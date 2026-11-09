"""
Byte-level Byte-Pair Encoding (BPE) tokenizer, written from scratch (Milestone 2).

This is the kind of tokenizer real LLMs use (GPT-2/3/4, Llama, etc.). The idea:

  1. Start with the 256 possible bytes as the base vocabulary. Any text, in any
     language or encoding, is therefore always representable — there are no
     "unknown" characters, unlike the char-level tokenizer in Milestone 1.
  2. Repeatedly find the most frequent adjacent pair of tokens and merge it into
     a single new token. Each merge adds one entry to the vocabulary.
  3. Stop when the vocabulary reaches the target size.

Pretokenization: before BPE, the text is split into chunks (runs of non-space
and runs of space) with a regex. Merges only happen *within* a chunk, never
across the boundary. Real BPE tokenizers do this, and it matters in practice:
chunks are short, so training and encoding are fast, and encoded chunks can be
cached (the word "authentication" is encoded once, not every time it appears).

Nothing here is imported from a tokenizer library; the training, encoding and
decoding are all implemented by hand so the mechanism is fully visible.
"""

import json
import re

from special_tokens import SpecialTokensMixin

# Cap the per-tokenizer encode memo so a long-lived serving process fed many
# distinct inputs cannot accumulate retained strings/token-lists without bound.
_CACHE_MAX = 100_000

# Split into runs of non-whitespace or runs of whitespace. Together these cover
# every character with no loss, so decode(encode(x)) == x.
_SPLIT = re.compile(r"\S+|\s+")


def _get_pair_counts(ids, counts=None):
    """Count how often each adjacent pair of ids occurs. (a, b) -> frequency."""
    counts = counts if counts is not None else {}
    for a, b in zip(ids, ids[1:]):
        counts[(a, b)] = counts.get((a, b), 0) + 1
    return counts


def _merge(ids, pair, new_id):
    """Replace every occurrence of `pair` in `ids` with `new_id`."""
    out = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer(SpecialTokensMixin):
    kind = "bpe"

    def __init__(self, merges=None, vocab=None, special=None):
        # merges: dict mapping (int, int) pair -> new int id, in learned order.
        self.merges = merges or {}
        # vocab: dict mapping id -> bytes. Built from merges on top of 0..255.
        self.vocab = vocab or {i: bytes([i]) for i in range(256)}
        self._rank = None       # pair -> merge order, built lazily for encode
        self._cache = {}        # chunk str -> encoded ids, memoized
        self.special = dict(special) if special else {}

    def _base_vocab_size(self):
        return len(self.vocab)

    @staticmethod
    def _chunks(text):
        return _SPLIT.findall(text)

    def train(self, text, vocab_size, verbose=False):
        """Learn merges from `text` until the vocabulary reaches `vocab_size`."""
        assert vocab_size >= 256, "vocab_size must be at least 256 (the byte base)"
        num_merges = vocab_size - 256

        # Each chunk becomes a list of byte ids; merges stay inside a chunk.
        seqs = [list(c.encode("utf-8")) for c in self._chunks(text)]
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special = {}
        self._special_inverse = None

        for m in range(num_merges):
            counts = {}
            for s in seqs:
                _get_pair_counts(s, counts)
            if not counts:
                break
            pair = max(counts, key=counts.get)
            new_id = 256 + m
            seqs = [_merge(s, pair, new_id) for s in seqs]
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose and (m < 3 or (m + 1) % 500 == 0):
                print(f"  merge {m + 1}/{num_merges}: {pair} -> {new_id} "
                      f"({counts[pair]} occurrences)", flush=True)

        self._rank = None
        self._cache = {}
        return self

    def _ranks(self):
        if self._rank is None:
            # Insertion order of merges IS their priority (earliest = lowest rank).
            self._rank = {pair: i for i, pair in enumerate(self.merges)}
        return self._rank

    def _encode_chunk(self, chunk):
        if chunk in self._cache:
            return self._cache[chunk]
        ids = list(chunk.encode("utf-8"))
        rank = self._ranks()
        # Repeatedly merge the lowest-rank pair present, until none remain.
        while len(ids) >= 2:
            pairs = set(zip(ids, ids[1:]))
            candidate = min(pairs, key=lambda p: rank.get(p, float("inf")))
            if candidate not in rank:
                break
            ids = _merge(ids, candidate, self.merges[candidate])
        # Bound the memo so a serving process fed endless distinct inputs can't
        # grow memory without limit; evict oldest (FIFO) past the cap.
        if len(self._cache) >= _CACHE_MAX:
            self._cache.pop(next(iter(self._cache)))
        self._cache[chunk] = ids
        return ids

    def _encode_ordinary(self, text):
        """Text -> list of token ids (per-chunk, with memoization)."""
        out = []
        for chunk in self._chunks(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def _decode_ordinary(self, ids):
        """List of token ids -> text."""
        data = b"".join(self.vocab[int(i)] for i in ids)
        return data.decode("utf-8", errors="replace")

    def to_dict(self):
        # JSON keys must be strings, so pairs are stored as "a,b".
        return self._with_special({
            "kind": self.kind,
            "merges": {f"{a},{b}": nid for (a, b), nid in self.merges.items()},
        })

    @classmethod
    def from_dict(cls, obj):
        merges = {}
        vocab = {i: bytes([i]) for i in range(256)}
        # Merge ids are assigned in learned order, so sorting by id restores
        # priority even if a serializer reordered the keys.
        for key, nid in sorted(obj["merges"].items(), key=lambda kv: int(kv[1])):
            a, b = (int(x) for x in key.split(","))
            merges[(a, b)] = nid
            vocab[nid] = vocab[a] + vocab[b]
        return cls(merges=merges, vocab=vocab)._restore_special(obj)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


if __name__ == "__main__":
    # Tiny self-check so you can see it work: python tokenizer_bpe.py
    sample = ("the cryptographic handshake authenticates the server "
              "before any application data is encrypted. ") * 50
    tok = BPETokenizer().train(sample, vocab_size=320, verbose=True)
    s = "the server authenticates the handshake"
    ids = tok.encode(s)
    print(f"\nvocab size: {tok.vocab_size}")
    print(f"text:    {s!r}")
    print(f"tokens:  {ids}  ({len(ids)} tokens vs {len(s)} chars)")
    print(f"decoded: {tok.decode(ids)!r}")
    assert tok.decode(ids) == s, "round-trip failed"
    print("round-trip OK")
