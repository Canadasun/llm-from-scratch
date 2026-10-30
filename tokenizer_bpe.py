"""
Byte-level Byte-Pair Encoding (BPE) tokenizer, written from scratch (Milestone 2).

This is the kind of tokenizer real LLMs use (GPT-2/3/4, Llama, etc.). The idea:

  1. Start with the 256 possible bytes as the base vocabulary. Any text, in any
     language or encoding, is therefore always representable — there are no
     "unknown" characters, unlike the char-level tokenizer in Milestone 1.
  2. Repeatedly find the most frequent adjacent pair of tokens in the corpus and
     merge it into a single new token. Each merge adds one entry to the vocab.
  3. Stop when the vocabulary reaches the target size.

Encoding new text = apply the learned merges, in the order they were learned.
Decoding = look each token's bytes up and concatenate, then UTF-8 decode.

Nothing here is imported from a tokenizer library; the training, encoding and
decoding are all implemented by hand so the mechanism is fully visible.
"""

import json


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


class BPETokenizer:
    def __init__(self, merges=None, vocab=None):
        # merges: dict mapping (int, int) pair -> new int id, in learned order.
        self.merges = merges or {}
        # vocab: dict mapping id -> bytes. Built from merges on top of 0..255.
        self.vocab = vocab or {i: bytes([i]) for i in range(256)}

    @property
    def vocab_size(self):
        return len(self.vocab)

    def train(self, text, vocab_size, verbose=False):
        """Learn merges from `text` until the vocabulary reaches `vocab_size`."""
        assert vocab_size >= 256, "vocab_size must be at least 256 (the byte base)"
        num_merges = vocab_size - 256

        ids = list(text.encode("utf-8"))
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

        for m in range(num_merges):
            counts = _get_pair_counts(ids)
            if not counts:
                break  # corpus fully merged (tiny inputs only)
            # Most frequent adjacent pair.
            pair = max(counts, key=counts.get)
            new_id = 256 + m
            ids = _merge(ids, pair, new_id)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose and (m < 5 or m % 500 == 0):
                print(f"  merge {m + 1}/{num_merges}: {pair} -> {new_id} "
                      f"({counts[pair]} occurrences)")
        return self

    def encode(self, text):
        """Text -> list of token ids, applying learned merges in order."""
        ids = list(text.encode("utf-8"))
        # Apply merges in the order they were learned (that order matters).
        for pair, new_id in self.merges.items():
            ids = _merge(ids, pair, new_id)
        return ids

    def decode(self, ids):
        """List of token ids -> text."""
        data = b"".join(self.vocab[int(i)] for i in ids)
        return data.decode("utf-8", errors="replace")

    def save(self, path):
        # JSON keys must be strings, so pairs are stored as "a,b".
        obj = {
            "merges": {f"{a},{b}": nid for (a, b), nid in self.merges.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        merges = {}
        vocab = {i: bytes([i]) for i in range(256)}
        # Rebuild in insertion order (Python dicts preserve it; JSON too).
        for key, nid in obj["merges"].items():
            a, b = (int(x) for x in key.split(","))
            merges[(a, b)] = nid
            vocab[nid] = vocab[a] + vocab[b]
        return cls(merges=merges, vocab=vocab)


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
