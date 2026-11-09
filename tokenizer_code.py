"""Code-aware byte-level BPE with an incremental trainer.

Differences from tokenizer_bpe.BPETokenizer (kept unchanged for M2/M3):
  - pretokenization splits identifiers, short digit runs, punctuation, newlines
    and indentation, instead of whitespace-separated words;
  - training counts each distinct chunk once with its frequency and updates
    pair counts only for chunks that contain the merged pair, so a 16k
    vocabulary trains in minutes rather than hours.
"""

import heapq
import json
import re
from collections import Counter, defaultdict

from special_tokens import SpecialTokensMixin

PATTERN_VERSION = 1
_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)"""
    r"""| ?[^\W\d_][^\W_]*(?:_[^\W_]*)*"""
    r"""| ?_+[^\W_]*(?:_[^\W_]*)*"""
    r"""| ?\d{1,3}"""
    r"""| ?[^\s\w]+"""
    r"""|\r?\n"""
    r"""| {4}| {2}| \t?|\t+"""
    r"""|\s""",
    re.UNICODE,
)
_CACHE_MAX = 200_000


def pretokenize(text):
    chunks = _PATTERN.findall(text)
    if "".join(chunks) != text:
        raise ValueError("pretokenizer lost characters")
    return chunks


class CodeBPETokenizer(SpecialTokensMixin):
    kind = "bpe_code"

    def __init__(self, merges=None, special=None):
        self.merges = dict(merges or {})
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), nid in sorted(self.merges.items(), key=lambda kv: kv[1]):
            self.vocab[nid] = self.vocab[a] + self.vocab[b]
        self.special = dict(special) if special else {}
        self._cache = {}

    def _base_vocab_size(self):
        return len(self.vocab)

    def train(self, text_or_counts, vocab_size, verbose=False, min_frequency=2):
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256")
        counts = text_or_counts if isinstance(text_or_counts, Counter) else Counter(pretokenize(text_or_counts))
        words = [list(chunk.encode("utf-8")) for chunk in counts]
        freqs = list(counts.values())
        pair_counts = Counter()
        where = defaultdict(set)
        for wi, (word, freq) in enumerate(zip(words, freqs)):
            for pair in zip(word, word[1:]):
                pair_counts[pair] += freq
                where[pair].add(wi)
        heap = [(-c, pair) for pair, c in pair_counts.items()]
        heapq.heapify(heap)
        self.merges, self.vocab, self.special, self._cache = {}, {i: bytes([i]) for i in range(256)}, {}, {}
        next_id = 256
        while next_id < vocab_size and heap:
            neg, pair = heapq.heappop(heap)
            current = pair_counts.get(pair, 0)
            if current != -neg:
                if current > 0:
                    heapq.heappush(heap, (-current, pair))
                continue
            if current < min_frequency:
                break
            self.merges[pair] = next_id
            self.vocab[next_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            changed = Counter()
            for wi in where.pop(pair, ()):
                word, freq = words[wi], freqs[wi]
                i, out = 0, []
                while i < len(word):
                    if i + 1 < len(word) and word[i] == pair[0] and word[i + 1] == pair[1]:
                        out.append(next_id)
                        i += 2
                    else:
                        out.append(word[i])
                        i += 1
                if out == word:
                    continue
                for p in zip(word, word[1:]):
                    changed[p] -= freq
                for p in zip(out, out[1:]):
                    changed[p] += freq
                    where[p].add(wi)
                words[wi] = out
            for p, delta in changed.items():
                if not delta:
                    continue
                pair_counts[p] += delta
                if pair_counts[p] <= 0:
                    pair_counts.pop(p, None)
                else:
                    heapq.heappush(heap, (-pair_counts[p], p))
            pair_counts.pop(pair, None)
            if verbose and (next_id - 255) % 1000 == 0:
                print(f"  merge {next_id - 255}/{vocab_size - 256} ({current} occurrences)", flush=True)
            next_id += 1
        return self

    def _encode_chunk(self, chunk):
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        merges = self.merges
        while len(ids) >= 2:
            best, best_rank = None, None
            for pair in zip(ids, ids[1:]):
                rank = merges.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best, best_rank = pair, rank
            if best is None:
                break
            out, i = [], 0
            while i < len(ids):
                if i + 1 < len(ids) and ids[i] == best[0] and ids[i + 1] == best[1]:
                    out.append(best_rank)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = out
        if len(self._cache) >= _CACHE_MAX:
            self._cache.pop(next(iter(self._cache)))
        self._cache[chunk] = ids
        return ids

    def _encode_ordinary(self, text):
        out = []
        for chunk in _PATTERN.findall(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def _decode_ordinary(self, ids):
        return b"".join(self.vocab[int(i)] for i in ids).decode("utf-8", errors="replace")

    def to_dict(self):
        return self._with_special({
            "kind": self.kind, "pattern_version": PATTERN_VERSION,
            "merges": [[a, b, nid] for (a, b), nid in sorted(self.merges.items(), key=lambda kv: kv[1])],
        })

    @classmethod
    def from_dict(cls, obj):
        if obj.get("pattern_version") != PATTERN_VERSION:
            raise ValueError("unsupported bpe_code pattern version")
        merges = {}
        for expected, (a, b, nid) in enumerate(obj["merges"], start=256):
            if nid != expected or a >= nid or b >= nid:
                raise ValueError("bpe_code merges must be contiguous and reference earlier ids")
            merges[(a, b)] = nid
        return cls(merges=merges)._restore_special(obj)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))
