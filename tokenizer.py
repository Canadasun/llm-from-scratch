"""
Character-level tokenizer (Milestone 1).

The simplest tokenizer that works: every distinct character in the training
text becomes one token. There is no training step and no external vocabulary
file format to learn, so you can focus on the model itself. Milestone 2 swaps
this for a byte-pair-encoding (BPE) tokenizer, which is what real LLMs use.
"""

import json


class CharTokenizer:
    kind = "char"

    def __init__(self, chars):
        # Sorted for determinism, so the same corpus always yields the same ids.
        self.chars = sorted(set(chars))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}

    @property
    def vocab_size(self):
        return len(self.chars)

    @classmethod
    def from_text(cls, text):
        return cls(text)

    def encode(self, text):
        return [self.stoi[c] for c in text]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"chars": self.chars}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["chars"])

    # --- uniform (de)serialization so a checkpoint can embed the tokenizer ---

    def to_dict(self):
        return {"kind": self.kind, "chars": self.chars}

    @classmethod
    def from_dict(cls, d):
        return cls(d["chars"])
