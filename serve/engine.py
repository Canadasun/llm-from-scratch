"""
Inference engine: hold a loaded model + tokenizer and generate text.

This is framework-agnostic (no HTTP here) so it can be unit-tested directly and
reused by any transport. It mirrors the secure load used in sample.py.
"""

import os

import torch

import tokenization
from model import GPT, GPTConfig

DEFAULT_CKPT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints", "ckpt.pt")


class InferenceEngine:
    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        self.block_size = model.cfg.block_size
        self.vocab_size = model.cfg.vocab_size

    @classmethod
    def from_checkpoint(cls, path=DEFAULT_CKPT, device="cpu"):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Checkpoint {path} not found. Train first (see docs/GPU.md) or "
                f"pass --ckpt.")
        # Secure load: weights_only=True so a crafted checkpoint cannot execute
        # arbitrary code via pickle. Our checkpoints hold only tensors + plain data.
        ckpt = torch.load(path, map_location=device, weights_only=True)
        tokenizer = tokenization.from_dict(ckpt["tokenizer"])
        model = GPT(GPTConfig(**ckpt["model_cfg"]))
        model.load_state_dict(ckpt["model_state"])
        return cls(model, tokenizer, device=device)

    def _encode_prompt(self, prompt):
        p = prompt
        if getattr(self.tokenizer, "kind", None) == "char":
            # A char tokenizer has no id for a character it never saw; drop those
            # rather than raising on arbitrary user input.
            p = "".join(c for c in p if c in self.tokenizer.stoi)
        ids = self.tokenizer.encode(p)
        if not ids:
            # Fall back to a newline, but a char tokenizer whose vocab lacks "\n"
            # would raise here, so guard it and fall back to token id 0.
            try:
                ids = self.tokenizer.encode("\n")
            except Exception:
                ids = []
            if not ids:
                ids = [0]
        return ids

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=128, temperature=0.8, top_k=None,
                 seed=None):
        """Generate a continuation. Returns a dict with the full text, the
        continuation only, and token counts. Pure function of its inputs when
        `seed` is set (reproducible)."""
        # A per-call generator keeps a seeded request reproducible without
        # touching the process-wide RNG, so concurrent requests can't perturb
        # each other's seeded output (torch.manual_seed would be a shared race).
        gen = None
        if seed is not None:
            gen = torch.Generator(device=self.device)
            gen.manual_seed(seed)
        ids = self._encode_prompt(prompt)
        idx = torch.tensor([ids], dtype=torch.long, device=self.device)
        out = self.model.generate(
            idx, max_new_tokens, temperature=temperature, top_k=top_k,
            generator=gen)
        out_ids = out[0].tolist()
        full = self.tokenizer.decode(out_ids)
        completion = self.tokenizer.decode(out_ids[len(ids):])
        return {
            "text": full,
            "completion": completion,
            "prompt_tokens": len(ids),
            "completion_tokens": len(out_ids) - len(ids),
        }
