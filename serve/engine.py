"""
Inference engine: hold a loaded model + tokenizer and generate text.

This is framework-agnostic (no HTTP here) so it can be unit-tested directly and
reused by any transport. It mirrors the secure load used in sample.py.
"""

import os

import torch

import tokenization
from model import GPT, GPTConfig
from special_tokens import ENDOFTEXT, USER
from serve.validation import ValidationError
from serve.lifecycle import check_generation

DEFAULT_CKPT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints", "ckpt.pt")


class InferenceEngine:
    def __init__(self, model, tokenizer, device="cpu", quantize=False):
        model = model.to(device).eval()
        # cfg is read off the model before quantization wraps it (the quantized
        # module keeps the same .cfg attribute, but we grab the dims up front).
        self.block_size = model.cfg.block_size
        self.vocab_size = model.cfg.vocab_size
        self.quantized = False
        if quantize:
            model = self._quantize(model)
            self.quantized = True
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        special_id = getattr(tokenizer, "special_id", None)
        self.stop_ids = frozenset(
            i for i in (special_id(name) for name in (ENDOFTEXT, USER)) if i is not None
        ) if special_id else frozenset()

    def _decode(self, ids):
        if getattr(self.tokenizer, "special_ids", None):
            return self.tokenizer.decode(ids, skip_special=True)
        return self.tokenizer.decode(ids)

    @staticmethod
    def _quantize(model):
        """Apply dynamic int8 quantization to the Linear layers (attention/MLP/
        head). CPU-only: it roughly halves generation latency with negligible
        quality change (measured bits-per-byte delta ~+0.004, well within the
        regression gate). Dynamic quant needs no calibration data."""
        return torch.ao.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8)

    @classmethod
    def from_checkpoint(cls, path=DEFAULT_CKPT, device="cpu", quantize=False):
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
        engine = cls(model, tokenizer, device=device, quantize=quantize)
        from evals.evidence import file_sha256
        engine.checkpoint_sha256 = file_sha256(path)
        return engine

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
        check_generation()
        ids = self._encode_prompt(prompt)
        if len(ids) > self.block_size:
            raise ValidationError(
                f"prompt has {len(ids)} tokens; model context limit is {self.block_size}")
        idx = torch.tensor([ids], dtype=torch.long, device=self.device)
        out = self.model.generate(
            idx, max_new_tokens, temperature=temperature, top_k=top_k,
            generator=gen, check_cancelled=check_generation, stop_ids=self.stop_ids)
        check_generation()
        out_ids = out[0].tolist()
        if self.stop_ids and len(out_ids) > len(ids) and out_ids[-1] in self.stop_ids:
            out_ids = out_ids[:-1]
        full = self._decode(out_ids)
        completion = self._decode(out_ids[len(ids):])
        return {
            "text": full,
            "completion": completion,
            "prompt_tokens": len(ids),
            "completion_tokens": len(out_ids) - len(ids),
        }
