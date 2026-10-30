"""
Load a trained checkpoint into an evaluatable (model, tokenizer) pair, and build
a fresh random-init model of the same shape as a baseline.

The load mirrors sample.py exactly, including the secure `weights_only=True`
guard so a malicious checkpoint cannot execute code through pickle.
"""

import os
import sys

import torch

# Make the repo-root modules (model, tokenization, ...) importable no matter the
# current working directory when the suite is run.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import tokenization                       # noqa: E402
from model import GPT, GPTConfig          # noqa: E402

DEFAULT_CKPT = os.path.join(_REPO_ROOT, "checkpoints", "ckpt.pt")


def pick_device(requested="auto"):
    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_checkpoint(ckpt_path=DEFAULT_CKPT, device="auto"):
    """Return (model, tokenizer, cfg_dict) from a checkpoint, in eval mode."""
    if not os.path.exists(ckpt_path):
        raise SystemExit(
            f"Checkpoint {ckpt_path} not found. Train first (python train.py "
            "--preset gpu) or copy a downloaded ckpt.pt into checkpoints/."
        )
    device = pick_device(device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    tokenizer = tokenization.from_dict(ckpt["tokenizer"])
    model = GPT(GPTConfig(**ckpt["model_cfg"]))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, tokenizer, ckpt["model_cfg"]


def random_baseline(cfg_dict, device="auto", seed=0):
    """A same-shape, *untrained* model. Comparing against it proves that training
    — not the architecture or the tokenizer alone — is what produced any skill."""
    device = pick_device(device)
    torch.manual_seed(seed)
    model = GPT(GPTConfig(**cfg_dict))
    model.to(device).eval()
    return model


def safe_encode(tokenizer, text):
    """Encode text to ids. A char tokenizer has no id for an unseen character, so
    drop those (as sample.py does); BPE is byte-level and encodes anything."""
    if getattr(tokenizer, "kind", None) == "char":
        text = "".join(c for c in text if c in tokenizer.stoi)
    return tokenizer.encode(text)
