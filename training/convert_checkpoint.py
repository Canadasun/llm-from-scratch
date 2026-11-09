"""Convert a pre-format-2 checkpoint (M2/M3 era) to the current format.

    python -m training.convert_checkpoint --src old.pt --dst new.pt

The legacy file stored the causal mask as a persistent buffer. The current model
rebuilds it, so conversion drops those keys and verifies that the converted
weights produce the same logits under both attention backends.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tokenization  # noqa: E402
from model import GPT, GPTConfig, set_attention_backend, upgrade_state_dict  # noqa: E402
from train import CHECKPOINT_FORMAT, atomic_save, file_sha256  # noqa: E402


def convert(src, dst, atol=1e-5):
    src, dst = Path(src), Path(dst)
    if src.resolve() == dst.resolve():
        raise ValueError("write the converted checkpoint to a new path")
    if dst.exists():
        raise ValueError(f"{dst} exists; choose a new path")
    blob = torch.load(src, map_location="cpu", weights_only=True)
    state = blob["model_state"]
    dropped = sorted(set(state) - set(upgrade_state_dict(state)))
    model = GPT(GPTConfig(**blob["model_cfg"]))
    model.load_state_dict(state)
    model.eval()
    tokenizer = tokenization.from_dict(blob["tokenizer"])
    generator = torch.Generator().manual_seed(0)
    length = min(64, model.cfg.block_size)
    idx = torch.randint(tokenizer.vocab_size, (2, length), generator=generator)
    previous = set_attention_backend("manual")
    try:
        with torch.no_grad():
            manual, _ = model(idx)
        set_attention_backend("sdpa")
        with torch.no_grad():
            sdpa, _ = model(idx)
    finally:
        set_attention_backend(previous)
    max_diff = float((manual - sdpa).abs().max())
    if max_diff > atol:
        raise ValueError(f"attention backends disagree by {max_diff}; refusing to convert")
    converted = {k: v for k, v in blob.items() if k != "model_state"}
    converted["model_state"] = model.state_dict()
    converted["format_version"] = CHECKPOINT_FORMAT
    converted["converted_from"] = {"sha256": file_sha256(src), "name": src.name,
                                   "dropped_keys": len(dropped)}
    atomic_save(converted, dst)
    reloaded = GPT(GPTConfig(**blob["model_cfg"]))
    reloaded.load_state_dict(torch.load(dst, map_location="cpu", weights_only=True)["model_state"])
    reloaded.eval()
    with torch.no_grad():
        again, _ = reloaded(idx)
    if float((again - manual).abs().max()) > atol:
        raise ValueError("converted checkpoint does not reproduce the source logits")
    return {"src_sha256": converted["converted_from"]["sha256"], "dst_sha256": file_sha256(dst),
            "dropped_keys": len(dropped), "max_backend_logit_diff": max_diff}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args(argv)
    print(json.dumps(convert(args.src, args.dst), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
