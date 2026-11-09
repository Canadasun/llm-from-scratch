"""Measure training throughput for the Phase 4 model sizes on the current device.

    python -m training.benchmark --presets gpu_base gpu_xl --out bench.json

Uses synthetic token ids, so it measures compute only; the real loader adds a
little overhead. Each preset tries micro-batches from largest to smallest and
records the fastest one that fits.
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import PRESETS  # noqa: E402
from model import GPT, GPTConfig, attention_backend  # noqa: E402
from train import (TensorSplit, configure_optimizer, make_autocast, native_bf16,  # noqa: E402
                   pick_device, train_step)


def measure(preset, device, vocab_size, batch_sizes, steps, warmup, compile_model):
    cfg = PRESETS[preset]()
    cfg.grad_accum_steps = 1
    if vocab_size:
        cfg.vocab_size = vocab_size
    model_cfg = GPTConfig(vocab_size=cfg.vocab_size, block_size=cfg.block_size, n_layer=cfg.n_layer,
                          n_head=cfg.n_head, n_embd=cfg.n_embd, dropout=cfg.dropout)
    data = TensorSplit(torch.randint(cfg.vocab_size, (cfg.block_size * 64,), generator=torch.Generator().manual_seed(0)))
    result = {"preset": preset, "vocab_size": cfg.vocab_size, "block_size": cfg.block_size,
              "n_layer": cfg.n_layer, "n_head": cfg.n_head, "n_embd": cfg.n_embd, "attempts": []}
    best = None
    for batch_size in batch_sizes:
        cfg.batch_size = batch_size
        model = optimizer = None
        try:
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            model = GPT(model_cfg).to(device)
            result["params"] = model.num_params()
            optimizer = configure_optimizer(model, cfg, device)
            if compile_model:
                model = torch.compile(model)
            autocast_ctx, scaler, amp_name = make_autocast(device)
            generator = torch.Generator().manual_seed(1)
            for _ in range(warmup):
                loss = train_step(model, optimizer, scaler, data, cfg, device, autocast_ctx, generator)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            began = time.perf_counter()
            for _ in range(steps):
                loss = train_step(model, optimizer, scaler, data, cfg, device, autocast_ctx, generator)
            final_loss = loss.item()
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            seconds = time.perf_counter() - began
            attempt = {"batch_size": batch_size, "ok": True, "seconds": round(seconds, 3),
                       "tokens_per_second": round(steps * batch_size * cfg.block_size / seconds, 1),
                       "final_loss": round(final_loss, 4), "amp_dtype": amp_name}
            if device.startswith("cuda"):
                attempt["peak_memory_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            result["attempts"].append(attempt)
            if best is None or attempt["tokens_per_second"] > best["tokens_per_second"]:
                best = attempt
        except torch.cuda.OutOfMemoryError:
            result["attempts"].append({"batch_size": batch_size, "ok": False, "error": "out of memory"})
        finally:
            del model, optimizer
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        if best is not None and len([a for a in result["attempts"] if a["ok"]]) >= 2:
            break
    result["best"] = best
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--presets", nargs="+", default=["gpu_base", "gpu_xl"], choices=list(PRESETS))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--vocab-size", type=int, default=None)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[24, 16, 12, 8, 6, 4, 2, 1])
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    device = pick_device(args.device)
    report = {"device": device, "torch": str(torch.__version__), "python": platform.python_version(),
              "attention_backend": attention_backend(), "compile": args.compile,
              "steps": args.steps, "warmup": args.warmup, "results": []}
    if device.startswith("cuda"):
        props = torch.cuda.get_device_properties(0)
        report.update({"gpu": props.name, "gpu_count": torch.cuda.device_count(),
                       "gpu_memory_gb": round(props.total_memory / 2**30, 1),
                       "bf16_reported": torch.cuda.is_bf16_supported(),
                       "bf16_native": native_bf16(device),
                       "compute_capability": list(torch.cuda.get_device_capability(0))})
    for preset in args.presets:
        r = measure(preset, device, args.vocab_size, args.batch_sizes, args.steps, args.warmup, args.compile)
        report["results"].append(r)
        print(json.dumps({k: r[k] for k in ("preset", "params", "best")}), flush=True)
    text = json.dumps(report, indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
