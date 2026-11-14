import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

import tokenization
from data.shards import write_shards
from model import GPT, GPTConfig
from train import atomic_save
from training.sft import build_parser, tune


def verify(device, report):
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA resume qualification requires an actual GPU")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with tempfile.TemporaryDirectory(prefix="llmfs-resume-") as temporary:
        root = Path(temporary)
        text = "".join(f"def f{i}(x):\n    return x + {i}\n\n" for i in range(60))
        tokenizer = tokenization.build_with_specials("bpe_code", text * 2, vocab_size=300, bpe_train_chars=10 ** 6)
        torch.manual_seed(42)
        config = GPTConfig(vocab_size=tokenizer.vocab_size, block_size=64,
                           n_layer=2, n_head=2, n_embd=32, dropout=0.1)
        base = root / "base" / "base.pt"
        atomic_save({"model_state": GPT(config).state_dict(), "model_cfg": config.__dict__,
                     "tokenizer": tokenizer.to_dict()}, base)
        data = root / "data"
        data.mkdir()
        for split, size in (("train", 24), ("validation", 6)):
            rows = [{"id": f"{split}-{i}", "family": f"{split}-{i}", "kind": "implement",
                     "prompt": f"Write f{i}.", "response": f"def f{i}(x):\n    return x + {i}\n"} for i in range(size)]
            (data / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        (data / "manifest.json").write_text(json.dumps({"schema": 1, "purpose": "synthetic resume check only"}))
        replay = root / "replay"
        write_shards([(f"def g{i}(x):\n    return x * {i}\n", "val" if i % 8 == 0 else "train")
                      for i in range(80)], tokenizer, replay, shard_tokens=500)

        def args(output, extra=()):
            return build_parser().parse_args([
                "--base", str(base), "--data", str(data), "--replay-dir", str(replay),
                "--replay-fraction", "0.3", "--output", str(root / output), "--steps", "6",
                "--batch-size", "2", "--grad-accum-steps", "2", "--warmup", "2",
                "--context", "64", "--eval-interval", "3", "--learning-rate", "1e-3",
                "--device", device, "--upsample", "", *extra])

        tune(args("continuous"))
        tune(args("interrupted", ["--stop-at-step", "3"]))
        tune(args("resumed", ["--resume", str(root / "interrupted" / "ckpt.pt")]))
        full = torch.load(root / "continuous" / "ckpt.pt", map_location="cpu", weights_only=True)
        resumed = torch.load(root / "resumed" / "ckpt.pt", map_location="cpu", weights_only=True)
        mismatches = [key for key, value in full["model_state"].items()
                      if not torch.equal(value, resumed["model_state"][key])]
        if mismatches:
            raise ValueError(f"resumed model differs in tensors: {mismatches}")
        for key in ("sampler", "mix_rng", "scaler_state", "best", "step"):
            if full[key] != resumed[key]:
                raise ValueError(f"resume state differs: {key}")
        for key in ("torch_rng", "replay_rng"):
            if not torch.equal(full[key], resumed[key]):
                raise ValueError(f"resume RNG differs: {key}")
        if len(full["cuda_rng"]) != len(resumed["cuda_rng"]) or any(
                not torch.equal(a, b) for a, b in zip(full["cuda_rng"], resumed["cuda_rng"])):
            raise ValueError("resume CUDA RNG differs")
        result = {"passed": True, "device": device, "torch_version": str(torch.__version__),
                  "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
                  "dropout": 0.1, "steps": 6, "interrupt_at": 3, "replay_fraction": 0.3,
                  "model_tensors_identical": len(full["model_state"]), "rng_states_identical": True,
                  "precision": full["meta"]["amp_dtype"]}
        Path(report).write_text(json.dumps(result, indent=2) + "\n")
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare continuous and interrupted SFT with dropout and replay.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--report", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(verify(args.device, args.report)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
