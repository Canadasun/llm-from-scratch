import argparse
import hashlib
import json
import math
import os
import time
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F

from config import TrainConfig
from data.shards import LEGACY_PREFIX, UNIFORM_WINDOWS, ShardSplit, load_manifest
from evals.loader import load_checkpoint
from train import atomic_save, atomic_write_text, configure_optimizer, file_sha256, lr_for, make_autocast, train_step
from training.control_plan import load as load_plan


@contextmanager
def deterministic_runtime():
    previous = {"deterministic": torch.are_deterministic_algorithms_enabled(),
                "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
                "threads": torch.get_num_threads(), "tf32": torch.backends.cuda.matmul.allow_tf32,
                "flash": torch.backends.cuda.flash_sdp_enabled(),
                "efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
                "math": torch.backends.cuda.math_sdp_enabled()}
    previous["cudnn_sdp"] = torch.backends.cuda.cudnn_sdp_enabled() if hasattr(torch.backends.cuda, "cudnn_sdp_enabled") else None
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    if previous["cudnn_sdp"] is not None:
        torch.backends.cuda.enable_cudnn_sdp(False)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous["deterministic"], warn_only=previous["warn_only"])
        torch.set_num_threads(previous["threads"])
        torch.backends.cuda.matmul.allow_tf32 = previous["tf32"]
        torch.backends.cuda.enable_flash_sdp(previous["flash"])
        torch.backends.cuda.enable_mem_efficient_sdp(previous["efficient"])
        torch.backends.cuda.enable_math_sdp(previous["math"])
        if previous["cudnn_sdp"] is not None:
            torch.backends.cuda.enable_cudnn_sdp(previous["cudnn_sdp"])


def tensor_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def warm_start(base, expected_sha256, tokenizer, device, seed):
    if file_sha256(base) != expected_sha256:
        raise ValueError("base checkpoint sha256 does not match the accepted artifact")
    torch.manual_seed(seed)
    model, base_tokenizer, model_cfg = load_checkpoint(str(base), device)
    if base_tokenizer.to_dict() != tokenizer.to_dict():
        raise ValueError("base tokenizer differs from the corpus tokenizer")
    return model, base_tokenizer, model_cfg


@torch.no_grad()
def measure(model, plan, arrays, device, batch_size=4):
    previous = model.training
    model.eval()
    windows, losses = plan["windows"], []
    context = plan["block_size"]
    for start in range(0, len(windows), batch_size):
        selected = windows[start:start + batch_size]
        block = torch.from_numpy(np.stack([
            np.asarray(arrays[row["shard"]][row["start"]:row["start"] + context + 1], dtype=np.int64)
            for row in selected])).to(device)
        logits, _ = model(block[:, :-1].contiguous())
        per_token = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                    block[:, 1:].contiguous().reshape(-1), reduction="none")
        values = per_token.reshape(len(selected), context).mean(dim=1).cpu().tolist()
        if not all(math.isfinite(value) for value in values):
            raise ValueError("nonfinite fixed-window evaluation loss")
        losses.extend({"id": row["id"], "stratum": row["stratum"], "group_sha256": row["group_sha256"],
                       "loss_nats": value} for row, value in zip(selected, values))
    summary = {}
    for stratum in sorted({row["stratum"] for row in losses}):
        values = [row["loss_nats"] for row in losses if row["stratum"] == stratum]
        average = sum(values) / len(values)
        summary[stratum] = {"windows": len(values), "mean_loss_nats": average, "perplexity": math.exp(average)}
    model.train(previous)
    return {"precision": "float32", "summary": summary, "window_losses": losses}


def paired_difference(treatment, reference, seed=1730):
    left = {row["id"]: row for row in treatment["window_losses"] if row["stratum"] == "code_uniform"}
    right = {row["id"]: row for row in reference["window_losses"] if row["stratum"] == "code_uniform"}
    if not left or left.keys() != right.keys():
        raise ValueError("paired code-window membership mismatch")
    groups = [left[key]["group_sha256"] for key in sorted(left)]
    if len(set(groups)) != len(groups) or any(left[key]["group_sha256"] != right[key]["group_sha256"] for key in left):
        raise ValueError("code windows are not independent repository pairs")
    differences = np.array([left[key]["loss_nats"] - right[key]["loss_nats"] for key in sorted(left)], dtype=np.float64)
    if not np.isfinite(differences).all():
        raise ValueError("nonfinite paired loss")
    generator = np.random.default_rng(seed)
    bootstrap = differences[generator.integers(len(differences), size=(2000, len(differences)))].mean(axis=1)
    low, high = np.quantile(bootstrap, [0.025, 0.975]).tolist()
    return {"repositories": len(differences), "mean_delta_nats": float(differences.mean()),
            "bootstrap_95_interval": [low, high], "resamples": 2000, "seed": seed,
            "passes": len(differences) >= 64 and float(differences.mean()) <= -0.01 and high < 0}


def assess(report):
    base, control, treatment = report["base"], report["arms"]["control"], report["arms"]["treatment"]
    if control["sampling_mode"] != LEGACY_PREFIX or treatment["sampling_mode"] != UNIFORM_WINDOWS:
        raise ValueError("arm labels do not match the predeclared sampling modes")
    control_cfg = {key: value for key, value in control["train_cfg"].items() if key != "sampling_mode"}
    treatment_cfg = {key: value for key, value in treatment["train_cfg"].items() if key != "sampling_mode"}
    if control_cfg != treatment_cfg:
        raise ValueError("arm recipes differ beyond the sampling mode")
    for row in (control, treatment):
        if (row["completed_steps"] != report["recipe"]["steps"] or row["attempted_steps"] != report["recipe"]["steps"] or row["skipped_steps"]
                or row["initial_optimizer_states"] != 0 or row["initial_tensor_sha256"] != base["tensor_sha256"]):
            return {"eligible_for_capability_screen": False, "reason": "incomplete_or_unmatched_initialization_or_overflow"}
    for field in ("initial_cpu_rng_sha256", "initial_device_rng_sha256", "initial_batch_rng_sha256", "initial_scaler_state"):
        if control[field] != treatment[field]:
            raise ValueError("arm initialization differs: " + field)
    versus_control = paired_difference(treatment["evaluation"], control["evaluation"])
    versus_base = paired_difference(treatment["evaluation"], base["evaluation"])
    prose_delta = (treatment["evaluation"]["summary"]["prose_uniform"]["mean_loss_nats"]
                   - base["evaluation"]["summary"]["prose_uniform"]["mean_loss_nats"])
    passed = versus_control["passes"] and versus_base["passes"] and prose_delta <= 0.02
    return {"eligible_for_capability_screen": passed, "versus_control": versus_control, "versus_base": versus_base,
            "prose_uniform_delta_vs_base": prose_delta, "prose_retained_within_0_02_nats": prose_delta <= 0.02,
            "decision": "run_observed_capability_screen" if passed else "stop_pilot_keep_accepted_base",
            "limitations": ["Single training seed; bootstrap covers repository-window variation, not training-seed variation.",
                            "Lower language-model loss does not establish functional coding or security capability."]}


def run_arm(label, mode, args, plan, arrays, manifest, tokenizer, base_tensor_sha, run_started):
    model, _, model_cfg = warm_start(args.base, args.expect_base_sha256, tokenizer, args.device, args.seed)
    if plan["block_size"] > model_cfg["block_size"]:
        raise ValueError("validation context exceeds the base model context")
    cfg = TrainConfig(block_size=plan["block_size"], n_layer=model_cfg["n_layer"], n_head=model_cfg["n_head"],
                      n_embd=model_cfg["n_embd"], dropout=model_cfg["dropout"], tokenizer=tokenizer.kind,
                      vocab_size=tokenizer.vocab_size, batch_size=args.batch_size, grad_accum_steps=args.accumulation,
                      learning_rate=args.learning_rate, max_iters=args.steps, lr_decay_iters=args.steps,
                      warmup_iters=args.warmup, sampling_mode=mode, seed=args.seed)
    optimizer = configure_optimizer(model, cfg, args.device)
    autocast, scaler, precision = make_autocast(args.device)
    if scaler.is_enabled():
        scale_state = scaler.state_dict()
        scale_state["scale"] = 1024.0
        scaler.load_state_dict(scale_state)
    torch.manual_seed(args.seed)
    batch_generator = torch.Generator().manual_seed(args.seed + 1)
    row = {"sampling_mode": mode, "train_cfg": cfg.__dict__, "initial_tensor_sha256": tensor_digest(model),
           "initial_optimizer_states": len(optimizer.state), "initial_scaler_state": scaler.state_dict(),
           "initial_cpu_rng_sha256": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
           "initial_device_rng_sha256": hashlib.sha256(torch.cuda.get_rng_state(args.device).cpu().numpy().tobytes()).hexdigest()
           if args.device.startswith("cuda") else None,
           "initial_batch_rng_sha256": hashlib.sha256(batch_generator.get_state().numpy().tobytes()).hexdigest(),
           "precision": precision, "history": []}
    if row["initial_tensor_sha256"] != base_tensor_sha or optimizer.state:
        raise ValueError("arm did not start from identical weights and fresh optimizer")
    out = Path(args.output) / label
    out.mkdir()
    data = ShardSplit(args.data_dir, "train", manifest)
    completed, skipped, attempted = 0, 0, 0
    model.train()
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    for step in range(args.steps):
        if time.monotonic() - run_started > args.max_seconds:
            break
        for group in optimizer.param_groups:
            group["lr"] = lr_for(step, cfg)
        previous_scale = scaler.get_scale()
        loss = train_step(model, optimizer, scaler, data, cfg, args.device, autocast, batch_generator)
        attempted += 1
        skipped += scaler.get_scale() < previous_scale
        completed += scaler.get_scale() >= previous_scale
        if not torch.isfinite(loss):
            raise ValueError("nonfinite control experiment training loss")
        if attempted % 20 == 0 or attempted == args.steps:
            record = {"attempt": attempted, "optimizer_steps": completed, "skipped_steps": skipped,
                      "train_loss": loss.item(), "lr": optimizer.param_groups[0]["lr"],
                      "elapsed_seconds": round(time.monotonic() - run_started, 1)}
            row["history"].append(record)
            print(json.dumps({"arm": label, **record}), flush=True)
            atomic_write_text(json.dumps(row, indent=2) + "\n", out / "progress.json")
    row.update(completed_steps=completed, attempted_steps=attempted, skipped_steps=skipped,
               target_tokens_processed=attempted * cfg.batch_size * cfg.grad_accum_steps * cfg.block_size,
               peak_memory_gib=torch.cuda.max_memory_allocated() / 2 ** 30 if args.device.startswith("cuda") else None)
    row["evaluation"] = measure(model, plan, arrays, args.device, args.eval_batch_size)
    filename = "final.pt" if completed == args.steps and not skipped else "partial.pt"
    checkpoint = out / filename
    atomic_save({"model_state": model.state_dict(), "model_cfg": model_cfg, "tokenizer": tokenizer.to_dict(),
                 "training": {"method": "weights-only fresh-optimizer sampler control", "step": completed,
                              "parent_checkpoint_sha256": args.expect_base_sha256,
                              "corpus_hash": args.expect_data_sha256, "sampling_mode": mode,
                              "validation_plan_sha256": args.expect_plan_sha256,
                              "initial_optimizer_states": 0, "actual_context": cfg.block_size,
                              "optimizer_resume_available": False}}, checkpoint)
    row["checkpoint"] = str(checkpoint)
    row["checkpoint_sha256"] = file_sha256(checkpoint)
    atomic_write_text(json.dumps(row, indent=2) + "\n", out / "result.json")
    del optimizer, scaler, model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return row


def run(args):
    if not 1 <= args.steps <= 200 or min(args.batch_size, args.accumulation, args.eval_batch_size, args.warmup) < 1:
        raise ValueError("experiment requires 1..200 attempts and positive batch/accumulation/evaluation/warmup sizes")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0 or args.warmup > args.steps or args.max_seconds <= 0:
        raise ValueError("invalid learning schedule or deadline")
    out = Path(args.output)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("experiment output must be a new or empty directory")
    if Path(args.base).resolve().is_relative_to(out.resolve()):
        raise ValueError("experiment output cannot contain the accepted base")
    manifest, digest, tokenizer = load_manifest(args.data_dir, verify=True, expect_sha256=args.expect_data_sha256)
    plan, arrays = load_plan(args.validation_plan, args.data_dir, manifest, args.expect_plan_sha256, digest)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("the declared CUDA experiment cannot silently fall back to CPU")
    started = time.monotonic()
    with deterministic_runtime():
        base_model, _, model_cfg = warm_start(args.base, args.expect_base_sha256, tokenizer, args.device, args.seed)
        if plan["block_size"] > model_cfg["block_size"]:
            raise ValueError("plan context exceeds the accepted model context")
        initial_sha = tensor_digest(base_model)
        report = {"protocol": "base-sampler-control-v1", "base_checkpoint_sha256": args.expect_base_sha256,
                  "data_manifest_sha256": digest, "validation_plan_sha256": args.expect_plan_sha256,
                  "recipe": {"steps": args.steps, "batch_size": args.batch_size, "accumulation": args.accumulation,
                             "context": plan["block_size"], "learning_rate": args.learning_rate,
                             "warmup": args.warmup, "seed": args.seed, "initial_fp16_scale": 1024.0},
                  "environment": {"device": args.device, "torch_version": str(torch.__version__),
                                  "gpu": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
                                  "deterministic_algorithms": True, "attention_backend": "sdpa_math",
                                  "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"]},
                  "base": {"tensor_sha256": initial_sha, "evaluation": measure(base_model, plan, arrays, args.device, args.eval_batch_size)},
                  "arms": {}, "accepted_model_changed": False}
        del base_model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        out.mkdir(parents=True, exist_ok=True)
        atomic_write_text(json.dumps(report, indent=2) + "\n", out / "experiment.json")
        for label, mode in (("control", LEGACY_PREFIX), ("treatment", UNIFORM_WINDOWS)):
            report["arms"][label] = run_arm(label, mode, args, plan, arrays, manifest, tokenizer, initial_sha, started)
            atomic_write_text(json.dumps(report, indent=2) + "\n", out / "experiment.json")
        report["assessment"] = assess(report)
        report["elapsed_seconds"] = round(time.monotonic() - started, 1)
        if file_sha256(args.base) != args.expect_base_sha256:
            raise ValueError("accepted base changed during experiment")
        atomic_write_text(json.dumps(report, indent=2) + "\n", out / "experiment.json")
        print(json.dumps(report["assessment"], indent=2), flush=True)
        return report


def generate_screen(args):
    from evals.bench_run import generate_all
    report = json.loads(Path(args.experiment).read_text())
    assessment = assess(report)
    if not assessment["eligible_for_capability_screen"]:
        print("Primary criterion not met; capability generation skipped.", flush=True)
        return 0
    if file_sha256(args.base) != report["base_checkpoint_sha256"]:
        raise ValueError("capability baseline is not the accepted checkpoint")
    checkpoints = {"base": args.base, **{label: row["checkpoint"] for label, row in report["arms"].items()}}
    with deterministic_runtime():
        for label, checkpoint in checkpoints.items():
            expected = report["base_checkpoint_sha256"] if label == "base" else report["arms"][label]["checkpoint_sha256"]
            if file_sha256(checkpoint) != expected:
                raise ValueError("arm checkpoint changed before capability generation")
            for chat in (False, True):
                result = generate_all(checkpoint, args.bench_dir, limit=25, chat=chat, max_new_tokens=384, device=args.device)
                mode = "chat" if chat else "completion"
                path = Path(args.output) / f"programs-{label}-{mode}.json"
                atomic_write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", path)
                print(json.dumps({"generated": str(path), "programs": len(result["programs"])}), flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Bounded weights-only prefix-versus-uniform base-training control.")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("run")
    train.add_argument("--base", required=True)
    train.add_argument("--expect-base-sha256", required=True)
    train.add_argument("--data-dir", required=True)
    train.add_argument("--expect-data-sha256", required=True)
    train.add_argument("--validation-plan", required=True)
    train.add_argument("--expect-plan-sha256", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--steps", type=int, default=200)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--accumulation", type=int, default=4)
    train.add_argument("--eval-batch-size", type=int, default=4)
    train.add_argument("--learning-rate", type=float, default=1e-5)
    train.add_argument("--warmup", type=int, default=20)
    train.add_argument("--seed", type=int, default=1337)
    train.add_argument("--max-seconds", type=float, default=3600)
    train.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    generate = sub.add_parser("generate")
    generate.add_argument("--experiment", required=True)
    generate.add_argument("--base", required=True)
    generate.add_argument("--bench-dir", required=True)
    generate.add_argument("--output", required=True)
    generate.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    if args.command == "generate":
        return generate_screen(args)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
