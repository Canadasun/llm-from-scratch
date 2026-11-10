import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from data.shards import ShardSplit, load_manifest
from evals.loader import load_checkpoint
from special_tokens import ASSISTANT, ENDOFTEXT, USER
from train import StopRequest, atomic_save, atomic_write_text, file_sha256, make_autocast, pick_device

AUTHORED = ("secure-authored", "refusal", "insufficient-info")


def encode_row(row, tokenizer, context):
    user, assistant, eot = (tokenizer.special_id(t) for t in (USER, ASSISTANT, ENDOFTEXT))
    prompt = [user] + tokenizer.encode(row["prompt"]) + [assistant]
    full = prompt + tokenizer.encode(row["response"]) + [eot]
    if len(full) - 1 > context:
        return None
    x = torch.tensor(full[:-1], dtype=torch.long)
    y = torch.tensor(full[1:], dtype=torch.long)
    y[:len(prompt) - 1] = -100
    return x, y


def load_rows(path, tokenizer, context):
    rows, encoded, dropped = [], [], 0
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        pair = encode_row(row, tokenizer, context)
        if pair is None:
            dropped += 1
            continue
        rows.append(row)
        encoded.append(pair)
    if not encoded:
        raise ValueError(f"no usable rows in {path}")
    return rows, encoded, dropped


def parse_upsample(text):
    out = {}
    for item in filter(None, (text or "").split(",")):
        kind, factor = item.split("=")
        out[kind.strip()] = int(factor)
        if out[kind.strip()] < 1:
            raise ValueError("upsample factors must be at least 1")
    return out


def schedule(rows, upsample):
    return [i for i, row in enumerate(rows) for _ in range(upsample.get(row["kind"], 1))]


def pad(pairs, device):
    width = max(len(x) for x, _ in pairs)
    x = torch.zeros((len(pairs), width), dtype=torch.long)
    y = torch.full((len(pairs), width), -100, dtype=torch.long)
    for i, (a, b) in enumerate(pairs):
        x[i, :len(a)] = a
        y[i, :len(b)] = b
    return x.to(device), y.to(device)


class Sampler:
    def __init__(self, order, seed):
        self.order, self.seed, self.epoch, self.position = list(order), seed, 0, 0
        self._shuffle()

    def _shuffle(self):
        self.perm = list(self.order)
        random.Random(self.seed * 1000003 + self.epoch).shuffle(self.perm)

    def take(self, n):
        out = []
        while len(out) < n:
            if self.position >= len(self.perm):
                self.epoch, self.position = self.epoch + 1, 0
                self._shuffle()
            out.append(self.perm[self.position])
            self.position += 1
        return out

    def state(self):
        return {"epoch": self.epoch, "position": self.position}

    def restore(self, state):
        self.epoch, self.position = state["epoch"], state["position"]
        self._shuffle()


def lr_at(step, peak, warmup, total, floor=0.1):
    if step < warmup:
        return peak * (step + 1) / warmup
    ratio = min(1.0, (step - warmup) / max(1, total - warmup))
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * ratio)))


@torch.no_grad()
def response_loss(model, pairs, device, autocast_ctx, batch_size=8):
    model.eval()
    total, count = 0.0, 0
    for start in range(0, len(pairs), batch_size):
        x, y = pad(pairs[start:start + batch_size], device)
        with autocast_ctx():
            logits, _ = model(x, y)
        loss = torch.nn.functional.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1),
                                                 ignore_index=-100, reduction="sum")
        total += loss.item()
        count += (y != -100).sum().item()
    model.train()
    return total / count


@torch.no_grad()
def replay_loss(model, split, context, device, autocast_ctx, batches=8, batch_size=8, seed=7919):
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(batches):
        x, y = split.sample(batch_size, context, generator)
        with autocast_ctx():
            _, loss = model(x.to(device), y.to(device))
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def tune(args):
    out = Path(args.output)
    base = Path(args.base).resolve()
    if base.parent == out.resolve():
        raise ValueError("write the candidate to a new directory, never beside the base checkpoint")
    base_sha = file_sha256(base)
    if args.expect_base_sha256 and base_sha != args.expect_base_sha256:
        raise SystemExit(f"base checkpoint sha256 {base_sha} does not match the pinned value")
    data_dir = Path(args.data)
    data_sha = file_sha256(data_dir / "manifest.json")
    if args.expect_data_sha256 and data_sha != args.expect_data_sha256:
        raise SystemExit(f"instruction manifest sha256 {data_sha} does not match the pinned value")
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    model, tokenizer, model_cfg = load_checkpoint(str(base), device)
    model.train()
    context = min(args.context, model_cfg["block_size"])
    if any(tokenizer.special_id(t) is None for t in (USER, ASSISTANT, ENDOFTEXT)):
        raise ValueError("base tokenizer lacks chat special tokens")
    rows, train_pairs, dropped_train = load_rows(data_dir / "train.jsonl", tokenizer, context)
    val_rows, val_pairs, dropped_val = load_rows(data_dir / "validation.jsonl", tokenizer, context)
    if {r["family"] for r in rows} & {r["family"] for r in val_rows}:
        raise ValueError("training and validation families overlap")
    upsample = parse_upsample(args.upsample)
    order = schedule(rows, upsample)
    replay = replay_val = None
    replay_sha = None
    if args.replay_dir and args.replay_fraction > 0:
        manifest, replay_sha, replay_tok = load_manifest(args.replay_dir, verify=not args.skip_replay_hash,
                                                         expect_sha256=args.expect_replay_sha256)
        if replay_tok.to_dict() != tokenizer.to_dict():
            raise ValueError("replay shards use a different tokenizer from the base checkpoint")
        replay = ShardSplit(args.replay_dir, "train", manifest)
        replay_val = ShardSplit(args.replay_dir, "val", manifest)
    autocast_ctx, scaler, amp_dtype = make_autocast(device)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                                   {"params": no_decay, "weight_decay": 0.0}],
                                  lr=args.learning_rate, betas=(0.9, 0.95),
                                  **({"fused": True} if device.startswith("cuda") else {}))
    sampler = Sampler(order, args.seed)
    replay_gen = torch.Generator().manual_seed(args.seed + 1)
    mix = random.Random(args.seed + 2)
    step, history, best = 0, [], None
    meta = {"method": "response-only instruction tuning with base-corpus replay",
            "base_checkpoint_sha256": base_sha, "instruction_manifest_sha256": data_sha,
            "replay_manifest_sha256": replay_sha, "replay_fraction": args.replay_fraction if replay else 0.0,
            "upsample": upsample, "context": context, "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps, "learning_rate": args.learning_rate,
            "warmup": args.warmup, "steps": args.steps, "seed": args.seed, "amp_dtype": amp_dtype,
            "train_rows": len(rows), "train_rows_dropped_for_context": dropped_train,
            "validation_rows": len(val_rows), "validation_rows_dropped_for_context": dropped_val,
            "schedule_rows": len(order), "torch_version": str(torch.__version__)}
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if state["meta"]["base_checkpoint_sha256"] != base_sha or state["meta"]["instruction_manifest_sha256"] != data_sha:
            raise ValueError("resume state belongs to a different base or instruction set")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scaler.load_state_dict(state["scaler_state"])
        sampler.restore(state["sampler"])
        replay_gen.set_state(state["replay_rng"])
        version, internal, gauss = state["mix_rng"]
        mix.setstate((version, tuple(internal), gauss))
        torch.set_rng_state(state["torch_rng"])
        step, history, best = state["step"], state["history"], state["best"]
        meta["resumed_from_step"] = step
    out.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        base_val = response_loss(model, val_pairs, device, autocast_ctx)
        record = {"step": 0, "validation_loss": base_val}
        if replay_val is not None:
            record["replay_val_loss"] = replay_loss(model, replay_val, context, device, autocast_ctx)
        history.append(record)
        print(json.dumps(record), flush=True)
    stop = StopRequest().install()
    started = time.monotonic()
    last_save = started
    state_reason = "completed"

    def save_state():
        atomic_save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                     "scaler_state": scaler.state_dict(), "sampler": sampler.state(),
                     "replay_rng": replay_gen.get_state(), "mix_rng": list(mix.getstate()),
                     "torch_rng": torch.get_rng_state(), "step": step, "history": history, "best": best,
                     "meta": meta, "model_cfg": model_cfg, "tokenizer": tokenizer.to_dict()}, out / "ckpt.pt")

    def write_status(state):
        status = {"state": state, "completed_steps": step, "max_iters": args.steps, "best_val_loss": best,
                  "meta": meta, "elapsed_seconds": round(time.monotonic() - started, 1),
                  "ckpt_sha256": file_sha256(out / "ckpt.pt")}
        if (out / "best.pt").exists():
            status["best_sha256"] = file_sha256(out / "best.pt")
        atomic_write_text(json.dumps(status, indent=2) + "\n", out / "status.json")
        atomic_write_text("".join(json.dumps(r) + "\n" for r in history), out / "train_log.jsonl")

    try:
        while step < args.steps:
            if stop.reason:
                state_reason = stop.reason
                break
            if args.deadline_seconds and time.monotonic() - started > args.deadline_seconds:
                state_reason = "deadline"
                break
            if args.stop_at_step and step >= args.stop_at_step:
                state_reason = "stopped"
                break
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, args.learning_rate, args.warmup, args.steps)
            optimizer.zero_grad(set_to_none=True)
            total = 0.0
            for _ in range(args.grad_accum_steps):
                if replay is not None and mix.random() < args.replay_fraction:
                    x, y = replay.sample(args.batch_size, context, replay_gen)
                    x, y = x.to(device), y.to(device)
                else:
                    x, y = pad([train_pairs[i] for i in sampler.take(args.batch_size)], device)
                with autocast_ctx():
                    _, loss = model(x, y)
                    loss = loss / args.grad_accum_steps
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite training loss")
                scaler.scale(loss).backward()
                total += loss.item()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            step += 1
            if step % args.eval_interval == 0 or step == args.steps:
                record = {"step": step, "train_loss": total,
                          "validation_loss": response_loss(model, val_pairs, device, autocast_ctx),
                          "lr": optimizer.param_groups[0]["lr"],
                          "elapsed_seconds": round(time.monotonic() - started, 1)}
                if replay_val is not None:
                    record["replay_val_loss"] = replay_loss(model, replay_val, context, device, autocast_ctx)
                history.append(record)
                print(json.dumps(record), flush=True)
                if best is None or record["validation_loss"] < best:
                    best = record["validation_loss"]
                    atomic_save({"model_state": model.state_dict(), "model_cfg": model_cfg,
                                 "tokenizer": tokenizer.to_dict(),
                                 "instruction_tuning": {**meta, "step": step, "validation_loss": best}},
                                out / "best.pt")
            if args.checkpoint_minutes and time.monotonic() - last_save > args.checkpoint_minutes * 60:
                save_state()
                write_status("running")
                last_save = time.monotonic()
    finally:
        stop.restore()
    save_state()
    if file_sha256(base) != base_sha:
        raise SystemExit("base checkpoint changed during tuning")
    write_status(state_reason)
    return {"state": state_reason, "step": step, "best_val_loss": best,
            "base_val_loss": history[0]["validation_loss"] if history else None}


def build_parser():
    ap = argparse.ArgumentParser(description="Response-only instruction tuning from a pretrained checkpoint.")
    ap.add_argument("--base", required=True)
    ap.add_argument("--expect-base-sha256")
    ap.add_argument("--data", required=True)
    ap.add_argument("--expect-data-sha256")
    ap.add_argument("--replay-dir")
    ap.add_argument("--expect-replay-sha256")
    ap.add_argument("--skip-replay-hash", action="store_true")
    ap.add_argument("--replay-fraction", type=float, default=0.2)
    ap.add_argument("--upsample", default=",".join(f"{k}=8" for k in AUTHORED))
    ap.add_argument("--output", required=True)
    ap.add_argument("--resume")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum-steps", type=int, default=4)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--eval-interval", type=int, default=200)
    ap.add_argument("--checkpoint-minutes", type=float, default=15)
    ap.add_argument("--deadline-seconds", type=float)
    ap.add_argument("--stop-at-step", type=int)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=2026)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    print(json.dumps(tune(args)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
