"""
Train the from-scratch GPT: char or BPE tokenizer, CPU or GPU.

Examples
--------
CPU, quick, char-level:
    python data/build_corpus.py
    python train.py --preset cpu

GPU on pinned token shards (python -m data.shards ...), stopping safely before a
free-GPU session ends and resuming in the next session:
    python train.py --preset gpu_base --data-dir DATA --out-dir RUN --deadline-seconds 30000
    python train.py --resume RUN/ckpt.pt --data-dir DATA --out-dir RUN2 --deadline-seconds 30000

The script auto-detects CUDA. On GPU it uses mixed precision (autocast) and,
for fp16, a gradient scaler. The trained checkpoint embeds its tokenizer, so
`sample.py` needs only the checkpoint.
"""

import argparse
import contextlib
import hashlib
import json
import math
import os
import signal
import time
import tempfile

import torch

import tokenization
from config import PRESETS, TrainConfig
from model import GPT, GPTConfig

CORPUS = os.path.join("data", "corpus.txt")
CHECKPOINT_FORMAT = 2


def atomic_save(blob, path):
    """Replace a checkpoint only after its complete contents reach disk."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".checkpoint-", dir=directory)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(blob, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_text(text, path):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".status-", dir=directory)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def file_sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def save_training_checkpoint(path, model, model_cfg, tokenizer, optimizer,
                             scaler, cfg, completed_steps, corpus_hash, device,
                             batch_generator=None, best_val_loss=None, data_kind="text"):
    raw_model = getattr(model, "_orig_mod", model)
    atomic_save({
        "format_version": CHECKPOINT_FORMAT,
        "model_state": raw_model.state_dict(), "model_cfg": model_cfg.__dict__,
        "tokenizer": tokenizer.to_dict(), "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(), "train_cfg": cfg.__dict__,
        "completed_steps": completed_steps, "corpus_hash": corpus_hash, "data_kind": data_kind,
        "rng_state": torch.get_rng_state(), "training_device": device,
        "cuda_rng_state": torch.cuda.get_rng_state_all() if device.startswith("cuda") else [],
        "mps_rng_state": torch.mps.get_rng_state() if device.startswith("mps") else None,
        "batch_rng_state": batch_generator.get_state() if batch_generator is not None else None,
        "best_val_loss": best_val_loss, "torch_version": str(torch.__version__),
    }, path)


def save_inference_checkpoint(path, model, model_cfg, tokenizer, step, val_loss, corpus_hash):
    raw_model = getattr(model, "_orig_mod", model)
    atomic_save({"format_version": CHECKPOINT_FORMAT, "model_state": raw_model.state_dict(),
                 "model_cfg": model_cfg.__dict__, "tokenizer": tokenizer.to_dict(),
                 "training": {"step": step, "val_loss": val_loss, "corpus_hash": corpus_hash}}, path)


def pick_device(requested):
    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TensorSplit:
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def sample(self, batch_size, block_size, generator, doc_aligned=True):
        ix = torch.randint(len(self.data) - block_size, (batch_size,), generator=generator)
        x = torch.stack([self.data[i:i + block_size] for i in ix])
        y = torch.stack([self.data[i + 1:i + 1 + block_size] for i in ix])
        return x, y


def to_device(x, y, device):
    if device.startswith("cuda"):
        return (x.pin_memory().to(device, non_blocking=True),
                y.pin_memory().to(device, non_blocking=True))
    return x.to(device), y.to(device)


def get_batch(data, block_size, batch_size, device, generator=None):
    """Pick batch_size random windows of length block_size.

    x is the window; y is the same window shifted one token right, so the model
    learns to predict the next token at every position at once.
    """
    split = data if hasattr(data, "sample") else TensorSplit(data)
    return to_device(*split.sample(batch_size, block_size, generator), device)


@torch.no_grad()
def estimate_loss(model, splits, cfg, device, autocast_ctx):
    model.eval()
    out = {}
    for name, data in splits.items():
        generator = torch.Generator().manual_seed(cfg.seed + 7919)
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            x, y = to_device(*data.sample(cfg.batch_size, cfg.block_size, generator, cfg.doc_aligned), device)
            with autocast_ctx():
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def lr_for(it, cfg):
    """Linear warmup, then cosine decay to min_lr_ratio * learning_rate."""
    if cfg.warmup_iters and it < cfg.warmup_iters:
        return cfg.learning_rate * (it + 1) / cfg.warmup_iters
    if cfg.lr_schedule == "constant":
        return cfg.learning_rate
    min_lr = cfg.learning_rate * cfg.min_lr_ratio
    decay_iters = cfg.lr_decay_iters or cfg.max_iters
    if it >= decay_iters:
        return min_lr
    ratio = (it - cfg.warmup_iters) / max(1, decay_iters - cfg.warmup_iters)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (cfg.learning_rate - min_lr)


def configure_optimizer(model, cfg, device):
    raw_model = getattr(model, "_orig_mod", model)
    decay, no_decay = [], []
    for _, p in raw_model.named_parameters():
        if p.requires_grad:
            (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": cfg.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    kwargs = {"lr": cfg.learning_rate, "betas": (0.9, cfg.beta2)}
    if device.startswith("cuda"):
        kwargs["fused"] = True
    return torch.optim.AdamW(groups, **kwargs)


def make_autocast(device):
    use_amp = device.startswith("cuda")
    amp_dtype = (torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported()
                 else torch.float16)
    if use_amp:
        def autocast_ctx():
            return torch.autocast(device_type="cuda", dtype=amp_dtype)
    else:
        autocast_ctx = contextlib.nullcontext
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))
    return autocast_ctx, scaler, (str(amp_dtype).replace("torch.", "") if use_amp else "float32")


def train_step(model, optimizer, scaler, split, cfg, device, autocast_ctx, generator):
    optimizer.zero_grad(set_to_none=True)
    total = None
    for _ in range(cfg.grad_accum_steps):
        x, y = to_device(*split.sample(cfg.batch_size, cfg.block_size, generator, cfg.doc_aligned), device)
        with autocast_ctx():
            _, loss = model(x, y)
            loss = loss / cfg.grad_accum_steps
        scaler.scale(loss).backward()
        total = loss.detach() if total is None else total + loss.detach()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    scaler.step(optimizer)
    scaler.update()
    return total


def load_or_build_tokenizer_and_ids(text, cfg, device):
    """Build the tokenizer and encode the corpus, caching both to disk so the
    one-time BPE training + encoding cost is never paid twice for the same
    corpus/settings."""
    # Hash the FULL corpus, not just length + first/last 200 chars: an interior
    # edit that keeps those identical would otherwise collide and silently reuse
    # the previous corpus's cached tokens.
    corpus_hash = hashlib.sha256(text.encode()).hexdigest()
    sig = hashlib.sha256(
        f"{cfg.tokenizer}|{cfg.vocab_size}|{cfg.bpe_train_chars}|"
        f"{len(text)}|{corpus_hash}".encode()
    ).hexdigest()[:16]
    os.makedirs("data", exist_ok=True)
    cache = os.path.join("data", f".cache_{cfg.tokenizer}_{sig}.pt")

    if os.path.exists(cache):
        blob = torch.load(cache, weights_only=True)
        tok = tokenization.from_dict(blob["tok"])
        ids = blob["ids"]
        print(f"Loaded tokenizer+ids cache ({cache})", flush=True)
        return tok, ids

    print(f"Building {cfg.tokenizer} tokenizer...", flush=True)
    t = time.time()
    tok = tokenization.build(cfg.tokenizer, text, vocab_size=cfg.vocab_size,
                             bpe_train_chars=cfg.bpe_train_chars, verbose=True)
    print(f"Encoding corpus with {tok.kind} tokenizer...", flush=True)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    print(f"Tokenizer ready in {time.time() - t:.1f}s "
          f"(vocab {tok.vocab_size}, {len(ids):,} tokens)", flush=True)
    atomic_save({"tok": tok.to_dict(), "ids": ids}, cache)
    return tok, ids


class StopRequest:
    def __init__(self):
        self.reason = None
        self._previous = {}

    def install(self):
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                self._previous[sig] = signal.signal(sig, self._handle)
            except ValueError:
                pass
        return self

    def restore(self):
        for sig, handler in self._previous.items():
            signal.signal(sig, handler)
        self._previous = {}

    def _handle(self, signum, frame):
        self.reason = f"signal-{signal.Signals(signum).name}"


def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(PRESETS), default="cpu")
    ap.add_argument("--tokenizer", choices=["char", "bpe"], default=None,
                    help="override the preset's tokenizer")
    ap.add_argument("--vocab-size", type=int, default=None)
    ap.add_argument("--bpe-train-chars", type=int, default=None,
                    help="chars the BPE tokenizer learns merges from")
    ap.add_argument("--batch-size", type=int, help="micro-batch size; fixed when resuming")
    ap.add_argument("--grad-accum-steps", type=int, help="micro-batches per optimizer step; fixed when resuming")
    ap.add_argument("--max-iters", type=int, default=None,
                    help="total optimizer steps for this run (not the LR schedule length)")
    ap.add_argument("--lr-decay-iters", type=int, default=None,
                    help="cosine schedule length in steps (default: the preset's max_iters); fixed when resuming")
    ap.add_argument("--learning-rate", type=float, default=None)
    ap.add_argument("--warmup-iters", type=int, default=None)
    ap.add_argument("--eval-interval", type=int, default=None)
    ap.add_argument("--eval-iters", type=int, default=None)
    ap.add_argument("--resume", help="resume a training checkpoint, preserving optimizer and RNG")
    ap.add_argument("--out-dir", help="checkpoint directory")
    ap.add_argument("--checkpoint-interval", type=int, default=None,
                    help="save every N completed steps (default: evaluation interval)")
    ap.add_argument("--checkpoint-minutes", type=float, default=None,
                    help="also save when this many minutes passed since the last save")
    ap.add_argument("--deadline-seconds", type=float, default=None,
                    help="stop with a final checkpoint once this many seconds have passed since start")
    ap.add_argument("--stop-at-step", type=int, default=None,
                    help="stop with a resumable checkpoint after this completed step")
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    ap.add_argument("--corpus", default=CORPUS,
                    help="training corpus file (default data/corpus.txt). Point at a "
                         "mixed prose+code corpus from data/build_training_corpus.py.")
    ap.add_argument("--data-dir", help="pinned token-shard directory from python -m data.shards")
    ap.add_argument("--expect-data-sha256", help="refuse to train unless the shard manifest has this sha256")
    ap.add_argument("--skip-shard-hash", action="store_true",
                    help="check shard sizes but not full content hashes (faster resume on huge sets)")
    ap.add_argument("--log-interval", type=int, default=50)
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the model (PyTorch 2.x, CUDA)")
    return ap


def main(argv=None):
    started = time.monotonic()
    ap = build_arg_parser()
    args = ap.parse_args(argv)

    resumed = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    if resumed is not None:
        required = {"train_cfg", "optimizer_state", "scaler_state", "completed_steps",
                    "corpus_hash", "rng_state", "training_device", "cuda_rng_state",
                    "batch_rng_state"}
        if required - resumed.keys():
            ap.error("checkpoint lacks training state; inference-only checkpoints cannot resume")
        if resumed["batch_rng_state"] is None:
            ap.error("checkpoint lacks batch RNG state; it predates exact resume")
        cfg = TrainConfig(**resumed["train_cfg"])
    else:
        cfg = PRESETS[args.preset]()
        if not cfg.lr_decay_iters:
            cfg.lr_decay_iters = cfg.max_iters
    fixed_on_resume = {"batch_size": args.batch_size, "grad_accum_steps": args.grad_accum_steps,
                       "lr_decay_iters": args.lr_decay_iters, "learning_rate": args.learning_rate,
                       "warmup_iters": args.warmup_iters}
    for name, value in fixed_on_resume.items():
        if value is None:
            continue
        if value <= 0 or (resumed is not None and value != getattr(cfg, name)):
            ap.error(f"{name.replace('_', '-')} must be positive and cannot change on exact resume")
        setattr(cfg, name, value)
    if args.tokenizer:
        cfg.tokenizer = args.tokenizer
    if args.vocab_size:
        cfg.vocab_size = args.vocab_size
    if args.bpe_train_chars is not None:
        cfg.bpe_train_chars = args.bpe_train_chars
    if args.max_iters is not None:
        cfg.max_iters = args.max_iters
    if args.eval_interval is not None:
        cfg.eval_interval = args.eval_interval
    if args.eval_iters is not None:
        cfg.eval_iters = args.eval_iters
    if args.out_dir:
        cfg.out_dir = args.out_dir
    checkpoint_interval = args.checkpoint_interval or cfg.eval_interval
    if cfg.max_iters < 1 or checkpoint_interval < 1 or args.checkpoint_interval == 0:
        ap.error("max-iters and checkpoint-interval must be positive")
    if cfg.eval_interval < 1 or cfg.eval_iters < 1 or args.log_interval < 1:
        ap.error("eval-interval, eval-iters and log-interval must be positive")
    if args.stop_at_step is not None and args.stop_at_step < 1:
        ap.error("stop-at-step must be positive")
    deadline = started + args.deadline_seconds if args.deadline_seconds else None

    torch.manual_seed(cfg.seed)
    device = pick_device(args.device)
    if resumed is not None and device != resumed["training_device"]:
        ap.error("resume requires the original training device for RNG compatibility")
    if device == "cpu":
        torch.set_num_threads(os.cpu_count() or 1)
    print(f"Preset: {args.preset} | device: {device} | tokenizer: {cfg.tokenizer}",
          flush=True)

    if args.data_dir:
        from data.shards import ShardSplit, load_manifest
        try:
            manifest, corpus_hash, tokenizer = load_manifest(
                args.data_dir, verify=not args.skip_shard_hash, expect_sha256=args.expect_data_sha256)
        except (OSError, ValueError, KeyError) as exc:
            ap.error(f"token shards rejected: {exc}")
        splits = {s: ShardSplit(args.data_dir, s, manifest) for s in ("train", "val")}
        data_kind = "shards"
        cfg.tokenizer = tokenizer.kind
        cfg.vocab_size = tokenizer.vocab_size
    else:
        if args.expect_data_sha256:
            ap.error("--expect-data-sha256 needs --data-dir")
        if not os.path.exists(args.corpus):
            raise SystemExit(f"{args.corpus} not found. Run: python data/build_corpus.py "
                             f"(prose) and/or python data/build_training_corpus.py (prose+code)")
        with open(args.corpus, encoding="utf-8") as f:
            text = f.read()
        corpus_hash = hashlib.sha256(text.encode()).hexdigest()
        if resumed is None or corpus_hash == resumed["corpus_hash"]:
            tokenizer, data = load_or_build_tokenizer_and_ids(text, cfg, device)
            n_val = max(1, int(len(data) * cfg.val_split))
            splits = {"train": TensorSplit(data[:-n_val]), "val": TensorSplit(data[-n_val:])}
        data_kind = "text"
    if resumed is not None and (corpus_hash != resumed["corpus_hash"]
                                or data_kind != resumed.get("data_kind", "text")):
        ap.error("corpus differs from the resume checkpoint")
    if any(len(split) <= cfg.block_size for split in splits.values()):
        ap.error("both training and validation splits need more than block_size tokens")
    print(f"Data: {data_kind} sha256={corpus_hash[:16]} | train {len(splits['train']):,} tokens"
          f" | val {len(splits['val']):,} tokens", flush=True)

    model_cfg = GPTConfig(
        vocab_size=tokenizer.vocab_size, block_size=cfg.block_size,
        n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.n_embd, dropout=cfg.dropout,
    )
    model = GPT(model_cfg).to(device)
    if resumed is not None:
        if model_cfg.__dict__ != resumed["model_cfg"] or tokenizer.to_dict() != resumed["tokenizer"]:
            ap.error("model or tokenizer differs from the resume checkpoint")
        model.load_state_dict(resumed["model_state"])
    print(f"Model parameters: {model.num_params():,}", flush=True)

    autocast_ctx, scaler, amp_name = make_autocast(device)
    optimizer = configure_optimizer(model, cfg, device)
    if args.compile and hasattr(torch, "compile"):
        print("Compiling model (first iters are slow)...", flush=True)
        model = torch.compile(model)

    batch_generator = torch.Generator().manual_seed(cfg.seed + 1)
    start, best_val = 0, None
    if resumed is not None:
        optimizer.load_state_dict(resumed["optimizer_state"])
        scaler.load_state_dict(resumed["scaler_state"])
        torch.set_rng_state(resumed["rng_state"])
        batch_generator.set_state(resumed["batch_rng_state"])
        if device.startswith("cuda"):
            states = resumed["cuda_rng_state"]
            for index in range(min(len(states), torch.cuda.device_count())):
                torch.cuda.set_rng_state(states[index], index)
        if device.startswith("mps"):
            if resumed.get("mps_rng_state") is None:
                ap.error("checkpoint lacks MPS RNG state")
            torch.mps.set_rng_state(resumed["mps_rng_state"])
        start = resumed["completed_steps"]
        best_val = resumed.get("best_val_loss")
        if start > cfg.max_iters:
            ap.error("max-iters is a total step count and cannot precede the resume step")

    os.makedirs(cfg.out_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.out_dir, "ckpt.pt")
    best_path = os.path.join(cfg.out_dir, "best.pt")
    log_path = os.path.join(cfg.out_dir, "train_log.jsonl")
    tokens_per_step = cfg.batch_size * cfg.block_size * cfg.grad_accum_steps

    def save(step):
        save_training_checkpoint(ckpt_path, model, model_cfg, tokenizer, optimizer, scaler, cfg,
                                 step, corpus_hash, device, batch_generator, best_val, data_kind)

    def log(record):
        with open(log_path, "a") as fh:
            fh.write(json.dumps(record) + "\n")

    def finish(state, step):
        status = {"state": state, "completed_steps": step, "max_iters": cfg.max_iters,
                  "best_val_loss": best_val, "data_sha256": corpus_hash, "data_kind": data_kind,
                  "device": device, "amp_dtype": amp_name, "tokens_per_step": tokens_per_step,
                  "elapsed_seconds": round(time.monotonic() - started, 1),
                  "resumed_from_step": resumed["completed_steps"] if resumed is not None else None,
                  "ckpt_sha256": file_sha256(ckpt_path)}
        if os.path.exists(best_path):
            status["best_sha256"] = file_sha256(best_path)
        atomic_write_text(json.dumps(status, indent=2) + "\n", os.path.join(cfg.out_dir, "status.json"))
        print(f"Run {state} at step {step}/{cfg.max_iters}; saved {ckpt_path}", flush=True)
        return status

    def evaluate(step):
        nonlocal best_val
        losses = estimate_loss(model, splits, cfg, device, autocast_ctx)
        print(f"iter {step:5d}/{cfg.max_iters} ({100 * step / cfg.max_iters:3.0f}%) | "
              f"train {losses['train']:.4f} | val {losses['val']:.4f} | "
              f"{time.monotonic() - started:5.1f}s", flush=True)
        log({"step": step, "eval_train_loss": losses["train"], "val_loss": losses["val"],
             "elapsed_seconds": round(time.monotonic() - started, 2)})
        if not math.isfinite(losses["val"]):
            raise SystemExit("non-finite validation loss; stopping without overwriting checkpoints")
        if step > 0 and (best_val is None or losses["val"] < best_val):
            best_val = losses["val"]
            save_inference_checkpoint(best_path, model, model_cfg, tokenizer, step, best_val, corpus_hash)

    stopper = StopRequest().install()
    try:
        print(f"Training for {cfg.max_iters} iters ({tokens_per_step:,} tokens/step)...", flush=True)
        last_save = time.monotonic()
        window_start, window_tokens, step_loss = time.monotonic(), 0, None
        for it in range(start, cfg.max_iters):
            for g in optimizer.param_groups:
                g["lr"] = lr_for(it, cfg)

            if it % cfg.eval_interval == 0 and (it > start or resumed is None):
                evaluate(it)
                window_start, window_tokens = time.monotonic(), 0

            step_loss = train_step(model, optimizer, scaler, splits["train"], cfg, device,
                                   autocast_ctx, batch_generator)
            window_tokens += tokens_per_step
            done = it + 1
            if done % args.log_interval == 0:
                loss_value = step_loss.item()
                if not math.isfinite(loss_value):
                    raise SystemExit("non-finite training loss; stopping without overwriting checkpoints")
                elapsed = time.monotonic() - window_start
                record = {"step": done, "train_loss": loss_value, "lr": lr_for(it, cfg),
                          "tokens_per_second": round(window_tokens / elapsed, 1) if elapsed > 0 else None,
                          "elapsed_seconds": round(time.monotonic() - started, 2)}
                if device.startswith("cuda"):
                    record["max_memory_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
                log(record)
                print(json.dumps(record), flush=True)
                window_start, window_tokens = time.monotonic(), 0

            reason = stopper.reason
            if reason is None and deadline is not None and time.monotonic() >= deadline:
                reason = "deadline"
            if reason is None and args.stop_at_step is not None and done >= args.stop_at_step and done < cfg.max_iters:
                reason = "stop-at-step"
            if reason is not None and done < cfg.max_iters:
                save(done)
                return finish(reason, done)
            minutes_due = (args.checkpoint_minutes is not None
                           and time.monotonic() - last_save >= args.checkpoint_minutes * 60)
            if (done % checkpoint_interval == 0 or minutes_due) and done < cfg.max_iters:
                save(done)
                last_save = time.monotonic()

        evaluate(cfg.max_iters)
        save(cfg.max_iters)
        return finish("completed", cfg.max_iters)
    finally:
        stopper.restore()


if __name__ == "__main__":
    main()
