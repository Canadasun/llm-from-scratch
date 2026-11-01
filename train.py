"""
Train the from-scratch GPT (Milestone 2): char or BPE tokenizer, CPU or GPU.

Examples
--------
CPU, quick, char-level (Milestone 1 behaviour):
    python data/build_corpus.py
    python train.py --preset cpu

GPU (Colab/Kaggle), bigger model + BPE tokenizer (Milestone 2 target):
    python train.py --preset gpu

The script auto-detects CUDA. On GPU it uses mixed precision (autocast) and,
for fp16, a gradient scaler. The trained checkpoint embeds its tokenizer, so
`sample.py` needs only the checkpoint.
"""

import argparse
import hashlib
import os
import time
import tempfile

import torch

import tokenization
from config import PRESETS, TrainConfig
from model import GPT, GPTConfig

CORPUS = os.path.join("data", "corpus.txt")


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


def save_training_checkpoint(path, model, model_cfg, tokenizer, optimizer,
                             scaler, cfg, completed_steps, corpus_hash, device):
    raw_model = getattr(model, "_orig_mod", model)
    atomic_save({
        "model_state": raw_model.state_dict(), "model_cfg": model_cfg.__dict__,
        "tokenizer": tokenizer.to_dict(), "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(), "train_cfg": cfg.__dict__,
        "completed_steps": completed_steps, "corpus_hash": corpus_hash,
        "rng_state": torch.get_rng_state(), "training_device": device,
        "cuda_rng_state": torch.cuda.get_rng_state_all() if device.startswith("cuda") else [],
        "mps_rng_state": torch.mps.get_rng_state() if device.startswith("mps") else None,
    }, path)


def pick_device(requested):
    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_batch(data, block_size, batch_size, device):
    """Pick batch_size random windows of length block_size.

    x is the window; y is the same window shifted one token right, so the model
    learns to predict the next token at every position at once.
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    if device.startswith("cuda"):
        # Pinned + non-blocking transfer overlaps copy with compute.
        return (x.pin_memory().to(device, non_blocking=True),
                y.pin_memory().to(device, non_blocking=True))
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, splits, cfg, device, autocast_ctx):
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            x, y = get_batch(data, cfg.block_size, cfg.batch_size, device)
            with autocast_ctx():
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def lr_for(it, cfg):
    """Linear warmup then constant. Keeps the bigger GPU model stable early."""
    if cfg.warmup_iters and it < cfg.warmup_iters:
        return cfg.learning_rate * (it + 1) / cfg.warmup_iters
    return cfg.learning_rate


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(PRESETS), default="cpu")
    ap.add_argument("--tokenizer", choices=["char", "bpe"], default=None,
                    help="override the preset's tokenizer")
    ap.add_argument("--vocab-size", type=int, default=None)
    ap.add_argument("--bpe-train-chars", type=int, default=None,
                    help="chars the BPE tokenizer learns merges from")
    ap.add_argument("--max-iters", type=int, default=None)
    ap.add_argument("--resume", help="resume a training checkpoint, preserving optimizer and RNG")
    ap.add_argument("--out-dir", help="checkpoint directory")
    ap.add_argument("--checkpoint-interval", type=int, default=None,
                    help="save every N completed steps (default: evaluation interval)")
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    ap.add_argument("--corpus", default=CORPUS,
                    help="training corpus file (default data/corpus.txt). Point at a "
                         "mixed prose+code corpus from data/build_training_corpus.py.")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the model (PyTorch 2.x, CUDA)")
    args = ap.parse_args()

    resumed = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    if resumed is not None:
        required = {"train_cfg", "optimizer_state", "scaler_state", "completed_steps",
                    "corpus_hash", "rng_state", "training_device", "cuda_rng_state"}
        if required - resumed.keys():
            ap.error("checkpoint lacks training state; inference-only checkpoints cannot resume")
        cfg = TrainConfig(**resumed["train_cfg"])
    else:
        cfg = PRESETS[args.preset]()
    if args.tokenizer:
        cfg.tokenizer = args.tokenizer
    if args.vocab_size:
        cfg.vocab_size = args.vocab_size
    if args.bpe_train_chars is not None:
        cfg.bpe_train_chars = args.bpe_train_chars
    if args.max_iters is not None:
        cfg.max_iters = args.max_iters
    if args.out_dir:
        cfg.out_dir = args.out_dir
    checkpoint_interval = args.checkpoint_interval or cfg.eval_interval
    if cfg.max_iters < 1 or checkpoint_interval < 1 or args.checkpoint_interval == 0:
        ap.error("max-iters and checkpoint-interval must be positive")

    torch.manual_seed(cfg.seed)
    device = pick_device(args.device)
    if resumed is not None and device != resumed["training_device"]:
        ap.error("resume requires the original training device for RNG compatibility")
    if device == "cpu":
        torch.set_num_threads(os.cpu_count() or 1)
    print(f"Preset: {args.preset} | device: {device} | tokenizer: {cfg.tokenizer}",
          flush=True)

    if not os.path.exists(args.corpus):
        raise SystemExit(f"{args.corpus} not found. Run: python data/build_corpus.py "
                         f"(prose) and/or python data/build_training_corpus.py (prose+code)")
    with open(args.corpus, encoding="utf-8") as f:
        text = f.read()
    corpus_hash = hashlib.sha256(text.encode()).hexdigest()
    if resumed is not None and corpus_hash != resumed["corpus_hash"]:
        ap.error("corpus differs from the resume checkpoint")

    tokenizer, data = load_or_build_tokenizer_and_ids(text, cfg, device)
    n_val = max(1, int(len(data) * cfg.val_split))
    splits = {"train": data[:-n_val], "val": data[-n_val:]}
    if any(len(split) <= cfg.block_size for split in splits.values()):
        ap.error("both training and validation splits need more than block_size tokens")

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

    # Mixed precision on CUDA: bf16 if the GPU supports it (no scaler needed),
    # otherwise fp16 with a gradient scaler. CPU/MPS run in fp32.
    use_amp = device.startswith("cuda")
    amp_dtype = (torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported()
                 else torch.float16)
    if use_amp:
        def autocast_ctx():
            return torch.autocast(device_type="cuda", dtype=amp_dtype)
    else:
        import contextlib
        autocast_ctx = contextlib.nullcontext
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    if args.compile and hasattr(torch, "compile"):
        print("Compiling model (first iters are slow)...", flush=True)
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    start = 0
    if resumed is not None:
        optimizer.load_state_dict(resumed["optimizer_state"])
        scaler.load_state_dict(resumed["scaler_state"])
        torch.set_rng_state(resumed["rng_state"])
        if device.startswith("cuda"):
            torch.cuda.set_rng_state_all(resumed["cuda_rng_state"])
        if device.startswith("mps"):
            if resumed.get("mps_rng_state") is None:
                ap.error("checkpoint lacks MPS RNG state")
            torch.mps.set_rng_state(resumed["mps_rng_state"])
        start = resumed["completed_steps"]
        if start > cfg.max_iters:
            ap.error("max-iters is a total step count and cannot precede the resume step")

    ckpt_path = os.path.join(cfg.out_dir, "ckpt.pt")

    print(f"Training for {cfg.max_iters} iters...", flush=True)
    t0 = time.time()
    for it in range(start, cfg.max_iters):
        for g in optimizer.param_groups:
            g["lr"] = lr_for(it, cfg)

        if it % cfg.eval_interval == 0:
            losses = estimate_loss(model, splits, cfg, device, autocast_ctx)
            print(f"iter {it:5d}/{cfg.max_iters} ({100 * it / cfg.max_iters:3.0f}%) | "
                  f"train {losses['train']:.4f} | val {losses['val']:.4f} | "
                  f"{time.time() - t0:5.1f}s", flush=True)

        x, y = get_batch(splits["train"], cfg.block_size, cfg.batch_size, device)
        with autocast_ctx():
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if (it + 1) % checkpoint_interval == 0 and it + 1 < cfg.max_iters:
            save_training_checkpoint(ckpt_path, model, model_cfg, tokenizer,
                                     optimizer, scaler, cfg, it + 1, corpus_hash, device)

    os.makedirs(cfg.out_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.out_dir, "ckpt.pt")
    save_training_checkpoint(ckpt_path, model, model_cfg, tokenizer, optimizer,
                             scaler, cfg, cfg.max_iters, corpus_hash, device)
    print(f"Saved checkpoint to {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
