"""
Train the from-scratch GPT on data/corpus.txt (Milestone 1, CPU).

Run:
    python data/prepare.py            # writes data/corpus.txt
    python train.py                   # trains, saves checkpoints/ckpt.pt

This is a complete, readable training loop: batching, an AdamW optimizer,
gradient clipping, periodic train/val loss estimates, and a checkpoint save.
"""

import os

import torch

from config import TrainConfig
from model import GPT, GPTConfig
from tokenizer import CharTokenizer

CORPUS = os.path.join("data", "corpus.txt")


def get_batch(data, block_size, batch_size, device):
    """Pick batch_size random windows of length block_size.

    x is the window; y is the same window shifted one token to the right, so
    the model learns to predict the next token at every position at once.
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, splits, cfg, device):
    """Average loss over a few batches of each split, so the printed numbers
    are less noisy than a single-batch loss."""
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            x, y = get_batch(data, cfg.block_size, cfg.batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def main():
    cfg = TrainConfig()
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if not os.path.exists(CORPUS):
        raise SystemExit("data/corpus.txt not found. Run: python data/prepare.py")

    with open(CORPUS, encoding="utf-8") as f:
        text = f.read()

    tokenizer = CharTokenizer.from_text(text)
    print(f"Corpus: {len(text):,} chars, vocab size {tokenizer.vocab_size}")

    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n_val = int(len(data) * cfg.val_split)
    splits = {"train": data[:-n_val], "val": data[-n_val:]}

    model_cfg = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout,
    )
    model = GPT(model_cfg).to(device)
    print(f"Model parameters: {model.num_params():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    for it in range(cfg.max_iters + 1):
        if it % cfg.eval_interval == 0:
            losses = estimate_loss(model, splits, cfg, device)
            print(f"iter {it:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

        x, y = get_batch(splits["train"], cfg.block_size, cfg.batch_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

    os.makedirs(cfg.out_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.out_dir, "ckpt.pt")
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_cfg": model_cfg.__dict__,
            "chars": tokenizer.chars,
        },
        ckpt_path,
    )
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
