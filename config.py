"""
Training configuration, with a CPU preset and a GPU preset.

The CPU preset is deliberately small so the model trains in minutes on a laptop
with no GPU. The GPU preset is the Milestone 2 step up — a ~10-30M parameter
model sized to train on a free Colab/Kaggle GPU (e.g. a T4). Pick one with
`--preset {cpu,gpu}` in train.py; individual fields can still be overridden.
"""

from dataclasses import dataclass


@dataclass
class TrainConfig:
    # --- model size ---
    block_size: int = 128      # context length in tokens
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1

    # --- optimization ---
    batch_size: int = 32
    learning_rate: float = 3e-4
    max_iters: int = 2000
    eval_interval: int = 200   # how often to print train/val loss
    eval_iters: int = 20       # batches averaged when estimating loss
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_iters: int = 100    # linear LR warmup (helps the bigger GPU model)

    # --- tokenizer ---
    tokenizer: str = "char"    # "char" or "bpe"
    vocab_size: int = 1024     # BPE target vocab (ignored for char)
    bpe_train_chars: int = 500_000

    # --- data / io ---
    val_split: float = 0.1     # fraction of the corpus held out for validation
    seed: int = 1337
    out_dir: str = "checkpoints"


def cpu_preset():
    """Small model, trains on CPU in minutes. Char-level by default."""
    return TrainConfig()


def gpu_preset():
    """~15M-param model + BPE, for a free cloud GPU. Milestone 2 target."""
    return TrainConfig(
        block_size=256,
        n_layer=6,
        n_head=6,
        n_embd=384,
        dropout=0.2,
        batch_size=64,
        learning_rate=5e-4,
        max_iters=5000,
        eval_interval=500,
        eval_iters=50,
        warmup_iters=200,
        tokenizer="bpe",
        vocab_size=1024,          # pure-Python BPE train is ~1 min at this size
        bpe_train_chars=300_000,  # slice BPE learns merges from (one-time, cached)
    )


def gpu_large_preset():
    """~33M-param model + a larger BPE vocab, for the Milestone-3 corpus
    (RFCs + NIST). Still fits a free T4 with mixed precision. Bigger context,
    deeper, wider than `gpu`, and trained longer."""
    return TrainConfig(
        block_size=512,
        n_layer=10,
        n_head=8,
        n_embd=512,
        dropout=0.2,
        batch_size=24,          # sized to fit a 16 GB T4 at block 512
        learning_rate=5e-4,
        max_iters=6000,
        eval_interval=500,
        eval_iters=50,
        warmup_iters=300,
        tokenizer="bpe",
        vocab_size=2048,        # larger vocab for a bigger, more varied corpus
        bpe_train_chars=600_000,
    )


PRESETS = {"cpu": cpu_preset, "gpu": gpu_preset, "gpu_large": gpu_large_preset}
