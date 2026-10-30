"""
Training configuration for Milestone 1 (CPU-friendly defaults).

These numbers are deliberately small so the model trains in minutes on a laptop
CPU with no GPU. They are the knobs you will grow in Milestone 2 once you move
to a cloud GPU: bigger block_size, n_layer, n_head, n_embd and more iterations.
"""

from dataclasses import dataclass


@dataclass
class TrainConfig:
    # --- model size (kept small for CPU) ---
    block_size: int = 128      # context length in tokens
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1

    # --- optimization ---
    batch_size: int = 32
    learning_rate: float = 3e-4
    max_iters: int = 3000
    eval_interval: int = 250   # how often to print train/val loss
    eval_iters: int = 50       # batches averaged when estimating loss
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # --- data / io ---
    val_split: float = 0.1     # fraction of the corpus held out for validation
    seed: int = 1337
    out_dir: str = "checkpoints"
