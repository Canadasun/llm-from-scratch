# Phase 1: Training engine, measured on Kaggle

Status: exit criterion met on 2026-11-09. A real GPU resume passed across two Kaggle sessions, and throughput for
the 65M and 117M configurations was measured on the hardware Phase 4 will use.

## How the Kaggle runs work

- `training/kaggle/driver.py` publishes a pinned code archive (`llmfs-code`, commit `5059aa26`, archive sha256
  `ed77d3c4…`) and pushes a private script kernel with internet disabled. No GitHub or cloud credential is sent to
  Kaggle.
- `training/kaggle/session.py` checks the archive hash, the data manifest hash and, when resuming, the pinned
  checkpoint hash before running anything.
- `driver fetch` downloads the output and refuses any `ckpt.pt` or `best.pt` whose sha256 differs from the value in
  `status.json`.

## Hardware

Tesla T4 x2 (one GPU used), 14.6 GB, compute capability 7.5, torch 2.11.0+cu130, Python 3.13. The T4 reports bf16
but has no native bf16, so training uses fp16 autocast with a gradient scaler. Attention uses SDPA.

## Throughput (20 steps after 3 warmup, block 1024)

| Preset | Parameters | Batch | Tokens per second | Peak memory |
|---|---|---|---|---|
| `gpu_base` | 65.0M | 24 | 31,997 | 9.56 GB |
| `gpu_base` | 65.0M | 16 | 31,089 | 6.67 GB |
| `gpu_xl` | 117.3M | 24 | 16,165 | 13.54 GB |
| `gpu_xl` | 117.3M | 16 | 14,856 | 9.55 GB |

The benchmark presets use the smoke vocabulary. The Phase 2 tokenizer has 16,384 tokens, which adds output-layer
work, so Phase 4 should expect slightly lower figures.

## Resume across sessions

| Session | Kernel | Start step | End step | Stop reason | Checkpoint sha256 |
|---|---|---|---|---|---|
| A2 | `llmfs-phase1-session-a2` v2 | 0 | 483 | deadline | `5344c927…5cc7` |
| B | `llmfs-phase1-session-b` v1 | 483 (resumed) | 941 | deadline | `1848312…255d` |

- Both sessions stopped at their own deadline (600 s of training plus a reserve) and wrote a final checkpoint.
- Session B loaded the checkpoint from the private dataset `llmfs-phase1-state-a` only after its sha256 matched the
  pinned value. `status.json` records `resumed_from_step: 483`. The learning-rate schedule continued from step 483,
  not from warmup.
- Training ran at about 27,000 tokens per second (batch 8, accumulation 4, 32,768 tokens per step).
- The smoke shards are tiny, so the model memorises them: train loss 0.29 against validation loss 4.53 at step 500.
  That shows the loop, resume and best-validation checkpoint work. It says nothing about model quality.

## Sizing Phase 4

corpus-v3 has 631.2M training tokens.

| Preset | One epoch on one T4 | Kaggle hours per epoch |
|---|---|---|
| `gpu_base` (65M) | about 5.5 hours | about 6 |
| `gpu_xl` (117M) | about 11 hours | about 12 |

Kaggle gives 30 GPU hours per week, so one week covers about five epochs of `gpu_base`, or about 2.5 of `gpu_xl`.
Recommendation: train `gpu_base` for about 2 epochs (about 1.3B tokens, roughly 20 tokens per parameter) in
deadline-limited sessions chained by resume. Move to `gpu_xl` only if `gpu_base` misses the preview targets.
Using the second T4 with data-parallel training could nearly double throughput. It is not built yet.

## Operational note

Kaggle left session A2 queued for about nine hours. A second, stuck session (`llmfs-phase1-session-a` v3) was
also queued. After the stuck session was stopped in the web UI and A2 was pushed again, A2 started within two
minutes. If a kernel stays queued for more than an hour, check for other queued sessions under the account.
