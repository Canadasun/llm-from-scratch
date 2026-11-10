# Phase 4: Base training on Kaggle

Status: stopped on 2026-11-10 by the roadmap stop rule (validation loss flat for 15% of the budget). The accepted
base is the pilot `best.pt` at step 8,000.

## Setup

- Code `llmfs-code` at `783f3daa` (archive sha256 `c30e0cfd…6573`), data `llmfs-corpus-v3` (manifest
  `57af602d…`, about 2 GB of decontaminated Python, `bpe_code` tokenizer, vocab 16,384).
- `gpu_base`: 12 layers, 10 heads, width 640, block 1,024, about 65M parameters.
- Batch 16 x accumulation 4 = 65,536 tokens per step. Planned 38,000 steps (about 2.49B tokens, about 4 epochs),
  cosine decay over 38,000, peak learning rate 6e-4. fp16 autocast with a gradient scaler on one Tesla T4.
- Each session resumes from the previous `ckpt.pt`, passed as a private Kaggle dataset and pinned by sha256.

## Sessions

| Session | Kernel | Steps | Hours | Result | `ckpt.pt` sha256 |
|---|---|---|---|---|---|
| Smoke | `llmfs-p4-s1` | 0 to 495 | 0.3 | loss falling, resume state published | `75493c85…` |
| Pilot | `llmfs-p4-s2` | 495 to 9,176 | 7.0 | best val 1.566 at step 8,000 | `f3685f83…` |
| Main 1 | `llmfs-p4-s3` | 9,176 to 19,097 | 7.0 | no new best; stop rule met | `748f82f9…` |

Throughput about 23,000 tokens per second, peak memory 8.5 GB. `driver fetch` verified every checkpoint against
`status.json`.

Accepted base: pilot `best.pt`, step 8,000, sha256 `1164213e635d376946ec331d5ff238cf89f492788554a953010f59bd5f57a593`.

## Validation loss

| Step | Validation | Training (eval) |
|---|---|---|
| 500 | 4.157 | 4.169 |
| 3,000 | 1.749 | 1.737 |
| 6,000 | 1.600 | 1.488 |
| 8,000 | **1.566** | 1.397 |
| 11,000 | 1.573 | 1.301 |
| 14,000 | 1.569 | 1.234 |
| 17,000 | 1.576 | 1.173 |
| 19,000 | 1.585 | 1.132 |

The best validation loss came at step 8,000, about 0.85 epochs. Over the next 11,097 steps (29% of the budget)
training loss fell from 1.40 to 1.13 while validation loss did not improve. The model is memorising the corpus
rather than generalising. More passes over the same 2 GB will not help; more distinct data is the next lever.

## Frozen benchmarks (pilot `best.pt`)

Programs: `evals/results/phase4/programs-pilot.json`, greedy decoding, 384 new tokens. Scored in the CI sandbox by
`bench-score.yml` (run 42802344546) against bench manifest `0a97d6da…`.

| Metric | M3 | Pilot | Preview target |
|---|---|---|---|
| Parses as valid Python | 9.1% | **63.4%** | at least 80% |
| HumanEval first-sample pass | 0/157 | **2/157** (1.3%) | at least 10% |
| HumanEval+ | 0/157 | **3/157** | (reported) |
| MBPP / MBPP+ | 0/377, 0/372 | 0/377, 0/372 | (reported) |
| Unsafe recommendation (safe choice) | 2/15 | 4/15 | no regression from M3 |
| Prompt injection (safe choice) | 7/15 | 5/15 | (reported) |
| Abstention (safe choice) | 4/15 | 5/15 | (reported) |

HumanEval prompts end in a signature and docstring, and 82% of completions parse. MBPP prompts are plain English
with no signature, and only 55% parse: the base model drifts into repeated imports, `>>>` transcripts or filler.
That is an instruction-following gap for Phase 5, not a pretraining data fault.

The base misses the preview targets. Per the roadmap it goes to Phase 5 as the base, and the result is recorded.
