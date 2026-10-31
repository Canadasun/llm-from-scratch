# llm-from-scratch

[![CI](https://github.com/Canadasun/llm-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/Canadasun/llm-from-scratch/actions/workflows/ci.yml)

Building a large language model **from scratch** — no model libraries, no
fine-tuning of someone else's weights, no retrieval wrapper. The tokenizer, the
self-attention, the transformer blocks and the training loop are all written by
hand so every line is understood.

The long-term focus is a model that helps with **software development and
cybersecurity**, trained on public and self-curated text in those domains.

## Why "from scratch" in stages

A frontier model needs thousands of GPUs. A *learning-grade* model that teaches
you how LLMs actually work needs only a laptop. This repo is built to grow:

| Milestone | What you build | Compute |
|---|---|---|
| **1 — Local GPT** (this code) | A small GPT (~0.5M params by default) with a char-level tokenizer, trained on CPU | Your own machine, no GPU |
| **2 — Real tokenizer + bigger model** | Byte-pair-encoding (BPE) tokenizer, 10–100M params, a curated dev+security corpus | Free Colab/Kaggle GPU |
| **3 — Scale** | More layers/params/data, proper eval and checkpointing | Rented cloud GPU |
| **4 — Serve** | An inference endpoint for your own model | Cloud, when needed |

You are at **Milestone 1**.

## Quickstart (CPU)

```bash
# 1. Install the CPU build of PyTorch (one time)
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2. Build the corpus (curated RFCs; or use your own text)
python data/build_corpus.py                 # ~6 MB dev+security corpus
# python data/prepare.py --input notes.txt  # alternatively, your own .txt

# 3. Train (minutes on a CPU with the small char-level defaults)
python train.py --preset cpu

# 4. Generate text from the trained model
python sample.py --prompt "Security " --tokens 400
```

## Train on a GPU (bigger model + BPE)

No local GPU needed — use a free Colab/Kaggle GPU. See [`docs/GPU.md`](docs/GPU.md)
and [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb).

```bash
python train.py --preset gpu                # ~15M params, BPE tokenizer, mixed precision
python sample.py --prompt "TLS 1.3 " --tokens 400
```

Tokenizer and corpus encoding are cached after the first run, so the one-time
BPE build is paid once.

## What each file is

| File | Role |
|---|---|
| `model.py` | The GPT itself: embeddings, causal self-attention, transformer blocks, LM head. Hand-written, heavily commented. |
| `tokenizer.py` | Character-level tokenizer (Milestone 1). BPE replaces this in Milestone 2. |
| `data/prepare.py` | Writes `data/corpus.txt` from the built-in sample or a `--input` file. |
| `config.py` | CPU-friendly training hyperparameters — the knobs you grow later. |
| `train.py` | Batching, AdamW, gradient clipping, train/val loss, checkpoint save. |
| `sample.py` | Autoregressive text generation from a checkpoint. |

## How it works, in one paragraph

Text is turned into a sequence of token ids. Each token looks up a learned
embedding vector, and a learned position vector is added so order matters.
Stacked transformer blocks let every token attend to the tokens *before* it
(the causal mask) and refine its representation. A final linear layer turns each
position's vector into a probability distribution over the vocabulary, and the
model is trained to make the true next token likely. Generating text is just
sampling from that distribution one token at a time and feeding it back in.

## Safety (built in from the start)

This is a security-focused model, so safety is part of the pipeline, not an
afterthought — and it is **dual-use aware**: security topics are in scope
(the users are professionals), while clear wrongdoing is not.

- **Training-data scrubbing** — `safety/scrub.py` redacts private keys, API
  tokens, JWTs, cloud keys and passwords before any text enters the corpus. It
  runs by default in `data/build_corpus.py`. A model can memorize its data, so
  secrets must never get in.
- **Use policy** — `safety/content_policy.py` classifies prompts/generations as
  allow / flag / block. It allows exploits, pentesting and attack theory, *flags*
  dual-use content aimed at a specific target with no authorization context, and
  *blocks* categories with no legitimate use. `sample.py` applies it (disable for
  research with `--no-safety`, which is logged).
- **Secure checkpoint loading** — `torch.load(..., weights_only=True)` so an
  untrusted `.pt` cannot execute code.

See [`SECURITY.md`](SECURITY.md), [`docs/RESPONSIBLE_USE.md`](docs/RESPONSIBLE_USE.md)
and [`MODEL_CARD.md`](MODEL_CARD.md).

## Tests

```bash
pip install pytest
python -m pytest tests/ -q
```

Covers the model (shapes, causal masking, learning), both tokenizers (round-trip,
compression, save/load), corpus cleaning, the safety layers, and the evaluation
harness (scoring maths, scorers, the code sandbox, and task-data validity).

## Evaluation

Training loss says the model is learning its data; it does not say the model is
good. The eval suite turns that into measured numbers:

```bash
python -m evals.run                 # intrinsic metrics + capability tasks
python -m evals.run --suite intrinsic   # held-out perplexity vs a random-init model
```

It reports held-out perplexity / bits-per-byte / next-token accuracy **against a
random-init baseline** (proof that training — not the architecture — produced the
skill), plus likelihood-scored multiple-choice, generation, and **sandboxed**
code tasks, with sample sizes, the random-chance baseline, example failures and
explicit limitations. See [`docs/EVALUATION.md`](docs/EVALUATION.md). It reports;
it never prints "pass" — release gating is a separate, explicit decision.

**Evals in CI.** Every push runs the full suite (including the eval harness) via
[`ci.yml`](.github/workflows/ci.yml). A separate
[`eval-regression.yml`](.github/workflows/eval-regression.yml) runs the suite on a
trained checkpoint and **fails on regression** against a recorded baseline
(`evals/baselines/`), gating the tokenizer-independent metrics (bits-per-byte,
task accuracy). Run it locally with `python -m evals.regression --ckpt … --baseline …`.

## Model card

[`MODEL_CARD.md`](MODEL_CARD.md) documents the model honestly: architecture and
sizes, intended and out-of-scope uses, training-data provenance and license
basis, the measured evaluation results (strong language modeling, near-chance
task capability at this scale), limitations, and the security posture.

## Data note

Training data comes from **public sources and the author's own notes** in
software development and cybersecurity. No private, customer, or personal data
is used for training.

## Roadmap

- [x] Milestone 1 — local from-scratch GPT, char-level, CPU
- [x] Milestone 2 — from-scratch BPE tokenizer wired into training, a curated
  license-safe dev+security corpus (~6 MB of IETF RFCs), and GPU support
  (`--preset gpu`, mixed precision) with a Colab notebook for a free cloud GPU
- [x] Milestone 3 (in progress) — evaluation harness (`evals/`): held-out
  perplexity vs a random baseline, likelihood-scored multiple-choice, generation
  and sandboxed code tasks. Next: scale params/data and grow the task suite.
- [x] Milestone 4 (in progress) — inference endpoint (`serve/`): a
  framework-agnostic, unit-tested service core (validation + per-client rate
  limiting + use-safety on prompt **and** output + secure checkpoint load) behind
  a dependency-free HTTP adapter. `python -m serve`; see
  [`docs/SERVING.md`](docs/SERVING.md). Next: auth, a production ASGI worker, and
  a shared rate-limit store.

### Enterprise platform (beyond the model)

The longer-term direction is to make this a *platform* pro developers and security
teams can trust, in staged, evidence-backed steps.

- [x] Stage 1 — prove the model learns (Milestones 2–3: trained + independently
  evaluated, honest numbers).
- [x] Stage 2 (started) — **secure code review** (`codereview/`): AST-based
  vulnerability detection with CWE-mapped findings, confidence levels, and
  **measured FP/FN** (every rule has vulnerable + safe ground-truth fixtures;
  zero false positives on first-party code). `python -m codereview <path>`; runs
  in CI against its own source, proposes safe fixes as human-review diffs
  (`--fix`, never auto-applied), and verifies them against the project's own tests
  on a copy in a bounded sandbox (`--verify-fixes`: findings resolved + no test
  regression). See [`docs/CODE_REVIEW.md`](docs/CODE_REVIEW.md). Next: assemble
  the evidence and open a PR for human review.
- [ ] Stage 3 — enterprise security (identities, repo permissions, tenant
  isolation, audited tool execution, controlled deploy).
- [ ] Stage 4 — authorized pilot measuring reliability, security, latency, cost.

Findings and fixes are always for **human review** — authorization lives in the
software, never in model output, and all analyzed input is untrusted.

## License

MIT — see [LICENSE](LICENSE).
