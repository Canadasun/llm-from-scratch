# Evaluation suite

Training loss tells you the model is *learning its training data*. It does **not**
tell you the model is good, safe, or ready. This suite exists to replace "the
samples look impressive" with numbers you can track and defend.

It is deliberately small to start (a hand-authored starter set), and designed to
grow toward the 100–200 validated tasks the project's enterprise direction calls
for.

## Run it

```bash
python -m evals.run                      # all evals on checkpoints/ckpt.pt
python -m evals.run --suite intrinsic    # just perplexity vs a random baseline
python -m evals.run --suite tasks        # just the capability tasks
python -m evals.run --limit 20           # quick smoke run
python -m evals.run --json results.json  # also save machine-readable results
python -m evals.run --ckpt path/to/ckpt.pt --device cuda
```

The report **never prints "pass"**. It reports; release gating is a separate,
explicit human decision (see below).

## What it measures

### 1. Intrinsic quality (works on any base model)
On a **held-out** text (`evals/data/heldout_security.txt`, independently authored
and kept out of the training corpus so this is generalisation, not memorisation):

| metric | meaning |
|---|---|
| **perplexity** | how surprised the model is per token (lower = better) |
| **bits-per-byte** | same idea, but tokenizer-independent — the fair cross-model number |
| **next-token top-1 / top-5** | how often the true next token is the model's #1 / top-5 guess |

Each is reported for the **trained** model **and** for a fresh **random-init**
model of the same shape. The gap between them is the proof that *training* — not
the architecture or tokenizer — produced the skill.

### 2. Capability tasks
- **Multiple-choice** (`*_mc.jsonl`): scored by **likelihood** — the model doesn't
  have to *write* the answer, it just has to find the correct choice least
  surprising. This is how base (non-chat) models are evaluated, and it gives real
  signal from a small model. Compared against the **random-guess baseline**.
- **Generation** (`generation.jsonl`): the model generates a continuation, matched
  by exact / substring / regex.
- **Code** (`code_python.jsonl`): the model's output is written to `solution.py`
  and run against unit tests **in a sandboxed subprocess** — throwaway working
  directory, a minimal environment (no parent-process secrets), CPU/memory/
  file-size rlimits, and a wall-clock timeout. Generated code is untrusted input
  and is never imported or exec'd in-process.

## Honesty rules baked in
- Sample sizes are always shown (`acc (correct/n)`).
- The MC random-chance baseline is printed, so near-chance reads as near-chance.
- A few concrete failures are listed every run.
- Limitations are printed every run: this is a ~10M-param **base** model, not
  instruction-tuned; a near-zero code pass rate is the expected baseline until the
  model is scaled or tool-assisted.

## First baseline — 2026-10-30 (ckpt: ~10.7M params, BPE vocab 1024, 5000 iters)

Held-out intrinsic (trained vs a random-init twin of the same shape):

| metric | trained | random-init |
|---|---|---|
| perplexity | **15.60** | 1,272.65 |
| bits/byte | **2.15** | 5.59 |
| next-token top-1 | **44.7 %** | 0.1 % |
| next-token top-5 | **69.1 %** | 0.8 % |

Capability tasks (overall **23.0 %**, 14/61; MC random-chance ≈ 25 %):
security 40.0 % (8/20) · software 26.7 % (4/15) · reasoning 12.5 % (1/8) ·
generation 10.0 % (1/10) · code 0.0 % (0/8).

**Reading this honestly:** the intrinsic numbers are a decisive win — on text it
has never seen, the model predicts the next token correctly 44.7 % of the time
versus 0.1 % untrained, and its perplexity is ~80× better than random. It
genuinely learned to model the domain's language. But the capability tasks sit
**at or near random chance**: a 10M-param model trained on ~6 MB of RFCs has
*not* acquired transferable security/software knowledge, reasoning, or coding
ability. This is exactly the expected baseline, and it is the whole point of the
suite — the fluent RFC-looking samples could fool you; these numbers don't. This
is the line Milestone 3 (more and broader data, a bigger model, longer training,
later RAG + tools) must move.

## Adding tasks
Append JSON lines to the files in `evals/data/`. Schemas:

```jsonc
// multiple choice (answer is the index of the correct choice)
{"id":"...","category":"security","type":"multiple_choice",
 "prompt":"Q: ...\nA: ","choices":["...","..."],"answer":0}

// generation (scorer: contains | exact | regex | contains_any)
{"id":"...","category":"...","type":"generation",
 "prompt":"...","scorer":"contains","target":"...","max_new_tokens":16}

// code (test is a unittest module that imports `solution`)
{"id":"...","category":"code","type":"code",
 "prompt":"...\n```python\n","test":"import unittest\nfrom solution import ...\n..."}
```

`tests/test_evals.py` validates every task file on each test run (parse, unique
ids, in-range answers), so a malformed task is caught immediately.

## Regression gate & CI

Two layers run in CI (`.github/workflows/`):

**1. Always-on (`ci.yml`)** — every push/PR runs the full test suite, which
exercises the eval harness itself: task-file validation, likelihood/scoring
maths, the sandboxed code runner, and the regression-comparison logic. No GPU,
network, or checkpoint needed, so it is fast and reliable. If the eval machinery
breaks, the build goes red.

**2. Model regression (`eval-regression.yml`)** — runs the suite on a trained
checkpoint and **fails if it regresses** below a recorded baseline
(`evals/baselines/m2.json`). Run it locally too:

```bash
# run the suite on a checkpoint and gate it against the baseline
python -m evals.regression --ckpt checkpoints/ckpt.pt --baseline evals/baselines/m2.json
# or compare an already-produced results file
python -m evals.run --json out.json
python -m evals.regression --results out.json --baseline evals/baselines/m2.json
```

Exit code is **1 on regression**, 0 otherwise. This does *not* assert the model
is good — it only catches a change that quietly makes it **worse** (a tokenizer
bug, a broken checkpoint, a scoring change).

**What is gated, and why only some metrics.** A baseline's `enforce` list names
the metrics that actually fail the build. The M2 baseline enforces only
**bits-per-byte** and **overall task accuracy**, because those are
**tokenizer-independent** and so compare fairly across models with different
vocabularies (M2 vocab 1024 vs M3 vocab 2048). **Perplexity** and **next-token
top-k** are per-token and therefore tokenizer-dependent — they are recorded for
reference but not gated across a vocab change. Default tolerances: bits/byte may
worsen by ≤ 0.15 absolute; task accuracy may drop by ≤ 0.05 absolute (overridable
per baseline).

When a new milestone's numbers are measured and accepted, add a new baseline file
(e.g. `evals/baselines/m3.json`) and point the workflow at it.

## Release gating (separate from this report)
Per the project's enterprise direction, production promotion should be **blocked**
on things this suite is a building block for — tenant-isolation defects, secret
leakage, benchmark contamination, critical-capability regressions, unauthorized
high-impact tool execution. Those are explicit human decisions informed by
evidence, not an accuracy threshold printed by a script. No claims of 100 %
accuracy or guaranteed prompt-injection immunity.
