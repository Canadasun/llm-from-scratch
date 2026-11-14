# Verified instruction seed and bounded pilot

The next experiment replaces mixed-format mined instructions with a small,
project-authored MIT code-only seed. This is a test of the training approach,
not enough data for broad capability claims or a replacement for the accepted base.

## Dataset

`data/verified_tasks.py` contains 30 distinct development utilities with explicit
specifications, 143 fixed input/output or exception cases, and one targeted buggy
implementation per task. The builder executes only this first-party authored
code in isolated Python subprocesses with five-second timeouts. It never executes
downloaded code or model outputs. All reference cases pass; all 30 mutants fail
at least one case. These tests are authored with the solutions and do not constitute
independent human review or a proof of correctness.

There are 18 training, six validation and six test problem families, with no shared
family or exact solution across splits. Three instruction phrasings per family
produce 54/18/18 rows, not 90 independent problems. Every response is plain Python,
with no summary/review/refusal mixing or upsampling. All tasks share first-party
repository provenance; this is a problem-family split, not a repository holdout.
The accepted tokenizer encodes every train/validation row in at most 193 tokens.

The builder checks public HumanEval/MBPP reference containment (13-word n-grams,
50% threshold, eligible references only), plus exact repository evaluation snippets
including nested preview suites. It fails on a detected match. This does not detect
all paraphrase/semantic leakage. Private independent holdouts are never read.
The manifest binds source, builder, split contents, tests and benchmark references.

```bash
python -m data.build_verified_sft \
  --out ~/llmfs-data/sft-verified-seed-2026-11-14 \
  --refs ~/llmfs-data/bench/reference_texts.jsonl
```

Builds require an empty output directory and are reproducible. The evidence is
in [the committed manifest](verified-seed-2026-11-14.json).

## Checks and GPU gate

75 focused tests passed across authored data, contamination and training/resume.
They include planted benchmark leakage, split-family duplication, wrong references,
surviving defects, input mutation and missing/empty contamination references.
The existing full-suite baseline failures are described in the recovery report.

`python -m training.verify_sft_resume --device cuda --report resume.json` requires
a real GPU. It compares six continuous steps with three steps plus resume, using
dropout 0.1 and replay. Every model tensor and RNG state must match exactly before
the candidate training step is allowed. The CPU version passes; that is recorded
separately and does not qualify GPU resume.

## Pilot budget and acceptance

Use the pinned Phase 4 base (`1164213e…f593`), 30 optimization steps, learning rate
1e-5, warmup five, batch four with two accumulation steps, 512-token context,
30% replay, no upsampling, validation every ten steps. Candidate output is separate
from the base. Use the existing free Kaggle T4 worker only; no paid infrastructure.

Generate the first 25 tasks per frozen code suite, with unchanged greedy decoding
and 384-token limits. Compare both completion and chat modes separately between
base and candidate, plus the existing behavior categories. Generated code must
only be executed in the existing Linux sandbox. The pilot subset is not the full
qualification benchmark. Advance to full benchmarks only after a coding improvement,
no parse-rate decrease and no behavior-category regression in the matched-mode
comparison. Do not promote from training loss or parse rate alone. A failed pilot
keeps the accepted base and stops expansion of this seed experiment.
