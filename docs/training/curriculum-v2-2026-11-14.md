# Broader Python/security curriculum experiment

This follows the failed 18-training-family seed experiment. The accepted Phase 4
base remains the starting point. No production floors or frozen benchmark tasks
are changed. No promotion follows from lower training loss alone.

## Failure diagnosis

[Aggregate diagnostics](failure-analysis-2026-11-14.json) were computed by parsing
existing program text and reading sandbox verdicts, never by executing generated
programs on the host. They contain no benchmark prompts, test inputs or answers.

Of 52 distinct outputs, base chat has 42 syntax failures; seven of its ten parseable
outputs are empty modules. Candidate chat has 43 syntax failures and four empty
modules. Completion mode has 35 parseable programs for each model, but 13 have no
top-level function. Sandboxed results also show output mismatches and recursion,
index, type and value errors. Base/plus verdicts reuse programs and are not 100
independent problems. Generic worker exits remain unclassified, not labelled logic
bugs without evidence. Python failures do not establish cybersecurity skill.

## Data

Add 48 new first-party MIT problem families in `data/curriculum_tasks.py`: 30 train,
nine validation and nine test. Preserve all 30 original seed families and their
split assignments. Totals are 48/15/15 independent problem families, not 390
independent tasks. There are 52 Python-focused and 26 security-focused labels;
security tasks are also Python programs.

Every family has three implementation phrasings, one repair input containing a
targeted defect, and one correct-code preservation input. All responses are complete
Python functions with no prose or Markdown. This gives 240/75/75 instruction rows.
The repair and preservation rows teach conditional changes, while retaining the
same code-only response format. No evaluation solution was used as an authoring
source. New tasks include authorization checks, input budgets, SQL/argv construction,
serialization, settings, Unicode, network literals, HMAC and parser edge cases.

All 401 authored reference cases pass, all 78 targeted mutants are caught, and
input-preservation checks pass. Tests and solutions were authored together: these
are executable checks, not independent human review or proof of security. Helpers
have explicit limited contracts; lexical path checks are not filesystem confinement,
and metadata checks are not complete upload/session/authentication systems.

The builder checks final repair/preservation prompts as well as implementations
against public references and nested repository evaluations. It also rejects
renamed equivalent code structures across splits, unsupported provenance and
duplicate families/solutions. All variants remain with their family. There is no
repository holdout because this dataset is wholly first-party; independent private
holdouts are never accessed. N-gram/exact checks do not establish semantic independence.

```bash
python -m data.build_verified_sft --curriculum expanded \
  --out ~/llmfs-data/sft-curriculum-v2-2026-11-14 \
  --refs ~/llmfs-data/bench/reference_texts.jsonl
```

[Manifest and per-family test evidence](curriculum-v2-manifest-2026-11-14.json).
The output directory must be empty. Both train and validation files pass secret-pattern
screening (the test file also passes). The longest train row is 523 tokens and the
longest validation row 471 with the accepted tokenizer. A 768-token training
context retains all rows; using 512 would silently drop two repair/preservation rows.

130 focused tests pass for data, contamination and training/resume. Linux CI is
required for the existing sandbox tests; the documented Mac baseline failures are
not grounds to disable the sandbox.

## Predeclared bounded trial

One free Kaggle T4 experiment, from pinned accepted base `1164213e…f593`, with
pinned code/data/replay. Require exact CUDA resume verification first. Run at most
100 optimization steps, context 768, batch four, accumulation two, learning rate
1e-5, ten warmup steps, 30% corpus replay, uniform row sampling, validation every
20 steps, and a ten-minute training deadline. Keep candidate artifacts separate.
Expected instruction exposure is approximately 2.3 row passes, rather than a long
memorization run. Choose the saved candidate only by instruction validation loss.

Use the unchanged first-25-task screen in each frozen suite with greedy decoding
and 384 generated tokens. Generate matched base/candidate completion and chat
outputs and score only in the existing Linux sandbox. This screen has already been
observed in prior experiments and is not an independent final qualification set.

The primary completion screen must improve at least one functional suite, decrease
none, retain overall syntax rate and retain each behavior category. Chat must not
regress on functional suites, syntax rate or behavior categories. No unavailable
executions are acceptable as evidence. Only a passing screen justifies full frozen
benchmark qualification; that is a separate measurement, not automatic promotion.
A failed screen retains the accepted base and ends this bounded trial. No paid
resources, hosting changes or relaxation of release gates are part of this work.
