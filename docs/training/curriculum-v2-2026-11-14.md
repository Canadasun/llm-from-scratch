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
split assignments. Totals are 48/15/15 labelled problem families, not 390
independent tasks. Related utilities may share algorithms, so a distinct family ID
does not prove statistical or semantic independence. There are 52 Python-focused and 26 security-focused labels;
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
pinned code/data/replay. Require exact CUDA resume verification first on the small
dropout/replay test model with deterministic algorithms enabled; this tests the
resume mechanism rather than every accelerator/model configuration. Run at most
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

## Measured outcome: rejected, base retained

The [private free-GPU run](https://www.kaggle.com/code/idowuaye/llmfs-curriculum-pilot-20261114)
completed the resume gate, 100 training steps and all four generation stages.
No train or validation rows were dropped. The best aggregate instruction validation
loss was at step 100: 1.087790 to 0.633408; replay validation loss rose from
1.657006 to 1.672926. These mixed instruction losses include repair/preservation
prompts containing code, so they must not be interpreted as novel-code-generation
scores or compared directly with the seed's implementation-only validation loss.

The exact newly generated base programs, complete behavior vectors and relevant
inference/benchmark metadata matched the previously scored base outputs. Those
base sandbox results were reused rather than re-executed. The evidence records
their original run IDs and content hashes. The new candidate outputs were scored
in fresh Linux sandbox runs with a passing canary and no unavailable cases.

| Metric | Base completion | Candidate completion | Base chat | Candidate chat |
|---|---|---|---|---|
| Valid Python, distinct programs | 35/52 | 31/52 | 10/52 | 17/52 |
| HumanEval | 1/25 | 1/25 | 0/25 | 0/25 |
| HumanEval+ | 1/25 | 1/25 | 0/25 | 0/25 |
| MBPP | 0/25 | 0/25 | 0/25 | 0/25 |
| MBPP+ | 0/25 | 0/25 | 0/25 | 0/25 |
| Abstention, safe choice | 5/15 | 5/15 | 5/15 | 5/15 |
| Prompt injection, safe choice | 5/15 | 5/15 | 5/15 | 5/15 |
| Unsafe recommendation, safe choice | 4/15 | 4/15 | 4/15 | 4/15 |

The paired comparisons show no new or lost functional passes. The primary
completion screen fails both improvement and syntax-retention requirements;
chat retention passes, but increased syntax validity does not establish useful
coding. No full qualification, further curriculum steps or rollout follows.
The accepted Phase 4 base remains unchanged. This observed public subset does
not establish a model ranking or broad cybersecurity competence.

`training/pilot_decision.py` applies the predeclared checks and rejects incompatible
checkpoint/suite/task/decoding identities, infrastructure errors and inconsistent
score totals. Thirteen decision-controller tests pass. A separate check of the
same controller rejects the previous failed seed artifacts. The data/training code
passed [Linux CI](https://github.com/Canadasun/llm-from-scratch/actions/runs/43632855908)
with 580 passed and one skipped; final branch checks are visible on the PR.

Evidence:

- [Pinned configuration](curriculum-v2-kaggle-2026-11-14.json)
- [GPU status, history, identity and baseline-reuse evidence](curriculum-v2-gpu-2026-11-14.json)
- [Matched-mode decision](curriculum-v2-decision-2026-11-14.json)
- [Aggregate output diagnostics](curriculum-v2-failures-2026-11-14.json)
- Raw outputs and scores: `evals/results/curriculum_v2/`
- [Candidate completion sandbox run](https://github.com/Canadasun/llm-from-scratch/actions/runs/43634660362)
- [Candidate chat sandbox run](https://github.com/Canadasun/llm-from-scratch/actions/runs/43634662106)

The two controlled authored-data trials have not produced functional coding gains.
Pause this sequence of small-curriculum SFT trials rather than launch another
longer run. The next research step is to investigate base pretraining coverage,
generalization and format transfer, then choose a materially different, measured
experiment. This result alone does not prove whether capacity, pretraining data,
instruction diversity or their interaction is the dominant limitation.
