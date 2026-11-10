# Phase 3: Freeze evaluation before training

Status: suites frozen on 2026-11-09, before any Phase 4 training run. Targets below are fixed and must not move
after training starts.

## Benchmarks (evaluation only)

| Suite | Source (pinned revision) | Licence | Tasks scored | Cases |
|---|---|---|---|---|
| HumanEval | `openai/openai_humaneval` @ `7dce6050` | MIT | 157 of 164 | 1,103 |
| HumanEval+ | `evalplus/humanevalplus` @ `d32357cf` | Apache-2.0 | 157 of 164 | 121,003 |
| MBPP | `evalplus/mbppplus` @ `b2d74c91` (original three asserts, corrected) | Apache-2.0 (MBPP CC BY 4.0) | 377 of 378 | 1,169 |
| MBPP+ | `evalplus/mbppplus` @ `b2d74c91` | Apache-2.0 | 372 of 378 | 39,972 |

- `evals/bench.py` reads each source's tests with `ast` and `ast.literal_eval` (plus folding of constant
  arithmetic such as `3 * 19`). Nothing from a benchmark is executed to build the suites.
- Tasks whose tests need randomness, loops, imports or a reference function are **excluded and listed with the
  reason** in `evals/data/bench-manifest.json`, rather than scored with a weaker check.
- HumanEval+ and MBPP+ tasks with more than 1,000 inputs keep the first 1,000 (58 tasks); the manifest lists them.
- `evals/data/bench-manifest.json` pins every source sha256 and every built suite sha256. `python -m evals.bench
  verify` refuses suites that differ.

## Scoring

- Programs run only in the existing bubblewrap + seccomp sandbox. New batch mode: up to 100 calls per sandbox
  launch, typed values (tuple, set, frozenset, dict with non-string keys, complex, inf/nan), float tolerance
  matching EvalPlus (1e-6 default, or the task's own), and `set`/`sequence` comparison where the original test
  compared that way. Booleans never equal integers. Expected values stay in the controller.
- The single-call path used by the production suite is unchanged.
- `evals/bench_run.py generate` produces first greedy samples (no sandbox needed); `score` runs them in the
  sandbox (no model needed). This lets generation run on a GPU or this Mac and scoring run on a Linux CI runner.
- "Parses as Python" is `ast.parse` on the extracted program.

## Behaviour suite

`evals/data/preview/behaviour_mc.jsonl`: 45 project-authored, likelihood-scored items (exact, no keyword
heuristics) in three categories of 15: prompt injection, unsafe recommendation and abstention. None overlaps the
instruction-tuning data (13-gram containment, 10% threshold).

## Decontamination

`data/decontaminate.py` flags any training row containing at least 25% of a benchmark task's 13-word n-grams
(prompt plus reference solution). Containment, unlike Jaccard, catches a copied function inside a large file.

- corpus-v2: 21 rows from 7 repositories (one is literally an evals-data repository; the rest are
  LeetCode/Codewars/challenge solutions) matching 20 benchmark tasks. **All rows of those 7 repositories
  (165) were removed**, giving corpus-v3: shard manifest `57af602d753c003ae446721d9ad25144c56a432e3b5daf52925a9b6ed9fafcfd`,
  631,218,245 train tokens. Shards were cut without re-tokenizing; 32 sampled documents decode exactly to source.
- Rescan of corpus-v3 at 25%: 0 rows. Instruction set rebuilt from corpus-v3 (sft-v2): 0 rows.
- Kaggle private datasets: `llmfs-corpus-v3`, `llmfs-sft-v2`.

## Preview baseline

`evals/baselines/preview.json` is a plainly named research-preview baseline. `production.json` is untouched and
its hash is frozen inside `preview.json`.

| Metric | Preview target |
|---|---|
| Repo code tasks | at least 10/18 |
| HumanEval first-sample pass | at least 10% |
| Secure development | at least 6/12 |
| Security multiple choice | at least 60% |
| Output that parses as valid Python | at least 80% |
| Unsafe recommendation rate | no regression from M3, reviewed by hand |

Measured before Phase 4 (benchmarks scored in the CI sandbox from the committed programs; the existing-suite
figures come from the 2026-10-31 and 2026-11-04 evaluator runs):

| Metric | M3 (`6115d7a5`) | Instruction candidate (`50664e4e`) | Target |
|---|---|---|---|
| Repo code tasks | 0/18 | 0/18 | at least 10/18 |
| HumanEval first-sample pass | 0/157 | 0/157 | at least 10% |
| HumanEval+ | 0/157 | 0/157 | (reported) |
| MBPP / MBPP+ | 0/377, 0/372 | 0/377, 0/372 | (reported) |
| Secure development | 1/12 | 2/12 | at least 6/12 |
| Security multiple choice | 18/45 (40%) | 19/45 (42%) | at least 60% |
| Parses as valid Python | 9.1% | 2.4% | at least 80% |
| Unsafe recommendation (safe choice) | 2/15 | 2/15 | no regression from M3 |
| Prompt injection (safe choice) | 7/15 | 7/15 | (reported) |
| Abstention (safe choice) | 4/15 | 6/15 | (reported) |

Three-way choices, so chance is 5/15. Both models pick the unsafe recommendation more often than chance.

CI job `preview-baseline` rebuilds the suites from the pinned sources, rescores the committed programs in the
sandbox and fails if any number or frozen hash differs.

## Private holdout

Claude Code wrote a private holdout of 20 original coding tasks (97 cases) and 15 behaviour items. It is stored
outside the repository and is never committed or shared with Codex. Only its manifest hash is recorded here:
`a09363350b6653ef1592b06e09fe01cdb4dd4e60399c3f8e79827b88aba02273`.

## Reference-solution check

`.github/workflows/bench-canonical.yml` (manual, owner-approved) runs every official reference solution against its
converted suite inside the sandbox on an ephemeral GitHub-hosted runner, with read-only permissions, no persisted
credentials and a 120-minute job limit. It uploads a per-suite report of passes, wrong outputs, exceptions, timeouts,
worker failures and sandbox errors.

- First run (2026-11-10): 1,031 of 1,063 passed. All 32 failures were converter or checker bugs: dict key order
  lost when writing suites, HumanEval+ checked against the original HumanEval solutions instead of its corrected
  ones, `assert f(x)` treated as `== True` instead of truthiness, `True == 1` rejected although the official checks
  use Python `==`, and 100-call batches sharing one CPU budget. No expected value or input was changed.
- Second run after the fixes: 1,063 of 1,063 passed (157, 157, 377, 372), with no timeouts or sandbox errors.
- Suite hashes were refrozen. Task and case counts did not change, and the preview programs regenerated
  identically. `production.json` is untouched.
