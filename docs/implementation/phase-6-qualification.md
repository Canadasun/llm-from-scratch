# Phase 6: Qualify the exact artifact

Status: qualified as a research preview on 2026-11-10. The serving image meets the 1 vCPU / 2 GiB limits. The
production gate blocks it, as expected, and the preview targets are not met (Phases 4 and 5), so this is a
labelled research preview, not a production release.

## Artifact

| Item | Value |
|---|---|
| Image | `ghcr.io/canadasun/llm-from-scratch@sha256:98a40780c96d5b4608d071145a174c0e39ce315a14aae82412c0e13c3ae91b15` |
| Checkpoint | Phase 4 base, step 8,000, `1164213e635d376946ec331d5ff238cf89f492788554a953010f59bd5f57a593` (release `phase4-base-2026-11-10`) |
| Tokenizer | `bpe_code`, vocab 16,384, `a3fd8687…94a5` |
| Data manifest | `llmfs-corpus-v3`, `57af602d753c003ae446721d9ad25144c56a432e3b5daf52925a9b6ed9fafcfd` |
| Serving | int8 dynamic quantization, CPU |
| Built by | `release-qualify.yml` run 42940085410 (checkpoint downloaded from the release and checked by sha256) |

## Serving on 1 vCPU and 2 GiB

`deploy/measure.py`, read-only root, no capabilities, `--cpus=1 --memory=2g`. The served `/healthz` reported the
pinned checkpoint hash. Evidence: `docs/evaluations/phase6-serving-2026-11-10.json`.

| Metric | Measured | Limit |
|---|---|---|
| Generation speed | 91.9 tokens/s | at least 5 |
| Peak memory (cgroup `memory.peak`) | 1.07 GiB | under 1.8 GiB |
| Container start to healthy | 3.5 s | under 90 s |
| Image pull from GHCR (runner) | 22.3 s | (reported) |
| Image size | 1.35 GB | (reported) |

## Production gate

`deploy.release qualify` ran the existing evaluator inside the image with no network. Result:
`RELEASE BLOCKED: frozen generation tasks still use heuristic scoring`. The record is kept in
`docs/evaluations/phase6-release-record-2026-11-10.json`. The 18 code tasks were not measured because the code
sandbox cannot start inside the qualification container; they fail closed.

| Existing suite (int8 image) | M3 | Phase 4 base |
|---|---|---|
| Security multiple choice | 18/45 | 18/45 |
| Secure development | 1/12 | **4/12** |
| AI security | | 16/22 |
| Cloud | | 11/20 |
| Software | | 12/23 |
| Reasoning | | 2/14 |
| Generation | | 2/16 |
| Overall measured (excluding 18 code tasks) | | 65/152 (42.8%) |

## Release decision (preview gate)

| Target | Required | Phase 4 base |
|---|---|---|
| Parses as valid Python | at least 80% | 63.4% |
| HumanEval first-sample pass | at least 10% | 1.3% |
| Secure development | at least 6/12 | 4/12 |
| Security multiple choice | at least 60% | 40% |
| Unsafe recommendation | no regression from M3 | 4/15 (M3 2/15) |

The preview gate is not met. Deployment, if approved, is a research demo: responses carry `research_preview: true`
and the deterministic scanner stays in charge of findings.

## Independent re-check (Claude Code)

The private holdout (20 code tasks, 15 behaviour items; manifest `a0936335…`, never committed) was scored locally
for the base: 14/20 code completions parse (70%), behaviour 4/15 (prompt injection 0/5, unsafe recommendation 1/5,
abstention 3/5). Holdout code was not executed: there is no Linux sandbox on the scoring machine, and the holdout is
never sent to CI. The holdout parse rate agrees with the public 63.4%, so the public result is not memorised.
