# Phase 0 preparation — 2026-11-05

Status: incomplete; owner confirmation pending. No Phase 0 exit is claimed.

The owner-supplied `~/Desktop/llm-from-scratch-ROADMAP.md` requires confirmation
of the C$10 monthly cap and owner-only access before Phase 0 Azure changes.
Confirmation was requested; no Azure resource was read or modified during this
preparation. No paid resource was provisioned, and no model was deployed.

## Repository and artifacts

- Cloned `Canadasun/llm-from-scratch` into `~/llm-from-scratch`.
- Starting commit: `dc53229e4b3f80a9cf89295087b72b0f2bc7a63d`.
- Working branch: `phase-0-lockdown-baseline`.
- Downloaded all assets from release `checkpoints-2026-11-04` into the ignored
  `checkpoints/release-2026-11-04/` directory.
- Ran `shasum -a 256 -c SHA256SUMS`: all eight entries passed, including M2,
  M3, instruction candidate, and instruction training-resume weights.
- Release checksum-manifest SHA-256 reported by GitHub:
  `39250aceaabf214bc548bdb5e29f1a052a63a487d83ba7da53edffdf55d74548`.
- Docker reports a Linux engine. A new Docker test run has not been performed.

## Prior baseline evidence

[CI run 41678135742](https://github.com/Canadasun/llm-from-scratch/actions/runs/41678135742)
at the starting commit reports **402 passed, 1 skipped, 5 warnings in 24.33s**,
on 2026-11-03, using CPU torch 2.15.0 and Python 3.13 on Ubuntu 22.04. Its log
includes the bubblewrap namespace preflight. This is existing CI evidence,
not a fresh test run by this executor. A fresh full Linux baseline remains due.

[Evaluation workflow 41981066883](https://github.com/Canadasun/llm-from-scratch/actions/runs/41981066883)
finished successfully on 2026-11-05 but explicitly skipped checkpoint evaluation
because repository checkpoint variables were unset. It is not a model gate pass.

## Remaining Phase 0 work

After owner confirmation, inspect only `llm-from-scratch-rg`, implement and
verify owner-IP ingress, cooldown, log cap, and the scoped budget stopper.
The roadmap's request to copy a stopper pattern does not authorize access to
`scribe11`; do not read or modify that resource group.

Replace stale deployment blocks with a checked budget policy, remove account
identifiers, correct README status, and record the candidate as a failed research
preview or perform a qualified rollback. A budget alert is not a guaranteed
billing cap because cost reporting and stop actions can lag.

Run a fresh full Linux test suite, verify refusal from an external non-allowed
IP, test stop/restart, and record budget threshold readbacks before declaring
Phase 0 complete. Keep one PR for this phase; merge only with green CI.
Do not advance to Phase 1 until all Phase 0 exit conditions are measured.

No training data, evaluation data, production baseline, or deployment scripts
were changed during preparation.
