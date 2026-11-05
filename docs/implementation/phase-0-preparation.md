# Phase 0 preparation — 2026-11-05

Status: implementation underway; not all Phase 0 exits have passed.

The owner-supplied `~/Desktop/llm-from-scratch-ROADMAP.md` requires confirmation
of the C$10 monthly cap and owner-only access before Phase 0 Azure changes.
The owner confirmed both on November 5. Subsequent Azure changes are scoped to
this project's resource group. No model was promoted.

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
- Docker reports a Linux engine. The fresh Docker baseline returned 383 passed,
  4 failed, 14 errors, 2 skipped, 5 warnings in 20.99 seconds. Its image lacked
  `/usr/bin/python3`, which the independent sandbox requires. This is a failed
  baseline attempt, not a passing test result. Use the established Ubuntu CI
  runner for the next full sandbox test rather than weakening isolation.

## Prior baseline evidence

[CI run 41678135742](https://github.com/Canadasun/llm-from-scratch/actions/runs/41678135742)
at the starting commit reports **402 passed, 1 skipped, 5 warnings in 24.33s**,
on 2026-11-03, using CPU torch 2.15.0 and Python 3.13 on Ubuntu 22.04. Its log
includes the bubblewrap namespace preflight. This is existing CI evidence,
not a fresh test run by this executor. A fresh full Linux baseline remains due.

[Evaluation workflow 41981066883](https://github.com/Canadasun/llm-from-scratch/actions/runs/41981066883)
finished successfully on 2026-11-05 but explicitly skipped checkpoint evaluation
because repository checkpoint variables were unset. It is not a model gate pass.

## Applied controls

- Azure ingress readback: exactly one Allow rule for the detected owner's IPv4
  address, with a /32 prefix. The address is not committed.
- Log Analytics daily quota readback: 0.1 GB, status `RespectQuota`.
- Monthly budget `llm-monthly-10cad`: C$10, current actual spend C$0.00 at setup.
  Notifications read back as C$5 actual alert, C$8 actual stop and C$10 forecast
  stop. Both stop thresholds reference the new action group.
- Consumption workflow `llm-budget-stopper` uses a system-assigned managed
  identity. A custom role grants only app read and stop; assignment is scoped to
  `llm-serve`, not its resource group or subscription.
- Stopper run `08584077235460169782292598068CU21` reached `Succeeded`;
  app `runningStatus` was observed as `Stopped`, then `Running` after restart.
  The initial test command failed because its workflow-success read raced Azure's
  run completion. Both states and successful run were verified separately, and
  polling now waits for the same new run to finish before restarting.
- Callback credentials are kept in memory/private temporary request files and
  are not printed or committed. Existing budget recipients are reused privately.
- The existing failed candidate stays an owner-only research preview. This is
  cheaper than a new model deployment, and is not a production promotion or
  running-digest attestation. Full artifact qualification remains due in Phase 6.

## Remaining Phase 0 work

Complete and measure cooldown and non-owner ingress refusal, then obtain a
fresh full Linux CI test result.
The roadmap's request to copy a stopper pattern does not authorize access to
`scribe11`; do not read or modify that resource group.

Deployment entry points now check the C$10 budget policy and retain the separate
Phase 7 go-live gate. The legacy provisioning path that created storage was
removed. README and status identify the candidate as a failed research preview.
A budget alert is not a guaranteed
billing cap because cost reporting and stop actions can lag.

Run a fresh full Linux test suite, verify refusal from an external non-allowed
IP, test stop/restart, and record budget threshold readbacks before declaring
Phase 0 complete. Keep one PR for this phase; merge only with green CI.
Do not advance to Phase 1 until all Phase 0 exit conditions are measured.

No training data, evaluation data, or production baseline was changed.
