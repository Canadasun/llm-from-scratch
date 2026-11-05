# Phase 0 lockdown evidence — 2026-11-05

Status: Phase 0 exit measurements passed; final PR CI/merge pending.

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
  not a production promotion. For the cooldown revision, its existing registry
  tag was resolved to the immutable image index
  `sha256:2eb13a12f45df6f710c9b5ae32f49dda7cd7574d07e74bfa8fec657a2dc9d669`.
  This does not attest the original mutable-tag revision. Full artifact
  qualification remains due in Phase 6.
- The first cooldown update was rejected with `ContainerAppOperationInProgress`:
  Azure reported the app as stopped before its provisioning operation finished.
  Polling now waits for provisioning completion as well as running status.
  The corrected retry passed. No failed attempt is counted as a gate pass.
- Current revision readback: `llm-serve--phase0-cooldown`, provisioning
  `Succeeded`, running `Running`, latest ready revision matching, single revision
  mode, 100% traffic, min replicas 0/max replicas 1, 1 vCPU/2 GiB retained, and
  configured cooldown reduced from 300 to 30 seconds. The app was stopped during
  the change to avoid simultaneous SQLite writers.
- Owner health returned HTTP 200 in 18.690887 seconds after restart. A subsequent
  replica-list read returned 0 replicas. These observations establish successful
  restart and idle scale-to-zero, not an exact 30-second billing interval or the
  smallest possible stable cooldown. Azure also has scale-down stabilization.
- [Linux CI at c90e6aa](https://github.com/Canadasun/llm-from-scratch/actions/runs/42032829325)
  reports **410 passed, 1 skipped, 5 warnings in 23.31 seconds**, with the
  required bubblewrap tests and first-party security scans passing.
- The same CI run's `owner-ingress` job measured **HTTP 403** from GitHub's
  external runner, both normally and with an `X-Forwarded-For: 127.0.0.1`
  header. These probes ran after the new cooldown revision was configured.

## Policy and remaining work

Final evidence changes must have green PR CI before merge. The phase PR is
[PR 10](https://github.com/Canadasun/llm-from-scratch/pull/10).
The roadmap's request to copy a stopper pattern does not authorize access to
`scribe11`; do not read or modify that resource group.

Deployment entry points now check the C$10 budget policy and retain the separate
Phase 7 go-live gate. The legacy provisioning path that created storage was
removed. README and status identify the candidate as a failed research preview.
A budget alert is not a guaranteed
billing cap because cost reporting and stop actions can lag.

The three budget thresholds, external ingress refusal, stop/restart and fresh
Linux test count have now been measured. No later phase has been executed.
The local Kaggle credential file is absent; account creation, phone verification
and owner-installed credentials are required before the Phase 1 GPU smoke run.

No training data, evaluation data, or production baseline was changed.

## Operating the controls

Run from the repository root, using the Azure CLI's current subscription or an
explicit `SUBSCRIPTION` environment variable. No account identifier or recipient
is stored in the scripts.

```bash
bash deploy/allow-my-ip.sh
python3 -m deploy.lockdown verify
python3 -m deploy.lockdown setup-stopper
python3 -m deploy.lockdown test-stopper
```

The IP helper replaces all ingress rules with the detected owner's IPv4 /32;
`--ip ADDRESS` explicitly selects one address. `setup-stopper` reuses an existing
budget recipient unless `ALERT_EMAIL` is provided. `test-stopper` causes a brief
outage and restarts only after both stop and workflow completion are verified;
on failure inspect the app before manually restarting it. The workflow only
stops the app and cannot restart it or operate on other apps.

Budgets and log caps are delayed controls, not guaranteed billing ceilings.
Review costs regularly, stop manually if approaching C$8, and do not restart
after a budget stop until costs and the next billing period have been checked.
The small existing storage account remains until the Phase 7 migration; stopping
the app does not delete storage or stop every possible resource-group charge.
No GPU VM, paid plan, ACR or database was created.

References: [Azure budget automation](https://learn.microsoft.com/en-us/azure/cost-management-billing/manage/cost-management-budget-scenario),
[scale behavior](https://learn.microsoft.com/en-us/azure/container-apps/scale-app#scale-behavior),
[daily cap limitations](https://learn.microsoft.com/en-us/azure/azure-monitor/logs/daily-cap).
