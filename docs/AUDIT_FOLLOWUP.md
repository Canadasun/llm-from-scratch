# Enterprise audit implementation and remaining work

This follow-up implements local code fixes. It does not establish enterprise
readiness or improve the trained model's instruction-following capability.

## Implemented

- Store-owned key lifecycle transactions serialize separate managers. Local
  SQLite uses `BEGIN IMMEDIATE`, including across separate DB connections.
  Rotation's new key and retirement of the old key commit or roll back together.
- Rate limiting only evicts fully refilled buckets. When all tracked budgets
  remain spent, new identities are denied instead of resetting an existing
  client's budget. This limiter is still process-local.
- Likelihood scoring preserves the available left context for every prediction.
  Long candidates require more forward passes; remeasure capability baselines
  before comparing reports produced with the former algorithm.
- Oversized tokenized prompts return HTTP 400 with the model's context limit.
  Requests within that limit still use a sliding context during generation.
- Inference has a nonblocking generation concurrency cap, configurable with
  `python -m serve --max-concurrent 2`. Excess work gets HTTP 503. This caps
  generation. HTTP connection workers are separately bounded with
  `--max-connections 32`, including idle keep-alive clients. Excess connections
  get a best-effort 503 without creating a worker. Malformed/duplicate length
  headers and unsupported transfer encoding are rejected with connection close.
- Explicit unauthorized intent no longer becomes `allow` merely because a
  prompt contains "lab". Sensitive protective investigation is flagged instead
  of automatically blocked. Regex classification remains a heuristic and never
  establishes authorization.
- Code tasks now use the independent JSON-function evaluator described in
  [CODE_EVALUATION.md](CODE_EVALUATION.md), with mandatory bubblewrap/seccomp
  isolation. Expected answers and verdict logic stay outside the worker. The
  legacy unittest helper still refuses untrusted code; only reviewed fixtures
  may explicitly opt into that helper's trusted mode.
- The model-evaluation workflow no longer silently passes without an artifact.
  It downloads an HTTPS checkpoint, checks its SHA-256, and runs on PRs, version
  tags, manual dispatch, and a nightly schedule.
- Training saves periodic atomic checkpoints with optimizer/scaler, completed
  step count, training configuration, full corpus hash, and device RNG state.
  Interrupted runs can resume without restarting optimization. Small invalid
  splits and nonpositive step/save counts produce clear errors.

## Using resumable training

```bash
python train.py --preset gpu --checkpoint-interval 100 --out-dir checkpoints/run1
python train.py --resume checkpoints/run1/ckpt.pt --max-iters 6000 --device cuda
```

`--max-iters` is the total target step count, not extra steps. Resume requires
the same corpus, tokenizer/model configuration, and device string. CPU equivalence
is covered by a continuous-vs-resumed training test. CUDA/MPS state is saved but
accelerator resume has not been validated here; identical results across hardware,
PyTorch versions or compilation modes are not guaranteed. Old inference-only
checkpoints still load for inference but cannot resume training. Existing trained
checkpoint artifacts were not modified.

## Repository configuration still required

Configure `MODEL_CHECKPOINT_URL` and `MODEL_CHECKPOINT_SHA256` repository variables
with the candidate artifact, or provide both on manual workflow dispatch. Missing
configuration fails the evaluation workflow. Configure branch/release protection
to require the check. The artifact publisher must ensure the selected checkpoint
belongs to the candidate release; a repository variable alone cannot prove that
binding. No cloud deployment or repository setting was changed by this follow-up.

## Remaining work, in priority order

### P0 — release blockers

1. Deploy and operationally harden the independent evaluator. JSON-returning
   function tasks are implemented and tested locally; arbitrary unittest,
   stateful, repository-level, and multi-language tasks still need independent
   protocols. Add fleet quotas, job isolation/patching procedures and monitoring.
   Configure a Linux host with bubblewrap/libseccomp and working namespaces;
   unavailability blocks the evaluation gate rather than silently lowering scores.
   - DONE: active `sandbox_preflight()` runs a known-correct canary through the
     real sandbox (a present-but-broken bwrap/seccomp/namespaces reports
     unavailable); `python -m evals.code_executor --preflight` (exit 0/1, JSON)
     is the CI/health-check hook; `evals.run` records `meta.sandbox`; an
     unexecuted code task is reported UNMEASURED (excluded from accuracy, loud
     banner) instead of scored 0, and the gate blocks on unavailability via the
     `code_execution` and incomplete-suite checks. See docs/CODE_EVALUATION.md.
   - REMAINING: a deployed multi-host service with real fleet quotas, job
     isolation/patching procedures, and independent protocols for unittest,
     stateful, repository-level, and multi-language tasks.
2. Establish a useful narrow model workflow with independently measured secure
   coding and instruction-following. Current weights and training data quality
   are unchanged. Passing infrastructure tests does not establish model skill.
3. Move organizational persistence to a transactional database with DB-enforced
   tenant boundaries. SQLite `nolock=1` over a network share cannot provide
   cross-process exclusion; externally enforce one writer until it is replaced.
4. Add SSO/MFA, service identities, controlled provisioning/offboarding, immutable
   identity ownership, and project/repository permissions.
5. Add customer data lifecycle controls: retention/deletion, key management,
   tenant-scoped access, approved logging, and tested backup/restore.
6. HTTP worker admission is bounded locally. Still add distributed tenant quotas,
   absolute request deadlines, cancellation, and a production server/proxy.
   Validate under sustained load; socket inactivity timeouts do not stop a peer
   that keeps sending data slowly, and local admission is not a fleet quota.

### P1 — organizational usability and operations

- Version and verify release artifacts, data manifests, tokenizer and policies;
  validate the candidate-to-checkpoint binding, rollout and rollback procedures.
- Re-run independent code/security benchmarks, measure false positives/negatives
  by vulnerability class, and evaluate prompt injection and data exfiltration.
- Add permission-aware retrieval with citations, freshness and cache revocation.
- Add distributed observability, readiness checks, alerting and incident runbooks.
- Broaden policy evaluation beyond keyword heuristics, including legitimate
  investigative requests, negation, obfuscation, languages and mixed intent.
- Validate CUDA/MPS/compiled training resume and interruption/recovery at scale;
  add dataset provenance, document splits, deduplication and contamination tests.
- Add IDE/PR integrations, an accessible review UI, findings ownership and
  exceptions, API specification/client examples, and controlled feedback intake.

### Later differentiators

Evidence-backed findings, security regression replay, independently verified
patches, organization-specific exposure analysis, machine-enforced authorization
for security actions, revocable organizational knowledge, and calibrated
abstention remain product development work.
