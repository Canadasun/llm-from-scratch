# Readiness work and measured outcome — November 1, 2026

This follow-up implemented local controls and gathered missing evidence. It did
**not** make the current weights a useful production Python model, alter Azure,
start paid training, or certify enterprise readiness. The release remains blocked
for measured reasons, not merely missing implementation.

## Four requested blockers

| Blocker | Work completed | Current decision |
|---|---|---|
| Model coding capability | Measured all 170 tasks against the cached immutable image referenced by the Azure deployment, with quantized inference and independent host-side code verdicts. Prepared a code-first, longer-context free-GPU experiment and notebook. | **Failed:** 0/18 coding, 2/12 secure development; 57/170 overall includes heuristic generation scores. No GPU is accessible here; no improved weights are claimed. |
| Broader real-repository coverage | Scanned pinned Requests, Flask and PyYAML checkouts (167 Python files). Independently reproduced and fixed 9/9 injected digest/TLS/YAML defects across real source files. Protected explicit non-security/protocol hashes from automatic rewriting. | **Improved, narrowly qualified.** Mutation coverage does not establish natural-vulnerability precision/recall, full upstream compatibility, or general remediation. |
| Actual serving-image qualification | Added a controller that can measure legacy cached images without modifying them, injecting new model code, pulling images, or contacting Azure ingress. Verified immutable image/checkpoint/tokenizer/source fingerprints and kept expected answers outside the image. | **Measured and failed capability.** Azure metadata still uses a mutable tag; its live pulled digest is not attested. This local cached digest must not be represented as proof of the currently running container's identity. |
| Live enterprise controls | Collected allowlisted Azure management metadata, added optional organization JWT/MFA-claim verification and immediate local offboarding, exercised local backup/restore and tenant isolation, and blocked a second provisioning path under the $0 budget. | **Local code implemented; live controls remain incomplete/unmeasured.** No identity policy, storage migration, cloud configuration, live recovery/load/rollback exercise or rollout was performed. |

## Recorded evidence

- [Immutable image task evaluation](evaluations/serving-image-2026-11-01.json):
  `ghcr.io/canadasun/llm-serve@sha256:2eb13a12f45df6f710c9b5ae32f49dda7cd7574d07e74bfa8fec657a2dc9d669`.
  All 18 code tasks executed; none were silently treated as a measured failure
  because isolation was unavailable. The controller records code completions and
  independent verdicts. This is a full capability-task run, not an intrinsic-
  metrics run or a passing production qualification.
- [Release decision](evaluations/serving-image-decision-2026-11-01.json): capability
  floors fail. Heuristic generation scores and unverified live digest also block.
- [Pinned real-source qualification](evaluations/real-repositories-2026-11-01.json):
  upstream URLs, exact commits, source hashes, natural findings, and separate
  controlled mutation outcomes. Upstream checkouts were not modified or imported
  on the host. Their full test suites were not executed or claimed to pass.
- [Azure management snapshot](security/azure-readiness-2026-11-01.json) and
  [control assessment](security/azure-controls-2026-11-01.json): TLS ingress rejects
  insecure access; no configured platform/Entra authentication settings were
  returned, identity type is None, the image is tagged, and storage is Azure Files.
  These reads did not contact `/healthz` or `/generate`, wake a replica, list secret
  values, or mutate resources. Existing subscription expenditure was not audited;
  no claim that already-running infrastructure costs exactly zero is made.
- [Local recovery exercise](security/local-recovery-2026-11-01.json): restored
  synthetic identities, preserved tenant read/list boundaries, found no plaintext
  API key in the backup, and verified private backup file permissions. This is not
  evidence of live backup retention, encryption, RPO/RTO or disaster recovery.

## Organization identity implementation

Install `requirements-identity.txt` (also included in development tests and the
next container build). Set `--secured --organization-policy /trusted/policy.json`
or `LLM_ORGANIZATION_POLICY` with:

```json
{
  "issuer": "https://YOUR-TRUSTED-ISSUER",
  "audience": "YOUR-APPLICATION-AUDIENCE",
  "required_amr": ["mfa"],
  "max_token_age_seconds": 3600,
  "jwks": {"keys": [{"kty": "RSA", "kid": "KEY-ID", "use": "sig", "alg": "RS256", "n": "PUBLIC-MODULUS", "e": "AQAB"}]},
  "bindings": {"EXTERNAL-SUBJECT": {"enabled": true, "principal_id": "LOCAL-PRINCIPAL", "tenant_id": "LOCAL-TENANT"}}
}
```

The policy is administrator-controlled, regular, size-bounded, not group/world
writable, and read on every authentication. Keys are pinned public RSA keys;
signature, algorithm, issuer, audience, expiry, not-before, issuance age and
required authentication-method claims are validated. Roles/tenant claims do not
create authority: the existing local principal and its explicit binding decide.
Disabling a binding takes effect without a token cache. Organization mode refuses
legacy API keys rather than allowing them to bypass the organization policy.
Unknown keys and token-supplied key URLs are rejected. Key rotation is an explicit
identity-owner operation; no discovery or JWKS URL is fetched from token input.

The JWT implementation uses the documented [PyJWT RS256 verification API](https://pyjwt.readthedocs.io/en/stable/usage.html).
A signed MFA claim is an assertion from the trusted issuer, **not** proof that the
organization has enrolled users or configured an effective MFA/offboarding policy.
No real issuer/audience/public-key binding was supplied, configured or enabled in
Azure during this work.

## Reproduce locally

```bash
# Real checkouts must already match the pinned commits in qualify_repositories.py.
LLM_EVAL_BWRAP=/path/to/bwrap python -m codereview.qualify_repositories \
  --root /path/to/checkouts --json repository-evidence.json

# Cached image only: no pull, no image replacement, no Azure traffic.
LLM_EVAL_BWRAP=/path/to/bwrap python -m evals.serving_image \
  --image ghcr.io/canadasun/llm-serve@sha256:2eb13a12f45df6f710c9b5ae32f49dda7cd7574d07e74bfa8fec657a2dc9d669 \
  --json image-evidence.json

python -m enterprise.recovery --report local-recovery.json
python -m enterprise.readiness assess --snapshot azure-readiness.json --report controls.json

# CPU-safe plan only. Actual training needs an existing operator-confirmed FREE CUDA runtime.
python -m training.free_gpu --output checkpoints/code-first-gpu-experiment --plan
```

The [free-runtime notebook](../notebooks/train_code_free_gpu.ipynb) provides the
handoff. Runtime availability/billing status must be confirmed by its operator;
this repository does not create GPU instances, subscribe to a plan, provide a paid
fallback, or auto-promote the resulting model. The experiment starts fresh at
context 1024, uses a code-first licensed corpus and periodic checkpoints, and has
a total wall-clock limit. It may still fail quality gates. Training/validation
loss alone cannot clear coding, security, abstention or injection-resistance gates.

## Work requiring external access/evidence to close

1. An accessible existing free GPU (or sufficient explicitly allocated local
   compute) and an actual independent improvement in coding performance. A
   notebook or a new preset is not a trained model.
2. A reviewed dataset of natural vulnerabilities/fixes and full application
   compatibility tests beyond the nine controlled mutations.
3. The Azure owner's supported persistence/identity configuration, immutable image
   rollout and live digest evidence, within an explicitly approved zero-cost
   deployment strategy. Current scripts do not bypass the $0 prohibition.
4. Real organizational MFA/offboarding, access reviews, secret lifecycle, audit
   retention, backup restoration, load/recovery and incident/rollback exercises.

Do not mark these complete from local unit tests or change the production floors
to hide the failures. No Azure rollout, new cloud resource, or paid compute was
performed in this follow-up.

## Local validation

The final regression run completed with **402 passed, 1 skipped**, including real
namespace isolation, identity signature/claim/offboarding tests, restore boundaries,
new security witnesses and zero-budget provisioning guards. Five warnings concern
existing PyTorch quantization deprecations. Shell syntax and diff whitespace checks
also passed. These tests are software evidence, not production model qualification.
