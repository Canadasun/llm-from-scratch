# November 1 audit remediation — local changes, Azure budget $0

No cloud resources, deployments, paid training, remote database, or Redis service
were created or changed by this work. `deploy/canary.sh canary|promote` now refuses
before any cloud command. Local build, qualify and verify remain available.
Existing Azure costs/configuration have not been inspected or changed; this is
not an assertion that an already-running subscription currently costs zero.

## Implemented controls

- Writer exclusion uses atomic directory creation, never overwrite-and-reread.
  Locks do not expire or get stolen: a paused process must not lose ownership to
  an unfenced second writer. A crash leaves an occupied lock requiring offline
  recovery after **all holders are stopped**. Legacy lock files also fail closed.
- SQLite `nolock` is refused before opening a database. Even a correct writer
  lock cannot protect unlocked SQLite readers during a commit. Normal locking on
  suitable local storage is supported; network-share SQLite is not qualified.
  Nothing automatically migrates or deletes an existing database.
- Repository execution defaults to sandboxed mode. `--trusted` / `trusted=True`
  explicitly opt into host execution. Bubblewrap is mandatory for untrusted jobs;
  missing isolation is a failed result. Memory, process, descriptor, CPU and file
  limits apply, output is disk-backed with a bounded diagnostic tail, and child
  process groups are killed/reaped. A dedicated non-root worker UID is required;
  RLIMIT_NPROC is per OS user, not a replacement for production cgroup limits.
- Copying preserves symlinks rather than following host targets; scanning skips
  symlinks, and patch proposals use scanner file/size limits.
- The default untrusted fix workflow requires an independent behavioral security
  witness. The first supported grammar is a one-argument digest function returning
  `hashlib.md5/sha1(value).hexdigest()`. Controller-owned inputs/expected SHA-256
  digests establish an output mismatch before and passing results after the fix.
  Expected answers stay outside the sandbox. Unsupported shapes/rules fail closed
  to manual review. Generated witnesses are exported with PR artifacts. Replay after applying a
  patch with `python -m codereview.security_tests --repo PROJECT --evidence
  codereview_security_tests.json`; expected answers are rebuilt by the controller,
  never trusted from the exported file. This
  establishes algorithm replacement for that grammar, not password-storage
  suitability, application-wide exploit resistance, or model coding ability.
- Generation checks cancellation/deadlines between tokens/forward passes. HTTP
  peer disconnection cancels its request; nested service budgets retain the
  shorter deadline and generation slots release on cancellation. A running native
  kernel cannot be forcibly interrupted: production process supervision is still
  required for hard deadlines.
- An optional shared Redis quota adapter atomically charges tenant and principal
  buckets, uses server time/expiry, and fails closed on backend errors. It is wired
  into secured serving via `--quota-redis-url` or `LLM_QUOTA_REDIS_URL`; it requires
  an **existing** endpoint and the optional `redis` client. Remote endpoints require
  TLS; only loopback development endpoints can use plain Redis. No Redis instance
  is provisioned and none is required for local single-process use.
- Contradictory/negated keyword hits are rejected more conservatively. Lexical and
  regex scorers remain explicitly heuristic: production qualification refuses
  them, including attempts to relabel their results as exact. This intentionally
  blocks today's generation suite pending reviewed replacement tasks.
- Category regression guards now use a predeclared practical accuracy-loss budget,
  not the baseline's Wilson lower bound. This is a policy threshold, **not** a
  significance/non-inferiority claim. Wilson intervals remain descriptive.
- Scoring is versioned anew. Historical results are not silently rewritten:
  remeasure baseline and candidate under the same new scoring version before a
  valid regression comparison.
- Production capability floors are checked separately from development regression
  checks. The policy is `evals/baselines/production.json`; floors are initial
  engineering acceptance targets, not proof of enterprise fitness. Image
  qualification requires complete current-suite outcomes, independent code
  execution, quantization, checkpoint/tokenizer/source fingerprints, matching
  policy and suite hashes, and a digest-pinned image.
- `python -m deploy.release qualify --image registry/repo@sha256:... --record out.json`
  runs that image locally with no network, a read-only root, non-root UID and
  resource limits. It uses the serving quantizer via `evals.run --quantize`.
  A host/container unable to run the independent sandbox **fails qualification**;
  privileges are not silently relaxed. Docker itself/images must already be
  available locally; this work did not build/pull images or run paid jobs.
- `python -m deploy.release verify --image registry/repo@sha256:... --record out.json`
  recomputes gated outcomes and checks evidence bindings. Records are protected
  CI artifacts, not cryptographic attestations: their publisher/policy must be
  trusted and access controlled. Authenticated functional probes are mandatory
  in the retained, currently disabled promotion path; HTTP 200 with a wrong
  answer fails. Latency samples measure generation and include failed requests.

## Still not achieved — do not call these fixed

1. **Model capability**: no newly trained/qualified model. The latest recorded
   candidate remains 0/18 coding tasks. No paid GPU work is authorized here.
2. **Breadth of secure remediation**: only a narrow reviewed grammar has an
   independent generated witness. Real-repository qualification and independent
   witnesses for TLS, YAML, debug configuration and other vulnerability classes
   remain work. Unsupported cases refuse verified status by default.
3. **New frozen evaluation evidence**: replace heuristic generation tasks with
   reviewed exact/executable/security-specific tests; extend injection,
   authorization and abstention tests; remeasure under scoring v3. Do not merely
   change labels or tune acceptance thresholds to get a pass.
4. **Served artifact qualification**: tooling is implemented, but no current
   deployment image has passed it. The existing weak model and heuristic suite
   deliberately cannot qualify. No image rollout was attempted.
5. **Fleet operations**: shared quotas have a local cross-client integration test;
   availability, concurrency, recovery and sustained load across real hosts have
   not been established. Process-level hard deadlines and a qualified proxy/server
   remain operational requirements. Provisioning new infrastructure is forbidden
   under this task's $0 Azure constraint.
6. **Enterprise controls**: Azure owner must demonstrate identity/offboarding,
   tenant/data isolation, supported persistence, secrets, retention, backup/restore,
   incident response and rollback. No claim about live controls is made here.

Use local/nonproduction workflows while these remain open. Do not enable paid
services, overlap Azure revisions, or bypass qualification to hide capability or
operational gaps.
