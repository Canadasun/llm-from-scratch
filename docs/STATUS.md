# Honest production-readiness status

This states plainly what the build is and is not, in response to the external
audit of commit e6b115b. It is deliberately not promotional.

## Headline

November 14: [training recovery and real-checkpoint CPU pilot](training/recovery-2026-11-14.md)
fixes SFT checkpoint/data integrity and resume defects. A small validation sample
improved during eight training steps; capability improvement is unproven and the
accepted base remains unchanged. The report records remaining baseline test failures.

The subsequent [verified-code GPU pilot](training/verified-seed-2026-11-14.md)
completed exact CUDA resume and 30 training steps. Sandboxed matched-mode scoring
found no functional coding gain and a small chat syntax regression, so the candidate
was rejected. Linux CI passed 525 tests with one skipped. Python and cybersecurity
research priorities are recorded in [the queue](training/PRIORITIES.md).

The subsequent [broader curriculum trial](training/curriculum-v2-2026-11-14.md)
trained on 48 labelled families with 401 reference cases across 78 total families.
After 100 GPU steps, chat syntax improved but functional coding did not; completion
syntax regressed. This candidate was also rejected. Further small-curriculum SFT
steps are paused pending investigation of base pretraining and format transfer.

**This is not a production cybersecurity/Python coding assistant.** It is a
from-scratch research model with a hardened evaluation and serving harness. Neither software changes nor passing infrastructure tests establish enterprise
readiness; the model's demonstrated coding ability remains inadequate.

## What the measured evidence actually shows

On the frozen 170-task suite (independent evaluator, sandboxed code execution):

- Overall capability ~33% (MC-weighted); **coding 0/18**; secure_development
  ~2/12; generation weak. The instruction candidate beats the prior baseline but
  "30% → 33.5%" does not establish useful programming ability.
- Intrinsic language modeling genuinely improved (bits/byte 2.15 → ~1.77), i.e.
  it models the domain's *form*, not its *competence*.

The release gate (`docs/RELEASE_GATES.md`, `evals/baselines/release-gate.example.json`)
encodes production floors; the current model **fails** them, which is the correct,
honest result.

## Why code cannot fix the headline (audit Critical / High #1-#2)

Coding ability is a function of training data, context length, and evaluation —
not of the serving or evaluation code. Moving it requires work this repo's code
changes cannot substitute for:

- **Training data**: appropriately licensed, verified Python fixes; realistic
  projects; secure-coding examples and clean-code counterexamples; and cases that
  require refusal or clarification. (Public / self-authored only — never customer
  or PII data.)
- **Context**: 512 tokens is inadequate for repository-level work.
- **A real GPU training run** on free Kaggle quota, after validated resume and
  throughput smoke tests. Paid GPU resources require separate owner approval.
- **Evaluation split by repository and problem family** so memorization cannot
  masquerade as progress, with hidden tests and frozen sets.

More training *steps alone* will not fix weak data, short context, or weak
evaluation.

## The end-to-end workflow (audit High #2)

The intended product is one narrow, verified loop, not broad autonomy:

> authorized repository → security finding → proposed patch → isolated
> functional/security verification → reviewed PR (human approval required)

The pieces exist and are now bound together: `codereview/` scans (with import
resolution and declared coverage limits, `codereview/COVERAGE.md`), proposes
fixes for a limited class set, verifies on a copy that the finding is removed AND
tests still pass (in a bubblewrap sandbox for untrusted repos, fail-closed), and
`codereview/pr.py` emits a PR bound to the verified commit and exact patch hash.
What is still missing for a product claim: a generated reproducing test per
finding, and qualification across real repositories.

## What still requires evidence (not yet established here)

- Qualify the **exact served artifact**: evaluate the quantized checkpoint inside
  the deployment image and bind image digest + checkpoint/tokenizer/evaluator
  versions + results into one release record (`docs/RELEASE_GATES.md`).
- Enterprise operations from the Azure workstream: SSO/MFA and offboarding,
  tenant isolation and a transactional datastore (lock-disabled SQLite is now refused; atomic writer locks supplement normal
  locking on supported local storage — see
  [AUDIT_FOLLOWUP.md](AUDIT_FOLLOWUP.md)), secret handling, audit retention,
  tested backup/restore, quotas and load behavior, incident response, rollback.
- Reviewable UX: IDE/CLI or PR integration with exact diffs, reproducible
  findings, visible verification status, and clear refusal reasons.

See [AUDIT_FOLLOWUP.md](AUDIT_FOLLOWUP.md) for the itemized status of every audit
finding (which are fixed in code, and which are capability/ops work).


## November 1 local remediation

[Detailed status and limitations](AUDIT_REMEDIATION_2026-11-01.md) supersede older
claims that the lease, keyword scorer or optional promotion checks completely
resolved those findings. The owner approved a C$10 monthly hosting plan on November 5. New model
rollouts still require qualification and the separate Phase 7 go-live approval.
See [Phase 0 evidence](implementation/phase-0-preparation.md).
The default untrusted review workflow now requires a narrowly supported independent
behavioral witness; unsupported rules require manual review. Model qualification,
real-repository validation and live enterprise evidence remain outstanding.


## Further readiness work (November 1)

See [measured readiness resolution](READINESS_RESOLUTION_2026-11-01.md) for the
170-task cached-image measurement, three pinned real-repository scans and 9/9
controlled mutations, optional organization identity, local restore exercise, and
free-GPU training handoff. The model still scores 0/18 coding tasks. Neither the
local controls nor the new evidence constitutes a passing enterprise release.
