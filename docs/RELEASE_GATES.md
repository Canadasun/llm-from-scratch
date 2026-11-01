# Release gates vs. regression gates

Two separate questions, kept separate on purpose (external audit: do not conflate
"did not regress" with "meets a minimum quality bar"):

1. **Regression gate** (CI, `evals/regression.py` against `evals/baselines/m3.json`):
   did this change make the model *worse* than the recorded baseline? It is
   a predeclared practical loss budget (not a statistical significance test) and is a
   development regression guard. It never asserts the model is *good*.

2. **Release gate** (a deliberate, pre-release check): does the model meet the
   **absolute minimum capability** required to present it for a given use? This
   is enforced by a baseline's optional `absolute_floors` block, which the same
   `check()` evaluates as separate `min:<metric>` checks, recomputed from task
   outcomes. It is NOT wired into CI, because the current model does not meet
   production floors and that must not block development — it is run explicitly
   before any release claim.

```bash
# Release gate: the candidate must clear the absolute floors, not merely avoid regressing.
python -m evals.regression --results run.json --baseline evals/baselines/release-gate.example.json
```

## Capability gates to measure independently before a release claim

Per the audit, a release as a Python security/coding assistant requires
INDEPENDENT measurement (hidden tests, frozen sets, split by repository and
problem family so memorization cannot look like progress):

- **Executable correctness** on unseen Python tasks (the sandboxed function
  evaluator; code tasks must actually run — `code_execution` gate).
- **Vulnerability precision and recall** by supported vulnerability class.
- **Patch quality**: the fix removes the vulnerability *and* preserves behavior
  (the verification workflow: finding removed on re-scan AND tests still pass).
- **Safety**: no unsafe recommendations (negative examples / `forbidden`), no
  missed authorization boundaries, prompt-injection resistance.
- **Abstention**: the model declines or asks for clarification when evidence or
  verification is insufficient, rather than fabricating.

`absolute_floors` can enforce the quantitative ones once those tasks exist and a
frozen evaluation set is measured. The qualitative ones (injection resistance,
abstention) need dedicated task families; they are tracked in
[AUDIT_FOLLOWUP.md](AUDIT_FOLLOWUP.md) and are not yet met.

## Qualify the EXACT served artifact

A release record must bind, and be measured on, the exact thing that serves
traffic (audit): the quantized checkpoint **inside the deployment image**, not an
unquantized local checkpoint. Bind into one record: checkpoint SHA-256, image
digest, tokenizer, evaluator version, scoring version, and results. The baseline
`meta` already carries checkpoint/suite/heldout SHAs, scoring_version and
code_evaluator; the image digest and a quantized-artifact evaluation are the
remaining pieces and must be added to the release record before any release.


## November 1 enforcement update (supersedes optional release procedure above)

See [AUDIT_REMEDIATION_2026-11-01.md](AUDIT_REMEDIATION_2026-11-01.md).
`production.json` and `python -m deploy.release qualify|verify` enforce minimum
capability and immutable, quantized-image evidence. Heuristic generation scorers
cannot qualify, even if their reported accuracy is high. The old example remains
illustrative only. Historical v2 results require remeasurement under scoring v3.
Azure canary/promotion is blocked under the user's $0 Azure budget. These controls
do not themselves establish model capability or operational enterprise readiness.
