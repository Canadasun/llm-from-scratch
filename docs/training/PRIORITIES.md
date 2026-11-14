# Queued development priorities

Owner direction, November 14, 2026: make this from-scratch model as capable as
possible at cybersecurity and Python programming. This is a long-term research
objective. A claim of being the smartest model requires independent comparative
evidence and is not established by this project.

## Latest completed experiment

The verified code-only seed pilot completed with exact GPU resume and sandboxed
matched-mode scoring. It produced no functional coding gain and a small chat
syntax regression. The candidate was rejected and the Phase 4 base retained.
See [results](verified-seed-2026-11-14.md). Do not expand this failed seed run by
simply adding more steps.

## Queued after the pilot

1. Use actual failing outputs to choose the next dataset gaps: Python function
   correctness, debugging and tested repairs; cybersecurity explanations and
   secure implementations. Distinguish these response formats explicitly.
2. Grow distinct first-party or licence-approved problem families with executable
   expected-behavior cases, targeted insecure/incorrect variants, edge cases and
   clean counterexamples. Prioritize input handling, access control, secret
   handling, safe database/subprocess usage, and reliable Python error handling.
3. Keep problem-family and repository provenance explicit. Keep all variants and
   repairs of a problem in one split. Check benchmark contamination before training;
   never use independent evaluation answers or private/customer data as training.
4. Run bounded free-GPU trials from this project's accepted weights, with replay,
   deterministic resume and held-out monitoring. Revisit corpus diversity and
   model capacity when repeated measured failures justify it.
5. Require functional coding improvements and security-category retention on the
   unchanged benchmarks before accepting a new model. Report losses, parse rates,
   functional results, sample sizes and regressions separately.

The from-scratch constraint remains: no imported pretrained weights or external
model API standing in for this model. Paid compute requires separate owner approval.
The deterministic scanner/verifier remains responsible for findings and verified
patches; training a candidate does not authorize a hosting rollout.
