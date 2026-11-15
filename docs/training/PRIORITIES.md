# Queued development priorities

Owner direction, November 14, 2026: make this from-scratch model as capable as
possible at cybersecurity and Python programming. This is a long-term research
objective. A claim of being the smartest model requires independent comparative
evidence and is not established by this project.

## Latest completed experiment

The [broader curriculum trial](curriculum-v2-2026-11-14.md) expanded training from
18 to 48 labelled families, with 78 total families and 401 passing reference cases.
It passed the resume check and completed 100 GPU steps. Chat syntax improved, but
functional coding did not, and completion syntax regressed. The candidate was
rejected and the Phase 4 base retained. The preceding seed trial also failed.
Stop this sequence of small-curriculum SFT trials; do not simply add more steps.

## Queued after the pilot

1. The [base diagnosis](base-diagnosis-2026-11-14.md) found that document-prefix
   sampling made 58% of train tokens unreachable. After review of the versioned
   sampler repair, prepare the bounded prefix-versus-uniform base-pretraining
   control described there. Use a separate weights-only warm start with fresh
   optimizer/RNG; changing old exact resumes is forbidden. Stratify uniform code
   and prose validation and retain the existing capability gates. No larger model
   or new corpus is yet justified as the first intervention.
2. If justified by that investigation, grow distinct first-party or licence-approved problem families with executable
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
