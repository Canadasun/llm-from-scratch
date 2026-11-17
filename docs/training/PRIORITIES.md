# Queued development priorities

Owner direction, November 14, 2026: make this from-scratch model as capable as
possible at cybersecurity and Python programming. This is a long-term research
objective. A claim of being the smartest model requires independent comparative
evidence and is not established by this project.

## Latest completed experiment

The [sampler control](sampler-control-results-2026-11-14.md) completed 200 updates
per arm. Uniform sampling passed the predeclared code-body loss criterion versus
both prefix control and unchanged base, with prose retention. It did not improve
functional coding and regressed completion syntax and prompt-injection accuracy.
Both candidates were rejected; the accepted base remains unchanged. Do not extend
this pilot merely because loss improved.

## Earlier curriculum result

The [broader curriculum trial](curriculum-v2-2026-11-14.md) expanded training from
18 to 48 labelled families, with 78 total families and 401 passing reference cases.
It passed the resume check and completed 100 GPU steps. Chat syntax improved, but
functional coding did not, and completion syntax regressed. The candidate was
rejected and the Phase 4 base retained. The preceding seed trial also failed.
Stop this sequence of small-curriculum SFT trials; do not simply add more steps.

## Queued after the pilot

1. The [offline syntax diagnosis](syntax-diagnosis-2026-11-16.md) is complete:
   treatment loses 13 previously valid completion programs and gains five;
   nine of ten parseable chat outputs are empty. No observed valid full response
   is made invalid by extraction. The [original format probe](format-probe-2026-11-16.md)
   now has 12 families and 24 paired prompts; isolated verification passed all
   references and rejected empty/wrong-entry/pass-only/wrong-value controls.
   Next measure accepted-model outputs on both formats, preserving the six
   evaluation families and recording stop reasons/token counts, before a new
   training proposal. No model capability has yet been measured on this probe. The
   [base diagnosis](base-diagnosis-2026-11-14.md) and completed control establish
   that repaired body exposure improves loss, but not useful coding in this budget.
   Inspect corpus format/provenance without training on benchmark answers.
   Before more GPU training, predeclare a distinct
   data/format hypothesis, independent family-level evaluation and bounded budget.
   Retain uniform sampling for new experiments and preserve historical sampling
   on exact resumes. No larger model or immediate run extension is justified by
   this pilot alone.
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
