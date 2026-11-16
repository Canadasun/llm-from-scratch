# Bounded base-sampler control

Protocol declared before GPU training. [Measured results](sampler-control-results-2026-11-14.md)
report a primary loss-criterion pass but no functional coding gain, with syntax
and prompt-injection regressions; both candidates are rejected. This tests the document-body exposure
repair from PR #31. It does not retrain or replace the served model automatically.

## Initialization and the only arm difference

Start both arms from accepted weights `1164213e635d376946ec331d5ff238cf89f492788554a953010f59bd5f57a593`.
`training.sampler_control` imports weights/model/tokenizer only. Each arm gets a
fresh AdamW optimizer, scaler and RNG; it never imports the step-9,176 optimizer.
Initial tensor hashes, optimizer-state counts, scaler states, CPU/device/batch RNG
hashes and full recipes are recorded and checked for equality.

Control uses `legacy_doc_prefix_v1`; treatment uses `uniform_windows_v1`. Identical
recipe otherwise: 200 attempts per arm, batch eight, accumulation four, context
1,024, LR 1e-5, 20 warmup steps, cosine decay, weight decay 0.1, clipping 1.0,
seed 1337 and batch seed 1338. That is 6,553,600 target tokens processed per arm.
Both use deterministic algorithms, SDPA's math backend, TF32 disabled and the
same hardware/runtime. FP16 starts with scale 1,024 for both arms; any skipped
optimizer update makes the comparison ineligible rather than silently changing
the effective budget. A one-hour core deadline also blocks incomplete runs.

Evaluate only the fixed endpoint, not the best-looking intermediate checkpoint.
Artifacts are weights-only warm starts with explicit parent/data/plan hashes and
`optimizer_resume_available: false`. Continuing them requires a separately named
experiment with fresh optimizer state, not a claim of exact optimizer resume.

## Validation frozen before training

[Full plan](sampler-control-validation-2026-11-14.json), sha256
`3ee5f39fc87341ac1385792be86187fcadc3e1474e799aadeff540f774e67016`.
It binds corpus/shard/tokenizer hashes and 320 token-window coordinates, not code
text or benchmark answers. Building it reads validation only and verifies each
selected document's token sequence against its stored shard. Private and corpus
test holdouts are untouched. Uploaded datasets and kernel are private.

Seed 1729 selects one file from each of 128 distinct Stack repository metadata
groups and 32 prose documents. Only documents at least 2,050 tokens long qualify;
659 code groups and 50 prose documents are eligible. Each selected document has a
prefix window and a uniformly selected legal in-document window of 1,024 targets.
The four strata are code-prefix/code-uniform (128 each) and prose-prefix/prose-uniform
(32 each). Code-uniform has 105,596/131,072 targets beyond the document's initial
prefix; prose-uniform has 32,517/32,768. This deliberately diagnoses long-document
coverage, not whole-corpus or globally token-weighted performance.

Every model is scored on the same windows in float32, with no gradient updates.
Code is repository-balanced; cross-repository similarities can still exist.
Prose chunks share source documents/groups and are secondary observations, not
32 independent security tasks. Prior mixed validation losses are not reused as
interchangeable baselines for this plan.

## Predeclared primary gate

Treatment code-uniform loss must beat **both** the control and the unchanged
accepted base by at least 0.01 nats per target token. Both paired mean differences
must have a 95% bootstrap interval entirely below zero, using 2,000 repository-pair
resamples with seed 1730. There must be at least 64 distinct code repository pairs
(the real plan has 128). Treatment prose-uniform mean loss may rise by no more than
0.02 nats relative to base. Prefix and all source-stratified results are reported,
even if the primary criterion fails.

The single training seed is a limitation: these intervals describe validation-window
variation, not variation across independently trained seeds. Language-model loss
improvement is not proof of functional coding, safe behavior or broad model ranking.

## Conditional capability screen and stopping

If the primary gate passes, generate the unchanged observed first-25-task code
screen for base/control/treatment in completion and chat modes, greedy with 384
new tokens, plus the existing behavior categories. Generated code runs only in
the existing Linux sandbox. Base scores may be reused only after exact output,
behavior and relevant inference/benchmark metadata equality is checked.

This observed public subset is not independent final qualification. Report functional
and behavior changes separately from language-model loss. No candidate is accepted
or deployed from this pilot; full frozen/preview qualification and separate hosting
authorization remain required. If the primary gate fails, skip capability generation,
retain the accepted base and stop the bounded trial. Never add steps to manufacture
a pass. Paid compute and infrastructure changes are outside this experiment.

## Implementation checks

The control-plan/warm-start/assessment tests cover deterministic paired selection,
SHA and tokenizer mismatch, boundary/pair integrity, fresh optimizers despite a
poisoned old optimizer payload, identical initialization, the 200-attempt cap,
meaningful effect size, prose retention, distinct repositories, mismatched RNG
and recipes, overflow/incomplete comparisons, and rechecking the gate before
generation. Seventeen new tests and 49 existing sampler/training checks pass
locally. GPU execution and its measured results are recorded in the linked report.
