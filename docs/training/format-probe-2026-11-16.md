# Original response-format probe

This implements the next diagnostic from the
[syntax investigation](syntax-diagnosis-2026-11-16.md), not another training run.
The [public prompt manifest](format-probe-2026-11-16.json) has 12 newly authored
families and 24 prompts: one complete-function request and one indented-body
request per family. Six families are assigned to training-side diagnostics and
six to evaluation. Both variants inherit the family split. None of these rows
has been exported into a training corpus or used to update weights.

The tasks cover sequence operations, simple timestamps/budgets and literal
scope/outcome checks. They are small functional diagnostics, not a security audit
or a representative cybersecurity benchmark. Public families and shared general
programming concepts do not establish hidden or semantic independence. These
evaluation families must stay out of future training, including repair variants.
No external benchmark answers, private holdouts or customer data were read.

## What is measured

`data/format_probe_tasks.py` contains original specifications, canonical bodies
and controller-side expected cases. The prompt manifest contains no reference
responses or cases. `evals.format_probe` checks three separate properties:

1. Python syntax after the prescribed assembly. A body is appended to the exact
   supplied signature; a module is used as supplied. No trimming, dedenting,
   Markdown extraction or repair is applied.
2. Exactly one top-level synchronous function with the required name and positional
   argument names, no decorators/defaults/extra argument forms, and an implementation
   beyond a docstring or `pass`. Empty modules and nested replacement definitions
   cannot count as successful entry points.
3. Actual output correctness across the authored cases, using the existing Linux
   bubblewrap/libseccomp controller. Expected values stay outside the worker.
   Function shape alone is not evidence of correctness or secure behavior.

The verifier checks both reference formats and four negative controls for every
family: empty output, wrong entry, wrong value and pass-only output. Reference
responses must pass; every negative control must fail. Wrong-value controls have
valid syntax and shape, so their rejection must come from execution results.
Unavailable isolation or evaluator errors fail verification; there is no host
execution fallback. The source and verifier/executor/worker hashes bind the report.

## Reproduction

Build prompts without executing their code:

```
python -m evals.format_probe --out docs/training/format-probe-2026-11-16.json
```

Verify on the isolated Linux CI host:

```
python -m evals.format_probe --verify --out format-probe-verification.json
```

The dedicated `Verify original format probe` workflow installs the same isolation
dependencies as the existing evaluator and uploads the verification artifact.
Unit tests cover family pairing, reference shapes, signature/empty-output controls,
no AST execution, functional-vs-shape separation and sandbox failure handling.
Reference verification is not model capability measurement. No new candidate is
accepted or deployed, and no paid compute or inference is part of this phase.

## How this constrains the next training proposal

Once the harness is verified, measure the accepted model on both formats with the
same task semantics, checkpoint, greedy decoding and generation budget. Bind raw
responses, prompt/source hashes, token counts and stop reasons. Execute responses
only in the existing sandbox; report syntax, shape, functional results and paired
task changes separately. Do not select prompts or repairs using favorable scores.

If the model fails complete functions while body continuations work, a response-
boundary/format intervention becomes a testable hypothesis. If both fail functional
cases, formatting alone is insufficient. Neither outcome justifies automatic more
training. A future proposal must freeze its data, family exclusions, initialization,
budget and success criteria before training, require functional improvement in
both formats without security-category regression, and retain the existing frozen
qualification gates. The accepted model remains unchanged throughout this phase.
