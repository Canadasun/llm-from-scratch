# Sampler control results

Training ran November 15 UTC; sandbox scoring and reporting followed November 16.
The [protocol](sampler-control-2026-11-14.md) was committed before training.
The private free-Kaggle run was
`idowuaye/llmfs-prefix-uniform-pilot-20261114`, version 1, using source commit
`5be0f45bf81b7052c8ea213040583eb0dcb46198`. No paid compute was used.

## Controlled learning result

Both arms completed 200/200 optimizer updates without skipped steps, processing
6,553,600 target tokens each. Initial tensors, CPU/device/batch RNG, scaler and
fresh optimizer state matched. Only the sampling mode differed. The core run took
1,404.5 seconds; conditional generation took 2,086.6 seconds. The CUDA uniform
replay resume check also passed, with all 29 tensors and RNG state identical.

Mean validation loss, nats per target token:

| Stratum | Windows | Accepted base | Prefix control | Uniform treatment |
| --- | ---: | ---: | ---: | ---: |
| Code prefix | 128 | 1.571483 | 1.552291 | 1.551259 |
| Code document body | 128 | 1.590121 | 1.570149 | 1.531643 |
| Prose prefix | 32 | 1.389507 | 1.382837 | 1.320361 |
| Prose document body | 32 | 3.031514 | 3.005891 | 2.946072 |

The predeclared primary criterion passed. Treatment code-body loss improved by
0.038505 nats versus control (paired 95% bootstrap interval for treatment minus
control: [-0.041908, -0.035107]) and 0.058478 versus unchanged base
([-0.062209, -0.054593]). Both exceed the required 0.01 nats. Prose-body loss
improved by 0.085443 nats versus base, passing the retention criterion.

This supports repairing document-body exposure. It is one training seed on a
deliberately long-document diagnostic, not a whole-corpus estimate. Bootstrap
intervals describe repository-window variation, not training-seed variation.

## Capability result and decision

All four new Linux sandbox scoring runs succeeded. HumanEval+ and MBPP+ counts
equal their respective original-suite counts in every row below. The plus suites
share prompts; do not count them as independent functional tasks. Syntax uses
52 unique generated programs per mode (including the existing fixture tasks).

| Model / mode | HumanEval | MBPP | Parses as Python | Abstention | Injection | Safe recommendation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base / completion | 1/25 | 0/25 | 35/52 | 5/15 | 5/15 | 4/15 |
| Control / completion | 1/25 | 0/25 | 33/52 | 5/15 | 4/15 | 4/15 |
| Treatment / completion | 1/25 | 0/25 | 27/52 | 5/15 | 3/15 | 5/15 |
| Base / chat | 0/25 | 0/25 | 10/52 | 5/15 | 5/15 | 4/15 |
| Control / chat | 0/25 | 0/25 | 8/52 | 5/15 | 4/15 | 4/15 |
| Treatment / chat | 0/25 | 0/25 | 10/52 | 5/15 | 3/15 | 5/15 |

Neither arm improves functional coding. The treatment loses eight syntactically
valid completion programs and two prompt-injection answers versus base. One
additional safe-recommendation answer does not offset those regressions.
These small observed samples do not establish population-level security effects.

Applying the existing matched-mode pilot capability checks rejects both candidates:
[control decision](../../evals/results/sampler_control/decision-control.json),
[treatment decision](../../evals/results/sampler_control/decision-treatment.json).
Those decisions are separate from the successful predeclared language-model-loss
criterion. No full qualification, further training, checkpoint promotion or
deployment follows automatically. The bounded trial stops here.

Next research phase: investigate completion syntax failures and secure instruction
transfer offline, retaining all observed-screen limitations. Inspect error classes
and corpus format/provenance without copying benchmark answers into training.
Predeclare a new data/format hypothesis, independent family-level evaluation and
a separate bounded budget before another GPU trial. The sampler repair remains
appropriate for new experiments, but lower loss alone does not justify extending
this candidate or claiming that it is a smarter coding/security assistant.

## Evidence and artifact binding

The [raw experiment](../../evals/results/sampler_control/experiment.json) records
all 320 per-window losses for all three models, recipes, initial hashes, update
counts, checkpoint hashes and the assessment. The assessment was recomputed
locally and matched. Generated-output metadata matched the respective recorded
checkpoint hashes, and the existing secret-pattern scan found no matches.

The six generated program files are committed alongside that report. Generated
code is executed only by the existing Linux bubblewrap/libseccomp workflow.
The regenerated base programs, behavior vectors and all inference metadata except
generation time exactly match the earlier verified-seed screen, allowing reuse
of its raw sandbox scores. [Reuse provenance](../../evals/results/sampler_control/base-score-reuse.json)
records source/current hashes and the original scoring workflow runs.
The [new sandbox-run manifest](../../evals/results/sampler_control/sandbox-runs.json)
binds four workflow runs, raw score hashes and generated-input hashes. Raw scores
and exact task-level results are committed in that same directory.

Validation: 66 targeted local tests pass. The implementation's Linux CI passed
625 tests with one skipped; the image scan and remaining required checks passed.

Accepted base SHA256:
`1164213e635d376946ec331d5ff238cf89f492788554a953010f59bd5f57a593`.
Control weights SHA256:
`1b7e69a3bcec96683be1ec4550bad458fb428219115677709086bb5a1f9a8bcc`.
Treatment weights SHA256:
`b9e020f8b35ee2928864ec9270b2a17c49c3e0f2ab9530a115f333aad6bcb7b7`.
Trial checkpoints remain separate weights-only artifacts; their optimizer state
cannot be resumed exactly. The accepted model remains unchanged.
