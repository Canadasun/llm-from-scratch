# Syntax and instruction-format diagnosis

PR #33 is merged. This follows its rejected sampler candidates using the six
already committed generation files. No model inference, GPU training, generated
code execution, benchmark answers, private holdouts or score changes are needed.

## Reproducible offline analysis

```
python -m evals.syntax_diagnosis \
  --programs-dir evals/results/sampler_control \
  --out docs/training/syntax-diagnosis-2026-11-16.json
```

The [raw report](syntax-diagnosis-2026-11-16.json) binds all six input hashes and
records task-level syntax categories, program hashes, extraction diagnostics and
paired transitions. It contains no generated source or benchmark solutions.
`ast.parse` inspects syntax only; it never executes a program. Five tests cover
that boundary, input pairing, extraction mismatch, fences and the possibility
that stop sequences cut a valid multiline string.

## What changed

| Model / mode | Valid | Indentation | Unclosed construct | Other syntax | Empty among valid |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base completion | 35 | 1 | 10 | 6 | 9 |
| Prefix control completion | 33 | 1 | 6 | 12 | 10 |
| Uniform treatment completion | 27 | 9 | 7 | 9 | 8 |
| Base chat | 10 | 0 | 23 | 19 | 7 |
| Prefix control chat | 8 | 1 | 22 | 21 | 4 |
| Uniform treatment chat | 10 | 2 | 19 | 21 | 9 |

Each row has 52 unique programs, including the existing fixture tasks. Plus-suite
duplicates are not counted again. Empty modules pass Python parsing, so parse
rates cannot substitute for functional execution or entry-point presence. Nine
of the treatment's ten valid chat programs have no functions and are empty;
the remaining one contains a function. Neither fact establishes correctness.

Treatment completion loses 13 previously parseable programs and gains five.
The 13 losses comprise five indentation failures, five other syntax failures
and three unclosed constructs. HumanEval-format valid outputs move from 20/27 to
19/27; MBPP-format valid outputs move from 15/25 to 8/25. This localizes most net
syntax loss to whole-program responses on this observed subset. It does not prove
a causal format effect; suites also differ in task content.

Across all six files, zero saved programs are invalid when the corresponding
uncut response is parseable. For HumanEval completion the original prompt is
reconstructed from the saved program and exact existing stop extraction; for
chat/MBPP the uncut text may include prose or fences. This detects a particular
extraction failure, not all possible extraction problems. Do not change stop
rules to improve this observed screen. Unclosed strings/brackets are common,
but saved files lack actual generated token counts and terminal stop reasons;
calling these failures proven token-budget exhaustion would be unjustified.

## Training-format evidence and next experiment requirements

The prior curriculum's **training file only** has 240 rows: 144 implementation,
48 repair and 48 preserve-correct. All 240 responses start with raw `def ` and
none with a Markdown fence. Its SHA256 is
`0d560c7977ce6899636790831964273a0d86684a89096d15f3e45eab2e543a03`.
`training.sft.encode_row` supplies user/assistant/EOT tokens around those rows;
their absence from literal prompt strings is expected. Base pretraining has no
user/assistant occurrences, as recorded in the earlier base diagnosis. The two
sampler arms did base training, not instruction tuning, and cannot establish that
an instruction protocol has been learned. Earlier small SFT trials also failed.

The next testable hypothesis is that response-boundary and complete-function
supervision need explicit measurement separately from body-continuation syntax.
Before any further GPU run:

1. Build an original, repository/family-separated format probe with paired
   complete-module and indented-body requests, plus empty-output and wrong-entry
   controls. Use authored task semantics, not these benchmark prompts or answers.
   Require actual function/entry-point presence in addition to parsing and execute
   correctness cases only in the existing isolated Linux harness.
2. Audit exposure counts and response lengths for each format in training only.
   Preserve family splits and contamination checks. A formatting probe is not
   independent qualification of general coding or cybersecurity capability.
3. Freeze the hypothesis, training/evaluation data hashes, matched initialization,
   bounded budget and functional/security retention criteria before a new trial.
   Record generated token counts and stop reasons to distinguish censoring from
   malformed code. Keep greedy decoding and observed benchmark scores unchanged.

This diagnosis authorizes no extra steps on the rejected candidates. Accepted
weights, production/preview gates and private holdouts remain unchanged. No paid
compute is involved. A useful assistant still requires demonstrated functional
and security improvements, not merely better loss or more parseable outputs.
