# Instruction tuning of the from-scratch checkpoint

## First measured experiment — 2026-10-31

160 CPU updates took about 351 seconds after initialization/initial validation.
Held-out response loss decreased from **2.5040 to 1.2278** (51.0%). On the
unchanged 170-task benchmark, accuracy increased from **51/170 (30.0%)** to
**57/170 (33.5%)**. MC increased from 51/136 to 55/136, generation from 0/16 to
2/16, and coding remained **0/18**. Bits/byte improved slightly from 1.7800 to
1.7744 (see the exact recorded intrinsic value in the JSON evidence).

**The candidate was not promoted.** Reasoning dropped from 7/14 to 6/14, which
fails its existing category regression floor even though overall results improved.
Eight tasks became correct and two became incorrect. The public seed suite and
lexical generation checks do not establish broader expertise or statistically
robust gains; fresh function-generation samples still contained invalid code.
The accepted baseline and original serving checkpoint are unchanged.

Local weights: `checkpoints/instruction-candidate/ckpt.pt`. Training provenance is
in [`training/instruction-candidate-2026-10-31.json`](training/instruction-candidate-2026-10-31.json),
with the [independent benchmark comparison](evaluations/instruction-candidate-2026-10-31.md)
and [raw results](evaluations/instruction-candidate-2026-10-31.json). Weight files
are local ignored artifacts, not uploaded by the source-code commit.

## Training workflow

`finetune.py` trains the project's own M3 weights on response-only instruction
examples. It does not load anyone else's pretrained weights, invoke an external
model, provision cloud compute, or update the serving endpoint.

The seed builder in `data/build_instructions.py` authors 368 small examples:
Python function templates plus developer/security explanations. It never reads
evaluation tasks. The split has 300 training examples and 68 validation examples
from disjoint exercise families, preventing a renamed instance of the same
template from crossing the split. The dataset is a small seed curriculum, not
comprehensive cybersecurity instruction data. Exact benchmark-prompt copying is
checked by a test; semantic overlap in a small public domain suite remains a
limitation, not a claim of contamination-proof evaluation.

```bash
python data/build_instructions.py
python finetune.py --ckpt checkpoints/ckpt.pt \
  --output checkpoints/instruction-candidate/ckpt.pt \
  --steps 160 --context 192 --batch-size 2 --threads 2
```

The starting inference-only checkpoint lacks optimizer state, so this starts a
new AdamW optimizer. Prompt and padding tokens are excluded from supervised
loss. Every fifth update replays a window from the first 300,000 characters of
the existing source corpus to reduce forgetting. The original checkpoint is
never overwritten. Existing candidate paths are refused; choose a new run
directory for another attempt. `--ckpt` may point to a prior candidate to continue
from its weights with a **new** optimizer; this is not an exact optimizer resume.

Validation is token-weighted response loss, measured across every held-out
example. The candidate `ckpt.pt` is saved only when validation loss improves over
the source checkpoint. `last.pt` contains the latest weights, optimizer, and RNG
state for diagnosis/future resume support; the instruction-tuning CLI does not
yet provide exact optimizer-resume support. The JSON training log records data
hashes, source checkpoint hash, hyperparameters, replay source hash, and progress.
The replay file hash covers the full source file; selection is its first 300,000
characters and random windows of the configured context length.

The loss-selected candidate must then be checked independently:

```bash
python -m evals.run --ckpt checkpoints/instruction-candidate/ckpt.pt \
  --threads 2 --task-timeout 120 --json evals/results-instruction-candidate.json
python -m evals.review --results evals/results-instruction-candidate.json \
  --baseline evals/baselines/m3.json --markdown evals/review-instruction-candidate.md
```

Linux bubblewrap/libseccomp are required for generated-code evaluation. A smaller
instruction loss alone does not establish better security judgment, code
correctness, calibrated uncertainty, or production readiness. Keep the accepted
benchmark baseline unchanged when a candidate regresses; do not lower the floor
to accommodate a training run. Neither training nor evaluation promotes a
candidate into production.

For larger improvements, the curriculum needs broader independently verified
code/answers, richer instruction variation, private held-out tasks, and more
compute. Those are future work; repeating the small seed indefinitely risks
overfitting and forgetting.
