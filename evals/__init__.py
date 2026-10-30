"""
Evaluation suite for the from-scratch LLM.

This is the "turn 'looks good' into 'measured'" layer. It does three things a
training-loss number cannot:

1. Intrinsic metrics (perplexity, bits-per-byte, next-token accuracy) on a
   held-out set the model never trained on, compared against a random-init
   baseline so improvement is *proven*, not assumed.
2. Capability tasks — multiple-choice (scored by likelihood, which works on a
   base model), free-generation (matched by exact/contains/regex), and code
   (generated, then unit-tested in a sandboxed subprocess).
3. An honest report: per-category accuracy, sample sizes, the random-chance
   baseline, failure examples, and explicit limitations.

Design rules (from the project's enterprise direction):
- Never claim readiness from loss alone.
- Report sample sizes and limitations; show failures.
- Execute code in isolated, resource-limited environments.
- Keep held-out eval data out of the training corpus (contamination).
"""
