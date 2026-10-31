"""Safety package for the llm-from-scratch project.

Three layers, applied from the start rather than bolted on later:

- scrub          : remove secrets / PII from training data (data-in safety)
- content_policy : dual-use-aware check on prompts and generations (use safety)
- (secure loading of checkpoints is handled in model/IO code via weights_only)

See docs/RESPONSIBLE_USE.md and MODEL_CARD.md for the policy these enforce.
"""
