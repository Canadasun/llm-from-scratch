"""
Milestone 4 — inference serving.

A small, production-minded serving layer for the from-scratch model. The design
keeps the *decision logic* framework-agnostic and fully unit-testable, with the
HTTP transport as a thin shell on top:

    engine.py      load a checkpoint once; turn a prompt into generated text
    validation.py  validate + bound untrusted request JSON (caps, types, ranges)
    ratelimit.py   per-client token-bucket rate limiting (in-process)
    service.py     orchestration: validate -> rate-limit -> safety -> generate
    http_server.py stdlib HTTP adapter (no heavyweight dependency) over service

Security posture (see also MODEL_CARD.md and SECURITY.md):
  - Every request field is untrusted input: validated and hard-bounded before use.
  - The use-safety policy (safety/content_policy.py) gates both prompt and output.
  - The model never executes anything; it only returns text.
  - Checkpoints load with weights_only=True (no pickle code execution).
"""
