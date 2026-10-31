"""
Inference service: the transport-agnostic request pipeline.

    validate  ->  rate-limit  ->  use-safety (prompt)  ->  generate
                                               ->  use-safety (output)

Returns (status_code, body_dict). An HTTP adapter just maps this to a response.
Every request gets a request id, a latency measurement and one structured log
line. Nothing here depends on a web framework, so it is fully unit-testable.
"""

import json
import logging
import time
import uuid

from safety import content_policy
from serve.validation import ValidationError, validate_generate


class InferenceService:
    def __init__(self, engine, rate_limiter=None, safety=True, logger=None):
        self.engine = engine
        self.rate_limiter = rate_limiter
        self.safety = safety
        self.log = logger or logging.getLogger("serve")

    def health(self):
        return {
            "status": "ok",
            "block_size": self.engine.block_size,
            "vocab_size": self.engine.vocab_size,
            "tokenizer": getattr(self.engine.tokenizer, "kind", "?"),
            "safety": self.safety,
        }

    def handle_generate(self, payload, client="-"):
        rid = uuid.uuid4().hex[:12]
        t0 = time.monotonic()

        def done(status, body):
            body["request_id"] = rid
            dt_ms = round((time.monotonic() - t0) * 1000, 1)
            body.setdefault("latency_ms", dt_ms)
            self._log(rid, client, status, dt_ms, body)
            return status, body

        # 1. Validate untrusted input.
        try:
            params = validate_generate(payload)
        except ValidationError as e:
            return done(400, {"error": "invalid_request", "detail": str(e)})

        # 2. Rate limit (after cheap validation, before expensive generation).
        if self.rate_limiter is not None and not self.rate_limiter.allow(client):
            return done(429, {"error": "rate_limited",
                              "detail": "too many requests; slow down"})

        # 3. Use-safety on the prompt.
        caution = None
        if self.safety:
            d = content_policy.check(params["prompt"])
            if d.action == "block":
                return done(403, {"error": "blocked_prompt",
                                  "detail": d.reason, "category": d.category})
            if d.action == "flag":
                caution = content_policy.caution_text(d)

        # 4. Generate.
        try:
            result = self.engine.generate(
                prompt=params["prompt"],
                max_new_tokens=params["max_new_tokens"],
                temperature=params["temperature"],
                top_k=params["top_k"],
                seed=params["seed"])
        except Exception as e:  # keep internals off the wire
            self.log.exception("generation failed rid=%s", rid)
            return done(500, {"error": "generation_failed"})

        # 5. Use-safety on the output.
        if self.safety:
            d = content_policy.check(result["completion"])
            if d.action == "block":
                return done(403, {"error": "withheld_output",
                                  "detail": d.reason, "category": d.category})
            if d.action == "flag" and caution is None:
                caution = content_policy.caution_text(d)

        body = {
            "completion": result["completion"],
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "params": {k: params[k] for k in
                       ("max_new_tokens", "temperature", "top_k", "seed")},
        }
        if caution:
            body["caution"] = caution
        return done(200, body)

    def _log(self, rid, client, status, dt_ms, body):
        rec = {"rid": rid, "client": client, "status": status, "ms": dt_ms}
        if "error" in body:
            rec["error"] = body["error"]
        if status == 200:
            rec["completion_tokens"] = body.get("completion_tokens")
        self.log.info(json.dumps(rec))
