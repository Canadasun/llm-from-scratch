"""
The enterprise security pipeline wired in FRONT of the inference service.

This is the Stage-3 increment that makes `/generate` a multi-tenant,
authenticated, audited endpoint without changing the model or the inner service:

    authenticate (API key) -> authorize ("model:infer", RBAC + key scope)
    -> delegate to InferenceService (validate -> rate-limit -> use-safety ->
       generate -> use-safety) -> audit the outcome

Design rules it keeps (from the platform directive):
- **Authorization is decided in software, never by model output.** The decision
  to serve a request is made here, before the model runs.
- **Every path emits exactly one audit event** — auth failure, authz denial,
  a safety block, an internal error, or success — so the trail is complete.
- **Rate limiting is keyed by the authenticated principal**, not the socket IP:
  with real identity we can throttle a tenant's principal rather than whatever
  address it happens to connect from.
- Inference is a tenant-scoped capability, not a per-resource action, so there is
  no resource id — authorization is RBAC + key-scope only. The existing
  cross-tenant resource isolation (store + authz + gateway) is unchanged.

The checkpoint, the model, and the inner `InferenceService` are untouched; this
composes around them. Returns `(status, body)` exactly like the inner service, so
the same HTTP adapter maps it to a response.
"""

import uuid

from enterprise.authz import authorize

INFER_ACTION = "model:infer"


class SecuredInferenceService:
    # The HTTP adapter checks this to decide whether to pull a bearer token.
    requires_auth = True

    def __init__(self, inner, store, audit, authorizer=authorize,
                 action=INFER_ACTION):
        self.inner = inner
        self.store = store
        self.audit = audit
        self.authorizer = authorizer
        self.action = action

    def health(self):
        # Liveness stays open (no auth): it leaks no tenant data and is the
        # standard target for load-balancer / orchestrator probes. It does
        # advertise that the endpoint requires auth.
        info = self.inner.health()
        info["auth"] = "required"
        return info

    def handle_generate(self, payload, api_key=None, client="-", request_id=None):
        rid = request_id or uuid.uuid4().hex[:12]

        # 1. authenticate. `client` (socket origin) is recorded for the trail but
        #    is never trusted for identity — only the API key is.
        principal, key = (self.store.authenticate(api_key)
                          if api_key else (None, None))
        if principal is None:
            self.audit.record(rid, self.action, "deny", "auth_failed",
                              detail=f"client={client}")
            return 401, {"error": "unauthorized", "request_id": rid}

        tid, pid = principal.tenant_id, principal.id

        # 2. authorize the infer capability (RBAC + key-scope ceiling; no resource)
        d = self.authorizer(principal, self.action, None,
                            scopes=getattr(key, "scopes", ()))
        if not d.allow:
            self.audit.record(rid, self.action, "deny", "forbidden",
                              tenant_id=tid, principal_id=pid, detail=d.reason)
            return 403, {"error": "forbidden", "reason": d.reason,
                         "request_id": rid}

        # 3. delegate to the inner service. Rate-limit + validation + use-safety
        #    + generation all happen there; we key the limiter by principal so a
        #    tenant's identity, not its IP, is what gets throttled.
        status, body = self.inner.handle_generate(payload, client=pid)

        # 4. audit the outcome. A 200 is an allowed success; anything else
        #    (validation 400, rate-limit 429, safety block 403, internal 500) is
        #    recorded with its cause so the trail reflects what actually happened.
        # Adopt the inner service's request id as the single correlation id, so
        # the audit event, the response body and the inner's structured log line
        # all carry the SAME id (previously the wrapper audited a different id
        # than the one returned to the client, so they couldn't be matched up).
        rid = body.get("request_id", rid)
        body["request_id"] = rid
        body["tenant_id"] = tid
        if status == 200:
            self.audit.record(rid, self.action, "allow", "ok",
                              tenant_id=tid, principal_id=pid,
                              detail=f"tokens={body.get('completion_tokens')}")
        else:
            self.audit.record(rid, self.action, "deny",
                              body.get("error", "error"),
                              tenant_id=tid, principal_id=pid,
                              detail=f"status={status}")
        return status, body
