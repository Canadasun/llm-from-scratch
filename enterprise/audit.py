"""
Structured, redacted audit log.

Every authorization decision and action becomes one immutable event. Free-text
fields are passed through the corpus secret/PII scrubber before storage, so an
audit trail never becomes a place secrets leak. Events are queryable (e.g. all
denials for a tenant) to support investigation.
"""

import json
import time
from collections import deque
from dataclasses import asdict, dataclass, field

from safety.scrub import scrub


@dataclass(frozen=True)
class AuditEvent:
    ts: float
    request_id: str
    tenant_id: str
    principal_id: str
    action: str
    resource_id: str
    decision: str          # "allow" | "deny"
    outcome: str           # "ok" | "error" | reason code
    detail: str = ""

    def to_dict(self):
        return asdict(self)


class AuditLog:
    def __init__(self, sink=None, redact=True, max_events=10000):
        """sink: optional callable(dict) for an external, durable log line (e.g.
        a file/stdout writer) — the in-memory buffer is process-local and is lost
        on restart, so a real deployment should pass a sink. redact: scrub
        secrets/PII from `detail` before storing. max_events: the in-memory
        buffer is a bounded ring (oldest dropped) so a long-running or
        auth-failure-flooded process cannot grow memory without bound; the sink,
        if set, still receives every event."""
        self._events = deque(maxlen=max_events)
        self._sink = sink
        self._redact = redact

    def record(self, request_id, action, decision, outcome,
               tenant_id="-", principal_id="-", resource_id="-", detail=""):
        if self._redact and detail:
            detail = scrub(detail, pii=True)[0]   # scrub returns (text, counts)
        ev = AuditEvent(ts=time.time(), request_id=request_id,
                        tenant_id=tenant_id or "-", principal_id=principal_id or "-",
                        action=action, resource_id=resource_id or "-",
                        decision=decision, outcome=outcome, detail=detail)
        self._events.append(ev)
        if self._sink:
            self._sink(ev.to_dict())
        return ev

    def events(self, tenant_id=None, decision=None):
        out = self._events
        if tenant_id is not None:
            out = [e for e in out if e.tenant_id == tenant_id]
        if decision is not None:
            out = [e for e in out if e.decision == decision]
        return list(out)

    def as_jsonl(self):
        return "\n".join(json.dumps(e.to_dict()) for e in self._events)
