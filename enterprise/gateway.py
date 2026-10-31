"""
The enterprise request pipeline:

    authenticate (API key) -> resolve resource within tenant -> authorize
    -> perform the action -> audit

Returns (status_code, body). Mirrors the serving layer's shape so an HTTP adapter
can map it directly. Every path — including denials and auth failures — produces
exactly one audit event. A missing resource and a cross-tenant resource return
the SAME 404, so existence never leaks across tenants.
"""

import uuid

from enterprise.authz import authorize


class Gateway:
    def __init__(self, store, audit, authorizer=authorize):
        self.store = store
        self.audit = audit
        self.authorizer = authorizer

    def handle(self, api_key, action, resource_id=None, request_id=None,
               perform=None, detail=""):
        rid = request_id or uuid.uuid4().hex[:12]

        # 1. authenticate
        principal, key = self.store.authenticate(api_key) if api_key else (None, None)
        if principal is None:
            self.audit.record(rid, action, "deny", "auth_failed", detail=detail)
            return 401, {"error": "unauthorized", "request_id": rid}

        tid, pid = principal.tenant_id, principal.id

        # 2. resolve resource within the tenant (isolation enforced by the store)
        resource = None
        if resource_id is not None:
            resource = self.store.get_resource(principal, resource_id)
            if resource is None:
                # not found OR other tenant — identical response, no existence leak
                self.audit.record(rid, action, "deny", "not_found_or_forbidden",
                                  tenant_id=tid, principal_id=pid,
                                  resource_id=resource_id, detail=detail)
                return 404, {"error": "not_found", "request_id": rid}

        # 3. authorize (RBAC + tenant + key scope)
        d = self.authorizer(principal, action, resource,
                            scopes=getattr(key, "scopes", ()))
        if not d.allow:
            self.audit.record(rid, action, "deny", "forbidden",
                              tenant_id=tid, principal_id=pid,
                              resource_id=resource_id or "-", detail=d.reason)
            return 403, {"error": "forbidden", "reason": d.reason, "request_id": rid}

        # 4. perform
        try:
            result = perform(principal, resource) if perform else None
        except Exception as e:
            self.audit.record(rid, action, "allow", "error",
                              tenant_id=tid, principal_id=pid,
                              resource_id=resource_id or "-", detail=str(e))
            return 500, {"error": "action_failed", "request_id": rid}

        # 5. audit success
        self.audit.record(rid, action, "allow", "ok",
                          tenant_id=tid, principal_id=pid,
                          resource_id=resource_id or "-", detail=detail)
        body = {"request_id": rid, "tenant_id": tid}
        if result is not None:
            body["result"] = result
        return 200, body
