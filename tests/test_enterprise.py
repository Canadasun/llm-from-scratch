"""
Tests for the enterprise security layer.

The headline is the P0 gate: ZERO cross-tenant access. A principal from tenant A
must never read, act on, or learn the existence of tenant B's resources —
checked at the store, the authorizer AND the gateway, for every action.
"""

import pytest

from enterprise.audit import AuditLog
from enterprise.authz import ALL_ACTIONS, Decision, authorize, permissions_for
from enterprise.gateway import Gateway
from enterprise.keys import generate_key, hash_key, verify_key
from enterprise.models import ApiKey, Principal, Resource, Tenant
from enterprise.store import Store


@pytest.fixture
def world():
    store = Store()
    store.add_tenant(Tenant("tA", "Acme"))
    store.add_tenant(Tenant("tB", "Globex"))
    store.add_principal(Principal("pA", "tA", ("member",)))
    store.add_principal(Principal("pB", "tB", ("admin",)))
    store.add_principal(Principal("pView", "tA", ("viewer",)))
    store.add_principal(Principal("pScoped", "tA", ("admin",)))

    keys = {}
    for pid, scopes in [("pA", ()), ("pB", ()), ("pView", ()),
                        ("pScoped", ("model:infer",))]:
        plain, h = generate_key()
        store.add_api_key(ApiKey(f"k_{pid}", h, store._principals[pid].tenant_id,
                                 pid, scopes))
        keys[pid] = plain

    store.add_resource(Resource("rA", "tA", "document"))
    store.add_resource(Resource("rB", "tB", "document"))
    audit = AuditLog()
    return store, Gateway(store, audit), audit, keys


# ---- keys ----

def test_keys_only_hash_stored_and_verify():
    plain, h = generate_key()
    assert plain.startswith("sk_") and h != plain and len(h) == 64
    assert verify_key(plain, h)
    assert not verify_key(plain + "x", h)
    assert hash_key(plain) == h


def test_authenticate(world):
    store, _, _, keys = world
    p, k = store.authenticate(keys["pA"])
    assert p.id == "pA" and k.tenant_id == "tA"
    assert store.authenticate("sk_wrong") == (None, None)


# ---- RBAC ----

def test_role_permissions():
    assert "resource:read" in permissions_for(("viewer",))
    assert "resource:write" not in permissions_for(("viewer",))
    assert "resource:delete" in permissions_for(("admin",))


def test_authorize_rbac(world):
    store, *_ = world
    viewer = store._principals["pView"]
    assert authorize(viewer, "resource:read").allow
    assert not authorize(viewer, "resource:write").allow
    assert not authorize(viewer, "bogus:action").allow


def test_key_scope_narrows_permissions(world):
    store, *_ = world
    admin = store._principals["pScoped"]       # admin role, key scoped to infer
    assert authorize(admin, "resource:write").allow                      # role allows
    assert not authorize(admin, "resource:write", scopes=("model:infer",)).allow


# ---- tenant isolation: the P0 gate ----

def test_store_never_returns_cross_tenant_resource(world):
    store, *_ = world
    pA = store._principals["pA"]
    assert store.get_resource(pA, "rA") is not None     # own tenant: ok
    assert store.get_resource(pA, "rB") is None         # other tenant: hidden
    assert store.get_resource(pA, "nonexistent") is None
    assert [r.id for r in store.list_resources(pA)] == ["rA"]


def test_authorize_blocks_cross_tenant(world):
    store, *_ = world
    pA = store._principals["pA"]
    rB = store._resources["rB"]
    assert authorize(pA, "resource:read", rB).reason == "cross-tenant access denied"


def test_gateway_denies_all_cross_tenant_access(world):
    store, gw, audit, keys = world
    # tenant A principal attempts every action against tenant B's resource
    for action in sorted(ALL_ACTIONS):
        status, body = gw.handle(keys["pA"], action, resource_id="rB")
        assert status != 200, f"cross-tenant {action} must not succeed"
        assert status == 404      # identical to a nonexistent id — no existence leak
    # and it is indistinguishable from a resource that does not exist at all
    s_missing, _ = gw.handle(keys["pA"], "resource:read", resource_id="does-not-exist")
    assert s_missing == 404
    # every one of those attempts was audited as a denial
    assert all(e.decision == "deny" for e in audit.events(tenant_id="tA"))
    assert len(audit.events(decision="deny")) >= len(ALL_ACTIONS)


# ---- gateway pipeline ----

def test_gateway_auth_failure(world):
    _, gw, audit, _ = world
    status, body = gw.handle("sk_bad", "model:infer")
    assert status == 401
    assert audit.events(decision="deny")[-1].outcome == "auth_failed"


def test_gateway_forbidden_same_tenant(world):
    store, gw, audit, keys = world
    # viewer may read but not write its own tenant's resource
    status, body = gw.handle(keys["pView"], "resource:write", resource_id="rA")
    assert status == 403 and "lacks" in body["reason"]


def test_gateway_success_runs_action_and_audits(world):
    store, gw, audit, keys = world
    status, body = gw.handle(keys["pA"], "resource:read", resource_id="rA",
                             perform=lambda p, r: {"id": r.id})
    assert status == 200
    assert body["result"] == {"id": "rA"} and body["tenant_id"] == "tA"
    assert audit.events(tenant_id="tA", decision="allow")[-1].outcome == "ok"


# ---- audit redaction ----

def test_audit_redacts_secrets_in_detail():
    audit = AuditLog()
    ev = audit.record("rid1", "model:infer", "allow", "ok",
                      detail="leaked AKIAIOSFODNN7EXAMPLE in a log line")
    assert "AKIAIOSFODNN7EXAMPLE" not in ev.detail      # scrubbed
    assert "[REDACTED_AWS_KEY]" in ev.detail            # redaction marker remains


def test_audit_filtering(world):
    store, gw, audit, keys = world
    gw.handle(keys["pA"], "resource:read", resource_id="rA")   # allow
    gw.handle(keys["pA"], "resource:read", resource_id="rB")   # deny (cross-tenant)
    assert len(audit.events(tenant_id="tA")) == 2
    assert len(audit.events(tenant_id="tA", decision="deny")) == 1


def test_audit_log_is_bounded_but_sink_sees_all():
    from enterprise.audit import AuditLog
    seen = []
    log = AuditLog(sink=seen.append, max_events=3)
    for i in range(10):
        log.record(f"r{i}", "model:infer", "allow", "ok")
    assert len(log.events()) == 3        # bounded ring buffer
    assert len(seen) == 10               # durable sink still receives every event


def test_secured_response_id_matches_audit_id():
    from enterprise.store import Store
    from enterprise.audit import AuditLog
    from enterprise.models import Tenant, Principal, ApiKey
    from enterprise.keys import generate_key
    from serve.secured import SecuredInferenceService

    store = Store()
    store.add_tenant(Tenant("t", "T"))
    store.add_principal(Principal("p", "t", ("member",)))
    plain, h = generate_key()
    store.add_api_key(ApiKey("k", h, "t", "p", ()))
    audit = AuditLog()

    class FakeInner:
        def handle_generate(self, payload, client="-"):
            return 200, {"completion": "x", "completion_tokens": 3,
                         "request_id": "INNER_ID"}
        def health(self):
            return {"status": "ok"}

    svc = SecuredInferenceService(FakeInner(), store, audit)
    status, body = svc.handle_generate({"prompt": "hi"}, api_key=plain)
    assert status == 200
    ev = audit.events()[-1]
    assert body["request_id"] == ev.request_id   # response correlatable to audit
