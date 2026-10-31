"""
Tests for authorized API-key issuance, rotation and revocation
(enterprise/keymgmt.py).

Invariants under test:
- only an admin (or a key scoped to apikey:manage) can manage keys; a member -> 403;
- management is tenant-scoped: an admin cannot mint/rotate/revoke for another
  tenant's principal or key (indistinguishable from not existing -> 404);
- issued scopes can only NARROW, never exceed, the grantor's own authority;
- a revoked key stops authenticating immediately; an expired key stops at expiry;
- rotation issues a new working key and retires the old one (immediately, or
  after a grace window during which the old key still works), linking lineage;
- every operation emits exactly one audit event and never records the secret.
"""

import pytest

from enterprise.audit import AuditLog
from enterprise.keymgmt import KeyManager
from enterprise.keys import generate_key
from enterprise.models import ApiKey, Principal, Tenant
from enterprise.store import Store


class Clock:
    """A movable clock so expiry/grace are deterministic."""
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def world():
    clock = Clock()
    store = Store(clock=clock)
    store.add_tenant(Tenant("tA", "Acme"))
    store.add_tenant(Tenant("tB", "Globex"))
    store.add_principal(Principal("adminA", "tA", ("admin",)))
    store.add_principal(Principal("memberA", "tA", ("member",)))
    store.add_principal(Principal("adminB", "tB", ("admin",)))

    # a bootstrap key for each actor so they can be "the actor" (role drives authz)
    keys = {}
    for pid in ("adminA", "memberA", "adminB"):
        plain, h = generate_key()
        store.add_api_key(ApiKey(f"boot_{pid}", h,
                                 store._principals[pid].tenant_id, pid, (),
                                 created_at=clock.t))
        keys[pid] = plain

    audit = AuditLog()
    km = KeyManager(store, audit, clock=clock)
    return store, km, audit, clock, keys


def P(store, pid):
    return store._principals[pid]


# ---------- issuance ----------

def test_admin_issues_working_key(world):
    store, km, audit, clock, _ = world
    status, body = km.issue(P(store, "adminA"), "memberA", scopes=("model:infer",))
    assert status == 200 and body["api_key"].startswith("sk_")
    # the freshly minted key authenticates as the target principal
    principal, key = store.authenticate(body["api_key"])
    assert principal.id == "memberA" and key.scopes == ("model:infer",)
    assert audit.events(decision="allow")[-1].outcome == "issued"


def test_member_cannot_issue(world):
    store, km, audit, _, _ = world
    status, body = km.issue(P(store, "memberA"), "memberA")
    assert status == 403
    assert audit.events(decision="deny")[-1].outcome == "forbidden"


def test_cannot_issue_for_other_tenant_principal(world):
    store, km, audit, _, _ = world
    # admin of A tries to mint a key for a principal in tenant B
    status, body = km.issue(P(store, "adminA"), target_principal_id="adminB")
    assert status == 404                       # no existence leak across tenants
    assert audit.events(decision="deny")[-1].outcome == "not_found_or_forbidden"


def test_issued_scope_cannot_exceed_grantor(world):
    store, km, audit, _, _ = world
    # admin acting with a key scoped to {manage, infer} may not grant
    # resource:write — the grant can only narrow the actor's own authority
    status, body = km.issue(P(store, "adminA"), "memberA",
                            scopes=("resource:write",),
                            actor_scopes=("apikey:manage", "model:infer"))
    assert status == 403 and "beyond your own" in body["reason"]
    assert audit.events(decision="deny")[-1].outcome == "scope_exceeds_grantor"


def test_issue_with_ttl_expires(world):
    store, km, _, clock, _ = world
    status, body = km.issue(P(store, "adminA"), "memberA", ttl_seconds=100)
    key_plain = body["api_key"]
    assert store.authenticate(key_plain)[0] is not None      # valid now
    clock.t += 101                                           # past expiry
    assert store.authenticate(key_plain) == (None, None)     # expired -> rejected


# ---------- revocation ----------

def test_revoke_stops_authentication(world):
    store, km, audit, _, _ = world
    _, body = km.issue(P(store, "adminA"), "memberA")
    plain, key_id = body["api_key"], body["key_id"]
    assert store.authenticate(plain)[0] is not None
    status, _ = km.revoke(P(store, "adminA"), key_id)
    assert status == 200
    assert store.authenticate(plain) == (None, None)         # revoked -> rejected
    assert audit.events(decision="allow")[-1].outcome == "revoked"


def test_cannot_revoke_other_tenant_key(world):
    store, km, audit, _, _ = world
    # a key that lives in tenant A
    _, body = km.issue(P(store, "adminA"), "memberA")
    key_id = body["key_id"]
    # admin of B tries to revoke it -> 404, and A's key still works
    status, _ = km.revoke(P(store, "adminB"), key_id)
    assert status == 404
    assert store.authenticate(body["api_key"])[0] is not None


# ---------- rotation ----------

def test_rotate_immediate_retires_old_key(world):
    store, km, audit, _, _ = world
    _, issued = km.issue(P(store, "adminA"), "memberA", scopes=("model:infer",))
    old_plain, old_id = issued["api_key"], issued["key_id"]

    status, body = km.rotate(P(store, "adminA"), old_id)
    assert status == 200
    new_plain = body["api_key"]
    # old key dead immediately, new key works and carries the same scopes
    assert store.authenticate(old_plain) == (None, None)
    p, k = store.authenticate(new_plain)
    assert p.id == "memberA" and k.scopes == ("model:infer",)
    # lineage recorded on the retired key
    old_rec = store.get_api_key(P(store, "adminA"), old_id)
    assert old_rec.revoked and old_rec.replaced_by == body["key_id"]
    assert audit.events(decision="allow")[-1].outcome == "rotated"


def test_rotate_with_grace_keeps_old_key_until_expiry(world):
    store, km, audit, clock, _ = world
    _, issued = km.issue(P(store, "adminA"), "memberA")
    old_plain, old_id = issued["api_key"], issued["key_id"]

    status, body = km.rotate(P(store, "adminA"), old_id, grace_seconds=50)
    assert status == 200
    new_plain = body["api_key"]
    # during the grace window BOTH keys authenticate (no outage on swap)
    assert store.authenticate(old_plain)[0] is not None
    assert store.authenticate(new_plain)[0] is not None
    assert audit.events(decision="allow")[-1].outcome == "rotated_grace"
    # after the grace window the old key is gone, the new one remains
    clock.t += 51
    assert store.authenticate(old_plain) == (None, None)
    assert store.authenticate(new_plain)[0] is not None


def test_cannot_rotate_other_tenant_key(world):
    store, km, _, _, _ = world
    _, issued = km.issue(P(store, "adminA"), "memberA")
    status, _ = km.rotate(P(store, "adminB"), issued["key_id"])
    assert status == 404


# ---------- audit hygiene ----------

def test_audit_never_records_the_secret(world):
    store, km, audit, _, _ = world
    _, body = km.issue(P(store, "adminA"), "memberA")
    plaintext = body["api_key"]
    assert all(plaintext not in e.detail for e in audit.events())

def test_one_audit_event_per_operation(world):
    store, km, audit, _, _ = world
    before = len(audit.events())
    _, body = km.issue(P(store, "adminA"), "memberA")       # +1
    km.rotate(P(store, "adminA"), body["key_id"])           # +1
    km.revoke(P(store, "memberA"), body["key_id"])          # +1 (denied, still audited)
    assert len(audit.events()) == before + 3
