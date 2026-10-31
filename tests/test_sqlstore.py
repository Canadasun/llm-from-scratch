"""
Tests for the persistent SQLite-backed store (enterprise/sqlstore.py).

The headline is PERSISTENCE: keys and their lifecycle survive a process restart
(modeled by closing the store and reopening the same file). Plus it must uphold
the SAME tenant-isolation invariants as the in-memory store, and be a drop-in for
the Gateway and KeyManager.
"""

import pytest

from enterprise.audit import AuditLog
from enterprise.gateway import Gateway
from enterprise.authz import ALL_ACTIONS
from enterprise.keymgmt import KeyManager
from enterprise.keys import generate_key
from enterprise.models import ApiKey, Principal, Resource, Tenant
from enterprise.sqlstore import SqlStore


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def _seed(store):
    store.add_tenant(Tenant("tA", "Acme"))
    store.add_tenant(Tenant("tB", "Globex"))
    store.add_principal(Principal("pA", "tA", ("member",)))
    store.add_principal(Principal("pB", "tB", ("admin",)))
    store.add_resource(Resource("rA", "tA", "document"))
    store.add_resource(Resource("rB", "tB", "document"))


# ---- persistence (the whole point) ----

def test_key_survives_reopen(tmp_path):
    path = str(tmp_path / "keys.db")
    plain, h = generate_key()
    s1 = SqlStore(path)
    _seed(s1)
    s1.add_api_key(ApiKey("k1", h, "tA", "pA", ("model:infer",), created_at=1000.0))
    s1.close()

    # brand new store object on the SAME file = a "restart"
    s2 = SqlStore(path)
    principal, key = s2.authenticate(plain)
    assert principal is not None and principal.id == "pA"
    assert key.scopes == ("model:infer",)       # scopes round-tripped
    s2.close()


def test_revocation_persists_across_reopen(tmp_path):
    import dataclasses
    path = str(tmp_path / "keys.db")
    plain, h = generate_key()
    s1 = SqlStore(path)
    _seed(s1)
    rec = s1.add_api_key(ApiKey("k1", h, "tA", "pA", (), created_at=1000.0))
    s1.update_api_key(dataclasses.replace(rec, revoked=True))
    s1.close()

    s2 = SqlStore(path)
    assert s2.authenticate(plain) == (None, None)   # still revoked after restart
    s2.close()


def test_expiry_uses_injected_clock(tmp_path):
    clock = Clock()
    path = str(tmp_path / "keys.db")
    plain, h = generate_key()
    s = SqlStore(path, clock=clock)
    _seed(s)
    s.add_api_key(ApiKey("k1", h, "tA", "pA", (), created_at=1000.0, expires_at=1100.0))
    assert s.authenticate(plain)[0] is not None     # before expiry
    clock.t = 1101.0
    assert s.authenticate(plain) == (None, None)     # after expiry
    s.close()


# ---- tenant isolation (same invariant as in-memory, enforced in SQL) ----

def test_tenant_isolation_in_sql(tmp_path):
    s = SqlStore(str(tmp_path / "k.db"))
    _seed(s)
    pA = s.get_principal("pA")
    assert s.get_resource(pA, "rA") is not None     # own tenant
    assert s.get_resource(pA, "rB") is None          # other tenant hidden
    assert s.get_resource(pA, "missing") is None
    assert [r.id for r in s.list_resources(pA)] == ["rA"]
    s.close()


def test_get_api_key_is_tenant_scoped(tmp_path):
    s = SqlStore(str(tmp_path / "k.db"))
    _seed(s)
    _, h = generate_key()
    s.add_api_key(ApiKey("kA", h, "tA", "pA", (), created_at=1000.0))
    pA, pB = s.get_principal("pA"), s.get_principal("pB")
    assert s.get_api_key(pA, "kA") is not None        # owner tenant
    assert s.get_api_key(pB, "kA") is None            # other tenant can't see it
    s.close()


# ---- drop-in for Gateway: the P0 cross-tenant gate still holds ----

def test_gateway_p0_gate_with_sqlstore(tmp_path):
    s = SqlStore(str(tmp_path / "k.db"))
    _seed(s)
    plain, h = generate_key()
    s.add_api_key(ApiKey("kA", h, "tA", "pA", (), created_at=1000.0))
    gw = Gateway(s, AuditLog())
    for action in sorted(ALL_ACTIONS):
        status, _ = gw.handle(plain, action, resource_id="rB")   # tenant B resource
        assert status == 404, f"cross-tenant {action} must be 404"
    s.close()


# ---- drop-in for KeyManager: issue/rotate/revoke, persisted ----

def test_keymanager_issue_then_persist(tmp_path):
    path = str(tmp_path / "k.db")
    clock = Clock()
    s1 = SqlStore(path, clock=clock)
    s1.add_tenant(Tenant("tA", "Acme"))
    s1.add_principal(Principal("admin", "tA", ("admin",)))
    s1.add_principal(Principal("user", "tA", ("member",)))
    km = KeyManager(s1, AuditLog(), clock=clock)
    status, body = km.issue(s1.get_principal("admin"), "user", scopes=("model:infer",))
    assert status == 200
    issued_key = body["api_key"]
    s1.close()

    # restart: the issued key still authenticates
    s2 = SqlStore(path, clock=clock)
    p, k = s2.authenticate(issued_key)
    assert p.id == "user" and k.scopes == ("model:infer",)
    s2.close()


def test_keymanager_rotate_persists(tmp_path):
    path = str(tmp_path / "k.db")
    clock = Clock()
    s1 = SqlStore(path, clock=clock)
    s1.add_tenant(Tenant("tA", "Acme"))
    s1.add_principal(Principal("admin", "tA", ("admin",)))
    km = KeyManager(s1, AuditLog(), clock=clock)
    _, issued = km.issue(s1.get_principal("admin"), "admin")
    _, rot = km.rotate(s1.get_principal("admin"), issued["key_id"])
    old_plain, new_plain = issued["api_key"], rot["api_key"]
    s1.close()

    s2 = SqlStore(path, clock=clock)
    assert s2.authenticate(old_plain) == (None, None)   # old retired, persisted
    assert s2.authenticate(new_plain)[0] is not None    # new works after restart
    s2.close()


# ---- URI path (nolock) for network shares (Azure Files / SMB) ----

def test_uri_path_with_nolock_opens_and_roundtrips(tmp_path):
    # A `file:...?nolock=1` URI must open, create the schema, and persist a key
    # across a reopen — this is the mode used on an Azure Files mount where SMB
    # can't provide SQLite's byte-range locks. (Here it just exercises the URI
    # path on a local file; the point is that uri=True is honoured.)
    db = tmp_path / "keys.db"
    uri = f"file:{db}?nolock=1"
    plaintext, key_hash = generate_key()

    s1 = SqlStore(uri)
    s1.add_tenant(Tenant("demo", "Demo"))
    s1.add_principal(Principal("demo-user", "demo", ("member",)))
    s1.add_api_key(ApiKey("demo-key", key_hash, "demo", "demo-user", ()))
    s1.close()

    s2 = SqlStore(uri)                       # reopen same file via URI
    principal, key = s2.authenticate(plaintext)
    assert principal is not None and principal.id == "demo-user"
    assert key.id == "demo-key"
    s2.close()


def test_plain_path_still_opens_without_uri(tmp_path):
    # A non-URI path must behave exactly as before (locking on).
    s = SqlStore(str(tmp_path / "plain.db"))
    s.add_tenant(Tenant("t", "T"))
    assert s.get_principal("nobody") is None
    s.close()
