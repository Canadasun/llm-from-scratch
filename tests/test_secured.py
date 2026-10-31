"""
Tests for the enterprise security pipeline wired in front of the serving layer
(serve/secured.py). This is the Stage-3 increment that makes /generate
authenticated, tenant-scoped and audited.

The invariants under test:
- no API key / bad key  -> 401, audited as an auth failure (model never runs);
- authenticated but role/scope forbids infer -> 403, audited;
- authenticated + authorized -> the inner service runs and the success is audited
  with the tenant attached;
- rate limiting is keyed by the authenticated principal, not the socket IP;
- a safety block from the inner service is surfaced AND audited (one event per
  request on every path);
- health stays open and advertises that auth is required.
"""

import pytest
import torch

from enterprise.audit import AuditLog
from enterprise.keys import generate_key
from enterprise.models import ApiKey, Principal, Tenant
from enterprise.store import Store
from model import GPT, GPTConfig
from tokenizer import CharTokenizer
from serve.engine import InferenceEngine
from serve.ratelimit import RateLimiter
from serve.secured import SecuredInferenceService
from serve.service import InferenceService


@pytest.fixture(scope="module")
def engine():
    text = ("zero trust architecture verifies every request. tls protects data "
            "in transit. least privilege limits access.\n")
    tok = CharTokenizer.from_text(text)
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=32,
                    n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    return InferenceEngine(GPT(cfg), tok, device="cpu")


@pytest.fixture
def secured(engine):
    """A secured service over two tenants, plus the plaintext keys to use."""
    store = Store()
    store.add_tenant(Tenant("tA", "Acme"))
    store.add_tenant(Tenant("tB", "Globex"))
    store.add_principal(Principal("pMember", "tA", ("member",)))
    store.add_principal(Principal("pScoped", "tB", ("member",)))

    keys = {}
    for pid, scopes in [("pMember", ()), ("pScoped", ("resource:read",))]:
        plain, h = generate_key()
        store.add_api_key(ApiKey(f"k_{pid}", h,
                                 store._principals[pid].tenant_id, pid, scopes))
        keys[pid] = plain

    audit = AuditLog()
    inner = InferenceService(engine, rate_limiter=None, safety=True)
    svc = SecuredInferenceService(inner, store, audit)
    return svc, audit, keys


GOOD = {"prompt": "zero trust ", "max_new_tokens": 8, "seed": 1}


# ---------- authentication ----------

def test_missing_key_is_401_and_audited(secured):
    svc, audit, _ = secured
    status, body = svc.handle_generate(GOOD, api_key=None)
    assert status == 401 and body["error"] == "unauthorized"
    assert "completion" not in body                 # model never ran
    evs = audit.events(decision="deny")
    assert evs and evs[-1].outcome == "auth_failed"


def test_bad_key_is_401(secured):
    svc, audit, _ = secured
    status, _ = svc.handle_generate(GOOD, api_key="sk_not_a_real_key")
    assert status == 401
    assert audit.events(decision="deny")[-1].outcome == "auth_failed"


# ---------- authorization ----------

def test_member_can_infer_and_success_is_audited(secured):
    svc, audit, keys = secured
    status, body = svc.handle_generate(GOOD, api_key=keys["pMember"])
    assert status == 200
    assert "completion" in body and body["tenant_id"] == "tA"
    ok = audit.events(tenant_id="tA", decision="allow")
    assert ok and ok[-1].outcome == "ok" and ok[-1].action == "model:infer"


def test_key_scope_without_infer_is_403_and_audited(secured):
    # pScoped's role (member) allows model:infer, but its key is scoped to
    # resource:read only -> the key narrows away the infer capability.
    svc, audit, keys = secured
    status, body = svc.handle_generate(GOOD, api_key=keys["pScoped"])
    assert status == 403 and "scope" in body["reason"]
    assert "completion" not in body
    assert audit.events(tenant_id="tB", decision="deny")[-1].outcome == "forbidden"


# ---------- rate limiting is per-principal ----------

def test_rate_limit_is_keyed_by_principal(engine):
    store = Store()
    store.add_tenant(Tenant("tA", "Acme"))
    store.add_principal(Principal("p1", "tA", ("member",)))
    store.add_principal(Principal("p2", "tA", ("member",)))
    keys = {}
    for pid in ("p1", "p2"):
        plain, h = generate_key()
        store.add_api_key(ApiKey(f"k_{pid}", h, "tA", pid, ()))
        keys[pid] = plain

    # capacity 1, no refill within the test -> each principal gets exactly one.
    limiter = RateLimiter(capacity=1, refill_per_sec=1.0, clock=lambda: 1000.0)
    inner = InferenceService(engine, rate_limiter=limiter, safety=False)
    svc = SecuredInferenceService(inner, store, AuditLog())

    assert svc.handle_generate(GOOD, api_key=keys["p1"])[0] == 200
    assert svc.handle_generate(GOOD, api_key=keys["p1"])[0] == 429   # p1 exhausted
    assert svc.handle_generate(GOOD, api_key=keys["p2"])[0] == 200   # p2 unaffected


# ---------- safety block is surfaced and audited ----------

def test_safety_block_is_surfaced_and_audited(secured, monkeypatch):
    from safety import content_policy
    svc, audit, keys = secured

    class _D:
        action, reason, category = "block", "disallowed", "test"
    monkeypatch.setattr(content_policy, "check", lambda text: _D())

    status, body = svc.handle_generate(GOOD, api_key=keys["pMember"])
    assert status == 403 and body["error"] == "blocked_prompt"
    # the request was authenticated+authorized, so it is a tenant event, audited
    last = audit.events(tenant_id="tA")[-1]
    assert last.decision == "deny" and last.outcome == "blocked_prompt"


# ---------- health ----------

def test_health_is_open_and_advertises_auth(secured):
    svc, _, _ = secured
    h = svc.health()
    assert h["status"] == "ok" and h["auth"] == "required"


# ---------- one audit event per request ----------

def test_exactly_one_audit_event_per_request(secured):
    svc, audit, keys = secured
    svc.handle_generate(GOOD, api_key=keys["pMember"])      # 200
    svc.handle_generate(GOOD, api_key=None)                 # 401
    svc.handle_generate(GOOD, api_key=keys["pScoped"])      # 403
    assert len(audit.events()) == 3
