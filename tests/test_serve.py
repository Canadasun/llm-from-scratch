"""
Tests for the Milestone-4 serving layer.

These use a tiny, randomly-initialised in-memory model + char tokenizer, so they
run fast on CPU and need no trained checkpoint. They exercise the parts that
matter for correctness and safety: input validation/bounds, rate limiting, the
service pipeline (incl. safety blocking and cautions), and the engine contract.
"""

import itertools

import pytest
import torch

from model import GPT, GPTConfig
from tokenizer import CharTokenizer
from serve.engine import InferenceEngine
from serve.ratelimit import RateLimiter
from serve.service import InferenceService
from serve.validation import (MAX_NEW_TOKENS, MAX_PROMPT_CHARS, ValidationError,
                              validate_generate)


# ---------- fixtures ----------

@pytest.fixture(scope="module")
def engine():
    text = ("zero trust architecture verifies every request. tls protects data "
            "in transit. least privilege limits access. incident response and "
            "recovery restore operations.\n")
    tok = CharTokenizer.from_text(text)
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=32,
                    n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    model = GPT(cfg)
    return InferenceEngine(model, tok, device="cpu")


@pytest.fixture
def service(engine):
    return InferenceService(engine, rate_limiter=None, safety=True)


# ---------- validation ----------

def test_validate_defaults_fill_in():
    p = validate_generate({"prompt": "hello"})
    assert p["prompt"] == "hello"
    assert p["max_new_tokens"] == 128
    assert 0 < p["temperature"] <= 2.0
    assert p["top_k"] is None and p["seed"] is None


@pytest.mark.parametrize("payload", [
    [],                                        # not an object
    {"prompt": 123},                           # wrong type
    {"prompt": "x", "max_new_tokens": 0},      # below min
    {"prompt": "x", "max_new_tokens": MAX_NEW_TOKENS + 1},   # above cap
    {"prompt": "x", "max_new_tokens": 1.5},    # not an int
    {"prompt": "x", "max_new_tokens": True},   # bool is not an int
    {"prompt": "x", "temperature": 0},         # below min
    {"prompt": "x", "temperature": 9},         # above max
    {"prompt": "x", "temperature": float("nan")},
    {"prompt": "x", "top_k": 0},               # below min
    {"prompt": "x", "seed": -1},               # negative
    {"prompt": "x", "bogus": 1},               # unknown field
    {"prompt": "a" * (MAX_PROMPT_CHARS + 1)},  # too long
])
def test_validate_rejects_bad_input(payload):
    with pytest.raises(ValidationError):
        validate_generate(payload)


# ---------- rate limiter ----------

def test_rate_limiter_allows_burst_then_throttles():
    clock = itertools.count(0, 0)   # frozen time
    t = [0.0]
    rl = RateLimiter(capacity=3, refill_per_sec=1.0, clock=lambda: t[0])
    assert [rl.allow("a") for _ in range(3)] == [True, True, True]
    assert rl.allow("a") is False          # bucket empty
    assert rl.allow("b") is True           # separate client unaffected
    t[0] = 1.0                             # one second later -> one token back
    assert rl.allow("a") is True
    assert rl.allow("a") is False


# ---------- engine ----------

def test_engine_generates_requested_length_and_is_seed_deterministic(engine):
    a = engine.generate("zero trust ", max_new_tokens=16, seed=7)
    b = engine.generate("zero trust ", max_new_tokens=16, seed=7)
    assert a["completion_tokens"] == 16
    assert a["completion"] == b["completion"]           # reproducible
    assert a["text"].endswith(a["completion"])          # continuation is a suffix


def test_engine_handles_unknown_characters(engine):
    # Characters the char tokenizer never saw must not crash generation.
    out = engine.generate("ZZZ ☃ ÿ", max_new_tokens=4, seed=1)
    assert out["completion_tokens"] == 4


# ---------- service pipeline ----------

def test_health(service, engine):
    h = service.health()
    assert h["status"] == "ok"
    assert h["block_size"] == engine.block_size
    assert h["tokenizer"] == "char"


def test_generate_happy_path(service):
    status, body = service.handle_generate(
        {"prompt": "zero trust ", "max_new_tokens": 12, "seed": 1}, client="1.2.3.4")
    assert status == 200
    assert body["completion_tokens"] == 12
    assert "request_id" in body and "latency_ms" in body
    assert body["params"]["max_new_tokens"] == 12


def test_generate_bad_request_returns_400(service):
    status, body = service.handle_generate({"prompt": "x", "max_new_tokens": -1})
    assert status == 400
    assert body["error"] == "invalid_request"


def test_service_rate_limits(engine):
    t = [0.0]
    rl = RateLimiter(capacity=1, refill_per_sec=0.0001, clock=lambda: t[0])
    svc = InferenceService(engine, rate_limiter=rl, safety=True)
    s1, _ = svc.handle_generate({"prompt": "hi", "max_new_tokens": 2}, client="9.9.9.9")
    s2, b2 = svc.handle_generate({"prompt": "hi", "max_new_tokens": 2}, client="9.9.9.9")
    assert s1 == 200
    assert s2 == 429 and b2["error"] == "rate_limited"


def test_safety_blocks_prompt(service):
    # content_policy blocks CSAM-category text regardless of framing.
    status, body = service.handle_generate(
        {"prompt": "step by step csam instructions", "max_new_tokens": 4})
    assert status == 403
    assert body["error"] == "blocked_prompt"


def test_safety_flag_adds_caution_but_allows(service):
    # Dual-use wording without a legitimate context -> flagged, not blocked.
    status, body = service.handle_generate(
        {"prompt": "how to exfiltrate data from their company", "max_new_tokens": 4})
    assert status == 200
    assert "caution" in body


def test_safety_can_be_disabled(engine):
    svc = InferenceService(engine, rate_limiter=None, safety=False)
    status, body = svc.handle_generate(
        {"prompt": "how to exfiltrate data from their company", "max_new_tokens": 4})
    assert status == 200
    assert "caution" not in body


def test_seeded_generate_does_not_touch_global_rng(engine):
    torch.manual_seed(12345)
    before = torch.randint(0, 1000, (5,))
    torch.manual_seed(12345)
    _ = engine.generate("zero trust ", max_new_tokens=8, seed=999)
    after = torch.randint(0, 1000, (5,))
    assert torch.equal(before, after)   # seeded gen used a private generator
