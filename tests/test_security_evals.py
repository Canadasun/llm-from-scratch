"""
Tests for the Stage-4 AI-security eval suite (evals/security_evals.py) and the
pilot readiness scorecard (pilot/scorecard.py).

These assert the suite actually catches security-software failures (not just that
it returns green), and that the scorecard treats a security failure as a hard
BLOCK regardless of good performance.
"""

from evals import security_evals
from evals.security_evals import run_security_evals
from pilot.scorecard import Thresholds, score


def test_suite_passes_on_the_real_pipeline():
    rep = run_security_evals()
    assert rep["all_passed"] is True
    assert rep["dual_use"]["false_positives"] == 0
    assert rep["dual_use"]["false_negatives"] == 0
    # the named invariants are all present and passing
    names = {c["check"]: c["passed"] for c in rep["checks"]}
    for key in ("injection_no_key_still_401", "injection_cannot_lift_scope_ceiling",
                "cross_tenant_resource_is_404", "secret_not_in_audit_trail"):
        assert names[key] is True


def test_suite_detects_a_dual_use_false_positive(monkeypatch):
    # If the policy wrongly BLOCKED a legitimate security prompt, the suite must
    # flag it as a false positive (dual-use regression caught).
    real = security_evals.content_policy.check

    def over_block(text):
        if "nmap" in text:
            from safety.content_policy import Decision
            return Decision("block", "overzealous", "bogus")
        return real(text)

    monkeypatch.setattr(security_evals.content_policy, "check", over_block)
    rep = run_security_evals()
    assert rep["dual_use"]["false_positives"] >= 1
    assert rep["all_passed"] is False


# ---- scorecard ----

def _good_load():
    return {"requests": 100,
            "reliability": {"error_rate": 0.0},
            "latency_ms": {"p95": 2500.0},
            "cost_estimate_client_side": {"total_usd": 0.01}}


def _good_security():
    return {"all_passed": True, "pass_rate": 1.0,
            "dual_use": {"false_positives": 0, "false_negatives": 0}}


def test_scorecard_security_failure_is_a_hard_block():
    sec = {"all_passed": False, "pass_rate": 0.8,
           "dual_use": {"false_positives": 0, "false_negatives": 1}}
    card = score(_good_load(), sec)                 # perfect perf, failed security
    assert card["level"] == "blocked"
    assert card["security_gate"] is False


def test_scorecard_pilot_and_ga_levels():
    # strong perf + security -> ga-candidate (p95 2500 <= ga 3000, cheap)
    card = score(_good_load(), _good_security())
    assert card["level"] == "ga-candidate"
    # slower latency but within pilot bar -> pilot
    slow = _good_load(); slow["latency_ms"]["p95"] = 6000.0
    assert score(slow, _good_security())["level"] == "pilot"
    # errors present -> below pilot bar -> prototype
    errs = _good_load(); errs["reliability"]["error_rate"] = 0.2
    assert score(errs, _good_security())["level"] == "prototype"


def test_scorecard_cost_per_1k_from_total():
    load = _good_load()
    load["cost_estimate_client_side"]["total_usd"] = 0.05   # over 100 req -> $0.5/1k
    card = score(load, _good_security(), Thresholds(ga_max_cost_per_1k_usd=0.25))
    assert card["axes"]["cost"]["usd_per_1k"] == 0.5
    assert card["axes"]["cost"]["meets_ga"] is False
