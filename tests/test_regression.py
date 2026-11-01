"""
Tests for the eval regression gate (evals/regression.py).

Pure comparison logic — crafted candidate/baseline dicts, no model needed.
"""

import json
import os

from evals.regression import check

BASE = {
    "enforce": ["bits_per_byte", "overall_acc"],
    "intrinsic": {"trained": {"perplexity": 15.6, "bits_per_byte": 2.15,
                              "top1_acc": 0.447, "top5_acc": 0.691}},
    "summary": {"overall": {"acc": 0.23}},
    "tolerances": {"bits_per_byte_abs_max_increase": 0.15,
                   "overall_acc_abs_max_drop": 0.05},
}


def _cand(bpb=2.15, acc=0.23, ppl=15.6, top1=0.447):
    return {
        "intrinsic": {"trained": {"perplexity": ppl, "bits_per_byte": bpb,
                                  "top1_acc": top1, "top5_acc": 0.691}},
        "summary": {"overall": {"acc": acc}},
    }


def _get(checks, metric):
    return next(c for c in checks if c["metric"] == metric)


def test_identical_passes():
    ok, checks = check(_cand(), BASE)
    assert ok is True


def test_improvement_passes():
    ok, checks = check(_cand(bpb=1.80, acc=0.40), BASE)
    assert ok is True
    assert _get(checks, "bits_per_byte")["ok"]
    assert _get(checks, "overall_acc")["ok"]


def test_bits_per_byte_regression_fails():
    ok, checks = check(_cand(bpb=2.40), BASE)   # +0.25 > 0.15 tolerance
    assert ok is False
    assert _get(checks, "bits_per_byte")["ok"] is False


def test_small_bpb_worsening_within_tolerance_passes():
    ok, checks = check(_cand(bpb=2.25), BASE)   # +0.10 <= 0.15 tolerance
    assert ok is True


def test_overall_acc_drop_fails():
    ok, checks = check(_cand(acc=0.17), BASE)   # -0.06 > 0.05 tolerance
    assert ok is False
    assert _get(checks, "overall_acc")["ok"] is False


def test_non_enforced_metric_does_not_fail_build():
    # perplexity balloons, but it is not in `enforce` -> recorded, not gated.
    ok, checks = check(_cand(ppl=9999.0), BASE)
    assert ok is True
    ppl = _get(checks, "perplexity")
    assert ppl["skipped"] is True and ppl["ok"] is True


def test_missing_enforced_candidate_metric_fails_not_skipped():
    # Integrity: a metric the baseline GATES ON that the candidate does not
    # report must FAIL the gate — absent evidence is never "no regression".
    # (Previously this silently passed, so an empty candidate {} could pass.)
    cand = _cand()
    del cand["intrinsic"]["trained"]["bits_per_byte"]
    ok, checks = check(cand, BASE)
    assert ok is False
    c = _get(checks, "bits_per_byte")
    assert c["skipped"] is False and c["ok"] is False


def test_empty_candidate_cannot_pass_against_populated_baseline():
    ok, checks = check({}, BASE)
    assert ok is False


def test_committed_m2_baseline_parses_and_gates():
    path = os.path.join(os.path.dirname(__file__), "..", "evals", "baselines", "m2.json")
    with open(path) as f:
        base = json.load(f)
    assert base["enforce"] == ["bits_per_byte", "overall_acc"]
    # an M3-like candidate that improves on the fair metrics must pass
    ok, _ = check(_cand(bpb=1.9, acc=0.35), base)
    assert ok is True
    # a candidate that regresses bits/byte must fail
    ok2, _ = check(_cand(bpb=2.6), base)
    assert ok2 is False


def test_category_floor_uses_predeclared_loss_budget():
    # Small category (n=14): the floor is the baseline's wide Wilson lower bound,
    # so a single borderline flip (7/14 -> 6/14), statistically indistinguishable
    # from the baseline, must NOT fail (the old fixed 5% drop wrongly did).
    base = {"enforce_categories": ["reasoning"], "enforce": ["overall_acc"],
            "summary": {"overall": {"acc": 0.5},
                        "by_category": {"reasoning": {
                            "n": 14, "correct": 7, "acc": 0.5,
                            "wilson_95": [0.2679920, 0.7320079]}}}}
    noise = {"summary": {"overall": {"acc": 0.5},
                         "by_category": {"reasoning": {"n": 14, "correct": 6, "acc": 6/14}}}}
    ok, checks = check(noise, base)
    assert ok is False
    assert _get(checks, "category:reasoning")["ok"] is False
    ok, _ = check(noise, base, {"category_acc_abs_max_drop": 0.08})
    assert ok is True

    # A genuine collapse (3/14 = 0.214) falls below the Wilson lower bound -> FAIL.
    collapse = {"summary": {"overall": {"acc": 0.5},
                            "by_category": {"reasoning": {"n": 14, "correct": 3, "acc": 3/14}}}}
    ok2, checks2 = check(collapse, base)
    assert ok2 is False
    assert _get(checks2, "category:reasoning")["ok"] is False


def test_category_floor_scales_tight_for_large_samples():
    # Large category (n=200): the recomputed Wilson interval is tight, so even a
    # ~10-point drop is a real regression and must fail. (No stored interval ->
    # the floor is recomputed from n/correct.)
    base = {"enforce_categories": ["security"], "enforce": ["overall_acc"],
            "summary": {"overall": {"acc": 0.5},
                        "by_category": {"security": {"n": 200, "correct": 100, "acc": 0.5}}}}
    cand = {"summary": {"overall": {"acc": 0.5},
                        "by_category": {"security": {"n": 200, "correct": 80, "acc": 0.4}}}}
    ok, checks = check(cand, base)
    assert ok is False
    assert _get(checks, "category:security")["ok"] is False
