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
