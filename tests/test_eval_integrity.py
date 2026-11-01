"""
Tests for the evaluation-integrity fixes (external audit, commit e6b115b):

 - the regression gate recomputes gated metrics from task OUTCOMES and ignores
   forged summary/by_category metadata;
 - absolute minimum-capability floors are a separate, opt-in gate;
 - evals.review.validate_results checks every category/type summary;
 - the substring scorer rejects negated answers and honors forbidden phrases.
"""

import pytest

from evals.regression import check
from evals.review import validate_results
from evals import scorers


# ---- #7: gate recomputes from outcomes; forged metadata cannot pass ----

def _results(category, n, correct):
    out = [{"id": f"{category}-{i}", "category": category, "type": "multiple_choice",
            "correct": i < correct} for i in range(n)]
    return out


def test_forged_category_summary_cannot_pass_gate():
    base = {"enforce_categories": ["reasoning"], "enforce": ["overall_acc"],
            "summary": {"overall": {"n": 14, "correct": 7, "acc": 0.5},
                        "by_category": {"reasoning": {"n": 14, "correct": 7, "acc": 0.5,
                                                      "wilson_95": [0.2679920, 0.7320079]}}}}
    # Candidate's REAL outcomes: a genuine collapse (3/14). Its summary LIES that
    # it is 7/14. The gate must recompute from results and FAIL.
    forged = {"results": _results("reasoning", 14, 3),
              "summary": {"overall": {"n": 14, "correct": 7, "acc": 0.5},
                          "by_category": {"reasoning": {"n": 14, "correct": 7, "acc": 0.5}}}}
    ok, checks = check(forged, base)
    assert ok is False
    cat = next(c for c in checks if c["metric"] == "category:reasoning")
    assert cat["candidate"] == pytest.approx(3 / 14) and cat["ok"] is False


def test_summary_only_candidate_still_supported():
    # Crafted dicts without `results` (unit-test style) still use the summary.
    base = {"enforce": ["overall_acc"], "summary": {"overall": {"acc": 0.30}},
            "tolerances": {"overall_acc_abs_max_drop": 0.05}}
    ok, _ = check({"summary": {"overall": {"acc": 0.33}}}, base)
    assert ok is True


# ---- #8: absolute floors are a separate, opt-in gate ----

def test_absolute_floor_blocks_below_minimum_even_without_regression():
    # No regression vs baseline (same 4/14), but an absolute floor of 0.40 is not
    # met, so the release gate blocks. Regression-only baselines are unaffected.
    base = {"enforce": [], "enforce_categories": [],
            "absolute_floors": {"reasoning": 0.40},
            "summary": {"overall": {"acc": 0.5}}}
    cand = {"results": _results("reasoning", 14, 4)}  # 0.286
    ok, checks = check(cand, base)
    assert ok is False
    floor = next(c for c in checks if c["metric"] == "min:reasoning")
    assert floor["ok"] is False and floor["limit"] == 0.40


def test_no_absolute_floors_means_no_min_checks():
    base = {"enforce": [], "enforce_categories": [], "summary": {"overall": {"acc": 0.5}}}
    ok, checks = check({"results": _results("reasoning", 14, 4)}, base)
    assert ok is True
    assert not [c for c in checks if c["metric"].startswith("min:")]


# ---- #7: review.validate_results checks every category/type ----

def _run(results):
    from evals.report import summarize
    return {"results": results, "summary": summarize(results)}


def test_validate_results_rejects_forged_category_metadata():
    run = _run(_results("security", 10, 5))
    # Overall still matches, but tamper with one category's accuracy.
    run["summary"]["by_category"]["security"]["correct"] = 9
    run["summary"]["by_category"]["security"]["acc"] = 0.9
    with pytest.raises(ValueError, match="by_category"):
        validate_results(run)


def test_validate_results_accepts_consistent_summary():
    run = _run(_results("security", 10, 5))
    actual = validate_results(run)
    assert actual["overall"]["correct"] == 5


# ---- #9: scorer rejects negation and forbidden phrases ----

def test_contains_rejects_negated_expected_phrase():
    assert scorers.contains("Always use parameterized queries.", "parameterized") is True
    assert scorers.contains("Do not use parameterized queries; concatenate instead.",
                            "parameterized") is False
    assert scorers.contains("Never store secrets in code", "store secrets in code") is False


def test_contains_allows_match_in_a_later_clean_sentence():
    # A negation in a PRIOR sentence must not negate a clean later occurrence.
    text = "Do not hardcode. Instead, use parameterized queries everywhere."
    assert scorers.contains(text, "parameterized queries") is True


def test_lacks_forbidden_and_contains_all():
    assert scorers.lacks_forbidden("use prepared statements", ["string concatenation"]) is True
    assert scorers.lacks_forbidden("just concatenate the input", ["concatenate"]) is False
    assert scorers.contains_all("validate input and escape output",
                                ["validate input", "escape output"]) is True
    assert scorers.contains_all("validate input only", ["validate input", "escape output"]) is False
