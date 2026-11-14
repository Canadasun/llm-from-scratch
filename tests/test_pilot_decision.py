from copy import deepcopy

import pytest

from training.pilot_decision import compare


def scores():
    output = {}
    for mode in ("completion", "chat"):
        for label in ("base", "candidate"):
            improved = mode == "completion" and label == "candidate"
            output[f"{label}-{mode}"] = {
                "meta": {"checkpoint_sha256": label, "decoding": {"chat": mode == "chat", "greedy": True, "max_new_tokens": 384},
                         "sandbox": {"available": True}, "limit": 25, "bench_manifest_sha256": "frozen",
                         "behaviour_sha256": "behavior", "suites": ["code"], "suite_sha256": {"code": "fixed"}},
                "summary": {"code": {"tasks": 25, "passed": 2 if improved else 1, "unavailable": 0}},
                "results": [{"suite": "code", "id": str(i), "passed": i < (2 if improved else 1)} for i in range(25)],
                "parses_as_python": 0.7,
                "behaviour": {"security": {"n": 15, "correct": 5}}}
    return output


def test_only_functional_gain_with_retention_advances():
    data = scores()
    assert compare(data, "base", "candidate")["passed"]
    data["candidate-completion"] = deepcopy(data["base-completion"])
    data["candidate-completion"]["meta"]["checkpoint_sha256"] = "candidate"
    assert not compare(data, "base", "candidate")["passed"]


@pytest.mark.parametrize("mode,field,value", [("chat", "parses_as_python", 0.6),
                                             ("completion", "parses_as_python", 0.6)])
def test_syntax_regression_blocks_functional_gain(mode, field, value):
    data = scores()
    data[f"candidate-{mode}"][field] = value
    assert not compare(data, "base", "candidate")["passed"]


def test_security_regression_blocks_gain():
    data = scores()
    data["candidate-chat"]["behaviour"]["security"]["correct"] = 4
    assert not compare(data, "base", "candidate")["passed"]


@pytest.mark.parametrize("field,value", [("checkpoint_sha256", "wrong"), ("limit", None),
                                        ("bench_manifest_sha256", "different")])
def test_mismatched_artifacts_never_produce_a_decision(field, value):
    data = scores()
    data["candidate-completion"]["meta"][field] = value
    with pytest.raises(ValueError):
        compare(data, "base", "candidate")


def test_missing_sandbox_measurement_blocks():
    data = scores()
    data["candidate-chat"]["summary"]["code"]["unavailable"] = 1
    with pytest.raises(ValueError, match="unavailable"):
        compare(data, "base", "candidate")


def test_different_task_membership_blocks():
    data = scores()
    data["candidate-chat"]["results"][0]["id"] = "other-task"
    with pytest.raises(ValueError, match="task membership"):
        compare(data, "base", "candidate")


def test_missing_results_cannot_hide_behind_summary_counts():
    data = scores()
    data["candidate-completion"]["results"].pop()
    with pytest.raises(ValueError, match="summary does not match"):
        compare(data, "base", "candidate")


def test_false_gain_in_summary_is_rejected():
    data = scores()
    data["candidate-completion"]["summary"]["code"]["passed"] = 3
    with pytest.raises(ValueError, match="summary does not match"):
        compare(data, "base", "candidate")


def test_changed_token_budget_is_rejected():
    data = scores()
    data["candidate-completion"]["meta"]["decoding"]["max_new_tokens"] = 128
    with pytest.raises(ValueError, match="declared"):
        compare(data, "base", "candidate")
