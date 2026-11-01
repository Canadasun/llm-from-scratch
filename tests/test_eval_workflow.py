"""Evaluation integrity and developer review workflows, without model weights."""

import copy
import json

import pytest

from evals.evidence import CODE_EVALUATOR, SCORING_VERSION, suite_sha256
from evals.regression import check
from evals.report import summarize
from evals.review import build_review, main, make_baseline
from evals.tasks import load_tasks


def measured():
    results = [
        {"id": "mc", "category": "security", "type": "multiple_choice", "correct": True,
         "n_choices": 4, "detail": "chose 0, answer 0"},
        {"id": "code", "category": "code", "type": "code", "correct": False,
         "execution_status": "failed", "detail": "worker exited unsuccessfully"},
    ]
    return {"results": results, "summary": summarize(results),
            "intrinsic": {"trained": {"bits_per_byte": 1.8}},
            "meta": {"checkpoint_sha256": "a" * 64, "suite_sha256": "b" * 64,
                     "heldout_sha256": "c" * 64, "scoring_version": SCORING_VERSION,
                     "code_evaluator": CODE_EVALUATOR, "complete": True, "suite": "all",
                     "limit": None, "tasks_total": 2}}


def test_mc_reference_is_compared_to_mc_only_and_intervals_are_bounded():
    summary = measured()["summary"]
    assert summary["overall"]["acc"] == .5
    assert summary["by_type"]["multiple_choice"]["acc"] == 1
    assert summary["mc_random_baseline"] == .25
    for metrics in summary["by_type"].values():
        lo, hi = metrics["wilson_95"]
        assert 0 <= lo <= metrics["acc"] <= hi <= 1


@pytest.mark.parametrize("field", ["suite_sha256", "heldout_sha256", "scoring_version", "code_evaluator"])
def test_regression_rejects_incomparable_evidence(field):
    run = measured()
    baseline = make_baseline(run)
    assert check(run, baseline)[0]
    altered = copy.deepcopy(run)
    altered["meta"][field] = "different"
    assert not check(altered, baseline)[0]


def test_smoke_run_cannot_pass_full_suite_gate_or_become_baseline():
    run = measured()
    baseline = make_baseline(run)
    run["meta"]["complete"] = False
    run["meta"]["limit"] = 1
    assert not check(run, baseline)[0]
    with pytest.raises(ValueError, match="complete"):
        make_baseline(run)


def test_missing_task_outcomes_cannot_pass_or_become_baseline():
    run = measured()
    baseline = make_baseline(run)
    run["results"].pop()
    assert not check(run, baseline)[0]
    with pytest.raises(ValueError):
        make_baseline(run)


def test_duplicate_or_reordered_results_cannot_pass_full_suite_gate():
    run = measured()
    baseline = make_baseline(run)
    run["results"] = run["results"][::-1]
    assert not check(run, baseline)[0]


def test_category_decline_cannot_hide_behind_overall_score():
    run = measured()
    baseline = make_baseline(run)
    # Forging the summary metadata has NO effect: the gate recomputes category
    # accuracy from task OUTCOMES, so a lie in either direction is ignored
    # (audit: category-metadata forgery must not change the verdict).
    forged = copy.deepcopy(run)
    forged["summary"]["by_category"]["security"]["acc"] = 0
    assert check(forged, baseline)[0]
    # A REAL decline in the outcomes (the security task now fails) blocks it.
    declined = copy.deepcopy(run)
    declined["results"][0]["correct"] = False
    declined["summary"] = summarize(declined["results"])
    assert not check(declined, baseline)[0]


@pytest.mark.parametrize("change", ["unavailable", "timeout", "nan", "miscount"])
def test_baseline_requires_complete_measured_evidence(change):
    run = measured()
    if change == "unavailable":
        run["results"][1]["execution_status"] = "unavailable"
    elif change == "timeout":
        run["results"][1]["detail"] = "timed_out (>120s)"
    elif change == "nan":
        run["intrinsic"]["trained"]["bits_per_byte"] = float('nan')
    else:
        run["summary"]["overall"]["correct"] = 99
    with pytest.raises(ValueError):
        make_baseline(run)


def test_review_escapes_generated_markdown_and_filters_failures():
    run = measured()
    run["results"][1]["detail"] = '![image](https://example.invalid) <script>bad</script> | fake'
    text = build_review(run, make_baseline(run))
    assert "![image]" not in text and "<script>" not in text
    assert "PASS — regression" in text
    assert "No failures in the selected category" in build_review(run, category="security")


def test_cli_writes_review_and_new_baseline_but_refuses_overwrite(tmp_path):
    results = tmp_path / 'run.json'
    baseline = tmp_path / 'baseline.json'
    review = tmp_path / 'review.md'
    results.write_text(json.dumps(measured()))
    assert main(['--results', str(results), '--markdown', str(review),
                 '--write-baseline', str(baseline)]) == 0
    original = baseline.read_bytes()
    assert 'Failure triage' in review.read_text()
    with pytest.raises(SystemExit) as error:
        main(['--results', str(results), '--write-baseline', str(baseline)])
    assert error.value.code == 2 and baseline.read_bytes() == original


def test_suite_fingerprint_changes_for_answers_and_order():
    tasks = [{"id": "a", "answer": 0}, {"id": "b", "answer": 1}]
    assert suite_sha256(tasks) != suite_sha256(tasks[::-1])
    changed = copy.deepcopy(tasks)
    changed[0]["answer"] = 1
    assert suite_sha256(tasks) != suite_sha256(changed)


@pytest.mark.parametrize("change", ["boolean_answer", "unknown_scorer", "empty_target", "duplicates"])
def test_invalid_task_definitions_fail_with_file_and_line(tmp_path, change):
    task = {"id": "x", "category": "security", "type": "multiple_choice",
            "prompt": "Q?", "choices": ["a", "b"], "answer": 0}
    if change == "boolean_answer":
        task['answer'] = True
    elif change in {"unknown_scorer", "empty_target"}:
        task.update(type="generation", scorer="typo" if change == "unknown_scorer" else "contains", target="")
    path = tmp_path / 'tasks.jsonl'
    path.write_text(json.dumps(task) + '\n' + (json.dumps(task) + '\n' if change == "duplicates" else ''))
    with pytest.raises(ValueError, match=r'tasks.jsonl:[12]'):
        load_tasks([str(path)])
