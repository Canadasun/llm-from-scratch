from dataclasses import replace
import json

import pytest

from data.format_probe_tasks import TASKS
from evals import format_probe as probe
from evals.code_executor import Verdict


def test_paired_formats_stay_in_family_split():
    suite = probe.build()
    assert suite["families"] == 12 and len(suite["rows"]) == 24
    train = {r["family"] for r in suite["rows"] if r["split"] == "train"}
    evaluation = {r["family"] for r in suite["rows"] if r["split"] == "evaluation"}
    assert len(train) == len(evaluation) == 6 and not train & evaluation
    for family in train | evaluation:
        rows = [r for r in suite["rows"] if r["family"] == family]
        assert {r["format"] for r in rows} == {"module", "body"}
        assert len({r["split"] for r in rows}) == 1
    assert all('"expected"' not in json.dumps(row) and '"response"' not in json.dumps(row)
               for row in suite["rows"])


def test_build_rejects_family_duplication(monkeypatch):
    monkeypatch.setattr(probe, "TASKS", (TASKS[0], replace(TASKS[0], split="evaluation")))
    with pytest.raises(ValueError, match="family"):
        probe.build()


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.family)
def test_reference_shape_and_body_boundary(task):
    assert probe.check_shape(task, "body", task.body)["format"]
    assert probe.check_shape(task, "module", task.header + task.body)["format"]
    assert not probe.check_shape(task, "body", task.header + task.body)["format"]
    assert not probe.check_shape(task, "module", "")["format"]
    assert not probe.check_shape(task, "body", "    pass\n")["format"]


def test_shape_rejects_wrong_signature_and_disguised_empty_body():
    task = TASKS[0]
    for source in ("def wrong(items):\n    return []\n",
                   "def value_runs(other):\n    return []\n",
                   task.header + '    "only docs"\n',
                   task.header + "    def nested():\n        return []\n",
                   "@decorator\n" + task.header + task.body):
        assert not probe.check_shape(task, "module", source)["format"]
    assert not probe.check_shape(task, "module", " " * 140000)["format"]
    with pytest.raises(ValueError, match="format"):
        probe.prompt(task, "invented")


def test_syntax_check_never_executes_code(tmp_path):
    target = tmp_path / "marker"
    response = TASKS[0].header + f"    open({str(target)!r}, 'w').write('bad')\n    return []\n"
    assert probe.check_shape(TASKS[0], "module", response)["format"]
    assert not target.exists()


def test_shape_failure_never_calls_executor(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("must not execute")
    monkeypatch.setattr(probe, "evaluate_cases", forbidden)
    assert not probe.score_response(TASKS[0], "module", "")["functional"]


def test_functional_failure_is_separate_from_valid_shape(monkeypatch):
    calls = []
    def executor(source, name, cases, **kwargs):
        calls.append((source, name, cases))
        return Verdict("failed", False, 0, len(cases))
    monkeypatch.setattr(probe, "evaluate_cases", executor)
    result = probe.score_response(TASKS[0], "body", "    return None\n")
    assert result["syntax"] and result["format"] and not result["functional"]
    assert calls[0][0].startswith(TASKS[0].header)
    monkeypatch.setattr(probe, "evaluate_cases", lambda *a, **k: Verdict("unavailable", False))
    with pytest.raises(RuntimeError, match="sandbox"):
        probe.score_response(TASKS[0], "body", TASKS[0].body)


def test_verification_fails_closed_without_linux_sandbox(monkeypatch):
    monkeypatch.setattr(probe, "sandbox_preflight", lambda: {"available": False})
    with pytest.raises(RuntimeError, match="no host fallback"):
        probe.verify()
