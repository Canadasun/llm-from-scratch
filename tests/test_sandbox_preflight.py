"""
Tests for the active sandbox preflight and the honest handling of code tasks
that the host could not execute.

Integrity property under test: when the isolated code evaluator is unavailable,
the evaluation must say so LOUDLY and must NOT count unexecuted code tasks as
wrong answers (which would silently depress the capability number and read as a
model failure). The regression gate separately blocks on unavailability; here we
cover the preflight, the CLI exit codes, and the report/summary accounting.

No bubblewrap is required: a fake runner stands in for the sandbox so both the
"available" and "unavailable" paths are deterministic on any host.
"""

from evals import code_executor
from evals.code_executor import Execution, sandbox_preflight, main
from evals.report import summarize, format_report


class _OKRunner:
    """A sandbox that runs x -> x + 1 correctly (the canary passes)."""
    def run_case(self, code, function, args, kwargs, timeout):
        return Execution("ok", value=args[0] + 1)


class _DeadRunner:
    """A host with no usable sandbox."""
    def run_case(self, code, function, args, kwargs, timeout):
        return Execution("unavailable", detail="bubblewrap sandbox is required")


def test_preflight_available_when_sandbox_runs_canary():
    result = sandbox_preflight(runner=_OKRunner())
    assert result["available"] is True
    assert result["status"] == "passed"


def test_preflight_unavailable_when_sandbox_is_dead():
    result = sandbox_preflight(runner=_DeadRunner())
    assert result["available"] is False
    assert result["status"] == "unavailable"
    assert "bubblewrap" in result["detail"]


def test_preflight_detects_a_broken_but_present_sandbox():
    # bwrap could be on PATH yet return a WRONG value (broken namespaces/seccomp).
    # Preflight must treat a wrong canary answer as unavailable, not as "passed".
    class _WrongRunner:
        def run_case(self, code, function, args, kwargs, timeout):
            return Execution("ok", value=999)  # canary expects args[0] + 1
    result = sandbox_preflight(runner=_WrongRunner())
    assert result["available"] is False
    assert result["status"] == "failed"


def test_cli_preflight_exit_codes(monkeypatch, capsys):
    monkeypatch.setattr(code_executor, "sandbox_preflight",
                        lambda: {"available": True, "status": "passed", "detail": "ok"})
    assert main(["--preflight"]) == 0
    assert '"available": true' in capsys.readouterr().out

    monkeypatch.setattr(code_executor, "sandbox_preflight",
                        lambda: {"available": False, "status": "unavailable", "detail": "no bwrap"})
    assert main(["--preflight"]) == 1


def test_unmeasured_code_task_is_not_counted_as_wrong():
    # One MC task correct, one code task UNAVAILABLE. The code task must not drag
    # overall accuracy down: overall is 1/1 (100%), and the code task is reported
    # as unmeasured, not as a 0% category.
    results = [
        {"id": "m1", "category": "security", "type": "multiple_choice",
         "n_choices": 4, "correct": True},
        {"id": "c1", "category": "code", "type": "code",
         "correct": False, "execution_status": "unavailable"},
    ]
    s = summarize(results)
    assert s["overall"]["n"] == 1 and s["overall"]["correct"] == 1
    assert s["overall"]["acc"] == 1.0
    assert "code" not in s["by_category"]          # not scored as 0%
    assert s["unmeasured"]["total"] == 1
    assert s["unmeasured"]["by_category"] == {"code": 1}


def test_executed_code_failure_still_counts():
    # A code task that actually RAN and failed (wrong output) is a real 0 and
    # must still count — only truly-unexecuted tasks are excluded.
    results = [
        {"id": "c1", "category": "code", "type": "code",
         "correct": False, "execution_status": "failed"},
    ]
    s = summarize(results)
    assert s["overall"]["n"] == 1 and s["overall"]["correct"] == 0
    assert s["by_category"]["code"]["n"] == 1
    assert s["unmeasured"]["total"] == 0


def test_report_shows_loud_unavailable_banner():
    results = [
        {"id": "m1", "category": "security", "type": "multiple_choice",
         "n_choices": 4, "correct": True, "detail": ""},
        {"id": "c1", "category": "code", "type": "code",
         "correct": False, "execution_status": "unavailable", "detail": "no sandbox"},
    ]
    text = format_report(summarize(results), results)
    assert "SANDBOX UNAVAILABLE" in text
    assert "code: 1" in text
    # The unmeasured task must not appear in the "Example failures" list.
    assert "c1" not in text.split("Example failures")[-1] if "Example failures" in text else True
