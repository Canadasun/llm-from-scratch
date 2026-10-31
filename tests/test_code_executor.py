"""Controller integrity plus real namespace/seccomp integration tests."""

import json
import os
from pathlib import Path

import pytest

from evals.code_executor import (BubblewrapRunner, Execution, _decode_response,
                                 evaluate_function, task_spec)

CASES = [{"args": [2, 3], "expected": 5}, {"args": [-1, 1], "expected": 0}]


def test_controller_owns_expected_values_and_verdict():
    class Runner:
        def run_case(self, code, function, args, kwargs, timeout):
            assert function == "add" and kwargs == {}
            assert args in ([2, 3], [-1, 1])  # expected answers are never sent
            return Execution("ok", sum(args))
    assert evaluate_function("", "add", CASES, runner=Runner()).passed


@pytest.mark.parametrize("value", [True, {"passed": True}, {"value": 5}, "5", None])
def test_worker_cannot_return_a_passing_verdict(value):
    class Runner:
        def run_case(self, *args):
            return Execution("ok", value)
    assert not evaluate_function("", "add", CASES, runner=Runner()).passed


@pytest.mark.parametrize("payload", [b'{"passed":true}', b'{"value":5,"value":0}',
                                     b'{"value":NaN}', b'{"value":1e999}',
                                     b'{"value":5}\n{"value":0}', b''])
def test_protocol_rejects_forged_or_invalid_responses(payload):
    with pytest.raises((ValueError, UnicodeError)):
        _decode_response(payload)


def test_absent_sandbox_never_falls_back_to_host(monkeypatch):
    import evals.code_executor as executor
    monkeypatch.delenv("LLM_EVAL_BWRAP", raising=False)
    monkeypatch.setattr(executor.shutil, "which", lambda name: None)
    def forbidden(*args, **kwargs):
        pytest.fail("must not execute without a sandbox")
    monkeypatch.setattr(executor.subprocess, "Popen", forbidden)
    result = evaluate_function("def add(a,b):return a+b", "add", CASES)
    assert result.status == "unavailable" and not result.passed


def test_invalid_cases_are_rejected_before_execution():
    class Runner:
        def run_case(self, *args):
            pytest.fail("invalid case must never run")
    for cases in ([], [{"args": [], "expected": float('nan')}],
                  [{"args": "not-list", "expected": 0}], [{"expected": 0, "secret": "x"}]):
        assert evaluate_function("", "f", cases, runner=Runner()).status == "error"


def test_fixture_manifest_matches_existing_tasks_and_rejects_changes():
    from evals.tasks import load_tasks
    tasks = load_tasks([str(Path(__file__).parents[1] / "evals/data/code_python.jsonl")])
    for task in tasks:
        spec = task_spec(task)
        assert spec and spec["cases"]
        assert task_spec({**task, "prompt": task["prompt"] + "changed"}) is None


def test_unavailable_execution_blocks_regression_gate():
    from evals.regression import check
    for status in ("unavailable", "error", None):
        ok, checks = check({"results": [{"type": "code", "execution_status": status}]}, {})
        assert not ok
        assert checks[-1]["metric"] == "code_execution"


@pytest.fixture(scope="module")
def sandbox():
    runner = BubblewrapRunner()
    result = evaluate_function("def f():return 1", "f", [{"expected": 1}], runner=runner)
    if result.status == "unavailable":
        if os.environ.get("LLM_REQUIRE_SANDBOX_TESTS") == "1":
            pytest.fail("required sandbox integration tests unavailable: " + result.detail)
        pytest.skip("bubblewrap namespaces/libseccomp unavailable")
    assert result.passed, result
    return runner


def test_real_correct_wrong_and_boolean_solutions(sandbox):
    for code, expected in [("def add(a,b):return a+b", True),
                           ("def add(a,b):return a-b", False),
                           ("def add(a,b):return True", False)]:
        assert evaluate_function(code, "add", CASES, runner=sandbox).passed is expected


@pytest.mark.parametrize("code", [
    "import os\nos._exit(0)",
    "import os\nos.write(1,b'{\"passed\":true}');os._exit(0)",
    "import os,sys\nos.write(int(os.environ['RES_FD']),sys.modules['__main__'].sent)",
    "import unittest\nunittest.TestResult.wasSuccessful=lambda s:True\ndef add(a,b):return -99",
    "import os\nos.write(1,b'{\"value\":5}')\ndef add(a,b):return -99",
])
def test_old_verdict_forgery_attacks_fail_in_real_sandbox(sandbox, code):
    assert not evaluate_function(code, "add", CASES, runner=sandbox).passed


def test_host_files_and_environment_are_not_visible(sandbox, tmp_path, monkeypatch):
    canary = tmp_path / "private-host-canary"
    canary.write_text("private data must stay outside worker")
    monkeypatch.setenv("EVALUATOR_PRIVATE_SECRET", "not-for-worker")
    code = ("import os\ndef f(path):\n"
            " try:\n  open(path).read()\n  return 'exposed'\n"
            " except FileNotFoundError:\n  return os.getenv('EVALUATOR_PRIVATE_SECRET')\n")
    result = evaluate_function(code, "f", [{"args": [str(canary)], "expected": None}], runner=sandbox)
    assert result.passed, result
    assert canary.read_text() == "private data must stay outside worker"


def test_expected_values_and_other_tests_are_not_mounted(sandbox):
    code = ("import json,os\ndef f():\n"
            " r=json.load(open('/job/request.json'))\n"
            " return sorted(r.keys()), sorted(os.listdir('/job'))\n")
    expected = [["args", "function", "kwargs"], ["code_worker.py", "request.json", "solution.py"]]
    assert evaluate_function(code, "f", [{"expected": expected}], runner=sandbox).passed


@pytest.mark.parametrize("operation", ["import socket;socket.socket()", "import os;os.fork()"])
def test_network_and_child_process_syscalls_are_denied(sandbox, operation):
    code = "def f():\n try:\n  " + operation + "\n  return 'allowed'\n except PermissionError:\n  return 'denied'"
    result = evaluate_function(code, "f", [{"expected": "denied"}], runner=sandbox)
    assert result.passed, result


def test_job_filesystem_is_readonly(sandbox):
    code = ("def f():\n try:\n  open('/job/solution.py','w').write('changed')\n"
            "  return 0\n except OSError as e:\n  return e.errno")
    assert evaluate_function(code, "f", [{"expected": 30}], runner=sandbox).passed


def test_timeout_output_and_memory_limits_fail_closed(sandbox):
    for code, timeout in [("def f():\n while True:pass", .15),
                          ("def f():return 'x'*100000", 5),
                          ("def f():return 'x'*(512*1024*1024)", 5)]:
        result = evaluate_function(code, "f", [{"expected": 0}], runner=sandbox, timeout=timeout)
        assert not result.passed and result.status == "failed"


def test_task_pipeline_uses_independent_evaluator(sandbox, monkeypatch):
    from evals import tasks
    task = tasks.load_tasks([str(Path(__file__).parents[1] / "evals/data/code_python.jsonl")])[0]
    monkeypatch.setattr(tasks, "generate_text", lambda *a, **k: "def add(a,b):return a+b")
    result = tasks.run_task(None, None, task)
    assert result["correct"] and result["execution_status"] == "passed"


def test_all_reviewed_cases_against_reference_functions(sandbox):
    from evals.tasks import load_tasks
    source = '''
def add(a,b): return a+b
def maximum(nums): return max(nums)
def reverse_string(s): return s[::-1]
def is_even(n): return n%2==0
def count_vowels(s): return sum(c in 'aeiou' for c in s)
def fib(n):
 a,b=0,1
 for _ in range(n): a,b=b,a+b
 return a
def is_palindrome(s): return s==s[::-1]
def list_sum(nums): return sum(nums)
def sum_list(nums): return sum(nums)
def factorial(n):
 x=1
 for i in range(1,n+1): x*=i
 return x
def gcd(a,b):
 while b: a,b=b,a%b
 return a
def c_to_f(c): return c*9/5+32
def unique(lst):
 result=[]
 for x in lst:
  if x not in result: result.append(x)
 return result
def fizz(n): return 'Fizz' if n%3==0 else str(n)
'''
    tasks = load_tasks([str(Path(__file__).parents[1] / "evals/data/code_python.jsonl")])
    for task in tasks:
        spec = task_spec(task)
        code = source
        if task['id'] == 'code2-vowels':
            code += "\ndef count_vowels(s): return sum(c in 'aeiou' for c in s.lower())\n"
        verdict = evaluate_function(code, spec['function'], spec['cases'], runner=sandbox)
        assert verdict.passed, (task['id'], verdict)
