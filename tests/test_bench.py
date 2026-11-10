import ast
import json
import math
import os

import pytest

from evals import bench as B
from evals.code_executor import BubblewrapRunner, Execution, evaluate_cases, evaluate_function, values_equal
from evals.code_worker import decode_value, encode_value


def test_codec_round_trips_python_types():
    value = [1, 2.5, "s", None, True, (1, (2, 3)), {1, 2}, frozenset({"a"}), {"k": [1]}, {1: "x", (2, 3): None},
             complex(1, -2), float("inf"), {"$tuple": [1]}]
    encoded = encode_value(value)
    json.dumps(encoded, allow_nan=False)
    back = decode_value(json.loads(json.dumps(encoded)))
    assert back[:12] == value[:12] and back[12] == {"$tuple": [1]}
    assert encode_value({"b", "a", "c"}) == {"$set": ["a", "b", "c"]}
    assert math.isnan(decode_value(encode_value(float("nan"))))
    with pytest.raises(TypeError):
        encode_value(object())
    with pytest.raises(ValueError):
        decode_value({"$float": "1e999"})


def test_values_equal_follows_benchmark_semantics():
    assert values_equal(0.1 + 0.2, 0.3) and not values_equal(0.31, 0.3)
    assert values_equal(3.0, 3) and values_equal(3, 3.0)
    assert values_equal(True, 1) and values_equal(0, False) and not values_equal(True, 2)
    assert values_equal([1, 0], [True, False]) and not values_equal("1", True)
    assert not values_equal([1, 2], (1, 2)) and values_equal((1, [2.0000001]), (1, [2.0]))
    assert values_equal({1, 2}, {2, 1}) and not values_equal({"a": 1}, {"a": 1, "b": 2})
    assert values_equal(float("nan"), float("nan")) and not values_equal(1.0, float("nan"))
    assert values_equal(1.001, 1.0, atol=0.01) and not values_equal(1.1, 1.0, atol=0.01)


def _assert(src, name="f"):
    return B.assert_case(ast.parse(src).body[0].test, name)


def test_assert_parsing_covers_benchmark_forms():
    case, atol = _assert("assert f([1, 2], k=3) == (4, 5)")
    assert case == {"args": [[1, 2]], "kwargs": {"k": 3}, "expected": {"$tuple": [4, 5]}} and atol == 0
    assert _assert("assert f(3 * 19, -2 ** 3, 'ab' + 'c') == [3, 19]")[0]["args"] == [57, -8, "abc"]
    assert _assert("assert abs(f(1.0) - 2.0/3.0) < 1e-6") == ({"args": [1.0], "expected": 2 / 3}, 1e-6)
    assert _assert("assert math.isclose(f(2), 3.14, rel_tol=0.001)")[1] == pytest.approx(0.00314)
    assert _assert("assert f(4) is False")[0] == {"args": [4], "expected": False, "compare": "is"}
    assert _assert("assert not f(4)")[0] == {"args": [4], "expected": False, "compare": "truth"}
    assert _assert("assert f(4)")[0] == {"args": [4], "expected": True, "compare": "truth"}
    case, _ = _assert("assert set(f((3, 4), (4, 5))) == set((4,))")
    assert case["compare"] == "set" and case["expected"] == {"$set": [4]}
    assert _assert("assert tuple(f([1])) == tuple([1])")[0]["compare"] == "sequence"
    for bad in ("assert f(x) == 1", "assert f(len('a')) == 1", "assert g(1) == 1", "assert f(2 ** 100000) == 1",
                "assert f(*a) == 1", "assert f(1) > 2"):
        with pytest.raises(B.Unsupported):
            _assert(bad)


def test_assert_block_rejects_loops_and_imports():
    body = ast.parse("print\nassert f(1) == 2\n'doc'\nassert True").body
    assert len(B.assert_cases(body, "f")[0]) == 1
    for bad in ("import random\nassert f(1) == 2", "for i in range(3):\n    assert f(i) == i"):
        with pytest.raises(B.Unsupported):
            B.assert_cases(ast.parse(bad).body, "f")


HE_PLUS = '''
import numpy as np

def assertion(out, exp, atol):
    exact_match = out == exp
    if atol == 0 and is_floats(exp):
        atol = 1e-6
    if not exact_match and atol != 0:
        assert np.allclose(out, exp, rtol=1e-07, atol=atol)
    else:
        assert exact_match


def check(candidate):
    inputs = [[[1, 2], 3], [[], inf]]
    results = [(1, 2.5), None]
    for i, (inp, exp) in enumerate(zip(inputs, results)):
        assertion(candidate(*inp), exp, 1e-06)
'''

MBPP_PLUS_SET = '''
def assertion(out, exp, atol):
    out = set(out)
    exp = set(exp)
    assert out == exp

inputs = [[(3, 4), (4, 5)]]
results = [(4,)]
for i, (inp, exp) in enumerate(zip(inputs, results)):
    assertion(similar(*inp), exp, 0)
'''


def test_literal_io_parses_both_layouts_and_rejects_reference_functions():
    cases, atol, total = B.literal_io_tasks({"test": HE_PLUS})
    assert total == 2 and atol == 1e-6
    assert cases[0] == {"args": [[1, 2], 3], "expected": {"$tuple": [1, 2.5]}}
    assert cases[1]["args"] == [[], {"$float": "inf"}] and cases[1]["expected"] is None
    cases, _, _ = B.literal_io_tasks({"test": MBPP_PLUS_SET})
    assert cases == [{"args": [{"$tuple": [3, 4]}, {"$tuple": [4, 5]}], "expected": {"$set": [4]}, "compare": "set"}]
    ref = HE_PLUS.replace("    results = [(1, 2.5), None]\n", "").replace(
        "for i, (inp, exp) in enumerate(zip(inputs, results)):\n        assertion(candidate(*inp), exp, 1e-06)",
        "for inp in inputs:\n        assertion(candidate(*inp), ref_func(*inp), 0)")
    with pytest.raises(B.Unsupported, match="reference function"):
        B.literal_io_tasks({"test": ref})


class LocalRunner:
    def __init__(self):
        self.batches = []

    def run_batch(self, code, function, calls, timeout):
        self.batches.append(len(calls))
        namespace = {}
        exec(compile(code, "<trusted-test>", "exec"), namespace)
        out = []
        for call in calls:
            try:
                value = namespace[function](*decode_value(call["args"]), **decode_value(call["kwargs"]))
                out.append(("ok", decode_value(json.loads(json.dumps(encode_value(value))))))
            except Exception as exc:
                out.append(("error", type(exc).__name__))
        return Execution("ok", value=out)


def test_evaluate_cases_batches_and_compares_modes():
    cases = [{"args": [i], "expected": {"$tuple": [i, i * 0.1]}} for i in range(250)]
    runner = LocalRunner()
    assert evaluate_cases("def f(i):\n    return (i, i / 10)\n", "f", cases, runner=runner).passed
    assert runner.batches == [100, 100, 50]
    assert not evaluate_cases("def f(i):\n    return [i, i / 10]\n", "f", cases, runner=LocalRunner()).passed
    sets = [{"args": [[3, 1, 3]], "expected": {"$set": [1, 3]}, "compare": "set"}]
    assert evaluate_cases("def f(x):\n    return list(dict.fromkeys(x))\n", "f", sets, runner=LocalRunner()).passed
    seq = [{"args": [[2, 1]], "expected": [1, 2], "compare": "sequence"}]
    assert evaluate_cases("def f(x):\n    return tuple(sorted(x))\n", "f", seq, runner=LocalRunner()).passed
    verdict = evaluate_cases("def f(x):\n    return 1 // x\n", "f", [{"args": [0], "expected": 0}], runner=LocalRunner())
    assert not verdict.passed and "ZeroDivisionError" in verdict.detail
    for bad in ([], [{"args": [1]}], [{"args": [1], "expected": 1, "compare": "fuzzy"}]):
        assert evaluate_cases("def f(x): return x", "f", bad, runner=LocalRunner()).status == "error"


@pytest.fixture(scope="module")
def sandbox():
    runner = BubblewrapRunner()
    result = evaluate_function("def f():return 1", "f", [{"expected": 1}], runner=runner)
    if result.status == "unavailable":
        if os.environ.get("LLM_REQUIRE_SANDBOX_TESTS") == "1":
            pytest.fail("required sandbox integration tests unavailable: " + result.detail)
        pytest.skip("bubblewrap namespaces/libseccomp unavailable")
    return runner


def test_real_sandbox_batch_handles_types_errors_and_forgery(sandbox):
    cases = [{"args": [{"$tuple": [1, 2]}], "expected": {"$set": [1, 2]}},
             {"args": [{"$tuple": []}], "expected": {"$set": []}}]
    assert evaluate_cases("def f(t):\n    assert isinstance(t, tuple)\n    return set(t)\n", "f", cases,
                          runner=sandbox).passed
    for payload in (b'{"passed": true}', b'{"values": [{"ok": 1}]}',
                    b'{"values": [{"ok": {"$set": [9]}}, {"ok": {"$set": [9]}}]}'):
        forged = f"import os\nos.write(1, {payload!r})\nos._exit(0)\ndef f(t):\n    return None\n"
        assert not evaluate_cases(forged, "f", cases, runner=sandbox).passed
    verdict = evaluate_cases("def f(t):\n    return t[5]\n", "f", cases, runner=sandbox)
    assert not verdict.passed and "IndexError" in verdict.detail
    slow = evaluate_cases("def f(t):\n    while True:\n        pass\n", "f", cases, runner=sandbox, timeout=2)
    assert slow.status == "failed"
    code = ("import json,os\ndef f(t):\n    r = json.load(open('/job/request.json'))\n"
            "    return sorted(r), sorted(os.listdir('/job'))\n")
    expected = [{"args": [1], "expected": {"$tuple": [["calls", "function"], ["code_worker.py", "request.json", "solution.py"]]}}]
    assert evaluate_cases(code, "f", expected, runner=sandbox).passed


def test_real_sandbox_truth_identity_and_per_call_cpu(sandbox):
    regex = "import re\ndef f(s):\n    return re.search('^a', s)\n"
    truthy = [{"args": ["abc"], "expected": True, "compare": "truth"},
              {"args": ["xyz"], "expected": False, "compare": "truth"}]
    assert evaluate_cases(regex, "f", truthy, runner=sandbox).passed
    one = "def f(x):\n    return 1\n"
    assert evaluate_cases(one, "f", [{"args": [0], "expected": True}], runner=sandbox).passed
    assert not evaluate_cases(one, "f", [{"args": [0], "expected": True, "compare": "is"}], runner=sandbox).passed
    burn = "import time\ndef f(x):\n    end = time.process_time() + 4\n    while time.process_time() < end:\n        pass\n    return x\n"
    cases = [{"args": [i], "expected": i} for i in range(3)]
    assert evaluate_cases(burn, "f", cases, runner=sandbox, timeout=120).passed


def test_frozen_manifest_pins_sources_and_counts():
    manifest = json.loads(B.FROZEN.read_text())
    assert set(manifest["suites"]) == set(B.SUITES)
    for name, src in B.SOURCES.items():
        assert manifest["sources"][name]["sha256"] == src["sha256"]
    assert manifest["suites"]["humaneval"]["tasks"] >= 150 and manifest["suites"]["mbpp"]["tasks"] >= 370
    assert all(len(m["sha256"]) == 64 for m in manifest["suites"].values())


def test_bench_run_program_extraction():
    from evals import bench_run as R
    prompt = "def add(a, b):\n    \"\"\"Add.\"\"\"\n"
    assert R.humaneval_program(prompt, "    return a + b\n\n\ndef other():\n    pass\n") == prompt + "    return a + b\n\n"
    assert R.mbpp_program("def f(x):\n    return x\n```\nassert f(1) == 1") == "def f(x):\n    return x\n"
    assert R.mbpp_program("Sure:\n```python\ndef f(x):\n    return x\n```") == "def f(x):\n    return x\n"
    assert R.chat_program("```python\ndef f(x):\n    return x\n") == "def f(x):\n    return x\n"
    assert R.parses("def f(:\n") is False and R.parses("x = 1\n")
    text, special = R.model_input({"prompt": "Write f."}, "mbpp", chat=True)
    assert special and text.startswith("<|user|>") and text.endswith("<|assistant|>")
    assert R.model_input({"prompt": "Write f."}, "mbpp", chat=False) == ("Write f.\n```python\n", False)
    rows = [{"suite": "mbpp", "passed": True, "parses": True, "status": "passed"},
            {"suite": "mbpp", "passed": False, "parses": False, "status": "syntax-error"}]
    assert R.summarize(rows)["mbpp"] == {"tasks": 2, "passed": 1, "pass_at_1": 0.5, "parses": 0.5, "unavailable": 0}


def test_preview_baseline_freezes_hashes_and_leaves_production_alone():
    from evals import preview as V
    preview = json.loads(V.PREVIEW.read_text())
    assert preview["frozen"] == V.frozen_hashes()
    assert preview["targets"]["humaneval_pass_at_1"]["min"] == 0.10
    assert preview["targets"]["repo_code_tasks"] == {"min_correct": 10, "n": 18}
    for name, entry in preview["measured"].items():
        path = V.PROGRAMS / f"programs-{name}.json"
        assert V.sha256(path) == entry["programs_sha256"]
        assert json.loads(path.read_text())["meta"]["checkpoint_sha256"] == entry["checkpoint_sha256"]


def test_canonical_outcomes_are_classified():
    from evals.bench import classify
    from evals.code_executor import Verdict
    assert classify(Verdict("passed", True, 3, 3)) == "passed"
    assert classify(Verdict("failed", False, 1, 3, "case 2 output mismatch")) == "output_mismatch"
    assert classify(Verdict("failed", False, 0, 3, "case 1 raised ValueError")) == "exception"
    assert classify(Verdict("failed", False, 0, 3, "suite timeout")) == "timeout"
    assert classify(Verdict("failed", False, 0, 3, "worker exited unsuccessfully")) == "worker_failure"
    assert classify(Verdict("error", False, 0, 3, "execution infrastructure error")) == "sandbox_error"


def test_suite_lines_keep_dict_insertion_order():
    case, _ = _assert("assert f({'p': 1, 'A': 2, 'B': 3}) == False")
    line = json.dumps(case, ensure_ascii=False)
    assert list(json.loads(line)["args"][0]) == ["p", "A", "B"]
