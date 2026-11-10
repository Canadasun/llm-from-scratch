"""Evaluation-only coding benchmarks converted to sandboxed JSON function cases.

    python -m evals.bench build --raw ~/llmfs-data/bench-raw --out ~/llmfs-data/bench
    python -m evals.bench verify --out ~/llmfs-data/bench

Sources are pinned by Hugging Face revision and file sha256. Test files are only
parsed with ast; literals are read with ast.literal_eval and nothing from a
source is executed. Tasks whose tests cannot be expressed as literal cases are
excluded and listed with a reason, never scored with a weaker check.
"""

import argparse
import ast
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from evals.code_worker import encode_value  # noqa: E402

HF = "https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}"
SOURCES = {
    "humaneval": {"repo": "openai/openai_humaneval", "revision": "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
                  "path": "openai_humaneval/test-00000-of-00001.parquet", "licence": "MIT",
                  "sha256": "2f2871a15fbc95b6c683043359f4ed8e144c5a1c4f24f25f66bc51f598dfcfb6", "file": "he.parquet"},
    "humaneval_plus": {"repo": "evalplus/humanevalplus", "revision": "d32357cf319e50e9c8d8dab5ea876c72b0fd321b",
                       "path": "data/test-00000-of-00001-5973903632b82d40.parquet", "licence": "Apache-2.0",
                       "sha256": "4436f5c03d77c17e0cbc57543b90665b5c1266f55a43992a5ed7922cd34a7558",
                       "file": "heplus.parquet"},
    "mbpp_plus": {"repo": "evalplus/mbppplus", "revision": "b2d74c91837c3f2a20c1299ae98133cbe7cfa077",
                  "path": "data/test-00000-of-00001-d5781c9c51e02795.parquet", "licence": "Apache-2.0",
                  "sha256": "dc20030b3788fccf617444edcb34138ef13d7e4fafd17bfcb8c1279dbb12399b",
                  "file": "mbppplus.parquet"},
}
MAX_CASES = 1000
SUITES = ("humaneval", "humaneval_plus", "mbpp", "mbpp_plus")


class Unsupported(ValueError):
    pass


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def fetch(raw_dir):
    raw = Path(raw_dir).expanduser()
    raw.mkdir(parents=True, exist_ok=True)
    for name, src in SOURCES.items():
        path = raw / src["file"]
        if not path.exists():
            url = HF.format(**src)
            with urllib.request.urlopen(url, timeout=120) as resp:
                path.write_bytes(resp.read())
        digest = sha256_bytes(path.read_bytes())
        if digest != src["sha256"]:
            raise ValueError(f"{name}: sha256 {digest} does not match the pinned {src['sha256']}")
    return raw


def _rows(raw, name):
    import pyarrow.parquet as pq
    return pq.read_table(raw / SOURCES[name]["file"]).to_pylist()


_FOLD_OPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
             ast.Div: lambda a, b: a / b, ast.FloorDiv: lambda a, b: a // b, ast.Mod: lambda a, b: a % b,
             ast.Pow: lambda a, b: a ** b}
_NAMES = {"inf": float("inf"), "nan": float("nan")}


def _fold(node, depth=0):
    if depth > 64:
        raise Unsupported("literal nests too deeply")
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id in _NAMES:
        return _NAMES[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _fold(node.operand, depth + 1)
        if not isinstance(value, (int, float, complex)) or isinstance(value, bool):
            raise Unsupported("non-literal value")
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp) and type(node.op) in _FOLD_OPS:
        a, b = _fold(node.left, depth + 1), _fold(node.right, depth + 1)
        numeric = all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in (a, b))
        if isinstance(node.op, ast.Pow) and not (numeric and abs(b) <= 128 and abs(a) <= 10**6):
            raise Unsupported("non-literal value")
        if isinstance(node.op, ast.Mult) and not numeric:
            n = b if isinstance(b, int) else a
            if not isinstance(n, int) or isinstance(n, bool) or abs(n) > 10**5:
                raise Unsupported("non-literal value")
        elif not numeric and not (isinstance(node.op, ast.Add) and type(a) is type(b)
                                  and isinstance(a, (str, list, tuple))):
            raise Unsupported("non-literal value")
        try:
            return _FOLD_OPS[type(node.op)](a, b)
        except (ArithmeticError, TypeError, ValueError) as exc:
            raise Unsupported("non-literal value") from exc
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        items = [_fold(e, depth + 1) for e in node.elts]
        return {ast.List: list, ast.Tuple: tuple, ast.Set: set}[type(node)](items)
    if isinstance(node, ast.Dict) and None not in node.keys:
        return {_fold(k, depth + 1): _fold(v, depth + 1) for k, v in zip(node.keys, node.values)}
    raise Unsupported("non-literal value")


def _literal(node):
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        pass
    try:
        return _fold(node)
    except (TypeError, RecursionError) as exc:
        raise Unsupported("non-literal value") from exc


def _call_case(call, name):
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.func.id != name:
        raise Unsupported("assertion does not call the function directly")
    if any(isinstance(a, ast.Starred) for a in call.args) or any(k.arg is None for k in call.keywords):
        raise Unsupported("starred arguments")
    return ([_encode(_literal(a)) for a in call.args],
            {k.arg: _encode(_literal(k.value)) for k in call.keywords})


def _encode(value):
    try:
        return encode_value(value)
    except (TypeError, ValueError) as exc:
        raise Unsupported(f"unencodable value: {exc}") from exc


def _case(args, kwargs, expected, compare="exact"):
    case = {"args": args, "expected": _encode(expected)}
    if kwargs:
        case["kwargs"] = kwargs
    if compare != "exact":
        case["compare"] = compare
    return case


def _wrapped(node, name):
    if (isinstance(node, ast.Call) and getattr(node.func, "id", None) in ("tuple", "list", "set")
            and len(node.args) == 1 and not node.keywords):
        inner = node.args[0]
        if isinstance(inner, ast.Call) and getattr(inner.func, "id", None) == name:
            return node.func.id, inner
    return None, None


def _container(node, kind):
    if isinstance(node, ast.Call) and getattr(node.func, "id", None) == kind and len(node.args) == 1:
        node = node.args[0]
    value = _literal(node)
    if not isinstance(value, (list, tuple, set, frozenset)):
        raise Unsupported("unsupported comparison")
    return set(value) if kind == "set" else list(value)


def assert_case(test, name):
    """One `assert` statement to (case, atol). Supports ==, truthiness, abs(x - y) < tol, math.isclose."""
    if isinstance(test, ast.Compare) and len(test.ops) == 1:
        left, op, right = test.left, test.ops[0], test.comparators[0]
        if isinstance(op, ast.Is) and isinstance(right, ast.Constant) and isinstance(right.value, bool):
            args, kwargs = _call_case(left, name)
            return _case(args, kwargs, right.value, "is"), 0.0
        if isinstance(op, ast.Eq):
            kind, inner = _wrapped(left, name)
            if kind:
                args, kwargs = _call_case(inner, name)
                return _case(args, kwargs, _container(right, kind), "set" if kind == "set" else "sequence"), 0.0
            if isinstance(left, ast.Call) and getattr(left.func, "id", None) == name:
                args, kwargs = _call_case(left, name)
                return _case(args, kwargs, _literal(right)), 0.0
            if isinstance(right, ast.Call) and getattr(right.func, "id", None) == name:
                args, kwargs = _call_case(right, name)
                return _case(args, kwargs, _literal(left)), 0.0
        if (isinstance(op, (ast.Lt, ast.LtE)) and isinstance(left, ast.Call) and getattr(left.func, "id", None) == "abs"
                and len(left.args) == 1 and isinstance(left.args[0], ast.BinOp) and isinstance(left.args[0].op, ast.Sub)):
            inner = left.args[0]
            args, kwargs = _call_case(inner.left, name)
            expected = _literal(inner.right)
            if not isinstance(expected, (int, float)) or isinstance(expected, bool):
                raise Unsupported("tolerance check on a non-number")
            return _case(args, kwargs, float(expected)), float(_literal(right))
        raise Unsupported("unsupported comparison")
    if isinstance(test, ast.Call) and isinstance(test.func, ast.Attribute) and test.func.attr == "isclose" \
            and getattr(test.func.value, "id", None) == "math" and len(test.args) == 2:
        args, kwargs = _call_case(test.args[0], name)
        expected = float(_literal(test.args[1]))
        tols = {k.arg: float(_literal(k.value)) for k in test.keywords}
        atol = max(tols.get("abs_tol", 0.0), tols.get("rel_tol", 1e-9) * abs(expected))
        return _case(args, kwargs, expected), atol
    if isinstance(test, ast.Call) and getattr(test.func, "id", None) == name:
        args, kwargs = _call_case(test, name)
        return _case(args, kwargs, True, "truth"), 0.0
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        args, kwargs = _call_case(test.operand, name)
        return _case(args, kwargs, False, "truth"), 0.0
    raise Unsupported("unsupported assertion")


def assert_cases(statements, name):
    cases, atol = [], 0.0
    for stmt in statements:
        if isinstance(stmt, ast.Pass) or (isinstance(stmt, ast.Expr) and (
                isinstance(stmt.value, ast.Constant)
                or (isinstance(stmt.value, ast.Name) and stmt.value.id == "print"))):
            continue
        if not isinstance(stmt, ast.Assert):
            raise Unsupported(f"unsupported statement {type(stmt).__name__}")
        if isinstance(stmt.test, ast.Constant) and stmt.test.value is True:
            continue
        case, tol = assert_case(stmt.test, name)
        cases.append(case)
        atol = max(atol, tol)
    if not cases:
        raise Unsupported("no assertions")
    return cases, atol


def _assertion_mode(tree):
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "assertion"), None)
    if fn is None:
        raise Unsupported("no assertion helper")
    src = ast.unparse(fn)
    if "is not None" in src or "isinstance(out" in src:
        raise Unsupported("non-standard result check")
    return "set" if "set(out)" in src else "exact"


def literal_io_tasks(row):
    tree = ast.parse(row["test"])
    check = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "check"), None)
    scope = check if check is not None else tree
    found = {}
    for node in ast.walk(scope):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("inputs", "results"):
                found[node.targets[0].id] = _literal(node.value)
    if set(found) != {"inputs", "results"}:
        raise Unsupported("expected values are computed by a reference function")
    if len(found["inputs"]) != len(found["results"]):
        raise Unsupported("inputs and results differ in length")
    mode = _assertion_mode(tree)
    calls = [n for n in ast.walk(scope) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "assertion"]
    if len(calls) != 1 or len(calls[0].args) != 3:
        raise Unsupported("non-standard result check")
    first = calls[0].args[0]
    if not (isinstance(first, ast.Call) and isinstance(first.func, ast.Name)
            and len(first.args) == 1 and isinstance(first.args[0], ast.Starred)):
        raise Unsupported("non-standard candidate call")
    atol = float(_literal(calls[0].args[2]))
    cases = []
    for inp, exp in zip(found["inputs"][:MAX_CASES], found["results"][:MAX_CASES]):
        if mode == "set":
            if not isinstance(exp, (list, tuple, set, frozenset)):
                raise Unsupported("set comparison on a non-collection")
            exp = set(exp)
        cases.append(_case([_encode(a) for a in inp], {}, exp, mode))
    return cases, atol, len(found["inputs"])


def _mbpp_entry(row):
    first = ast.parse(row["test_list"][0] if isinstance(row["test_list"], list)
                      else ast.literal_eval(row["test_list"])[0]).body[0]
    for node in ast.walk(first):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id not in {
                "set", "sorted", "list", "tuple", "abs", "isclose", "round", "str", "int", "float", "len"}:
            return node.func.id
    raise Unsupported("cannot find the function name")


def _tests(row):
    tests = row["test_list"]
    return tests if isinstance(tests, list) else ast.literal_eval(tests)


def mbpp_prompt(row, entry):
    return (f"{row['prompt'].strip()}\nYour code should pass this test:\n{_tests(row)[0].strip()}\n\n"
            f"Write the Python function `{entry}`.\n")


def build(raw_dir, out_dir):
    raw = fetch(raw_dir)
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    suites = {s: [] for s in SUITES}
    excluded = {s: {} for s in SUITES}
    reference = []
    plus_truncated = {}
    for row in _rows(raw, "humaneval"):
        tid = row["task_id"]
        reference.append({"id": tid, "text": row["prompt"] + row["canonical_solution"]})
        try:
            tree = ast.parse(row["test"])
            check = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "check")
            cases, atol = assert_cases(check.body, "candidate")
        except (Unsupported, StopIteration, SyntaxError) as exc:
            excluded["humaneval"][tid] = str(exc) or "no check function"
            continue
        suites["humaneval"].append({"id": tid, "prompt": row["prompt"], "entry_point": row["entry_point"],
                                    "atol": atol, "cases": cases})
    for row in _rows(raw, "humaneval_plus"):
        tid = row["task_id"]
        try:
            cases, atol, total = literal_io_tasks(row)
        except (Unsupported, SyntaxError) as exc:
            excluded["humaneval_plus"][tid] = str(exc)
            continue
        if total > len(cases):
            plus_truncated[tid] = total
        suites["humaneval_plus"].append({"id": tid, "prompt": row["prompt"], "entry_point": row["entry_point"],
                                         "atol": atol, "cases": cases})
    for row in _rows(raw, "mbpp_plus"):
        tid = f"Mbpp/{row['task_id']}"
        reference.append({"id": tid, "text": row["prompt"] + "\n" + row["code"]})
        try:
            entry = _mbpp_entry(row)
        except (Unsupported, SyntaxError, ValueError, IndexError) as exc:
            excluded["mbpp"][tid] = excluded["mbpp_plus"][tid] = str(exc)
            continue
        prompt = mbpp_prompt(row, entry)
        try:
            body = [ast.parse(t).body[0] for t in _tests(row)]
            cases, atol = assert_cases(body, entry)
            suites["mbpp"].append({"id": tid, "prompt": prompt, "entry_point": entry, "atol": atol, "cases": cases})
        except (Unsupported, SyntaxError, IndexError) as exc:
            excluded["mbpp"][tid] = str(exc)
        try:
            cases, atol, total = literal_io_tasks(row)
            if total > len(cases):
                plus_truncated[tid] = total
            suites["mbpp_plus"].append({"id": tid, "prompt": prompt, "entry_point": entry, "atol": atol,
                                        "cases": cases})
        except (Unsupported, SyntaxError) as exc:
            excluded["mbpp_plus"][tid] = str(exc)
    files = {}
    for name, tasks in suites.items():
        data = "".join(json.dumps(t, ensure_ascii=False) + "\n" for t in tasks)
        (out / f"{name}.jsonl").write_text(data, encoding="utf-8")
        files[name] = {"tasks": len(tasks), "cases": sum(len(t["cases"]) for t in tasks),
                       "sha256": sha256_bytes(data.encode("utf-8")), "excluded": dict(sorted(excluded[name].items()))}
    ref = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in reference)
    (out / "reference_texts.jsonl").write_text(ref, encoding="utf-8")
    manifest = {"schema": 1, "max_cases_per_task": MAX_CASES,
                "sources": {k: {f: v[f] for f in ("repo", "revision", "path", "licence", "sha256")}
                            for k, v in SOURCES.items()},
                "mbpp_note": "mbpp uses the original three asserts as corrected in evalplus/mbppplus; "
                             "mbpp_plus uses its extended literal inputs",
                "truncated_to_max_cases": dict(sorted(plus_truncated.items())),
                "suites": files,
                "reference_texts": {"rows": len(reference), "sha256": sha256_bytes(ref.encode("utf-8")),
                                    "use": "decontamination only"}}
    (out / "bench-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify(out_dir, manifest_path):
    out = Path(out_dir).expanduser()
    manifest = json.loads(Path(manifest_path).read_text())
    for name, meta in manifest["suites"].items():
        digest = sha256_bytes((out / f"{name}.jsonl").read_bytes())
        if digest != meta["sha256"]:
            raise ValueError(f"{name}: suite sha256 {digest} does not match the frozen {meta['sha256']}")
    return manifest


def load_suite(out_dir, name, manifest_path=None):
    if manifest_path:
        verify(out_dir, manifest_path)
    path = Path(out_dir).expanduser() / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def canonical(raw_dir, out_dir, workers=4):
    """Run each reference solution through the real sandbox against its converted suite.

    Reference solutions are downloaded code, so this only runs inside the
    bubblewrap sandbox used for model outputs; it is meant for a CI runner.
    """
    from concurrent.futures import ThreadPoolExecutor
    from evals.code_executor import evaluate_cases, sandbox_preflight
    preflight = sandbox_preflight()
    if not preflight["available"]:
        raise SystemExit(f"sandbox unavailable: {preflight['detail']}")
    raw = fetch(raw_dir)
    solutions = {}
    for suite, source in (("humaneval", "humaneval"), ("humaneval_plus", "humaneval_plus")):
        for row in _rows(raw, source):
            solutions[suite, row["task_id"]] = row["prompt"] + row["canonical_solution"]
    for row in _rows(raw, "mbpp_plus"):
        for suite in ("mbpp", "mbpp_plus"):
            solutions[suite, f"Mbpp/{row['task_id']}"] = row["code"]
    jobs = [(name, task) for name in SUITES for task in load_suite(out_dir, name)]

    def run(job):
        name, task = job
        verdict = evaluate_cases(solutions[name, task["id"]], task["entry_point"], task["cases"],
                                 atol=task["atol"], timeout=300)
        return name, task["id"], verdict

    failures = []
    tally = {s: dict.fromkeys(OUTCOMES, 0) for s in SUITES}
    with ThreadPoolExecutor(workers) as pool:
        for name, tid, verdict in pool.map(run, jobs):
            outcome = classify(verdict)
            tally[name][outcome] += 1
            if outcome != "passed":
                failures.append({"suite": name, "id": tid, "outcome": outcome, "status": verdict.status,
                                 "cases_passed": verdict.cases_passed, "cases": verdict.cases_total,
                                 "detail": verdict.detail})
    return {"suites": tally, "failures": failures}


OUTCOMES = ("passed", "output_mismatch", "exception", "timeout", "worker_failure", "sandbox_error")


def classify(verdict):
    detail = verdict.detail or ""
    if verdict.passed:
        return "passed"
    if verdict.status != "failed":
        return "sandbox_error"
    if "timeout" in detail:
        return "timeout"
    if "output mismatch" in detail:
        return "output_mismatch"
    if " raised " in detail:
        return "exception"
    return "worker_failure"


FROZEN = HERE / "data" / "bench-manifest.json"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--raw", required=True)
    b.add_argument("--out", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--out", required=True)
    v.add_argument("--manifest", default=str(FROZEN))
    c = sub.add_parser("canonical")
    c.add_argument("--raw", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--workers", type=int, default=4)
    c.add_argument("--report")
    args = ap.parse_args(argv)
    if args.cmd == "canonical":
        verify(args.out, FROZEN)
        report = canonical(args.raw, args.out, args.workers)
        text = json.dumps(report, indent=2)
        if args.report:
            Path(args.report).write_text(text + "\n")
        print(text)
        return 1 if report["failures"] else 0
    if args.cmd == "build":
        manifest = build(args.raw, args.out)
        print(json.dumps({k: {f: m[f] for f in ("tasks", "cases", "sha256")} | {"excluded": len(m["excluded"])}
                          for k, m in manifest["suites"].items()}, indent=2))
    else:
        verify(args.out, args.manifest)
        print("suites match the frozen manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
