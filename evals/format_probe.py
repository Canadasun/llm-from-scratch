"""Build and verify an original, public family-separated response-format probe."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

from data.format_probe_tasks import TASKS
from evals.code_executor import MAX_CODE_BYTES, evaluate_cases, sandbox_preflight

FORMATS = ("module", "body")


def prompt(task, mode):
    if mode == "module":
        instruction = "Return only a complete Python function with this exact signature, without Markdown fences."
    elif mode == "body":
        instruction = "Return only the function body, indented by four spaces. Do not repeat the signature or use Markdown fences."
    else:
        raise ValueError("unknown format")
    return f"{task.spec}\n{instruction}\n{task.header}"


def assemble(task, mode, response):
    if mode not in FORMATS:
        raise ValueError("unknown format")
    return response if mode == "module" else task.header + response


def check_shape(task, mode, response):
    """No stripping, dedenting, fence extraction, code repair or host execution."""
    source = assemble(task, mode, response)
    if len(source.encode()) > MAX_CODE_BYTES:
        return {"syntax": False, "format": False, "reason": "size_limit"}
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return {"syntax": False, "format": False, "reason": "syntax"}
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        return {"syntax": True, "format": False, "reason": "one_function_required"}
    node = tree.body[0]
    args = node.args
    expected = [name.strip() for name in task.arguments.split(",")]
    if (node.name != task.name or [arg.arg for arg in args.args] != expected
            or args.posonlyargs or args.kwonlyargs or args.vararg or args.kwarg
            or args.defaults or node.decorator_list):
        return {"syntax": True, "format": False, "reason": "signature"}
    # Pass/docstring-only bodies and nested replacement functions are not useful
    # entry-point implementations even though Python accepts their syntax.
    meaningful = [n for n in node.body if not isinstance(n, ast.Pass)
                  and not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
                           and isinstance(n.value.value, str))]
    if not meaningful or any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for n in node.body):
        return {"syntax": True, "format": False, "reason": "missing_implementation"}
    return {"syntax": True, "format": True, "reason": "accepted_shape"}


def score_response(task, mode, response):
    shape = check_shape(task, mode, response)
    if not shape["format"]:
        return {**shape, "functional": False, "status": "format_rejected"}
    verdict = evaluate_cases(assemble(task, mode, response), task.name, task.cases, timeout=10)
    if verdict.status in ("unavailable", "error"):
        raise RuntimeError(f"sandbox measurement failed: {verdict.status}")
    return {**shape, "functional": verdict.passed, "status": verdict.status,
            "cases_passed": verdict.cases_passed, "cases_total": verdict.cases_total}


def build():
    families = [task.family for task in TASKS]
    if len(set(families)) != len(families):
        raise ValueError("family must occur once; variants inherit its split")
    rows, structures = [], {}
    for task in TASKS:
        if task.split not in ("train", "evaluation"):
            raise ValueError("unknown split")
        tree = ast.parse(task.header + task.body)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                node.name = "function"
            elif isinstance(node, ast.Name):
                node.id = "variable"
            elif isinstance(node, ast.arg):
                node.arg = "argument"
        key = ast.dump(tree, include_attributes=False)
        if key in structures and structures[key] != task.split:
            raise ValueError("renamed reference structure overlaps splits")
        structures[key] = task.split
        for mode in FORMATS:
            rows.append({"id": f"{task.family}:{mode}", "family": task.family,
                         "split": task.split, "format": mode, "entry_point": task.name,
                         "prompt": prompt(task, mode)})
    source = Path(__file__).resolve().parents[1] / "data" / "format_probe_tasks.py"
    return {"schema": "original_format_probe_v1", "provenance": "Project-authored public diagnostic; no benchmark/private data imported.",
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "families": len(families), "cases": sum(len(task.cases) for task in TASKS),
            "rows": rows, "limitations": ["Public small diagnostic, not independent final qualification.",
                "Related programming concepts can overlap across families; family IDs do not prove semantic independence.",
                "No model outputs have been measured. Evaluation families must not enter training."]}


def verify():
    suite = build()  # Refuse an invalid split before executing any reference.
    preflight = sandbox_preflight()
    if not preflight["available"]:
        raise RuntimeError("Linux bubblewrap/libseccomp sandbox required; no host fallback")
    results = []
    for task in TASKS:
        for mode in FORMATS:
            reference = task.header + task.body if mode == "module" else task.body
            wrong_name = "def wrong_entry():\n    return None\n"
            wrong_value = task.header + "    return None\n" if mode == "module" else "    return None\n"
            controls = {"reference": reference, "empty": "", "wrong_entry": wrong_name,
                        "wrong_value": wrong_value,
                        "pass_only": task.header + "    pass\n" if mode == "module" else "    pass\n"}
            for name, response in controls.items():
                result = score_response(task, mode, response)
                if result["functional"] != (name == "reference"):
                    raise AssertionError(f"{task.family}:{mode}:{name}: incorrect control verdict")
                results.append({"family": task.family, "split": task.split, "format": mode,
                                "control": name, **result})
    root = Path(__file__).resolve().parents[1]
    return {"schema": "format_probe_verification_v1", "suite_sha256": hashlib.sha256(
                json.dumps(suite, sort_keys=True).encode()).hexdigest(),
            "implementation_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                      for name in ("evals/format_probe.py", "evals/code_executor.py", "evals/code_worker.py")},
            "sandbox": preflight, "passed": True, "results": results,
            "scope": "Reference/control verification only; no model capability measurement."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = verify() if args.verify else build()
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
