"""Independent behavioral regression witnesses for a deliberately narrow fix.

Supported: a top-level one-argument function returning
hashlib.md5/sha1/sha256(argument).hexdigest(). The source function is executed in
our JSON sandbox; expected digests and verdicts stay in the controller. Other
shapes/rules are unsupported and cannot receive a security-verified verdict.
This is a digest-algorithm witness, not proof of password-storage suitability.
"""
import ast
import hashlib
from pathlib import Path

from evals.code_executor import evaluate_function

INPUTS = ["", "hello", "' OR 1=1 --", "安全", "a" * 1024]


def witness_source(path, line):
    source = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for fn in tree.body:
        if not isinstance(fn, ast.FunctionDef) or not fn.lineno <= line <= fn.end_lineno:
            continue
        if fn.decorator_list or len(fn.args.args) != 1 or fn.args.posonlyargs or fn.args.kwonlyargs or fn.args.vararg or fn.args.kwarg or fn.args.defaults or len(fn.body) != 1:
            return None
        arg = fn.args.args[0].arg
        expected = f"return hashlib.ALGORITHM({arg}).hexdigest()"
        normalized = ast.unparse(fn.body[0])
        if normalized not in [expected.replace("ALGORITHM", name) for name in ("md5", "sha1", "sha256")]:
            return None
        # Reconstruct just this reviewed grammar, not decorators/import side effects.
        return ("import hashlib\n" + ast.unparse(fn) + "\n\ndef security_probe(value):\n"
                f"    return {fn.name}(value.encode('utf-8'))\n")
    return None


def reproduce(path, line, *, runner=None):
    source = witness_source(path, line)
    if source is None:
        return {"status": "unsupported", "detail": "no reviewed behavioral witness for this code shape"}
    cases = [{"args": [value], "expected": hashlib.sha256(value.encode()).hexdigest()}
             for value in INPUTS]
    verdict = evaluate_function(source, "security_probe", cases, runner=runner)
    return {"status": verdict.status, "cases_passed": verdict.cases_passed,
            "cases_total": verdict.cases_total, "detail": verdict.detail,
            "test_source": source, "cases": cases}


def main():
    """Replay exported locations against fresh source and controller-owned cases."""
    import argparse
    import json
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--evidence", required=True)
    args = ap.parse_args()
    try:
        root = Path(args.repo).resolve(strict=True)
        locations = json.loads(Path(args.evidence).read_text())
        if not isinstance(locations, list) or not 1 <= len(locations) <= 100:
            raise ValueError("one to 100 witness locations are required")
        results = []
        for item in locations:
            rel, line = item["file"], item["line"]
            if not isinstance(rel, str) or Path(rel).is_absolute() or type(line) is not int or line < 1:
                raise ValueError("invalid witness location")
            target = root / rel
            if target.is_symlink() or not target.resolve(strict=True).is_relative_to(root):
                raise ValueError("witness path escapes repository")
            # Never trust exported expected answers, source or verdicts. Rebuild
            # from this reviewed controller and the current repository source.
            result = reproduce(target, line)
            results.append({"file": rel, "line": line, "status": result["status"],
                            "detail": result["detail"]})
        print(json.dumps(results, indent=2))
        return 0 if all(r["status"] == "passed" for r in results) else 1
    except (ValueError, OSError, KeyError, TypeError) as exc:
        ap.exit(1, f"SECURITY VERIFICATION BLOCKED: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
