import argparse
import ast
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


def analyze(root):
    root = Path(root)
    reports = {}
    for path in sorted(root.glob("programs-*.json")):
        label = path.stem.removeprefix("programs-")
        score_path = root / f"scores-{label}.json"
        programs, scores = json.loads(path.read_text()), json.loads(score_path.read_text())
        if programs["meta"]["checkpoint_sha256"] != scores["meta"]["checkpoint_sha256"]:
            raise ValueError("program/score checkpoint identity mismatch")
        syntax, shapes, failures = Counter(), Counter(), Counter()
        for row in programs["programs"].values():
            try:
                tree = ast.parse(row["program"])
            except (SyntaxError, ValueError, RecursionError, MemoryError) as error:
                category = "indentation" if isinstance(error, IndentationError) else "syntax_or_parser_error"
                syntax[category] += 1
            else:
                shapes["parseable"] += 1
                if not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in tree.body):
                    shapes["no_top_level_function"] += 1
                if not tree.body:
                    shapes["empty_module"] += 1
        for row in scores["results"]:
            detail = row.get("detail", "")
            if row["status"] == "passed":
                category = "passed"
            elif row["status"] == "syntax-error":
                category = "syntax_error"
            elif "output mismatch" in detail:
                category = "wrong_output"
            elif match := re.search(r"raised (\w+)", detail):
                category = "runtime_" + match.group(1)
            elif detail == "worker exited unsuccessfully":
                category = "worker_exit_cause_not_established"
            else:
                category = "other_" + row["status"]
            failures[category] += 1
        reports[label] = {"unique_generated_programs": len(programs["programs"]),
                          "syntax_counts": dict(syntax), "static_shape_counts": dict(shapes),
                          "suite_evaluations": len(scores["results"]), "outcomes": dict(failures),
                          "inputs_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                            for p in (path, score_path)}}
    if not reports:
        raise ValueError("no program/score pairs found")
    return {"reports": reports, "scope": "aggregate diagnostics only; no benchmark prompts or answers emitted",
            "limitations": ["Base/plus suite evaluations reuse programs and are not independent problems.",
                            "Worker exits do not establish whether a failure was logic, runtime or isolation.",
                            "No independent cybersecurity capability test is inferred from Python failures."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Classify existing model failures without executing generated code.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    report = analyze(args.root)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
