"""Offline AST-only diagnosis of saved generations; never execute or repair code."""

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path

from evals.bench_run import STOPS, _cut, chat_program, mbpp_program


def syntax(source):
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        message = error.msg.lower()
        if isinstance(error, IndentationError):
            category = "indentation"
        elif any(term in message for term in ("never closed", "unterminated", "unexpected eof", "incomplete input")):
            category = "unclosed_construct"
        else:
            category = "other_syntax"
        # Do not retain source lines or interpreter messages containing source text.
        return {"parses": False, "category": category, "line": error.lineno}
    except (ValueError, RecursionError, MemoryError):
        return {"parses": False, "category": "parser_limit_or_invalid_input", "line": None}
    functions = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    return {"parses": True, "category": "valid", "functions": len(functions),
            "empty_module": not tree.body}


def diagnose(generated):
    chat = generated["meta"]["decoding"]["chat"]
    rows = {}
    for key, item in sorted(generated["programs"].items()):
        suite = key.split(":", 1)[0]
        if suite not in ("humaneval", "mbpp"):
            raise ValueError("unsupported suite")
        completion, program = item["completion"], item["program"]
        result = syntax(program)
        if suite == "humaneval" and not chat:
            used = _cut(completion, STOPS)
            # The saved program includes its prompt. Recover that prefix without
            # loading benchmark cases, answers or any private holdout.
            if not program.endswith(used):
                raise ValueError("saved HumanEval program disagrees with extraction")
            prefix = program[:len(program) - len(used)] if used else program
            uncut = prefix + completion
            changed = used != completion
        else:
            expected = chat_program(completion) if chat else mbpp_program(completion)
            if program != expected:
                raise ValueError("saved program disagrees with extraction")
            uncut = completion
            changed = program.strip() != completion.strip()
        uncut_result = syntax(uncut)
        rows[key] = {**result, "program_sha256": hashlib.sha256(program.encode()).hexdigest(),
                     "completion_characters": len(completion), "extraction_changed": changed,
                     "completion_starts_indented": completion.startswith((" ", "\t")),
                     "completion_contains_fence": "```" in completion,
                     "uncut_parses": uncut_result["parses"],
                     "extraction_valid_to_invalid": uncut_result["parses"] and not result["parses"],
                     "extraction_invalid_to_valid": not uncut_result["parses"] and result["parses"]}
    summary = dict(Counter(row["category"] for row in rows.values()))
    return {"checkpoint_sha256": generated["meta"]["checkpoint_sha256"], "chat": chat,
            "programs": len(rows), "categories": summary,
            "empty_modules": sum(row.get("empty_module", False) for row in rows.values()),
            "valid_without_functions": sum(row["parses"] and row.get("functions", 0) == 0 for row in rows.values()),
            "by_suite": {suite: dict(Counter(row["category"] for key, row in rows.items()
                                             if key.startswith(suite + ":")))
                         for suite in ("humaneval", "mbpp")},
            "extraction_valid_to_invalid": sum(row["extraction_valid_to_invalid"] for row in rows.values()),
            "extraction_invalid_to_valid": sum(row["extraction_invalid_to_valid"] for row in rows.values()),
            "rows": rows}


def report(directory):
    directory = Path(directory)
    models, hashes = {}, {}
    for label in ("base", "control", "treatment"):
        for mode in ("completion", "chat"):
            path = directory / f"programs-{label}-{mode}.json"
            generated = json.loads(path.read_text())
            if generated["meta"]["decoding"]["chat"] != (mode == "chat"):
                raise ValueError("filename and decoding mode disagree")
            models[f"{label}-{mode}"] = diagnose(generated)
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    transitions = {}
    for label in ("control", "treatment"):
        for mode in ("completion", "chat"):
            base, candidate = [models[f"{name}-{mode}"]["rows"] for name in ("base", label)]
            if base.keys() != candidate.keys():
                raise ValueError("task membership differs")
            transitions[f"{label}-{mode}"] = {
                "valid_to_invalid": [key for key in base if base[key]["parses"] and not candidate[key]["parses"]],
                "invalid_to_valid": [key for key in base if not base[key]["parses"] and candidate[key]["parses"]]}
    return {"schema": "syntax_diagnosis_v1", "input_sha256": hashes, "models": models,
            "transitions_vs_base": transitions,
            "limitations": ["AST parsing only; no execution, repairs or benchmark score changes.",
                            "Unclosed constructs do not prove token-budget exhaustion: stop reasons and token counts were not saved.",
                            "Uncut chat/MBPP text can include prose/fences; its parse rate is not a capability score.",
                            "Observed benchmark outputs are diagnostic only and must not become training examples."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--programs-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    Path(args.out).write_text(json.dumps(report(args.programs_dir), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
