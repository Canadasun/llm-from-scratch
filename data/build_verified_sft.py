import argparse
import ast
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from data import decontaminate
from data.contamination import find_contamination
from data.verified_tasks import TASKS


WORKER = '''
import copy
import json
import sys
payload = json.load(sys.stdin)
namespace = {}
exec(compile(payload["source"], "<project-authored-task>", "exec"), namespace)
function = namespace[payload["function"]]
results = []
for case in payload["cases"]:
    args = copy.deepcopy(case["args"])
    before = copy.deepcopy(args)
    try:
        value = function(*args)
    except Exception as error:
        passed = type(error).__name__ == case.get("raises")
    else:
        passed = "raises" not in case and json.dumps(value, sort_keys=True) == json.dumps(case["expected"], sort_keys=True)
    results.append(passed and args == before)
json.dump({"results": results}, sys.stdout)
'''
FILES = {"train": "train.jsonl", "validation": "validation.jsonl", "test": "holdout_test.jsonl"}
PROMPTS = (
    "Write the Python implementation described below. Return only executable Python code, without Markdown or explanation.",
    "Implement this utility in Python. Output only the function definition; do not provide a review, refusal or summary.",
    "Complete this development task with Python code only. Preserve the specified signature and edge-case behavior.",
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def function_name(source):
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("authored task must contain exactly one function definition")
    return tree.body[0].name


def run_cases(source, cases):
    payload = {"source": source, "function": function_name(source), "cases": cases}
    process = subprocess.run([sys.executable, "-I", "-c", WORKER], input=json.dumps(payload),
                             capture_output=True, text=True, timeout=5)
    if process.returncode:
        raise ValueError(f"authored task runner failed: {process.stderr[-1000:]}")
    return json.loads(process.stdout)["results"]


def validate_task(task):
    if len(task.cases) < 4 or len({json.dumps(c["args"], sort_keys=True) for c in task.cases}) != len(task.cases):
        raise ValueError(f"{task.family}: need at least four distinct test inputs")
    correct = run_cases(task.source, task.cases)
    if not all(correct):
        raise ValueError(f"{task.family}: reference failed cases {[i for i, ok in enumerate(correct) if not ok]}")
    old, new = task.mutation
    if task.source.count(old) != 1:
        raise ValueError(f"{task.family}: mutation must replace exactly one location")
    mutant = task.source.replace(old, new, 1)
    killed = [i for i, ok in enumerate(run_cases(mutant, task.cases)) if not ok]
    if not killed:
        raise ValueError(f"{task.family}: targeted defect survived every test")
    return {"family": task.family, "split": task.split, "cases": len(correct),
            "source_sha256": digest(task.source.encode()),
            "cases_sha256": digest(json.dumps(task.cases, sort_keys=True).encode()),
            "mutation_sha256": digest(mutant.encode()), "mutation_killed_by_cases": killed}


def structural_key(source):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.name = "function"
        elif isinstance(node, ast.Name):
            node.id = "variable"
        elif isinstance(node, ast.arg):
            node.arg = "argument"
    return ast.dump(tree, include_attributes=False)


def instruction_variants(task, include_repairs):
    rows = [("implement", str(index), instruction + "\n\n" + task.spec)
            for index, instruction in enumerate(PROMPTS)]
    if include_repairs:
        buggy = task.source.replace(*task.mutation, 1)
        rows += [
            ("repair", "repair", "Repair the function below to meet the specification. Return only the complete corrected Python function, without explanation or Markdown.\n\n"
             + task.spec + "\n\nCurrent code:\n" + buggy),
            ("preserve-correct", "preserve", "Check the function against the specification. If it already meets it, return the function unchanged; otherwise repair it. Output only complete Python code.\n\n"
             + task.spec + "\n\nCurrent code:\n" + task.source),
        ]
    return rows


def build(out, refs, eval_dir, tasks=TASKS, include_repairs=False):
    out, refs, eval_dir = Path(out), Path(refs), Path(eval_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("output directory must be empty")
    if not refs.is_file() or not eval_dir.is_dir():
        raise ValueError("public benchmark references and repository evaluation directory are required")
    families, sources, seen_prompts, structures = set(), set(), set(), {}
    rows = {split: [] for split in FILES}
    verification = []
    decontaminate._init(refs, 13, 0.5)
    if not decontaminate._STATE["refs"]:
        raise ValueError("public references contain no eligible benchmark texts")
    evaluation_files = sorted(set(eval_dir.rglob("*.jsonl")) | set(eval_dir.rglob("heldout_security.txt")))
    evaluation_dirs = sorted({eval_dir, *(path.parent for path in evaluation_files)})
    source_root = Path(__file__).resolve().parents[1]
    authoring_files = sorted({task.source_file for task in tasks})
    if not set(authoring_files) <= {"data/verified_tasks.py", "data/curriculum_tasks.py"}:
        raise ValueError("only built-in project-authored task sources are supported")

    def check_overlap(text, family):
        if any(find_contamination(text, str(directory)) for directory in evaluation_dirs) or decontaminate.row_hits(text):
            raise ValueError(f"{family}: overlaps a public benchmark")

    for task in tasks:
        if task.split not in FILES or task.family in families or task.source in sources:
            raise ValueError("duplicate task family, duplicate solution or invalid split")
        families.add(task.family)
        sources.add(task.source)
        structure = structural_key(task.source)
        if structure in structures and structures[structure] != task.split:
            raise ValueError("renamed code structure overlaps splits")
        structures[structure] = task.split
        text = task.spec + "\n" + task.source
        check_overlap(text, task.family)
        verification.append(validate_task(task))
        for kind, variant, prompt in instruction_variants(task, include_repairs):
            check_overlap(prompt + "\n" + task.source, task.family)
            if prompt in seen_prompts:
                raise ValueError("duplicate instruction prompt")
            seen_prompts.add(prompt)
            rows[task.split].append({"id": f"verified-{task.family}-{variant}", "family": task.family,
                                     "kind": kind, "prompt": prompt, "response": task.source,
                                     "licences": ["MIT"], "source": task.source_file,
                                     "verification_cases": len(task.cases),
                                     **({"domain": task.domain} if include_repairs else {})})
    if not all(rows.values()):
        raise ValueError("all three problem-family splits must be populated")
    payloads = {split: "".join(json.dumps(row, sort_keys=True) + "\n" for row in values).encode()
                for split, values in rows.items()}
    split_key = {"train": "train", "validation": "val", "test": "test"}
    manifest = {"schema": 1, "name": "llmfs-verified-curriculum-v2" if include_repairs else "llmfs-verified-sft-seed-v1", "licence": "MIT",
                "provenance": "new project-authored tasks; no external model or downloaded code",
                "scope": "code-only authored pilot data; instruction/repair/preservation variants are not independent tasks",
                "split": "explicit disjoint problem families; all tasks are authored in this repository",
                "rows": {split_key[k]: len(v) for k, v in rows.items()},
                "families": dict(Counter(task.split for task in tasks)),
                "domains_by_family": dict(Counter(task.domain for task in tasks)),
                "kinds_by_split": {split_key[k]: dict(Counter(row["kind"] for row in values)) for k, values in rows.items()},
                "sha256": {split_key[k]: digest(v) for k, v in payloads.items()},
                "authoring_sha256": digest(Path(__file__).with_name("verified_tasks.py").read_bytes()),
                "authoring_files_sha256": {path: digest((source_root / path).read_bytes()) for path in authoring_files},
                "builder_sha256": digest(Path(__file__).read_bytes()),
                "contamination": {"reference_sha256": digest(refs.read_bytes()), "ngram_size": 13,
                                  "repository_evaluation_sha256": {str(path.relative_to(eval_dir)): digest(path.read_bytes())
                                                                   for path in evaluation_files},
                                  "containment_threshold": 0.5,
                                  "eligible_public_references": len(decontaminate._STATE["refs"]),
                                  "hits": 0, "private_holdout_read": False,
                                  "limitations": "exact snippets and 13-gram containment; not semantic independence"},
                "verification": {"reference_cases_passed": sum(item["cases"] for item in verification),
                                 "targeted_mutants_killed": len(verification), "tasks": verification}}
    out.mkdir(parents=True, exist_ok=True)
    for split, data in payloads.items():
        (out / FILES[split]).write_bytes(data)
    (out / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="Verify project-authored coding tasks and build code-only SFT splits.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--refs", required=True)
    parser.add_argument("--curriculum", choices=("seed", "expanded"), default="seed")
    parser.add_argument("--eval-dir", default=str(Path(__file__).resolve().parents[1] / "evals" / "data"))
    args = parser.parse_args(argv)
    tasks = TASKS
    if args.curriculum == "expanded":
        from data.curriculum_tasks import TASKS as extra_tasks
        tasks = TASKS + extra_tasks
    manifest = build(args.out, args.refs, args.eval_dir, tasks, include_repairs=args.curriculum == "expanded")
    print(json.dumps({key: manifest[key] for key in ("name", "rows", "families", "sha256")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
