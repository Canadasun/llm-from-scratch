"""Developer workflow: inspect failures, compare runs, and record measured baselines.

    python -m evals.review --results run.json --baseline evals/baselines/m3.json
    python -m evals.review --results run.json --markdown review.md
    python -m evals.review --results run.json --write-baseline candidate.json

Recording a baseline is explicit; it never happens during evaluation or review.
"""

import argparse
import json
import re
import math
import copy
from pathlib import Path

from evals.evidence import CODE_EVALUATOR, SCORING_VERSION
from evals.regression import check
from evals.report import summarize


def validate_results(run):
    # Reject JSON's nonstandard NaN/Infinity extensions before deriving or
    # writing any evidence artifact, even in a non-gating metric or metadata.
    json.dumps(run, allow_nan=False)
    results = run.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError("task results are required")
    seen = set()
    for result in results:
        if not isinstance(result, dict) or not isinstance(result.get("id"), str) or result["id"] in seen:
            raise ValueError("unique task IDs are required")
        if type(result.get("correct")) is not bool or not isinstance(result.get("category"), str):
            raise ValueError("task outcomes must contain a boolean correct and a category")
        if result.get("type") not in {"code", "generation", "multiple_choice"}:
            raise ValueError("unknown result type")
        seen.add(result["id"])
    # Never record a headline that contradicts the underlying task outcomes.
    actual = summarize(results)
    for key in ("n", "correct", "acc"):
        if run.get("summary", {}).get("overall", {}).get(key) != actual["overall"][key]:
            raise ValueError("overall summary does not match task outcomes")
    return actual


def make_baseline(run, name="measured-baseline"):
    summary = validate_results(run)
    meta = run.get("meta", {})
    if meta.get("complete") is not True or meta.get("suite") != "all" or meta.get("limit") is not None:
        raise ValueError("baseline requires a complete, unlimited all-suite run")
    if meta.get("tasks_total") != len(run["results"]):
        raise ValueError("baseline task count is incomplete")
    for field in ("checkpoint_sha256", "suite_sha256", "heldout_sha256"):
        if not re.fullmatch(r"[a-f0-9]{64}", meta.get(field, "")):
            raise ValueError(f"baseline requires {field}")
    if meta.get("scoring_version") != SCORING_VERSION or meta.get("code_evaluator") != CODE_EVALUATOR:
        raise ValueError("baseline requires current scoring and evaluator versions")
    intrinsic = run.get("intrinsic", {}).get("trained", {})
    if type(intrinsic.get("bits_per_byte")) not in (int, float) or not math.isfinite(intrinsic["bits_per_byte"]) or intrinsic["bits_per_byte"] <= 0:
        raise ValueError("measured intrinsic metrics are required")
    for result in run["results"]:
        if result.get("detail", "").startswith("timed_out"):
            raise ValueError("baseline cannot include timed-out tasks")
        if result["type"] == "code" and result.get("execution_status") not in {"passed", "failed"}:
            raise ValueError("independent code evaluation must be available")
    recorded_meta = copy.deepcopy(meta)
    recorded_meta["task_ids"] = [r["id"] for r in run["results"]]
    if "ckpt" in recorded_meta:
        recorded_meta["ckpt"] = Path(recorded_meta["ckpt"]).name
    return {"name": name,
            "note": "Measured with the current suite and independent evaluator. Regression floor only; not a production-readiness claim.",
            "enforce": ["bits_per_byte", "overall_acc"],
            "enforce_categories": sorted(summary["by_category"]),
            "intrinsic": {"trained": copy.deepcopy(intrinsic)}, "summary": summary, "meta": recorded_meta,
            "tolerances": {"bits_per_byte_abs_max_increase": .15, "overall_acc_abs_max_drop": .05}}


def _cell(value):
    text = str(value).replace("\n", " ").replace("\r", " ")
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    for character in "\\`|[]!*_":
        text = text.replace(character, "\\" + character)
    return text


def build_review(run, baseline=None, category=None):
    summary = validate_results(run)
    meta = run.get("meta", {})
    overall = summary["overall"]
    lines = ["# Model evaluation review", "",
             f"Accuracy: **{overall['correct']}/{overall['n']} ({overall['acc']:.1%})**.",
             "This measures the recorded tasks; it does not establish production readiness.", "",
             f"Checkpoint SHA-256: `{_cell(meta.get('checkpoint_sha256', 'unrecorded'))}`",
             f"Suite SHA-256: `{_cell(meta.get('suite_sha256', 'unrecorded'))}`", "",
             "## Results by task type", "",
             "| Type | Correct / tasks | Accuracy | Wilson 95% interval |",
             "|---|---:|---:|---:|"]
    for kind, metrics in summary["by_type"].items():
        lo, hi = metrics["wilson_95"]
        lines.append(f"| {_cell(kind)} | {metrics['correct']}/{metrics['n']} | {metrics['acc']:.1%} | {lo:.1%}–{hi:.1%} |")
    if summary["mc_random_baseline"] is not None:
        lines += ["", f"MC random-guess reference: {summary['mc_random_baseline']:.1%}; compare with MC-only accuracy above."]
    lines += ["", "Intervals describe uncertainty on these small task samples, not performance on all real-world work.",
              "", "## Categories", "", "| Category | Correct / tasks | Accuracy |", "|---|---:|---:|"]
    for name, metrics in summary["by_category"].items():
        lines.append(f"| {_cell(name)} | {metrics['correct']}/{metrics['n']} | {metrics['acc']:.1%} |")
    if baseline is not None:
        passed, checks = check(run, baseline)
        lines += ["", "## Regression comparison", "",
                  "PASS — regression checks passed." if passed else "FAIL — regression or incompatible/incomplete evidence.",
                  "", "| Check | Outcome | Note |", "|---|---|---|"]
        for item in checks:
            state = "SKIP" if item["skipped"] else ("PASS" if item["ok"] else "FAIL")
            lines.append(f"| {_cell(item['metric'])} | {state} | {_cell(item.get('note', ''))} |")
    failures = [r for r in run["results"] if not r["correct"] and (category is None or r["category"] == category)]
    lines += ["", "## Failure triage", "", "| Task | Category | Type | Evidence |", "|---|---|---|---|"]
    for result in failures:
        lines.append("| " + " | ".join(_cell(result.get(k, "")) for k in ("id", "category", "type", "detail")) + " |")
    if not failures:
        lines.append("| — | — | — | No failures in the selected category |")
    lines += ["", "Model weights were not changed by this review.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Review measured model results without loading a model")
    parser.add_argument("--results", required=True)
    parser.add_argument("--baseline")
    parser.add_argument("--markdown", help="write a review artifact")
    parser.add_argument("--category", help="filter the failure-triage table")
    parser.add_argument("--write-baseline", help="explicitly record a new baseline to a NEW path")
    parser.add_argument("--name", default="measured-baseline")
    args = parser.parse_args(argv)
    try:
        run = json.loads(Path(args.results).read_text())
        baseline = json.loads(Path(args.baseline).read_text()) if args.baseline else None
        text = build_review(run, baseline, args.category)
        if args.write_baseline:
            recorded = make_baseline(run, args.name)
            payload = json.dumps(recorded, indent=2, allow_nan=False) + "\n"
            # Prevent accidental overwriting of an accepted regression floor.
            with open(args.write_baseline, "x", encoding="utf-8") as stream:
                stream.write(payload)
        if args.markdown:
            Path(args.markdown).write_text(text)
        print(text)
        return 1 if baseline is not None and not check(run, baseline)[0] else 0
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
