"""
Load tasks from JSONL files and run one task against a model.

Each line of a .jsonl task file is one task object:

  multiple_choice:
    {"id","category","type":"multiple_choice","prompt","choices":[...],"answer":N}
  generation:
    {"id","category","type":"generation","prompt","scorer":"contains|exact|regex|
      contains_any","target": "..." | ["...",...], "max_new_tokens":N}
  code:
    {"id","category","type":"code","prompt","test":"<unittest source>","timeout":N}

A task result is a dict: {id, category, type, correct(bool), detail, n_choices?}.
"""

import json
import os
import re

import torch

from evals import scorers
from evals.scoring import choose_by_likelihood, generate_text


def load_tasks(paths):
    tasks = []
    seen = set()
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                task = json.loads(line)
                try:
                    validate_task(task)
                    if task["id"] in seen:
                        raise ValueError("duplicate task id")
                except (ValueError, TypeError, KeyError) as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
                seen.add(task["id"])
                tasks.append(task)
    return tasks


def validate_task(task):
    if not isinstance(task, dict):
        raise ValueError("task must be an object")
    for field in ("id", "category", "prompt"):
        if not isinstance(task.get(field), str) or not task[field].strip():
            raise ValueError(f"nonempty {field} is required")
    kind = task.get("type")
    if kind == "multiple_choice":
        choices = task.get("choices")
        if not isinstance(choices, list) or len(choices) < 2 or any(not isinstance(c, str) or not c.strip() for c in choices):
            raise ValueError("at least two nonempty choices are required")
        if len(set(choices)) != len(choices):
            raise ValueError("choices must be distinct")
        answer = task.get("answer")
        if type(answer) is not int or not 0 <= answer < len(choices):
            raise ValueError("answer must index a choice")
    elif kind == "generation":
        scorer, target = task.get("scorer", "contains"), task.get("target")
        if scorer not in {"contains", "contains_any", "exact", "regex"}:
            raise ValueError("unknown generation scorer")
        targets = target if scorer == "contains_any" else [target]
        if not isinstance(targets, list) or not targets or any(not isinstance(t, str) or not t.strip() for t in targets):
            raise ValueError("nonempty scorer targets are required")
        if scorer == "regex":
            try:
                re.compile(target)
            except re.error as exc:
                raise ValueError("invalid scoring regex") from exc
    elif kind == "code":
        if not isinstance(task.get("test"), str) or not task["test"].strip():
            raise ValueError("code task needs reviewed test metadata")
    else:
        raise ValueError("unknown task type")
    if "max_new_tokens" in task and (type(task["max_new_tokens"]) is not int or not 1 <= task["max_new_tokens"] <= 512):
        raise ValueError("max_new_tokens must be between 1 and 512")


def run_task(model, tokenizer, task, device="cpu", seed=1337):
    ttype = task["type"]
    base = {"id": task.get("id"), "category": task.get("category", "uncategorized"),
            "type": ttype}

    if ttype == "multiple_choice":
        best, means = choose_by_likelihood(
            model, tokenizer, task["prompt"], task["choices"], device)
        correct = best == task["answer"]
        return {**base, "correct": correct, "n_choices": len(task["choices"]),
                "chosen": best, "answer": task["answer"],
                "detail": f"chose {best}, answer {task['answer']}"}

    if ttype == "generation":
        torch.manual_seed(seed)
        out = generate_text(
            model, tokenizer, task["prompt"],
            max_new_tokens=task.get("max_new_tokens", 64),
            temperature=task.get("temperature", 0.6),
            top_k=task.get("top_k", 40), device=device)
        scorer = task.get("scorer", "contains")
        target = task["target"]
        if scorer == "exact":
            correct = scorers.exact_match(out, target)
        elif scorer == "regex":
            correct = scorers.regex_match(out, target)
        elif scorer == "contains_any":
            correct = scorers.contains_any(out, target)
        else:
            correct = scorers.contains(out, target)
        return {**base, "correct": correct, "detail": repr(out[:80])}

    if ttype == "code":
        from evals.code_executor import evaluate_function, task_spec
        spec = task_spec(task)
        if spec is None:
            return {**base, "correct": False, "execution_status": "unavailable",
                    "detail": "reviewed function cases missing or task content changed"}
        torch.manual_seed(seed)
        out = generate_text(
            model, tokenizer, task["prompt"],
            max_new_tokens=task.get("max_new_tokens", 256),
            temperature=task.get("temperature", 0.4),
            top_k=task.get("top_k", 40), device=device)
        verdict = evaluate_function(scorers.extract_code(out), spec["function"],
                                    spec["cases"], timeout=task.get("timeout", 10))
        return {**base, "correct": verdict.passed, "detail": verdict.detail,
                "execution_status": verdict.status,
                "cases_passed": verdict.cases_passed, "cases_total": verdict.cases_total}

    raise ValueError(f"unknown task type: {ttype!r}")


def default_task_files(data_dir=None):
    data_dir = data_dir or os.path.join(os.path.dirname(__file__), "data")
    names = ["security_mc.jsonl", "software_mc.jsonl", "cloud_mc.jsonl",
             "aisec_mc.jsonl", "reasoning_mc.jsonl", "generation.jsonl",
             "code_python.jsonl", "secure_development.jsonl"]
    return [os.path.join(data_dir, n) for n in names
            if os.path.exists(os.path.join(data_dir, n))]
