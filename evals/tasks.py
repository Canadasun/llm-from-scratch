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

import torch

from evals import scorers
from evals.scoring import choose_by_likelihood, generate_text


def load_tasks(paths):
    tasks = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                tasks.append(json.loads(line))
    return tasks


def run_task(model, tokenizer, task, device="cpu", seed=1337):
    ttype = task["type"]
    base = {"id": task.get("id"), "category": task.get("category", "uncategorized"),
            "type": ttype}

    if ttype == "multiple_choice":
        best, means = choose_by_likelihood(
            model, tokenizer, task["prompt"], task["choices"], device)
        correct = best == task["answer"]
        return {**base, "correct": correct, "n_choices": len(task["choices"]),
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
        torch.manual_seed(seed)
        out = generate_text(
            model, tokenizer, task["prompt"],
            max_new_tokens=task.get("max_new_tokens", 256),
            temperature=task.get("temperature", 0.4),
            top_k=task.get("top_k", 40), device=device)
        passed, detail = scorers.run_python_unittest(
            out, task["test"], timeout=task.get("timeout", 10))
        return {**base, "correct": passed, "detail": detail}

    raise ValueError(f"unknown task type: {ttype!r}")


def default_task_files(data_dir=None):
    data_dir = data_dir or os.path.join(os.path.dirname(__file__), "data")
    names = ["security_mc.jsonl", "software_mc.jsonl", "cloud_mc.jsonl",
             "aisec_mc.jsonl", "reasoning_mc.jsonl", "generation.jsonl",
             "code_python.jsonl"]
    return [os.path.join(data_dir, n) for n in names
            if os.path.exists(os.path.join(data_dir, n))]
