"""Score a checkpoint on the frozen coding benchmarks (first greedy sample).

    python -m evals.bench_run --ckpt checkpoints/ckpt.pt --bench-dir ~/llmfs-data/bench --json out.json
    python -m evals.bench_run --ckpt sft.pt --bench-dir DIR --chat --suites humaneval mbpp

HumanEval prompts are completed as code and cut at the usual stop sequences.
MBPP prompts ask for a function; the answer is read up to a closing code fence.
Every program is checked with ast.parse (the "parses as Python" metric) and then
run only inside the bubblewrap sandbox against controller-held cases.
"""

import argparse
import ast
import hashlib
import json
import re
import time
from datetime import datetime, timezone

from evals.bench import FROZEN, SUITES, load_suite, verify

STOPS = ("\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n#", "\nassert ")
MBPP_STOPS = ("```", "\nassert ", "\nprint(", "\nif __name__")
SANDBOX_TIMEOUT = 120


def _cut(text, stops):
    cut = len(text)
    for stop in stops:
        i = text.find(stop)
        if i != -1:
            cut = min(cut, i)
    return text[:cut]


def humaneval_program(prompt, completion):
    return prompt + _cut(completion, STOPS)


def mbpp_program(completion):
    head = completion.split("```", 1)[0]
    if head.strip() and parses(head) or "```" not in completion:
        return _cut(head, MBPP_STOPS[1:]).strip() + "\n"
    return chat_program(completion)


def chat_program(completion):
    match = re.search(r"```(?:python)?\s*\n(.*?)(?:```|$)", completion, re.DOTALL)
    body = match.group(1) if match else completion
    return _cut(body, MBPP_STOPS[1:]).strip() + "\n"


def parses(program):
    try:
        ast.parse(program)
        return True
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return False


def model_input(task, suite, chat):
    if chat:
        from special_tokens import ASSISTANT, USER
        text = task["prompt"]
        if suite.startswith("humaneval"):
            text = f"Complete this Python function.\n\n```python\n{task['prompt']}```"
        return f"{USER}{text}{ASSISTANT}", True
    if suite.startswith("humaneval"):
        return task["prompt"], False
    return task["prompt"] + "\n```python\n", False


def program_for(task, suite, completion, chat):
    if chat:
        return chat_program(completion)
    if suite.startswith("humaneval"):
        return humaneval_program(task["prompt"], completion)
    return mbpp_program(completion)


def generate(model, tokenizer, text, special, max_new_tokens, device):
    import torch
    from special_tokens import ENDOFTEXT, USER
    ids = tokenizer.encode(text, allowed_special="all") if special else tokenizer.encode(text)
    block = model.cfg.block_size
    max_new_tokens = min(max_new_tokens, block // 2)
    ids = ids[-(block - max_new_tokens):]
    stop = [i for i in (tokenizer.special_id(ENDOFTEXT), tokenizer.special_id(USER)) if i is not None]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens, temperature=1.0, top_k=1, stop_ids=stop or None)
    return tokenizer.decode(out[0, len(ids):].tolist(), skip_special=True)


def score(programs, suites, workers=4):
    from concurrent.futures import ThreadPoolExecutor
    from evals.code_executor import evaluate_cases
    jobs = [(suite, task) for suite, tasks in suites.items() for task in tasks]

    def run(job):
        suite, task = job
        program = programs[(suite.replace("_plus", ""), task["id"])]
        ok = parses(program)
        verdict = evaluate_cases(program, task["entry_point"], task["cases"], atol=task["atol"],
                                 timeout=SANDBOX_TIMEOUT) if ok else None
        return suite, task["id"], ok, verdict

    results = []
    with ThreadPoolExecutor(workers) as pool:
        for suite, tid, ok, verdict in pool.map(run, jobs):
            results.append({"suite": suite, "id": tid, "parses": ok,
                            "passed": bool(verdict and verdict.passed),
                            "status": verdict.status if verdict else "syntax-error",
                            "detail": verdict.detail if verdict else ""})
    return results


def summarize(results):
    out = {}
    for suite in sorted({r["suite"] for r in results}):
        rows = [r for r in results if r["suite"] == suite]
        out[suite] = {"tasks": len(rows), "passed": sum(r["passed"] for r in rows),
                      "pass_at_1": round(sum(r["passed"] for r in rows) / len(rows), 4),
                      "parses": round(sum(r["parses"] for r in rows) / len(rows), 4),
                      "unavailable": sum(r["status"] == "unavailable" for r in rows)}
    return out


def run(ckpt, bench_dir, suites=SUITES, limit=None, chat=False, max_new_tokens=384, device="auto",
        progress=print):
    from evals.code_executor import sandbox_preflight
    from evals.evidence import file_sha256
    from evals.loader import load_checkpoint, pick_device
    frozen = verify(bench_dir, FROZEN)
    preflight = sandbox_preflight()
    if not preflight["available"]:
        raise SystemExit(f"sandbox unavailable, benchmarks not measured: {preflight['detail']}")
    device = pick_device(device)
    model, tokenizer, cfg = load_checkpoint(ckpt, device)
    loaded = {s: load_suite(bench_dir, s)[:limit] if limit else load_suite(bench_dir, s) for s in suites}
    programs, completions = {}, {}
    started = time.time()
    for suite, tasks in loaded.items():
        base = suite.replace("_plus", "")
        for task in tasks:
            key = (base, task["id"])
            if key in programs:
                continue
            text, special = model_input(task, base, chat)
            completion = generate(model, tokenizer, text, special, max_new_tokens, device)
            completions[key] = completion
            programs[key] = program_for(task, base, completion, chat)
            if len(programs) % 25 == 0:
                progress(f"  generated {len(programs)} programs ({time.time() - started:.0f}s)")
    results = score(programs, loaded)
    summary = summarize(results)
    parse_rate = round(sum(parses(p) for p in programs.values()) / max(1, len(programs)), 4)
    return {"summary": summary, "parses_as_python": parse_rate, "results": results,
            "programs": {f"{s}:{t}": p for (s, t), p in programs.items()},
            "meta": {"ckpt": str(ckpt), "checkpoint_sha256": file_sha256(ckpt),
                     "bench_manifest_sha256": hashlib.sha256(FROZEN.read_bytes()).hexdigest(),
                     "suite_sha256": {s: frozen["suites"][s]["sha256"] for s in suites},
                     "decoding": {"greedy": True, "max_new_tokens": max_new_tokens, "chat": chat,
                                  "stops": list(STOPS)},
                     "limit": limit, "device": str(device), "sandbox": preflight,
                     "evaluated_at": datetime.now(timezone.utc).isoformat(),
                     "n_layer": cfg["n_layer"], "block_size": cfg["block_size"],
                     "tokenizer": tokenizer.kind, "vocab_size": tokenizer.vocab_size}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bench-dir", required=True)
    ap.add_argument("--suites", nargs="+", choices=SUITES, default=list(SUITES))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    out = run(args.ckpt, args.bench_dir, args.suites, args.limit, args.chat, args.max_new_tokens, args.device)
    print(json.dumps({"summary": out["summary"], "parses_as_python": out["parses_as_python"]}, indent=2))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
