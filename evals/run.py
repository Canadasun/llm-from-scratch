"""
Run the evaluation suite against a checkpoint and print an honest report.

    python -m evals.run                         # all evals on checkpoints/ckpt.pt
    python -m evals.run --suite tasks           # skip intrinsic (faster)
    python -m evals.run --suite intrinsic
    python -m evals.run --limit 20 --json out.json
    python -m evals.run --ckpt path/to/ckpt.pt --device cpu

Exit code is 0 on success; the report never "passes/fails" the model — it
reports. Release gating is a separate, explicit decision (see docs/EVALUATION.md).
"""

import argparse
import json
import os
import signal
from datetime import datetime, timezone

import torch

from evals.evidence import CODE_EVALUATOR, SCORING_VERSION, file_sha256, suite_sha256

from evals.loader import DEFAULT_CKPT, load_checkpoint, random_baseline, pick_device
from evals import report as report_mod
from evals.scoring import corpus_metrics
from evals.tasks import default_task_files, load_tasks, run_task


class _TaskTimeout(Exception):
    pass


def _run_task_guarded(model, tokenizer, task, device, timeout):
    """Run one task, but never let a single slow task (e.g. a long CPU
    generation) stall the whole suite. On timeout the task is recorded as an
    incorrect result marked 'timed_out' and evaluation continues.

    Uses SIGALRM, so it is effective only on the main thread on Unix (the CLI and
    CI case); with timeout falsy it is a plain call, unchanged behaviour.
    """
    if not timeout:
        return run_task(model, tokenizer, task, device)

    def _handler(signum, frame):
        raise _TaskTimeout()

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, float(timeout))
    try:
        return run_task(model, tokenizer, task, device)
    except _TaskTimeout:
        return {"id": task.get("id"),
                "category": task.get("category", "uncategorized"),
                "type": task.get("type"), "correct": False,
                "detail": f"timed_out (>{timeout}s)"}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)

HELDOUT = os.path.join(os.path.dirname(__file__), "data", "heldout_security.txt")


def run_intrinsic(model, tokenizer, cfg, device):
    if not os.path.exists(HELDOUT):
        return None
    with open(HELDOUT, encoding="utf-8") as f:
        text = f.read()
    trained = corpus_metrics(model, tokenizer, text, device)
    base_model = random_baseline(cfg, device)
    baseline = corpus_metrics(base_model, tokenizer, text, device)
    return {"trained": trained, "baseline": baseline}


def evaluate(ckpt=DEFAULT_CKPT, device="auto", suite="all", limit=None,
             progress=None, task_timeout=None):
    """Run the eval suite and return {"summary","intrinsic","results","meta"}.

    This is the reusable core shared by the CLI (main) and the regression gate
    (evals.regression). `progress` is an optional callable(str) for status lines.
    `task_timeout` (seconds) caps any single capability task so one pathologically
    slow task cannot stall the whole run; a timed-out task counts as incorrect.
    """
    say = progress or (lambda *_: None)
    if suite not in {"all", "tasks", "intrinsic"}:
        raise ValueError("unknown evaluation suite")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be a positive integer")
    device = pick_device(device)
    checkpoint_hash = file_sha256(ckpt)
    model, tokenizer, cfg = load_checkpoint(ckpt, device)
    say(f"Loaded {ckpt} | device {device} | tokenizer {tokenizer.kind} "
        f"vocab {tokenizer.vocab_size} | block {cfg['block_size']} "
        f"layers {cfg['n_layer']}")

    all_tasks = load_tasks(default_task_files())
    suite_hash = suite_sha256(all_tasks)
    selected_tasks = all_tasks[:limit] if limit is not None else all_tasks
    intrinsic = None
    if suite in ("all", "intrinsic"):
        say("Running intrinsic metrics (held-out perplexity vs random init)...")
        intrinsic = run_intrinsic(model, tokenizer, cfg, device)

    results = []
    summary = {"by_category": {}, "overall": {"n": 0, "correct": 0, "acc": 0.0},
               "mc_random_baseline": None}
    sandbox = None
    if suite in ("all", "tasks"):
        # If the suite contains code tasks, actively verify the isolated evaluator
        # works on this host BEFORE scoring, so an unavailable sandbox is surfaced
        # as "not measured" (and blocks the gate) rather than silently scoring
        # every code task 0 and looking like a model failure.
        if any(t.get("type") == "code" for t in selected_tasks):
            from evals.code_executor import sandbox_preflight
            sandbox = sandbox_preflight()
            if not sandbox["available"]:
                say(f"  SANDBOX UNAVAILABLE — code tasks will NOT be measured "
                    f"({sandbox['detail']}); install bubblewrap+libseccomp or use "
                    f"an approved host.")
        tasks = selected_tasks
        say(f"Running {len(tasks)} capability tasks...")
        for i, task in enumerate(tasks, 1):
            r = _run_task_guarded(model, tokenizer, task, device, task_timeout)
            results.append(r)
            if r.get("detail", "").startswith("timed_out"):
                say(f"  {i}/{len(tasks)}  TIMED OUT: {task.get('id')}")
            elif i % 10 == 0:
                say(f"  {i}/{len(tasks)}")
        summary = report_mod.summarize(results)

    if file_sha256(ckpt) != checkpoint_hash or suite_sha256(all_tasks) != suite_hash:
        raise ValueError("checkpoint or reviewed code cases changed during evaluation")
    return {"summary": summary, "intrinsic": intrinsic, "results": results,
            "meta": {"ckpt": ckpt, "device": device,
                     "checkpoint_sha256": checkpoint_hash,
                     "suite_sha256": suite_hash,
                     "heldout_sha256": file_sha256(HELDOUT) if os.path.exists(HELDOUT) else None,
                     "scoring_version": SCORING_VERSION, "code_evaluator": CODE_EVALUATOR,
                     "sandbox": sandbox,
                     "evaluated_at": datetime.now(timezone.utc).isoformat(),
                     "torch_version": str(torch.__version__), "seed": 1337,
                     "suite": suite, "limit": limit, "tasks_total": len(all_tasks),
                     "complete": suite == "all" and limit is None and intrinsic is not None,
                     "num_params": model.num_params(),
                     "tokenizer": tokenizer.kind, "vocab_size": tokenizer.vocab_size,
                     "block_size": cfg["block_size"], "n_layer": cfg["n_layer"]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=2, help="CPU threads (default 2)")
    ap.add_argument("--suite", choices=["all", "tasks", "intrinsic"], default="all")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of capability tasks (quick smoke run)")
    ap.add_argument("--json", default=None, help="also write results as JSON here")
    ap.add_argument("--task-timeout", type=float, default=None,
                    help="seconds cap per capability task (prevents a slow task "
                         "from stalling the run; timed-out tasks count incorrect)")
    args = ap.parse_args()
    if args.threads < 1:
        ap.error("threads must be positive")
    torch.set_num_threads(args.threads)

    out = evaluate(args.ckpt, args.device, args.suite, args.limit,
                   progress=lambda s: print(s, flush=True),
                   task_timeout=args.task_timeout)
    summary, intrinsic, results = out["summary"], out["intrinsic"], out["results"]

    print("\n" + report_mod.format_report(summary, results, intrinsic))

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"summary": summary, "intrinsic": intrinsic,
                       "results": results, "meta": out["meta"]}, f, indent=2)
        print(f"\nWrote JSON results to {args.json}")


if __name__ == "__main__":
    main()
