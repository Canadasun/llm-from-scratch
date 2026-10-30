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

from evals.loader import DEFAULT_CKPT, load_checkpoint, random_baseline, pick_device
from evals import report as report_mod
from evals.scoring import corpus_metrics
from evals.tasks import default_task_files, load_tasks, run_task

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--suite", choices=["all", "tasks", "intrinsic"], default="all")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of capability tasks (quick smoke run)")
    ap.add_argument("--json", default=None, help="also write results as JSON here")
    args = ap.parse_args()

    device = pick_device(args.device)
    model, tokenizer, cfg = load_checkpoint(args.ckpt, device)
    print(f"Loaded {args.ckpt} | device {device} | "
          f"tokenizer {tokenizer.kind} vocab {tokenizer.vocab_size} | "
          f"block {cfg['block_size']} layers {cfg['n_layer']}", flush=True)

    intrinsic = None
    if args.suite in ("all", "intrinsic"):
        print("Running intrinsic metrics (held-out perplexity vs random init)...",
              flush=True)
        intrinsic = run_intrinsic(model, tokenizer, cfg, device)

    results = []
    summary = {"by_category": {}, "overall": {"n": 0, "correct": 0, "acc": 0.0},
               "mc_random_baseline": None}
    if args.suite in ("all", "tasks"):
        tasks = load_tasks(default_task_files())
        if args.limit:
            tasks = tasks[:args.limit]
        print(f"Running {len(tasks)} capability tasks...", flush=True)
        for i, task in enumerate(tasks, 1):
            results.append(run_task(model, tokenizer, task, device))
            if i % 10 == 0:
                print(f"  {i}/{len(tasks)}", flush=True)
        summary = report_mod.summarize(results)

    print("\n" + report_mod.format_report(summary, results, intrinsic))

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"summary": summary, "intrinsic": intrinsic,
                       "results": results}, f, indent=2)
        print(f"\nWrote JSON results to {args.json}")


if __name__ == "__main__":
    main()
