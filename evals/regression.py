"""
Regression gate: fail the build when a candidate checkpoint is meaningfully
WORSE than a recorded baseline.

This is the "evals in CI" enforcement point. It does NOT assert the model is
good (see docs/EVALUATION.md — release gating is a separate human decision). It
only catches *regressions*: a change that quietly tanks perplexity, next-token
accuracy, or task accuracy (e.g. a tokenizer bug, a broken checkpoint, a scoring
change) should turn the pipeline red.

A baseline file (evals/baselines/*.json) records reference metrics and the
tolerances. Metrics not present in the baseline are skipped, so a baseline can
gate on just the stable intrinsic numbers if desired.

    # compare an existing results file
    python -m evals.regression --results out.json --baseline evals/baselines/m2.json
    # run the suite on a checkpoint and compare in one step
    python -m evals.regression --ckpt checkpoints/ckpt.pt --baseline evals/baselines/m2.json

Exit code: 0 = no regression, 1 = regression, 2 = usage/IO error.
"""

import argparse
import json
import sys

# Default tolerances (overridable per-baseline via a "tolerances" block).
#   *_rel_max_increase : metric may rise by at most this FRACTION (lower is better)
#   *_abs_min          : metric may fall by at most this ABSOLUTE amount (higher is better)
DEFAULT_TOLERANCES = {
    "perplexity_rel_max_increase": 0.10,   # perplexity may worsen by <=10%
    "bits_per_byte_abs_max_increase": 0.15,
    "top1_acc_abs_max_drop": 0.03,
    "top5_acc_abs_max_drop": 0.03,
    "overall_acc_abs_max_drop": 0.05,
}


def _intrinsic(d):
    return ((d or {}).get("intrinsic") or {}).get("trained") or {}


def _overall_acc(d):
    return ((d or {}).get("summary") or {}).get("overall", {}).get("acc")


def check(candidate, baseline, tolerances=None):
    """Compare candidate results to a baseline. Returns (ok, checks) where checks
    is a list of {metric, baseline, candidate, limit, direction, ok, skipped}."""
    tol = dict(DEFAULT_TOLERANCES)
    tol.update(baseline.get("tolerances") or {})
    if tolerances:
        tol.update(tolerances)

    # Optional allow-list of metrics to actually gate on. If a baseline sets
    # "enforce", any metric outside it is recorded for reference but not enforced.
    # This is how a cross-tokenizer baseline (e.g. M2 vocab 1024 vs M3 vocab 2048)
    # gates only on tokenizer-independent metrics (bits_per_byte, overall_acc) and
    # leaves perplexity / top-k as non-gating reference values.
    enforce = baseline.get("enforce")

    cand_i, base_i = _intrinsic(candidate), _intrinsic(baseline)
    checks = []

    def record(metric, base_v, cand_v, limit, direction, ok, note=""):
        enforced = enforce is None or metric in enforce
        skipped = base_v is None or cand_v is None or not enforced
        if not enforced and note == "":
            note = "not enforced"
        checks.append({"metric": metric, "baseline": base_v, "candidate": cand_v,
                       "limit": limit, "direction": direction,
                       "ok": True if skipped else ok,
                       "skipped": skipped, "note": note})

    # Lower-is-better, relative ceiling on increase.
    for metric, tkey in [("perplexity", "perplexity_rel_max_increase"),
                         ("bits_per_byte", "bits_per_byte_abs_max_increase")]:
        bv, cv = base_i.get(metric), cand_i.get(metric)
        if bv is None or cv is None:
            record(metric, bv, cv, None, "lower_better", True, "skipped")
            continue
        if tkey.endswith("rel_max_increase"):
            limit = bv * (1 + tol[tkey])
        else:
            limit = bv + tol[tkey]
        record(metric, bv, cv, limit, "lower_better", cv <= limit)

    # Higher-is-better, absolute floor on drop.
    higher = [
        ("top1_acc", "top1_acc_abs_max_drop", base_i.get("top1_acc"), cand_i.get("top1_acc")),
        ("top5_acc", "top5_acc_abs_max_drop", base_i.get("top5_acc"), cand_i.get("top5_acc")),
        ("overall_acc", "overall_acc_abs_max_drop", _overall_acc(baseline), _overall_acc(candidate)),
    ]
    for metric, tkey, bv, cv in higher:
        if bv is None or cv is None:
            record(metric, bv, cv, None, "higher_better", True, "skipped")
            continue
        limit = bv - tol[tkey]
        record(metric, bv, cv, limit, "higher_better", cv >= limit)

    ok = all(c["ok"] for c in checks)
    return ok, checks


def format_report(ok, checks):
    lines = ["Regression gate (candidate vs baseline):"]
    for c in checks:
        if c["skipped"]:
            reason = c["note"] or "missing in baseline or candidate"
            cand = f"{c['candidate']:.4g}" if c["candidate"] is not None else "-"
            lines.append(f"  - {c['metric']:<14} SKIP ({reason}); candidate={cand}")
            continue
        arrow = "≤" if c["direction"] == "lower_better" else "≥"
        status = "OK  " if c["ok"] else "FAIL"
        lines.append(
            f"  [{status}] {c['metric']:<14} candidate={c['candidate']:.4g} "
            f"{arrow} limit={c['limit']:.4g}  (baseline={c['baseline']:.4g})")
    lines.append(f"\nRESULT: {'PASS — no regression' if ok else 'FAIL — regression detected'}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Eval regression gate")
    ap.add_argument("--baseline", required=True, help="baseline JSON to compare against")
    ap.add_argument("--results", help="an existing results JSON (from evals.run --json)")
    ap.add_argument("--ckpt", help="run the suite on this checkpoint instead of --results")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--suite", choices=["all", "tasks", "intrinsic"], default="all")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--task-timeout", type=float, default=120,
                    help="per-task seconds cap so a slow task can't stall the "
                         "gate (default 120; 0 disables)")
    args = ap.parse_args(argv)

    try:
        with open(args.baseline) as f:
            baseline = json.load(f)
    except OSError as e:
        print(f"error: cannot read baseline: {e}", file=sys.stderr)
        return 2

    if args.results:
        try:
            with open(args.results) as f:
                candidate = json.load(f)
        except OSError as e:
            print(f"error: cannot read results: {e}", file=sys.stderr)
            return 2
    elif args.ckpt:
        from evals.run import evaluate
        candidate = evaluate(args.ckpt, args.device, args.suite, args.limit,
                             progress=lambda s: print(s, flush=True),
                             task_timeout=(args.task_timeout or None))
    else:
        print("error: pass --results or --ckpt", file=sys.stderr)
        return 2

    ok, checks = check(candidate, baseline)
    print("\n" + format_report(ok, checks))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
