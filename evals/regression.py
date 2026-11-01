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

from evals.report import _metrics, summarize

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


def _category_floor(base_category, tol):
    """Predeclared practical regression tolerance, not a significance test.

    Confidence intervals remain descriptive in reports. The baseline interval
    alone cannot establish a candidate's equivalence or non-inferiority.
    """
    bv = base_category.get("acc")
    if bv is None:
        return None
    return max(0.0, bv - tol.get("category_acc_abs_max_drop", 0.05))


def _trusted_summary(run):
    """The summary to GATE ON. Never trust a run's self-reported summary for a
    gated metric: when the run carries its own task outcomes, recompute the
    summary from them so edited `summary`/`by_category` metadata cannot make a
    failing run pass (audit: category-accuracy metadata forgery). Falls back to
    the recorded summary only when no raw results are present (e.g. a baseline
    artifact, or a crafted comparison with no results)."""
    results = run.get("results")
    if isinstance(results, list) and results:
        return summarize(results)
    return run.get("summary", {})


def check(candidate, baseline, tolerances=None):
    """Compare candidate results to a baseline. Returns (ok, checks) where checks
    is a list of {metric, baseline, candidate, limit, direction, ok, skipped}."""
    tol = dict(DEFAULT_TOLERANCES)
    tol.update(baseline.get("tolerances") or {})
    if tolerances:
        tol.update(tolerances)

    # Gated accuracy metrics are recomputed from task outcomes, not trusted from
    # the (attacker-editable) candidate summary. See _trusted_summary.
    cand_summary = _trusted_summary(candidate)
    base_summary = _trusted_summary(baseline)

    # Optional allow-list of metrics to actually gate on. If a baseline sets
    # "enforce", any metric outside it is recorded for reference but not enforced.
    # This is how a cross-tokenizer baseline (e.g. M2 vocab 1024 vs M3 vocab 2048)
    # gates only on tokenizer-independent metrics (bits_per_byte, overall_acc) and
    # leaves perplexity / top-k as non-gating reference values.
    enforce = baseline.get("enforce")

    cand_i, base_i = _intrinsic(candidate), _intrinsic(baseline)
    checks = []

    base_meta, cand_meta = baseline.get("meta", {}), candidate.get("meta", {})
    for field in ("suite_sha256", "heldout_sha256", "scoring_version", "code_evaluator"):
        if base_meta.get(field) is not None:
            same = cand_meta.get(field) == base_meta[field]
            checks.append({"metric": field, "baseline": 1, "candidate": int(same),
                           "limit": 1, "direction": "higher_better", "ok": same,
                           "skipped": False, "note": "evaluation provenance must match"})
    if base_meta.get("complete") is True:
        complete = cand_meta.get("complete") is True
        base_n = baseline.get("summary", {}).get("overall", {}).get("n")
        cand_n = candidate.get("summary", {}).get("overall", {}).get("n")
        complete = complete and base_n == cand_n and len(candidate.get("results", [])) == base_n
        if "task_ids" in base_meta:
            complete = complete and [r.get("id") for r in candidate.get("results", [])] == base_meta["task_ids"]
        checks.append({"metric": "complete_suite", "baseline": 1, "candidate": int(complete),
                       "limit": 1, "direction": "higher_better", "ok": complete,
                       "skipped": False, "note": "full suite required; smoke runs cannot pass"})

    def record(metric, base_v, cand_v, limit, direction, ok, note=""):
        enforced = enforce is None or metric in enforce
        # A metric the baseline GATES ON that is present in the baseline but
        # MISSING from the candidate is a FAIL, not a skip: absent evidence must
        # never read as "no regression" (otherwise an empty candidate {} would
        # pass against a populated baseline).
        if enforced and base_v is not None and cand_v is None:
            checks.append({"metric": metric, "baseline": base_v, "candidate": None,
                           "limit": limit, "direction": direction, "ok": False,
                           "skipped": False,
                           "note": note or "candidate missing required metric"})
            return
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
        ("overall_acc", "overall_acc_abs_max_drop",
         base_summary.get("overall", {}).get("acc"), cand_summary.get("overall", {}).get("acc")),
    ]
    for metric, tkey, bv, cv in higher:
        if bv is None or cv is None:
            record(metric, bv, cv, None, "higher_better", True, "skipped")
            continue
        limit = bv - tol[tkey]
        record(metric, bv, cv, limit, "higher_better", cv >= limit)

    # Overall accuracy must not hide a security/coding decline behind gains in
    # another category. Category floors are opt-in for historical baselines.
    #
    # Use an explicit practical loss budget. This is a policy threshold, not
    # evidence of statistical equivalence; qualify releases independently.
    for category in baseline.get("enforce_categories", []):
        base_category = base_summary.get("by_category", {}).get(category, {})
        candidate_category = cand_summary.get("by_category", {}).get(category, {})
        bv, cv = base_category.get("acc"), candidate_category.get("acc")
        limit = _category_floor(base_category, tol)
        passed = bv is not None and cv is not None and limit is not None and cv >= limit
        checks.append({"metric": f"category:{category}", "baseline": bv, "candidate": cv,
                       "limit": limit, "direction": "higher_better", "ok": passed,
                       "skipped": False,
                       "note": "regression floor = baseline accuracy minus predeclared drop tolerance"})

    # Absolute minimum-capability floors, kept SEPARATE from the regression
    # comparison above (audit: do not conflate "did not regress" with "meets a
    # minimum quality bar"). Optional: enforced only when a baseline declares
    # them, so the CI regression gate stays regression-only while a RELEASE gate
    # can require real minimums. Each is an absolute accuracy the candidate's
    # recomputed outcomes must meet, independent of the baseline's own score.
    floors = baseline.get("absolute_floors") or {}
    for key, minimum in sorted(floors.items()):
        if key == "overall":
            cv = cand_summary.get("overall", {}).get("acc")
        else:
            cv = cand_summary.get("by_category", {}).get(key, {}).get("acc")
        ok_floor = cv is not None and cv >= minimum
        checks.append({"metric": f"min:{key}", "baseline": minimum, "candidate": cv,
                       "limit": minimum, "direction": "higher_better", "ok": ok_floor,
                       "skipped": False,
                       "note": "absolute minimum-capability floor (release gate)"})

    code_results = [r for r in candidate.get("results", []) if r.get("type") == "code"]
    if code_results:
        available = all(r.get("execution_status") in ("passed", "failed") for r in code_results)
        checks.append({"metric": "code_execution", "baseline": 1, "candidate": int(available),
                       "limit": 1, "direction": "higher_better", "ok": available,
                       "skipped": False, "note": "all code tasks require an available independent evaluator"})
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
        cand = f"{c['candidate']:.4g}" if c["candidate"] is not None else "MISSING"
        lim = f"{c['limit']:.4g}" if c["limit"] is not None else "-"
        base = f"{c['baseline']:.4g}" if c["baseline"] is not None else "-"
        lines.append(
            f"  [{status}] {c['metric']:<14} candidate={cand} "
            f"{arrow} limit={lim}  (baseline={base})")
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
