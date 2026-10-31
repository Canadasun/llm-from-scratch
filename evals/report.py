"""
Aggregate task results into an honest report.

Honest means: per-category accuracy *with sample sizes*, the random-chance
baseline for multiple-choice (so near-chance results read as near-chance, not as
success), a few concrete failure examples, and an explicit limitations note.
"""

from collections import defaultdict
import math


def _metrics(correct, n):
    if not n:
        return {"n": 0, "correct": 0, "acc": 0.0, "wilson_95": None}
    p, z = correct / n, 1.959963984540054
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return {"n": n, "correct": correct, "acc": p,
            "wilson_95": [max(0.0, center - margin), min(1.0, center + margin)]}


def _mc_chance(results):
    """Expected accuracy of random guessing over the MC tasks present."""
    chances = [1.0 / r["n_choices"] for r in results
               if r["type"] == "multiple_choice" and r.get("n_choices")]
    return sum(chances) / len(chances) if chances else None


def summarize(results):
    by_cat = defaultdict(lambda: {"n": 0, "correct": 0})
    by_type = defaultdict(lambda: {"n": 0, "correct": 0})
    for r in results:
        c = by_cat[r["category"]]
        c["n"] += 1
        c["correct"] += int(r["correct"])
        t = by_type[r["type"]]
        t["n"] += 1
        t["correct"] += int(r["correct"])
    total_n = sum(c["n"] for c in by_cat.values())
    total_correct = sum(c["correct"] for c in by_cat.values())
    return {
        "by_category": {k: _metrics(v["correct"], v["n"])
                        for k, v in sorted(by_cat.items())},
        "by_type": {k: _metrics(v["correct"], v["n"]) for k, v in sorted(by_type.items())},
        "overall": _metrics(total_correct, total_n),
        "mc_random_baseline": _mc_chance(results),
    }


def format_report(summary, results, intrinsic=None, max_failures=5):
    lines = []
    lines.append("=" * 62)
    lines.append("  EVALUATION REPORT — from-scratch LLM")
    lines.append("=" * 62)

    if intrinsic:
        lines.append("\nIntrinsic quality on held-out text (lower perplexity /")
        lines.append("bits-per-byte = better; vs an untrained same-shape model):")
        t = intrinsic.get("trained", {})
        b = intrinsic.get("baseline", {})
        lines.append(f"  {'metric':<16}{'trained':>14}{'random-init':>16}")
        for key, label in [("perplexity", "perplexity"),
                           ("bits_per_byte", "bits/byte"),
                           ("top1_acc", "next-tok top1"),
                           ("top5_acc", "next-tok top5")]:
            tv, bv = t.get(key), b.get(key)
            if tv is None:
                continue
            fmt = (lambda v: f"{v:.3f}") if key.endswith("acc") else (lambda v: f"{v:,.2f}")
            bstr = fmt(bv) if bv is not None else "-"
            lines.append(f"  {label:<16}{fmt(tv):>14}{bstr:>16}")
        if t.get("tokens"):
            lines.append(f"  (scored on {t['tokens']:,} tokens / {t.get('bytes', 0):,} bytes)")

    lines.append("\nCapability tasks — accuracy by category (n = sample size):")
    for cat, v in summary["by_category"].items():
        lines.append(f"  {cat:<28} {v['acc']*100:5.1f}%   ({v['correct']}/{v['n']})")
    o = summary["overall"]
    lines.append(f"  {'-'*28} {'-'*5}")
    lines.append(f"  {'OVERALL':<28} {o['acc']*100:5.1f}%   ({o['correct']}/{o['n']})")
    if summary["mc_random_baseline"] is not None:
        mc = summary.get("by_type", {}).get("multiple_choice")
        if mc:
            lines.append(f"  MC-only accuracy: {mc['acc']*100:.1f}% ({mc['correct']}/{mc['n']})")
        lines.append(f"  (random-guess baseline on the MC tasks ≈ "
                     f"{summary['mc_random_baseline']*100:.1f}%)")

    failures = [r for r in results if not r["correct"]][:max_failures]
    if failures:
        lines.append("\nExample failures (first few):")
        for r in failures:
            lines.append(f"  [{r['category']}/{r['id']}] {r['type']}: {r.get('detail','')}")

    lines.append("\nLimitations:")
    lines.append("  - This is a base model, not instruction-tuned; it is")
    lines.append("    scored on likelihood (MC) and raw generation, not chat ability.")
    lines.append("  - A small, hand-authored starter suite: treat numbers as a")
    lines.append("    baseline to improve against, not a readiness claim.")
    lines.append("  - Code tasks compare sandboxed function outputs in an independent")
    lines.append("    controller. Missing isolation/cases fails closed; tests are finite.")
    lines.append("=" * 62)
    return "\n".join(lines)
