"""
Stage-4 pilot readiness scorecard (directive P8).

Combines the four measured axes — reliability, latency, cost and security — into
an explicit, threshold-based readiness verdict, so "is this ready for an
authorized pilot?" is answered by numbers against stated bars, not by vibes.

Readiness levels (honest, and deliberately conservative):
- **blocked**   : a security gate failed (a cross-tenant leak, an auth bypass, or
                  a dual-use FP/FN). No amount of good latency compensates — this
                  is a hard stop, mirroring the directive's P8 release gates.
- **prototype** : security gate passes, but reliability/latency are below the
                  pilot bar. Usable for demos, not for real users.
- **pilot**     : security gate passes AND reliability/latency meet the pilot bar.
                  Ready for an authorized, limited pilot with real users.
- **ga-candidate**: meets the stricter GA bar on every axis. (Not a GA claim —
                  a signal that the measured axes clear the higher bar.)

No axis is asserted as a guarantee: the security evals measure the security
software over a seed corpus, and the perf/cost numbers are a snapshot under the
load that produced them. The scorecard reports the inputs so a human decides.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Thresholds:
    # pilot bar
    max_error_rate: float = 0.0          # no request errors tolerated for a pilot
    max_p95_ms: float = 8000.0           # p95 latency ceiling (CPU serving)
    max_cost_per_1k_usd: float = 1.0     # est. cost per 1,000 requests
    # stricter GA bar
    ga_max_p95_ms: float = 3000.0
    ga_max_cost_per_1k_usd: float = 0.25


def _cost_per_1k(load):
    c = load.get("cost_estimate_client_side") or {}
    n = max(1, load.get("requests", 0))
    return (c.get("total_usd", 0.0) / n) * 1000.0


def score(load_summary, security_report, thresholds=None):
    """Return a scorecard dict. `load_summary` is pilot.loadtest.summarize(...)
    output; `security_report` is evals.security_evals.run_security_evals()."""
    t = thresholds or Thresholds()
    rel = load_summary.get("reliability", {})
    lat = load_summary.get("latency_ms", {})
    error_rate = rel.get("error_rate", 1.0)
    p95 = lat.get("p95", float("inf"))
    cost_1k = _cost_per_1k(load_summary)

    du = security_report.get("dual_use", {})
    security_gate = (security_report.get("all_passed", False)
                     and du.get("false_positives", 1) == 0
                     and du.get("false_negatives", 1) == 0)

    axes = {
        "security": {
            "pass_rate": security_report.get("pass_rate", 0.0),
            "dual_use_fp": du.get("false_positives"),
            "dual_use_fn": du.get("false_negatives"),
            "gate_passed": security_gate,
        },
        "reliability": {"error_rate": error_rate,
                        "meets_pilot": error_rate <= t.max_error_rate},
        "latency": {"p95_ms": p95, "meets_pilot": p95 <= t.max_p95_ms,
                    "meets_ga": p95 <= t.ga_max_p95_ms},
        "cost": {"usd_per_1k": round(cost_1k, 5),
                 "meets_pilot": cost_1k <= t.max_cost_per_1k_usd,
                 "meets_ga": cost_1k <= t.ga_max_cost_per_1k_usd},
    }

    if not security_gate:
        level = "blocked"
    else:
        pilot_perf = (axes["reliability"]["meets_pilot"]
                      and axes["latency"]["meets_pilot"]
                      and axes["cost"]["meets_pilot"])
        ga_perf = (pilot_perf and axes["latency"]["meets_ga"]
                   and axes["cost"]["meets_ga"])
        level = "ga-candidate" if ga_perf else "pilot" if pilot_perf else "prototype"

    return {"level": level, "security_gate": security_gate, "axes": axes,
            "thresholds": {"max_error_rate": t.max_error_rate,
                           "max_p95_ms": t.max_p95_ms,
                           "max_cost_per_1k_usd": t.max_cost_per_1k_usd}}


def format_scorecard(card):
    a = card["axes"]
    lines = [f"PILOT READINESS: {card['level'].upper()}", "=" * 40]
    s = a["security"]
    lines.append(f"  security    : gate {'PASS' if s['gate_passed'] else 'FAIL'}  "
                 f"(pass_rate {s['pass_rate']}, dual-use FP={s['dual_use_fp']} "
                 f"FN={s['dual_use_fn']})")
    r = a["reliability"]
    lines.append(f"  reliability : error_rate={r['error_rate']}  "
                 f"{'OK' if r['meets_pilot'] else 'BELOW BAR'}")
    l = a["latency"]
    lines.append(f"  latency     : p95={l['p95_ms']}ms  "
                 f"pilot={'OK' if l['meets_pilot'] else 'NO'} "
                 f"ga={'OK' if l['meets_ga'] else 'NO'}")
    c = a["cost"]
    lines.append(f"  cost        : ${c['usd_per_1k']}/1k req  "
                 f"pilot={'OK' if c['meets_pilot'] else 'NO'} "
                 f"ga={'OK' if c['meets_ga'] else 'NO'}")
    if not card["security_gate"]:
        lines.append("  -> BLOCKED: a security gate failed; fix before any pilot.")
    return "\n".join(lines)
