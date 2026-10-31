"""
Client-side pilot harness: drive the live `/generate` endpoint under load and
report the numbers the Stage-4 pilot gate needs — reliability, latency, security
and cost — measured from the caller's side.

Why client-side: it works against the deployed endpoint *as it is today*, with no
redeploy and no server changes. If the server was started with `--metrics` the
harness also pulls `GET /metrics` for the authoritative server-side view
(including active-compute cost); otherwise it estimates cost from observed busy
time so a report is always produced.

Standard library only (urllib + threads) — no extra dependency, so it runs
anywhere the repo does. It is a measurement client, not a server, so it is kept
out of the self-reviewed package set; it performs outbound HTTP to a
*user-supplied* URL by design.
"""

import json
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, field

from serve.metrics import CostModel, _percentile


@dataclass
class Result:
    status: int
    latency_ms: float
    ok: bool
    completion_tokens: int = 0
    error: str = ""


def _post_generate(base_url, api_key, prompt, max_new_tokens, seed, timeout):
    """One POST /generate. Returns a Result; network/HTTP errors become a Result
    with a synthetic status (0 for transport failure) rather than raising, so one
    bad request never aborts the run."""
    url = base_url.rstrip("/") + "/generate"
    body = json.dumps({"prompt": prompt, "max_new_tokens": max_new_tokens,
                       "seed": seed}).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", "Bearer " + api_key)
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (user URL, by design)
            payload = json.loads(resp.read().decode("utf-8"))
            dt = (time.monotonic() - t0) * 1000.0
            return Result(resp.status, dt, resp.status == 200,
                          int(payload.get("completion_tokens", 0) or 0))
    except urllib.error.HTTPError as e:
        dt = (time.monotonic() - t0) * 1000.0
        return Result(e.code, dt, False, error=f"http_{e.code}")
    except Exception as e:  # transport error, timeout, bad JSON
        dt = (time.monotonic() - t0) * 1000.0
        return Result(0, dt, False, error=type(e).__name__)


def _get_json(base_url, path, api_key=None, timeout=10):
    """GET a JSON endpoint. Returns (status, obj_or_None)."""
    url = base_url.rstrip("/") + path
    req = urllib.request.Request(url, method="GET")
    if api_key:
        req.add_header("Authorization", "Bearer " + api_key)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, None
    except Exception:
        return 0, None


def security_probes(base_url, api_key, timeout=15):
    """Black-box security checks on the deployed endpoint. Each probe asserts a
    *property the surrounding software must enforce* (never the model's text):

    - health is reachable and advertises that auth is required;
    - no credential  -> 401 (the model must not run unauthenticated);
    - a bad credential -> 401 (a wrong key is not a valid identity);
    - a valid credential -> 200 (the happy path still works).

    Returns a list of {name, pass, detail}. These encode the enterprise
    invariants as live assertions against the real endpoint.
    """
    probes = []

    s, h = _get_json(base_url, "/healthz", timeout=timeout)
    probes.append({
        "name": "health_reachable_and_auth_advertised",
        "pass": s == 200 and isinstance(h, dict) and h.get("auth") == "required",
        "detail": f"status={s} auth={(h or {}).get('auth')}",
    })

    r = _post_generate(base_url, None, "zero trust ", 8, 1, timeout)
    probes.append({
        "name": "no_key_rejected_401",
        "pass": r.status == 401,
        "detail": f"status={r.status}",
    })

    r = _post_generate(base_url, "sk_not_a_real_key", "zero trust ", 8, 1, timeout)
    probes.append({
        "name": "bad_key_rejected_401",
        "pass": r.status == 401,
        "detail": f"status={r.status}",
    })

    if api_key:
        r = _post_generate(base_url, api_key, "zero trust ", 8, 1, timeout)
        probes.append({
            "name": "valid_key_accepted_200",
            "pass": r.status == 200,
            "detail": f"status={r.status}",
        })
    return probes


@dataclass
class LoadConfig:
    base_url: str
    api_key: str = ""
    requests: int = 50
    concurrency: int = 4
    prompt: str = "zero trust architecture "
    max_new_tokens: int = 48
    timeout: float = 60.0
    seeds: list = field(default_factory=lambda: list(range(1000)))


def run_load(cfg):
    """Fire cfg.requests requests across cfg.concurrency worker threads against
    /generate. Returns the list of Result. Uses deterministic, incrementing seeds
    (no RNG) so a run is reproducible."""
    results = []
    results_lock = threading.Lock()
    next_i = [0]
    index_lock = threading.Lock()

    def worker():
        while True:
            with index_lock:
                i = next_i[0]
                if i >= cfg.requests:
                    return
                next_i[0] += 1
            seed = cfg.seeds[i % len(cfg.seeds)]
            r = _post_generate(cfg.base_url, cfg.api_key, cfg.prompt,
                               cfg.max_new_tokens, seed, cfg.timeout)
            with results_lock:
                results.append(r)

    threads = [threading.Thread(target=worker)
               for _ in range(max(1, cfg.concurrency))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def summarize(results, wall_seconds, cost_model=None):
    """Turn raw Results into the client-side pilot report dict."""
    cost_model = cost_model or CostModel()
    n = len(results)
    ok = sum(1 for r in results if r.ok)
    status_counts = Counter(r.status for r in results)
    lat = sorted(r.latency_ms for r in results)
    busy_s = sum(r.latency_ms for r in results) / 1000.0
    tokens = sum(r.completion_tokens for r in results)
    wall = max(1e-9, wall_seconds)
    errors = n - ok
    return {
        "requests": n,
        "wall_seconds": round(wall, 3),
        "throughput_rps": round(n / wall, 3),
        "reliability": {
            "success": ok,
            "errors": errors,
            "error_rate": round(errors / n, 4) if n else 0.0,
            "success_rate": round(ok / n, 4) if n else 0.0,
            "status_counts": {str(k): v for k, v in sorted(status_counts.items())},
        },
        "latency_ms": {
            "count": len(lat),
            "mean": round(sum(lat) / len(lat), 2) if lat else 0.0,
            "p50": round(_percentile(lat, 50), 2),
            "p90": round(_percentile(lat, 90), 2),
            "p95": round(_percentile(lat, 95), 2),
            "p99": round(_percentile(lat, 99), 2),
            "max": round(lat[-1], 2) if lat else 0.0,
        },
        "throughput": {
            "completion_tokens": tokens,
            "tokens_per_second": round(tokens / wall, 2),
        },
        "cost_estimate_client_side": cost_model.estimate(busy_s, n),
    }


def run_pilot(cfg):
    """Full pilot: security probes, then a load run, then (if available) the
    server-side /metrics view. Returns one report dict."""
    report = {"endpoint": cfg.base_url, "config": {
        "requests": cfg.requests, "concurrency": cfg.concurrency,
        "max_new_tokens": cfg.max_new_tokens}}

    report["security"] = security_probes(cfg.base_url, cfg.api_key, cfg.timeout)
    report["security_all_pass"] = all(p["pass"] for p in report["security"])

    t0 = time.monotonic()
    results = run_load(cfg)
    wall = time.monotonic() - t0
    report["load"] = summarize(results, wall)

    s, server_metrics = _get_json(cfg.base_url, "/metrics", timeout=cfg.timeout)
    report["server_metrics"] = server_metrics if s == 200 else None
    report["server_metrics_available"] = server_metrics is not None
    return report


def format_report(report):
    """Human-readable one-screen summary."""
    lines = []
    lines.append(f"PILOT REPORT — {report['endpoint']}")
    lines.append("=" * 60)
    lines.append("Security (invariants enforced in software, not the model):")
    for p in report["security"]:
        mark = "PASS" if p["pass"] else "FAIL"
        lines.append(f"  [{mark}] {p['name']}  ({p['detail']})")
    lines.append(f"  -> {'ALL PASS' if report['security_all_pass'] else 'FAILURES PRESENT'}")
    lines.append("")

    ld = report["load"]
    rel, lat = ld["reliability"], ld["latency_ms"]
    lines.append(f"Load: {ld['requests']} requests, "
                 f"{report['config']['concurrency']} concurrent, "
                 f"{ld['wall_seconds']}s wall")
    lines.append(f"  reliability : success_rate={rel['success_rate']} "
                 f"error_rate={rel['error_rate']}  statuses={rel['status_counts']}")
    lines.append(f"  latency(ms) : p50={lat['p50']} p95={lat['p95']} "
                 f"p99={lat['p99']} max={lat['max']} mean={lat['mean']}")
    lines.append(f"  throughput  : {ld['throughput_rps']} req/s, "
                 f"{ld['throughput']['tokens_per_second']} tok/s")
    c = ld["cost_estimate_client_side"]
    lines.append(f"  cost (est.) : ${c['total_usd']} over {c['active_seconds']}s "
                 f"busy  ({c['caveat']})")
    lines.append("")
    if report["server_metrics_available"]:
        lines.append("Server /metrics available — authoritative server-side view "
                     "included in JSON.")
    else:
        lines.append("Server /metrics not exposed (endpoint started without "
                     "--metrics): cost/latency above are client-side estimates.")
    return "\n".join(lines)
