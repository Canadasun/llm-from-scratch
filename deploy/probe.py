"""Mandatory authenticated functional probes; no cloud management operations."""
import argparse
import json
import math
import os
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

# Exact expected behavior, maintained independently of model answers.
SMOKE_CASES = [
    {"prompt": "Reply with exactly: Use parameterized queries.",
     "expected": "Use parameterized queries."},
    {"prompt": "Reply with exactly: Never execute untrusted Python on the host.",
     "expected": "Never execute untrusted Python on the host."},
]


def request(base, payload, key=None):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    req = Request(base + "/generate", data=json.dumps(payload).encode(), headers=headers)
    started = time.monotonic()
    try:
        with urlopen(req, timeout=40) as response:
            body = response.read(65537)
            if len(body) > 65536:
                raise ValueError("oversized response")
            return response.status, json.loads(body), time.monotonic() - started
    except HTTPError as exc:
        return exc.code, {}, time.monotonic() - started


def probe(base, key, samples=20, min_success=0.95, max_p95_ms=3000):
    url = urlsplit(base)
    if url.scheme != "https" or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError("a credential-free HTTPS revision URL is required")
    if not key:
        raise ValueError("PROBE_KEY is required; generation probes cannot be skipped")
    if type(samples) is not int or samples < 20 or not 0 < min_success <= 1 or not math.isfinite(max_p95_ms) or max_p95_ms <= 0:
        raise ValueError("invalid probe thresholds")
    base = base.rstrip("/")
    payload = {"prompt": "x", "max_new_tokens": 4}
    if request(base, payload)[0] != 401:
        raise ValueError("unauthenticated generation was not rejected")
    latencies = []
    successes = 0
    for i in range(samples):
        case = SMOKE_CASES[i % len(SMOKE_CASES)]
        status, body, seconds = request(base, {"prompt": case["prompt"],
            "max_new_tokens": 64, "seed": 1337, "temperature": 0.1}, key)
        latencies.append(seconds * 1000)
        if status == 200 and body.get("completion", "").strip() == case["expected"]:
            successes += 1
    p95 = sorted(latencies)[math.ceil(0.95 * samples) - 1]
    if successes / samples < min_success or p95 > max_p95_ms:
        raise ValueError(f"functional gate failed: {successes}/{samples}; p95={p95:.0f}ms")
    return {"success_rate": successes / samples, "p95_ms": p95}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--min-success", type=float, default=0.95)
    ap.add_argument("--max-p95-ms", type=float, default=3000)
    args = ap.parse_args()
    try:
        print(json.dumps(probe(args.base, os.environ.get("PROBE_KEY"), args.samples,
                               args.min_success, args.max_p95_ms)))
    except (ValueError, OSError) as exc:
        ap.exit(1, f"PROMOTION BLOCKED: {exc}\n")


if __name__ == "__main__":
    main()
