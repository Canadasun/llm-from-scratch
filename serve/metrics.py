"""
Stage-4 pilot instrumentation: in-process reliability / latency / cost metrics.

The pilot gate in the platform directive asks for numbers on **reliability,
latency and cost** (security is covered by the enterprise tests + the pilot
probes). This module is the server-side half: a thread-safe collector plus a
transparent wrapper that times every request the serving layer handles.

Design rules it keeps:
- **It composes, it does not change behaviour.** `MeteredInferenceService` wraps
  the existing (secured or plain) service, forwards every call and its result
  unchanged, and only *observes* status, latency and token counts. Remove it and
  the endpoint behaves identically.
- **Aggregate only — no tenant data, no prompts, no completions** ever enter the
  metrics. A snapshot is safe to expose at an open `/metrics` endpoint because it
  leaks nothing a health check doesn't: only counts, rates and timings. This
  keeps tenant isolation intact (a per-tenant breakdown could leak existence).
- **Cost is an explicit, documented *estimate*,** not a billing figure. On a
  scale-to-zero consumption container, cost tracks active (busy) compute time, so
  we accumulate measured busy-seconds and multiply by a configurable rate. The
  rate and the caveat travel with the number.
- **Bounded memory.** Latency samples live in a fixed-size ring, so percentiles
  are computed from a bounded recent window regardless of how long it runs.

Thread-safety: the HTTP server is threaded, so every mutation is under one lock.
The clock is injectable for deterministic tests.
"""

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CostModel:
    """A documented estimate of cost per unit of work, defaulting to Azure
    Container Apps *consumption* pricing (pay only for active resources; idle
    replicas scaled to zero cost nothing). These are approximate list prices in
    USD and are meant to give the pilot an order-of-magnitude figure, not a bill.

    Billing on consumption is per active vCPU-second and GiB-second, so we charge
    the *measured busy time* (sum of request latencies) against the provisioned
    shape (vcpu / memory_gib). Request charges apply after a monthly free grant.
    """
    vcpu: float = 1.0                     # provisioned vCPU per replica
    memory_gib: float = 2.0               # provisioned memory per replica
    usd_per_vcpu_second: float = 0.000024
    usd_per_gib_second: float = 0.000003
    usd_per_million_requests: float = 0.40
    free_requests_per_month: int = 2_000_000

    def estimate(self, active_seconds, requests):
        """Return a cost breakdown for `active_seconds` of busy compute spread
        over `requests` requests. Idle time is free (scale-to-zero), so only
        busy time is charged."""
        compute = active_seconds * (
            self.vcpu * self.usd_per_vcpu_second
            + self.memory_gib * self.usd_per_gib_second)
        billable_requests = max(0, requests - self.free_requests_per_month)
        request_cost = billable_requests / 1_000_000 * self.usd_per_million_requests
        return {
            "active_seconds": round(active_seconds, 3),
            "compute_usd": round(compute, 6),
            "request_usd": round(request_cost, 6),
            "total_usd": round(compute + request_cost, 6),
            "model": {
                "vcpu": self.vcpu,
                "memory_gib": self.memory_gib,
                "usd_per_vcpu_second": self.usd_per_vcpu_second,
                "usd_per_gib_second": self.usd_per_gib_second,
                "usd_per_million_requests": self.usd_per_million_requests,
                "free_requests_per_month": self.free_requests_per_month,
            },
            "caveat": ("estimate only: active-compute-time model for a scale-to-"
                       "zero consumption container; not a billing figure"),
        }


def _percentile(sorted_samples, q):
    """Linear-interpolated percentile (q in [0,100]) over an already-sorted list.
    Returns 0.0 for an empty list so a snapshot is always well-formed."""
    n = len(sorted_samples)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_samples[0]
    rank = (q / 100.0) * (n - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return sorted_samples[lo]
    frac = rank - lo
    return sorted_samples[lo] * (1 - frac) + sorted_samples[hi] * frac


@dataclass
class Metrics:
    """Thread-safe, aggregate-only request metrics with a bounded latency window.

    `sample_size` caps how many recent latency samples are retained for the
    percentile computation (bounded memory); counts and sums are exact and
    unbounded. `clock` is injectable for tests."""
    sample_size: int = 4096
    cost_model: CostModel = field(default_factory=CostModel)
    clock: callable = time.monotonic

    def __post_init__(self):
        self._lock = threading.Lock()
        self._start = self.clock()
        self._total = 0
        self._status_counts = {}          # exact status -> count
        self._latencies = deque(maxlen=self.sample_size)   # ms, recent window
        self._latency_sum_ms = 0.0        # exact, all requests (for busy time)
        self._latency_count = 0           # exact count behind the sum
        self._completion_tokens = 0

    def record(self, status, latency_ms, completion_tokens=0):
        """Observe one handled request. `latency_ms` is end-to-end wall time for
        the request as measured by the wrapper; `completion_tokens` is 0 for any
        non-success path. Never raises on bad inputs — metrics must not break the
        request path — it clamps instead."""
        status = int(status)
        latency_ms = max(0.0, float(latency_ms))
        completion_tokens = max(0, int(completion_tokens or 0))
        with self._lock:
            self._total += 1
            self._status_counts[status] = self._status_counts.get(status, 0) + 1
            self._latencies.append(latency_ms)
            self._latency_sum_ms += latency_ms
            self._latency_count += 1
            self._completion_tokens += completion_tokens

    def snapshot(self):
        """Return a JSON-serialisable aggregate view. Safe to expose publicly:
        contains only counts, rates, timings and a cost estimate."""
        with self._lock:
            now = self.clock()
            uptime = max(1e-9, now - self._start)
            total = self._total
            status_counts = dict(self._status_counts)
            samples = sorted(self._latencies)
            latency_sum_ms = self._latency_sum_ms
            tokens = self._completion_tokens

        ok = status_counts.get(200, 0)
        client_err = sum(c for s, c in status_counts.items() if 400 <= s < 500)
        server_err = sum(c for s, c in status_counts.items() if s >= 500)
        errors = client_err + server_err
        active_seconds = latency_sum_ms / 1000.0

        latency = {
            "count": len(samples),
            "mean_ms": round(sum(samples) / len(samples), 2) if samples else 0.0,
            "p50_ms": round(_percentile(samples, 50), 2),
            "p90_ms": round(_percentile(samples, 90), 2),
            "p95_ms": round(_percentile(samples, 95), 2),
            "p99_ms": round(_percentile(samples, 99), 2),
            "max_ms": round(samples[-1], 2) if samples else 0.0,
            "window": self.sample_size,
        }
        return {
            "uptime_seconds": round(uptime, 2),
            "requests": total,
            "requests_per_second": round(total / uptime, 4),
            "status_counts": {str(s): c for s, c in sorted(status_counts.items())},
            "reliability": {
                "success": ok,
                "client_errors_4xx": client_err,
                "server_errors_5xx": server_err,
                "error_rate": round(errors / total, 4) if total else 0.0,
                "server_error_rate": round(server_err / total, 4) if total else 0.0,
                "success_rate": round(ok / total, 4) if total else 0.0,
                "unauthorized_401": status_counts.get(401, 0),
                "forbidden_403": status_counts.get(403, 0),
                "rate_limited_429": status_counts.get(429, 0),
            },
            "latency": latency,
            "throughput": {
                "completion_tokens": tokens,
                "tokens_per_second": round(tokens / uptime, 2),
            },
            "cost": self.cost_model.estimate(active_seconds, total),
        }


class MeteredInferenceService:
    """Transparent instrumentation wrapper. Place it OUTERMOST (around the secured
    service) so it sees every outcome, including 401/403 before the model runs:

        MeteredInferenceService(SecuredInferenceService(InferenceService(...)))

    It forwards `handle_generate` and `health` unchanged, mirrors the inner
    service's `requires_auth` so the HTTP adapter still passes a bearer token
    through, and adds a `metrics()` method the adapter exposes at `/metrics`.
    """

    def __init__(self, inner, metrics=None):
        self.inner = inner
        self.metrics = metrics or Metrics()

    @property
    def requires_auth(self):
        # Mirror the inner service so the HTTP adapter's bearer-token handling is
        # unchanged by wrapping. A plain inner service has no such attribute.
        return getattr(self.inner, "requires_auth", False)

    def health(self):
        return self.inner.health()

    def metrics_snapshot(self):
        return self.metrics.snapshot()

    def handle_generate(self, payload, **kwargs):
        # Measure end-to-end wall time here, independent of any latency the inner
        # service may self-report, so the number reflects the whole handled path.
        t0 = time.monotonic()
        status, body = self.inner.handle_generate(payload, **kwargs)
        dt_ms = (time.monotonic() - t0) * 1000.0
        tokens = body.get("completion_tokens", 0) if isinstance(body, dict) else 0
        self.metrics.record(status, dt_ms, tokens)
        return status, body
