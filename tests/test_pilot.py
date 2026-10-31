"""
Tests for the Stage-4 pilot instrumentation:

- `serve/metrics.py`  — the aggregate metrics collector + cost model, and the
  transparent `MeteredInferenceService` wrapper;
- `pilot/loadtest.py` — the client harness, exercised end-to-end over real HTTP
  against an in-process server (a fake auth'd service, so no checkpoint/torch).

The invariants under test:
- percentiles / rates / cost are computed correctly and the snapshot is always
  well-formed (even empty);
- recording is thread-safe (no lost counts under concurrency);
- the wrapper is transparent — it forwards status/body unchanged, mirrors
  `requires_auth`, passes the bearer key through, and only observes;
- the pilot client reports reliability/latency/throughput and its security
  probes assert the real 401/200 behaviour over the wire;
- GET /metrics is served only when the service exposes it, and never leaks.
"""

import json
import threading
import time
from http.server import ThreadingHTTPServer

import pytest

from serve.http_server import make_handler
from serve.metrics import CostModel, Metrics, MeteredInferenceService, _percentile


# ---------- percentiles ----------

def test_percentile_edges_and_interpolation():
    assert _percentile([], 50) == 0.0
    assert _percentile([42.0], 99) == 42.0
    s = [10.0, 20.0, 30.0, 40.0]
    assert _percentile(s, 0) == 10.0
    assert _percentile(s, 100) == 40.0
    assert _percentile(s, 50) == pytest.approx(25.0)   # between 20 and 30


# ---------- metrics collector ----------

def test_snapshot_is_wellformed_when_empty():
    snap = Metrics().snapshot()
    assert snap["requests"] == 0
    assert snap["reliability"]["error_rate"] == 0.0
    assert snap["latency"]["p95_ms"] == 0.0
    assert snap["cost"]["total_usd"] == 0.0


def test_reliability_breakdown_and_error_rate():
    m = Metrics()
    for _ in range(7):
        m.record(200, 10.0, completion_tokens=5)
    m.record(401, 1.0)
    m.record(403, 1.0)
    m.record(500, 2.0)
    snap = m.snapshot()
    rel = snap["reliability"]
    assert snap["requests"] == 10
    assert rel["success"] == 7
    assert rel["client_errors_4xx"] == 2
    assert rel["server_errors_5xx"] == 1
    assert rel["error_rate"] == pytest.approx(0.3)
    assert rel["server_error_rate"] == pytest.approx(0.1)
    assert rel["unauthorized_401"] == 1 and rel["forbidden_403"] == 1
    assert snap["throughput"]["completion_tokens"] == 35


def test_latency_percentiles_from_known_samples():
    m = Metrics()
    for v in range(1, 101):          # 1..100 ms
        m.record(200, float(v))
    lat = m.snapshot()["latency"]
    assert lat["count"] == 100
    assert lat["p50_ms"] == pytest.approx(50.5, abs=0.6)
    assert lat["p99_ms"] == pytest.approx(99.0, abs=1.0)
    assert lat["max_ms"] == 100.0


def test_latency_window_is_bounded():
    m = Metrics(sample_size=100)
    for _ in range(1000):
        m.record(200, 5.0)
    snap = m.snapshot()
    assert snap["requests"] == 1000               # exact count unbounded
    assert snap["latency"]["count"] == 100         # sample window bounded


def test_record_clamps_bad_input_and_never_raises():
    m = Metrics()
    m.record(200, -5.0, completion_tokens=-3)      # negatives clamp to 0
    m.record("200", 1.0, completion_tokens=None)   # coercible / None token
    snap = m.snapshot()
    assert snap["requests"] == 2
    assert snap["throughput"]["completion_tokens"] == 0


def test_cost_estimate_is_zero_at_start_and_grows_with_busy_time():
    model = CostModel(vcpu=1.0, memory_gib=2.0)
    assert model.estimate(0.0, 0)["total_usd"] == 0.0
    lo = model.estimate(10.0, 100)["total_usd"]
    hi = model.estimate(20.0, 100)["total_usd"]
    assert hi > lo > 0.0


def test_cost_free_request_grant_applies():
    model = CostModel(free_requests_per_month=1000)
    # under the free grant -> no request charge, only compute
    c = model.estimate(1.0, 500)
    assert c["request_usd"] == 0.0


def test_record_is_threadsafe():
    m = Metrics()
    n_threads, per = 16, 500

    def worker():
        for _ in range(per):
            m.record(200, 1.0, completion_tokens=1)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    snap = m.snapshot()
    assert snap["requests"] == n_threads * per
    assert snap["throughput"]["completion_tokens"] == n_threads * per


# ---------- metered wrapper transparency ----------

class _FakeSecured:
    """A minimal stand-in for SecuredInferenceService: auth'd, token-bearing,
    returns a 401 without a key and a 200 with completion_tokens otherwise."""
    requires_auth = True

    def __init__(self):
        self.calls = []

    def health(self):
        return {"status": "ok", "auth": "required"}

    def handle_generate(self, payload, api_key=None, client="-"):
        self.calls.append((payload, api_key, client))
        if api_key != "good":
            return 401, {"error": "unauthorized", "request_id": "r"}
        return 200, {"completion": "x", "completion_tokens": 7, "request_id": "r"}


def test_wrapper_mirrors_requires_auth_and_forwards_bearer():
    inner = _FakeSecured()
    m = MeteredInferenceService(inner)
    assert m.requires_auth is True                 # mirrored for the HTTP adapter
    status, body = m.handle_generate({"prompt": "hi"}, api_key="good", client="c")
    assert status == 200 and body["completion_tokens"] == 7
    # the kwargs were forwarded unchanged
    assert inner.calls[-1] == ({"prompt": "hi"}, "good", "c")


def test_wrapper_records_outcomes_including_pre_model_401():
    inner = _FakeSecured()
    m = MeteredInferenceService(inner)
    m.handle_generate({"prompt": "hi"}, api_key="good")
    m.handle_generate({"prompt": "hi"}, api_key=None)     # 401 before model
    snap = m.metrics_snapshot()
    assert snap["requests"] == 2
    assert snap["reliability"]["success"] == 1
    assert snap["reliability"]["unauthorized_401"] == 1
    assert snap["throughput"]["completion_tokens"] == 7


def test_wrapper_without_requires_auth_on_plain_inner():
    class Plain:
        def handle_generate(self, payload, client="-"):
            return 200, {"completion_tokens": 1}
        def health(self):
            return {"status": "ok"}
    m = MeteredInferenceService(Plain())
    assert m.requires_auth is False


# ---------- end-to-end over real HTTP (no torch) ----------

@pytest.fixture
def live_server():
    """A real ThreadingHTTPServer serving a metered fake auth'd service on an
    ephemeral port. Yields (base_url, metered)."""
    metered = MeteredInferenceService(_FakeSecured())
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(metered))
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{port}", metered
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_metrics_endpoint_served_only_when_supported(live_server):
    from pilot.loadtest import _get_json
    base, _ = live_server
    s, obj = _get_json(base, "/metrics")
    assert s == 200 and "reliability" in obj and "cost" in obj
    # and it leaks nothing request-level: no prompt text, and the only
    # "completion" token is the aggregate count field (never a completion body).
    blob = json.dumps(obj)
    assert "prompt" not in blob
    assert blob.replace("completion_tokens", "").find("completion") == -1


def test_metrics_endpoint_404_when_unsupported():
    # A service without metrics_snapshot must NOT expose /metrics.
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(_FakeSecured()))
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        from pilot.loadtest import _get_json
        s, _ = _get_json(f"http://127.0.0.1:{port}", "/metrics")
        assert s == 404
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_security_probes_against_live_endpoint(live_server):
    from pilot.loadtest import security_probes
    base, _ = live_server
    probes = security_probes(base, api_key="good")
    by = {p["name"]: p["pass"] for p in probes}
    assert by["health_reachable_and_auth_advertised"] is True
    assert by["no_key_rejected_401"] is True
    assert by["bad_key_rejected_401"] is True
    assert by["valid_key_accepted_200"] is True


def test_run_pilot_end_to_end(live_server):
    from pilot.loadtest import LoadConfig, run_pilot
    base, metered = live_server
    cfg = LoadConfig(base_url=base, api_key="good", requests=20, concurrency=4)
    report = run_pilot(cfg)
    assert report["security_all_pass"] is True
    assert report["load"]["requests"] == 20
    assert report["load"]["reliability"]["success_rate"] == 1.0
    assert report["load"]["latency_ms"]["p95"] >= 0.0
    # server-side /metrics was pulled and agrees the load happened
    assert report["server_metrics_available"] is True
    assert report["server_metrics"]["requests"] >= 20


def test_run_pilot_reports_errors_when_key_bad(live_server):
    from pilot.loadtest import LoadConfig, run_pilot
    base, _ = live_server
    cfg = LoadConfig(base_url=base, api_key="", requests=10, concurrency=2)
    report = run_pilot(cfg)
    # no key -> every /generate is a 401 -> error_rate 1.0, security still 'pass'
    # for the no-key/bad-key probes but the valid-key probe is absent
    assert report["load"]["reliability"]["error_rate"] == 1.0
    assert report["load"]["reliability"]["status_counts"].get("401") == 10
