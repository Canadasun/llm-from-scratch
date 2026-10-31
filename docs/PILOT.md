# Stage 4 — authorized pilot instrumentation

The platform directive's pilot gate asks for evidence on four axes —
**reliability, security, latency, cost** — measured on the *live* endpoint, not
in theory. This stage adds the instrumentation to produce those numbers.

Two halves, composed so neither changes the model or the request behaviour:

| half | where | what it does |
|---|---|---|
| server-side | `serve/metrics.py` | aggregate, thread-safe metrics + a transparent `MeteredInferenceService` wrapper + `GET /metrics` |
| client-side | `pilot/` | drives the live endpoint under load; reports reliability / security / latency / cost from the caller's side |

## Server-side metrics (`serve/metrics.py`)

`MeteredInferenceService` wraps the (secured or plain) service **outermost**, so
it observes every outcome — including a `401`/`403` decided *before* the model
runs — and forwards status and body completely unchanged. Remove it and the
endpoint behaves identically; it only measures.

- **Aggregate only.** The snapshot is counts, rates, timings and a cost estimate
  — **no prompts, no completions, no tenant ids**. That is why `/metrics` can be
  open like `/healthz`: it leaks nothing a health check doesn't, and a per-tenant
  breakdown (which could leak tenant existence) is deliberately absent.
- **Bounded memory.** Latency percentiles are computed from a fixed-size ring of
  recent samples; counts and sums are exact and unbounded.
- **Cost is an explicit estimate**, not a bill. On a scale-to-zero consumption
  container, cost tracks *active* compute time, so it accumulates measured
  busy-seconds × a configurable `CostModel` rate (default: Azure Container Apps
  consumption list prices). The rate and the caveat travel with every number.

Turn it on:

```bash
python -m serve --secured --metrics            # GET /metrics now served
curl -s localhost:8000/metrics | python -m json.tool
```

## Client-side pilot (`python -m pilot`)

Runs against the **deployed endpoint as it is today** — no redeploy, no server
change required — using only the standard library.

```bash
# full pilot: security probes + load + (if present) server /metrics
python -m pilot --url https://<app>.azurecontainerapps.io --key sk_... -n 100 -c 8

# security smoke check only
python -m pilot --url https://<app>... --key sk_... --probe-only

# machine-readable
python -m pilot --url ... --key ... --json > pilot.json
```

The key comes from `--key` or `PILOT_API_KEY` (so it need not hit shell history),
and nothing is written anywhere — the report goes to stdout. The process exits
non-zero if a security probe fails or the load run saw any error, so it can gate
a CI/CD step.

### Security probes — invariants as live assertions

Each probe asserts a property the **surrounding software** must enforce (never
the model's text), checked black-box over the wire:

- health is reachable and advertises `"auth": "required"`;
- **no credential → 401** (the model must not run unauthenticated);
- **a bad credential → 401** (a wrong key is not an identity);
- a valid credential → 200 (the happy path still works).

These caught a real mistake during development: a run accidentally pointed at a
*stale, unsecured* dev server returned `200` to the no-key probe, and the pilot
failed loudly instead of reporting bogus "healthy" numbers — which is exactly the
guardrail intended.

### What the load run reports

Client-side reliability (success/error rate, status breakdown), latency
percentiles (p50/p90/p95/p99/max), throughput (req/s and tok/s), and a
client-side cost estimate. If the endpoint exposes `/metrics`, the authoritative
server-side view is attached as well.

## A real run (local, real M3 checkpoint, CPU)

```
Security: ALL PASS (auth:required; no-key 401; bad-key 401; valid 200)
Load:     30 requests, 4 concurrent
  reliability : success_rate=1.0  error_rate=0.0
  latency(ms) : p50≈4950  p95≈6087  p99≈6103
  throughput  : 0.76 req/s, 24 tok/s
  cost (est.) : ~$0.005 over ~151s busy compute
```

The latency reflects the from-scratch model generating on **CPU** (the deploy
target is a CPU consumption container); it is the honest baseline a pilot exists
to surface, and the number to beat if the model is later served on GPU or quantised.

## AI-security eval suite (`python -m evals.security_evals`)

The pilot's security axis is more than "auth returns 401". `evals/security_evals.py`
runs the real enterprise pipeline (authn → authz → audit) around a **stub inner
service** — because every invariant is enforced by the surrounding software, not
by model output — and reports a pass rate plus measured dual-use FP/FN:

- **prompt-injection resistance (system level):** adversarial prompt text never
  changes an authorization outcome (no key → 401, bad key → 401, a key scoped
  away from infer → 403), whatever the prompt says;
- **authorization ceiling:** a scoped key cannot exceed its grant via the body;
- **cross-tenant isolation:** the P0 gate, re-run as a measured eval (every
  tenant-A action on a tenant-B resource → 404);
- **audit redaction / no exfiltration:** a secret placed in a prompt never lands
  in the audit trail; the open health endpoint leaks no tenant data;
- **dual-use correctness:** the policy must BLOCK truly harmful categories and
  ALLOW legitimate security work (this model is *for* security professionals),
  measured as false positives (legit blocked) and false negatives (harmful
  allowed) over a labelled corpus. The bar is **FP = 0** on security topics.

It exits non-zero on any failure, so it gates CI (and is also asserted by
`tests/test_security_evals.py`). Honest scope: it measures the security
*software* over a seed corpus — not a model-level injection-immunity guarantee.

## Pilot readiness scorecard (`--scorecard`)

`pilot/scorecard.py` combines the four axes into one threshold-based verdict
(directive P8), run with `python -m pilot --url … --key … --scorecard`:

- **blocked** — a security gate failed (cross-tenant leak, auth bypass, or a
  dual-use FP/FN). A hard stop; good latency never compensates.
- **prototype** — security passes, but reliability/latency are below the pilot bar.
- **pilot** — security passes AND reliability/latency meet the pilot bar.
- **ga-candidate** — clears the stricter GA bar on every axis (a signal, not a
  GA claim).

The scorecard prints the inputs behind the verdict so a human makes the call; no
axis is asserted as a guarantee.

## Honest scope

This measures the endpoint; it does not change its capacity. The server metrics
are **per-process** (per replica), like the in-memory rate limiter and key store
— a fleet-wide view needs the shared metrics/quota store noted in
`ENTERPRISE_SECURITY.md`. The cost figure is an order-of-magnitude estimate from
list prices, not a billing export. The pilot client performs outbound HTTP to a
**user-supplied** URL, so it is kept out of the self-reviewed package set by
design.
