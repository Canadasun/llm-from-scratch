"""
Run the inference server.

    python -m serve                       # 127.0.0.1:8000, checkpoints/ckpt.pt
    python -m serve --port 8080 --ckpt path/to/ckpt.pt
    python -m serve --host 0.0.0.0        # expose (be deliberate about this)
    python -m serve --no-safety           # research mode; logged
    python -m serve --secured             # require an API key, tenant-scoped + audited
    python -m serve --secured --key-db keys.db   # + PERSISTENT keys (survive restart)

Then:
    curl localhost:8000/healthz
    curl -s localhost:8000/generate -H 'content-type: application/json' \
         -d '{"prompt":"Zero trust architecture ","max_new_tokens":80,"seed":1}'

With --secured, a demo tenant + key are printed at startup; pass the key as:
    curl -s localhost:8000/generate -H 'authorization: Bearer sk_...' \
         -H 'content-type: application/json' -d '{"prompt":"TLS ","seed":1}'
"""

import argparse
import logging

from serve.engine import DEFAULT_CKPT, InferenceEngine
from serve.http_server import serve
from serve.ratelimit import RateLimiter
from serve.secured import SecuredInferenceService
from serve.service import InferenceService


def _make_store(key_db):
    """Return the store backing --secured. A path gives a PERSISTENT SQLite store
    (keys survive restarts); no path gives the in-memory store (ephemeral demo)."""
    import time
    if key_db:
        from enterprise.sqlstore import SqlStore
        return SqlStore(key_db, clock=time.time)
    from enterprise.store import Store
    return Store()


def _bootstrap_demo_tenant(store):
    """Ensure a demo tenant/principal exists and return a usable API key.

    - In-memory store, or a fresh DB with no demo key yet: create the tenant +
      principal, mint a key, and return (plaintext, is_new=True) so it is printed
      once (only the hash is stored).
    - A persistent DB that already has the demo principal: do NOT recreate or
      re-mint — existing keys stay valid across the restart — and return
      (None, is_new=False). This is what makes --secured keys STABLE.

    Real deployments issue keys through the admin flow (enterprise.keymgmt); this
    is just the runnable demo identity.
    """
    from enterprise.keys import generate_key
    from enterprise.models import ApiKey, Principal, Tenant

    if store.get_principal("demo-user") is not None:
        return None, False                      # already provisioned (persisted)

    store.add_tenant(Tenant("demo", "Demo Tenant"))
    store.add_principal(Principal("demo-user", "demo", ("member",)))
    import time
    plaintext, key_hash = generate_key()
    store.add_api_key(ApiKey("demo-key", key_hash, "demo", "demo-user", (),
                             created_at=time.time()))
    return plaintext, True


def main():
    ap = argparse.ArgumentParser(description="llm-from-scratch inference server")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--rate-capacity", type=int, default=10,
                    help="max burst requests per client")
    ap.add_argument("--rate-per-sec", type=float, default=1.0,
                    help="sustained requests/second per client")
    ap.add_argument("--no-safety", action="store_true",
                    help="disable the use-safety policy (research use; logged)")
    ap.add_argument("--secured", action="store_true",
                    help="require an API key; requests are tenant-scoped + audited")
    ap.add_argument("--key-db", default=None,
                    help="path to a SQLite key store (persists keys across "
                         "restarts); omit for an in-memory demo store")
    ap.add_argument("--metrics", action="store_true",
                    help="collect aggregate reliability/latency/cost metrics and "
                         "expose them at GET /metrics (Stage-4 pilot)")
    ap.add_argument("--max-concurrent", type=int, default=2,
                    help="maximum simultaneous generations; excess requests get 503")
    ap.add_argument("--max-connections", type=int, default=32,
                    help="maximum HTTP connection workers, including idle keep-alive clients")
    ap.add_argument("--quantize", action="store_true",
                    help="dynamic int8 quantization (CPU): ~2x faster generation, "
                         "negligible quality change")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(message)s")

    log = logging.getLogger("serve")
    if args.no_safety:
        log.warning("use-safety policy DISABLED (--no-safety)")

    engine = InferenceEngine.from_checkpoint(args.ckpt, device=args.device,
                                             quantize=args.quantize)
    if args.quantize:
        log.info("int8 dynamic quantization enabled (faster CPU generation)")
    limiter = RateLimiter(capacity=args.rate_capacity, refill_per_sec=args.rate_per_sec)
    service = InferenceService(engine, rate_limiter=limiter, safety=not args.no_safety,
                               max_concurrent=args.max_concurrent)

    if args.secured:
        import json as _json
        from enterprise.audit import AuditLog
        store = _make_store(args.key_db)
        # Durable-ish sink: emit every audit event as one JSON line to the
        # "audit" logger (stdout), so the trail survives the process in the
        # container/host log aggregator instead of living only in bounded memory.
        _audit_log = logging.getLogger("audit")
        audit = AuditLog(sink=lambda ev: _audit_log.info(_json.dumps(ev)))
        demo_key, is_new = _bootstrap_demo_tenant(store)
        service = SecuredInferenceService(service, store, audit)
        backing = f"persistent SQLite ({args.key_db})" if args.key_db else "in-memory (ephemeral)"
        if is_new:
            # Printed to stdout (not the structured log) so it is easy to copy.
            print("\n  [--secured] demo API key (tenant 'demo', role 'member'):\n"
                  f"    {demo_key}\n"
                  "  Pass it as:  -H 'authorization: Bearer <key>'\n"
                  f"  Key store: {backing}.\n", flush=True)
        else:
            print(f"\n  [--secured] key store: {backing} — existing keys remain "
                  "valid (not reprinted). Mint/rotate via enterprise.keymgmt.\n",
                  flush=True)

    if args.metrics:
        # Outermost wrapper so it observes every outcome (incl. 401/403 before
        # the model runs). Purely observational — it changes no behaviour.
        from serve.metrics import MeteredInferenceService
        service = MeteredInferenceService(service)
        print("  [--metrics] aggregate metrics exposed at GET /metrics\n",
              flush=True)

    serve(service, host=args.host, port=args.port, max_connections=args.max_connections)


if __name__ == "__main__":
    main()
