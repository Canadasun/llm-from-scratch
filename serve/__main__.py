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
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(message)s")

    log = logging.getLogger("serve")
    if args.no_safety:
        log.warning("use-safety policy DISABLED (--no-safety)")

    engine = InferenceEngine.from_checkpoint(args.ckpt, device=args.device)
    limiter = RateLimiter(capacity=args.rate_capacity, refill_per_sec=args.rate_per_sec)
    service = InferenceService(engine, rate_limiter=limiter, safety=not args.no_safety)

    if args.secured:
        from enterprise.audit import AuditLog
        store = _make_store(args.key_db)
        audit = AuditLog()
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

    serve(service, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
