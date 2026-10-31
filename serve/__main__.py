"""
Run the inference server.

    python -m serve                       # 127.0.0.1:8000, checkpoints/ckpt.pt
    python -m serve --port 8080 --ckpt path/to/ckpt.pt
    python -m serve --host 0.0.0.0        # expose (be deliberate about this)
    python -m serve --no-safety           # research mode; logged
    python -m serve --secured             # require an API key, tenant-scoped + audited

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


def _bootstrap_demo_tenant():
    """Build an in-memory store with one tenant/principal and a fresh API key.

    This is a DEMO identity so `--secured` is runnable end to end; it is NOT a
    production key store. Real deployments back the store with a database (with
    row-level security) and issue keys through an admin flow. The plaintext key
    is shown once here and only its hash is kept.
    """
    from enterprise.audit import AuditLog
    from enterprise.keys import generate_key
    from enterprise.models import ApiKey, Principal, Tenant
    from enterprise.store import Store

    store = Store()
    store.add_tenant(Tenant("demo", "Demo Tenant"))
    store.add_principal(Principal("demo-user", "demo", ("member",)))
    plaintext, key_hash = generate_key()
    store.add_api_key(ApiKey("demo-key", key_hash, "demo", "demo-user", ()))
    return store, AuditLog(), plaintext


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
        store, audit, demo_key = _bootstrap_demo_tenant()
        service = SecuredInferenceService(service, store, audit)
        # Printed to stdout (not the structured log) so it is easy to copy and
        # is not mistaken for an audit record. Demo only — see _bootstrap docs.
        print("\n  [--secured] demo API key (tenant 'demo', role 'member'):\n"
              f"    {demo_key}\n"
              "  Pass it as:  -H 'authorization: Bearer <key>'\n"
              "  This is an in-memory demo identity, not a production key store.\n",
              flush=True)

    serve(service, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
