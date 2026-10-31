"""
Run the inference server.

    python -m serve                       # 127.0.0.1:8000, checkpoints/ckpt.pt
    python -m serve --port 8080 --ckpt path/to/ckpt.pt
    python -m serve --host 0.0.0.0        # expose (be deliberate about this)
    python -m serve --no-safety           # research mode; logged

Then:
    curl localhost:8000/healthz
    curl -s localhost:8000/generate -H 'content-type: application/json' \
         -d '{"prompt":"Zero trust architecture ","max_new_tokens":80,"seed":1}'
"""

import argparse
import logging

from serve.engine import DEFAULT_CKPT, InferenceEngine
from serve.http_server import serve
from serve.ratelimit import RateLimiter
from serve.service import InferenceService


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
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(message)s")

    if args.no_safety:
        logging.getLogger("serve").warning("use-safety policy DISABLED (--no-safety)")

    engine = InferenceEngine.from_checkpoint(args.ckpt, device=args.device)
    limiter = RateLimiter(capacity=args.rate_capacity, refill_per_sec=args.rate_per_sec)
    service = InferenceService(engine, rate_limiter=limiter, safety=not args.no_safety)
    serve(service, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
