"""
Run the Stage-4 pilot against a live (or local) endpoint.

    # against the live Azure endpoint (key from the --secured startup log)
    python -m pilot --url https://<app>.azurecontainerapps.io \
                    --key sk_... -n 100 -c 8

    # security probes only (no load), useful as a smoke check
    python -m pilot --url http://127.0.0.1:8000 --key sk_... --probe-only

    # machine-readable
    python -m pilot --url ... --key ... --json > pilot.json

The key is read from --key or the PILOT_API_KEY env var so it need not appear in
your shell history. Nothing is written anywhere; the report goes to stdout.
"""

import argparse
import json
import os
import sys

from pilot.loadtest import (LoadConfig, format_report, run_pilot,
                            security_probes)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stage-4 pilot: reliability / "
                                 "security / latency / cost of a live endpoint")
    ap.add_argument("--url", required=True, help="base URL of the endpoint")
    ap.add_argument("--key", default=os.environ.get("PILOT_API_KEY", ""),
                    help="API key (Bearer); or set PILOT_API_KEY")
    ap.add_argument("-n", "--requests", type=int, default=50)
    ap.add_argument("-c", "--concurrency", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--prompt", default="zero trust architecture ")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--probe-only", action="store_true",
                    help="run only the security probes (no load)")
    ap.add_argument("--json", action="store_true",
                    help="emit the raw report as JSON instead of text")
    args = ap.parse_args(argv)

    if args.probe_only:
        probes = security_probes(args.url, args.key, args.timeout)
        report = {"endpoint": args.url, "security": probes,
                  "security_all_pass": all(p["pass"] for p in probes)}
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            for p in probes:
                print(f"  [{'PASS' if p['pass'] else 'FAIL'}] {p['name']}  "
                      f"({p['detail']})")
            print("  -> " + ("ALL PASS" if report["security_all_pass"]
                             else "FAILURES PRESENT"))
        return 0 if report["security_all_pass"] else 1

    cfg = LoadConfig(base_url=args.url, api_key=args.key,
                     requests=args.requests, concurrency=args.concurrency,
                     prompt=args.prompt, max_new_tokens=args.max_new_tokens,
                     timeout=args.timeout)
    report = run_pilot(cfg)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_report(report))

    # Exit non-zero if the pilot found a reliability or security problem, so it
    # can gate a CI/CD step. "Healthy" = all security probes pass AND no errors.
    healthy = (report["security_all_pass"]
               and report["load"]["reliability"]["error_rate"] == 0.0)
    return 0 if healthy else 1


if __name__ == "__main__":
    sys.exit(main())
