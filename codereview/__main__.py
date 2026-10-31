"""
Secure code review CLI.

    python -m codereview path/to/repo            # scan + print report
    python -m codereview file.py --json out.json # also write machine-readable
    python -m codereview . --fail-on high        # exit 1 if any >= high (for CI)

By default it REPORTS and exits 0 (gating is an explicit choice, like the eval
suite). --fail-on turns it into a gate.
"""

import argparse
import json
import sys

from codereview.findings import SEVERITY_ORDER
from codereview.report import format_report, summarize
from codereview.scanner import scan_path


def main(argv=None):
    ap = argparse.ArgumentParser(description="static secure code review (Python)")
    ap.add_argument("path", help="file or directory to scan")
    ap.add_argument("--json", help="write findings as JSON here")
    ap.add_argument("--max-findings", type=int, default=100)
    ap.add_argument("--fail-on", choices=["critical", "high", "medium", "low", "none"],
                    default="none", help="exit 1 if any finding is at least this severe")
    args = ap.parse_args(argv)

    scan = scan_path(args.path)
    print(format_report(scan, max_findings=args.max_findings))

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"summary": summarize(scan["findings"]),
                       "findings": [x.to_dict() for x in scan["findings"]],
                       "files_scanned": scan["files_scanned"],
                       "skipped": scan["skipped"]}, f, indent=2)
        print(f"\nWrote JSON to {args.json}")

    if args.fail_on != "none":
        threshold = SEVERITY_ORDER[args.fail_on]
        worst = max((SEVERITY_ORDER[x.severity] for x in scan["findings"]),
                    default=-1)
        if worst >= threshold:
            print(f"\nFAIL: a finding at or above '{args.fail_on}' was reported.")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
