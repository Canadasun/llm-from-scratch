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
from codereview.scanner import collect_fixes, scan_path


def main(argv=None):
    ap = argparse.ArgumentParser(description="static secure code review (Python)")
    ap.add_argument("path", help="file or directory to scan")
    ap.add_argument("--json", help="write findings as JSON here")
    ap.add_argument("--max-findings", type=int, default=100)
    ap.add_argument("--fail-on", choices=["critical", "high", "medium", "low", "none"],
                    default="none", help="exit 1 if any finding is at least this severe")
    ap.add_argument("--fix", action="store_true",
                    help="also print proposed patch diffs for auto-fixable findings "
                         "(suggestions only — nothing is written)")
    ap.add_argument("--verify-fixes", action="store_true",
                    help="apply the proposed fixes to a COPY of the repo and run "
                         "--test-cmd before/after to verify them (EXECUTES tests)")
    ap.add_argument("--test-cmd", default=None,
                    help="test command for --verify-fixes, e.g. 'python -m pytest -q'")
    ap.add_argument("--timeout", type=int, default=180,
                    help="per-run timeout (seconds) for --verify-fixes")
    args = ap.parse_args(argv)

    scan = scan_path(args.path)
    print(format_report(scan, max_findings=args.max_findings))

    if args.fix:
        proposals = collect_fixes(args.path)
        print("\n" + "=" * 66)
        print("  PROPOSED FIXES — review before applying; NOTHING is written")
        print("=" * 66)
        if not proposals:
            print("No auto-fixable findings. The remaining issue types (shell/SQL")
            print("injection, eval/exec, hardcoded secrets) need manual restructuring.")
        for p in proposals:
            print(f"\n[{p.rule_id} {p.cwe}] {p.file}:{p.line}  "
                  f"(fix confidence: {p.confidence})")
            print(f"  {p.description}")
            print(f"  caveat: {p.caveat}")
            print(p.diff.rstrip("\n"))

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"summary": summarize(scan["findings"]),
                       "findings": [x.to_dict() for x in scan["findings"]],
                       "files_scanned": scan["files_scanned"],
                       "skipped": scan["skipped"]}, f, indent=2)
        print(f"\nWrote JSON to {args.json}")

    if args.verify_fixes:
        if not args.test_cmd:
            print("\nerror: --verify-fixes requires --test-cmd "
                  "(e.g. --test-cmd 'python -m pytest -q')", file=sys.stderr)
            return 2
        from codereview.verify import format_verification, verify_fixes
        print(f"\n[!] --verify-fixes will EXECUTE: {args.test_cmd!r} on a copy of "
              f"{args.path} (before and after applying fixes).")
        result = verify_fixes(args.path, args.test_cmd, timeout=args.timeout)
        print(format_verification(result))

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
