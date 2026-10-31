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
    ap.add_argument("path", nargs="?", help="file or directory to scan")
    ap.add_argument("--self-metrics", action="store_true",
                    help="report precision/recall of the detector against the "
                         "built-in labelled corpus (no scan target needed)")
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
                    help="per-run timeout (seconds) for --verify-fixes/--prepare-pr")
    ap.add_argument("--prepare-pr", action="store_true",
                    help="build a PR artifact (title, evidence body, combined patch) "
                         "from VERIFIED fixes — writes files, NO git/network")
    ap.add_argument("--open-pr", action="store_true",
                    help="EXPLICIT: create a branch, commit the verified fixes "
                         "(refuses if tree is dirty or fixes unverified). Add --push "
                         "to also push and open a PR via gh.")
    ap.add_argument("--push", action="store_true",
                    help="with --open-pr: push the branch and run gh pr create")
    ap.add_argument("--out-dir", default=".",
                    help="where --prepare-pr writes its artifact files")
    args = ap.parse_args(argv)

    if args.self_metrics:
        from codereview.metrics import compute_metrics, format_metrics
        print(format_metrics(compute_metrics()))
        return 0

    if not args.path:
        ap.error("path is required (or use --self-metrics)")

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

    if args.prepare_pr or args.open_pr:
        if not args.test_cmd:
            print("\nerror: --prepare-pr/--open-pr require --test-cmd "
                  "(e.g. --test-cmd 'python -m pytest -q')", file=sys.stderr)
            return 2
        from codereview.pr import (apply_fixes_to_branch, prepare_pr,
                                    push_and_create_pr, write_artifact)
        print(f"\n[!] Preparing PR: this EXECUTES {args.test_cmd!r} on a copy to "
              f"verify fixes (before/after).")
        artifact = prepare_pr(args.path, args.test_cmd, timeout=args.timeout)
        print(f"\nVerdict: {artifact.verdict} — {artifact.note}")
        print(f"Branch:  {artifact.branch}")
        print(f"Title:   {artifact.title}")
        print(f"Files:   {', '.join(artifact.files_changed) or '(none)'}")

        if args.prepare_pr:
            body_path, patch_path = write_artifact(artifact, args.out_dir)
            print(f"\nWrote PR body -> {body_path}\nWrote patch    -> {patch_path}")
            print("No git actions taken. Review, then run with --open-pr to branch.")

        if args.open_pr:
            if not artifact.verified:
                print(f"\nRefusing --open-pr: fixes are not verified "
                      f"(verdict={artifact.verdict}).", file=sys.stderr)
                return 1
            base, branch = apply_fixes_to_branch(args.path, artifact)
            print(f"\nCommitted verified fixes on branch '{branch}' (from '{base}').")
            if args.push:
                url = push_and_create_pr(args.path, branch, artifact.title,
                                         artifact.body)
                print(f"Opened PR: {url}")
            else:
                print("Branch is local only. Push it and open a PR when ready "
                      "(or re-run with --push).")

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
