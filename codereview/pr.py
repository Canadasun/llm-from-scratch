"""
Assemble review evidence into a pull-request artifact, and (only behind an
explicit action) open the PR (Stage 2, increment 3).

Two authority levels, kept separate on purpose:
  - prepare_pr(): runs the scan + the fix verification and builds the PR title,
    body (an evidence report) and a combined patch. It has NO side effects — no
    git, no network. Safe to run anywhere.
  - apply_fixes_to_branch() / push_and_create_pr(): the side-effectful git/gh
    operations. They refuse unless the fixes are VERIFIED and the working tree is
    clean, and they are only reached via the CLI's explicit --open-pr flag.

The PR only ever contains fixes whose verdict is "verified" (findings resolved AND
the project's tests still pass). Everything is for human review; nothing is
auto-merged. Authorization is a human decision, never the tool's.
"""

import difflib
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime

from codereview.fixes import apply_fixes, propose_fixes
from codereview.report import summarize
from codereview.scanner import scan_path
from codereview.verify import _apply_to_tree, propose_all, verify_fixes


@dataclass
class PRArtifact:
    verified: bool
    branch: str
    title: str
    body: str
    patch: str
    files_changed: list = field(default_factory=list)
    verdict: str = ""
    note: str = ""


def _combined_patch(repo_dir, proposals):
    by_file = {}
    for p in proposals:
        by_file.setdefault(p.file, []).append(p)
    chunks = []
    for path, props in sorted(by_file.items()):
        with open(path, encoding="utf-8") as f:
            src = f.read()
        patched = apply_fixes(src, props)
        rel = os.path.relpath(path, repo_dir)
        chunks.append("".join(difflib.unified_diff(
            src.splitlines(keepends=True), patched.splitlines(keepends=True),
            fromfile=f"a/{rel}", tofile=f"b/{rel}", n=3)))
    return "".join(chunks)


def build_pr_body(scan, verification, proposals, repo_dir, patch):
    s = summarize(scan["findings"])
    v = verification
    L = []
    L.append("## Automated secure code-review fixes")
    L.append("")
    L.append("Produced by `codereview/` (static AST analysis). **Human review "
             "required — do not auto-merge.** Authorization is a human decision.")
    L.append("")
    L.append("### Verification")
    if v.before is not None and v.after is not None:
        L.append(f"- Tests before: **{'PASS' if v.before.passed else 'FAIL'}** "
                 f"(rc={v.before.returncode}, {v.before.duration_s}s)")
        L.append(f"- Tests after:  **{'PASS' if v.after.passed else 'FAIL'}** "
                 f"(rc={v.after.returncode}, {v.after.duration_s}s)")
    L.append(f"- Findings: {v.findings_before} → {v.findings_after} "
             f"(resolved {len(v.resolved)}, introduced {len(v.introduced)})")
    L.append(f"- Verdict: **{v.verdict}** — {v.note}")
    L.append("")

    if proposals:
        L.append("### Applied fixes (verified)")
        L.append("| rule | CWE | location | change |")
        L.append("|---|---|---|---|")
        for p in proposals:
            rel = os.path.relpath(p.file, repo_dir)
            L.append(f"| {p.rule_id} | {p.cwe} | `{rel}:{p.line}` | {p.description} |")
        L.append("")
        # surface the caveats so the reviewer sees behaviour-change warnings
        caveats = sorted({p.caveat for p in proposals if p.caveat})
        if caveats:
            L.append("Caveats to check:")
            for c in caveats:
                L.append(f"- {c}")
            L.append("")
        L.append("<details><summary>Combined diff</summary>")
        L.append("")
        L.append("```diff")
        L.append(patch.rstrip("\n"))
        L.append("```")
        L.append("</details>")
        L.append("")

    L.append("### All findings (including ones NOT changed by this PR)")
    if scan["findings"]:
        L.append("| severity | conf | rule | CWE | location |")
        L.append("|---|---|---|---|---|")
        for f in scan["findings"]:
            rel = os.path.relpath(f.file, repo_dir)
            L.append(f"| {f.severity} | {f.confidence} | {f.rule_id} | {f.cwe} | "
                     f"`{rel}:{f.line}` |")
    else:
        L.append("None from the current rule set.")
    L.append("")
    L.append("Issue types that need manual restructuring (shell/SQL injection, "
             "`eval`/`exec`, hardcoded secrets) are reported but **not** changed here.")
    L.append("")
    L.append("### How this was produced & limitations")
    L.append("- Findings come from deterministic AST rules, not model output; each "
             "cites a **CWE**, not a CVE.")
    L.append("- Each fix was verified: apply to a copy → finding removed → the "
             "project's tests still pass.")
    L.append("- Static analysis has no taint tracking; **absence of findings is "
             "not proof of security**. Review every change.")
    return "\n".join(L)


def prepare_pr(repo_dir, test_cmd, branch=None, timeout=180):
    """Run scan + verification and build a PRArtifact. No git, no network."""
    scan = scan_path(repo_dir)
    verification = verify_fixes(repo_dir, test_cmd, timeout=timeout)
    proposals = propose_all(repo_dir)
    verified = verification.verdict == "verified"
    files = sorted({os.path.relpath(p.file, repo_dir) for p in proposals})
    patch = _combined_patch(repo_dir, proposals) if proposals else ""
    branch = branch or f"security/codereview-fixes-{datetime.now():%Y%m%d-%H%M%S}"
    if verified and proposals:
        title = f"Security: {len(proposals)} verified code-review fix(es)"
    else:
        title = (f"Security: {len(scan['findings'])} code-review finding(s) "
                 f"for review")
    body = build_pr_body(scan, verification, proposals, repo_dir, patch)
    return PRArtifact(verified=verified, branch=branch, title=title, body=body,
                      patch=patch, files_changed=files,
                      verdict=verification.verdict, note=verification.note)


def write_artifact(artifact, out_dir="."):
    """Write the PR body and the patch to files (no git). Returns the paths."""
    body_path = os.path.join(out_dir, "CODEREVIEW_PR.md")
    patch_path = os.path.join(out_dir, "codereview_fixes.patch")
    with open(body_path, "w", encoding="utf-8") as f:
        f.write(artifact.body + "\n")
    with open(patch_path, "w", encoding="utf-8") as f:
        f.write(artifact.patch)
    return body_path, patch_path


# ----------------- side-effectful git/gh operations (explicit opt-in) -----------

def _git(args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def apply_fixes_to_branch(repo_dir, artifact):
    """Create artifact.branch, apply the verified fixes, and commit them. Refuses
    unless the fixes are verified and the working tree is clean. Local only (no
    push). Returns (base_branch, new_branch)."""
    if not artifact.verified:
        raise RuntimeError("refusing: fixes are not verified")
    st = _git(["status", "--porcelain"], repo_dir)
    if st.returncode != 0:
        raise RuntimeError(f"not a git repo or git error: {st.stderr.strip()}")
    if st.stdout.strip():
        raise RuntimeError("refusing: working tree is not clean "
                           "(commit or stash first)")
    base = _git(["rev-parse", "--abbrev-ref", "HEAD"], repo_dir).stdout.strip()
    cb = _git(["checkout", "-b", artifact.branch], repo_dir)
    if cb.returncode != 0:
        raise RuntimeError(f"could not create branch: {cb.stderr.strip()}")
    _apply_to_tree(propose_all(repo_dir))
    _git(["add", "-A"], repo_dir)
    msg = (artifact.title + "\n\nAutomated, verified secure code-review fixes "
           "(codereview/). Human review required; do not auto-merge.")
    cm = _git(["commit", "-m", msg], repo_dir)
    if cm.returncode != 0:
        raise RuntimeError(f"commit failed: {cm.stderr.strip()}")
    return base, artifact.branch


def push_and_create_pr(repo_dir, branch, title, body, remote="origin"):  # pragma: no cover
    """Push the branch and open a PR via gh. Network + GitHub auth required.
    Separated so the local branch step can be used/tested without it."""
    push = _git(["push", "-u", remote, branch], repo_dir)
    if push.returncode != 0:
        raise RuntimeError(f"push failed: {push.stderr.strip()}")
    pr = subprocess.run(["gh", "pr", "create", "--title", title, "--body", body],
                        cwd=repo_dir, capture_output=True, text=True)
    if pr.returncode != 0:
        raise RuntimeError(f"gh pr create failed: {pr.stderr.strip()}")
    return pr.stdout.strip()
