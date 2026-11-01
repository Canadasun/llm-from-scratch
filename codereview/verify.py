"""
Verify proposed fixes against a project's own tests (Stage 2, increment 2).

A proposed patch is only trustworthy if, once applied, it (a) actually removes
the finding and (b) does not break the project's tests. This module proves that:

    copy the repo  ->  run tests (baseline)  ->  apply the reviewed fixes to the
    copy  ->  re-scan (findings gone?)  ->  run tests again (still pass?)  ->  verdict

Everything happens on a COPY in a temp dir; the original repo is never modified.

Sandboxing — honest scope. The test command runs in a child process with:
  - a throwaway working directory (a copy of the repo),
  - a minimal environment (no parent-process secrets are inherited),
  - CPU-time and file-size rlimits and a wall-clock timeout.
It does NOT provide filesystem or network isolation — that needs OS containers/
namespaces, which is out of scope here. So run this against a TRUSTED repo (e.g.
this project's own tests) or inside a container you already trust. Running an
untrusted repo's tests is arbitrary code execution; do that only in a real
sandbox.
"""

import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field

try:
    import resource
except ImportError:                       # pragma: no cover - non-unix
    resource = None

from codereview.fixes import apply_fixes, propose_fixes
from codereview.scanner import scan_path

_IGNORE = shutil.ignore_patterns(
    ".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache", ".mypy_cache",
    "node_modules", "build", "dist", ".tox", "checkpoints", "rfc_cache",
    "nist_cache", "*.pt", "*.pyc", ".cache_*")


def _limits(cpu_seconds, fsize_mb=128):
    def hook():                           # pragma: no cover - runs in child
        if resource is None:
            return
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        fbytes = fsize_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_FSIZE, (fbytes, fbytes))
    return hook


@dataclass
class CmdResult:
    returncode: int
    passed: bool
    duration_s: float
    tail: str


@dataclass
class VerificationResult:
    verdict: str                 # verified | rejected_regression | baseline_failing
                                 # | no_fixes | no_change | error
    before: CmdResult = None
    after: CmdResult = None
    findings_before: int = 0
    findings_after: int = 0
    resolved: list = field(default_factory=list)   # (relpath, rule_id, line)
    introduced: list = field(default_factory=list)
    files_changed: list = field(default_factory=list)
    note: str = ""


def _isolation_prefix(workdir):
    """A bubblewrap sandbox for running an UNTRUSTED repo's tests: a fresh
    namespace with NO network (--unshare-all), no host home or credentials, a
    read-only system runtime, and the repo copy bind-mounted writable. Returns an
    argv prefix, or None if bwrap is unavailable. The sandbox uses the SYSTEM
    /usr/bin/python3 (not this venv), so an untrusted repo must carry its own
    dependencies / run on a prepared host — never the evaluator's interpreter."""
    bwrap = shutil.which("bwrap")
    if not bwrap or not os.path.isfile("/usr/bin/python3"):
        return None
    return [bwrap, "--unshare-all", "--die-with-parent", "--new-session",
            "--cap-drop", "ALL",
            "--tmpfs", "/", "--ro-bind", "/usr", "/usr",
            "--ro-bind-try", "/lib", "/lib", "--ro-bind-try", "/lib64", "/lib64",
            "--ro-bind-try", "/bin", "/bin", "--ro-bind-try", "/sbin", "/sbin",
            "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
            "--bind", workdir, workdir, "--chdir", workdir,
            "--clearenv", "--setenv", "PATH", "/usr/bin:/bin",
            "--setenv", "HOME", workdir, "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
            "--"]


def run_command(test_cmd, cwd, timeout=180, *, trusted=True):
    """Run a test command in `cwd` under resource limits and a scrubbed env.
    'passed' means exit code 0.

    trusted=True (default): the current behaviour — run on the host with this
    interpreter and a minimal env. Only for a repo you already trust (e.g. this
    project's own tests), since it is NOT filesystem/network isolated.

    trusted=False: run inside a bubblewrap sandbox (no network, no host home/
    credentials, read-only runtime, repo bind-mounted writable). If working
    isolation is unavailable this FAILS CLOSED (returns a non-passing result)
    rather than executing an untrusted repo with host access."""
    argv = shlex.split(test_cmd)
    t0 = time.monotonic()
    if trusted:
        if argv and argv[0] in ("python", "python3"):
            argv[0] = sys.executable       # use this interpreter / its packages
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
               "HOME": cwd, "PYTHONDONTWRITEBYTECODE": "1"}
    else:
        prefix = _isolation_prefix(cwd)
        if prefix is None:
            return CmdResult(-1, False, 0.0,
                             "isolation_unavailable: refusing to run an untrusted "
                             "repo without a working bubblewrap sandbox")
        if argv and argv[0] in ("python", "python3"):
            argv[0] = "/usr/bin/python3"   # the sandbox's system interpreter
        argv = prefix + argv
        env = {}                            # bwrap --clearenv / --setenv control the env
    try:
        proc = subprocess.run(
            argv, cwd=cwd, env=env, capture_output=True, text=True,
            timeout=timeout,
            preexec_fn=_limits(timeout) if resource else None)
    except subprocess.TimeoutExpired:
        return CmdResult(-1, False, round(time.monotonic() - t0, 2), "timeout")
    except Exception as e:                 # pragma: no cover - defensive
        return CmdResult(-1, False, round(time.monotonic() - t0, 2), f"runner error: {e}")
    out = (proc.stdout or "") + (proc.stderr or "")
    tail = "\n".join(out.strip().splitlines()[-8:])
    return CmdResult(proc.returncode, proc.returncode == 0,
                     round(time.monotonic() - t0, 2), tail)


def _finding_keys(scan, root):
    return {(os.path.relpath(f.file, root), f.rule_id, f.line)
            for f in scan["findings"]}


def _apply_to_tree(proposals):
    """Apply proposals (grouped by their .file) to files in place. Returns the
    list of files changed."""
    by_file = {}
    for p in proposals:
        by_file.setdefault(p.file, []).append(p)
    changed = []
    for path, props in by_file.items():
        with open(path, encoding="utf-8") as f:
            src = f.read()
        patched = apply_fixes(src, props)
        if patched != src:
            with open(path, "w", encoding="utf-8") as f:
                f.write(patched)
            changed.append(path)
    return changed


def verify_fixes(repo_dir, test_cmd, timeout=180, *, trusted=True):
    """Copy repo_dir, propose+apply fixes on the copy, and verify tests + findings.
    Returns a VerificationResult. The original repo_dir is never modified.

    trusted=False runs the repo's tests inside a bubblewrap sandbox and fails
    closed (verdict 'error') if isolation is unavailable — use it for any repo
    you do not fully trust (audit: isolate every repository job)."""
    tmp = tempfile.mkdtemp(prefix="codereview_verify_")
    dst = os.path.join(tmp, "repo")
    try:
        shutil.copytree(repo_dir, dst, ignore=_IGNORE)

        before_scan = scan_path(dst)
        proposals = propose_all(dst)
        if not proposals:
            return VerificationResult(
                verdict="no_fixes", findings_before=len(before_scan["findings"]),
                findings_after=len(before_scan["findings"]),
                note="No auto-fixable findings to verify.")

        before = run_command(test_cmd, dst, timeout, trusted=trusted)
        if not trusted and before.returncode == -1 and "isolation_unavailable" in before.tail:
            return VerificationResult(
                verdict="error", before=before,
                findings_before=len(before_scan["findings"]),
                findings_after=len(before_scan["findings"]),
                note="Refused: untrusted repo requires a working sandbox (no isolation available).")
        before_keys = _finding_keys(before_scan, dst)

        changed = _apply_to_tree(proposals)

        after_scan = scan_path(dst)
        after_keys = _finding_keys(after_scan, dst)
        after = run_command(test_cmd, dst, timeout, trusted=trusted)

        resolved = sorted(before_keys - after_keys)
        introduced = sorted(after_keys - before_keys)

        if not before.passed:
            verdict = "baseline_failing"
            note = ("Baseline tests did not pass, so a regression can't be "
                    "attributed to the fixes. Fix the baseline first.")
        elif not after.passed:
            verdict = "rejected_regression"
            note = "A proposed fix broke the tests — DO NOT apply."
        elif resolved and not introduced:
            verdict = "verified"
            note = "Fixes resolve findings and the tests still pass."
        else:
            verdict = "no_change"
            note = "Fixes applied but no net finding reduction."

        return VerificationResult(
            verdict=verdict, before=before, after=after,
            findings_before=len(before_keys), findings_after=len(after_keys),
            resolved=resolved, introduced=introduced,
            files_changed=[os.path.relpath(c, dst) for c in changed], note=note)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def propose_all(root):
    """Collect fix proposals for every file under root (paths point into root)."""
    from codereview.scanner import iter_python_files
    out = []
    for path in iter_python_files(root):
        try:
            with open(path, encoding="utf-8") as f:
                src = f.read()
        except OSError:
            continue
        out.extend(propose_fixes(src, filename=path))
    return out


def format_verification(r):
    lines = ["=" * 66, "  FIX VERIFICATION", "=" * 66,
             f"Verdict: {r.verdict.upper()} — {r.note}"]
    if r.before is not None:
        lines.append(f"Tests before: {'PASS' if r.before.passed else 'FAIL'} "
                     f"(rc={r.before.returncode}, {r.before.duration_s}s)")
    if r.after is not None:
        lines.append(f"Tests after:  {'PASS' if r.after.passed else 'FAIL'} "
                     f"(rc={r.after.returncode}, {r.after.duration_s}s)")
    lines.append(f"Findings: {r.findings_before} -> {r.findings_after}   "
                 f"resolved: {len(r.resolved)}   introduced: {len(r.introduced)}")
    if r.files_changed:
        lines.append("Files changed by the fixes:")
        for c in r.files_changed:
            lines.append(f"  - {c}")
    if r.resolved:
        lines.append("Resolved findings:")
        for relpath, rule, line in r.resolved:
            lines.append(f"  - {rule} at {relpath}:{line}")
    if r.after is not None and not r.after.passed:
        lines.append("\nTest output (tail):")
        lines.append(r.after.tail)
    lines.append("=" * 66)
    return "\n".join(lines)
