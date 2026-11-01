"""
Verify proposed fixes against a project's own tests (Stage 2, increment 2).

A proposed patch is only trustworthy if, once applied, it (a) actually removes
the finding and (b) does not break the project's tests. This module proves that:

    copy the repo  ->  run tests (baseline)  ->  apply the reviewed fixes to the
    copy  ->  re-scan (findings gone?)  ->  run tests again (still pass?)  ->  verdict

Everything happens on a COPY in a temp dir; the original repo is never modified.

Repository tests are untrusted by default: bubblewrap isolates filesystem and
network, resource limits bound CPU/memory/processes/files, and output is disk
backed. Explicit trusted=True is only for code already trusted as the OS user.
This does not make repository-owned tests independent security evidence.
"""

import os
import shlex
import signal
import math
from pathlib import Path
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


def _limits(cpu_seconds, fsize_mb=8):
    def hook():                           # pragma: no cover - runs in child
        if resource is None:
            return
        resource.setrlimit(resource.RLIMIT_CPU, (math.ceil(cpu_seconds), math.ceil(cpu_seconds)))
        resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
        resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
        resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
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
    security_tests: list = field(default_factory=list)


def _isolation_prefix(workdir):
    """A bubblewrap sandbox for running an UNTRUSTED repo's tests: a fresh
    namespace with NO network (--unshare-all), no host home or credentials, a
    read-only system runtime, and the repo copy bind-mounted writable. Returns an
    argv prefix, or None if bwrap is unavailable. The sandbox uses the SYSTEM
    /usr/bin/python3 (not this venv), so an untrusted repo must carry its own
    dependencies / run on a prepared host — never the evaluator's interpreter."""
    bwrap = os.environ.get("LLM_EVAL_BWRAP") or shutil.which("bwrap")
    if not bwrap or not os.path.isfile("/usr/bin/python3"):
        return None
    return [bwrap, "--unshare-all", "--die-with-parent", "--new-session",
            "--cap-drop", "ALL", "--uid", "65534", "--gid", "65534",
            "--tmpfs", "/", "--ro-bind", "/usr", "/usr",
            "--ro-bind-try", "/lib", "/lib", "--ro-bind-try", "/lib64", "/lib64",
            "--ro-bind-try", "/bin", "/bin", "--ro-bind-try", "/sbin", "/sbin",
            "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
            "--bind", workdir, workdir, "--chdir", workdir,
            "--ro-bind", str(Path(__file__).with_name("job_worker.py")), "/job-runner.py",
            "--clearenv", "--setenv", "PATH", "/usr/bin:/bin",
            "--setenv", "HOME", workdir, "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
            "--"]


def run_command(test_cmd, cwd, timeout=180, *, trusted=False):
    """Run a test command in `cwd` under resource limits and a scrubbed env.
    'passed' means exit code 0.

    trusted=True (explicit opt-in): the current behaviour — run on the host with this
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
        if argv and (argv[0] in ("python", "python3") or argv[0] == sys.executable):
            argv[0] = "/usr/bin/python3"   # the sandbox's system interpreter
        argv = prefix + ["/usr/bin/python3", "-I", "-B", "/job-runner.py", str(timeout)] + argv
        env = {}                            # bwrap --clearenv / --setenv control the env
    if not argv or not math.isfinite(timeout) or timeout <= 0:
        return CmdResult(-1, False, 0.0, "invalid command or timeout")
    if not trusted and (resource is None or os.geteuid() == 0):
        return CmdResult(-1, False, 0.0, "isolation_unavailable: untrusted jobs require a non-root POSIX worker")
    try:
        # Disk-backed, file-size-limited output cannot grow parent memory without bound.
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            proc = subprocess.Popen(argv, cwd=cwd, env=env, stdout=stdout, stderr=stderr,
                                    start_new_session=True,
                                    preexec_fn=_limits(timeout) if resource and trusted else None)
            timed_out = False
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                # Reap descendants even if the command's parent exits normally.
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
            if timed_out:
                return CmdResult(-1, False, round(time.monotonic() - t0, 2), "timeout")
            tails = []
            for stream in (stdout, stderr):
                stream.seek(0, os.SEEK_END)
                stream.seek(max(0, stream.tell() - 8192))
                tails.append(stream.read().decode("utf-8", errors="replace"))
            tail = "\n".join("".join(tails).strip().splitlines()[-8:])
            return CmdResult(proc.returncode, proc.returncode == 0,
                             round(time.monotonic() - t0, 2), tail)
    except Exception as exc:
        return CmdResult(-1, False, round(time.monotonic() - t0, 2), f"runner error: {exc}")


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


def verify_fixes(repo_dir, test_cmd, timeout=180, *, trusted=False):
    """Copy repo_dir, propose+apply fixes on the copy, and verify tests + findings.
    Returns a VerificationResult. The original repo_dir is never modified.

    trusted=False runs the repo's tests inside a bubblewrap sandbox and fails
    closed (verdict 'error') if isolation is unavailable — use it for any repo
    you do not fully trust (audit: isolate every repository job)."""
    tmp = tempfile.mkdtemp(prefix="codereview_verify_")
    dst = os.path.join(tmp, "repo")
    try:
        shutil.copytree(repo_dir, dst, ignore=_IGNORE, symlinks=True)

        before_scan = scan_path(dst)
        proposals = propose_all(dst)
        if not proposals:
            return VerificationResult(
                verdict="no_fixes", findings_before=len(before_scan["findings"]),
                findings_after=len(before_scan["findings"]),
                note="No auto-fixable findings to verify.")

        witnesses = []
        if not trusted:
            from codereview.security_tests import reproduce
            for proposal in proposals:
                if proposal.rule_id != "PY-WEAK-HASH":
                    return VerificationResult(verdict="error", note=
                        "No independent behavioral security test for this rule; manual review required.")
                witness = reproduce(proposal.file, proposal.line)
                if witness["status"] != "failed" or witness.get("cases_passed") != 0 or witness.get("detail") != "case 1 output mismatch":
                    return VerificationResult(verdict="error", note=
                        "Security failure was not independently reproduced (sandbox or witness unavailable); refusing verified status.",
                        security_tests=[witness])
                witnesses.append({"file": os.path.relpath(proposal.file, dst),
                                  "line": proposal.line, "before": witness})
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

        for witness in witnesses:
            witness["after"] = reproduce(os.path.join(dst, witness["file"]), witness["line"])
        if any(w["after"]["status"] != "passed" for w in witnesses):
            return VerificationResult(verdict="error", before=before, after=after,
                security_tests=witnesses, note="Independent security test did not pass after patching.")

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
            files_changed=[os.path.relpath(c, dst) for c in changed], note=note,
            security_tests=witnesses)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def propose_all(root):
    """Collect fix proposals for every file under root (paths point into root)."""
    from codereview.scanner import collect_fixes
    return collect_fixes(root)


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
    if r.security_tests:
        lines.append(f"Independent security regression witnesses: {len(r.security_tests)}")
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
