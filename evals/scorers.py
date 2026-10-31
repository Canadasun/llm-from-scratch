"""
Scorers for free-generation and code tasks.

Text scorers are simple and transparent (exact / contains / regex). The code
scorer runs the model's output against unit tests in a SEPARATE subprocess —
never imported or exec'd in this process — with a throwaway working directory, a
wall-clock timeout, CPU/memory/file-size rlimits, a scrubbed environment, and a
best-effort network namespace when the host supports it.

Honest isolation scope: this is resource-limited, not a security sandbox. The
subprocess still runs with the evaluator's OS user, so generated code can READ
files the user can read. Run this only on model output you are willing to
execute, in a disposable environment (local dev / throwaway CI). Strong
isolation (no host filesystem, no escalation) requires an OS sandbox
(a container, bubblewrap, etc.) and is out of scope for this stdlib harness.

Scoring integrity: a pass requires BOTH a clean exit AND a per-run random
sentinel that is printed only after the tests actually run and all succeed — so
generated code cannot score a pass by exiting 0 early (e.g. os._exit(0) at
import) before the assertions execute.
"""

import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile

# resource is POSIX-only; degrade gracefully elsewhere (limits just won't apply).
try:
    import resource
except ImportError:                       # pragma: no cover - non-POSIX
    resource = None


def exact_match(output, expected):
    return output.strip() == expected.strip()


def contains(output, expected):
    return expected.strip().lower() in output.lower()


def regex_match(output, pattern):
    return re.search(pattern, output) is not None


def contains_any(output, expecteds):
    low = output.lower()
    return any(e.strip().lower() in low for e in expecteds)


def extract_code(text):
    """Pull the first fenced code block out of model output, or return the raw
    text if there is no fence (the tiny model rarely produces clean fences)."""
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return (m.group(1) if m else text).strip()


def _limit_resources(cpu_seconds=5, mem_mb=512, fsize_mb=8):
    """preexec hook: cap CPU time, address space and file size for the child."""
    if resource is None:                  # pragma: no cover
        return
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    nbytes = mem_mb * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_AS, (nbytes, nbytes))
    except (ValueError, OSError):         # pragma: no cover - some platforms
        pass
    fbytes = fsize_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_FSIZE, (fbytes, fbytes))


_BWRAP_OK = None       # cached probe result for the bubblewrap sandbox


def _bwrap_prefix(workdir):
    """Strong isolation when bubblewrap (`bwrap`) is installed: a fresh namespace
    with NO host filesystem except a read-only runtime and the writable workdir,
    and no network. Returns an argv prefix, or None if bwrap is absent or a probe
    shows the sandbox can't run Python here (then we fall back). The probe is
    cached so it runs at most once."""
    global _BWRAP_OK
    if not shutil.which("bwrap"):
        return None
    base = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session",
            "--tmpfs", "/", "--ro-bind", "/usr", "/usr",
            "--ro-bind-try", "/lib", "/lib", "--ro-bind-try", "/lib64", "/lib64",
            "--ro-bind-try", "/bin", "/bin", "--ro-bind-try", "/sbin", "/sbin",
            "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
    if _BWRAP_OK is None:
        try:
            probe = subprocess.run(base + [sys.executable, "-I", "-c",
                                           "import unittest"],
                                   capture_output=True, timeout=10)
            _BWRAP_OK = probe.returncode == 0
        except Exception:                      # pragma: no cover - defensive
            _BWRAP_OK = False
    if not _BWRAP_OK:
        return None
    return base + ["--bind", workdir, workdir, "--chdir", workdir, "--"]


def _net_isolation_prefix():
    """Fallback when bwrap is absent: run in a rootless network namespace if the
    host allows it (blocks network exfiltration). NOT filesystem isolation."""
    if not shutil.which("unshare"):
        return []
    try:
        probe = subprocess.run(["unshare", "-rn", "true"],
                               capture_output=True, timeout=5)
        if probe.returncode == 0:
            return ["unshare", "-rn", "--"]
    except Exception:                          # pragma: no cover - defensive
        pass
    return []


# Runner executed in the child. The sentinel is read from an inherited pipe fd
# (SENT_FD) BEFORE any untrusted code is imported, and written back on SENT_FD's
# partner (RES_FD) only after every test actually ran and passed. The sentinel
# never appears in argv or the environment, so generated code can't read it there
# and forge a pass (and an early os._exit(0) never reaches the write).
_RUNNER = (
    "import os, sys, unittest\n"
    "sent = os.read(int(os.environ['SENT_FD']), 128)\n"
    "os.close(int(os.environ['SENT_FD']))\n"
    "res = unittest.TextTestRunner(verbosity=1).run(\n"
    "    unittest.TestLoader().loadTestsFromName('test_solution'))\n"
    "ok = res.wasSuccessful() and res.testsRun > 0\n"
    "if ok:\n"
    "    os.write(int(os.environ['RES_FD']), sent)\n"
    "os.close(int(os.environ['RES_FD']))\n"
    "sys.exit(0 if ok else 1)\n"
)


def run_python_unittest(solution_code, test_code, timeout=10):
    """Write `solution_code` as solution.py and `test_code` as a test that
    imports it, then run the tests in a separate, resource-limited subprocess
    (bubblewrap-isolated when available, else a best-effort network namespace).
    Returns (passed: bool, detail: str). A pass requires a clean exit AND an
    unforgeable per-run sentinel delivered over a pipe, so neither an early
    os._exit(0) nor reading argv/env can fake success. See the module docstring
    for the (honest, limited) isolation scope.
    """
    code = extract_code(solution_code)
    sentinel = os.urandom(24)
    with tempfile.TemporaryDirectory(prefix="llm_eval_") as d:
        with open(os.path.join(d, "solution.py"), "w") as f:
            f.write(code)
        with open(os.path.join(d, "test_solution.py"), "w") as f:
            f.write(test_code)
        with open(os.path.join(d, "_runner.py"), "w") as f:
            f.write(_RUNNER)

        sent_r, sent_w = os.pipe()
        res_r, res_w = os.pipe()
        os.write(sent_w, sentinel)             # fits the pipe buffer
        os.close(sent_w)
        # Minimal environment; the child gets the temp dir on its import path and
        # the two pipe fd NUMBERS (the sentinel value travels through the pipe).
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
               "PYTHONPATH": d, "PYTHONDONTWRITEBYTECODE": "1",
               "SENT_FD": str(sent_r), "RES_FD": str(res_w)}
        prefix = _bwrap_prefix(d) or _net_isolation_prefix()
        argv = prefix + [sys.executable, "-s", "-E", "_runner.py"]
        try:
            proc = subprocess.run(
                argv, cwd=d, env=env, capture_output=True, text=True,
                timeout=timeout, pass_fds=(sent_r, res_w),
                preexec_fn=_limit_resources if resource else None)
        except subprocess.TimeoutExpired:
            return False, "timeout"
        except Exception as e:                 # pragma: no cover - defensive
            return False, f"runner error: {e}"
        finally:
            for fd in (sent_r, res_w):
                try:
                    os.close(fd)
                except OSError:
                    pass
        try:
            received = os.read(res_r, 128)
        except OSError:                         # pragma: no cover - defensive
            received = b""
        finally:
            os.close(res_r)
        # Integrity gate: a clean exit is NOT enough — only the sentinel coming
        # back proves the tests actually ran to completion and all passed.
        ok = proc.returncode == 0 and received == sentinel
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        if proc.returncode == 0 and not ok:
            return False, "exited 0 without completing tests (no sentinel)"
        return ok, (detail[-1] if detail else ("ok" if ok else "failed"))
