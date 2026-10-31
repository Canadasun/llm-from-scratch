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


def _net_isolation_prefix():
    """Return an argv prefix that runs the child in a new (rootless) network
    namespace when the host supports it, so generated code can't exfiltrate over
    the network. Returns [] if unavailable, so the scorer still works."""
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


# Runner executed in the child: it RUNS the tests and only prints the sentinel
# (passed in as argv[1]) after every test has actually run and succeeded. This is
# what makes an early os._exit(0) unable to fake a pass.
_RUNNER = (
    "import sys, unittest\n"
    "suite = unittest.TestLoader().loadTestsFromName('test_solution')\n"
    "res = unittest.TextTestRunner(verbosity=1).run(suite)\n"
    "if res.wasSuccessful() and res.testsRun > 0:\n"
    "    sys.stdout.write(sys.argv[1]); sys.stdout.flush(); sys.exit(0)\n"
    "sys.exit(1)\n"
)


def run_python_unittest(solution_code, test_code, timeout=10):
    """Write `solution_code` as solution.py and `test_code` as a test that
    imports it, then run the tests in a separate, resource-limited subprocess.
    Returns (passed: bool, detail: str). A pass requires a clean exit AND the
    per-run sentinel, so early-exit tricks cannot fake success. See the module
    docstring for the (honest, limited) isolation scope.
    """
    code = extract_code(solution_code)
    sentinel = "PASS_" + secrets.token_hex(16)
    with tempfile.TemporaryDirectory(prefix="llm_eval_") as d:
        with open(os.path.join(d, "solution.py"), "w") as f:
            f.write(code)
        with open(os.path.join(d, "test_solution.py"), "w") as f:
            f.write(test_code)
        with open(os.path.join(d, "_runner.py"), "w") as f:
            f.write(_RUNNER)
        # Minimal environment: the child inherits no parent-process secrets, but
        # still gets the temp dir on its import path so the test can import the
        # generated solution.
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
               "PYTHONPATH": d, "PYTHONDONTWRITEBYTECODE": "1"}
        argv = (_net_isolation_prefix()
                + [sys.executable, "-s", "-E", "_runner.py", sentinel])
        try:
            proc = subprocess.run(
                argv,
                cwd=d,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                preexec_fn=_limit_resources if resource else None,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
        except Exception as e:                 # pragma: no cover - defensive
            return False, f"runner error: {e}"
        # Integrity gate: clean exit is NOT enough — the sentinel proves the
        # tests actually ran to completion and all passed.
        ok = proc.returncode == 0 and sentinel in (proc.stdout or "")
        out = (proc.stdout or "").replace(sentinel, "").strip()
        detail = (proc.stderr or out or "").strip().splitlines()
        if proc.returncode == 0 and not ok:
            return False, "exited 0 without completing tests (no sentinel)"
        return ok, (detail[-1] if detail else ("ok" if ok else "failed"))
