"""
Scorers for free-generation and code tasks.

Text scorers are simple and transparent (exact / contains / regex). The code
scorer actually runs the model's output against unit tests, but only inside a
locked-down subprocess: isolated interpreter (-I), a throwaway working
directory, a wall-clock timeout, and CPU/memory/file-size rlimits. Generated
code is untrusted input and is treated as such — never imported or exec'd in
this process.
"""

import os
import re
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


def run_python_unittest(solution_code, test_code, timeout=10):
    """Write `solution_code` as solution.py and `test_code` as a test that
    imports it, then run the test in a sandboxed subprocess. Returns
    (passed: bool, detail: str).
    """
    code = extract_code(solution_code)
    with tempfile.TemporaryDirectory(prefix="llm_eval_") as d:
        with open(os.path.join(d, "solution.py"), "w") as f:
            f.write(code)
        with open(os.path.join(d, "test_solution.py"), "w") as f:
            f.write(test_code)
        # Minimal environment: the child inherits no parent-process secrets, but
        # still gets the temp dir on its import path so the test can import the
        # generated solution. Isolation comes from the throwaway cwd, the rlimits
        # and the timeout, not from -I (which would also drop the cwd import).
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
               "PYTHONPATH": d, "PYTHONDONTWRITEBYTECODE": "1"}
        try:
            proc = subprocess.run(
                [sys.executable, "-s", "-E", "-m", "unittest", "-q", "test_solution"],
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
        ok = proc.returncode == 0
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        return ok, (detail[-1] if detail else ("ok" if ok else "failed"))
