"""Text scorers and an explicit TRUSTED-code test runner.

Untrusted generated code is refused by this legacy unittest runner. The JSON
function evaluator in evals.code_executor provides independent comparisons with
mandatory isolation. A unittest runner sharing a
Python interpreter with a solution cannot attest its own integrity: the solution
can inspect runner globals, monkeypatch unittest, or alter the tests. A separate
sandboxed execution service with an independent verdict is required for arbitrary
unittest tasks, which this helper does not support.

`trusted=True` is only for reviewed fixtures or code the caller already trusts
to execute as its OS user. It is never enabled by model output or task JSON.
Bubblewrap is used when available as defense in depth, not as proof of scoring
integrity. Timeout cleanup kills the process group and always closes pipes.
"""

import os
import re
import signal
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


# Words that invert the meaning of a following expected phrase, so that merely
# containing the phrase is NOT evidence of the right answer. "Do NOT use
# parameterized queries" contains "parameterized" but recommends the opposite.
_NEGATORS = ("not", "never", "don't", "dont", "do not", "avoid", "without",
             "no ", "shouldn't", "should not", "instead of", "rather than",
             "except", "unless")
# How far back (characters) to look for a negator before an expected phrase.
_NEGATION_WINDOW = 40


def _is_negated(low_output, phrase_start):
    """True if a negation word appears shortly before phrase_start, in the same
    sentence (we stop at sentence boundaries so a prior sentence's 'not' does
    not spuriously negate this one)."""
    window = low_output[max(0, phrase_start - _NEGATION_WINDOW):phrase_start]
    window = re.split(r"[.!?;\n]", window)[-1]  # current sentence fragment only
    return any(re.search(r"(?:^|\b)" + re.escape(n.strip()) + r"\b", window)
               for n in _NEGATORS)


def contains(output, expected):
    """Substring match that FAILS when the expected phrase is negated nearby, so
    an answer recommending the opposite of the expected concept is not scored
    correct (audit: negated security advice passed a plain substring check)."""
    low, needle = output.lower(), expected.strip().lower()
    if not needle:
        return False
    positions = [m.start() for m in re.finditer(re.escape(needle), low)]
    if not positions:
        return False
    # Reject contradictory mentions too, rather than letting a later keyword
    # launder an unsafe recommendation. This remains a heuristic, not a semantic
    # correctness verdict; release qualification rejects heuristic scoring.
    for start in positions:
        suffix = re.split(r"[.!?;\n]", low[start + len(needle):])[0][:80]
        if _is_negated(low, start) or re.search(
                r"\b(unnecessary|unsafe|useless|insecure|not needed|not required|"
                r"should be avoided|must be avoided|should not|must not)\b", suffix):
            return False
    return True


def regex_match(output, pattern):
    return re.search(pattern, output) is not None


def contains_any(output, expecteds):
    return any(contains(output, e) for e in expecteds)


def contains_all(output, expecteds):
    """Every expected phrase must be present and non-negated."""
    return all(contains(output, e) for e in expecteds)


def lacks_forbidden(output, forbidden):
    """True when NONE of the forbidden phrases appear (case-insensitive). Used to
    reject unsafe recommendations even when the expected phrase is also present."""
    low = output.lower()
    return not any(f.strip().lower() in low for f in (forbidden or []))


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


# Completion marker for reviewed fixtures only. This runner and the solution
# share an interpreter, so the marker cannot attest adversarial test execution.
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


def run_python_unittest(solution_code, test_code, timeout=10, *, trusted=False):
    """Run reviewed code only; untrusted code requires an independent evaluator."""
    if trusted is not True:
        return False, "execution_disabled: untrusted code requires an independent evaluator"
    if os.name != "posix":
        return False, "execution_disabled: POSIX process isolation is required"
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
        received = b""
        try:
            # Files keep captured output out of parent memory; the child file-size
            # rlimit bounds each stream. Read just the diagnostic tail afterwards.
            with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
                proc = subprocess.Popen(
                    argv, cwd=d, env=env, stdout=stdout, stderr=stderr,
                    start_new_session=True, pass_fds=(sent_r, res_w),
                    preexec_fn=_limit_resources if resource else None)
                try:
                    proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    return False, "timeout"
                finally:
                    # Includes descendants that outlive a normally exiting parent.
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait()
                os.set_blocking(res_r, False)
                try:
                    received = os.read(res_r, 128)
                except BlockingIOError:
                    received = b""
                for stream in (stdout, stderr):
                    stream.seek(0, os.SEEK_END)
                    stream.seek(max(0, stream.tell() - 8192))
                out = stdout.read().decode("utf-8", errors="replace")
                err = stderr.read().decode("utf-8", errors="replace")
        except Exception as e:
            return False, f"runner error: {e}"
        finally:
            for fd in (sent_r, res_w, res_r):
                os.close(fd)
        # This completion marker catches accidents in trusted fixtures only;
        # it is deliberately NOT described as an unforgeable security boundary.
        ok = proc.returncode == 0 and received == sentinel
        detail = (err or out or "").strip().splitlines()
        if proc.returncode == 0 and not ok:
            return False, "exited 0 without completing tests (no sentinel)"
        return ok, (detail[-1] if detail else ("ok" if ok else "failed"))
