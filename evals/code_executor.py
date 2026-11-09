"""Independent JSON-function evaluator. Expected answers stay in the controller.

Linux bubblewrap isolation is mandatory: there is no host-execution fallback.
The worker can return values, never a trusted verdict. It receives one case's
arguments, not expected answers or the rest of the suite. This tests observable
function behavior; it does not prove that a solution is secure or generalizes.
"""

import json
import hashlib
import ctypes
import ctypes.util
import errno
import math
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from evals.code_worker import decode_value, encode_value

MAX_CODE_BYTES = 128 * 1024
MAX_JSON_BYTES = 64 * 1024
MAX_CASES = 100
MAX_BATCH = 100
MAX_SUITE_CASES = 1000
MAX_BATCH_BYTES = 8 * 1024 * 1024
WORKER = Path(__file__).with_name("code_worker.py")
CASE_MANIFEST = Path(__file__).with_name("data") / "code_cases.json"


def _seccomp_filter(stream):
    """Emit a portable syscall deny filter applied by bwrap after namespace setup.

    Function tasks do not need sockets, child processes or namespace changes.
    Refuse execution if libseccomp cannot build the required filter.
    """
    library = ctypes.util.find_library("seccomp")
    if not library:
        return False
    lib = ctypes.CDLL(library)
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_export_bpf.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    ctx = lib.seccomp_init(0x7FFF0000)  # SCMP_ACT_ALLOW
    if not ctx:
        return False
    try:
        for name in ("fork", "vfork", "clone", "clone3", "unshare", "setns",
                     "mount", "umount2", "pivot_root", "ptrace", "process_vm_readv",
                     "process_vm_writev", "open_by_handle_at", "bpf", "perf_event_open",
                     "keyctl", "add_key", "request_key", "socket", "socketpair",
                     "connect", "bind", "listen", "accept", "accept4", "socketcall",
                     "kill", "tkill", "tgkill"):
            syscall = lib.seccomp_syscall_resolve_name(name.encode())
            if syscall >= 0 and lib.seccomp_rule_add(ctx, 0x00050000 | errno.EPERM, syscall, 0) != 0:
                return False
        return lib.seccomp_export_bpf(ctx, stream.fileno()) == 0
    finally:
        lib.seccomp_release(ctx)


def task_spec(task):
    """Select reviewed cases; never turn arbitrary unittest source into code."""
    try:
        with CASE_MANIFEST.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
    except (OSError, ValueError):
        return None
    if not isinstance(manifest, dict) or manifest.get("schema") != "function-json-v1":
        return None
    if not isinstance(manifest.get("tasks"), dict):
        return None
    spec = manifest["tasks"].get(task.get("id"))
    digest = hashlib.sha256(json.dumps(
        {"prompt": task.get("prompt"), "test": task.get("test")}, sort_keys=True).encode()).hexdigest()
    return spec if isinstance(spec, dict) and spec.get("task_sha256") == digest else None


@dataclass(frozen=True)
class Execution:
    status: str
    value: object = None
    detail: str = ""


@dataclass(frozen=True)
class Verdict:
    status: str
    passed: bool
    cases_passed: int = 0
    cases_total: int = 0
    detail: str = ""


def _json_bytes(value, limit=MAX_JSON_BYTES):
    data = json.dumps(value, allow_nan=False, ensure_ascii=True).encode()
    if len(data) > limit:
        raise ValueError("JSON value exceeds size limit")
    return data


def _reject_constant(value):
    raise ValueError("non-finite JSON constant")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _decode_response(data):
    value = json.loads(data, parse_constant=_reject_constant, object_pairs_hook=_unique_object)
    if not isinstance(value, dict) or set(value) != {"value"}:
        raise ValueError("worker must return exactly one value field")
    _json_bytes(value)  # also rejects numeric overflow such as 1e999
    return value["value"]


def _equal(actual, expected):
    # Python's True == 1 must not let a boolean solution pass numeric tests.
    pending = [(actual, expected)]
    while pending:
        actual, expected = pending.pop()
        if isinstance(actual, bool) or isinstance(expected, bool):
            if type(actual) is not type(expected) or actual != expected:
                return False
        elif isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
            if actual != expected:
                return False
        elif type(actual) is not type(expected):
            return False
        elif isinstance(expected, list):
            if len(actual) != len(expected):
                return False
            pending.extend(zip(actual, expected))
        elif isinstance(expected, dict):
            if actual.keys() != expected.keys():
                return False
            pending.extend((actual[k], v) for k, v in expected.items())
        elif actual != expected:
            return False
    return True


def _decode_batch(data, count):
    payload = json.loads(data, parse_constant=_reject_constant, object_pairs_hook=_unique_object)
    if not isinstance(payload, dict) or set(payload) != {"values"} or not isinstance(payload["values"], list):
        raise ValueError("worker must return exactly one values list")
    if len(payload["values"]) != count:
        raise ValueError("worker returned the wrong number of values")
    out = []
    for item in payload["values"]:
        if not isinstance(item, dict) or len(item) != 1:
            raise ValueError("invalid batch item")
        if "ok" in item:
            out.append(("ok", decode_value(item["ok"])))
        elif "error" in item and isinstance(item["error"], str):
            out.append(("error", item["error"][:64]))
        else:
            raise ValueError("invalid batch item")
    return out


def values_equal(actual, expected, atol=0.0):
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(expected, float):
        if not isinstance(actual, (int, float)):
            return False
        if math.isnan(expected):
            return isinstance(actual, float) and math.isnan(actual)
        if math.isinf(expected):
            return actual == expected
        return actual == expected or abs(actual - expected) <= (atol or 1e-6) + 1e-7 * abs(expected)
    if isinstance(expected, int):
        if isinstance(actual, float) and not isinstance(expected, bool):
            return actual == expected
        return type(actual) is int and actual == expected
    if isinstance(expected, complex):
        return isinstance(actual, complex) and abs(actual - expected) <= (atol or 1e-6)
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, (list, tuple)):
        return len(actual) == len(expected) and all(values_equal(a, e, atol) for a, e in zip(actual, expected))
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(values_equal(actual[k], v, atol) for k, v in expected.items())
    return actual == expected


COMPARE_MODES = ("exact", "set", "sequence")


def _coerce(value, mode):
    if mode == "set" and isinstance(value, (list, tuple, set, frozenset)):
        try:
            return set(value)
        except TypeError:
            return value
    if mode == "sequence" and isinstance(value, (list, tuple)):
        return list(value)
    return value


def evaluate_cases(code, function, cases, *, atol=0.0, timeout=60, runner=None, batch=MAX_BATCH):
    """Run tagged-JSON cases in sandboxed batches; expected values never leave the controller."""
    try:
        if not isinstance(code, str) or len(code.encode("utf-8")) > MAX_CODE_BYTES:
            raise ValueError("solution exceeds size limit")
        if not isinstance(function, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", function):
            raise ValueError("invalid function name")
        if not isinstance(cases, list) or not 1 <= len(cases) <= MAX_SUITE_CASES:
            raise ValueError("one to 1000 cases are required")
        if not 1 <= batch <= MAX_BATCH:
            raise ValueError("invalid batch size")
        calls, expected, modes = [], [], []
        for case in cases:
            if (not isinstance(case, dict) or set(case) - {"args", "kwargs", "expected", "compare"}
                    or "expected" not in case or case.get("compare", "exact") not in COMPARE_MODES):
                raise ValueError("invalid case fields")
            if not isinstance(case.get("args", []), list) or not isinstance(case.get("kwargs", {}), dict):
                raise ValueError("case args/kwargs have invalid types")
            modes.append(case.get("compare", "exact"))
            calls.append({"args": case.get("args", []), "kwargs": case.get("kwargs", {})})
            expected.append(decode_value(case["expected"]))
    except (ValueError, TypeError, UnicodeError, RecursionError, OverflowError) as exc:
        return Verdict("error", False, detail=str(exc))
    runner = runner or BubblewrapRunner()
    deadline = time.monotonic() + timeout
    for start in range(0, len(calls), batch):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return Verdict("failed", False, start, len(cases), "suite timeout")
        chunk = calls[start:start + batch]
        try:
            result = runner.run_batch(code, function, chunk, min(remaining, 30))
        except (OSError, ValueError, subprocess.SubprocessError):
            return Verdict("error", False, start, len(cases), "execution infrastructure error")
        if result.status != "ok":
            status = "unavailable" if result.status == "unavailable" else "failed"
            return Verdict(status, False, start, len(cases), result.detail)
        for offset, (kind, value) in enumerate(result.value):
            index = start + offset
            if kind != "ok":
                return Verdict("failed", False, index, len(cases), f"case {index + 1} raised {value}")
            if not values_equal(_coerce(value, modes[index]), expected[index], atol):
                return Verdict("failed", False, index, len(cases), f"case {index + 1} output mismatch")
    return Verdict("passed", True, len(cases), len(cases), "all function cases passed")


class BubblewrapRunner:
    """One fresh namespace per call. Runtime selection is deployment configuration.

    Only /usr and dynamic-library paths are exposed read-only. The job contains
    solution.py, code_worker.py, and arguments; no host home, tests, expected
    values, credentials, or other jobs are mounted. No writable scratch mount is
    provided. Kernel/runtime vulnerabilities remain outside this boundary.
    """

    def __init__(self, executable=None):
        self.executable = executable or os.environ.get("LLM_EVAL_BWRAP") or shutil.which("bwrap")
        self.python = "/usr/bin/python3"

    def _command(self, directory, seccomp_fd):
        if not self.executable or not os.path.isfile(self.python):
            return None
        return [self.executable, "--unshare-all", "--die-with-parent", "--new-session",
                "--cap-drop", "ALL", "--uid", "65534", "--gid", "65534",
                "--tmpfs", "/", "--ro-bind", "/usr", "/usr",
                "--ro-bind-try", "/lib", "/lib", "--ro-bind-try", "/lib64", "/lib64",
                "--dir", "/dev", "--ro-bind", "/dev/null", "/dev/null",
                "--ro-bind", directory, "/job", "--chdir", "/job",
                "--remount-ro", "/", "--clearenv", "--setenv", "LANG", "C.UTF-8",
                "--seccomp", str(seccomp_fd),
                "--", self.python, "-I", "-B", "/job/code_worker.py"]

    def run_case(self, code, function, args, kwargs, timeout):
        result = self._run(code, {"function": function, "args": args, "kwargs": kwargs}, timeout, MAX_JSON_BYTES)
        if result.status != "ok":
            return result
        try:
            return Execution("ok", value=_decode_response(result.value))
        except (ValueError, UnicodeError, RecursionError):
            return Execution("failed", detail="invalid worker response")

    def run_batch(self, code, function, calls, timeout):
        result = self._run(code, {"function": function, "calls": calls}, timeout, MAX_BATCH_BYTES)
        if result.status != "ok":
            return result
        try:
            return Execution("ok", value=_decode_batch(result.value, len(calls)))
        except (ValueError, TypeError, UnicodeError, RecursionError):
            return Execution("failed", detail="invalid worker response")

    def _run(self, code, request, timeout, max_output):
        if os.name != "posix" or not self.executable:
            return Execution("unavailable", detail="bubblewrap sandbox is required")
        with tempfile.TemporaryDirectory(prefix="llm_function_") as directory:
            root = Path(directory)
            (root / "solution.py").write_text(code, encoding="utf-8")
            shutil.copyfile(WORKER, root / "code_worker.py")
            (root / "request.json").write_bytes(_json_bytes(request, max_output))
            with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr, tempfile.TemporaryFile() as seccomp:
                try:
                    if not _seccomp_filter(seccomp):
                        return Execution("unavailable", detail="libseccomp filter is required")
                except (OSError, AttributeError):
                    return Execution("unavailable", detail="libseccomp filter could not initialize")
                seccomp.seek(0)
                command = self._command(directory, seccomp.fileno())
                if command is None:
                    return Execution("unavailable", detail="sandbox Python runtime is unavailable")
                try:
                    proc = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                            stdout=stdout, stderr=stderr, env={},
                                            close_fds=True, pass_fds=(seccomp.fileno(),),
                                            start_new_session=True)
                except OSError:
                    return Execution("unavailable", detail="sandbox could not start")
                try:
                    proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    return Execution("failed", detail="worker timeout")
                finally:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait()
                stdout.seek(0)
                data = stdout.read(max_output + 1)
                stderr.seek(0)
                diagnostic = stderr.read(2048)
                if proc.returncode:
                    # No generated exception text is reflected into the verdict.
                    if diagnostic.startswith(b"bwrap:"):
                        return Execution("unavailable", detail="required sandbox isolation could not start")
                    return Execution("failed", detail="worker exited unsuccessfully")
                if len(data) > max_output:
                    return Execution("failed", detail="worker output exceeds limit")
                return Execution("ok", value=data)


def evaluate_function(code, function, cases, *, timeout=10, runner=None):
    """Compare isolated function outputs against controller-owned JSON cases.

    `runner` is a trusted dependency-injection seam, never a task JSON option.
    Timeout is a total case-suite budget. Cases are validated before execution.
    """
    try:
        if not isinstance(code, str) or len(code.encode("utf-8")) > MAX_CODE_BYTES:
            raise ValueError("solution exceeds size limit")
        if not isinstance(function, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", function):
            raise ValueError("invalid function name")
        if not isinstance(cases, list) or not 1 <= len(cases) <= MAX_CASES:
            raise ValueError("one to 100 cases are required")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 0 < timeout <= 120:
            raise ValueError("timeout must be finite and between zero and 120 seconds")
        for case in cases:
            if not isinstance(case, dict) or set(case) - {"args", "kwargs", "expected"} or "expected" not in case:
                raise ValueError("invalid case fields")
            if not isinstance(case.get("args", []), list) or not isinstance(case.get("kwargs", {}), dict):
                raise ValueError("case args/kwargs have invalid types")
            _json_bytes(case)
    except (ValueError, TypeError, UnicodeError, RecursionError, OverflowError) as exc:
        return Verdict("error", False, detail=str(exc))
    runner = runner or BubblewrapRunner()
    deadline = time.monotonic() + timeout
    for index, case in enumerate(cases):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return Verdict("failed", False, index, len(cases), "suite timeout")
        try:
            result = runner.run_case(code, function, case.get("args", []), case.get("kwargs", {}), remaining)
        except (OSError, ValueError, subprocess.SubprocessError):
            return Verdict("error", False, index, len(cases), "execution infrastructure error")
        if result.status != "ok":
            status = "unavailable" if result.status == "unavailable" else "failed"
            return Verdict(status, False, index, len(cases), result.detail)
        if not _equal(result.value, case["expected"]):
            return Verdict("failed", False, index, len(cases), f"case {index + 1} output mismatch")
    return Verdict("passed", True, len(cases), len(cases), "all function cases passed")


# A trivial, controller-owned canary: a correct `x -> x + 1`. If the real sandbox
# cannot run even this and return the right value, the host cannot measure code
# at all, so the evaluation must BLOCK rather than silently score code tasks 0.
_PREFLIGHT_FUNCTION = "llm_eval_canary"
_PREFLIGHT_CODE = "def llm_eval_canary(x):\n    return x + 1\n"
_PREFLIGHT_CASES = [{"args": [41], "expected": 42}]


def sandbox_preflight(runner=None):
    """Actively verify the isolated evaluator works on THIS host.

    Runs a known-correct canary function through the REAL sandbox and checks the
    returned value. This is a true execution check, not a PATH probe: a host with
    `bwrap` installed but broken user namespaces or libseccomp reports
    `available=False`, so the gate can block instead of trusting a dead sandbox.

    Returns {"available": bool, "status": str, "detail": str}. `runner` is a
    trusted dependency-injection seam for tests, never a task field.
    """
    verdict = evaluate_function(_PREFLIGHT_CODE, _PREFLIGHT_FUNCTION,
                                _PREFLIGHT_CASES, timeout=10, runner=runner)
    available = verdict.status == "passed"
    detail = ("sandbox executed the canary function correctly" if available
              else (verdict.detail or verdict.status))
    return {"available": available, "status": verdict.status, "detail": detail}


def main(argv=None):
    """CLI for ops/CI/monitoring: `python -m evals.code_executor --preflight`
    prints a one-line JSON verdict and exits 0 when the sandbox is usable, 1 when
    it is not. Wire this into deployment health checks and CI before an eval run."""
    import argparse
    ap = argparse.ArgumentParser(description="Independent code evaluator sandbox")
    ap.add_argument("--preflight", action="store_true",
                    help="verify the sandbox works on this host (exit 0 usable, 1 not)")
    args = ap.parse_args(argv)
    if args.preflight:
        result = sandbox_preflight()
        print(json.dumps(result, sort_keys=True))
        return 0 if result["available"] else 1
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
