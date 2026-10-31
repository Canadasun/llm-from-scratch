# Independent function evaluation

`evals/code_executor.py` compares generated Python functions against reviewed
JSON input/output cases. The controller retains expected values and decides
pass/fail. A fresh sandbox receives only the solution, function name, and one
case's arguments. There is no pass sentinel, test code, expected answer, or
controller process memory available to the solution.

This is a local execution backend/library, not a deployed multi-user service.

## Requirements and use

Use Linux with an up-to-date kernel, working unprivileged user namespaces,
`/usr/bin/python3`, bubblewrap and libseccomp. For example, on Debian/Ubuntu:

```bash
sudo apt-get install bubblewrap libseccomp2
python -m pytest -q tests/test_code_executor.py
python -m evals.run --suite tasks
```

`LLM_EVAL_BWRAP=/absolute/path/to/bwrap` selects a deployment-managed binary when
it is not on PATH. This is trusted deployment configuration, never a task field.
Set `LLM_REQUIRE_SANDBOX_TESTS=1` to fail rather than skip integration tests if
isolation is unavailable. Both CI workflows install the isolation dependencies;
unit-test CI requires integration tests to actually run. A host that prohibits
namespaces must use an approved execution host; there is no host fallback and no
automatic relaxation of host security settings.

```python
from evals.code_executor import evaluate_function

verdict = evaluate_function(
    "def add(a, b): return a + b",
    "add",
    [{"args": [2, 3], "expected": 5}],
    timeout=10,
)
```

## Isolation and limits

- Bubblewrap creates private user, PID, network, IPC, and other namespaces;
  capabilities are dropped, and the sandbox runs as UID/GID 65534 internally.
- Only read-only runtime paths and a read-only job directory are mounted. No
  host home, credentials, test suite, other jobs, writable scratch, or host
  `/proc` is exposed. The controller uses an empty environment.
- A required libseccomp filter denies sockets, fork/clone, namespace changes,
  mounts, ptrace, process-memory APIs, kernel key operations, and other sensitive
  calls. There is no optional weaker execution mode.
- Each worker has hard CPU, memory, file-size, descriptor, process-count and core
  limits. A controller-enforced total timeout kills the worker process group.
  Output goes to size-limited files, not unbounded parent-memory capture.
- Responses must be one bounded JSON object containing only `value`. Duplicate
  keys, non-finite numbers, multiple responses, invalid output, unsuccessful
  exits and missing responses fail. Booleans cannot pass integer expectations.
- Changing stdout, monkeypatching unittest, or exiting zero cannot supply a
  trusted verdict. A solution can fabricate a *value*, but it must match the
  controller's expected value for every case, which is the observable contract.

The namespace and seccomp policy follows bubblewrap's explicit-policy model:
see the [upstream security documentation](https://github.com/containers/bubblewrap#sandbox-security).
It does not make kernel vulnerabilities disappear. Run hostile evaluations on
dedicated disposable hosts without production secrets, with host-level quotas
and patching. The read-only `/usr` runtime is trusted infrastructure; avoid
installing sensitive application data there.

## Cases, task binding and reporting

The 18 existing Python tasks have reviewed cases in `evals/data/code_cases.json`.
The manifest binds each task's prompt and legacy test text with SHA-256, so an
edited or unknown task cannot silently reuse another task's cases. The legacy
unittest source is retained in the original dataset but never executed by this
path. Review/update manifest cases and hashes deliberately when changing tasks.

Task results include `execution_status`, `cases_passed`, and `cases_total`.
`passed`/`failed` are completed evaluator outcomes; `unavailable`/`error` are
infrastructure or configuration failures. The regression gate rejects the latter
regardless of accuracy thresholds. Finite cases test observable behavior, not
the security of the source or its ability to generalize. Public fixtures are
not hidden benchmarks; private release cases should remain controller-side too.

## Remaining scope

The protocol supports synchronous JSON-compatible function calls. It does not
support arbitrary unittest source, persistent objects, filesystem projects,
external packages, network-dependent tasks, or multiple languages. Those need
separate reviewed protocols with an equally independent oracle. The old
`run_python_unittest(..., trusted=True)` helper is still only for trusted fixtures;
it must not be used to grade arbitrary model output.
