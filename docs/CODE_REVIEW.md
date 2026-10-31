# Secure code review (enterprise platform — Stage 2)

The first "excellent workflow" of the enterprise platform: **read code → find real
vulnerabilities → report with evidence** (patch-proposal and test-running are the
next increments). It is the highest-priority capability in the platform plan —
secure coding that finds *real* issues with an acceptable false-positive rate.

## Why it is built the way it is

- **Deterministic analysis, not the model.** Findings come from `ast`-based rules,
  not from model-generated text. A ~32M from-scratch model cannot be trusted to
  find vulnerabilities; the platform's value is *verifiable* analysis + evidence.
- **Accuracy is measured, not claimed.** Every rule ships with a vulnerable
  fixture (must be caught) and a safe fixture (must not false-positive) under
  `codereview/testdata/`. `tests/test_codereview.py` enforces **zero false
  positives on the safe fixture and zero false negatives on the vulnerable one**.
- **Untrusted input.** Files are parsed, never imported or executed. The scanner
  is read-only and bounded (file count/size, skips vendored/hidden dirs).
- **Honest reporting.** Findings cite a **CWE** (never a fabricated CVE), carry a
  **severity** and a separate **confidence**, and the report always prints
  limitations. "No findings" is explicitly *not* "secure".

## Run it

```bash
python -m codereview path/to/repo              # scan + report
python -m codereview file.py --json out.json   # machine-readable output
python -m codereview . --fail-on high          # exit 1 if any finding >= high (CI gate)
python -m codereview path/to/repo --fix        # also print proposed patch diffs
```

By default it reports and exits 0; `--fail-on` turns it into a gate.

## Proposing fixes (`--fix`)

For the subset of findings with a **safe, deterministic, behaviour-preserving**
fix, the tool proposes a concrete unified diff — **it never writes anything**.
Proposing a patch and applying it are different authority levels (applying is a
later increment, gated on human approval and a PR).

Auto-fixable: weak hash (`md5`/`sha1` → `sha256`), `yaml.load` → `yaml.safe_load`,
`verify=False` → `verify=True`, `debug=True` → `debug=False`. Each diff carries a
**caveat** where the change can alter behaviour (e.g. changing a hash invalidates
previously stored digests), so the reviewer decides.

Deliberately **not** auto-fixed — shell/SQL injection, `eval`/`exec`, hardcoded
secrets — because a correct fix requires restructuring, not a local edit. Those
stay as manual findings.

Every proposed fix is verified in tests with a **propose → apply → re-scan** loop:
applying it must remove the finding and leave valid Python.

## Verifying fixes against the project's tests (`--verify-fixes`)

A patch is only trustworthy if it removes the finding **and** doesn't break the
project. `--verify-fixes` proves that end to end:

```
copy the repo → run tests (baseline) → apply the reviewed fixes to the copy →
re-scan (findings gone?) → run tests again (still pass?) → verdict
```

```bash
python -m codereview . --verify-fixes --test-cmd "python -m pytest -q"
```

Everything happens on a **copy** in a temp dir — the original is never modified.
Verdicts: `verified` (findings resolved, tests still pass), `rejected_regression`
(a fix broke a test — do not apply), `baseline_failing` (tests were already red,
so nothing can be attributed), `no_fixes`, `no_change`.

**Sandbox scope (honest).** The test command runs in a child process with a
throwaway working dir, a **scrubbed environment (no parent-process secrets)**,
CPU-time and file-size rlimits, and a wall-clock timeout. It does **not** provide
filesystem or network isolation — that needs OS containers/namespaces, which is
out of scope here. Running a project's tests is arbitrary code execution, so use
this against a **trusted** repo (this project's own tests) or inside a container
you already trust.

## Rules (initial set)

| rule id | CWE | severity | what it catches |
|---|---|---|---|
| PY-DANGEROUS-EVAL | CWE-95 | critical | `eval()`/`exec()` on a runtime value |
| PY-SHELL-INJECTION | CWE-78 | high | `subprocess(..., shell=True)`, `os.system` with a built command |
| PY-SQL-INJECTION | CWE-89 | high | `.execute()` with a string built at runtime (not parameterised) |
| PY-INSECURE-DESERIALIZATION | CWE-502 | high | `pickle.loads`, `yaml.load` without `SafeLoader` |
| PY-TLS-VERIFY-DISABLED | CWE-295 | high | `verify=False`, `ssl._create_unverified_context` |
| PY-HARDCODED-SECRET | CWE-798 | high (med conf) | string literal assigned to a secret-named variable |
| PY-WEAK-HASH | CWE-327 | medium | `hashlib.md5`/`sha1` (ok only for non-security checksums) |
| PY-FLASK-DEBUG | CWE-489 | medium | `app.run(debug=True)` |

Confidence is separate from severity: a `high` severity with `medium` confidence
is a serious issue *if* it is real, and a human verifies it. AST scoping keeps
false positives down (a literal is told from a runtime string, a keyword from a
coincidence) — verified on the first-party packages, which scan clean.

## Example

```
[CRITICAL high   conf] PY-DANGEROUS-EVAL  CWE-95
    app/handlers.py:23
    eval() on a runtime value executes arbitrary code
    > return eval(user_input)
    fix: Avoid eval/exec on dynamic input; use ast.literal_eval or a dispatch table.
```

## Assembling a PR for human review (`--prepare-pr` / `--open-pr`)

The last step turns findings + verified fixes into a reviewable pull request.
Two authority levels are kept separate on purpose:

```bash
# build the PR artifact — evidence body + combined patch. NO git, NO network.
python -m codereview . --prepare-pr --test-cmd "python -m pytest -q"
#   -> writes CODEREVIEW_PR.md (the evidence report) and codereview_fixes.patch

# EXPLICIT opt-in: create a branch and commit the VERIFIED fixes (refuses on a
# dirty tree or unverified fixes). Local only unless you add --push.
python -m codereview . --open-pr --test-cmd "python -m pytest -q"
python -m codereview . --open-pr --push --test-cmd "python -m pytest -q"   # + gh pr create
```

`prepare_pr()` has **no side effects**. The git/gh operations live behind
`--open-pr`, which **only ever commits `verified` fixes**, refuses a dirty working
tree, and never auto-merges. The PR body carries the verification evidence (tests
before/after, findings resolved), the applied-fix table **with caveats**, the
combined diff, all findings (including the manual-only ones), and the limitations
— so a human reviews a complete, honest artifact.

## Where this sits in the plan & what's next

This is **Stage 2** of the endorsed staged plan (Stage 1 = prove the model learns,
done). Increments:
1. ✅ **Patch proposals** — safe, deterministic fixes proposed as diffs for a
   subset of rules, with caveats; verified by a propose→apply→re-scan loop; never
   auto-applied (`--fix`).
2. ✅ **Verify fixes against the project's tests** — copy the repo, run the test
   command before/after applying the reviewed fixes in a bounded sandbox, and
   confirm findings are resolved with no test regression (`--verify-fixes`).
3. ✅ **Evidence-backed report + PR** — assemble findings + verified fixes + test
   evidence into a PR artifact (`--prepare-pr`, no side effects); open the PR
   behind an explicit, guarded opt-in that only commits verified fixes
   (`--open-pr`). Never auto-merged; authorization is a human decision.
4. Extend language/rule coverage; add FP/FN *rate* tracking across a larger
   labelled corpus.

With increments 1–3 done, the secure-code-review workflow is complete end to end:
**read → find (measured FP/FN) → propose (verified diffs) → verify (tests) → PR
(human review).**

Stage 3 (enterprise security: identities, repo permissions, tenant isolation,
audited tool execution) and Stage 4 (authorized pilot) follow. See the project's
enterprise direction for the full 9-phase plan and release gates.
