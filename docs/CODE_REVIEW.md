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
```

By default it reports and exits 0; `--fail-on` turns it into a gate.

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

## Where this sits in the plan & what's next

This is **Stage 2** of the endorsed staged plan (Stage 1 = prove the model learns,
done). Next increments, in order:
1. **Patch proposals** — for a subset of rules, generate a concrete, human-review
   diff (never auto-applied).
2. **Verify fixes** — run the project's tests in the sandbox before/after.
3. **Evidence-backed report + PR** — open a pull request for human review (P6).
4. Extend language/rule coverage; add FP/FN *rate* tracking across a larger
   labelled corpus.

Stage 3 (enterprise security: identities, repo permissions, tenant isolation,
audited tool execution) and Stage 4 (authorized pilot) follow. See the project's
enterprise direction for the full 9-phase plan and release gates.
