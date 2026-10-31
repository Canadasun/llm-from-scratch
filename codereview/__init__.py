"""
Secure code review — the first "excellent workflow" of the enterprise platform
(Stage 2 of the endorsed staged plan; see the enterprise direction).

The workflow is: read a repo -> find real vulnerabilities -> (propose a patch) ->
(run tests) -> produce an evidence-backed report. This package implements the
analysis core.

Design principles (non-negotiable, from the platform directive):
  - The FINDINGS come from deterministic, auditable rules (Python's `ast`), not
    from model-generated text. A ~32M model cannot be trusted to find vulns; the
    platform's value is verifiable analysis + evidence, not a fluent opinion.
  - Every analyzed file is UNTRUSTED INPUT. It is parsed, never imported or
    executed. The scanner is read-only and bounded (file count/size limits).
  - Honesty is enforced: each rule ships with ground-truth vulnerable AND safe
    fixtures, so false-positive / false-negative rates are MEASURED (tests), not
    claimed. No fabricated CVEs; findings cite CWE ids and carry a confidence.
  - Nothing here decides authorization or auto-applies a fix. Patches are
    proposed for human review (later increment); high-impact action needs a
    human.

Layout:
  findings.py  the Finding/Severity/Confidence types
  rules.py     AST-based detectors, each mapped to a CWE
  scanner.py   walk files, parse, run rules, collect findings (read-only, bounded)
  report.py    format findings + an honest summary (confirmed vs potential, limits)
  __main__.py  CLI: python -m codereview <path> [--json out.json]
"""
