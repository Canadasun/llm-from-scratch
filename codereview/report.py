"""
Format scan results into an honest, evidence-backed report.

Honest means: findings carry a CWE, a severity AND a confidence; potential
(heuristic) findings are not dressed up as proven; limitations are always
printed; and "no findings" is never reported as "secure".
"""

from collections import Counter

from codereview.findings import SEVERITY_ORDER


def summarize(findings):
    return {
        "total": len(findings),
        "by_severity": dict(Counter(f.severity for f in findings)),
        "by_rule": dict(Counter(f.rule_id for f in findings)),
    }


def format_report(scan, max_findings=100):
    findings = scan["findings"]
    s = summarize(findings)
    lines = []
    lines.append("=" * 66)
    lines.append("  SECURE CODE REVIEW — static analysis report")
    lines.append("=" * 66)
    lines.append(f"Files scanned: {scan['files_scanned']}   "
                 f"Findings: {s['total']}   Skipped: {len(scan['skipped'])}")
    if s["by_severity"]:
        order = sorted(s["by_severity"],
                       key=lambda k: -SEVERITY_ORDER.get(k, 0))
        lines.append("By severity: " +
                     ", ".join(f"{k} {s['by_severity'][k]}" for k in order))

    if not findings:
        lines.append("\nNo findings from the current rule set. This is NOT a")
        lines.append("proof of security — it only means these specific checks did")
        lines.append("not fire. See limitations below.")
    else:
        lines.append("")
        for f in findings[:max_findings]:
            lines.append(f"[{f.severity.upper():<8} {f.confidence:<6} conf] "
                         f"{f.rule_id}  {f.cwe}")
            lines.append(f"    {f.file}:{f.line}")
            lines.append(f"    {f.message}")
            if f.snippet:
                lines.append(f"    > {f.snippet}")
            if f.remediation:
                lines.append(f"    fix: {f.remediation}")
            lines.append("")
        if s["total"] > max_findings:
            lines.append(f"... {s['total'] - max_findings} more (raise --max-findings)")

    lines.append("Limitations (read before acting):")
    lines.append("  - Static pattern analysis, no data-flow/taint tracking: a")
    lines.append("    'high confidence' finding is a strong lead, not a proven")
    lines.append("    exploit. 'medium' may be a false positive (e.g. a weak hash")
    lines.append("    used for a non-security checksum). A human verifies each one.")
    lines.append("  - Covers a fixed set of Python rules (see codereview/rules.py);")
    lines.append("    absence of findings does not mean the code is secure.")
    lines.append("  - No CVEs are asserted; findings cite CWE classes only.")
    lines.append("  - Fixes are suggestions for human review, not auto-applied.")
    lines.append("=" * 66)
    return "\n".join(lines)
