"""
Tests for the secure code-review engine.

The important ones are the ground-truth checks: every seeded vulnerability in
testdata/vulnerable.py must be found (no false negatives), and testdata/safe.py
must produce ZERO findings (no false positives). That is how we keep the tool's
accuracy honest instead of asserting it.
"""

import os
from collections import Counter

from codereview.findings import Finding
from codereview.rules import analyze_source
from codereview.report import format_report, summarize
from codereview.scanner import scan_path
from codereview.__main__ import main

HERE = os.path.dirname(__file__)
TESTDATA = os.path.normpath(os.path.join(HERE, "..", "codereview", "testdata"))
VULN = os.path.join(TESTDATA, "vulnerable.py")
SAFE = os.path.join(TESTDATA, "safe.py")

EXPECTED = {
    "PY-SHELL-INJECTION": 2,
    "PY-DANGEROUS-EVAL": 1,
    "PY-INSECURE-DESERIALIZATION": 2,
    "PY-WEAK-HASH": 1,
    "PY-TLS-VERIFY-DISABLED": 1,
    "PY-HARDCODED-SECRET": 1,
    "PY-FLASK-DEBUG": 1,
    "PY-SQL-INJECTION": 1,
    "PY-INSECURE-TEMPFILE": 1,
    "PY-WEAK-RANDOM": 1,
    "PY-DJANGO-DEBUG": 1,
    "PY-TARFILE-TRAVERSAL": 1,
    "PY-JINJA-AUTOESCAPE-OFF": 1,
    "PY-BIND-ALL-INTERFACES": 1,
}
TOTAL_EXPECTED = sum(EXPECTED.values())   # 16


def _scan_file(path):
    with open(path, encoding="utf-8") as f:
        return analyze_source(f.read(), filename=path)


# ---- ground truth: no false negatives ----

def test_vulnerable_fixture_all_rules_fire():
    counts = Counter(f.rule_id for f in _scan_file(VULN))
    assert dict(counts) == EXPECTED


def test_every_rule_has_cwe_and_valid_fields():
    for f in _scan_file(VULN):
        assert f.cwe.startswith("CWE-")
        assert f.severity in {"critical", "high", "medium", "low", "info"}
        assert f.confidence in {"high", "medium", "low"}
        assert f.snippet            # evidence present
        assert f.line > 0


# ---- ground truth: no false positives ----

def test_safe_fixture_produces_no_findings():
    findings = _scan_file(SAFE)
    assert findings == [], f"false positives: {[f.rule_id for f in findings]}"


# ---- a few focused rule unit tests ----

def test_shell_true_literal_is_lower_confidence_than_dynamic():
    dyn = analyze_source("import subprocess\nsubprocess.run(cmd, shell=True)\n")
    lit = analyze_source("import subprocess\nsubprocess.run('ls', shell=True)\n")
    assert dyn[0].confidence == "high"
    assert lit[0].confidence == "medium"


def test_sql_parameterised_is_not_flagged():
    assert analyze_source("cur.execute('SELECT 1 WHERE a = ?', (x,))\n") == []


def test_yaml_safe_load_not_flagged():
    assert analyze_source("import yaml\nyaml.safe_load(d)\n") == []


def test_hardcoded_secret_placeholder_ignored():
    assert analyze_source("password = 'changeme'\n") == []
    assert len(analyze_source("password = 'S3cr3tL0ngValue!'\n")) == 1


def test_hardcoded_secret_matches_words_not_substrings():
    # must NOT flag names that merely contain a secret word as a substring
    assert analyze_source('tokenizer = "char"\n') == []
    assert analyze_source('sort_key = "name"\n') == []
    # must flag real secret-ish names
    assert len(analyze_source('api_key = "AKIAEXAMPLE1234567"\n')) == 1
    assert len(analyze_source('access_token = "ya29.longtokenvalue"\n')) == 1
    assert len(analyze_source('AWS_SECRET_ACCESS_KEY = "abcd1234longsecret"\n')) == 1


def test_parse_error_is_reported_not_raised():
    out = analyze_source("def broken(:\n")
    assert len(out) == 1 and out[0].rule_id == "PY-PARSE-ERROR"


# ---- scanner ----

def test_scanner_finds_vuln_and_not_safe():
    scan = scan_path(TESTDATA)
    assert scan["files_scanned"] == 2
    rule_ids = Counter(f.rule_id for f in scan["findings"])
    assert dict(rule_ids) == EXPECTED        # safe.py contributes nothing


def test_scanner_respects_max_files():
    scan = scan_path(TESTDATA, max_files=1)
    assert scan["files_scanned"] == 1


def test_findings_sorted_most_severe_first():
    scan = scan_path(TESTDATA)
    sevs = [f.severity for f in scan["findings"]]
    # critical (eval) must come before high, high before medium
    assert sevs == sorted(sevs, key=lambda s:
                          -{"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}[s])


# ---- report ----

def test_report_has_counts_and_limitations():
    scan = scan_path(TESTDATA)
    text = format_report(scan)
    assert "SECURE CODE REVIEW" in text
    assert "Limitations" in text
    assert summarize(scan["findings"])["total"] == TOTAL_EXPECTED


def test_report_empty_is_not_called_secure():
    scan = scan_path(SAFE)
    text = format_report(scan)
    assert "NOT" in text and "proof of security" in text


# ---- CLI gate ----

def test_cli_fail_on_high_returns_1(capsys):
    assert main([VULN, "--fail-on", "high"]) == 1


def test_cli_report_only_returns_0(capsys):
    assert main([SAFE]) == 0


def test_cli_fail_on_critical_safe_returns_0(capsys):
    assert main([SAFE, "--fail-on", "critical"]) == 0
