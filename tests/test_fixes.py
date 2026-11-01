"""
Tests for patch proposals (codereview/fixes.py).

The important property: a proposed fix, once applied, actually REMOVES the
finding and leaves valid Python. That verification loop (propose -> apply ->
re-scan) is what makes a suggested patch trustworthy rather than cosmetic.
"""

import ast
import os
from collections import Counter

from codereview.fixes import apply_fixes, propose_fixes
from codereview.rules import analyze_source
from codereview.__main__ import main

HERE = os.path.dirname(__file__)
TESTDATA = os.path.normpath(os.path.join(HERE, "..", "codereview", "testdata"))
VULN = os.path.join(TESTDATA, "vulnerable.py")
SAFE = os.path.join(TESTDATA, "safe.py")


def _read(p):
    with open(p, encoding="utf-8") as f:
        return f.read()


def test_only_safe_fixable_rules_are_proposed():
    props = propose_fixes(_read(VULN), filename=VULN)
    rules = Counter(p.rule_id for p in props)
    assert dict(rules) == {
        "PY-WEAK-HASH": 1,
        "PY-INSECURE-DESERIALIZATION": 1,   # yaml.load only; pickle is NOT auto-fixed
        "PY-TLS-VERIFY-DISABLED": 1,
        "PY-FLASK-DEBUG": 1,
    }


def test_no_fixes_proposed_for_safe_code():
    assert propose_fixes(_read(SAFE), filename=SAFE) == []


def test_each_individual_fix_resolves_its_finding():
    checks = [
        ("import hashlib\nhashlib.md5(x)\n", "hashlib.sha256"),
        ("import yaml\nyaml.load(d)\n", "yaml.safe_load"),
        ("import requests\nrequests.get(u, verify=False)\n", "verify=True"),
        ("app.run(debug=True)\n", "debug=False"),
    ]
    for src, expected_text in checks:
        props = propose_fixes(src)
        assert len(props) == 1
        patched = apply_fixes(src, props)
        assert expected_text in patched
        ast.parse(patched)                       # still valid Python
        assert analyze_source(patched) == []     # finding is gone


def test_apply_all_fixes_reduces_findings_and_keeps_valid_python():
    src = _read(VULN)
    before = len(analyze_source(src, filename=VULN))
    patched = apply_fixes(src, propose_fixes(src, filename=VULN))
    ast.parse(patched)                           # multi-edit result still parses
    after = len(analyze_source(patched, filename=VULN))
    assert after == before - 4                   # the 4 auto-fixable ones resolved


def test_yaml_load_with_explicit_unsafe_loader_removes_loader_keyword():
    # A reviewed Loader must be removed as part of the call replacement.
    src = "import yaml\nyaml.load(d, Loader=yaml.Loader)\n"
    assert "yaml.safe_load(d)" in apply_fixes(src, propose_fixes(src))


def test_fix_carries_a_caveat():
    props = propose_fixes("import hashlib\nhashlib.md5(x)\n")
    assert props[0].caveat        # non-empty: reviewer is warned about behaviour change


def test_cli_fix_flag_prints_proposals_and_exits_0(capsys):
    rc = main([VULN, "--fix"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "PROPOSED FIXES" in out
    assert "NOTHING is written" in out
    assert "yaml.safe_load" in out


def test_cli_fix_does_not_modify_the_file():
    before = _read(VULN)
    main([VULN, "--fix"])
    assert _read(VULN) == before          # suggestion-only: file untouched


def test_fix_handles_non_ascii_before_target():
    # A non-ASCII character earlier on the line shifts ast byte offsets vs string
    # indices; the fix must still splice correctly and yield valid Python.
    src = 'import hashlib\nx = "é"; h = hashlib.md5(b"a").hexdigest()\n'
    proposals = propose_fixes(src)
    assert any(p.rule_id for p in proposals)
    new = apply_fixes(src, proposals)
    ast.parse(new)                      # not corrupted
    assert "sha256" in new and "é" in new
