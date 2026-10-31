"""
Tests for fix verification (codereview/verify.py).

Synthetic tiny repos run the stdlib `unittest` (fast, no third-party deps). They
prove the end-to-end guarantee: a fix is "verified" only when it removes the
finding AND the tests still pass; a fix that breaks a test is rejected; and the
original repo is never modified.
"""

import os
import sys

import pytest

from codereview.verify import run_command, verify_fixes

TEST_CMD = f"{sys.executable} -m unittest -q"


def _write(d, name, text):
    with open(os.path.join(d, name), "w", encoding="utf-8") as f:
        f.write(text)


# ---- run_command ----

def test_run_command_pass_fail_timeout(tmp_path):
    d = str(tmp_path)
    assert run_command(f"{sys.executable} -c \"import sys;sys.exit(0)\"", d, 10).passed
    assert not run_command(f"{sys.executable} -c \"import sys;sys.exit(1)\"", d, 10).passed
    slow = run_command(f"{sys.executable} -c \"import time;time.sleep(5)\"", d, 1)
    assert slow.passed is False and slow.tail == "timeout"


# ---- verify_fixes ----

def _md5_repo(tmp_path, test_body):
    d = str(tmp_path)
    _write(d, "mod.py",
           "import hashlib\n\n\ndef checksum(b):\n    return hashlib.md5(b).hexdigest()\n")
    _write(d, "test_mod.py",
           "import unittest\nfrom mod import checksum\n\n\n"
           "class T(unittest.TestCase):\n" + test_body + "\n")
    return d


def test_verified_when_fix_resolves_and_tests_pass(tmp_path):
    # test only checks the return type, so md5->sha256 keeps it green
    d = _md5_repo(tmp_path, "    def test_type(self):\n"
                            "        self.assertIsInstance(checksum(b'x'), str)")
    r = verify_fixes(d, TEST_CMD, timeout=30)
    assert r.verdict == "verified"
    assert r.before.passed and r.after.passed
    assert r.findings_after == r.findings_before - 1
    assert any(rule == "PY-WEAK-HASH" for _, rule, _ in r.resolved)
    assert "mod.py" in r.files_changed


def test_rejected_when_fix_breaks_a_test(tmp_path):
    # test pins the exact md5 digest, so switching to sha256 must FAIL -> rejected
    empty_md5 = "d41d8cd98f00b204e9800998ecf8427e"
    d = _md5_repo(tmp_path, "    def test_exact(self):\n"
                            f"        self.assertEqual(checksum(b''), '{empty_md5}')")
    r = verify_fixes(d, TEST_CMD, timeout=30)
    assert r.verdict == "rejected_regression"
    assert r.before.passed and not r.after.passed


def test_no_fixes_when_only_unfixable_findings(tmp_path):
    d = str(tmp_path)
    _write(d, "mod.py", "def run(x):\n    return eval(x)\n")   # eval: not auto-fixable
    _write(d, "test_mod.py",
           "import unittest\n\n\nclass T(unittest.TestCase):\n"
           "    def test_ok(self):\n        self.assertTrue(True)\n")
    r = verify_fixes(d, TEST_CMD, timeout=30)
    assert r.verdict == "no_fixes"


def test_original_repo_is_never_modified(tmp_path):
    d = _md5_repo(tmp_path, "    def test_type(self):\n"
                            "        self.assertIsInstance(checksum(b'x'), str)")
    before = open(os.path.join(d, "mod.py"), encoding="utf-8").read()
    verify_fixes(d, TEST_CMD, timeout=30)
    after = open(os.path.join(d, "mod.py"), encoding="utf-8").read()
    assert before == after and "md5" in after    # untouched (still the original)
