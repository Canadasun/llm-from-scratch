"""
Tests for PR assembly (codereview/pr.py).

prepare_pr() must have NO side effects (no git). The git-local step
(apply_fixes_to_branch) is tested in a throwaway `git init` repo, and must refuse
unverified fixes or a dirty tree. Nothing here touches a real remote or runs gh.
"""

import os
import subprocess
import sys

import pytest

from codereview.pr import (PRArtifact, apply_fixes_to_branch, prepare_pr,
                           write_artifact)

TEST_CMD = f"{sys.executable} -m unittest -q"


def _write(d, name, text):
    with open(os.path.join(d, name), "w", encoding="utf-8") as f:
        f.write(text)


def _md5_repo(d, exact=False):
    _write(d, "mod.py",
           "import hashlib\n\n\ndef checksum(b):\n    return hashlib.md5(b).hexdigest()\n")
    body = ("    def test_type(self):\n"
            "        self.assertIsInstance(checksum(b'x'), str)")
    _write(d, "test_mod.py",
           "import unittest\nfrom mod import checksum\n\n\n"
           "class T(unittest.TestCase):\n" + body + "\n")


# ---- prepare_pr: no side effects ----

def test_prepare_pr_verified_builds_patch_and_body(tmp_path):
    d = str(tmp_path)
    _md5_repo(d)
    art = prepare_pr(d, TEST_CMD, branch="security/test", timeout=30, trusted=True)
    assert art.verified is True
    assert art.verdict == "verified"
    assert "mod.py" in art.files_changed
    assert "hashlib.sha256" in art.patch
    assert "do not auto-merge" in art.body.lower()
    assert "CWE-327" in art.body
    assert "verified code-review fix" in art.title
    # no files created, no git dir
    assert not os.path.exists(os.path.join(d, "CODEREVIEW_PR.md"))
    assert not os.path.exists(os.path.join(d, ".git"))


def test_prepare_pr_unfixable_is_not_verified(tmp_path):
    d = str(tmp_path)
    _write(d, "mod.py", "def run(x):\n    return eval(x)\n")
    _write(d, "test_mod.py",
           "import unittest\n\n\nclass T(unittest.TestCase):\n"
           "    def test_ok(self):\n        self.assertTrue(True)\n")
    art = prepare_pr(d, TEST_CMD, timeout=30, trusted=True)
    assert art.verified is False
    assert art.patch == ""
    assert "for review" in art.title


def test_write_artifact_creates_files(tmp_path):
    d = str(tmp_path)
    _md5_repo(d)
    art = prepare_pr(d, TEST_CMD, timeout=30, trusted=True)
    out = str(tmp_path / "out")
    os.makedirs(out)
    body_path, patch_path = write_artifact(art, out)
    assert os.path.exists(body_path) and os.path.exists(patch_path)
    assert "hashlib.sha256" in open(patch_path).read()


# ---- git-local step in a throwaway repo ----

def _init_git_repo(d):
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    subprocess.run(["git", "init", "-q"], cwd=d, check=True, env=env)
    subprocess.run(["git", "config", "user.email", "t@test.local"], cwd=d, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=d, check=True)
    subprocess.run(["git", "add", "-A"], cwd=d, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "baseline"], cwd=d, check=True)


def _file_on_branch(d, branch, path):
    return subprocess.run(["git", "show", f"{branch}:{path}"], cwd=d,
                          capture_output=True, text=True).stdout


def test_apply_fixes_to_branch_commits_on_new_branch(tmp_path):
    d = str(tmp_path)
    _md5_repo(d)
    _init_git_repo(d)
    base = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=d,
                          capture_output=True, text=True).stdout.strip()
    art = prepare_pr(d, TEST_CMD, branch="security/fixes-x", timeout=30, trusted=True)
    ret_base, branch = apply_fixes_to_branch(d, art)
    assert ret_base == base and branch == "security/fixes-x"
    # new branch has the fix; base branch is untouched
    assert "hashlib.sha256" in _file_on_branch(d, branch, "mod.py")
    assert "hashlib.md5" in _file_on_branch(d, base, "mod.py")


def test_apply_refuses_dirty_tree(tmp_path):
    d = str(tmp_path)
    _md5_repo(d)
    _init_git_repo(d)
    art = prepare_pr(d, TEST_CMD, timeout=30, trusted=True)
    _write(d, "mod.py", "# uncommitted edit\nimport hashlib\n")   # dirty
    with pytest.raises(RuntimeError, match="not clean"):
        apply_fixes_to_branch(d, art)


def test_apply_refuses_unverified():
    art = PRArtifact(verified=False, branch="b", title="t", body="", patch="")
    with pytest.raises(RuntimeError, match="not verified"):
        apply_fixes_to_branch(".", art)


# ---- patch integrity: bind verification to the commit + exact patch ----

def test_apply_refuses_when_repo_drifts_after_verification(tmp_path):
    d = str(tmp_path)
    _md5_repo(d)
    _init_git_repo(d)
    art = prepare_pr(d, TEST_CMD, branch="security/fixes-drift", timeout=30, trusted=True)
    # The repo moves on (a new commit) AFTER verification: the artifact is bound
    # to the old commit and must be refused rather than applied to drifted code.
    _write(d, "another.py", "x = 1\n")
    subprocess.run(["git", "add", "-A"], cwd=d, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "drift"], cwd=d, check=True)
    with pytest.raises(RuntimeError, match="drift"):
        apply_fixes_to_branch(d, art)


def test_apply_refuses_tampered_patch(tmp_path):
    d = str(tmp_path)
    _md5_repo(d)
    _init_git_repo(d)
    art = prepare_pr(d, TEST_CMD, branch="security/fixes-tamper", timeout=30, trusted=True)
    art.patch = art.patch + "\n# tampered\n"           # hash no longer matches
    with pytest.raises(RuntimeError, match="hash"):
        apply_fixes_to_branch(d, art)


def test_apply_commits_the_exact_verified_patch(tmp_path):
    d = str(tmp_path)
    _md5_repo(d)
    _init_git_repo(d)
    art = prepare_pr(d, TEST_CMD, branch="security/fixes-exact", timeout=30, trusted=True)
    assert art.base_commit and art.patch_sha256
    _base, branch = apply_fixes_to_branch(d, art)
    # The committed content is exactly the verified fix.
    assert "hashlib.sha256" in _file_on_branch(d, branch, "mod.py")
    # The commit message records the binding.
    msg = subprocess.run(["git", "log", "-1", "--format=%B", branch], cwd=d,
                         capture_output=True, text=True).stdout
    assert art.base_commit in msg and art.patch_sha256 in msg
