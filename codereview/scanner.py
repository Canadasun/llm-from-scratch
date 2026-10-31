"""
Walk a path and run the rules on every Python file — read-only and bounded.

Files are opened and parsed, never imported or executed (they are untrusted
input). Limits on file count and size keep a hostile or huge repo from
exhausting resources.
"""

import os

from codereview.fixes import propose_fixes
from codereview.rules import analyze_source

MAX_FILE_BYTES = 2_000_000
MAX_FILES = 5000
SKIP_DIRS = {".git", ".venv", "venv", "env", "node_modules", "__pycache__",
             ".mypy_cache", ".pytest_cache", "build", "dist", ".tox"}


def iter_python_files(root):
    if os.path.isfile(root):
        if root.endswith(".py"):
            yield root
        return
    for dirpath, dirnames, filenames in os.walk(root):
        # prune noisy / vendored / hidden directories in place
        dirnames[:] = [d for d in dirnames
                       if d not in SKIP_DIRS and not d.startswith(".")]
        for f in sorted(filenames):
            if f.endswith(".py"):
                yield os.path.join(dirpath, f)


def scan_path(root, max_files=MAX_FILES, max_bytes=MAX_FILE_BYTES):
    """Return {"findings": [...sorted...], "files_scanned": n, "skipped": [...]}"""
    findings = []
    scanned = 0
    skipped = []
    for path in iter_python_files(root):
        if scanned >= max_files:
            skipped.append((path, "max_files_reached"))
            continue
        try:
            if os.path.getsize(path) > max_bytes:
                skipped.append((path, "too_large"))
                continue
            with open(path, encoding="utf-8", errors="replace") as fh:
                src = fh.read()
        except OSError as e:
            skipped.append((path, str(e)))
            continue
        findings.extend(analyze_source(src, filename=path))
        scanned += 1
    findings.sort(key=lambda f: f.sort_key())
    return {"findings": findings, "files_scanned": scanned, "skipped": skipped}


def collect_fixes(root, max_files=MAX_FILES, max_bytes=MAX_FILE_BYTES):
    """Walk a path and return the proposed fixes (FixProposal) for every file.
    Read-only: this proposes diffs, it never writes."""
    proposals = []
    scanned = 0
    for path in iter_python_files(root):
        if scanned >= max_files:
            break
        try:
            if os.path.getsize(path) > max_bytes:
                continue
            with open(path, encoding="utf-8", errors="replace") as fh:
                src = fh.read()
        except OSError:
            continue
        proposals.extend(propose_fixes(src, filename=path))
        scanned += 1
    return proposals
