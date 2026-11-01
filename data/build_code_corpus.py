"""Build a license-safe PYTHON CODE corpus for the next training step.

Why: the current corpus is security/network PROSE (RFCs + NIST) with almost no
code, which is why the model scores ~0 on Python tasks. This adds real,
idiomatic, license-safe Python plus original secure-coding / insecure-counter /
refusal examples, so a future run can actually learn to read and write Python.

Sources (all public or self-authored; no customer/PII data — see data/README.md):
  1. The local CPython standard library (PSF License — permissive, redistributable).
     Read from this interpreter's own stdlib directory; fully OFFLINE and $0.
  2. Original secure-coding examples / insecure counterexamples / refusal turns
     (data/build_code_examples.py) — written for this project.

Every file is secret-scrubbed (defense in depth), the result is checked for
train/eval contamination (data/contamination.py), and a manifest records the
sources, licenses, sizes and hashes. Output and caches are git-ignored.

    python data/build_code_corpus.py                 # ~40 MB default cap
    python data/build_code_corpus.py --max-bytes 8000000
"""

import argparse
import hashlib
import os
import sys
import sysconfig

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from data import build_code_examples            # noqa: E402
from data.contamination import assert_clean      # noqa: E402
from safety import scrub as _scrub               # noqa: E402

EVAL_DIR = os.path.join(os.path.dirname(HERE), "evals", "data")
# Stdlib subdirectories to skip: tests, bundled third-party, GUI demos, and
# anything that is not representative idiomatic application Python.
_SKIP_DIRS = {"test", "tests", "idlelib", "turtledemo", "lib2to3", "ensurepip",
              "site-packages", "__pycache__", "config-3", "venv"}
_MAX_FILE_BYTES = 128 * 1024          # skip very large generated/data modules


def iter_stdlib_files(stdlib_dir):
    """Yield (module_label, source_text) for stdlib .py files, deterministically
    (sorted), skipping tests/bundled/oversized files."""
    for dirpath, dirnames, filenames in os.walk(stdlib_dir):
        dirnames[:] = sorted(d for d in dirnames
                             if d not in _SKIP_DIRS and not d.startswith("test_"))
        for name in sorted(filenames):
            if not name.endswith(".py") or name.startswith("test_"):
                continue
            path = os.path.join(dirpath, name)
            try:
                if os.path.getsize(path) > _MAX_FILE_BYTES:
                    continue
                with open(path, encoding="utf-8", errors="replace") as fh:
                    src = fh.read()
            except OSError:
                continue
            label = os.path.relpath(path, stdlib_dir)
            yield label, src


def build(stdlib_dir=None, max_bytes=40_000_000, scrub=True):
    """Assemble the code corpus string + a manifest dict. Pure/deterministic for
    a given stdlib and cap; performs the contamination check before returning."""
    stdlib_dir = stdlib_dir or sysconfig.get_paths()["stdlib"]
    parts, files, total = [], 0, 0
    scrub_counts = {}

    # 1) Self-authored examples first (small, high-signal, always included).
    examples_text = build_code_examples.render()
    if scrub:
        examples_text, counts = _scrub.scrub_secrets(examples_text)
        for k, v in counts.items():
            scrub_counts[k] = scrub_counts.get(k, 0) + v
    parts.append("# ===== self-authored secure/insecure/refusal examples =====\n")
    parts.append(examples_text)
    total += len(examples_text.encode("utf-8"))

    # 2) CPython stdlib (PSF), until the byte cap.
    parts.append("\n# ===== CPython standard library (PSF License) =====\n")
    for label, src in iter_stdlib_files(stdlib_dir):
        if scrub:
            src, counts = _scrub.scrub_secrets(src)
            for k, v in counts.items():
                scrub_counts[k] = scrub_counts.get(k, 0) + v
        chunk = f"\n# --- stdlib: {label} ---\n{src}\n"
        encoded = len(chunk.encode("utf-8"))
        if total + encoded > max_bytes:
            break
        parts.append(chunk)
        files += 1
        total += encoded

    corpus = "".join(parts)
    # Anti-memorisation: the evaluation must not leak into training.
    assert_clean(corpus, EVAL_DIR)

    manifest = {
        "schema": "code-corpus-v1",
        "sources": [
            {"name": "self-authored examples", "license": "original (project)",
             "blocks": len(build_code_examples.examples())},
            {"name": "CPython standard library", "license": "PSF License",
             "root": stdlib_dir, "files": files},
        ],
        "bytes": len(corpus.encode("utf-8")),
        "max_bytes": max_bytes,
        "sha256": hashlib.sha256(corpus.encode("utf-8")).hexdigest(),
        "scrubbed": scrub,
        "secret_matches_redacted": scrub_counts,
        "contamination_checked_against": os.path.relpath(EVAL_DIR, os.path.dirname(HERE)),
        "policy": "public or self-authored only; no customer/PII data",
    }
    return corpus, manifest


def main():
    ap = argparse.ArgumentParser(description="Build a license-safe Python code corpus")
    ap.add_argument("--out", default=os.path.join(HERE, "corpus_code.txt"))
    ap.add_argument("--manifest", default=os.path.join(HERE, "code_corpus_manifest.json"))
    ap.add_argument("--stdlib", default=None, help="stdlib dir (default: this interpreter's)")
    ap.add_argument("--max-bytes", type=int, default=40_000_000)
    ap.add_argument("--no-scrub", action="store_true")
    args = ap.parse_args()

    corpus, manifest = build(args.stdlib, args.max_bytes, scrub=not args.no_scrub)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(corpus)
    import json
    with open(args.manifest, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f">> wrote {args.out}: {manifest['bytes']:,} bytes, "
          f"{manifest['sources'][1]['files']} stdlib files + "
          f"{manifest['sources'][0]['blocks']} example blocks")
    print(f">> sha256 {manifest['sha256'][:16]}…  scrub redactions: "
          f"{manifest['secret_matches_redacted'] or 'none'}")
    print(f">> contamination check vs {manifest['contamination_checked_against']}: PASSED")


if __name__ == "__main__":
    main()
