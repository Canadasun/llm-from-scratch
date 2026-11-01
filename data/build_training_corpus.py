"""Assemble the TRAINING corpus = prose (RFC/NIST) + Python code, for the
coding-capability run.

The prose corpus (data/corpus.txt from build_corpus.py) teaches the security/
networking domain; the code corpus (build_code_corpus.py, offline from the local
CPython stdlib + self-authored examples) teaches Python. This mixes them into one
file for train.py (`--corpus`), runs the train/eval contamination guard on the
COMBINED text, and reports the composition.

Offline and $0: the code half reads the local stdlib; the prose half is whatever
build_corpus.py already produced (optional). Nothing is downloaded here.

    python data/build_corpus.py                      # optional: prose half (needs net)
    python data/build_training_corpus.py             # writes data/corpus_training.txt
    python train.py --preset gpu_xl --corpus data/corpus_training.txt
"""

import argparse
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from data.build_code_corpus import build as build_code     # noqa: E402
from data.contamination import assert_clean                 # noqa: E402

EVAL_DIR = os.path.join(os.path.dirname(HERE), "evals", "data")
_PROSE_DEFAULT = os.path.join(HERE, "corpus.txt")


def build_training_corpus(prose_path=_PROSE_DEFAULT, code_max_bytes=40_000_000, scrub=True):
    """Return (corpus_text, info). Reads the prose corpus if present, builds the
    code corpus offline, concatenates them, and asserts no eval contamination."""
    parts, info = [], {"prose_bytes": 0, "code_bytes": 0}

    prose = ""
    if prose_path and os.path.exists(prose_path):
        with open(prose_path, encoding="utf-8") as fh:
            prose = fh.read()
    if prose.strip():
        parts.append("# ===== domain prose (RFCs + NIST) =====\n")
        parts.append(prose)
        info["prose_bytes"] = len(prose.encode("utf-8"))

    code, code_manifest = build_code(max_bytes=code_max_bytes, scrub=scrub)
    parts.append("\n# ===== Python code corpus =====\n")
    parts.append(code)
    info["code_bytes"] = len(code.encode("utf-8"))
    info["code_manifest"] = code_manifest

    corpus = "".join(parts)
    assert_clean(corpus, EVAL_DIR)        # combined text must stay eval-clean
    info["bytes"] = len(corpus.encode("utf-8"))
    info["sha256"] = hashlib.sha256(corpus.encode("utf-8")).hexdigest()
    return corpus, info


def main():
    ap = argparse.ArgumentParser(description="Build the prose+code training corpus")
    ap.add_argument("--prose", default=_PROSE_DEFAULT,
                    help="prose corpus (default data/corpus.txt; skipped if absent)")
    ap.add_argument("--out", default=os.path.join(HERE, "corpus_training.txt"))
    ap.add_argument("--code-max-bytes", type=int, default=40_000_000)
    ap.add_argument("--no-scrub", action="store_true")
    args = ap.parse_args()

    corpus, info = build_training_corpus(args.prose, args.code_max_bytes, scrub=not args.no_scrub)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(corpus)
    total = info["bytes"] or 1
    print(f">> wrote {args.out}: {info['bytes']:,} bytes "
          f"(prose {info['prose_bytes']:,} = {100*info['prose_bytes']//total}%, "
          f"code {info['code_bytes']:,} = {100*info['code_bytes']//total}%)")
    print(f">> sha256 {info['sha256'][:16]}…  contamination check vs evals/data: PASSED")
    if not info["prose_bytes"]:
        print(">> note: no prose corpus found (data/corpus.txt). Code-only training "
              "corpus. Run data/build_corpus.py first to include the RFC/NIST prose.")


if __name__ == "__main__":
    main()
