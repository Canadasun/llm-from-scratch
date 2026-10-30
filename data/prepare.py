"""
Prepare a training corpus for Milestone 1.

Default corpus: a small built-in sample so the repo trains out of the box with
no download. To train on your own software-dev / cybersecurity text instead,
point --input at a .txt file (e.g. a concatenation of man pages, RFCs, your own
notes, or public documentation) and this script will use that.

The output is a single UTF-8 text file at data/corpus.txt, which train.py reads.
"""

import argparse
import os

SAMPLE = """\
The transformer reads a sequence of tokens and predicts the next one.
A language model is just that prediction, repeated.
Security begins at the boundary: authenticate, then authorize, then audit.
Least privilege means a component can do its job and nothing more.
A packet crosses a trust boundary the moment it leaves your control.
Encryption protects confidentiality; signatures protect integrity.
Defense in depth: no single control is allowed to be the only one.
A model that memorizes its training data has not learned to generalize.
Gradient descent walks downhill on the loss surface, one small step at a time.
Attention lets every token look back at the tokens that came before it.
""" * 200  # repeated so the tiny demo has enough characters to train on


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None,
                    help="Path to a .txt corpus. If omitted, a built-in sample is used.")
    ap.add_argument("--output", default=os.path.join(os.path.dirname(__file__), "corpus.txt"))
    args = ap.parse_args()

    if args.input:
        with open(args.input, encoding="utf-8", errors="ignore") as f:
            text = f.read()
        print(f"Using corpus from {args.input}: {len(text):,} characters")
    else:
        text = SAMPLE
        print(f"Using built-in sample corpus: {len(text):,} characters")
        print("Tip: pass --input yourfile.txt to train on your own text.")

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
