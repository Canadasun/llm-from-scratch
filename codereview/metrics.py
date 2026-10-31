"""
Measure the detector's precision / recall against a LABELLED corpus.

"Accuracy is measured, not claimed" needs a number. This scores the current rule
set against a frozen ground-truth label file (codereview/testdata/labels.json):

    precision = TP / (TP + FP)   — of what we flagged, how much was real
    recall    = TP / (TP + FN)   — of the real issues, how much we caught
    F1        = harmonic mean

A finding is matched by (file, rule_id, line). The label file is frozen, so if a
rule change starts missing a labelled issue (recall drops) or flags a clean line
(precision drops), the metric — and the test that asserts it — catches it.

Honest scope: these rates are over THIS labelled corpus. They are a regression
guard, not a claim about all code. The way to make recall a real-world number is
to GROW the corpus with independent, harder cases (including known misses) and
watch the number move — that is the intended next step.
"""

import json
import os

from codereview.scanner import scan_path

LABELS_PATH = os.path.join(os.path.dirname(__file__), "testdata", "labels.json")
CORPUS_DIR = os.path.join(os.path.dirname(__file__), "testdata")


def _found_keys(corpus_dir):
    scan = scan_path(corpus_dir)
    return {(os.path.relpath(f.file, corpus_dir), f.rule_id, f.line)
            for f in scan["findings"]}


def generate_labels(corpus_dir):
    """Produce a labels dict from the current scan (used to FREEZE ground truth
    once; not run during measurement)."""
    keys = sorted(_found_keys(corpus_dir))
    files = set()
    for dirpath, _dirs, names in os.walk(corpus_dir):
        for n in names:
            if n.endswith(".py"):
                files.add(os.path.relpath(os.path.join(dirpath, n), corpus_dir))
    flagged_files = {k[0] for k in keys}
    return {
        "expected": [{"file": f, "rule_id": r, "line": ln} for f, r, ln in keys],
        "clean": sorted(files - flagged_files),
    }


def compute_metrics(corpus_dir=CORPUS_DIR, labels_path=LABELS_PATH):
    with open(labels_path, encoding="utf-8") as f:
        labels = json.load(f)
    expected = {(e["file"], e["rule_id"], e["line"]) for e in labels["expected"]}
    found = _found_keys(corpus_dir)
    tp = found & expected
    fp = found - expected
    fn = expected - found
    precision = len(tp) / (len(tp) + len(fp)) if (tp or fp) else 1.0
    recall = len(tp) / (len(tp) + len(fn)) if (tp or fn) else 1.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return {
        "tp": len(tp), "fp": len(fp), "fn": len(fn),
        "precision": precision, "recall": recall, "f1": f1,
        "false_positives": sorted(fp), "false_negatives": sorted(fn),
        "labelled_findings": len(expected),
    }


def format_metrics(m):
    L = ["=" * 60, "  DETECTOR METRICS (vs labelled corpus)", "=" * 60,
         f"labelled findings: {m['labelled_findings']}",
         f"TP={m['tp']}  FP={m['fp']}  FN={m['fn']}",
         f"precision={m['precision']:.3f}  recall={m['recall']:.3f}  "
         f"F1={m['f1']:.3f}"]
    if m["false_positives"]:
        L.append("False positives:")
        for f, r, ln in m["false_positives"]:
            L.append(f"  - {r} at {f}:{ln}")
    if m["false_negatives"]:
        L.append("False negatives (labelled but missed):")
        for f, r, ln in m["false_negatives"]:
            L.append(f"  - {r} at {f}:{ln}")
    L.append("Note: rates are over this labelled corpus — a regression guard, "
             "not a claim about all code.")
    L.append("=" * 60)
    return "\n".join(L)
