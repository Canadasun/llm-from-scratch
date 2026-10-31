"""
Tests for detector metrics (codereview/metrics.py).

These assert the detector scores perfectly against its FROZEN labelled corpus —
so any future rule change that introduces a false positive on the safe fixture or
misses a labelled issue on the vulnerable fixture turns the build red.
"""

import json

from codereview.metrics import (CORPUS_DIR, LABELS_PATH, compute_metrics,
                                format_metrics, generate_labels)


def test_labels_file_matches_current_detection():
    # The frozen labels must equal what generate_labels produces now; if a rule
    # change alters detection, this flags that the labels need review.
    with open(LABELS_PATH, encoding="utf-8") as f:
        frozen = json.load(f)
    live = generate_labels(CORPUS_DIR)
    assert frozen["expected"] == sorted(
        live["expected"], key=lambda e: (e["file"], e["rule_id"], e["line"]))
    assert frozen["clean"] == live["clean"]


def test_precision_recall_are_perfect_on_labelled_corpus():
    m = compute_metrics()
    assert m["fp"] == 0, f"false positives: {m['false_positives']}"
    assert m["fn"] == 0, f"false negatives: {m['false_negatives']}"
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["labelled_findings"] == 16


def test_format_metrics_mentions_scope_caveat():
    text = format_metrics(compute_metrics())
    assert "precision" in text and "recall" in text
    assert "not a claim about all code" in text
