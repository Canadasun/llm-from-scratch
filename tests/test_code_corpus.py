"""
Tests for the license-safe Python code corpus and the train/eval contamination
guard (Part 1 of the GPU-training-run data work).
"""

import os

import pytest

from data import build_code_examples
from data.build_code_corpus import build
from data.contamination import find_contamination, assert_clean

EVAL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "evals", "data")


# ---- self-authored examples ----

def test_examples_cover_secure_and_refusal():
    names = [n for n, _ in build_code_examples.examples()]
    assert any(n.startswith("secure-") for n in names)
    assert any(n.startswith("refusal-") for n in names)
    text = build_code_examples.render()
    assert "parameterised query" in text.lower() or "parameterized" in text.lower()
    assert "A:" in text          # refusal turns present


# ---- contamination guard ----

def test_contamination_guard_catches_planted_eval_text():
    # Build a corpus that deliberately embeds a real eval string -> must be caught.
    import json
    sample = None
    with open(os.path.join(EVAL_DIR, "security_mc.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                sample = json.loads(line)["prompt"]
                break
    assert sample and len(sample) >= 40
    dirty = "def f():\n    pass\n" + sample + "\nmore code\n"
    hits = find_contamination(dirty, EVAL_DIR)
    assert hits, "planted eval prompt was not detected"
    with pytest.raises(ValueError, match="contamination"):
        assert_clean(dirty, EVAL_DIR)


def test_clean_corpus_passes_guard():
    assert find_contamination("def add(a, b):\n    return a + b\n", EVAL_DIR) == []


# ---- builder ----

def test_build_is_deterministic_and_clean():
    corpus1, m1 = build(max_bytes=300_000)
    corpus2, m2 = build(max_bytes=300_000)
    assert m1["sha256"] == m2["sha256"]          # deterministic for a given cap
    assert corpus1 == corpus2
    assert m1["sources"][1]["license"] == "PSF License"
    assert m1["sources"][0]["blocks"] >= 1       # examples included
    assert m1["bytes"] <= 300_000
    # build() runs assert_clean internally; it returned, so the corpus is clean.
    assert_clean(corpus1, EVAL_DIR)


def test_build_includes_examples_and_stdlib():
    corpus, m = build(max_bytes=2_000_000)
    assert "self-authored secure/insecure/refusal examples" in corpus
    assert "CPython standard library (PSF License)" in corpus
    assert m["sources"][1]["files"] >= 1
    assert m["policy"].startswith("public or self-authored")


# ---- Part 2: gpu_xl preset + prose+code training mixer ----

from config import PRESETS
from data.build_training_corpus import build_training_corpus


def test_gpu_xl_preset_is_longer_context_and_larger():
    c = PRESETS["gpu_xl"]()
    assert c.block_size == 1024            # doubled context for code
    assert c.vocab_size == 4096
    assert c.n_layer >= 12 and c.n_embd >= 768
    assert c.tokenizer == "bpe"


def test_training_mixer_includes_code_and_is_eval_clean(tmp_path):
    corpus, info = build_training_corpus(prose_path=None, code_max_bytes=400_000)
    assert "Python code corpus" in corpus
    assert info["code_bytes"] > 0 and info["prose_bytes"] == 0   # no prose given
    # build_training_corpus runs assert_clean internally; re-affirm here.
    assert_clean(corpus, EVAL_DIR)


def test_training_mixer_combines_prose_and_code(tmp_path):
    prose = tmp_path / "prose.txt"
    prose.write_text("# domain prose\nTLS is a protocol for secure transport.\n" * 50,
                     encoding="utf-8")
    corpus, info = build_training_corpus(prose_path=str(prose), code_max_bytes=400_000)
    assert "domain prose (RFCs + NIST)" in corpus and "Python code corpus" in corpus
    assert info["prose_bytes"] > 0 and info["code_bytes"] > 0
