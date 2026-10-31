"""
Tests for the evaluation harness.

These check the *machinery* (scoring maths, scorers, the sandbox, task running
and report aggregation) on a tiny model, plus validity of the hand-authored task
data. They do not assert the trained model hits any accuracy — that is measured,
not unit-tested.
"""

import glob
import json
import os

import pytest
import torch

from model import GPT, GPTConfig
from tokenizer import CharTokenizer

from evals import scorers
from evals.scoring import (choose_by_likelihood, completion_nll,
                           corpus_metrics, generate_text)
from evals.tasks import load_tasks, run_task
from evals import report as report_mod

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "evals", "data")


@pytest.fixture(scope="module")
def tiny():
    text = "hello world. secure code and tests. the quick brown fox.\n" * 20
    tok = CharTokenizer.from_text(text)
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=32,
                    n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    torch.manual_seed(0)
    model = GPT(cfg).eval()
    return model, tok


# --- scoring maths -------------------------------------------------------

def test_completion_nll_is_finite_and_positive(tiny):
    model, tok = tiny
    nll, n = completion_nll(model, tok, "hello ", "world", device="cpu")
    assert n == len("world")
    assert nll > 0 and nll != float("inf")


def test_choose_by_likelihood_returns_valid_index(tiny):
    model, tok = tiny
    best, means = choose_by_likelihood(model, tok, "hello ", ["world", "xqzkj"],
                                       device="cpu")
    assert best in (0, 1)
    assert len(means) == 2
    assert all(m > 0 for m in means)


def test_corpus_metrics_has_expected_keys(tiny):
    model, tok = tiny
    m = corpus_metrics(model, tok, "secure code and tests.\n" * 5, device="cpu")
    for k in ("perplexity", "bits_per_byte", "top1_acc", "top5_acc", "tokens"):
        assert k in m
    assert m["tokens"] > 0
    assert 0.0 <= m["top1_acc"] <= 1.0
    assert m["perplexity"] > 0


def test_generate_returns_only_continuation(tiny):
    model, tok = tiny
    out = generate_text(model, tok, "hello ", max_new_tokens=5, device="cpu")
    assert isinstance(out, str)
    assert len(out) >= 1


# --- scorers -------------------------------------------------------------

def test_text_scorers():
    assert scorers.exact_match("  yes ", "yes")
    assert scorers.contains("the Password field", "password")
    assert scorers.regex_match("TLS 1.3 here", r"1\.3")
    assert scorers.contains_any("use encryption", ["encrypt", "hash"])
    assert not scorers.contains("nope", "password")


def test_extract_code_prefers_fenced_block():
    text = "sure:\n```python\nx = 1\n```\ndone"
    assert scorers.extract_code(text) == "x = 1"
    assert scorers.extract_code("x = 2") == "x = 2"


# --- sandboxed code execution -------------------------------------------

CORRECT = "def add(a, b):\n    return a + b\n"
WRONG = "def add(a, b):\n    return a - b\n"
TEST = ("import unittest\nfrom solution import add\n"
        "class T(unittest.TestCase):\n"
        "    def test(self):\n        self.assertEqual(add(2, 3), 5)\n"
        "if __name__ == '__main__':\n    unittest.main()\n")


def test_sandbox_passes_correct_code():
    ok, _ = scorers.run_python_unittest(CORRECT, TEST)
    assert ok is True


def test_sandbox_fails_wrong_code():
    ok, _ = scorers.run_python_unittest(WRONG, TEST)
    assert ok is False


def test_sandbox_times_out_on_infinite_loop():
    ok, detail = scorers.run_python_unittest(
        "def add(a, b):\n    while True:\n        pass\n", TEST, timeout=3)
    assert ok is False


# --- task running and report --------------------------------------------

def test_run_mc_task(tiny):
    model, tok = tiny
    task = {"id": "t", "category": "x", "type": "multiple_choice",
            "prompt": "hello ", "choices": ["world", "zzz"], "answer": 0}
    r = run_task(model, tok, task, device="cpu")
    assert r["type"] == "multiple_choice"
    assert isinstance(r["correct"], bool)
    assert r["n_choices"] == 2


def test_summarize_and_report_run():
    results = [
        {"id": "1", "category": "a", "type": "multiple_choice", "correct": True,
         "n_choices": 4, "detail": ""},
        {"id": "2", "category": "a", "type": "multiple_choice", "correct": False,
         "n_choices": 4, "detail": ""},
        {"id": "3", "category": "b", "type": "code", "correct": False, "detail": "x"},
    ]
    s = report_mod.summarize(results)
    assert s["overall"]["n"] == 3
    assert s["overall"]["correct"] == 1
    assert abs(s["by_category"]["a"]["acc"] - 0.5) < 1e-9
    assert abs(s["mc_random_baseline"] - 0.25) < 1e-9
    text = report_mod.format_report(s, results)
    assert "EVALUATION REPORT" in text
    assert "random-guess baseline" in text


# --- authored data validity (catches data-entry mistakes) ---------------

def test_all_task_files_parse_and_are_valid():
    files = glob.glob(os.path.join(DATA_DIR, "*.jsonl"))
    assert files, "no task files found"
    tasks = load_tasks(files)
    assert len(tasks) >= 50, "expected a meaningful starter suite"
    ids = set()
    for t in tasks:
        assert t["id"] not in ids, f"duplicate id {t['id']}"
        ids.add(t["id"])
        assert "category" in t and "type" in t
        if t["type"] == "multiple_choice":
            assert len(t["choices"]) >= 2
            assert 0 <= t["answer"] < len(t["choices"]), f"bad answer in {t['id']}"
        elif t["type"] == "generation":
            assert "target" in t and "prompt" in t
        elif t["type"] == "code":
            assert "test" in t and "solution" in t["test"]
        else:
            raise AssertionError(f"unknown type {t['type']} in {t['id']}")


def test_sandbox_rejects_early_exit_zero():
    # Integrity (scoring can't be faked): a solution that exits 0 before the
    # tests actually run must NOT score a pass.
    evil = "import os\ndef add(a, b):\n    return a + b\nos._exit(0)\n"
    ok, _ = scorers.run_python_unittest(evil, TEST)
    assert ok is False
