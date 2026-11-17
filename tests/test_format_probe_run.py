import copy
import hashlib
import json
from pathlib import Path

import pytest

from evals import format_probe_run as run
from evals.format_probe import FORMATS, build, prompt
from data.format_probe_tasks import TASKS
from special_tokens import ASSISTANT, ENDOFTEXT, USER
from tokenization import from_dict


def fixture():
    state = json.loads(Path("evals/results/format_probe/tokenizer.json").read_text())
    tokenizer = from_dict(state)
    stops = sorted(tokenizer.special_id(t) for t in (ENDOFTEXT, USER, ASSISTANT))
    saved = {"schema": "format_probe_responses_v1", "meta": {
        "checkpoint_sha256": run.BASE_SHA, "probe_sha256": run.digest(build()),
        "source_sha256": build()["source_sha256"], "max_new_tokens": 384,
        "context": 1024, "vocab_size": tokenizer.vocab_size,
        "decoding": "temperature_1_top_k_1", "seed_per_response": 1337,
        "wrapper": "user_prompt_assistant", "stop_ids": stops,
        "precision": "float32", "backend": "deterministic_math_sdpa", "truncation": "forbidden"},
        "responses": {}}
    for task in TASKS:
        for mode in FORMATS:
            text = f"{USER}{prompt(task, mode)}{ASSISTANT}"
            response = task.header + task.body if mode == "module" else task.body
            ids = tokenizer.encode(response) + [stops[0]]
            saved["responses"][f"{task.family}:{mode}"] = {
                "response": response, "token_ids": ids,
                "prompt_tokens": len(tokenizer.encode(text, allowed_special="all")),
                "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
                **run.finish_tokens(ids, stops, 384)}
    return saved, state


def test_roundtrip_and_stop_reason_precedence():
    saved, state = fixture()
    run.validate(saved, state)
    assert run.finish_tokens([1, 2], [3], 2)["stop_reason"] == "token_budget"
    assert run.finish_tokens([1, 3], [3], 2)["stop_reason"] == "special_token"
    for ids in ([], [3, 1], [1]):
        with pytest.raises(ValueError):
            run.finish_tokens(ids, [3], 2)


@pytest.mark.parametrize("field,value", [("response", "changed"), ("prompt_tokens", 1),
                                        ("stop_reason", "token_budget"), ("prompt_sha256", "wrong")])
def test_tampered_rows_fail_before_execution(field, value, monkeypatch):
    saved, state = fixture()
    next(iter(saved["responses"].values()))[field] = value
    monkeypatch.setattr(run, "score_response", lambda *a: pytest.fail("must not execute"))
    with pytest.raises(ValueError):
        run.score(saved, state)


def test_partial_wrong_checkpoint_or_tokenizer_rejected():
    saved, state = fixture()
    partial = copy.deepcopy(saved)
    partial["responses"].pop(next(iter(partial["responses"])))
    with pytest.raises(ValueError, match="membership"):
        run.validate(partial, state)
    saved["meta"]["checkpoint_sha256"] = "wrong"
    with pytest.raises(ValueError, match="checkpoint"):
        run.validate(saved, state)
    with pytest.raises(ValueError, match="tokenizer"):
        run.validate(saved, {})


def test_summary_keeps_formats_and_splits_separate(monkeypatch):
    saved, state = fixture()
    monkeypatch.setattr(run, "score_response", lambda *a: {
        "syntax": True, "format": True, "functional": False, "status": "failed"})
    result = run.score(saved, state)
    assert len(result["summary"]) == 4
    assert all(row == {"tasks": 6, "syntax": 6, "format": 6, "functional": 0}
               for row in result["summary"].values())


def test_reference_report_preserves_mode_and_boolean_format(monkeypatch):
    from evals import format_probe as probe
    from evals.code_executor import Verdict
    monkeypatch.setattr(probe, "TASKS", TASKS[:1])
    monkeypatch.setattr(probe, "sandbox_preflight", lambda: {"available": True})
    monkeypatch.setattr(probe, "evaluate_cases", lambda source, *a, **k:
                        Verdict("failed", False) if "return None" in source else Verdict("passed", True))
    result = probe.verify()
    assert {row["mode"] for row in result["results"]} == {"module", "body"}
    assert all(type(row["format"]) is bool for row in result["results"])
