import copy
import json

import pytest

from evals.bench_run import humaneval_program
from evals.syntax_diagnosis import diagnose, report, syntax


def generated(completion, program, chat=False, suite="humaneval"):
    return {"meta": {"decoding": {"chat": chat}, "checkpoint_sha256": "a" * 64},
            "programs": {f"{suite}:example": {"completion": completion, "program": program}}}


def test_classification_and_no_execution(tmp_path):
    target = tmp_path / "must-not-exist"
    assert syntax(f"open({str(target)!r}, 'w').write('bad')")["parses"]
    assert not target.exists()
    assert syntax("  return 2")["category"] == "indentation"
    assert syntax("x = [1,")["category"] == "unclosed_construct"
    assert syntax("def :")["category"] == "other_syntax"
    assert syntax("")["empty_module"]


def test_humaneval_extraction_can_cut_a_valid_multiline_string():
    prompt = "def example():\n"
    completion = '    text = """hello\n# literal text\n"""\n    return text\n'
    value = generated(completion, humaneval_program(prompt, completion))
    before = copy.deepcopy(value)
    result = diagnose(value)
    assert result["extraction_valid_to_invalid"] == 1
    assert result["rows"]["humaneval:example"]["category"] == "unclosed_construct"
    assert value == before


def test_empty_cut_keeps_prompt_and_verifies_extractor():
    completion = "\ndef other():\n    pass\n"
    result = diagnose(generated(completion, "def example():\n    pass\n"))
    assert result["rows"]["humaneval:example"]["uncut_parses"]
    with pytest.raises(ValueError, match="disagrees"):
        diagnose(generated("    return 1\n", "different"))


def test_chat_fence_and_mismatched_program():
    completion = "Here is code:\n```python\ndef example():\n    return 1\n```"
    result = diagnose(generated(completion, "def example():\n    return 1\n", chat=True))
    assert result["extraction_invalid_to_valid"] == 1
    with pytest.raises(ValueError, match="disagrees"):
        diagnose(generated(completion, "pass\n", chat=True))


def test_report_pairs_tasks_and_binds_input_bytes(tmp_path):
    for label in ("base", "control", "treatment"):
        for mode in ("completion", "chat"):
            value = generated("pass\n", "pass\n", chat=mode == "chat")
            (tmp_path / f"programs-{label}-{mode}.json").write_text(json.dumps(value))
    result = report(tmp_path)
    assert len(result["input_sha256"]) == 6
    assert result["transitions_vs_base"]["treatment-chat"]["valid_to_invalid"] == []
    path = tmp_path / "programs-treatment-chat.json"
    value["programs"] = {}
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="membership"):
        report(tmp_path)
