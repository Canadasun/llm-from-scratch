import json
from dataclasses import replace

import pytest

from data.build_verified_sft import FILES, build, instruction_variants, validate_task
from data.curriculum_tasks import TASKS
from data.verified_tasks import TASKS as SEED_TASKS
from training.failure_analysis import analyze


@pytest.mark.parametrize("task", TASKS, ids=lambda task: task.family)
def test_new_reference_and_targeted_bug(task):
    result = validate_task(task)
    assert result["cases"] >= 4 and result["mutation_killed_by_cases"]


def resources(tmp_path):
    refs = tmp_path / "public.jsonl"
    refs.write_text(json.dumps({"id": "external", "text": " ".join(f"unrelated{i}" for i in range(60))}) + "\n")
    eval_dir = tmp_path / "evals"
    eval_dir.mkdir()
    tasks = [next(task for task in TASKS if task.split == split) for split in FILES]
    return refs, eval_dir, tasks


def test_curriculum_keeps_repair_and_correct_variants_together(tmp_path):
    refs, eval_dir, tasks = resources(tmp_path)
    manifest = build(tmp_path / "out", refs, eval_dir, tasks, include_repairs=True)
    assert manifest["name"] == "llmfs-verified-curriculum-v2"
    assert manifest["rows"] == {"train": 5, "val": 5, "test": 5}
    families = []
    for split, filename in FILES.items():
        rows = [json.loads(line) for line in (tmp_path / "out" / filename).read_text().splitlines()]
        assert {row["kind"] for row in rows} == {"implement", "repair", "preserve-correct"}
        assert len({row["response"] for row in rows}) == 1
        assert all(row["source"] == "data/curriculum_tasks.py" for row in rows)
        families.append({row["family"] for row in rows})
    assert not any(a & b for i, a in enumerate(families) for b in families[i + 1:])


def test_seed_splits_remain_unchanged():
    combined = SEED_TASKS + TASKS
    assert len({task.family for task in combined}) == len(combined)
    assert all(combined[i] is task for i, task in enumerate(SEED_TASKS))


def test_repair_prompt_is_checked_for_contamination(tmp_path):
    refs, eval_dir, tasks = resources(tmp_path)
    prompt = next(prompt for kind, _, prompt in instruction_variants(tasks[0], True) if kind == "repair")
    (eval_dir / "planted.jsonl").write_text(json.dumps({"id": "planted", "prompt": prompt}) + "\n")
    with pytest.raises(ValueError, match="overlaps a public benchmark"):
        build(tmp_path / "out", refs, eval_dir, tasks, include_repairs=True)
    assert not (tmp_path / "out").exists()


def test_renaming_a_solution_cannot_move_it_to_another_split(tmp_path):
    refs, eval_dir, tasks = resources(tmp_path)
    original = tasks[0]
    from data.build_verified_sft import function_name
    renamed = original.source.replace(function_name(original.source), "renamed_function", 1)
    tasks[1] = replace(original, family="different-label", split="validation", source=renamed)
    with pytest.raises(ValueError, match="structure overlaps splits"):
        build(tmp_path / "out", refs, eval_dir, tasks, include_repairs=True)


def test_nonbuiltin_provenance_rejected(tmp_path):
    refs, eval_dir, tasks = resources(tmp_path)
    tasks[0] = replace(tasks[0], source_file="/tmp/downloaded.py")
    with pytest.raises(ValueError, match="built-in project-authored"):
        build(tmp_path / "out", refs, eval_dir, tasks, include_repairs=True)


def test_failure_analysis_never_executes_programs_or_emits_solutions(tmp_path):
    marker = tmp_path / "must-not-exist"
    payload = f'open({str(marker)!r}, "w").write("benchmark-answer-secret")'
    meta = {"checkpoint_sha256": "a" * 64}
    (tmp_path / "programs-example.json").write_text(json.dumps({"meta": meta, "programs": {
        "one": {"program": payload}, "two": {"program": "def f(\n"}, "three": {"program": ""}}}))
    (tmp_path / "scores-example.json").write_text(json.dumps({"meta": meta, "results": [
        {"status": "failed", "detail": "case 1 output mismatch"},
        {"status": "failed", "detail": "worker exited unsuccessfully"}]}))
    report = analyze(tmp_path)
    assert not marker.exists()
    assert "benchmark-answer-secret" not in json.dumps(report)
    stats = report["reports"]["example"]
    assert stats["static_shape_counts"]["empty_module"] == 1
    assert stats["outcomes"]["worker_exit_cause_not_established"] == 1


def test_failure_analysis_rejects_wrong_checkpoint_pair(tmp_path):
    (tmp_path / "programs-example.json").write_text(json.dumps({"meta": {"checkpoint_sha256": "a"}}))
    (tmp_path / "scores-example.json").write_text(json.dumps({"meta": {"checkpoint_sha256": "b"}}))
    with pytest.raises(ValueError, match="identity mismatch"):
        analyze(tmp_path)
