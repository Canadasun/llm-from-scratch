import json
from dataclasses import replace

import pytest

from data import build_verified_sft as builder
from data.verified_tasks import TASKS, Task, case


@pytest.mark.parametrize("task", TASKS, ids=lambda task: task.family)
def test_authored_reference_and_targeted_defect(task):
    result = builder.validate_task(task)
    assert result["cases"] >= 4
    assert result["mutation_killed_by_cases"]


def inputs(tmp_path):
    refs = tmp_path / "references.jsonl"
    refs.write_text(json.dumps({"id": "unrelated", "text": " ".join(f"external{i}" for i in range(50))}) + "\n")
    eval_dir = tmp_path / "evals"
    eval_dir.mkdir()
    tasks = [next(task for task in TASKS if task.split == split) for split in builder.FILES]
    return refs, eval_dir, tasks


def test_build_is_reproducible_and_keeps_families_disjoint(tmp_path):
    refs, eval_dir, tasks = inputs(tmp_path)
    manifest = builder.build(tmp_path / "first", refs, eval_dir, tasks)
    assert manifest == builder.build(tmp_path / "second", refs, eval_dir, tasks)
    split_families = []
    for filename in builder.FILES.values():
        first = (tmp_path / "first" / filename).read_bytes()
        assert first == (tmp_path / "second" / filename).read_bytes()
        rows = [json.loads(line) for line in first.splitlines()]
        assert all(row["kind"] == "implement" and row["licences"] == ["MIT"] for row in rows)
        split_families.append({row["family"] for row in rows})
    assert not any(a & b for i, a in enumerate(split_families) for b in split_families[i + 1:])
    assert manifest["verification"]["targeted_mutants_killed"] == 3
    assert manifest["contamination"]["private_holdout_read"] is False


def test_incorrect_reference_never_publishes_dataset(tmp_path):
    refs, eval_dir, tasks = inputs(tmp_path)
    tasks[0] = replace(tasks[0], source=tasks[0].source.replace(".lower()", ""))
    with pytest.raises(ValueError, match="reference failed"):
        builder.build(tmp_path / "out", refs, eval_dir, tasks)
    assert not (tmp_path / "out").exists()


def test_ineffective_test_suite_rejected():
    task = Task("missed-edge", "train", "test", "def f(x):\n    return x + 1\n",
                [case([i], i + 1) for i in range(4)], ("x + 1", "x + 1 if x >= 0 else 0"))
    with pytest.raises(ValueError, match="survived every test"):
        builder.validate_task(task)


def test_mutation_of_inputs_rejected():
    source = "def f(values):\n    values.append(1)\n    return values\n"
    task = Task("mutates", "train", "test", source,
                [case([[i]], [i, 1]) for i in range(4)], ("append(1)", "append(2)"))
    with pytest.raises(ValueError, match="reference failed"):
        builder.validate_task(task)


def test_planted_public_benchmark_overlap_blocks_build(tmp_path):
    refs, eval_dir, tasks = inputs(tmp_path)
    refs.write_text(json.dumps({"id": "planted", "text": tasks[0].spec + "\n" + tasks[0].source}) + "\n")
    with pytest.raises(ValueError, match="overlaps a public benchmark"):
        builder.build(tmp_path / "out", refs, eval_dir, tasks)


def test_duplicate_family_across_splits_rejected(tmp_path):
    refs, eval_dir, tasks = inputs(tmp_path)
    tasks[1] = replace(tasks[1], family=tasks[0].family)
    with pytest.raises(ValueError, match="duplicate task family"):
        builder.build(tmp_path / "out", refs, eval_dir, tasks)


def test_missing_benchmark_references_fail_closed(tmp_path):
    refs, eval_dir, tasks = inputs(tmp_path)
    with pytest.raises(ValueError, match="references"):
        builder.build(tmp_path / "out", refs.with_name("missing"), eval_dir, tasks)


def test_nested_evaluation_overlap_blocks_build(tmp_path):
    refs, eval_dir, tasks = inputs(tmp_path)
    nested = eval_dir / "preview"
    nested.mkdir()
    (nested / "behaviour.jsonl").write_text(json.dumps({"id": "planted", "prompt": tasks[0].spec}) + "\n")
    with pytest.raises(ValueError, match="overlaps a public benchmark"):
        builder.build(tmp_path / "out", refs, eval_dir, tasks)


def test_empty_public_references_fail_closed(tmp_path):
    refs, eval_dir, tasks = inputs(tmp_path)
    refs.write_text("")
    with pytest.raises(ValueError, match="no eligible benchmark texts"):
        builder.build(tmp_path / "out", refs, eval_dir, tasks)
