import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from data.shards import LEGACY_PREFIX, UNIFORM_WINDOWS, load_manifest, sha256_file, write_shards
from model import GPT, GPTConfig
from tokenizer import CharTokenizer
from train import atomic_save
from training import control_plan, sampler_control


def setup(tmp_path):
    train = [f"def f{i}(value):\n    return value + {i}\n" * 4 for i in range(4)]
    val = [f"def g{i}(item):\n    return item * {i}\n" * 4 for i in range(4)]
    prose = ["Protocol framing uses an explicit length and rejects invalid input.\n" * 4,
             "Security checks use documented boundaries and fixed expected behavior.\n" * 4]
    tokenizer = CharTokenizer("".join(train + val + prose)).add_special_tokens()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    rows = [{"text": text, "source": "bigcode/the-stack-dedup", "group": f"stack:repo{i}"} for i, text in enumerate(val)]
    rows += [{"text": text, "source": "rfc-nist-prose", "group": "prose:shared"} for text in prose]
    raw = "".join(json.dumps(row) + "\n" for row in rows).encode()
    (corpus / "val.jsonl").write_bytes(raw)
    (corpus / "corpus-manifest.json").write_text(json.dumps({"files": {"val.jsonl": {"sha256": hashlib.sha256(raw).hexdigest()}}}))
    (corpus / "holdout_test.jsonl").write_bytes(b"\xffmust-not-read")
    data = tmp_path / "shards"
    write_shards([(text, "train") for text in train] + [(row["text"], "val") for row in rows], tokenizer,
                 data, shard_tokens=10000)
    plan = control_plan.build(data, corpus, block_size=16, code_documents=2, prose_documents=1)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan, sort_keys=True, indent=2) + "\n")
    config = GPTConfig(vocab_size=tokenizer.vocab_size, block_size=16, n_layer=1, n_head=2, n_embd=16, dropout=0.1)
    torch.manual_seed(17)
    base = tmp_path / "base" / "base.pt"
    atomic_save({"model_cfg": config.__dict__, "model_state": GPT(config).state_dict(), "tokenizer": tokenizer.to_dict(),
                 "optimizer_state": {"must_not_inherit": True}}, base)
    manifest, digest, _ = load_manifest(data)
    args = SimpleNamespace(base=str(base), expect_base_sha256=sha256_file(base), data_dir=str(data),
                           expect_data_sha256=digest, validation_plan=str(plan_path), expect_plan_sha256=sha256_file(plan_path),
                           output=str(tmp_path / "run"), steps=2, batch_size=2, accumulation=1, eval_batch_size=2,
                           learning_rate=1e-3, warmup=1, seed=1337, max_seconds=60, device="cpu")
    return args, plan, manifest, tokenizer, corpus


def test_plan_is_reproducible_paired_and_does_not_export_corpus_text(tmp_path):
    args, plan, manifest, _, corpus = setup(tmp_path)
    assert plan == control_plan.build(args.data_dir, corpus, 16, 2, 1)
    control_plan.validate(plan, manifest)
    assert len(plan["windows"]) == 6
    assert "def g" not in json.dumps(plan)
    assert plan["private_holdout_read"] is False and plan["test_split_read"] is False
    loaded, _ = control_plan.load(args.validation_plan, args.data_dir, manifest, args.expect_plan_sha256, args.expect_data_sha256)
    assert loaded == plan


def test_wrong_plan_hash_and_corpus_identity_are_rejected(tmp_path):
    args, _, manifest, _, _ = setup(tmp_path)
    with pytest.raises(ValueError, match="sha256"):
        control_plan.load(args.validation_plan, args.data_dir, manifest, "0" * 64, args.expect_data_sha256)
    with pytest.raises(ValueError, match="different corpus"):
        control_plan.load(args.validation_plan, args.data_dir, manifest, args.expect_plan_sha256, "0" * 64)


def test_plan_rejects_cross_document_windows(tmp_path):
    _, plan, manifest, _, _ = setup(tmp_path)
    plan["windows"][1]["start"] = plan["windows"][1]["doc_end"] - 1
    with pytest.raises(ValueError, match="crosses a document"):
        control_plan.validate(plan, manifest)


def test_plan_rejects_unpaired_documents(tmp_path):
    _, plan, manifest, _, _ = setup(tmp_path)
    plan["windows"][1]["group_sha256"] = "other"
    with pytest.raises(ValueError, match="different documents"):
        control_plan.validate(plan, manifest)


def test_plan_selection_fails_for_insufficient_repositories(tmp_path):
    args, _, _, _, corpus = setup(tmp_path)
    with pytest.raises(ValueError, match="not enough"):
        control_plan.build(args.data_dir, corpus, 16, 5, 1)


def test_two_arm_training_warm_starts_weights_and_resets_optimizer(tmp_path):
    args, _, _, _, _ = setup(tmp_path)
    before = Path(args.base).read_bytes()
    report = sampler_control.run(args)
    assert Path(args.base).read_bytes() == before
    control, treatment = report["arms"]["control"], report["arms"]["treatment"]
    assert control["initial_tensor_sha256"] == treatment["initial_tensor_sha256"] == report["base"]["tensor_sha256"]
    assert control["initial_optimizer_states"] == treatment["initial_optimizer_states"] == 0
    assert control["completed_steps"] == treatment["completed_steps"] == 2
    assert control["sampling_mode"] == LEGACY_PREFIX and treatment["sampling_mode"] == UNIFORM_WINDOWS
    for row in (control, treatment):
        checkpoint = torch.load(row["checkpoint"], map_location="cpu", weights_only=True)
        assert "optimizer_state" not in checkpoint
        assert checkpoint["training"]["parent_checkpoint_sha256"] == args.expect_base_sha256
        assert checkpoint["training"]["optimizer_resume_available"] is False
    assert report["assessment"]["eligible_for_capability_screen"] is False
    report["assessment"]["eligible_for_capability_screen"] = True
    experiment = Path(args.output) / "experiment.json"
    experiment.write_text(json.dumps(report))
    assert sampler_control.generate_screen(SimpleNamespace(experiment=str(experiment))) == 0
    with pytest.raises(ValueError, match="new or empty"):
        sampler_control.run(args)


def test_bad_base_hash_cannot_start_experiment(tmp_path):
    args, _, _, tokenizer, _ = setup(tmp_path)
    with pytest.raises(ValueError, match="accepted artifact"):
        sampler_control.warm_start(args.base, "0" * 64, tokenizer, "cpu", 1337)


def test_tokenizer_mismatch_cannot_warm_start(tmp_path):
    args, _, _, _, _ = setup(tmp_path)
    with pytest.raises(ValueError, match="tokenizer differs"):
        sampler_control.warm_start(args.base, args.expect_base_sha256, CharTokenizer("abc").add_special_tokens(), "cpu", 1337)


def test_protocol_cannot_exceed_step_cap(tmp_path):
    args, *_ = setup(tmp_path)
    args.steps = 201
    with pytest.raises(ValueError, match="1..200"):
        sampler_control.run(args)


def assessment_report():
    def evaluation(code, prose):
        return {"summary": {"prose_uniform": {"mean_loss_nats": prose}},
                "window_losses": [{"id": str(i), "stratum": "code_uniform", "group_sha256": f"group-{i}",
                                   "loss_nats": code} for i in range(128)]}

    common = {"completed_steps": 200, "attempted_steps": 200, "skipped_steps": 0,
              "initial_optimizer_states": 0, "initial_tensor_sha256": "same",
              "initial_cpu_rng_sha256": "cpu", "initial_device_rng_sha256": "cuda",
              "initial_batch_rng_sha256": "batch", "initial_scaler_state": {"scale": 1024}}
    return {"recipe": {"steps": 200}, "base": {"tensor_sha256": "same", "evaluation": evaluation(2.0, 2.0)},
            "arms": {"control": {**common, "sampling_mode": LEGACY_PREFIX, "train_cfg": {"sampling_mode": LEGACY_PREFIX, "seed": 1337}, "evaluation": evaluation(2.02, 2.0)},
                     "treatment": {**deepcopy(common), "sampling_mode": UNIFORM_WINDOWS, "train_cfg": {"sampling_mode": UNIFORM_WINDOWS, "seed": 1337}, "evaluation": evaluation(1.95, 2.01)}}}


def test_predeclared_assessment_requires_meaningful_gain_over_control_and_base():
    report = assessment_report()
    assert sampler_control.assess(report)["eligible_for_capability_screen"]
    for row in report["arms"]["treatment"]["evaluation"]["window_losses"]:
        row["loss_nats"] = 1.995
    assert not sampler_control.assess(report)["eligible_for_capability_screen"]


def test_prose_degradation_blocks_code_gain():
    report = assessment_report()
    report["arms"]["treatment"]["evaluation"]["summary"]["prose_uniform"]["mean_loss_nats"] = 2.1
    assert not sampler_control.assess(report)["eligible_for_capability_screen"]


@pytest.mark.parametrize("field,value", [("initial_optimizer_states", 1), ("completed_steps", 199), ("skipped_steps", 1)])
def test_unmatched_steps_or_optimizer_invalidate_comparison(field, value):
    report = assessment_report()
    report["arms"]["treatment"][field] = value
    assert not sampler_control.assess(report)["eligible_for_capability_screen"]


def test_different_initial_rng_is_not_accepted():
    report = assessment_report()
    report["arms"]["treatment"]["initial_batch_rng_sha256"] = "different"
    with pytest.raises(ValueError, match="initialization differs"):
        sampler_control.assess(report)


def test_repeated_code_repository_invalidates_bootstrap():
    report = assessment_report()
    rows = report["arms"]["treatment"]["evaluation"]["window_losses"]
    rows[1]["group_sha256"] = rows[0]["group_sha256"]
    with pytest.raises(ValueError, match="repository pairs"):
        sampler_control.assess(report)


def test_other_recipe_changes_invalidate_sampler_comparison():
    report = assessment_report()
    report["arms"]["treatment"]["train_cfg"]["seed"] = 42
    with pytest.raises(ValueError, match="recipes differ"):
        sampler_control.assess(report)
