import json

import pytest
import torch

import tokenization
from data.shards import manifest_sha256, write_shards
from model import GPT, GPTConfig
from special_tokens import ASSISTANT, USER
from train import atomic_save
from training import sft

TEXT = "".join(f"def f{i}(x):\n    return x + {i}\n\n" for i in range(60))


def _setup(tmp_path, dropout=0.0):
    tok = tokenization.build_with_specials("bpe_code", TEXT * 2, vocab_size=300, bpe_train_chars=10 ** 6)
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=64, n_layer=2, n_head=2, n_embd=32, dropout=dropout)
    base = tmp_path / "base" / "ckpt.pt"
    atomic_save({"model_state": GPT(cfg).state_dict(), "model_cfg": cfg.__dict__, "tokenizer": tok.to_dict()}, base)
    data = tmp_path / "sft"
    data.mkdir()
    kinds = ["implement", "refusal", "secure-authored", "summarise"]
    for name, n, offset in (("train", 24, 0), ("validation", 6, 100)):
        rows = [{"id": f"{name}{i}", "family": f"fam{offset + i}", "kind": kinds[i % 4],
                 "prompt": f"Write f{i}.", "response": f"def f{i}(x):\n    return x + {i}\n"} for i in range(n)]
        rows.append({"id": f"{name}-long", "family": f"long{offset}", "kind": "implement",
                     "prompt": "Explain " + "x " * 200, "response": "no"})
        (data / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (data / "manifest.json").write_text(json.dumps({"schema": 1}))
    shards = tmp_path / "shards"
    docs = [(f"def g{i}(y):\n    return y * {i}\n", "val" if i % 8 == 0 else "train") for i in range(80)]
    write_shards(docs, tok, shards, shard_tokens=500)
    return tok, base, data, shards


def _args(base, data, shards, out, **extra):
    argv = ["--base", str(base), "--data", str(data), "--replay-dir", str(shards), "--replay-fraction", "0.3",
            "--output", str(out), "--steps", "6", "--batch-size", "2", "--grad-accum-steps", "2",
            "--warmup", "2", "--context", "64", "--eval-interval", "3", "--device", "cpu",
            "--learning-rate", "1e-3"]
    for key, value in extra.items():
        argv += [f"--{key.replace('_', '-')}", str(value)]
    return sft.build_parser().parse_args(argv)


def test_loss_covers_only_the_response(tmp_path):
    tok, *_ = _setup(tmp_path)
    x, y = sft.encode_row({"prompt": "Write f1.", "response": "return 1"}, tok, 64)
    assistant = tok.special_id(ASSISTANT)
    cut = x.tolist().index(assistant)
    assert x[0].item() == tok.special_id(USER)
    assert (y[:cut] == -100).all() and (y[cut:] != -100).all()
    assert tok.decode(y[cut:].tolist(), skip_special=True) == "return 1"
    assert sft.encode_row({"prompt": "a " * 100, "response": "b"}, tok, 64) is None


def test_upsampling_repeats_authored_rows():
    rows = [{"kind": "implement"}, {"kind": "refusal"}, {"kind": "secure-authored"}]
    order = sft.schedule(rows, sft.parse_upsample("refusal=3,secure-authored=2"))
    assert sorted(order) == [0, 1, 1, 1, 2, 2]


def test_tuning_writes_candidate_and_never_touches_base(tmp_path):
    _, base, data, shards = _setup(tmp_path)
    before = base.read_bytes()
    result = sft.tune(_args(base, data, shards, tmp_path / "run"))
    assert result["state"] == "completed" and result["step"] == 6
    status = json.loads((tmp_path / "run" / "status.json").read_text())
    assert status["meta"]["train_rows_dropped_for_context"] == 1
    assert status["meta"]["replay_manifest_sha256"] == manifest_sha256(shards)
    assert status["meta"]["replay_sampling_mode"] == "uniform_windows_v1"
    assert (tmp_path / "run" / "best.pt").exists() and "best_sha256" in status
    assert "replay_val_loss" in json.loads((tmp_path / "run" / "train_log.jsonl").read_text().splitlines()[0])
    assert base.read_bytes() == before
    with pytest.raises(ValueError):
        sft.tune(_args(base, data, shards, base.parent))
    with pytest.raises(SystemExit):
        sft.tune(_args(base, data, shards, tmp_path / "pinned", expect_base_sha256="0" * 64))


@pytest.mark.parametrize("dropout", [0.0, 0.1])
def test_resumed_tuning_matches_continuous(tmp_path, dropout):
    _, base, data, shards = _setup(tmp_path, dropout=dropout)
    sft.tune(_args(base, data, shards, tmp_path / "full"))
    first = sft.tune(_args(base, data, shards, tmp_path / "a", stop_at_step=3))
    assert first["state"] == "stopped" and first["step"] == 3
    sft.tune(_args(base, data, shards, tmp_path / "b", resume=tmp_path / "a" / "ckpt.pt"))
    full = torch.load(tmp_path / "full" / "ckpt.pt", weights_only=False)
    resumed = torch.load(tmp_path / "b" / "ckpt.pt", weights_only=False)
    assert resumed["step"] == full["step"] == 6
    for key, value in full["model_state"].items():
        assert torch.equal(value, resumed["model_state"][key]), key


def test_resume_rejects_changed_training_settings(tmp_path):
    _, base, data, shards = _setup(tmp_path)
    sft.tune(_args(base, data, shards, tmp_path / "first", stop_at_step=3))
    with pytest.raises(ValueError, match="resume configuration mismatch: learning_rate"):
        sft.tune(_args(base, data, shards, tmp_path / "resumed",
                       resume=tmp_path / "first" / "ckpt.pt", learning_rate="2e-3"))


def test_manifest_detects_modified_instruction_file(tmp_path):
    _, base, data, shards = _setup(tmp_path)
    (data / "manifest.json").write_text(json.dumps({"sha256": {"train": "0" * 64}}))
    with pytest.raises(ValueError, match="train split does not match"):
        sft.tune(_args(base, data, shards, tmp_path / "run"))


def test_preversion_sft_resume_preserves_replay_distribution(tmp_path, monkeypatch):
    from data.shards import LEGACY_PREFIX, UNIFORM_WINDOWS
    original = sft.ShardSplit.sample

    def checked(self, *args, **kwargs):
        assert kwargs["sampling_mode"] == LEGACY_PREFIX
        return original(self, *args, **kwargs)

    monkeypatch.setattr(sft.ShardSplit, "sample", checked)
    _, base, data, shards = _setup(tmp_path, dropout=0.1)
    sft.tune(_args(base, data, shards, tmp_path / "full", replay_sampling_mode=LEGACY_PREFIX))
    sft.tune(_args(base, data, shards, tmp_path / "first", replay_sampling_mode=LEGACY_PREFIX, stop_at_step=3))
    state = torch.load(tmp_path / "first" / "ckpt.pt", weights_only=True)
    state["meta"].pop("replay_sampling_mode")
    torch.save(state, tmp_path / "preversion.pt")
    sft.tune(_args(base, data, shards, tmp_path / "resumed", resume=tmp_path / "preversion.pt"))
    a = torch.load(tmp_path / "full" / "ckpt.pt", weights_only=True)
    b = torch.load(tmp_path / "resumed" / "ckpt.pt", weights_only=True)
    assert b["meta"]["replay_sampling_mode"] == LEGACY_PREFIX
    assert all(torch.equal(value, b["model_state"][key]) for key, value in a["model_state"].items())
    with pytest.raises(ValueError, match="replay_sampling_mode"):
        sft.tune(_args(base, data, shards, tmp_path / "changed", resume=tmp_path / "preversion.pt",
                       replay_sampling_mode=UNIFORM_WINDOWS))


@pytest.mark.parametrize("option,value", [("replay_fraction", 1), ("replay_fraction", -0.1),
                                         ("grad_accum_steps", 0), ("eval_interval", 0)])
def test_invalid_training_settings_fail_before_loading(option, value, tmp_path):
    args = _args(tmp_path / "missing", tmp_path, tmp_path, tmp_path / "run", **{option: value})
    with pytest.raises(ValueError):
        sft.tune(args)
