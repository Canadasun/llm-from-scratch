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


def _setup(tmp_path):
    tok = tokenization.build_with_specials("bpe_code", TEXT * 2, vocab_size=300, bpe_train_chars=10 ** 6)
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=64, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
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
    assert (tmp_path / "run" / "best.pt").exists() and "best_sha256" in status
    assert "replay_val_loss" in json.loads((tmp_path / "run" / "train_log.jsonl").read_text().splitlines()[0])
    assert base.read_bytes() == before
    with pytest.raises(ValueError):
        sft.tune(_args(base, data, shards, base.parent))
    with pytest.raises(SystemExit):
        sft.tune(_args(base, data, shards, tmp_path / "pinned", expect_base_sha256="0" * 64))


def test_resumed_tuning_matches_continuous(tmp_path):
    _, base, data, shards = _setup(tmp_path)
    sft.tune(_args(base, data, shards, tmp_path / "full"))
    first = sft.tune(_args(base, data, shards, tmp_path / "a", stop_at_step=3))
    assert first["state"] == "stopped" and first["step"] == 3
    sft.tune(_args(base, data, shards, tmp_path / "b", resume=tmp_path / "a" / "ckpt.pt"))
    full = torch.load(tmp_path / "full" / "ckpt.pt", weights_only=False)
    resumed = torch.load(tmp_path / "b" / "ckpt.pt", weights_only=False)
    assert resumed["step"] == full["step"] == 6
    for key, value in full["model_state"].items():
        assert torch.equal(value, resumed["model_state"][key]), key
