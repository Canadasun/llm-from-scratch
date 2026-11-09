import json
import math
from pathlib import Path

import pytest
import torch

import tokenization
from config import TrainConfig
from model import GPT, GPTConfig, KVCache, set_attention_backend
from special_tokens import ASSISTANT, ENDOFTEXT, SPECIAL_TOKENS, USER
from tokenizer import CharTokenizer
from tokenizer_bpe import BPETokenizer


def tiny_model(vocab=40, block=24, layers=2, heads=2, embd=16, seed=0):
    torch.manual_seed(seed)
    model = GPT(GPTConfig(vocab_size=vocab, block_size=block, n_layer=layers, n_head=heads,
                          n_embd=embd, dropout=0.0))
    return model.eval()


@pytest.fixture
def manual_backend():
    previous = set_attention_backend("manual")
    yield
    set_attention_backend(previous)


def test_sdpa_and_manual_attention_agree():
    model = tiny_model()
    idx = torch.randint(40, (3, 24), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        sdpa, _ = model(idx)
        previous = set_attention_backend("manual")
        try:
            manual, _ = model(idx)
        finally:
            set_attention_backend(previous)
    assert torch.allclose(sdpa, manual, atol=1e-5)


@pytest.mark.parametrize("backend", ["sdpa", "manual"])
def test_cached_logits_match_full_forward(backend):
    previous = set_attention_backend(backend)
    try:
        model = tiny_model()
        idx = torch.randint(40, (2, 20), generator=torch.Generator().manual_seed(2))
        with torch.no_grad():
            full, _ = model(idx)
            cache = KVCache(len(model.blocks))
            first, _ = model(idx[:, :7], cache=cache)
            middle, _ = model(idx[:, 7:12], cache=cache)
            steps = [model(idx[:, i:i + 1], cache=cache)[0] for i in range(12, 20)]
        stitched = torch.cat([first, middle, *steps], dim=1)
        assert cache.length == 20
        assert torch.allclose(stitched, full, atol=1e-5)
    finally:
        set_attention_backend(previous)


@pytest.mark.parametrize("prompt_len,new_tokens", [(3, 10), (5, 40)])
def test_cached_and_uncached_generation_match(prompt_len, new_tokens):
    model = tiny_model()
    idx = torch.randint(40, (2, prompt_len), generator=torch.Generator().manual_seed(3))
    greedy_cached = model.generate(idx, new_tokens, top_k=1, use_cache=True)
    greedy_plain = model.generate(idx, new_tokens, top_k=1, use_cache=False)
    assert torch.equal(greedy_cached, greedy_plain)
    sampled_cached = model.generate(idx, new_tokens, generator=torch.Generator().manual_seed(9))
    sampled_plain = model.generate(idx, new_tokens, generator=torch.Generator().manual_seed(9), use_cache=False)
    assert torch.equal(sampled_cached, sampled_plain)
    assert greedy_cached.shape == (2, prompt_len + new_tokens)


def test_generation_stops_at_stop_token_and_validates_budget():
    model = tiny_model()
    idx = torch.tensor([[1, 2, 3]])
    out = model.generate(idx, 50, stop_ids=set(range(40)))
    assert out.shape == (1, 4)
    assert model.generate(idx, 0).shape == (1, 3)
    with pytest.raises(ValueError):
        model.generate(idx, -1)
    with pytest.raises(ValueError):
        model.generate(idx, 2.5)


def test_mask_is_not_persisted_and_legacy_weights_still_load():
    model = tiny_model()
    state = model.state_dict()
    assert not any(k.endswith(".attn.mask") for k in state)
    legacy = dict(state)
    for i in range(len(model.blocks)):
        legacy[f"blocks.{i}.attn.mask"] = torch.tril(torch.ones(24, 24)).view(1, 1, 24, 24)
    fresh = tiny_model(seed=5)
    fresh.load_state_dict(legacy)
    idx = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        assert torch.equal(fresh(idx)[0], model(idx)[0])


def test_converter_upgrades_legacy_checkpoint(tmp_path):
    from training.convert_checkpoint import convert
    tok = CharTokenizer("abcdefghij \n")
    model = tiny_model(vocab=tok.vocab_size)
    legacy = dict(model.state_dict())
    for i in range(len(model.blocks)):
        legacy[f"blocks.{i}.attn.mask"] = torch.tril(torch.ones(24, 24)).view(1, 1, 24, 24)
    src = tmp_path / "legacy.pt"
    torch.save({"model_state": legacy, "model_cfg": model.cfg.__dict__, "tokenizer": tok.to_dict()}, src)
    report = convert(src, tmp_path / "converted.pt")
    assert report["dropped_keys"] == 2 and report["max_backend_logit_diff"] < 1e-5
    blob = torch.load(tmp_path / "converted.pt", weights_only=True)
    assert blob["format_version"] == 2 and not any(k.endswith(".attn.mask") for k in blob["model_state"])
    with pytest.raises(ValueError):
        convert(src, tmp_path / "converted.pt")
    with pytest.raises(ValueError):
        convert(src, src)


@pytest.mark.parametrize("kind", ["char", "bpe"])
def test_special_tokens_are_explicit_and_round_trip(kind):
    text = "def add(a, b):\n    return a + b\n<|user|> literal\n"
    tok = tokenization.build_with_specials(kind, text, vocab_size=300)
    assert [tok.special_id(s) for s in SPECIAL_TOKENS] == list(range(tok.vocab_size - 3, tok.vocab_size))
    plain = tok.encode(text)
    assert not set(plain) & tok.special_ids
    assert tok.decode(plain) == text
    turn = f"{USER}add a and b{ASSISTANT}def add(a, b): return a + b{ENDOFTEXT}"
    ids = tok.encode(turn, allowed_special="all")
    assert ids[0] == tok.special_id(USER) and ids[-1] == tok.special_id(ENDOFTEXT)
    assert tok.decode(ids) == turn
    assert ENDOFTEXT not in tok.decode(ids, skip_special=True)
    back = tokenization.from_dict(json.loads(json.dumps(tok.to_dict())))
    assert back.to_dict() == tok.to_dict() and back.encode(turn, allowed_special="all") == ids
    with pytest.raises(ValueError):
        tok.encode("x", allowed_special={"<|nope|>"})


def test_legacy_tokenizer_dicts_are_unchanged():
    bpe = BPETokenizer().train("aaab aaab aaab baab abab", vocab_size=260)
    assert set(bpe.to_dict()) == {"kind", "merges"}
    assert tokenization.from_dict(bpe.to_dict()).vocab_size == bpe.vocab_size == 256 + len(bpe.merges)
    char = CharTokenizer("abc")
    assert char.to_dict() == {"kind": "char", "chars": ["a", "b", "c"]}
    with pytest.raises(ValueError):
        tokenization.from_dict({"kind": "char", "chars": ["a"], "special": {ENDOFTEXT: 7}})


def documents(n=60):
    return [f"def f{i}(x):\n    return x * {i} + {i % 7}\n" * (1 + i % 4) for i in range(n)]


def build_shards(path, val_fraction=0.2):
    from data.shards import write_shards
    tok = CharTokenizer("".join(documents())).add_special_tokens()
    return tok, write_shards(documents(), tok, path, shard_tokens=900, val_fraction=val_fraction,
                             source={"kind": "test"})


def test_shards_are_pinned_split_by_document_and_tamper_evident(tmp_path):
    from data.shards import ShardSplit, document_split, load_manifest
    tok, (manifest, digest) = build_shards(tmp_path / "set")
    eot = tok.special_id(ENDOFTEXT)
    assert manifest["splits"]["train"]["documents"] + manifest["splits"]["val"]["documents"] == 60
    assert len(manifest["splits"]["train"]["shards"]) > 1
    val_docs = [d for d in documents() if document_split(d, 0.2) == "val"]
    assert manifest["splits"]["val"]["documents"] == len(val_docs) > 0
    loaded, loaded_digest, loaded_tok = load_manifest(tmp_path / "set", expect_sha256=digest)
    assert loaded_digest == digest and loaded_tok.to_dict() == tok.to_dict()
    split = ShardSplit(tmp_path / "set", "train", loaded)
    x, y = split.sample(16, 20, torch.Generator().manual_seed(0), doc_aligned=True)
    assert x.shape == y.shape == (16, 20) and torch.equal(x[:, 1:], y[:, :-1])
    for row in x:
        assert tok.decode(row[:4].tolist()) == "def "
    long_x, _ = split.sample(16, 200, torch.Generator().manual_seed(0), doc_aligned=True)
    assert (long_x == eot).any()
    assert split._cumulative(201)[1] <= len(split)
    with pytest.raises(ValueError):
        split.sample(1, 100_000, torch.Generator())
    with pytest.raises(ValueError):
        load_manifest(tmp_path / "set", expect_sha256="0" * 64)
    shard = tmp_path / "set" / manifest["splits"]["train"]["shards"][0]["file"]
    raw = bytearray(shard.read_bytes())
    raw[0] ^= 1
    shard.write_bytes(bytes(raw))
    with pytest.raises(ValueError):
        load_manifest(tmp_path / "set")
    load_manifest(tmp_path / "set", verify=False)
    with pytest.raises(ValueError):
        build_shards(tmp_path / "set")


def test_cosine_schedule_and_parameter_groups():
    import train
    cfg = TrainConfig(learning_rate=1e-3, warmup_iters=10, max_iters=110, lr_decay_iters=110, min_lr_ratio=0.1)
    assert train.lr_for(0, cfg) == pytest.approx(1e-4)
    assert train.lr_for(9, cfg) == pytest.approx(1e-3)
    assert train.lr_for(60, cfg) == pytest.approx(1e-4 + 0.5 * (1 + math.cos(math.pi * 0.5)) * 9e-4)
    assert train.lr_for(110, cfg) == pytest.approx(1e-4) == train.lr_for(500, cfg)
    cfg.lr_schedule = "constant"
    assert train.lr_for(60, cfg) == 1e-3
    model = tiny_model()
    optimizer = train.configure_optimizer(model, TrainConfig(weight_decay=0.1), "cpu")
    decay, no_decay = optimizer.param_groups
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decay["params"]) and all(p.dim() < 2 for p in no_decay["params"])
    assert sum(p.numel() for g in optimizer.param_groups for p in g["params"]) == model.num_params()


def shard_run(tmp_path, monkeypatch, *args):
    import train
    monkeypatch.setitem(train.PRESETS, "cpu", lambda: TrainConfig(
        block_size=16, n_layer=1, n_head=2, n_embd=16, batch_size=3, grad_accum_steps=2,
        eval_interval=3, eval_iters=2, max_iters=8, warmup_iters=2, dropout=0.1))
    monkeypatch.setattr(train.torch, "set_num_threads", lambda n: None)
    return train.main(["--device", "cpu", "--data-dir", str(tmp_path / "set"), "--log-interval", "1", *args])


def test_interrupted_shard_run_resumes_exactly(tmp_path, monkeypatch):
    _, (_, digest) = build_shards(tmp_path / "set")
    full = shard_run(tmp_path, monkeypatch, "--out-dir", str(tmp_path / "continuous"),
                     "--expect-data-sha256", digest)
    assert full["state"] == "completed" and full["completed_steps"] == 8
    part = shard_run(tmp_path, monkeypatch, "--out-dir", str(tmp_path / "s1"), "--stop-at-step", "5")
    assert part["state"] == "stop-at-step" and part["completed_steps"] == 5
    resumed = shard_run(tmp_path, monkeypatch, "--out-dir", str(tmp_path / "s2"),
                        "--resume", str(tmp_path / "s1" / "ckpt.pt"))
    assert resumed["state"] == "completed" and resumed["resumed_from_step"] == 5
    a = torch.load(tmp_path / "continuous" / "ckpt.pt", weights_only=True)
    b = torch.load(tmp_path / "s2" / "ckpt.pt", weights_only=True)
    for key, value in a["model_state"].items():
        assert torch.equal(value, b["model_state"][key]), key
    assert a["best_val_loss"] == b["best_val_loss"]
    best = torch.load(tmp_path / "continuous" / "best.pt", weights_only=True)
    assert "optimizer_state" not in best and best["training"]["val_loss"] == a["best_val_loss"]
    log = [json.loads(line) for line in (tmp_path / "continuous" / "train_log.jsonl").read_text().splitlines()]
    assert any("tokens_per_second" in r for r in log) and any("val_loss" in r for r in log)
    status = json.loads((tmp_path / "s2" / "status.json").read_text())
    assert status["data_kind"] == "shards" and len(status["ckpt_sha256"]) == 64


def test_deadline_stops_with_resumable_checkpoint_and_pins_are_enforced(tmp_path, monkeypatch):
    _, (_, digest) = build_shards(tmp_path / "set")
    status = shard_run(tmp_path, monkeypatch, "--out-dir", str(tmp_path / "d"), "--deadline-seconds", "0.000001")
    assert status["state"] == "deadline" and status["completed_steps"] == 1
    assert torch.load(tmp_path / "d" / "ckpt.pt", weights_only=True)["completed_steps"] == 1
    with pytest.raises(SystemExit):
        shard_run(tmp_path, monkeypatch, "--out-dir", str(tmp_path / "x"), "--expect-data-sha256", "f" * 64)
    with pytest.raises(SystemExit):
        shard_run(tmp_path, monkeypatch, "--out-dir", str(tmp_path / "y"),
                  "--resume", str(tmp_path / "d" / "ckpt.pt"), "--grad-accum-steps", "4")
    build_shards(tmp_path / "other", val_fraction=0.3)
    import train
    with pytest.raises(SystemExit):
        train.main(["--device", "cpu", "--data-dir", str(tmp_path / "other"), "--out-dir", str(tmp_path / "z"),
                    "--resume", str(tmp_path / "d" / "ckpt.pt")])


def test_signal_request_is_recorded():
    import signal
    import train
    stopper = train.StopRequest()
    stopper._handle(signal.SIGTERM, None)
    assert stopper.reason == "signal-SIGTERM"


def test_free_gpu_plan_chains_sessions_without_overwriting(tmp_path):
    from training.free_gpu import plan
    first = plan(tmp_path / "run1", steps=100, batch_size=8, max_seconds=30600, data_dir=tmp_path / "set",
                 preset="gpu_base", grad_accum_steps=16, lr_decay_iters=100, expect_data_sha256="a" * 64)
    verify, train_cmd = first["commands"]
    assert verify[-4:] == ["verify", str((tmp_path / "set").resolve()), "--expect-sha256", "a" * 64]
    assert train_cmd[train_cmd.index("--deadline-seconds") + 1] == "30000"
    assert "--grad-accum-steps" in train_cmd and "--resume" not in train_cmd
    second = plan(tmp_path / "run2", steps=100, max_seconds=30600, data_dir=tmp_path / "set",
                  resume=tmp_path / "run1" / "ckpt.pt")
    cmd = second["commands"][1]
    assert "--resume" in cmd and "--batch-size" not in cmd and "--preset" not in cmd
    with pytest.raises(ValueError):
        plan(tmp_path / "run1", data_dir=tmp_path / "set", resume=tmp_path / "run1" / "ckpt.pt")
    with pytest.raises(ValueError):
        plan(tmp_path / "run3", max_seconds=600, reserve_seconds=600)


def test_engine_stops_on_special_tokens_and_hides_them():
    from serve.engine import InferenceEngine
    tok = CharTokenizer("abcdefgh \n").add_special_tokens()
    model = tiny_model(vocab=tok.vocab_size, block=32)
    engine = InferenceEngine(model, tok)
    assert engine.stop_ids == {tok.special_id(ENDOFTEXT), tok.special_id(USER)}
    out = engine.generate("abc", max_new_tokens=20, seed=1)
    assert "<|" not in out["completion"]
    assert InferenceEngine(tiny_model(vocab=10), CharTokenizer("abcdefghij")).stop_ids == frozenset()


def test_benchmark_runs_on_cpu(tmp_path, monkeypatch):
    from training import benchmark
    monkeypatch.setitem(benchmark.PRESETS, "tiny", lambda: TrainConfig(
        block_size=16, n_layer=1, n_head=2, n_embd=16, vocab_size=64, dropout=0.0))
    out = tmp_path / "bench.json"
    benchmark.main(["--presets", "tiny", "--device", "cpu", "--steps", "1", "--warmup", "0",
                    "--batch-sizes", "2", "--out", str(out)])
    report = json.loads(out.read_text())
    assert report["results"][0]["best"]["tokens_per_second"] > 0


def test_bpe_tokenizer_survives_key_sorted_json(tmp_path):
    from data.shards import load_manifest, write_shards
    text = "".join(documents()) * 3
    tok = tokenization.build_with_specials("bpe", text, vocab_size=420)
    shuffled = json.loads(json.dumps(tok.to_dict(), sort_keys=True))
    back = tokenization.from_dict(shuffled)
    sample = documents()[7] + ENDOFTEXT
    assert back.encode(sample, allowed_special="all") == tok.encode(sample, allowed_special="all")
    assert back.to_dict() == tok.to_dict()
    _, digest = write_shards(documents(), tok, tmp_path / "bpe", shard_tokens=500, val_fraction=0.2)
    _, loaded_digest, loaded = load_manifest(tmp_path / "bpe", expect_sha256=digest)
    assert loaded_digest == digest and loaded.encode(sample, allowed_special="all") == tok.encode(sample, allowed_special="all")


@pytest.mark.parametrize("capability,reported,expected", [((7, 5), True, False), ((8, 0), True, True),
                                                          ((8, 6), False, False)])
def test_bf16_only_on_native_hardware(monkeypatch, capability, reported, expected):
    import train
    monkeypatch.setattr(train.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(train.torch.cuda, "get_device_capability", lambda index=0: capability)
    monkeypatch.setattr(train.torch.cuda, "is_bf16_supported", lambda: reported)
    assert train.native_bf16("cuda") is expected
    assert train.native_bf16("cpu") is False
