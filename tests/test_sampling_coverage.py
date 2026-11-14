import hashlib
import json

import numpy as np
import pytest
import torch

from config import TrainConfig, restore_train_config
from data.shards import LEGACY_PREFIX, LEGACY_RANDOM, UNIFORM_WINDOWS, ShardSplit
from training.base_audit import corpus_profile, coverage


def split(tmp_path, lengths=(400,)):
    records = []
    for index, size in enumerate(lengths):
        filename, idx = f"data{index}.bin", f"data{index}.idx"
        np.arange(size, dtype=np.uint16).tofile(tmp_path / filename)
        np.array([0], dtype=np.uint64).tofile(tmp_path / idx)
        records.append({"file": filename, "index": idx, "tokens": size})
    return ShardSplit(tmp_path, "train", {"splits": {"train": {"shards": records}}})


def test_long_document_reproducer_and_uniform_fix(tmp_path):
    data = split(tmp_path)
    legacy, _ = data.sample(1000, 16, torch.Generator().manual_seed(17), sampling_mode=LEGACY_PREFIX)
    uniform, targets = data.sample(1000, 16, torch.Generator().manual_seed(17), sampling_mode=UNIFORM_WINDOWS)
    assert torch.equal(legacy[:, 0], torch.zeros(1000, dtype=torch.long))
    assert uniform[:, 0].max() > 350
    assert torch.equal(uniform[:, 1:], targets[:, :-1])
    report = coverage([0], 400, 16)
    assert report["legacy_unreachable_tokens"] == 383
    assert report["uniform_reachable_tokens"] == 400
    default, _ = data.sample(1000, 16, torch.Generator().manual_seed(17))
    assert torch.equal(default, uniform)


def test_uniform_uses_legal_starts_and_can_reach_last_token(tmp_path, monkeypatch):
    data = split(tmp_path, (20, 40))
    seen = []

    def last_pick(high, size, generator):
        seen.append(high)
        return torch.full(size, high - 1, dtype=torch.long)

    monkeypatch.setattr(torch, "randint", last_pick)
    x, y = data.sample(1, 16, torch.Generator(), sampling_mode=UNIFORM_WINDOWS)
    assert seen == [(20 - 16) + (40 - 16)]
    assert x[0, 0] == 23 and y[0, -1] == 39
    assert data._cumulative(17, UNIFORM_WINDOWS)[1] == 28
    assert data._cumulative(17, LEGACY_PREFIX)[1] == 60


@pytest.mark.parametrize("mode,aligned", [(LEGACY_PREFIX, True), (LEGACY_RANDOM, False)])
def test_explicit_legacy_mode_matches_historical_algorithm(tmp_path, mode, aligned):
    data = split(tmp_path, (80, 120))
    data.starts = [np.array([0, 13, 70]), np.array([0, 37, 119])]
    generator = torch.Generator().manual_seed(97)
    picks = torch.randint(200, (100,), generator=generator).tolist()
    expected = []
    for position in picks:
        size, offset = (80, position) if position < 80 else (120, position - 80)
        if aligned:
            starts = data.starts[0 if position < 80 else 1]
            offset = int(starts[np.searchsorted(starts, offset, side="right") - 1])
        offset = min(offset, size - 17)
        expected.append(list(range(offset, offset + 16)))
    x, _ = data.sample(100, 16, torch.Generator().manual_seed(97), sampling_mode=mode)
    assert x.tolist() == expected


def test_short_shards_are_excluded_without_losing_last_valid_window(tmp_path):
    data = split(tmp_path, (10, 17))
    x, y = data.sample(3, 16, torch.Generator(), sampling_mode=UNIFORM_WINDOWS)
    assert data._cumulative(17, UNIFORM_WINDOWS)[1] == 1
    assert x[:, 0].tolist() == [0, 0, 0] and y[:, -1].tolist() == [16, 16, 16]


def test_fresh_config_is_uniform_but_old_config_preserves_legacy():
    assert TrainConfig().sampling_mode == UNIFORM_WINDOWS
    old = TrainConfig().__dict__.copy()
    old.pop("sampling_mode")
    assert restore_train_config(old).sampling_mode == LEGACY_PREFIX
    old["doc_aligned"] = False
    assert restore_train_config(old).sampling_mode == LEGACY_RANDOM


def test_coverage_accounts_for_overlaps_and_end_clipping():
    result = coverage([0, 5, 30], 40, 16)
    assert result["legacy_reachable_tokens"] == 39
    assert result["legacy_unreachable_tokens"] == 1
    with pytest.raises(ValueError, match="document-start"):
        coverage([1, 8], 20, 4)


@pytest.mark.parametrize("mode", ["typo", False])
def test_unknown_sampler_rejected(tmp_path, mode):
    with pytest.raises(ValueError, match="unsupported"):
        split(tmp_path).sample(1, 16, torch.Generator(), sampling_mode=mode)


def corpus(tmp_path):
    from tokenizer import CharTokenizer
    marker = tmp_path / "must-not-exist"
    text = f'open({str(marker)!r}, "w").write("must-not-run")\n'
    files = {}
    for split_name in ("train", "val"):
        row = {"text": text, "source": "test-code", "group": split_name, "licences": ["MIT"]}
        raw = (json.dumps(row) + "\n").encode()
        (tmp_path / f"{split_name}.jsonl").write_bytes(raw)
        files[f"{split_name}.jsonl"] = {"sha256": hashlib.sha256(raw).hexdigest()}
    (tmp_path / "corpus-manifest.json").write_text(json.dumps({"files": files}))
    (tmp_path / "holdout_test.jsonl").write_bytes(b"\xffmust-not-read")
    return CharTokenizer(text).add_special_tokens(), marker


def test_corpus_audit_does_not_execute_code_or_read_test_split(tmp_path):
    tokenizer, marker = corpus(tmp_path)
    report = corpus_profile(tmp_path, tokenizer, 16, samples=2)
    assert not marker.exists()
    assert report["test_split_read"] is False and report["private_holdout_read"] is False
    assert report["train_validation_group_overlap"] == 0
    assert report["splits"]["train"]["code_sample"]["syntax"]["parsed"] == 1
    assert "must-not-run" not in json.dumps(report)


def test_corpus_audit_rejects_unpinned_content(tmp_path):
    tokenizer, _ = corpus(tmp_path)
    with (tmp_path / "train.jsonl").open("a") as stream:
        stream.write((tmp_path / "train.jsonl").read_text())
    with pytest.raises(ValueError, match="does not match"):
        corpus_profile(tmp_path, tokenizer, 16, samples=2)


def test_corpus_audit_reports_shared_groups_without_hiding_them(tmp_path):
    tokenizer, _ = corpus(tmp_path)
    row = json.loads((tmp_path / "val.jsonl").read_text())
    row["group"] = "train"
    raw = (json.dumps(row) + "\n").encode()
    (tmp_path / "val.jsonl").write_bytes(raw)
    manifest = json.loads((tmp_path / "corpus-manifest.json").read_text())
    manifest["files"]["val.jsonl"]["sha256"] = hashlib.sha256(raw).hexdigest()
    (tmp_path / "corpus-manifest.json").write_text(json.dumps(manifest))
    assert corpus_profile(tmp_path, tokenizer, 16, samples=2)["train_validation_group_overlap"] == 1
