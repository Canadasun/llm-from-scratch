import json

import numpy as np

import tokenization
from data import decontaminate as D
from data.shards import filter_shards, load_manifest, write_shards

REF = ("def separate_paren_groups(paren_string):\n    result = []\n    current = []\n    depth = 0\n"
       "    for c in paren_string:\n        if c == '(':\n            depth += 1\n            current.append(c)\n"
       "        elif c == ')':\n            depth -= 1\n            current.append(c)\n            if depth == 0:\n"
       "                result.append(''.join(current))\n                current.clear()\n    return result\n")


def _docs(n=40):
    return [f"def helper_{i}(value):\n    total = value * {i} + {i * 3}\n    return total - {i}\n" for i in range(n)]


def test_scan_flags_copies_inside_larger_files(tmp_path):
    refs = tmp_path / "refs.jsonl"
    refs.write_text(json.dumps({"id": "HumanEval/1", "text": REF}) + "\n")
    rows = [{"text": _docs()[0] + "\n" + REF + "\n" + "\n".join(_docs()[1:30]), "group": "copied"},
            {"text": "\n".join(_docs()), "group": "clean"},
            {"text": REF.replace("depth", "level").replace("current", "buf").replace("result", "out"), "group": "renamed"}]
    data = tmp_path / "train.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in rows))
    hits = D.scan(refs, [data], "text", containment=0.5, workers=1)[str(data)]
    assert [h["group"] for h in hits] == ["copied"] and hits[0]["hits"][0]["ref"] == "HumanEval/1"


def test_filter_shards_drops_documents_without_retokenizing(tmp_path):
    docs = _docs()
    tok = tokenization.build_with_specials("bpe_code", "".join(docs) * 3, vocab_size=320,
                                           bpe_train_chars=10 ** 6)
    src = tmp_path / "src"
    write_shards([(d, "val" if i % 10 == 0 else "train") for i, d in enumerate(docs)], tok, src, shard_tokens=200)
    train_docs = [d for i, d in enumerate(docs) if i % 10]
    drop = {("train", 0), ("train", 7), ("train", len(train_docs) - 1)}
    manifest, digest = filter_shards(src, tmp_path / "out", drop, {"note": "test"})
    loaded, again, tok2 = load_manifest(tmp_path / "out", expect_sha256=digest)
    assert loaded["splits"]["train"]["documents"] == len(train_docs) - 3
    assert loaded["source"]["removed_documents"] == {"train": 3, "val": 0}
    texts = []
    for rec in loaded["splits"]["train"]["shards"]:
        t = np.fromfile(tmp_path / "out" / rec["file"], dtype=np.uint16)
        s = np.fromfile(tmp_path / "out" / rec["index"], dtype=np.uint64).astype(np.int64)
        for a, b in zip(s, np.append(s[1:], t.size)):
            texts.append(tok2.decode(t[a:b].tolist(), skip_special=True))
    assert texts == [d for i, d in enumerate(train_docs) if ("train", i) not in drop]
