"""Versioned uint16 token shards with document boundaries.

    python -m data.shards stdlib --out DIR --vocab-size 4096
    python -m data.shards text --out DIR --input a.txt b.txt --vocab-size 8192
    python -m data.shards verify DIR
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import tokenization  # noqa: E402
from special_tokens import ENDOFTEXT, SPECIAL_TOKENS  # noqa: E402

FORMAT = "llmfs-token-shards"
VERSION = 1
SPLITS = ("train", "val")


def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def document_split(text, val_fraction):
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return "val" if int.from_bytes(digest[:8], "big") / 2**64 < val_fraction else "train"


class _ShardWriter:
    def __init__(self, out_dir, split, shard_tokens):
        self.out_dir, self.split, self.shard_tokens = Path(out_dir), split, shard_tokens
        self.tokens, self.starts, self.records, self.documents = [], [], [], 0
        self.count = 0

    def add(self, ids):
        self.starts.append(self.count)
        self.tokens.append(np.asarray(ids, dtype=np.uint16))
        self.count += len(ids)
        self.documents += 1
        if self.count >= self.shard_tokens:
            self.flush()

    def flush(self):
        if not self.tokens:
            return
        index = len(self.records)
        name = f"{self.split}_{index:04d}"
        data = np.concatenate(self.tokens)
        starts = np.asarray(self.starts, dtype=np.uint64)
        bin_path, idx_path = self.out_dir / f"{name}.bin", self.out_dir / f"{name}.idx"
        data.tofile(bin_path)
        starts.tofile(idx_path)
        self.records.append({"file": bin_path.name, "index": idx_path.name,
                             "tokens": int(data.size), "documents": int(starts.size),
                             "sha256": sha256_file(bin_path), "index_sha256": sha256_file(idx_path)})
        self.tokens, self.starts, self.count = [], [], 0


def write_shards(documents, tokenizer, out_dir, shard_tokens=1 << 26, val_fraction=0.005,
                 source=None):
    eot = tokenizer.special_id(ENDOFTEXT)
    if eot is None:
        raise ValueError("tokenizer needs the <|endoftext|> special token")
    if tokenizer.vocab_size > 65536:
        raise ValueError("uint16 shards need a vocabulary of at most 65,536 tokens")
    if not 0 < val_fraction < 1 or shard_tokens < 1:
        raise ValueError("val_fraction must be in (0, 1) and shard_tokens positive")
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"{out} is not empty; shard sets are immutable")
    out.mkdir(parents=True, exist_ok=True)
    writers = {s: _ShardWriter(out, s, shard_tokens) for s in SPLITS}
    for item in documents:
        text, split = item if isinstance(item, tuple) else (item, None)
        if not text:
            continue
        split = split or document_split(text, val_fraction)
        if split not in writers:
            raise ValueError(f"unknown split {split!r}; test documents belong in a separate holdout file")
        writers[split].add(tokenizer.encode(text) + [eot])
    for w in writers.values():
        w.flush()
    if not writers["train"].records or not writers["val"].records:
        raise ValueError("both splits need at least one document; add data or adjust val_fraction")
    tok_dict = tokenizer.to_dict()
    (out / "tokenizer.json").write_text(json.dumps(tok_dict, separators=(",", ":"), ensure_ascii=False))
    manifest = {"format": FORMAT, "version": VERSION, "dtype": "uint16",
                "vocab_size": tokenizer.vocab_size, "eot_id": eot,
                "tokenizer_sha256": sha256_bytes(canonical_json(tok_dict).encode("utf-8")),
                "val_fraction": val_fraction, "source": source or {},
                "splits": {s: {"shards": w.records,
                               "tokens": sum(r["tokens"] for r in w.records),
                               "documents": sum(r["documents"] for r in w.records)}
                           for s, w in writers.items()}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest, manifest_sha256(out)


def manifest_sha256(data_dir):
    manifest = json.loads((Path(data_dir) / "manifest.json").read_text())
    return sha256_bytes(canonical_json(manifest).encode("utf-8"))


def load_manifest(data_dir, verify=True, expect_sha256=None):
    root = Path(data_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("format") != FORMAT or manifest.get("version") != VERSION:
        raise ValueError("unsupported shard format")
    digest = manifest_sha256(root)
    if expect_sha256 and digest != expect_sha256:
        raise ValueError(f"shard manifest sha256 {digest} does not match the pinned {expect_sha256}")
    tok_text = (root / "tokenizer.json").read_text()
    if sha256_bytes(canonical_json(json.loads(tok_text)).encode("utf-8")) != manifest["tokenizer_sha256"]:
        raise ValueError("tokenizer.json does not match the manifest")
    for split in SPLITS:
        for rec in manifest["splits"][split]["shards"]:
            bin_path, idx_path = root / rec["file"], root / rec["index"]
            if bin_path.stat().st_size != rec["tokens"] * 2 or idx_path.stat().st_size != rec["documents"] * 8:
                raise ValueError(f"{rec['file']} size does not match the manifest")
            if verify and (sha256_file(bin_path) != rec["sha256"] or sha256_file(idx_path) != rec["index_sha256"]):
                raise ValueError(f"{rec['file']} hash does not match the manifest")
    return manifest, digest, tokenization.from_dict(json.loads(tok_text))


class ShardSplit:
    def __init__(self, data_dir, split, manifest):
        root = Path(data_dir)
        self.shards, self.starts, weights = [], [], []
        for rec in manifest["splits"][split]["shards"]:
            self.shards.append(np.memmap(root / rec["file"], dtype=np.uint16, mode="r"))
            self.starts.append(np.fromfile(root / rec["index"], dtype=np.uint64).astype(np.int64))
            weights.append(rec["tokens"])
        self.tokens = int(sum(weights))
        self.weights = weights
        self._eligible = {}

    def __len__(self):
        return self.tokens

    def _cumulative(self, span):
        if span not in self._eligible:
            usable = [w if w >= span else 0 for w in self.weights]
            if not sum(usable):
                raise ValueError("no shard has more than block_size tokens")
            self._eligible[span] = (np.cumsum(usable), int(sum(usable)))
        return self._eligible[span]

    def sample(self, batch_size, block_size, generator, doc_aligned=True):
        import torch
        span = block_size + 1
        cumulative, total = self._cumulative(span)
        picks = torch.randint(total, (batch_size,), generator=generator).tolist()
        rows = []
        for p in picks:
            s = int(np.searchsorted(cumulative, p, side="right"))
            shard = self.shards[s]
            offset = p - (int(cumulative[s - 1]) if s else 0)
            if doc_aligned:
                starts = self.starts[s]
                offset = int(starts[np.searchsorted(starts, offset, side="right") - 1])
            offset = min(offset, shard.size - span)
            rows.append(np.asarray(shard[offset:offset + span], dtype=np.int64))
        block = torch.from_numpy(np.stack(rows))
        return block[:, :-1].contiguous(), block[:, 1:].contiguous()


def iter_text_files(paths):
    for path in paths:
        yield Path(path).read_text(encoding="utf-8", errors="replace")


def iter_stdlib(max_bytes):
    import sysconfig
    from data.build_code_corpus import iter_stdlib_files
    total = 0
    for _, src in iter_stdlib_files(sysconfig.get_paths()["stdlib"]):
        if total >= max_bytes:
            return
        total += len(src.encode("utf-8"))
        yield src


def _build(args, documents, source):
    docs = list(documents)
    sample = "".join(docs)[:args.bpe_train_chars]
    tok = tokenization.build_with_specials("bpe", sample, vocab_size=args.vocab_size,
                                           bpe_train_chars=args.bpe_train_chars, verbose=True)
    staging = Path(tempfile.mkdtemp(prefix=".shards-", dir=Path(args.out).resolve().parent))
    manifest, digest = write_shards(docs, tok, staging / "set", shard_tokens=args.shard_tokens,
                                    val_fraction=args.val_fraction, source=source)
    os.replace(staging / "set", args.out)
    staging.rmdir()
    print(json.dumps({"out": args.out, "manifest_sha256": digest, "vocab_size": manifest["vocab_size"],
                      "train_tokens": manifest["splits"]["train"]["tokens"],
                      "val_tokens": manifest["splits"]["val"]["tokens"]}, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("stdlib", "text"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--vocab-size", type=int, default=4096)
        p.add_argument("--bpe-train-chars", type=int, default=300_000)
        p.add_argument("--shard-tokens", type=int, default=1 << 26)
        p.add_argument("--val-fraction", type=float, default=0.02)
        if name == "stdlib":
            p.add_argument("--max-bytes", type=int, default=8_000_000)
        else:
            p.add_argument("--input", nargs="+", required=True)
    v = sub.add_parser("verify")
    v.add_argument("data_dir")
    v.add_argument("--expect-sha256")
    args = ap.parse_args(argv)
    if args.cmd == "verify":
        manifest, digest, _ = load_manifest(args.data_dir, verify=True, expect_sha256=args.expect_sha256)
        print(json.dumps({"manifest_sha256": digest, "vocab_size": manifest["vocab_size"],
                          "train_tokens": manifest["splits"]["train"]["tokens"],
                          "val_tokens": manifest["splits"]["val"]["tokens"]}, indent=2))
        return 0
    if args.cmd == "stdlib":
        import platform
        _build(args, iter_stdlib(args.max_bytes),
               {"kind": "cpython-stdlib", "licence": "PSF-2.0", "python": platform.python_version(),
                "max_bytes": args.max_bytes, "special_tokens": list(SPECIAL_TOKENS)})
    else:
        _build(args, iter_text_files(args.input),
               {"kind": "text-files", "files": [{"name": Path(p).name, "sha256": sha256_file(p)}
                                                for p in args.input],
                "special_tokens": list(SPECIAL_TOKENS)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
