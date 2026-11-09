"""Build the Phase 2 training corpus: licensed code, capped security prose,
repository-level splits, a manifest, a code-aware tokenizer and token shards.

    python -m data.build_phase2 --out DIR --stack-files 12 --target-bytes 2000000000
    python -m data.build_phase2 --out DIR --sources cpython --target-bytes 50000000

Hugging Face data is read as Parquet with pyarrow; nothing from a dataset is
executed. The token comes from HF_TOKEN or the huggingface_hub token file and is
only ever sent as a request header to huggingface.co.
"""

import argparse
import json
import os
import random
import sys
import tarfile
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import tokenization  # noqa: E402
from data import pipeline as P  # noqa: E402
from data.shards import manifest_sha256, sha256_file, write_shards  # noqa: E402

STACK = {"name": "bigcode/the-stack-dedup", "revision": "17cad72c886a2858e08d4c349a00d6466f54df63",
         "subdir": "data/python", "files": 144}
CPYTHON = {"name": "python/cpython", "tag": "v3.13.9",
           "url": "https://github.com/python/cpython/archive/refs/tags/v3.13.9.tar.gz"}
STACK_COLUMNS = ("content", "max_stars_repo_licenses", "max_stars_repo_name", "max_stars_repo_path")
EXCLUDED_SOURCES = {
    "CyberNative/Code_Vulnerability_Security_DPO": "owner decision pending: 424 Python rows, appears model-generated "
                                                   "(distillation concern), sampled rows contain corrupted code",
    "code-search-net/code_search_net": "no per-row licence; requires per-repository verification before use",
    "OWASP Cheat Sheets": "CC BY-SA share-alike; owner decision required before ingesting",
    "MITRE CWE": "terms of use not yet reviewed",
}
_DEDUP = P.NearDuplicateIndex()


def hf_token():
    token = os.environ.get("HF_TOKEN")
    if token:
        return token.strip()
    path = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "token"
    return path.read_text().strip() if path.exists() else None


def download(url, dest, token=None, expect_sha256=None):
    dest = Path(dest)
    if dest.exists() and (expect_sha256 is None or sha256_file(dest) == expect_sha256):
        return dest
    request = urllib.request.Request(url, headers={"User-Agent": "llm-from-scratch-data/1"})
    if token and url.startswith("https://huggingface.co/"):
        request.add_header("Authorization", f"Bearer {token}")
    partial = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(request, timeout=120) as resp, open(partial, "wb") as fh:
        while True:
            block = resp.read(1 << 22)
            if not block:
                break
            fh.write(block)
    if expect_sha256 and sha256_file(partial) != expect_sha256:
        partial.unlink()
        raise ValueError(f"{url} does not match the pinned sha256")
    os.replace(partial, dest)
    return dest


def stack_rows(cache, n_files, seed):
    import pyarrow.parquet as pq
    token = hf_token()
    if not token:
        raise SystemExit("The Stack is gated: accept its terms on huggingface.co, then run `hf auth login` "
                         "with a read-only token or set HF_TOKEN.")
    order = list(range(STACK["files"]))
    random.Random(seed).shuffle(order)
    for index in order[:n_files]:
        name = f"data-{index:05d}-of-{STACK['files']:05d}.parquet"
        url = f"https://huggingface.co/datasets/{STACK['name']}/resolve/{STACK['revision']}/{STACK['subdir']}/{name}"
        path = download(url, Path(cache) / "stack" / name, token=token)
        table = pq.read_table(path, columns=list(STACK_COLUMNS))
        file_sha = sha256_file(path)
        for row in table.to_pylist():
            yield {"source": STACK["name"], "text": row["content"] or "",
                   "licences": list(row["max_stars_repo_licenses"] or []),
                   "group": f"stack:{row['max_stars_repo_name']}", "file": name, "file_sha256": file_sha}


def cpython_rows(cache):
    path = download(CPYTHON["url"], Path(cache) / f"cpython-{CPYTHON['tag']}.tar.gz")
    skip = ("/test/", "/tests/", "/idlelib/", "/turtledemo/", "/lib2to3/", "/ensurepip/")
    with tarfile.open(path) as tar:
        for member in sorted(tar.getmembers(), key=lambda m: m.name):
            name = member.name
            if not member.isfile() or "/Lib/" not in name or not name.endswith(".py"):
                continue
            if any(s in name for s in skip) or Path(name).name.startswith("test_"):
                continue
            text = tar.extractfile(member).read().decode("utf-8", errors="replace")
            module = name.split("/Lib/", 1)[1]
            yield {"source": CPYTHON["name"], "text": text, "licences": ["PSF-2.0"],
                   "group": f"cpython:{module.split('/')[0]}", "file": module, "file_sha256": sha256_file(path)}


def prose_rows(path):
    if not path or not Path(path).exists():
        return
    text = Path(path).read_text(encoding="utf-8")
    for i, part in enumerate(t for t in text.split("\n\n\n") if t.strip()):
        yield {"source": "rfc-nist-prose", "text": part, "licences": None, "prose": True,
               "group": f"prose:{i // 50}", "file": Path(path).name, "file_sha256": None}


def _prefilter(row):
    if row.get("prose"):
        text, counts = P.scrub(row["text"])
        return {**row, "text": text, "verdict": {"text": text, "licences": ["IETF-RFC/NIST-public"],
                                                 "scrubbed": counts, "key": P.content_key(text)}}
    verdict = P.prefilter(row["text"], row["licences"], near_dup=_DEDUP)
    return {**row, "verdict": verdict}


def contamination_index(eval_dir):
    from data.contamination import _eval_strings
    texts = [t for _, t in _eval_strings(eval_dir)] if Path(eval_dir).exists() else []
    return P.ContaminationIndex(texts)


def build(args):
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; builds are immutable")
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache)
    sources = []
    if "prose" in args.sources:
        sources.append(prose_rows(args.prose))
    if "cpython" in args.sources:
        sources.append(cpython_rows(cache))
    if "stack" in args.sources:
        sources.append(stack_rows(cache, args.stack_files, args.seed))
    stats, dedup, exact = {}, P.NearDuplicateIndex(), set()
    contamination = contamination_index(args.eval_dir)
    docs = {"train": out / "train.jsonl", "val": out / "val.jsonl", "test": out / "holdout_test.jsonl"}
    handles = {k: open(v, "w", encoding="utf-8") for k, v in docs.items()}
    code_bytes = prose_bytes = 0
    started = time.monotonic()

    def rows():
        for source in sources:
            yield from source

    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for row in pool.map(_prefilter, rows(), chunksize=64):
                st = stats.setdefault(row["source"], P.SourceStats(row["source"]))
                st.rows_in += 1
                v = row["verdict"]
                reason = v.get("reason")
                is_prose = bool(row.get("prose"))
                if reason is None and v["key"] in exact:
                    reason = "exact-duplicate"
                if reason is None and not is_prose:
                    new, _ = dedup.add_if_new(signature=v["signature"])
                    if not new:
                        reason = "near-duplicate"
                if reason is None and contamination.hits(v["text"]):
                    reason = "eval-contamination"
                size = len(v.get("text", "").encode("utf-8"))
                if reason is None and is_prose and prose_bytes + size > args.max_prose_fraction * args.target_bytes:
                    reason = "prose-cap"
                if reason is not None:
                    st.dropped[reason] += 1
                    continue
                exact.add(v["key"])
                split = P.split_for_group(row["group"], args.val_fraction, args.test_fraction)
                handles[split].write(json.dumps({"text": v["text"], "source": row["source"], "group": row["group"],
                                                 "licences": v["licences"]}) + "\n")
                st.rows_out += 1
                st.bytes_out += size
                st.splits[split] += 1
                for lic in v["licences"]:
                    st.licences[lic] += 1
                for k, n in v["scrubbed"].items():
                    st.scrubbed[k] += n
                if is_prose:
                    prose_bytes += size
                else:
                    code_bytes += size
                if code_bytes + prose_bytes >= args.target_bytes:
                    break
    finally:
        for h in handles.values():
            h.close()
    total = code_bytes + prose_bytes
    manifest = {"schema": 1, "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "elapsed_seconds": round(time.monotonic() - started, 1),
                "target_bytes": args.target_bytes, "bytes": total, "code_bytes": code_bytes,
                "prose_bytes": prose_bytes, "code_fraction": round(code_bytes / total, 4) if total else 0,
                "allowed_licences": sorted(P.ALLOWED_LICENCES), "split": {"by": "repository",
                "val_fraction": args.val_fraction, "test_fraction": args.test_fraction},
                "sources": {"stack": STACK if "stack" in args.sources else None,
                            "cpython": CPYTHON if "cpython" in args.sources else None},
                "excluded_sources": EXCLUDED_SOURCES,
                "contamination": {"method": "13-gram overlap with evals/data", "eval_ngrams": len(contamination.grams)},
                "stats": {k: v.to_dict() for k, v in stats.items()},
                "files": {p.name: {"sha256": sha256_file(p), "bytes": p.stat().st_size} for p in docs.values()}}
    (out / "corpus-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("bytes", "code_fraction", "elapsed_seconds")}), flush=True)
    return manifest


def iter_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)["text"]


def stratified_sample(path, max_chars, seed):
    rng = random.Random(seed)
    by_source = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            by_source.setdefault(row["source"], []).append(row["text"])
    total = sum(len(t) for texts in by_source.values() for t in texts) or 1
    parts = []
    for source, texts in sorted(by_source.items()):
        budget = max_chars * sum(len(t) for t in texts) / total
        rng.shuffle(texts)
        used = 0
        for t in texts:
            if used >= budget:
                break
            parts.append(t)
            used += len(t)
    rng.shuffle(parts)
    return "".join(parts)


def tokenize(args):
    out = Path(args.out)
    sample = stratified_sample(out / "train.jsonl", args.tokenizer_chars, args.seed)
    tok = tokenization.build_with_specials("bpe_code", sample, vocab_size=args.vocab_size,
                                           bpe_train_chars=len(sample), verbose=True)
    held = "".join(list(iter_jsonl(out / "val.jsonl"))[:2000])
    ids = tok.encode(held) if held else []
    if held and tok.decode(ids) != held:
        raise SystemExit("tokenizer round-trip failed on validation text")
    bpt = len(held.encode("utf-8")) / len(ids) if ids else None
    shard_dir = out / "shards"
    docs = [(t, "train") for t in iter_jsonl(out / "train.jsonl")] + [(t, "val") for t in iter_jsonl(out / "val.jsonl")]
    corpus = json.loads((out / "corpus-manifest.json").read_text())
    _, digest = write_shards(docs, tok, shard_dir, source={"corpus_manifest_sha256": sha256_file(out / "corpus-manifest.json"),
                                                           "code_fraction": corpus["code_fraction"]})
    report = {"vocab_size": tok.vocab_size, "validation_bytes_per_token": round(bpt, 3) if bpt else None,
              "shard_manifest_sha256": digest, "tokenizer_sample_chars": len(sample)}
    (out / "tokenizer-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=str(Path.home() / ".cache" / "llmfs-data"))
    ap.add_argument("--sources", nargs="+", default=["cpython", "stack", "prose"], choices=["cpython", "stack", "prose"])
    ap.add_argument("--stack-files", type=int, default=12)
    ap.add_argument("--prose", default=str(HERE / "corpus.txt"))
    ap.add_argument("--eval-dir", default=str(HERE.parent / "evals" / "data"))
    ap.add_argument("--target-bytes", type=int, default=2_000_000_000)
    ap.add_argument("--max-prose-fraction", type=float, default=0.08)
    ap.add_argument("--val-fraction", type=float, default=0.01)
    ap.add_argument("--test-fraction", type=float, default=0.01)
    ap.add_argument("--vocab-size", type=int, default=16384)
    ap.add_argument("--tokenizer-chars", type=int, default=50_000_000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--skip-tokenize", action="store_true")
    args = ap.parse_args(argv)
    build(args)
    if not args.skip_tokenize:
        tokenize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
