"""Find training rows that copy evaluation benchmarks.

    python -m data.decontaminate --refs ~/llmfs-data/bench/reference_texts.jsonl \
        --input ~/llmfs-data/corpus-v2/train.jsonl --field text --out hits.json
    python -m data.decontaminate apply --corpus ~/llmfs-data/corpus-v2 --hits hits.json \
        --refs ~/llmfs-data/bench/reference_texts.jsonl --out ~/llmfs-data/corpus-v3

A row is flagged when it contains at least --containment of any reference
task's 13-word n-grams (prompt plus reference solution). Containment, unlike
Jaccard, still catches a copied function inside a much larger file.
"""

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from data.pipeline import ngrams  # noqa: E402

_STATE = {}


def load_refs(path, n=13, min_grams=8):
    refs, index = [], defaultdict(list)
    for line in Path(path).expanduser().read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        grams = ngrams(row["text"], n)
        if len(grams) < min_grams:
            continue
        rid = len(refs)
        refs.append({"id": row["id"], "grams": len(grams)})
        for g in grams:
            index[g].append(rid)
    return refs, dict(index)


def _init(refs_path, n, containment):
    refs, index = load_refs(refs_path, n)
    _STATE.update(refs=refs, index=index, n=n, containment=containment)


def row_hits(text):
    counts = defaultdict(int)
    index = _STATE["index"]
    for g in ngrams(text, _STATE["n"]):
        for rid in index.get(g, ()):
            counts[rid] += 1
    hits = []
    for rid, c in counts.items():
        ref = _STATE["refs"][rid]
        share = c / ref["grams"]
        if share >= _STATE["containment"]:
            hits.append({"ref": ref["id"], "containment": round(share, 3)})
    return hits


def _scan(job):
    path, start, stop, field = job
    out = []
    with open(path, "rb") as fh:
        fh.seek(start)
        while fh.tell() < stop:
            offset = fh.tell()
            raw = fh.readline()
            if not raw:
                break
            row = json.loads(raw)
            text = row[field] if isinstance(field, str) else "\n".join(row[f] for f in field)
            hits = row_hits(text)
            if hits:
                out.append({"offset": offset, "group": row.get("group"), "id": row.get("id"), "hits": hits})
    return out


def _jobs(path, field, chunk_bytes=32 << 20):
    size = Path(path).stat().st_size
    jobs, start = [], 0
    with open(path, "rb") as fh:
        while start < size:
            fh.seek(min(start + chunk_bytes, size))
            fh.readline()
            stop = min(fh.tell(), size)
            jobs.append((str(path), start, stop, field))
            start = stop
    return jobs


def scan(refs_path, inputs, field, n=13, containment=0.5, workers=8):
    results = {}
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(str(refs_path), n, containment)) as pool:
        for path in inputs:
            path = str(Path(path).expanduser())
            hits = [h for chunk in pool.map(_scan, _jobs(path, field)) for h in chunk]
            results[path] = hits
    return results


def apply(corpus_dir, hits_path, out_dir, refs_path):
    import hashlib
    import shutil
    from data.shards import filter_shards, sha256_file
    corpus, out = Path(corpus_dir).expanduser(), Path(out_dir).expanduser()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    out.mkdir(parents=True, exist_ok=True)
    hits = json.loads(Path(hits_path).read_text())
    groups = sorted({r["group"] for rows in hits.values() for r in rows})
    refs = sorted({h["ref"] for rows in hits.values() for r in rows for h in r["hits"]})
    drop, files, dropped_rows = set(), {}, {}
    for split, name in (("train", "train.jsonl"), ("val", "val.jsonl"), ("test", "holdout_test.jsonl")):
        src, dst = corpus / name, out / name
        index, kept, removed = 0, 0, 0
        h = hashlib.sha256()
        with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
            for line in fin:
                row = json.loads(line)
                if row["group"] in groups:
                    removed += 1
                    if split != "test" and row["text"]:
                        drop.add((split, index))
                else:
                    fout.write(line)
                    h.update(line.encode("utf-8"))
                    kept += 1
                if split != "test" and row["text"]:
                    index += 1
        files[name] = {"sha256": h.hexdigest(), "bytes": dst.stat().st_size, "rows": kept}
        dropped_rows[name] = removed
    note = {"decontamination": {"method": "13-gram containment against HumanEval and MBPP prompts and reference "
                                          "solutions; whole repositories dropped",
                                "references_sha256": sha256_file(refs_path), "groups": groups, "refs_hit": refs}}
    _, digest = filter_shards(corpus / "shards", out / "shards", drop, note)
    manifest = json.loads((corpus / "corpus-manifest.json").read_text())
    manifest["files"] = files
    manifest["parent_corpus_manifest_sha256"] = sha256_file(corpus / "corpus-manifest.json")
    manifest["decontamination"] = {**note["decontamination"], "rows_dropped": dropped_rows}
    (out / "corpus-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = json.loads((corpus / "tokenizer-report.json").read_text())
    report["shard_manifest_sha256"] = digest
    (out / "tokenizer-report.json").write_text(json.dumps(report, indent=2) + "\n")
    shutil.copyfile(hits_path, out / "decontamination-hits.json")
    return {"groups": len(groups), "refs": len(refs), "rows_dropped": dropped_rows, "shard_manifest_sha256": digest}


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if argv[:1] == ["apply"]:
        ap = argparse.ArgumentParser()
        ap.add_argument("--corpus", required=True)
        ap.add_argument("--hits", required=True)
        ap.add_argument("--refs", required=True)
        ap.add_argument("--out", required=True)
        a = ap.parse_args(argv[1:])
        print(json.dumps(apply(a.corpus, a.hits, a.out, a.refs), indent=2))
        return 0
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refs", required=True)
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--field", nargs="+", default=["text"])
    ap.add_argument("--containment", type=float, default=0.5)
    ap.add_argument("--n", type=int, default=13)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    field = args.field[0] if len(args.field) == 1 else args.field
    results = scan(args.refs, args.input, field, args.n, args.containment, args.workers)
    Path(args.out).write_text(json.dumps(results, indent=1) + "\n")
    print(json.dumps({p: {"rows_flagged": len(h), "refs": len({x["ref"] for r in h for x in r["hits"]})}
                      for p, h in results.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
