import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from data.shards import load_manifest, sha256_file
from special_tokens import ENDOFTEXT

SCHEMA = "llmfs-sampler-validation-v1"


def build(data_dir, corpus_dir, block_size=1024, code_documents=128, prose_documents=32, seed=1729):
    if min(block_size, code_documents, prose_documents) < 1:
        raise ValueError("context and stratum sample sizes must be positive")
    data_dir, corpus_dir = Path(data_dir), Path(corpus_dir)
    manifest, manifest_sha, tokenizer = load_manifest(data_dir, verify=True)
    corpus_manifest = json.loads((corpus_dir / "corpus-manifest.json").read_text())
    records = []
    shards = {}
    for rec in manifest["splits"]["val"]["shards"]:
        starts = np.fromfile(data_dir / rec["index"], dtype=np.uint64).astype(np.int64)
        if not len(starts) or starts[0] != 0 or np.any(np.diff(starts) <= 0) or starts[-1] >= rec["tokens"]:
            raise ValueError("invalid validation document index")
        ends = np.r_[starts[1:], rec["tokens"]]
        shards[rec["file"]] = np.memmap(data_dir / rec["file"], mode="r", dtype=np.uint16)
        records.extend({"shard": rec["file"], "doc_start": int(start), "doc_end": int(end)}
                       for start, end in zip(starts, ends))
    rows, hasher = [], hashlib.sha256()
    with (corpus_dir / "val.jsonl").open("rb") as stream:
        for raw in stream:
            hasher.update(raw)
            row = json.loads(raw)
            if row["text"]:
                rows.append(row)
    if hasher.hexdigest() != corpus_manifest["files"]["val.jsonl"]["sha256"]:
        raise ValueError("validation JSON does not match its pinned corpus manifest")
    if len(rows) != len(records):
        raise ValueError("validation JSON and shard document counts differ")
    span = block_size + 1
    code, prose = defaultdict(list), []
    for index, (row, rec) in enumerate(zip(rows, records)):
        if rec["doc_end"] - rec["doc_start"] < 2 * span:
            continue
        if row["source"] == "bigcode/the-stack-dedup" and row["group"].startswith("stack:"):
            code[row["group"]].append(index)
        elif row["source"] == "rfc-nist-prose":
            prose.append(index)
    if len(code) < code_documents or len(prose) < prose_documents:
        raise ValueError("not enough eligible validation repositories or prose documents")
    rng = random.Random(seed)
    selected = [("code", rng.choice(code[group])) for group in rng.sample(sorted(code), code_documents)]
    selected += [("prose", index) for index in rng.sample(prose, prose_documents)]
    windows = []
    for source, index in selected:
        row, rec = rows[index], records[index]
        encoded = tokenizer.encode(row["text"]) + [tokenizer.special_id(ENDOFTEXT)]
        stored = shards[rec["shard"]][rec["doc_start"]:rec["doc_end"]]
        if not np.array_equal(np.asarray(encoded, dtype=np.int64), stored):
            raise ValueError("selected JSON document does not match the stored token sequence")
        offset = rng.randrange(len(encoded) - block_size)
        group_sha = hashlib.sha256(row["group"].encode()).hexdigest()
        for region, relative in (("prefix", 0), ("uniform", offset)):
            windows.append({"id": f"{source}-{index}-{region}", "stratum": f"{source}_{region}",
                            "group_sha256": group_sha, "document_index": index,
                            **rec, "start": rec["doc_start"] + relative,
                            "targets_beyond_legacy_prefix": min(block_size, relative)})
    return {"schema": SCHEMA, "shard_manifest_sha256": manifest_sha,
            "tokenizer_sha256": manifest["tokenizer_sha256"],
            "corpus_manifest_sha256": sha256_file(corpus_dir / "corpus-manifest.json"),
            "validation_json_sha256": hasher.hexdigest(), "block_size": block_size, "seed": seed,
            "eligibility_min_tokens": 2 * span, "code_repositories": code_documents,
            "prose_documents": prose_documents, "windows": windows,
            "selection": "one file per seeded distinct Stack repository; seeded long prose documents; uniform legal in-document offsets",
            "private_holdout_read": False, "test_split_read": False,
            "limitations": ["Validation belongs to the existing public corpus and is not final qualification.",
                            "Code windows are repository-balanced, not globally token-weighted.",
                            "Prose chunks may share documents/groups; prose uncertainty is not treated as independent."]}


def validate(plan, manifest):
    if plan.get("schema") != SCHEMA or plan.get("block_size", 0) < 1:
        raise ValueError("unsupported or invalid validation plan")
    span = plan["block_size"] + 1
    records = {rec["file"]: rec for rec in manifest["splits"]["val"]["shards"]}
    identities, groups, documents, pairs = set(), defaultdict(set), defaultdict(set), {}
    counts = defaultdict(int)
    for window in plan["windows"]:
        if window["id"] in identities or window["stratum"] not in {"code_prefix", "code_uniform", "prose_prefix", "prose_uniform"}:
            raise ValueError("duplicate window identity or invalid stratum")
        identities.add(window["id"])
        rec = records.get(window["shard"])
        if rec is None or not 0 <= window["doc_start"] <= window["start"] < window["doc_end"] <= rec["tokens"]:
            raise ValueError("window document bounds do not match validation shards")
        if window["start"] + span > window["doc_end"]:
            raise ValueError("validation window crosses a document boundary")
        if window["doc_end"] - window["doc_start"] < 2 * span:
            raise ValueError("validation document does not meet the long-document criterion")
        if window["stratum"].endswith("prefix") and window["start"] != window["doc_start"]:
            raise ValueError("prefix window is not a document prefix")
        counts[window["stratum"]] += 1
        groups[window["stratum"]].add(window["group_sha256"])
        documents[window["stratum"]].add(window["document_index"])
        pair = (window["stratum"].split("_")[0], window["document_index"])
        identity = (window["shard"], window["doc_start"], window["doc_end"], window["group_sha256"])
        if pair in pairs and pairs[pair] != identity:
            raise ValueError("paired prefix/uniform windows refer to different documents")
        pairs[pair] = identity
    for source, expected in (("code", plan["code_repositories"]), ("prose", plan["prose_documents"])):
        for region in ("prefix", "uniform"):
            key = f"{source}_{region}"
            if counts[key] != expected or len(documents[key]) != expected:
                raise ValueError("validation stratum size or document uniqueness mismatch")
        if documents[f"{source}_prefix"] != documents[f"{source}_uniform"]:
            raise ValueError("prefix and uniform documents are not paired")
    if len(groups["code_uniform"]) != plan["code_repositories"]:
        raise ValueError("code validation reuses a repository")


def load(path, data_dir, manifest, expected_plan_sha256, expected_manifest_sha256):
    path = Path(path)
    if sha256_file(path) != expected_plan_sha256:
        raise ValueError("validation plan sha256 mismatch")
    plan = json.loads(path.read_text())
    if plan["shard_manifest_sha256"] != expected_manifest_sha256 or plan["tokenizer_sha256"] != manifest["tokenizer_sha256"]:
        raise ValueError("validation plan belongs to a different corpus/tokenizer")
    validate(plan, manifest)
    arrays, boundaries = {}, {}
    for rec in manifest["splits"]["val"]["shards"]:
        arrays[rec["file"]] = np.memmap(Path(data_dir) / rec["file"], mode="r", dtype=np.uint16)
        starts = np.fromfile(Path(data_dir) / rec["index"], dtype=np.uint64).astype(np.int64)
        boundaries[rec["file"]] = dict(zip(starts.tolist(), np.r_[starts[1:], rec["tokens"]].tolist()))
    for window in plan["windows"]:
        if boundaries[window["shard"]].get(window["doc_start"]) != window["doc_end"]:
            raise ValueError("planned document is not in the pinned index")
    return plan, arrays


def main(argv=None):
    parser = argparse.ArgumentParser(description="Freeze source-stratified, repository-balanced validation token windows.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--code-documents", type=int, default=128)
    parser.add_argument("--prose-documents", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise ValueError("validation plans are immutable; use a new output path")
    plan = build(args.data_dir, args.corpus_dir, args.block_size, args.code_documents, args.prose_documents, args.seed)
    manifest, _, _ = load_manifest(args.data_dir, verify=True)
    validate(plan, manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"plan_sha256": sha256_file(out), "windows": len(plan["windows"]),
                      "code_repositories": plan["code_repositories"], "prose_documents": plan["prose_documents"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
