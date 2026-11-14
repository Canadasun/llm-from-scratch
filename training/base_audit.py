import argparse
import ast
import hashlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from data.shards import LEGACY_PREFIX, UNIFORM_WINDOWS, ShardSplit, load_manifest
from special_tokens import ASSISTANT, ENDOFTEXT, USER


def coverage(starts, tokens, block_size):
    if tokens < 1 or block_size < 1:
        raise ValueError("token count and block size must be positive")
    span = block_size + 1
    starts = np.asarray(starts, dtype=np.int64)
    if not len(starts) or starts[0] != 0 or np.any(np.diff(starts) <= 0) or starts[-1] >= tokens:
        raise ValueError("invalid document-start index")
    lengths = np.diff(np.r_[starts, tokens])
    reachable, right = 0, 0
    if tokens >= span:
        for start in np.minimum(starts, tokens - span):
            end = int(start) + span
            reachable += max(0, end - max(right, int(start)))
            right = max(right, end)
    return {"tokens": int(tokens), "documents": len(starts),
            "long_documents": int((lengths > span).sum()),
            "legacy_reachable_tokens": reachable, "legacy_unreachable_tokens": int(tokens) - reachable,
            "uniform_reachable_tokens": int(tokens) if tokens >= span else 0,
            "uniform_legal_windows": max(0, int(tokens) - span + 1)}


def shard_profile(root, manifest, tokenizer, block_size):
    root = Path(root)
    output = {}
    special = {name: tokenizer.special_id(name) for name in (ENDOFTEXT, USER, ASSISTANT)}
    for split in ("train", "val"):
        totals, counts = Counter(), Counter()
        records = []
        for rec in manifest["splits"][split]["shards"]:
            starts = np.fromfile(root / rec["index"], dtype=np.uint64).astype(np.int64)
            tokens = np.memmap(root / rec["file"], mode="r", dtype=np.uint16)
            item = coverage(starts, len(tokens), block_size)
            totals.update(item)
            ends = np.r_[starts[1:], len(tokens)]
            if np.any(tokens[ends - 1] != special[ENDOFTEXT]):
                raise ValueError("document does not end in the pinned EOT token")
            for offset in range(0, len(tokens), 1 << 22):
                chunk = tokens[offset:offset + (1 << 22)]
                if np.any(chunk >= tokenizer.vocab_size):
                    raise ValueError("shard token exceeds the tokenizer vocabulary")
                for name, token in special.items():
                    counts[name] += int(np.count_nonzero(chunk == token))
            records.append({"file": rec["file"], **item})
        total = dict(totals)
        total["legacy_unreachable_fraction"] = total["legacy_unreachable_tokens"] / total["tokens"]
        total["special_token_counts"] = dict(counts)
        total["shards"] = records
        output[split] = total
    return output


def corpus_profile(root, tokenizer, block_size, samples=512, seed=1729):
    root = Path(root)
    manifest = json.loads((root / "corpus-manifest.json").read_text())
    output, group_sets = {}, {}
    for split in ("train", "val"):
        rows, empty, source_rows, source_bytes, licences, groups = 0, 0, Counter(), Counter(), Counter(), Counter()
        reservoir, eligible = [], 0
        generator = random.Random(seed)
        hasher = hashlib.sha256()
        filename = f"{split}.jsonl"
        with (root / filename).open("rb") as stream:
            for raw in stream:
                hasher.update(raw)
                row = json.loads(raw)
                text = row["text"]
                rows += 1
                empty += not bool(text)
                source_rows[row["source"]] += 1
                source_bytes[row["source"]] += len(text.encode("utf-8"))
                licences.update(row["licences"])
                groups[row["group"]] += 1
                if row["source"] == "rfc-nist-prose" or not text:
                    continue
                eligible += 1
                if len(reservoir) < samples:
                    reservoir.append(text)
                else:
                    index = generator.randrange(eligible)
                    if index < samples:
                        reservoir[index] = text
                if rows % 100000 == 0:
                    print(json.dumps({"stage": "corpus_profile", "split": split, "rows": rows}), flush=True)
        if hasher.hexdigest() != manifest["files"][filename]["sha256"]:
            raise ValueError(f"{filename} does not match its corpus manifest")
        group_sets[split] = set(groups)
        syntax, features, imports = Counter(), Counter(), Counter()
        for text in reservoir:
            try:
                tree = ast.parse(text)
            except (SyntaxError, ValueError, RecursionError, MemoryError) as error:
                syntax[type(error).__name__] += 1
                continue
            syntax["parsed"] += 1
            prefix = tokenizer.decode(tokenizer.encode(text)[:block_size + 1])
            prefix_line = prefix.count("\n") + 1
            nodes = list(ast.walk(tree))
            for name, types in (("functions", (ast.FunctionDef, ast.AsyncFunctionDef)),
                                ("returns_and_raises", (ast.Return, ast.Raise))):
                selected = [node for node in nodes if isinstance(node, types)]
                features[name] += len(selected)
                features[name + "_starting_beyond_prefix"] += sum(node.lineno > prefix_line for node in selected)
            roots = set()
            for node in nodes:
                if isinstance(node, ast.Import):
                    roots.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    roots.add(node.module.split(".")[0])
            for module in ("hashlib", "hmac", "ssl", "secrets", "cryptography", "sqlite3", "subprocess", "pytest", "unittest"):
                imports[module] += module in roots
        output[split] = {"rows": rows, "empty_rows": empty, "content_sha256": hasher.hexdigest(),
                         "source_rows": dict(source_rows), "source_bytes": dict(source_bytes),
                         "licence_metadata_rows": dict(licences), "metadata_groups": len(groups),
                         "largest_group_row_fraction": max(groups.values()) / rows,
                         "top10_groups_row_fraction": sum(sorted(groups.values(), reverse=True)[:10]) / rows,
                         "code_sample": {"seed": seed, "eligible_files": eligible, "sample_files": len(reservoir),
                                         "python_version": sys.version.split()[0], "syntax": dict(syntax),
                                         "ast_features": dict(features), "imported_modules_file_counts": dict(imports)}}
    return {"splits": output, "train_validation_group_overlap": len(group_sets["train"] & group_sets["val"]),
            "private_holdout_read": False, "test_split_read": False,
            "limitations": ["Downloaded code is parsed as data, never executed.",
                            "Stack groups are repositories; CPython groups are top-level modules and prose groups are chunk batches.",
                            "Static imports and parseable syntax do not establish functional or security quality.",
                            "AST feature estimates are from a file-uniform code sample, not token-weighted."]}


def measure_loss(data_dir, ckpt, expected_sha256, pairs=24, block_size=1024, seed=1729):
    import torch
    from evals.loader import load_checkpoint
    from train import file_sha256
    if file_sha256(ckpt) != expected_sha256:
        raise ValueError("checkpoint does not match the pinned base")
    torch.set_num_threads(4)
    manifest, digest, _ = load_manifest(data_dir, verify=True)
    model, _, cfg = load_checkpoint(str(ckpt), "cpu")
    if block_size > cfg["block_size"]:
        raise ValueError("diagnostic context exceeds the model context")
    model.eval()
    output = {}
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for split in ("train", "val"):
            data = ShardSplit(data_dir, split, manifest)
            modes = {}
            for mode in (LEGACY_PREFIX, UNIFORM_WINDOWS):
                generator.manual_seed(seed)
                losses = []
                for index in range(pairs):
                    x, y = data.sample(1, block_size, generator, sampling_mode=mode)
                    _, loss = model(x, y)
                    losses.append(loss.item())
                    if (index + 1) % 8 == 0:
                        print(json.dumps({"stage": "loss", "split": split, "mode": mode, "windows": index + 1}), flush=True)
                average = sum(losses) / len(losses)
                variance = sum((loss - average) ** 2 for loss in losses) / max(1, len(losses) - 1)
                modes[mode] = {"windows": pairs, "tokens_per_window": block_size, "mean_loss_nats": average,
                               "perplexity": math.exp(average), "window_loss_standard_error": math.sqrt(variance / pairs),
                               "window_losses": losses}
            output[split] = modes
    if file_sha256(ckpt) != expected_sha256:
        raise ValueError("base checkpoint changed during read-only diagnosis")
    return {"checkpoint_sha256": expected_sha256, "shard_manifest_sha256": digest, "seed": seed,
            "device": "cpu", "precision": "float32", "torch_version": str(torch.__version__), "splits": output,
            "limitations": ["Fixed diagnostic windows, not a capability benchmark or a causal training ablation.",
                            "The two distributions sample different content; their loss gap alone cannot prove a cause.",
                            "Historical prefix losses and uniform-window losses are not interchangeable baselines."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Audit token exposure and base losses without executing corpus code or training.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--corpus-dir")
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--ckpt")
    parser.add_argument("--expect-checkpoint-sha256")
    parser.add_argument("--windows", type=int, default=24)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if min(args.block_size, args.samples, args.windows) < 1:
        parser.error("block size, samples and windows must be positive")
    if args.ckpt:
        if not args.expect_checkpoint_sha256:
            parser.error("checkpoint measurement requires its pinned sha256")
        report = measure_loss(args.data_dir, args.ckpt, args.expect_checkpoint_sha256, args.windows, args.block_size)
    else:
        manifest, digest, tokenizer = load_manifest(args.data_dir, verify=True)
        report = {"shard_manifest_sha256": digest, "block_size": args.block_size,
                  "coverage": shard_profile(args.data_dir, manifest, tokenizer, args.block_size)}
        if args.corpus_dir:
            report["corpus"] = corpus_profile(args.corpus_dir, tokenizer, args.block_size, args.samples)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"out": args.out, "completed": True}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
