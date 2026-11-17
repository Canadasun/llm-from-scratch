"""Pinned inference and isolated scoring of the original format probe."""

import argparse
import hashlib
import json
import platform
from pathlib import Path

from data.format_probe_tasks import TASKS
from evals.format_probe import FORMATS, build, prompt, score_response
from evals.code_executor import sandbox_preflight
from special_tokens import ASSISTANT, ENDOFTEXT, USER

BASE_SHA = "1164213e635d376946ec331d5ff238cf89f492788554a953010f59bd5f57a593"
TOKENIZER_SHA = "bcefb776782e7c728c19a39376746bf94816700f9ce2c4b4513a09122d5e10ac"


def tokenizer_digest(state):
    return hashlib.sha256(json.dumps(state, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def finish_tokens(ids, stops, budget):
    if not ids or len(ids) > budget:
        raise ValueError("invalid generated length")
    terminal = ids[-1] if ids[-1] in stops else None
    if any(token in stops for token in ids[:-1]):
        raise ValueError("generation continued beyond stop")
    if terminal is None and len(ids) != budget:
        raise ValueError("early termination without a recorded stop")
    return {"generated_tokens": len(ids), "response_tokens": len(ids) - int(terminal is not None),
            "stop_reason": "special_token" if terminal is not None else "token_budget",
            "terminal_token_id": terminal}


def generate(checkpoint, expected, output, device):
    import torch
    from evals.loader import load_checkpoint
    from training.sampler_control import deterministic_runtime
    from train import file_sha256
    if expected != BASE_SHA or file_sha256(checkpoint) != expected:
        raise ValueError("measurement requires the unchanged accepted base")
    suite = build()
    with deterministic_runtime():
        model, tokenizer, _ = load_checkpoint(checkpoint, device)
        model.eval()
        if tokenizer_digest(tokenizer.to_dict()) != TOKENIZER_SHA:
            raise ValueError("accepted tokenizer changed")
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).with_name("tokenizer.json").write_text(json.dumps(tokenizer.to_dict(), sort_keys=True) + "\n")
        stops = sorted(tokenizer.special_id(name) for name in (ENDOFTEXT, USER, ASSISTANT))
        budget = 384
        result = {"schema": "format_probe_responses_v1", "meta": {
            "checkpoint_sha256": expected, "probe_sha256": digest(suite),
            "source_sha256": suite["source_sha256"], "max_new_tokens": budget,
            "context": model.cfg.block_size, "vocab_size": tokenizer.vocab_size,
            "decoding": "temperature_1_top_k_1", "seed_per_response": 1337,
            "wrapper": "user_prompt_assistant", "stop_ids": stops,
            "torch": str(torch.__version__), "python": platform.python_version(),
            "device": str(device), "precision": "float32", "truncation": "forbidden",
            "backend": "deterministic_math_sdpa"}, "responses": {}}
        for task in TASKS:
            for mode in FORMATS:
                key = f"{task.family}:{mode}"
                text = f"{USER}{prompt(task, mode)}{ASSISTANT}"
                ids = tokenizer.encode(text, allowed_special="all")
                if len(ids) + budget > model.cfg.block_size:
                    raise ValueError("prompt plus generation exceeds context; no silent truncation")
                torch.manual_seed(1337)
                idx = torch.tensor([ids], dtype=torch.long, device=device)
                with torch.no_grad():
                    out = model.generate(idx, budget, temperature=1.0, top_k=1, stop_ids=stops)
                generated = out[0, len(ids):].tolist()
                end = finish_tokens(generated, stops, budget)
                visible = generated[:end["response_tokens"]]
                result["responses"][key] = {"response": tokenizer.decode(visible),
                    "token_ids": generated, "prompt_tokens": len(ids),
                    "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(), **end}
                Path(output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
                print(json.dumps({"finished": key, **end}), flush=True)
        if file_sha256(checkpoint) != expected:
            raise ValueError("accepted checkpoint changed")


def validate(saved, tokenizer_state):
    from tokenization import from_dict
    if tokenizer_digest(tokenizer_state) != TOKENIZER_SHA:
        raise ValueError("tokenizer mismatch")
    tokenizer = from_dict(tokenizer_state)
    suite = build()
    meta = saved["meta"]
    if saved["schema"] != "format_probe_responses_v1" or meta["checkpoint_sha256"] != BASE_SHA:
        raise ValueError("wrong schema or checkpoint")
    if meta["probe_sha256"] != digest(suite) or meta["source_sha256"] != suite["source_sha256"]:
        raise ValueError("probe changed")
    if (meta["max_new_tokens"] != 384 or meta["decoding"] != "temperature_1_top_k_1"
            or meta["seed_per_response"] != 1337 or meta["wrapper"] != "user_prompt_assistant"
            or meta["truncation"] != "forbidden" or meta["context"] != 1024
            or meta["vocab_size"] != tokenizer.vocab_size or meta["precision"] != "float32"
            or meta["backend"] != "deterministic_math_sdpa"
            or meta["stop_ids"] != sorted(tokenizer.special_id(t) for t in (ENDOFTEXT, USER, ASSISTANT))):
        raise ValueError("inference recipe changed")
    expected = {r["id"] for r in suite["rows"]}
    if set(saved["responses"]) != expected:
        raise ValueError("incomplete or unexpected task membership")
    for task in TASKS:
        for mode in FORMATS:
            row = saved["responses"][f"{task.family}:{mode}"]
            text = f"{USER}{prompt(task, mode)}{ASSISTANT}"
            if row["prompt_sha256"] != hashlib.sha256(text.encode()).hexdigest():
                raise ValueError("prompt mismatch")
            if row["prompt_tokens"] != len(tokenizer.encode(text, allowed_special="all")):
                raise ValueError("prompt token count mismatch")
            ids = row["token_ids"]
            if any(type(token) is not int or not 0 <= token < meta["vocab_size"] for token in ids):
                raise ValueError("invalid token ID")
            if not isinstance(row["response"], str) or type(row["prompt_tokens"]) is not int or row["prompt_tokens"] < 1:
                raise ValueError("invalid response or prompt count")
            if row["prompt_tokens"] + meta["max_new_tokens"] > meta["context"]:
                raise ValueError("context overflow")
            if any(row[key] != value for key, value in finish_tokens(ids, meta["stop_ids"], 384).items()):
                raise ValueError("stop metadata mismatch")
            if tokenizer.decode(ids[:row["response_tokens"]]) != row["response"]:
                raise ValueError("response text disagrees with recorded tokens")
    return suite


def score(saved, tokenizer_state):
    validate(saved, tokenizer_state)  # Check all rows before any execution.
    preflight = sandbox_preflight()
    if not preflight["available"]:
        raise RuntimeError("sandbox unavailable; no host fallback")
    rows = {}
    for task in TASKS:
        for mode in FORMATS:
            key = f"{task.family}:{mode}"
            rows[key] = {"family": task.family, "split": task.split, "mode": mode,
                         **score_response(task, mode, saved["responses"][key]["response"])}
    summary = {}
    for split in ("train", "evaluation"):
        for mode in FORMATS:
            selected = [r for r in rows.values() if r["split"] == split and r["mode"] == mode]
            summary[f"{split}-{mode}"] = {"tasks": len(selected), **{
                name: sum(row[name] for row in selected) for name in ("syntax", "format", "functional")}}
    return {"schema": "format_probe_scores_v1", "meta": saved["meta"],
            "responses_sha256": digest(saved), "sandbox": preflight,
            "summary": summary, "results": rows,
            "scope": "Observed public format diagnostic; no qualification or training authorization."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("generate", "score"))
    parser.add_argument("--checkpoint")
    parser.add_argument("--expect-checkpoint-sha256", default=BASE_SHA)
    parser.add_argument("--responses")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()
    if args.command == "generate":
        generate(args.checkpoint, args.expect_checkpoint_sha256, args.out, args.device)
    else:
        result = score(json.loads(Path(args.responses).read_text()),
                       json.loads(Path(args.responses).with_name("tokenizer.json").read_text()))
        Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
