"""Response-only instruction tuning of this project's own pretrained weights.

Writes candidate checkpoints only. Benchmark examples are never training input.
This starts a new optimizer; it is not an exact resume of inference-only weights.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch

from evals.evidence import file_sha256
from evals.loader import load_checkpoint
from train import atomic_save, pick_device


def encode_examples(path, tokenizer, context):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    encoded = []
    for row in rows:
        prompt = tokenizer.encode(row['prompt'])
        response = tokenizer.encode(row['response'])
        if not prompt or not response or len(prompt) + len(response) > context:
            raise ValueError(f"example {row['id']} is empty or exceeds context; no silent truncation")
        full = torch.tensor(prompt + response, dtype=torch.long)
        targets = full[1:].clone()
        targets[:len(prompt) - 1] = -100
        encoded.append((full[:-1], targets))
    if not encoded:
        raise ValueError("empty instruction dataset")
    return rows, encoded


def batch(examples, batch_size, rng, device):
    indexes = torch.randint(len(examples), (batch_size,), generator=rng).tolist()
    selected = [examples[i] for i in indexes]
    width = max(len(x) for x, _ in selected)
    x = torch.zeros((batch_size, width), dtype=torch.long)
    y = torch.full((batch_size, width), -100, dtype=torch.long)
    for index, (tokens, targets) in enumerate(selected):
        x[index, :len(tokens)] = tokens
        y[index, :len(targets)] = targets
    return x.to(device), y.to(device)


@torch.no_grad()
def response_loss(model, examples, device):
    training = model.training
    model.eval()
    total, count = 0.0, 0
    try:
        for x, y in examples:
            _, loss = model(x[None].to(device), y[None].to(device))
            tokens = (y != -100).sum().item()
            total += loss.item() * tokens
            count += tokens
    finally:
        model.train(training)
    return total / count


def tune(ckpt, data, output, steps=160, batch_size=2, learning_rate=3e-5,
         context=192, device='cpu', threads=2, seed=2026, replay_path=None):
    if steps < 1 or batch_size < 1 or threads < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("positive steps, batch size, threads, and learning rate required")
    if Path(output).resolve() == Path(ckpt).resolve():
        raise ValueError("candidate output must not overwrite the base checkpoint")
    if Path(output).exists():
        raise ValueError("candidate output exists; choose a new run directory")
    root = Path(data)
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    device = pick_device(device)
    model, tokenizer, cfg = load_checkpoint(ckpt, device)
    if context > cfg['block_size'] or context < 2:
        raise ValueError("invalid training context")
    training_rows, training = encode_examples(root/'train.jsonl', tokenizer, context)
    validation_rows, validation = encode_examples(root/'validation.jsonl', tokenizer, context)
    if {r['family'] for r in training_rows} & {r['family'] for r in validation_rows}:
        raise ValueError("training and validation exercise families overlap")
    if {r['prompt'].strip() for r in training_rows} & {r['prompt'].strip() for r in validation_rows}:
        raise ValueError("training and validation prompts overlap")
    # Remove the source checkpoint from consideration as a training destination.
    source_hash = file_sha256(ckpt)
    base_loss = response_loss(model, validation, device)
    print(f"Base held-out response loss: {base_loss:.4f}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=.01)
    rng = torch.Generator().manual_seed(seed)
    replay = None
    if replay_path:
        text = Path(replay_path).read_text()[:300000]
        replay = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        if len(replay) <= context:
            raise ValueError("replay corpus too short")
    best, history = base_loss, []
    metadata = {"base_checkpoint_sha256": source_hash, "method": "response-only supervised instruction tuning",
                "train_sha256": file_sha256(root/'train.jsonl'),
                "validation_sha256": file_sha256(root/'validation.jsonl'),
                "replay_sha256": file_sha256(replay_path) if replay_path else None,
                "seed": seed, "context": context, "learning_rate": learning_rate,
                "batch_size": batch_size, "device": device, "torch_version": str(torch.__version__),
                "base_validation_loss": base_loss}
    started = time.monotonic()
    model.train()
    for step in range(1, steps + 1):
        if replay is not None and step % 5 == 0:
            offsets = torch.randint(len(replay) - context, (batch_size,), generator=rng).tolist()
            x = torch.stack([replay[i:i+context] for i in offsets]).to(device)
            y = torch.stack([replay[i+1:i+context+1] for i in offsets]).to(device)
        else:
            x, y = batch(training, batch_size, rng, device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite training loss; candidate not promoted")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step % 40 == 0 or step == steps:
            val_loss = response_loss(model, validation, device)
            record = {"step": step, "train_loss": loss.item(), "validation_loss": val_loss,
                      "elapsed_s": round(time.monotonic() - started, 2)}
            history.append(record)
            print(json.dumps(record), flush=True)
            if val_loss < best:
                best = val_loss
                atomic_save({"model_state": model.state_dict(), "model_cfg": cfg,
                             "tokenizer": tokenizer.to_dict(),
                             "instruction_tuning": {**metadata, "step": step, "validation_loss": val_loss}}, output)
            atomic_save({"model_state": model.state_dict(), "model_cfg": cfg,
                         "tokenizer": tokenizer.to_dict(), "optimizer_state": optimizer.state_dict(),
                         "rng_state": torch.get_rng_state(), "batch_rng_state": rng.get_state(),
                         "instruction_tuning": {**metadata, "step": step, "validation_loss": val_loss}},
                        str(Path(output).with_name('last.pt')))
            Path(output).with_suffix('.training.json').write_text(json.dumps(
                {**metadata, "best_validation_loss": best, "history": history}, indent=2)+'\n')
    if file_sha256(ckpt) != source_hash:
        raise ValueError("base checkpoint changed during training")
    return {"base_validation_loss": base_loss, "best_validation_loss": best,
            "candidate_saved": best < base_loss, "history": history}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='checkpoints/ckpt.pt')
    parser.add_argument('--data', default='data/instructions')
    parser.add_argument('--output', default='checkpoints/instruction-candidate/ckpt.pt')
    parser.add_argument('--steps', type=int, default=160)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=3e-5)
    parser.add_argument('--context', type=int, default=192)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--replay', default='data/corpus.txt')
    args = parser.parse_args()
    print(json.dumps(tune(args.ckpt, args.data, args.output, args.steps, args.batch_size,
                          args.learning_rate, args.context, args.device, args.threads,
                          seed=args.seed, replay_path=args.replay)))
