"""
Generate text from a trained checkpoint (Milestone 1).

Run:
    python sample.py --prompt "Security " --tokens 400
"""

import argparse
import os

import torch

from model import GPT, GPTConfig
from tokenizer import CharTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join("checkpoints", "ckpt.pt"))
    ap.add_argument("--prompt", default="\n")
    ap.add_argument("--tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=None)
    args = ap.parse_args()

    if not os.path.exists(args.ckpt):
        raise SystemExit(f"Checkpoint {args.ckpt} not found. Train first: python train.py")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    tokenizer = CharTokenizer(ckpt["chars"])
    model = GPT(GPTConfig(**ckpt["model_cfg"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Any prompt character the model never saw is dropped (char-level limitation).
    prompt = "".join(c for c in args.prompt if c in tokenizer.stoi) or "\n"
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)

    out = model.generate(idx, args.tokens, temperature=args.temperature, top_k=args.top_k)
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
