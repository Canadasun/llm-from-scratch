"""
Generate text from a trained checkpoint (Milestone 1).

Run:
    python sample.py --prompt "Security " --tokens 400
"""

import argparse
import os

import torch

import tokenization
from model import GPT, GPTConfig
from safety import content_policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join("checkpoints", "ckpt.pt"))
    ap.add_argument("--prompt", default="\n")
    ap.add_argument("--tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--no-safety", action="store_true",
                    help="disable the use-safety policy (research use; logged)")
    args = ap.parse_args()

    if not os.path.exists(args.ckpt):
        raise SystemExit(f"Checkpoint {args.ckpt} not found. Train first: python train.py")

    # Use-safety: check the prompt before generating.
    if not args.no_safety:
        d = content_policy.check(args.prompt)
        if d.action == "block":
            raise SystemExit(f"[safety] prompt refused: {d.reason} [{d.category}]")
        if d.action == "flag":
            print(content_policy.caution_text(d))
    else:
        print("[safety] policy DISABLED via --no-safety (research mode)")

    # Secure load: weights_only=True so a malicious checkpoint cannot execute
    # arbitrary code via pickle. Checkpoints here hold only tensors + plain data.
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    tokenizer = tokenization.from_dict(ckpt["tokenizer"])
    model = GPT(GPTConfig(**ckpt["model_cfg"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    prompt = args.prompt
    if tokenizer.kind == "char":
        # A char tokenizer has no token for a character it never saw; drop those.
        prompt = "".join(c for c in prompt if c in tokenizer.stoi) or "\n"
    ids = tokenizer.encode(prompt)
    if not ids:
        ids = tokenizer.encode("\n")
    idx = torch.tensor([ids], dtype=torch.long)

    out = model.generate(idx, args.tokens, temperature=args.temperature, top_k=args.top_k)
    text = tokenizer.decode(out[0].tolist())

    # Use-safety: check the generation before showing it.
    if not args.no_safety:
        d = content_policy.check(text)
        if d.action == "block":
            raise SystemExit(f"[safety] generation withheld: {d.reason} [{d.category}]")
        if d.action == "flag":
            print(content_policy.caution_text(d))

    print(text)


if __name__ == "__main__":
    main()
