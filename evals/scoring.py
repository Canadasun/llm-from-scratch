"""
Scoring primitives shared by every eval.

The important one is `completion_nll`: the negative log-likelihood the model
assigns to a candidate continuation. Likelihood scoring is how base (non-chat)
models are evaluated on multiple-choice tasks — you don't ask the model to
*say* the answer, you ask which answer it finds least surprising. That gives a
real signal even from a small model that cannot yet write fluent answers.
"""

import math

import torch
import torch.nn.functional as F

from evals.loader import safe_encode


@torch.no_grad()
def completion_nll(model, tokenizer, prompt, completion, device="cpu"):
    """Negative log-likelihood (in nats) the model assigns to `completion`
    following `prompt`.

    Returns (nll_sum, n_tokens). Length-normalised score is nll_sum / n_tokens;
    lower means the model finds that continuation more likely.
    """
    block_size = model.cfg.block_size
    prompt_ids = safe_encode(tokenizer, prompt)
    comp_ids = safe_encode(tokenizer, completion)
    n_comp = len(comp_ids)
    if n_comp == 0:
        return math.inf, 0

    full = prompt_ids + comp_ids
    # The model sees at most block_size tokens; keep the whole completion at the
    # tail and crop the prompt from the left if the pair is too long.
    if len(full) > block_size + 1:
        full = full[-(block_size + 1):]
    if len(full) < 2:
        # Need at least one context token to predict the first completion token.
        full = safe_encode(tokenizer, "\n") + full
        if len(full) < 2:
            return math.inf, n_comp
    n_comp = min(n_comp, len(full) - 1)

    x = torch.tensor([full[:-1]], dtype=torch.long, device=device)
    y = torch.tensor(full[1:], dtype=torch.long, device=device)
    logits, _ = model(x)                       # (1, T, V)
    logits_comp = logits[0, -n_comp:, :]       # only the completion positions
    y_comp = y[-n_comp:]
    nll = F.cross_entropy(logits_comp, y_comp, reduction="sum")
    return nll.item(), n_comp


def choose_by_likelihood(model, tokenizer, prompt, choices, device="cpu"):
    """Return (best_index, per_choice_mean_nll) picking the least-surprising
    choice (length-normalised, so longer answers aren't unfairly penalised)."""
    means = []
    for ch in choices:
        nll_sum, n = completion_nll(model, tokenizer, prompt, ch, device)
        means.append(nll_sum / n if n else math.inf)
    best = min(range(len(means)), key=lambda i: means[i])
    return best, means


@torch.no_grad()
def corpus_metrics(model, tokenizer, text, device="cpu"):
    """Intrinsic quality on held-out `text`: token perplexity, bits-per-byte and
    next-token top-1 / top-5 accuracy. bits-per-byte is tokenizer-independent, so
    it is the fair number to compare across models or tokenizers.
    """
    block_size = model.cfg.block_size
    ids = safe_encode(tokenizer, text)
    total_nll = 0.0
    total_tokens = 0
    top1 = 0
    top5 = 0

    # Non-overlapping windows of up to block_size+1 tokens.
    step = block_size
    for start in range(0, max(1, len(ids) - 1), step):
        window = ids[start:start + block_size + 1]
        if len(window) < 2:
            break
        x = torch.tensor([window[:-1]], dtype=torch.long, device=device)
        y = torch.tensor(window[1:], dtype=torch.long, device=device)
        logits, _ = model(x)
        lg = logits[0]                          # (T, V)
        total_nll += F.cross_entropy(lg, y, reduction="sum").item()
        total_tokens += y.numel()
        top1 += (lg.argmax(-1) == y).sum().item()
        k = min(5, lg.size(-1))
        top5 += (lg.topk(k, dim=-1).indices == y.unsqueeze(-1)).any(-1).sum().item()

    if total_tokens == 0:
        return {"tokens": 0}
    mean_nll = total_nll / total_tokens
    n_bytes = max(1, len(text.encode("utf-8")))
    return {
        "tokens": total_tokens,
        "bytes": n_bytes,
        "perplexity": math.exp(mean_nll),
        "bits_per_byte": (total_nll / math.log(2)) / n_bytes,
        "top1_acc": top1 / total_tokens,
        "top5_acc": top5 / total_tokens,
    }


@torch.no_grad()
def generate_text(model, tokenizer, prompt, max_new_tokens=64,
                  temperature=0.8, top_k=None, device="cpu"):
    ids = safe_encode(tokenizer, prompt) or safe_encode(tokenizer, "\n")
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k)
    full = tokenizer.decode(out[0].tolist())
    # Return only the newly generated continuation.
    return full[len(tokenizer.decode(ids)):]
