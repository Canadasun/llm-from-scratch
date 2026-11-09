"""
A GPT-style decoder-only transformer, written from scratch.

Nothing here is imported from a model library: the attention, the transformer
block, the embeddings and the final language-model head are all implemented by
hand with plain PyTorch tensor ops. The point of Milestone 1 is to understand
every line, so the code favours clarity over cleverness.

Shapes use these letters in comments:
    B = batch size
    T = time steps (tokens in the context window, <= block_size)
    C = channels / embedding dimension (n_embd)
    H = number of attention heads (n_head)
"""

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int          # number of distinct tokens
    block_size: int          # maximum context length
    n_layer: int = 4         # number of transformer blocks
    n_head: int = 4          # number of attention heads
    n_embd: int = 128        # embedding dimension (must be divisible by n_head)
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal (look-back-only) mask.

    Each token produces a query, a key and a value. A token's new
    representation is a weighted average of the values of every token at or
    before it, where the weights come from query-key similarity. The causal
    mask is what makes this a *language model*: position t may never attend to
    a position in its future.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head

        # One linear layer produces query, key and value together (3 * n_embd).
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        # Projection applied after the heads are recombined.
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)

        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        self.dropout = cfg.dropout
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size),
                             persistent=False)

    def forward(self, x, layer_cache=None):
        B, T, C = x.shape

        # Project to q, k, v, then split the last dim into three C-sized chunks.
        q, k, v = self.qkv(x).split(C, dim=2)

        # Reshape each into (B, H, T, head_dim) so heads run in parallel.
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        past = 0
        if layer_cache is not None:
            if layer_cache.get("k") is not None:
                past = layer_cache["k"].size(2)
                k = torch.cat([layer_cache["k"], k], dim=2)
                v = torch.cat([layer_cache["v"], v], dim=2)
            layer_cache["k"], layer_cache["v"] = k, v
        total = past + T
        mask = self.mask[:, :, past:total, :total]

        if attention_backend() == "sdpa":
            dropout_p = self.dropout if self.training else 0.0
            if past == 0:
                y = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)
            elif T == 1:
                y = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
            else:
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p)
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            att = att.masked_fill(~mask, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        # Recombine heads back into (B, T, C).
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))


class MLP(nn.Module):
    """Position-wise feed-forward network: expand 4x, non-linearity, project back."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = F.gelu(self.fc(x))
        x = self.proj(x)
        return self.dropout(x)


class Block(nn.Module):
    """One transformer block: pre-norm attention, then pre-norm MLP, both with
    residual connections. 'Pre-norm' (LayerNorm before the sublayer) is what
    makes deep transformers train stably."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x, layer_cache=None):
        x = x + self.attn(self.ln1(x), layer_cache)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg

        # Token embedding: a lookup table mapping each token id to a vector.
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        # Position embedding: a learned vector per position, so the model knows
        # token order (attention itself is order-agnostic without this).
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)

        # Language-model head: project back to vocabulary-sized logits.
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # Weight tying: share weights between the input embedding and the output
        # head. A standard trick that saves parameters and usually helps.
        self.head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self):
        # nn.Module.parameters() already yields each tensor once, so the tied
        # head/embedding weight is counted a single time here. (Subtracting it
        # again, as this once did, UNDER-counted the model by vocab*n_embd.)
        return sum(p.numel() for p in self.parameters())

    def load_state_dict(self, state_dict, strict=True, assign=False):
        return super().load_state_dict(upgrade_state_dict(state_dict), strict=strict, assign=assign)

    def forward(self, idx, targets=None, cache=None):
        B, T = idx.shape
        past = cache.length if cache is not None else 0
        assert past + T <= self.cfg.block_size, "sequence longer than block_size"

        pos = torch.arange(past, past + T, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)   # (B, T, C)
        x = self.drop(x)
        for i, block in enumerate(self.blocks):
            x = block(x, cache.layers[i] if cache is not None else None)
        if cache is not None:
            cache.length = past + T
        x = self.ln_f(x)
        logits = self.head(x)                         # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Cross-entropy over the flattened (B*T, vocab) logits vs targets.
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None,
                 generator=None, check_cancelled=None, stop_ids=None, use_cache=True):
        """Autoregressive sampling: predict the next token, append it, repeat.

        `generator` is an optional torch.Generator; when given, sampling draws
        from it instead of the global RNG. That keeps a seeded request
        reproducible even when other requests sample concurrently (the global
        RNG is process-wide and would otherwise be a shared, racy resource)."""
        if type(max_new_tokens) is not int or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a non-negative integer")
        self.eval()
        stops = None
        if stop_ids:
            stops = torch.tensor(sorted(set(int(i) for i in stop_ids)), device=idx.device)
        finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
        cache = None
        for _ in range(max_new_tokens):
            if check_cancelled is not None:
                check_cancelled()
            if use_cache and idx.size(1) <= self.cfg.block_size:
                if cache is None:
                    cache = KVCache(len(self.blocks))
                    logits, _ = self(idx, cache=cache)
                else:
                    logits, _ = self(idx[:, -1:], cache=cache)
            else:
                cache = None
                logits, _ = self(idx[:, -self.cfg.block_size:])
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1, generator=generator)
            idx = torch.cat([idx, next_id], dim=1)
            if stops is not None:
                finished |= torch.isin(next_id[:, 0], stops)
                if bool(finished.all()):
                    break
        return idx


class KVCache:
    def __init__(self, n_layer):
        self.layers = [{} for _ in range(n_layer)]
        self.length = 0


_BACKEND = ["sdpa" if hasattr(F, "scaled_dot_product_attention") else "manual"]


def attention_backend():
    return _BACKEND[0]


def set_attention_backend(name):
    if name not in ("sdpa", "manual"):
        raise ValueError(f"unknown attention backend: {name!r}")
    if name == "sdpa" and not hasattr(F, "scaled_dot_product_attention"):
        raise ValueError("this torch build has no scaled_dot_product_attention")
    previous = _BACKEND[0]
    _BACKEND[0] = name
    return previous


def upgrade_state_dict(state_dict):
    return {k: v for k, v in state_dict.items()
            if not (k.endswith(".attn.mask") and k.startswith("blocks."))}
