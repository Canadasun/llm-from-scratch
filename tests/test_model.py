"""Regression tests for the from-scratch GPT model."""

import torch

from model import GPT, GPTConfig


def tiny_cfg(vocab=37, block=16):
    return GPTConfig(vocab_size=vocab, block_size=block, n_layer=2, n_head=2,
                     n_embd=32, dropout=0.0)


def test_forward_shapes_and_loss():
    cfg = tiny_cfg()
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (4, cfg.block_size))
    logits, loss = model(x, x)
    assert logits.shape == (4, cfg.block_size, cfg.vocab_size)
    assert loss.ndim == 0 and loss.item() > 0


def test_forward_without_targets_has_no_loss():
    cfg = tiny_cfg()
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    logits, loss = model(x)
    assert loss is None
    assert logits.shape[0] == 2


def test_causal_mask_blocks_the_future():
    """A token's logits must not change when FUTURE tokens change. If they do,
    the causal mask is broken and the model is cheating by seeing ahead."""
    cfg = tiny_cfg()
    model = GPT(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, cfg.block_size))
    with torch.no_grad():
        base, _ = model(x)
        x2 = x.clone()
        x2[0, -1] = (x2[0, -1] + 1) % cfg.vocab_size  # change only the LAST token
        changed, _ = model(x2)
    # Every position except the last must be identical.
    assert torch.allclose(base[0, :-1], changed[0, :-1], atol=1e-6)
    # The last position is allowed to differ.
    assert not torch.allclose(base[0, -1], changed[0, -1])


def test_generate_respects_block_size_and_length():
    cfg = tiny_cfg(block=8)
    model = GPT(cfg).eval()
    idx = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(idx, max_new_tokens=20)
    assert out.shape == (1, 21)          # original + 20 new
    assert out.max().item() < cfg.vocab_size


def test_generation_is_deterministic_with_seed():
    cfg = tiny_cfg()
    model = GPT(cfg).eval()
    idx = torch.zeros((1, 1), dtype=torch.long)
    torch.manual_seed(0)
    a = model.generate(idx, 15)
    torch.manual_seed(0)
    b = model.generate(idx, 15)
    assert torch.equal(a, b)


def test_param_count_excludes_tied_head():
    cfg = tiny_cfg()
    model = GPT(cfg)
    raw = sum(p.numel() for p in model.parameters())
    assert model.num_params() == raw - model.head.weight.numel()


def test_overfits_tiny_batch():
    """Sanity check that training actually reduces loss (learning works)."""
    cfg = tiny_cfg()
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (4, cfg.block_size))
    y = torch.randint(0, cfg.vocab_size, (4, cfg.block_size))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    first = None
    for _ in range(50):
        _, loss = model(x, y)
        if first is None:
            first = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first  # loss went down
