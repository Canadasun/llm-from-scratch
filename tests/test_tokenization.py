"""Tests for the tokenizer factory and checkpoint-embedded tokenizers."""

import torch

import tokenization
from model import GPT, GPTConfig


def test_build_char():
    tok = tokenization.build("char", "TLS handshake\n")
    assert tok.kind == "char"
    assert tok.decode(tok.encode("TLS")) == "TLS"


def test_build_bpe():
    text = "the handshake authenticates the server " * 30
    tok = tokenization.build("bpe", text, vocab_size=300, bpe_train_chars=None)
    assert tok.kind == "bpe"
    assert tok.vocab_size <= 300
    assert tok.decode(tok.encode("the server")) == "the server"


def test_build_unknown_kind_raises():
    try:
        tokenization.build("nope", "x")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_from_dict_roundtrip_char():
    tok = tokenization.build("char", "abcABC 123\n")
    back = tokenization.from_dict(tok.to_dict())
    assert back.kind == "char"
    assert back.encode("abc") == tok.encode("abc")


def test_from_dict_roundtrip_bpe():
    tok = tokenization.build("bpe", "crypto crypto handshake handshake " * 20,
                             vocab_size=300, bpe_train_chars=None)
    back = tokenization.from_dict(tok.to_dict())
    assert back.kind == "bpe"
    assert back.encode("crypto handshake") == tok.encode("crypto handshake")


def test_from_dict_unknown_kind_raises():
    try:
        tokenization.from_dict({"kind": "mystery"})
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_checkpoint_roundtrip_like_train_and_sample(tmp_path):
    """Mirror what train.py saves and sample.py loads, for a BPE tokenizer."""
    tok = tokenization.build("bpe", "secure channel secure channel " * 30,
                             vocab_size=300, bpe_train_chars=None)
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=16, n_layer=1,
                    n_head=2, n_embd=16)
    model = GPT(cfg)
    ckpt = tmp_path / "ckpt.pt"
    torch.save({"model_state": model.state_dict(),
                "model_cfg": cfg.__dict__,
                "tokenizer": tok.to_dict()}, ckpt)

    blob = torch.load(ckpt, weights_only=True)   # secure load, as in sample.py
    tok2 = tokenization.from_dict(blob["tokenizer"])
    model2 = GPT(GPTConfig(**blob["model_cfg"]))
    model2.load_state_dict(blob["model_state"])
    assert tok2.encode("secure channel") == tok.encode("secure channel")
