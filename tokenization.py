"""
Tokenizer factory: one place that knows about both tokenizers.

train.py and sample.py use this so they stay tokenizer-agnostic — the choice of
char vs BPE is a single flag, and a trained checkpoint carries its tokenizer with
it (as a dict) so sampling needs nothing else.
"""

from special_tokens import SPECIAL_TOKENS
from tokenizer import CharTokenizer
from tokenizer_bpe import BPETokenizer
from tokenizer_code import CodeBPETokenizer

_REGISTRY = {"char": CharTokenizer, "bpe": BPETokenizer, "bpe_code": CodeBPETokenizer}


def build_with_specials(kind, text, vocab_size=1024, bpe_train_chars=500_000, verbose=False):
    tok = build(kind, text, vocab_size=max(256, vocab_size - len(SPECIAL_TOKENS)) if kind in ("bpe", "bpe_code") else vocab_size,
                bpe_train_chars=bpe_train_chars, verbose=verbose)
    return tok.add_special_tokens(SPECIAL_TOKENS)


def build(kind, text, vocab_size=1024, bpe_train_chars=500_000, verbose=False):
    """Create and fit a tokenizer of the given kind on `text`.

    - char: vocabulary is every distinct character (vocab_size is ignored).
    - bpe:  learn merges up to `vocab_size`. Because pure-Python BPE training is
            slow, it trains on the first `bpe_train_chars` characters (a
            representative slice) and is then applied to the whole corpus.
    """
    if kind == "char":
        return CharTokenizer.from_text(text)
    if kind == "bpe":
        slice_ = text[:bpe_train_chars] if bpe_train_chars else text
        if verbose:
            print(f"Training BPE (vocab {vocab_size}) on {len(slice_):,} chars...",
                  flush=True)
        return BPETokenizer().train(slice_, vocab_size=vocab_size, verbose=verbose)
    if kind == "bpe_code":
        slice_ = text[:bpe_train_chars] if bpe_train_chars else text
        return CodeBPETokenizer().train(slice_, vocab_size=vocab_size, verbose=verbose)
    raise ValueError(f"unknown tokenizer kind: {kind!r}")


def from_dict(d):
    """Rebuild a tokenizer from the dict stored in a checkpoint."""
    kind = d.get("kind")
    if kind not in _REGISTRY:
        raise ValueError(f"unknown tokenizer kind in checkpoint: {kind!r}")
    return _REGISTRY[kind].from_dict(d)
