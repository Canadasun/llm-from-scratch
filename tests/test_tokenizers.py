"""Regression tests for both tokenizers."""

from tokenizer import CharTokenizer
from tokenizer_bpe import BPETokenizer, _merge, _get_pair_counts


# --- char-level ---

def test_char_roundtrip():
    text = "TLS 1.3 handshake\nauthenticates the server."
    tok = CharTokenizer.from_text(text)
    assert tok.decode(tok.encode(text)) == text


def test_char_vocab_is_sorted_and_deterministic():
    a = CharTokenizer.from_text("banana")
    b = CharTokenizer.from_text("ananab")
    assert a.chars == b.chars          # order independent of input order
    assert a.chars == sorted(set("banana"))


def test_char_save_load(tmp_path):
    tok = CharTokenizer.from_text("abcABC 123\n")
    p = tmp_path / "char.json"
    tok.save(p)
    loaded = CharTokenizer.load(p)
    assert loaded.chars == tok.chars
    assert loaded.encode("abc") == tok.encode("abc")


# --- BPE ---

def test_bpe_merge_helper():
    assert _merge([1, 2, 1, 2, 3], (1, 2), 99) == [99, 99, 3]


def test_bpe_pair_counts():
    counts = _get_pair_counts([1, 2, 1, 2])
    assert counts[(1, 2)] == 2
    assert counts[(2, 1)] == 1


def test_bpe_roundtrip_ascii():
    text = "the server authenticates the handshake " * 20
    tok = BPETokenizer().train(text, vocab_size=300)
    s = "the handshake authenticates the server"
    assert tok.decode(tok.encode(s)) == s


def test_bpe_roundtrip_unicode():
    """Byte-level BPE must handle any UTF-8, including non-ASCII, with no UNK."""
    text = "café — naïve — 🔐 secure — café " * 20
    tok = BPETokenizer().train(text, vocab_size=300)
    s = "café 🔐 naïve"
    assert tok.decode(tok.encode(s)) == s


def test_bpe_compresses():
    text = "authentication authorization authentication " * 50
    tok = BPETokenizer().train(text, vocab_size=320)
    ids = tok.encode("authentication authorization")
    assert len(ids) < len("authentication authorization")


def test_bpe_save_load(tmp_path):
    text = "handshake handshake handshake crypto crypto " * 20
    tok = BPETokenizer().train(text, vocab_size=300)
    p = tmp_path / "bpe.json"
    tok.save(p)
    loaded = BPETokenizer.load(p)
    assert loaded.merges == tok.merges
    sample = "handshake crypto"
    assert loaded.encode(sample) == tok.encode(sample)
    assert loaded.decode(loaded.encode(sample)) == sample


def test_bpe_vocab_size_respected():
    tok = BPETokenizer().train("abcabcabc " * 50, vocab_size=270)
    assert tok.vocab_size <= 270
