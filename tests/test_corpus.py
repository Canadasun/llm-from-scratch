"""Regression tests for corpus cleaning (no network required)."""

import importlib.util
import os

# Load data/build_corpus.py as a module without needing a package.
_spec = importlib.util.spec_from_file_location(
    "build_corpus",
    os.path.join(os.path.dirname(__file__), "..", "data", "build_corpus.py"),
)
build_corpus = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_corpus)


def test_clean_rfc_strips_page_furniture():
    raw = (
        "RFC 8446                    TLS 1.3                   August 2018\n"
        "\n"
        "   The handshake authenticates the peers.\n"
        "\f"
        "   More normal content here.\n"
        "                                                           [Page 12]\n"
    )
    out = build_corpus.clean_rfc(raw)
    assert "August 2018" not in out       # running header removed
    assert "[Page 12]" not in out         # footer removed
    assert "\f" not in out                 # form-feed removed
    assert "handshake authenticates" in out
    assert "More normal content" in out


def test_clean_rfc_collapses_blank_runs():
    raw = "line one\n\n\n\n\nline two\n"
    out = build_corpus.clean_rfc(raw)
    assert "\n\n\n" not in out


def test_clean_nist_strips_furniture_and_dehyphenates():
    raw = (
        "NIST SP 800-207\n"
        "Zero trust is a set of principles.\n"
        "This publication is available free of charge from:\n"
        "https://doi.org/10.6028/NIST.SP.800-207\n"
        "iii\n"
        "42\n"
        "Security config-\nuration matters.\n"
    )
    out = build_corpus.clean_nist(raw)
    assert "NIST SP 800-207" not in out        # running header removed
    assert "available free of charge" not in out
    assert "doi.org" not in out                # DOI line removed
    assert "\niii\n" not in out and "\n42\n" not in out   # page numbers removed
    assert "configuration matters" in out      # hyphenation rejoined
    assert "set of principles" in out


def test_nist_catalog_is_nonempty_and_well_formed():
    assert len(build_corpus.NIST) >= 20
    for sp, (title, url) in build_corpus.NIST.items():
        assert url.startswith("https://nvlpubs.nist.gov/") and url.endswith(".pdf")
        assert title


def test_bpe_encode_cache_is_bounded(monkeypatch):
    import tokenizer_bpe
    monkeypatch.setattr(tokenizer_bpe, "_CACHE_MAX", 8)
    tok = tokenizer_bpe.BPETokenizer()           # base byte vocab, no merges
    for i in range(40):
        tok.encode(f"distinct-chunk-{i} ")
    assert len(tok._cache) <= 8                   # no unbounded growth
