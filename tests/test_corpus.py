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
