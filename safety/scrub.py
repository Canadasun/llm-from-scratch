"""
Scrub secrets and PII from training text (data-in safety).

Why this exists, and why it runs by default:

  A model trained on text can memorize and later emit verbatim strings from that
  text. For a model aimed at cybersecurity and software work, the training data
  is exactly the kind of material that tends to contain live credentials —
  private keys, API tokens, passwords in config snippets, and so on, especially
  once you add your own notes under data/sources/. Redacting those before
  training means the model can never learn or leak them.

Design:
  - `scrub_secrets` targets credentials/keys/tokens. It is always safe to run and
    is applied by default to all corpus text.
  - `scrub_pii` targets personal data (email addresses). It is optional, because
    the public RFC corpus uses illustrative example addresses that are part of
    the technical text; it is meant for user-supplied sources.
  - Each match is replaced by a clear placeholder like [REDACTED_PRIVATE_KEY], so
    the surrounding prose still reads sensibly and you can see where redaction
    happened.

These patterns are deliberately conservative: they aim to catch well-known,
high-confidence secret formats with few false positives, not to be a complete
DLP engine. Strengthen them as the corpus grows.
"""

import re

# (name, compiled pattern, replacement)
_SECRET_PATTERNS = [
    ("PRIVATE_KEY",
     re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----.*?-----END (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----",
                re.DOTALL),
     "[REDACTED_PRIVATE_KEY]"),
    ("AWS_ACCESS_KEY_ID", re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[REDACTED_AWS_KEY]"),
    ("GITHUB_TOKEN", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"), "[REDACTED_GITHUB_TOKEN]"),
    ("GITHUB_PAT", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"), "[REDACTED_GITHUB_TOKEN]"),
    ("SLACK_TOKEN", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"), "[REDACTED_SLACK_TOKEN]"),
    ("GOOGLE_API_KEY", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"), "[REDACTED_GOOGLE_API_KEY]"),
    ("JWT", re.compile(r"\beyJ[A-Za-z0-9_-]{5,}\.eyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\b"),
     "[REDACTED_JWT]"),
    ("BEARER", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{16,}"), "Bearer [REDACTED_TOKEN]"),
    # key = value style assignments for common secret names
    ("ASSIGNED_SECRET",
     re.compile(r"""(?ix)\b(api[_-]?key|secret(?:[_-]?key)?|access[_-]?token|
                    auth[_-]?token|password|passwd|pwd|client[_-]?secret)\b
                    \s*[:=]\s*['"]?([A-Za-z0-9/+_\-\.]{8,})['"]?"""),
     r"\1=[REDACTED_SECRET]"),
    # password embedded in a URL: scheme://user:pass@host
    ("URL_PASSWORD",
     re.compile(r"\b([a-z][a-z0-9+.\-]*://[^\s:/@]+):([^\s:/@]+)@"),
     r"\1:[REDACTED_SECRET]@"),
]

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")


def scrub_secrets(text):
    """Redact credentials/keys/tokens. Returns (clean_text, {name: count})."""
    counts = {}
    for name, pat, repl in _SECRET_PATTERNS:
        text, n = pat.subn(repl, text)
        if n:
            counts[name] = counts.get(name, 0) + n
    return text, counts


def scrub_pii(text):
    """Redact personal data (email addresses). Returns (clean_text, {name: count})."""
    text, n = _EMAIL.subn("[REDACTED_EMAIL]", text)
    return text, ({"EMAIL": n} if n else {})


def scrub(text, pii=False):
    """Run secret scrubbing (always) and, optionally, PII scrubbing."""
    text, counts = scrub_secrets(text)
    if pii:
        text, pc = scrub_pii(text)
        counts.update(pc)
    return text, counts


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="Scrub secrets/PII from a text file.")
    ap.add_argument("path")
    ap.add_argument("--pii", action="store_true", help="also redact email addresses")
    ap.add_argument("--inplace", action="store_true", help="overwrite the file")
    args = ap.parse_args()

    original = open(args.path, encoding="utf-8", errors="ignore").read()
    cleaned, counts = scrub(original, pii=args.pii)
    if args.inplace:
        open(args.path, "w", encoding="utf-8").write(cleaned)
        print(f"Scrubbed in place: {args.path}")
    else:
        sys.stdout.write(cleaned)
    if counts:
        print(f"\nRedactions: {counts}", file=sys.stderr)
    else:
        print("\nNo secrets/PII found.", file=sys.stderr)
