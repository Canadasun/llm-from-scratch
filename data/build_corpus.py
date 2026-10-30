"""
Build a software-dev / cybersecurity training corpus (Milestone 2).

Primary source: IETF RFCs. They are published as clean plain text, are freely
redistributable (RFC 8721 / the IETF Trust TLP), and cover exactly this
project's domains — TLS, HTTP, DNS/DNSSEC, OAuth, JWT/JOSE, SSH, X.509, IPsec,
QUIC, email, URIs and more. That makes them an ideal license-safe starting
corpus, with none of the privacy concerns of scraped data.

You can extend the corpus by dropping your own `.txt` / `.md` files (public
docs, your own notes) into data/sources/ — they are appended after the RFCs.

Output: data/corpus.txt (UTF-8), which train.py / the Milestone 2 trainer reads.

NOTE on data policy: public + self-authored text only. No customer, personal,
or Waine data is ever used for training.
"""

import argparse
import os
import re
import sys
import time
import urllib.request

HERE = os.path.dirname(__file__)
# Make the repo root importable so we can use the safety package when this file
# is run directly as `python data/build_corpus.py`.
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
from safety import scrub as _scrub  # noqa: E402

# A curated set of RFCs across security, networking and web development.
RFCS = {
    # TLS / crypto
    8446: "TLS 1.3", 5246: "TLS 1.2", 5280: "X.509 PKI certificates",
    6797: "HTTP Strict Transport Security", 2104: "HMAC",
    8017: "PKCS #1 RSA", 7748: "Elliptic curves for security",
    # HTTP / web
    9110: "HTTP semantics", 9111: "HTTP caching", 9112: "HTTP/1.1",
    9113: "HTTP/2", 9114: "HTTP/3", 6455: "WebSocket protocol",
    7235: "HTTP authentication", 6265: "HTTP cookies",
    # Auth / tokens
    6749: "OAuth 2.0", 6750: "OAuth 2.0 bearer tokens", 7519: "JSON Web Token",
    7515: "JSON Web Signature", 7516: "JSON Web Encryption",
    7517: "JSON Web Key", 7518: "JSON Web Algorithms", 6238: "TOTP",
    # DNS
    1034: "DNS concepts", 1035: "DNS implementation",
    4033: "DNSSEC introduction", 4034: "DNSSEC resource records",
    4035: "DNSSEC protocol", 8484: "DNS over HTTPS",
    # Transport / IP
    9293: "TCP", 768: "UDP", 791: "IPv4", 8200: "IPv6",
    9000: "QUIC transport", 4301: "IPsec architecture",
    # SSH
    4251: "SSH architecture", 4252: "SSH authentication",
    4253: "SSH transport layer", 4254: "SSH connection protocol",
    # Email
    5321: "SMTP", 5322: "Internet message format",
    # Formats / foundations
    8259: "JSON", 3986: "URI generic syntax", 2119: "Requirement key words",
    7231: "HTTP/1.1 semantics (legacy)", 7540: "HTTP/2 (legacy)",
}


def clean_rfc(text):
    """Strip RFC page furniture so the model learns prose, not pagination."""
    text = text.replace("\f", "\n")                       # form-feed page breaks
    lines = []
    for line in text.splitlines():
        # Running headers/footers: "RFC 8446 ... August 2018" and "[Page 12]".
        if re.match(r"^RFC \d+\s", line) and re.search(r"\d{4}\s*$", line):
            continue
        if re.search(r"\[Page \d+\]\s*$", line):
            continue
        lines.append(line.rstrip())
    out = "\n".join(lines)
    out = re.sub(r"\n{3,}", "\n\n", out)                  # collapse blank runs
    return out.strip() + "\n"


def fetch_rfc(num, timeout=30):
    url = f"https://www.rfc-editor.org/rfc/rfc{num}.txt"
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=os.path.join(HERE, "corpus.txt"))
    ap.add_argument("--cache", default=os.path.join(HERE, "rfc_cache"),
                    help="Downloaded RFCs are cached here so re-runs are offline.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only use the first N RFCs (for a quick test).")
    ap.add_argument("--no-scrub", action="store_true",
                    help="skip secret scrubbing (NOT recommended).")
    ap.add_argument("--scrub-pii", action="store_true",
                    help="also redact emails in user-supplied sources.")
    args = ap.parse_args()

    os.makedirs(args.cache, exist_ok=True)
    parts = []
    redactions = {}  # running total across all sources
    nums = list(RFCS)[: args.limit] if args.limit else list(RFCS)

    def _accumulate(counts):
        for k, v in counts.items():
            redactions[k] = redactions.get(k, 0) + v

    for i, num in enumerate(nums, 1):
        cache_path = os.path.join(args.cache, f"rfc{num}.txt")
        try:
            if os.path.exists(cache_path):
                raw = open(cache_path, encoding="utf-8").read()
                status = "cached"
            else:
                raw = fetch_rfc(num)
                open(cache_path, "w", encoding="utf-8").write(raw)
                status = "fetched"
                time.sleep(0.3)  # be polite to the server
            cleaned = clean_rfc(raw)
            if not args.no_scrub:
                cleaned, counts = _scrub.scrub_secrets(cleaned)
                _accumulate(counts)
            header = f"\n\n===== RFC {num}: {RFCS[num]} =====\n\n"
            parts.append(header + cleaned)
            print(f"[{i:2d}/{len(nums)}] RFC {num:<4} {RFCS[num]:<32} "
                  f"{len(cleaned):>8,} chars ({status})", flush=True)
        except Exception as e:
            print(f"[{i:2d}/{len(nums)}] RFC {num:<4} SKIPPED: {e}", file=sys.stderr, flush=True)

    # Append any user-supplied sources.
    src_dir = os.path.join(HERE, "sources")
    if os.path.isdir(src_dir):
        for name in sorted(os.listdir(src_dir)):
            if name.lower().endswith((".txt", ".md")):
                path = os.path.join(src_dir, name)
                body = open(path, encoding="utf-8", errors="ignore").read()
                # User sources get secret scrubbing and, optionally, PII scrubbing.
                if not args.no_scrub:
                    body, counts = _scrub.scrub(body, pii=args.scrub_pii)
                    _accumulate(counts)
                parts.append(f"\n\n===== source: {name} =====\n\n" + body)
                print(f"[+] local source {name}: {len(body):,} chars", flush=True)

    corpus = "\n".join(parts)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(corpus)
    mb = len(corpus.encode("utf-8")) / 1e6
    print(f"\nWrote {args.output}: {len(corpus):,} chars ({mb:.1f} MB)")
    if args.no_scrub:
        print("WARNING: secret scrubbing was DISABLED (--no-scrub).")
    elif redactions:
        print(f"Safety: redacted secrets/PII -> {redactions}")
    else:
        print("Safety: secret scrub ran; nothing matched.")


if __name__ == "__main__":
    main()
