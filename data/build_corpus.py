"""
Build a software-dev / cybersecurity training corpus (Milestone 2/3).

Two license-safe public sources, plus your own files:

1. IETF RFCs — published as clean plain text, freely redistributable
   (RFC 8721 / the IETF Trust TLP), covering TLS, HTTP, DNS/DNSSEC, OAuth,
   JWT/JOSE, SSH, X.509, IPsec, QUIC, email, routing, encodings and more.
2. NIST Special Publications (SP 800 series) — works of the U.S. federal
   government, in the public domain. These add security *guidance and
   explanatory prose* (zero trust, incident response, risk management, secure
   development, logging, testing) that RFCs' protocol specs don't cover — the
   Milestone-3 bet for broadening the model beyond narrow standards text.
   Fetched as PDF and extracted with PyMuPDF (`pip install pymupdf`).

You can also drop your own `.txt` / `.md` files (public docs, your own notes)
into data/sources/ — they are appended after the fetched sources.

Output: data/corpus.txt (UTF-8), which the trainer reads.

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

# A browser-like UA: rfc-editor.org is fine with the default, but nvlpubs.nist.gov
# returns 403 to the stock Python urllib agent.
_USER_AGENT = "Mozilla/5.0 (llm-from-scratch corpus builder; public-domain docs)"


def _http_get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


# A curated set of RFCs across security, networking and web development.
RFCS = {
    # --- TLS / crypto ---
    8446: "TLS 1.3", 5246: "TLS 1.2", 5280: "X.509 PKI certificates",
    6797: "HTTP Strict Transport Security", 2104: "HMAC",
    8017: "PKCS #1 RSA", 7748: "Elliptic curves for security",
    8439: "ChaCha20-Poly1305", 5869: "HKDF", 2898: "PBKDF2 (PKCS #5)",
    5652: "Cryptographic Message Syntax", 6960: "OCSP",
    6962: "Certificate Transparency", 8032: "EdDSA", 7914: "scrypt",
    5116: "AEAD interface", 4086: "Randomness requirements for security",
    7525: "TLS recommendations (BCP)", 9325: "TLS recommendations (BCP, 2022)",
    8996: "Deprecating TLS 1.0 and 1.1", 8555: "ACME", 7030: "EST",
    6066: "TLS extensions", 8018: "PKCS #5 v2.1",
    # --- HTTP / web ---
    9110: "HTTP semantics", 9111: "HTTP caching", 9112: "HTTP/1.1",
    9113: "HTTP/2", 9114: "HTTP/3", 6455: "WebSocket protocol",
    7235: "HTTP authentication", 6265: "HTTP cookies", 6454: "Web Origin",
    7034: "X-Frame-Options", 6585: "Additional HTTP status codes",
    7541: "HPACK header compression", 9204: "QPACK header compression",
    5789: "HTTP PATCH", 6570: "URI Template", 3875: "CGI",
    # --- Auth / tokens / OAuth ---
    6749: "OAuth 2.0", 6750: "OAuth 2.0 bearer tokens", 7519: "JSON Web Token",
    7515: "JSON Web Signature", 7516: "JSON Web Encryption",
    7517: "JSON Web Key", 7518: "JSON Web Algorithms", 6238: "TOTP",
    4226: "HOTP", 7636: "OAuth PKCE", 8252: "OAuth for native apps",
    8414: "OAuth server metadata", 7662: "OAuth token introspection",
    7009: "OAuth token revocation", 8705: "OAuth mutual-TLS",
    9068: "JWT profile for OAuth access tokens", 8628: "OAuth device grant",
    6819: "OAuth 2.0 threat model", 7591: "OAuth dynamic client registration",
    # --- DNS ---
    1034: "DNS concepts", 1035: "DNS implementation",
    4033: "DNSSEC introduction", 4034: "DNSSEC resource records",
    4035: "DNSSEC protocol", 8484: "DNS over HTTPS", 7858: "DNS over TLS",
    9250: "DNS over QUIC", 6891: "EDNS(0)", 8499: "DNS terminology",
    # --- Email security ---
    5321: "SMTP", 5322: "Internet message format", 6376: "DKIM",
    7208: "SPF", 7489: "DMARC", 3207: "SMTP over TLS", 8314: "Email TLS",
    # --- Transport / IP / routing ---
    9293: "TCP", 768: "UDP", 791: "IPv4", 8200: "IPv6",
    9000: "QUIC transport", 4301: "IPsec architecture", 4271: "BGP-4",
    2328: "OSPF v2", 826: "ARP", 792: "ICMP", 4443: "ICMPv6",
    2131: "DHCP", 8415: "DHCPv6", 4861: "IPv6 Neighbor Discovery",
    1918: "Private IPv4 address space", 4291: "IPv6 addressing",
    # --- Encodings / formats / foundations ---
    8259: "JSON", 3986: "URI generic syntax", 2119: "Requirement key words",
    8174: "RFC 2119 clarification", 6901: "JSON Pointer", 6902: "JSON Patch",
    7396: "JSON Merge Patch", 8949: "CBOR", 4648: "Base16/32/64 encodings",
    3339: "Date and time on the Internet", 5646: "Tags for languages",
    2045: "MIME part one", 6838: "Media type specifications", 5234: "ABNF",
    # --- Security concepts / BCPs ---
    4949: "Internet Security Glossary v2", 3552: "Security considerations guidelines",
    6973: "Privacy considerations", 7258: "Pervasive monitoring is an attack",
    4107: "Guidelines for cryptographic key management",
    # --- legacy HTTP (kept: lots of real-world prose references them) ---
    7231: "HTTP/1.1 semantics (legacy)", 7540: "HTTP/2 (legacy)",
}

# NIST Special Publications (U.S. federal government = public domain). Two URL
# layouts on nvlpubs.nist.gov: newer docs under SpecialPublications/, older ones
# under Legacy/SP/. Values are (title, url).
_NIST_NEW = "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.{}.pdf"
_NIST_OLD = "https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication{}.pdf"

NIST = {
    "800-207":  ("Zero Trust Architecture", _NIST_NEW.format("800-207")),
    "800-61r2": ("Computer Security Incident Handling Guide", _NIST_NEW.format("800-61r2")),
    "800-53r5": ("Security and Privacy Controls", _NIST_NEW.format("800-53r5")),
    "800-63-3": ("Digital Identity Guidelines", _NIST_NEW.format("800-63-3")),
    "800-171r2": ("Protecting Controlled Unclassified Information", _NIST_NEW.format("800-171r2")),
    "800-218":  ("Secure Software Development Framework (SSDF)", _NIST_NEW.format("800-218")),
    "800-128":  ("Security-focused Configuration Management", _NIST_NEW.format("800-128")),
    "800-40r4": ("Enterprise Patch Management Planning", _NIST_NEW.format("800-40r4")),
    "800-190":  ("Application Container Security Guide", _NIST_NEW.format("800-190")),
    "800-37r2": ("Risk Management Framework", _NIST_NEW.format("800-37r2")),
    "800-204":  ("Security Strategies for Microservices", _NIST_NEW.format("800-204")),
    "800-83r1": ("Malware Incident Prevention and Handling", _NIST_NEW.format("800-83r1")),
    "800-77r1": ("Guide to IPsec VPNs", _NIST_NEW.format("800-77r1")),
    "800-52r2": ("Guidelines for TLS Implementations", _NIST_NEW.format("800-52r2")),
    "800-47r1": ("Managing the Security of Information Exchanges", _NIST_NEW.format("800-47r1")),
    "800-184":  ("Guide for Cybersecurity Event Recovery", _NIST_NEW.format("800-184")),
    "800-150":  ("Guide to Cyber Threat Information Sharing", _NIST_NEW.format("800-150")),
    "800-115":  ("Technical Guide to Security Testing and Assessment", _NIST_OLD.format("800-115")),
    "800-92":   ("Guide to Computer Security Log Management", _NIST_OLD.format("800-92")),
    "800-94":   ("Guide to Intrusion Detection and Prevention Systems", _NIST_OLD.format("800-94")),
    "800-30r1": ("Guide for Conducting Risk Assessments", _NIST_OLD.format("800-30r1")),
    "800-123":  ("Guide to General Server Security", _NIST_OLD.format("800-123")),
    "800-44ver2": ("Guidelines on Securing Public Web Servers", _NIST_OLD.format("800-44ver2")),
    "800-95":   ("Guide to Secure Web Services", _NIST_OLD.format("800-95")),
    "800-146":  ("Cloud Computing Synopsis and Recommendations", _NIST_OLD.format("800-146")),
    "800-153":  ("Guidelines for Securing Wireless LANs", _NIST_OLD.format("800-153")),
    "800-162":  ("Guide to Attribute Based Access Control (ABAC)", _NIST_OLD.format("800-162")),
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
    return _http_get(url, timeout).decode("utf-8", errors="replace")


def clean_nist(text):
    """Strip NIST-PDF furniture (running headers, page numbers, the free-of-charge
    DOI boilerplate) and rejoin words that PDF extraction split across lines."""
    text = text.replace("\f", "\n")
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            lines.append("")
            continue
        if re.match(r"^NIST\s+SP\s+800-\d+", s):               # running header
            continue
        if s.startswith("This publication is available free of charge"):
            continue
        if re.match(r"^https?://doi\.org/\S+$", s):            # DOI line
            continue
        if re.match(r"^\d+$", s):                              # bare page number
            continue
        if re.match(r"^[ivxlcdm]{1,6}$", s):                   # roman-numeral page
            continue
        lines.append(line.rstrip())
    out = "\n".join(lines)
    # Rejoin words hyphenated across a line break: "config-\nuration" -> "configuration".
    out = re.sub(r"([a-z])-\n([a-z])", r"\1\2", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip() + "\n"


def fetch_pdf_text(url, timeout=90):
    """Download a PDF and extract its text. Requires PyMuPDF (`pip install
    pymupdf`); raises a clear error if it is missing so NIST can be skipped
    without crashing the RFC build."""
    try:
        import pymupdf
    except ImportError as e:                                   # pragma: no cover
        raise RuntimeError(
            "PyMuPDF is required for NIST PDFs: pip install pymupdf "
            "(or run with --no-nist)"
        ) from e
    data = _http_get(url, timeout)
    doc = pymupdf.open(stream=data, filetype="pdf")
    try:
        return "\n".join(page.get_text() for page in doc)
    finally:
        doc.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=os.path.join(HERE, "corpus.txt"))
    ap.add_argument("--cache", default=os.path.join(HERE, "rfc_cache"),
                    help="Downloaded RFCs are cached here so re-runs are offline.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only use the first N RFCs (for a quick test).")
    ap.add_argument("--no-nist", action="store_true",
                    help="skip the NIST SP 800 publications (RFCs only).")
    ap.add_argument("--nist-limit", type=int, default=None,
                    help="Only use the first N NIST publications.")
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

    # NIST SP 800 publications (public domain). Extracted text is cached so
    # re-runs are offline and the slow PDF parse is paid once.
    if not args.no_nist:
        nist_ids = list(NIST)[: args.nist_limit] if args.nist_limit else list(NIST)
        for i, sp in enumerate(nist_ids, 1):
            title, url = NIST[sp]
            cache_path = os.path.join(args.cache, f"nist_{sp}.txt")
            try:
                if os.path.exists(cache_path):
                    raw = open(cache_path, encoding="utf-8").read()
                    status = "cached"
                else:
                    raw = fetch_pdf_text(url)
                    open(cache_path, "w", encoding="utf-8").write(raw)
                    status = "fetched"
                    time.sleep(0.3)
                cleaned = clean_nist(raw)
                if not args.no_scrub:
                    cleaned, counts = _scrub.scrub_secrets(cleaned)
                    _accumulate(counts)
                header = f"\n\n===== NIST SP {sp}: {title} =====\n\n"
                parts.append(header + cleaned)
                print(f"[{i:2d}/{len(nist_ids)}] NIST SP {sp:<10} {title[:32]:<32} "
                      f"{len(cleaned):>8,} chars ({status})", flush=True)
            except Exception as e:
                print(f"[{i:2d}/{len(nist_ids)}] NIST SP {sp:<10} SKIPPED: {e}",
                      file=sys.stderr, flush=True)

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
