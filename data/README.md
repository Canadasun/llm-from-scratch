# Training data

## Data policy

Training data is **public or self-authored only**. No customer, personal, or
private business data is ever used to train this model.

## Sources

### IETF RFCs (primary corpus)

`build_corpus.py` downloads a curated set of ~46 RFCs and assembles them into
`corpus.txt`. RFCs are published as clean plain text and are freely
redistributable under the IETF Trust Legal Provisions (see RFC 8721). They were
chosen because they densely cover this project's domains:

- **Transport/crypto:** TLS 1.2/1.3, X.509, HMAC, PKCS#1, elliptic curves, IPsec
- **Web:** HTTP/1.1–3, WebSocket, cookies, HTTP auth, HSTS
- **Identity:** OAuth 2.0, JWT, JOSE (JWS/JWE/JWK/JWA), TOTP
- **Naming:** DNS, DNSSEC, DNS-over-HTTPS
- **Networking:** TCP, UDP, IPv4/IPv6, QUIC
- **SSH, SMTP/email, JSON, URIs**

Downloaded RFCs are cached under `data/rfc_cache/` so re-runs work offline. The
cache and the generated `corpus.txt` are git-ignored (they are reproducible).

### Your own text (optional)

Drop `.txt` or `.md` files into `data/sources/` and they are appended to the
corpus after the RFCs — e.g. your own notes, or public documentation you have
the right to use.

## Rebuild

```bash
python data/build_corpus.py            # full corpus (~6 MB)
python data/build_corpus.py --limit 5  # quick test with 5 RFCs
```

## Tokenizer note

The from-scratch BPE tokenizer (`tokenizer_bpe.py`) is trained **once** on this
corpus and saved to a JSON file, then reused for both training and inference.
Pure-Python BPE training is intentionally simple and therefore slow on large
inputs; training the tokenizer on a representative slice (a few hundred KB) and
reusing it is the normal workflow.
