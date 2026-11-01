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

### Python code corpus (for the coding-capability training step)

`build_code_corpus.py` assembles a **license-safe Python code** corpus — the
piece the RFC/NIST prose lacks, and the reason the model scores ~0 on coding:

- **CPython standard library** (PSF License — permissive, redistributable), read
  from this interpreter's own stdlib directory, so it is fully **offline** and
  costs nothing. Tests, bundled third-party (`lib2to3`, `ensurepip`), GUI demos
  and oversized files are skipped.
- **Original secure-coding / insecure-counterexample / refusal examples**
  (`build_code_examples.py`) — written for this project (license-safe). These
  teach the secure-vs-insecure contrast and give abstention examples.

Every file is secret-scrubbed, a manifest records sources/licenses/sizes/hashes,
and the result is checked against the evaluation set for **train/eval
contamination** (`contamination.py`) — none of the held-out text or task
prompts/targets may appear verbatim in the corpus, so memorisation cannot look
like progress. When repo-level code is added later, keep the eval **split by
repository and problem family** (hold out whole repos/families) to preserve this.

```bash
python data/build_code_corpus.py                 # ~40 MB default cap, offline
python data/build_code_corpus.py --max-bytes 8000000
```

Output `corpus_code.txt` and `code_corpus_manifest.json` are git-ignored
(reproducible).

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
