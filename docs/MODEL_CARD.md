# Model card — llm-from-scratch

A living model card. It is filled in as the project advances; today it describes
the Milestone 1/2 state honestly, including what the model cannot yet do.

## Overview

- **Name:** llm-from-scratch
- **Type:** decoder-only transformer (GPT-style), implemented from scratch.
- **Purpose:** an assistant for cybersecurity, software development and
  engineering management — see [RESPONSIBLE_USE.md](RESPONSIBLE_USE.md).
- **Status:** educational / in development. Not production-ready.

## Current capability (be honest)

| | Milestone 1 | Milestone 2 (in progress) |
|---|---|---|
| Tokenizer | character-level | byte-level BPE (from scratch) |
| Parameters | ~0.8M (CPU default) | grows (10–100M on GPU) |
| Training data | sample / any `.txt` | ~6 MB curated IETF RFCs + your sources |
| Hardware | laptop CPU | free cloud GPU |

At Milestone 1 the model produces text in the *style* of its training data; it
is far too small to be factually reliable or to reason. **Do not rely on its
output.** It exists to make the architecture and training process concrete.

## Training data

Public and self-authored text only. The primary corpus is IETF RFCs (freely
redistributable), covering TLS, HTTP, DNS/DNSSEC, OAuth, JWT/JOSE, SSH, X.509,
IPsec, QUIC, email, JSON and URIs. No customer, personal, or private business
data is used. Secrets and (optionally) PII are redacted before training by
`safety/scrub.py`. See [`../data/README.md`](../data/README.md).

## Safety

- Dual-use-aware use policy: `safety/content_policy.py` (allow / flag / block).
- Training-data scrubbing: `safety/scrub.py`.
- Secure checkpoint loading (`weights_only=True`). See [`../SECURITY.md`](../SECURITY.md).

## Limitations and risks

- Tiny model: outputs are unreliable, may be nonsensical, and must not be trusted
  for security or engineering decisions.
- Corpus bias: RFC-heavy, so it reflects protocol/standards language, not the
  breadth of real-world code or operations.
- The safety layers are a transparent first pass, not a guarantee.
- No alignment / instruction tuning yet; it is a base language model.

## Evaluation

Regression tests in `tests/` cover the model (shapes, causal masking,
learning), both tokenizers (round-trip, compression), corpus cleaning and the
safety layers. A capability/quality evaluation harness is planned for Milestone 3.
