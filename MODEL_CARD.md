# Model Card — llm-from-scratch

**Latest remeasurement (2026-10-31):** the existing M3 checkpoint scored 51/170
overall (30.0%), 51/136 multiple-choice (37.5%), 0/18 independently sandboxed code
tasks, 0/16 generation tasks, and 1/12 new secure-development scenarios. Bits per
byte is 1.78003. No weights were changed. Historical results below use earlier
suites and are not directly comparable. See the
[measured review](docs/evaluations/m3-2026-10-31.md) and
[evaluation workflow](docs/EVALUATION.md) for reproducible evidence and limits.

A small, decoder-only GPT language model **written and trained from scratch** for
the **software-development and cybersecurity** domains. Every component — the
byte-pair tokenizer, the attention, the transformer blocks, the training loop and
the evaluation harness — is hand-implemented in plain PyTorch. Nothing is
imported from a pretrained-model library.

This card follows the usual model-card structure (details, intended use, data,
evaluation, limitations, security/ethics) and is written to be **honest about what
the model cannot do**. Its purpose is a learning and portfolio artifact, not a
production assistant.

- **Repository:** https://github.com/Canadasun/llm-from-scratch
- **License:** MIT (see `LICENSE`)
- **Author:** Idowu Peter Ayeni
- **Card version / date:** 1.0 — 2026-10-30

---

## 1. Model details

- **Architecture:** decoder-only (GPT-style) transformer — token + learned
  positional embeddings, causal multi-head self-attention, pre-norm residual
  blocks, a final LayerNorm, and a language-model head whose weights are **tied**
  to the token embedding.
- **Tokenizer:** a from-scratch **byte-level BPE** tokenizer with regex
  pre-tokenization (a character-level tokenizer is also included for the CPU
  preset).
- **Objective:** next-token prediction (causal language modeling), cross-entropy
  loss.
- **Framework:** PyTorch. Trains on CPU (tiny preset) or a single free cloud GPU
  (e.g. Colab Tesla T4) with mixed precision, LR warmup and gradient clipping.

### Configurations (`config.py`)

| preset | layers | heads | d_model | context | vocab | params | corpus | compute |
|---|---|---|---|---|---|---|---|---|
| `cpu` | 4 | 4 | 128 | 128 | char | ~0.2M | sample | laptop CPU, minutes |
| `gpu` (Milestone 2) | 6 | 6 | 384 | 256 | 1024 (BPE) | **~10.7M** | ~6 MB RFCs | free T4, ~50 min |
| `gpu_large` (Milestone 3) | 10 | 8 | 512 | 512 | 2048 (BPE) | **~31.8M** | ~19 MB RFCs + NIST | free T4, ~2 h |

These are **small** models by modern standards (production assistants are
1,000–10,000× larger and trained on vastly more data). Scale expectations are set
accordingly throughout this card.

---

## 2. Intended use

**Primary intended uses**
- An educational reference for how an LLM is built end to end.
- A portfolio artifact demonstrating the full pipeline: tokenizer → model →
  training → **evaluation** → (planned) serving.
- A **base language model** over software/security *text* — useful for studying
  domain language modeling and as a substrate for later retrieval and tool use.

**Intended users:** the author; engineers and students learning LLM internals;
reviewers evaluating the author's work. The domain framing targets **security
practitioners**, so the model is **dual-use-aware**: legitimate security topics
(vulnerability classes, pentesting concepts, exploit *theory*) are in scope and
are not treated as disallowed content. Authorization and safety are enforced in
the **surrounding software**, never delegated to model-generated text.

**Out-of-scope / explicitly NOT intended**
- ❌ A chat assistant or instruction-following model. It is **not instruction-tuned**.
- ❌ A source of authoritative security, legal, financial or medical advice.
- ❌ Autonomous action, code execution, or any high-impact decision without a human
  in the loop.
- ❌ Generating working attack tooling for use against systems you are not
  authorized to test.
- ❌ Any production deployment on its current measured capability (see §4).

---

## 3. Training data

Training text comes from **public or self-authored sources only**. **No private,
customer, personal (PII), or Waine Appointments data is ever used for training.**

| source | license basis | what it contributes |
|---|---|---|
| **IETF RFCs** | freely redistributable (RFC 8721 / IETF Trust TLP) | protocol & security specs — TLS, HTTP, DNS/DNSSEC, OAuth/JWT, email security, routing, crypto |
| **NIST SP 800 series** | works of the U.S. federal government — **public domain** | security *guidance* — zero trust, incident response, risk management, secure SDLC, logging, cloud, ABAC |
| **Author's own notes** | self-authored | dev/security prose in the author's words |

- **Milestone 2 corpus:** ~6 MB (≈46 RFCs).
- **Milestone 3 corpus:** ~19.2 MB (≈113 RFCs + ≈27 NIST SP 800 documents).
- **Secret/PII scrubbing runs as a gate** (`safety/scrub.py`, invoked by
  `data/build_corpus.py`) before any text enters the corpus: API keys, tokens,
  JWTs, cloud keys, private keys and passwords are redacted. A model can memorize
  its data, so this is applied up front, not after the fact. (M3 build redacted
  12 such items from the public corpus.)
- **Preprocessing:** NIST PDFs are text-extracted (PyMuPDF), running headers /
  footers / boilerplate stripped, and de-hyphenated; corpus is cached between runs.
- **Known gaps:** English-only; skewed toward standards/specification prose, which
  is formal and not representative of code, conversation, or general text.

---

## 4. Evaluation

Full methodology: `docs/EVALUATION.md`. The harness (`evals/`) reports numbers and
**never prints "pass"** — release gating is a separate, explicit human decision.

It measures three things and always shows sample sizes, a random-chance baseline,
example failures, and limitations:

1. **Intrinsic quality** on **held-out** text (independently authored, kept out of
   the training corpus): perplexity, tokenizer-independent **bits-per-byte**, and
   next-token top-1/top-5 accuracy — each reported for the trained model **and** a
   fresh random-init twin of the same shape, so the gap proves *training* produced
   the skill.
2. **Capability tasks** (91 hand-authored, validated): likelihood-scored
   multiple-choice (security, software, cloud/DevSecOps, AI-security, reasoning),
   plus generation and **sandboxed** code tasks.
3. Code tasks run generated code in an **isolated subprocess** (throwaway dir,
   minimal env with no parent secrets, CPU/memory/file-size rlimits, timeout) —
   never imported or exec'd in-process.

### Current measured results — Milestone 2 checkpoint (2026-10-30)
*(~10.7M params, BPE vocab 1024, 5000 iters, ~6 MB corpus)*

Held-out intrinsic (trained vs random-init twin):

| metric | trained | random-init |
|---|---|---|
| perplexity | **15.60** | 1,272.65 |
| bits/byte | **2.15** | 5.59 |
| next-token top-1 | **44.7 %** | 0.1 % |
| next-token top-5 | **69.1 %** | 0.8 % |

Capability (overall **23.0 %**, 14/61; MC random chance ≈ 25 %): security 40.0 %,
software 26.7 %, reasoning 12.5 %, generation 10.0 %, code 0.0 %.

**Honest reading:** the intrinsic win is decisive — on unseen text the model
predicts the next token 44.7 % of the time vs 0.1 % untrained, ~80× better
perplexity than random. It genuinely learned the domain's *language*. But the
capability tasks sit **at or near random chance**: a 10M-param model on ~6 MB of
RFCs has **not** acquired transferable security/software knowledge, reasoning, or
coding ability. Fluent, RFC-looking samples could fool a reader; these numbers do
not. This is the expected baseline for the scale.

### Milestone 3 results (2026-10-30)
*(~31.8M params, BPE vocab 2048, block 512, 6000 iters, ~19.2 MB RFC+NIST corpus)*

Held-out intrinsic (trained vs random-init twin):

| metric | M3 trained | random-init | M2 |
|---|---|---|---|
| bits/byte (fair, tokenizer-independent) | **1.78** | 5.43 | 2.15 |
| perplexity¹ | 12.82 | 2,380.92 | 15.60 |
| next-token top-1¹ | 49.3 % | 0.0 % | 44.7 % |
| next-token top-5¹ | 70.4 % | 0.1 % | 69.1 % |

Capability (overall **33.0 %**, 30/91; MC random chance ≈ 25 %): security 43.3 %,
ai_security 50.0 %, cloud 41.7 %, reasoning 37.5 %, software 33.3 %,
generation 0.0 %, code 0.0 %.

¹ tokenizer-dependent (vocab 2048 vs M2's 1024) — shown for reference; bits/byte
is the fair cross-model number.

**Measured outcome:** the bigger model + broader corpus improved language modeling
on the fair metric (bits/byte **2.15 → 1.78**) and moved multiple-choice capability
from near-chance to clearly above it (**23.0 % → 33.0 %**, chance ≈ 25 %), with the
knowledge categories that now have matching training text — security, cloud,
AI-security — at **41–50 %**. Generation and code remain at 0 %, as expected for a
non-instruction-tuned base model at this scale. The eval **regression gate**
confirms no regression vs the M2 baseline. Full detail: `docs/EVALUATION.md`.

---

## 5. Limitations and biases

- **Scale:** at ~10–32M parameters and tens of MB of data, this is orders of
  magnitude below models that exhibit general capability. Expect weak factual
  reliability and little transferable reasoning.
- **Not instruction-tuned:** it continues text; it does not follow instructions or
  hold a conversation. Multiple-choice is scored by likelihood for this reason.
- **Hallucination:** it will produce fluent, authoritative-sounding text that is
  wrong. Do not rely on any output as fact without independent verification.
- **Domain/style bias:** trained mostly on standards/specification prose — biased
  toward that register and the viewpoints of those bodies; weak elsewhere.
- **Language:** English only.
- **Memorization risk:** small models can memorize training text; this is why
  secret scrubbing runs *before* training and why intrinsic metrics use a
  **held-out**, contamination-free set.

---

## 6. Security considerations

Designed for a security-practitioner audience, so security posture is explicit:

- **Dual-use by design:** legitimate security content is in scope and not refused;
  the model is not a safety control.
- **Authorization is enforced in software, not by model text.** Model output,
  retrieved documents, repositories, web pages and tool responses are treated as
  **untrusted input** (relevant to prompt-injection — OWASP LLM Top 10).
- **Sandboxed code evaluation:** any model-generated code executed by the eval
  harness runs isolated, resource-limited, with no access to parent-process
  secrets, and is never run in-process.
- **Secure checkpoint loading:** checkpoints are loaded with `weights_only=True`
  to avoid arbitrary-code execution via pickle.
- **No training on sensitive data:** see §3 — public/self-authored only.
- **Scope for any security research tied to this project:** authorized testing
  against the author's **own** devices on an isolated lab network only — never
  third-party systems.

---

## 7. Ethical considerations & caveats

- This model should **not** drive decisions that affect people. It is a learning
  artifact.
- No claims of 100 % accuracy, safety, or prompt-injection immunity are made.
- **Environmental:** training is a single ~2 h run on one shared free-tier GPU —
  deliberately modest; no large-scale compute was used.
- **Provenance:** this card describes a hand-built model; capability figures are
  the author's measured results from `evals/`, reproducible from the checkpoint.

---

## 8. How to reproduce the evaluation

```bash
# with a trained checkpoint at checkpoints/ckpt.pt
python -m evals.run                      # intrinsic + all capability tasks
python -m evals.run --suite intrinsic    # perplexity/bits-per-byte vs random-init
python -m evals.run --ckpt path/to/ckpt.pt --device cuda
```

The report is self-describing: it prints sample sizes, the random-chance baseline,
example failures, and the limitations above on every run.
