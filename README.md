# llm-from-scratch

Building a large language model **from scratch** — no model libraries, no
fine-tuning of someone else's weights, no retrieval wrapper. The tokenizer, the
self-attention, the transformer blocks and the training loop are all written by
hand so every line is understood.

The long-term focus is a model that helps with **software development and
cybersecurity**, trained on public and self-curated text in those domains.

## Why "from scratch" in stages

A frontier model needs thousands of GPUs. A *learning-grade* model that teaches
you how LLMs actually work needs only a laptop. This repo is built to grow:

| Milestone | What you build | Compute |
|---|---|---|
| **1 — Local GPT** (this code) | A small GPT (~0.5M params by default) with a char-level tokenizer, trained on CPU | Your own machine, no GPU |
| **2 — Real tokenizer + bigger model** | Byte-pair-encoding (BPE) tokenizer, 10–100M params, a curated dev+security corpus | Free Colab/Kaggle GPU |
| **3 — Scale** | More layers/params/data, proper eval and checkpointing | Rented cloud GPU |
| **4 — Serve** | An inference endpoint for your own model | Cloud, when needed |

You are at **Milestone 1**.

## Quickstart (CPU)

```bash
# 1. Install the CPU build of PyTorch (one time)
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2. Prepare a corpus (built-in sample, or your own text)
python data/prepare.py                      # built-in sample
# python data/prepare.py --input notes.txt  # your own .txt

# 3. Train (minutes on a CPU with the small defaults)
python train.py

# 4. Generate text from the trained model
python sample.py --prompt "Security " --tokens 400
```

## What each file is

| File | Role |
|---|---|
| `model.py` | The GPT itself: embeddings, causal self-attention, transformer blocks, LM head. Hand-written, heavily commented. |
| `tokenizer.py` | Character-level tokenizer (Milestone 1). BPE replaces this in Milestone 2. |
| `data/prepare.py` | Writes `data/corpus.txt` from the built-in sample or a `--input` file. |
| `config.py` | CPU-friendly training hyperparameters — the knobs you grow later. |
| `train.py` | Batching, AdamW, gradient clipping, train/val loss, checkpoint save. |
| `sample.py` | Autoregressive text generation from a checkpoint. |

## How it works, in one paragraph

Text is turned into a sequence of token ids. Each token looks up a learned
embedding vector, and a learned position vector is added so order matters.
Stacked transformer blocks let every token attend to the tokens *before* it
(the causal mask) and refine its representation. A final linear layer turns each
position's vector into a probability distribution over the vocabulary, and the
model is trained to make the true next token likely. Generating text is just
sampling from that distribution one token at a time and feeding it back in.

## Data note

Training data comes from **public sources and the author's own notes** in
software development and cybersecurity. No private, customer, or personal data
is used for training.

## Roadmap

- [x] Milestone 1 — local from-scratch GPT, char-level, CPU
- [~] Milestone 2 — **in progress:** from-scratch BPE tokenizer (`tokenizer_bpe.py`)
  and a curated, license-safe dev+security corpus (`data/build_corpus.py`, ~6 MB
  of IETF RFCs) are done. Still to do: wire the BPE tokenizer into training and
  move to a free cloud GPU.
- [ ] Milestone 3 — scale params/data, evaluation harness
- [ ] Milestone 4 — inference endpoint

## License

MIT — see [LICENSE](LICENSE).
