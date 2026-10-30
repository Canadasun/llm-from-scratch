# Training on a GPU (Milestones 2–3)

Your laptop has no GPU, so the bigger model trains on a **free** cloud GPU —
Google Colab or Kaggle both give you one. `train.py` already supports CUDA
(mixed precision, LR warmup, pinned transfers); you just run it there.

## Quickest path: Google Colab

1. Open <https://colab.research.google.com> → **File ▸ Open notebook ▸ GitHub** →
   paste `Canadasun/llm-from-scratch` → open `notebooks/train_colab.ipynb`.
2. **Runtime ▸ Change runtime type ▸ T4 GPU**.
3. **Runtime ▸ Run all.**

The notebook installs PyMuPDF (for the NIST PDFs), builds the corpus, then runs:

```bash
python train.py --preset gpu_large   # Milestone 3: ~32M params, RFC + NIST corpus
python sample.py --prompt "Zero trust architecture " --tokens 400
python -m evals.run                  # measure it (see docs/EVALUATION.md)
```

It downloads the trained `checkpoints/ckpt.pt` at the end so you keep it.

## The presets

| | cpu | gpu (M2) | gpu_large (M3) |
|---|---|---|---|
| block_size | 128 | 256 | 512 |
| layers / heads / n_embd | 4 / 4 / 128 | 6 / 6 / 384 | 10 / 8 / 512 |
| parameters | ~0.8M | ~11M | ~32M |
| tokenizer | char | BPE 1024 | BPE 2048 |
| iters | 2000 | 5000 | 6000 |
| corpus | any | RFCs (~6 MB) | RFCs + NIST (~25–40 MB) |
| T4 time | — | ~10–50 min | **~2 hours** |

`gpu_large` is the Milestone-3 step up: deeper, wider, longer context, a larger
vocabulary, and trained on the broader RFC + NIST corpus. It still fits a free
T4 with mixed precision (batch 24 at block 512). The first run spends a few
minutes building the BPE tokenizer; that result is cached (`data/.cache_*.pt`)
so later runs skip it.

Shorten the run with `--max-iters 3000` (~1 hour) if you want a faster first
look; fall back to the smaller, faster M2 model with `--preset gpu`.

## Useful flags

```bash
python train.py --preset gpu_large --max-iters 3000    # shorter run (~1 hour)
python train.py --preset gpu_large --compile           # torch.compile (PyTorch 2.x)
python data/build_corpus.py --no-nist                  # RFCs only (skip NIST PDFs)
python data/build_corpus.py --limit 10 --nist-limit 3  # tiny corpus for a smoke test
python train.py --preset cpu --tokenizer bpe           # BPE on CPU (slow, for testing)
python train.py --device cpu                           # force CPU anywhere
```

## Kaggle

Create a new Notebook, enable a GPU accelerator (Settings ▸ Accelerator ▸ GPU),
then in a cell:

```python
!git clone https://github.com/Canadasun/llm-from-scratch && cd llm-from-scratch \
  && python data/build_corpus.py && python train.py --preset gpu \
  && python sample.py --prompt "TLS 1.3 " --tokens 400
```

## Notes

- PyTorch is preinstalled on Colab/Kaggle, so no install step is needed there.
- The safety layers (data scrubbing, use policy, secure checkpoint loading) apply
  unchanged on GPU — see [RESPONSIBLE_USE.md](RESPONSIBLE_USE.md).
- Even at ~15M parameters this is a small model: expect fluent-looking
  protocol/standards prose, not reliable facts. Scaling is Milestone 3.
