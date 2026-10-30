# Training on a GPU (Milestone 2)

Your laptop has no GPU, so the bigger model trains on a **free** cloud GPU —
Google Colab or Kaggle both give you one. `train.py` already supports CUDA
(mixed precision, LR warmup, pinned transfers); you just run it there.

## Quickest path: Google Colab

1. Open <https://colab.research.google.com> → **File ▸ Open notebook ▸ GitHub** →
   paste `Canadasun/llm-from-scratch` → open `notebooks/train_colab.ipynb`.
2. **Runtime ▸ Change runtime type ▸ T4 GPU**.
3. **Runtime ▸ Run all.**

The notebook clones the repo, builds the corpus, then runs:

```bash
python train.py --preset gpu     # ~15M-param model, BPE tokenizer
python sample.py --prompt "TLS 1.3 " --tokens 400
```

It downloads the trained `checkpoints/ckpt.pt` at the end so you keep it.

## What `--preset gpu` does

| | cpu preset | gpu preset |
|---|---|---|
| block_size | 128 | 256 |
| layers / heads / n_embd | 4 / 4 / 128 | 6 / 6 / 384 |
| parameters | ~0.8M | ~15M |
| tokenizer | char | BPE (vocab 1024) |
| iters | 2000 | 5000 |
| precision | fp32 | bf16/fp16 autocast |

On a T4 this trains in roughly 10–30 minutes. The first run spends ~1 minute
building the BPE tokenizer; that result is cached (`data/.cache_*.pt`) so later
runs skip it.

## Useful flags

```bash
python train.py --preset gpu --max-iters 8000          # train longer
python train.py --preset gpu --vocab-size 2048         # larger BPE vocab (slower build)
python train.py --preset gpu --compile                 # torch.compile (PyTorch 2.x)
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
