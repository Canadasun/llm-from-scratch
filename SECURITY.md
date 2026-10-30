# Security

This project is a cybersecurity/software-development LLM built from scratch. Its
own pipeline is held to the standards it is meant to teach.

## Reporting a vulnerability

Open a private report via GitHub Security Advisories on this repository, or
contact the maintainer. Please do not file public issues for security problems.

## Pipeline security practices

- **Safe checkpoint loading.** Model checkpoints are loaded with
  `torch.load(..., weights_only=True)`. A PyTorch checkpoint is a pickle and can
  execute arbitrary code on load; `weights_only=True` restricts loading to
  tensors and plain data, so an untrusted `.pt` file cannot run code. Never load
  a checkpoint from an untrusted source without this.
- **No secrets in training data.** `safety/scrub.py` redacts private keys, API
  tokens, JWTs, cloud keys and passwords before any text is written to the
  corpus; it runs by default in `data/build_corpus.py`. Disabling it
  (`--no-scrub`) prints a warning. A model can memorize and emit its training
  data, so secrets must never enter it.
- **Public/self-authored data only.** No customer, personal, or private business
  data is used for training. See `data/README.md`.
- **Transport.** Corpus sources are fetched over HTTPS.
- **Dependencies.** Minimal dependencies (PyTorch); pin and audit before any
  deployment. Run `pip install` only from trusted indexes.
- **Untrusted inputs.** Treat any downloaded corpus file or checkpoint as
  untrusted data: keep it out of the import path and pass it as data, not code.

## Use safety

The model is gated by a dual-use-aware policy (`safety/content_policy.py`) and
governed by `docs/RESPONSIBLE_USE.md`. The policy is a transparent first layer,
not a guarantee; it is designed to be read, tested and strengthened as the model
grows in capability.
