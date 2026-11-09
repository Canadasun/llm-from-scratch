# Phase 2: Data pipeline and tokenizer

Status: built and published as private Kaggle datasets on 2026-11-09. Training on this data has not run yet;
the Kaggle smoke run is waiting on the Phase 1 sessions (Kaggle notebook outage).

## What was built

| Piece | File |
|---|---|
| Code-aware byte-level BPE with an incremental trainer (`bpe_code`) | `tokenizer_code.py` |
| Licence allowlist, parse/generated/minified filters, payload rules, secret scrub, MinHash near-dedup, repo split, 13-gram contamination screen | `data/pipeline.py` |
| Corpus builder: download, gate, split, manifest, stratified tokenizer sample, token shards | `data/build_phase2.py` |
| Instruction-set builder: mined functions, mined insecure-pattern reviews, authored rows | `data/build_sft.py`, `data/sft_authored.py` |
| Tests | `tests/test_data_pipeline.py` (16), `tests/test_sft.py` (6) |

Nothing from a dataset is executed: Parquet is read with pyarrow and Python files are only parsed with `ast`.

## Sources

| Source | Pin | Licence handling |
|---|---|---|
| `bigcode/the-stack-dedup`, `data/python` | revision `17cad72c886a2858e08d4c349a00d6466f54df63`, 10 of 144 files | per-row `max_stars_repo_licenses`, every licence must be on the allowlist |
| CPython stdlib | tag `v3.13.9` | PSF-2.0 |
| RFC and NIST prose (existing `data/corpus.txt`) | file hash in corpus manifest | capped at 8% |

Allowlist: MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, ISC, PSF-2.0. Missing or unknown licences are dropped.

Excluded pending an owner decision (recorded in `corpus-manifest.json`):
`CyberNative/Code_Vulnerability_Security_DPO` (looks model-generated, sampled rows corrupted),
`code-search-net/code_search_net` (no per-row licence), OWASP Cheat Sheets (CC BY-SA), MITRE CWE (terms not reviewed).

## Corpus v2 (`llmfs-corpus-v2`)

| Target | Result |
|---|---|
| about 2 GB | 2,000,002,741 bytes |
| at least 80% code | 99.0% (1.98 GB code, 19.2 MB prose) |
| at least 3 bytes per token, vocab 8k to 16k | 3.152 on validation at 16,384 |
| tokens | 631,277,601 train, 8,001,206 validation |

Shard manifest sha256: `ec128b2390c8b292e60c955c88cdff5732935bd86c5fd6659fdef611328baa41`.
Split by repository: 1% validation, 1% held-out test (`holdout_test.jsonl`, never tokenized into shards).

Stack rows: 526,892 in, 450,062 kept. Drops: licence not allowlisted 33,282; Python parse failed 25,277;
generated marker 13,807; minified 2,465; data literal 1,200; low alphanumeric 355; too large 268; near-duplicate 68;
exact duplicate 31; payload rules 77 (keylogger 32, ransomware 28, reverse shell 7, other 10).
Secrets scrubbed in place in Stack rows: 24,757, mostly assigned secrets and URL passwords.
Kept licences: MIT 301,577; Apache-2.0 102,490; BSD-3 38,130; BSD-2 9,207; PSF 461; ISC 389.

Contamination: 13-gram overlap against `evals/data` removed no rows. Phase 3 replaces this with near-duplicate
checks against HumanEval and MBPP.

## Instruction set v1 (`llmfs-sft-v1`)

| Split | Rows |
|---|---|
| train | 7,659 |
| validation | 421 |
| held-out test | 426 |

Train by kind: implement-from-docstring 4,705; summarise 1,609; security review of real code 1,257;
authored secure-coding 36; refusal 27; insufficient information 25.

- Mined functions are top-level, undecorated, documented, and self-contained: every free name is a builtin or an
  absolute import, which is included in the answer. Docstrings must be a clean English sentence.
- Security reviews come from 11 narrow rules (unsafe YAML, shell strings, deserialization, SQL string formatting,
  disabled TLS checks, fast password hashes, `random` for secrets, `mktemp`, dynamic `eval`, unfiltered tar
  extraction, exposed debug servers). Each answer quotes the line, gives the risk conditionally where taint is
  unknown, names the fix, and states it is not a full review. Spot checks found two false-positive patterns
  (HIBP SHA-1 range hashing, SQL identifier concatenation); both are excluded and tested.
- 100 authored rows: 40 secure-coding, 30 refusals with safe alternatives, 30 insufficient-information answers,
  each a different topic. Authored rows are about 1% of the set, so Phase 5 should upsample them.
- Mined rows inherit the corpus repository split. Authored rows split by family. No prompt appears in two splits.
  Every row keeps its repository and licence.

Training-file sha256: `a108d2b333806569bcf5f04653f349a6dfabbf47bef47546ce3ebd725c8fb99f`.

## Reproduce

```bash
python3.13 -m venv .venv-data && .venv-data/bin/pip install -r requirements-data.txt numpy pytest
huggingface-cli login   # accept the Stack terms first
.venv-data/bin/python -m data.build_phase2 --out ~/llmfs-data/corpus-v2 --sources prose cpython stack \
    --stack-files 10 --target-bytes 2000000000 --vocab-size 16384 --tokenizer-chars 50000000 --workers 14
.venv-data/bin/python -m data.build_sft --corpus ~/llmfs-data/corpus-v2 --out ~/llmfs-data/sft-v1 --workers 12
```

The corpus build took 85 minutes on this Mac; the instruction build took about one minute.

## Still open

- Kaggle smoke run on `llmfs-corpus-v2` (after the Phase 1 sessions run).
- Owner decisions on the four excluded sources.
