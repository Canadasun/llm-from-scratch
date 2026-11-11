# Phase 5: Instruction tuning

Status: two candidates trained on 2026-11-10. Neither beats the Phase 4 base on the frozen suite without a category
regression, so the base stays accepted, as the roadmap's exit rule requires.

## Method

`training/sft.py` on Kaggle (one Tesla T4), from the Phase 4 base `best.pt` (step 8,000, `1164213e…`). Loss on the
response tokens only (`<|user|>` prompt `<|assistant|>` response `<|endoftext|>`), 20% of each batch is
`llmfs-corpus-v3` replay, batch 8 x accumulation 4, learning rate 1e-4 with 100 warmup steps, cosine to 10%. The
base is pinned by sha256 and never written. `best.pt` is the step with the lowest instruction validation loss.

| | Candidate 1 | Candidate 2 |
|---|---|---|
| Kernel | `llmfs-p5-s1` | `llmfs-p5-s2` |
| Data | `sft-v2`, manifest `83554e2a…` | `sft-v3`, manifest `0b50c77c…` (#23) |
| Train rows (implement / summarise / review / authored) | 4,705 / 1,609 / 1,257 / 88 | 5,489 / 825 / 1,257 / 88 |
| Upsampling | authored kinds 8x | secure-authored 8x, refusal and insufficient-info 2x |
| Steps run, best step | 1,500, 300 | 1,000, 300 |
| Instruction validation loss (base, best) | 1.969, 1.257 | 1.908, 1.231 |
| Replay validation loss (base, at best) | 1.576, 1.592 | 1.576, 1.574 |
| `best.pt` sha256 | `903906c8…` | `a3cdbb4b…` |
| Wall time | 18 minutes | 14 minutes |

Both `sft-v2` and `sft-v3` passed the 13-gram benchmark decontamination with no hits. The check was confirmed to flag
a planted HumanEval reference.

## Frozen benchmarks

Scored in the CI sandbox by `bench-score.yml`. Behaviour items do not depend on the decoding mode.

| Metric | Base (completion) | Candidate 1 (chat) | Candidate 2 (chat) | Candidate 2 (completion) |
|---|---|---|---|---|
| Parses as valid Python | **63.4%** | 21.8% | 26.6% | 48.8% |
| HumanEval | **2/157** | 1/157 | **2/157** | 0/157 |
| HumanEval+ | **3/157** | 1/157 | 2/157 | 0/157 |
| MBPP | 0/377 | 0/377 | **1/377** | **1/377** |
| MBPP+ | 0/372 | 0/372 | **1/372** | 0/372 |
| Abstention (safe choice) | 5/15 | **10/15** | **10/15** | **10/15** |
| Unsafe recommendation (safe choice) | 4/15 | 4/15 | **5/15** | **5/15** |
| Prompt injection (safe choice) | **5/15** | 4/15 | 4/15 | 4/15 |
| Scoring run | 42802344546 | 42921635445 | 42935819392 | 42935838922 |

Programs: `evals/results/phase4/programs-pilot.json`, `evals/results/phase5/programs-sft2-chat.json` and
`programs-sft2-completion.json` (candidate 1's programs are on branch `phase-5-sft-eval`).

## What the outputs show

- Candidate 1 answered about 135 MBPP requests with a one-line description ("This function returns…") and about 50
  with a refusal. Prompts that contained code were mostly summarise rows, and 27 refusal rows were repeated 8 times.
- Candidate 2 fixed the descriptions (2 left). MBPP prompts include a test `assert` line, and the model now often
  continues in the security-review format ("Fix: … This covers the issue above only…"), 108 times. About 160 answers
  start as code but break, for example `def next_smallest_palindrome(99):`, or loop on whitespace.
- Abstention doubled in both candidates, from the authored insufficient-information rows.

## Conclusion

The tuning engine works and teaches behaviour from small authored sets (abstention 5/15 to 10/15). The 65M base does
not keep its code ability through tuning on about 7,700 mined rows: the parse rate falls and the model mixes up
response formats. The base stays the accepted model. Closing the gap would need instruction data with tests per row
(not possible from mined code without running it) or a larger base, not more tuning steps.
