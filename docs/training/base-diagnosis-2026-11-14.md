# Base diagnosis: inaccessible document bodies

The next intervention is to correct base-data exposure. This audit establishes a
sampling defect, not how much fixing it will improve coding. The accepted base
checkpoint is unchanged; no new GPU training or deployment occurred.

## Confirmed defect

The historical sampler draws a random token position, snaps it back to the
containing document's start, and takes 1,025 tokens (1,024 inputs and shifted
targets). Long files repeat their prefixes; later code cannot be reached. Longer
documents get more draws because the initial position is token-weighted.

The verified training-state checkpoint `f3685f83…9df7` records `doc_aligned=True`,
context 1,024, batch 16, accumulation four and the expected corpus hash. The source
at `783f3daa` has the same sampler. Base training, validation and old SFT replay all
used it. The union of **all possible** sampled input/target intervals, including
shard-end clipping, gives these bounds:

| Split | Stored tokens | Possibly reachable | Never reachable | Unreachable |
|---|---|---|---|---|
| Train | 631,218,245 | 265,338,648 | 365,879,597 | 57.96% |
| Validation | 8,001,206 | 2,780,211 | 5,220,995 | 65.25% |

This is possible exposure, not a claim that every reachable token was visited.
Increasing steps with the old sampler cannot expose the inaccessible bodies.
There are 153,127 long training and 1,606 long validation documents.

All shard/tokenizer hashes match `57af602d…fcfd`. Indexes are ordered and in range;
all document ends contain EOT; token IDs fit the vocabulary. One empty train text
row explains the JSON/shard document-count difference.

## Data, format and history

- The accepted artifact has 70,224,640 parameters, 12 layers, width 640, ten heads,
  context 1,024 and a 16,384-entry code BPE tokenizer. These dimensions are consistent.
  Historical overrides, not current preset defaults, determine its actual recipe.
- The shards contain 441,439 train and 4,670 validation EOTs, but **zero user or
  assistant delimiter IDs**. Reserved IDs exist; raw pretraining supplied no chat
  examples. This is expected for a raw code LM, not independently a tokenizer bug.
  It helps frame the completion/chat gap without proving SFT cannot teach chat.
- Prose is 0.72% of train bytes versus 20.24% of validation bytes. Mixed validation
  loss is not a code-only generalization metric. Metadata groups are disjoint, but
  only Stack groups are repositories: CPython uses top-level modules, prose chunk
  batches. This does not establish a whole-repository CPython holdout.
- In file-uniform samples of 512 code files per split (seed 1729, Python 3.13.16),
  511 parse in each split. 1,565/3,075 train function definitions and 1,352/2,890
  validation definitions start beyond the prefix. Return/raise counts beyond it
  are 938/1,946 and 1,125/2,073. Parsing and imports do not prove functional or
  security quality. The sample is not token-weighted.
- Logs contain finite numeric values and a consistent schedule. Prefix-sampled
  train/validation losses were 1.3968/1.5661 at step 8,000 and 1.1315/1.5849 at
  step 19,000. This supports overfitting of exposed prefixes, not adequate learning
  of the whole corpus. The earlier conclusion that only more distinct data could
  help was premature. 524,288,000 nominal target tokens at the accepted step do
  not imply 83% unique coverage when most document bodies have zero probability.

Existing functional benchmark results for these weights remain valid. Neither
authored-data trial solved a new task. The sampler defect and unseen role protocol
must inform the interpretation; a larger model is not yet a proven remedy.

## Read-only loss check

Unchanged accepted weights, CPU float32, 24 seeded 1,024-token windows per cell:

| Split | Prefix loss, nats | Uniform loss | Prefix perplexity | Uniform perplexity |
|---|---|---|---|---|
| Train | 1.1855 | 1.8108 | 3.27 | 6.12 |
| Validation | 1.5917 | 1.7364 | 4.91 | 5.68 |

Window standard errors are 0.097/0.142 for train and 0.104/0.106 for validation.
These distributions sample different content; the small loss comparison is not a
causal ablation or capability benchmark. Train/validation source mixes also differ.
Historical prefix losses and uniform losses are not interchangeable baselines.
The base hash was verified before and after measurement. Corpus code was parsed
as data and never executed; test splits/private holdouts were not read.

## Repair and compatibility

New base runs, direct shard draws and SFT replay use `uniform_windows_v1`. Legal
start positions have equal probability across shards, weighted by `tokens - context`.
There is no document-start snap or overweighted clipped tail. Every token in an
eligible shard can occur in an input or target. EOT markers stay intact; windows
may cross EOT, without a promise of attention isolation between documents.

Exact resumes retain `legacy_doc_prefix_v1`, or the historical random/clipped mode
when `doc_aligned=False`. New checkpoints record their mode explicitly. Changing
the mode on exact resume is rejected. SFT checkpoints without a replay-mode field
retain historical prefix sampling. This preserves trajectories and does not repair
existing weights retroactively. Changing distribution requires a separate experiment.

Tests cover the unreachable-body reproducer, final-window reachability, cross-shard
probabilities, historical snapping/clipping, explicit training/validation/replay
wiring, and identical resumed weights for pre-version checkpoints. The CPU
dropout/replay resume verifier passes with uniform sampling. GPU verification of
the new mode is deferred to the next experiment.

## Next controlled experiment

Prepare a separate weights-only warm-start path from the pinned accepted inference
checkpoint, with fresh optimizer/RNG and explicit lineage. Do not combine step-8,000
weights with step-9,176 optimizer state or call a changed distribution exact resume.
Compare legacy-prefix control with uniform treatment using identical initialization,
data and schedule. An initial cap of 200 free-T4 steps per arm, batch eight x
accumulation four, context 1,024, LR 1e-5 and 20 warmup steps gives 6.55M target
tokens per arm. This pilot is not enough to recover all missed training.

Predeclare fixed code/prose-stratified uniform validation windows alongside the
historical metric. Compare both arms under both distributions; require improved
uniform code validation beyond measurement noise before expanding. Then compare
unchanged functional/behavior benchmarks with matched decoding. Preview/production
gates remain intact. A sampler fix or lower loss alone never qualifies a served model.

## Evidence and reproduction

- [Token/source/AST audit](base-data-audit-2026-11-14.json)
- [Historical checkpoint/configuration and curves](base-history-audit-2026-11-14.json)
- [Read-only losses](base-loss-audit-2026-11-14.json)
- [Uniform replay resume, CPU](uniform-resume-cpu-2026-11-14.json)

```bash
python -m training.base_audit --data-dir ~/llmfs-data/corpus-v3/shards \
  --corpus-dir ~/llmfs-data/corpus-v3 --samples 512 --out base-data-audit.json
python -m training.base_audit --data-dir ~/llmfs-data/corpus-v3/shards \
  --ckpt BASE.pt --expect-checkpoint-sha256 1164213e635d376946ec331d5ff238cf89f492788554a953010f59bd5f57a593 \
  --windows 24 --out base-loss-audit.json
```
