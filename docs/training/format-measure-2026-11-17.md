# Accepted-model response-format measurement

Inference-only run on the unchanged accepted base, using the
[original 24-prompt format probe](format-probe-2026-11-16.md). No training,
optimizer, checkpoint writes, private-holdout access or deployment occurred.

## Frozen measurement

Private free-GPU kernel `idowuaye/llmfs-format-measure-20261117`, version 1,
used source commit `f4caafdd90ad9c8c10a246abbe49115b046206e1` and the
[pinned run configuration](format-measure-kaggle-2026-11-17.json). Inference took
208.3 seconds on the existing Kaggle GPU allocation. The checkpoint SHA256 was
verified before and after generation:
`1164213e635d376946ec331d5ff238cf89f492788554a953010f59bd5f57a593`.

Both module and body requests use the same user/prompt/assistant wrapper,
float32 deterministic math attention, temperature 1/top-k 1, seed 1337 reset for
each response and a fixed 384-token generation budget. No prompt truncation or
response cleanup is permitted. EOT/user/assistant tokens terminate generation.
This tests instruction-conditioned responses; body requests do **not** prefill
the assistant with a function header, so it is not a raw code-continuation test.

All 24 responses reached exactly 384 generated tokens with no terminal special
token. Their stored text, token IDs, prompt counts and terminal metadata were
verified with the exact pinned accepted tokenizer before scoring. All prompt
plus generation lengths fit context 1,024. The existing secret-pattern scan found
no matches. Raw responses and session provenance are committed under
`evals/results/format_probe/`; no response is executed on the host.

## Observations and limits

The [raw scores](../../evals/results/format_probe/scores-base.json) from isolated
workflow [44002813300](https://github.com/Canadasun/llm-from-scratch/actions/runs/44002813300)
passed sandbox preflight and bind the exact response digest. Task-level results,
raw responses, tokenizer and [provenance](../../evals/results/format_probe/provenance.json)
are committed together. The corrected reference/control report also passed in
workflow 44002813233 and retains both mode labels and boolean format verdicts.

| Split / mode | Tasks | Valid syntax | Required shape | Functional passes |
| --- | ---: | ---: | ---: | ---: |
| Training-side / module | 6 | 1 | 0 | 0 |
| Training-side / body | 6 | 0 | 0 | 0 |
| Evaluation / module | 6 | 0 | 0 | 0 |
| Evaluation / body | 6 | 0 | 0 | 0 |

Functional zeros here mean no acceptable function was produced; no shape-valid
model response reached execution. The sandbox canary and reference controls did
execute in isolation. The six evaluation families remain excluded from training.

Every response fails required function shape. Of the 12 module responses, 11 fail
syntax and one parses without the single required function. All 12 assembled
body responses fail syntax. Every family therefore fails the format gate before
functional execution. All responses exhausting the budget establishes censoring
under this recipe; it does not prove that extending the budget would yield a
valid or correct program.

This result does not isolate algorithmic reasoning from instruction/format failure:
there is no valid entry point to test. It also does not prove that the underlying
base cannot continue raw Python, because both conditions use the role protocol
absent from base pretraining. The prior observed completion benchmark already
showed limited raw-code ability. Public small-family results are diagnostic,
not independent final qualification or a security benchmark. Reference output
cases do not prove every property of an implementation, such as absence of input
mutation. No capability improvement or candidate acceptance follows.

## Tooling correction

The new aggregation test found that the existing verifier used `format` both for
the requested mode and its boolean validity verdict, overwriting the mode label.
Reports now use `mode` for module/body and `format` for the boolean. Historical
reference/control pass/fail verdicts are unaffected; their original artifacts
remain unchanged. A regression test preserves both fields, and the corrected
reference verifier runs again in Linux CI.

## Next research decision

Keep the accepted weights and all qualification gates unchanged. Do not add
training steps or generation tokens in response to this failed screen. Before
another training proposal, use a separately frozen matched-task raw-header
continuation diagnostic to distinguish code continuation from role/response-boundary
transfer. Preserve these six evaluation families outside training. Audit format
exposure and response lengths using training-only data; earlier small instruction
trials already failed, so merely repeating them is not justified.

Any subsequent trial must declare a distinct hypothesis, train/evaluation hashes,
matched initialization and a bounded budget before training. Require actual coding
gains, format retention and existing security-category retention, followed by full
frozen qualification before model acceptance. This measurement alone does not
establish a training or deployment decision.
