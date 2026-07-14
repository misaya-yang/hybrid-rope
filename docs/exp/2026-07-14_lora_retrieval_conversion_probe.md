# LoRA retrieval-conversion probe (seed 42, 2026-07-14)

## Status

This is supporting, mechanistic, single-seed evidence. It does not modify or
replace any paper result. The experiment reused the matched 300-step Geo and
EVQ LoRA adapters and kept both source adapters unchanged.

## Bottom line

The 50-step, answer-only, 8K retrieval micro-tune did not convert EVQ's lower
16K gold-answer NLL into robust autoregressive retrieval. On 20-case held-out
true-16K S-NIAH and KV probes, Geo and EVQ were both 0% before and after the
micro-tune. On the frozen 19-row capability slice, both substrates' mean
gold-answer NLL became slightly worse after the micro-tune.

The result rejects the simple hypothesis that a small amount of short-context
retrieval supervision automatically turns the EVQ substrate advantage into
long-context task ability. It does not reject the narrower frequency-substrate
mechanism: EVQ still has much lower 16K gold-answer NLL and stronger
counterfactual source dependence than Geo.

The subsequent attention gate found a real EVQ target-ranking advantage, but
the matched sparse-conversion pilot did not turn it into retrieval. Target
block rank is therefore not a sufficient conversion criterion.

A follow-up 5090 causal decomposition shows that the ranking signal is not
merely correlational. Removing the answer block from the 32 frozen retrieval
heads worsened EVQ by `+0.5270` NLL and Geo by effectively zero; removing it
from every head worsened EVQ by `+1.5055` NLL and Geo by effectively zero.
Dense EVQ therefore uses real, distributed remote-answer information. The
failure is downstream: that information moves the correct tokens into the top
thousands, not to top-1, and neither score sparsity nor oracle block inclusion
closes the remaining readout gap.

## Geo identity

The `native_geo` arm used here is standard Llama RoPE: its inverse-frequency
tensor is equal to the model's original geometric grid. The injection gate
measured a maximum buffer error of `1.80e-08`.

The separate schedule `tau=0, midpoint=true` is not original RoPE. It shifts
every frequency by the same half-cell factor, `0.9025614848`. At 16K, this
midpoint schedule improved base-model diagnostic NLL from `14.8667` to
`13.7662`, but source-pair consistency stayed at 20% for both schedules.
Therefore a midpoint-Geo result must not be labeled as original RoPE.

## Counterfactual frequency and adapter swap

The 16K source-dependence canary separates training-time adaptation from the
runtime frequency tensor.

| Adapter / runtime frequency | Answer NLL | Source-pair consistency |
| --- | ---: | ---: |
| Geo / native | 15.526 | 0% |
| Geo / EVQ cross-swap | 14.134 | 0% |
| EVQ / native cross-swap | 9.246 | 60% |
| EVQ / EVQ | 9.073 | 80% |

Runtime EVQ frequencies improve Geo NLL without creating source binding. The
EVQ-trained adapter retains most of its source binding after swapping back to
native frequencies, while the matched EVQ runtime tensor adds an incremental
gain. In this probe, the dominant effect is training-time LoRA/frequency
co-adaptation rather than a runtime-frequency-only effect.

## Fifty-step retrieval micro-tune

Both matched 300-step adapters received the same 50-step, answer-only update on
500 deterministic synthetic retrieval rows (490 train, 10 validation), with
maximum length 8K, effective batch size 8, learning rate `2e-5`, and seed 42.
Training and frozen-evaluation answers had zero exact overlap. The outputs were
saved as separate adapters; neither source adapter was overwritten.

| Substrate | Runtime | Final reported train loss |
| --- | ---: | ---: |
| Geo | 10.73 min | 0.03231 |
| EVQ | 10.73 min | 0.08435 |

The low in-distribution loss did not predict 16K retrieval conversion, so it is
reported only as a training-health observation.

## Frozen capability slice at 16K

The frozen slice contains 19 RULER-style, NoLiMa, and Qasper rows. The metric
column is exact match outside Qasper; the single post-tune EVQ nonzero event was
one fragile `yes` match and is not treated as evidence of ability.

| Substrate | Stage | Mean metric | Mean gold NLL | RULER NIAH exact |
| --- | --- | ---: | ---: | ---: |
| Geo | matched 300-step adapter | 0.0281 | 7.0634 | 0% |
| Geo | + 50-step retrieval tune | 0.0000 | 7.4889 | 0% |
| EVQ | matched 300-step adapter | 0.0044 | 4.8513 | 0% |
| EVQ | + 50-step retrieval tune | 0.0526 | 5.2295 | 0% |

EVQ's gold-NLL advantage over Geo is `2.2121` before and `2.2594` after the
micro-tune, but it is not expressed as robust exact-match retrieval.

## Twenty-case held-out generation probe

| Substrate | Stage | S-NIAH 8K | S-NIAH true 16K | KV true 16K |
| --- | --- | ---: | ---: | ---: |
| Geo | before | 100% | 0% | 0% |
| Geo | after | 100% | 0% | 0% |
| EVQ | before | 85% | 0% | 0% |
| EVQ | after | 90% | 0% | 0% |

The S-NIAH prompts were tokenizer-checked: the nominal 8K and 16K examples
contained 7,822 and 15,793 input tokens, and both the question and needle were
retained.

The legacy KV generator had a measurement bug: its nominal 8K and 16K prompts
contained only 3,358 and 6,827 input tokens. Its former Geo 95% result is
invalid as 16K evidence. After calibrating the pair count and adding a
fail-closed token-length gate, all 20 corrected KV prompts contained
15,940--16,207 input tokens. All four corrected cells scored 0%. The bug did
not affect the S-NIAH result above.

## True-16K attention-score gate

Retrieval heads were frozen by pooled Geo/EVQ reciprocal block rank at 8K,
then evaluated on ten 16K passkey cases. Every evaluated prompt contained
exactly 16,384 tokens.

| Measure over 32 frozen retrieval heads | Geo | EVQ |
| --- | ---: | ---: |
| Median target-block hit@16 | 18.75% | 64.06% |
| Range of case-level hit@16 | 3.12--34.38% | 62.50--68.75% |
| Median sparse/dense answer-mass ratio | 0.000 | 1.033 |

EVQ beat Geo on hit@16 in all ten paired cases; the median paired difference
was +48.44 percentage points. This passes the explicitly exploratory 16K gate.
It does not override the preregistered 32K stop: the earlier 32K mass-gain
criterion failed.

## Dense versus sparse 16K pilot

The passing 16K gate triggered a ten-case checkpoint-only pilot with identical
128-token blocks, 16 selected remote blocks, a 1,024-token local window, and
four sink tokens. `score` selects blocks by exact per-query-head maximum q-k;
`fixed` uses the same budget without content-dependent selection. No rotary
positions were reordered.

| Substrate | Mode | Exact match | Answer NLL | Change from dense |
| --- | --- | ---: | ---: | ---: |
| Geo | dense | 0% | 15.5263 | -- |
| Geo | score | 0% | 14.8400 | -0.6863 |
| Geo | fixed | 0% | 14.6804 | -0.8459 |
| EVQ | dense | 0% | 9.0726 | -- |
| EVQ | score | 0% | 9.0521 | -0.0204 |
| EVQ | fixed | 0% | 9.9192 | +0.8467 |

The score-mode NLL difference-in-differences is `-0.6659`: sparsification
helped Geo more, not EVQ. Full-budget sparse and dense logits matched exactly
for both arms (`max_abs=0`, identical top-1), so this is not an implementation
parity failure. The pilot stop conditions fired and the 100-case expansion was
not run.

## 5090 causal decomposition at 16K

The follow-up reused the same unchanged matched step-300 adapters and the same
ten passkey cases. Prompt prefill remained dense. Each intervention changed
only answer-side decode attention, preserved original rotary KV indices, and
used the same 128-token block geometry. Small absolute differences from the
earlier RTX Pro 6000 table (at most `0.017` NLL for dense) are cross-GPU numeric
drift; all causal contrasts below are paired within the 5090 run.

| Answer-side intervention | Geo NLL | Geo change | EVQ NLL | EVQ change | EVQ-minus-Geo effect |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense | 15.5093 | -- | 9.0627 | -- | -- |
| score only on 32 frozen retrieval heads | 15.5090 | -0.0003 | 8.9982 | -0.0645 | -0.0642 |
| remove gold block on 32 frozen retrieval heads | 15.5090 | -0.0004 | 9.5897 | +0.5270 | +0.5274 |
| keep only sink, local, and gold on 32 frozen heads | 15.3454 | -0.1639 | 8.8728 | -0.1899 | -0.0260 |
| remove gold block on every head | 15.4998 | -0.0095 | 10.5683 | +1.5055 | +1.5151 |

All exact-match cells remained 0%. The selected-head deletion hurt EVQ in all
ten cases, while the all-head deletion also hurt EVQ in all ten cases and at
every registered needle depth. The selected-head effect is about 35% of the
all-head effect in magnitude, but the interventions are not additive, so this
ratio is descriptive rather than a head-wise variance attribution. The causal
path is distributed beyond the 32 heads selected by best block rank.

The all-head deletion changes EVQ's geometric-mean correct-token probability
by a factor of `exp(1.5055) = 4.51`; Geo is unchanged. This establishes genuine
source use, but the EVQ probability is still only `1.16e-4` with the source
present.

## Matched-budget oracle inclusion

To distinguish selector misses from collateral pruning, a second intervention
forced the gold block into every head's score-selected set while replacing the
lowest-score selected block. It therefore kept the score mode's token budget
exactly fixed.

| Substrate | Dense NLL | Score NLL | Forced-gold NLL | Forced-gold minus score |
| --- | ---: | ---: | ---: | ---: |
| Geo | 15.5093 | 14.8298 | 14.8246 | -0.0052 |
| EVQ | 9.0627 | 9.0586 | 9.0244 | -0.0341 |

Forced inclusion improved EVQ by only `0.0341` NLL relative to ordinary score
selection, with a substrate interaction of `-0.0289`. Missing the gold block
is therefore not the main reason global score sparsity fails to convert EVQ
into exact retrieval. Score sparsity already preserves EVQ's dense NLL; it
cannot create a decoding policy that dense inference itself lacks.

## Correct-token rank and generation diagnostics

The evaluator additionally recorded strict correct-token rank without storing
or reporting any passkey text. Percentages below are fractions of all
teacher-forced answer tokens.

| Arm / intervention | Answer NLL | First-token median rank | Median mean-token rank | Top-10 | Top-100 | Top-1000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Geo dense | 15.5093 | 33,774.5 | 33,533.5 | 0.0% | 0.0% | 0.0% |
| EVQ dense | 9.0627 | 2,043.0 | 1,506.3 | 3.3% | 23.3% | 60.0% |
| EVQ, gold block removed on every head | 10.5683 | 3,781.5 | 4,035.7 | 0.0% | 0.0% | 43.3% |

EVQ moves the target by more than an order of magnitude in rank and the gold
block causally accounts for a large part of that movement. The first token is
nevertheless still around rank 2,000, so 0% exact match is not a near-miss at
top-1. Across all ten cases, generation had zero gold containment, always used
the full 32-token budget without EOS, and began with the same token.

## Chat-boundary and decode sanity

The frozen passkey builder, unlike the frozen RULER/NoLiMa/LongBench builders,
stores raw prompt IDs even though LoRA training used the Llama-3 chat template.
This was tested as a possible harness explanation without changing the frozen
artifact. The diagnostic added the tokenizer's official five-token user prefix
and five-token assistant boundary, removed exactly ten filler tokens before the
query, and kept every prompt at exactly 16,384 tokens with a unique retained
passkey span.

| Substrate | Raw NLL | Chat-wrapped NLL | Chat exact / containment | Chat first-token median rank |
| --- | ---: | ---: | ---: | ---: |
| Geo | 15.5093 | 15.1504 | 0% / 0% | 33,120 |
| EVQ | 9.0627 | 9.0929 | 0% / 0% | 5,922 |

Chat framing did not restore retrieval. On one chat-wrapped EVQ case, the
custom KV loop and Transformers `model.generate` produced exactly the same 32
token IDs. Full-budget custom attention also matched dense logits exactly in
every run. The zero generation result is therefore not explained by a missing
chat boundary, an off-by-one KV loop, or the custom attention implementation.

## Interpretation and stop decision

Lower language-model NLL, recoverable source signal, attention routing, and
autoregressive answer production are separate gates. The causal result now
shows that EVQ improves the first two and that dense inference genuinely uses
the remote source. It also rejects the proposed dense-dilution explanation at
16K: deleting the source is highly damaging, while score sparsity and even
matched-budget oracle inclusion do little. The remaining bottleneck is the
learned task/readout path that must turn a rank-thousands token into a generated
answer.

Stop this line. Do not continue the 50-step recipe, expand the sparse pilot, or
search sparse budgets post hoc on these ten cases. The retrieval-head and
oracle-block interventions have now answered the narrow causal question; more
sparse variants would not address the observed output-rank bottleneck.

A new 16K retrieval fine-tune would answer a different question--whether the
model can learn 16K retrieval--and could not support a zero-shot conversion
claim. If that question is pursued, the next controlled experiment should use
matched true-16K chat-formatted supervision, checkpoint rank/NLL/containment/EM
early, and stop if EVQ does not move toward top-1 faster than Geo. It should not
repeat the failed 8K, 50-step recipe.

## Artifacts and verification

Raw outputs are in the ignored directory
`results/lora_sparse_conversion_s42_20260714/`. The relevant source adapters
were not overwritten; the two stage-2 adapters were saved as separate
artifacts.

The current-script attention gate is under `phase0_16k_gate_v2/`; the matched
sparse pilot is under `phase1_16k_passkey_pilot_v2/`; corrected KV generation
results are under `true16k_kv_n20/`.

The ignored RTX-5090 bundle contains separate raw and sanitized outputs for the
selected-head causal run, all-head gold deletion, matched-budget oracle
inclusion, correct-token rank, and chat-boundary diagnostic. A sanitized
manifest records the SHA256 of each Geo, EVQ, and summary JSON. No passkey text
is present in the manifest or this report. The manifest SHA256 is
`e546460030f952047a10fdd8ed46acb9a41a0a31de8b3cd86e8dd3a4e25e521b`.

Verification completed:

- local focused tests: 28 passed;
- server focused tests: 6 passed;
- all 500 stage-2 samples passed real-tokenizer preprocessing;
- the 20-case four-model held-out sweep completed;
- the corrected true-16K KV four-model sweep completed;
- the current-script true-16K attention gate completed and passed;
- the matched dense/score/fixed pilot completed and hit its stop condition;
- full-budget sparse/dense logits matched exactly for both arms;
- the latest focused test run passed 8/8 tests;
- all earlier result files were copied locally before the prior shutdown;
- the 5090 causal code passed 12/12 focused tests locally and on the server;
- all five 5090 raw-result pairs were copied locally and summarized;
- full-budget custom attention matched dense logits exactly in every 5090 run;
- custom KV decode matched Transformers generation exactly for the tested case;
- the new 5090 instance was intentionally left running at the user's request;
- no paper file or reported paper number was changed.
