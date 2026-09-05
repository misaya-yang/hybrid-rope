# Native-preserving far-pass chord retrofit: experiment report

- **Date:** 2026-08-21/22
- **Status:** complete internal negative; not manuscript evidence
- **Decision:** stop the present CE-only retrofit route
- **Machine-path-free receipt:**
  [`../evidence/FAR_PASS_CHORD_RESULTS_20260821.json`](../../evidence/FAR_PASS_CHORD_RESULTS_20260821.json)

## 1. Decision first

The experiment established a usable systems substrate but not a usable
long-context adapter.

Requests whose total budget is at most 4K delegate to the untouched released
Native attention path and are bitwise identical in the registered smoke.
Long requests use one Flash-compatible augmented attention call with 160
Q/K/V dimensions per head rather than a second softmax. The strongest variant
trains 9.96M parameters, approximately 0.67% of the 1.485B checkpoint.

That construction prevents short-request regression, but none of the four
trained variants learned strict autoregressive retrieval. The strongest
teacher-forced improvement was not capability: adding V/O transport reduced
8K validation NLL from `8.370` to `4.956`, yet the first answer token was top-1
on `0/64` held-out rows and fresh RULER remained zero. Reweighting the first
token and then replacing the virtual query gap with a physical continuous 8K
prefix each moved the rank distribution, but the final continuous run still
achieved only `1/64` first-token top-1 and `0/64` exact answer-plus-EOS.

Therefore:

1. do not promote this result into the ICLR manuscript;
2. do not spend another seed on the same protocol;
3. do not sweep frequencies, ranks, gains, learning rates, or step counts;
4. retain the exact-Native dispatch and augmented Flash operator as reusable
   infrastructure;
5. if the route is resumed, train the missing retrieval alignment directly
   before asking next-token CE to discover a pointer circuit indirectly.

## 2. Reviewer question and experimental contract

The test asked whether a frozen released OLMo-2-0425-1B-Instruct checkpoint
could gain 8K/16K retrieval through a small long-request-only adapter while
making short-request retention a construction property.

For head query (q_i), key (k_j), and relative position

\[
\Delta=p_j-p_i,
\]

the score branch adds

\[
g_\ell\,u_i^\top[I-R_{\Omega_F}(\Delta)]v_j
\]

to the complete Native logit. The eight fixed far-pass wavelengths are
log-spaced from 32,768 to 163,840 tokens, so their phase does not wrap through
16K. The chord is zero at equal positions and has a measured far/near response
ratio of `15.80`. Native Q/K/V/O and the Native frequency tensor remain frozen.

The variants were cumulative and used one seed (`20260821`):

| Arm | Added trainable path | Training exposure | Objective |
| --- | --- | --- | --- |
| QK-gap | 4.72M Q/K residual and score gain | physical 4K; query/answer phase shifted to 8K or 16K | answer+EOS CE |
| QKVO-gap | QK plus learned residual V/O, 9.96M total | same virtual gap | answer+EOS CE |
| QKVO-gap-first | same operator | same virtual gap | 0.5 first token CE + 0.5 continuation/EOS CE |
| QKVO-continuous-8K | same operator and loss | physical 8K positions 0...8191; latter half recursively residual-active | same weighted CE |

All training rows came from independently tokenized FineWeb-Edu natural text.
Each row contains a unique eight-token anchor in the passage and supervises
the following eight source tokens plus immediate EOS. No RULER or NIAH row or
generator was used for training. The gap view contains 896 train and 128
validation rows. The continuous view pairs disjoint packed source rows into
448 train and 64 validation examples; its anchor is always in the first
physical 4K and its query is at the end of the physical 8K sequence.

The three gap runs each use 300 optimizer steps, global batch 8, and 9.828M
input tokens. The continuous run holds steps and global rows fixed but consumes
19.658M input tokens. Consequently it identifies the value of the complete
physical-training intervention, not a pure key-count effect at matched token
compute.

## 3. Results

| Arm | Teacher-forced NLL, 8K | Teacher-forced NLL, 16K | First token, 64 held-out rows | Full exact | Fresh core-4 RULER |
| --- | ---: | ---: | ---: | ---: | ---: |
| QK-gap | `8.376 -> 7.533` | `8.485 -> 7.142` | not separately registered | `0` | `0.0` at 8K/16K |
| QKVO-gap | `8.370 -> 4.956` | `8.481 -> 5.827` | `0/64`, median rank `479.5/283.5` at 8K/16K | `0/64` teacher-forced | `0.0` at 8K/16K |
| QKVO-gap-first | `8.370 -> 5.359` | `8.481 -> 6.137` | `1/64` at each length; median rank `288.5/309.5` | `0/64` teacher-forced | gated off |
| QKVO-continuous-8K | `7.996 -> 5.865` | not trained/evaluated | `1/64`, median rank `63.0` | `0/64` autoregressive | gated off |

The fresh RULER subset used the pinned official generator, 20 rows for each of
`niah_single_1`, `niah_multikey_2`, `niah_multikey_3`, and `vt` at both 8K and
16K. Released Native also scored `0.0`, so this subset is a floor-valued
capability check, not evidence of candidate degradation. The QKVO-gap arm was
also zero. A full 13-family run was not justified after the independent
natural-span capability gate failed.

The continuous 8K run completed all 300 steps with finite loss and gradients:
19,658,400 input tokens in 769.26 seconds, 25.55k tok/s overall, and a 25.43GB
allocated-memory peak. Its active-shape smoke sustained 27.39k tok/s and
verified all 160 trainable tensors had nonzero gradients. The result is not a
runtime failure.

## 4. What the ablation chain says

### Q/K routing alone is insufficient

The QK-only residual lowered teacher-forced NLL, so the branch was not
numerically inert. It nevertheless produced no strict sequence and no fresh
task score. A probability change cannot be promoted to generation capability.

### Content transport is necessary as an operator, but was not sufficient

Giving the same augmented softmax learned V dimensions and a learned O
projection more than doubled the adapter parameter count and produced the
largest NLL decrease. The first-token result stayed zero. Most of the apparent
gain therefore came from easier teacher-forced continuation after the true
retrieval decision, not from locating the remote answer.

### Objective alignment matters, but did not close the gap

Assigning half the loss to the first supervised token improved the 8K median
rank from `479.5` to `288.5` and produced one top-1 row. It worsened aggregate
continuation NLL relative to the unweighted arm and did not produce any strict
sequence. This is a directional diagnostic, not capability.

### Physical exposure matters, but is not the root cause

The virtual-gap rows expose only about 4K keys, and the middle prefix never
passes recursively through the long-query residual. A real 8K request exposes
about 8K keys, changes the softmax denominator, and makes positions 4096 onward
residual-active at every layer. Training on that matched physical structure
improved the first-token median rank from `288.5` to `63.0`. Top-1 remained
`1/64` and exact remained zero. The mismatch was real but secondary.

## 5. First-principles update for practical LoRA

The failed route clarifies that mature-model spectral adaptation has at least
four independent contracts:

1. **Retention:** preserve the Native short function exactly. Length dispatch
   already solves this part more strongly than a retention regularizer.
2. **Representation:** provide a position operator capable of separating far
   evidence. The `I-R(Delta)` chord supplies such a basis without replacing
   Native frequencies, but this run does not identify it as optimal.
3. **Transport:** carry attended content through V/O into the residual stream.
   The QKVO arm supplies capacity, yet capacity alone did not create a pointer.
4. **Credit assignment:** tell the small adapter which source positions must
   influence each answer token. Generic answer CE over a 1.485B frozen model
   gave a weak, delayed signal and optimized continuation probability without
   reliably solving retrieval.

This moves the next question away from frequency allocation. The highest-value
next gate is a source-aligned objective on the same independent natural rows:
for each supervised answer token, the correct source-token position is already
known. Train the residual score route against that source position (or an
equivalent attention-context target), then use answer CE only as the capability
endpoint. A minimal future comparison would be:

- current far-pass QKVO operator plus direct source-position alignment;
- same-parameter content-only positional residual control;
- released Native, all on fresh rows and strict answer-plus-EOS generation.

Only after the internal natural gate clears should the candidate receive fresh
8K/16K RULER and 2Wiki evaluation. Only after a positive candidate exists does
a second seed or a frequency-shape attribution control have decision value.

## 6. Claim boundary

This experiment supports an implementation and diagnosis, not a paper method.
It does not show that phase-chord frequencies are better than another residual
basis, that the adapter solves LoRA practicality, or that long-context
capability improved. It does show why teacher-forced NLL, content capacity,
first-token weighting, and virtual position matching cannot individually
substitute for a strict autoregressive gate.
