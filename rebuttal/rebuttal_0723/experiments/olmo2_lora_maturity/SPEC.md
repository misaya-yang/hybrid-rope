# OLMo-2 maturity × EVQ-LoRA conversion screen

Current execution status: `STOPPED_AFTER_HELD_OUT_TRANSFER_GATE`.

The completed full-transplant recipe is no longer propagated to step-20K/30K.
Any future mature-model training must first follow
`theory_results/OLMO2_MATURITY_ADAPTATION_NEXT_EXPERIMENT_20260726.md`, which
adds progressive Native-to-EVQ morphing and explicit behavior retention. That
file is design-only and does not authorize GPU execution.

## Reviewer/AC concern

This package serves `R27bE.2`, `R27bE.5`, `AC.2`, and `AC.4`: it asks whether
an EVQ frequency intervention followed by parameter-efficient adaptation at
**at most 4K context** can produce language-modeling and source-dependent
capability gains when evaluated beyond the adaptation length on a modern
1.485B-parameter architecture.

## Hard length contract

- Every optimizer step has sequence length at most 4,096.
- 8K, 16K, and 32K are evaluation-only lengths.
- A run trained at 8K or longer cannot answer this package's question.
- The earlier 16K one-token arms are retained only as negative diagnostics;
  they are not templates for the new conversion experiment.

## Existing evidence

At the released step-5K checkpoint, 16K QKVO answer-only LoRA greatly improved
gold-token NLL and rank but did not produce held-out top-1 retrieval. This
separates “the training task moved the target logit” from “the model acquired
a robust source-dependent decision rule.” It does not identify whether the
limitation is base-model maturity, training data, objective, or the reachable
LoRA function class.

The first factorized 16K diagnostic further localizes the failure:

| context bundle | value pool | exact | median rank |
| --- | --- | ---: | ---: |
| train | train | 100% | 1 |
| train | unseen evaluation | 100% | 1 |
| evaluation | train | 6.9% | 21.5 |
| evaluation | unseen evaluation | 0% | 1023.5 |

This rules out unseen answer tokens or a generic vocabulary-readout bottleneck
as the primary explanation. A second factorial diagnostic localizes the
remaining failure:

| position geometry | template | exact |
| --- | --- | ---: |
| train | train | 100% |
| train | unseen evaluation | 42--44% |
| unseen evaluation | train | 14--17% |
| unseen evaluation | unseen evaluation | 0% |

Switching validation/test documents has negligible effect. The evaluation
distractor counts are already a subset of the training counts. The primary
failure is therefore discrete overfitting to the three training source-position
bins; template shift is secondary, and document split is not a first-order
failure. Value shift compounds the full held-out corner but does not cause it
by itself.

A step-5K joint intervention then varied eight templates, twelve positions,
and added paired-margin training simultaneously. It failed to fit even its
training-distribution canary and finished at 16K with exact 0, NLL 11.768,
median rank 1188.5, source gap 0.0026, and 49.2% positive swap-follow. This is
worse than the standard EVQ-LoRA diagnostic (NLL 11.180, rank 1019, source gap
0.0197). The result rejects that overloaded recipe at step 5K; it does not
reject dense position coverage as the isolated intervention.

The first step-30K dense-position arm also trained at 16K and therefore lies
outside the new 4K conversion contract. It supervised only 1,200 answer tokens.
After 300 steps its correctly matched 16K four-cell diagnostic was:

| context | value pool | exact | median rank | source gap |
| --- | --- | ---: | ---: | ---: |
| train | train | 2.8% | 23.5 | 0.00075 |
| train | unseen evaluation | 0% | 959 | 0.00318 |
| evaluation | train | 0% | 28 | 0.00034 |
| evaluation | unseen evaluation | 0% | 762.5 | 0.00302 |

This arm did not learn a source-dependent rule even in its train-context
corner. It must not be resumed or treated as evidence that a 4K full-token
LoRA recipe is ineffective.

## Historical pre-Instruct plan: smallest missing evidence

The remainder of this section through the old positive gate is retained only
to explain how the completed run was designed. It is superseded by the current
stop status and the progressive-morph design linked above; do not execute it.

Use the official step-30K checkpoint first and test whether a 4K-only
EVQ-LoRA curriculum can beat the untouched native-RoPE checkpoint on:

1. held-out natural-text NLL at 4K/8K/16K/32K;
2. disjoint source-dependent retrieval with deletion/swap counterfactuals;
3. a task not used for adaptation, initially RULER single-needle.

The first screen does not prepay for native-RoPE+LoRA controls. A native-LoRA
control is triggered only by a materially positive EVQ-LoRA result.

## Historical data contract

- Official LongAlign-10k, pinned at
  `12f17c4baff1001f0d44c4f8feab09ee2ee8c6dc`.
- Official OLMo-2 Tulu-3 SFT mixture, pinned at
  `d91a0785ade02942520280fb484866fce41e448f`.
- Official OLMo-2 SFT rendering is used. Assistant labels cover assistant
  content and EOS only.
- One deterministic LongAlign row set is rendered at 4K/8K/16K. Every selected
  record fits untruncated at 16K; shorter views preserve BOS and the tail that
  contains the query and answer.
- A separate 4K Tulu replay set supplies general instruction/readout examples.
- Training data is never used as downstream evaluation data.

## Historical execution order — do not execute

1. Verify all three checkpoints and tokenizer identity off GPU.
2. Screen step-30K first because it has the highest chance of conversion.
3. Use 5090-fast execution: BF16, Flash-only SDPA, fused AdamW,
   no activation checkpointing, persistent Inductor cache, and the completed
   16K `max-autotune-no-cudagraphs` receipt unless a two-step shape check
   contradicts it.
4. Stage A starts fresh from step-30K with EVQ endpoint frequencies and QKVO
   LoRA rank 64 / alpha 128. Train only contiguous 4K natural sequences with
   full-token next-token loss until approximately 20M supervised tokens have
   been consumed. This is the direct analogue of the successful dense-label
   8B adaptation, not a one-token proxy.
5. Stage B continues from Stage A while remaining at 4K. Half of optimizer
   micro-batches replay natural text; half use dense paired binding examples.
   Each binding sequence contains 8--16 key/value sources and answer slots.
   Sourced and value-swapped pairs keep background, keys, templates, and actual
   token positions fixed. Start with CE only.
6. Stage-B training positions are recorded as actual source and query token
   indices and actual relative distance, not requested fractions. Training
   values come from a balanced 512-token pool; validation values are disjoint.
7. Evaluate the untouched native checkpoint, EVQ-injected base, Stage A, and
   Stage B. Only after a material EVQ-LoRA positive result, train native-RoPE
   with the identical curriculum to isolate the generic LoRA contribution.
8. Propagate a locked winning recipe to step-20K/10K only if the maturity
   trend is still reviewer-relevant. Do not tune a different recipe per stage.

The source-causal evaluation must split the old all-at-once “evaluation
context bundle” into one-factor changes:

1. train versus paraphrased query template;
2. train versus held-out document source;
3. fixed training bins versus held-out source-position bins, plus a dense
   position sweep for the continuous-position candidate;
4. seen versus held-out distractor count/placement;
5. train versus unseen value tokens.

Run the unchanged train-bundle and full held-out-bundle corners as anchors.
Report accuracy by position decile so success cannot be caused by averaging
easy locations. The intervention is successful only if gains survive the
individual context shifts and their full composition. A broad value pool is
not a priority because unseen values already reach 100% exact under the train
context bundle. A document-domain expansion is likewise not a first-round
intervention because the observed validation/test document swap is neutral.

## Historical positive gate and stop condition

A Stage-A candidate advances only if it:

- improves 16K held-out natural-text NLL over the EVQ-injected base by at
  least 0.2 NLL on the screening set, with most rows moving in the same
  direction;
- causes no more than 0.1--0.15 NLL regression at 4K.

A Stage-B candidate advances to long-context capability evaluation only if it
simultaneously:

- raises 4K held-out full-vocabulary exact/top-1 above the baseline floor;
- increases source deletion/swap dependence in the correct direction;
- preserves the Stage-A 4K natural-text NLL within 0.1;
- generalizes to unseen key, value, document, template, and held-out
  in-range distance intervals.

Only then run 8K/16K/32K source-causal tasks and RULER. NLL/rank improvement
without source-causal top-1 is language-modeling evidence, not capability
conversion. Margin, readout, rank-128, or extra templates are diagnostic
branches triggered by the observed failure mode; they are not a grid.

## Completed held-out-task stop gate

After the positive `niah_single_1` result, the frozen adapters were screened
at 4K on two official RULER tasks not used for adaptation. The registered rule
was to stop before 8K whenever the final EVQ adapter scored below 50% at 4K.

| Arm | `niah_multikey_3` | `vt` |
| --- | ---: | ---: |
| Untouched Native | 55% | 25% |
| Native Stage A | 30% | 8% |
| Native-LoRA final | 40% | 15% |
| EVQ injected | 0% | 0% |
| EVQ Stage A | 0% | 0% |
| EVQ-LoRA final | 0% | 0% |

Both tasks therefore stopped at 4K. No 8K transfer run, extra LoRA seed, rank
sweep, or new mechanism arm is authorized by this package. The completed
result supports task-specific 2x NIAH conversion but falsifies broad transfer
and no-forgetting claims for the current full-EVQ transplant recipe. The
Native Stage-A addendum also shows that LongAlign-only full-token adaptation
causes substantial task narrowing before the routing stage; retention cannot
be treated as an EVQ-only problem.

## Completed n=100 source-gap audit

The aggregate 8K screen contains 50 rows within and 50 rows beyond the maximum
3,933-token source-to-answer gap observed in the 1,024 frozen routing-training
rows. Strict exact decomposes as:

| Arm | Within training-gap support | Beyond training-gap support |
| --- | ---: | ---: |
| Native | 0/50 | 0/50 |
| EVQ seed 20260725 | 48/50 | 21/50 |
| EVQ seed 20260726 | 48/50 | 19/50 |

This CPU-only post-hoc audit does not authorize another run. It narrows
`69/100` and `67/100` to a reproducible but strongly distance-sensitive
same-task conversion result. The full record is
`theory_results/OLMO2_N100_GAP_STRUCTURE_AUDIT_20260726.md`.

## Completed fresh all-long-gap n=100 confirmation

One bounded inference-only follow-up reused the three frozen final adapters
on a new 8K n=100 set. Every row has a source-to-generation gap above 3,933
tokens, and query/source/value/answer identities are disjoint from routing
train, calibration, and the original n=100 set.

| Arm | Strict exact |
| --- | ---: |
| Native seed 20260725 | 0/100 |
| EVQ seed 20260725 | 49/100 |
| EVQ seed 20260726 | 48/100 |

The two EVQ seeds agree on 81/100 correctness outcomes. This confirms
same-task capability beyond observed training-distance support, but it does
not overturn the negative UUID/VT transfer gate or authorize another
mechanism, rank sweep, or training run. The evidence owner is
`theory_results/OLMO2_FRESH_ALL_LONG_GAP_N100_20260727.md`.

## Invalid historical hybrid screen

The earlier zero-training partial-pair, log-blend, and per-head hybrid screens
are **not valid hybrid evidence**. Their evaluator saved the Native frequency
buffer without cloning it and then patched the same buffer to EVQ in place.
The supposed Native snapshot therefore changed with the model.

A receipt-level reconstruction audited all 28 historical hybrid outputs:

- 0/28 active hashes match the frequency tensor declared by their hybrid
  metadata;
- 28/28 active hashes exactly match the tensor produced by the buffer-aliasing
  bug;
- consequently, the recorded zero scores cannot establish the performance of
  any partial-pair, blended, or per-head Native/EVQ hybrid.

The evaluator now clones both Native and EVQ snapshots. Twelve CPU identity
tests cover nine pair partitions, two log-frequency blends, and a per-head
partition. These tests passed under the server's pinned Torch/Transformers
runtime. This correction does not authorize a GPU rerun or promote hybrid into
the current experiment route.

Frozen audit archive SHA-256:
`eae8efd0015ab1a0ae5114ba9fc6367da5aa9f023408063bce8441e1f90b1d6c`.
