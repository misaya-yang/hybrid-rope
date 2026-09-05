# Conditional P3: finite same-QK Native attention KL

## Lifecycle and entrance

**Status: `ENTRANCE_FAILED / NOT_EXECUTED`.** No model outcome exists. The
independent K32 owner returned `UNRESOLVED`, not the required
`CONFIRMED_CROSSING`; therefore this asset remains unused. Do not run it,
reinterpret the historical 20-row pilot, or treat the prepared input split as
mechanism evidence.

The sole hypothesis is that K32 normalized-index Native compatibility, if
independently confirmed, is accompanied by a smaller finite perturbation of
the checkpoint's Native attention distribution than physical-x. This is a
diagnosis of a frozen result, not a method candidate or profile selector.

## Frozen identities

- Qwen2.5-0.5B-Instruct artifact: weight SHA-256
  `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`.
- Native / physical / index tensors:
  `6d1e10125bd0468a7cf91c6175a3af31c1bffca24592cf5630f0f8402a8746e3`,
  `b61a58f3e84429e00eaac69a0d9ab43abf89bc193987b2bcbcd3ab3bccd455fb`,
  `8c19ab976f71d30c6409f78a661209a8535ef9f101e8bf42f5bfce6f7817dc5f`.
- Both candidates use s=2 and amplitude `1.0512928913614359`; Native uses
  amplitude one. There is no gain, table, layer or head search.
- Input receipt SHA-256
  `4a3fdcdd037dc02b77b94d910f550dc93d5ce368215bf769c6a2b9bbbc4ae6ce`,
  binding 8 calibration and 8 untouched confirmation documents. It excludes
  128 explicitly supplied natural-input identities and reads no model outcome.
- Mathematical owner:
  [`native_attention_kl.py`](../../../../../scripts/analysis/native_attention_kl.py).

## Exact estimand

Run the checkpoint only with Native RoPE. At every one of its 24 layers,
capture the same Native hidden-state-derived **pre-RoPE** Q/K for all 14 query
heads and two KV heads. Candidate hidden states are never substituted. Apply
the complete finite physical/index frequency tensors analytically, including
Q/K-side amplitude and the model attention scale.

The primary per-head/query value is

```
KL(softmax(Native logits) || softmax(candidate logits))
```

over every causal key from position zero through the query. The implementation
uses split-half RoPE and contiguous GQA head repetition. FP64 analysis is the
estimand; it is not asserted bitwise equal to the BF16 FlashAttention kernel.

### Query and aggregation lock

Each 32768-token input scores final target tokens at positions
`32512..32767`. Their aligned next-token logits come from **all 256 query
positions `32511..32766`**. No query subsampling is allowed. For each document:

1. average equally over the 256 locked queries, every query head, and every
   layer;
2. then average the resulting document values equally over the eight fixed
   documents.

Calibration and confirmation use exactly this aggregation. Layer/head/query
breakdowns are retained only for diagnosis and cannot become weights,
subsets, parameters or follow-up candidates.

Mechanically retain Native-attention-weighted signed slot-delta means and
variances, total-delta variance, off-diagonal cancellation, and the gain term.
These describe aggregate cancellation/reinforcement; they do not identify a
particular slot pair or decompose task loss.

## Guards, split use and decision

Native frequency with gain one must give KL zero within FP64 numerical
tolerance on every inspected cell; nonfinite Q/K/logits/KL or an identity
mismatch stops the instrument. Record checkpoint/data/tensor/code hashes,
runtime, raw per-document/layer/head/query KL and complete aggregation.

Calibration cannot alter the formula, query set, weights or aggregation.
Freeze its raw output/code hashes before opening confirmation. On the eight
confirmation documents, pair physical-minus-index document KL and bootstrap
documents jointly (10,000 replicates, seed `202609028`, percentile 95% CI):

- mean difference > 0 with CI wholly above zero: `SUPPORTS_LOCAL_KL_EXPLANATION`;
- mean difference < 0 with CI wholly below zero: `CONTRADICTS_LOCAL_KL_EXPLANATION`;
- otherwise: `UNRESOLVED_LOCAL_KL_EXPLANATION`.

Calibration/confirmation directions and every document value are reported
regardless of the confirmation decision. No additional document, query,
profile, metric or seed is added after seeing the result.

## Claim ceiling and closed-route distinction

This checkpoint-aware finite logit calculation uses real Native Q/K and signed
slot contributions. It is not the checkpoint-free phase-risk scalar, direct
distance map, or first-order kappa/Fisher ordering closed by `INDEX.md` §3.4.

Even a positive result supports only: on these fixed Native hidden states and
documents, one frozen table produces smaller local attention-distribution KL
and aggregate slot cancellation/reinforcement of the recorded sign. It does
**not** show that KL predicts Native NLL, RULER, candidate quality, causality,
V/downstream-layer feedback, or long capability. It cannot authorize a
selector, correction, G(x;K), new curve, gain change or SOTA expansion.
