# OLMo-2 Instruct 8K n=100 Gap-Structure Audit

Date: 2026-07-26

Status: `RAW_BACKED_DUAL_COPY_FROZEN_POST_HOC`

Concern mapping: `R27bE.2`, `R27bE.5`, `AC.2`

## Audit contract

1. **Reviewer/AC concern.** Determine whether the completed 8K NIAH result
   demonstrates retrieval across a genuinely extrapolative source-to-query
   distance, rather than merely processing an 8K container whose relevant
   source remains within the 4K training-distance support.
2. **Existing evidence.** Matched Native and EVQ 4K-only LoRA score `0/100`
   and `69/100` strict exact at 8K; a second EVQ seed scores `67/100`.
3. **Smallest missing evidence.** Decompose those same frozen rows by whether
   their source-to-generation-boundary gap was observed during routing
   training.
4. **Smallest executable plan.** CPU-only analysis of the frozen training rows
   and three aligned per-example prediction files; no new data, inference, or
   training.
5. **Stop condition.** Stop after one outcome-independent split and descriptive
   fixed-width bands. Do not search thresholds for a favorable result.

This is a post-hoc diagnostic, not a preregistered endpoint. It changes the
safe interpretation of the completed experiment but does not change any raw
prediction or reported aggregate.

## Outcome-independent distance split

For routing training row \(i\), define

\[
g_i^{\mathrm{train}}
=p_i^{\mathrm{answer\ start}}-p_i^{\mathrm{source\ answer}}.
\]

Across all 1,024 frozen 4K routing rows, this gap ranges from 62 to 3,933
tokens. No routing-training row exceeds 3,933.

For each 8K evaluation row, define the comparable generation-boundary gap

\[
g_i^{\mathrm{eval}}
=L_i^{\mathrm{input}}-p_i^{\mathrm{source\ answer}}.
\]

The primary split is therefore fixed at 3,933 tokens using training data only:

- **within observed training-gap support:** \(g_i^{\mathrm{eval}}\le3933\);
- **beyond observed training-gap support:** \(g_i^{\mathrm{eval}}>3933\).

The frozen 8K set contains exactly 50 rows in each group. Native and both EVQ
seeds have identical `row_sha256`, reference, source position, source row,
input length, and local index for all 100 rows.

## Primary result

| Arm | Within support | Beyond support | Difference |
| --- | ---: | ---: | ---: |
| Native, seed 20260725 | 0/50 | 0/50 | 0 pp |
| EVQ, seed 20260725 | 48/50 (96%) | 21/50 (42%) | 54 pp |
| EVQ, seed 20260726 | 48/50 (96%) | 19/50 (38%) | 58 pp |

Wilson 95% intervals for the beyond-support cells are `[29.38%, 55.77%]`
and `[25.86%, 51.85%]`. Within versus beyond support gives:

| EVQ seed | Odds ratio | Fisher exact two-sided \(p\) |
| ---: | ---: | ---: |
| 20260725 | 33.14 | \(2.56\times10^{-9}\) |
| 20260726 | 39.16 | \(2.59\times10^{-10}\) |

These tests quantify the pre-specified support split; they are not a license to
interpret post-hoc subgroup \(p\)-values as a preregistered primary endpoint.

The fixed 1,024-token bands show the same distance decay. Both EVQ seeds are
exact on every row below a 3,072-token gap. At gaps of at least
7,168 tokens, seed 20260725 is `3/13` and seed 20260726 is `2/13`.

## Seed stability

Across all 100 rows, the two EVQ seeds agree on correctness for 90 rows:
63 are correct for both, six only for seed 20260725, four only for seed
20260726, and 27 for neither.

The distance split localizes all seed disagreement:

| Gap group | Both correct | Seed 25 only | Seed 26 only | Both wrong | Agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| Within support | 48 | 0 | 0 | 2 | 100% |
| Beyond support | 15 | 6 | 4 | 25 | 80% |

The paired seed difference is not detectable (`exact McNemar p=0.754`); the
important replicated fact is the common distance-dependent degradation.

## Metric and error audit

For seed 20260725, the within-support strict and official scores are both
`48/50`. Beyond support, strict exact is `21/50` and official substring match
is `23/50`; both official-only successes contain the gold digits inside an
incorrect first number. Seed 20260726 has no strict/official discrepancy.
Strict first-number exact therefore remains the correct primary metric.

Native emits no number on all 100 rows. EVQ failures are mostly wrong first
numbers rather than empty generations, so the distance effect is not an
evaluator crash or a simple refusal to answer.

## Reviewer-safe interpretation

The aggregate `69/100` and `67/100` results are real, aligned, and
reproducible, but they do **not** establish uniform retrieval over an 8K
source-to-query distance. Half of the official 8K rows keep the relevant source
within the maximum distance seen during 4K routing training, and both EVQ
seeds solve `48/50` of those rows.

The stronger and more precise capability statement is:

> Under 4K-only adaptation, EVQ converts the mature OLMo-2 Instruct model from
> 0/50 Native strict exact to 21/50 and 19/50 on 8K NIAH rows whose
> source-to-generation gap exceeds every routing-training gap. The same
> adapters reach 48/50 on the within-support half, showing a large and
> reproducible distance-dependent decline.

This remains useful extrapolative capability evidence because Native is zero
in both groups and the EVQ seeds remain positive beyond training support. It is
weaker than a claim of solved 2x retrieval and must be reported with the gap
stratification. Together with the zero held-out UUID/VT results, it supports a
narrow same-task conversion claim, not broad downstream conversion.

## Fresh confirmatory follow-up

The post-hoc split was followed by a new inference-only evaluation whose
selection rule was fixed to require every source-to-generation gap to exceed
3,933 tokens. The fresh 100 rows are also identity-disjoint from routing
training, calibration, and the original n=100 set.

| Arm | Fresh all-long-gap strict exact |
| --- | ---: |
| Native, seed 20260725 | 0/100 |
| EVQ, seed 20260725 | 49/100 |
| EVQ, seed 20260726 | 48/100 |

The two EVQ Wilson 95% intervals are `[39.42%, 58.65%]` and
`[38.46%, 57.68%]`. This prospectively confirms the qualitative claim that
the frozen EVQ adapters retain nonzero autoregressive retrieval beyond every
training gap while Native remains zero. It does not erase distance
sensitivity: the fresh set has a mean gap of 5,781 tokens versus 6,155 for
the original beyond-support subset, and its exact rate must not be treated as
universal. Full raw-backed record:
`OLMO2_FRESH_ALL_LONG_GAP_N100_20260727.md`.

## Reproducibility

Inputs:

- routing training rows SHA-256:
  `1d7837e40c7bbc5a52d56cd2e111a515521ed6c6b5e561d40cf66e527887bea2`;
- Native n=100 predictions SHA-256:
  `cb1e63b1abb1f3fc0ddafaa6897baa633e62397881c86d3c89666b47f3e873c6`;
- EVQ seed-20260725 predictions SHA-256:
  `bf11d96c20c8854a72233a5aea12f6703d28d6da4a190497febd284e0a4981ed`;
- EVQ seed-20260726 predictions SHA-256:
  `57454509b6656d63113d63d4e5b3c844cb7cbe30a9daa1347c0a930d6de7bb2c`.

Analysis:

- script:
  `experiments/olmo2_lora_maturity/analyze_n100_gap_structure.py`;
- script SHA-256:
  `1b2abd381473af9ae905bca1be6b887437c233b95f8a4f7ef677d2ce1eb32054`;
- result JSON SHA-256:
  `3e2b39b2003e8e071ee9341d10195ad5909947af4f7a1bc02e52a8c122245992`;
- regression test SHA-256:
  `289099bb9c673e62eccdc0045cf1faa9879aaadb37490e66a4ce5f4929fe4ed3`;
- local regression gate: `9 passed`.
- frozen inventory SHA-256:
  `5a53b3f5110a8ca89e0325170926df7cdf459f032cb4c2d7dfb039b5027fae34`;
- local and remote archive SHA-256:
  `07751754fa73a8e991e11e545497bbd40bfc0e17dc3ba1f36e930de9f2209596`.

The result JSON contains the complete 100-row alignment and per-example gap,
group, and correctness fields. It is derived solely from the previously
frozen prediction package; no GPU was used. The archive is read-only on the
server and has an identical local copy.
