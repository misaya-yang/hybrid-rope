# OLMo-2 1B 4K-Only Routing Conversion

Date: 2026-07-26

Status: `RAW_BACKED_DUAL_COPY_FROZEN`

Concern mapping: `R27bE.2`, `R27bE.5`, `AC.2`

Evidence role: post-submission mature-model supporting evidence

## Question and claim boundary

This experiment asks whether a mature, approximately 1B-parameter Instruct
model can convert a fixed EVQ frequency substrate into autoregressive 2x
long-context retrieval ability using only 4K LoRA training.

It does **not** isolate pure interior Cosh allocation, establish full RULER or
downstream superiority, or solve 4x extrapolation. Native and EVQ use the same
checkpoint, training rows and order, token budget, rank-64 Q/K/V/O LoRA,
optimizer, and evaluation rows; their fixed `inv_freq` tables are the intended
scientific difference. A second EVQ training seed tests stability, while the
matched Native comparison is the first seed.

## Protocol

| Item | Value |
| --- | --- |
| Checkpoint | `allenai/OLMo-2-0425-1B-Instruct` |
| Checkpoint SHA-256 | `36d044c73655bb904f822915e6294ba3dae8e6e1af5e703e9d452f2d6a3a294f` |
| Physical training length | 4,096 tokens only |
| Stage A | 611 steps, 20,016,360 full-token CE tokens |
| Routing stage | 300 steps; `routing, routing, natural` family pattern |
| Routing-data generator | official RULER `scripts/data/synthetic/niah.py`; custom train/calibration templates and disjoint generated rows |
| Adaptation | Q/K/V/O LoRA, rank 64, alpha 128; LM head frozen |
| Batch | micro-batch 4, accumulation 2, global batch 8 |
| Optimizer | fused AdamW, betas `(0.9, 0.95)`, weight decay 0 |
| Learning rate | Stage A `1e-4`; routing `5e-5`; cosine schedule |
| Seeds | matched Native/EVQ `20260725`; EVQ replication `20260726` |
| Generation task | official RULER `niah_single_1`, greedy autoregressive |
| Primary metric | strict first generated number equals the gold number |

The completed runs did not save Adam moments. The exact optimizer
configuration, seeds, code, data, and deterministically reconstructed
micro-batch row order are frozen, but a nonexistent optimizer state is not
claimed as retained.

## Frequency and adapter identity

| Artifact | SHA-256 |
| --- | --- |
| Native `inv_freq` float32 | `dde15c31724177356ae954d6e11fb337e6fccef56e4520a905cac3f0d9885b34` |
| EVQ `inv_freq` float32 | `917a52426b4ac986545c8ec73b115daae3c6515d6b9047f09d30c972ea1a4607` |
| Native final adapter, seed 20260725 | `6570ab94aec68431dd4e261eb3ef342ef72253357aa65a0d37b0a018df2f3f8d` |
| EVQ final adapter, seed 20260725 | `95ceeb70117c73233915760a9756b9b2a98416ec188b125054da8ced75cad16a` |
| EVQ Stage-A adapter, seed 20260726 | `805dfd8412de2b35c1d4fd639a64e9f14a5e91de2af2e2ee0877b87c4a795a38` |
| EVQ final adapter, seed 20260726 | `fdf6dfc249cb216c3effe22a2ee96fe439a5aff9a11a99007f022c5fe7c3b085` |

## Natural-text NLL

Mean NLL after the final routing stage:

| Arm | 4K | 8K | 16K |
| --- | ---: | ---: | ---: |
| Native, seed 20260725 | 2.2354 | 3.7353 | 4.8511 |
| EVQ, seed 20260725 | 2.5480 | 2.7033 | 2.9245 |
| EVQ, seed 20260726 | 2.5418 | 2.6891 | 2.8989 |

EVQ trades worse in-range 4K NLL for substantially better 8K/16K NLL. The
second EVQ seed closely reproduces this trajectory.

## Routing calibration

Both EVQ runs use the complete held-out 128-pair calibration as the formal
value:

| Training seed | Answer NLL | Median full-vocabulary rank | Token exact | Source-preference positive |
| ---: | ---: | ---: | ---: | ---: |
| 20260725 | 0.2196 | 1 | 96.22% | 96.22% |
| 20260726 | 0.1226 | 1 | 97.92% | 96.48% |

For seed 20260726, the corresponding initial 16-pair probe was NLL `8.1598`,
median rank `53`, and token exact `15.63%`. The formal seed-20260725 value is
96.22%, not the 97.9%/100% figures printed by an eight-pair intermediate
training-log slice.

## Autoregressive RULER results

The initial matched screen used 20 official `niah_single_1` examples per
length:

| Length | Native strict exact | EVQ strict exact |
| ---: | ---: | ---: |
| 4K | 20/20 | 19/20 |
| 8K | 0/20 | 14/20 |
| 16K | 0/20 | 1/20 |

At 4K, EVQ's official substring score is 20/20, but strict first-number exact
is 19/20 because one prediction appended a digit. Strict exact is therefore
the primary number.

The same 8K generator seed was then extended from 20 to 100 examples. The
first 20 rows and their substantive predictions exactly match the earlier
run.

| Arm / training seed | Strict exact | Official substring | Wilson 95% CI for strict exact |
| --- | ---: | ---: | ---: |
| Native / 20260725 | 0/100 | 0/100 | `[0.00%, 3.70%]` |
| EVQ / 20260725 | 69/100 | 71/100 | `[59.37%, 77.22%]` |
| EVQ / 20260726 | 67/100 | 67/100 | `[57.31%, 75.44%]` |

For the matched seed-20260725 comparison, EVQ-only correct is 69 and
Native-only correct is 0; the exact two-sided McNemar value is
`3.39e-21`. On the 80 newly added examples, Native is 0/80 and EVQ is
55/80. Across the two EVQ training seeds on the same 100 rows, 63 examples
are correct for both, 6 only for seed 20260725, 4 only for seed 20260726,
and 27 for neither.

The two EVQ percentages are seed-stability evidence on the same test rows;
they must not be pooled as 136 independent successes out of 200.

### Post-hoc source-gap decomposition

The aggregate 8K score is not uniform over source-to-generation distance.
The 1,024 frozen 4K routing rows have a maximum source-to-answer gap of 3,933
tokens. Splitting the same 100 aligned evaluation rows at that
training-derived threshold gives:

| Arm | Gap at most 3,933 | Gap above 3,933 |
| --- | ---: | ---: |
| Native / 20260725 | 0/50 | 0/50 |
| EVQ / 20260725 | 48/50 | 21/50 |
| EVQ / 20260726 | 48/50 | 19/50 |

Within-versus-beyond odds ratios are 33.14 and 39.16 for the two EVQ seeds
(two-sided Fisher exact \(p=2.56\times10^{-9}\) and
\(2.59\times10^{-10}\)). All seed disagreement occurs beyond training-gap
support. This is a post-hoc diagnostic, not a preregistered endpoint, but it
materially narrows the interpretation: EVQ produces reproducible
beyond-training-gap retrieval that Native lacks, while performance declines
sharply with distance. The result is not uniform solved 2x retrieval.
Full audit:
`OLMO2_N100_GAP_STRUCTURE_AUDIT_20260726.md`.

### Fresh all-long-gap confirmation

A follow-up generated a new, disjoint 8K n=100 set in which every
source-to-generation gap exceeds 3,933 tokens. It reused the three frozen
adapters and performed inference only:

| Arm / training seed | Strict exact | Wilson 95% CI |
| --- | ---: | ---: |
| Native / 20260725 | 0/100 | `[0.00%, 3.70%]` |
| EVQ / 20260725 | 49/100 | `[39.42%, 58.65%]` |
| EVQ / 20260726 | 48/100 | `[38.46%, 57.68%]` |

The fresh query, source-key, source-value, and answer identities have zero
overlap with routing train, calibration, and the original 8K n=100 set. The
two EVQ seeds agree on 81/100 correctness outcomes; 39 are correct for both,
10 only for the first seed, nine only for the second, and 42 for neither.
This confirms the qualitative beyond-training-gap result without relying on
the post-hoc 50-row subset. The exact rate is not invariant to distance
composition: the fresh set has a smaller mean gap than the original
beyond-support subset. Full record:
`OLMO2_FRESH_ALL_LONG_GAP_N100_20260727.md`.

## Leakage and pairing audit

Routing train/calibration versus the 4K/8K/16K evaluation sets have exactly
zero overlap for:

- query identifiers;
- gold answers;
- source keys;
- source values.

The 8K n=100 extension preserves zero overlap on all four fields. The original
8K and 16K cells share 19/20 query/value identities and are therefore a paired
length comparison, not independent datasets. Training and evaluation use the
same official RULER NIAH generator family. The rows, identities, values, and
templates are disjoint, but the protocol is benchmark-family matched. This is
within-task transfer, not benchmark-independent or unseen-task evidence.

## Held-out-task transfer audit

A follow-up 4K-only inference screen tested whether the frozen adapters retain
or transfer capability beyond the numeric `niah_single_1` task used for
routing adaptation. It used the same 20 rows per arm and the official
autoregressive RULER metric. UUID retrieval has one reference and is exact;
variable tracking reports mean reference recall:

- `niah_multikey_3`: UUID keys and values with many key/value distractors;
- `vt`: five-hop variable tracking, a different task family.

| Arm | UUID distractor retrieval | Variable tracking |
| --- | ---: | ---: |
| Untouched Native | 55% | 25% |
| Native-LoRA, Stage A only | 30% | 8% |
| Native-LoRA, final | 40% | 15% |
| EVQ injected, no adapter | 0% | 0% |
| EVQ Stage A, 20M natural tokens | 0% | 0% |
| EVQ-LoRA, final | 0% | 0% |

Predictions are non-empty. EVQ arms typically copy a wrong UUID distractor on
`niah_multikey_3` and repeat the queried numeric value rather than output
variable names on `vt`. The failure is therefore not an empty-generation or
scoring artifact.

This localizes two effects. LongAlign-only Stage-A adaptation narrows existing
Native capability before any routing supervision: UUID retrieval falls from
55% to 30%, and variable-tracking recall from 25% to 8%. The routing stage
partially recovers these Native scores to 40%/15%, but not to the untouched
level. Separately, full EVQ frequency replacement immediately drives both
tasks to zero; strong natural-text NLL after Stage A does not recover them.
The routing stage learns the trained numeric single-needle behavior but does
not transfer to these held-out tasks. Because both final EVQ scores were below
the pre-registered 50% 4K competence gate, neither task was run at 8K and no
second-seed transfer evaluation was launched.

## Frozen evidence

Seven read-only evidence packages retain adapters, raw JSON/JSONL outputs,
per-example predictions, logs, data/code snapshots, reconstructed row order,
and file-level SHA-256 inventories:

| Package | Files | Inventory SHA-256 |
| --- | ---: | --- |
| Matched Native/EVQ seed 20260725 | 51 | `717b894e6924043165b1b06f292f2f0c8e30532b942e4c0fd93cf9c16544ef2d` |
| Matched 8K n=100 extension | 7 | `5fd00a1ab7f35cbd279d6fc0c9c2702586e783905c2e873b101132b12b33769b` |
| EVQ seed-20260726 replication | 10 | `b2a997ce74ce9ec929f13182d72a98cf9204535a2d101a1b14c7ea78f901ac7e` |
| Held-out 4K RULER transfer audit | 17 | `933861f150a3f3da9a21409317d1dce992f86dc7bd65355bd9ce7295a5c229a4` |
| Native Stage-A held-out transfer addendum | 8 | `50b16aa16de00acbb69e2f97b9b1ccd82cee5c6e4ced16c1aeb22aecbe7db869` |
| 8K n=100 source-gap audit | 3 | `5a53b3f5110a8ca89e0325170926df7cdf459f032cb4c2d7dfb039b5027fae34` |
| Fresh all-long-gap 8K n=100 | 26 | `fa49290be2675ce07eb7bc698cf7a4e521d964dba55c2e67c1fa5811f952463b` |

The combined archive SHA-256 is
`a48505d5c51be4f60d3c87bb118924b935d17cda26eb31416acf4b024d07d383`.
The separate held-out-task archive SHA-256 is
`bc5f17507750f6400bf8142b52a5447c68c7c67ece6fb59a65150cc097323836`.
The Native Stage-A addendum archive SHA-256 is
`4e0665c6c222b9279444a148ec52d4224c66d5f4412b176a00730c218035c9fd`.
The CPU-only source-gap audit archive SHA-256 is
`07751754fa73a8e991e11e545497bbd40bfc0e17dc3ba1f36e930de9f2209596`.
The fresh all-long-gap archive SHA-256 is
`e91e67ce1bc316a562037930e7a375ee4eb711b72ea5961da33b74afab06d0de`.

## Reviewer-safe conclusion

Under this matched mature-model protocol, 4K-only counterfactual routing LoRA
fits both substrates at 4K, but Native fails on all 100 8K examples while EVQ
achieves 69/100 strict exact. A second EVQ training seed gives 67/100 on the
same rows. On the 50 rows whose source-to-generation gap exceeds every routing
training gap, the two EVQ seeds score 21/50 and 19/50 while Native remains
0/50. On a fresh, disjoint set containing only beyond-training-gap rows,
Native remains 0/100 while the same EVQ adapters score 49/100 and 48/100.
This supports reproducible, distance-sensitive autoregressive capability
conversion beyond training-gap support for the tested OLMo-2 1B Instruct,
EVQ-plus-LoRA, single-task setting; it does not establish uniform solved 2x
retrieval.

The held-out-task audit directly rejects a broader interpretation. Even
Native LongAlign-only LoRA loses substantial 4K task capability, while full
EVQ injection drives both held-out tasks to zero and neither natural Stage A
nor the final routing stage restores them. The result does not establish pure
Cosh attribution, general downstream transfer, complete RULER performance,
no catastrophic forgetting, or reliable 4x capability.
