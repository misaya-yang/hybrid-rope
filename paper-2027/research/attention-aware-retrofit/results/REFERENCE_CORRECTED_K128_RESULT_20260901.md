# Reference-corrected K128 coupling (2026-09-01)

## Passport and decision

- **Status:** `S2_S4_COMPLETE / OLD_16K_NEGATIVE_RECOVERED / COORDINATE_SUPERIORITY_UNRESOLVED`
- **Checkpoint:** the exact Gemma-1.1 redistribution bound by the
  [confirmed Native reference receipt](../evidence/GEMMA_NATIVE_REFERENCE_CONFIRMED_20260901.json).
- **Reference:** `L_config=8192`, independently confirmed operating
  `L_ref=4096`; training history is not relabeled.
  The 16K target is 4x this operating reference but only 2x the configured
  8192 length; it is not described as 4x the documented training context.
- **Law:** unchanged `x_H=.7382780681078285`, `x_L=.366403835112904`,
  `G(x)=clip((x_H-x)/(x_H-x_L),0,1)`, `omega'=omega*s^-G(x)`, `c=.074`.
- **Registration:**
  [`REFERENCE_CORRECTED_K128_PREFLIGHT_20260901`](../preflights/REFERENCE_CORRECTED_K128_PREFLIGHT_20260901.md).
- **RULER owner:**
  [`REFERENCE_CORRECTED_K128_S2_RULER_RECEIPT_20260901.json`](../evidence/REFERENCE_CORRECTED_K128_S2_RULER_RECEIPT_20260901.json).
- **Natural continuation owner:**
  [`REFERENCE_CORRECTED_K128_S2_NLL_RECEIPT_20260901.json`](../evidence/REFERENCE_CORRECTED_K128_S2_NLL_RECEIPT_20260901.json).
- **Completed s4 owners:**
  [`RULER curve`](../evidence/REFERENCE_CORRECTED_K128_S4_RULER_RECEIPT_20260901.json)
  and [`natural continuation curve`](../evidence/REFERENCE_CORRECTED_K128_S4_NLL_RECEIPT_20260901.json).

**The old 16K zero is recovered without refitting the law.** On the same new
16K rows, the old config-reference s2 table remains zero, while the corrected
reference s4 physical table scores `.7250`. Its single static maximum profile
scores `.8775/.7975/.7250` at 4K/8K/16K and passes both measured 4K retention
gates. This supports a reference/request-scale explanation of the prior
negative; it does not establish a pure reference effect independently of the
associated scale and gain change.

Physical x is **not** the demonstrated winning coordinate: s4 index scores
`.9450/.8650/.7950`. Pointwise intervals narrowly favor index, but a
three-length multiplicity sensitivity does not establish uniform superiority
in either direction. The exact continuous law is not identified by this panel.

The reference-correct s2 physical profile passes both tested operating gates:
4K RULER macro improves from `.8700` to `.9650`, while natural 4K suffix NLL
increases only `.025083` (geometric-mean PPL retention about `.975`). At 8K it
recovers RULER `.8600` from Native zero and reduces fresh natural NLL from
`10.548809` to `3.163365`.

However, physical and normalized-index profiles are not reliably separated.
Their 8K macro difference is only `.0050`, with paired 95% interval
`[-.0350,.0475]`. YaRN is nominally ahead at 8K but that comparison also has
an interval spanning zero. This is successful one-hop static extension under
the confirmed reference, not privileged-coordinate identification or SOTA.

## 1. Frozen construction and held-out inputs

Target 8192 fixes `s=8192/4096=2`; the ratio is not searched. Both physical
and normalized-index coordinates use the independently confirmed reference.
The actual Native inverse-frequency tensor and checkpoint config are unchanged.

Physical s2 tensor SHA-256:
`10bdecdb270c576caf99bb6a9aa9da16b4f9c04ec1af52c3021553f9d21bbd59`.
Normalized-index s2 tensor SHA-256:
`f58a9f478dadc2b3cb18251fcb156ebecdbab9637dcf913530090168ec55a610`.
Their shared amplitude is `1.0512928913614359`.

The official-equation YaRN baseline uses the installed HF initializer,
`original_max_position_embeddings=4096`, factor two and its published
amplitude `1.0693147180559945`. The realized tensor is
`cdf74bbded4041842edfef7f5eeb8861b9a52ed5e1688296876c828963b7eb89`;
equation parity passes with maximum absolute difference `3.73e-9` while exact
tensor identity is separately hash-bound. This is not the repository's
different fixed-index YaRN-style operator.

Fresh official core-four RULER uses seed `202609023`, 20 rows per task and
length, manifest
`d5380f1b59bb454633717b4101c24709bde6e3b813c1c8f1d1b4447177754e89`.
It supplied no P0 selection evidence. All arms have identical rows, tokenizer,
greedy decoding, scorer and maximum-profile lifetime. Complete generated IDs
and EOS metadata are preserved alongside the unchanged official metric.

## 2. Complete s2 RULER curve

| Fixed profile | 4K single / mk2 / mk3 / VT | 4K macro | 8K single / mk2 / mk3 / VT | 8K macro |
| --- | --- | ---: | --- | ---: |
| Native | `1/.95/.80/.73` | `.8700` | `0/0/0/0` | `.0000` |
| physical-x | `1/1/1/.86` | `.9650` | `1/.95/.85/.64` | `.8600` |
| normalized-index | `1/1/.95/.83` | `.9450` | `1/1/.80/.62` | `.8550` |
| YaRN | `1/1/1/.81` | `.9525` | `1/1/.75/.84` | `.8975` |

At 4K, physical-minus-Native is `.0950`, paired 95% interval
`[.0400,.15256]`. Physical-minus-index is `.0200`, interval
`[-.0050,.0525]`. At 8K, physical-minus-YaRN is `-.0375`, interval
`[-.1025,.0300]`. Do not rank coordinates or methods from these small,
unresolved contrasts.

Intervals use 10,000 paired row-bootstrap replicates, seed `202609024`,
stratified within the four fixed tasks. They do not estimate checkpoint,
training-seed or task-family uncertainty. The reproducer is
[`summarize_reference_corrected_ruler.py`](../../../../scripts/analysis/summarize_reference_corrected_ruler.py).

## 3. Independent natural-text operating check

Thirty-two fresh documents of at least 16K tokens were selected in source
order, excluding **all 96** P0 natural-document hashes. Each length scores
the same 256-token suffix per document. The holdout manifest is
`e02d46994cd0fd927fe02d9f28bfddb1c707b87ffeb498141376ed8c0cfe1c32`.

| Profile | 4K NLL | change vs Native | 8K NLL |
| --- | ---: | ---: | ---: |
| Native | 3.220942 | 0 | 10.548809 |
| physical-x | 3.246025 | +.025083 | 3.163365 |
| normalized-index | 3.240990 | +.020049 | 3.158023 |
| YaRN | 3.242039 | +.021098 | 3.168670 |

All three static long profiles have finite loss, small measured 4K cost and
large 8K improvement. Their tiny mutual NLL differences are not presented as
a meaningful ranking. PPL retention means `exp(mean NLL_Native - mean NLL_arm)`,
not an average of per-document PPL ratios.

Raw natural rows SHA-256:
`80611b3e48fe8ca8b6343a972d74e70bb901a96c74e57982c15e8c61e7b67c07`;
run-manifest SHA-256:
`2aed73f359dbf5736c7a774b7ae2fa82767239df6e837352be4c2bfdeb4f2959`.
The 256-row evaluation took 92.096 seconds, with peak reserved memory
7,107,248,128 bytes. These are held-out teacher-forced outcomes, not additional
autoregressive capability trials.

## 4. Stage-2 entrance and frozen s4 construction

The preregistered stage-2 entrance passes: physical single-key is `20/20` at
8K and its macro exceeds Native by `.8600`; the separate 4K RULER and PPL
operating gates pass as well.

The fixed second target is 16384, hence `s=4`. It uses the same confirmed
reference, unchanged `G` and coefficient, and one static maximum profile at
4096/8192/16384. No 16K outcome selects the profile. A paired old
config-reference s2 table is a diagnostic control, not a new candidate.

Physical s4 tensor SHA-256 is
`be5c2b3b4ce01d7fe6020cb01d9041e10aad93b9cdb03e989e64b8fa17561423`;
index is `1b908f90aebccc006521b3840b5662c217caa040d173c974527d33c7ea9e9849`.
Their amplitude is `1.102585782722872`. YaRN4 tensor is
`33279a0999c3224cec4c89ad6711e21bbb65f83df71a7b4eb75be83da0c903e5`,
with published amplitude `1.138629436111989`. No runtime table switching or
KV-cache rewriting occurs.

## 5. Complete s4 length curve

Each row below is one maximum-16K static profile, not a different table chosen
at each evaluation length. RULER uses the same seed and 20 rows per task as
stage 1. Natural scores reuse the same 32 holdout documents and target suffixes;
these are paired profile comparisons, not additional independent documents.

| Fixed profile | 4K RULER | 8K RULER | 16K RULER | 4K NLL | 8K NLL | 16K NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native | .8700 | .0000 | .0000 | 3.220942 | 10.548809 | 15.691767 |
| physical-x | .8775 | .7975 | .7250 | 3.295702 | 3.246521 | 3.131111 |
| normalized-index | .9450 | .8650 | .7950 | 3.271593 | 3.223082 | 3.110515 |
| YaRN | .8950 | .8400 | .6575 | 3.244954 | 3.221684 | 3.113652 |

All s4 task vectors, in single / mk2 / mk3 / VT order:

| Profile | 4K | 8K | 16K |
| --- | --- | --- | --- |
| Native | `1/.95/.80/.73` | `0/0/0/0` | `0/0/0/0` |
| physical-x | `1/1/.65/.86` | `1/1/.60/.59` | `1/.85/.20/.85` |
| normalized-index | `1/1/.90/.88` | `1/1/.85/.61` | `1/.90/.40/.88` |
| YaRN | `1/1/.90/.68` | `1/1/.75/.61` | `.95/.65/.25/.78` |

The physical 4K RULER-retention ratio is `1.008621`; its natural PPL retention
is `.927966`, both above `.875`. Its 4K NLL increase is `.074760`, paired
document 95% interval `[.041515,.110733]`. This is measurable Native cost,
not exact Native preservation. Index and YaRN also pass the point gates.

Physical-minus-index RULER differences are `-.0675/-.0675/-.0700`; pointwise
95% intervals are `[-.1325,-.0025] / [-.1300,-.0050] / [-.1425,-.0025]`.
As a **post hoc multiplicity sensitivity**, the same bootstrap draws with
Bonferroni coverage over these three lengths give
`[-.145,.010] / [-.1425,.0075] / [-.1575,.010]`. All span zero. This is a
consistent sample-level tilt toward index, not multiplicity-controlled proof
of dominance, equivalence, or a K trend. The correction is per fixed contrast
family, not a global correction over every comparison in the project.

At 16K, physical-minus-YaRN is `.0675`, pointwise 95% interval
`[-.0175,.1550]`; no SOTA/ranking claim follows. The interval machinery verifies
complete paired rows and valid bounded scores, retains all observations, and
resamples within fixed task strata. It does not assume Gaussian individual
scores or independent arms. With only 20 rows/task, bootstrap coverage and
external generalization remain limited.

Natural s4 raw rows SHA-256:
`6d22653e3ef7efaa6c08ed4c0910d5b91b5ba8d3ff3c2c2b0653c9751636711e`;
run-manifest SHA-256:
`90534fb5920cb80d8eb05f92a48e32ca33f04f3c2b7d7c9e045438dbe381723d`.
The 384-row panel took 362.921 seconds while sharing the GPU with RULER;
peak reserved memory was 7,107,248,128 bytes. Neither elapsed time nor a live
utilization sample is promoted to a matched performance benchmark.

## 6. Paired old-reference control and causal update

The preregistered old `L_ref=8192,s=2,c=.074` table was evaluated on exactly
the same fresh 16K cells as the corrected `L_ref=4096,s=4,c=.074` table.
Its original tensor hash
`fbd2f80a462f3271e65a8cdc9f3acb81b46c4afcf79c3be73509c36ac1f0bf0b`
and amplitude `1.0512928913614359` were unchanged; it was not silently
reconstructed as s4. All four task scores remain zero (80 generations).

Raw result / examples / run-manifest SHA-256 respectively:

- `0f3e067c22249131876ca55eea6ccf3bd14183b5cacdfdd3e4d450161ff5eaa9`
- `55ec9c4e35a4e9951850aa43fe7f8c058a4bec2cc09f91118e7091df2c4fd99d`
- `949e2edd901e94c24c648f4e0ad864782a12474b18aecc0c152bee2779a36736`

The control's data manifest matches the s4 panel exactly. Therefore the old
zero cannot be treated as an intrinsic 16K capability ceiling for this K128
artifact. Correcting the reference/request contract is sufficient for large
recovery in this protocol. Because reference changes the coordinate and
target/reference changes both dilation and analytic gain, the experiment
does not isolate those components individually.

Updated interpretation:

1. A direct 4x maximum profile remains useful at the tested 1x/2x/4x points;
   the previous one-hop-only explanation is not needed for this artifact.
2. Frozen physical G is not falsified as a useful long backbone here, but its
   privilege over the normalized-index construction is not established.
3. Independent Native reference calibration is an explicit dependency of this
   engineering pipeline. It is not evidence that a config-only
   checkpoint-plus-s rule works universally.
4. K, architecture, model size and checkpoint training are not causally
   separated. P2 now fills the fixed same-generation Qwen s2 baseline gaps;
   no new curve, residual, G(x;K), per-model gain or s8 branch follows.

These are core-four retrieval plus teacher-forced natural continuation
results. They do not substitute for complete RULER-13, natural QA/F1,
arbitrary-scale/continuous-interval guarantees, or a submission SOTA claim.
