# Reference-corrected K128 coupling (2026-09-01)

## Passport and decision

- **Status:** `S2_COMPLETE / S4_CONDITIONAL_STAGE_OPENED`
- **Checkpoint:** the exact Gemma-1.1 redistribution bound by the
  [confirmed Native reference receipt](../evidence/GEMMA_NATIVE_REFERENCE_CONFIRMED_20260901.json).
- **Reference:** `L_config=8192`, independently confirmed operating
  `L_ref=4096`; training history is not relabeled.
- **Law:** unchanged `x_H=.7382780681078285`, `x_L=.366403835112904`,
  `G(x)=clip((x_H-x)/(x_H-x_L),0,1)`, `omega'=omega*s^-G(x)`, `c=.074`.
- **Registration:**
  [`REFERENCE_CORRECTED_K128_PREFLIGHT_20260901`](../preflights/REFERENCE_CORRECTED_K128_PREFLIGHT_20260901.md).
- **RULER owner:**
  [`REFERENCE_CORRECTED_K128_S2_RULER_RECEIPT_20260901.json`](../evidence/REFERENCE_CORRECTED_K128_S2_RULER_RECEIPT_20260901.json).
- **Natural continuation owner:**
  [`REFERENCE_CORRECTED_K128_S2_NLL_RECEIPT_20260901.json`](../evidence/REFERENCE_CORRECTED_K128_S2_NLL_RECEIPT_20260901.json).

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

## 4. Consequence and conditional next stage

The preregistered stage-2 entrance passes: physical single-key is `20/20` at
8K and its macro exceeds Native by `.8600`; the separate 4K RULER and PPL
operating gates pass as well.

The next fixed target is 16384, hence `s=4`. It uses the same confirmed
reference, unchanged `G` and coefficient, and one static maximum profile at
4096/8192/16384. No 16K outcome selects the profile. A paired old
config-reference s2 table is a diagnostic control, not a new candidate.

Only that completed stage can determine whether the former 16K zero was
resolved by the joint reference/request-scale correction. The s2 result alone
does not establish 4x behavior, explain the reference and gain components
separately, identify K causally, or validate arbitrary-scale universality.
