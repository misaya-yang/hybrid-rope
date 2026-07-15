# MLA K=16 short-context YaRN component ablation (seed 42)

## Status and claim boundary

This is a single-seed, supporting/mechanistic pilot. It does not modify or
promote any primary paper claim. The experiment asks whether official YaRN's
frequency correction can erase the EVQ substrate advantage in a scarce rotary
channel regime (`d_rope=32`, `K=16`).

The decisive prediction was preregistered in the evaluator before the results
were available: at 4K and 8K, `freq_only` should leave an absolute
Geo-minus-EVQ NLL gap greater than `0.02`. The kill condition was that
`freq_only` flattened this gap at both decisive lengths.

## Protocol

- Architecture: 432,194,560-parameter MLA; code field `head_dim=64`,
  `d_rope=32`, `d_nope=32`, `K=16`, `kv_lora_rank=256`.
- Training: seed 42, sequence length 512, 99,999,744 tokens, FineWeb-Edu,
  2% explicitly supervised Passkey mix, base 500K.
- Substrates: native endpoint geometric RoPE and endpoint EVQ-Cosh with
  `tau=1.414`.
- `tau=1.414` is retained as the historical Primary III empirical
  `d_eff=128` operating convention. It is not re-derived from the shortened
  512-token training length, `head_dim`, or `d_rope`.
- Evaluation lengths: 1K, 2K, 4K, and 8K, corresponding to YaRN scales 2, 4,
  8, and 16 relative to the 512-token training length.
- Operators: `raw`, `freq_only`, `mscale_only`, and `full`.
- Native uses official native-grid YaRN equations. EVQ uses the same equations
  through the virtual-coordinate derived-YaRN path and is labeled
  "YaRN-derived on endpoint EVQ-Cosh."
- Natural text: mean NLL/PPL over eight frozen offsets per length.
- Passkey: 100 held-out teacher-forced cases per condition, split into 25 cases
  at each length; the metric is `NLL_wrong - NLL_correct` and is diagnostic,
  not an autoregressive capability claim.

Matched controls passed before scoring: both arms have identical initial
trainable-weight, row-order, Passkey-selector, training-data, validation-data,
and model-config hashes; only their frequency hashes differ.

## Natural-text results

Absolute PPL (lower is better):

| Operator | Substrate | 1K | 2K | 4K | 8K |
|---|---|---:|---:|---:|---:|
| raw | Native | 102.926 | 212.282 | 315.422 | 470.719 |
| raw | EVQ | **82.058** | **177.258** | **272.040** | **412.034** |
| freq_only | Native | **53.408** | **57.775** | 65.837 | 117.936 |
| freq_only | EVQ | 54.451 | 57.830 | **62.673** | **92.871** |
| mscale_only | Native | 105.403 | 234.148 | 384.240 | 618.662 |
| mscale_only | EVQ | **83.047** | **191.888** | **317.727** | **519.139** |
| full | Native | **52.714** | **55.068** | 56.863 | 85.455 |
| full | EVQ | 53.837 | 55.187 | **55.410** | **71.550** |

Substrate gap `NLL(Native) - NLL(EVQ)` (positive favors EVQ):

| Operator | 1K | 2K | 4K | 8K |
|---|---:|---:|---:|---:|
| raw | +0.2266 | +0.1803 | +0.1480 | +0.1332 |
| freq_only | -0.0193 | -0.0009 | **+0.0492** | **+0.2389** |
| mscale_only | +0.2384 | +0.1990 | +0.1901 | +0.1754 |
| full | -0.0211 | -0.0022 | +0.0259 | +0.1776 |

Difference-in-differences
`I_op = (EVQ_op - EVQ_raw) - (Native_op - Native_raw)`:

| Operator | 1K | 2K | 4K | 8K |
|---|---:|---:|---:|---:|
| freq_only | +0.2459 | +0.1813 | +0.0987 | -0.1058 |
| mscale_only | -0.0118 | -0.0187 | -0.0421 | -0.0422 |
| full | +0.2477 | +0.1825 | +0.1221 | -0.0444 |

## Passkey diagnostic

Mean teacher-forced NLL gap (positive favors the correct key):

| Operator | Substrate | 1K | 2K | 4K | 8K |
|---|---|---:|---:|---:|---:|
| raw | Native | +0.1396 | -0.0864 | -0.1104 | -0.1078 |
| raw | EVQ | -0.0293 | -0.1011 | +0.0229 | -0.1468 |
| freq_only | Native | +2.1140 | +1.4337 | +0.3206 | -0.0517 |
| freq_only | EVQ | +1.9663 | +1.7797 | +0.7022 | -0.0226 |
| mscale_only | Native | +0.1217 | +0.0026 | -0.0590 | -0.1418 |
| mscale_only | EVQ | -0.0424 | -0.1939 | -0.0248 | +0.0151 |
| full | Native | +2.2783 | +2.3696 | +1.0810 | +0.0963 |
| full | EVQ | +1.8906 | +1.6779 | +1.0990 | +0.1226 |

At 8K all gaps are close to zero, so this run does not support an 8K Passkey
capability claim. The Passkey rows remain mechanism diagnostics only.

## Preregistered decisions

1. **P1 passed at the decisive lengths.** `freq_only` flattened or slightly
   reversed the substrate gap at 1K/2K, but the gap reopened to `+0.0492` at
   4K and `+0.2389` at 8K. Thus frequency correction does not fully substitute
   for EVQ in the long-extrapolation, K=16 regime.
2. **P2 failed.** EVQ `mscale_only` was not approximately as good as EVQ
   `full`; at 8K their PPLs were 519.139 and 71.550. The mscale component is
   useful after frequency correction, not as a standalone substitute for it.
3. **P3 failed.** EVQ `raw` beat Native `raw` at every length, but it did not
   beat Native `full`. At 8K the relevant PPLs were 412.034 and 85.455.
4. **The kill condition did not trigger.** `freq_only` did not flatten the
   substrate gap at both 4K and 8K.

## Mechanistic interpretation

The result is narrower than “YaRN cannot replace EVQ.” At modest extrapolation
(1K/2K), frequency correction does erase the substrate gap. At 4K/8K, where
only 16 rotary frequency channels are available, a measurable EVQ advantage
remains after the same correction. This is consistent with the proposed
scarce-channel stress mechanism.

The best absolute 8K result is EVQ + full derived-YaRN (PPL 71.550), compared
with Native + full official YaRN (85.455). Frequency correction supplies the
large range-extension gain, while mscale improves both substrates only when
combined with that correction. The poor `mscale_only` rows and strong `full`
rows show an interaction rather than two independently sufficient mechanisms.

These observations are single-seed, short-training supporting evidence. They
should not replace the three-seed Primary III result or be promoted into a
universal YaRN/EVQ claim.

## Artifacts and verification

Raw outputs are kept under the ignored narrow directory
`results/mla_yarn_short_s42_20260714/`.

- `raw_results.json`: `c4c3ed3e3d9a9814b1320c37f8f099926e7a636a8007f2a0651a2b96c2e562c5`
- `analysis.json`: `ce46819ad40bffe2b5b68f2b3c0f2deb1548089a84ad5ca98ce0378ddcbeaee2`
- sanitized manifest: `132a4ae32db3e8b3c4ff03f28da918229a1cf17537ae8b2f30a66f13c83348a3`
- Native checkpoint: `d475c1f2032792bf8a3fac1b9a1750d6bf08a1beb487b59e928218e91ea369ec`
- EVQ checkpoint: `d1a48930ff541d60e928d4b5d5c114dbe2593fe1d1147607e9277379abbd1530`

Both checkpoints remain on the experiment machine. Only result JSON, logs,
frequency tensors, and checkpoint metadata were copied into the local ignored
results directory.

Preservation boundary: the original 2K token tensor and all pre-existing
checkpoints were never opened for write. Training used a read-only 512-token
view and wrote new checkpoints to separate directories. The fresh seed-42
initial weights are identified by SHA256 but were not separately saved as an
initial-weight checkpoint.
