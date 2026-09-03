# Repository Fixed-Ramp Mechanism Probe

Date: 2026-07-14  
Status: complete checkpoint-only, inference-only seed-42 diagnostic  
Scope: 151.9M RTX-5090 small-model track; supporting/mechanistic evidence only

## 1. Question and controls

The existing six-cell result showed that the repository fixed-ramp scaler is
useful and interacts strongly with endpoint EVQ-Cosh, but it did not identify
which implementation choice causes that complementarity. This probe reuses the
same two 500M-token checkpoints, eight frozen natural-text offsets, and 100
Passkey cases. It evaluates:

- `repo_scale_only`: the broad 20%--90% smoothstep divisor `scale^r`, with the
  folded temperature coefficient set to zero;
- `repo_full`: the registered scaler `scale^r * temperature^(0.5r)`;
- `official_linear_shared_index`: official linear-ramp weights applied as the
  same index-wise frequency multiplier to both substrates, without mscale;
- repo frequency variants combined with half or full official YaRN mscale.

The shared-index control is descriptive and must not be called official YaRN
on EVQ. No training or paper-table update was performed.

## 2. Reproduction gate

New `raw` and `repo_full` outputs reproduce the prior complete artifact exactly
for both substrates: maximum per-offset natural-text NLL difference `0.0` and
maximum per-case Passkey NLL-gap difference `0.0`.

## 3. Frequency-ramp ablation

The substrate gap is `NLL(Native) - NLL(EVQ)`; positive values favor EVQ.
Negative interaction means the operator improves EVQ more than Native.

| Length | Operator | Native PPL | EVQ PPL | Substrate gap | Interaction |
| ---: | --- | ---: | ---: | ---: | ---: |
| 4K | repo_scale_only | 34.53 | 30.92 | +0.1106 | -0.0201 |
| 4K | repo_full | 34.04 | 30.72 | +0.1024 | -0.0120 |
| 4K | official linear shared-index | 30.64 | 30.89 | -0.0084 | +0.0989 |
| 8K | repo_scale_only | 65.34 | 40.92 | +0.4680 | -0.3138 |
| 8K | repo_full | 63.61 | 39.22 | +0.4835 | -0.3293 |
| 8K | official linear shared-index | 36.03 | 37.73 | -0.0460 | +0.2002 |
| 16K | repo_scale_only | 101.53 | 57.50 | +0.5686 | -0.3917 |
| 16K | repo_full | 97.36 | 53.99 | +0.5897 | -0.4127 |
| 16K | official linear shared-index | 42.58 | 49.05 | -0.1415 | +0.3185 |

The broad `scale^r` smoothstep already accounts for nearly all of the repository
scaler's behavior. The folded temperature term gives a modest additional gain,
especially at 16K, but is not the source of complementarity. Applying an
identical index multiplier is also insufficient: the narrow official linear
ramp rescues Native more strongly and reverses the substrate ordering.

Thus the useful interaction is specific to the repository scaler's broad,
gradual frequency deformation. It extends the EVQ-trained spectral allocation
without immediately driving both substrates into the same operator-saturated
regime. The result does not establish that this ramp is novel or generally
better than other by-parts scalers.

## 4. Passkey check

| Operator | Native mean gap / sign rate | EVQ mean gap / sign rate |
| --- | ---: | ---: |
| raw | 0.820 / 67% | 0.830 / 69% |
| repo_scale_only | 0.951 / 75% | 1.209 / 85% |
| repo_full | 0.975 / 77% | 1.301 / 87% |
| official linear shared-index | 2.296 / 99% | 2.491 / 100% |

Passkey follows the same pattern: the repo ramp improves EVQ more while staying
below saturation; the aggressive official-linear control nearly saturates both.

## 5. Combining repo frequencies with YaRN mscale

| Length | Operator | Native PPL | EVQ PPL | Substrate gap | Interaction |
| ---: | --- | ---: | ---: | ---: | ---: |
| 4K | repo_full | 34.04 | 30.72 | +0.1024 | -0.0120 |
| 4K | repo_full + half mscale | 33.45 | **30.35** | +0.0972 | -0.0068 |
| 4K | repo_full + official mscale | **33.39** | 30.38 | +0.0946 | -0.0042 |
| 8K | repo_full | 63.61 | 39.22 | +0.4835 | -0.3293 |
| 8K | repo_full + half mscale | **59.51** | 35.96 | +0.5038 | -0.3497 |
| 8K | repo_full + official mscale | 59.71 | **35.64** | +0.5161 | -0.3619 |
| 16K | repo_full | 97.36 | 53.99 | +0.5897 | -0.4127 |
| 16K | repo_full + half mscale | **86.03** | 45.25 | +0.6425 | -0.4655 |
| 16K | repo_full + official mscale | 89.02 | **44.93** | +0.6837 | -0.5067 |

Adding mscale materially improves the repository ramp and strengthens its EVQ
interaction. It does not produce the lowest absolute PPL: full official/derived
YaRN remains better at 4K/8K/16K (`30.21/33.16/33.16` Native and
`30.06/33.07/32.94` EVQ). The operating choice is therefore objective-specific:

- lowest absolute PPL: full official/derived YaRN;
- strongest visible EVQ complementarity: repo full plus official mscale at
  8K/16K;
- best Native PPL within the repo-frequency family: half mscale at 8K/16K;
- best EVQ PPL within the repo-frequency family: official mscale at 8K/16K.

Further tuning on these same eight offsets would be single-seed overfitting, so
no larger coefficient or blend sweep was run.

## 6. Artifacts

- Ramp probe raw SHA256: `f97a5759a996929b58d0188da30720d07cf3632da653de326c313daa0214be5d`
- mscale probe raw SHA256: `62dc5f0e665a1055c83c2e285c3ea4a6b32f8e675153d5b8b53ddfab965df7bc`
- Sanitized manifest SHA256: `b0154fab5809e380a26a71d9be1a56a4040525afeaa53c58c95f41b7883c2bb8`
