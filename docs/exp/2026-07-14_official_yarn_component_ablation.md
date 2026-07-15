# Official YaRN Frequency-versus-mscale Ablation

Date: 2026-07-14  
Status: complete checkpoint-only, inference-only seed-42 diagnostic  
Scope: 151.9M RTX-5090 small-model track; supporting/mechanistic evidence only

## 1. Question and protocol

This experiment asks why full official/derived YaRN nearly removes the raw
NLL gap between Native RoPE and endpoint EVQ-Cosh (`tau=1.5`). It reuses the
two existing 500M-token seed-42 checkpoints and decomposes YaRN into:

- `raw`: original substrate frequency and attention scaling `1.0`;
- `freq_only`: official YaRN frequency transform, attention scaling `1.0`;
- `mscale_only`: original substrate frequency, official YaRN mscale;
- `full`: official YaRN frequency transform and official YaRN mscale.

Native uses the pinned official native-grid equations. EVQ uses the same
equations through the existing virtual-coordinate path and is labeled
**YaRN-derived on EVQ**. No training, continuation, adapter, data preparation,
or paper-table update was performed.

Natural-text evaluation uses the same eight frozen offsets at 4K, 8K, and 16K.
Passkey retains the exact existing 100 cases across 2K/4K/8K/16K so that its
case identity remains byte-comparable; the 2K identity condition is not part
of the 24-cell long-context attribution matrix.

## 2. Full-path reproduction gate

Before interpreting the ablation, the new `raw` and `full` paths were compared
against the complete prior six-cell artifact for both substrates:

| Gate | Maximum absolute difference |
| --- | ---: |
| Raw natural-text per-offset NLL | 0.0 |
| Full natural-text per-offset NLL | 0.0 |
| Raw 100-case Passkey NLL-gap | 0.0 |
| Full 100-case Passkey NLL-gap | 0.0 |

All 100 Passkey case identities also match exactly. Thus `full` reproduces the
existing official/derived YaRN cells rather than defining a new harness.

## 3. Natural-text attribution

The substrate gap is `NLL(Native) - NLL(EVQ)`, so a positive value favors EVQ.
The interaction is

```text
I_op = [NLL(EVQ+op) - NLL(EVQ raw)]
     - [NLL(Native+op) - NLL(Native raw)].
```

| Length | Operator | Native NLL / PPL | EVQ NLL / PPL | Substrate gap | `I_op` |
| ---: | --- | ---: | ---: | ---: | ---: |
| 4K | raw | 3.9472 / 51.79 | 3.8568 / 47.31 | +0.0904 | +0.0000 |
| 4K | freq_only | 3.4223 / 30.64 | 3.4168 / 30.47 | +0.0055 | +0.0849 |
| 4K | mscale_only | 3.9244 / 50.62 | 3.8336 / 46.23 | +0.0908 | -0.0003 |
| 4K | full | 3.4083 / 30.21 | 3.4033 / 30.06 | +0.0050 | +0.0854 |
| 8K | raw | 4.8360 / 125.97 | 4.6819 / 107.97 | +0.1542 | +0.0000 |
| 8K | freq_only | 3.5842 / 36.02 | 3.5719 / 35.58 | +0.0124 | +0.1418 |
| 8K | mscale_only | 4.8127 / 123.07 | 4.6357 / 103.10 | +0.1770 | -0.0228 |
| 8K | full | 3.5013 / 33.16 | 3.4987 / 33.07 | +0.0027 | +0.1515 |
| 16K | raw | 5.3784 / 216.67 | 5.2014 / 181.52 | +0.1770 | +0.0000 |
| 16K | freq_only | 3.7513 / 42.58 | 3.7184 / 41.20 | +0.0330 | +0.1440 |
| 16K | mscale_only | 5.3782 / 216.64 | 5.1726 / 176.37 | +0.2057 | -0.0287 |
| 16K | full | 3.5014 / 33.16 | 3.4948 / 32.94 | +0.0066 | +0.1704 |

`freq_only` removes about 94%, 92%, and 81% of the raw substrate gap at
4K, 8K, and 16K. In contrast, `mscale_only` preserves or slightly enlarges
the gap. The frequency correction is therefore the mechanism that primarily
equalizes Native and EVQ in this checkpoint-only setting. mscale further
improves absolute NLL after frequency correction, especially at 8K/16K, but
does not erase the substrate advantage by itself.

## 4. Passkey attribution

Passkey is the existing teacher-forced `NLL_wrong - NLL_correct` diagnostic.
It was explicitly supervised at 2K and is already near saturation under the
frequency-transformed operators.

| Operator | Native mean gap / sign rate | EVQ mean gap / sign rate |
| --- | ---: | ---: |
| raw | 0.820 / 67% | 0.830 / 69% |
| freq_only | 2.294 / 99% | 2.494 / 100% |
| mscale_only | 0.820 / 68% | 0.831 / 69% |
| full | 3.122 / 100% | 3.189 / 100% |

The retrieval diagnostic agrees with natural text: frequency correction causes
the large gain and near-saturation, mscale alone is nearly neutral, and full
YaRN raises the continuous gap further after the frequency correction.

## 5. Rebuttal-safe conclusion

For these two raw-trained 151.9M seed-42 checkpoints, official YaRN suppresses
the observable EVQ-versus-Native substrate gap mainly through its wavelength-
driven frequency correction/interpolation. That correction rescues the weaker
Native long-range phase behavior enough to approach the EVQ arm. The mscale
attention-temperature term is useful in combination but is not the equalizing
mechanism on its own.

This is a single-seed, checkpoint-only mechanism diagnostic. It does not show
that EVQ replaces YaRN, does not test the full YaRN continuation-training
recipe, and does not alter any primary claim or number in `paper/`.

## 6. Artifacts

- Raw result SHA256: `9a50505196f76d364da2033b2c05bb772b903c0bdf06ae0c19bc745b77d7ed3e`
- Analysis SHA256: `d1afdd31cd921f9cb6cfcd823195b035f081ba6394b44b075e206b8e037f75bf`
- Sanitized manifest SHA256: `4ace64fef5655b8d47ab013ae5ac83f1987f10b152a791a1fddbb7a991b89bdb`
- Reference six-cell artifact SHA256: `82438325d444ef6ea61520bb411fe60dd75d436655c74e9fc3183aefca6c95ad`
- Native checkpoint SHA256: `91b8a7629c8c1255fb4c8f246dae0b28abeebde74e451aa733422375f58ec8c6`
- EVQ checkpoint SHA256: `25349632db990e8465c9ee3e58d4ec5a460c6c1a5b84dee1c155be6e2ffd078c`
