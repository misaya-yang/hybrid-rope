# 151.9M Native-RoPE / EVQ 500M Result Audit

Date: 2026-07-13
Status: complete single-seed diagnostic; not promoted into the paper
Raw artifact: `data/curated/native_rope_evq_150m_s42_500m_20260713.json`

## 1. Scope and machine separation

This experiment belongs only to the RTX 5090 small-model track. It does not
stand in for the RTX Pro 6000 LLaMA-3-8B/LoRA track and cannot answer whether a
small scratch model solves downstream QA. The 5090 questions are narrower:

1. Does endpoint EVQ retain a raw extrapolation advantage after 500M tokens?
2. Does a matched range scaler help EVQ more than native RoPE?
3. Does the historical repository fixed-ramp scaler behave like a useful
   operator even though it is not official YaRN?
4. Does the present result resolve the previously observed training-saturation
   reversal?

## 2. Contract and validity checks

- Model: 151,898,880 parameters, 12 layers, width 768, 12 heads,
  `d_head=64`, tied embeddings.
- Training: length 2,048; 499,998,720 tokens; seed 42; 4,069 optimizer
  steps; global batch 60; AdamW; BF16 autocast.
- Data: FineWeb-Edu plus 10,088,448 deterministic Passkey tokens (2.0177%).
- Pairing: identical initial-trainable hash
  `fb176482...2452`, identical row-order hash `2fee60f7...3953`, and distinct
  registered endpoint frequency hashes.
- Held-out natural-text evaluation comes from FineWeb-Edu shard 004, which is
  absent from the training-shard list.
- Training time was 3,044 s for Native and 2,973 s for EVQ on one RTX 5090.

The first automatic evaluation stopped before model inference because PyTorch
2.8 rejected a `TorchVersion` metadata object under `weights_only=True`. The
checkpoint loader was changed only to allowlist that exact safe metadata type;
future checkpoints serialize the version as a plain string. No model tensor,
frequency operator, data, offset, or metric changed. The repaired evaluator
passed all local and server regression tests before evaluation.

## 3. Natural-text NLL/PPL

Each entry is the mean over the same eight frozen validation offsets. At 4K,
8K, and 16K, raw EVQ has lower NLL at all 8/8 offsets.

| Length | Native raw | EVQ raw | EVQ raw delta | Native + official YaRN | EVQ + YaRN-derived | EVQ scaled delta | Native + fixed-ramp | EVQ + fixed-ramp | EVQ fixed-ramp delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2K | 30.09 | 29.96 | -0.4% | 30.09 | 29.96 | -0.4% | 30.09 | 29.96 | -0.4% |
| 4K | 51.79 | 47.31 | -8.6% | 30.21 | 30.06 | -0.5% | 34.04 | 30.72 | -9.7% |
| 8K | 125.97 | 107.97 | -14.3% | 33.16 | 33.07 | -0.3% | 63.61 | 39.22 | -38.3% |
| 16K | 216.67 | 181.52 | -16.2% | 33.16 | 32.94 | -0.7% | 97.36 | 53.99 | -44.5% |

The official native-grid YaRN implementation is pinned to
`jquesnelle/yarn@995db5b`. Applying the same equations to endpoint EVQ uses a
virtual frequency coordinate and must be called **YaRN-derived on EVQ**, not
official native-grid YaRN.

The final-2K-token NLL check gives the same qualitative result and rules out a
simple easy-prefix dilution explanation:

| Length | Native raw / EVQ raw | Native official / EVQ derived | Native fixed-ramp / EVQ fixed-ramp |
| ---: | ---: | ---: | ---: |
| 8K | 5.840 / 5.590 | 3.465 / 3.458 | 5.303 / 4.046 |
| 16K | 6.275 / 5.977 | 3.611 / 3.590 | 5.741 / 5.248 |

## 4. Difference-in-differences: what is actually super-additive?

For a lower-is-better NLL endpoint, define the interaction

```text
I_op = [NLL(EVQ+op) - NLL(EVQ raw)]
     - [NLL(Native+op) - NLL(Native raw)].
```

`I_op < 0` means that the operator improves EVQ more than Native.

| Length | Official/derived YaRN interaction | Repository fixed-ramp interaction |
| ---: | ---: | ---: |
| 4K | +0.085 | -0.012 |
| 8K | +0.151 | **-0.329** |
| 16K | +0.170 | **-0.413** |

Therefore:

- This run **does not establish super-additive interaction with official
  YaRN** at the target-matched scales. Official YaRN is so effective here that
  it brings both substrates to almost the same NLL; it helps Native more in
  absolute NLL because Native starts worse.
- The historical **repository fixed-ramp does show a strong positive
  complementarity with EVQ** at 8K and 16K. It is also useful by itself: at 16K
  it reduces Native PPL from 216.67 to 97.36 and EVQ PPL from 181.52 to 53.99.
- The fixed-ramp result makes it a legitimate independent range-scaler
  candidate. It is not YaRN, and novelty relative to other by-parts/ramp
  scalers has not been established.

## 5. Zero-shot operator test versus the full YaRN training recipe

This six-cell evaluation injects each operator only at inference into models
trained with their raw 2K substrates. That is a valid **non-fine-tuned YaRN
operator** diagnostic, which the YaRN paper discusses, but it is not a
reproduction of the paper's headline context-extension recipe. The published
64K model used 400 long-context fine-tuning steps, and its 128K model continued
for another 200 steps.

The two scalers also share only a broad endpoint intuition; their actual
operators differ materially. At `d_head=64`, base 500K, and scale 8:

- official YaRN keeps channels 0--5 unchanged, linearly transitions through
  channels 6--14, fully divides channels 15--31 by 8, and multiplies rotary
  sine/cosine amplitudes by `mscale=1.208` (about `1.459x` on QK logits);
- repo fixed-ramp uses a much broader smoothstep over channels 6--28, reaches
  about an 8.8-fold low-frequency divisor because its temperature term is
  folded into frequency, and applies no attention mscale;
- fixed-ramp applies the same index-wise divisor to Native and EVQ, whereas the
  YaRN-derived EVQ arm obtains its ramp from a virtual frequency coordinate.
  That extension is reasonable and matches official YaRN on a native grid, but
  it is not a canonical operator published by the YaRN authors.

Consequently, "high frequencies unchanged, low frequencies stretched" does
not imply the same interaction. Official YaRN is both more aggressive over the
middle/low band and includes a second attention-temperature mechanism. Here it
nearly saturates natural-text PPL for both substrates, leaving little substrate
gap. The present table cannot determine whether that equalization comes mainly
from the correction range or from `mscale`; a checkpoint-only phase-versus-
mscale ablation would answer that before any new training.

## 6. Passkey NLL-gap

Passkey is explicitly supervised at 2K. The primary value below is the
continuous teacher-forced `NLL_wrong - NLL_correct` gap over 100 cases; the
sign rate is diagnostic.

| Operator | Native gap / sign rate | EVQ gap / sign rate | EVQ minus Native gap |
| --- | ---: | ---: | ---: |
| Raw | 0.820 / 67% | 0.830 / 69% | +0.010 |
| Official / derived YaRN | 3.122 / 100% | 3.189 / 100% | +0.067 |
| Repository fixed-ramp | 0.975 / 77% | 1.301 / 87% | **+0.325** |

Official YaRN closes the retrieval task for both arms, so the 100% sign rate
cannot distinguish their substrates. The fixed-ramp remains below saturation
and shows a larger EVQ advantage. This is consistent with complementarity but
is still one seed and one synthetic task. Evaluation secrets have zero overlap
with training secrets, but the task format itself is supervised.

## 7. Does 500M settle the training-saturation reversal?

No. The observed answer at 500M is clear but narrower:

- There is **no reversal yet** in this MHA run: raw EVQ is 14.3%/16.2% lower
  PPL at 8K/16K.
- 500M tokens are only about 3.29 tokens per parameter. This is substantially
  more training than the historical 100M pilot but should not be called a
  saturated 151.9M language model.
- The historical 1B reversal came from a different MLA architecture, length,
  corpus, and protocol. It cannot be transferred to this MHA pair.

The clean next test is a **from-scratch 1B matched pair**, with one LR schedule
registered for the full 1B run and checkpoints at 500M, 750M, and 1B. The
current 500M checkpoints do not contain optimizer state and their cosine
schedule already ended at 500M, so restarting them is only an exploratory
continuation, not a clean saturation trajectory.

## 8. Bottom line

This 500M run supports two narrow findings:

1. Endpoint EVQ still improves raw long-range NLL/PPL over native RoPE at this
   training budget; the raw reversal has not occurred here.
2. The historical fixed-ramp is a useful, distinct range scaler and has a
   strong positive interaction with EVQ. At the tested target-matched
   **zero-shot** scales, official/derived YaRN is better in absolute PPL but
   does not show the claimed super-additive EVQ interaction because both
   substrates converge to nearly the same NLL. This does not resolve the
   separate question of matched YaRN long-context continuation training.

This is single-seed mechanistic evidence. It neither establishes downstream
capability nor settles the fully trained regime. No further data preparation or
GPU experiment was started.
