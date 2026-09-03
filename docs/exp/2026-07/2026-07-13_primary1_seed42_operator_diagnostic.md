# Primary I Seed-42 Operator Diagnostic — 2026-07-13

## Status and bottom line

This is a **new-lineage, single-seed, undertrained diagnostic**. It does not
replace the historical three-seed Primary I artifact and has not been promoted
to `data/curated/` or the paper.

The 454.2M Geo and EVQ substrate runs completed normally. At 8K, the
YaRN-equation-derived operator produced PPL 66.016 on midpoint-Geo and 64.110
on EVQ (EVQ: -2.89% PPL, `Delta NLL=-0.0293`), while global teacher-forced
passkey retrieval was 89.0% versus 94.5%. This is a real positive direction,
but the effect is modest. The much larger separation belongs to the repository's
custom fixed-ramp scaler, not to official YaRN.

Most importantly, both trained substrates use the midpoint grid
`u=(k+0.5)/K`. The pinned official YaRN equations are therefore applied through
a virtual-dimension extension. The correct label for both evaluated arms is
**YaRN-derived**, not a faithful official-YaRN method reproduction.

## Artifact receipt

The small raw artifacts, logs, and exact server-side runners were copied to the
ignored local snapshot `results/rebuttal_primary1_20260713/`. Model checkpoints
and token tensors were deliberately excluded.

| Artifact | SHA256 |
| --- | --- |
| 454M operator JSON | `f904f80d612a15151bb924de532bd491d27b62964e7203cb3bd94e8590747e28` |
| 454M training summary | `65c4fe2b10d1283857e92f4b950fdd102f9596daa9bc0b741ac7584285260303` |
| 125M training summary | `6269c2b1821b1e333dfb3bde4456036deffc543a20b9db7cebc53b7f32e23f50` |
| Completed operator log | `b39cd119d2ada19c4f663a4185cb930c922dbe64ab09fb996905bde33b5e8b75` |

The copied hashes match the source machine byte-for-byte. The server-side
`official_yarn.py` hash is
`2871aa51977b6eb02f6a270de1bdbe5292632c512da20275901c7f8e537df2da`,
identical to the tracked implementation. The pinned upstream reference is
`jquesnelle/yarn@995db5b575e75230b3384d658f8b944c9662f775`.

## Protocol

| Field | 454M diagnostic |
| --- | --- |
| Seed | 42 |
| Model | 454.2M parameters; 24 layers, hidden 1024, 16 heads, head dim 64 |
| Training data | FineWeb-Edu, 100M tokens, 10% passkey mix |
| Train length | 2,048 |
| RoPE base | 500,000 |
| Substrates | midpoint-Geo `tau=0`; EVQ-Cosh `tau=1.5` |
| Optimization | BF16 AdamW, LR `2e-4`, batch 4, 12,207 steps |
| Runtime | Flash SDPA and `torch.compile(mode="default")` |
| Operator evaluation | fixed scale 8; 2K/4K/8K/12K/16K PPL; 2K–12K PK |
| YaRN parameters | `beta_fast=32`, `beta_slow=1`, `mscale=1.207944`, transition channels 5–15 |

The batch-4 run has half as many optimizer updates as the historical batch-2
recipe. It is explicitly new lineage. No YaRN operator was used during
substrate training.

The native endpoint parity gate passed exactly:

- native-path maximum absolute error: `0.0`;
- generic-inv-frequency-path maximum absolute error: `0.0`;
- virtual native-grid coordinate error: `3.55e-15`;
- expected and obtained `mscale`: `1.2079441541679836`.

This proves formula parity on the native endpoint geometric grid. It does not
turn a midpoint-grid application into official YaRN.

## Training health

Both 454M arms trained to completion without numerical divergence. Their loss
trajectories are nearly indistinguishable.

| Arm | Train time | Mean first 10 logged losses | Mean last 20 | Post-train PPL@2K | PK retrieval, 2K/4K/8K aggregate |
| --- | ---: | ---: | ---: | ---: | ---: |
| midpoint-Geo | 40.1 min | 7.9518 | 3.7831 | 64.774 | 70.67% |
| EVQ-Cosh | 39.7 min | 7.9482 | 3.7917 | 65.575 | 84.67% |

The model is still undertrained: PPL around 65 at the training length is not a
well-trained language-model regime. These runs test mechanism direction, not
production-quality language modeling.

## Final six-cell operator result

All PPL values below come from the same 1M-token validation cache with eight
chunks per length. They are directly comparable within this table.

| Substrate | Operator | 2K | 4K | 8K | 12K | 16K |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| midpoint-Geo | none | 81.277 | 69.423 | 122.415 | 185.462 | 220.554 |
| EVQ-Cosh | none | 81.910 | 61.300 | 101.826 | 150.787 | 182.538 |
| midpoint-Geo | repo fixed-ramp | 82.439 | 51.642 | 66.861 | 106.475 | 142.023 |
| EVQ-Cosh | repo fixed-ramp | 84.782 | 53.759 | 59.601 | 70.612 | 87.690 |
| midpoint-Geo | YaRN-derived | 89.463 | 58.545 | 66.016 | 72.816 | 75.131 |
| EVQ-Cosh | YaRN-derived | 86.937 | 56.835 | 64.110 | 70.728 | 73.149 |

### NLL view

NLL is `ln(PPL)`. Negative `Delta NLL` favors EVQ.

| Operator | Geo NLL@8K | EVQ NLL@8K | Delta NLL@8K | Delta NLL@12K | Delta NLL@16K |
| --- | ---: | ---: | ---: | ---: | ---: |
| none | 4.8074 | 4.6233 | -0.1842 | -0.2070 | -0.1892 |
| repo fixed-ramp | 4.2026 | 4.0877 | -0.1149 | -0.4107 | -0.4822 |
| YaRN-derived | 4.1899 | 4.1606 | -0.0293 | -0.0291 | -0.0267 |

The YaRN-derived operator lowers extrapolation PPL strongly for both
substrates, but also compresses their difference to roughly 2.6–2.9% PPL from
8K through 16K. At 2K, applying fixed scale 8 incurs the expected short-context
cost: +10.1% PPL for Geo and +6.1% for EVQ relative to their raw substrates.

### Teacher-forced passkey retrieval

Each length uses five depths and ten trials per depth. The global value covers
200 trials over 2K, 4K, 8K, and 12K. The 16K length has PPL but no PK cell in
this evaluator.

| Substrate | Operator | 2K | 4K | 8K | 12K | Global |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| midpoint-Geo | none | 100% | 48% | 52% | 56% | 64.0% |
| EVQ-Cosh | none | 100% | 66% | 62% | 54% | 70.5% |
| midpoint-Geo | repo fixed-ramp | 98% | 98% | 56% | 52% | 76.0% |
| EVQ-Cosh | repo fixed-ramp | 100% | 98% | 92% | 72% | 90.5% |
| midpoint-Geo | YaRN-derived | 96% | 92% | 84% | 84% | 89.0% |
| EVQ-Cosh | YaRN-derived | 100% | 100% | 86% | 92% | 94.5% |

Autoregressive exact match is 0% for all six cells on this seed-0 prompt set,
so it provides no discrimination here. The paper-defined PK metric is the
teacher-forced NLL-gap retrieval rate above.

## Matched contrasts

### EVQ versus Geo under the same operator

- Raw substrate at 8K: `Delta NLL=-0.1842`, 16.8% lower PPL, +6.5 percentage
  points global PK.
- Repo fixed-ramp at 8K: `Delta NLL=-0.1149`, 10.9% lower PPL, +14.5 points
  global PK. At 12K/16K the PPL reduction grows to 33.7%/38.3%.
- YaRN-derived at 8K: `Delta NLL=-0.0293`, 2.9% lower PPL, +5.5 points global
  PK. The PPL advantage remains only 2.6–2.9% at 12K/16K.

### Operator versus no scaler on the same substrate

- On midpoint-Geo, YaRN-derived scaling changes PPL by -46.1% at 8K, -60.7%
  at 12K, and -65.9% at 16K; global PK rises by 25 points.
- On EVQ, YaRN-derived scaling changes PPL by -37.0% at 8K, -53.1% at 12K,
  and -59.9% at 16K; global PK rises by 24 points.

This is strong evidence that range scaling helps both trained substrates. It is
only modest evidence for an additional EVQ advantage under the YaRN-derived
operator.

## Validation-slice sensitivity

The post-training summaries used a separate 5M-token validation cache, whereas
the six-cell table uses a 1M-token cache. Both evaluate only eight chunks per
length. Absolute PPL must not be mixed across those files.

The warning is material: in the post-training summary, EVQ is 17.5% better than
Geo at 8K but 3.9% worse at 16K; in the six-cell raw-substrate table it is 16.8%
better at 8K and 17.2% better at 16K. The 16K sign change means that a larger,
fixed validation budget is required before making a 16K raw-substrate claim.

## 125M directional run

The separate 151.9M run used batch 12 and only 4,069 optimizer steps. It is not
a scale-matched reproduction of either the 454M run or the historical protocol.

| Arm | 2K PPL | 4K | 8K | 12K | 16K | PK retrieval, 2K/4K/8K aggregate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| midpoint-Geo | 61.054 | 99.393 | 178.079 | 202.106 | 242.163 | 80.67% |
| EVQ-Cosh | 61.370 | 97.178 | 166.459 | 180.757 | 217.413 | 46.00% |

EVQ improves long-length PPL by 6.5% at 8K and about 10.2–10.6% at 12K/16K,
yet retrieval collapses: Geo reaches 100%/84%/58% at 2K/4K/8K, while EVQ
reaches only 48%/46%/44%. This is direct evidence that lower aggregate language
modeling NLL is not equivalent to retrieval capability. It also prevents any
universal cross-scale EVQ claim from this seed.

## Operational incident

The watchdog logged 24 operator-evaluation launches between 19:27 and 20:21.
Earlier failure logs were overwritten, so their individual causes are not fully
recoverable. The 125M GPU run overlapped from 19:53 to 20:20, and the evaluator
had neither a shared GPU lease nor per-arm resume artifacts. Any failed attempt
therefore restarted the full six-cell sweep.

After the 125M job ended, the 20:21 launch ran continuously and wrote the final
JSON at 20:37. The watchdog then reported `ALL COMPLETE` with zero GPU memory in
use. Only that final atomic JSON is used in this report.

Before another paid rerun, the evaluator should acquire the same GPU lock as
training and save each completed substrate/operator cell atomically so an
interruption only reruns the missing cells.

## Rebuttal-safe interpretation

Safe:

> In a new single-seed, undertrained 454M diagnostic, a pinned
> YaRN-equation-derived scaler substantially improved both midpoint-Geo and EVQ
> substrates. EVQ retained a modest NLL and teacher-forced retrieval advantage;
> the much larger separation was specific to our repository-defined fixed-ramp
> scaler.

Not supported:

- “The official YaRN comparison reproduces the historical 100% versus 61%
  result.”
- “EVQ has a large universal synergy with official YaRN.”
- “The single-seed new lineage replaces the historical Primary I table.”

A true official-YaRN control still requires a native endpoint-Geo substrate.
Any application to EVQ remains a clearly labeled YaRN-derived generalization;
a method-level YaRN comparison would additionally require matched continuation
or fine-tuning rather than only an eval-time operator swap.

## Validation performed in this audit

- Parsed all copied JSON artifacts successfully.
- Verified source and local SHA256 equality for the four key artifacts.
- Confirmed upstream YaRN pin and exact native-grid parity.
- Fresh local tests: `22 passed` across
  `test_official_yarn_parity.py` and
  `test_official_yarn_capability_eval.py`.
- No paper source, PDF, curated metric, or reported experimental number was
  changed.
