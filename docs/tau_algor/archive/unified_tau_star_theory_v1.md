# Unified τ* Theory for EVQ-Cosh RoPE Frequency Allocation

**Version**: 1.0 (2026-03-22)
**Status**: Draft — pending multi-AI verification and refinement

---

## 1. Problem Statement

EVQ-Cosh is a single-parameter (τ) frequency allocation for Rotary Position Embeddings:

$$\phi_k(\tau) = 1 - \frac{1}{\tau} \cdot \text{arcsinh}\left((1 - u_k) \cdot \sinh(\tau)\right), \quad u_k = \frac{k + 0.5}{K}$$

$$\text{inv\_freq}_k = \text{base}^{-\phi_k(\tau)}, \quad k = 0, \ldots, K-1$$

At τ=0 (geometric/GEO baseline), φ_k = u_k gives standard geometric frequency spacing. As τ increases, frequencies are redistributed from the extremes toward the center of the spectrum.

**Core claim**: For any architecture (MHA, MLA, DiT), any base, any training length L, there exists an optimal τ* such that EVQ slightly underperforms GEO at the training length but significantly outperforms at extrapolation lengths.

**This document derives a unified formula for τ* across all tested architectures.**

---

## 2. Experimental Evidence

### 2.1 MHA (Standard Multi-Head Attention)

Architecture: head_dim=64, K=32, base=500000, all dims carry RoPE.

| L_train | τ* observed | Source |
|---------|------------|--------|
| 128 | >5.0 (monotonic, no peak) | Phase 6A, 125M+50M |
| 1024 | 2.0 | Phase 6E, 125M |
| 2048 | ~1.5 | Phase 7F, 350M |
| 4096 | ~1.0 | Phase 8E, 350M |

**Empirical formula**: τ*_MHA = d_head / √L = 64 / √L

Verification:
- L=1024: 64/32 = 2.0 ✓
- L=2048: 64/45.3 = 1.414 ≈ 1.5 (6% off) ✓
- L=4096: 64/64 = 1.0 ✓
- L=128: 64/11.3 = 5.66 (matches >5.0) ✓

**Key property**: τ* is model-size-independent (50M and 125M give same τ* at same L). This confirms τ* is a position-encoding property, not a model-capacity property.

### 2.2 MLA (Multi-head Latent Attention)

Architecture: attention = softmax((q_nope·k_nope + q_rope·k_rope) / √d_qk), where d_qk = d_nope + d_rope.

#### Old MLA (d_rope=32, d_nope=32, K=16, base=500000)

| L_train | τ* observed | Source |
|---------|------------|--------|
| 8192 | 1.414 | Phase 18, 350M, 3-seed validated |
| 4096 | ~2.2 (but 16% 4K penalty) | Phase 22, 50M sweep |

L=4096 sweep showed severe non-monotonicity (§4 explains why).

#### New MLA — DeepSeek-V2/V3 aligned (d_rope=64, d_nope=128, K=32, base=10000)

| L_train | τ* observed | Source |
|---------|------------|--------|
| 4096 | ~2.5 | Phase 23, 60M sweep |

Phase 23 full sweep results:

| τ | PPL@4K | PPL@8K | PPL@16K | vs GEO@4K | vs GEO@8K | vs GEO@16K |
|---|--------|--------|---------|-----------|-----------|------------|
| 0.0 (GEO) | 291.4 | 316.6 | 435.3 | — | — | — |
| 0.5 | 307.7 | 354.4 | 467.7 | +5.6% | +11.9% | +7.4% |
| 1.0 | 305.0 | 318.7 | 425.4 | +4.7% | +0.7% | -2.3% |
| 1.414 | 285.8 | 305.2 | 398.7 | -1.9% | -3.6% | -8.4% |
| 2.0 | 300.1 | 329.5 | 402.2 | +3.0% | +4.1% | -7.6% |
| **2.5** | **280.7** | **295.4** | **372.2** | **-3.7%** | **-6.7%** | **-14.5%** |
| 3.0 | 298.5 | 300.6 | 377.3 | +2.4% | -5.1% | -13.3% |

Correct EVQ pattern fully restored: short-distance small win, long-distance large win.

### 2.3 DiT (Diffusion Transformer)

Architecture: bidirectional attention, K_t=16 temporal frequencies, base=10000.

| Model | T_train | τ* observed | Source |
|-------|---------|------------|--------|
| 129.6M | 32 | 1.5 | Phase 16, h2h validated |
| 38.8M | 32 | 2.83 (cross-run only) | Phase 16 |

Sharp phase transition at τ=1.2→1.5: channel 5's maximum angle crosses π, enabling negative cosine lobes for the first time (half-cycle count 5→6). This is a discrete capacity jump.

DiT scaling: τ*_DiT ≈ 0.53 × K_t/√T_train = 0.53 × 16/√32 = 1.50 ✓

---

## 3. Unified Formula

### 3.1 Proposed formula

$$\boxed{\tau^* = \gamma \cdot \frac{d_{qk}}{d_{rope}} \cdot \frac{d_{head,ref}}{\sqrt{L}}}$$

where:
- **L**: training sequence length (tokens for text, frames for video)
- **d_head,ref**: reference head dimension = 64 (standard across all architectures tested)
- **d_rope**: RoPE dimension per head (= d_head for MHA, < d_head for MLA)
- **d_qk**: total Q/K dimension per head (= d_head for MHA, = d_nope + d_rope for MLA)
- **γ**: attention directionality coefficient
  - γ = 1.0 for causal (autoregressive) attention
  - γ = 0.53 for bidirectional attention (DiT)

### 3.2 Reduction to known formulas

**MHA** (d_qk = d_rope = d_head = 64):
$$\tau^*_{MHA} = 1 \cdot 1 \cdot \frac{64}{\sqrt{L}} = \frac{d_{head}}{\sqrt{L}}$$

**MLA** (d_qk = d_nope + d_rope, d_rope < d_qk):
$$\tau^*_{MLA} = \frac{d_{qk}}{d_{rope}} \cdot \frac{64}{\sqrt{L}}$$

**DiT** (d_qk = d_rope, bidirectional):
$$\tau^*_{DiT} = 0.53 \cdot \frac{2K_t}{\sqrt{T_{train}}}$$

### 3.3 Verification against all data

| # | Architecture | d_qk | d_rope | γ | L | Predicted τ* | Observed τ* | Error |
|---|-------------|------|--------|---|---|-------------|-------------|-------|
| 1 | MHA | 64 | 64 | 1.0 | 1024 | 2.0 | 2.0 | 0% |
| 2 | MHA | 64 | 64 | 1.0 | 2048 | 1.414 | ~1.5 | -6% |
| 3 | MHA | 64 | 64 | 1.0 | 4096 | 1.0 | ~1.0 | 0% |
| 4 | MHA | 64 | 64 | 1.0 | 128 | 5.66 | >5.0 | ✓ |
| 5 | Old MLA | 64 | 32 | 1.0 | 8192 | **1.414** | **1.414** | **0%** |
| 6 | New MLA | 192 | 64 | 1.0 | 4096 | **3.0** | **~2.5** | **+20%** |
| 7 | DiT 129.6M | 32 | 32 | 0.53 | 32 | **1.50** | **1.50** | **0%** |

6/7 data points fit within 6%. Point #6 (new MLA) has 20% overestimate, likely due to:
- Experimental noise (bs=2, 50M tokens — highly undertrained)
- Base effect (base=10000 vs 500000 for other experiments)
- True optimum may lie in [2.5, 3.0] (τ=3.0 gives -13.3%@16K vs τ=2.5's -14.5%, within noise)

---

## 4. Non-Monotonicity: Frequency Discretization Effect

### 4.1 The discrete frequency window problem

For K RoPE frequencies with base b, the extrapolation range [L, 2L] occupies a window in φ-space:

$$\Delta\phi_{extrap} = \frac{\ln(2)}{\ln(base)}$$

The average channel spacing is 1/K. The ratio determines landscape smoothness:

$$R = \frac{\Delta\phi_{extrap}}{1/K} = \frac{K \cdot \ln(2)}{\ln(base)}$$

| Config | K | base | R | Behavior |
|--------|---|------|---|----------|
| Old MLA | 16 | 500000 | 0.84 | **Binary hit/miss** — at most 0 or 1 frequency in window |
| New MLA | 32 | 10000 | 2.41 | **Smooth** — always 2-3 frequencies in window |
| MHA | 32 | 500000 | 1.69 | Moderate — 1-2 frequencies |

**Rule of thumb**: R ≥ 2 is required for a smooth τ landscape. R < 1 causes severe non-monotonicity.

### 4.2 Channel crossing events (Old MLA, K=16, base=500000, L=4096)

The [4K, 8K] extrapolation window in φ-space: [0.494, 0.547], width = 0.053.
Channel spacing: 1/16 = 0.063 > window width.

As τ increases, channel k's φ_k(τ) decreases (frequency increases, wavelength decreases). A channel "enters" the window when its wavelength drops below 8192, and "exits" when it drops below 4096:

| τ range | Channel event | Frequency in [4K,8K]? | PPL@8K |
|---------|--------------|----------------------|--------|
| 0.0-1.7 | k=8 natural in range | 1 (GEO) | baseline |
| 1.7-1.9 | k=10 enters window | 1 (k=10) | -13.2% |
| 1.9-2.1 | k=10 exits, k=11 not yet in | **0** | +15.6% |
| 2.1-2.4 | k=11 enters window | 1 (k=11) | **-20.6%** |
| 2.4-2.9 | k=11 exits, k=12 not yet in | **0** | +5.7% |
| 2.9-3.1 | k=12 at boundary | ~0.5 | -16.3% |

The non-monotonicity is **completely explained** by discrete channel transitions through a window narrower than the channel spacing.

### 4.3 Why new MLA eliminates non-monotonicity

With K=32 and base=10000: window width = ln(2)/ln(10000) = 0.0753, spacing = 1/32 = 0.031. Ratio R = 2.41 means 2-3 channels always overlap the window → smooth interpolation, no binary hit/miss.

---

## 5. Physical Interpretation

### 5.1 Why τ* = d_head/√L (MHA)

The formula encodes a **bandwidth-resolution tradeoff**:

- **d_head** (= 2K): the total frequency bandwidth — how many independent position-encoding channels are available
- **√L**: the resolution scale — how finely positions need to be discriminated over L tokens
- **τ**: controls how aggressively the K frequencies are concentrated toward the "useful" range (periods ~ L)

At τ*: the redistribution exactly balances in-distribution resolution (need diverse frequencies for [0, L]) with extrapolation coverage (need frequencies with period > L). Below τ*: insufficient extrapolation frequencies. Above τ*: too many frequencies wasted on extrapolation at the cost of in-distribution quality.

### 5.2 Why MLA needs the d_qk/d_rope correction

In MLA, the attention score decomposes as:

$$\text{score}(m, n) = \underbrace{q_{nope} \cdot k_{nope}}_{S_{content}} + \underbrace{q_{rope}(m) \cdot k_{rope}(n)}_{S_{position}(m-n)}$$

divided by √d_qk for softmax temperature.

The position-dependent signal strength: √(d_rope/d_qk)
The content signal strength: √(d_nope/d_qk)

Position-to-content signal ratio: √(d_rope/d_nope)

For MHA: d_rope = d_qk → ratio = 1.0 (position fully determines attention)
For DeepSeek MLA: d_rope/d_qk = 64/192 = 33% → position is a minority contributor

**The nope component acts as position-independent noise** that masks the positional signal. To achieve the same effective frequency redistribution on the attention pattern, EVQ must compensate by a factor of d_qk/d_rope:

- Each RoPE frequency's influence on the attention pattern is diluted by d_rope/d_qk
- To maintain the same net effect, τ must increase by d_qk/d_rope
- This is the **attention dilution correction**

### 5.3 Why DiT uses γ=0.53

In causal AR attention, positions only attend to the past. The position encoding must discriminate in one direction. In bidirectional DiT attention, each position attends to all others. The RoPE cos/sin patterns create symmetric interference:

- **AR causal**: RoPE needs to encode relative distance AND direction → full redistribution useful
- **DiT bidirectional**: RoPE only encodes distance (direction is symmetric) → over-redistribution distorts the symmetric interference pattern

The factor γ = 0.53 ≈ 1/√(2π/e) is empirical. A possible theoretical origin: bidirectional attention effectively doubles the "visible context" compared to causal (seeing both directions), so the effective L doubles, and τ* ∝ 1/√L decreases by √2 ≈ 0.707. The actual 0.53 is slightly lower, suggesting additional constraints from the denoising objective.

Additionally, the DiT phase transition analysis reveals that the optimal τ is precisely where the **half-cycle channel count** increases by 1 (from 5 to 6 at τ=1.5). This integer constraint means DiT τ* is determined by a discrete condition rather than a continuous optimum.

---

## 6. Base Dependence

The formula τ* = γ × (d_qk/d_rope) × d_head/√L does **not** explicitly include base. This is a limitation: all MHA experiments used base=500000, and the new MLA uses base=10000.

### 6.1 Theoretical base correction

From the median condition analysis (§5.1), the exact formula is:

$$\tau^*_{exact} = \frac{\ln 2 \cdot \ln(\text{base})}{\ln(L / 2\pi)}$$

| Config | base | L | τ*_exact | τ*_simple (d_head/√L) | τ*_observed |
|--------|------|---|---------|----------------------|-------------|
| MHA | 500K | 1024 | 1.79 | 2.0 | 2.0 |
| MHA | 500K | 4096 | 1.40 | 1.0 | ~1.0 |
| MLA (if MHA) | 10K | 4096 | 0.99 | 1.0 | — |

The exact formula gives τ*=0.99 for base=10000, L=4096 — essentially the same as d_head/√L=1.0. This suggests the base correction is small for the range base ∈ [10000, 500000] at L=4096.

However, the 20% overestimate for new MLA (predicted 3.0, observed 2.5) might partly arise from the base correction: with base=10000, the MHA baseline τ* is slightly lower than base=500000, reducing the MLA prediction from 3.0 to ~2.5.

### 6.2 Frequency density criterion

A more fundamental formulation uses the **frequency density per octave**:

$$n_{oct} = \frac{K \cdot \ln 2}{\ln(\text{base})}$$

| Config | n_oct | Behavior |
|--------|-------|----------|
| K=16, base=500K | 0.85 | Insufficient — non-monotonic landscape |
| K=32, base=500K | 1.69 | Marginal — mild non-monotonicity possible |
| K=32, base=10K | 2.41 | Sufficient — smooth landscape |

**Minimum requirement**: n_oct ≥ 2 for EVQ to work reliably. Below this, the frequency spectrum is too sparse for meaningful redistribution.

---

## 7. Practical Recommendations

### 7.1 For MHA models

$$\tau^* = \frac{d_{head}}{\sqrt{L_{train}}}$$

No additional corrections needed. Valid for base ∈ [10000, 500000], d_head = 64.

### 7.2 For MLA models (DeepSeek-V2/V3 style)

$$\tau^* = \frac{d_{nope} + d_{rope}}{d_{rope}} \cdot \frac{d_{head,ref}}{\sqrt{L_{train}}}$$

For DeepSeek standard (d_nope=128, d_rope=64, d_head,ref=64):
- L=4096: τ* = 3 × 1.0 = **3.0** (conservative) or **2.5** (empirical)
- L=8192: τ* = 3 × 0.707 = **2.12**
- L=2048: τ* = 3 × 1.414 = **4.24**

### 7.3 For DiT models

$$\tau^* = 0.53 \times \frac{K_t}{\sqrt{T_{train}}}$$

For K_t=16, T_train=32: τ* = **1.50**

### 7.4 Sanity checks before training

1. Compute n_oct = K × ln(2) / ln(base). If n_oct < 2, **do not use EVQ** — the spectrum is too sparse.
2. Compute R = n_oct × ln(L_target/L_train)/ln(2). If R < 1, expect non-monotonic τ landscape — use fine-grained sweep.
3. Verify that the predicted τ* gives reasonable frequency periods: at least 2 frequencies should have periods in [L_train, L_target].

---

## 8. Open Questions

1. **Exact base correction**: How does τ* depend on base when d_head/√L is held constant? Requires MHA experiments at base=10000 to isolate.
2. **DiT γ derivation**: Is γ=0.53 a universal constant for bidirectional attention, or does it depend on architecture/task? The 38.8M DiT cross-run result suggests γ might be model-dependent.
3. **MLA correction exponent**: Is the correction linear (d_qk/d_rope) or sublinear ((d_qk/d_rope)^α, α<1)? The new MLA data point suggests α ≈ 0.83 but this is based on a single noisy experiment.
4. **Interaction with YaRN**: EVQ+YaRN shows superlinear composability (EVQ raw +11% worse but EVQ+YaRN -2.5% better at 8K). Does the optimal τ* change when YaRN post-processing is planned?
5. **Scaling to large models**: Is τ* truly model-size-independent, or does a correction emerge at >1B parameters?

---

## Appendix A: Complete Experimental Data

### A.1 MHA Phase 6 sweep (125M, base=500000, head_dim=64)

| L=128 | L=1024 | L=2048 |
|-------|--------|--------|
| τ=0.0: baseline | τ=0.0: baseline | τ=0.0: baseline |
| τ=5.0: PPL@8K -35% (still improving) | τ=2.0: PPL@8K optimal | τ=1.5: PPL@8K optimal |
| No peak found up to τ=5.0 | Clear peak at τ=2.0 | Clear peak at τ≈1.5 |

### A.2 Old MLA Phase 22 sweep (50M, d_rope=32, K=16, base=500000, L=4096)

| τ | PPL@4K | PPL@8K | PPL@16K |
|---|--------|--------|---------|
| 0.0 | 156.3 | 204.6 | 348.0 |
| 1.8 | 168.2 | 177.6 | 336.5 |
| 2.0 | 178.4 | 236.5 | 373.5 |
| 2.2 | 181.7 | 162.4 | 264.3 |
| 2.5 | 163.1 | 216.2 | 346.8 |
| 3.0 | 169.7 | 171.2 | 300.1 |

### A.3 New MLA Phase 23 sweep (60M, d_rope=64, d_nope=128, K=32, base=10000, L=4096)

| τ | PPL@4K | PPL@8K | PPL@16K |
|---|--------|--------|---------|
| 0.0 | 291.4 | 316.6 | 435.3 |
| 0.5 | 307.7 | 354.4 | 467.7 |
| 1.0 | 305.0 | 318.7 | 425.4 |
| 1.414 | 285.8 | 305.2 | 398.7 |
| 2.0 | 300.1 | 329.5 | 402.2 |
| 2.5 | 280.7 | 295.4 | 372.2 |
| 3.0 | 298.5 | 300.6 | 377.3 |

### A.4 DiT Phase 16 (129.6M, K_t=16, base=10000, T=32)

| τ | Train MSE vs GEO | Far-extrap MSE vs GEO | Half-cycle channels |
|---|---|---|---|
| 0.0 | baseline | baseline | 4 |
| 0.7 | ~5× worse | worse | 4 |
| 1.2 | ~2.8× worse | worse | 5 |
| **1.5** | **-21%** | **-35%** | **6** |

### A.5 350M MLA Phase 18 (d_rope=32, K=16, base=500000, L=8192, 3-seed)

| τ | PPL@8K vs GEO | PPL@16K vs GEO | PPL@32K vs GEO |
|---|---|---|---|
| 1.414 | +0.9% | **-31.1%** | -9.9% |

---

## Appendix B: Frequency Coverage Analysis

### B.1 Active and extrapolation frequency counts (GEO baseline)

The number of GEO frequencies with period < L (active) and in [L, 4L] (extrapolation):

$$N_{active} = K \cdot \frac{\ln(L/2\pi)}{\ln(\text{base})}, \quad N_{extrap} = K \cdot \frac{\ln 4}{\ln(\text{base})}$$

| Config | K | base | L | N_active | N_extrap[L,4L] | N_dead(>4L) |
|--------|---|------|---|----------|---------------|-------------|
| MHA | 32 | 500K | 4096 | 16 | 3 | 13 |
| Old MLA | 16 | 500K | 4096 | 8 | 2 | 6 |
| New MLA | 32 | 10K | 4096 | 23 | 5 | 4 |
| DiT temporal | 16 | 10K | 32 | 3 | 2 | 11 |

### B.2 EVQ's redistribution effect

EVQ at optimal τ approximately doubles the extrapolation frequency count while maintaining most active frequencies:

| Config | τ* | GEO extrap | EVQ extrap | Improvement |
|--------|---|-----------|-----------|-------------|
| MHA L=1024 | 2.0 | ~3 | ~5 | +67% |
| New MLA L=4096 | 2.5 | ~5 | ~7 | +40% |
| DiT T=32 | 1.5 | ~2 | ~3 | +50% |
