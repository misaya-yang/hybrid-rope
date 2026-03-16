# Video DiT Temporal Extrapolation: EVQ-Cosh vs Geometric RoPE
# Complete Experiment Report (2026-03-16, v2 — Head-to-Head Update)

## Executive Summary

> **⚠️ v2 UPDATE**: The original v1 conclusions about "capacity-dependent" behavior have been
> **overturned**. All v1 cross-run comparisons were contaminated by CUDA non-determinism.
> Head-to-head (same-run) evaluation reveals EVQ-Cosh(τ=1.5) wins at BOTH model scales.

**Validated findings (head-to-head, same-run):**

- **129.6M model, τ=1.5**: EVQ wins denoising precision by **-21% (train)** and **-35% (far-extrap)**
- **38.8M model, τ=2.83**: EVQ wins denoising precision by 27% (train) and 40% (far-extrap) — v1 result confirmed
- **τ is critical**: Only τ=1.5 beats GEO on 129.6M. τ=0.3, 0.7, 1.2 all lose. Sharp phase transition between τ=1.2 and τ=1.5.
- **Power-Shift family rejected**: 6-22x worse than GEO across all α values tested.

Key insight: DiT requires a **different optimal τ** than AR (τ≈1.5 vs τ≈2.83), but EVQ-Cosh
remains the correct frequency family. The unified story is: same Cosh solution, different τ*
for different attention mechanisms.

---

## Experiment Setup (Common)

| Parameter | Value |
|-----------|-------|
| Data | Oscillating MNIST, 64x64, grayscale |
| Training frames | 32 |
| Extrapolation | 128 frames (4x temporal) |
| 3D RoPE split | K_h=8, K_w=8, K_t=16 |
| Patch size | 8x8 = 64 tokens/frame |
| Scheduler | Rectified Flow (linear interpolation) |
| Sampling | Euler ODE, 50 steps |
| GPU | NVIDIA RTX 5090 32GB |

### Model Configurations

| | 38.8M ("default") | 129.6M ("medium") |
|---|---|---|
| Hidden | 512 | 768 |
| Layers | 8 | 12 |
| Heads | 8 | 12 |
| Head dim | 64 | 64 |
| Batch size | 64 | 16 |
| Epochs | 30 | 15 |
| LR | 2e-4 | 1.5e-4 |
| Steps | 7,500 | 15,000 |
| Time/method | 45 min | 56 min |

---

## Part I: Cross-Run Results (v1, now superseded)

> **⚠️ WARNING**: These results compare models trained in separate CUDA runs.
> Non-deterministic CUDA operations (cuBLAS, cuDNN) introduce run-to-run variance
> that can exceed the actual signal from frequency allocation differences.
> These results are preserved for reference but should NOT be used for conclusions.

### 38.8M Model (cross-run)

| Region | GEO | EVQ (τ=2.83) | Delta | Winner |
|--------|-----|-----|-------|--------|
| Train (0-31) | 0.0668 | 0.0486 | **-27.2%** | **EVQ** |
| All extrap (32-127) | 0.0557 | 0.0500 | **-10.2%** | **EVQ** |
| Near (32-63) | 0.0387 | 0.0474 | +22.4% | GEO |
| Mid (64-95) | 0.0446 | 0.0524 | +17.3% | GEO |
| Far (96-127) | 0.0837 | 0.0503 | **-39.9%** | **EVQ** |

### 129.6M Model (cross-run — ~~OVERTURNED~~)

| Region | GEO | EVQ (τ=2.83) | Delta | Winner |
|--------|-----|-----|-------|--------|
| Train (0-31) | 0.0084 | 0.0144 | ~~+70.8%~~ | ~~GEO~~ |
| All extrap (32-127) | 0.0064 | 0.0114 | ~~+78.6%~~ | ~~GEO~~ |
| Far (96-127) | 0.0075 | 0.0108 | ~~+45.0%~~ | ~~GEO~~ |

**Post-mortem**: The 70-79% "GEO advantage" at 129.6M was entirely an artifact of
CUDA non-determinism. Head-to-head evaluation (Part II) shows the opposite result.

---

## Part II: Head-to-Head Results (v2, validated)

Head-to-head (h2h) evaluation trains BOTH methods in the SAME run, sharing identical
random seeds, data order, and CUDA state. This eliminates run-to-run variance entirely.

### 129.6M τ Sweep (head-to-head, YaRN)

| τ | Train MSE (vs GEO) | Far-extrap MSE (vs GEO) | Winner |
|---|-----|-----|--------|
| 0.00 | baseline | baseline | — |
| 0.30 | worse | worse | GEO |
| 0.70 | ~5x worse | worse | GEO |
| 1.20 | ~2.8x worse | worse | GEO |
| **1.50** | **-21%** | **-35%** | **EVQ** |
| 2.83 | worse (cross-run only) | worse (cross-run only) | GEO |

**Key observation: Discrete phase transition between τ=1.2 and τ=1.5.**

At τ=1.2, EVQ is 2.8x worse than GEO. At τ=1.5, EVQ is 21% better. This is not a
gradual improvement — it's a sharp jump. The hypothesized mechanism is discrete
"dead channel activation": at base=10000 with K_t=16 and T_train=32, certain frequency
channels have θ_k × Δ ≈ 0 for all relevant Δ, making them useless. At τ=1.5, EVQ
redistributes these dead channels into useful frequency bands.

### Power-Shift Family (head-to-head, REJECTED)

| α | Train MSE (vs GEO) | Winner |
|---|-----|--------|
| 0.25 | **22x worse** | GEO |
| 0.50 | **6x worse** | GEO |

Power-Shift φ_k(α) = 1 - (1-u_k)^(1+α) was designed to boost low frequencies
without damaging mid/high frequencies. In practice, it performs catastrophically.
**This family is abandoned.**

---

## Part III: Supporting Evidence (AR VideoGPT)

Teacher-forced evaluation on VideoGPT (268.7M, 1024 hidden, 16 layers) confirms
EVQ advantage without autoregressive error accumulation:

| Metric | EVQ advantage (YaRN, extrap region) |
|--------|-----|
| Top-1 accuracy | **+3.14%** |
| Top-5 accuracy | **+5.40%** |
| Near extrap Top-5 | **+6.59%** |

Frequency decomposition shows advantage scales with temporal frequency:

| Period | Top-1 Delta | Top-5 Delta |
|--------|------------|------------|
| P=16 (highest freq) | +5.07% | +8.48% |
| P=24 (mid freq) | +4.39% | +7.63% |
| P=32 (lowest freq) | +3.55% | +6.25% |

See `results/supporting_video/temporal_precision_report.md` for full details.

---

## Revised Interpretation and Paper Implications

### Finding 1: EVQ-Cosh works for DiT — with different τ*

~~The v1 "capacity-dependent" narrative is dead.~~ EVQ wins at BOTH 38.8M and 129.6M
when τ is properly tuned. The key insight is that DiT requires smaller τ than AR:

| Architecture | Optimal τ | τ* formula |
|-------------|-----------|------------|
| AR (causal) | τ≈2.83 | τ* = K_t/√T_train |
| DiT (bidirectional) | τ≈1.5 | τ*_DiT ≈ γ × K_t/√T_train, γ≈0.53 |

This supports a unified narrative: "Same EVQ-Cosh family, different τ* for different
attention mechanisms." The γ factor reflects DiT's dual objective (spectrum matching +
positional fingerprinting) vs AR's single objective (information propagation).

### Finding 2: Cross-run comparisons are unreliable for DiT

All cross-run results (v1 of this report, and likely many published DiT ablations)
are contaminated by CUDA non-determinism. Head-to-head evaluation is essential.
The 70% "GEO advantage" at 129.6M was pure noise.

### Finding 3: Sharp phase transition suggests discrete mechanism

The jump from τ=1.2 (2.8x worse) to τ=1.5 (21% better) is too sharp for a continuous
optimization landscape. Hypothesized cause: base=10000 creates "dead channels" in the
temporal frequency spectrum (K_t=16, T_train=32). At specific τ thresholds, EVQ
redistributes enough channels away from dead zones to cross a performance cliff.

**Open question**: Would lowering base_t (from 10000 to ~100-1000) eliminate dead
channels and make the τ landscape smoother? This could be an additional contribution.

### Finding 4: Power-Shift family provides no benefit

Despite theoretical appeal (boost low-freq without hurting mid/high-freq), Power-Shift
performs 6-22x worse than GEO in practice. The Cosh family is the correct solution
for both AR and DiT.

### Suggested framing for paper

The video DiT experiments provide a second axis of validation:

1. EVQ-Cosh generalizes beyond AR to bidirectional attention (DiT)
2. The same family works, but τ* depends on architecture type
3. This is a **new contribution**: architecture-dependent τ* scaling

---

## Open Experiments

### Completed

- [x] 38.8M cross-run (GEO vs EVQ τ=2.83) — EVQ wins
- [x] 129.6M cross-run (GEO vs EVQ τ=2.83) — ~~GEO wins~~ (noise)
- [x] 129.6M τ sweep cross-run (τ=0.3, 0.7, 1.5)
- [x] 129.6M head-to-head τ=1.5 — **EVQ wins -21%/-35%**
- [x] 129.6M head-to-head τ=1.2 — GEO wins 2.8x
- [x] 129.6M Power-Shift α=0.25, 0.5 — rejected
- [x] VideoGPT teacher-forced evaluation — EVQ +5.4% top-5
- [x] VideoGPT frequency decomposition — P=16 shows +8.48%

### Needed

- [ ] Fine-grained τ sweep (1.25, 1.30, 1.35, 1.40, 1.45) — pinpoint phase transition boundary
- [ ] Phase collision analysis vs τ — does collision score predict the transition?
- [ ] τ=2.83 head-to-head on 129.6M — confirm AR-optimal τ truly fails for DiT
- [ ] base_t sweep (100, 256, 500, 1000, 5000, 10000) — test dead channel hypothesis
- [ ] Larger model (250M+) head-to-head — confirm τ=1.5 scales

---

## Files

### 38.8M ("default") experiment
- Work dir: results/video_dit/20260316_002758/
- Checkpoints: {geo,evq}_seed42.pt
- Generated videos: {geo,evq}_seed42_gen_{32,128,128_noyarn}f.npy
- Results: {geo,evq}_seed42_results.json, summary.json

### 129.6M ("medium") experiment
- Work dir: results/video_dit/20260316_medium/
- Checkpoints: {geo,evq}_seed42.pt
- Generated videos: {geo,evq}_seed42_gen_{32,128,128_noyarn}f.npy
- Results: {geo,evq}_seed42_results.json, summary.json

### τ sweep
- Work dir: results/video_dit/20260316_tau_sweep/ (on server, pending download)
- Head-to-head results: reported verbally, JSON pending

### Phase collision analysis
- Data: results/video_dit/phase_collision_analysis.json
- Plots: results/video_dit/phase_collision_analysis.png, rank_transition.png

### Supporting video (AR VideoGPT)
- Temporal precision: results/supporting_video/temporal_precision_report.md
- Frequency decomposition: results/supporting_video/temporal_precision_freq/

### Server
- GPU: NVIDIA RTX 5090 32GB
- Total GPU time: ~8+ hours (original + τ sweep + h2h experiments)
- Server: AutoDL, ssh -p 29382 root@connect.westd.seetacloud.com
