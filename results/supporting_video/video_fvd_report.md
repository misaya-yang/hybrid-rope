# Video Temporal FVD Verification Report

**Date**: 2026-03-15
**Hardware**: NVIDIA RTX PRO 6000 Blackwell 96GB
**Model**: VideoGPT 268.7M (1024 hidden, 16 layers, 16 heads, head_dim=64)
**Train**: 32 frames (2048 tokens), Extrapolate to: 128 frames (8192 tokens, 4x)

---

## Experiment 1: Standard Moving MNIST (linear motion, 2 digits)

Trained from scratch, 16 epochs (~2h/arm). N=256 generated videos.

### PPL (128f, 4x extrapolation)

| | geo_k16 | evq_k16 | EVQ advantage |
|--|---------|---------|---------------|
| PPL (raw) | 7.053 | **6.095** | **-13.6%** |
| PPL (YaRN) | 6.984 | **4.633** | **-33.7%** |

### Generation Quality (128f)

| Metric | geo_k16 | evq_k16 | Delta |
|--------|---------|---------|-------|
| I3D FVD ↓ | **2.254** | 2.272 | +0.8% |
| pixel FVD ↓ | **64.15** | 64.34 | +0.3% |
| MSE ↓ | 0.01823 | 0.01823 | 0% |
| Token acc ↑ | 77.52% | 77.59% | +0.1% |
| Temporal coherence (real) | 0.0557 | 0.0557 | — |
| Temporal coherence (gen) | 0.0266 | 0.0263 | -1.1% |

**Conclusion**: PPL strongly favors EVQ (13-34%), but generation quality is identical. Both models produce videos that are 2x smoother than real data (coherence 0.026 vs 0.056). The task is too simple for FVD to discriminate.

---

## Experiment 2: Oscillating MNIST (sinusoidal motion, 3 digits, periods 16/24/32)

Fine-tuned from Exp 1 checkpoints, 4 epochs (~30min/arm), LR=9e-5 (0.3x base). N=256.

### PPL (128f, 4x extrapolation)

| | geo_k16 | evq_k16 | EVQ advantage |
|--|---------|---------|---------------|
| PPL (raw) | 4.832 | **4.440** | **-8.1%** |
| PPL (YaRN) | 4.787 | **3.481** | **-27.3%** |

### FVD Metrics (128f)

| Metric | geo_k16 | evq_k16 | Delta | Winner |
|--------|---------|---------|-------|--------|
| I3D FVD ↓ | 2.992 | **2.973** | -0.6% | EVQ |
| pixel FVD ↓ | 79.50 | **79.08** | -0.5% | EVQ |
| Temporal I3D FVD ↓ | 3.653 | **3.634** | -0.5% | EVQ |
| Temporal pixel FVD ↓ | 3.621 | **3.541** | -2.2% | EVQ |

### Temporal Quality Metrics (128f)

| Metric | geo_k16 | evq_k16 | Delta | Winner |
|--------|---------|---------|-------|--------|
| FVMD-lite ↓ | 2.208 | **2.179** | -1.3% | EVQ |
| Motion FFT JSD ↓ | 0.268 | **0.266** | -0.7% | EVQ |
| Dynamic degree ratio ↑ | 0.375 | **0.380** | +1.3% | EVQ |
| NoRepeat ↑ | **0.940** | 0.939 | -0.1% | geo |
| MSE ↓ | **0.03009** | 0.03009 | 0% | tie |
| Token acc ↑ | **70.07%** | 70.0% | -0.1% | tie |

### Temporal Autocorrelation (key diagnostic)

| Lag (frames) | Real | geo_k16 gen | evq_k16 gen | Closer to real? |
|--------------|------|-------------|-------------|-----------------|
| 8 (P₁/2) | 0.191 | 0.558 | **0.553** | EVQ (slightly) |
| 16 (=P₁) | 0.316 | 0.395 | **0.392** | EVQ (slightly) |
| 24 (=P₂) | 0.335 | 0.198 | **0.196** | EVQ (slightly) |
| 32 (=P₃) | 0.404 | -0.086 | **-0.085** | EVQ (slightly) |

Both models fail equally at lag 32 — real autocorrelation is 0.404, generated is -0.085.

### Temporal Coherence

| | Real | geo gen | evq gen |
|--|------|---------|---------|
| Coherence | 0.0640 | 0.0311 | **0.0315** |
| Dynamic degree | 0.0272 | 0.0102 | **0.0103** |

Generated videos have only ~38% of real motion intensity.

---

## Cross-Experiment Summary

### Oscillating is harder than linear

| Metric | Linear (EVQ) | Oscillating (EVQ) | Change |
|--------|-------------|-------------------|--------|
| Token acc | 77.6% | 70.0% | ↓7.6pp |
| MSE | 0.018 | 0.030 | ↑67% |
| I3D FVD | 2.27 | 2.97 | ↑31% |
| Dynamic ratio | ~0.47 | 0.38 | ↓19% |

### EVQ wins PPL everywhere, generation is always ~tied

| Dataset | PPL advantage | Best generation metric advantage |
|---------|---------------|----------------------------------|
| Linear MNIST | 13.6-33.7% | <1% (tied) |
| Oscillating MNIST | 8.1-27.3% | <2.2% (tied) |

---

## Diagnosis: Why PPL advantage doesn't translate to generation quality

1. **Sampling collapses distribution differences**: Temperature=0.9 + top-k=50 means both models sample from nearly identical top-token sets. An 8% PPL difference means the *full distribution* is better, but sampling only uses the top of the distribution.

2. **Token-level errors average out**: At 70-77% token accuracy, ~30% of tokens are "wrong" for both models. The *pattern* of errors is similar — mostly off-by-one pixel values. FVD and other distribution metrics can't distinguish these.

3. **Both models are equally "too smooth"**: Generated motion intensity is only 38% of real (dynamic_degree_ratio). Both models default to conservative, smooth motion at extrapolation. The failure mode is the same for both allocations.

4. **Autocorrelation collapses identically**: Both models lose oscillation periodicity at lag 32 (real=0.40, gen=-0.09). Neither allocation preserves long-range temporal structure — this is a model capacity / training issue, not a frequency allocation issue.

---

## Potential approaches to amplify the difference

### A. Reduce sampling temperature (e.g., 0.3 or greedy)
- **Theory**: Lower temperature forces the model to commit to its top prediction. If EVQ's top-1 is more often correct, greedy decoding should amplify the quality gap.
- **Risk**: Both models might produce identical greedy outputs (same top-1 token).
- **Cost**: ~1h (generation only, no retraining).

### B. Measure top-1 accuracy directly (no sampling)
- **Theory**: Skip generation entirely. For each position in the test set, check if the model's argmax prediction matches the ground truth. This directly measures the PPL advantage in terms of prediction quality.
- **Cost**: ~10 min (just forward pass on test data, no generation).
- **Note**: This is essentially PPL in a different form, but more interpretable for the paper.

### C. Increase extrapolation ratio (8x or 16x)
- **Theory**: At 4x, both models are still "okay" (70% accuracy). At 8x or 16x, both degrade but EVQ should degrade slower, creating a larger gap.
- **Requires**: Test data with 256f or 512f, which needs regenerating.
- **Cost**: ~2h (data gen + eval only, reuse fine-tuned checkpoints).

### D. Per-frame quality curve (not aggregate)
- **Theory**: Instead of a single FVD number, plot per-frame MSE/accuracy as a function of frame index. At early extrapolation frames (33-64), both are good. At late frames (96-128), EVQ should be measurably better.
- **Cost**: ~10 min (just analyze existing generated videos... but we didn't save them).
- **Requires**: Re-generate and save videos, then compute per-frame metrics.

### E. Use lower temperature + per-frame curve together
- Generate at temp=0.1 (near greedy)
- Save per-frame token accuracy
- Plot accuracy vs frame index for both models
- **This is the most promising combination**: amplifies differences AND shows where they occur.
- **Cost**: ~1h.

### F. Larger model or longer training
- More parameters / more epochs might let the models differentiate more.
- **Cost**: 5-10h. Not recommended given budget constraints.
