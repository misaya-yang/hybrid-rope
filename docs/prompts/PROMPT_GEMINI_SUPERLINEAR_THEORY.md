# Prompt: Theoretical Analysis of Superlinear Synergy between Training-time and Inference-time RoPE Frequency Optimization

## Context

We study RoPE (Rotary Position Embedding) frequency allocation for Transformer language models. There are two orthogonal approaches to improving length extrapolation:

1. **Training-time frequency optimization (EVQ)**: Replace the standard geometric frequency schedule ω_k = b^{-k/N} with a variationally optimal schedule derived from minimizing phase-collision energy. The closed-form solution is:
   φ_k(τ) = 1 - (1/τ) · arcsinh((1-u_k) · sinh(τ))
   where τ > 0 controls the redistribution strength. This compresses high-frequency spacing (~0.6×) and expands low-frequency spacing (~1.4×).

2. **Inference-time frequency scaling (YaRN)**: At inference time, apply a progressive frequency warp to the trained model's frequencies, scaling them to handle longer contexts than seen during training. YaRN uses a ramp function that interpolates between full scaling (low frequencies) and no scaling (high frequencies).

## Experimental Data (350M params, base=500K, L_train=2048, 10% passkey-retrieval mix)

Passkey retrieval accuracy at different extrapolation lengths:

| Method | Type | PK@2K (1×) | PK@4K (2×) | PK@8K (4×) |
|--------|------|------------|------------|------------|
| Geo (standard RoPE) | train | 100% | 42% | 46% |
| Geo + YaRN | train+infer | 100% | 100% | 62% |
| Geo + NTK-aware | train+infer | 100% | 100% | 50% |
| EVQ τ=1.5 | train | 100% | 82% | 60% |
| **EVQ + YaRN** | **train+infer** | **100%** | **100%** | **98%** |
| **EVQ + NTK-aware** | **train+infer** | **100%** | **100%** | **88%** |

Key observation: The combination is **superlinear**.
- Geo + YaRN improves 8K from 46% → 62% (+16pp)
- EVQ alone improves 8K from 46% → 60% (+14pp)
- If additive, EVQ + YaRN should give ~76%. Actual: **98%** (+36pp over Geo+YaRN)

## Your Task

Provide a rigorous theoretical analysis explaining **why** training-time frequency optimization (EVQ) and inference-time frequency scaling (YaRN) exhibit superlinear synergy. Specifically:

1. **Decompose the positional encoding problem** into training-time representation quality and inference-time extrapolation capability. What mathematical framework explains why these are multiplicative rather than additive?

2. **Analyze the phase structure**: EVQ expands low-frequency channel spacing during training. YaRN scales frequencies at inference. How does better-separated low-frequency spacing during training create representations that are more amenable to inference-time rescaling?

3. **Information-theoretic perspective**: Can you formulate the synergy in terms of mutual information I(position; representation)? EVQ presumably increases this quantity during training. How does higher positional mutual information interact with YaRN's frequency remapping?

4. **Phase collision analysis**: Standard RoPE's low-frequency channels suffer phase collisions at long distances (cos(ω_k Δ) ≈ cos(ω_j Δ) for nearby ω_k, ω_j). EVQ separates these frequencies. Does this separation create a "cleaner" spectral basis that YaRN can more effectively rescale without introducing new collisions?

5. **Propose a formal bound or inequality** (even conjectural) that captures the superlinear interaction. For example, something of the form:
   Retrieval(EVQ + YaRN) ≥ Retrieval(Geo + YaRN) + f(spacing_gain) · Retrieval(EVQ - Geo)
   where f > 1 captures the amplification.

Please be mathematically rigorous. Use signal processing, information theory, or variational analysis as appropriate. If you need to make assumptions, state them explicitly.
