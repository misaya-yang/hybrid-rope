# Attention interface: is G_attn = E[Φᵀ F_z Φ] a Fisher information?

Independent audit note. All statements numerically verified (scripts: verify_core.py,
inline numpy checks; MC rel err ≤ 3e-3 = sampling noise).

## Setup (exact, T1)

For one head, the RoPE part of the pre-softmax logit between query m and key n (Δ = m−n):

    l_z(Δ) = Φ(Δ) f_z,   Φ(Δ) = [cos ω₁Δ, sin ω₁Δ, …, cos ω_KΔ, sin ω_KΔ] ∈ R^{2K}
    C_k = q_c r_c + q_s r_s,   D_k = q_s r_c − q_c r_s   (exact bilinear in q,r)

## Shape analysis of G_attn = E[Φᵀ F_z Φ]

Φ ∈ R^{(2L−1)×2K}. The expression ΦᵀF_zΦ is well-typed only for F_z ∈ R^{(2L−1)×(2L−1)}
(logit space). The only consistent reading is F_z = l_z l_zᵀ, giving (verified, MC 2e-3):

    G_attn = E[Φᵀ l_z l_zᵀ Φ] = (ΦᵀΦ)(Σ + θθᵀ)(ΦᵀΦ),   Σ = Cov(f_z), θ = E[f_z].

Reading F_z = f_z f_zᵀ (coefficient space) is dimensionally ill-typed; the pullback
E[(Φᵀf)(Φᵀf)ᵀ] exists but is not of the form ΦᵀFΦ.

## Fisher analysis (verified)

Model: f_z ~ P (mean θ, cov Σ), l = Φf + ε, ε ~ N(0, σ²I). The marginal is exactly Gaussian:
p(l;θ) = N(Φθ, σ²I + ΦΣΦᵀ), so the Fisher is closed-form (T1):

    I(θ) = Φᵀ(σ²I + ΦΣΦᵀ)⁻¹Φ         [MC-verified 1.7e-3; θ-independent: 3e-3]

Expansion (T2, valid for σ² > ‖ΦΣΦᵀ‖):

    I(θ) = σ⁻²ΦᵀΦ − σ⁻⁴ΦᵀΦΣΦᵀΦ + O(σ⁻⁶)
    abs err ∝ σ⁻⁶ (ratio 15.9 per 2× σ at σ = 32, 64, 128 ✓); rel err ∝ σ⁻⁴.

Limits (both verified):
- σ → ∞:  I → σ⁻²ΦᵀΦ  — the static Gram, i.e. exactly the object collision/effrank theory studies.
- σ → 0:  I → Φᵀ(ΦΣΦᵀ)⁺Φ  [verified 9e-4].

## Answers

1. **G_attn is not a Fisher information.** It is the static-Gram-sandwiched coefficient
   second moment. Its covariance part ΦᵀΦΣΦᵀΦ equals −σ⁴ × the Fisher's leading data
   correction: coefficient heterogeneity *reduces* information about the mean positional
   structure (expected: more prior spread ⇒ more ambiguity).
2. The mean part θθᵀ does not enter the Fisher at all (location invariance, verified).
3. Static collision/effrank theory = leading-order Fisher geometry (σ⁻²ΦᵀΦ) — exact in the
   isotropic-coefficient limit. G_attn measures the Σ-corrections. Order of operations:
   measure G_attn (Σ, θ) from a checkpoint → test anisotropy → only then consider a
   Σ-weighted objective. **Do not define a new unified loss from this object.**
4. Relation to the finite-K question: static effrank/logdet objectives implicitly assume
   Σ ∝ I (isotropic usage). The honest "strict relation" is: static rank bounds are
   realized iff attention uses all positional directions equally; G_attn measures the gap.

## Estimation cost (concrete)

Repo has checkpoints: results/weekend_sweep/*.pt (e.g. train_tinystories_25000000_2048.pt),
results/ ≈ 30 GB. Recipe: load checkpoint via repo training code → one forward pass over
N token pairs → per head extract (C_k, D_k) via the exact bilinear formulas →
G_attn = (1/N)Σ_z (Φᵀl_z)(Φᵀl_z)ᵀ. Cost O(N·2K·(2L−1)); no gradients, no training.

Caveats: pre-softmax only (softmax nonlinearity is outside this object); RoPE part only
(MLA uses partial RoPE — per-head decomposition still exact on the RoPE part); uncentered
vs centered matters for the θ-direction. Loader path pending code audit (task 1).
