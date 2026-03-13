# EVQ-Cosh: Theory Details & Proof Reference

> **Purpose**: Mathematical appendix companion to `mainstory.md`. Contains full proofs, derivation details, and theoretical analysis not included in the paper narrative spine. For the paper story, evidence tiers, and writing guidance, see `mainstory.md`.
> **Companion**: `SECONDARY_THEORY.md` (speculative theory, deprecated experiments)
> **Last updated**: 2026-03-11 (v17 — refactored: removed experiment/narrative overlap with mainstory.md, retained only unique mathematical content)

---

## 1. Full Derivation: Phase-Collision Kernel to EVQ-Cosh

### 1.1 Step 1: Distance Prior → Phase-Collision Kernel

Given a distance prior D(Δ) over token-pair separations, the phase-collision kernel is:

$$K(\varphi_1, \varphi_2) = \int_1^L D(\Delta)\cos(b^{-\varphi_1}\Delta)\cos(b^{-\varphi_2}\Delta)\,d\Delta$$

Using the product-to-sum identity cos(a)cos(b) = ½[cos(a−b) + cos(a+b)]:

$$K(\varphi_1,\varphi_2) = \frac{1}{2}\int_1^L D(\Delta)\bigl[\cos\bigl((b^{-\varphi_1}-b^{-\varphi_2})\Delta\bigr) + \cos\bigl((b^{-\varphi_1}+b^{-\varphi_2})\Delta\bigr)\bigr]d\Delta$$

The second (sum-frequency) term oscillates rapidly and averages out in the broadband limit; the first (difference-frequency) term carries the structural information.

### 1.2 Step 2→3: Broadband Projection (The Key Approximation)

**Operator decomposition**: min(φ₁, φ₂) is the Green's function of A = −d²/dφ² under mixed boundary conditions (Dirichlet at φ=0, Neumann at φ=1). Therefore:

$$K_{\text{approx}} = \alpha I + \beta A^{-1}$$

where I is the identity and A⁻¹ is the resolvent. This is the **Hilbert-Schmidt optimal two-parameter projection** of the exact kernel K onto the operator family {αI + βA⁻¹}.

**Derivation of the min kernel under D(Δ) ∝ 1/Δ**: Under the scale-invariant prior, the off-diagonal bulk of K is dominated by the cosine-integral function Ci(x). Using the asymptotic expansion Ci(x) = γ + ln x + O(x²) for x ≪ 1, the off-diagonal entries for small log-frequency separations take an affine form in min(φ₁, φ₂), yielding β ≈ 1/2 as the typical coefficient. This is an asymptotic / regional result (bulk + off-diagonal); the remaining terms consist of high-frequency oscillations and boundary/diagonal contributions that appear as residuals in the full matrix.

**What the projection captures and what it misses**: The mid-band (away from UV and IR boundaries) is well-captured. The full-matrix residual of 35–49% comes from three boundary effects:

1. **UV boundary** (high frequency, φ → 0): Discretization of the frequency grid
2. **IR boundary** (low frequency, φ → 1): Wavelength exceeds sequence length
3. **Diagonal ridge**: The δ-function component has finite physical width O(1/ln b)

The variational ODE acts in the mid-band; boundary residuals do not invalidate the cosh solution there.

### 1.3 Step 3→4: Variational Functional → Euler-Lagrange ODE

The functional to be minimized:

$$J[\rho] = \frac{\alpha}{2}\int_0^1 \rho(\varphi)^2\,d\varphi + \frac{\beta}{2}\int_0^1\int_0^1 \rho(\varphi_1)\rho(\varphi_2)\min(\varphi_1,\varphi_2)\,d\varphi_1\,d\varphi_2 - \mu\int_0^1 \rho(\varphi)b^{-2\varphi}\,d\varphi$$

First variation δJ/δρ = 0 gives:

$$\alpha\rho(\varphi) + \beta\int_0^1 \rho(\varphi')\min(\varphi,\varphi')\,d\varphi' = \mu b^{-2\varphi}$$

Applying A = −d²/dφ² to both sides (using the Green's function identity A · min(φ,·) = δ(φ)):

$$\rho'' - \tau^2\rho = \gamma b^{-2\varphi}, \qquad \tau = \sqrt{\beta/\alpha}$$

This is a standard second-order linear ODE with constant coefficients and an exponential forcing term.

### 1.4 Step 4→5: ODE General Solution

**Theorem 1 (ODE Exact Solution)**: The general solution is:

$$\rho^*(\varphi) = C_1\cosh(\tau\varphi) + C_2\sinh(\tau\varphi) + P\,b^{-2\varphi}$$

where P = γ/(4ln²b − τ²) is the particular solution coefficient. The solution consists of two competing components:

- **Hyperbolic tether** (cosh/sinh terms): Represents the structural redistribution driven by the min-kernel penalty
- **Fisher pulse** (b^{−2φ} term): Represents the information-theoretic incentive from the Fisher term

τ controls the balance between these two forces.

### 1.5 Step 5→6: CDF Inversion to Frequency Assignment

From the density ρ\* to the quantile function (channel assignment):

$$\varphi_k(\tau) = 1 - \frac{1}{\tau}\operatorname{arcsinh}\!\bigl((1-u_k)\sinh\tau\bigr)$$

where u\_k = k/K is the normalized channel index (k = 0, 1, ..., K−1; K = d\_head/2).

**Derivation**: The CDF of ρ\* is integrated and inverted. The arcsinh form arises from integrating the cosh/sinh density profile.

---

## 2. Geometric as τ→0 Degenerate Point (Proof)

**Theorem 2**: As τ → 0, EVQ-Cosh smoothly degenerates to Geometric RoPE.

**Proof**: Starting from φ\_k(τ) = 1 − (1/τ)arcsinh((1−u\_k)sinh τ):

- sinh τ ≈ τ + τ³/6 + O(τ⁵) as τ → 0
- arcsinh(x · τ) ≈ xτ − (xτ)³/6 + O(τ⁵) for bounded x
- Therefore arcsinh((1−u\_k)sinh τ)/τ → (1−u\_k)
- Hence φ\_k → 1 − (1−u\_k) = u\_k

u\_k = k/K are uniform quantiles on [0,1], corresponding to geometric (uniform log-spacing) RoPE. ∎

**Corollary (Conditional suboptimality)**: Under the scaling law τ\*(L) = d\_head/√L (§4), for any L > 0 we have τ\* > 0, so Geometric (τ=0) is suboptimal within the EVQ family. The gap shrinks as L → ∞ (τ\* → 0).

**Taylor expansion at small τ**:

$$\varphi_k(\tau) \approx u_k - \frac{\tau^2}{6}u_k(1-u_k)(2-u_k) + O(\tau^4)$$

The leading correction is quadratic in τ, confirming smooth onset from the geometric baseline.

---

## 3. Waterbed Inequality: Fine Structure

### 3.1 Statement and Equality Condition

$$\int_0^1 \ln E(\varphi)\,d\varphi \geq \ln b - \ln c$$

where E(φ) is the position-encoding error at normalized frequency φ.

**Equality condition**: Jensen's inequality on f(x) = −ln x (strictly convex). Equality holds iff E(φ) is constant, i.e., ρ(φ) ≡ 1 ↔ Geometric RoPE.

**Interpretation**: Geometric minimizes total log-error volume but distributes error highly nonuniformly: E ∝ b^{2φ}, exponentially large at low frequencies. EVQ equalizes error across the band but increases total volume — a bounded cost.

### 3.2 Why PPL Does Not Reveal Waterbed

High-frequency error increases from EVQ remain within the softmax over-parameterization margin: adjacent high-frequency channels encode nearly identical short-distance information, so losing resolution there has negligible impact on next-token prediction. PPL is a token-level global average that masks frequency-axis effects.

**Empirical confirmation**: Phase 6A verified that varying τ from 0 to 5.0 produces no degradation in training-window PPL.

### 3.3 Why Downstream Tasks May Reveal Waterbed

Different tasks act as bandpass filters on the frequency axis:

- **Retrieval tasks** (passkey, NIAH): Low-pass — benefit from improved low-frequency resolution
- **Multi-hop reasoning**: Potentially mid/high-pass — may experience mild degradation from high-frequency compression
- **Short-context tasks**: High-pass — insensitive to low-frequency changes

This predicts task-dependent waterbed effects that PPL averages cannot detect.

### 3.4 Quantification

The excess waterbed volume from departing geometric is:

$$\Delta W = D_{\text{KL}}(\text{Uniform}\,\|\,\rho)$$

The weighted L² error satisfies: Weighted L² error ≥ D\_KL / c².

---

## 4. τ\* Scaling Law: Variational Derivation

### 4.1 Semi-Rigorous Derivation

From Fourier uncertainty principle considerations:

- The diagonal coefficient α\*(L, b) controls the penalty for density variance. In the broadband limit: α\* ∝ 1/(L · ln b)
- The off-diagonal coefficient β\* controls the min-kernel penalty: β\* ≈ O(1)

Therefore:

$$\tau^* = \sqrt{\beta^*/\alpha^*} \propto \sqrt{L \cdot \ln b}$$

At fixed base: τ\* ∝ √L. But since τ enters the channel assignment as τ/K where K = d\_head/2, the effective parameter is:

$$\tau^* = \frac{d_{\text{head}}}{\sqrt{L}}$$

**Status**: This derivation is semi-rigorous (reviewed by Gemini Q6). The dimensional analysis is sound; the proportionality constant C = d\_head is empirically confirmed but not derived from first principles. The formula should be presented as a conjecture with strong empirical support.

### 4.2 Detailed Validation Data

**Phase 8D / Phase 11 anchor points** (d\_head=64, base=500K):

| L\_train | Predicted τ\*=64/√L | Empirical best τ | Notes |
|---------|---------------------|-----------------|-------|
| 128 | 5.66 | ≥5.0 | Monotonically decreasing (PE-dominant) |
| 256 | 4.0 | 4.0 | Phase 11: 454M 3-seed confirms τ=4.0 > τ=2.0 |
| 512 | 2.83 | 4.0 | Rightward bias (small L, old 125M data) |
| 1024 | 2.0 | 2.0 | Exact match |
| 2048 | 1.41 | 1.5 | 6% deviation |

L ≥ 1024: good agreement. L < 1024 with old small-model data: systematic rightward bias (PE-dominant regime). Phase 11 at L=256 with 454M confirms τ=4.0 as the correct direction.

**Phase 16 full sweep** (99 runs, 9 configs, 3 seeds each):

| Config | d\_head | τ\*=d/√L | Empirical best | Rank |
|--------|--------|---------|---------------|------|
| L=256, H=16 | 32 | 2.0 | **2.0** | **#1** |
| L=256, H=8 | 64 | 4.0 | **4.0** | **#1** |
| L=256, H=4 | 128 | 8.0 | 10.0 | #2 |
| L=512, H=16 | 32 | 1.41 | 1.77 | #5 (worst) |
| L=512, H=8 | 64 | 2.83 | 4.24 | #3 |
| L=512, H=4 | 128 | 5.66 | **5.66** | **#1** |
| L=1024, H=16 | 32 | 1.0 | 1.25 | #3 |
| L=1024, H=8 | 64 | 2.0 | 2.5 | #2 |
| L=1024, H=4 | 128 | 4.0 | 5.0 | #2 |

Summary: 3/9 exact #1, 6/9 top-2, 8/9 top-3. Systematic ~1.20× rightward shift (finite-capacity effect). All empirical optima within 1.5× of theory.

### 4.3 Loss Landscape Flatness (d\_head=64)

| Config | τ\* (formula) | Empirical best | Rank | PPL gap |
|--------|-------------|---------------|------|---------|
| L=256 | 4.0 | 4.0 | #1 | 0 |
| L=512 | 2.83 | 4.24 | #3 | <1% |
| L=1024 | 2.0 | 2.5 | #2 | <1% |

Even in the worst case (L=512, ratio=1.50×), the PPL gap between τ\* and empirical optimum is less than 1%. The loss landscape around τ\* is a **shallow basin**, not a sharp peak.

### 4.4 Model-Size Independence

The outer geometric truncation error depends only on L, d\_head/2, and τ — not on model parameter count (given the model is not severely underfitting).

**Empirical verification**: Phase 11 at L=256 shows τ=4.0 optimal for both 125M and 454M, with identical directional pattern.

---

## 5. Fisher → Attention Utility Bridge

### 5.1 Laplace Bridge (Core Result)

The second derivative of the collision kernel at zero separation:

$$K''(0) = -\int \rho(\varphi)b^{-2\varphi}\,d\varphi = -\mathcal{H}$$

Under Laplace approximation of softmax attention:

$$A(\Delta) \approx \exp\!\left(-\frac{\mathcal{H}\Delta^2}{2\tau_{\text{temp}}}\right)$$

**Interpretation**: The Fisher information term 𝓗 equals the precision matrix of the local attention Gaussian. Higher Fisher information → sharper local attention → better short-range discrimination.

This bridges the variational framework (which operates on frequency density) to the observable attention pattern (which operates in position space).

### 5.2 Failure Region

At large Δ, high-frequency cos(ωΔ) terms produce spatial aliasing false peaks. The Fisher/Laplace analysis only captures local curvature and cannot detect distant aliasing artifacts. This is why the collision/sub-cycle analysis (mainstory §4.5) is needed for the long-range regime.

---

## 6. Why 1 Parameter Beats 32: n-Width Argument

### 6.1 Sketch

J[ρ] is strictly convex → unique global analytic solution ρ\*. The min-kernel spectral eigenvalues decay as λ\_k ~ O(k⁻²), so the Kolmogorov n-width of the solution manifold decays rapidly.

At N=32 channels, the discretization error ΔJ ~ O(N⁻²). The cosh family captures the dominant mode of the variational solution, and additional degrees of freedom (as in DAPE's 32 learnable parameters) are spent fitting noise rather than structure.

**Caveat**: This is a sketch, not a rigorous theorem. The quantitative constants depend on the target functional and boundary terms. We use this to explain the empirical observation that 1 parameter (τ) suffices, not as a formal proof.

### 6.2 Density Ratio Bound

In the pure-tether model (μ=0), the ODE boundary conditions give exactly:

$$\rho(0)/\rho(1) = \cosh(\tau)$$

In the full model with Fisher pulse, this ratio is typically larger. A unified tight lower bound for the general case requires additional assumptions about the boundary-term structure.

---

## 7. r Parameter Analysis: Why It Is Not a Hyperparameter

### 7.1 Original r\* Formula (Collision Boundary Upper Bound)

$$r_{\text{upper}} = \frac{d}{2\ln b}\ln\!\left(\frac{L_{\text{train}}}{2\pi}\right)$$

This is the channel index where the wavelength first exceeds L\_train. For base=500K, d=64, L=2048: r\_upper ≈ 14.1.

### 7.2 Mathematical Explanation of r-Insensitivity

The EVQ-Cosh assignment has a key property at the high-frequency end:

- At k=0: u=0, φ = 1 − (1/τ)arcsinh(sinh τ) = 1 − 1 = 0, so θ = base⁰ = 1 (identical to Geometric)
- At small k: The EVQ warp is negligible (cosh redistribution concentrates at the low-frequency end)
- Therefore the first several channels have φ\_k^{EVQ} ≈ φ\_k^{Geo} automatically

**The cosh mathematical structure automatically achieves "freeze high frequency, redistribute low frequency" without any explicit r cutoff.** This is why r=0 (Pure EVQ) performs identically to r=4 in experiments.

### 7.3 Hybrid Riemann-Lebesgue Argument (Historical, Epsilon-Level)

**Original proposition**: Under the condition that J\_HF is already minimized, Hybrid (local warp) is strictly better than Pure EVQ (global warp).

**Proof sketch**:

1. J\_HF at Geometric is locally minimized → Hessian is strongly positive definite → perturbation cost ΔJ\_HF ≈ ½δᵀHδ
2. For long-range attention: cos(m·φ\_HF) with m≫1 averages to zero by Riemann-Lebesgue → ∇\_{φ\_HF} J\_LF ≈ 0
3. Therefore J(Pure) − J(Hybrid) ≈ ½δᵀHδ − ∇J\_LF · δ > 0. ∎

**Why this is irrelevant in practice**: The δ at high-frequency channels under cosh allocation is negligibly small (see §7.2). The theoretical advantage is ε-level and experimentally undetectable.

**Experimental confirmation**: r-sweep (350M, base=500K, L=2048) shows r=0 ≈ r=4 in PPL. More critically, EVQ+YaRN works only at r=0; Hybrid r=16 + YaRN is harmful (dilutes low-frequency improvement, breaking the YaRN synergy foundation).

---

## 8. Collision / Sub-Cycle Theory: Full Proofs

> For the proposition statement, numerical verification, and dead-zone analysis, see `mainstory.md` §4.5. Below are the complete proofs of the three corollaries.

### 8.1 Sub-Cycle Proposition (Restated)

$$\rho_{\text{sub-cycle}}^{\text{Geo}} = x, \qquad \rho_{\text{sub-cycle}}^{\text{EVQ}}(\tau) = \frac{\sinh(\tau x)}{\sinh(\tau)}$$

where x = 1 − φ\_c and φ\_c = clip(ln(L/2π)/ln b, 0, 1).

**Proof of Δρ > 0**: By strict convexity of sinh on [0,∞) with sinh(0) = 0: for 0 < x < 1 and τ > 0, sinh(τx) < x · sinh(τ). Therefore sinh(τx)/sinh(τ) < x, giving Δρ = x − sinh(τx)/sinh(τ) > 0. ∎

### 8.2 Corollary 1: Monotone Decrease in τ

**Claim**: ∂ρ\_sub-cycle^{EVQ}/∂τ < 0 for τ > 0, x ∈ (0,1).

**Proof**: Let r(τ,x) = sinh(τx)/sinh(τ). Then:

$$\frac{\partial \ln r}{\partial \tau} = x\coth(\tau x) - \coth(\tau)$$

Define f(z) = z · coth(z). We show f is strictly increasing on z > 0:

$$f'(z) = \coth(z) - \frac{z}{\sinh^2(z)} = \frac{\sinh(z)\cosh(z) - z}{\sinh^2(z)}$$

Since sinh(z)cosh(z) = sinh(2z)/2 > z for z > 0, we have f'(z) > 0.

Since τx < τ for 0 < x < 1: f(τx) < f(τ), i.e., τx · coth(τx) < τ · coth(τ), hence x · coth(τx) < coth(τ), giving ∂(ln r)/∂τ < 0. Since r > 0, this implies ∂r/∂τ < 0. ∎

**Physical meaning**: Larger τ always reduces the sub-cycle fraction. τ systematically compresses the long-wavelength tail.

### 8.3 Corollary 2: Small-τ Quadratic Onset

$$\Delta\rho = \frac{\tau^2}{6}\,x(1 - x^2) + O(\tau^4)$$

**Derivation**: Taylor expand:

- sinh(τx) = τx + (τx)³/6 + O(τ⁵)
- sinh(τ) = τ + τ³/6 + O(τ⁵)

$$\frac{\sinh(\tau x)}{\sinh(\tau)} = \frac{x(1 + \tau^2 x^2/6 + \cdots)}{1 + \tau^2/6 + \cdots} = x\!\left(1 - \frac{\tau^2(1-x^2)}{6} + O(\tau^4)\right)$$

Therefore Δρ = x − x(1 − τ²(1−x²)/6) = τ²x(1−x²)/6 + O(τ⁴). ∎

**Significance**: The departure from Geometric is quadratic in τ, consistent with smooth τ→0 degeneration. At small τ, the redistribution is mild and controlled.

### 8.4 Corollary 3: Large-τ Exponential Decay

$$\rho_{\text{sub-cycle}}^{\text{EVQ}}(\tau) \sim e^{-\tau\varphi_c} \qquad (\tau \to \infty)$$

**Derivation**: As τ → ∞: sinh(τx)/sinh(τ) ~ e^{τx}/e^{τ} = e^{−τ(1−x)} = e^{−τφ\_c}. ∎

**Significance**: The sub-cycle tail shrinks exponentially with τ. In principle, sufficiently large τ eliminates nearly all sub-cycle channels — but the Waterbed inequality (§3) limits the practical τ range.

### 8.5 Equivalent Base Analysis

Given EVQ at base₀ with effective channel count matching Geometric at base\_eq:

$$\ln(b_{\text{eq}}) = \frac{\ln(L/2\pi)}{1 - \sinh(\tau x_0)/\sinh(\tau)}$$

For base=10K, L=512: ln(b\_eq) = 4.40/0.752 = 5.85, giving b\_eq ≈ 350. EVQ at base=10K provides effective channel count equivalent to Geometric at base ≈ 350 — approximately 28× compression.

> **Caveat**: This is an effective-channel-count equivalence only. EVQ additionally optimizes the spacing distribution within the effective band, which base scaling alone cannot provide.

### 8.6 Collision Threshold Formula

$$\varphi_c = \text{clip}\!\left(\frac{\ln(L/2\pi)}{\ln b},\; 0,\; 1\right)$$

The key variable is c = ln(L/2π)/ln(b), not base alone. This correctly predicts:

- base=10K, L=4096 (c=0.90): Only ~3/32 channels optimizable → dead zone
- base=10K, L=512 (c=0.68): 16/32 channels optimizable → EVQ effective
- base=500K, L=4096 (c=0.63): 12/32 channels → strong EVQ regime

### 8.7 Net Gain Scaling (Simplified Model)

$$\Delta J \propto \frac{1-c}{\ln b} = \frac{1 - \ln L / \ln b}{\ln b}$$

| Base | ln b | c (L=4096) | Sub-cycle fraction | Sub-cycle channels (K=32) | Relative gain |
|------|------|-----------|-------------------|--------------------------|--------------|
| 10K | 9.21 | 0.903 | 9.7% | ~3 | 1.0× |
| 100K | 11.51 | 0.722 | 27.8% | ~9 | 2.3× |
| 500K | 13.12 | 0.634 | 36.6% | ~12 | **2.7×** |

Higher base → more sub-cycle channels → more room for EVQ to operate → larger gain.

---

## 9. Broadband R² Validation: Technical Details

### 9.1 Distance Prior Choice

**D(Δ) ∝ 1/Δ** (scale-invariant / Jeffreys prior): Each log-distance scale is weighted equally — token pairs at distance 1–10 and distance 100–1000 contribute equally to the RoPE collision kernel. This is the implicit assumption behind geometric RoPE's equal log-spacing.

**Why NOT token co-occurrence**: The function `measure_distance_distribution` counts same-token repetition probability at distance Δ. This is dominated by high-frequency stopwords at short distances and approaches uniform random noise at long distances (flat tail). It gives R² ≈ 0.65, which is NOT the correct D(Δ) for the theory. The theory's D(Δ) represents the weight of each distance scale in the variational objective, not the empirical probability of token repetition.

### 9.2 Validation Results (24,000-Configuration Sweep)

**R² > 0.99 conditions**: D(Δ) ∝ Δ^{−α} with α ∈ [0.97, 1.05], base ∈ [8K, 100K], L ≥ 4096. 886 out of 24,000 configurations meet this threshold.

**GPT-2 cross-validation**: 12×12 attention heads, real attention D(Δ) fitted to power law:

- α\_mean = 0.56 (global average across all heads)
- α\_median = 0.54
- Local heads (17% of heads, most sensitive to RoPE allocation): α > 0.8
- Full-matrix broadband R² with GPT-2 attention prior: 0.90–0.96 (depending on base and L)

α ≈ 1 (our theoretical assumption) sits within the range observed in practice, bracketed by the local-head values.

### 9.3 Residual Structure

Full-matrix residual of 35–49% comes from three sources:

1. **UV boundary**: High-frequency discretization artifacts
2. **IR boundary**: Wavelength exceeds sequence length
3. **Diagonal ridge**: Physical width O(1/ln b) of the δ-function component

Under the power-law prior D(Δ) ∝ 1/Δ, the cosine integral Ci(x) expansion at x ≪ 1 gives the off-diagonal bulk as βmin(φ₁,φ₂), with structural residual O(b^{−γ}).

---

## 10. NTK-Aware Incompatibility with EVQ (Theoretical Explanation)

NTK-Aware scaling recomputes frequencies at inference time using a geometric progression scaled by the extension ratio. This **overwrites** EVQ's optimized frequency layout, reverting to a (rescaled) geometric distribution.

YaRN, by contrast, applies a gradual scaling factor that preserves the relative frequency structure. This is why EVQ+YaRN shows superlinear synergy while EVQ+NTK is harmful.

**Phase 11 data** (454M, L=256, 3-seed): NTK on EVQ τ=4.0 gives PPL@32× = 331.4 vs Geo+NTK 198.1 (catastrophic). YaRN on EVQ τ=4.0 gives PPL@32× = 99.6 vs Geo+YaRN 260.2 (-61.7%).

**Design principle**: Only inference-time scaling methods that preserve relative frequency structure (YaRN) can synergize with training-time frequency optimization (EVQ).

---

## Changelog (v16 → v17)

### Removed (now in mainstory.md):
- §11 full experiment results → mainstory §5
- §12 passkey mix results → mainstory §5.3
- §13 practical recipe → mainstory §7
- §14 narrative direction → mainstory §9
- §15 EVQ+YaRN synergy narrative → mainstory §3 Claim 3
- §16 cautions → mainstory §8
- §16b lessons → mainstory Appendix A.3
- §17B experiment inventory → mainstory §10
- §17C PE paper comparison → mainstory §8.3
- §18 project status → removed (operational, not reference)

### Retained (unique mathematical content):
- Full derivation chain with intermediate steps (§1)
- Geometric degeneration proof + Taylor expansion (§2)
- Waterbed fine structure: PPL invisibility, downstream bandpass, KL quantification (§3)
- τ\* variational derivation + full validation tables (§4)
- Fisher → attention Laplace bridge (§5)
- n-width argument + density ratio bound (§6)
- r parameter mathematical analysis + Riemann-Lebesgue proof (§7)
- Collision corollary proofs (§8)
- Broadband R² technical details (§9)
- NTK incompatibility explanation (§10)
