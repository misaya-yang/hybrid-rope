# Core theory questions for a stronger model — EVQ-Cosh (2026-07-20)

Purpose: self-contained mathematical questions distilled from an independent
adversarial audit. Each is a RESIDUAL open problem — not already closed by the
existing self-audits (`FULL_PAPER_INTEGRITY_AUDIT_20260713.md`,
`THEORY_REBUTTAL_MATHEMATICAL_AUDIT_20260711.md`,
`EXPERIMENT_THEORY_REVIEW_20260720.md`). Ordered roughly by depth/leverage.
Notation: φ∈[0,1] log-frequency, ω(φ)=b^(-φ), K=d_rot/2 channels, density
ρ≥0, ∫ρ=1; surrogate C_app[ρ]=α/2∫ρ² + β/2∬ρ(φ)ρ(ψ)min(φ,ψ)dφdψ;
cosh minimizer ρ_τ=τ·cosh(τ(1−φ))/sinh τ, τ=√(β/α); deployed rule τ*=d_eff/√L.

---

## Q1 — Is the exact theorem a tautology? (content of "cosh minimizes C_app")

min(φ,ψ) is the Green kernel of −∂² on [0,1]. Differentiating the EL equation
αρ+βg+ν=0 (g=∫ρmin, g''=−ρ) twice FORCES ρ''−(β/α)ρ=0, whose positive mass-1
solution is the cosh family for EVERY (α,β). So as (α,β) vary, the minimizer
family IS exactly the one-parameter cosh family — the ansatz and the optimizer
coincide by construction. The logical direction is cosh→surrogate (a tractable
quadratic form reverse-engineered to have a cosh minimizer).

**Question.** Does "cosh is the unique minimizer of C_app" have content beyond
the choice of C_app? Concretely: can the cosh allocation be derived as the
minimizer of an INDEPENDENTLY stated objective — the exact oscillatory collision
quadratic form ⟨ρ,Kρ⟩ with K(φ,ψ)=∫D(Δ)cos(ω₁Δ)cos(ω₂Δ)dΔ, or a stated
RoPE-attention / LM loss — rather than the surrogate built to yield it? If not,
the theorem should be restated as a closed-form representation / solvability
result, and ALL RoPE relevance must rest on an independent functional test (with
a non-cosh control and a held-out prior). State precisely what, if anything,
selects cosh from RoPE physics rather than from the surrogate's Green-kernel
structure.

---

## Q2 — The deployed cosh is the exact minimizer of NOTHING (τ_surr vs τ_deploy)

The paper's own fit gives α≈1/d_rot as a DISCRETIZATION artifact (kernel diagonal
~1/2 × channel spacing Δφ=1/K) and β∼L^(−0.22) as an O(1) power law, hence the
theorem's minimizer τ_surr=√(β/α)≈√d·L^(−0.11). Numerically at d=64, b=500K:
τ_surr≈6.2 (L=2048) and ≈5.7 (L=4096), versus deployed τ*=d/√L≈1.41 and 1.00 —
a factor 4.4×–5.7×, GROWING with L and maximal in the primary L=2048–4096 regime.
The densities are near-disjoint: concentration ratio ρ(0)/ρ(1)=cosh τ is ≈257:1
at τ_surr vs 2.18:1 deployed; ‖ρ_surr−ρ_deploy‖₁≈0.90–0.93. (The existing audit
P0.4 concedes the EXPONENT mismatch √d·L^(−0.11) vs d·L^(−1/2) but does not
compute the factor or the density distance, nor note that the deployed point is
far outside the surrogate's own optimum.)

**Question.** Is there ANY gauge-invariant / normalization-invariant convention
for (α,β) under which √(β/α) reproduces d·L^(−1/2) (not √d·L^(−0.11))? Because
K_app=αδ+βmin is invariant under joint rescaling of (α,β), only a RATIO is
physical — identify the observable that fixes it. If no such convention exists,
the exact tier and the deployed rule are unrelated and the theorem is a
family-SHAPE statement with zero claim that the deployed τ optimizes C_app.
Also: if the 24–92% "functional validation" table is evaluated at the mild
deployed τ (≈1.5, as in Fig.1), then it validates the DEPLOYMENT RULE, not the
theorem — state which τ the table uses.

---

## Q3 — The "structural d_head factor" is a normalization convention

From F=½S_χ²−λU with S_χ²∼τ⁴/d^a and U∼(d/L)^b·τ², stationarity gives
τ*∼d^((a+b)/2)·L^(−b/2). The proposition picks a=1 (d-normalized stiffness) and
b=1 (utility carries a d/L prefactor) → d¹·L^(−1/2). But a=1 is a convention:
the UNNORMALIZED stiffness (a=0) gives τ*∼d^(1/2) — matching the surrogate tier
(Q2) and CONTRADICTING the proposition's d¹. The L^(−1/2) exponent (b=1, from the
diffuse-softmax 1/L Jacobian) appears robust; the d-factor appears conventional.

**Question.** Which line of the derivation fixes a=1 and b=1 from physics rather
than convention? Is there a normalization-INDEPENDENT observable (e.g. a
dimensionless ratio of measured PPL curvatures, or a fitted exponent from a
controlled d-sweep at fixed L) that determines the d-exponent? If only L^(−1/2)
is convention-independent, restate the proposition as deriving ONLY the L-exponent
and label the d_head factor a convention.

---

## Q4 — The scaling balance REQUIRES U=O(τ²) linear, but U is labeled "KL" (which is O(τ⁴))

The only softmax KL computed is D_KL(p‖p_{εg})=½ε²gᵀJ_sm g+O(ε³), which at
ε=τ² (the EVQ displacement) is O(τ⁴) with zero first variation. Meanwhile
U=(d/L)[Q₀+τ²Q₁+…] is LINEAR in ρ_τ=1+τ²η+… — a probability-transport /
phase-variance first moment, not a KL. The balance τ*²=45λQ₁·d²/L REQUIRES
U=O(τ²): if U were the O(τ⁴) KL, both ½S_χ² and λU would be O(τ⁴) and
stationarity would fix a coefficient RATIO, not yield τ²=d²/L. (The audit T-02
concedes the order/mislabel error; what is NOT isolated is this structural
dependency — the derivation only works if the driver is the O(τ²) linear
functional, while its NAME ("KL gain") and its 1/L factor are imported from the
O(τ⁴) KL curvature.)

**Question.** Under the honest definition of U as a transport first moment (not a
KL): (a) what fixes the ABSOLUTE normalization of U (the exponent b in
U∼(d/L)^b) well enough to call the d-factor structural rather than conventional
(links to Q3)? (b) what is the correct physical reading of the 1/L factor once it
is NOT imported from the KL Taylor curvature? (c) rewrite the balance so the
O(τ²)-vs-O(τ⁴) statement is internally consistent, and state explicitly that
τ*∝d/√L survives ONLY because U is linear in ρ.

---

## Q5 — The collision mechanism is sign-indefinite (and "shaping reduces ∫w/ρ²" is false)

(a) E_off=Σ_{i<j}K_ij²/(K_ii K_jj) is the squared mutual coherence of the
position-feature Gram over ALL position pairs. Minimizing it yields a
near-orthogonal spreading code (good for worst-case position DISCRIMINATION).
But autoregressive attention needs a SIGNED, distance-resolved criterion: nearby
positions should COHERE (smooth relative-position structure / local attention)
and distant positions should decorrelate. min(φ,ψ) penalizes cumulative spectral
overlap uniformly with no notion of which off-diagonal coherences help vs hurt.

(b) The finite-channel mechanism "a shaped ρ reduces the weighted inverse-density
load ∫w/ρ²" is FALSE for the only weight obtainable without a task model, w≡1:
by Jensen with f(x)=1/x² convex, ∫1/ρ²dφ ≥ (∫1/ρ dφ)² ≥ 1 = ∫1·dφ, equality iff
ρ≡1. For the cosh density, ∫1/ρ²dφ = 1.004 (τ=0.5), 1.05 (τ=1), 1.59 (τ=2),
11.6 (τ=4) — shaping STRICTLY INCREASES the quantization distortion. The asserted
reduction requires w pre-concentrated where ρ is large, i.e. w chosen to MATCH ρ
(circular).

**Question.** (i) Derive the sign / distance weighting of the off-diagonal
penalty from the LM loss (or the exact RoPE attention), and determine whether the
LM-favorable allocation is the E_off MINIMIZER or requires preserving
near-diagonal coherence that E_off minimization destroys. (ii) Does there exist
any NON-CIRCULAR weight w derived from RoPE (e.g. the phase-variance q(Lb^(−φ))
of the transport proxy, or a stationary-phase importance) for which the cosh ρ
reduces ∫w/ρ² below the uniform density? If not, delete the "reduces ∫w/ρ²"
mechanism and present the finite-channel bounds purely as self-approximation
statements.

---

## Q6 — Three inconsistent "exact kernels"; which does cosh minimize / validate against?

The paper uses three different content-free kernels under a uniform distance
prior: (i) K=∫D(Δ)cos(ω₁Δ)cos(ω₂Δ)dΔ (cosine-PRODUCT, no sum-frequency term, no
content weights) — used for "validation"; (ii)
K=(1/2L)[sin((ω₁−ω₂)L)/(ω₁−ω₂)+sin((ω₁+ω₂)L)/(ω₁+ω₂)] — KEEPS the
cos((ω₁+ω₂)Δ) sum term — used to FIT α,β; (iii) the true RoPE Gram
z_ij=Re Σ_k α_{ij,k}e^{ir_ijω_k}, content-weighted by
α_{ij,k}=q_{i,k}conj(k_{j,k}). The cosh shape is FIT from (ii) (with sum term)
but VALIDATED against (i) (without); neither carries the content weights that the
paper's own Fisher coefficient depends on.

**Question.** Reconcile the fit kernel (ii) and the validation kernel (i). Then:
does the cosh minimizer survive when K is the content-weighted, causal-distance
RoPE Gram (iii) rather than the uniform-D cos-only kernel? Do the content weights
|α_{ij,k}|² factor out of the allocation problem, and if not, what allocation do
they imply?

---

## Q7 — Fisher forcing is the ONLY model-dependent term, and it is dropped

The forcing γb^(−2φ) carries η_F(φ_k)=(1/2s_att²)E[w_{ℓhij}|α_{ℓhij,k}|²] — the
activation-conditioned Fisher coefficient, i.e. the ONLY place the trained
model's actual attention statistics enter the allocation. The deployed
homogeneous pure-tether branch depends only on τ=√(β/α) fit to kernel geometry,
hence is entirely model-free / universal. So the deployed "exact minimizer" is
independent of the model it is deployed in PRECISELY because the one
model-dependent term is discarded — the Fisher utility that would tie EVQ to
trained attention enters the theory and is immediately removed.

**Question.** (a) On a checkpoint, measure η_F (hence λ_F·η̄_F/α) and bound the
forced-branch warp correction and residual R_F at the DEPLOYED τ (the paper's own
protocol). The "controlled residual / O(1/log b)" argument uses an L¹/CDF bound
amplified by sinh τ/τ in the inverse-CDF — at τ=4 this amplification is ≈13.6, so
"mass-small" is not "pointwise-small"; quantify the true residual. (b) If the
forcing is genuinely negligible at deployed τ, reposition the method as a
model-free closed-form allocation and demote Fisher forcing from a justification
to a discarded refinement.

---

## Q8 — The 24–92% "functional validation" has no discriminating independence

α,β are fit to the exact kernel K on the same discrete grid (diagonal matched by
construction via α=mean(K_ii)·Δφ; off-diagonal β by least squares against
min(φ_i,φ_j)), so the surrogate's quadratic form approximates K on those very
configs by construction. The validation then minimizes this FIT surrogate and
reports collision reduction against the SAME K on the SAME 12 configs — no
held-out distance prior D, no out-of-sample (L,b,K), undisclosed τ. The measured
collision score C=Σ_{i<j}K_ij²/(K_ii K_jj) is a NONLINEAR function of K, distinct
from the quadratic form ⟨ρ,K_app ρ⟩ the surrogate minimizes. A collision-only
oracle search reaches comparable (~65%) reduction.

**Question.** Does the cosh allocation beat (a) a generic monotone reallocation
and (b) the surrogate's OWN minimizer at τ_surr, on a HELD-OUT distance prior and
out-of-sample (L,b,K), by more than the collision-only oracle (with a global, not
local, optimizer)? Without this, state precisely what the 24–92% establishes
beyond "a fit surrogate transfers on its training configs."

---

### How to read these
- Q1, Q2, Q4, Q5 are the deepest: they ask whether the exact tier has any
 RoPE-specific content at all, and whether the deployed rule is connected to it.
- Q2+Q3 together determine whether "τ=d/√L is theory-motivated" survives in any
 form stronger than "a flat empirical basin."
- Q5, Q6, Q8 are the mechanism/bridge questions a theory reviewer will press.
- A defensible rebuttal needs a crisp answer to at least Q2 (is the gap a
 convention or a disconnection?) and Q4 (is the balance internally consistent
 once U is honestly defined?), since those two underwrite the conditional tier.
