# Auditor 3/6 — Bessel-defense subsection verification

**Scope.** Independent symbolic re-derivation of every math claim in the
new \subsection{Why constant α (...)}\label{sec:why-constant-alpha} of
`paper/appendix/a1_proofs.tex` (commit a358d6d / 2b33a59 — current `main`).
All work performed with sympy 1.14 plus scipy 1.17 in
`/Users/misaya.yanghejazfs.com.au/neurIPS-2026/hybrid-rope/audit_v2/scripts/verify_bessel_substitution.py`.
No reuse of paper-side scripts.

**Severity tags:** P0 = math error in the published paper / hand-wave a
reviewer would attack; P1 = wording divergence between user's paraphrased
summary and paper text / step works with caveats; P2 = stylistic.

---

## Step A. Verbatim extract of the new subsection (lines 99–108 of `a1_proofs.tex`)

The new subsection is a single dense paragraph (no `\paragraph` breaks)
plus one numbered display equation `(eq:const-vs-sp)`.  Verbatim:

> **Section heading**
> `\subsection{Why constant \texorpdfstring{$\alpha$}{alpha} (and not the continuum stationary-phase coefficient)}`
> `\label{sec:why-constant-alpha}`
>
> **Body, line 102:**
> "A natural objection asks why C_app uses a constant diagonal α δ(φ−ψ)
> rather than the φ-dependent coefficient implied by stationary-phase
> asymptotics of the exact kernel. With **λ := ln b**, **D₀ := D(0)**,
> near ψ=φ a stationary-phase expansion of K(φ_1,φ_2) = ∫ D(Δ)
> cos(ω_1 Δ) cos(ω_2 Δ) dΔ gives leading diagonal coefficient
> α_sp(φ) = π D₀ / (L λ b^{−φ}) = (π D₀ / Lλ) b^φ, which grows
> exponentially in φ. The difference is not a higher-order correction:"
>
> **Eq. const-vs-sp (line 103-107):**
> ∫₀¹ ρ²(φ) [α_K − α_sp(φ)] dφ
>   = ‖ρ‖₂² / (2K) − (π D₀ / Lλ) ∫₀¹ ρ²(φ) b^φ dφ.
>
> **Body, line 108:**
> "...which vanishes in the joint limit K→∞ AND L ln b / b → ∞, but is
> O(K⁻¹ + (L ln b)⁻¹·b) otherwise. Replacing α by α_sp(φ) converts the
> homogeneous Euler-Lagrange ODE ρ'' − τ² ρ = 0 into the
> variable-coefficient equation
>     **−[ α_sp(φ) ρ ]'' + β ρ = 0**,
> whose closed-form solution is a modified-Bessel combination
>     **ρ(φ) ∝ e^{−λφ} [ A I_0(x(φ)) + B K_0(x(φ)) ]**
> with **x(φ) = (2r/λ) e^{−λφ/2}**, **r = √(β L ln b / (π D₀))**.
> This Bessel density has heavier right tail than cosh and, in some
> parameter regimes (x_0 → ∞, x_1 → 0), violates positivity at φ=1
> unless an active inequality constraint is added.
> We deliberately use constant α for three reasons.
> (i) Closed-form CDF: the cosh density admits the elementary inverse-CDF
>     φ_k(τ) = 1 − τ⁻¹ arsinh((1 − u_k) sinh τ), which is the prerequisite
>     for inverse-CDF channel quantization; the Bessel density does not.
> (ii) Manifest positivity: ρ_τ ≥ τ/sinh τ > 0 on [0,1] for all τ > 0,
>     while the variable-α Bessel solution can violate ρ > 0 at the
>     right endpoint.
> (iii) Discrete-grid fit, not continuum asymptotic: paper-deployed
>     α ≈ 1/d_rot is fitted to K on the discrete K ∈ {16, 32} grid
>     (§sec:tau-scaling), and the resulting cosh allocation is
>     functionally validated against the exact kernel by the 24-92%
>     collision reduction across 12 configurations
>     (§sec:surrogate-validation). C_app is therefore a deliberately
>     discrete surrogate, not a claim about continuum-limit
>     asymptotics; (eq:const-vs-sp) is the precise statement of how
>     the two diverge."

### Deltas vs. user's paraphrased summary

| User said | Paper actually has | Tag |
|:---|:---|:---|
| "α_K = 1/(2K)" | Paper writes "constant diagonal α δ(φ−ψ)" with no explicit α_K = 1/(2K) inside this subsection — the value 1/(2K) is established earlier in `sec:tau-scaling` (a1:289). | P2 (the identity (eq:const-vs-sp) implicitly uses α_K = 1/(2K) by the appearance of the ‖ρ‖²/(2K) term on RHS, which is consistent with α_K = 1/(2K).) |
| `r` related to "√(β/γ)" | Paper: `r = √(βL ln b / (πD₀))`. Equivalent because γ := πD₀/(Lλ_paper) = πD₀/(L ln b), so β/γ = β L ln b/(πD₀). | OK — equivalent. |
| BC system "A·P(x_1) + B·R(x_1) = 0; A·P(x_0) + B·R(x_0) = γτ²/λ" with P, R defined | **The paper does NOT contain this BC system anywhere.** The subsection mentions BCs only implicitly ("violates positivity at φ=1") and the user's stated form is from outside the paper. | **P1** — this BC system was paraphrased by the user but is not in the paper. |
| λ in this section is the §3.7 transport multiplier | **`λ := ln b`** — explicitly redefined inside the subsection. Different symbol from §3.7 transport multiplier λ. | **P1** wording risk — same symbol two meanings, but disclosed in the first sentence of the subsection. |

---

## Step B. CLAIM 1 — const-vs-sp identity

**Claim.** ∫₀¹ ρ²(φ)[α_K − α_sp(φ)] dφ = ‖ρ‖² / (2K) − (πD₀ / Lλ) ∫₀¹ ρ² · b^φ dφ.

**Independent re-derivation (sympy):**

```
LHS = ∫₀¹ ρ²(φ)·[1/(2K) − (πD₀/Lλ) b^φ] dφ
    = (1/(2K)) ∫₀¹ ρ² dφ − (πD₀/Lλ) ∫₀¹ ρ²·b^φ dφ        [linearity of ∫]
    = ‖ρ‖² / (2K) − (πD₀/Lλ) ∫₀¹ ρ² · b^φ dφ              [‖ρ‖² := ∫ρ²]
```

Sympy `simplify(LHS_part1 − RHS_part1)` and `simplify(LHS_part2 − RHS_part2)`
both return 0.

**VERDICT: PASS (P0=0, P1=0).** Identity is the trivial linearity-of-integral
factoring of α_K = 1/(2K) (a constant) out of the integral. No hidden
algebraic content. The substantive content is in the *interpretation*
(the two terms scale separately as K⁻¹ and (L ln b)⁻¹·b respectively),
which the paper states correctly via the joint-limit comment.

---

## Step C. CLAIM 2 — Variable-α ODE → modified Bessel equation

**Claim.** With α_sp(φ) = γ exp(λφ) (where γ = πD₀/(Lλ_paper) and
λ_paper = ln b), the substitution

  ρ(φ) = e^{−λφ} · y(x(φ)),       x(φ) = (2r/λ) e^{−λφ/2}

reduces the ODE −[α_sp(φ) ρ]'' + β ρ = 0 to the modified Bessel equation

  x² y''(x) + x y'(x) − x² y(x) = 0,

with r = √(β/γ) (= √(β L ln b/(π D₀)) when γ = πD₀/(L ln b), matching
the paper's r).

**Independent derivation (sympy + analytic chain rule):**

1. **Cancellation in α_sp · ρ:**
   H(φ) := α_sp(φ) · ρ(φ) = γ e^{λφ} · e^{−λφ} y(x) = γ y(x(φ)).
   ⇒ The first thing that happens is the e^{λφ} (in α_sp) and e^{−λφ} (in ρ)
   cancel, so H(φ) is just γ y(x(φ)).

2. **Compute d²H/dφ² via chain rule.**
   dx/dφ = −(λ/2) (2r/λ) e^{−λφ/2} = −r e^{−λφ/2} = −(λ/2) x.
   d²x/dφ² = (λ/2)·(λ/2)·x = (λ²/4) x.
   So d²(γ y(x(φ)))/dφ² = γ [ y''(x) (dx/dφ)² + y'(x) d²x/dφ² ]
                        = γ [ y''(x) (λ²/4) x² + y'(x) (λ²/4) x ]
                        = γ (λ²/4) [ x² y''(x) + x y'(x) ].

3. **Compute β·ρ in x.**
   x = (2r/λ) e^{−λφ/2} ⇒ e^{−λφ/2} = (λ/(2r)) x ⇒ e^{−λφ} = (λ/(2r))² x²
                                                               = λ² x² / (4 r²).
   So β ρ = β e^{−λφ} y(x) = β λ² x² / (4 r²) · y(x).

4. **Assemble the ODE.**
   Original: −H'' + β ρ = 0  ⇒
     −γ (λ²/4) [ x² y'' + x y' ] + β λ²/(4 r²) · x² y = 0.
   Divide by γ (λ²/4):
     −[ x² y'' + x y' ] + (β/(γ r²)) x² y = 0.
   Multiply by −1:
     x² y'' + x y' − (β/(γ r²)) x² y = 0.
   Setting **β/(γ r²) = 1 ⇒ r² = β/γ ⇒ r = √(β/γ)** gives the *exact*
   modified Bessel equation x² y'' + x y' − x² y = 0.

5. **Cross-check r = √(β L ln b / (πD₀)).**
   Paper's α_sp(φ) = (πD₀/Lλ_paper) b^φ.  Identifying the user's
   exponential form γ e^{λφ} = γ exp(λ·φ) with the paper's
   (πD₀/Lλ_paper) b^φ = (πD₀/Lλ_paper) e^{(ln b) φ}, we get
   λ ≡ ln b = λ_paper, and γ = πD₀/(Lλ_paper) = πD₀/(L ln b).
   So β/γ = β L ln b/(πD₀), and r = √(β L ln b/(πD₀)) — matches the
   paper's r exactly.

6. **Numerical sanity:** I evaluated H'' for y = I_0 at φ = 0.3,
   λ = 0.7, γ = 1.3, r = 2 by sympy auto-diff and by the closed-form
   formula γ(λ²/4)(x²y''+xy') with y = I_0, y' = I_1, y'' = I_0 − I_1/x.
   Result: both are 130.6943, exact to printed precision.  Verified the
   modified Bessel equation x²y''+xy'-x²y=0 holds numerically for
   y = I_0 at x = 1.7 (residual 8.9e-16). ✅

**VERDICT: PASS (P0=0, P1=0).** The substitution genuinely reduces the
variable-α ODE to the modified Bessel equation, with r = √(β/γ) =
√(β L ln b/(πD₀)).  The paper's r matches exactly.  Independent of
sympy this is a textbook trick: Liouville-type substitution
ρ = e^{−λφ} y converts a Cauchy-Euler-times-exp into Bessel.

**Note (no paper bug):** the paper writes the substitution and the
final ρ(φ) = e^{−λφ}[A I_0 + B K_0] form *correctly* and gives the
correct r. The intermediate algebra is omitted (one would need to
include 4-5 lines to write it out) — that is a reasonable
omission for an appendix subsection focused on motivation, not on
re-deriving Bessel reduction.  See §F.

---

## Step D. CLAIM 3 — BC system

**User's claim.**
With ρ(φ) = e^{−λφ} [A I_0(x) + B K_0(x)], applying ρ'(0) = −τ², ρ'(1) = 0
and defining
  P(x) := I_0(x) + (x/2) I_1(x),   R(x) := K_0(x) − (x/2) K_1(x),
the BC system reduces to:
  A·P(x_1) + B·R(x_1) = 0
  A·P(x_0) + B·R(x_0) = γ τ² / λ.

**Independent derivation (sympy auto-diff plus modified-Bessel
identities I_0' = I_1, K_0' = −K_1):**

1. ρ(φ) = e^{−λφ} [A I_0(x(φ)) + B K_0(x(φ))].
2. d/dφ [I_0(x(φ))] = I_1(x) · dx/dφ = I_1(x) · (−(λ/2) x) = −(λ/2) x I_1(x).
   d/dφ [K_0(x(φ))] = −K_1(x) · dx/dφ = −K_1(x) · (−(λ/2) x) = (λ/2) x K_1(x).
3. ρ'(φ) = −λ e^{−λφ} [A I_0 + B K_0]
          + e^{−λφ} [ A · (−(λ/2) x I_1) + B · ((λ/2) x K_1) ]
        = −λ e^{−λφ} [ A (I_0 + (x/2) I_1) − B · 0 ]    ← no, redo:
   Let me redo carefully:
   = −λ e^{−λφ} [ A I_0 + B K_0 ]  + e^{−λφ}·(−(λ/2) x)·[A I_1 − B K_1]
   = e^{−λφ} [ −λ (A I_0 + B K_0) − (λ/2) x (A I_1 − B K_1) ]
   = −λ e^{−λφ} [ A (I_0 + (x/2) I_1) + B (K_0 − (x/2) K_1) ]
   = **−λ e^{−λφ} [ A · P(x) + B · R(x) ].**

4. Numerical sympy cross-check (4 random tuples (φ, A, B, λ, r))
   confirms identity ρ'(φ) − ( −λ e^{−λφ} [ A·P + B·R ] ) = 0 in
   sympy + numerical (all diffs 0.00e+00).  ✅ The pre-factors P(x) and
   R(x) are EXACTLY as the user states.

5. Apply BCs:
   - At φ = 0: x(0) = 2r/λ =: x_0, e^{0} = 1 ⇒
     ρ'(0) = −λ [A P(x_0) + B R(x_0)].
     If we set ρ'(0) = −τ² (the §3 BC, which the paper inherits), then
       A P(x_0) + B R(x_0) = **τ²/λ**.
   - At φ = 1: x(1) = (2r/λ) e^{−λ/2} =: x_1, e^{−λ} ≠ 0 ⇒
     ρ'(1) = −λ e^{−λ} [A P(x_1) + B R(x_1)] = 0 ⇒
       A P(x_1) + B R(x_1) = **0**.

**Discrepancy with user's claim.** User wrote
"A P(x_0) + B R(x_0) = γ τ²/λ" but my derivation gives just **τ²/λ**
(no γ).  Origin of the user's γ:
- *Either* the user is reusing α_sp(0) = γ e^0 = γ as a multiplier
  on ρ'(0). Indeed if one consistently re-derives the BC ρ'(0) from
  the variable-α Euler-Lagrange (rather than inheriting the §3
  constant-α BC verbatim), one gets α_sp(0) ρ'(0) = −β ⇒
  γ ρ'(0) = −β ⇒ ρ'(0) = −β/γ.  Then with τ̃² := β/γ,
  ρ'(0) = −τ̃² and the system gives A P(x_0) + B R(x_0) = τ̃²/λ
  = β/(γλ).  Multiplying both sides by γ gives
  γ·(A P + B R)|x_0 = β/λ — but the user's claim has γ on the
  RHS only, not LHS, which doesn't follow from any single consistent
  derivation.
- *Or* the user mixed conventions: kept τ from §3 (where τ² := β/α_K
  with α_K = 1/(2K) constant, so β = τ² α_K = τ²/(2K)) and tried to
  rewrite β/(γλ) in terms of τ.  In that case the RHS would be
  β/(γλ) = τ²·α_K/(γλ) = τ²/(2K γ λ) — *not* γτ²/λ.

Either way, **the user's RHS γτ²/λ is dimensionally inconsistent
with both natural derivations**, off by a factor that depends on
how the §3 and §sec:why-constant-alpha conventions are reconciled.

**Critical fact:** The paper does NOT contain a BC system. The user's
paraphrased BC system was generated by the user (or a downstream
agent), not extracted from the paper. So this discrepancy is **not a
P0 paper bug** — but the user should not write it down in any future
revision without reconciling it.  The two-row BC system if one wants
to add it would be:

  A · [I_0(x_1) + (x_1/2) I_1(x_1)] + B · [K_0(x_1) − (x_1/2) K_1(x_1)] = 0,
  A · [I_0(x_0) + (x_0/2) I_1(x_0)] + B · [K_0(x_0) − (x_0/2) K_1(x_0)] = τ²/λ,

with x_0 = 2r/λ, x_1 = x_0 e^{−λ/2}, r = √(β L ln b/(πD₀)).

**VERDICT: PARTIAL.**
- Pre-factors P(x) = I_0(x) + (x/2) I_1(x), R(x) = K_0(x) − (x/2) K_1(x):
  **PASS** (sympy + numerical exact agreement).
- BC at φ=1 "A P(x_1) + B R(x_1) = 0": **PASS**.
- BC at φ=0: my derivation gives "A P(x_0) + B R(x_0) = τ²/λ", user says
  "γτ²/λ".  **User's RHS is wrong by a factor of γ relative to the most
  natural derivation.** Tagged P1 because **paper does not actually
  publish this BC system**, so it does not invalidate any paper claim.
  Severity P1, not P0, because no paper text needs fixing — just the
  user's mental note for future revisions.

---

## Step E. CLAIM 4 — Three reasons for choosing cosh

### 4(i) — Closed-form CDF of cosh density

**Claim.** F_τ(φ) = 1 − sinh(τ(1−φ))/sinh τ; F_τ' = ρ_τ(φ); F_τ(0) = 0,
F_τ(1) = 1; F_τ⁻¹(u) = 1 − τ⁻¹ arsinh((1−u) sinh τ).

**Sympy:**
```
dF/dφ = τ cosh(τ(1−φ))/sinh τ  ✓ (identical to ρ_τ; sympy diff = 0)
F(0) = 0  ✓
F(1) = 1  ✓
F(F⁻¹(u)) − u = 0  ✓ (sympy simplify returns 0)
```

**VERDICT: PASS.**

### 4(ii) — Positivity ρ_τ(φ) ≥ τ/sinh τ > 0

**Claim.** ρ_τ ≥ τ/sinh τ > 0 on [0,1] for τ > 0.

**Sympy + analytic:**
```
ρ_τ'(φ) = −τ² sinh(τ(1−φ))/sinh τ.
For τ > 0 and φ ∈ [0,1):  τ(1−φ) ∈ (0, τ], so sinh(τ(1−φ)) > 0.
⇒ ρ_τ'(φ) < 0 on (0,1),  ρ_τ' = 0 at φ = 1.
⇒ ρ_τ is strictly decreasing on [0,1], minimum at φ = 1.
ρ_τ(1) = τ cosh(0)/sinh τ = τ/sinh τ.
For τ > 0: sinh τ > 0, so τ/sinh τ > 0.
```

**VERDICT: PASS.**

### 4(iii) — 24-92% collision reduction across 12 configs

**Test.** Read `tab:surrogate-validation` from `a1_proofs.tex` lines
117-148 and extract every `($-X\%$)` cell.

**Result:**
```
Extracted: ['92', '86', '73', '56', '38', '24', '45', '41', '37', '34', '28', '26']
Count = 12 (matches "across 12 configurations" claim)
Min = 24%, Max = 92%
```

**VERDICT: PASS.**  Range exactly 24-92%; the paper text "24-92%
collision reduction across 12 configurations" matches the table to
the digit.  Of the 12 rows: 6 are sweep-L (text K=32 b=500K); 2 are
sweep-b (text K=32 L=2048); 4 are sweep-b (video K=16 L=32). All 12
have Δ_C strictly negative ranging −24% to −92%.

---

## Step F. Hand-wave audit of the new subsection

The subsection is one paragraph, packing ~7 distinct math claims into
a single block.  I list each with a hand-wave assessment.

| # | Claim in §sec:why-constant-alpha | Status | Sev |
|:---:|:---|:---|:---:|
| F0 | "stationary-phase expansion of K(φ_1,φ_2) gives leading diagonal coefficient α_sp(φ) = πD₀/(Lλ b^{−φ})" | NOT PROVED in section. Reader expected to know stationary-phase = πD₀/(L ω) and ω(φ) = b^{−φ}, ω = ω̇₀ φ. (Standard but not cited.) | P1 |
| F1 | "joint limit K→∞ AND L ln b/b → ∞" makes (eq:const-vs-sp) vanish | OK by inspection of (eq:const-vs-sp): each term scales as K⁻¹ and (L ln b)⁻¹·b respectively. **Verified.** | OK |
| F2 | "is O(K⁻¹ + (L ln b)⁻¹·b) otherwise" | OK because b/(L ln b) → 0 only when L ln b → ∞ at rate ≥ b. So in practical regimes (L ~ 4096, b ~ 500K), L ln b ~ 5.4·10⁴ vs b ~ 5·10⁵, ratio ~ 0.1, **NOT vanishing** ⇒ correction term not actually small. **The paper acknowledges this implicitly** ("vanishes in the joint limit ... but is O(...) otherwise"). Honest disclosure — not a hand-wave per se, but a reviewer might press: "in the deployed regime b=500K, L=4096, what is the actual size of (eq:const-vs-sp)?" | **P1**: a one-row numerical display would be safer. |
| F3 | "−[α_sp ρ]'' + β ρ = 0 ... whose closed-form solution is e^{−λφ}[A I_0(x) + B K_0(x)] ... r = √(βL ln b/(πD₀))" | The substitution + reduction is *correct* (verified Step C) but the paper does not show the 4-5 lines of algebra. Reviewer-acceptable for an appendix cite-block, but a **PARANOID reader will ask for the steps**. | **P1**: add a 2-line `\paragraph{Bessel reduction.}` showing the chain-rule derivation for full safety. |
| F4 | "in some parameter regimes (x_0 → ∞, x_1 → 0), violates positivity at φ=1 unless an active inequality constraint is added" | I numerically tested 5 (λ, r, τ²) tuples covering the (small x_1) regime; in **all 5**, ρ(1) emerged ≈ −10⁻⁶ to −10⁻⁴ (strictly negative). **Confirmed.** This is a real positivity violation, fully consistent with the paper's claim. (The mechanism: K_0(x_1) → +∞ as x_1 → 0, but the BC forces a small *negative* B coefficient; with K_1(x_1)/x_1 → 1/x_1², R(x_1) → −∞ which dominates and forces ρ(1) < 0.) | **OK**, supported. |
| F5 | "(i) Closed-form CDF: φ_k(τ) = 1 − τ⁻¹ arsinh((1 − u_k) sinh τ)" | Verified in Step E(i). | **PASS** |
| F6 | "(ii) Manifest positivity: ρ_τ ≥ τ/sinh τ > 0 on [0,1] for τ > 0" | Verified in Step E(ii). | **PASS** |
| F7 | "(iii) ... 24-92% collision reduction across 12 configurations" | Verified in Step E(iii). | **PASS** |

**Summary of hand-waves found:**

- **F0** (P1): "Stationary-phase expansion of K(φ_1,φ_2) gives α_sp(φ) =
  πD₀/(Lλ b^{−φ})" — paper asserts but does not derive. A reviewer
  unfamiliar with stationary-phase asymptotics for kernel pairs will
  not be able to verify the πD₀/L factor or the (Lλ)⁻¹ scaling.
  Recommend: add a sentence "(see van Kampen 1957 / standard result for
  ∫f(Δ) cos(ωΔ) dΔ ≈ πf(0)/ω as ω→∞, applied at ω = Lω̇₀ = L ln b · b^{−φ})".
- **F2** (P1): "O(K⁻¹ + (L ln b)⁻¹·b) otherwise" — claim is correct but
  the *bigness* of the second term in deployed regimes (b/(L ln b) ~ 9
  for L=4096, b=500K) is not numerically disclosed. A reviewer could
  argue "the (eq:const-vs-sp) deviation is therefore *huge* in your
  deployed regime — why does cosh still work?" Paper's answer is
  reason (iii) (discrete grid + functional validation), but this is
  somewhat *post hoc*. Recommend: a single sentence noting
  "in deployed regimes (b=500K, L=4096), b/(L ln b) ≈ 9, so the
  stationary-phase form would *severely* deviate from the discrete
  α≈1/d_rot fit; the discrete-grid α is the right object."
- **F3** (P1): The Bessel reduction (4 lines of chain-rule algebra) is
  declared but not shown. Recommend: insert a single equation showing
  the substitution step.

**No P0 hand-waves.** Every math claim verifies; the section's
claims, as a whole, are correct. The hand-waves are about *exposition*,
not about *truth*.

---

## Final summary

| Claim | Verdict | Severity |
|:---|:---:|:---:|
| 1 — const-vs-sp identity (linearity of ∫) | **PASS** | OK |
| 2 — Variable-α ODE → modified Bessel, r = √(β/γ) | **PASS** | OK (paper-internal correct) |
| 3 — BC system in (A, B) with rows P, R | **PARTIAL** | P1 (user's RHS off by γ; paper does not contain BC system, no paper bug) |
| 4(i) — Closed-form invertible CDF | **PASS** | OK |
| 4(ii) — Positivity ρ_τ ≥ τ/sinh τ | **PASS** | OK |
| 4(iii) — Tab:surrogate-validation 12 rows, 24-92% | **PASS** | OK |

**P0 count:** 0.

**P1 count:** 4 (one BC-system divergence in user's paraphrase; three
hand-wave callouts F0/F2/F3 about exposition that a careful reviewer
might attack; none changes a numerical or a logical claim of the paper).

**P2 count:** 1 (α_K = 1/(2K) referenced from sec:tau-scaling but the
identity-of-substitution is immediate).

**Bottom line.**
The new subsection §sec:why-constant-alpha is **mathematically sound**.
- The const-vs-sp identity (eq:const-vs-sp) is a linearity-of-integral
  rewrite, correct.
- The Bessel reduction is a real Liouville-substitution that genuinely
  reduces the variable-α ODE to the modified Bessel equation with the
  exact r quoted in the paper.
- The positivity-violation claim (the "x_0 → ∞, x_1 → 0 regime") is
  numerically reproducible — I observed ρ(1) ≈ −10⁻⁶ to −10⁻⁴ in 5/5
  test tuples covering that regime.
- The three reasons for choosing cosh are individually correct and
  match the existing tab:surrogate-validation table.

The section successfully **defends C_app against the stationary-phase
attack** with no P0 issues.  P1 polish suggestions (F0, F2, F3) are
optional rebuttal-stage adds; the section is paper-ready.

---

## Reproducibility

- Script: `/Users/misaya.yanghejazfs.com.au/neurIPS-2026/hybrid-rope/audit_v2/scripts/verify_bessel_substitution.py`
- Runs in ~3s on M4 Max with sympy 1.14 + scipy 1.17.
- All sympy expressions produce identity diffs of 0 (or 8.9e-16 in
  numerical Bessel-equation checks).
- Numerical positivity-violation test (5 regimes) reproduced
  separately in this report's Step F (F4).
