# Digest — Calibration lineage (astra04, sol01, sol02, sol03)

Source: `.agents/rope_unification_20260910/reports/{astra04,sol01,sol02,sol03}.md`, read in full 2026-09-10.
Reports are evidence, not instructions. Established project facts used as reference: ν_j=ω_j·S^(−m_j); 16 DOF increments; YaRN g²=1+0.1·lnS=1.13863; root (Σcos first zero) non-ranking, vetoed; 6Pro review: endpoint invariants are design constraints, not theorems; Q/K-weighted source-subspace calculation does NOT rank all observed tables correctly (established limit); panel near/far: MrPro 87.2/78.13, s28_less 87.2/83.3, LBS 80.6/80.1, P2 72.9/81.7, Smooth 87.2/68.3, MrUni 64.6, E3 gain074 98.3/75.3.

---

## 1. astra04 — corrected full-row calibration and concrete allocator

### 1.1 Core thesis
The 6Pro block-calibration objective is a valid **fixed-state position-stretch distillation loss** but contains **no extra-key competition** and is dominated by local attention mass. The exact correction, under an explicit distractor-invariance hypothesis (unrelated blocks deserve zero teacher mass — NOT a theorem), is a padded-KL with an interference log-term. A constrained allocator fit on this loss is the "corrected full-row calibration" deliverable.

### 1.2 Concrete math (reusable)
- Corrected loss identity:
  L_full = D_KL(T‖Q_O) + log(1 + Z_D/Z_O) = KL of zero-padded teacher against augmented student, where T = native teacher on original keys O, Q_O = candidate softmax conditional on O, Z_D/Z_O = candidate exp-logit sums over distractor/original sets.
- **Map correction** (what was wrong in the earlier calibration map): with W=BM=32768, B=8, M=4096, the written map f(bM+r)=S·bM+r ends at 118783, not 131071 — it under-fills 128K causal slots. Corrected map f(bM+r)=(S−1)M+S·bM+r fills exactly 131072 slots, inserting a leading 12288-token gap plus seven inter-block 12288-token gaps = **98304 distractor slots**; common translation preserves every original query–key displacement. Caveat retained: correction does NOT turn stitched raw states into an actual 128K forward pass.
- Local-dominance decomposition: D_KL(T‖Q_O) = D_KL(p‖q) + Σ_b p_b·D_KL(T_b‖Q_b); a teacher with p_local=0.999 permits arbitrarily bad far-block preservation at negligible loss.
- Balanced answer-free objective:
  L_bal = D_KL(p‖q) + B⁻¹·Σ_b D_KL(T_b‖Q_b) + log(1+Z_D/Z_O); compute conditionals from block log-softmax (avoid underflow); report p_b, per-block conditional KL, and distractor mass separately.
- Identity-map guard: same block-balanced original-support loss at f(p)=p with D empty; require fitting aggregate no worse than MrPro's value (preserves an existing reference, no free short/long tradeoff weight).
- Exact gradient (finite-phase, retains cross-pair effects through summed logits + softmax):
  ∂z_k/∂x_j = σ·d′_k·ν_j·[C_jk·sin(d′_k ν_j) − D_jk·cos(d′_k ν_j)];
  balanced original-support terms: (q_b−p_b)Q_b(k) + B⁻¹(Q_b(k)−T_b(k));
  distractor penalty: Q_aug(k)−Q_O(k) on O, Q_aug(k) on D.

### 1.3 Tags
- Padded-KL identity, map arithmetic, 98304 count, gradient vs finite differences: [已验证-derivation] (CPU-checked, see 1.5).
- Distractor-invariance hypothesis (inserts should get zero teacher mass): [假设].
- Local-dominance failure mode of the unbalanced loss: [已验证-derivation] (exact decomposition) with [部分证据] practical relevance.
- Counterexample that penalty ≠ functional retention (donor with identical K necessarily competes; 4× copied K/V gives duplicate marginal = T with no justified penalty): [已验证-derivation] (algebraic counterexample).
- Fit-on-existing-data transferability claim: [假设] — falsifiable rule, explicitly not a task-success theorem.

### 1.4 Proposed rule — the "corrected full-row calibration" + constrained allocator (exact recipe)
Coordinates: x_j = −log(ν_j/ν_ref); fix x_0 and x_63 to MrPro endpoints; constraints x_{j+1}−x_j ≥ ε (ε = numerical order tolerance, NOT a frequency prior); for causal comparison additionally fix Σ_j x_j = Σ_j x_j^Mr. 61 free dimensions; no prescribed knees, signed per-slot labels, or geometry-roughness objective.
Data: existing captures — `experiments/nongeometric_screen/pro_block_calibration.py`: fit docs 0:4, validate docs 4:6, M=4096, S=4, 4 query heads but **only q=32767**, all raw K/V for two KV heads, stored states are native/common-gain states from the actual layer run. Distractor fill: for each fit doc, its own 32K Q with its own K on O; the 98304 gap slots filled with the other three fit docs' K (same layer, correct KV-head), internal order preserved, averaged over the three cyclic donor orders; validation docs never used as fit donors; no new GPU capture needed.
Procedure: start at MrPro; minimize mean fit L_bal (single declared block-stretch intervention) with the identity-map guard; average docs → layers → heads equally; stream full keys with logsumexp; solve with a constrained optimizer (SLSQP + analytic derivatives), not a one-shot Taylor step; freeze ONE final table at fitting-objective convergence; evaluate docs 4,5 once; no checkpoint search using task answers; if solver can't improve under constraints, keep MrPro and report the calibration gave no direction. Post-hoc controls: v = x_fit − x_Mr; ±v scaled by largest factor ≤ 1 preserving order (reverse control; both directions keep endpoints + cumulative compression). Positive/reverse/MrPro evaluated with identical gain on newly generated independent full-model long tasks.

### 1.5 CPU numerics
- Padded-KL vs KL+log1p identity: T=[.7,.2,.1], original logits [1,−.5,.2], donor logits [.4,−.7] → both sides 0.45529644748142606, difference 0.
- Analytic combined gradient vs centered finite differences on seeded 3-block/21-key/5-frequency fixture: max abs error 2.3480405640652346e-10 (validates the derivative, not the model-quality hypothesis).
- Map endpoints and 98304-slot count checked. No GPU jobs.

### 1.6 Obstructions
- Single query position (q=32767) per document → cannot identify behavior across query positions or separate query-local vs block-local effects.
- Stitched donor K generated under different prefixes/native RoPE → query/donor compatibility differs from real insertion; artificial doc-start/sink states repeated in gaps create sink competition absent from a real 128K document.
- Zero-padded teacher assumes inserts should be ignored (hypothesis, with counterexamples above); penalty tests original-key identity retention, stronger than functional retention.
- Native/common-gain calibration states ≠ operational Native gain=1; full-prefill candidate states must be evaluated separately.
- Objective-transfer warnings from ingested sources: output-KD NLL 3.94 vs score-fit 9.99 with score-fit having *smaller* position-response error yet both missing the frozen retrieval answer (`20260909_gpu/REPORT.md`); ROUND11 OLMO §9 downgrades mechanism-narrative claims; BM cross-cache JSON shows fixed-cache improvement ≠ generated-answer improvement; E16/G16/U16 appendix-B illustrates why the declared calibration prior must remain explicit.
- Nearly uniform low-mass block teachers may encode noise; balanced-loss improvement alone cannot establish useful far relations.

### 1.7 KKT relevance
The allocator is a bona fide constrained program (equalities: endpoints + sum; inequalities: monotone order) — KKT applies; astra04 solves numerically (SLSQP) rather than stating multipliers. Cross-pair coupling retained through full softmax logits — avoids the pair-additive assumption that would make per-slot KKT decoupling legitimate.

### 1.8 Conflicts with established facts
None direct. Notably astra04's calibration runs on **real attention logits**, so it sidesteps (does not contradict) the failed Q/K-weighted source-subspace ranking — but the single-query capture limit means it cannot be claimed to fix the general "calibration quantities must rank all tables" requirement; it declares itself a frozen-compatibility estimator, not a derivation of EVQ's spacing cost or a scratch optimum. Failure at dense-128K after stitched-row success falsifies transfer of this estimator only, not allocation DOF in general.

---

## 2. sol01 — what survives of MrRoPE as mathematics + EVQ-compatible allocation rule

### 2.1 Core thesis / verdict
**MrRoPE survives only as a cumulative frequency-allocation coordinate system** (ν_j=ω_j·S^(−m_j), m_j=Σ_{d<j} ε_d, λ_d=S^(ε_d), anchored endpoints, optional ε_d≥0). It does NOT supply an exact mixed-radix encoding theorem or a task-optimal allocation rule. MrPro's arithmetic increments are an assumption, not an optimizer output.

### 2.2 Concrete math (reusable)
- Exact mixed-radix: P_1=1, P_j=Π_{d<j} r_d, a_j(n)=⌊n/P_j⌋ mod r_j, n=Σ a_j P_j. RoPE phases φ_j(n)=(n/P_j) mod 2π drop the floor, use modulus 2π, noninteger radices ⇒ parameterization by adjacent cumulative ratios is algebra only; no carry/unique-decoding/no-collision-range transfer.
- Off-by-one: ν_j = ω_j/Π_{d=1}^{j−1} λ_d ⇒ **λ_{D_r} affects no frequency**; observable table determines only λ_1..λ_{D_r−1}. Fixed convention: transition frequencies j=l..h, m_l=0, m_h=1, N=h−l gap increments ε_q, Σ ε_i=1; slot h fully scaled; allocation variables live on gaps. (Matches the project's 16-DOF-on-increments fact.)
- MrPro exact closed forms: **ε_q^Pro = 2q/(N(N+1))**, **m_q^Pro = q(q+1)/(N(N+1))**; m_q^Uni − m_q^Pro = q(N−q)/(N(N+1)) > 0 for 0<q<N ⇒ Pro strictly delays all internal compression at equal endpoints — the only rigorous allocation principle, and it is comparative (vs MrUni), not optimality.
- YaRN "regressive" proof scope: A(r_j)=c+(S−1)r_j, c=β−Sα; direction needs (S−1)c ≥ 0 ⇒ holds at S=4, α=1, β=32; universal claim fails when c<0 (sign flip reverses the AM–GM term). Valid for target deployment, invalid as a theorem for all S.
- Log-frequency positions: φ_j(m) = −log_b ν_j = φ_j⁰ + (log S/log b)·m_j.

### 2.3 EVQ-compatible allocation rule (exact recipe)
Data needed: frozen calibration examples (native-window conditional logits per layer/head/query for D_0; frozen-model slot/source statistics including cross-slot structure for W).
- D_0(m) = E[(z(m;x) − z(0;x))²] over x,ℓ,h,n≤L_0, z = actual pre-softmax attention logit (measured native-window damage).
- C_T(m) = Σ_{a,b} W_ab · K_{L_T}(φ_a(m), φ_b(m)), K_L = exact finite-window EVQ kernel (closed form via cosine integrals, cf. sol02); **W ⪰ 0 must come from frozen-model slot/source statistics, not equal slot weights** (explicitly because cross terms in expected logit error are retained by the transition review).
- Rule: m* = argmin_m C_T(m) s.t. m_l=0, m_h=1, 0 ≤ m_{j+1}−m_j, D_0(m) ≤ δ, with δ chosen as the **measured native-window distortion of an accepted baseline (e.g., MrPro)** — a budget, not an invented tradeoff scalar. Returns one Pareto allocation; rejects candidates whose long-range collision gain costs more pretrained damage than the baseline's.
- Local form: g=∇C_T(m⁰), D_0 increment ≈ ½uᵀHu, endpoint-preserving tangent cone, trust region Δ: u* = −√(2Δ/(gᵀH⁻¹g))·H⁻¹g (add quadratic subproblem with active monotonicity faces). Marginal rule: spend extension where exact-EVQ benefit per unit measured native-logit damage is largest, including cross-slot couplings; per-slot independent argmin unjustified.
- Recommended next comparison (no grid): from deployed MrPro, estimate g + HVP-based H⁻¹g, generate ONE endpoint-preserving monotone candidate at MrPro's measured budget; pre-GPU reject if exact nonlinear recomputation loses predicted improvement or source-weighted benefit is dominated by unweighted proxy; else compare against MrPro on the paired 128K tasks.
- From-scratch is a different problem: (θ*,m*) = argmin E[L(f_{θ,m})] + γ·C_{p_target}(m); D_0 constraint meaningless (no pretrained m=0); scratch optimum may compress early channels more because projections adapt; frozen evidence does not establish scratch allocation.

### 2.4 Tags
- Mixed-radix vs RoPE distinction, λ_{D_r}-unobservable, off-by-one convention, ε/m closed forms, Pro-vs-Uni delay inequality, YaRN scope restriction, single-frequency and λ₂=S counterexamples, N=1 Pro=Uni: [已验证-derivation] (exact-fraction CPU checks: Σ_q ε_q=1 for N=2,3,17,18; ε₁=1/153, ε₁₇=1/9, Σ₁¹⁶ m_q=16/3 — matching `ROPE_MRPRO_BM_CONSTRUCTION_ANALYSIS_20260908.md`).
- "Delayed compression is desirable": design choice interpretation, [假设] as a principle (arithmetic sequence not derived from any loss).
- C_T-with-D_0-budget rule: [假设] — explicitly a candidate-generation rule, not a success theorem; full nonlinear recomputation + paired task eval required.

### 2.5 CPU numerics
Exact-fraction verification of the Pro construction (above) and source/hash verification only. No model jobs.

### 2.6 Obstructions
- BM counterexample blocks upgrading spectral smoothness to the missing theorem: BM lowers geometric roughness/distortion, moves every internal slot slower than MrPro, adds 50% to Σ internal m — yet loses 7.29 pts on the paired Qwen3B 128K six-task aggregate (heterogeneous task effects, prefix formation); conversely OLMo-BM strongly beats MrPro there. ⇒ neither smaller distortion, delayed compression, nor exponent mass is universal task utility.
- Softmax, V-mixing, layerwise prefix formation are outside the cosine kernel; cross-cache evidence says prefix formation matters.
- Falsifiable checks: exact-radix empty-product; index counterexample (λ₂=S scales slots 3,4 not "dimension 2"); progressive-limit checks; **EVQ proxy check** (if exact C_T improves but paired outcomes worsen, the collision objective lacks necessary source/task structure — do not rescue by claiming geometry success; Smooth/BM history makes this live); frozen-vs-scratch discriminator.

### 2.7 KKT relevance
Highest of the four: the rule is a KKT problem by construction (endpoint equalities, monotonicity inequalities, scalar budget inequality D_0≤δ, trust-region tangent cone, active-face QP). The marginal interpretation (benefit per unit damage equalized across slots incl. cross-slot coupling) is the KKT stationarity reading.

### 2.8 Conflicts with established facts
None; fully consistent with the panel record (Smooth 87.2/68.3 geometry-vs-far failure, BM loss, MrPro as canonical baseline) and with the 16-DOF increments fact. Its demand that W come from measured frozen statistics is the acknowledgment — not solution — of the established Q/K-weighted-subspace non-ranking limit: sol01's rule presumes a *good enough* W from real logit statistics, which is exactly what the failed subspace calculation could not deliver as a ranker.

---

## 3. sol02 — EVQ derivation, exact quantiles, finite-window correction

### 3.1 Core thesis
EVQ-Cosh has a clean exact theorem but **narrower than the phrase "exact EVQ"**: cosh is the unique minimizer of the *projected pure-tether* functional, not of the original finite-window collision functional unless the kernel projection is exact. At finite L the correct object retains the exact K_L — continuum quantile problem or constrained finite-table gap problem. Sharp slot transitions are over-penalized by EVQ's local δ-ridge replacement ⇒ density smoothness is not a theorem-backed deployment criterion.

### 3.2 Concrete math (reusable)
- Projected pure-tether functional: C_app[ρ] = (α/2)∫ρ² + (β/2)∬ρ(φ)ρ(ψ)min(φ,ψ)dφdψ, ρ≥0, ∫ρ=1; min kernel = Green kernel of −d²/dφ² (g(0)=0, g′(1)=0), PSD; strictly convex/coercive for α>0.
- Stationarity ⇒ ρ″ − τ²ρ = 0, τ=√(β/α), BCs ρ′(1)=0, ρ′(0)=−τ² ⇒ **ρ_τ(φ) = τ·cosh(τ(1−φ))/sinh τ**; positive ⇒ nonnegativity KKT constraint inactive; β=0 ⇒ uniform.
- Fisher-retained surrogate ODE: ρ″ − τ²ρ = (4μ_F c²/α)e^{−2cφ}, particular solution P·e^{−2cφ}, P = (4μ_F c²/α)/(4c²−τ²) (off resonance); pure cosh is exact only after removing the Fisher term; positivity there needs a KKT check.
- Exact finite-window kernel (c=log b, ω=e^{−cφ}, D_L(Δ)=1/(Δ log L) on [1,L]):
  K_L(φ,ψ) = [Ci(Lδ)−Ci(δ) + Ci(Lσ)−Ci(σ)]/(2 log L), δ=|ω−ν|, σ=ω+ν (φ≠ψ); diagonal via limit lim_{a→0}[Ci(La)−Ci(a)] = log L ⇒ K_L(φ,φ) = ½ + [Ci(2Lω)−Ci(2ω)]/(2 log L).
- Quantile law: F_τ(φ)=1−sinh(τ(1−φ))/sinh τ; **Q_τ(u) = 1 − (1/τ)asinh((1−u)sinh τ)**; Q_τ′(u)=1/ρ_τ(Q_τ(u)) strictly increasing ⇒ every equal-mass grid has gaps increasing toward low frequency. Exact gaps: g_k = (1/τ)[asinh((1−u_k)sinh τ) − asinh((1−u_{k+1})sinh τ)]. Small-τ: Q_τ(u) = u − u(1−u)(2−u)τ²/6 + O(τ⁴).
- Convention split: u_k=k/K anchors fastest endpoint only; **u_k=(k+½)/K (deployed practical/finite-transport table) anchors neither endpoint** ⇒ endpoint-anchoring propositions do NOT describe the deployed EVQ table; any MrRoPE comparison must freeze one convention (half-cell shifts).
- Continuum exact form: E_L[Q] = ½∬K_L(Q(u),Q(v))dudv − μ_F∫e^{−2cQ(u)}du over nondecreasing Q; δE/δQ(u) = ∫∂₁K_L(Q(u),Q(v))dv + 2cμ_F e^{−2cQ(u)}; **no cosh ODE exists for this exact functional**.
- Finite table: E_{L,K}(φ) = (1/2K²)ΣΣK_L(φ_i,φ_j) − (μ_F/K)Σe^{−2cφ_i}, 0≤φ₀≤…≤φ_{K−1}≤1; exact gap gradient ∂E/∂g_r = Σ_{m≥r}[(1/K²)Σ_j ∂₁K_L(φ_m,φ_j) + (2cμ_F/K)e^{−2cφ_m}] (cumulative force); ∂₁K_L(φ,ψ) = (c·ω(φ)/log L)∫₁ᴸ sin(ω(φ)t)cos(ω(ψ)t)dt (no Ci differentiation needed).
- Nonlocal ridge decomposition: K_L = [c·min(φ,ψ) − γ + h_c(φ−ψ)]/log L + E_L, h_c(x) = −½log(1−e^{−2c|x|}); multiplier ĥ_c(k) = (π/2|k|)coth(π|k|/2c) − c/k², ĥ_c(0)=π²/12c; positive and decreasing in |k| ⇒ delta replacement overcharges every nonconstant mode; first correction ⟨ρ,h_c*ρ⟩ = A₀‖ρ‖₂² − (π⁴/720c³)‖ρ′‖₂² + … is unbounded below at high wavenumber with endpoint terms ⇒ unsafe to optimize truncated; retain full K_L.

### 3.3 Tags
- All of 3.2 (cosh theorem, quantile/gap formulas, Ci kernel + diagonal, ridge multiplier, finite-table gradient): [已验证-derivation], with the exact scope statements.
- Transplant obstruction — position-independent invertible Q/K maps cannot exactly change a frozen model's rotary frequency multiset on an open interval: [已验证-derivation] (proved in appendix; decisive for frozen deployment).
- min kernel PSD, strict convexity, uniqueness, increasing gaps, τ→0 geometric limit, single crossing, self-consistency, finite transport bounds: [已验证-derivation].
- Cosine-only kernel as *the* collision cost (omits full sin/cos subspace, signed content coefficients, learned Q/K use, nonlinear attention): [假设] (modeling).
- τ ∝ d_head/√L scaling: [假设] inside stated small-τ/diffuse-attention/full-RoPE-MHA/positive-Q₁/fixed-λ model — a scaling form, not a universal optimum.
- Kernel-fit R², 99-run win counts, coefficient c=1: empirical only [部分证据].

### 3.4 Proposed rule
- Frozen deployment: **displacement-budgeted finite-window step**, min_φ E_{L_target,K}(φ) + ½Σ w_i(φ_i−φ_i⁰)², monotone gaps, hard-fixed slots where association evidence says critical; w_i measured from the frozen model (held-out loss curvature / controlled slot perturbations), NOT from cosine geometry. EVQ = broadband prior; MrRoPE-style selective preservation = displacement weights/hard constraints.
- Scratch: minimize the exact finite-window gap objective under one stated grid convention, then train as candidate; collision term is a prior only.
- Decision consequence: don't choose EVQ-vs-MrRoPE by smoothness or Cosh alone; evaluate candidates on exact E_{L,K} at training AND target windows, record per-gap cumulative forces, measure frozen slot sensitivity; downstream long-context task is the acceptance criterion.

### 3.5 CPU numerics
None reported beyond closed-form derivations (no numerical experiment logged).

### 3.6 Obstructions
- The projected-kernel approximation is "the decisive approximation behind Cosh", nonuniform at diagonal/endpoints; the deployed table uses the midpoint convention the anchoring theory doesn't describe; exact functional has no closed-form density solution (no ODE) — anything cosh-shaped is necessarily surrogate-derived; truncated high-wavenumber expansions unbounded below.

### 3.7 KKT relevance
Explicit and central: nonnegativity of ρ (inactive at the cosh optimum, needs a check in the Fisher-augmented case), monotone-gap and endpoint constraints with complementary slackness on the finite table, stationarity = cumulative force zero for interior gaps. sol02's E_{L,K} with gap gradient is the cleanest *exact differentiable objective* any of the four reports hands to a KKT allocator — but its weights/sensitivity (w_i) remain measured externals.

### 3.8 Conflicts with established facts
None. Consistent with "static collision ≠ extrapolation mechanism" (sol02 itself limits the kernel's claim scope), and with the full-row-subspace limit (cosine-only kernel explicitly acknowledged as incomplete).

---

## 4. sol03 — audit of the Pro first-principles proposals

### 4.1 Core thesis
The three Pro documents agree on the right distinctions but never finish the unification: EVQ-Cosh is an allocation rule for **training/co-adaptation under a declared surrogate**, not a mature-checkpoint transplant rule; MrRoPE is a **strong training-free non-uniform baseline + structural prior**, not derived from EVQ, geometry not shown to predict task utility. Correct unification = shared allocation framework with **lifecycle-specific evidence**, not one curve optimal in both regimes.

### 4.2 Surviving math (reusable)
- Normalized log-frequency coordinates x_k = −log ω_k = a + R·z_k, 0=z₀≤…≤z_{K−1}=1; fix sampled endpoints a, a+R, pair count K, operator, gain ⇒ isolated interior allocation (the clean causal object).
- **Exact radix bridge**: r_k = ω_k/ω_{k+1} = exp(x_{k+1}−x_k), Π r_k = e^R ⇒ mixed-radix schedule and EVQ allocation live on the SAME simplex {Δx_k ≥ 0, Σ Δx_k = R}: MrRoPE = factorization of the span into adjacent radices; EVQ = criterion for how much span each factor receives.
- Frozen allocation is signed/content-conditioned: ∂m/∂s_ij = α_ij·⟨∇_{o_i}m, v_j−o_i⟩ — more attention mass helps only if the written value points the right downstream way.
- Full sin/cos Gram and slow-collapse results remain valid within their positional surrogate (with the caveat that the strongest fixed-support experiment is not wholly in the ωL≪1 deep limit).

### 4.3 Audit verdicts — which Pro proposals died and why (VETOED)
1. **"Position-function redundancy ⇒ cheap frozen slots"** — REJECTED. K pairs sharing ω give positional-function span 2, but with Q=K=I_{2K} the content kernel diag(R(ωΔ),…,R(ωΔ)) has rank 2K ⇒ redundant positions may still carry independent content channels. Any frozen score charging zero cost for coincident positional columns fails this test. [vetoed]
2. **"A slow pair is suppressed by softmax"** — REJECTED. Softmax removes a rowwise constant, not a nearly distance-independent term varying with key content; leading slow-frequency qᵀk may be semantically decisive. [vetoed]
3. **"A static scalar gain resolves local/long phase conflict"** — REJECTED. Gain changes logit scale, not simultaneous phase constraints (no-wrap interval condition makes it explicit). [vetoed]
4. **"Lower geometric distortion ⇒ higher long utility"** — UNSUPPORTED, self-contradicted by the dossiers (stable rank/phase coverage/condition number/frequency movement cannot be renamed task utility); consistent with, and reinforced by, Smooth_MrBudget (pending raw-artifact verification) and the project's root veto. [vetoed as ranking principle]
5. **"The same analytic allocation applies scratch and frozen"** — UNSUPPORTED, contradicted by ordered-slot coupling and installation-history decomposition. [vetoed]
Surviving assumptions (1–4 in 4.2) tagged [已验证-derivation] (items 1–2 exact algebra; item 4 chain-rule identity).

### 4.4 Proposed rule — the "constrained allocator" (framework + KKT law)
Unified allocation problem (framework, not observability claim):
min_{u≥0, 1ᵀu=R} J_pos(u;μ) − γ·U(u;θ,D) + λ·D_N(u;θ), u_k = Δx_k = log r_k, subject to fixed endpoints, ordered slots, fixed operator/gain, declared min-gap. J_pos = declared EVQ positional surrogate (definition preserved, no invented proxy); U = task functional (paired content-fork/full-trajectory margins on dev instances).
**KKT interior-gap marginal-allocation law**: −∂_{u_k}J_pos + γ·∂_{u_k}U − λ·∂_{u_k}D_N = ν (constant across active gaps), complementary inequalities at u_k=0 — each unit of log-radix is assigned until net marginal value equalizes. This is "EVQ defines positional marginal value, MrRoPE supplies radix coordinates + span conservation."
- Scratch branch: λ=0, θ co-adapts, u^scratch = argmin J_EVQ(u; μ_train) then train θ (or joint (θ,u) with β·J_EVQ regularizer, eq. 4). Cosh retained exactly if it is that surrogate's minimizer; r_k* = e^{u_k*}.
- Frozen branch (Qwen2.5-3B 32K→128K, x = x⁰+δx): ONE trust-region candidate δx* = argmax_{C} g_Lᵀδx − (λ/2)δxᵀH_Nδx − (ρ/2)‖Bδx−d₀‖²; g_L = signed derivative of a predeclared long functional target; H_N = pullback Fisher or estimated native-output-KL quadratic; B maps slot frequency changes → adjacent log-radix changes; d₀ = MrRoPE/EVQ structured radix proposal for 4×; C = endpoint policy + order + trust radius + single global static table. Stationarity: (λH_N + ρBᵀB)δx = g_L + ρBᵀd₀; project in the same PD metric with inequalities.
- Fallback ladder (important honesty): no long-labeled dev data ⇒ g_L=0 ⇒ result is a **prior-constrained compatibility candidate**, NOT a predicted long optimum; no native calibration either ⇒ use external MrRoPE unchanged; optimum underdetermined.
- Data needed: one frozen dev set of legal paired compact/far relations + one native calibration set; four fixed arms (native, official MrRoPE, EVQ-prior-only, single function-calibrated solution); decision endpoint = native-gated strict paired long generation with full trajectory/EOS; geometry = diagnostic only. No candidate grid; P2 and E1-slot28 are priors only after tiny-sample conditionality + owner artifacts confirmed, must not create extra search arms.

### 4.5 CPU numerically checkable items
Permutation check (frequency-only permutation must change behavior; joint permutation of full rotary pairs + Q/K output coordinates + norm params must preserve logits — joint-parity failure is an implementation bug). Repeated-frequency rank-2-vs-rank-2K construction. Row-shift check: adding a constant to all key logits in a row leaves attention unchanged but changes raw-logit-MSE allocation loss ⇒ any functional/native cost must be row-shift invariant (this kills sol01's D₀ as literally written as an MSE — it is a damage *budget* on measured logits, and sol03's check shows raw MSE is the wrong functional; resolution: the invariance demand applies to the loss used inside the optimization). Phase-conflict feasibility check: 3|ω| ≤ 4η_N/d_N + η_L/d_L — if violated, no single frequency meets both exact phase targets; must trade off via other slots/margin. All four are scalar-arithmetic, CPU-checkable; no numerics were logged.

### 4.6 Evidence-status caveats (audit's own limits)
Files audited are analysis/plan documents, not primary run artifacts; reported model outcomes are owner-level claims needing manifest/raw-output verification; Smooth_MrBudget, P2, E1-slot28 remain coordination-level claims in this report (paper PDF, MrRoPE full paper, server JSON, failure transcripts not ingested).

### 4.7 KKT relevance
Sol03 delivers the explicit KKT marginal law (eq. 2) and the equality-constrained simplex geometry ({u≥0, 1ᵀu=R}) on which all four reports' allocations live — the strongest formal KKT statement in the lineage.

### 4.8 Conflicts with established facts
None; sol03's rejections are exactly aligned with the project vetoes (root non-ranking, static collision ≠ mechanism, Q/K-weighted subspace insufficient, endpoint invariants as design constraints — cf. its own "fix the chosen endpoint policy" as constraint, not theorem).

---

## 5. Cross-report alignment map

- Shared coordinate object (all four): span-R simplex of adjacent log-radices / cumulative exponents — [已验证-derivation]. astra04's x_j + endpoint+sum constraints, sol01's (m, δ), sol02's gaps g_r, sol03's u_k are the same feasible set in different notations.
- Two distinct calibration sources proposed: (a) astra04 — distillation KL on real native attention logits of stitched 32K captures (answer-free, fit-now, no GPU); (b) sol01/sol03 — frozen-model logit distortion D₀ / native-output KL curvature H_N. Both are *functional* (softmax/logit-level) calibrations, deliberately bypassing the geometric-subspace route that failed to rank.
- Exact kernel supply: sol02 gives the closed-form K_L + ∂₁K_L + cumulative-gap gradient that sol01's C_T and sol03's J_pos were pointing at; sol02 also proves the cosh form does NOT apply to the exact finite-window problem, which scopes every "EVQ says X" claim in sol01/sol03 to the surrogate.
- Common acceptance veto (all four): downstream paired long-task evaluation decides; geometry/calibration objects are candidate generators and diagnostics.

---

## 6. Lineage verdict (5 lines)

1. Calibration-derived quantities can serve as the F functional (or its constraint set) in a KKT allocation problem ONLY in their *functional* forms — astra04's answer-free L_bal on real logits, sol01's budgeted C_T-with-D₀≤δ, sol03's g_L/H_N trust region — because those are differentiable, measured, row-shift-honest objectives on the installed function.
2. Pure geometric calibration objects (positional Gram, cosine kernel alone, static collision/root, subspace coverage, raw logit MSE) are diagnostics only — each is independently vetoed by a counterexample (BM, Smooth, rank-2-vs-2K, row-shift, root non-ranking).
3. The strongest single calibration object delivered is **astra04's corrected full-row package: exact padded-KL identity + block-balanced L_bal + 98304-slot map correction + analytically verified gradients + the endpoint/order/sum-constrained SLSQP allocator** — the only recipe fit-able on existing captures before any new GPU work, with the identity-map guard against MrPro.
4. Its closest KKT-structured rival is sol01's constrained argmin C_T s.t. D₀≤δ with MrPro-measured δ, armed by sol02's exact K_L and gap-gradient and formalized by sol03's equal-marginal law (eq. 2); all three share the same feasible simplex.
5. No report licenses a task-independent curve: every rule is a candidate generator whose acceptance is native-gated strict paired long generation; sol03's fallback ladder (g_L=0 ⇒ "compatibility candidate"; no calibration ⇒ MrPro unchanged) is the honest floor.
