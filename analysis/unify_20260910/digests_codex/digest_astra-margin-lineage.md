# Digest — Astra margin lineage (astra01 / astra05 / astra06 / astra08)

Source reports: `.agents/rope_unification_20260910/reports/astra01.md`, `astra05.md`, `astra06.md`, `astra08.md`.
Assignment theme (skimmed from `.agents/rope_unification_20260910/assignments/astra*.md`): derive a concrete EVQ+MrRoPE unification with a constructive allocation law, correct scratch-vs-frozen framing, CPU-only checks, full-text ingestion, no candidate grids, no geometry-proxy task-success theorems.

All four reports share one spine: **replace EVQ's unsigned cosine energy with a role-conditioned (source-vs-distractor) signed margin and its covariance**, recover EVQ and Mr-type behavior as special cases, and give an exact finite solver (inverse-C law / integer DP / projector transport) plus an identification recipe. None of them uses the RoPE-bound first-zero root to rank tables — no root-veto triggered. None contradicts the panel numbers; all four cite Smooth's geometry-better/task-worse reversal as a binding constraint.

---

## Astra01 — Signal vs interference: role-conditioned margin, matched-filter allocation law

### 1. Core thesis
EVQ's quadratic `E_D[B_ρ(Δ)²]` is a squared *aggregate* response and cannot be a universal allocation objective because changing the semantic role of the same lag reverses the preferred ordering; the right common object is a role-conditioned source-vs-distractor score margin `M = ∫Z_e dη` with mean `μ` and covariance `v`, and the allocation law is the generalized matched filter `ρ* ∝ C⁻¹h`.

### 2. Concrete math delivered
- Exact EVQ kernel: `K(x,y) = E_{Δ~D} cos(ω(x)Δ)cos(ω(y)Δ)`, `ω(x)=b^{-x}`, and `⟨ρ,Kρ⟩ = E_D B_ρ(Δ)²`, `B_ρ(Δ)=∫cos(ω(x)Δ)ρ(x)dx` [已验证-derivation — reproduces the EVQ note].
- Response field per role/lag draw e: `Z_e(x) = Re{A_{+,e}(x) e^{iω(x)Δ_{+,e}} − A_{−,e}(x) e^{iω(x)Δ_{−,e}}}` with complex signed content coefficients; `h(x)=E Z_e(x)`, `C(x,y)=Cov(Z_e(x),Z_e(y))`; for measure η: `μ_η=∫h dη`, `v_η=∬C dηdη`. Exact for the stipulated field; C is PSD by construction and carries signed cross-frequency cancellation [已验证-derivation, 假设-laden: h,C prescribed].
- Standardized margin `t = μ/√v`. Gaussian margin ⇒ `Pr(M≤0)=Φ(−t)` exactly; distribution-free Cantelli ⇒ `Pr(M≤0) ≤ 1/(1+t²)` [已验证-derivation]. Constant threshold absorbed via h→h−a since η(I)=1.
- Allocation law: `max_{ρ≥0, ∫ρ=1, ⟨h,ρ⟩>0} ⟨h,ρ⟩/√⟨ρ,Cρ⟩` ≡ (homogeneity) convex `min_{w≥0, ⟨h,w⟩=1} ½⟨w,Cw⟩`; if `C⁻¹h ≥ 0`: `ρ*(x) = [C⁻¹h](x) / ∫[C⁻¹h]` with `t*² = ⟨h,C⁻¹h⟩` (Cauchy–Schwarz in C-inner product) [已验证-derivation]. Positivity binding ⇒ active-set KKT: `Cw ≥ λh, w ≥ 0, w(Cw−λh)=0, ⟨h,w⟩=1` [已验证-derivation — this is literally the KKT system].
- Multi-role: maximize `min_r t_r`; for fixed t the constraints `‖C_r^{1/2}ρ‖ ≤ ⟨h_r,ρ⟩/t` + mass + positivity are convex ⇒ bisection gives global optimum of the continuum surrogate [已验证-derivation]. Protected frequency part contributes fixed mean/cov cross terms; formulation stays affine in remaining measure.
- EVQ recovery: nuisance model `Z_e(x)=h0 + ξ_e cos(ω(x)Δ_e)`, `Eξ=0, Eξ²=1`, ξ⟂Δ ⇒ `h=h0` (constant), `C=K_exact`; maximizing standardized margin *exactly* minimizes EVQ's energy [已验证-derivation, conditional]. EVQ = protected-constant-signal special case, not a general remote-content model [作者明确声明].
- Cosh recovery from covariance approximation `C_app = αI + βG`, `G(x,y)=min(x,y)`: `αρ+βGρ=λ ⇒ ρ''−τ²ρ=0, ρ'(1)=0 ⇒ ρ*(x)=τ cosh(τ(1−x))/sinh τ`, `τ²=β/α`; tail form `v = α∫(T')² + β∫T²`, `T(x)=sinh(τ(1−x))/sinh τ` [已验证-derivation]. Nonconstant h ⇒ forced equation `αρ''−βρ=λh''`, `αρ'(1)=λh'(1)` with free-boundary conditions — no automatic Cosh [已验证-derivation].
- White-noise caveat: `αI` is a frequency-field white noise (`Var∫ρ dW = α∫ρ²`), NOT K iid channel noises (those give `K⁻¹∫v(x)ρ dx`, linear in density — cannot produce `α∫ρ²`); finite-resolution realizable alternative `k_ε(x−y)=ε⁻¹(1−|x−y|/ε)_+`, `C_ε=αk_ε+βmin`, diagonal `α/(Kε)` diverges as ε→0 at fixed K — density and finite-K limits do not commute [已验证-derivation; equivalence of real nuisance to shared-bin structure = 假设, declared].
- Finite-K quantile realization: midpoint-quantile atoms `x_k=F⁻¹((k+½)/K)` give `W₁(η_K,ρdx) ≤ R/(2K)` (proof; endpoint-pinned variant `≤R/K`); with L_h-, L_C-Lipschitz h,C and d=W₁: `|μ_K−μ|≤L_h d=:e_μ`, `|v_K−v|≤2L_C d=:e_v`, then `t_K ≥ (μ−e_μ)/√(v+e_v)` and corresponding Cantelli/Gaussian bounds [已验证-derivation; author flags it can be weak at K=64, 128K lags since `L_C ~ α/ε²+β`].
- Frozen labeled design: product space (slot label k, candidate frequency x): `h_k(x)`, `C_{kℓ}(x,y)`, `η_Ω=K⁻¹Σ_k δ_{(k,x_k)}`; label constraint prevents replacing one trained slot with redundant frequencies; density-only on x discards learned slot association [已验证-derivation (exactness of Eq.1 on product space); 假设 for stationarity of the field].

### 3. Status tags per claim
- Same-lag role-reversal counterexample: 3 frequencies at Δ=2π, A=(1,½,¼), B=(1,0.9,¼): `B_A=0`, `B_B=1.809017`; EVQ quadratic prefers A (0 vs 3.27254). Wrong-key model (margin=B, Gaussian σ=√3): pairwise error 0.5 vs 0.148142. Positional model (margin=3−B): 0.041632 vs 0.245848. Ordering flips with role. [已验证-derivation — exact Gaussian score model; NOT LM outcomes (author says so)]
- "Single unsigned cosine energy cannot be universal objective" [已验证-derivation relative to the stipulated score models; the score models' fit to real LM behavior = 假设].
- τ=√(β/α) ≠ empirical d/√L rule unless separately demonstrated; historical mismatch documented [部分证据—agrees with the project's own audit; the d/√L identification itself is rejected].
- Exact oscillatory K ≠ Brownian+identity C in general; historical exact-kernel optima had truncated densities / collision strengths outside trained basin [部分证据—cites STRONG_MODEL_THEORY_VERDICT, not re-derived].
- MrPro = heuristic member, not an exact optimizer; making it solve (4) by manufacturing h=Cρ_Pro is inverse-optimal-control with zero predictive content, explicitly rejected [已验证-argument].
- Native-window cache rephasing is exact for local rotary contraction of that cache but cannot identify long-context h,C from native-only observation; two response families agreeing up to W can disagree beyond with opposite optimal allocation [部分证据—constructive argument + test reference].
- No root-based ranking anywhere; no conflict with panel; endpoints/anchors treated as declared constraints [consistent with established facts].

### 4. Proposed allocation rule (recipe)
Population-design version: (i) specify role/lag distribution and source-vs-distractor contrast classes; (ii) estimate h(x), C(x,y) from the stipulated/observed response field; (iii) solve `Cw=λh`-KKT active set (5), clip only by active set not naively; (iv) emit frequencies as equal-weight quantiles of `ρ*=w/∫w`. Frozen Qwen version: keep slot labels and anchors, optimize the finite labeled product-space version, NOT the density quantile table. Data needed: signed short-role coefficients from native-window Q/K measurement; long-range h,C needs transport assumption or actual long forwards.

### 5. CPU numbers
(1) Sign counterexample errors above. (2) 800-point discretization of `I+4G` (τ=2): solved density vs analytic Cosh max error 2.986e−7 [已验证-numeric]. (3) Smooth covariance `0.3e^{−|x−y|/0.15}+min(x,y)`, h=1+0.4x, Cosh density, K=16/64 quantiles: mean errors 3.37e−4/2.55e−4, variance errors 1.333e−3/5.03e−4; conservative analytic bounds 1.25e−2/3.125e−3 and 0.1875/0.046875 — bounds valid but loose [已验证-numeric, author notes bounds are weak at K=64].

### 6. Obstructions / negative results
- No universal non-identifiability theorem adopted; the *usable* obstruction is missing role/label/learned-response information, exhibited by the sign counterexample.
- From-scratch law is not a solution of training: trained h,C change with ρ (co-adaptation); table-crossings evidence against transplanting response laws.
- At fixed K the optimal measure can have atoms; strict spacing/max-occupancy/exact endpoints are extra architecture constraints, not guaranteed by C⁻¹h.
- Union bounds over multi-distractor are loose; positive mean vs one random key insufficient vs 128K competitors; multi-key contrast classes required.
- Cantelli conditional-vs-marginal Gaussianity warning: use `E_Δ Φ(−t_Δ)` or unconditional Cantelli, never `Φ(−μ/σ)` after lag mixing.

### 7. KKT relevance
Directly a KKT generator: (4)/(5) is the KKT solution of the constrained variational problem; the three-band structure enters as regime splits of h (protected/constant = low band; oscillatory remote = high band; transition where positivity binds); finite-K error bounds (8) supply the certified bridge from continuum F to the 64-channel table. For F = near damage + far capability: `v` (covariance of coherent nuisance at wrong lags) = near-damage term; `μ` at the remote target lag = far-capability term; multi-role min-t formulation is exactly the max-min of the two.

### 8. Disagreements with established facts
None material. It sharpens them: warns τ is not d/√L, warns αI is not iid channel noise, warns quantile bound weak at K=64 — all consistent with the red-lines. The proposed next test (does margin moment structure separate MrPro/P2/BM/Smooth on existing behavior) is aligned with the panel as ground truth.

---

## Astra05 — Role-conditioned nonlinear modes and exact mode-selective transport

### 1. Core thesis
EVQ's squared cosine and MrRoPE's positive cosine appear as different terms of one signal-vs-distractor **log-partition margin** `M = log Z_S − log Z_D`; independent isotropic distractors do NOT produce the squared-kernel cost (coherence is essential), and when a *joint integer relation* among frequencies carries the source computation, the exact minimal-intervention allocation is a projector transport `ν=(I−P_R)ω+S⁻¹P_Rω` — a genuinely different direction from monotone m-ramps.

### 2. Concrete math delivered
- Exact per-key rotary score at fixed pre-RoPE Q/K: `z_t(ν)=b_t+Σ_j{a_tj cos(ν_j d_t)+b_tj sin(ν_j d_t)}`; `Z_A(ν)=Σ_{t∈A}e^{z_t}`, `M=log Z_S−log Z_D`, `p(S|S∪D)=σ(M)` — no Fourier stationarity/independence/small-change assumptions needed [已验证-derivation]. Caveat: conditional mass if S∪D omits keys; M is an attention observable, not an answer theorem.
- EVQ/Mr bridge: `K_ν(d)=K⁻¹Σ_j cos(ν_jd)`; deterministic signal `μK_ν(d_s)`, distractors i.i.d. distances, **shared coherent** score `ξ_t K_ν(d_t)`, `ξ_t~N(0,σ²)`: `E e^{z_D}=E_{p_D}exp{½σ²K_ν(d)²}`; `M̄(ν)=μK_ν(d_s) − log N − log E exp{½σ²K_ν²}`; weak-noise: `−M̄ = log N − μK_ν(d_s) + ½σ²E_D K_ν(d)² + O(σ⁴)` [已验证-derivation]. The three terms: Mr-like positive-cosine signal, EVQ-like squared-kernel nuisance (squared AFTER summing), −log N distractor multiplicity. Jensen: `E M ≥ M̄` [已验证].
- Crucial limitation: `α,β` of the Cosh surrogate are NOT identified by this bridge; Cosh is not thereby the exact optimizer of M̄ [author states; matches 03_theory.tex:74–110].
- Isotropic counterexample: `z_D(d)=Σ_j[A_j cos(ν_j d)+B_j sin(ν_j d)]`, `A_j,B_j iid N(0,σ²/K)` ⇒ `Var z_D = σ²` independent of ν,d; `E e^{z_D}=e^{σ²/2}` — no squared kernel appears [已验证-derivation]. General Gaussian coefficients: `log E e^{u^T x_ν(d)} = m^T x_ν(d) + ½ x_ν(d)^T C x_ν(d)` — coherence structure of C determines whether collisions cost anything.
- Nonlinear joint-mode expansion (exact, per-key, position-varying coefficients): `e^{z_t}=e^{z_{t,−B}} Σ_{n∈Z^|B|} c_{t,n} e^{i(n^Tν_B)d_t}`, `c_{t,n}=Π_{j∈B} I_{n_j}(r_tj) e^{−in_jφ_tj}` (Bessel I); role transforms `H_{A,n}(κ)=Σ_{t∈A} e^{z_{t,−B}}c_{t,n}e^{iκd_t}`, `Z_A=Σ_n H_{A,n}(n^Tν_B)`; exact derivative `∇_{ν_B}M = Re Σ_n n {H'_{S,n}/Z_S − H'_{D,n}/Z_D}` [已验证-derivation; builds on ROPE_SOFTMAX_MIXED_FREQUENCY_DERIVATION_20260910]. Same harmonic can help or hurt depending on signed role contrast — |c_n| or period length alone cannot choose stretch/preserve/suppress.
- Truncation rule: if |error_A| ≤ η_A Z_A, η_A<1, log-margin error ≤ `−log(1−η_S)−log(1−η_D)` [已验证-derivation — numerical rule, not capability threshold].
- Transport law: `min_ν ‖ν−ω‖² s.t. Rν = Rω/S` ⇒ `P_R=R^T(RR^T)†R`, `ν=(I−P_R)ω+S⁻¹P_Rω`; every row relation retimes exactly (`n^Tν=n^Tω/S`), every orthogonal relation preserved exactly; composition `T_R(S)T_R(U)=T_R(SU)` at fixed R (no scale-path ambiguity) [已验证-derivation — the projector formula is the KKT solution of its own QP].
  - Two-frequency envelope n=(1,−1), c=(ω₁+ω₂)/2, g=ω₁−ω₂: `ν₁=c+g/(2S), ν₂=c−g/(2S)` — beat stretches, carrier retained; **one frequency moves UP**, excluded by any compression-only box ν_j≤ω_j; box-constrained variant `ν₂=ω₂, ν₁=ω₂+g/S` shifts the carrier [已验证-derivation].
  - Three-frequency curvature n=(1,−2,1), κ=ω_j−2ω_{j+1}+ω_{j+2}: `ν_B=ω_B−(1−S⁻¹)κ(1,−2,1)/6` — mean and linear slope unchanged, curvature clock ÷S; alternating-sign changes, not monotone smoothing [已验证-derivation].
  - Uniform PI recovered when row(R)=full space; block PI when R spans a coordinate block; full-rank R kills carrier preservation ⇒ local-vs-long conflict can force global PI [已验证-derivation].

### 3. Status tags
- Bridge M̄ decomposition [已验证-derivation under declared coherent-nuisance model]; the model's fit to real Qwen distractors = 假设, and the isotropic counterexample shows generic "noise" does not supply it.
- Mode transport is a valid *hypothesis family* [已验证-derivation]; identification of any causally useful R in Qwen = **not established** [假设/未证实 — author says so explicitly: bias coefficient at layer27/head8 has far-mass 2.576e−8 (MrPro) / 2.055e−7 (Smooth) beyond first quarter, negligible; "cannot supply that identification"].
- R-selection contract: (a) material normalized contribution after full-row partition, (b) sign/phase supports source-vs-distractor separation in source-only paired worlds, (c) the intended coordinate extension demands the retiming; relations present in Q/K biases fail this [部分证据 — operational recipe, no data yet satisfies it].
- "If no relation satisfies the contract, outcome is NO justified intervention; do not pick a harmonic by maximizing exposed 128K answers" — anti-overfitting rule, aligned with red-lines [已验证-argument].
- Rejection of universal-closure vetoes, of the old nullband.py "wrapped channel unconditionally safe" claim (contradicted by mixed modes), of causal alignment asserted without measuring [作者立场, consistent with the panel's demand for evidence].
- Smooth beats MrPro on source-weak exposure/unweighted distortion yet worse 128K outputs (cites ROPE_ALLOCATION_SUBSPACE_DERIVATION_20260910:151–204, 208–281) ⇒ rules out choosing R by minimizing total source-weak energy / operator distance / frequency norm [部分证据 — this matches established panel fact].
- Carrier-removal pilot: −73.85% on background derivative objective alongside severe long VT/UUID degradation ⇒ beat preservation alone cannot promote the family [部分证据 — cites existing pilot record].

### 4. Proposed allocation rule
(i) Fix a source computation with a spatial transformation requiring a coarse coordinate to stretch by S; (ii) identify integer relation rows R from exact role transforms on fixed pre-RoPE Q/K + explicit source set S + full competing key set D + source→target coordinate map (synthetic assays or defensible annotations — attention magnitude insufficient); (iii) deploy `ν=(I−P_R)ω+S⁻¹P_Rω`; (iv) validate by exact finite replay of Z_S/Z_D against an equal-size orthogonal-intervention control, then actual generated answers; native/local controls mandatory since exact retiming does not preserve raw logits. Data needed: role-labeled captures — same gap Astra08 quantifies.

### 5. CPU numbers
- Formula-grid b=10⁶, K=64, S=4: slots36−37 first difference: period 76740.5661→306962.2644, changes −7.2809%/+9.0352%, ‖change‖/‖PI change‖=0.10690. Slots28–30 second difference: period 70286.2105→281144.8420, changes −0.471216%/+1.169499%/−0.725638%, ratio 0.01069. Stored Qwen period 70285.94 (initializer rounding documented) [已验证-numeric; reproducible snippet in report].
- Continuation (requested family, 29 candidates = 15 adjacent first-differences + 14 adjacent second-differences inside slots 24–39), starting from verified FP32 MrPro table, retargeting Native clock /4: `ν_c = ν_M + n(n^Tω_native/4 − n^Tν_M)/(n^Tn)`. All finite/positive/strictly decreasing, endpoints+outside slots bitwise preserved, FP64 projection identities pass, deployed relation-clock error ≤1.0494e−5; gain 1.138629436111989; Mr Σm=29.333333268998935 [已验证-numeric, SHAs given: native 138c99b1…, Mr 33cbe3a4…].
- Key correction: **none of these relations already retimes Native by 4** — early middle first differences run FASTER than Native despite every raw frequency being slowed; second-difference periods mostly shorten. The family changes actual joint clocks; it is not a disguised Mr ramp [已验证-numeric].
- Table highlights: d1_s24..26 push one slot beyond native (outside compression-only box; still positive/ordered); ratio Mr/native from 0.920 to 1.901; d1 max phase change at 128K up to 58.31 rad; d2 candidates all inside box, ratios 0.841–1.169, max phase change ≤6.91 rad, ΔΣm tiny (±0.002) [已验证-numeric]. Zero-sum relations preserve raw frequency sum, NOT the Σm budget — not same-compression-budget controls (author reports deltas rather than silently fixing).
- Files: `.agents/rope_unification_20260910/code/joint_mode_candidates.py`, `joint_mode_candidates.json` with `delta_log_period=−log(ν_c/ν_M)` per entry for contraction with parent `gradient_log_period`; first-order loss prediction is a direction filter only, cannot forecast 58-rad finite changes.

### 6. Obstructions / negative results
- Independent isotropic noise ⇒ rotation-invariant variance ⇒ EVQ squared-kernel unjustified without stated coherence (kills lazy "background noise" derivations).
- Magnitude-only or period-only mode selection is impossible: signed role contrast decides.
- Full-rank interaction across the whole space forces global PI (local-vs-long conflict made precise).
- No GPU/model intervention run; no identified active relation for Qwen; advantage over MrRoPE unproven.

### 7. KKT relevance
Contributes a new *constraint geometry* for the middle-band 16 DOF: the transport is the KKT solution of minimal-displacement s.t. exact harmonic-retiming constraints; it shows the compression-only box `ν_j≤ω_j` is itself a substantive mechanism restriction (excludes carrier-preserving envelope transport) — a claim about the feasible set of the KKT problem, not F's terms. It also supplies the coherent-nuisance covariance structure (joint sums/differences with sine phases and cross-slot covariance) that F's near-damage term needs before any α,β are meaningful. The 29-candidate table is an exact finite family inside the free slots 24–39 with both endpoints bitwise preserved — consistent with the endpoint design constraints.

### 8. Disagreements with established facts
None. It reinforces: root not used for ranking (notes Mr's first-zero does not derive the arithmetic increments — §3.2.2 assumes them); Smooth reversal treated as binding; endpoints kept as constraints; gain and MrPro table SHAs verified from run contracts.

---

## Astra06 — Adversarial audit + exact finite-channel softmax-mass allocation

### 1. Core thesis
When the requested computation is softmax source mass, the right criterion is stronger than standardized SNR: minimize `J = v/2 − μ` (with μ deterministic-signal, Gaussian-envelope coherent distractors), which admits an **exact integer dynamic program allocating K equal channels to B frequency bins**; constant protected μ recovers a discrete Cosh law, a signed coherent remote μ recovers the Mr-type preservation term — both inside one solver.

### 2. Concrete math delivered
- Row probability: `p_* = 1/[1+Σ_t exp(D_t−μ)]`; distractor MGF envelope `E exp(D_t) ≤ exp(b_t+v_t/2)` (Gaussian one exact realization; envelope can be established without Gaussianity). Markov gives `P[p_*<ρ] ≤ min{1, [ρ/(1−ρ)] Σ_t exp(b_t+v_t/2−μ)}` — **no independence across keys** [已验证-derivation]. Random source requires joint MGF `E exp(D_t−S)`, not mean substitution.
- Common b=0,v: minimizing `v/2−μ` exactly minimizes the bound; sufficient condition `μ − v/2 ≥ log N + log[ρ/(1−ρ)] + log(1/δ)` [已验证-derivation]. Contrast with SNR: shrinking both μ,v raises μ/√v but can lower mass — the log N term is absolute-scale, absent from squared-SNR.
- Declared linear readout: contrast ≥ `r + a·p_* − b(1−p_*)`; certified if `r+aρ−b(1−ρ)>0`; one-hot copying construction is a nonvacuous instance [假设-conditional — must not be assumed satisfied by real Transformers].
- Channel-count model: B bins, spacing Δ=1/B, `x_i=iΔ`, `ω_i=exp(−x_i)`; integer counts `n_i`, `Σn_i=K`, `p_i=n_i/K`; nuisance covariance `C_ij=(α/Δ)1{i=j}+β min(x_i,x_j)` (independent bin noise variance α/Δ + shared Brownian increments βΔ — shared frequency-local noise, all channels in a bin see the same bin nuisance) [假设-declared, testable].
- Exact variance: `μ(n)=K⁻¹Σh_i n_i`; `v(n)=K⁻²[(α/Δ)Σn_i² + βΔΣT_i²]`, `T_i=Σ_{j≥i}n_j` (min-kernel identity) [已验证-derivation].
- Integer program: `min_{n∈Z₊^B, Σn=K} Σ_i[(α/2Δ)n_i² + (βΔ/2)T_i² − K h_i n_i]`; DP `F_i(t)=(βΔ/2)t² + min_{0≤n≤t}{(α/2Δ)n² − K h_i n + F_{i+1}(t−n)}`, backtrack from `F_1(K)`; O(BK²) time, O(BK) memory, no local minima; endpoint/occupancy constraints by restricting n-range; source `h_i=A` constant (protected) or `h_i=A cos(ω_iD−θ_i)` (aligned remote, signed phase) [已验证-derivation — this is a fully constructive KKT-consistent solver].
- Frozen labeled variant: bin-dependent label-independent covariance keeps DP exact with `h_{j,i}` and consecutive-label bin assignment and prefix sums; heterogeneous slot covariance destroys the decomposition [已验证 within declared model].
- Discrete vs continuum Cosh: constant-h relaxed optimum `Cp=λ1` ⇒ `p_{i+1}−(2+βΔ²/α)p_i+p_{i−1}=0`, `p_{B+1}=p_B` ⇒ `p_i ∝ cosh[κ(B+½−i)]`, `κ=arcosh(1+βΔ²/2α)`; `κ/Δ→√(β/α)` as Δ→0; exact finite-K answer is the integer DP, not (7) [已验证-derivation]. Nonconstant h: `Cp−h=λ1` occupied / `≥λ` empty; continuum `αρ''−βρ=h''`, `αρ'(1)=h'(1)` + free boundaries — no universal Cosh [已验证-derivation; note the forcing coefficient differs from Astra01's standardized-margin version because the optimized quantity is the exponential-moment mass bound, not a homogeneous ratio — authors' formulations differ accordingly].

### 3. Status tags
- Softmax-mass bound and DP [已验证-derivation]. Envelope validity on real logits = 假设, flagged as the empirical gap.
- "Astra02's continuous-kernel equilibrium is not Cosh" [audit verdict on a sibling report; treat as 部分证据 pending that report's digest].
- Pairwise-SNR non-dominance counterexample: means 0.8 var 1 independent ⇒ all-positive prob Φ(0.8)²=0.6211719; perfectly correlated means 0.5 ⇒ Φ(0.5)=0.6914625 — better every pairwise margin, worse joint retrieval [已验证-numeric]. Sufficient dominance condition: common coupling `M_A=m_A+DZ`, `M_B=m_B+DZ`, same diagonal scale, same joint noise, m_A≥m_B componentwise ⇒ nested events [已验证-derivation; "substantial structure, not implied by covariance summaries"].
- Identical mean/covariance does not determine nonlinear task failure; Cantelli bounds are certificates, not orderings [已验证-argument].
- Joint torus containment strictly stronger than marginal phase coverage: common PI preserves the one-parameter orbit on retimed lags; independent compressions generally do not [已验证-argument].
- Transplant rigidity conditional: `A^T R_new(d) B = R_old(d)` on an open lag interval ⇒ matching spectra, but does not exclude low-rank compensation/unused channels/learned adaptation, and does not mandate a displacement schedule [已验证-derivation + author caveat].
- Astra02 integer-lag audit: finite integer-lag Gram has finite rank on signed measures, not strictly PD ⇒ uniqueness doesn't transfer; uniform measure on [0,π] has zero cosine moments at all positive integer lags (diffuse minimizer) [已验证-derivation; flagged as assumption counterexample, not the RoPE [1/base,1] case].

### 4. Proposed allocation rule
Use bound (1)/criterion `v/2−μ` as objective target when the computation is attention mass; solve DP (6) exactly under the declared shared-bin+nested covariance with the signed source h_i; emit K locations by repeating `ω_i` n_i times (equal amplitude — this allocates channel COUNTS, not weights); preserve labels in frozen mode; require B to be a stated physical covariance resolution (refining B at fixed α changes the model). Data needed: h_i from source role (protected constant or remote lag D, phase θ_i); α,β from nuisance field estimation under real long prefill.

### 5. CPU numbers
Declared check B=16,K=64,α=1,β=4 (τ=2):
- Constant h=20: DP counts `[8,7,6,6,5,4,4,4,3,3,3,3,2,2,2,2]`; variance 2.2095337; direct quadratic matches DP; uniform counts 2.4609375; real-relaxed optimum 2.2038758; discrete Cosh (7) vs direct C⁻¹ solve agree to 2.78e−17; independent exhaustive B=3,K=4 enumeration gives (2,1,1) variance 2.875 [已验证-numeric].
- Bound at N=131071, ρ=0.9, μ=20: failure probability ≤ **0.0073393** ⇒ P[mass≥0.9]≥0.99266 — theorem non-vacuous at 128K in-stated-model [已验证-numeric].
- Oscillatory source h_i=20cos(2e^{−x_i}) (only change): counts `[0,…,0,4,7,9,12,15,17]`, μ=13.5074, v=6.4801, bound >1 ⇒ uninformative [已验证-numeric; honest demonstration that constant-signal Cosh and remote preservation demand different allocations and that optimization cannot conjure capacity].
- Reproducer snippet included in report.

### 6. Obstructions / negative results (audit section)
Historical errors catalogued with paths: challenger_2's "cosine Taylor converges only for small phase" (cosine is entire); reviewer_1's fabricated condition number >10⁴ and 81.2% curve-fit residual reinterpreted as Hessian curvature; explorer_survey_1 mislabeling FullLagP2 U as a checkpoint content statistic (it is native geometry, no learned Q/K); worker_empirical_r3's "fixed channel count ⇒ information-theoretic impossibility of simultaneous short/long improvement" (count conservation gives no task-risk conservation law) and its reversed Cosh direction; blueprint's native-table-swap claim contradicting weight/table-crossing evidence (exact routing needs identical weights+table+gain+state); BM diagnostic = two selected instances only. Live numerical owner `ROPE_LOCAL_FAILURE_SYNTHESIS_20260908.md §6` retracts the broad old non-identifiability theorem — retraction overrides preserved rhetoric. Smooth reversal re-confirmed with numbers: remote unresolved response 0.0494352 vs 0.235928, C26 0.000159185 vs 0.000240986, C32768 35.3793 vs 42.5180, lower projection-operator MSE at all ranges and lower weighted remote weak response in all 36 layers × 3 cutoffs — yet worse development task ⇒ **forbids any positive combination of those audited geometry costs from selecting the better frozen table** [部分证据 — matches established panel fact].

### 7. KKT relevance
This is the closest of the four to an *executable* KKT system for F: it fixes the objective (mass bound ⇒ `v/2−μ`), fixes the feasible set (integer counts summing to K = the real 64-channel constraint the continuum density theories ignore), and solves it exactly (DP = discrete KKT, no local minima). The three-band structure appears as h-regimes: constant h → discrete Cosh (low band), signed oscillatory h at remote lag D → mass shifted to where cos(ω_iD−θ_i)>0 (far band), transition = free-boundary positivity set. The N source-projector is demoted to a constraint diagnostic (`tr(N G_far(ν))` checked post-transport), never an objective [aligned with red-lines].

### 8. Disagreements with established facts
None. Explicitly defers MrPro's arithmetic progression (not derived), refuses to identify real Qwen h,C, and repeats that Cosh is an assumption-dependent surrogate. Minor care: its bin support is [Δ,1] not [0,1] — author flags not to call the right-endpoint rule "endpoint-pinned RoPE"; any integration must state support convention.

---

## Astra08 — Identifiable role margins and a finite source-calibrated estimator

### 1. Core thesis
The margin program becomes an allocation algorithm only when h,C are *identified* from labeled data; the repository's six natural-text 32K captures structurally cannot do it (no role labels — a role-swap leaves every observable identical while reversing the correct answer), so the usable rule is supervised finite fitting of one ordered 64-vector against role-labeled source-window counterfactual tasks with output-sensitivity weighting, exact replay, and acceptance on actual full-model outputs.

### 2. Concrete math delivered
- Role sets from the input parser (source relation labels, not circuit claims): `T_e` requested key+ordinal record; `H_e` same-key other ordinals (hard distractors); `O_e` other records; `B_e` background; `Q_e` question tokens [部分证据 — constructions exist: `prepare_support_oracle.py:14–47`, `retention_evidence.py:26–52`, `nosa_position/test_data.py:59–74` 2×2 content/query-swap, `refcarry_audit/prepare_mrcr_pairs.py:55–101`]. Maximizing T-attention in every head is invalid (question-key copies compete; targets are not uniformly useful).
- Exact finite replay at captured implementation: split-half pair k, `C_jk=q_k k_jk + q_{k+K} k_j(k+K)`, `S_jk=q_{k+K} k_jk − q_k k_j(k+K)`; `z_ej(ν;d)=a_e Σ_k[C_ejk cos(d_jν_k)+S_ejk sin(d_jν_k)]`, `a_e=module.scaling × common_gain²` (do not double-multiply; `pro_block_calibration.py:60–77` distinguishes runtime-BF16 vs mathematical replay KL); `p_ej` = softmax over **every causal key**; `o_e=Σ p_ej v_ej`. No approximation [已验证-derivation, matches repo code].
- Diagnostics: `M_{T,rest}=log Z_T − log Z_{all\T}` (gives p(T)=σ(M)) and `M_{T,H}=log Z_T − log Z_H` (conditional p(T|T∪H)) — report separately [已验证].
- Output-level label: `U_e = log P_model(y_e|prompt_e)` under single-token tokenizer-verified distinct answer strings at the actual output boundary, rejecting shared-first-token candidate pairs; plus signed margin against other record values [design, 部分证据].
- Sensitivity weighting: `g_e,lh = ∂U_e/∂o_e,lh` (one backward pass at source checkpoint); value reward `r_e,lh,j = g^T v_e,lh,j` (signed, task-conditioned, includes W_O/residual/downstream); `J_lin(ν;d') = mean_e Σ_lh g^T[o_e,lh(ν;d') − o_e,lh(ν₀;d')] = Σ_j[p_j(ν;d')−p_j(ν₀;d')] r_j` with ν₀ = MrPro at same gain [已验证-derivation of estimator; 假设 for proposal validity — source Jacobians on source states, simultaneous all-layer changes interact].
- Coordinate stretch for calibration prompts: `g_S(p)=S⌊p/B⌋+(p mod B)`, B=4096, S=4 — preserves within-block distances, multiplies block-start offsets; explicit artificial transform, NOT a 128K forward (adds no distractors, no hidden-state recompute) [declared limitation].
- Parameterization: `x_k=log ν_k`, shared 64-vector, init `x₀=log ν_MrPro`, constraints `x_k≥x_{k+1}`, endpoints pinned `x_0=x_native,0`, `x_63=x_MrPro,63` for pure interior allocation; endpoint moves must be declared extra variables; channel zeroing outside positive log parameterization [consistent with endpoint-as-constraint fact].
- Near-term compatibility: `D_src(x)=mean over natural source rows KL[p_native(ω;d) ‖ p(e^x;d)]`, constraint `D_src(x) ≤ D_src(x₀)` plus no worsening of measured source-family utility vs x₀ — "no worse than reference on this captured objective", not "safe vs Native" [已验证-definition].
- Optimization: projected gradient / sequential constrained optimization of −J_lin with trust-region/backtracking acceptance judged on exact finite trigonometric logits + full softmax + constraints; never accept a large step on local-quadratic grounds; no candidate lattice; Native/MrPro are init/controls [algorithm spec].
- Guarantees [all 已验证-derivation]:
  - `|z_j(ν)−z_j(ν₀)| ≤ 2a Σ_k √(C_jk²+S_jk²) |sin[d_j(ν_k−ν₀k)/2]| =: ε_j`; `log Z_A` is 1-Lipschitz in max logit error ⇒ `|ΔM_{T,D}| ≤ max_{j∈T}ε_j + max_{j∈D}ε_j` — sign of a role margin cannot flip if it exceeds the bound (finite change, signed coefficients; role-ranking certificate, not answer guarantee).
  - Pinsker: `KL[p₀‖p_ν]≤δ ⇒ |p_ν(T)−p_0(T)|≤√(δ/2)` (may be vacuous for rare targets; average KL gives average bounds only).
  - `‖o_ν−o_0‖ ≤ 2V·TV(p_ν,p_0) ≤ V√(2δ)` for ‖v_j‖≤V (measure actual o-difference; cancellation makes norm bounds loose).
  - Taylor with Hessian bound: `U(o+δo) ≥ U(o)+g^Tδo−(H/2)‖δo‖²` — **H is not provided by the source capture**, so J_lin carries no guaranteed-improvement claim [honest limitation].

### 3. Status tags
- "Six existing captures cannot evaluate the margin objective": each has one last-natural-token query per document, four sampled heads, Native frequencies at MrPro's common gain, all causal keys — no question queries, no target/distractor roles [已验证-audit of capture contents].
- Minimal information-theoretic obstruction: swap requested-answer meaning leaving all six captures identical ⇒ every geometry/KL/covariance value identical but desirable signed margin reversed ⇒ no deterministic selector of unlabeled summaries chooses right in both worlds [已验证-derivation; missing ROLE information at observed interface, not impossibility from richer data].
- J_lin acceptance-on-actual-outputs, labeled-capture requirement, family-disjoint held-out evaluation, cluster-resampling (position_overnight/report.py:88–104) [design; 假设 that estimator survives its own held-out single-layer validation — if it fails, "do not proceed as if it had identified a frequency mechanism"].
- Scratch vs frozen framing: `E_task L(θ*(ν),ν)` vs `E_task L(θ₀,ν)`; source Jacobians at θ₀ cannot identify scratch-optimal density; Cosh = assumption-dependent design prior for scratch regime [已验证-argument].
- Astra05's transport "becomes actionable only after role-sensitive modes are identified; this report supplies an observable alternative that does not need an unknown P" [positioning claim].

### 4. Proposed allocation rule (exact recipe)
(1) New labeled capture: source-window record tasks (family-paired ordinals + value swaps), raw Q / all K / V, correct single-token answer label, explicit T/H/O/B/Q roles, per-head output sensitivities `g=∂U/∂o`, plus reused natural full rows. (2) Fit shared ordered x=log ν (64 params) maximizing J_lin(x;d') s.t. `D_src(x)≤D_src(x₀)` and family-utility non-worsening, via constrained projected gradient with exact finite trig evaluation and trust-region acceptance. (3) Optionally replace J_lin by full-model `U_e(e^x; stretched prompt)` (frequency-only supervised fitting). (4) Freeze table; evaluate on family-disjoint source tasks AND actual 128K generations vs MrPro-at-identical-gain (operational Native separately). (5) It may legitimately output "no supported update, retain existing table." Without step-1 data, any returned numeric Qwen table = fabrication (author's word).

### 5. CPU numbers
1,000 random five-pair changes with signed distances to 131,072 and arbitrary finite old/new frequencies: exact finite score bound never violated; minimum slack (|score change| − bound) = −0.04345434 [已验证-numeric — verifies the bound formula, not calibration]. Rare-key comparison reproduces the ranking reversal: exact log mass 10.00285611 > second-cumulant 5.08417605 < constant-score-1 competitor 5.15888308 [已验证-numeric].

### 6. Obstructions / negative results
- Full second-cumulant covariance insufficient for finite softmax competition (rare [10,0,…,0] block vs flat block — `test_full_covariance_probe.py:69–80`) ⇒ generator must evaluate the actual partition.
- Raw/centered score MSE blind to row-common components: layer-0 MSE 1.305e−5 coexists with KL 7.196134; centered MSE 0.48446 (`core_query_diagnostics.json:4–16`, `FOLLOWUP_RESULTS_20260909.md:64`).
- Target-top1 preservation insufficient: target mass 0.50075→0.20339 under output KD while top1 kept (`FOLLOWUP_RESULTS:43–53`); attention-only and value-only errors vary by layer and don't add (lines 55–68).
- Local bounds don't imply generation recovery: PostMetric4 10/32, PreMetric4 11/32 vs Quest reference 13/32; native-pair geometric bounds did not imply recovery (`RESULT_20260908.md:64–99,112–129`).
- Gold-reference phase surgery can break already-correct reads when the query already contains the offset (`refcarry_audit/README.md:103–108`) — known target identity ≠ universally correct phase transform.
- Does NOT independently certify Smooth/E1/P2 numeric tables from parent reports (explicit scope discipline; the argument doesn't depend on them).

### 7. KKT relevance
Astra08 supplies the missing *data side* of the KKT program: F's far-capability term becomes measurable as g-weighted J_lin (output units, signed, per-head); F's near-damage term becomes the constrained compatibility divergence D_src with exact certificate bounds (ε_j margin non-flip, Pinsker); the feasible set is exactly the 16-DOF-equivalent interior (monotone ordered log-frequencies with pinned endpoints); the acceptance rule is hold-out on real behavior, i.e., the panel criterion itself. It is the only one of the four whose recipe can run today modulo one new capture.

### 8. Disagreements with established facts
None; it is the most conservative report. It agrees the root is never used, endpoints are constraints (pins them and demands declaration if moved), and refuses any proxy-substitution ("calling dominant keys 'correct targets' would repeat the project's proxy substitution").

---

## Cross-report consistency check vs established facts

| Fact | astra01 | astra05 | astra06 | astra08 |
|---|---|---|---|---|
| Root non-ranking respected | yes (criticizes first-zero as selector) | yes (root cited only descriptively) | yes (unused) | yes (unused) |
| Panel numbers (MrPro 87.22/78.13, Smooth far-collapse 68.3, etc.) | not contradicted; Smooth reversal used | not contradicted; reversal used to bar energy-min selection | reversal re-audited with geometry values | not contradicted (deferred certification) |
| Endpoints m=0/m=1 as constraints | anchors retained as constraints | candidates bitwise preserve both endpoints | endpoint restrictions by range, support caveat [Δ,1] flagged | endpoints pinned; moves declared |
| 16 DOF slots 24–39 | product-space label measure | all 29 candidates inside 24–39 | bin counts n_i (K=64 total) | full 64-vector ordered, endpoints fixed |
| transport= lens not definition | no water-bed language; KKT active set | transport is a QP with stated constraints | mass objective, exact DP | J_lin + constraints, no transport claim |
| EVQ = collision-density theory | recovered as constant-h special case | squared term = coherent nuisance only | discrete vs continuum Cosh distinguished | Cosh = design prior, not solved risk |
| b=10⁶ / K=64 / S=4 / W=32768→131072 grid | consistent | verified SHAs, gain 1.1386, Σm=29.3333 | generic K=64 check | d up to 131072, gain handling explicit |

One genuine internal divergence (for integration, not a veto): Astra01's nonconstant-h forced equation is `αρ''−βρ=λh''` while Astra06's is `αρ''−βρ=h''` with boundary `αρ'(1)=h'(1)` — different because Astra01 maximizes homogeneous SNR ⟨h,ρ⟩/√⟨ρ,Cρ⟩ while Astra06 minimizes the exponential-moment mass bound v/2−μ (no ratio normalization; λ absorbed). Both derivations are internally consistent; whichever F-term is adopted fixes which forcing law is correct.

## Lineage verdict (5 lines)
1. Beyond EVQ's collision kernel, the margin lineage adds **sign and role**: EVQ's `E[B²]` is recovered exactly only as the covariance of coherent nuisance against a protected constant signal (`h=h₀`, `C=K_exact`); the unsigned energy cannot rank (same-lag role reversal), so μ (positive source response) is a genuinely new F-term, not a rename.
2. It adds **interference structure**: the relevant v is a full covariance C (signed cross-frequency cancellation/reinforcement), with the sharp conditional that isotropic independent noise yields NO squared-kernel cost at all — EVQ's kernel is a modeling assumption about nuisance coherence, now falsifiable.
3. It adds **multiplicity and scale**: Astra06's `μ − v/2 ≥ log N + …` mass criterion fixes what SNR-style t misses (shrinking μ and v both can raise t while destroying attention mass) and ships an exact O(BK²) integer DP that allocates real K=64 channel counts with endpoints/occupancy as constraints.
4. It adds **mode-selectivity**: Astra05's `ν=(I−P_R)ω+S⁻¹P_Rω` opens an allocation direction (carrier-preserving, alternating-sign, sign-change under compression-only box) that the monotone m-ramp simplex structurally excludes — with a bitwise-verified 29-candidate table in slots 24–39 ready as exact replay hypotheses.
5. Verdict on usability as an F-term: **yes, conditionally** — role-conditioned margins are usable as F's two terms (near damage = coherent-nuisance variance v / D_src divergence; far capability = signed μ / g-weighted J_lin) with KKT solvers already existing (active-set inverse-C, DP, projector-QP), but the binding gap is identification: h,C,R are not obtainable from existing captures (Astra08's role-swap obstruction), so the next decisive action is the labeled source-window capture + first testing whether margin moments separate the existing MrPro/P2/BM/Smooth panel outcomes before any new table is proposed.
