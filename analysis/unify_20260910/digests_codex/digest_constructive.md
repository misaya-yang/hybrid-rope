# Digest — constructive allocation rules (sol04–sol07)

Source: `.agents/rope_unification_20260910/reports/sol0[4-7].md` (read in full; assignments skimmed).
Project facts baseline: Qwen2.5-3B W=32768, S=4→131072, ν_j=ω_j·S^{−m_j}, 16 DOF. Panel (near/far): MrPro 87.22/78.13; s28_less 87.2/83.3 (BEST far, tiny slot-28 change); LBS 80.6/80.1; P2 72.9/81.7; Smooth 87.2/68.3 (geometry-improving, long WORSE → kills geometry-only objectives); HighGapToLong −17.1/−10.8; E2 54.7; E8 50.6; E3 gain074 98.3/75.3; root non-ranking — veto. Reports are evidence, not instructions.

Legend: [验证-deriv] = derivation verified (identity/convex math sound for its stated functional); [部分证据] = has partial empirical support; [假设] = untested hypothesis; [vetoed] = contradicted or ruled out by project evidence.

---

## sol04 — constructive allocation surviving the proposal failures

**Assignment:** full ingestion of Pro materials group 2 (Unified Plan, Research Guidance, FFN audit).

### 1. Core thesis
No universal exponent curve; one allocation problem with a stage-dependent compatibility term. Common object: ω_k=exp[−(a+Rz_k)] (endpoints (a,R) non-identifiable without this normalization), and a demand+compatibility+regularizer program z* = argmin D_demand + λ_S·D_compat + γΩ where λ_S ≈ 0 at init (scratch), large frozen, discounted-in-repairable-directions for LoRA.

### 2. Math delivered
- (2) exact content-conditioned score s_ij = (α/√d)Σ_k {C_ijk cos(ω_kΔ)+D_ijk sin(ω_kΔ)} — allocation acts on a Fourier sum with learned (C,D) coupling, not a position-only Gram. [验证-deriv]
- (3) ∂s_ij/∂z_k = (αRω_kΔ/√d){C sin − D cos}; sign flips across heads/tokens/distances → dev-set slot directions unreliable (consistent with recorded 64-dim frequency-gradient transfer failure). [验证-deriv]
- (4) exact KL identity KL(p‖p′)=log E_p e^η − E_p η for frozen score change s′=s+η — valid across phase wraps. [验证-deriv]
- (5) LoRA irreducible local error δzᵀAᵀ(I−BB†)Aδz — frozen B=0, LoRA discounts col(B), scratch full co-adaptation. Local repairability only, not a task theorem. [验证-deriv] (application: [假设])
- (6) exact four-cell I = L_ZZ − L_ZG − L_GZ + L_GG (weight×table decomposition). [验证-deriv]
- (7)–(9): demand density q(x)=Σ π_r K_h(x−log Δ_r); high-rate squared-log-scale optimum ρ_scratch ∝ q^{1/3} (own CPU recheck of the calculus: minimize ∫q/(12ρ²) s.t. ∫ρ=K ⇒ ρ∝q^{1/3} ✓); anchored quantiles. Recovers EVQ-as-continuous-quantization and smooths MrRoPE radix demands. [验证-deriv for the stated functional; task-value link: [假设]]
- (11)/(12): convex program max bᵀq − (τ/2)Σ(q_{k+1}−q_k)² s.t. ½Σ f_k q_k² ≤ ε_N, order-preservation Q; τ=0 water-filling q_k(λ)=clip_{[0,1]}(b_k/(λf_k)) + λ-bisection + weighted isotonic projection. [验证-deriv — it is a real solver emitting one table]

### 3. Tags summary
Normalization/identifiability [验证-deriv]; geometry-only selection [vetoed] (its own counterexample 1 = Smooth_MrBudget pattern); "move all slow slots" [vetoed] (C_ijk varies across candidates, softmax does not row-cancel); permutation-decoupling [验证-deriv counterexample]; E1-slot-28-as-law [vetoed → reclassified as seed for sign of b_28/f_28]; (11) as behavioral winner [假设].

### 4. Constructive allocation rule (exact recipe)
**Frozen Qwen2.5-3B, 32K→128K ("compatibility-constrained radix motion"):**
- Inputs: native table ω_k⁰ (slot identity retained); declared relation-scale set R (radix place values + within-place neighborhoods) with mixture weights π_r (predeclared, NOT benchmark-tuned) → benefit b_k from coverage of the 32K→128K demand in (7); measured cost f_k = native compatibility curvature from exact attention KL (4) via symmetric small single-slot perturbations on FIXED native Q/K, fixed sampled layers/queries (if LoRA: f_k ← residual curvature from (5)); scalars ε_N (deployment contract, fixed BEFORE any long-task eval), τ.
- Steps: (1) compute b_k, f_k; (2) solve (11) by bisection on budget multiplier λ + weighted isotonic projection (order constraint); (3) deploy ω_k = ω_k⁰·4^{−q_k}; (4) freeze + hash the single table for every sequence length; (5) verify exact fixed-Q/K KL → complete-model output KL/task deltas → only then untouched 128K generation. No feasible table ⇒ failure of the demand/cost model at this checkpoint, not a theorem against static tables.
- Output m-profile: q ∈ [0,1]^K monotone under order constraint — generically a 0-prefix / interior-transition / 1-suffix shape (i.e., three-band-like emerges from water-filling + isotonic projection, not postulated).
**Scratch:** quantile rule (8)–(9) on the same demand q(x) — an initialization rule only; behavioral value requires matched training.
**Comparator (decisive, not EVQ-vs-MrRoPE labels):** native (no motion) vs standard MrRoPE q^MR vs rule-(11) table at identical endpoints/gain/decoder/frozen weights; never retune gain while evaluating allocation (FFN audit: historical Z/Y differed in amplitude — not a pure allocation contrast).

"Constructive allocation surviving the proposal failures" — survives: Smooth geometry-win/behavior-loss (via f_k term), low-frequency redundancy error, permutation coupling error, train/frozen sign flips (PPL 7.14→76.20 while effective rank 4.57→12.54 — reported in guidance), EOS/contains-answer proxy endpoints, single-slot overgeneralization. Generated candidates: exactly ONE table from (11) (explicitly "not a candidate grid").

### 5. CPU numerics
None computed in-report beyond the (8) derivation (reproduced ✓). Cited: PPL 7.14→76.20, rank 4.57→12.54 (guidance lines 30–34); E1 slot-28 dev-sample positive (treated as seed only).

### 6. Obstructions
Universal slot gradient cannot be estimated from small dev sets (sign-changing (3)); recorded 64-dim gradient non-transfer; KL is a valid local readout but a scalar gain cannot repair candidate ordering at fixed Q/K (lines 248–277); any allocator validated only by KL/rank/mass/EOS has not met the task endpoint.

### 7. KKT relevance
(11) IS the KKT problem; (12) is its water-filling solution: interior slots equalize marginal benefit/cost b_k/(λf_k) at common multiplier λ; box [0,1] + isotonic give bang-bang ends with graded bridge ⇒ three-band structure EMERGES (never imposed). The most explicit "marginal-equalization solver" of the four — but its inputs are proxies (see §8).

### 8. Conflicts with facts
- f_k from fixed-Q/K KL and b_k from radix coverage: E7 (local-output preservation failed as downstream selector, −9.5 pp long) and E8 (strong conditional target-mass score, −13.9 pp) show KL/mass-type curvatures do NOT rank behavior. sol04 demotes them to cost/constraint and keeps task as endpoint — partial mitigation, but water-filling q_k=b_k/(λf_k) still RANKS slots by proxy ratio. [部分风险]
- Does NOT predict Smooth should win (counterexample 1 is exactly Smooth); does NOT rank by root; handles s28 as a sign-seed only (consistent with 83.3 far). No conflict with panel; conflict risk is proxy-input validity, deferred to matched comparator 1 vs 3.

---

## sol05 — audit + corrected transported-row calibration rule

**Assignment:** Pro group 3 incl. newest 6Pro attachment; audit full-row block calibration, transport map, local+long preservation.

### 1. Core thesis
6Pro's full-row calibration is a material step above geometry-only proxies (uses signed content-conditioned Q/K coefficients + full softmax denominator) but is NOT yet a valid allocation rule: (a) a single full-row KL weights remote strata by native mass π_L≪1 — full-row coverage ≠ long-dependency coverage; (b) the fixed block map T(p)=SM⌊p/M⌋+(p mod M) with only the last query position aliases locality to one block origin. Corrected rule: constrained, stratum-balanced transported-row one-shot projected step from MrPro.

### 2. Math delivered
- (x₀, A, {a_i/A}) decomposition of log-frequencies: translation / support width / internal allocation; MrRoPE radix increments = added log-gaps a_i = a_i⁰+(m_i−m_{i−1})log S. [验证-deriv]
- EVQ change of variables h=φ′ exact; J[h]=½∫[α/h+β(1−u)²h]du ⇒ Cosh optimal FOR THAT FUNCTIONAL only (independently numerically reviewed). [验证-deriv (functional-scoped); task claim: [vetoed]]
- Ridge decomposition J_r(c₀,ν)=J_r(c_r*,ν)+(c₀−c_r*)ᵀ(G_r+ζI)(c₀−c_r*), ζ>0: representational fit vs cost of making an EXISTING checkpoint use the basis. [验证-deriv]
- KL chain rule KL(P‖Q)=KL(P_G‖Q_G)+Σ_g P_G(g)KL(P(·|g)‖Q(·|g)) — remote conditional geometry multiplied by group mass. [验证-deriv]
- Transport boundary algebra: T(p)−T(q)=SM(b_p−b_q)+(r_p−r_q); straddling pair: distance 1 → (S−1)M+1. [验证-deriv (CPU)]
- Corrected objective L_nat/L_locT/L_longT with conditional C_g + group-odds B_g terms; constrained QP direction. [假设 — untested as behavior predictor]

### 3. Tags summary
Mass-blindness counterexample [验证-deriv counterexample]; transport-map-as-scale-identity [假设, proposal self-admits]; current capture thinness (only q=32767, 4 heads/doc, same offset per row — code audit pro_block_calibration.py:43–46) [验证, from code]; corrected rule [假设]; frozen-hidden-states conditionality (new table changes earlier layers → later Q/K stale) [部分证据, acknowledged limitation].

### 4. Constructive allocation rule (exact recipe)
- Inputs: start at exact MrPro log-frequency vector x^M; full-key captures with signed Q/K coefficients + native exact & runtime logprobs + exact replay checks; stratified query offsets per fit doc/layer (immediately before/after each 4096 boundary, block middle, doc tail); heads rotated across ≥2 docs AND ≥2 offsets; maps = identity + block-stretch at ≥2 pre-fixed origins (current, M/2-shifted) as NUISANCE variables (report by origin; never average away a sign reversal); 4–5 fully held-out docs.
- Steps: (1) autograd gradients g_T=∇L_longT, g_N=∇L_nat, g_L=∇L_locT at x^M (exact sin/cos/softmax); (2) convex QP δ*=argmin g_Tᵀδ+(μ/2)‖δ‖² s.t. δ₀=δ_{K−1}=0, 1ᵀδ=0 (centroid held), g_Nᵀδ≤0, g_Lᵀδ≤0; (3) if no direction with g_Tᵀδ<0 → declared OBSTRUCTION: no local Pareto improvement over MrPro under these rows/maps — do not force a candidate by reweighting; (4) else normalize δ*, deterministic halving line search from the monotonicity limit for the largest η with exact nonlinear L_nat(x⁺)≤L_nat(x^M), L_locT(x⁺)≤L_locT(x^M), L_longT(x⁺)<L_longT(x^M), gaps >0; (5) build mirror x⁻=x^M−ηδ* (same η, shrink only for ordering) — matched reverse-direction control; (6) pre-task acceptance: held-out docs/origins show no protected-component increase beyond replay tolerance, separate (not summed) decreases in transported-long conditional KL and group-odds, no hidden head/layer regressions; (7) MrPro/x⁺/x⁻ at common gain on full prefills; paired local relations at both sides of a boundary + remote binding at controlled expanded distances; then dense 128K (added-key competition present).
- Output m-profile: one ±ηδ* perturbation of MrPro's m-curve (endpoints + Σx preserved — stays on the MrPro band face, interior bridge slots move).
- Anti-overfit discipline: ONE projected direction + deterministic line search; iteration only if held-out components agree in sign.

### 5. CPU numerics (reproduced ✓)
- Boundary stretch: (S−1)M+1 = 3·4096+1 = **12289** ✓.
- Remote-mass suppression: conditional KL of a flipped 90/10 split = 0.9·ln9+0.1·ln(1/9) = **1.7578 nats**, ×π_R=10⁻³ ⇒ **0.001758 nats** to full-row loss ✓.
- Boundary-crossing probability for distance d<M under randomized origin = d/M; expected mapped distance Sd but mixture {d, (S−1)M+d} — expectation-matching inadequate for periodic features ✓ (argument sound).

### 6. Obstructions
- Full-row KL is mass-blind to low-mass remote strata (decisive as stated).
- Capture as run is far thinner than "all heads, full rows" suggests.
- Frozen-state conditioning: exact only given captured hidden states; calibration nominates a direction, only full prefill tests survival.
- Natural text cannot identify WHICH low-mass remote keys carry task dependency — target is "preserve native behavior under coordinate transport," not "create long-range capability."
- Gain non-orthogonality: result valid only at declared gain.

### 7. KKT relevance
QP KKT multipliers on g_Nᵀδ≤0, g_Lᵀδ≤0 = first-order Pareto filter, not marginal equalization across slots — the direction maximizes transported-long descent per unit norm subject to zero first-order degradation elsewhere. No three-band implication by construction (it inherits MrPro's bands and moves interior bridge gaps under 1ᵀδ=0, δ₀=δ_{K−1}=0).

### 8. Conflicts with facts
None with the panel: anchored AT MrPro, protects local (Smooth-proof via separate identity-map terms), treats geometry as diagnostic. Its transported-long KL/odds components are exactly the E7/E8-class preservation proxies — sol05 concedes they are direction predictors requiring full-prefill behavioral acceptance, which is the honest posture the panel demands. Compatible with s28 (a small bridge-gap edit is within its output class).

---

## sol06 — RTGA: relation-targeted gap allocation (QP with actual constructive solution)

**Assignment:** audit subspace transport, softmax harmonics, gap-budget theories; unify with a real solution, refuse proxy→performance jumps.

### 1. Core thesis
There IS a clean common allocation variable (extra log-gap budget) but NO evidence-grounded common proxy objective across regimes. Evidenced feasible face for frozen Qwen: m_j=0 (j≤23), m_j=1 (j≥40), monotone between ⇒ transition gaps e_i≥−a_i⁰ with Σ_{i=23}^{39} e_i = log S — a polytope parameterization, not an objective. Replace proxy ranking with RTGA: hit measured mixed-frequency relation targets κ_n=S^{−t_n}κ_n(ω) via one constrained QP + a signed task-risk term c that is structurally absent at scratch and present when frozen.

### 2. Math delivered
- κ_n(ν)=nᵀν appear in the exact exponential-softmax expansion with content-dependent Bessel coefficients (cited derivation). [验证-deriv]
- Linearization κ_n(xʳ+v)≈κ_n(xʳ)−a_nᵀv, (a_n)_j=n_jν_jʳ — off-diagonal a_n a_nᵀ coupling for difference modes (adjacent smoothing cannot reproduce it); matches subspace graph coupling Σ‖P_ij‖²(m_i−m_j)² independently. [验证-deriv]
- Closed form v*=(AᵀWA+ζQ)⁻¹(AᵀWb−c) pre-inequalities (ζ>0, AᵀWA+ζQ≻0); with polytope F: convex QP, global optimum, active-set/KKT yields the finite table. [验证-deriv — genuine solver]
- Exact mixed-frequency retiming identity ⇒ hard native prefix + common /S suffix is an EXACT special case: within-block relations hit their clocks exactly; only cross-block relations enter the bridge QP. [验证-deriv]
- MGDA min-norm convex combination of per-cell gradients as signed direction when multiple short/long calibration cells conflict. [验证-deriv (standard); application 假设]
- Counterexample 1 (decisive): NO nonnegative combination of the audited geometry costs is a sufficient selector — Smooth has lower unweighted unresolved exposure, lower local distortion, lower Q/K-weighted operator MSE at every lag range, and lower Q/K-weighted unresolved response at every cutoff in ALL 36 layers, yet worse 128K. Positive weights / more layers cannot repair it. [vetoed class: geometry-only; 验证-deriv argument]

### 3. Tags summary
Gap-budget polytope [验证-deriv]; D_j=W·S^{m_j} as hard failure boundary [vetoed → risk feature] (MrPro successes beyond danger band in row evidence); arc-clock/filter-bank dichotomy as a binary law [vetoed as law → 假设] (slots 28/29 complete ~12.37/9.97 native turns yet have conditional interventions — defeats one-turn account); mode 28−2(29)+30 ~70K period exists but relative amplitude ranges 0.00049 (amp .3) → 0.294 (amp 2) — low |nᵀν| alone is not a cost [部分证据, cited]; Cosh as frozen target [vetoed] (deployment τ failed to select the best nearby point in recorded multi-seed comparison); Cosh as scratch regularizer [部分证据]; universal displacement-1 preservation + extrapolation compatibility [vetoed] (exact conflict, cited lines 123–134); RTGA behavior [假设].

### 4. Constructive allocation rule (RTGA, exact recipe)
- Inputs: native ω and reference xʳ (e.g., MrPro); polytope F = {order, endpoints, 0≤m≤1, Σe_i=log S over bridge 23→39}; finite interaction set {n} = per head/layer/row LARGEST measured Bessel/Fourier coefficients (or low-order set with certified coefficient-mass tail) from native conditional content on a frozen, benchmark-isolated calibration corpus; target clocks t_n∈[0,1] labeled by whether INTERVENTION on that relation improves declared short/long self-supervised risk (NOT turn count; conflicts → continuous t_n or dual membership); weights w_n; c = full-model signed gradient of declared frozen calibration losses w.r.t. x INCLUDING changed prefixes and readout (detached Q/K norms provably insufficient, cited general-allocation derivation 74–112); multi-cell → MGDA; Q = source-subspace commutator/transport quadratic as STABILIZER ONLY.
- Steps: (1) build A, b; (2) solve ONE QP (closed form if unconstrained; active-set/KKT with F); (3) evaluate EXACT nonlinear residuals κ_n(ν)−κ_nᵀ...κ_nᵗᵃʳ afterward; relinearize only if needed; (4) deploy +v* and mirror −v* (norm- and endpoint-matched) vs MrPro and strongest P2/E1 baselines; report exact nonlinear relation residuals, short-window calibration risk, 32K/128K tasks, independent holdout. Geometry = diagnostics only.
- Output m-profile: v* added to reference gaps; when targets separate, structure = native prefix (m≈0) / coupled bridge QP / common retimed suffix (m≈1) — three-band DERIVED in the separated case, plus exact special case = hard prefix + uniform /S suffix.
- Scratch regime: min_{W,x} R_train(W,x)+λJ_rep(x) with EVQ gap functional as regularizer; c=∇_x R (learned implicitly by data); equal training exposure/update sets mandatory for family comparisons.

### 5. CPU numerics
No new computation; re-derives/quotes: slot-28/29 native turns 12.37/9.97 (cited from extrapolation review:51–69); mixed mode 28−2(29)+30 ≈70K period, amplitude 0.00049→0.294 across content amplitudes .3→2 (cited, not independently reproduced here); budget identity Σ_{i=23}^{39}e_i=log4.

### 6. Obstructions
Explicit claim boundary: the QP proves a solution EXISTS for stated finite targets — it does not prove the targets improve capability; universal interaction weights and bridge width are unidentified (must be measured); exact preservation vs extrapolation conflict means the residual conflict is exposed only through the QP value; Smooth result forecloses any nonneg geometry selector.

### 7. KKT relevance
Highest of the four in formal terms: unconstrained closed form + polytope active-set are literally KKT; the budget equality Σe=log S carries a multiplier equalizing marginal relation-residual cost across gap coordinates; box/order constraints give complementarity at m=0/m=1 (bang-bang prefix/suffix). Three-band: DERIVED for the separated case, EMERGENT otherwise.

### 8. Conflicts with facts
None found — sol06 is the most panel-consistent report: it explicitly reconciles Smooth (counterexample 1), HighGapToLong/E2/E8 (feasible-face restriction to tested m-profiles, not universal law), s28 (bridge slots 24–39 exactly where RTGA moves), root non-ranking (relation set measured, not ranked by root). Tension: c requires full-network signed gradients (GPU-ish) and t_n requires per-relation interventions — the recipe is the most input-hungry of the four.

---

## sol07 — nongeometric screen audit + marginal-decision-utility allocation rule

**Assignment:** full audit of experiments/nongeometric_screen source + result reviews; determine which measurable computations explain winner/loser ordering; separate implementation numerical failure from theory failure.

### 1. Core thesis
The screen supports NO geometry-only rule. The computation that best explains a REAL win is the signed answer-decision margin under the actual frozen model, decomposed into prefix-formation and readout contributions: nearly-additive continuous margin changes cross a discrete greedy-decoding boundary. Rule: allocate log-gap mass by cross-fitted marginal decision utility (donor→recipient transfers), recomputing after every edit; EVQ = scratch prior only; gain = independent amplitude coordinate.

### 2. Math delivered
- Decision margin M_t(ν,g)=ℓ_t(y*;ν,g)−log Σ_{c∈C} exp ℓ_t(c;ν,g); finite-candidate ΔM on gold-prefix trajectories (correct sign, includes values, output projection, later layers). [部分证据 — currently explains ONE winner case; predictive ordering across unseen examples untested]
- Factorial prefix/read decomposition P=M(a,0)−M(0,0), R=M(0,a)−M(0,0), I=M(a,a)−M(a,0)−M(0,a)+M(0,0). [验证-measurement]
- Utility: Û_a=Q_{0.2,e}[Σ_t w_{e,t}ΔM]−λ_N[ΔNLL−ε_N]₊−λ_S Regress_a−λ_B BF16Mismatch_a. [假设]

### 3. Actual experiment audit table (reproduced from report; all Qwen3B numbers dev-evidence unless marked transfer)
| Intervention | Outcome | Reading |
|---|---|---|
| E1 slot 28, predecessor exponent | 32K tie; 128K 78.125→**83.333%**, 2W/0L | Local headroom; margin explained by prefix+readout |
| E1 slot 29, successor exponent | 32K **+8.333 pp**; 128K **−0.208 pp**, 2/1 | Short gain = one QA row; long VT gain & multiquery loss cancel |
| E1 pair 28+29 | 32K tie; 128K **−4.167 pp**, 0/1 | Individual gains do NOT compose ⇒ recompute after every edit |
| MrPro gain .074 | 32K 98.333%; 128K 75.347%, 4/2 | Gain = amplitude, large length-dependent effect (= panel E3 98.3/75.3 ✓) |
| BM gain .074 | 32K 100%; 128K 70%, 6/4 | vs same-gain MrPro: table +1.667 short / **−5.347 long** |
| FullLagP2 gain .074 | 32K 72.917%; 128K 81.667% | Long QA/VT gain, multikey/FWE cost; +3.542 long / −14.306 short vs official MrPro; sharp gap at 29, high gaps ~unchanged ⇒ no EVQ-style high-gap donation |
| E7 local projection | 32K +2.778 pp; 128K **−9.514 pp**, 6/7 | Local-output preservation FAILED as downstream selector |
| E8 zero slot 51 | long subset **−13.889 pp** | Strong conditional target-mass score didn't predict generation |
| LongBridge slower/faster | slower −6.667/+1.944; faster 0/−4.167 | Direction matters; slower = VT-specific with FWE cost (= LBS panel row region) |
| Smooth(MrBudget) | 32K tie; 128K **−9.792 pp** | Min roughness at fixed budget insufficient (= panel 68.3 ✓) |
| HighGapToLong | **−17.083 / −10.764 pp**, 0/7 | Donation quantum + recipient construction fails; EVQ training theory untested |
| Layer/group/E9/E10 | 12-row ties | Non-results for ranking; never mix 12-row with 36-row macros |

Frozen cross-model slot-28 TRANSFER is heterogeneous: Qwen7B 32K tie, 128K −1.667 pp (1/1); OLMo1B 4K −3.333, 16K +3.819 (4/2) from a low baseline ⇒ checkpoint-relative recalibration is the defensible hypothesis; no completed independent holdout exists.

### 4. What explains ordering / what doesn't (rule-relevant math findings)
- Slot-28 multikey 4-cell margins Mr/Mr, Mr/E1, E1/Mr, E1/E1 = −2.125, −1.125, −1.000, **+0.250** nats; prefix +1.125, readout +1.000, factorial remainder +0.250 (CPU recheck: −2.125+1.125+1.000+0.250=+0.250 ✓; report's P/R labels cross the two column orders but sums are consistent). Strongest current mechanistic measurement.
- Old conditional selector (select.py:24–68, mass-held reconstruction): nominated BOTH E1 winners but ALSO E2/E8 losers, layer/group ties; cannot explain the failed 28+29 pair ⇒ filter, not objective.
- NLL: sub-millinat shifts in E1 wins; guard, not selector. Chord separation: periodic/nonmonotone; slot 28 WINS while reducing it at 3 of 5 audited distances ⇒ [vetoed] as selector. Smooth: geometry up, task down ⇒ [vetoed]. HighGapToLong: endpoints preserved, declared budget moved, loses everywhere ⇒ geometry+budget = descriptors/constraints, not utility.
- **E7 precision separation (the implementation-vs-theory call):** linear predicted NMSE 7.9366844e−8; ideal finite relative-phase 7.9364028e−8; FP32 absolute-coordinate 7.9325473e−8; BF16 absolute-coordinate 1.2807392e−5 ⇒ BF16 ≈ **161× squared error = 12.7× amplitude** (CPU recheck: 161.4 / √=12.70 ✓). Derivative correctly predicted the IDEAL intervention; BF16 arithmetic broke deployment AND the protected local object was separately insufficient downstream — two independent failures cleanly separated.
- select.py underflow bug: old_probability·exp(δ) could underflow; log-weight normalization (select.py:49–58) repaired it and moved the E5 runner-up from layer 27 → 32; layer-27 records = valid outcomes of an invalidly ranked proposal (selector implementation failure, not result corruption).

### 5. Constructive allocation rule (exact recipe)
- Inputs: target-checkpoint MrPro + validated endpoints + gain; gold-prefix calibration examples grouped by task × physical relation distance, with answer token y* and competitor set C; COMPLETE forward passes on a small grounded set of adjacent transfers (conditional replay only pre-orders); ε_N, λ_N, λ_S, λ_B predeclared; folds A/B select, fold C sign confirmation.
- Output representation: monotone gap table x_j=log ν_j − log ν_{j+1}≥0, Σx_j=R; a transfer a=(i→j, η) moves η from donor gap i to recipient j, reconstruct cumulatively.
- Steps: (1) form FINITE donor→recipient transfers sized by an existing adjacent exponent step or a declared physical phase quantum (no arbitrary grids); (2) rank by cross-fitted Û_a (lower 0.2 quantile of weighted ΔM — one threshold crossing cannot dominate; minus NLL-violation hinge, minus Regress on baseline-correct examples, minus measured BF16 mismatch) — NOT target mass, NLL, chord, roughness; (3) apply the best positive transfer, RECOMPUTE (28+29 non-additivity forbids composition); (4) stop when no transfer has positive lower-confidence utility; (5) freeze; test untouched examples.
- Output m-profile: MrPro face + a sequence of small validated single transfers (s28_less is exactly one manually-run instance).
- From-scratch arm: J_train(x,g)=min_W E L(W,x,g)+ρΩ_EVQ(x); Cosh = initialization/regularizer only; "at an interior constrained optimum, active gaps equalize marginal training value, with KKT inequalities at zero gaps."

### 6. Predictive accuracy of the rule against the panel rows (per team-lead request)
- s28_less 83.3 far BEST: rule's margin utility explains it prospectively-consistent (+5.208 pp, 2/0) ✓; the Q_{0.2} + recompute design also explains why naive combination FAILED (28+29 pair −4.167) — an additive-utility rule would have predicted a double win, so this is the rule's sharpest qualitative edge over any linear proxy ✓.
- Smooth −9.792 long: predicted non-winning (utility contains no geometry term) ✓ — no conflict, unlike every geometry-ranked rule.
- P2 81.7 far / 72.9 near: Regress/short-task term predicts the −14.3 pp short penalty the far-only view hides ✓ (retrospective).
- HighGapToLong −17.1/−10.8: large-quantum multi-slot donation; lower-quantile utility with per-edit recompute would have rejected it ✓ (retrospective).
- E2 54.7 / E8 50.6 losers: mass/target-score selector failures — rule REPLACES that selector (self-reported failure of its own predecessor) ✓.
- E3/gain074 98.3/75.3: gain held out as separate amplitude coordinate ✓ consistent.
- HONEST CAVEAT: full forward-pass margin ordering across unseen examples is UNTESTED — the rule currently explains 1 winner mechanistically and the remainder by construction; sol07 pre-registers exactly the fold-A/B/C experiment to falsify itself ("If ordering holds…" / failure ⇒ fall back to coarser-gap-block full-forward loss, never geometry-only).

### 7. KKT relevance
The greedy loop IS discrete complementary slackness on the gap simplex: apply transfer while marginal decision utility > 0, stop when no positive lower-confidence utility remains (KKT inequality at zero transfer); scratch statement is literal marginal-equalization-with-KKT-inequalities. Three-band: NOT implied — profile is whatever transfers survive; MrPro face is the starting point, so structure is inherited + locally patched (s28 = single-slot bridge edit).

### 8. Conflicts with facts
None with the panel — it is the only report written entirely FROM the panel rows. Root non-ranking ✓ (select.py audit concedes ranking failure); Smooth ✓; s28 83.3 is its central evidence, not ignored ✓. Its risks are internal: margins are on gold prefixes (teacher-forced, state-specific — the same frozen-state objection sol05 raises, partially answered by sol07's insistence on complete forward passes and free generation as endpoint), and the 2-win/0-loss s28 cell is tiny-n paired development evidence.

---

## Lineage verdict (5 lines)
1. Closest to a KKT solver: **sol07's greedy marginal-decision-utility transfer rule** (discrete complementary slackness on Σx=R, equalize-marginal-value stated for scratch) — sol06's RTGA is the most FORMAL QP/KKT object and sol04's (11)/(12) the cleanest water-filling equalizer, but sol07 is the only rule whose equalized quantity (ΔM) uses an input that has already explained a real panel win.
2. Required input measurement: teacher-forced answer-decision margins ΔM from COMPLETE forward passes on a grouped (task × distance) gold-prefix cohort with A/B/C folds — vs sol04's fixed-Q/K KL curvature f_k (E7/E8 evidence says KL-class proxies mis-rank) and sol06's Bessel-coefficient relation set + full-model signed gradient (heaviest, untested).
3. Three-band: sol06 derives native-prefix/bridge/retimed-suffix in the separated case; sol04 gets it emergently from water-filling + isotonic projection; sol07 inherits MrPro's bands and single-slot-patches them (s28 is its executed instance).
4. All four reports independently veto geometry-only selection and honor Smooth (87.2/68.3) and HighGapToLong; none ranks by root; none ignores 83.3 — the lineage is consistent with the panel, the disagreement is proxy-input validity (KL/mass vs margin).
5. GPU distinguishability from MrPro is near: sol07's pre-registered A/B/C margin cohort is a few forward passes, and its next table pick would face MrPro (and s28_less, whose 83.3 it already claims to explain) at common gain; sol04's (11) arm and sol06's ±v* mirror each need one calibration run — a single 32K/128K matched-gain task row separates the three candidate m-profiles from MrPro without waiting for training.
