# First-principles theory memo: RoPE post-hoc extrapolation for mature frozen checkpoints

- **Date:** 2026-09-02
- **Status:** PARTIALLY RETRACTED THEORY-ONLY synthesis; working history, not a
  proof owner; no experiments planned, no code, no manuscript text. Supersedes
  no owner; historically audited
  [`COMMON_DIRECTION_FEASIBILITY_AND_BASIN_BARRIER_THEORY_20260902.md`](COMMON_DIRECTION_FEASIBILITY_AND_BASIN_BARRIER_THEORY_20260902.md)
  (see its PARTIALLY SUPERSEDED block and §G.13 here).
- **Method:** three isolated first-principles derivations (exact algebra /
  scale continuation / task circuits) plus one adversarial attacker; this memo
  keeps only intersections supported by independent derivation, or propositions
  whose disagreement is explained (§G). "Multiple agents agreed" is never used
  as evidence; every load-bearing item below was re-derived or owner-checked
  by the synthesizer.
- **Convention:** all numbers carry owners in §I. Native window `L`, target
  scale `S` (deployment on `[0, SL]`), native geometric spectrum
  `ω_k = b^(−2k/d) = b^(−k/K)` (`K = d/2` slots, slot 0 fastest).

> **Controlling audit correction (2026-09-02):** this memo is working theory
> history, not a canonical proof owner. Fact D (38-row “16K Hotpot”) is invalid
> for claim/gate use and Fact E lacks a recovered raw owner. T4's “conditioning
> exactly S” divides upper bounds and is false as stated; T5's arbitrary-epsilon
> checkpoint construction and unrestricted off-arc intersection claim are not
> proved; T7 contains an incorrect novelty ratio. Later sections that rely on
> those items—including task radius, disconnected-basin, non-identifiability,
> and method-class conclusions—are superseded by this correction. Retain only
> explicitly scoped exact identities and the exact transplant/compatibility
> results after independently checking their assumptions.
>
> **T1 proof correction (2026-09-03):** its arc-length step treated a vector
> of per-slot scalings as one scalar and did not prove uniformity. The
> conclusion survives only after first equating the arcs' tangent rays and then
> comparing their lengths. The corrected proof is owned by
> [`ROPE_OPTIMALITY_IDENTIFIABILITY_AND_CONDITIONAL_EQUATIONS_20260903`](../../foundations/ROPE_OPTIMALITY_IDENTIFIABILITY_AND_CONDITIONAL_EQUATIONS_20260903.md#45-correction-to-the-historical-pi-arc-proof).

## 0. Inputs used at the time (not all remain valid)

**A — intervention validity.** Old custom-attention-wrapper gradient
conclusions are void (failed z=0 parity). The current intervention replaces
only native RoPE `inv_freq`. At z=0 it reproduces full-vocabulary logits
6/6 checkpoints, max error 0.

**B — ordered coupling.** Same final frequency multiset permuted across rotary
slots collapses OLMo/Qwen. The learned object includes the ordered pairing
rotary-subspace ↔ frequency, not the unordered spectrum.

**C — the working retrofit.** Frozen `log_s4`: `ω′_k = ω_k·4^(−m_k)`;
request-static, pure-z, no routing, no weight training, no KV change.
Passes the 1x PG-19 retention gate at `0.875302`. (Terminological debt,
flagged once: "pure-z" is the inherited wrapper-era name; the realized
intervention is fact A's `inv_freq` replacement.)

**D — INVALID assay input (constructed HotpotQA, 38 Native-short-correct
samples; do not use).**
8K EM/F1: Native `0/.002`, log_s4 `.605/.677`, YaRN-4 `.579/.656`
(diff `+.0215`, CI `[−.161, .202]`, unresolved). 16K EM/F1: Native `0/0`,
log_s4 `.079/.153`, YaRN-4 `.395/.500` (diff `−.3466`, CI `[−.496, −.198]`,
log_s4 clearly loses). 16K correct-EOS: Native `0/38`, log_s4 `22/38`,
YaRN `36/38`. The same log_s4 is strong on RULER-13 at 16K (`0.49859` vs
official YaRN-4 `0.1056`) and beats YaRN there. These recorded numbers have no
recovered raw owner and come from a short-correct, constructed-filler stress
with a mechanical prompt-tail boundary. They establish no Hotpot benchmark,
task radius, gate outcome, or comparison to RULER.

**E — unverified gain-sweep input.** Gain sweep on log_s4 (Native-short F1 `.796`):
`g=.9→.036`, `1.0→.588`, `1.05→.608`, `1.1026→.680`, `1.15→.692`,
`1.2→.641`, `1.3→.439`. `g=1.15` frozen by 2Wiki short-only criterion fails
Hotpot; unit gain is more reasonable on the joint evidence. Finite interior
optimum; task/checkpoint instability. The raw owner was not recovered. These
points may describe the session but cannot establish non-universality, a
threshold event, or a gate until the executed protocol and rows are verified.

**F — retired route.** The 18-sample → 64-D pure-z behavioral-gradient
direction improved dev margins but failed the unopened holdout and is exited
from the main line. Nothing below premises "margin-gradient is the only
allowed next move."

### 0.1 Provenance of `m_k` (independently re-verified this session)

The working movement mask is a deterministic function of `(L, Ω0)` alone,
constructed by `scripts/analysis/export_uniqueness_budgeted_tables.py`
(hash-pinned tables):

1. causal pair-count distance measure on the Native window: weight
   `∝ (L − Δ)`, no content, no task labels, no long-context data;
2. per-slot conditional uniqueness `u_k ∈ [0,1]`: leave-others-out regression
   of slot k's window-weighted phase curve `(cos ω_kΔ, sin ω_kΔ)` onto all
   other slots' phase curves; `u_k` = residual energy fraction (the part of
   slot k's Native-window phase curve not linearly reconstructible from the
   others);
3. movement `m_k = (1 − normalized u_k)^2`; arithmetic-era rule
   `ω′ = ω(1−m) + (ω/s)m`; the log law reuses the same frozen mask as
   `ω′ = ω·s^(−m)`.

**The split is true at the object level and false at the meta level**
(attacker-verified, P1-1):

- *Genuinely Native-only:* the 64 slot values (above); the two-parameter
  `G_4` compression (fit to the frozen movement vector only; data boundary
  recorded in the CPU owner); the gain `c = 0.074` (selected on the 1x PG-19
  retention boundary only); the s2 zero-refit table and the OLMo→Qwen 64-point
  transport, which are real out-of-sample successes of the geometry-only
  stage (s2 retention `0.9840/1.0412`, transport `0.6725` inside the
  pre-registered parity band).
- *Contaminated at the model-selection level:* the mask exponent was chosen
  against long outcomes (`LENGTH_CONDITIONED_BUDGETED_RETROFIT_RESULT_20260822`:
  `p=2` retained over `p=1` because it scored better at 8K RULER, `0.5825`
  vs `0.5525`); the log law itself was chosen against long outcomes
  (`SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831` §3: log beats arithmetic at
  2x/4x/8x core-4); the amplitude family `1 + c·log(s)` is YaRN's.

Precise statement that survives: **log_s4 is a design family (uniqueness
budget rule + exponent 2 + exponent-space embedding) selected with long-range
outcomes; conditional on the family, the per-slot allocation and the G_4
compression are deterministic functions of Native geometry; the gain was
1x-selected.** Every "identifiable from Native data alone" sentence in this
memo carries that asterisk.

One more structural fact: official YaRN-4 is likewise a Native-only
construction (base + scale only), yet log_s4 and YaRN-4 have inverted outcomes
on RULER-16K vs Hotpot-16K (fact D). Two admissible Native-only
constructions with opposite long-task outcomes is the empirical shadow of the
non-identifiability theorem (T5).

## A. Exact identities

**A1 — score decomposition.** At any attention layer, with the retrofit
changing only `inv_freq`,

    s_ij(Δ) = Σ_k Re[ c_k(x_i, x_j) · e^{iω_kΔ} ],   Δ = i − j ≥ 0,

with `c_k = z^q_k · conj(z^κ_k)` the complex product of the rotary-projected
query/key components. At the first attention layer `c_k` is exactly frozen
(weights and embeddings unchanged). At deeper layers `c_k` co-moves with the
table through activations — it is frozen as a function of the input, not as a
number. Everything below that uses "frozen c_k" applies strictly to a fixed
input (or to layer 1); the input-averaged statements carry this implicitly.

**A2 — zero-lag invariant.** `s(0) = Σ_k Re c_k` is independent of the table.

**A3 — mean-power identity.** For distinct frequencies the Besicovitch mean
power over lags is `½Σ_k |c_k|²`, independent of the table and of the
slot-frequency pairing. **Correction to the invariants report:** if a retrofit
*merges* two slots onto one frequency, mean power becomes
`½|c_1 + c_2|²` on the merged line, which is **not** bounded above by
`½(|c_1|² + |c_2|²)` — aligned coefficients raise it (e.g. `c_1 = c_2 = 1`:
`2 → 4`). Mean power is invariant under table changes that keep frequencies
distinct; merging can move it in either direction.

**A4 — derivative identities.** For `n ≥ 1`,
`s⁽ⁿ⁾(0) = Σ_k Re[ c_k (iω_k)^n ]`; every derivative of order ≥ 1 depends on
the table. Together with A1's real-analyticity in Δ this gives:

**A5 — Fourier injectivity.** The score function `s(·)` on any nontrivial
interval determines the complex spectral measure `ν = Σ_k c_k δ_{ω_k}`
bijectively (a trigonometric polynomial known on an interval is known
everywhere; with finite K, Prony/character-averaging recovers `(ω_k, c_k)`).
Consequence: at fixed content, retrofitting is exactly the operation of moving
the support of ν while freezing its weights.

**A6 — per-slot waterbed identity (log family).** For `ω′_k = ω_k S^(−m_k)`,
the novelty ratio (deployed phase range / Native phase range per slot)

    r_k(S) := ω′_k·SL / (ω_k·L) = S^(1−m_k)

satisfies `r_k(S) · S^{m_k} = S` exactly. `m_k = 0`: pure extrapolation
(`r_k = S`, all novelty); `m_k = 1`: full position interpolation
(`r_k = 1`, no novelty). This is an identity, not a trade-off model; the
trade-off enters only once a cost is attached to novelty and to blur
(§C, T7).

**A7 — composition identity.** For a frozen mask,
`ω(s₁s₂) = ω(s₁)·s₂^(−m)` (repo audit, hash-owned). Formula-level only; see
T6 for why this carries no behavioral content.

**A8 — universal confusability.** Two lags Δ, Δ′ are confusable by all
contents (all coefficient realizations spanning `ℂ^K`) iff
`ω_k(Δ − Δ′) ∈ 2πℤ` for every k. For a geometric spectrum with
`b^{1/K} ∉ ℚ` (verified for OLMo `b = 5e5, K = 64` and Qwen
`b = 1e6, K = 64`), this has no solution with `Δ ≠ Δ′` — no exact
confusability at any finite lag. (Approximate, content-specific confusability
is governed by the window Gram matrix, not by this exact identity.)

**A9 — Native-computable uniqueness diagnostic.** With the causal pair-count
measure `w(Δ) ∝ (L − Δ)` and design rows
`(cos ω_kΔ, sin ω_kΔ)_k`, the leave-others-out residual fraction `u_k` (0.1)
is an exact function of `(L, Ω0)`. It is the diagonal of a leave-block-out
coherence profile of the window Gram matrix — the Native-side currency in
which "how much slot k can move for free" is priced.

## B. Theorems (stated with full assumptions)

**T1 — PI uniqueness, curve level.**
*Assumptions:* retrofit acts by per-slot scalings `ω′_k = ρ_k ω_k`,
`ρ_k ∈ (0, 1]`; the joint phase arc `γ(t) = (ω_k t) mod 2πℤ`,
`γ′(t) = (ρ_k ω_k t) mod 2πℤ`; injectivity of `t ↦ tω mod 2πℤ` on the
relevant ranges (sufficient: one pair of frequencies with irrational ratio —
holds for OLMo/Qwen adjacent slots; Qwen has distant relations
`ω_{k+32} = ω_k/1000`, irrelevant here); finite arcs, not closures.
*Claim:* `γ′([0, SL]) = γ([0, L])` as subsets of the torus iff `ρ ≡ 1/S`.
*Corrected proof:* both arcs are embedded and share the origin. Set equality
first makes their tangent rays there identical, so `ρ ⊙ ω = cω` for some
`c>0`. Since every `ω_k>0`, `ρ_k=c` for all `k`. Only then may arc length be
compared: `cSL‖ω‖=L‖ω‖`, hence `c=1/S`.
*Edge cases:* (i) closure-equality instead of arc-equality is vacuous — with
an irrational ratio both closures are the same subtorus for every ρ;
(ii) rational spectra (e.g. integer bases with small K) lose uniqueness at
finite arcs once `L` exceeds the common period; (iii) the set-equivalence has
zero sup-norm robustness: an ε-arc perturbation can move ρ by O(ε)-relative
amounts; (iv) operational form (attacker P2-4): the requirement for no new
phase exposure is arc *containment* `γ′([0, SL]) ⊆ γ([0, L])`; the same
origin-tangent argument first forces a uniform `ρ=c`, and endpoint/length
comparison then gives `c ≤ 1/S`; the theorem's equality `ρ = 1/S` is
the boundary case where the full deployed interval is used. Mod 2π creates no
additional finite-arc equivalences under the injectivity assumption. *Attribution correction:* no Lindemann–Weierstrass is needed for the
curve-level theorem; the operative number-theoretic condition elsewhere is
`deg_ℚ(β) > K − 1`, and for `b = 10000` one actually has relations
`ω_{k+16} = ω_k/10`, so closure-level non-uniqueness there is real.

**T2 — transplant rigidity.**
*Assumptions:* frozen, content-independent, invertible Q/K reparametrization
acting within the RoPE sandwich form:
`A^T R(ω′Δ) B = R(ωΔ)` for all Δ, with `R` the block-diagonal rotation.
*Claim:* the frequency multisets coincide up to sign: `{|ω′_k|} = {|ω_k|}`.
*Proof (attacker P2-1 simplification):* at Δ = 0 the identity gives
`A^T B = I`, hence `R(ω′Δ) = (A^T)^{−1} R(ωΔ) A^T` — the two rotation
families are similar for every Δ. Equality of spectra
`{e^{±iω′_kΔ}} = {e^{±iω_kΔ}}` for all Δ forces multiset equality directly.
Consequence: no static Q/K map can move a slot's frequency; fact B
(permutation collapse) is consistent with — and stronger than — this, since
even keeping the multiset and permuting the pairing collapses behavior.

**T3 — compatibility modulus.**
For a table change `ω → ω′` at fixed coefficients,

    |s′(Δ) − s(Δ)| ≤ Σ_k |c_k| · min(2, |ω′_k − ω_k|·Δ),

tight over coefficient assignments of the given moduli (choose phases to align
all terms). Softmax propagation with per-logit error ≤ ε:
`‖p′ − p‖₁ ≤ e^{2ε} − 1` (derivation: `p′_i/p_i ∈ [e^{−2ε}, e^{2ε}]`; the
invariants report's `2(e^{2ε} − 1)` is valid but slack).
*Corollary (weight-blind limitation):* a retrofit diagnostic that uses only
`{|c_k|}` or `{|c_k|²}` (diagonal energies) cannot determine the realized
direction or provide a tight function-specific guarantee without the ordered
coefficient/frequency pairing. A coarse worst-case modulus such as the bound
above remains available. For example,
at `K = 2`, `ω = (1, 2)`: pairings `c = (2, 1)` vs `c = (1, 2)` share all
slot energies and the frequency multiset, yet `s(π)` flips sign (`−1` vs
`+1`); and `c = (1, i)` vs `(1, −i)` share `|c_k|²` while `s₂(Δ) = s₁(−Δ)`.
Compatibility is a property of the *paired* measure `(c_k, ω_k)`, i.e. of the
checkpoint's embedding of frequencies into subspaces.

**T4 — pointwise horizon bound; former exact-conditioning theorem retracted.**
For a table perturbation `ω′ → ω′ + δω′`, the valid first-order statement is

`|δs(Δ)| ≤ Δ Σ_k |c_k||δω′_k|`.

Thus the coefficient-budget Lipschitz upper bound grows at most linearly with
the evaluated horizon. It is invalid to divide this upper bound at `SL` by a
different upper bound on the actual Native-range deviation and conclude that
their ratio is at most or exactly `S`; the Native denominator may vanish to
higher order. For example, with one real cosine score at small `ω`, the actual
sup-deviation ratio between `[0,4L]` and `[0,L]` approaches `16`, not `4`.
No exact condition number in the realized score sup norm is established here.

**T5 — non-identifiability questions; former theorem package retracted.**
At table level, exact score agreement for all contents on an interval can force
the same complex spectral measure under the stated real-analyticity and Fourier
injectivity assumptions. That scoped fact survives.

The checkpoint-level construction `h(Δ)=1-cos(ωΔ)` provides only an
`O(S^-2)` Native-range bound for its selected `ω`; at fixed `S` it does not
prove the former “for every ε>0” claim while retaining `O(1)` long difference.
Likewise, distinct torus one-parameter subgroups need not intersect only at the
origin. A counterexample is `ω=(1,3)`, non-uniform `ρ=(1,1/3)`: at `t=π` both
arcs reach `(π,π)` modulo `2π`. Any off-arc claim therefore needs explicit
arithmetic and finite-range assumptions checked for the concrete spectrum.

The current evidence motivates a working question—how the frozen checkpoint
responds to ordered phase configurations outside its measured exposure—but it
does not prove uniform Native-only non-identifiability or impossibility of a
predictive functional.

**T6 — semigroup vacuity.**
Require `F(S₁S₂) = F(S₂) ∘ F(S₁)` with per-slot multiplicative action
`ω → ω·f_k(S)`. The multiplicative Cauchy equation forces
`f_k(S) = S^(−m_k)` (for continuous/monotone f; Hamel pathologies are
behaviorally irrelevant since any installation is endpoint-observed).
Composition is algebraically automatic for any frozen per-slot power-law
profile (hence for the whole log family) — it is not a distinguishing virtue
of log_s4 (attacker P2-5); the arithmetic law is not closed under rescaling
(D2), but closure is a property of the parametrization, not of behavior.
Moreover: (i) flow structure is unobservable under endpoint-only installation
(off-orbit behavior = gauge); (ii) exact table composition carries zero
behavioral content — log_s4 composes exactly as a table while frozen s4→s8
continuation fails behaviorally; (iii) the repo's own audit already limits the
composition identity to "scale consistency of the formula only".
*Verdict:* semigroup structure is neither an axiom to impose nor a derivable
behavioral property; it is a bookkeeping identity of the log parametrization.
Its only residual design content — forbidding S-dependent refitting of the
mask when a single table must serve a scale interval (requirement 6 of the
problem) — is a *constraint of the problem statement*, not an optimality
principle. The s8 result shows that composition is not sufficient for behavior;
it neither proves a finite universal radius nor falsifies composition as a
possible necessary design constraint. As a bookkeeping identity it survives.

**T7 — fixed-profile phase accounting (descriptive; former divergence theorem
retracted).** For the memo's fixed 4x-installed table definition,

`N_k(S)=ω_kL(S·4^(−m_k)-1)`,

the correct ratio, where the denominator is positive, is

`N_k(2S)/N_k(S)=2+1/(S·4^(−m_k)-1)`.

The excess above two decreases with `S`; the previous S-independent formula
and “excess growing in S” claim are algebraically wrong. If the table is instead
recomputed as `ω′=ωS^(−m)`, it is a different continuation and must be analyzed
separately. Phase range and PI-style blur can still be recorded as descriptive
coordinates, but neither proves a finite behavioural radius or failure of a
method class.

**T8 — gain (exact argmax identity plus unverified finite sweep).**
*What survives:* multiplying one fixed set of attention scores by `g > 0`
preserves that set's argmax and changes concentration. This local identity does
not determine later-layer key mass, generation quality, or a gain method class.

The unrecovered seven-point sweep was reported as inconsistent with one smooth
convex toy balance. Conditional on protocol validity it would reject only that
toy shape; it cannot identify a discrete threshold, portable optimum, or
mechanism. The exploratory headwise report records lower teacher-forced loss
with a lower Hotpot point estimate, an adaptive within-panel association rather
than proof that gain causes a sharpening shortcut.

**T9 — three-layer framing; former per-task-radius inference retracted.**
I. *Representation capacity* — property of Ω alone: aliasing lattice
(A8), separation/wrap structure of the phase map on `[0, SL]`.
II. *Compatibility* — property of (θ, Ω′): the retention modulus of T3/T4,
measured on `[0, L]` (e.g. the 1x PG-19 gate).
III. *Circuit robustness* — property of (θ, Ω′, T):

    R_T(θ, Ω′) := sup{ S ≥ 1 : on every decision node visited by task T at
                       scale S under Ω′, margin − accumulated perturbation
                       budget (T3 chain through softmax/value/residual steps)
                       stays ≥ 0 }.

Fact D cannot define any task radius. In the separate exploratory headwise
protocol, retention and variable-length Hotpot point estimates order two tables
differently, so retention did not rank that QA panel. Without a validated
binary task-success threshold this does not prove formal necessity or
sufficiency. The 2026-08-31 `0.875302` gate belongs to a different PG-19
protocol and must not be compared with headwise gate numbers. The three layers
remain a useful bookkeeping framework, not an identified theory.

## C. Derived surrogate quantities (each with its approximation entry point)

1. **Uniqueness budget `u_k`** (A9). Exact as a number; its role as the
   *correct Native cost currency for movement* is a design assumption:
   linear leave-others-out redundancy may not be the currency the frozen
   network actually uses (approximation enters here).
2. **Normalized index `x_k = ln(c_k/c_orth)`, `c_k = Lω_k/2π`** (cycles of
   Native coverage). Exactly affine in slot index for a geometric lattice;
   `x` is the coordinate in which OLMo→Qwen transport and the two-parameter
   compression live. Approximation: using coverage cycles as the sole coupling
   coordinate ignores amplitude/content structure of the checkpoint.
3. **Two-parameter compression `G_4(x)`** (clipped affine, boundaries
   `x_H ≈ 0.7383`, `x_L ≈ 0.3664`): OLMo movement MAE `0.001223`; zero-refit
   Qwen geometry MAE `0.014303` vs transport `0.002174`. Approximation:
   fitted shape; boundaries are fitted values, not analytic constants.
4. **Saturation onset `Δ_k^sat = 2/|ω′_k − ω_k|`**: first-order heuristic —
   the lag where linearized phase deviation reaches O(1); actual onset is
   content- and margin-dependent.
5. **Worst-case envelope `B̂(Δ) = Σ|c_k| min(2, |Δω_k|Δ)`** using Native slot
   norms: worst case over contents of given moduli; realized contents are
   typically far smaller. Usable as a parametric sufficient condition once
   margins are supplied; unusable alone as a prediction.
6. **Band decomposition** (fast `r_k > 32`, mid `1 ≤ r_k ≤ 32`, slow
   `r_k < 1` cycles): descriptive; the thresholds 1 and 32 are conventions,
   not derived constants.
7. **Margin hierarchy `M_RULER ≫ M_Hotpot`**: latent and currently posited
   from outcomes — no margin in the current evidence is bounded independently
   of who survived (attacker P1-4). With margins free, the three-layer stack
   fits any survival pattern: at present it is a redescription, not an
   explanation. The connection to fact F is direct: the one independent
   margin-like measurement attempted in this project (18 samples → 64-D
   behavioral gradient) failed its unopened holdout — 64 DOF on 18 samples —
   so the measurement that would have supplied margins has already
   empirically collapsed. Until margins are measured independently, the
   three-layer theory is an envelope/ceiling theory, not a quantitative
   account of the RULER/Hotpot split.
8. **Fixed-table phase-range ratio** (T7): for
   `N_k(S)=ω_kL(S·4^(−m_k)−1)`, the exact ratio is
   `2+1/(S·4^(−m_k)−1)` when defined. It is descriptive and has no established
   behavioural weight.

## D. What the current data identify

1. **Ordered pairing is real and checkpoint-level** (fact B): behavior is not
   a function of the unordered spectrum; the slot index is shared across
   content pairs, so the coupling lives in the checkpoint, not per content
   pair.
2. **The log (exponent-space) law beats arithmetic in the tested s4/s8
   protocols.** Arithmetic's
   effective exponent `q_k(S) = −log(ω′_k/ω_k)/log S` drifts below `m_k` with
   RMS error growing in S (`0.01099/0.02195/0.03200` at s=2/4/8, worst pair
   k=21), it is not closed under rescaling, and it distorts geometric ratios.
   The matched-gain operational gate is arithmetic `0.869584` vs log
   `0.875302`; this identifies neither a universal mechanism nor a scientific
   discontinuity at the threshold.
3. **Movement closeness is continuity evidence, not equivalence** (attacker
   P1-8): the C2 two-parameter mask reproduces OLMo movement at MAE
   `0.001223` (max `0.044`) and lands retention `0.870971` vs `0.875302` for
   the original mask — a small behavioral difference straddling a knife-edge
   registered gate, while preserving Qwen long behavior (64K/128K core-4
   `0.6775/0.5725`, gates pass) and missing the strict OLMo operating point.
   Movement metrics certify geometry, not functional equivalence. The only O(1)
   jumps in the record are same-multiset permutation collapses — discrete
   rearrangements of the *pairing* coordinate, compatible with continuity in
   movement distance.
4. **Fact D is invalid for claim use.** It identifies no task radius or
   stopping-vs-hop mechanism.
5. **Fact E is unverified.** Its unrecovered finite sweep cannot establish a
   portable optimum or gain class boundary.
6. **Frozen s4 → s8 continuation ceiling** (NLL/RULER owners): continuing the
   frozen mask to 8x does not preserve the 4x operating point in those
   protocols. T7 supplies no theorem explaining this outcome.
7. **Coordinate relativity**: normalized-index vs physical-x relations vary
   with K and checkpoint (K32/K128 owners); `x` transfers across the two
   K=64 checkpoints, ordinal slot index is not established as equivalent.
8. **z=0 parity** (fact A) validates the intervention surface itself: the
   retrofit path is causally clean.

## E. What remains unidentified

A useful working description of the missing mechanism is the frozen network's
task- and checkpoint-conditioned response to ordered phase changes outside its
measured training exposure. Current owners do not identify a per-task radius,
behaviorally optimal continuation, circuit margins, hop transmission factors,
slot-usage weights, aggregation slack, or EOS margins. T4/T5 do not prove that
these quantities are inaccessible from Native data; they merely leave the
question open. Geometry-only ceilings, envelopes, and gain argmax identities
remain necessary-condition tools, not sufficient predictors of long-task
success.

## F. Superseded candidate global theory

> **Entire section superseded.** It preserves the original fitted-family
> reasoning for audit history, but relies on invalid Fact D and the retracted
> T4/T5/T7 claims. Its behavioral “eliminations,” off-arc total-novelty,
> task-radius, conditioning-metric, and global-theory answers are not current
> conclusions. Use §H and `INDEX.md` instead.

**The object.** An order-preserving, per-slot multiplicative endpoint map

    ω_k(S) = ω_k · S^(−m_k),   m_k monotone decreasing in Native cycle
    coverage c_k = Lω_k/2π (equivalently, monotone increasing in the
    uniqueness deficit 1 − u_k),

with corners anchored by exact arguments: `m = 1` (full PI) where coverage is
below a threshold (`c_L = e^{x_L}·c_orth ≈ 7.78` OLMo cycles, fitted
boundary; `c_orth = 1/(1−b^{−1/K}) ≈ 5.394`) — there the slot's Native phase
curve carries little unique Native information (A9), so PI is nearly free and
buys de-aliasing; `m = 0` above a higher threshold
(`c_H = e^{x_H}·c_orth ≈ 11.29` cycles) — there the slot carries
irreplaceable Native information, so it must not move; a narrow transition
band between. The map is an *endpoint map*, not
an ODE: no flow structure is claimed or needed (T6 shows it would be gauge),
and the semigroup appears only as the bookkeeping identity A7.

**Why this object — and the circularity audit of that "why".**
The exclusion argument must be stated honestly (attacker P1-6): the
waterbed identity A6 is the *definition* of `r_k = S^{1−m_k}`; the novelty
calculus of T7 is calculus on that definition; the corner structure is a
description of the fitted curve; and the exclusion criteria
(composition-closed, order-preserving, coverage-threshold shape) are
satisfied *by construction* by the log family. Choosing criteria your
candidate satisfies structurally, then declaring it the survivor, is
circular. The renaming objection lands: A6, the novelty calculus, and the
corner picture are the log_s4 curve in new notation — bookkeeping, not
identification.

What remains as genuine content is exactly two behavioral eliminations plus
the negative theorems:

(i) **Arithmetic law eliminated behaviorally**: fails the 1x retention gate
(`0.869584 < 0.875`) with quantified exponent drift
(RMS `0.01099/0.02195/0.03200` at s=2/4/8) — not eliminated by elegance.
(ii) **YaRN corner profile eliminated behaviorally on one axis**: fails the
same retention gate (`0.6588`) and RULER-16K (`0.1056`) — while winning
Hotpot-16K (fact D), so this elimination is axis-specific, not global.
(iii) Order preservation is *motivated* (not forced) by fact B + T2: full
same-multiset permutations collapse (Qwen 64K `0.7000 → 0`, OLMo 1x NLL
`3.10423 → 6.86493`); but full permutation is sufficient while monotone
order preservation is not necessary — the Qwen self-construction transfer's
realized table has order crossings at pairs 1 and 18 and still scored
`0.7000` at 64K (`SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md`,
Qwen-construction-transfer section; owner located on the final pass after
the claim was first removed for lack of one). Monotonicity remains a
sufficient structural assumption, no longer a candidate necessary condition.
(iv) The negative theorems (T2 rigidity, T3 weight-blind vacuity, T4/T5
identifiability bounds, T6 semigroup vacuity) constrain every future design
independently of log_s4 — this is the part of the memo that is not renaming.
(v) Alternatives considered — spectral-measure evolution (descriptive,
selects no law), phase-exposure sets (negative theorem: deployed vs Native
joint arcs are distinct one-parameter subgroups, so joint set-novelty is
total for every nontrivial retrofit; only per-slot projected arcs
discriminate), variational continuation (vacuous until the functional is
named; the meaningful functional is exactly the unidentifiable tolerance of
§E), transport maps (content reduces to order preservation) — none supplies
the missing tolerance functional. The endpoint map survives as the fitted
design family with the two behavioral eliminations above, not by derivation.

**Answers to the sub-questions of §1 (as posed).**
A. The constant direction `dx/dτ = m` is locally valid because all Native
diagnostics see only rates (first order in log S, T4), and globally fails
because the tolerance functional is nonlinear in joint-phase novelty — the
fitted m is calibrated at S = 4 and its optimality does not transport (T7,
s8 ceiling).
B. General continuations need not be curved as *curves* — endpoint maps make
the question a gauge artifact; what must be non-constant is the mask's
behavioral validity across S, which is exactly the unidentifiable.
C. Semigroup: neither axiom nor derivable property; a bookkeeping identity
whose only design content (forbid S-dependent refitting of a single table) is
a constraint of requirement 6, falsified as an optimality principle by the
s8 ceiling (T6).
D. The more natural object is the endpoint map with a budget accounting, not
an ODE; the natural metric is the score-norm conditioning of T4, derived from
the exact score equation, not postulated.
E. One S-table for all of `[0, SL]` is feasible as a construction (log_s4 is
one), but its *effective* radius is task-dependent (T9): the same table at
the same physical 16K is good on RULER and bad on Hotpot because the two
tasks read different functions of the joint phase configuration — RULER-style
retrieval reads local content-conditioned matches that survive moderate phase
novelty, multihop QA reads cross-distractor lag-coherence in the mid band and
EOS chains that do not.

## G. Adversarial audit (attacker findings + synthesizer corrections)

> **Superseded audit record:** the later repository audit found fatal defects in
> T4/T5/T7 and invalidated Fact D. Statements below that say all P1 repairs were
> complete or that no P0 remained describe the earlier session only.

Protocol: three isolated builders + one attacker that saw only the reports
and the fact sheet; every load-bearing item was re-derived or owner-checked
by the synthesizer; attacker severities were independently re-verified
against repo owners before acceptance. The attacker's final delivery
(P2-4…P2-7) arrived after first completion and is integrated in a second
pass (containment form of T1, semigroup-as-falsified-norm reframing, the
restored pair-1,18 fact, the terminological flag). **P0 (fatal, retract):
NONE.** The
only candidate was the long-information-leakage attack; it does not rise to
P0 because the object-level construction is genuinely Native-only and has
real blind out-of-sample successes (§0.1). All findings below are P1
(REPAIR, applied) or P2 (cosmetic, applied).

**G.1 (P1-1) Native-only split — object level true, meta level false.**
The per-slot values, G_4 compression, 1x-selected gain, s2 zero-refit and
Qwen transport are Native-only; the design *family* (exponent p=2; log-vs-
arithmetic choice) was selected against long outcomes. Repaired in §0.1 with
owners. This is the most important audit result: the flagship instance does
not violate the problem's constraint at the value level, but no sentence of
the form "identifiable from Native data alone" may omit that asterisk.

**G.2 (P1-2) Mean-power direction.** Merging slots shifts mean power by
`Re(c₁c̄₂) ∈ [−|c₁||c₂|, +|c₁||c₂|]`; sign content-dependent; "can only
drop" retracted. Repaired in A3. (Attacker and synthesizer converged
independently.)

**G.3 (P1-3) "Discontinuity" retracted; conditioning + off-arc replace it.**
Table level: exact Native agreement forces identical tables (analyticity +
Fourier injectivity), so non-identifiability of tables is exactly the
conditioning theorem T4 (amplification S, tight). Checkpoint level: dormancy
construction survives as a model-class statement. Retrofit-specific content
is the off-arc statement, **with the uniform-ρ exclusion**: for
`ρ ≡ 1/S` (pure PI) the deployed arc is contained in the Native arc — no
off-arc configurations are created at all, and PI's cost is entirely Layer-II
blur/resolution loss (consistent with T1). Off-arc novelty and its
unidentifiable tolerance exist only for non-uniform masks. Repaired in
T4/T5.

**G.4 (P1-4) Margin circularity.** No margin is bounded independently of
outcomes; with margins free the three-layer stack fits any survival pattern.
Downgraded to an envelope/ceiling theory; the fact F connection recorded
(the one independent margin measurement attempted collapsed: 64 DOF on 18
samples). Repaired in C7.

**G.5 (P1-5) Novelty calculus.** Retract the constant multiplier
`2(1+2^{m−1})` (three independent errors); keep the super-doubling
inequality and the budget structure; coordinate-count divergence is
descriptive, not behavioral. Repaired in T7. (Attacker and synthesizer
converged independently.)

**G.6 (P1-6) Exclusion argument is circular; renaming objection lands.**
A6 is a definition, the novelty calculus is calculus on it, corner structure
describes the fitted curve, and the exclusion criteria are satisfied by
construction. Genuine content narrows to two behavioral eliminations
(arithmetic law; YaRN on the retention/RULER axis) plus the negative
theorems. Repaired in §F. Direct answer to the renaming question: yes, for
A6/T7/corners; no, for T1–T6 and the two eliminations.

**G.7 (P1-7) Gain toy model falsified by sweep shape.** Observed loss
1−F1 has non-monotone derivative (convexity violated between g=1.0 and
1.05); smooth two-parameter convex family cannot produce it; optimum
location has zero predictive content; the `~16x` collapse below g=1 is a
threshold-like event. Survivors: argmax invariance (exact, single-step
ranking only), sharpening/decisiveness reading with repo corroboration,
non-universality at tested points (trivial but true). Repaired in T8.

**G.8 (P1-8) C2 is continuity evidence.** MAE `0.001223` → retention
`0.870971` vs `0.875302`: continuous, knife-edge at the registered gate.
The only O(1) jumps on record are permutation collapses (pairing coordinate).
B is continuous but ill-conditioned in S, pairing-sensitive. Repaired in D3.

**G.9 (P1-9) Protocol mixing.** 0.875302 (08-31 PG-19 gate protocol) and
0.6588 (09-02 headwise protocol, 20 docs / final-512 scoring) must not be
compared; within the headwise protocol the II-not-necessary/not-sufficient
logic survives (0.7171 vs 0.6588 retention; 0.21169 vs 0.27644 Hotpot F1).
Repaired in T9; registry split in §I.

**G.10 (P2-1) Transplant proof simplified.** Δ=0 gives similarity directly;
spectra equality gives the multiset. Repaired in T2.

**G.11 (P2-2) OLMo parameter slip.** The invariants report wrote K=32 where
head_dim=128 gives K=64; the irrationality conclusion is unaffected
(2⁵·5⁶ exponents divisible by neither 32 nor 64; 10⁶ exponents not by 64).
This memo states K=64.

**G.12 (P2-3) Confusability vs beats — compatible once levels are
separated.** Exact universal confusability ⟺ per-slot phase collision (A8),
no beat frequencies involved. `deg_ℚ(β) > K−1` governs a *different* object:
ℚ-linear relations among frequencies → closure dimension and window-Gram
redundancy; the two number-theoretic audits address different objects and do
not contradict. Beats `(ω_k − ω_l)` describe the Δ-dependence of energies
and window-Gram inner products `|G_kl| = |sinc((ω_k−ω_l)L/2)|`, not scores.
The circuits report's "distractor beats" must be read as content-weighted
inter-slot interference in the score: the discriminative signal between two
candidate keys at lag Δ is the single measure
`Σ_k Re[(c_k(j*) − c_k(j_w)) e^{iω_kΔ}]`, and near-duplicate distractors are
resolved by how the *difference coefficients* interfere across slots — beat
structure is the envelope of that interference, not a separate mechanism.
No contradiction with A8.

**G.13 Pre-registered synthesizer corrections** (independent re-derivation,
applied above): Lindemann–Weierstrass misattribution corrected (T1); the
voided §7.3 margin-gradient route of the basin/barrier note superseded by
fact F (that note's convex-hull algebra remains valid mathematics, its route
unauthorized). The "pair-1,18 crossing survives at 0.7000" claim was first
removed from §F for lack of an owner; the final pass located its owner
(`SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md`, Qwen construction
transfer: realized table has crossings at pairs 1 and 18, 64K core-4
`0.7000`) and restored it in §F(iii) with the owner attached.

**G.14 What the audit leaves standing.** The surviving inventory is exactly
§H's list. The audit changed the memo's center of gravity: from "an endpoint
map defensible by exclusion" to "a family of impossibility/conditioning
theorems plus two behavioral eliminations, around which log_s4 sits as the
audited fitted design family". Nothing in the audit produced a new
constructive law, and none was sought.

**G.15 Falsifiability of the three-layer account (logical conditions, not
plans).** The envelope theory of C7/T9 becomes quantitative only if margins
are supplied from outside its own survival pattern: e.g. discriminative
score margins between correct keys and nearest distractors measured under
Native vs retrofit tables, per-hop transmission factors, and the EOS margin
profile along Ω′-induced paths. Any such measurement that shows Layer-II
retention predicting Layer-III outcomes, or a task radius exceeding its
weakest-link bound, would falsify the current reading. This memo proposes no
measurement; it records what a future theory would have to make identifiable.

## H. Final verdict

> **Corrected verdict:** this memo retains scoped exact identities, T1/T2/T3
> under their assumptions, and the T6 algebraic observation that composition is
> not sufficient for behavior. T4, the checkpoint/off-arc parts of T5, and T7
> are retracted; T8/T9 are bounded interpretations, not class theorems. Fact D
> is invalid, Fact E is unverified, and the headwise panel is exploratory. The
> memo neither proves a disconnected basin or universal non-identifiability nor
> identifies a predictive mature-checkpoint theory.

## I. Number-to-owner registry (cited in this memo)

| Number | Owner |
| --- | --- |
| log_s4 1x PG-19 retention `0.875302`, task retention `0.915103` | `results/CPU_LOW_DIM_COUPLING_LAW_20260901.md` §2 (numerical owner `SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md`) |
| arithmetic 1x retention `0.869584`; q-drift RMS `0.01099/0.02195/0.03200`, worst pair 21 | `SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md` §2 |
| log_s4 RULER-13 `0.71397/0.66705/0.49859`; official YaRN-4 `0.4314/0.2431/0.1056` | `SESSION_BINARY_S4_REAL_CONTEXT_RESULT_20260823.md` line ~293; `SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md` line ~110 (two matrices exist; cite per matrix) |
| YaRN-4 retention gate `0.6588` | `HEADWISE_FACTORIZED_Z_AND_SCALE_FLOW_RESULT_20260902.md` line ~175 |
| G_4 boundaries, movement MAE `0.001223`, Qwen holdout MAEs | `CPU_LOW_DIM_COUPLING_LAW_20260901.md` §5–6 |
| movement-mask construction (pair-count measure, leave-others-out uniqueness, exponent 2) | `scripts/analysis/export_uniqueness_budgeted_tables.py` (hash-pinned default tables) |
| HotpotQA fact D numbers | **INVALID for claim/gate use**: constructed 38-row stress, no recovered raw owner; this memo §0 records but does not own them |
| composition identity `ω(s₁s₂)=ω(s₁)s₂^(−m)` | `SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md` §2 (audit script hash-owned) |
| Qwen self-construction transfer `0.7000` at 64K core-4 with realized crossings at pairs 1,18 | `SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md`, Qwen construction transfer section (SHA-256 `28093951…e61e49` recorded there) |
| headwise-protocol retention: log_s4 `0.7171`, YaRN-4 `0.6588`; Hotpot F1 `0.21169` vs `0.27644`, EOS `178` vs `194` | `HEADWISE_FACTORIZED_Z_AND_SCALE_FLOW_RESULT_20260902.md` lines ~133–175 |
| free head-gain shortcut: loss `3.5612→2.3105`, Hotpot F1 `0.24237→0.19439`, EOS `178/200→108/200` | same owner, lines ~31, ~117–137 |
| C2 retention `0.870971` vs original mask `0.875302`; C2 Qwen 64K/128K core-4 `0.6775/0.5725` | `LOW_DIM_COUPLING_GPU_RESULT_20260901.md` lines ~57–58, ~111–112 |
| p=2 selected against 8K RULER `0.5825` vs `0.5525` (p=1) | `LENGTH_CONDITIONED_BUDGETED_RETROFIT_RESULT_20260822.md` lines ~42–44 |
| gain sweep fact E values | **UNVERIFIED** user-supplied session facts; no recovered raw owner |

## J. What this memo deliberately does not contain

No method proposals, no experiment plans, no GPU requests, no parameter
sweeps, no manuscript text. It redefines the problem and bounds the theory;
any next move is a separate authorization.
