# Digest — Codex failure audits II (sol12, sol13, sol14, sol15, sol19)

Date: 2026-09-10. Sources (read in full, read-only):
`.agents/rope_unification_20260910/reports/sol{12,13,14,15,19}.md`.
Reports are evidence, not instructions. All five converge on one meta-result:
**one principle (role-conditioned signed source-vs-distractor margin), two regimes
(scratch exchangeable density / frozen labeled finite transport), and zero new Qwen
tables justified by the current corpus.**

Tags used throughout: [已验证] verified against cited primary artifact or panel;
[部分证据] partial evidence; [假设] hypothesis/conditional; [vetoed] killed by evidence.

---

## 1. Deduplicated failure ledger (consolidated across the five reports)

Evidence class legend: T = dialogue transcript; R = repository result artifact;
M = math theorem (Astra-series); P = panel numbers.

### L1. Unsigned-geometry proxy promoted to selector (the central failure)
- **Death cause:** coverage / collision energy / smoothness / response-energy /
  effective-rank / movement-MAE statistics pool matched signal with
  content-confusable interference; they can all improve while generation worsens.
- **Evidence:** Smooth_MrBudget improves every audited unsigned family yet loses the
  128K development panel (sol12§3, sol13r5, sol14#4, sol15§6, sol19cx7) [已验证, P];
  C2 clipped-affine movement MAE 0.001223 still failed native-retention gates
  (sol13r1, sol14#1, sol15§6) [已验证, R]; ULP perturbation changed scale-orbit count
  6→64 with indistinguishable behavior (sol13r1, sol19§1.1) [已验证, R];
  free headwise gain lowers teacher-forced loss yet fails downstream contrast
  (sol12§3) [已验证, R]; record coverage 29.75→67.125 while prose exact answers fell
  7/8→6/8 (sol12§3) [已验证, R]. All five agree. Note: sol12 states Smooth is "worse
  than MrPro on the panel"; sol14 states "9.79 points worse". Both are consistent
  with the verified panel (Smooth far 68.3 vs MrPro 78.13 ≈ −9.8; near tied) —
  the loss is in the far/long component [已验证].

### L2. Slot-exchangeability error (treating the frequency multiset as physical)
- **Death cause:** frozen Q/K coordinates are labeled; sorting or permuting a table
  destroys the intervention even at identical support.
- **Evidence:** same-multiset permutation OLMo NLL 3.10423→6.86493, Qwen 64K core-4
  0.70→0 (sol13r2, sol14#2, sol19§1.5) [已验证, R].
- **Scope agreement:** sol12, sol19, sol15 all restrict the verdict to *frozen*
  deployment; a joint permutation of frequencies **and** learned coefficient slots is
  equivalent, and scratch channels may co-adapt [已验证]. Disagreement nuance:
  the *archived* universal non-identifiability theorem over-generalized this fact
  (see L11).

### L3. Local derivative promoted to finite intervention
- **Death cause:** Taylor/Jacobian scores on full S=4 tables accumulate many phase
  turns; recorded linear predictions had 71–468% relative error; phase shifts of
  22.74 and 90.97 rad voided local gating.
- **Evidence:** sol13r3 [已验证, R]; sol19§1.2 (remediation retired local Taylor
  gating) [已验证]; sol12§4 (scale-transport log-mean-exp "neither long-text error
  probability nor certified bound") [已验证, R]. Consistent across sol12/13/19.

### L4. Independent-channel additivity (cross terms discarded)
- **Death cause:** the shared-table derivative contains cross-key signed coherence and
  shared-head/W_O cancellation; identical per-key nonnegative energies give total
  derivatives 0 vs 4.
- **Evidence:** sol13r4 [已验证, R]; carried by sol12 (full Σ in SRI), sol19 cx4
  (diagonal score can reverse ordering) [部分证据 for the general repair].

### L5. Smoothness-as-mechanism and its mirror error ("BM failed")
- **Death cause:** smoothness is a numerical/solver constraint, not a performance
  principle; but BM's OLMo strength (41.67% vs 7.09% on 16K RULER) and Qwen7B failure
  (0 wins/3 losses/15 ties; QA 50%→0) are both real — checkpoint-conditioned.
- **Evidence:** sol12§7+addendum [已验证, R]; sol13r5 [已验证]; sol15§6 [已验证].
  All five refuse both "smoothing is universally beneficial" and "the family is dead".

### L6. Exact-EVQ / Cosh / Z conflation and frozen transplantation
- **Death cause:** for the exact finite-window collision kernel the unique measure
  optimum is **finite atomic**, not a positive Cosh density (Astra02); Cosh follows
  only from the delta-plus-min (αI+βG) approximation; support retargeting reversed a
  Cosh scratch gain; Z was repeatedly mislabeled as Cosh.
- **Evidence:** sol12§8 [已验证, M]; sol13r6 [已验证, M]; sol14 [已验证];
  sol15§3 four-object separation [已验证, M]; sol19§2.1 ("Cosh = conditional
  relaxation") [已验证, M]. This is the single most-agreed mathematical fact.
- **Sub-failure:** iid channel noise is rotation-invariant → produces *no*
  frequency-dependent collision term; claiming collision benefit needs coherent,
  lag-conditioned nuisance (sol12cx2, sol13cx3, sol15§3) [已验证 math].

### L7. MrRoPE first-zero story over-extended into a derivation
- **Death cause:** MrRoPE **assumes** arithmetic radix increments and selects Qwen
  boundaries empirically; the progressive interior profile is not derived from
  coherent-cosine positivity.
- **Evidence:** sol13r7 [部分证据→已验证 of the assumption, via MrRoPE paper lines];
  sol15§5 [已验证]; sol19§1.5 ("satisficing", not optimum) [已验证]; sol12§2
  ("high-freq unchanged / low-freq /s / monotonicity / 44.8% limit" all demoted from
  laws to design choices; an earlier useful table was nonmonotone) [已验证, T].

### L8. P2/E1 residual shape mistaken for portable cause
- **Death cause:** FullLagP2−MrPro is 84.26% symmetric energy, only 13.18% cubic —
  shape energy says what differs, not what caused gains; P2 reverses across
  checkpoints/tasks (+3.54 long / −14.31 short on panel).
- **Evidence:** sol13r8 [已验证, R]; sol12§7 (conditional observations, not to be
  averaged) [已验证]; sol19§1.5 ("premature phase saturation" and "architecture
  scrambling" causal stories demoted to [假设]).

### L9. Attention/selection ≠ generation (prefix, readout, decoder collapse)
- **Death cause:** cross-cache diagnostic — BM reading retrieves the correct answer
  from a MrPro-formed prefix, BM-formed prefix stays wrong under MrPro reading;
  oracle/late-KV pruning only partially restores; 128K BM diagnosis — compact content
  solvable at original positions but not from dense-prefill KV state.
- **Evidence:** sol12§3 [已验证, R]; sol13r10 [部分证据]; sol14 [已验证, R];
  all five carry the rule: generated correctness remains the decision endpoint;
  attention source-selection is mechanism evidence only [已验证].

### L10. Regime conflation (one law for scratch / adapter / frozen)
- **Death cause:** weight×table crossings, Geo↔Cosh runtime swaps (7.14↔76.20 /
  7.16↔23.05 PPL), and permutation collapse show co-adaptation changes the statistical
  object; the objective semantics must differ per regime.
- **Evidence:** sol15§1 [已验证, manuscript]; sol12§1 [已验证, T]; sol13r11,
  sol14, sol19§2 [已验证]. All five agree; this is the accepted replacement frame.

### L11. Over-claimed impossibility/metatheory (retirements of archived verdicts)
- **Death cause (of the claims, not the direction):**
  (a) *universal non-identifiability* over "any structural metric" — the permutation
  construction only bounds permutation-invariant maps; ULP only refutes the
  discontinuous orbit count; one C2 observation does not construct the ε→0 family
  (sol19§1.1, sol13 scope note, sol12 "tests certify invalid premises") [部分证据 for
  class-specific bounds; the universal form is [vetoed]];
  (b) *slot-19 Fisher/Hessian* — "81.2% of residual in slot 19" ≠ curvature; no
  measured Fisher/Hessian artifact exists; empirical-Fisher=Hessian and MAE=L1 steps
  are math errors (sol19§1.2) [vetoed];
  (c) *"six necessary-and-sufficient observables"* — no sufficiency proof; the
  remediation itself demoted T to coordinates (sol19§1.3) [vetoed];
  (d) *support-retargeting mechanism* — common-scalar multiplication preserves
  adjacent log-interval ratios for both uniform and nonuniform z; the reversal is
  observation, not mechanism (sol19§1.4) [reversal 已验证; explanation vetoed];
  (e) *"VICTORY CONFIRMED" audit* and the teamwork-preview review as closure
  certificates — internally inconsistent with the unsupported claims they coexisted
  with (sol19§1.5) [vetoed];
  (f) *universal 1x–2x exchange-rate law* over seven heterogeneous protocols —
  rows mix architectures/metrics/gains/regimes; Video-DiT exception; support
  retargeting reverses scratch ordering (sol12 addendum) [vetoed];
  (g) *τ≈d_head/√L* — `verify_tau_unified.py` "PASS" is a script convention
  (15 hand-entered anchors, mean rel. err 9.6%, max 33.3%) (sol15§3) [假设 at best].
- **Note:** sol13 and sol19 *agree and explicitly state* the archived universal ban
  is too strong; sol12 concurs via the Type-III continuity objection. This narrows —
  but does not overturn — the panel's root non-ranking veto.

### L12. Scale-transport confounding then over-generalization
- **Death cause:** beta statistic mixed window/truncation with positional effects;
  the resulting table overwrote the MrPro middle (exponent ≈.0065→≈.912) and harmed
  short behavior; equal phase under rescaled distance ≠ replay of the successful
  computation (it generated the withdrawn dynamic-bifocal proposal).
- **Evidence:** sol12§4 [已验证, R]. Fix agreed by sol12/13/19: transport a declared
  *labeled operation distribution* (source, hard distractor, prefix state, readout
  assumptions), never distance alone.

### L13. Candidate-generation churn substituting for resolving the unknown
- **Death cause:** new-curve/arms/gate loops without a discriminating measurement;
  16K assay at floor (0/32–1/32); 8x8 panel wording-sensitive; MrPro+CoPE hybrid 8/8
  simple retrieval / 0/8 UUID / skipped tracking — neither superiority nor
  attribution.
- **Evidence:** sol12§5 [已验证, T]; sol13r12; sol19§4. Consistent.

### L14. Failed direct-optimization routes (pre-condition for any new optimizer)
- **Death cause:** 64-dim behavior-gradient route failed its unopened holdout;
  direct-z fixed-support pilot failed its declared gate; renaming to signed
  gradient/MGDA/QP does not repair the missing population definition.
- **Evidence:** sol12§6 [已验证, R]. Constraint: optimizer must be *downstream* of a
  defensible statistical object.

### L15. Process/workflow failures (sol14's ledger, corroborated by sol12§5, sol13r12)
- F1 scope expansion (oversized appendix rolled back) [已验证, T];
  F2 proxy-as-outcome (exact length×checkpoint×overlay×metric matrix only specified
  after correction) [已验证, T]; F3 OOM "diagnosis" by intervention not evidence
  [已验证, T]; F4 ETA before protocol check (continuation broke seed-42 batch
  protocol) [已验证, T]; F5 file-gate while GPU idle ("still training" while dead;
  CUDAGraph overwrite crash) [已验证, T]; F6 harness complexity → import/tokenizer/
  inv_freq/path failures [已验证, T]; F7 454M checkpoint loaded under generic 500M
  config (4 layers random) — run correctly invalidated [已验证, T]; F8 ended with the
  requested evaluation never validly completed [已验证, T].
- Status ladder carried into all rules: proposed ≠ implemented ≠ launched ≠
  checkpointed ≠ evaluated ≠ accepted.

### L16. Estimand narrower than narrative
- **Death cause:** passkey evaluators separate teacher-forced NLL-gap from
  autoregressive exact match; the zero-training tournament optimizes NLL, not source
  binding; sparse-memory pilot had a query-blind shortcut, then no baseline query
  learnability.
- **Evidence:** sol12 addendum [已验证, R]. Also: L=256 scratch report's
  "monotone advantage / tau=4 verifies optimum" is stronger than its own table
  (32x gain slightly < 16x; only tau 2 vs 4 swept) [部分证据].

---

## 2. sol14's role-conditioned allocation rule — exact recipe

Headline rule: *allocate labeled positional channels to maximize useful-source margin
relative to content-conditioned interference, subject to explicit native-operation
constraints* — one principle, two regimes.

**Scratch branch:** channels exchangeable before learning; prescribe signal field h and
nuisance covariance C; conditional matched-filter density
ρ*(x) = [C⁻¹h(x)]₊ / ∫[C⁻¹h(y)]₊dy, with active-set KKT when positivity binds;
EVQ-Cosh is recovered only under constant signal + nested slow-tail covariance.

**Frozen branch (exact recipe):** for operation role r (useful source vs one competing
key), content instance u, pre-RoPE-fixed margin

m_r(ν;u) = Σ_j a_rj(u)·cos(d_r·ν_j) + b_rj(u)·sin(d_r·ν_j) + c_r(u),
μ_r(ν)=E[m_r], v_r(ν)=Var(m_r).

Boxed optimization:

max over ν∈F  min over r∈R_far   μ_r(ν)/√(v_r(ν)+ε_r²)
subject to    μ_q(ν) − γ_q·√(v_q(ν)+ε_q²) ≥ 0   for q ∈ R_native.

F fixes endpoint/support policy, positivity, ordering, declared movement limits; ε
covers estimation + upstream-transport uncertainty. Probability content: Cantelli
P[m_r≤0] ≤ v_r/(v_r+μ_r²); union bound over distractors gives a source-selection
probability bound. Explicitly a *conditional attention* result, not a
generated-answer theorem.

**Acceptance protocol:** local convex approximation only proposes a direction; a table
is accepted only after exact finite trigonometric replay of every native and far
constraint, then the requested full-model behavior test. If signed moments fail to
rank the observed MrPro/Smooth/P2 outcomes correctly, the rule is missing the
operative role/value-readout/upstream effect — do **not** add another unsigned penalty.

**6-step decision contract (Qwen2.5-3B 32K→128K):** (1) freeze checkpoint, tokenizer,
MrPro/P2/Smooth tables, prompts, labels, decoding, provenance; (2) on independent
calibration rows record pre-RoPE Q/K, compute exact signed source-vs-every-material-
competitor margins split native/far, include cross-slot covariance (Frobenius
projection norms are not a substitute); (3) gate: statistic must first reject Smooth
as universal improvement and expose the P2 short/long trade — failure means stop,
the measurement contract is wrong; (4) from the MrPro/evidence-supported feasible face
solve ONE maximin with native probability constraints — no curve sweeps, no slot
chosen from held-out answers; (5) certify exact finite table (ordering/endpoints,
native constraints, far margins, full trig replay), compare one intervention vs an
equal-size relation-preserving control + reused baselines; (6) test generated
long-context tasks + native retention; attention selection is mechanism, generated
correctness is the endpoint. Verdict: no supplied artifact contains the role-conditioned
moments needed to emit a defensible new table [已验证 as a coverage statement].

Sol13's isomorphic-but-more-robust variant: z_r = (μ_rᵀf_r − ε_r‖f_r‖ − η_r)/‖L_rᵀf_r‖
with η_r=log((M_r−1)(1−a_r)/a_r), max-min over far roles, hard z_r≥z̄_r native
constraints, conic inner step lower-bounding the mean / upper-bounding the sd, second-
derivative box bound √(u²+u⁴), exact reevaluation after. Sol12's SRI and sol19's
J_r = μ²/(2v) − log N are the same family.

---

## 3. sol15's claim audit — the defensible unification statement

### The precise wording defended (sol15§7, verbatim):

> "A RoPE design allocates a finite set of equal-amplitude rotary channels over
> frequency while choosing which positional score relations to preserve or retime.
> For a declared source/competitor population, the operative quantity is the signed
> log-partition margin generated by the table and its learned content coefficients.
> In an exchangeable scratch-design model with allocation-independent useful signal
> and shared coherent local-plus-nested nuisance, minimizing a softmax failure bound
> reduces to the EVQ surrogate and yields Cosh in the continuum; its finite resolved
> equal-count version is an integer allocation problem. In a frozen checkpoint,
> channel labels and learned coefficients must be retained, so allocation is a
> constrained finite transport problem. MrRoPE supplies a structured native-relative
> dilation family, while role-qualified harmonic constraints can justify exact
> mixed-mode retiming. None of these restricted reductions establishes a
> checkpoint-independent LM-optimal curve."

Common exact object (eq.1): per-row z_t(ν)=c_t+Σ_j{A_tj cos(ν_j d_t)+B_tj sin(ν_j d_t)};
role margin M_r(ν)=log Σ_{S_r} e^z − log Σ_{D_r} e^z; single-source
p(S_r|S_r∪D_r)=σ(M_r). Probability certificate (eq.2, Markov on joint MGF, no
independence needed): Pr[p_*<q] ≤ min{1, q/(1−q)·Σ_t e^{b_t+v_t/2−μ}}. Equal
zero-mean/variance case → correct scalar objective is v/2−μ with unavoidable log|D_r|
threshold; pairwise SNR alone can improve while source mass worsens.
Finite-count solver (Astra06, eqs.4–5): integer DP over bins, global in O(BK²);
for constant h the real relaxation obeys a discrete Cosh recurrence converging to the
continuum law; assumes shared-within-bin noise + nested Brownian covariance, B =
physical correlation resolution; frozen ordered slots keep a DP only in the additive
special case — heterogeneous cross-slot covariance destroys the recurrence.

### Every claim graded:

**SUPPORTED [已验证]:**
1. Exponent allocation is a real design variable (fixed-range changes learned behavior).
2. Causal fixed-range effect + range×weight co-adaptation (02_identification; three paired seeds; tail NLL 512/1K/2K; retargeting reverses ordering).
3. Cosh = unique optimizer **of the declared surrogate** C_app = (α/2)∫ρ²+(β/2)∫S_ρ² (03_theory exactly scoped).
4. Regime split is real and decisive (Geo↔Cosh runtime swaps 7.14↔76.20 / 7.16↔23.05 PPL).
5. MrRoPE as native-relative dilation coordinate (ν_j=ω_j S^{−m_j}, nonnegative edge increments summing to 1) and empirical baseline.
6. Conditional constructions: role-margin object (1)/(2), finite-count DP (4)/(5), exact projected mixed-mode retiming (6) — each under explicit assumptions [部分证据, conditional].
7. Manuscript as written survives the audit (bounded claims; Round-2 review read the contribution the same way).

**UNSUPPORTED — refuses to claim [vetoed]:**
1. Universal Cosh or MrPro optimality for LM quality.
2. Deriving τ ≈ d_head/√L from current theory ("PASS" = script convention; 9.6% mean / 33.3% max anchor error) [假设].
3. Treating iid channel noise as the EVQ α∫ρ² term (rotation-invariant → no such term).
4. Treating the exact finite-window kernel optimum as a smooth density (it is finite atomic; Astra02) — plus: the atomic theorem does NOT transfer to finite integer-lag Grams where uniqueness can fail, and does not say an LM should repeat frequencies.
5. Transplanting an unlabeled density to a frozen checkpoint.
6. Selecting tables by smoothness / effective rank / residual energy / SNR alone / unqualified harmonic magnitude.
7. Claiming task success from fixed-state attention or geometry.
8. Historical-theory-note extras: literal continuous Hilbert–Schmidt projection onto αI+βG is undefined (αI not HS in infinite L²; a finite Galerkin resolution must be declared); the "dominant Cosh component" is a heuristic scale argument.
9. Astra05 (6): exact *conditional on a role-qualified relation set R* — not a way to discover R; order can fail; some exact beat-preserving moves accelerate individual frequencies.
10. Astra04 stitched-block KL: can rank a fixed-state intervention; cannot establish whole-model transfer (cached native states ≠ 128K forward; zero-padding asserts distractor irrelevance; equal block weighting is an evaluation prior).
11. Four-object separation enforced (exact-atomic optimum / smooth Cosh surrogate / finite-K integer counts / frozen labeled table) — no two are interchangeable.

Evidence-check ceiling [已验证 each]: sparse-memory 6144 rows = role heterogeneity demo
(relations 4096/4096 margin 9.1994 vs content 399/2048 margin −0.87595), synthetic
baseline only; FRESH_FINEWEB = one-checkpoint frozen tail-NLL; dose-response = failed
joint gate, self-declared non-selector; OLMo movement fit = reconstruction only;
EVQ-LoRA probe collapse = real but scoped negative; operator-cache partition-invariance
≠ sparse-token-selection result.

---

## 4. sol19's historical closure audit — what is CLOSED, at what confidence

Explicit non-closure: "the historical record does **not** prove a universal
impossibility theorem for frequency allocation" and does not close the causal loop
with any finite observable set. Constructive frequency allocation remains OPEN.

CLOSED / retracted:
| # | Direction | Status | Confidence (sol19's words) |
|---|---|---|---|
| C1 | Unordered/permutation-invariant spectral summary as frozen certificate | CLOSED [vetoed] | "decisive"/"strong proof" — permutation collapse + labeled-slot logic |
| C2 | Discrete scale-orbit count as physical selector | CLOSED [vetoed] | "not stable" — ULP discontinuity, direct |
| C3 | Small unweighted table error (MAE) as behavior certificate | CLOSED [vetoed] | strong — C2 0.001223 gate failure |
| C4 | Independent treatment of fixed-support vs support-retargeting | CLOSED [已验证 coupling] | strong experimentally; **mechanism remains OPEN [假设]** |
| C5 | Static geometry as task-success objective | CLOSED as sufficient [vetoed] | strong — "diagnostic, not objective" |
| C6 | Universal non-identifiability theorem (worker_falsification_1 quantifier) | RETRACTED to class boundary [vetoed as stated] | proof gaps enumerated (permutation/ULP/C2 scope) |
| C7 | Slot-19 Hessian/Fisher curvature claims | RETRACTED [vetoed] | no artifact; two math errors (Fisher≠Hessian; MAE≠L1) |
| C8 | Six "necessary-and-sufficient" observables | RETRACTED [vetoed] | no sufficiency proof; category mixing |
| C9 | Support-retargeting *explanation* (uniform-vs-nonlinear ratio story) | RETRACTED; observation survives [vetoed] | algebra refutes the claimed asymmetry |
| C10 | "VICTORY CONFIRMED" audit + teamwork_preview review as closure certificates | UNRELIABLE [vetoed] | coexisted with unsupported claims; internal inconsistency |
| C11 | Taylor/local gating for finite table moves | CLOSED, replaced by remediation path integral | remediation record [已验证]; but path integral is an identity "not a cheap pre-hoc predictor"; finite rotation bound "safe but too loose to rank" |
| — | Same-multiset collapse applied to *scratch* | NOT closed that way [已验证 frozen-only] | weights may co-adapt during training |
| — | FullLagP2 "premature saturation/scrambling" causal stories | hypotheses [假设] | not identified mechanisms |
| — | MrRoPE-Pro | satisficing heuristic [部分证据] | cutoffs/transition not derived optima |

Missing evidence named: the calibrated, source-labeled moment object at
layer/operation level and its transport stability — "not another hand-designed
frequency curve."

---

## 5. Carried-claim register with tags

- [已验证] Smooth geometry-win/task-loss = decisive anti-proxy evidence (panel far 68.3 vs 78.13; sol14 "9.79 pts" = far component).
- [已验证] Labeled-slot dependence of frozen checkpoints (permutation collapse 3.10→6.86 / 0.70→0).
- [已验证] Astra02: exact finite-window optimum finite atomic ≠ Cosh density.
- [已验证] Regime split: Geo↔Cosh runtime swap PPL 7.14↔76.20 / 7.16↔23.05; one table ≠ one object across regimes.
- [已验证] Distractor-count penalty required: η_r = log(H(1−a_r)/a_r) (sol12), log((M_r−1)(1−a_r)/a_r) (sol13), −log N_r term (sol19), log|D_r| threshold (sol15).
- [已验证] Prefix-formation vs reading vs generation separable (cross-cache; BM 128K diagnosis; record-coverage vs prose-answer dissociation).
- [已验证] BM conditional reality: OLMo strong (41.67% vs 7.09%), Qwen7B failed (QA 50→0), Qwen3B NLL +.00425@16K / unresolved 8K,32K.
- [已验证] Scale-transport beta confounding (.0065→.912 middle overwrite); transport must carry labeled operation distribution.
- [已验证] Direct-gradient (64-dim) and direct-z fixed-support routes failed their gates.
- [已验证] Universal 1x–2x exchange-rate law inadmissible over heterogeneous protocols.
- [部分证据] Full signed-margin max-min recipe itself — never run end-to-end; component counterexample checks pass on paper (sol13 cx2 reproduces Smooth slot-28 reversal +0.99946→−0.99954 analytically [已验证 math]).
- [部分证据] Fixed-support allocation activity in scratch (L=256 tau=4 three-seed raw-PPL; 454M EVQ×YaRN passkey table) — supports "allocation active", not the universal story.
- [假设] EVQ-Cosh recovery assumptions (exchangeable channels, allocation-constant signal, αI+βG covariance form) — declared, untested on Qwen.
- [假设] Transport envelope (‖E c_r−μ_r‖≤ε_r, Cov⪯L_rL_rᵀ) stability under table change; upstream drift voids the bound.
- [假设] τ≈d_head/√L law; "dominant Cosh component" scale argument.
- [vetoed] Universal non-identifiability as stated; slot-19 Fisher claims; six-observable sufficiency; support-retargeting mechanism story; VICTORY-CONFIRMED closure; iid-noise→α∫ρ²; unlabeled density→frozen; geometry-only selection; task success from fixed-state attention; universal Cosh/MrPro optimality; MrRoPE-first-zero derivation of interior profile; strict monotonicity as law (one earlier useful table was nonmonotone — but dropping all structure also unsupported, permuted deployments fail [已验证]).
- [vetoed→rescoped] Root non-ranking veto: the panel decision stands, but sol19/sol13/sol12 agree its *theorem-level* justification must be narrowed to the tested model-blind class.
- [consensus, 已验证] "No new Qwen table is justified by the current corpus" — all five reports state this independently.

---

## 6. Constructive next rules (recipe form, consolidated)

All five give the same gate-first falsification recipe; the union:

1. **Freeze provenance.** Checkpoint, tokenizer, tables {MrPro, Smooth, E1 slot28, P2}, prompts, source/distractor labels, prefixes, decoding, scorers, table hashes. (sol14 step1, sol19§4, sol12§7)
2. **Measure the signed object, not another curve.** On benchmark-isolated, held-out calibration rows: pre-RoPE Q/K → exact signed source-vs-each-hard-distractor margins per role (native and far), with cross-slot covariance / finite-MGF, retaining layer/head/relation/lag labels. Qualification = compact correct answer + source-deletion/content fork; distractors content-confusable. Do NOT fit on generation labels. (sol12 decision, sol13 next-calc, sol14 steps2, sol15§8, sol19§4)
3. **Pre-registration gate.** The statistic must reproduce the known contrasts *before* any optimization: reject Smooth on its failing rows (slot-28 sign reversal must come out right), expose P2's +long/−short trade, keep E1's small positive direction without over-claiming. Fail → stop; report the missing causal layer (source labeling / prefix formation / value transport / decoder trajectory). Feasibility fail on native∧far → report incompatible roles (a legitimate obstruction, evidence for why a single cold table may need routing/adaptation). (all five)
4. **Solve exactly one allocation on the evidenced face.** MrPro face (j≤23 native, j≥40 /4, 16 interior labeled displacements free — the smallest complete departure, not a law). Route A: if exchangeable within resolved bins + shared local-plus-nested covariance adequate → finite-count integer DP (Astra06 recurrence), ordered labeled assignment in its additive special case. Route B: if a few signed joint relations dominate → exact projected retiming (I−P_R)ω+S^{−1}P_Rω on the qualified row span, then exact finite log-partition evaluation. Elsewhere: conic local max-min step (sol13). No curve sweeps; optimizer returns "not identified" rather than manufacturing a table. (sol12/13/14/15/19)
5. **Certify finite, then one intervention.** Exact trigonometric replay of every native + far constraint; second-order/remainder box bound; compare ONE frozen candidate vs reused baselines + equal-size relation-preserving (or matched-mirror) control on independent long rows + native retention. Adapter regime: alternate (x, adapter) against the same SRI target, re-measure μ,Σ after adaptation, keep native protected set; claim co-adapted allocation only with matched adaptation controls. (sol12/13/14/15)
6. **Status discipline.** proposed/implemented/launched/checked/completed/qualified/supported are distinct; least-cost discriminating comparison; results must change the next decision.

---

## 7. KKT relevance map (which failure evidences which term/constraint exclusion)

Objective terms:
- **Signed mean μ_r present, not |response|²:** L1 (Smooth energy-win/behavior-loss) vetoes any unsigned quadratic term (collision energy ∫cos², coverage, response energy) as the objective — the mean must be source-signed per label. [vetoed → term excluded]
- **Variance/covariance denominator with FULL joint Σ:** L4 (cross-key coherence, per-key 0-vs-4 derivatives) and sol19 cx4 (negative-covariance cancellation) exclude diagonal/per-slot-additive Σ in the deployed shared-table KKT; the Gram/cross terms ARE the decision. L6-sub (iid isotropic noise is rotation-invariant) excludes the α∫ρ² ridge unless a coherent shared field is declared — the αI term survives only as a *declared finite-bin / resolved-noise* modeling choice (sol15). [已验证 exclusions]
- **−log N_r / η_r crowd term:** L1 + panel far-band dilution + counterexample 4 ("positive pairwise margin insufficient at 128K") exclude any pure pairwise-SNR objective; the KKT multiplier structure must carry the distractor-multiplicity threshold term. [已验证 necessity, at bound level]

Variable/parameterization:
- **ν_j as ordered labeled vector:** L2 permutation collapse excludes multiset/density/sorted parameterizations of the frozen problem — no sorting-post-projection step; KKT gradients are per slot-label, and the stationarity conditions only make sense in labeled coordinates. [已验证]
- **Exchangeability only in scratch:** L10 (Geo↔Cosh runtime swap) excludes shared-variable pooling across regimes; scratch density KKT (active-set [C⁻¹h]₊, sol14) is valid there, frozen needs the labeled transport program. [已验证]

Constraints / feasible set F:
- **Native hard constraints (z_q ≥ z̄ / chance-constraint), not penalties:** L9-native rows and task-floor observations (sol13r9) exclude folding native retention into the objective as a soft term. [部分证据]
- **Trust box + exact-replay certification on any local step:** L3 (71–468% linear error) excludes unbounded local KKT/Taylor steps; stationarity conditions hold only inside a certified remainder box. [已验证]
- **Monotonicity/endpoints/`/s` tails in F only when the chosen face demands them:** sol12§2 (nonmonotone useful table; 44.8% "limit" was a conditional artifact) demotes these from laws to face constraints; sol19's (5) keeps ordering as a declared frozen-DP constraint — see conflict note §8. [部分证据]
- **Zero-mean degeneracy:** sol12cx2/sol13cx3 — if μ_r≡0 the KKT system has no positive-margin solution reachable by covariance reduction; the correct optimizer output is "not identified". [已验证 math]
- **Atomic vs density optimizer class:** L6 — the exact finite-window kernel's KKT lives over measures with finite-atomic optima (Astra02); the smooth Cosh Euler–Lagrange solution belongs to the surrogate. Solving one and installing the other is a class error. [已验证]

---

## 8. Conflicts with established facts — flags

> **FLAG 1 (loud): scope of the non-ranking veto.** Project fact: root non-ranking —
> veto. sol19 (with sol13's scope note and sol12's Type-III objection) RETRACTS the
> archived *universal* non-identifiability theorem that partially dressed that veto.
> The panel decision itself is untouched (it is an empirical ranking outcome), but any
> downstream text that cites "no low-dimensional law is identifiable — theorem" is now
> contradicted by two audited reports; the surviving claim is class-bounded
> ("no *tested model-blind unordered* statistic certifies frozen deployment").

> **FLAG 2: monotonicity constraint status.** sol19's frozen program (5) hard-codes
> ν_0≥…≥ν_{K−1}; sol12 records a useful nonmonotone table and demotes strict order to
> design choice; sol13 places ordering in F "only if part of the chosen deployment
> face". Not fatal (MrPro face is monotone, so (5) is a facet, not a law) but the
> reports do not state this reconciliation — do not inherit monotonicity as universal.

> **FLAG 3: Smooth panel phrasing.** sol14's "9.79 points worse" is a panel aggregate
> framing; the verified near/far rows show Smooth near-TIES MrPro (87.2 vs 87.22) and
> the entire loss is far (68.3 vs 78.13). Use the near/far decomposition, not a single
> delta, in any claim about Smooth.

> **FLAG 4 (minor): model-record naming.** sol14/sol12 say "Qwen-1.5B@64K" for
> FullLagP2; canonical project record is 1.485B. Same experiment, rounding — cite
> 1.485B in carried claims.

> **FLAG 5: E3_BM naming (known Q7).** None of these five reports resolves it; sol12's
> Qwen7B "BM" is the boundary-matched family, not identified as E3_BM. Treat as
> untouched open question; no conflict, but no help either.

> **FLAG 6: no report contradicts the 6Pro endpoints-as-design-constraints fact** —
> sol13/sol14's MrPro-face recipe is fully consistent with it (endpoints fixed, 16
> interior slots free).

No report disputes: ν_j=ω_j S^{−m_j} coordinate, Qwen2.5-3B W=32768 S=4→131072 setup,
or the Smooth-decisive-anti-proxy status. All five *add* the same missing-evidence
verdict: the role-conditioned signed moment object has never been measured.

---

## Lineage verdict (5 lines)

1. DEAD mechanisms, consolidated: unsigned-geometry selectors (collision/coverage/
   smoothness/rank/energy/MAE/orbit-count); permutation-invariant or unlabeled-density
   certificates for frozen checkpoints; local/Taylor-only and Fisher-curvature
   shortcuts; per-slot additive energies; Cosh-or-atomic-EVQ transplantation to frozen;
   first-zero derivation of MrRoPE's interior; universal exchange-rate and
   six-observable closure claims.
2. RESCAPED (not dead): labeled signed-margin allocation itself — the veto is
   class-bounded, constructive frequency allocation remains open per sol19+sol13+sol12.
3. The one unification statement all five converge on and sol15 grades defensible
   (verbatim §7): allocation = finite equal-amplitude rotary channels + a declared
   source/competitor population whose operative quantity is the signed log-partition
   margin; EVQ-Cosh is its exchangeable-scratch covariance reduction (finite form =
   integer DP); frozen = constrained finite transport on labeled slots with MrRoPE's
   dilation family as feasible scaffold; "none of these restricted reductions
   establishes a checkpoint-independent LM-optimal curve."
4. Blocking evidence gap (unanimous): content-conditioned, source-labeled mean/
   covariance (or finite-MGF) of real source-vs-hard-distractor contrasts on the
   MrPro/Smooth/P2/E1 rows — never measured; no new Qwen table is authorized until it
   is, gate-first against the known contrasts.
5. Panel facts (near/far rows, non-ranking veto, 6Pro endpoints, Smooth decisive
   anti-proxy) are CONFIRMED, not contradicted, by all five audits; only the
   theorem-dress around the veto is downgraded.
