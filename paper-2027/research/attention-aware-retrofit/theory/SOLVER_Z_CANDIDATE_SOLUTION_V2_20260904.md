# SOLVER Z — Candidate Solution V2 (post-veto re-derivation)

> **2026-09-08 锚点身份更正**
> 原文把s2的2.987191/.359361与s4锚点拟作同一full-p2路径；实际s2是C2、s4是
> full legacy-p2，movement不同。后续[原综合更正](../analysis/SINGLE_TABLE_ROPE_OPEN_PROBLEMS_HANDOFF_20260904.md)
> 已撤回混合拟合。本页损伤面/天花板不能作为同路径实测推断继续使用，不能仅称
> “噪声较大”而保留前提。详见[本地复核](../../../../docs/research/ROPE_LOCAL_FAILURE_SYNTHESIS_20260908.md)。

- **Date:** 2026-09-04. **Author:** Solver Z. **Status:** solver candidate, pending round-2 adjudication. Supersedes `SOLVER_Z_CANDIDATE_SOLUTION_20260904.md` (V1), which was vetoed.
- **Trigger:** `ATTACK_2_ON_SOLVER_Z_20260904.md` adjudicated VETO UPHELD (team-lead). Every finding below is dispositioned.
- **Method constraint honored:** no GPU, no model runs; only repo-recorded numbers plus local CPU arithmetic (python). All tables in §3 regenerated from a single canonical script this round to eliminate transcription risk (minor item 3).

## 0. Attack-disposition table

| Item | Finding | Disposition | Where |
|---|---|---|---|
| F1 | "provably/never ≥6" ceiling contradicts own §2.5 class-rejection; attacker countermodel (W1) fits every anchor exactly and admits s5–s8 | **Conceded + re-derived.** Answer line is now an explicit model-set-conditional conjunction (S1, S8). The word "provably" is removed from every ceiling statement; the only "proven" labels attach to the algebraic split identity (S2), the Tier-1 monotone bounds *given declared assumptions* (S4), and the *existence* of witnesses (S8). P-Z1 re-issued in interval form (S16) that does not entail its own negation | S1, S8, S16 |
| F2 | §2.4 bands blended class-only and corner recipes; s5 band needed (f=.15,r=1) which misses s2 task by 0.0156; "~r≤0.3" escape threshold wrong (real r≤0.685 at f=1) | **Conceded + re-derived.** Set A (power exponents [1,2]) and Set B (s2-consistent saturating witnesses) are printed in separate sections, never blended (S6/S7 vs S8). The r≤0.3 escape claim is deleted entirely along with the blended recipe it belonged to | S6–S8 |
| F3 | headline "8x excluded unless sublinear" contradicted statement 9 and the in-class (r=1) 28/81 Iso-s8 rows | **Re-derived + relabeled.** One unified s8 statement (S10). The 28/81 evidence is kept (S7) with its exponent-dependency and s2-inconsistency labels, not deleted | S7, S10 |
| F4 | statement 2 "proven/structurally wrong" presupposed unowned α·d separability (unit-gain table control never run; HANDOFF §8) | **Relabeled.** The split is proven *as algebra at the anchor* (telescoping identity, assumption-free); its portability across s is assumption A1; "structurally wrong" downgraded to "not supported by owned arms". q≈2.83 makes the gain/table attribution α-dependent, noted | S2, ledger A1 |
| Minor 1 | 86.04% vs 86.26% | **Fixed.** Both reported: gain = 86.26% of the arm's total damage increase, 86.04% of the gate budget | S2 |
| Minor 2 | RULER zero-cross value | **Fixed.** 47,759 ≈ 47.8K tokens (was 48K) | S12 |
| Minor 3 | s7 row PG/task transposition | **Re-derived.** All Iso bands regenerated from one canonical script; S7 table is the only printed source | S7 |
| Minor 4 | LAW-Z needed enumerated masks and a defined capability screen | **Relabeled.** LAW-Z demoted to "decision aid"; mask set enumerated {legacy-u p2, exact-u Iso}; capability screen deferred to E2 rather than invented | S5, S13 |
| Minor 5 | statement 6 "proven mechanism" | **Relabeled.** Degeneracy claim made conditional on PG-damage monotonicity in α, which is owned only at two points (native vs α0) | S5 |
| Minor 6 | statement 11 unfalsifiability note | **Kept**, and fit explicitly labeled descriptive/postdiction | S12 |
| Extra (adjudication note) | §2.5 "proven by exhaustion" rested on noise-dominated point estimates (owner bootstrap intervals cross zero) | **Relabeled.** S9 states this; it is the load-bearing premise of the identifiability result, not a discarded weakness | S9 |

Team-lead pre-authorization on record: if the honest re-derivation lands on path-(C) conjunction + frozen split + corrected sensitivity bands, deliver that as the solution. It does; that is S1.

## 1. The answer (statements S1–S16)

**S1 — Answer (path C, conjunction).** The maximum feasible zero-training extrapolation factor S for one static p2 table + one static gain under the two separate 0.875 budgets is **not identifiable from repository numbers**. Specifically: (i) the demonstrated feasible installed object remains the incumbent (p2, s=4, c=.074), PG .875302 / task .915103; (ii) under the declared power-exponent class Set A, the p2 ceiling is 4–5x and the Iso ceiling is 6–7x — printed as *sensitivity values*, not bounds; (iii) there **exist** separable, monotone, s2-consistent damage models (witness W1, constructed in §3.4) that admit feasible windows at s=5,6,7,8 — so no p2 factor in 5–8 is provably excluded under the s2-consistent class either; (iv) arms with α>α0 and c∈{.074,.10} at s≥5 (plus s8/c=.05) fail the PG gate up to an explicitly named kink-pathology exception (§3.5); (v) resolving the ceiling needs at minimum one off-ray task cell per factor (E1). This conjunction is the solution; no single "S=_" claim is defensible.

**S2 — Tier-0 split identity (proven algebra, then one conditional).** The incumbent arm's PG-19 damage telescopes exactly: D_total = D_gain + D_table = .114885 + .018302 = .133187 nats, retention .875302 (gate .875; literal .88 fails — marginal). Gain is 86.26% of the arm's total damage and 86.04% of the .133531 budget; **PG headroom above the incumbent is 0.000344 nats**. At the anchor this is definitional, no modeling assumption. Two conditionals: portability of the split across (α,d) is assumption A1 (separability; the unit-gain table control was never run — HANDOFF §8); and because the s2 PG point implies gain damage ∝ α^2.831 (q fit, §3.4), the 86/14 attribution is α-dependent, not a structural constant.

**S3 — Gate reversal (measured, owned).** At 1x, gain *helps* downstream and *hurts* likelihood: the factorial's gain-only row (owner: NATIVE_ISOTONIC_PROFILE_RESULT §3, 3.085932 / 0.891469 / 0.378216 / retention 1.095851) versus table-only hurting both. Consequence (structure, given A1+A2): PG caps α from above, the five-task gate caps α from below; the feasible set at each s is a window [α_lo(s), α_hi(s)]. Non-obvious corollary: more gain is not monotone-protective for downstream — the task gate is the binding constraint below α0.

**S4 — Tier-1 bounds (proven given A1+A2, no exponent class).** For any separable damage model with both components monotone nondecreasing in their argument, consistent with the owned anchors: (i) any arm with d≥d0 and α≤α0 has five-task retention ≤ **.915103** (the anchor's own value — displacement can only add table-task damage, gain benefit is capped by b(α0)); (ii) any arm with α≥α0, d≥d0 has PG retention ≤ **.875302**, so passing .875 requires total joint damage rise over the anchor ≤ **0.000344 nats**; (iii) the s2 anchor forces D_gain(α≤.5α0) ≤ .016144 = 14.1% of D_gain(α0): gain damage concentrates in the last octave of α, and no measurement exists above α0.

**S5 — Stage-Z protocol degeneracy (conditional; decision-aid status).** If PG damage is nondecreasing in α across the owned range — owned only at two points (native, α0), so this is an assumption, not a measurement — the PG-calibrated gain rule in the Stage-Z preflight selects c=.05 at every s>4 (the only grid value that can pass PG by S4(ii) reasoning), and c=.074/.10 arms are then pre-excluded on PG, not capability. The protocol therefore reports ceiling ≈4 for p2 whenever any saturating model admits more. LAW-Z is demoted accordingly: the window calculator is a decision aid over the enumerated mask set {legacy-u p2, exact-u Iso}, not a capability metric, and its mask inputs are measured marginals, not derived quantities.

**S6 — Set A sensitivity ceiling, p2 (declared class; NOT s2-consistent).** Model: D_PG(α,d)=Dg(α/α0)^q + t_g(d/d0)^p, Δtask = t_t(d/d0)^r − b(α/α0)^f, exponents {q,p,f,r}∈{1,1.5,2}. Feasible-window census over the 81 exponent grid: s≤4.5 → 81/81; s=5 → **27/81, and every surviving member has r=1** (table-task damage exactly linear in displacement — a labeled dependency, the p2 analogue of the V1 Iso-s8 issue); s≥5.5 → 0/81. Band values at protocol arms: s5/c.05 PG [.8916,.9122] task [.8154,.8653]; s5/c.06 PG [.8756,.8842] task [.8413,.8804]; s6/c.05 PG [.8773,.8947] task [.7712,.8501]; s8/c.05 PG [.8528,.8660] task [.6905,.8273]. Verdict under Set A: ceiling 4–5 (p2), s6+ task-infeasible. Set A best fit misses s2 task by .0064 (see S9) — these are sensitivity numbers, not bounds.

**S7 — Set A sensitivity ceiling, Iso (same class; s2-Iso anchor does not exist).** Census: s≤6 → 81/81; s6.5 → 60/81; s7 → 54/81; s8 → **28/81** (kept per F3 disposition). Bands at protocol gains (regenerated, no transposition): s5/c.05 PG [.9058,.9247] task [.8881,.9280]; s6/c.05 PG [.8947,.9084] task [.8613,.9199]; s7/c.05 PG [.8853,.8936] task [.8364,.9130] (at c=.052: PG [.8814,.8861] task [.8435,.9166]); s8/c.05 PG [.8757,.8814] task [.8119,.9083] (at c=.052: PG [.8673,.8773] task [.8158,.9163]). Verdict under Set A: Iso sensitivity ceiling 6–7 at mid-band, 8 only in the 28/81 corner. Caveats printed with the numbers: corner rows are exponent-dependent; there is no s2-Iso anchor so s2-consistency is untestable for this mask; task bands exceeding 1 at large α are artifacts of the unbounded benefit term and are clamped interpretively at 1.

**S8 — Identifiability result (proven existence; the load-bearing new statement).** Two explicit witnesses:
- **W1 (saturating, s2-consistent):** q=2.8311147; table-PG piecewise-linear through (x=.5, 0), (x=1, .018302), (x=log6/log4, .0185), plateau beyond; table-task piecewise-linear through (x=.5, .0023142), (x=1, .062383), (x=log6/log4, .07475), plateau beyond; b linear (f=1). It reproduces s2 PG exactly (.016144 predicted = .016144 observed) and s2 task exactly (.359361 = .359361), and all four anchor arms by construction. Admitted windows: s5 α∈[.0808,.1027] (c∈[.0502,.0638]); s6 α∈[.0980,.1026] (c∈[.0547,.0573]); s7 α∈[.0980,.1026] (c∈[.0504,.0527]); s8 α∈[.0980,.1026] (c∈[.0471,.0494]). ⇒ **No p2 factor 5–8 is provably infeasible from repo numbers under s2-consistent separable monotone damage.**
- **W2 (power, s2-consistent):** f=.15, r=2, q=2.8311, p=1; s2 task fit .3593534 (8e-6 off). Predicts (p2,s6,c=.05) task .7920 — infeasible. ⇒ No factor 5–8 is provably *feasible* either: W2 is equally consistent with every owned number.
A third legal member — table-task damage steep to x=1 then exactly flat (most extreme saturation) — gives s6/c.05 task **.9030**, passing. The repository 1x statistics contain no datum that separates these; the three-arm incumbent evidence lies on the single collinear ray α=.074·log s.

**S9 — Noise dominance (why the s2 veto cannot rescue Set A, relabeled per adjudication note).** Set A's rejection of s2 rests on point estimates only: best Set A member misses s2 task by .0064 macro — inside the owner's paired-bootstrap interval for that comparison, which crosses zero ([-.036407,+.073927] task; [-.025070,+.000937] PG; NATIVE_ISOTONIC_PROFILE_RESULT §3, owner rows). "Proven by exhaustion" is a deterministic proof on statistically unresolved anchors. This does not weaken Set A as a *sensitivity* family; it forbids using s2 to certify Set A as *the* model.

**S10 — Unified s8 statement (replaces V1's contradictory pair).** p2 at s8: every model that does not place a gain-damage kink within 1.4% of α0 fails the PG gate at all protocol gains — Set A bands [.8528,.8660] at c=.05, [.7411,.8189] at c=.074; W1's continuing power gives .8714. Only the named kink-pathology (S4-iii, §3.5) could pass, and only barely. Iso at s8: passes both gates inside the 28/81 Set A corner (S7), untestable against s2. Neither arm is decidable at Tier-1. One sentence, both masks, same logic — the V1 pairing of "8x excluded" with "no measured mask s8" while printing 28/81 rows is withdrawn.

**S11 — Near-model-free exclusion list (tightened, kink-exception named).** Given A1+A2 only (no exponent class), and excluding "kink pathologies" — models whose joint damage rise over the interval between an anchor and the arm is ≤ 0.000344 nats despite α or d strictly increasing (i.e., saturation placed exactly at the selected operating point) — the following fail the PG gate: all p2 arms with c∈{.074,.10} at s∈{5,…,8} (α/α0 ≥ 1.161) and (p2, s=8, c=.05) (α/α0 = 1.0135). The c=.05 arms at s∈{5,6,7} survive PG (α<α0) and are decided by the task gate, which S8 shows is unresolvable. Label: Derived-under-assumption, not Observation.

**S12 — Capability side (descriptive only; note kept).** RULER-13 exact-match log2-quadratic fit p(t)=−.060769t²+.013846t+.713974 (reproduces owner §4.1 points; a=−.04949337/b=.1050045 fit): p(32K)≈.209, p(5x)≈.418, extrapolated zero-cross ≈47.8K tokens (47,759). Postdiction; never a capability claim. The §4.3 dilution-vs-phase decomposition is **uninstantiable**: inputs ρ, n0, μ, B, w, η, Hbar, κ_η all unavailable in repo, and a single length cannot separate log-q dilution from power-law phase decay — this is why the answer cannot route around the identifiability wall via mechanism.

**S13 — Selection-law status.** No frozen (table, gain) selection law arguable to ≥5x exists, because the law's inputs are under-identified: both coefficients (c, and the table's displacement exponents p, r) are identified by at most three collinear arms plus one off-ray point (s2) whose comparison is noise-dominated (S9). LAW-Z is retained solely as the §3.3 window calculator with S5's demotion label.

**S14 — (C)-core, the minimal relaxations.** Z as posed is unresolvable because four assumptions bind simultaneously: A1 separability (never controlled), collinearity of owned arms, saturating-model freedom past the anchors (S8), and noise-dominated s2 (S9). Minimal relaxations, named precisely: **R1** one off-ray task cell per factor (breaks collinearity — E1); **R2** the unit-gain table control at 1x (directly tests A1); **R3** ≥40 replicates of an s2 comparison to resolve S9; **R4** a second evaluation length (tests the kink exception and §4.3). Any one of R1–R2 upgrades part of the answer; all four upgrade the ceiling question itself. Each is ≤ a handful of already-authorized Stage-Z-scale runs.

**S15 — Deployable today (unchanged).** The only installed object meeting both literal budgets is (p2, s4, c=.074): PG .875302 (marginal band [.875,.88)), task .915103. Recommended probe, not claim: (p2, s4, c=.063) is predicted to clear *both* 0.88-literal readings under Set A — PG [.8904,.9034] (q=2.83 point: .9129), task [.8887,.9009] — with monotone-model plateau extremes .8753/.9151 and sampling noise per S9. It buys literal-.88 headroom on a marginal arm, zero additional extrapolation.

**S16 — P-Z1′ (re-issued frozen prediction; interval adjudication, contains no self-negation).** Before any new run, pre-register: (a) observed (p2, s=6, c=.05) five-task retention lands in exactly one adjudication cell: **[.7712,.8501]** → power class retained, saturating witnesses (W1=.8671, flat-table=.9030) refuted at factor 6 → Set A ceiling 4–5 stands as a bound, not a sensitivity; **[.86,.915]** → power class refuted at factor 6, table-task damage saturates past the anchor → the ceiling question reopens toward 8x at off-grid gains c∈[.047,.057]; **≥.915103** → Tier-1 bound S4(i) broken → A1/A2 (separability or monotonicity) refuted outright — every outcome is a result, and the prediction is the map, not a number; (b) observed (p2, s=8, c=.05) PG retention **<.875**, with passage (≥.875) counting as discovery of an α0-coincident damage kink — which is itself reported as selection-artifact evidence, not as 8x capability. Post-outcome this remains postdiction until E1/E2 land.

## 2. Derivations

### 2.1 Tier-0 identity (§ S2)
α0 = .074·log4 = .1025857827228719. Installed-arm damage .133186539 → exp(−·) = .87530180 (owner-recorded .875302 ✓ within rounding). Dg = −log(1.095851-normalized gain-only row) = .114884539; t_g = .133186539 − .114884539 = .018302. Budget .133531047 (exp-of-.875); headroom = .000344854 nats. Gain share: .114885/.133187 = .86258; /budget = .86036. Task tolerance: 1−.875 = .125 of NT=.345134307 → .043141788 macro; anchor Δtask = t_t − b = .062383 − .033082 = .029301 ≤ .043142 ✓ (.915103 ✓).

### 2.2 Tier-1 bounds (S4)
Given A1+A2: d≥d0 ⇒ t_t(d) ≥ t_t(d0); α≤α0 ⇒ b(α) ≤ b(α0). Retention_task(s,d) = (NT − t_t + b)/NT ≤ (NT − t_t(d0) + b(α0))/NT = .915103. Symmetric for PG. (iii): W1's q solves .114885·.5^q = .016144 ⇒ q = log(.114885/.016144)/log2 = 2.8311147; any monotone model has D_g(.5α0) ≤ .016144 by the s2 PG anchor (table-PG ≥ 0, A4).

### 2.3 Set A windows and bands (S6/S7)
Feasibility on the 81-member exponent grid {1,1.5,2}^4: α_hi(s) solves Dg(α/α0)^q + t_g(log s/log4)^p = .133531 (exists iff table-PG term < budget); α_lo(s) solves b(α/α0)^f = t_t(log s/log4)^r − .043142 (clipped at 0). Census and arms printed in S6/S7. The r=1 dependency at p2-s5 (27/81 all r=1) and the 28/81 Iso-s8 corner are the two F2/F3-relevant structures; neither is blended across sets anymore.

### 2.4 Witnesses (S8)
W1/W2/W3 (flat-table) definitions in S8; verification digits: W1 s2-PG .016144 vs .016144, s2-task .359361 vs .359361 (exact by construction: the piecewise nodes are the anchors themselves); W2 s2-task .3593534 (8e-6 off), s2-PG not exactly reproduced (its p=1 power gives .02529 at x=.5 — W2 is included because task, the binding gate at s6, is reproduced; a p-piecewise variant reproduces both but was not needed). Flat-table member: s6/c.05 task .9030; W1 same arm .8671; Set A band max .8501.

### 2.5 Near-model-free exclusions (S11)
(p2, s, c=.074), s≥5: α/α0 = log s/log4 ≥ 1.161 ⇒ even with table-PG flat at .018302, monotone gain damage with strictly-increasing members violates headroom .000344 unless kinked at α0. c=.10 same at ratio ≥1.569. (p2, s8, c=.05): α/α0 = 1.0135135 ⇒ joint rise must be ≤ .000344 over a 1.35% interval. c=.05 at s∈{5,6,7}: α<α0 ⇒ no PG information at all — survival is the assumption-free side.

### 2.6 Descriptive fits (S12)
§4.1 quadratic reproduced (boundaries 3.9115/4.0057); RULER-13 fit p(t)=−.060769t²+.013846t+.713974, p(3)=.2086, p(log2 5)=.4185, root at t=47759 → 47.8K.

## 3. Assumption ledger

| ID | Assumption | Status / owner |
|---|---|---|
| A1 | Damage separable in (α,d) | **Unowned** — unit-gain table control never run (HANDOFF §8); all S4/S11 claims conditional |
| A2 | Component damages monotone nondecreasing in their argument | Plausible, untested past anchors; S5 explicitly conditional on it |
| A3 | s2 anchor consistency as model-class veto | **Invalid** — noise-dominated (S9, owner bootstrap intervals cross zero) |
| A4 | Table damages ≥ 0 | Definitional |
| A5 | Set A exponent grid [1,2] | Declared sensitivity family, not a claim about the world |
| A6 | Anchor marginals (t_g, t_t, Iso t^I) port unchanged across s | Needed by every table in §2.3; untested |
| A7 | No kink pathology at α0 | Used only in S11's "up to" scope; P-Z1′(b) tests it |
| A8 | Mask marginals are complete descriptors of mask quality | Assumed; E2 adjudicates |

## 4. Discriminating experiments (each separates two named explanations)

- **E1 — off-ray gain cells: class vs saturation vs interaction.** Arms (p2,s5,c=.06), (p2,s6,c=.06) (PG-safe, α<α0) and (p2,s4,c=.10) (α>α0 at the anchor displacement). Readings: (i) s5/s6 task in Set A positions ⇒ power class; at/above W1/flat positions ⇒ saturating family (table damage plateau past anchor); (ii) (s4,c=.10) PG: <.875 ⇒ no right-plateau at α0 (S11 hardened, A7 disposed); ≥.875 ⇒ kink confirmed ⇒ *all* α>α0 exclusions die and the headroom argument collapses to the anchors; (iii) any task reading exceeding both separable predictions by >2·SE ⇒ A1 violated ⇒ interaction model. Disables the S8 ambiguity without long-context evaluation.
- **E2 — Iso arms (s6, s7) at c≈.052:** tests A8/mask efficiency — "p2's Set-A ceiling of 5 is mask-specific" (Iso passes s6 where p2 fails) vs "the task ceiling is a gain-policy property that both masks share."
- **E3 — two-length PG probe at s=4 and s=8 (short/long halves):** only design that separates dilution from phase decay for §4.3 inputs; also re-measures the headroom at two lengths, testing whether 0.000344 nats is a length-specific accident.

## 5. Status table

| Claim | Label |
|---|---|
| Split identity + 86.26%/86.04% + headroom .000344 | **Proven** (algebra at anchor) |
| Gate reversal (gain helps task, hurts PG) | **Observation** (owner factorial row) |
| Tier-1 bounds .915103 / .875302 / q=2.831 | **Derived** under A1+A2 |
| Set A ceilings 4–5 (p2) / 6–7 (Iso), 27/81 & 28/81 corners | **Sensitivity** (declared class; not s2-consistent) |
| Existence of W1/W2/flat witnesses admitting or refuting s5–s8 | **Proven** (constructive) |
| Unidentifiability of the true ceiling from repo | **Proven** given A1+A2 (S8 ⇒ no member of the consistent class is decidable) |
| Stage-Z degeneracy | **Derived** under A2 (two-point monotonicity only) |
| RULER decay fit / 47.8K zero | **Descriptive postdiction** only |
| c=.063 clears .88 at s4 | **Set A prediction**, plateau + sampling caveats (S15) |
| V1 "provably never ≥6" / blended corner bands / dual s8 statements | **Withdrawn** (F1/F2/F3) |
| V1 "8x near-model-free excluded" unqualified | **Conceded** (F3, W1) |

## 6. Owner register (values → source, paths relative to the repository root)

- Incumbent double gate, L0 guard, s2 zero-refit (PG 2.987191, task .359361, core-4 .5150), 2x2 weights-by-table, s8 core-4 rows: `paper-2027/research/attention-aware-retrofit/results/SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md`
- Gain-only / Iso factorial rows and the two bootstrap intervals ([-.025070,+.000937], [-.036407,+.073927]): `paper-2027/research/attention-aware-retrofit/results/NATIVE_ISOTONIC_PROFILE_RESULT_20260903.md`
- Protocol (factors 4–8 × gains .05/.074/.10, PG-calibrated gain, exporter law m_k = −log(ω_s4/ω_native)/log4): `paper-2027/research/attention-aware-retrofit/preflights/CONSTRAINED_FRONTIER_AND_2X4X_LORA_PREFLIGHT_20260904.md`
- §4.1 quadratic, §4.3 two-ceiling decomposition and input list, §8 unit-gain-control absence, §9 do-not-revive list, §10 safe statements: `paper-2027/research/attention-aware-retrofit/analysis/SINGLE_TABLE_ROPE_OPEN_PROBLEMS_HANDOFF_20260904.md`
- Selection tree / incumbent provenance: `paper-2027/research/attention-aware-retrofit/theory/SINGLE_STATIC_LOG_P2_SELECTION_AND_LORA_20260903.md`
- K32 family (distinct, not p2 evidence): `paper-2027/research/attention-aware-retrofit/results/K32_NORMALIZED_INDEX_FULL13_CONFIRMATION_RESULT_20260901.md`
- Derived quantities (§2.1, §2.3–§2.6): local CPU python recomputation, this memo. V1: `.../theory/SOLVER_Z_CANDIDATE_SOLUTION_20260904.md` (superseded). Attack: `.../theory/ATTACK_2_ON_SOLVER_Z_20260904.md`.
