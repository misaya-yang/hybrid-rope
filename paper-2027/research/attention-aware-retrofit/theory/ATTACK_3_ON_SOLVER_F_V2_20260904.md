# Attack 3 on SOLVER_F_CANDIDATE_SOLUTION_V2_20260904 (second-wave adversarial review)

- **Date:** 2026-09-04
- **Role:** veto-seeking reviewer (Solver F V2 targeted); attacks must be NEW
  relative to ATTACK_1_ON_SOLVER_F_20260904 and must audit the §0 dispositions.
- **Verification actually run:** full read of V2, round-1 memo, attack 1,
  handoff §2/§3.2/§5/§8, preflight, all three observation owners
  (EVQ/Obs1, OLMO2/Obs2, LOG_P2/Obs3); trainer script
  `scripts/train/train_log_p2_phase_transfer_lora.py` (pair-view contract,
  lines 42–71); Python recomputation of every cited number, the pair-fraction
  formulas (analytic + brute-force Monte-Carlo over the causal triangle), the
  GeLU tail bounds (erf-GELU and tanh-GELU variants), the composite-retention
  arithmetic, the 16K↔32K feature-separation feasibility numbers, and the
  band-endpoint derivations re-done from P1–P4 independently.

## 1. VERDICT: **NO-FATAL-BREACH**

Lemma F1′ survives a step-by-step severe audit (certified below, with one
carrier sentence to be corrected). All arithmetic reproduces. The A2/A3/A4/A5/
A6/A7/A8/A9 dispositions close their aired holes as far as I can test on this
machine. No FATAL at the veto standard was found. But two attacks are real
**soundness bugs that change falsification labels as printed** and must be
fixed before any decision-file pre-registration: (A-1) â_hi is not an upper
bound under P2 as stated (missing +B·o_S — a full counterexample given);
(A-2) T1a/E1's slope-1-in-nominal-band-units test is not a consequence of P1
(effective-count semantics per the memo's own S4/A9 machinery) and can cancel
the one authorized reveal while recording a premise-consistent world as a
P1-window falsification. Both are one-term/one-line repairs inside the same
architecture with S6(iii) discrimination preserved post-fix, so per the
adjudication standard they are graded MINOR-to-MODERATE, not FATAL — a stricter
reading of "a disposition whose fix does not fully close the aired hole" could
re-grade A-1 against the §0 A1 row ("re-derived, not patched" is not true of
the band's *upper* endpoint as printed). I state that boundary openly rather
than inflating the grade.

## 2. Attacks, ranked

### A-1 — MODERATE (band-soundness bug; repairable, discrimination preserved) — â_hi omits the numerator's upward degree of freedom that its own â_lo term prices

**Target:** S4 â_hi(S) = E_rows[P̂(Ĝ − Δ_S·ρ̂_− + ε̂)] and the claim
"under P1–P4: predicted accuracy at length S ∈ [â_lo(S), â_hi(S)]";
S6(iii) "if measured > â_hi at some S, the optimistic clause (P1-ε̂ sign or
ρ̂) is falsified upward"; S6(i) veto "reveal iff … â_hi(8x) ≥ τ".

**Mechanism.** S3-P2's defining clause is two-sided: **|∂z_R/∂octave| ≤ B**,
extended to blind octaves with λ ∈ [0, B]. The lower endpoint subtracts
B·o_S for worst-case numerator *decay* — correct. The upper endpoint assumes
zero numerator *improvement*. An environment in which every premise holds and
z_R grows at exactly +B per octave beyond 16K (permitted by the |·| clause;
nothing in P1–P4 excludes numerator improvement beyond window — in fact
Obs2's own substrate story is a numerator *gaining* from 8K to EVQ-adapted
states) yields blind margins up to B·o_S above â_hi's argument, hence
a(S) > â_hi(S) with P1–P4 all true.

**Demonstration.** Take ε = 0, ρ = ρ̂_−, λ = B (all inside the premise class).
True margin = Ĝ − Δ_S·ρ̂_− + B·o_S > â_hi's argument Ĝ − Δ_S·ρ̂_− + ε̂
whenever B·o_S > ε̂. o_S = 1 at 8x: with B̂ anywhere ≥ a couple tenths nats
(the in-window source-band decay B̂ is measured from E1 "source-band margin
decay", whose sign is negative in-window but whose |·| bound licenses the
positive extension), the violation is large. Then S6(iii) records
"optimistic clause falsified upward" — a *premise-consistent* outcome is
logged as a falsification. And S6(i): â_hi as printed is below the licensed
optimistic endpoint, so the veto can fire (declining the single authorized
reveal) while the memo's stated veto justification — "paying to avoid a
reveal whose optimistic premise-case prediction is below usefulness" — is
false in exactly the worlds it declines. Note the internal tension with the
memo's own §8.3 ("a numerator curve … with out-of-window slope = B is the
band's honest worst case — the width … is therefore expected to be large"):
the worst case is applied to one endpoint only. The alternative reading —
"shift ≤ λ·o_S" is a *one-sided decay-only* clause — makes â_hi valid but
then (i) contradicts the |·| clause two sentences earlier, and (ii) silently
assumes "the numerator cannot improve beyond window", which is precisely the
round-1 A1 hidden numerator invariance re-entering through the other door.
Either way the printed band's printed premises are mismatched.

**Grade:** MINOR-to-MODERATE. Repair: â_hi += B·o_S (one term; estimators,
rules, and architecture untouched). **Post-fix S6(iii) still discriminates:**
outside-the-symmetrized-band again implies a genuine premise violation, the
upward-falsification label becomes sound, and the veto's justification
sentence becomes true. Pre-fix it mislabels evidence, which is why this is not
a taste note. Disposition-adjacent consequence: §0's A1 row "Re-derived, not
patched … rebuilt as two-sided band (S4)" is not fully achieved — the two-
sided band is one-and-a-half-sided.

### A-2 — MODERATE (decision-gate semantics bug; patch-level, discrimination preserved) — T1a's slope-1 gate is stated in nominal band units but P1 licenses growth only in effective-count units; a premise-consistent substrate can be vetoed and recorded as a P1 falsification

**Target:** S5 T1a ("pooled within-row regression of band-cumulative
distractor LSE on ln n over n∈{4K,8K,12K,16K} … require slope ∈ [1−0.15,
1+0.15] on the CI") and its scope claim "a failure falsifies P1/P2";
E1 ("slope of Ĝ vs ln n′ ≈ −1" over nominal truncation counts n′);
S6(i) ("reveal iff … T1 passes").

**Mechanism.** The memo's own P1 (and round-1 A1a, which S3 keeps) defines
the dilution law in **effective** competitor counts, and S4 spends an entire
parameter ρ̂ ("median effective/nominal competitor ratio with its dispersion")
precisely because nominal ≠ effective — the A9 correction, which the §0 table
marks "Fixed". LSE_D over the cumulative *nominal* bands with attention
concentration that varies across distance bands (n_eff(n) = ρ(n)·n, ρ
decreasing as far bands add near-zero-mass keys) gives
dLSE/dln n_nom = 1 + dln ρ(n)/dln n, which departs from 1 while P1 holds
*exactly* (ε = 0). Such a world fails the ±0.15 gate ⇒ the protocol (i)
records "falsifies P1-window" (a false attribution to a premise that never
failed — the constant-ρ *modeling choice*, the thing §8.4 admits is "not a
theorem", is what died), and (ii) declines the reveal (T1 pass is required in
S6(i)). A premise-consistent environment is denied its one-shot unconditional
demonstration leg on a mislabeled pre-text. E1's crossed-predictions claim
suffers the same way: "H-dilution vs H-phase predict opposite T1a results"
holds only under n-flat ρ; H-dilution-with-declining-ρ predicts slope ≠ 1
and mimics the H-phase signature, collapsing the discrimination E1 advertises
("no outcome is compatible with 'nothing was learned'" becomes false — a
sublinear slope is compatible with both hypotheses).

**Demonstration.** Concrete: per-query attention over distractors with
exponential distance scale λ: at 16K rows with λ = 4K, effective count grows
as ~min(n, 1/λ)-ish; LSE over nominal bands {4K,8K,12K,16K} saturates and
regression slope ≈ 0.6–0.85 with tight CIs — T1a fails hard while P1 (in
effective counts) is satisfied to ε ≈ 0. The mass-vs-count arithmetic also
exposes a related wording slip in S3: "76.6% of key–query pair **mass**" —
my computation: pair *counts* beyond 16K are exactly the memo's
(1−D/n)² = 25.0/56.3/76.6% (verified analytic + Monte-Carlo over the causal
triangle), but *attention-mass-weighted* fractions are far below that (≈1%
at λ=2K scale; a few % at realistic λ=4K) — the mass statement understates
nothing for the concession (count is the conservative statement for
population coverage), but the word "mass" should be "count".
**Fix:** regress on ln n_eff measured in-window via the same participation-
ratio statistic that defines ρ̂, or relabel the failure target as "constant-ρ
in-window (§8.4 choice) + P1-window jointly" and price the ρ-slope range into
the band width. Architecture and S6(iii) discrimination preserved.

### A-3 — MINOR — the band dropped P̂'s own fit uncertainty that round 1 carried, and the 32x lower endpoint is near-vacuous; "licensed by what can and cannot be named" over-names the coverage

**Target:** S4 ("Band width is licensed by what can and cannot be named");
S6(iii) four reading rules; §8.3–8.4 disclosures.

**Demonstration.** (i) Round-1 S5 recorded predictions "with intervals
inherited from ε̂ **and fit uncertainty**"; V2's endpoint terms are
Δ·ρ̂_±, ε̂, B·o_S only — no (α̂, ĝ_c) CI. For a 2-parameter logistic evaluated
partly *outside its fit support* (at 8x, â_lo's argument reaches ~1–1.5 nats
below the in-window margin range; at 32x, ≈2.08·ρ̂_+ + ε̂ + 3B̂ ≈ 2.5–3+ nats
below), the endpoint depends exponentially on α̂ whose sampling error from a
few hundred rows is unpropagated. P3's literal words ("the *fitted* map P̂ …
apply") license *using* the fit, but the band's width does not price its
uncertainty — so the sentence "licensed by what can and cannot be named"
is too strong for its own list. (ii) Consequence for S6(iii) reading rules:
at 32x, â_lo ≈ E[P̂(deep-left tail)] ≈ 0, so "measured ≪ â_lo ⇒ P1–P4
falsified" can essentially never fire at 32x — the 32x test is upper-side-
only plus the ordering check; the rules do not say so. (iii) The lower
endpoint's approach to zero is *sound* (vacuous, not violated) so this is
disclosure-grade, not unsoundness. Fix: propagate a bootstrap CI of P̂ into
both endpoints (or state â_lo(32x) as informational only).

### A-4 — MINOR (wording inside the certified lemma) — S2 step 2's carrier sentence is mechanically false as written; the construction is saved by step 1's own constant, via the value path

**Target:** S2(2): "c₀ is carried as a learnable constant direction in the
residual stream (a value-aligned BOS-type key contributes a fixed vector at
every position)".

**Demonstration.** Through softmax attention a single BOS-type key's value
contributes p₀·v₀ at every position, and p₀ = e^a/(e^a+(n−1)e^b) = Θ(1/n)
for any fixed logit gap a−b — the contribution decays at exactly the rate
the gate is built to read, so it cannot carry a *fixed* offset (a persistent
sink needs a logit gap growing like ln n, which reintroduces the very
n-dependence). The correct carrier is already in the construction: the
shared value constant c in v_j = c + ξ_j (step 1) survives the attention
mean at every position (h_n = c + m_n) and survives RMSNorm: with x =
‖m_n‖/‖c‖ = Θ(√(d/n)), the normalized feature x/√(1+x²) is strictly
monotone in x hence monotone in n, so one threshold c₀ separates both
scales — which answers the lead's RMSNorm make-or-break question directly
(verified analytically; the renormalized gap 16K↔32K stays ≈29% of the
smaller value). RoPE makes key-path constants impossible for off-diagonal
logits (R(Δ) rotation); the rotation-invariant self-position term q̃·k̃ is an
alternative but unnecessary route. Fix is one sentence (name the value-path
shared constant as carrier); the lemma's claim is unaffected (see §4).

### A-5 — MINOR (scope note) — S1/S2's class restriction must be read as banning any >16K-length *computation*, including protocol-authored blind-length probes; the user firewall's words ("8x/16x/32x rows never enter…") arguably ban only the sealed rows

**Demonstration.** The S2 environment pair is distinguishable by running the
model on synthetic 32K strings (the gate fires there), and by direct weight
inspection (the gate's weights exist in E and not E′). The lemma's
indistinguishability is over *row statistics at ≤16K* only. The memo's class
("trains and selects only on ≤4x-visible statistics") does exclude those
routes, but handoff §8's literal rule ("8x/16x/32x rows and outcomes never
enter model or hyperparameter selection") most naturally scopes to the sealed
rows. Nothing in the delivered protocol exploits the gap (S12 keeps it clean
and the single-reveal structure forbids synthetic blind probing in spirit),
so no decision changes — but S1(i)'s headline "no protocol can guarantee"
should say "no protocol forbidden blind-length computation" to avoid a
reviewer finding the theorem class narrower than the firewall letter.
Wording only; MINOR by definition.

## 3. Repair-that-must-not-count (had these been FATAL) — recorded for the fix round

- For A-1: widening ε̂ or ρ̂_± to swallow the missing numerator upside would
  count as parameter-bloat of the assumption (the exact move round-1 A1 ruled
  non-admissible); the only admissible fix is the stated ±B term on â_hi or an
  explicit one-sided-decave restatement of P2 with its disclosure.
- For A-2: re-tuning η from ±0.15 to fit whatever slope the substrate shows
  (post-E1) would make T1 unfalsifiable-by-construction; the admissible fix is
  effective-count units or an honest relabel of the falsification target.

## 4. Certified clean (said plainly, because survival matters)

1. **Lemma F1′ survives every step I could attack.**
   *Step 1:* bias-free 1/n-class feature exists; RMSNorm preserves it (mono-
   tonicity proof above); realized per-position separability works via the
   norm/√variance reading, and I checked the concentration budget: relative
   std of the head-concat mean = 1/√(2·1024) ≈ 2.2% vs the 16K↔32K gap of
   29.3% = 13.3σ, so a strict order-statistic gap over the finite realized
   visible sample (~10⁵ answer positions, extremes ≈ 4–5σ) is realizable; the
   pure second-order mean-shift reading is per-position noise-dominated
   (fluctuation σ/√n ≫ shift σ²/n) unless σ_w is scaled ≳10³ (also feasible,
   constructor-free) — the memo names both readings, so no step *fails*;
   threshold interval (k/32K, k/16K) nonempty ✓.
   *Step 2:* weight-space constants are realizable in bias-free RoPE (shared
   value constant + RMSNorm scale γ read through W₁; self-position logit term
   rotation-invariant) — see A-4 for the one false sentence.
   *Step 3:* bounds verified numerically: |GELU(−t)| ≤ (t+1)e^{−t²/2} and
   |GELU′(−t)| ≤ (t+1)e^{−t²/2} hold at t ∈ {0.5…60} (the true bound is
   ≤ φ(t), so the printed one is loose in the right direction); g=60 ⇒
   log₁₀ ≈ −779.94 ≈ "−780" ✓; the T-step trajectory argument is standard
   Lipschitz-compounding absorbed in C ✓. BF16/float question resolved as the
   lead suspected — the contribution underflows *exactly to zero* (erf-based
   Φ(−t) saturates at t ≳ 6; tanh-GELU decays as e^{−ct³}, even faster), so
   realized float trajectories are bitwise identical and no float-computable
   statistic can see the gate: finite precision **strengthens** the lemma;
   confirmed, not nitpicked.
   *Step 4:* consistency checks pass — β > task margin realizable with the
   same W that produces g (split c₀ within the gap; both margins scale with
   W); answer spans at ≤16K see n ≤ 16K (causality) ✓; the 18K–32K crossing
   region is unvisited ≤4x ✓; second-order leakage ≤10⁻⁷⁸⁰ ✓.
   With step 2's sentence fixed: **certified as a construction-level derived
   result.** S1(i), S3's P3-pricing, S6's veto-status downgrade, and S14 all
   stand on it.
2. **All numerics reproduced clean.** Pair counts: 1−(2D/n−(D/n)²) ≡
   (1−D/n)² = 0.2500/0.5625/0.765625 ✓ (formula is exactly the uniform-
   causal-triangle pair fraction; brute-force MC 0.2501/0.5623/0.7653 ✓);
   attacker-1's 46.7/74.2/87.3% ✓ (source-uniform × query-at-end population;
   the memo's "differ only in which query set is averaged" is TRUE, checked);
   composite 0.875302·e^{0.014107} = 0.8877374, direct exp(−(3.090126−
   2.971047)) = 0.8877377 ✓ (memo's "to 7 decimals" should say 6 — trivia);
   0.875302 = exp(2.971047−3.104234) ✓; −ln0.88 = 0.127833, −ln0.875 =
   0.133531 ✓; ln2/ln4/ln8 ✓; τ anchor: Obs3 §5 frozen 16K core-4 = 0.4025 ✓;
   Obs3 CI [+0.009616,+0.019405] ✓ owner; Obs2 column labels ✓ (21.5/17.5/4.0
   EVQ exact vs 22.0/0/0 Native exact; 42.44/31.63/5.03 vs 72.19/2.02/0.38
   macro — "0%" is QA-exact ✓); Obs1 ✓ (18.75→64.06, +1.5055 vs −0.0095,
   rank 2043.0, 0.1126 vs 0.2110); o_S = "1/2/3 octaves" decodes correctly as
   {1,2,3} = log₂(S/16K) ✓ (matches Δ_S = ln2/ln4/ln8 convention).
3. **Disposition audit.** A2 ✓ (three forbidden strings actually deleted; veto
   is compute-avoidance; false-negative capacity named; measured lengths
   unconditional). A3 ✓ (61.0% EOS at 8K and 200/200 empty are OLMo-owner
   facts, now correctly attributed; note the 200-empty fact carries 100% EOS
   so it motivates D2's *content* half, not the terminal half — the sentence
   leans on it for both; immaterial). A4 ✓ performed, not promissory: Obs3
   records frozen 3.104234 itself (§3 table) and the table hash (§2), so the
   row-set identity check is a manifest comparison on the work machine and the
   six-decimal NLL agreement already near-certifies it; the blended 0.8982 is
   dropped and replaced by own-row gate evaluation ✓. A5 ✓ (frozen passes per
   owner; protocol-vs-substrate attribution corrected; (a′)/(a) split legal).
   A6 ✓ two-pass scheme costed honestly; my forward-equivalent count gives
   ≈+55–75% vs the memo's ≈+50% — an under-estimate, not a breach (no budget
   claim gates the protocol; 300 steps preserved). A7 ✓ (gate moved to base
   weights — the A7(i) hole is genuinely closed; gradients now quantified —
   A7(ii) closed; both value and gradient gaps proven; only step-2 carrier
   wording remains, A-4 above). A8 ✓ substantially (Spearman deleted, slope
   CI has power — with the units caveat A-2; the 1x anchor *is* defined —
   see §4.5 — so ε̂ is not undefined). A9 ✓ (columns fixed; nominal/effective
   gap routed through ρ̂ — but see A-2: the correction was applied to S4 and
   NOT to S5/E1).
4. **S13 demonstration-language sweep.** Every occurrence of "demonstration/
   demonstrated" (S1(ii), S6(iii), S8, S13 bullets 1–2, S14, E2) is confined
   to measured lengths; 32x-when-floored is explicitly routed under P1–P4
   (β); the ordering test (8x ≥ 16x ≥ 32x) is a *mechanism* credit rule,
   conservative in the right direction (non-monotone in-band outcomes are
   premise-consistent, and the memo only withholds credit when it fails — it
   never claims falsification from ordering alone). No silent support for
   unmeasured lengths found. The post-reveal "direction" attribution (which
   premise failed) is legitimately obtainable post-reveal from the observable
   G = z_R − LSE_D decomposition on opened rows — not a pre-reveal
   identifiability claim.
5. **Surface 5 (geometry carve-out) — firewall-clean as adjudicated.** Sealed
   lengths are protocol constants (8x≡32K by definition — zero information).
   Answer-span positions sit at row tails for all core-4 families (NIAH
   variants and VT place query+answer at the end by generator construction),
   so "100% of answer-span readout queries have prefixes > any training row"
   is TRUE for the answer-span population. Positions are generator-determined
   for training rows too — the identical arithmetic already uses ≤4x rows'
   positions — so blind positions carry no outcome information pre-run (no
   model has been run). Source-depth is uniform-in-haystack ⇒ the blind
   source-distance distribution is broader; that variation is exactly what
   ρ̂_± is priced to absorb (§8.4 discloses it). Auditability note (not an
   attack): S6(ii) should record the generator code+seed hashes and *derive*
   geometry rather than opening sealed row files pre-reveal — same bits,
   better provenance.
6. **Surface 3 (1x anchor) — the "ε̂ undefined" kill-shot fails.** The anchor
   cannot separate the c-curvature term from in-window distractor
   phase/density drift (both are cross-length level effects; ĉ bounds a
   bundle, so "bounds the visible c-coefficient" overstates attribution — and
   the level comparison requires same-generator exchangeable held-out content
   at 1x/2x/4x, available since core-4 exists at 4K/8K/16K; content offsets
   average out in expectation over the row pool, with bootstrap SE to be
   folded in). But for the band's purposes ε̂ only needs to upper-bound
   within-window growth deviation from ln-ratio — which the anchor deviation
   does, conservatively (bundle ⊇ ε). ε̂ is defined and the band stands; the
   mechanism-attribution sentence and an unpropagated anchor SE are
   honesty-level fixes. The slope/level conflation the lead hypothesized is
   real *as an attribution overstatement* but does not un-define ε̂.
7. **Compliance.** Endpoints: every band/veto/promotion rule compares against
   greedy exact, never NLL/margin-as-capability (S6(iii), S4 label line,
   S15) ✓. Single table+gain, gates separate, τ untouched ✓. E2's fresh-QK
   arm adjudicated against the do-not-revive list: it tests Obs3's own
   published falsifier ("param-matched fresh-QK passing 4x on the NEW signal
   ⇒ budget/underfit") on a new objective at 300 steps — a two-sided
   attribution test, not a rank/alpha/gain/step sweep of the failed 96-step
   objective. NOT a revival. S7 stays "Interpretation" and its only consumers
   are E2/E3 crossed predictions (conditional tests) ✓. Trainer-contract
   claims spot-checked against `train_log_p2_phase_transfer_lora.py` (three
   independent 128×2×L pair-view roots, no cross-length identity check —
   consistent with the exchangeable-content reading of the anchor, and
   consistent with S12).

## 4b. What a revision must change (consolidated, minimal)

1. S4: â_hi(S) = E_rows[P̂(Ĝ − Δ_S·ρ̂_− + ε̂ **+ B·o_S**)] and one clause
   naming which λ-direction each endpoint prices.
2. S5/E1: slope test stated in ln n_eff (participation-ratio counts, same
   statistic that defines ρ̂) or the failure label broadened to "P1-window ∧
   constant-ρ (§8.4)" with the veto text updated to match; E1's
   "opposite predictions" sentence re-qualified.
3. S4/S6(iii): propagate (α̂,ĝ_c) bootstrap CI into endpoints; state that
   â_lo(32x) ≈ 0 (upper-side-only falsification power at 32x).
4. S2(2): replace the BOS-key carrier sentence with the shared-value-constant
   route (value path; RMSNorm-monotonicity one-liner).
5. S1: scope clause "…and performs no computation at >16K physical length,
   including on protocol-generated rows"; S3 "pair mass" → "pair count".
6. S6(ii): record generator hash + seed as the geometry read, not row files.

None of these touches the architecture, the impossibility result, the
demonstration leg, or any owned number.
