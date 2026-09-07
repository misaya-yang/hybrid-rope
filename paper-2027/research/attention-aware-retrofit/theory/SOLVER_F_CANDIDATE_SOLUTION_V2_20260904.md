# Solver F V2: identifiability-delimited LoRA conversion — re-derived under the A1/A2 veto

> **External-model archive:** not project evidence, instruction, authorization,
> or a current verdict. Re-establish every claim from canonical owners.

- **Date:** 2026-09-04 (round 2; supersedes `SOLVER_F_CANDIDATE_SOLUTION_20260904.md` for solver-F use)
- **Status:** solver candidate memo; **not a claim owner**; no new measurement;
  no run authorized. All numbers are repo-owned Observations or arithmetic
  derived from them (derivations inline). Owner citations corrected per attack
  items A3/A9.
- **Question (F):** unchanged from round 1.
- **Path:** still **(C) hybrid**, but re-derived. FATAL A1 accepted: the round-1
  claim "A1 assumes nothing beyond testable dilution" is withdrawn. Round-1 A1
  is replaced by a four-clause premise set P1–P4 with **disclosed untestable
  members** and a predictor â that is a **licensed prediction set (band)**, not
  a point. FATAL A2 accepted: the â-veto is re-scoped to disclosed
  compute-avoidance only; unconditional post-reveal demonstration for *measured*
  lengths is admitted; S13 taxonomy re-derived accordingly.

## 0. Attack-disposition table (binding items → disposition)

| Item | Disposition in this memo |
| --- | --- |
| **A1 (FATAL)** | **Re-derived, not patched.** â rebuilt as two-sided band (S4) over named premise set P1–P4 (S3), with numerator beyond-window drift entering as explicit class-bounded parameter λ; transport premise P3 named with zero visible consequences; blind-prediction dependence on untestable clauses **quantified** (S3, P3-row): 100% of blind readout queries and 25.0%/56.3%/76.6% of key–query pair mass at 8x/16x/32x sit beyond trained distances. "Nothing else is assumed" and "entire visible consequence set testable at ≤4x" **withdrawn**; §4 minimality restated honestly (premise set, not minimal testable assumption). |
| **A2 (FATAL)** | **Conceded + re-derived.** Rule (i) veto survives ONLY as compute-avoidance; the phrases "dilution reserve measurably insufficient", "impossibility-shaped outcome", "no protocol could have done better" are **deleted** and replaced (S6); false-negative capacity of the veto is named. S1 trichotomy corrected to admit **unconditional demonstration at measured lengths** (S1, S13). "Under P1–P4" binds the â-machinery, between-length mechanism claims, and unrevealed lengths — never a measured 8x/16x greedy number (S13). S13 taxonomy re-derived per band semantics. |
| **A3 (moderate)** | **Fixed.** The 200-empty-predictions and 61%-EOS facts are attributed to **Obs2 (OLMo)**, not Obs1; S8's motivating sentence rewritten with per-owner citations (Obs1 owns exact-0/rank-2043 only). |
| **A4 (moderate)** | **Split.** Single-owner core kept: Obs3's paired +0.014107 [CI] 1x PG-19 improvement (one owner). The 0.8877 level relabeled **derived-composite** (SCALE/Handoff §3.1 frozen retention × Obs3 paired delta; legal only if gate row-set identity is verified — verification made a protocol step). The five-task 0.8982 blended retention **dropped**; replaced by Obs3's within-owner −0.00582 [CI crosses 0] and a rule that re-runs the gate rather than inheriting levels. The 0.03 budget **labeled as a registered assumption** with its falsifier. |
| **A5 (moderate)** | **Fixed.** Frozen substrate passes its own gate (0.875302, owner); "substrate failed the gate" attribution deleted. All-checkpoints-fail is now a **protocol-caused** negative (adapter PG-regression), with the substrate verdict left untouched. Gate split: (a′) pass-retention (gate) vs (a) beats-frozen (headroom-monitor, promotion-preferred). |
| **A6 (moderate)** | **Fixed honestly.** "+1 forward, greedy-visited prefix" corrected to the two-pass **sequence-level student exposure** scheme (no-grad greedy decode of answer span; gradient pass over the substituted sequence; ≈+50% wall). What it fixes: **content** exposure bias on the answer span. What it does NOT fix: **positional-context exposure** of a query inside a 32K row — that half is named as premise P3, not solved. |
| **A7 (moderate)** | **Re-stated + lemma added.** S1's gate moved to **base weights** (environments differ in frozen weights; the adapter is whatever the protocol selects), bias-free class (OLMo: no additive projection biases), threshold carried by a learnable constant key/value direction; **Lemma F1′** makes the exponential-flatness argument explicit and proves **value AND gradient** gaps both vanish at ≤4x (GeLU tails: |GELU(−g)|, |GELU′(−g)| ≤ (g+1)e^{−g²/2}; verified example: g=60 → ~10⁻⁷⁸⁰; exact separation on the finite visible sample by order-statistic gap). |
| **A8 (moderate)** | **Redesigned.** Spearman-on-4-bands deleted. T1a is now a pooled within-row LSE-vs-ln n regression with SE across 12–16K rows, slope ∈ 1±0.15 on the CI, plus a cross-length 1x-anchor that bounds the visible c-term coefficient; T1's claim **re-scoped**: it falsifies only within-window violations of dilution clause P1, with declared power limit (drift confined to Δ>16K is unfalsifiable at ≤4x — Lemma F1′ says no test can). |
| **A9 (minor)** | **Fixed.** Obs2 carrier contrast now column-labeled: QA exact 21.5%/17.5%/4.0% (EVQ) vs 22.0%/**0%/0%** (Native); RULER-13 macro 42.44/31.63/5.03 (EVQ) vs 72.19/**2.02/0.38** (Native) — the "0%" is the QA-exact column, not the RULER macro. Δ_S corrected: nominal token counts of sealed blind rows are geometry and legal; **effective** competitor counts are attention quantities and unmeasurable pre-reveal; the nominal→effective gap enters the band through ρ̂, measured at ≤4x, never as "measured blind ratios". |

Arithmetic verified this round: 1−(2D/n−(D/n)²) at D=16K gives 0.2500/0.5625/0.7656;
0.875302·e^{0.014107}=0.887737 = exp(−(3.090126−2.971047)) to 7 decimals;
ln2/ln4/ln8 = 0.693/1.386/2.079; −ln0.88=0.127833, −ln0.875=0.133531 nats.

## 1. The solution in 15 numbered statements

**S1 — Impossibility theorem, corrected trichotomy (proven half, restated headline).**
Fix any protocol that trains and selects only on ≤4x-visible statistics, one
static table/gain, LoRA of any fixed rank on {q,k,v,o}. Then: (i) no protocol
can *guarantee* 8x+ behavior **prospectively** — Lemma F1′ below; (ii) a
protocol that *reveals* blind rows can **demonstrate** useful 8x/16x/32x
capability unconditionally for the measured lengths (a greedy-exact endpoint
measured under a validated firewall needs no assumption; round 1's S1 headline
erased this and is corrected); (iii) pre-reveal, the only sound epistemic
object is a prediction **conditional on a disclosed premise set** (S3–S4).
Guarantee / demonstration / conditional prediction are three different things;
round-1 memo conflated (i)'s content with (iii)'s. Label: derived result +
correction.

**S2 — Lemma F1′ (gradient-indistinguishable length gate; bias-free class; value AND gradient gaps vanish).**
*Claim.* In a bias-free transformer class (OLMo-style: no additive
projection biases; GeLU; RMSNorm) there exist environments E, E′ differing only
in **base** weights, and a task law, such that for any protocol as in S1: on
every row of length ≤16K and every prefix thereof, per-token loss values and
loss gradients with respect to *all* parameters (base and LoRA) differ between
E and E′ by ≤ (1+g)·e^{−g²/2}·C, for a separation g the constructor chooses
arbitrarily large; while 8x greedy answer accuracy differs by ≥ 0.5.
*Proof sketch (4 steps).*
(1) **n-feature without biases.** A head with near-uniform attention and values
v_j = c + ξ_j (ξ iid, zero-mean, E‖ξ‖²=σ²) has output h_n = c + (1/n)Σξ_j.
Any downstream GeLU unit at operating point c converts the deviation's second
moment into a *mean* shift: E[GeLU(wᵀh_n)] = wᵀGeLU(c) + ½·GeLU″-like curvature
term · σ²_w/n + O(n⁻²) — a monotone 1/n feature (and analogously √variance ~
n^{−1/2}), realized with no additive bias; RMSNorm preserves it.
(2) **Threshold without a bias parameter.** Scale the unit's weight by gain W
so its pre-activation is W·(c₀ − k/n) with c₀ placed strictly between the
visible range [k/16K, …] and the blind range [0, k/32K]; c₀ is carried as a
learnable constant direction in the residual stream (a value-aligned
BOS-type key contributes a fixed vector at every position), so the offset is a
*weight-space* constant, not a bias tensor.
(3) **Flatness (this is the lemma's work).** For n ≤ 16K the pre-activation is
≤ −g with g = W·(k/16K − c₀)·… > 0; GeLU satisfies |GeLU(−t)| ≤ (t+1)e^{−t²/2}
and |GeLU′(−t)| ≤ e^{−t²/2}(1+t) (Gaussian-tail bound), so the logit
contribution and its **derivative w.r.t. every upstream parameter** are ≤
poly(g)e^{−g²/2}. The adapter's gradient path through the gate is multiplied
by the same factor, hence *both value and gradient* trajectories of any
gradient-based protocol agree to ≤ e^{−Ω(g²)} at each of its finitely many
steps, and by choosing g ≥ √(2·ln(C·T/δ)) the entire 300-step trajectory —
hence the selected adapter — is identical at any fixed selection precision δ.
On the finite realized visible sample the strict order-statistic gap between
max_{≤16K}(1/n) and min_{≥32K}(1/n) of the feature makes the contribution
*numerically exact-zero* (verified: g=60 gives log₁₀ ≈ −780).
(4) **Blind flip.** At n ≥ 32K the pre-activation is ≥ +g′ and the unit
contributes O(β) to the answer-logit difference; β is set > the task's
margin-to-argmax, flipping exact correctness on ~all blind rows. The 18K–32K
transition region is visited only inside blind rows, so it never touches
≤4x-visible states. □
*Consequences.* (a) No ≤4x statistic — loss, NLL, margin, band curve, rank,
attention, or any *gradient* of any loss — identifies 8x behavior. (b) The
gate environments specifically violate the transport clause P3 below: this is
why P3 must be *named*. Label: derived result (ε-version at distribution
level; δ-version on realized samples; construction, not a claim about any repo
model).

**S3 — The premise set, with an explicit visible-consequence column (round-1 A1 replaced).**
The transfer claim is conditional on all of:
- **P1 (dilution extension).** LSE_D growth between competitor counts n₁<n₂ is
  ln(n₂/n₁) + ε, |ε| ≤ ε̂. *Visible consequences:* within-window slope at
  1x/2x/4x band boundaries (T1a) is a **necessary** consequence; the extension
  to n₂ > 16K has **none** (S2). ε̂ is measured in-window and *assumed carried*
  — disclosed.
- **P2 (numerator drift class).** |∂z_R/∂ octave| ≤ B along the source
  distance axis, B measured in-window from source-band margin decay (T1a
  anchor + E1), then **extended** to blind octaves: blind numerator shift ≤
  λ·o_S with λ ∈ [0, B] an explicit unmeasured parameter and o_S = log₂(S/16K)
  = 1/2/3 octaves at 8x/16x/32x. *Visible consequences:* within-window
  boundedness of β̂ (measurable); beyond-window behavior **none** — disclosed.
  Round 1's A1b ("flat within measured bounds") was this clause dressed as a
  test; withdrawn.
- **P3 (population transport).** The fitted margin→accuracy map P̂ and the
  within-window margin distribution apply to the blind query population.
  *Visible consequences:* **none** — the blind population is the blind rows
  (S2's gate violates exactly this). **Dependence quantification:** in 8x/16x/32x
  rows, 100% of answer-span readout queries have prefixes longer than any
  training row, and 25.0%/56.3%/76.6% of all causal key–query pairs sit at
  relative distances >16K never present in any training row (uniform-position
  arithmetic; attacker-1's 46.7/74.2/87.3% differ only in which query set is
  averaged). Therefore **most — in the query-population sense, all — of the
  blind prediction mass rides on P3, the clause with zero in-window
  consequences.** This is conceded, not concealed.
- **P4 (no adversarial beyond-window response).** S2's gate-shaped response
  functions are excluded from the environment class; equivalently, the
  protocol's conditional claims are quantified over P4-respecting environments
  only. *Visible consequences:* none; P4 is the honest name for "the model is
  not pathological beyond support".
*Minimality, restated honestly:* this is a four-clause premise set with two
fully untestable members (P3, P4) and two partially untestable (P1, P2
extensions); it is not "the weakest testable assumption" (round-1 §4 claim
withdrawn — no such element exists in the smooth-function hierarchy; S2 proves
every candidate has an invisible-violation class). It is the *smallest named
set* whose testable projections (P1-window, P2-window) are what T1/E1
actually check, and whose untestable projections are what the band must
honestly widen.

**S4 — â rebuilt: licensed prediction set (two-sided, derived from P1–P4).**
Let Ĝ be the inference-time same-target counterfactual margin on 1x/2x/4x
held-out rows, P̂(·)=σ(α̂(·−ĝ_c)) the monotone fit of greedy-exact correctness
on those same rows, Δ_S the **nominal** competitor-count log-shift from
geometry (8x: ln2, 16x: ln4, 32x: ln8), ρ̂ the ≤4x-measured median effective/
nominal competitor ratio with its dispersion, ε̂ the T1a within-window slope
deviation. Then, under P1–P4:

  â_lo(S) = E_rows[ P̂( Ĝ − Δ_S·ρ̂_+ − ε̂ − B·o_S ) ]
  â_hi(S) = E_rows[ P̂( Ĝ − Δ_S·ρ̂_− + ε̂ ) ]
  predicted accuracy at length S ∈ [â_lo(S), â_hi(S)]

Band width is **licensed by what can and cannot be named**: Δ·ρ̂ term covers
A9's nominality, ε̂ covers the testable dilution slack, B·o_S covers the
numerator family λ, and the transport clause P3 is **not** widened further —
disclosed limitation: if P3 itself is false (S2 class) the band is void
*whatever its width*, which is why S6's post-reveal rules treat the band as a
falsification target, not a confidence object. Best case (λ=0, ε=0) recovers
round 1's â; round 1 presented the best case as *the* prediction — that was
the FATAL. Label: derived (band) conditional on P1–P4; the construction
(â_lo/â_hi estimators) is ≤4x-computable and firewall-legal; the band's
*endpoints* are compared only against greedy exact endpoints post-reveal.

**S5 — T1, re-scoped and re-powered (falsifies the dilution clause only).**
On held-out ≤4x rows (no training rows reused for the fit): (T1a) pooled
within-row regression of band-cumulative distractor LSE on ln n over
n∈{4K,8K,12K,16K} × rows (12–16K panel), per adapter and frozen: require
slope ∈ [1−0.15, 1+0.15] on the CI (SE from row-level bootstrap; Spearman
deleted per A8 — with 4 bands it only detects perfect monotonicity); plus a
cross-length **1x anchor** (a c·(log₂S−1)(log₂S−2)-family term is zero at
2x/4x but 2c at 1x, so the 1x band curve bounds the visible c-coefficient ĉ;
ε̂ := max(T1a residual, ĉ-driven 1x anchor deviation)). (T1b) adapter−frozen
per-band mean logit shift on source-band and distractor-band keys, |shift| ≤
0.25 nats across the four ≤16K bands (detects learned band-specific policy
*in-window* only). **Scope of T1:** necessary-condition check on P1-window and
P2-window and adapter policy; a failure falsifies P1/P2 (the run becomes an
S6(iii)-style premise falsification *without* a reveal); a pass licenses
nothing — it is an anti-falsification check, and saying more re-invites A1.
Power limit stated: drift confined to Δ>16K is undetectable, and by Lemma F1′
no ≤4x statistic can detect it.

**S6 — Decision rules under band semantics (A2-conformant epistemics).**
- **(i) Reveal rule (compute-avoidance only).** Open the single blind reveal
  iff: both retention gates pass (S9), T1 passes, and â_hi(8x) ≥ τ=0.40.
  The veto's status: **a disclosed bet against paying for a reveal whose
  optimistic premise-case prediction is below usefulness.** A no-veto protocol
  could have demonstrated success unconditionally had capability existed
  despite P1–P4 (false-negative capacity named). The veto is never reported
  as "dilution reserve measurably insufficient", never as an
  "impossibility-shaped outcome", never as "no protocol could have done
  better" (all three strings deleted from S13).
- **(ii) Pre-registration.** (â_lo, â_hi) triplets for 8x/16x/32x, B̂, ε̂, ρ̂,
  P̂ parameters, τ, and this rule are written to a hash-bound decision file
  before the reveal; only row geometry (lengths, answer-span positions) of
  sealed blind rows is read earlier — counts, not outcomes (A9).
- **(iii) Post-reveal reading (single reveal, all three lengths together).**
  measured a(S) ∈ band ⇒ premise set **survives falsification** at S (never
  "confirmed"); measured a(S) ≥ τ ⇒ **unconditional demonstration** of useful
  capability at S (no P1–P4 conditional attaches to this number; S2's theorem
  bars *guarantee*, not *measurement*); measured a(S) ≪ â_lo(S) ⇒ P1–P4
  falsified jointly with a direction worth reporting (distractor nonstationarity
  beyond 16K, numerator drift > B, or transport failure) — a real, owned
  negative that no ≤4x test could have caught; measured â_lo < a(S) < â_hi but
  a(32x) ≥ τ with o=3 ⇒ band machinery earns its keep only if the *ordering*
  (8x ≥ 16x ≥ 32x) holds; if measured > â_hi at some S, the optimistic clause
  (P1-ε̂ sign or ρ̂) is falsified upward and no promotion is claimed beyond
  lengths already covered by published evidence.

**S7 — C1/C2/C3 synthesis of the three observations (unchanged in substance; citations corrected).**
Conversion at ≥2x needs jointly: (C1) substrate carrier; (C2) trainable-or-
inherited transport/readout capacity (V/O); (C3) a source-forcing signal at
the prefixes decoding visits. Obs1 lacked C3 (generic CE; links 1–2 moved,
link-3 flat: exact 0%, rank 2043); Obs2 lacked *dense* ≤4x exposure (its
long exposure was position-sparse, physical ≤4K with explicit position IDs —
per its own intervention section) and its 16K (=4x) weakness is exactly the
λ-band of P2 untested by any dense row; Obs3 lacked C2 (fresh Q/K rank-8).
The prepared prototype is the first arm with C1+C2+C3(teacher-placed); this
memo adds the P-set/band accounting and the S8 placement fix. Residual
tension preserved and disclosed: Obs2 converted with only Q/K trainable —
read as C2-present-by-parenthood (Stage-A V/O inherited trained); E2 tests it.
Labels: Interpretation (falsifiable by E2/E3 crossed predictions).

**S8 — Link-3 placement fix, correctly quoted and correctly costed (A3/A6).**
The same-target margin term is retained but *not* credited with link 3:
Obs1 shows source dependence and transport can both be present at the teacher
state with free-run exact still 0 (rank 2043). The nonterminal failure
evidence is OLMo-owned: Native arm 8K EOS 61.0% and 16K 200/200 empty
predictions (OLMO2 owner; round-1 misattribution to Obs1 corrected). The
repair is **sequence-level student exposure** on the answer span: (1)
no-grad greedy decode of the answer+EOS span under the current adapter;
(2) one gradient pass of CE + counterfactual margin (margin now including the
terminal pre-EOS→EOS contrast; round-1 `source_effect[:, :-1]` exclusion
fixed) on the *substituted* sequence. This is teacher-forcing on the visited
sequence (scheduled-sampling one-epoch), NOT per-token autoregressive
DAgger: cost ≈ one no-grad prefill + L decode steps + one grad pass ≈ +50%
wall/step at answer spans ≤ ~16 tokens; budget stays 300 steps. What it fixes:
content exposure bias on the answer span. What it does **not** fix: the
positional-context half (a query inside a real 32K row sits in an attention
context never present at training) — that half **is P3**, named, not solved;
no claim on this memo's side may present D2 as closing P3. Falsifier: D2
passes 4x while prototype fails 4x ⇒ content-exposure component established;
D2 ≡ prototype in-window ⇒ D2 mechanism refuted, blind gap then purely P1/P2/
P3 territory.

**S9 — Retention, single-owner core and re-attributed failures (A4/A5).**
Owner-internal fact (Obs3, one owner, one table): its c=.074 adapter improved
paired 1x PG-19 by +0.014107 [+0.009616, +0.019405] and 4x by +0.018742, on
the exact retained float32 table hash. Composite (needs verification step,
labeled derived-composite, not Observation): frozen 1x retention 0.875302
(Handoff §3.1 / SCALE owner) × e^{0.014107} = 0.887737, consistent with
direct exp(−(3.090126−2.971047)) = 0.887738 because the two owners record the
same frozen NLL 3.104234 and table hash — row-set identity of the Native arm
is nevertheless a **protocol step to verify**, after which the level claim
becomes owner-grade. Dropped per A4: the blended five-task retention 0.8982
(cross-owner) — instead, the five-task gate must be evaluated on the
protocol's own Native-vs-adapter rows at ≤4x (cheap; already in Stage F
order); Obs3's within-owner −0.00582 interval is cited only as evidence that
*rank-8* adapters on this substrate showed no significant 1x five-task cost.
Rules: (a′) HARD gate: adapted 1x PG-19 retention ≥ 0.875 AND adapted 1x
five-task retention ≥ 0.875, separate, evaluated on checkpoints {250,275,300}
with strict-0.88 status reported; (a) HEADROOM MONITOR (promotion-preferred,
not a gate): paired 1x PG-19 improvement over frozen — registered assumption
that the Obs3 effect transplants to r64 QKVO; its falsifier is any accepted
checkpoint with paired-1x regression, and the 0.03 five-task drop budget for
QKVO is likewise a **registered assumption from one rank-8 observation**, not
measured at rank 64. Failure attribution: frozen passes the gate per owner;
so no checkpoint passing (a′) is a **protocol negative** (objective/schedule/
placement drove a PG-regression) and closes this protocol without a substrate
verdict; a frozen-gate failure would be a substrate negative (did not occur).

**S10 — Substrate and carrier premise (unchanged substance; column labels per A9).**
Run on s4/c=.074 (or Stage-Z-selected factor, with carrier re-measured frozen
at ≤4x before any GPU stage F cost). Carrier is measured: frozen RULER-13
0.71397/0.66705/0.49859 at 4K/8K/16K vs Native 0.71308/0/0.00385 (SCALE
owner). Obs2's substrate contrast, column-labeled: under identical
continuations, EVQ vs Native — 2Wiki exact 21.5%/17.5%/4.0% vs
22.0%/0%/0% (QA table); RULER-13 macro 42.44/31.63/5.03 vs 72.19/2.02/0.38.
The carrier is substrate-borne; that is the premise P1–P4 ride on, and the
matched Native-table arm of E3 re-tests it under the new objective.

**S11 — Placement arms with parameter-count receipts (unchanged).**
Frozen / fresh-QK / prototype-QKVO(teacher) / D2-QKVO, data, order, tokens,
optimizer, 300 steps identical; actual trainable counts matched within 1%
(handoff §5's GQA rank≠capacity warning binding). QK runs first per the
preflight ordering; E2 runs it against its crossed predictions (S7).

**S12 — Data contract (with the geometry carve-out made explicit).**
Training rows exclusively physical 4K/8K/16K identifiable pair views + 1x
replay; 8x/16x/32x rows generated pre-training and sealed: **token counts and
answer-span positions (row geometry) may be read for Δ_S·ρ̂ band arithmetic;
no model output, no margin, no score, no correctness on sealed rows enters
any statistic.** All ≤4x fits (P̂, ε̂, B̂, ρ̂, gates, checkpoint choice) on
1x/2x/4x held-out rows only; one reveal, all three lengths; S13 conditions
frozen in the decision file.

**S13 — What each outcome licenses (re-derived taxonomy).**
- Reveal + gates + T1 pass + measured a(8x),a(16x) ≥ τ: **unconditional
  demonstration** of useful blind generated capability at the measured lengths
  for this checkpoint/family/protocol (firewall-clean because of S12);
  "under P1–P4" attaches only to (α) the band machinery's between-length
  mechanism language (the −ln2 per octave shift), (β) any length not measured
  (e.g. 32x when floored or unopened), (γ) the veto decision.
- Reveal + measured ∈ bands: premise set survives falsification; demonstration
  at measured lengths as above if ≥ τ.
- Reveal + measured ≪ â_lo: P1–P4 falsified; publishable negative with
  direction (S6 iii); no rescue sweeps (revival list respected).
- Veto taken (â_hi(8x) < τ): compute declined under optimistic premise case;
  **reported as a decision, not an evidence claim**; an unvetoed protocol
  retains the possibility of unconditional demonstration that this one bought
  out. If GPU is cheap, the protocol's recommended default is to reveal anyway
  and let the falsification test run — the veto exists to protect a scarce
  single-reveal seal, nothing more.
- Gate failure: protocol negative (S9 attribution).

**S14 — Why not pure impossibility (the option the adjudication allowed).**
Pure impossibility would require proving both horns of A1 close *every*
≤4x-computable predictor. That is false as a claim about epistemic products:
S2 bars *guarantees* and *pre-reveal identification*, but the reveal itself
is ≤4x-computable-and-then-measured and yields unconditional demonstration
(S1-ii), and bands are falsifiable premise-set tests (S6-iii). What *is*
proved: the veto and any pre-reveal pass/fail claim can never be evidence
about the blind lengths, so the only defensible positive object is a
measured one. The hybrid survives with these corrections; a pure-impossibility
memo would have thrown away the demonstration leg and is therefore not
delivered.

**S15 — Compliance.** No rank/alpha/gain/LR/step sweep on the failed 96-step
Q/K objective; no routing/dual-table substitutes; no proxy promoted as
capability (margins/LSE/NLL are gate inputs, band inputs, or premise
falsifiers only; the endpoint compared to predictions is greedy exact); no
non-admissible moves from the attack are used — P2's λ is a premise with a
measured bound, not "A1b's parenthetical renamed", τ is unmoved, veto status
is downgraded exactly as ordered, and no new mechanism story is told about
numerator stability without a premise (P2 says "class-bounded, unmeasured
beyond window", which is the disclosure, not a story).

## 2. Observations the solution must explain; consistency

All three (same coverage as round 1, unaffected by the veto — it targeted the
predictor's premises and epistemics, not the attributions): Obs1 (links 1–2
moved, link-3 flat: C3-absence at decode states + in-window EVQ damage);
Obs2 (conversion ≤2x, 4x weakness = untested λ-band of P2 under position-
sparse exposure; carrier substrate-borne); Obs3 (capacity C2-absent +
teacher-placement; also supplies S9's owner-internal headroom fact and
created the composite-retention arithmetic that is now verification-gated).
Consistent with all three. Open tension (disclosed, tested by E2): Obs2's
success with trainable-Q/K-only, read as C2-by-parenthood.

## 3. Link-level attribution (owners; falsifiers) — unchanged except citations

| Failure | Broken link | Evidence (owner) | Falsifier |
| --- | --- | --- | --- |
| Obs1 (hit@16 18.75→64.06%; gold-block deletion +1.5055 vs −0.0095; 16K exact 0%; first-token rank 2043) | 3 (top-1 at decode states) | `EVQ_8B_ADAPTATION_EVIDENCE_20260724` Levels 2–3 | free-run rank ≈1 with scorer-side failure (owner records greedy exact 0%); or D2-invariance (S8 falsifier) |
| Obs2 (QA F1 24.84→21.48→8.57; RULER 42.44→31.63→5.03; physical ≤4K + explicit position IDs to 16K) | 4 (dilution reserve, dense regime untrained) + substrate-bound ceiling | `OLMO2_1B_SELECTIVE_QK_PHASE_ADAPTATION_20260729` | source-band numerator collapse inside [4K,16K] with slope-1 distractor LSE ⇒ link-1 phase, not dilution (T1a on that substrate) |
| Obs3 (PG-19 +0.0141/+0.0187; five-task ≤0 CIs cross 0; core-4 −.0100/+.0025/−.0225 at 1x/2x/4x) | 2 + 3-capacity/placement; **in-window**, no link-4 claim admissible | `LOG_P2_QK_LORA_GAIN_MATCHED_RESULT_20260904` §1,§4,§5 | param-matched fresh-QK passes 4x on the new signal ⇒ budget/underfit, refuting C2; or QKVO-teacher also fails 4x ⇒ move cause from C2 to C3 |

## 4. Premise-set analysis (replaces round-1 "weakest testable assumption")

- **What is testable:** P1-window and P2-window projections (T1a incl. 1x
  anchor, E1 slope), adapter band policy (T1b), gates (≤4x).
- **What is not testable at ≤4x, and was previously disguised as testable:**
  P1/P2 extension beyond 16K; P3 in full; P4. Lemma F1′ is the proof that no
  ≤4x statistic (value or gradient) can test them.
- **Dependence:** all blind readout queries ride on P3; between-length band
  shape rides on P1/P2-extensions; the *measurement* rides on nothing.
- **Minimal relaxation names (unchanged, relabeled):** R1 = any structural
  beyond-window restriction (P1–P4 is such a package, now honestly priced);
  R2 = one 8x dev row in selection (breaks the class; named, rejected);
  R3 = drop a retention gate (Obs1 shows the ungated conversion-to-zero path
  is cheap; the gate, not the budget, is what hurts). The delivered answer
  keeps all constraints and pays for it with disclosed untestability.

## 5. Protocol deltas over the prepared prototype (final list)

1. D2 sequence-level student exposure + EOS-transition margin (S8, correct
   cost).
2. Inference-margin + band-LSE module on 1x/2x/4x held-out rows; fit P̂, ε̂,
   B̂, ρ̂; write decision file (bands, rule S6) + hash; geometry-only readout
   of sealed rows.
3. T1 module (S5 scope and power disclosure).
4. Checkpoint acceptance {250,275,300} on (a′) separate 0.875 gates (own
   rows; verify Native row-set identity before quoting composite levels),
   (a) headroom monitor reported not gated.
5. Arms frozen/QK/QKVO/D2-QKVO param-matched + Native-table control on pass
   (E3).
6. Budget 300 steps unchanged; ≈+50% wall accepted as the honest D2 cost.
7. Recommended default when blind reveal is cheap: skip veto, reveal, and let
   the falsification test decide (S13).

## 6. Discriminating experiments (≤3, predictions crossed)

**E1 — In-window dilution/numerator assay (no GPU training).** Frozen +
candidate adapters on 4x held-out rows with competitor truncation
n′∈{4K,8K,12K,16K}, queries ≥4K from the cut edge: H-dilution (P1-window)
⇒ slope of Ĝ vs ln n′ ≈ −1 with CI, accuracy tracks P̂; H-phase (link-1 decay)
⇒ source-band collapse, slope ≠ −1. Outputs ε̂, B̂ that *define* the band; the
two hypotheses predict opposite T1a results — no outcome is compatible with
"nothing was learned".
**E2 — Placement arms (fresh-QK vs prototype-QKVO vs D2-QKVO, param-matched,
same 300 steps).** C2-capacity: QK fails 4x, QKVO passes. Budget-underfit
(re-vives Obs3 attribution falsifier, which is allowed — it is a test, not a
sweep): QK passes 4x. Exposure-bias: QKVO(teacher) passes 4x but D2's blind
gap is the exposure-bias estimate (and by S13 a D2-only demonstration is
unconditional for its measured lengths). E2/E3 arms' blind gap is *pure P1/
P2/P3 content*: if D2 and prototype both pass 4x and differ only in blind
outcome, the between-length mechanism claim is what is being tested.
**E3 — Carrier ablation (Native-table vs s4, identical D2-QKVO contract).**
Substrate-borne carrier (Obs2-predicted): Native fails 8x/16x. LoRA-created
carrier (rival): Native matches s4. Opposite predictions, same budget, and
the preflight-mandated attribution control for any pass.

## 7. Honest status per main statement

| Statement | Status |
| --- | --- |
| S1 trichotomy | **Proven** (impossibility half); demonstration/correction clauses are protocol-epistemic statements |
| S2 Lemma F1′ | **Derived result** (explicit construction; ε-version disclosed; bias-free feasibility argued in GeLU/RMSNorm class, not run) |
| S3 premise set | **Conditional by declaration**; untestable members named and priced (the correction A1 demanded) |
| S4 band predictor | **Derived given P1–P4**; estimators ≤4x-computable; P3-dependence conceded quantitatively |
| S5 T1 | **Design**, re-scoped to dilution-clause falsification with power limits |
| S6 rules | **Pre-registered protocol rules**; veto status = compute-avoidance (not evidence) |
| S7 synthesis | **Interpretation**, falsifiable via E2/E3 |
| S8 D2 placement | **Working hypothesis**; content-exposure half addressed, positional half = P3 (disclosed) |
| S9 owner-internal facts | **Observation** (Obs3 table) + **derived-composite** (0.8877, verification-gated) + **registered assumptions** (transplant, 0.03) |
| S10 carrier | **Observation** (two owners, column-labeled) |
| S11–S12 | Protocol rules |
| S13 taxonomy | **Pre-registration** (A2-conformant) |
| S14 why not pure impossibility | **Derived argument** (demonstration leg intact) |
| S15 compliance | Statement of record |

## 8. Remaining attack surface (self-declared)

1. Lemma F1′ is a construction over a model class; a reviewer may demand an
   instantiated 2-layer numeric example — the exponential-flatness bound
   ((g+1)e^{−g²/2}; g=60 ⇒ ~10⁻⁷⁸⁰ verified arithmetically) makes the
   qualitative claim robust, but a fully written explicit-weight example would
   strengthen it further.
2. τ=0.40 remains author-anchored (s4 frozen core-4 16K=0.4025, a ≤4x number);
   the band rule is disclosed and pre-registered, but τ itself is operational.
3. B̂ is fitted from 3 anchor lengths + 4 within-row bands; a numerator curve
   in the Lipschitz class with in-window slope near zero and out-of-window
   slope = B is the band's honest worst case — the *width* of the 32x band is
   therefore expected to be large and E1 should report it as such (a wide 32x
   band is not a defect; pretending it is narrow was the round-1 sin).
4. ρ̂ (effective/nominal competitor ratio) is measured only for in-window
   attention patterns; blind heads could sparsify differently — folded into
   the Δ_S·ρ̂_± term, but that term is a modeling choice, not a theorem.
5. The E1 truncation assay changes which queries exist near the cut; edge-
   excluded powering (20+20/cell) must pass a no-GPU dry count before any
   adapter spend.
