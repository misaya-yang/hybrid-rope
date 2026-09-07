# Attack 1 on SOLVER_F_CANDIDATE_SOLUTION_20260904 (adversarial review)

> **External-model archive:** not project evidence, instruction, authorization,
> or a current verdict. Re-establish every claim from canonical owners.

- **Date:** 2026-09-04
- **Role:** veto-seeking reviewer (Solver F targeted)
- **Target:** `SOLVER_F_CANDIDATE_SOLUTION_20260904.md` S1–S15, §2–§8
- **Verification actually run:** full read of the memo, handoff, preflight, all three
  observation owners, and `scripts/train/train_log_p2_phase_transfer_lora.py`;
  Python recomputation of every cited or derived number (§5 below).

## 1. VERDICT: **VETO**

Two FATALs stand (A1, A2), each striking a core statement (A1's claimed minimality
and the derivation of S5; S1's claimed shape-of-answer and its use as a permanent
reveal-veto in S5 rule (i)/S13(i)). Six further attacks are MODERATE/MINOR and do
not independently reach the veto bar. Arithmetic is clean — I found no misderived
number; the memo's owner-table citations reproduce exactly, and I say so plainly.

---

## 2. Attacks, ranked by severity

### A1 — FATAL — the S5 reweighting formula is not licensed by A1 as stated; A1's "nothing else is assumed" is false, and its "strictly weaker than any fitted continuation law" is false in the opposite direction

**Target statements:** S3 (identity + A1a + A1b, "Nothing else is assumed"),
S5 ("under A1, the blind prediction is pure reweighting
â(S) = E_rows[P̂(Ĝ − Δ_S − ε̂)]"), §4 minimality argument, S2's role-claim.

**Break mechanism.** â(S) shifts the *entire* visible margin distribution by the
*LSE_D-growth term only*. Unpacked through the memo's own identity
Ĝ(S) = z_R(S) − LSE_D(S), the formula asserts:

1. **Numerator invariance beyond the window:** z_R(8x on visible-analogue
   queries) = z_R(4x). But A1a quantifies *only* LSE_D growth ("LSE_D(n₂) −
   LSE_D(n₁) = ln(n₂/n₁) + ε"), and A1a's ε̂ is defined as deviation of the
   **band-LSE (distractor) curve** — the numerator never appears in either
   A1a or T1a. A1b bounds the *adapter's* per-band shift within [4K,16K]
   (≤0.25 nats, T1b) and the frozen numerator's profile "flat **within
   measured within-window bounds**" — i.e., within-window only. Nothing in
   A1-as-written constrains z_R at distances >16K.
2. **Population invariance:** the expectation E_rows is over the *visible*
   query population, carried "unchanged" to 8x/16x/32x rows.

On (2), I computed the distance-population shift concretely. With RULER-style
placement (source uniform in the haystack, query near the row end), the
fraction of blind-length (query, source) pairs whose physical distance exceeds
**every distance that exists in any training row** (max 16K in a 16K row):
**46.7% at 8x, 74.2% at 16x, 87.3% at 32x** (H = S−2000; verified in Python).
So even if per-query numerators were invariant, nearly half of 8x is composed
of query types that do not exist in the calibration population; the memo's
S3 line "the margin distribution … estimated in-domain and carried across
unchanged" is doing exactly the work it claims not to be assuming — and that
sentence, read literally, *names the assumption* immediately after "Nothing
else is assumed."

On (1), the memo's own S2 is the witness: ℓ(Δ) = Σγ_k[a_k cos(ω′_kΔ)+b_k
sin(ω′_kΔ)] is **oscillatory in Δ** and S2 itself states the composed numerator
is "not finite-basis in Δ" and that the 128-span argument "does not rescue the
general problem." A frozen numerator that decays or re-phases between 16K and
32K is compatible with every named clause of A1, passes T1a (all bands ≤16K),
passes T1b (bands ≤16K), matches all ≤4x margins within ε — and collapses
8x accuracy. This is precisely the handoff's *phase* channel in §4.3
(S_max ≤ min(S_phase, S_dilution)): the memo cites a two-ceiling decomposition
in its own owner and then builds a predictor that assumes one of the two
ceilings is non-binding beyond the window — without naming that as an
assumption.

**Why this is not pre-empted by §8.2.** §8.2 concedes "arbitrary smooth
post-window drift remains unbounded (that is A1's content)". Correct — drift
that A1 *fails to exclude*. The break here is different: â requires the
*absence* of post-window numerator drift as a **derivation premise**, so the
unboundedness is not merely weakening the guarantee, it means **â is not
derived from the stated A1**. Dilemma, both horns fatal:
- Horn 1: keep A1 as written → S5's "under A1, the blind prediction is pure
  reweighting" is a derivation gap (the step needs numerator+population
  invariance that A1 does not contain). S7's table row "the reweighting
  formula is proven given A1" is false.
- Horn 2: add the invariances to A1 to license â → A1 now contains constant
  (degree-0) continuation of the numerator past the window. That **is** a
  fitted continuation law — the strongest kind — so §4's claim that A1 is
  "strictly weaker than any fitted continuation law" and the §3/S3 claim
  that A1's "entire visible consequence set is testable at ≤4x" are both
  false (the numerator-invariance clause has *zero* visible consequences;
  it is unfalsifiable in principle, yet the formula depends on it
  materially — 46.7–87.3% of blind prediction mass is governed by it).

**What the solver must RE-DERIVE (not patch):** either restate â so it is
actually a function of only the named assumptions (e.g., predict a *two-sided
band* in which numerator drift enters as an explicit unmeasured parameter —
which changes every S5/S13 decision rule), or list the invariances inside A1
and re-do the minimality argument honestly (A1 then sits *between* "no
assumption" and the c·(log₂S−1)(log₂S−2) family — incomparable, weaker than
nothing, and its decisive clause is untestable, so T1's "falsifies A1 before
any blind row" role collapses to "falsifies the dilution clause of A1 only").
Renaming A1b's parenthetical to cover beyond-window numerators is
parameter-bloat of the assumption with a new mechanism story — inadmissible.

### A2 — FATAL — S1's theorem, as the memo itself states it, voids the epistemic status of S5 rule (i) and S13(i); the impossibility claim also overreaches its headline

**Target statements:** S1 (headline "the answer must be conditional … or an
impossibility statement"), S5 rule (i) ("proceed only if … â(8x) ≥ 0.40"),
S11 ("all three blind lengths open together exactly once"), S13(i)
("prospective conditional negative — … because the dilution reserve was
**measurably** insufficient; this is the impossibility-shaped outcome … per S1
**no protocol could have done better** without an assumption change").

**Break mechanism.** S1 proves: for every protocol and every ε, there are two
environments ε-close on *all* protocol-visible ≤4x quantities (hence with
equal or ε-close â, since â is computed from those quantities through a
2-parameter logistic fit — a continuous map of them) while 8x exact accuracy
differs by ≥0.5. Take the environment pair (A, B) with â(A) ≈ â(B) = 0.395 <
τ, acc₈ₓ(A) = 0.85, acc₈ₓ(B) = 0.30 (the existence is S1's own construction;
S1's constraint class includes the memo's protocol — the class fixpoint is
the memo's own argument). Rule (i) permanently forecloses the single
authorized reveal in *both* environments. Consequences inside the memo's own
framework:

- "the dilution reserve was **measurably** insufficient" — false by S1: no
  ≤4x measurement identifies 8x insufficiency; the insufficiency is
  *inferred under A1*, the very assumption S1 says cannot be certified. The
  memo polices "proxy is never capability" as a hard rule elsewhere; rule
  (i)/S13(i) violate it against themselves — an A1-shadow is reported as a
  measurement.
- "no protocol could have done better" — false as stated: a protocol without
  the veto reveals and *unconditionally demonstrates* delivery on A (the
  handoff's Stage F is exactly such a measurement program). At best the
  decline saves compute; it cannot earn an impossibility label.
- "this is the impossibility-shaped outcome delivered without ever reading
  blind rows" — the outcome has the *shape* but not the *content* of
  impossibility: by S1 the pre-reveal state is compatible with both success
  and failure at 8x, so the decline is informationally empty about actual
  capability. A recorded "prospective conditional negative" would enter the
  repo as close-adjacent evidence against a protocol class that S1 itself
  says this number cannot adjudicate.

**Headline overreach (second facet).** S1's bolded conclusion "the answer
must be conditional … or an impossibility statement" is a false
trichotomy-exclusion: there is a third delivered shape — conditional
*prediction* before reveal, **unconditional post-reveal demonstration** for
the measured lengths. The memo practices this at S5 rule (iii) (measured
within interval ⇒ claim stands) yet S12 binds even the *measured* 8x/16x
outcomes to "conditional on A1 (never dropped) … because S1 says dropping it
is unsound". S1 says no ≤4x-visible *guarantee* is unsound-free; it says
nothing against an unconditional assertion grounded in post-freeze
*measurement* of the asserted quantity. The condition is load-bearing only
for â's calibration claim and for 32x-when-floored — both already handled
(S12 excludes floored 32x; S13(iv) handles wide intervals). As written, the
memo gives an untestable assumption veto power *and* hostage power over its
own success claim.

**Decision impact:** the author's brief asks whether the protocol can
*deliver* capability. Rule (i)+S11 can permanently discard the one authorized
reveal — and the memo instructs that this discarding be *reported as an
impossibility-shaped result* — with S1 itself proving the discarding is not
warranted by evidence. That can change how a future true positive is recorded.

**Minimal repair that does NOT count:** relabeling â "conditional on A1"
(it already says so), or moving τ to a lower value (doesn't touch the
epistemic status), or converting S13(i) to "A1-predicted negative" while
keeping the veto (the veto's justification is what breaks — the veto may
remain as a *cost* rule, disclosed as compute-avoidance only; it cannot be an
evidence-producing outcome). What must be re-derived: rule (i)'s status as
*decision* vs *evidence*, S13(i)'s report label, S1's headline
exhaustiveness, and S12's conditional on measured lengths.

### A3 — MODERATE (verified owner misattribution) — S7's "Obs1's 16K Native arm produced 200 empty predictions" is Obs2's fact, from a different model

`EVQ_8B_ADAPTATION_EVIDENCE_20260724` (Obs1) records 10 passkey cases at 16K
with exact 0% and first-token rank ~2043; it contains no "200 empty
predictions" anywhere. The quoted fact is in
`OLMO2_1B_SELECTIVE_QK_PHASE_ADAPTATION_20260729` (Obs2): "Native's 16K EOS
rate is not capability: all 200 predictions are empty" (OLMo-2 1.485B,
2WikiMQA). The memo's *point* (nonterminal decoding is a recorded failure)
survives on Obs2 alone, so this is not independently fatal — but the memo
markets S6/§3 attributions as owner-checked, and an owner-misquote in the
sentence that motivates the memo's central S7/S8 design change is a chain-of-custody
failure the solver must fix by re-citation, not by keeping the inference.

### A4 — MODERATE — S9's "five-task 1x retention → ≈0.8982" is a cross-owner blend presented as arithmetic on one owner's table

The PG-19 half is sound (verified: frozen 3.104234 appears identically in
handoff §3.1 and Obs3 §3, so exp(2.971047−3.090126)=0.887738 is legitimate
and does cross 0.88). The five-task half is not: Obs3 §4 reports only the
*delta* (−0.00582) on its own "identical fixed rows", never its absolute
frozen macro; 0.9151→0.8982 subtracts that delta from the **SCALE owner's**
0.315833 and divides by the SCALE owner's Native 0.345134. Row-set identity
across the two owners is assumed and unverifiable from the cited owners
(handoff §3.2's own instruction is "do not pool"). §7's label "arithmetic on
owned Observations" is therefore correct for 0.8877 and false-for-its-label
for 0.8982. The 0.03 damage budget (rule b) is likewise licensed by a single
rank-8 Q/K observation at 0.0058 — 5.15× headroom at 8× the trainable
parameters and a different (margin-bearing, V/O-touching) objective whose 1x
five-task damage has never been measured on this substrate. "Largely
dissolved" exceeds what the analogy supports; the gate rule itself is legal
(≤4x selection is in-contract).

### A5 — MODERATE — S9's failure attribution is backwards: "If no checkpoint passes (a), the substrate (not the protocol) failed the gate"

Rule (a) is "strictly beats frozen on paired 1x PG-19". The substrate's
frozen arm already passes the owner-defined gate (0.875302 ≥ 0.875; handoff
§3.1). If adapters at steps {250,275,300} all regress PG-19, the damage is
*caused by the training the protocol chose* — the protocol side. Also note
rule (a) is *stricter than the owner gate* (retention ≥0.875 never required
beating frozen); a stricter self-imposed gate is legal, but its failure must
be reported as a protocol/adapter failure, or the record will mis-close the
s4 substrate for all future arms on the strength of a mislabeled negative.
Fix is re-attribution wording — but it changes what a negative closes, so it
is listed despite being one sentence.

### A6 — MINOR-to-MODERATE — S7/S8's "+1 forward per step" is inconsistent with the guarantee "placed on the prefix distribution greedy decoding visits"

Verified against the script: the prototype is fully teacher-forced (fixed
labels, 2 forwards/step: correct + counterfactual; `source_effect[:, :-1]`
excludes EOS exactly as the memo says — lines 164–167). A *sequential*
student rollout over an answer span of L tokens costs L extra forwards, not
1 (L=2 → ×2 step cost, not the memo's ~×1.5; the 2Wiki-style span budget is
32 → ×17). The only +1-forward implementation is a single parallel
re-sampling pass, whose sampled tokens are each conditioned on the *teacher*
prefix — that is not the greedy-decode prefix distribution the claim invokes.
Separately (and not curable by more forwards): training visits positions
≤16K only, while 32x greedy decode visits answer positions in 128K context —
student-forcing fixes the token-history half of exposure bias, not the
positional-context half; A1b's ≤0.25-nat band bound constrains per-band key
logits at ≤16K and does not bound the decode-tree distribution beyond it.
The memo's own §5 ("~2x compute per step") partially discloses the cost
tension but never reconciles "+1 forward" with "the prefix distribution
greedy decoding visits".

### A7 — MINOR — S1's construction sketch has two fixable holes it should name

(i) "a rank-1 o_proj **LoRA** bias gated on that direction" — the
environment pair is defined as base weights + task distribution; a gate in
the *trained* LoRA is chosen by the protocol, not the environment. The
construction must place the gate in the base-weight o_proj delta, and
"bias" is outside the stated class (the protocol's own LoRA is bias-free).
(ii) The indistinguishability list ("all training losses … statistics")
omits **gradients** — every protocol in the class consumes gradients through
training. I checked the rescue: for θ strictly above the realized ≤4x proxy
support with gap g, the gate's value at ≤4x states decays like
w·e^{−k(θ−x)} and its derivative like w·k·e^{−kg}, so with k ≈ ln(w/ε)/g and
w bounded below by the 8x flip margin, both value and gradient gaps → 0 as
ε → 0. The construction likely survives — but the proof as *stated* doesn't
quantify over gradients, and "proven" in §7 should be "proven modulo a
gradient-indistinguishability lemma".

### A8 — MINOR — T1a's trend test has almost no power at the memo's own design

Band Spearman on n=4 points can only take |ρ| ∈ {1.0, 0.8, 0.6, 0.4, 0.2, 0}
(verified by enumeration). "|ρ| < 0.8" therefore flags only *perfect*
monotone drift; any single band inversion passes. Disclosed numbers, but the
memo sells T1 as the ≤4x-falsifier of the dilution clause; its trend
component rejects only one ordering class out of 24.

### A9 — MINOR — metric-column mixing and the S5 Δ parenthetical

S8's "Obs2's matched Native arm (0% at 8K/16K)" quotes the all-references-found
column (0%/0%) while the official-macro column (2.02%/0.38%) is quoted as
"31.63% vs 2.02%" one paragraph earlier — both in-owner, but the column
switch is unlabeled. And S5's "(Δ uses measured effective competitor ratios,
not nominal length)" contradicts its own Δ₈ₓ=ln2: n_eff beyond 16K is not
measurable under the firewall; ln2 *is* the nominal-length ratio (secondary
instance of A1's hidden invariances).

---

## 3. What is NOT under attack (checked, clean)

- **Arithmetic (surface G): fully reproduced, zero errors.** 0.875301/0.887738/
  0.915103/0.89824; −ln0.875=0.133531, −ln0.88=0.127833; ln2/ln4/ln8 =
  0.693147/1.386294/2.079442; τ=0.40 vs 0.4025 anchor; Obs3 core-4 deltas
  −.0100/+.0025/−.0225; Obs2 QA 24.84→21.48→8.57 and RULER 42.44→31.63→5.03;
  Obs1 hit@16 18.75→64.06, rank 2043, +1.5055 deletion, QA 0.1126 vs 0.2110;
  S8's frozen RULER-13 0.71397/0.66705/0.49859 vs Native 0.71308/0/0.00385;
  E1's "~25% sample loss" from a 4K margin in 16K rows.
- **Script claims:** schedule `2x,4x,2x,4x,1x` = FAMILY_PATTERN lines 19;
  EOS-exclusion `source_effect[:, :-1]` line 167; counterfactual teacher-forcing
  identity + assert lines 140–143; (128,2,L) pair views line 59; QKVO r64/a128
  defaults; BF16/Flash-only enforcement lines 188–198; margin softplus m=1,w=1.
- **Compliance (surface F):** no breach of single table+gain (T1 hook is
  eval-time instrumentation ≤16K only), blind firewall (T1/E1/â/S9-selection all
  ≤16K), do-not-revive list (no sweep of the failed 96-step Q/K objective;
  no routing/dual-table). E2 running QK and QKVO in parallel deviates from the
  preflight's sequential opening rule but the memo discloses the deviation and
  its identification rationale; not a breach.
- **Obs2 exposure attribution** ("physical ≤4K with explicit position IDs to
  16K", "never trained against dense distractor mass"): faithful to owner; the
  owner's warning ("must not be described as training without long-position
  exposure") concerns position coverage, not distractor density — the memo's
  density claim is distinct and consistent.
- **S1 vs the memo's own o_proj usage:** not a contradiction — the memo's
  protocol is inside the impossibility class and responds with A1+T1, exactly
  as S1 says a sound protocol must.
- **τ=0.40 and ε̂-from-measurement:** disclosed at §8.2/§8.3; repetition would
  be re-listing pre-emptions, so they are not counted as attacks.

## 4. If the solver survives

A1 and A2 both force re-derivation, not wording: A1 because â's derivation
depends on invariances the named A1 excludes by its own text (both horns
change what §3/§4/S7 certify); A2 because rule (i) and S13(i) grant an
object the memo's own theorem says is unmeasurable, the decision status of a
one-shot irreversible reveal and call its firing an impossibility-shaped
evidence event. A3–A5 are fixable by re-citation and re-attribution but each
currently misstates an owner or a causal close and would fail an independent
owner check as written.
