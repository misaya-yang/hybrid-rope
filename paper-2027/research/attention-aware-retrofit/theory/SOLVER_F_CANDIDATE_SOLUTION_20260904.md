# Solver F: identifiability-delimited LoRA conversion protocol (candidate solution)

> **External-model archive:** not project evidence, instruction, authorization,
> or a current verdict. Re-establish every claim from canonical owners.

- **Date:** 2026-09-04
- **Status:** solver candidate memo; **not a claim owner**; no new measurement.
  Routing into `INDEX.md` is deferred to the author decision. All cited numbers
  are repo-owned Observations or arithmetic derived from them (derivations
  shown). No GPU run was performed or authorized.
- **Question (F):** can a small-data/few-step LoRA trained only on physical
  1x replay + 2x/4x rows, on one static table + fixed gain, deliver useful
  blind 8x/16x/32x generated capability under separate ≥0.875 retention gates?
- **Chosen path:** **(C) hybrid** — a robust impossibility theorem delimiting
  what no constrained protocol can guarantee, plus the protocol that is optimal
  relative to the weakest named, ≤4x-falsifiable structural assumption.
- **Base facts used (owners):**
  `main_0726:rebuttal/rebuttal_0723/theory_results/EVQ_8B_ADAPTATION_EVIDENCE_20260724.md`,
  `main_0726:rebuttal/rebuttal_0723/theory_results/OLMO2_1B_SELECTIVE_QK_PHASE_ADAPTATION_20260729.md`,
  `main_0726:paper-2027/research/attention-aware-retrofit/results/LOG_P2_QK_LORA_GAIN_MATCHED_RESULT_20260904.md`,
  `main_0726:paper-2027/research/attention-aware-retrofit/results/SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md`
  (via handoff §3.1),
  [`SINGLE_TABLE_ROPE_OPEN_PROBLEMS_HANDOFF_20260904`](../../attention-aware-retrofit/analysis/SINGLE_TABLE_ROPE_OPEN_PROBLEMS_HANDOFF_20260904.md),
  [`CONSTRAINED_FRONTIER_AND_2X4X_LORA_PREFLIGHT_20260904`](../../attention-aware-retrofit/preflights/CONSTRAINED_FRONTIER_AND_2X4X_LORA_PREFLIGHT_20260904.md),
  `scripts/train/train_log_p2_phase_transfer_lora.py`.
  Length convention: 1x=4K, 2x=8K, 4x=16K, blind 8x/16x/32x = 32K/64K/128K.

## 1. The solution in 15 statements

**S1 — Impossibility theorem (the negative half, proven).** For *any* protocol
that trains and selects only on statistics computable from rows of physical
length ≤4x, under one fixed static table/gain and LoRA of any fixed rank on any
of {q,k,v,o}, and for any ε>0, there exist two environments (base weights +
task distribution) whose induced laws differ by <ε on every protocol-visible
quantity (all training losses, all ≤4x NLL/margin/attention/rank/accuracy
statistics, at any finite selection precision δ) while differing by ≥0.5 in 8x
greedy exact accuracy. *Justification (construction):* the residual stream of a
softmax transformer carries a prefix-count proxy: take a head whose value
outputs are a shared constant direction plus token noise; its output deviation
after averaging over n keys concentrates at rate n^(−1/2), and a frozen GELU/
ReLU unit reading that direction exposes a monotone mean shift in 1/√n. A
rank-1 o_proj LoRA bias gated on that direction with threshold θ placed inside
the strict gap max(proxy over the finitely many realized ≤4x states) <
min(proxy at 8x) shifts every 8x answer argmax and, by scaling gate sharpness
up and output weight down at fixed product ε, leaves every ≤4x statistic
within ε. Gate-only models also leave the prototype's counterfactual margin
and all prepared diagnostics within ε of ungated ones at ≤4x. Hence: **an
unconditional "guarantees useful 8x+" claim is unavailable to any protocol in
the constraint class; the answer must be conditional (named structural
assumption + ≤4x-falsifiable pre-test) or an impossibility statement.** Labels:
derived result (exact-ε version; construction over model class, not a claim
that any repo model has a gate).

**S2 — The one layer of genuine identifiability (remark, not rescue).** For a
*frozen single Q/K pair*, ℓ(Δ)=Σ_k γ_k[a_k cos(ω′_kΔ)+b_k sin(ω′_kΔ)] lies in a
fixed 2K=128-dimensional span independent of the adapter (the table is static;
LoRA moves coefficients, not frequencies), so ≥128 distinct visible relative
distances determine that pair's entire distance curve in exact arithmetic.
This does not rescue the general problem: composition through depth puts the
coefficients themselves in functions of prefix content, hidden states are not
finite-basis in Δ, and S1's gate lives at the composed level. It does justify
one thing: the *dilution* part of the blind prediction is combinatorial
(competitor count is known exactly at every length), while the *functional
shape* part is assumption — which is where A1 is placed. Label: derived result
(single-pair case); Interpretation (its role in the design).

**S3 — A1, the weakest transfer assumption (named).** For the per-query
generated-answer margin Ĝ = logp(gold answer+EOS | context) − logp(gold |
same context, remote source swapped) (the prototype's own same-target
counterfactual, evaluated at inference), the blind-length behavior is assumed
only through the identity Ĝ(S) = z_R(S) − LSE_D(S) with:
(A1a) **distractor log-combinatorics:** LSE_D(n₂) − LSE_D(n₁) = ln(n₂/n₁) + ε
between lengths, where n is the actual unmasked effective competitor count,
|ε| ≤ ε̂ and ε̂ is the maximal within-window deviation of the measured
band-LSE curve from slope 1 (T1a) — so ε̂ is *measured at ≤4x, not chosen*;
(A1b) **numerator/adapter band-stability:** the frozen numerator's distance
profile and the adapter's per-band treatment effect are flat within measured
within-window bounds (T1a source-band + T1b), i.e., the adapter may not have
learned a distance-specific policy inside [4K,16K] that is absent beyond.
Nothing else is assumed: the margin *distribution*, the margin→accuracy *map*,
the answer prior, and all EOS behavior are estimated in-domain and carried
across unchanged. A1 is strictly weaker than any fitted continuation law (the
§5 c·(log₂S−1)(log₂S−2) family violates A1a wherever its visible curvature is
nonzero, and A1 is exactly "extrapolate only the combinatorial term with
visible curvature capped at ε̂"). It is not *uniquely* weakest — no weakest
element exists in a smooth-function hierarchy — but it is the weakest member
whose entire visible consequence set is testable at ≤4x. Label: structural
assumption (working hypothesis by construction; naming is the deliverable).

**S4 — The ≤4x pre-test T1 that can falsify A1 before any blind row is read.**
On held-out 1x/2x/4x rows only, for the selected adapter and for frozen:
(T1a) split each 4x row's key positions into four distance bands (0–4K, 4K–8K,
8K–12K, 12K–16K); per retrieval query and head, compute band distractor LSE
and mean distractor logit; regress measured LSE_D against ln(n) over the
cumulative band boundaries n∈{4K,8K,12K,16K} *within the same rows*; require
slope ∈ [1−η, 1+η] with η=0.15 pre-registered, and require band-mean logits to
show no monotone trend (|Spearman ρ across band index| < 0.8). A visible
super-logarithmic LSE growth or drifting band means is exactly how a nonzero
c-term or phase-driven distractor nonstationarity first appears; failing T1a
voids the transfer claim without touching 32K.
(T1b) adapter-minus-frozen per-band mean logit shift, separately on source-band
and distractor-band keys; require |shift| ≤ 0.25 nats uniformly across the four
bands (a learned "only boost keys 12K–16K" policy is then disqualified from
any blind extrapolation). All of T1 is computable with the prepared
adapter-aware evaluator plus a band-restricted attention-stats hook; it never
needs an 8x row. Label: diagnostic design; its discriminative power is
calibrated by S5, which is the stated predicted relationship.

**S5 — Calibrated transfer predictor with a pre-registered decision rule (the
discriminating diagnostic).** On the same held-out ≤4x rows, fit a monotone
map P̂(Ĝ)=σ(α(Ĝ−g_c)) from measured counterfactual margin to *greedy exact
correctness* (this fit is in-domain: both quantities exist at 1x/2x/4x —
recorded core-4 at 4K/8K/16K for s4 are all ≤4x). Then, under A1, the blind
prediction is pure reweighting: â(S) = E_rows[P̂(Ĝ − Δ_S − ε̂)], Δ_8x=ln2,
Δ_16x=ln4, Δ_32x=ln8 (verified arithmetic; Δ uses measured effective
competitor ratios, not nominal length). *Decision rule, frozen before opening
blind rows:* (i) proceed only if both retention gates pass (S11) and T1 passes
and â(8x) ≥ 0.40 (author-registered usefulness floor; note 0.40 is anchored
only on published ≤4x values — s4 frozen core-4 16K = 0.4025 — never on blind
outcomes); (ii) record the triple (â(8x), â(16x), â(32x)) with intervals
inherited from ε̂ and fit uncertainty; (iii) after the single blind reveal:
measured within interval ⇒ conditional transfer claim stands (labeled "under
A1"); measured ≪ prediction ⇒ A1 falsified, report the falsification, no
rescue sweep; measured ≫ prediction ⇒ A1 falsified upward, no promotion
beyond the longest length actually covered by prior evidence. The diagnostic
genuinely discriminates: it predicts failure (rule i abort + rule iii), so a
pass is not a constructed win.

**S6 — Why previous failures are three different broken links, and what was
never simultaneously present (attribution; owners in §3).** Conversion at ≥2x
required three conditions at once: (C1) a substrate carrying the long signal
(carrier), (C2) trainable transport/readout capacity (V/O present in the
trainable graph, fresh or inherited), (C3) a learning signal whose gradient
vanishes unless the answer logit flows through the remote span, evaluated at
the prefixes greedy decoding actually visits. Obs1 had C1,C2, failed C3
(generic LongAlpaca CE can be paid by the prior; rank 2043 with 64% hit@16 is
the direct measurement); Obs2 had C2,C3 but its exposure was position-sparse
(physical ≤4K with explicit position IDs to 16K), so it never trained against
dense distractor mass — its 16K weakness is where A1's Δ term was untrained;
Obs3 had C1,C3-at-teacher-states but not C2 (fresh Q/K-only rank-8, no
trainable V/O) and not C3-at-decode-states. The prepared prototype is the
first arm with C1+C2+C3(teacher) — and has no answer at all for A1/accounting
(S1–S5) or for decode-state placement (S7).

**S7 — The same-target counterfactual margin does NOT fix link 3 alone; the
patch is placement, and it is justified or replaced (hard-rule item).**
Retention of the margin is justified for links 1–2 only: CE fixes the answer
token's logit level against the vocabulary (top-1 needs level), margin fixes
source-dependence (long-range needs dependence), and Obs1 proves level+
dependence at the teacher state can still leave argmax wrong at decode state
(teacher-forced first-token rank 2043 is computed on gold prefixes; free-run
exact 0%). The link-3 repair is therefore *not* the margin term but **S8**:
evaluate CE and margin at **student-forced** answer prefixes — one-step
self-rollout of the answer span (sampled token re-enters, no gradient through
sampling, +1 forward pass per step) — so the top-1 guarantee is placed on the
prefix distribution greedy decoding visits, and add the terminal-EOS
transition to the contrasted positions (the prototype's
`source_effect[:, :-1]` excludes EOS; Obs2's failure signature at 8K was a
61% EOS rate, and Obs1's 16K Native arm produced 200 empty predictions —
nonterminal decoding is a recorded, not hypothetical, failure mode). This is
a change in the *input distribution the loss sees*, not a new loss name on the
same objective, and is explicitly one of the unresolved causes handoff §5
itself lists ("output-level credit remains too weak"). Falsifier: if the
teacher-forced arm passes all ≤4x gates and transfers while S8 adds nothing,
exposure-bias is refuted as a component; if S8 transfers where the prototype
fails in-window, exposure-bias is established. Labels: working hypothesis
(placement fix); conditional derivation (the margin/link-1–2 vs CE/link-3
decomposition).

**S8 — Substrate choice: s4/c=.074, and the carrier premise is measured, not
assumed.** §5 requires "the static table already exposes useful long carriers"
for any transfer. On this substrate the *frozen, zero-training* arm already
scores RULER-13 macro 0.71397/0.66705/0.49859 at 4K/8K/16K (owner:
SCALE_CONSISTENT_LOG_PROFILE via handoff §3.1) against Native 0.71308/0/0.00385
— the carrier exists through 4x without any adapter. Obs2's matched-Native
arm (0% at 8K/16K under the identical continuation) shows the carrier is a
substrate property, not a LoRA property. Stage F therefore runs on s4/c=.074
if Stage-Z retains s4; if Stage-Z selects a smaller factor, the carrier
premise must be re-measured frozen at 2x/4x before Stage F consumes GPU. If
Stage-Z's selected factor exceeds 4 with gates passing, that arm replaces s4
and S5's calibration baseline length changes accordingly (Δ measured from the
largest trained physical length). Label: Observation (frozen scores) +
protocol rule.

**S9 — The retention-gate answer (the "no headroom" tension is largely
dissolved by an owned measurement).** The brief notes s4/c=.074 sits at 0.875302
with ~zero headroom. But the *completed* Obs3 adapter on the exact same table
improved paired 1x PG-19 NLL from 3.104234 to 3.090126 against Native
2.971047, i.e., retention 0.8753 → **0.8877** (derived arithmetic from the
owner's own table; crosses the strict 0.88 gate), and five-task 1x retention
0.9151 → ≈0.8982 (also derived; passes). A likelihood-repairing adapter on
this substrate *creates* NLL-gate headroom because the substrate's 1x damage
is partly a weight/table mismatch that any 1x-replay-bearing CE reduces. The
gate rule is therefore made active, not passive: an intermediate checkpoint is
acceptable only if it (a) strictly beats frozen on paired 1x PG-19 and
(b) drops the 1x five-task macro by ≤0.03 (vs observed 0.0058 at rank 8;
0.03 is the registered budget for rank-64 QKVO), checked at steps {250, 275,
300} — selection on ≤4x statistics is legal under the data contract. If no
checkpoint passes (a), the substrate (not the protocol) failed the gate and the
run is a stop, reported as such. Labels: Observation (Obs3 deltas) +
arithmetic (retention conversions) + rule.

**S10 — Parameter placement and matched controls are part of the solution, not
an afterthought.** Minimum screen: frozen s4 / QK arm / QKVO arm with *actual
trainable parameter counts* matched within 1% and receipts, identical data,
order, tokens, optimizer, schedule (handoff §5's GQA rank≠capacity warning is
binding: rank 64 QK ≠ rank 64 QKVO in parameters). Obs2+Obs3 jointly predict
QK fails at 4x already (transport), so a QK-only pass on the new objective
would falsify the Obs3 attribution (see §3 falsifier column). QKVO opens only
through the preflight's own ordering (QK first; QKVO if routing succeeds but
oracle-routed answer/EOS is weak) — except that this solution's arm E2 (§6)
runs both from the start *because* the attribution prediction is now explicit
and two-sided.

**S11 — Data contract and firewall, restated as hard protocol properties (not
a solution per se, but the solution's validity conditions).** Training rows:
physical 4K/8K/16K identifiable correct/deranged pair views only, + 1x replay;
8x/16x/32x rows generated before training but sealed; no selection statistic
(early stop, checkpoint choice, gate, τ, α̂, ĝ_c, ε̂) may reference sealed
rows; all three blind lengths open together exactly once; every prediction of
S5 is written to a hash-bound decision file before opening. Source-
counterfactual identity check (`counterfactual_host[positions]=correct`
preserves teacher forcing; verified token-change assert) already exists in the
prototype and is retained.

**S12 — What the protocol's success would license, stated pre-run.** A pass
(all gates, T1 pass, measured ∈ interval at 8x and 16x, 32x may floor)
licenses: **"small LoRA trained at ≤4x converts frozen-carrier capability to
useful blind 8x–16x generated capability, conditional on A1 (which passed its
falsifiable pre-test), on this checkpoint/family"** — the conditional is part
of the claim, never dropped, because S1 says dropping it is unsound. It does
not license "LoRA solves RoPE extension", 32x if floored, or cross-checkpoint
universality (multi-seed + second model remain required before any general
method claim, per handoff §8).

**S13 — What each failure mode is reported as (the result is informative in
all directions).** (i) T1 fails or â(8x)<0.40 at selection time: *prospective
conditional negative* — the protocol declined the blind reveal because the
dilution reserve was measurably insufficient; this is the impossibility-shaped
outcome delivered without ever reading blind rows, and per S1 no protocol could
have done better without an assumption change. (ii) Retention fails:
substrate-gate negative (S9 rule (a)). (iii) Blind measured ≪ prediction:
A1-falsification result — first repo evidence that distractor nonstationarity
or learned length-gating dominates past 4x, which per S1 no calibration could
have caught; publishable negative. (iv) Blind passes but prediction was wide:
transfer established but the calibration machinery is not validated — downgrade
S5 to descriptive. Each outcome closes exactly one named object.

**S14 — Minimal relaxation that restores a guarantee (theorem companion to
S1).** The gate construction in S1 violates A1 but is invisible to ≤4x
statistics; hence the *minimal* relaxations are exactly: (R1) any structural
restriction excluding length-gated response beyond the training window — A1 is
one such; stronger alternatives (global power-law fit through 2x/4x) are
already shown unidentified by §5; or (R2) one blind-proximate observation
(e.g., a single 8x *dev* row entering selection — breaks the stated constraint
class, so it is named and rejected); or (R3) dropping the retention gate
(Obs1 shows the ungated path: conversion to zero-capability at scale is cheap
— EVQ-LoRA *improved* 16K NLL by 1.51 nats while exact stayed 0; the gate is
what makes this problem hard, not the training budget). The delivered answer
keeps all constraints and chooses R1 via A1+T1.

**S15 — Anti-revival compliance.** Nothing here is a rank/alpha/gain/LR/step
sweep on the failed 96-step Q/K objective, nothing re-adds routing/dual
tables, and no proxy is promoted as capability: NLL, margin, band-LSE, and
rank statistics appear only as (i) gates, (ii) inputs to a calibration whose
*output* is compared to greedy exact endpoints, or (iii) falsification tests
of a named assumption. The deltas over the prepared prototype are exactly:
S5 calibration+decision file, S7/S8 student-forced placement and EOS contrast,
S9 active checkpoint acceptance, S10 receipts, S4 pre-test — each traceable to
a named broken link in §3, not to hyperparameter taste.

## 2. Which recorded observations the solution must explain

It must explain all three (they are the problem statement's evidence base), and
it is consistent with all three:

- **Obs1 (Llama-3-8B EVQ-LoRA):** explained as C3-absence (no source-forcing
  signal, teacher-placed) → links 1–2 moved, link 3 flat (exact 0, rank 2043);
  the −0.39 NLL regression at 8K and 303-example QA deficit are in-window
  damage of a generic-CE objective, predicted by the framework to be
  orthogonal to conversion.
- **Obs2 (OLMo-2 1.485B Q/K continuation on EVQ):** explained as the one
  successful conversion below the dense-exposure bar: carrier (C1) present on
  EVQ and absent on matched Native (identical training → 8K 31.63% vs 2.02%);
  the 16K decay (5.03%) is the S5 dilution term Δ=ln2 from 8K→16K *untrained
  densely* (its long exposure was position-sparse: physical ≤4K, explicit
  position IDs — a different q_eff; the solution's 2x/4x dense rows are the
  relaxation that Obs2 could not use while keeping "physical 2x/4x" literal).
  No inconsistency.
- **Obs3 (log-p2 Q/K rank-8 96-step):** explained as C2-absence (no trainable
  V/O) plus teacher-state-only signal: PG-19 up, core-4 flat **even at 4K**
  (in-window!) — the framework predicts failure before any extrapolation
  question arises. Also explains S9 (same run created 1x retention headroom).
  No inconsistency.

Residual tension admitted: Obs2 shows conversion is possible with **only** Q/K
trainable — but only because its V/O capacity was *inherited trained* (Stage-A
parent), which the framework classifies as C2-present-by-parenthood, not C2-
absent. The prediction that discriminates: a fresh QK-only arm on the new
objective must fail at 4x (§6 E2); if it instead succeeds, S6/C2 is refuted.

## 3. Link-level attribution of previous failures

| Failure | Broken link | Evidence (owner) | What would falsify this attribution |
| --- | --- | --- | --- |
| Obs1: 300-step r64 QKVO EVQ-LoRA, NLL↑ hit@16 18.75→64.06%, gold-block deletion +1.5055 nats causal, exact 16K = 0%, first-token rank 2043, QA ≤ Native | **3 (top-1 readout)** at decode states; links 1–2 demonstrably moved | [`EVQ_8B_ADAPTATION_EVIDENCE_20260724`](../../../../rebuttal/rebuttal_0723/theory_results/EVQ_8B_ADAPTATION_EVIDENCE_20260724.md) Levels 2–3 | Showing argmax was correct but terminal-EOS/scoring masked it (owner records greedy exact 0% with rank 2043, so argmax itself wrong); or showing free-run (non-teacher) rank ≈1 at 16K (would move blame to the rank diagnostic, not link 3) — resolved by the S8 student-state margin assay: D2 converts ⇒ placement was the gap; D2 fails where prototype passes in-window ⇒ placement refuted |
| Obs2: 4x (16K) weak after strong 2x transfer: QA 24.84→21.48→8.57 F1; RULER 42.44→31.63→5.03; 4x exposure was position-sparse ≤4K physical | **4 (dilution reserve)** at the untrained dense regime; **1 intact** where carrier present (matched Native arm's 0 at 8K/16K localizes the rest to the substrate, not the recipe) | [`OLMO2_1B_SELECTIVE_QK_PHASE_ADAPTATION_20260729`](../../../../rebuttal/rebuttal_0723/theory_results/OLMO2_1B_SELECTIVE_QK_PHASE_ADAPTATION_20260729.md) (incl. "explicit position IDs" caveat §Intervention) | Measuring on that substrate the T1a band curve and the margin distribution: if the 16K *numerator* z_R had collapsed (source-band gain negative vs 12K band) rather than margins shifting down by ≈ln2 with shape preserved, the attribution is link-1 phase cost, not link-4 dilution |
| Obs3: paired PG-19 up (+0.0141/+0.0187), five-task ≤0, core-4 deltas −.0100/+.0025/−.0225 at 1x/2x/4x | **2+3 (transport capacity + placement)**; the failure is *in-window* so no link-4 claim is even admissible | [`LOG_P2_QK_LORA_GAIN_MATCHED_RESULT_20260904`](../results/LOG_P2_QK_LORA_GAIN_MATCHED_RESULT_20260904.md) §1, §4, §5 | A parameter-matched fresh-QK arm *with* S7/S8 signals passing 4x (would show rank-8/96-step underfit was the cause, i.e., budget not placement — then Obs2-parenthood reading of C2 is wrong); or the same-objective QKVO arm failing 4x too (would move cause from C2 to C3, i.e., counterfactual credit itself insufficient) |

Cross-check on pooling discipline: the three attributions are different
(link 3-placement / link 4-dilution / links 2–3-capacity), which is why the
handoff's "do not pool" warning is respected rather than papered over.

## 4. Structural assumption, weakness argument, pre-test, decision rule

A1 + T1 are S3–S5. Consolidated statement:

- **Assumed object:** only the blind-length *reweighting* of the margin
  distribution by the known competitor-count log-growth, with within-window
  measured slack ε̂; and adapter band-flatness within measured limits.
- **Why weakest available:** every alternative candidate assumption either
  (a) is strictly stronger (fit continuation laws through 2x/4x — §5
  unidentified; single-frequency phase-noise models — excludes A1-compatible
  environments; "network has no length detector" — itself untestable and
  stronger than A1 since A1 merely refuses to *license* extrapolation of
  detector-shaped terms), or (b) leaves nothing to assume *and* nothing to
  guarantee (S1). The theorem makes "no assumption" unavailable; the visible-
  curvature cap ε̂ makes A1 the assumption with the smallest visible-consequence
  footprint that still yields the reweighting formula.
- **≤4x falsifier:** T1a slope/trend + T1b band-flatness (§4=S4), plus S5
  rule (i) which can *pre-call* the protocol's own failure.
- **Pre-registered decision rule:** S4(η=0.15, ρ<0.8, ±0.25 nats) +
  S5 (τ=0.40, one reveal, three frozen predictions, four-way interpretation
  per S13) + S9 (gates on {250,275,300}) — all thresholds fixed here, before
  any new run, anchored only on published ≤4x numbers or author utility.

## 5. Protocol spec: deltas over the prepared prototype

Prototype (kept): s4/c=.074 substrate; pair-view data at 4K/8K/16K + 1x
replay; `2x,4x,2x,4x,1x` 300-step schedule; answer+EOS CE + same-target
counterfactual softplus margin (m=1, w=1); QKVO r64/a128; BF16 Flash-only;
identity/stop battery.

1. **Loss placement (S7/S8):** add student-forced view per row (one-step
   self-rollout over the answer span, no gradient through sampling; +1
   forward/step); apply CE + margin at visited prefixes; include the terminal
   EOS transition in the contrasted set (fix `source_effect[:, :-1]` to also
   contrast the pre-EOS→EOS position; EOS dependence on source swap is the
   nonterminal-failure probe).
2. **Inference-margin module (S5):** same-target margin Ĝ and band-LSE
   stats computed by the adapter-aware evaluator on 1x/2x/4x held-out rows;
   fit P̂ (2 params, monotone logistic); write the S5 decision file
   (predictions, τ, rule) + hash before the single 32K/64K/128K reveal.
3. **T1 module (S4):** band-restricted attention statistics hook (no materialization
   beyond existing per-head logits; Flash-compatible via per-band LSE from the
   masked-logit stats the prepared evaluator already exposes, else a
   documented chunked recompute ≤16K rows only).
4. **Acceptance rules (S9/S11):** checkpoint set {250,275,300} selected only on
   paired-1x-beats-frozen + five-task drop ≤0.03 + T1; far rows never enter.
5. **Arms (S10):** frozen / fresh-QK (param-matched) / prototype-QKVO /
   D2-QKVO; plus Native-table param-matched LoRA control run only on a pass,
   for substrate attribution (handoff Stage F step 8). Receipts: actual
   trainable counts, exposure hashes (already in prototype).
6. **Budget unchanged** (300 steps; the +1 forward per step is the only cost
   increase — ~2x compute per step at r64 on 4K/8K/16K rows; on the 32 GiB
   machine this fits because rows are processed per-variant sequentially in
   the prototype).

## 6. Discriminating experiments (≤3)

**E1 — Margin-reserve dilution assay (no training; prereq to everything).**
On s4 frozen and candidate adapters, 4x held-out rows: compute per-query Ĝ vs
*truncated competitor count* n′∈{4K,8K,12K,16K} (prefix truncation — an
in-domain operation at ≤4x). Competing explanations: H-dilution (A1a) predicts
Ĝ(n′) ≈ Ĝ(16K) − ln(16K/n′) with slope 1 in ln n′ and accuracy following
P̂; H-phase (link-1 decay of the numerator with distance) predicts numerator
Ĝ-band collapse *inside* [4K,16K] and LSE slope ≠1 — distinguishable outputs,
opposite predictions for T1 and for â(8x). E1 outcome routes the run: pass →
S5 prediction; fail → S13(i) prospective negative without blind reveal.

**E2 — Placement arms at fixed budget (fresh-QK vs prototype-QKVO vs
D2-QKVO, parameter-matched receipts).** Predictions are crossed: C2-capacity
story predicts QK fails 4x, QKVO passes 4x; budget-underfit story predicts QK
passes on core-4 at 4x with the new signal (refutes Obs3 attribution);
exposure-bias story predicts QKVO(teacher) passes 4x but D2 is required for
the blind lengths (D2−QKVO blind gap is the exposure-bias estimate). Any two
arms swapping outcomes refutes one of the §3 attributions — no outcome is
consistent with "everything we said already".

**E3 — Carrier premise ablation (s4 vs matched Native-table, identical
D2-QKVO contract).** Obs2 predicts Native fails 8x/16x while s4 transfers
(C1 is substrate-borne). Rival "LoRA creates the carrier" predicts Native
transfers equally. Also required by preflight Stage F step 8 for causal table
attribution if the s4 arm passes.

## 7. Honest status per main statement

| Statement | Status |
| --- | --- |
| S1 impossibility | **Proven** (derived result; ε-construction over the model class; caveat: quantized-weight realizability of the gate head is a construction, not an observation) |
| S2 single-pair identifiability | **Proven** (trig-span argument), with its non-portage to composed networks stated |
| S3 A1 | **Conditional** — the named assumption; by construction a working hypothesis |
| S4 T1 | **Design**; discriminative only relative to A1's visible consequences (proven that drift of the fitted form must show slope error; arbitrary invisible post-16K deviations are not excludable — S1 says none are) |
| S5 calibration + rule | **Conditional protocol**; the reweighting formula is proven *given A1*; τ/η registered pre-hoc |
| S6 synthesis of conditions | **Interpretation** (explains all three observations; falsifiable via §2 residual tension + E2) |
| S7 margin/link-3 decomposition | **Conditional argument + design** (justified against Obs1 measurements; the placement fix itself is a working hypothesis) |
| S8 substrate/carrier premise | **Observation** (frozen 0.667/0.499 at 2x/4x; Native arm contrast in Obs2) |
| S9 retention arithmetic | **Arithmetic on owned Observations** (0.8877/0.8982 are derived from owner table values; gate budget 0.03 is a registered rule) |
| S10 parameter matching | Protocol rule (handoff-mandated) |
| S11 firewall | Protocol rule |
| S12 licensed claim | Pre-registration |
| S13 outcome taxonomy | Pre-registration |
| S14 relaxations | **Derived** (R1: by S1; R3: supported by Obs1's ungated NLL-wins-with-zero-capability) |
| S15 compliance | Statement of record |

## 8. Where a hostile reviewer should attack first (stated by the author of this memo)

1. S1's gate needs a nonlinear unit to read a length proxy — a reviewer may
   demand an explicit 2-layer transformer instance rather than the
   concentration argument; the proof survives at ε (not exact) strength, and
   exact-strength claims were deliberately not made.
2. ε̂ from a 4-point band regression bounds only the fitted-form deviation;
   arbitrary smooth post-window drift remains unbounded (that is A1's content,
   disclosed, not concealed).
3. τ=0.40 is author-anchored (0.875-gate convention and the s4 16K=0.4025
   published value), not derived from blind data; if challenged as arbitrary,
   the rule still discriminates because it is frozen before the reveal.
4. E1's prefix truncation changes *which* queries exist near the cut; the assay
   must restrict to queries whose positions are far from the truncation edge
   (≥4K margin), which reduces effective sample by ~25% and must be powered
   at 20+20 rows/cell before launch.
