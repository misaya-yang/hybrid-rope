# Solver-Z candidate solution: the two-gate budget-window law for the zero-training ceiling

- **Date:** 2026-09-04
- **Status:** solver candidate analysis (independent-solver round 1). Repository
  reading + deterministic arithmetic only. No run, no GPU, no new measurement.
  Not a result owner; all input numbers are cited to existing owners. Prospective
  predictions here bind only if frozen before any new outcome is read.
- **Question (Z):** one global request-static RoPE table + one fixed scalar gain
  at every length, zero weight training; Native damage budget PG-19 tail
  retention >= 0.875 (literal 0.88 reported separately) AND five-task
  downstream retention >= 0.875, measured separately; what is the maximum
  feasible extrapolation factor S, and by what selection law?
- **Headline:** the 1x PG-19 budget of the incumbent is 86.3% gain and 13.7%
  table (exact identity). Feasibility is therefore a *window* in gain at each
  table factor: PG-19 caps the gain from above; the downstream task gate caps
  it from *below* (the gain helps 1x downstream), and the table's displacement
  raises the task floor faster than the PG cap falls. Under this law the p2
  family's first jointly-infeasible factor is 6 for every gain in the Stage-Z
  grid and every separable damage model with exponents in [1,2]; s5 survives
  only in the linear-task-damage corner. A different, already-measured mask
  (exact-u isotonic) roughly doubles the retention window (feasible to s6-s7).
  8x is excluded for every measured mask at the 1x gates unless task damage
  scales sublinearly in displacement; and the capability/dilution side of the
  ceiling is unresolvable-as-asked from repo numbers (inputs unmeasured).
  The prepared Stage-Z mechanism (select gain by 1x PG calibration) is
  predicted to *understate* the ceiling by one to two factors and to report
  ceiling 4; window-center gain selection is the fix, and three added cells
  decide the remaining unknown.

## 1. THE SOLUTION IN 15 STATEMENTS

1. **Exact split identity (proven).** At 1x PG-19 the incumbent's NLL increase
   over Native (2.971047 → 3.104234) telescopes exactly as
   `0.133187 = 0.114885 + 0.018302`, where 0.114885 is the gain-only cost
   (Native table + `g=1.102586`: NLL 3.085932, retention 0.891469) and
   0.018302 is the p2-table marginal at that gain. The gain consumes **86.3%**
   of the 0.133531-nat (retention 0.875) budget; the table consumes 13.7%.
   Inputs: `main_0726:paper-2027/research/attention-aware-retrofit/results/NATIVE_ISOTONIC_PROFILE_RESULT_20260903.md` §3 factorial +
   `main_0726:paper-2027/research/attention-aware-retrofit/results/SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md` §6.
2. **Reframing (proven corollary of 1).** The descriptive quadratic boundary
   "factor ≈ 3.91–4.01" (handoff §4.1) is the *gain* boundary of the ray
   `g = 1 + c·log s`, `c=.074`, not a table/phase boundary. The handoff's
   Stage-Z interpretation line "no gain rescues 1x PG-19 ⇒ the p2 table
   movement sets the path-specific bound" is structurally wrong: at PG
   unit-gain the p2 table displacement costs only ≈ 0.018–0.041 nats even at
   s8 (retention ≥ 0.96 extrapolation range) — the PG gate constrains the
   *gain*, and the *table* is constrained elsewhere.
3. **The second gate is reversed (proven from owners).** At 1x the fixed gain
   *helps* five-task downstream retention: gain-only macro 0.378216 vs Native
   0.345134 (retention 1.095851). The p2 table costs 0.062383 macro at the
   anchor gain. So PG-19 caps α = g−1 from above; the task gate needs gain to
   *offset* table damage — it caps α from below. Feasibility at factor s is a
   window `W(s) = [α_lo(s), α_hi(s)]`, and `α_lo` grows with table displacement
   faster than `α_hi` shrinks.
4. **Model class (declared, conditional).** Anchor four owned 1x numbers
   (Native; gain-only; p2-s4; p2-s2) and assume separable damage
   `D_PG(α,d) = D_g(α/α0)^q + t_g(d/d0)^p`,
   `Δtask(α,d) = b(α/α0)^f − t_t(d/d0)^r`, exponents in [1,2], anchors
   `D_g=0.114885, t_g(s4)=0.018302, b=0.033082, t_t(s4)=0.062383,
   α0=0.1025858, d0=log 4`. This is the weakest class consistent with
   monotone/convex damage; every band below is computed over the class.
5. **p2-family ceiling (conditional on 4, computed).** `W(4)` is nonempty in
   81/81 exponent models (α ∈ [.0597,.1029] — the incumbent sits at the PG
   corner, which is exactly why boundary selection found .074). `W(5)` is
   nonempty in 27/81 models, only when r=1, α ∈ [.0908,.1014] ⇒ c ≈ .056–.062.
   `W(5.5)=W(6)=W(7)=W(8)=∅` in 81/81. Independently: **(p2, s=6) fails the
   0.875 task gate for any grid gain, ≤ 0.860 even at the most favorable model
   corner** (and fails PG-19 outright at c ≥ .074); **(p2, s=8, c=.05)** fails
   *both* gates (PG ≤ .866, task ≤ .827). So the maximum feasible factor for
   the p2 mask is **4 measured, 5 only in the linear-task-damage corner, never
   ≥ 6.**
6. **Why Stage-Z-as-written will report 4 (proven mechanism + bands).** PG
   damage is strictly increasing in α, so the protocol's "lowest-NLL gain per
   factor" calibration selects c=.05 at *every* factor. Its five evaluated arms
   are (s, c=.05). Predicted task retentions: s4 [.863,.884] (straddles the
   gate — the protocol risks rejecting even the incumbent geometry), s5
   [.815,.883], s6 [.771,.860], s7 [.730,.837], s8 [.689,.827]. Every band
   above s4 fails the task gate in most of the class. The protocol therefore
   cannot demonstrate factor > 4 even where the window law says 5 is feasible,
   because it under-selects the gain. **This is the sense in which new
   selection theory changes which arm should win: the winning gain at each
   factor is the window midpoint `α*(s)`, not the PG minimizer.**
7. **Selection law (frozen, pre-outcome, computable from owned numbers).**
   `LAW-Z: choose table mask to minimize measured damage efficiency
   e_t = t_t(d0)/d0 subject to a nonempty capability screen; set
   s = min(target, s_max(mask)); set α*(s) = mid[W(s)], c* = α*(s)/log s`.
   Window endpoints solve `D_g(α/α0)^q = ε_PG − t_g(d0)(log s/d0)^p` (upper)
   and `b(α/α0)^f = t_t(d0)(log s/d0)^r − 0.125·N_T` (lower). Mid-model values:
   c*(4)=.063, c*(4.5)=.064; corner s5: c*≈.059; Iso s6: c*≈.051; Iso s7:
   c*≈.052.
8. **Strict-0.88 upgrade (falsifiable, same table).** LAW-Z's s4 arm
   (p2 log-s4, **c≈.063, g=1.0873**) is predicted to pass *both* gates at the
   literal 0.88: PG ∈ [.890,.894] (robust across q ∈ [1,4]) and five-task
   ∈ [.888,.904]. The incumbent's marginal status is an artifact of
   boundary selection (0.26% budget slack); backing the gain off the PG corner
   buys strict-0.88 with predicted no 1x cost. Long-length capability at
   c=.063 vs .074 is untested (gain matters at 16K per the L0 guard) — this
   claim is 1x-only.
9. **Mask damage efficiency (measured).** Two completed masks share the
   construction and differ in measured 1x marginal damage at the same s4
   displacement: p2 costs 0.0183 PG-nats and 0.0624 macro; exact-u isotonic
   costs 0.0065 PG-nats (0.36×) and 0.0438 macro (0.70×). LAW-Z applied to the
   isotonic mask gives `W(s)` nonempty in 81/81 models through s=6, 60/81 at
   s=6.5, 54/81 at s=7, 28/81 at s=8; mid-model s7 arm: PG .884, task .883
   (passes .88). Predicted retention ceiling: **isotonic ≈ 6–7 where p2 ≈ 4–5.**
   No *measured* mask has a retention-feasible s8.
10. **The data reject the moderate model class (proven by exhaustion).** The
    observed p2-s2 1x five-task macro (0.359361) exceeds every separable
    power-law prediction over exponents in [0.5,2]² anchored at the s4/gain-only
    points (max predicted 0.35293). The s2 PG point similarly forces gain
    damage super-quadratic (effective exponent ≳ 2.8) or table-marginal
    nonadditivity. Consequence: the repo cannot identify the 2-D damage
    surface — the *only* quantities that decide whether the ceiling is 4 or 5
    (the task-damage exponent r) are unidentifiable from completed arms because
    all three table arms lie on the coupled ray α = .074·log s. One off-ray
    factor-5 cell and one factor-6 cell at 1x resolve it (§5 E1).
11. **Retrodiction of the old s8 failure (conditional corroboration).** The
    s8/c=.10 arm had α=.2079 = 2.03α0 ⇒ predicted PG retention .55–.77, i.e.
    it was 1x-PG-dead before any phase story; its measured 1x core-4 0.5925
    (vs .7550 for s4/L1) is consistent with gain overdose. The s8 failure is
    not evidence against 5–8x tables in general; it is (mostly) evidence
    against 8x *gains*.
12. **Capability side (descriptive, must not be merged with retention).** No
    arm in the repo has ever demonstrated generated capability at ≥ 5x while
    passing both 1x gates: the s8 arm collapses capability (single-key-3
    0.00/0.20/0.25, early stop) and gates; the gated incumbent is only measured
    through 4x (RULER-13 .71397/.66705/.49859 @4K/8K/16K). A log2-quadratic
    extrapolation through the incumbent's own three points predicts
    **RULER-13 ≈ 0.209 at 32K (8x), ≈ 0.418 at 5x**, zero-cross near 45K;
    the single measured 8x cell that exists (s8/c=.10 core-4 0.3025) is the
    same degraded band. Descriptive, not a bound.
13. **Dilution/phase arm of §4.3 is uninstantiable from repo numbers
    (path-C component).** `S_dilution` needs ρ, n0, μ, B; `S_phase` needs w_k,
    η, b, Hbar_j, κ_η,j with verified curvature — every one of these is
    UNAVAILABLE: no attention-mass/margin measurement exists for any gated arm
    beyond what §12 cites, and a single-length margin cannot separate `log q`
    dilution decay from power-law phase decay (both monotone in S). Therefore
    *the factor ceiling that mixes retention and long capability cannot be
    closed-form derived from the repo at all*; what *is* derivable now is the
    1x double-gate feasibility statement (statements 5, 9), which already caps
    the p2 family below 8x without touching dilution.
14. **Answer to Z, defensible now.** Largest *demonstrated* S = 4x (incumbent,
    marginal at 0.88). Largest *retention-feasible* S under LAW-Z: p2 family
    4–5 (first joint infeasibility provably at 6); isotonic family 6–7
    (8 on the hair, only under linear task-damage scaling); no measured mask
    reaches 8x inside the gates under the s2-consistent corner. Largest
    *demonstrated useful capability* at ≥ 5x under passing gates: none —
    predicted 0.35–0.45 at 5x, ~0.21–0.30 at 8x (descriptive). Zero-training
    8x with both gates and useful generation is, on repo evidence, a
    capability- and mask-efficiency question, not a PG-19 question.
15. **Minimal relaxations that could change the answer (named exactly,
    §7).** (i) two off-ray 1x task cells at factors 5, 6 (identifies r →
    settles 4 vs 5 vs model-class failure); (ii) one CPU-exportable isotonic
    s6/s7 arm evaluated on both 1x gates + fresh core-4 (tests whether the
    ceiling is mask damage-efficiency or shared geometry); (iii) two-length
    logit-margin mass probe on the incumbent (the only way to instantiate the
    dilution ceiling). Nothing weaker in (i)/(iii) suffices: one factor cannot
    identify an exponent; one length cannot separate the two decay laws.

## 2. CORE DERIVATION AND EVIDENCE CHAIN (every equation checked)

All inputs from owners: `main_0726:paper-2027/research/attention-aware-retrofit/results/NATIVE_ISOTONIC_PROFILE_RESULT_20260903.md` §1/§3,
`main_0726:paper-2027/research/attention-aware-retrofit/results/SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md` §6/§7,
[`SINGLE_TABLE_ROPE_OPEN_PROBLEMS_HANDOFF_20260904`](../../attention-aware-retrofit/analysis/SINGLE_TABLE_ROPE_OPEN_PROBLEMS_HANDOFF_20260904.md) §4.1.
Python (numpy, stdlib) verification outputs appended verbatim.

**2.1 Split identity.** α0 = 0.074·log 4 = 0.1025857827228719 (matches recorded
attention scaling 1.102585782722872).

```
gain-only damage D_g   = 3.085932 - 2.971047461 = 0.114884539   (ret .891469)
table marginal  t_g(4) = 3.104234 - 3.085932    = 0.018302000
total                  = 0.133186539 = exp -> 0.87530180  == owner 0.875302
gate budgets: -log(.875) = 0.133531393 ; -log(.88) = 0.127833372
share of budget consumed by gain = 0.8626
```
The split is exact by telescoping at fixed anchors; its *portability* across
s is the model assumption (statement 4).

**2.2 Task-side anchors.** `b = 0.378216 − 0.345134307 = 0.033082` (gain
benefit at α0), `t_t(4) = 0.378216 − 0.315833 = 0.062383`; task gate
`macro ≥ 0.875·N_T = 0.301992` ⇔ `t_t − b ≤ 0.043142`; incumbent net marginal
`0.062383 − 0.033082 = 0.029301 ≤ 0.043142` ✓ (matches measured retention
0.915103). Isotonic marginals: `t_g^I(4) = 3.092470 − 3.085932 = 0.006538`,
`t_t^I(4) = 0.378216 − 0.334461 = 0.043755`.

**2.3 Window computation (all 81 exponent combinations, {1,1.5,2}^4).**

```
=== p2 mask ===   feasible models / 81, alpha union over feasible models
 s=4  : 81/81  [.0597,.1029]      mid-model c*=.0629
 s=4.5: 81/81  [.0761,.1020]      mid-model c*=.0637
 s=5  : 27/81  [.0908,.1014]      (all feasible rows have r=1)
 s=5.5:  0/81   s=6: 0/81   s=6.5: 0/81   s=7: 0/81   s=8: 0/81
=== Iso mask ===
 s=4..6: 81/81;  s=6.5: 60/81;  s=7: 54/81;  s=8: 28/81
 Iso mid-model arms: s5 PG .9166/task .9072; s6 PG .8983/task .8951;
                     s7 PG .8838/task .8831; s8 window EMPTY (gap .0045)
```

**2.4 Frozen per-arm bands (q,p,f,r ∈ {1,1.5,2} plus the s2-consistent corner
f=0.15, q=2.8).** PG band / task band, Stage-Z grid gains and LAW-Z gains:

```
p2 s4 c=.05  : PG [.9085,.9317]  task [.8630,.8840]  <- protocol arm, straddles
p2 s5 c=.05  : PG [.8916,.9236]  task [.8154,.8826]  <- protocol arm, mostly fails
p2 s5 c=.06  : PG [.8756,.8885]  task [.8413,.8851]  <- LAW-Z s5 (corner only)
p2 s6 c=.05  : PG [.8773,.9028]  task [.7712,.8603]  <- TASK FAILS ALL MODELS
p2 s6 c=.06  : PG [.8508,.8658]  task [.7946,.8716]  <- BOTH FAIL ALL MODELS
p2 s8 c=.05  : PG [.8517,.8660]  task [.6894,.8273]  <- BOTH FAIL ALL MODELS
p2 s8 c=.074 : PG [.7411,.8189]                      <- statement 11
p2 s8 c=.10  : PG [.5986,.7708]                      <- old arm retrodiction
Iso s6 c=.052: PG [.8911,.9083]  task [.8673,.9306]  <- LAW-Z candidate, passes
Iso s7 c=.052: PG [.8814,.8872]  task [.8435,.9177]  <- passes PG .88; task straddles
Iso s8 c=.052: PG [.8626,.8773]  task [.8114,.9163]  <- discriminating cell
```
And the α-free table bound: at unit gain the task gate caps the p2 table at
retention ≤ .790 already at s5 (linear-damage best case), so **the table,
not the gain, is what the task gate kills** — the two gates bind different
variables (statement 3).

**2.5 Model-class exhaustion at the s2 point (proves the identification gap).**
Observed s2 five-task macro 0.359361; best separable prediction over exponents
in [0.5,2]² is 0.352930. PG at s2: observed damage 0.016144 vs class minimum
0.0333 (pure quadratic) — consistent only with effective gain exponent ≳ 2.8.
Both facts push toward the s2-consistent corner (superlinear displacement
damage, saturated gain benefit), under which p2 s5 task = .849–.852 →
**p2 ceiling = 4, Iso ceiling = 6** (s7 fails at .816 even with saturated
benefit). The corner and the r=1 corner disagree only about factor 5 (p2) and
7–8 (Iso); Stage-Z measures exactly those cells.

**2.6 Independent verification of repo fits.** The §4.1 quadratic reproduces
exactly (fit a=−0.0494934, b=0.1050045 vs repo −0.0494925, 0.105004; boundary
S = 3.9115 / 4.0057 vs 3.91/4.01) — confirming statement 2's reading, since
the fit ray holds c fixed at the gain that itself sits at the PG corner.
RULER-13 decay fit through (.713974, .667051, .498590) at log2-lengths
(0,1,2): `p(t) = −.060769 t² + .013846 t + .713974`; p(3)=0.2086,
p(log2 5)=0.4185, zero-cross t=3.543 (≈45K). Descriptive.
The log-law composition identity `ω(s1s2)=ω(s1)s2^{-m}` is owned
(SCALE_CONSISTENT §2); LAW-Z inherits it trivially (both masks use the log
installation).

## 3. ASSUMPTION LEDGER

| Quantity | Value/def | Source | Independent of candidate? | If wrong |
| --- | --- | ---: | --- | --- |
| Native 1x PG-19 NLL | 2.971047461 | owner OBS | yes (table-free) | all bands shift |
| Gain-only 1x PG NLL / task | 3.085932 / 0.378216 | owner OBS (isotonic §3 factorial) | yes (Native-table arm; defined before p2 selection existed) | split identity dies with statement 1 |
| p2-s4 anchors | 3.104234 / 0.315833 | owner OBS | incumbent (post-hoc by nature) | window endpoints off |
| p2-s2 anchors | 2.987191 / 0.359361 | owner OBS | zero-refit frozen | exhaustion result (§2.5) changes |
| Iso-s4 marginals | 3.092470 / 0.334461 | owner OBS | independent construction, frozen before s4 c-selection reuse | statements 9 Iso rows fall |
| Gate budgets | 0.133531 / 0.043142 macro | arithmetic on 0.875 | yes | none (definition) |
| Separability + exponents ∈[1,2] | model class | FROZEN HERE (declared §2, never tuned) | yes — frozen before any new outcome | all ceilings conditional; interaction can reopen s6+ (E1 detects: both-extra-pass outcome) |
| Table-damage displacement exponent r | ∈[1,2], unresolved | UNIDENTIFIED from repo (§2.5) | — | decides 4 vs 5 (p2), 6 vs 8 (Iso) |
| Phase support requirement s ≥ target S | geometric reading | hypothesis (composition identity only; 32K s4 behavior UNAVAILABLE) | — | if s4-table supports 5–8x capability, factors decouple from reach and the whole window question becomes gain-only |
| Dilution inputs ρ, n0, μ, B; phase-surrogate inputs w, η, b, Hbar, κ | — | **UNAVAILABLE** (no owner) | — | §4.3 ceiling stays uninstantiable; statements 13 |
| Capability decay fit coefficients | §2.6 | post-hoc on owned cells | no | statement 12 is descriptive; falsifiable at 32K |
| Five-task macro sampling noise | unquantified in owners | — | — | bands within ±~.02 retention of gates (s4/c.05, Iso s6/s7 task) are genuinely uncertain |

Post-hoc usage disclosure: the four anchors include the incumbent, which was
outcome-selected at the PG boundary. LAW-Z is frozen as a *procedure*; its
predictions above are then independent of any candidate's future outcomes.
The s2/s4/gain-only exhaustion (§2.5) uses only completed owners.

## 4. STRONGEST FROZEN NUMERIC FALSIFIABLE PREDICTION

**P-Z1.** In Stage-Z as written (PG-calibrated gain ⇒ c=.05 at every factor),
the five-task 1x retention of the **(p2, s=5, c=.05)** arm is **< 0.875**
(predicted band [0.815, 0.883]; passes only if r ≤ ~0.3, refuted by §2.5),
and the **(p2, s=6, c=.05)** arm's five-task retention is **≤ 0.860** — first
joint gate failure at factor **6 for LAW-Z gains, at factor 5 for the
protocol's own gains, and the protocol will report ceiling 4.**
Counter-prediction boundary: if any (p2, s ≥ 6, c ∈ {.05,.074,.10}) arm passes
the five-task gate at ≥ 0.875, then separability or the exponent class is
false (α·d interaction), and this solution's ceiling statements are vetoed by
a single row.
Companion frozen claims: (p2, s4, c=.063) PG ∈ [.890,.894], task ∈ [.888,.904]
(strict 0.88 pass); (Iso, s6, c=.052) PG ∈ [.891,.908] and task ≥ .867 (pass in
all s2-consistent models). RULER-13 of the incumbent at 32K = 0.209 ±
(band [0.05, 0.35]; ≥ 0.35 refutes the decay description; ≤ 0.05 also refutes
the quadratic arm of it).

## 5. DISCRIMINATING EXPERIMENTS (max 3, each names two explanations)

**E1 — two off-ray 1x cells: (p2, s=5, c=.06) and (p2, s=6, c=.06), five-task
+ PG-19.** Distinguishes (i) *linear task-damage corner* (r=1 ⇒ predicts s5
passes both, s6 fails both) vs (ii) *s2-consistent corner* (r≥2 ⇒ predicts
both fail the task gate) vs (iii) *separable-class failure* (both pass ⇒
α·d interaction; ceiling reopens above 6 and statements 5/14 must be
withdrawn). No other completed or planned arm separates (i) from (ii);
each outcome changes the named answer to Z (5 / 4 / >5).
**E2 — LAW-Z isotonic arms (Iso, s=6, c=.052), CPU-exported from the existing
frozen exact-u builder, then 1x double gate + fresh core-4 at 1x/2x/6x-length.**
Distinguishes (i) *mask damage-efficiency sets the ceiling* (Iso passes the
1x gates where p2 fails ⇒ search for lower-e_t masks is the live route to
gate-passing 8x) vs (ii) *shared geometric/readout ceiling* (Iso fails gates
anyway, or passes gates yet collapses on fresh core-4 at 6x despite retention
margin ⇒ the boundary is frozen-readout compatibility, not damage per
displacement ⇒ zero-training 8x is closed as a class question, and the
honest statement becomes (F)-route dependency). The measured p2-vs-Iso
efficiency ratio (0.70 task, 0.36 PG) makes (i) vs (ii) genuinely open.
**E3 — two-length margin probe on the incumbent: target-vs-distractor logit
identity `G = LSE_R(z) − LSE_D(z)` at 16K and 32K on ~10 identifiable pair
probes (inference-only, table unchanged).** Distinguishes (i) *dilution-limited*
(numerator intact, G falls ≈ log q from added distractors; target mass at 16K
already near ρ ⇒ instantiates §4.3 with measured μ, n0 ⇒ a real S_dilution
number becomes derivable) vs (ii) *phase/coupling-limited* (G falls faster than
log q, or A_R itself decays by 16K ⇒ the capability ceiling is phase-support,
and LAW-Z's retention ceiling is not the binding one anyway). A single length
cannot separate them (both monotone), which is why exactly two lengths are
required.

## 6. HONEST STATUS PER STATEMENT

| # | Status |
| --- | --- |
| 1, 2, 3 | **Proven** (exact arithmetic on owned numbers) |
| 4 | declared model class (frozen here) |
| 5, 6, 9 | **Conditional** on 4 (computed bands; 6 additionally needs PG monotonicity in α, owned only at two points) |
| 7 | procedure, frozen; its inputs are conditional |
| 8 | **Conditional + falsifiable**; 1x-only scope |
| 10 | **Proven** (exhaustion over declared class, arithmetic in §2.5) |
| 11 | conditional retrodiction (old s8 arm's 1x PG was never measured) |
| 12 | **Descriptive** (post-hoc fit, three points) |
| 13 | **Unresolved-as-asked** with named missing inputs |
| 14 | answer at the labeled strength of 5/9/12/13; "largest demonstrated" row is Observation-bounded (4x) |
| 15 | derivation-backed identifiability claims; conditional on 4/10 |

Hostile-expert notes. (a) No NLL/attention proxy is presented as capability:
§12 and E3 keep RULER exact-match as the endpoint and PG-19 as retention
currency only. (b) No revived selector: nothing here uses boundary counts,
Gram/Ky-Fan, operator norms, or slot permutations. (c) The tautology test:
D_g, t_g, b, t_t are fixed by *completed owners including two non-p2 arms*;
they are not re-fit to any candidate table or to Stage-Z outcomes; the exponent
class was declared before the bands were computed; the residual
post-hocness is the incumbent's own boundary anchoring, disclosed in §3.
(d) Single-checkpoint (OLMo-2-0425-1B-Instruct) scope; bands are
family-specific, never a global RoPE upper bound. (e) The Iso arms'
capability risk is owned by a real negative (fresh core-4 −.145/−.175 at
4K/8K for Iso s4) and is not washed away by their better retention margins —
E2 must gate on capability, and a pass there still licenses only "retention-
feasible 6–7", not a deployment claim.

## 7. THE PATH-(C) COMPONENT AND WHY NOTHING WEAKER RESCUES SOLVABILITY

Two sub-claims are individually unresolvable-as-asked from the repo:

1. **The dilution/phase ceiling** (§4.3's `S_dilution`, `κ_η` machinery):
   every input (μ, n0, B, ρ; w, η, b, Hbar, κ) is UNAVAILABLE and none can be
   made available by arithmetic on owned numbers. Minimal relaxation: E3
   (one arm, two lengths, inference-only). A single-length version cannot work
   because dilution (`log q`) and phase-decay predictions are both monotone and
   are only separable by their *rate across lengths*. A pure-CPU version cannot
   work because μ is a property of realized attention on real distractors, not
   of the frequency tensor (frequency-multiset-preserving permutations already
   prove realized behavior is not spectrally determined: scale-consistent
   owner §6 fork).
2. **The exponent r that decides 4 vs 5 (p2) and 6 vs 8 (Iso):** unidentifiable
   from completed arms *because* every table arm lies on the coupled ray
   α ∝ log s (three collinear points in a 4-parameter surface; §2.5 shows the
   surface is not even inside the moderate separable class). Minimal relaxation:
   E1's two off-ray cells. One cell cannot identify an exponent; a sweep of
   the existing ray (Stage-Z's factor grid at PG-selected gain) is collinear by
   construction and cannot identify it either — which is precisely why the
   prepared protocol, restated, is not a solution.

If both relaxations are refused, the correct answer to Z is the conjunction:
**"4x demonstrated; ≥6x p2-family infeasible at 1x under any separable
homogeneous-growth damage model; 8x feasible in no measured mask's 1x window
under the s2-consistent corner; and the true ceiling between 4 and 8 is not
identifiable from any statistic computable from the repository as it stands."**

## 8. INPUT-VALUE REGISTER (everything load-bearing, with owners)

```
2.971047461  Native 1x PG-19                 NATIVE_ISOTONIC_PROFILE_RESULT_20260903 §3
3.085932     gain-only 1x PG (ret .891469)   same §3; matches SCALE_CONSISTENT §6 2x2 row 1
3.104234     p2 s4 1x PG   (ret .875302)     SCALE_CONSISTENT §6
2.987191     p2 s2 1x PG   (ret .983986)     SCALE_CONSISTENT §7
3.092470     Iso s4 1x PG  (ret .885660)     NATIVE_ISOTONIC §1/§3
0.345134307  Native five-task macro          NATIVE_ISOTONIC §3
0.378216     gain-only five-task (ret 1.095851)  "
0.315833     p2 s4 five-task (ret .915103)   SCALE_CONSISTENT §6
0.359361     p2 s2 five-task (ret 1.041220)  SCALE_CONSISTENT §7
0.334461     Iso s4 five-task (ret .969075)  NATIVE_ISOTONIC §3
.71397/.66705/.49859  incumbent RULER-13 @4K/8K/16K  SCALE_CONSISTENT §6
.5925/.4575/.4625/.3025  s8 L1 core-4 @1x/2x/4x/8x    SCALE_CONSISTENT §3
0.074/1.102585782722872  incumbent gain                SCALE_CONSISTENT §6
gate map .875/.88 -> .133531/.127833 nats              HANDOFF §2
```

*Prepared by solver Z, round 1. Not routed as a claim owner; parent should
either route via INDEX.md after attack round, or discard.*
