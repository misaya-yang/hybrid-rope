# Attack 2 on the Solver-Z candidate solution (SOLVER_Z_CANDIDATE_SOLUTION_20260904.md)

> **External-model archive:** not project evidence, instruction, authorization,
> or a current verdict. Re-establish every claim from canonical owners.

- **Date:** 2026-09-04
- **Role:** hostile expert reviewer (attacker 2). Job: veto hunt, not repair.
- **Method:** full re-derivation in Python/numpy of §2.1–§2.6 (all 81/108-model
  window counts, all §2.4 bands, §2.5 exhaustion, fits), line-by-line owner
  verification of the §8 register, and construction of an explicit anchor-exact
  countermodel.
- **Verdict preview: VETO** (three FATALs: F1 the "provably at 6" answer;
  F2 the §2.4 band table is not one recipe and its band-tops violate the memo's
  own s2 anchor; F3 the 8x-exclusion sentence contradicts the memo's own 28/81
  computation; plus one "proven"-mislabel FATAL-adjacent, F4).

## 1. VERDICT: **VETO**

## 2. Attacks, ranked

### F1 — FATAL. Statement 14's "(first joint infeasibility provably at 6)" and statement 5's "never ≥ 6" are false as delivered: an explicit monotone model that fits ALL five owned 1x task anchors AND all four owned PG anchors EXACTLY, including s2, admits (p2, s6) through both gates.

**Break mechanism.** The memo's engine is a trilemma it cannot escape:

- Horn A — the s2 point is trusted exact: then §2.5's exhaustion holds only
  for exponents restricted to (f,r) ∈ [0.5,2]², but the bands the memo
  actually delivers (F2 below) are computed over a "corner" that violates
  that same s2 point by 0.0156 — 2.4× the gap used to "prove" the rejection.
  The memo cannot use s2-exactness to reject a class and simultaneously use
  s2-inconsistent corner models to set band ceilings. And statement 10's
  consequence ("the repo cannot identify the 2-D damage surface") then means
  statement 14's "provably" is a word attached to an unidentified surface —
  §7 sub-claim 2 says exactly this: "the exponent r … unidentifiable from
  completed arms". "Provably at 6" (§7's own conjunction says "under any
  separable homogeneous-growth damage model") is delivered in statement 14
  WITHOUT that qualifier on the ANSWER line.
- Horn B — the s2 point is noisy (the ledger itself: "Five-task macro
  sampling noise unquantified in owners… bands within ±~.02 retention of
  gates are genuinely uncertain"; and the owner's OWN paired bootstrap for a
  difference of these very macros is [−0.0364, +0.0739] ⇒ SE ≈ 0.028, and for
  PG [−0.02507, +0.000937] ⇒ SE ≈ 0.0066 nats, both in
  NATIVE_ISOTONIC_PROFILE_RESULT §3): then "proven by exhaustion" (statement
  10) is a deterministic proof over point estimates whose 1σ noise exceeds
  the task-side gap (0.00643) by ~4×, and the class is NOT rejected — it is
  missed by 0.23σ. A proof by exhaustion over a noisy anchor is not a proof.
- Either horn, statement 14's "provably" and statement 5's flat "never ≥ 6"
  fail: Horn A kills the band table, Horn B kills statement 10, and the
  countermodel kills the conclusion under both.

**Demonstration (constructed countermodel, all numbers recomputed).**
Monotone, separable, fits every owned 1x point exactly:
- gain benefit b(α) = 0.033082·(α/α0), f = 1;
- p2 table task damage t_t(L), L = log s/log 4, piecewise-linear monotone:
  t_t(0.5) = 0.002314, t_t(1) = 0.062383, t_t(1.29248) = 0.07475;
- PG gain damage exponent q = 2.8312 (gives gain damage 0.016144 at α0/2);
  PG table damage 0 at L=0.5, 0.018302 at L=1, 0.0185 at L=1.29248 (monotone).

Anchor-fit table:

| Owned 1x point | Observed | Countermodel |
| --- | ---: | ---: |
| Native PG / task | 0 / 0.345134 | 0 / 0.345134 (exact) |
| gain-only PG / task | 3.085932 / 0.378216 | exact (anchors) |
| p2-s2 PG / task | 2.987191 / 0.359361 | 0.016144 / 0.359361 (exact) |
| p2-s4 PG / task | 3.104234 / 0.315833 | 0.133187 / 0.062383−b → 0.315833 (exact) |
| Iso-s4 PG / task | 3.092470 / 0.334461 | independent iso marginals, unconstrained by p2 curve |

Prediction under this model: at s6, α_hi = α0·((0.133531−0.0185)/0.114885)^{1/2.83} = 0.103032
(c ≈ 0.0575); task: b(α_hi) = 0.033226 ≥ t_t(1.29248) − 0.043142 = 0.031608 ⇒
**both 0.875 gates pass at (p2, s=6)**. The memo's §3 ledger lists the
separability assumption's "if wrong" cell as "interaction can reopen s6+
(E1 detects)" — but this countermodel needs NO interaction: pure separability,
monotone damage, saturating (concave-past-s4) table displacement damage. The
only prior it breaks is convexity-in-power-of-displacement — the exact
property §2.5 "proves" the data rejects. So the memo's own rejection of the
power-law class is the license for the countermodel; statement 10 cannot
reject convex power laws and statement 14 cannot then "prove" 6-infeasibility
under saturating alternatives. **What the solver must RE-DERIVE:** statement
14's ceiling must become the §7 conjunction verbatim ("not identifiable from
repository statistics"), with 4–5/6–7 demoted to labeled class-conditionals;
statement 5's "never ≥ 6" and P-Z1's counter-prediction framing must be
re-derived against the reopenability the memo itself concedes in §7.
**Repair that must NOT count:** adding "(conditional on 4)" footnote text
without changing the ANSWER line — the answer line "provably at 6" is the
delivered decision, and re-labelling is not re-derivation. Also not admissible:
invoking "the s2-consistent corner is more conservative" — the corner's own
PG leg forces table-PG-damage at s2 to be ≤ 0 (q=2.8 gives −0.000356 nats;
q=2.8312 gives exactly 0, i.e., zero table contribution at s2), which is
itself a non-power-law admission the memo never surfaces.

### F2 — FATAL (H-standard: miscomputed bands). The §2.4 band table is NOT the output of its stated recipe, and its band-tops are generated by a model that violates the memo's own s2 anchor.

**Break mechanism.** §2.4 header: bands over "q,p,f,r ∈ {1,1.5,2} plus the
s2-consistent corner f=0.15, q=2.8". I recomputed every row.
The corner is "s2-consistent" only when r ≈ 2 (exact-fit solve: f=0.15 ⇒
r = 2.0007). But the recipe leaves r ∈ {1,1.5,2} free, so it contains
(f=0.15, r=1), which predicts s2 macro 0.343758 vs observed 0.359361 —
**off by 0.015603, 2.4× the 0.00643 gap that §2.5 uses to "prove" the class
rejection.** And the printed bands use corner-r1 in SOME rows and not others:

| Row (my recomputation) | class-only task max | class∪corner(f=.15,r free) max | PRINTED max | corner-r1 used? |
| --- | ---: | ---: | ---: | --- |
| p2 s4 c=.05 | .8840 | .9096 | **.8840** | NO |
| p2 s5 c=.05 | .8653 | .8826 | **.8826** | YES |
| p2 s6 c=.05 | .8581 | .8603 | **.8603** | YES |
| p2 s7 c=.05 | .8372 | .8414 | **.8372** | NO |
| p2 s8 c=.05 | .8273 | .8273 | .8273 | irrelevant (x>1) |
| Iso s6 c=.052 | .9232 | .9306 | **.9306** | YES |
| Iso s7 c=.052 | .9166 | .9177 | **.9177** | YES |

PG rows confirm the same split: p2 s4 PG printed [.9085,.9317] = class-only
(with corner q=2.8 the max is .9449); p2 s5 PG printed top .9236 and s6 top
.9028 and s8 bottom .8517 all require q=2.8. So at least three different
model sets produced the rows of one "frozen" table.

**Downstream damage to delivered claims.**
(a) P-Z1 asserts "(p2, s=5, c=.05) five-task < 0.875 (predicted band
[0.815, 0.883]; passes only if r ≤ ~0.3, refuted by §2.5)". The assertion
contradicts its own band's upper endpoint, and the stated escape condition is
wrong twice over: the pass threshold is r ≤ 0.685 at f=1 (not 0.3; ~0.39 at
f=1.5, ~0.12 at f=2), and the ONLY models that push s5 past 0.875 at c=.05
are corner-r1-type (f≲0.4, r≤1.26) — which §2.5's s2 argument refutes by the
same 0.0156 margin, but the memo never makes that argument; its actual
refutation chain ("r≤~0.3 refuted by §2.5") does not close the gap it names.
Under the s4/s7 recipe (class-only) the s5 band is [.815,.865] and P-Z1 is
trivially consistent — so the memo's own table proves P-Z1 only if the frozen
band is RE-printed. A frozen prediction whose stated band admits its negation
is a compliance failure under §8 success rules ("prospective predictions bind
only if frozen"): what is frozen is incoherent.
(b) Statement 5: "(p2, s=6) … ≤ 0.860 even at the most favorable model corner"
— true over the class (.8581) but false over the memo's own corner-inclusive
band (.8603 > .860), i.e., the sentence and §2.4's s6 row in the same document
differ by 0.0003 through contradictory recipes.
(c) Statement 6's s4 straddle narrative uses [.863,.884]; under the header
recipe it is [.863,.9096] — the direction is conservative here but the
selectivity (drop the corner exactly where it would widen toward "passes")
is the problem: bands are not recipe-frozen, so statement 6's "proven
mechanism + bands" and statement 9's iso ceilings are band-shopping.
**Re-derivation required:** recompute every §2.4/§2.3 band under ONE declared
model set that is internally consistent with the s2 anchor used in §2.5
(i.e., constrain (f,r) to the s2-fit manifold r=r(f) or prune r=1 when
f<0.5), then re-issue P-Z1 and statement 5/6/9's endpoint numbers.
**Repair that must NOT count:** keeping the current numbers and declaring the
corner "r=2 fixed" — that silently changes half the table's endpoints
(iso s6 top .9306→.9232 etc.) without re-derivation, i.e., a new selection of
the model set after seeing which endpoints the argument needs (parameter
re-strategy, not wording).

### F3 — FATAL as headline/statement-9 text (answer-shift). "8x is excluded for every measured mask … unless task damage scales sublinearly" is refuted by the memo's own 28/81 computation; linear r=1 — inside the class, the same corner statement 5 leans on — already admits Iso s8.

**Demonstration.** My class-only recomputation reproduces statement 9 exactly:
Iso W(8) nonempty in **28/81** models; census by r: {r=1: 27, r=1.5: 1,
r=2: 0}. Sublinear r<1 is not even in the declared class [1,2]²⁴; so the
"unless sublinear" escape names a region that is neither needed nor
sufficient-by-its-own-argument. Statement 9's closing sentence "No *measured*
mask has a retention-feasible s8" contradicts statement 9's own preceding
sentence ("28/81 at s=8") — the iso mask IS measured (§3 factorial) and IS
retention-feasible at s8 in 28/81 class models. The headline delivers iso
ceiling ≤ 7; statement 14 delivers "6–7 (8 on the hair, only under linear
task-damage scaling)". Two different ceilings are delivered to Z in one memo,
and the escape condition attached to the stronger one is mathematically
misidentified. Because the user's answer to Z is the ceiling number, this is
not harmless.
**Re-derivation required:** a single answer line, with the iso-8 row carrying
its r=1 dependency and the (rejected-by-the-memo's-own-data) class status of
that feasibility made explicit — i.e., the answer must state under WHICH
model set s8 is/isn't excluded, consistent with F1's reopenability (the same
saturating countermodel, run at Iso with L=2: t_t(s8)=0.075 ⇒ Iso 8x passes
more easily than p2 6x).
**Repair that must NOT count:** deleting "28/81 at s=8" from statement 9 to
make the sentence true (evidence removal), or re-wording "sublinear"→"linear"
without also fixing statement 14's interplay with statement 10 (under the
rejected class, iso-8-feasible models are class models; if the class is dead,
the surviving r=1 corner feasibility is exactly the F1 epistemics).

### F4 — FATAL label violation. Statement 2 is labeled "proven" but its load-bearing quantitative claim ("at PG unit-gain the p2 table displacement costs only ≈ 0.018–0.041 nats even at s8, retention ≥0.96") is never measured at unit gain and is separability-conditional — the very property statement 10 says the data rejects.

**Demonstration.** The repo's owned PG arms at p2 tables are all at α0
(s2/s4/iso-s4, all on the ray α = c·log s with c fixed or boundary-selected);
the handoff's own Stage-F controls list ("candidate table with unit gain")
is required-not-run (HANDOFF §8: "Required controls … candidate table with
unit gain, Native table with candidate gain"). At α0, the s4 marginal IS
0.018302 by telescoping (exact, statement 1). At unit gain, the marginal is
t_g(4)|_{α=0} — a different quantity under any α·d interaction; §2.5's own PG
leg already forces q≈2.83 on the gain part (i.e., non-additive attribution at
s2 unless t_g(s2)=0 exactly), so the memo has measured evidence that the
gain/table split is not stable across α on the PG leg. The verdict that the
handoff's Stage-Z interpretation is "structurally wrong" is therefore proven
only inside separability; under truth it may be right — but the label and the
"proven corollary" claim in §6's honesty table are false as stated. This
also infects the headline's framing ("the incumbent's ≈4x boundary is a
GAIN-ray boundary, not a table boundary"): §2.6 correctly shows the fit is
the c=.074 ray, but calling the boundary "gain-side" presupposes that the
table's PG contribution does not grow with α — unowned.
**Re-derivation required:** relabel statement 2 (and the headline clause)
explicitly conditional on α·d separability, and downgrade "structurally
wrong" to "not supported by the owned arms (unit-gain table control
unmeasured)". If separability is assumed, statement 10's "proven rejection"
must be correspondingly softened (interaction is the alternative the memo
itself prints in §4: "separability or the exponent class is false").

### MINOR items (fixable without re-derivation; listed for completeness)

1. Statement 1 / headline: "gain consumes 86.3% of the 0.133531-nat budget" —
   recomputed: 0.114885/0.133531 = **86.04%**; table share 13.71%; the two
   sum to 99.74% (0.26% slack, which statement 8 itself later uses). 86.26%
   is the share of CONSUMED damage (0.114885/0.133187) — §2.1's printed label
   "share of budget consumed by gain = 0.8626" mislabels its own quantity.
2. Statement 12: "zero-cross near 45K" — recomputed root t=3.5435 ⇒
   4096·2^3.5435 = **47.8K**. (Fit coefficients p(3)=0.20859, p(log2 5)=0.41850
   verified exact; §4.1 quadratic a=−0.0494934, b=0.1050045 and S=3.9116/4.0060
   reproduce.)
3. §2.3: "Iso s7 PG .8838/task .8831" — recomputed mid-model values are
   PG 0.88312 / task 0.88380: the two numbers are **transposed** (statement 9's
   "passes .88" survives: both ≥0.88).
4. Statement 7 LAW-Z: "capability screen" is an unparameterized predicate in
   a procedure declared "frozen", and the mask set over which e_t is minimized
   is never enumerated (only p2/Iso have measured e_t = 0.0450/0.0316
   macro-nats per log-unit — recomputed from register: 0.062383/1.38629 and
   0.043755/1.38629, ratio 0.70 ✓). As written the law generates no new table
   and its "minimize e_t" would (unconstrained) prefer near-identity masks with
   s_max≈1. It is a decision-aid naming Iso + midpoint gains, not a frozen
   selection procedure. Needs one sentence of parameterization, no re-derivation.
5. Statement 11: "consistent with gain overdose" — the only owned
   gain-vs-1x-capability datum at s8 is L0 core-4 0.0167 vs L1 0.5925
   (SCALE §3/§4): removing gain is catastrophic, so overdose is
   unfalsifiable-but-permitted by owned data; label already "conditional
   corroboration" — admissible under the user's mechanism rule only because
   nothing in statement 14 consumes it. Keep flagged.
6. Statement 6 "proven mechanism": the "c=.05 at every factor" selection
   follows from strict monotonicity of real PG damage in α, owned at two
   points (α=0 and α0) — disclosed in the §6 honesty table ("6 additionally
   needs PG monotonicity"); §2.5's own q≈2.8 finding means the two-point
   ownership is doing real work. Disclosed → not a separate attack, but the
   "proven" header word should read "conditional" for consistency with the
   memo's own honesty table.

### Verified-clean (attack surfaces that did NOT break; recorded so adjudication is not re-run)

- **§8 register:** every value checked against owners. 2.971047461, 3.085932
  (isotonic §3 AND SCALE §6 2x2 row 1 ✓ — the gain-only arm is genuinely the
  Native-table+inherited-gain cell), 3.104234/0.315833 (SCALE §6 ✓),
  2.987191/0.359361 (SCALE §7 ✓), 3.092470/0.334461 (isotonic §1/§3 ✓),
  0.345134307/0.378216/1.095851 (isotonic §3 ✓), RULER triplet (SCALE §6 ✓),
  s8 core-4 row (SCALE §3 ✓), gate map (HANDOFF §2 ✓). No misowned input
  found. Cross-mask suite identity: the isotonic §3 table re-reports the p2
  s4 row bit-identically to SCALE §6 ⇒ same five-task evaluator/suite;
  the efficiency comparison in statement 9 is anchor-legal.
- **Window arithmetic:** p2 s4 81/81 [.0597,.1029]; s4.5 81/81 [.0761,.1020];
  s5 27/81 all-r=1 [.0908,.1014]; s5.5–s8 0/81. Iso 81/81 to s6; 60/81 s6.5;
  54/81 s7; 28/81 s8. ALL reproduced exactly from the register + §2.2 marginals.
  (b=0.033082, t_t=0.062383, gate allowance 0.043142, α-free s5 bound:
  1−0.072423/0.345134 = 0.7902 ✓ ".790".)
- **§2.5 task exhaustion number:** class max over (f,r)∈[0.5,2]² of
  NT+B·0.5^f−TT·0.5^r = 0.352931 at (f=0.5, r=2) ✓ printed 0.35293; PG class
  min 0.033297 ✓ "0.0333"; q solving 0.114885·0.5^q=0.016144 = 2.8312
  ✓ "≳2.8". (The PROOF status is attacked in F1; the arithmetic is right.)
- **P-Z1 s6 number:** class max task retention at s6,c=.05 = .85807 ≤ .860 ✓
  (but see F2(b) — the printed .8603 corner value contradicts it).
- **Split telescoping, §2.2 identities, statement 9's 0.36×/0.70× ratios
  (0.3572/0.7014), c* values (mid-model c*(4)=.06292, c*(4.5)=.06391,
  Iso s6 .05155, Iso s7 .05227), Iso s8 mid-model window gap (task shortfall
  0.0047 at PG corner at c*=.0522… ~ printed .0045)** — reproduce.
- **Compliance (G):** e_t is measured task-macro damage per log-unit — not a
  revived operator-bound/Gram/boundary-count selector; retention/capability
  kept separate in statements 11–14; 0.875/0.88 discipline maintained;
  single-checkpoint scope disclosed (§6(d)); isotonic s6/s7 export is legal
  via the existing builder lineage (`scripts/analysis/derive_native_isotonic_profile.py`
  owns m; ω_k·s^{−m_k} is the owned composition identity, SCALE §2) — E2's
  "CPU-exportable" does not overclaim a nonexistent script.

## 3. What survives a successful defense

If the solver can re-derive F2's band table under one anchor-consistent recipe
AND F1's countermodel is excluded by something already owned (it is not —
it fits every 1x number), the statements 1–4, 9-censuses, the identifiability
claims (§7), E1/E2's discriminating design, and statement 8's strict-0.88 arm
all stand. The vetoed parts are the epistemic labels ("proven", "provably",
"excluded") and the two answer lines that consume them (statement 5 "never ≥6",
statement 14 "provably at 6", headline "8x is excluded… sublinear", statement
2 "proven"). The delivered answer to Z should collapse toward the memo's own
§7 conjunction — which is arguably the honest solution, but as written the
memo answers Z twice differently (7 vs 8; never-6 vs reopens-on-single-row)
and calls a class-conditional sensitivity analysis a law.

*Attacker 2. Each FATAL was demonstrated by recomputation or explicit
construction, not alleged.*
