# Editorial Decision — "RoPE Has a Spectral Budget" (ICLR 2027 submission)

**Panel:** simulated five-seat peer review, full mode (academic-paper-reviewer v1.11.1)
**Date:** 2026-08-27 · **Model:** qwen3.8-max (all seats and synthesis) · **Manuscript state:** `main_0726` @ f9804fb
**Provenance:** Phase 0 field analysis configured four card-backed seats; the Devil's Advocate ran as the fixed fifth seat. All five seats read the full manuscript read-only and committed their reports without cross-referencing peer outputs. The editorial synthesis below traces every point to a Phase 1 report; where the editor independently recomputed a disputed fact, the recomputation is labeled EDITORIAL CHECK.

---

## 1. Decision

## **MAJOR REVISION**

**Score summary (ICLR scale):**

| Seat | Overall | Soundness | Presentation | Contribution | Confidence |
|---|---|---|---|---|---|
| EIC — Journal-Fit Reviewer | 6 | 3 | 2 | 3 | 4 |
| R1 — Methodology | 6 | 3 | 3 | 3 | 4 |
| R2 — Domain | 6 | 3 | 3 | 3 | 4 |
| R3 — Perspective | 6 | 3 | 3 | 3 | 4 |
| DA — Devil's Advocate | (unscored) | 3 CRITICAL / 7 MAJOR / 3 MINOR | | | |

**Rationale.** The panel is unanimous at 6/10 with lean-accept instincts: the identification of interior allocation $z$ as a separately controllable finite-RoPE coordinate, executed with a pinned-support three-seed control and a phase-invariant exact geometry, is a durable and reusable contribution, and the paper's self-policing (co-adaptation table, target-matched reversal, exponential tie) is above venue norm. However, three Devil's Advocate CRITICAL findings were validated — at least in corrected form — as mandatory claim-scoping obligations, and panel protocol does not permit an Accept to finalize while a validated CRITICAL stands. Every mandatory fix is achievable from existing data and closed forms (scoping sentences, a prior-sensitivity table, a phase-invariant surrogate validation, task-list documentation, citation hygiene); new compute is confined to the optional strengthening track. The decision is therefore **Major Revision**, not Reject: no reviewer located a defect that falsifies the core identification claim, and one attempted falsification (R2's Proposition 2 charge) was refuted by editorial recomputation (see §4.1).

---

## 2. Decision letter to the authors

Dear authors,

Your submission was reviewed by a five-seat panel (journal-fit, methodology, domain, perspective, devil's advocate). All four scoring reviewers place the paper at 6/10 — marginally above threshold — and lean toward acceptance **conditional on a claim-scoping revision**. The panel's judgment of the science:

1. **The core contribution survives scrutiny.** The pinned-support control design is recognized by all seats as a genuine methodological contribution to the RoPE literature; the phase-invariant pair-subspace geometry is verified correct by two independent recomputations (the headline $r_2 = 2.00$ for the 23 slow pairs reproduces at 2.0003/2.001); the co-adaptation crossing is singled out by three seats as the paper's most valuable exhibit. One reviewer's claim of a factor-of-2 error in Proposition 2 was refuted by an editorial recomputation (§4.1) — the proposition as stated is correct.

2. **What must change is the distance between your front matter and your appendices.** Your own appendix already contains the honest versions of every contested claim: the target-matched reversal, the exponential indistinguishability, the ramp's constructed dependence, the single-trajectory scope labels, the 7/9 rule-coverage cap. The abstract, intro, and §2/§4 headline sentences must carry the same scoping. Three CRITICAL findings (DA-1, DA-2, DA-3, adjudicated in §3) reduce to this single structural obligation plus two definitional gaps (the split rule and causal-distance measure).

3. **The constructive pillar needs rescoping, not removal.** The factorial does not behaviorally distinguish the Cosh shape from a matched exponential; the mature protocols do not use the zero-search rule's τ. The evidence supports "non-uniform interior allocation is a controllable, causally active coordinate; here is one admissible closed-form instance." Say exactly that.

4. **Cheap, closed-form analyses would materially strengthen the theory.** A phase-invariant re-run of your surrogate validation and a three-prior sensitivity table require minutes of CPU with formulas you already have; both are requested, not as new science, but to remove the internal contradiction between §3.1's phase-invariance doctrine and the cosine-slice validation.

A revised manuscript addressing the mandatory items in §5 (Themes A, B, and the starred items of C–E) will be in strong accept territory per the panel's stated positions. Optional Theme F items would raise the ceiling but are not required for the decision.

---

## 3. Devil's Advocate CRITICAL adjudication (mandatory visible record)

Panel rule: every DA CRITICAL must be visibly adjudicated here; a validated or unresolved CRITICAL blocks silent Accept. All three are adjudicated below. None is silently bypassed; none vetoes revision.

### DA-1 — Target-matched reversal undisclosed in body — **VALIDATED (with scope correction)**

- **Facts confirmed.** The reversal exists (+0.060/+0.227/+0.460 nats, FMRoPE favoured 3/3 seeds at every OOD length under target-matched support; App. Table right block) and is never named in the abstract, §2.1, or Fig. 1c. Independently raised by EIC-W1 (MAJOR), R1-W1 (MAJOR), and echoed in R2-S1. Four-seat consensus plus DA.
- **Scope correction to the DA's reading.** The DA states "the sign of the paper's central causal claim is an artifact of convention." This overstates the case: the identification claim actually made in §2.1 ("fixed support + different $z$ ⇒ different trained-model behaviour") holds under **both** policies — changing $z$ changes behaviour either way; what is convention-conditional is the **direction of benefit**, not the identifiability of $z$. The Table's own "Reading" paragraph concedes exactly this.
- **Required response (blocks Accept until done):** one sentence each in the abstract and §2.1, plus a Fig. 1c caption clause, stating that per-length support retargeting reverses the tested ordering and that the fixed-support result isolates the deployment case (one table serving many lengths). No new experiment required; the DA's support-sweep rebuttal evidence is recorded as optional Theme F2.
- **Status:** VALIDATED → mandatory revision item A1.

### DA-2 — Frozen headline lacks scalar-support control; ramp corroboration is by construction — **VALIDATED IN PART**

- **Validated parts.**
  (i) The same-support log-linear control is one point ($s{=}4$) with no $s$-sweep — raised compatibly by R1-W4 and EIC-W4;
  (ii) the movement-profile ramp is fit to the derived table by movement-MSE, so its agreement corroborates the movement profile, not an independent allocation (the paper concedes this; R1-W4 and R2-W2 concur);
  (iii) "+52.54 over official YaRN" is confounded by endpoint convention while the paper's own pinned-endpoint YaRN-family ramp reaches 61.04 (EIC-W4/W5; R3-W3 anchor-shopping concern);
  (iv) EDITORIAL CHECK confirms DA-6's definitional gap infecting this headline: "split rule" appears three times in `04_experiments.tex` (lines 20, 28, 29) and is defined nowhere in sections/, appendix/, or tables/; the "Native causal-distance measure" is likewise never defined.
- **Not validated.** The DA's demand for unpinned NTK-style arms as a *condition of the claim* misfires: the paper's stated claim is precisely at matched support, and moving support is the orthogonal $(a,R)$ coordinate the paper's decomposition exists to separate. The matched-support contrast is internally valid for its stated claim by construction.
- **Required response (blocks Accept until done):** relabel §4.1 "zero-training deployment" as controlled zero-training instantiation (EIC-W4); carry both anchors in the abstract's delta (the 0.56 matched-support control and YaRN's 7.94); define the split rule and causal-distance measure with a reviewer-runnable recipe (Theme D2); keep the ramp's constructed status visible at its citation sites. Optional: $s$-sweep and independent-profile ramp (Theme F2).
- **Status:** VALIDATED IN PART → mandatory items A2, D2; DA's unpinned-arm condition recorded as optional F2.

### DA-3 — Factorial contradicts shape-specificity narrative — **VALIDATED IN CORRECTED FORM**

- **Editorial correction to the DA's record.** The DA quotes the body as asserting "the exponential control rules out a generic 'any deformation works' explanation." EDITORIAL CHECK (grep of all compiled sections): no such sentence exists. §2.1's actual language ("supplies configuration and shape breadth") is disciplined, as R1-W3 independently noted. The verbatim-contradiction form of the CRITICAL is therefore dismissed.
- **The evidential core stands, by five-seat consensus.** Rule point − Geo = −0.00988, CI contains 0, $p{=}0.125$, 7/12 configurations; matched exponential ties the rule ($+0.00074$, $p{=}0.836$); the only arm clearing 0.05 is the off-rule 1.25× multiplier ($p{=}0.027$, unadjusted across contrasts); the 99-run staging result (7/9 configuration means) carries no test; the abstract nonetheless names the closed-form construction as the paper's constructive product with its uniqueness clause. Sources: R1-W3 (MAJOR), EIC-W2 (MAJOR), R2-W2 (MAJOR), R3-W6, DA-3 — the panel's single most densely supported finding.
- **Required response (blocks Accept until done):** rescope contribution (iii) to "one admissible closed-form instance of a behaviorally undistinguished shape family" (or pre-register and win the shape contrast at the 1.00× multiplier — recorded as the stronger option); state the factorial's power limitation where it is cited (§2.1); declare which factorial contrasts are confirmatory vs exploratory and adjust or disclose the 1.25× multiplicity; align the "zero-search rule" language with the actual 7/9 coverage. Items A2, D6.
- **Status:** VALIDATED IN CORRECTED FORM → mandatory items A2, D6.

**CRITICAL tally for the record: 3 adjudicated, 3 validated (1 fully, 1 in part, 1 in corrected form), 0 unresolved, 0 refuted.** Because validated CRITICALs remain, an Accept is not available at this round; all three reduce to claim-scoping, definitional, and presentation fixes that the revision roadmap below operationalizes.

---

## 4. Corrections to the review record (author protections)

The editor verified disputed facts before synthesis. Three reviewer claims do not survive verification and must not be conceded in any rebuttal:

### 4.1 R2-W5 (Proposition 2 "factor-of-2 error") — **REFUTED**

R2 claimed the stated deficit constant belongs to $1-c_{\omega\nu}$, not $2-\lVert Q\rVert_F^2$. EDITORIAL CHECK: recomputed the exact block-whitened cross-Gram from the paper's own closed form at three scales. At $(x,y)=(0.05,0.10)$ the exact-to-leading ratio for $2-\lVert Q_{x,y}\rVert_F^2$ against $\tfrac{19}{12600}(x^2-y^2)^2$ is **1.00058** — reproducing the appendix's own quoted ratio to five decimal places — and the ratio tends to 1 as $\epsilon\to 0$ across $(0.02,0.05)$ (1.00016) and $(0.08,0.03)$ (1.00036). **Proposition 2 is correct as stated.** R2's rebuttal-facing claim of a "ratio 2.0011 against the stated formula" is not reproducible. Authors should answer Q3/W5 by pointing at this check, not by correcting the proposition.

### 4.2 DA-3's verbatim quotation — **NOT IN MANUSCRIPT**

As adjudicated in §3: the quoted sentence does not exist in the compiled text. The finding survives only in its aggregate-narrative form.

### 4.3 R2-W7(vi) co-adaptation $r_2$ discrepancy — **CONVENTION DISPUTE, NOT CONFIRMED ERROR**

R2 recomputed 4.78/13.68 vs printed 4.57/12.54 under the stated uniform prior. The manuscript does not pin the exact prior/grid for that table, so this is an under-specification (fixable by caption, item C4), not a demonstrated error. Do not treat as an arithmetic correction without first checking the authors' own convention.

Also noted: DA's alternative explanation (b) (amplitude rescaling) was withdrawn by the DA itself in its naive form; the residual (Native-$z$-with-amplitude / derived-$z$-without-amplitude cells) is optional evidence, recorded under F.

---

## 5. Consensus analysis and Revision Roadmap

Consensus = ≥3 seats raising the same defect (DA counted as one seat). All roadmap items trace to named sources; the roadmap is grouped, **not ranked**. ★ marks items that are mandatory for the decision (Themes A, B and starred C/D/E items); the rest are strongly recommended or optional.

### Consensus clusters

| # | Cluster | Seats | Roadmap |
|---|---|---|---|
| C1 | Target-matched reversal undisclosed in front matter | EIC-W1, R1-W1, DA-1 (+R2-S1, R3-W3 context) | ★A1 |
| C2 | Cosh shape behaviorally undistinguished; witness-vs-winner; rule point non-significant | EIC-W2, R1-W3, R2-W2, R3-W6, DA-3 | ★A2, ★D6 |
| C3 | Positioning vs YaRN/LongRoPE/NTK-aware; taxonomy misclassification; 32 uncited bib entries | R2-W1, R3-W3, DA-12 (+EIC §5) | ★B1–B3 |
| C4 | Frozen headline evidence base (one checkpoint, undocumented nine-task subset, single $s$, ramp-by-construction, single YaRN baseline) | R1-W4, EIC-W4/W5, DA-2, DA-6 (+R2-W4) | ★A2, ★D1–D2, F2 |
| C5 | Mature-scale claims on single seeds/trajectories, pooled in prose | EIC-W8, R1-W10, DA-8 (+R2-W4, R3 soundness note) | ★A4, F1 |
| C6 | Theory predicts no behaviour; title's "budget" reads as predictor | EIC-W7, R2-W8, R3-W1, DA-4/DA-5 | C3, ★A5 |
| C7 | Surrogate validated on the cosine slice the theory discredits; surrogate↔true objective unbounded; surrogate lacks $L$/$b$ | R2-W3, R3-W4, DA-4 | ★C1 |
| C8 | τ rule not followed by mature protocols; $L_{\rm train}$ ambiguous under phase exposure | R1-W2, DA-7 (+EIC-Q4) | ★D3 |
| C9 | Presentation hygiene: EVQ never expanded, forward references, arm-name density | EIC-W3, R2-W7(i) (+EIC-Q5) | E1–E2 |
| C10 | Orphaned artifacts (`table_evq_ramp.tex`, `a4_supporting_experiments.tex`), dangling refs | R1-W9, R2-W7(v), DA-11 — EDITORIAL CHECK: both files exist and are not `\input` by main.tex | E3 |
| C11 | Separation-prior sensitivity unquantified; whitening removes energy | R3-W2 (+DA-4 partial) | C2 |
| C12 | Reproducibility gaps (factorial grid values, DiT hyperparameters, compute budget, movement-map sensitivity) | R1-W8, DA-6 (+R2-W4) | ★D2, D4 |

### Theme A — Claim scoping and disclosure (all mandatory; no new experiments)

- **★A1 — Surface the target-matched reversal.** Abstract lines 10–12 ("improves every tested OOD length in every seed, identifying $z$ during co-adaptation"), §2.1 lines 16–22, Fig. 1c caption, contribution (i): add that under per-length support retargeting the ordering reverses 3/3 seeds (App. values), and that the fixed-support column isolates the one-table-many-lengths deployment case. *Sources: EIC-W1, R1-W1, DA-1.*
- **★A2 — Rescope the constructive and deployment claims.** Contribution (iii) and abstract lines 12–14: "witness" is acceptable only paired with an explicit statement that the factorial does not separate the Cosh shape from a matched exponential (rule point $p{=}0.125$, CI ∋ 0; tie $p{=}0.836$). §4.1: "zero-training deployment" → "controlled zero-training instantiation" unless ≥2 tuning-free baselines are added. Abstract line 8: carry both anchors (0.56 matched-support control; YaRN 7.94). Where the in-window tax and routing requirement appear in §5, also name them in §4.1's deployment paragraph (two-table routing, ~30–45 RULER-point in-window cost on mature models). *Sources: EIC-W2/W4/W5, R1-W3, R2-W2, R3-W6, DA-2, DA-3, DA-10.*
- **★A4 — Carry scope labels into the body.** §4.3's 750M sentence must say seed-42-only; "repeat … through 1.485B" → "consistent direction in the tested single-trajectory protocols"; state the exact sign-test $p$ for the $n{=}3$ seed contrasts ($\ge 1/64$ per length if sign-exchangeable); add the Qwen derived−log-linear interval [0.25, 17.50] to the body where the point estimate is quoted. *Sources: EIC-W8, R1-W8/W10, R3-W8, DA-8.*
- **★A5 — Title/budget-metaphor clause.** One sentence in the intro: the budget bounds what the basis can supply; behaviour is settled by training (Table 1 is the evidence that $r_2$ does not predict PPL). This defuses C6 without touching the title; a title softening remains optional. *Sources: EIC-W7, R2-W8, R3-W1, DA-4/DA-5.*

### Theme B — Positioning and literature (mandatory)

- **★B1 — NTK-aware paragraph + taxonomy reclassification.** §2 must engage NTK-aware/dynamic-NTK scaling (zero occurrences in compiled text — EDITORIAL CHECK; anchors: LongLoRA; Fu et al. 128K data engineering) as the canonical "move $R$, keep $z$ uniform" family; recategorize YaRN's ramp and LongRoPE's searched schedules as interior allocations *combined with* support transport; restate the novelty as "$z$ was never isolated at pinned support, and no prior construction derives it in closed form." Also: relabel the 151.9M control arm "geometric (FMRoPE identity)" or state that FMRoPE at $\theta=L_{\rm train}$ *is* the geometric grid. *Sources: R2-W1, R3-W3, DA-12, DA strongest counter-argument.*
- **★B2 — Cite the classical lineage** the geometry re-derives: frame potential / Welch bound / principal angles / Landau-type density / quantization distortion. Naming them strengthens, not weakens, the claim. *Source: R3-W5.*
- **★B3 — Bibliography hygiene.** 32 of 73 entries are never cited (EDITORIAL CHECK: exactly 32, including kazemnejad2023impact, ALiBi, XPos, KERPLE, CLEX, SelfExtend, PoSE, Found/Lost-in-the-Middle). Either engage them (kazemnejad2023impact is the paper's closest motivating prior and must be cited) or prune. Disambiguate the three HoPE entries. *Sources: R2-W7(iii/iv), DA-12.*

### Theme C — Theory–metric bridge (CPU-only; ★C1 mandatory, rest strongly recommended)

- **★C1 — Phase-invariant surrogate validation.** Re-run the 12-configuration surrogate sweep with the closed-form $c_{\omega\nu}$ (App. A1 Eq. 3) instead of the cosine-feature kernel, per §3.1's own phase-invariance doctrine; additionally evaluate the true $\bar c(z;\tau)$ for the Cosh family over a τ-sweep and report argmin displacement from the surrogate's operating rule. If the ordering survives, the constructive section is materially strengthened; if it does not, say so — the paper's own standards already license negative results. *Sources: R2-W3, R3-W4, DA-4.*
- **C2 — Prior-sensitivity table.** Slow-cluster $r_2$ and full-table $r_2$ under three priors (uniform, power-law, checkpoint-derived attention-distance histogram) plus one energy-weighted collision variant; state which headline numbers move. Minutes of CPU with existing closed forms. *Source: R3-W2.*
- **C3 — One falsifiable geometry→behaviour prediction.** Either predict the in-window/OOD crossover length from a stated geometric functional across the 432M/750M/1.485B protocols, or show $r_2/\bar c$ predicts the sign of the trained contrast across the 12 factorial configurations. One success converts the theory from taxonomy to explanation; a reported failure is also acceptable under the paper's own epistemics. *Sources: R3-W1, R2-W8, EIC-Q2, DA-4.*
- **C4 — Co-adapt table caption.** Pin the exact prior and grid that produced 4.57/12.54 (R2's uniform-prior recomputation gives 4.78/13.68; see §4.3). *Source: R2-W7(vi).*

### Theme D — Statistical and reproducibility hygiene (★ items mandatory)

- **★D1 — Document the nine-task subset.** List the nine confirmation-only tasks, state the selection rule, give per-task scores for all 13 RULER families (and the Qwen four), and record the freeze artifact (date/hash) supporting "frozen before evaluation." *Sources: R1-W4, DA-6, DA-8.*
- **★D2 — Define the split rule and the Native causal-distance measure.** Both are referenced (split rule: three times in §4.1) and never defined — EDITORIAL CHECK. Provide the definitions plus a reviewer-runnable recipe that regenerates 60.47/61.04 from the public checkpoint with all choices fixed before any OOD number. *Sources: DA-6, R1-W8.*
- **★D3 — Per-protocol τ provenance table.** For every mature protocol, the $(c, L_{\rm train}, u_k)$ triple and whether τ was rule-generated or separately fixed (identification τ=4 rule-consistent; OLMo-2 τ=2, 750M τ=1.5, MLA/8B τ=1.414, video τ=1.5=0.53×rule — none rule-generated); state the $L_{\rm train}$ tie-break under phase-exposure training. "Zero-learned-parameter" must be explicitly scoped to gradients, not protocol design. *Sources: R1-W2, DA-7, EIC-Q4.*
- **★D6 — Multiplicity and contrast family.** Declare which of the five factorial contrasts are confirmatory vs exploratory; report an adjusted $p$ (or pre-specified hierarchy) for the claim-bearing one; treat the 1.25× $p{=}0.027$ accordingly. *Sources: R1-W3, DA-3.*
- **D4 — Design-values tables.** Factorial grid enumeration (2 bases × 2 lengths × 3 head dims — values never listed), DiT hyperparameters (LR/batch/steps/axis partition), 8B LoRA LR/dropout/steps, revision-pinned eval corpora, and a compute-budget appendix entry. *Sources: R1-W8, DA-6.*
- **D5 — Arithmetic and unit hygiene.** Footnote that differences are computed from unrounded values and CIs are percentile intervals (59.92 vs 59.91; 52.54 vs 52.53; +0.9% vs +1.13%); name the co-adaptation bootstrap's resampling unit ("configuration-level" over a single configuration is vacuous as written); add anchor/document-level intervals for the 151.9M contrasts plus one micro-batch-geometry swap to show the seed effect is geometry-free. *Sources: R1-W5/W6/W7, EIC-W6.*
- **D7 — Protocol reconciliation.** One shared table (same checkpoint, length, scorer) reconciling the frozen route (54.40/60.47) with the matched continuation route (6.13/5.03), with an explicit account of the gap — or scope §4.2 out of the persistence narrative. *Source: DA-9.*

### Theme E — Presentation and artifact hygiene

- **E1 — Expand EVQ at first use**, including the abstract and macro definition (`\evq` currently renders only "EVQ-Cosh" — EDITORIAL CHECK). *Sources: EIC-W3(i)/Q5, R2-W7(i).*
- **E2 — First-pass legibility.** Forward-pointer in §2.1 to §3.4 for the Cosh/τ/anchor machinery; move the four-family classification sentence into §1; add an arm-dictionary box (Geo, Native, FMRoPE, anchored EVQ, derived, log-linear, ramp, long profile, session policy); relocate Eq. (2)'s machinery or label "instantiated in §4.1." *Sources: EIC-W3/W9.*
- **E3 — Orphan artifacts.** `tables/table_evq_ramp.tex` (454M, 3-seed, PK@8K 100±0%) and `appendix/a4_supporting_experiments.tex` exist but are never `\input` (EDITORIAL CHECK); the former references an undefined `\ref{sec:ramp-scaler}`. Include them with definitions and correct statistical unit, or remove them from the release archive. Fix the dangling "App. B.7" references (App. B ends at B.6) and Table 10's undefined MLA wavelength-blend row. *Sources: R1-W9, R2-W7(v), DA-11.*
- **E4 — Venue residue.** main.tex preserves an ICML-formatted fallback venue block; remove before submission. *Source: DA §4.*

### Theme F — Optional strengthening (compute-permitting; not required for the decision)

- **F1** Second independently trained arm at ≥750M or a second MLA seed (best value-per-compute against C5). *Sources: R1-W10, DA-8.*
- **F2** Support × allocation sweep ($s\in\{2,4,8,16\}$, paired best-of comparisons) and an independent ramp profile (official YaRN α/β with pinned endpoints, evaluated on held-out rows). Satisfies DA-1/DA-2 rebuttal evidence if it succeeds. *Sources: DA-1, DA-2.*
- **F3** A second frozen checkpoint, or task-resampling beyond the fixed nine (intervals currently exclude task-population uncertainty). *Sources: R2-W4, DA-8.*
- **F4** Map LeRoPE's published learned profiles into $z$-coordinates (and/or a matched-budget learned-table run) — the cheapest corroboration of the coordinate outside the authors' surrogate. *Sources: R2-W6, DA §3(g).*
- **F5** One anchored exact-range control at a realistic base ($b\ge 10$K) within the 50–150M budget, answering the small-base objection; state the factorial's two bases (never given). *Sources: R2-W4, R1-W8(i), Q1.*

---

## 6. What happens next

Per panel protocol, a Major Revision decision triggers revision coaching (Phase 2.5): a Socratic pass in which the authors triage every item as `will_address` / `wont_address` / `not_on_point`, producing an immutable roadmap + author-adjudication sidecar. The authors may also say "just fix it" to skip coaching and proceed directly to implementation.

**Editorial note on rebuttal posture:** should any of these findings return as external referee reports, the authors are protected on three points verified in this record — Proposition 2's constant is correct (§4.1), the quoted "rules out" sentence does not exist in the manuscript (§4.2), and the co-adaptation numbers are a convention dispute pending caption, not a demonstrated error (§4.3).
