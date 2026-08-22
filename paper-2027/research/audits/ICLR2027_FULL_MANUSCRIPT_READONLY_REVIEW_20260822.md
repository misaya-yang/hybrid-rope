# ICLR 2027 full-manuscript read-only review

- **Date:** 2026-08-22
- **Status:** internal read-only analysis record; not manuscript prose; no
  manuscript file was modified
- **Reviewer scope:** complete source read of `paper-2027/` against
  `AGENTS.md`, `HANDOFF.md`, `research/README.md`, the canonical synthesis,
  the narrative-reflection audit, and the exact-range raw-hash owner JSON
- **Method:** every section, appendix, table, statement, checklist, and
  provenance document read in full; mechanical checks run without writing
  into the manuscript tree

## 1. Executive conclusion

The manuscript is submission-viable and scientifically coherent. Its central
claim — interior allocation `z` of a finite RoPE table is a separately
identifiable training-time variable at fixed sampled support — is supported
by an evidence chain that is unusually disciplined about protocol identity,
seed scope, and endpoint separation. The three hard theorems (budget
identity, collapse proposition, transplant obstruction) are exact where
exactness matters and are honestly scoped. No claim-ceiling violation from
`AGENTS.md` was found in the current text.

The dominant remaining risk is not truth but **conversion**: the paper still
asks a hurried reviewer to absorb many scoped results before experiencing one
discovery. The narrative-reflection diagnosis stands after this full read.
The second risk class is **unfinished venue-validity work**: four checklist
items remain open, and the interrupted RULER-13 / pending 2Wiki retrofit
checks do not affect the current manuscript but bound what may still be
promoted before the deadline.

## 2. Coverage inventory

Read completely:

- `main.tex`; all nine body sections (`00_abstract` … `08_ai_use`)
- all six appendix files (`a1_proofs` … `a6_mature_scale`), including the
  full 613-line proof appendix
- all seven table files under `tables/`
- `SUBMISSION_CHECKLIST.md`, `CHANGES_FROM_NEURIPS2026.md`
- `HANDOFF.md`, `research/README.md`,
  `ICLR2027_RESEARCH_SYNTHESIS_20260819.md`,
  `audits/ICLR2027_NARRATIVE_REFLECTION_20260821.md`,
  `EXACT_RANGE_151M_3SEED_RESULT_20260820.json` (head)

Mechanical observations:

- compiled PDF: 30 pages total; body page limit enforced mechanically by
  `compile.sh` (last recorded pass); PDF built 2026-08-21 09:15
- `refs/references.bib`: 66 entries; citations concentrated in Related Work
  (24 `\citep/\citet` lines) plus intro/experiments/discussion
- figures actually referenced: exactly three
  (`fig_evidence_overview.pdf`, `fig_method_overview.pdf`,
  `fig_frequency_geometry.pdf`); twelve additional legacy/diagnostic PDFs
  sit unused in `figs/`, including `fig_lerope_profile_oracle.pdf`, which
  encodes an **internal-only** falsification result and should never reach
  the supplement package unreviewed
- `rope-spectral-budget-iclr2027-supplement.zip` exists at repository root,
  timestamped with the current PDF (2026-08-21 09:15)

## 3. Section-by-section findings

### 3.1 Abstract (~186 words)

Opens with the strongest quantitative hook (23 slow pairs, 46 nominal
dimensions, r2 = 2.00), states the decomposition, the construction, the
three-seed identification, the co-adaptation account, the 31.1% MLA flagship,
the persistence ladder, and correctly labels 8B as adaptation. Compliant with
all locked presentation rules (no Cosh trade-off sentence, no SOTA language).
Minor tension: it is dense — five distinct numeric claims before the reader
has the frame. Acceptable for ICLR; do not add further load.

### 3.2 Introduction

Six functional paragraphs now match the narrative plan (budget observation →
decomposition → three-seed answer → mechanism → construction →
consequences). The Figure 1 evidence-overview panel mapping is accurate
against the underlying owners. Contributions (i)–(iii) match the claim
architecture. Residual issue: paragraphs 2–4 still carry audit-grade
qualifiers ("Because the endpoints are identical, no scalar base can
reproduce…"), which are correct but heavy; the discovery would land harder
if one qualifier moved to its nearest claim boundary later in the section.

### 3.3 Related Work

The intervention-level taxonomy (transport / support / allocation) is clean
and non-misleading. LeRoPE is positioned exactly within its ceiling:
learned/fixed-table evidence, the 63.6% fixed-table retention quoted as
external primary-source fact, explicitly not mechanism validation. The
Evaluation paragraph correctly labels RULER endpoints as task-family-adapted
length transfer. No over-claim found. The DAPE/learned-table distinction is
stated twice (here and App. E) — slightly redundant but defensible.

### 3.4 Theory

- Budget identity (Thm 1): two-line proof, correct as written; the appendix
  properly disclaims Shannon-rank/log-det scope.
- Collapse proposition: fourth-order constant `19/12600` with a numerical
  ratio check (1.00058 at x=0.05, y=0.10); softmax-centred limit stated
  under a nondegeneracy condition. Sound.
- Co-adaptation (Table 1 / tab:coadapt): 2x2 frozen crossing with bootstrap
  CI on the interaction contrast [-5.17, -3.04]; the "static criteria do not
  predict trained quality" discipline sentence is present and load-bearing.
- Obstruction theorem: continuous-interval generator-similarity argument
  plus an integer-position one-step spectral variant (handles the 2π alias);
  scoped to fixed position-independent invertible Q/K maps with an explicit
  complementary-case sentence.
- Construction: convex surrogate → cosh density → midpoint quantiles →
  deployed rule; the surrogate is repeatedly labelled a modelling choice,
  and c=1 is labelled a zero-search operating prior. Fully compliant with
  the finite-tau ceiling.

Theory verdict: no defect found; boundary statements match the canonical
report's scopes.

### 3.5 Experiments

Protocol separation verified: fixed-support identification (151.9M, three
seeds, paired anchors), M4 factorial (50.9M, 12 configurations,
pre-specified), MLA flagship (432M, three seeds), range composition (454M,
three seeds), 750M continuation (seed 42), 1.485B from-initialisation,
1.485B matched Q/K adaptation, 8B matched LoRA, video DiT (two seeds). Each
subsection names its metric, control, and seed scope. Numbers cross-checked:

- exact-range contrasts (+0.026 / -0.281 / -0.176 / -0.146) match the
  raw-hash-receipted JSON owner to rounding
- 31.1% MLA reduction (138.8→95.6) consistent between abstract, §4.2,
  App. C, and Fig. 1b
- 41→61 / 53→100 retrieval composition matches tab:evq-ramp
- 45.1/24.4 @16K and 0%→77.5% strict AR match tab:750m
- 122/128 and 126/128 document counts match the App. F from-init paragraph
- 2Wiki token-F1 triple and the 8B deletion contrast (-0.0095/+1.5055)
  match App. F; NLL and autoregressive endpoints stay separate everywhere

Attack surfaces that remain (disclosed by design):

1. **Target-matched reversal:** under retargeting FMRoPE wins 3/3 seeds at
   every OOD length (App. Table). The text frames this as the expected
   interaction and keeps the fixed column as the identifier — correct per
   locked decisions, but a hostile reviewer reading Appendix E can construct
   a "method loses when support is chosen optimally" narrative; the
   Discussion pointer ("interact when chosen jointly") is currently the only
   defence.
2. **M4 rule-vs-Geo weakness:** 7/12 with p = 0.125 and a CI touching zero;
   the section wisely rests on family direction (best member 10/12, matched
   exponential 9/12) rather than rule superiority. Keep this framing intact
   under review pressure.
3. Mature rows are single-seed by construction; captions state seed scope.

### 3.6 Discussion

Five paragraphs, each mapped to one evidence layer; ends on the budget
thesis. No audit vocabulary, no negative-result ledger replay. This section
already meets the narrative standard the intro has not fully reached.

### 3.7 Ethics / Reproducibility / AI-use statements

All three present before the bibliography as required. The AI-use statement
matches the author-confirmed 2026-08-19 factual coverage and must not be
narrowed. The reproducibility statement accurately points to the supplement
(entrypoint, aggregate, figure generators, curated snapshots, CPU tests).

## 4. Appendix findings

- `a1_proofs` (613 lines): complete derivations for the cross-Gram, both
  theorems, the proposition, existence/uniqueness of the surrogate minimiser
  (direct method + strict convexity; nonnegativity inactive a posteriori),
  Wasserstein/histogram quantisation bounds with an honest vacuity caveat at
  large tau, the surrogate self-consistency identity (exact internal check,
  no trained-loss claim), operating-point scaling separating shape-selection
  from scale-selection, stiffness-sweep model sensitivity, and the chi-square
  load axiomatisation explicitly marked "not a uniqueness claim". This is a
  genuine asset; none of it should be cut for length.
- `a2_experiment_details`: hyperparameters, reproducibility snapshot, 750M
  and video-DiT details including the base-1000 diagnostic. Consistent with
  main text (tau = 1.5 video, tau = 1.414 MLA).
- `a3_supporting_results`: MLA table with per-cell std and the
  wavelength-blend operator named distinctly from YaRN-style — nomenclature
  compliant. States the +1.1% 8K cost here (correctly absorbed from the old
  abstract trade-off sentence).
- `a4_supporting_experiments`: L=256 composition study; correctly framed as
  "among the tested settings".
- `a5_identification`: the strongest provenance appendix — pinned-variable
  inventory, data shard revisions and tokenizer revision hashes, arm
  definitions, anchor-paired metric definition, per-seed table, M4 design and
  absolute values, boundary-tau arms explicitly denied basin/boundary status,
  WikiText-2 corpus difference disclosed, the YaRN-style operator fully
  specified, and the learned-comparator identity clarified (32-param learned
  table ≠ DAPE).
- `a6_mature_scale`: routing-conversion rows kept separate (0/100 vs 69/100
  replication vs independent 67/100 — never called seed variance), selective
  Q/K protocol, bootstrap CIs conditioned on the single trained pair with an
  explicit "not training-seed variability" disclaimer, from-init values
  including the adverse 2K direction (+0.0724), 8B probability/routing/
  deletion endpoints separated, and the endpoint-scope paragraph closing with
  task-family-adapted transfer. Exemplary.

## 5. Claim-ceiling compliance check

Checked against the `AGENTS.md` ceilings and locked decisions:

| Ceiling | Status |
| --- | --- |
| Static geometry ≠ LM-quality/extrapolation predictor | respected (§3.3 discipline paragraph) |
| Low-frequency collapse = redundancy in metric, not unused | respected ("may still carry content") |
| Retrofit obstruction limited to fixed invertible Q/K | respected (theorem scope + complement sentence) |
| Cosh unique only under stated surrogate | respected (repeatedly) |
| Finite tau = fallible zero-search prior; no basin claims | respected (App. E boundary arms; App. A multiplier language) |
| Exact-range owns pure allocation identification | respected (owner-routed, three training seeds, anchors not counted as seeds) |
| 8B = adaptation; trend stops at 1.485B | respected (abstract, §4.3, Discussion) |
| RULER/NIAH = task-family adaptation | respected (Related Work, App. F) |
| LeRoPE = related evidence, not validation | respected (Related Work, Discussion) |
| OLMo scratch = same init/same recipe, not bitwise | respected (App. F wording) |
| No cross-protocol variance splicing | respected (bootstrap conditioning disclaimer) |

No violation found. Two presentation-level risks remain (target-matched
reversal readability; M4 rule-contrast weakness), both currently inside
their allowed framing.

## 6. Venue validity status

From `SUBMISSION_CHECKLIST.md`, still open before upload:

1. Final submission-day build (clean dir, both passes, BibTeX, hash
   receipt, visual page review).
2. Author final number-vs-owner review after layout freeze.
3. NeurIPS contingency: if accepted (notification Sep 24), third-person
   citation and contribution-boundary statement before the Sep 25 upload.
4. OpenReview title/abstract equality with the final PDF.
5. Live deadline/policy recheck (checked 2026-08-19; must be repeated).

The distinctness audit is done and documented (8-word shingle overlap 1.02%
in the nine-page body; materially different claim/theory/evidence).
Dual-submission timing is permitted by ICLR policy per the 2026-08-19 check.

## 7. Ranked residual risks (decision-leverage order)

1. **Reading contract (score ceiling).** Strongest science, partially
   converted into ICLR-style conceptual force. Highest-leverage action
   remains a full-PDF narrative pass, especially Intro ¶2–4; no new evidence
   needed.
2. **Target-matched reversal surface (technical credibility).** Disclosed
   and framed, but defended in only one Discussion clause. Consider whether
   the fixed-column identification logic can be restated once more crisply
   near the App. E pointer — without promoting the reversal.
3. **Open checklist items (venue validity).** Five actions above; all cheap
   but mandatory.
4. **Supplement hygiene.** Twelve unused figure PDFs in `figs/`, including
   the internal-only LeRoPE oracle figure; the packager leak scan passed on
   2026-08-20, but re-verify after any further change that shipped `figs/`
   contents remain intentional.
5. **Interrupted retrofit confirmation (no current manuscript impact).**
   RULER-13 v3 output must be inspected on restart; partial directories must
   not be promoted; core-4 reuse requires exact cell identity. This gates
   any late promotion of the zero-training retrofit, which is currently NOT
   in the manuscript and should stay out unless completed cleanly.

## 8. Recommended sequence (no edit performed in this review)

1. Full-PDF visual/narrative pass targeting the reading contract only.
2. Complete or consciously drop the RULER-13 confirmation matrix per handoff
   rules (requires explicit authorisation if GPU time is needed).
3. Re-run `./compile.sh` and the iclr2027 packager profile; record receipts
   in the handoff.
4. Execute the five open checklist items in the last 72 hours before the
   deadline, ending with OpenReview title/abstract equality.

## 9. Figure visual inspection (2026-08-22 addendum)

Method: the three referenced figure PDFs were rendered to PNG (pdftoppm,
400 dpi for the flagged panel), the relevant `main.pdf` pages (2–5) were
rendered for in-context review, and `make_fig_evidence_overview.py` was read
against the curated MLA JSON owner
(`data/curated/table18_mla_3seed_aggregate.json`).

### 9.1 Verified correct

- **Figure 1(a)** per-seed grey lines, mean, and all four annotated contrasts
  (+0.026 / −0.28 / −0.18 / −0.15) match the raw-hash owner JSON exactly;
  the script asserts these values bit-for-bit. In-window +0.026 is honestly
  visible above zero.
- **Figure 1(c)** bar heights match the App. F triple
  (25.99/24.84, 0.07/21.48, 0/8.57); the callout ΔNLL −0.01→+1.51 rounds the
  owner's −0.0095/+1.5055 correctly; legend uses compliant `Native` /
  `EVQ-Cosh` nomenclature.
- **Figure 2(c)** heatmap reproduces Table 1 exactly (7.14 / 76.20 / 23.05 /
  7.16); caption matches.
- **Figure 3(a)** three-curve allocation-vs-range story matches §3.5;
  endpoint circles mark the pinned support correctly.
- **Figure 3(b)** white-to-blue sequential palette with the red slow-pair box
  (23 pairs) matches both the caption and the handoff palette decision;
  colorbar labelled c canonical redundancy [0,1].
- No anonymity leaks inside any figure; fonts render cleanly; in-context
  page renders show no overlap with body text or captions.

### 9.2 Defect found: Figure 1(c) occluded bar label

The 2x EVQ-Cosh bar's value label "21.5" is **fully occluded by the red
"8B causal deletion" callout box**. Five of six bars carry visible labels;
this one does not, and it is precisely the value that carries the caption's
claim ("carries real-document QA beyond the physical training window").
Root cause in `make_fig_evidence_overview.py`: bar labels draw at
`value + 0.6` (y ≈ 22.1 of ylim 33) while the callout sits at axes fraction
(0.98, 0.78) top-right, overlapping that region. Fix options: move the
callout to mid-right below y-fraction 0.6, raise ylim, or left-shift it.
This requires regenerating the PDF and recompiling — not done in this
read-only review.

### 9.3 Numeric inconsistency found: "+0.9%" vs "+1.1%"

Figure 1(b) annotates the 8K in-window PPL change as **+0.9%**; App. C text
states **+1.1%**. Verification against the curated JSON:

- underlying per-seed data are identical in both places (figure uses
  `progression["100%"]`, the table uses `extended`; same values);
- true quantity (mean over seeds of per-seed relative change, and equally
  the ratio of unrounded means 35.77/35.44) is **+0.93%**, i.e. +0.9%;
- the text's +1.1% comes from dividing *rounded* table entries
  (35.8/35.4 − 1 = +1.13%).

The figure is right; the appendix sentence is a rounding artifact and should
be corrected to +0.9% (or "~1%") at the next edit. Low stakes but
reviewer-visible once both numbers are compared.

### 9.4 Legibility margin

Panel (a)/(b) annotations use 5.8–6.2 pt source sizes ("3/3" markers, bar
labels, percentage tags). They are legible at 130 dpi full-page render and
in the compiled PDF at 100% zoom, but sit at the print-legibility floor;
recommend the author's final visual pass include a print-scale zoom on
Figure 1.

## 10. Limits of this review


- Static source analysis; no fresh compile was run (builds write artifacts,
  and this review is read-only by instruction).
- Owner verification was spot-checked at hash-receipt level for the
  exact-range JSON and the MLA curated aggregate, and consistency-checked
  elsewhere (cross-section, table-text, appendix-main agreement); a full
  per-number owner sweep exists historically
  (`CHANGES_FROM_NEURIPS2026.md` §7) but was not re-executed line by line
  here.
- All three referenced figures were rendered and visually inspected
  (§9); the twelve unused figure PDFs in `figs/` were not rendered, and the
  author visual-review checklist item remains necessary for the final PDF.
