# REVISION BRIEF v2 — positioning upgrade + panel-response revision (execution input for Codex — cycle closed 2026-08-28)

**Date:** 2026-08-27 (v2, post-audit) · **Status:** author-commissioned; supersedes no authority —
[`NARRATIVE_GUIDE.md`](NARRATIVE_GUIDE.md) still governs narrative discipline,
[`HANDOFF.md`](HANDOFF.md) governs live state, routed canonical owners govern every number.
**Inputs synthesized:** (1) the qwen five-seat panel (`research/external-reviews/qwen-panel-20260826/`,
decision: Major Revision, starred mandatory items in `06_editorial_decision.md` §5);
(2) the ICLR-2026 RoPE cohort (`~/Downloads/RoPE_Papers/Markdown/`);
(3) the author's restated core positioning (2026-08-27, recorded verbatim in §1 below).

**Cycle closed (2026-08-28):** this brief drove the ICLR 2027 revision cycle to completion — executed in four receipted passes logged in [`research/CODEX_CLAUDE_PAPER_REVIEW_LOG.md`](research/CODEX_CLAUDE_PAPER_REVIEW_LOG.md), outcome committed at 93d7eac.

This brief is a plan, not a manuscript. No science changes here; every number
must still be sourced from the routed canonical owner.

**v2 audit record:** this brief was cross-audited on 2026-08-27 by three
independent read-only auditors (factual accuracy / normative consistency /
Codex executability). 13 of 18 factual claims verified against primary files;
stale or unsupported items were corrected below and the corrections are logged
in §10. If a §-reference below conflicts with the compiled PDF, the compiled
numbering + file path is authoritative (§5 carries the mapping).

---

## 1. The author's core, codified (and one precision correction)

Author statement (2026-08-27): we identified a point almost nobody in RoPE
optimization attends to. A RoPE frequency decomposes into support/base and
exponent: with $x_k=-\log\omega_k=a+Rz_k$, a geometric table pins the exponent
sequence $z_k=k/(K{-}1)$, and every scalar-base or range method changes only
$(a,R)$ — the base — while carrying the exponents along. We isolate the exponent
coordinate $z$. We study extrapolation on $z$ and prove its effectiveness with
locked causal variables from three angles: zero-training, from-scratch
pretraining, and LoRA adaptation. Extrapolation is one use of $z$: LeRoPE-type
evidence shows in-window gains are also possible, and a method that wins both
in-window and out-of-window may exist — but that is others' related work and
our own unproven thinking. GPU budget limits experimental depth; breadth is
extensive. The core is variable isolation: the standalone effect of $z$ is
strong, plus synergy studies and three application routes. Theory and
experiment are nearly closed-loop. Goal: a 30-second skim convinces innovation,
usefulness, rigor, comprehensiveness → solid accept; use all 9 body pages.

**Precision correction (must be applied everywhere):** "all methods do not
change the exponent" is not literally true — YaRN ramps and LongRoPE's searched
schedules move per-channel exponents, and LeRoPE learns them. The defensible,
panel-verified novelty sentence is:

> Prior work either moves support and carries the exponents along (PI,
> NTK/base-scaling, YaRN, LongRoPE), or learns exponents without a closed form
> (LeRoPE/AdaRoPE). **No prior work isolates the exponent coordinate at fixed
> support, measures its causal effect, or derives it in closed form.**

Use exactly this structure. Never write "no one changes the exponent."

---

## 2. Target identity: hybrid coordinate-and-method paper

### 2.1 What the ICLR-2026 cohort rewards

From the seven cohort papers (PPE, RePo, Group Representational PE, Decoupling
Positional/Symbolic, Selective RoPE, Deconstructing Positional Information,
MrRoPE):

- Method papers win with a named, memorable construction and one-line identity
  (RePo, MrRoPE, PPE, Selective RoPE).
- Analysis papers win with a bold conceptual act, not a hedge (Decoupling…,
  Deconstructing…, Group Representational PE).
- Accepted colon titles take exactly two grammars (verified against the seven
  papers): **Acronym: Expansion** (PPE, RePo, MrRoPE) or a **From→To
  trajectory** ("Deconstructing X: From A to B"). Never "Hook: Claim Is True".
- Accepted analysis papers still lead with a *positive object* (a decomposition,
  a decoupling, a representation), not a measurement.

### 2.2 Why neither pure identity fits us

- **Pure analysis undersells:** we own a zero-learned-parameter closed-form
  allocation (EVQ-Cosh) with genuine multi-seed wins (151.9M fixed-support 3/3;
  MLA $K{=}16$ 3/3 seeds at 16K — per-protocol scope, §3 T4) and a zero-cost
  advantage over every rival family (LeRoPE: learned; YaRN: boundary
  heuristics; LongRoPE: combinatorial search). "Constructive witness" rhetoric
  buries this asset. (Do NOT quote a LongRoPE candidate count — the manuscript
  and bib carry none; verify any number against a routed source before use.)
- **Pure method overreaches:** the factorial rule point is not confirmatory
  (rule−Geo $p{=}0.125$, CI ∋ 0; matched exponential ties the rule at
  $p{=}0.836$; by the sign-flip $p$-criterion only the off-rule 1.25× arm is
  unadjusted-significant, $p{=}0.027$ — while the 95% bootstrap CIs of the
  0.75× and matched-exponential arms also exclude 0; report both criteria,
  per Table `m4` caption: $p$ = unadjusted exact sign-flip, CI = configuration
  bootstrap); no mature protocol uses the zero-search rule's τ; the 60.47%
  frozen headline is generated by the checkpoint-derived table, which is *not*
  EVQ-Cosh. A method-paper lead invites reviewers to strike exactly these
  points.

### 2.3 The hybrid positioning (target)

**A new finite-table coordinate, causally identified; and the first
zero-parameter closed-form method on it — with the shape-specificity boundary
stated by us, not discovered by reviewers.**

The paper leads with the coordinate and the causal design (strongest legs),
presents EVQ-Cosh as the constructive method that operationalizes the
coordinate, and reports the lifecycle breadth. Retire the word "witness"
everywhere (6 manuscript occurrences, verified: `00_abstract.tex:12`,
`01_intro.tex:44,77`, `03_theory.tex:9,187,231`); replace with "closed-form
construction" plus the explicit boundary sentence (§6, row M3). **This is an
authorized override — see §9 override items O1/O2; NARRATIVE_GUIDE and HANDOFF
witness mentions are amended in the same pass.**

---

## 3. Evidence hierarchy (canonical ranking — also becomes a manuscript table)

Every claim must cite its tier; routes never pool across tiers.

| Tier | Evidence | Unit & scope | What it licenses |
|---|---|---|---|
| **T0 — Exact geometry** | Budget identity (Thm 1), collapse law (Prop 2), transplant obstruction (Thm 3), co-adaptation crossing | theorems | Diagnosis of the available basis. Never cited as a behavioural predictor. |
| **T1 — Pure-$z$ causal anchors** | (a) 151.9M fixed-support, 3 training seeds; (b) frozen same-support OLMo/Qwen controls | seeds / deterministic intervention | **The causal claim.** Everything fixed except $z$: support, $K$, operator, gain, routing, data, rows, decoder. (a) identifies $z$ during co-adaptation; (b) after co-adaptation. |
| **T2 — Persistence & synergy routes** | Matched LoRA on the mature OLMo checkpoint (+8B causal probe), 750M continuation | single trained pairs/trajectories | $z$ remains actionable combined with adaptation; per the synergy contract these are combined interventions, not additional pure-$z$ identification. |
| **T3 — Construction specificity** | 50.9M factorial; staged 99-run rule study | configurations | Shape *family* confirmed; Cosh *shape* and rule point directional only — stated explicitly in body (dual criteria: sign-flip $p$ vs bootstrap CI). |
| **T4 — Breadth** | MLA $K{=}16$ (EVQ vs Geo at fixed support, 16K = 2× OOD, 3/3 seeds; per-protocol EVQ deployment evidence at 16K only — no claim past 16K or against the Geo+wavelength-blend overlay arm); video DiT (1 seed, coordinate-breadth scope check) | per protocol | Architecture/modality breadth; each protocol separately scoped. |

**Corollary for the prose:** "the standalone effect of $z$ is strong" is a
T1 sentence. Synergy and lifecycle results are T2. Never let a T2/T4 result
carry a T1 sentence, and never let T0 predict behaviour. The MLA 16K result may
be cited as method deployment evidence **only with its protocol scope attached**
(16K, EVQ vs Geo, fixed architecture; τ=1.414 is an empirical $d_{eff}{=}128$
operating convention per the curated record, not a theorem).

**Depth-limit honesty (one sentence, compiled §5 closing or §6):** computational
budget caps per-protocol depth (several routes are single-trajectory); the
design choice was breadth across intervention stages, architectures, and
modalities, with seed replication concentrated where causal identification
lives (T1). This converts the GPU limitation from an apology into a stated
design logic.

---

## 4. The 30-second skim (innovation · usefulness · rigor · comprehensiveness)

The skim path is: **title → abstract → Fig. 1 → Contributions.** Each must land
all four signals without reading a single table.

### 4.1 Title — DECISION POINT for the author (default: option 2)

Cohort-calibrated options (both legal colon grammars respected; no "Hook:
Claim" form):

1. **`One Rotary Budget, Many Positional Bases`** — assertive, covers geometry +
   allocation consequence, memorable. (Editor's preference.)
2. **`RoPE Has a Spectral Budget`** — keep current; already cohort-consistent
   assertive style; cheapest (ecosystem branding preserved). **DEFAULT: proceed
   with this option unless the author responds.**
3. **`Decoupling Support and Exponent in Rotary Position Embeddings`** — most
   cohort-grammatical; names the coordinate directly; loses the budget image.
4. Method-forward variant (only if the author funds the §8 F-track extension):
   add a From→To trajectory subtitle, e.g. current title +
   `: From Collapsed Dimensions to Zero-Parameter Allocation`.

Considered and rejected for the lead: an Acronym: Expansion colon title (e.g.
"EVQ-Cosh: …") is cohort-legal but over-weights the method arm against §2.2.

Whichever is chosen must remain true of the *scoped* abstract (no governance
reading of "budget").

**Blast radius of any title change (list all, do not improvise):** `\title` +
header comment in `main.tex`; HANDOFF §2 contract line; NARRATIVE_GUIDE
reviewer-memory sentence; label `fig:spectral-budget-overview`
(`01_intro.tex`); `figs/fig_spectral_budget_scaling.{pdf,py}`;
`SUPPLEMENT_README.md`; supplement archive filename generated at
`scripts/package_supplement.py` (the ZIP name is branding-locked — a title
change implies a renamed archive, note it in the receipt).

### 4.2 Abstract skeleton (≤165 words; ONE number group; frozen-first order)

Rules: (i) identifiers (151.9M, MLA, $K{=}16$) are labels, not statistics;
(ii) the single number group is the paired frozen anchor defined in S3; (iii)
the mature frozen result is presented before the paired-training result (locked
presentation order, `research/CODEX_CLAUDE_PAPER_REVIEW_LOG.md`); (iv) this
number-group definition replaces the current HANDOFF abstract receipt (override
O3).

- **S1 (innovation):** the exponent coordinate nobody isolates — geometric RoPE
  fixes $z$; base/range methods move support and carry $z$ along; at fixed
  support $z$ is a separate design coordinate.
- **S2 (rigor, qualitative):** phase-invariant exact geometry: a fixed spectral
  budget whose nominal dimensions can collapse onto nearly one effective basis
  (supply-side statement; behaviour is settled by training — the defusing
  clause for ★A5 lives here or in the §5.1 body pointer).
- **S3 (rigor + causal effect, scoped; carries THE number group):** pure-$z$
  causal identification, frozen-first: the mature frozen fixed-support
  intervention raises RULER 0.56% → 60.47%, calibrated against the YaRN
  in-window anchor 7.94% — **this pair is the abstract's one number group**
  (satisfies ★A2's dual-anchor requirement); subordinate clause: the
  three-seed 151.9M from-training intervention improves every tested OOD length
  in all seeds, *with the one-clause scope*: per-length support retargeting
  reverses the ordering (the fixed-support result is the one-table-many-lengths
  deployment case). This clause is mandatory (panel DA-1).
- **S4 (usefulness, method-forward):** EVQ-Cosh: zero-learned-parameter,
  closed-form allocation; wins at fixed support (labels: 151.9M; MLA at 16K),
  deployed across zero-training / matched-adaptation / from-training routes;
  shape-specificity honestly bounded ("one closed-form point of a confirmed
  deformation family").
- **S5 (comprehensiveness, scoped):** breadth across lifecycle routes,
  architectures, and a video modality, in the tested protocols; in-window
  improvement is related work's evidence (LeRoPE), ours is OOD-anchored.

Word-budget guidance: the current abstract is 159 words; S3's paired anchors +
scope clause are the expansion risk — compress S1/S2/S5, not S3's anchors.

### 4.3 Fig. 1 spec (three panels, one row)

- **(a) The coordinate:** $x_k = a + Rz_k$ decomposition; base/range methods vs
  exponent moves; "fixed support + different $z$ ⇒ different basis".
- **(b) The geometry:** collapse picture; 23 slow pairs, $r_2 = 2.00$ under the
  stated prior (label the prior in the caption). Current panel (b) already
  matches — keep.
- **(c) The causal design + routes:** pure-$z$ contrast diagram (all knobs
  locked, 30 interiors move) with the three-seed win and the three routes
  (zero-training / adaptation / from-training) as labelled lanes.
- Caption carries: the scope clause (retargeting reversal), the prior, seed
  counts, and the MLA 16K protocol scope if MLA appears. A reader who reads
  *only* the caption must get the honest version.

**Scope gate:** panel (a)/(c) changes require regenerating
`figs/fig_evidence_overview.pdf` (generator:
`figs/make_fig_evidence_overview.py`). Caption/label-level edits are Codex
scope; a panel redesign (new decomposition diagram, route lanes) is §8 decision
point 5 — do not restore the old multi-protocol montage (HANDOFF guard).

### 4.4 Contributions list (rewrite; four bullets aligned to the four signals)

- **(i) Innovation+Rigor:** expose the exponent coordinate $z$; causally
  identify its behavioural effect via pure-$z$ contrasts (T1 anchors), fixing
  every scalar-base degree of freedom.
- **(ii) Rigor:** exact phase-invariant geometry — budget identity, collapse
  law, co-adaptation boundary, transplant obstruction (T0).
- **(iii) Usefulness:** EVQ-Cosh as a zero-parameter closed-form construction;
  wins at fixed support in multi-seed protocols (MLA cited only at 16K scope);
  shape-specificity boundary stated (T3); zero-cost against
  learned/searched/heuristic rivals.
- **(iv) Comprehensiveness:** three protocol-separated lifecycle routes, MLA
  architecture, video modality (T2/T4); synergy contract stated.

---

## 5. Nine-page budget (body currently 8 — use the full 9)

All additions are gap-driven (panel items / author's evidence hierarchy) —
this satisfies the HANDOFF no-page-filler rule; nothing is added for volume.

**Compiled section map (authoritative for every anchor below):** §1 Intro
(`01_intro`) · §2 Identifying the Allocation Axis, §2.1 Fixed-support training
intervention (`02_identification`) · §3 Related Work (`02_related`) · §4
Theory, §4.1–§4.4, Thm 1 at §4.2 (`03_theory`) · §5 Consequences across
Model-Building Regimes, §5.1 frozen / §5.2 matched LoRA / §5.3 from-training
(`04_experiments`) · §6 Discussion (`05_discussion`). NOTE: earlier panel
documents cite filenames as section numbers (their "§4.1" = `04_experiments`);
every anchor below uses compiled numbers + file paths.

| Space | Add | Closes panel item |
|---|---|---|
| Compiled §3 Related Work (`02_related.tex`, +~0.35p) | **Zero-cost comparison table**: rows = base/range (NTK-aware, PI, YaRN, LongRoPE), learned-$z$ (LeRoPE, AdaRoPE), ours; columns = moves support / moves exponents / isolates at fixed support / closed form / search or training cost. LongRoPE cost cell: "combinatorial search" — no candidate count unless verified against a routed source. Add the classical-lineage sentence (frame potential / Welch bound / principal angles / Landau density / quantization distortion) at the point where allocation geometry first meets coding-theoretic language (★B2). Add the NTK-aware/dynamic-NTK paragraph (zero occurrences currently in compiled text) and reclassify YaRN/LongRoPE as allocation + support transport. Cite `kazemnejad2023impact` (in bib, never cited). | ★B1, ★B2, B3, DA-12 |
| Compiled §5 intro (`04_experiments.tex:13`, +~0.3p) | **Evidence-hierarchy table** (§3 above) — one glance gives reviewers the rigor map and the depth-breadth design logic. | panel C5/DA-8 scope, author's 证据等级 |
| Compiled §2.1 (`02_identification.tex`, +1 sentence) | Retargeting-reversal disclosure (abstract S3 clause mirrored here). Authorized ★A1 exception to the appendix-only rule — see §9 O2. | ★A1 |
| Compiled §5.1 (`04_experiments.tex`, +2 sentences) | Dual-anchor calibration: 0.56 matched-support control and YaRN 7.94 — **both values already sit in this subsection's table rows (`04_experiments.tex:48,50`); reference them, do not duplicate**. In-window tax per model from Table `ruler` 1× column (compute and cite per-model values: OLMo ≈ 30–45 points, LLaMA-8B ≈ 17 — do NOT reuse the panel's "30–45" blanket range, it is OLMo-only), plus the two-table routing requirement, named in body, not only appendix. | ★A2, DA-10 |
| Compiled §2.1/§5.1 (+~0.15p) | Shape-specificity paragraph: rule point directional ($p{=}0.125$, CI ∋ 0), family confirmed, boundary ours to state; report the dual criteria (sign-flip $p$ vs bootstrap CI, per Table `m4` caption); confirmatory-vs-exploratory contrast declaration. | ★A2, ★D6 |
| Compiled §6 Discussion (`05_discussion.tex`, +2 sentences) | In-window gains = LeRoPE's related-work evidence; "both-window" methods = explicitly unproven future direction; depth-limit design-logic sentence (§3 corollary). | DA-10, author's 内外兼得 policy |
| Compiled §4.2, after Thm 1 (`03_theory.tex:99`) | keep Codex's title-interpretation sentence (already added 08-26). | A5 (partially absorbed) |

**Authorized compression targets** (if the additions push past 9 pages; do not
cut elsewhere without author direction): the pre-2024 historical prose in
`02_related.tex:10–45`; protocol-repetition detail in compiled §5.1 that
duplicates §2.1; categories per NARRATIVE_GUIDE "repeated history, reviewer
rebuttal framing, metric ledgers, protocol detail". Appendix theory stays.

**Zero-space fixes** (each owner-routed; none may invent a number):

- Expand EVQ at first use (macro `main.tex:67` currently renders only
  "EVQ-Cosh"; `math_commands.tex` is never `\input` — dead code, ignore).
- Define the **split rule** in body/appendix from the canonical owner:
  `research/attention-aware-retrofit/results/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md`
  (label-free model-relative projection; OLMo pairs 20→22, Qwen 28→31).
- Per-protocol **τ provenance** footnote/table in the appendix: the
  $(c, L, \text{grid})$ triple, rule-generated vs fixed, and per-protocol
  deviations. Sources: `../rebuttal/rebuttal_0723/theory_results/TAU_TRUE_ROLE_AND_OPERATING_RULE_AUDIT.md`,
  `appendix/a1_proofs.tex:97`, `appendix/a3_supporting_results.tex:15`. Note:
  the 0.53× video-DiT deviation is in the manuscript
  (`a2_experiment_details.tex:96–97`); the OLMo τ=2 vs rule τ≈1 deviation is
  derived — show it as computed, not quoted.
- **Nine-task list + selection rule + freeze artifact** in App.: owners =
  `evidence/METHOD_SELECTION_LEDGER_20260823.json` (ruler_core4 = method
  selection; ruler_unseen9 = post-freeze confirmation) +
  `evidence/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULTS_20260823.json` (per-arm
  `results_sha256`); the nine task names = intersection with Table `ruler`
  (13-family per-task scores already in the current PDF). Include the D1
  per-task scores for all 13 RULER families + the Qwen four, and the D2
  reviewer-runnable recipe that regenerates 60.47/61.04.
- ★A4 scope labels: per-protocol seed labels; the sign-test $p$; the Qwen
  derived−log-linear interval $[0.25, 17.50]$ — all in the appendix near their
  tables.
- E2: forward pointer, arm-dictionary box, classification-sentence move, Eq-2
  label. E4: remove the ICML fallback block (`main.tex:19–21` header comment +
  `venue_icml_fallback/` directory per panel E4).
- Orphan artifacts: `tables/table_evq_ramp.tex` and
  `appendix/a4_supporting_experiments.tex` exist but are never `\input` —
  include with definitions or remove from the release archive.
- Bib: prune or engage the 32 uncited entries (exact count re-verified
  2026-08-27) — **run the prune AFTER the new comparison-table cites land**, or
  the new cites' targets get swept.
- REMOVED (stale, verified 2026-08-27): the panel's dangling-ref items
  (App. B.7; Table 10 wavelength-blend row) no longer exist — current source
  and build contain neither, zero undefined references. Do not chase them.

**Author-input items (Codex must NOT invent these — surface them in the
completion report instead):**

- The **Native causal-distance measure** (referenced once, `03_theory.tex:47`,
  defined nowhere in the repo; the nearest analysis note self-disclaims as
  non-manuscript material). Either the author supplies the definition or the
  reference is deleted.
- Title choice (§8.1), F-track compute (§8.2), Fig. 1 redesign (§8.5),
  OLMo parameter-count verification (§8.4).

---

## 6. Claim-language ledger (replaces the corresponding NARRATIVE_GUIDE claim
prose on adoption — override O1; the guide's claim rules are prose bullets, not
table rows)

| # | Claim | Level | Allowed formulation |
|---|---|---|---|
| M1 | Coordinate | STRONG | "$z$ is a third independent design coordinate; no prior work isolates it at fixed support or derives it in closed form." |
| M2 | Pure-$z$ effect | STRONG, scope-carrying | "At fixed support, changing only $z$ changes trained behaviour" + the retargeting-reversal clause whenever the direction is mentioned. |
| M3 | Method family | STRONG | "Non-uniform interior allocation is a confirmed, actionable deformation family across zero-training, adaptation, and from-training routes." |
| M4 | Cosh shape | DIRECTIONAL ONLY | "One closed-form point of the family; shape-specificity unresolved by the factorial (rule point $p{=}0.125$); by the unadjusted sign-flip criterion the 1.25× arm is the only $p{<}0.05$ contrast, while 0.75× and matched-exponential 95% CIs also exclude 0 — directionally consistent family, unresolved shape." Never "optimal", never "the shape". |
| M5 | Rule τ | DIRECTIONAL | "A zero-search prior validated directionally (7/9 configuration means)"; list per-protocol deviations (owner-routed, §5); "zero-learned-parameter" scoped to gradients, not protocol design. |
| M6 | In-window gains | RELATED WORK | Attribute to LeRoPE; our protocols are OOD-anchored; both-window methods = unproven future work. |
| M7 | Depth | HONEST-DESIGN | §3 corollary sentence; single-trajectory labels carried into body prose. |
| F1–F7 | Forbidden | — | SOTA / globally or near-optimal / necessary or sufficient / "we beat FMRoPE/YaRN" / static rank as monotone performance predictor / slow bands as dead or reclaimable / the mature LoRA protocol as converged or multi-seed / 50.9M→mature as a scaling law / presenting LeRoPE evidence as validation of EVQ / uniqueness beyond "unique only for the stated convex surrogate" / calling the frozen derived or ramp profiles EVQ-Cosh. |

---

## 7. Panel mandatory items → resolution map

Each starred row inherits the FULL sub-requirement list from
`06_editorial_decision.md` §5; the right column names landing sites, not the
complete spec — Codex executes every sub-requirement of each row it touches
(including: A2's "controlled zero-training instantiation" relabel; B1's
FMRoPE-identity control-arm relabel; B3's three-HoPE disambiguation; C1's
true-$\bar c(z;\tau)$ τ-sweep + argmin displacement; D2's reproducible recipe;
D3's $L_{train}$ tie-break).

| Panel item (06_editorial_decision §5) | Resolved by |
|---|---|
| ★A1 reversal disclosure | §4.2 S3, §4.3 caption, §5 row 3 (authorized exception O2) |
| ★A2 dual anchors + rescoping + deployment label | §4.2 S3 (paired number group in the abstract), §5 row 4, §6 M3/M4/M5 |
| ★A4 scope labels in body | §5 zero-space fixes (seed labels, sign-test $p$, Qwen interval $[0.25,17.50]$, τ table) |
| ★A5 budget-metaphor clause | Codex's Thm-1 sentence (kept) + §4.2 S2 defusing clause ("budget bounds supply; behaviour settled by training") |
| ★B1–B3 positioning/literature | §5 row 1 (comparison table + lineage sentence + NTK paragraph + kazemnejad cite) |
| ★C1 phase-invariant surrogate validation + C2 three-prior sensitivity table | **both required by the decision letter** — CPU-only re-run of the surrogate AND the three-prior sensitivity table (existing closed forms; no GPU); appendix table; if ordering changes, report it (the paper's standards license negative results) |
| ★D1 nine-task documentation | §5 zero-space fixes (per-task scores for 13 RULER families + Qwen four) |
| ★D2 split rule + distance measure | §5 zero-space fixes (split rule owner-routed; distance measure → author input) |
| ★D3 τ provenance | §5 zero-space fixes |
| ★D6 multiplicity | §5 row 5 (dual criteria) |
| E1–E4 hygiene | §5 zero-space fixes (EVQ expansion; bib prune after new cites; orphans; E2/E4 enumerated) |
| F-track (optional experiments) | author decision points (§8); recommended: one **rule-fixed-τ mature run** as an F-track extension (new experiment, author-gated; not a listed F-item in the decision) — converts M4/M5 from directional toward confirmatory |
| Deferred/declined (recorded, not silent) | C3 prediction test — deferred, needs new compute; C4 coadapt caption — absorbed by §4.3 "label the prior in the caption"; D4/D5 presentation minors — folded into the E-hygiene pass; D7 T2 persistence framing — addressed by §3 T2 labels |

Author protections for any rebuttal (editorial-verified, do not concede):
Prop. 2 constant is correct (R2's factor-2 charge refuted); the DA-quoted
"rules out" sentence does not exist in the manuscript; co-adapt $r_2$ values
are a caption-pinning issue, not a demonstrated error.

**Erratum to the earlier delta note** (`07_post_review_delta_note.md`): the
HEAD abstract's "frozen 1.485B OLMo checkpoint" was CONSISTENT with the
canonical record — `data/curated/frozen_fixed_support_mature_20260823.json`
records checkpoint OLMo-2-0425-1B-Instruct with `parameters: 1485000000`, and
`a6_mature_scale.tex:6,13` agrees. The "factual error" framing is withdrawn:
Codex's "billion-scale" was a clarity choice, not a correction. The naming-vs-
count tension ("1B" name, 1.485B recorded count) is now §8 decision point 4.

---

## 8. DECISION POINTS for the author (Codex must not resolve these)

1. **Title** — §4.1 options 1–4; default option 2 (keep current) for
   single-pass execution.
2. **F-track compute** — at minimum recommended: one rule-fixed-τ mature run
   with known $L_{train}$ (F-track extension, new experiment, author-gated;
   upgrades M5); optionally the pre-registered shape contrast at larger
   configuration population (upgrades M4).
3. **In-window ambition** — keep M6 as written (LeRoPE-attributed, ours
   unproven), or authorize a small in-window arm later; no in-window claim
   enters this revision without data.
4. **OLMo frozen-checkpoint parameter count** — the model is named
   OLMo-2-0425-1B-Instruct but the canonical record and `a6` both state 1.485B
   parameters. Verify against the HF model card; if the true count differs,
   align the manuscript (`a6:6,13`), the curated record, and every "1.485B"
   label in one pass. Codex flags, does not decide.
5. **Fig. 1 panel (a)/(c) redesign** — caption/label edits are Codex scope;
   new decomposition diagram and route lanes = figure-regeneration work the
   author must authorize.

## 9. Execution constraints for Codex

- **Read before editing:** NARRATIVE_GUIDE + HANDOFF + this brief (and the
  review-log append-only workflow in HANDOFF); on conflict, canonical owner >
  HANDOFF > NARRATIVE_GUIDE > this brief.
- Never compile or modify the immutable `paper/` baseline; work only in
  `paper-2027/`.
- No number, metric, or protocol identity changes without the routed owner.
- Keep the three evidence routes un-pooled; keep FMRoPE a control, never a
  narrative antagonist; do not restart the FMRoPE body-prose loop.
- **Authorized overrides (amend the governing docs in the same pass, then log
  in HANDOFF):** O1 retire "witness" — supersedes NARRATIVE_GUIDE witness
  prose (:33, :71–74, :120–124, :150) and HANDOFF reader path (:69); this brief
  outranks the guide on this term only. O2 the ★A1 retargeting-reversal clause
  in the abstract and compiled §2.1 is a panel-mandated disclosure, an
  authorized exception to "the reversal lives in the appendix with body
  pointers" and to HANDOFF's "must not become the body narrative" (one clause +
  one sentence, not a narrative loop). O3 the abstract's paired-anchor number
  group replaces the current HANDOFF abstract receipt.
- Verify with `compile.sh` (≤9 body pages via the `page:bodyend` check, zero
  undefined refs/citations, overfull <5pt, anonymous, fonts embedded; note the
  TODO/FIXME/TBD gate — unresolved author items cannot be parked as source
  markers; surface them in the completion report instead); update HANDOFF
  receipts (append-only review log); no commit/push/upload/GPU unless the
  author explicitly directs.

---

## 10. Audit log (v1 → v2 corrections, 2026-08-27)

Three independent read-only audits (facts / consistency / executability) ran
against this brief. Accepted corrections:

- **Stale item removed:** dangling refs (App. B.7; Table 10 blend row) — the
  current build has neither; zero undefined references (2-auditor confirmation).
- **Cohort grammar corrected:** colon titles = Acronym: Expansion OR From→To
  (PPE/RePo/MrRoPE are counterexamples to v1's "From→To only" claim).
- **Unsupported numbers removed:** "LongRoPE ~48K candidates" (no repo source);
  "30–45 RULER-point in-window tax" replaced by per-model Table `ruler` values
  (LLaMA-8B ≈ 17 is outside the old range); τ-deviation range must be
  owner-routed per protocol.
- **Dual statistical criteria added** to M4/shape-specificity (sign-flip $p$ vs
  bootstrap CI, per Table `m4` caption).
- **MLA scope fixed:** T4 contradiction resolved — MLA 3/3 is a per-protocol
  EVQ deployment win at 16K only (not past 16K, not vs the blend overlay);
  DiT remains coordinate breadth.
- **Compiled section numbers** replace filename numbering throughout §5
  (every v1 anchor from §2 onward was off by one vs the compiled PDF).
- **Panel gaps closed:** C2 three-prior sensitivity table added (letter
  mandates it); ★A2 dual anchors routed into the abstract's number group;
  ★B2 lineage sentence, ★A4 sub-items, E2/E4 enumerated; "F2" mislabel fixed
  (rule-fixed-τ run is an F-track extension, not a listed F-item); deferred
  items recorded, not silent.
- **Executability:** title default + blast radius; page-budget compression
  targets; split-rule owner path; nine-task owner paths; bib-prune ordering;
  Native causal-distance measure moved to author-input; compile.sh marker gate
  noted; abstract skeleton restructured (frozen-first, ≤165 words).
- **Erratum:** the delta note's "HEAD 1.485B was factually wrong" withdrawn —
  the canonical record itself states 1.485B for the frozen OLMo checkpoint
  (naming-vs-count verification is now §8.4).
