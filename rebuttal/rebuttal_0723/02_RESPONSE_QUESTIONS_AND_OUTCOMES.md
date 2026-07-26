# Response questions inventory and final outcomes — Submission 11628

**Authority:** `00_REVIEWER_SCORES_AND_AC_METAREVIEW.md` only
**Evidence routing:** `01_REBUTTAL_PLAYBOOK.md`, `theory_results/EXPERIMENT_REPORT_20260724.md`, mature-model owners, Desktop drafts
**Date:** 2026-07-26
**Mode:** inventory + honest outcome (not a paste package)
**Principles:** (P1) answer named concerns only; (P2) no GPU invent-work for gaps already closed offline

This file lists **every concern we must answer** in the retained OpenReview panel,
states **what response we already have**, and judges the **final sendable result**
per item. It does not invent new experiments or claim score changes.

---

## Score board (baseline)

| Source | Overall | Confidence | Role in flip path |
| --- | ---: | ---: | --- |
| `Dz6s` | **4** Borderline accept | 3 | Protect 4; soft aim 5 |
| `27bE` | **3** Borderline reject | 4 | Highest ROI: 3→4 |
| `zWsa` | **2** Reject | 5 | Hard; aim 2→3 via dual-knob identity |
| AC `XLtL` | Metareview only | — | Recommendation moves only if novelty + control + stronger eval |

---

## 1. Area Chair `XLtL` — four decision gates

AC does not score; it states what would **materially change the recommendation**.

### 1.1 `AC.1` — Novelty vs FMRoPE / dead channels

| Field | Content |
| --- | --- |
| **Ask** | Technical distinction from FMRoPE and prior dead-frequency observations; related-work positioning; matched comparison. |
| **Existing answer** | Concede missing Oka citation. **Identity:** FMRoPE = geometric range/base for a declared context; EVQ = fixed nominal base, closed-form inverse-CDF **interior allocation**. Dead channels = supporting intuition, not the main claim. **Direct control:** fixed matched-range Cosh wins the seed-42 interior-allocation comparison; target-retargeted FMRoPE wins under the target-aware deployment contract. Seed-42 is sendable; the 3-seed aggregate remains CONDITIONAL until raw promotion. |
| **Best numbers** | Seed-42 exact-range (fixed): Cosh−FMR NLL **−0.478 / −0.205 / −0.113** at 512/1K/2K. Target-retargeted: **+0.061 / +0.182 / +0.279** (FMR stronger). Primary I still supports substrate×range complementarity. |
| **Gap** | Missing citation is real. Not universal win over FMRoPE. 3-seed not yet sendable as aggregate. |
| **Outcome** | **ANSWERABLE NOW** for “different method + complementarity.” **Not** “EVQ dominates FMRoPE.” |

### 1.2 `AC.2` — Empirical validation too small / diagnostic-heavy

| Field | Content |
| --- | --- |
| **Ask** | Stronger scale, stronger benchmarks, base sensitivity, real tasks — not only small-model diagnostics. |
| **Existing answer** | Submitted Primary I–III package + **OLMo-2 1.485B** CF AR/NIAH + 13-task family RULER + **matched LLaMA-8B** NLL/RULER + submitted Appendix D, Table 23 8B anchor + the raw-hash-backed 1.485B step-0→1,000 run. The M4 exact-range factorial adds two-base, three-head-dimension mechanistic coverage. Endpoint taxonomy: NLL ≠ AR ≠ RULER ≠ QA. |
| **Best numbers** | OLMo 8K strict NIAH **0/100 vs 69/100, 67/100**; LLaMA 16K RULER **0.295% vs 14.03%**; OLMo RULER **37.5 / 21.3 / 6.1%** at 4K/8K/16K. |
| **Gap** | Task-family adapted (not clean unseen-task). 4× weak/zero. Multi-seed controlled pretrain still strongest **below 1B**. No broad instruction-following / production suite. |
| **Outcome** | **PARTIALLY CLOSED.** Enough for “not only 454M PPL toys”; **not** “production generality.” |

### 1.3 `AC.3` — Surrogate → Cosh → operating rule; shape vs tuning

| Field | Content |
| --- | --- |
| **Ask** | Separate exact theory, modeling choices, empirical τ, matched analytic schedules; disentangle allocation from parameterization. |
| **Existing answer** | Four epistemic layers (Table-1 discipline). Independent τ sweep (selected 5; rule 5.657 within **0.0119 NLL**). The three-seed M4 factorial fixes sampled extrema/log-span across two bases, two training lengths, and three head dimensions while comparing Cosh multipliers and a deformation-matched exponential. |
| **Best numbers** | Formula-Cosh/\(1.25\times\)-Cosh/matched-exponential minus Geo weighted OOD NLL **−0.009879/−0.012100/−0.010619**; \(1.25\times\) Cosh favors 10/12 configurations. |
| **Gap** | Small-τ asymptotics ≠ τ≈4 theorem. Rule is fallible basin prior. |
| **Outcome** | **ANSWERABLE NOW** if claims stay narrow (allocation axis + useful default, not global Cosh optimum). |

### 1.4 `AC.4` — What can change the recommendation

| Field | Content |
| --- | --- |
| **Ask** | Priority order: (i) clear novelty over FMRoPE, (ii) controlled comparison, (iii) stronger evaluation. Else state limits. |
| **Existing answer** | Same three blocks as 1.1–1.3 in that order. No fake “all solved.” |
| **Outcome** | **Defensible package exists.** Flip not guaranteed; package is **triage-complete for response**, not **acceptance-complete**. |

---

## 2. Reviewer `Dz6s` (rating 4) — three concerns

### 2.1 `RDz6s.1` — Evidence too narrow / diagnostic endpoints

| Field | Content |
| --- | --- |
| **Ask** | Synthetic / PE-dom / NLL-gap retrieval do not show real-world long-context applicability; want mature models and stronger tasks. |
| **Existing answer** | Agree diagnostic vs capability separation. Add OLMo strict AR NIAH + LLaMA/OLMo RULER + mature NLL. Keep Primary I as **mechanism complementarity**, not capability claim. |
| **Gap** | No broad multi-downstream / chat / instruction suite. 4× open. |
| **Outcome** | **Protect the 4.** Can soften toward stronger accept if AR/RULER are clean and limited. Overclaim → risk of **losing** the 4. |

### 2.2 `RDz6s.2` — Matched YaRN not conclusive vs optimized Geo+YaRN

| Field | Content |
| --- | --- |
| **Ask** | Matched scale shows complementarity; does not rule out better Geo+YaRN / channel search closing the gap. |
| **Existing answer** | **Concede fully.** Claim only: same fixed repository “YaRN” scaler, scale \(s=8\), substrate changes leverage. Target-aware FMR/YaRN can beat raw EVQ. No official tuned-YaRN redo. |
| **Outcome** | **ANSWERABLE BY NARROWING** — already Dz6s-aligned if we do not overclaim. |

### 2.3 `RDz6s.3` — Theory–practice layers

| Field | Content |
| --- | --- |
| **Ask** | Separate (i) surrogate math, (ii) exact-kernel checks, (iii) post-training results. |
| **Existing answer** | Same as `AC.3` / `R27bE.1`: Table-1 + τ + schedules. |
| **Outcome** | **ANSWERABLE NOW.** |

---

## 3. Reviewer `zWsa` (rating 2, conf 5) — four explicit score-move questions

zWsa states **when score would increase**. Answer these in order.

### 3.1 `RzWsa.1` — Novelty vs Oka et al. / FMRoPE

| Field | Content |
| --- | --- |
| **Ask** | Clear technical novelty of EVQ over FMRoPE; missing citation and dead-channel overlap. |
| **Existing answer** | Cite Oka. Dead channels not claimed novel. Construction/stage/object differ (range vs interior allocation). AdamW/YaRN standard for related-theme ≠ same method. |
| **Outcome** | **Technically defendable.** Confidence 5 makes a full flip difficult; use the opposite outcomes to define the tested parameterizations, not to claim “we win FMR.” |

### 3.2 `RzWsa.2` — Direct matched FMRoPE comparison

| Field | Content |
| --- | --- |
| **Ask** | Matched settings; **advantages or complementarity** both raise score. |
| **Existing answer** | Dual result meets **complementarity** criterion. The seed-42 study is the direct target-aware FMRoPE control; the three-seed M4 factorial strengthens fixed-range allocation identification across base/length/head settings. Deployment: retargeted FMR remains stronger. |
| **Outcome** | **Meet stated bar for complementarity.** Do not claim pure dominance. |

### 3.3 `RzWsa.3` — RULER (or effective context)

| Field | Content |
| --- | --- |
| **Ask** | Include RULER even at small scale; score increases if method improves RULER. |
| **Existing answer** | OLMo 13-task family RULER + LLaMA matched 13-family RULER. Protocol: task-family adapted; 4× weak. |
| **Outcome** | **Satisfies “include RULER.”** Partial on “improves” (2× yes under protocol; 4× no; 8K LLaMA Native can win in-window). |

### 3.4 `RzWsa.4` — Scale ~1B–7B

| Field | Content |
| --- | --- |
| **Ask** | Validate at least ~1B, discuss transfer to 1B–7B. |
| **Existing answer** | 1.485B OLMo + 8B LLaMA (matched + submitted Appendix D, Table 23), plus the 1.485B step-0→1,000 from-scratch trajectory. Not multi-seed 7B full pretraining. |
| **Outcome** | **Scale bar partially met** (mature adaptation, not full pretrain at 7B). |

**zWsa panel judgment:** best realistic move **2 → 3** if dual-knob + RULER + scale are crisp; **2 → 4+ unlikely** at conf 5 without unseen-task / multi-seed large pretrain.

---

## 4. Reviewer `27bE` (rating 3, conf 4) — five concerns

### 4.1 `R27bE.1` — Approximation chain not separately examined

| Field | Content |
| --- | --- |
| **Ask** | C_app, pure-tether, small-τ vs τ≈4; separate rationales for Cosh shape vs operating rule. |
| **Existing answer** | Four layers; Cosh unique only for stated convex surrogate; pure-tether = modeling choice; τ rule = empirical basin. |
| **Outcome** | **ANSWERABLE NOW** with Table-1 discipline + ablations. |

### 4.2 `R27bE.2` — Small models, one base, one lineage

| Field | Content |
| --- | --- |
| **Ask** | Persist across bases, head dims, scales (esp. \(b\ge 500\mathrm{K}\), larger \(d_{\mathrm{head}}\)). |
| **Existing answer** | The three-seed M4 exact-range factorial directly covers bases **500K/1M**, training lengths **256/1024**, and \(d_{\mathrm{head}}=32/64/128\), while the separate 1.485B/8B experiments answer scale. |
| **Wording trap** | Do not convert the 50.9M factorial into scale evidence or claim uniform improvement in every configuration; use it for base/head robustness and attribution. |
| **Outcome** | **ANSWERED AT TWO TIERS:** M4 for controlled base/head variation; 1.485B/8B for scale. |

### 4.3 `R27bE.3` — DAPE confounds shape vs capacity / tuning

| Field | Content |
| --- | --- |
| **Ask** | Operator fixed; only fixed schedules vary; DAPE tuning budget unclear. |
| **Existing answer** | **Direct tuning answer:** the reported DAPE row received a dedicated PE-learning-rate sweep at `10x` and `100x`; the better `100x` row was reported (`455.3` PPL@8K versus `477.7`, both seed 42). Attribution then moves to the M4 exact-range factorial: same operator, initialization, data/order, optimizer, budget, sampled extrema, and log-span; only interior frequencies differ. |
| **Outcome** | **ANSWERABLE NOW.** The tuning-budget question is direct; M4 removes learned capacity and scalar range from the shape-attribution comparison. |

### 4.4 `R27bE.4` — Independent τ + matched non-Cosh schedules

| Field | Content |
| --- | --- |
| **Ask** | Explicit ablation: tuned τ under Cosh; alternative schedules at matched τ. |
| **Existing answer** | Direct sweep: selected \(\tau=5\), rule \(5.657\) within `0.0119 NLL`. M4: the best \(0.75\times/1.00\times/1.25\times\) Cosh multiplier varies across configurations (2/4/6 wins), and the deformation-matched exponential is competitive. |
| **Outcome** | **CLOSED for the requested ablation type** (with fallible-rule honesty). |

### 4.5 `R27bE.5` — Held-out base + larger pre-specified scratch

| Field | Content |
| --- | --- |
| **Ask** | Held-out base config **and** larger-scale pre-specified training run. |
| **Existing answer** | **Base/head branch:** the new M4 factorial includes base 1M across two training lengths, three head dimensions, and three seeds under exact-range controls. **Scratch branch:** OLMo step-1000 is post-submission raw-hash-backed and reviewer-usable: same public step-0 initialization, scientific recipe, data-order prefix, 1,000 steps, and 2.097B counted tokens. Geo/EVQ PPL is `161.19/167.45`, `163.88/156.87`, and `182.73/159.64` at 4K/8K/16K. |
| **Outcome** | **BOTH REQUESTED BRANCHES ANSWERED AT THEIR PROPER TIERS.** M4 is mechanistic base/head evidence; OLMo is one large early-training trajectory with different trainer implementations, so call it same-initialization/same-scientific-recipe rather than bitwise paired. |

**27bE panel judgment:** highest realistic move **3 → 4** if theory + fixed
schedules + the direct DAPE tuning answer + raw-hash-backed scratch + mature
1.485B/8B evidence are tight and non-apologetic. The held-out-base subrequest
remains partial.

---

## 5. Master numbered checklist (all questions we must respond to)

Flattened list for drafting / paste audit. Every item maps to a stable ID in `00_`.

| # | ID | One-line question | Response status | Send now? |
| ---: | --- | --- | --- | --- |
| 1 | `AC.1` / `RzWsa.1` | Novelty vs FMRoPE / dead channels? | Dual-knob identity + cite Oka | **Yes** |
| 2 | `AC.1` / `RzWsa.2` | Matched FMRoPE comparison? | Seed-42 direct FMR control; M4 fixed-range factorial support | **Yes** |
| 3 | `AC.2` / `RDz6s.1` | Real tasks / less diagnostic? | AR NIAH + RULER + mature NLL | **Yes**, collar limits |
| 4 | `AC.2` / `RzWsa.3` | RULER? | OLMo + LLaMA family RULER | **Yes**, task-adapted |
| 5 | `AC.2` / `RzWsa.4` / `R27bE.2` | ~1B–7B scale? | 1.485B + 8B | **Yes**, not multi-seed pretrain |
| 6 | `AC.2` / `R27bE.2`/`.5` | Other bases / held-out base? | M4 exact-range 500K/1M factorial across three head dimensions | **Yes**, mechanistic tier |
| 7 | `AC.3` / `RDz6s.3` / `R27bE.1` | Separate theory layers? | Table-1 + text | **Yes** |
| 8 | `AC.3` / `R27bE.4` | Independent τ + non-Cosh schedules? | Direct sweep + M4 Cosh multipliers/matched exponential | **Yes** |
| 9 | `AC.3` / `R27bE.3` | Shape vs DAPE capacity? | DAPE budget answer + M4 exact-range schedules | **Yes** |
| 10 | `RDz6s.2` | Optimized Geo+YaRN baseline? | Narrow to matched-scale interaction | **Yes** (narrow) |
| 11 | `R27bE.5` | Larger pre-specified scratch? | 1.485B step-0→1,000, 2.097B tokens | **Yes**, single trajectory |
| 12 | `AC.4` | Material decision change package? | 1+2+3–5 in order | **Yes as package** |

**Do not answer as new science (out of panel / invent-work):**

- Universal SOTA over YaRN/LongRoPE/DAPE
- Global optimality of τ or Cosh
- Clean 4× AR capability as solved
- Unseen-task RULER transfer as claimed
- Official multi-seed 7B full-pretraining generality
- Failed post-sub MLA scarcity as win

---

## 6. Evidence we already have (by block)

### Block A — Submitted mechanism (paper)

| Evidence | Answers | Grade |
| --- | --- | --- |
| Primary I EVQ×YaRN 454M 3-seed | `RDz6s.2` complementarity; substrate | Primary |
| Primary III MLA scarce-channel 3-seed | allocation under scarce \(d_{\mathrm{rot}}\) | Primary |
| Primary II PE-dom seed-42 | diagnostic only after DAPE concede | Diagnostic |
| Appendix 8B A4 LoRA PPL | scale *anchor* only | Supporting |

### Block B — Post-submission theory / shape (rebuttal reports)

| Evidence | Answers | Grade |
| --- | --- | --- |
| Independent τ sweep | `R27bE.1`/`.4`, `AC.3` | Ready |
| M4 exact-range factorial, 3 seeds / 12 configurations | `R27bE.1`–`.4`, `AC.1/.3`, `RzWsa.1/.2` | Workstation-raw-backed; core attribution |
| Fixed non-Cosh schedules 3-seed | `R27bE.3`/`.4` | Ready |
| Held-out base 1M / d=128 3-seed | `R27bE.2`/`.5` | AUTHOR_CONFIRMED_NOT_PROMOTED; no external numbers |
| Phase18 base 10K vs 500K seed-42 | base not unique to 500K | Supporting |
| Exact-range s42 dual FMR | `AC.1`, `RzWsa.1`–`.2` | Ready |
| Exact-range 3-seed aggregate | same | CONDITIONAL |
| FMR/YaRN deploy boundary diagnostics | `RDz6s.2`, dual-knob | Ready |

### Block C — Mature scale / capability

| Evidence | Answers | Grade |
| --- | --- | --- |
| OLMo-2 1.485B CF NLL + 8K strict NIAH | `RzWsa.3`/`.4`, `RDz6s.1`, `AC.2` | Ready, task-family |
| OLMo 13-task RULER family | `RzWsa.3` | Ready, single seed |
| LLaMA-8B matched NLL + RULER | `RzWsa.3`/`.4` | Ready, single seed |
| OLMo scratch step-1000 | `R27bE.2/.5`, `AC.2/.4` | Raw-hash-backed; send with single-trajectory/trainer boundary |

---

## 7. Final panel outcome (honest)

### 7.1 What the package **can** do

1. **Answer every named concern with either evidence or an honest limit** — no silent holes on FMRoPE, RULER, τ, schedules, DAPE, base, scale.
2. **Use the opposite fixed-range and target-retargeted FMR outcomes to identify the tested knobs** — strongest novelty clarification for zWsa/AC.
3. **Correct base narrative** — principal 500K ≠ “no other base”; use tiered inventory.
4. **Give 27bE a clean 3→4 path** — theory layers + fixed schedules + DAPE tuning + completed scratch + mature models, without a new GPU run.
5. **Protect Dz6s 4** — keep complementarity claim; do not claim tuned-YaRN dominance or solved real-world suite.

### 7.2 What the package **cannot** honestly claim

1. Guaranteed score flips or acceptance.
2. Replacement of FMRoPE / optimized YaRN.
3. Universal Cosh / τ optimality.
4. Production multi-seed large-scale pretrain generality.
5. Clean 4× AR / broad unseen-task transfer.
6. The older CONDITIONAL 151.9M three-seed exact-range or held-out-base
   aggregates as if they were the M4 owner.

### 7.3 Score-path summary (not a prediction)

| Reviewer | Now | Best defendable move | What drives it | What still blocks |
| --- | ---: | --- | --- | --- |
| `Dz6s` | 4 | **4 → 5 target** | Mature endpoints + theory clarity | Still limited real-world breadth |
| `27bE` | 3 | **3 → 4** | M4 exact-range/tau/non-Cosh attribution + DAPE tuning + scratch + 1.485B/8B | M4 remains mechanistic; large scratch remains one trajectory |
| `zWsa` | 2 | **2 → 3** | Novelty dual-knob + RULER + scale | Conf 5; no FMR dominance; task-adapted |
| AC | meta | Soften rejection risk | Package of 1–3 above | Generality / novelty residual |

**Overall:** rebuttal is **defendable and sendable as triage**, not a **fake
“solved rebuttal.”** No mandatory new GPU experiment is required. Optional
promotion of conditional aggregates is a provenance task, not a reason to
rerun the completed scratch experiment.

### 7.4 Recommended send order (per reviewer)

| Reviewer | Lead with | Then | Collar |
| --- | --- | --- | --- |
| **zWsa** | Novelty distinction + direct FMR control | RULER + 1.485B/8B | No replacement; task-adapted; 4× open |
| **27bE** | M4 exact-range attribution + τ sensitivity | DAPE tuning; scratch; mature scale | Rule fallible; M4 is mechanistic |
| **Dz6s** | Endpoint taxonomy + mature AR/RULER | Matched YaRN narrowed; theory layers | Not optimized-YaRN contest |
| **AC** | Map to `AC.1`→`.4` in that priority | Point to same evidence | Remaining limits in one paragraph |

---

## 8. File links

| Role | Path |
| --- | --- |
| Authority | `00_REVIEWER_SCORES_AND_AC_METAREVIEW.md` |
| Strategy / wording | `01_REBUTTAL_PLAYBOOK.md` |
| This inventory | `02_RESPONSE_QUESTIONS_AND_OUTCOMES.md` |
| Numeric entry | `theory_results/EXPERIMENT_REPORT_20260724.md` |
| M4 exact-range factorial | `theory_results/M4_EXACT_RANGE_FACTORIAL_RESULT_20260726.md` |
| Exact-range s42 | `theory_results/MATCHED_RANGE_COSH_500M_S42_20260724.md` |
| Exact-range 3-seed | `theory_results/MATCHED_RANGE_COSH_500M_3SEED_20260724.md` (CONDITIONAL) |
| Paste drafts (Desktop) | `~/Desktop/EVQ_COSH_OPTIMIZED_REBUTTAL_DRAFTS.md` |

---

## 9. Status line

```
Response inventory: COMPLETE for all stable IDs in 00_
Send-ready: YES for the core using seed-42 FMR and no held-out-base numbers; NO for the unpromoted three-seed FMR or held-out-base aggregates
Fake “all concerns solved”: NO
New GPU required for this inventory: NO
```
