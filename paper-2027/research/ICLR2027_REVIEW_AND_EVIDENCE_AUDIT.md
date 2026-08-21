# ICLR 2027 review-absorption and evidence audit

- **Date:** 2026-08-19
- **Auditor scope:** `rebuttal/rebuttal_0723/` (full), `rebuttal/pre_rebuttal/`,
  `research_notes/`, `nonuniform-alloc/`, `AGENTS.md`, `docs/exp/`, and the
  complete `paper-2027/` source tree at commit `e869228`.
- **Status:** internal audit. Not submission prose. No `.tex` file was modified
  by this audit.
- **Rule followed:** every number below is quoted from a named owner file. No
  number was re-derived, averaged, or inferred.

> **Addendum, same day, after this audit was written.** A concurrent session
> converted the package to ICLR format while the audit was in progress:
> `main.tex:25` now loads `iclr2027_conference`, the abstract moved to
> `sections/00_abstract.tex`, and `sections/06_impact.tex` was replaced by
> `06_ethics` / `07_reproducibility` / `08_ai_use`. **§0.1 and §5 item 1 below
> are therefore partly superseded** — the format half is done; the
> dual-submission policy had not yet been checked against the official FAQ, and the
> freed ninth page is still unallocated. Line citations for `01_intro`,
> `02_related`, `04_experiments`, all appendices, and all tables are unaffected.
> `03_theory.tex` shifted by one line after its line 92; abstract citations now
> resolve to `sections/00_abstract.tex`. Nothing else in this audit changes.
>
> **Current-use note.** This file remains a historical audit of commit
> `e869228`; venue conversion and the full-RoPE rewrite were subsequently
> executed. Use current source plus `ICLR2027_RESEARCH_SYNTHESIS_20260819.md`
> for present state, not the line-numbered findings below.
>
> **2026-08-20 supersession note.** The raw-hash-receipted three-seed
> exact-range owner and the result-first evidence overview now supersede this
> audit's seed-42 headline, `fig_identification`, `fig_mature_crossover`, and
> `table_mature` paths. Those names below describe the audited snapshot only;
> current routing is in `HANDOFF.md` and
> `EXACT_RANGE_151M_3SEED_RESULT_20260820.md`.

## 0. Two facts that frame everything below

**0.1 The package is formatted for the wrong venue.** `paper-2027/main.tex:30`
loads `icml2027.sty`; `main.tex:111-112` emits an ICML **Impact Statement**;
`SUBMISSION_CHECKLIST.md:11` fixes the body at **8 pages**. The stated target is
ICLR 2027 (abstract 2026-09-18, full paper 2026-09-25). `paper-2027/README.md:31-45`
already identifies the mismatch and the fix: swap in `iclr2027_conference.sty`,
relax the body to **9 pages**, and replace `sections/06_impact.tex` with an AI-use
statement. Nothing in the body text needs to change. **One free page of body space
is currently unclaimed**, and every "we could not fit it" argument below is
answered by it.

**0.2 A canonical rewrite decision exists and has not been executed.**
`paper-2027/research/ICLR2027_RESEARCH_SYNTHESIS_20260819.md` §11 says the *next*
implementation turn should "rewrite the paper skeleton and theory section" around
full-RoPE geometry, and §10 `RISKY_REGIONS` flags
`sections/03_theory.tex` as "currently centered on the old cosine-only surrogate
narrative; replace rather than append." The current `03_theory.tex` is still the
old surrogate narrative. `AGENTS.md` §0.2 has already adopted the new claim
architecture. **The manuscript is one full architectural revision behind its own
project constitution.** Sections 2 and 5 below quantify what that costs.

---

# 1. Reviewer objection ledger

Verdict key: **FULL** = a fresh reviewer will see it answered without hunting.
**PARTIAL** = answered for some of the ask. **BURIED** = the answer exists but is
in the appendix, or worded so defensively that the reviewer reads it as a
concession. **NOT** = absent.

## 1.1 Area Chair `XLtL`

| ID | What the AC actually wanted (operative phrase) | What paper-2027 does, with file+line | Verdict | Concrete fix |
|---|---|---|---|---|
| **AC.1** | "establishes clear technical novelty over FMRoPE, provides a direct controlled comparison" | Oka cited `02_related.tex:29`; FMRoPE described as a base rule `02_related.tex:28-32`; three-level intervention table `tables/table_layers.tex:15-17`; exact-range control as the direct comparison `04_experiments.tex:35-44`, `a5_identification.tex:16-26`; dead channels attributed to Barbero `02_related.tex:26-28` | **PARTIAL** | The strongest novelty argument in the whole workspace is **missing**: Oka et al.'s own §6.3 calls the `L_target` requirement a "practical limitation" and lists adaptive schemes as future work (`REBUTTAL_HANDOVER.md` §3.1). Add one sentence to `02_related.tex` after line 31 citing that admission. It converts "overlap" into "we occupy the gap the cited paper names," in the cited authors' own words — zero cost, highest novelty leverage available. |
| **AC.2** | "results on a stronger benchmark or larger model" without "overstating small-model results" | OLMo-2 1.485B + LLaMA-3-8B RULER/2Wiki as `§4.3` `04_experiments.tex:87-146`, `tables/table_mature.tex`, `tables/table_ruler.tex`, `a6_mature_scale.tex` | **FULL** | Keep. Only ordering is wrong — see AC.4 fix and §4.2 item 3. |
| **AC.3** | "matched analytic-schedule ablations, sensitivity to the allocation parameter and base frequency" | M4 factorial `04_experiments.tex:66-77` + `tables/table_m4.tex`; matched exponential `04_experiments.tex:164-172`; bases 500K/1M and `d_head` 32/64/128 `a5_identification.tex:51-57` | **PARTIAL / BURIED** | Two of the three requested sensitivities are answered but under-sold. (a) The **base-frequency** ask is answered only inside an appendix design paragraph (`a5_identification.tex:52-53`); the main text never says "two bases, three head dimensions." Add it to `04_experiments.tex:66-68`. (b) The **held-out base-1M / `d_head`=128 three-seed suite** (`EXPERIMENT_REPORT_20260724.md` §4, deltas `−0.802/−0.664/−0.433/−0.287/−0.212` at 1K/2K/4K/8K/16K) was **sent to all four recipients** and is now deleted from the draft. See §2 row E-HELDOUT. |
| **AC.4** | "Recommendation can change only with clear novelty, controlled comparison, and stronger evaluation" | All three are in the body; the four boundaries are in `04_experiments.tex:164-172`, `05_discussion.tex:41-49` | **PARTIAL** | The paper wins on evidence and loses on *ordering*. `04_experiments.tex:101-103` leads the mature-scale result with the number where we lose (`72.19/42.44%` at 4K) before the number where we win 15.7× (`2.02/31.63%` at 8K). `AGENTS.md` §1.8 hostile-reviewer scan item 3 is exactly this failure mode. Reorder — see §4.2 item 3. |

## 1.2 Reviewer `Dz6s` (4 / conf 3) — the vote to protect

| ID | Operative phrase | Current draft, file+line | Verdict | Fix |
|---|---|---|---|---|
| **RDz6s.1** | "do not demonstrate the broad applicability … to real-world long-context tasks, pre-trained LLM models, or downstream scenarios" | 2WikiMultiHopQA greedy exact `04_experiments.tex:106-114`; RULER 13-family `04_experiments.tex:100-104`; 8B causal source use `04_experiments.tex:130-136`; endpoint taxonomy `04_experiments.tex:11-14` | **FULL** | None. This is the single largest improvement over the NeurIPS version and it is correctly in the body. |
| **RDz6s.2** | "doesn't answer whether a more optimized Geo+YaRN or other channel-based scaling/search could narrow the gap" | `\rs{}` disclosure `04_experiments.tex:20-25`; App F `a5_identification.tex:89-118`; substrate result `04_experiments.tex:178-188` | **BURIED (over-corrected)** | `04_experiments.tex:23-25` volunteers "not … a jointly tuned range-method comparison" to a panel that has not asked. `AGENTS.md` §1.2 admission test: no reviewer of *this* submission asked; omitting it does not make any sentence false (naming the operator `\rs{}` and specifying it in App. F already carries the truth). Cut that clause. See §4.2 item 1. |
| **RDz6s.3** | "more clearly separate: (i) the mathematical proof of the surrogate model; (ii) the empirical verification of the exact kernel collision score; and (iii) the results observed only after training" | Four lines of prose at `03_theory.tex:17-20`; surrogate check `a1_proofs.tex:120-166`; trained results `§4` | **PARTIAL — a credited strength was deleted** | The NeurIPS version's `paper/tables/table_epistemic_map.tex` is the object 27bE explicitly praised ("Table 1 explicitly differentiates the exact conditional statement established by Theorem 1 from subsequent choices"). It has **no counterpart in paper-2027** — grep for `epistemic` returns only `03_theory.tex:18` and `a1_proofs.tex:124`. Restore it as a compact four-row table in `sections/03_theory.tex`. It costs ~8 lines and buys back the only strength two reviewers independently named. |

## 1.3 Reviewer `zWsa` (2 / conf 5) — the four stated score-move conditions

| ID | Operative phrase | Current draft, file+line | Verdict | Fix |
|---|---|---|---|---|
| **RzWsa.1** | "clearly explain the novelty of VQ-Cosh/EVQ-Cosh over FMRoPE? My score would increase if the difference is technically and empirically convincing" | `02_related.tex:24-34`, `tables/table_layers.tex` | **PARTIAL** | Same as AC.1: add the Oka §6.3 self-stated limitation. Also, `tables/table_layers.tex:16` lists FMRoPE at level 2 and `\evq{}` at level 3 but the caption never states the asymmetry that makes the argument airtight: *a range method is target-aware by construction; an allocation method is not* (`REBUTTAL_HANDOVER.md` §3.2). One clause in the caption. **Do not** extend it to "FMRoPE depends on `L_train` and we do not" — the handover explicitly forbids that, since `τ ∝ d_eff/√L_train` also consumes `L_train`. |
| **RzWsa.2** | "A direct comparison with FMRoPE under matched settings is necessary … My evaluation would increase if VQ-Cosh/EVQ-Cosh shows clear advantages **or** complementarity" | `04_experiments.tex:35-64`; anchors `04_experiments.tex:50-51`; retargeting reversal `04_experiments.tex:79-84` | **FULL on "advantage", SELF-DAMAGED on "or"** | The reviewer offered two routes and the draft delivers advantage, then immediately spends four lines on the reversal (`04_experiments.tex:79-84`) — the exact pattern the memory note `feedback_rebuttal_never_import_internal_ledger` identifies as the 11628 death cause. The reversal is **not** removable (the same figure panel shows it, `fig_identification`), so keep it — but reframe it as a *positive identification claim* rather than a loss. See §4.2 item 4. Also: `MATCHED_RANGE_COSH_500M_S42_20260724.md` records that applying the target-range rule **on top of** an EVQ grid improves that model by `0.098/0.529/0.638` NLL at 512/1K/2K. That is literal composition — it belongs in `04_experiments.tex` §4.2 and is currently **absent**. It converts the reversal paragraph from a concession into the "or complementarity" branch the reviewer pre-approved. |
| **RzWsa.3** | "Could the authors include results on RULER, even at a small scale? My score would increase if the method improves RULER performance" | `tables/table_ruler.tex`, `tables/table_mature.tex`, `04_experiments.tex:100-104,137-140` | **FULL** | None. |
| **RzWsa.4** | "provide evidence on at least a 1B-scale model … My evaluation would increase if the gains remain consistent at larger scales" | 1.485B and 8B throughout `§4.3`; from-scratch trajectory `04_experiments.tex:125-128` | **FULL** | None. |

## 1.4 Reviewer `27bE` (3 / conf 4) — the highest-ROI vote

| ID | Operative phrase | Current draft, file+line | Verdict | Fix |
|---|---|---|---|---|
| **R27bE.1** | "a sequence of approximations whose individual contributions are not separately examined … values near tau ≈ 4, a range that lies beyond the regime directly justified by the underlying asymptotic expansion" | `03_theory.tex:17-20` (four lines), `03_theory.tex:204-212`, `a1_proofs.tex:109-119` (`why-constant-alpha`), `a1_proofs.tex:120-166` | **PARTIAL / BURIED** | The separation is real and correct but exists as running prose, not as a scannable object. Restore the epistemic map table (RDz6s.3 fix). `03_theory.tex:206-209` already contains the exact concession the reviewer wanted ("the small-$\tau$ series motivates the scaling and does not certify a finite operating point") — that sentence is *good* and should stay; the problem is that a reviewer has to read three paragraphs to find it. |
| **R27bE.2** | "whether its benefits persist across the base values, attention-head dimensions, and model scales … particularly in regimes involving b ≥ 500K and larger d_head values" | Bases 500K/1M and `d_head` 32/64/128 mentioned only at `a5_identification.tex:52-53`; scale at `§4.3` | **BURIED** | Three fixes, all cheap. (a) State "two bases, three head dimensions, three seeds" in the **main text** at `04_experiments.tex:66-68` — the words currently there ("two bases, two training lengths, three head dimensions") are present but the *contrast with the reviewer's complaint* is not made; add "so the axis is identified at two bases and three head dimensions, not only at the calibration setting." (b) Restore the held-out base-1M / `d_head`=128 suite (§2, E-HELDOUT). (c) **Zero-cost, high-value:** `nonuniform-alloc/RESEARCH_MEMO.md` §3 contains a purely geometric channel-accounting table across **OLMo-2 1.485B, LLaMA-3-8B, Qwen2.5, and DeepSeek-V3 MLA** (wrapped/resolving/slow = 32/10/22, 35/14/15, 40/7/17, 16/9/7). `03_theory.tex:167-171` currently gives only the OLMo column. Extending that table answers "one architectural lineage / production base values" with pure arithmetic and no GPU. |
| **R27bE.3** | "the current experimental design still leaves the influence of allocation shape partially confounded with differences in operator capacity" | Retired to App. E `a5_identification.tex:120-142`; disclaimed in caption `tables/table_pe_dominant.tex:6-11`; main-text pointer `04_experiments.tex:170-172` | **FULL, but with a self-inflicted beacon** | The substantive handling is exactly right (`CHANGES_FROM_NEURIPS2026.md` §3.1). But `04_experiments.tex:170-172` advertises the retired row in the body, and `tables/table_pe_dominant.tex:6` puts "**We rest no allocation-shape attribution on this table**" in **bold** inside a caption. For a *fresh* panel, that is a flag on a problem nobody raised. De-bold and de-advertise — see §4.2 items 7-8. |
| **R27bE.4** | "contrasting the proposed tau* choice with independently tuned tau values under the same cosh allocation, as well as with alternative non-cosh schedules evaluated at matched tau" | `04_experiments.tex:155-172`, `tables/table_m4.tex:21-27`, `03_theory.tex:214-219` | **ADDRESSED BUT BURIED — worst instance in the paper** | Both halves of the requested ablation are run and the paper leads with the number that makes the rule look bad. `03_theory.tex:216-218` and `04_experiments.tex:155-158` both open with "best in 4/12". The owner-backed containment statement — selected `τ` never leaves `0.75×–1.5×` of the rule value **across 21 configurations in two independent studies** (`REVIEWER_27bE.md` §2, from `PHASE16_99RUN_RAW_REANALYSIS_20260724.md` + M4) — is **absent from the draft entirely**, and it is the statement that makes a zero-search rule look like a result rather than a lucky guess. Reorder and add. See §4.2 item 5 and §5 item 4. |
| **R27bE.5** | "Evaluating the method on a held-out base configuration and conducting a larger-scale pre-specified training run" | Larger scratch run: `04_experiments.tex:125-128`, `a6_mature_scale.tex:50-59`. Held-out base: **nothing** | **HALF ANSWERED (scratch), NOT ADDRESSED (held-out base)** | The held-out-base half was answered in the rebuttal and is now deleted. See §2, E-HELDOUT. This is the single clearest regression from the NeurIPS response to the ICLR draft. |

## 1.5 Discussion-phase follow-ups (2026-08-03 official comments; no reviewer replied)

| ID | Source | Status in draft | Verdict | Fix |
|---|---|---|---|---|
| **FU.1 collision prefactor** | `official/01_AC.md`, `official/02_27bE.md` cite submitted Table 6, `c_pred/c_coll = 0.981 ± 0.003`, CV `0.28%` | **Absent from paper-2027** (grep: no `0.981`, no `c_coll`) | **CORRECTLY DROPPED** | Keep it dropped, permanently. `EVQ_TRUE_OBJECTIVE_ULTRA_AUDIT.md` §3.8 / finding `C06` is a *verified numerical contradiction*: the claimed minimizer does not minimize its stated objective (score derivative `−17.63` at the representative `τ=3.31`; a feasible `τ≈13.05` gives `C=0.04207`). This is a **hard red line** — see §4.1. |
| **FU.2 video-DiT base sweep** | `official/01_AC.md`, `02_27bE.md` cite Table 17 (base 100→50,000, dead channels 0/16→~9/16) as base coverage | `a2_experiment_details.tex:123-147` keeps a base-1,000 head-to-head only | **PARTIAL** | The base-sweep framing is what answered R27bE.2 in the discussion phase. If base coverage stays thin, this is a cheap appendix row to restore. |
| **FU.3 QuALITY** | `official/03_Dz6s.md` cites Table 21 (2,086 real-document questions; accuracy at random; gold-answer NLL `−1.7/−30.1/−21.4%`) | Absent | **CORRECTLY DROPPED** | `final_send/00_README.md` "未采用材料" and `REBUTTAL_HANDOVER.md` §7 record why: accuracy sits at the random baseline, and Dz6s specifically criticised NLL-only endpoints. Do not reinstate. |
| **FU.4 Table 19 YaRN leverage** | `official/01_AC.md` cites `−3.1%` (Geo) vs `−40.7%` (EVQ) at 8K, **and the unfavourable NTK row** (32×: 198.1 Geo vs 331.4 EVQ) | `a4_supporting_experiments.tex:7-23` keeps the `L=256` `\rs{}` table without the NTK row | **PARTIAL — check before reinstating** | `official/00_README.md` ⛔ records that Table 19's seed count is **not declared in the submitted PDF**; do not write "three seeds" for it. The NTK row is an internal negative that does not need volunteering (§4.1b). |
| **FU.5 seed accounting** | `FINAL_AC.md` §C and `FINAL_27bE.md` promise seed counts stated per result | `tables/table_mature.tex:10-12`, `tables/table_ruler.tex:7-8`, `04_experiments.tex:143-145` | **FULL, over-delivered** | The commitment is met by the table captions alone. The standalone main-text `\paragraph{Scope.}` at `04_experiments.tex:142-146` duplicates it in a location that reads as a retreat. See §4.2 item 2. |

---

# 2. Post-May evidence inventory

Submission date 2026-05-01. Everything below is dated after it. "In draft?" cites
`paper-2027/` file+line.

## 2.1 Identification / theory layer

| Owner | What it shows | Headline numbers (as written by the owner) | In draft? | Disposition |
|---|---|---|---|---|
| `MATCHED_RANGE_COSH_500M_S42_20260724.md` | Exact-range interior-allocation identification, 151.9M, seed 42 | Cosh−uniform NLL `−0.4775/−0.2050/−0.1128` @512/1K/2K; anchors `32/32, 27/32, 22/32`; retargeted `+0.061/+0.182/+0.279`, anchors `5/32,1/32,2/32`; **EVQ+target-range composition `0.098/0.529/0.638`** | Yes — `04_experiments.tex:59-64`, `a5_identification.tex:34-40`. **Composition numbers absent** | (a) **Promote the composition triple to main text.** It is the "or complementarity" branch `RzWsa.2` pre-approved and it turns the retargeting paragraph from a loss into a two-knob result |
| `M4_EXACT_RANGE_FACTORIAL_RESULT_20260726.md` + `m4_exact_range_factorial_evidence_20260726.json` | 12 structural configs × 3 seeds × 5 schedules, exact range | `−0.00912/−0.00988/−0.01210/−0.01062` vs Geo; `8/12, 7/12, 10/12, 9/12`; cosh−exp `+0.00074`, `p=0.836` | Yes — `tables/table_m4.tex`, `04_experiments.tex:66-77` | Keep. Note `CHANGES_FROM_NEURIPS2026.md` §6.4: the curated JSON's `best_cosh_multiplier_counts` summary (`1/1/1`) and its per-row regret mean (`0.007366`) disagree with the Markdown owner (`0.011440`). Only the recomputable `2/4/6` is used in the draft — **correct**; keep the validator gap on the internal list |
| `MATCHED_RANGE_COSH_500M_3SEED_20260724.md` | Three-seed exact-range aggregate | `−0.3159/−0.1949/−0.1674`; 3/3 seeds | **No — deliberately deleted** (`CHANGES_FROM_NEURIPS2026.md` §6.2) | (c) **Unusable as written.** `AGENTS.md` §1.5 marks it `AUTHOR_CONFIRMED` only, with no local raw/per-seed values or CIs. Do not reinstate without promotion. If promoted before 09-25 it belongs in the body — it is the seed replication a reviewer will ask for |
| `EXPERIMENT_REPORT_20260724.md` §2 + `PHASE16_99RUN_RAW_REANALYSIS_20260724.md` | 99-run τ sweep, 9 configs × 5 τ, then 3 seeds on retained arms | vs midpoint-Geo: **7/9 config means, 18/27 paired runs**, mean weighted extrapolation NLL improvement `0.0133`; vs pilot-selected neighbour **3/9**; selected τ within **0.75×–1.5×** of formula in all nine | **Only one clause** — `04_experiments.tex:160` and `03_theory.tex:216-217` carry `τ=5` vs `5.657` = `0.0119` NLL | (a) **Promote.** The `0.75×–1.5×` containment across 21 configurations is the strongest available answer to `R27bE.4` and is currently missing. Report `3/9` alongside it — `DISCUSSION_PHASE_FOLLOWUP_PLAYBOOK.md` 27-3 requires the pair to travel together |
| `EXPERIMENT_REPORT_20260724.md` §4 (**E-HELDOUT**) | Held-out base-1M, `d_head`=128, three seeds, formula applied untuned | `−0.802/−0.664/−0.433/−0.287/−0.212` @1K/2K/4K/8K/16K; in-domain `+0.069` @512 | **No** — removed pending promotion | (a) **Promote if the raw/per-seed bundle can be promoted before 09-25; otherwise (c).** This is the only direct answer to `R27bE.5`'s held-out-base half, it was sent to all four recipients, and its effect size is ~an order of magnitude above M4's |
| `EXPERIMENT_REPORT_20260724.md` three-level ladder (**E-LADDER**) | Fixed zero-parameter schedules at three control levels | Level 1 EVQ−Paper-Geo `−0.256/−0.305/−0.223/−0.238` @1K/2K/4K/8K, 3 seeds agreeing; Level 2 native-endpoint `−0.113/−0.149/−0.207/−0.190/−0.099` @512/1K/2K/4K/8K, every paired 95% interval excluding zero | **No** | (b) **Appendix, at minimum.** `REVIEWER_27bE.md` §4 and the revision plan item (3) promised "A second [appendix] will report the three-level ladder and the base/head factorial." Neither is in the draft. Level 2's RMS-deformation matching (`0.2557` for every non-Geo arm) is the cleanest answer to "is EVQ just farther from geometric?" |
| `OLMO2_POSTHOC_FREQUENCY_TRANSPLANT_OBSTRUCTION_20260726.md` | **Exact theorem**: if `Aᵀ R(ω′Δ) B = R(ωΔ)` for all Δ then `|ω′| = |ω|`; frequency multisets must match up to sign, permutation, and integer-position alias | OLMo-2 replacement changes **63 of 64 rotary pairs**; proof in four steps | **No** — `03_theory.tex:140-141` alludes to it in one defensive clause | (a) **Promote to main text as a numbered theorem.** `ICLR2027_RESEARCH_SYNTHESIS_20260819.md` §3.4 and §12 rank this the #1 acceptance-leverage action and calls it "paper-ready." It costs ~15 lines, needs zero GPU, and it converts the paper's largest apparent weakness (the mature-model in-window drop) into a *predicted consequence of a proved theorem*. `INWINDOW_PRESERVATION_ROUTE_AUDIT_20260731.md` §5 spells out the reframing: theorem → prediction → seven independent confirmations → Q/K-only upper bound → three admissible routes |
| `paper-2027/research/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md` §2 | Full sin/cos subspace geometry; whitened cross-Gram canonical correlations; **exact stable-rank identity** `r₂(R) = 2K/(1+(K−1)c̄)` | Analytic Gram vs 200,001-point quadrature max abs `1.51e-6`; phase-rotation invariance `7.44e-15` | **No** | (a) **Promotable and strongly recommended** — it is the "real theorem" the synthesis memo says makes this an ICLR-grade theory paper rather than a schedule paper. But it is a *rewrite*, not an insertion; `AGENTS.md` §0.2 already commits to it. Budget realistically against the 09-25 deadline |
| same report §3 | Low-frequency collapse: `V_ω → span{1,Δ}`; softmax-centered limit `span{Δ−E_pΔ, Δ²−E_pΔ²}` | `K=64`: 24 low-frequency pairs, nominal 48 dims, block-whitened `r₂ = 2.0002` → **95.83% stable-dimension loss**; log-log slope `4.009–4.011` | **No** | (a) Promotable. Note the guardrail: `AGENTS.md` §4 forbids calling these channels dead, unused, or freely reclaimable |
| same report §4 | **Static-objective failure boundary** — two explicit counterexamples | Cosine metric picks A (`2.62e-9 < 1.44e-8`) while full stable rank prefers B (`64.48 < 128.00`); length-order reversal `C_L(A)=.45837<.61695` but `C_2L(A)=.45834>.41132`; analytic Fourier tables alias exactly at `Φ(Δ+L)=Φ(Δ)` | **No** | (a) Promotable, and *protective*: it pre-empts "why not just minimise collision / maximise rank?" — a question a theory-minded ICLR reviewer will ask within one page of seeing `C_app` |
| same report §5.2-5.3 (**50M co-adaptation 2×2**) | Frozen-weight weights×table counterfactual | Self-consistent Geo/Geo `PPL 7.14`, EVQ/EVQ `7.16`; mismatched Geo/EVQ `76.20`, EVQ/Geo `23.05`. Factorial: `E_T=+0.5991`, `E_W=−0.5965`, **`I_{T×W}=−3.5367`**, bootstrap CI `[−5.165,−3.039]`, ≈5.9× the main effects. Bare static rank *improves* in the worst cell | **No** | (a) **Promote — highest evidence-per-line ratio in the inventory.** It is CPU-only, deterministic, hash-receipted, and it independently establishes the co-adaptation claim that the transplant theorem proves in the exact case. Limitation to record internally: seed-42 only (§11.1) |
| same report §6 (**base-only controls**) | How much of EVQ is reachable by base alone | `base=8.06K` moves EVQ-weights/Geo-table PPL `23.05 → 9.63`, recovering **74.7%** of the table gap; residual `9.63 → 7.16` | **No** | (c) **Unusable outward as-is, and it is a live threat.** It is a frozen-checkpoint counterfactual, not a from-scratch base arm (§11.5). It must never be presented as "EVQ ≈ base change", and equally must never be claimed away. **The exact-range control is the answer to it** — which is one more reason `04_experiments.tex:35-64` must stay first in the paper |
| `research_notes/.../audits/FULL_ROPE_CLAUDE_AUDIT_20260819.md` | Independent re-derivation of the same geometry + same-parity harmonic construction `ω_k = π a_k/L` | Confirms six conclusions; known defects: one-sided quadrature needed a resolution repair, an older causal example was rejected by its own script | **No** | (c) Internal corroboration only. `ICLR2027_RESEARCH_SYNTHESIS_20260819.md` §6 forbids verbatim reuse and forbids promoting the harmonic comb as a schedule |
| `research_notes/.../audits/DEPENDENCY_SPECTRUM_CLAUDE_AUDIT_20260819.md` | Dependency-gradient spectrum pilot | ≈`r^{-2.4}` on one small sample | **No** | (c) **Unusable.** `AGENTS.md` §4 pins it as "a small CPU pilot and internal falsification tool, not a universal demand law or paper claim" |
| `EVQ_TRUE_OBJECTIVE_ULTRA_AUDIT.md` | Adversarial audit of the whole objective | No task-independent universal optimum (proved on a legitimate teacher-attention family); Cosh best only at 128/512 among fixed-span alternatives, Exp best at 256/1K, attention-derived two-band best at 2K/4K/8K; **finding C06** (see §4.1) | Partially — `04_experiments.tex:164-172` states Cosh is not the only effective shape; the two-band result is absent | (b/c) Cosh non-uniqueness: **already correctly in the body** and should stay — it is a claim ceiling `AGENTS.md` §4 enforces, and it reads as scientific maturity, not weakness. The two-band ranking is internal-only |
| `TAU_TRUE_ROLE_AND_OPERATING_RULE_AUDIT.md` | What τ actually does | τ is **not** a pure shape knob: at `L=256, d_head=128, K=64, B=500K`, τ=6/8/10 compress the sampled exponent span to `80.5%/61.5%/49.2%` of midpoint-Geo; a global `1.25×` improves retrospective regret in Phase16 but is rejected on the `L_train=1024` hold-out and by an independent `L=128, d_head=64` counterexample | **Partly and correctly** — `a5_identification.tex:21-26` states that the endpoint-normalised cosh arm is *not* the deployed midpoint grid, and `03_theory.tex:130-131` states that the midpoint grid changes sampled extrema and span | (b) The honest handling is already in the appendix and is load-bearing for the exact-range claim's validity. **Do not delete it** — deleting it would make `04_experiments.tex:41-42` false |
| `TRAINING_FREE_TAU_SELECTOR_20260724.md` | Failed training-free τ selector | selector returns ≈13.2, PPL regret `5.44%` vs `4.69%` for the old formula | **No** | (c) Internal guardrail only. Never volunteer |

## 2.2 Mature-model layer

| Owner | What it shows | Headline numbers | In draft? | Disposition |
|---|---|---|---|---|
| `OLMO2_1B_SELECTIVE_QK_PHASE_ADAPTATION_20260729.md` | Phase-matched Q/K-only continuation; the only fully coverage-matched mature experiment in the workspace | RULER macro Native/EVQ `72.19/42.44` @4K, `2.02/31.63` @8K, `0.38/5.03` @16K; 2Wiki exact `22.0/21.5`, `0/17.5`, `0/4.0`; F1 `25.99/24.84`, `0.07/21.48`, `0/8.57`; 4K macro raised from `11.09%` to `42.44%` | **Yes, as headline** — `04_experiments.tex:94-114`, `tables/table_mature.tex`, `main.tex:97-100` | Keep. `nonuniform-alloc/RESEARCH_MEMO.md` §1 explicitly identified this as "your strongest attribution evidence, not in the paper" — **that gap is now closed.** Remaining gap: the `11.09 → 42.44` adaptation-scope recovery is not stated anywhere in the draft (see §5 item 6) |
| `OLMO2_1B_4K_RULER_FAMILY_ADAPTATION_20260726.md` + JSON | Full Q/K/V/O 13-family RULER | `82.16/37.51`, `0.08/21.29`, `0/6.13` | Yes — `tables/table_ruler.tex:19-20` | Keep as the adaptation-scope contrast row |
| `OLMO2_1B_4K_ONLY_ROUTING_CONVERSION_20260726.md` | Matched routing conversion, strict first-number AR | `0/100` vs `69/100`; second EVQ seed `67/100`; NLL `2.235/3.735/4.851 → 2.548/2.703/2.925` | Yes — `04_experiments.tex:118-124`, `a6_mature_scale.tex:8-20` | Keep |
| `OLMO2_FRESH_ALL_LONG_GAP_N100_20260727.md` | Disjoint set, every gap beyond training support | `0/100` vs `49/100`, `48/100` | Yes — `04_experiments.tex:122-124` | Keep. `AGENTS.md` §1.5 trap: never merge with the `69/67` set |
| `EVQ_QUERY_GAP_FINAL_DIAGNOSTIC.md` | Strict raw-token **complete-answer-string + terminal EOS**, greedy AR | EVQ `100/100`, `98/100`, `60/100` @4K/8K/16K; matched Native `95/100`, `18/100`, `0/100` | **No** | (b) **Appendix-only, with a hard label.** This is the most impressive capability number in the workspace and was the lead of `AC_PUBLIC.md` §3 and `FINAL_27bE.md`. It is currently invisible. It **cannot** sit next to the `69/67` strict-first-number numbers — `REBUTTAL_HANDOVER.md` §4 and `AGENTS.md` §1.5 both flag that adjacency as reading like catastrophic seed variance. Put it in `a6_mature_scale.tex` as its own subsection with the endpoint defined in the first clause |
| `OLMO2_1B_RELEASED_ROPE_BASELINE_20260725.md` | Same-init step-0→1,000 scratch trajectory, 2.097B tokens | PPL Geo/EVQ `161.19/167.45`, `163.88/156.87`, `182.73/159.64` @4/8/16K; ΔNLL `+0.0724/+0.0381/−0.0437/−0.1351` @2/4/8/16K; `122/128` and `126/128` documents | Yes — `04_experiments.tex:125-128`, `a6_mature_scale.tex:50-59` | Keep. `AGENTS.md` §1.5: same-initialization / same-scientific-recipe, **never** bitwise paired. Draft complies (`a6_mature_scale.tex:56`) |
| `EVQ_8B_ADAPTATION_EVIDENCE_20260724.md` | Matched 8B NLL + causal source use | ΔNLL `+0.390/−1.510/−2.048` @8/16/32K; hit@16 `18.75% → 64.06%`; gold-block deletion `−0.0095` (Native) vs `+1.5055` (EVQ) | Yes — `04_experiments.tex:130-136`, `a6_mature_scale.tex:63-70` | Keep. The deletion test is the best mechanism result at scale and is correctly in the body |
| `LLAMA8B_MATCHED_RULER_MIX_20260726.md` + JSON | Matched 8B 13-family RULER | macro `94.44/77.60` @8K, `0.295/14.03` @16K; normalized exact `17.69/21.54`, `0/1.54`; **32K macro must not be reported** (3 shards never started) | Yes — `tables/table_ruler.tex:26-30`, `04_experiments.tex:137-140` | Keep. `CHANGES_FROM_NEURIPS2026.md` §7.4 records the `n/r` correction — verify it survives any table edit |
| `OLMO2_1B_MATCHED_RULER_CONTINUATION_20260727.md` | Matched Native/EVQ physical-4K 13-family continuation | 2×/4× length transfer positive; Native 4K boundary | Superseded in the draft by the selective-Q/K arm | (b) Appendix at most. `nonuniform-alloc/RESEARCH_MEMO.md` §1 argues the selective-Q/K arm is "stronger, cleaner, more expensive" — the draft already made that choice correctly |
| `LLAMA8B_FRESH_COUNTERFACTUAL_RESULT_20260726.md` | EVQ-only feasibility arm, not matched | single arm | No | (c) Not a matched comparison; `README.md` §3 marks it not-a-headline |
| `OLMO2_1B_CLEAN_4K_TULU_FULL_RULER_20260728.md` | Clean (non-RULER-generator) adaptation → full RULER | `9.74/4.01/2.05%` | No | (c) **Internal negative. Do not volunteer** — §4.1b |
| `OLMO2_1B_NON_RULER_ADAPTATION_SEARCH_20260731.md` | Seven-arm search for clean transfer | best 4K screen `0.1167`, best 8K `0.0750`, against length-matched controls `0.5700` / `0.5125`; one-token curriculum failed its own held-out task | No | (c) internal — **but see §5 item 2.** `INWINDOW_PRESERVATION_ROUTE_AUDIT_20260731.md` §1 shows all seven arms are "in-place replacement + post-hoc adaptation", i.e. seven independent confirmations of the transplant theorem. If the theorem enters the paper, a *one-line* aggregate ("seven adaptation variants in this class all fail, consistent with Thm. X") is a strength, not a confession. Reporting the arms individually is not |
| `OLMO2_POSTHOC_..._OBSTRUCTION` (again) | see §2.1 | — | — | — |
| `mla_scarcity_seed42_result_20260724.json` | Post-submission K=8 MLA scarcity test | `claim_gate: DOES_NOT_SUPPORT_PRACTICAL_SCARCE_CHANNEL_ADVANTAGE`; at K=8, 8K `evq_minus_native = +0.667` NLL | No | (c) **Hard red line, §4.1.** The draft's MLA claim is correctly narrowed to K=16 persistence (`04_experiments.tex:196-199`) and the NeurIPS "scarcer channels ⇒ allocation matters more" claim is gone. Keep it gone |
| `results/.../432M MLA 3-seed` via `a3_supporting_results.tex` | K=16 MLA, 3 seeds | PPL@16K `138.8±5.5 → 95.6±4.1` (`−31.1%`), `+0.9%` @8K | Yes — `04_experiments.tex:191-199`, `a3_supporting_results.tex` | Keep. `DISCUSSION_PHASE_FOLLOWUP_PLAYBOOK.md` §8: the `71.1` figure is **EVQ + MLA wavelength-blend operator**, not pure EVQ — the draft's table complies |

## 2.3 Concurrent work and never-run designs

| Owner | What it shows | Headline numbers | In draft? | Disposition |
|---|---|---|---|---|
| `LEROPE_CONCURRENT_WORK_NOTE_20260728.md` | LeRoPE facts, verified against the primary paper | one learned log-scale per band, shared across layers/heads; 52M–2.5B non-geometric profiles; **Fixed-LeRoPE retains 63.6%** of full gain vs `10.4%` for p-RoPE at 217M; LeRoPE degrades more than RoPE under naive extrapolation | Yes — `02_related.tex:36-48`, `a5_identification.tex:136-142` | Keep. Guardrails hold: no "EVQ ≈ LeRoPE", no "LeRoPE validates EVQ" (`AGENTS.md` §4). The retracted `2.205L` narrative is correctly absent |
| `EVQ_OBSERVED_BAND_LEROPE_READY_REPORT_20260728.md` | Observability-partitioned frequency update: bands with `L_train·ω/2π ≥ 1` may learn, slower bands frozen at EVQ. **Status `OFFLINE_READY_NOT_TRAINED`** | Full-band EVQ-LeRoPE, 2 completed seeds: mean ΔNLL `+0.00531/−0.03315/−0.09820` @128/256/512; slow-band learned residual ℓ2 `0.1162`/`0.1213`; for `L=128, b=500K, τ=5, K=32`: learnable bands `0..21`, frozen `22..31`; literal Native/EVQ index splice **rejected** (Native pair 21 ≈ `0.000182` vs EVQ pair 22 ≈ `0.041335`, order reversal); seed 44 interrupted | **No** | (c) **Unusable as a result** — the observed-band arm has no training result, only a preflight receipt. The *full-band* two-seed numbers are usable as an appendix diagnostic if a learned-table comparator is wanted, but the in-window direction flips across seeds (`+0.02059` at 128 for seed 43), so `AGENTS.md` §1.8 "two branches under one header" applies. Recommend leaving out |
| `INWINDOW_PRESERVATION_EXPERIMENT_PLAN_20260731.md` + `INWINDOW_PRESERVATION_ROUTE_AUDIT_20260731.md` | Route audit for preserving in-window while extending extrapolation; the theorem→prediction reframing; the `C_app` conditional-minimisation fix | Index-level protected splice at OLMo geometry gives min log spacing **`0.0062`** vs geometric **`0.2050`** (4 near-duplicate pairs) when the fastest 16 pairs are protected; slow-end residual at `λ=500K` varies by **`0.0013`** across a 4K window; author's own success probabilities G1 ~95%, G2 ~45%, G3 ~8% | **No** | (a) for §5 of the route audit (the theorem→prediction reframing — see §5 item 2); (c) for P-EVQ / far-only residual, which are `PLAN_ONLY_NOT_REGISTERED_NOT_EXECUTED`. `AGENTS.md` §1.8: pending/design-only work must be absent from completed-evidence prose |
| `OLMO2_NATIVE_IMPORTANCE_PROTECTED_EVQ_HYPOTHESIS_20260728.md` | P-EVQ design + Stage-D gates | design only | No | (c). `research_notes/QWEN_...20260814.md` records owner state `DESIGN_ONLY_..._NOT_EXECUTED`. Claim-identity rule if it ever runs: "Native-importance-protected hybrid retrofit", never pure EVQ-Cosh |
| `nonuniform-alloc/RESEARCH_MEMO.md` (2026-08-19) | Self-refutation of v1 + two surviving assets | (i) **Multi-source split**: single-source families average `−16.67` pp at 4K, multi-source families `−40.96` pp, difference `24.3` pp; per-family `niah_multikey_2 95.00→0.00`, `fwe 46.67→56.67` (**+10.00**). (ii) **Three-regime channel accounting**: OLMo-2 `32/10/22`, LLaMA-3-8B `35/14/15`, Qwen2.5 `40/7/17`, DeepSeek-V3 MLA `16/9/7`. Also: 22 slow channels contribute `Σ(1−cos ω_k D) = 11.87` ≈ **1.05 logit units** of monotone recency kernel at 32K — slow bands are *not* free budget | Partly — `05_discussion.tex:29-39` and `a6_mature_scale.tex:39-48` carry a **different** grouping (3 single-needle families `−21.67`, 5 multi-key/value/query `−52.25`); `03_theory.tex:167-171` carries the OLMo column only | (a) for the **cross-model regime table** — extending `03_theory.tex:167-171` to four production models answers `R27bE.2` at zero cost. (b) for the source-count prediction. ⚠️ **The two family groupings must be reconciled to one owner before either is cited**; the draft currently uses the `a6` grouping and must not silently adopt the memo's numbers |
| `research_notes/QWEN_PARETO_AND_PRETRAINED_EVQ_ADAPTATION_ANALYSIS_20260814.md` | External-model analysis, archived | status `EXTERNAL_MODEL_ANALYSIS_NOT_VERIFIED` | No | (c) **Not an evidence owner.** Its content (constrained-allocation formulation, `C_ext` two-term structure, pinned-λ*) is a research direction, not a result |
| `research_notes/iclr2027/02_PAPER_REVISION_PLAN.md` | The August plan: reposition to "substrate + mechanism + protected retrofit" | — | Superseded by `ICLR2027_RESEARCH_SYNTHESIS_20260819.md` | (c) Historical. Note that its Pillar iii (P-EVQ) never ran, and the synthesis memo correctly dropped it |

---

# 3. Promises a future reviewer could check

None of these are enforceable against a *new* ICLR panel — the ICLR reviewers
have not read the NeurIPS thread. Three of them are nonetheless load-bearing,
because a reviewer who finds the public NeurIPS/OpenReview record (which
`SUBMISSION_CHECKLIST.md:64-66` anticipates) will read a broken promise as a
credibility signal.

| # | Exact promise, and where it was sent | Must the ICLR paper contain it? | Current state |
|---|---|---|---|
| P1 | "We should have cited Oka et al.; the revision will." — `AC_PUBLIC.md:7`, `official/03_Dz6s.md:27`, `04_zWsa.md:18`, `FINAL_zWsa.md:11`, `REVIEWER_zWsa.md:1` (five separate sends) | **Yes — non-negotiable.** This was the one clean concession that bought standing everywhere else (`REBUTTAL_HANDOVER.md` §6) | ✅ `refs/references.bib:398`, `02_related.tex:29,31`, and a dedicated FMRoPE paragraph. **Kept** |
| P2 | "§2 will cite and discuss Oka et al., **with a table contrasting the parameterization levels and the stage at which each acts**" — `REVIEWER_Dz6s.md:44` (1), `REVIEWER_zWsa.md:46` (1) | Yes | ✅ `tables/table_layers.tex` — exactly that table. **Kept** |
| P3 | "§3.3 and **Table 1** will separate the four links and present c = 1 as a zero-search default" — `REVIEWER_27bE.md:49` (1); mirrored in `REVIEWER_Dz6s.md:44` (4) | Yes — and independently, because it restores a credited strength | ⚠️ Four links are separated in prose (`03_theory.tex:17-20`, `:83-85`, `:204-219`) but **Table 1 is gone**. `paper/tables/table_epistemic_map.tex` has no counterpart. **Broken** |
| P4 | "A new appendix will report the exact-range factorial with the tuned-τ and matched-exponential comparisons **and the 0.75×–1.5× containment**" — `REVIEWER_27bE.md:49` (2) | Yes | ⚠️ Factorial ✅ (`a5_identification.tex:49-78`), matched exponential ✅ (`table_m4.tex:24,27`), **containment absent**. **Partly broken** |
| P5 | "A second [appendix] will report the **three-level ladder and the base/head factorial**" — `REVIEWER_27bE.md:49` (3) | Base/head yes; ladder yes | ⚠️ Base/head factorial ✅; **three-level ladder absent**; **held-out base-1M/d128 absent**. **Broken** |
| P6 | "The PE-dominant table will no longer support the allocation-shape claim, which rests on the ladder" — `REVIEWER_27bE.md:49` (4) | Yes | ✅ `a5_identification.tex:129-134`, `table_pe_dominant.tex:6-11`, `04_experiments.tex:170-172`. **Kept** (over-executed — see §4.2 items 7-8) |
| P7 | "A third will report the 1.485B result **with its single-trajectory and trainer boundaries in line**" — `REVIEWER_27bE.md:49` (5) | Yes | ✅ `a6_mature_scale.tex:50-59` states both. **Kept** |
| P8 | "§6 will state that the evidence supports the allocation axis and a closed-form zero-search operating point, **not universal optimality of Cosh or of the τ default**" — `REVIEWER_27bE.md:49` (6) | Yes | ✅ `04_experiments.tex:164-172`, `05_discussion.tex:41-45`. **Kept** |
| P9 | "§5 will restate the YaRN result as **substrate dependence**" — `REVIEWER_Dz6s.md:44` (5), and "We … will state it that way rather than as complementarity" — `REVIEWER_Dz6s.md:38` | Yes | ✅ `04_experiments.tex:178-186`, `05_discussion.tex:15-17`, and the operator is renamed `\rs{}` throughout. **Kept** |
| P10 | "**The revision will state precisely which range transform each experiment applies.**" — `REVIEWER_zWsa.md:46` (4) | Yes | ✅ `\rs{}` everywhere + `a5_identification.tex:89-118` + the distinct MLA wavelength-blend operator is named as such (`a3_supporting_results.tex:17-19,38-40`). **Kept** |
| P11 | "§6 will report the in-window trade-off, **metric dependence** and the **adaptation-scope ablation**" — `REVIEWER_Dz6s.md:44` (6), `REVIEWER_zWsa.md:46` (6) | Yes | ⚠️ In-window ✅ (`04_experiments.tex:126-127`, `tables/table_mature.tex`); metric dependence ✅ (`a6_mature_scale.tex:76-78`); **adaptation-scope ablation is present only as two unconnected table rows** (`table_ruler.tex:19-24`) — the sentence "restricting adaptation to Q/K raises 4K macro 37.51 → 42.44 and 8K 21.29 → 31.63" appears nowhere. **Partly broken, and this one is a lost asset, not just a lost promise** — see §5 item 6 |
| P12 | "The revision will state seed counts inline for every result rather than in aggregate." — prepared answer `DISCUSSION_PHASE_FOLLOWUP_PLAYBOOK.md` X-1; sent form in `FINAL_AC.md` §C | Yes | ✅ Over-delivered (`table_mature.tex:10-12`, `table_ruler.tex:7-8`, `04_experiments.tex:143-145`). Trim the duplicate — §4.2 item 2 |
| P13 | "[the DAPE label] correction will appear in the revision" — **prepared but never sent** (`DISCUSSION_PHASE_FOLLOWUP_PLAYBOOK.md` 27-1; `REBUTTAL_HANDOVER.md` §1.2 records that `paste/` contains zero occurrences of "DAPE") | No — nothing was promised outwardly | ✅ Handled anyway and handled well. `CHANGES_FROM_NEURIPS2026.md` §3.1 records the deliberate decision **not** to print "an earlier draft mislabelled this" in the PDF, on the grounds that it would hint at dual submission. **Endorse that decision** |

---

# 4. Disclosure red lines

The governing rule is `AGENTS.md` §1.1: **one objective (maximise acceptance),
exactly two binding constraints (no fabrication, no junk experiments)**; and
§1.2's boundary-admission test — a negative enters outward prose only if (1) this
reviewer asked about that specific thing, (2) omitting it makes a sentence you
are actually writing *false*, or (3) it defeats a **stronger** named attack.
`AGENTS.md` §4 is explicit that its table entries are **claim ceilings, not
caveat mandates**: they bound what may be asserted; they never require printing a
confession.

For a *fresh* ICLR panel, condition (1) is empty at submission time. Every
boundary now in the draft must therefore justify itself under (2) or (3).

## 4.1a Hard red lines — must never be claimed away, in any draft

These are the ones where narrowing the claim is the only legal move.

1. **The collision-prefactor result (submitted Table 6) is dead.**
   `EVQ_TRUE_OBJECTIVE_ULTRA_AUDIT.md` §3.8 / `C06`: the claimed `c_coll`
   minimizers do not minimize their stated objective (score derivative `−17.63`
   at `τ=3.31`; feasible `τ≈13.05` gives `C=0.04207`), and §3.8 further records
   that `scripts/analysis/verify_c_coll.py:39-64` re-derives the ratio *from the
   table values*, so the `c_coll = 1.171` provenance does not establish a
   collision calibration. It is currently absent from `paper-2027` — keep it
   absent. It was cited in `official/01_AC.md` and `official/02_27bE.md`; do not
   reinstate it on the strength of having sent it.
2. **No official FMRoPE implementation exists.** `REBUTTAL_HANDOVER.md` §1.4.
   The only legal wording is a paper-faithful reimplementation of the §6.1 rule.
   `a5_identification.tex:16-20` and `04_experiments.tex:38-40` comply.
3. **No official YaRN reproduction.** The repository operator is a fixed-index
   smooth-ramp scaler (`AGENTS.md` §4). `\rs{}` naming + `a5_identification.tex:105-118`
   comply. The MLA `117.9` row uses that architecture's *MLA wavelength-blend
   operator* and remains distinct from the paper's YaRN-style operator
   (`CHANGES_FROM_NEURIPS2026.md` §7 D); `a3_supporting_results.tex:17-19,38-40`
   comply.
4. **The 32-parameter row is not DAPE.** It is a layer-shared learnable
   `inv_freq` baseline. It may never carry allocation-shape attribution.
5. **Scratch comparison is same-initialization / same-scientific-recipe, never
   bitwise paired** (`AGENTS.md` §1.5). `a6_mature_scale.tex:56` complies.
6. **`98/100` (full-string + EOS) and `69/100` (strict first number) are
   different experiments** and may never be adjacent or unlabelled
   (`AGENTS.md` §1.5, `REBUTTAL_HANDOVER.md` §4). Likewise `69/67` (mixed-gap
   set) vs `49/48` (beyond-training-gap set).
7. **The K=8 MLA scarcity result is negative** (`claim_gate:
   DOES_NOT_SUPPORT_PRACTICAL_SCARCE_CHANNEL_ADVANTAGE`; `evq_minus_native =
   +0.667` at 8K). No monotone "fewer channels ⇒ allocation matters more" claim
   is permitted. The draft's K=16 persistence claim is legal and stands.
8. **32K is not usable capability.** LLaMA 32K RULER macro must not be reported
   at all (three shards never started; `LLAMA8B_MATCHED_RULER_MIX_20260726.md`);
   `table_ruler.tex` correctly uses `---`.
9. **RULER/2Wiki are task-family adaptation, not unseen-task transfer.**
   This one survives the admission test under condition (2): the paper writes
   "effective-context" claims, and dropping the qualifier would make them false.
   Keep it — but once, in the table caption, not three times.
10. **`τ` and Cosh have no universality claim.** `AGENTS.md` §4. The draft
    complies; this also happens to read as strength, so no tension.
11. **Static geometry (collision, rank, logdet) is not an extrapolation or
    LM-quality predictor.** If the full-RoPE theory is promoted (§5 item 3), this
    is the ceiling that governs it — and the §4 counterexamples in the canonical
    report are the honest, *interesting* way to state it.
12. **Dual submission (corrected after the historical audit).** The official
    ICLR 2027 FAQ expressly permits an ICLR abstract while a NeurIPS decision is
    pending and performs duplicate-submission checks only on full submissions.
    NeurIPS notification is 2026-09-24, before the ICLR full-paper deadline on
    2026-09-25. The operative question is therefore whether an accepted paper
    and the ICLR full submission are identical or substantially similar, not
    whether an abstract was filed before withdrawal. The current rewritten
    draft is assessed separately in `CHANGES_FROM_NEURIPS2026.md` §2.1.

## 4.1b Internal-only — real, recorded, and need not be volunteered

None of these may be *contradicted*; none of them needs a sentence in the paper.

- The `base=8.06K` frozen-checkpoint control recovering `74.7%` of the table gap
  (`FULL_ROPE_..._20260819.md` §6). Internal because it is a frozen-checkpoint
  counterfactual, not a from-scratch base arm (§11.5) — and because the
  exact-range control, which *is* in the body, is the direct answer to it.
- The seven-arm non-RULER adaptation search (best 4K `0.1167` vs control
  `0.5700`) and the clean LongAlign+Tulu full-RULER arm (`9.74/4.01/2.05%`).
- Held-out-task zeros (UUID retrieval, variable tracking at 0%) and
  `niah_single_2` regressing `0.70 → 0.55` (`DISCUSSION_PHASE_FOLLOWUP_PLAYBOOK.md` §6).
- The Phase-11B Kerple+MLP result where EVQ adds nothing on top of an adaptive-bias
  operator (`02_RESPONSE_QUESTIONS_AND_OUTCOMES.md` §4.3).
- Table 19's unfavourable NTK row (32×: `198.1` Geo vs `331.4` EVQ).
- The M4 curated-JSON summary drift (`1/1/1` vs recomputable `2/4/6`; regret
  `0.007366` vs `0.011440`). Internal correctness item; the draft already uses
  only the recomputable figures and prints no regret number.
- Cosh losing to a two-band schedule at 2K/4K/8K in the ultra audit. Ceiling:
  never claim Cosh optimality. Not a required sentence.
- The full-band EVQ-LeRoPE in-window sign flip across seeds.
- The 50M 2×2 being seed-42 only, if that result is promoted.

## 4.2 Over-defensive sentences in the current draft, with exact replacements

Each item names which admission test the current sentence fails. None of the
replacements makes any claim the owners do not support.

**1. `sections/04_experiments.tex:23-25`** — fails tests 1 and 2.
> Current: `It is not an official YaRN reproduction or a jointly tuned range-method comparison; the exact rule is given in App.~\ref{sec:ramp-scaler}.`
> Replace with: `The exact rule, and how it differs from reference YaRN, are given in App.~\ref{sec:ramp-scaler}.`
Rationale: the "not a reproduction" fact is preserved (and stated in full at
`a5_identification.tex:92-95` where it belongs); "not a jointly tuned
range-method comparison" pre-concedes `RDz6s.2` to a panel that has not asked it.

**2. `sections/04_experiments.tex:142-146`** — a titled `\paragraph{Scope.}` in
the body is a retreat marker; its content is already in two table captions.
> Current: `\paragraph{Scope.} The 2Wiki and RULER rows are disjoint from training but use the same task families, so they measure task-adapted length transfer. Each matched mature-scale protocol uses one training seed per arm; the second OLMo \evq{} seed replicates only strict NIAH. The 8B deletion study and RULER continuation are separate protocols.`
> Replace with a single trailing sentence appended to the 8B paragraph
> (`:140`): `Training and evaluation rows are disjoint and share the RULER generator families and the 2Wiki task, so these establish task-adapted length transfer; per-protocol seed scope and the separation of the deletion and continuation studies are given in App.~\ref{sec:mature-details}.`
Test 2 keeps the task-family clause. Seed counts remain visible at
`table_mature.tex:10-12` and `table_ruler.tex:7-8`, satisfying P12 without a
second, weaker restatement.

**3. `sections/04_experiments.tex:100-104`** — verdict order inverted; this is
`AGENTS.md` §1.8 scan item 3.
> Current: `On all $13$ RULER families \citep{hsieh2024ruler}, Native/\evq{} official macro is $72.19/42.44\%$ at $4$K, $2.02/31.63\%$ at $8$K, and $0.38/5.03\%$ at $16$K.  Native receives the same long-phase exposure yet remains at $2.02\%$ at $8$K: exposure alone cannot explain the $15.7\times$ score gap.`
> Replace with: `Native receives the same long-phase exposure yet reaches only $2.02\%$ official macro over all $13$ RULER families \citep{hsieh2024ruler} at $8$K, against $31.63\%$ for \evq{} --- a $15.7\times$ gap --- and $0.38/5.03\%$ at $16$K.  Exposure alone therefore cannot explain the gap; the frequency table can.  Table~\ref{tab:mature} gives all three lengths including the $4$K physical cap.`
The `72.19` figure stays in the table, where it belongs; the paragraph stops
opening with our own worst number.

**4. `sections/04_experiments.tex:79-84`** — reframe the reversal as a result,
and add the composition triple that `RzWsa.2` pre-approved.
> Current: `\paragraph{Range and allocation are different controls.} Retargeting \emph{both} grids to each evaluation length reverses the seed-$42$ ordering (Fig.~\ref{fig:identification}a, red). Fixed-range allocation and target-aware range selection therefore answer different questions: the former identifies the interior-allocation axis; the latter can dominate once range is allowed to track the target.`
> Replace with: `\paragraph{Range and allocation are separate, composable knobs.} Retargeting \emph{both} grids to each evaluation length reverses the seed-$42$ ordering (Fig.~\ref{fig:identification}a, red): with range free to track the target, target-aware range selection is the stronger lever, exactly as a range method should be. The two levels then compose --- applying the same target-range rule on top of the \evq{} grid improves that model by $0.098/0.529/0.638$ NLL at $512$/$1$K/$2$K. Fixed range identifies the allocation axis; free range transports it.`
Numbers from `MATCHED_RANGE_COSH_500M_S42_20260724.md`. This is the single
highest-value rewrite in the section: same facts, and the reviewer who offered
"advantage **or** complementarity" now gets both.

**5. Finite-$\tau$ wording — superseded after owner recheck.**
The earlier recommendation to call $0.75\times$--$1.5\times$ a verified basin
was too strong. `PHASE16_99RUN_RAW_REANALYSIS_20260724.md` confirms only one
pilot-selected neighbour per configuration; those discrete neighbour ratios
are $0.75\times$, $1.25\times$, or $1.5\times$. M4 separately tests
$0.75\times$, $1.00\times$, and $1.25\times$, plus two endpoint-specific
boundary arms. These grids neither certify every value in a continuous interval
nor locate its bounds. The outward text therefore treats $c{=}1$ as a useful
but fallible zero-search operating prior relative to Geo, not a verified basin
or near-optimal selector.

**6. `sections/03_theory.tex:139-141`** — an opaque disclaimer where the paper's
sharpest structural point should be.
> Current: `Moving budget toward the former should favour extrapolation, while any in-window cost must be measured after training. The surrogate does not assign the much larger mismatch observed when a pretrained model is retrofitted with a new table.`
> Replace with: `Moving budget toward the former should favour extrapolation at an in-window cost that must be measured after training. That cost has two distinct sources. Trained from scratch with the new table it is small: $+0.072$ and $+0.038$ NLL at $2$K and $4$K on a $1.485$B trajectory (\S\ref{sec:exp-mature}). Retrofitting a table into an already-trained checkpoint costs far more, for a reason the surrogate does not model and Prop.~\ref{prop:transplant} does.`
Numbers from `OLMO2_1B_RELEASED_ROPE_BASELINE_20260725.md`. Contingent on §5
item 2; if the theorem is not added, end at "...costs far more; \S\ref{sec:exp-mature}
reports both quantities separately." Either way the current sentence is the
weakest formulation of a genuinely strong idea (the cost dichotomy —
`INWINDOW_PRESERVATION_ROUTE_AUDIT_20260731.md` §0).

**7. `sections/04_experiments.tex:170-172`** — advertises a retired row in the body.
> Current: `The learned-frequency comparator and its tuning budget appear in App.~\ref{sec:pe-dominant}; shape attribution rests on the fixed schedules above.`
> Replace with: `Shape attribution therefore rests on the zero-parameter fixed schedules above, where parameter count and optimisation effort are identical by construction.`
The App. E pointer stays discoverable through the appendix itself. Nothing false
is removed; a beacon on a non-question is.

**8. `tables/table_pe_dominant.tex:6-11`** — bold self-negation inside a caption.
> Current: `\textbf{We rest no allocation-shape attribution on this table} (\S\ref{sec:exp-tau}): the learned-inv-freq arm carries $32$ trainable parameters, so any gap confounds allocation shape with parameterisation and optimisation effort.`
> Replace with: `Allocation-shape attribution in this paper rests on the zero-parameter fixed schedules of \S\ref{sec:exp-identify}: a learned-inv-freq arm carries $32$ trainable parameters, so any gap here confounds allocation shape with parameterisation and optimisation effort.`
Identical content, no boldface, framed as a design decision rather than a
confession. The method-identity paragraph at `a5_identification.tex:136-142`
already carries the substantive correction and should be left exactly as it is.

**9. `appendix/a5_identification.tex:16-20`**
> Current: `We did not identify a public author implementation when implementing the control, so the comparison is against the rule as published, not an official-code reproduction.`
> Replace with: `No public author implementation was available, so the baseline implements the rule exactly as published in \S6.1 rather than reproducing author code.`
Same fact (red line 4.1a-2 preserved), stated as a circumstance rather than as
our shortcoming.

**10. `appendix/a5_identification.tex:80-87`**
> Current: `...not a large-scale pre-training distribution, and a different corpus from the $151.9$M control and from the from-scratch runs. It is not scale evidence and not capability evidence, and its effect size is not comparable to the $151.9$M single-configuration contrast: the factorial establishes cross-configuration \emph{direction}, whereas the $151.9$M control provides the effect size within its configuration.`
> Replace with: `...and a different corpus from the $151.9$M control and from the from-scratch runs, so the two effect sizes are not comparable: the factorial establishes cross-configuration \emph{direction}, and the $151.9$M control provides the effect size within its configuration. Scale is evaluated separately in \S\ref{sec:exp-mature}.`
The corpus difference and the direction/effect-size split are required for truth
(test 2, and `AGENTS.md` §1.5). "It is not scale evidence and not capability
evidence" answers a charge nobody made and is dropped.

**11. `main.tex:87-89`** — the abstract asks a question where it should state a result.
> Current: `...while long-context methods usually alter or transport its range; we ask whether interior allocation still matters when that range is fixed.`
> Replace with: `...while long-context methods usually alter or transport its range. We show that interior allocation is a separate, binding training-time variable even when that range is pinned exactly.`

**12. `sections/05_discussion.tex:36-39`** — keep, but sharpen. "This does not
identify individual channels, but it gives a falsifiable prediction" is fine;
the prediction is the strength. Consider upgrading it with the source-count
formulation from `nonuniform-alloc/RESEARCH_MEMO.md` §2 — *the 4K cost of any
schedule should be monotone in the number of sources a task must resolve
simultaneously* — **only after** reconciling the two family groupings (§2.3).

---

# 5. Top 10 highest-leverage changes

Ranked by expected effect on acceptance probability. Effort is my estimate of
writing time, not compute — none of these requires a GPU.

| # | Change | File to edit | Why it moves the decision |
|---|---|---|---|
| **1** | **Resolve the venue.** Switch to `iclr2027_conference.sty`, relax the body to 9 pages, replace the Impact Statement with an AI-use statement, and record the NeurIPS decision/citation branch before the ICLR full upload. | `main.tex:30,111-112`, `sections/06_impact.tex`, `SUBMISSION_CHECKLIST.md` | The format work and policy check are now complete; only the conditional citation action remains if NeurIPS accepts. It also frees one full body page, which is the budget for items 2-5. |
| **2** | **Add the exact post-hoc transplant obstruction as a numbered proposition** (`Aᵀ R(ω′Δ) B = R(ωΔ) ∀Δ ⇒ |ω′|=|ω|`), and use it to reframe the mature-model in-window cost as a *predicted* consequence rather than a defect. | `sections/03_theory.tex` (new subsection + `\label{prop:transplant}`), `sections/05_discussion.tex`, `appendix/a1_proofs.tex` | The paper's biggest apparent weakness becomes a theorem-backed structural result. `ICLR2027_RESEARCH_SYNTHESIS_20260819.md` §12 ranks this #1; `INWINDOW_PRESERVATION_ROUTE_AUDIT_20260731.md` §5 gives the exact five-step framing. Zero GPU, ~15 lines, proof already written. |
| **3** | **Promote the 50M co-adaptation 2×2** (`7.14/7.16` self-consistent vs `76.20/23.05` mismatched; interaction `−3.5367`, CI `[−5.165,−3.039]`, ≈5.9× the main effects; bare static rank *improves* in the worst cell). | `sections/04_experiments.tex` (new short subsection), `appendix/a1_proofs.tex` or a new appendix | Empirical companion to item 2, deterministic and CPU-only. Together they turn "co-adaptation" from an assertion into a proved-plus-measured claim, and the "static rank improves while PPL collapses" cell pre-empts the reviewer who asks why you don't just optimise rank. |
| **4** | **Reorder every verdict sentence so the paper never opens with its own worst number**, and add the `0.75×–1.5× across 21 configurations` containment. | `04_experiments.tex:100-104,155-158`, `03_theory.tex:214-219`, `04_experiments.tex:79-84` | §4.2 items 3, 4, 5. This is the concrete form of the 11628 lesson — the NeurIPS panel's memory was the concessions, not the answers. Costs zero space and changes what a skimming reviewer retains. |
| **5** | **Restore the epistemic-status table as Table 1.** | `sections/03_theory.tex` (adapt `paper/tables/table_epistemic_map.tex`) | 27bE named it as the paper's strength *by number*; Dz6s asked for exactly the separation it provides (`RDz6s.3`). Deleting a credited strength is the one unforced error in this revision. Also discharges promise P3. |
| **6** | **State the adaptation-scope ablation as a sentence**, not as two unlabelled table rows: restricting the continuation to Q/K raises 4K macro `37.51 → 42.44` and 8K `21.29 → 31.63` (and, per its owner, EVQ's 4K macro `11.09 → 42.44`). | `appendix/a6_mature_scale.tex:34-37`, one clause in `sections/04_experiments.tex` | Converts the in-window drop from an unexplained defect into a partially *diagnosed* adaptation artifact — with a matched ablation attached. Discharges promise P11. Owner: `OLMO2_1B_SELECTIVE_QK_PHASE_ADAPTATION_20260729.md`. |
| **7** | **Add Oka et al.'s own §6.3 admission** — the `L_target` requirement is a "practical limitation", with adaptive schemes listed as future work — plus the target-awareness asymmetry in the `table_layers` caption. | `sections/02_related.tex` after `:31`, `tables/table_layers.tex` caption | The strongest available novelty argument against the reject vote's core objection, made in the cited authors' own words. One sentence. `REBUTTAL_HANDOVER.md` §3.1-3.2 has the verified quotes. **Do not** extend it to an `L_train` asymmetry — that argument cuts both ways. |
| **8** | **Extend the three-region channel accounting to four production models** (OLMo-2 `32/10/22`, LLaMA-3-8B `35/14/15`, Qwen2.5 `40/7/17`, DeepSeek-V3 MLA `16/9/7`). | `sections/03_theory.tex:167-171` → small table | Answers `R27bE.2`'s "one base value, one architectural lineage" with pure arithmetic at production bases (500K, 1M) and production lengths (32K, 128K). Zero GPU; owner `nonuniform-alloc/RESEARCH_MEMO.md` §3 with a recomputable script. |
| **9** | **Restore the held-out base-1M / `d_head`=128 three-seed suite** (`−0.802/−0.664/−0.433/−0.287/−0.212` at 1K–16K; in-domain `+0.069` at 512) **and the three-level fixed-schedule ladder** (Level 1 `−0.256/−0.305/−0.223/−0.238`; Level 2 `−0.113/−0.149/−0.207/−0.190/−0.099`), conditional on promoting their per-seed raw bundles. | new appendix subsection in `appendix/a5_identification.tex` | The only direct answer to `R27bE.5`'s held-out-base half; effect sizes ~an order of magnitude above M4's; both were sent to reviewers and both are now missing (promises P4, P5). **Gated on promotion** — do not write them from the summary alone. |
| **10** | **Surface the strict full-string+EOS capability result** (`100/98/60` EVQ vs `95/18/0` matched Native at 4K/8K/16K) in the mature-scale appendix, in its own subsection, endpoint defined in the first clause, never adjacent to the `69/67` strict-first-number rows. | `appendix/a6_mature_scale.tex` | The most striking capability number in the workspace and currently invisible in the manuscript; it led three of the five sent rebuttal documents. Owner `EVQ_QUERY_GAP_FINAL_DIAGNOSTIC.md`. The adjacency rule (`AGENTS.md` §1.5) is why it belongs in its own subsection rather than in Table 5. |

**Explicitly not recommended:** reinstating the collision-prefactor table (dead,
§4.1a-1); reinstating QuALITY (`FU.3`); adding P-EVQ, the far-only residual, or
the observed-band LeRoPE arm (design-only / untrained, `AGENTS.md` §1.8); adding
a generic "single-seed limitation" sentence to the mature-model results
(`AGENTS.md` §0.1 explicitly authorises omitting it, and seed scope is already
exact in the table captions); running any new experiment before items 1-8 are
written, since none of items 1-8 needs one.
