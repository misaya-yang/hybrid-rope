# Audit 02 — Cross-Reference + Label Completeness

**Auditor**: 2/8 (parallel)
**Scope**: paper/main.tex, paper/sections/, paper/appendix/, paper/tables/
**Read-only**. No source modifications.
**Source state**: HEAD = `a358d6d`. main.tex 2026-04-27 09:09. main.pdf 2026-04-27 09:09. main.aux/main.log are stale (2026-04-24 20:22, see §"Build artefact note" below).

---

## Map summary

- **Files scanned**: 1 main + 5 body sections (01_intro, 02_related, 03_theory, 05_experiments, 06_limitations) + 4 appendix (a1_proofs, a2_experiment_details, a3_supporting_results, a4_supporting_experiments) + 10 tables.
- **Distinct labels defined**: 107 (excluding commented-out labels marked `% \label{...} % unused`).
- **Distinct labels referenced**: 70.
- **Total ref/eqref/Cref calls**: 124 (51 a1 + 10 a2 + 6 a3 + 10 a4 + 2 main + 2 intro + 4 related + 37 theory + 21 experiments + 0 limitations + table-caption refs).
- **Broken refs**: 0.
- **Duplicate labels**: 0.
- **Dead labels (defined but no `\ref` / `\eqref` / `\Cref`)**: 37 (mostly in appendix a1; analyzed below).
- **`??` placeholders in current main.pdf**: 0 (verified via `pdftotext main.pdf | grep '??'`).

Per-file label/ref counts (defined / referenced):
- main.tex: 1 / 2
- sections/01_intro: 0 / 2
- sections/02_related: 0 / 4
- sections/03_theory: 11 / 37
- sections/05_experiments: 7 / 21
- sections/06_limitations: 1 / 0
- appendix/a1_proofs: 60 / 51
- appendix/a2_experiment_details: 8 / 10
- appendix/a3_supporting_results: 6 / 6
- appendix/a4_supporting_experiments: 4 / 10
- tables/*: 10 labels / 21 refs (across all 10 table files combined)

---

## Findings

### Broken refs (P0)

**None.** All 70 referenced labels resolve to a definition in paper/.

### `??` placeholders in PDF (P0)

**None.** `pdftotext paper/main.pdf | grep '\?\?'` returns 0 hits.

### Duplicate labels (P0)

**None.** Every label name has exactly one defining `file:line`.

### Dead labels — defined but never referenced (P1)

The 37 orphans below are defined in source but no `\ref{}` / `\eqref{}` / `\Cref{}` (or equivalent) targets them anywhere in paper/. PDF still renders the underlying object correctly (since `\label` just records); the labels are simply load-bearing. Most are P2-stylistic; the few P1-flagged below have a small risk of confusing readers who follow named anchors.

Equation labels (all in `paper/appendix/a1_proofs.tex` — surrogate / waterbed / discrete-continuous / pure-tether / leff blocks):

| label | file:line | severity |
|---|---|---|
| `eq:rho-tau-closed` | appendix/a1_proofs.tex:38 | P2 |
| `eq:min-kernel-psd` | appendix/a1_proofs.tex:45 | P2 |
| `eq:forced-ode` | appendix/a1_proofs.tex:55 | P2 |
| `eq:waterbed-functional` | appendix/a1_proofs.tex:164 | P2 |
| `eq:waterbed-tight` | appendix/a1_proofs.tex:173 | P2 |
| `eq:waterbed-corollary` | appendix/a1_proofs.tex:178 | P2 |
| `eq:D-B-evq` | appendix/a1_proofs.tex:188 | P2 |
| `eq:S-chi2-evq` | appendix/a1_proofs.tex:189 | P2 |
| `eq:M-rho-evq` | appendix/a1_proofs.tex:190 | P2 |
| `eq:waterbed-evq-scale` | appendix/a1_proofs.tex:195 | P2 |
| `eq:green-identity` | appendix/a1_proofs.tex:219 | P2 |
| `eq:tau-leff` | appendix/a1_proofs.tex:476 | P2 |
| `eq:eta-F` | appendix/a1_proofs.tex:501 | P2 |
| `eq:delta-rho-F-pointwise` | appendix/a1_proofs.tex:513 | P2 |
| `eq:tether-L2-correct` | appendix/a1_proofs.tex:519 | P2 |
| `eq:warp-bound` | appendix/a1_proofs.tex:526 | P2 |
| `eq:RF` | appendix/a1_proofs.tex:533 | P2 |
| `eq:tether-deficit` | appendix/a1_proofs.tex:539 | P2 |
| `eq:chi2-load-identity` | appendix/a1_proofs.tex:553 | P2 |
| `eq:lambda-true` | appendix/a1_proofs.tex:567 | P2 |
| `eq:wasserstein-bound` | appendix/a1_proofs.tex:586 | P2 |
| `eq:density-bound` | appendix/a1_proofs.tex:591 | P2 |
| `eq:evq-discrete-bound` | appendix/a1_proofs.tex:597 | P2 |
| `eq:kernel-bound` | appendix/a1_proofs.tex:603 | P2 |
| `eq:high-res-distortion` | appendix/a1_proofs.tex:610 | P2 |

Section-anchor and theorem labels:

| label | file:line | severity | note |
|---|---|---|---|
| `sec:broader-impact` | main.tex:68 | P2 | Standalone NeurIPS appendix anchor; convention to keep. |
| `sec:conclusion-limitations` | sections/06_limitations.tex:2 | P1 | The body §5 "Conclusion and Limitations" anchor. Body uses §5 implicitly; appendix never `\ref`'s back. P1 because dropping `04_predictions.tex`/`07_conclusion.tex` from the input list (Round 5) likely orphaned this. |
| `sec:exp-setup` | sections/05_experiments.tex:5 | P2 | §4.1 anchor, never cross-referenced. Safe. |
| `sec:waterbed` | sections/03_theory.tex:86 | P2 | §3.6 anchor. The text references *Proposition 1* and *Appendix `sec:waterbed-proof`* instead of pointing at the body subsection. Stylistically fine. |
| `sec:leff-remark` | sections/03_theory.tex:112 | P2 | Diagnostic-extension paragraph; not cited. |
| `sec:floor-higher-order` | appendix/a1_proofs.tex:141 | P2 | Section header for the floor proposition; only the proposition itself is referenced (also dead — see below). |
| `prop:floor-higher-order` | appendix/a1_proofs.tex:146 | P2 | Proposition stated but never cited. |
| `sec:self-consistency` | appendix/a1_proofs.tex:200 | P2 | Section anchor for the self-consistency theorem; not cited. |
| `thm:self-consistency` | appendix/a1_proofs.tex:205 | P2 | Theorem stated but never cited. |
| `sec:discrete-continuous-gap` | appendix/a1_proofs.tex:579 | P2 | Section anchor (a1 §A.16); not cited. |
| `fig:attn-mechanism` | appendix/a3_supporting_results.tex:56 | P1 | Figure 6 in PDF, with a real `\label`, but body text never `\ref`'s to "Figure 6"; the surrounding paragraph in a3 references it only via the natural-language caption. **Probable bug**: §3.6 / §A waterbed text discusses this mechanism without a back-link. |
| `tab:pe-yarn-l256` | appendix/a4_supporting_experiments.tex:15 | P1 | Table 23 in PDF (`L=256` YaRN composition study). Listed in the task's expected appendix-coverage set. The caption itself supplements `tab:pe-dominant`, but no body or appendix paragraph runs `\ref{tab:pe-yarn-l256}`. **Probable bug**: §4.3 says "A complementary τ-sweep at Ltrain=256 (Appendix Fig.~\ref{fig:pe-dominant}) confirms..." — the *figure* is referenced but the *table* with the actual numbers is not. |

### Number sequence gaps (P1)

**Figures (PDF order)**: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10. No gaps.
- Figure 7 ("750M frequency dynamics") has its `\label{fig:freq-dynamics}` commented out (`% \label{...} % unused` at appendix/a3_supporting_results.tex:63) — figure renders correctly but cannot be `\ref`'d. Acceptable since it is never referenced. P2.

**Tables (PDF order)**: 1 through 25, sequential. No gaps.
- Table 13 in PDF is "750M continued pretraining" — its `\label{tab:750m-supporting}` is commented out (`tables/table6_750m_continue_supporting.tex:3`). Same story as fig:freq-dynamics: renders correctly, no `\ref`'s.

**fig:method-overview → fig:evq-yarn → fig:pe-dominant** are PDF figures 1, 2, and 10 respectively (all reachable; intermediate figures 3-9 are appendix figures, also reachable).

---

## \input audit (P1)

**main.tex \input order (lines 49-53 body, 62-65 appendix)**:
```
49  \input{sections/01_intro}     -> §1 Introduction
50  \input{sections/02_related}   -> §2 Related Work
51  \input{sections/03_theory}    -> §3 Theory and Method
52  \input{sections/05_experiments} -> §4 Experiments
53  \input{sections/06_limitations} -> §5 Conclusion and Limitations
[bib]
62  \input{appendix/a1_proofs}
63  \input{appendix/a2_experiment_details}
64  \input{appendix/a3_supporting_results}
65  \input{appendix/a4_supporting_experiments}
```

- **Body section sequence**: 1, 2, 3, 4, 5. **PASS** (sensible).
- **04_predictions.tex** and **07_conclusion.tex** are intentionally NOT \input'd. **CONFIRMED**:
  - `sections/04_predictions.tex` contains only `% Content merged into §3 Theory (subsection 3.8 Predictions)`.
  - `sections/07_conclusion.tex` contains only `% Conclusion merged into §5 (Conclusion and Limitations) to fit 9-page body.`.
- The handover statement that these are stubs and not in the input list is correct.
- No body section number is skipped. The `0X_*.tex` filename convention is informational, not LaTeX-binding.

---

## Appendix-label coverage table (task §3)

Every required label EXISTS. Body-citation status (whether body sections `paper/sections/*.tex` cite the label) below:

| label | defined? | file:line | body-cited? | citing body locations |
|---|---|---|---|---|
| sec:proofs | Yes | appendix/a1_proofs.tex:2 | Yes | sections/03_theory.tex:37; :41 |
| sec:lambda-curvature | Yes | appendix/a1_proofs.tex:562 | No | (cited from tables/table_lambda_cv.tex:2 — internal cross-link) |
| sec:surrogate-validation | Yes | appendix/a1_proofs.tex:100 | Yes | sections/03_theory.tex:15; :32 |
| sec:mechanism-isolation | Yes | appendix/a1_proofs.tex:408 | Yes | sections/03_theory.tex:32 |
| sec:pure-tether-bound | Yes | appendix/a1_proofs.tex:494 | Yes | sections/03_theory.tex:48 |
| subsec:fisher-ext | Yes | appendix/a1_proofs.tex:50 | Yes | sections/03_theory.tex:64 |
| sec:waterbed-proof | Yes | appendix/a1_proofs.tex:159 | Yes | sections/03_theory.tex:88 |
| sec:leff-derivation | Yes | appendix/a1_proofs.tex:441 | Yes | sections/03_theory.tex:88; :113 |
| sec:stiffness-derivation | Yes | appendix/a1_proofs.tex:356 | Yes | sections/03_theory.tex:93; :103 |
| sec:chi2-load | Yes | appendix/a1_proofs.tex:546 | Yes | sections/03_theory.tex:113 |
| sec:tau-scaling | Yes | appendix/a1_proofs.tex:278 | Yes | sections/03_theory.tex:115 |
| sec:lambda-cv | Yes | appendix/a1_proofs.tex:243 | Yes | sections/03_theory.tex:115 |
| sec:mla-results | Yes | appendix/a3_supporting_results.tex:4 | Yes | sections/03_theory.tex:19; :81; sections/05_experiments.tex:11; :51 |
| sec:supporting-experiments | Yes | appendix/a4_supporting_experiments.tex:2 | Yes | sections/05_experiments.tex:9; :11 |
| sec:tau-correction | Yes | appendix/a2_experiment_details.tex:107 | No | (cited from a1_proofs.tex:328 and a4_supporting_experiments.tex:58) |
| sec:lora-regime | Yes | appendix/a1_proofs.tex:393 | No | (cited from a4_supporting_experiments.tex:51) |
| tab:lambda-cv | Yes | tables/table_lambda_cv.tex:3 | Yes | sections/03_theory.tex:113 |
| tab:multiscale | Yes | tables/table1_multiscale_raw_ppl.tex:3 | Yes | sections/05_experiments.tex:51; :61 |
| tab:capability | Yes | tables/table3_capability_passkey.tex:3 | Yes | sections/05_experiments.tex:61 |
| tab:evidence-tier | Yes | tables/table_evidence_tier.tex:7 | Yes | sections/01_intro.tex:9; sections/05_experiments.tex:11 |
| tab:method-comparison | Yes | tables/table_method_comparison.tex:7 | Yes | sections/02_related.tex:3; :17 |
| tab:epistemic-map | Yes | tables/table_epistemic_map.tex:5 | Yes | sections/03_theory.tex:11 |
| tab:phase11-leverage | Yes | tables/table5_phase11_leverage.tex:3 | Yes | sections/05_experiments.tex:23 |
| tab:evq-yarn | Yes | tables/table2_evq_yarn_main.tex:3 | Yes | sections/05_experiments.tex:21; :23 |
| tab:pe-dominant | Yes | tables/table4_pe_dominant.tex:3 | Yes | sections/05_experiments.tex:41 |
| tab:pe-yarn-l256 | Yes | appendix/a4_supporting_experiments.tex:15 | **No** | **DEAD — never referenced** |
| tab:lora-8b | Yes | appendix/a4_supporting_experiments.tex:37 | No | (cited from a1_proofs.tex:403 and a4_supporting_experiments.tex:51) |
| tab:quality-nll | Yes | appendix/a3_supporting_results.tex:73 | No | (cited from a3_supporting_results.tex:69 and a4_supporting_experiments.tex:61) |
| tab:surrogate-validation | Yes | appendix/a1_proofs.tex:108 | No | (cited from a1_proofs.tex:104) |
| fig:orthogonal | Yes | appendix/a1_proofs.tex:274 | Yes | sections/03_theory.tex:115; sections/05_experiments.tex:23 |
| fig:tau-sweep | Yes | appendix/a1_proofs.tex:322 | No | (cited from a1_proofs.tex:97; :284; :316) |
| fig:method-overview | Yes | sections/03_theory.tex:8 | Yes | sections/03_theory.tex:19 |
| fig:evq-yarn | Yes | sections/05_experiments.tex:29 | Yes | sections/05_experiments.tex:21; :23 |
| fig:pe-dominant | Yes | appendix/a4_supporting_experiments.tex:10 | Yes | sections/05_experiments.tex:41 |

The task-required set: 33 / 34 labels are referenced *somewhere* in paper/. The single fully orphan label is `tab:pe-yarn-l256` (Table 23 in PDF).

---

## Intra-appendix consistency (task §5)

For each appendix file, every `\ref` / `\eqref` / `\Cref` resolves to a label defined in paper/:

| appendix file | refs issued | unresolved |
|---|---|---|
| a1_proofs.tex | 51 | **0** |
| a2_experiment_details.tex | 10 | **0** |
| a3_supporting_results.tex | 6 | **0** |
| a4_supporting_experiments.tex | 10 | **0** |

All clear. **No broken intra-appendix refs.**

---

## Build artefact note (informational, not a finding)

`paper/main.aux` and `paper/main.log` are dated 2026-04-24 20:22 — three days older than `paper/main.pdf` (2026-04-27 09:09). The stale `main.log` contains:

```
LaTeX Warning: Reference `sec:exp-supporting' on page 7 undefined on input line ...   (×4)
LaTeX Warning: There were undefined references.
```

These warnings refer to a since-removed body subsection (Round 5 deleted `\subsection{Supporting evidence}\label{sec:exp-supporting}` from `05_experiments.tex`). Confirmed via `grep -rn 'exp-supporting' paper/` — only stale `main.log` and `EDIT_CHANGELOG.md` mention it; neither current source nor current PDF contains a reference.

`paper/main.aux` also contains `\newlabel{tab:mla-reversal}{{6}{9}{...}}` — a label that no longer exists in source either. Confirmed gone.

**Recommendation**: re-run `pdflatex; bibtex; pdflatex; pdflatex` once before submission so `main.aux` / `main.log` reflect the actual `main.tex`. This is **not** a content bug — the current `main.pdf` is correct.

---

## Summary

- **0 P0 findings** (no broken refs, no duplicates, no `??` in PDF).
- **2 P1 findings** worth fixing before submission:
  1. `tab:pe-yarn-l256` (Table 23, "L=256 YaRN composition study") is defined at `appendix/a4_supporting_experiments.tex:15` but **never referenced**. §4.3 (`sections/05_experiments.tex:41`) says "*A complementary τ-sweep at Ltrain=256 (Appendix Fig.~\ref{fig:pe-dominant}) confirms…*" — the figure visualization is cited but the underlying *table* with the numbers is not. **Recommended fix**: in `sections/05_experiments.tex:41`, after "Appendix Fig.~\ref{fig:pe-dominant}", add "(numbers in Appendix Table~\ref{tab:pe-yarn-l256})". Or in `a4_supporting_experiments.tex` body text, add a "Table~\ref{tab:pe-yarn-l256} reports..." sentence.
  2. `fig:attn-mechanism` (Figure 6, "Waterbed verification in learned attention patterns 750M") is labeled at `appendix/a3_supporting_results.tex:56` but never referenced anywhere in paper/. The figure renders standalone in §A.7 of appendix. The body §3.6 / §3.7 on the waterbed mechanism could plausibly back-link here. **Recommended fix**: optional — either drop the label (matches `% \label{fig:freq-dynamics} % unused` convention used for orphan labels in same file), or add `\ref{fig:attn-mechanism}` in the appendix paragraph that introduces the figure (currently §A.7 §"Attention distance redistribution" introduces Figure 6 only via natural-language proximity).
  3. (Bonus) `sec:conclusion-limitations` at `sections/06_limitations.tex:2` is also unused, but body §5 is implicitly addressable, so this is closer to P2.
- **35 P2 findings**: dead equation/section/theorem labels in `appendix/a1_proofs.tex`. These are fine to keep — orphan equation labels are common in long proofs and don't affect the rendered PDF — but a future cleanup pass could comment them out following the `% \label{...} % unused` convention already established in `a3_supporting_results.tex:63`, `a3_supporting_results.tex:67`, `a2_experiment_details.tex:17,66,75,199`, and `tables/table6_750m_continue_supporting.tex:3`.
- **\input order**: correct. Body produces sections 1, 2, 3, 4, 5; `04_predictions.tex` and `07_conclusion.tex` are confirmed empty stubs not in the input list.
- **Intra-appendix refs**: all 77 internal references resolve.
- **Build artefacts**: stale `main.aux`/`main.log` from 2026-04-24 do not match the 2026-04-27 `main.pdf`. Not a content bug, but a fresh `pdflatex` run before submission is recommended so the warnings (which reference obsolete labels) clear.

**Bottom line**: cross-reference integrity of the camera-ready paper is **clean** at the P0 level. The two P1 items above are stylistic / discoverability nits, not correctness bugs — the PDF reads correctly without them. Recommend the small `tab:pe-yarn-l256` body-citation insertion before submission since this is one of the few defined-but-unreferenced numbered floats and the body section discusses exactly its content.
