# Audit v2 / 04 — Cross-Reference + Label Completeness (Round-2 regression)

**Auditor**: 4/6 (parallel)
**Scope**: paper/main.tex, paper/sections/, paper/appendix/, paper/tables/, paper/refs/
**Read-only**. No source modifications.
**Source state**: HEAD = `616edd7`. Working-tree has uncommitted edits in `paper/appendix/a1_proofs.tex`, `paper/sections/03_theory.tex`, `paper/sections/05_experiments.tex`, and `paper/main.pdf` (the round-2 edits this audit is reviewing). main.pdf timestamp 2026-04-27 16:38; latest source 2026-04-27 16:27 (a1_proofs.tex). PDF is newer than all sources — reflects the working-tree edits.

---

## Methodology

1. Built a label-defining map from every uncommented `\label{...}` across `paper/{main.tex,sections/,appendix/,tables/}` (total 108 active labels, 9 explicitly commented-out as `% \label{...} % unused`).
2. Built a ref-using map from every `\ref{...}`, `\eqref{...}`, `\Cref{...}`, `\cref{...}` (total 103 ref calls, 75 distinct keys).
3. Built a cite map from every `\cite/\citep/\citet/\citealp/\citealt{...}` (44 distinct bibkeys) and intersected with the 44 entries in `paper/refs/references.bib`.
4. Confirmed the prior-round dead-label list against the new state via per-key `grep` count.
5. PDF placeholder check: `pdftotext paper/main.pdf - | grep -F -c '??'` (used `-F` to disable regex — earlier `?\?` regex returns false positives).
6. \input order verified by reading `paper/main.tex:49-65` and `wc -l` on the would-be-skipped stubs.
7. Bib coverage: `comm -23 cite_keys bib_keys` gives missing bibkeys; `comm -13` gives unused bib entries.

Source diff vs prior audit: `git diff a358d6d HEAD` plus `git diff HEAD` (working tree). Round-2 edits live in two layers: (a) commit `616edd7` (caption metadata + std insertion + theory consistency), and (b) uncommitted working-tree changes (the new "Why constant α" subsection + §3.2 K_app reframing + §3.7 Q1>0 sentence).

---

## Map summary (current state, including working-tree edits)

| metric | count | vs prior round |
|---|---|---|
| Distinct active labels defined | 108 | -9 net (3 P1 cleared: tab:pe-yarn-l256/fig:attn-mechanism cleared, sec:conclusion-limitations removed; 1 added: sec:why-constant-alpha + 1 added: eq:const-vs-sp; many others changed cosmetically) |
| Distinct ref keys used | 75 | +5 (Q1 grid ref, KL convention ref, ref to tab:pe-yarn-l256, ref to fig:attn-mechanism, ref to eq:const-vs-sp) |
| Total ref/eqref/Cref calls | 103 | -21 net (commented-out checklist references) |
| Total cite calls | 44 distinct bibkeys | unchanged |
| Bib entries in references.bib | 44 | unchanged |
| Broken refs | **0** | unchanged |
| Duplicate labels | **0** | unchanged |
| `??` placeholders in main.pdf | **0** | unchanged |
| Dead labels (defined, no `\ref`) | **33** | -2 (35 → 33: two equation labels awakened, one section label added) |

Per-file label/ref counts (active labels / ref calls):

| file | labels | refs |
|---|---|---|
| paper/main.tex | 1 | 1 |
| sections/01_intro.tex | 0 | 1 |
| sections/02_related.tex | 0 | 4 |
| sections/03_theory.tex | 11 | 16 |
| sections/05_experiments.tex | 7 | 8 |
| sections/06_limitations.tex | 0 | 0 (now 1-line comment-only stub) |
| appendix/a1_proofs.tex | 62 | 36 |
| appendix/a2_experiment_details.tex | 8 | 8 |
| appendix/a3_supporting_results.tex | 6 | 7 |
| appendix/a4_supporting_experiments.tex | 4 | 7 |
| tables/* (10 files) | 9 active | 15 |

---

## NEW labels and refs (this round)

### `sec:why-constant-alpha` — NEW orphan (P1)

**Status**: defined exactly once at `paper/appendix/a1_proofs.tex:100` (working-tree only; not yet committed). **No `\ref{sec:why-constant-alpha}` exists anywhere in `paper/`.**

§3.2 (the K_app reframing paragraph after Action 6) at `paper/sections/03_theory.tex:32` instead contains the natural-language anchor:
> ... see Appendix~\ref{sec:proofs} ``Why constant $\alpha$''

i.e., it links to the umbrella Appendix A (`sec:proofs`) and uses the subsection title as a textual hint, **not** a clickable anchor to the precise §A.5 (the rendered subsection number per `pdftotext paper/main.pdf` line 1756).

**Severity**: **P1** (not P0). Reasoning:
1. The label is not broken; it is simply not pointed-to. Readers can still navigate via the umbrella `sec:proofs` jump + textual title match, which the rendered PDF does correctly (line 558–560: "see Appendix A `Why constant α`").
2. The new subsection is internally cited via `\eqref{eq:const-vs-sp}` (a1_proofs.tex:108), so its content is not a hanging end node.
3. It IS load-bearing in the sense that §3.2 was rewritten this round specifically to defer the constant-α justification to this subsection; a precise anchor would aid reviewer click-through.

(a) Defined exactly once: **PASS** — single defining file:line.
(b) §3.2 cites this label: **FAIL** — §3.2 cites `sec:proofs`, not `sec:why-constant-alpha`. Recommended fix: change `Appendix~\ref{sec:proofs} ``Why constant $\alpha$''` to `Appendix~\ref{sec:why-constant-alpha}` at `paper/sections/03_theory.tex:32`.
(c) No duplicate: **PASS** — only one definition site.

### Other new ref targets (all resolve)

| target label | defining file:line | new citing file:line | resolves? |
|---|---|---|---|
| `sec:lambda-cv` | appendix/a1_proofs.tex:254 | sections/03_theory.tex:108 ("Q_1>0 verified across the tested grid... Appendix~\ref{sec:lambda-cv}") | **PASS** — also previously cited at 03_theory.tex:115 |
| `sec:lambda-curvature` | appendix/a1_proofs.tex:573 | sections/03_theory.tex:93 ("convention $\tfrac{1}{2}$ factors from the KL Taylor expansion are absorbed into $\lambda$, Appendix~\ref{sec:lambda-curvature}") | **PASS** — also cited at tables/table_lambda_cv.tex:2 |
| `eq:const-vs-sp` | appendix/a1_proofs.tex:106 | appendix/a1_proofs.tex:108 (self-citation in same paragraph: "$\eqref{eq:const-vs-sp}$ is the precise statement...") | **PASS** |
| `eq:rho-tau-closed` (was dead in prior round) | appendix/a1_proofs.tex:38 | appendix/a1_proofs.tex:40 ("the $\tau{\to}0$ limit of~\eqref{eq:rho-tau-closed}") | **PASS** — newly awakened |
| `eq:S-chi2-evq` (was dead in prior round) | appendix/a1_proofs.tex:200 | appendix/a1_proofs.tex:197 ("Eq.~\eqref{eq:S-chi2-evq} below uses the un-normalized integral") | **PASS** — newly awakened |

PDF render confirmation (from `pdftotext paper/main.pdf`):
- Line 824: `Q1(L, b)>0 ... Appendix A.10` (= `sec:lambda-cv` per appendix numbering).
- Line 771: `KL Taylor expansion are absorbed into λ (Appendix A.18)` (= `sec:lambda-curvature` per appendix numbering).
- Line 1111: `numbers in Appendix Table 23` (= `tab:pe-yarn-l256`).
- Line 4835: `Figure 6 illustrates the waterbed crossover...` (= `fig:attn-mechanism`).

---

## Resolved-this-round labels (prior P1 → cleared)

All three prior P1 dead labels are resolved or removed:

| prior P1 issue | prior file:line | round-2 status |
|---|---|---|
| `tab:pe-yarn-l256` (Table 23, never referenced) | appendix/a4_supporting_experiments.tex:15 | **RESOLVED** — `sections/05_experiments.tex:41` now reads "Appendix Fig.~\ref{fig:pe-dominant}, numbers in Appendix Table~\ref{tab:pe-yarn-l256}". Verified in PDF L1111. (Fix landed in commit 616edd7.) |
| `fig:attn-mechanism` (Figure 6, never referenced) | appendix/a3_supporting_results.tex:58 | **RESOLVED** — appendix/a3_supporting_results.tex:44 added introducing paragraph: "Figure~\ref{fig:attn-mechanism} illustrates the waterbed crossover in learned attention patterns of the 750M continue@4K checkpoint, complementing the surrogate-level analysis in \S\ref{sec:waterbed-proof}." Verified in PDF L4835. |
| `sec:conclusion-limitations` (was at 06_limitations.tex:2) | sections/06_limitations.tex:2 | **CLEARED BY DELETION** — `06_limitations.tex` is now a 1-line comment-only stub: `% Conclusion + Limitations merged inline at end of §5.4 for body 9-page budget.`. The label no longer exists. \input is still present in `main.tex:53` but the file emits no content. |

---

## Awakened-prior-dead-labels check

The 35 prior-round dead labels were verified for awakenings (i.e., a new `\ref` added that points to one of the previously orphan labels). Two awakenings, both legitimate (not accidental):

| label | prior file:line | new citing site | new ref text |
|---|---|---|---|
| `eq:rho-tau-closed` | a1_proofs.tex:38 | a1_proofs.tex:40 | "...is also the $\tau{\to}0$ limit of~\eqref{eq:rho-tau-closed}." (added in commit 616edd7 along with PSD/convexity tightening) |
| `eq:S-chi2-evq` | a1_proofs.tex:200 | a1_proofs.tex:197 | "Eq.~\eqref{eq:S-chi2-evq} below uses the un-normalized integral" (added in commit 616edd7 to disambiguate normalization convention) |

Neither awakening introduces a broken ref or duplicate label. All other 33 prior dead labels remain dead — no accidental awakenings.

The remaining 33 dead labels (P2 unless noted):

- **Equation labels in a1_proofs.tex (23)**: `eq:min-kernel-psd`, `eq:forced-ode`, `eq:waterbed-functional`, `eq:waterbed-tight`, `eq:waterbed-corollary`, `eq:D-B-evq`, `eq:M-rho-evq`, `eq:waterbed-evq-scale`, `eq:green-identity`, `eq:tau-leff`, `eq:eta-F`, `eq:delta-rho-F-pointwise`, `eq:tether-L2-correct`, `eq:warp-bound`, `eq:RF`, `eq:tether-deficit`, `eq:chi2-load-identity`, `eq:lambda-true`, `eq:wasserstein-bound`, `eq:density-bound`, `eq:evq-discrete-bound`, `eq:kernel-bound`, `eq:high-res-distortion`. All P2 (PDF renders correctly).
- **Section/theorem labels (10)**: `sec:broader-impact` (P2 — NeurIPS appendix anchor convention), `sec:exp-setup` (P2), `sec:waterbed` (P2), `sec:leff-remark` (P2), `sec:floor-higher-order` (P2), `prop:floor-higher-order` (P2), `sec:self-consistency` (P2), `thm:self-consistency` (P2), `sec:discrete-continuous-gap` (P2), and the **new** `sec:why-constant-alpha` (**P1**, see above).

---

## PDF placeholder check

```
$ pdftotext paper/main.pdf - | grep -F -c '??'
0
```

**0 `??` placeholders** in `paper/main.pdf` (timestamp 2026-04-27 16:38). PDF is newer than all source files (latest source = a1_proofs.tex 16:27), so it reflects the working-tree edits.

`grep -F` (literal-string mode) was used to avoid false positives from `?\?` regex matching every line. The PDF has been re-built post working-tree edits.

Build artefact note: `git diff HEAD paper/main.pdf` reports the working-tree PDF (2,153,092 bytes) is **smaller than** the committed `a358d6d` PDF (2,156,085 bytes); diff is `Bin -3KB`. This is consistent with the round-2 text edits being smaller than the prior version (the new §A.5 subsection adds ~1.3 KB but the §3.2 simplifications + §5.4 takeaway removal subtract more). No compilation error.

---

## \input order (paper/main.tex)

`paper/main.tex` body inputs (lines 49-53):

```
49  \input{sections/01_intro}
50  \input{sections/02_related}
51  \input{sections/03_theory}
52  \input{sections/05_experiments}
53  \input{sections/06_limitations}
```

Appendix inputs (lines 62-65): a1_proofs, a2_experiment_details, a3_supporting_results, a4_supporting_experiments. Same as prior round.

- **04_predictions.tex**: NOT \input'd. File is 1 line, comment-only: `% Content merged into §3 Theory (subsection 3.8 Predictions)`. **PASS**.
- **07_conclusion.tex**: NOT \input'd. File is 1 line, comment-only: `% Conclusion merged into §5 (Conclusion and Limitations) to fit 9-page body.`. **PASS**.
- **06_limitations.tex**: STILL \input'd at line 53, but file is now also a 1-line comment-only stub: `% Conclusion + Limitations merged inline at end of §5.4 for body 9-page budget.`. The \input therefore emits zero LaTeX output. **CONFIRMED stubbed**. P2 — leaving the \input in place is harmless (LaTeX silently includes a comment-only file), but a future cleanup could remove `\input{sections/06_limitations}` from `main.tex:53` to avoid the dangling stub. The Conclusion+Limitations text was relocated to `sections/05_experiments.tex` §4.4 ("Robustness and supporting evidence") tail, which the round-2 working-tree edit removed (commit 616edd7 added it; the round-2 edit deletes it). **NEW concern (P2): the conclusion paragraph that was added to §4.4 in commit 616edd7 has been deleted in the working-tree edit at sections/05_experiments.tex:61, leaving the paper without an explicit conclusion section.** This is a content-organization issue, not a xref bug — flagging as P2 stylistic for parent auditor team to consider.

---

## Bib check

- 44 distinct bibkeys cited across `paper/main.tex`, `sections/`, `appendix/`, `tables/`.
- 44 entries in `paper/refs/references.bib`.
- `comm -23 cite_keys.txt bib_keys.txt` (cited but not in bib) = **0 missing**. **PASS**.
- `comm -13 cite_keys.txt bib_keys.txt` (in bib but not cited) = **0 unused**. Tight 1:1.
- No new `\cite{}` was added in the round-2 edits (`git diff a358d6d HEAD paper/refs/references.bib` is empty).

---

## Summary

- **0 P0 findings** (no broken refs, no duplicate labels, no `??` in PDF).
- **1 P1 finding** (newly introduced this round):
  1. **`sec:why-constant-alpha` is a new orphan label.** Defined at `paper/appendix/a1_proofs.tex:100` (working-tree, uncommitted), but `paper/sections/03_theory.tex:32` reaches it only via the umbrella `Appendix~\ref{sec:proofs}` plus a natural-language `"Why constant α"` hint. Recommended fix: change `Appendix~\ref{sec:proofs} ``Why constant $\alpha$''` to `Appendix~\ref{sec:why-constant-alpha}` at `paper/sections/03_theory.tex:32` so the precise §A.5 anchor is clickable. The PDF rendering is correct as-is, so this is discoverability, not correctness.
- **3 P1 fixes from prior round are cleanly resolved**:
  1. `tab:pe-yarn-l256` now cited at `sections/05_experiments.tex:41` ("numbers in Appendix Table~\ref{tab:pe-yarn-l256}").
  2. `fig:attn-mechanism` now cited at `appendix/a3_supporting_results.tex:44` (introducing paragraph added).
  3. `sec:conclusion-limitations` removed entirely (`06_limitations.tex` is now a 1-line stub).
- **2 P2 awakenings** (intended): `eq:rho-tau-closed` and `eq:S-chi2-evq` are newly cited internally in `a1_proofs.tex` after the round-2 PSD/convexity tightening and the normalization-convention disambiguator paragraph.
- **0 accidental awakenings** of the other 33 prior dead labels — they all remain dead exactly as the prior audit catalogued.
- **\input order**: still 01, 02, 03, 05, 06 in body; `04_predictions.tex` and `07_conclusion.tex` confirmed not \input'd; `06_limitations.tex` is \input'd but emits zero content (1-line comment-only stub).
- **Bib**: 44 cited / 44 in bib / 0 missing / 0 unused. Clean.
- **PDF**: 0 placeholders, newer than all sources, reflects working-tree edits.

**Bottom line**: cross-reference integrity remains **clean at the P0 level** in round-2. The single P1 nit is the new orphan `sec:why-constant-alpha` — the new §A.5 subsection's label is not pointed at by any `\ref`, even though §3.2 was rewritten specifically to set up that subsection as the constant-α justification target. A 1-character edit at `sections/03_theory.tex:32` (`sec:proofs` → `sec:why-constant-alpha`) closes the gap and converts the textual title hint into a clickable anchor.

Secondary observation (not strictly xref): the round-2 working-tree edit at `sections/05_experiments.tex:61` removed the "Takeaway: RoPE allocation shape is a third design axis..." sentence that commit 616edd7 had added as the de-facto conclusion (since `06_limitations.tex` is now empty). The paper currently ends §4.4 abruptly without a closing statement, but this is content organization, not cross-reference integrity. Flagging for parent auditor to consider.
