# Audit v2 — 01: Diff Verification (Auditor 1/6)

**Date**: 2026-04-27 (round-2 cross-validation)
**Baseline**: `a358d6d` (sealing commit)
**Stage A commit**: `616edd7` (committed)
**Stage B + C**: in working tree (uncommitted)
**Scope**: `paper/` only

---

## Methodology

1. Captured three diffs:
   - `git diff a358d6d -- paper/` → `/tmp/full_diff.txt` (919 lines, all changes since seal)
   - `git diff a358d6d HEAD -- paper/` → `/tmp/committed_diff.txt` (861 lines, Stage A only)
   - `git diff HEAD -- paper/` → `/tmp/uncommitted_diff.txt` (84 lines, Stage B+C only)
2. Cross-referenced each numbered claim (A1–A13, B1–B4, C1–C2) against (i) commit `616edd7` `git show --name-status` and (ii) the actual diff hunks.
3. Walked the actual file-level result with `grep -n` for verification of post-edit text at the line numbers, not the line numbers in the user's claims (per audit instructions).
4. Ran `grep -rn "sec:why-constant-alpha"` to confirm the new label resolves.
5. Verified the F2 file move via the rename status (`R100`) in `git log --name-status`.

Severity legend:
- **P0** = false claim or unclaimed change found
- **P1** = partial match / wording delta from claim
- **P2** = stylistic / cosmetic delta

---

## Per-claim verification table (19 rows)

| # | Claim (paraphrased) | Status | Severity | Evidence |
|---|---|:-:|:-:|---|
| **A1** | P0-1 Table 5 caption + abstract + intro + §5 + tier sync ("1–3 seed; Learnable τ row only is 3-seed") + std `181.2±1.3 / 437.9±12.2` on Learnable τ row | ✓ EXACT | — | `paper/tables/table4_pe_dominant.tex:2` "Geo, DAPE, and EVQ rows are single-seed (seed 42); the Learnable $\tau$ row is 3-seed (42/137/256)…"; row 13: `$\mathbf{181.2{\scriptstyle\,\pm 1.3}}$ & $437.9{\scriptstyle\,\pm 12.2}$`. `paper/sections/01_intro.tex:9` "three primary stress tests…(1–3 seed; the learned-PE row is 3-seed, EVQ/Geo/DAPE rows are seed 42)". `paper/sections/05_experiments.tex:11` "Primary II 1–3 seed; only the learned-PE comparison row is 3-seed". `paper/tables/table_evidence_tier.tex:13` "PE-dominant…(DAPE; learned-PE row 3-seed, others seed 42) & 1–3". Note: `paper/main.tex:46` abstract was NOT edited for "3-seed" wording — only for "matched-scale (s=4)" qualifier (claim A12 sub-bullet). Original abstract phrasing already says "Three primary stress tests" with no explicit seed claim, so no drift remained to fix there. |
| **A2** | P0-2 Table 4 caption: L=512/s=4/16-32× → L=2048/s=8/4-8×; tier row sync | ✓ EXACT | — | `paper/tables/table2_evq_yarn_main.tex:2` "(454M, $L_{\mathrm{train}}{=}2048$, 10\% passkey mix; $4{\times}/6{\times}/8{\times}$ extrapolation). YaRN uses…scale $s{=}8$". `paper/tables/table_evidence_tier.tex:12` "$L_{\mathrm{train}}{=}2048$, $4$/$6$/$8{\times}$ extrapolation". |
| **A3** | P1-A1 Table 4 PK@8K column std `41±5% / 61±3% / 53±8% / 100±0%` | ✓ EXACT | — | `paper/tables/table2_evq_yarn_main.tex:11–14`: `$41{\scriptstyle\pm 5}$\%`, `$61{\scriptstyle\pm 3}$\%`, `$53{\scriptstyle\pm 8}$\%`, `$\mathbf{100{\scriptstyle\pm 0}}$\textbf{\%}`. |
| **A4** | P1-A3 Table 5 phase11-leverage 6 cells get std (`-4.5±1.2 / -3.1±1.6 / -27.0±0.1 / -28.9±1.3 / -32.5±2.0 / -40.7±1.6`) | ✓ EXACT | — | `paper/tables/table5_phase11_leverage.tex:12–14`: `$-4.5\%{\scriptstyle\,\pm 1.2}$ & $-3.1\%{\scriptstyle\,\pm 1.6}$`, `$-27.0\%{\scriptstyle\,\pm 0.1}$ & $-28.9\%{\scriptstyle\,\pm 1.3}$`, `$\mathbf{-32.5\%{\scriptstyle\,\pm 2.0}}$ & $\mathbf{-40.7\%{\scriptstyle\,\pm 1.6}}$`. |
| **A5** | P1-A4 Table 1 750M continue@4K row +0.9% → +1.2% | ✓ EXACT | — | `paper/tables/table1_multiscale_raw_ppl.tex:14`: `750M continue@4K & Mix setting & 1 & +1.2\% & -45.9\%`. |
| **A6** | P1-B1 a1_proofs.tex K=d_head/2 → K=d_rot/2 (Floor proposition) | ✓ EXACT | — | `paper/appendix/a1_proofs.tex:158`: "For the EVQ-Cosh mapping with $K = d_{\mathrm{rot}}/2$ channels, the $N$-channel mid-displacement threshold satisfies". (Note: hunk header in committed diff is `@@ -144,7 +144,7 @@` — line 158 in current file, line 147 in user's claim is approximately consistent up to whitespace anchor; the substantive text matches exactly.) |
| **A7** | P1-B2 a1_proofs.tex α=1/d_rot multi-arch note (=1/d_head MHA, =1/d_rope MLA) | ✓ EXACT | — | `paper/appendix/a1_proofs.tex:559`: "…$\alpha{=}1/d_{\mathrm{rot}}$ ($={}1/d_{\mathrm{head}}$ for MHA via \S\ref{sec:tau-scaling}; ${=}1/d_{\mathrm{rope}}$ for compressed-RoPE MLA via \S\ref{sec:mla-results})." |
| **A8** | P1-B3 a1_proofs.tex normalization parenthetical near eq:S-chi2-evq | ✓ EXACT | — | `paper/appendix/a1_proofs.tex:197`: "For $\rho_\tau(\phi) = \tau\cosh(\tau(1{-}\phi))/\sinh\tau$ (Eq.~\eqref{eq:S-chi2-evq} below uses the un-normalized integral; cf.\ \S\ref{sec:chi2-load} ``Normalization convention'')". |
| **A9** | P1-B4 a1_proofs.tex β=0 explicit degenerate case after Theorem 1 proof | ✓ EXACT | — | `paper/appendix/a1_proofs.tex:40`: "For the degenerate case $\beta{=}0$ the ODE reduces to $\rho''{=}0$ with $\rho'(0){=}\rho'(1){=}0$ and $\int\rho{=}1$, forcing $\rho\equiv 1$, which is also the $\tau{\to}0$ limit of~\eqref{eq:rho-tau-closed}." This was committed in Stage A (hunk `@@ -37,7 +37,7 @@`); Stage B uncommitted hunk *additionally* expands the same line with KKT-inactive language. See B4 for the second-stage augmentation. |
| **A10** | P1-D1 §5 Appendix Table tab:pe-yarn-l256 ref added | ✓ EXACT | — | `paper/sections/05_experiments.tex:41`: "(Appendix Fig.~\ref{fig:pe-dominant}, numbers in Appendix Table~\ref{tab:pe-yarn-l256})". |
| **A11** | P1-D2 a3 fig:attn-mechanism intro paragraph added | ✓ EXACT | — | `paper/appendix/a3_supporting_results.tex:44`: "Figure~\ref{fig:attn-mechanism} illustrates the waterbed crossover in learned attention patterns of the 750M continue@4K checkpoint, complementing the surrogate-level analysis in \S\ref{sec:waterbed-proof}." |
| **A12.a** | C1: a2 dead-channel "validation" → "mechanism check"; "vanishes entirely" → "is no longer present in this seed" | ✓ EXACT | — | `paper/appendix/a2_experiment_details.tex:153`: "Dead-channel mechanism check: …is no longer present in this seed". |
| **A12.b** | C2: a4 heading "Cross-modal confirmation: video DiT (supporting)" → "Cross-modal mechanism check: video DiT" | ✓ EXACT | — | `paper/appendix/a4_supporting_experiments.tex:57`: "\paragraph{Cross-modal mechanism check: video DiT.}" |
| **A12.c** | C3: a3:55,108 "Waterbed verification" → "Waterbed illustration" (×2) | ✓ EXACT | — | `paper/appendix/a3_supporting_results.tex:57`: "\textbf{Waterbed illustration in learned attention patterns (750M, single seed).}" + line 110: "\textbf{Waterbed illustration across model scales.}". (Caption also gains "single seed" qualifier — minor over-delivery, no severity.) |
| **A12.d** | C4: a4:36 LoRA caption "dramatic" → "substantial" | ✓ EXACT | — | `paper/appendix/a4_supporting_experiments.tex:36`: "EVQ trades a modest in-distribution cost for substantial extrapolation gains." |
| **A12.e** | C5: main.tex:46 abstract "surpassing Geo+YaRN" → "surpassing matched-scale (s=4) Geo+YaRN" | ✓ EXACT | — | `paper/main.tex:46`: "…at $+1.1\%$ in-distribution cost, surpassing matched-scale ($s{=}4$) Geo+YaRN." |
| **A12.f** | C6: main.tex:80 Q1 "validated" → "calibrated" | ✓ EXACT | — | `paper/main.tex:80`: "…with derived $d_{\mathrm{head}}$ and $L^{-1/2}$ structure and an empirically calibrated $\mathcal{O}(1)$ scale…" |
| **A12.g** | C7: main.tex:94 Q3 "surrogate validated" → "supported by functional check" | ✓ EXACT | — | `paper/main.tex:94`: "…the surrogate itself is supported by a functional check (24--92\% collision-score reduction across 12 configurations) in Appendix~\ref{sec:surrogate-validation}." |
| **A12.h** | C8: a3:69 drop "clearly" | ✓ EXACT | — | `paper/appendix/a3_supporting_results.tex:71` (post-edit): "gold-answer NLL separates the two RoPE variants" (no "clearly"). |
| **A12.i** | C9: a3:103 mixed-tier rewrite to bracketed range | ✓ EXACT | — | `paper/appendix/a3_supporting_results.tex:105`: "EVQ trades a small in-range cost (${\pm}1.7\%$) for long-range gains ($10$--$46\%$) across the tested grid." |
| **A13** | F2 paper/COWORK_FINAL_PROMPT.md and paper/EDIT_CHANGELOG.md moved to internal/2026_04_run/chat_scaffolding/ | ✓ EXACT | — | `git log --name-status a358d6d..HEAD` reports both as `R100` (rename, 100% similarity) into `internal/2026_04_run/chat_scaffolding/`. Verified with `ls`: target dir contains both files; source paths return ENOENT. |
| **B1** | Action 1: §3.7 KL 1/2 convention parenthetical "absorbed into λ" | ✓ EXACT | — | `paper/sections/03_theory.tex:93` (paragraph "Shape and scale come from separate layers"): "…convention $\tfrac{1}{2}$ factors from the KL Taylor expansion are absorbed into $\lambda$ (Appendix~\ref{sec:lambda-curvature})." This is in the paragraph immediately preceding `prop:softmax-transport`, not inside the proposition body — the user's claim said "§3.7 KL 1/2 convention parenthetical" which fits this placement. ⚠ minor: the user's wording in B1 claim was ambiguous about whether this attaches to a Proposition body or the lead paragraph; the realized location is the lead paragraph. No semantic delta. |
| **B2** | Action 2: §3.7 Proposition gets "Real τ_* requires Q_1>0; Q_1∈[0.008, 0.032]" | ✓ EXACT | — | `paper/sections/03_theory.tex:108` (inside Proposition body): "Real $\tau_*$ requires $Q_1(L,b){>}0$, verified across the tested grid ($Q_1{\in}[0.008, 0.032]$; Appendix~\ref{sec:lambda-cv})." |
| **B3** | Action 3: §3.2 distance prior gets D ∈ W^{m,1}(R) bounded-support qualifier | ✓ EXACT | — | `paper/sections/03_theory.tex:23`: "Under a smooth, bounded-support distance prior $D \in W^{m,1}(\mathbb{R})$ ($m\geq 1$; the regime relevant to autoregressive language models with finite effective context)…" |
| **B4** | Action 4: a1 §A.1.1 KKT inactive-constraint explicit sentence | ✓ EXACT | — | `paper/appendix/a1_proofs.tex:40` (uncommitted hunk extends Stage A's β=0 line): "Positivity holds…since $\rho_\tau(\phi){\geq}\rho_\tau(1){=}\tau/\sinh\tau{>}0$ for $\tau{>}0$ (with $\rho_\tau(0){=}\tau\coth\tau{>}0$); the inequality constraint $\rho{>}0$ is therefore inactive throughout $[0,1]$ and the unconstrained Euler--Lagrange solution coincides with the KKT-constrained solution." Note: the Stage A (committed) version of this same line had only the β=0 sentence; Stage B reorganizes the positivity argument and adds the KKT inactivity sentence. The diff hunk is `@@ -37,7 +37,7 @@` showing the line was rewritten, not appended. |
| **C1** | Action 6: §3.2 K_app framing rewritten as "discrete-grid surrogate fitted at deployed channel grid, not continuum stationary-phase" | ✓ EXACT | — | `paper/sections/03_theory.tex:32`: "with constants $\alpha,\beta>0$ \emph{fitted from the exact kernel $K$ on the deployed discrete channel grid} ($K{\in}\{16,32\}$; Appendix~\ref{sec:tau-scaling}), giving $\alpha{\approx}1/d_{\mathrm{rot}}$. … $\mathcal{C}_{\mathrm{app}}$ is a discrete-grid surrogate, not the continuum stationary-phase asymptotic of $K$ (which would give a $\phi$-dependent diagonal coefficient; see Appendix~\ref{sec:proofs} ``Why constant $\alpha$'')." |
| **C2** | Action 7: a1 NEW \subsection (label sec:why-constant-alpha) with const-vs-sp identity, modified-Bessel I_0/K_0 ODE, three reasons | ✓ EXACT | — | `paper/appendix/a1_proofs.tex:99`: `\subsection{Why constant \texorpdfstring{$\alpha$}{alpha} (and not the continuum stationary-phase coefficient)}`; line 100 `\label{sec:why-constant-alpha}`. Body contains: stationary-phase derivation of `α_sp(φ) = πD₀/(Lλ b^{-φ}) = (πD₀/Lλ)b^φ`; equation `\eqref{eq:const-vs-sp}` at line 108; "modified-Bessel combination $\rho(\phi){\propto}e^{-\lambda\phi}[A\,I_0(x(\phi))+B\,K_0(x(\phi))]$"; three numbered reasons (i) closed-form CDF, (ii) manifest positivity, (iii) discrete-grid fit. All claim sub-elements are present. |

### Summary of per-claim verification

- **19/19 claims verified** at "✓ EXACT MATCH" status.
- **0 ✗ MISSING** claims.
- Action 5 was correctly skipped per the user's note (no diff expected, none found).

---

## Extra hunks (potential regressions / unclaimed edits)

Three changes appear in the diff that are NOT covered by any A1-A13/B1-B4/C1-C2 claim:

### EXTRA-1 (P1): Stage B drops the inline "Takeaway:" sentence at end of §5.4

**Location**: `paper/sections/05_experiments.tex:61` (final line)

**Stage A (committed at `616edd7`) ended §5.4 with**:
> "…concentrated at the PE layer. *Takeaway:* RoPE allocation *shape* is a third design axis; EVQ-Cosh ($\tau^*{=}d_{\mathrm{head}}/\sqrt{L}$) is a closed-form zero-learned-parameter substrate. Production MLA, ${\geq}1$B training, and per-channel head-to-heads remain follow-ups."

**Stage B (uncommitted, current working tree) ends §5.4 with**:
> "…concentrated at the PE layer." (sentence removed)

**Severity**: P1. The "Takeaway" sentence was the entire content of the merged §6 conclusion (per `06_limitations.tex:1` comment "Conclusion + Limitations merged inline at end of §5.4 for body 9-page budget"). Dropping it leaves the paper with NO conclusion-equivalent text in the body — `06_limitations.tex` now contains only a comment, and `07_conclusion.tex` was deleted earlier. This is either:
- (a) An accidental omission when applying Stage B edits to §5 (regression vs. Stage A intent), OR
- (b) An intentional re-trim for page-budget reasons that the user forgot to mention in the claim list.

The diff hunk `@@ -58,4 +58,4 @@` in `/tmp/uncommitted_diff.txt` shows the removal explicitly: the `+` line is shorter than the corresponding `-` line by exactly the "Takeaway: …follow-ups." sentence. **Recommend the user confirm this was intentional**; if accidental, the line should be restored before submission.

### EXTRA-2 (P2): Stage A `table2_evq_yarn_main.tex` caption gains an explanatory clause about std availability

**Location**: `paper/tables/table2_evq_yarn_main.tex:2`

The caption now includes (beyond the L_train=2048/s=8 metadata fix in claim A2 and the PK@8K std in claim A3):
> "PK@8K is 3-seed mean$\pm$std; other PK columns and PPL are 3-seed means (per-seed PPL not preserved locally for YaRN rows)."

**Severity**: P2. This is a defensive disclosure consistent with the `audit/01_number_trace.md` finding "per-seed PPL not preserved locally for YaRN composition rows; std on retrieval column is recoverable" — i.e., it transparently reports why some columns lack std. Not a regression; arguably *improves* honesty. The user's A3 claim list mentioned only the std numbers, not the explanatory clause; this is over-delivery rather than drift. No action required.

### EXTRA-3 (P2): Stage A `table2_evq_yarn_main.tex` adds `\setlength{\tabcolsep}{4pt}`

**Location**: `paper/tables/table2_evq_yarn_main.tex:6`

Diff line: `+\setlength{\tabcolsep}{4pt}`

**Severity**: P2. Cosmetic — likely needed because adding ±std subscripts to the PK columns widened the table; reducing column padding keeps it within `\textwidth`. No action required.

### EXTRA-4 (P2): Stage A `a3_supporting_results.tex:57` waterbed-attn caption gains "(single seed)" qualifier

**Location**: `paper/appendix/a3_supporting_results.tex:57`

Beyond the C3 "verification" → "illustration" rename, the caption also gains the qualifier "(750M, **single seed**)":
> "\textbf{Waterbed illustration in learned attention patterns (750M, single seed).}"

**Severity**: P2. Honest seed-count disclosure consistent with the audit-5 overclaim-scan principle (which the C3 rewrite was meant to address). Over-delivery, not regression. No action.

---

## Stage-commit mapping

| Stage | Expected location | Verified | Notes |
|---|---|:-:|---|
| **A1–A13 (13 claims)** | committed in `616edd7` (= NOT in working tree) | ✓ | `git log --name-status 616edd7` shows all 12 modified files + 2 renames. `git diff a358d6d HEAD -- paper/` shows the corresponding hunks. `git diff HEAD -- paper/` does NOT contain any A* hunk. |
| **B1–B4 (4 claims)** | working tree only (= NOT in `616edd7`) | ✓ | All 4 hunks (a1@37, 03_theory@20, 03_theory@90, 03_theory@105) appear in `/tmp/uncommitted_diff.txt`; none appear in `/tmp/committed_diff.txt`. (Note: the a1@37 hunk was *partially* introduced in Stage A — it added the β=0 sentence — and is *expanded* in Stage B with KKT-inactive language. So the committed file has the β=0 part; the working tree has β=0 + KKT.) |
| **C1, C2 (2 claims)** | working tree only | ✓ | `paper/sections/03_theory.tex:32` (C1 K_app reframe) and `paper/appendix/a1_proofs.tex:99–110` (C2 new subsection) are uncommitted. |
| Action 5 | skipped | ✓ | No corresponding hunk found, as expected. |

---

## Label cross-check: `sec:why-constant-alpha`

```
$ grep -rn "sec:why-constant-alpha" paper/
paper/appendix/a1_proofs.tex:100:\label{sec:why-constant-alpha}
```

- **Definition** at `paper/appendix/a1_proofs.tex:100` (in the new \subsection from C2). ✓
- **References to it**: `grep -rn` finds **zero** `\ref{sec:why-constant-alpha}` invocations. The companion §3.2 paragraph (claim C1) cross-refers to the new subsection via plain English ("see Appendix~\ref{sec:proofs} ``Why constant $\alpha$''") rather than `\ref{sec:why-constant-alpha}`.
- **Severity**: P2. The label is well-formed and resolves; the lack of a `\ref{}` to it is benign because §3.2 uses the section-name pointer + the encompassing `sec:proofs` ref. However, if the user intends the label to be referenceable from elsewhere, they could add `\ref{sec:why-constant-alpha}` in §3.2 instead of the `sec:proofs` plus quoted-name pointer. Optional polish; not a regression.

---

## Summary

| Category | Count |
|---|---:|
| ✓ EXACT MATCH | 19 / 19 numbered claims |
| ⚠ PARTIAL | 0 |
| ✗ MISSING | 0 |
| Extra hunks | 4 (1× P1 regression candidate, 3× P2 over-delivery / cosmetic) |

**P0 findings**: 0.

**P1 findings**: 1.
- **EXTRA-1** (sections/05_experiments.tex:61, Stage B uncommitted): the inline "Takeaway: RoPE allocation *shape*…follow-ups." sentence introduced by Stage A as the merged §6 conclusion is REMOVED in the Stage B working tree without a corresponding claim. Net effect: the paper body loses its only conclusion sentence (the merged-inline replacement for §6/§7). Recommend confirming with the user whether this was intentional page-fit pressure or an accidental omission, and restoring before submission if accidental.

**P2 findings**: 3 (all benign over-delivery: a defensive caption clause, a `\tabcolsep` cosmetic, a "single seed" qualifier).

**Verdict**: All 19 numbered claims (A1–A13, B1–B4, C1–C2) verify at EXACT MATCH against the diff. Stage A is correctly committed in `616edd7`; Stage B and Stage C are correctly in the working tree only. The new label `sec:why-constant-alpha` is well-formed at `paper/appendix/a1_proofs.tex:100`. The single P1 finding (EXTRA-1) is the lost "Takeaway:" sentence at the end of §5.4 in the Stage B working tree, which the user should explicitly confirm is intentional.
