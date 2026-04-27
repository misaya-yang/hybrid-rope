# Audit 7/8: PDF Rendering + NeurIPS 2026 Format Compliance

Date: 2026-04-27
Auditor: 7 of 8 (parallel)
PDF audited: `paper/main.pdf` (after fresh recompile this run)
Source baseline: HEAD `a358d6d paper: tighten body to 9-page boundary; add 2026-04-27 handover doc`

---

## Verdict (TL;DR)

**No P0 issues. No P1 issues. Two P2 polish notes.** PDF is submission-ready
from a format-compliance standpoint.

| Check | Result |
|---|---|
| Compile errors | 0 |
| Overfull \hbox / \vbox | 0 |
| Undefined references | 0 (after fresh recompile) |
| `??` placeholders in PDF text | 0 |
| Type 3 fonts | 0 (all 39 fonts are Type 1) |
| PDF metadata anonymity (Title/Author/Subject/Keywords) | clean (all empty) |
| Body length | 9 pages (Page 9 ends with the expected handover sentence) |
| References start on page 10 | yes |
| All 13 required tables present and not truncated | yes |
| Style file is real `neurips_2026.sty` (not renamed 2025) | confirmed |
| Forbidden page-cheats (\vspace{-, \addtolength, \linespread, \baselinestretch) | none |
| Q5 checklist `\answerYes{}` | confirmed |
| Figure EXIF/path leaks | none |

---

## 1. Compile log summary

### Recompile
PDF in repo (`paper/main.pdf`) timestamp 2026-04-27 09:09:16; .tex/.bib sources
also stamped 2026-04-27 09:09:16 (likely via git checkout). However, the
`main.log`/`main.aux` carried in the repo was older (2026-04-24 20:22:48) and
contained 4 stale "Reference `sec:exp-supporting' undefined" warnings —
**stale, not present in source**:
- `grep -n "sec:exp-supporting" paper/sections/*.tex paper/appendix/*.tex paper/main.tex` returns nothing.
- The labels in the source are `sec:exp-yarn`, `sec:exp-pe`, `sec:exp-mla`, `sec:exp-robust`, `sec:exp-setup`, `sec:experiments`.

I ran a fresh full compile sequence:

```
cd paper && pdflatex -interaction=nonstopmode main.tex
         && bibtex main
         && pdflatex -interaction=nonstopmode main.tex
         && pdflatex -interaction=nonstopmode main.tex
```

All four invocations completed without error. Final PDF: 40 pages,
2,106,513 bytes, generated 2026-04-27 11:30:57 CST.

### Counts on the fresh log

| Class | Count |
|---|---|
| `^! ` (LaTeX errors) | 0 |
| `^Overfull` | 0 |
| `^Underfull` | 9 (all hbox/vbox cosmetic, badness ≤10000, hyphenation-driven) |
| `LaTeX Warning: ... undefined` | 0 |
| Citation / reference warnings | 0 |

The 9 `Underfull \hbox` notices are all paragraph-level fill warnings (e.g.
`Broadband sur-ro-`, `point-wise ap-prox-i-ma-tion`); they do not produce
visible defects in the typeset PDF and are normal for a column-constrained
LaTeX document. Severity: P2 cosmetic only.

The single `LaTeX Warning: 'h' float specifier changed to 'ht'` is a NeurIPS
common pattern and not a problem.

---

## 2. Page count + body boundary

- Total pages: 40 (matches handover expectation: 9 body + 31 refs/appendix/checklist).
- Body length: **9 pages exactly**.
- Page 9 last sentence (per `pdftotext -layout -f 9 -l 9`):
  > "RoPE allocation shape—not only range—is a third design axis; EVQ-Cosh
  > (τ\* = d_head/√L) is a closed-form zero-parameter substrate downstream
  > methods inherit. Production MLA, ≥1B training, and per-channel
  > head-to-heads are natural follow-ups."
  Matches the handover-specified ending verbatim.
- Page 10 starts with `References` and the Bai et al. LongBench citation, as
  expected.
- No `??` placeholders anywhere in the PDF text dump (`grep -c '??'
  /tmp/paper_fresh.txt` → 0).

Page-9 boundary status: **CLEAN**.

---

## 3. Per-table visibility (13 required tables)

Table-name → number → page mapping derived from `\label{...}` lookups and
`pdftotext -layout` page-by-page extraction:

| Handover name | tab:label | PDF Table # | Page | Visible | Caption self-contained |
|---|---|---|---|---|---|
| EVQ×YaRN | tab:evq-yarn | 4 | 9 | yes | yes (matched-scale `s=4`, what 8/12/16K means) |
| PE-dominant | tab:pe-dominant | 5 | 9 | yes | yes (DAPE protocol, learned-vs-closed-form contrast) |
| evidence-tier | tab:evidence-tier | 3 | 8 | yes | yes |
| method-comparison | tab:method-comparison | 1 | 4 | yes | yes (RoPE-family scope clarified) |
| epistemic-map | tab:epistemic-map | 2 | 5 | yes | yes (variational core vs. recipe split labeled) |
| multiscale | tab:multiscale | 21 | 34 | yes | yes (single-seed 750M annotation present) |
| capability | tab:capability | 25 | 37 | yes | yes (links to Table 4 for full-sequence vs per-doc PPL) |
| lambda-cv | tab:lambda-cv | 7 | 18 | yes | yes (CV 0.28%, λ_∞=0.96±0.005, leave-one-out RMSE) |
| surrogate-validation | tab:surrogate-validation | 6 | 15 | yes | yes (12-config functional surrogate) |
| lora-8b | tab:lora-8b | 24 | 37 | yes | yes (single-seed, r=64, LongAlign-10k) |
| pe-yarn-l256 | tab:pe-yarn-l256 | 23 | 37 | yes | yes (visualizes Table 5 as supporting) |
| phase11-leverage | tab:phase11-leverage | 20 | 33 | yes | yes (matched-scale YaRN at L=256, NTK-aware control) |
| mla-results | tab:mla | 19 | 33 | yes | yes (3-seed mean±std MLA validation) |

All 13 are fully rendered with no row truncation or column overflow.

### Figures
10 figures total in the PDF (Figure 1–10). Spot-checked Figure 1 (page 4),
Figure 2 (page 8), Figure 10 (page 37): all captions are self-contained
(specify model size / seeds / context lengths / which panel shows what).

---

## 4. Font audit (`pdffonts`)

```
39 fonts embedded. All Type 1.
0 Type 3 fonts.        ← NeurIPS hard requirement satisfied
```

Embedded face list (deduped: 30 unique faces; 9 are duplicate subset entries
that pdftex reuses across pages):
- LMRoman {5,6,7,8,9,10,12} regular/italic/bold and LMRomanCaps10
- LMSans8-Regular
- LMMono10 regular/italic
- LMMath {Italic,Symbols,Extension} {5,6,7,9,10}
- AMS {MSAM,MSBM} {7,10}

All Latin Modern + AMS, all subsetted Type 1, all embedded. Standard
LaTeX/pdfTeX output via `\usepackage{lmodern}` + `\usepackage[T1]{fontenc}` in
`main.tex` lines 7–8.

Status: **PASS** (NeurIPS 2026 hard requirement: zero Type 3).

---

## 5. Metadata anonymity (`pdfinfo`)

```
Title:
Subject:
Keywords:
Author:
Creator:    LaTeX with hyperref
Producer:   pdfTeX-1.40.27
```

- Title / Author / Subject / Keywords: **all empty** (anonymous).
- Creator / Producer: pdfTeX strings — these are explicitly allowed by NeurIPS
  (they are toolchain fingerprints, not author identity).
- First-page typeset author block (page 1): "Anonymous Author(s) / Affiliation
  / Address / email" — confirmed via `pdftotext -f 1 -l 1`.

Status: **PASS**.

---

## 6. Style file + body discipline

### `\usepackage{neurips_2026}`
`paper/main.tex` line 3: `\usepackage[nonatbib]{neurips_2026}` — confirmed.
`paper/main.tex` line 4: `\usepackage{natbib}` — explicitly loaded after
`nonatbib` option, standard idiom.

### `neurips_2026.sty` is real (not renamed 2025)
`diff paper/neurips_2025.sty paper/neurips_2026.sty` shows substantive
differences:
- Header line 22: `\ProvidesPackage{neurips_2026}[2026-01-29 NeurIPS 2026 ...]`
- 2026-only `eandd` (Evaluations and Datasets) track option (replaces the
  2025 `dandb` option).
- 2026-only `nonanonymous` option.
- 2026 enforces minimum font sizes via `\renewcommand{\tiny}{\fontsize{6pt}...}`
  / `\scriptsize{7pt}` / `\footnotesize{8pt}` (lines 168–177) — this is a
  2026 anti-cheat hardening that 2025 lacked.
- Updated location string ("Sydney" vs "Sydney, Australia"), updated
  `\answerYes`/`\answerNo`/`\answerNA` formatting, etc.

Confirmed: this is the actual 2026 style file.

### Geometry from `neurips_2026.sty`
- `textheight=9in`, `textwidth=5.5in` (lines 125–126)
- Body normalsize 10pt on 11pt baseline (`\@xpt\@xipt`, line 145)
- Floors on tiny=6pt, scriptsize=7pt, footnotesize=8pt enforced by the .sty.

### Forbidden page-cheat grep
```bash
grep -rn '\\vspace{-' main.tex sections/ appendix/ tables/   → 0 hits
grep -rn '\\addtolength.*textheight' ...                     → 0 hits
grep -rn '\\setlength.*textfloatsep' ...                     → 0 hits
grep -rn '\\renewcommand.*baselinestretch' ...               → 0 hits
grep -rn '\\linespread' ...                                  → 0 hits
```

All four NeurIPS-forbidden page-stretching tricks: **absent**.

### Local font shrinking inside floats
Tables and appendix tables use `\small` and `\scriptsize` to fit content:
- `\small`: 18 occurrences (mostly in tables/ and appendix/)
- `\scriptsize`: 3 occurrences (table5_phase11_leverage.tex,
  table_evidence_tier.tex, table_method_comparison.tex)

This is **allowed** by NeurIPS (the prohibition is on globally shrinking the
body; local shrinking inside `\table`/`\figure` floats is standard practice
and is bounded below by the .sty floors of 7pt/6pt). Severity: not a
violation.

Status: **PASS**.

---

## 7. EXIF / figure metadata audit

`exiftool` is **not installed** locally; used `strings` + `pdfinfo` fallback.

### `pdfinfo` on every figure PDF
Eleven figure PDFs in `paper/figs/*.pdf`. All show only:
- `Creator: Matplotlib v3.10.8, https://matplotlib.org`
- `Producer: GPL Ghostscript 9.55.0`

No `Author`, `Title`, `Keywords`, or `Subject` fields populated on any figure
PDF.

### `strings | grep -iE "Author|Username|/Users/|/home/"` over `paper/figs/*.{pdf,png}`
Zero matches across all figure files.

Recommendation: install `exiftool` for a deeper audit of XMP/IPTC sidecar
metadata in the PNG files; current evidence is **clean** but `strings` only
catches plain-text leaks. Severity: P2 (not blocking — the matplotlib
fingerprint is generic and `strings` finds no path leak).

---

## 8. Q5 checklist confirmation (commit 88e4126)

`paper/main.tex` line 104–108:
```
\item \textbf{Open access to data and code}
\begin{itemize}
\item \textbf{Question:} Does the paper provide open access to the data and code?
\item \textbf{Answer:} \answerYes{}
\item \textbf{Justification:} ...anonymous supplementary code archive...
```

Q5 is **`\answerYes{}`** as specified by the handover. Justification text
covers: code archive scope (initializer, surrogate analyses, eval, configs),
upstream dataset accessibility (FineWeb-Edu, TinyStories, QuALITY, RULER),
deferred items (model checkpoints withheld for double-blind / size), and
points to a reproducibility README.

Full checklist tally (paper/main.tex): 11×Yes, 4×N/A, 0×No. Q9 (Q&A about
license / human subjects / safeguards) is `\answerNA{}` (line 149) which is
appropriate for a methodology paper without released model weights.

Status: **PASS**.

---

## 9. Summary

### P0 (submission-blocking)
**None.** Compile clean, no Type 3 fonts, metadata anonymous, body exactly
9 pages, no `??`, no `\vspace{-`.

### P1 (must fix before camera-ready)
**None.** No overfull boxes, all required captions self-contained, no EXIF
leaks detected by `strings`/`pdfinfo`.

### P2 (polish)
1. **Stale `main.log`/`main.aux` shipped in working tree** — the .log/.aux
   in the repo (Apr 24) lagged behind the .tex sources (Apr 27), so anyone
   reading `paper/main.log` casually would see 4 phantom "undefined
   reference: sec:exp-supporting" warnings that vanish on a fresh compile.
   Recommend either committing fresh post-recompile auxiliaries or
   `.gitignore`-ing them and noting that `main.pdf` is the only artifact
   that gets versioned. (My fresh recompile has produced clean
   `main.log`/`main.aux`/`main.pdf` in your working tree — you may want to
   `git status` to decide whether to commit those.)
2. **`exiftool` not installed locally**, so the EXIF audit fell back to
   `strings`/`pdfinfo`. Both are clean, but installing
   `brew install exiftool` and re-running gives a more thorough XMP/IPTC
   sidecar check on the PNG figures before submission.
3. **9 `Underfull \hbox`** badness warnings remaining are minor
   hyphenation-fill artifacts in the body and appendix; cosmetic only.
   None produce visible bad-set lines in the typeset PDF.

### What was actually checked vs. asked
- Recompile sequence: ran fully, fresh log clean.
- 13 specific tables: enumerated, page-mapped, visually inspected for
  truncation — all pass.
- Font audit: 39 fonts, 0 Type 3.
- Metadata: 4 of 4 anonymity fields empty.
- Style file: confirmed real 2026 (diff against 2025 sty has 19+ substantive
  changes including the 2026-specific font-size floors).
- 4 forbidden page-cheat patterns: zero matches each.
- Q5 status: `\answerYes{}` confirmed at main.tex:107.
- Figure EXIF: zero leaks under available tooling.

**Overall: PASS for NeurIPS 2026 submission format.**
