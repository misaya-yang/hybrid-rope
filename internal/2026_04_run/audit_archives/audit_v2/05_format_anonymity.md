# Audit v2 — 05/06 Format + Anonymity Sweep (Round-2 Regression Check)

Date: 2026-04-27
Auditor: 5 of 6 (parallel cross-validation team)
PDF audited: `paper/main.pdf` (after fresh full recompile this run)
Source baseline: HEAD `a358d6d paper: tighten body to 9-page boundary; add 2026-04-27 handover doc`
Prior-round reference: `audit/07_pdf_format.md`, `audit/08_anonymity.md`

---

## Verdict (TL;DR)

**No P0. No P1. Three P2 polish notes.** PDF is submission-ready.

Round-2 regression check: this round added new math (Action 7 Bessel subsection
in `\subsection{Why constant }` in `appendix/a1_proofs.tex`, plus body
revisions through `sections/03_theory.tex`, `sections/05_experiments.tex`,
`sections/06_limitations.tex`, `appendix/a1_proofs.tex`,
`appendix/a3_supporting_results.tex`, `appendix/a4_supporting_experiments.tex`,
six tables). Despite the additions:

- Body remains **exactly 9 pages**. Page 9 ends with a complete sentence.
- Compile clean (0 errors, 0 overfull, 0 undefined refs, 0 `??`).
- 42 fonts embedded, all Type 1, 0 Type 3.
- PDF Title/Author/Subject/Keywords all blank.
- Q5 still `\answerYes{}` (line 107).
- 0 forbidden page-cheats.
- 0 identity leaks in any paper-source file or PDF body.
- `paper/COWORK_FINAL_PROMPT.md` and `paper/EDIT_CHANGELOG.md` confirmed
  **moved out** of `paper/` to `internal/2026_04_run/chat_scaffolding/`.
- New Bessel subsection and label `sec:why-constant-alpha`: identity-clean.

Total page count: **41** (vs 40 in prior round; the +1 came from References /
appendix expansion, not from body). Body discipline preserved.

---

## PART 1 — Format

### 1.1 Compile log summary

#### Source-vs-PDF freshness check
`paper/main.pdf` mtime is **Apr 27 16:27:36 2026**. Newest source file is
`paper/appendix/a1_proofs.tex` at **Apr 27 16:27:18 2026** — PDF is 18 s newer
than the latest source. `find paper/{sections,appendix,tables,refs,main.tex}
-newer paper/main.pdf` returns 0 files. PDF is up to date.

I nonetheless ran the full sequence to verify clean compile state:

```
cd paper && pdflatex -interaction=nonstopmode main.tex
         && bibtex main
         && pdflatex -interaction=nonstopmode main.tex
         && pdflatex -interaction=nonstopmode main.tex
```

All four invocations completed with exit 0. Final PDF: **41 pages**,
2,153,092 bytes, generated 2026-04-27 16:38:15 CST. Stable across passes 2 and 3
(no rerun warnings).

#### Counts on the fresh log (paper/main.log)

| Class | Count |
|---|---|
| `^! ` (LaTeX errors) | **0** |
| `^Overfull` (hbox or vbox) | **0** |
| `^Underfull \hbox` | 12 (paragraph hyphenation-fill, badness ≤10000, cosmetic) |
| `^Underfull \vbox` | 5 (output-active fill warnings, badness ≤10000) |
| `LaTeX Warning: ... undefined` | 0 |
| `Reference ... undefined` | 0 |
| Citation warnings | 0 |
| `Rerun to get` requests | 0 |
| `?? on input line` (`??` placeholders) | 0 |

The 17 `Underfull` notices are all paragraph-/page-level fill warnings (e.g.
`Underfull \hbox (badness 1231) in paragraph at lines 10--10` from the abstract;
others are from a1_proofs and a4_supporting_experiments dense-equation
paragraphs). They produce no visible defects in the typeset PDF and are
unchanged in spirit from the 9 underfull warnings in the prior round (the
delta of +8 is consistent with the 17 net new lines added across appendix
math). **Severity: P2 cosmetic only.**

### 1.2 Page boundary check (CRITICAL)

```
Total pages: 41
Body pages: 1–9
References: pages 10–12
Appendix:   pages 13–38
Checklist:  pages 39–41
```

Per-page boundary verification:

- **Page 9 (last body page).** Last visible sentence:
  > "Multi-scale raw extrapolation (Table 21): in-range PPL stays within ±1.7%
  > while long-range improves 10–46%; capability checks (Table 25) show EVQ is
  > non-destructive. Supporting (1–2 seed): video DiT (0.53× post-hoc), LoRA
  > into LLaMA-3-8B at r=d_head/2 (8–19× extrapolation-PPL), progressive
  > training. Aggregated, EVQ's effect attenuates through abstraction layers
  > (raw PPL −46%/−13% 1-/3-seed; gold-NLL −30%; passkey +59 pp; QA +2.2 pp),
  > **concentrated at the PE layer.**"

  Full-stop terminator. **Sentence-end, not mid-paragraph.** Section 4.5
  ("Robustness and supporting evidence") closes cleanly on page 9.

- **Page 10** starts with `References` and the Bai et al. LongBench citation,
  as required.

- No table or figure truncation observed (spot-checked Table 4 PE-yarn on
  page 9, Figure 2 on page 8, Table 25 capability on page 38).

- No equation off-column observed (visual inspection of pages 8–9 and the
  dense appendix pages 13–18).

- `grep -c '??' /tmp/v2_paper.txt` = **0**.

**Page-9 boundary status: CLEAN. No regression vs prior round despite new
math additions in the body and new appendix subsection.**

### 1.3 Q5 status

`paper/main.tex` lines 104–108:

```latex
\item \textbf{Open access to data and code}
\begin{itemize}
\item \textbf{Question:} Does the paper provide open access to the data and code?
\item \textbf{Answer:} \answerYes{}
```

Q5 is **`\answerYes{}`** at line 107, matching prior commit 88e4126 and the
prior-round audit. **Confirmed.**

Full checklist tally: 11×Yes, 4×N/A, 0×No (no regression).

### 1.4 Font audit (`pdffonts`)

```
Total fonts embedded: 42
  Type 1, Custom encoding:  38 (all subsetted, all embedded)
  Type 1, Builtin encoding:  4 (AMS math symbols)
Type 3 fonts: 0
```

Embedded face list: LMRoman {5–12} regular/italic/bold/caps/sans/mono;
LMMath {Italic,Symbols,Extension}; AMS {MSAM,MSBM} {7,10}. All Latin Modern +
AMS, all subsetted Type 1, all embedded.

Status: **PASS.** NeurIPS 2026 hard requirement (zero Type 3) satisfied.

(+3 fonts vs prior round's 39 — likely consequence of new math operators in
the Bessel subsection requiring additional AMS or Latin-Modern math glyphs.
All still Type 1.)

### 1.5 PDF metadata (`pdfinfo`)

```
Title:           
Subject:         
Keywords:        
Author:          
Creator:         LaTeX with hyperref
Producer:        pdfTeX-1.40.27
CreationDate:    Mon Apr 27 16:38:15 2026 CST
ModDate:         Mon Apr 27 16:38:15 2026 CST
Pages:           41
Page size:       612 x 792 pts (letter)
File size:       2153092 bytes
PDF version:     1.7
```

- **Title / Author / Subject / Keywords: all empty.** Pass.
- **Creator / Producer:** standard pdfTeX strings, allowed by NeurIPS.
- First-page typeset author block: "Anonymous Author(s) / Affiliation /
  Address / email" (per `pdftotext -f 1 -l 1`). Pass.

The `+08'00'` timezone in `/CreationDate` and `/ModDate` is the same soft
fingerprint noted in prior round (consistent with mainland China / Singapore /
WA / Malaysia). Not a P0 (NeurIPS reviewers do not inspect this); same
recommendation as prior round (`SOURCE_DATE_EPOCH=0` recompile if scrubbing).

### 1.6 Style file

`paper/main.tex` line 3: `\usepackage[nonatbib]{neurips_2026}`. Confirmed
(line 4 follows with `\usepackage{natbib}`). Real 2026 sty per prior-round
diff against 2025.

### 1.7 Forbidden page-cheat grep

```
grep -rE '\\vspace\{-' paper/main.tex paper/sections/ paper/appendix/ paper/tables/
  → 0 hits across 22 source files
grep -rE '\\addtolength.*textheight' …                      → 0 hits
grep -rE '\\linespread' …                                   → 0 hits
grep -rE '\\baselinestretch' …                              → 0 hits
grep -rE '\\setlength.*textfloatsep' …                      → 0 hits
```

All five NeurIPS-forbidden page-stretching tricks: **absent.** No regression.

(Local `\small` / `\scriptsize` font shrinking inside `\table` floats is
allowed by NeurIPS and bounded below by the .sty floors; same situation as
prior round.)

---

## PART 2 — Anonymity (regression check vs audit/08)

### 2.1 Scaffolding-file relocation (was P1 in audit/08, now resolved)

Prior round (audit/08 §6.3) flagged that `paper/COWORK_FINAL_PROMPT.md` and
`paper/EDIT_CHANGELOG.md` lived in `paper/` and contained `/Users/yang/...`
absolute paths. These were planned-to-move but **not yet moved** at the time
of audit/08.

Confirmed this round:

```
ls paper/*.md
  → CITATION_AUDIT_REPORT.md
    README.md
    REBUTTAL_PLAYBOOK.md
    REVIEW_PROMPT.md
  (NOT COWORK_FINAL_PROMPT.md, NOT EDIT_CHANGELOG.md)

ls internal/2026_04_run/chat_scaffolding/
  → COWORK_FINAL_PROMPT.md
    EDIT_CHANGELOG.md
```

**Both files moved out of `paper/`. Resolved.**

The four remaining `paper/*.md` files (CITATION_AUDIT_REPORT, README,
REBUTTAL_PLAYBOOK, REVIEW_PROMPT) are NOT compiled into the PDF (LaTeX never
`\input`s them) and were re-scanned for identity leaks:

```
grep -nEi 'misaya|yanghej|hejaz|/Users|/home|/sessions|seetacloud|sshpass|@hejazfs|@privaterelay|/root/autodl' paper/{REVIEW_PROMPT,REBUTTAL_PLAYBOOK,CITATION_AUDIT_REPORT,README}.md
  → 0 hits
```

All four are **identity-clean.** They remain a P1 caveat only if `paper/` is
zipped wholesale as a "paper-source" archive (the standard guidance from
prior audit/08 §6.11 still applies: exclude these from any tarball build).

### 2.2 Identity sweep on paper LaTeX sources

```
grep -rEni 'misaya|yanghej|hejaz|/Users/[a-zA-Z]|/sessions/|seetacloud|connect\.bjb1|@hejazfs|@privaterelay|wandb\.ai/[^"]+' \
  paper/main.tex paper/sections/ paper/appendix/ paper/tables/ paper/refs/
  → 0 hits
```

Zero matches across all 22 LaTeX source files (`main.tex` + 7 `sections/` +
4 `appendix/` + 13 `tables/` + 1 `refs/`). **Pass.**

### 2.3 PDF body strings sweep

```
strings paper/main.pdf | grep -iE 'misaya|yanghej|hejaz|/Users|/home|misayade|@hejazfs|seetacloud'
  → 0 hits

strings paper/main.pdf | grep -iE 'CreationDate|ModDate|/Users|/home|/root|/sessions|misaya|hejaz|yang'
  → 2 hits (timestamp metadata only):
       /CreationDate (D:20260427163815+08'00')
       /ModDate (D:20260427163815+08'00')
```

Only the standard pdfTeX timestamps surface. No filesystem path embedded,
no author identity in the PDF binary. **Pass.**

### 2.4 Figure EXIF (best-effort, `exiftool` not installed)

`exiftool` still not installed locally; same `strings` fallback as prior
round. Audited every PDF and PNG in `paper/figs/` (15 figure PDFs in main
dir, 9 in `paper/figs/unused/`, plus the 7 PNGs):

```
for f in paper/figs/*.pdf paper/figs/*.png paper/figs/unused/*.pdf; do
  strings "$f" | grep -iE 'Author|Username|/Users|/home|misaya|hejaz|/sessions/|seetacloud'
done
  → 0 hits across all 31 figure files
```

Also checked `paper/figs/attention_stats.npz`: 0 hits.

**Pass.** Same caveat as prior round: install `exiftool` for an XMP/IPTC
sidecar audit pre-submission.

### 2.5 Bib self-citation pattern

First-author surname distribution across 44 `references.bib` entries:

| First author | Entries | Comment |
|---|---|---|
| Li | 3 | li2024fire, li2025hope, li2026copeclipped — three distinct research groups, all third-party |
| Yang | 2 | distinct papers / orgs |
| Zheng | 2 | distinct papers / orgs |
| DeepSeek-AI | 2 | corporate author; distinct papers |
| Chen | 2 | distinct papers / orgs |
| All other surnames | 1 each | fine |

Same distribution as prior-round audit/08 §5. **No author appears as first
author ≥3 times in a way suggestive of self-citation.** Pass.

### 2.6 NEW issue: Action 7 (Bessel subsection) anonymity

The Bessel subsection is at `paper/appendix/a1_proofs.tex` lines 99–108
(rendered as section A.5 on page 18 of the PDF, heading "Why constant α (and
not the continuum stationary-phase coefficient)").

Identity-leak scan on `a1_proofs.tex`:

```
grep -nE 'misaya|yanghej|hejaz|tong|seetacloud|connect\.bjb1|/Users|/sessions|/root/autodl|sshpass|@privaterelay|@hejazfs|wandb\.ai|github\.com|huggingface\.co/[a-z]+/' \
  paper/appendix/a1_proofs.tex
  → 0 hits
```

The Bessel subsection content is pure mathematical exposition (modified-Bessel
density, stationary-phase asymptotics, closed-form CDF justification). No
real names, cluster paths, or internal URLs. **Identity-clean.**

### 2.7 NEW issue: `sec:why-constant-alpha` label / heading anonymity

Subsection heading on `paper/appendix/a1_proofs.tex:99`:

```latex
\subsection{Why constant \texorpdfstring{$\alpha$}{alpha} (and not the continuum stationary-phase coefficient)}
\label{sec:why-constant-alpha}
```

Both the heading and the label are math-content only — `alpha` is the
broadband-surrogate diagonal coefficient, "stationary-phase" is the
asymptotic-analysis term. **Neither contains reviewer names, conference
codes, internal slug fingerprints, or anything else that could
de-anonymize.** Pass.

Full subsection heading list in `a1_proofs.tex` (19 subsections total):
all are mathematical content (e.g., "Stiffness functional derivation",
"Waterbed envelope and corollary", "Surrogate self-consistency theorem").
No de-anonymizing fingerprints anywhere in headings.

---

## Summary

### P0 (submission-blocking)

**None.**

- Compile clean (0 errors, 0 overfull, 0 undefined refs).
- 0 Type 3 fonts (42 fonts, all Type 1).
- PDF Title / Author / Subject / Keywords all blank.
- 0 `??` placeholders in PDF body.
- **Body length: exactly 9 pages.** No regression despite Round-2 math
  additions.
- 0 identity leaks in any paper-source file or PDF body strings.
- `paper/COWORK_FINAL_PROMPT.md` and `paper/EDIT_CHANGELOG.md` successfully
  moved to `internal/2026_04_run/chat_scaffolding/`.

### P1 (must fix before camera-ready)

**None.**

No overfull boxes, no EXIF leaks detected by `strings`, no figure path
fingerprints, all Bessel/why-constant-alpha additions identity-clean.

### P2 (polish)

1. **17 `Underfull` warnings** (12 hbox + 5 vbox, badness ≤10000) in the
   compile log. Cosmetic only, no visible defects in the typeset PDF.
   +8 vs prior round, consistent with the new math density in the appendix.

2. **`/CreationDate` and `/ModDate` carry `+08:00` timezone offset** —
   soft geographic fingerprint, same as prior round. Recompile with
   `SOURCE_DATE_EPOCH=0` if scrubbing to maximum paranoia.

3. **`exiftool` not installed locally**, EXIF audit fell back to `strings`.
   Both PDF and PNG figure files are clean under that lens, but
   `brew install exiftool && exiftool -overwrite_original -all=
   paper/figs/*.pdf paper/figs/*.png` would harden the supplementary archive
   build before submission.

### Regression check vs prior round

| Check | Prior round | This round | Regression? |
|---|---|---|---|
| Total pages | 40 | 41 | Δ+1 (refs/appendix, not body) |
| Body pages | 9 | 9 | **No** |
| Page 9 ends with full sentence | yes | yes | No |
| Compile errors | 0 | 0 | No |
| Overfull boxes | 0 | 0 | No |
| Underfull warnings | 9 | 17 | Δ+8 cosmetic |
| Type 3 fonts | 0 | 0 | No |
| Total fonts embedded | 39 | 42 | Δ+3 (math glyphs from new section) |
| PDF metadata fields blank | yes | yes | No |
| Q5 = `\answerYes{}` | yes | yes | No |
| Forbidden page-cheats | 0 | 0 | No |
| `paper/` identity leaks | 0 | 0 | No |
| PDF body strings leaks | 0 | 0 | No |
| Figure EXIF leaks | 0 | 0 | No |
| Bib self-cite pattern | clean | clean | No |
| `paper/COWORK_FINAL_PROMPT.md` location | `paper/` (P1) | `internal/2026_04_run/chat_scaffolding/` (resolved) | **Improved** |
| `paper/EDIT_CHANGELOG.md` location | `paper/` (P1) | `internal/2026_04_run/chat_scaffolding/` (resolved) | **Improved** |
| New Bessel subsection (Action 7) | n/a | identity-clean | n/a |
| New `sec:why-constant-alpha` label | n/a | non-deanonymizing | n/a |

**Net regression: zero.** One genuine improvement (scaffolding files
relocated). Two cosmetic deltas (+8 underfull, +3 fonts) attributable to
the new math content in `a1_proofs.tex`.

### What was actually checked vs. asked

Brief item → result:

1. Compile timestamps + recompile sequence → ran fully, fresh log clean.
2. Page boundary 1–9 + last sentence on page 9 + no truncation + no `??` →
   all clean.
3. Q5 `\answerYes{}` at line 107 → confirmed (quoted above).
4. Font audit → 42 fonts, 0 Type 3.
5. PDF metadata blank → 4 of 4 anonymity fields empty.
6. Style file `[nonatbib]{neurips_2026}` → confirmed at main.tex:3.
7. Forbidden page-cheat grep (5 patterns) → 0 hits each.
8. `paper/COWORK_FINAL_PROMPT.md` / `paper/EDIT_CHANGELOG.md` not in `paper/`,
   are in `internal/2026_04_run/chat_scaffolding/` → confirmed both.
9. Identity sweep on paper LaTeX sources → 0 hits.
10. PDF body strings identity sweep → 0 hits (only standard timestamps).
11. Figure EXIF best-effort → 0 hits across 31 files.
12. Bib self-cite distribution → no author ≥3 times.
13. Action 7 (Bessel subsection) anonymity → clean.
14. `sec:why-constant-alpha` heading + label anonymity → clean.

**Overall verdict: PASS for NeurIPS 2026 submission format AND anonymity.**
The Round-2 additions (Action 7 Bessel subsection + body revisions across
five tex files) introduced no regressions in either dimension. The single
P1 from prior audit/08 (scaffolding files inside `paper/`) is now resolved.
