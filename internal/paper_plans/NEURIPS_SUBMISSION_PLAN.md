# NeurIPS Anonymous Submission Plan

> Canonical plan for the anonymous NeurIPS submission package.
> Scope: redline compliance, 9-page body budget, citation allocation, and execution order.
> Status: implemented in `paper_draft/submission/`.

---

## 1. Official Redlines

### 1.1 Current baseline

NeurIPS 2026 style files are not yet public. The current submission baseline is:

- NeurIPS 2025 Call for Papers:
  `https://neurips.cc/Conferences/2025/CallForPapers`
- Official NeurIPS style instructions:
  `https://media.neurips.cc/Conferences/NeurIPS2023/Styles/neurips_2023.pdf`

When the 2026 package is released, run a delta-compliance pass and only update format-specific details.

### 1.2 Hard constraints

1. Main paper body must be at most 9 pages.
   - Count from title/abstract through the end of conclusion.
   - References do not count.
   - Appendix does not count and must come after references.
2. The official NeurIPS style file must be used.
   - No `article + geometry` submission file.
3. No manual spacing or layout hacks.
   - No custom margins.
   - No negative `\vspace`.
   - No font size or line spacing overrides.
4. Anonymous submission only.
   - No author names, institutions, acknowledgments, or funding disclosure in the review version.
   - Self-citations must remain third-person and non-identifying.
5. Submission mode with line numbers must be used for page budgeting.
   - Do not budget pages in preprint mode.
6. The paper checklist must be present after references/appendix material.
7. Abstract target is 130-160 words.
   - This is an internal writing target, not a separate official redline.

### 1.3 Local corrections

These are not treated as current hard redlines:

- a mandatory standalone broader-impact section in the 9-page body
- an abstract hard cap of exactly 200 words
- keeping Figure 1 in the main body

---

## 2. Paper Thesis and Primary Claims

### 2.1 One-sentence thesis

**Standard RoPE fails not because inference-time scaling is too weak, but because its training-time geometric frequency allocation is a degenerate point of a broader variational optimum.**

### 2.2 Only three body-level claims

1. **Closed-form theory**
   - RoPE frequency allocation is a variational inverse problem.
   - After fixing the broadband surrogate, the ODE and closed-form solution are exact.
   - Geometric RoPE is the `tau = 0` degenerate limit.

2. **Extreme extrapolation beats learnable PE**
   - In the DAPE-style PE-dominant regime, EVQ beats learnable PE with `0` extra parameters.
   - This shows that PE quality itself can determine extreme extrapolation.

3. **EVQ unlocks YaRN**
   - The most important systems result is not `EVQ > Geo` alone.
   - It is **`EVQ + YaRN >> Geo + YaRN`**.
   - EVQ acts as the training-time foundation that makes inference-time scaling effective.

### 2.3 Explicitly demoted results

These are secondary/supporting only:

- `750M phase9f` Hybrid dynamics
- the single-seed `+40pp` passkey outlier
- `video confirms tau*=2.0`
- Hybrid strict-superiority as a main theorem
- an independent `r`-sweep main figure

---

## 3. Nine-Page Body Budget

| Block | Pages | Notes |
|------|------:|------|
| Title + Abstract | 0.45 | Abstract target 130-160 words |
| 1. Introduction | 0.95 | Problem, method, two killer results |
| 2. Related Work | 0.45 | Four short paragraphs only |
| 3. Theory / Method | 2.10 | Only paper-grade theory |
| 4. Predictions / Mechanism | 0.55 | Waterbed, spacing asymmetry, YaRN mechanism |
| 5. Experiments | 4.05 | Main empirical body |
| 6. Limitations / Scope | 0.30 | One short section |
| 7. Conclusion | 0.15 | Two-sentence close |
| **Total** | **9.00** | Budget against submission mode only |

---

## 4. Section Plan

### 4.1 Introduction

Must do:

- state the gap: inference-time scaling is heavily studied, training-time geometric frequency allocation is not
- state the method: variational formulation, ODE, closed-form EVQ-cosh, geometric as `tau=0`
- state the two killer empirical results
- end with at most 4 contributions

Numbers that must appear:

- `EVQ+YaRN@8K = 100% across 6/6 seeds`
- `Geo+YaRN = 61%-65%`
- `128->8K: EVQ 333.7 vs DAPE 455.3`
- `L=256: EVQ4+YaRN 99.6 vs Geo+YaRN 260.2`

Forbidden:

- no `+40pp` intro headline
- no video in intro
- no `750M phase9f` dynamics in intro

### 4.2 Related Work

Exactly four compact paragraphs:

1. rotary and relative positional methods
2. context extension / inference-time scaling
3. learnable or adaptive PE
4. long-context evaluation

### 4.3 Theory / Method

Body-level theory only:

- RoPE as frequency allocation
- exact kernel and broadband surrogate
- variational objective and ODE
- closed-form EVQ and geometric limit
- waterbed and scaling-law statement

Identity discipline:

- theorem: exact ODE / solution / geometric limit
- proposition: waterbed under explicit assumptions
- empirical law / conjecture: `tau*(L) = d_head / sqrt(L)`

### 4.4 Predictions / Mechanism

Do only three things:

1. waterbed intuition
2. bounded short-range cost
3. why EVQ should help inference-time scaling

### 4.5 Experiments

#### 5.1 Setup

One paragraph only:

- from-scratch
- `base=500K`
- `L_train`
- seeds
- only `inv_freq` changes
- all details to appendix

#### 5.2 Main systems result

Use:

- **Figure 2**
- **Table 2**

Claim:

- `EVQ+YaRN >> Geo+YaRN`
- training-time and inference-time optimization are orthogonal and complementary

#### 5.3 Extreme extrapolation and scaling-law confirmation

Use:

- **Figure 3**

Purpose:

- show closed-form EVQ beats learnable PE in DAPE-style extrapolation
- show `tau*` is directly confirmed at `L=256`
- tie theory to empirical prediction

#### 5.4 Robustness and capability preservation

Use:

- **Table 1**
- **Table 3**

Must cover:

- multi-scale raw length generalization
- `10% mix` raw retrieval gain
- `5% -> 10%` robustness gap
- empirical proposition: non-destructive + learned retrieval gain

`750M continue@4K` treatment:

- one supporting paragraph only
- `PPL@16K -45.9%`
- `8K AR exact 77.5% vs 0%`
- explicitly single-seed supporting evidence

#### 5.5 Rebuttal paragraph

One compact paragraph only:

- `base=10K` negative result becomes theory-confirming evidence
- larger-scale evidence is directionally consistent

### 4.6 Limitations / Scope

Must include:

- surrogate approximation is still the only approximation step
- `tau*` remains empirical law / conjecture
- main downstream evidence is still retrieval-heavy
- `750M continue` is single-seed
- video remains supportive, not co-primary

### 4.7 Conclusion

Two sentence target:

1. closed-form variational solution and geometric degeneracy
2. EVQ unlocks YaRN and materially improves long-context extrapolation

---

## 5. Body Figures and Tables

### 5.1 Main body only

Figures:

- **Figure 2**: `EVQ x YaRN`
- **Figure 3**: `DAPE-style extreme extrapolation + Phase 11`

Tables:

- **Table 1**: multi-scale raw PPL summary
- **Table 2**: killer table for `EVQ+YaRN`
- **Table 3**: capability-preserving + passkey mix

### 5.2 Appendix only

- Figure 1: 750M dynamics
- Figure 4: collision-block / dead zone
- full phase9f tables
- full phase15 tables
- `r`-sweep
- video temporal tables/figures
- implementation details and hyperparameters

### 5.3 Decision rule

Figure 1 is supporting only. Keeping it in the body would crowd out Figures 2 and 3 and weaken the main submission package.

---

## 6. Citation Plan

### 6.1 Target

- Body target: `26-34` references
- Enough for coverage, not a mini-survey

### 6.2 Citation buckets

| Bucket | Count | Body role |
|--------|------:|----------|
| Transformer / base PE background | 4-6 | Intro + Related |
| Inference-time scaling | 4-6 | Related + Section 5.2 |
| Learnable / adaptive PE | 3-5 | Section 5.3 |
| Long-context evaluation | 3-5 | Experiments |
| Model / config facts | 2-4 | Minimal factual support |
| Video / multimodal temporal RoPE | 2-4 | Appendix or discussion by default |

### 6.3 Section mapping

- Introduction:
  - RoFormer / RoPE
  - PI / YaRN / LongRoPE
  - DAPE / FIRE / Kerple
- Related Work:
  - carries most citations
- Theory / Method:
  - only cite what is genuinely necessary
- Experiments:
  - Section 5.2: YaRN / PI / LongRoPE
  - Section 5.3: DAPE / FIRE / Kerple
  - Section 5.4: LongBench / RULER / passkey-style evaluation

### 6.4 Citation redlines

1. Use `natbib` style commands.
2. No citation footnote dumping.
3. No identity-revealing self-citation phrasing.
4. Every non-obvious factual claim gets a citation or direct experiment support.
5. Every body figure/table must be referenced before it appears.

---

## 7. Implemented File Structure

The initial anonymous submission skeleton is implemented at:

```text
paper_draft/submission/
  main.tex
  neurips_2025.sty
  README.md
  sections/
    01_intro.tex
    02_related.tex
    03_theory.tex
    04_predictions.tex
    05_experiments.tex
    06_limitations.tex
    07_conclusion.tex
  tables/
    table1_multiscale_raw_ppl.tex
    table2_evq_yarn_main.tex
    table3_capability_passkey.tex
  refs/
    references.bib
```

Body figures referenced by the submission skeleton:

- `paper_draft/figs/fig2_evq_yarn_synergy.pdf`
- `paper_draft/figs/fig3_pe_dominant_scaling.pdf`

Supporting figure referenced for appendix planning:

- `paper_draft/figs/fig1_frequency_dynamics.pdf`

---

## 8. Execution Order

1. compile the anonymous skeleton in submission mode
2. lock Figure 2 and Figure 3 placement
3. finish Introduction and Section 5.2 first
4. write theory only after theorem/proposition/conjecture identities are fixed
5. write Section 5.3 next so the theory-to-experiment bridge is explicit
6. write robustness and limitation sections after the main claims are frozen
7. write abstract last

---

## 9. Acceptance Tests

### 9.1 Format compliance

1. submission mode compiles
2. main body <= 9 pages
3. anonymous review version contains no author-identifying material
4. no manual spacing hacks
5. all body figures/tables remain inside the 9-page limit

### 9.2 Content consistency

1. all `P0` claims have a main figure or main table
2. all `P2` evidence is explicitly supporting only
3. no single-seed result is promoted to a primary headline
4. video is not written as co-primary evidence
5. `tau*` is not promoted to an unconditional theorem

### 9.3 Citation checks

1. no dead bib entries
2. no uncited body figures/tables
3. no identity-revealing self-citation phrasing

