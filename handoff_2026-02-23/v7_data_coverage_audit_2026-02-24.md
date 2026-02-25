# v7 Data Coverage Audit (2026-02-24)

## Scope
- Goal: verify whether currently strongest **VALID** evidence is already reflected in the v7 anonymous paper.
- Paper target: `/Users/misaya.yanghejazfs.com.au/download/releases/neurips/hybrid_rope_neurips_submission_v7_anonymous.tex`
- Data sources:
  - `handoff_2026-02-23/01_结果真值表_VALID_INVALID.md`
  - `results/theory_2026-02-22/*`
  - `archives/batch_report_2026-02-23_downstream_eval/report/*`
  - `.../significance_seeded_v7_refresh.{json,csv}` (local refresh run)

## Coverage Verdict
- **A1 from-scratch scaling (50M/100M/350M): INCLUDED**
  - v7 Table 1 reports the expected 16K improvements:
    - 50M: `-10.2%`
    - 100M: `-13.5%`
    - 350M: `-13.7%`
- **A1 YaRN comparison (50M): INCLUDED**
  - v7 Table 2 keeps the corrected progressive YaRN comparison (`EXP_50M_YARN`).
- **A2 theory calibration: INCLUDED**
  - v7 Section 4.3 + Table 3 + Figure 4 contain:
    - `R^2_mid ≈ 0.9942` (`b=1e4`)
    - `R^2_mid ≈ 0.9954` (`b=5e5`)
    - `L/b` transition statement (crossing only at high ratio).
- **8B fair-protocol core table: INCLUDED**
  - v7 Table 4 values match `method_metrics_best_available.csv`:
    - Anchored LongBench `0.0717`
    - PI `0.0665`
    - YaRN `0.0656`
    - Baseline `0.0626`
    - Sigmoid `0.0687`

## 8B Relative Improvement Check (memory: “8–9%”)
- Anchored vs PI: `+7.8%`
- Anchored vs YaRN: `+9.3%`
- Anchored vs Baseline: `+14.5%`
- Anchored vs Sigmoid (ablation): `+4.4%`

These are exactly consistent with the v7 discussion paragraph and with the report CSV.

## Statistical Status (fair 8B)
- Task-level paired results remain non-significant (v7 already states this).
- Refresh run (`significance_seeded_v7_refresh.json`) per-sample block is **not claim-grade** here:
  - archived snapshot exposes preview-length per-sample traces (effective `n=18` pooled), not full `6*80`.
  - therefore task-level framing (`n=6`) should remain primary in main text.

## “Solid but not in main claim” items
- Reviewer-oriented generated artifacts under `artifacts/reviewer_2026-02-24/` (tuning/prior bridge variants) are useful diagnostics, but not all are protocol-locked against the main fair suite.
- Keep these as internal analysis or appendix-ready candidates, not core acceptance evidence unless protocol lock is explicitly documented.

## Action for v7 text
- No numeric correction required for the 8B fair table/discussion.
- Keep the current wording emphasis:
  - directional/mechanistic validation;
  - explicit non-significance disclosure;
  - unit clarification (`0--1` raw equals percentage-style points ×100).
