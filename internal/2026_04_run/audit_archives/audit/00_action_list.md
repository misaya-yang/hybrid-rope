# Action List — Synthesis of 8 Auditor Reports

**Generated**: 2026-04-27
**Source files**: `audit/01_number_trace.md` … `audit/08_anonymity.md`
**Paper commit at audit time**: `a358d6d` (clean working tree, in sync with origin/main)

---

## Counts

| Severity | Total | Categories |
|---|---:|---|
| **P0** | 3 | 2× drift in caption metadata, 1× missing reproducibility entry-point |
| **P1** | 33 | 7× numeric/std, 4× theory consistency, 9× caption verb, 2× cross-ref, 4× bib, 7× anonymity/packaging |
| **P2** | ~25 | dead labels, author truncations, regularity widening, stale aux/log, hyphenation underfulls — **default reject per stated principle** |

---

## P0 — must fix before submit

| # | finding | 严重度 | 时间 | GPU | 建议 |
|---|---|:-:|:-:|:-:|---|
| **P0-1** | `tables/table4_pe_dominant.tex:2` caption claims **"3-seed mean"**, but only Learnable τ row is 3-seed; Geo (184.9/513.7), DAPE (183.6/455.3), EVQ (182.0/333.7) are all single-seed (seed 42) per `docs/exp/2026-02-{24,25}_*.md`. Drift propagates to: `main.tex:46` (abstract "three primary 3-seed stress tests"), `sections/01_intro.tex:9`, `sections/05_experiments.tex:11,41`, `tables/table_evidence_tier.tex:13`. | P0 | 30 min | n | **APPLY**. Rewrite Table 5 caption to "1–3 seeds (Learnable τ row only is 3-seed; Geo/DAPE/EVQ are seed=42)". Soften abstract/intro to "three primary stress tests (3-seed where indicated)". Update `tab:evidence-tier` row to "1–3 seeds". Audit/01 §"Table 5 std proposals" gives drop-in LaTeX with `181.2±1.3 / 437.9±12.2` for the Learnable τ row. |
| **P0-2** | `tables/table2_evq_yarn_main.tex:2` caption claims `L_train=512, scale s=4, 16/24/32× extrapolation`. Source data (`docs/exp/2026-03-03_passkey_mix_results.md`) is **L_train=2048, scale=8** — extrapolation ratios are 4×/6×/8×. Numerical values match L=2048+s=8; the caption metadata is wrong. | P0 | 15 min | n | **APPLY**. Rewrite caption to "(454M, $L_{\mathrm{train}}=2048$, 10\% passkey mix, YaRN scale $s=8$); 8K/12K/16K is 4×/6×/8× extrapolation". Cross-check no body sentence relies on the old 16×/32× labeling. |
| **P0-3** | Q5 (`main.tex:108`) promises MLA 432M reproducibility code, but `scripts/core_text_phases/run_350m_mla32_500m.sh:6` invokes `run_gqa_evq_experiment.py` which **does not exist anywhere in the repo**. The runnable entry that produced Table 19 is missing. | P0 | 2-4h **or** 5 min wording | n | **DECIDE**. Option A (preferred, 2–4h): re-author `run_mla_432m_evq.py` from `mla_tau_optimization_v2.py` + `mla_patch.py`. Option B (5 min): soften Q5 to drop explicit "MLA 432M" code claim → "primary experiments (EVQ×YaRN 454M, PE-dominant 125M/454M); MLA 432M training script in camera-ready". |

---

## P1 — should fix before submit (sorted by ROI = impact / time)

### A. Numeric trace + std insertion (handover top priority)

| # | finding | 严重度 | 时间 | GPU | 建议 |
|---|---|:-:|:-:|:-:|---|
| A1 | Table 4 missing std on PPL (per-seed PPL not preserved locally for YaRN composition rows); std on **retrieval** column is recoverable | P1 | 10 min | n | **APPLY**: insert `±0.05 / ±0.03 / ±0.08 / ±0.00` on PK@8K column only; leave PPL means as-is. Drop-in cells in `audit/01_number_trace.md` §"Table 4 std proposals". Recover scale=8 per-seed PPL JSON from cluster for camera-ready. |
| A2 | Table 5 missing std (only Learnable τ row is 3-seed) | P1 | 5 min | n | **APPLY**: drop-in `181.2±1.3 / 437.9±12.2 / τ=1.140±0.003` from `audit/01`. Single-seed rows get no std. |
| A3 | (Bonus) `tables/table5_phase11_leverage.tex:12-14` std fully recoverable from `results/core_text/phase11/results_phase11_yarn.json` | P1 | 5 min | n | **APPLY (optional)**: drop-in `±1.2 / ±1.6 / ±0.1 / ±1.3 / ±2.0 / ±1.6` from `audit/01` (verified by `audit/scripts/verify_phase11_leverage.py`). |
| A4 | `tables/table1_multiscale_raw_ppl.tex:14` "+0.9%" disagrees with Table 6 (Geo=25.9, EVQ=26.2 → +1.16%) and source `13_UNIFIED_RESULTS_TABLE.md:16` (+1.5%) | P1 | 5 min | n | **APPLY**: change `+0.9%` → `+1.2%` for self-consistency with Table 6. |
| A5 | `appendix/a2_experiment_details.tex:42` lists 750M LR `1.5×10^-4` but `results/core_text/phase9f_750m_2k_1b/summary.json:255` records lr=3e-4 | P1 | 10 min | n | **VERIFY+APPLY**: check which 750M experiment 1.5e-4 refers to (continue@4K vs phase9f@2K). If Phase 9F is the load-bearing one, fix to 3e-4. |
| A6 | `appendix/a3_supporting_results.tex:55,62` 750M attention-viz numbers (295/432, 508-token, 2.5×) lack checked-in source script | P1 | 30 min | n | **DEFER or APPLY**: locate `results/attention_viz_v3` JSON; if exists, footnote the script. Otherwise add "computed offline; script available on request" footnote. |
| A7 | "454M" vs "350M" labeling — source data uses 350M for the same 454.2M-param model; reader may confuse it with the Table 1 "350M" row | P1 | 5 min | n | **APPLY**: insert footnote at first abstract occurrence: "454M = 454.2M parameters (24-layer transformer with $d_{\mathrm{head}}=64$); reported as 350M in some internal logs." |

### B. Theory / symbol consistency (auditor 4)

| # | finding | 严重度 | 时间 | GPU | 建议 |
|---|---|:-:|:-:|:-:|---|
| B1 | `appendix/a1_proofs.tex:147` writes "K = d_head/2"; per §3.1 convention should be `K = d_rot/2` for MLA generality | P1 | 1 min | n | **APPLY**: 1-char edit `head` → `rot`. |
| B2 | `appendix/a1_proofs.tex:548` says "α=1/d_rot derived in §sec:tau-scaling", but §sec:tau-scaling at a1:289 derives only the MHA-specific α≈1/(2K)=1/d_head form | P1 | 5 min | n | **APPLY**: rephrase to "α=1/d_rot (=1/d_head for MHA via §sec:tau-scaling, =1/d_rope for MLA via §sec:mla-results)". |
| B3 | S_χ² is used un-normalized at `a1:189` AND normalized at `§3.7:99` and `a1:386`; the disambiguating "Normalization convention" paragraph at `a1:548` appears 359 lines after first use | P1 | 5 min | n | **APPLY**: append at `a1:189` "(un-normalized integral; cf. §sec:chi2-load Normalization convention)". Optionally same parenthetical at `03_theory.tex:113` where `Var_{ρdφ}(1/ρ)` is named. |
| B4 | β=0 case is stated in body Theorem 1 (line 45) but proof at `a1:1-40` never explicitly substitutes β=0; mathematically OK but not literally proven | P1 | 5 min | n | **APPLY**: append at `a1_proofs.tex:40` one line: "For β=0 the ODE collapses to ρ''=0 with ρ'(0)=ρ'(1)=0 and ∫ρ=1, forcing ρ≡1, which is also the τ→0 limit of (37)." |

### C. Caption verbs / overclaim (auditor 5)

| # | finding | 严重度 | 时间 | GPU | 建议 |
|---|---|:-:|:-:|:-:|---|
| C1 | `appendix/a2_experiment_details.tex:153` (tab:dit-base1000 caption): "Dead-channel **validation** … phase transition … **vanishes entirely**" on single-seed (seed 42) DiT | P1 | 2 min | n | **APPLY**: "validation" → "mechanism check"; "vanishes entirely" → "is no longer present in this seed". |
| C2 | `appendix/a4_supporting_experiments.tex:57` heading "Cross-modal **confirmation**: video DiT (supporting)" — heading verb contradicts body's own "supporting mechanism check" | P1 | 2 min | n | **APPLY**: heading → "Cross-modal mechanism check: video DiT". |
| C3 | `appendix/a3_supporting_results.tex:55,108` two captions use "Waterbed **verification**" on 750M-single-seed and multiscale-mixed underlying rows | P1 | 5 min | n | **APPLY**: both occurrences "verification" → "illustration" (or "evidence"). |
| C4 | `appendix/a4_supporting_experiments.tex:36` (tab:lora-8b caption): "trades a modest cost for **dramatic** extrapolation gains" on single-seed LoRA | P1 | 2 min | n | **APPLY**: drop "dramatic"; numbers (8×, 19×) speak for themselves. |
| C5 | `paper/main.tex:46` abstract: "**surpassing** Geo+YaRN" missing matched-scale qualifier (qualifier already present in body §5.4 and `tab:mla` caption) | P1 | 1 min | n | **APPLY**: insert "matched-scale (s=4)" before "Geo+YaRN". |
| C6 | `paper/main.tex:80` Checklist Q1: "empirically **validated** O(1) scale" — λ=1 is calibrated, not validated | P1 | 1 min | n | **APPLY**: "validated" → "calibrated". |
| C7 | `paper/main.tex:94` Checklist Q3: "the surrogate itself is **validated** in Appendix" | P1 | 1 min | n | **APPLY**: "validated" → "supported by a functional check (24–92% collision-score reduction)". |
| C8 | `appendix/a3_supporting_results.tex:69` "gold-answer NLL **separates** the two RoPE variants **clearly**" on single-seed | P1 | 1 min | n | **APPLY**: drop "clearly". |
| C9 | `appendix/a3_supporting_results.tex:103` "EVQ **consistently** trades a small cost for **substantial** gains" mixes 1- and 3-seed rows | P1 | 2 min | n | **APPLY**: replace with bracketed range "EVQ trades a small in-range cost (±1.7%) for long-range gains (10–46%) across the tested grid". |

### D. Cross-references (auditor 2)

| # | finding | 严重度 | 时间 | GPU | 建议 |
|---|---|:-:|:-:|:-:|---|
| D1 | `tab:pe-yarn-l256` (Table 23) defined at `appendix/a4_supporting_experiments.tex:15` but never `\ref`'d. §5.2 cites the figure but skips the table with the actual numbers | P1 | 2 min | n | **APPLY**: in `sections/05_experiments.tex:41`, after "Appendix Fig.~\ref{fig:pe-dominant}", insert "(numbers in Appendix Table~\ref{tab:pe-yarn-l256})". |
| D2 | `fig:attn-mechanism` (Figure 6, Waterbed verification 750M) labeled at `appendix/a3_supporting_results.tex:56` but never `\ref`'d | P1 | 2 min | n | **APPLY**: add `\ref{fig:attn-mechanism}` in the introducing paragraph (the appendix §A.7 §"Attention distance redistribution" sentence currently introduces Figure 6 by proximity only). |

### E. Bib (auditor 3)

| # | finding | 严重度 | 时间 | GPU | 建议 |
|---|---|:-:|:-:|:-:|---|
| E1 | `paper/refs/references.bib:312,327` — `li2025hope` and `hua2025fope` show only one author + `and others`. Reviewer-visible flag (every other entry has full author lists) | P1 | 15 min | n | **APPLY**: expand both author lists from arXiv abs page (Li et al. HoPE / Hua et al. FoPE). |
| E2 | `paper/refs/references.bib:42` — `chen2024position` bibkey says 2024 but year=2023, arXiv 2306 (June 2023). Renders correctly as "Chen et al., 2023" | P1 | 5 min | n | **DEFER**: cosmetic; only the bibkey naming is misleading. Skip unless doing a rename pass. |
| E3 | `paper/refs/references.bib:278` — `sun2022xpos` bibkey says 2022 but year=2023 (ACL 2023) | P1 | 5 min | n | **DEFER**: cosmetic, same as E2. |
| E4 | `paper/refs/references.bib:302` — `qwen2026mhrope` claims ICLR 2026 venue; verify acceptance before camera-ready | P1 | n/a | n | **CAMERA-READY**: re-verify ICLR 2026 acceptance. If rejected, switch to arXiv-only citation. |

### F. Anonymity / packaging (auditor 6 + 8)

| # | finding | 严重度 | 时间 | GPU | 建议 |
|---|---|:-:|:-:|:-:|---|
| **F1** | TWO live SSH passwords in `internal/team/archive/recent_handoffs/2026-{02-27_8b_longinst,03-03_5090b}_handoff.md` — security incident regardless of paper | **P1+sec** | 5 min | n | **URGENT (rotate today, independent of submission)**. Also `git filter-repo` later if .git ever ships. |
| F2 | `paper/COWORK_FINAL_PROMPT.md:7` and `paper/EDIT_CHANGELOG.md:7` contain `/Users/yang/projects/...` paths. Not compiled into PDF, but inside `paper/` dir — would ship if user zips paper-source naively | P1 | 2 min | n | **APPLY**: move both files to `internal/` or delete (they are markdown chat scaffolding, not paper sources). |
| F3 | `scripts/2026-04/PAPER_HANDOVER_2026-04-27.md` contains review-strategy / score predictions / commit hashes — anonymity-clean but reviewer-game-theory disclosure | P1 | 0 min | n | **EXCLUDE**: add to rsync `--exclude` list at packaging time. |
| F4 | YAML configs promised by Q5/packaging guide don't exist; inline `CFG_*` dicts in `phase{11b,11c,17b,17c}_*.py` are runnable but mismatch packaging guide layout | P1 | 3h **or** 0 min | n | **DECIDE**. Option A (3h): externalize 3 YAML configs. Option B (recommended, 0 min wording): change Q5 from "configuration files" to "training scripts with hard-coded configurations". |
| F5 | Standalone passkey-mix prep script doesn't exist; logic fused into `scripts/core_text_phases/run_evq_sweep.py:645` (`maybe_wrap_with_passkey_mix`) | P1 | 2h | n | **CAMERA-READY**: carve out `scripts/data_prep/prepare_passkey_mix.py` (~80–120 LOC). Also fix `scripts/data_prep/README.md:9` mis-description. |
| F6 | English/anonymized supp-archive README missing; `docs/overview/REPRODUCE.md` is Chinese with `/root/autodl-tmp/...` paths | P1 | 1.5–2h | n | **CAMERA-READY**: translate + scrub into root-level `repro.md` per packaging guide. Replace cluster paths with `./data` and `./checkpoints`. |
| F7 | Cluster paths `/root/autodl-tmp/`, `/sessions/<harness-id>/` in `experiments/lora_evq_v2/*`, `scripts/2026-04/*.sh`, `scripts/analysis/unification_plot*.py` (~35 lines) | P1 | 30 min | n | **PRE-SUBMISSION**: run sed-replace at packaging time per `audit/08_anonymity.md` §8 checklist (specific commands provided). |

---

## P2 — default reject (style polish)

For completeness, ~25 P2 items identified. Per stated principle, **default reject** unless a free pass becomes available:

- **35 dead equation labels** in `appendix/a1_proofs.tex` (waterbed/leff/pure-tether blocks) — repo already uses `% \label{...} % unused` convention in places; cleanup is cosmetic.
- **5 author truncations** with `and others` beyond E1: `touvron2023llama2`, `grattafiori2024llama3`, `qwen2024qwen25`, `yang2024cogvideox`, `kong2024hunyuanvideo`, `roziere2023codellama` — standard for industry mega-author reports.
- **F2 (auditor 4)** Regularity widening C²→L² is one-sided benign; body could optionally note "proof relaxes to L²".
- **F5 (auditor 4)** §3.7 line 99 doesn't restate the d_eff=d_head calibration; already in §3.1 + §A.3.
- **F8 (auditor 4)** S_χ² axiom alias at `a1:552` resolved by paragraph adjacency to `a1:548`.
- **F9, F10 (auditor 5)** "robust default" applied to τ rule — defensible since `robust` here means low-sensitivity-to-grid; basin flatness is documented.
- **F20 (auditor 5)** "confirms" → "is consistent with" (borderline, prose-lint-adjacent).
- **F51, F52 (auditor 5)** "robustness/remarkably stable" headers in a2 — body line 218 already softens to "within the tested grid".
- **9 underfull \hbox** warnings — cosmetic hyphenation only; no visible defects.
- **Stale main.aux/main.log** in working tree (Apr 24 vs Apr 27 sources) — auditor 7's fresh recompile this run produced clean log; commit fresh aux/log or `.gitignore` them.
- **`exiftool` not installed** — fallback `strings`/`pdfinfo` audit was clean; install `brew install exiftool` for deeper XMP/IPTC sidecar pass on PNG figures before submission.

---

## Implementation order (ranked by simple time × impact)

| Stage | Items | Time | GPU | Impact |
|---|---|:-:|:-:|---|
| **1. Security** | F1 (rotate SSH passwords) | 5 min | n | independent of submission, do today |
| **2. P0 caption fixes** | P0-1 (Table 5 + abstract/intro/tier), P0-2 (Table 4 metadata) | 45 min | n | resolves every drift in audit/01 |
| **3. Top P1 std + cross-ref + theory** | A1–A4, A7, B1–B4, D1–D2 | 50 min | n | numeric/theory cleanup, all 1-line edits |
| **4. P1 caption verbs** | C1–C9 | 17 min | n | overclaim removal, all 1–2 word edits |
| **5. Pre-submission hygiene** | F2, F3, F7 | 30 min | n | move/exclude leak files; sed-replace cluster paths |
| **6. P0 entry-point** | P0-3 | 2-4h **or** 5 min | n | recover MLA runner OR soften Q5 wording |
| **7. Bib polish** | E1 (camera-ready prep) | 15 min | n | author list expansion |
| **8. P1 deferred (camera-ready)** | A5, A6, E4, F4 (Option B), F5, F6 | total ~5h | n | not submission-blocking |

**Total quick wins (stages 1–5, excluding P0-3 entry-point recovery)**: ≈2.5 hours, 0 GPU.
**Full coverage including MLA entry-point recovery**: ≈6 hours, 0 GPU.
**No GPU work required** for any item in this audit.

---

## File pointers

- Detailed findings: `audit/01_number_trace.md` … `audit/08_anonymity.md`
- Reproducible scripts: `audit/scripts/04_q_x_verify.py`, `04_Q1_grid.py`, `compute_stds.py`, `verify_phase11_leverage.py`
- Drop-in LaTeX for std insertions: `audit/01_number_trace.md` §"Tables EVQ×YaRN & PE-dominant std proposals"
- Pre-submission anonymization checklist (sed/grep commands): `audit/08_anonymity.md` §8
