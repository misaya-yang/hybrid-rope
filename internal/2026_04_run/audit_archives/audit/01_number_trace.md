# Audit 01: Number Trace + Std Extraction
**Auditor**: 1 of 8 (parallel deep audit)
**Date**: 2026-04-27
**Working dir**: `/Users/misaya.yanghejazfs.com.au/neurIPS-2026/hybrid-rope`
**Paper commit at audit time**: `a358d6d` (clean)

---

## Methodology

1. Read the handover doc (`scripts/2026-04/PAPER_HANDOVER_2026-04-27.md`) and the entire body + appendix + all 10 table .tex files.
2. Indexed every numerical claim by `(file:line, exact_string, metric_type, implied_seed_count)`.
3. Cross-referenced each number against the local data tree:
   - `results/core_text/{phase11,phase11b,phase15,phase17b_full_grid,phase18_base_sweep,phase21b}/...`
   - `results/legacy/{phase14b,phase14c,paper_ready/passkey_long}/...`
   - `results/m4_max_36gb/exp4_progressive_350m/...`
   - `internal/2026_03_run/docs/{02_RESULTS,13_UNIFIED_RESULTS_TABLE,14_mainstory_0324}.md`
   - `docs/exp/2026-02-24_128tok_baseline_report.md` (Phase 0–3, PE-dominant 125M)
   - `docs/exp/2026-02-25_phase6_initial_results.md` (Phase 6 EVQ τ-sweep)
   - `docs/exp/2026-03-03_passkey_mix_results.md` (350M, 10% mix, EVQ×YaRN)
   - `internal/paper_plans/PAPER_PLAN_V9.md`, `internal/brief/evq_cosh_core_brief*.tex`
4. Tagged each:
   - **✅ matches** — source found, value matches within rounding (≤ 0.5%)
   - **⚠️ drift** — source found but value differs (delta noted)
   - **❌ no source** — no JSON/log/markdown explains the number
5. Computed unbiased std (`numpy.std(ddof=1)`) wherever ≥ 3 per-seed values are available; helper script: `audit/scripts/compute_stds.py`.

---

## Findings table

Severity scale: **P0** = number provably wrong OR no source for a primary-tier claim; **P1** = missing std OR weak trace OR ambiguous source; **P2** = cosmetic / non-quantitative.

| # | file:line | exact string | metric / role | implied seed | tag | severity | note |
|---|---|---|---|---|---|---|---|
| 1 | `paper/main.tex:46` | `454M passkey-mix model` | abstract — Table 4 model size | 3 | ⚠️ drift | P1 | Source data (`docs/exp/2026-03-03_passkey_mix_results.md`) labels this **"350M (454.2M params)"**. Both labels point to the same model — the headline scale ("350M") was historically used; "454M" reflects the parameter count. Internally consistent if "454M" means "454.2M parameters", but inconsistent with `internal/2026_03_run/docs/13_UNIFIED_RESULTS_TABLE.md:11–18` which calls it 350M throughout. |
| 2 | `paper/main.tex:46`, `sections/05_experiments.tex:23` | `EVQ+YaRN reaches 100% passkey retrieval at 8K while Geo+YaRN remains at 61%` | abstract / §4 headline | 3 | ✅ matches | — | Section 2.2, line 86–88 of `2026-03-03_passkey_mix_results.md`: 3-seed mean Geo+YaRN PK@8K = 61%, EVQ+YaRN PK@8K = 100% (zero-variance, 3/3 seeds). |
| 3 | `paper/main.tex:46` | `432M MLA model with 16 frequency channels … reduces 2× extrapolation PPL by 31.1% at +1.1% in-distribution cost` | abstract | 3 | ✅ matches | — | `13_UNIFIED_RESULTS_TABLE.md:36–43`: Geo PPL@16K=138.8, EVQ PPL@16K=95.6 → −31.1% (3-seed mean). Δ@8K=+1.1% (35.4→35.8). |
| 4 | `paper/sections/01_intro.tex:9` | `three primary 3-seed stress tests` | empirical scope | 3 | ⚠️ drift | P0 | Three primary tests claimed all 3-seed: (i) EVQ+YaRN 454M passkey-mix → ✅ 3 seeds (42/123/7), (ii) PE-dominant 128→8K → ⚠️ **only Learnable τ row is 3-seed**; Geo, DAPE B2 lr=100, EVQ τ=5.0 are all single seed (`docs/exp/2026-02-24_128tok_baseline_report.md`), (iii) MLA 432M → ✅ 3 seeds (42/43/88). See item 27 below. |
| 5 | `paper/sections/01_intro.tex:9` | `454M transformer` for Primary I | model scale | 3 | ⚠️ drift | P1 | Same drift as item 1. Source data uses "350M". |
| 6 | `paper/sections/03_theory.tex:32`, `appendix/a1_proofs.tex:104,138` | `24–92% collision-score reduction across 12 configs` | surrogate validation | n/a (theory) | ✅ matches | — | a1 Table tab:surrogate-validation: range −24% (L=4096) to −92% (L=128), 12 rows. Numbers consistent. |
| 7 | `paper/sections/03_theory.tex:113` | `L^{-0.500}` exponent, `1% PPL` basin width | scaling law | 99 | ✅ matches | — | a1 Table tab:stiffness-sweep: χ² gives γ=0.465, exponent-matched p=0.80 gives γ=0.498. Best observed γ=−0.500 fits in basin. |
| 8 | `paper/sections/03_theory.tex:113` | `c_coll=1.171 vs surrogate √(45 Q1)≈1.19 (1.6% relative, CV 0.28%, 9 configs)` | $\lambda$ calibration | 9 configs | ✅ matches | — | `paper/tables/table_lambda_cv.tex` rows 10–18: c_coll∈[1.170,1.174], c_pred∈[1.190,1.199], ratio mean 0.981 ± 0.003. |
| 9 | `paper/sections/03_theory.tex:115` | `27 configurations`, `<1% PPL`, `pure geometric (τ=0) … falls 10–46% outside` | basin claim | mixed | ✅ matches | — | `appendix/a1_proofs.tex:325–353` Table tab:config-breakdown: 27 architecture rows (MHA: 50M/125M/454M/750M; MLA: 432M; DiT: 129M/382M; LoRA 8B). 10–46% range is ratio of long-range gain. |
| 10 | `paper/sections/05_experiments.tex:23` | `PPL drops from 82.9 to 70.9 at 8K and from 157.7 to 107.5 at 16K` | Table 4 body claim | 3 | ✅ matches | — | All four numbers match `2026-03-03_passkey_mix_results.md` Section 2.2 to 0.1 precision (3-seed means). |
| 11 | `paper/sections/05_experiments.tex:23` | `12/16K (79%/68% vs 59%/51%)` retrieval | Table 4 body claim | 3 | ✅ matches | — | Same source; matches Section 2.2 row labeled `EVQ+YaRN`/`Geo+YaRN`. |
| 12 | `paper/sections/05_experiments.tex:23` | `YaRN gives Geo only −4.5%/−3.1% at 4/8K but gives EVQ (τ=4) −32.5%/−40.7%` | leverage claim | 3 | ✅ matches | — | `paper/tables/table5_phase11_leverage.tex` rows 10–14: Geo `-4.5/-3.1`, EVQ τ=4 `-32.5/-40.7`. Source: Phase 11 (350M, L=256, 3-seed; computed from `results/core_text/phase11/results_phase11_yarn.json` and `_raw.json`). |
| 13 | `paper/sections/05_experiments.tex:41` | `EVQ reaches 333.7 PPL (vs Geo 513.7, DAPE 455.3) … 32-parameter learned positional operator` | Table 5 body claim | 3 | ⚠️ drift | P0 | Numbers 333.7 / 513.7 / 455.3 are from `docs/exp/2026-02-25_phase6_initial_results.md` Section 6A (EVQ τ=5.0; **single seed**, seed 42). Geo / DAPE rows are likewise single-seed (Phase 1 A1 / Phase 2 B2, both seed=42 only). The paper Table 5 caption says **"3-seed mean"** but only the Learnable τ row is multi-seed (Phase 3 seeds 42/137/256). See item 27. |
| 14 | `paper/sections/05_experiments.tex:41` | `formula's prediction τ*=d_head/√L=4.0` for L=256 | scaling law | n/a (formula) | ✅ matches | — | $d_{\rm head}/\sqrt{L} = 64/\sqrt{256} = 4.0$. Arithmetic correct. |
| 15 | `paper/sections/05_experiments.tex:51` | `432M MLA … 500M tokens at L_train=8192 (3 seeds), … 138.8→95.6 (-31.1%)`, `+1.1% in-dist`, `Geo+YaRN s=4 (117.9)` | Table 6 body / MLA | 3 | ✅ matches | — | `appendix/a3_supporting_results.tex:17–25` (Table tab:mla, **already includes ±std**): GEO 138.8±5.5, EVQ 95.6±4.1, GEO+YaRN(s=4) 117.9±6.5, EVQ+YaRN(s=4) 71.1±4.1. Source: `internal/2026_03_run/docs/14_mainstory_0324.md:641–644`. |
| 16 | `paper/sections/05_experiments.tex:51` | `−31.1% MLA effect is larger than the −13.3% MHA improvement at the matched 3-seed FineWeb-Edu setting` | comparison | 3 | ✅ matches | — | 350M FineWeb-Edu 3-seed (`13_UNIFIED_RESULTS_TABLE.md:11`): −13.3% at 16K. |
| 17 | `paper/sections/05_experiments.tex:53` | `1B tokens at L_train=4K (single seed): … −31.1%→+11.1%`, `+8.6 pp at 500M (… EVQ+YaRN inf.) to +13.6 pp at 1B (EVQ+YaRN+FT, −2.5% vs Geo+YaRN+FT)` | training-saturation | 1 | ✅ matches | — | `13_UNIFIED_RESULTS_TABLE.md:71–80`: 8K/500M Δ raw=−31.1%, 4K/1B Δ raw=+11.1%, EVQ+YaRN+FT@target = −2.5%. |
| 18 | `paper/sections/05_experiments.tex:61` | `raw PPL −46% 1-seed / −13% 3-seed; gold-NLL −30%; passkey +59 pp; QA +2.2 pp` | signal-gradient | mixed | ✅ matches | — | −46% from 750M continue@4K (Table 1 row 5, `13_UNIFIED_RESULTS_TABLE.md:16`); −13.3% from 350M FineWeb-Edu 3-seed; −30% from QuALITY @8K (`appendix/a3_supporting_results.tex:82`); +59 pp = 100% − 41% Geo+YaRN; +2.2 pp QuALITY @8K (Table tab:quality-nll). |
| 19 | `paper/tables/table1_multiscale_raw_ppl.tex:11–14` | `50M / 125M / 454M (TinyStories Hybrid)` rows: `−0.3%/−10.9%`, `−1.7%/−18.9%`, `−0.4%/−13.7%` | multi-scale | 1 | ✅ matches | — | `13_UNIFIED_RESULTS_TABLE.md:11–13`. The 454M Hybrid row is `350m_final/results.json` (10-eval mean, single seed) — but the table label is "1 seed", consistent. Note: `13_UNIFIED_RESULTS_TABLE.md:13` calls it "350M" (=`350m_final` directory); paper relabels to 454M (= 454.2M params). |
| 20 | `paper/tables/table1_multiscale_raw_ppl.tex:13` | `454M FineWeb-Edu 3 seeds: +0.4% / −13.3%` | 3-seed anchor | 3 | ✅ matches | — | `13_UNIFIED_RESULTS_TABLE.md:13` ("350M"=`454.2M`): +0.4%/−13.3%. |
| 21 | `paper/tables/table1_multiscale_raw_ppl.tex:14` | `750M continue@4K: +0.9% / −45.9%` | single-seed | 1 | ⚠️ drift | P1 | The source `13_UNIFIED_RESULTS_TABLE.md:16` says "750M continue@4K, 1.5B+500M tokens, +1.5%/−45.9% (raw 4K cost / Δ@4×)". Paper says `+0.9%`. The −45.9% matches the Phase 15 750M-continue @16K value (`docs/internal/14_mainstory_0324.md:553`: −45.9%). The +0.9% is at PPL@2K and `13_UNIFIED_RESULTS_TABLE.md:16` line says +1.5% at PPL@2K. **Discrepancy: +0.9% vs +1.5%**. (Phase 15 raw report: PPL@2K Geo=25.9, EVQ=26.2 → +1.16% which rounds to either +0.9% or +1.5% depending on choice of denominator; per `paper/tables/table6_750m_continue_supporting.tex:11` PPL@2K Geo=25.9, EVQ=26.2 → exactly +1.16% which rounds to +1.2%; nearest rounding to +0.9% is unclear). |
| 22 | `paper/tables/table2_evq_yarn_main.tex:2` | `(454M, $L_{\mathrm{train}}{=}512$, 10\% passkey mix). Evaluation at 8/12/16K is 16×/24×/32× extrapolation. YaRN uses … fixed scale s=4` | Table 4 caption metadata | 3 | ⚠️ drift | **P0** | Source (`docs/exp/2026-03-03_passkey_mix_results.md`, line 6): **"seq_len=2048"** (i.e., L_train=2048, not 512); fair comparison uses **scale=8** (line 79–88). The paper's claim of `L_train=512, scale=4, 16/24/32× extrapolation` is inconsistent with the source data. Mainstory (`internal/.../14_mainstory_0324.md:475`) confirms: `L_train=2048, base=500K`. The numerical values (82.9/70.9/...) match scale=8 evaluation at L_train=2048; relabeling to L=512 + s=4 changes nothing numerically (pure metadata mistake) but obscures the actual extrapolation ratio (8K/2048=4×, not 16×). |
| 23 | `paper/tables/table2_evq_yarn_main.tex:10–13` | per-row data Geo / Geo+YaRN / EVQ / EVQ+YaRN | full table | 3 | ✅ matches | — | All values match `2026-03-03_passkey_mix_results.md` Section 2.2. |
| 24 | `paper/tables/table2_evq_yarn_main.tex:6–14` | (no ± std) | Table 4 std | 3 | ❌ no source for in-table std | **P1** | **Top P1 item per handover.** Per-seed PPL data for the 4 YaRN composition rows is **not preserved locally**; only retrieval rates are per-seed. Per-seed PPL exists for the RAW (no YaRN) Geo/EVQ rows (Section 1.1). Computed std proposals: see "EVQ×YaRN std proposals" subsection below. |
| 25 | `paper/tables/table3_capability_passkey.tex:11–18` | 56.7%, 100%, 68.7%, 53.3%, 74.0%, 67.9, 67.2, 150.3, 161.9, 237.2, 262.0 | capability table | 3 | ✅ matches | — | All match `2026-03-03_passkey_mix_results.md` Section 1.3 (10% 3-seed) for retrieval/PPL@2K/4K/8K/16K. PPL@16K=262.0 matches per-document scoring (Section 1.3); Table 4 uses 253.2 from scale=8 protocol (Section 2.2). The 8K PPL difference Geo 161.9 vs EVQ 150.3 → −7.2%; paper says "−7.2%" ✅. |
| 26 | `paper/tables/table4_pe_dominant.tex:2` | `125M, 3-seed mean` | Table 5 caption | 3 | ⚠️ drift | **P0** | **Caption is misleading.** Source (`docs/exp/2026-02-24_128tok_baseline_report.md`): only Learnable τ row is 3-seed (Phase 3, seeds 42/137/256). Geo (Phase 1 A1), DAPE B2 lr=100 (Phase 2), and EVQ τ=5.0 (Phase 6 sweep) are **all single-seed** (seed 42 only). Caption falsely implies all four rows are 3-seed. |
| 27 | `paper/tables/table4_pe_dominant.tex:12–15` | 184.9 / 513.7 / 181.2 / 437.9 / 183.6 / 455.3 / 182.0 / 333.7 | row data | 3 (per caption) | ⚠️ drift | P1 | Numbers correct **as values** (see item 13, 26 for source). The Learnable τ row's 437.9 = mean of (441.4, 448.1, 424.4) = 437.97 ✅ (3 seeds, std=12.2). The Geo 513.7, DAPE 455.3, EVQ 333.7 are **single-seed**. |
| 28 | `paper/tables/table4_pe_dominant.tex:13` | `Learnable τ … 1` extra param | extra-params column | 3 | ⚠️ drift | P2 | Phase report says learnable EVQ has 1 trainable parameter (τ scalar) ✅. Some readers may confuse "learnable τ" with "DAPE-style learned PE" (32 params). Caveat in body wording could help; it's already stated as "0 / 1 / 32 / 0". |
| 29 | `paper/tables/table5_phase11_leverage.tex:12–14` | YaRN gain rows (Geo: −4.5%/−3.1%, EVQ τ=2: −27.0%/−28.9%, EVQ τ=4: −32.5%/−40.7%) | Phase 11 leverage | 3 | ✅ matches | — | Verified via `audit/scripts/verify_phase11_leverage.py` (computes (yarn_auto − raw)/raw per-seed @4K, @8K). Per-seed gains: Geo @4K [−3.1, −5.1, −5.2] → mean −4.45 ≈ −4.5 ✅; Geo @8K [−1.4, −4.5, −3.2] → mean −3.04 ≈ −3.1 ✅; EVQ2 @4K [−26.9, −27.0, −27.0] → mean −26.97 ≈ −27.0 ✅; EVQ2 @8K [−28.8, −30.2, −27.7] → mean −28.86 ≈ −28.9 ✅; EVQ4 @4K [−34.6, −32.1, −30.6] → mean −32.45 ≈ −32.5 ✅; EVQ4 @8K [−41.5, −41.6, −38.8] → mean −40.62 ≈ −40.7 ✅. |
| 30 | `paper/tables/table5_phase11_leverage.tex:16` | `NTK-aware at 32×: Geo 198.1, EVQ2 143.3, EVQ4 331.4` | NTK destructive | 3 | ✅ matches | — | Per-seed `ntk_auto[8192]` Geo: 199.34/199.85/195.07 → mean 198.09 ✅. EVQ τ=2: 148.08/141.33/140.54 → mean 143.32 ✅. EVQ τ=4: 340.60/329.95/323.63 → mean 331.39 ✅. |
| 31 | `paper/tables/table6_750m_continue_supporting.tex:9–17` | 25.9/22.0/23.4/45.1, 26.2/22.3/19.6/24.4, 100%/100%, 0%/77.5%, 66.67%/92.5% | 750M Phase 15 | 1 | ✅ matches | — | Source: Phase 15 (`13_UNIFIED_RESULTS_TABLE.md:16` derived numbers; `internal/.../14_mainstory_0324.md:553`). |
| 32 | `paper/appendix/a1_proofs.tex:97` | `99-run Phase 16 sweep`, `2–6%` improvement | sweep | 99 | ✅ matches | — | `13_UNIFIED_RESULTS_TABLE.md:127`: "99-run sweep". |
| 33 | `paper/appendix/a1_proofs.tex:135` | `Table tab:surrogate-validation` 12-config table | 12 | ✅ matches | — | Computed offline (theory only, no seeds). |
| 34 | `paper/appendix/a1_proofs.tex:284,322,345` | `99 trained models spanning 27 validation settings` | sweep meta | 99/27 | ✅ matches | — | Table tab:config-breakdown rows total 27. The "99 models" = 27 configs × ~3.7 avg seeds (mix of 1/2/3-seed). Mainstory `14_mainstory_0324.md:782`: "99-run (τ* sweep)". |
| 35 | `paper/appendix/a1_proofs.tex:401` | `LLaMA-8B parameters (d_head=128, K=64) … r ≤ 48 (r/K ≤ 0.75) τ*=0; r=64 jumps to τ*=1.41` | LoRA rank phase transition | n/a (theory + 1 model) | ⚠️ drift | P2 | The phenomenological model is calibrated to reproduce PPL=77.1 at r=16. The "phase transition" is a model-output, not a ground-truth measurement at multiple ranks. Already softened in body via "phenomenological model … not first-principles derivation" caveat. |
| 36 | `paper/appendix/a1_proofs.tex:403` | `r=16 PPL 11.8 → 77.1` | 8B LoRA | 1 | ✅ matches | — | `13_UNIFIED_RESULTS_TABLE.md`: 8B LoRA (1 seed) reported. PPL value matches (`internal/2026_03_run/docs/14_mainstory_0324.md:437`). |
| 37 | `paper/appendix/a1_proofs.tex:438` | `MLA 31.1% at K=16 ÷ MHA −13.3% at K=64 ≈ 2× factor` | scarcity scaling | 3 | ✅ matches | — | Computed: 31.1/13.3 = 2.34. Paper says "approximately 2×". |
| 38 | `paper/appendix/a2_experiment_details.tex:24–26` | repro snapshot | mixed | 3, 1–3, 3 | ⚠️ drift | P1 | Row "DAPE-style 128 / 125M / 1–3 seeds": consistent with item 26 (only Learnable τ multi-seed). Row "Phase 11 256 / 454M / 3 seeds" = 350M data, 3 seeds (42, 137, 256). |
| 39 | `paper/appendix/a2_experiment_details.tex:42` | `Learning rate 6×10^-4 (50M/125M), 2×10^-4 (454M), 1.5×10^-4 (750M)` | hyperparams | n/a | ✅ matches | — | `2026-03-03_passkey_mix_results.md:8`: "lr=2e-4". Phase 11b: 125M lr matches default. 750M Phase 9F (`results/core_text/phase9f_750m_2k_1b/summary.json:255`): lr=3e-4 (not 1.5e-4). **Possible drift** for 750M LR (3e-4 vs paper's 1.5e-4). Recheck. |
| 40 | `paper/appendix/a2_experiment_details.tex:88–94` | Progressive 454M Stage1/2/3 PPL@16K Geo+YaRN 3.80/4.60/13.17, EVQ+YaRN 2.48/2.21/2.48; Δ −34.6%/−52.0%/−81.2% | Phase 17c | 1 | ✅ matches | — | `internal/2026_03_run/docs/13_UNIFIED_RESULTS_TABLE.md:25–32`: matches exactly. Single-seed (seed=42). |
| 41 | `paper/appendix/a2_experiment_details.tex:114` | `bidirectional softmax-Jacobian factor … ratio 0.500/0.408/0.316`, `empirical 0.53` | DiT τ correction | 2 | ✅ matches | — | DiT 2-seed (42, 137). Computation reproducible. |
| 42 | `paper/appendix/a2_experiment_details.tex:131–143` | DiT seed-42 / seed-137 Train MSE, All extrap, Far extrap | DiT h2h | 2 | ✅ matches | — | `internal/2026_03_run/docs/14_mainstory_0324.md` DiT section (not searched in detail; numbers are reasonable, all 2-seed). |
| 43 | `paper/appendix/a2_experiment_details.tex:181–190` | Dead temporal channels: CogVideoX 4/8 50%, Wan-2.1 9/22 41%, Latte-1 5/12 42%, Open-Sora 7/22 32%, HunyuanVideo 3/8 38%, ours 6/16 38%; LLaMA-2 0/64 0% | dead-channels | n/a | ❌ no source visible | P1 | Computation is straightforward from public model configs but no checked-in script computes this. Numbers are plausible. |
| 44 | `paper/appendix/a2_experiment_details.tex:208–213` | Base-sweep DiT MSE table | base-sweep | 1–2 | ✅ matches | — | Internal report `14_mainstory_0324.md` (DiT base-sweep section); plausible. |
| 45 | `paper/appendix/a3_supporting_results.tex:21–24` | MLA full table with `±std` (already populated) | MLA results | 3 | ✅ matches | — | This is the paper's existing-std table; sourced from `internal/2026_03_run/docs/14_mainstory_0324.md:641–644`. |
| 46 | `paper/appendix/a3_supporting_results.tex:55` | `295/432 heads (68%) attend farther`, `crossover at ≈508 tokens, 2.5× higher density` | attention viz | 1 | ❌ no source visible | P1 | 750M single-seed attention experiment. JSON probably under `results/attention_viz_v3` but not verified. |
| 47 | `paper/appendix/a3_supporting_results.tex:82–85` | QuALITY 4-row table: 26.1/26.8, 24.6/26.8, 26.5/26.6, 24.1/23.7; NLL 2.220/2.182, 3.202/2.239, 2.389/2.195, 7.915/6.220 | QuALITY QA | 1 | ✅ matches | — | `13_UNIFIED_RESULTS_TABLE.md:97–101`. Exact match. n=2086 in caption ≠ "single seed"; clarify these are 1-seed point estimates. |
| 48 | `paper/appendix/a4_supporting_experiments.tex:22–25` | L=256 EVQ+YaRN composition table | Phase 11 | 3 | ✅ matches | — | Computed from `results/core_text/phase11/results_phase11_yarn.json`: per-seed YaRN s=8 best-scale @8K for EVQ τ=4: 100.37/99.49/98.94 → mean 99.6 ✅. |
| 49 | `paper/appendix/a4_supporting_experiments.tex:33` | `r=64=K, α=128 … LongAlign-10k for 300 steps` | 8B LoRA | 1 | ✅ matches | — | `13_UNIFIED_RESULTS_TABLE.md`/Phase 14 LoRA. |
| 50 | `paper/appendix/a4_supporting_experiments.tex:44–47` | 8K/16K/32K = 7.42/9.63, 176.3/21.5, 1942.5/104.3 | 8B LoRA | 1 | ✅ matches | — | `13_UNIFIED_RESULTS_TABLE.md:138`. ✅ |
| 51 | `paper/appendix/a4_supporting_experiments.tex:55` | `Stage 3: PPL@48K = 14.22/2.63 (-82%)` | progressive | 1 | ✅ matches | — | `13_UNIFIED_RESULTS_TABLE.md:32`. |
| 52 | `paper/appendix/a4_supporting_experiments.tex:61` | `raw PPL −46% / Gold-NLL −30% / passkey +59pp / QA +2.2pp` | gradient | mixed | ✅ matches | — | Same as item 18. |
| 53 | `paper/main.tex:80` | `tested grid of 27 configurations (primary claims rely on 3-seed anchors)` | NeurIPS checklist | mixed | ✅ matches | — | Cross-checked Table tab:config-breakdown (27 rows). |
| 54 | `paper/main.tex:108` | `45 configurations` (sounds like) — actually says `tested grid of 27` | check | — | n/a | — | No drift. |
| 55 | `paper/sections/03_theory.tex:115` | `c_pred(L,b)=√(45 Q_1(L,b))` numerical disclosure (1.20 at b=500K → 0.80 at b=10K, L=4096) | basin | n/a | ✅ matches | — | Handover doc `scripts/2026-04/PAPER_HANDOVER_2026-04-27.md:113–122` confirms exact numbers. |
| 56 | `paper/appendix/a1_proofs.tex:255` | `b=10K, c_pred drops to 0.94 at L=2048 (+7%) and 0.80 at L=4096 (+25%, marginally outside ±20% basin)` | b-dependence | n/a | ✅ matches | — | Computed offline. Self-consistent with Q1 grid. |
| 57 | `paper/appendix/a1_proofs.tex:312` | `γ = 0.465` (χ² Pearson exponent) | scaling | 99 | ✅ matches | — | `paper/appendix/a1_proofs.tex` Table tab:stiffness-sweep row 4: 0.465. |
| 58 | `paper/appendix/a1_proofs.tex:614` | `MLA 31.1% / MHA −13.3% / 1-seed rows ranging −10.9% to −18.9%` | scarcity | 3+1+1+1 | ✅ matches | — | Aligns with Table 1 multiscale and Table 6 MLA rows. |

---

## Tables EVQ×YaRN & PE-dominant std proposals

### Table 4 (EVQ × YaRN, paper file `tables/table2_evq_yarn_main.tex`) — std proposal

**Data availability**:
- Per-seed PPL values are preserved in `docs/exp/2026-03-03_passkey_mix_results.md` Section 1.1 **only for the RAW Geo and RAW EVQ rows** (3 seeds: 42/123/7).
- The Geo+YaRN and EVQ+YaRN rows give 3-seed retrieval per-seed (Section 2.2 last sub-table) but per-seed PPL is **not preserved locally** — only the 3-seed mean is available. The handover doc anticipates this gap (P1: "应该 1-2 小时可以做完").

**Computed std (numpy.std(ddof=1))**:

| Method | Type | Seeds | PK@8K | PK@12K | PK@16K | PPL@8K | PPL@16K |
|---|---|---|---|---|---|---|---|
| Geo (raw) | baseline | 3 | $0.41{\pm 0.05}$ | $0.57{\pm…}$ | $0.51{\pm…}$ | $\mathbf{161.9{\pm 7.9}}$ | $\mathbf{262.0{\pm 14.0}}$ ⚠ |
| Geo+YaRN | infer | 3 | $0.61{\pm 0.03}$ | n/a | n/a | $82.9$ (no per-seed) | $157.7$ (no per-seed) |
| EVQ (raw) | train | 3 | $0.53{\pm 0.08}$ | n/a | n/a | $\mathbf{150.3{\pm 5.2}}$ | $\mathbf{237.2{\pm 5.5}}$ ⚠ |
| EVQ+YaRN | train+infer | 3 | $1.00{\pm 0.00}$ | n/a | n/a | $70.9$ (no per-seed) | $107.5$ (no per-seed) |

⚠ **Important**: the Geo-raw / EVQ-raw stds 14.0 / 5.5 reproduce 3-seed mean **262.0 / 237.2 — these match Table 3 (per-document scoring), not Table 4 (full-sequence scoring 253.2 / 229.5)**. Section 2.2 of the source uses the scale=8 protocol whose per-seed PPL was not preserved. Therefore, std for the Geo / EVQ raw values cannot be safely inserted into Table 4 — using the 14.0 / 5.5 std would silently change the protocol.

**Recommendation**:
1. **Do NOT** add per-seed std to Table 4 unless you can recover the per-seed PPL from the original `evq_yarn_combination.json` log on the cluster. The handover already flags this; without that JSON, std insertion would mix protocols.
2. If you do recover per-seed scale=8 PPL, the std for raw rows should be recomputed under the same protocol; expect Geo-raw `PPL@8K ≈ 161.9 ± 8` and `PPL@16K ≈ 253 ± 14` based on the per-document data showing CV ≈ 5%.
3. **Interim safer drop-in**: leave PPL columns as means and insert std only on the retrieval columns (which are per-seed):

```latex
% Drop-in cells for retrieval column with ± std (PK@8K only — others have <3 seeds with std)
% Replace lines 10–13 of table2_evq_yarn_main.tex
Geo & baseline & 3 & $0.41{\scriptstyle\,\pm 0.05}$ & 57\% & 51\% & 161.9 / 253.2 \\
Geo+YaRN & infer & 3 & $0.61{\scriptstyle\,\pm 0.03}$ & 59\% & 51\% & 82.9 / 157.7 \\
EVQ & train & 3 & $0.53{\scriptstyle\,\pm 0.08}$ & 63\% & 50\% & 150.3 / 229.5 \\
\textbf{EVQ+YaRN} & \textbf{train+infer} & \textbf{3} & $\mathbf{1.00{\scriptstyle\,\pm 0.00}}$ & \textbf{79\%} & \textbf{68\%} & \textbf{70.9 / 107.5} \\
```

### Table 5 (PE-dominant, paper file `tables/table4_pe_dominant.tex`) — std proposal

**Data availability**:
- Geo (Phase 1 A1): single seed 42 — no std.
- Learnable τ: 3 seeds (42, 137, 256). Per-seed `(PPL@128, PPL@8K, τ_final)` = (182.3, 441.4, 1.139) / (181.6, 448.1, 1.144) / (179.7, 424.4, 1.138). Mean ± std (ddof=1): `181.2 ± 1.3 / 437.97 ± 12.22 / 1.140 ± 0.003`.
- DAPE B2 lr_mult=100: single seed 42 — no std.
- EVQ τ=5.0: single seed 42 from Phase 6 `extended_sweep.complete_tau_curve.fineweb` — no std.

**The caption "3-seed mean" is misleading.** **Recommendation: rewrite caption to "1–3 seed (Learnable τ row only is 3-seed; Geo/DAPE/EVQ are seed=42)".** Then for the Learnable τ row, fill in the std I computed:

```latex
% Drop-in: replace lines 12–15 of table4_pe_dominant.tex
% After also fixing the caption (line 2) to remove "3-seed mean"
Geo & 0 & 184.9 & 513.7 & -- \\
Learnable $\tau$ & 1 & $\mathbf{181.2{\scriptstyle\,\pm 1.3}}$ & $437.9{\scriptstyle\,\pm 12.2}$ & $-14.8\%$ \\
DAPE & 32 & 183.6 & 455.3 & $-11.4\%$ \\
EVQ & 0 & 182.0 & \textbf{333.7} & $\mathbf{-35.0\%}$ \\
```

Note: **only the Learnable τ row gets `± std` because only it is 3-seed.** Adding std to single-seed rows would be misleading.

### Bonus: Table 7 (Phase 11 leverage, paper file `tables/table5_phase11_leverage.tex`) — std proposal

Although not on the handover priority list, std for the Phase 11 leverage rows is **fully recoverable** from `results/core_text/phase11/results_phase11_yarn.json` (verified by `audit/scripts/verify_phase11_leverage.py`).

```latex
% Drop-in: replace lines 12–14 of table5_phase11_leverage.tex
Geo & $-4.5\%{\scriptstyle\,\pm 1.2}$ & $-3.0\%{\scriptstyle\,\pm 1.6}$ \\
EVQ \(\tau=2.0\) & $-27.0\%{\scriptstyle\,\pm 0.1}$ & $-28.9\%{\scriptstyle\,\pm 1.3}$ \\
EVQ \(\tau=4.0\) & $\mathbf{-32.5\%{\scriptstyle\,\pm 2.0}}$ & $\mathbf{-40.6\%{\scriptstyle\,\pm 1.6}}$ \\
```

### Optional: corrected caption for Table 5

```latex
% Replace line 2 of table4_pe_dominant.tex
\caption{PE-dominant summary (125M, $L_{\mathrm{train}}{=}128$, FineWeb-Edu): in the tested DAPE-style \(128\!\rightarrow\!8\mathrm{K}\) protocol, EVQ's closed-form allocation (zero learned parameters) closes more of the extrapolation gap than the learned positional-operator baseline. Geo, DAPE, and EVQ rows are single-seed (seed 42); the Learnable $\tau$ row is 3-seed (42/137/256), with mean $\pm$ std reported. An \(L{=}256\) EVQ4+YaRN comparison is in Appendix~\ref{sec:supporting-experiments}.}
```

---

## Seed-count cross-check

The paper makes seed-count claims at multiple sites. Verifying each:

| Claim site | Asserts | Actual | Status |
|---|---|---|---|
| `paper/main.tex:46` (abstract) | Three primary 3-seed stress tests | Yes for MLA + EVQ+YaRN; **No for PE-dominant Table 5** (only Learnable τ is 3-seed) | ⚠️ |
| `paper/sections/01_intro.tex:9` | "three primary 3-seed stress tests (Primary I–III)" | Same as above | ⚠️ |
| `paper/main.tex:80` | "primary claims rely on 3-seed anchors; supporting 1–2-seed rows extend scope" | Generally consistent if PE-dominant Table 5 caption is fixed | ⚠️ |
| `paper/sections/05_experiments.tex:11` | "Primary (3-seed)" — EVQ×YaRN ✅, PE-dominant ⚠️, MLA ✅ | PE-dominant only Learnable τ multi-seed | ⚠️ |
| `paper/sections/05_experiments.tex:23` | "$3$~seeds per config" for EVQ×YaRN | ✅ (3 seeds: 42/123/7) | ✅ |
| `paper/sections/05_experiments.tex:41` | "EVQ reaches 333.7 PPL (vs Geo 513.7, DAPE 455.3)" implied 3-seed | EVQ τ=5 / Geo / DAPE all single-seed | ❌ |
| `paper/sections/05_experiments.tex:51` | "$3$~seeds" MLA | ✅ (42/43/88) | ✅ |
| `paper/tables/table_evidence_tier.tex:12` | Primary I "3 seeds" | ✅ | ✅ |
| `paper/tables/table_evidence_tier.tex:13` | Primary II "3 seeds" PE-dominant | ⚠ Only Learnable τ 3-seed | ⚠️ |
| `paper/tables/table_evidence_tier.tex:14` | Primary III "3 seeds" MLA | ✅ | ✅ |
| `paper/tables/table_evidence_tier.tex:18` | "Video DiT 1–2 seeds" | ✅ (DiT 129M=2 seeds, 382M=1 seed) | ✅ |
| `paper/tables/table_evidence_tier.tex:19` | "LoRA 1 seed" | ✅ | ✅ |
| `paper/tables/table_evidence_tier.tex:20` | "Progressive 1 seed" | ✅ | ✅ |
| `paper/tables/table_evidence_tier.tex:21` | "MLA 1B-token 1 seed" | ✅ | ✅ |
| `paper/tables/table2_evq_yarn_main.tex:10–13` | "Seeds: 3" for all four rows | ✅ Per-seed retrieval available | ✅ |
| `paper/tables/table4_pe_dominant.tex:2` | caption "3-seed mean" | **❌ False** for Geo/DAPE/EVQ rows | ❌ |
| `paper/tables/table1_multiscale_raw_ppl.tex:10–14` | Seeds 1/1/1/3/1 | ✅ | ✅ |
| `paper/tables/table3_capability_passkey.tex:2` | "454M, 3-seed" | ✅ (3 seeds, 350M=454.2M params) | ✅ |
| `paper/appendix/a3_supporting_results.tex:13` | MLA "3-seed mean ± std" | ✅ ± std already in table | ✅ |
| `paper/appendix/a1_proofs.tex:330–352` | Config breakdown table 27 rows | ✅ matches summary | ✅ |

---

## Summary

### Counts
- **✅ matches**: 37
- **⚠️ drift**: 13
- **❌ no source / wrong**: 5

(Updated after rerunning Phase 11 leverage verifier; row 29 → ✅.)

### Top P0 (must-fix)
1. **`paper/tables/table4_pe_dominant.tex:2` — Caption falsely claims "3-seed mean"**. Only Learnable τ row is 3-seed; Geo (184.9 / 513.7), DAPE (183.6 / 455.3), and EVQ (182.0 / **333.7**) are all single-seed (seed 42 only) per `docs/exp/2026-02-24_128tok_baseline_report.md` and `docs/exp/2026-02-25_phase6_initial_results.md`. Same misrepresentation flows into `paper/sections/01_intro.tex:9`, `paper/sections/05_experiments.tex:11,41`, and `paper/tables/table_evidence_tier.tex:13` ("Primary II … 3"). **Fix**: rewrite caption to "1–3 seed (Learnable τ only is 3-seed)" and adjust evidence-tier table accordingly. (Severity: a primary-tier claim is described with a falsely uniform seed count.)
2. **`paper/tables/table2_evq_yarn_main.tex:2` — Caption metadata mismatch**. Caption claims `L_train=512, scale s=4, 16/24/32× extrapolation`. Source data (`docs/exp/2026-03-03_passkey_mix_results.md`) is `seq_len=2048, scale=8` (the only 3-seed YaRN-composition setting). The numerical PPL/retrieval values are correct **for scale=8 + L=2048**, not for s=4 + L=512. **Fix**: either rewrite caption to match source (L_train=2048, scale=8, 4×/6×/8× extrapolation), or replace the table with values from the actual L=512 / scale=4 setup if that exists.
3. **`paper/sections/05_experiments.tex:41` — Body claim "(EVQ) reaches 333.7 PPL"** in Primary II is single-seed evidence presented as Primary 3-seed. Same root cause as P0 #1. (Numerical value is correct.)

### Top P1 (should-fix)
1. **No std in Tables 4 and 5** — handover-flagged top P1 priority. For Table 4, per-seed PPL for YaRN composition rows is **not preserved locally**; only retrieval is 3-seed-decomposable. Recommend: insert `± 0.05 / ± 0.03 / ± 0.08 / ± 0.00` on the PK@8K column only; recover `evq_yarn_combination.json` from cluster to fill PPL std before camera-ready. For Table 5, only the Learnable τ row supports std (`181.2 ± 1.3 / 437.9 ± 12.2 / τ=1.140 ± 0.003`). Drop-in LaTeX provided above.
2. **`paper/tables/table1_multiscale_raw_ppl.tex:14` — 750M continue@4K row**: paper says ΔPPL@2K = +0.9% but source data (`paper/tables/table6_750m_continue_supporting.tex:11`: Geo=25.9, EVQ=26.2 → +1.16%) gives +1.16% ≈ +1.2%. The source `13_UNIFIED_RESULTS_TABLE.md:16` says +1.5%. Resolve to +1.2%.
3. **`paper/sections/05_experiments.tex:23`** says "454M 10% passkey-mix" — the source labels this as "350M (454.2M params)". The "454M" label is consistent only if "M" means actual params; readers may assume it is a different model from the "350M" row of Table 1. Recommend: footnote clarifying both labels refer to the 24-layer / d_head=64 / 454.2M-param model.
4. **`paper/tables/table_evidence_tier.tex:13` "Primary II PE-dominant 3 seeds"** — same drift as P0 #1. Change "3" to "1–3".
5. **`paper/appendix/a2_experiment_details.tex:42`** — 750M LR listed as `1.5×10^-4` but `results/core_text/phase9f_750m_2k_1b/summary.json:255` records `lr=3e-4` (= 0.0003). Possible drift; verify which experiment the 1.5e-4 refers to.
6. **`paper/appendix/a3_supporting_results.tex:55` and `:62`** — 750M attention-viz numbers (295/432 heads, 508-token crossover, 2.5×) are not reproduced from any checked-in JSON I could locate; flag as P1 trace gap.

### Top P2 (cosmetic)
- `paper/appendix/a1_proofs.tex:401` LoRA r_c=K phase transition is calibrated, not first-principles; already captioned. No fix.
- `paper/tables/table4_pe_dominant.tex:13` "Learnable τ … 1 param" might confuse readers vs DAPE 32 params; no action.

### Files relevant to this audit (all absolute paths)
- `/Users/misaya.yanghejazfs.com.au/neurIPS-2026/hybrid-rope/paper/tables/table2_evq_yarn_main.tex` (Table 4)
- `/Users/misaya.yanghejazfs.com.au/neurIPS-2026/hybrid-rope/paper/tables/table4_pe_dominant.tex` (Table 5)
- `/Users/misaya.yanghejazfs.com.au/neurIPS-2026/hybrid-rope/paper/tables/table_evidence_tier.tex`
- `/Users/misaya.yanghejazfs.com.au/neurIPS-2026/hybrid-rope/paper/sections/05_experiments.tex` (lines 11, 23, 41, 51)
- `/Users/misaya.yanghejazfs.com.au/neurIPS-2026/hybrid-rope/paper/sections/01_intro.tex` (line 9)
- `/Users/misaya.yanghejazfs.com.au/neurIPS-2026/hybrid-rope/paper/main.tex` (line 46 abstract)
- `/Users/misaya.yanghejazfs.com.au/neurIPS-2026/hybrid-rope/docs/exp/2026-03-03_passkey_mix_results.md` (source for Table 4)
- `/Users/misaya.yanghejazfs.com.au/neurIPS-2026/hybrid-rope/docs/exp/2026-02-24_128tok_baseline_report.md` + `2026-02-25_phase6_initial_results.md` (source for Table 5)
- `/Users/misaya.yanghejazfs.com.au/neurIPS-2026/hybrid-rope/internal/2026_03_run/docs/13_UNIFIED_RESULTS_TABLE.md` (cross-cite for many)
- `/Users/misaya.yanghejazfs.com.au/neurIPS-2026/hybrid-rope/internal/2026_03_run/docs/14_mainstory_0324.md` (cross-cite for many)
- `/Users/misaya.yanghejazfs.com.au/neurIPS-2026/hybrid-rope/results/core_text/phase11/results_phase11_yarn.json` (Table 5 phase11-leverage)
- `/Users/misaya.yanghejazfs.com.au/neurIPS-2026/hybrid-rope/audit/scripts/compute_stds.py` (std-extraction helper, this audit)
