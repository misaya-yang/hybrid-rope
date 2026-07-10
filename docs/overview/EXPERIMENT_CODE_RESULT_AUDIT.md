# Experiment Code/Result Audit

Purpose: answer the reviewer-facing question "is there code for this experiment,
does the code path implement the claimed protocol, and is there a JSON/result
artifact?" This is deliberately separate from checkpoint/data provenance. A row
can be code-supported and still lack checkpoint hashes.

Status scale:

- `A`: current code plus current JSON/curated JSON supports the reported value.
- `B`: code exists and writes JSON, but the exact raw JSON is archival/branch or
  curated rather than fully packaged in the compact branch.
- `C`: code exists, but current branch only has a Markdown report for the row.
- `D`: missing or broken code/result chain.

## Main Audit Matrix

| Experiment / claim row | Code support | Implementation health | JSON/result artifact | Status |
| --- | --- | --- | --- | --- |
| Table 2 EVQ x YaRN, 454M, 10% passkey mix | `scripts/core_text_phases/run_evq_sweep.py`, `scripts/supporting_eval/eval_passkey_scratch.py`, figure reader `scripts/figures/fig2_evq_yarn_orthogonality.py`; `phase14c_multiscale_evq_yarn.py` is supporting only. | Healthy for reported metric: PK helper writes NLL-gap JSON, figure reads curated 10% 3-seed JSON, and wording is now teacher-forced NLL-gap. | Current: `data/curated/table2_evq_yarn_454m_passkey_10pct.json`. Archival branch also has `data/results_5090b/evq_yarn_10pct_allseeds.json`. | `A` for table values; `B` for raw run-level replay. |
| Table 4 / Fig. 3(a) PE-dominant 128-to-8K diagnostic | `scripts/core_text_phases/phase11b_125m_dape.py` trains/evals Geo, EVQ, and DAPE-style rows and writes per-run `result.json` plus `all_results.json`. | Healthy for panel/table scope. Seed scope must stay explicit: Geo/DAPE/EVQ are seed 42; learnable tau is multi-seed mean/std. | Current: `data/curated/fig3_extreme_128.json`. | `A` for Table 4 / panel (a). |
| Fig. 3(b,c) Phase 11 scaling panels | `scripts/figures/fig3_pe_dominant_scaling.py` expects Phase 11 raw and YaRN JSONs; `phase11b_125m_dape.py`, `phase11_L256_extrap.py`, and `phase11_yarn_eval.py` are the relevant generators. | Code path is clear, but compact branch does not include the expected `results/core_text/phase11/results_phase11_raw.json` and `results_phase11_yarn.json`. | Visible archival branch has `results/phase11/results_phase11_raw.json` and `results/phase11/results_phase11_yarn.json`; compact branch does not. | `B`; add curated fallbacks or restore sanitized JSON before claiming one-command figure regeneration. |
| Primary MLA 8K/500M 3-seed stress test | `scripts/core_text_phases/run_gqa_evq_experiment.py` trains MLA/GQA/MHA and writes per-run `results.json`, `inv_freq.npy`, and work-dir `summary.json`; `scripts/core_text_phases/eval_extended_3seeds.py` evaluates 3 seeds and writes `eval_3seeds_full_results.json`. | Improved in this audit: eval now explicitly uses checkpoint-loaded `inv_freq` before YaRN and resolves both current `350m_mla_tau...` and historical `350m_tau...` run IDs. Paper wording now distinguishes code `head_dim=64`, `d_rope=32`, and empirical `d_eff=128` tau convention. | Current: `results/eval_3seeds_full_results.json`, plus supporting `results/350m_mla32_results_final.json` and `results/eval_extended_results.json`. | `A` for eval values; `B` for per-run training JSON/checkpoint bundle. |
| 1B/4K MLA reversal and YaRN+FT row | `run_350m_4k_1b.sh` / `run_350m_4k_v2_1b.sh` launch `run_gqa_evq_experiment.py`; `scripts/core_text_phases/yarn_finetune_eval.py` performs baseline, inference-only YaRN, YaRN+FT, and writes `yarn_ft_s*_seed*_results.json`. | Code support exists and now resolves both current and historical run IDs. Launch wrappers remain historical because they encode external server assumptions. | The compact branch retains the scoped paper summary, but neither the historical Phase18/19 reports nor exact 1B/4K baseline/YaRN+FT JSONs are tracked. The visible archival branch also does not close this gap. | `C`: code-backed and paper-summary-backed, but not report/JSON-backed in the current branch. |
| Phase 22-23 MLA tau/window diagnosis | `scripts/core_text_phases/run_gqa_evq_experiment.py`, `mla_tau_optimization.py`, and `mla_tau_optimization_v2.py` support the analysis path. | Mechanism explanation is plausible and code-backed, but it is a supporting diagnostic, not a primary claim. | Current: `results/PHASE22_23_MLA_TAU_SWEEP_REPORT.md`; no compact JSON artifact found for the tables in that report. | `C`; keep as diagnostic unless JSON is recovered. |
| Phase 16 tau operating-rule sweep | `scripts/core_text_phases/phase16_formula_optimality_sweep.py` is a full harness with `result.json`, `status.json`, `events.jsonl`, and report `summary.json` outputs. | Syntax compiles. It is a large harness; use existing reports unless rerunning in the proper environment. | Current docs: `docs/exp/2026-03-09_phase16_formula_optimality_sweep_results.md`; current result folders include only local/smoke reports. Archival branch has `results/phase16/rsweep_results_final.json`. | `B` for code/result support; current compact JSON is incomplete. |
| Phase 21b QuALITY downstream support | `scripts/core_text_phases/phase21b_quality_eval_clean.py`; full-eval report `docs/exp/2026-03-12_phase21b_454m_full_eval_report.md`; curated aggregate `data/curated/quality_454m_full_eval.json`. | The ignored n=2086 aggregate was recovered with SHA256 identity and sanitized into the tracked curated JSON. The older `phase21b_quality_454m_report.json` is an n=200 accuracy-only pilot and must not be used as Table 21 provenance. | Raw-JSON-backed Table 21 values; older pilot retained only to explain the stale Figure 8. | `B` supporting-only; probability-space signal, accuracy inconclusive. |
| 750M continuation support | `scripts/core_text_phases/phase15_750m_2k_to_4k_continue_ckpt_eval.py` plus historical report. | Code exists; row is single-seed supporting evidence only. | Current: `docs/exp/2026-03-06_phase15_750m_2k_to_4k_continue_results.md`; archival branch has `results/phase15/phase15_continue_summary.json`. | `B/C`; do not use as primary scale proof. |

## Code-Level Findings

### Training and JSON outputs exist

- `run_gqa_evq_experiment.py` constructs the MLA `inv_freq` from `d_rope` for
  MLA, logs a short hash, saves `model.pt`, saves `inv_freq.npy`, writes
  per-run `results.json`, and writes work-dir `summary.json`.
- `audit_training_artifacts.py` can inspect recovered run directories and report
  actual cached tokens, train-loop used tokens, dropped tokens from the batch
  floor, and presence of `results.json` / `model.pt` / `inv_freq.npy`.
- `phase11b_125m_dape.py` writes per-run `result.json` and aggregate
  `all_results.json`.
- `phase14c_multiscale_evq_yarn.py` writes per-run `result.json` and
  `passkey_nll.json`, but it is a 50M/125M supporting multiscale check, not the
  full 454M Table 2 reproduction.
- `yarn_finetune_eval.py` writes `yarn_ft_s*_seed*_results.json`.

### Fixed in this audit

- `eval_extended_3seeds.py` and `yarn_finetune_eval.py` no longer rely on
  implicit local-tensor aliasing for YaRN. They require checkpoint `inv_freq`,
  clone the loaded table, print a short hash, and apply YaRN to the loaded table.
- Those same eval scripts now resolve both current training-script run IDs
  (`350m_mla_tau1.414_seed42`) and historical run IDs
  (`350m_tau1.41_seed42`). Before this change, a fresh run from the current
  training script could be missed by the eval script.
- Paper prose no longer calls the MLA `tau=1.414` setting `d_eff=d_head=128`.
  The code field `head_dim` is 64 and `d_rope` is 32; `d_eff=128` is now stated
  only as an empirical operating convention.

### Remaining code/result gaps

- 1B/4K is not "no-code"; it is `code + Markdown report` in the compact branch,
  with exact run JSON absent. The next useful action is to recover the external
  run directory's `results.json`, `summary.json`, and `yarn_ft_s*.json`, or rerun
  the launcher after sanitizing paths.
- Fig. 3(b,c) are not one-command reproducible from compact current JSON because
  the expected Phase 11 JSON files are not packaged in the current branch.
- Old `results.json` files do not record exact token math. Recovered run
  directories should be audited with `audit_training_artifacts.py` before using
  nominal 500M/1B names as proof of exact consumed tokens.
- Some historical launch wrappers remain external/server scripts. Treat them as
  provenance clues unless paired with sanitized JSON manifests.
- Four supporting LLaMA8B JSON files are zero-byte/invalid in the current branch:
  `results/llama8b_longbench/niah/anchored_sigmoid_passkey.json`,
  `results/llama8b_longbench/niah/baseline_passkey.json`,
  `results/llama8b_longbench/passkey_tf/anchored_sigmoid_niah.json`, and
  `results/llama8b_longbench/passkey_tf/baseline_niah.json`. These should not be
  cited as result artifacts.

## Judgment on the 1B Question

The 1B/4K reversal has code support. The training launcher calls the shared MLA
training entrypoint, that entrypoint writes JSON by design, and the YaRN+FT eval
script writes JSON by design. The current compact branch simply does not contain
the exact JSONs behind the historical Phase18 report reference.

So the correct reviewer-safe statement is:

> The 1B/4K row is code-backed and report-backed, but not JSON-backed in the
> compact branch. It should remain a limitation/root-cause target until the exact
> baseline and YaRN+FT JSON files are recovered or regenerated.
