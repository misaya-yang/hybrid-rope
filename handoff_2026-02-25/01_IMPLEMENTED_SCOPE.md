# 01 Implemented Scope (This Round)

Date: 2026-02-25  
Workspace: `/Users/misaya.yanghejazfs.com.au/neurIPS-2026/hybrid-rope`

## A) LongBench pipeline reconstruction

Implemented:

- `scripts/eval_longbench.py`
  - official parity mode switches
  - 21-task support (`lb21`)
  - chat template path
  - middle truncation mode
  - official max_new_tokens policy
- vendored official config:
  - `scripts/longbench_official_config/dataset2prompt.json`
  - `scripts/longbench_official_config/dataset2maxlen.json`

## B) Runner protocol lock propagation

Implemented:

- `scripts/run_eval.py`
  - parity/task-set passthrough
  - writes lock metadata (`baseline_protocol_lock.json`) with protocol knobs
- `scripts/run_sota_downstream_eval.py`
  - supports `lb21` and parity-mode parameters in downstream orchestration

## C) Statistical rigor upgrade

Implemented:

- `scripts/import_2024/significance_test.py`
  - FDR correction support (BH/BY)
  - outputs `p_raw`, `p_fdr_bh`, `p_fdr_by`, `claim_grade`
  - claim policy report generation for automatic wording downgrade on non-significance

## D) New reviewer-facing theory scripts

Implemented:

- `scripts/import_2024/longbench_pipeline_audit.py`
- `scripts/import_2024/functional_residual_real_prior.py`
- `scripts/import_2024/theorem3_adversarial_bimodal.py`
- `scripts/import_2024/export_schedule_from_prior.py`

## E) Documentation already landed

Updated/added:

- `AI_HANDOFF.md`
- `handoff_2026-02-23/plan.md`
- `handoff_2026-02-23/anchored_sigmoid_math_note.md`
- `handoff_2026-02-23/longbench_pipeline_parity.md`
- `handoff_2026-02-23/theory_appendix_extensions.md`

## F) Artifacts already available

- `artifacts/reviewer_2026-02-25/longbench_parity_report.json`
- `artifacts/reviewer_2026-02-25/longbench_pipeline_parity.md`

## G) Known compatibility risk

- Historical LongBench outputs that do not contain per-sample raw fields cannot be used for upgraded per-sample significance.
- Required field example: `per_sample_scores_raw`.
- Consequence: legacy files can still be used for descriptive comparison, but not for strict paired significance.
