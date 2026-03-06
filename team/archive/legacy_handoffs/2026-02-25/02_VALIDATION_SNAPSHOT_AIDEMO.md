# 02 Validation Snapshot (aidemo)

Date: 2026-02-25  
Environment: `aidemo`  
Conda path: `~/miniconda3/bin/conda`

## 1) Python compile check

Command:

```bash
~/miniconda3/bin/conda run -n aidemo python -m py_compile \
  scripts/eval_longbench.py \
  scripts/run_eval.py \
  scripts/run_sota_downstream_eval.py \
  scripts/import_2024/significance_test.py \
  scripts/import_2024/longbench_pipeline_audit.py \
  scripts/import_2024/functional_residual_real_prior.py \
  scripts/import_2024/theorem3_adversarial_bimodal.py \
  scripts/import_2024/export_schedule_from_prior.py
```

Result: pass (exit code 0).

## 2) Script smoke checks

### 2.1 CLI availability

Command pattern:

```bash
~/miniconda3/bin/conda run -n aidemo python <script> --help
```

Scripts checked:

- `longbench_pipeline_audit.py`
- `functional_residual_real_prior.py`
- `theorem3_adversarial_bimodal.py`
- `export_schedule_from_prior.py`

Result: pass (all help outputs returned normally).

### 2.2 Minimal functional run with synthetic inputs

Synthetic files used:

- `/tmp/hybrid_rope_smoke/candidate.json`
- `/tmp/hybrid_rope_smoke/reference.json`
- `/tmp/hybrid_rope_smoke/prior.json`

Executed:

- parity audit
- real-prior functional residual (reduced grid)
- Theorem 3 adversarial bimodal scan (reduced grid)
- schedule export from prior

Result: pass.

Generated smoke outputs:

- `/tmp/hybrid_rope_smoke/parity_report.json`
- `/tmp/hybrid_rope_smoke/parity_report.md`
- `/tmp/hybrid_rope_smoke/functional/functional_residual_real_prior.json`
- `/tmp/hybrid_rope_smoke/theorem3/theorem3_fragility_map.json`
- `/tmp/hybrid_rope_smoke/schedule_from_prior.{json,npy,pt}`

Persisted copies (for repo traceability):

- `artifacts/reviewer_2026-02-25/smoke/parity_report.json`
- `artifacts/reviewer_2026-02-25/smoke/parity_report.md`
- `artifacts/reviewer_2026-02-25/smoke/functional_residual_real_prior.json`
- `artifacts/reviewer_2026-02-25/smoke/theorem3_fragility_map.json`
- `artifacts/reviewer_2026-02-25/smoke/schedule_from_prior.json`

## 3) Existing reviewer artifact check

Confirmed present:

- `artifacts/reviewer_2026-02-25/longbench_parity_report.json`
- `artifacts/reviewer_2026-02-25/longbench_pipeline_parity.md`

## 4) Interpretation

- Core remediation scripts are runnable in current local environment.
- The pipeline is ready for full-data execution once server-side run gates are met.
- Statistical strict mode still depends on fresh per-sample outputs from the new evaluator.
