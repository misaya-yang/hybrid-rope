# Batch Report Bundle (2026-02-23)

This folder contains the recovered data and report artifacts for:

- `results/llama8b_fair_v2_longbench_stable_20260223_0150`

## Structure

- `data/llama8b_batch_20260223_dataonly.tgz`: remote data-only archive (no `final_lora` weights).
- `data/raw/`: extracted archive payload.
- `logs/remote_status_snapshot_clean.txt`: remote runtime snapshot at bundle time.
- `report/BATCH_REPORT_CN.md`: consolidated Chinese report.
- `report/*.csv`: completion matrix, method metrics, and task-level scores.
- `report/build_report.py`: report builder script from local extracted files.

## Rebuild Report

```bash
python3 batch_report_2026-02-23_downstream_eval/report/build_report.py
```

## Notes

- At bundle time, `anchored_sigmoid` was complete in both downstream profiles.
- `downstream_eval_autorun` had an autopilot backfill run for `yarn`, then was intentionally stopped at `2026-02-23 23:49 CST` by cost policy (`>30 min` remaining, low incremental value).
- Decision log: `logs/remote_shutdown_decision_2026-02-23_2349_CST.txt`.
