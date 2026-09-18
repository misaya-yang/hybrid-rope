# Kanana official-runtime comparison

Status: **complete**. The canonical score owner is [RESULT.md](RESULT.md); this
README only maps code and compact artifacts and does not maintain a second set
of results.

## Compact reports

| Endpoint | Report |
|---|---|
| 64K two-task three-arm pilot | [reports/pilot2_three_arm.json](reports/pilot2_three_arm.json) |
| 64K TailSpline/YaRN Full-13x10 | [reports/full13x10.json](reports/full13x10.json) |
| 64K TailSpline/MrPro/YaRN Full-13x10 | [reports/full13x10_three_arm.json](reports/full13x10_three_arm.json) |
| 128K complete-context InfiniteBench English QA | [reports/qa128k_two_arm.json](reports/qa128k_two_arm.json) |

Large generation rows remain under the experiment server root
`/root/autodl-tmp/today_rope_plan_20260914/kanana_yarn_tailspline_64k_20260918`
and are identified by hashes in the result owner. They are not copied into Git.

## Reusable code

- `prepare.py`, `prepare_parallel_server.sh`, `run_server.sh`: frozen 64K input,
  table and two-arm execution path.
- `prepare_mrpro.py`, `run_mrpro_full13.sh`, `report_three_arm_full.py`: canonical
  MrRoPE completion and three-arm recomputation at 64K.
- `prepare_qa128k.sh`, `prepare_qa128k_tables.py`, `run_qa128k_three_arm.sh`:
  untruncated 128K English-QA assets and execution. The completed run used only
  TailSpline S=4 and official runtime YaRN factor 4.4; MrRoPE was skipped before
  start.
- `report_qa_two_arm.py`, `finish_qa128k_two_arm.sh`: strict paired completion
  and compact report generation.

The other `run_qa64k_*` and queue wrappers are retained as execution history or
recovery tools. Their presence does not define a pending experiment.
