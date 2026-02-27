# LongBench Pipeline Parity Guide

This document tracks parity between local `scripts/eval_longbench.py` and official LongBench behavior.

## Locked parity knobs
Use these settings for parity runs:

- `--prompt_source official`
- `--chat_template auto` (or `on`)
- `--truncate_mode middle`
- `--max_new_tokens_policy official`
- `--score_scale pct` (paper table presentation)
- `--strict_parity_check`

Official config files vendored in repo:
- `scripts/longbench_official_config/dataset2prompt.json`
- `scripts/longbench_official_config/dataset2maxlen.json`

## Audit command

```bash
python scripts/import_2024/longbench_pipeline_audit.py \
  --candidate_json <local_eval_json> \
  --reference_json <official_eval_json_or_baseline_json> \
  --candidate_model_alias hybrid_lora \
  --prefer_pct \
  --tolerance_abs 1.0 \
  --output_json artifacts/reviewer_2026-02-25/longbench_parity_report.json \
  --output_md artifacts/reviewer_2026-02-25/longbench_pipeline_parity.md
```

## Acceptance threshold
- Per-task absolute difference `<= 1.0` (pct scale) on audit subset.
- Ranking consistency must hold.
- If either fails, do not start full 21-task rerun.

## Notes
- Keep raw scores (`score_raw`) for statistics; display `score_pct` in main tables.
- For fairness comparisons, keep decode settings and manifest fixed across methods.
