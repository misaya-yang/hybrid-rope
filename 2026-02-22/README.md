# 2026-02-22 Package

This folder is an isolated package for Gemini review, so it does not disturb legacy scripts.

## Files

- `scripts/run_llama8b_fair_suite.py`
  - New fair training script for Llama-3-8B-Instruct LoRA.
  - No forward monkey patch.
  - No `config.rope_scaling` mutation.
  - RoPE update by in-place `inv_freq.copy_()` only.

- `EXPERIMENT_CALIBRATION_PLAN.md`
  - Multi-gate calibration and follow-up experiment schedule.

## Quick Run

```bash
cd /root/autodl-tmp/dfrope/hybrid-rope
/root/miniconda3/bin/python 2026-02-22/scripts/run_llama8b_fair_suite.py \
  --method anchored_hybrid \
  --run_name fair_anchor_test_01
```

Calibration only mode:

```bash
/root/miniconda3/bin/python 2026-02-22/scripts/run_llama8b_fair_suite.py \
  --method baseline \
  --run_name fair_baseline_calib \
  --calibration_only
```
