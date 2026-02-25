# Fast Tuned Cross-Model Runbook (2026-02-25)

## Goal
Run future cross-model LoRA experiments with:
- tuned anchored schedule (`anchor=4, slope_raw=20, center_ratio=0.70`)
- higher throughput on RTX PRO 6000 (96GB)
- unchanged legacy scripts (new pipeline only)

## New scripts (legacy untouched)
- launcher: `scripts/cross_model_finetune_fast_tuned.sh`
- trainer: `scripts/train_cross_model_lora_fast_tuned.py`

The launcher points to the new trainer by default and writes outputs to:
- `artifacts/cross_model_fast_tuned/`

## Default profile (cost-performance balanced)
- `MAX_STEPS=400`
- `PER_DEVICE_BATCH=2`
- `GRAD_ACCUM=1`
- `GRAD_CHECKPOINTING=0`
- `MAX_SEQ_LEN=16384`
- `LOAD_IN_4BIT=1`
- `ATTN_IMPLEMENTATION=auto`

Anchored tuned parameters:
- `ANCHOR_FACTOR_DEFAULT=4`
- `ANCHORED_SLOPE_RAW=20`
- `ANCHORED_CENTER_RATIO=0.70`

## Why 400 steps
Based on completed baseline log (`llama_3_8b_instruct_baseline_1337.log`) on this server:
- 10-200 steps: major drop
- 210-400 steps: smaller but still useful drop
- 410-600 steps: very small marginal gain

Operational choice:
- keep 400 as default for better cost-efficiency
- only use 600 when a specific final run needs maximum convergence evidence

## Launch commands

```bash
cd /root/autodl-tmp/dfrope/hybrid-rope
bash scripts/cross_model_finetune_fast_tuned.sh
```

Optional smoke before full run:

```bash
cd /root/autodl-tmp/dfrope/hybrid-rope
PER_DEVICE_BATCH=4 MAX_STEPS=50 bash scripts/cross_model_finetune_fast_tuned.sh
```

If OOM appears in smoke:
- fallback order:
  1. `PER_DEVICE_BATCH=3`
  2. `PER_DEVICE_BATCH=2` and keep `GRAD_ACCUM=1`
  3. enable checkpointing only if required: `GRAD_CHECKPOINTING=1`

## Must-log fields (for paper traceability)
Each run summary must include:
- `inv_sha256`
- `rope.schedule_params`:
  - `anchor_factor_requested`
  - `anchor_factor_effective`
  - `slope_raw`
  - `center_ratio`
- training hyperparams:
  - `max_steps`
  - `per_device_train_batch_size`
  - `gradient_accumulation_steps`
  - `gradient_checkpointing`

## Note on flash-attn
Current server environment reported no `flash_attn` package.
With `ATTN_IMPLEMENTATION=auto`, runtime falls back to supported backends (currently `sdpa` observed).
