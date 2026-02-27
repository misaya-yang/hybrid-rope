# Fast Tuned Cross-Model Runbook (2026-02-25)

## Goal
Run future cross-model LoRA experiments with:

- tuned anchored schedule (`anchor=4, slope_raw=20, center_ratio=0.70`)
- stable high GPU utilization on RTX PRO 6000 (96GB)
- strict artifact traceability for downstream eval and paper evidence

Detailed handoff contract is maintained at:

- `handoff/2026-02-23/05_GPU最大化与LoRA资产交接规范_2026-02-25.md`

## New scripts (legacy untouched)

- launcher: `scripts/cross_model_finetune_fast_tuned.sh`
- trainer: `scripts/train_cross_model_lora_fast_tuned.py`
- registry builder: `scripts/build_model_registry.py`

The launcher writes outputs to:

- `artifacts/cross_model_fast_tuned/` (default)

## Default profile (server-validated)

- `MAX_STEPS=400`
- `PER_DEVICE_BATCH=1`
- `GRAD_ACCUM=1`
- `GRAD_CHECKPOINTING=1`
- `MAX_SEQ_LEN=16384`
- `LOAD_IN_4BIT=1`
- `ATTN_IMPLEMENTATION=auto`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (if unset, launcher auto-sets)

Anchored tuned parameters:

- `ANCHOR_FACTOR_DEFAULT=4`
- `ANCHORED_SLOPE_RAW=20`
- `ANCHORED_CENTER_RATIO=0.70`

## Why 400 steps

Based on completed baseline logs on the same server profile:

- 10-200 steps: large loss drop
- 210-400 steps: smaller but still meaningful drop
- 410-600 steps: small marginal gain versus cost

Operational choice:

- keep `400` as default for iterative experiments
- only run `600` when a final evidence run explicitly requires it

## Launch commands

```bash
cd /root/autodl-tmp/dfrope/hybrid-rope
bash scripts/cross_model_finetune_fast_tuned.sh
```

Profile override examples:

```bash
# Try a faster profile only after smoke passes
cd /root/autodl-tmp/dfrope/hybrid-rope
PER_DEVICE_BATCH=2 GRAD_ACCUM=1 GRAD_CHECKPOINTING=1 bash scripts/cross_model_finetune_fast_tuned.sh
```

OOM fallback order:

1. `PER_DEVICE_BATCH=1`
2. keep `GRAD_ACCUM=1`
3. keep `GRAD_CHECKPOINTING=1`

## Must-log fields (paper traceability)

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

## Registry refresh (mandatory after each batch)

```bash
cd /root/autodl-tmp/dfrope/hybrid-rope
python scripts/build_model_registry.py
```

Then inspect:

- `artifacts/model_registry/lora_weights.tsv`
- `artifacts/model_registry/latest_by_method.tsv`

## Note on flash-attn

Current server may not have `flash_attn` installed.
With `ATTN_IMPLEMENTATION=auto`, runtime falls back to supported backends (typically `sdpa`).
