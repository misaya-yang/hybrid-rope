# Isolated Experiment Pipelines

This folder contains newly created pipelines that are isolated from legacy/running jobs.

## Current Entry Points

- `new_lora_longalpaca_train.py`
  - Locked base model: `Meta-Llama-3-8B-Instruct` (8K-native).
  - Builds mixed instruction dataset (`LongAlpaca/LongQA + WikiText-derived`).
  - Launches low-compute LoRA training (`800` steps, `rank=32`) through
    `scripts/train_cross_model_lora_fast_tuned.py` with `inv_freq.copy()` path.

- `new_eval_longbench.py`
  - Evaluates full `lb21` + per-sample traces.
  - Runs NIAH/Passkey multi-length checks.
  - Writes reproducibility manifest with code/model hashes.

- `attn/`
  - `new_lora_longalpaca_attnbias_train.py`: attention-integrated LoRA training entry.
  - `new_eval_longbench_attnbias.py`: full lb21 + NIAH/Passkey evaluator for attention-integrated runs.
  - `attn_patch_llama_attention_bias.py`: runtime attention-bias monkey patch (default off).
  - `attn_audit_readonly.py`: read-only process/status auditor.
  - `next_attn_lora_queue.sh`: queued launcher for post-sacred-run execution.

## Why isolated

- Running production experiments are treated as read-only.
- New experiments must not mutate active scripts.
- Output roots are segregated under `artifacts/new_dataset_v1/`.
