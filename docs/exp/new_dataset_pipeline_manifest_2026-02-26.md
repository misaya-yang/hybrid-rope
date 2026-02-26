# New Dataset Experiment Manifest (Isolated Pipeline)

Last updated: 2026-02-26

## Objective

Test Anchored-Sigmoid under a better long-context instruction mixture while freezing all other protocol dimensions.

## Hard Locks

- Base model: `Meta-Llama-3-8B-Instruct` (8K-native, `max_position_embeddings=8192`)
- Forbidden: `Llama-3.1`, any 128K-native model
- RoPE path: `inv_freq.copy()` injection only (no HF `rope_scaling` config)
- Theta: `500000`
- LoRA rank: `32`
- Steps: `800`
- Keep learning rate / optimizer / batch policy aligned with current fast protocol:
  - `learning_rate=2e-5`
  - `per_device_train_batch_size=2`
  - `gradient_accumulation_steps=1`
  - `optimizer=paged_adamw_8bit`
  - `warmup_steps=50`
  - `lr_scheduler_type=cosine`

## Dataset Policy

- Primary: LongAlpaca (or LongQA high-quality subset)
- Stability fallback: `70% long-instruction + 30% WikiText-derived continuation instructions`
- Data preparation is isolated inside:
  - `scripts/isolated/new_lora_longalpaca_train.py`
  - outputs to `artifacts/new_dataset_v1/prepared_data/`

## Pipeline Entrypoints

### 1) Training

```bash
~/miniconda3/bin/conda run -n aidemo python scripts/isolated/new_lora_longalpaca_train.py \
  --longalpaca_path /root/autodl-tmp/dfrope/datasets/LongAlpaca-12k.json \
  --longqa_path /root/autodl-tmp/dfrope/datasets/LongQA.jsonl \
  --max_steps 800 \
  --lora_rank 32 \
  --methods baseline,anchored_sigmoid \
  --seeds 42,1337
```

### 2) Evaluation

```bash
~/miniconda3/bin/conda run -n aidemo python scripts/isolated/new_eval_longbench.py \
  --train_output_root artifacts/new_dataset_v1/runs \
  --eval_output_root artifacts/new_dataset_v1/eval \
  --methods baseline,anchored_sigmoid \
  --seeds 42,1337 \
  --ctx 16384 \
  --batch_size 2
```

## Required Outputs

- Training manifest: `artifacts/new_dataset_v1/train_manifest.json`
- Eval manifest: `artifacts/new_dataset_v1/eval_manifest.json`
- Full LongBench: `artifacts/new_dataset_v1/eval/*/longbench_lb21.json`
- NIAH: `artifacts/new_dataset_v1/eval/*/niah/`
- Passkey: `artifacts/new_dataset_v1/eval/*/passkey/`
- Repro manifests: `artifacts/new_dataset_v1/eval/*/repro_manifest/`

## Audit Note

Current Qwen run is sacred and untouched. This new pipeline is isolated by file names and output roots.
