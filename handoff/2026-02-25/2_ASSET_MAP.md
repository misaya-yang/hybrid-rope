# Asset Map (Base + LoRA + inv_freq Triad)

Last updated: 2026-02-25

The only correct way to load an experiment condition is the triad:
- Base model path
- LoRA adapter path (optional)
- `custom_inv_freq` tensor path (optional, but required for custom schedules)

## Server Paths (52592)

Repo root:
- `/root/autodl-tmp/dfrope/hybrid-rope`

Model cache:
- `/root/autodl-tmp/dfrope/ms_models`

Qwen base model (recommended mainline):
- `/root/autodl-tmp/dfrope/ms_models/Qwen/Qwen2___5-7B-Instruct`

Alternative Qwen base (older):
- `/root/autodl-tmp/dfrope/ms_models/Qwen/Qwen2-7B-Instruct`

LoRA run roots (fast tuned runs):
- `/root/autodl-tmp/dfrope/hybrid-rope/artifacts/cross_model_fast_tuned_b1_gc/`

Each run dir must contain:
```text
<run_dir>/
  final_lora/
  artifacts/
    summary.json
    custom_inv_freq.pt
```

## How to locate assets reliably (no guessing)

Generate the registry:
```bash
cd /root/autodl-tmp/dfrope/hybrid-rope
/root/miniconda3/bin/python scripts/build_model_registry.py \
  --repo_root . \
  --model_cache_dir /root/autodl-tmp/dfrope/ms_models \
  --run_roots artifacts/cross_model_fast_tuned_b1_gc
```

Then read:
- `artifacts/model_registry/lora_weights.tsv` (complete triads)
- `artifacts/model_registry/latest_by_method.tsv` (latest usable per model/method)

## Local machine (macOS)

This repo may not store large base model weights locally.
Use local runs mainly for:
- theory scripts (`scripts/import_2024/*.py`)
- plotting and table aggregation

