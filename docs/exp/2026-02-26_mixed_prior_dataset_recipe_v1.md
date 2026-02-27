# Mixed Prior Dataset v1 (Token-Budget Locked)

## Goal
Construct a LoRA finetune corpus that faithfully instantiates the paper's mixed distance prior `D(Delta)`:

- `power_law_base` (long-range smooth semantic prior)
- `bimodal_reasoning` (mid-band/local multihop prior)
- `uniform_scaffold` (uniform control scaffold)

The ratio is enforced by **token budget**, not row count.

## Locked Recipe
- Token ratio target (default):
  - `power_law_base = 0.50`
  - `bimodal_reasoning = 0.40`
  - `uniform_scaffold = 0.10`
- Default target total tokens: `200,000,000`
- Response-only filter:
  - prompt tokens masked to `-100`
  - assistant tokens supervised
  - `min_supervised_tokens >= 64`
  - no supervision fallback when assistant span is out-of-window (sample is dropped)
  - truncation policy aligns with trainer: `head-tail keep + drop middle` (`head_cap=500`)
  - assistant boundary is resolved by `offset_mapping` (strict mode)

## Script
- Builder: `/Users/yang/projects/hybrid-rope/scripts/prepare_mixed_prior_dataset_v1.py`

Main outputs:
- `mixed_prior_finetune.jsonl`
- `train.jsonl`, `valid.jsonl`, `test.jsonl`
- `mix_manifest.json`
- `quality_report.md`
- `label_mask_preview.json`

## Required Pre-flight Checks
The builder always prints token-share check:

- `[DATA CHECK] token_ratio power_law_base=... bimodal_reasoning=... uniform_scaffold=...`

It also writes `label_mask_preview.json` with first 3 samples:
- `first_64_input_ids`
- `first_64_labels`
- `last_64_input_ids`
- `last_64_labels`
- `check_has_masked_prefix`
- `check_has_supervised_suffix`

## Example Commands
Dry run (fast gate):

```bash
python scripts/prepare_mixed_prior_dataset_v1.py \
  --tokenizer_path /root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct \
  --wikitext_path /root/autodl-tmp/wikitext_data/train.txt \
  --bimodal_jsonl_paths /root/autodl-tmp/dfrope/datasets/LongAlpaca_msdownload/LongAlpaca-12k.min64.jsonl,/root/autodl-tmp/dfrope/datasets/LongQA.jsonl \
  --target_total_tokens 200000000 \
  --dry_run --dry_run_tokens 2500000
```

Full build:

```bash
python scripts/prepare_mixed_prior_dataset_v1.py \
  --tokenizer_path /root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct \
  --wikitext_path /root/autodl-tmp/wikitext_data/train.txt \
  --bimodal_jsonl_paths /root/autodl-tmp/dfrope/datasets/LongAlpaca_msdownload/LongAlpaca-12k.min64.jsonl,/root/autodl-tmp/dfrope/datasets/LongQA.jsonl \
  --scaffold_jsonl_paths /root/autodl-tmp/dfrope/datasets/sharegpt_clean.jsonl \
  --target_total_tokens 250000000 \
  --powerlaw_ratio 0.50 --bimodal_ratio 0.40 --scaffold_ratio 0.10 \
  --max_seq_len 16384 --min_supervised_tokens 64
```

## Train Entry (No Dynamic Mixing)
`new_lora_longinst_train_v1.py` now supports direct loading of prebuilt mixed dataset:

```bash
python scripts/isolated/longinst/new_lora_longinst_train_v1.py \
  --base_model_path /root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct \
  --mixed_dataset_dir /root/autodl-tmp/dfrope/hybrid-rope/artifacts/datasets/mixed_prior_v1_YYYYMMDD_HHMMSS \
  --mixed_dataset_split train \
  --max_steps 800
```

Or through the orchestrator:

```bash
python scripts/isolated/longinst/run_llama8k_theory_v1.py \
  ... \
  --mixed_dataset_dir /root/autodl-tmp/dfrope/hybrid-rope/artifacts/datasets/mixed_prior_v1_YYYYMMDD_HHMMSS \
  --mixed_dataset_split train
```

## Audit Field Contract
`mix_manifest.json` includes required source-level audit fields:
- `source_name`
- `lang`
- `task_type`
- `count`
- `token_ratio`
- `sha256`
- `filter_rule`
