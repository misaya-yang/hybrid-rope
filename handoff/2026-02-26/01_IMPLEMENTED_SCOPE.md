# 01 Implemented Scope (2026-02-26)

## A) LoRA artifact contract hardening

- `scripts/train_cross_model_lora_fast_tuned.py`
  - exports canonical `<run_dir>/final_lora/`
  - keeps root adapter files for backward compatibility
  - writes `artifacts_contract.adapter_layout` and `adapter_resolved_path`

## B) Registry compatibility upgrade

- `scripts/build_model_registry.py`
  - supports ready-detection on `root_adapter|final_lora`
  - adds columns: `adapter_layout`, `adapter_resolved_path`, `root_adapter_*`, `final_lora_*`

## C) Plan B evaluator control surface

- `scripts/plan_b_eval_longbench.py`
  - supports `--task_set`, `--tasks`, `--max_samples_per_task`
  - supports stress controls:
    - `--niah_needles_per_prompt`
    - `--niah_trials_per_cell`
    - `--passkey_trials_per_cell`
  - forwards `manifest_json` to NIAH/Passkey
  - supports adapter directories under both root and `final_lora`

## D) Mixed long-instruction data builder (new)

- `scripts/prepare_long_instruction_mix.py` (new)
  - multi-source JSONL ingestion (messages/conversations/instruction schemas)
  - language/task/length-bucket ratio controls
  - outputs `train.txt/valid.txt/test.txt` and `mix_manifest.json`
  - records source audit fields (`source_name/lang/task_type/count/token_ratio/sha256/filter_rule`)

## E) Unified result schema lock

- `scripts/eval_longbench.py`
- `scripts/eval_niah_recall.py`
- `scripts/eval_passkey_teacher_forcing.py`

All now include required fields:
- `protocol_lock`
- `manifest_json`
- `per_sample_scores_raw`
- `inv_sha256`
