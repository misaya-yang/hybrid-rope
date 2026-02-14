# Workspace Cleanup And Data Sync (2026-02-15)

## Scope

- Goal: pull back remote experiment data only (no model weights), then reorganize code and result assets for advisor-facing review.
- Workspace: `e:/rope/hybrid-rope`
- Remote root: `/root/autodl-tmp/dfrope/hybrid-rope`

## What Was Synced

- Synced additional result directories from remote via a `tar` package with strict excludes:
  - `*.safetensors`
  - `*.pt`
  - `*.bin`
  - `*.pth`
  - `optimizer*`
  - `pytorch_model*`
- Retrieved directories include:
  - `results/passkey_long`
  - `results/qwen_3way_compare`
  - `results/qwen_comparison`
  - `results/qwen_int4_vs_base_only`
  - `results/rope_scaling_v2`
  - `results/sigmoid_debug`
  - `results/sigmoid_v3`
  - `results/smoke_train_local`
  - `results/smoke_train_local_v2`
  - `results/train_700m_local`
  - `results/train_700m_wikitext`
  - `results/train_freq_comparison`

## Result Folder Reorganization

### 1) High-priority advisor bundle

- Created: `results/paper_ready/`
- Contents: copied lightweight evidence files only (`.json/.md/.txt/.csv/.pdf/.png/.log`) from key experiments.
- Use this folder first when preparing slides/report.

### 2) Low-priority archive bucket

- Created: `results/archive_low_priority/`
- Moved low-priority/diagnostic directories:
  - `llama8b_fair_lora_probe_bs2`
  - `llama8b_fair_lora_probe_bs4`
  - `llama8b_fair_lora_probe_bs4_mem`
  - `llama8b_fair_lora_probe_bs5`
  - `llama8b_fair_lora_probe_bs6`
  - `llama8b_fair_lora_probe_bs8`
  - `llama8b_fair_lora_smoke`
  - `sigmoid_debug`
  - `sigmoid_v3`
  - `smoke_train_local`
  - `smoke_train_local_v2`
  - `train_700m_local`
  - `train_700m_wikitext`
  - `model_downloads`

### 3) Local weight quarantine

- Created: `results/_weights_quarantine/`
- Moved all existing weight-like files from `results/` into quarantine:
  - `*.safetensors`, `*.pt`, `*.bin`, `*.pth`
- Purpose: keep experiment data and model artifacts physically separated.

## Code Reorganization

- Created: `tools/remote_legacy/`
- Moved root-level historical remote helpers into this directory:
  - all `upload_*.py`
  - all `download_*.py`
  - old one-off root `run_*.py` remote launchers
  - `seetacloud_plink.py`
  - `remote_start_night_run_9h_extended.py`
  - `remote_tail_night_run_9h_extended.py`
- Result: project root now focuses on core directories (`scripts/`, `docs/`, `results/`, `artifacts/`).

## Recommended Working Convention (from now on)

- New training/eval code goes to `scripts/`.
- New remote upload/download helper goes to `tools/` (not project root).
- Advisor-facing outputs should be copied to `results/paper_ready/`.
- Temporary and smoke/probe runs should be moved to `results/archive_low_priority/`.

## Advisor Delivery Bundle

- Created: `results/advisor_package_2026-02-15/`
- Contains numbered sections (`01` to `07`) with only core evidence, key scripts, and live status logs.
- Entry file: `results/advisor_package_2026-02-15/INDEX.md`
