# LongInst Core Review And Handoff (2026-02-27)

## Scope
- Core trainer: `scripts/isolated/longinst/new_lora_longinst_train_v1.py`
- Core orchestrator: `scripts/isolated/longinst/run_llama8k_theory_v1.py`
- Core paired stats: `scripts/isolated/longinst/paired_stats_llama8k_theory_v1.py`

Goal: remove high-risk logic that can silently waste GPU budget, then lock a safe run path for next-day execution.

## High-Risk Findings And Fixes

1. `run_llama8k_theory_v1.py` Stage-A plan text and real tau grid were inconsistent.
- Fix: docstring now matches the real job table (`tau=0.4/0.6/0.8` for A2/A3/A4).

2. `run_llama8k_theory_v1.py` EVQ best-pick score had a hidden tau bias term.
- Old behavior: mean delta plus a manual penalty toward `tau≈2.0`.
- Risk: could pick a non-best EVQ candidate and waste full-lb21 runs.
- Fix: best-pick now uses pure gate mean delta with deterministic tie-breakers only.

3. `run_llama8k_theory_v1.py` registry fields could diverge from actually applied runtime schedule.
- Fix: registry now stores applied `rope_schedule/evq_tau` used in the emitted train command.

4. `new_lora_longinst_train_v1.py` default schedule could accidentally fall back to anchored sigmoid.
- Fix: default `--rope_schedule` switched to `evq_cosh`.
- Added explicit warning when `anchored_sigmoid` is selected.

5. `paired_stats_llama8k_theory_v1.py` naming was anchored-specific, causing handoff confusion after EVQ switch.
- Fix: generalized wording to Method-A vs Geometric.
- Added `--method_a_jsons` while keeping legacy `--anchored_jsons` compatible.

## Data-Safety Status (Already Enforced)
- Response-only masking no longer forces a tail reserve when assistant span is missing.
- Samples with assistant out-of-window are dropped, not force-supervised.
- `pad_token_id` is explicitly set to `eos_token_id` when missing.
- Wiki sampling cap can auto-expand beyond requested limit when token budget demands it.

## Pre-Run Checklist (Tomorrow)
1. Confirm venv and imports:
```bash
cd /Users/yang/projects/hybrid-rope
.venv/bin/python - <<'PY'
import transformers,datasets,peft,trl,accelerate,torch
print("env_ok")
PY
```

2. Confirm mixed dataset manifest is valid:
```bash
cd /Users/yang/projects/hybrid-rope
ls -lah <MIXED_DATASET_DIR>/mix_manifest.json <MIXED_DATASET_DIR>/train.jsonl
```

3. Dry-run one gate job before full queue:
```bash
cd /Users/yang/projects/hybrid-rope
.venv/bin/python scripts/isolated/longinst/run_llama8k_theory_v1.py \
  --base_model_path <BASE_MODEL_PATH> \
  --longalpaca_path <LONGALPACA_JSONL> \
  --longqa_path <LONGQA_JSONL> \
  --wikitext_train_path <WIKITEXT_TXT> \
  --mixed_dataset_dir <MIXED_DATASET_DIR> \
  --mixed_dataset_split train \
  --longbench_local_data_dir <LONGBENCH_DATA_DIR> \
  --qwen_seed42_json <QWEN_S42_JSON> \
  --qwen_seed1337_json <QWEN_S1337_JSON> \
  --morning_reference_json <MORNING_REF_JSON> \
  --no-execute
```

4. Launch execution only after command audit:
```bash
cd /Users/yang/projects/hybrid-rope
.venv/bin/python scripts/isolated/longinst/run_llama8k_theory_v1.py <same args> --execute
```

## Expected Outputs
- Registry CSV: `docs/exp/llama8k_theory_v1_registry.csv`
- Report MD: `docs/exp/llama8k_theory_v1_report.md`
- Run manifest: `artifacts/llama8k_theory_v1/stats/run_manifest.json`
- Per-run logs: `artifacts/llama8k_theory_v1/logs/*.log`

## Notes
- `.venv/` and local dry-run artifacts are environment/runtime assets and are not part of git deliverables.
- If Stage-B gate fails (`B1/B2`), orchestrator will stop `B3/B4` automatically by design.
