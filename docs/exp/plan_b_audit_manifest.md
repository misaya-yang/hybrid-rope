# Plan B Audit Manifest (2026-02-25)

This document is the fixed checklist for the strict audit + recovery path.
It is designed to be machine-readable and human-auditable.

## 1) Running Script Protection

- Sacred run policy:
  - The currently running script/process must be read-only.
  - No kill, no overwrite, no in-place edits.
  - Recovery work must use isolated new files only.
- Protected process snapshot:
  - `scripts/train_cross_model_lora_fast_tuned.py`
  - Active args (observed): `--method anchored_sigmoid --base_model_path .../Qwen2___5-7B-Instruct --data_dir /root/autodl-tmp/wikitext_data --max_steps 400`.

## 2) Audit Report v2 Gate

For each audit cycle, fill the following fields:

- Running script(s):
- Flaw 1 Model Integrity (Meta-Llama-3-8B-Instruct 8K only): `PASS/WARNING/CRITICAL`
- Flaw 2 Version Trap (no Llama-3.1, no 128K-native): `PASS/WARNING/CRITICAL`
- Flaw 3 Dataset Quality (long instruction/retrieval tuning data): `PASS/WARNING/CRITICAL`
- Flaw 4 Metric & Capability Health (LongBench absolute level + instruction sanity): `PASS/WARNING/CRITICAL`
- Flaw 5 Reproducibility & Traces (full-task + per-sample + hashes): `PASS/WARNING/CRITICAL`
- Overall Verdict: `FULL PASS / MINOR WARNING / CRITICAL FLAW`

## 3) Plan B Files (Isolated)

- Training launcher: `scripts/plan_b_train_anchored_v2.py`
- Evaluation launcher: `scripts/plan_b_eval_longbench.py`
- This manifest: `docs/exp/plan_b_audit_manifest.md`

## 4) Reproducibility Contract

- Base model lock:
  - Must contain `Meta-Llama-3-8B-Instruct`
  - Must have `max_position_embeddings = 8192`
- Dataset lock:
  - Must be long-context instruction style data (`jsonl`)
  - Must not be plain WikiText pipeline
- Output lock:
  - `artifacts/plan_b_runs/plan_b_manifest.json`
  - `artifacts/plan_b_eval/plan_b_eval_manifest.json`
  - LongBench full `lb21` json with per-sample traces
  - NIAH + Passkey outputs

## 5) Suggested Execution Order

1. Audit current run and archive evidence in this manifest.
2. Prepare long-instruction jsonl data.
3. Launch Plan B training (baseline + anchored, seeds 42/1337).
4. Launch Plan B evaluation (LongBench lb21 full + NIAH + Passkey).
5. Run significance and FDR policy report.

