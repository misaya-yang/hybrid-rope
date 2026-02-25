# AUDIT REPORT v2 (2026-02-25 21:20 CST)

## Running script(s)

- `PID 73144`: `scripts/train_cross_model_lora_fast_tuned.py --method anchored_sigmoid --base_model_path /root/autodl-tmp/dfrope/ms_models/Qwen/Qwen2___5-7B-Instruct --data_dir /root/autodl-tmp/wikitext_data --max_steps 400 ...`
- GPU snapshot during audit: RTX PRO 6000, `90437MiB / 97887MiB`, `100%` util.

## Flaw 1 Model Integrity

- **CRITICAL**
- Required red line: `Meta-Llama-3-8B-Instruct` (8K-native).
- Observed running model: `Qwen2.5-7B-Instruct`.
- Config evidence:
  - Qwen config: `model_type=qwen2`, `max_position_embeddings=32768`
  - Llama-3 config on server: `model_type=llama`, `max_position_embeddings=8192`
- Verdict: current run cannot support the strict 8K-native extension narrative.

## Flaw 2 Version Trap

- **WARNING**
- Positive: no evidence of `Llama-3.1` in current running command.
- Risk: model family drift still exists (Qwen path instead of locked Meta-Llama-3-8B-Instruct).

## Flaw 3 Dataset Quality

- **CRITICAL**
- Observed training data path: `/root/autodl-tmp/wikitext_data` (`train.txt` ~11MB).
- For strict claim, expected long-context instruction/retrieval style data.
- Verdict: dataset does not satisfy the required training-data quality gate.

## Flaw 4 Metrics & Capability Health

- **WARNING**
- Current run is mid-training; no checkpoint/summary emitted yet for this run.
- Cannot conclude LongBench baseline health from this active run alone.
- Existing repo docs indicate prior baseline recovery signals, but this active run has no finished metric artifact yet.

## Flaw 5 Reproducibility & Traces

- **WARNING**
- Training script has summary outputs and inv-frequency hash path, but this active run has not yet emitted final summary/checkpoints.
- Full LongBench per-sample traces are available in eval pipeline code, not from this in-progress training process.

## Overall Verdict

- **CRITICAL FLAW**
- Triggered by Flaw 1 and Flaw 3 under the strict red-line prompt.

## Action Taken

- No modification or interruption to the running process.
- Created isolated recovery pipeline files:
  - `scripts/plan_b_train_anchored_v2.py`
  - `scripts/plan_b_eval_longbench.py`
  - `docs/exp/plan_b_audit_manifest.md`
