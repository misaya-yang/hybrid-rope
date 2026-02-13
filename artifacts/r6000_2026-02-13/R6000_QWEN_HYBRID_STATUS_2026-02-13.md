# R6000 Qwen Hybrid LoRA Status (2026-02-13)

## Current Training Job
- Machine: `connect.bjb1.seetacloud.com:42581` (RTX PRO 6000)
- Process: `run_qwen_hybrid_lora_train.py`
- PID: `7482`
- Start Time: `2026-02-13 15:47:24`
- Training mode: `DATASET_MODE=tinystories`
- Sequence length: `8192`
- Target tokens: `40,000,000`
- Max steps: `500`
- Precision: `4bit + bf16 compute`

## Live Progress Snapshot
- Data build finished: `39,993,344` usable tokens (`4882` chunks)
- Trainer has entered optimization loop
- Latest logged record:
  - `loss=3.64`
  - `grad_norm=2.266`
  - `learning_rate=6e-05`
  - `epoch=0.01639`
- Rough progress in progress-bar stream: `~3%`
- GPU (snapshot): `~31.7GB / 97.9GB`, util `~92-100%`

## Scripts Synced Into This Repo
- `artifacts/r6000_2026-02-13/scripts/run_qwen_hybrid_lora_train.py`
- `artifacts/r6000_2026-02-13/scripts/run_qwen_eval_suite.py`

## Logs Synced Into This Repo
- `artifacts/r6000_2026-02-13/logs/qwen_hybrid_lora_run_live.log`

## Planned Evaluation (after LoRA finishes)
`run_qwen_eval_suite.py` compares 3 variants:
1. `base` (Qwen2.5-7B-Instruct)
2. `base_yarn8` (inference-side YaRN scaling)
3. `hybrid_lora` (our Hybrid-RoPE + LoRA adapter)

Metrics:
- PPL at lengths: `8192 / 16384 / 24576 / 32768`
- Passkey multiple-choice accuracy (non-PPL retrieval)
- KV retrieval multiple-choice accuracy (non-PPL retrieval)

Output target:
- `/opt/dfrope/results/qwen_hybrid_lora/eval_suite.json`
