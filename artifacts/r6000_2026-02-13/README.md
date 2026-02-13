# R6000 Artifacts (2026-02-13)

This folder tracks the Qwen2.5-7B Hybrid-LoRA run on the R6000 machine.

## Contents
- `scripts/`: exact training/evaluation scripts used on server
- `logs/`: live training log snapshots synced from server
- `R6000_QWEN_HYBRID_STATUS_2026-02-13.md`: current progress and run configuration

## Server Result Paths
- Training log: `/opt/dfrope/results/qwen_hybrid_lora/run.log`
- Final adapter (after finish): `/opt/dfrope/results/qwen_hybrid_lora/final_lora`
- Eval output (after finish): `/opt/dfrope/results/qwen_hybrid_lora/eval_suite.json`
