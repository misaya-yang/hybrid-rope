# Hybrid-RoPE Research Repository

This repository is the paper workspace for long-context RoPE research.
It contains code, experiment evidence, analysis docs, and server snapshots.

## What This Repo Is For

- Primary goal: support paper writing with auditable experiment evidence.
- Scope:
  - From-scratch scaling experiments (50M / 100M / 350M / 700M lines).
  - Llama/Qwen long-context adaptation and fair LoRA comparisons.
  - Sigmoid-RoPE analytical study (phase collision, fitting, robustness).
  - Training-time validation pipeline (`sigmoid_rope_experiments`).
- Rule: no model weights/checkpoints are committed as evidence artifacts.

## Read In This Order

1. `AI_HANDOFF.md`
2. `docs/EXPERIMENT_MASTER_SUMMARY_2026-02-21.md`
3. `docs/RESEARCH_STORYLINE_2026-02-21.md`
4. `docs/EXPERIMENT_ENVIRONMENT_2026-02-21.md`
5. `docs/SERVER_EVIDENCE_SYNC_2026-02-21.md`
6. `results/advisor_package_2026-02-15/INDEX.md`

## Canonical Evidence Locations

- Curated advisor package:
  - `results/advisor_package_2026-02-15/`
- Latest server evidence snapshot (data/logs only):
  - `server_artifacts_2026-02-21/`
- Sigmoid-RoPE project outputs:
  - `sigmoid_rope_experiments/data/`
  - `sigmoid_rope_experiments/results/`
- Historical machine-specific snapshots:
  - `artifacts/`
  - `server_artifacts_2026-02-13/`

## Main Code Entrypoints

- 8B post-eval pipeline:
  - `scripts/run_8b_post_eval.py`
- 8B LoRA variant training:
  - `scripts/train_llama8b_lora_variant.py`
- NIAH eval:
  - `scripts/eval_niah_recall.py`
- LongBench eval:
  - `scripts/eval_longbench.py`
- Sigmoid phase pipelines:
  - `sigmoid_rope_experiments/run_all.py`
  - `sigmoid_rope_experiments/run_phase2.py`
  - `sigmoid_rope_experiments/run_phase3.py`
  - `sigmoid_rope_experiments/run_phase4.py`

## Notes

- If you need fresh remote evidence, run:
  - `tools/sync_server_evidence_data_only.ps1`
- The repository may contain ongoing-run logs; check timestamps before citing numbers.
