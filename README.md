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

1. `knowledge_base/README.md`
2. `knowledge_base/ALL_IN_ONE.md`
3. `knowledge_base/00_项目与结论总览.md`
4. `AI_HANDOFF.md`
5. `docs/AI_ENVIRONMENT_SNAPSHOT_2026-02-23.md`
6. `docs/RESEARCH_STORYLINE_2026-02-21.md`
7. `docs/EXPERIMENT_ENVIRONMENT_2026-02-21.md`

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
