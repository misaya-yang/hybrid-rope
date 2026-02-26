# Hybrid-RoPE Research Repository

This repository is the paper workspace for long-context RoPE research.
It contains code, experiment evidence, analysis docs, and server snapshots.

## Start Here (New Collaborators)

1. `AI_HANDOFF.md` (what to run, where outputs land)
2. `docs/README.md` (documentation index + citation rules)
3. `docs/EXPERIMENT_REGISTRY.md` (single source of truth for VALID/PENDING/INVALID)
4. `docs/REPO_STRUCTURE_CN.md` (repo “file manager”: where to put new things)
5. `docs/exp/EXPERIMENT_INVENTORY.md` (claim-ready inventory mirror)
6. `docs/exp/plan_b_audit_manifest.md` (strict red-line audit + recovery checklist)

## What This Repo Is For

- Primary goal: support paper writing with auditable experiment evidence.
- Scope:
  - From-scratch scaling experiments (50M / 100M / 350M / 700M lines).
  - Llama/Qwen long-context adaptation and fair LoRA comparisons.
  - Sigmoid-RoPE analytical study (phase collision, fitting, robustness).
  - Training-time validation pipeline (`sigmoid_rope_experiments`).
- Rule: no model weights/checkpoints are committed as evidence artifacts.

## Repo Layout (High Level)

- `docs/`: paper-facing docs + protocols + experiment registry (start here)
- `scripts/`: repo-level runnable entrypoints (training/eval/audit)
- `rope/`: core RoPE schedule/injection code
- `results/`: curated result bundles + paper-ready small artifacts
- `paper_exports/`: dated paper export packages (tables/figures/json, no weights)
- `artifacts/`: machine/cluster snapshots and small manifests
- `archives/`: historical snapshots/batch reports (kept for traceability, not “current”)
- `experiments/`: one-off or side projects (non-core; see per-folder README)

## Read In This Order (Paper Work)

1. `knowledge_base/README.md`
2. `knowledge_base/ALL_IN_ONE.md`
3. `knowledge_base/00_项目与结论总览.md`
4. `docs/README.md`
5. `docs/env/AI_ENVIRONMENT_SNAPSHOT_2026-02-23.md`
6. `docs/notes/RESEARCH_STORYLINE_2026-02-21.md`
7. `docs/env/EXPERIMENT_ENVIRONMENT_2026-02-21.md`

## Canonical Evidence Locations

- Curated advisor package:
  - `results/advisor_package_2026-02-15/`
- Latest server evidence snapshot (data/logs only):
  - `archives/server_artifacts_2026-02-21/`
- Sigmoid-RoPE project outputs:
  - `sigmoid_rope_experiments/data/`
  - `sigmoid_rope_experiments/results/`
- Historical machine-specific snapshots:
  - `artifacts/`
  - `archives/server_artifacts_2026-02-13/`

## Main Code Entrypoints

- 8B post-eval pipeline:
  - `scripts/run_8b_post_eval.py`
- 8B LoRA variant training:
  - `scripts/train_llama8b_lora_variant.py`
- Cross-model fair LoRA batch training:
  - `scripts/cross_model_finetune.sh`
  - `scripts/train_cross_model_lora.py`
- NIAH eval:
  - `scripts/eval_niah_recall.py`
- LongBench eval:
  - `scripts/eval_longbench.py`
- Attention-integrated isolated pipeline:
  - `train.py` (LLaMA-3-8B anchored/static/dynamic penalty trainer)
  - `scripts/isolated/attn/new_eval_longbench_attnbias.py`
  - `scripts/isolated/attn/next_attn_lora_queue.sh`
- Attention distance prior estimation:
  - `scripts/run_attn_hist.py`
- Sigmoid phase pipelines:
  - `sigmoid_rope_experiments/run_all.py`
  - `sigmoid_rope_experiments/run_phase2.py`
  - `sigmoid_rope_experiments/run_phase3.py`
  - `sigmoid_rope_experiments/run_phase4.py`

## Notes

- If you need fresh remote evidence, run:
  - `tools/sync_server_evidence_data_only.ps1`
- The repository may contain ongoing-run logs; check timestamps before citing numbers.
