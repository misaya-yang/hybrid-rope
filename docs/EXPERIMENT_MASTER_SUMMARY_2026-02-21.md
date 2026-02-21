# Experiment Master Summary

Updated: `2026-02-21`

This file is the single summary for advisor reporting and paper drafting context.

## 1. Completed Experiment Lines

### A. From-Scratch Scaling Line

- Scope: 50M / 100M / 350M / 700M style frequency comparisons.
- Key evidence:
  - `results/advisor_package_2026-02-15/01_scaling_from_scratch/`
  - `results/advisor_package_2026-02-15/06_700m_trainfreq/`
- Typical reported finding:
  - hybrid-like schedules improve long-context PPL versus pure geometric baselines.

### B. Llama Long-Context Shape/Theta Controls

- Scope: shape-preserving and theta-boundary ablations.
- Key evidence:
  - `results/advisor_package_2026-02-15/02_llama_long_context/`

### C. Fair 8B LoRA Baselines

- Scope: YaRN / PI / Hybrid / PI-soft under matched budget.
- Key evidence:
  - `results/advisor_package_2026-02-15/03_llama8b_fair_lora/`
  - `results/advisor_package_2026-02-15/04_niah_and_retrieval/`
- Raw server mirror:
  - `server_artifacts_2026-02-21/results/llama8b_fair_lora_suite_20260214/`
  - `server_artifacts_2026-02-21/results/llama8b_post_eval_20260214/`

### D. Qwen Cross-Model Line

- Scope: qwen hybrid-lora and cross-model checks.
- Key evidence:
  - `results/advisor_package_2026-02-15/05_qwen_and_cross_model/`

### E. Sigmoid-RoPE Phase 1-3

- Scope:
  - analytical formula validation
  - fine search and parameter refit
  - model selection, sensitivity, passkey debug
- Key evidence:
  - `sigmoid_rope_experiments/data/`
  - `sigmoid_rope_experiments/results/`
  - `sigmoid_rope_experiments/data/phase3/`
  - `sigmoid_rope_experiments/results/phase3/`

## 2. In-Progress Line

### Sigmoid-RoPE Phase 4 (Training-Time Validation)

- Script:
  - `sigmoid_rope_experiments/run_phase4.py`
- Current status:
  - running on remote RTX PRO 6000 server
  - two-model matched training (standard vs sigmoid frequencies)
  - periodic logs already synced
- Live evidence snapshot:
  - `server_artifacts_2026-02-21/sigmoid_rope_experiments/run_phase4.log`
  - `server_artifacts_2026-02-21/sigmoid_rope_experiments/data/training_log_standard.csv`
  - `server_artifacts_2026-02-21/sigmoid_rope_experiments/data/training_log_sigmoid.csv`

## 3. Environment Summary

- Local repo:
  - `e:/rope/hybrid-rope`
- Remote repo:
  - `/root/autodl-tmp/dfrope/hybrid-rope`
- Remote runtime (phase4):
  - Python `3.12.3`
  - Torch `2.8.0+cu128`
  - GPU `NVIDIA RTX PRO 6000 Blackwell Server Edition`

Detailed environment notes:

- `docs/EXPERIMENT_ENVIRONMENT_2026-02-21.md`

## 4. Evidence Integrity Notes

- `server_artifacts_2026-02-21/` is mirrored from server using data-only sync.
- Weights/checkpoints are excluded by policy.
- Curated stable package remains:
  - `results/advisor_package_2026-02-15/`

## 5. Recommended Citation Sources

- Stable, advisor-facing claims:
  - cite `results/advisor_package_2026-02-15/`
- Latest in-progress status:
  - cite `server_artifacts_2026-02-21/` with timestamp language.
