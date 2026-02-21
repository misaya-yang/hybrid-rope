# Research Storyline (Paper-Oriented)

Updated: `2026-02-21`

## 1. Problem

Long-context extrapolation with standard geometric RoPE often shows strong degradation when sequence length exceeds the trained range. The project investigates whether frequency allocation redesigns (Hybrid/Sigmoid-style) improve stability and downstream retrieval.

## 2. Experiment Arc

### Stage A: Scaling and Failure Diagnostics

- From-scratch small/medium models (50M, 100M, 350M, 700M lines).
- Goal:
  - identify whether frequency schedule changes reduce long-length collapse.
- Evidence:
  - `results/advisor_package_2026-02-15/01_scaling_from_scratch/`
  - `results/advisor_package_2026-02-15/06_700m_trainfreq/`

### Stage B: Llama Long-Context Frequency Shape Controls

- Controlled comparisons on shape/theta variants.
- Goal:
  - isolate which frequency-shape factors correlate with long-context stability.
- Evidence:
  - `results/advisor_package_2026-02-15/02_llama_long_context/`

### Stage C: Fair 8B LoRA Baselines

- Llama-3-8B fair setup with YaRN / PI / Hybrid / PI-soft.
- Goal:
  - compare perplexity and downstream long-context behavior under matched training budget.
- Evidence:
  - `results/advisor_package_2026-02-15/03_llama8b_fair_lora/`
  - `results/advisor_package_2026-02-15/04_niah_and_retrieval/`
  - latest raw mirror: `server_artifacts_2026-02-21/results/llama8b_fair_lora_suite_20260214/`

### Stage D: Sigmoid-RoPE Mechanistic Line

- Phase 1-3:
  - formula validation
  - fine search and refit
  - model selection + sensitivity + passkey debug
- Phase 4:
  - training-time validation with two matched GPT models (standard vs sigmoid frequencies)
- Evidence:
  - local canonical outputs: `sigmoid_rope_experiments/data/`, `sigmoid_rope_experiments/results/`
  - server snapshot: `server_artifacts_2026-02-21/sigmoid_rope_experiments/`
  - zero-shot tensor mechanism artifact:
    - `artifacts/neurips_zero_shot_mechanism/`

## 3. Main Claim Structure For Drafting

1. Standard RoPE has measurable long-context instability under extrapolation.
2. Frequency redesign can improve theoretical collision metrics.
3. Zero-shot frequency replacement can fail (implementation + adaptation caveats).
4. Training-time integration is required for fair capability conclusions.
5. Robustness and downstream retrieval must accompany perplexity claims.

## 4. Current Live Status

- Phase4 training-time experiment is active on server.
- Progress snapshot has been synced into:
  - `server_artifacts_2026-02-21/sigmoid_rope_experiments/run_phase4.log`
  - `server_artifacts_2026-02-21/sigmoid_rope_experiments/data/training_log_standard.csv`
  - `server_artifacts_2026-02-21/sigmoid_rope_experiments/data/training_log_sigmoid.csv`

## 5. Citation Policy Inside This Repo

- Use curated package for stable numbers:
  - `results/advisor_package_2026-02-15/`
- Use server snapshot for "latest in-progress" statements:
  - `server_artifacts_2026-02-21/`
