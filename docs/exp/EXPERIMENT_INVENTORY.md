# Experiment Inventory (Authoritative Index Mirror)

Last updated: 2026-02-25

This file is a **human/AI-friendly index** of experiments, artifacts, and whether they are usable for paper claims.

Source of truth:
- `docs/EXPERIMENT_REGISTRY.md` is the only authoritative registry.
- If this file conflicts with `docs/EXPERIMENT_REGISTRY.md`, follow `docs/EXPERIMENT_REGISTRY.md`.

Status legend:
- `VALID`: protocol-clean, traceable, paper-usable.
- `PENDING`: incomplete, missing gates, or needs re-run.
- `INVALID`: broken protocol or wrong scale; do not cite.
- `DEPRECATED`: explicitly banned for main claims (may be used in failure-mode discussion only).

For each item, keep 6 fields:
- `ID`, `Status`, `Claim`, `Artifacts`, `Reproduce`, `Notes`.

## VALID (Paper-Ready)

### Tier 1: From-scratch scaling (TinyStories)

| ID | Status | Claim | Artifacts | Reproduce | Notes |
|---|---|---|---|---|---|
| `EXP_50M_3SEED` | `VALID` | Hybrid improves long-context PPL vs geometric (3 seeds). | `results/evidence_chain_50m_3cfg3seed/results.json` | `python archives/a100/unified_search_3cfg_3seed.py` | See `docs/EXPERIMENT_REGISTRY.md` for exact protocol. |
| `EXP_50M_YARN` | `VALID` | Hybrid beats YaRN under the same TinyStories setup. | `results/50m_yarn_compare_v2/results.json` | `python artifacts/a100_2026-02-13/scripts/run_50m_yarn_compare.py` | YaRN is highly sensitive; this experiment is used as controlled contrast. |
| `EXP_100M_FINAL` | `VALID` | Hybrid improves 16K PPL at 100M scale. | `artifacts/a100_2026-02-13/data/100m_scaling/` | See run metadata under artifacts folder. | Registry notes mention a unified script; do not invent entrypoints. |
| `EXP_350M_FINAL` | `VALID` | Hybrid improvement persists at 350M scale. | `artifacts/a100_2026-02-13/data/350m_final/results.json` | `python archives/a100/run_350m_final.py` | Chunk eval protocol; keep identical chunking if re-running. |

### Tier 1.5-2: Mechanism / Phase-4 evidence

| ID | Status | Claim | Artifacts | Reproduce | Notes |
|---|---|---|---|---|---|
| `EXP_PHASE4_124M` | `VALID` | Sigmoid-like schedules reduce long-context PPL collapse vs standard. | `sigmoid_rope_experiments/data/ppl_vs_length.csv` | `python sigmoid_rope_experiments/run_phase4_corrected.py` | Must ensure non-byte tokenizer and non-synthetic dataset per frozen protocol. |
| `EXP_COLLISION_D` | `VALID` | Shape dominates base in collapse regime (e.g., 22x -> 1.08x). | `results/anchored_sigmoid_v3_followup/` | See `results/README.md` + run logs in folder. | Used as "waterbed / collapse" support evidence. |
| `EXP_ATTN_D_DIST` | `VALID` | Attention distance distribution is approximately power-law. | `results/attention_distribution/` | Use attention probing script referenced by registry. | This is a theory-to-practice bridge; keep layers/sequence sampling documented. |

## PENDING (Must finish gates before citing)

| ID | Status | Claim | Artifacts | Reproduce | Notes |
|---|---|---|---|---|---|
| `EXP_8B_FAIR_LORA` | `PENDING` | Fair-protocol 8B comparison under identical injection path and budget. | Server: `/root/autodl-tmp/dfrope/hybrid-rope/results/overnight_8h/summary/` | `python scripts/run_llama8b_fair_suite.py` / `archives/2026-02-22/scripts/run_overnight_8h.py` | Must pass LongBench parity, full lb21, per-sample traces, and multi-seed before main claims. |

## INVALID (Do not cite)

| ID | Status | Claim | Artifacts | Reproduce | Notes |
|---|---|---|---|---|---|
| `Task3/Task4 (2026-02-22 mentor plan)` | `INVALID` | N/A | `results/theory_2026-02-22/mentor_plan_execution_summary.json` | N/A | Wrong dataset/tokenizer mode (`Synthetic-Passkey`, byte tokenizer), PPL scale anomaly; only for debugging postmortem. |

## DEPRECATED (Do not cite in main results)

| ID | Status | Claim | Artifacts | Reproduce | Notes |
|---|---|---|---|---|---|
| `BAD_8B_LORA_OLD` | `DEPRECATED` | N/A | `docs/EXPERIMENT_REGISTRY.md` (deprecated table) | N/A | Confounded protocol: mixes HF `rope_scaling` and monkey patch paths. Use only for "failure mode" narrative if needed. |
| `BAD_50M_BASE_300K` | `DEPRECATED` | Base-shape interaction failure mode. | Registry entry | N/A | Only for limitations/discussion; never used for positive claims. |
| `BAD_ZERO_SHOT_SWAP` | `DEPRECATED` | Zero-shot swap causes collapse. | Registry entry | N/A | Keep as limitations: "needs training to adapt". |

## New Evidence Requirements (Hard Gate)

Any new experiment becomes `VALID` only when all are true:
1. Protocol is frozen and auditable (base/tokenizer/template/decode/injection path/manifest).
2. Has traceability: `code_hash + env_freeze + config + data/model identifiers`.
3. LongBench claims use full task set (`lb21`), not preview-only.
4. Saves per-sample traces (including truncation meta and failure types).
5. Statistics: paired bootstrap CI + permutation/sign-flip p-value + effect size + FDR (BH/BY).
