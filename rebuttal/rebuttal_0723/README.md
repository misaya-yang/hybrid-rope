# Rebuttal 0723

Last audited: 2026-07-26
Status: `review_received / internal_draft / needs_author_input`

This directory is the single working entry point for the current rebuttal.
Start from the retained Reviewer 27bE and author-supplied AC concerns, then
select only evidence that directly answers them.

## 1. Start here

| Order | File | Purpose |
| ---: | --- | --- |
| 1 | `00_REVIEWER_SCORES_AND_AC_METAREVIEW.md` | Authoritative retained concern text and stable IDs |
| 2 | `01_REBUTTAL_PLAYBOOK.md` | Result-first response strategy, concern routes, English wording, and send gate |
| 3 | `theory_results/REVIEWER_USABLE_EVIDENCE_LEDGER_20260726.md` | Compact positive-evidence index with exact numbers and mandatory limits |
| 4 | `theory_results/INTERNAL_NEGATIVE_AND_DIAGNOSTIC_LEDGER_20260726.md` | One consolidated index for negative, diagnostic, superseded, and design-only material |
| 5 | `theory_results/EVQ_COSH_REBUTTAL_PRINCIPLES.md` | Method identity, theory boundaries, experiment rules, and stopping logic |

The full reports below remain evidence owners. The two ledgers are routing
documents; they do not replace protocol, raw hashes, or uncertainty.

## 2. Current response spine

1. Concede the missing FMRoPE citation/direct comparison.
2. Define EVQ narrowly as finite training-time frequency-grid allocation.
3. Use fixed schedules and, after promotion, exact-range controls to identify
   allocation separately from scalar range.
4. Correct the “small-model only” impression with already-submitted 8B support,
   then lead with new mature 1.485B and matched 8B results.
5. Separate NLL/PPL, causal source use, strict generation, and RULER.
6. Claim task-adapted 2× length transfer; disclose that broad unseen-task and
   reliable 4× capability remain open.

## 3. Positive standalone evidence owners

### Method identity and theory attribution

| Owner | Role | Status |
| --- | --- | --- |
| `theory_results/EXPERIMENT_REPORT_20260724.md` | Fixed schedules, \(\tau\), held-out base/head, native-span controls, FMR boundary | Active numeric entry |
| `theory_results/MATCHED_RANGE_COSH_500M_3SEED_20260724.md` + JSON | Exact-range, three-seed interior-allocation identification | Conditional: raw/per-seed promotion pending |
| `theory_results/MATCHED_RANGE_COSH_500M_S42_20260724.md` | Seed-42 raw provenance for exact-range | Retain until aggregate promotion |
| `theory_results/PHASE16_99RUN_RAW_REANALYSIS_20260724.md` | Fallible operating-rule audit | Active guardrail |
| `theory_results/TRAINING_FREE_TAU_SELECTOR_20260724.md` | Failed training-free selector | Negative guardrail |
| `theory_results/FREQUENCY_DEFINITION_MANIFEST.json` | Paper-Geo, Std-Geo, and EVQ identity | Implementation contract |

### Mature-model and capability evidence

| Owner | Role | Status |
| --- | --- | --- |
| `theory_results/OLMO2_1B_4K_ONLY_ROUTING_CONVERSION_20260726.md` | 1.485B actual-parameter, 4K-only counterfactual LoRA; NLL and 8K strict exact | Positive, task-family matched |
| `theory_results/OLMO2_1B_4K_RULER_FAMILY_ADAPTATION_20260726.md` + JSON | Complete 13-task 4K/8K/16K task-family-adapted RULER | Positive supporting, single seed |
| `theory_results/EVQ_8B_ADAPTATION_EVIDENCE_20260724.md` | Matched LLaMA-8B NLL, remote-source dependence, rank, and QA boundary | Positive mechanism/supporting |
| `theory_results/LLAMA8B_MATCHED_RULER_MIX_20260726.md` + JSON | Matched LLaMA-8B Native/EVQ 8K/16K RULER-family adaptation | Positive 2× supporting; 32K negative |
| `theory_results/OLMO2_1B_RELEASED_ROPE_BASELINE_20260725.md` + JSON | 1.485B step-0→1,000 from-scratch scale-transfer comparison | Conditional: Markdown/JSON/raw reconciliation required |

### Next experiment

| Owner | Purpose | Status |
| --- | --- | --- |
| `theory_results/LLAMA8B_COUNTERFACTUAL_REBUTTAL_PLAN_20260726.md` | Matched Native/EVQ LLaMA-8B counterfactual continuation | Design only; no result or GPU authorization |

## 4. Negative and internal owners

Use `theory_results/INTERNAL_NEGATIVE_AND_DIAGNOSTIC_LEDGER_20260726.md`
instead of browsing these files as an action queue. Important owners include:

- `OLMO2_N100_GAP_STRUCTURE_AUDIT_20260726.md`;
- `OLMO2_1B_CLEAN_4K_TULU_FULL_RULER_20260728.md`;
- `OLMO2_1B_NON_RULER_ADAPTATION_SEARCH_20260731.md`;
- `OLMO2_POSTHOC_FREQUENCY_TRANSPLANT_OBSTRUCTION_20260726.md`;
- `EVQ_TRUE_OBJECTIVE_ULTRA_AUDIT.md`;
- `TAU_TRUE_ROLE_AND_OPERATING_RULE_AUDIT.md`;
- `EVQ_4X_AND_8K_NO_HARM_THEORY_20260726.md`.

Files dated after the current audit date are internal-only until their actual
run/transfer timestamps and raw packages are reconciled.

## 5. Superseded routing documents

The following are retained for history but are not current execution or
response entry points:

- `OLMO2_1B_OVERNIGHT_EXPERIMENT_SUMMARY_20260726.md`;
- `EVQ_Cosh_NeurIPS2026_Rebuttal_Experiment_Design.md`;
- `ROPE_RANGE_SHAPE_MAPPING_THEORY_AND_5090_PLAN_20260724.md`;
- `OLMO2_MATURITY_ADAPTATION_NEXT_EXPERIMENT_20260726.md`;
- `MLA_YARN_OPERATOR_PARITY_5090_PLAN.md`.

Do not delete standalone evidence owners merely because their conclusions are
summarized in a ledger.

## 6. Current blockers

1. Exact-range three-seed per-seed raw/hash/CI promotion.
2. OLMo step-1,000 Markdown versus curated-JSON/raw reconciliation.
3. Correction of future-dated OLMo metadata.
4. Promotion of selected untracked reports, JSONs, and experiment packages to
   canonical `main`.
5. LLaMA counterfactual continuation, only if the authors choose to run the
   final score-changing experiment.

No file under `paper/` is modified by this workspace organization.
