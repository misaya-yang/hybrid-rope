# Rebuttal 0723

Last audited: 2026-07-27
Status: `review_received / internal_draft / sendable_core`

This directory is the single working entry point for the current rebuttal.
Start from the retained full OpenReview panel in
`00_REVIEWER_SCORES_AND_AC_METAREVIEW.md` (AC `XLtL`, reviewers `Dz6s`,
`zWsa`, `27bE`), then select only evidence that directly answers those
concerns.

## 1. Start here

| Order | File | Purpose |
| ---: | --- | --- |
| 1 | `00_REVIEWER_SCORES_AND_AC_METAREVIEW.md` | Authoritative retained concern text and stable IDs |
| 2 | `01_REBUTTAL_PLAYBOOK.md` | Result-first response strategy, concern routes, English wording, and send gate |
| 3 | `02_RESPONSE_QUESTIONS_AND_OUTCOMES.md` | Numbered full-panel question inventory, existing answers, and final send outcomes |
| 3b | `03_STRATEGY_REVIEW_AND_OPTIMIZATION.md` | External-view critique of the response strategy; its P0/P1 items are applied in `01_` |
| 4 | `theory_results/REVIEWER_USABLE_EVIDENCE_LEDGER_20260726.md` | Compact positive-evidence index with exact numbers and mandatory limits |
| 5 | `theory_results/INTERNAL_NEGATIVE_AND_DIAGNOSTIC_LEDGER_20260726.md` | One consolidated index for negative, diagnostic, superseded, and design-only material |
| 6 | `theory_results/EVQ_COSH_REBUTTAL_PRINCIPLES.md` | Method identity, theory boundaries, experiment rules, and stopping logic |

The full reports below remain evidence owners. The two ledgers are routing
documents; they do not replace protocol, raw hashes, or uncertainty.

## 2. Current response spine (wide, four blocks)

1. **Submitted mechanism package:** Primary I EVQ×YaRN (three-seed,
   fixed-transform substrate leverage), Primary III MLA scarce-channel
   (three-seed), submitted 750M strict AR, 8B LoRA, and video-DiT breadth.
   For the DAPE question, answer the verified `10x/100x` PE-learning-rate
   sweep directly; use fixed schedules for shape attribution.
2. **Theory attribution:** four epistemic layers; independent τ sweep; the
   three-seed M4 exact-range factorial across base/length/head settings; and a
   deformation-matched non-Cosh schedule (allocation axis, not Cosh universal
   optimum).
3. **FMRoPE:** concede missing citation; training-grid allocation ≠ range
   retarget; controlled comparison with honest retargeted-FMRoPE boundary.
4. **Stronger eval / scale:** mature 1.485B/8B AR and RULER under task-family
   protocols; surface submitted Appendix D, Table 23 and the 750M strict-AR
   row; use the raw-hash-backed 1.485B step-0→1,000 scratch comparison for the
   requested pre-specified scale run.
5. Separate NLL/PPL, NLL-gap PK, strict AR, RULER, and QA; claim task-adapted
   2× transfer; keep 4× and unseen-task open.
6. Do not tunnel-vision on LoRA-only; do not upgrade supporting DiT/progressive
   rows; do not use failed post-sub MLA scarcity as a win.

## 3. Positive standalone evidence owners

### Method identity and theory attribution

| Owner | Role | Status |
| --- | --- | --- |
| `theory_results/M4_EXACT_RANGE_FACTORIAL_RESULT_20260726.md` | Three-seed exact-range Cosh/non-Cosh factorial across base, training length, and head dimension | Completed workstation-backed mechanistic owner; core attribution evidence |
| `theory_results/EXPERIMENT_REPORT_20260724.md` | Fixed schedules, \(\tau\), held-out base/head, native-span controls, FMR boundary | Active numeric entry |
| `theory_results/M4_EXACT_RANGE_FACTORIAL_RESULT_20260726.md` + curated JSON | 50.9M exact-range factorial over base, training length, head dimension, \(\tau\), and matched exponential | Raw/hash-backed supporting/mechanistic evidence |
| `theory_results/MATCHED_RANGE_COSH_500M_3SEED_20260724.md` + JSON | Exact-range, three-seed interior-allocation identification | Conditional: raw/per-seed promotion pending |
| `theory_results/MATCHED_RANGE_COSH_500M_S42_20260724.md` | Seed-42 raw-backed exact-range control | Reviewer-usable, single seed |
| `theory_results/PHASE16_99RUN_RAW_REANALYSIS_20260724.md` | Fallible operating-rule audit | Active guardrail |
| `theory_results/TRAINING_FREE_TAU_SELECTOR_20260724.md` | Failed training-free selector | Negative guardrail |
| `theory_results/FREQUENCY_DEFINITION_MANIFEST.json` | Paper-Geo, Std-Geo, and EVQ identity | Implementation contract |

### Mature-model and capability evidence

| Owner | Role | Status |
| --- | --- | --- |
| `theory_results/OLMO2_1B_4K_ONLY_ROUTING_CONVERSION_20260726.md` | 1.485B actual-parameter, 4K-only counterfactual LoRA; NLL and 8K strict exact | Positive, task-family matched |
| `theory_results/OLMO2_1B_4K_RULER_FAMILY_ADAPTATION_20260726.md` + JSON | Complete 13-task 4K/8K/16K task-family-adapted RULER | Positive supporting, single seed |
| `theory_results/OLMO2_1B_MATCHED_RULER_CONTINUATION_20260727.md` | Matched Native/EVQ physical-4K, 13-family continuation at 1.485B | Positive 2×/4× length transfer; Native 4K boundary |
| `theory_results/EVQ_8B_ADAPTATION_EVIDENCE_20260724.md` | Matched LLaMA-8B NLL, remote-source dependence, rank, and QA boundary | Positive mechanism/supporting |
| `theory_results/LLAMA8B_MATCHED_RULER_MIX_20260726.md` + JSON | Matched LLaMA-8B Native/EVQ 8K/16K RULER-family adaptation | Positive 2× supporting; 32K negative |
| `theory_results/OLMO2_1B_RELEASED_ROPE_BASELINE_20260725.md` | 1.485B step-0→1,000 from-scratch scale-transfer comparison | Post-submission raw-hash-backed; single trajectory |
| `theory_results/olmo2_1b_released_rope_baseline_20260725.json` | Earlier released-Geo baseline snapshot only | Native-only scope; not the owner of the later paired EVQ result |
| `theory_results/LLAMA8B_FRESH_COUNTERFACTUAL_RESULT_20260726.md` | Fresh EVQ-only LLaMA counterfactual feasibility arm | Complete single arm; not a matched comparison or headline |

### Counterfactual classification

| Owner | Purpose | Status |
| --- | --- | --- |
| `theory_results/LLAMA8B_COUNTERFACTUAL_REBUTTAL_PLAN_20260726.md` | Matched Native/EVQ LLaMA-8B counterfactual continuation | Design only; not required for the current response; do not run |

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

## 6. Optional promotion gates

1. Promotion of the older 151.9M exact-range three-seed per-seed raw/hash/CI
   bundle if that separate result is ever used.
2. Promotion of the older held-out base/head aggregate if that separate result
   is ever used.
3. Correction of future-dated OLMo metadata before using those specific
   future-dated owners.
4. Promotion of selected untracked reports, JSONs, and experiment packages to
   canonical `main`.

None of these gates blocks the current core response. The M4 factorial owns the
core multi-configuration exact-range attribution; the older 151.9M aggregates
remain optional. The step-1,000 scratch result is already reviewer-usable; the
native-only JSON does not supersede its later canonical Markdown owner. No new
LLaMA counterfactual experiment is required or authorized.

No file under `paper/` is modified by this workspace organization.
