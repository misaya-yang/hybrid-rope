# Opus 4.8 Issue Resolution Ledger

Purpose: track how each Opus 4.8 issue has been handled. This file complements
`docs/overview/OPUS48_REVIEW_AUDIT_CHECKLIST.md`: the checklist is the reviewer
attack inventory; this ledger records the current resolution state, evidence,
and remaining gate for each item.

Resolution labels:

- `Resolved for wording`: the paper/docs now scope the claim safely, but no new
  experiment was added.
- `Resolved in code`: a code path was patched or test coverage was added.
- `Evidence-gated`: code/report evidence exists, but exact JSON/checkpoint/data
  artifacts are missing.
- `Experiment-gated`: the reviewer concern requires a new or recovered
  experiment before it can be defended as solved.
- `Concede/scope`: the correct rebuttal is to concede the limitation and avoid
  relying on the row.

## Ledger

| ID | Resolution state | What was done | Evidence | Remaining gate |
| --- | --- | --- | --- | --- |
| O48-01 | Experiment-gated | Novelty is scoped as closed-form allocation, not unique superiority over all schedules. | `docs/overview/OPUS48_REVIEW_AUDIT_CHECKLIST.md`, `docs/overview/PAPER_DESCRIPTION_AUDIT.md` | Add rebased-Geo/fixed-interpolation control or concede missing baseline. |
| O48-02 | Resolved for wording | Surrogate language is kept as surrogate-derived/functionally validated, not full-attention-derived. | `paper/sections/03_theory.tex`, `docs/overview/OPUS48_REVIEW_AUDIT_CHECKLIST.md` | None for wording; stronger theory would be new work. |
| O48-03 | Resolved for wording | Constant-alpha tractability remains an explicit modeling choice. | `paper/appendix/a1_proofs.tex`, `docs/overview/OPUS48_REVIEW_AUDIT_CHECKLIST.md` | None unless adding variable-alpha experiments. |
| O48-04 | Concede/scope | Tau rule is treated as an operating default/basin selector, not a global optimum. | `docs/overview/OPUS48_REVIEW_AUDIT_CHECKLIST.md`, `docs/overview/OPUS48_COMPLETION_AUDIT.md` | Direct `L_eff^J` measurement if the rebuttal wants to defend scaling more strongly. |
| O48-05 | Resolved for wording | Flat basin is framed as supporting robust non-geometric allocation, not a sharp tau optimum. | `docs/overview/PAPER_DESCRIPTION_AUDIT.md`, `docs/overview/OPUS48_AUDIT_CONTROL_CENTER.md` | None for current claim scope. |
| O48-06 | Resolved for wording | PK is consistently scoped as teacher-forced NLL-gap unless AR exact is explicitly named. | `paper/sections/05_experiments.tex`, `paper/tables/table3_capability_passkey.tex`, `docs/overview/DATA_PREPARATION.md` | Add multi-seed AR exact only if claiming AR retrieval. |
| O48-07 | Resolved for wording | EVQ x YaRN claim now emphasizes higher matched-scale YaRN leverage on the EVQ substrate. | `paper/sections/05_experiments.tex`, `docs/overview/RESULT_PROVENANCE_MANIFEST.md` M1 | None for matched-scale claim; tuned-scaler claim remains unsupported. |
| O48-08 | Experiment-gated | Matched-scale scope is explicit; tuned Geo+YaRN/LongRoPE-style dominance is not claimed. | `docs/overview/RESULT_PROVENANCE_MANIFEST.md` M1, `docs/overview/OPUS48_AUDIT_CONTROL_CENTER.md` | Add tuned-scale controls or concede baseline gap. |
| O48-09 | Resolved for wording | PE-dominant Geo/DAPE/EVQ rows are seed-42 scoped. | `paper/tables/table4_pe_dominant.tex`, `docs/overview/RESULT_PROVENANCE_MANIFEST.md` M2 | Add two more seeds if using as stronger primary evidence. |
| O48-10 | Resolved for wording | 128-to-8K result is framed as PE-dominant diagnostic, not ordinary downstream evidence. | `docs/overview/RESULT_PROVENANCE_MANIFEST.md` M2, `docs/overview/PAPER_DESCRIPTION_AUDIT.md` | None for diagnostic use. |
| O48-11 | Resolved for wording | MLA `tau=1.414` is now described as empirical `d_eff=128`, distinct from code `head_dim=64` and `d_rope=32`. | `paper/sections/05_experiments.tex`, `paper/appendix/a3_supporting_results.tex`, `docs/overview/OPUS48_FORENSIC_AUDIT_REPORT.md` | Direct tau ablations still needed to defend the convention. |
| O48-12 | Experiment-gated | Gap is explicitly recorded; no implication remains that direct MLA `tau=d_rope/sqrt(L)` is solved. | `docs/overview/OPUS48_REVIEW_AUDIT_CHECKLIST.md`, `docs/overview/OPUS48_COMPLETION_AUDIT.md` | Run/report `tau=d_rope/sqrt(L)` and code-`head_dim/sqrt(L)` ablations. |
| O48-13 | Evidence-gated | 1B raw EVQ reversal is promoted to a real limitation/root-cause target, not hidden. | `docs/overview/OPUS48_FORENSIC_AUDIT_REPORT.md`, `docs/overview/RESULT_PROVENANCE_MANIFEST.md` M4, `docs/overview/OPUS48_ARTIFACT_RECOVERY_RUNBOOK.md` | Recover exact JSON/checkpoint/data manifests or remove numeric rebuttal reliance. |
| O48-14 | Evidence-gated | 1B row is classified as code-backed/report-backed but not compact-JSON-backed; external recovery steps are now specified. | `docs/overview/EXPERIMENT_CODE_RESULT_AUDIT.md`, `docs/overview/OPUS48_FORENSIC_AUDIT_REPORT.md`, `docs/overview/OPUS48_ARTIFACT_RECOVERY_RUNBOOK.md` | Import sanitized external manifests or rerun. |
| O48-15 | Resolved for wording | Old MLA-32/base500K is called a scarce-channel stress test, not production-identical DeepSeek MLA. | `docs/overview/OPUS48_REVIEW_AUDIT_CHECKLIST.md`, `docs/overview/RESULT_PROVENANCE_MANIFEST.md` M3 | None for scoped claim; production-like arm would be new evidence. |
| O48-16 | Concede/scope | LoRA is kept supporting/exploratory because matched Geo+LoRA is missing. | `paper/appendix/a4_supporting_experiments.tex`, `docs/overview/PAPER_DESCRIPTION_AUDIT.md` | Add matched Geo+LoRA before using as EVQ-specific transfer proof. |
| O48-17 | Resolved for wording | LoRA phase-transition language is not used as primary theory evidence. | `docs/overview/OPUS48_REVIEW_AUDIT_CHECKLIST.md`, `docs/overview/PAPER_DESCRIPTION_AUDIT.md` | None unless expanding LoRA claims. |
| O48-18 | Resolved for wording | Video/DiT evidence is kept supporting and conditioned on dead-channel/base sensitivity. | `docs/overview/PAPER_CLAIMS_MAP.md`, `docs/overview/OPUS48_AUDIT_CONTROL_CENTER.md` | No broad video claim unless packaged multi-seed evidence is promoted. |
| O48-19 | Concede/scope | Progressive training remains single-seed/supporting and is not durability proof. | `docs/exp/2026-03-11_phase17c_2048_continue_results.md`, `docs/overview/OPUS48_REVIEW_AUDIT_CHECKLIST.md` | Multi-seed progressive rerun if used for durability. |
| O48-20 | Resolved for wording | Scale claims remain mechanism-study scoped; no frontier-scale validation is claimed. | `paper/sections/05_experiments.tex`, `docs/overview/OPUS48_AUDIT_CONTROL_CENTER.md` | Larger from-scratch or controlled adaptation experiments if claiming scale. |
| O48-21 | Resolved for wording | Downstream accuracy is a non-regression/capacity-limited check, not main evidence. | `paper/sections/05_experiments.tex`, `docs/overview/OPUS48_FORENSIC_AUDIT_REPORT.md` | None for current scope. |
| O48-22 | Experiment-gated | Missing tuned LongRoPE2/CoPE/tuned-scale YaRN baselines are explicit open gaps. | `docs/overview/OPUS48_AUDIT_CONTROL_CENTER.md`, `docs/overview/OPUS48_COMPLETION_AUDIT.md` | Add baselines or concede. |
| O48-23 | Resolved for wording | Key caveats are promoted into overview/rebuttal docs and old docs are quarantined. | `docs/overview/OPUS48_AUDIT_CONTROL_CENTER.md`, `docs/overview/OPUS48_COMPLETION_AUDIT.md`, old-doc audit notices | Keep paper/rebuttal synced with these caveats. |
| O48-24 | Resolved for wording | The historical paper-local rebuttal playbook was retired; its overclaims remain prohibited in the current triage path. | `rebuttal/REVIEWER_TRIAGE_PLAYBOOK.md`, `docs/overview/PAPER_DESCRIPTION_AUDIT.md` | Review again before any final rebuttal submission. |
| O48-25 | Resolved in code / Evidence-gated | Eval scripts now use checkpoint-loaded `inv_freq` explicitly and resolve historical/current MLA run IDs; manifest and historical-script docs separate code support from artifact support. | `eval_extended_3seeds.py`, `yarn_finetune_eval.py`, `tests/test_yarn_checkpoint_inv_freq.py`, `docs/overview/HISTORICAL_SCRIPT_STATUS.md` | External artifacts still need sanitized manifests before promotion. |

## Code-Side Fixes Landed

- `scripts/core_text_phases/eval_extended_3seeds.py`
  - requires checkpoint `attn.rope.inv_freq`
  - clones the loaded buffer before YaRN scaling
  - prints short `inv_freq` hashes
  - resolves current `350m_mla_tau...` and historical `350m_tau...` run IDs
- `scripts/core_text_phases/yarn_finetune_eval.py`
  - same checkpoint-loaded `inv_freq` enforcement for baseline, YaRN, and YaRN+FT
  - same current/historical run-id resolution
- `scripts/core_text_phases/audit_rope_checkpoint.py`
  - offline checkpoint frequency-family audit
- `scripts/core_text_phases/audit_training_artifacts.py`
  - train-cache token math and run-artifact audit
- `scripts/core_text_phases/make_artifact_manifest.py`
  - sanitized external artifact ledger with hashes and tensor metadata

## Tests Covering the Fixes

- `tests/test_yarn_checkpoint_inv_freq.py`
- `tests/test_artifact_manifest.py`
- `tests/test_training_artifact_audit.py`
- `tests/test_opus48_audit_docs.py` for O48 coverage, audit-doc links,
  manifest-hash freshness, and stale high-risk paper/rebuttal phrases

These tests prove helper behavior and artifact-reporting logic. They do not
prove the missing 1B artifacts, because those files are not present in the
compact repository.
