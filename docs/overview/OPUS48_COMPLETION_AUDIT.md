# Opus 4.8 Objective Completion Audit

Purpose: verify the active Opus 4.8 objective against the current repository
state. This file is not a success claim. It records which requested deliverables
are currently proven by files/tests and which remain open because required
artifacts or experiments are missing.

Objective interpreted:

> Build a rigorous review checklist for all Opus 4.8 issues, deeply audit
> experiment code and paper descriptions, organize the findings into docs, and
> resolve issues one by one where possible.

Deliverable format:

- The repository-facing deliverables are Markdown files under `docs/overview/`
  plus small audit scripts/tests under `scripts/` and `tests/`.
- The `@documents` request was treated as a request for organized repository
  documents, not a polished `.docx`; no Word/Google Docs artifact is currently
  generated.

## Requirement Coverage

| Requirement | Evidence in current tree | Status | Remaining gap |
| --- | --- | --- | --- |
| List all Opus 4.8 attacks/vulnerabilities. | `docs/overview/OPUS48_REVIEW_AUDIT_CHECKLIST.md` has O48-01 through O48-25 with severity, status, assessment, and action. | Covered | Keep updated if new reviewer comments arrive. |
| Resolve issues one by one where possible. | `docs/overview/OPUS48_ISSUE_RESOLUTION_LEDGER.md` maps O48-01 through O48-25 to fixes, evidence, and remaining gates. | Covered at audit/documentation level | Experiment-gated items still require new or recovered evidence. |
| Provide rebuttal-ready response strategy. | `docs/overview/OPUS48_REBUTTAL_RESPONSE_MATRIX.md` maps reviewer attacks to response posture, safe wording, evidence, and forbidden claims. | Covered for current evidence | Update if new experiments or recovered artifacts change a gate. |
| Provide a single advisor-facing Markdown summary. | `docs/overview/OPUS48_REBUTTAL_MASTER_BRIEF.md` compresses the rebuttal decision, defend/scope table, 1B answer, P0 gates, and suggested rebuttal paragraph. | Covered | Keep it synchronized with the audit stack after new artifacts/experiments. |
| Distinguish closed, partly closed, open, and do-not-defend issues. | Same checklist plus `docs/overview/OPUS48_AUDIT_CONTROL_CENTER.md` attack buckets. | Covered | Some scientific gaps remain open by design. |
| Investigate the 1B MLA anomaly from code/provenance rather than assumption. | `docs/overview/OPUS48_FORENSIC_AUDIT_REPORT.md` sections 3 and 10; `docs/overview/EXPERIMENT_CODE_RESULT_AUDIT.md`; `docs/overview/RESULT_PROVENANCE_MANIFEST.md` M4. | Partly covered | Exact 1B JSON/checkpoint/data hashes are still missing from the compact branch. |
| Define how to recover missing external artifacts. | `docs/overview/OPUS48_ARTIFACT_RECOVERY_RUNBOOK.md` gives sanitized external-machine commands, expected fields, and acceptance criteria. | Covered as a runbook | Actual artifacts still need to be recovered. |
| Decide whether 1B is same-config longer training. | `docs/overview/OPUS48_FORENSIC_AUDIT_REPORT.md` section 3 compares 8K/500M vs 4K/1B fields and says no. | Covered for current evidence | A true same-config rerun would be needed for a token-scaling ablation. |
| Audit EVQ/RoPE/YaRN frequency loading. | `eval_extended_3seeds.py` and `yarn_finetune_eval.py` now require checkpoint `inv_freq`, clone it after load, print hash, and apply YaRN to that loaded table; `tests/test_yarn_checkpoint_inv_freq.py` covers helpers. | Covered for code path | Exact checkpoints still need offline `audit_rope_checkpoint.py` runs. |
| Add checkpoint audit script. | `scripts/core_text_phases/audit_rope_checkpoint.py`. | Covered | Must be run on recovered artifacts before promoting old rows. |
| Audit token-count logic. | `docs/overview/OPUS48_FORENSIC_AUDIT_REPORT.md` section 6; `scripts/core_text_phases/audit_training_artifacts.py`; `tests/test_training_artifact_audit.py`. | Covered for code math | Old `results.json` files do not include exact token math. |
| Audit eval protocol distinctions. | `docs/overview/OPUS48_FORENSIC_AUDIT_REPORT.md` section 7; `docs/overview/PAPER_DESCRIPTION_AUDIT.md`; paper/table wording patches for teacher-forced PK and per-document/full-sequence distinctions. | Covered for documented rows | Exact 1B replay JSON still absent. |
| Audit dataset provenance. | `docs/overview/OPUS48_FORENSIC_AUDIT_REPORT.md` section 8; `docs/overview/DATA_PREPARATION.md` audit note. | Partly covered | 1B external tensor hashes/source manifests missing. |
| Audit paper table/result provenance. | `docs/overview/RESULT_PROVENANCE_MANIFEST.md`; `docs/overview/PAPER_CLAIMS_MAP.md`; `docs/overview/EXPERIMENT_CODE_RESULT_AUDIT.md`. | Covered at compact-repo level | Some raw/checkpoint-level provenance is missing. |
| Identify and patch obvious code bugs/inconsistencies. | Eval run-id resolution and checkpoint-loaded `inv_freq` handling patched; tests added. | Covered for found code issues | No claim that all possible historical server-side divergences are fixed. |
| Audit paper descriptions against evidence. | `docs/overview/PAPER_DESCRIPTION_AUDIT.md`; patched `paper/sections/05_experiments.tex`, `paper/appendix/a3_supporting_results.tex`, `paper/appendix/a4_supporting_experiments.tex`, `paper/tables/table3_capability_passkey.tex`, and `paper/REBUTTAL_PLAYBOOK.md`. | Covered for reviewed statements | Continue rechecking if paper prose changes. |
| Organize the docs so future readers do not use stale registry claims. | `docs/overview/OPUS48_AUDIT_CONTROL_CENTER.md`; audit notices in `docs/overview/PROJECT_OVERVIEW.md`, `docs/overview/EXPERIMENT_REGISTRY.md`, `docs/overview/EXPERIMENT_INVENTORY.md`, `docs/overview/METHODOLOGY.md`, and `docs/overview/DATA_PREPARATION.md`. | Covered | None known. |
| Prevent audit-doc and paper-wording regressions. | `tests/test_opus48_audit_docs.py` checks O48 coverage, allowed resolution labels, core markdown links, manifest hashes, and high-risk old phrases in paper/rebuttal. | Covered for structural regressions | Does not prove missing external artifacts. |
| Provide minimal experiment plan. | `docs/overview/OPUS48_FORENSIC_AUDIT_REPORT.md` section 13 and `docs/overview/OPUS48_REVIEW_AUDIT_CHECKLIST.md` action plan. | Covered | Running the plan is outside current no-training audit scope. |

## Resolved by This Audit Pass

- Paper/rebuttal language no longer treats EVQ as universal long-context SOTA.
- PK is scoped as teacher-forced NLL-gap unless explicitly marked AR exact.
- PE-dominant Geo/DAPE/EVQ rows are seed-42-scoped.
- MLA `tau=1.414` is described as empirical `d_eff=128`, distinct from code
  `head_dim=64` and `d_rope=32`.
- MLA caption/prose no longer claims EVQ+YaRN is best at 8K; it says every
  extrapolated length.
- 1B/4K MLA is marked as supporting/limitation, not same-config longer training.
- Historical launch wrappers are quarantined as provenance clues.
- Old overview/registry documents now defer to the Opus 4.8 audit stack.
- Eval scripts now make the checkpoint frequency source explicit for YaRN.

## Still Open

These are not documentation omissions; they require external artifacts or new
experiments.

| Open item | Why it matters | Required evidence |
| --- | --- | --- |
| Exact 1B/4K baseline and YaRN+FT JSONs. | Without them, the 1B row is report-backed but not compact-JSON-backed. | Raw `results.json` / `yarn_ft_s*.json` plus SHA256. |
| 1B and primary MLA checkpoint/data hashes. | Needed to prove schedule labels, data source, and loaded `inv_freq`. | `make_artifact_manifest.py` output plus `audit_rope_checkpoint.py` output. |
| Direct MLA tau ablations. | Opus 4.8 explicitly attacks the empirical `d_eff=128` convention. | `tau=d_rope/sqrt(L)` and code-`head_dim/sqrt(L)` rows under fixed config. |
| Rebased-Geo/fixed-interpolation control. | Addresses novelty/simple-schedule attack. | Training-time control with same endpoints/range budget. |
| Tuned Geo+YaRN/LongRoPE-style controls. | Addresses matched-scale vs tuned-scaler scope. | Clearly scoped baseline results or explicit concession. |
| Geo+LoRA matched control. | Needed before LoRA can support EVQ-specific transfer. | Same long-context LoRA data/budget with Geo substrate. |
| Additional PE-dominant seeds. | Turns seed-42 diagnostic into stronger primary evidence. | Geo/DAPE/EVQ seeds beyond 42, or keep diagnostic-only. |

## Completion Decision

The documentation and code-audit scaffolding is substantially complete for the
current compact repository: every Opus 4.8 issue has a status, evidence pointer,
and reviewer-safe handling rule.

The broader scientific self-rescue is not fully complete because several P0
reviewer attacks still require recovered artifacts or new experiments. Do not
mark the objective as fully achieved until either:

1. those external-artifact/experiment gates are closed, or
2. the final paper/rebuttal explicitly concedes those gaps and removes any claim
   that depends on them.
