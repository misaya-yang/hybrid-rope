# Opus 4.8 Total-Attack Dossier

Purpose: one-page operating brief for rebuttal preparation. This is the file to
read when deciding what to defend, what to scope down, and what evidence must be
recovered before a claim can be promoted.

This dossier does not create new results. It compresses the current audit stack:

- `docs/overview/OPUS48_AUDIT_CONTROL_CENTER.md`
- `docs/overview/OPUS48_REVIEW_AUDIT_CHECKLIST.md`
- `docs/overview/OPUS48_ISSUE_RESOLUTION_LEDGER.md`
- `docs/overview/EXPERIMENT_CODE_RESULT_AUDIT.md`
- `docs/overview/RESULT_PROVENANCE_MANIFEST.md`
- `docs/overview/PAPER_DESCRIPTION_AUDIT.md`
- `docs/overview/OPUS48_ARTIFACT_RECOVERY_RUNBOOK.md`

## Current Decision

The document layer is ready for rebuttal use. The scientific self-rescue is not
fully closed because several reviewer attacks require recovered artifacts or new
experiments.

Allowed high-level claim:

> EVQ-Cosh identifies training-time RoPE frequency allocation as a finite
> spectral-budget design axis. The strongest current evidence supports a
> mechanism claim: EVQ changes the trained frequency substrate on which matched
> inference-time scaling acts.

Do not upgrade this into a universal long-context, tuned-scaler, or production
deployment claim.

## Defend

| Claim to defend | Evidence | Rebuttal posture |
| --- | --- | --- |
| EVQ is a closed-form, zero-learned-parameter training-time allocation. | `scripts/lib/rope/schedules.py`; paper theory; RoPE tests. | Defend as a mechanism and implementation claim. |
| EVQ x YaRN Table 2 is a matched-scale substrate/range composition result. | `data/curated/table2_evq_yarn_454m_passkey_10pct.json`; `paper/tables/table2_evq_yarn_main.tex`. | Defend only under fixed matched YaRN scale; PK is teacher-forced NLL-gap. |
| Primary MLA 8K/500M is the strongest systems stress test. | `data/curated/table18_mla_3seed_aggregate.json`; MLA eval scripts; original per-seed/checkpoints unavailable. | Defend only the printed 3-seed aggregate with empirical `d_eff=128` convention. |
| PE-dominant Table 4 is a diagnostic. | `data/curated/fig3_extreme_128.json`; `paper/tables/table4_pe_dominant.tex`. | Defend as seed-scoped PE isolation, not broad learned-PE dominance. |
| YaRN eval frequency source is explicit in current code. | `eval_extended_3seeds.py`; `yarn_finetune_eval.py`; `tests/test_yarn_checkpoint_inv_freq.py`. | Defend current code path; still audit exact historical checkpoints if recovered. |

## Concede Or Scope

| Attack | Correct handling |
| --- | --- |
| 1B/4K MLA raw EVQ reversal | Treat as real limitation and root-cause target. It is code-backed/report-backed, not compact-JSON-backed. |
| 1B as longer-training ablation | Do not defend. It is not same-config: train length, data/provenance, seed coverage, and artifact completeness differ. |
| Direct MLA tau convention attack | Scope. `tau=1.414` is empirical `d_eff=128`; direct `tau=d_rope/sqrt(L)` and code-`head_dim/sqrt(L)` ablations remain open. |
| Tuned Geo+YaRN / LongRoPE-style baselines | Concede missing tuned-scaler dominance evidence. Table 2 is matched-scale only. |
| LoRA 8B transfer | Supporting only until matched Geo+LoRA control exists. |
| Video/progressive/750M | Supporting only; do not use as primary durability or production-scale proof. |
| PE-dominant seeds | Geo/DAPE/EVQ rows are seed 42; add seeds before stronger statistical wording. |

## Never Say

- EVQ replaces YaRN, LongRoPE, LongRoPE2, DAPE, FIRE, or learned PE.
- EVQ+YaRN beats every tuned Geo+YaRN or range-scaling baseline.
- Table 2 PK is autoregressive exact match.
- The 1B row proves training durability.
- The 1B row is a same-config token-scaling continuation of the 8K/500M MLA row.
- MLA `tau=1.414` is theoretically forced by released code `head_dim=64` or
  `d_rope=32`.
- Supporting LoRA/video/progressive rows are primary evidence.

## 1B Answer

Short answer:

> The 1B/4K MLA row is not being hidden; it is the main limitation. Current repo
> evidence says it is code-backed and report-backed, but exact raw JSON,
> checkpoint `inv_freq` hashes, and data hashes are missing from the compact
> branch. Because it also differs from the primary MLA setup, it is not a
> same-config longer-training ablation. We therefore treat it as a root-cause
> target and do not use it as primary support.

Evidence state:

| Layer | State |
| --- | --- |
| Training/eval code | Exists; current eval scripts now enforce checkpoint-loaded `inv_freq`. |
| Markdown report | Unavailable in the current checkout; only summary-level references survive. |
| Exact compact JSON | Missing. |
| Checkpoint/data hashes | Missing. |
| Same-config token scaling | Not established. |

Promotion rule:

- No exact JSON: limitation only.
- JSON but no checkpoint/data hash: supporting diagnostic only.
- JSON plus checkpoint/data manifests: reviewer-grade supporting row.
- Same-config rerun: valid token-scaling ablation.

## P0 Gates

| Gate | Close with |
| --- | --- |
| 1B exact JSON/provenance | Run `docs/overview/OPUS48_ARTIFACT_RECOVERY_RUNBOOK.md` on recovered external artifacts. |
| Primary MLA checkpoint provenance | Run `scripts/core_text_phases/audit_rope_checkpoint.py` on exact checkpoints. |
| Token-count/data provenance | Run `scripts/core_text_phases/audit_training_artifacts.py` and `make_artifact_manifest.py`. |
| MLA tau convention | Add `tau=d_rope/sqrt(L)` and code-`head_dim/sqrt(L)` ablations, or keep explicit concession. |
| Simple-schedule/tuned-scaler baseline | Add rebased-Geo / tuned Geo+YaRN controls, or keep baseline gap explicit. |

## Rebuttal Order

1. Lead with the narrow mechanism claim.
2. Cite Table 2 as matched-scale EVQ x YaRN, not tuned-scaler dominance.
3. Cite 8K/500M MLA as the strongest 3-seed systems stress test.
4. Treat PE-dominant Table 4 as diagnostic and seed-scoped.
5. Acknowledge 1B/4K as the limitation and explain why it is not same-config.
6. Do not cite supporting LoRA/video/progressive rows unless the question asks
   about scope or future work.

## If Only One More Thing Can Be Done

Recover the 1B and primary MLA artifact manifests first. Without exact JSON,
checkpoint `inv_freq` hashes, and data hashes, the paper can still be defended
as a scoped mechanism study, but the 1B objection cannot be fully closed.
