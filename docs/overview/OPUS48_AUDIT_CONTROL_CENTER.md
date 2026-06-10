# Opus 4.8 Audit Control Center

Purpose: one-stop navigation for the Opus 4.8 review self-rescue. This file is
the top-level reviewer checklist; the detailed evidence lives in the linked
audit docs. It should be read before older overview/registry documents.

Authoritative audit stack:

| Need | Use this file |
| --- | --- |
| Single advisor-facing rebuttal brief | `docs/overview/OPUS48_REBUTTAL_MASTER_BRIEF.md` |
| Issue-by-issue Opus 4.8 checklist | `docs/overview/OPUS48_REVIEW_AUDIT_CHECKLIST.md` |
| Per-issue resolution ledger | `docs/overview/OPUS48_ISSUE_RESOLUTION_LEDGER.md` |
| Rebuttal-safe response matrix | `docs/overview/OPUS48_REBUTTAL_RESPONSE_MATRIX.md` |
| Requirement-level completion audit | `docs/overview/OPUS48_COMPLETION_AUDIT.md` |
| Original forensic prompt answer | `docs/overview/OPUS48_FORENSIC_AUDIT_REPORT.md` |
| External artifact recovery runbook | `docs/overview/OPUS48_ARTIFACT_RECOVERY_RUNBOOK.md` |
| Code/result/JSON support matrix | `docs/overview/EXPERIMENT_CODE_RESULT_AUDIT.md` |
| Paper wording vs evidence audit | `docs/overview/PAPER_DESCRIPTION_AUDIT.md` |
| Result artifact provenance ledger | `docs/overview/RESULT_PROVENANCE_MANIFEST.md` |
| Historical script quarantine | `docs/overview/HISTORICAL_SCRIPT_STATUS.md` |
| Paper figure/table map | `docs/overview/PAPER_CLAIMS_MAP.md` |

## Current Verdict

Defensible core:

- EVQ-Cosh is a closed-form, zero-parameter training-time frequency allocation.
- The paper can defend RoPE frequency allocation as a finite spectral-budget
  axis complementary to inference-time range scaling.
- The best evidence remains the 454M matched-scale EVQ x YaRN table and the
  8K/500M 3-seed MLA scarce-channel stress test.

Non-defensible as broad claims:

- EVQ is not a YaRN/LongRoPE/LongRoPE2 replacement.
- EVQ+YaRN is not shown to beat tuned Geo+YaRN or every tuned range-scaling
  baseline.
- PK is not autoregressive exact retrieval unless explicitly marked that way.
- The 1B/4K MLA row is not a same-config longer-training ablation and is not
  compact-JSON-backed.
- MLA `tau=1.414` is an empirical `d_eff=128` convention, not a theorem derived
  from released code `head_dim=64` or `d_rope=32`.

## Opus 4.8 Attack Buckets

| Bucket | Main risk | Current status | Action gate |
| --- | --- | --- | --- |
| Novelty/baselines | Missing rebased-Geo/fixed-interpolation and tuned scaler controls. | Open | Concede scope or run a minimal control. |
| Theory/surrogate | Variational result is exact only for a fitted broadband surrogate. | Partly closed | Use "surrogate-derived and functionally validated" wording. |
| Tau rule | `tau=d_eff/sqrt(L)` is a basin/default, not a global optimum. | Partly closed | Do not over-defend; add `L_eff^J` measurement only if available. |
| Passkey metric | PK is teacher-forced NLL-gap. | Closed for wording | Never call Table 2/3 PK AR exact. |
| Primary II | Geo/DAPE/EVQ rows are seed 42 at extreme 128-to-8K extrapolation. | Partly closed | Keep diagnostic and seed-scoped. |
| MLA tau | Strongest MLA row uses empirical `d_eff=128` convention. | Partly closed | Add direct `tau=d_rope/sqrt(L)` and code-`head_dim/sqrt(L)` ablations if possible. |
| 1B anomaly | Raw EVQ reverses in 4K/1B supporting MLA report. | Open limitation | Treat as root-cause target unless exact JSON/checkpoint/data manifests are recovered. |
| LoRA 8B | Missing matched Geo+LoRA control. | Open | Supporting only; do not use as proof. |
| Video/progressive | Useful mechanism checks, but not broad SOTA evidence. | Partly closed | Keep supporting-only and name single-seed/base sensitivity. |
| Provenance | Historical wrappers and compact branch artifacts diverge. | Partly closed | Use sanitized manifests before promoting old runs. |
| Paper wording | Some old docs/rebuttal language overclaimed. | Partly closed | Use `docs/overview/PAPER_DESCRIPTION_AUDIT.md` before editing text. |

## P0 Checklist

- [x] Create an Opus 4.8 issue inventory with severity/status/actions.
- [x] Create a prompt-structured forensic audit report.
- [x] Separate code support from JSON/result support.
- [x] Patch eval scripts so YaRN uses checkpoint-loaded `inv_freq` explicitly.
- [x] Add checkpoint and artifact audit helpers.
- [x] Fix paper wording for PK teacher-forced scope.
- [x] Fix paper wording for MLA `d_eff=128` empirical convention.
- [x] Fix MLA prose/caption that implied EVQ+YaRN was best at 8K.
- [x] Mark 1B/4K as supporting/limitation, not same-config longer training.
- [ ] Recover exact 1B baseline and YaRN+FT JSON/checkpoint/data manifests.
- [ ] Run checkpoint `inv_freq` audit on exact primary MLA and 1B artifacts.
- [ ] Run direct MLA tau ablations or keep them as explicit open gaps.

## 1B Decision Rule

Use this rule in rebuttal and paper edits:

| Evidence state | Allowed treatment |
| --- | --- |
| Markdown report only, no exact JSON/checkpoint/data hashes | limitation/root-cause target only |
| Exact JSON recovered, but no checkpoint/data hashes | supporting negative/diagnostic row only |
| JSON + checkpoint `inv_freq` hashes + data manifest recovered | reviewer-grade supporting row |
| Same-config 8K/500M-to-1B rerun with fixed data/seeds/eval | valid token-scaling ablation |

The current compact branch is in the first state for the 1B/4K row.

## Old Document Quarantine

Older files such as `docs/overview/PROJECT_OVERVIEW.md`,
`docs/overview/EXPERIMENT_REGISTRY.md`, and
`docs/overview/EXPERIMENT_INVENTORY.md` contain useful historical context but
predate the Opus 4.8 audit. They must not override the audit stack above. In
particular:

- "paper-ready" labels from early Hybrid/EVQ registry files are historical.
- old "unique authoritative source" claims are no longer true.
- old broad EVQ+YaRN or tau-formula language should be rechecked against
  `docs/overview/PAPER_DESCRIPTION_AUDIT.md`.
- historical launch scripts are provenance clues until promoted by
  `make_artifact_manifest.py`.

## Commands for New Artifact Promotion

When an external run directory is recovered:

```bash
python scripts/core_text_phases/make_artifact_manifest.py \
  --entry 1b_mla_seed42=<recovered-run-dir> \
  --output <sanitized-manifest.json> \
  --rope-audit \
  --inspect-tensors

python scripts/core_text_phases/audit_training_artifacts.py \
  --work-dir <recovered-work-dir> \
  --dataset fineweb-edu \
  --seq-len 4096 \
  --batch-size 12 \
  --train-tokens 1000000000

python scripts/core_text_phases/audit_rope_checkpoint.py \
  --checkpoint <checkpoint.pt> \
  --rope-dim 32 \
  --base 500000 \
  --tau 1.414
```

Do not paste private absolute paths into public docs. Use sanitized path hints
and SHA256 hashes.
