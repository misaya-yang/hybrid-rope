# EVQ-Cosh Overview Docs

Start here if you are reviewing the paper, preparing rebuttal text, or checking
whether an experiment row is supported by code and artifacts.

## Current Authority Order

1. `ai-handoff.md` for current worktree state and known breakage.
2. `rebuttal/README.md` and `rebuttal/rebuttal_playbook.md` for the current response-only workflow.
3. `docs/overview/RESULT_PROVENANCE_MANIFEST.md` for reviewer-safe result identity and hashes.
4. `docs/overview/PAPER_CLAIMS_MAP.md`, `docs/overview/REPRODUCE.md`, and `docs/overview/DATA_PREPARATION.md` for navigation and reproduction.

The Opus 4.8 files below are a supporting audit stack. They do not override the 2026-07-11/12 rebuttal theory, provenance, or control-room documents.

## Supporting Opus 4.8 Audit Stack

Read these first for reviewer-facing decisions:

| Question | File |
| --- | --- |
| What single Markdown should I send to an advisor? | `docs/overview/OPUS48_REBUTTAL_MASTER_BRIEF.md` |
| What is the current reviewer-safe position? | `docs/overview/OPUS48_AUDIT_CONTROL_CENTER.md` |
| What did Opus 4.8 attack? | `docs/overview/OPUS48_REVIEW_AUDIT_CHECKLIST.md` |
| How was each issue handled? | `docs/overview/OPUS48_ISSUE_RESOLUTION_LEDGER.md` |
| How should rebuttal answer each attack? | `docs/overview/OPUS48_REBUTTAL_RESPONSE_MATRIX.md` |
| Which requirements are covered and which remain open? | `docs/overview/OPUS48_COMPLETION_AUDIT.md` |
| What is the detailed 1B/MLA forensic audit? | `docs/overview/OPUS48_FORENSIC_AUDIT_REPORT.md` |
| How do we recover external artifacts to close P0 gates? | `docs/overview/OPUS48_ARTIFACT_RECOVERY_RUNBOOK.md` |
| Does a row have code, implementation, and JSON/result support? | `docs/overview/EXPERIMENT_CODE_RESULT_AUDIT.md` |
| Does paper wording stay within evidence? | `docs/overview/PAPER_DESCRIPTION_AUDIT.md` |
| Which files/hashes support current claims? | `docs/overview/RESULT_PROVENANCE_MANIFEST.md` |

## Fast Rules

- Table 2 PK means teacher-forced NLL-gap retrieval unless a row explicitly says
  autoregressive exact match.
- The 1B/4K MLA row is code-backed and report-backed, but not compact-JSON-backed.
- The 1B/4K MLA row is not a same-config longer-training ablation of the 8K/500M
  primary MLA result.
- MLA `tau=1.414` is an empirical `d_eff=128` operating convention, distinct
  from released code fields `head_dim=64` and `d_rope=32`.
- EVQ x YaRN is a matched-scale substrate/range complementarity claim, not a
  tuned-scaler SOTA claim.
- LoRA, video, progressive, and 750M rows are supporting unless explicitly
  re-audited and promoted.

## Reproduction And Data

| Need | File |
| --- | --- |
| Figure/table/script map | `docs/overview/PAPER_CLAIMS_MAP.md` |
| Reviewer reproduction paths | `docs/overview/REPRODUCE.md` |
| Data source notes and caveats | `docs/overview/DATA_PREPARATION.md` |
| Terms and metric definitions | `docs/overview/TERMS_AND_PROTOCOLS.md` |
| Current vs historical scripts | `docs/overview/HISTORICAL_SCRIPT_STATUS.md` |

## Historical Docs

The following files are useful for background, but they predate the Opus 4.8
audit stack and must not override it:

- `docs/overview/PROJECT_OVERVIEW.md`
- `docs/overview/EXPERIMENT_REGISTRY.md`
- `docs/overview/EXPERIMENT_INVENTORY.md`
- `docs/overview/METHODOLOGY.md`

If these disagree with the Opus 4.8 audit stack, use the audit stack.
