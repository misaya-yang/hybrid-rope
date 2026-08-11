# EVQ-Cosh Overview Docs

Start here if you are reviewing the paper, preparing rebuttal text, or checking
whether an experiment row is supported by code and artifacts.

## Current Authority Order

1. `rebuttal/rebuttal_0723/00_REVIEWER_SCORES_AND_AC_METAREVIEW.md` for
   retained reviewer and AC concerns.
2. `rebuttal/rebuttal_0723/01_REBUTTAL_PLAYBOOK.md` for current response and
   evidence routing.
3. `rebuttal/rebuttal_0723/theory_results/EVQ_COSH_REBUTTAL_PRINCIPLES.md`
   for method and theory boundaries.
4. `docs/overview/RESULT_PROVENANCE_MANIFEST.md` for reviewer-safe result
   identity and hashes.
5. `docs/overview/PAPER_CLAIMS_MAP.md`, `docs/overview/REPRODUCE.md`, and
   `docs/overview/DATA_PREPARATION.md` for navigation and reproduction.

## Reviewer-Facing Checks

| Question | File |
| --- | --- |
| What did reviewers and the AC actually ask? | `rebuttal/rebuttal_0723/00_REVIEWER_SCORES_AND_AC_METAREVIEW.md` |
| What evidence and wording answer each concern? | `rebuttal/rebuttal_0723/01_REBUTTAL_PLAYBOOK.md` |
| What is ready, conditional, negative, or pending? | `rebuttal/rebuttal_0723/theory_results/REVIEWER_USABLE_EVIDENCE_LEDGER_20260726.md` |
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

The following files are useful for background, but they predate the current
reviewer/AC routing and must not override it:

- `docs/overview/PROJECT_OVERVIEW.md`
- `docs/overview/EXPERIMENT_REGISTRY.md`
- `docs/overview/EXPERIMENT_INVENTORY.md`
- `docs/overview/METHODOLOGY.md`

If these disagree with the current rebuttal playbook, evidence ledgers, or
provenance manifest, use the current sources.
