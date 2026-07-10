# Rebuttal 7 — July 22 rebuttal command center

This directory is the single entry point for the July 22 rebuttal campaign. It combines two independent committee simulations, the portable evidence ledger, and the no-new-experiment response strategy.

## Authoritative review inputs

- `neurips_2026_review_committee_output.md` (Fable5): SHA-256 `520ff82bb04c4d552f1d36a5573ef613fc17d9bb15838b780c8f16de1d751864`; 37,871 bytes; 224 lines.
- `REVIEW_COMMITTEE_FULL_V2.md` (GPT Pro): SHA-256 `07fac44c2080bf0a3440cf092596c0656ee1c2246246e486e5fb3e7ed4b9d942`; 36,665 bytes; 308 lines.

Both review packets are preserved verbatim. Response documents may summarize them but must not silently alter reviewer wording.

## Working documents

- `FABLE5_RESPONSE_AND_FIX_LEDGER.md`: all 18 questions, dispositions, completed fixes, evidence boundaries, and concise draft responses.
- `GPT_PRO_RESPONSE_CROSSWALK.md`: all 17 GPT Pro questions mapped to tracked evidence, safe concessions, and claim boundaries.
- `IGNORED_ASSET_RECONCILIATION.md`: complete ignored-asset inventory, portable evidence tiers, all 18 response updates, experiment order, and company-computer handoff.
- `EXPERIMENT_PRIORITY_PLAN.md`: server-time queue, protocols, decision rules, and the role of the proposed 7B/8B fine-tuning work.
- `NO_SERVER_FIX_LOG.md`: code, paper, provenance, and presentation fixes completed without new experiments.
- `BRANCH_AND_EVIDENCE_FINAL_AUDIT.md`: final branch-ancestry, ignored-result, and battle-readiness decision record.
- `rebuttal/REBUTTAL_RESPONSE_DRAFT.md`: formal response prose for the four highest-impact questions (shape--scale, tuned base, AR exact match, and QuALITY).

## July 22 battle spine

1. Open with the recovered AR exact-match separation: at 8K, Geo+YaRN has 61.3% teacher-forced retrieval but 0.0% AR exact across every seed; EVQ+YaRN reaches 58.0% mean AR exact.
2. Concede and fix the QuALITY provenance error: only the full `n=2086` Gold-NLL result remains in the claim chain.
3. Separate theory layers: cosh is conditional on the variational surrogate; `tau` is a calibrated operating/basin rule.
4. Defend the intended high-base, finite-channel regime while retaining the reported small-base negative boundary.
5. Close with the zero-learned-parameter and zero-new-inference-operation deployment story, without claiming FLOP savings or downstream SOTA.

The files listed above are the authoritative July packet. Older response notes under the repository-level `rebuttal/` directory are archival unless this README or the current ledgers link them explicitly.

## Current readiness

`draft_with_placeholders`. The package is sufficient to conduct the rebuttal from current evidence: all 35 simulated questions are indexed, the highest-value recovered result is raw-backed, and presentation/provenance corrections are explicit. It is not a claim that every reviewer-requested control is complete. New training, missing matched controls, and checkpoint-derived mechanism measurements remain placeholders; no missing result is represented as completed.
