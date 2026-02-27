# Handoff Packages (Dated)

This repository uses **dated handoff packages** for experiment operations and cross-machine continuity.

## Where To Start

- One-glance ops card: `AI_HANDOFF.md`
- Latest handoff package: `handoff/2026-02-26/0_README.md`

## Directory Convention

All handoff packages live under:

- `handoff/YYYY-MM-DD/`

Each package should contain (minimum):

- `0_README.md` (entrypoint + 3-command reproduction)
- `1_PROTOCOL_LOCK.md` (templates/decoding/truncation/manifest invariants)
- `2_ASSET_MAP.md` (base model + LoRA + inv_freq tensor triplets)
- `3_RUNBOOK.md` (commands for train/eval/stats/export)
- `README.md` (package overview)

Optional but recommended:

- `01_IMPLEMENTED_SCOPE.md`
- `02_VALIDATION_SNAPSHOT.md`
- `03_DEEP_REVIEW_FINDINGS.md`
- `4_RECOVERY_AND_CLEANUP.md`

## Update Policy

- Do **not** rewrite old packages. Add a new dated folder when decisions change.
- Any paper-facing claim must have a traceable run manifest (code hash, config, data hash, inv_freq hash).

