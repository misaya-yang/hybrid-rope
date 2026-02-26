# Project File Manager (NeurIPS Ops)

Last updated: 2026-02-26

## 1) One-screen map

- `train.py`: attention-integrated LLaMA-3-8B training entrypoint.
- `scripts/isolated/attn/`: all isolated new experiments (safe to iterate).
- `docs/exp/`: experiment plans, inventories, and audit reports.
- `docs/exp/reports/`: dated human-readable reports.
- `artifacts/reviewer_YYYY-MM-DD/`: machine-readable snapshots and hash manifests.
- `results/` and `paper_exports/`: paper-facing compact outputs only.

## 2) Keep root clean

Allowed at repo root:
- project-level docs (`README.md`, `AI_HANDOFF.md`)
- global config (`.gitignore`)
- core canonical entry (`train.py`)
- top-level project folders

Not allowed at repo root:
- temporary logs (`*.log`, `*.txt`)
- ad-hoc audit outputs
- one-off scripts with date suffixes

## 3) Where to place new files

- New isolated training/eval/audit scripts:
  - `scripts/isolated/attn/`
- New experiment decision docs:
  - `docs/exp/`
- New run-time report markdown:
  - `docs/exp/reports/`
- JSON/CSV snapshots for analysis:
  - `artifacts/reviewer_YYYY-MM-DD/`

## 4) Evidence contract (paper-usable)

Any claim is citable only if these exist:
- run config
- model/method lock
- per-sample traces (for significance)
- seed
- script hash manifest

Missing one item means `PENDING` or `INVALID` in `docs/exp/EXPERIMENT_INVENTORY.md`.

## 5) Cleanup policy

- Always `quarantine-first`, no direct delete.
- Quarantine location on server:
  - `/root/autodl-tmp/dfrope/trash/hybrid-rope/<date>/`
- Retention:
  - keep 7 days, then delete after second confirmation.

## 6) Fast operator commands

```bash
# status + latest inventory
git status -sb
sed -n '1,120p' docs/exp/EXPERIMENT_INVENTORY.md

# run isolated queue
bash scripts/isolated/attn/next_attn_lora_queue.sh

# verify hash pack
cat artifacts/reviewer_2026-02-26/attn_hashes_2026-02-26.txt
```
