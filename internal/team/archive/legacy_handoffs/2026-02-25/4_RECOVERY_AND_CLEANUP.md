# Recovery & Cleanup (Quarantine-First)

Last updated: 2026-02-25

This document defines the safe cleanup procedure for the server (disk pressure) without breaking reproducibility.

## 1) Quarantine policy (hard)

- First pass: **move only**, do not delete.
- Retention: 7 days.
- A move must be recorded in `docs/exp/SERVER_CLEANUP_MANIFEST.md` with:
  - original path
  - quarantine path
  - size
  - what evidence remains in-place
  - restore command

## 2) Quarantine root

Quarantine root (writable on this server):
- `/root/autodl-tmp/dfrope/trash/hybrid-rope/2026-02-25/`

Note:
- `/autodl-pub/data` is mounted read-only for `trash/` on this server, so we quarantine under `/root/autodl-tmp/dfrope/trash/` instead.

## 3) Restore

Restore is always:
```bash
mv <quarantine_path> <original_path>
```

No renames, no edits, no tar required.

## 4) What is safe to move first

Move only `checkpoint-*` folders when final artifacts already exist:

- `results/train_freq_comparison/700m_orig_*/checkpoint-*`
  - keep: `model/`, `results.json`
- `results/smoke_train_local*/run_*/checkpoint-*`
  - keep: `final_model/`, `results.json`, `training.log`

If unsure, do not move it.

## 5) Second pass deletion (manual confirmation only)

After 7 days, we can permanently delete quarantined items **only if**:
- the corresponding paper exports exist (`paper_exports/...`)
- registry/inventory references still resolve without the checkpoints
- there is no open issue requiring the raw checkpoint
