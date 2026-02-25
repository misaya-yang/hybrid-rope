# Server Cleanup Manifest (Quarantine-First)

Last updated: 2026-02-25

Remote server:
- Host: `connect.bjb1.seetacloud.com`
- Port: `52592`
- Repo: `/root/autodl-tmp/dfrope/hybrid-rope`

Goal:
- Free disk safely without breaking reproducibility.
- **No `rm` in the first pass**: only move to quarantine with a 7-day retention window.

Policy (hard):
1. Generate this manifest before moving anything.
2. Never move paths that are currently referenced by a running PID.
3. Keep “paper-ready” evidence files (`results.json`, `summary.json`, `report.md`, plots) in-place.
4. Only move heavy `checkpoint-*` directories when a corresponding `final_model/` or `model/` exists.

## Quarantine Root

Quarantine root (writable on this server):
- `/root/autodl-tmp/dfrope/trash/hybrid-rope/2026-02-25/`

Note:
- `/autodl-pub/data` is mounted read-only for `trash/` on this server, so we quarantine under `/root/autodl-tmp/dfrope/trash/` instead.

Restore rule:
- Move back to the original path (same directory name). No transformations required.

## Disk Snapshot (before)

Fill with server `df -h` and top `du -sh` outputs.

- `df -h` (trimmed):
  - `AutoFS:fs1 10T used 4.4T (44%) mounted on /autodl-pub/data`
  - `/dev/sda2 879G used 17G (3%)`

- top `du -sh` (paths under repo):
  - before (2026-02-25 18:31 CST): `ms_models=86G`, `hybrid-rope/results=30G`, `hybrid-rope/artifacts=1.3G`
  - after  (2026-02-25 18:35 CST): `hybrid-rope/results=11G`, `trash/hybrid-rope/2026-02-25=20G`

## Candidate Large Checkpoints

These are the first-line candidates (move checkpoints, keep final outputs):

1. `results/train_freq_comparison/700m_orig_*/checkpoint-*`
   - keep: `model/`, `results.json`
2. `results/smoke_train_local*/run_*/checkpoint-*`
   - keep: `final_model/`, `results.json`, `training.log`

Additional candidates (>1G) must be listed explicitly in the table below before any move.

## Move Log (quarantine actions)

| Timestamp | Original Path | Quarantine Path | Size | Keep-in-place evidence | Restore Command |
|---|---|---|---:|---|---|
| `2026-02-25 18:35 CST` | `results/train_freq_comparison/700m_orig_20260214_140024/checkpoint-800` | `/root/autodl-tmp/dfrope/trash/hybrid-rope/2026-02-25/results/train_freq_comparison/700m_orig_20260214_140024/checkpoint-800` | `6.5G` | keep `model/`, `results.json` in-place | `mv /root/autodl-tmp/dfrope/trash/hybrid-rope/2026-02-25/results/train_freq_comparison/700m_orig_20260214_140024/checkpoint-800 /root/autodl-tmp/dfrope/hybrid-rope/results/train_freq_comparison/700m_orig_20260214_140024/` |
| `2026-02-25 18:35 CST` | `results/smoke_train_local/run_20260214_210643/checkpoint-20` | `/root/autodl-tmp/dfrope/trash/hybrid-rope/2026-02-25/results/smoke_train_local/run_20260214_210643/checkpoint-20` | `6.5G` | keep `final_model/`, `results.json`, `training.log` in-place | `mv /root/autodl-tmp/dfrope/trash/hybrid-rope/2026-02-25/results/smoke_train_local/run_20260214_210643/checkpoint-20 /root/autodl-tmp/dfrope/hybrid-rope/results/smoke_train_local/run_20260214_210643/` |
| `2026-02-25 18:35 CST` | `results/smoke_train_local_v2/run_20260214_211041/checkpoint-20` | `/root/autodl-tmp/dfrope/trash/hybrid-rope/2026-02-25/results/smoke_train_local_v2/run_20260214_211041/checkpoint-20` | `6.5G` | keep `final_model/`, `results.json`, `training.log` in-place | `mv /root/autodl-tmp/dfrope/trash/hybrid-rope/2026-02-25/results/smoke_train_local_v2/run_20260214_211041/checkpoint-20 /root/autodl-tmp/dfrope/hybrid-rope/results/smoke_train_local_v2/run_20260214_211041/` |

## Notes / Exceptions

- If a run is still needed for future debugging, keep checkpoints until the final paper-exported plots/tables exist.
- If a directory contains mixed important/temporary files, prefer moving only `checkpoint-*` rather than the whole run directory.
