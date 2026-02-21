# Server Evidence Sync Record

Updated: `2026-02-21`

## 1. Objective

Pull back all relevant experiment evidence from server to local repository while excluding weights/checkpoints.

## 2. Local Destination

- Snapshot root:
  - `server_artifacts_2026-02-21/`
- Synced content groups:
  - `server_artifacts_2026-02-21/results/`
  - `server_artifacts_2026-02-21/logs/`
  - `server_artifacts_2026-02-21/sigmoid_rope_experiments/data/`
  - `server_artifacts_2026-02-21/sigmoid_rope_experiments/results/`
  - `server_artifacts_2026-02-21/sigmoid_rope_experiments/*.log`

## 3. Exclusion Policy

The sync excludes common weight/checkpoint artifacts:

- `*.safetensors`, `*.pt`, `*.bin`, `*.pth`
- `optimizer*`, `pytorch_model*`, `training_args.bin`
- `**/checkpoint-*`, `**/checkpoints/*`
- `**/_weights_quarantine/*`

## 4. Script Used

- `tools/sync_server_evidence_data_only.ps1`

Usage:

```powershell
powershell -ExecutionPolicy Bypass -File tools/sync_server_evidence_data_only.ps1 `
  -RemoteHost connect.bjb1.seetacloud.com `
  -RemotePort 42581 `
  -RemoteUser root `
  -RemotePassword "<PASSWORD>" `
  -RemoteRepoRoot "/root/autodl-tmp/dfrope/hybrid-rope" `
  -LocalRepoRoot "." `
  -LocalTargetRel "server_artifacts_2026-02-21"
```

## 5. Notes

- If an experiment is still running, rerun the sync to refresh latest logs and csv metrics.
- This snapshot is intended as evidence mirror, not a replacement for curated summaries in `results/advisor_package_2026-02-15/`.
