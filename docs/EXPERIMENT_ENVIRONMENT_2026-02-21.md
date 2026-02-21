# Experiment Environment

Updated: `2026-02-21`

## 1. Local Workspace

- Path: `e:/rope/hybrid-rope`
- OS shell used for orchestration: PowerShell
- SSH tools:
  - `C:\Users\Admin\.ssh\plink.exe`
  - `C:\Users\Admin\.ssh\pscp.exe`

## 2. Remote Primary Server

- Repo path: `/root/autodl-tmp/dfrope/hybrid-rope`
- Main experiment host uses:
  - Python: `3.12.3`
  - PyTorch: `2.8.0+cu128`
  - GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition` (96GB class)

## 3. Known Operational Constraints

- External HuggingFace access may be blocked/unreliable.
- Prefer local model/data paths already downloaded on server.
- Commit policy: no model weights or checkpoints.
- Keep reproducible evidence:
  - `json/csv/log/pdf/png`

## 4. Phase4 Runtime Configuration (Current)

- Script:
  - `sigmoid_rope_experiments/run_phase4.py`
- Active launch arguments:
  - `--passkey_repeats 5`
  - `--passkey_lengths 1024,2048,4096,8192,16384`
- Stabilization fixes already applied:
  - fixed RoPE method name conflict (`apply_rotary`)
  - tokenizer vocab-size safety (`len(tokenizer)` and token range guard)
  - dual-model memory safety (`micro_batch` halving after autotune)

## 5. Evidence Sync Strategy

- Raw server evidence is mirrored to:
  - `server_artifacts_2026-02-21/`
- Sync script:
  - `tools/sync_server_evidence_data_only.ps1`
