# Experiment Environment Snapshot (ASCII-safe)

Last updated: 2026-02-21
Purpose: provide a clear and reproducible environment snapshot for the next AI.

## 1) Repository and Working Paths

- Local repo root: `e:\rope\hybrid-rope`
- Remote repo root: `/root/autodl-tmp/dfrope/hybrid-rope`
- Main experiment subdir: `/root/autodl-tmp/dfrope/hybrid-rope/sigmoid_rope_experiments`

## 2) Remote Server (actual run machine)

- Hostname: `autodl-container-q70yg9yhtt-d2becc30`
- OS: `Linux 5.15.0-78-generic (Ubuntu)`
- Python: `3.12.3`
- GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition`
- VRAM: `97887 MiB` (about 96 GB)
- Driver: `590.44.01`

From run log (`run_phase4_corrected_live.log`):
- Torch: `2.8.0+cu128`
- CUDA available: `True`
- CUDA device count: `1`
- Device name: `NVIDIA RTX PRO 6000 Blackwell Server Edition`

## 3) Model/Data Locations Used in This Run

- Local model root on server: `/root/autodl-tmp/dfrope/ms_models`
- Local dataset root on server: `/root/autodl-tmp/dfrope/ms_datasets`
- Tokenizer used by phase4 corrected run:
  - `hf:/root/autodl-tmp/dfrope/ms_models/EleutherAI/gpt-neox-20b`
  - vocab size: `50277`
- Training data mode in this completed run:
  - dataset name in log: `LongBench-local`
  - token count in log: `24000000`

## 4) Network and Access Constraints

- HF external access may be restricted in this environment.
- Experiments rely on locally available model/data folders above.
- Remote access was done via `plink/pscp` from Windows.

## 5) Main Script and Command Used

- Script: `sigmoid_rope_experiments/run_phase4_corrected.py`
- Command used:

```bash
cd /root/autodl-tmp/dfrope/hybrid-rope/sigmoid_rope_experiments
/root/miniconda3/bin/python -u run_phase4_corrected.py --include_alpha_star
```

## 6) Completed Status for This Run

- Process status now: finished (no `run_phase4_corrected.py` process running).
- Main log:
  - `/root/autodl-tmp/dfrope/hybrid-rope/sigmoid_rope_experiments/run_phase4_corrected_live.log`
- Final output timestamp group:
  - around `2026-02-21 23:37` for summary CSV/JSON.

## 7) Key Output Files (must exist)

- `data/phase4_corrected_summary.json`
- `data/ppl_vs_length.csv`
- `data/passkey_fixed_results.csv`
- `data/positional_loss.csv`
- `results/training_curves_all.pdf`
- `results/ppl_vs_length.pdf`
- `results/passkey_fixed.pdf`
- `results/positional_loss.pdf`
- `results/freq_comparison_trained.pdf`

Absolute examples:
- `/root/autodl-tmp/dfrope/hybrid-rope/sigmoid_rope_experiments/data/phase4_corrected_summary.json`
- `/root/autodl-tmp/dfrope/hybrid-rope/sigmoid_rope_experiments/data/ppl_vs_length.csv`
- `/root/autodl-tmp/dfrope/hybrid-rope/sigmoid_rope_experiments/data/passkey_fixed_results.csv`
- `/root/autodl-tmp/dfrope/hybrid-rope/sigmoid_rope_experiments/data/positional_loss.csv`

## 8) Important Reproducibility Notes

- Four variants were run under the same training framework:
  - `standard`, `sigmoid`, `anchored20`, `anchored_alpha`
- Shared settings in corrected run:
  - same seed family, same initialization baseline, same data source, same optimizer/scheduler family.
- Known issue to audit next:
  - Passkey remains 0% for all variants in this run despite protocol fix.
  - Do not treat low PPL alone as final success.

## 9) Local Machine Snapshot (for quick compute tasks)

- Local OS: Windows (PowerShell workflow)
- Local GPU (user-provided): RTX 4070 Super
- Recommended local usage:
  - pure compute diagnostics, curve fitting, phase-collision scans, plotting.

## 10) Next AI startup checklist

1. Read this file first.
2. Read `AI_HANDOFF_NEXT_AI_2026-02-21.md` for full context and task plan.
3. Verify file timestamps and hash/size of summary outputs.
4. Re-run only diagnostics first (do not start new expensive training immediately).

