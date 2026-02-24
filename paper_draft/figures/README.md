# Figure Assets for Paper Draft

## Figure 3 (main text)
- File: `figure3_theory_warning.pdf` (`.png` for preview)
- Source script: `scripts/import_2024/theory_experiment_plot.py`
- Main command:
  - `python scripts/import_2024/theory_experiment_plot.py --gamma 1.03 --base 10000 --L 16384`
- Real frequency tensors:
  - Loaded from `scripts/import_2024/real_inv_freq_20260223.json`
  - Includes `sigmoid` and `anchored_sigmoid` tensors exported from the 2026-02-23 8B fine-tuned checkpoints.

## Intended narrative
- Left panel: theoretical `rho(phi)` band vs actual warps.
- Right panel: corrected increasing `E_diag(phi)` curve.
- Use this as a warning figure: overly aggressive warps are flagged by deviation from the bounded-amplitude/theoretical-safe region.
