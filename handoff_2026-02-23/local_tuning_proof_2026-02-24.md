# Local Tuning Proof (2026-02-24)

## Objective
On local machine (`aidemo`), tune anchored-sigmoid schedule parameters to better align with theory band (`rho_diag` to `rho_cosh`) across long contexts, without additional expensive training.

## Method

- Script:
  - `scripts/import_2024/tune_anchored_sigmoid_to_theory.py`
- Search space:
  - `anchor_factor in [2, 30]`
  - `slope_raw in [4, 40]`
  - `center_ratio in [0.30, 0.70]`
- Contexts:
  - `16K, 32K, 64K` (joint objective)
- Metrics:
  - RMSE to mid-theory target
  - in-band coverage
  - monotonicity penalty

## Main result

Stable parameter region found:
- `center_ratio = 0.70` (consistent under loose/mid/strict settings)
- `slope_raw ~= 20`
- `anchor_factor ~= 3..5`

Recommended practical setting:
- `anchor_factor=4`, `slope_raw=20`, `center_ratio=0.70`

## Next-Day execution note (2026-02-25)

- Use the tuned setting above as the only schedule change in the next protocol-locked rerun.
- Keep checkpoint/tokenizer/manifest/decode/seed unchanged.
- Run order: `E2` first (shape signal), then `E1` full table refresh, then paired significance.
- Detailed commands: `handoff_2026-02-23/tomorrow_tuned_param_runbook_2026-02-25.md`.

## Quantitative gains vs current anchored_sigmoid

Strict setting (`band_tol=0.05`) over `16K/32K/64K`:
- loss improvement: about `50%+` on all three contexts
- coverage gain:
  - `16K: 0.110 -> 0.612`
  - `32K: 0.110 -> 0.612`
  - `64K: 0.110 -> 0.590`

Mid setting (`band_tol=0.10`) over `16K/32K/64K`:
- loss improvement: about `59%~61%`
- coverage gain:
  - `16K: 0.236 -> 0.812`
  - `32K: 0.236 -> 0.812`
  - `64K: 0.236 -> 0.782`

## Artifacts

- `artifacts/reviewer_2026-02-24/tuning_loose_16k_32k_64k/tuned_params.json`
- `artifacts/reviewer_2026-02-24/tuning_mid_16k_32k_64k/tuned_params.json`
- `artifacts/reviewer_2026-02-24/tuning_strict_16k_32k_64k/tuned_params.json`
- `artifacts/reviewer_2026-02-24/tuning/sensitivity_summary.md`
- `artifacts/reviewer_2026-02-24/tuning/summary_strict.md`
