# Sigmoid-RoPE Experiments

This subproject contains the phase-based workflow for sigmoid frequency allocation research.

## Canonical Layout

```text
sigmoid_rope_experiments/
  src/                        # shared core modules
  experiments/                # phase-1 style experiment modules
  data/                       # csv/json outputs
  results/                    # figures (pdf/png)
  archive_server_snapshots/   # legacy one-off server pulls
  run_all.py                  # phase-1 bundle
  run_phase2.py               # fine search + formula refit
  run_phase3.py               # model selection + robustness + passkey debug
  run_phase4.py               # training-time validation
```

## Entrypoints

- Phase 1 baseline:
  - `python run_all.py`
- Phase 2:
  - `python run_phase2.py`
- Phase 3:
  - `python run_phase3.py`
- Phase 4:
  - `python run_phase4.py`

## Data Ownership

- Keep all outputs in `data/` and `results/`.
- Do not commit weights/checkpoints.
- Use top-level server mirror for cross-machine evidence:
  - `../server_artifacts_2026-02-21/`
