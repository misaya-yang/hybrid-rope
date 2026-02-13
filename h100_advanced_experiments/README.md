# H100 Advanced Experiments

This folder is a reproducible workspace for large-scale RoPE frequency experiments
on 2xH100 (or larger clusters), focused on:

- verifying whether `hybrid_a0.2_t100k` can replace high-theta geometric RoPE;
- scaling validation from small models to `~1.5B`;
- producing paper-ready artifacts (JSON + CSV + figures + markdown summaries).

## Quick Start

1. Read the protocol:
   - `docs/H100_1P5B_EXPERIMENT_PLAN_CN.md`
   - `docs/H100_1P5B_RUNBOOK_CN.md`
   - `docs/H100_RENT_DECISION_BRIEF_CN.md`
2. Fill/adjust matrix:
   - `configs/experiment_matrix_1p5b.yaml`
3. Run environment checks:
   - `bash scripts/bootstrap_h100_env.sh`
4. Run training/eval with your own launcher and save raw JSON into:
   - `results/raw/`
5. Generate charts/tables:
   - `python scripts/plot_h100_results.py --input-dir results/raw --output-dir results/processed`

## Structure

- `configs/`: experiment matrix and schema
- `docs/`: protocol, runbook, AI operator prompt
- `scripts/`: environment checks and result plotting
- `results/`: raw and processed outputs
- `figures/`: exported paper figures (optional mirror of processed figures)
- `logs/`: runtime logs
- `tmp/`: temporary files

## Notes

- Keep all scripts and metrics, but do not commit large model weights.
- Use consistent tokenizer/data split/seed policy across variants.
- Any interrupted run should still keep partial `results.json` for recovery.
