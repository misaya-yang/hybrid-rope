# Scripts

This directory keeps only scripts that are still useful for the current paper package or the immediately next evaluation wave.

## Structure

- `scripts/train.py`: core training entrypoint kept for from-scratch and continued-pretraining runs
- `scripts/core_text_phases/`: the main Phase 8–15 text experiment chain
- `scripts/supporting_eval/`: generic evaluation utilities that are still useful but are not themselves the core experiment chain
- `scripts/data_prep/`: dataset preparation helpers still needed for reproducibility and next-step runs
- `scripts/figures/`: paper figure generators
- `scripts/video_temporal/`: temporal-only video extrapolation support
- `scripts/mac_train/`: M4 Max local experiments (legacy, ~20GB assumption)
- `scripts/m4_max_36gb/`: M4 Max 36GB local experiments — progressive chain, base sweep, τ landscape, 350M scaling
- `scripts/lib/rope/`: retained local RoPE schedule and injection code

## Working Rule

If a script cannot be tied to a paper figure, a paper table, a core experiment report, or the next planned evaluation step, it should not live here.
