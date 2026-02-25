# Experiment Environment Snapshot (Updated)

Last updated: 2026-02-21 (after teacher-forcing passkey sanity-check run)
Purpose: one-file environment + artifact map for reproducible continuation.

## 1) Repository and Paths

- Local repo root: `e:\rope\hybrid-rope`
- Remote repo root: `/root/autodl-tmp/dfrope/hybrid-rope`
- Main experiment subdir: `/root/autodl-tmp/dfrope/hybrid-rope/sigmoid_rope_experiments`

## 2) Remote Machine (actual run node)

- Host: `autodl-container-q70yg9yhtt-d2becc30`
- OS: `Linux 5.15.0-78-generic (Ubuntu)`
- Python: `3.12.3`
- Torch: `2.8.0+cu128`
- GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition`
- VRAM: `97887 MiB` (about 96 GB)
- CUDA available: `True`

## 3) Data / Tokenizer / Models Used

- Local model mirror on remote: `/root/autodl-tmp/dfrope/ms_models`
- Local dataset mirror on remote: `/root/autodl-tmp/dfrope/ms_datasets`
- Tokenizer used in phase4 and sanity checks:
  - `hf:/root/autodl-tmp/dfrope/ms_models/EleutherAI/gpt-neox-20b`
  - vocab size `50277`

Checkpoint root used by latest sanity run:
- `/root/autodl-tmp/dfrope/hybrid-rope/sigmoid_rope_experiments/checkpoints`

## 4) Access Constraints

- External HuggingFace access may be blocked.
- Prefer local mirrors and existing checkpoints.
- Remote control workflow: `plink` and `pscp` from Windows.

## 5) Latest Script Added and Executed

New script:
- `scripts/run_passkey_sanity_check.py`

Executed remotely with:

```bash
cd /root/autodl-tmp/dfrope/hybrid-rope
/root/miniconda3/bin/python -u scripts/run_passkey_sanity_check.py \
  --repeats 20 \
  --model_root tmp_phase4_compare \
  --fallback_model_root sigmoid_rope_experiments/checkpoints
```

Run log:
- `/root/autodl-tmp/dfrope/hybrid-rope/logs/run_passkey_sanity_check.log`

## 6) Latest Result (Teacher-Forcing Hit Rate)

| Model | L1 | L2-4K-H | L2-4K-M | L2-4K-E | L2-8K-H | L2-8K-M | L2-8K-E | L3-12K-H | L3-12K-E |
|---|---|---|---|---|---|---|---|---|---|
| Sigmoid | 75.0% | 50.0% | 70.0% | 75.0% | 70.0% | 80.0% | 65.0% | 55.0% | 45.0% |
| Standard | 65.0% | 40.0% | 80.0% | 85.0% | 40.0% | 45.0% | 65.0% | 70.0% | 35.0% |
| Anchored-alpha | 65.0% | 50.0% | 55.0% | 55.0% | 65.0% | 70.0% | 55.0% | 40.0% | 50.0% |
| Anchored-20 | 50.0% | 45.0% | 40.0% | 40.0% | 30.0% | 45.0% | 40.0% | 50.0% | 55.0% |

Important:
- This resolves the previous all-zero issue in generation-based passkey.
- Current run uses `repeats=20`; variance remains non-trivial. Re-run with higher repeats before final claims.

## 7) Synced Back to Local (this session)

New passkey sanity outputs:
- `results/phase4_passkey_sanity/results.json`
- `results/phase4_passkey_sanity/summary.md`
- `results/phase4_passkey_sanity/results.csv`
- `results/phase4_passkey_sanity/aggregated.csv`
- `logs/run_passkey_sanity_check.log`

Phase4 core data synced:
- `sigmoid_rope_experiments/data/phase4_corrected_summary.json`
- `sigmoid_rope_experiments/data/ppl_vs_length.csv`
- `sigmoid_rope_experiments/data/positional_loss.csv`
- `sigmoid_rope_experiments/data/training_log_standard.csv`
- `sigmoid_rope_experiments/data/training_log_sigmoid.csv`
- `sigmoid_rope_experiments/data/training_log_anchored20.csv`
- `sigmoid_rope_experiments/data/training_log_anchored_alpha.csv`
- `sigmoid_rope_experiments/data/passkey_fixed_results.csv`
- `sigmoid_rope_experiments/data/passkey_results_v3.csv`
- `sigmoid_rope_experiments/data/passkey_rootcause_probe.csv`
- `sigmoid_rope_experiments/data/passkey_rootcause_nexttoken.csv`
- `sigmoid_rope_experiments/data/passkey_sanity_probe.csv`
- `sigmoid_rope_experiments/data/passkey_sanity_probe_summary.csv`

Archive copy:
- `archives/server_artifacts_2026-02-21/results/phase4_passkey_sanity/`
- `archives/server_artifacts_2026-02-21/logs/run_passkey_sanity_check.log`

## 8) Next-Run Checklist

1. Keep this teacher-forcing metric as primary passkey sanity metric.
2. Re-run with `--repeats 60` or `--repeats 100`.
3. Add CI/statistical significance before paper-facing figure.
4. Keep generation exact-match as secondary diagnostic only.
