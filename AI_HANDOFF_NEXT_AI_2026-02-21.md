# AI Handoff (Clean) - 2026-02-21

Purpose: this file is the clean handoff baseline for the next AI (Gemini/Claude/ChatGPT) without requiring full chat history.

## 1) Current Project State

- Repo root: `e:\rope\hybrid-rope`
- Main remote experiment root: `/root/autodl-tmp/dfrope/hybrid-rope`
- Core subproject: `sigmoid_rope_experiments`
- Hardware used in latest run: `NVIDIA RTX PRO 6000 Blackwell Server Edition (96GB)`
- Network constraint: external HF access is limited; use local model/data mirrors first.

## 2) Why We Pivoted the Passkey Protocol

Previous passkey evaluation based on greedy generation produced all-zero accuracy and was confounded by decoding priors (models often refuse to emit digits, even when retrieval signal exists).

We therefore added a new **teacher-forcing sanity check**:

- Compare token-level CE on:
  - `Prompt + TruePasskey`
  - `Prompt + FalsePasskey`
- Success criterion: `TrueLoss < FalseLoss`
- This measures retrieval preference and avoids generation-format collapse.

## 3) New Script Added (This Session)

- `scripts/run_passkey_sanity_check.py`

Design implemented:

1. Level 1 Near-Copy Control:
- Context `256`
- Needle near the suffix with ~20-token gap

2. Level 2 Distance Buckets:
- Context `4096` and `8192`
- Depths: `hard=10%`, `medium=50%`, `easy=90%`

3. Level 3 Mild Extrapolation:
- Context `12000`
- Depths: `hard=10%`, `easy=90%`

Models evaluated:
- `Standard`
- `Sigmoid`
- `Anchored-alpha`
- `Anchored-20`

## 4) Latest Sanity-Check Run (Executed on Remote)

Remote command used:

```bash
cd /root/autodl-tmp/dfrope/hybrid-rope
/root/miniconda3/bin/python -u scripts/run_passkey_sanity_check.py \
  --repeats 20 \
  --model_root tmp_phase4_compare \
  --fallback_model_root sigmoid_rope_experiments/checkpoints
```

Key result table (hit rate):

| Model | L1 | L2-4K-H | L2-4K-M | L2-4K-E | L2-8K-H | L2-8K-M | L2-8K-E | L3-12K-H | L3-12K-E |
|---|---|---|---|---|---|---|---|---|---|
| Sigmoid | 75.0% | 50.0% | 70.0% | 75.0% | 70.0% | 80.0% | 65.0% | 55.0% | 45.0% |
| Standard | 65.0% | 40.0% | 80.0% | 85.0% | 40.0% | 45.0% | 65.0% | 70.0% | 35.0% |
| Anchored-alpha | 65.0% | 50.0% | 55.0% | 55.0% | 65.0% | 70.0% | 55.0% | 40.0% | 50.0% |
| Anchored-20 | 50.0% | 45.0% | 40.0% | 40.0% | 30.0% | 45.0% | 40.0% | 50.0% | 55.0% |

Interpretation:
- Generation-based passkey all-zero was a misleading bottleneck.
- Teacher-forcing protocol can now distinguish model variants and depths.
- Current numbers are still noisy (`repeats=20`) and should be rerun with higher repeats before paper claims.

## 5) Evidence Synced Back to Local

Main local outputs:

- `results/phase4_passkey_sanity/results.json`
- `results/phase4_passkey_sanity/summary.md`
- `results/phase4_passkey_sanity/results.csv`
- `results/phase4_passkey_sanity/aggregated.csv`
- `logs/run_passkey_sanity_check.log`

Additional synced phase4 data:

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

Archived copy:

- `server_artifacts_2026-02-21/results/phase4_passkey_sanity/*`
- `server_artifacts_2026-02-21/logs/run_passkey_sanity_check.log`

## 6) What Next AI Should Do First

1. Re-run sanity check with stronger statistics:
- `--repeats 100` (or at least 60)
- Keep identical checkpoint/model roots

2. Add confidence intervals and significance:
- bootstrap CI per cell
- pairwise delta vs Standard in same cell

3. Keep both metrics in report:
- teacher-forcing retrieval hit (primary for small/base models)
- generation exact-match (secondary diagnostic only)

4. Do not claim final superiority from this table alone:
- It is a protocol sanity result, not full downstream benchmark closure.

## 7) Cautions

- `tmp_phase4_compare/` in this repo currently contains CSV files only; checkpoints are in `sigmoid_rope_experiments/checkpoints` on remote.
- Avoid deleting historical artifact folders before finishing manuscript evidence packaging.
- For final paper figures, use repeated runs and CI bars to avoid reviewer criticism on variance.
