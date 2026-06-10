# Opus 4.8 Artifact Recovery Runbook

Purpose: provide exact, sanitized steps for recovering external AutoDL/server
artifacts needed to close the remaining Opus 4.8 P0 gates. This runbook is
read-only: it hashes, audits, and summarizes artifacts; it does not train or
evaluate models.

Use this when you have access to an external machine or archive that still has
1B/4K MLA, primary MLA, or other raw run directories.

## P0 Targets

| Target | Why it matters | Minimum artifact set |
| --- | --- | --- |
| 1B/4K MLA Geo and EVQ raw runs | Current compact repo has Markdown report only. | run dirs with `results.json`, `model.pt`, `inv_freq.npy`, train/val cache tensors. |
| 1B/4K MLA YaRN and YaRN+FT eval outputs | Needed to verify Phase18 target/beyond-target claims. | `yarn_ft_s*_seed*_results.json` or equivalent eval JSON plus command/config notes. |
| 8K/500M primary MLA checkpoints | Needed to prove checkpoint-level `inv_freq` provenance for primary MLA. | per-seed `model.pt`/checkpoint, `inv_freq.npy`, `results.json`, data hashes. |
| Data caches for 1B and primary MLA | Needed to distinguish true FineWeb-Edu from Pile/OWT/FineWeb/C4 cache labels. | `train_*.pt`, `val_*.pt`, source/prep script/config notes. |

## Copy Nothing Sensitive First

Before copying anything into the public repo:

- Do not copy raw checkpoints or token tensors into `docs/`, `paper/`, or
  reviewer supplement paths.
- Do not paste absolute external paths into public docs.
- Do not copy private logs containing account names, hostnames, tokens, or
  service URLs.
- First generate sanitized JSON manifests on the external machine and review
  those manifests locally.

## Step 1: Generate A Sanitized Artifact Manifest

Run from the repository checkout on the external machine, replacing the
placeholder paths with the recovered run directories.

```bash
python scripts/core_text_phases/make_artifact_manifest.py \
  --entry mla_1b_4k_geo_seed42=<recovered-geo-run-dir> \
  --entry mla_1b_4k_evq_seed42=<recovered-evq-run-dir> \
  --entry mla_500m_8k_geo_seed42=<recovered-primary-geo-run-dir> \
  --entry mla_500m_8k_evq_seed42=<recovered-primary-evq-run-dir> \
  --output /tmp/opus48_artifact_manifest.json \
  --inspect-tensors \
  --rope-audit \
  --base 500000 \
  --rope-dim 32 \
  --d-rope 32 \
  --d-head 64 \
  --d-eff 128 \
  --tau 1.414
```

Expected properties of `/tmp/opus48_artifact_manifest.json`:

- `path_policy` is `sanitized_path_hints_only`.
- each run has SHA256 entries for `results.json`, `model.pt`, and
  `inv_freq.npy` where present.
- tensor entries include shape/dtype/numel, not absolute paths.
- `rope_audit.actual_inv_freq.sha256` is present for checkpoints with RoPE
  buffers.
- `rope_audit.classification` is consistent with the claimed Geo/EVQ label.

## Step 2: Audit Train-Cache Token Math

Run once per work directory that contains the training cache tensors and run
subdirectories.

For 1B/4K MLA:

```bash
python scripts/core_text_phases/audit_training_artifacts.py \
  --work-dir <recovered-1b-work-dir> \
  --dataset fineweb-edu \
  --seq-len 4096 \
  --batch-size 12 \
  --train-tokens 1000000000 \
  --val-tokens 5000000 \
  --output /tmp/opus48_1b_training_artifacts.json
```

For 8K/500M MLA:

```bash
python scripts/core_text_phases/audit_training_artifacts.py \
  --work-dir <recovered-500m-work-dir> \
  --dataset fineweb-edu \
  --seq-len 8192 \
  --batch-size 6 \
  --train-tokens 500000000 \
  --val-tokens 5000000 \
  --output /tmp/opus48_500m_training_artifacts.json
```

Expected fields:

- `train_cache.shape`, `dtype`, and `numel`
- `token_math.actual_tokens_in_cache`
- `token_math.used_tokens_by_train_loop`
- `token_math.dropped_tokens_from_batch_floor`
- `run_artifacts[*].results_json`
- `run_artifacts[*].model_pt`
- `run_artifacts[*].inv_freq_npy`

## Step 3: Audit Individual Checkpoints

Use this when a checkpoint is recovered but the directory layout is unclear.

```bash
python scripts/core_text_phases/audit_rope_checkpoint.py \
  --checkpoint <checkpoint-or-run-dir> \
  --base 500000 \
  --rope-dim 32 \
  --d-rope 32 \
  --d-head 64 \
  --d-eff 128 \
  --tau 1.414
```

Reviewer-useful output fields:

- `source.kind`
- `source.first_buffer`
- `source.num_buffers`
- `actual_inv_freq.sha256`
- `actual_inv_freq.first_8`
- `actual_inv_freq.last_8`
- `phi_grid.sha256`
- `closest_reference`
- `classification`

## Step 4: Acceptance Criteria

The 1B/4K row can be upgraded from "Markdown report only" only if all are true:

- exact raw/eval JSON files are found and hashed.
- the recovered Geo and EVQ checkpoints have distinct, label-consistent
  `inv_freq` audits.
- training cache tensor source and shape are documented.
- token math shows actual cached/used tokens rather than only nominal `1B`.
- seed list is explicit; if only seed 42 is found, the row remains single-seed
  supporting/negative evidence.
- no manifest contains private absolute paths or host/user identifiers.

The row is still not a same-config longer-training ablation unless the recovered
artifacts prove identical train length, data source, seeds, model config, and
eval protocol to the 8K/500M primary MLA row except for token budget.

## Step 5: Public Import

After local review:

1. copy only sanitized JSON manifests into a curated/audit location.
2. add their hashes to `docs/overview/RESULT_PROVENANCE_MANIFEST.md`.
3. update `docs/overview/EXPERIMENT_CODE_RESULT_AUDIT.md` status for the row.
4. update `docs/overview/OPUS48_ISSUE_RESOLUTION_LEDGER.md` for O48-13/O48-14.
5. rerun:

```bash
python tests/test_opus48_audit_docs.py
python tests/test_artifact_manifest.py
python tests/test_training_artifact_audit.py
git diff --check
```

Only then should the recovered row be cited in a rebuttal.
