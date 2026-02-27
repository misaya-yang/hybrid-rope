# Protocol Lock (Qwen Evidence Chain)

Last updated: 2026-02-25

This file defines the **non-negotiable** protocol knobs for anything that enters the main evidence chain.

## 1) Equality Contract (must match within a comparison)

All methods compared in the same table/figure must share:
- Same **base checkpoint** weights
- Same **tokenizer** and chat template behavior
- Same **evaluation manifest** (exact same sample indices per task)
- Same **decode** settings
- Same **RoPE injection path** (`inv_freq.copy_()`-style)
- Same **context length** and truncation policy

If any deviation occurs: mark that row `INVALID` and do not cite.

## 2) LongBench “official parity” knobs (paper default)

These are the paper-default settings for LongBench:

- `--prompt_source official`
- `--chat_template auto` (must use `apply_chat_template(..., add_generation_prompt=True)` when available)
- `--truncate_mode middle`
- `--max_new_tokens_policy official`
- `--score_scale pct` (store `raw` too, but paper shows `pct`)
- `--max_samples_per_task 0` (full set, not preview)

Paired evaluation is enforced by:
- `--manifest_json artifacts/manifests/longbench_manifest_<model>_ctx<ctx>_seed<seed>.json`

## 3) Decode settings (deterministic)

Greedy decoding only:
- `do_sample = False`
- `temperature = None`
- `top_p = None`
- `use_cache = True`

## 4) Anchored-sigmoid tuned parameters (locked)

For anchored-sigmoid runs in this evidence chain, the tuned schedule is locked to:
- `anchor_factor = 4`
- `slope_raw = 20`
- `center_ratio = 0.70`

Every run must record:
- `custom_inv_freq.pt` path
- `inv_sha256` (checksum)

## 5) Repro manifest outputs (required)

Every LongBench run must emit:
- `baseline_gold.yaml`
- `env_freeze.txt`
- `code_hash.txt`

Under:
- `artifacts/repro_manifest/<run_id>/`

## 6) Claim policy (statistics)

Claims are **graded** by FDR-adjusted p-values:
- If `p_fdr_bh < 0.05`: “significant improvement”
- Else: “directional improvement consistent with theory (p=..., FDR-adjusted p=...)”

The claim text should be generated from the stats report (no freehand wording).

