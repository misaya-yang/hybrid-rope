# 2026-02-26 Root-Cause Analysis (LLaMA longinst_v1 tradeoff)

## Symptom
- Gate eval result (pct):
  - `qasper`: base `42.4927` -> LoRA `44.3389` (`+1.8461`)
  - `musique`: base `19.1116` -> LoRA `11.0840` (`-8.0276`)
- This is a strong task tradeoff and fails gate stop criteria.

## Evidence
1. Dataset domain mismatch (primary)
- Artifact: `artifacts/longinst_v1/data/stats.json` on server.
- `selected_long=1680`, `selected_wiki=1`, and preview samples are overwhelmingly continuation-style.
- Many samples use prompt pattern `Long-context continuation... Continue the passage...`, which is weakly aligned with multi-hop QA tasks like Musique.

2. Sample quality signal
- Artifact: `artifacts/longinst_v1/data/preview_20.txt`.
- User/assistant pairs are mostly continuation generation; QA supervision density for exact-answer behavior is low.
- This explains why `qasper` can improve (extractive tolerance) while `musique` collapses (strict multi-hop answer requirements).

3. Pipeline bug that hid detailed diagnostics
- In `scripts/isolated/longinst/new_lora_longinst_train_v1.py`, gate step overwrote raw `qasper_musique_compare.json` with compact summary.
- Consequence: per-sample traces/comparison payload were lost after run, making diagnosis harder.

4. Wrong default base model path (safety issue)
- Same script defaulted to Qwen path, although this run passed LLaMA path explicitly.
- This is a high-risk footgun for future runs.

## Fixes applied
- File: `scripts/isolated/longinst/new_lora_longinst_train_v1.py`
- Changes:
  1) Set `DEFAULT_BASE_MODEL` to `Meta-Llama-3-8B-Instruct`.
  1.1) Set default dataset paths to `LongAlpaca-12k + LongQA` (instead of continuation-only local dump).
  2) Add synthetic long-QA rows (`--synthetic_ratio`, default `0.30`) to reduce continuation-only bias.
  3) Add continuation-dominance guard (`--allow_continuation_dominant_corpus`, default `false`).
     - If continuation-like ratio > 0.70, script fails fast with explicit message.
  4) Preserve raw eval JSON:
     - gate raw: `qasper_musique_compare_raw.json`
     - gate summary: `qasper_musique_compare.json`
     - full raw: `longbench_full_compare_raw.json`
     - full summary: `longbench_full_compare.json`
  5) `parse_compare_scores` now supports both raw `comparison` format and compact `gate_scores` format.

## Practical next run recommendation
- Keep the same fast training lock (`800 steps`, `seq_len=8192`, `r=32`, `sdpa`, `q/k/v/o`, response-only).
- Use the patched script with synthetic QA enabled.
- Run gate (`qasper,musique`) first; only if pass then run lb21 full.

## Why this is code + data, not pure theory failure
- Anchored RoPE + LoRA did not uniformly fail (`qasper` improved).
- Collapse is concentrated on task type (multi-hop QA), strongly consistent with supervision mismatch and insufficient QA-style constraints.
