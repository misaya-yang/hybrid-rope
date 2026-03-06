# Advisor Package Index (2026-02-15)

- Generated at: `2026-02-15 01:23:12`
- Package root: `results/advisor_package_2026-02-15`
- File count: `36`
- Rule: data and logs only, no model/checkpoint weights.

## 0) Current training snapshot

- Remote fair suite process is alive and sequential pipeline is intact.
- Completed variants: `yarn`, `pi`.
- Running variant: `hybrid` (latest observed around `step 40/600`, loss from `5.13 -> 3.00`, GPU reserved around `81.66 GB`).
- Waiting daemon is alive: `run_8b_post_eval.py --wait_for_suite ...`.

## 1) Core evidence highlights

- From-scratch scaling:
  - `50M 3cfgx3seed`: `hybrid_a0.2_t100k` at 16K (`17.324`) better than `geo_500k` (`18.207`).
  - `350M final`: at 16K, `hybrid_a0.2_t100k=12.646` vs `geo_500k=14.653`.
- LLaMA long-context collapse control:
  - `llama_shape_theta_min`: `geo_10k` collapse ratio `22.026x`, `sigmoid_t100k` `1.077x`.
- Fair 8B LoRA baseline run (ongoing):
  - `yarn` done: PPL@16K `6.0566`, PPL@32K `6.2702`.
  - `pi` done: PPL@16K `6.1369`, PPL@32K `6.3100`.
  - `hybrid` currently training.
- Qwen line:
  - `qwen_hybrid_lora`: retrieval metrics remain high, but PPL increases vs base in this run protocol.

## 2) Folder map

- `01_scaling_from_scratch`
  - 50M/350M key JSON + 350M run log.
- `02_llama_long_context`
  - shape/theta boundary and collapse evidence.
- `03_llama8b_fair_lora`
  - A800 summary, fair-suite logs, `yarn`/`pi` summaries, `hybrid` run log.
- `04_niah_and_retrieval`
  - NIAH base matrix and passkey results.
- `05_qwen_and_cross_model`
  - Qwen hybrid-LoRA and cross-model comparison outputs.
- `06_700m_trainfreq`
  - 700M train/eval outputs currently synced.
- `07_docs_and_scripts`
  - advisor docs and the key automation scripts used for this stage.

## 3) Notes for advisor talk track

- The 8B fair suite is still running; this package includes latest live logs and finished variant summaries.
- `06_700m_trainfreq` contains synced outputs, but PPL values are clipped (`22026.46`) in current JSON, so this subline should be presented as "protocol check in progress", not final headline evidence.

