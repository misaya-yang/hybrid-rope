# Phase 17B: 454M `512 -> 1024` Segmented Continuation

## Intent

验证 staged length extension 是否优于继续固定在 `L=512` 上训练，并检查这种收益是否同时出现在 `Geo` 和 `EVQ` 上。

## Protocol

- Base stage:
  使用 phase17 的 `454M @ L=512` checkpoint 作为 continuation 起点
- Continue stage:
  把训练长度切到 `L=1024`
- Dataset:
  延续 phase17 的 `proof-pile-2`
- Methods:
  `Geo` vs `EVQ`
- EVQ hyperparameter:
  `tau* = d_head / sqrt(L_new) = 64 / sqrt(1024) = 2.0`
- Fresh data:
  continuation 数据从 flat token 源里显式跳过 base stage 已消费的 token，再重新 chunk 成 `1024`
- Segmentation:
  stage-2 内部保存 `25% / 50% / 75% / 100%` checkpoints

## Defaults

- Model:
  `454M` (`24L x 1024H x 16 heads`, `d_head=64`)
- Effective batch:
  对齐 phase17，设为 `20`；实现上使用 `micro_bs=10, grad_accum=2`
- Continue tokens:
  `1.0B`
- Base tokens already consumed:
  `1.0B`
- Data start token:
  `1.0B`
- Passkey mix:
  `5%`，与 phase17 保持一致
- Eval lengths:
  final=`1K, 2K, 4K, 8K, 16K, 32K`
- Checkpoint eval lengths:
  `1K, 4K, 8K, 16K`

## Why This Design

- `tau` 必须跟新的训练长度走，不能沿用 `L=512` 的 `2.83`
- “新数据”不能只是换 cache 名，而是要换 token offset
- 分段 checkpoint 能回答两个问题:
  - staged continuation 是否在 stage-2 内部持续改善长程能力
  - `Geo` 和 `EVQ` 的 stage-2 斜率谁更好

## Run Shape

1. Pilot:
   单 seed，先跑 `Geo` 和 `EVQ` 各一条，确认 `1024` 显存、checkpoint eval 和 fresh-data offset 都正常
2. Full:
   跑完整 stage-2 token budget
3. Compare:
   与 phase17 的 `L=512` raw trajectory 对照，判断 `512 -> 1024` staging 是否减轻 long-range regression

## Outputs

- Summary:
  `phase17b_454m_512_to_1024_continue_summary.json`
- Per-run:
  `seed{seed}/{method}/result.json`
- Per-checkpoint progress:
  `seed{seed}/{method}/checkpoint_eval_progress.json`
- Data provenance:
  `data/train_{dataset}_{tag}_{start}_{tokens}_{seq}.json`
