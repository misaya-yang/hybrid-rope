# Table 23 LoRA Rebuttal Worksheet

日期：2026-06-10

用途：把用户新增的 Geo+LoRA 控制组整理成 rebuttal 可用的 Table 23。当前 workspace 只找到 LoRA 脚本/计划，没有找到可引用的结果 JSON/CSV；因此本文件是数据接收和落表模板，不填造数字。

## 0. 当前 Workspace 状态

已确认：

- 旧论文 LoRA 表是 Base vs EVQ-LoRA 两行，见 `paper/appendix/a4_supporting_experiments.tex:35-51`。
- LoRA v2 目录说明它是 supporting experiment，不属于 primary claim tier，见 `experiments/lora_evq_v2/README.md:1-17`。
- LoRA 计划设定 LLaMA-3-8B-Instruct、EVQ tau=1.414、LoRA rank=64、alpha=128、LongAlign-10k、8K/16K/32K PPL，见 `experiments/lora_evq_v2/EXPERIMENT_PLAN.md:27-70`。
- `scripts/2026-04/README.md:7-21` 列出 GEO seeds 42/43/44 与 EVQ seeds 43/44 的训练/评估 pipeline。
- `scripts/2026-04/PAPER_HANDOVER_2026-04-27.md:80-91` 把 Geo+LoRA control 列为高 ROI rebuttal 实验。
- `scripts/2026-04/PAPER_HANDOVER_2026-04-27.md:161-165` 说明当时 Geo+LoRA baseline 脚本未跑。
- `experiments/lora_evq_v2` 与 `scripts/2026-04` 当前没有可直接落表的 `.json` / `.csv` 结果文件。
- 本轮在允许范围内复查 `experiments/lora_evq_v2`、`scripts/2026-04`、`docs`、`paper`、`rebuttal`，仍未发现可引用的 Geo+LoRA exact result table；现有文件仍是计划、脚本或 handover note。

结论：Geo+LoRA 新结果如果已经完成，应来自当前 repo 之外或未同步目录。不能在 rebuttal 里写具体数字，除非把结果文件或数字补进来。

## 1. 必须收齐的数据字段

每个 row 至少需要：

| 字段 | 必须性 | 说明 |
| --- | --- | --- |
| Method | 必须 | Base / Geo+LoRA / EVQ-LoRA |
| Seed scope | 必须 | single seed or seeds list; if mixed, every metric must say which seeds |
| Checkpoint | 必须 | same pretrained checkpoint for Base/Geo/EVQ |
| LoRA rank | 必须 | expected r=64 |
| LoRA alpha | 必须 | expected alpha=128 |
| Steps | 必须 | expected 300 steps; if stage2 exists, separate table |
| Training data | 必须 | LongAlign-10k or exact subset |
| Frequency schedule | 必须 | Base native Geo; Geo+LoRA no frequency injection; EVQ-LoRA tau=1.414 |
| Eval set | 必须 | WikiText2/PPL dataset or exact eval corpus |
| PPL@8K | 必须 | in-distribution native length; used for cost split |
| PPL@16K | 必须 | 2x extrapolation |
| PPL@32K | 必须 | 4x extrapolation |
| RULER / LongBench / passkey | 可选 | keep separate from PPL table unless robust |
| Zero-training scaler baseline | P1 | Geo + Dynamic NTK / YaRN at 16K/32K, eval-only |

## 2. Seed-Level Raw Table

Fill one row per evaluated seed. Do not average before this table exists.

| Method | Seed | Adapter/checkpoint id | Rank/alpha | Steps | Data | PPL@8K | PPL@16K | PPL@32K | Notes |
| --- | ---: | --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| Base | n/a | TBD | n/a | 0 | n/a | TBD | TBD | TBD | no LoRA |
| Geo+LoRA | 42 | TBD | 64/128 | 300 | TBD | TBD | TBD | TBD | matched control |
| Geo+LoRA | 43 | TBD | 64/128 | 300 | TBD | TBD | TBD | TBD | matched control |
| Geo+LoRA | 44 | TBD | 64/128 | 300 | TBD | TBD | TBD | TBD | matched control |
| EVQ-LoRA | 42 | TBD | 64/128 | 300 | TBD | TBD | TBD | TBD | tau=1.414 |
| EVQ-LoRA | 43 | TBD | 64/128 | 300 | TBD | TBD | TBD | TBD | tau=1.414 |
| EVQ-LoRA | 44 | TBD | 64/128 | 300 | TBD | TBD | TBD | TBD | tau=1.414 |

If EVQ seed 42 is the old checkpoint named `evq_r64_tau1414`, mark that explicitly instead of silently pooling it with newer stage1 runs.

## 3. Rebuttal Table Template

Only use this after seed-level rows are verified.

| Method | Control role | Seeds | PPL@8K | PPL@16K | PPL@32K | In-dist. cost vs Base | Extrap. gain vs Base | Incremental EVQ gain vs Geo+LoRA |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | pretrained checkpoint | n/a | TBD | TBD | TBD | -- | -- | -- |
| Geo+LoRA | adaptation control | TBD | TBD | TBD | TBD | TBD | TBD | -- |
| EVQ-LoRA | frequency injection + same adaptation | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Recommended deltas:

- In-dist. cost vs Base at 8K:
  - Geo cost = `(Geo8K - Base8K) / Base8K`
  - EVQ total cost = `(EVQ8K - Base8K) / Base8K`
  - EVQ incremental cost = `(EVQ8K - Geo8K) / Geo8K`
- Extrapolation gain vs Base at 16K/32K:
  - Geo gain = `(GeoL - BaseL) / BaseL`
  - EVQ gain = `(EVQL - BaseL) / BaseL`
  - EVQ incremental gain = `(EVQL - GeoL) / GeoL`

## 4. Interpretation Rules

| Outcome | Rebuttal interpretation | Claim to avoid |
| --- | --- | --- |
| Geo+LoRA no meaningful extrapolation gain; EVQ-LoRA large gain | Strong control: LoRA/LongAlign alone does not explain extrapolation | Do not say from-scratch industrial durability is proven |
| Geo+LoRA has similar 8K cost but little 16K/32K gain | Cost split helps: much of in-dist cost is adaptation, extrapolation is EVQ-specific | Do not call all +30% “EVQ cost” |
| Geo+LoRA also gains at 16K/32K, but less than EVQ | Attribute only incremental gap to EVQ | Do not claim EVQ uniquely solves LoRA extension |
| Geo+LoRA equals or beats EVQ-LoRA | LoRA row becomes supporting caution, not defense | Do not hide the control |
| Geo + zero-training scaler equals EVQ-LoRA | Claim becomes competitive/supporting, not unique | Do not say raw Geo is the only relevant baseline |

## 5. Safe Response Paragraph Once Filled

Use only after exact numbers are inserted.

> We agree that the original LoRA row could not isolate frequency injection from LoRA/LongAlign adaptation. We therefore added a matched Geo+LoRA control on the same pretrained checkpoint, data, rank, and adaptation schedule. The table separates adaptation cost from EVQ-specific frequency injection: Base -> Geo+LoRA measures the cost of the LoRA path itself, while Geo+LoRA -> EVQ-LoRA measures the incremental effect of EVQ under the same adaptation budget. We will report this row as supporting evidence on a heavily pretrained checkpoint, not as from-scratch trillion-token validation.

## 6. Do Not Write

- “EVQ-LoRA proves EVQ scales to industrial pretraining.”
- “The full +30% at 8K is EVQ’s cost.”
- “Geo+LoRA has no effect” unless the table actually proves it.
- “EVQ-LoRA beats default LLaMA context extension” unless the zero-training Geo scaler reference is included.
- “RULER/LongBench improves” unless exact benchmark results support it.
