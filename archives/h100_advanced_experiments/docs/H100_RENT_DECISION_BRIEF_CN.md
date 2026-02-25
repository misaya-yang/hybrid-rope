# H100 租用决策简报（1.5B 实验）

## 结论先行

如果目标是“尽快拿到可投稿证据”，建议从 `2xH100, 72 小时`起步。  
理由：能覆盖 `P0 + P1 + 部分 P2`，既有主结论，也有下游任务支撑。

## 三档预算建议

### 档位 A（48h，最低可行）

- 能做：
  - 3 配置单 seed 完整跑通
  - 最多 1~2 配置补 3 seeds
  - 16K 主指标图
- 风险：
  - 统计显著性偏弱
  - 32K/下游任务可能不完整

### 档位 B（72h，推荐）

- 能做：
  - 3 配置 × 3 seeds（主结论完整）
  - 16K/32K PPL 对比
  - 1 套长任务（LongBench 或 RULER）
- 收益：
  - 论文叙事闭环基本齐全

### 档位 C（120h，冲击更强证据）

- 能做：
  - 完整 3-seed + 长任务双评测（LongBench + RULER）
  - 常规任务稳定性（lm-eval）
  - 更完整 ablation（alpha/theta 扫描）
- 收益：
  - 结果更稳健， rebuttal 更有底气

## 最小“可投稿包”清单

1. 主表：`Config | PPL@2048 | PPL@16384 | PPL@32768 | mean±std`
2. 图 1：`PPL vs Length`
3. 图 2：`PPL@16K/32K bar`
4. 图 3：`Train loss vs tokens`
5. 下游表：至少一组长上下文 benchmark

## 何时可以说“可以冲顶会”

满足以下三条更稳：

1. `hybrid_a0.2_t100k` 在 1.5B、3 seeds 下持续优于 `geo_500k`（16K+）
2. 32K 仍保留优势或至少持平，同时 2K 不明显退化
3. 下游长任务（LongBench/RULER）也体现同向收益

如果只满足 1 和 2，也已经是很强 workshop / 主会 short paper 水平。
