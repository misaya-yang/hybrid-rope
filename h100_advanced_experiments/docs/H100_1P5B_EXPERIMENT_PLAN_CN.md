# 1.5B on 2xH100 实验总方案（可落地版）

## 0. 目标与核心主张

本轮要验证的不是“任意方法都能涨点”，而是一个可投稿主张：

- 主张 A：`hybrid_a0.2_t100k` 在长上下文（16K/32K）可稳定接近或优于 `geo_500k`；
- 主张 B：在保持短上下文质量（2K/4K）不明显退化的前提下，Hybrid 可降低对超高 `theta` 的依赖；
- 主张 C：该现象可在更大模型（1.5B）和更真实任务（LongBench/RULER + 常规任务）上复现。

## 1. 资源与边界

- 机器：2xH100 80GB（NVLink 优先）
- 并行策略：推荐 1 个作业使用双卡 FSDP/ZeRO，避免频繁切换导致吞吐波动
- 训练精度：BF16（必须）
- 序列长度：
  - 训练：2048（主设置，和既有 50M/350M 保持可比）
  - 评测：`[2048, 4096, 8192, 12288, 16384, 32768]`

## 2. 模型与数据（1.5B 实用配置）

### 2.1 模型配置建议（目标约 1.4B~1.6B）

```yaml
vocab_size: 50304
hidden_size: 2048
num_layers: 24
num_heads: 16
head_dim: 128
intermediate_size: 5632
max_position_embeddings: 2048
dropout: 0.0
```

说明：
- 该规模在 2xH100 + BF16 + FSDP 下可运行；
- 如果首轮吞吐太低，可降到 `hidden=1792` 或 `num_layers=22`，先完成结论再回升规模。

### 2.2 数据配方（比 TinyStories 更接近真实）

建议三层数据策略（可按预算裁剪）：

1. 主预训练语料（高占比）  
   - FineWeb（主干）  
   - SlimPajama（补充多域）
2. 长文增强（中占比）  
   - PG-19（长篇结构）
3. 验证/评测集合（固定不变）  
   - TinyStories validation（和历史结果对齐）
   - LongBench / LongBench v2 / RULER（长上下文泛化）

## 3. 频率配置矩阵（主实验）

最小必跑（确保论文核心结论）：

1. `geo_500k`（强基线）
2. `hybrid_a0.2_t100k`（当前冠军）
3. `anchpoly_p3.9_omf0.3_t500k`（替代结构）

可选增强（预算允许再加）：

4. `geo_100k`
5. `hybrid_a0.1_t100k`
6. `hybrid_a0.3_t100k`

种子策略：
- 主线至少 3 seeds：`[42, 123, 7]`
- 若时间紧：先单 seed 选出前 2，再做 3-seed 完整对比。

## 4. 分阶段执行（保证可产出）

### Phase P0：8 小时内拿到第一版可用结论

- 每个配置先跑 `10%` token budget 的短跑；
- 输出：
  - train loss 曲线
  - `PPL@2048` 和 `PPL@16384`
- Gate：
  - 若 `PPL@2048` 明显崩（相对最佳 > 15%），直接淘汰该配置。

### Phase P1：主实验（可投稿核心）

- 保留前 2~3 个配置，跑满预设 token budget（建议 20B~60B，按租用时长选）
- 固定同一验证集切片策略；
- 每个配置保存：
  - `results.json`
  - `loss_curve.jsonl`
  - `eval_per_length.json`

### Phase P2：下游验证（决定“顶会力度”）

- 长上下文任务：LongBench / LongBench v2 / RULER
- 常规能力任务：lm-eval-harness 上 4~6 个任务（HellaSwag/PIQA/ARC/Wino 等）
- 目标：证明 Hybrid 不只是 PPL 特例，而是“长上下文收益 + 主流任务不退化”。

## 5. 成功判据（建议论文口径）

至少满足其一：

1. 在 `16K` 或 `32K`，`hybrid_a0.2_t100k` 均值优于 `geo_500k`（3 seeds）；
2. 在长任务集（LongBench/RULER）整体分数优于 `geo_500k`，且短任务不显著退化；
3. 频率风险指标（collision/OOD surrogate）与长程指标显著相关，支撑解释性叙事。

## 6. 图表清单（至少产出以下 6 张）

1. `Train Loss vs Tokens`（每配置一条线）
2. `PPL vs Length`（2K→32K）
3. `PPL@16K Bar (mean±std)`（主比较图）
4. `Short-vs-Long Scatter`（x:2K PPL, y:16K/32K PPL）
5. `LongBench/RULER Radar or Grouped Bars`
6. `Risk Surrogate vs 16K PPL`（解释性图）

## 7. 常见失败模式与应对

1. 2K 质量掉太多  
   - 降低 hybrid alpha（0.2 -> 0.15）或加一点高频保护
2. 16K 没提升  
   - 增大 `theta_base`（100k -> 200k）并复测
3. 指标抖动大  
   - 增加 seed 或统一评测 chunk 数（至少 10）
4. 结果不可复现  
   - 固化 tokenizer/data slicing/seed；每步写日志与版本信息

## 8. 交付物（GitHub 可直接上传）

- `configs/experiment_matrix_1p5b.yaml`
- `results/raw/*.json`（无权重）
- `results/processed/*.csv`
- `results/processed/*.md`
- `results/processed/figures/*.png`
- `docs/H100_1P5B_EXPERIMENT_PLAN_CN.md`
- `docs/H100_1P5B_RUNBOOK_CN.md`

---

## 参考资料（用于方案依据）

- RoFormer（RoPE）：https://arxiv.org/abs/2104.09864  
- YaRN：https://arxiv.org/abs/2309.00071  
- LongBench：https://arxiv.org/abs/2308.14508  
- LongBench v2：https://arxiv.org/abs/2412.15204  
- RULER：https://arxiv.org/abs/2404.06654  
- Chinchilla scaling law：https://arxiv.org/abs/2203.15556  
- NVIDIA H100 官方规格：https://www.nvidia.com/en-us/data-center/h100/  
- PyTorch FSDP 文档：https://pytorch.org/docs/stable/fsdp.html  
- Hugging Face Accelerate FSDP：https://huggingface.co/docs/accelerate/main/en/usage_guides/fsdp  
- FlashAttention 项目：https://github.com/Dao-AILab/flash-attention  
- FineWeb 数据卡：https://huggingface.co/datasets/HuggingFaceFW/fineweb  
- SlimPajama 数据卡：https://huggingface.co/datasets/cerebras/SlimPajama-627B  
