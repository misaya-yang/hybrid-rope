# NeurIPS 2026 冲刺总计划

> 最后更新：2026-02-22
> 目标：系统性补全证据链，让论文达到 NeurIPS 主会录用水平

---

## 0. 当前位置 vs 目标差距

```
你已有的 ✅                          NeurIPS 需要的 ❌
──────────────────                  ──────────────────
50M/100M/350M 从零训练              13B+ 规模验证
NIAH passkey 评测                   RULER / LongBench 等真实任务
4 方法对比（baseline/PI/YaRN/ours） + LongRoPE / LongRoPE2 / CLEX / NTK
Phase Collision 机理分析             形式化理论 + proof sketch
单架构（LLaMA）                      跨架构（Mistral / Qwen）
```

---

## 1. 论文定位（决定一切）

### 核心主张

> **Frequency spectrum shape is an effective and underexplored dimension for improving RoPE long-context extrapolation. We prove there is no universal optimal frequency allocation — the optimal shape depends on the distance distribution prior D(Δ), and provide a principled framework for designing task-specific frequency allocations.**

### 三大贡献

| # | 贡献 | 类型 | 当前状态 |
|---|------|------|---------|
| C1 | 理论框架：距离先验 → 频谱形状 → 外推性能 | Theory | ⚠️ 需要形式化 |
| C2 | Anchored Hybrid 方法 + 从零训练/LoRA 验证 | Method | 🔄 8B 运行中 |
| C3 | Phase Collision 机理指标 + 诊断工具 | Analysis | ✅ 已有数据 |

---

## 2. 实验矩阵（完整版）

### Phase A：当前 8B 实验（已在进行 🔄）

| 实验 | 方法 | 规模 | 评测 | 预计完成 |
|------|------|------|------|---------|
| A1 | baseline/PI/YaRN/anchored_hybrid | 8B LoRA 300步 | NIAH | 02-23 凌晨 |

### Phase B：补全 SOTA 基线（优先级 P0）

> [!IMPORTANT]
> 这是拉开与普通投稿差距最大的地方。必须把 LongRoPE2 和 CLEX 加进来。

| 实验 | 方法 | 说明 | 工作量 | 算力需求 |
|------|------|------|--------|---------|
| B1 | + LongRoPE | 非均匀 rescale，progressive YaRN 变体 | 实现 LongRoPE 频率搜索 | 1×GPU 4h |
| B2 | + LongRoPE2 | Needle-driven PPL + 进化搜索 | 复现核心逻辑 | 1×GPU 8h |
| B3 | + CLEX | 连续长度外推 | 集成 CLEX 频率方案 | 1×GPU 4h |
| B4 | + NTK-aware | 动态 NTK 缩放 | 简单实现 | 1×GPU 2h |
| B5 | + Code Llama ABF | Adjusted Base Frequency | 简单改 base | 1×GPU 1h |

**实现策略**：所有方法统一用你的 `inv_freq.copy_()` 公平协议。不同方法只是生成不同的 `custom_inv_freq`。这是你的独特框架优势——把所有方法拉到同一个比较平面。

### Phase C：下游任务基准（优先级 P0）

| 基准 | 任务类型 | 长度范围 | 指标 | 说明 |
|------|---------|---------|------|------|
| **RULER** | 合成 + 真实混合 | 4K-128K | Accuracy | NeurIPS reviewer 最认的 long-ctx 基准 |
| **LongBench** | 真实任务（QA/摘要/多跳） | 4K-32K | F1/ROUGE | 最广泛使用的真实任务基准 |
| **InfiniteBench** | 超长检索+理解 | 100K+ | Accuracy | 展示极端外推能力 |
| **PPL@length** | 语言建模 | 2K-128K | Perplexity | 你已有，补充更多长度点 |

**最低要求**：RULER + LongBench。有这两个 reviewer 就不会质疑评测不足。

### Phase D：规模提升（优先级 P1）

| 实验 | 规模 | 训练/推理 | 说明 | 算力需求 |
|------|------|----------|------|---------|
| D1 | 13B LoRA | 训练 300步 | 证明方法在更大规模有效 | 2×GPU 8h |
| D2 | 70B eval-only | 纯推理 | 只替换频率跑 PPL，不训练 | 4×GPU 2h |
| D3 | 8B 全参数微调 | 训练 1000步 | 证明不依赖 LoRA | 4×GPU 12h |

**70B eval-only 很重要**：只需要加载模型 → 替换 inv_freq → 跑 PPL/NIAH。不需要训练，但能证明方法在 70B 级别也有效。这是性价比最高的实验。

### Phase E：跨架构验证（优先级 P2）

| 模型 | 架构差异 | 说明 |
|------|---------|------|
| Mistral-7B | GQA + Sliding Window | RoPE 变体，验证通用性 |
| Qwen2.5-7B | 不同 head_dim | 你已有部分数据 |
| Phi-3-mini | 不同 tokenizer + 训练数据 | Microsoft 架构 |

**最低要求**：至少 1 个非 LLaMA 架构。Mistral 最合适。

### Phase F：消融实验（优先级 P1）

| 消融 | 变量 | 目的 |
|------|------|------|
| F1 | rigid_j0 = {4, 8, 12, 16, 24} | 锚定维度数的影响 |
| F2 | 训练步数 = {100, 200, 300, 500} | 收敛速度对比 |
| F3 | 上下文长度 = {8K, 16K, 32K} | 训练长度的影响 |
| F4 | 不同 hybrid shape（sigmoid vs polynomial vs linear） | 形状的影响 |

---

## 3. 论文产出物清单

### 主论文图表

| 编号 | 内容 | 数据来源 |
|------|------|---------|
| Fig.1 | 频谱形状示意图（geo vs hybrid vs sigmoid） | 可视化 |
| Fig.2 | Phase Collision 机理图（θ_i × Δ 矩阵） | 已有 |
| Fig.3 | RULER/LongBench 主实验结果（雷达图/柱状图） | Phase C |
| Fig.4 | NIAH 热力图（depth × length × method） | Phase A |
| Fig.5 | PPL vs Length 曲线（所有方法 + 所有规模） | Phase A+D |
| Fig.6 | 消融实验结果 | Phase F |

| 编号 | 内容 | 数据来源 |
|------|------|---------|
| Tab.1 | 主实验对比（8B/13B × 6+ 方法 × RULER/LongBench） | Phase B+C |
| Tab.2 | 从零训练结果（50M→350M 一致性） | 已有 |
| Tab.3 | 跨架构结果 | Phase E |
| Tab.4 | 消融实验 | Phase F |

---

## 4. 理论贡献形式化

### 需要包含的理论内容

1. **Theorem 1**：给定 distance prior D(Δ)，最优频率分配 f* = argmin E[Phase Collision]
2. **Proposition 1**：uniform D(Δ) → geometric freq 最优（recover standard RoPE）
3. **Proposition 2**：power-law D(Δ) → hybrid freq 结构更优
4. **base ≈ 0.3L 规则**的信息论推导（Nyquist-Shannon 框架）

这部分不需要实验，纯数学推导，可以并行准备。

---

## 5. 时间线

### 假设 NeurIPS 2026 截稿 May 22

```
Week 1  (02-22 ~ 02-28): Phase A 结果 + Phase B 实现
Week 2  (03-01 ~ 03-07): Phase B 跑完 + Phase C 启动
Week 3  (03-08 ~ 03-14): Phase C/D 并行跑
Week 4  (03-15 ~ 03-21): Phase E/F + 理论写作
Week 5  (03-22 ~ 03-28): 论文初稿
Week 6  (03-29 ~ 04-04): 补实验 + 修改
Week 7+ (04-05 ~ 05-22): 迭代打磨 + 内审
```

---

## 6. 概率评估

| 完成阶段 | NeurIPS 接收概率 | 说明 |
|---------|-----------------|------|
| 只有 Phase A（当前 8B） | 20-30% | 规模小、评测少、baseline 少 |
| + Phase B（加 LongRoPE2/CLEX） | 30-40% | 有说服力的 SOTA 对比 |
| + Phase C（RULER + LongBench） | 40-50% | 评测全面，不被挑评测问题 |
| + Phase D（13B + 70B eval） | **50-60%** | 规模够了，跨规模一致性 |
| + Phase E（Mistral 跨架构） | **55-65%** | 通用性验证完整 |
| + 理论形式化 | **60-70%** | 有理论深度，不只是 empirical |

> [!TIP]
> **性价比最高的投入**：Phase B（加 SOTA baseline）和 Phase C（RULER + LongBench）。
> 这两个 Phase 完成后概率直接从 20% → 50%。

---

## 7. 多卡使用规划

| 卡数 | 任务 | 耗时 |
|------|------|------|
| 1卡 | Phase A/B（8B LoRA 各方法） | ~2-3 天 |
| 2卡 | Phase D1（13B LoRA） | ~8h |
| 4卡 | Phase D2（70B eval-only） | ~2h |
| 4卡 | Phase D3（8B 全参数） | ~12h |
| 1卡 | Phase E（Mistral/Qwen 7B） | ~1 天 |
| 1卡 | Phase F（消融，可并行 8 方法） | ~2 天 |

**总计**：~7-10 天密集 GPU 使用。如果有 4 卡并行，可压缩到 4 天。

---

## 8. 文件结构

```
neurips_plan/
├── MASTER_PLAN.md          ← 本文件
├── baselines/
│   ├── longrope.py         ← LongRoPE 频率生成器
│   ├── longrope2.py        ← LongRoPE2 进化搜索
│   ├── clex.py             ← CLEX 连续外推
│   ├── ntk_aware.py        ← 动态 NTK
│   └── abf.py              ← Code Llama ABF
├── benchmarks/
│   ├── run_ruler.py        ← RULER 基准评测
│   ├── run_longbench.py    ← LongBench 评测
│   └── run_infinitebench.py
├── analysis/
│   ├── plot_radar.py       ← 论文图表生成
│   ├── plot_niah_heatmap.py
│   └── plot_ppl_curve.py
└── theory/
    └── proofs.tex           ← 理论推导
```
