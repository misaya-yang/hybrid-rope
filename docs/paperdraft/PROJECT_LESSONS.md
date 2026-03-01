# 项目复盘与决策记录

> **用途**: 记录项目关键决策和教训，防止后续 AI 或协作者重复犯错
> **创建日期**: 2026-03-01

---

## 1. 最大的教训：LoRA 微调是一条死路

### 1.1 时间线

| 时间 | 事件 | 花费 |
|------|------|------|
| 2月初 | 50M/125M/350M TinyStories from-scratch 实验 → **成功** | 低（单卡数天） |
| 2月中 | 转向 Llama-3-8B LoRA 微调，试图展示大模型效果 | ~¥400-500 |
| 2月14日 | A1(geometric) LoRA 成功，A2(EVQ τ=1.5) LoRA **崩溃** | |
| 2月下旬 | 尝试修复 A2 崩溃（降 τ、调 rank、调 lr）→ 全部失败 | ~¥200-300 |
| 2月28日 | 最终结论：LoRA 范式与 EVQ 不兼容 → **回归 from-scratch** | |

**总计浪费约 ¥700-800，约 1.5-2 周时间。**

### 1.2 崩溃根因分析

EVQ τ=1.5 对预训练模型的频率分配产生了 5x 量级的中频段扰动。LoRA rank-32（仅占总参数 0.1%）无法桥接如此大的频率流形差异。

**"不可能三角"**：
- LoRA 太轻 → 学不会新频率分配
- LoRA 太重 → 灾难性遗忘
- 无法隔离"频率感知能力"这一单独维度

### 1.3 正确的认知

EVQ 是一个 **design-time principle**（设计时原则），不是 **post-hoc extension**（事后扩展）。它应该从模型训练第一步就介入，而不是通过微调注入到已有模型中。

**类比**: 就像你不能通过微调把 ALiBi 注入到用 RoPE 训练的模型里一样，EVQ 的非线性频率重映射也需要从头训练。

### 1.4 但 LoRA 实验并非完全浪费

Llama-3-8B 和 Qwen-2.5-7B 的 **受控协议 LoRA 评估**（使用共享 inv_freq.copy() 注入、无 rope_scaling API）实际上提供了：
- 跨模型族的 waterbed trade-off 验证
- 5 种 schedule 的公平对比
- 双 seed 复现性证据

这些作为"受控验证实验"写进了论文 Section 5.3-5.5，**是论文的重要组成部分**。

关键区别：
- ❌ 试图用 LoRA 让 EVQ 在预训练模型上"工作" → 失败
- ✅ 用 LoRA 作为受控协议验证 waterbed trade-off 的结构性预测 → 成功

---

## 2. 50M/125M/350M from-scratch 从一开始就是正确路径

### 2.1 为什么

1. **DAPE 先例**: NeurIPS 2024 仅用 125M from-scratch 就被接受
2. **控制变量**: from-scratch 训练中 RoPE schedule 是唯一差异
3. **Scaling law 证据**: 50M → 125M → 350M → 500M 展示趋势
4. **理论验证**: from-scratch 直接验证 frequency allocation 的效果，无预训练 confound

### 2.2 已有数据：两代方法全部赢了

**第一代：Anchored-Sigmoid（Legacy，但数据有效）**

| 模型 | Geometric PPL@16K | Anchored-Sigmoid PPL@16K | Δ |
|------|-------------------|--------------------------|---|
| 50M (3 seeds) | 19.386 ± 2.007 | 17.404 ± 1.564 | **-10.1%** |
| 100M | 10.888 | 9.417 | **-13.5%** |
| 350M | 14.653 | 12.651 | **-13.7%** |

**第二代：EVQ τ-sweep（Current，论文主线）**

| 模型 | Geometric PPL@16K | EVQ (τ=1.5) PPL@16K | Δ |
|------|-------------------|---------------------|---|
| 50M | 33.316 | 29.697 | **-10.9%** |
| 125M (seed42) | 34.153 | 27.699 | **-18.9%** |
| 125M (seed137) | 28.502 | 26.860 | **-5.8%** |

### 2.3 核心发现：改善是结构性的

两种完全不同的 warp 函数（sigmoid-based vs cosh-based），在 5 个不同的（规模×种子）组合上，**全部一致地优于 geometric**。这不是偶然。

这意味着 500M FineWeb-Edu 实验的方向性结果是高度可预测的——**重训一定赢**。唯一的变量是赢多少。

### 2.4 TinyStories 的隐患

TinyStories 是合法的 NeurIPS 数据集（DAPE 没有用但 Base of RoPE 也没有用 LongBench），但可能被审稿人攻击为：
- 语法结构太简单
- 距离分布可能偏向短距离
- 不代表真实 LLM 训练场景

**解决方案**: 500M FineWeb-Edu 实验堵住这个攻击面。

---

## 3. 关键决策记录

### 3.1 数据集选择

| 决策 | 选项 | 最终选择 | 原因 |
|------|------|---------|------|
| From-scratch 数据 | TinyStories / FineWeb-Edu / SlimPajama | TinyStories (已完成) + FineWeb-Edu sample-10BT (500M 实验) | TinyStories 已有数据；FineWeb-Edu 更强但需要下载 |
| LoRA 数据 | WikiText-style | WikiText-style | 与 baseline 论文一致 |

### 3.2 评估指标选择

| 指标 | 是否必须 | 原因 |
|------|---------|------|
| PPL@多长度 | ✅ 必须 | 最基础的 PE 评测 |
| Passkey retrieval | ✅ 必须 | 纯测位置编码质量，不依赖语言能力 |
| NIAH | ✅ 推荐 | PE 论文标准评测 |
| LongBench | ❌ 非必须 | 需要指令跟随能力，from-scratch 小模型做不到 |

### 3.3 Baseline 选择

| 方法 | 论文中比较 | 500M 实验中比较 |
|------|----------|---------------|
| Geometric (τ=0) | ✅ | ✅ |
| PI | ✅ (8B LoRA) | ✅ (推理时零成本) |
| YaRN | ✅ (8B LoRA) | ❌ (预算不够) |
| Diagonal-only | ✅ (8B LoRA) | ❌ |
| EVQ (τ>0) | ✅ | ✅ |

---

## 4. 下一步优先级

### P0: 500M FineWeb-Edu 实验
- 配置已就绪: `docs/exp/prompt_500m_experiment.md`
- 脚本已修改: `scripts/m4_evq_sweep/run_evq_sweep.py` (500m tier)
- 数据: FineWeb-Edu sample-10BT (~10B tokens, 28.5GB)
- 评估: PPL@多长度 + Passkey (teacher-forcing NLL gap) + 推理时 PI baseline

### P1: 论文最终修改
- 500M 数据写入 Section 5.2
- Figure 2: PPL vs τ 曲线图
- Passkey 结果写入新 Section

### P2: Camera-ready 准备
- 开源代码和数据
- 许可证声明
- Reproducibility checklist 完善

---

## 5. 预算跟踪

| 项目 | 预估花费 | 实际花费 | 备注 |
|------|---------|---------|------|
| 50M/125M/350M TinyStories | ~¥100-200 | ~¥150 | 已完成 |
| Llama-3-8B LoRA 各种尝试 | 预算外 | ~¥700-800 | ⚠️ 大部分浪费 |
| Qwen-2.5-7B 双 seed LoRA | ~¥200 | ~¥200 | 已完成，论文需要 |
| 500M FineWeb-Edu (计划) | ~¥100-200 | TBD | 单卡 RTX PRO 6000 |
| **总计** | | ~¥1200+ | |
