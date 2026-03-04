# CHE (Chomsky Hierarchy Evaluation) 实验计划

> **创建时间**：2026-03-04
> **状态**：紧急 — Reviewer 生死线，必须有这个 benchmark
> **动机**：DAPE (NeurIPS 2024) 用 CHE 评测 PE 方法的 length generalization，是 PE 论文的标准 benchmark。我们必须跑并证明 EVQ 在结构化任务上也优于 Geometric。
> **参考**：Delétang et al. (ICLR 2023) "Neural Networks and the Chomsky Hierarchy"; DAPE (NeurIPS 2024) arXiv:2405.14722

---

## 1. CHE Benchmark 概述

CHE 来自 Delétang et al. (2022)，包含 15 个按 Chomsky 层级分类的序列预测任务。
DAPE 论文使用的 CHE 配置：

| 项 | 值 |
|----|-----|
| 任务数 | 15（覆盖 Regular / DCF / CS） |
| 训练长度 | 40 |
| 训练步数 | 200K steps |
| 评估长度 | 41 → 500 |
| 模型 | 5 层, 8 heads, d_model=256 |
| D_DAPE | 64 |
| 最大随机位置 L | 2048 |
| 指标 | Accuracy（随机基线：多数任务 50%，部分 20%） |
| Baselines | NoPE, RoPE, T5 Bias, ALiBi, Kerple, FIRE, CoPE, DAPE-* |

### 1.1 任务列表（按 Chomsky 层级）

**Regular (R)** — Finite-State Automata 可解：
1. Even Pairs
2. Parity Check ‡ (permutation-invariant)
3. Cycle Navigation
4. Modular Arithmetic (Simple)
5. Modular Arithmetic

**Deterministic Context-Free (DCF)** — 需要确定性栈：
6. Reverse String
7. Stack Manipulation
8. Solve Equation

**Context-Free (CF)**：
9. Missing Palindrome / Equal Repeats
10. NDStack Manipulation / Compare Occurrence

**Context-Sensitive (CS)** — 需要有界带：
11. Bucket Sort ‡ (permutation-invariant)
12. Binary Addition
13. Duplicate String
14. Interlocked Pairing / Odds First
15. Divide by 2 / Binary Multiplication

> ‡ = permutation-invariant task，PE 信息理论上不影响

### 1.2 为什么这对我们是生死线

- **DAPE 是 NeurIPS 2024 accepted paper**，reviewer 100% 会拿来对比
- DAPE 展示了自适应 PE 在 CHE 上的优势（permutation-variant tasks 上 beat all static PE）
- 如果我们不跑 CHE：reviewer 会说 "只跑了 PPL 和 passkey，没有结构化泛化评估"
- 如果我们跑了 CHE 且 EVQ ≥ RoPE-Geo：reviewer 无话可说
- 如果 EVQ < RoPE-Geo：我们需要知道并做 damage control（例如 argue CHE 是 permutation-invariant 所以不区分）

---

## 2. 实验设计

### 2.1 核心原则

- **严格复现 DAPE 设置**：用他们的 model config（5L/8H/d256），不用我们的 350M，确保苹果对苹果
- **额外加 125M 和 350M**：展示 scale consistency
- **Multi-seed**：每个 (task, method, model) 跑 3 seeds

### 2.2 模型配置

| 配置名 | Layers | Heads | d_model | d_head | FFN | Params | 用途 |
|--------|--------|-------|---------|--------|-----|--------|------|
| CHE-Small (DAPE复现) | 5 | 8 | 256 | 32 | 1024 | ~5M | 与 DAPE 直接对比 |
| 125M | 12 | 12 | 768 | 64 | 3072 | ~125M | Scale consistency |
| 350M | 24 | 16 | 1024 | 64 | 4096 | ~350M | Scale consistency |

### 2.3 PE 方法

| Method | 类型 | 配置 |
|--------|------|------|
| Geo (RoPE baseline) | trained | base=10000 (标准 RoPE) |
| EVQ τ=optimal | trained | τ 根据 τ*=d_head/√L_train 计算 |
| NoPE | trained | 无位置编码 |

> 对 CHE-Small (d_head=32, L_train=40): τ* = 32/√40 ≈ 5.06
> 对 125M (d_head=64, L_train=40): τ* = 64/√40 ≈ 10.12
> 对 350M (d_head=64, L_train=40): τ* = 64/√40 ≈ 10.12
>
> **注意**：L_train=40 对我们的 τ* 公式是极端小值，可能需要 τ-sweep {2, 5, 10, 15}

### 2.4 训练配置

| 项 | 值 | 备注 |
|----|-----|------|
| 训练长度 | 40 | 与 DAPE 一致 |
| 训练步数 | 200K | 与 DAPE 一致 |
| Batch size | TBD | 查 DAPE repo，推测 64-128 |
| Optimizer | AdamW | 标准 |
| LR | TBD | 查 DAPE repo |
| Seeds | {42, 123, 7} | 与我们其他实验一致 |

### 2.5 评估配置

| 项 | 值 |
|----|-----|
| 评估长度范围 | 41 → 500（步长可 discretize: 50, 100, 200, 300, 500） |
| 指标 | Accuracy (%) |
| 每个长度的 test 样本数 | TBD（查 DAPE repo，推测 1000） |

### 2.6 优先级排序

**Phase 1 — 最高优先级（立即跑）**：
- CHE-Small × {Geo, EVQ} × {seed 42, 123, 7} × 15 tasks
- 共 90 runs，每个 ~200K steps on small model → 估计每个 <30min on 5090
- 总时间：~45 GPU-hours（可多任务并行）

**Phase 2 — 高优先级**：
- 125M × {Geo, EVQ} × {seed 42, 123, 7} × 15 tasks（选 top-5 最有区分度的 tasks）
- 350M × {Geo, EVQ} × {seed 42, 123, 7} × top-5 tasks

**Phase 3 — 可选**：
- + NoPE baseline（作为 lower bound）
- + YaRN/NTK-aware inference-time combination（如果 EVQ 在 CHE 上也有超线性效应 → 论文加分项）

---

## 3. 实施路径

### 3.1 代码依赖

```
# DAPE 官方代码
git clone https://github.com/chuanyang-Zheng/DAPE.git

# 原始 CHE benchmark
git clone https://github.com/google-deepmind/neural_networks_chomsky_hierarchy.git
```

### 3.2 需要做的修改

1. **集成 EVQ 频率到 DAPE 的训练框架**：
   - DAPE repo 已有 RoPE/ALiBi/Kerple/FIRE baseline
   - 需要在其 PE 模块中添加 EVQ φ_k(τ) = 1 - (1/τ)arcsinh((1-u_k)sinhτ) 计算
   - 或直接在我们自己的代码库中实现 CHE data loader

2. **CHE Data Loader**：
   - 从 `neural_networks_chomsky_hierarchy` repo 获取任务生成器
   - 需要把 JAX/DeepMind 格式转为 PyTorch DataLoader
   - 或直接用 DAPE repo 中已有的 PyTorch CHE 实现

3. **评估脚本**：
   - 生成 eval_len ∈ {50, 100, 200, 300, 500} 的测试集
   - 计算 per-task accuracy
   - 汇总为 Chomsky 层级平均 accuracy

### 3.3 预计时间线

| 阶段 | 时间 | 内容 |
|------|------|------|
| Day 0 (今天) | 2-3h | Clone repos, 理解 DAPE CHE 代码, 集成 EVQ |
| Day 0-1 | overnight | 启动 CHE-Small Phase 1 (90 runs) |
| Day 1 | 数据分析 | 出 CHE-Small 结果, 判断方向 |
| Day 1-2 | 按需 | 125M/350M Phase 2 |

---

## 4. 预期结果与应对策略

### 4.1 最佳情况：EVQ > Geo on most tasks
- 直接加入论文 Table/Figure
- 特别是 permutation-variant tasks（Reverse, Cycle Navigation 等）EVQ 应该受益

### 4.2 中性情况：EVQ ≈ Geo
- 说明 EVQ 的频率重分配在结构化任务上不损害性能
- 配合 PPL/passkey/retrieval 优势仍然是加分

### 4.3 最差情况：EVQ < Geo on some tasks
- 分析：是否集中在 permutation-invariant tasks（PE 本身无关）？
- 论文中 argue：EVQ 优化了 extrapolation regime, CHE 的 regular tasks 在训练长度内就能解，不是 EVQ 的目标场景
- Damage control：只报 extrapolation regime 的结果（eval_len >> train_len）

### 4.4 关键看点

- **Length generalization curve**：不是看 eval@40 的 accuracy，而是看 eval@500 vs eval@40 的**衰减速度**
- EVQ 的价值在于长程外推 → 如果 EVQ 在 eval@500 上比 Geo 好，即使 eval@40 差一点也是大胜
- 特别关注 Regular 层级的 permutation-variant tasks（Cycle Navigation, Modular Arithmetic）

---

## 5. 论文整合方案

### 如果结果好 → 正文 Section 5 新增 0.4 页
- 新增 Table: CHE accuracy (Geo vs EVQ) at eval_len=500, 按 Chomsky 层级分组
- 用 Appendix 放完整 15-task × multi-length 结果
- 叙事："EVQ 不仅改善 PPL 和 retrieval，在结构化泛化任务上也保持/改善了 length generalization"

### 如果结果中性/差 → Appendix 一页
- 放在 Appendix 中证明我们做了这个 evaluation
- 解释为什么 CHE 的短序列 regime (L=40→500) 不是 EVQ 的主要优化场景

---

## 6. 参考文献

- Delétang et al. "Neural Networks and the Chomsky Hierarchy" (ICLR 2023): https://arxiv.org/abs/2207.02098
- DAPE: https://arxiv.org/abs/2405.14722
- DAPE GitHub: https://github.com/chuanyang-Zheng/DAPE
- CHE GitHub: https://github.com/google-deepmind/neural_networks_chomsky_hierarchy
