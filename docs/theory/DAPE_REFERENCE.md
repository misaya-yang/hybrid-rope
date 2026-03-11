# DAPE: NeurIPS 2024 中稿模版参考

> **用途**: 作为同类论文的中稿参考，指导实验设计和论文写作的最低要求
> **创建日期**: 2026-03-01
> **论文**: "DAPE: Data-Adaptive Positional Encoding for Length Extrapolation"
> **会议**: NeurIPS 2024 (Poster)
> **arXiv**: 2405.14722

---

## 1. 论文基本信息

| 项目 | DAPE | 我们的论文 |
|------|------|---------|
| 标题 | Data-Adaptive Positional Encoding for Length Extrapolation | RoPE Scaling as a Variational Inverse Problem |
| 会议 | NeurIPS 2024 | 目标 NeurIPS 2026 |
| 核心贡献 | 数据自适应 PE，可动态调整 | 变分逆问题框架 + EVQ 闭式解 + Waterbed 定理 |
| 理论深度 | 低（无变分推导） | **高**（ODE 推导 + 7个 Appendix 证明） |
| 实验规模 | 125M from-scratch | 50M-350M from-scratch + 8B/7B LoRA |

---

## 2. DAPE 的实验设置（我们的参照基准）

### 2.1 模型规模

- **主实验: 125M 参数 from-scratch**
- 训练长度: 128 tokens
- 测试长度: 最长到 8192 tokens（64x 外推）
- 没有任何 7B/13B 模型的 from-scratch 训练

### 2.2 数据集

- Arxiv, Books3, CHE (Chinese)
- 不是 TinyStories（比我们的数据集更严肃）

### 2.3 评估指标

- PPL at multiple lengths
- 下游任务（非 LongBench，是自选的较简单任务）
- **没有 Passkey/NIAH**

### 2.4 Baselines

- Standard PE (learned absolute)
- RoPE
- ALiBi
- Kerple
- FIRE
- 以及它们的 data-adaptive 变体

### 2.5 统计

- 多 seed 运行
- 误差棒报告

---

## 3. DAPE 被接受的关键因素分析

### 3.1 优势（我们也有或更强的）

1. **新颖的视角**: DAPE 提出 PE 应该 data-adaptive → 我们提出 PE 是变分逆问题（**更深刻**）
2. **清晰的框架**: 简单直观 → 我们的框架也清晰，且有完整数学推导（**更强**）
3. **多 baseline 对比**: 6+ 种方法 → 我们 5 种（**可比**）
4. **一致性好的实验**: 训练/评估协议一致 → 我们也有（**可比**）

### 3.2 DAPE 的弱点（我们没有的）

1. **理论浅**: 没有变分推导，没有最优性证明 → **我们在理论上远超**
2. **仅 125M**: 没有 scaling law 证据 → **我们有 50M-350M（+计划 500M）**
3. **没有 waterbed 分析**: 没有解释 trade-off → **我们有完整的 waterbed 理论+验证**
4. **没有跨模型验证**: 仅一个模型架构 → **我们有 Llama + Qwen 两个家族**

---

## 4. 我们相对于 DAPE 的优势总结

| 维度 | DAPE (NeurIPS 2024) | 我们 |
|------|---------------------|------|
| **理论** | 无变分推导 | ODE 推导 + 3 Proposition + Waterbed 定理 + 7 Appendix |
| **闭式解** | 无 | EVQ 公式，单参数 τ，零启发式 |
| **Scaling** | 125M only | 50M → 125M → 350M (→ 500M planned) |
| **跨模型** | 1 architecture | 2 families (Llama-3, Qwen-2.5) |
| **Trade-off 分析** | 无 | Waterbed: 理论预测 + 实验验证 (Table 4-5) |
| **统计** | 多 seed | 多 seed + 双 seed 跨模型 + FDR 校正 |
| **代码行数** | 较复杂 | EVQ 仅 ~10 行 Python |

**结论**: 如果 DAPE 用 125M 和无理论推导就能中 NeurIPS 2024 poster，我们的论文在理论深度和实验广度上都有显著优势。

### 4.1 额外优势：跨方法一致性

我们有一个 DAPE 完全没有的证据维度：**两代不同方法（anchored-sigmoid 和 EVQ）在所有规模上都一致优于 geometric**。这证明改善是频率重分配的结构性优势，而非特定 warp 的偶然结果。这大幅增强了结论的可信度。

---

## 5. 从 DAPE 学到的实验设计准则

### 5.1 必须做到的

- [x] From-scratch 训练（至少一个规模点）
- [x] 多 baseline 对比
- [x] 多长度评估（train length → 外推长度）
- [x] 多 seed 运行
- [x] 误差棒报告

### 5.2 加分项（DAPE 没有但我们可以有）

- [x] Passkey retrieval（PE 论文标准评测，Base of RoPE 用了）
- [x] NIAH（近年标准）
- [ ] Scaling law 图（PPL improvement vs model size）← 750M 已有，可做完整图
- [x] 跨模型族验证
- [ ] SCROLLS task-specific finetuning（见 §5.4）

### 5.3 不需要的

- LongBench **zero-shot**（需要指令跟随能力，454M from-scratch 做不了）
- 标准 NLU benchmarks（PE 论文不需要）
- 真正的超长上下文（128K+）评测（小模型 OOM，且不是 PE 论文的重点）

### 5.4 SCROLLS 下游评测方案（FIRE 路线，已确认可行）

FIRE (ICLR 2024) 用 125M 和 350M 做了 SCROLLS 7 个子任务，方法是 task-specific finetuning（不是 zero-shot 指令跟随）。我们的 454M 比 FIRE 的最大模型还大。

**FIRE 的 SCROLLS 配置**:

| 参数 | 值 |
|------|-----|
| 模型 | Base=125M (12L/12H/768d), Large=350M (24L/16H/768d), head_dim=64 |
| 预训练 | C4, L=2048, AdamW, LR=6e-4(Base)/3e-4(Large), batch=256, 600k steps |
| SCROLLS 微调 | L=8192, LR=1e-5, batch=128, 25k steps, dropout=0.1 |
| 子任务 | Qasper, NarrativeQA, QuALITY, ContractNLI, QMSum, GovReport, SummScreenFD |
| FIRE Large 平均分 | 27.05 (所有 PE 方法中最高) |

**我们的操作方案**:

1. 取 454M 3-seed EVQ checkpoint 和 Geo checkpoint（同一 pretrain recipe）
2. 在 QMSum / GovReport / QuALITY 上分别 finetune（L=8192, LR=1e-5, 25k steps）
3. 比较 EVQ-finetuned vs Geo-finetuned 的 ROUGE/F1
4. PE 是唯一自变量，归因干净

**优先级**: 17c 多 seed > LaTeX 骨架 ≥ SCROLLS finetune

---

## 6. 审稿人可能的对比攻击及回应

| 审稿人问题 | 回应 |
|-----------|------|
| "DAPE 是 data-adaptive 的，你的不是" | EVQ 通过 τ 参数化了距离先验的形状；task-conditional τ 是自然扩展（Future Work） |
| "DAPE 在更严肃的数据集上评测" | 我们的 500M 实验将使用 FineWeb-Edu，比 DAPE 的数据集更大更新 |
| "DAPE 的方法更通用" | EVQ 的优势是理论基础：我们知道为什么它工作，以及何时它不工作（waterbed） |
| "你们模型太小" | DAPE 用 125M 被 NeurIPS 接受；FIRE 用 125M/350M 被 ICLR 接受；我们有 50M–750M 五点 scaling chain，是 PE allocation 文献中最大规模 |
| "没有下游任务" | DAPE 也没有下游照样中了；我们可做 SCROLLS task-specific finetuning（FIRE 路线，454M > FIRE Large 350M），EVQ vs Geo 同 recipe 微调，归因干净 |

---

## 7. 其他 NeurIPS 2024 PE 论文参考

### Base of RoPE (NeurIPS 2024)
- 2B from-scratch + 7B 微调
- 关键发现：PPL 低 ≠ 检索好 → 必须加合成检索任务
- 我们的学习：**Passkey 评测必须加**

### CREAM (NeurIPS 2024)
- 7B 微调 LLaMA-2
- 有 Passkey + LongBench
- 实验规模比我们大但理论比我们弱

### YaRN (ICLR 2024)
- 7B/13B 微调
- 是我们论文中直接比较的 baseline 之一
- 我们在 Appendix I 中展示 YaRN 的隐含先验与 TinyStories 不匹配
