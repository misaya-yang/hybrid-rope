# NeurIPS 2026 终极路线图：不再烧钱版

> **创建日期**: 2026-03-01
> **核心原则**: 以中稿为唯一目标，不做任何非必要实验
> **预算上限**: 500M 实验 ≤ ¥200（单卡 RTX PRO 6000，~2天）

---

## ⛔ 红线：以下实验不做

| 诱惑 | 为什么不做 | 替代方案 |
|------|----------|---------|
| 1B from-scratch | 太贵太慢（5天+），且 DAPE 用 125M 就中了 | 500M 足够 |
| 8B from-scratch | 不可能（70天+） | LoRA 数据已有 |
| 更多 LoRA 微调 | 已证明 EVQ+LoRA 不兼容，已浪费 ¥700+ | 已有数据够用 |
| LongBench on from-scratch | 小模型没有指令跟随能力 | PPL + Passkey |
| 额外数据集（C4, The Pile 等） | FineWeb-Edu 已足够，不值得数据工程 | FineWeb-Edu sample-10BT |
| Layer-wise EVQ | Future work，不是 V1 论文该做的 | 写进 Conclusion |
| 更多 base 值实验 | Nice to have，非必须 | 已有 b=10⁴ 和 b=10⁶ |

---

## ✅ 论文主线（定稿版）

### 叙事弧线（一句话版）

> 我们把 RoPE 频率分配公式化为变分逆问题，推导出精确解 EVQ-Cosh，
> 证明所有频率重分配都受 Waterbed 不等式约束，
> 并在 50M-500M from-scratch + 8B/7B LoRA 上验证了理论预测。

### 详细叙事节奏

| Section | 内容 | 核心论点 | 状态 |
|---------|------|---------|------|
| §1 Intro | 现有方法都是启发式的，没人知道为什么 | 我们提供 theory-first 答案 | ✅ |
| §2 Related Work | PI/YaRN/NTK 各自隐含什么先验 | 统一视角 | ✅ |
| §3.1-3.3 | Kernel 分解 → 联合泛函 → ODE | 数学推导 | ✅ |
| §3.4 | ODE 精确解 → EVQ warp 公式 | **Theorem 1** | ✅ |
| §3.5 | Uniform/Power-law/Bimodal 结构定理 | **Prop 1-3** | ✅ |
| §4.1 | Waterbed 不等式 | 不可能同时改善所有距离 | ✅ |
| §4.2 | 有限 base 校准 | 理论≠空谈，数值验证 | ✅ |
| §5.1 | 实验协议 | 两条线：from-scratch + LoRA | ✅ |
| §5.2 | **From-scratch scaling (50M→500M)** | EVQ 一致优于 geometric | 🔴 需要 500M |
| §5.3 | **Passkey retrieval** | 纯位置编码质量测试 | 🔴 需要 500M |
| §5.4 | LoRA 受控评估 (Llama+Qwen) | 跨模型 waterbed 验证 | ✅ |
| §5.5 | Waterbed trade-off 实证 | Table 4-5 方向性验证 | ✅ |
| §6 | Limitations | 诚实：broadband 近似、规模、trade-off | ✅ |
| §7 | Conclusion | Task-aware scheduling 是方向 | ✅ |

### 微调实验在论文中的正确定位

**微调不是用来"证明 EVQ 好用"的，而是用来"验证 Waterbed 的结构性预测"的。**

叙事逻辑：
1. From-scratch 实验证明 → EVQ 确实改善长上下文 PPL（§5.2）
2. 但理论预测不可能所有任务都改善（Waterbed, §4.1）
3. LoRA 实验在真实大模型上验证了这个预测（§5.4-5.5）
4. Retrieval ↑ + Multi-hop ↓ = 结构化 trade-off（Table 4-5）
5. 这恰恰是理论的胜利，不是方法的失败

**因此微调实验有意义，但意义不是"EVQ 在 8B 上也好用"，而是"Waterbed 理论的实证验证"。**

---

## 🔴 唯一剩余实验：500M FineWeb-Edu

### 为什么必须做

| 攻击面 | 没有 500M 时 | 有了 500M 后 |
|--------|------------|------------|
| "数据集太简单" | TinyStories 确实简单 | FineWeb-Edu 是 web 文本 |
| "模型太小" | 350M 是最大 | 500M，且有 scaling 趋势 |
| "没有 Passkey" | 只有 PPL | PPL + Passkey 双指标 |
| "只有一个数据集" | TinyStories only | TinyStories + FineWeb-Edu |

### 实验规格（不可更改）

| 参数 | 值 | 原因 |
|------|---|------|
| 模型 | 500M (d=1024, L=28, H=16) | 比 350M 大一个台阶 |
| 数据 | FineWeb-Edu sample-10BT | 严肃 web 文本，~10B tokens |
| 训练量 | 500M tokens | 预算 regime |
| Schedule | geometric (τ=0) vs EVQ (τ=1.5) | 只比较这两个 |
| 评估 | PPL@2K/4K/8K/16K + Passkey@2K-16K | 双指标 |
| Seed | 42 | 单 seed（预算限制，50M 已有 3-seed） |
| 额外 baseline | PI (推理时零成本) | 不需要额外训练 |
| 训练时间 | ~1-2 天 (RTX PRO 6000) | |
| Prompt | `docs/exp/prompt_500m_experiment.md` | 已定稿 |

### 预期结果

基于 5 个已有数据点（两代方法×三个规模），PPL@16K 改善预期 **10-20%**。Passkey 预期 EVQ 在更长上下文下保持更高准确率。

### 产出物

1. 写入论文 Table 2（添加 500M 行）
2. 新增 Passkey 结果表
3. 可选：PPL vs τ 曲线图（Figure 2）

---

## 📊 论文中的表格/图清单

| 编号 | 内容 | 状态 | 数据来源 |
|------|------|------|---------|
| Table 1 | 有限 base 校准 R² | ✅ | 理论计算 |
| Table 2 | From-scratch scaling PPL | 🔴 需加 500M | results/paper_ready/ |
| Table 3 | Llama-3-8B 5 schedules | ✅ | LoRA 实验 |
| Table 4 | Qwen-2.5-7B 双 seed 总分 | ✅ | artifacts/reviewer_2026-02-25/ |
| Table 5 | Qwen task-family 分解 | ✅ | 同上 |
| Table 6-9 | Appendix 配置表 | ✅ | 手动 |
| Fig 1 | EVQ warp curves | ✅ | plot_evq_warp_v2.py |
| Fig 2 | PPL vs τ 曲线（可选） | 🟡 500M 后可做 | evq_analysis.py |
| **NEW** | Passkey 结果 | 🔴 需要 500M | 500M 实验 |

---

## 📅 时间线

| 日期 | 里程碑 | 备注 |
|------|--------|------|
| 3月1日 | paperdraft 铁桶文档完成 | ✅ 本次完成 |
| 3月1-2日 | 500M 实验代码校验 + 数据下载 | Claude Code 执行 prompt |
| 3月3-5日 | 500M geometric + EVQ 训练 | RTX PRO 6000 |
| 3月5-6日 | Passkey + PI baseline 评估 | 推理 only |
| 3月7日 | 数据写入论文 | Table 2 + Passkey 表 |
| 3月8-15日 | 论文 final polish | 图表美化、语言润色 |
| 3月下旬 | 导师 review | |
| 4月 | 修改 + 校对 | |
| 5月中 | **提交 NeurIPS 2026** | |

---

## 🗂️ 仓库整理计划

```
hybrid-rope/
├── submission/           ← 📦 干净投稿包（可复现代码+数据+论文）
├── docs/paperdraft/      ← 📋 铁桶文档（理论+决策+路线图）
├── paper_exports/        ← 📄 LaTeX 源
├── rope/                 ← 🔧 核心库代码
├── train.py              ← 🔧 训练入口
├── scripts/m4_evq_sweep/ ← 🔧 主实验脚本
├── knowledge_base/       ← 📖 参考（以 paperdraft 为准）
├── artifacts/            ← 📊 LoRA 实验数据
├── results/paper_ready/  ← 📊 From-scratch 数据
├── results/350m_final/   ← 📊 350M 数据
└── _archive/             ← 🗄️ 归档（所有其他东西）
```

**所有非上述路径的东西都应该移到 `_archive/`。**
