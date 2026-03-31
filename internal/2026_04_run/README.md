# EVQ-Cosh 2026年四月冲刺工作区

> 目标：NeurIPS 2026 投稿，将论文从 borderline 推到 weak accept
> 论文源码：`../paper/`
> 上次更新：2026-03-31

---

## 给 AI 的快速上下文

这是一篇关于 **RoPE 位置编码频率分配** 的 NeurIPS 2026 论文（EVQ-Cosh）。核心贡献是：用变分法推导出 cosh 密度族作为最优频率分配，给出半解析公式 τ* = λ·d_head/√L（λ≈1.13 由校准得到）。论文实验在 MHA/MLA/progressive training 上均显著优于 Geometric RoPE。

**当前关键瓶颈**：τ* 公式中的 λ 是校准常数而非解析推导，使理论看起来"金玉其外"。3月31日的分析证明了代理泛函内部的精确自洽性（新定理），但 λ 本身无法闭合——它吸收了 softmax transport 的离散修正。

**四月核心策略**：不追求闭合 λ，改为在行文上从"最优公式"转向"原则性框架"叙事，同时补强实验（progressive 3-seed、λ cross-validation table）。

---

## 文件夹结构

```
2026_04_run/
├── README.md                              ← 本文件（总索引）
│
├── docs/                                  ← 所有文档
│   │
│   │  ── 策略与规划 ──
│   ├── 01_四月冲刺计划.md                   ← 实验优先级 + GPU 预算 + 时间线
│   ├── 02_论文行文策略.md                   ← ★ 核心：narrative tension 分析 + 修改方案 ABC
│   │
│   │  ── 理论分析 ──
│   ├── 03_六大理论问题.tex/.pdf             ← 6 个开放问题（已审核修正，可发导师）
│   ├── 04_lambda闭合分析.md                 ← ★ λ 闭合尝试完整报告 + 新定理
│   ├── 05_理论分析报告_0331.md              ← S_χ², collision, surrogate 的数值分析
│   │
│   │  ── 实验指南 ──
│   ├── 06_tau策略决策.md                    ← Progressive training 中 retarget vs delayed
│   ├── 07_P0实验手册_progressive三种子.md   ← 最高优先级实验的具体命令
│   ├── 08_P1P2P3实验清单.md                 ← QuALITY 多种子、LongRoPE2 等
│   │
│   │  ── 参考 ──
│   ├── 09_全项目文件索引.md                  ← 论文各 .tex、数据、脚本的路径映射
│   └── 10_实验盘点报告_0331.md               ← ★ 全量实验审计：已引用 vs 未挖掘
│
├── figs/                                  ← 图表
│   ├── 方法图_v3.pdf                       ← EVQ 方法示意图（最新版，待插入论文）
│   └── 方法图_v3.png
│
├── scripts/                               ← 分析脚本
│   └── generate_theory_problems_pdf.py     ← 编译六大理论问题 PDF 的脚本
│
└── results/                               ← 实验结果（待填充）
```

---

## 四月待办（按优先级）

### P0 — 不做就 reject

| 编号 | 任务 | 文档 | 状态 |
|------|------|------|------|
| P0-1 | 论文 narrative 重构：从"optimal formula"转向"principled framework" | `02_论文行文策略.md` | 待开始 |
| P0-2 | τ* optimality basin 可视化移入正文（展示 λ 偏移 <1% PPL gap） | `02_论文行文策略.md` | 待开始 |
| P0-3 | Progressive training 3-seed 复现 | `07_P0实验手册.md` | 待开始 |

### P1 — 做了加分明显

| 编号 | 任务 | 文档 | 状态 |
|------|------|------|------|
| P1-1 | λ cross-validation table 加入 Appendix | `04_lambda闭合分析.md` | 待开始 |
| P1-2 | 代理自洽性定理写入 Appendix（新发现） | `04_lambda闭合分析.md` | 待开始 |
| P1-3 | QuALITY downstream 多种子 | `08_P1P2P3实验清单.md` | 待开始 |
| P1-4 | tau_diagnostic 消融写入 Appendix（collision-only vs full，数据已有） | `10_实验盘点报告_0331.md` | 待开始 |
| P1-5 | Qwen LongBench 负面结果写入 Limitations（plug-and-play boundary） | `10_实验盘点报告_0331.md` | 待开始 |

### P2 — Nice to have

| 编号 | 任务 | 文档 | 状态 |
|------|------|------|------|
| P2-1 | EVQ + LongRoPE2 > Geo + LongRoPE2 实验（师兄建议） | `08_P1P2P3实验清单.md` | 待开始 |
| P2-2 | 方法图 v3 插入论文 | `figs/方法图_v3.pdf` | 待开始 |
| P2-3 | LLaMA-8B LoRA 4-method 对比表进 Appendix | `10_实验盘点报告_0331.md` | 待开始 |
| P2-4 | Video frequency decomposition 详细表格 | `10_实验盘点报告_0331.md` | 待开始 |

---

## 关键理论结论（截至 3月31日）

1. **λ ≈ 1.13 ± 0.16**（9 配置, CV=14%），由 Phase 16 的 99 次训练得到
2. **新定理**：τ²T₂(τ) + T₁(τ) = τcoth(τ)，证明代理平衡 τ=√(β/α) 是精确解
3. **λ 不可闭合**：它吸收了连续代理到离散物理核的 softmax transport 常数
4. **Q6 公式已修正**：小τ展开的符号和因子原来都写错了
5. **S_χ²(τ) = τ⁴/45 + O(τ⁶)**（精确 Taylor 首项）

---

## 论文源码位置

论文在 `../paper/` 目录，关键文件：

- `sections/03_theory.tex` — 核心理论（Proposition 3, τ* scaling law）
- `sections/05_experiments.tex` — 实验结果
- `appendix/a1_proofs.tex` — 完整证明（surrogate ODE, stiffness, LoRA phase transition）
- 详见 `docs/09_全项目文件索引.md`
