# AI Handoff Prompt: EVQ-Cosh NeurIPS 2026 论文

> **用途**: 粘贴给任何 AI 助手（Claude Code / Claude / GPT / Gemini），让它快速进入项目上下文
> **最后更新**: 2026-03-01
> **使用方法**: 直接说"请读这个文件，然后开始工作"

---

## 项目概述

我们正在准备一篇 NeurIPS 2026 投稿：**"RoPE Scaling as a Variational Inverse Problem: Exact Frequency Allocation and the Waterbed Trade-off"**。

核心贡献：证明 RoPE 频率分配是一个变分逆问题，推导出闭式最优解 EVQ-Cosh（由单参数 τ 控制），并证明 Waterbed 不等式约束了任何分配的性能边界。

## 当前状态

### 已完成的实验 ✅

1. **128-Token PE Quality Test** (FineWeb-Edu, 125M):
   - EVQ fixed τ=1.5 vs Geometric: **-18.3% PPL@8K**
   - EVQ learnable (1 param) vs DAPE (32 params): **EVQ 赢 3-14%**
   - Learnable τ 跨 3 seed 收敛到 **1.141 ± 0.003**
   - 详细数据: `docs/paperdraft/EXPERIMENT_RESULTS_128TOK.md`

2. **TinyStories From-Scratch Scaling** (50M-350M):
   - EVQ τ=1.5 vs Geometric @16K: -10.9% (50M), -18.9% (125M seed42)
   - 数据: `results/paper_ready/evq_tau_sweep/evq_sweep_paper_table.csv`

3. **LoRA Downstream Waterbed** (8B Llama-3, 7B Qwen-2.5):
   - Retrieval +2.50, Multi-hop -2.69（方向完美复现 Waterbed 理论预测）
   - 数据: `results/paper_ready/llama8b_fair_lora_suite_20260214/`

### 已知问题 ⚠️

1. **Algorithm 1 (D(Δ)→τ* 预测器) 失败**: 残差 35-49%，预测值无意义。降级为 Appendix 理论工具。
2. **Learnable τ gap**: τ_learned=1.14 vs τ_sweep=1.5。训练目标 vs 外推目标不一致。
3. **Softplus 死区**: τ_init < 0.1 时梯度消失。建议 init ≥ 0.5 或换 exp 参数化。

### 待完成实验 ⬜

见下方 `PROMPT_NEXT_EXPERIMENTS.md` 部分。

## 文件导航

**写论文时按顺序读这些文件：**

1. `docs/paperdraft/THEORY_IRONCLAD.md` — 理论权威参考
2. `docs/paperdraft/EXPERIMENT_RESULTS_128TOK.md` — 128-tok 实验数据
3. `docs/paperdraft/LATEX_SNIPPETS.md` — 可直接粘贴的 LaTeX 段落
4. `docs/paperdraft/FINAL_ACTION_PLAN.md` — 行动方案 v4
5. `docs/paperdraft/PAPER_FILE_INDEX.md` — 全部文件总索引

**做实验时读：**

6. `docs/paperdraft/EXPERIMENT_AUDIT_V4.md` — 实验设计审核
7. `docs/paperdraft/LEARNABLE_TAU_DESIGN.md` — Learnable τ 设计+结果
8. `rope/learnable_evq.py` — 核心实现代码
9. `scripts/m4_evq_sweep/run_evq_sweep.py` — 训练 pipeline

**论文 LaTeX 源文件：**

10. `submission/paper/hybrid_rope_neurips.tex` — 当前版本

## 论文叙事 (v4)

```
Theory → ODE → cosh 族（N/2 维压缩到 1D）
    → 在 PE-dominant regime 中 learnable τ 收敛（1.141 ± 0.003）
    → EVQ (1 param) 击败 DAPE (32 params)
    → 跨协议/跨数据集 τ=1.5 稳定
    → Waterbed 不等式：理论预测 + 实验验证
```

核心卖点：**理论导出的 1 参数方法优于无理论的 32 参数方法。**

## ⚠️ 致 AI 助手的警告

请勿犯以下错误（本项目历史上曾出现过）：

1. ❌ "cosh 是 ansatz/假设" → 是 ODE 的精确齐次解
2. ❌ "Algorithm 1 能可靠预测 τ" → 不能，残差 35-49%
3. ❌ "Learnable τ 收敛到 sweep 最优" → 收敛到 1.14，sweep 最优是 1.5
4. ❌ "τ→0 退化定理没价值" → 它是统一性声明
5. ❌ "Waterbed 说明 EVQ 没用" → Waterbed 是可检验预测，不是否定
6. ❌ "三重验证成立" → Algorithm 1 失败，只有二重（sweep ≈ learnable 方向）

---

*Handoff 创建: 2026-03-01*
