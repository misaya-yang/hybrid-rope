# Paper Draft Status (论文进度追踪)

> **最后更新**: 2026-02-27
> **论文标题**: "RoPE Frequency Allocation as a Variational Inverse Problem"
> **目标会议**: NeurIPS 2026
> **截止日期**: ~9 weeks from now
> **当前版本**: V5
> **编译路径**: `paper_exports/neurips_v5_fig/hybrid_rope_neurips_v5.tex`

---

## 1. 版本演进

| 版本 | 日期 | 主要变化 |
|------|------|----------|
| V1-V3 | ~2026-02-13 | 初始框架：anchored-sigmoid, hybrid 实验 |
| V4 | 2026-02-25 | 加入 cross-model Qwen 验证，tradeoff framing |
| **V5** | **2026-02-27** | **全面改写为 EVQ/τ/Waterbed**，移除所有 anchored-sigmoid 叙述 |

---

## 2. 页面预算 (NeurIPS: ≤9 pages main content)

| 部分 | 页数 | 状态 |
|------|------|------|
| Main content (Sec 1-7) | **8** | ✅ 完成 |
| **剩余空间** | **1** | ⏳ 留给 τ-sweep 结果表 |
| References | ~1 | ✅ |
| Appendix A-H | ~10 | ✅ |
| **总计** | ~19 | 编译通过，0 warnings |

---

## 3. 各 Section 状态

### Main Content

| Section | 标题 | 状态 | 备注 |
|---------|------|------|------|
| 1 | Introduction | ✅ 完成 | EVQ/τ framing |
| 2 | Related Work | ✅ 完成 | |
| 3.1 | RoPE as frequency allocation | ✅ 完成 | |
| 3.2 | Phase-collision energy | ✅ 完成 | |
| 3.3 | Joint objective with Fisher fidelity | ✅ 完成 | |
| 3.4 | Exact Variational Quantization (EVQ) | ✅ 完成 | |
| 3.5 | Structural theorems | ✅ 完成 | |
| **Figure 1** | **EVQ warp curves** | ✅ **已插入** | 2 panels: warp + deviation |
| 4.1 | Waterbed inequality | ✅ 完成 | |
| 4.2 | Finite-base calibration | ✅ 完成 | |
| 4.3 | Implicit priors of existing methods | ✅ 完成 | |
| 5.1 | Protocol summary | ✅ 完成 | |
| 5.2 | From-scratch TinyStories scaling | 🟡 **数据已就绪** | EVQ τ-sweep 50M+125M 数据已 Paper-ready，待替换 LaTeX |
| 5.3 | Controlled-protocol LoRA evaluation | 🔴 **待 8B 数据** | 需 EXP_EVQ_8B_LONGINST 结果 |
| 5.4 | Waterbed trade-off verification | 🟡 **数据已就绪** | 125M 双种子验证: 无 waterbed, 待替换 LaTeX |
| 6 | Limitations | ✅ 完成 | EVQ framing |
| 7 | Conclusion | ✅ 完成 | EVQ framing |

### Appendices

| Appendix | 标题 | 状态 |
|----------|------|------|
| A | Proofs of Theorems 1 and 2 | ✅ 新增 |
| B | Proof of Proposition 1 (uniform prior) | ✅ |
| C | Proof of Proposition 2 (power-law prior) | ✅ |
| D | Proof of Proposition 3 (resonance condition) | ✅ |
| E | Diagonal approximation and residual analysis | ✅ |
| F | Structural stability beyond diagonal | ✅ |
| G | Non-asymptotic and discretization guarantees | ✅ |
| H | Experimental Hyperparameter Configurations | ✅ 已更新为 EVQ 名 |

---

## 4. 🔴 关键缺口 (Critical Gaps)

### ~~Gap 1: 实验数据与理论框架不匹配~~ ✅ 已解决

**数据已就绪** (2026-02-27): 5090 τ-sweep 完成，双种子交叉验证通过。

**已有数据**:
- [x] 50M τ-sweep PPL 表 (τ = 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0) — `results/paper_ready/evq_tau_sweep/50m_seed42.json`
- [x] 125M τ-sweep PPL 表 (τ = 0.0, 0.2, 1.5, cross-seed) — `results/paper_ready/evq_tau_sweep/125m_seed{42,137}.json`
- [x] Waterbed 验证: 双种子均无 waterbed 效应
- [x] Phase collision vs τ: τ=1.5 最低 (0.268)

**待做**: 将数据写入 LaTeX Section 5.2 和 5.4 替换旧占位数据。
**CSV**: `results/paper_ready/evq_tau_sweep/evq_sweep_paper_table.csv`

### Gap 2: 8B 大模型验证

**问题**: 无 8B 模型使用 EVQ 的实验结果。

**解决方案**: τ=1.5 已确认为最优值，longinst 管线已优化就绪。

**需要的数据**:
- [ ] Llama-3-8B EVQ τ=1.5 vs baseline，LongBench-21 分数
- [ ] NIAH heatmap
- [ ] Passkey 准确率

### Gap 3: Figure 2

**问题**: 论文只有 1 张图 (EVQ warp curves)，空间允许再加 1 张。

**候选**: PPL vs τ 曲线图 — 数据已就绪，可用 `scripts/m4_evq_sweep/evq_analysis.py` 生成。

---

## 5. 编译与导出

```bash
# 编译论文
cd paper_exports/neurips_v5_fig/
pdflatex hybrid_rope_neurips_v5.tex
pdflatex hybrid_rope_neurips_v5.tex  # 两次以生成引用

# 生成 Figure 1
python plot_evq_warp_v2.py
# → evq_warp_curves.pdf / .png

# 生成 τ-sweep 分析图 (待数据就绪)
python scripts/m4_evq_sweep/evq_analysis.py --input <results_final.json>
```

---

## 6. 更新日志

| 日期 | 操作 | 操作者 |
|------|------|--------|
| 2026-02-27 | 创建本文档；V5 完成编译 | Claude (Cowork) |
| 2026-02-27 | Figure 1 插入 | Claude (Cowork) |
| 2026-02-27 | Limitations/Conclusion/Appendix 改写 | Claude (Cowork) |
| 2026-02-27 | Gap 1 解决: τ-sweep 数据 Paper-ready, 双种子验证通过 | Claude (Cowork) |
