# Paper Draft Status (论文进度追踪)

> **最后更新**: 2026-02-28
> **论文标题**: "RoPE Scaling as a Variational Inverse Problem: Exact Frequency Allocation and the Waterbed Trade-off"
> **目标会议**: NeurIPS 2026
> **截止日期**: ~9 weeks from now
> **当前版本**: V6
> **编译路径**: `paper_exports/neurips_v6_fig/hybrid_rope_neurips_v6.tex`

---

## 1. 版本演进

| 版本 | 日期 | 主要变化 |
|------|------|----------|
| V1-V3 | ~2026-02-13 | 初始框架：anchored-sigmoid, hybrid 实验 |
| V4 | 2026-02-25 | 加入 cross-model Qwen 验证，tradeoff framing |
| V5 | 2026-02-27 | 全面改写为 EVQ/τ/Waterbed，移除所有 anchored-sigmoid 叙述 |
| **V6** | **2026-02-28** | **理论优先叙事重构；τ-sweep 数据替换；Discussion 新增；NeurIPS 2026 格式** |

---

## 2. 页面预算 (NeurIPS: ≤9 pages main content)

| 部分 | 页数 | 状态 |
|------|------|------|
| Main content (Sec 1-7) | **~9** | ✅ 完成 (含新 Discussion subsection) |
| References | ~1 | ✅ |
| Appendix A-H | ~10 | ✅ |
| **总计** | ~20 | 编译通过，0 errors |

---

## 3. 各 Section 状态

### Main Content

| Section | 标题 | 状态 | 备注 |
|---------|------|------|------|
| 1 | Introduction | ✅ **V6 重写** | 更锐利的 theory-first framing；显式列出现有方法的隐含先验 |
| 2 | Related Work | ✅ **V6 更新** | "How this work differs" 段落扩展，含 implicit priors insight |
| 3.1 | RoPE as frequency allocation | ✅ 完成 | |
| 3.2 | Phase-collision energy | ✅ 完成 | |
| 3.3 | Joint objective with Fisher fidelity | ✅ 完成 | |
| 3.4 | Exact Variational Quantization (EVQ) | ✅ 完成 | |
| 3.5 | Structural theorems | ✅ 完成 | |
| **Figure 1** | **EVQ warp curves** | ✅ **已插入** | 2 panels: warp + deviation |
| 4.1 | Waterbed inequality | ✅ 完成 | |
| 4.2 | Finite-base calibration | ✅ 完成 | |
| 4.3 | Implicit priors of existing methods | ✅ **V6 精简** | 核心内容前移至 Related Work |
| 5.1 | Protocol summary | ✅ 完成 | |
| 5.2 | From-scratch τ-sweep (50M–125M) | ✅ **V6 替换** | τ-sweep 表格 + 三点分析 + 双种子验证 |
| **5.3** | **Discussion: spectral redistribution** | ✅ **V6 新增** | 凸性/边界密度/离散化间隙假说 |
| 5.4 | Controlled-protocol LoRA evaluation | 🟡 **现有数据** | Llama-3-8B 5 schedules + Qwen dual-seed |
| 5.5 | Waterbed trade-off verification | ✅ **V6 重写** | 重新解读：training-regime dependency 而非 intrinsic |
| 6 | Limitations | ✅ **V6 更新** | training-regime dependency + 规模覆盖限制 |
| 7 | Conclusion | ✅ **V6 重写** | 理论优先 + 两个实验 regime 的对比叙事 |

### Appendices

| Appendix | 标题 | 状态 |
|----------|------|------|
| A | Proofs of Theorems 1 and 2 | ✅ |
| B | Proof of Proposition 1 (uniform prior) | ✅ |
| C | Proof of Proposition 2 (power-law prior) | ✅ |
| D | Proof of Proposition 3 (resonance condition) | ✅ |
| E | Diagonal approximation and residual analysis | ✅ |
| F | Structural stability beyond diagonal | ✅ |
| G | Non-asymptotic and discretization guarantees | ✅ |
| H | Experimental Hyperparameter Configurations | ✅ |

---

## 4. 关键缺口 (Critical Gaps)

### ~~Gap 1: 实验数据与理论框架不匹配~~ ✅ 已解决

V6 已将 τ-sweep 数据写入 Section 5.2。

### Gap 2: 8B 大模型 from-scratch / LongInst 验证

**问题**: 当前 8B 数据仅来自短步 WikiText LoRA。

**需要的数据**:
- [ ] Llama-3-8B EVQ τ=1.5 + LongInst 数据 + 800步，LongBench-21 分数
- [ ] NIAH heatmap
- [ ] Passkey 准确率

**状态**: 实验脚本已就绪 (`run_llama8k_theory_v1.py`)，数据集正在准备中。

### Gap 3: Figure 2 (PPL vs τ 曲线)

**问题**: 论文只有 1 张图 (EVQ warp curves)，空间允许再加 1 张。

**候选**: PPL vs τ 曲线图 — 数据已就绪，可用 `scripts/m4_evq_sweep/evq_analysis.py` 生成。

---

## 5. 编译与导出

```bash
# 编译 V6 论文
cd paper_exports/neurips_v6_fig/
pdflatex hybrid_rope_neurips_v6.tex
pdflatex hybrid_rope_neurips_v6.tex  # 两次以生成引用

# 生成 Figure 1
python plot_evq_warp_v2.py
# → evq_warp_curves.pdf / .png
```

---

## 6. V5→V6 变更摘要

| 变更 | 说明 |
|------|------|
| Abstract | 从 waterbed-defensive framing 改为 theory-first positive framing |
| Introduction | 更锐利的开头：显式列出 PI/YaRN/NTK 的隐含先验，指出它们都没有被推导 |
| Related Work | "How this work differs" 扩展，含 implicit priors analysis |
| Section 4.3 | 精简（核心内容前移至 RW） |
| Section 5.2 | 旧 TinyStories 3-scale 表 → 新 τ-sweep 表（50M/125M, 双种子） |
| Section 5.3 (新) | Discussion: convexity → φ<u → Fisher increase, boundary density ratio, discretization gap hypothesis |
| Section 5.5 | Waterbed 解读从 "intrinsic zero-sum" 改为 "training-regime dependency" |
| Limitations | 删除 "not defeatable"，新增 from-scratch scale coverage limitation |
| Conclusion | 理论优先叙事 + from-scratch vs LoRA 两个 regime 对比 |
| NeurIPS year | neurips_2025 → neurips_2026 |

---

## 7. 更新日志

| 日期 | 操作 | 操作者 |
|------|------|--------|
| 2026-02-27 | 创建本文档；V5 完成编译 | Claude (Cowork) |
| 2026-02-27 | Figure 1 插入 | Claude (Cowork) |
| 2026-02-27 | Limitations/Conclusion/Appendix 改写 | Claude (Cowork) |
| 2026-02-27 | Gap 1 解决: τ-sweep 数据 Paper-ready, 双种子验证通过 | Claude (Cowork) |
| **2026-02-28** | **V6 全面升级: 叙事重构 + τ-sweep 数据写入 + Discussion 新增** | **Claude (Cowork)** |
