# paperdraft/ 文件夹说明

> **目标会议**: NeurIPS 2026
> **论文标题**: "RoPE Scaling as a Variational Inverse Problem: Exact Frequency Allocation and the Waterbed Trade-off"
> **截止日期**: ~2026年5月中旬

---

## 文档清单

| 文件 | 用途 | 优先级 |
|------|------|--------|
| **THEORY_IRONCLAD.md** | 理论体系的绝对权威参考，含推导链条、定理证明要点、审稿防御速查 | 🔴 最高 |
| **PROJECT_LESSONS.md** | 项目决策记录和教训（尤其是 LoRA 弯路），含预算和时间线 | 🟡 高 |
| **DAPE_REFERENCE.md** | NeurIPS 2024 中稿论文 DAPE 的模版参考，含对比分析 | 🟡 高 |
| discussion_evq_cosh_analysis.md | Discussion Section 的详细分析和数学审查记录 | 🟢 参考 |

## ⚠️ 致所有 AI 助手的核心指令

1. **理论已验证**: 三大定理数学正确，推导完整。不要质疑 cosh 的来源或 τ 的物理意义
2. **From-scratch 是正路**: 50M/125M/350M/500M from-scratch 训练是正确的实验范式
3. **LoRA 不是 EVQ 的正确验证方式**: 但受控 LoRA 实验仍有价值（waterbed 验证）
4. **编号注意**: 论文中的 Theorem 1/2 ≠ 知识库中的 Theorem 1/2/3。以论文 V5/V6 为准
5. **不要使用知识库中的夸张措辞**: 如"绝对最优"、"趋于无穷"等，论文中已修正为审慎表述
