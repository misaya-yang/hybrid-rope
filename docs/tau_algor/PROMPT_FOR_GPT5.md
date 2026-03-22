请详细阅读这个文件夹中的理论部分：hybrid-rope/docs/tau_algor，特别是以下文件：

- `TAU_SCALING_DERIVATION.md` — 完整的理论推导链和已失败的尝试
- `TAU_FIRST_PRINCIPLES_ANALYSIS_2026-03-22.md` — 推导链断裂点的精确诊断
- `TAU_STATIC_VS_DYNAMIC_EXPERIMENT_2026-03-22.md` — 数值实验结果
- `TAU_UNIFIED_THEORY.md` — 当前的统一理论框架

同时阅读论文正文 `paper/sections/03_theory.tex` 和附录 `paper/appendix/a1_proofs.tex` 中的完整变分推导。

---

## 问题

EVQ-Cosh 的核心理论链是：

```
D(Δ) ∝ 1/Δ  →  K(φ₁,φ₂) = ∫D(Δ)cos(ω₁Δ)cos(ω₂Δ)dΔ
→  K ≈ αδ(φ₁-φ₂) + β·min(φ₁,φ₂)   (broadband surrogate)
→  ρ'' - τ²ρ = γb^{-2φ},  τ = √(β/α)   (Euler-Lagrange ODE)
→  ρ_τ(φ) = cosh(τ(1-φ))   (唯一稳态解)
→  φ_k(τ) = 1 - (1/τ)arcsinh((1-u_k)sinh(τ))   (CDF inversion)
```

这条链严格地推导出了 **cosh 族** 是最优频率密度。但链中没有出现 τ 的具体取值。

实验发现 τ* = d_head / √L（99次实验 R² > 0.99）。问题是：**L^{-0.5} 这个指数能从理论推导吗？**

## 已有结论

1. **broadband surrogate 的 α, β 不携带足够的 L 信息**：数值拟合 α ~ L^{-0.051}, β ~ L^{-0.221}, 所以 τ_surr = √(β/α) ~ L^{-0.085}，远达不到 L^{-0.5}
2. **自洽 surrogate**（在 EVQ 分配点而非均匀网格上拟合 α*, β*）改善到 L^{-0.17}，仅填了 1/3 的 gap
3. **直接在 discrete exact kernel 上优化任何静态目标**（L2 collision, mutual coherence, condition number, weighted collision, 等 7+ 种），最优 τ 都是 10-15 且几乎不依赖 L。说明静态 collision landscape 单调偏向极端再分配
4. 结论：L^{-0.5} 大约 1/3 来自静态 kernel 几何，2/3 来自训练动力学

## 你的任务

我不满意"2/3 来自训练动力学"这个结论。我认为应该存在一条**纯理论路径**能推导出 τ* ∝ L^{-0.5}（或至少 L^{-0.4} 以上）。

请你：

1. **独立审视**上面的推导链和数值实验，找出我们可能遗漏的理论机制
2. **提出新的变分泛函或目标函数**，使得其最优 τ 的 L-scaling 接近 L^{-0.5}。不需要是闭式的——如果能给出一个数学上 well-defined 的优化问题，其解的 scaling 能推导出 L^{-0.5}，就算成功
3. **考虑以下方向**（但不限于）：
   - 信息论：频率分配的 mutual information 或 Fisher information 关于位置的表达
   - 有限通道效应：K 个离散通道的相干结构如何引入额外的 L 依赖
   - 注意力机制的几何：softmax(QK^T/√d) 中的 √d 归一化与 τ 的关系
   - 有效维度：d_head 个频率通道在序列长度 L 下的有效自由度
   - 随机矩阵理论：K×K kernel 矩阵的谱分布如何随 L 变化
4. 如果你确认 L^{-0.5} 确实无法从纯静态理论推导，请给出**最强的理论 bound**（比 L^{-0.17} 更紧），并说明紧化需要什么额外假设

输出格式：数学推导为主，结论明确，不要泛泛而谈。
