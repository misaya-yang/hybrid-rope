
╔══════════════════════════════════════════════════════════════════════╗
║            THEORY PROBLEMS: COMPREHENSIVE STATUS REPORT            ║
║                    EVQ-Cosh NeurIPS 2026                           ║
║                    Analysis Date: 2026-03-31                       ║
║                    ★ v2: Analysis 4 已修正 ★                       ║
╚══════════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ANALYSIS 1: Small-τ Asymptotic (COMPLETED ✓)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Result: Surrogate 泛函 J[ρ] 在 τ < ~1.4 时减少 collision

数值验证 (α=β=1):
  τ=0.1: ΔJ = -0.033%
  τ=0.5: ΔJ = -0.705% (T2 通道间相关下降 -3.2%)
  τ=1.0: ΔJ = -1.522% ← collision 最优点
  τ=1.5: ΔJ = +0.103% ← 开始变差
  τ=2.0: ΔJ = +5.801%

分解:
  T1 (α/2·∫ρ²) = 密度集中度 → τ增大时 T1 一直增加
  T2 (β/2·∫∫ρρ·min) = 通道间相关 → τ增大时 T2 一直减少
  J = T1 + T2 在 τ ≈ 1 时取最小值

含 Fisher utility 的完整泛函:
  τ=1.0: ΔJ_full = -3.98% (最优)
  τ=2.0: ΔJ_full = -1.48% (仍好于 geometric)
  τ=3.0: ΔJ_full = +17.0% (变差)

Paper implication:
  → 可作为 Corollary: EVQ 在小 τ 时降低 collision (T2 下降补偿 T1 上升)
  → 完整 J[ρ] 的最优来自 collision 和 Fisher utility 的 tradeoff
  → 论文的 "collision + Fisher tradeoff" 叙事是正确的

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ANALYSIS 2: MLA 16-Channel Discretization (COMPLETED ✓)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Result: Formula-predicted τ creates negligible frequency shifts for MLA

Key findings:
  • τ_formula = d_rope/√L gives τ ≈ 0.25-0.5 for MLA at L=1024-4096
  • At these τ values, per-channel shift ≈ 5-13% of channel spacing
  • This is near-noise-level for K=16 channels

Paper implication:
  → MLA 的 τ* 公式理论上偏小, 但实验中用了 empirical τ sweep
  → 建议在 MLA 段落注明: 公式给出初始点, 实际用 sweep 微调

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ANALYSIS 3: λ Universality from Phase 16 Data (COMPLETED ✓)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Result: λ ≈ 1.17 ± 0.13 (CV = 11.1%, parabolic fits only)

From 99 runs across 9 configurations (L ∈ {256,512,1024}, d_head ∈ {32,64,128}):

  Cross-table of mean λ:
                L=256    L=512    L=1024
  Dh=32:        1.060    1.315    1.250
  Dh=64:        1.089    0.937    1.169
  Dh=128:       1.256    0.864    1.271

  Log-linear regression: τ* ≈ 0.967 · d_head^0.944 · L^(-0.439)
  Theory predicts:       τ* ∝ d_head^1.0   · L^(-0.5)

  Deviation: Δα = -0.057, Δβ = +0.061  (both small!)
  Residual: ~19.4% multiplicative error

Paper implication:
  → Q1 (λ closure) PARTIALLY CLOSED: λ≈1.17 近似普适
  → Scaling law 在 ~6% 指数精度内得到验证
  → 建议加 λ cross-validation table 到 appendix

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ANALYSIS 4: Surrogate Kernel Validity (COMPLETED ✓ — v2 修正)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

★ v1 结论有误, 已修正 ★

v1 错误: "EVQ 在所有设置下增加 collision → collision paradox"
原因: v1 算的是离散通道 collision (Σ_{i≠j} K(i,j)²),
      但论文理论是连续密度泛函 J[ρ]，两者不等价。

v2 修正结论:
  • 连续 surrogate 泛函 J[ρ] 在小 τ 时确实减少 (Analysis 1 验证)
  • 离散通道 collision 在所有 τ 增加 — 但这是因为离散版混合了
    T1 (密度集中度) 和 T2 (通道间相关) 的效果, T1 增加占主导
  • 论文的理论框架 (collision + Fisher tradeoff) 是正确的
  • Surrogate 近似有局限 (R² ≈ 0.25-0.73), 但定性结论 OK

补充发现 (仍然有价值):
  • Effective dimensionality increase 是 EVQ 的一个有用的直观解释:
    d_head=64, τ=2.5: active freqs at Δ=1024 从 14→17 (+21%)
  • 这和 waterbed inequality 的叙事一致: 牺牲高频精度, 换取低频覆盖

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SYNTHESIS: Status of Six Open Theory Problems
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q1 (λ closure): PARTIALLY CLOSED ✓
  λ ≈ 1.17 is universal (CV=11%). Empirically calibrated.
  Paper action: 加 cross-validation table, 标注为 semi-analytic.

Q2 (Surrogate validity): RESOLVED ✓ (v2 修正)
  Surrogate 在连续泛函层面定性正确。
  离散 collision 增加是预期行为 (T1 主导), 不构成矛盾。
  Paper action: 无需大改。可加注 surrogate 是泛函层面的近似。

Q3 (DiT 0.53× factor): NOT ADDRESSED
  Low priority, 保留为 acknowledged limitation.

Q4 (LoRA phase transition): NOT ADDRESSED
  Medium priority, 需单独实验.

Q5 (Progressive amplification): NOT ADDRESSED
  Low priority, Phase 17F 已经验示.

Q6 (τ* at L≥4096): PARTIALLY RESOLVED
  Scaling law 在 L ∈ {256,512,1024} 验证。更大 L 待验证。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

★ PAPER RECOMMENDATION (v2 修正) ★
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

论文理论框架基本正确, 不需要翻筋动骨的改动。建议:

1. 加 λ ≈ 1.17 cross-validation table (Appendix)
2. MLA 段落: 注明公式给初始点, 实际用 empirical sweep 微调
3. 可选: 加 effective dimensionality 的直觉解释到 waterbed 段落
4. 加模型方法图 (老师要求)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ERRATUM: v1 → v2 修改说明
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

v1 的 Analysis 4 声称发现 "collision paradox" 并建议 reframe 论文理论。
这是因为 v1 混淆了离散通道 collision 和连续密度泛函。
经过验证, 连续泛函在小 τ 时确实减少 collision, 论文理论正确。
v1 的 "Option A: reframe" 建议撤回。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
