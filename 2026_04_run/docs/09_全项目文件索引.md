# EVQ-Cosh 项目关键文件映射

> 最后更新：2026-03-31
> 用途：快速定位项目中的核心文件、数据、分析脚本

---

## 1. 论文源码

| 文件 | 内容 |
|------|------|
| `paper/main.tex` | 主文件 |
| `paper/sections/01_intro.tex` | Introduction |
| `paper/sections/02_related.tex` | Related Work |
| `paper/sections/03_theory.tex` | **核心理论**: broadband functional J[ρ], ODE, EVQ-Cosh 闭式解, τ* scaling law, waterbed inequality |
| `paper/sections/04_predictions.tex` | 理论预测 |
| `paper/sections/05_experiments.tex` | **实验结果**: MHA/MLA/progressive training |
| `paper/sections/06_limitations.tex` | Limitations |
| `paper/sections/07_conclusion.tex` | Conclusion |
| `paper/REBUTTAL_PLAYBOOK.md` | Rebuttal 攻防策略 |

---

## 2. 理论文档

| 文件 | 内容 | 重要度 |
|------|------|--------|
| `docs/tau_algor/unified_tau_star_theory_v2.md` | **τ* 统一理论**: τ*=√(β/α), 架构特定公式 | ★★★ |
| `docs/tau_algor/mla_linear_vs_sqrt_correction_v1.md` | **MLA 修正因子**: κ_dilute = d_qk/d_rope (线性非根号) | ★★★ |
| `2026_04_run/docs/theory_problems.pdf` | **六大开放理论问题** Q1-Q6 | ★★★ |
| `2026_04_run/docs/theory_analysis_report_20260331.md` | **2026-03-31 理论分析综合报告** (本次分析) | ★★★ |
| `docs/tau_algor/TAU_REGIME_THEORY_2026-03-24.md` | τ regime 理论 | ★★ |
| `docs/tau_algor/TAU_THEORY_DEEP_ANALYSIS_2026-03-24.md` | τ 深度分析 | ★★ |
| `docs/tau_algor/TAU_HABITABLE_ZONE.md` | τ 宜居带概念 | ★ |

---

## 3. 实验报告 (按时间)

| 文件 | 内容 | 关键结论 |
|------|------|----------|
| `docs/exp/2026-02-26_full_experiment_report.md` | **Phase 0-3 完整报告**: learnable τ 实验 | learnable τ 失败; in-dist PPL 对 τ 不敏感(<2%) |
| `docs/exp/2026-03-09_phase16_formula_optimality_sweep_results.md` | **Phase 16 τ* sweep**: 99 runs, 9 配置 | τ* scaling law 经验验证 |
| `docs/exp/2026-03-14_staged_diagnostic_report.md` | Staged continuation 诊断 | tau retarget 不是 burden 的原因 |
| `docs/exp/2026-03-20_gqa_mla_125m_compression_ablation.md` | GQA/MLA 125M 消融 | MLA 31.1% PPL 提升 |
| `docs/exp/2026-03-11_test3_broadband_r2_validation.md` | Broadband R² 验证 | Surrogate 精度数据 |

---

## 4. 核心代码

| 文件 | 内容 | 重要度 |
|------|------|--------|
| `scripts/lib/rope/inject.py` | **RoPE injection**: EVQ-Cosh 频率注入主逻辑 | ★★★ |
| `scripts/lib/rope/learnable_evq.py` | **Learnable τ 实现** (531行): softplus(raw_tau), Taylor 稳定梯度, Algorithm 1 | ★★★ |
| `scripts/lib/rope/schedules.py` | Progressive training τ schedule | ★★ |
| `scripts/lib/rope/attn_hist.py` | Attention 距离分布采集 | ★ |
| `scripts/core_text_phases/phase17f_progressive_tau_burden_probe.py` | **Phase 17F**: 三臂实验 geo/evq_dynamic/evq_frozen | ★★ |
| `scripts/core_text_phases/mla_patch.py` | MLA 架构 patch | ★★ |
| `scripts/core_text_phases/mla_tau_optimization_v2.py` | MLA τ 优化 v2 | ★ |

---

## 5. 关键数据

| 路径 | 内容 | 用于 |
|------|------|------|
| `results/m4_max_36gb/D_attention_per_head.npy` | GPT-2 per-head attention (12×12×1023) | Per-head τ 分析 |
| `results/m4_max_36gb/test3_attention_prior_results.json` | 144 heads power-law fits + 分类 | Head specialization |
| `results/theory/phase16_formula_optimality_sweep_local_m4_wikitext/` | **Phase 16 数据**: 99 runs, 9 配置 × 多 τ × 多 seed | λ 提取、τ* 验证 |
| `results/theory/phase16_.../runs/*/result.json` | 每个 run 的 PPL@{L, 2L, 4L, 8L} | τ* 寻优 |
| `results/theory/phase16_.../batch_probes/*.json` | 9 个配置的 batch probe | 显存规划 |
| `results/m4_max_36gb/theory_verification_results.json` | 理论验证结果 | Broadband R² 等 |

---

## 6. 本次分析 (2026-03-31) 关键发现映射

### Analysis 1 → 论文 Proposition 3
- **结论**: ΔC(τ) = -(2β/45)·τ² < 0 (surrogate 下 EVQ 严格降碰撞)
- **脚本**: sympy 推导 (session-only, 未持久化)
- **论文影响**: 可加入 §3 Theory 作为 Corollary
- **注意**: 仅在 surrogate 近似下成立

### Analysis 2 → MLA 章节 §5
- **结论**: MLA 16 channels 在 formula τ 下频移仅 5-13% channel spacing
- **论文影响**: 需软化 MLA 理论声明; 强调 empirical τ sweep
- **相关文件**: `docs/tau_algor/mla_linear_vs_sqrt_correction_v1.md`

### Analysis 3 → τ* Scaling Law 验证 (§3 + §5)
- **结论**: λ ≈ 1.17 ± 0.13 (CV=11%), τ* ∝ d_head^0.94 · L^(-0.44)
- **数据来源**: `results/theory/phase16_.../runs/*/result.json` (99 runs)
- **论文影响**: 加 λ cross-validation table; 定位为 "empirically calibrated"

### Analysis 4 → Surrogate 有效性验证 (v2 修正)
- **v1 错误**: 混淆离散通道 collision 和连续密度泛函, 错误声称 "collision paradox"
- **v2 修正结论**: 连续泛函在小 τ 时确实减少 collision, 论文理论正确
- 离散 collision 增加是 T1 (密度集中度) 主导, 不构成矛盾
- Effective dimensionality increase 仍是有用的直觉补充
- **论文影响**: 不需要大改, 理论框架正确

---

## 7. 六大理论问题状态速查

| ID | 问题 | 状态 | 下一步 |
|----|------|------|--------|
| Q1 | λ 闭式解 | ✅ 部分解决 | λ≈1.17 作为经验常数写入论文 |
| Q2 | Surrogate 有效边界 | ✅ 已验证 (v2) | 连续泛函定性正确, 无需大改 |
| Q3 | DiT 0.53× 因子 | ❌ 未处理 | 低优先级, 标注为 limitation |
| Q4 | LoRA 相变 r_c=d_head/2 | ❌ 未处理 | 中优先级, 需单独实验 |
| Q5 | Progressive 放大机制 | ❌ 未处理 | Phase 17F 已经验证, 低优先级 |
| Q6 | τ* 在 L≥4096 有效性 | ✅ 部分解决 | 机制是 dimensional, 非 collision |

---

## 8. Deadline 提醒

| 日期 | 事项 |
|------|------|
| **2026-05-04** | NeurIPS 2026 Abstract 提交 |
| **2026-05-06** | NeurIPS 2026 Full Paper 提交 |

---

## 9. 快速导航：如果你想...

- **改理论章节** → 先读 `theory_analysis_report_20260331.md`, 再改 `03_theory.tex`
- **验证 τ* 公式** → Phase 16 数据在 `results/theory/phase16_.../`
- **查 learnable τ 失败原因** → `scripts/lib/rope/learnable_evq.py` + `docs/exp/2026-02-26_full_experiment_report.md`
- **理解 MLA 修正** → `docs/tau_algor/mla_linear_vs_sqrt_correction_v1.md`
- **准备 rebuttal** → `paper/REBUTTAL_PLAYBOOK.md` + 本报告的 Analysis 4
- **决定是否开新实验** → 先确认 Q2 (surrogate validity) 在论文中如何处理
