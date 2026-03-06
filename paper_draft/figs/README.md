# Paper Theory-Narrative-Asset Matrix

> 目的：把**理论严谨性**、**正文叙事优先级**、**主图主表资产**统一到一张可执行矩阵上。
> 原则：先决定“什么是 paper-grade primary claim”，再决定图表和正文怎么分配版面。
> 理论高标准来源：`/Users/misaya.yanghejazfs.com.au/AI-Imam-pdf/EVQ_Cosh_Theory_Optimized_Rigorous_2026.tex`

---

## 当前稳定图资产

| Figure | 文件 | 生成脚本 | 当前定位 |
|--------|------|----------|----------|
| Figure 1 | `fig1_frequency_dynamics.pdf` / `.png` | `scripts/figures/fig1_neurips.py` | supporting mechanism / dynamics |
| Figure 2 | `fig2_evq_yarn_synergy.pdf` / `.png` | `scripts/figures/fig2_evq_yarn_orthogonality.py` | **main empirical figure** |

## Submission implementation

- Canonical anonymous submission plan:
  [NEURIPS_SUBMISSION_PLAN.md](/Users/misaya.yanghejazfs.com.au/neurIPS-2026/hybrid-rope/paper_draft/NEURIPS_SUBMISSION_PLAN.md)
- Implemented review-version skeleton:
  [main.tex](/Users/misaya.yanghejazfs.com.au/neurIPS-2026/hybrid-rope/paper_draft/submission/main.tex)

---

## 决策级矩阵

> 这张表回答的不是“哪条结论有趣”，而是“哪条结论最能把稿子从 weak accept 推到 strong accept / spotlight”。

| 层级 | Claim | 为什么 reviewer 会在意 | 当前证据强度 | 主文资产 | 当前判断 |
|------|-------|-------------------------|--------------|----------|----------|
| **P0** | **`EVQ+YaRN >> Geo+YaRN`** | 这把工作从“又一个训练时 PE tweak”升级成“解锁现有 inference-time scaling 的训练底座” | **A+**：`8K=100% across 6/6 seeds`；10% fair scale=8 下 `PPL@8K 70.9 vs 82.9`，`PPL@16K 107.5 vs 157.7` | **Figure 2 + Table 2** | **当前最强 killer result** |
| **P0** | **DAPE-style extreme extrapolation 下，closed-form EVQ > learnable PE** | 直接对标 learnable/adaptive PE 文献，证明 “PE quality itself” 可以决定远程 PPL | **A / A-**：128-token 64x 外推到 8K，EVQ `333.7` vs DAPE `455.3`，且 EVQ 为 `0` 额外参数 | **Figure 3 + Table 4** | **当前最强理论-实验一体化卖点** |
| **P1** | L=256 / Phase 11 同时确认 `τ*` 与 EVQ×YaRN 杠杆 | 把 scaling law、PE-dominant、systems synergy 三件事绑在一起 | **A / A+**：`EVQ4+YaRN 99.6 vs Geo+YaRN 260.2` at 32x；YaRN leverage `10x` | **Figure 3 + Table 5** | spotlight 级综合证据 |
| **P1** | 多规模 raw length generalization | 证明 EVQ 不是仅在 PE-dominant toy regime 有效 | **A**：50M-350M 方向一致；350M 3-seed `PPL@16K -13.3%` | Table 1 | strong accept 稳定器 |
| **P1** | capability-preserving + retrieval gain | 证明代价是 frequency-local，不是 capability-destructive | **B+ / A-**：10% mix raw retrieval `+10.0pp@4K / +12.7pp@8K`；未训练 retrieval 无破坏性证据 | Table 3 | 支撑“不是只会降 PPL” |
| **P2** | collision-block / `base=10K` dead zone | 最容易被攻击的负面结果，必须反转为理论证据 | **A**：理论预测 + 负面实验确认 | Figure 4 + Table 6 | rebuttal 必需，不是 headline |
| **P2** | 750M dynamics + 4K-continue full EVQ | 给出 Geo regression 与更大规模 full-EVQ continue 的 supporting picture | **B / B+**：single-seed，但轨迹与 final head-to-head 都清晰 | Figure 1 + Table 7 | supporting only |
| **P2** | video temporal transfer | 抬高论文上限，证明不是 text-only artifact | **B- / B**：2-seed + strong EVQ×YaRN synergy，但尚未达到 text 主线强度 | Appendix 主图 / 正文轻触 | 上限提升项，不替代主锤 |

---

## 理论严谨矩阵

> 使用原则：先问“这条结论在数学上属于哪一类”，再决定它能不能进正文 theorem、正文 proposition、经验命题，还是只能放 conjecture / appendix。

| T-ID | 理论对象 | 数学地位 | 正文身份 | 关键前提 / 使用边界 | 可以直接说 | 不可过说 |
|------|----------|----------|----------|---------------------|------------|----------|
| T1 | 精确 phase-collision kernel 与 `Ci` 闭式 | **Exact proposition** | 正文 preliminaries | `D(Δ)` 已定义；对角线需用连续延拓理解 | “精确 kernel 有闭式” | 不能把 `Ci(0)` 当普通有限值直接代入 |
| T2 | Green 算子 `A^{-1}` 与 broadband surrogate | **Exact lemma + modeling convention** | 正文唯一近似步 | 连续模型中必须写成 `αI+βA^{-1}`；系数标定在有限 `N` 矩阵层面进行 | “唯一近似步是 surrogate `αI+βA^{-1}`” | 不能写成“连续 HS 投影到 `αδ+βmin`” |
| T3 | 变分问题的存在唯一性、仿射极小解、KKT | **Exact theorem** | 正文理论主干 | `α>0, β≥0`；密度极小解需要非负约束，必要时转 KKT | “仿射极小解唯一；密度极小解唯一” | 不能把密度极小解无条件等同于仿射 Euler-Lagrange 解 |
| T4 | forced ODE、非共振/共振闭式解 | **Exact theorem/corollary after surrogate** | 正文主定理 | surrogate 已固定；`τ=2 ln b` 共振分支必须单列 | “ODE→闭式解链条在 surrogate 固定后是精确的” | 不能漏掉共振分支 |
| T5 | pure tether、CDF/反 CDF、全局 warp 几何 | **Exact theorem/proposition** | 正文最干净的 closed-form theory | 纯 tether (`μ_F=0`) 下成立 | “Geometric 是 `τ→0` 退化点；全局间距重分配严格成立” | 不能降级成仅靠 Taylor 的局部现象 |
| T6 | Waterbed 下界 | **Exact proposition under explicit assumptions** | 正文 theorem / proposition | 需显式写 `E>0` 与 `∫ln(EI)≥0`；等号判别还需 `E=1/I` | “Waterbed 下界在这些假设下严格成立” | 不能写成无条件零假设不等式 |
| T7 | Fisher → 注意力效用桥梁 | **Asymptotic / local heuristic bridge** | 正文解释性 remark | 只在 Laplace / local-curvature 近似下有意义；大 `Δ` 失效 | “Fisher 项提供局部注意力精度的解释桥梁” | 不能写成全距离严格等价定理 |
| T8 | `τ*` scaling law | **Conditional proposition if scale bounds are proved; otherwise empirical law / conjecture** | 正文 conjecture / empirical law | 需要统一标度界才能升级为条件性理论 | “当前实验支持 `τ*≈d_head/√L`” | 不能写成无条件 theorem |
| T9 | collision-block / dead zone | **Simplified analytic model + empirical validation** | 正文机制解释 | 属于简化量纲分析，不是完整连续变分理论的严格推出 | “collision-block 模型成功预测 dead zone” | 不能宣称与主变分理论同级严格 |
| T10 | Riemann-Lebesgue / Hybrid 严格优越性 | **Mathematically possible but practically negligible; experimentally falsified as design principle** | Appendix caution only | 量级是 epsilon 级，不再指导方法选择 | “可作为理论备注解释为何曾考虑 Hybrid” | 不能再做正文主 theorem / 主方法 |

---

## 叙事-图表-表格矩阵

| 模块 | 主叙事任务 | 理论依赖 | 主文资产 | 适合放哪里 | 当前状态 |
|------|------------|----------|----------|------------|----------|
| M1 | 证明 EVQ 不是 heuristic，而是 closed-form solution | T1 + T2 + T3 + T4 + T5 | 正文定理 + Algorithm box | Intro / Theory | ✅ |
| M2 | 证明 Geometric 是退化点，不是 generic optimum | T5 + T6 | Theory section + Table 1 | Theory / Results 前置 | ✅ |
| M3 | 证明极限长外推下 EVQ 比 learnable PE 更强 | T5 + T8 | **Figure 3 + Table 4** | 主文 Section 5 前半 | ✅ |
| M4 | **证明 EVQ 解锁 YaRN** | T2 + T5 + T7 | **Figure 2 + Table 2** | 主文中心结果段 | ✅ |
| M5 | 解释 bounded short-range cost 与 long-range gain 的非对称性 | T5 + T6 + T7 | Figure 1(a) + Table 1 | 机制解释 | ✅ |
| M6 | 证明 capability-preserving + retrieval gain | T6 + multi-seed retrieval evidence | Table 3 | 次主结果段 | ✅ |
| M7 | 反转 `base=10K` 负面结果 | T9 | Figure 4 + Table 6 | rebuttal / appendix 可回指 | ⏳ |
| M8 | 提供 Geo regression / EVQ foundation 的直观轨迹证据 | T10 + supporting experiment | Figure 1(b)(c) + Table 7 | supporting only | ✅ |
| M9 | 提供跨模态迁移上限 | T5 的迁移直觉 + video evidence | Appendix figure/table | appendix / discussion | ⚠️ |

---

## 当前推荐的主文版面预算

| 位置 | 资产 | 作用 |
|------|------|------|
| 主图 1 | **Figure 2** `fig2_evq_yarn_synergy.*` | 直接承载 `EVQ+YaRN >> Geo+YaRN` |
| 主图 2 | **Figure 3** | [fig3_pe_dominant_scaling.pdf](/Users/misaya.yanghejazfs.com.au/neurIPS-2026/hybrid-rope/paper_draft/figs/fig3_pe_dominant_scaling.pdf)；承载 DAPE-style extreme extrapolation + Phase 11 `τ*` 直观确认 |
| Supporting 图 | Figure 1 `fig1_frequency_dynamics.*` | 给机制和 750M dynamics，不再承担主 headline |
| Rebuttal / mechanism 图 | Figure 4（待做） | 把 `base=10K` 负面结果转成理论证据 |
| 主表 1 | Table 1 | 跨规模 raw PPL 一致性 |
| 主表 2 | **Table 2** | PE baselines + `EVQ+YaRN` 公平比较；当前最值钱表格 |
| 主表 3 | Table 3 | capability-preserving + retrieval evidence |

---

## 写正文时的硬规则

1. 只有 `T1/T3/T4/T5/T6` 能进正文 theorem / proposition 区。
2. `T2` 必须写成“唯一近似步 / modeling convention”，不能伪装成 exact theorem。
3. `T7` 只能当解释性 bridge，不承担主证明职责。
4. `T8` 当前只能写成 `empirical law` / `conjecture`；未来补齐标度界再升级。
5. `T9` 是简化模型，不与主变分理论同级。
6. `T10` 只保留为 appendix caution，不再做主方法叙事。
7. `+40pp` 单 seed 极值、`5%→10%` 旧反对称叙事、以及 `video confirms τ*=2.0` 都不能回到正文 headline。

---

## 下一步最值钱的资产

1. **Figure 4: Collision-block / base dead zone**
   - 内容：`base` vs collision fraction / predicted gain + `base=10K` 负面结果
   - 作用：把最容易被攻击的负面结果变成理论验证

2. **1.5B 主 benchmark 表 / 图（未来）**
   - 内容：真实复杂下游任务上的 clean win
   - 作用：决定稿子是否能从 strong accept contender 进一步抬向 oral 区间

---

## 命名规则

- `figN_<semantic_name>.pdf`
- `figN_<semantic_name>.png`
- `figN` 对应论文最终编号
- `semantic_name` 使用方法学语义，不使用 `final`, `v3`, `neurips` 之类临时命名
