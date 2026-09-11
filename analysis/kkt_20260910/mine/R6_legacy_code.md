# R6：CPU 可跑的数学/最优化代码资产盘点

**日期**：2026-09-10
**任务**：为"求解 KKT 问题"盘点仓库里现成的、CPU 可跑的数学/最优化代码件。
**纪律**：只读；除本文件外未修改仓库任何文件；未写入 `~/.codex`。本文所有结论均带出处（文件路径 + 行号）。证据分级：[已验证]（我实际读码/复算确认）/ [部分证据]（读码确认存在，但未端到端运行）/ [假设]（推定）/ [叙事-未验证]（来自报告/transcript 文字，未在代码中落实）。
**红线遵守**：本文不把任何静态几何代理量（Σcos 根、碰撞能、覆盖率、平滑度、有效秩、Gram、能量）推荐为 F 的分项或选择子；凡此类代码只标注"机器可复用、目标已红线"。

---

## 0. 一句话结论（给 workflow-3 的 K3 用）

**求解 KKT 的现成拼装方案已经存在于仓库里，且全部纯 CPU、无 GPU 依赖**：

1. **坐标系**：`docs/research/rope_allocation_20260910/code/sol16_frequency_calibration_reference.py` 的 `helmert_zero_sum(width)` + `MrProAllocation16` 给出**Δ-单纯形上的无约束 16 维坐标**，`eta=0` 逐位精确等于 MrPro-Pro，且已通过 `gradcheck`。这是 K3 直接该用的坐标。
2. **解算器**：`experiments/nongeometric_screen/smooth_budget.py:8 solve(n,budget)` 是**带完整 primal/dual KKT 证书的 active-set 解算器**（凸，面枚举 + 全局最优性证书 + BM 恢复自检）。它解的正是"ΣΔ=1 约束下的最小粗糙度分配"，其内点条件 `∂F/∂Δ_j = μ` 就是水填充。
3. **带约束的罚函数求解器**：`scripts/analysis/rope_transport/run_pareto_search.py` 的 `_eval_free` 是 `dstar + 50.0 * over`（`over = max(0, risk@target − budget)`）——**现有代码里最接近"最小化窗内损失 s.t. 远距能力预算"这一 KKT 形式的实现**，配 `differential_evolution` 全局搜索 + 命名算子播种。
4. **isotonic / PAVA**：`scripts/analysis/derive_native_isotonic_profile.py:isotonic_nondecreasing` 与 `scripts/experiments/scale_transport/math.py:25 bounded_isotonic`（带上下界的加权递减 PAVA）。INTEGRATION §4.3 的 sol04 "水填充+isotonic，三段结构在解中涌现" 在仓库里**没有对应脚本**，只有这两件通用 PAVA 可复用。
5. **缺件**：**没有任何字面 DP（水床后向递推）实现**；`dependency_spectrum_audit.py` 的 `lloyd_max_coverage`（交替 assign/优化）是唯一的最近邻，但目标是覆盖率（红线）。

---

## 1. 逐文件清单

格式：**算什么 / 输入 / 输出 / 依赖 / CPU 可跑 / 落盘 receipt**。

### 1.1 单纯形 / KKT / 最优化

#### `experiments/nongeometric_screen/smooth_budget.py`（65 行）★最高价值
- **算什么**：`min εᵀLε` s.t. `ε≥0`、`Σε=1`、`Σ(逆序索引)ε=budget`，L 为 Dirichlet 三对角 Laplacian（端点零）。两个等式约束 ⇒ 解在一个 15/16 维单纯形截面上。面枚举（前导/尾随 active 零集）+ 每个面解一次 KKT 线性系统 + 完整可行性/驻点/对偶符号检查。
- **输入**：`solve(n, budget)` 纯数字；`construct(tables, lo, hi)` 需要 tables JSON（含 `Native`/`MrPro` 的 `values_float32` 与 `gain`）。
- **输出**：`(eps, certificate)`，certificate 含 `budget_actual / total / roughness / minimum_increment / active_zero_indices_1based / dual_multipliers / equality_residual / stationarity_free_residual`（L23-26）。不可行时报 `ValueError('No certified solution found for the requested budget')`（L27）——**fail-closed**。
- **依赖**：`numpy` + `argparse/json/pathlib`。**无 torch、无 scipy**。[已验证]
- **CPU 可跑**：是。
- **落盘**：写 `planned_controls/smooth_budget/<target>.json`（L54）与 `queue/044{0,1}_*.json`（L58-59）。
- **关键常数**：`solve(n,(n-1)/3)` = `Smooth_MrBudget`；`solve(n,(n-1)/2)` = `MrUni`；L33-34 用 `6i(n+1−i)/(n(n+1)(n+2))` 做 **BM 恢复自检**，`atol=1e-11`。[已验证]
- **边界**：`qwen3b (23,40)`、`olmo1b (14,32)`（L52）。
- **scope 自述**（L47）："Unique nonnegative minimum roughness at fixed cumulative budget; not a task-optimality result"。[已验证-代码注释]

#### `docs/research/rope_allocation_20260910/code/sol16_frequency_calibration_reference.py`（142 行）★最高价值
- **算什么**：MrPro 初始化的 16 自由度频率标定器 CPU 参考实现。
  - `helmert_zero_sum(rows)`（L14-23）：返回 `{x∈R^rows : Σx=0}` 的**正交归一基**（经典 Helmert 构造，列 j 为 `1/√(j(j+1))` 前 j 项、第 j+1 项 `−j/√(j(j+1))`）。
  - `mrpro_increments(width)`（L26-29）：`ε_i = 2i/(width(width+1))`。
  - `MrProAllocation16`（L32-78）：`increments() = softmax(log ε_ref + Basis @ eta)`（L65-67）⇒ 自动 Δ≥0、ΣΔ=1；`exponents()` 拼 `[0]*24 + cumsum + [1]*24`（L69-75）；`inv_freq() = ω * exp(−log(S)·exponents)`（L77-78）。
- **输入**：`eta ∈ R¹⁶`（`nn.Parameter`）；超参 `pair_count=64, head_dim=128, rope_theta=1e6, scale=4, low=23, high=40`。
- **输出**：64 维 inv_freq；`DifferentiableQwenRotaryEmbedding`（L81-107，**去掉 HF `@torch.no_grad` 的可微 rotary**，相位 FP32 + autocast 禁用）。
- **依赖**：`torch`（float64 路径）+ `math`。**无 scipy**。[已验证]
- **CPU 可跑**：是（`self_check()` 全部在 CPU，L110-137，末尾打印 `PASS`）。
- **落盘**：无。是 contract/reference，不是 runner（L3-4）。
- **关键性质**：[已验证-代码] `eta=0` 逐位等于 MrPro（docstring L38 + L114 assert_close atol=2e-16）；16 个标量全部可辨识（`eta` 维度 = width−1 = 16，L59）；L136-137 对 `softmax(log_ref + basis@z)` 做 `gradcheck(eps=1e-6)`。

#### `scripts/analysis/rope_transport/nullband.py`（585 行）★可行域精确描述
- **算什么**：相安全盒 + 频带重参数化 + 覆盖残差。核心是三条：
  - `phase_safety_box(omega_native, *, native_length, target_length)`（docstring 明写）：盒 = `w'_k·L_tgt ≤ w_k·L_nat` **OR** `w_k·L_nat ≥ 2π`；换到 `phi = −log(w)/log(base)` 坐标后变成**下界** `phi'_k ≥ phi_k + log(s)/log(base)`，盒外无约束。
  - `free_band_table(native, z, *, scale, band_start, rope_base, floor=None, name)`：`increments = logaddexp(0, clip(z,−60,60))`（softplus，**注意是 softplus 而非 softmax**）→ `cumulative=[0,cumsum]` → `phi_band = a+(b−a)·cumulative/total`；随后 running-max floor + 1e-9 严格单调强制。**按构造可行**。
  - `free_band_delta_table(native, z, delta, *, band_start, rope_base, start_delta=0.0, name)`：同上但带内总 log 跨度**自由**（`a=phi[j0]+start_delta`, `b=phi[-1]+delta`）。docstring："there is no safety floor: feasibility on the risk axis is expressed by the objective, not by the parametrisation, which is what makes the two-objective frontier searchable."（**这句就是 KKT 罚函数法的正当性陈述**）
  - `encode_band(native, table, *, band_start, rope_base)`：把表反解回 `(z, delta, start_delta)` 作为搜索种子；会拒绝移动了 band_start 以下槽的表。
- **依赖**：`numpy` only。[已验证]
- **CPU 可跑**：是。**落盘**：无（纯函数库）。

#### `scripts/analysis/rope_transport/run_pareto_search.py`（439 行）★罚函数法模板
- **算什么**：两目标（窗内 D\* vs 远距风险）帕累托搜索，`differential_evolution` 全局优化 + 罚函数。
- **目标函数逐字**（`_eval_free`）：
  ```python
  over = max(0.0, stats[f"risk@{target}"] - budget)
  return index, stats["dstar"] + 50.0 * over, stats
  ```
  ——`over` 是**单边不等式约束违反量**，`50.0` 是罚系数。这是仓库里对"min L_near s.t. L_far 预算"最直接的转写。
- **输入**：`vector = [z (dim−2), delta, start_delta]`；`band_start`；`budget`。
- **输出**：`(index, penalized_objective, stats)`；`stats` 含 D0/d\* 与每目标 `phase_excess_risk` 的 mean/max turns。
- **依赖**：`numpy` + `multiprocessing`；有 **fail-closed CPU-only 门**：`_require_no_cuda()` 在 `CUDA_VISIBLE_DEVICES` 被设置或 torch 已导入时 raise。[已验证-读码]
- **CPU 可跑**：是（设计如此）。
- **落盘**：未在头 110 行确认；脚本自述 "historical failed selections do not authorize a new sweep"。

#### `scripts/analysis/demand_companding_numeric.py`（1520 行）★数值泛函工具箱
- **算什么**：高码率量化理论下的密度/失真/headroom 全套数值工具 + 一个带钉扎约束的 SLSQP 求解器。
- **模块 docstring（L12-30）给出的核心公式**（逐字）：
  - `D[rho] = (12 K**2)**-1 ∫ m(x)/rho(x)**2 dx`
  - Euler 解 `rho = m**(1/3)/Z`
  - `D* = Z**3/(12 K**2)`
  - headroom `H(m) = 1 − Z**3`
  - λ 族 `m_lambda = (1−λ)m + λ/|I|`
  - 逐点四次方程 `alpha*rho**4 + nu*rho**3 − 2*(m+epsilon) = 0`
  - 标量驻点方程 `F(t)=alpha*t**4 + beta*t**2 − gain*t`，`F'(t)=4αt³+2βt−gain`
- **关键签名**：
  - `high_rate_density(...)` → `rho ∝ cbrt(m)`、`H_m13`、`H`（Jensen 截断）、`H_differential`
  - `D_star(x, demand, K=None)`；`compression_headroom`；`H`
  - `quantile_from_density(x, density, K, *, endpoint=True)`：逆梯形 CDF；`endpoint=True` 用 `u_k=k/(K−1)` 强制精确落在定义域两端，`endpoint=False` 用 `u_k=(k+1/2)/K`；**CDF 平坦段被保留为重复分位数**，注释明写"so zero-demand/collision behavior [is] visible rather than adding a hidden positive floor"（**这条与"允许重复频率"的 KKT 变量声明直接相关**）
  - `capp_components(x, rho, *, alpha, beta)`：`beta_kernel = ∫ tail(φ)² dφ`，`tail = 1 − CDF`（即 `∫∫min(φ,ψ)ρρ = ∫(1−F)²`）
  - `capp_discrete_objective(values, *, alpha, beta)`：Voronoi 单元格离散化 `edges=[0, midpoints..., 1]`、`rho = 1/(n·widths)`、`beta_term = 0.5β·masses @ min(points_i, points_j) @ masses`
  - **`conditional_capp_optimize(K, protected_values, *, alpha=1.0, beta=1.0, protected_indices=None, min_gap=1e-6, max_iter=300)`**：**SLSQP 带钉扎/保护频率**，两种模式（gap 模式：保护点切分区间成锚，锚内放自由点；index 模式）。返回 `status ∈ {NUMERICAL_SMALL_SCALE_ONLY, BLOCKED_NONCONVEX_OR_SOLVER}`，并显式带 `global_optimum_claim: False`、`convexity: "not_established_for_discrete_parameterization"`、`solver: "scipy.optimize.SLSQP"`。
- **依赖**：`numpy`；**scipy 在函数内部惰性导入**（AST 检查：`scipy.optimize` 是唯一非 stdlib 导入）。[已验证]
- **CPU 可跑**：是。**落盘**：`write_csv` / `write_plot` 存在。

#### `scripts/analysis/verify_general_rope_allocation.py`（155 行）★MGDA
- **算什么**：经典 MGDA 最速下降方向（凸包最小范数方向）+ 数值证书。
- **签名**：
  ```python
  common_direction(gradients, tolerance=1e-12, max_iterations=20000)
      -> dict(direction=v, weights=weights, norm_squared=..., dual_gap=gap,
              slopes=slopes, descent_certified=max(slopes)<-1e-12,
              solver_converged=gap<=tolerance, iterations=...)
  ```
- **另含**：`rotary_binary_loss(x, records)` —— 精确有限 sin/cos 玩具似然 + 解析梯度（有限差分验证 `<1e-7`）；`two_clock_minimum(q, scale, grid=2000)`，`risk(y) = 2 − sinc(y−q) − sinc(scale·y−q)` 网格搜索。
- **依赖**：`pathlib` + stdlib only。[已验证] **CPU 可跑**：是。**落盘**：写含脚本 SHA 的 JSON。
- **自述**："The solver is classical MGDA, not claimed novel."（[已验证-代码注释]）

#### `scripts/analysis/third_axis_ceiling.py`（358 行）— 机器可复用，目标已红线
- **算什么**：有效秩/碰撞目标下的频率表搜索（**目标 = 红线项，禁用**）。
- **可复用机器**：`search_best_found(pairs, log_max, span, weights, *, restarts, steps)` 用 `z = cat([0], cumsum(softmax(logits)))` 的 **softmax-cumsum 单纯形重参数化** + Adam。这个重参数化**独立于其红线目标**可用。
- **依赖**：`torch`（CPU 张量）。**CPU 可跑**：是。
- **自述警告**（docstring）：`r2` 是"静态、相位不变的属性……**不是** LM 质量或外推预测器"；"到 2K 的数值间隙不能被归因……除非有本脚本不提供的全局证书"。[已验证-代码注释]
- **落盘**：无。

#### `scripts/lib/rope/knot_allocation.py`（12k）
- **算什么**：5 自由度结点分配（7 个归一化 pair-index 结点：2 端点钉死 + 5 内点）。正 knot gap 的 softmax 归一化累计和参数化 ⇒ 实现表恒严格有序；单参数 `support_factor` 拉伸慢端点。
- **关键常数**：`INTERIOR_KNOTS = 5`、`TOTAL_KNOTS = 7`、`MAX_GAP_LOGIT_DELTA = 4.0`、`METHOD_ID = "frozen_checkpoint_z5_knot_allocation_v1"`。
- **依赖**：`numpy` + `torch`。**CPU 可跑**：是。**落盘**：无（是 nn.Module）。
- **另有**：`knot_positions`、`interior_knot_u`、`_piecewise_linear`、`init_gap_logits_from_table`。

#### `scripts/lib/rope/fixed_support_z.py`（8.4k）
- `FixedSupportZRotaryEmbedding`、`install_fixed_support_z`。定支撑 z 参数化（另一条无约束坐标路线）。[部分证据-仅 AST 扫描]

### 1.2 Isotonic / PAVA

#### `scripts/analysis/derive_native_isotonic_profile.py`（302 行）
- `isotonic_nondecreasing(values)` —— **精确 PAVA**（相邻块合并，加权）。
- `pinned_profile(uniqueness)`：target = `clip(1 − uniqueness, 0, 1)`，钉 `movement[0]=0`、`movement[-1]=1`，内点 PAVA，断言非降。
- 另含精确数学：`triangular_fourier(alpha, length)`（三角窗精确傅里叶变换）、`phase_gram(native, length, dps)`（mpmath 精确 cos/sin Gram）、`conditional_uniqueness(...)`（逐对 Schur 补条件唯一性 / 边际迹，clip 到 [0,1]）。
- **依赖**：`mpmath` + `torch` + `numpy`。[已验证]
- **CPU 可跑**：是。**落盘**：CSV/图（脚本内含 `write_csv` 类似路径）。

#### `scripts/experiments/scale_transport/math.py`（84 行）★带界 PAVA
- **签名**：
  ```python
  bounded_isotonic(values, weights, lower, upper)
      """Decreasing weighted PAVA with per-slot bounds; never reorders slots."""
  ```
  逐元素起块 `[start,end,weight,total,lo,hi,clip(total/w,lo,hi)]`，违反 `blocks[-2][6] < blocks[-1][6]` 时合并（`lo=max`, `hi=min`，不可行则 `ValueError('infeasible pooled bounds')`）；出口对结果做三条契约断言（非升、≥lower、≤upper）。**这是唯一带**上下界**的 PAVA**（`derive_native_isotonic_profile` 那只带钉扎端点）。
- 另含 `quantile_moments(a,b)`（有限分布分位矩，用于 β 估计）、`estimate_beta(a,b,ratio=2.)`、`replay(q,k,v,wo,frequencies,gain,positions,*,response=False)`（**完整可见 key 归一化的精确 rotary 重放**，`q=[H,Q,D], k/v=[KV,L,D], wo=[hidden,H*D]`，split-half 布局，只物化选中 query 不物化 L×L）、`background(q,k,frequencies,gain,distances)`。
- **依赖**：`numpy` + `torch` + `math`。**CPU 可跑**：是。**落盘**：无。
- **测试**：`tests/test_scale_transport_math.py` 存在。

### 1.3 频率表 → 指标（评估器）

#### `scripts/analysis/rope_transport/transport.py`（369 行）
- docstring 给出精确代理对象：`D* = min_{M,N} E_{D~w} || Mᵀ R_Ω'(D) N − R_Ω(D) ||_F²`，且 "For independent q/k with unit second-moment matrices, `E_{q,k}[(qᵀAk − qᵀBk)²] = ||A−B||_F²` exactly"。
- `hard_swap_residual(omega_src, omega_dst, support, weight)`（**闭式**）：
  ```python
  d = 2 * src.size
  delta = np.outer(np.asarray(support), dst - src)
  inner = 4.0 * np.cos(delta).sum(axis=1)
  return float(2.0 * d - float(np.asarray(weight) @ inner))
  ```
- `transport_residual(..., *, rank=None, max_iter=60, tol=1e-12, ridge=1e-12)` → `TransportResult(hard_swap, repaired, relative_hard_swap, relative_repaired, repairability, iterations, converged, rank, query_map, key_map, history)`；ALS "returns a feasible residual, an **upper bound** on the surrogate's minimum"。
- **依赖**：`numpy` only。**CPU 可跑**：是。**落盘**：无（`run_analysis.py` 负责）。

#### `scripts/analysis/rope_transport/nullband.py` 的两个评估器
- `phase_excess_risk(omega_native, omega_new, *, native_length, target_length)`：`trained = src·L_nat`；`deployed = dst·L_tgt`；`wrapped = trained ≥ 2π`；`unseen = clip(min(deployed,2π) − trained, 0, 2π)`；`excess = where(wrapped, 0, unseen)`；`turns = excess/2π`。返回 `per_channel_turns, mean_turns, max_turns, channels_at_risk`。
- `coverage_residual(..., deployed_points=512, trained_points=8192, weight=None)`：`c(D) = 4K − 4 max_{D'} Σ_k cos(w'_k D − w_k D')`。
- **docstring 关键判决**："Per-channel phase coverage is the wrong benefit axis, and the repository's own RULER numbers say so: the minimal-displacement table that is perfectly phase-safe by that measure scores zero." 以及 "Position interpolation drives this to exactly zero by construction, so the quantity is only informative together with the in-window cost `D*`."（**⇒ 显式两项目标结构**）
- `turn_budget_table(native, *, scale, beta, native_length, rope_base, ramp_turns=0.0, name)`：`turns = ω·L_nat/2π`；`keep = 1 if turns ≥ β`（或从 β−ramp_turns 的 smoothstep）；`arr = ω·keep + (ω/scale)·(1−keep)`。doc："`beta=1` with `ramp_turns=0` is the exact safety floor. `beta=32` with `ramp_turns=31` reproduces YaRN's schedule shape."

#### `scripts/analysis/rope_transport/weights.py`（105 行）
- `distance_weight(family, *, length, min_distance=0, alpha=1.0, max_points=4096, empirical=None)`，family ∈ {`causal`（weight = L−d，"the content-free ground truth for a packed training sequence"）、`uniform`、`powerlaw`（`(1+d)^{−α}`）、`empirical`}；`_subsample` 按 bin 保总权重（加权质心）。

#### `scripts/analysis/rope_transport/conditioning.py`（126 行）
- `basis_matrix`、`weighted_gram`、`pair_uniqueness(omega, support, weight, *, ridge=1e-10)`（逐对 Schur 补条件唯一性 ∈ (0,1]，用 `np.linalg.lstsq` 故意不解显式逆）、`range_resolvability(..., *, trained_omega=None, trained_length=None)` → entropy_effective_rank / stable_rank / min-max 特征值 / `phase_safe_fraction`。
- **注意**：目标为有效秩/Schur 唯一性 ⇒ **红线项**（INTEGRATION §5"几何-无符号类（全灭）"）。

#### `scripts/analysis/audit_finite_k_coupling.py`（195 行）★精确有限元投影
- **算什么**：**精确有限单元格投影 vs 点采样**的对照（重要：K3 若要离散化 F，这是唯一给出"单元格平均"精确闭式的件）。
- `clipped_affine(x, *, x_high, x_low) = clip((x_high−x)/(x_high−x_low), 0, 1)`
- `clipped_affine_antiderivative`（分段闭式）
- `cell_average(centers, *, spacing, x_high, x_low) = (F(upper) − F(lower))/spacing`
- 另含 `c_orth = 1/(1 − b^{−1/pairs})`、`x_zero = log(native_length/(2π·c_orth))`、`centers = x_zero − spacing·slots`、`eta = (x_high−x_low)/spacing`
- **默认常数**：`x_high=0.7382780681078285`、`x_low=0.366403835112904`；模型网格 `olmo_k64 (64,5e5,4096)`、`qwen_k64 (64,1e6,32768)`、`qwen_k32 (32,1e6,32768)`。
- **依赖**：`numpy` + stdlib。**CPU 可跑**：是。**落盘**：无。

#### `scripts/analysis/audit_finite_k_coupling.py` 的姊妹：`scripts/analysis/finite_k_cosh_regret_audit.py`（218 行）
- `cosh_quantile(mass, tau) = 1 − asinh((1−mass)·sinh(tau))/tau`（**EVQ-cosh φ 的精确分位数**）
- `continuous_components(tau)` → (‖ρ‖²₂, ‖Vρ‖²₂) 解析
- `histogram_components(pairs, tau)`：精确等质量分位直方图的二次型分量，含精确 Volterra 项 `width/(K²)·(r² − r + 1/3)`
- `surrogate_value(density_l2, volterra_l2, *, alpha, tau) = 0.5α(l2 + τ²·volterra_l2)`
- `asymptotic_constant(alpha, tau) = α/24·(τ² − τ·tanh τ)`
- `audit_tau(tau)` 验证 `K^{−2}` regret 标度与常数。
- **依赖**：`math`/`json`/stdlib only。[已验证] **CPU 可跑**：是。
- scope 行显式弃权 r2 / LM loss / 表选择。

#### `scripts/analysis/native_attention_kl.py`（137 行）
- `native_attention_kl(q, k, query_positions, native_inv, active_inv, *, gain=1.0, attention_scale=None)` → 逐 head/query `kl[Hq,Q]`、`mean_kl`、`slot_logit_delta_mean/variance`、`total_logit_delta_variance`、`off_diagonal_cancellation = Var_p(Σδ) − Σ Var_p(δ)`（负 = 抵消）、gain 矩。
- 内部 `_slot_logits` 隐式实现 RoPE：`a = q[:p]·k[:,:p] + q[p:]·k[:,p:]`，`b = q[:p]·k[:,p:] − q[p:]·k[:,:p]`，logits `= attention_scale·(a·cos(φ) + b·sin(φ))`。
- **依赖**：`numpy` only。**CPU 可跑**：是。**落盘**：无。
- docstring 边界："a mathematical diagnostic on Native latent vectors, **not** bitwise BF16 kernel equivalence, a V/downstream-feedback model, a task-loss predictor, or a profile selector."

#### `scripts/analysis/attention_phase_demand.py`（144 行）⚠ 有一个死 import
- `_demand(mass, inv_freq, bins)`：`demand = distance_probability @ (1 − cos(delta ⊗ omega))`，即 **`m(φ) = Σ_d p(d)(1 − cos(ω(φ)d))`**，梯形归一。
- `phase_demand(collection, bins=64)`：读 `.npz`（键 `inv_freq`/`mass`）。
- `layerwise_plan(...)`：`rho = cbrt((1−lam)*demand + lam)` → `quantile_phi` → `endpoint_anchored_omega`。
- **依赖断裂**（我已复现）：`layerwise_plan` 导入 `rebuttal.rebuttal_0723.experiments.demand_companding_5090.schedule`（`quantile_phi`/`endpoint_anchored_omega`）。`python3 -c "from rebuttal.rebuttal_0723.experiments.demand_companding_5090.schedule import quantile_phi"` → **ModuleNotFoundError**（该目录下只有 `functional_split_zero_training_5090`、`olmo2_function_constrained_lerope_5090`、`olmo2_sparse_spectral_retrofit_5090`、`resolution_aware_zero_training_5090`）。**⇒ `--r3-output` 路径在本分支已死**（代码归档在 `main_0726` 分支）。**只有 `phase_demand` 输出路径可用**（它不依赖那个 import）。[已验证]

#### `scripts/lib/rope/attn_hist.py`（120 行）
- `accumulate_distance_histogram(q, k, query_positions, max_distance, hist, block_q=128)`（torch，CPU 张量可用）
- `fit_power_law(hist, d_min=8, d_max=None)` → `{"alpha": −slope, "r2", "n_points"}`（log-log polyfit）
- `bootstrap_alpha_ci(per_sample_hists, n_bootstrap=1000, seed=42, ...)`

#### `scripts/analysis/third_axis_ceiling.py`
- `measure_weights(length, kind)`：`causal`（`p(d) ∝ L−d` 精确）或 `uniform`。这是 `weights.py:causal` 的另一份实现。

#### `scripts/analysis/lerope_profile_oracle.py`（416 行）
- 权重 = "`J_logomega^T (diag(p) − p p^T) J_logomega` 的逐带对角，在冻结 checkpoint 上平均"。
- `high_rate_profile(phi_nodes, w_nodes, bands)`：log w 分段对数线性插值 → `rho = exp(log_w/3)` → 中点逆 CDF。
- `compute_metrics`（支撑感知 log 波长 RMSE、支撑归一化形状 RMSE、EVQ↔LeRoPE 分段投影 α + 分类）。
- 转写 LeRoPE θ（arXiv:2607.10134v1 Fig.6）的**显示舍入值**，明示"not checkpoint data"。要求一个硬编码 SHA256（`32770ef1…`）的规范 probe JSON——**我未在本地验证该 SHA 对应的文件存在**。
- **依赖**：`numpy` + `matplotlib`（+ 一条路径用 torch）。

#### `scripts/analysis/dependency_spectrum_audit.py`（457 行）— 机器可复用
- **链条**：`K(r) → p(x) → ρ*(x) → θ_k`。
- 可用件：`Kernel` 类；`kernel_catalog(L=8192, W=1024)`（8 个命名核）；`quantile_grid(K, density_x, x_grid, theta_lo, theta_hi)`；`golden_max(f, a, b, tol=1e-10, max_iter=200)`；`lloyd_max_coverage(K, kern, x_lo, x_hi, init_x, iters=40)`（交替 assign/优化 + 坐标上升抛光）；`density_candidates`（指数 1 / 1/3 / 1/2）。
- **红线警告**：其目标 `coverage` / `stable_rank` 全是红线项。docstring 自述的结构性发现值得引用："**the ADDITIVE objective … is separable in k → all channels collapse to one point; coverage alone cannot define an allocation without an interference/budget term.**"（[已验证-代码注释] —— 这句话本身可作为 F 必须含非可分项的论据）
- **依赖**：`numpy` only。[已验证]
- **落盘**：写 `results/dependency_spectrum_audit_20260819/summary.json` —— **该目录在本分支不存在**（09-06 瘦身时剪除），无持久化 receipt。[已验证]

### 1.4 表格构造器（零/少参数臂，可作为单纯形上的候选点）

| 文件 | 函数 | 构造式 | 依赖 | 落盘 |
|---|---|---|---|---|
| `experiments/nongeometric_screen/gap_budget_transfer.py` | `prepare` | `gaps = log(ref[:-1]/ref[1:])`；`budget = mean(gaps[donors])`；`new_gaps[donors] -= budget/len(donors)`；`new_gaps[recipients] += budget/len(recipients)`；断言 `new_gaps.sum()==gaps.sum()`（**精确 log 范围守恒**） | numpy+stdlib | JSON receipt |
| `experiments/nongeometric_screen/scale_taper.py` | `prepare` | `weight = clip(log(W/period)/log(scale),0,1)`；`values = exp(log(ref) + weight*log(bm/ref))`，端点精确保 | numpy | `planned_controls/scale_taper.json`+queue |
| `experiments/nongeometric_screen/long_bridge.py` | `prepare` | period ∈ [32768,131072] 的槽；`step = 1/target_length`；反号对 LongBridgeSlower/Faster；记录 `phase_shift_at_native`/`phase_shift_at_target` | numpy+stdlib | JSON receipt |
| `experiments/nongeometric_screen/prepare_gap_probe.py` | `prepare` | 三臂 G1(24,48)/G2(36,36)/G3(36,48) over 槽 28,29，分子/306；断言三条恒等式 `gaps==[a ln4,(b−a)ln4,−b ln4]`、`center=−(a+b)ln4/2`、`sum(gaps)==0`（**精确三 gap 守恒**） | numpy+stdlib | `planned_controls/gap_probe.json`+queue |
| `scripts/analysis/build_boundary_matched_mrpro.py` | 149 行 | `increments(n) = [Fraction(6i(n+1−i), n(n+1)(n+2))]`；`cumulative(n,q) = Fraction(q(q+1)(3n+2−2q), n(n+1)(n+2))`；`independent_minimum(n)` 高消解 `Lz=1` 后归一，**断言等于 `increments(n)`**；`roughness(values)` = 相邻差平方和（端点零） | **stdlib only**（Fraction/math/csv/json） | `docs/research/ROPE_MRPRO_BM_CANDIDATE_20260908.json` + 2 CSV |
| `scripts/lib/rope/boundary_matched.py` | `boundary_matched_inv_freq(native_inv_freq, *, base, reference_length, scale)` | `m_q = q(q+1)(3N+2−2q)/(N(N+1)(N+2))`；32 圈/1 圈过渡 `turns = w·reference_length/(2π)`；`gain = 1+0.1·ln(scale)` | torch | (FP32 tensor, metadata) |
| `scripts/lib/rope/gap_capped.py` | `cumulative_minimum(width)` | 上限 `Fraction(2, width+1)`；`m_q = max(0, 1−2(N−q)/(N+1))`。scope："Componentwise least cumulative compression … **no task-gain guarantee**." | stdlib | 否 |
| `scripts/analysis/rope_transport/tables.py` | `budgeted_transport(base_table, uniqueness, *, scale, exponent=1.0, name=None)` | `move = (1 − norm_uniqueness)^exponent`；`arr = ω(1−move) + (ω/scale)·move`；doc："Displacement budget inversely proportional to measured in-window uniqueness … YaRN's fixed wavelength ramp is the **crude binary special case** of this rule." | numpy | 否 |
| `scripts/analysis/export_uniqueness_budgeted_tables.py` | 210 行 | `movement = (1 − normalized_uniqueness)^exponent`；`target = native(1−movement) + (native/factor)·movement`；**含 s=2.0/s=4.0 两张表的冻结 SHA256（hash-drift gate）**；原子写 .npy + receipt.json | numpy+torch(CPU) | .npy + receipt |
| `docs/research/rope_allocation_20260910/code/joint_mode_candidates.py` | 161 行 | 最低阶关系时钟的精确 rank-one 重定时。`FIRST, LAST = 24, 39`；`project(reference, native, n)`：`target = n·native/S`；`candidate = reference + n*(target − n·reference)/(n·n)`。**含 NATIVE_SHA / MR_SHA 硬校验**（`138c99b1…` / `33cbe3a4…`）。29 个零和候选 | numpy+stdlib | JSON |
| `scripts/lib/rope/schedules.py` | 369 行 | 调度工厂。`geometric_inv_freq`；`evq_cosh_phi(n_freqs, tau, midpoint=True)`；`evq_cosh_inv_freq`；`maxent_dilation_factors(pair_count, *, target_factor, lambda_)`（MaxEnt 密度 ∝ exp(λτ) 的中点分位：`r_i = [1 + q_i(target_factor^λ − 1)]^{1/λ}`）；`maxent_dilation_inv_freq`（反序耦合，doc 引 rearrangement inequality）；`build_inv_freq(method, head_dim, base, max_seq_len, rigid_j0=12, anchor_factor=0.0, tau=None, midpoint=True)` 支持 baseline/pi/yarn/anchored_hybrid/sigmoid/anchored_sigmoid/evq_cosh/evq_exp | torch | 否 |
| `scripts/lib/rope/official_yarn.py` | 16k | `find_correction_dim` / `find_correction_range` / `linear_ramp_mask(min_idx, max_idx, dim)` / `get_mscale(scale)` / `yarn_mscale(scale, attn_factor=1.0)` / `official_yarn_on_inv_freq` / `shared_index_yarn_control_on_inv_freq` / `repo_fixed_ramp_inv_freq` / `parity_vs_official_source` | torch | 否 |

### 1.5 非 CPU / 非数学件（登记以便排除）

- `experiments/nongeometric_screen/select.py`（191 行）：条件 selected-key 重放。`phase_rotate` / `prepare_record` / `replay(data, frequencies, head_mask=None, max_distance=None)` → `dict(gain, nmse, head_gain, target_mass, target_support, output_delta, max_selected_logit_delta)`；含恒等自检门（`identity['nmse']>1e-8 or max_selected_logit_delta>0.01` → raise）。`proposals(tables)` 枚举 E1_s{24..39}_{less,more}、E2_boundary{39,41}、E2_tail、E8_zero{40..63}。`aggregate(records)` → target_gain/split_gains/robust_gain/natural_output_nmse。**`prepare_record` 里 `.to('cuda')`（L25）⇒ 按写法非 CPU 可跑**。
- `experiments/nongeometric_screen/worker.py`：`Worker` 类，`install_table` 校验 `shape==(64,)` 且有限非负，装到 `model.model.rotary_emb`。需 `transformers`+`torch`+CUDA。
- `experiments/nongeometric_screen/capture.py`：`capture_row(worker, folder, row_id, ids, query_positions, targets, metadata)`；`rotate(x, cos, sin)`；`reference_positions(tokenizer, ids, references)`。需 CUDA（`device='cuda'` 硬编码）。
- `experiments/position_overnight/run_all.py`：远程 subprocess launcher（`/root/autodl-tmp/position_overnight_20260909`），**不是 CPU 数学件**。
- `experiments/position_overnight/report.py`：`paired_bootstrap(pairs, replicates, seed)`（"Resample clusters, preserving their rows; report the paired row-mean estimand"）、`quantile`、`cluster_id`、`budget_label`、`summarize`。**stdlib/numpy，CPU 可用**（收据统计）。
- `experiments/position_overnight/reuse.py`：`candidate_rows`、`append_missing`。
- `scripts/experiments/single_table_generation.py`（48k）：`target_score`、`condition_prompt`、`evaluate`、`retention`、`greedy`、`gold_prefix_trace`、`load_runtime`、`guard_resources`。需模型+GPU。

### 1.6 结果目录状态

- `experiments/nongeometric_screen/`：**本地只有 README.md + .py**，无 receipt。活动 run 在 `/root/autodl-tmp/nongeometric_screen_20260909`；"compact receipts are copied into the ignored local `results/` directory"。README 最新协议（2026-09-10）："focus on 128K … Do not automatically complete 32K panels."
- `experiments/position_overnight/`：`config.json` 指向 NOSA-1B(16K, topk_blocks 64) 与 Qwen2.5-3B-Instruct(prefix 8192, keep 0.25, horizon 512)。
- `analysis/unify_20260910/`：三份权威 md 齐全（INTEGRATION 130 行 / NEXT_DERIVATION 146 行 / STARTING_POINT 115 行），另 `tables/`、`digests/`、`digests_codex/`、`raw/`。
- `docs/research/rope_allocation_20260910/`：codex 归档权威副本（`agents/`、`assignments/`、`code/`、`coverage/`、`evidence/`、`recovered/`、`source_inputs/`、`archive_manifest.json`）。

---

## 2. 可直接复用的函数签名清单（按 KKT 求解链排列）

### 2.1 决策坐标系（Δ-单纯形 → 无约束）
```python
# docs/research/rope_allocation_20260910/code/sol16_frequency_calibration_reference.py
helmert_zero_sum(rows: int, *, dtype=torch.float64) -> Tensor[rows, rows-1]   # L14-23
mrpro_increments(width: int, *, dtype=torch.float64) -> Tensor[width]         # L26-29
class MrProAllocation16: eta: Parameter[16]; increments(); exponents(); inv_freq()  # L32-78
```
**推荐用法**：`Δ = softmax(log ε_MrPro + Helmert @ eta)`，`eta∈R¹⁶`，无约束；`eta=0` = MrPro。[已验证]

```python
# scripts/analysis/rope_transport/nullband.py
free_band_table(native, z, *, scale, band_start, rope_base, floor=None, name=None)
free_band_delta_table(native, z, delta, *, band_start, rope_base, start_delta=0.0, name=None)
encode_band(native, table, *, band_start, rope_base) -> (z, delta, start_delta)
```
**用途**：`free_band_delta_table` 是"自由总跨度"版本（把 ΣΔ=1 松绑成可搜的 delta），正是罚函数搜索所需的坐标。

```python
# scripts/analysis/third_axis_ceiling.py
z = cat([0.], cumsum(softmax(logits)))   # 在 search_best_found 内
# scripts/lib/rope/knot_allocation.py
knot_positions / interior_knot_u / init_gap_logits_from_table
```

### 2.2 解算器
```python
# experiments/nongeometric_screen/smooth_budget.py
solve(n, budget) -> (eps, certificate)     # active-set + 完整 primal/dual KKT 证书  L8-27
construct(tables, lo, hi) -> dict(Smooth_MrBudget, MrUni, construction)   # L30-47

# scripts/analysis/demand_companding_numeric.py
conditional_capp_optimize(K, protected_values, *, alpha=1.0, beta=1.0,
                          protected_indices=None, min_gap=1e-6, max_iter=300)
conditional_capp_from_native_frequencies(...)
solve_pointwise_quartic / quartic_density_solution / quartic_root / solve_quartic_balance
capp_discrete_objective(values, *, alpha, beta)

# scripts/analysis/verify_general_rope_allocation.py
common_direction(gradients, tolerance=1e-12, max_iterations=20000)
rotary_binary_loss(x, records)             # 解析梯度，有限差分 <1e-7
two_clock_minimum(q, scale, grid=2000)

# scripts/analysis/rope_transport/run_pareto_search.py
_eval_free(task)      # 罚函数目标：dstar + 50.0*max(0, risk@target − budget)
differential_evolution(pool, *, dim, lower, upper, band_start, budget,
                       popsize, iters, seed, seeds=None)
```

### 2.3 isotonic / PAVA
```python
# scripts/analysis/derive_native_isotonic_profile.py
isotonic_nondecreasing(values)         # 精确 PAVA（加权块合并）
pinned_profile(uniqueness)             # 钉端点 + PAVA，断言非降

# scripts/experiments/scale_transport/math.py
bounded_isotonic(values, weights, lower, upper)   # L25-42，带逐槽上下界的递减加权 PAVA
```

### 2.4 频率表 → 指标
```python
# scripts/analysis/rope_transport/transport.py
hard_swap_residual(omega_src, omega_dst, support, weight)            # 闭式 D0
transport_residual(omega_src, omega_dst, support, weight, *, rank=None,
                   max_iter=60, tol=1e-12, ridge=1e-12) -> TransportResult

# scripts/analysis/rope_transport/nullband.py
phase_safety_box(omega_native, *, native_length, target_length)
phi_floor(...); phase_excess_risk(omega_native, omega_new, *, native_length, target_length)
coverage_residual(omega_native, omega_new, *, native_length, target_length,
                  deployed_points=512, trained_points=8192, weight=None)
turn_budget_table(native, *, scale, beta, native_length, rope_base, ramp_turns=0.0, name=None)

# scripts/analysis/rope_transport/weights.py
distance_weight(family, *, length, min_distance=0, alpha=1.0, max_points=4096, empirical=None)

# scripts/analysis/audit_finite_k_coupling.py
clipped_affine(x, *, x_high, x_low); clipped_affine_antiderivative(...)
cell_average(centers, *, spacing, x_high, x_low)

# scripts/analysis/native_attention_kl.py
native_attention_kl(q, k, query_positions, native_inv, active_inv, *, gain=1.0, attention_scale=None)

# scripts/analysis/attention_phase_demand.py
_demand(mass, inv_freq, bins)          # m(φ) = Σ_d p(d)(1−cos(ω(φ)d))
phase_demand(collection, bins=64)

# scripts/analysis/finite_k_cosh_regret_audit.py
cosh_quantile(mass, tau); continuous_components(tau); histogram_components(pairs, tau)
surrogate_value(density_l2, volterra_l2, *, alpha, tau); asymptotic_constant(alpha, tau)

# scripts/analysis/third_axis_ceiling.py
measure_weights(length, kind); characteristic(weights, t); gram_blocks(omega, weights)

# scripts/experiments/scale_transport/math.py
quantile_moments(a, b); estimate_beta(a, b, ratio=2.)
replay(q, k, v, wo, frequencies, gain, positions, *, response=False)   # 精确完整可见-key 重放
background(q, k, frequencies, gain, distances)
```

### 2.5 证书/校验模式（非函数，但是可抄的工程范式）
- **SHA 冻结门**：`export_uniqueness_budgeted_tables.py`（s=2.0/4.0 表 hash）、`joint_mode_candidates.py`（NATIVE_SHA/MR_SHA）、`worker.py:sha/digest`、`rope_transport/run_pareto_search.py` 的脚本 SHA 写进 receipt。
- **恒等自检门**：`select.py:123-126`（identity nmse/logit-delta）、`sol16:110-137 self_check()`、`smooth_budget.py:33-34`（BM 恢复）。
- **fail-closed CPU 门**：`run_pareto_search._require_no_cuda()`。
- **诚实状态枚举**：`conditional_capp_optimize` 的 `{NUMERICAL_SMALL_SCALE_ONLY, BLOCKED_NONCONVEX_OR_SOLVER}` + `global_optimum_claim: False`。

---

## 3. 可作 F 零件的公式 / 定义 / 约束（带出处与证据等级）

标注沿用权威文档的四级：`[已验证]`/`[部分证据]`/`[假设]`/`[叙事-未验证]`。**代码中的每一行我实际读过。**

1. **Δ-单纯形与三坐标互换**：`ν_j = ω_j·S^{−m_j}`，`λ_j = S^{Δ_j}`，`Δ_j = m_{j+1}−m_j`，`Δ∈R¹⁷, Δ≥0, ΣΔ=1`。
   出处：`analysis/unify_20260910/NEXT_DERIVATION_KKT_PROBLEM.md:25-27`；代码实现 `docs/research/rope_allocation_20260910/code/sol16_frequency_calibration_reference.py:65-78`。[已验证=代码]

2. **守恒的正确表述**：`m_40 − m_23 = 1 ⇒ Σ_{j=23}^{39} Δ_j = 1`；**17 个过渡 gap 的总跨度锁定为 `ln S`**（原生 3.6697 + 额外 1.3863 = 5.0560 nats），**不是**"17 个 gap 之和 = ln S"。可行域 = 16 维单纯形。
   出处：`NEXT_DERIVATION_KKT_PROBLEM.md:43`（并明示两种口径不可混用）；`INTEGRATION_20260910.md:24,30,50`。[已验证]

3. **水床 = ΣΔ=1 的路径形**："任何窄化一处间隔必以拓宽他处为偿"；`∫ln E ≥ ln b − ln c`，取等 iff 均匀。
   出处：`NEXT_DERIVATION_KKT_PROBLEM.md:33`。[已验证=数学（文件自述）]

4. **端点锁定 I1 / I2**：`m_j=0 (j≤23)`；`m_j=1 (j≥40)`。
   出处：`NEXT_DERIVATION_KKT_PROBLEM.md:41-42`，**强度口径**为"强基线设计约束，非零容忍定理"，`s28_less` 暗示槽 28 附近有 ~0.033 可回收量。[已验证=面板]

5. **KKT 三段结构预言**（待证明，不预设成立）：`Δ_j=0` 的 bank 平台（`∂F/∂Δ_j(0) > μ`）、`Δ_j>0` 的过渡桥（边际率相等 = **水填充**）、`m=1` 尾部平台。
   出处：`NEXT_DERIVATION_KKT_PROBLEM.md:53-57`。[假设-待证；有 sol04/sol06 的数值涌现作为 [部分证据]，见 `INTEGRATION_20260910.md:79`]

6. **目标泛函**：`F[Δ] = L_near[m(Δ)] + L_far[m(Δ)]`，`Δ→m` 线性双射。
   出处：`NEXT_DERIVATION_KKT_PROBLEM.md:47`。[假设-建模]

7. **L_near 候选载体**：`Σ_j w_j(r_j)(ν_j−ω_j)²`（F3 的积分形式，**动态算子量**，非静态几何代理），`w_j` 应随 `r_j`（bank 圈数）增长。**F6 OLMo 反例禁止它单独成 F**。
   出处：`STARTING_POINT_YARN_VS_MRPRO.md:103`；`INTEGRATION_20260910.md:92`。[部分证据+禁令]

8. **L_far 必须经 η**：`η_j = −∂log ν_j/∂log S`。
   YaRN：`η_Y(t,S) = t/[S(1−t)+t]`，S→∞ 时 `η_Y→0`（**饱和**）。
   MrPro：`η_M = m_q`（**幂律不饱和**）。
   出处：`STARTING_POINT_YARN_VS_MRPRO.md:50-53`。[已验证-推导]

9. **标准化 YaRN 的 m 坐标形式**：`ν_j^Y = ω_j(1 − t + t/S)` ⇒ `m_Y(t) = −log[1−(1−1/S)t]/log S`，**m_Y′>0 且 m_Y″>0**（递增且凸）。
   出处：`STARTING_POINT_YARN_VS_MRPRO.md:19-21`。[已验证-推导+代码对账；**并因此证伪"YaRN 递减 vs Pro 递增"叙事**]

10. **MrPro-Pro**：`ν_j^M = ω_j S^{−m_q}`，`m_q = q(q+1)/(N(N+1))`（二次）。**是设计假设，不是任务目标的解**。
    出处：`STARTING_POINT_YARN_VS_MRPRO.md:25-27`。代码：`sol16:26-29`（`ε_i=2i/(width(width+1))`）。[已验证-论文原文]

11. **高码率量化理论**（`demand_companding_numeric.py` docstring L12-30）：
    `D[ρ] = (12K²)^{−1}∫ m(x)/ρ(x)² dx`；Euler `ρ = m^{1/3}/Z`；`D* = Z³/(12K²)`；`H(m) = 1 − Z³`。
    λ 族 `m_λ = (1−λ)m + λ/|I|`。逐点四次 `αρ⁴ + νρ³ − 2(m+ε) = 0`。[部分证据-代码注释与实现一致，未独立复算]

12. **KKT 最优先复用泛函**（codex 理论核心 §4.1，`NEXT_DERIVATION_KKT_PROBLEM.md:104-106` 转述）：
    `J[h] = (1/2)∫₀¹[α/h(u) + β(1−u)²h(u)]du`，`h>0`，`∫h=1`，`h=Q′(u)`。
    离散化后 = 单纯形 `{a_i>0, Σa_i=A}` 上的**可微凸优化**。`1/h` 项 = 间隔 `a_i` 的凸惩罚。[已验证-推导（文件自述在 codex 核心 §4.1 复核过严凸性/唯一正解/ρ″=τ²ρ）]

13. **EVQ-cosh φ 的精确分位数**：`φ_k = 1 − asinh((1−u_k)·sinh τ)/τ`；代码 `cosh_quantile(mass, tau) = 1 − asinh((1−mass)·sinh(tau))/tau`（`finite_k_cosh_regret_audit.py`）。`ω_k = base^{−φ_k}`。[已验证-代码]

14. **相安全盒（精确可行集）**：`w'_k·L_tgt ≤ w_k·L_nat` OR `w_k·L_nat ≥ 2π`；log 坐标下 = **下界** `φ'_k ≥ φ_k + log(s)/log(base)`（unwrapped block 内），盒外无约束。
    出处：`scripts/analysis/rope_transport/nullband.py` docstring。[已验证]

15. **评级的正确姿势（nullband 自身判决）**："Per-channel phase coverage is the wrong benefit axis … Position interpolation drives this to exactly zero by construction，所以该量只有与窗内代价 `D*` 联用才有信息"；"The binary phase-safe fraction makes the retrofit problem degenerate: the feasible set is a box, `D0` is separable, and the constrained optimum is simply 'move as little as the box demands'. **The graded version is the one with structure.**"
    出处：`nullband.py` docstring（`coverage_residual` / `phase_excess_risk`）。[已验证-代码注释，且是**建模级**论据]

16. **运输残差代理**：`D* = min_{M,N} E_{D~w}||MᵀR_Ω'(D)N − R_Ω(D)||²_F`；`D0 = E||R_dst − R_src||²_F`（硬交换闭式）。
    出处：`transport.py` docstring；`run_analysis.py`。[已验证]

17. **覆盖残差**：`c(D) = min_{D'}||R_Ω'(D) − R_Ω(D')||²_F = 4K − 4 max_{D'} Σ_k cos(w'_k D − w_k D')`。出处：`nullband.py:coverage_residual`。[已验证-代码]

18. **联合模式输运算子**：`ν_c = ν_M + n(nᵀω_native/4 − nᵀν_M)/(nᵀn)`，`n∈{[1,−1]×15, [1,−2,1]×14}`（29 个零和候选，保 Σ频率、端点逐位固定、严格递减）。**关键实测：MrPro 对 29 个相邻关系时钟的 0/29 做了 ×4 重定时**；3/29 候选把某槽推得比 MrPro 还快——**null band 假设被联合模式否定**。
    出处：`INTEGRATION_20260910.md:72`；代码 `docs/research/rope_allocation_20260910/code/joint_mode_candidates.py`。[部分证据-CPU 复算（文件自标 NO_ROLE_OR_CAPABILITY_QUALIFICATION）]

19. **sol15 §7 统一陈述**（唯一可防御措辞，`INTEGRATION_20260910.md:58` 逐字）——本文不复制，只标注：**"这些受限归约中没有一条确立与 checkpoint 无关的 LM 最优曲线。"** [叙事-未验证，但被五份审计共同背书]

20. **可辨识性边界**（`INTEGRATION_20260910.md:71`）：自由系数规差下任意密度可被吸收；`"唯一最优密度"式声明一律降格为"给定声明协方差下的解"`。[部分证据]

21. **水床的幸存形式（sol17，含精确 DP）**：固定对数 `Σn_i=K` + 嵌套协方差 `C_il = (α/Δ)1{i=l} + β·min(x_i,x_l)` 下精确 DP
    `min Σ_i[α/(2Δ)n_i² + (βΔ/2)T_i² − K h_i n_i]`，`T_i = Σ_{l≥i} n_l`，
    后向递推 `F_i(t) = (βΔ/2)t² + min_{0≤n≤t}{ (α/2Δ)n² − K h_i n + F_{i+1}(t−n) }`，`O(BK²)` 全局最优。
    **`βΔT_i²/2` 就是水床写成守恒律**；iid 通道噪声会杀死该项 ⇒ 水床要求**声明的相干/嵌套协方差**。常数 h 归约出离散 Cosh 递推。
    出处：`INTEGRATION_20260910.md:67`。[部分证据-文档公式，**仓库内无对应实现**]

---

## 4. 已核实数字（每条带出处）

| 数字 | 值 | 出处 |
|---|---|---|
| Qwen2.5-3B 部署几何 | `W=32768, S=4, L=131072, θ=10⁶, K=64` | `INTEGRATION_20260910.md:50` |
| 原生频率 zero-based | `ω_j = θ^{−j/64}`，末对 `θ^{−63/64}≈1.24e-6`（**非 θ^{−1}**） | `INTEGRATION_20260910.md:50` |
| gain | `g² = 1 + 0.1·ln4 = 1.1386294` | `INTEGRATION_20260910.md:50`；代码 `scale_transport/math.py` 与 `boundary_matched.py`（`gain = 1+0.1·ln(scale)`） |
| 过渡段总 log 跨度 | `ln S = 5.0560 nats`（native 3.6697 + 额外 1.3863） | `NEXT_DERIVATION_KKT_PROBLEM.md:43` |
| 面板 near/far（36 行） | MrPro **87.22/78.13**；s28_less 87.2/**83.3**；LBS 80.6/80.1；P2 72.9/81.7；MrUni 64.6；Smooth 87.2/68.3；pair28_29 **−4.17pp**；HighGapToLong −17.1/−10.8；E2 54.7；E8 50.6；E3 gain074 98.3/75.3 | `INTEGRATION_20260910.md:52`；`NEXT_DERIVATION_KKT_PROBLEM.md:50-51,84-94` |
| OLMo-2-0425-1B 16K 350 条 | BM **41.67%** vs MrPro **7.09%**（+34.59pp，156W/9L；multikey_3 双方皆 0；EOS BM119/MrPro197） | `INTEGRATION_20260910.md:52` |
| prefix/read 交叉 | −2.125 / −1.125 / −1.0 / +0.25（收益不限于固定 Q/K 读出） | `INTEGRATION_20260910.md:52` |
| 32K 全模型 CE 梯度（4 行） | `‖grad‖ 50.9–688.7`；**响应集中于被 MrPro 面冻结的快槽 0–6**，过渡槽 24–39 几乎无响应 | `INTEGRATION_20260910.md:52` |
| MrPro m 值 | `m24=.0065 … m39=.8889`；E1_s28 `m28=.0654`；P2 `m36–39=1`；E2 `m40=1.1557` | `NEXT_DERIVATION_KKT_PROBLEM.md:29` |
| YaRN/MrPro 槽24 相对降频 | Qwen N=17,S=4：YaRN **4.4118%** vs MrPro **0.9020%**；Llama3 N=17,S=16：5.5147% vs 1.7958% | `STARTING_POINT_YARN_VS_MRPRO.md:40` |
| 局部旋转导数扰动平方和比 | Qwen 现配置 `Σ(ν^M−ω)²/Σ(ν^Y−ω)² = 0.4841` | `STARTING_POINT_YARN_VS_MRPRO.md:42` |
| OLMo 反例 | MrPro 扰动同样降到 ~**47.8%**，但独立 72 条长端 MrPro 2.78% < YaRN 6.94% ≪ **BM 51.32%** | `STARTING_POINT_YARN_VS_MRPRO.md:75` |
| A/B 交点 | Qwen 4× A={24–37} B={38–39}；Llama3 16× A={19–24} B={25–34} | `STARTING_POINT_YARN_VS_MRPRO.md:69` |
| 四格族 Σm 实测范围 | `−0.01826 … +0.00185`（**零和频移 ΔΣm≠0**）；P2 质心 Σm 34.18 > MrPro 29.33 | `INTEGRATION_20260910.md:30`；`NEXT_DERIVATION_KKT_PROBLEM.md:76` |
| Smooth 反代理判决 | Smooth_MrBudget 几何全赢、far 端 **68.3 vs MrPro 78.13**（near 打平） | `INTEGRATION_20260910.md:28` |
| 根（Σcos 首零）证伪 | MrUni 82.2K > MrPro 80.3K 而 32K 64.6≪87.2；E2/P2 同根 109.1K 一崩一 81.7；s28 修复 +5.2pp 时根几乎不动 80.7 vs 80.3 | `NEXT_DERIVATION_KKT_PROBLEM.md:103`；`STARTING_POINT_YARN_VS_MRPRO.md:86` |
| `smooth_budget` 预算常数 | `solve(n,(n-1)/3)` → Smooth_MrBudget；`solve(n,(n-1)/2)` → MrUni；BM 恢复自检 `atol=1e-11` | `smooth_budget.py:31-34` |
| `smooth_budget` 边界 | qwen3b (23,40)、olmo1b (14,32) | `smooth_budget.py:52` |
| `prepare_gap_probe` 基线 | `m28=30/306, m29=42/306`，step `6/306` | `prepare_gap_probe.py:17` |
| `prepare_gap_probe` 三臂分子 | G1(24,48)/G2(36,36)/G3(36,48) | `prepare_gap_probe.py:15` |
| `audit_finite_k_coupling` 默认盒 | `x_high=0.7382780681078285, x_low=0.366403835112904` | `audit_finite_k_coupling.py`（默认参数） |
| `sol16` DOF 与精度 | `eta ∈ R¹⁶`（width−1=16）；`eta=0` 与 MrPro 逐位差 `atol=2e-16`；gradcheck `eps=1e-6` PASS | `sol16_frequency_calibration_reference.py:59,114,136-137` |
| `knot_allocation` 常数 | 5 内点 / 7 结点 / `MAX_GAP_LOGIT_DELTA=4.0`；`METHOD_ID="frozen_checkpoint_z5_knot_allocation_v1"` | `knot_allocation.py` |
| `joint_mode_candidates` 哈希 | `NATIVE_SHA='138c99b109d7affbfba059e435670918fe4531bce4709b6e86f3f22f7ef80f6e'`；`MR_SHA='33cbe3a40994ac2a79126a14ce30282867bd6d49b74c1ada4d4cce8a7e76016f'`；`SCALE=4.0, FIRST,LAST=24,39` | `joint_mode_candidates.py:16-21` |
| `run_pareto_search` 罚系数 | `50.0`；`over = max(0, risk@target − budget)` | `run_pareto_search.py:_eval_free` |

---

## 5. 死路登记册（已证伪/已失败，不得再试）

> 本节**全部依权威文档**；凡是代码里能复核的另标代码出处。红线系条目重申自用户纪律。

1. **静态几何代理量入 F**（Σcos 首根、碰撞能、覆盖率、平滑度、有效秩、Gram、能量、movement-MAE、轨道计数、response energy）。反例：Smooth_MrBudget 几何全赢但 far 端 68.3 vs MrPro 78.13；根排序与能力排序显著失序。
   出处：`INTEGRATION_20260910.md:28,84`；`NEXT_DERIVATION_KKT_PROBLEM.md:103,111`；`STARTING_POINT_YARN_VS_MRPRO.md:86`。[已验证=双源 CPU+面板]
   **代码侧注意**：`third_axis_ceiling.py`（stable_rank）、`dependency_spectrum_audit.py`（coverage）、`conditioning.py`/`export_uniqueness_budgeted_tables.py`（Gram/Schur uniqueness）的目标**全在红线内**；其**机器**（重参数化、PAVA、证书范式）可复用，**目标不可**。

2. **Σm（质心）当守恒量**。`ΔΣm≠0` 实测 `−0.01826…+0.00185`；Σm 是**自由决策变量**。
   出处：`INTEGRATION_20260910.md:30`；`NEXT_DERIVATION_KKT_PROBLEM.md:107`。[已验证-实测]

3. **"所有赢家同向移预算"**。6Pro 已推翻强表述；只允许弱版（bank 边缘卸载 ∪ 危险区右段完成）。
   出处：`NEXT_DERIVATION_KKT_PROBLEM.md:76,112`。[已验证]

4. **"17 个 log-gap 之和 = ln S"** 的错误口径。正确 = 总跨度锁定 + 原生部分分开计。
   出处：`NEXT_DERIVATION_KKT_PROBLEM.md:43,114`。[已验证]

5. **"YaRN 递减 vs MrPro 递增"叙事**。两者都凸都递增（`m_Y″>0`）；Qwen S=4 上逐槽 `max|Δm| = 0.034`。
   出处：`STARTING_POINT_YARN_VS_MRPRO.md:9,21`；`INTEGRATION_20260910.md:87`。[已验证-推导+代码对账]

6. **逐槽可加性**。pair28_29 → **−4.17pp**（负！）；原因 = 跨 key 相干与共享 head/W_O 对消（同非负能量总导数 0 vs 4）。
   出处：`INTEGRATION_20260910.md:52,86`。[已验证-面板]
   **推论**：slot 级收益非加性 ⇒ **逐槽贪心列表法先天失效**（sol07 的正确解释器）。

7. **冻结 checkpoint 上的密度/多重集/排序参数化**（坐标类错误）。同多重集置换 NLL 3.104→6.865、Qwen core-4 0.70→0；联合置换频率+学习系数槽才是恒等。
   出处：`INTEGRATION_20260910.md:85`。[已验证]

8. **scratch 密度移植到冻结**：Geo↔Cosh 运行时互换 PPL 7.14↔76.20 / 7.16↔23.05。
   出处：`INTEGRATION_20260910.md:85`。[已验证]

9. **解①对象装④对象**（精确-原子最优 / 光滑-Cosh 代理 / 有限-K 整数计数 / 冻结-带标签表，两两不可互换）。
   出处：`INTEGRATION_20260910.md:69,85`。[已验证-多份审计一致]

10. **Taylor/Jacobian 分数跨全 S=4 表**（相对误差 71–468%，相位 22.74/90.97 rad）。
    出处：`INTEGRATION_20260910.md:86`。[已验证]

11. **无符号二次项入 F / iid 噪声 → α∫ρ²**（旋转不变，无碰撞项）。
    出处：`INTEGRATION_20260910.md:84`。[已验证]
    **代码侧对照**：`verify_general_rope_allocation.py` 的 `rotary_binary_loss` 是**带符号**的（用 sin/cos 的显式相位），所以它不踩这条；而 `demand_companding_numeric` 的 `capp_discrete_objective` 用了 `min(points_i, points_j)` 核（**非**平移不变的），也不踩。

12. **不可辨识定理外衣**：`universal 不可辨识定理只约束已测的 model-blind unordered 类`。引用措辞必须缩窄为"没有**已测的 model-blind 无序**统计量能认证冻结部署"。
    出处：`INTEGRATION_20260910.md:87,99`（FLAG-1）。[已验证-裁决]

13. **直接优化两条死路**：64 维行为梯度未开 holdout 即败；direct-z 定支撑 pilot 挂门。**优化器必须在可辩护统计对象下游**。
    出处：`INTEGRATION_20260910.md:88`。[已验证]
    **对 K3 的直接影响**：`third_axis_ceiling.search_best_found`、`knot_allocation` 的 `init_gap_logits_from_table` 这类"直接对表做无约束优化"路径，**历史上已挂门**；K3 的合法性必须建立在 §8-1 的角色矩门或 §8-3 的 sol18 测试之上。

14. **attention ≠ generation**：cross-cache 上 BM 读 MrPro 前缀能对、MrPro 读 BM 前缀仍错；record coverage 29.75→67.1 而散文精确答 7/8→6/8。
    出处：`INTEGRATION_20260910.md:88`。[已验证]
    **对 R6 的直接影响**：`select.py:replay` 的 conditional replay **自述**"Conditional selected-key replay; omitted key changes and upstream state changes are not modeled. Whole-model tests are mandatory."

15. **τ ≈ d_head/√L**（Phase 19 "PASS" 是脚本约定，9.6% 均值/33.3% 最大锚误差）。出处：`INTEGRATION_20260910.md:87`；`MEMORY.md` Phase 19 段。[已验证]

16. **`attention_phase_demand.py --r3-output` 路径已死**（ModuleNotFoundError，我本地复现）。不要在其上建任何东西；只有 `phase_demand` 输出路径可用。[已验证-本地复现]

17. **`results/dependency_spectrum_audit_20260819/` 不存在**（脚本会写那里，但 09-06 瘦身已剪除）。**不要假设该 receipt 可读**。[已验证]

18. **`experiments/nongeometric_screen/` 本地无 receipt**（只有 README + .py）。任何"读本地 receipt 取数"的做法都不成立。[已验证]

---

## 6. 与权威文档的冲突 / 不一致（须显式指出）

### C1. 约束集：经验端点锁 vs 代数相安全盒
- **权威文档侧**：I1/I2（`m_j=0 (j≤23)`、`m_j=1 (j≥40)`）是**面板导出的经验端点锁**，文档自述"可证伪性：少数违例证明代价高，不排除某个小位移自由度的存在"（`NEXT_DERIVATION_KKT_PROBLEM.md:41`）。
- **代码侧**：`nullband.phase_safety_box` 给出的是**精确代数盒**（`w'_k·L_tgt ≤ w_k·L_nat` OR `w_k·L_nat ≥ 2π`），换元后是逐槽**下界**，且 doc 明确"盒外无约束"。
- **冲突点**：代码的盒**允许** `m_j≠0`（只要相位安全），I1 不允许。**这两个可行域不是同一个**。且 `low/high` 边界的合法性来源不同（面板 vs 代数）。
- **建议**：K3 必须声明用哪个可行域；若用 Helmert 坐标（自动 ΣΔ=1），则 I1/I2 是**额外的盒约束**而非结构性约束，需要在坐标里显式加（或作为罚项）。

### C2. 预算固定 vs 预算自由
- **权威文档侧**：`Σ_{j=23}^{39} Δ_j = 1` 是**恒等式**（`NEXT_DERIVATION_KKT_PROBLEM.md:43`，"不是修辞"）。
- **代码侧**：`nullband.free_band_delta_table` **故意松绑**带内总 log 跨度（`start_delta`/`delta` 自由），docstring 解释理由："there is no safety floor: feasibility on the risk axis is expressed by the objective, not by the parametrisation"。
- **冲突点**：若 §1.3 的守恒是硬的，则 `free_band_delta_table` 搜索空间**超出可行域**（它允许 `m_40−m_23 ≠ 1`）；若守恒是软的（= 一个应被验证的约束），则由 `free_band_delta_table` + 罚函数是正当的。
- **建议**：这是 KKT 建模的第一个必须裁决的分叉。注意文档 §1.2 说水床"在路径形下就是 ΣΔ=1"，把守恒当恒等式；而 `run_pareto_search` 的整个设计依赖 `free_band_delta_table`。**两边只能选一边。**

### C3. 16 DOF 坐标的两个版本（文档已自认，但代码只有一份）
- `INTEGRATION_20260910.md:104`（FLAG-6）：sol16 Helmert 保序构造 vs sol18 槽坐标加约束，"**选 sol16 形式做无约束求解器，出表前用 sol18 坐标报槽值**"。
- 代码现状：`docs/research/rope_allocation_20260910/code/sol16_frequency_calibration_reference.py` **存在**；sol18 的槽坐标形式在 `code/` 目录下**没有对应文件**（只有 `astra07_lifecycle.py`、`astra09_rule.py`、`full_model_response.py`、`joint_mode_candidates.py`）。
- **⇒ 兑现 FLAG-6 的裁决需要先实现 sol18 坐标（约 10 行），或直接从 `MrProAllocation16.exponents()` 反算。**

### C4. "水填充 + isotonic"（sol04）在仓库里没有实现
- `INTEGRATION_20260910.md:79` 把 sol04 列为四条可执行配方之首（"水填充+isotonic——三段结构在解中**涌现**（K1 定理的数值伙伴）"）。
- 代码现状：**仓库里没有 sol04 脚本**。水填充只能从 `smooth_budget.solve` 的内点条件（凸问题的 active-set KKT ⇒ 自由面上 `∂F/∂Δ_j = μ` 常数）间接获得；isotonic 只有 `derive_native_isotonic_profile.isotonic_nondecreasing` 与 `scale_transport/math.py:bounded_isotonic`。
- **⇒ K1 的"数值伙伴"需要**从这两件拼装**，不是现成的。

### C5. sol17 的 DP 只在文档里
- `INTEGRATION_20260910.md:67` 给出精确 DP（`O(BK²)` 后向递推）与水床守恒项 `βΔT_i²/2`。
- 代码现状：**无任何 DP 实现**。`dependency_spectrum_audit.py:lloyd_max_coverage` 是最近邻（交替 assign/优化），但目标是 coverage（红线）且是连续松弛不是 DP。
- **⇒ 若 K1/K3 要用水床的 DP 形式做对照或作为 F 的一个可分近似，必须**新写**。

### C6. sol15/sol14 的"带符号 log-partition margin"在代码中无评估器
- `INTEGRATION_20260910.md:64`（共同精确对象）：`z_t(ν) = c_t + Σ_j{A_tj cos(ν_j d_t) + B_tj sin(ν_j d_t)}`；`M_r(ν) = log Σ_{S_r}e^z − log Σ_{D_r}e^z`；`p* = σ(M_r)`。四家同式的分母/稀释项 `η_r = log((M_r−1)(1−a_r)/a_r) = log(H(1−a_r)/a_r) = −log N_r = log|D_r|`。
- 代码现状：**没有以角色（native/far × source/hard-distractor）分组的 margin 计算器**。最接近的是 `native_attention_kl.py`（但它是 KL，不是 margin）与 `scale_transport/math.py:background`（`logsumexp(z) − log k`，**只有分母项，无分子角色**）。
- **⇒ 与 §8-1 "标签化角色矩从未被测"一致：这是唯一阻塞证据，且**缺工具**。K3 若要在可辩护统计对象下游求解，这个评估器是**必须新写**的第一件。**

### C7. `attention_phase_demand.layerwise_plan` 的 `ρ = cbrt(...)` 与文档的 `ρ ∝ m^{1/3}` 一致，但该函数不可用
- 文档侧：`demand_companding_numeric.py` docstring 的 Euler 解 `rho = m**(1/3)/Z`（[部分证据]）。
- 代码侧：`layerwise_plan` 里的 `rho = cbrt((1−lam)*demand + lam)` 与之一致，**但该函数因死 import 不可运行**（见 C-死路 16）。
- **⇒ 该逻辑必须从 `demand_companding_numeric.py` 的 `high_rate_density` 路径重建。**

### C8. 数字口径不一致（文档已裁决，代码未统一）
- `INTEGRATION_20260910.md:100-102`（FLAG-3/FLAG-4）：Smooth 的"差 9.79 分"必须改成 near/far 分解；"Qwen-1.5B" 规范记为 **1.485B**。
- 代码现状：`smooth_budget.py` 输出的 receipt 只有 `scope` 字符串，不带 near/far 分解；命名用 `qwen3b`/`olmo1b`。
- **⇒ 若 K3 的输出去对账 §3 的 14 点表，命名与分解口径必须按 FLAG-3/4 修正。**

---

## 7. 开放问题（给 workflow-3）

1. **C2 的裁决**：ΣΔ=1 是硬恒等式（→ 用 Helmert 坐标 16 DOF + 罚项）还是应被验证的软约束（→ 用 `free_band_delta_table` 的 18 维自由坐标）？这一条决定了 K3 用哪个坐标系。
2. **C6 的工具缺口**：谁来实现带角色标签的 `margin` / 角色矩计算器（§8-1 的阻塞件）？在它落地前，K3 的解只能在"不可辩护统计对象"下游做，**与 §5 死路 13 直接冲突**。
3. **`L_far` 的可计算形式**（文档 Q9，"兼容成本的可计算形式未完成"）：`nullband.phase_excess_risk`（mean/max turn over budget）是一个**候选**——它是动态的（依赖 `L_nat`/`L_tgt` 真实旋转差）、带符号的、经 `η` 的。**这与 §4.2/§6 的 L_far 要求是否相容，需要判定。** 注意文档 §1.4 明确否定了"单槽弧语言"（"r_j>1 的槽超 D_j 后圆周已覆盖，故 L_far 不能按单槽弧语言定义"），而 `phase_excess_risk` 恰恰是**逐槽**的 ⇒ **可能踩这条**。
4. **`run_pareto_search` 当年在 `dstar`（运输残差）上的搜索结果是否已落盘？** 若已落盘，它是 K3 的现成 baseline；若未落盘，该脚本的历史结论不可引用。
5. **罚系数 50.0 的来源**：`_eval_free` 的 `50.0` 是手选还是有标定？若非标定，K2/K3 需要重新标定（罚系数决定解的可行性违反量）。
6. **`x_high/x_low` 默认值**（`0.7382780681078285 / 0.366403835112904`）在 `audit_finite_k_coupling.py` 中的来源与适用范围（我只看到它们是默认参数，未追到推导）。
7. **`lerope_profile_oracle` 的硬编码 SHA `32770ef1…` 对应文件是否在本分支存在？** 我未在本地验证；若不存在，该脚本的 `high_rate_profile` 路径不可复现（`compute_metrics` 的可复用性随之存疑）。
8. **`official_yarn.py` 的 `parity_vs_official_source`** 是仓库里唯一的"与官方实现逐位对账"工具。K4 若要引 YaRN 的 `m_Y(t)` 公式，应先用它确认本仓库的 YaRN 实现与公式一致（避免 `STARTING_POINT` §1 提到的"论文最终表格用的确切提交未核对"问题）。
9. **`smooth_budget.solve` 能否直接承载 K1？** 它是**最小粗糙度**（`εᵀLε`）目标，不是 `L_near+L_far`。它的价值是**证书模式**与**BM 恢复自检**，不是 F 本身。K1 若要"三段结构涌现"，需要把 F 的 `∂L_near/∂m` 形状替换进 L 的位置——这是一个**泛函替换**，不是参数调整。

---

## 8. 覆盖度

**读了（本轮 + 前序）**：
- 三份权威 md 全文：`INTEGRATION_20260910.md`（130 行）、`NEXT_DERIVATION_KKT_PROBLEM.md`（146 行）、`STARTING_POINT_YARN_VS_MRPRO.md`（115 行）。
- `docs/research/rope_allocation_20260910/code/sol16_frequency_calibration_reference.py`（全文 142 行）、`joint_mode_candidates.py`（head 60 行）。
- `scripts/experiments/scale_transport/math.py`（全文 84 行）。
- `scripts/analysis/`：`demand_companding_numeric.py`（关键区间）、`verify_general_rope_allocation.py`、`build_boundary_matched_mrpro.py`、`derive_native_isotonic_profile.py`（1-130）、`audit_finite_k_coupling.py`、`finite_k_cosh_regret_audit.py`、`dependency_spectrum_audit.py`（全文）、`native_attention_kl.py`、`attention_phase_demand.py`、`third_axis_ceiling.py`、`export_uniqueness_budgeted_tables.py`、`lerope_profile_oracle.py`（1-270）。
- `scripts/analysis/rope_transport/`：`nullband.py`（全文 585 行）、`run_pareto_search.py`（1-110）、`transport.py`、`weights.py`（全文）、`tables.py`、`conditioning.py`（126 行）、`run_analysis.py`。
- `scripts/lib/rope/`：`schedules.py`（369 行）、`boundary_matched.py`、`gap_capped.py`、`attn_hist.py`（120 行）、`knot_allocation.py`（head 50）、`official_yarn.py`（签名扫描）、`__init__.py`。
- `experiments/nongeometric_screen/`：`smooth_budget.py`（全文）、`prepare_gap_probe.py`（全文）、`scale_taper.py`（全文）、`gap_budget_transfer.py`、`long_bridge.py`、`select.py`（全文 191 行）、`capture.py`（head 45 + `capture_row`）、`worker.py`（head 45）、`README.md`。
- `experiments/position_overnight/`：`run_all.py`、`config.json`、`report.py`、`reuse.py`。

**AST 扫描但未逐行读**：`scripts/lib/rope/` 的 `fixed_support_z.py`、`hat_projection.py`、`learnable_evq.py`、`target_free.py`、`inject.py`、`generation_contract.py`、`length_conditioned_budgeted.py`；`scripts/analysis/` 的 `full_rope_collision_audit.py`、`optimize_scale_conjugacy.py`、`rope_qk_operator_gram.py`、`shared_frequency_response.py`、`analyze_cached_frequency_response.py`、`attention_fisher_50m_probe.py`、`scale_orbit_validation.py`、`export_static_rope_baselines.py`、`export_frozen_coupling_transport.py`、`maxent_dilation_allocation.py`、`dilation_allocation_fork.py`、`verify_signed_lag_kway_gap.py`、`finite_scale_covariance.py`、`audit_qwen_p2_full_lag.py`、`analyze_p2_transfer_mechanism.py`、`allocate_binding_states.py`、`analyze_binding_states.py`、`tau_direct_optimization.py`、`tau_exact_derivation.py`、`verify_softmax_v2.py`、`compute_eta_vp.py`、`summarize_native_reference_calibration.py`、`industrial_128k_feasibility.py`；`scripts/experiments/single_table_generation.py`（48k）、`matched_transfer_round.py`；`experiments/nongeometric_screen/` 的 `project.py`、`operators.py`、`distance_operator.py`、`holdout_eval.py`、`paired_summary.py`、`summarize.py`、`finish_screen.py`。
**其中我认为对 KKT 求解最可能有增量价值、建议下一轮补读的 4 个**：`tau_direct_optimization.py`（含 `optimize_tau`，是"直接优化"死路的原始现场，值得看失败模式）、`optimize_scale_conjugacy.py`（含 `optimize_one` / `_project_condition`）、`full_rope_collision_audit.py`（含 `_d_optimal_greedy` —— **一个 D-最优贪心算法，可能可移植**）、`maxent_dilation_allocation.py` / `dilation_allocation_fork.py`（MaxEnt 分配的数值实现）。

**跳过（有意）**：`paper-2027/`、`tests/`（除 `test_scale_transport_math.py` 的存在性）、`internal/`、`rebuttal/`（除死 import 的验证）、`results/`、`data/`、`outputs/`、`analysis/unify_20260910/raw*`（2.9MB transcript 镜像，属 T 系列 agent 的领域）、`digests/`/`digests_codex/`（属 A 系列）。**未读 `~/.codex` 任何内容，未写入任何文件。**

**未验证的强声明**（须在使用前打回）：`NEXT_DERIVATION_KKT_PROBLEM.md:106` 声称 `J[h]` 的"严格凸、唯一正解、ρ″=τ²ρ 全链条在 codex 核心 §4.1 复核过"——我只在 `demand_companding_numeric.py` 的 docstring 里看到同一泛函族的**数值**实现，**没有在仓库代码里找到该凸性/唯一性证明**。该条按 [部分证据] 处理。
