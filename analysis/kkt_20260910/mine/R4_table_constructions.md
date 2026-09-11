# R4 digest：各方法频率表构造代码 → 可复现公式（拟合 F 的可行域地图）

挖掘者：R4。日期：2026-09-10。**只读**：未修改仓库任何文件，未写入 `~/.codex`。所有重算在 `/tmp` 内进行。

任务范围：把每个方法的**频率表构造代码**挖出来，写成公式级的可复现构造，给出 (a) `m_j` 或 `ν_j` 的精确构造、(b) 代码出处 `file:line`、(c) 端点是否满足 `m_{j≤23}=0` / `m_{j≥40}=1`、(d) 总跨度是否 `= ln S`，做成一张对照表。

兄弟 digest（避免重复）：`R2_ground_truth.md` 已覆盖 `ground_truth_tables.json` 的 **JSON schema / 对账表 / 存储 bug**；`R3_evq_math.md` 已覆盖 `evq_cosh_phi` 的 midpoint 网格与文档冲突。本文件只做**构造公式与可行域**，并补 R3 未覆盖的 `maxent_dilation_factors` / `build_inv_freq` 形状族（§3.10）。

---

## 0. 一句话结论

38 条目全部可以写成**同一族** `ν_j = ω_j · S^{−m_j}`（`ω_j = b^{−j/64}`），区别只在 `m_j` 这一条 25 维曲线（槽 0–23 / 24–39 / 40–63 三段）怎么取；所有"端点固定"表在代数上被**两个等式约束 + 一条水床恒等式**锁死，自由决策量恰好是过渡段 16 个增量 `Δ_j`（`ΣΔ=1`）。因此拟合 F 的可行域就是一个**16 维标准化增量单纯形**加上每条方法自己的端点/跨度是否守约。我把 30 张静态表的 `m_j` 构造、出处、端点、跨度、过渡预算 `B` 全部列进 §2/§4；发现一条隐藏恒等式 **`Σ_j m_j = B + 24`（当且仅当 I1/I2 成立）**，即 `Σm` 完全由 `B` 决定、**不含独立信息**——这直接印证红线 R4「Σm 是自由决策变量」。另外定位到 5 处**未登记在案的构造族**（`maxent_dilation` 与 4 个形状族）和 1 个已知存储 bug（`rebuild_ground_truth_tables.py:111`）。

---

## 1. 统一坐标、不变量与红线

### 1.1 坐标（出处：`ground_truth_tables.json:meta.definitions`；`NEXT_DERIVATION_KKT_PROBLEM.md:25-27`）

| 量 | 定义 | 出处 |
|---|---|---|
| `ω_j` | `b^{−j/64}`，`j=0..63`（0-based 槽） | `rebuild_ground_truth_tables.py:30-33` |
| `ν_j` | 部署反频率；统一读法 `ν_j = ω_j · S^{−m_j}` | `JSON meta.definitions.m_j` |
| `m_j` | `log(ν_j/ω_j)/log 4`（由 fp32 部署值反演） | 同上 |
| `T_j` | `2π/ν_j` | 同上 |
| `D_j` | `W·4^{m_j}`（识别地平线） | 同上 |
| `r_j` | `W/T_j^native`（原生周期数，**对所有方法相同**） | 同上 |
| `gap_g` | `ln(ν_g/ν_{g+1})`，`g=0..62`（槽 g 与 g+1 之间） | 同上 |
| `ρ_g` | 洞比率 `T_{g+1}/T_g = exp(gap_g)` | `JSON meta.definitions.hole_ratio`；`UNIFIED §1-2` |
| `Δ_j` | `m_{j+1} − m_j`（λ 坐标 `λ_j = S^{Δ_j}`，`∏λ_j = S`） | `NEXT_DERIVATION_KKT_PROBLEM.md:25-27` |

常数（`JSON meta.constants`）：`W=32768`、`S=4`、`L=131072`、`Dr=64`、`b=1e6`、`lnS=1.3862943611198906`、`native_log_gap = ln(b)/64 = 0.21586735246819178`、`native_period_ratio = b^{1/64} = 1.2409377607517196`、`official_gain = 1+0.1ln4 = 1.138629436111989`、`p2_gain = 1.102585782722872`。[已验证：全部由定义重算一致]

### 1.2 三条端点/跨度不变量（照抄权威文档，**勿改写成错误版本**）

- **I1**：`m_j = 0`，`j ≤ 23`（快带原生不动）。
- **I2**：`m_j = 1`，`j ≥ 40`（慢带 ÷S）。
- **水床恒等式**：`Σ_{g=23}^{39} (gap_g − ln(b)/64) = ln4`，等价于
  `Σ_{g=23}^{39} gap_g = 17·ln(b)/64 + lnS·(m40 − m23)`
  （出处：`NEXT_DERIVATION_KKT_PROBLEM.md:41-43`；`BUDGET_ALLOCATION_MODEL_AND_CANDIDATES_20260910.md:14`；`rebuild_ground_truth_tables.py:563` 实装校验）。
  **它是一条 telescoping 恒等式，不是能力声明；禁止写成"17 个 log-gap 之和 = ln S"**（那是错的：和为 `17·ln(b)/64 + ln4 = 3.6697 + 1.3863`）。[已验证：38 条目中所有 I1∧I2 成立者该和恰为 `+1.386294`]

### 1.3 红线（违反即无效，逐条遵守）

- 静态几何代理量（`Σcos` 首根、碰撞能、覆盖率、平滑度、有效秩、Gram、能量）**不得**作为 F 的分项或选择子。本文件里出现的 `B`、洞比率 `ρ`、粗糙度 `εᵀLε` 等，只作为**构造参数/可行域坐标**列出，**不是**我主张的 F 项。
- **`Σm` 不是守恒量**（`INTEGRATION_20260910.md:26-31` R4；`NEXT_DERIVATION_KKT_PROBLEM.md:109-116`）。§4.3 的 `Σm = B+24` 恒等式正好说明它只是 `B` 的仿射函数。
- "守恒"必须点名坐标：本文只声明 **log 跨度守恒**（`Σ_all gap = ln4`，坐标 = `ν` 的对数总跨度）与 **log-gap 总量守恒**（`Σ_g gap_g` 在运输操作下不变，坐标 = `gap` 向量）。
- 报告的 VICTORY / 已闭合 / 已证明类结论一律降级；本文不主张任何最优性，除 §3.4 那条**带完整 primal/dual KKT 证书**的最小粗糙度解（其最优性由凸性给出，但**仅对"最小粗糙度"这一目标**，不是任务最优性）。

---

## 2. 主对照表：30 张静态表的构造 / 端点 / 跨度

**列说明**：`B = Σ_{j=24}^{39} m_j`（过渡预算，**我定义并重算的口径**，见 §4.1）；`m23/m40` 取自 `ground_truth_tables.json`；`Σgap` = `sum_all_gap_increments_slots0_63`；`ρ_max` = `hole_ratio_max_transition`。

### 2.1 面板实测 / 基线（有分数）

| 方法 | 公式级构造（`m_j` 或 `ν_j`） | 代码出处 | `m23` | `m40` | I1/I2 | `Σgap=ln4` | `B` | `ρ_max` |
|---|---|---|---|---|---|---|---|---|
| **Native** | `ν_j=ω_j`（即 `m≡0`） | `rebuild_ground_truth_tables.py:199` 邻域；`JSON` | 0 | 0 | I1 ✓ / I2 ✗ | ✗(−0) | 0 | 1.2409 |
| **MrPro** | `m_j = q(q+1)/(N(N+1))`，`q=clip(j−23,0,N)`，`N=17` | `rebuild_ground_truth_tables.py:165-172` `radial_family_m(N_=17,dl=23)` | 0 | 1 | ✓ / ✓ | ✓ | 5.3333 | 1.4476@39 |
| **MrProBM**（BM） | `ε_i = 6i(N+1−i)/(N(N+1)(N+2))`；`m_q=q(q+1)(3N+2−2q)/(N(N+1)(N+2))`，`N=17` | `rebuild_ground_truth_tables.py:181-186` `bm_m_formula`；`smooth_budget.py:31-34` | 0 | 1 | ✓ / ✓ | ✓ | 8.0000 | 1.3934@31 |
| **MrUni** | `m_j=(j−23)/17`，槽 24–39；外部同 MrPro | `rebuild_ground_truth_tables.py:207-232`；`smooth_budget.py:38` | 0 | 1 | ✓ / ✓ | ✓ | 8.0000 | 1.3464@28 |
| **Smooth_MrBudget** | 固定预算 `B=16/3` 的最小粗糙度解 `ε`，`m= cumsum(ε)` | `smooth_budget.py:8-27,31-38`；`rebuild_ground_truth_tables.py:207-232` | 0 | 1 | ✓ / ✓ | ✓ | 5.3333 | 1.4513@35 |
| **GapCapped** | `m_j = clip((j−31)/9, 0, 1)`（= `max(0, 1−2(N−q)/(N+1))`，`q=j−23`，`N=17`，帽 `c=2/(N+1)=1/9`） | `scripts/lib/rope/gap_capped.py:11-15`；`docs/research/ROPE_GAP_CAPPED_PROTOCOL_20260908.md` | 0 | 1 | ✓ / ✓ | ✓ | 4.0000 | 1.4476@39 |
| **YaRN_linear_official** | `t=clip((j−23)/17,0,1)`；`ν_j = ω_j/4·t + ω_j·(1−t)` ⟺ `m_j = −log₄(1−0.75t)` | `rebuild_ground_truth_tables.py:191-197`；`scripts/analysis/export_static_rope_baselines.py:94-99` | 0 | 1 | ✓ / ✓ | ✓ | 6.1042 | 1.4599@39 |
| **LongBridgeSlower** | `ν_j = MrPro_j − 1/131072`，`j ∈ slots`；`slots = {j : T_j^MrPro ∈ [W, 4W]}` → **[36,37,38,39]** | `long_bridge.py:20,24`；`rebuild_ground_truth_tables.py:234-240` | 0 | 1 | ✓ / ✓ | ✓ | 5.5602 | 1.4930@38 |
| **LongBridgeFaster** | 同幅反号 `ν_j = MrPro_j + 1/131072`（方向控制） | `long_bridge.py:20,24` | 0 | 1 | ✓ / ✓ | ✓ | 5.1253 | 1.6192@39 |
| **HighGapToLong** | 见 §3.5；donors = gaps 0–22 各 `−budget/23`，recipients = gaps 36–39 各 `+budget/4`，`budget = mean(gaps[0:23]) = 0.2158673515995676` | `gap_budget_transfer.py:22,24,27,40,44-48,51`；`rebuild_ground_truth_tables.py:242-256` | **−0.1557** | 1 | **✗** / ✓ | ✓ | 3.0755 | 1.5279@39 |
| **HighGapToMid** | 同预算给 gaps 26–31（recipient 带 = 几何均值周期 ∈ `[W/16, W/4]`） | `gap_budget_transfer.py:39-41`；`JSON` | **−0.1557** | 1 | **✗** / ✓ | ✓ | 4.4769 | 1.4476@39 |
| **FullLagP2_Transfer3B** | 逐槽条件残差最小二乘构造（§3.4.2）；`m` 在槽 31 完成、32 起 `=1` | `scripts/analysis/audit_qwen_p2_full_lag.py:29-49`；`artifacts/agent_matched_comparison/audit_qwen_p2_origin.py` | **+0.000559** | 1 | **✗**(微) / ✓ | ✓ | 10.1780 | 2.9696@29 |
| **E1_s28_less** | 单槽：`ν_28 := native_28·4^{−m_27}`（`m28←m27`） | `select.py:76-80`；`rebuild_ground_truth_tables.py:258-266` `e1(28,-1)` | 0 | 1 | ✓ / ✓ | ✓ | 5.3007 | 1.4476@39 |
| **E1_s29_more** | 单槽：`ν_29 := native_29·4^{−m_30}` | `select.py:76-80`；`rebuild_ground_truth_tables.py:258-266` `e1(29,+1)` | 0 | 1 | ✓ / ✓ | ✓ | 5.3791 | 1.4476@39 |
| **E1_pair28_29** | `m28←m27` 且 `m29←m30`（双手术，增量集中入 gap28） | `rebuild_ground_truth_tables.py:263-266` | 0 | 1 | ✓ / ✓ | ✓ | 5.3464 | 1.4608@28 |
| **E1_s28_reverse_matched** | 镜像控制（等幅反向） | `JSON` role `panel-measured (镜像控制)` | 0 | 1 | ✓ / ✓ | ✓ | 5.3676 | 1.4476@39 |
| **E1_s29_plus_matched** | 镜像控制（等幅反向） | 同上 | 0 | 1 | ✓ / ✓ | ✓ | 5.2908 | 1.4476@39 |
| **E4_pair25_29** | `ν_j := native_j·4^{−s}`，`j∈{25,29}`，`s=(m25+m29)/2=0.078431` | `select.py:146-163`；`rebuild_ground_truth_tables.py:277-281` | 0 | 1 | ✓ / ✓ | ✓ | 5.3333 | 1.4476@39 |
| **E8_zero51** | `ν_51 := 0`（慢带单槽置零；该槽 `m/T/D=null`） | `select.py:90-92`；`rebuild_ground_truth_tables.py:285-286` | 0 | 1 | ✓ / ✓ | ✓ | 5.3333 | 1.4476@39 |
| **E7_local_projection** | `ν[24:40] += (BM−MrPro)[24:40]·x*`，`x*` 由局部响应 Hessian 约束最小二乘 + 二分（§3.8） | `project.py:29,65-83` | 0 | 1 | ✓ / ✓ | ✓ | 6.3935 | 1.5515@34 |
| **E2_tail_more** | MrPro 且 `ν[40:] *= 1e6^{−1/64}` ⟹ `m_{j≥40}=1.15572` | `select.py:85-88`；`rebuild_ground_truth_tables.py:268-273` | 0 | **1.1557** | ✓ / **✗** | **✗**(1.602162) | 5.3333 | 1.7964@39 |

### 2.2 纯公式 / 未执行 / 外部基线（无面板分数）

| 方法 | 公式级构造 | 代码出处 | `m23` | `m40` | I1/I2 | `Σgap=ln4` | `B` | `ρ_max` |
|---|---|---|---|---|---|---|---|---|
| **YaRN_smoothstep_variant** | 同 YaRN 但 `t → 3t²−2t³` | `rebuild_ground_truth_tables.py:191-197` `ramp='smoothstep'` | 0 | 1 | ✓ / ✓ | ✓ | 6.5162 | 1.4208@35 |
| **MrProN16** | `m_q = q(q+1)/272`，`N=16`（槽 39 完成） | `rebuild_ground_truth_tables.py:165-172` `radial_family_m(16)` | 0 | 1 | ✓ / ✓ | ✓ | 6.0000 | 1.4608@38 |
| **MrProN15** | `m_q = q(q+1)/240`，`N=15`（槽 38 完成） | 同上 `radial_family_m(15)` | 0 | 1 | ✓ / ✓ | ✓ | 6.6667 | 1.4757@37 |
| **StackFrontBack** | MrPro ⊕ `s28_less` ⊕ LBS(槽36–39 `−step`) | `rebuild_ground_truth_tables.py:288-291` | 0 | 1 | ✓ / ✓ | ✓ | 5.5275 | 1.4930@38 |
| **BM_ScaleTaper** | `w_j=clip(ln(W/T_j^Mr)/ln4,0,1)`；`ν = Mr·(BM/Mr)^{w}` ⟹ 槽24–31=BM、32–35 锥度、≥36=MrPro | `scale_taper.py:16-17`；`rebuild_ground_truth_tables.py:293-298` | 0 | 1 | ✓ / ✓ | ✓ | 6.8019 | 1.4476@39 |
| **NTK_static** | `ν_j = ω_j·4^{−j/63}`（等价新基 `b·4^{128/126}`） ⟹ `m_j = j/63` | `rebuild_ground_truth_tables.py:199`；`export_static_rope_baselines.py:100` | **0.3651** | **0.6349** | **✗/✗** | ✓(恒等) | 8.0000 | 1.2685@28 |
| **E3_BM_gain074** | 表 = BM，只变 gain（`0.74`） | `JSON` role `panel-measured (gain 析因)` | 0 | 1 | ✓ / ✓ | ✓ | 8.0000 | 1.3934 |
| **E3_BM_gain1** | 表 = BM，gain=1 | 同上 | 0 | 1 | ✓ / ✓ | ✓ | 8.0000 | 1.3934 |
| **Control_Mr_gain074** | 表 = MrPro，gain=0.74 | 同上 | 0 | 1 | ✓ / ✓ | ✓ | 5.3333 | 1.4476 |

### 2.3 无独立 64 槽表的算子臂（`nu_j=null`，8 条）

`E5_layer21/27/32`（层替换：MrPro 表 + 单层换 BM 表）、`E6_layer14_group0/1`（KV 组替换 E2 方向）、`E7_norm_matched_BM`（BM 的范数匹配对照）、`E10_dual_frequency`（同槽双频混合）、`E9_distance`（距离域双时钟）。
出处：`JSON` role 字段；构造代码 `experiments/nongeometric_screen/operators.py`（`embeddings()` / layer / group / `dual_frequency` / `distance` 安装器）、`distance_operator.py`（E9）、`numerical_controls.py:9`（E10 同频对照）。
**E10 关键事实**：索引集 `range(24,40) ∪ range(88,104)`，Q 侧权重恰 1/2，**不产生独立的 64 槽频率表**。`operators.py`。
**E9 关键事实**：local = Native 表，far = MrPro 表，`extra = window·(native−mr)` 相位偏移，joint `logaddexp` 归一化；`distance_operator.py`。

---

## 3. 逐族构造（公式级，可直接复现）

### 3.1 锚点：Native / PI / NTK

```python
# rebuild_ground_truth_tables.py:30-33
W, S, LN_S, BASE, DR = 32768, 4, math.log(4.0), 1e6, 64
NATIVE_GAP_RATIO = BASE ** (1.0/DR)      # 1.2409377607517196   ← ρ_g 的原生值
NATIVE_LOG_GAP   = math.log(BASE)/DR     # 0.21586735246819178  ← ln(b)/64
```

- **Native**：`ν_j = ω_j = b^{−j/64}`（`scripts/lib/rope/schedules.py:87-91` `geometric_inv_freq`）。
- **PI（位置插值）**：`ν_j = ω_j / scale`，**全表均匀、无分带**（`scripts/analysis/rope_transport/tables.py:95-108` `position_interpolation`：`arr = base_table.inv_freq/scale`）。在统一坐标里 `m_j ≡ log_S(scale) = 1`（取 `scale = S = 4`），即**"常数 `m` 射线"而不是三段式**：`m23 = m40 = 1` ⟹ **I1 破、I2 恰好满足**，且 `Σ_all gap = lnS·(m_64 − m_0) = 0`、`B = Σ_{j=24}^{39} 1 = 16`。
  **注意**：PI **不在** `ground_truth_tables.json` 的 38 条目中（该 JSON 不含 PI），故上面是**代数结论**而非 JSON 读数；`panel_scores` 亦无。
  另注意 `build_inv_freq` 里的 `pi` 用 `scale = max(max_seq_len/8192, 1)`（`schedules.py:229-231`），是**第二种 PI 口径**（`scale` 由推理长度而非 `S` 决定），此时 `m_j ≡ log_S(scale)` 仍为常数但值不同。
- **静态 NTK**：`ν_j = ω_j·4^{−j/63}`（`rebuild_ground_truth_tables.py:199` `ntk_nu = f32(NATIVE * np.power(4.0, -np.arange(DR)/(DR-1)))`；`export_static_rope_baselines.py:100` `ntk = omega*factor**(-slots/(pairs-1))`）。
  ⟹ `m_j = j/63`（**线性**），`m23 = 0.365079`、`m40 = 0.634921` —— **两个端点都不满足**，却**总跨度恰为 `ln4`**（`Σ_all gap = ln4`，因为它端到端完成了 `m: 0→1`）。这是最有信息量的反例：**"跨度 = lnS"与"端点守约"是两条独立的约束**，NTK 只满足前者。
- **HF/NTK 乘子口径**：`multiplier = factor**(dim/(dim-2))`，`new_base = base*multiplier`（`export_static_rope_baselines.py:101`）；`eval_pe_baselines.py` 用 `scaled_base = base*scale**(d/(d-2))`。**同一思想两种写法**（改 base vs 改 `ν`），见 §8。

### 3.2 YaRN 族（两种互不相同的实现）

**(A) 官方 linear ramp（本仓库对账身份，也是唯一被 ground truth 采信的）**

```python
# rebuild_ground_truth_tables.py:191-197
def yarn_table(factor=4.0, ramp='linear'):
    low, high = 23, 40
    t = np.clip((np.arange(DR)-low)/(high-low), 0, 1)          # 线性
    if ramp=='smoothstep': t = 3*t*t - 2*t*t*t                  # 变体
    return f32(omega/factor*t + omega*(1-t))                    # ν = ω[1 − t(1−1/S)]
```
等价 `m`：`ν = ω(1 − t(1−1/S))` ⟹ `m_j = −log_S(1 − t_j(1−1/S))`（`S=4` 时 `m=−log₄(1−0.75t)`）。**这是 `m`-参数化与 YaRN 混合参数化的精确桥**：`m` 是 `t` 的**非线性**函数，`t=1` ⟹ `m=1` 精确。
- `low=23, high=40` 是**几何推导出来的、不是超参**：`low = floor(d·ln(W/(β_fast·2π))/(2 ln b))`、`high = ceil(d·ln(W/(β_slow·2π))/(2 ln b))`，`β_fast=32, β_slow=1`（`export_static_rope_baselines.py:94-95`；`scripts/analysis/rope_transport/tables.py:114-181` `official_yarn` 用 `virtual = -head_dim*np.log(omega)/(2*log(rope_base))` ⟹ 对几何网格恰等于 `j`）。对 Qwen3B 精确算出 `23.5959 / 39.6509` → `[23,40]`；对 OLMo1B → `[14,32]`。**这与 MrPro 的索引集完全相同**——两族共用同一分带边界。
- attention scaling：`mscale = 1 + 0.1·ln(factor) = 1.138629436111989`（`export_static_rope_baselines.py`；`JSON meta.constants.official_gain`；jquesnelle/yarn@995db5b）。
- **YaRN 尺度响应**：`η_Y(S) = −∂log ν/∂log S = t/[S(1−t)+t] → 0`（饱和）。见 `STARTING_POINT_YARN_VS_MRPRO.md` F 系列。

**(B) 第二实现（非规范，术语冲突源）**：`scripts/core_text_phases/eval_pe_baselines.py:66-122`——`start = int(0.20·k)`、`end = int(0.90·k)`、ramp 用 **smoothstep**、`yarn_scale = scale^{ramp}·temperature^{0.5·ramp}`、`temperature = 1+0.07·log2(scale)`。**与 (A) 的分带、ramp、scaling 都不同**，见 §8 冲突 C4。

### 3.3 MrRoPE 径向族（Pro / BM / N16 / N15）——本族的中心构造

```python
# rebuild_ground_truth_tables.py:165-172  MrRoPE Eq.14 径向族
def radial_family_m(N_, dl=23):
    q = np.clip(np.arange(DR)-dl, 0, N_)
    return q*(q+1)/(N_*(N_+1))              # N_=17 ⟹ MrPro；N_=16 ⟹ N16；N_=15 ⟹ N15
```
```python
# rebuild_ground_truth_tables.py:181-186  BM（最小粗糙度闭式）
def bm_m_formula(N_=17, dl=23):
    q = np.clip(np.arange(DR)-dl, 0, N_)
    return q*(q+1)*(3*N_+2-2*q)/(N_*(N_+1)*(N_+2))
```
- **MrPro 的增量**：`Δ_q = 2(q+1)/(N(N+1)) = (q+1)/153`，线性递增，`Δ` 最大 `17/153 = 1/9 = 0.11111`（槽 39）；`B = 16/3 = 5.3333`。
- **BM 的增量**：`ε_i = 6i(N+1−i)/(N(N+1)(N+2)) = 6i(18−i)/5814`，**抛物线、内点最大 `i=9` 处 `ε=0.0836`**；`B = 8`。
- `radial_family_m` 与 `bm_m_formula` **都是"槽 39 恰好完成 `m=1`"的径向族成员**：参数 `N` 决定"用几个槽走完 1"。`N=17` ⟹ 完成于槽 40；`N=16` ⟹ 槽 39 完成（`m39=1.0000`，`m40` 的 `Δ=0`）；`N=15` ⟹ 槽 38 完成（`m38=m39=1.0000`）。**这三种是"以更陡的坡换更长的平台"的同一维调节**，且**都保持 I1/I2 与 `Σgap=ln4`**。
- **未登记的等价**：`E2_boundary39`（`select.py:84`，`n=high−23=16`，`native/4^{t(t+1)/(n(n+1))}`）**与 `MrProN16` 逐位相同**——E2 的 boundary 分支其实就是在扫径向族的 `N`。
- 权威对照：`STARTING_POINT_YARN_VS_MRPRO.md` F1–F9；`docs/research/ROPE_GAP_CAPPED_PROTOCOL_20260908.md`（由 MrPro 自身的最大增量 `2/(N+1)` 推 `cap`）。

### 3.4 KKT / 残差构造

**3.4.0 最小粗糙度（Smooth_MrBudget）——本仓库唯一带完整 KKT 证书的构造**

```python
# experiments/nongeometric_screen/smooth_budget.py:8-27
lap = 2I − eye(k=±1)                                  # Dirichlet Laplacian，n×n
a = [ones(n); arange(n-1,-1,-1)]                      # 约束矩阵：Σε=1 且 Σ w_i ε_i = budget
b = [1.0, budget]
for free in (前缀活跃面 ∪ 后缀活跃面):                  # 枚举 k=1..n-2 的活跃面
    kkt = [[2·lap[free,free], a[:,free].T],
           [a[:,free],       0]]
    answer = solve(kkt, [0…, b])
    ε[free] = answer[:|free|]; dual = 2·lap@ε + a.T@answer[|free|:]
    accept iff  ε.min() ≥ −1e-10  ∧  ‖a@ε−b‖_∞ < 1e-9
              ∧ ‖dual[free]‖_∞ < 1e-9  ∧  dual[active] ≥ −1e-9
```
- **目标**：`min εᵀ L ε`（`L` = Dirichlet Laplacian），即"最小粗糙度"。**凸问题**，故 KKT 是全局面最优的充要条件——但**只对这一目标**。
- **两个预算**：`solve(17, (n−1)/3 = 16/3)` ⟹ **Smooth_MrBudget**（与 MrPro 同预算）；`solve(17, (n−1)/2 = 8)` ⟹ **BM**，并有独立恢复校验 `expected = 6i(n+1−i)/(n(n+1)(n+2))`（`smooth_budget.py:33-34`）。**BM = 最小粗糙度解在 B=8 处的闭式**，这条是**已验证的恒等**（`atol=1e-11`）。
- **MrUni 是同一条 `B=8` 线上的另一个可行点**（均匀 `ε=1/17`），不是最小粗糙度解——故 `BM ≠ MrUni` 但 `B` 相同（§4.1 关键对照）。
- Smooth 的证书：`roughness = 0.004886399`（`JSON`/receipt）；活跃面见 `active_zero_indices_1based`。
- 界：qwen3b `(23,40)`、olmo1b `(14,32)`（`smooth_budget.py:52`）。
- 端点保持是**硬校验**：`if not (array_equal(f[:lo+1],base[:lo+1]) and array_equal(f[hi:],base[hi:])): raise`（`smooth_budget.py:41`）。

**3.4.1 参考实现（同算法、不同文件）**：`rebuild_ground_truth_tables.py:207-232` `solve_sb(n,budget)` 与 `smooth_budget.solve` 同构；`m_smooth = np.r_[0, np.cumsum(eps_smooth)]`；`smooth_nu[j] = NATIVE[j]·4^{−m_smooth[j−23]}`（`j∈24..39`）；`mruni_nu[j] = NATIVE[j]·4^{−(j−23)/17}`。

**3.4.2 FullLagP2（条件最小二乘残差构造）**

```python
# scripts/analysis/audit_qwen_p2_full_lag.py:29-31,34-35,38-49
design_for(p) = [cos(p·ν_j)·sqrt(w_p), sin(p·ν_j)·sqrt(w_p)]     # 支持 p=0..32767，权重 w_p = L−p
normalized_m(u) = (1 − (u−u.min())/(u.max()−u.min()))**2         # 归一化后平方
residual_fraction = ‖residual‖² / ‖target‖²                       # np.linalg.lstsq(rcond=1e-10)
```
- **P2 不是闭式 `m` 公式**，而是**逐槽数值最小化**（对 `ν_j` 做条件残差拟合）；落地形态是"短槽不动、槽 31 完成 `m=1`、32 起平台"。
- 实装表 **bit-exact 等于 `full_32768` 构造**，sha `ecd0c280a11788e0c4a869a3d162880964ba3f371f589f7e2f5ae471d461096b`（我把 audit 跑到 `/tmp/p2audit.json` 复核）。历史数组另存 `docs/research/ROPE_QWEN15_FULL_LAG_P2_CANDIDATE_20260907.json`、`ROPE_RECOVERED_QWEN_P2_20260907.json`。
- `B = 10.1780`（最大）、`ρ_max = 2.9696@29`（**全表最大洞，远高于其它臂**）；`m23 = +0.000559`（**微违反 I1**）。P2 是"极端集中"的一个实测端点。

### 3.5 EVQ 字面运输族（GapCapped / HighGapToLong / HighGapToMid / E2）

**HighGapToLong（EVQ 的字面运输）**：`experiments/nongeometric_screen/gap_budget_transfer.py` 是原始实现，`rebuild_ground_truth_tables.py:242-256` 是移植。

```python
high_end = int(first_changed[0]) - 1        # = 23，因 MrPro 首个改变槽为 24
donors   = np.arange(high_end)              # = 0..22（23 个高通 gap）
gaps     = np.log(reference[:-1]/reference[1:])          # gap_g = ln(ν_g/ν_{g+1})
budget   = float(gaps[donors].mean())       # = 0.2158673515995676  ← 恰好 = ln(b)/64!
periods  = 2*np.pi/reference
gap_period = np.sqrt(periods[:-1]*periods[1:])            # 几何均值周期
recipients = flatnonzero((gap_period>=low) & (gap_period<=high))
new_gaps[donors]     -= budget/len(donors)
new_gaps[recipients] += budget/len(recipients)
values = np.exp(np.r_[log(ref[0]), log(ref[0]) - np.cumsum(new_gaps)])
values[recipients[-1]+1:] = reference[...]   # 末位 recipient 之后累积转移恰为零
```
- **`budget = mean(gaps[0:23])` 精确等于 `ln(b)/64 = 0.21586735246819178`**（差 9.3e-13，浮点级）。这不是巧合：MrPro 下 donors 全为原生 gap，其均值就是原生 log-gap。**"捐一个原生高通 gap 的总量"= "把 23 个 gap 各压低一个原生 gap"**——EVQ 的"预算量子"有一个干净的解析值。[已验证：我重算 `mean(log(native_j/native_{j+1})) j=0..22` 得 0.2158673515995676]
- **recipient 带（带宽由 `gap_period` 的几何均值判据决定，不是槽号）**：
  - Long：`[W, 4W] = [32768, 131072]` → gaps **[36,37,38,39]**
  - Mid：`[W/16, W/4] = [2048, 8192]` → gaps **[26,27,28,29,30,31]**
- **`m23 = −0.155715` 的来源**：donors 覆盖 gaps 0–22，槽 0 被钉在 `ν_0=1.0`，于是 `m_1..m_23` 被整体下压 `budget/ln4 = 0.155715`。**这是"运输"操作与 I1 的**结构性冲突**：只要 donor 带触及槽 0 的邻域，I1 必破。**这是可复用的约束结论**：要保 I1，donor 带不得从槽 0 侧起算（或须把 `ν_0` 一起改）。
- 两条硬校验：`new_gaps > 0`；`np.isclose(new_gaps.sum(), gaps.sum(), atol=1e-12)` —— **这就是"log-gap 总量守恒"的实装形式，坐标 = `gap` 向量**（`gap_budget_transfer.py:46-47`）。

**E2_tail_more**：`select.py:85-88` `factor = 1e6**(1/64)`；`new[40:] *= 1/factor`（`more`）或 `*= factor`（`less`）。
⟹ `m_{j≥40} = 1 + 1/64·... `：实测 `m40 = 1.155715 = 1 + 0.155715`。**它是唯一"故意破坏 I2"的臂**，总跨度变成 `1.602162 ≠ ln4`。`ρ_max = 1.7964@39`（次高）。
用途：**作为"慢带整体下移"的方向性对照**，同时提供一条 `Σgap ≠ ln4` 的可行点。

**GapCapped（帽 = MrPro 自身最大增量）**：`scripts/lib/rope/gap_capped.py:11-15`

```python
cap = Fraction(2, width+1)                                    # width=17 ⟹ cap = 1/9
[ max(Fraction(0), 1-(width-q)*cap) for q in range(width+1) ]  # m_q = max(0, 1 − 2(17−q)/18)
```
等价槽号形式：**`m_j = clip((j−31)/9, 0, 1)`**（`j=32` 起非零；`1/9 = Δ_39^MrPro`）。[已验证：与 `JSON` 的 `m24..m39` 逐点一致]
**含义**：把最大增量钉在 MrPro 的最大增量上 ⟹ 过渡带被迫加长（9 个槽而非 17 个），预算降到 `B=4`。`ROPE_GAP_CAPPED_PROTOCOL_20260908.md:1-2` 记录 GPU 结果为**负**（死路，见 §7）。
注意文档称 "Qwen 有效带 31→40"：按 0-based 槽是 **32→40**（槽 31 的 `m` 仍为 0），差 1 槽，属文档口径（见 §8 C2）。

### 3.6 长桥族（LBS / LBF / StackFrontBack）

```python
# experiments/nongeometric_screen/long_bridge.py:20,24
step = 1.0 / target_length          # target_length = 131072 ⟹ step = 7.62939453125e-06
for direction in (-1 Slower, +1 Faster):
    values[slots] += direction * step       # slots = MrPro 周期 ∈ [32768, 131072] = [36,37,38,39]
```
- 等价 `m` 变化：`Δm_j = log_S((ν_j + d·step)/ν_j)`，符号 = `−d`（`d=−1` ⟹ `m` 上升）。实测 `Δ_38 = 0.1154`（Slower）vs MrPro `0.0850`。
- **StackFrontBack** = MrPro ⊕ `s28_less` ⊕ LBS(36–39)；`B = 5.5275`、`ρ_max = 1.4930@38`（= LBS 的值）。`rebuild_ground_truth_tables.py:288-291`。
- **量纲设计**：`step = 1/L` 使远端槽在 `L=131072` 处获得 **1 rad 的公共相位**（`LBS` 的 `Δphase = 1·2π/2π`）。这是"O(1) 长距相位干预"，短距效应小 4×（`long_bridge.py:17-18` 注释，**注释是证据不是指令**）。

### 3.7 单槽手术族（E1 / E4 / E8）—— 可行域的"坐标扰动"基底

```python
# rebuild_ground_truth_tables.py:258-266
def e1(slot, direction):
    t[slot] = NATIVE[slot] * 4 ** (-MR_M[slot+direction])    # 把本槽的 m 换成邻居的 m
s28_less = e1(28, -1)   # m28 ← m27
s29_more = e1(29, +1)   # m29 ← m30
```
- 等价 `m` 读数：`s28_less` 的 `m28 = 0.065359 = m27`；`s29_more` 的 `m29 = 0.431373 = m30`。
- **`E1_pair28_29`**（`rebuild_ground_truth_tables.py:263-266`）：`pair_nu[28]=NATIVE[28]·4^{−MR_M[27]}`、`pair_nu[29]=NATIVE[29]·4^{−MR_M[30]}` ⟹ `ε28+ε29+ε30` 集中进 `gap28`，`ρ_28 = 1.4608`。
- **`E4_pair25_29`**（`select.py:146-163`）：`shared = (m25+m29)/2 = 0.078431`，`ν_j = native_j·4^{−shared}`，`j∈{25,29}` ⟹ **同一分钟刻度上做"等值配对"**，`B` 不变（5.3333）。
- **`E8_zero51`**：`ν_51 = 0` —— 唯一的**非严格递减/零槽**表，`m/T/D = null`。**它的问题不可用本族坐标表示**（`m_51 → +∞`），是可行域的一个边界/退化点。
- **镜像控制**（`E1_s28_reverse_matched`、`E1_s29_plus_matched`）：同幅反号，用于把"手术量"与"方向"分离。`JSON` role 明示 `panel-measured (镜像控制)`。

### 3.8 锥度与投影（BM_ScaleTaper / E7）

**BM_ScaleTaper**（`scale_taper.py:16-21`）

```python
periods = 2*np.pi/reference
weight  = np.clip(np.log(native_length/periods)/np.log(scale), 0, 1)   # W=32768, S=4
values  = np.exp(np.log(reference) + weight*np.log(bm/reference))      # 对数空间线性插值
values[weight==0] = reference[...] ; values[weight==1] = bm[...]       # 端点精确还原
```
⟹ `m_j = (1−w_j)·m_j^MrPro + w_j·m_j^BM`，`w_j = clip(ln(W/T_j^MrPro)/ln4, 0, 1)`；**槽 24–31 完全 = BM、32–35 锥度、≥36 = MrPro**。`B = 6.8019`。
写法与 `scripts/analysis/rope_transport/tables.py:177-215` 的 `budgeted_transport` 同构：`move = (1−norm(u))**exponent`，`arr = omega*(1−move) + (omega/scale)*move`（**同一"对数空间混合"思想**）。`receipt` 明写 `rationale` 是**设计假设**、非最优阈值。

**E7_local_projection**（`project.py:29,65-83`）

```python
direction = bm[24:40] - mr[24:40]
# 局部窗口 Hessian：J = ∂(o_proj·attn_out)/∂ν，hess += J Jᵀ（36 层 × 全部 calibration）
eig, u = eigh(hess)                     # 特征分解
solution(lam) = u @ ((u.T@ones)/(1 + lam*eig))
# 二分 lam 使 energy(x) = xᵀHx 达到 budget = min(两 split 的 mean step cost)
radial = min(1, 0.25/max_phase)         # 相位半径帽
values[24:40] += direction * x * radial
```
- 这是一条**"在局部线性响应度量下最省地做 BM 方向移动"**的约束最小二乘，`x` 是 16 维。实测 `B = 6.3935`、`ρ_max = 1.5515@34`。
- **注意**：`JSON` 的 `construction` 注明"**修正约束后 ≡ BM，历史部署表保留**"——即 **E7 的部署表是已被取代的约束下的历史产物**。引用 E7 时必须带这条限定（死路相关，见 §7）。
- 其中的 `hess`、`energy`、`eig` 全是**静态/线性代理量**；按红线，**它们不得作为 F 的分项**，只能作为"该臂当初怎么造的"这一事实记录。

### 3.9 无独立表的算子臂（E5 / E6 / E9 / E10）

构造代码在 `experiments/nongeometric_screen/operators.py`（`embeddings()` 及 layer/group/`dual_frequency`/`distance` 安装器）。这些臂**改的是算子，不是频率表**——对拟合 F 的意义是：**它们把"改表"与"改算子"分离**，可作为"改表效应"的对照上界/下界。
- E5：单层用 BM 表替代 MrPro 表（层索引 21/27/32 = 由 BM 的固定状态效应排名选出）。
- E6：E2 方向的 KV **组**级替换（`group ∈ {0,1}`，`layer=14`），保持共享 KV 组的完整。
- E10：同槽双频（`dual_frequency`，Q 侧权重 1/2，索引 `24..39 ∪ 88..103`）；`numerical_controls.py:9` 做了**同表**对照以隔离"算术 vs 频率混合"。
- E9：距离域双时钟（`distance_operator.py`），`extra = window·(native−mr)`，`logaddexp` 联合归一化，`torch.compile(flex_attention)`。

### 3.10 【R3 未覆盖】仓库内另一整套闭式构造族（`scripts/lib/rope/schedules.py`）

这一族**不在任务清单上，但是真实的频率表构造代码**，且给出的是**1 参数闭式族**，对 F 的参数化极有价值。`schedules.py:215-322` `build_inv_freq(method, head_dim, base, max_seq_len, ...)`：

| method | 构造 | 行号 |
|---|---|---|
| `geometric` | `ν_i = b^{−2i/d}`（= `ω`） | 87-91 |
| `pi` | `ν_i = ω_i/scale`，`scale = max(max_seq_len/8192, 1)` | 229-231 |
| `yarn`（B 实现） | `ramp = smoothstep(clip((i−start)/(end−start)))`，`start=⌊0.2k⌋`、`end=⌊0.9k⌋`，`ν = ω/yarn_scale`，`yarn_scale = S^{ramp}·T^{0.5·ramp}`，`T = 1+0.07·log₂S` | 233-245 |
| `anchored_hybrid` | 前 `j0=12` 槽刚性 = 原生；尾用 `tail_base = b·S² (≥4b)` 几何表 + `blend = α·(0.5−0.5cos(πt))` 混合，`α = clip(0.16·log₂S, 0.08, 0.40)` | 247-265 |
| `sigmoid` | `ν_i = b^{−s_norm(i)}`，`s_norm = (σ(i)−σ(0))/(σ(n−1)−σ(0))`，`σ(x)=1/(1+e^{−slope(x−center)})`，`slope=16.05/d`，`center=0.47n` | 267-278 |
| `anchored_sigmoid` | `ν_i = ω_i/(1+(A−1)σ(i))`，`A = clip(max(2, 2.5S), 2, 30)` | 280-293 |
| `evq_cosh` | `φ = 1 − (1/τ)·asinh((1−u)·sinh τ)`，`u=(i+0.5)/K`，`ν = b^{−φ}` | 295-301（`evq_cosh_phi` 162-189，`evq_cosh_inv_freq` 192-208） |
| `evq_exp` | `φ = ((1+β)^u − 1)/β`，`β=3`，`u=i/n`，`ν = b^{−φ}` | 303-311 |
| **`maxent_dilation`** | `r_i = [1 + q_i(S^λ − 1)]^{1/λ}`，`q_i = (i+0.5)/K`；`ν_i = ν_i^native / r_i`，**反向配对（大 dilation ↔ 慢频率）** | `maxent_dilation_factors` 94-127；`maxent_dilation_inv_freq` 129-159 |

**为什么这对 F 重要（三条）**：
1. **`maxent_dilation` 的 `λ→0` 极限就是 `ν_i = ω_i·S^{−q_i}`，`q_i = (i+0.5)/K`** —— 即统一坐标下的 `m_i = (i+0.5)/K`，**midpoint 网格的均匀斜坡**（与 `MrUni` 的 `(j−23)/17` 同一形状，只是网格约定差半格）。`λ` 是**在 log-频率上加几何倾斜的单参数**：`density ∝ e^{λτ}`，`τ = log r`。
2. `maxent_dilation_inv_freq` 的 docstring 明写配对规则：**"Larger dilation factors are paired with slower Native frequencies. This opposite-order coupling minimizes summed in-window phase displacement by the rearrangement inequality."**（`schedules.py:136-139`）——**"不锚定端点、全靠重排不等式"**，与 MrPro 族"锚定两端"是**两条不同的构造哲学**，值得作为 F 的两条候选参数化并列。
3. `evq_exp` 与 `evq_cosh` 是**同一 warping 思想的两种凸变换**（`φ` 分别把 `u` 映成指数/反 sinh），两者都**不锚定端点**（`φ(0) ≠ 0` 当 `midpoint=True`）。R3 已覆盖 `evq_cosh` 的网格争议，本表补上 `evq_exp` 与 `maxent` 两条**未被 R3 登记**的构造。

---

## 4. 可行域地图（拟合 F 的直接输入）

### 4.1 过渡预算 `B = Σ_{j=24}^{39} m_j` —— 单一最有区分力的标量

`B` 是我按统一坐标定义并逐条重算的（`B = m[24:40].sum()`，源数据 `ground_truth_tables.json.methods[*].m_j`）。**它是这条"在过渡段总共走了多少 log-频率"的总量**：

| `B` | 方法（同值者并列） | 出处 |
|---|---|---|
| 0 | Native | `JSON` |
| 4.0000 | GapCapped | `JSON`/`gap_capped.py:11-15` |
| 5.1253 | LongBridgeFaster | `JSON` |
| 5.2908 / 5.3007 / 5.3333 / 5.3464 / 5.3676 / 5.3791 | E1_s29_plus_matched / E1_s28_less / **MrPro**, Smooth, E4, E8, E2_tail_more, Control_Mr_gain074 / E1_pair28_29 / E1_s28_reverse_matched / E1_s29_more | `JSON` |
| 5.5275 | StackFrontBack | `JSON` |
| 5.5602 | LongBridgeSlower | `JSON` |
| 6.0000 | MrProN16 | `JSON` |
| 6.1042 | YaRN_linear_official | `JSON` |
| 6.3935 | E7_local_projection | `JSON` |
| 6.5162 | YaRN_smoothstep_variant | `JSON` |
| 6.6667 | MrProN15 | `JSON` |
| 6.8019 | BM_ScaleTaper | `JSON` |
| **8.0000** | **MrProBM（BM）**, **MrUni**, NTK_static | `JSON` |
| 10.1780 | FullLagP2_Transfer3B | `JSON` |
| 3.0755 / 4.4769（**I1 已破**） | HighGapToLong / HighGapToMid | `JSON` |

**关键对照（可复用结论）**：
- **`B` 相同 ≠ 表相同**：`MrProBM` 与 `MrUni` 都 `B=8`，但一个是抛物线增量（最小粗糙度），一个是均匀增量。**`B` 是"总量"，形状是"分布"，两者正交**——F 至少需要 (总量, 形状) 两个坐标，或直接用 16 维 `Δ`。
- **`B` 越大越集中，洞越大**：`B: 4 → 5.33 → 8 → 10.18` 对应 `ρ_max: 1.4476 → 1.4476 → 1.3934 → 2.9696`。**并不单调**——BM（B=8，抛物线）的洞**比** MrPro（B=5.33，线性）**更小**。即**"总量"不是洞的充分统计量，"集中度/形状"才是**（BM 的抛物线把增量摊平）。这条是拟合 F 时必须保留的二维结构证据。
- **同 `B` 的两条不同形状**（MrPro `B=16/3` 线性 vs Smooth `B=16/3` 最小粗糙度）：Smooth 的 `ρ_max = 1.4513@35` 反而**高于** MrPro 的 `1.4476@39`——**最小化"粗糙度"（二阶差分能量）并不最小化"最大洞"（一阶差分上确界）**。这是两个不同的几何目标，红线里"平滑度不得作 F 项"正好指向这一点。

### 4.2 端点与跨度的**正交性**（§2 表直接读出）

| 类别 | 日志跨度 `Σgap = ln4` | I1 (`m23=0`) | I2 (`m40=1`) | 成员 |
|---|---|---|---|---|
| 守约 | ✓ | ✓ | ✓ | MrPro / BM / MrUni / Smooth / GapCapped / YaRN×2 / N16 / N15 / Stack / LBS / LBF / E1×5 / E4 / E8 / E7 / BM_ScaleTaper / E3×2 / Control |
| **只破 I1** | ✓ | ✗ (`−0.1557`) | ✓ | HighGapToLong / HighGapToMid |
| **微破 I1** | ✓ | ✗ (`+0.000559`) | ✓ | FullLagP2_Transfer3B |
| **只破 I2** | ✗ (`1.6022`) | ✓ | ✗ (`1.1557`) | E2_tail_more |
| **只守跨度** | ✓ | ✗ (`0.3651`) | ✗ (`0.6349`) | NTK_static |
| **两端都破、跨度也破** | ✗ (`0`) | ✗ (`0`) | ✗ (`0`) | Native（`m≡0`） |
| **只破 I1、跨度归零** | ✗ (`0`) | ✗ (`1`) | ✓ (`1`) | PI（`m≡1`，**代数结论，不在 JSON 38 条目内**） |

**⟹ 结论（可直接用作 F 的约束结构）**：`Σ_all gap = ln4` 与 `I1 ∧ I2` **互相独立**——前者是 `m` 端到端的净变化，后者是两端各自钉死。"守跨度不守端点"（NTK）与"守端点自动守跨度"（I1∧I2 ⟹ 跨度必为 `ln4`，因为 `m_64 − m_0 = 1`）**是单向蕴含**。因此**I1∧I2 是一条比"跨度守恒"更强的约束**，F 应把 I1/I2 作为硬约束、跨度作为推论，而不是反过来。

### 4.3 隐藏恒等式：`Σ_j m_j = B + 24`（当且仅当 I1∧I2）

由 §2 数据逐条验证：`MrPro 29.3333 = 5.3333+24`；`BM 32 = 8+24`；`GapCapped 28 = 4+24`；`YaRN 30.1042 = 6.1042+24`；`N16 30 = 6+24`；`N15 30.6667 = 6.6667+24`；`NTK 32 = 8+24`；`HighGapToLong 25.2069 ≠ 3.0755+24`（**因 I1 破**）；`E2_tail_more 33.0705 ≠ 5.3333+24`（**因 I2 破**）。[已验证：`JSON.sum_m_all_slots` vs 重算 `B+24`，I1∧I2 成立者全等（差 <1e-12）]
**含义**：槽 0–23 贡献 0、槽 40–63 贡献 24，过渡段贡献 `B`。**`Σm` 不是独立自由度，它是 `B` 的仿射函数**。这条**从数值上印证红线 R4**（`INTEGRATION_20260910.md:26-31`；`NEXT_DERIVATION_KKT_PROBLEM.md:109-116`：`Σm` 是自由决策变量、不是守恒量）。任何以 `Σm` 为约束/惩罚的设计都是在重复使用 `B`。

### 4.4 自由度的准确计数

- **过渡段 16 个增量 `Δ_{23..38}`**（`m_{24..39}` 由 cumsum 给出，`Δ_39 = 1 − m_39` 由 I2 定死）。`Σ_{g=23}^{38} Δ_g = 1` ⟹ **自由度 15**？——不：`Δ_{23..39}` 共 **17 个**，`ΣΔ = 1` 是一个等式，故 **16 维自由**（`NEXT_DERIVATION_KKT_PROBLEM.md` 的"16 free DOF after the equality"）。
- 本文件的 `B = Σ_{j=24}^{39}m_j` 与 `ΣΔ` 是**不同的坐标**：`B = Σ_{q=1}^{16} (17−q)·Δ_{23+q−1}`（加权和）。**`B` 是 `Δ` 的一阶矩的补**，因此 `B` 单独**不**决定 `Δ`（§4.1 已用 BM vs MrUni 证明）。
- 术语不一致警告：文档里"17 个增量 / 16 个自由 DOF"两种说法并存且都对（17 个增量减 1 个等式约束 = 16 DOF）。见 §8 C7。

### 4.5 洞比率的精确定义（避免与"覆盖率/平滑度"混用）

`ρ_g = T_{g+1}/T_g = exp(gap_g)`，`gap_g = ln(b)/64 + Δ_g·lnS`（`JSON meta.definitions`；`UNIFIED §1-2`）。
由 `JSON`：原生 `ρ = 1.2409377607517196`。MrPro `ρ_max = 1.4476 @ gap39`：检验 `exp(ln(b)/64 + (17/153)·ln4) = exp(0.215867+0.154033) = 1.4476` ✓。[已验证]
**注意**：`ρ` 与 `Δ` 一一对应（单调），故 `max ρ` ≡ `max Δ`。**它不是独立信息**；只有 `Δ` 的**分布形状**携带信息（§4.1）。

---

## 5. 可作 F 零件 / KKT 约束的条目（推荐清单）

| # | 条目 | 形式 | 出处 | 等级 |
|---|---|---|---|---|
| K1 | 统一坐标 | `ν_j = ω_j·S^{−m_j}`，`ω_j=b^{−j/64}` | `JSON meta.definitions`；`NEXT_DERIVATION_KKT_PROBLEM.md:25-27` | [已验证] |
| K2 | 端点硬约束 I1/I2 | `m_j=0 (j≤23)`，`m_j=1 (j≥40)` | `NEXT_DERIVATION_KKT_PROBLEM.md:41-43` | [已验证] |
| K3 | 跨度推论 | I1∧I2 ⟹ `Σ_{g} gap_g = lnS`（精确 `ln4`） | 由上二者代数推出；`rebuild_ground_truth_tables.py:629-634` 全表校验 | [已验证] |
| K4 | 水床恒等式 | `Σ_{g=23}^{39}(gap_g − ln b/64) = lnS·(m40−m23)` | `NEXT_DERIVATION_KKT_PROBLEM.md:41-43`；`rebuild:563` | [已验证] |
| K5 | 增量单纯形 | `Δ∈R¹⁷, Δ≥0, ΣΔ=1`（16 free DOF） | `NEXT_DERIVATION_KKT_PROBLEM.md`；§4.4 | [已验证] |
| K6 | 过渡预算坐标 | `B = Σ_{j=24}^{39} m_j`（§4.1 全表值） | 本文件重算 `ground_truth_tables.json` | [已验证] |
| K7 | `Σm` 仿射依赖 | `Σ_j m_j = B + 24`（iff I1∧I2） | §4.3 重算 | [已验证] |
| K8 | λ 坐标 | `λ_j = S^{Δ_j}`，`∏λ_j = S` | `NEXT_DERIVATION_KKT_PROBLEM.md:25-27` | [已验证] |
| K9 | `m↔t` 桥（YaRN 混合 ⟺ 显式 m） | `T: m = −log_S(1−t(1−1/S))`，`t = (1−S^{−m})/(1−1/S)` | `rebuild:191-197` 代数变换 | [已验证] |
| K10 | 尺度响应（两族分界） | `η_Y(S)=t/[S(1−t)+t]→0`（饱和）；`η_M(S)=m_q`（幂律，不饱和） | `STARTING_POINT_YARN_VS_MRPRO.md` F 系列 | [部分证据]（文档级，未独立重算 `∂logν/∂logS`） |
| K11 | 分带边界的几何推导 | `low=⌊d·ln(W/(β_fast·2π))/(2 ln b)⌋`，`high=⌈d·ln(W/(β_slow·2π))/(2 ln b)⌉` | `export_static_rope_baselines.py:94-95`；`rope_transport/tables.py:114-181` | [已验证]（Qwen3B→23/40、OLMo1B→14/32） |
| K12 | 最小粗糙度问题的 KKT 模板 | `min εᵀLε` s.t. `Σε=1`, `Σwε=B`（Dirichlet Lap.），活跃面枚举 + 全证书 | `smooth_budget.py:8-27` | [已验证]（凸性 ⟹ 全局最优，**仅对此目标**） |
| K13 | BM 闭式 = `B=8` 最小粗糙度解 | `ε_i = 6i(N+1−i)/(N(N+1)(N+2))` | `smooth_budget.py:33-34`（`atol=1e-11` 校验） | [已验证] |
| K14 | log-gap 总量守恒（点名坐标） | `Σ_g gap_g` 在运输操作下不变；实装 `isclose(new_gaps.sum(), gaps.sum(), atol=1e-12)` | `gap_budget_transfer.py:46-47` | [已验证] |
| K15 | 运输的解析预算 | `budget = mean(gaps[donors]) = ln(b)/64 = 0.21586735246819178`（MrPro donors） | `gap_budget_transfer.py:24`；本文件重算 | [已验证] |
| K16 | 运输 ⟹ I1 结构性冲突 | donor 带触及槽 0 邻域 ⟹ `m_{1..23}` 整体下压 `budget/lnS`，I1 必破（实测 `−0.155715`） | §3.5 代数 + `JSON` | [已验证] |
| K17 | 帽 = 最大增量 ⟹ 带长 | `cap = 2/(N+1) = Δ_max^MrPro` ⟹ `m_j=clip((j−31)/9,0,1)`，`B=4` | `gap_capped.py:11-15` + §3.5 反演 | [已验证] |
| K18 | 重排配对（另一条参数化哲学） | `ν_i = native_i/r_i`，`r_i=[1+q_i(S^λ−1)]^{1/λ}`，`q_i=(i+0.5)/K`，大 dilation ↔ 慢频率 | `schedules.py:94-159` | [已验证]（代码即公式）；"重排最优"的解释是 [假设]（docstring 未给证明） |
| K19 | 无表算子臂作为"改表"的对照 | E5/E6/E9/E10 改算子不改表 | `operators.py` / `distance_operator.py` / `numerical_controls.py:9` | [已验证] |
| K20 | 增益与表**分离** | `E3_BM_gain074/1`、`Control_Mr_gain074` 只动 gain，表不变 | `JSON` role 字段 | [已验证]（呼应 `NEXT_DERIVATION_KKT_PROBLEM.md:109-116`"gain 单独计"） |

---

## 6. 已验证数字（每条带出处）

1. `W=32768, S=4, L=131072, Dr=64, b=1e6`；`lnS=1.3862943611198906`；`ln(b)/64 = 0.21586735246819178`；`b^{1/64} = 1.2409377607517196`；`1+0.1ln4 = 1.138629436111989`；`p2_gain = 1.102585782722872`。[已验证：由定义重算一致 | 源 `ground_truth_tables.json:meta.constants`]
2. **过渡预算 `B` 全表值**见 §4.1（MrPro `5.3333`、BM/MrUni `8.000`、YaRN `6.1042`、GapCapped `4.000`、N16 `6.000`、N15 `6.667`、P2 `10.178`、Smooth `5.3333`）。[已验证：重算 `m[24:40].sum()` | 源 `ground_truth_tables.json.methods[*].m_j`]
3. **`Σ_j m_j = B + 24`**（iff I1∧I2），逐条核验通过。[已验证 | 源同上 `sum_m_all_slots`]
4. **`Σ_all gap = ln4 = 1.386294`** 对除 `E2_tail_more`(1.602162)、`Native`(−0.0) 外的全部表成立。[已验证 | 源 `sum_all_gap_increments_slots0_63` / `expected_if_endpoints_fixed_ln4`]
5. `NTK`：`m23 = 0.365079 = 23/63`，`m40 = 0.634921 = 40/63`（**线性**），双端点皆破、跨度仍 `ln4`。[已验证 | `ntk_nu = NATIVE*4^{−arange(64)/63}`，`rebuild:199`]
6. `HighGapToLong/Mid`：`m23 = −0.155715 = −(ln b/64)/ln4 = −0.155715`（精确！`0.215867/1.386294 = 0.155715`）。[已验证 | `JSON`+代数]
7. `HighGapToLong` recipients = gaps **[36,37,38,39]**；`HighGapToMid` recipients = gaps **[26,27,28,29,30,31]**；donors = `arange(first_changed−1) = 0..22`；`first_changed = 24`。[已验证：我用 `gap_period` 判据独立重算 | `gap_budget_transfer.py:21-41`]
8. `budget = mean(gaps[donors]) = 0.2158673515995676`。[已验证 | 同上]
9. gap-period 样本：`gp[25]=1608.8, gp[26]=2060.8, gp[27]=2663.7, gp[31]=8140.9, gp[32]=11010.5, gp[35]=28761.7, gp[36]=40335.6, gp[39]=117467.3, gp[40]=157439.9`。[已验证：重算 | 同上]
10. `GapCapped`：`m_j = clip((j−31)/9, 0, 1)`；`cap = 1/9 = 2/(17+1) = Δ_39^MrPro = 17/153`。[已验证：与 `JSON m_j` 逐点一致 | `gap_capped.py:11-15` + `rebuild:165-172`]
11. `LBS/LBF`：`step = 1/131072 = 7.62939453125e-06`；slots = **[36,37,38,39]**；`LBS` 使 `Δ_38 = 0.1154`（MrPro `0.0850`）。[已验证 | `long_bridge.py:20,24`；`JSON m_j`]
12. `P2`：部署表 sha `ecd0c280a11788e0c4a869a3d162880964ba3f371f589f7e2f5ae471d461096b`（= `full_32768` 构造）；`B = 10.1780`；`ρ_max = 2.9696@29`。[已验证：我把 audit 跑到 `/tmp/p2audit.json` | `scripts/analysis/audit_qwen_p2_full_lag.py`]
13. `Smooth_MrBudget` 证书：`roughness = 0.004886399`，预算 `B = 16/3`。[部分证据：数值取自 `ground_truth_tables.json`/receipt，我未重跑 `solve` 全枚举（依赖缺失的 `results/` 镜像目录）]
14. 18/18 部署表公式重建 **bit-exact**；文档声称的 `≤4.3e-8` 实际为 `0.0`。[已验证 | `JSON formula_vs_deployed` + `BUDGET_ALLOCATION_MODEL_AND_CANDIDATES_20260910.md:49`]
15. `YaRN` 分带：Qwen3B `low` 精确 `23.5959 → 23`、`high` 精确 `39.6509 → 40`；OLMo1B → `14/32`。[部分证据：由 `export_static_rope_baselines.py:94-95` 公式重算，未跑 `blocal`/`bhigh` 的官方实现交叉]
16. `m_j=null` 的槽：仅 `E8_zero51` 的槽 51（`ν=0`）。[已验证 | `JSON m_j` 中唯一 null]
17. `maxent_dilation` 的 `λ→0` 极限：`log_r = q·log_s` ⟹ `r_i = S^{q_i}` ⟹ `m_i = q_i = (i+0.5)/K`。[已验证：代码 `schedules.py:118-123` 的 `abs(tilt)<1e-8` 分支]
18. `evq_cosh_phi` 闭式：`φ = 1 − (1/τ)·asinh((1−u)·sinh τ)`，`u=(i+0.5)/K`；`τ→0` 时 `φ→u`。[已验证 | `schedules.py:182-189`；与 R3 §网格结论一致]
19. `nv_j` 反演一致性：我由 `1e6^{−arange(64)/64}` 重生成 Native，与部署数组最大绝对差 `4.29e-8`（fp32 存储舍入级）。[已验证 | `/tmp` 重算 vs `ground_truth_tables.json.methods.Native.nu_j`]
20. `MrPro` 快带 vs Native：`m_{j≤23} ≡ 0`，逐位相对差 `≤8.2e-8`（fp32 舍入）。[已验证：**修正** `rebuild:111` 的形状不匹配 bug 后重算]

---

## 7. 死路 / 已证伪（不得再试）

| # | 机制 | 失败证据 | 出处 | 等级 |
|---|---|---|---|---|
| D1 | **GapCapped**（把最大增量钉在 MrPro 的最大增量 `2/(N+1)` 上，靠加长过渡带来降洞） | GPU 结果为负；且 `ρ_max=1.4476@39` 与 MrPro 的 `1.4476@39` **完全相同** —— 帽操作**没有降低最大洞**，只削掉了中间段的洞。机制层面的解释：最大洞由 I2 端点强迫的 `Δ_39=1/9` 决定，帽等于它 ⟹ 洞不变 | `docs/research/ROPE_GAP_CAPPED_PROTOCOL_20260908.md:1-2`；`§4.5` 代数 | [部分证据]（负结果文档级；我的代数解释 [假设]） |
| D2 | **HighGapToLong（EVQ 字面运输：把高通 gap 搬给慢带）** | 36 行面板 `score_32K = 70.1389 / score_128K = 67.3611`，`wins=0 / losses=7` —— **面板最差**；且 `m23=−0.1557` 破 I1 | `JSON` `panel_scores`（源 `results/nongeometric_screen_20260909/results/HighGapToLong/summary.json`） | [已验证] |
| D3 | **HighGapToMid**（同预算改给中频） | 16/36 行中止、无裁决（`role: deferred`）。**不能引用为"中频更好/更差"的证据** | `JSON` role | [已验证]（仅"无裁决"这一事实） |
| D4 | **E2_tail_more**（慢带整体再 ÷1e6^{1/64}） | `score_32K=100.0 / score_128K=54.7222`，`losses=2`；`Σgap=1.6022 ≠ ln4`，破 I2 | `JSON` `panel_scores` | [已验证] |
| D5 | **E7_local_projection 的原始约束版** | `JSON` 明写"**修正约束后 ≡ BM，历史部署表保留**" ⟹ 部署表是**被取代的约束**下的产物，不得当作 E7 方法族的代表 | `JSON.methods.E7_local_projection.construction` | [部分证据]（文档级；我未重跑修正版） |
| D6 | **用 `Σm` 当守恒量 / 约束** | 红线 R4；且 §4.3 证 `Σm = B+24`，不含独立信息 | `INTEGRATION_20260910.md:26-31`；`NEXT_DERIVATION_KKT_PROBLEM.md:109-116` | [已验证]（红线 + 我的恒等式） |
| D7 | **把"17 个过渡 log-gap 之和 = ln S"当恒等式** | 真值 = `17·ln(b)/64 + ln4 = 3.6697+1.3863 = 5.0560`；权威文档明令禁止该措辞 | `NEXT_DERIVATION_KKT_PROBLEM.md:41-43`；`rebuild:563` | [已验证] |
| D8 | **"MrUni = 全表 ÷4"** | 假：MrUni 只在槽 24–39 与 MrPro 不同，槽 40+ 与 MrPro 逐位相同；`sum_m = 32 = 8+24`（若全表 ÷4 则 `sum_m` 应为 64） | `JSON.methods.MrUni.construction` 明写"GLM 复核已纠正 UNIFIED'全表÷4'表述" | [已验证] |
| D9 | **用静态几何代理量当 F 项** | 红线：`Σcos` 首根、碰撞能、覆盖率、平滑度、有效秩、Gram、能量 | `INTEGRATION_20260910.md:26-31` R1-R5 | [已验证]（红线） |
| D10 | **E8（慢带单槽置零）** | `score_128K = 50.5556`，`losses=2`；且该槽 `m/T/D` 不可定义，破坏本族坐标 | `JSON` `panel_scores` | [已验证] |
| D11 | **`fast_band_bitwise_equal_native` 字段** | `rebuild_ground_truth_tables.py:111` 用 24 元素切片对 64 元素数组做 `np.array_equal` ⟹ **恒为 False**（38 条全假）。**该字段不可引用**；正确事实是 `m_{j≤23}≡0`、相对差 `≤8.2e-8` | `rebuild_ground_truth_tables.py:111`；我的重算 | [已验证]（我独立复现并修正） |

---

## 8. 矛盾 / 术语不一致（含与权威文档的冲突点）

| # | 冲突 | 双方来源 | 我的判定 |
|---|---|---|---|
| C1 | **`BUDGET_ALLOCATION_MODEL_AND_CANDIDATES_20260910.md:43-47` 记录 Stack/N16/N15 的 max hole = `1.86 / 1.76 / 1.80`**，但 `ground_truth_tables.json` 给出 `1.4930 / 1.4608 / 1.4757` | `docs/research/BUDGET_ALLOCATION_MODEL_AND_CANDIDATES_20260910.md:43-47` **vs** `analysis/unify_20260910/tables/ground_truth_tables.json` | **文档数字不可复现**：从 `m_j` 出发按 `ρ_g = exp(ln(b)/64 + Δ_g lnS)` 重算只能得到 JSON 值。**权威文档 `INTEGRATION_20260910.md` 未涉及此数字**，故按"以可复现数值为准"取 JSON。必须显式标注为冲突。 |
| C2 | **GapCapped 有效带 `31→40`（文档）vs `32→40`（0-based 槽实算）** | `ROPE_GAP_CAPPED_PROTOCOL_20260908.md` **vs** `gap_capped.py:11-15` + `JSON m_j` | 槽 31 的 `m` 仍为 0（`1−2·9/18 = 0`），首个非零在槽 32。文档口径差 1 槽。 |
| C3 | **`UNIFIED §1-2` 的槽号与 `JSON` 差 1** | `NEXT_DERIVATION_KKT_PROBLEM.md` / UNIFIED 系 **vs** `ground_truth_tables.json:meta.definitions` | 见 `R2_ground_truth.md` 已登记；本文按 **0-based 槽、`gap_g` 介于槽 g/g+1** 的 JSON 口径写。 |
| C4 | **两套互不相同的 YaRN 实现**：(A) `[23,40]` 线性 ramp + `1+0.1ln4`（对账身份）；(B) `[0.2k, 0.9k]` smoothstep ramp + `1+0.07log₂S` 温度项 | `rebuild_ground_truth_tables.py:191-197` / `export_static_rope_baselines.py:94-99` / `rope_transport/tables.py:114-181` **vs** `scripts/core_text_phases/eval_pe_baselines.py:66-122` **vs** `scripts/lib/rope/schedules.py:233-245`（第三套：`scale^{ramp}·T^{0.5ramp}`） | **三套**。任何"YaRN"引用必须点名哪一套。ground truth 只对账 (A)。 |
| C5 | **"MrUni = 全表 ÷4"** | `BUDGET_ALLOCATION_MODEL_AND_CANDIDATES_20260910.md:11` **vs** `JSON.methods.MrUni.construction`（GLM 已纠正） | 见 D8。**BUDGET 文档该行错误**。 |
| C6 | **`INTEGRATION_20260910.md:17` 出现 `Σ_j m_j = log S / log S … 即 Σm 固定于过渡段`**，与同文档 R4"`Σm` 是自由决策变量"**自相矛盾** | `analysis/unify_20260910/INTEGRATION_20260910.md:17` **vs** 同文件 `:26-31` R4 | 按 R4 为准（红线），且 §4.3 的 `Σm = B+24` 表明 `Σm` 随 `B` 变、**不固定**。**该行表述应作废**。 |
| C7 | **"17 个增量" vs "16 个自由 DOF"** | `NEXT_DERIVATION_KKT_PROBLEM.md`（16 free DOF）**vs** 多处"17 gaps" | 两者都对（17 增量 − 1 等式 = 16 DOF），但**混用会算错维度**。建议写 `Δ∈R¹⁷, ΣΔ=1 ⟹ 16 DOF`。 |
| C8 | **`rebuild:111` 的 `fast_band_bitwise_equal_native` 恒 False** | 代码 bug vs 38 条目的 JSON 字段 | 见 D11。 |
| C9 | **E1 的"镜像/配对"公式未在构造脚本中恢复**：`E1_s28_reverse_matched`、`E1_s29_plus_matched` 在 `rebuild_ground_truth_tables.py` 的构造段里没有对应函数，只有部署反演值 | `JSON.methods.*.nu_used == 'deployed'` 且 sources 无公式出处 | **构造公式未挖掘到**（见 §9 Q3）。 |
| C10 | **`pi` 有两种口径**：`ω/scale`（`tables.py:95-108`，严格位置插值）vs `scale = max(max_seq_len/8192,1)`（`schedules.py:229-231`，按推理长度而非 `S`） | 两处代码 | 第二种的 `scale` 与 `S=4` 无固定关系；引用"PI"须点名。 |

---

## 9. 开放问题

- **Q1**：`B`（过渡预算）与面板分数**无单调关系**（§4.1）：`B=4`（GapCapped）与 `B=5.33`（MrPro）洞相同，`B=8`（BM）洞反而更小。那么 F 里"总量"与"形状"各占多少权重？**需要把 16 维 `Δ` 的形状指标（如 `max Δ`、`Δ` 的方差）单独作为坐标**，而不是用 `B` 一个标量。这是拟合 F 的第一优先问题。
- **Q2**：`E1_s28_less`（`B=5.3007`，`score_32K=87.2222 / 128K=83.3333`，`wins=2/losses=0`）是**面板上唯一 128K 高于 32K 的臂**；`MrPro` 是 `87.2222/78.1250`。`E1_s28_less` 与 `BS_ScaleTaper`、`Smooth_MrBudget` 的差是否就在 `Δ_28` 上？**单槽 `Δ_28` 的边际效应是否可量化**（`E1_pair28_29` 把它加倍后 `128K` 掉到 `73.9583`）。
- **Q3**：E1 镜像控制（C9）与 `E1_s28_less` 的**精确构造公式**未在 `rebuild_ground_truth_tables.py` 中恢复；`select.py` 的 `proposals()` 只生成"more"方向（`direction=-1,1` 都生成，但命名 `less/more` 与部署表的对应需逐位核对）。**建议以部署张量反演为准并显式标注"公式未恢复"**。
- **Q4**：`FullLagP2` 的 `m` 在槽 31 完成而非槽 40，其 `B=10.178` 远超所有臂。**"提前完成"是否等价于"把 I2 的锚点从 40 前移到 31"？**若是，则 I2 可能是 `m_{j≥40}=1` 的**一族**约束 `m_{j≥j_h}=1`，`j_h` 成为额外自由参数。这个泛化未被任何文档提出（**我的推断，[假设]**）。
- **Q5**：`maxent_dilation`（K18）的"重排不等式最优"只有 docstring 断言（`schedules.py:136-139`），**未给证明**，且**从未进入 36 行面板**。它是否与 MrPro 族在同一可行域上？`λ≠0` 时 `Σ_all gap ≠ ln4`（无端点锚定），故**它属于 §4.2 表中"未分类"的第六类**。**建议补一条对账**：把 `maxent_dilation(λ)` 与 `MrPro` 的 `Δ` 分布画在一起。
- **Q6**：`NTK_static` 守跨度不守端点却**从未在本地面板测过**（`role: external-baseline (equation)`，`panel_scores: null`）。它是**唯一能分离"跨度"与"端点"两条约束的实验臂**，但缺数据。
- **Q7**：`E2_boundary39 ≡ MrProN16`（§3.3）—— `select.py:84` 的 `E2_boundary{high}` 与径向族 `N = high−23` 是同一构造。**`E2_boundary41`（n=18）是否 ≡ N18？**未核对（`proposals()` 会生成它，但 JSON 里无该条目）。我能算出的部分：`m_j = t(t+1)/342`，`t=clip(j−23,0,18)` ⟹ `m40 = 17·18/342 = 0.894737 ≠ 1`，**破 I2**（`m41 = 1` 才完成）。所以 N18 若存在，它属于"把 I2 锚点从 40 后移到 41"的那一类，与 Q4 的 P2 反向。**[假设]**
- **Q8**：`Smooth_MrBudget` 的 `roughness = 0.004886399` 我未独立重跑（缺 `results/nongeometric_screen_20260909/` 镜像）。**该数字与 KKT 证书的完整 `dual_multipliers` 建议单独复核一次**。

---

## 10. 覆盖声明

**读了**（全文或指定区间）：
- `analysis/unify_20260910/tables/rebuild_ground_truth_tables.py`（676 行，重点 L30-33、L106-114、L136-141、L165-298、L510-634）；`tables/ground_truth_tables.json`（321 KB，**只用程序化重算读取，未逐字通读**）；`tables/GROUND_README.md`（摘要级）。
- `analysis/unify_20260910/`：`INTEGRATION_20260910.md`（L17、L26-31、L50-52、L64-72、L84-88、L97-106）；`NEXT_DERIVATION_KKT_PROBLEM.md`（L25-27、L41-43、L45-55、L78-97、L109-116）；`STARTING_POINT_YARN_VS_MRPRO.md`（F1–F9）。
- `experiments/nongeometric_screen/`：`gap_budget_transfer.py`（L20-52 全文）、`smooth_budget.py`（全文 65 行）、`long_bridge.py`（L17-26）、`scale_taper.py`（全文 56 行）、`select.py`（全文 191 行）、`project.py`（全文 96 行）、`numerical_controls.py`（全文 34 行）。`operators.py`、`distance_operator.py`：**只读了摘要级描述，未逐行读**。
- `scripts/lib/rope/`：`gap_capped.py`（L9-20）、`schedules.py`（L1-205、L211-325 全文）。
- `scripts/analysis/`：`export_static_rope_baselines.py`（L92-102）、`rope_transport/tables.py`（L95-108、L114-181 摘要）、`audit_qwen_p2_full_lag.py`（L29-49 摘要级；**实装跑了该脚本**，输出 `/tmp/p2audit.json`）。
- `scripts/core_text_phases/eval_pe_baselines.py`（L66-122 摘要级）。
- `docs/research/`：`BUDGET_ALLOCATION_MODEL_AND_CANDIDATES_20260910.md`（L11、L14、L16-20、L24-33、L43-49）；`ROPE_GAP_CAPPED_PROTOCOL_20260908.md`（L1-2 + 协议段）。
- 兄弟 digest：`mine/R2_ground_truth.md`（§0-1.4）、`mine/R3_evq_math.md`（grep 级交叉核对，确认 §3.10 不在其覆盖内）。

**未读 / 跳过**：
- `ground_truth_tables.json` 的 `reconciliation`（126 项）与 `mismatches`（10 条）**逐条内容**——已由 `R2_ground_truth.md` 覆盖，本文只用到 L563/L629-634 两条水床校验。
- `experiments/nongeometric_screen/worker.py`、`*.json` receipts、`docs/research/` 下 P1–P8 其它 `sol*.md`、`astra*.md`、所有 `analysis/kkt_20260910/mine/A*.md` / `T*.md`（属其它挖掘者的分工）。
- `paper-2027/` 全部 tex、`tests/`、`docs/theory/EVQ_COSH_THEORY.tex`（R3 已覆盖）。
- `results/nongeometric_screen_20260909/`：**本机/仓库/历史树均不存在**（09-06 瘦身删除），故面板分数只能取自 `ground_truth_tables.json` 的转写值，**我无法重跑任何面板臂**。

**未能复核的事项**：18 条以外的 `nu_used=='formula'` 条目的部署一致性（无部署对象）；E1 镜像控制的构造公式（C9/Q3）；`Smooth` 的 KKT 完整证书（Q8）；`maxent_dilation` 的重排最优性证明（Q5）。

**一次未重复的声明**：本文件未修改仓库任何文件；唯一写盘位置为 `/tmp`（`/tmp/p2audit.json` 等）与本输出文件本身。
