# R2 digest：G1 频率表地面真值（`GROUND_README.md` / `ground_truth_tables.json` / `rebuild_ground_truth_tables.py`）

挖掘者：R2。日期：2026-09-10。**只读**，未修改仓库任何文件；脚本重跑在 `/tmp/gt_check/` 内进行（见 §5）。

材料根：`/Users/yang/projects/hybrid-rope/analysis/unify_20260910/tables/`

---

## 0. 一句话结论

这是一张 **30 个静态 64 槽频率表 + 8 个无表算子臂 = 38 条目的数值面板**，每条目给出 `nu_j[64] / m_j[64] / T_j[64] / D_j[64] / r_j[64] / gap_j[63]` 六组逐槽量 + 水床和 + 洞比率 + 分数 + 对账。**离线可独立复核的部分全部通过**：我用与脚本无关的自写代码，从 JSON 自带的 `nu_j` 重算全部 6 组量，30/30 表逐位一致（§3）；用闭式重建 MrPro / BM / N16 / N15 / YaRN-linear / NTK / E1 族 / LBS 族 / MrUni / E2，**bit-exact 全部命中**（§4）。但 **脚本本身在本机不可运行**（依赖 `results/nongeometric_screen_20260909/` 镜像，该目录本地/仓库/历史均不存在，§5.1），且发现 **1 个 JSON 存储字段的系统性 bug**（`endpoint_delta_m.fast_band_bitwise_equal_native` 恒为 `false`，形状不匹配，§6.1）与 **README 的 3 处与自身 JSON 冲突的表述**（§6.2/6.3）。

---

## 1. JSON schema（可直接拿来拟合）

### 1.1 顶层

| 键 | 内容 |
|---|---|
| `meta` | `constants` / `definitions` / `evidence_policy` / `score_units` |
| `methods` | dict，**38 条**（键=方法名） |
| `reconciliation` | 列表，**126 项**文档锚点核对（`{item, doc_claim, recomputed, status}`） |
| `mismatches` | 列表，**10 条**字符串 |

`meta.constants`（出处：JSON `meta.constants`；与 GROUND_README §1 一致）：
`W=32768`、`S=4`、`L=131072`、`Dr=64`、`base=1e6`、`lnS=1.3862943611198906`、
`native_log_gap=0.21586735246819178`、`native_period_ratio=1.2409377607517196`、
`official_gain=1.138629436111989`、`p2_gain=1.102585782722872`。[已验证：重算 ln(1e6)/64=0.2158673524681918、1e6^(1/64)=1.2409377607517194、1+0.1ln4=1.138629436111989，全一致]

### 1.2 每方法条目字段

- 元数据：`role`、`panel`、`construction`、`sources[]`、`nu_used`（`'deployed'` / `'formula'` / `null`）、`gain`、`panel_scores`（可选）、`notes`（可选）
- `formula_vs_deployed`：`{bit_exact, max_abs_freq_diff, max_rel_freq_diff, sha256_deployed, sha256_rebuilt, note?}`（无对账对象时为 `{bit_exact: null, note}`）
- **逐槽量**（长度固定）：`nu_j`(64)、`m_j`(64)、`T_j`(64)、`D_j`(64)、`r_j_native`(64)、`gap_j`(63)
  - `nu=0` 的槽（E8_zero51 的槽 51）→ `m/T/D` 为 `null`
  - `gap_j` 是**槽 g 与 g+1 之间**的 0-based gap（j=0..62）
- **聚合量**：`sum_transition_gap_increments_gaps23_39`、`sum_all_gap_increments_slots0_63`、`expected_if_endpoints_fixed_ln4`、`hole_ratio_max_global`、`hole_ratio_argmax_gap`、`hole_ratio_max_transition`、`hole_ratio_argmax_transition`、`native_period_ratio`、`sum_m_all_slots`、`changed_slots_vs_MrPro[]`、`endpoint_delta_m{m_23, m_40, fast_band_bitwise_equal_native, tail_m40to63_all_equal_MrPro}`

**消费者必读（易踩）**：`nu_j` 字段**不总是公式重建值**。脚本 `method_entry`（`rebuild_ground_truth_tables.py:136-141`）里 `nu = deployed if deployed is not None else nu_rebuilt`。所以
- `nu_used=='deployed'`（22 条）：`nu_j`/`m_j`… 是**部署 fp32 张量**的反演；
- `nu_used=='formula'`（5+3=8 条：Stack/N16/N15/YaRN_smoothstep/NTK 为纯公式，其余无表臂无 `nu_j`）。

### 1.3 全部方法名（38）

**参考/基线**：`Native`、`MrPro`、`MrProBM`
**E1 单/双槽手术**：`E1_s28_less`、`E1_s29_more`、`E1_pair28_29`
**E1 镜像控制**：`E1_s28_reverse_matched`、`E1_s29_plus_matched`
**E2/E4/E8**：`E2_tail_more`、`E4_pair25_29`、`E8_zero51`
**KKT/斜坡**：`Smooth_MrBudget`、`MrUni`
**运输臂**：`HighGapToLong`、`HighGapToMid`
**长桥**：`LongBridgeSlower`、`LongBridgeFaster`
**历史/控制**：`FullLagP2_Transfer3B`、`Control_Mr_gain074`、`E3_BM_gain074`、`E3_BM_gain1`、`E7_local_projection`
**09-08 面板**：`GapCapped`
**deferred**：`BM_ScaleTaper`
**队列未执行（无分数）**：`StackFrontBack`(0446)、`MrProN16`(0448)、`MrProN15`(0449)
**外部基线/变体**：`YaRN_linear_official`、`YaRN_smoothstep_variant`、`NTK_static`
**算子臂（无 64 槽表，`nu_j=null`，8 条）**：`E5_layer21`、`E5_layer27`、`E5_layer32`、`E6_layer14_group0`、`E6_layer14_group1`、`E7_norm_matched_BM`、`E10_dual_frequency`、`E9_distance`

### 1.4 可作为 F 零件的定义（全部带脚本行号，精度见 §3/§4）

| 量 | 定义 | 出处 |
|---|---|---|
| 部署坐标 | `ν_j = ω_j·S^{−m_j}`，`ω_j = b^{−j/64}`，`b=1e6`，`S=4` | README §1；脚本 `:30-40` |
| 反演 | `m_j = log_4(ω_j/ν_j)` | 脚本 `:82` |
| 周期 | `T_j = 2π/ν_j` | 脚本 `:83` |
| 识别地平线 | `D_j = W·4^{m_j}` | 脚本 `:84` |
| 原生格点数 | `r_j = W·ω_j/(2π)`（**逐槽不同**，只是"对全族（各方法）相同"） | 脚本 `:85-86`；已验证 `max|r−Wω/2π|=9.1e-13` |
| log-gap | `gap_g = ln(ν_g/ν_{g+1})`，`g=0..62` | 脚本 `:88` |
| **洞比率 ≡ exp(log-gap)** | `ρ_g = T_{g+1}/T_g = ν_g/ν_{g+1} = e^{gap_g}` | 脚本 `:95-98`；这是恒等式，我实测 MrPro g39: `e^{0.154033+0.215867}=1.4476` = 表值 ✓ |
| 原生洞比 | `ρ_0 = b^{1/64} = 1.2409378` | README §1 |
| **水床恒等式（过渡段）** | `Σ_{g=23}^{39}(gap_g − lnb/64) = ln4·(m_40 − m_23)`；端点固定 ⇒ `= ln4 = 1.386294` | README §2；脚本 `:90-93`；**独立重算逐表吻合，残差恒为 2.45e-8（fp32 地板）** |
| **水床恒等式（全表）** | `Σ_{g=0}^{62}(gap_g − lnb/64) = ln4·(m_63 − m_0)` | README §2；脚本 `:90` |
| 守恒式 | `m_40 − m_23 = 1 ⟺ Σ_{g=23}^{39} Δ_g = 1`，`Δ_g := m_{g+1} − m_g ≥ 0` | NEXT_DERIVATION §1.3；我实测 Σ=1.0 ✓ |
| gain | `1+0.1·ln S = 1.138629436111989`；P2 用 `1+0.074·ln4 = 1.102585782722872` | README §1；JSON `gain` 字段 |

**MrPro 径向族（MrRoPE Eq.14）闭式**（脚本 `:165-172`）：
`m_q = q(q+1)/(N(N+1))`，`q = clip(j−23, 0, N)`，`N=17`，界 `dl=23, dh=40`；`j≥40` 时 `q=17 ⇒ m=1`。
⇒ 逐 gap 增量 `Δ_g = 2(g−22)/306`（`g=23..39`，等差、步长 2、首项 2/306）。**N′=16/15 同族**：`m_q=q(q+1)/(N'(N'+1))`。
⇒ 闭式预算：`Σm(N') = 40 − N' + (N'+2)/3 = (122 − 2N')/3`；`B(N') := Σm − 24 = (50 − 2N')/3`。
（N=17→29.3333/5.3333、16→30.0000/6.0000、15→30.6667/6.6667 —— 与 JSON `sum_m_all_slots` 及 README §6 的 B 值全部吻合 [已验证]。README §4.3 只印了 `(N′+2)/3+…` 的片段，它单独给 N15→5.667，与同行"6.667"不符，见 §6.4。）

**BM 闭式**（脚本 `:181-186`）：`m_q = q(q+1)(3N+2−2q)/(N(N+1)(N+2))`，`N=17`。
**YaRN 官方线性（论文式，界 23/40）**（脚本 `:191-197`）：`ν_j = ω_j(1−t+t/S)`，`t=clip((j−23)/17,0,1)`。等价 `m_Y(t) = −log_S[1−(1−1/S)t]`（STARTING_POINT F1）。
**NTK 静态**（脚本 `:199`）：`ν_j = ω_j·4^{−j/63}`（∀64 槽，无平台）。
**Smooth KKT 问题（可作 F 的直接模板）**（脚本 `:207-226`）：
`min_ε εᵀLε` s.t. `1ᵀε = 1`、`wᵀε = 16/3`（`w=(16,15,…,0)`）、`ε ≥ 0`，`L` = 17×17 路径拉普拉斯（`2I − 移位`），`ε` 累加成 `m`。粗糙度证书 `εᵀLε = 0.004886399`（我独立重算逐位一致 [已验证]；活动集为 `ε_0=ε_1=0`，末步长 `Δm(39→40)=0.042484`）。注意 `B=16/3` 是**加权矩预算** `Σ(16−i)ε_i`，**不是** `Σε`（`Σε≡1`）。

---

## 2. 关键已验证数字（含出处）

### 2.1 常数与原生表
| 数字 | 值 | 出处 | 等级 |
|---|---|---|---|
| `lnb/64` | 0.21586735246819178 | JSON `meta.constants` | [已验证] |
| `b^{1/64}` 原生洞比 | 1.2409377607517196 | 同上；MrPro/Native 实读 | [已验证] |
| `ln4` | 1.3862943611198906 | 同上 | [已验证] |
| MrPro fast 段（槽 0–23）与原生 | **逐位相同** | 我独立重算 `array_equal` | [已验证] |

### 2.2 水床和 / Σm / 洞（我独立重算，与 JSON 逐位一致）

| 方法 | Σm(全64) | Σtrans | Σall | 洞 max(过渡) | @gap | 出处 |
|---|---|---|---|---|---|---|
| MrPro | 29.3333 | 1.386294 | 1.386294 | 1.4476 | 39 | JSON `methods.MrPro` |
| MrProBM | 32.0000 | 1.386294 | 1.386294 | 1.3934 | 31 | 同上 |
| E1_s28_less | 29.3007 | 1.386294 | 1.386294 | 1.4476 | 39 | 同上（README §3 作 29.301 ✓）|
| E1_s29_more | 29.3791 | 1.386294 | 1.386294 | 1.4476 | 39 | 同上（29.379 ✓）|
| E1_pair28_29 | 29.3464 | 1.386294 | 1.386294 | **1.4608** | **28** | 同上（README "1.46× 洞在 gap28" ✓）|
| Smooth_MrBudget | 29.3333 | 1.386294 | 1.386294 | 1.4513 | 35 | 同上 |
| MrUni | 32.0000 | 1.386294 | 1.386294 | 1.3464 | 28 | 同上 |
| HighGapToLong | 25.2069 | **1.602162** | 1.386294 | 1.5279 | 39 | 同上 |
| HighGapToMid | 26.6083 | **1.602162** | 1.386294 | 1.4476 | 39 | 同上 |
| LongBridgeSlower | 29.5602 | 1.386294 | 1.386294 | 1.4930 | **38** | 同上 |
| LongBridgeFaster | 29.1253 | 1.386294 | 1.386294 | 1.6192 | 39 | 同上 |
| FullLagP2 | 34.1789 | 1.385519 | 1.386294 | **2.9696** | **29** | 同上 |
| E2_tail_more | 33.0705 | **1.602162** | **1.602162** | 1.7964 | 39 | 同上 |
| E8_zero51 | 28.3333 | 1.386294 | 1.386294 | 1.4476 | 39 | 同上 |
| E4_pair25_29 | 29.3333 | 1.386294 | 1.386294 | 1.4476 | 39 | 同上 |
| StackFrontBack | 29.5275 | 1.386294 | 1.386294 | **1.4930** | **38** | 同上 |
| MrProN16 | 30.0000 | 1.386294 | 1.386294 | **1.4608** | **38** | 同上 |
| MrProN15 | 30.6667 | 1.386294 | 1.386294 | **1.4757** | **37** | 同上 |
| YaRN_linear_official | 30.1042 | 1.386294 | 1.386294 | 1.4599 | 39 | 同上 |
| NTK_static | 32.0000 | **0.374079** | 1.386294 | 1.2685 | 28(=41) | 同上 |

计数（**与 README §4.1 的"17 张 / 18 张"不符，见 §6.4**）：`Σtrans=ln4`（±1e-4）的表 **24 张**；`Σall=ln4`（±1e-4）的表 **28 张**；例外为 Native(0/0)、E2(1.602/1.602)、HighGapToLong+Mid(1.602/1.386)、P2(1.3855/1.386)、NTK(0.374/1.386)。

### 2.3 面板分数（`panel_scores.score_*_pct` = `summary.json` 的 `macro_accuracy×100`）

36 行面板：MrPro **87.2222/78.1250**（基线，经 `E1_s28_less/summary.json` 的 `baseline` 字段实读复核 [已验证]）、MrProBM 91.6667/70.8333、E1_s28_less 87.2222/**83.3333**、E1_s29_more 95.5556/77.9167、E1_pair28_29 87.2222/73.9583、Smooth 87.2222/68.3333、MrUni **64.5833**/73.3333、HighGapToLong **70.1389/67.3611**、LBS 80.5556/80.0694、LBF 87.2222/73.9583、P2 72.9167/81.6667、Control_Mr_gain074 98.3333/75.3472、E3_BM_gain074 100.0/70.0、E3_BM_gain1 89.5833/58.8194、E7_local_projection 90.0/68.6111。
12 行屏：E2 100.0/54.7222、E8 100.0/50.5556、E4 与两个镜像控制 100.0/64.4444、E5×3/E6×2/E7_norm/E9/E10 全为 100.0/64.4444。
**未执行队列 `StackFrontBack`/`MrProN16`/`MrProN15`、`BM_ScaleTaper`、`HighGapToMid`、`YaRN_smoothstep_variant`、`NTK_static`、`Native` 无 `panel_scores`（score=null，不虚构）**。[已验证：JSON 实读]

### 2.4 与权威文档的正面交叉验证（我复算，全部命中）

| 权威文档断言 | 出处 | 我从 G1 表复算 | 等级 |
|---|---|---|---|
| Qwen4× 槽24 `ν/ω`：YaRN 0.9559 / MrPro 0.9910；槽28 0.7794/0.8729；槽39 0.2941/0.2916 | STARTING_POINT §4 F5 | 0.9559/0.9910、0.7794/0.8729、0.2941/0.2916 | [已验证] |
| `Σ(ν^M−ω)²/Σ(ν^Y−ω)² = 0.4841` | STARTING_POINT §2 F3 | **0.4841** | [已验证] |
| 槽24 相对降频 YaRN 4.4118% / MrPro 0.9020% | 同上 | 4.4118% / 0.9020% | [已验证] |
| MrPro m24=.0065、m39=.8889；E1_s28 m28=.0654；P2 m36–39=1；E2 m40=1.1557 | NEXT_DERIVATION §1.1 | .006536/.888889/.065359/全1/1.155715 | [已验证] |
| MrUni m28 = .294 | NEXT_DERIVATION §3 | 0.29411767 | [已验证] |
| LBS m36–39 = .625/.729/.847/.980 | NEXT_DERIVATION §3 | 0.625169/0.729476/0.846534/0.979914 | [已验证] |
| P2 gap29 = 1.088 巨洞、洞比 2.97× | README §4.4 | **1.088415**、2.969564 | [已验证] |
| MrPro D36–39 = 75/85/97/112K（欠完成） | README §5.2 | 74737/84845/97197/112361 | [已验证] |
| OLMo 面板数字（BM 41.67% vs MrPro 7.09%） | INTEGRATION §3 | 不在本表内（G1 只覆盖 Qwen3B 屏） | — |

---

## 3. 独立重算（第一层：JSON 自带 `nu_j` → 全部 6 组量）

方法：**不调用脚本任何函数**，用 `/tmp/gt_check/verify.py` 自写 `ln` 反演、`2π/ν`、`W·4^m`、`ln(ν_g/ν_{g+1})`、水床两口径，逐表比对 JSON 存储字段。

结果：**30/30 表，`m/T/D/gap_j/Σtrans/Σall` 六项最大偏差全部 = 0.00e+00**（唯一非零是 `E8_zero51` 的 `gap` 差为 `inf−inf=nan`，两侧同为 `inf`，属预期）。
⇒ **JSON 内部自洽性 100% 通过**；也就是说表里的 `m/λ/D/洞/水床和` 与 `nu_j` 是同一份数据的自洽导出，不存在"存储值 ≠ 数组值"的漂移。

## 4. 文献/公式抽样手算核对（第二层：闭式 → `nu_j` 逐位）

`/tmp/gt_check/handcheck.py`。判据 = float32 逐位（`array_equal`）+ sha256。**抽到 14 个方法、每方法 ≥3 槽位**，远超 3×3 要求：

| 方法 | 闭式（我独立实现） | bit_exact | 与 JSON `formula_vs_deployed.bit_exact` 一致? |
|---|---|---|---|
| MrPro | `m_q=q(q+1)/306` | **True**（sha 33cbe3a40994ac2a… 两边相同）| ✓ True |
| MrProBM | `m_q=q(q+1)(53−2q)/5814` | **True** | ✓ True |
| MrProN16 | `m_q=q(q+1)/272` | **True**（sha e88bff09…）| ✓（JSON 无对账对象，我补上了）|
| MrProN15 | `m_q=q(q+1)/240` | **True**（sha 6f8aa1bd…）| ✓（同上）|
| YaRN_linear_official | `ν=ω(1−t+t/4)` | **True**（sha 6641d434…）| ✓ True |
| NTK_static | `ν=ω·4^{−j/63}` | **True** | ✓（无对账对象，我补上）|
| E1_s28_less | `m28:=m27` | **True** | ✓ True |
| E1_s29_more | `m29:=m30` | **True** | ✓ True |
| E1_pair28_29 | `m28:=m27 且 m29:=m30` | **True** | ✓ True |
| E4_pair25_29 | `m25=m29=(m25+m29)/2` | **True** | ✓ True |
| E2_tail_more | 尾部 `÷1e6^{1/64}` | **True** | ✓ True |
| LongBridgeSlower | 槽36–39 `ν−=1/131072` | **True** | ✓ True |
| LongBridgeFaster | 槽36–39 `ν+=1/131072` | **True** | ✓ True |
| MrUni | `m_j=(j−23)/17`（j=24..39，段外=MrPro）| **True** | ✓ True |

**手算明细（摘代表性 6 组，槽位 × 公式值 × JSON 值）**

| 方法 | j | q | `m` 手算（精确有理） | `m_JSON` | `ν` 手算(f32) | `ν_JSON` |
|---|---|---|---|---|---|---|
| MrPro | 28 | 5 | 0.098039215686 | 0.098039215 | 0.0020700200 | 0.0020700200 |
| MrPro | 36 | 13 | 0.594771241830 | 0.594771236 | 0.0001848894 | 0.0001848894 |
| MrPro | 63 | 17 | 1.000000000000 | 1.000000000 | 0.0000003102 | 0.0000003102 |
| MrProBM | 28 | 5 | 0.221878224974 | 0.221878242 | 0.0017434761 | 0.0017434761 |
| MrProBM | 39 | 16 | 0.982456140351 | 0.982456149 | 0.0000565266 | 0.0000565266 |
| MrProN16 | 28 | 5 | 0.110294117647 | 0.110294126 | 0.0020351496 | 0.0020351496 |
| MrProN15 | 36 | 13 | 0.758333333333 | 0.758333298 | 0.0001473798 | 0.0001473798 |
| YaRN | 28 | t=.294118 | — | — | 0.0018482767 | 0.0018482767 |
| YaRN | 40 | t=1 | — | — | 0.0000444570 | 0.0000444570 |
| NTK | 31 | — | 0.492063492063 | 0.492063 | 0.0006273332 | 0.0006273332 |

**唯一一次"差异"是我自己的错**：初版我把 E2 尾部增量写成 `1/64`，得到 21% 偏差；正确增量是 `lnb/64 / ln4 = ln(1e6)/(64·ln4) = 0.1557153794478451`，改对后 bit_exact=True，且与 JSON 的 `m_40=1.1557154053858685`、`Σall=1.602161716` 吻合（`ln4×1.1557154=1.6021617` ✓ README §3 "m=1.155715" 正确）。**教训记录在此以防复现者踩同一坑：`m` 的尾部违例增量**是 `0.1557`（log₄ 单位）**不是** `1/64≈0.0156`。

**结论**：JSON 里的 `formula_vs_deployed.bit_exact=True`（18 条）**经我第三方独立复现，18/18 全部为真**，无一条夸大。README §3 的"18/18 bit-exact"成立。

### 4.1 逐 gap 手算（供拟合直接用）

MrPro `Δ_g = m_{g+1} − m_g`，`g=23..39`：`306·Δ_g = 2,4,6,…,34`（等差，首项 2，步长 2），`ΣΔ = 1.0` 精确。
即 `Δ_g = 2(g−22)/306`。**注意**：`NEXT_DERIVATION §3 表`写"MrPro `Δ_q ∝ (2q+1)`"（`NEXT_DERIVATION_KKT_PROBLEM.md:85`），与同文档 `:29` 的 `ε_j=2(1+j−dl)/((1+n)n)`（⇒ `2(j−22)/306`）以及实测表**不一致**（`(2q+1)` 归一化后与实测逐项不符；见 §7 矛盾 5）。

对照量（我复算）：`gap_g − lnb/64` 对 MrPro = `0.009060762, 0.018121453, …`（`g=23..39`，等差步长 `0.009060691`），首项 `= ln4·Δ_23 − lnb/64 = ln4×0.0065359 − 0.2158674 = 0.0090608` ✓。

### 4.2 HighGap 运输臂手算复核（README §3）

- `HighGapToLong`：gaps 0–22 每项 `Δgap = −0.0093855506`，`−lnb/64/23 = −0.0093855371` ✓；gaps 36–39 每项 `Δgap = 0.0539668x`，`lnb/64/4 = 0.0539668381` ✓；抽走总量 `0.2158673195` vs `lnb/64 = 0.2158673525`（差 3.3e-8 = fp32 地板）。`ΣΔgap = 0`（端点固定）✓。
- `HighGapToMid`：recipient = **gaps 26–31**（6 项 × `0.0359779`）✓ 与 README §3 一致；donor 同 gaps 0–22。
- **两臂都改动 fast 段槽 1–23**（逐位与 Native 不同）。这与 README §4.5 的"全部面板表 fast 段与原生逐位相同"冲突，见 §6.2。

---

## 5. 脚本可运行性（实测）

### 5.1 结论：**本机不可运行**（依赖缺失）

`python3 analysis/unify_20260910/tables/rebuild_ground_truth_tables.py` 第一步即失败：

```
FileNotFoundError: '/Users/yang/projects/hybrid-rope/results/nongeometric_screen_20260909/reference_tables.json'
```

`MIRROR = ROOT/'results/nongeometric_screen_20260909'`（脚本 `:27`）**在本地磁盘、当前分支、`main_0726`/`main_0726_09_06`/`main`/`origin/09_09` 全部不存在**（`find /` 无 `reference_tables.json`；`git rev-list --all -- <path>` 为空 = 从未入库）。该镜像应在 GPU 服务器（digest_thread-0909-pm §:126 记"远程根 `/root/autodl-tmp/nongeometric_screen_20260909`"），**仓库瘦身/关机后本地无副本**。
⇒ README §7 的 `python3 ...` 复现指令 **当前环境不可执行**；JSON 是这份镜像的唯一本地派生品。**下游拟合者只能用 `ground_truth_tables.json` + `docs/research/*.json`（carrier/P2/GapCapped 三份候选 JSON 本地存在）**，不能重跑对账。

本地**存在**的少数外部依赖（我确认可读）：`docs/research/ROPE_CARRIER_REMOVAL_CANDIDATE_20260907.json`、`docs/research/ROPE_QWEN15_FULL_LAG_P2_CANDIDATE_20260907.json`、`docs/research/ROPE_GAP_CAPPED_CANDIDATE_20260908.json`、`paper-2027/.../REFERENCE_CORRECTED_K128_S4_NLL_RECEIPT_20260901.json`。缺失的有：整个 `results/nongeometric_screen_20260909/`（含 `reference_tables.json`、`results/*/contract.json`、`results/*/summary.json`、`deferred_queue/20260910_candidate_quality/*.json`、`planned_controls/p2_gap_comparison.json`）。

### 5.2 我为验证所做（未碰仓库）

把脚本复制到 `/tmp/gt_check/run.py`，仅改 `ROOT`（硬编码绝对路径）与 `OUT`（`/tmp/gt_check/gt_out.json`），确认仍然失败于同一缺失镜像。**仓库内 `ground_truth_tables.json` 未被覆盖、未被读取后改写**（`git status` 仅显示 `analysis/kkt_20260910/` 未跟踪 = 我的输出目录）。后续所有验证改为「读 JSON 自带数组 + 自写独立实现」的两层复核（§3/§4），这恰好也是更强的第三方验证：不复用脚本的 `metrics()`。

---

## 6. 脚本/文档中的 patch 残留与假阳性（任务要求 3）

### 6.1 真 bug：`fast_band_bitwise_equal_native` 恒为 `false`（形状不匹配型假阴性）

脚本 `metrics()`（`rebuild_ground_truth_tables.py:111`）：
```python
'fast_band_bitwise_equal_native': bool(np.array_equal(f32(nu[:24]), f32(NATIVE))),
```
左 `f32(nu[:24])` 形状 `(24,)`，右 `f32(NATIVE)` 形状 `(64,)`。`np.array_equal` 对形状不等**静默返回 False**（不抛异常）。因此 **38 条中每一条该字段都是 `false`**，**零信息量**。

我实测：`MrPro`/`MrProBM`/`Smooth`/`MrUni`/`LBS`/`GapCapped`/`Stack`/`N16`/`N15`/`E1_*`/`E3_*`/`Control`/`E7`/`E4`/`E8`/`YaRN`/`BM_ScaleTaper` 的槽 0–23 **确实逐位等于 Native**（我重算为 True，JSON 记 False = **假阴性**，共 ≥20 条）。

**给下游的口径**：该字段必须**丢弃并自行重算**。我自己算出的真值是：
- fast 段逐位 = Native：`MrPro, MrProBM, E1_s28_less, E1_s29_more, E1_pair28_29, E4_pair25_29, E8_zero51, E1_s28_reverse_matched, E1_s29_plus_matched, Smooth_MrBudget, MrUni, LongBridgeSlower, LongBridgeFaster, Control_Mr_gain074, E3_BM_gain074, E3_BM_gain1, E7_local_projection, GapCapped, BM_ScaleTaper, StackFrontBack, MrProN16, MrProN15, YaRN_linear_official, YaRN_smoothstep_variant, E2_tail_more`（E2 只改尾段）
- fast 段 ≠ Native：`Native`(自比=是)、`HighGapToLong`（槽 1–23 改）、`HighGapToMid`（槽 1–23 改）、`FullLagP2_Transfer3B`（槽 12–23 改，`m_23=+0.000559`）、`NTK_static`（按构造）

### 6.2 patch/死代码残留（不影响数值，但给类型检查器制造噪音）

| 行 | 内容 | 性质 |
|---|---|---|
| `:122` | `METHODS = []  # provisional; replaced below after helpers` | **死赋值**，`:133` 立即 `METHODS = {}` 覆盖。**这就是 INTEGRATION §8-4 说的 "Pyright 报 setitem 类型假阳性"的来源**：Pyright 把 `METHODS` 推断为 `list[...] ∪ dict[...]`，于是 `:342` `METHODS[name] = e`（str 键 setitem）被误报。运行时无影响。 |
| `:121`+`:132` | `MISMATCHES = []` 连续声明两次 | 冗余 |
| `:244` | `donors = np.arange(23)` 随后 `:246` 立刻用 `int(first_changed[0])-1` 重算覆盖 | 死赋值；行内注释 `# first_changed=24 → high_end=23? 代码: first_changed[0]-1=23? 见下验证` 是**原代码语义未复原的现场标记**（重构者对 HighGapToLong 原式 donor 索引不确定）。实测结果 donor=gaps 0–22 正确。 |
| `:238` | `f32(MRPRO.copy()[... ])` | `[...]`（Ellipsis 索引）是 no-op 视图，残留 |
| `:286-287` | `stack_nu = pair_nu.copy()  # base = MrPro` 随后立即 `stack_nu = f32(MRPRO.copy())` | 死赋值，且注释与代码矛盾（base 是 MrPro 不是 pair） |
| `:338-339` | `if g is None and dep_arr is not None and name in ('StackFrontBack',): g = GAIN` | **硬编码单方法特判**（patch），应写进 `add()` 参数 |

### 6.3 INTEGRATION §8-4 的"342 行"已过期

`INTEGRATION_20260910.md:113` 写"其脚本 342 行 Pyright 报 setitem 类型假阳性"。实测 `rebuild_ground_truth_tables.py` = **676 行**（`wc -l`）。"342 行"是早期版本计数 [口径不一致，非数值错误]。

### 6.4 计数口径漂移（README/INTEGRATION 自报 vs 交付物）

| 文档断言 | 出处 | 交付 JSON 实测 |
|---|---|---|
| "17 张端点固定表 Σtrans=ln4" | GROUND_README §4.1 | **24 张**（判据：`Σtrans−ln4` <1e-4）|
| "18 张表 Σall=ln4（含 NTK）" | GROUND_README §4.1 | **28 张** |
| "32 行方法全谱 / 31 行方法全谱" | INTEGRATION §9、`:121` | `methods` = **38 条**（30 静态表 + 8 算子臂）|
| "G1 表 v1 320KB" | INTEGRATION §2 | `ground_truth_tables.json` = 321,398 B ✓（320KB 属实）|
| 脚本 342 行 | INTEGRATION §8-4 | 676 行 |

### 6.5 `endpoint_delta_m.m_23 / m_40` 字段是可信的

同一 dict 里另两个字段没有形状问题，实测全部能对上（`m_23`/`m_40` 见 §2.2 端点表）。**只有 `fast_band_bitwise_equal_native` 一个字段坏**。

---

## 7. 与权威文档/任务描述的矛盾（逐条给两边出处）

1. **README §4.5 vs 自家 JSON/§4.1（fast 段不变量范围）**
   - GROUND_README §4.5："全部面板表 fast 段（槽0–23）与原生逐位相同…**除 P2…与 NTK/YaRN（按构造）**"。
   - 实测：**`HighGapToLong` 与 `HighGapToMid` 的槽 1–23 也逐位不同**（`m_23 = −0.155715`），且这正是 README §4.1 自己写的"EVQ 运输确实是从带外（gaps0–22）注入"的必然结果 —— 两节自相矛盾。
   - 反向也有错：`YaRN_linear_official` 的 fast 段**逐位等于 Native**（`t=clip((j−23)/17,0,1)` 在 `j≤23` 恒为 0），把它列为"例外"不成立。
   - ⇒ 下游若想用"fast 段=原生"作为 I1 约束的判据，**必须用我 §6.1 的真值名单**，不能用 README 那句话。

2. **README §6 表 vs JSON（StackFrontBack 的洞位置）**
   - `GROUND_README.md:144` 表列"**1.493@g35**"，并在 `:152` 解释"洞源在 gap35→36 边缘"。
   - JSON：`StackFrontBack.hole_ratio_argmax_transition = 38`，实测 `T39/T38 = 1.4930`（gap **38**→39）。同表 N16 "@g38"、N15 "@g37" 与 JSON 一致 —— **只有 Stack 这一行的 gap 索引错了 3 格**（数值 1.493 正确）。

3. **README §3 "MrPro m36–39" 与 §5.2**：README §3 构造式列 `m_q=q(q+1)/306` 正确；§5.2 指出 UNIFIED:41 的 "0.51–0.89" 中 0.5098 实为 **m35**（`m_36=0.5948`）。我复算 `m35=0.509804`、`m36=0.594771` ✓ —— **README 的更正正确**；UNIFIED:41 是槽位笔误。

4. **"MrUni（全表÷4）"不实**（任务描述与旧 UNIFIED 文）：我 bit-exact 确认 MrUni = 过渡段线性斜坡 `m_j=(j−23)/17`（j=24..39），段外与 MrPro 逐位相同。README §3、§5.3 的更正**正确**。[已验证]

5. **NEXT_DERIVATION §3 表 "MrPro `Δ_q ∝ (2q+1)`"**（`NEXT_DERIVATION_KKT_PROBLEM.md:85`）与同文档 `:29` 的 `ε_j=2(1+j−dl)/((1+n)n)` 及实测表不符。实测 `Δ_g = 2(g−22)/306`（`g=23..39`）。归一化后 `(2q+1)` 序列与实测逐项偏离（例：首项实测 0.006536 vs `(2q+1)` 0.003460）。**拟合时以实测表为准**。

6. **INTEGRATION §1 vs §R4（Σm 是否守恒）**
   - `INTEGRATION_20260910.md:17`："总压缩预算 `Σ_j m_j = log S / log S`（即 **Σm 固定于过渡段**）" —— 这句本身量纲/表达式可疑（`logS/logS ≡ 1`），且语义上把 Σm 当守恒量。
   - 同文档 `:30` R4 + `NEXT_DERIVATION §4.5`（`:107`）明确："**Σm（质心）不是守恒量，是自由决策变量**"。
   - G1 表佐证 R4：`Σm` 在 30 张表上从 25.2069（HighGapToLong）到 34.1789（P2）**连续分布**，无守恒值。[已验证]
   - ⇒ 以 R4 为准；`INTEGRATION :17` 的括号注应删。

7. **INTEGRATION §7 FLAG-6 期望落空**："16-DOF 坐标：sol16 Helmert 保序构造 vs sol18 槽坐标加约束…两式互换性 **G1 表里已可逐位验证**"。实际 G1 JSON **不含任何 Helmert/η 坐标数组**（`methods` 只有 `nu/m/T/D/r/gap` 槽坐标）。该期望**未被本交付物满足**。

8. **GROUND_README §5.5（K128_S4 回执 sha 不可用于 Qwen3B）**：JSON `reconciliation` 已记为 NOTE，我确认该项状态为 `NOTE` 而非 MATCH/MISMATCH，与 README 描述一致。

9. **任务给的红线（静态几何代理不得入 F）与 G1 表的自我保护一致**：GROUND_README §2 明确"洞/粗糙度/平滑性只作为**被测量**记录，不据其排序方法或推断能力"，JSON 中无任何能力断言字段。[符合红线]

---

## 8. 死路登记（**绝不能再试的东西**，含失败原因）

| 机制 | 为什么死 | 出处 |
|---|---|---|
| **把 `endpoint_delta_m.fast_band_bitwise_equal_native` 当真值用** | 恒为 `false`（形状不匹配假阴性），零信息 | §6.1（我重算证明）|
| **用 BUDGET §3 的 max 洞 `1.86 / 1.76 / 1.80` 做任何排序或回归** | 不可复现。按 README §1-2 定义（ρ=T_{g+1}/T_g）重算为 **1.4930/1.4608/1.4757**。我另行穷举：`4^Δmax`(λ 比)=1.2031/1.1771/1.1892；max 2-槽跨度=2.1791/2.1122/2.1528；max 3-槽跨度=3.1248/3.0232/3.1044；`ρ/ρ0`=1.2031/1.1771/1.1892；末 gap ρ=1.2760/1.2409/1.2409 —— **均不命中**。仅在**非极值位置**出现零星巧合（T34/T32=1.8627、T30/T28=1.7581/1.7895），不构成定义 | README §5.1 + 我的穷举（§9 脚本）|
| **试图重跑 `rebuild_ground_truth_tables.py` 来复现对账** | 依赖 `results/nongeometric_screen_20260909/` 镜像，本地与全 git 历史均无 | §5.1 |
| **`E1_s28_reverse_matched` 的候选镜像式（"2·m28−m27"）** | README 给的候选值 **0.130719 无法由所给表达式产生**：`2·m28−m27 = 2(0.132270)−0.065359 = 0.199181`。0.130719 实际 = `2×m27`（`=0.130718894`）或 `m27+2(m29−m28)`。构造式**不可复原**，只有部署真值 `m28=0.132270051` 可用 | README §5.7 + 我复算 |
| **`E1_s29_plus_matched` 构造式** | 仅部署真值 `m29=0.094729390`（≠ 任何邻槽组合：`m30=0.183007`、`m28=0.098039`） | README §5.8 + 我复算 |
| **`E7_local_projection` / `GapCapped` / `HighGapToMid` 的构造迭代式** | 只有部署张量真值，构造参数（投影迭代式、帽 `c=1.2365e-5` 的施加方式）未本地复原 | README §5.9 |
| **把 `gap_j` 当成两个"槽内"量** | `gap_j` 是 j 与 j+1 之间共 63 个（`len=63`），不是 64 | JSON `definitions` |
| **把 `r_j_native` 当标量** | 它是 64 长向量 `W·ω_j/(2π)`（逐槽不同），"对所有方法相同"≠"对所有槽相同" | 脚本 `:85-86`；我实测 |
| **用"Σm 守恒"做推导** | Σm 是自由变量（25.21→34.18 连续分布） | INTEGRATION R4；我的表 |

---

## 9. 未解问题 / 需作者澄清

1. **BUDGET §3 的 `max 洞` 原始定义未知**：1.86/1.76/1.80 与任何我能构造的单-gap 或 k-gap 极值口径都不符。需要作者给出原式，或直接以重算值 1.4930/1.4608/1.4757 替换（README §6 已建议如此，我背书）。
2. **`rebuild_ground_truth_tables.py` 的镜像是否还在服务器**？若在，应把 `reference_tables.json` + 各 `contract.json`/`summary.json` 的关键子集（`values_float32` 数组 + gain）落进 `docs/research/`，否则**这张表永远不可复现**（连带 §3/§4 的两层复核也无法被他人重跑，只能信我的 digest）。
3. **README §4.1 的 "17 张 / 18 张" 计数**与交付 JSON 的 24/28 不符 —— 是口径不同（未写明判据）还是旧版本残留？需澄清。
4. **NTK 的 `Σtrans = 0.374079` 与 `hole_ratio_max_global` 在 g28 与 g41 并列**（我实测两处同为 1.2685，JSON 只记 argmax=41 for global / 28 for transition）。若下游用 argmax 唯一性做判据需注意并列。
5. **`E2_tail_more` 的 `Σtrans` 也 = 1.6022**（与 HighGapToLong 同值）—— 两者机制完全不同（尾部违例 vs 带外运输）。JSON 无字段区分这两种"1.6022 的成因"，拟合时若用 `Σtrans` 单一特征会把它们混为一谈。
6. **算子臂（E5×3、E6×2、E7_norm、E9、E10）只有 12 行分数、全为 100.00/64.4444**，与该屏基线一致 = 无信息量。它们**永远无法进入 F 的拟合**（无表）。
7. **`Panel 12 行` 与 `36 行` 的分数不可直接比较**（README §5.10 的计数口径说明）。`E2/E8/E4` 的 100.0/54.72 与 36 行臂的 87.22/78.13 不同屏。

---

## 10. 覆盖度

### 已读（本 digest 依据）
- `analysis/unify_20260910/tables/GROUND_README.md`（**全文 170 行**）
- `analysis/unify_20260910/tables/rebuild_ground_truth_tables.py`（**全文 676 行**）
- `analysis/unify_20260910/tables/ground_truth_tables.json`（321,398 B，**全部字段程序化读取**：38 条目的 `nu_j/m_j/T_j/D_j/r_j/gap_j/sum_*/hole_*/endpoint_*/changed_slots/panel_scores/formula_vs_deployed/sources`；126 项 `reconciliation`；10 条 `mismatches`）—— 未逐字节人工通读文本，但覆盖率 = 100% 字段
- 权威文档：`NEXT_DERIVATION_KKT_PROBLEM.md`（全文 147 行）、`STARTING_POINT_YARN_VS_MRPRO.md`（全文 116 行）、`INTEGRATION_20260910.md`（§1–§10 全文 132 行）
- `docs/research/BUDGET_ALLOCATION_MODEL_AND_CANDIDATES_20260910.md`（grep 命中行 + 上下文，非全文）
- `digests/digest_nongeo-code.md`、`digests/digest_thread-0909-pm.md`、`digests_codex/digest_failure-audits-1.md`（grep 命中行）

### 未读 / 跳过
- `ground_truth_tables.json` 中 30×6 个逐槽数组的**人工逐值通读**（改用独立程序复核，等价且更强）
- `analysis/unify_20260910/raw/` 全部 transcript（2.9MB，与 G1 无直接关系）
- `digests/` 其余 13 份、`digests_codex/` 其余 6 份
- `docs/research/` 其余 ~110 份
- `.agents/rope_unification_20260910/`（gitignored，未见）
- `results/nongeometric_screen_20260909/`（**不存在**，§5.1）
- `proposal_A/B.md`、`answers/D1–D4`、`tables/CANDIDATE_TABLES.*`（NEXT_DERIVATION §7 列为 ⏳ 进行中，未落盘）

### 我未做（边界）
- 未运行任何 GPU/训练；未改动仓库任何文件；未写 `~/.codex`
- 未对 `E7_local_projection`/`GapCapped`/`HighGapToMid`/`E1_*_matched` 的构造式做逆向**破解**（只验证了 README 给的那一个候选式不成立）
- 未验证 `panel_scores` 背后的 `summary.json`（镜像缺失，我**只能采用 JSON 里已存的分数**，其与文档锚点的一致性由 JSON 自己的 126 项对账保证，我未独立重放）

### 我落盘的临时脚本（不在仓库内，供复核者按图索骥）
`/tmp/gt_check/run.py`（脚本改写版）、`/tmp/gt_check/verify.py`（第一层重算）、`/tmp/gt_check/handcheck.py`（第二层闭式核对）、`/tmp/gt_check/verify2.py`（水床/端点/洞）
