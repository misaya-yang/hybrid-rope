# R10 — 远距离能力侧（L_far）的经验基础汇编

任务定位：为 KKT 目标泛函 F = L_near + L_far 的 **far 侧** 提供经验基础。本文只做 **材料挖掘与组织**，不做推导。
红线遵循：静态几何代理量（Σcos 首根、碰撞能、覆盖率、平滑度、有效秩、Gram、能量）**不得**作为 F 的分项或选择子；Σm 不是守恒量；报告中 VICTORY/已闭合/已证明 类结论一律降级。
证据分级：[已验证] / [部分证据] / [假设] / [叙事-未验证]。所有结论均带文件路径 + 行/节号。

---

## 0. 阅读范围与出处可用性警告（必读）

本节先声明 **哪些数字在本机可独立复核、哪些只能引用文档**，以免下游把转述当一手证据。

### 0.1 缺失的原始数据目录（覆盖缺口）
- `results/nongeometric_screen_20260909/`（远程 `/root/autodl-tmp/nongeometric_screen_20260909` 的同步副本）**本机不存在**。`analysis/unify_20260910/digests/digest_panel-results.md:3` 明确该 digest 的仓库根是 `/Users/[REDACTED_AUTHOR].yanghejazfs.com.au/paper_project/hybrid-rope/`——**另一台机器**。
- `planned_controls/evidence_distances_20260910.json` **本机不存在**（digest_panel-results.md:56 全仓 find 无匹配）。
- 后果：36 行 panel 的**逐行原始 JSONL** 与距离联表在本机无法一手复核。
- 本机可用的**一手替代物**：
  - `docs/research/NONGEOMETRIC_MECHANISM_TRANSFER_20260910.md`（630 行，LOCAL，含 P2/LBS/HighGapToLong 全表与距离段）
  - `docs/research/rope_allocation_20260910/source_inputs/RoPE_Allocation_Theory_Questions_for_Pro_20260910.md`（36 行 panel 表 + 逐任务变化）
  - digest_panel-results.md 本身（注明其数字来自服务器文件，且其 §2.5 用本地 ruler.jsonl 复核了**得分部分**并声明全部一致）

### 0.2 距离数值的出处强度（分档）
| 距离对象 | 数值 | 本地一手出处 | 等级 |
|---|---|---|---|
| MK2 128K 末 query→目标首 token | 88,725 | NONGEOMETRIC_MECHANISM_TRANSFER_20260910.md:348 | [已验证-本地] |
| MK2 两条 128K 行正确答案数字 token 距离 | 117,487–117,493 与 95,676–95,682 | 同上 :443 | [已验证-本地] |
| single-key 两条行 | 51,403–51,409 与 29,024–29,030 | 同上 :444 | [已验证-本地] |
| multiquery row 0 | 混 70 – 120,933 | 同上 :445 | [已验证-本地]（但见 §4.3 警告） |
| VT max/min | 105,880 / 20,563 | `ROPE_GLM_6PRO_REVIEW_AND_VALIDATION_20260910.md:33`（GLM 复核稿转述服务器文件） | [部分证据] |
| vt_131072_1 = 127K 行 | 127,xxx | **无本地任何文字出处** | [叙事-未验证] |
| FWE 行距离 | — | **定义无效**，见 §4.4 | 作废 |

### 0.3 本文主要引用文件清单
- `analysis/unify_20260910/NEXT_DERIVATION_KKT_PROBLEM.md`（权威问题模板，146 行）
- `analysis/unify_20260910/STARTING_POINT_YARN_VS_MRPRO.md`（权威起点 F1–F9，116 行）
- `analysis/unify_20260910/INTEGRATION_20260910.md`（主集成，131 行）
- `analysis/unify_20260910/digests/digest_panel-results.md`（36 行 panel 密度最高来源，220 行）
- `docs/research/NONGEOMETRIC_MECHANISM_TRANSFER_20260910.md`（LOCAL 一手，630 行）
- `docs/research/ROPE_ALLOCATION_SUBSPACE_DERIVATION_20260910.md`（U 度量与反例原文）
- `docs/research/rope_allocation_20260910/source_inputs/RoPE_Allocation_Theory_Questions_for_Pro_20260910.md`
- `docs/research/ROPE_BM_128K_DIAGNOSIS_20260908.md`、`ROPE_BM_CROSS_CACHE_20260908.md`、`ROPE_BM_TRANSFER_RESULT_20260908.md`
- `docs/research/ROPE_QWEN7_BM_TRANSFER_20260908.md`、`ROPE_OLMO_BM_RESULT_20260908.md`、`ROPE_OLMO_BM_EXTRA_RULER_RESULT_20260908.md`、`ROPE_OLMO_BM_FIVE_QA_RESULT_20260908.md`、`ROPE_BM_NLL_CROSS_MODEL_20260908.md`
- `docs/research/ROPE_BM_SELECTIVE_GAIN_20260908.md`、`ROPE_BM_SCALE_CAP_20260908.md`、`ROPE_MRPRO_BM_CONSTRUCTION_ANALYSIS_20260908.md`、`ROPE_MRPRO_TRANSITION_REVIEW_20260908.md`、`ROPE_LOCAL_FAILURE_SYNTHESIS_20260908.md`
- `docs/research/TWO_CORE_CONTINUATION_RESULTS_20260910.md`、`docs/research/GLM review — ROPE_GLM_6PRO_REVIEW_AND_VALIDATION_20260910.md`

---

## 1. L_far 必须复现的现象清单（核心交付）

以下 10 条是 **任何候选 F 的 far 侧建模都不得违反的经验事实**。每条给出：现象、出处、等级、对 F 的约束。

### P1 【分配敏感行 vs 饱和行】——far 侧损失不是全任务均匀的
36 行 panel = 6 任务 ×（32K 2 行 + 128K 4 行）。基线 MrPro(gain .1) = 32K **87.2222** / 128K **78.1250**（score_sum 29.2167；三来源一致，digest_panel-results.md §2.1 与 §2.3）。
**分配敏感的代表性证据**（`E1_s28_less`）：32K 87.2222（持平）/ 128K **83.3333**（+5.2083），**2W/0L**，且**只改两行**：
- `niah_multikey_2_131072_2`：0 → 1
- `niah_multiquery_131072_3`：.75 → 1

**饱和行的代表**：niah_single_2 在 Qwen3B / Qwen7B 的 32K 与 128K、MrPro 与 BM 两臂上**全部为 1.0**（`ROPE_QWEN7_BM_TRANSFER_20260908.md` JSON `baseline_extreme_cells` 把 3 个 niah 任务在两个长度上都标为 extreme cell）。
**对 F 的约束**：L_far 必须能被少数行的**离散跳变**驱动；不能写成对所有任务平均的连续损失。这与 F7/F8（softmax 竞争 σ(D)、只需把个别读取推过阈值）一致（STARTING_POINT_YARN_VS_MRPRO.md §6 [部分证据]）。

### P2 【同增益对照揭示"分配 vs 增益"分离】——BM@1 的反例
| 方法 | 32K | 128K | 说明 |
|---|---:|---:|---|
| MrPro@gain .1 | 87.2222 | 78.1250 | 主基线 |
| BM@gain .1 | 91.6667 | 70.8333 | 32K +4.44pp、128K **−7.29pp** |
| BM@gain 1 | 89.5833 | 58.8194 | 长端比 BM@.1 再 **−12.0139**，而 NLL 反而改善 0.023–0.035 |

出处：digest_panel-results.md §2.3 增益网格；`ROPE_BM_TRANSFER_RESULT_20260908.md`（32K 87.22 vs 91.67，2W/0L/10T；128K 78.12 vs 70.83，4W/4L/16T）。
**对 F 的约束**：**长的 NLL / 语言建模损失与远距任务能力可以反向**。L_far 不能等同于 NLL 项；若 F 的 far 侧写成 perplexity，会给出错误符号。

### P3 【几何更优但 128K 更差】——Smooth_MrBudget 反例（红线 1 原始证据）
- Smooth_MrBudget：32K 87.2222（持平）/ 128K **68.3333**（vs MrPro 78.1250，**−9.7917**）。
- 同一材料中其**静态几何量更优**：结构表 roughness MrPro .013071896 vs Smooth .004886399；slots-in-32K–128K 同为 4（source_inputs/RoPE_Allocation_Theory_Questions_for_Pro_20260910.md 结构表）。
- 逐任务：long MK2 .75→.25、QA .5→.25（退化），MQ .9375→1、VT .75→.85（个别改善）——同上文件。
- **U 度量反向**：U(ν)=E_{d~p_far}‖Nx_ν(d)‖²=tr(N G_far(ν))，Smooth MrBudget **.0494352** < MrPro **.235928**（更小=更优），未加权局部畸变 .000159185 < .000230655（ROPE_ALLOCATION_SUBSPACE_DERIVATION_20260910.md:166,170 附近的 CPU 表）。
- 原文自述（同上 :192–194）：「U and unweighted local distortion are **not a sufficient selector**. Smooth MrBudget improves both quantities relative to MrPro in this calculation, yet its existing 128K development score is worse.」
- Q/K 加权版 U_H = tr(N H N G_far(ν)) 在 36 层、cutoff 1e−6/1e−8/1e−10 下 Smooth 仍全面更低（同上 :258–275）——即加权也救不回反例。
等级：[已验证-本地 CPU 计算]。**对 F 的约束**：任何以"更小失真/更小 U"为选择子的 F 直接被此反例否。

### P4 【U 几乎相同但输出不同】——E1_s28_less 反例
- E1 s28 less：U = **.235928**，与 MrPro **完全相同**（ROPE_ALLOCATION_SUBSPACE_DERIVATION_20260910.md:170 与 MrPro 行）。
- 但 128K 任务分 **83.3333 vs 78.1250**（+5.2083，2W/0L）。
- 原文（同上 :194–195）：「E1 s28 has essentially the same U as MrPro despite different task outputs.」
等级：[已验证-本地 CPU + panel 转述]。**对 F 的约束**：U 不是充分统计量；far 侧损失必须包含 U 之外的自由度（阶段/整定位置 → 见 P5）。

### P5 【"动哪个槽"比"动多少"更决定远距结果】
- **E1 只把 m28 从 .0980 降到 .0654**（其余槽不动），128K +5.2083（digest §2.3；NEXT_DERIVATION_KKT_PROBLEM.md §1.4 anchor：m28 .098→.065 得 32K 持平 / 128K **+5.2**）。
- 相反极端：m28 .098→.294 得 32K **−22.6**（同 anchor）。
- **LBS**：把 slots 36–39 改为 .6252/.7295/.8465/.9799（D = 77,954 / 90,082 / 105,953 / 127,473），128K 80.5556/**80.0694**（VT .75→.95；3W/3L）。
- GLM 复核稿 §1 修正(5)：**E1 s28_less 并未改动 slot 36–39 的 D**，却修好了被引用作"视界证据"的 89K 行 —— 直接打掉"必须延长 36–39 视界"的因果解读。
- **P2**：m30 = .8506、m31 = .9979、m32+ = 1.0、m28 = .0741；gap28 **+0.149445**、gap29 **+0.809123**（部署绝对 gap ln(13267/4468)=1.088）；而 high gaps 0–23 总改动仅 **+0.0007754**（max 0.0775%）。对照 HighGapToLong 的 high gaps 总改动 **+0.2158674**（约 280×）。128K 81.6667（+3.5417），短端 72.9167（**−14.3056 / −25.4167**）。
等级：[已验证-本地（NONGEOMETRIC）]。**对 F 的约束**：F 对 Δ 的梯度是**强非均匀、非凸、slot 定位敏感**的；linear/二次代理不成立（见 §5 死路）。

### P6 【三段结构有代码级证据，但"交点随 S 移动"必须有 S-依赖】
- MrPro vs YaRN 频率交叉表（STARTING_POINT_YARN_VS_MRPRO.md §4，F5）：Qwen 4× 槽39 YaRN 0.2941 vs MrPro 0.2916（MrPro **更慢**），槽24/28 MrPro 更接近原生。A/B 集：Qwen {24–37}/{38–39}；Llama3 16× {19–24}/{25–34}。
- η 行为（F4）：η_Y(t,S)=t/[S(1−t)+t] → S→∞ 时 η_Y→0（**饱和**）；η_M = m_q **不饱和**。
等级：[已验证-推导+代码对账]。**对 F 的约束**：K1（三段定理）要证的正是"交点为何随 S 移动"（STARTING_POINT §8 第 3 点）。

### P7 【端点不是零容差定理，而是强基线设计约束】
- I1：m_j=0（j≤23 高频恒等）；I2：m_j=1（j≥40 尾部精确 ÷S）。守恒表述：17 个过渡 gap 总和锁定 ln S（native 3.6697 + extra 1.3863 = 5.0560 nats）（NEXT_DERIVATION_KKT_PROBLEM.md §1.3）。
- GLM 复核稿 §1 修正(3)：端点行为是**强基线设计约束，不是零容差定理**；修正(4)：*「所有赢家朝一个方向搬预算」为假*——质心表 MrPro 29.333333/34.666667、E1 s28_less 29.300653/34.699347、LBS 29.560179/34.439821、**P2 34.178915/29.821085（方向相反）**。
- INTEGRATION R4：Σm 质心是**自由决策变量**，零和频移 ΔΣm 实测 −0.01826…+0.00185；「任何守恒论证必须先点名坐标」。
等级：[已验证-推导+面板]。**对 F 的约束**：端点可作约束但不得写成硬等式分裂；ΣΔ=ln S 的守恒必须点名"过渡 gap 总跨度为 ln S"这一坐标。

### P8 【反例：局部扰动更小 ≠ 任务更好】
- F3：Qwen 现配置 Σ(ν^M−ω)²/Σ(ν^Y−ω)² = 0.4841（MrPro 局部旋转导数平方扰动约为 YaRN 一半）[已验证-CPU]。
- **OLMo 反例**（F6）：MrPro 对 YaRN 的局部扰动同样降到 ~47.8%，但 72 行长端 **MrPro 2.78% < YaRN 6.94% ≪ BM 51.32%**（STARTING_POINT_YARN_VS_MRPRO.md §5）。
- 即：减小局部扰动**不自动**换来胜利；BM 在 OLMo 上以相反方向拿到大幅优势。
等级：[已验证-独立实验]。**对 F 的约束**：L_near 的候选载体 Σ_j(ν_j−ω_j)² **不得单独成 F**（F6 明文禁止）；须与 far 侧联合。

### P9 【attention ≠ generation：覆盖更全但生成更差】
- `TWO_CORE_CONTINUATION_RESULTS_20260910.md:134`：prose16 中完整目标 Record 覆盖的层/head 单元数（共 72）均值 **29.75 → 67.125**，但答案仍由 **7/8 降为 6/8**。具体 007 完整目标覆盖 28/72 → 67/72，输出从正确 `bdqdrgdhzpn` 变为错误 `bdtfdetytmn`。原文：「这是更完整字面保留与更差生成并存的实测，**不能把覆盖率当充分能力指标**」。
- 交叉缓存对照（`ROPE_BM_CROSS_CACHE_20260908.md:44–49`）：4 格表 前缀形成×读取：
  - MrPro/MrPro = 3954314 正确，100% 召回
  - MrPro/BM = 3954314 正确，100%
  - BM/BM = 9289114 错误，20%，` ZOFPD.` 后 EOS
  - BM/MrPro = 9289114 错误，20%
  结论：「两例的成败/召回均**跟随前缀来源**」——预填充形成的状态（而非 decode 期读取表）控制这两例。
- VT/MrPro 来源换 BM 读取仍列出全部 5 变量，但后续说明变成 `However, it seems there is a mistake in the text.` → **官方召回保持 ≠ 完整输出一致**（同文件 :51–54）。
- 覆盖率类量属红线静态代理，**不得入 F**。等级：[已验证-独立实验]。**对 F 的约束**：L_far 的对象是**带符号判定裕度**（NONGEOMETRIC:104–108），须分别计前缀形成与前缀读出两项贡献。

### P10 【PHASE 差大本身不是失败的充分条件】
- `ROPE_BM_128K_DIAGNOSIS_20260908.md`：MK 案例中正确 multikey record 起于 token 35173，lag 95697，BM vs MrPro 最大相位差 ≈ **31.25 rad**（槽 28）；VT 起于 token 3986，lag 127052，最大 ≈ **41.49 rad**。
- 原文：「**P 仍能满分，说明相位差大本身不是失败的充分条件**」；「不能将最大相位差的槽 28 自动称最敏感槽」。
- 竞争 key 单点编辑：替换易混记录 key 后 MrPro 仍正确（3954314），BM 从 9289114 变成另一个错误数 4068207 —— 说明 BM 的失败是**绑定错误**而非"找不到"。
等级：[已验证-独立实验]。**对 F 的约束**：L_far 不能用"最大相位差/最大偏移量"这类**极值散度**作分项（红线）。

### P11 【远距失败的具体形态：多键误绑 + 非响应】
- `ROPE_BM_TRANSFER_RESULT_20260908.md` §原始答案复核：真实 multi-key 误绑（bizarre-inhabitant → 3954314，BM 输出属 bizarre-slime 的 9289114）；QA 答案 `P⊆NP⊆PP⊆PSPACE` 属**非响应**（未作答），状态 NO_LONG_GAIN。
- 逐任务 128K：MrPro {100, 75, 93.75, 75, 75, 50} vs BM {100, 50, 100, 75, 75, 25}。
- OLMo 独立任务（`ROPE_OLMO_BM_EXTRA_RULER_RESULT_20260908.md`，350 行 / 16K / 50 per task）：MrPro→BM：niah_single_1 .20→.90、niah_single_3 .00→.50、niah_multikey_1 .12→.60、**niah_multikey_3 .00→.00（两臂均未解决）**、niah_multivalue .05→.605、cwe .006→.052、qa_2 .12→.26；macro .0709→.4167（+34.59pp），156W/9L。
等级：[已验证-独立实验]。**对 F 的约束**：L_far 至少要有两个可分辨的失败模式自由度——(a) 多键竞争下的**错误绑定**，(b) 目标存在但不被读出的**非响应**。单一标量无法同时坍缩这两种。

---

## 2. 逐模型 / 逐长度敏感性总表（L_far 的定标数据）

### 2.1 Qwen2.5-3B-Instruct（W=32768, L=131072, S=4, θ=1e6, K=64）
| 方法 | 32K | 128K | 短端变化 | 长端变化 | 出处 |
|---|---:|---:|---|---|---|
| MrPro@gain.1（基线） | 87.2222 | 78.1250 | — | — | digest §2.1 |
| BM@gain.1 | 91.6667 | 70.8333 | +4.4445 | −7.2917 | BM_TRANSFER |
| BM@gain.074 | 100.0 | 70.0 | — | — | digest §2.3 |
| MrPro@gain.074 | 98.3333 | 75.3472 | — | — | digest §2.3 |
| E1_s28_less | 87.2222 | **83.3333** | 0 | **+5.2083** | digest §2.3 |
| FullLagP2_Transfer3B（P2） | 72.9167 | **81.6667** | **−14.3056 / −25.4167** | **+3.5417 / +6.3194** | NONGEOMETRIC / digest |
| LongBridgeSlower（LBS） | 80.5556 | 80.0694 | — | +1.9444 | NONGEOMETRIC |
| LongBridgeFaster | 87.2222 | 73.9583 | — | −4.1667（仅 QA row2 1→0） | NONGEOMETRIC |
| Smooth_MrBudget | 87.2222 | 68.3333 | — | −9.7917 | digest §2.3 |
| HighGapToLong | 70.1389 | 67.3611 | — | **0 improved / 7 worsened** | NONGEOMETRIC |
| MrUni | 64.5833 | 73.3333 | — | −4.7917 | digest §2.3 |
| E7_local_projection | 90.0 | 68.6111 | — | — | digest §2.3 |
| E2_tail_more（12 行子集） | 100 | 54.7222 | — | — | digest §2.3 |
| E8_zero51（12 行子集） | 100 | 50.5556 | — | — | digest §2.3 |
| 12 行打平集：E10、E1_s28_reverse_matched、E1_s29_plus_matched、E4、E5、E6、E7_norm_matched、E9 | — | — | — | 与基线同分 | digest §2.3 |
| E1_s28_less（12 行子集口径） | 100 | **64.7222** | 0 | 0 | digest §2.3 |

P2 的 128K 逐任务：1 / .5 / 1 / .9 / .75 / .75；对照 same-gain MrPro：1 / .75 / .9375 / .75 / .8333 / .25（NONGEOMETRIC §Positive-case / digest）。
LBS 的 128K 逐任务变化：VT .75→.95（升），FWE .75→.6667（降），3W/3L。
HighGapToLong 的 128K 逐任务：single-key 1、multikey .5、multiquery .875、VT .75、FWE .6667、QA .25。

### 2.2 Qwen2.5-7B-Instruct（`ROPE_QWEN7_BM_TRANSFER_20260908.md` + JSON，18 行）
- 32K：BM **80.00** vs MrPro **83.33**；128K：BM **71.11** vs MrPro **84.44**；**0W/3L/15T**。
- JSON 逐任务：两臂在 32K 与 128K 上 **niah_single_2 / niah_multikey_2 / niah_multiquery 全部 1.0**（饱和）。
- MrPro 128K：vt .9、fwe .6667、qa_1 .5；BM 128K：vt .6、fwe .6667、**qa_1 0.0**。
等级：[已验证]。**对 F 的约束**：7B 上 BM 在 QA 上归零，是"非响应"模式在更大模型上的复现。

### 2.3 OLMo-2-0425-1B-Instruct（W=4096, base 500000, l/h=14/32）
| 面板 | MrPro 32K/128K | BM 32K/128K | 备注 |
|---|---|---|---|
| dev 36 行 | 37.22 / 14.93 | **79.44 / 49.03** | BM 全面占优 |
| seed_replication 72 行 | 37.85 / **2.78** | 81.81 / **51.32** | **44W/0L/28T** |
| MrUni | 76.88 / 32.12 | — | 长端介于两者 |
| OfficialYaRN | 54.38 / 6.94 | — | 长端崩 |
| S8（scale=8）32K | MrPro .69 / BM 6.94 | — | 双双贴地 |
| S=8 4K | MrPro 4.17 / BM 23.89 | — | — |

出处：`ROPE_OLMO_BM_RESULT_20260908.md` + JSON。
- JSON 逐任务（seed_replication）**4K**：MrPro {single_2 .75, multikey_2 .25, multiquery .1875, vt .25, fwe .5833, qa_1 .25} → BM {1.0, 1.0, .875, .7, .5833, .75}。
- **16K**：MrPro {0,0,0,0,.1667,0} → BM {1.0, .375, .4375, .1, .4167, .75}。
- Native 24 行 4K 参考 .75625，qa_1 1.0（BM 把 QA 从 1.0 掉到 .75）。
- **独立任务**（EXTRA_RULER，350 行 16K）：见 P11。
- **自然文本迁移**（FIVE_QA，631 长输入配对）：BM **25.44%** vs MrPro **21.62%**（+3.82pp，95% CI [+1.32,+6.29]）；逐任务 hotpot +4.91、2wiki +3.88、qasper +5.11、narrativeqa +1.96、multifieldqa +3.23。短输入 147 行 +2.00pp，CI [−3.56,+7.47]（**优势不明确**）。
等级：[已验证]。**对 F 的约束**：同一 (L_near, L_far) 泛函的**最优解依模型而变**——OLMo 上"更激进的慢频压缩"（BM）长端优于 MrPro，与 Qwen 上 BM 长端劣于 MrPro **符号相反**。F 必须显式依赖模型/数据（至少是窗口比 W/L 与 θ）。

### 2.4 NLL 侧：长端 NLL 不能解释长端任务退化
`ROPE_BM_NLL_CROSS_MODEL_20260908.md`（30 行）：
- OLMo 4K/8K/16K：Native 2.83505/—/—；MrPro 3.21062/3.25655/3.68798；BM 2.95506/2.95486/**2.86206**（16K BM 16/16 文档更低，mean diff −0.82592，95% CI [−1.08299, −0.62445]；4K BM 仍高于 Native +0.12001）。
- Qwen3B 8K/16K/32K：Native 2.27607/2.14317/2.03850；MrPro 2.32333/2.18886/2.08899；BM 2.32581/2.19311/2.08789（32K diff **−0.00111 ≈ 0**）。
- 关键结论（原文）：「该结果**不支持**把 128K 的任务退化解释成原生范围内普遍语言模型损坏。」
- 88K/128K 连续文本（NONGEOMETRIC:181–185）：slot 28 差 −0.00155/−0.00121 on Proof-Pile，+0.00359/+0.00420 on PG19；slot 29 在 +0.00304 内；样本为 tail-512 NLL，非全文档 PPL，材料自述**不构成等价性证明**。
等级：[已验证]。**对 F 的约束**：L_far ≠ 长文 PPL；但在 far 侧仍可保留一个弱约束——**分配不得使原生窗内 NLL 显著变差**（窗内不失忆的硬要求）。

---

## 3. 距离-桶证据（distance-bucket）与其适用边界

### 3.1 可达的距离观测
- Mk2 128K 末 query → 目标值首 token = **88,725**（NONGEOMETRIC:348），即通常引用的 **89K** 行。
- 两条 128K multikey 行正确答案 digit token 距离 = **117,487–117,493**、**95,676–95,682**（→ 117K / 96K）。
- single-key 两行 = **51,403–51,409**、**29,024–29,030**。
- multiquery row 0 混 **70 – 120,933**（→ 106K/117K/… 的混淆来源）。
- VT：min 20,563 / max 105,880 [部分证据]。
- vt_131072_1 的 127K 行：**无本地出处**。

### 3.2 与分配结果的联表（转述自 digest §5.4，服务器数据）
digest_panel-results.md §5.4 给出 128K 的 distance×method 表（89K/106K/127K/96K/117K 行）。本机**无法一手复核**（见 §0.1）。

### 3.3 必须记录的证伪与警告
- **GLM 复核稿 §证据距离复核**结论：「**不能据此将 75–112K 宣称为已验证的危险带**」。
- **FWE 距离定义无效**：答案词 `cyuvqn` 在全文中出现 **10785** 次、`lobxbq` 出现 **4793** 次——"答案位置"不是良定义的单点，故 FWE 行**不得**进入任何距离-桶统计。
- **MQ 行混合距离**（70–120,933）：同一次评分聚合了差异极大的物理关系长度，不能作为单一距离桶样本。
- Mk2 的 89K/117K/96K 是**字面目标 token 几何**，材料自述「not measured internal read paths」（NONGEOMETRIC:446）。
等级：结构事实 [已验证]；"危险带" 结论 [叙事-未验证] 且已被本地材料明确限制。

---

## 4. 迁移差异（模型间不可直接搬运）

| 对象 | 事实 | 出处 | 等级 |
|---|---|---|---|
| OLMo 规则迁移 | 源规则冻结为"穿过 MrPro 过渡的 **5/17** 分数位置"，四舍五入到最近目标槽，用其前驱指数替换；bounds 14/32 → slot 19；其他频率与 gain 不动；**无任何目标结果参与选择/调参** | NONGEOMETRIC §Frozen transfer | [已验证-协议] |
| Qwen7B / OLMo 的参数匹配 | 两模型的 native 网格与 MrPro bounds 与 3B 一致，故 slot-28 表**精确迁移** | NONGEOMETRIC §Frozen transfer | [已验证] |
| Qwen7B 特有约束 | 必须保留基线的 **4096-token positionwise MLP** 实现 | 同上 | [已验证-协议] |
| 结论口径 | 「If fixed transfer fails but recalibration succeeds, the claim is about an **adaptation procedure**; it is not a universally better frequency table.」 | NONGEOMETRIC §Frozen transfer | [已验证-方法学] |
| OLMo 局部量一致性 | MrPro 对 YaRN 的局部旋转导数扰动在 OLMo 上也 ~47.8%（与 Qwen 的 0.4841 同量级），但长端排序 **反号** | STARTING_POINT §5（F6） | [已验证] |
| OLMo BM 局部粗糙度 | OLMo l14/h32/N18：roughness R 0.01169590643 → 0.001754385965；terminal increment ×3/(N+2)=15% | `ROPE_MRPRO_BM_PROTOCOL_20260908.md` | [已验证-协议] |

**对 F 的约束**：F 中至少要有两个可随模型切换的标签变量：窗口比（W/L，Qwen 4× vs OLMo 32×）与 θ/base。把"更激进的压缩更好"当普遍规律是错的（OLMo 反例）。

---

## 5. 已证伪 / 已失败机制（死路登记，不得再试）

| # | 机制 | 失败表现 | 失败原因 | 出处 | 等级 |
|---|---|---|---|---|---|
| D1 | Smooth_MrBudget（几何/失真更优） | 128K 68.3333 vs MrPro 78.1250；U .0494352 更小仍更差 | **U / 局部畸变不是充分选择子** | SUBSPACE_DERIVATION:166,192–194 | [已验证] |
| D2 | HighGapToLong（把预算搬向低频） | 32K 70.1389 / 128K 67.3611，**0 improved / 7 worsened**；high gaps 改动 +.2158674（P2 的 280×） | 高频段过度改动 → 近端崩塌 | NONGEOMETRIC / digest §2.3 | [已验证] |
| D3 | E2_tail_more / E8_zero51 | 12 行子集 100 / 54.7222 与 100 / 50.5556（长端比基线 64.4444 更差） | 尾部一刀切 | digest §2.3 | [已验证] |
| D4 | MrUni（均匀插值） | 32K 64.5833（远低于 87.2）/ 128K 73.3333 | 均匀 ≠ 最优；中前段扰动过大 | digest §2.3；BM_CONSTRUCTION §5 | [已验证] |
| D5 | SelectiveGain（选择性增益） | 4K/16K 80.6944/44.7917；把 16K 密集多键检索 25%→0%、VT 15%→0% | 增益调整破坏远距检索（**相位与增益不可互换**） | `ROPE_BM_SELECTIVE_GAIN_20260908.md` | [已验证] |
| D6 | UniformMatchedGain | 4K/16K 79.4444/44.2361 vs BM 79.4444/49.0278 | 同上 | 同上 | [已验证] |
| D7 | BMCappedS4 / BMFreq8Gain4 | 4K/32K 73.4028/0% 与 29.0972/5.5556% | 均劣于原 S8 BM 23.8889/6.9444 | `ROPE_BM_SCALE_CAP_20260908.md` | [已验证] |
| D8 | E7_local_projection | 32K 90.0 / 128K 68.6111 | BF16 绝对坐标旋转误差 1.28074e−5 vs 局部线性预测 7.93668e−8（~160×），**误差由有限精度旋转主导** | NONGEOMETRIC §Numerical approximation audit | [已验证] |
| D9 | E1_s28_reverse_matched / E1_s29_plus_matched | 12 行子集与基线同分（0 变化） | 反向/邻槽对照**无特异性**：单一槽的等量反向改动不产生对应反向效果 | digest §2.3 | [已验证] |
| D10 | "延迟压缩" 候选 m_q\* = max{0, 1−(N−q)c} | 解析上等同于**收窄过渡带的 MrUni** | 构造上与已失败的 MrUni 同族 | `ROPE_MRPRO_BM_CONSTRUCTION_ANALYSIS_20260908.md` §5 | [已验证-解析] |
| D11 | 静态几何代理选型（Σcos 首根、碰撞能、覆盖率、平滑度、有效秩、Gram、能量） | 根与能力排序**失序**：MrUni 82.2K > MrPro 80.3K 而 32K 64.6 ≪ 87.2；E2/P2 同根反向 | 根 = 诊断量，**禁入 F**（红线 1 / INTEGRATION R1） | STARTING_POINT §6；INTEGRATION R1 | [已验证-双源] |
| D12 | YaRN 的"递减 vs MrPro 递增"机制解释 | 两者 **都凸都递增**（m_Y′>0, m_Y″>0） | 论文口径在 HF 实现下**无表格载体**；mx|Δm|=0.034 | STARTING_POINT §1（F1/F2）+ digest_mrrope-evq §D8 | [已验证] |
| D13 | ψ 投影假说（用单一投影解释过渡表差异） | a\* = 2.1477690111869907，只解释差异能量的 **13.1846%**；槽 30/31 移动方向与 P2 相反 | 投影基不完备 | `ROPE_MRPRO_TRANSITION_REVIEW_20260908.md` §C1 | [已验证] |
| D14 | 覆盖率/完整保留作为能力指标 | 覆盖 29.75→67.125 而答案 7/8→6/8 | attention ≠ generation | TWO_CORE:134 | [已验证] |
| D15 | 纯 decode 表修复（只换读取表） | BM/MrPro = BM/BM = 错误答案，成败跟随**前缀来源** | 预填充状态主导 | BM_CROSS_CACHE:44–59 | [已验证] |
| D16 | 覆盖率型分组贪心（KeyDiff 均分句段） | prose256：完整目标 Record/key/value 在所有层/head 均**未保留**（全记录完整保留 ~4.00%） | 未纠正记录选择缺失 | TWO_CORE:136 | [已验证，但材料自述只否定该具体分组] |
| D17 | "窗内训练白送外推" | ZC vs ZF p=0.0076 | 已证伪 | MEMORY（round12 教训） | [已验证-历史] |
| D18 | 报告中的 VICTORY / 已闭合 / 已证明 类结论 | — | **红线：一律降级处理** | INTEGRATION R5 / 任务红线段 | 规则 |

---

## 6. 矛盾与约定不一致（供下游裁决）

1. **U 度量 vs 任务结果**：U 与未加权局部畸变在 Smooth_MrBudget 上**双优**，任务却更差；加权版 U_H 也救不回。→ 与"最小化远距表示损失"的直觉**直接矛盾**。（SUBSPACE_DERIVATION:192–194 vs panel §2.3）
2. **OLMo 与 Qwen 的符号冲突**：BM 在 OLMo 长端优于 MrPro（51.32 vs 2.78），在 Qwen3B 长端劣于 MrPro（70.83 vs 78.12），在 Qwen7B 同样劣（71.11 vs 84.44）。→ "更激进慢频压缩更好"不是普遍规律。
3. **"所有赢家朝一个方向搬预算"**：GLM 复核稿 §1 修正(4) 判为**假**，质心表 P2 = 34.178915/29.821085 方向相反。
4. **"必须延长 slot 36–39 视界"**：GLM 复核稿 §1 修正(2) 指出 D_j **不是未见相位边界**（native 36–39 在 W 内已转约 2.199/1.772/1.428/1.151 圈）；修正(5) 指出 E1 s28_less **根本没动 36–39 的 D** 却修好了 89K 行。
5. **FWE 距离定义**：答案词在全文重复 10785 / 4793 次 → 距离不可定义，与"FWE 行可用于距离桶"冲突（GLM 复核稿 §证据距离复核）。
6. **paper 表格与论文附录**：MrRoPE 论文表格所用确切 commit **未核对**（STARTING_POINT §1 末），故 F1/F2 对账以公开仓库 + HF 实现为准。
7. **本机 vs 服务器数据**：36 行 panel 与 evidence_distances 的一手文件在**另一台机器**；本地只有文档转述。任何以"89K/106K/127K 距离联表"为硬约束的下游推导必须先复算。
8. **51.32% 口径**：`ROPE_LOCAL_FAILURE_SYNTHESIS_20260908.md` §6 明确 —— **81.2% 是 OLMo 曲线拟合的平方残差占比，不是 Jacobian 敏感度**，不得混用。

---

## 7. 开放问题

1. **89K 行到底由什么修好？** E1 s28_less 只动 m28（.098→.065），未动 36–39 的 D，却修好了被引作视界证据的 89K 行；LBS 动了 36–39 也修好 VT。两条通路（早期槽 vs 尾部槽）**哪个是必要的**？
2. **距离桶能否成立？** 需要重新定义 FWE 距离（规避重复答案词）、拆开 MQ 的混合距离，才能谈 75–112K 危险带。
3. **P2 的巨大短端代价（−14.3/−25.4）** 是否可被更低增益或更窄的高频区吸收，从而使 F 的解落在 Pareto 前沿而非单点？
4. **前缀形成 vs 读出** 的贡献可加性：MK 案例 margin 分解为 −2.125 / −1.125 / −1.000 / +0.250（NONGEOMETRIC:134），交互项仅 +0.250 nat（BF16 分辨率量级）；这个"近似可加"是否在其它行成立？
5. **λ 坐标下的 far 侧对象**：F 的 far 侧应依赖 {η_j} 的分布（STARTING_POINT §8 第 2 点），但**没有材料给出 ∂(能力)/∂η 的实验形状**。
6. **U 的自由度缺口**：什么量能在 U 相同的 E1_s28_less 与 MrPro 之间区分（P4）？候选在材料中**未被测量**。
7. **27 vs 26 个方法目录**（digest §7）；`0446/0448/0449/0450/0451` 未运行；`0441` 试点缺行；A7 反驳未结（digest §7）。
8. **P2 在 32K 上的 −25.4167（同增益口径）** 的机制未解释：是高频段改动（总 +.0007754）还是 gap29 +.809123？

---

## 8. 覆盖说明

### 已读（本机一手）
- 权威文档三件：`NEXT_DERIVATION_KKT_PROBLEM.md`、`STARTING_POINT_YARN_VS_MRPRO.md`（全文 116 行）、`INTEGRATION_20260910.md`
- `digests/digest_panel-results.md`（220 行，含 §2.1–§7）
- `docs/research/NONGEOMETRIC_MECHANISM_TRANSFER_20260910.md`（第 60–200、340–352、438–448 行区间精读）
- `docs/research/ROPE_ALLOCATION_SUBSPACE_DERIVATION_20260910.md`（U 段 60–120、160–200、240–280）
- `docs/research/rope_allocation_20260910/source_inputs/RoPE_Allocation_Theory_Questions_for_Pro_20260910.md`（36 行 panel 表、结构表、E7 BF16 分解）
- `ROPE_BM_CROSS_CACHE_20260908.md`（全文）、`ROPE_BM_128K_DIAGNOSIS_20260908.md`、`ROPE_BM_TRANSFER_RESULT_20260908.md`、`ROPE_QWEN7_BM_TRANSFER_20260908.md`(+JSON)、`ROPE_OLMO_BM_RESULT_20260908.md`(+JSON)、`ROPE_OLMO_BM_EXTRA_RULER_RESULT_20260908.md`(+JSON)、`ROPE_OLMO_BM_FIVE_QA_RESULT_20260908.md`、`ROPE_BM_NLL_CROSS_MODEL_20260908.md`、`ROPE_BM_SELECTIVE_GAIN_20260908.md`、`ROPE_BM_SCALE_CAP_20260908.md`、`ROPE_MRPRO_BM_CONSTRUCTION_ANALYSIS_20260908.md`、`ROPE_MRPRO_TRANSITION_REVIEW_20260908.md`、`ROPE_LOCAL_FAILURE_SYNTHESIS_20260908.md`、`TWO_CORE_CONTINUATION_RESULTS_20260910.md`、`GLM review — ROPE_GLM_6PRO_REVIEW_AND_VALIDATION_20260910.md`、`ROPE_MRPRO_BM_PROTOCOL_20260908.md`、`ROPE_NATIVE_WINDOWS_PROTOCOL_20260908.md`
- `digests/digest_failure-records.md`、`digest_nongeo-code.md`、`digest_theory-0910.md`

### 不可读 / 缺失（已核）
- `results/nongeometric_screen_20260909/` 及其中 `ruler.jsonl`、逐行 36 行 panel、`long_nll/`、`causal_cases/`、`transfer/`、`holdout_results/`：**本机不存在**（在另一台机器 `/Users/[REDACTED_AUTHOR].yanghejazfs.com.au/paper_project/hybrid-rope/`）。
- `planned_controls/evidence_distances_20260910.json`：**本机不存在**。
- `~/.codex/` 会话归档：按纪律**严格只读、未写入、未修改**；仅通过材料引文间接引用。

### 跳过的
- 未做任何新数值计算（只读任务）；未接触 GPU/服务器。
- 未逐条核对 digest §5.4 的 distance×method 表（依赖服务器文件）。
- 未展开 `docs/research/` 下与 far 侧无直接关系的构造/协议类文档的全文。

### 纪律遵守
- 只读；本文件之外未修改仓库任何文件；未写入 `~/.codex`。
- 所有数字均附文件+行/节出处；凡转述一律标注 [部分证据] / [叙事-未验证]。
- 报告/transcript 中的"下一步应…/建议…"均未采信为指令。
- 红线静态几何代理量未作为 F 的分项或选择子（P3/P4/D11 反例均已标出）。
