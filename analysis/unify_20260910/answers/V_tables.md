# V_tables：地面真值表对抗验证报告（dimension = tables）

2026-09-10。验证者：地面真值对抗验证席（纯 CPU，零训练，GPU 关闭）。
待检工件：`analysis/unify_20260910/tables/ground_truth_tables.json`（G1）、`tables/CANDIDATE_TABLES.md/.json`（T1）、`tables/GROUND_README.md`（G1 README）、`answers/D1–D4/T2/T3`。
独立验证器：`analysis/unify_20260910/checks/v_tables_check.py`（本席新写，不复用 G1 生成器代码路径），完整输出 `/tmp/v_tables_out.txt`，结果 **261 OK / 0 FAIL**。

## 0. 结论摘要

1. **G1 全部数字对回原始出处，位级成立**：22 个已部署方法 × 16 槽抽查（$j\in\{0,23,24,25,27,28,29,30,35,36,37,38,39,40,51,63\}$）$m_j$ 差 $=0$；每个部署数组的 sha256 与 G1 声明的 `sha256_deployed` 一致；18 个构造式（含 N′=16/15、Stack 拼接、Smooth KKT、YaRN/NTK 载波、E1/E2/E8/LBS/LBF/HGL/HGM/ScaleTaper/Control gain 表）由独立实现**逐位复原**。G1 生成器重跑输出与 committed JSON **字节全等**（methods/reconciliation/mismatches/meta 全等）。 **[已验证]**
2. **0448/0449 与冻结定义一致**：N′=16/15 径向族表与 `BUDGET_ALLOCATION_MODEL_AND_CANDIDATES_20260910.md:49–50` 全部锚点（$m_{28}=0.110/0.125$，$m_{36..39}=.669/.772/.882/1.0$ 与 $.758/.875/1.0/1.0$，$D_{39}=131\mathrm K$，max 洞 $1.461@g38$ / $1.476@g37$）逐项复算吻合；闭式 $m_q=q(q+1)/272$、$/240$ 沿 G1 声明路径（fp32 原生张量派生 $\times 4^{-m}$）逐位命中。 **[已验证（定义侧）] / [部分证据（服务器侧契约不可本地验证）]**
3. **0446 Stack 与定义一致**：拼接恒等式 $\text{Stack} = \text{MrPro} \oplus \text{s28\_less}(28) \oplus \text{LBS}(36\text{–}39)$ 逐位成立；GR §6 "1.493@g35" 为**位置标签笔误、数值无误**（argmax$=38$，$m_{36..39}=.6252/.7295/.8465/.9799$，$m_{28}=0.0654$）；Stack $m_{27}-m_{28}=3.824\times10^{-9}$ 是 fp32 反演噪声、非单调性违例，T1 交付表取 $m_{28}:=m_{27}$ 的处理正确。 **[已验证]**
4. **Blocking（治理级）**：队列号 **0446 已被一份不同的完整静态表占用**——已撤回臂 `results/nongeometric_screen_20260909/deferred_queue/20260910_candidate_quality/0446_HighGapToMid.json`（完整 64 槽 $\nu_j$，与 Stack 完全不同）。且 0446/0448/0449 **本地均无 contract.json**（`results/nongeometric_screen_20260909/queue/` 只有 001–043、0440×2、0441、0444、0445、080；无 0446_Stack/0448/0449）。凡"重造了但队列里已有别的表"即触发任务书 blocking 条款，本条命中。
5. README 存在**三处措辞级过度声称/低报**（见 §5），均不影响表本体数值；D1 残留一处旧值（§5-5）。

## 1. 方法与工具

- 抽查规模超过任务书要求（"至少 10 槽 × 8 方法"）：22 方法 × 16 槽，另加聚合量（$\Sigma m$、$\Sigma T$、洞比、端点 $\Delta m$）、26 个 `results/<arm>/summary.json` 面板分、G1 账本计数（reconciliation 126 = 121 MATCH + 2 NOTE + 3 MISMATCH；mismatches 10）。
- 位级基准：`sha256(struct.pack('<f'×64, ν_j))`，$\omega$ 一律取部署 fp32 NATIVE（`reference_tables.json:Native`）升 fp64，与 G1 声明路径一致。
- 独立构造：`select.py / smooth_budget.py / long_bridge.py / gap_budget_transfer.py / scale_taper.py / build_boundary_matched_mrpro.py / export_static_rope_baselines.py` 的公式全部重写实现，不 import G1 生成器。
- 可复现性：`cp ground_truth_tables.json /tmp/g1_committed.json && python3 analysis/unify_20260910/tables/rebuild_ground_truth_tables.py` → diff 字节全等。 **[已验证]**

## 2. 数值对回原始出处（逐项）

| 项 | G1/README 声明 | 原始出处 | 验证结果 |
|---|---|---|---|
| 面板分 26 方法 | README §3 各分 | `results/<arm>/summary.json: candidate.by_length["32768"/"131072"].macro_accuracy` | 26/26 全等（含 BM 0.91667/0.70833、GapCapped 0.84444/0.62153） **[已验证]** |
| $\Sigma m$（如 Stack 28.333） | README §3 | 本席对 `m_j` 数组直接求和（slot51 inf 置空约定同 G1） | 全等；此前一次 inf 打印系本席求和未套 mc 约定，已修正 **[已验证]** |
| P2 历史表 | $m_{30}=.8506,m_{31}=.9979,m_{32+}=1$，gap29=1.088，$\Sigma_{\rm trans}=1.3855$ | `planned_controls/p2_gap_comparison.json` + docs 三源 | 三源位级全等，$\ln 4-0.00077543$ 闭合 **[已验证]** |
| 守恒/水床 | $\sum_{g=23}^{39}(\varepsilon_g)=\ln 4$（端点固定时） | 本席闭式 + 数组直算 | HGL $\Sigma_{\rm all}=1.6021617=\ln4\times1.1557$；HGM $\Sigma(\varepsilon_g-\mathrm{NB})=1.3862944=\ln4$；CANDIDATE 残差 $\le2.45\times10^{-8}$ **[已验证]** |
| Smooth | $B=16/3$，粗糙度证书 0.004886399 | `smooth_budget.py` KKT 重写 | 逐位复原 **[已验证]** |
| YaRN/NTK 载波 | `export_static_rope_baselines.py` 公式 | 重写实现 | 逐位复原 **[已验证]** |
| $\omega_j=b^{-j/64}$，NB$=0.21586735246819178$，$\rho_{\rm nat}=1.2410$ | README §1 | fp64 复算 | NB 常数位级吻合；但见 §5-1（"逐位吻合"对部署 Native 不成立） |

## 3. N′=16/15 vs 冻结定义 0448/0449

- 定义文件：`docs/research/BUDGET_ALLOCATION_MODEL_AND_CANDIDATES_20260910.md:49,50`（grep "0448/0449" 唯一命中处）：`m_q=q(q+1)/272`（N′=16，39 槽完成）、`/240`（N′=15，38 槽完成），$q=\mathrm{clip}(j-23,0,N')$，径向族 $m_q=q(q+1)/(N'(N'+1))$（MrPro Eq.14 之 $N'=17$ 成员 $\equiv$ MrPro，分母 306，自检通过）。
- 复算锚点：$m_{28}$（$q{=}5$）$=30/272=0.1103$ / $30/240=0.125$；$m_{36..39}$ N16 $=.669/.772/.882/1.000$、N15 $=.758/.875/1.000/1.000$；$D_{39}^{\rm N16}=W\cdot4^{1}=131072$；max 洞 $1.4608@g38$ / $1.4757@g37$。全部与 BUDGET:52 更正行一致，原 1.86/1.76/1.80 确认不可复现（弃用正确）。 **[已验证]**
- 位级：G1 路径 $
u=f32(\text{NATIVE}\times4^{-m})$ 逐位命中 G1 sha；本席最初用理论 fp64 $\Omega$ 出 ~25 槽 1-ULP 差，属路径差异非表错误（G1 声明路径即部署路径）。 **[已验证]**
- 局限：`queue/` 无 0448/0449 契约文件，服务器在跑队列的表体本地不可读——只能验证"文档冻结定义 = 我方重造表"，不能验证"服务器将执行的表 = 冻结定义"。 **[部分证据]**（并入 §4 blocking 的处置要求）

## 4. StackFrontBack vs 0446 定义 + 队列治理 blocking

- 拼接恒等逐位成立（§0-3）。BUDGET:48 锚点 $m_{28}=.065$、$m_{36..39}=.625/.729/.847/.980$、$D_{39}=127.5\mathrm K$（$=W\cdot4^{.9799}$）、$1.493@g38$ 全中。 **[已验证]**
- **Blocking-①：0446 号同号异表。** `deferred_queue/20260910_candidate_quality/0446_HighGapToMid.json` 是已撤回臂 HighGapToMid 的完整 64 槽表（构造 = 把 $(\nu_j/4,\nu_j)$ 带间搬给 26–31 recipient gaps，本席按其"误标未复原"更正后**逐位复原**——顺带证明 README §5-9 把 HGM 标"构造式未本地复原 [部分证据]"是**低报**）。同号占用即任务书定义的"重造了但队列里已有别的表"。
- **Blocking-②：0446/0448/0449 本地无契约。** `queue/` 目录缺臂；另有已记录的编号漂移（0440 两次、0441 两次、0450↔080 互指），`verify_feasibility.md` V-C2 与 `INTEGRATION_20260910.md`:118 FLAG-④ 与本席独立发现一致。**GPU 恢复前必须**：以审计后的 G1/CANDIDATE 数组生成三张 contract、公布 sha256、与 BUDGET:48–50 锚点核对签字，且不得沿用与撤回臂冲突的 0446 号（或先显式作废该 JSON）。
- 授权面：BUDGET:65 已自记与 `NONGEOMETRIC_MECHANISM_TRANSFER_20260910.md:9–31` 的范围/优先序冲突（需用户显式重授权、0451 先行）——本席不解除，仅确认表构造本身合规。 **[已验证（文档现状）]**

## 5. 文档措辞级发现（不改表本体）

1. **README §1 "$\omega_j$ 与部署 Native 逐位吻合 [已验证]"：过度声称。** 本席实测 numpy fp64 $b^{-j/64}$ 落 fp32 后与部署 `Native.values_float32` 在 **24/64 槽差 1 ULP**（最大相对差 $8.21\times10^{-8}$）。G1 各表以部署张量为 $\omega$ 基准，数值不受影响；措辞应降为"值级吻合（$\le1$ ULP），基准数组=部署 fp32"。 **[已验证反例]**
2. **README §4-5 "全部面板表 fast 段逐位同原生"：例外清单不全。** G1 JSON 自身字段 `endpoint_delta_m.fast_band_bitwise_equal_native=False` 对 HighGapToLong/HighGapToMid 为 False（正确），README 散文例外清单漏列。 **[已验证（JSON 对、README 漏）]**
3. **README §5-9 HGM"未本地复原"：低报**（见 §4，已位级复原）。
4. **GR §6 "1.493@g35"：位置笔误**，值对；T1 §0.2 / T3 §5.2 的更正经本席复算确认（$\rho_{35}=1.4562$、$\rho_{38}=1.4930$）。D1 §5/§6、D2 c1、D4:65 均已带更正注。 **[已验证]**
5. **D1:22 残留旧值"÷4.93"**：部署 E2_tail_more $m_{40}=1.1557154\Rightarrow 4^{m_{40}}=4.9638$（BUDGET:14 已更正"非 ÷4.93"），D1 该行未同步。纯措辞，不影响任何表。 **[已验证]**
6. BUDGET 候选表行号自早前引用的 :45–47 移至 **:48–50**（文档增订），本席 grep 定位，无实质差异。

## 6. 与否决清单合规

对照 `ROPE_LOCAL_FAILURE_SYNTHESIS_20260908.md`：本报告不引入静态洞率外推作机制解释、不把"碰撞率"当充分统计、不以未执行实验虚构效果（0446/0448/0449 一律"构造已验证、判决未回"）、E8 归 I2/尾带、E2/E8 12 行不与 36 行直比、MrUni 不作端点违例证据、循环防护门（组件"已验证-开发面板"不得当结论）均遵守。 **[已验证]**
