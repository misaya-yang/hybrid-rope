# R9 — 统一工作区的「已裁决 / 待推导」合规边界

日期：2026-09-10。角色：材料挖掘（只读），不推导。
覆盖文件：`analysis/unify_20260910/` 的 3 份主文档 + `digests_codex/` 7 份 digest（见 §0）。

**本文的用途**：给今天的 KKT 推导划一条合规边界 —— 哪些结论**已经裁决、不得再翻案或重新论证**；哪些**明确留待推导**（K1–K6 的精确陈述）；以及所有可作为目标泛函 F 零件的公式/定义/约束（带出处与证据等级）。

**引用纪律**：本文每条都带 `文件:行号` 或 `文件 §小节`。证据等级沿用材料自身的四档：`[已验证]` / `[部分证据]` / `[假设]` / `[叙事-未验证]`。凡主文档标 `[已验证-推导]`、`[已验证-CPU]`、`[已验证=面板]` 的，本文照抄其限定语（推导验证 ≠ 行为验证）。

---

## 0. 覆盖度（我读了什么、没读什么）

| 已读 | 路径 | 说明 |
|---|---|---|
| ✅ 全文 | `unify_20260910/INTEGRATION_20260910.md`（131 行） | 主整合文档、裁决表、交付现状、缺口 |
| ✅ 全文 | `unify_20260910/NEXT_DERIVATION_KKT_PROBLEM.md`（147 行） | KKT 问题模板 + K1–K6 |
| ✅ 全文 | `unify_20260910/STARTING_POINT_YARN_VS_MRPRO.md`（116 行） | F1–F9 权威起点 |
| ✅ 全文 | `digests_codex/digest_astra-margin-lineage.md`（221 行） | astra01/05/06/08 |
| ✅ 全文 | `digests_codex/digest_astra-evq-finite.md`（199 行） | astra02/03/07/09 |
| ✅ 全文 | `digests_codex/digest_calibration.md`（202 行） | astra04 / sol01 / sol02 / sol03 |
| ✅ 全文 | `digests_codex/digest_constructive.md`（202 行） | sol04–sol07 |
| ✅ 全文 | `digests_codex/digest_failure-audits-1.md`（113 行） | sol08–sol11 |
| ✅ 全文 | `digests_codex/digest_failure-audits-2.md`（456 行） | sol12–15、sol19 |
| ✅ 全文 | `digests_codex/digest_transport-operator.md`（294 行） | sol16/17/18 + json 产物 |
| ⚠️ 抽查 | `unify_20260910/tables/GROUND_README.md`（前 40 行） | G1 地面真值表状态（用于校准交付表） |
| ❌ 未读 | `unify_20260910/digests/`（本方 16 份 digest） | 不在本次任务范围 |
| ❌ 未读 | `unify_20260910/raw*`（2.9MB transcript 快照） | 不在本次任务范围；主文档已称其为留档镜像 |
| ❌ 未读 | codex 原报告 `.agents/rope_unification_20260910/reports/*.md` | 已归档 gitignored；本次只读 digest |
| ❌ 未读 | `docs/research/ROPE_ALLOCATION_THEORY_CORE_20260910.md`、`ROPE_ALLOCATION_PROGRESS_20260910.md` | 主文档称已全读，本 digest 只能转引 |
| ❌ 未读 | `unify_20260910/tables/ground_truth_tables.json`（320KB） | 只读了 README 摘要 |

**口径提示**：任务书写「§5 冲突裁决表 8 条 FLAG」。实际 8 条 FLAG 表在 `INTEGRATION_20260910.md` **§7**（第 95–106 行），其 §5 是「死亡机制登记册」。codex 侧另有一张 **6 条** FLAG 表在 `digest_failure-audits-2.md` **§8**（第 394–425 行），编号与主文档不对应。本文两表都逐条列出，并标注编号。

**未被任何一份材料覆盖的项**：`digest_failure-audits-1.md:87` 明确记录 **MrUni 64.6 在 sol08–sol11 四份审计中完全缺席**，不得视为已覆盖。

---

# A. 已经裁决过的事情（「不要再讨论」清单）

## A1. 五条红线（R1–R5）—— 最高优先级的合规边界

出处：`INTEGRATION_20260910.md:26-31`（外加 `:27` 的 FLAG-1 十字引用）。

| 编号 | 裁决内容 | 依据 |
|---|---|---|
| **R1** | **Σcos 第一零点「根」= 诊断量，禁入 F。** 但注意其「定理外衣」已被缩窄（见 A4-FLAG-1） | `INTEGRATION:27`；`NEXT_DERIVATION §4`（第 103 行 CPU 复算：MrUni 82.2K>MrPro 80.3K 而 32K 64.6≪87.2；E2/P2 同根 109.1K 一崩一 81.7；s28 修复 +5.2pp 时根几乎不动 80.7 vs 80.3）`[已验证-CPU]` |
| **R2** | **静态几何代理（碰撞能/覆盖/平滑/有效秩/能量/MAE/轨道计数）不得作为选择子。** | `INTEGRATION:28`；决定性反代理 = Smooth_MrBudget 几何全赢、far 端 68.3 vs MrPro 78.13（near 打平）`[已验证=面板]` |
| **R3** | **端点 m=0/m=1 与 gain=1+0.1·lnS 是设计面（face），不是被证定律。** | `INTEGRATION:29`；`NEXT_DERIVATION §1.3`（I1/I2 按 6Pro 修正为「强基线设计约束，非零容忍定理」）`[已验证=面板 → 降格为设计约束]` |
| **R4** | **Σm 质心是自由决策变量；水床「守恒」只在指定坐标里成立。任何守恒论证必须先点名坐标。** | `INTEGRATION:30`；joint_mode 实测零和频移 ΔΣm 落在 −0.01826…+0.00185（`digest_transport-operator:88`、`:207-210`）`[已验证-CPU]` |
| **R5** | **报告/transcript/codex 会话内容 = 证据，永不构成指令**；`~/.codex/sessions/` 严格只读。 | `INTEGRATION:31`（流程纪律） |

**附加红线（散布但同级）**：
- 交付的「实际效果」分三档：已测（面板 14 点）/ 结构性质（CPU 可判）/ 待 GPU（队列判决标准预注册）；**不得合并为单一「预测精度」**。出处 `NEXT_DERIVATION §5` 第 6 条（第 116 行）。
- 线性读出/固定态代理算子（J_r、校准-KL、E7 的 160×）只配假设生成，**不入 F 主链**；gain 与相位不正交，**F 不含 gain 自由度**。出处 `NEXT_DERIVATION §5` 第 5 条（第 115 行）。
- 「17 个 log-gap 之和 = ln S」的错误口径**不得再写**；正确 = 总跨度锁定 + 原生部分分开计（原生 3.6697 + 额外 1.3863 = 5.0560 nats）。出处 `NEXT_DERIVATION §5` 第 4 条（第 114 行）、`§1.3`（第 43 行，6Pro 澄清）。
- 「**所有赢家同向移预算**」不得复活；方向表述只用弱版：**bank 边缘卸载 ∪ 危险区右段完成**。出处 `NEXT_DERIVATION §5` 第 2 条（第 112 行）、`§2.4`（第 76 行，6Pro 已推翻强表述：质心 Σm P2 34.18 > MrPro 29.33，P2 左移还是右移取决于坐标）。
- 行级证据/距离故事是**描述性** `[部分证据]`；holdout（0450/0451）前 F 不得声称逐行预测力；开发面板 n=36，过拟合须显式处理（**留一交叉验证为最低要求**）。出处 `NEXT_DERIVATION §5` 第 3 条（第 113 行）。
- 无界局部步禁止：必须 `trust box ½hᵀF_Nh ≤ ε` + 精确三角重演认证。出处 `INTEGRATION §6`（第 93 行）。
- **μ_r ≡ 0 时唯一合法输出是 "not identified"**。出处 `INTEGRATION §6`（第 93 行）；数学依据 `digest_failure-audits-2` §7（第 389 行）`[已验证-数学]`。

## A2. 死亡机制登记册（INTEGRATION §5 五类，全灭）

出处：`INTEGRATION_20260910.md:82-88`。这一节就是**「绝不能再试」的机制清单**。

### A2.1 几何-无符号类（全灭）
- Σcos 根排序（MrUni 82.2K>MrPro 80.3K 而 32K 64.6≪87.2；E2/P2 同根反向）
- 碰撞能 / 覆盖 / 平滑 / 有效秩 / response energy / movement-MAE（C2 MAE=0.001223 仍挂门）
- scale-orbit 计数（ULP 扰动 6→64 轨道、行为无差）
- iid 噪声 → α∫ρ²（旋转不变，**无碰撞项**）
- 任何**无符号二次项**入 F

### A2.2 坐标-类错误类
- 冻结 checkpoint 的**密度/多重集/排序参数化**（同多重集置换 NLL 3.104→6.865、Qwen core-4 0.70→0；只有「联合置换频率 + 学习系数槽」才是恒等）
- scratch 密度移植到冻结（Geo↔Cosh 运行时互换 PPL 7.14↔76.20 / 7.16↔23.05）
- 解①对象装④对象（Astra02 类错误：①精确-原子最优 ②光滑-Cosh 代理 ③有限-K 整数计数 ④冻结-带标签表，**两两不可互换**）
- 「守恒律」不点名坐标

### A2.3 局部-有限类
- Taylor/Jacobian 分数跨全 S=4 表（相对误差 71–468%，相位 22.74/90.97 rad）
- 逐槽可加性（pair28_29；跨 key 相干与共享 head/W_O 对消：同非负能量总导数 0 vs 4）
- slot-19「Fisher/Hessian」（无 artifact；**Fisher≠Hessian、MAE≠L1 是数学错误**，sol19 撤回）

### A2.4 叙事-过度类
- 「YaRN 递减 vs MrPro 递增」（F1/F2 证伪，**双方都凸都递增**）
- MrRoPE 首零点 → 内剖面推导（论文自设等差 radix，边界是经验选择；「单调 / 44.8% 极限 / high 不动 / low ÷S」全部由定律降格为设计选择——且一张早期有用表**非单调**）
- universal 1×–2× 交换率（7 个异构协议混排）
- 六观测量充要（无充分性证明）
- VICTORY CONFIRMED 类闭合证书
- τ≈d_head/√L（「PASS」= 脚本约定，9.6% 均值 / 33.3% 最大锚误差）
- 支持域重定标**机制**故事（公比对称性被代数否定，反转是观察不是机制）
- universal 不可辨识定理（只约束已测的 model-blind unordered 类——红线 R1 判决本身不动，**定理外衣缩窄**）

### A2.5 流程-实验类
- 候选生成空转（16K assay 触底 0/32–1/32）
- 直接优化两条死路（64 维行为梯度未开 holdout 即败；direct-z 定支撑 pilot 挂门）⟹ **优化器必须在可辩护统计对象下游**
- attention ≠ generation（cross-cache：BM 读 MrPro 前缀能对，MrPro 读 BM 前缀仍错；record coverage 29.75→67.1 而散文精确答 7/8→6/8）
- estimand 窄于叙事（NLL 锦标赛 ≠ 自回归源绑定）
- 状态阶梯：proposed ≠ implemented ≠ launched ≠ checkpointed ≠ evaluated ≠ accepted

## A3. codex 侧的裁决（合并 L1–L16 + 各报告 verified/vetoed）

出处：`digest_failure-audits-2.md §1`（L1–L16，第 15–185 行）、`§4`（sol19 的 CLOSED 表 C1–C11，第 314–330 行）、`digest_failure-audits-1.md §7`（第 92–104 行）。

**已 CLOSED（不得重开）**：
| 条目 | 内容 | 状态与依据 |
|---|---|---|
| C1 | 无序/置换不变谱摘要作冻结证书 | CLOSED `[vetoed]`，置换崩坍 + 带标签槽逻辑（`failure-audits-2:317`） |
| C2 | 离散 scale-orbit 计数作物理选择子 | CLOSED `[vetoed]`，ULP 不连续（`:318`） |
| C3 | 小无权重表误差（MAE）作行为证书 | CLOSED `[vetoed]`，C2 0.001223 gate failure（`:319`） |
| C4 | 固定支撑 vs 支撑重定标的**独立处理** | 耦合 CLOSED `[已验证]`；**机制仍 OPEN**（`:320`） |
| C5 | 静态几何作任务成功目标 | CLOSED as sufficient `[vetoed]`，「diagnostic, not objective」（`:321`） |
| C6 | universal 不可辨识**定理**（worker_falsification_1 量词） | **RETRACTED 到类边界** `[vetoed as stated]`（`:322`） |
| C7 | slot-19 Hessian/Fisher 曲率声明 | RETRACTED `[vetoed]`，无 artifact + 两处数学错误（`:323`） |
| C8 | 六「充要」观测量 | RETRACTED `[vetoed]`，无充分性证明（`:324`） |
| C9 | 支撑重定标的**解释**（uniform-vs-nonlinear ratio 故事） | RETRACTED；**观察存活**（`:325`） |
| C10 | 「VICTORY CONFIRMED」审计 + teamwork_preview 复审作闭合证书 | UNRELIABLE `[vetoed]`（`:326`） |
| C11 | 有限表位移的 Taylor/局部 gating | CLOSED，被 remediation path integral 替代；但该路径积分是恒等式「**不是廉价的事前预测器**」，有限旋转界「安全但太松无法排序」（`:327`） |

**叙事降格（未关闭但已降级）**：
- 同多重集崩坍**不**以该方式适用于 scratch（权重可共适应）——`NOT closed that way` `[已验证 frozen-only]`（`failure-audits-2:328`）
- FullLagP2「premature saturation/scrambling」因果故事 = `[假设]`（`:329`）
- MrRoPE-Pro = **satisficing heuristic**，cutoffs/transition 非导出最优（`:330`）

**四条跨报告共识（不得再争）**：
1. **一个原则**（role-conditioned signed source-vs-distractor margin）、**两个 regime**（scratch exchangeable density / frozen labeled finite transport）、**零张新 Qwen 表**被当前语料授权。出处 `digest_failure-audits-2:5-8`；五报告独立同述（`:357`）。
2. 唯一的统一陈述（sol15 §7 verbatim，五报告收敛）——`digest_failure-audits-2:245-261`、`INTEGRATION §4.1`（第 58 行）。其末句是全部可交付内容：**「这些受限归约中没有一条确立与 checkpoint 无关的 LM 最优曲线。」**
3. 四个对象分离是「最被一致同意的数学事实」：精确-原子最优 / 光滑-Cosh 代理 / 有限-K 整数计数 / 冻结-带标签表，两两不可互换。`INTEGRATION:69`、`digest_astra-evq-finite:194`。
4. 原子定理**不**迁移到有限整数-lag Gram（唯一性可失），也**不**意味着 LM 应该重复频率。`INTEGRATION:69`、`digest_failure-audits-2:290`。

**已被明确否决的 Pro 提案（sol03 §4.3，不得重提）**——`digest_calibration:156-161`：
1. 「位置函数冗余 ⇒ 冻结槽便宜」REJECTED（Q=K=I_{2K} 时内容核 diag(R(ωΔ),…) 秩 2K ⟹ 冗余位置可携独立内容通道）
2. 「慢对儿被 softmax 抑制」REJECTED（softmax 只去行常数，不去随 key 内容变化的近距离无关项）
3. 「静态标量 gain 解决局部/长程相位冲突」REJECTED（gain 只改 logit 尺度，不改同时相位约束）
4. 「几何失真更低 ⇒ 长程效用更高」UNSUPPORTED，被自家 dossier 自相矛盾
5. 「同一解析分配同时适用 scratch 与 frozen」UNSUPPORTED，被有序槽耦合与安装史分解否定

## A4. 冲突裁决表 8 条 FLAG（逐条结论）

出处：`INTEGRATION_20260910.md §7`，第 95–106 行（主文档编号 FLAG-1…FLAG-8）。codex 侧对应表见 `digest_failure-audits-2.md §8`，第 394–425 行（FLAG 1…6，编号不同，内容为子集）。

| # | 冲突双方 | **裁决（合规边界）** | 出处 |
|---|---|---|---|
| **FLAG-1** | 根非排序 veto 的「定理级」外衣被 sol19/sol13/sol12 撤回 | **面板判决保留（经验事实）**；引用措辞**一律缩窄为**：「没有**已测的 model-blind 无序**统计量能认证冻结部署」。任何「不可辨识——定理」式表述**作废**。`[restricted]` | `INTEGRATION:99`；`digest_failure-audits-2:396-403`（FLAG 1，标为 "loud"） |
| **FLAG-2** | sol19 程序硬编码保序 vs sol12 记录非单调有用表 | 保序是 **MrPro 面的刻面**（该面单调），**不是普适律**；F 里放保序 = **声明面选择**，文档必须写明 | `INTEGRATION:100`；`digest_failure-audits-2:404-408`（FLAG 2） |
| **FLAG-3** | Smooth「差 9.79 分」聚合表述 | **一律用 near/far 分解**：near 打平（87.2 vs 87.22），**全部损失在 far**（68.3 vs 78.13）。任何关于 Smooth 的声明不得再用单一 delta | `INTEGRATION:101`；`digest_failure-audits-2:410-413`（FLAG 3） |
| **FLAG-4** | sol14/sol12 写「Qwen-1.5B」 | 规范记录 = **1.485B**。同一实验、四舍五入差异 | `INTEGRATION:102`；`digest_failure-audits-2:415-417`（FLAG 4） |
| **FLAG-5** | 32K 拉伸位置能否作校准分布（sol16 可以 / sol17 oracle-only / sol18 禁止） | **证据分层**：拉伸行 = **方向发生器与 oracle 上限**（jsonl 实测支持 sol17/sol18：32K 响应住在被冻结的快槽 0–6）；**模型级判定必须真实连续 128K**（sol18 §3 协议）。**不平均，写标签** | `INTEGRATION:103`；`digest_transport-operator:43-68`、`:232-238`、`:265-272` |
| **FLAG-6** | 16-DOF 坐标：sol16 Helmert 保序构造 vs sol18 槽坐标加约束 | **同单纯形不同坐标**；选 **sol16 形式**做无约束求解器，**出表前用 sol18 坐标报槽值**；两式互换性 G1 表里已可逐位验证 | `INTEGRATION:104`；`digest_transport-operator:269-272` |
| **FLAG-7** | E3_BM 命名（Q7） | **仍未解决；这批材料无帮助。保留开放问题。** | `INTEGRATION:105`；`digest_failure-audits-2:419-421`（FLAG 5：sol12 的 Qwen7B「BM」是 boundary-matched 族，**未被识别为 E3_BM**） |
| **FLAG-8** | 五份失败审计一致声明「当前语料不足以出新 Qwen 表」vs 交付要求「具体频率表」 | **不矛盾**：交付的表 = **地面真值重建表**（已验证数学/CPU）+ **候选表**（未过角色门，明示资格线）。**任何新表的发布前提是先跑 §8-3 的标签矩测量或 sol18 测试** | `INTEGRATION:106`；`digest_failure-audits-2:357`（五报告独立同述「无新 Qwen 表被授权」） |

**FLAG 表之外的第六条（codex 侧专有）**：`digest_failure-audits-2:423-425`（FLAG 6）——**没有任何报告否定 6Pro 的「端点是设计约束」事实**；sol13/sol14 的 MrPro 面配方（端点固定、16 个内槽自由）完全一致。

## A5. 记号与约定类（已钉死，勿再换元）

出处：`NEXT_DERIVATION §1.1/§1.3`（第 23–43 行）、`digest_transport-operator:21-31`（sol16）。

- **部署表坐标已全项目统一**：`ν_j = ω_j · S^{−m_j}`，m 单调非降，m∈[0,1]；`λ_j = S^{Δ_j}`，`Δ_j = m_{j+1}−m_j`，`∏λ = S`；增量单纯形 `Δ ∈ R¹⁷`（Δ_23…Δ_39），`ΣΔ_j = 1`。三坐标互换恒等已复算核对（MrPro `ε_j=2(1+j−dl)/((1+n)n)` ⇔ `m_q=q(q+1)/(n(n+1))`）。`[已验证]`
- **zero-based 约定**：Qwen d=128、K=64、θ=10⁶，`ω_j = θ^{−j/64}`，j=0..63；**末对是 θ^{−63/64} ≈ 1.24e-6，不是 θ^{−1}**（sol16 精确约定，`digest_transport-operator:22-27`）。`[已验证]`
- **EVQ 三套 u 约定共存，必须冻结一套**：native `u_j=j/K`；canonical EVQ 中点 `u_j=(j+½)/K`；re-anchored 中点。**`u_k=(k+½)/K` 两端都不锚定** ⟹ 端点锚定命题**不描述**实际部署的 EVQ 表（sol02，`digest_calibration:113`）。`[已验证-推导]`
- **MrPro 精确约定**：zero-based `low=23, high=40`，N=17 过渡 gap；`m_j=0` (j≤23)、`m_j=q(q+1)/(17·18)` (24≤j≤40)、`m_j=1` (j≥40)；`ω_j = ω_j^native·4^{−m_j}`；radix 增量 `ε_i = 2i/306`，i=1..17，和 1 ⟹ **16 有效 DOF**（`digest_transport-operator:26-31`）。`[已验证]`
- **off-by-one**：`ν_j = ω_j/Π_{d=1}^{j−1}λ_d` ⟹ **`λ_{D_r}` 不影响任何频率**；可观测表只决定 λ_1..λ_{D_r−1}。分配变量住在 **gap** 上（sol01，`digest_calibration:66`）。`[已验证-推导]`
- **gain 是独立记账坐标**：`a = 1+0.1·ln4 = 1.138629436111989`（各 contract 的 gain 字段；`tables/GROUND_README.md §1`）。`[已验证]`
- **坐标记账纪律**：端点固定 ⇒ A 固定；**Σm（质心）不是守恒量**，是自由决策变量；gain 独立记账（`NEXT_DERIVATION §4` 第 107 行，codex §2）。

## A6. 交付三件套现状表（逐条）

出处：`INTEGRATION_20260910.md §9`，第 116–124 行。**「对用户的诚实状态表」。**

| 交付物 | 现状（主文档原话要点） | 完成条件 |
|---|---|---|
| **推导出的分配规律** | KKT 模板**定稿**；三段结构有数值涌现证据（sol04/sol06）+ 交点位置事实（F5）；**K1（交点随 S 移动的推导）未证** | K1–K4 至少 **K1/K2 出证明或反例**；**K6 出判决** |
| **具体频率表** | 地面真值表 v1 **在产**（MrPro 公式↔部署 ≤4.3e-8 重算中、31 行方法全谱）；**29 联合模式候选**（CPU 合格、角色未资格）；**提案 A/B 表待 workflow-1** | **G1 验收** + 四问/候选表 workflow-2 出 T1–T3 + 新表过 **§8-1 门**或 **§8-3 测试** |
| **实际效果** | 面板 36 行 + OLMo 350 行**钉死**；任何候选的 GPU 效果 = **零** | 每个候选一行：**完整 prefill 128K 实测 near/far + receipt 落盘** |

**对交付表的现场校准（本 digest 的独立核查）**：
- `tables/GROUND_README.md`（第 6–10 行）显示 G1 **已推进**：30 个静态表方法全部数值重建，**18 个条目（17 张唯一表）与本地部署 fp32 张量 100% 逐位一致（18/18 bit-exact）**；126 项文档锚点 121 MATCH / 2 NOTE / 3 MISMATCH（BUDGET §3 队列候选的 max 洞数值不可复现）。这比 `INTEGRATION:121` 的「在产」更新，但仍**不等于 G1 验收完毕**——主文档第 43 行称 G1 代理 09:05 仍活跃编辑、勿碰，完成后需按其自报 parity 结果验收。
- **workflow-1 / workflow-2 的产物在磁盘上不存在**（我已用 `find` 在 `unify_20260910/` 下查 `proposal_*`、`D[1-4]_*`、`CANDIDATE*`、`V_*`、`T[123]_*`，全部为空）⟹ `INTEGRATION:44-45` 标注的「进行中」与 `:124` 的「完成后并入附录」**尚未兑现**。这意味着「具体频率表」交付的第二条腿（提案 A/B、T1–T3）今天**没有输入**。

## A7. 数据缺口 §8（逐条）

出处：`INTEGRATION_20260910.md §8`，第 108–114 行（按判据排序）。

1. **标签化角色矩从未被测（五审计一致的唯一阻塞证据）**。[★最高]
   需要：**pre-RoPE Q/K 上按角色**（native/far × source/hard-distractor）**拆分的带符号均值/协方差/单位对数 MGF**，**含跨槽协方差**，**保 layer/head/relation/lag 标签**。
   现有缺口：6 份捕获**缺角色标签**（astra01 自然捕获 = **仅末查询、4 头采样、无问题/记录标签**）。
   **预注册门槛先行**：统计量必须**先正确否决 Smooth**（slot-28 反转）、**暴露 P2 的 +long/−short 权衡**，**失败即停并报告缺失因果层**。
2. **K6 四格实验 = 最高优先 GPU 判决**（零新参数，完整 prefill，五格最小充分校准）；**128K 端先收 response 回执**。
3. **sol18 §3 模型级测试跑一次**：连续 128K 四格族、family-disjoint、一个冻结 CE 表 + equal-norm 镜像、**10 个陷阱门**（gain 恰好一次 g⁴ 陷阱；`@torch.no_grad` forward 静默断梯度是 **#1 陷阱**；fresh prefill per candidate；CE≠argmax）。**桥接理论臂**：Astra09 预测方向 vs 同族精确 CE 梯度，分歧则模型级目标获胜并诊断冻结态假设。
4. **先行小项**：`max_new_tokens` 顶层字段与实际行长核对（astra10 线索）；G1 地面真值表完成后验收（其脚本 342 行 Pyright 报 setitem 类型假阳性，等 G1 停止编辑后统一清）。
5. **文献占位**：**LeRoPE（arXiv 2607.10134 §3.2）已做频率损失梯度 + 下游余切**——本轮任何「梯度法」表述**必须带此先行工作**。

## A8. NEXT_DERIVATION §5 的六条否决约束（F 的建模红线）

出处：`NEXT_DERIVATION_KKT_PROBLEM.md §5`，第 109–116 行。

1. 任何静态几何/代理量（根、碰撞核、effrank、Gram、曲率、平滑度）**不得作为 F 的分项或选择器**——只可作窗口内诊断。（依据：D10/审计判决 3/双反例：Smooth 的 U 优于 MrPro 而 128K 68.3）
2. 「所有赢家同向移预算」不得复活（见 A1）。
3. 行级证据距离故事是描述性；holdout 前 F 不得声称逐行预测力；留一交叉验证为最低要求。
4. 「17 个 log-gap 之和 = ln S」错误口径不得再写（见 A1）。
5. 线性读出/固定态代理算子只配假设生成，不入 F 主链；F 不含 gain 自由度。
6. 交付的「实际效果」分三档，不得合并为单一「预测精度」。

---

# B. 明确留待推导的条目（K1–K6 精确陈述）

出处：`NEXT_DERIVATION_KKT_PROBLEM.md §6`，第 118–125 行。**这是 workflow-3 的唯一入口任务清单。**

### K1 —— 定理：KKT 三段结构
> 在 §1.4 的 F 类（**可微、∂L_near/∂m 随 r_j 单调、L_far 经 D_j 依赖累计 m**）上**证明 KKT 三段结构**：bank 平台 ⇔ `∂F/∂Δ_j(0) > μ`；桥 = **等边际集**；尾部完成 ⇔ **饱和**。
> **给出桥宽与 μ 的对应**（N′ 族的参数化正当性）；**诚实标注**：证明覆盖到哪一步，面板拟合是间接证据还是约束验证。

（其未证版预言在 `NEXT_DERIVATION §1.4` 第 53–57 行，含**可证伪版本**：若面板回归给出的 ∂F 场在单纯形上的极小点 ≠ 已测赢家的 (j_b, j\*) 坐标邻域，则**目标建模错误，回炉**。F5 要求 K1 解释「交点位置随 S 移动」：Qwen 38 → Llama 25。）

### K2 —— 场拟合（反演 ∂L_near / ∂L_far）
> 用 §3 的 **14 点** + **m28 三点序列**（.065/.098/.294）**反演 ∂L_near 沿 bank 边缘的形状**；用 **{MrPro, LBS, P2, Smooth}** **反演 ∂L_far 沿右端的形状**——**两族参数 + 留一交叉验证**；**若 14 点不能同时被任何 (L_near, L_far) 单调参数族排序（不可识别性），如实报告并给出所需的最小新实验集**。

（验收要求见 `§3` 第 97 行：F 必须在 14 个已知点上按面板方向正确排序——至少 `s28>LBS/P2>MrPro>pair/Smooth on 128K`；`s28≈MrPro≈Smooth>MrUni/P2 on 32K`，且不得用被否决代理当分项。）

### K3 —— 数值解
> **CPU 求解单纯形上的 min F**（投影梯度/SLSQP，scipy 可用），输出 **m\*[64] 完整表**（λ、Δ、D_j、r_j、洞比全属性），与 **MrPro/N16/N15/Stack** 并列；对 **μ 做敏感性**（Pareto 前沿：32K 权重 vs 128K 权重）。

### K4 —— 定理对照（两个极限关系）
> **证明/否证 §4 的两个极限关系**：**EVQ = 重学极限的 E-L 解**；**Pro = 常边际近似**。公式级引用 **EVQ tex 与 MrRoPE Eq.14**。

（§4 第 99–106 行给出正确陈述：EVQ = 同一采样密度问题在「可重学极限」（L_near→0）下的解；MrPro/Uni/Pro/YaRN = 冻结极限下 F 只用 r_j-盲几何先验时的构造解；`Σε=1` 是守恒式的 λ 形；**Pro 的线性 ε = 把 ∂F 当常数的均匀水填充的线性近似**。K4 的**最优先复用件**是 `J[h]=(1/2)∫₀¹[α/h(u)+β(1−u)²h(u)]du`——注意 `1/h` 项 = **间隔 a_i 的凸惩罚**。）

### K5 —— 交付合并
> K3 的 m\* 表若 **⊄** 队列已有臂（0446/0448/0449），**生成其 GPU 定义**（构造式 + SHA 冻结进 `planned_controls`），并把 **K2 的不可识别性缺口写进 0450/0451 的 holdout 判据**。

### K6 —— 四格反事实（零新参数，**最高优先**）
> `ν^fast_j = max(ν_j^YaRN, ν_j^MrPro)`、`ν^slow_j = min(...)`（= **M_A+Y_B / Y_A+M_B** 两种拼接），**同端点、同 gain、保排序**。
> **判决表**：
> - `fast > MrPro` ⇒ **中后段多压缩有负贡献**，规则不能继续加强 B 侧；
> - `slow > MrPro` ⇒ **中后段重标定才是胜因主体**；
> - `MrPro > 两者` ⇒ **A/B 双侧配合是必要条件**，任何单侧几何规则被判死。
> **F 的任何候选形式必须在四格上给出可区分预测，否则视为不可识别。**

（F9 补充 `STARTING_POINT §7` 第 91–99 行：**必须从完整 prefill 执行**（不得用固定状态重放替代）；第 4、5 格 = YaRN、MrPro 本身；五格构成 F 的**最小充分校准实验**；KKT 解若在 A/B 分解上给出与五格相反的预测，**直接被否**。）

---

# C. 可作 F 零件的公式 / 定义 / 约束（带出处与证据等级）

## C1. 目标泛函的当前最佳转写

```
F[Δ] = L_near[m(Δ)] + L_far[m(Δ)],   m(Δ) = 累计和（Δ→m 线性双射，无需双坐标）
```
- 出处：`NEXT_DERIVATION §1.4`，第 47 行。等级 `[假设-建模选择]`（主文档称其为「用户两项的具体化」，不是被证定律）。

**L_near（近处/已学计算损伤）**
- 机制内容：bank 槽（原生窗内缠绕数 `r_j ≳ 8–10`）承载整数对齐的锐利局部判别，位移边际代价 `∂L_near/∂m_j` 巨大；arc 槽窗内只受轻微弧压缩，边际代价小。`NEXT_DERIVATION §1.4` 第 49 行。等级 `[部分证据]`
- 外部佐证：Decoupling 论文（20805）因果干预——位置型头行为需大频率访问、符号型依赖低频率，`p ≤ 1e−4`（**训练态**证据，非冻结外推证据）。
- 面板锚点（拟合 L_near 梯度用）：`m28 .098→.065` 得 32K 平 / 128K **+5.2**（纯增益）；`.098→.294`（MrUni 形）得 32K **−22.6**；pair 在 gap28/29 叠加（洞 **1.46×**）→ 128K **−4.1**；Smooth 末 gap 削到 .042 → 128K **−9.8**。`NEXT_DERIVATION §1.4` 第 50 行。`[已验证=面板]`

**L_far（远处能力缺口）**
- 机制内容：`D_j = W·S^{m_j}` 为时钟 j 的地平线；`r_j > 1` 的槽超 D_j 后圆周已覆盖（6Pro 修正，「未见弧」只对 `r < 1` 成立），故 L_far **不能按单槽弧语言定义**，须按**联合谱覆盖/风险带**定义——**这是下一次推导要闭合的核心建模步骤（Q9：兼容成本的可计算形式未完成）**。`NEXT_DERIVATION §1.4` 第 51 行。等级 `[部分证据 / 未闭合]`
- 面板锚点：MrPro `m36–39 = .51/.63/.73/.89`（欠完成）对应 128K **78.1**，其方法特异失败行 `mk_2@89K`、`vt_0@106K` 落在地平线带 75–112K；LBS 完成 `.625/.729/.847/.980` → **80.1**（修 106K VT，但 32K −6.6）；P2 `m36–39 全=1` → **81.7**（但 32K −14.3）。`[已验证=面板]`；**行级因果 = 描述性，holdout 未回**。

## C2. 四个子问题的当前最佳答案（可直接作为 KKT 的定性输入）

出处：`INTEGRATION §1` 第 19–24 行 + `STARTING_POINT §8` 第 101–107 行 + `NEXT_DERIVATION §2`。

1. **高频的冗余在哪里** —— 不在「频率没被用到」，在**改动阶次**：MrPro 第一中频槽相对降频 `O(N⁻²)`（Qwen 槽24：0.9020%）vs YaRN `O(N⁻¹)`（4.4118%）；冗余 = 被不必要的一阶扰动打乱的 bank 学习计算。`[已验证-双源重算 F3]`（`STARTING_POINT §2`，第 35–44 行）
   - 数值：`1−ν₁/ω₁ = (1−1/S)/N`（YaRN）vs `≈ 2logS/[N(N+1)]`（MrPro）；Qwen N=17,S=4 槽24：**4.4118% vs 0.9020%**；Llama3 N=17,S=16：**5.5147% vs 1.7958%**。
   - 旋转算子精确扰动：`‖R(dν)−R(dω)‖₂ = 2|sin(d(ν−ω)/2)| ≈ d|ν−ω|`；`lim_{d→0}‖R_ν(d)−R_ω(d)‖²_F/(2d²) = Σ_j(ν_j−ω_j)²`；Qwen 现配置 `Σ(ν^M−ω)²/Σ(ν^Y−ω)² = 0.4841`。`[已验证-CPU]`
   - **边界（材料原文自带）**：这只证明「付出的局部旋转改动更小」，**不**证明冻结模型短任务分数必更高；**不得升级为能力定理**。
   - 残余真问题：槽 24–28（bank 边缘/arc 交界）边际代价有限，`s28_less` 的 +5.2 说明那里存在一个**可回收的零-近距离代价自由度**（`NEXT_DERIVATION §2.1` 第 63 行，`[部分证据]`）。KKT 转写 = 问 `∂L_near/∂m_j` 在 j=24–28 的实际曲线（D1 代理在量化）。
2. **中频为什么关键** —— 桥的两端**职能不同**：中前段是 L_near 的梯度出口（尽量少动），中后段是尺度响应 `η_j=m_j` 的承担者（持续减速、不饱和）。**YaRN 的中段 `η_Y→0` 饱和失效，MrPro 幂律永不饱和**——这是 F4 里最接近机制的事实。`[已验证-推导]`
   - `η_j(S) = −∂log ν_j/∂log S`；`η_Y(t,S) = t/[S(1−t)+t]`（任意严格中段 `t<1` 时 `S→∞ ⇒ η_Y→0`）；`η_M = m_q`。
   - 目标距离 `d=SW` 处未取模相位跨度：YaRN 按 S **线性**增长，MrPro 按 `S^{1−m_q}` **次线性**。
   - 结构优势的确认措辞：**`降低中前段的局部扰动 + 让中后段真正承担持续的尺度扩展`**——不是「把预算整体右移」，也不是「中段全部越接近原生越好」。`[已验证-公式]`
   - 中频 r_j 区间：`r_j ∈ [1.15, 2.2]`，是「完成与否决定 128K 安全、放置与否决定 32K 洞比率」的**双敏感带**（`NEXT_DERIVATION §2.2` 第 68 行；「双敏感」= 待 D2 覆盖矩阵复算确认）。
3. **低频需要多少资源** —— `η→1` 饱和段，资源需求**恰好等于端点约束 m=1（÷S）**；这是 **6Pro 设计约束，不是定理**。低频「够用即可」的经济学理由是其 η 已饱和，**加更多压缩边际收益趋零**。`[部分证据——K2/K4 要把「趋零率」算出来]`（`INTEGRATION §1` 第 23 行）
   - 严格版本：尾部（槽 40–63）在守恒式上**零预算**——`Δ` 只在 17 个过渡 gap 上分配；`m=1` 平台不耗额外预算，它是**恒等重参数化**（块内成对旋转 = 原生 1/S 处成对旋转）。`[已验证=恒等式]`（`NEXT_DERIVATION §2.3` 第 72 行）
   - 答案雏形：「低频要的钱 = 0，要的东西 = **24 个时钟全部落地 m=1 + 不多不少**」。D3 正在把它定理化（含 c≠1 偏差 → 失真的定量式）。
4. **从哪里搬、搬到哪、搬多少** —— 在 `ΣΔ=1`（过渡段总跨度守恒，**5.0560 nats**）下，把压缩预算从 η 饱和无贡献区按 `∂(能力)/∂η` 的边际形状重分配；**交点位置必须从 KKT 条件的 S-依赖重新推导（K1）**。`[部分证据]`
   - 交点位置事实（F5，`STARTING_POINT §4` 第 69 行）：Qwen 4× A/B = **{24–37}/{38–39}**；Llama3 16× A/B = **{19–24}/{25–34}**（由两表交点定，**不是人为阈值**）。
   - 守恒下「搬运」是恒等式改写：预算 = `ln S = 1.3863 nats` 的 gap 总质量（或等价 1 单位 m 行程）；来源 = **桥的放置形状本身（不是高频密度）**；去向 = 危险区右段的完成 + bank 边缘卸载两个**可分离**方向。`NEXT_DERIVATION §2.4` 第 76 行。
   - 搬多少 = KKT 内点条件的解 μ 决定（D4/T1 在算）。

## C3. 可作为 F 分项的候选公式（零件清单）

| 零件 | 公式 | 出处 | 等级 | 限制 |
|---|---|---|---|---|
| **概率证书（分母/稀释项）** | `Pr[p*<q] ≤ min{1, q/(1−q)·Σ_t e^{b_t+v_t/2−μ}}`（Markov on joint MGF，**无需独立性**） | sol15 eq.2 / astra06 §2：`INTEGRATION §4.2`（第 65 行）、`digest_astra-margin-lineage:113` | `[已验证-必要]` | 四家同式：`η_r = log((M_r−1)(1−a_r)/a_r)`（sol13）= `log(H(1−a_r)/a_r)`（sol12）= `−log N_r`（sol19）= `log|D_r|`（sol15） |
| **正确的标量目标** | 等均值等方差时正确目标 = **`v/2 − μ`**（绝对 log N 项，比 SNR 强）；pairwise SNR 可以「改善」而源质量恶化 | `INTEGRATION:65`；`digest_astra-margin-lineage:114` | `[已验证-推导]` | 充分条件 `μ−v/2 ≥ log N + log[ρ/(1−ρ)] + log(1/δ)` |
| **连续极限泛函（KKT 最优先复用件）** | `J[h] = (1/2)∫₀¹[α/h(u) + β(1−u)²h(u)]du`，`h>0`，`∫h=1`，`h=Q′(u)` | `NEXT_DERIVATION §4`（第 105 行） | `[已验证-推导]`（严格凸、唯一正解、`ρ″=τ²ρ` 全链条在 codex 核心 §4.1 复核过） | `1/h` 项 = **间隔 a_i 的凸惩罚**；离散化后 = 单纯形 `{a_i>0, Σa_i=A}` 上可微凸优化 |
| **Cosh 解（代理）** | `ρ_τ(x) = τ·cosh(τ(1−x))/sinh τ`，`τ²=β/α`；分位数 `Q_τ(u) = 1 − (1/τ)·asinh((1−u)sinh τ)`，`Q_τ′` 严格增 ⟹ 间隙向低频递增 | sol02 §3.2：`digest_calibration:107-112` | `[已验证-推导]`，**但只属代理 `C_app`** | 精确有限窗泛函**不存在 cosh ODE**（sol02 §3.6） |
| **精确有限窗核 + 原子均衡** | `K_L(φ,ψ)=[Ci(Lδ)−Ci(δ)+Ci(Lσ)−Ci(σ)]/(2 log L)`，`δ=|ω−ν|, σ=ω+ν`；对角 `K_L(φ,φ)=½+[Ci(2Lω)−Ci(2ω)]/(2 log L)` | sol02 §3.2：`digest_calibration:110-111` | `[已验证-推导]` | 精确泛函的极小是**有限原子**，非正 Cosh 密度（astra02） |
| **原子均衡 KKT 系统** | 势 `V_μ(ω)=∫K(ω,ν)dμ(ν)−qω²`，`c_μ=∫V_μ dμ`；最优：`V_μ* ≥ c_μ*` 于 I 全体、`= c_μ*` 于 supp；位置方程 `V′(x_i)=0`；**对偶间隙证书** `0 ≤ E_q(μ)−E_q(μ*) ≤ g(μ) = c_μ − min_I V_μ` | astra02 §2：`digest_astra-evq-finite:21-25` | `[已验证-推导]` | 原子数**无 L-一致界**（S=4 拉长可改变最优表结构）；无约束测度解 ≠ 有限 K 等权解 |
| **有限通道整数 DP（离散 KKT，可执行）** | `min_{n∈Z₊^B, Σn=K} Σ_i[(α/2Δ)n_i² + (βΔ/2)T_i² − K h_i n_i]`，`T_i=Σ_{l≥i}n_l`；`F_i(t)=(βΔ/2)t² + min_{0≤n≤t}{(α/2Δ)n² − K h_i n + F_{i+1}(t−n)}`，O(BK²) 全局最优 | astra06 §2 / sol17：`digest_astra-margin-lineage:118`、`digest_transport-operator:97-99` | `[已验证-推导]`（在其**声明的**共享 bin + 嵌套协方差下） | 协方差 `C_ij=(α/Δ)1{i=j}+β·min(x_i,x_j)` 是**声明假设**；**iid 逐通道噪声会杀死碰撞项** |
| **盒装优化（sol14，KKT 直接模板）** | `max_ν min_{r∈R_far} μ_r(ν)/√(v_r+ε_r²)` s.t. `μ_q−γ_q√(v_q+ε_q²) ≥ 0 ∀q∈R_native` | `INTEGRATION §4.2`（第 66 行）；`digest_failure-audits-2:206-208` | `[部分证据]` | **从未端到端运行过**；预注册门槛：先正确否决 Smooth 的 slot-28 符号反转（解析复算 +0.99946→−0.99954 已验证数学，sol13cx2）通过后才允许出表 |
| **认证比率（sol13，更稳健变体）** | `z_r = (μ_rᵀf_r − ε_r‖f_r‖ − η_r)/‖L_rᵀf_r‖`，`η_r=log((M_r−1)(1−a_r)/a_r)`；硬约束 `z_r ≥ z̄_r`（r∈R_Native） | `digest_failure-audits-2:237-241`；`digest_astra-evq-finite:75` | `[已验证-推导]` | Cantelli `P[Z≤η]≤1/(1+z²)`；完整生成需额外读出保持前提 |
| **输运算子（保关系重定时）** | `ν_c = ν_M + n(nᵀω_native/4 − nᵀν_M)/(nᵀn)`，`n∈{[1,−1]×15, [1,−2,1]×14}`；carrier-preserving 形式 `ν=(I−P_R)ω+S⁻¹P_Rω`，`P_R=R^T(RR^T)†R` | sol18/Astra05：`INTEGRATION §4.2`（第 72 行）、`digest_transport-operator:88`、`digest_astra-margin-lineage:69` | `[已验证-CPU]` | **角色资格线未过**：文件自标 `NO_ROLE_OR_CAPABILITY_QUALIFICATION`；**打破 `ν_j≤ω_j` 压缩-only 盒** |
| **守恒/单纯形约束** | `Δ=(Δ_23,…,Δ_39)∈R¹⁷`，`Δ_j≥0`，`ΣΔ_j=1`；`m_40−m_23=1`；可行域 = **16 维单纯形**（最后一个 Δ 由等式消去） | `NEXT_DERIVATION §1.3`（第 43 行） | `[已验证]` | 正确表述是 **17 个过渡 gap 的总跨度锁定 ln S**（原生 3.6697 + 额外 1.3863 = 5.0560 nats），**不是**「17 个 gap 之和 = ln S」 |
| **端点不变量 I1/I2** | I1: `m_j=0, j≤23`（高频恒等）；I2: `m_j=1, j≥40`（尾部精确 ÷S 平台） | `NEXT_DERIVATION §1.3`（第 41–42 行） | `[已验证=面板]` 但按 6Pro 修正为 **强基线设计约束，非零容忍定理** | 少数违例证明代价高，**不排除某个小位移自由度的存在**（`s28_less` 暗示槽 28 附近有 ~0.033 可回收量） |
| **scratch 分支密度规则** | `ρ*(x) = [C⁻¹h(x)]₊ / ∫[C⁻¹h(y)]₊ dy`（active-set KKT 当 positivity binding 时） | `INTEGRATION §4.2`（第 70 行） | `[已验证-推导]` | **不能直接裁剪代替 active-set 求解**（astra10 修正）；EVQ-Cosh 是其常数信号 + 嵌套慢尾协方差特例 |
| **冻结分支兼容性约束** | `D_src(x) = mean over natural source rows KL[p_native(ω;d) ‖ p(e^x;d)]`，约束 `D_src(x) ≤ D_src(x₀)` | astra08 §2：`digest_astra-margin-lineage:165` | `[已验证-定义]` | 「no worse than reference on this captured objective」，**不是**「safe vs Native」 |
| **Native 保留 trust region** | `½hᵀF_Nh ≤ ε`；无约束时闭式 `h* = −√(2ε/(g_LᵀF_N⁻¹g_L))·F_N⁻¹g_L` | sol08/sol11：`digest_failure-audits-1:58,65` | `[已验证-方程]` | 打包与共适应失败正是「保留必须是约束、不能是可交易的惩罚」的理由 |
| **四格反事实的交互项** | 交互项 `= F11 − F10 − F01 + F00`（K6 的判定表，零新增参数） | `INTEGRATION §6`（第 92 行）、`STARTING_POINT §7` | `[设计-预注册]` | 必须完整 prefill |
| **置换不变性（缺信息项的精确陈述）** | `E_q(μ)=½∫₁ᴸ p(t)f_μ(t)²dt − q∫ω²dμ` 只依赖**无标签频率测度**；冻结 checkpoint 依赖 slot↔有符号 Q/K 内容的配对；**置换 slot 配对不改 E_q 但改学习计算** | astra02 §3：`digest_astra-evq-finite:28` | `[已验证-推导]` | ⟹ **任何 slot 级中带规则不能由该信息单独导出** |
| **约束条件：中带 = 活跃约束集** | 两时钟特例：要求局部族精确保留 + 长程族精确重时 ⟹ 独占局部成分 `ω_j` 原生、独占重时成分 `ω_j/S`；**中带 = 冲突/耦合的活跃计算集合**，不是由单一旋转数决定的区间 | astra02 §3：`digest_astra-evq-finite:28` | `[已验证-推导]` | 兼容性冲突使联合需求**不可行** |

## C4. 三带结构：材料里已经存在的「KKT 解形状」证据

- **预言（待证）**：`NEXT_DERIVATION §1.4` 第 53–57 行——**「最优解形态自动是三段：Δ_j=0 的 bank 平台（∂F/∂Δ_j(0)>μ）、Δ_j>0 的过渡桥（边际率相等 = 水填充）、Δ 之后 m=1 的尾部平台（能力饱和）」，即「业界主流的分段规则 = KKT 解的结构形式」**。
- **数值涌现证据（sol04）**：(11)/(12) 的水填充 + isotonic 投影，「box [0,1] + isotonic give bang-bang ends with graded bridge ⟹ **三段结构 EMERGES（从未被强加）**」。出处 `digest_constructive:24,46`。solver 是 `max bᵀq − (τ/2)Σ(q_{k+1}−q_k)² s.t. ½Σ f_k q_k² ≤ ε_N`，阶保序 Q，`q_k(λ)=clip_{[0,1]}(b_k/(λf_k))` + λ 二分。`[已验证-推导（那个泛函）；任务链接 = 假设]`
- **导出证据（sol06 RTGA）**：`v* = (AᵀWA+ζQ)⁻¹(AᵀWb−c)`（ζ>0，`AᵀWA+ζQ ≻ 0`）；多项式 F 上加 active-set ⟹ 全局最优；**三段结构在 separated 情形被 DERIVED，其余 EMERGENT**；精确特殊情形 = 硬前缀 + 均匀 /S 后缀。出处 `digest_constructive:108-129`。`[已验证-推导（解存在）；目标改善能力 = 未证]`
- **反证/张力（sol07）**：贪心 donor→recipient 边际决策效用规则**不蕴含三段**——profile 就是存活的 transfer 序列；MrPro 面是起点 ⟹ 结构 = 继承 + 局部补丁（s28 = 单槽桥编辑）。出处 `digest_constructive:190`。
- **起点事实（F5）**：A/B 交点集合由两表交点定，不是人为阈值 ⟹ **K1 要证的是「为什么交点位置随 S 移动」**（F5 的 Qwen 38 → Llama 25 移动必须可从 KKT 条件的 S-依赖重 derive）。`STARTING_POINT §8.3` 第 105 行。

## C5. 可执行求解器谱系（四条配方，全部受「标签捕获数据门」阻塞）

出处 `INTEGRATION §4.3`（第 74–80 行），等级 `[设计]`：

1. **sol04** 水填充 + isotonic —— 三段结构在解中**涌现**（K1 定理的数值伙伴）
2. **sol05** 分层输运行 Pareto QP（`1ᵀδ=0` 持质心，镜像 ±ηδ* 对照）
3. **sol06** RTGA QP（关系目标 κ_n）—— 分离情形下三段被导出
4. **sol07** 贪心 donor→recipient 边际决策决策效用转移：`ΔM` 来自**完整前向**、`Q₀.₂` 分位、**每次编辑后重算**——这是 pair28_29 不可组合性（−4.17pp）的正确解释器
   - 校准件：astra04 修正映射 `f(bM+r)=(S−1)M+S·bM+r` 填满 131072、**98304 干扰槽**；padded-KL 恒等式；`L_bal` 块平衡 + 恒等映射 guard；SLSQP。sol01 预算化 `argmin C_T s.t. D₀≤δ`（δ 取 MrPro 实测）。sol03 等边际 KKT 律 + 回退梯子。

---

# D. 矛盾与口径不一致（材料之间 / 与权威文档）

| # | 矛盾点 | 双方出处 | 主文档裁决 |
|---|---|---|---|
| D1 | **非排序 veto 的定理外衣**：sol19/sol13/sol12 撤回「universal 不可辨识定理」vs 红线 R1 的表述 | `digest_failure-audits-2:322,356`（C6 RETRACTED、`[vetoed→rescoped]`）↔ `INTEGRATION:27`（R1） | **面板判决保留，引用措辞缩窄**（FLAG-1）。红线 R1 的**判决本身不动** |
| D2 | **保序约束地位**：sol19 冻结程序 (5) 硬编码 `ν_0≥…≥ν_{K−1}` vs sol12 记录一张**非单调**有用表 | `digest_failure-audits-2:404-408` | 保序 = MrPro 面的**刻面**，不是普适律（FLAG-2）；**报告未自行写明此调和**，不得继承为普适 |
| D3 | **Smooth 损失口径**：sol14「9.79 分更差」（聚合）vs 验证过的 near/far 行（near 打平、全部损失在 far） | `digest_failure-audits-2:410-413` | 一律 near/far 分解（FLAG-3） |
| D4 | **模型命名**：sol14/sol12「Qwen-1.5B@64K」vs 规范 1.485B | `digest_failure-audits-2:415-417` | 规范用 **1.485B**（FLAG-4） |
| D5 | **32K 拉伸位置能否作校准分布**：sol16 §step-4 可以 / sol17 §5 oracle-only（三个具名盲点）/ sol18 §3 禁止 | `digest_transport-operator:43,112,176,265-272` | **证据分层**，不平均，写标签（FLAG-5）。jsonl 事实（32K 响应住在冻结快槽 0–6）**站 sol17/sol18** |
| D6 | **16-DOF 坐标**：sol16 `ε(η)=softmax(log ε^Mr+Bη)`（Helmert，构造式保序）vs sol18 在槽 24–39 上优化 `x_j`（约束式保序） | `digest_transport-operator:269-272` | 同单纯形不同坐标；选 sol16 形式求解、sol18 坐标报值（FLAG-6） |
| D7 | **「总尺度守恒」是否坐标相关**：joint_mode `ΔΣm ≠ 0`（−0.01826…+0.00185） | `digest_transport-operator:272` | **限定而非反驳**：sol16 的「exact total scale」在 ε-空间为真、m-空间不真；**任何守恒律必须先点名坐标**（R4） |
| D8 | **E3_BM 命名**（Q7） | `digest_failure-audits-2:419-421`；sol12 的 Qwen7B「BM」是 boundary-matched 族，**未识别为 E3_BM** | **仍未解决；保留开放问题**（FLAG-7） |
| D9 | **MrUni 64.6 缺席** | `digest_failure-audits-1:87`：sol08–sol11 四份审计中**完全缺席**，无解释 | **digest 不得视为已覆盖**；middle-recipient 控制 70.14/67.36 是 **HighGapToLong 的控制**，不是 MrUni |
| D10 | **Smooth 是否可独立验证**：sol08 hedge「Smooth 不能从我的语料独立验证」vs sol10/sol11 直接档案检视 | `digest_failure-audits-1:88` | 按 T+N 类处理（**两次独立检视**）；sol08 只是缺覆盖 |
| D11 | **HighGapToLong 的 0/36 输出无法按名单独恢复** | `digest_failure-audits-1:90` | `[部分证据]` 级 caveat：仅对话级留存 |
| D12 | **astra01 与 astra06 的非恒定 h 强迫方程不同**：`αρ″−βρ=λh″` vs `αρ″−βρ=h″`（边界 `αρ′(1)=h′(1)`） | `digest_astra-margin-lineage:214` | 两者**内部自洽**；**F 采哪一项就固定哪条 forcing 律正确**——不是 veto，是集成项 |
| D13 | **astra06 的 bin 支撑是 `[Δ,1]` 不是 `[0,1]`** | `digest_astra-margin-lineage:148` | 任何集成必须**声明支撑约定**；不要把右端点规则叫作「endpoint-pinned RoPE」 |
| D14 | **s28 的 2 胜 0 负是 tiny-n 配对开发证据** | `digest_constructive:193`；`digest_failure-audits-1:89` | **不得把 83.3 当确认**，只作当前最佳方向假设 |
| D15 | **sol07 的 P/R 标签跨了两个列序，但和一致** | `digest_constructive:167` | CPU 复核 `−2.125+1.125+1.000+0.250=+0.250 ✓` |
| D16 | **solid 内部数字与旧文档不符**：astra02 修正 `EVQ_COSH_THEORY.tex:120` 的「Hilbert-Schmidt 投影」字面不成立、`:330-350` 的 `τ~K/√L` 由该投影推不出 | `digest_astra-evq-finite:63` | `[已验证-推导]`；`tau_static_vs_dynamic_experiment.py:71-83` 的 α 拟合口径有误（未减 min 核对角、非精确 Ci） |
| D17 | **astra09 与 16 增量面存在设计张力**：astra09 允许 j≤23 与 j≥40 的频率也动（盒内小步），偏离既定冻结面；astra03 明确保留该面 | `digest_astra-evq-finite:184` | 「实际频率步」与「抽象计数再分配」是**两个不同对象**，集成时须对齐 |

---

# E. 死路登记（已证伪/已失败，含失败原因 —— 绝不能再试）

出处合并：`INTEGRATION §5`（第 82–88 行）、`digest_failure-audits-1 §1+§7`、`digest_failure-audits-2 §1+§5`、`digest_calibration §4.3`。

| 机制 | 失败原因（机制级） | 关键证据 |
|---|---|---|
| Σcos 首零 / 根排序（任何变体） | 根与能力排序**显著失序** | MrUni 82.2K > MrPro 80.3K 而 32K 64.6≪87.2；E2/P2 同根 109.1K 一崩一 81.7；s28 +5.2pp 时根几乎不动 80.7 vs 80.3 `[已验证-CPU]` |
| 碰撞能 / 覆盖率 / 平滑度 / 有效秩 / response energy / movement-MAE | 把匹配信号与内容可混淆干扰**混池**；可以全部改善而生成变差 | Smooth_MrBudget 每个已审无符号族都改善却输 128K 面板（`digest_failure-audits-2:23-34`）；C2 MAE=0.001223 仍挂 native gate |
| scale-orbit 计数 | **不连续**（ULP 扰动 6→64 轨道、行为无差） | `digest_failure-audits-2:29` |
| iid 通道噪声 → `α∫ρ²` | 旋转不变 ⟹ **无频率依赖碰撞项**；`αI` 是频率场白噪声（`Var∫ρ dW = α∫ρ²`），**不是** K 个 iid 通道噪声（后者给 `K⁻¹∫v(x)ρ dx`，密度线性） | `digest_astra-margin-lineage:23`；`digest_failure-audits-2:78-79` `[已验证-数学]` |
| **任何无符号二次项入 F** | 同上（全灭类） | `INTEGRATION:84` |
| 冻结 checkpoint 上的**密度/多重集/排序**参数化 | 冻结 Q/K 坐标是**带标签**的；排序/置换毁掉干预 | 同多重集置换 OLMo NLL 3.10423→6.86493、Qwen core-4 0.70→0；OLMo log-s4 内置换 +3.760692 PG-19 NLL |
| scratch 密度移植到冻结 | 共适应改变统计对象 | Geo↔Cosh 运行时互换 PPL 7.14↔76.20 / 7.16↔23.05；weights×table crossing 3.400/3.251 |
| 解①对象装④对象（Astra02 类错误） | 四个对象**两两不可互换** | `INTEGRATION:69`、`digest_failure-audits-2:297` |
| 「守恒律」不点名坐标 | joint_mode 零和频率下 ΔΣm 仍 ≠ 0 | `digest_transport-operator:88,272` |
| Taylor/Jacobian 分数跨全 S=4 表 | 多次相位缠绕；线性预测相对误差 71–468%，相位 22.74/90.97 rad | `digest_failure-audits-2:52-54` |
| 逐槽可加性 | 共享表的导数含跨 key 带符号相干与共享 head/W_O 对消 | pair28_29 −4.17pp；同非负能量总导数 0 vs 4 |
| slot-19「Fisher/Hessian」曲率 | **无 artifact**；`Fisher≠Hessian`、`MAE≠L1` 是数学错误 | `digest_failure-audits-2:323` |
| 「YaRN 递减 vs MrPro 递增」 | F1/F2 证伪：**两者都凸都递增**（`m_Y′>0, m_Y″>0`） | `STARTING_POINT §1-2`；`INTEGRATION:87` |
| MrRoPE 首零点 → 内剖面推导 | 论文**自设**等差 radix；边界是**经验选择**；一张早期有用表**非单调** | `digest_failure-audits-2:81-88`（L7） |
| universal 1×–2× 交换率 | 7 个**异构协议**混排（架构/指标/gain/regime） | `digest_failure-audits-2:132-134` |
| 六观测量充要 | **无充分性证明**；category mixing | `digest_failure-audits-2:324` |
| VICTORY CONFIRMED / teamwork_preview 作闭合证书 | 与其共存的无支撑声明**内部不一致** | `digest_failure-audits-2:326` |
| `τ ≈ d_head/√L` | 「PASS」= **脚本约定**（15 个人工锚点，均值相对误差 9.6%、最大 33.3%）；HS 投影下实际给 `α=O(1/n)`、`τ~√n` | `digest_astra-evq-finite:27,63` |
| 支持域重定标的**机制**故事（公比对称性） | **代数否定**：共同标量乘法对 uniform 与 nonuniform z 都保持相邻 log-区间比；反转是**观察不是机制** | `digest_failure-audits-2:325`（C9） |
| universal 不可辨识**定理**（as stated） | 证明缺口被逐条枚举；只约束置换不变映射 / ULP 不连续轨道计数 | `digest_failure-audits-2:322`（C6） |
| 候选生成空转 | 无判别性测量；16K assay 触底 0/32–1/32；8×8 面板措辞敏感 | `digest_failure-audits-2:151-155`（L13） |
| 直接优化（64 维行为梯度；direct-z 定支撑） | 前者**未开 holdout 即败**；后者挂声明门；改名 signed gradient/MGDA/QP 不修缺失的总体定义 | `digest_failure-audits-2:157-163`（L14） |
| attention 代理 → 生成（E7/E8） | cross-cache：BM 读 MrPro 前缀能对，MrPro 读 BM 前缀仍错；E7 局部保持 −9.514pp；E8 零槽 51 −13.889pp | `digest_constructive:157-158`；`digest_failure-audits-2:98-105` |
| HighGapToLong（「从高抽、给低」） | 捐赠量子 + 接受者构造双败；**middle-recipient 控制也败**（70.14/67.36，0W/7L） | `digest_failure-audits-1:20` |
| LongBridge（组相位平移） | 只测相位原点/方向，**不是组内分辨率分配**；Slower −6.67/+1.94，Faster 不对称 | `digest_failure-audits-1:21` |
| P2 作普适规则 | Qwen-3B 32K 72.92 vs matched-gain 98.33；跨 checkpoint/任务反转；cross-model 不平稳（2× ≠ 4×/8× 律） | `digest_failure-audits-1:22` |
| LeRoPE `ρ ∝ w^{1/3}` | 结构性 softmax 曲率下**落在 EVQ 之外**（α=−0.957） | `digest_failure-audits-1:26` |
| 假设距离先验的 universal 密度 | 逐频加性效用**把全部通道塌到同一频率**（除非加干扰）；排序随核选择互换 | `digest_failure-audits-1:27` |
| Arcsine（scratch note） | kernel-to-potential 论证无效；数值形状**非 U 型** | `digest_failure-audits-1:28` |
| 「任何改变的静态表都精确无短程伤害」 | 只有**原生频率多重集**（至别名）允许；近似无伤害是**风险约束** | `digest_failure-audits-1:29` |
| Emulation rank bounds → 再适应推断 | 需要 loss-local 模型 + 联合 Q/K 预算 | `digest_failure-audits-1:30` |
| 「慢对儿被 softmax 抑制」 | softmax 只去**行常数**，不去随 key 内容变化的近距离无关项 | `digest_calibration:158` |
| 「位置函数冗余 ⇒ 冻结槽便宜」 | Q=K=I_{2K} 时内容核秩 2K ⟹ 冗余位置可携独立内容通道 | `digest_calibration:157` |
| 「静态标量 gain 解决局部/长程相位冲突」 | gain 改 logit 尺度，**不改同时相位约束** | `digest_calibration:159` |
| 「同一解析分配同用于 scratch 与 frozen」 | 被有序槽耦合与安装史分解否定 | `digest_calibration:161` |
| `αI` 作精确 K 的替代 | `L²[0,1]` 上恒等算子**非 Hilbert-Schmidt** ⟹ `‖K_L−αI−βG‖_HS` 对任何 α≠0 为 ∞；字面连续 HS 投影**不存在** | `digest_astra-evq-finite:27` |
| 高阶波数截断展开 | `⟨ρ,h_c*ρ⟩` 的高波数项**下无界**（含端点项）⟹ **不安全** | `digest_calibration:116` |

**同时被否决但方向「被 rescoped 而非死掉」的**（不得当作全死）：`[vetoed→rescoped]` 根非排序 veto；labeled signed-margin allocation 本身（sol19+sol13+sol12：constructive frequency allocation **remains OPEN**）。出处 `digest_failure-audits-2:356`、`:442`。

---

# F. 未解问题清单（材料自报的 OPEN 项）

1. **Q9 / L_far 的可计算形式**：`L_far` 不能按单槽弧语言定义，须按**联合谱覆盖/风险带**定义——**兼容成本的可计算形式未完成**。`NEXT_DERIVATION §1.4` 第 51 行。
2. **K1 未证**：交点位置随 S 移动的推导（Qwen 38 → Llama 25）。`INTEGRATION:120`。
3. **K2 不可识别性风险**：14 点能否被任何 (L_near, L_far) 单调参数族排序——若不能，**如实报告并给出所需的最小新实验集**。`NEXT_DERIVATION §6`（第 121 行）。
4. **标签化角色矩从未被测**（见 A7-1，唯一阻塞证据）。`INTEGRATION §8-1`。
5. **E3_BM 命名**（FLAG-7）—— 保留开放。
6. **支持域重定标的机制**：观察已验证，**解释被拒**，机制仍 OPEN `[假设]`（C4）。`digest_failure-audits-2:320`。
7. **`α,β` 未被任何桥识别**：Astra05 明确 `α,β` of the Cosh surrogate 不被该桥识别；Cosh **不因此**成为 M̄ 的精确最优。`digest_astra-margin-lineage:65`；与 `03_theory.tex:74-110` 一致。
8. **真实 Qwen 的 `h, C, R` 不可得**：Astra08 角色交换阻碍 ⟹ 下一决定性动作 = **带标签源窗捕获** + 先测「margin moments 是否分离现有 MrPro/P2/BM/Smooth 面板结果」。`digest_astra-margin-lineage:221`。
9. **`t_n` 需要逐关系干预；`c` 需要全网带符号梯度**（sol06 是四份里输入最饥渴的）。`digest_constructive:132`。
10. **Astra09 全槽自由度 vs 16 增量面不相容**：须投影到增量面或放弃盒规则（分歧点）。`digest_astra-evq-finite:184`。
11. **`gain×相位不正交`**（6Pro 第 6 点）：F 不含 gain 自由度，但 gain 与相位的耦合机制本身未解。`INTEGRATION §7`（FLAG-6 相邻）/`NEXT_DERIVATION §5` 第 5 条。
12. **Σm 质心是自由决策变量**——「搬多少」的第一刀（K6 四格）尚未跑。`INTEGRATION §8-2`。
13. **G1 地面真值表验收未完**：脚本 342 行 Pyright 假阳性待清；3 项 MISMATCH（BUDGET §3 队列候选 max 洞数值不可复现）。`INTEGRATION §8-4`、`tables/GROUND_README.md §5`。
14. **`max_new_tokens` 顶层字段与实际行长核对**（astra10 线索）—— 先行小项，未做。`INTEGRATION §8-4`。
15. **LeRoPE 先行工作占位**：任何「梯度法」表述必须带 arXiv 2607.10134 §3.2。`INTEGRATION §8-5`。
16. **workflow-1/2 产物缺失**（本 digest 核查）：`proposal_A/B`、`verify_*`、`answers/D1–D4`、`tables/CANDIDATE_TABLES`、`answers/T2/T3`、`answers/V_*` **在磁盘上不存在** ⟹ 交付表第二条腿今天无输入。

---

## 附：本 digest 的核心一句话

**已裁决的边界**：几何/无符号/静态代理（R2）、根（R1）、Σm 守恒的坐标依赖性（R4）、端点作设计面（R3）、四个对象的不可互换性、两 regime 的分离、以及「无新 Qwen 表被当前语料授权」——**这些都不需要也不能在今天重新论证**。
**留待推导的只有 K1–K6**，其中 **K1（三段结构的 KKT 定理）+ K6（四格反事实）** 是 INTEGRATION §9 明写的「完成条件」的最低门槛。

——完——
