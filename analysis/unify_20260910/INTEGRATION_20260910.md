# 统一推导总整合：62 个代理资产的裁决与下一次推导入口

日期：2026-09-10。状态：汇总文档（推导准备材料，不是推导本身，更不是执行指令）。
本文合并两条独立工作线的全量阅读成果：

- **本方线**：32 个子代理（16 份会话/代码/结果 digest，`digests/`；另有 20260909 夜间 8 会话与 0910 批量 16 会话的回传摘要）＋ 本轮两个进行中 workflow（提案 A/B＋验证器；地面真值表→四问→候选表→终审）。
- **codex 线（thread 01a0806f，只读跟踪）**：30 代理归档（28 报告＋27 回执＋sol20/astra10 两份回传整理），已 7 份 digest 全量读完（`digests_codex/`），仓库权威副本在 `docs/research/rope_allocation_20260910/`（codex commit 525dc15）。

阅读优先级：**本文 → `NEXT_DERIVATION_KKT_PROBLEM.md`（问题最终模板与 K1–K6）→ `STARTING_POINT_YARN_VS_MRPRO.md`（F1–F9 权威起点）→ 各 digest**。任何单一 digest 与本文冲突时，以本文为准；本文与一手报告冲突时，以一手报告＋当场重算为准。

---

## 1. 问题陈述的最终形态（用户权威重构）

**我们根本不是搬运。** 搬运（transport）只是本理论视角的产物——它给出水床不等式。业界最强的是**分段规则**：高频不动、中频过渡、低频 ÷S。真正的问题是：

> RoPE 这个**非均匀离散傅立叶通道系统**，在有限 K 个等幅旋转槽、总压缩预算 Σ_j m_j = log S / log S（即 Σm 固定于过渡段）之下，**是否存在最优解**？高频（native/近距离）损失最小，低频获得最强能力——若最优性成立，业界分段形态应当是 **KKT 解被推导出来的形状**，而不是前提。

四子问题的当前最佳答案（详细推导输入见 `STARTING_POINT_YARN_VS_MRPRO.md` §8；裁决依据 §5/§6）：

1. **高频的冗余在哪里**——**D1 判决**：分两个意义。(i) **弧覆盖冗余＝100% 但兑现价值＝0**：j≤39 全部 r_j≥1.15、训练弧覆盖率 c_T=1，m>0 买不到任何弧安全——EVQ 式"压缩高频腾预算"是无对价出售（HighGapToLong 0 升 7 降）。(ii) **密度冗余（EVQ α∫ρ²）是重学习概念**，冻结下无定义（有序表反例）。真实自由预算只有一处：**MrPro 自压在 bank 边缘的 Σm(24..28)＝0.2288 ln S**，已兑现 0.0326（s28→+5.21pp@128K、32K 零成本），其余 0.196 的兑现以"落点洞 ρ≤1.44"为前提、整块未测。bank 核心 j≤23 零自由度（且**无单槽测量**——D1 点名的证据缺口）；尾部零冗余（E8 删一条 −13.89pp/12 行）。旧表述"改动阶次 O(N⁻²) vs O(N⁻¹)"保留为 D2/D4 的进入阶论证（T3 定理 3）。[answers/D1_high_freq_redundancy.md]
2. **中频为什么关键**——**D2 判决（三重合）**：中频带 24–39 ＝ 唯一自由度带（I1/I2 之间）∩ 风险距离支撑带（D_j=L·4^−(1−m_j) 把全部缺口精确映进 89–127K 证据行区间）∩ 双重端约束交汇（窗内容忍——5 个桥形不同的表 32K 分数逐位同 87.2222；bank 敏感——s28 的 0.29 圈相位差就能翻 mk_2@89K）。D2 并交付**双端风险泛函** R=U+αH+βΦ：5 面板点严格排序可行、参数被钉进 0.04% 体积的近唯一细缝（ρ₀∈[1.370,1.386]、p≈0.5、α∈[3.70,4.02]、β∈[0.55,1.33]）、三承重墙各被一对相邻序钉死；留出 12 点 τ=+0.576、**幅度不外推**（定位为排序描述子，C5 合规）。[answers/D2_mid_freq.md + D2_fit5]
3. **低频需要多少资源**——**D3 判决：需要的是资格而非预算**。尾部内部 gap 对预算贡献 −6.5e-9≈0；ln S 只在路径无关的坡里花一次。覆盖 128K 需要 ⌈log_ρS⌉=7（含绕回余量 8）条互异慢钟——由 ÷S 原生梯重放**免费**供应（原像带 (8192,40663]＝槽 34–40）。真实需求＝全部 24 个尾槽**恰好 m=1**：E2 过冲（c=1.1557）→ 错位 6362 位置＝18.9%W、Δθ₄₀=1.13 rad、−9.7pp/12 行；E8 删**一条**钟 → −13.9pp，且 E2/E8 断同两行（共享必要通道）。非恒定尾被联合码 R1（块内相对比值保持）判死——§2.2 证明非常数 m 破坏它。[answers/D3_low_freq.md]
4. **从哪里搬、搬到哪、搬多少**——**D4 判决（推导出的分配规律＋表）**：盲目标（位置无关洞罚）的解恰是均分桥＝MrUni（已崩 64.58），**盲目标连同 argmin 一起作废为错设证据**；洞必须按所在 T 带计价（bank 权重⇒水填充要求 ε_g 左低右高——**右载 ramp 方向被导出**，与业界 Eq.14 形状同构）。**规则不引入新形状：保持 Eq.14 径向 ramp，只把完成边界 dh=40→39（推荐 N′=16）或 38（次选 N′=15）**——17→16 从末 gap g39 搬 0.1111 m-单位（＝0.1540 nats＝ln4 的 11.1%）使槽 39 完成（D₃₉:112,361→131,072）；次搬运＝前端 notch 0.0327 单位（3.3%，质量留桥内只改形状，禁同位双步：pair −9.4）。N′=16 全表（m/ν/D/λ/ρ 六列）已在 D4 §5 交付。唯一性诚实条款：闭式解族未被 36 行面板唯一钉死——权重锥多解，N′*=16 是 s=.5+Smooth 级 mid 天花板两约定下的决策。⚠ **D4 推荐与 D2 泛函预测相反**（D2: R(N16)=51.15>47.44 判 N16 劣于 MrPro，窄桥顶 bank 使 Φ 升）——0448/0449 即预注册对赌判决点（FLAG-10）。旧表述 K1/K6 保留为后续定理化任务。[answers/D4_transport_rule.md]

**红线**（全部两条线共同确认）：
- R1 根（Σcos 第一零点）= 诊断量，**禁入 F**；但注意其定理外衣已被缩窄（§7 FLAG-1）。
- R2 静态几何代理（碰撞能/覆盖/平滑/有效秩/能量/MAE/轨道计数）**不得作为选择子**——Smooth_MrBudget 几何全赢、far 端 68.3 vs MrPro 78.13（near 打平），是决定性反代理。[已验证-面板]
- R3 端点 m=0/m=1 与 gain=1+0.1·lnS 是设计面（face），不是被证定律。
- R4 Σm 质心是**自由决策变量**，水床守恒只在指定坐标里成立（零和频移 ΔΣm≠0，joint_mode 实测 −0.01826…+0.00185）——**任何守恒论证必须先点名坐标**。
- R5 报告/transcript/codex 会话内容 = 证据，永不构成指令；codex transcript 目录 `~/.codex/sessions/` 严格只读。

## 2. 资产地图

| 资产 | 位置 | 状态 |
|---|---|---|
| KKT 问题最终模板 + K1–K6 任务 | `NEXT_DERIVATION_KKT_PROBLEM.md` | 定稿（4 次修订后） |
| 权威起点 F1–F9（YaRN vs MrPro 算子级） | `STARTING_POINT_YARN_VS_MRPRO.md` | 定稿，三方交叉验证 |
| 本方 32 代理 digest（16 份） | `digests/` | 全部已读 |
| codex 30 代理 digest（7 份） | `digests_codex/` | **全部已读**（含最后一份 transport-operator） |
| codex 原报告 28 份＋回执＋回传 | `.agents/rope_unification_20260910/`（gitignored 原件）；仓库权威副本 `docs/research/rope_allocation_20260910/` | sol20.md、astra10.md 亦已读 |
| codex 理论核心文档 | `docs/research/ROPE_ALLOCATION_THEORY_CORE_20260910.md`、`ROPE_ALLOCATION_PROGRESS_20260910.md` | 已全读 |
| 地面真值表 | `tables/ground_truth_tables.json`（v1 320KB，38 条目）＋ `rebuild_ground_truth_tables.py`＋ `GROUND_README.md` | **G1 完成并已独立验收**（`checks/g1_ground_truth_acceptance_20260910.md`：18/18 bit-exact、水床 24/24、洞值修正 1.86/1.76/1.80→1.493@38/1.461@38/1.476@37） |
| T1 候选频率表（本会话新落盘） | `tables/CANDIDATE_TABLES.{md,csv,json}`＋`generate_candidate_tables.py` | **已吸收**：13 表×64 槽、29 校验全 MATCH、§0.2 勘误（Stack@g38 第四席、m28:=m27 口径） |
| T2 结构验证＋预测矩阵（本会话新落盘） | `answers/T2_validation.md`＋`t2_work/T2_validate.{py,out}` | **已吸收**：恒等式全绿、U 矩阵、τ=0.778、K1–K9、R1 均分桥防呆 |
| 提案 A/B＋数学/证据/可行性验证 | workflow-1（wf_c2d1c25c：25 启动 / 16 回传） | **完成**。proposers 双双失败（prompt 过长/6 次停摆）→ 三验证器直查 BUDGET/UNIFIED 规范载体；报告 `verify_math.md`/`verify_evidence.md`/`verify_feasibility.md` 已全部吸收，11 阻断＋5 可行性修正已于本日应用到两文档（文内"验证修订记录"） |
| 四问推导 D1–D4、候选表 T1–T3、终审 V×4 | workflow-2（wf_f8c14342） | **V×4 终审已落地并回收（2026-09-10 11:00，台账 `checks/V4_FINAL_VERDICTS.md`）**：D1/D2/D3/D4/T3＋T1＋T2 均已落盘并吸收进 §1/§3/§7/§8/§9。V×4 回收状态：journal 现 25 行，**仅 tables 席（a6b5cebdaf30c237e）有官方裁决**（8 项 claim＋2 项 blocking；回收席重跑 `checks/v_tables_check.py`＝261 OK/0 FAIL，[已复核]）；math/veto/predictions 三席转录中断、无官方裁决，结论一律标**[重建-非官方]**（veto 实质零产出；math 席崩于 T3 四个极限定理之前未执行；任务书口径 math/feasibility/evidence/tables 与实际在册维度 math/veto/predictions/tables 不符）。重跑三席（veto ac82fd9e5ac62af37、predictions ae2ce49a3ae7e47e7、math a9fcda922ac40e803）11:08 快照仍在途、无 result **[V4_FINAL_VERDICTS.md §0]**。〔刷新 2026-09-10：重跑三席**已全部落盘**——V_predictions(11:20)/V_veto(11:30)/V_math(11:50)，四席官方裁决在册，本行 [重建-非官方] 口径仅对 11:08 快照有效；UNCONFIRMED 九项对账见 FLAG-11 更新、吸收总记录见 §8-9〕 |
| V×4 终审回收台账（本会话新落盘） | `checks/V4_FINAL_VERDICTS.md` | **已落盘**：四席裁决回收＋独立现算复算（逐项标 [已复核]/[未复核]）；新发现＝G1 `fast_band_bitwise_equal_native` 死旗标、D4 §4 mid 带界裁决（g29–35，文档自洽）；UNCONFIRMED 九项清单（§6）回主循环 |
| 原始 transcript 快照（本方） | `raw/`（2.9MB，只读镜像） | 留档 |

## 3. 地面事实（钉死数字，两条线共用）

部署对象：Qwen2.5-3B，W=32768，S=4，L=131072，θ=10⁶，K=64；ν_j=ω_j·S^{−m_j}；ω_j=θ^{−j/64}（**zero-based**，末对 θ^{−63/64}≈1.24e-6，非 θ^{−1}——sol16 精确约定）；自由增量 Δ_j 在槽 24–39（16 个），j≤23 m=0，j≥40 m=1；gain g²=1+0.1·ln4=1.1386294。

面板（36 行，near/far；12 行筛选面板另注，两口径**禁止直比**）：MrPro **87.22/78.13**；s28_less 87.2/**83.3**（已知最强 far；+5.2 全部来自 2 行开发翻盘 2/0 W/L）；LBS 80.6/80.1（3W/3L；同时翻好 mk_2@89K **并改坏** mk_2@96K）；P2 72.9/81.7（6W/6L，short −14.3）；MrUni 64.6/73.3（**过渡段线性斜坡、端点逐位同 MrPro**——"全表÷4"标签不实，是桥形失败非端点违例）；Smooth 87.2/68.3（R2 反代理；机制行已反向更正——抬 m36–39、从 bank 抽、毁中程梯级）；pair28_29 **−4.17pp**（逐槽可加性已死；交互 −9.17pp）；HighGapToLong −17.1/−10.8（**I1 侧唯一被测端点违例**）；E2 54.7、E8 50.6（**12 行面板，基线 64.4444**——从未与 78.125 比较过；E2 有效尾比÷4.9638 非÷4.93；E8 槽 51∈尾带→**I2 侧**证据）；gain074 行 **98.3/75.3 是 Control_Mr_gain074**（旧误标"E3 gain074"）；E3_BM_gain074＝100.0/70.0、E3_BM_gain1＝89.58/58.82。OLMo-2-0425-1B 16K 350 条：BM 41.67% vs MrPro 7.09%（+34.59pp，156W/9L；multikey_3 双方皆 0，多重绑定未解决；EOS BM119/MrPro197，评分口径不得混用——sol20 回传）。prefix/read 交叉：−2.125/−1.125/−1.0/+0.25（收益不限于固定 Q/K 读出）。32K 全模型 CE 梯度 4 行（`full_model_response_native.jsonl`）：‖grad‖ 50.9–688.7，**响应集中于被 MrPro 面冻结的快槽 0–6**，过渡槽 24–39 几乎无响应——128K 过渡带问题需要真实长相位（sol18 实测，支持 sol17 的分层立场）。128K response 终态 **UNKNOWN——使用前必须先收回执**（progress 文档纪律）。

**T1/T2 新增地面事实（2026-09-10 晚，全部 [已验证]=CPU 复算）**：
- **T2 (a) 恒等式全绿**：12 张闭式表（ramp N′=13..17 ⊕ 均分族 ⊕ notch 变体）Σ桥=ln4 残差 ≤6.7e-16；ρ_max 恒等式任务口径 `4^{1/N'}` 少乘原生因子——正确式 **均分族 ρ_max=ρ_nat·4^{1/N′}、ramp 族 ρ_max=ρ_nat·4^{2/(N′+1)}**（闭式核验偏差 ≤4.4e-16）。ramp 族 ρ_max 梯子：1.5127/1.4929/1.4757/1.4608/1.4476（N′=13→17）[T2 (a)4]。
- **T1 Δ1（第四独立席位确认）**：Stack max 洞值 1.4930 无误、位置 **g38 非 g35**（GR §6/D1 §5/D2 c3 位置标签笔误）；G1 Stack 数组 m28 比 m27 低 3.8e-9 为 **fp32 反演伪影**（s28 构造两槽应相等），交付表取 m28:=m27；T2 (a)2 同判。**Stack 洞集与 LBS 逐位相同**，差异只在 g27→1.2409、g28→1.3710——D1 §6"Stack g35 洞 1.493>1.459"前提作废，0446 洞超调判定按此修正（不改主判据）。〔更新 2026-09-10〕V×4 predictions 席重建＋回收席现算再次**[已复核]**确认该口径：Stack vs LBS 逐 gap ρ 仅差 g27（1.2409 vs 1.2984，Δ−0.0575）与 g28（1.3710 vs 1.3103，Δ+0.0607），m 数组仅差槽 28；argmax=g38、1.4929753892033155（G1 存储字段同值）[V4_FINAL_VERDICTS.md §1-3、§4]。
- **覆盖 U 合计**（T2 b1，桥内弧钟暴露数）：MrPro 17 / s28 17 / LBS 13 / Smooth 13 / **P2 0** / N16 14 / N15 10 / Stack 13——三候选 U 全部低于 MrPro，方向与赢家机制（后端完成）一致；P2 的 U=0 是其长端赢与短端崩（1.088 巨洞+完成过快）共同的结构底座。
- **排序复现独立验收**（T2 c1/c2）：D2 发表点 (ρ₀,p,α,β)=(1.372,0.5,3.920,1.234) 在我方特征下 4 条不等式裕量复现到 2e-3；Chebyshev 可行窗 18/217 格与 D2 窗一致；8 方法（含 3 留出臂）Kendall τ=+0.778，漏 3 对（LBF/MrPro 参数敏感、pair/MrUni 过罚）——与 D2"局部排序描述子、幅度不外推"定位一致 [已验证]。32K 端 (Φ,H₃₂) 拟合 max|resid|=3.93pp<一行（4.17/8.33），Smooth/pair 双反例证明**洞必须按所在 T 带位置计价**（D4 §2 独立复现）。
- **HGL 快带例外（V×4 回收席 [已复核]，2026-09-10 11:00 新增）**：HighGapToLong 槽 1–23 的 m_j **全部偏离原生**（m1=−0.006770 … m23=−0.1557154；HighGapToMid 同构造、`endpoint_delta_m.m_23` 双臂同值），GROUND_README.md:101（§4-5）"全部面板表 fast 段（槽0–23）与原生逐位相同"的例外清单**漏列 HGL/HGM**。回收席对 G1 存组 `nu_j`→f32 与部署 Native 逐位比较进一步裁决：真实 fast 段例外清单＝**四臂 {HighGapToLong, HighGapToMid, FullLagP2_Transfer3B, NTK_static}**（26/30 等、4/30 不等），**YaRN 两变体 fast 段实际逐位等于 Native**（README 将其列入例外属过度排除）；另 G1 字段 `endpoint_delta_m.fast_band_bitwise_equal_native` 为生成器死旗标（`rebuild_ground_truth_tables.py:111` 拿 24 元切片比 64 元全数组，含 Native 在内 30 个数组方法恒 False），**不得作表体正确旁证**。[V4_FINAL_VERDICTS.md §1-5、§3-①]

## 4. 幸存理论核心（KKT 的零件清单）

### 4.1 sol15 §7 统一陈述（五份审计报告共同背书的可防御措辞）

> RoPE 设计 = 在有限个等幅旋转通道上分配频率，同时选择保留/重定时哪些位置分数关系。对声明的 source/competitor 总体，作用量是表格与其学习内容系数产生的**带符号 log-partition margin**。在交换式从头设计模型中（信号与分配无关、共享相干局部＋嵌套 nuisance 协方差），最小化 softmax 失败界归约为 EVQ 代理、连续极限给出 Cosh；其有限分辨等计数版本是**整数分配问题**。在冻结 checkpoint 中必须保留通道标签与学习系数，分配是**带约束的有限输运问题**。MrRoPE 提供结构化的 native 相对膨胀族；带角色限定的谐波约束可证成精确混合模式重定时。**这些受限归约中没有一条确立与 checkpoint 无关的 LM 最优曲线。**

这段话就是统一理论当前可交付的全部；KKT 推导的任务不是推翻它，而是把"声明的总体"变成**可测对象**、把"受限归约"之间的桥变成**定理或反例**。

### 4.2 逐件零件

- **共同精确对象**（sol15 eq.1）：每行 z_t(ν)=c_t+Σ_j{A_tj cos(ν_j d_t)+B_tj sin(ν_j d_t)}；角色 margin M_r(ν)=log Σ_{S_r}e^z−log Σ_{D_r}e^z；单源 p*=σ(M_r)。**带符号、按槽标签、带内容系数**——三缺一即死（L1/L2/L4）。
- **概率证书**（Markov 于联合 MGF，无需独立性）：Pr[p*<q] ≤ min{1, q/(1−q)·Σ_t e^{b_t+v_t/2−μ}}。等均值等方差时正确标量目标是 **v/2−μ**（绝对 log N 项，比 SNR 强）；pairwise SNR 可以"改善"而源质量恶化。分母/稀释项四家同式：η_r=log((M_r−1)(1−a_r)/a_r)（sol13）= log(H(1−a_r)/a_r)（sol12）= −log N_r（sol19）= log|D_r|（sol15）。[已验证-必要]
- **sol14 冻结分支的盒装优化**（KKT 的直接模板）：max_ν min_{r∈R_far} μ_r(ν)/√(v_r+ε_r²) s.t. μ_q−γ_q√(v_q+ε_q²) ≥ 0 ∀q∈R_native；F 定端点/支撑/保序/声明位移限；Cantelli 给失败概率界。**从未端到端运行过**；其预注册门槛（先正确否决 Smooth 的 slot-28 符号反转：解析复算 +0.99946→−0.99954 已验证数学，sol13cx2）通过后才允许出表。[部分证据]
- **水床的幸存形式**（sol17）：固定对数 Σn_i=K＋嵌套协方差 C_il=(α/Δ)1{i=l}+β·min(x_i,x_l) 下精确 DP：min Σ_i[α/(2Δ)n_i²+(βΔ/2)T_i²−K h_i n_i]，T_i=Σ_{l≥i}n_l，后向递推 F_i(t)=(βΔ/2)t²+min_{0≤n≤t}{(α/2Δ)n²−K h_i n+F_{i+1}(t−n)}，O(BK²) 全局最优。**βΔT_i²/2 就是水床写成守恒律**（质量推向慢端 ⇒ 每个远尾碰撞负载上升）；iid 通道噪声会杀死该项——水床要求**声明的相干/嵌套协方差**。常数 h 归约出离散 Cosh 递推；源对齐振荡 h 产出 Mr 式正的远端响应。
- **连续极限**：J[h]=(1/2)∫₀¹[α/h(u)+β(1−u)²h(u)]du，h=Q′(u)；离散模拟 = 单纯形 {a_i>0, Σa_i=A} 凸问题。**KKT 最优先复用件**（NEXT_DERIVATION §4）。sol16 的 softmax-Helmert 参数化 ε(η)=softmax(log ε^Mr+Bη)（B 为 17×16 正交零和 Helmert 基）把冻结 MrPro 面变成 η∈R¹⁶ 无约束坐标，η=0 逐位等于 MrPro——**优化器的现成坐标系**；CPU 参考实现 gradcheck PASS。
- **原子性与四对象分离**（astra02/sol15，最被一致同意的数学事实）：精确有限窗口碰撞泛函的唯一测度最优是**有限原子**，不是正 Cosh 密度；Cosh 只从 delta-plus-min(αI+βG) 代理流出。四对象——①精确-原子最优 ②光滑-Cosh 代理 ③有限-K 整数计数 ④冻结-带标签表——**两两不可互换**；解出一个装上另一个是类错误。原子定理也**不**迁移到有限整数-lag Gram（唯一性可失），也不意味着 LM 应该重复频率。
- **从头分支密度规则**：ρ*(x)=[C⁻¹h(x)]₊/∫[C⁻¹h]₊（active-set KKT 当positivity binding 时——**不能直接裁剪代替 active-set 求解**，astra10 修正）；EVQ-Cosh 是其常数信号+嵌套慢尾协方差特例（h=h₀, C=K_exact）。
- **可辨识性边界**（astra07）：自由系数规差下任意密度可被吸收（a₂=ρ₁a₁/ρ₂ 保持全部已实现 margin）；α∫ρ² 需要共享噪声＋等载荷 w∝ρ；独立噪声给不同 Neyman 律 ρ*∝σ|w|。**"唯一最优密度"式声明一律降格为"给定声明协方差下的解"。**
- **联合模式输运算子**（sol18/Astra05）：ν_c=ν_M+n(nᵀω_native/4−nᵀν_M)/(nᵀn)，n∈{[1,−1]×15, [1,−2,1]×14}（29 个 CPU 候选，全部零和⇒保 Σ频率、端点逐位固定、严格递减）；关键实测：**MrPro 对 29 个相邻关系时钟的 0/29 做了 ×4 重定时**（ramp 重定时的是槽，不是低阶关系时钟）；3/29 候选把某槽推得比 MrPro 还快（框内允许）——**null band 假设被联合模式否定：重定时一个关系同时保正交载体，有时要求加速，纯压缩斜坡排除了它**。carrier-preserving 形式 ν=(I−P_R)ω+S⁻¹P_Rω 打破 ν_j≤ω_j 的压缩-only 盒。[已验证-CPU；角色资格线未过：文件自标 NO_ROLE_OR_CAPABILITY_QUALIFICATION]

### 4.3 求解器谱系（四条可执行配方，全部受"标签捕获数据门"阻塞）

1. **sol04** 水填充+isotonic——三段结构在解中**涌现**（K1 定理的数值伙伴）。
2. **sol05** 分层输运行 Pareto QP，1ᵀδ=0 持质心，镜像 ±ηδ* 对照（与 sol18 的 equal-norm mirror 同族）。
3. **sol06** RTGA QP（关系目标 κ_n）——分离情形下三段被导出。
4. **sol07** 贪心 donor→recipient 边际决策效用转移：ΔM 来自**完整前向**、Q₀.₂ 分位、**每次编辑后重算**——这是 pair28_29 不可组合性（−4.17pp）的正确解释器：slot 级收益非加性 ⟹ 逐槽列表法先天失效。
（校准件：astra04 修正映射 f(bM+r)=(S−1)M+S·bM+r 填满 131072、98304 干扰槽；padded-KL 恒等式；L_bal 块平衡＋恒等映射 guard；SLSQP。sol01 预算化 argmin C_T s.t. D₀≤δ（δ 取 MrPro 实测）。sol03 等边际 KKT 律＋回退梯子。）

## 5. 死亡机制登记册（合并两线，按病因分类）

**几何-无符号类（全灭）**：Σcos 根排序（MrUni 82.2K>MrPro 80.3K 而 32K 64.6≪87.2；E2/P2 同根反向）；碰撞能/覆盖/平滑/有效秩/response energy/movement-MAE（C2 MAE=0.001223 仍挂门）；scale-orbit 计数（ULP 扰动 6→64 轨道、行为无差）；iid 噪声→α∫ρ²（旋转不变，无碰撞项）；任何无符号二次项入 F。
**坐标-类错误类**：冻结 checkpoint 的密度/多重集/排序参数化（同多重集置换 NLL 3.104→6.865、Qwen core-4 0.70→0；联合置换频率+学习系数槽才是恒等）；scratch 密度移植到冻结（Geo↔Cosh 运行时互换 PPL 7.14↔76.20 / 7.16↔23.05）；解①对象装④对象（Astra02 类错误）；"守恒律"不点名坐标。
**局部-有限类**：Taylor/Jacobian 分数跨全 S=4 表（相对误差 71–468%，相位 22.74/90.97 rad）；逐槽可加性（pair28_29；跨 key 相干与共享 head/W_O 对消：同非负能量总导数 0 vs 4）；slot-19 "Fisher/Hessian"（无 artifact；Fisher≠Hessian、MAE≠L1 是数学错误，sol19 撤）。
**叙事-过度类**："YaRN 递减 vs MrPro 递增"（F1/F2 证伪，双方都凸都递增）；MrRoPE 首零点→内剖面推导（论文自设等差 radix，边界是经验选择；"单调/44.8% 极限/high 不动/low ÷S" 全部从定律降格为设计选择——且一张早期有用表**非单调**）；universal 1×–2× 交换率（7 个异构协议混排）；六观测量充要（无充分性证明）；VICTORY CONFIRMED 类闭合证书；τ≈d_head/√L（"PASS"=脚本约定，9.6% 均值/33.3% 最大锚误差）；支持域重定标机制故事（公比对称性被代数否定，反转是观察不是机制）；universal 不可辨识定理（只约束已测的 model-blind unordered 类——**红线 R1 的判决本身不动，定理外衣缩窄**）。
**流程-实验类**：候选生成空转（16K assay 触底 0/32–1/32）；直接优化两条死路（64 维行为梯度未开 holdout 即败；direct-z 定支撑 pilot 挂门——**优化器必须在可辩护统计对象下游**）；attention≠generation（cross-cache：BM 读 MrPro 前缀能对，MrPro 读 BM 前缀仍错；record coverage 29.75→67.1 而散文精确答 7/8→6/8）；estimand 窄于叙事（NLL 锦标赛≠自回归源绑定）；状态阶梯 proposed≠implemented≠launched≠checkpointed≠evaluated≠accepted。

## 6. KKT 最终模板（指向 NEXT_DERIVATION，勿在此重复）

`NEXT_DERIVATION_KKT_PROBLEM.md` 给出：F = L_near（bank 学习计算损伤，∂ 随 r_j 增长）+ μ·L_far（距离带能力）目标；水床 = 过渡总跨度守恒；坐标 = 带标签有序 ν 向量；L_near 候选载体 = Σ_j w_j(r_j)(ν_j−ω_j)²（F3 的积分形式，**不得单独成 F**——F6 OLMo 反例禁止）；L_far 经 η_j=m_j 分布；交点位置的 S-依赖重推导 = **K1**；K6 = 四格反事实（ν^fast=max(νY,νM)/ν^slow=min 的 F00/F10/F01/F11 判决表，零新增参数，必须完整 prefill，交互项 = F11−F10−F01+F00）。
**负数据清单（禁止项已并入红线 R1–R5 与 §5）**：无符号项、对角 Σ、纯 pairwise SNR、冻结问题上的密度参数化、无界局部步（必须 trust box ½hᵀF_Nh≤ε＋精确三角重演认证）、μ_r≡0 时唯一合法输出是 "not identified"。

## 7. 冲突裁决表（两线材料互相矛盾处，本文判定）

| # | 冲突 | 裁决 |
|---|---|---|
| FLAG-1 | 根非排序 veto 的"定理级"外衣被 sol19/sol13/sol12 撤回 | **面板判决保留（经验事实），引用措辞一律缩窄为**："没有**已测的 model-blind 无序**统计量能认证冻结部署"。任何"不可辨识——定理"式表述作废。 |
| FLAG-2 | sol19 程序硬编码保序 vs sol12 记录非单调有用表 | 保序是 **MrPro 面的刻面**（该面单调），不是普适律；F 里放保序=声明面选择，文档必须写明。 |
| FLAG-3 | Smooth "差 9.79 分"聚合表述 | 一律用 near/far 分解：near 打平（87.2 vs 87.22），**全部损失在 far**。 |
| FLAG-4 | sol14/sol12 "Qwen-1.5B" | 规范记录 **1.485B**。 |
| FLAG-5 | 32K 拉伸位置能否作校准分布（sol16 可以/sol17 oracle-only/sol18 禁止） | **证据分层**：拉伸行=方向发生器与 oracle 上限（jsonl 实测支持 sol17/sol18：32K 响应住在被冻结的快槽）；**模型级判定必须真实连续 128K**（sol18 §3 协议）。不平均，写标签。 |
| FLAG-6 | 16-DOF 坐标：sol16 Helmert 保序构造 vs sol18 槽坐标加约束 | 同单纯形不同坐标；选 sol16 形式做无约束求解器，出表前用 sol18 坐标报槽值；两式互换性 G1 表里已可逐位验证。 |
| FLAG-7 | E3_BM 命名（Q7） | **数字层已由 verify_evidence/verify_feasibility 双重核销**：98.3333/75.3472＝Control_Mr_gain074（BUDGET:32 旧误标），E3_BM_gain074＝100.0/70.0、E3_BM_gain1＝89.58/58.82；行标签已更正。命名沿革（为何 BM 臂占 E3 号）仍开放，不影响任何数字。 |
| FLAG-8 | 五份失败审计一致声明"当前语料不足以出新 Qwen 表" vs 交付要求"具体频率表" | 不矛盾：**交付的表 = 地面真值重建表（已验证数学/CPU）+ 候选表（未过角色门，明示资格线）**。任何新表的发布前提是先跑 §8-3 的标签矩测量或 sol18 测试。 |
| FLAG-9 | **Astra01 vs Astra06 欧拉–拉格朗日律分歧**（codex digest-1 点名）：SNR-ratio 目标给 `αρ″−βρ=λh″`（λ=乘子），mass-bound 目标给 `αρ″−βρ=h″`（无 λ） | **F 项的选择直接决定密度律**——KKT 推导的第一分岔，不可回避、不可平均。裁决路径：两律各自代入判决集（sol13cx2 slot-28 符号反转 +0.99946→−0.99954；Smooth/P2/E1 三例排序），哪个 F 在标签矩上给出正确排序（astra08 门）才采纳哪个律；§8-1 数据到位前两律都记 [假设]。配套收录 astra06 可执行界 `μ−v/2 ≥ logN + log[ρ/(1−ρ)] + log(1/δ)` 为 far 侧候选约束；astra09 有限步规则（θ*=θ₀+clip(−g/L,±r_box)，默认框 1 rad⇒128K 附加相位 ≤0.25 rad）与 16 增量面冲突（它动所有槽），集成时必须先投影到面内。 |
| FLAG-10 | **D2 与 D4 对 N′ 族给出相反的可证伪预测**：D4（天花板+加权 Pareto，s=.5）推荐 N′\*=16/次选 15（预测长端 ≥78.13）；D2（校准泛函 R=U+αH+βΦ）判 N16/N15 **劣于** MrPro（R=51.15/51.51 vs 47.44——窄桥把预算顶进 bank，Φ 项上升），并预测 Stack 优于全部 5 表（R=46.55<s28 46.67） | **不用口头调和——0448/0449/0446 即预注册对赌判决点**（T3 §5.3 已把此张力写进定理 4 的诚实条款）。判决表：N16≥MrPro ⇒ D2 的 Φ 线性形或 bank 窗 (24,29) 证伪；N16<MrPro ⇒ D4 的 s=.5+mid 天花板约定证伪；Stack<s28 ⇒ 可加性死（D1 §2.3 的洞簇超可加先验赢）。两泛函共享的输入（G1 表、面板分数）都已独立验收，分歧纯在目标函数形——这正是"假设组织器"设计要的张力。**〔T2 追加，本会话〕** 第四席（T2 自算 Cheb 参数）方向与 D2 发表完全一致：Stack 47.65（全表最优预测）/ N16 52.64 / N15 53.04 vs MrPro 48.60〔T2 c4〕；且 T2 新发现 **notch 在 ramp16 底座上不再免费**（N16notch R=54.32——ramp16 的 ε27+ε28 合并后 ρ₂₈=1.3882 越线，Stack 底座上 s28 洞恰落 1.3710≈容差线）：若 0446 判可加，D4 §6.3 的"ramp16⊕notch"分支必须摊薄 notch 预算（≤2 gap）或改用 ramp17 底座。判决表升级为 T2 (d) 的 **K1–K9 预注册死刑标准**（K2＝vt_0@106K 行分不动为最重行级死刑——D_j 穿越→行修复因果链断；K7＝0450 上 notch 修复 0/16 复现→c1 可行窗降为 dev 内参数化）。 |
| FLAG-11 | **V×4 终审三席无官方裁决**：journal 仅 line 25 tables 席在册；math（62 行转录、崩于 T3 质心段 KeyError 'LBF'）/veto（58 行、无最终文本）/predictions（95 行、末两步 shell 工具错误截断）三席中断于复算途中，无裁决 JSON、`V_math.md`/`V_veto.md`/`V_predictions.md` 均未写出 | **裁决：三席结论在重跑席回传前一律 [重建-非官方]，终稿只可引用 V4_FINAL_VERDICTS 的 [已复核] 子集**。缺口清单＝V4 §6 UNCONFIRMED 九项（T3 四定理/EVQ→cosh/ε_j↔守恒换算无人执行；D4 §2 全局界证明未独立复证；c1 可行窗两席网格互差 (1.390,0.55)、(1.371,0.45) 待 math 重跑裁决；Kendall τ=+0.852(25/27) 未复算；G1 生成器重跑字节全等仅席上声明；ruler.jsonl 逐行 id 级翻转仅一致性复核；veto 全维度悬空；predictions (d)"事后 vs 预测"与 (e) 最强/最弱句未产出；0450↔080 漂移细节未展开）[V4_FINAL_VERDICTS.md §0/§2/§4/§5/§6]。**〔裁决更新 2026-09-10——本条"在重跑席回传前"前置条件已满足，[重建-非官方] 标签解除，九项对账：①T3 四定理＝V_math §5 复核（Thm1/2a/3 [已验证]、Thm4 [部分证据]、ε↔守恒与 NLC 乘数全对、无阻断性失败）✓；②D4 §2 全局界＝V_math §2 [已验证-限定定义]（字面 F_arc 退化合句判死→已传导 D4:24）✓；③c1 窗互差＝**仍不判**——math 席明示未复核（V_math §1.3），veto 席复现主窗 [1.370,1.386]/α[3.70,4.02]，孤点口径见 T2:83（margin 0.0%/p=0.55），不阻断回队；④τ=0.852(25/27)＝**作废**——V_predictions 逐字节复算 T2_validate 输出一致、只认发表口径 τ=+0.778（27 对 24 一致 3 漏）[已验证复算]；⑤G1 重跑字节全等＝V_tables:9 独立复现（09-10 更正编辑**之前**的状态；此后脚本已有意分叉——死旗标修复＋sources 行号，重生成批次待授权 §8-7/§8-9）；⑥ruler 行级翻转＝V_predictions §0-2 [已验证]；⑦veto 全维度＝V_veto 在册（3 contradicted＋4 overstated，**全部已传导**）；⑧predictions (d)(e)＝在册；⑨0450↔080＝V_tables Blocking-① 展开（治理入 §8-7④）〕。V-blocking 更正吸收计数见 §8-9 |
| FLAG-12 | **tables 席 verdict #5 论据自我矛盾（回收席新发现）**："README §4-5 fast 段例外清单漏列"结论方向成立（HGL/HGM 槽 1–23 实测偏离原生，[已复核]），但其括号论据"（G1 JSON 字段对 HGL/HGM=False，数据正确）"不成立——该字段是生成器 shape bug 死旗标（`rebuild_ground_truth_tables.py:111` 24 元切片 vs 64 元全数组），对含 Native 的全部 30 个数组方法恒 False，无判别力；另 README §1 ω"逐位吻合"实测 **24/64 槽 1-ULP**（最大相对差 1.178e-7）[已复核] | **裁决：结论保留、论据作废替换**。真实 fast 段例外清单＝{HGL, HGM, FullLagP2_Transfer3B, NTK_static} 四臂，YaRN 两变体 fast 段逐位等于 Native→README §4-5 例外清单改四臂口径（补 HGL/HGM、删 YaRN），生成器该行改 `f32(nu[:24])` vs `f32(NATIVE[:24])`；连带传导：D4 §4 行标签"mid（g29–36/37）"改"mid（g29–35）"（本席以 g29–35 逐位复现 D4_transport_rule.md:92 发表值——文档自洽，分歧是 math 席验证者带选取 off-by-one，非文档错误）；D1:22 残留"÷4.93"改 4^m40=**4.9638**（BUDGET:14 已更正、D1 未同步）[V4_FINAL_VERDICTS.md §1-4/5/7、§3-①②] |

## 8. 数据缺口与可执行下一步（按判据排序）

1. **标签化角色矩从未被测**（五审计一致的唯一阻塞证据）：需要 pre-RoPE Q/K 上按角色（native/far × source/hard-distractor）拆分的带符号均值/协方差/单位对数 MGF，含跨槽协方差，保 layer/head/relation/lag 标签；现有 6 份捕获缺角色标签（astra01 自然捕获=仅末查询、4 头采样、无问题/记录标签）。**预注册门槛先行**：统计量必须先正确否决 Smooth（slot-28 反转）、暴露 P2 的 +long/−short 权衡，失败即停并报告缺失因果层。
2. **K6 四格实验 = 最高优先 GPU 判决**（零新参数，完整 prefill，五格最小充分校准）；128K 端先收 response 回执。
3. **sol18 §3 模型级测试跑一次**：连续 128K 四格族、family-disjoint、一个冻结 CE 表 + equal-norm 镜像、10 个陷阱门（gain 恰好一次 g⁴ 陷阱；`@torch.no_grad` forward 静默断梯度是 #1 陷阱；fresh prefill per candidate；CE≠argmax）。桥接理论臂：Astra09 预测方向 vs 同族精确 CE 梯度，分歧则模型级目标获胜并诊断冻结态假设。
4. 先行小项：**astra10 的 max_new_tokens 线索已结案**（`checks/max_new_tokens_reconciliation_20260910.md`）：五-QA 文件顶层 `max_new_tokens=16` 是 `_from_model_config` 回显、**有效 cap=64**（778 行 max=64，两臂各 45 行恰停 64，对称截断，自然 F1 分数仍可用；其 EOS 计数 725/728 与 OLMo RULER 的 119/197 属不同实验不得混引）；screen qualification 文件行级字段自洽。规则：生成分数引用前以行级长度分布核定有效 cap。G1 地面真值表 v1 已落盘（38 条目、18/18 bit-exact、126 锚点、10 条 mismatch 如实记录，含队列候选洞值修正 1.86/1.76/1.80→1.493/1.461/1.476）；脚本收尾等 G1 停笔后统一清。**洞位置终裁**：Stack max 洞 **@g38**（本会话第三独立重算：Stack 继承 LBS 尾段，g35=1.4562、g38=1.4930；**T1 §0.2-1 与 T2 R2 第四、五席一致确认**，JSON 自记 `hole_ratio_argmax_transition=38`）——D2/D4/G1-README 的 @g35 为位置误标、值一致；BUDGET §3 已按 g38 修；**D2（两处）/D4（§3 表）/G1-README（§6 表＋归因句）文内旧标已于本日终稿一致性轮全部改为 @g38＋〔更正〕注**。连带：**D1 §6 的 0446 预测前提作废**（T2 R2：Stack 洞集≡LBS 逐位，g35=1.4562=LBS g35；Stack−LBS 洞差只在 g27→1.2409/g28→1.3710），D1 §5-6/§6 已按此更正——Stack 是"双手术、洞不新增"的干净可加性检验。G1 Stack 数组 m27/m28 3.8e-9 反演伪影的交付口径＝**m28:=m27**（T1 §0.2-4，与 T2 (a)2 同判）。
5. **文献占位**：LeRoPE（arXiv 2607.10134 §3.2）已做频率损失梯度+下游余切——本轮任何"梯度法"表述必须带此先行工作；**且 LeRoPE 已做 in-window 多干扰项检索——"我们也做 in-window"不构成新颖性**（切割落在冻结放置规则与 ±v 控制）。CoPE（2602.05258）非"检索新增"：09-07 已知先例并测过 Mr+CoPE 臂（128K 检索 8/8、UUID 0/8，V-D11 由此生效）。
6. **验证修复已应用（本会话）**：verify_math §4 的 11 项阻断 + verify_feasibility §4 的 ①②③⑤ 已逐条写入 BUDGET/UNIFIED（两文档头部各有"验证修订记录"；判定规则缺项以【缺口】标出待作者冻结阈值）。INTEGRATION 的 §1/§2/§3/§7 同步更新。**采纳判定**（wf1 三席合议式结论）：BUDGET/UNIFIED 的代数核与骨架可采纳；候选数字层修复后方可引用；`[BUDGET]:45` 的"已验证"在 0450/0451 出数前按 contradicted 处理（C1 循环门）。
7. **GPU 恢复前的治理清单**（全部未授权，需用户决定）〔更新 2026-09-10 11:00 现状口径：**0441 在跑；0446→0448→0449→0450→0451 未授权（UNQUEUED，需 USER 显式重授权，与 NONGEOMETRIC:9-31 no-new-32K 协议冲突未解除——本文档不主张超出此口径的队列状态）**〕：① 队列范围/优先序与 NONGEOMETRIC:9-31 协议冲突——重授权 or 改 long-only+0451 先行；② 0450 契约不含 Stack/LBS（A 通过分支无 holdout 落点）；③ `evidence_distances_20260910.json` 本地缺失→Core-C 不可判；④ 队列编号治理：0446 号与已撤回臂 `deferred_queue/20260910_candidate_quality/0446_HighGapToMid` 同号复用、"0450" 在本地队列的真实契约是 `queue/080_E1_holdout16.json`（编号漂移）〔更新 2026-09-10，V×4 tables 席 blocking [已复核（目录列举）]：`queue/` 在册仅 001–043、**0440×2**（P2_LongPrecheck128 与 Smooth_MrBudget 同号两份）、0441、0444、0445、080——无 0446_Stack/0448/0449 契约；`deferred_queue/20260910_candidate_quality/0446_HighGapToMid.json` 在册而 BUDGET:48 以 0446 指 StackFrontBack（同号异表）；0450↔080 互指漂移细节仍未展开 [V4_FINAL_VERDICTS.md §1-8]〕；⑤ 判定规则未闭合分支+缺功效估计（83% 线锚 2 行 dev）；⑥〔T2 R7 追加〕**Stack/N16/N15 三表从未部署、无 sha 可对**（公式重建级资产）——GPU 回队时先按 T2 §0 闭式对账（T2 数组即验收值，`tables/CANDIDATE_TABLES.json` 为机读基准），回执与对账不符则 T2 (b)–(d) 全部预测作废重算。〔更新 2026-09-10〕V×4 tables 席 blocking 佐证本 duty 为实：G1 中 N16 的 `formula_vs_deployed` 即记"公式重建，无本地部署张量可对账"（服务器侧执行表不可本地验证 caveat [已复核]），`queue/` 亦无 0448/0449 契约——**处置要求维持：contract+sha 登记后方可回队**；执行授权冲突（NONGEOMETRIC:9-31）席上未解除、回收席亦不解除 [V4_FINAL_VERDICTS.md §1-2/8]。
8. **T1/T2 已吸收（本会话）＋ R1 构造护栏**：候选表全量交付见 `tables/CANDIDATE_TABLES.{md,csv,json}`（幂等生成器 `generate_candidate_tables.py`，hard assert 不过不产出）；行级预测矩阵＋K1–K9 见 `answers/T2_validation.md` (d)。**R1（T2 (e)1，防呆红线）**：任务口径"均分桥 N′=13..17"与 T1 实际交付"ramp Eq.14 族"**不是同一闭式**——uniform N′=17＝已判死 MrUni（64.58/73.33，死于 bank 负载 Φ=30.39 而非洞）。**下游任何候选构造必须用 ramp 闭式 m_q=q(q+1)/(N′(N′+1))；按"均分桥"字面执行得到的将是 MrUni 方向表。** T2 对两族恒等式同检全绿，仅 ramp 族为合法候选域。
9. **wf3 整合层记录（consolidate-unify-20260910，本会话）**：workflow（run wf_bd01e12c-81b，脚本 `~/.claude/commands/consolidate-unify-20260910.js`）**未完成运行**——Verify:critic 席 6 次尝试全部 180s 停摆，脚本 null-guard 对"停摆即抛出"无效，整个 run 终止、**无返回对象构造**；其 7 个 agent() 结果全部从 `<session>/subagents/workflows/wf_bd01e12c-81b/journal.jsonl` 恢复采用，Verify/fixer 阶段改由主循环内联执行（不 resume：同脚本必再停摆）。**分工与吸收**：整合席（后台）完成文档同步——RECOVER 四席、本文件 wf1 附录更新、`INDEX.md`（新建 67 行）、STARTING §0、NEXT（146→184 行重写）、GR/TABLES 席的 §4-5/§1 更正；主循环完成全部实质性 V-blocking 修复——官方 V 席报告回传后按 blocking 清单逐条改 **D1×4、D2×2、D3×3、D4×13、T3×5、T2×8、CANDIDATE×5＋再生成（MISMATCH:0）**，本午后段又落 **D4 带标签×1、GR×6、rebuild.py×5、generate.py×6＋再生成、NEXT×4、STARTING×2、INDEX×8、本文件×5**。台账：`checks/V4_FINAL_VERDICTS.md` 保持 11:08 历史快照不追改（官方四席 `answers/V_*.md` 在册后其 §4/§5 [重建-非官方] 口径由 FLAG-11 更新行取代）。**复核席结论被复算推翻的首例（verify-the-verifier 纪律）**：V_math §4 断言"T3 §2 step-1 把 TEX 的 \(\int_\varphi^1\) 转写成 \(\int_0^\varphi\)"——复算两处原文：[TEX] eq:stationarity_integral（`docs/theory/EVQ_COSH_THEORY.tex:181`）本即 \(\int_0^1\)（Fredholm 全域核），T3:80 转写逐字一致，**该勘误项不成立，T3 未改**。**G1 重生成批次（待授权，勿擅自跑）**：脚本三处更正（:111 死旗标修复、sources 行号 45–47→48–50、max 洞声称历史化）使当前脚本重跑 ≠ committed JSON——重生成将改 G1 哈希、波及 glue 对账，与 HGM 构造回填一并等 GPU 批次/用户授权时执行。

## 9. 交付三件套现状（对用户的诚实状态表）

| 交付物 | 现状 | 完成条件 |
|---|---|---|
| **推导出的分配规律** | **已成形（D4+T3+D2）**：盲目标＝均分桥＝MrUni（崩）被否证为错设；位置计价 ⇒ 水填充要求右载 ramp——**业界分段形状由目标函数导出而非前提**；规律＝"Eq.14 形不变、完成边界 dh=40→39/38、加 bank-notch 二选一"（D4 §5）；T3 给出统一泛函 J=A+H+P 及四定理极限（EVQ=重学极限体解、MrPro=线性化精确解 Eq.14 代数回收、YaRN=进入阶 O(N⁻¹) 边界违例、D4 族=冻结极限解），I1/I2 从外生约束升为**罚极限 emerge**。K1（交点随 S 移动）仍未证；权重锥非单点（N′∈14..17 均可为某权重 argmin） | K1–K4 至少 K1/K2 出证明或反例；K6/F9 出判决；J 的 A 系数形状辨识（§8-1 门） |
| **具体频率表** | 地面真值表 G1 v1 **已独立验收**（`checks/g1_ground_truth_acceptance_20260910.md`：MrPro 公式重算 8.37e-8、水床恒等式 24/24 零违例、E1 手术恰 {28}/{29}/{28,29}）；**T1 全量交付**：13 表×64 槽（径向族 N′=13..17 主交付〔N17≡MrPro bit-exact、N16/N15≡0448/0449 冻结公式 MATCH〕＋stack_0446＋ramp16/15_notch 条件候选＋均分族参考〔uniform17≡MrUni 身份检查 MATCH〕），每表 m/ν/T/D/r/λ/ρ 全列＋29 项机器校验全 MATCH＋守恒 hard assert，`tables/CANDIDATE_TABLES.{md,csv,json}`＋幂等生成器；T2 独立闭式对账三方一致（4.1e-7 打印舍入位内） | 新表的 GPU 资格仍锁在 §8-1 标签门/§8-3 测试、§8-7 治理清单与冻结队列判决（K1–K9）之后 |
| **实际效果** | 面板 36 行＋OLMo 350 行钉死；任何候选的 GPU 效果**零** | 每个候选一行：完整 prefill 128K 实测 near/far＋receipt 落盘；**先决：§8-7 治理清单（重授权/long-only）** |

**workflow-1 裁决附录（2026-09-10，本会话吸收）**：proposers 双双失败（prompt 过长＋6 次停摆）→ 未生成 proposal_A/B.md；三验证器按分派口径直查规范载体 BUDGET/UNIFIED。判定汇总：数学席＝"恒等式层是真代数、历史撤回错误零重犯；11 项阻断修完（约半天）后候选队列（重判洞资格）与理论骨架可采纳"；证据席＝"条件通过——数字可信（126 锚点 121 匹配、无虚构）、标签须改（6 项）"；可行性席＝"构造层完全可行（A/B 表 CPU 生成＋恒等式自检过＋成本实测 4–7h 带内），证据措辞与授权层不可原样入队"。全部修复已应用（见 §8-6）。未决：proposer 若补跑须改提示词（缓存重放同 prompt 必再失败）——其职能现由 D4/T3 答案覆盖，暂不重跑。

**workflow-2 落盘状态（本会话吸收，〔更新 2026-09-10 11:00〕）**：D1–D4＋T3＋**T1（候选表全量）＋T2（结构验证/预测矩阵，含 `t2_work/T2_validate.py` 复现链）**均已落盘吸收——判决进 §1 四子问题、地面事实进 §3、对赌更新进 FLAG-10、护栏与 R1 进 §8-8、表交付进上表。**V×4 终审已落地并回收**（替代原"仅剩终审在途"状态）：回收台账 `checks/V4_FINAL_VERDICTS.md`——tables 席**官方裁决在册**（journal line 25；回收席重跑 `checks/v_tables_check.py`＝261 OK/0 FAIL [已复核]）；math/veto/predictions 三席转录中断、**[重建-非官方]**，重跑席（veto ac82fd9e5ac62af37、predictions ae2ce49a3ae7e47e7、math a9fcda922ac40e803）11:08 快照在途无 result。T2 自带合规自查逐条过（无静态几何→能力跳接、无虚构分数、gain 不入目标、E8 不作 bank 证据）。D2/D4 对赌见 FLAG-10（K1–K9 判决表）。**主循环剩余事项（11:00 口径）**：① 重跑三席回传后裁决 UNCONFIRMED 九项（V4_FINAL_VERDICTS §6，见 FLAG-11）；② V4 新发现传导轮：G1 生成器死旗标行修复、GROUND_README §4-5 例外清单改四臂口径（补 HGL/HGM、删 YaRN）＋§1 ω 措辞改"≤1 ULP（24/64 槽）"、D4 §4 行标签改"mid（g29–35）"、D1:22 "÷4.93"→4.9638 同步（见 FLAG-12）；③ 队列维持 **0441 在跑、0446–0451 未授权**，任何新 GPU 任务须待用户显式重授权（§8-7）；④ 本 workflow 收口后的**最终 commit+push 由主循环执行**（本会话无 git 权限）。

**〔刷新 2026-09-10 午后——11:00 口径四项处置〕** ①**完成**：重跑三席全部落盘，UNCONFIRMED 九项对账结果见 FLAG-11 更新行（唯 c1 窗孤点悬而不判、G1 重生成待授权）。②**完成**：传导轮全部落地——G1 生成器死旗标行已在脚本修复（committed JSON 未重生成，属预期分叉，见 §8-9）、GR §4-5 改 26/30 全量口径✓、GR §1 ω"逐位吻合"→24/64 槽 1-ULP✓、D4 §4 行标签→mid（g29–35）✓、D1:22 ÷4.93→4.9638✓；另 GR §4-1 守恒按 V_math §1.1 重写（17 张→望远镜 30/30＋24/30＋6 例外）、§5-9 HGM 升入复原清单、§6 附 0446 同号异表治理警示。③**维持不变**（本刷新未接触任何队列/GPU 操作）。④**执行中**（Task #4：终稿整理→commit→push）。

## 10. codex 活线程连接

`codex://threads/01a0806f-3df5-74b1-bc56-bf00d89d238e`（rollout 118MB，只读）。至 2026-09-10 12:56Z：codex 端已归档（commit 525dc15 把 30 代理材料+两份理论文档推入 `docs/research/`），用户指令"提交并推送"已由 codex 执行；未发现本会话材料之外的新理论内容。权威起点附件：`~/.codex/attachments/6f22c629-45a0-41e8-a383-4ad135436f7c/pasted-text.txt`（F1–F9 原文）；沙箱脚本 `sandbox:/mnt/data/mrrope_research/mrrope_yarn_verified_analysis.{py,json}` 未在本地仓库，若 codex 后续落库在 §2 补链接。

——完——下一次推导从 §6 指向的 NEXT_DERIVATION 文件开始，红线 §1，缺口 §8。
