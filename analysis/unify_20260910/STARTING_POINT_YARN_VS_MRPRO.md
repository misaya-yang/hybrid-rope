# 推导的权威起点：YaRN vs MrPro 的已验证算子级对比

状态：用户与 codex 会话（thread 01a0806f）于 2026-09-10 共同确认此材料"值得作为推导起点"。〔更新 2026-09-10 晚〕本文件在 `INTEGRATION_20260910.md` §2 资产地图中列为"权威起点 F1–F9，定稿，三方交叉验证"，阅读优先级为 INTEGRATION → `NEXT_DERIVATION_KKT_PROBLEM.md` → 本文件（INTEGRATION_20260910.md:9 文档头"阅读优先级"行）；D1–D4/T1–T3/V×4 轮未推翻 F1–F9 任何一条（`checks/V4_FINAL_VERDICTS.md` 全文无杀 F 项裁决；INTEGRATION §1-1 明确保留 F3 进入阶论证为"D2/D4 的进入阶论证（T3 定理 3）"）。问题状态自本起点之后的推进见下节「当前位置」。
本文件的 F1–F9 由三方独立来源交叉验证：
(a) 用户粘贴的核查分析（原文：`~/.codex/attachments/6f22c629-45a0-41e8-a383-4ad135436f7c/pasted-text.txt`，对照 HF/原始 YaRN 与公开 MrRoPE 代码重算）；
(b) 本方 workflow 摘要 `digests/digest_mrrope-evq.md` §D8（本方独立重算：Qwen S=4 上 HF-YaRN 与 MrPro 逐槽 max|Δm|=0.034，论文"YaRN 回退"口径在 HF 实现下无表格载体）；
(c) codex 会话 30 代理归档（`.agents/rope_unification_20260910/`，digests_codex/ 摘要进行中）。

**红线更新**：本文件 F8 与我方 root-veto（红线 1）是同一件事的两个来源，互相印证。任何后续推导不得再使用"YaRN 递减 vs MrPro 递增"作为机制解释（F1/F2 已证伪）。

---

## 0. 当前位置 2026-09-10（D1–D4 / T1–T3 / V×4 轮之后）〔新增 2026-09-10〕

**问题域（自由度与守恒钉死）。** 部署对象 Qwen2.5-3B，W=32768，S=4，L=131072，θ=b=10⁶，K=64 槽，ν_j=ω_j·S^{−m_j}，ω_j=θ^{−j/64}（zero-based）[INTEGRATION_20260910.md §3]。I1（m_j=0，j≤23）与 I2（m_j=1，j≥40）是**设计面**、非被证定律（红线 R3，INTEGRATION §1）；真实自由度=**槽 24–39 共 16 个**（INTEGRATION §3"自由增量 Δ_j 在槽 24–39（16 个）"）。17 个桥 gap 总跨度 17×0.2158674+ln4=5.0560 nats（本会话按 `tables/GROUND_README.md` §1 原生 log-gap 与 §2 水床恒等式现算），其中被守恒锁定的搬运量只有 Σ(gap−native)=ln S=1.386294（GROUND_README §2）；原生周期比 ρ_nat=b^{1/64}=1.2409378，ρ_g=ρ_nat·4^{Δm_g}[tables/CANDIDATE_TABLES.md §0]。官方 gain a=1.138629436 独立记账、永不入 F/J[T2_validation.md §(e)8]。

**闭式族（T1 已交付全量）。** 候选构造族=ramp Eq.14：m_q=q(q+1)/(N′(N′+1))，q=clip(j−23,0,N′)，N′=13..17——13 表×64 槽 m/ν/T/D/r/λ/ρ 全列、29 项机器校验全 MATCH[T1 `tables/CANDIDATE_TABLES.md` §5.2]。**N′=17 与 MrPro 位级同表**（ramp17_vs_G1_MrPro max|Δm|=3.686e-8，G1 声明 ν 张量 bit-exact[T1 §5.2 首行；`checks/V4_FINAL_VERDICTS.md` §2 复现]）。ramp 族 ρ_max 梯子 1.5127@g35 / 1.4929@g36 / 1.4757@g37 / 1.4608@g38 / 1.4476@g39（N′=13→17）[T2_validation.md §0 D4 行、§(a)4]。**R1 构造护栏：任何下游候选构造必须用 ramp 闭式**——"均分桥"字面执行得到的将是 MrUni 方向表[T2_validation.md §(e)1；INTEGRATION §8-8]。

**MrUni 判死＝桥形失败，非端点违例。** "全表÷4"标签不实：实为过渡段线性斜坡、段外逐位同 MrPro（bit-exact 证实）[GROUND_README §5-3、§3 MrUni 行]；死于 bank 负载 Φ=30.39（Σ₂₄²⁸m=0.8824），而洞 ρ_max=1.3464 全绿[T2_validation.md §(e)1、§c3]；实测 32K/128K=64.58/73.33（基线 MrPro 87.2222/78.1250，GROUND_README §3）。这条判决反向支持本文件 F6/F8：洞/几何指标不排序能力。

**最佳预测候选与排序（T2 c4，[假设]级，GPU 效果零）。** 风险泛函 R=U+αH+βΦ 发表点 (ρ₀,p,α,β)=(1.372,0.5,3.920,1.234)，Chebyshev 可行窗 ρ₀∈[1.370,1.386]、p∈{0.5,0.55}（18/217 格）[T2_validation.md §c1；两席网格孤点 (1.371,0.45)、(1.390,0.55)〔刷新 2026-09-10：官方 math 重跑席明示不再复核该拟合窗（V_math §1.3 [未复核]）、(1.390,0.55) 维持 T2:83"边界格 margin 0.0%"口径、(1.371,0.45) 未获采纳——悬而不判，回队判决不依赖窗内孤点〕]。候选序：**Stack(0446) R=47.65（全表最优预测）< MrPro 48.60 < N16(0448) 52.64 < N15(0449) 53.04 < N16notch 54.32**[T2_validation.md §c4；V4 §2 独立重写特征逐位复现]。定位红线：R 仅局部排序描述子、幅度不外推（8 方法 τ=+0.778、漏 3 对[T2 §c2]）；**D4 天花板模型（N′*=16 主选）与 D2-R（N16/N15 劣于 MrPro）方向相反（INTEGRATION §7 FLAG-10 / T2 §(e)5），0448/0449 即预注册对赌判决点，禁止先行采信任何一方**。三候选覆盖暴露 U（N16 14 / N15 10 / Stack 13）全部低于 MrPro 17，方向与"后端完成"赢家机制一致[T2 §b1；INTEGRATION §3]。

**洞终值（V×4 终审后）。** Stack 最大洞 **1.4930@g38**（精确值 1.4929753892033155；旧标 "@g35" 为位置笔误、值无误——T1 §0.2-1 / T2 §(e)2 / V4 §1-3 三席一致 [已复核]）；**Stack 洞集与 LBS 逐位相同**，差异只在 g27→1.2409、g28→1.3710[T2 §(e)2；V4 §1-3]；s28_less 全局最大洞仍是 g39=1.4476——即 MrPro 自家洞[V4 §2 洞位表 9/9 复现]；N16 1.4608@g38、N15 1.4757@g37[GROUND_README §6]。

**notch 分支约束。** notch 在 ramp16 底座**不免费**：ε₂₇+ε₂₈ 合并成单洞 ρ₂₈=1.3882，越出容差窗；在 Stack/MrPro 底座 s28 洞 1.3710 恰落在线上[T2_validation.md §c4 注；V4 §2]。故若 0446 判可加，D4 §6.3 的"ramp16⊕notch"分支必须把 notch 预算摊薄到 **≤2 个 gap** 或改用 **ramp17 底座**[INTEGRATION §7 FLAG-10]。

**证伪合同＝K1–K9 预注册死刑标准**[T2_validation.md §(d)；升级记录见 INTEGRATION §7 FLAG-10 末注]。最重一行为 K2（Stack vt_0@106K 行分 ≤0.2 不动→"D_j 地平线穿越⇒行修复"因果链断，N′ 族的行级含义降回纯描述性）；K9（0446/0448/0449 三者 128K 全部 ≤78.125→"分配规则胜 MrRoPE"总 claim 死，收为框架+机制+否证记录）。注意**撞名**：此 K1–K9 是 GPU 判决合同，与 `NEXT_DERIVATION_KKT_PROBLEM.md` 的定理任务 K1–K6（K1=交点位置随 S 移动的重推导、仍未证[INTEGRATION §9；NEXT_DERIVATION:120]；K6=本文件 §7 四格反事实[INTEGRATION §6]）是两套编号，引用时须点名文件。

**队列状态（不扩述）。** 0441 在跑；0446→0448→0449→0450→0451 **未排队（UNQUEUED），需用户显式重授权**（与 NONGEOMETRIC:9-31 no-new-32K 协议冲突；治理清单 INTEGRATION §8-7①）。V×4 tables 席 blocking 在册：`deferred_queue/20260910_candidate_quality/0446_HighGapToMid.json` 与 BUDGET:48 的 0446=StackFrontBack **同号异表**；`queue/` 下 0446_Stack/0448/0449 **无本地契约**（在册仅 001–043、0440×2、0441、0444、0445、080），contract+sha 登记后方可回队[V4_FINAL_VERDICTS.md §1 blocking ①②]。

**V×4 终审状态与勘误传导结果。**〔刷新 2026-09-10〕四席官方裁决**全部在册**——tables（独立验证器重跑 261 OK/0 FAIL）＋重跑三席 V_predictions(11:20)/V_veto(11:30)/V_math(11:50)；V4_FINAL_VERDICTS 台账＝11:08 历史快照。原"待传导"五项处置：① G1 死旗标（24-vs-64 形状比较）**脚本已修**、committed JSON 未重生成（G1 重生成批次待授权，INTEGRATION §8-7）；② fast 段例外四臂＝{HGL, HGM, P2, NTK}——`GROUND_README` §4-5 **已改** 26/30 全量口径（补 HGL/HGM、删 YaRN）✓；③ ω"逐位吻合"→ 24/64 槽 1-ULP（最大相对差 1.178e-7）**已改** ✓；④ `D1:22` ÷4.93→4.9638 **已改** ✓；⑤ D4 §4 行标签"mid（g29–36/37）"→"mid（g29–35）"**已改** ✓。另 V_math §1.1 守恒重写传导：`GROUND_README` §4-1 已改望远镜 30/30＋24/30＋6 例外口径。本文件 §1–§4 的 YaRN vs MrPro 闭式对照不涉及上述臂数组，不因此改变。

## 1. 锁定比较对象（F1/F2）

中段参数化：l,h 为中段边界，N=h−l，q=j−l，t=q/N。

- **标准代码 YaRN**（jquesnelle 原始仓库 + Transformers 实现，维度索引上的频率比线性混合）：

  ν_j^Y = ω_j (1 − t + t/S)

  写成累计压缩指数：m_Y(t) = −log[1−(1−1/S)t]/log S，**m_Y′>0 且 m_Y″>0**——标准 YaRN 的对数压缩增量同样递增。[已验证-推导+代码对账]

- **MrRoPE-Pro**（radix 对数增量等差 ⇒ 二次累计压缩）：

  ν_j^M = ω_j S^{−m_q}，m_q = q(q+1)/(N(N+1))

  二次形式是**设计假设**（论文自述来自等差 radix 假定），不是任何任务目标的解。[已验证-论文原文]

- 论文附录的"regressive"推导是对**圈数 r_j** 线性插值的另一表达，与维度线性不等价；公开 mrRoPE 仓库的 YaRN 构造函数也是维度线性版（圈数版在其未被调用的 `yarn2()` 中）。论文最终表格用的确切提交未核对，不作为本起点依据。[部分证据-仓库版本待对]

**推论（F2）**：MrPro > YaRN 的胜负**不发生在增量单调性上**——两者都凸都递增。必须找别的算子级差别。这就是为什么需要 F3/F4。

## 2. 已验证的差别一：中前段扰动的阶（F3）

第一中频槽 q=1 的相对降频：

- YaRN：1−ν₁/ω₁ = (1−1/S)/N = **O(N⁻¹)**
- MrPro：1−S^{−2/[N(N+1)]} ≈ 2logS/[N(N+1)] = **O(N⁻²)**

数值 [已验证-重算]：Qwen N=17,S=4 槽24：YaRN 降 4.4118% vs MrPro 0.9020%；Llama3 N=17,S=16：5.5147% vs 1.7958%。

旋转算子的精确扰动：‖R(dν)−R(dω)‖₂ = 2|sin(d(ν−ω)/2)| ≈ d|ν−ω|（d|ν−ω|≪1）；全谱局部极限 lim_{d→0}‖R_ν(d)−R_ω(d)‖²_F/(2d²) = Σ_j(ν_j−ω_j)²。Qwen 现配置 **Σ(ν^M−ω)²/Σ(ν^Y−ω)² = 0.4841**——MrPro 进入旋转算子的局部导数平方扰动约为 YaRN 一半。[已验证-CPU]

边界（材料原文自带，保留）：这证明"付出的局部旋转改动更小"，**不**证明冻结模型短任务分数必更高；不得升级为能力定理。

## 3. 已验证的差别二：尺度响应 η_j（F4，更关键）

定义 η_j(S) = −∂log ν_j(S)/∂log S（上下文再扩 1%，该时钟降频约百分之几）：

- YaRN：**η_Y(t,S) = t/[S(1−t)+t]**；任意严格中段 t<1 时 S→∞ ⇒ η_Y→0，ν_Y→ω(1−t)>0。**中段时钟随目标长度增长逐渐停止减速（饱和）。**
- MrPro：**η_M = m_q**，只要 m_q>0 就以确定幂律持续减速，不饱和。

目标距离 d=SW 处的未取模相位跨度：YaRN 严格中段按 S **线性**增长（Wω[S(1−t)+t]），MrPro 按 **S^{1−m_q} 次线性**增长。两种方法对长尺度计算的重标定力度有本质不同的增长规律。[已验证-推导]

## 4. 双向结构与随 S 移动的交点（F5）

实际频率交叉（MrPro 并非全域更接近原生）：

| 设置/槽 | ν/ω YaRN | ν/ω MrPro | MrPro 行为 |
|---|---:|---:|---|
| Qwen 4× 槽24 | 0.9559 | 0.9910 | 更接近原生 |
| Qwen 4× 槽28 | 0.7794 | 0.8729 | 更接近原生 |
| Qwen 4× 槽39 | 0.2941 | 0.2916 | **更慢（多压缩）** |
| Llama3 16× 槽19 | 0.9449 | 0.9820 | 更接近原生 |
| Llama3 16× 槽26 | 0.5588 | 0.5208 | 更慢 |
| Llama3 16× 槽32 | 0.2279 | 0.1492 | 明显更慢（ν_M/ν_Y=0.6544，周期长 +52.8%） |
| Llama3 16× 槽34 | 0.1176 | 0.0850 | 明显更慢 |

A/B 集合（由两表交点定，不是人为阈值）：Qwen 4× A={24–37} B={38–39}；Llama3 16× A={19–24} B={25–34}。

**确认的结构优势**：`降低中前段的局部扰动 + 让中后段真正承担持续的尺度扩展`——不是"把预算整体右移"，也不是"中段全部越接近原生越好"。[已验证-公式]

## 5. 反例与解释力的最低要求（F6）

**OLMo 反例**：MrPro 对 YaRN 的局部旋转导数扰动同样降到 ~47.8%，但独立 72 条实验长端 MrPro 2.78% < YaRN 6.94% ≪ BM 51.32%。减少局部扰动**不自动**换来任务胜利。[已验证-独立实验]

因此任何候选 F（连同其 KKT 解）必须**同时容纳**：
1. MrPro 在论文模型（Llama3/YaRN 对比）上的优势（F3/F4 侧）；
2. BM 在 OLMo 上的大幅优势（与 F3 方向相反的例子）;
3. BM 与 Smooth 在现 Qwen 上的分化（Smooth 87.2/68.3 vs MrPro 87.22/78.13，几何更优者任务更差——我方红线 1 的原始证据）。
只用"更少破坏原生"或"更早完成插值"解释其中任何一个，不够。

## 6. 非线性放大与根理论的正确地位（F7/F8）

- 读出是 softmax 竞争 p\*=σ(D)，D=z\*−log Σ_{k≠\*} e^{z_k}：频率改动只需把个别读取的竞争优势推过阈值即可产生离散跳变收益（RULER 79.9→86.6、KV 检索 9%→27% 而非全线提升），**不需要**平均几何损失下降。这与"静态几何代理不得入 F"一致：平均意义上的几何改善与任务胜负只在分布尾部相关。[部分证据]
- 论文 B_ν(d)=Σcos(dν_j) 零点理论 = iid 分量 + 相似 key k\*=q+ε 假设下的平均分差模型（Base of RoPE Bounds Context Length，arXiv 2405.14591），原文自认多层堆叠不严格；我方复算给出更强的判决：根与能力排序失序（MrUni 82.2K>MrPro 80.3K 而 32K 64.6≪87.2；E2/P2 同根反向）。**根 = 诊断量，禁入 F。**[已验证-双源]
- 高频保持 R(dω) 不变 + 慢频统一缩放 R(Sd·ω/S)=R(dω) 两个恒等式**拼不出**全网精确重演（多尺度拼接后网络状态重形成；我方 prefill/read 交叉实验已显示收益不限于固定 Q/K 读出）。[已验证-实验]

## 7. 四格反事实（F9，下一批 GPU 判决的最高优先设计）

定义 ν^fast_j=max(ν_j^Y,ν_j^M)、ν^slow_j=min(ν_j^Y,ν_j^M)（=M_A+Y_B / Y_A+M_B）。零新增曲线参数、同端点、同频率数、同 gain、保排序。**必须从完整 prefill 执行**（不得用固定状态重放替代）。〔更新 2026-09-10〕五格反事实未执行，其"下一批"承诺由队列治理接管：INTEGRATION §8-2 仍列 K6 四格实验为"GPU 恢复前最高优先判决"，但当前队列现实是 0441 在跑、0446–0451 待用户重授权（见 §0"队列状态"），且受 NONGEOMETRIC:9-31 no-new-32K 协议约束（INTEGRATION §8-7①）；执行时其交互项判据 F11−F10−F01+F00 原样保留[INTEGRATION §6]。

| 结果 | 对分配规律的含义 |
|---|---|
| fast > MrPro | B 侧（中后段多压缩）有负贡献；规则不得再加强后段压缩 |
| slow > MrPro | 更强的中后段重标定有价值，MrPro 的 A 侧保留并非该设置的正确取舍 |
| MrPro > fast 且 > slow | A/B 双侧配合是胜因本体；任何单侧规则被判死 |

（第四、五格 = YaRN、MrPro 本身。）这五格构成 F 的**最小充分校准实验**：KKT 解若在 A/B 分解上给出与五格相反的预测，直接被否。

## 8. 对 KKT 问题陈述的输入（并入 NEXT_DERIVATION_KKT_PROBLEM.md 的增量）

1. **L_near 的候选载体出现**：Σ_j(ν_j−ω_j)²（加权 d² 后的局部算子导数扰动，即 F3 的积分形式）。它是**动态算子量**（依赖真实旋转差）而非静态几何代理——但 F6 禁止它单独成 F；候选形式 L_near = w_j·(ν_j−ω_j)² 中 w_j 应随 r_j（bank 占用圈数）增长，与我方"∂L_near/∂m 随 r_j 单调"假设一致。〔更新 2026-09-10〕D2/T2 轮把该载体落地为风险泛函 R=U+αH+βΦ 并给出定位裁决：8 方法 τ=+0.778、漏 3 对，"局部排序描述子、幅度不外推"[T2_validation.md §c2/§c3]；且洞必须按其所在 T 带计价（Smooth/pair 双反例[T2 §c3]；D4 §2 独立复现[V4 §2]）。原句 w_j 随 r_j 增长仍是 L_near 侧待证假设，不是在册的 R 形式。
2. **L_far 必须经 η_j**：远距能力不再是"根更远"而是"该槽承担多少尺度重标定"——η_j=m_j（幂律族内）。F 的 far 侧应依赖 {η_j} 的分布（中后段是否有人持续减速），这给 KKT 桥条件一个新的等边际对象：**边际 far 收益 per unit Δ_j ∝ ∂(能力)/∂η 沿槽位形状**。
3. **三段结构获得真实代码证据**：MrPro 的 A/B 交点 + YaRN 的 η 饱和失效 = "最优解自然产生分段"的候选形态描述；K1 定理要证的是**为什么交点位置随 S 移动**（F5 的 Qwen 38→Llama 25 移动必须可从 KKT 条件的 S-依赖重 derive）。
4. **四子问题的更新答案骨架**：高频冗余 = bank 段被 O(N⁻¹) 阶不必要扰动（冗余在"改动阶次"不在"频率未用"）；中频关键 = 它同时是 L_near 的梯度出口（前段）和 η 的承担者（后段）——**桥的两端职能不同**；低频资源 = η→1 饱和段（I2 的经济学理由）；搬运 = 在 ΣΔ=1 守恒下把压缩预算从 η 饱和无贡献区搬到 ∂能力/∂η 最大处——四格实验（F9）是这一定量化的第一刀。
5. **论文公式层面的缺口**（材料原文结论，保留为诚实边界）："还没有从这些事实中推出一条足以判断会稳定超过 MrRoPE 的新分配公式"；二次累计压缩未被证明为任务最优。KKT（K1–K4）正是补这一步的框架。〔更新 2026-09-10〕"推出"一步已有 CPU 级交付：D4 分配规律="Eq.14 径向 ramp 形不变、完成边界 dh=40→39（主选 N′=16）/38（次选 N′=15）、次搬运=前端 notch 0.0327 m-单位（3.3%）"，含权重锥非单点的唯一性诚实条款[INTEGRATION §1-4]；T1 已交全量表、T2 已交预测矩阵与 K1–K9 合同。但 GPU 效果仍为零、"稳定超过"仍不可判断[INTEGRATION §9"实际效果"行]，且 D2-R 与 D4 对 N′ 族方向相反（FLAG-10）——原文判决解除前，本文件 §0 的全部候选序均为 [假设]。

## 9. 工件位置

- 粘贴原文：`~/.codex/attachments/6f22c629-45a0-41e8-a383-4ad135436f7c/pasted-text.txt`（只读）
- codex 沙箱中的可复现脚本/数值：`sandbox:/mnt/data/mrrope_research/mrrope_yarn_verified_analysis.{py,json}`（不在本地仓库；〔更新 2026-09-10〕codex 归档已落库——仓库权威副本 `docs/research/rope_allocation_20260910/`，codex commit 525dc15，见 INTEGRATION §10；沙箱脚本本体仍未入库）
- 公开代码对账对象：`raw.githubusercontent.com/jquesnelle/yarn/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py`；`github.com/mattian7/mrRoPE`；根理论出处 arXiv:2405.14591
- codex 30 代理归档：`.agents/rope_unification_20260910/`（28 报告 + coverage 回执 + corpus 120MB；digest 于 `digests_codex/` 进行中）〔更新 2026-09-10〕7 份 digest 已全部读完[INTEGRATION §2 资产地图]
- 本方独立重算：`digests/digest_mrrope-evq.md` §D8、`scripts/analysis/rope_softmax_harmonic_audit.py`
- 〔新增 2026-09-10〕本轮（D1–D4/T1–T3/V×4）工件（均在 `analysis/unify_20260910/` 下）：地面真值 `tables/ground_truth_tables.json`+`tables/GROUND_README.md`（G1，独立验收 18/18 bit-exact[INTEGRATION §2]）；候选表全量 `tables/CANDIDATE_TABLES.{md,csv,json}`+幂等生成器（T1，29 校验全 MATCH[§5.2]）；四问推导 `answers/D1_high_freq_redundancy.md`–`D4_transport_rule.md`；结构验证与预测矩阵 `answers/T2_validation.md`（含 K1–K9 死刑标准 §(d)）；统一定理草案 `answers/T3_unification.md`；总整合 `INTEGRATION_20260910.md`；终审复算台账 `checks/V4_FINAL_VERDICTS.md`。下一次推导入口顺序：INTEGRATION → `NEXT_DERIVATION_KKT_PROBLEM.md` → 本文件[INTEGRATION:9]。
