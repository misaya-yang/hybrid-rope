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

1. **高频的冗余在哪里**——不在"频率没被用到"，在**改动阶次**：MrPro 在第一中频槽的相对降频是 O(N⁻²)（Qwen 槽24：0.9020%）而 YaRN 是 O(N⁻¹)（4.4118%）；冗余 = 被不必要的一阶扰动打乱的 bank 学习计算。[已验证-双源重算 F3]
2. **中频为什么关键**——桥的两端**职能不同**：中前段是 L_near 的梯度出口（尽量少动），中后段是尺度响应 η_j=m_j 的承担者（持续减速、不饱和）。YaRN 的中段 η_Y→0 饱和失效，MrPro 幂律永不饱和——这是 F4 里最接近机制的事实。[已验证-推导]
3. **低频需要多少资源**——η→1 饱和段，资源需求恰好等于端点约束 m=1（÷S）；这是 6Pro 设计约束，不是定理。低频"够用即可"的经济学理由是其 η 已饱和，加更多压缩边际收益趋零。[部分证据——K2/K4 要把"趋零率"算出来]
4. **从哪里搬、搬到哪、搬多少**——在 ΣΔ=1（过渡段总跨度守恒，5.0560 nats）下，把压缩预算从 η 饱和无贡献区按 ∂(能力)/∂η 的边际形状重分配；交点位置（Qwen A/B={24–37}/{38–39}，Llama 16× A/B={19–24}/{25–34}）**必须从 KKT 条件的 S-依赖重新推导**（K1），搬多少的第一刀是四格反事实实验（K6/F9）。

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
| 地面真值表 | `tables/ground_truth_tables.json`（v1 320KB）＋ `rebuild_ground_truth_tables.py` | **G1 代理正在修正**（09:05 活跃编辑，勿碰；完成后需按其自报 parity 结果验收） |
| 提案 A/B＋数学/证据/可行性验证 | workflow-1（wf_c2d1c25c：25 启动 / 16 回传） | **进行中**，落盘后并入 §9 附录 |
| 四问推导 D1–D4、候选表 T1–T3、终审 V×4 | workflow-2（wf_f8c14342） | **进行中** |
| 原始 transcript 快照（本方） | `raw/`（2.9MB，只读镜像） | 留档 |

## 3. 地面事实（钉死数字，两条线共用）

部署对象：Qwen2.5-3B，W=32768，S=4，L=131072，θ=10⁶，K=64；ν_j=ω_j·S^{−m_j}；ω_j=θ^{−j/64}（**zero-based**，末对 θ^{−63/64}≈1.24e-6，非 θ^{−1}——sol16 精确约定）；自由增量 Δ_j 在槽 24–39（16 个），j≤23 m=0，j≥40 m=1；gain g²=1+0.1·ln4=1.1386294。

面板（36 行，near/far）：MrPro **87.22/78.13**；s28_less 87.2/**83.3**（已知最强 far）；LBS 80.6/80.1；P2 72.9/81.7；MrUni 64.6；Smooth 87.2/68.3（R2 反代理）；pair28_29 **−4.17pp**（逐槽可加性已死）；HighGapToLong −17.1/−10.8；E2 54.7；E8 50.6；E3 gain074 98.3/75.3。OLMo-2-0425-1B 16K 350 条：BM 41.67% vs MrPro 7.09%（+34.59pp，156W/9L；multikey_3 双方皆 0，多重绑定未解决；EOS BM119/MrPro197，评分口径不得混用——sol20 回传）。prefix/read 交叉：−2.125/−1.125/−1.0/+0.25（收益不限于固定 Q/K 读出）。32K 全模型 CE 梯度 4 行（`full_model_response_native.jsonl`）：‖grad‖ 50.9–688.7，**响应集中于被 MrPro 面冻结的快槽 0–6**，过渡槽 24–39 几乎无响应——128K 过渡带问题需要真实长相位（sol18 实测，支持 sol17 的分层立场）。128K response 终态 **UNKNOWN——使用前必须先收回执**（progress 文档纪律）。

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
| FLAG-7 | E3_BM 命名（Q7） | 仍未解决；这批材料无帮助。保留开放问题。 |
| FLAG-8 | 五份失败审计一致声明"当前语料不足以出新 Qwen 表" vs 交付要求"具体频率表" | 不矛盾：**交付的表 = 地面真值重建表（已验证数学/CPU）+ 候选表（未过角色门，明示资格线）**。任何新表的发布前提是先跑 §8-3 的标签矩测量或 sol18 测试。 |

## 8. 数据缺口与可执行下一步（按判据排序）

1. **标签化角色矩从未被测**（五审计一致的唯一阻塞证据）：需要 pre-RoPE Q/K 上按角色（native/far × source/hard-distractor）拆分的带符号均值/协方差/单位对数 MGF，含跨槽协方差，保 layer/head/relation/lag 标签；现有 6 份捕获缺角色标签（astra01 自然捕获=仅末查询、4 头采样、无问题/记录标签）。**预注册门槛先行**：统计量必须先正确否决 Smooth（slot-28 反转）、暴露 P2 的 +long/−short 权衡，失败即停并报告缺失因果层。
2. **K6 四格实验 = 最高优先 GPU 判决**（零新参数，完整 prefill，五格最小充分校准）；128K 端先收 response 回执。
3. **sol18 §3 模型级测试跑一次**：连续 128K 四格族、family-disjoint、一个冻结 CE 表 + equal-norm 镜像、10 个陷阱门（gain 恰好一次 g⁴ 陷阱；`@torch.no_grad` forward 静默断梯度是 #1 陷阱；fresh prefill per candidate；CE≠argmax）。桥接理论臂：Astra09 预测方向 vs 同族精确 CE 梯度，分歧则模型级目标获胜并诊断冻结态假设。
4. 先行小项：**astra10 的 max_new_tokens 线索已结案**（`checks/max_new_tokens_reconciliation_20260910.md`）：五-QA 文件顶层 `max_new_tokens=16` 是 `_from_model_config` 回显、**有效 cap=64**（778 行 max=64，两臂各 45 行恰停 64，对称截断，自然 F1 分数仍可用；其 EOS 计数 725/728 与 OLMo RULER 的 119/197 属不同实验不得混引）；screen qualification 文件行级字段自洽。规则：生成分数引用前以行级长度分布核定有效 cap。G1 地面真值表 v1 已落盘（38 条目、18/18 bit-exact、126 锚点、10 条 mismatch 如实记录，含队列候选洞值修正 1.86/1.76/1.80→1.493/1.461/1.476）；脚本收尾等 G1 停笔后统一清。
5. **文献占位**：LeRoPE（arXiv 2607.10134 §3.2）已做频率损失梯度+下游余切——本轮任何"梯度法"表述必须带此先行工作。

## 9. 交付三件套现状（对用户的诚实状态表）

| 交付物 | 现状 | 完成条件 |
|---|---|---|
| **推导出的分配规律** | KKT 模板定稿；三段结构有数值涌现证据（sol04/sol06）+交点位置事实（F5）；K1（交点随 S 移动的推导）未证 | K1–K4 至少 K1/K2 出证明或反例；K6 出判决 |
| **具体频率表** | 地面真值表 v1 在产（MrPro 公式↔部署 ≤4.3e-8 重算中、31 行方法全谱）；29 联合模式候选（CPU 合格、角色未资格）；提案 A/B 表待 workflow-1 | G1 验收＋四问/候选表 workflow-2 出 T1–T3＋新表过 §8-1 门或 §8-3 测试 |
| **实际效果** | 面板 36 行＋OLMo 350 行钉死；任何候选的 GPU 效果**零** | 每个候选一行：完整 prefill 128K 实测 near/far＋receipt 落盘 |

（workflow-1/workflow-2 完成后，其产物并入本节附录：proposal_A/B、verify_math/evidence/feasibility、answers/D1–D4、tables/CANDIDATE_TABLES、answers/T2/T3、answers/V×4。）

## 10. codex 活线程连接

`codex://threads/01a0806f-3df5-74b1-bc56-bf00d89d238e`（rollout 118MB，只读）。至 2026-09-10 12:56Z：codex 端已归档（commit 525dc15 把 30 代理材料+两份理论文档推入 `docs/research/`），用户指令"提交并推送"已由 codex 执行；未发现本会话材料之外的新理论内容。权威起点附件：`~/.codex/attachments/6f22c629-45a0-41e8-a383-4ad135436f7c/pasted-text.txt`（F1–F9 原文）；沙箱脚本 `sandbox:/mnt/data/mrrope_research/mrrope_yarn_verified_analysis.{py,json}` 未在本地仓库，若 codex 后续落库在 §2 补链接。

——完——下一次推导从 §6 指向的 NEXT_DERIVATION 文件开始，红线 §1，缺口 §8。
