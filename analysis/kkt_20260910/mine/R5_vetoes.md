# R5 — 合并否决清单（Dead-Mechanism Registry）

日期：2026-09-10。作者：R5 采矿代理（只读）。
产出物定位：**推导用的否决清单**，不是推导本身，不是执行指令。
输入范围：`/Users/yang/projects/hybrid-rope/analysis/unify_20260910/digests/` 全部 16 份 digest（约 712 KB）＋三份权威锚定文档（`INTEGRATION_20260910.md`、`NEXT_DERIVATION_KKT_PROBLEM.md`、`STARTING_POINT_YARN_VS_MRPRO.md`）。

**用法**：KKT 推导的每一段推理、每一个候选 F 分项、每一个"新判据"，逐条对照本表。判违规的三条件（沿用 `digest_failure-records.md` §4 用法）：(a) 未标证据等级；(b) 未给出处；(c) 未给出被撤回版本的精确弱化形式。

**证据等级词表**（项目铁律）：`[已验证]`＝有可检索原始数据/数组/恒等式；`[部分证据]`＝开发面板、小样本、单 seed 或口径受限；`[假设]`＝未被匹配证据唯一支持；`[叙事-未验证]`＝只在报告/转述中出现、无可检索凭证。

**引用约定**：`D:<file>:L###` 指 `digests/` 下某 digest 的行号；digest 内部再引一手文档时我保留其原路径名。**路径口径警告**：多数 digest 把仓库写成 `/Users/[REDACTED_AUTHOR].yanghejazfs.com.au/paper_project/hybrid-rope/`，而本机实际工作目录是 `/Users/yang/projects/hybrid-rope/`——同一仓库两个挂载名，引用时勿当两个仓库（见 §5 矛盾 1）。

---

## 0. 一句话根因（本清单的排序原则）

> 循环的根因是推理链反复失效，且失效结论没有退出；已有证据反复否定的是**从某个局部量到实际能力的跳接**；后续却把该局部量**改名、增加自由度，或把错误前提重新当起点**。
> ——`D:digest_failure-records.md:L131`（引 SYNTHESIS §2 行 31–34，一手：`docs/research/ROPE_LOCAL_FAILURE_SYNTHESIS_20260908.md`）

因此本清单的**分组不是按"谁死了"，而是按"怎么死的"**：几何量死在"无符号/与内容无关"；坐标死法死在"对象身份错配"；局部死法死在"有限范围外推"；叙事死法死在"把观察升成定律"；流程死法死在"判据与执行治理"。

每一项给出四元组：**机制名 / 为什么死 / 决定性证据（出处＋等级） / 复活形态与误用警告**。最后一项是防重复踩坑的关键——死机制从不会以原名回来。

---

## 1. 几何-无符号类（全灭）

> 类判词（`INTEGRATION_20260910.md` §5 首段）：**"任何无符号二次项入 F"** 一律否决。理由：冻结部署的真实对象是 `z_t(ν)=c_t+Σ_j{A_tj cos(ν_j d_t)+B_tj sin(ν_j d_t)}`，**带符号、按槽标签、带内容系数——三缺一即死**（`INTEGRATION_20260910.md` §4.2）；一切无符号几何聚合把这些全部丢掉。

### G-1 Σcos 首根（RoPE-bound 根排序）作为 F 分项或障碍定理
- **为什么死**：根排序与能力排序**显著失序**，且它本身是"可重学网格"式静态量（对所有维等权、不含内容系数、不含训练弧）。
- **决定性证据**：`[已验证-CPU 复算]` `NEXT_DERIVATION_KKT_PROBLEM.md` §4 第四条与 `D:digest_mrrope-evq.md:L74,L104,L108`：MrUni 82.2K > MrPro 80.3K 而其 32K 64.6 ≪ 87.2；E2/P2 同根 109.1K 一崩（54.7）一存（81.7）；s28 修复 +5.2pp 时根几乎不动（80.7 vs 80.3）。文档措辞："**必须排除出 F**"。
- **复活形态与误用警告**：(a) 改名成"频谱条件数""相位拥挤度""RoPE-bound 上界"再入 F；(b) 被 MrRoPE 论文 §4.4 Fig.5 的 `B_θ(m)` 背书——但该量是**均匀相位的静态余弦和**，同族于 r₂，只能预测"编码势"不能预测"被使用的计算"（`D:digest_mrrope-evq.md:L108`）；(c) 论文自报 28K bound 与其自身 128K NIAH 实测差 **4.6×**，连量级都不对（同上 L74）。注：**根非排序的"定理级"外衣已被撤回**（`INTEGRATION` FLAG-1），红线 R1 的判决保留为**经验事实**，措辞必须缩窄为"没有任何**已测的 model-blind 无序**统计量能认证冻结部署"。

### G-2 cos-only 碰撞核 / 精确 Ci 核作为预测器
- **为什么死**：cos-only 核是"**半个故事**"——完整 RoPE 几何还缺另外三个 sin/cos Gram 分量；且静态碰撞 ≠ 外推机制。
- **决定性证据**：`[已验证-数学，但机制链接被否]` `D:digest_theory-core.md:L67,L110,L201`（一手：`paper-2027/research/foundations/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md`、记忆 `full-rope-collision-audit.md`）；`D:digest_evq-code.md:L120,L134`（一手：`EVQ_NONLOCAL_KERNEL_CORRECTION_20260910.md`——正斜率展开 −π⁴/(720c³)‖ρ′‖²，delta 近似在每个非零波数**过罚**变化）。核本身闭式恒等式无误（`D:digest_theory-core.md:L110`），但其作为选择器的含义被 C5/Table 1 否定。
- **复活形态与误用警告**：(a) "精确核比 cos-only 好，所以更该进 F"——精确核只是修了**建模步**，文档自定位"a specific correction to a modeling step, **not** evidence that a sharp transition improves a frozen model"；(b) 用 `J[h]=(1/2)∫[α/h+β(1−u)²h]du` 里的粗糙度项反过来给"平滑的表更好"背书（见 §5 矛盾 3 的辨析）。

### G-3 覆盖 / 未访问弧比例（coverage / unvisited arc）
- **为什么死**：联合学习对象是（相位向量 × 内容系数 × 竞争 key），**边际覆盖不够**；且 45 号槽在参考窗内早已转过整圈，"未见弧"前提不成立。
- **决定性证据**：`[已验证-面板+复算]` `D:digest_failure-records.md:L188`（V-D16，一手 `ROPE_EXTRAPOLATION_FAILURE_AND_LIMITS_20260910.md` §3）：E1 槽 28/29 在参考窗内已转 **12.37 / 9.97 圈**，无法用单圆未访问解释；`n_cyc` 序列 1.17/1.34/1.62/1.95（槽 36–39）见 `D:digest_pro-materials.md` 与 GLM 复核，**否定 UNIFIED_BUDGET §1/§2 的"超出 D_j 进入未训练相位弧"**（见 §5 矛盾 2）。
- **复活形态与误用警告**：(a) 换成"相位空间覆盖率""token 距离测度覆盖"；"六观测量充要"是最常见的包装（`INTEGRATION` §5 叙事-过度类点名）；(b) 只有 `r_j<1` 的槽才允许"未见弧"语言（`NEXT_DERIVATION_KKT_PROBLEM.md` §1.4 L_far 段）。

### G-4 平滑度 / 粗糙度（表几何的平滑性）
- **为什么死**：Smooth(MrBudget) 在**几何意义上全赢**、能力上全输。
- **决定性证据**：`[已验证-面板]` `INTEGRATION_20260910.md` §1 红线 R2 与 §5：Smooth 的 U 在**全部 36 层三个 cutoff** 都优于 MrPro，其 128K 开发分 **68.3 vs 78.13**；near 打平（87.2 vs 87.22），**全部损失在 far**（FLAG-3 要求一律用 near/far 分解，不许写"差 9.79 分"聚合）。一手：`ROPE_ALLOCATION_SUBSPACE_DERIVATION_20260910.md`；`PARALLEL_NONGEOMETRIC_20X10_AUDIT_20260910.md`。
- **复活形态与误用警告**：最小粗糙度的解**退化成 Full PI / 近常数表**——文档原话"任何只最小化 U 的规则会退化成 Full PI/近常数表，不能解分配问题""**代理不得升为部署规则**"（`D:digest_theory-0910.md:L246`）。

### G-5 有效秩 / effrank / r₂（Rényi-2 恒等式）
- **为什么死**：静态 r₂ 与 LM 性能**反向**。
- **决定性证据**：`[已验证-两规模]` `D:digest_thread-0908-night.md:L99,L147`（night06 M3/M4）：50M 冻结 2×2 中 GeoW+EVQT 使 static r₂ **4.57→12.54**，PPL 反崩 **7.14→76.20**；interaction −3.5367 CI[−5.165,−3.039]≈5.9× 主效应；151.9M 两 seed 复核 interaction 3.400/3.251。论文侧 Table 1 同结论（`D:digest_mrrope-evq.md:L56`）。
- **复活形态与误用警告**：r₂ 的闭式 `r₂(R)=2K/[1+(K−1)c̄]` 是**真数学**（trace 恒等式，数值 4e−14 内，`D:digest_paper-state.md:L131`）——正因为它是真恒等式，最容易被当成"有效秩提升⇒表征变好⇒能力提升"的桥。桥不存在。

### G-6 Gram / QK 算子 Gram / 加权 U_H / Raylesigh 最差方向
- **为什么死**：**加权不救排序**。线性/加权算子族（cos 碰撞核→U→QK-Gram 加权 U_H→最差方向 Rayleigh→bias-KL）**逐级加码，Smooth-vs-MrPro 反例在每一级都不翻转**。
- **决定性证据**：`[已验证-CPU，恒等式 1.776e-14]` `D:digest_evq-code.md:L92,L116,L131,L134,L260`；原话："U and unweighted local distortion are **not a sufficient selector**"、"even its Q/K-weighted version does **not** rank all observed tables correctly"。同时"Gram 不能"清单明确：**不能说激活分布、不能说 softmax KL、不能说任务损失、不能说语义信息损失**。
- **复活形态与误用警告**：(a) 正则化最差方向表（0.1→1e-8 放大 5–15 个数量级）"保留为**诊断**，**不作选择器**"；(b) 数值分辨率声明必须绑定到**具体矩阵**（曾把 1e-10 施于 Gram 特征值，而设计矩阵奇异值 1e-10 对应 Gram 谱 1e-20，该放大数字被弃用）；(c) astra03 明确"**上一轮 Q/K projection-matrix Gram 不是所需测量**"（`D:digest_thread-0910-batch.md:L65`）。

### G-7 能量类代理（独立相位能量 χ_i、response energy、背景二次目标）
- **为什么死**：两类独立失败。(a) χ_i 丢失 i≠k 相干项——**两个算例能量同 (4,1)，统一换频后输出导数 0 vs 4**；(b) 背景二次目标改善与能力改善**同时发生但不共变**。
- **决定性证据**：`[已验证-代数算例]` `D:digest_failure-records.md:L63,L182`（V-D10；一手 REVIEW-0907 行 129–146）：χ_i=||∂y/∂φ_i||²（或其×距离²）**不得**直接推逐频率压缩量；应改用**有符号共享频率响应**。`[已验证-开发面板]` V-A3（`D:digest_failure-records.md:L139`）：原 Carrier 背景目标改善同时长 UUID/VT **全 0**。
- **复活形态与误用警告**：(a) "用非负响应直接推压缩方向"是三种被禁偷懒替代之一；(b) 共享频率响应的正确形式是 `J_{q,j}=Σ_h W_{O,h}Σ_i Δ_qi ∂o/∂φ`，`G_jk=(1/Q)Σ_q J_qjᵀ J_qk`——**同 query 内跨 key/head 带符号求和，query 间取平均**；投影时是重排不是新可观测量（`D:digest_failure-records.md:L267`；`D:digest_thread-core.md:L55`）。

### G-8 movement-MAE / 耦合律距离
- **为什么死**：C2 重建 movement MAE=**0.001223** 仍未过 Native 门；"足够接近有效表就保能力"被否。且"小 MAE 保证保持"被 C2 案例直接否定。
- **决定性证据**：`[已验证-历史实验]` `D:digest_failure-records.md:L232`（4.7 表 C1 行）、`L56`（OLMo 约 11.2865/7.7814 turns、Qwen 约 10.7764/7.4297 turns；两模型槽 19 的 C2 movement=0）；`D:digest_thread-core.md:L54` 条件②。
- **复活形态与误用警告**：不得以 MAE 放行，**也不得把轻微门槛失败写成崩溃**（双向禁止）。

### G-9 轨道计数（scale-orbit count / ULP 扰动）
- **为什么死**：ULP 扰动使轨道数 **6→64** 而行为无差；Exact/ULP 与有效 p2 否定该选择规则。
- **决定性证据**：`[已验证]` `INTEGRATION_20260910.md` §5 几何-无符号类；`D:digest_failure-records.md:L233`（4.7 表 C3 行："只作代数性质记录，**不用于挑表**"）。

### G-10 iid 噪声 → α∫ρ²（旋转不变项）
- **为什么死**：旋转不变，**没有碰撞项**；水床要求**声明的相干/嵌套协方差**，iid 通道噪声会杀死 `βΔT_i²/2` 项。
- **决定性证据**：`[已验证-推导]` `INTEGRATION_20260910.md` §4.2 水床幸存形式（sol17）。

### G-11 弦距离 D² 作为全局"分离度"指标
- **为什么死**：弦距离 `D²(Δ;ν)=4Σw_j sin²(Δν_j/2)` 对压缩**非单调**（Δν=2π 分量为 0、减半反而最大）。
- **决定性证据**：`[已验证-审计计算]` `D:digest_nongeo-code.md:L54,L125`（一手 `PARALLEL_NONGEOMETRIC_20X10_AUDIT_20260910.md`）：s28_less 在 5 个测试距离中有 **3 个降低**分离度（Δ² 变化 +0.1623/−0.3651/+0.5320/−1.5277/−0.4050）→ 实验 5′ 的 GPU 前置门**CPU 判负**。
- **复活形态与误用警告**：原话"**提高频率 ⇒ 周期性位置分离普遍增大**"是错的（V-D18），必须作为 VETO 记住；任何"分离度优化器"在开卡前先跑这 5 点回测。

### G-12 attention mass / attention top-1 代理
- **为什么死**：top-1 只需目标 logit 最大（权重 .4/.3/.3 即反例）；A(target)≥0.5 **非** top-1 必要条件；attention top-1 ≠ 生成正确。softmax dilution 只是**简化模型**。
- **决定性证据**：`[已验证-反例]` `D:digest_failure-records.md:L143`（V-A7）、`L174`（V-D2）；`D:digest_thread-core.md:L55`。

### G-13 有界相位码鸽笼界 / 静态障碍定理 ⇒ 实践天花板
- **为什么死**：界在 K=64 **极松**（差 ≤2π√K/M）；外部定理绑定随机 Q/K 模型与正则幅度假设；manuscript 移植障碍**只否定全算子精确等价**。
- **决定性证据**：`[已验证-数学]` `D:digest_failure-records.md:L189`（V-D17）：**不得**建 128K/1M 实践天花板，**也不得**反读成"冷换表不可能有益"——BM/P2/E1 的条件收益必须作为反权重保留。

### G-14 一切无符号二次项（兜底条款）
- **决定性证据**：`INTEGRATION_20260910.md` §6 负数据清单："无符号项、对角 Σ、纯 pairwise SNR"；`NEXT_DERIVATION_KKT_PROBLEM.md` §5.1。
- **复活形态与误用警告**：改名/换参数/换统计量包装都不算新机制。机检法（`D:digest_failure-records.md:L241,L316`）：**新统计量与谱系表已有量做代数恒等变形比对，命中即套用旧裁决**。

---

## 2. 坐标-类错误类

> 类判词（`INTEGRATION_20260910.md` §5）：对象身份错配——**解出的对象 ≠ 装上的对象**。

### C-1 冻结 checkpoint 上的密度/多重集/排序参数化
- **为什么死**：冻结权重下**槽位—频率配对不可任意交换**；同 multiset 仅置换槽位即崩塌。
- **决定性证据**：`[已验证]` `D:digest_thread-0910-batch.md:L80,L102,L105`：OLMo NLL **3.10423→6.86493**、Qwen core-4 **0.70→0**（一手：`CPU_LOW_DIM_COUPLING_LAW_20260901.md:38,61`；`SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md`）。数学原因：`B(Δ)=Σcos(ν_jΔ)` **置换不变**，真实 `z_qk` 带 `C_j/S_j` 内容系数 → 同一几何指标可对应 0.70 和 0 分（`D:digest_thread-core.md:L54`）。
- **复活形态与误用警告**：(a) "joint permutation of 频率＋学习系数 才是恒等"——不要用"频率必须单调降序"代表模型不可变约束，**降序是约定不是定理**（旧 p2 槽 1/18 交叉仍有效，V-D22）；(b) 唯一幸存结构条件"槽位—频率配对不可任意交换"**不帮选方法**（不产生候选优势预测）——不要拿它当 F 的项。

### C-2 scratch 密度移植到冻结态
- **为什么死**：训练期分配与冻结期放置是**两个范式**；换表使 PPL 崩。
- **决定性证据**：`[已验证-两规模]` `D:digest_paper-state.md:L106,L137`（一手 `paper-2027/sections/03_findings.tex`）：50M Geo 权重换 Cosh 表 PPL **7.14→76.20**；反向 **7.16→23.05**；151.9M 两 seed 重复。且《Spectral Budget》一文**从未测冻结 32K/128K 部署面板**（`D:digest_mrrope-evq.md:L56`）。
- **复活形态与误用警告**：移植障碍定理（Thm.3）只否定"频率多重集可被固定线性 Q/K 映射吸收"的**精确等价**，不否定条件收益。

### C-3 四对象互换（**最被一致同意的数学事实**）
- **为什么死**：精确有限窗口碰撞泛函的唯一测度最优是**有限原子**，不是正 Cosh 密度。
- **决定性证据**：`[已验证-证明+CPU 对偶间隙证书]` `D:digest_thread-0910-batch.md:L64`（astra02，全纯函数恒等定理排除无限支撑；原子数对 L 无普适上界）；`INTEGRATION_20260910.md` §4.2 原子性与四对象分离。
- **四个对象必须两两分开**：①精确-原子最优 ②光滑-Cosh 代理 ③有限-K 整数计数 ④冻结-带标签表。**解出一个装上另一个就是类错误**。
- **复活形态与误用警告**：(a) 原子定理**不**迁移到有限整数-lag Gram（唯一性可失）；(b) 原子定理**不**意味着 LM 应该重复频率；(c) 早期 astra02 的"缺归一化"点破——有限分辨率 Hilbert–Schmidt 投影需先指定分辨率，τ 缩放依赖该归一化选择。

### C-4 "守恒律"不点名坐标 / Σm 当守恒量
- **为什么死**：水床守恒只在**指定坐标**里成立（零和频移 ΔΣm≠0，joint_mode 实测 −0.01826…+0.00185）。**Σm 不是守恒量**，是自由决策变量。
- **决定性证据**：`[已验证-实测]` `INTEGRATION_20260910.md` §1 红线 R4 与 §4 记账纪律；`NEXT_DERIVATION_KKT_PROBLEM.md` §4 末段。
- **复活形态与误用警告**：(a) 写"总预算守恒"而不说是**17 个过渡 gap 的总跨度**锁定在 ln S；(b) 用质心 Σm 论证"赢家同向"——6Pro 已推翻（P2 34.18 > MrPro 29.33，P2 是右移还是左移取决于坐标，`NEXT_DERIVATION` §2.4）。

### C-5 "17 个 log-gap 之和 = ln S"的错误口径
- **为什么死**：**正确表述是 17 个过渡 gap 的总跨度锁定为 ln S**（原生 3.669745 + 额外 1.3863 = **5.056039 nats**），不是"17 个 gap 之和等于 ln S"。两种口径不可混用。
- **决定性证据**：`[已验证]` `NEXT_DERIVATION_KKT_PROBLEM.md` §1.3 守恒条与 §5.4；`D:digest_pro-materials.md`（6Pro 澄清）。6Pro 原始 Δsum 争议数值 5.38381→5.28431 见 `D:digest_pro-materials.md:L65`。
- **复活形态与误用警告**：这是**必写错的约定**，已三次复发。写成"ΣΔ=1"时须同时给坐标（m 坐标下的总行程）与原生部分分开计。

### C-6 gain 与相位混为一个自由度
- **为什么死**：gain 只乘 logit（gain²）改变**温度**，**不恢复相位弧**；gain×相位**不正交**但也不是正交分解；F **不含 gain 自由度**。
- **决定性证据**：`[已验证]` `D:digest_failure-records.md:L190`（V-D18）；`D:digest_theory-0910.md:L27`（AUDIT 第 42–46 行）；四格实测：BM 表不动、gain 1.1478→1，NLL 改善 0.02–0.035 nats 但 128K **70.83→58.82（−12.01pp）**；实现中 gain 同时乘 Q 和 K → logit 乘 **gain²**。
- **复活形态与误用警告**：(a) 把 BM(gain0.074) 的 32K +12.778pp 记作 gain 的独立因果效应——**是表×gain 打包与长度取舍**，须先补 `MrPro×gain0.074` 臂完成 2×2 factorial（V-B5）；(b) gain 是**独立幅度变量**，应独立记账。

### C-7 端点 m=0/m=1 与 gain=1+0.1·lnS 当被证定律
- **为什么死**：它们是**设计面（face）**，不是被证定律。I1/I2 是"强基线设计约束，**非零容忍定理**"——少数违例证明代价高，不排除某个小位移自由度的存在（s28_less 恰暗示槽 28 附近有 **~0.033** 的可回收量）。
- **决定性证据**：`[已验证-面板]` `NEXT_DERIVATION_KKT_PROBLEM.md` §1.3 I1/I2 段；`INTEGRATION_20260910.md` §1 红线 R3。反例：HighGapToLong（槽0–23 反向放大）32K −17.1/128K −10.8、36 行 0 提升；E8 单槽置零 → 128K 50.6；MrUni 带内左倾（m28=.294）→ 32K 64.6；E2 推深至 ÷4.93 → 128K 崩至 54.7。
- **复活形态与误用警告**：(a) 保序是 **MrPro 面的刻面**（该面单调），不是普适律；F 里放保序 = 声明面选择，**文档必须写明**（FLAG-2）；(b) sol19 程序硬编码保序要按此纠正。

### C-8 MrUni 身份混淆
- **为什么死**：本项目面板的 `MrUni` = **Δ 均匀 1/17，累计左倾压 bank 边缘（m28=.294）**；MrRoPE 论文的 MrRoPE-Uni 是**带内常数**。两身份不可互换。
- **决定性证据**：`D:digest_mrrope-evq.md` §7 Q1；`D:digest_panel-results.md` A5。

### C-9 物理-x 坐标特权
- **为什么死**：`c_orth=(1−b^{−1/K})^{−1}`、`x_i=log(L_ref·ω_i/(2π·c_orth))`、`G(x)=clip((x_H−x)/(x_H−x_L),0,1)`、`ω'_i=ω_i·s^{−G(x_i)}`（xH=.7382780681078285, xL=.366403835112904, c=.074）——该坐标化**没有特权**：K32 crossing 中 **index .9232 通过与 retention、physical .8595 不通过**。
- **决定性证据**：`[已验证]` `D:digest_thread-0908-night.md` §5 与 C 表。

### C-10 旧 log-p2 / FullLagP2 / P2Middle / 原始 Carrier / Native-sector Carrier 互相引用成绩
- **为什么死**：不同身份；8 月已发现 stride16 高频混叠、log-p2 用 2048 整点格；数值修复历史不能互相覆盖。
- **决定性证据**：`[已验证]` `D:digest_failure-records.md:L163`（V-C2）。scale-transport 提案（09-07）：**仅 λ=0 通过**（5 点 guard），提案实际同时做了三件事（槽 24–39 全部比 MrPro 慢 0.28205–0.85724×；槽 23→24 log 间距 0.22493→1.48037；投影产生 8 对相邻等频只剩 56 个不同频率；j=24 累计压缩指数 0.0065→0.912；短 QA F1 0.425 vs Native 0.498465）。
- **复活形态与误用警告**：裁决明确——**不能把该提案失败理解为"适当修改中段已经失败"**；整个 Pro 理论未被否定，**guard 也不等于真实能力**（REVIEW-0907 行 22）。

### C-11 从 radix 投影直接推机制
- **为什么死**：radix 语言看得见形状，**看不见 r_j、D_j、距离**。
- **决定性证据**：`[已验证-三投影链论证]` `D:digest_mrrope-evq.md:L102`：机制层必须用 m/Δ；radix 保留为**记号层**（Figure 1）。

---

## 3. 局部-有限类

> 类判词（`INTEGRATION_20260910.md` §5）：局部量在**有限范围/有限分辨率**外失效。

### L-1 Taylor / Jacobian 分数跨全 S=4 表
- **为什么死**：相位漂移超数弧度仍用局部二次外推；有界正弦算子使无约束二次增长非法。
- **决定性证据**：`[已验证]` `D:digest_failure-records.md:L65`：Mr 处**单点** Jacobian 外推旧 p2 相对误差 **71.15–468.12**；相位 22.74/90.97 rad。`D:digest_thread-0910-batch.md:L108`（台账 "Local Taylor extrapolation" 警告）。
- **复活形态与误用警告**：允许形式——局部 Taylor/Fisher **只在正则性与 trust region 内**保留；必须 `(1/2)hᵀF_Nh≤ε` ＋ 精确三角重演认证（`INTEGRATION_20260910.md` §6）。sol06 的 RTGA 与 astra03 都据此加了**精确残差/有限改动认证**。**不得**接成"下一个能力优化器"（V-E3）。

### L-2 逐槽可加性
- **为什么死**：slot 级收益**非加性** → 逐槽列表法先天失效。
- **决定性证据**：`[已验证-36 行面板]` `INTEGRATION_20260910.md` §3：pair28_29 **−4.17pp**，且 36 行 **0 提升**（洞比 1.46× 超调）；`NEXT_DERIVATION` §3 表 pair(28+29) 128K **74.0**，**低于任何单槽单独结果**。
- **复活形态与误用警告**：(a) 跨 key 相干与共享 head/W_O 对消——同非负能量总导数 **0 vs 4**；(b) 正确解释器是 **sol07 全前向、每次编辑后重算的边际效用转移**（`INTEGRATION` §4.3-4）。

### L-3 slot-19 "Fisher/Hessian 敏感度"
- **为什么死**：81.2% 是**曲线拟合平方残差占比**，非 Jacobian/内容模长/损失敏感度；且 Fisher≠Hessian、MAE≠L1 是**数学错误**。
- **决定性证据**：`[已验证]` `D:digest_failure-records.md:L179`（V-D7）；`INTEGRATION` §5 局部-有限类："slot-19 'Fisher/Hessian'（**无 artifact**；Fisher≠Hessian、MAE≠L1 是数学错误，sol19 撤）"。
- **复活形态与误用警告**："slot19 最敏感""harmonic mismatch""prefill 污染""softmax 稀释导致断崖"整体降为 hypothesis，**带错误数学推理的强版本直接撤回**。

### L-4 局部投影/保护选择器直接生成表（E7）
- **为什么死**：E7 局部投影 128K **−9.514pp**（判负）。且其"161× 局部线性化失败"的论据**作废**。
- **决定性证据**：`[已验证-BF16 实现层]` `D:digest_nongeo-code.md:L251`、`D:digest_theory-0910.md:L27`：NMSE 线性预测 7.9366844e−8 ≈ ideal 7.9364028e-8 ≈ FP32 7.9325473e-8，**BF16 1.2807392e-5**；161× 在 BF16 实现层，且 NMSE 比是**平方范数比非幅值比**。
- **复活形态与误用警告**：(a) 不要把"E7 判负"理解为"局部化必然错"——错的只是"**用一个局部保护准则直接生成表去跑**"；(b) 但 subspace 稿的精确算子陈述（Uniform-PI/selective-PI）**不受 E7 影响**（`D:digest_theory-0910.md:L246` 的"同教训、不同死法"辨析）。

### L-5 Fisher → 注意力效用（隐含等价假设）
- **为什么死**：泛函里 `b^{−2φ}=ω(φ)²` 是标准正弦频率估计 Fisher 标度律（这点**成立**），但"Fisher 保真奖励 ⇔ 注意力效用改善"是**未宣称**的隐含等价假设。
- **决定性证据**：`D:digest_theory-core.md:L150,L160,L202`（IRONCLAD §7.2："DAPE 等全部信息论 PE 论文共有"）。另注意：泛函里**没有显式 ρ≥0 不等式约束**——正性由解自动满足，四文件与后续审计**均未讨论**，写论文时是可被审稿人问的小空子（`L151`，`[假设]`级风险）。

### L-6 无界局部步 / 校准-KL / 正则化最差方向当选择器
- **决定性证据**：`NEXT_DERIVATION_KKT_PROBLEM.md` §5.5：线性读出/固定态代理算子（J_r、校准-KL、E7 160×）**只配假设生成，不入 F 主链**；`D:digest_evq-code.md:L134`。

### L-7 冻结态选择器分数（E8 类）
- **为什么死**：E8 强冻结态选择分数对应 128K **−13.889pp**（前 12 行），multiquery/FWE 回归。
- **决定性证据**：`[部分证据，判负]` `D:digest_failure-records.md:L142`（V-A6）；`D:digest_thread-0909-pm.md` §3 分数表（E8 0/−13.889pp）。
- **复活形态与误用警告**：代理不含"有用计算的损失"项；**min U 的解是 full PI**。

### L-8 有限维相码鸽笼界的越界使用（与 G-13 同源，此处强调"有限"）
- 见 G-13。关键是**界在 K=64 极松**，不得建立实践天花板。

---

## 4. 叙事-过度类

> 类判词：**把观察升成定律、把约定升成定理、把单点升成机制。**

### N-1 "YaRN 递减 vs MrPro 递增" 叙事
- **为什么死**：**双方都凸都递增**（F1/F2 双源重算）。
- **决定性证据**：`[已验证-双源重算]` `STARTING_POINT_YARN_VS_MRPRO.md` F1/F2；`NEXT_DERIVATION_KKT_PROBLEM.md` §3 YaRN 行明写"叙事**作废**"。

### N-2 三频带（high/mid/low）是数学定理
- **为什么死**：三带是连续谱上**不同渐近机制的实际过渡**；middle band **不是第三种数学上 distinct 的信息种类**。
- **决定性证据**：`[已验证-数学]` `D:digest_failure-records.md:L187`（V-D15）。

### N-3 "所有赢家同向移预算"
- **为什么死**：6Pro 已推翻强表述（Σm：P2 34.18 > MrPro 29.33，**方向取决于坐标**）。
- **决定性证据**：`NEXT_DERIVATION_KKT_PROBLEM.md` §2.4 与 §5.2。**只许用弱版**：`bank 边缘卸载 ∪ 危险区右段完成`。

### N-4 低频覆盖一个圆 ⇒ 更高频率安全
- 见 G-3。`V-D16`。

### N-5 universal 1×–2× 交换率
- **为什么死**：**7 个异构协议混排**。
- **决定性证据**：`INTEGRATION_20260910.md` §5 叙事-过度类。

### N-6 τ≈d_head/√L 的 "PASS"
- **为什么死**："PASS" 是**脚本约定**，非物理结论；且 9.6% 均值 / 33.3% 最大**锚误差**。
- **决定性证据**：`D:digest_failure-records.md:L45`（COSH-REVIEW §2）：surrogate √(β/α)=**6.244/5.704** vs 部署规则 d/√L=**1.414/1.000**，比值 **4.42/5.70** —— τ 的解析缺口有**直接反证**（V-D13）。
- **复活形态与误用警告**：τ 的符号/量级直觉可以留，**解析确定部署 τ 的定理不能留**。

### N-7 Cosh 是"真实 RoPE 应当采用"的被发现最优族
- **为什么死**：**由目标形式选出，不由拟合结果发现**——一旦选定常系数局部平方项 + min-kernel，任何 α、β 拟合都返回 Cosh 族。
- **决定性证据**：`[已验证-逻辑辨析]` `D:digest_failure-records.md:L184`（V-D12，一手 COSH-REVIEW §1.2）。

### N-8 高拟合 R² ⇒ 真实需求服从该先验
- **为什么死**：先验曾被**反向扫描挑选**——24,000 个配置中找到 886 个 R²_mid>0.99；GPT-2 attention 统计不是目标模型的任务敏感距离分布。
- **决定性证据**：`[已验证]` `D:digest_failure-records.md:L186,L44`（V-D14、COSH-REVIEW §1.1 行 43）。

### N-9 二元任务分数的 crossover ⇒ 强非线性隐藏态交互机制
- **为什么死**：multikey margin 序列 **(−2.125, −1.125, −1.000, +0.250)** 近似可加（prefix +1.125、reading +1.000、**交互仅 +0.250**，BF16 度量）。
- **决定性证据**：`[已验证-同前缀可检]` `D:digest_failure-records.md:L191`（V-D19）；`INTEGRATION` §3 prefix/read 交叉 −2.125/−1.125/−1.0/+0.25。
- **复活形态与误用警告**：先按 **margin-跨阈值机制**解释；"非交互"不可证但不得预设。

### N-10 family-close（用搜索失败给族封顶）
- **为什么死**：搜索结论只描述**被测试域/目标/算法/预算**；置零保留内容通道；collapse 只证明依赖被移除旋转。
- **决定性证据**：`[已验证]` `D:digest_failure-records.md:L192`（V-D20）；`D:digest_nongeo-code.md:L54`（"**别关族**"）。
- **四条被点名禁止的家族级关闭**：(i)"300–500 点搜索失败 ⇒ 静态表家族在 +5pp 封顶"；(ii)"若干形状失败 ⇒ support 约束表家族关闭"；(iii)"慢频置零崩溃 ⇒ 证明弧恢复机制/那些维是死重"；(iv)"NLL 可加＋任务差 ⇒ 否证逐槽可加"。第四条有多重反例（可加 logit 经阈值化可产生不可加 accuracy，multikey 即例）。

### N-11 "training-free + static + nongeometric 无人占据"的新颖性声明
- **为什么死**：MrRoPE-Pro 本身逐维 radix 进度、中段累计指数非线性、**非单一全局几何级数**；YaRN 也是分段；LeRoPE 已逐频学习+重训+组合。
- **决定性证据**：`[已验证-先行工作]` `D:digest_failure-records.md:L193`（V-D21）；`D:digest_nongeo-code.md:L54`（"MrRoPE 本身就是非几何 radix 日程"）；`INTEGRATION` §8.5：LeRoPE（arXiv 2607.10134 §3.2）已做频率损失梯度+下游余切。
- **复活形态与误用警告**：新颖性必须落到**新选择规则 / 内容依赖 / 允许的非单调性 / 预测理论**。

### N-12 支持域重定标（support-retargeting）的机制故事
- **为什么死**：公比对称性被**代数否定**；反转是**观察**不是机制。
- **决定性证据**：`[已验证-事实，机制未识别]` `D:digest_failure-records.md:L180`（V-D8）：`R(s)=R0+ln(s)` 坐标定义域仅相应正频率、非退化 support 域；B→sB 增量是 **(K−1)ln(s)/K** 不是 +ln(s)。

### N-13 VICTORY / 已闭合 / 已证明 类证书
- **处理**：项目红线——报告里这类结论**一律降级处理**。新增纪律（`D:digest_failure-records.md:L243`）：**反向也禁止**——不能为了"统一失败理论"把正例、混合结果和后来纠正删掉；**不能一次修 bug 把真实负结果洗掉**。

### N-14 "测试全绿 = 定理被证明"
- **为什么死**：`test_challenger_remediation_verification.py` 把 VETO 字典与 fiber 名字**写死再检查**，属**重述结论**；自洽测试不能替代独立参照。
- **决定性证据**：`[已验证]` `D:digest_failure-records.md:L177`（V-D5）＋工具修复反例 `L64`：新 NumPy 入口 rotate_half 反号，独立绝对旋转修正后偏差 4.09798→**1.44e−15**，而**旧自检对同一错误函数做有限差分仍通过**。

### N-15 非可识别性定理的扩大（三条已撤回）
- **撤回内容**：(a) "任何 Phi"（含完整有序表本身）；(b) "Phi 不同、Y 相近"违反 Y=f(Phi)；(c) 任意 ε 趋零仍失败。
- **保留的最小版本**（可证明）：固定权重/输入/gain/评分下，`Phi(T_A)=Phi(T_B)` 而 `Y(T_A)≠Y(T_B)` ⇒ 不存在只读 Phi 精确解释这两结果的**单值函数**（`D:digest_failure-records.md:L173`，V-D1）。
- **复活形态与误用警告**：**不要**用"不可辨识——定理"式表述（`INTEGRATION` FLAG-1 明令作废）。

### N-16 LoRA 容量错误界
- **撤回**：rank-r 更新除以头数；best-found 残差当"不可修复下界"；各向同性平均当最坏情形界。
- **正解**：每头切片可各有 rank-r；一般 LoRA 可读旧 Q/K 未保留的 hidden 方向（`D:digest_failure-records.md:L175`，V-D3）。

### N-17 "全部低维条件永久淘汰"的二元淘汰矩阵 + 三 fiber 定理
- **撤回**：具体统计量的反例**只否定其声明的充分性/不变性**；"缺少预测、未满足前提、未测量均**不能**填 VETO"。改用 §4.7 的 C1–C8 **最小裁决表**（`D:digest_failure-records.md:L176,L226-238`）。

### N-18 其他已撤回强断言（速查，全部带出处）
| 断言 | 撤回依据 | 出处 |
|---|---|---|
| 由槽位范数不均推任务对 attention 敏感 / 3-nat 损失下界 | attention logit 不是词表 logit | `D:digest_failure-records.md:L174` |
| "Qwen 实际损失变化 −1.81 nats、Taylor 预测 −252.04 nats" | 脚本硬设 m_ref=.25、delta_m=.044109；算的是单个 cos 不是 LM 损失，单位不是 nats | 同上 `L178` |
| "cos 收敛半径有限"；路径积分=免费预测器；旋转差算子界直接界定 LM 损失 | cos 是整函数，问题是低阶截断误差；路径积分是微积分基本定理 | 同上 `L181` |
| CoPE 光滑幅度窗 ⇒ 光滑修改频率一定改善注意力 | 频率映射改谱原子位置 ≠ 乘窗（Δ=0,C=1 时频率置 0 得 1、幅度置 0 得 0） | 同上 `L183` |
| 把旧 p2 新提案尾部标"继承已验证的本方优势"；把 44.8%（1.44759 倍）上限当普遍约束 | 旧 p2 有效结果不证明本方低频加速有效；该上限仅条件性计算 | 同上 `L194` |
| 三频带 / 1-32-cycle 阈值证明三个普遍功能模块 | 见 N-2 | 同上 `L187` |
| "A(target)≥0.5 是 top-1 必要条件" / softmax dilution 是完整模型 | 简化模型 | `D:digest_thread-core.md:L55` |
| G=JᵀJ 的 head/token-first 相等是新 observable | 是**重排**非新 observable | `D:digest_thread-core.md:L55` |
| C_{h,k}/S_{h,k} 均值模长代表 slot 功能重要性 | 正负可相消；不得解释 C2 slot-19 或 permutation collapse | `D:digest_thread-core.md:L55` |
| FullLagP2 1.5B/64K 胜 MrPro ⇒ uniqueness UkU_k 机制 | 128K/3B 已现排序交叉 | `D:digest_thread-core.md:L55` |

---

## 5. 流程-实验类

> 类判词：**判据、口径与调度的治理失败**——这类失败最贵，因为它让上面的错误结论获得"已验证"的外衣。

### P-1 弱基线上宣称进步 / 强基线进入太晚
- **决定性证据**：`[已验证]` `D:digest_thread-main.md:L51,L204`：候选从 Mean **0/32** 提到 10/32、11/32，补实际 **Quest32 基线 13/32** 后"进步"被反超（且其描述缓存只有候选约一半）。OLMo BM 开发面板 24胜1负先报告，独立复核后**仍只覆盖 OLMo**。
- **复发警告**：任何"胜率"先问**基线是谁、是否同预算**；复核必须**换题号非换种子**。

### P-2 用小开发收益授权大扩展矩阵
- **决定性证据**：`[部分证据]` `D:digest_failure-records.md:L213`（V-F1，OVERNIGHT §4.3）：9 项长矩阵耗 **63.8% 作业时间**未形成可比主结论。扩展前必须有"改动影响**正确/干扰区分**"的依据。

### P-3 事后放松判据并当作原判据通过
- **决定性证据**：`[已验证]` `D:digest_failure-records.md:L214`（V-F2）：先写"三项收益保留才进入后续"，遇 128K 混合又用 64K 收益转入 LoRA——"**不能把重写判据当作原判据通过**"。要求**事前固定主终点与各结果分支行动**。

### P-4 来源可追溯 ⇒ 效果可外推
- **决定性证据**：`[已验证]` `D:digest_failure-records.md:L215`（V-F3）：P2Middle 教训——模型、表组成、gain 全变的搬运须先有**跨模型依据**。P2Middle 实测：长 UUID 0 vs Mr 12.5；VT 65 vs 62.5；短能力退化。

### P-5 诊断工具正确 ⇒ 问题被解答
- `D:digest_failure-records.md:L216`（V-F4）：trace 首样两臂在**格式 token**即分歧；全层状态漂移与数值误差限制固定 Q/K 解释。

### P-6 准备 / 文档 / 代码完成 / hash 核对 / Git 提交 ⇒ 科研目标完成
- `[已验证]` `D:digest_failure-records.md:L217`（V-F5）：**无已证实方法突破/超 MrRoPE/可升格主贡献之前，一切"工程完成"不计为研究进展**；hash 只在新增/修改/传输边界核对。

### P-7 把同一 prompt 的多答案/多 head 当独立样本增大功效
- `D:digest_failure-records.md:L218`（V-F6）：**样本单位是 prompt**；"high power"需要目标效应量与配对不一致率，**两次开发胜利不是可靠效应量估计**。

### P-8 要求方法先具备完整理论/复杂校准组件才许有效
- `D:digest_failure-records.md:L219`（V-F7）："不要求补全理论，不以复杂诊断作为方法的必需组件"；简单规则不需要完整最优性定理。

### P-9 计划文档中的成功百分比/时长估计当校准概率
- `D:digest_failure-records.md:L220`（V-F8）：它们是 **planning guesses**；新耗时方案先按真实吞吐核算。

### P-10 与适配不足的旧方案比较后把全部收益归给网格
- `D:digest_failure-records.md:L221`（V-F9，COSH-REVIEW 行 9–13 原文）："零训练、不接触原窗口外距离、只允许短 LoRA，**都不是本次任务自动继承的要求**。新方法需要与相同训练/适配范式下的 Cosh 比较"。

### P-11 把未测写成否证；把轻微门槛失败写成崩溃
- `D:digest_failure-records.md:L222`（V-F10）：三件套判据分开——**科学预测 / 实用验收门槛 / 工程检查**；同时报告效果量与逐行得失。
- **最重要的"未测"实例**：LoRA 正式训练 59/128 步后 SIGTERM（`exit=-15`），只保存最终 adapter 的设计导致无最终 adapter ⇒ **既不能判有效也不能判无效**（V-A8，`L144`；LEDGER 行 900 原文："wall times are not cloud billing. SIGTERM during formal training was an author-requested wrap-up interruption, **not** a scientific or software failure verdict"）。

### P-12 口径混淆类（V-C 组，6 条）
| ID | 被禁形式 | 出处 |
|---|---|---|
| V-C1 | 用变体（如 sqrt 增益 Y2）的零分证明"官方方法上限" | `D:digest_failure-records.md:L162` |
| V-C2 | 把 arithmetic 旧表 / log-p2 / FullLag 当同一方法互相引用 | 同上 `L163` |
| V-C3 | 把 Carrier 的 YaRN 外底座与 Native-sector 的 MrPro 底座串成一个干预 | 同上 `L164` |
| V-C4 | 部分任务集均值（9 项 450 行 62.22）对比论文完整 13 项 macro（53.2） | 同上 `L165` |
| V-C5 | 把 s2/s4/s8 的门槛与分数拼成"同一物理长度断崖" | 同上 `L166` |
| V-C6 | 把 EOS 口径统计（14/50 次结束）改写成"14/50 失败"或反之 | 同上 `L167` |

### P-13 尾部-512 NLL 两文档 OOD 摘要冒充全文档困惑度或八文档面板
- `D:digest_failure-records.md:L223`（V-F11）：live OOD 摘要每 source/length **只有 2 份文档**（可用 8 份），测的是**长前缀后 tail-512 NLL**。

### P-14 拿 stale 的 root `development_summary.json` 当权威证据
- `D:digest_failure-records.md:L224`（V-F12）：权威入口是每个 method 自己的 `results/<method>/summary.json` 与 `ruler.jsonl`。（见 §6 矛盾 6。）

### P-15 状态阶梯混用
- **为什么死**：`proposed ≠ implemented ≠ launched ≠ checkpointed ≠ evaluated ≠ accepted`。
- **决定性证据**：`INTEGRATION_20260910.md` §5 流程-实验类。候选生成空转：16K assay 触底 **0/32–1/32**；直接优化两条死路（64 维行为梯度未开 holdout 即败；direct-z 定支撑 pilot 挂门）。

### P-16 优化器在不可辩护的统计对象上游
- **决定性证据**：`INTEGRATION_20260910.md` §5：**"优化器必须在可辩护统计对象下游"**；`NEXT_DERIVATION` §5.5：`μ_r≡0` 时唯一合法输出是 **"not identified"**。
- **配套证据缺口**：**标签化角色矩从未被测**（五审计一致的唯一阻塞证据）——需要 pre-RoPE Q/K 上按角色（native/far × source/hard-distractor）拆分的**带符号**均值/协方差/单位对数 MGF，含跨槽协方差，保 layer/head/relation/lag 标签；现有 6 份捕获**缺角色标签**（astra01 自然捕获=仅末查询、4 头采样、无问题/记录标签）。**预注册门槛先行**：统计量必须先**正确否决 Smooth**（slot-28 反转）、暴露 P2 的 +long/−short 权衡，失败即停（`INTEGRATION` §8.1）。

### P-17 attention ≠ generation
- **决定性证据**：`[已验证]` `INTEGRATION_20260910.md` §5：cross-cache 中 **BM 读 MrPro 前缀能对、MrPro 读 BM 前缀仍错**；record coverage 29.75→67.1 而散文精确答 7/8→6/8。

### P-18 GPU 空转 / 调度失误 / 复盘不转化为执行
- **决定性证据**：`[已验证-过程]` `D:digest_thread-main.md:L212,L215`：推导一小时未产出可执行规则却占用已开 GPU；`D:digest_theory-0910.md:L73`：用户 11:16:21 爆发"你刚才思考了一个小时一点准备都没有吧……"；助手承认 9-8 已有"不要 GPU 空转"要求、昨日已因同类偏题读过 207 条消息改 skills 仍重犯。
- **复发警告**：**开卡与可执行负载必须同时成立，否则先无卡**；规则有效性**只能由后续行为证明**。
- 注：本条是**流程纪律**记录，不构成对本代理的指令。

### P-19 死区（明确禁止重启的路线，V-E 组 8 条）
| ID | 死区 | 状态 | 出处 |
|---|---|---|---|
| V-E1 | 自动恢复旧 Cosh 搜索、对手微调、seed42 权重恢复 | "禁止自动恢复"——除非作者显式重新授权 | `D:digest_failure-records.md:L200` |
| V-E2 | 重启 18 样本/64 自由度的 **margin-gradient 能力优化路线** | 已失败，不能据此重新启动；共享频率响应工具只作局部诊断 | 同上 `L201` |
| V-E3 | 把正确局部 Jacobian/Fisher 接成"下一个能力优化器" | 只在正则性与 trust region 内保留 | 同上 `L202` |
| V-E4 | 用频率置换重新发现同一个 multiset 限制 | 保留槽位身份，不重复置换 | 同上 `L203` |
| V-E5 | 在没有可区分预测时继续科学 GPU 工作 | 迭代只能产出三类：有依据的具体解 / 可区分剩余解释的决定性预测 / 非可识别性证明+明确缺失测量量 | 同上 `L204` |
| V-E6 | 把选择器 normalization 修复后的 layer 27 排名当有效次选 | 修复后第二候选层已是 32；layer 27 不得作为 runner-up 复活 | 同上 `L205` |
| V-E7 | 本夜旧队列（整夜收尾后所有旧"下一步启动"文字失效） | 需新授权/新材料审查后才可启动 | 同上 `L206` |
| V-E8 | 静态碰撞指标当外推机制解释（旧 full-rope 审计遗产） | 静态 collision ≠ extrapolation 机制；cos-only kernel 只是半个故事 | 同上 `L207` |

### P-20 A/B 组跳接（代理→能力、单因素归因，速查）
- **A 组（代理 ⇒ 能力，9 条）**：V-A1 低崩溃比/低 PPL 比值⇒更稳定；V-A2 Gram/曲率/重构残差/phase risk/覆盖/平滑变好⇒生成变好；V-A3 背景响应下降⇒能力提高；V-A4 NLL/源依赖/target block rank 提高⇒正确输出；V-A5 PPL 改善⇒指令遵循成功；V-A6 冻结态选择器分数⇒全模型改善；V-A7 attention top-1/mass≥0.5⇒生成正确；V-A8 LoRA loss/smoke⇒改善；V-A9 数值修复/数组合法性/hash 一致/guard 取消⇒效果提升。（`D:digest_failure-records.md:L135-145`）
- **A5 的实测反例（最锋利）**：qa16k 三臂 303 题——Base-Native F1 **23.09%** / Native-LoRA 21.10% / EVQ-LoRA **11.26%**；exact 33/25/4；而**同一批 adapter 的 PPL 显示 EVQ 16K 24.068 vs Native 108.958**（`L141`）。EVQ-LoRA PPL 最好（32K 127.9 vs 991.5）但 F1/exact 最差。
- **B 组（单因素归因，6 条）**：V-B1 一次候选失败⇒认定训练量/架构为根因；V-B2 修 bug 后低分⇒"容量是唯一瓶颈"（**双向都不许**）；V-B3 一个 routing/oracle 控制无效⇒attention 不是瓶颈；V-B4 早压缩⇒128K 相位饱和 / 头层数差⇒3B 错配 / 最少 orbit⇒频率缺口失败；V-B5 BM(gain0.074) 32K +12.778pp 记作 gain 独立因果；V-B6 s29 的 32K +8.333pp 说成广泛短上下文恢复（**增益全部来自一行 QA**）。（`L151-156`）

---

## 6. 单列：被明确点名"不得进入 F"的量

> 依据优先级：`NEXT_DERIVATION_KKT_PROBLEM.md` §5（F 的建模红线，6 条）> `INTEGRATION_20260910.md` §1 红线 R1–R5 与 §6 负数据清单 > 各 digest。冲突处按 §7 处理。

| # | 量 | 判决原文 / 依据 | 出处 |
|---|---|---|---|
| 1 | **Σcos 首根（RoPE-bound 标量根）** | "**必须排除出 F**"（CPU 复算证伪根排序与能力排序显著失序） | `NEXT_DERIVATION` §4 第四条；红线 R1 |
| 2 | **碰撞核**（cos-only 核 / Ci 核 / 碰撞能） | §5.1 点名"根、碰撞核、effrank、Gram、曲率、平滑度"不得作为 F 的分项或选择器 | `NEXT_DERIVATION` §5.1 |
| 3 | **effrank / 有效秩 / r₂** | 同上 | 同上 |
| 4 | **Gram**（含 QK 算子 Gram、加权 U_H、cross-Gram） | 同上；且"Gram 不能"清单：不能说激活分布/softmax KL/任务损失/语义信息损失 | 同上；`D:digest_evq-code.md:L260` |
| 5 | **曲率** | 同上 | 同上 |
| 6 | **平滑度**（作为表的静态几何代理） | 同上——**注意与 J[h] 的粗糙度项的区分，见 §7 矛盾 3** | 同上 |
| 7 | **任何无符号二次项** | "无符号项、对角 Σ、纯 pairwise SNR" 负数据清单 | `INTEGRATION` §6 |
| 8 | **对角 Σ** | 同上 | 同上 |
| 9 | **纯 pairwise SNR** | 同上；等均值等方差时正确标量目标是 **v/2−μ**（含绝对 log N 项，比 SNR 强） | 同上；§4.2 |
| 10 | **冻结问题上的密度参数化** | 负数据清单 | 同上 |
| 11 | **线性读出/固定态代理算子**（J_r、校准-KL、E7 160×） | "只配假设生成，**不入 F 主链**" | `NEXT_DERIVATION` §5.5 |
| 12 | **gain 自由度** | "gain×相位不正交，**F 不含 gain 自由度**" | 同上 §5.5 |
| 13 | **覆盖 / 未访问弧比例** | 联合学习对象的边际覆盖不够；E1 槽 28/29 窗内已转 12.37/9.97 圈 | `D:digest_failure-records.md:L188` |
| 14 | **静态几何代理全体**（碰撞能/覆盖/平滑/有效秩/能量/MAE/轨道计数） | R2："**不得作为选择子**"——Smooth_MrBudget 几何全赢、far 端 68.3 vs MrPro 78.13 是决定性反代理 | `INTEGRATION` §1 红线 R2 |
| 15 | **attention mass / attention top-1** | top-1 ≠ 生成正确；A(target)≥0.5 非 top-1 必要条件 | `D:digest_failure-records.md:L143,L174` |
| 16 | **弦距离全局"分离度"** | 对压缩非单调；s28 成功改动在 3/5 距离降低分离度 | `D:digest_nongeo-code.md:L125` |
| 17 | **无界局部步** | 必须 `(1/2)hᵀF_Nh≤ε` ＋ 精确三角重演认证 | `INTEGRATION` §6 |
| 18 | **"17 个 log-gap 之和 = ln S"** 的错误口径 | "不得再写" | `NEXT_DERIVATION` §5.4 |
| 19 | **"所有赢家同向移预算"** | "不得复活" | 同上 §5.2 |
| 20 | **行级预测力声明** | holdout（0450/0451）前 F 不得声称逐行预测力 | 同上 §5.3 |
| 21 | **I1/I2/gain 作为"被证定律"的写法** | 是设计面（R3）；I1 是"强基线设计约束，非零容忍定理" | `INTEGRATION` §1 R3；`NEXT_DERIVATION` §1.3 |
| 22 | **可辨识性边界外推**（"唯一最优密度"式声明） | §4.2 astra07：自由系数规差下任意密度可被吸收；"唯一最优密度"式声明**一律降格为"给定声明协方差下的解"** | `INTEGRATION` §4.2 |
| 23 | **四对象互换** | ①精确-原子最优 ②光滑-Cosh 代理 ③有限-K 整数计数 ④冻结-带标签表——**两两不可互换** | 同上 §4.2 |
| 24 | **μ_r≡0 时的任何唯一解** | "唯一合法输出是 **not identified**" | `INTEGRATION` §6 |
| 25 | **三频带作为定理** | 见 N-2 | `D:digest_failure-records.md:L187` |

---

## 7. 单列：被明确点名"只能作窗口内诊断"的量

> 共同语义：**可以算、可以报、可以用于生成假设与排错，但不得作为 F 的分项/选择子，不得从它跳接到能力结论。** 判违规的形态是"变好 ⇒ 更好"。

| # | 量 | 允许的用途（原文措辞） | 出处 |
|---|---|---|---|
| 1 | **Σcos 根** | "**诊断量，禁入 F**"（红线 R1 明写"根 = 诊断量"） | `INTEGRATION` §1 R1 |
| 2 | **U / U_H（加权或非加权）/ QK 算子 Gram 排序** | 只能当**诊断/解释候选**，不能当预测主张；"U 与本地失真不是充分 selector" | `D:digest_paper-state.md:L162`（U13）；`D:digest_evq-code.md:L131` |
| 3 | **正则化最差方向放大表**（0.1→1e-8 放大 5–15 数量级） | "保留为**诊断**，**不作选择器**" | `D:digest_evq-code.md:L134` |
| 4 | **C2 movement MAE / 耦合律距离** | 只描述冻结 hidden 局部块响应；不以 MAE 放行 | `D:digest_failure-records.md:L232` |
| 5 | **共享频率响应 J_r / G_jk** | "只描述冻结 hidden 局部块响应"；工具只作**局部诊断** | `D:digest_failure-records.md:L267,L201` |
| 6 | **局部 Jacobian / Fisher** | 只在正则性与 **trust region** 内保留；不作全局退休也不作全局优化目标 | 同上 `L202` |
| 7 | **attention mass / top-1 / A(target)** | 简化模型观测；须与主分**分开报告** | 同上 `L143,L174` |
| 8 | **静态 r₂ / collision** | 静态碰撞 ≠ 外推机制；不得作为选择子 | 同上 `L207`；`D:digest_thread-0908-night.md:L99` |
| 9 | **独立相位能量 χ_i** | 丢失 i≠k 相干项；需改用**有符号**共享频率响应 | 同上 `L182` |
| 10 | **E8 冻结态选择器分数** | 代理最优而实际 −13.9pp；代理不含"有用计算的损失"项 | `D:digest_paper-state.md:L175` |
| 11 | **一阶导数/对易子 [M,P] 的图耦合** | 文档自我限定："only a **first-order diagnostic**，S=4 有限变换必须用精确特征响应核查" | `D:digest_theory-0910.md:L129` |
| 12 | **phase risk / 覆盖率 / 残差** | 只证明其实际覆盖的事实；"数学自洽、合法数组、相似几何和代理分数各自只证明其实际覆盖的事实" | `D:digest_failure-records.md:L277`（REVIEW-0907 行 250–257） |
| 13 | **数值不变性诊断**（全局 position-ID +1 平移的 exact 对称性在 BF16 下的实现检验） | 排队中（3 个正例 + MrPro 控制）；翻转将识别**算术敏感**而非任务失败 | `D:digest_failure-records.md:L301` |
| 14 | **筒仓式 B5 选择器（当前几何指标）** | "现有几何指标**只作诊断**"；需重写为**内容条件化带符号 margin** | 同上 `L299`；`D:digest_nongeo-code.md:L125` |
| 15 | **454M QuALITY full-eval** | "只能作 negative/downstream diagnostic，不能拿旧 n=200 pilot 的夸大 accuracy 当 claim" | `D:digest_thread-0908-night.md:L44` |
| 16 | **50M attention-Fisher / LM-gradient probe** | 作为机制边界/负结果短段或补图，**不能写成训练前 selector** | 同上 `L69` |
| 17 | **诊断工具全体** | "诊断工具可以帮助排错，**不能反过来成为方法必须具备的组成部分**" | `D:digest_failure-records.md:L277` |

---

## 8. 材料间矛盾与口径不一致（推导前必须先裁决）

| # | 冲突 | 两侧来源 | 当前裁决 / 处理建议 |
|---|---|---|---|
| 1 | **仓库路径**：digest 写 `/Users/[REDACTED_AUTHOR].yanghejazfs.com.au/paper_project/hybrid-rope/`，实际工作目录为 `/Users/yang/projects/hybrid-rope/` | 多数 digest 的来源清单 vs 本地环境 | 同一仓库两个挂载名。**引用时统一到 `/Users/yang/projects/hybrid-rope/`**，勿当两仓库。 |
| 2 | **"超出 D_j 进入未训练相位弧"** | `UNIFIED_BUDGET_ALLOCATION_THEORY_20260910.md` §1/§2 **vs** GLM/0452（槽 36–39 n_cyc=1.17/1.34/1.62/1.95，100% native 相位覆盖）与 GLM 复核"Native slots 36–39 already rotate >1 turn within W" | **否定 UNIFIED 一侧**：`r_j>1` 的槽超 D_j 后圆周已覆盖，"未见弧"只对 `r<1` 成立（`NEXT_DERIVATION` §1.4 L_far 段）。UNIFIED 是 8 份姊妹文档中**唯一未吸收该否定**的，其 §1/§2 正文需勘误。 |
| 3 | **Smooth "差 9.79 分"聚合表述** | 早期报告 vs 面板 near/far 分解 | FLAG-3：**一律用 near/far 分解**（near 打平 87.2 vs 87.22，**全部损失在 far**）。 |
| 4 | **"平滑度"禁入 F vs J[h] 的粗糙度项是"KKT 最优先复用件"** | `NEXT_DERIVATION` §5.1（禁平滑度入 F）**vs** 同文件 §4（`J[h]=(1/2)∫[α/h(u)+β(1−u)²h(u)]du` 为最优先复用件，其中第二项含 (1−u)²h） | **不矛盾但要写清**：§5.1 禁的是把**候选表的静态几何平滑度**当选择子/代理（Smooth 全赢全输是反例）；§4 的 β(1−u)²h 是泛函内部对**间隔 a_i 的凸惩罚**，有 KKT 边际含义。**推导文档必须显式声明这一区分**，否则是 P-20/V-A2 的复发。 |
| 5 | **保序是普适律还是面选择** | sol19 程序**硬编码保序** vs sol12 记录**非单调有用表**（旧 p2 槽 1/18 有频率交叉尖峰） | FLAG-2：保序是 **MrPro 面的刻面**，不是普适律；F 里放保序 = **声明面选择**，文档必须写明。 |
| 6 | **`development_summary.json` 权威性** | root 的 stale 文件 vs 各 method 自己的 `results/<method>/summary.json` | V-F12：权威入口是**每 method 自己的** summary.json 与 ruler.jsonl。另：`evidence_distances_20260910.json` 本机缺失（`NEXT_DERIVATION` 引用，实际在服务器 `planned_controls/`）。 |
| 7 | **`MrUni` 身份** | 面板 = Δ 均匀 1/17、m28=.294（左倾压 bank 边缘）vs MrRoPE 论文 MrRoPE-Uni = **带内常数** | 两身份不可互换；引用时必须带项目内定义。 |
| 8 | **面板基线不可比** | 12 行面板 vs 36 行面板 | **不得直接比较**；54.7/50.6 那两个数的 baseline 是 **64.4444**，不是 78.125。 |
| 9 | **两方法巧合同分 73.9583** | 面板分数表 | 引用行分时**必须同时给方法名**，否则无法分辨。 |
| 10 | **32K 拉伸位置作校准分布** | sol16 可以 / sol17 oracle-only / sol18 禁止 | FLAG-5 证据分层：拉伸行 = **方向发生器与 oracle 上限**（jsonl 实测支持 sol17/sol18：32K 响应住在被冻结的快槽 0–6）；**模型级判定必须真实连续 128K**。不平均，写标签。 |
| 11 | **16-DOF 坐标** | sol16 Helmert 保序构造 vs sol18 槽坐标加约束 | FLAG-6：同单纯形不同坐标；**选 sol16 形式做无约束求解器，出表前用 sol18 坐标报槽值**；两式互换性在 G1 表里可逐位验证。 |
| 12 | **"E3_BM" 命名** | 多份材料 | FLAG-7：**仍未解决**；这批材料无帮助。保留为开放问题。 |
| 13 | **"语料不足以出新 Qwen 表" vs 交付要求"具体频率表"** | 五份失败审计 vs 交付规格 | FLAG-8：**不矛盾**——交付的表 = 地面真值重建表（已验证数学/CPU）+ 候选表（未过角色门，**明示资格线**）。任何新表发布前提是先跑标签矩测量或 sol18 测试。 |
| 14 | **"16 个短会话批被杀"** 的假设 | 调度简报 vs 实测 | **不成立**：mtime 快照 ≠ 终止状态；须用 task_complete 事件 + 二次采样（`D:digest_thread-0910-batch.md` §0）。**且未发现任何被否决方向被直接重复**。 |
| 15 | **"180 次 rate_limit"** | grep 计数 vs 事件核查 | **假阳性**：每个 token_count 记录都含 `rate_limits` 与 `rate_limit_reached_type`；90 记录 × 2 = 180；所有 `rate_limit_reached_type` 均为 null。真实事件是一段 **112.1 分钟空闲窗口**（`D:digest_thread-0909-pm.md` §1）。 |
| 16 | **"两核/十候选"调度简报** | 简报 vs 该 7 会话簇实测 | 该簇"两核"（PC2 + PM/PMKeep）与"十候选"（E1–E10）**零命中**；加密任务载荷无法完全排除，但助手自报与最终报告一致指向不同任务（`D:digest_thread-0909-am.md` §0）。 |
| 17 | **HANDOFF.md 过时** | 文件 vs 实际状态 | 文件称"round 3 正在审"，实际 r01–r05 已存在、r05 无 review.md、r06–r10 从未运行。 |
| 18 | **E1 冻结规则跨模型迁移** | Qwen7B 32K tie / 128K −1.667pp；OLMo 4K −3.333 / 16K +3.819pp | `[部分证据]`；仍**显著低于其 BM 对照**；泛化未建立（`D:digest_failure-records.md:L98`）。 |
| 19 | **`6Pro 姐夫稿`归档状态** | 只经 UNIFIED §8 与 GLM_6PRO_REVIEW 引用 | **未归档 [已验证的缺席]**（`D:digest_pro-materials.md`）。**唯一可检索凭证是 codex attachments 里的 pasted-text.txt**（`~/.codex/attachments/178672bd-…/pasted-text.txt`，严格只读）。引用该稿的结论时须标明"经二手转述"。 |
| 20 | **GLM_6PRO_REVIEW 推翻 UNIFIED §3（质心/方向）但 UNIFIED §3 正文未更正** | 两文档 | **以 review 版本为准**；UNIFIED §3 正文**需要勘误**（`D:digest_pro-materials.md` §7.7）。 |

---

## 9. 与 F 建模直接相关的未决问题（从各 digest 未决清单合并去重）

1. **中段分歧机制**：为什么有效旧 p2 与 MrPro 在中段后半程（槽 30 起）采用不同压缩？"先解释，再用实际共享频率响应区分可保留的功能"是既定研究次序，**未完成**（`D:digest_failure-records.md:L293`）。
2. **support 反转机制**：support-retargeting 三子排序反转 `[已验证]`，**机制未识别**（同上 `L294`）。
3. **L/P 可见性诊断**：同可见集合、两干预时点的 oracle 因果诊断已备好**未启动**，收紧为"必须能改变一个具体设计取舍"才执行（同上 `L295`）。
4. **K1 未证**：交点随 S 移动的推导（Qwen A/B={24–37}/{38–39}，Llama 16× A/B={19–24}/{25–34}）未证明（`INTEGRATION` §9）。
5. **标签化角色矩从未被测**——唯一阻塞证据（见 P-16）。
6. **gain 的 2×2 factorial**：`MrPro×gain0.074` 臂未补（见 C-6）。
7. **s28/s29 方向性与可组合性**：等步长反转、邻槽控制、s28×s29 交互测试已固定**未评**。
8. **B5 重写**：需要绑定带符号内容竞争分数、在未见过数据上预测 margin 的选择器。
9. **row-wise 压缩算子定义**：相位附着对象（token vs query 重标定）、cache 合同、W 边界回跳反例——**定义前不得实现**（`D:digest_failure-records.md:L300`）。
10. **E10 收益归因**：频率混合 vs BF16 执行路径变化——同钟扩窗 NLL 控制未跑（同上 `L302`）。
11. **统一框架候选**：`phase-dependent, content-conditioned score changes + altered prefix states + amplitude/numerical effects + a task decision boundary` 是当前有用框架；"simpler two-force rule may emerge from them, **but has not yet been demonstrated**"（同上 `L304`）。
12. **rotary subspace 功能分配假设**：作者研究输入，**未验证**；若统一理论以其为前提必须显式标 `[假设]` 并给检验路径（同上 `L305`）。
13. **G_W(δ) 联合窗相关量**：依赖频率间距与距离测度；与 learned 距离分布、内容坐标的关系**未建立**（同上 `L307`）。
14. **K6 四格反事实** = 最高优先 GPU 判决（零新参数，完整 prefill）——`ν^fast=max(ν^Y,ν^M)`、`ν^slow=min(...)`，交互项 = F11−F10−F01+F00；**F 的任何候选形式必须在四格上给出可区分预测，否则视为不可识别**（`NEXT_DERIVATION` §6 K6）。
15. **`budget_*.tex` 残段**（prop:finite-budget 有限预算命题 + 因果权重 Gram）未并入当前稿也未删除——是统一理论"守恒预算"在论文坐标系里最接近的现成数学，**需裁决复活或归档**（`D:digest_paper-state.md:L240`）。

---

## 10. 复发的机检规则（把本清单变成可执行规则）

沿用 `D:digest_failure-records.md:L311-317` 的接口，并补两条本代理新增：

1. **文本层**：出现"⇒能力/⇒SOTA/⇒上限/证明……失败/淘汰全部/永久"与代理名词（PPL、NLL、margin、mass、Gram、曲率、残差、平滑、相位弧、覆盖、orbit、Jacobian、NMSE、**分离度、r₂、U、U_H、χ、MAE**）共现时，强制人工核对对应 VETO 行。
2. **数字层**：任何被引用的分数须带**六元组**（协议、任务集、样本单位、长度、gain、模型），否则触发 V-C4/V-B5/V-F6。
3. **谱系层**：新统计量与已知量做**代数恒等变形比对**，命中即套用旧裁决（防"改名复活"）。
4. **双向层**：既查"未测写成否证"（V-A8、V-D17、V-D20、V-F10），也查"代理写成分"（V-A 全组）。
5. **【新增】对象层**：每个公式先问"它属于四对象中的哪一个"（精确-原子 / 光滑-Cosh 代理 / 有限-K 整数计数 / 冻结-带标签表），并问"坐标系是 m、λ、Δ 还是别的"——两者任一未声明即触发 **C-3 / C-4**。
6. **【新增】坐标层**：任何"守恒/水床/预算"论证，**必须先点名坐标**，并区分"总跨度锁定"与"原生部分单独计"（触发 C-4 / C-5）。

---

## 11. 覆盖声明（诚实边界）

**已全文读完（16/16）**：`digest_failure-records.md`(318)、`digest_theory-core.md`(346)、`digest_panel-results.md`(221)、`digest_theory-0910.md`(326)、`digest_mrrope-evq.md`(241)、`digest_nongeo-code.md`(303)、`digest_evq-code.md`(283)、`digest_pro-materials.md`(178)、`digest_paper-state.md`(285)、`digest_thread-core.md`(234)、`digest_thread-main.md`(343)、`digest_thread-0907.md`(203)、`digest_thread-0908-night.md`(179)、`digest_thread-0909-am.md`(159)、`digest_thread-0909-pm.md`(162)、`digest_thread-0910-batch.md`(157)。另加三份权威锚定文档全文：`INTEGRATION_20260910.md`(131)、`NEXT_DERIVATION_KKT_PROBLEM.md`(146)、`STARTING_POINT_YARN_VS_MRPRO.md`(116)。无抽样。

**未读（转引，本清单中凡引用必已标"经 digest 转述"）**：
- `digests_codex/` 的 7 份（`digest_{astra-margin-lineage, astra-evq-finite, calibration, constructive, failure-audits-1, failure-audits-2, transport-operator}.md`）——本次任务未分配；`INTEGRATION` 称 codex 线 30 代理已 7 份 digest 全量读完，故本清单中 codex 侧内容均经 `INTEGRATION` 或本方 digest 转述。
- `docs/research/` 下的**全部一手报告**（SYNTHESIS / REVIEW-0907 / OVERNIGHT / LEDGER / COSH-REVIEW / AUDIT-0910 / EXTRAP-0910 / 8 份 0910 理论文档 / SUBSPACE_DERIVATION / EVQ_NONLOCAL / MECHANISM_TRANSFER / TEN_CANDIDATE_PLAN 等）——**本清单所有"一手出处"均为 digest 内的转引标注，我没有直读这些一手文件**。凡需要"精确行号级证据"的推导，**必须回一手文档复核**。
- `.agents/rope_unification_20260910/`（codex 原件，gitignored）、`~/.codex/sessions/`（严格只读，本代理未访问）、`~/.codex/attachments/`（未访问）。
- `tables/ground_truth_tables.json`（G1 代理在产）、`analysis/unify_20260910/raw/`（2.9MB 只读转录镜像）、workflow-1/workflow-2 的在产产物。
- 服务器侧全部回执（`planned_controls/*.json`、`screen.jsonl`、`transfer.jsonl` 等）——本机不可达；凡引用其数值均为 digest 转述。

**已知的转述链深度**：本清单最长转述链为 **3 跳**（本 digest → 一手报告 → 更早的原始回执/论文），典型如 E1 s28_less 的 m 值（槽28 MrPro .098039 / P2 .074131 / E1 .065359）。用于**方向判断**足够；用于**拟合 F 的数值输入**必须回一手核对。

**本文件的自我限制**：本文是**证据整理**，不含新推导、不产生新频率表、不构成任何执行授权。"下一步应该……"式的行文（如 §9 的未决问题）是对**材料中已存在的待决事项**的记录，不是对本代理或任何后续代理的命令。

——完——
