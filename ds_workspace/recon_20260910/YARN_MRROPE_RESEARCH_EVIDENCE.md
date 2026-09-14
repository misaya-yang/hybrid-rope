# YaRN→MrRoPE-Pro 研究依据报告（证据整理与核查）

## 2026-09-12复核：BM是否只改z

**相对同scale、同gain的MrRoPE-Pro，原版BM只改内部z；相对Native则不是。**
本节直接核对[BM实现](../../scripts/lib/rope/boundary_matched.py)、
[原协议](../../docs/research/ROPE_MRPRO_BM_PROTOCOL_20260908.md)、
[实际64槽CSV](../../docs/research/ROPE_MRPRO_BM_OLMO_SLOTS_20260908.csv)和
[论文坐标定义](../../paper-2027/sections/02_exponents.tex)，并用标准库math/Fraction复算；
没有运行模型、启动训练或改变GPU队列。

论文坐标为`x_j=-ln(nu_j)=a+R*z_j`。战役坐标为
`nu_j=omega_native,j*s^(-m_j)`，其中`omega_native,j=B^(-j/K)`。
当两张部署表均有`m_0=0,m_(K-1)=1`时：

`a=0; R=((K-1)/K)*ln(B)+ln(s)`，

`z_j=((j/K)*ln(B)+m_j*ln(s))/R`。

所以同B、K、s的表之间有`delta_z_j=ln(s)/R*delta_m_j`。
三段式的高频保持、低频全插值，是对可修改z的约束；不是z坐标之外的额外算子。
若对照是Native，低频端点还要除以s、R增加ln(s)，gain也由1变为
`1+0.1*ln(s)`，不能把整个Native→BM称为只改z。

OLMo归档数值复算结果：K64、B500000、W4096、s4，MrPro与BM均a=0、
R=14.303620846936004（来自实际FP32表），gain=1.138629436111989。
只有零基槽15–31共17个内部频率改变；0–14与32–63完全相同。
固定的还包括模型权重、槽位对应、普通RoPE旋转和位置ID规则（冻结对照中）。
这支持BM相对MrPro的纯z身份，不要求另外启动证明身份的GPU实验。

BM将MrPro的递增增量`epsilon_i=2i/[N(N+1)]`换成
`6i(N+1-i)/[N(N+1)(N+2)]`。N18时，中点q9的累计减速指数从
0.263158变为0.5，频率约为MrPro同槽的0.720123倍。
它将慢端附近集中的增量分散到中段，减小接入低频平台前的增量跳变。
最小粗糙度是这条闭式的构造理由，尚不是模型任务损失的最优性证明。

**只改z不等于保持sum(m)。** MrPro的sum(m)=113/3=37.6667；BM为
81/2=40.5，增加17/6。这个总量也是z向量的函数，发生变化仍然属于z干预。
已有同面板MrUni与BM恰有相同sum(m)=40.5，16K六任务分数32.12%对
51.32%，见[结果owner](../../docs/research/ROPE_OLMO_BM_RESULT_20260908.md)。
因此现有证据已说明总减速量不足以描述该面板行为，无需把BM/MrPro对照
错误升级为保持了所有“预算”的比较。

同理，C42与C42V24保持sum(m)=42、频带、平台和增量质心，是更进一步的
形状对照。它们与BM不混作同一方法；BM提供低复杂度实用构造，受控对解释
仅用总量/质心不能刻画全部z效果。后续现有5090复评结果以
[当前交接owner](../../docs/research/next_stage_20260912/5090_SOL_HANDOFF_20260912.md)
为准，不用旧350行开发集替代新面板。

研究判断：寻找YaRN/MrRoPE型改进与论文的z分配主线是同一设计问题。
BM在继承扩展端点与gain后改变中频分配，已有OLMo冻结任务/NLL证据；
当前LoRA属于该表上的权重适配，训练后收益同时包含权重变化，不能称冻结
纯z效应。Cosh的解析先验与BM的边界匹配是不同构造，不互相代替验证。

旧战役文档中的“频率收益最多±5pp”“只有gain有效”“所有频率代理都必然选错”
超出了已测面板和方向；D_pattern是逐样本变化量，不天然等于随机噪声下限。
保留对应失败记录，不据其停止整个中频分配方向。官方YaRN代码的索引ramp
与论文转数ramp的差异仍按下文§1.2明确标注，不能混成一个基线。

**2026-09-11** · 任务：整理 hybrid-rope 项目全部相关历史证据，供后续理论分析与实验共同使用。
范围：只研究"YaRN→MrRoPE-Pro 究竟改变了什么、在哪些条件下改善真实任务、为什么"。EVQ 等其他方向只保留对当前问题有直接约束的证据。
性质：证据整理与核查报告。未启动新 GPU 实验；只做了零 GPU 复算与服务器只读盘点。

证据基础：入口文档链（MASTER_SUMMARY、audit/pro_decision REPORT、RESEARCH_OWNER、index、MRROPE_RECONCILIATION）→ 28 份战役判决文档 → 两层代码（表代数 + runner）→ 本地同步 jsonl 抽查重算 → 服务器 `/root/autodl-tmp/{phase1_20260910,rope_decision_20260911}` 只读盘点 → MrRoPE 论文原文（本地 `~/Downloads/RoPE_Papers/Markdown/5551_MrRoPE_Mixed_radix_Rotary.md`，对应 arXiv 2601.22181v1）→ YaRN 官方实现（jquesnelle@995db5b，已重新拉取核对）。

**本报告新建立的事实**（此前任何单一文档都没有完整记录）：
1. 官方 YaRN（通道索引 ramp）**在 OLMo 上测过**：72 行复核集 16K = 6.94%（与 MrRoPE-Pro 2.78% 同为灾难）；此前文档"真 YaRN 从未测过"只对论文 A.2.1 前置变体成立。
2. **fresh_72 独立样本六臂验证已完成**（服务器 rope_decision_20260911，与全部旧提示零重叠）：官方 YaRN 与 MrRoPE-Pro 在 16K 上**几乎无差**（−0.5pp），同差 BM 约 −34pp；在 4096 窗内腿上 MrRoPE-Pro 反而差于官方 YaRN −24.5pp（名义 t≈−3.0，单次）。
3. LongBridge 符号对两腿已读（并行会话收完）：350 行面板 +11.83pp（t=5.85，65% 来自伪影任务 niah_single_3）；held-out 上 slower vs BM **翻负 −3.31pp（t=−1.47）** ⟹ 预注册判据链失败，方向关闭。
4. 服务器 2026-09-11 下午重启后 **GPU 当前不可见**（`nvidia-smi` No devices found），全部实验进程消失；多个在跑臂成为中断点（附 C）。

---

## 0. 一页摘要

**问题**：YaRN→MrRoPE-Pro 只改中段 18 根槽的拉伸预算装填（论文口径）；这个改动在什么条件下改善真实任务、为什么。

**核心答案（证据版）**：
1. **"YaRN→Pro 的收益"在我们的仪器上从未出现**。唯一在可比条件下（同 gain、同带规则、零训练）直接对比两者的三个数据点：OLMo 72 行集（YaRN 6.94 vs Pro 2.78，同级灾难）、OLMo fresh_72（YaRN 0.1406 vs Pro 0.1354，差 −0.5pp 不显著）、Qwen2.5-3B 36 行（无判别力，SE≈13pp）。论文报告的收益（LLaMA2/3、Qwen2.5-3B，PPL 0.03–0.3 / RULER 3–7pp）存在于我们的矩阵没有覆盖的模型×任务区。
2. **"官方 YaRN"与"论文里的 YaRN"是两个不同算子**（审计 P1，本报告复算确认）：官方代码在通道索引上线性 ramp，其 ε 序列**递增（后置）**，OLMo sum_m=38.49；论文 A.2.1 用转数混合，λ 递减（前置），sum_m=42.91。MrRoPE-Pro（ε 等差递增，更极端后置，sum_m=37.67）相对官方代码只是"后置程度的加深"，不是"前置→后置的反转"。相对论文变体才是反转。
3. **OLMo 上真正的巨大分界不在 YaRN↔Pro，而在"三段式后置族（YaRN-index / Pro / Uni）↔ 中置宽带族（BM 系）"**：0.07–0.14 vs 0.42–0.56（350 行 + 72 行 + fresh_72 三次一致，跨 RULER/NLL 两仪器）。这一族差异是 BM（本项目自创表）与文献方法的差异，不能回答"YaRN→Pro 改了什么"。
4. **条件收益的稳健结构是长度交易**：宽带 BM 族在 16K 检索 +6~10pp / 4K 窗内 −4.5~−6.4pp，在三块独立数据（holdout180 原始行、唯一提示修正、fresh_72）方向一致。论文侧的对称结构：Pro 的最大收益格在**训练窗内**（LLaMA2@4K PPL 5.72 vs YaRN 6.02）。
5. **gain 是比频率形状大一个数量级的杠杆**（同表 1.0→1.1386 = +38pp，350 行 t=15；fresh_72 上 gain 1.1191 = +8.4pp），且 gain×表是乘法交互（native@g1.0 在 16K=0）。论文协议（YaRN mscale=0.1·ln s+1）方向在这台模型上被强烈支持。

---

## 1. 我们实际上比较了什么

### 1.1 统一坐标系

全部表在一个坐标系里比较（`experiments/curvature_20260910/tables.py`，纯 numpy 参考实现）：

```
ω_j = θ^(−j/64)          原生频率（j=0..63，K=64 槽）
ν_j = ω_j · 4^(−m_j)     替换频率；m_j∈[0,1] 为压缩指数
m_j = ln(ω_j/ν_j)/ln4
```

**术语警示（战役文档曾混淆）**：
- `s`（小写）= 表构造的扩展倍数，本战役全部 s=4；ν 慢端 = ω/4。
- `S = Σm_j`（战役文档口语中的"预算"）是完全不同的量（BM=40.5，MrPro=37.67 等）。
- 论文的 `s_j = Π_{d<j} λ_d` 是累积因子（慢端 = S 倍减速 = m=1），与上面两个都不同。读任何旧文档时先确认它说的是哪一个。

模型几何（决定频带位置）：

| 模型 | θ | 训练窗 W | 频带（零基槽） | n | 出处 |
|---|---|---|---|---|---|
| OLMo-2-0425-1B-Instruct | 5e5 | 4096 | [15,32] | 18 | `band_from_turns(1,32,…)`，与归档部署表一致 |
| Qwen2.5-3B/1.5B | 1e6 | 32768 | [23,40] | 17 | 同上；论文附录 B 最佳 (23,40) 逐位一致 |

带边界规则三家相同：`j = head_dim·ln(W/(turns·2π))/(2·lnθ)`，YaRN `find_correction_dim` 取 floor/ceil，(α,β)=(1,32) 是转数窗（官方代码 L4-17 已重新核对；论文 Eq.26 同式）。

### 1.2 "YaRN"的五个不同对象（核查结论）

| # | 对象 | 定义 | ε 装填方向 | OLMo sum_m | 状态 |
|---|---|---|---|---|---|
| 1 | **官方 YaRN 代码**（jquesnelle@995db5b，pinned；HF 同族） | 通道索引 ramp：`t=(j−lo)/(hi−lo)`，`ν'/ν = 1−(1−1/s)·t`，`m = −ln(1−(1−1/s)t)/ln s` | **递增（后置）**，first_eps=0.031→last=0.111 | **38.4918** | 我方复算与 `formula_results.json`、本地 `prepared_controls_01/tables.json` 的 OfficialYaRN（sum_m=38.4918）三方逐位一致；与 Qwen 归档 `YaRN_linear_official` 对到 8.4e-8 |
| 2 | **论文 A.2.1 的 YaRN**（MrRoPE 叙述中的"regressive"） | 转数 `r_j=W·ω_j/2π` 上的线性混合：`ν'/ν = retain + (1−retain)/s`，`retain=clip((r−α)/(β−α))` | λ 递减 ⇒ **前置（凹 m）**，first=0.034→last=0.009 | **42.9117** | 论文 Eq.19–25 核实：证明依赖 `r_j²=r_{j+1}r_{j−1}`（转数几何），这只对转数混合版成立，**不描述官方代码** |
| 3 | 战役 FOUR_CORNERS 的"(0,0) YaRN 顶点" | ε 均匀 | 均匀 | 40.50 | **实为 MrRoPE-Uni（论文 Eq.13）**，错标已更正（commit 92c39a9）。P5 的预测对象 |
| 4 | `exact_yarn_olmo.json`（SATPRO P9 预注册臂） | = 对象 1 的 OLMo 版 | 递增 | 42.91* | 未跑（*该 JSON 是官方索引 ramp 但带内细节见 SATPRO；P9 待执行） |
| 5 | `yarn_turns_paper`（rope_decision/tables.py 新算子） | = 对象 2 的可运行版 | 前置 | 42.91 | probe_01 判定"官方索引与转数 YaRN 作为算子不同"；任务面板未测 |

**含义**：MRROPE_RECONCILIATION 的"YaRN 前置 → Pro 后置是唯一改动"这一叙述，**只对论文的 YaRN 变体（对象 2）成立**。对官方代码（对象 1），YaRN→Pro 是"后置→更后置"（sum_m 38.49→37.67，首槽 ε 0.031→0.0059）。审计 REPORT P1 已指出此点；本报告复算确认数值。

### 1.3 MrRoPE-Uni / MrRoPE-Pro 的本地复现

- **论文 Eq.14（已从原文核实）**：`ϵ_j = 2(1+j−dl)/((1+dh−dl)(dh−dl))`，`λ_j = S^{ϵ_j}`；带外 λ=1（Eq.16）；累积因子 `s_j=Πλ`（Eq.15）。
- **本地等价形式**：`m_mrpro`（tables.py L71）与 `m_incr_beta(0,…)`：`ε_q = 2q/(n(n+1))` 等差递增，`m_q = q(q+1)/(n(n+1))`，`m=1` 从带顶起。与论文逐位等价（q(q+1)/(n(n+1)) 是 2q/(n(n+1)) 的累积和）。
- **Uni（Eq.13）**：λ 常数 = S^{1/n} ⟺ ε 均匀 ⟺ m 线性。OLMo sum_m=40.50（与 BM 相同——同预算不同形状）。
- **精度**：本地 float64 构造对 Qwen 归档 MrPro 部署张量 bit-exact（G1 验证，差 ≤3.7e-8）；MrPro 在 OLMo 的 350 行臂（0.0709）用同一构造器。
- **作者原实现**：论文声明代码在补充材料（Reproducibility Statement），**仓库内没有、我们从未拿到**。所有"MrRoPE"臂都是本地复现（公式已对论文逐条核对）。标注：无原实现对照。

### 1.4 BM（MrProBM）——本项目自创表，不是文献方法

- 全名 "MrPro-BM"（boundary matched）。闭式：`ε_i = 6i(N+1−i)/(N(N+1)(N+2))` 即 `ε ∝ k(n+1−k)`（对称帐形=中置装填），`m = q(q+1)(3N+2−2q)/(N(N+1)(N+2))`。来源：外部 review 给出的闭式（`paper-2027/research/external-reviews/MRPRO_BOUNDARY_MATCHED_SOURCE_20260908.md`），数学上是 Σε=1 下 Dirichlet 最小粗糙度解。
- 代码：`scripts/lib/rope/boundary_matched.py` L13–53；协议 `docs/research/ROPE_MRPRO_BM_PROTOCOL_20260908.md`；OLMo 首测 commit `8ec9d1f`（2026-09-08）。
- 家族语言：`m_incr_beta(b)` 的 ε ∝ k^a(n+1−k)^b，b=0 即 MrRoPE-Pro、b=1 即 BM。**b 轴是"Pro↔BM"的连续插值**，这是战役中段形状扫描的实际对象。
- **部署对照关系**：350 行面板的"MrProBM/BM/部署表"三者同名同物；`turns_a1_b64` = 宽带版 BM（带 [12,32]，逐位核验）；`wide_b4` = 宽带 b=4。

### 1.5 Gain（幅度缩放）

- 全战役冻结 `1.138629436111989` = YaRN `get_mscale(4)` = `0.1·ln(4)+1`（官方代码 L37-39 已核对；论文附录 A.2 同式 t=0.1ln(s)+1，Eq. after 1112）。
- 它作用在 cos/sin 上（attention 幅度），与频率表正交。**所有历史臂比较都固定在此 gain**——gain×表交互被量化后（§3-E）这成为一个已知的边界条件。
- fresh_72 的 `gain_calibrated` = 同 BM 表、gain=1.1191034（gain_only_01/candidate_02）；`decision_calibrated` = 校准表 sum_m=46.59、gain=1.1121720（calibration_01/candidate_02）。

### 1.6 精度与索引细节（审计确认，引用时必须携带）

- **midpoint EVQ 身份更正**：旧 `evq_shift` 臂 = midpoint 位移装在 endpoint-native 网格上，全部槽乘 θ^(1/2K)（OLMo 1.1080 / Qwen 1.1140）。它是"合法的 midpoint 位移干预"，**不是** canonical EVQ 网格。三个 0.0 结果按此窄身份保留。
- **索引偏移**：m 坐标用 `u=j/K`（endpoint）重建部署表；EVQ canonical φ 用 `(j+0.5)/K`（midpoint，`scripts/lib/rope/schedules.py:162`）。新代码（rope_decision/tables.py）已把 `evq_endpoint_t1`（τ=0 恰为 native）与 `evq_midpoint_t1` 分开。
- **通道索引 vs 转数 ramp**：见 §1.2。任何"YaRN ramp"引用必须注明是哪一个。
- **BM archive vs rerun**：归档 MrProBM 与 2026-09-11 同名重跑差 24/350 分数、210/350 文本；表字典 SHA 相同而重建张量 20 槽差 ≤5.96e-8（float32 舍入路径不同）。归因未隔离——**旧文本不可当精确重放目标**。
- **旋转实现**：修正后的 `FrozenRoPE`（curvature/model.py，工作区 diff）只剥 stock 的 no_grad 外壳，保留 OLMo2 FP32 cos/sin；7 项 CPU 测试过（tests/test_rope_*）。历史任务分数不受影响（olmo_beta.py 直装表，不走该补丁）。

---

## 2. 作者怎样解释 YaRN→MrRoPE 的收益（核查版）

来源：论文原文（本地 markdown，对应 arXiv 2601.22181v1）+ YaRN 官方代码。逐条区分**恒等关系 / 建模假设 / 经验观察 / 未建立的因果**。

### 2.1 论文主张拆解

| 主张 | 类型 | 核查 |
|---|---|---|
| MrRoPE 框架：任何 RoPE 扩展 = 选 λ 向量（Eq.10） | 恒等（重参数化） | 成立；我方 m 坐标是其等价形式 |
| YaRN 的中间维 λ 递减（"regressive"，A.2.1） | 数学事实 | **只对转数混合版 YaRN 成立**（依赖 r_j 几何）；官方代码（索引 ramp）的 ε 递增。论文与官方实现之间存在未被论文承认的分歧 |
| "低位外推、高位插值、中间达到 S"是有效转换的原则（§3.2 开头） | 建模假设（从 YaRN 经验归纳） | 不可证伪形式；我方数据不支持其普适性（OLMo 上三段式后置族全灾难） |
| Pro 渐进转换"更好保留高频细节、更忠实外推中间区"（§3.2.2） | 机制假设 | 无直接注意力证据；论文的支撑是间接的（下两行） |
| Pro 提升理论上下文上界 1K→28K vs YaRN 6K（RoPE Bound Theory，Fig.5） | 数学（cosine 和的根）+ 解释 | 上界数字是恒等计算；"上界根 ⇒ 实际窗口"是推断。我方未复算 28K/6K（可零 GPU 复算，未做） |
| Pro 稳定中间维 attention 分布、匹配预训练模式（§4.4, Fig.6） | 经验观察 | 50 对 token 的 attention 采样图；无统计检验；模型 = LLaMA2/3。**从未在 OLMo 上测** |
| 渐进 ε（等差）是"期望的渐进序列"（§3.2.2 Eq.14 前的假设） | 假设 | 等差是"progressive"的最简选择，非推导最优；论文自己也扫了 (dl,dh)（附录 B.1）而没扫 ε 形状 |
| 实验设定 | — | **training-free / inference-only**（§4.1 明确 no fine-tuning）。⚠ 更正：我方 GPT6PRO 简报 Q6、COVERAGE 理论文档"MrRoPE 论文收益来自训练时用表的设定"的说法**与论文原文相反**，应全部改读 |

### 2.2 论文支撑实验（数字核实）

- PPL（proofpile，Table 1 + C.1）：LLaMA2-7B@4K **Pro 5.72 / Uni 5.84 / YaRN 6.02**（窗内行是 Pro 最大收益格）；LLaMA3-8B@128K Pro 2.34 / Uni 2.41 / YaRN 2.38。
- RULER 13 子任务（Table 2）：LLaMA3-8B 128K **Pro 86.6 vs YaRN 79.9**；Qwen2.5-3B 128K **Pro 53.2 vs YaRN 50.1**（两侧都只有 ~50%）。
- NIAH：Pro 把 LLaMA3 有效窗推到 ~96K。
- 超参：α=32/β=1（论文记法，即我方 (α,β)=(1,32)）跨模型近最优、不敏感（Fig.7/8）。
- **16× 的机制**：S 按目标重设（LLaMA2 16× 配置慢端 m=log₄16=2），不是同一张表硬拉。我方 s8 用 m≤1 测 8× 得全零，与该协议一致（是违反协议的应有结果，不是家族反例）。

### 2.3 论文、官方实现、我方解释三方冲突清单

1. **论文 YaRN（前置）vs 官方代码 YaRN（后置）**：A.2.1 的"regressive"定性不适用于官方部署代码。⇒ "YaRN→Pro 只改装填方向"的叙述需要限定；对官方代码而言 YaRN 与 Pro 同属后置族，差别是 ε 剖面陡度与 sum_m。
2. **论文 training-free vs 我方文档"训练时用表"**：见 §2.1 末行更正。此错误影响过 EVQ_LONGRANGE 的边界表述与 GPT6PRO Q6 的问题框架。
3. **论文收益 vs 我方零收益**：论文矩阵（强模型 / PPL+RULER / 窗内行多）与我方矩阵（OLMo-1B / 检索面板 / 16K 外推）无交集。两侧不矛盾但互不可推。
4. **"YaRN 顶点 = 均匀 ε"**（我方 FOUR_CORNERS 勘误）：论文里均匀 ε 是 Uni（Eq.13），YaRN 不是它。P5 预测对象已改名。

---

## 3. 实验总索引（按实际干预归类；计划/执行/验证分开标注）

主仪器：OLMo-2-0425-1B-Instruct（冻结，零训练）。面板：350 行选择面板（7 任务，开发用）；72/180 行 held-out；391 行自然 QA；72 行 fresh_72（全新提示，零重叠）；s8 面板（4K+32K）。连续：teacher-forced tail-512 NLL（16 篇 PG19 配对）@4096/16384 等。

### A. YaRN / MrRoPE / BM 三角直接对比（核心问题所在）

| 实验 | 模型/面板 | 干预 | 结果 | 状态 | 数据位置 |
|---|---|---|---|---|---|
| **三表复核集**（2026-09-08） | OLMo，72 行（4K×4+16K×8×6 任务，独立 seed） | MrPro / MrProBM / MrUni / **OfficialYaRN** 四臂同输入 | 16K：BM 51.32 / MrUni 32.12 / **OfficialYaRN 6.94** / MrPro 2.78；4K：81.81/76.88/54.38/37.85；BM 对 MrPro 44胜0负 | 执行完成，decision.json 存档 | `results/olmo_fast_screen_20260908/run_holdout_01/`、`run_controls_01/`（本地+服务器） |
| **OLMo NLL** | OLMo，16 篇 @16K | 同上两臂 | MrPro 3.68798 vs BM 2.86206（16/16 篇 BM 低，Δ −0.826，CI[−1.08,−0.62]） | 完成 | `docs/research/ROPE_OLMO_BM_NLL_RESULT_20260908.json` |
| **350 行面板端点** | OLMo，350 行 | MrPro（归档臂）vs MrProBM | **0.0709 vs 0.4167**（本报告本地重算复核 jsonl：0.0709/0.4167，350 行完整） | 完成且复审 | 本地 `work/jsonl/archive/{MrPro,MrProBM}.jsonl`；服务器 phase1 |
| **fresh_72 六臂**（2026-09-11） | OLMo，72 行全新（4096×24+16384×48，6 任务，与 2227 旧行零重叠，验证过 identity_check.json） | bm / b4wide / gain_calibrated / decision_calibrated / **yarn_index** / **mrpro**（后两者同 gain、同 native 网格构造的 fresh 对照） | @16384：b4wide 0.5729、gain_cal 0.5660、bm 0.4823、decision_cal 0.3885、**yarn_index 0.1406、mrpro 0.1354**；@4096：b4wide 0.7993、bm 0.775、gain_cal 0.7889、**yarn_index 0.5104、mrpro 0.2653**。配对：yarn−bm@16K **−34.2pp**（se 5.0）；pro−bm@16K **−34.7pp**（se 4.8）；**pro−yarn@16K −0.5pp（se 1.4，无差）**；**pro−yarn@4096 −24.5pp（se 8.1，t≈−3.0 名义显著）** | 执行完成+验证（equal-task macro，prompt SHA 配对，读数器 `read_validation.py` 经 test_rope_validation_statistics 测试） | 服务器 `/root/autodl-tmp/rope_decision_20260911/{validation_reference,validation_methods,validation_readout.json}` |
| **Qwen2.5-3B 36 行**（2026-09-08） | Qwen3B，12@32K+24@128K（论文 Table 2(b) 配置） | MrPro vs MrProBM | 32K 87.22/91.67；128K **78.13/70.83**（BM−Pro −7.29pp，6胜4负；归档自判 NO_LONG_GAIN） | 完成但**无判别力**（SE≈13pp） | `results/bm_transfer_20260908/run_qwen3_01/` |
| Qwen2.5-7B 36 行 | Qwen7B | 同上 | MrPro 83.33/84.44 vs BM 80.00/71.11（128K +13.3pp，同 SE 量级，不显著） | 完成，无判别力 | `results/nongeometric_screen_20260909/transfer/qwen7b/` |
| Qwen in-window NLL | Qwen3B 32K | MrPro vs BM | 2.08899 vs 2.08789（差 −0.001，CI 跨零） | 完成，零 | `ROPE_QWEN3_BM_NLL_RESULT_20260908.json` |
| qwen4x_power | Qwen1.5B @131K，29 篇×3 臂 | mrpro/bm/native | **只 2/87 行**（服务器重启杀死） | 中断 | 服务器 `phase1/qwen4x_power/rows.jsonl` |
| qlev 两长度 | Qwen1.5B @98305/131073 | mrpro/bm vs native（同进程） | mrpro−native −0.0395/−0.1730；**BM−mrpro −0.0014/−0.0074**（4 篇，无功率） | 完成（注意 native 参照带 gain=1.1386，非纯 native） | 服务器 `phase1/qlev_L*` |

### B. 中段形状族（b 轴：Pro↔BM 连续插值；350 行选择面板 + NLL）

b 扫描：MrRoPE(b0) 0.0709 → b0.25 0.1447 → b0.5 0.2320 → **BM(b1) 0.4167** → b2 0.5001 → **b3 0.5587**（NLL 2.8627→2.8267 单调改善）。局部复核：本地 jsonl 重算 b3=0.5587 ✓。状态：完成；**但 b3 的 +14.2pp 在 held-out 上归零**（§5）。宽带变体：`turns_a1_b64` 0.5384、`wide_b4` 0.5433（带 [12,32]）。C42/C42V24 受控对：同 S=42 差 10.73pp（t=5.47，RULER+NLL 双仪器同向——本战役唯一跨仪器验证的二阶效应）。数据：`work/jsonl/olmo*`、服务器 phase1。

### C. 频带边界

turns (α,β) 四变体（(1,16)→0.2651 即 turns_a1_b16 灾难边缘；(1,64)→宽带 0.5384；(0.5,32)/(2,32) 次之）；宽带 lo 14→11（+12pp 选择面板）；step 位置 hi=22（0.4893）vs hi=25（0.1121）——38pp 全在槽 22–24。P6 阶梯 hi19/20/21/23/24：**全部未跑**。状态：选择面板完成、out-of-sample 大多失败（§5）。

### D. ramp/阶跃/插值极端对照

native（4× 必崩，NLL 7.2）、interp m≡1（不可用）、step42（见 C）、smoothstep/power 家族（部署史：Qwen 归档 `YaRN_smoothstep_variant` 等未执行）。状态：完成，结论稳定（极端表两端都坏）。

### E. Gain 轴（最大杠杆）

| 干预 | 结果 | 状态 |
|---|---|---|
| BM 表 gain 1.0 vs 1.1386（350 行） | **0.0371 vs 0.4190（+38.2pp，t=15.05）**；审计重打分 900 行零错 | 完成且复审 |
| native 表两 gain（350+180 行） | 16K：g1.0 = **精确 0**（350 行零 EOS）；heldout：0.2361 vs 0.2351（4K 腿有分处无差） | 完成 ⟹ gain×表乘法交互 |
| a1_b64 两 gain（350 行） | 0.0903 vs 0.5384（+44.8pp） | 完成 |
| g2x2 其余 4 臂（mrpro/b3 × 2 gain） | **18/350 孤儿**（supervisor 死亡，重启后未恢复） | 中断 |
| gsweep（BM 表 gain 1.05/1.10/1.20） | 1.05、1.10 各 350 行完成（未读出汇总文档）；**1.20 只 83/350** | 部分 |
| fresh_72 gain_calibrated（BM 表，gain=1.1191） | @16K **+8.37pp**（se 5.3）vs BM；@4096 +1.39pp | 完成且独立样本 |
| 连续 NLL 侧 | native：g1.0 好 0.149；BM：gYarn 好 0.29（@4K）——**符号随表翻转** | 完成 |

### F. 代理目标/建模推导路线（全部失败，见 §5）

Fisher 二次（step_hi25 0.1121）、逐槽梯度（SNR<1）、12 静态泛函（三层受控对全灭）、Tstar/N 最大化/释放平台（六表全差于 BM）、condEVQ（350 行 +2.13pp t=1.11 净行 −1，NLL +0.036 更差）、走线（PURE TRADE + 拒绝线性）。状态：完成且多重复核——这是可信度最高的一组**否定**结果。

### G. 决策校准（rope_decision_20260911，Pro 提议的任务判决校准）

probe_01（4 个唯一冲突案例：stock 与可微路径 margin 逐位一致，修过的 EVQ τ=1 两版均 0 分）→ calibration_01（部分修复：short fwe 1/3→1，两个 long 保持）→ gain_only_01（同校准 gain-only 对照：无修复）→ calibration_02（无新增修复）→ **fresh_72 验证**：decision_calibrated @16K **−9.4pp vs BM**（se 6.4）、vs b4wide −18.4pp（se 5.3，t≈−3.5）——**校准候选在独立样本上失败，长程腿显著差于两个参照**；@4096 与两者打平。状态：完成且验证（这是"任务校准不泛化"的第一个独立样本判决）。注意：decision_calibrated 的初始化是 b4wide 类表（sum_m 46.59），失败归因（校准 vs 初始化）按 RESEARCH_OWNER 纪律留为开放，但其相对 b4wide 的 −18pp 已排除"校准无害增益"。

### H. EVQ（当前问题外，只留约束）

旧 evq_shift 三 τ 全 0（身份更正后是 midpoint 位移干预）；evq_calibration_01 因 GPU 竞争失败（无结果，非方法失败）。EVQ 冻结替换方向对"YaRN→Pro"问题的唯一约束：全局 OOD 改动（非三段式）在冻结模型上崩溃——与后置族灾难同向但更极端。

### I. 跨模型（对当前问题的约束：收益是否模型特异）

Qwen1.5B 九臂连续 @2×（表间 0.008 nats 零效应，vs OLMo 0.83）；Qwen3B 36 行（无功率）；幸存者移植 run_qwen3_02（a1_b64T/wide_b4T 两臂皆败，128K 最好 +0.3pp 平手）⟹ OLMo 宽带收益模型特异；qwen4x_power 中断。**跨模型侧没有任何一个有功率的 YaRN↔Pro 对比**。

### J. 长度轴与剂量

s8 面板（32768 行四臂全 0.0/48，4K 侧 0.66–0.76）⟹ m≤1 盒子在 8× 无覆盖；dose 三长度（NLL 侧 a*≡1 在 BM 端点，价差 0.255→0.301→0.826 随长度放大）；P1 scale8x_wide（8× 振幅处方）**未跑**。walk 剂量（180 行五档：16K +7.8~+10.9pp 饱和倒 U、4K −2.7~−6.4pp）。

### K. 自然 QA（391 行）

六臂（BM/b3/wide_b1/wide_b4/walk_a0p5/turns_a1_b16）全 0.127–0.137，pooled +0.04pp（t=0.05）。限定：87% 行 BM=0（地板），结论只能读作"13% 有信号行上无大效应"。**含义（对当前问题）**：论文的 PPL 型收益与我方检索型收益测的是不同东西；自然 QA 上家族全平。

### L. LongBridge 有符号对（唯一符号控制设计）

槽 28–31（OLMo 对应 Qwen 历史 36–39），ν±6.1e-5（公共相位 ∓1 rad@16K）。
350 行选择面板：**slow−fast = +11.83pp（se 2.02，t=5.85，63胜27负）**，slower vs BM +5.50pp（t=3.23）——
但 **65% 的效应集中在 niah_single_3**（held-out 判定过的伪影任务）。
held-out：**slower vs BM 翻负 −3.31pp（t=−1.47）**；faster 腿死于实例中断（60/180），符号对主统计量不可算。
按预注册判据链在"held-out 同号"一步失败 ⟹ **"慢端 ν 微调"方向不成立，关闭**。
`verdicts/LONGBRIDGE_RESULT · LONGBRIDGE_HOLDOUT · SIGNED_CONTROL_RESULT`（数据 `olmo_lb{,_h}/`）。

---

## 4. 哪些结果可信、可以复用（含修正记录）

### 4.1 强可信（跨仪器 / 独立样本 / 预注册命中，可直接引用）

| 结论 | 证据 | 复核 |
|---|---|---|
| **MrRoPE-Pro 在 OLMo-1B@16K 是灾难**（vs BM 族） | 350 行 0.0709、72 行 2.78%、fresh_72 0.1354（−34.7pp se 4.8）、NLL 3.688 vs 2.862 | 三块独立面板 + 两仪器；本地 jsonl 重算逐位吻合 |
| **官方索引 YaRN 在 OLMo-1B@16K 同为灾难，与 Pro 无差** | 72 行 6.94%；fresh_72 0.1406，pro−yarn −0.5pp（se 1.4） | 两块独立面板；表身份三方核验 |
| **gain 1.0→1.1386 在 BM 表上 +38pp** | 350 行 t=15；审计 900 行重打分零错；fresh_72 gain 1.1191 +8.4pp 同向 | 多重复核 |
| **8×（32768）m≤1 全家 0.0** | 48 行×4 臂 + 复读式输出形态学验证 | 预注册判据（绝对地板条款） |
| **NLL 与任务准确率沿频率轴反向** | 走线五档 + 三张独立构造表，两长度，排序一致 | 跨仪器双显著 |
| 12 静态泛函 + 6 条建模推导路线全部无预测力/负结果 | 三层受控对 | 多重独立判决 |
| **宽带族 16K/4K 长度交易** | holdout180（+6.5/−5.4）、唯一提示修正（b4wide +8.1±4.2/−4.5）、fresh_72（b4wide +9.1±4.7/+2.4*） | 三块独立数据同向（*fresh_72 4K 腿为 +2.4，与前两块 −5 的差异见 §4.3-6） |

### 4.2 有条件可信（保留效应量与不确定性，不得写成稳定获胜）

- **b4wide @16K +8~+10pp**（三块独立数据 +7.9~+9.6pp，SE 3.8~4.7，各 t≈1.7–2.1）：方向稳健、幅度中等；**4K 腿代价在前两块 −4.5~−6.4pp，fresh_72 为 +2.4pp（se 3.0）**——4K 代价的稳健性弱于 16K 收益，引用时必须并排。整行答对口径下 16K 收益缩水（holdout180：分数 +9.6 → 整行 +6.7）。
- ~~LongBridge 符号对~~（+11.8pp@350）：**已在 held-out 翻负（−3.31pp），关闭**——移入 §5.1 否定清单；保留作为"符号对设计也会被伪影任务污染"的案例（65% 效应来自 niah_single_3）。
- **C42/C42V24 同 S 差 10.73pp**：双仪器同向验证的真实二阶效应，但无理论分辨、不可外推（Var(ε) 族内 R²=0.009）。
- **step42 选择面板 +7.3pp**：已被 held-out 反转（§5），只作选择偏置的教学案例引用。

### 4.3 已被审计修正的口径（旧值 → 修正值）

1. **180 行 held-out 实为 120 唯一提示**（60 组重复，重复组内逐臂分数相同）：独立样本显著性全部重算。修正后 b4wide vs BM：16K 唯一提示宏 +7.92pp（分层 SE 3.81）/ 4K −6.16（3.45）；简单唯一提示均值 +8.13（4.21）。**方向与量级保留，旧 t 值作废**。审计数据 `audit/pro_decision_20260911/check_results.json.unique_holdout`。
2. **Qwen NLL 读数器胜负符号写反**（BM−MrRoPE<0 时印 MrRoPE 赢）：修正分支后重算，BM−mrpro@4× = −0.0074（mrpro 略好）但 n=4 无功率；"OLMo 与 Qwen NLL 响应相反"的旧说法**不得引用**。
3. **Qwen native 参照带 gain=1.1386**：表-vs-native 的 −0.17 nats 是"同 gain 频率对照"，不是纯 native 差。
4. **旧 evq_shift ≠ canonical EVQ**（§1.6）：三个 0.0 按窄身份保留。
5. **Qwen 131K NLL 对齐错误**（far=131073 +1 越界）：该作业不可用于细微比较（已停，native/mrpro 数组保留）。
6. fresh_72 的 4K 腿与前两块 heldout 的 4K 符号差异（+2.4 vs −5.4）：两块面板任务构成不同（fresh_72 是 6 任务等权 macro、每任务 4 短行；holdout180 是 60 行加权），且都在 ±2SE 内——**不足以宣称"4K 代价消失"**，只能并排报告。

### 4.4 计划/执行/验证状态区分（防混写）

- **仅计划未执行**：P1（scale8x@8×）、P5（Uni 在 350/180）、P6（step hi19-24 扫描）、P7（Qwen 检索式仪器）、P8（OLMo-2-7B）、P9–P12（exact_yarn/reverse_pro/hi23/hi24/satpro——注意 **P9 的盲预测已被 fresh_72 部分超越**：fresh_72 的 yarn_index 0.1406 落在 P9 区间 [0.46,0.60] 之外，面积账对官方 YaRN 失败的方向已现，但 P9 判据写的是选择面板+holdout 双段，严格判分仍待跑或不跑需决策）、g2x2 的 mrpro/b3 两列、gsweep 1.20（LongBridge 已关闭，见 §3-L）。
- **执行完成未读出**：gsweep 1.05/1.10（350 行×2，服务器有 jsonl，无读数文档）。
- **执行+验证完成**：§4.1 全部、fresh_72 六臂、LongBridge 两腿（判决：不成立）、gsweep 1.05/1.10（YaRN 解析值=实测最优）、g2x2 native/BM 列终版。

---

## 5. 历史实验究竟排除了什么

### 5.1 被证据否定的具体假设（勿重走）

| 假设 | 否定证据 |
|---|---|
| S=Σm 是操作变量 | C42/C42V24 同 S 差 10.73pp（双仪器）；S=42 四重奏跨 0.111 nats |
| 任何静态低维表泛函预测分数（12 个，含 forcing 形状） | 三层受控对；forcing 坐标下赢者相关 0.899 而结果差 0.115 nats |
| Fisher 二次代价 | 符号错（F₂₄=7842 恒正，实测 m=1 处 ΔNLL −0.0150） |
| 单槽梯度可导出表 | SNR<1（单槽 0.005–0.015 vs 文档 SE 0.005） |
| 均值 NLL 代理可选表 | NLL 与任务反向（三表+走线，双仪器显著） |
| 频谱空洞是一阶杀手 | step42/step25 同 7.76× 空洞，+7.3 vs −30.5pp（P2 失败驱动修订：覆盖一阶/空洞二阶） |
| "必须平滑 ramp" | step42 选择面板赢 BM（但见 out-of-sample 反转——此条降级为"平滑不是先验必要"） |
| EVQ/midpoint 位移零训练替换 | 三 τ 全 0 |
| 覆盖率越大越好 | interp 不可用 |
| OLMo 宽带赢家跨模型迁移 | run_qwen3_02 两臂皆败（128K −16.6/−10.6pp） |
| 任务判决校准（65 参数 greedy 前缀归纳）产出可部署表 | fresh_72：decision_calibrated @16K −9.4pp vs BM、−18.4pp vs b4wide |
| "两模型族方向相反"（Qwen MrRoPE>BM） | 36 行 SE≈13pp 无判别力；读数器符号错误已修 |
| 慢端 ν 微调（LongBridge 符号对） | 面板 +11.8pp 但 65% 集中于伪影任务 niah_single_3；held-out 翻负 −3.31pp |

### 5.2 仍有价值的条件收益（不可丢弃，不可夸大）

- **宽带 BM 族 @16K 检索 +8~10pp**（三块独立数据同向，见 §4.2）——以 4K 窗内代价换 16K 外推的交换，部署应长度条件化。
- **gain 轴**：+38pp 主效应 + 乘法交互 + fresh_72 gain 1.1191 +8.4pp；YaRN 解析 mscale 方向被支持，最优值未定（gsweep 未读完）。
- C42V24 的二阶效应（机制不明但真实）。

### 5.3 证据不足的判断（不要当结论引用）

- 跨模型上 YaRN/Pro/BM 的任何排序（全部仪器分辨率不足或中断）。
- "Qwen 上 Pro 优于 BM"（−7.3pp 与 +13.3pp 均在 SE 内）。
- 窗内腿 Pro 差于官方 YaRN −24.5pp（fresh_72 单次，名义 t≈3 但 4K 腿每任务 n=4，需独立复现）。
- 自然 QA 上"全家族无差异"（87% 地板行）。
- n_int/覆盖理论的部分样本外命中（8× 全零半预期；C42 对无分辨；n_int 与 Σm 共线）。

### 5.4 因实现/评测问题无法判定的

- qwen4x_power（2/87 行）、LongBridge 符号对 holdout 腿（faster 60/180，主统计量永久不可算，判决已按次要对比关闭）、g2x2 mrpro/b3 列（18/350）、gsweep 1.20（83/350）、s42h 的 length 分层读数（脚本未跑通；审计 unique 分析已给 step42 长端 −3.7pp 的修正值）、Qwen 131K BM 臂（对齐错误）。
- 旧 jsonl 无 token ID/prompt hash：任何"精确重放"类验证（Pro 提议的核心）在旧数据上不可能，须新生成。

### 5.5 为什么中频改造的选择集提升没有稳定泛化（证据链）

已建立的机制证据：
1. **D_pattern 恒定**：任何表都翻动 15–20% 的行；净方向由面板决定（选择面板 65–70% 建设性 / held-out ≈0%）。
2. **选择面板增益集中于伪影任务**（niah_single_3 贡献 +32~+46pp；该任务不在任何 held-out 里）。
3. **step42、平台成员、condEVQ 三次独立 out-of-sample 反转/归零**（+7.3→−3.4；+12~14→+2.6；+2.1→净行−1）。
4. 长程真增益的构成：多参考任务的逐项召回（+6.5pp 分数值 → +1.1pp 整行），不是解出新题。
5. SYNTHESIS 的定量解释：频率轴可达效应 ±5pp 低于评估翻动底噪（RATIO 2.65–37 vs gain 的 1.03）。

**已有证据能回答的**：选择面板提升不泛化是**可重复的测量学事实**（三次判决 + D_pattern 量化）。**尚不能回答的**：为什么选择面板会给这些表方向（single_3 类任务被什么机制改善）——这是机制问题（E1/E2 未做），不是测量问题。**纪律重申**：不得再把"平滑度/覆盖/NLL/余弦和/margin 变好"推成任务变好（NLL 已证反向；其余无任务级验证）；也不得把单次失败外推为方向不可能（宽带 16K 收益三块数据存活就是反例）。

---

## 6. 核心问题的证据结论

**Q：在可比条件下，哪里确实观察到了 YaRN→MrRoPE-Pro 的收益？哪里没有？哪些历史结果其实在比较 BM 的改造？**

1. **我们从未在任何一个有功率的对比中观察到 Pro 优于官方 YaRN（或反之）。** 两者的全部三次同台（OLMo 72 行、fresh_72、Qwen 36 行低功率）都给出"无差"或"同为灾难"。OLMo@16K：YaRN 0.141 ≈ Pro 0.135（Δ −0.5pp，se 1.4），同差 BM 约 −34pp。
2. **论文报告的 Pro>YaRN 收益存在于我们未覆盖的区格**：LLaMA2/3-8B、Qwen2.5-3B（强模型），PPL 主导 + 窗内行（Pro 最大收益格 = LLaMA2@4K PPL 5.72 vs 6.02），RULER 差距温和（3–7pp，且 Qwen 侧 ~50% 水平）。我们未测 LLaMA；OLMo-1B 上的等价物（PPL 型、窗内行）**从未跑过**——这是两侧矩阵缺失的另一半。
3. **OLMo 上与论文叙述正交的强事实**：三段式后置族（官方 YaRN / Pro / Uni）在 16K 检索上全灾难（0.07–0.14），中置宽带族（BM 系）0.42–0.57。这个 30–48pp 的分界不是"YaRN→Pro"的效应，而是**装填位置（后置 vs 中置）+ 带宽**的效应；它由 BM（自创表）与文献表构成，因此**回答的是"BM 为什么好"，不是"Pro 相对 YaRN 改了什么"**。
4. **历史结果中属于"BM 改造内部比较"、不能回答本问题的**：350 行面板的全部 +12~14pp"赢家"及其后的一切（b 轴扫描、宽带变体、C42 对、heldout 72/180、walk、natural、s8 四臂、dose、s42h 六臂、holdout 分层交易）——对照全是 BM 或 BM 变体。这些数据对"YaRN→Pro"仅提供一条间接约束：**后置族的失败模式（覆盖缺口）在中置宽带下消失**，即 Pro 的 ε 剖面在覆盖饥渴模型上是要害，与 MRROPE_RECONCILIATION 的"代价侧"读法一致。
5. **窗内腿的新线索（待复现）**：fresh_72 @4096 上 Pro 差于官方 YaRN −24.5pp（se 8.1）——与论文"Pro 窗内最好"方向相反，但语义不同（我方 4096 = 训练窗本身、S=4；论文 LLaMA2@4K 在 S=16 配置内）。这是"论文收益格在我方矩阵的最近邻"的第一个数据点，价值在复现后。
6. **Gain 的位置**：论文与我方都继承 YaRN mscale。我方测得它是最大杠杆且与表乘法交互——**任何"YaRN→Pro 为什么有效"的解释若不含幅度项，在我方数据上都不完整**（同表 gain 变化 38pp >> 表间差异）。

---

## 7. 关键未决问题（只列证据支持的；注明现有材料能否回答）

| # | 问题 | 现有材料能否回答 | 若不能，缺的最小证据 |
|---|---|---|---|
| 1 | **Pro 的论文收益到底是"强模型"还是"PPL/窗内任务矩阵"带来的？**（OLMo-1B 上 PPL 型窗内行从未测） | 不能 | OLMo-1B 上 proofpile 型 PPL @4K/16K，Pro vs 官方 YaRN vs Uni vs BM 四臂（无生成、纯 teacher-forced，约等于一次 NLL 会话成本）；或直接跑 P8（OLMo-2-7B 同族同窗四臂 48 行），把"规模"从"家族/窗"剥出 |
| 2 | **fresh_72 窗内腿 Pro−YaRN −24.5pp 是真的吗？** | 单次 72 行，名义显著但每任务 n=4 | 独立 72–150 行 4096 腿复现（fresh 提示再生成一批） |
| 4 | **gain 最优值与 gain×表交互全貌？** | 部分（BM/native/a1_b64 三点 + fresh_72 一点） | gsweep 1.05/1.10 已有数据先读出；补 1.20（83/350 续跑）与 g2x2 孤儿 4 臂（18/350 续跑）——都是续跑不是新实验 |
| 5 | **8× 处方（scale8x_wide）能否非零？**（覆盖理论主判决 P1） | 不能（未跑） | P1 一臂 72 行 s8 面板（表/判据/命令全部就绪） |
| 6 | **为什么后置族在覆盖饥渴模型上灾难、中置宽带存活？**（机制层） | 不能——全部现有证据是输入端统计×输出端分数 | E2 注意力遥测（1 行×4 表，逐槽 logit 归因解析可算：step25 vs step42 只差槽 22–24；Pro vs BM 逐槽 Δm 已知）。设计在 GPT6PRO_MECHANISM_BRIEF §5 |
| 7 | 跨模型符号（qwen4x_power） | 不能（2/87 行） | 原计划续跑（29 篇×3 臂@131K，~3–5 GPU 时） |

**优先级判断依据**（非新计划，只按信息量/成本排）：#6 是"为什么"的唯一通路且零生成成本；#1 是"论文收益能否在我方复现"的最小判别；#3/#4/#5 全是续跑已预注册实验。

---

## 附录 A：完整实验索引（干预 → 文件 → 数据）

**本地仓库**（`/Users/[REDACTED_AUTHOR].yanghejazfs.com.au/paper_project/hybrid-rope`）：
- 判决文档：`ds_workspace/recon_20260910/*.md`（51 份，见其 index.md §目录结构；本报告已覆盖全部主判决）
- 表代数（参考实现）：`experiments/curvature_20260910/tables.py`（m 坐标、全部构造器、verify CLI）
- 新算子（官方索引 vs 论文转数 YaRN、literal EVQ 两版）：`experiments/rope_decision_20260911/tables.py`
- BM 闭式：`scripts/lib/rope/boundary_matched.py`；官方 YaRN 等式：`scripts/lib/rope/official_yarn.py`；EVQ φ：`scripts/lib/rope/schedules.py:162`
- OLMo runner（350/72/180/391 面板）：`experiments/zerotrain_20260910/olmo_beta.py`（OLMO 常量 L65-66）
- 审计快照与复算：`ds_workspace/recon_20260910/audit/pro_decision_20260911/{REPORT.md,check_results.json,formula_results.json}` + 各 before 快照
- 验证读数器（fresh_72 口径）：`experiments/rope_decision_20260911/read_validation.py`
- 理论复算：`ds_workspace/recon_20260910/code/coverage_theory_20260911.py`
- 早期原始数据：`results/olmo_fast_screen_20260908/`（含 run_controls_01 官方 YaRN）、`results/bm_transfer_20260908/`、`results/nongeometric_screen_20260909/`
- 同步 jsonl：`ds_workspace/recon_20260910/work/jsonl/{archive,olmo,olmo_b3,olmo_wide,olmo_turns,olmo_c42,holdout,…}`
- 论文原文：`~/Downloads/RoPE_Papers/Markdown/5551_MrRoPE_Mixed_radix_Rotary.md`；YaRN 官方：jquesnelle/yarn@995db5b（`scaled_rope/LlamaYaRNScaledRotaryEmbedding.py`）
- 早期文档簇：`docs/research/ROPE_MRPRO_BM_*_20260908.*`、`analysis/unify_20260910/tables/ground_truth_tables.json`（38 方法部署表）

**服务器**（`ssh -p 27741 [REDACTED_EMAIL]`，只读盘点 2026-09-11）：
- `/root/autodl-tmp/phase1_20260910/`：olmo_pro（condEVQ/step42 350）、s8_out（72×4）、walk_out（180×5）、olmo_gain（350×2）、olmo_gain2x2（a1_b64 350×2 + mrpro_g1p0 18）、olmo_ngain{,_h}（native 两 gain 350/180×2）、olmo_gsweep（350/350/83）、natural_out（391×5+summary）、holdout（72×4）、holdout180（180×4）、s42_out（六臂 180）、dose_L{4097,8193,16385}、cont* 全家（NLL 仪器）、qlev_L*、qwen4x_power（2 行）、olmo_lb（350×2）、olmo_lb_h（60/180）
- `/root/autodl-tmp/rope_decision_20260911/`：probe_01、calibration_01/02、gain_only_01、fresh_72/（identity_check.json：72 新、零重叠 vs 2227 旧行/1337 唯一）、validation_{reference,methods}、validation_readout.json、validation_plan.json（SHA 6e91bf13…）
- 运行时：`/root/miniconda3/bin/python`；磁盘 48G/68G 用量

## 附录 B：关键重算命令

```bash
# 1. 全部表公式/sum_m 复算（本报告 §1.2 数值的来源）
python3 -c "from experiments.rope_decision_20260911.tables import formula_audit; import json; print(json.dumps(formula_audit(), indent=1))"

# 2. 表代数对部署表验证（Native/MrPro/YaRN_linear_official 三表 bit 级）
python3 experiments/curvature_20260910/tables.py verify analysis/unify_20260910/tables/ground_truth_tables.json

# 3. 本地 jsonl 重算（任何臂的分数独立复核）
python3 - <<'EOF'
import json
for line in open('ds_workspace/recon_20260910/work/jsonl/archive/MrPro.jsonl'):
    ...  # 累加 correct / 按 length_cap×task 分组
EOF

# 4. 审计复算（holdout 900 行重打分、唯一提示修正、gain 复核、旋子 dtype）
python3 ds_workspace/recon_20260910/audit/pro_decision_20260911/check_remote.py   # 经 ssh stdin，CPU only

# 5. fresh_72 读数（equal-task macro + 配对 Welch 区间）
ssh -p 27741 [REDACTED_EMAIL] "/root/miniconda3/bin/python -m experiments.rope_decision_20260911.read_validation --root /root/autodl-tmp/rope_decision_20260911"

# 6. LongBridge 350 腿配对读数（本报告 §3-L 数值的来源，可直接重跑）
#    load olmo_lb/{nu_m6p104em05,nu_p6p104em05}.jsonl 按 row_id 配对，slow−fast

# 7. 覆盖理论全部量（n_int、窗口律、8× 预测）
python3 ds_workspace/recon_20260910/code/coverage_theory_20260911.py
```

## 附录 C：无法访问 / 中断的证据（具体路径与影响）

| 项 | 状态 | 影响 |
|---|---|---|
| 服务器 GPU | 重启后 `nvidia-smi` No devices found，无实验进程 | 当前无法跑任何 GPU 作业；数据盘完好。恢复 GPU 需平台侧操作（非本任务范围） |
| qwen4x_power | 2/87 行 | 跨模型符号判决（预注册 QWEN4X_POWER_PREREG）无法出数；重跑需 3–5 GPU 时 |
| LongBridge holdout faster 腿 | 60/180 | 符号对主统计量永久不可算；判决已按 slower-vs-BM 次要对比关闭（翻负） |
| g2x2 mrpro/b3 四臂 | 18/350 | gain×表交互只有三点+1；Pro 的 gain 响应（对当前问题重要）缺 |
| gsweep gain=1.20 | 83/350 | mscale 最优性检验不完整；1.05/1.10 已有数据未读 |
| Qwen 131K BM 臂 | 对齐错误已停 | 该长度 BM−mrpro 只有 4 篇 qlev 数据 |
| MrRoPE 作者代码 | 从未获得 | 所有 MrRoPE 臂是本地复现（公式已对论文逐条核对，无实现级对照） |
| s42h length 分层 | 读数脚本未跑通 | step42 长端结论用审计 unique-prompt 值（−3.7pp）替代，pooled 值仍为文档值 |
| natural turns_a1_b16 jsonl | 目录中只见 summary（列表截断，未确认） | 若缺，六臂自然 QA 中该臂以 summary 为准 |

## 附录 D：本报告的核查方法与局限

- 复算：全部 sum_m/ε 剖面（两模型 × 6 算子）独立重算，与 `formula_results.json` 逐位一致；本地 8 个臂 jsonl 重算均值与文档一致；fresh_72 与 LongBridge 读数直接从服务器 jsonl 计算。
- 论文：从本地 markdown 原文核对 Eq.10/13/14/15/16/19-26、A.2.1 证明链、Table 1/2 数字、4.1 training-free 设定；YaRN 官方代码重新拉取核对 ramp/mscale。
- 未做：28K/6K cosine-bound 复算（零 GPU 可做，未做，标注于 §2.1）；服务器 jsonl 的全量重打分（抽查 8 臂 + 审计已做 900+700 行）；`_archive_20260911/` 23 份被覆盖早期文档未逐份重读（其结论已被后续判决取代，入口在 index.md）。
- 服务器盘点为只读（ls/wc/head/python 读文件）；未杀进程、未改文件（重启发生在盘点前，非本任务所为）。
