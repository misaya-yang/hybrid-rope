# A7 — `source_inputs/` 五份材料的挖掘 digest

范围：`docs/research/rope_allocation_20260910/source_inputs/` 下五份文件
（`pro6_allocation_analysis.md` 759 行、`mrrope_yarn_user_analysis.md` 489 行、
`Nongeometric_RoPE_Questions_for_GPT6Pro_20260910.md` 227 行、
`RoPE_Allocation_Core_Problem_for_Pro_20260910.md` 94 行、
`RoPE_Allocation_Theory_Questions_for_Pro_20260910.md` 247 行）。
对照权威文档：`analysis/unify_20260910/{INTEGRATION_20260910.md, NEXT_DERIVATION_KKT_PROBLEM.md, STARTING_POINT_YARN_VS_MRPRO.md}`。
本文只做提取与出处标注，不做推导。证据分级：[已验证] / [部分证据] / [假设] / [叙事-未验证]。

> **两条前置更正（重要，见 §2.3 与 §8-C1）**
> 1. 任务描述点名的四条"6Pro 修正条款"（**端点不变量口径 / gain 不正交 / Σm 恒等式 / r>1 后圆周已覆盖**）**并不逐字出现在 `pro6_allocation_analysis.md` 里**。它们逐字出现在
>    `/Users/yang/projects/hybrid-rope/docs/research/ROPE_GLM_6PRO_REVIEW_AND_VALIDATION_20260910.md` §"GLM：保留问题组织，撤下过强因果表述" 第 1–6 条（文件不在本次五份之列，由 grep 定位）。`NEXT_DERIVATION_KKT_PROBLEM.md:41/43/51/76/115` 以"（6Pro 修正）/（6Pro 澄清）/（6Pro 第 6 点）"引用它们。两份材料**实质内容重合但归属存在歧义**，下文两条并列给出，不合并。
> 2. `pro6_allocation_analysis.md` 自己**没有**任何"6Pro 修正条款"的自我编号；它是一份自包含的统一建模文档（带 8 条 arXiv 链接），面向"附件"。这是本次挖掘最需要提防的归因滑移。

---

## 0. 覆盖度

| 文件 | 行数 | 读法 | 状态标记 |
|---|---|---|---|
| `pro6_allocation_analysis.md` | 759 | 全文（含 §一–§八 + 8 条 arXiv 尾注） | 分析层文档，**无新增模型成绩**（:119 "这是上述坐标变换的数值结果，不是能力证明"；:131 "不是新增模型测评"） |
| `mrrope_yarn_user_analysis.md` | 489 | 全文（含 3 条 GitHub/arXiv 尾注） | 分析层 + CPU 重算，**自述"其中没有新增模型成绩"（:485）** |
| `Nongeometric_..._for_GPT6Pro_20260910.md` | 227 | 全文 | 对外问题包（证据快照 2026-09-10 06:08 UTC），含已完成实验数字 |
| `RoPE_Allocation_Core_Problem_for_Pro_20260910.md` | 94 | 全文 | 对外问题包（核心问题短版） |
| `RoPE_Allocation_Theory_Questions_for_Pro_20260910.md` | 247 | 全文 | 对外问题包（长版，证据更新至 2026-09-10 08:42 UTC） |

**未读**：`source_inputs/` 之外的 codex 线原始报告（`.agents/rope_unification_20260910/`，gitignored）、`digests_codex/` 七份、`analysis/kkt_20260910/mine/extracts/` 的 codex rollout 抽取文本。仅按需 grep 了 `digests/` 与 `docs/research/` 作交叉核对。

**分工关系**：五份材料中，`mrrope_yarn_user_analysis.md` 是**用户/我方**的算子级核查；`pro6_allocation_analysis.md` 是**外部 6Pro 的统一建模回复**；后三份是**我方对外的提问包**（= 问题原始表述）。三者角色不同，不得互相引用为独立证据。

---

## 1. 五份材料的定位与口径

| 文件 | 角色 | 关键口径自述 |
|---|---|---|
| `mrrope_yarn_user_analysis.md` | 用户自己写的 YaRN vs MrRoPE 算子级分析 | :77 "下面的定量比较，明确针对标准、可复现的代码版本"；:483 "我还没有从这些事实中推出一条已经足以判断会稳定超过 MrRoPE 的新分配公式" |
| `pro6_allocation_analysis.md` | 外部 6Pro 的统一建模 | :3 统一陈述；:68 "这三个量不应该统称为同一个预算"；:483 "这个分解用于组织假设和实验，不能直接当成整网收益定理" |
| `RoPE_Allocation_Core_Problem_...md` | 对外问题（短版） | :9 "服务器已经切到无 GPU 模式"；:94 "不要把回答停在'还需要更多实验'" |
| `RoPE_Allocation_Theory_Questions_...md` | 对外问题（长版） | :9 "现有 waterbed、smoothness、局部 gap 或 signed margin 分析均可重构，不能把其中任何一个当作最终答案" |
| `Nongeometric_..._for_GPT6Pro_...md` | 对外问题（10 问 + 证据包） | :9 "几何指标、attention mass、局部 NMSE 和 NLL 均不能替代下游任务" |

**共同部署口径（五份一致，可直接用作 F 的场景声明）** [已验证-自述]
Qwen2.5-3B-Instruct（revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`，`Nongeometric:15`）；36 层 / 16 Q heads / 2 KV heads / head_dim 128 ⇒ **64 个 rotary slots**；`W=32768`，`b=10^6`，`S=4`，`L=131072`；slot **从 0 开始**（`Nongeometric:15`，`Theory:17`）；权重冻结，从 prefill 起用同一频率规则（`Theory:17`）；**BF16 权重/旋转输出、FP32 相位计算、SDPA、greedy、repetition penalty=1**（`Nongeometric:24`，`Theory:17`）；任务原回答预算与 EOS 不改（`Nongeometric:24`）。

---

## 2. 6Pro 修正条款（逐条）

### 2.1 `pro6_allocation_analysis.md` 中实际写出的条款（本条为本次任务的主要交付）

**P1. 三个"预算"不可混称同一预算** [已验证-口径]
原文（:68）："附件已经正确地区分了 pair 数、总对数跨度和累计压缩量；这三个量不应该统称为同一个预算。"
出处：`pro6_allocation_analysis.md:68`（§一.1 末句）。与 `Theory_Questions:31`、`NEXT_DERIVATION:107`（codex §2 记账纪律）三处独立同述。

**P2. 固定端点 ≠ 固定累计压缩预算 B** [已验证-代数/口径]
原文（:117）："**固定端点不意味着必须固定 \(B\)；固定 \(B\) 是很好的因果实验控制，却不应成为所有新方法的硬约束。**"
配套定义（:110-116）：\(B=\sum_{q=1}^{N-1}m_q=\sum_{i=1}^{N}(N-i)\epsilon_i\)。
出处：`pro6_allocation_analysis.md:110-117`（§一.2）。authoritative 对应：`Theory_Questions:112`（"固定B是额外选择"）、`NEXT_DERIVATION:107`（"Σm（质心）不是守恒量，是自由决策变量"）、`INTEGRATION_20260910.md:30`（红线 R4）。

**P3. Σε=1 与总跨度增量的箱式恒等式** [已验证-代数]
原文（:96-104）：高端 \(m=0\)、低端 \(m=1\) 的边界条件下
\[
\sum_i\epsilon_i=1,\qquad \sum_i(a_i-a_i^{(0)})=\log S,\qquad \epsilon_i=m_i-m_{i-1}.
\]
出处：`pro6_allocation_analysis.md:96-104`。**这是 Σε=1 口径的原始写法**；注意它与"17 个 log-gap 之和 = ln S"是**不同**命题——见 §2.2-G1 与 §8-C2。

**P4. 统一陈述（配置层 / 使用层二分）** [叙事-未验证，但结构被 authoritative 部分采纳]
原文（:3）："它们都在配置有限的对数频谱间隔。EVQ 在固定频谱跨度内重新配置间隔；MrRoPE 为扩展窗口增加频谱跨度，再决定把新增间隔放在哪里。真正决定效果的，是这份配置能否支持目标尺度的计算，以及已有权重能否继续使用重新配置后的频率。"
出处：`pro6_allocation_analysis.md:3`、:5-11（配置层 / 使用层）。authoritative 对应：`INTEGRATION:58` 的"分配是带约束的有限输运问题"与 `:367-373`（几何 ≠ 使用）。

**P5. 频段不可写成三个独立效用函数之和** [部分证据]
原文（:381-402）："类似 \(U_{\mathrm{high}}+U_{\mathrm{mid}}+U_{\mathrm{low}}\) 的写法，除非额外证明独立性，否则过于粗糙。" 理由：完整 logit 误差平方**包含跨 pair 项**，不能默认逐槽损失可加。
出处：`pro6_allocation_analysis.md:381-402`。与 `Theory_Questions:124`（"完整logit误差平方包含 \(2\sum_{j<k}\mathbb E[\Delta z_j\Delta z_k]\) 交叉项"）**独立同述**。

**P6. gain 是独立记账项，不能归功于频率预算** [已验证-面板]
原文（:334-346）："当前默认 \(g=1+0.1\ln S\) 同时作用于 Q/K，在固定 raw Q/K 下使 logits 乘以 \(g^2\)。…你们的同表实验中，仅改变 gain 系数，MrPro 的 32K 分数就从 \(87.22\%\) 变成 \(98.33\%\)。因此，'窗口内保护很好'必须分清频率配置的贡献和 gain 的贡献。"
出处：`pro6_allocation_analysis.md:336-344`。对应面板行：`Theory_Questions:134/137`（MrPro/.1 = 87.2222%，MrPro/.074 = 98.3333%）。

**P7. EVQ 的严格最优性只属于该特定目标** [已验证-推导 + 边界自述]
原文（:256-260）："EVQ 的严格最优性属于这个特定目标。要推出更好的规则，必须回答：原目标中的间隔代价，怎样对应模型真正需要的尺度？它是否遗漏了冻结权重的使用成本？"
以及（:250-254）："**EVQ 不是简单地'增加低频通道'。它实际上让低频侧的通道在对数尺度上分得更开，同时让高频侧采样更密。**'慢时钟更分散'和'慢时钟更多'不是一回事。"
出处：`pro6_allocation_analysis.md:250-260`。

**P8. EVQ 的地位限定（承接 P7）** [部分证据]
原文（:487-495）："**EVQ 是'间隔预算目标'的一个严格特例，但现有材料还没有证明它是上述真实关系拟合目标的特例。**"
出处：`pro6_allocation_analysis.md:485-495`。authoritative 对应：`NEXT_DERIVATION:101`（EVQ = 可重学极限下的 E-L 解，[部分证据]）、`GLM_6PRO_REVIEW:53`（"它没有证明 EVQ 是真实整网目标的特例；6Pro 对这一点的限定正确"）。

**P9. 校准目标与位置映射假设（6Pro 提出的新规则生成法）** [假设]
原文（:499-584）：对同一 token 序列构造 \(p'=p\) 与块间拉伸 \(p=bM+r\mapsto p'=SbM+r\)；用真实 \(C_j,D_j\) 重算 \(z_{\nu,k}\)，最小化
\(\mathcal L_{\mathrm{cal}}(\nu)=\mathbb E_{\text{rows,maps}}\operatorname{KL}(a^{\mathrm{native}}\|\operatorname{softmax}(z_\nu))\)；
"对核心因果实验，还可以额外固定 \(\sum_jx_j=\sum_jx_j^{\mathrm{Mr}}\)"（:562-566）。
边界自述（:578）："它仍然有一个尚未解决的风险：保存的 Q/K 不会随新表改变，而真实完整 prefill 会改变。附件中的 E7 和缓存诊断已经证明，这个缺口不能忽略。"
出处：`pro6_allocation_analysis.md:507-584`。**这正是 `Theory_Questions:122` 记载的"此前 6 Pro 分析建议固定 MrPro 预算"的来源候选**，但 pro6 原文说的是**固定 Σx 的校准拟合**，**不是**"最小化 roughness"——归属需谨慎（见 §2.3）。

**P10. 三项核心实验设计** [设计假设]
- 实验一：同预算正向重分配 + 反向控制 \(x^+=x^{\mathrm{Mr}}+v\)、\(x^-=x^{\mathrm{Mr}}-v\)，\(v_0=v_{K-1}=0\)、\(\sum_jv_j=0\)（:632-676）。**这是"方向预测力"检验的原始设计**（作者：:664 "理论不仅要推荐一张表，还必须在看测试答案前给出方向"）。
- 实验二：三条件分离——原生短输入 / 块间位置拉伸（token 数不增加）/ 真实稠密长输入（:679-703）。
- 实验三：训练表 × 测试表交叉（几何训练/EVQ 训练 × 几何测试/EVQ 测试）分离"配置质量"与"冻结兼容性"（:705-732）。
出处：`pro6_allocation_analysis.md:632-732`。authoritative 对应：`STARTING_POINT:89-99`（F9 四格）、`NEXT_DERIVATION:125`（K6）。**注意：pro6 的实验一不是四格反事实，两者是不同设计，不要混并。**

### 2.2 四条点名条款的逐字出处（在 `ROPE_GLM_6PRO_REVIEW_AND_VALIDATION_20260910.md`，非本次五份）

> 文件：`/Users/yang/projects/hybrid-rope/docs/research/ROPE_GLM_6PRO_REVIEW_AND_VALIDATION_20260910.md`（2026-09-10）。`NEXT_DERIVATION:41/43/51/76/115` 以"6Pro 修正/澄清/第 N 点"引用它。另见 `digests/digest_paper-state.md:95` 的逐条摘要（与原文一致）。

**G1. 端点/守恒口径（"Σε=1 ⇒ 17 个 log-gap 总和为 ln S"是错的）** [已验证-代数]
原文（:7）："设 ε_i=m_i−m_{i−1}，固定 m_23=0、m_40=1，确有 Σ ε_i=1，因此新增对数跨度是 ln 4。**17 个实际 log-gap 的总和则约为 5.056039，而非 ln 4；其原生部分约为 3.669745。17 个增量在一条总量约束下仍有 16 个自由度。**把桥宽设为唯一参数是另加的设计限制，不能从守恒推出。"
authoritative 转写：`NEXT_DERIVATION:43`（"17 个过渡 gap 的总跨度锁定为 ln S（原生 3.6697 + 额外 1.3863 = 5.0560 nats）…两种口径不可混用"）、`:114`（否决约束第 4 条）。**内生自洽复核**：\(17\times\ln(10^6)/64=17\times0.2158735=3.67095\approx3.6697\)（末尾差异来自零基/一基约定），\(+1.3863=5.0560\)。✔ 与 `pro6_allocation_analysis.md:248` 的"等距表的间隔约为 0.21587"同一常数。

**G2. r>1 后圆周已覆盖（D_j 不是"未见相位边界"）** [已验证-代数]
原文（:9）："\(D_j=W S^{m_j}\) 精确表示：部署距离 d 在缩放后的未模坐标中对应 \(d/S^{m_j}\le W\)。可是 **Native 第 36–39 个频率在 W 内已经转过约 2.199、1.772、1.428、1.151 圈，各自圆周相位都已覆盖。超过 \(D_j\) 不等于进入一个从未见过的单频弧。**联合频率关系仍可能越出原生轨迹，但那需要联合模型；不能用四个独立 \(D_j\) 替代证明。"
authoritative 转写：`NEXT_DERIVATION:51`（"'未见弧'只对 r<1 成立…须按联合谱覆盖/风险带定义"）。**交叉来源**：`Theory_Questions:76-88` 的周期表给出同一组圈数（slot 36→2.199、37→1.772、38→1.428、39→1.151）。✔

**G3. Σm 恒等式（"所有赢家同方向移动预算"被推翻）** [已验证-表]
原文（:13-22）：用 ε_i 的真实槽位质心比较

| 表 | \(\sum_j m_j\) | 新增 log-gap 质心 |
|---|---:|---:|
| MrPro | 29.333333 | 34.666667 |
| E1 s28 less | 29.300653 | 34.699347 |
| LBS | 29.560179 | 34.439821 |
| P2 | 34.178915 | 29.821085 |

"E1 向后移，LBS 和 P2 向前移。不同方向都可能形成有条件收益，不能据此建立单一方向梯度。"
authoritative 转写：`NEXT_DERIVATION:76`、`INTEGRATION:30`（红线 R4）。**本次独立复算（用权威文档自身的公式）**：MrPro \(m_j=q(q+1)/306\)，\(q=j-23\)，\(j\ge40\) 取 1 ⇒ \(\sum_{j=0}^{63}m_j=\sum_{q=0}^{16}q(q+1)/306+24=16\cdot17\cdot18/6/306+24=5.3333+24=\mathbf{29.3333}\) ✔ 与表值逐位吻合。**同一复算给出质心的两条口径**：\(\sum_i i\epsilon_i=2\sum i^2/306=3570/306=11.6667\)，故槽位质心 \(=23+11.6667=\mathbf{34.6667}\) ✔。
→ **副产品结论（本次挖掘新增，供核对）**：`Theory_Questions:105-110` 与 `NEXT_DERIVATION:85` 的 **B = 16/3 = 5.3333** 是"过渡槽 23–39 的 m 之和"；GLM 表的 **Σ_j m_j = 29.3333** 是"全 64 槽之和（含尾部 24 槽 ×1）"。两者恒差 24，**不是同一量**，但两份材料都简称"预算/质心"（见 §8-C3）。

**G4. gain 与频率效应不正交** [已验证-推理]
原文（:26）："固定状态下 gain 使 logits 乘 \(g^2\)；\(\operatorname{softmax}(g^2 z_\nu)\) 同时依赖两者，通常存在非零交互项。四格对照不能证明幅度与相位机制在实际效果上互不影响。"
authoritative 转写：`NEXT_DERIVATION:115`（"gain×相位不正交（6Pro 第 6 点），F 不含 gain 自由度"）。**注意与 `pro6_allocation_analysis.md:336-342` 的关系**：pro6 说"固定 raw Q/K 下 logits 乘 \(g^2\)、不改变相位"；G4 说"整网里两者不正交"——**两条不矛盾**（前者是固定态恒等式，后者是否定"四格能分离机制"）。

**G5.（同节其余两条，供完整性）**
- LBS 也没"完成到 m=1"：槽 36–39 实际 m = .625169/.729476/.846534/.979914，D ≈ 77954/90082/105953/127473（:24）。
- 证据距离数据质量：FWE 距离定义不成立（答案词全文出现 10785/4793 次）、VT 只取 max 丢链、MQ 一行含 77 与 120934 双距离 ⇒"36 行并非同一种距离—准确率样本，不能据此将 75–112K 宣称为已验证的危险带"（:35-38）。

**G6.（`ROPE_GLM_6PRO_REVIEW` §"6Pro：可直接保留的数学"）** [已验证-CPU]
- 频谱三分解（高频端位置 \(x_0\) / 总跨度 A / 归一化间隔 \(a_i/A\)）"清楚地区分平移、扩大支持和内部重分配；Σm 只是一项可选额外控制，不等于总跨度"（:42）。
- EVQ 变量替换**精确**：以 α=1、β=4 的中点积分复核，两个坐标中的目标均约 **1.0373147207**，差约 **1.6e−11**；**KKT 最大残差约 1.1e−15**；64 个端点分位点、固定 Native 跨度的首末间隔 **.104825834/.391244468**，与 6Pro 数字一致（:44-51）。
- ridge 配方分解在 G+ζI 正定时等式成立（:53）。

### 2.3 归属歧义（必须记录）
- `NEXT_DERIVATION` 用"6Pro"指称上述 G1–G6；但 G1–G5 所在的文件标题是 "GLM / 6Pro 分析复核"，其第 1–6 条挂在 **"GLM：保留问题组织，撤下过强因果表述"** 小节下——即这些是**对 GLM 分析的撤稿条款**，其中引用了 6Pro 的数字/口径。**把 G1–G5 直接叫"6Pro 说的话"是转写层的归并，不是原文的自述。**（同型问题在 `A3_sol01_05.md` §7-C1 已出现过一次。）
- `Theory_Questions:122` 说"**用户此前提供的 6 Pro 分析建议固定 MrPro 预算最小化 roughness**，约束为外部增量端点零、ε_i≥0、Σε_i=1、B=16/3"。`pro6_allocation_analysis.md` 全文**没有出现"roughness / 粗糙度 / 平滑"作为推荐目标**；相反它在 :572 明确批评"不再只优化平滑度"。**该归因未在本次五份材料中找到载体，标记为待核。**

---

## 3. F1–F9 的原始出处：`mrrope_yarn_user_analysis.md` 的算子级事实与数值

> 本节 = `STARTING_POINT_YARN_VS_MRPRO.md` F1–F9 的**源文件**。凡 STARTING_POINT 已收录的，标注"已入起点"；源文件多出来的一并列出。

### 3.1 对象锁定（F1/F2）[已验证-推导+代码对账]
- 标准代码 YaRN（维度索引上频率比线性混合）：\(\boxed{\nu_j^Y=\omega_j(1-t+t/S)}\)，\(t=q/N\)，\(q=j-l\)，\(N=h-l\)（:13-29）。
- 写成累计压缩指数：\(m_Y(t)=-\log[1-(1-1/S)t]/\log S\)，**\(m_Y'>0\) 且 \(m_Y''>0\)** ⇒ 标准 YaRN 的累计压缩**同样凸、增量同样递增**（:51-63）。
- MrPro-Pro：\(\boxed{\nu_j^M=\omega_jS^{-m_q}}\)，\(m_q=q(q+1)/(N(N+1))\)（:37-42）；原文自述来自"假定 radix 的对数增量构成等差数列"（:45）。
- 论文附录 "regressive" 推导是对**旋转圈数 \(r_j\)** 线性插值，与按维度索引线性**不等价**；公开 mrRoPE 仓库的 YaRN 构造函数实际用的是维度线性版，圈数版在**未被调用的 `yarn2()`** 中（:73-75）。
- ⇒ **"YaRN 递减 → MrRoPE 递增"叙事作废**（:65-71）。已入起点 F2。

### 3.2 差别一：中前段扰动阶（F3）[已验证-重算]
- \(q=1\)：YaRN \(1-\nu_1^Y/\omega_1=(1-1/S)/N=O(N^{-1})\)；MrPro \(1-S^{-2/[N(N+1)]}\approx2\log S/[N(N+1)]=O(N^{-2})\)（:85-110）。
- 数值：Qwen \(N=17,S=4\) → YaRN 降 **4.4118%** vs MrPro **0.9020%**；Llama3 \(N=17,S=16\) → **5.5147%** vs **1.7958%**（:115-129）。
- 旋转差精确式：\(\|R(d\nu)-R(d\omega)\|_2=2|\sin(d(\nu-\omega)/2)|\approx d|\nu-\omega|\)（\(d|\nu-\omega|\ll1\)）（:137-148）。
- 全谱局部极限：\(\lim_{d\to0}\|R_\nu(d)-R_\omega(d)\|_F^2/(2d^2)=\sum_j(\nu_j-\omega_j)^2\)（:152-158）。
- Qwen 现配置：\(\sum_j(\nu_j^M-\omega_j)^2/\sum_j(\nu_j^Y-\omega_j)^2=\mathbf{0.4841}\)（:161-171）。
- **自带边界**（:175）："不能单独证明整网短任务分数更高…不能再次把这个局部结论升级成完整能力定理。" 已入起点 F3。

### 3.3 差别二：尺度响应 η（F4）[已验证-推导]
- \(\eta_j(S)=-\partial\log\nu_j(S)/\partial\log S\)（:189-194，物理含义："上下文长度增加 1%，这个时钟的频率大约降低多少百分比"）。
- YaRN：\(\boxed{\eta_Y(t,S)=t/[S(1-t)+t]}\)；任意严格中段 \(t<1\) 时 \(S\to\infty\Rightarrow\eta_Y\to0\)，且 \(\nu_Y\to\omega(1-t)>0\)（:202-230）⇒ **中段时钟随目标长度增长逐渐停止减速**。
- MrPro：\(\boxed{\eta_M=m_q}\)，只要 \(m_q>0\) 就持续幂律减速（:234-246）。
- 目标距离 \(d=SW\) 处未取模相位：\(d\nu_Y=W\omega[S(1-t)+t]\)（**S 线性**）；\(d\nu_M=W\omega S^{1-m_q}\)（**S^{1−m_q} 次线性**）（:248-262）。
- 已入起点 F4。

### 3.4 交点与 A/B 集合（F5）[已验证-公式]
源文件表（:270-286，与 `STARTING_POINT:59-67` 逐行相同）：

| 设置/槽 | \(\nu/\omega\) YaRN | \(\nu/\omega\) MrPro | MrPro 实际做了什么 |
|---|---:|---:|---|
| Qwen 4× 槽24 | 0.9559 | 0.9910 | 更接近原生 |
| Qwen 4× 槽28 | 0.7794 | 0.8729 | 更接近原生 |
| Qwen 4× 槽39 | 0.2941 | 0.2916 | 比 YaRN 更慢 |
| Llama3 16× 槽19 | 0.9449 | 0.9820 | 更接近原生 |
| Llama3 16× 槽26 | 0.5588 | 0.5208 | 更慢 |
| Llama3 16× 槽32 | 0.2279 | 0.1492 | 明显更慢 |
| Llama3 16× 槽34 | 0.1176 | 0.0850 | 明显更慢 |

- Llama3 槽32：\(\nu_M/\nu_Y=0.6544\) ⇒ **MrPro 周期比 YaRN 长约 52.8%**（:280-286）。
- A/B 集合：\(A=\{j:\nu_j^M>\nu_j^Y\}\)，\(B=\{j:\nu_j^M<\nu_j^Y\}\)；**由两张实际表的交点确定，不是人为阈值**（:400-415）。Qwen 4× A=槽24–37 / B=槽38–39；Llama3 16× A=槽19–24 / B=槽25–34。
- 结论句：\(\boxed{\text{降低中前段的局部扰动} + \text{让中后段真正承担持续的尺度扩展}}\)，且"不是笼统的'把预算往右移'，也不是'中段全部越接近原生越好'"（:288-298）。已入起点 F5。

### 3.5 OLMo 反例（F6）[已验证-独立实验]
源文件（:382）："**OLMo 配置下，MrPro 相对标准 YaRN 的局部旋转导数扰动，同样减少到约 47.8%。但独立72条实验中，长端 MrPro 为 2.78%，YaRN 为 6.94%，BM 达到 51.32%。**这个模型上，减少局部扰动并没有自动带来更好的任务结果。"
⇒ 任何 F 必须同时容纳三个事实：MrPro 在论文模型上的优势、BM 在 OLMo 上的大幅优势、BM/Smooth 在现 Qwen 上的分化（:384-390）。已入起点 F6。

### 3.6 非线性放大（F7）[部分证据]
- 槽28 从 YaRN 改到 MrPro：\(\Delta\nu=2.2174\times10^{-4}\)；同一差值在 \(d=32\) 处 \(d\Delta\nu\approx0.00710\) rad，在 \(d=131072\) 处 \(d\Delta\nu\approx\mathbf{29.0643}\) rad（:302-324）⇒ "一张只有几十个数的频率表，可以对长程计算产生很大的干预，同时保留大量局部计算"。
- "不需要训练"的精确含义（:326）："模型已经有相关内容识别、绑定和回答能力，换表改变的是这些计算在新的距离条件下是否还能正确连接，而不是凭空学会新知识。"
- 读出阈值机制：\(p_*=\dfrac{e^{z_*}}{e^{z_*}+\sum_{k\ne*}e^{z_k}}\)，\(D=z_*-\log\sum_{k\ne*}e^{z_k}\)，\(p_*=\sigma(D)\)（:338-350）。
- 论文侧数值：Llama3 128K RULER **79.9→86.6**；Infinite-Bench KV Retrieval **9%→27%**；"不是所有任务都有三倍提升"（:352）。
- 恒等式：\(R(Sd\cdot\omega/S)=R(d\omega)\)（:330）；局部关系 \(r\) 的扰动 \(r|\omega/S-\omega|=r\omega(1-1/S)\)（:303-310）。
- 边界（:334）："两种恒等式拼在一起，**不等于整个网络精确重演原生计算**"。已入起点 F7。

### 3.7 根理论（F8）[已验证-双源]
- 论文第 9 页用 \(B_\nu(d)=\sum_j\cos(d\nu_j)\) 的零点与中段 attention 曲线做解释（:358-366）。
- 我方的核查：该式对应 **query/key 分量独立同分布、相似 key 写成 \(k^*=q+\epsilon\)** 的特定模型，把"相似 key 与随机 key 的平均分差"化成无权重 cosine sum；原文自认多层堆叠使其不严格；出处 arXiv 2405.14591（`Base of RoPE Bounds Context Length`）（:368）。
- 判决：不能推出 \(B_\nu\) 首零点更远 ⇒ 真实冻结模型任务分数更高；MrPro 的算术递增 \(\epsilon_i\) 是**设计假设**，不是该 cosine 目标的最优解（:372-378）。已入起点 F8。

### 3.8 四格反事实（F9）[设计假设，未跑]
- \(\boxed{\nu^{\mathrm{fast}}_j=\max(\nu_j^Y,\nu_j^M)}\)（\(=M_A+Y_B\)）、\(\boxed{\nu^{\mathrm{slow}}_j=\min(\nu_j^Y,\nu_j^M)}\)（\(=Y_A+M_B\)）（:428-449）。
- 性质：零新增曲线参数、保持相同高低端/频率数量/gain、不破坏频率排序；**"必须从完整 prefill 起执行，不能换成固定状态重放"**（:451）。
- 判决表（:453-459）：fast>MrPro ⇒ 中后段额外压缩有负贡献；slow>MrPro ⇒ 更强的中后段重标定有价值、中前段保留并非正确取舍；MrPro>两者 ⇒ 两侧需配合，单侧规则被判死。已入起点 F9。

### 3.9 源文件中**未**进入 STARTING_POINT 的材料（取值时注意）
1. §五 的"论文理论未闭合"整段（:358-390），含 arXiv 2405.14591 的作用域说明。
2. §四 的"能力变化为什么可以很大"机制段（:302-354），含 29.06 rad 与 σ(D) 阈值论。**已入起点 F7 但源文件更细**。
3. §六 的"怎样真正超过 MrRoPE"论证段落与 A/B 分界表（:394-415）。
4. 尾注三条外部链接（:487-489）：jquesnelle/yarn 原仓、mattian7/mrRoPE 仓、arXiv 2405.14591。
5. **工件引用**（:485）：`sandbox:/mnt/data/mrrope_research/mrrope_yarn_verified_analysis.{py,json}` —— 与 `STARTING_POINT:112` 一致地标注"不在本地仓库"。**这两个文件本次不在仓库内，其数值不可复算**，一律按 [部分证据] 处理。

---

## 4. 向外提问的清单（= 问题原始表述，供校准今天的提法）

### 4.1 `RoPE_Allocation_Core_Problem_for_Pro_20260910.md`（短版，3 问）
- **目标（:5）**："从 EVQ 已有理论与有效分配规律出发，推导出更好的、能够跨模型和任务使用的分配规则，重点增强长程外推。**可以付出短程或中程性能代价；不要求所有任务同时获胜。**"
- 核心问题（:85）："**怎样从 EVQ 的频率资源分配思想出发，推导一种更好的分配规则，决定高、中、低频之间的取舍，尤其是目标长程下中频过渡的位置与形状？**"
- 交付四点（:89-92）：①明确数学问题（分配变量、目标、资源约束、"更好"对应什么量、与真实 rotary 计算的联系）；②从问题推出的规则/算法（**不是先选曲线再找解释**），并指出 EVQ 是特例/近似/对照；③可检查的推导 + 边界情况 + 小规模数值例子，特别是"为什么中频应在某处转折"；④少量有区分力的预测 + 什么结果会要求修改哪项假设。
- 明确禁令（:94）："不要把回答停在'还需要更多实验'，也不要把几何代理目标的改进直接写成模型任务必然提升。"
- 边界声明（:67）："\(k/K\) 与 \((k+1/2)/K\)、是否固定实际最低频率，并不等价。"（**离散约定口径**，与 `GLM_6PRO_REVIEW:51` 的"换成 midpoint 再锚定端点会产生略不同数字，不能混用离散约定"呼应。）

### 4.2 `RoPE_Allocation_Theory_Questions_for_Pro_20260910.md`（长版，6 问 + 交付）
- 目标（:7）："寻找整体上更好的频谱／压缩分配规则，**在权重冻结、零训练条件下超过 MrRoPE-Pro**，尤其争取明显的长上下文外推收益，同时尽可能跨任务、模型和长度成立。"
- 刻意打开的约束（:9）："高频不变、低频除S、中频分配及分段边界**均是可改的设计选择，不能从baseline概括反推硬约束**"；"不假设 MrPro 已经位于 Pareto frontier"；"现有 waterbed、smoothness、局部 gap 或 signed margin 分析均可重构，不能把其中任何一个当作最终答案"。
- 六问（:224-234）：
  1. 从失败机制到更好的整体规则，缺的最关键一步是什么？请给出**能同时容纳 BM 在 OLMo 大幅成功、在 Qwen 反转、P2 同 gain 条件收益、Smooth 固定预算失败**的解释。
  2. 如何把高/中/低的尺度角色变成下一张表的构造原则？**给公式或伪代码及可获取输入**；"只写'优化长程margin同时控制短程损失'还不构成方法"。
  3. waterbed 给的是限制还是自由度？固定累计 B 保留/排除了什么？"需要哪些最少条件才能把几何预算与实际长程能力联系起来"。
  4. 该规则如何使用已有正结果、又不拟合脆弱样本？"请提出能在看目标输出前冻结的规则与预测"。
  5. 最少哪几个实验能判定新规则真的走对了？（至多 3 个独立机制差异的规则 + 必要控制）
  6. 怎样形成可信但不过度声称的理论贡献？"能否提出一个**条件性命题或清晰可证伪预测**"。
- 交付（:236-240）：**少数真正可实现的新分配规则** + 最小决定性实验；每条说明机制/构造/适用条件/与 MrPro 差别/对已有结果的解释/可能失败的预测；"可以指出 Codex、5.6 Pro、此前 6 Pro 和 Qwen3.8-Max 都误判了什么；也请明确保留已经测到的条件性成功"；"不需要一份只列审查门槛的报告"。

### 4.3 `Nongeometric_RoPE_Questions_for_GPT6Pro_20260910.md`（10 问 + 交付）
- 目标与边界（:7-9）：同上；附加"**可以使用少量校准数据选择频率表，但必须区分'一个固定规则直接迁移'与'各 checkpoint 重新校准的算法'；不能用目标测试集调参后称作泛化**"；"几何指标、attention mass、局部 NMSE 和 NLL 均不能替代下游任务"。
- 十问（:196-214）逐条主题：
  1. 为什么从 YaRN 到 MrPro 都采用高/中/低频？**以 \(r_j=W\omega_j/(2\pi)\) 衡量训练窗口转圈数**；"为什么是三段、为什么 1/32 圈能作为边界？"（注：这是**提问里的一个未解释常量**，五份材料均未给出 1/32 的推导来源）。
  2. 如何把 signed margin 变成可泛化的分配规则？"目标 attention mass 改善却 E8 任务退化，否定了哪种充分条件？"
  3. s28 与 s29 是否标出有意义的"过渡结构"，还是两个偶然脆弱点？
  4. BM 的闭式最小粗糙度为什么会有任务意义？"**保持端点的 shape×budget 2×2**"如何定义。
  5. FullLagP2 的 conditional residual 是否抓住了有效信息？"min-max 及平方映射为何合理"。
  6. prefix/readout 的连续分解可以支持多强的理论结论？
  7. gain 与长度应该怎样与频率一起研究？"**补 MrPro×.074 可闭合 2×2**"。
  8. 精度改变的是测量、构造，还是可部署的研究对象？（BF16 NMSE 高出约 161 倍）
  9. **跨模型应运输什么？** 绝对 slot / normalized transition 位置 5/17 / \(W\omega_j\) / 相位周期数 / 训练分布统计 / 重新校准的功能坐标？
  10. 什么证据足以支持"超过 MrPro 且尽可能通用"，并决定是否转向动态核？
- 交付（:218）："先给出你认为最可信的 **1–3 个核心机制**…再给最值得执行的 **3–5 个小实验或现有结果重分析**"。
- 明令（:220）："不要把本文件中'尚未完成'写成结果。"

### 4.4 三份提问包的共同口径（用于校准我们今天的提法）[已验证-自述]
1. **静态表优先但非不可突破**（`Nongeometric:9`）；动态核需"精确定义、native-prefix 性质、cache 语义与一个能区分必要性和额外自由度的静态对照"（`Nongeometric:214`）。
2. **禁止把几何/NLL/attention-mass 当终点**（`Nongeometric:9`、`Theory:9`、`Core_Problem:94`）——与红线 R2 同源。
3. **要求事前可冻结的规则与预测**（`Theory:230`、`Core_Problem:92`）。
4. **明确保留已测到的条件性成功**，不许用一次失败否定整族（`Theory:238`、`Nongeometric:9`）。

---

## 5. 可作 F 零件的公式（精确形式，带出处与等级）

### 5.1 配置层（坐标与约束）
| # | 公式 | 出处 | 等级 |
|---|---|---|---|
| F-a | \(\nu_j=\omega_j S^{-m_j}\)，\(S=L/W\)；\(m_j=0\) 保留原频率，\(m_j=1\) 完整 PI | `Core_Problem:23-28`；`Nongeometric:17-22` | [已验证-代数] |
| F-b | \(\lambda_j=S^{\Delta_j}\)，\(\Delta_j=m_{j+1}-m_j\)，\(\prod\lambda=S\) | `NEXT_DERIVATION:26`（**非五份材料**，列出以便坐标系对齐） | [已验证-代数] |
| F-c | 对数频率坐标 \(x_j=-\log(\nu_j/\nu_{\rm ref})\)，\(a_i=x_i-x_{i-1}=\log(\nu_{i-1}/\nu_i)>0\)，\(A=\sum a_i\) | `pro6:29-42` | [已验证-代数] |
| F-d | **频谱三分解**：频率表 = 高频端位置 \(x_0\) + 总跨度 \(A\) + 间隔分配 \(\{a_i/A\}\) ⇒ 平移/扩支持/内部重分配是三种不同操作 | `pro6:47-56`、`pro6:60-67`（操作对照表） | [已验证-代数] |
| F-e | MrRoPE 边界条件：\(x_j=x_j^{(0)}+m_j\log S\)；\(a_i=a_i^{(0)}+\epsilon_i\log S\)，\(\epsilon_i=m_i-m_{i-1}\)；\(\sum_i\epsilon_i=1\)，\(\sum_i(a_i-a_i^{(0)})=\log S\) | `pro6:74-104` | [已验证-代数] |
| F-f | 过渡带额外 gap 约束：\(\log\dfrac{\nu_{l+i-1}}{\nu_{l+i}}=\dfrac{\ln b}{64}+\epsilon_i\ln S\)，\(\sum_{i=1}^{N}\epsilon_i=1\) | `Theory:98-102` | [已验证-代数] |
| F-g | 累计预算质心：\(B=\sum_{q=1}^{N-1}m_q=\sum_{i=1}^{N}(N-i)\epsilon_i=N-\sum_{i=1}^{N}i\epsilon_i\) | `pro6:110-116`；`Theory:105-110` | [已验证-代数]；**注意 B 的两种求和口径**（§8-C3） |
| F-h | 端点在 m 坐标的锁定：\(m_j=0,\ j\le23\)；\(m_j=1,\ j\ge40\)；\(m_{40}-m_{23}=1\Rightarrow\sum_{j=23}^{39}\Delta_j=1\) | `NEXT_DERIVATION:41-43`；`Theory:28` | 面板支持 [已验证=面板]，**但强度口径是"强基线设计约束，非零容忍定理"**（`NEXT_DERIVATION:41`、`GLM_6PRO_REVIEW:11`） |

### 5.2 目标泛函候选件
| # | 公式 | 出处 | 等级 |
|---|---|---|---|
| F-i | EVQ 原目标：\(\mathcal J[\rho]=\frac\alpha2\int_0^1\rho(\phi)^2d\phi+\frac\beta2\int_0^1S_\rho(t)^2dt\)，\(S_\rho(t)=\int_t^1\rho(\phi)d\phi\) | `pro6:136-148`；`Core_Problem:35-48` | [已验证-定义] |
| F-j | **EVQ 精确变量替换**：\(h(u)=\phi'(u)\)，\(\rho(\phi(u))=1/h(u)\)，\(S_\rho(\phi(u))=1-u\) ⇒ \(\mathcal J[h]=\frac12\int_0^1\left[\frac{\alpha}{h(u)}+\beta(1-u)^2h(u)\right]du\)，\(\int_0^1h=1\) | `pro6:151-194`；`GLM_6PRO_REVIEW:44-51`（数值验证 1.6e−11 / KKT 残差 1.1e−15） | [已验证-CPU 数值] |
| F-k | EVQ 一阶条件：\(-\frac{\alpha}{2h(u)^2}+\frac\beta2(1-u)^2+\lambda=0\Rightarrow h(u)=\sqrt{\dfrac{\alpha}{\beta(1-u)^2+2\lambda}}\) | `pro6:207-225` | [已验证-代数] |
| F-l | Cosh 解：\(h_\tau(u)=\dfrac{\sinh\tau}{\tau\sqrt{1+(1-u)^2\sinh^2\tau}}\)，\(\tau=\sqrt{\beta/\alpha}\)；对应密度 \(\rho_\tau(\phi)=\frac{\tau\cosh(\tau(1-\phi))}{\sinh\tau}\)，分位点 \(\phi(u;\tau)=1-\frac1\tau\operatorname{asinh}((1-u)\sinh\tau)\) | `pro6:230-238`；`Core_Problem:52-62` | [已验证-推导] |
| F-m | 局部算子导数扰动：\(\lim_{d\to0}\|R_\nu(d)-R_\omega(d)\|_F^2/(2d^2)=\sum_j(\nu_j-\omega_j)^2\)；逐 pair 精确式 \(\|R(d\nu)-R(d\omega)\|_2=2|\sin(d(\nu-\omega)/2)|\le\min\{2,|d(\nu-\omega)|\}\) | `mrrope:152-158`、`pro6:277-282`；`Theory:62-66` | [已验证-代数]；**作为 L_near 载体是候选，F6 OLMo 反例禁止其单独成 F**（`STARTING_POINT:103`） |
| F-n | 带正 ridge 的配方分解（**冻结兼容成本**）：\(J_r(c,\nu)=\mathbb E_{d\sim p_r}[f_r(d)-c^\top\Phi_\nu(d)]^2+\zeta\|c\|^2\)；\(G_r=\mathbb E[\Phi_\nu\Phi_\nu^\top]\)，\(b_r=\mathbb E[\Phi_\nu f_r]\)，\(c_r^*=(G_r+\zeta I)^{-1}b_r\)；精确分解 \(J_r(c_{0,r},\nu)=\underbrace{\mathbb E[f_r^2]-b_r^\top(G_r+\zeta I)^{-1}b_r}_{\text{表示质量}}+\underbrace{(c_{0,r}-c_r^*)^\top(G_r+\zeta I)(c_{0,r}-c_r^*)}_{\text{兼容成本}}\) | `pro6:406-470`；`GLM_6PRO_REVIEW:53` | 分解本身 [已验证-代数]；**"用于组织假设，不能直接当成整网收益定理"**（`pro6:483`） |
| F-o | 尺度响应 \(\eta_j(S)=-\partial\log\nu_j(S)/\partial\log S\)；YaRN \(\eta_Y=\frac{t}{S(1-t)+t}\)，MrPro \(\eta_M=m_q\) | `mrrope:189-246` | [已验证-推导] |
| F-p | 目标距离未取模相位：\(d\nu_Y=W\omega[S(1-t)+t]\)（S 线性） vs \(d\nu_M=W\omega S^{1-m_q}\)（次线性） | `mrrope:248-262` | [已验证-推导] |
| F-q | 读出竞争：\(p_*=\sigma(D)\)，\(D=z_*-\log\sum_{k\ne*}e^{z_k}\)；多 distractor 版 \(p_*=e^D/[e^D+n-1]\)（保持相同 mass 需 \(D\approx\ln(n-1)+\text{const}\)） | `mrrope:338-350`；`Theory:69-70` | [已验证-代数]；**"attention margin 与输出 token margin 不能混作同一数值"**（`Theory:70`） |
| F-r | 完整 logit 的带内容系数形式：\(z_{pt}^\nu=\frac{g^2}{\sqrt{128}}(q_p^\nu)^\top R_\nu(t-p)k_t^\nu\)；等价 \(z_t(\nu)=\frac{g^2}{\sqrt{128}}\sum_j[A_{tj}\cos(\Delta_t\nu_j)+B_{tj}\sin(\Delta_t\nu_j)]\) | `Theory:55-58`；`Nongeometric:160-163` | [已验证-代数] |
| F-s | 共同频移恒等式（±δ 同向平移）：\(A_\delta(d)=\cos(d\delta)A(d)+\sin(d\delta)Q(d)\)，\(Q(d)=\sum_j[-a_j\sin(d\nu_j)+b_j\cos(d\nu_j)]\) | `Theory:201-207` | [已验证-CPU，双向六距离误差 ≈8.9e−16]；**边界**："两臂保留带内所有绝对频差 \(\nu_j-\nu_k\)，因此主要是测试该带共同相位方向，不是在测试所有可能的带内 clock-spacing 重分配" |

### 5.3 只可作诊断、**不得入 F** 的静态几何量（红线 R2，材料中反复出现）
| 量 | 公式 | 出处 | 判决 |
|---|---|---|---|
| 平方弦距 | \(D^2(\Delta;\nu)=4\sum_jw_j\sin^2(\Delta\nu_j/2)\)；"不依赖绝对位置 p" | `Nongeometric:153-157` | 禁入 F（R2）。原文已否证单调解读：s28_less 在 1K/2K/4K/8K/16K 的改动为 **+.1623/−.3651/+.5320/−1.5277/−.4050**，s29_more 为 **−.1988/−.0022/+.0087/+.0340/+.1255** |
| Σcos 首根 | \(B_\nu(d)=\sum_j\cos(d\nu_j)\) | `mrrope:358-378`；`STARTING_POINT:86` | 禁入 F；"根排序与能力排序显著失序" |
| roughness | 相邻 \(\epsilon\) 差分平方和 | `Theory:120-122`、`Theory:150-155` | 禁入 F（Smooth 反例：几何全赢，far 68.3 vs 78.13） |
| 洞比率 / 覆盖 / 有效秩 / 能量 | — | `NEXT_DERIVATION:111`（**非五份材料**，列出以便对齐） | 禁入 F |
| 周期位置计数 | "周期位于 32K–128K 的槽数" | `Theory:150-155` | 只作描述，不作选择子 |

### 5.4 三份提问包里给出的其他方法构造（可作 F 的可行点/对照点）
- **BM**（Boundary-Matched）：\(\epsilon_i^{BM}=\frac{6i(N+1-i)}{N(N+1)(N+2)}\)，\(m_q^{BM}=\frac{q(q+1)(3N+2-2q)}{N(N+1)(N+2)}\)；**唯一最小化 \(\sum_{i=0}^{N}(\epsilon_{i+1}-\epsilon_i)^2\)**，约束 \(\epsilon_0=\epsilon_{N+1}=0\)、\(\sum\epsilon_i=1\)（`Theory:38-44`、`Nongeometric:46-53`）。严格中段 \(m_q^{BM}-m_q^{Mr}=\frac{2q(q+1)(N-q)}{N(N+1)(N+2)}>0\)（`Nongeometric:55-57`）；内部 q=1..N−1 的 exponent 总和：MrPro \((N-1)/3\)、BM \((N-1)/2\)（`Nongeometric:59`）。
- **MrUni** \(m_q=q/N\)，与 BM 同总和，用于排除"总预算足够"的最简解释（`Nongeometric:59`）。
- **FullLagP2**：\(X_j(d)=\sqrt{w_d}[\cos(d\omega_j),\sin(d\omega_j)]\)，\(w_d\propto W-d\)，`rcond=1e−10`；\(u_j=\frac{\min_B\|X_j-X_{-j}B\|_F^2}{\|X_j\|_F^2}\)，\(\bar u_j\) min-max 归一，\(m_j=(1-\bar u_j)^2\)，**全部 64 槽**使用，历史 gain \(g_{.074}\)（`Nongeometric:65-80`、`Theory:48`）。
- **BM_ScaleTaper**：\(T_j=2\pi/\nu_j^{Mr}\)，\(w_j=\mathrm{clip}[\ln(W/T_j)/\ln S,0,1]\)，\(\nu_j=\nu_j^{Mr}(\nu_j^{BM}/\nu_j^{Mr})^{w_j}\)（`Theory:220`）；**已实现/排队，尚无任务结果**。
- **E1 冻结迁移规则**：在目标 MrPro transition 的 \(5/17\) 处取最近 slot \(j=l+\mathrm{round}[(h-l)5/17]\)，用前一 slot 的 exponent 替换（`Nongeometric:41`）。
- **LongBridge**：\(\nu_j^\pm=\nu_j^{Mr}\pm\frac{1}{131072}\)，\(j\in\{36,37,38,39\}\)（`Theory:195-199`）；Slower 周期 35,446/50,830/74,190/110,764，Faster 32,637/45,245/62,864/87,285。

---

## 6. 已验证数字（带出处）

### 6.1 部署/结构常量
| 数字 | 值 | 出处 |
|---|---|---|
| 模型 | Qwen2.5-3B-Instruct, rev `aa8e7253…`, 36L/16Q/2KV/d128 | `Nongeometric:15` |
| rotary slots K | 64 | 同上 |
| W / b / S / L | 32768 / 10⁶ / 4 / 131072 | `Nongeometric:15`, `Theory:17` |
| 默认 gain | \(g_{.1}=1+0.1\ln4\approx1.1386294\)；固定态 logits ×\(g^2\) | `Nongeometric:31`, `Theory:26`；`pro6:336-342` |
| 备用 gain | \(g_{.074}=1+.074\ln4\approx1.1025858\) | `Nongeometric:61` |
| 零基约定 | \(j=0..63\)，末对 \(\omega_{63}=b^{-63/64}\) | `Nongeometric:15`、`Theory:17` |
| MrPro 边界 | \(l=23,h=40,N=17\) | `Nongeometric:26`, `Theory:28` |
| MrPro \(\epsilon\) | \(\epsilon_i^{Mr}=2i/[N(N+1)]=2i/306\) | `Nongeometric:34`, `Theory:32` |
| MrPro 周期表（slot 23/28/29/36/37/38/39/40/51） | 900 / 3,035 / 3,977 / 33,983 / 47,875 / 68,059 / 97,632 / 141,332 / 1,518,763 tokens | `Theory:76-86` |
| Native 在 32K 内圈数（slot 23/28/29/36/37/38/39/40/51） | 36.393 / 12.367 / 9.966 / 2.199 / 1.772 / 1.428 / 1.151 / 0.927 / 0.086 | `Theory:76-86`、`GLM_6PRO_REVIEW:9` |
| 过渡段 17 gap 总跨度 | ≈**5.056039** nats（原生部分 ≈**3.669745**） | `GLM_6PRO_REVIEW:7`、`digest_theory-core.md:283` |
| 等距间隔（native，base=1e6, K=64） | **0.21587346** = ln(10⁶)/64 | `pro6:248`；`tables/ground_truth_tables.json:8011` 实测同值 |
| Native 总对数跨度 / MrRoPE 跨度 | **13.5996 / 14.9859**（+**10.19%**） | `pro6:119`；独立佐证 `digest_theory-core.md:283`（"13.60→14.99"） |
| MrPro \(\sum_j m_j\)（全 64 槽）/ 内部 B | **29.3333** / **5.3333**（=16/3） | `GLM_6PRO_REVIEW:17`；`Theory:152` |
| BM / MrUni 的 B | **8**（= (N−1)/2） | `Theory:152`、`Nongeometric:59` |
| 新增 log-gap 槽位质心 | MrPro 34.6667 / E1 s28 34.6993 / LBS 34.4398 / P2 29.8211 | `GLM_6PRO_REVIEW:15-20` |
| roughness | MrPro **.013071896** / BM **.002063983** / MrUni **.006920421** / Smooth **.004886399** | `Theory:150-155`；`Theory:122`（Smooth 由 .0130719 降到 .0048864） |
| BM/MrUni/Smooth 位于 32K–128K 的槽数 | 5 / 4 / 4 | `Theory:150-155` |

### 6.2 Qwen3B 同输入开发面板（36 条，near/far）[已验证=面板；**不作独立泛化证明**]
`Theory:130-145` 与 `Nongeometric:86-98` 两处一致（后者为相对 MrPro 基线的 Δpp）：

| 表/gain | 32K | 128K | 出处 |
|---|---:|---:|---|
| MrPro / .1 | 87.2222% | 78.1250% | `Theory:134` |
| BM / .1 | 91.6667% | 70.8333% | `Theory:135` |
| MrUni / .1 | 64.5833% | 73.3333% | `Theory:136` |
| MrPro / .074 | 98.3333% | 75.3472% | `Theory:137` |
| BM / .074 | 100.0000% | 70.0000% | `Theory:138` |
| BM / gain=1 | 89.5833% | 58.8194% | `Theory:139` |
| **E1 s28_less / .1** | 87.2222% | **83.3333%** | `Theory:140` |
| E1 s29_more / .1 | 95.5556% | 77.9167% | `Theory:141` |
| s28+s29 组合 / .1 | 87.2222% | 73.9583% | `Theory:142` |
| E7 局部输出保护投影 / .1 | 90.0000% | 68.6111% | `Theory:143` |
| Smooth(MrBudget) / .1 | 87.2222% | **68.3333%** | `Theory:144` |

- **gain 与表效应的分离**：BM×.1 → BM×.074 的表效应 = 短 **+1.6667** / 长 **−5.3472 pp**（`Theory:138`）；MrPro 光换 gain 就使 32K **87.22→98.33**（`pro6:344`）。⇒ **"不同 gain 面板不能混算"**（两侧均声明）。
- Smooth 的失败分解：near 打平（87.2 vs 87.22），**全部损失在 far**（`INTEGRATION:101` FLAG-3）；具体为 MK2 .75→.25、QA .5→.25，同时 MQ .9375→1、VT .75→.85（`Theory:146`）。
- MrUni 长端：MK2 .75→.5、MQ→1、VT→.9、QA→.25（`Theory:148`）。
- E1 s28_less 的 slot 定义：\(\nu_{28}:0.00207001995\to0.00216595642\)（\(m_{28}:30/306\to20/306\)）；s29_more：\(\nu_{29}:0.00157984428\to0.00148275390\)（\(m_{29}:42/306\to56/306\)）（`Theory:46`、`Nongeometric:38-39`）。
- **s28_less 的未取模相位变化**：32K **+3.144 rad**，128K **+12.575 rad**，上述真实证据距离处 **+8.512 rad**（`Theory:92`）；"最大长度处接近整数圈，并不代表中间距离的变化小"。
- **BM 相对 MrPro 在 slot38 造成 −1.676 rad 相位变化**，虽然 40 之后完全不变（`Theory:88`）。

### 6.3 历史独立成功（不可丢）
**OLMo-2-0425-1B-Instruct（实际 1.485B）**，W=4096、base=500000、边界 14/32、S4、gain .1（`Nongeometric:105`、`Theory:163`）：

| 面板 | 4K | 16K |
|---|---:|---:|
| MrPro 开发 36 条 | 37.22% | 14.93% |
| BM 开发 36 条 | 79.44% | 49.03% |
| MrPro 独立 seed 72 条 | 37.85% | **2.78%** |
| **BM 独立 seed 72 条** | **81.81%** | **51.32%** |
| MrUni 同 72 条 | 76.88% | 32.12% |
| 官方 YaRN 同 72 条 | 54.38% | 6.94% |

- 独立集每任务 4K 四条 / 16K 八条，**与开发 prompt 不重叠**，BM vs MrPro **44 胜 0 负 28 平**（`Nongeometric:116`、`Theory:163`）。
- BM 优于**同 B** 的 MrUni ⇒ 排除"只看总预算"的最简解释（`Nongeometric:116`）。
- 同规则 S8 到 32K：MrPro **0.69%**、BM **6.94%**，接近地板；降低 scale/gain 未恢复（`Nongeometric:116`、`Theory:163`）。**"不能称解决 32K"**。
- **F6 反例数字**：MrPro 对 YaRN 的局部旋转导数扰动同样降到 ~**47.8%**，但长端 MrPro 2.78% < YaRN 6.94% ≪ BM 51.32%（`mrrope:382`）。
  （另注：`INTEGRATION:52` 记录 OLMo 16K **350 条**口径为 BM 41.67% vs MrPro 7.09%、+34.59pp（156W/9L）、multikey_3 双方皆 0、EOS BM119/MrPro197——**与上表 72 条口径不同，不得混算**。此为跨文档口径差异，见 §8-C4。）

**FullLagP2**（decoder repetition penalty=1.1，与当前面板不同，`Nongeometric:118`）：

| 模型/长度/n | 方法 | MK2 | VT | FWE |
|---|---|---:|---:|---:|
| Qwen1.5B/64K/n=8 | MrPro | 12.50 | 82.50 | 45.83 |
| 同上 | MrPro 同 gain .074 | 25.00 | 77.50 | 45.83 |
| 同上 | **FullLagP2** | **37.50** | **87.50** | **70.83** |
| Qwen1.5B/128K/n=8 | MrPro | 0 | 72.50 | 50.00 |
| 同上 | FullLagP2 | 0 | 85.00 | 50.00 |
| Qwen3B/64K/n=4 | MrPro | 50.00 | 95.00 | 75.00 |
| 同上 | FullLagP2 | 75.00 | 90.00 | 66.67 |

- P2 在 1.5B 64K 对官方 Mr **9胜13平2负**、对同 gain Mr **10胜12平2负**；3B 64K 只有 **1胜9平2负**且缺同 gain 对照（`Nongeometric:130`）。
- `Theory:74` 给出 P2 的 3B 开发集 128K 宏平均 **81.6667%**，MrPro 默认 gain **78.1250%**，同 gain 对照 **75.3472%**；`pro6:363` 把差值读作 **+6.32pp**（81.6667−75.3472）。**`pro6:363` 自己标注该数字"目前应按有待统一运行记录的开发结果使用，不算独立确认"。**
- `Core_Problem:74` 补充 P2 主要收益来自 **QA、变量追踪**，另有 multi-key 等代价；32K 代价大。
- `Theory:48` 补充 P2 前中段更贴 Native，**slot30/31 的 m 已达 .85056/.99785**（Mr 为 .18301/.23529），32 之后近似整段除 4，深尾 40–63 与 Mr 相同（表结构事实，[已验证-表]）。

### 6.4 跨模型冻结迁移（已完成）
`Theory:168-175`、`Nongeometric:134-141` 一致：

| 目标/面板 | 短端 Δpp | 长端 Δpp | NLL 差 |
|---|---:|---:|---|
| Qwen2.5-7B / 18 条（32K+128K） | 0 | **−1.667** | 8/16/32K：−.000045、+.002478、−.002505 |
| OLMo / 开发 36 条（4K+16K） | −3.333 | **+3.819** | 4/8/16K：−.000584、+.001093、−.006797 |

- 源规则在目标 MrPro 过渡区 5/17 位置取最近 slot：Qwen7B → slot 28，OLMo → slot 19（`Theory:168`）。
- 结论：**"不是统一赢家"**（`Theory:175`）；OLMo 长端 14.93%→18.75%，仍远低于同面板 BM 49.03%（`Nongeometric:141`）。

### 6.5 机制诊断数字
- **E7 精度拆解**（同冻结状态/同局部支持，输出 NMSE）：linear 预测 **7.93668e−8**；理想 relative-phase 精确有限改动 **7.93640e−8**；FP32 绝对坐标 **7.93255e−8**；BF16 绝对坐标 **1.28074e−5**（`Theory:179`、`Nongeometric:167`）。**"约 161 倍是平方误差之比，不能称线性近似失效 160 倍"**。
- **MK2 prefill/read 四格**（0=MrPro，1=E1）：s28 128K MK2 = (0,0,0,1)；s28 128K MQ = (.75,1,.75,1)；s29 128K VT = (.8,1,1,1)；s29 32K QA cached = (0,0,0,0)（`Nongeometric:173-179`）。
- 首分歧 digit 的正确 6 vs 错误 9 logit margin 依次 **−2.125、−1.125、−1.000、+.250**；prefill 单独 +1.125、read 单独 +1、交互余项 +.250 ⇒ **"二元四格像 AND，不等于强非加性交互"**（`Nongeometric:181`、`Theory:183`）。
- 正确完整答案 6683176，错误答案 9424151（`Nongeometric:180`）。
- **值互换反例**：互换两条记录的等 token 长度数字后两方法均错（输出 6624365/6624369）；全局位置 ID+1 使 E1 成功变失败（`Theory:185`）。
- **QA cached 路径失去原现象**：同表 E1 full-prefill 得分 1，切分最后 query 的 cached 路径已变 0，四格 cached 均 0（`Theory:187`、`Nongeometric:182`）。
- **LongBridge 完成结果**（`Theory:213-216`）：

| 方向 | 32K均分/相对Mr | 128K均分/相对Mr | 8/16/32K NLL 差 |
|---|---|---|---|
| Slower | 80.5556% / **−6.6667pp** | **80.0694% / +1.9444pp** | +.0010876／−.0002314／−.0002280 |
| Faster | 87.2222% / 持平 | 73.9583% / **−4.1667pp** | +.0002666／−.0007077／−.0008891 |

  Slower 全 36 条 3 胜 3 负；Faster 全 36 条 0 胜 1 负。**"给出长端正信号但存在短端代价，不是明显大幅超过 MrPro"**（`Theory:218`）。
- **16 篇自然文本尾 512 NLL**（`Nongeometric:101`）：s28 相对 MrPro = +0.000154/−0.000604/−0.000882；s29 = +0.000958/−0.000211/−0.001363 nats/token。**"不能叫全篇 PPL 或已证明等价"**。
- **E2 / E8 首筛 12 条**：E2 慢尾额外压缩 128K −9.722pp；E8 \(\nu_{51}=0\) −13.889pp；症状为 MQ 重复绑定/漏答、FWE 退化；**E8 选择器代理本来很好**（`Nongeometric:96-97`、`Theory:159`）。**"是'更多目标 attention mass 就够了'的反例"**。
- **E9/E10**：仅 12 条初筛全部得分持平，"不能称完整失败"（`Theory:159`）。
- **E9_distance 的 w**：用 **w=26 tokens**，来自校准数据中完整 primitive 键值记录的最大 token 跨度，**不是模型参考窗口 W=32768**（`Nongeometric:190`）。

---

## 7. 死路登记（五份材料中已证伪/已失败的机制，含失败原因）

| # | 机制 | 失败证据 | 出处 |
|---|---|---|---|
| D1 | **"YaRN 递减 vs MrRoPE 递增"** | 标准代码 YaRN 的 \(m_Y''>0\)，两者都凸都递增 | `mrrope:51-71`、`STARTING_POINT:9`（红线更新） |
| D2 | **"MrPro 所有中频都比 YaRN 更接近原生"** | 实际频率在 B 段**交叉**（Qwen 槽39、Llama3 槽25–34 更慢） | `mrrope:266-286` |
| D3 | **"把预算整体往右移"是胜因** | A/B 分界随 S 移动；"不是笼统的把预算往右移" | `mrrope:288-298` |
| D4 | **"更少破坏原生"⇒任务更强** | OLMo 反例：局部扰动同样降到 47.8%，长端 MrPro 2.78% < YaRN 6.94% ≪ BM 51.32% | `mrrope:382`、`STARTING_POINT:75` |
| D5 | **"中段全部越接近原生越好"** | 同上（B 段多压是 MrPro 相对 YaRN 的实际动作） | `mrrope:298` |
| D6 | **Σcos 首零点更远 ⇒ 任务分数更高** | 根排序与能力排序失序（MrUni 82.2K > MrPro 80.3K 而 32K 64.6≪87.2） | `mrrope:368-378`、`STARTING_POINT:86` |
| D7 | **高频保持 + 慢频除 S 两个恒等式 ⇒ 全网精确重演** | "不同频段采用不同尺度，网络内部还会重新形成状态"；prefill/read 交叉实验显示收益不只在固定 Q/K 读出 | `mrrope:334`、`STARTING_POINT:87` |
| D8 | **"少压缩所以相位分离更好"作为统一解释** | 平方弦距对频率不单调（s28 五距离 +.1623/−.3651/+.5320/−1.5277/−.4050） | `Nongeometric:157`、`Theory:181` |
| D9 | **E7 "一阶近似错了 160 倍"** | 161 倍是**平方误差**之比，出现在有限精度旋转，非线性化误差 | `Nongeometric:167` |
| D10 | **"更多目标 attention mass 就够了"** | E8 固定态代理很好而真实生成退化（−13.889pp） | `Nongeometric:97`、`Theory:159` |
| D11 | **固定 MrPro 预算最小化 roughness（Smooth_MrBudget）** | 凸问题解对（roughness .0130719→.0048864，约束误差 <1e−15），128K 仍低 **9.7917pp** | `Theory:122`、`INTEGRATION:28`（红线 R2 的决定性反代理） |
| D12 | **"两个局部正方向可加"（pair 28+29）** | 组合后 128K −4.17pp | `pro6:359`、`Theory:142`、`INTEGRATION:52` |
| D13 | **全频率 PI 一并除 S 会增加总频谱跨度** | "若统一 PI 使全部频率同时除 S，总频谱跨度本身不增加" | `Theory:112` |
| D14 | **要求 \(\epsilon_i\ge0\) 是正频率/有序的必要条件** | "同样是一个表族约束，不是正频率或有序频率的全部必要条件" | `Theory:112` |
| D15 | **gain .074 可当"arc 恢复证据"** | "只改幅度，不能当 arc 恢复证据；BM×gain 相对 Mr 的总差不是 gain 因果效应" | `Nongeometric:186` |
| D16 | **动态映射 \(G_t=\lceil(t+1)/W\rceil,\phi(t)=t/G_t\)** | "在 W 处向后跳变；token-attached mapping 与 query-dependent 重映射是不同核/cache 语义" | `Nongeometric:190` |
| D17 | **零面积/对称性/最小二乘下降 ⇒ 新规则能力依据** | 用 \(\psi(x)=x(1-x)(2x-1)\) 拟合 P2−Mr 仅解释 13.18% 几何差异能量，且在 slot29–31 拟反 | `Theory:124` |
| D18 | **逐槽相位 cost 可独立相加** | 完整 logit 误差平方含 \(2\sum_{j<k}\mathbb E[\Delta z_j\Delta z_k]\) 交叉项；softmax 与后续 readout 进一步耦合 | `Theory:124`、`pro6:402` |
| D19 | **有限搜索失败 ⇒ 静态表族无 headroom / 全族上界** | "有限搜索失败不能证明静态表族无 headroom"；同理"即使两臂都输，也只否证这两张表" | `Nongeometric:189`、`Theory:209` |
| D20 | **慢带 ν=0 ⇒ 维度无用** | "ν=0 后仍保留内容通道，不能把输出持平叫维度无用" | `Nongeometric:189` |

**注**：D1–D7 与 `STARTING_POINT` 的红线一致；D11 与 `INTEGRATION:28` 的 R2 一致。**五份材料未出现任何 VICTORY / 已闭合 / 已证明 类结论**，其中最接近的是 `Theory:44` "这个几何优化解已验证"（指 BM 的最小粗糙度**几何**定理）与 `Theory:122` "凸问题确实解对了"——**两者均已被材料自己限定为几何层，不是任务层**，本次按降级处理。

---

## 8. 矛盾与口径不一致

### C1. "6Pro 修正条款"的归属（本次最主要的口径问题）
- 任务描述把四条修正归给 `pro6_allocation_analysis.md`；实际**逐字出处**在 `docs/research/ROPE_GLM_6PRO_REVIEW_AND_VALIDATION_20260910.md` 的 **"GLM：…"** 小节第 1–6 条，而 `NEXT_DERIVATION:41/43/51/76/115` 以"6Pro 修正/澄清/第 6 点"引用之。
- `pro6_allocation_analysis.md` 自身**没有**这四条的形式化编号；它承载的是**同实质**的 P1/P2/P6/P8（§2.1）。
- ⇒ 引用时应**同时给两个路径**，或直接引 `ROPE_GLM_6PRO_REVIEW` 的具体行号。**不得把 G1–G5 写成"pro6 分析的原文"。**

### C2. "预算"一词至少三种用法（严重，会污染 F 的约束项）
| 用法 | 定义 | 出处 |
|---|---|---|
| 总压缩预算（用户/INTEGRATION 原话） | "Σ_j m_j = log S / log S（即 Σm 固定于过渡段）" | `INTEGRATION:17` |
| 总对数跨度 A（或过渡段总跨度） | \(A=\sum a_i\)；过渡段 17 gap = 5.056039 nats | `pro6:44`、`GLM_6PRO_REVIEW:7`、`NEXT_DERIVATION:43` |
| 累计压缩质心 B（内部 q=1..N−1） | \(B=\sum_{q=1}^{N-1}m_q\)，MrPro = 16/3 | `pro6:110-116`、`Theory:105-110` |
| 全表 Σ_j m_j | MrPro = 29.3333（含尾部 24 槽 ×1） | `GLM_6PRO_REVIEW:17` |
- 五份材料（`pro6:68`、`Theory:31`）与 `NEXT_DERIVATION:107` 一致要求**不得把三者统称同一预算**；但 `INTEGRATION:17` 的用户原话仍写成"Σ_j m_j = log S / log S"。**这是权威文档与源材料的直接冲突点**，落 F 时必须换元到一个具体坐标（红线 R4）。
- 本次复算补充：\(29.3333 = 5.3333 + 24\)，两个量恒差 24（尾部槽数），**不是同一量的两种写法**。

### C3. F5 的 A/B 交点位置在源文件与权威文档间**一致**，但**含义**不同
- 源文件（`mrrope:410-414`）把 A/B 定义为"**两张实际表**的逐个交点"，并强调"分界是由两张实际表的交点确定的，不是再挑一个 32K/128K 周期阈值"。
- `INTEGRATION:24` 把它当作"交点位置必须从 KKT 条件的 S-依赖重新推导（K1）"的**事实锚点**。
- 一致，无冲突；但 `NEXT_DERIVATION:105` 进一步把它升格为"三段结构获得真实代码证据"——**该升格在源文件里没有依据**（源文件只说这是两表交点，未说三段）。标记为[假设]。

### C4. OLMo 两个面板口径并存（数字不可混算）
- `Theory:163`/`Nongeometric:111-115`：**独立 seed 72 条**，BM 81.81%/51.32%，MrPro 37.85%/2.78%。
- `INTEGRATION:52`：**16K 350 条**，BM 41.67% vs MrPro 7.09%（+34.59pp，156W/9L），multikey_3 双方皆 0，EOS BM119/MrPro197。
- 两者**样本量、任务数、评分口径均不同**；`INTEGRATION:52` 已明确"评分口径不得混用"。

### C5. "1.485B" vs "1B" 命名
- `Nongeometric:105`、`Theory:163` 均写"OLMo-2-0425-1B-Instruct（**实际 1.485B 参数**）"；`INTEGRATION:102`（FLAG-4）规定规范记录为 **1.485B**。五份材料一致，无冲突，此处仅登记规范值。

### C6. 提问包的目标陈述 vs 权威问题重构
- 源材料（`Core_Problem:5`、`Theory:7`）的目标是"**从 EVQ 出发**推导更好的规则"。
- `NEXT_DERIVATION:11` 的目标是"**我们根本不是搬运**…RoPE 这个非均匀离散傅里叶变换是否有最优解，这也是一个 KKT 问题"。
- **不冲突但重心不同**：源材料把 EVQ 当**起点**，权威文档把 EVQ 当**可重学极限下的一个特例**（`NEXT_DERIVATION:101`）。两处对 EVQ 的"地位"表述强度不同，落 F 时须择一。
- 另：`Core_Problem:65` 自己已经预警——"若沿用完全相同的目标与约束，唯一最优性当然不能被另一条手工曲线击败；改进必须明确来自更准确的目标"——与 `NEXT_DERIVATION:106`（同一泛函族不同边界条件）一致。

### C7. 1/32 圈边界
- `Nongeometric:196` 提问"为什么是三段、**为什么 1/32 圈能作为边界**？"——**该常量在五份材料中均无推导来源**（MrPro 与 YaRN 原文均未给出，`mrrope:45`/`STARTING_POINT:27` 已注明二次形式是设计假设）。**登记为未解，不得当作已证事实引用。**

### C8. 离散约定（分位点）
- `Core_Problem:67`："\(k/K\) 与 \((k+1/2)/K\)、是否固定实际最低频率，并不等价。"
- `GLM_6PRO_REVIEW:51`："换成 midpoint 再锚定端点会产生略不同数字，**不能混用离散约定**。"
- `pro6:248` 的数值例子（0.10483/0.39124）与 `GLM_6PRO_REVIEW:51` 的 .104825834/.391244468 一致，说明 6Pro 的例子**用的是同一约定**；但材料未显式声明是 \(k/K\) 还是 \((k+1/2)/K\)。✅ 数值吻合，⚠️ 约定未书面化。

---

## 9. 未解问题（材料自述 + 本次观察）

1. **L_far 的可计算形式**：`NEXT_DERIVATION:51` 明确"L_far 不能按单槽弧语言定义，须按联合谱覆盖/风险带定义——这是下一次推导要闭合的核心建模步骤（Q9）"。[部分证据]
2. **EVQ 方向与冻结面板方向相反**：`pro6:250-254` 明确 EVQ "让高频侧采样更密"（数值：首间隔 0.10483 < 等距 0.21587）；而 `NEXT_DERIVATION:41` 的 I1（m=0，j≤23）与 HighGapToLong（−17.1/−10.8）表明**冻结态下从 bank 抽预算是死路**。`NEXT_DERIVATION:101` 用"可重学极限"解释这一反差，但**该极限与冻结态之间的桥（何时可迁移）尚未定量**。[假设]
3. **中频为什么关键、为什么是三段**：`Nongeometric:196` 提问未获答；`NEXT_DERIVATION:57` 的"三段 = KKT 解的结构形式"是**待证明的预言**，不是结论。
4. **s28 单槽成功是否为"过渡结构规律"**：`Nongeometric:200` 三选一（过渡区形状规律 / 任何近阈值输出都可能碰巧赢 / 特定 slot 有学习功能）**未决**；`Theory:185` 的"值互换失败、原点平移翻转"使结论必须保守。
5. **跨模型该运送什么**：`Nongeometric:212` 列出六个候选（绝对 slot / 5/17 归一化位置 / \(W\omega_j\) / 周期数 / 训练分布统计 / 重新校准的功能坐标），**未决**；当前 5/17 规则在 Qwen7B 长端 −1.667pp、OLMo 长端 +3.819pp（弱于该模型 BM 的 +34.59pp）。
6. **"1/32 圈"常量来源**（§8-C7）。
7. **增益与相位分离的四格对照**：`Nongeometric:208` 指出"补 MrPro×.074 可闭合 2×2"，**尚未做**。
8. **6Pro 校准方案的最终地位**：`pro6:578` 自述风险（保存的 Q/K 不随新表改变）；`ROPE_GLM_6PRO_REVIEW:52-55` 确认"仍是待检验的新规则生成方法"，且"旧 `calibration/` … 不能被称为 6Pro 所需的 Native 完整行数据"。[假设]
9. **E9/E10 的判读**：仅 12 条初筛持平，"不能称完整失败"（`Theory:159`）——需要什么样本量才能关闭，材料未给。
10. **F 的可识别性**：`NEXT_DERIVATION:121`（K2）提出"若 14 点不能同时被任何 (L_near, L_far) 单调参数族排序（不可识别性），如实报告并给出所需的最小新实验集"——**这是本次五份材料未覆盖的判断，属权威文档层**。

---

## 10. 跳过的内容 / 未做

- **未复算**：本 digest 中标注 [已验证-*] 的数字**绝大多数直接采信材料自述**，唯一由我独立重算的是 §2.2-G3 的两个量（\(\sum_j m_j=29.3333\)、槽位质心 34.6667）与 §2.2-G1 的 5.0560 自洽性检查。其余（4.4118%、0.4841、η 表、周期表、面板百分比、OLMo 72 条）**均未重算**——它们的一手复算在 `digests/digest_mrrope-evq.md`、`tables/ground_truth_tables.json` 与 `digests/digest_panel-results.md` 中，本次未读。
- **未读**：`digests_codex/`（7 份）、`analysis/kkt_20260910/mine/extracts/` 的 codex rollout 抽取、`.agents/rope_unification_20260910/` 原报告、`~/.codex/attachments/6f22c629-.../pasted-text.txt`（F1–F9 原文，`STARTING_POINT:5` 称其为 (a) 源）。**因此本文 F1–F9 的"原始出处"只到 `source_inputs/mrrope_yarn_user_analysis.md` 这一层，未回溯到 codex 粘贴原件。**
- **未验证**：`sandbox:/mnt/data/mrrope_research/mrrope_yarn_verified_analysis.{py,json}` 不可访问（`mrrope:485`、`STARTING_POINT:112`），其中承载的逐槽频率表与 A/B 逐槽交点**无法本地复算**。
- **未做**：任何 GPU/CPU 复算、任何新表的构造或评测。本文不含任何新增数字。
