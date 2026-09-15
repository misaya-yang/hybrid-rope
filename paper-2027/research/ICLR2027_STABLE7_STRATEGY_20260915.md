# ICLR 2027：面向四位审稿人平均 7 分的论文升级方案

日期：2026-09-15。状态：**分析与建议，尚未实施改稿或新增模型实验**。

**作者主张对齐更新：** 本方案按作者随后提供的截图调整了主张层级。论文的目标是提高模型在给定目标上下文窗口中的使用质量；内部频率设计是技术途径，固定频率端点是识别控制。第 4 节给出更新后的中心句、贡献、摘要和 Introduction 文案，并区分更一般的原生窗口研究与当前 TailSpline 构造。

审读对象为 [当前论文](../main.pdf) 的 62 页快照，逐页目视检查了前 9 页科学正文，并核对主文源码、相关证明、结果 owner 和已有审稿处置。快照 PDF SHA256 为 `87f79e89835816e8d34c85affade413be423d3d8e707ba8b5e9633e1e9f4fc89`；当时 HEAD 为 `a8e9cb177ecbafd46f54b323282a83fff1a23c1e`，工作树包含正在进行的论文与实验修改。另读取了作者指定任务「接管服务器实验进展」中的近期讨论，以及本地 MrRoPE、Round and Round、FoPE、STRING 稿件的相关部分。本方案评价这一快照；后续审稿和实验完成情况以各自 owner 为准。

本轮直接重聚合了 BM 自然 QA 的已有逐题分数，确认 631 条长输入的五任务宏平均；其他模型结论按已核对的源码、表格和结果报告使用，未声称重新评分所有生成文本。当前执行安排仍由 [本轮决策映射](COMPARATIVE_GAP_AND_DECISION_MAP_20260915.md) 与相应实验 owner 管理。

## 1. 判断：已有资产足以支撑有竞争力的 7 分稿，主要任务是提高贡献的可识别性与比较的完整性

我支持把“四审均分 7”作为这轮的合理目标。现有工作同时有固定端点的三 seed 识别实验、完整旋转对子空间分析、解析构造、8B 冻结模型的大样本任务收益，以及学习期和自然 QA 的补充成果。论文已经有足够的正面科学内容。

目前限制评分的主要因素是：

1. **目标与技术变量的层级不清。** 作者要解决的是给定上下文范围内的质量问题。“Frequency allocation is useful” 把技术变量的存在性放成了目标。论文应先讲频率设计带来的上下文使用质量改善，再以固定范围识别、完整对几何和运行条件说明贡献来源。
2. **理论到方法的关系没有被组织成最有力的论证。** 几何结论、Cosh 密度目标、TailSpline 边界目标各自成立，但它们之间存在动机联系，尚不存在完整的任务最优性推导。正文应解释每一部分解决的问题，而不是让读者自己补出一条并不存在的演绎链。
3. **最强部署结果缺少同协议的第二个主要基线。** Llama clean 的两长度结果很强，但只直接比较 MrRoPE-Pro。补 YaRN 能显著减少“该设置下 MrPro 是否偏弱”的疑问。
4. **部分好资产没有为当前主张发挥作用。** TailSpline 的自然 QA 持平不能代表整个 allocation 研究没有自然任务收益；BM 已有正面实例。另一方面，历史 OLMo 巨大收益也不能替代 TailSpline 的同协议跨模型确认。
5. **主图与主结果没有完全对齐。** clean 主表和 classic 曲线同页并置，增加了理解成本；最有辨识度的完整对几何主要靠公式和附录展示。

推荐路线：**以给定上下文窗口中的质量改进为中心，以 RoPE 内部频率分布为研究对象。TailSpline 承担主要部署结果，Cosh 承担学习期构造，BM 的对称边界实例补充自然任务证据。先重组现有成果，再补少量能改变审稿判断的比较。**

“稳定 7”无法由几次内部评分保证。目标应落实为四类审稿人都能找到支持接收的具体理由，并让至少两类审稿人有理由积极支持。`8/8/6/6` 只是均分 7 的算术示例，不是分数预测，也不假定 2027 官方表单逐一提供这些内部评分档位。[ICLR 2027 审稿指南](https://iclr.cc/Conferences/2027/ReviewerGuidelines)强调新知识与社区价值，并不把 SOTA 作为接收前提。

## 2. 现有成果应承担什么主张

| 资产 | 当前可支持的正面结论 | 在升级稿中的职责 |
|---|---|---|
| 151.9M，固定实际端点，三配对 seed；512/1024/2048 的 Cosh−Geo NLL 为 −0.281/−0.176/−0.146 | 仅改变内部 30 个频率即可改善学习后的外推；收益不由频率范围变化解释 | **核心科学识别**，保留在正文与首张科学图 |
| 完整 sin–cos 对、canonical overlap、有效秩恒等式；识别网格最慢 8 对的 16 坐标仅有约 2.11 的有效秩 | 频率数与有限窗内的有效位置方向数不同；单看 cosine 会遗漏交叉方向 | **核心结构解释**，用一个直观图展示 |
| Llama-3-8B，clean 16K：650 对；86.09% vs 82.71%，+3.39pp，区间 [1.53,5.34] | 同一 S4 表在中间长度有明确任务收益 | **全窗口目标的关键已有证据** |
| Llama-3-8B，clean 32K：2,600 对；68.27% vs 56.54%，+11.72pp，区间 [10.32,13.11]；12/13 任务均值正向 | 在目标长度有强且广于单一任务的配对收益 | **主要部署证据**；数字由未舍入分数计算 |
| OLMo-2-1B classic Full-13 AUC：17.36%→66.60% | TailSpline 在另一模型族的既有协议也有效 | 有价值的跨模型支持；与 clean 分开标示 |
| 432M MLA，三 seed；16K PPL 138.8→95.6；500M tokens | 学习期也能利用非均匀 allocation；实际配方包含 midpoint 支持变化 | **Cosh 学习价值**，不将它混称纯固定端点干预 |
| BM，OLMo 五项长输入自然 QA，631 题；宏 F1 21.62%→25.44%，+3.82pp | 解析 allocation 已有自然生成任务的成功实例 | 正文恢复一个紧凑支持结果；明确是 BM 与 OLMo |
| TailSpline，Llama Natural-QA631：41.08% vs 40.88%，+0.20pp，区间 [−1.53,1.89] | 该自然任务池尚未确认收益；实际输入约 3.7–16.3K | 与 RULER 分开报告；不包装成自然任务优势或等效性 |
| Native 8K：T−Native −2.14pp，区间 [−6.14,1.92]；PPL 相对增加 0.37% | 原生窗口已测，点估计代价较小；任务损失上界仍不精确 | **部署取舍证据**，补 clean Native 比较最有针对性 |

主要来源：[识别实验](../sections/02_identification.tex)、[几何证明](../appendix/a1_proofs.tex)、[Llama 结果 owner](../../docs/research/next_stage_20260912/TAILSPLINE_LLAMA_CLASSIC_RESULT_20260914.md)、[clean 16K 报告](../../experiments/iclr2027_three_track_sprint_20260915/reports/clean16k_tailspline_vs_mrpro.json)、[完整实验主文](../sections/04_experiments.tex)、[BM 自然 QA 原始结果记录](../../docs/research/ROPE_OLMO_BM_FIVE_QA_RESULT_20260908.json)。

BM 的既有区间 [1.32,6.29]pp 来自任务内配对行 bootstrap，表示所选池内的探索性不确定性；不要给它换成 TailSpline Natural-QA 的 source-context cluster 统计身份。本轮从已有逐题分数重算了 BM 点估计，结果一致。两项自然 QA 的模型、表、长度分层与统计单位均应各自保留。

## 3. 对标已录用论文：学习它们组织贡献的方式

### 3.1 MrRoPE 的启示

MrRoPE 是 **ICLR 2026 Oral**，可由 [官方 Oral 列表](https://iclr.cc/virtual/2026/events/oral)确认。对本次 ICLR 2027 投稿而言，它是上一届工作。

它的叙事非常集中：提出 mixed-radix 语言，将已有方法归入该语言，提出 uniform/progressive 两种策略，再用多长度、多个模型和实际应用评价这些策略。[论文](https://arxiv.org/abs/2601.22181)中的 Llama 是 S16、8K→128K；Qwen 是 S4、32K→128K。当前 TailSpline Llama 是 S4、8K→32K，不能直接把不同论文的绝对分数放在同一排行榜里比较。

我们的进一步贡献应明确写成：**通过 RoPE 内部频率设计，提高模型在给定目标窗口中的质量；以固定范围的受控识别和完整旋转对几何解释设计对象，以明确构造及多长度任务结果实现这一目标。**

MrRoPE 的 λ 已经逐维描述频率变换。`x=a+Rz` 很适合控制实验，但单独作为重参数化不足以建立创新性，也不能未经证明称它严格扩展了 MrRoPE 的表达空间。TailSpline 则是一个明确不同的构造：MrPro 的增量随过渡带递增，TailSpline 降低与完全插值低频尾部衔接时的额外 gap 跳变。

**我们应展示这个不同选择为什么值得研究、有什么精确性质、在哪些受控任务中有效。** 接收理由不应建立在“证明更多，因此应比 Oral 高分”上。

### 3.2 其他参照各提供一种写法

| 参照 | 可学习的组织方式 | 对本稿的具体启发 |
|---|---|---|
| [Round and Round](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e6d58fc68c0f3c36ae6e0e64478a69c0-Abstract-Conference.html) | 围绕一个熟悉解释的不足，逐步给出机制观察、分析与改法 | 用“相同频率范围为何仍会产生不同质量”组织发现；具体反例要有明确的被反驳对象 |
| [FoPE](https://proceedings.mlr.press/v267/hua25b.html) | 频域问题、方法部件与实验问题相对应 | 给完整对几何、Cosh 与 TailSpline 各一个明确职责，使读者能追踪每个部件的作用 |
| [STRING](https://proceedings.mlr.press/v267/schenck25a.html) | 将理论假设、保证的性质与目标应用连在一起 | 把有限窗几何、精确核等价与冻结模型表现分层，突出真正新增的结论 |

以上是对这些稿件行文的分析，不是以它们的录用结果推断我们被接收的概率。用户提供的目录也包含本项目旧稿，不能把目录中的每个文件都当作外部已录用证据。

## 4. 核心 claim：为给定上下文窗口设计质量更好的 RoPE

### 4.1 作者主张的准确层级

| 层级 | 应表达的内容 | 在论文中的位置 |
|---|---|---|
| 研究目标 | 提高模型在给定目标上下文窗口中的使用质量 | 标题意图、摘要开头、Introduction 第一段 |
| 技术途径 | 重新设计内部频率分布，使位置表示服务于目标区间与模型使用条件 | 方法与理论主体 |
| 科学识别 | 固定实际频率端点，确认内部设计本身能带来质量改善 | 受控实验与核心发现 |
| 当前实现 | 一张静态 TailSpline 表，在已测中间长度与目标长度取得任务收益；Cosh 提供学习期实例 | 正文贡献与主要结果 |
| 后续方向 | 在原生长度内直接设计 z，研究 s=1 的质量改善 | 简洁的研究展望；完成前不列为实验贡献 |

前版中心句仍先讲“同频率范围下表的表现不同”，把识别手段摆在了作者目标之前。当前应以质量目标起笔，再用这些控制解释改进从哪里来。

### 4.2 推荐中心表述

**中文：**

> 我们通过重新设计 RoPE 的内部频率分布，提高模型在给定目标上下文窗口中的任务表现。面向冻结部署，所提出的构造使用一张静态频率表，在中间长度与目标扩展长度均取得收益，无需更新权重或校准。

**英文：**

> We develop RoPE frequency designs that improve model performance within a prescribed context window. For frozen deployment, our construction uses a single static frequency table to improve task performance at both intermediate and target extension lengths, without weight updates or calibration.

这是一段总括；紧接的结果句应点明 clean RULER 和实际比较对象。当前成果可直接说在两个已测扩展长度取得收益；“对窗口内质量进行设计”是目标，不将它写成窗口内每个长度、每种任务都严格占优。

### 4.3 统一记号，明确评测对象

令原生训练长度为 `L_train`，目标服务上限为 `T = s L_train`。对这一目标设计一张固定表 `Ω_T`，观察其在预先声明长度集上的质量：

\[
\left\{Q(\ell;\Omega_T):\ell\in\mathcal L_T\right\},
\qquad \mathcal L_T\subseteq(0,T].
\]

这里的上下文窗口 `(0,T]` 与频率支持区间 `(a,R)` 是不同对象。正文先定义目标窗口，再介绍频率分解。上式用于说明一张表服务多个长度；沿用已有主终点和统计合同，不由此增加一个事后加权综合分数。

这种表述自然连接原生窗口质量与扩展窗口质量。窗口上限 T 描述任务范围，频率设计的价值由该范围内实际任务表现体现。背景用“原生窗口持续增长”即可；若要写具体模型已达到 200K 或百万长度，应在正式改稿时补对应来源，也不把训练长度、宣称窗口和评测长度混称。

### 4.4 三项贡献的可替换英文

1. **RoPE frequency design for context quality.** We develop explicit frequency allocations for learning and frozen deployment. TailSpline uses a single static table to improve RULER performance at both intermediate and target extension lengths, while Cosh provides complementary gains in learned extrapolation.
2. **Controlled identification of the source of improvement.** Paired training and frozen-model interventions show that interior frequency placement improves performance even when the sampled frequency endpoints are fixed. Additional controls distinguish changes in total displacement from residual shape and learned coordinate assignment.
3. **Finite-window structure and explicit constructions.** Full sine–cosine geometry characterizes the positional overlap of frequency pairs and reveals distinctions missed by cosine-only proxies. We use this structural perspective and the constraints of each operating setting to motivate explicit allocation objectives, with closed-form constructions and stated mathematical guarantees.

三项依次回答：带来什么能力收益、怎样识别改进来源、怎样理解并构造频率表。第三项中的“motivate”描述设计动机，不把完整对子空间指标与 TailSpline/Cosh 的目标函数写成未经建立的等价推导。BM 的自然结果作为有明确身份的补充实例。

### 4.5 建议摘要：保持无数字结果

> The value of a context window depends on how effectively a model uses the information it contains. We develop RoPE frequency designs to improve model quality within a prescribed context window. Controlled interventions show that redistributing interior frequencies improves learned extrapolation and frozen-model performance even when the frequency range is unchanged. Full sine–cosine geometry characterizes finite-window positional overlap, while learned-coordinate effects inform the distinction between learning a new allocation and modifying a pretrained model. For frozen deployment, we introduce TailSpline, a closed-form allocation that smooths the transition into the extended low-frequency tail. It uses a single static table and requires neither weight updates nor calibration. TailSpline substantially outperforms MrRoPE-Pro on RULER at both intermediate and target extension lengths, with a small observed native-window trade-off. Complementary Cosh experiments demonstrate gains when models learn with a redesigned frequency distribution. Together, these results connect RoPE frequency design to improved model quality within the intended context window.

该摘要是建议文案，未覆盖正在进行的论文源文件。它保留当前正面结果与简短的观测代价描述；未完成的 native 改进不出现在摘要。

### 4.6 Introduction 开头的建议文案

> Extending a pretrained model from a native length L to a target length T = sL specifies a range of inputs that the model is expected to serve. The resulting system must perform well at the shorter lengths within that range as well as near its upper limit. We study RoPE frequency design with this objective: improving the quality of context use within a prescribed window under a single static frequency table.
>
> YaRN and MrRoPE demonstrate how frequency rescaling can extend pretrained models beyond their original context lengths. Their rescaling policies also determine how rotary frequencies are distributed within the resulting range. We investigate this internal distribution as a means of improving model quality within the target window. Paired interventions hold the sampled frequency endpoints fixed and establish improvements from changing interior placement, while full sine–cosine analysis characterizes the positional structure of these changes.

第三段直接给出 TailSpline 的明确构造与 clean 2L/4L 收益，再以 Cosh 连接学习期设计。无需把领域描绘成“前人只追求最大长度”；MrRoPE 本身也测了多长度质量。本文进一步研究固定目标窗口中的频率设计、受控归因及解析构造。

准确的模型、数据与长度范围放在实验设置、结果句和表注中。摘要与 Introduction 围绕研究问题、贡献和已完成成果组织，不添加“only one model”之类脱离具体设置的自我降格式结论。实际支持范围仍如实交代。

### 4.7 s=1 的位置：研究空间与具体构造分别表达

作者关于 native 质量的方向成立：目标窗口可以取 `T=L_train`，在相同原生长度研究新的内部频率设计。**按作者最新决定，新 z 留到下一阶段，本次集中优化已完成论文的主张与行文。** 后续可以保持端点和频率跨度，直接重新设计内部坐标，再评价模型如何利用新表。

当前 [TailSpline 安装式](../sections/04_mature.tex) 为 `ω'_k=ω_k^N s^(−m_k)`、`g=1+0.1 ln s`。因此 `s=1` 时频率表和 gain 均退回 Native。原生质量改进须通过更一般的 z 构造研究，不能只把现有 TailSpline 的 s 设为 1。

当前 S4 表对原始 Native 的 8K 比较衡量的是**扩展部署的原生代价**；未来 s=1 新表对 Native 的比较衡量的是**不扩窗口时的能力改进**。两者回答不同问题。

展望中一句即可：

> The frequency-design formulation also accommodates redesign at the native context length, motivating future work on native-window quality.

### 4.8 当前稿件的具体替换点

| 当前位置 | 当前重心 | 建议重心 |
|---|---|---|
| 摘要首句 | RoPE typically ties placement to a single base | 目标窗口中的使用质量，再引出频率设计 |
| Introduction 第一项贡献 | Identify the value of allocation | RoPE frequency design for context quality |
| 理论节与方法节之间 | 重叠量之后直接出现构造 | 分清学习新基与冻结扩展，说明各目标的设计偏好 |
| 实验组织 | 先罗列不同协议 | 同一表在原生、中间、目标长度的任务表现，再展示识别与其他实例 |
| 结论末句 | Allocation is a practical design dimension | 总结明确构造带来的目标窗口质量改进 |

现有标题可以保留。若同步强调目标，推荐候选为 **Beyond the Base: RoPE Frequency Design for Better Context Utilization**；这只是标题候选，优先完成正文的主张对齐。

### 4.9 时代背景：更长上下文的计算可行性与实际使用质量

作者希望研究更符合当前长上下文发展的质量问题。这个动机可以成立，而不依赖“许多稀疏注意力已经不用 RoPE”的概括：

- **稀疏 softmax 的例子：** DeepSeek-V3.2-Exp 的官方 DSA 实现同时在 indexer 与主注意力中使用 RoPE。稀疏 token 选择和旋转位置表示能够共同存在。[官方实现](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py)
- **混合线性架构的例子：** Kimi Linear 明确在全部全局 MLA 层使用 NoPE，将位置与 recency 信息的建模交给 KDA 层。这说明部分模型改变了位置建模的承担方式；NoPE 不等于系统没有位置机制。[技术报告 §4、§6.1](https://arxiv.org/html/2510.26692v1)

因此，背景应提出一个跨架构都有意义的问题：随着更长输入在计算上变得可行，如何提高给定窗口中信息被有效使用的质量？本文选择 RoPE 频率设计这一具体对象，以已完成的结果回答它。稀疏、线性和混合架构提供背景，不成为未经验证的新应用主张。

**可用于 Introduction 的动机句：**

> As longer contexts become computationally accessible, improving how reliably models use information within a target window becomes an increasingly important objective. We address this objective through RoPE frequency design, developing explicit constructions that improve task performance at multiple lengths under a single static frequency table.

这是基于上述架构发展的研究定位判断，不是对所有现代模型瓶颈的统一归因。关于稀疏注意力的后续位置编码研究可以独立开展；当前论文继续用已有的 RoPE 构造、理论和任务证据完成这条质量主线。

## 5. 理论升级：强化已有贡献的解释力，准确连接构造

### 5.1 用完整对几何回答一个可记住的问题

核心问题应是：**两个不同频率，在模型实际服务的窗口里，是否提供不同的位置方向？**

先展示完整 sin–cos 对的图，再引入 canonical overlap 与

\[
r_2(\Gamma)=\frac{2K}{1+(K-1)\bar c}.
\]

读者应先理解“很多频率可能挤在近似同一位置子空间”，再看定义。保留识别网格的数值例子：8 个慢频 pair、16 个坐标、有效秩约 2.11；它没有使用不成立的 `ωL≪1` 近似。

正文写明这是 **block-whitened positional geometry**，刻画方向重叠；它不测量模型已经学到的各方向幅值或内容价值。把它直接叫模型容量或信息量会越出证据。

### 5.2 让两个反例承担积极的科学职责

[现有附录](../appendix/a1_proofs.tex)已经有：

- 相同端点的两张表，cosine-only collision 的优先次序与完整对有效秩相反。
- 完整对 overlap 的排序可以随 L、2L、4L 改变；单窗口最大秩的 parity lattice 还可能在窗口之外重复。

它们分别说明 **为什么必须看完整对**、**为什么必须声明服务区间**。这两点直接服务核心研究问题，不应只作为防御性限制散落在附录。

现有 `C_cos` 反例针对特定 cosine-only overlap 指标，不能直接宣称推翻 MrRoPE 使用的 `B_theta(d)=Σcos(dω)` bound。两者不是同一个量；若将来要批评该 bound，须另行对齐其假设与结论。

### 5.3 把 TailSpline 与 BM 写成同一边界构造的两个已知选择

已有证明可以组织为：相邻频率的额外 log gap 由 `ε_q log s` 决定，尾部完全插值区的额外 gap 为零。考虑

\[
J_{\alpha,\beta}(\epsilon)
=\sum_{q=1}^{n-1}(\epsilon_{q+1}-\epsilon_q)^2
+\alpha\epsilon_1^2+\beta\epsilon_n^2,
\qquad \sum_q\epsilon_q=1.
\]

这里仅用于并列解释已有的两个选择：TailSpline 为 `(α,β)=(0,1)`，BM 为 `(1,1)`。不新增参数搜索，也不声称已找到任务最优 α、β。

这可以同时做到：

1. 保留 TailSpline 的有限网格唯一解、正性、单调累计表和 O(K) 构造。
2. 解释它为何承担更大的高频入口跳变，以获得更小的尾部接缝跳变。
3. 将 BM 的自然任务结果接回同一数学构造，而不另开一条同等规模的方法主线。

对 Llama 的 n=17，直接由现有公式计算：TailSpline 的入口增量约 0.0857，MrPro 约 0.00654；尾端增量约 0.00952 与 0.11111。中间 `Σm_q` 分别约 9.943 与 5.333。**这些数字直观展示收益比较同时改变了位移总量与分配形状。** 它们是构造性质，不是胜因证明。

### 5.4 明确两种理论价值

- 完整对几何说明 allocation 改变的位置结构。
- Cosh/TailSpline 的变分或离散优化说明怎样从明确设计偏好得到可安装的解析表。

当前 Cosh 泛函不是完整 canonical-overlap 目标的精确推导，TailSpline 边界目标也不是它的离散化。将二者写成“几何与运行条件启发的明确构造，再由任务实验评价”，足以形成严肃贡献。

整数谱等价 corollary 保留一段简明结论即可；它解决精确位置核能否被固定坐标变换吸收的问题，经典谱相似性工具本身不宜承担主要新颖性。

### 5.5 有价值但不应升级成主证据的旧资产

[FullLagP2](../appendix/a9_recovered_design_evidence.tex)确有从完整对残差构造频率表的正面小面板例子，因此“几何从未进入实际构造”过于绝对。但其跨模型/任务结果混合、面板小，适合在讨论中用一句话指向完整附录。它目前不能解决 TailSpline 专属胜因问题。

**为了当前目标，不建议新开一个 checkpoint 校准器、拟合任务排名公式或普适最优表理论。** 构造继续只用公开 RoPE 参数；现有权重干预用于理解，不转作选表过程。

## 6. 九页正文和图表应怎样重排

### 6.1 先解决三处具体阅读摩擦

**第 7 页 Table 2 与 Figure 3(b)。** 主表给出 clean 16K/32K 的 +3.39/+11.72pp，图 3(b) 展示的是 classic Llama 曲线，其中 32K 仅约 +2.63pp。两者都有正确协议标记，但读者仍需自行解释差异。优先把主曲线换成 clean 两长度的绝对分数与配对差；classic 全曲线保留在单独支持面板或附录。未测的 clean 8K 位置不以 classic 点补齐。

**第 3 页 Figure 1(c)。** 该 crossing 使用另行派生的 factor-four runtime tables；两个匹配对角值是 3.426 与 3.479。它说明每套权重偏好对应表，不是 Figure 1(b) 的固定训练 support 优势复现。主图直接标注 `derived runtime tables; compare within each row`，用行内箭头表达结论。也可移到紧随首图的紧凑表，避免首个视觉同时承担两个容易混淆的排名任务。

**第 2 页控制表与第 6 页 Table 1。** 两处都在分类协议。合并成一个紧凑的控制表，省出几何示意与 BM 自然结果的位置。正文以科学问题和结果为段落主语，详细 runtime 身份集中在实验设置与附录。

### 6.2 建议的篇幅预算

| 部分 | 约页数 | 应让读者记住什么 |
|---|---:|---|
| 摘要、Introduction、贡献 | 1.0 | 同范围仍有可利用的内部设计问题；本文已给出实证方案 |
| 分解、控制识别、训练/冻结连接 | 1.6 | 改变的究竟是什么，哪些实验已排除了范围解释 |
| 完整对几何与必要反例 | 1.2 | 频率分开不代表有限窗位置方向分开 |
| TailSpline、BM 边界特例、Cosh | 1.3 | 方法公式、设计偏好、安装方式和明确性质 |
| 主要部署结果、成本、学习与自然任务支持 | 2.8 | 同一表的中间/目标长度收益；完整取舍 |
| 相关工作与结论 | 1.1 | 相对已有工作的知识增量与适用范围 |

合计约九页，是改稿预算，不是已经编译完成的排版。保持作者“第一页无图”的选择；参考文献和补充材料按 [ICLR 2027 作者指南](https://iclr.cc/Conferences/2027/AuthorGuidelines)放在正文之外。

### 6.3 图表按贡献组织

- **首张科学图：固定范围识别。** 频率位置示意、三 seed 效果；兼容性面板只表达行内比较。完整几何图可以作为紧邻的小面板或第二张图，优先复用已有 `fig_allocation_geometry`。
- **构造图：让读者看见边界选择。** 同一横轴显示 MrPro/TailSpline 的累计位移与增量；用少量标注显示两个接缝。BM 用明确的对称边界说明，避免突然出现第三套无解释的方法名。
- **主要部署图：clean 结果优先。** 16K/32K 的绝对性能与配对差；13 任务按 retrieval/tracking/aggregation/QA 分组，保留 multivalue 负格。第二模型若仍是 classic，显著标示，不与 clean 拼成同一种确认。
- **学习证据：保留三 seed 的可见性。** MLA 曲线或足够清楚的主表承担这一职责，完整训练进度留附录。不能只剩一句“更多实验见附录”。
- **结果汇总表：每行写明方法与模型。** TailSpline clean 16K/32K、原生代价、TailSpline 自然 QA、BM–OLMo 自然 QA、Cosh–MLA 各有独立行；不同指标不做跨行总平均。表太长时拆为“主要部署”和“学习/自然任务支持”两组。

BM 的正文恢复只需一个简短段落或表格行。Qwen 上的对应不利结果继续在同一证据指针下可见；不能只取 OLMo 胜格宣称 BM 普遍优越。

附录按证明、复现协议、完整结果、支持性实验导航即可。每个主结论在正文中应已有足以形成判断的信息；62 页材料的价值在于可追踪，不能要求审稿人通读才能发现最强理由。

## 7. 实验优先级：每项只解决一个足以影响接收判断的问题

下表是新增工作的建议，不改变当前 GPU 队列。样本量是可审阅的工作包；最终执行前沿用已有实验 owner 的预算和输入身份。

### P0：立即利用现有结果，无需 GPU

1. 重画 clean 主图；把 classic 与 clean 的数字对应关系讲清楚。
2. 正文恢复 BM 自然 QA 结果，保留独立统计身份；完善 allocation 总主张与 TailSpline 子主张的关系。
3. 把几何反例、Cosh/TailSpline 的理论职责、BM 的边界关系写清楚。
4. 复用现有 ProofPile-only 报告：其三长度 AUC 几乎持平，不能写成 PPL 全面领先。
5. 复用已经安排的 NIAH 热图。它增加长度/深度可读性，但与 RULER 检索覆盖重叠，新增样本的边际价值低于下面的直接比较。

### P1-A：在当前 clean Llama 输入上补 YaRN

**消除的疑问：** TailSpline 的优势是否仅因为 MrPro 在当前 S4 设置下不是强选择？

推荐完整方案：只增加 YaRN 一臂，32K 2,600 条，16K 650 条，共 **3,250 次新生成**；复用已完成的 T/P。使用已核对的公开 YaRN 公式、共同 decoder 与约定 gain；明确报告其实际频率端点是否与其他两臂相同。不能为形式上的“同端点”而静默改造 YaRN；受控变体须另起标签。

预算较紧时：预先固定使用当前每任务前 50 条，形成 16K/32K 各 650 条的 **三臂共同面板**，只新增 1,300 次 YaRN 生成；T/P 在完全相同的 row IDs 上重新聚合。原 T/P 32K 的 2,600 条确认结果仍保留，不把 YaRN 650 条分数混装成同样样本量的比较。

采用哪一个工作包应在查看 YaRN 输出前决定。若使用两项主比较 T>P 与 T>Y，保留既有 T/P 主估计，并为新增的联合优越性陈述计算同时区间。

结果解释：T 同时优于两者，部署价值更清楚；T 仅胜 P，就据实写出相对位置并考虑 S16 同场测试的解释价值。不能由 `T>P` 和 MrRoPE 论文中的 `P>Y` 推出当前 `T>Y`。

### P1-B：用 clean 原生窗口对照收紧代价判断

**消除的疑问：** 同一静态表的长端收益，会不会伴随不可接受的常用长度损失？

建议一个固定的 8K、13×50 source-order 面板，运行 **TailSpline + 原始 Native**，共 1,300 次新生成。Native 使用原始频率与原始 gain；TailSpline 仍安装同一 S4 表。现有 padded 130 条是有效历史证据，但不混入这个 clean 估计。

报告绝对分数、任务等权差和配对区间，并复用已完成的 Native PPL。若另有预算，可加 MrPro 8K 以形成统一 clean 方法曲线；它不是 T–Native 代价比较的必要条件。

50/task 不是保证非劣性的样本量。如果确实需要“损失不超过 δ”的声明，先给出应用上有意义的 δ 并据已有配对方差确定固定样本量；不能按结果反推界限。若仅报告代价，不额外创造非劣门槛。

### P1-C：第二模型的 clean 确认

**消除的疑问：** 最大样本的结果能否离开单个 checkpoint 和单一协议？

优先使用已有 OLMo-2-1B 权重，S4、16K、13×50 source-order 输入，原封不动使用公开参数算出的 canonical 表。T/P 两臂共 1,300 次生成；加 YaRN 为 1,950 次。它成本低于另开一个 8B 长窗口全套，又能把当前 classic OLMo 结果转为同协议支持。

这是对已有模型的协议确认，不称全新盲测模型，因为历史 OLMo 结果已参与方法发展。如果目标进一步提高跨架构广度，现有 Qwen 模型也可作为后续检验，但应事先固定模型、尺度与表，不以小样本结果挑选“能赢”的 checkpoint。

**P1 内的建议顺序：** 同场 YaRN → clean Native → 第二模型 clean。可按机器实际吞吐并行准备；不打断已经接近完成的授权任务。

### P2-A：S16、128K 用来回答与 MrRoPE 同尺度的竞争力

作者指定任务中已有 S16 TailSpline/MrPro 表、128K Full-13 的 130 条/臂和 ProofPile10 的准备回执；应复用。这些是准备状态，不是已完成性能结果。

128K 很有价值：它检验方法能否在 MrRoPE 公开 Llama 尺度下仍有竞争力。对“四审均分 7”的当前计划，它属于提升说服力的补充，不替代 S4 的基线与原生代价比较。

- 先完成已准备的固定工作包：两臂共 260 次 RULER 生成，另有两臂 PPL10。
- 13×10 是尺度可行性与粗效应测试，不能冒称与当前 clean 32K 的证据强度相同。
- 若希望用它支撑完整 S16 窗口结论，需要同一 S16 表的中间长度结果。先看极限点以决定资源投入是合理的；后续确认仍须使用事先固定的任务和样本，报告完整结果。
- S4 与 S16 是两张不同的静态表。S4 的 Native/16K 表现不能直接替代 S16 的对应点。
- 即使 128K 失利，S4 两长度收益仍成立；失败界定更大倍率的适用边界，不据此抹掉现有研究结果。

### P2-B：如果要强化 TailSpline 的应用收益，选一个真正覆盖扩展区间的自然任务

现有 Natural-QA631 的最长输入约 16.3K，不能检验 32K 的自然任务效果。给同一批短上下文再加基线有比较价值，但不能填上目标区间的应用证据。

建议后续只选一个清楚的应用类别，例如 [HELMET](https://princeton-nlp.github.io/HELMET/) 的 RAG 类别，采用官方任务、提示与评分，并在 Llama tokenizer 下确认实际长度。该基准提供受控长输入与多种应用类别，且明确指出简单 NIAH 不足以预测下游表现。

具体数据量、来源、生成上限和评估成本应在使用之前固定；这需要单独的数据/执行安排。若只是一个类别或其中一个数据集，报告相应名称，不能称 HELMET 总分。不要填充短文冒充天然 32K 长文，也不要在观察模型结果后挑选能产生正差的任务。

这一项若取得收益，会提高应用审稿人的支持力度；若持平，仍保留完整结果，主张集中在已测的检索/跟踪/聚合能力。现有 BM 自然成功实例已经使“allocation 从未改善自然任务”不成立。

### P2-C：E1 同位移对照有信息量，但不应成为论文停工条件

当前 T–C Full-13 AUC 为 −0.41pp，区间 [−2.63,1.82]，且 batch 1/2 不同。优先级低于补主比较基线。

若确需加强 TailSpline 专属边界解释，按既有方案只补 C 的 batch-1 390 次生成，复用匹配条件下的 T；LM 的 138 行仅在该诊断需要时补。使用原来的 Full-13 主终点，不根据 NIAH 子集正差更换终点。

- T>C：支持这组固定对照中的残余形状价值；仍不自动建立 tail-only 中介机制。
- 仍跨零：保留 TailSpline 是一个有效闭式完整构造的结论，也承认另一种同位移形状同样有竞争力。
- C>T：把它作为特定平滑偏好未获支持的结果；整个 allocation 命题与 T/P 方法收益不因此失效。

不建议当前启动大规模 head patching、边界搜索、额外 gain 网格或新预训练。它们服务的是更强机制/普适方法研究，尚非当前稳定接收路线中收益最高的缺口。

## 8. 针对四类审稿人的接收理由

| 审稿视角 | 应给他的主要正面理由 | 最容易压分的疑问 | 对应动作 |
|---|---|---|---|
| 理论与表示 | 一个定义精确的有限窗位置对象、非平凡反例、解析可安装构造 | 经典恒等式是否被包装成新理论？几何是否被误写成任务代理？ | 区分经典工具与 RoPE 新结论，展示完整对例子与边界构造 |
| RoPE/长上下文领域 | 沿着 YaRN/MrRoPE 的问题继续推进，提出不同的分配选择并取得强同场收益 | λ 已可逐维调整，z 到底新增什么？MrPro 是否偏弱？ | 强调控制识别；clean YaRN；有余力完成 S16 |
| 实验与复现 | 三 seed 识别、大样本 paired clean 结果、公开公式、完整负格 | 单模型、协议差异、原生损失估计宽 | clean Native 和第二模型；统一主表，保留结果身份 |
| 一般 ML/应用 | 读完前两页能解释新知识和实际收益，能看懂一张表的取舍 | 论文似乎只是多个小实验；自然任务是否完全没有收益？ | 单一中心句；TailSpline 主结果；Cosh/BM 有名有姓的支持结果 |

四种视角是本方案的组织工具，不是本轮新生成的四份独立审稿，也不是对真实审稿人构成的预测。

## 9. 如何使用已有审稿，避免越改越保守

已有 R03/R04 的独立模型审稿为 6 与 7，针对的是各自冻结输入。R05 开始一个模型内的四视角模拟，不等于四个独立审稿人；[审稿记录](pdf-review-rounds/20260915_astra_sol_five_rounds/README.md)已经说明这一点。

每条意见按三种情况处理：

1. **真实事实/表述问题：** 如读者混淆 clean/classic，或者方法效果与具体机制被连写。修复准确对应的句子和图表。
2. **值得投入的增量证据：** 同协议 YaRN、原生精度、第二模型确认。写入有边界的工作包。
3. **不属于当前 claim 的更强要求：** 普适任务最优表、纯几何必然预测 F1、所有模型全面胜出。明确科学层次，不把它们升级成接收前提。

重复出现的错误不自动成为事实。此前历史 MLA 记录缺失的判断已被 [正式撤回](pdf-review-rounds/20260915_two_rounds/README.md)；R04 中 OLMo EOS/non-EOS 被倒读的意见也已有 [处置](pdf-review-rounds/20260915_astra_sol_five_rounds/r04/disposition.md)。本方案不把两者作为补实验理由。

下一次重要复评宜在核心改稿和关键比较完成后，使用同一最终 PDF、独立上下文和中性的正式审稿维度，不输入目标分数，不让审稿人先看作者预期或其他评价。比较的是剩余决策性问题及其证据，而不是继续累计模型打出的 7。

## 10. 建议交付顺序与停止条件

### 第一批：完成现有资产即可做到的稿件升级

- 冻结三项核心贡献与主张层级。
- 修改 Introduction 和理论过渡；保留无数字摘要、第一页无图。
- 主图与 clean 结果对齐；恢复 BM 自然 QA 支持；合并重复协议表。
- 用正文回答“新知道了什么、怎样用、在哪些条件有效”。

这批工作不依赖新的 GPU 结果。完成后论文应已经比当前快照更容易被正确评价。

### 第二批：完成最能减少审稿分歧的比较

按 P1 固定选中的工作包执行。新结果不理想时修改结论，不反复换任务、扩样本或调方法直至出现正结果。

理想交付是：clean 两长度上有明确的三臂定位，原生任务代价有清楚的区间，第二模型有同协议支持。它们能直接支撑“质量改进且可部署”的价值判断。

### 第三批：依据已有资源加入尺度或自然应用补充

完成已准备的 S16 测试；若资源还允许，选择一个目标区间的自然应用评价。128K、自然应用、E1 三项各回答不同问题，不以数量多作为完成标准。

官方摘要截止为 **2026-09-18 23:59 AoE**，全文截止为 **2026-09-25 23:59 AoE**。[官方日期与格式](https://iclr.cc/Conferences/2027/AuthorGuidelines)支持这一时间安排。作者此前采用 9 月 17 日的内部目标，可用于先冻结标题、摘要与中心贡献；新结果只影响已明确分工的实验段落。建议 9 月 23 日冻结数字与主图，最后两天完成独立源码构建与最终 PDF 阅读。

### 稿件完成的实质标准

1. **一个中心命题。** 四类读者都能复述：为给定目标窗口设计频率表，以提高实际使用质量；频率分解、理论与受控实验服务于这一目标。
2. **每项贡献都有正文证据。** 识别、几何、构造和任务价值不靠泛泛的附录指针成立。
3. **主方法比较有相对位置。** 同协议常用基线、不同长度与原生代价清楚可查。
4. **正面结论足够明确。** 不把整篇研究弱化成“参数会影响结果”；也不合并不同构造的胜利。
5. **边界已被准确表达。** TailSpline 的自然 QA、E1、不同协议和跨模型反例保留，各自只影响对应主张。

## 11. 最终建议

**以给定上下文窗口中的质量改进作为中心贡献，组织成“目标窗口的质量问题 → 内部频率设计 → 受控识别与完整对结构 → 明确构造及多长度任务收益”的论文。** TailSpline 是最强的当前部署实例；Cosh 与 BM 分别提供学习价值和另一自然任务成功实例。更一般的频率设计也允许研究 s=1 的原生质量，完成前保留为后续方向。

优先重写新知识与结果之间的关系、用 clean 证据重排主图，再补 YaRN、clean Native 和第二模型 clean。128K 与目标区间自然任务用于提高覆盖与说服力。现有理论不必承担普适任务排序器的职责；每一个新增比较都应直接消除一个会影响接收判断的疑问。

这是我认为最有希望把当前“6–7 分波动”推进到“四审平均 7 分目标”的路线。依据是已有贡献的组合及明确可补的缺口，不是把内部模拟评分当作接受概率。

首次交付复核：当时本方案的 13 个仓库内文件链接均可达，最近一级研究索引已加入入口。按维护规则刷新了文件清单并运行文档检查；检查前后均有相同的 10 条来源快照不一致提示（涉及 8 个既有来源），没有新增链接或来源错误。作者主张对齐更新仅修订本方案，增加了一个已存在的 TailSpline 安装式链接；未修改论文源文件、实验结果或执行队列。
