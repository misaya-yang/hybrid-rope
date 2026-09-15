# ICLR 2027 深度优化计划：RoPE 内部频率分布的结构性设计

更新：2026-09-15。交付内容：问题定义、论证结构、可替换文案、逐文件修改计划与实验优先级。**本次修订计划，尚未覆盖论文源码或改变模型实验队列。**

## 当前适用范围：已完成必要性复核

[现状与修改必要性审计](audits/STATE_AND_REVISION_NECESSITY_AUDIT_20260915.md)对同一R07最终稿
重新区分了缺陷与可选优化。**本文的整套叙事重排、补图和新增实验建议不再作为现稿必改清单。**
现稿已具备受控识别、位置几何、解析构造和任务验证的完整论证链；其中许多建议属于写法偏好，
不足以判定原文有错。当前只保留引言问题陈述的局部增强供作者考虑，具体取舍以必要性审计为准。

新实验可增强未来结论，但不倒置为承认现有结果的前提；实际研究与执行安排沿用
[当前研究索引](../../docs/research/next_stage_20260912/index.md)。本文件后文保留为方案推演，
其中“先做/优先/必须”等措辞不构成新的执行指令。Native-Z5已有独立的初步结果与失败follow-up，
“新z后续再做”指新的研究路线，不能解读为仓库从未测试过native z。

## 1. 本轮重新确定的主线

### 1.1 论文重点解决什么问题

**RoPE 的频率表如何形成有限窗口中的位置基，以及如何据此分析、构造内部频率分布。**

更具体地说：

> RoPE 用有限个旋转通道提供不同频率的位置函数。然而，频率不同并不保证它们在有限上下文中提供不同的位置方向；调整频率还会改变预训练权重已经学会使用的位置基。因此，确定频率覆盖范围之后，仍有一个实质性的设计问题：内部频率应该怎样配置，才能形成适合目标范围和运行条件的位置表示？

这个问题由稿件已有的结构发现、受控实验与明确构造共同回答。**提高给定上下文窗口内的使用质量，是解决这一设计问题所取得的核心实证贡献。**

我此前两种表述各缺了一层：“allocation 有用”只陈述变量的作用；“提高窗口内质量”只陈述希望取得的效果。本轮用一个具体的表示设计问题组织全文。

### 1.2 科学问题、方法和贡献的分工

| 层级 | 本稿应表达的内容 |
|---|---|
| 科学问题 | 内部频率分布怎样形成有限窗口的位置结构，怎样设计这一分布 |
| 已有结构发现 | 不同频率可共享近似位置方向；完整 sin–cos 对与 cosine-only 指标可以给出不同几何排序 |
| 已有识别证据 | 固定实际端点仍可通过内部配置改善模型表现；范围策略与学得使用方式影响结果 |
| 主要方法 | TailSpline：在冻结扩展的外带条件下，以明确的尾部过渡目标构造频率表 |
| 核心实证贡献 | 同一张静态表在已测中间长度与目标长度取得明确任务收益 |
| 辅助实例 | Cosh 是为外推提出的频率搬运方式，正文占比最多约 20%；BM 只提供必要的对照和支持结果 |
| 后续研究 | 新 z 与 s=1 原生质量改善留到下一阶段；稀疏/线性架构的位置机制不扩成当前论文的新主张 |

保留标题 **Beyond the Base: Frequency Allocation in RoPE**。当前最需要调整的是标题之下的问题定义及论证，暂不再推荐以“better context utilization”直接替换论文身份。

### 1.3 这条主线的论证范围

频率重叠是结构事实，不能直接将低频称为“浪费的通道”：慢频也可能被模型用于内容处理。学得兼容性说明冻结安装需要尊重已有使用方式，但不单独承担一条机制论文主线。

完整对几何、TailSpline 的过渡目标与任务收益，各回答一个明确问题。它们构成分析、构造与验证的研究链；当前稿件不需要证明一个几何分数可以统一排序所有任务，也不应把不同构造写成同一全局最优原则的解。

## 2. 本轮依据：已读的版本、任务与作者纠正

审读了指定任务 **「升级论文并安排后续试验」** 中的作者要求、R03–R07 的分析与处置，以及当前源码。重点核实了作者的三项纠正：

- Cosh 的出发点是外推，是改变 z 的一种搬运方式，最多占正文约 20%。
- 已完成的 Llama clean 16K/32K 结果应充分呈现；准确写出设置与结果，不添加泛泛的自我降格式总结。
- 新 z 后续再做；给定窗口的质量改善应作为核心贡献，论文主线须回答更具体的科学问题。

最终依据更新为 **R07 完成修改后的稿件**：
[论文 PDF](../main.pdf)，63 页，科学正文 9 页；SHA256 为 649fdb68e7b39d2bcbd986db5263c2fda4cc8c10326a3377d4600faf59a2058f。本轮先检查 R05/R06，再读取 R07 最终处置、相关源码，并渲染检查最终稿第 1、4、5、7、8 页。之前的 62 页输入不再作为本计划的最新稿件状态。

已结合 [R05 处置](pdf-review-rounds/20260915_astra_sol_five_rounds/r05/disposition.md)、[R06 处置](pdf-review-rounds/20260915_astra_sol_five_rounds/r06/disposition.md)、[R07 最终处置](pdf-review-rounds/20260915_astra_sol_five_rounds/r07/disposition.md) 和 [参照稿评价分析](pdf-review-rounds/20260915_astra_sol_five_rounds/reference_single/assessment.md)，没有把旧输入上的问题直接认定为当前稿件仍存在的问题。

模型结论沿用已核对的论文、结果 owner 和便携记录；此前本任务已从 BM 的逐题分数重聚合自然 QA 点估计。本轮没有重跑模型或声称重新评分所有生成文本。

## 3. 同领域论文怎样把“想要的效果”变成科学问题

来源：作者提供的本地 RoPE 论文集。

本轮重点比较以下稿件的问题陈述、Introduction、理论到方法的衔接；没有将目录中的本项目旧稿视为独立外部证据，也没有声称逐篇完整审计所有数学证明。

| 参照 | 它真正抓住的问题 | 理论/方法如何回应 | 对本稿的启发 |
|---|---|---|---|
| MrRoPE，§1、§3.1–3.2 | 既有扩展策略缺少统一描述，中间频段的转换策略需要比较 | mixed-radix 参数化，再提出 uniform/progressive 转换 | 需要明确“描述频率变化”之外，我们新回答了什么 |
| FoPE，摘要、§1及方法动机 | 线性层、非线性和有限训练窗口造成的频谱破坏 | 频域分析对应 Fourier 组合与频率处理 | 先给具体的表示问题，再让方法部件回应它 |
| Round and Round，§1及贡献 | 距离衰减解释不足；不同频率实际上被如何使用 | 理论、模型内部观察与频率修改 | 不能把低频重叠自动等同于模型没有利用价值 |
| Selective RoPE，§1及贡献 | 仅衰减或仅旋转各缺少一类记忆管理能力 | 旋转与衰减的互补分析，引出输入相关旋转 | “更好 recall”是结果，缺失的具体能力才是问题 |
| Wavelet-based PE，§1、§3 | 作者从尺度使用与感受野约束切入分析既有编码 | 引入不同尺度的位置表示 | 多尺度应对应明确的表示结构，而不是只列更多长度 |
| STRING / GRAPE，§1及主要构造 | 怎样推广位置算子，同时保留相对性等结构性质 | 群与算子表征、可计算构造 | 我们保持标准 rotary 算子；频率内部配置是本稿对象 |
| Deconstructing Positional Information，§1 | 内容与位置如何在 attention logits 中耦合，如何影响学习 | 计算分解、针对性任务和训练偏置分析 | “模型如何使用位置”须有自己的证据，不能由位置几何自动推出 |
| PPE，摘要、§1 | token 合并丢失原有时空位置关系 | 保存多个原位置的编码方式 | 强主线应准确说出哪个对象、哪种结构出了问题 |

所用本地文件分别为 5551_MrRoPE_Mixed_radix_Rotary.md、11_ICML2025_Fourier_Position_Embedding.md、01_ICLR2025_Round_and_Round_We_Go.md、21436_Selective_Rotary_Positio.md、02_ICLR2025_Wavelet_Positional_Representation.md、12_ICML2025_Learning_RoPEs_STRING.md、20573_Group_Representational_P.md、3159_Deconstructing_Positional.md、10300_PPE_Positional_Preservat.md。

主要公开来源：[MrRoPE](https://arxiv.org/abs/2601.22181)、[FoPE](https://proceedings.mlr.press/v267/hua25b.html)、[Round and Round](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e6d58fc68c0f3c36ae6e0e64478a69c0-Abstract-Conference.html)、[Selective RoPE](https://arxiv.org/abs/2511.17388)、[Wavelet-based PE](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c131c8875c7b1133ffdad2b53cb10e91-Abstract-Conference.html)、[STRING](https://proceedings.mlr.press/v267/schenck25a.html)。

这些论文并不共同证明某一个科学立场。值得学习的是它们把“要改进的效果”落实成可分析的结构问题，并用对应实验评价自己的回答。

## 4. 我们相对已有工作的新知识，应怎样说清楚

### 4.1 对 MrRoPE 的定位

MrRoPE 的 λ 已逐维描述频率变化。我们的 x=a+Rz 分解适合分离控制变量，不能单靠这个分解宣称提出了更大的表达空间。

R07 已明确承认 radix products 与 cumulative displacement 的坐标等价，应保留。本稿的推进具体落在以下三点：

1. **从频率覆盖到有限窗位置结构。** 频率范围相同，完整 rotary pairs 的函数空间重叠仍可不同。给出精确对象、实际网格数值与显式反例。
2. **从整体表比较到内部变化的识别。** 固定实际端点的训练和冻结干预，明确内部配置的独立作用；其他控制回答位移与坐标使用的不同问题。
3. **从指定转换曲线到明确的过渡构造。** TailSpline 用一个可解释的离散目标处理进入低频尾部的过渡，给出有限网格闭式解，并在同场任务上评价完整构造。

这足以构成有辨识度的研究增量，不需要声称首次发现频率可以修改，也不需要把 MrRoPE 叙述为只研究最大长度。它本身已有多长度、多个模型与实际任务评价。

### 4.2 一个必须纠正的归因层级

固定 native 表、外带、端点与 gain 后，总 log 位移

\[
D=\sum_k\log(\omega_k^N/\omega_k')
\]

是内部频率配置的统计量。TailSpline–MrPro 改变 D 与形状，仍是在比较两种完整的内部频率设计。

**因此，这不构成对“内部配置带来方法收益”这一主张的外部混杂。** 若进一步声称“收益由尾部平滑这一个因素导致”，才需要匹配 D 的 C 对照。C 对照有信息量，但不能成为所有 allocation 结论和完整方法结果的先决条件。R07 已把这一点写入变量定义和方法说明，本计划将其列为保留项。

同理，固定 band 和总 increment mass 后，位移与 increment 质心并非两项独立控制。主文保留清楚的区分即可，不必在第一页让读者先理解一套控制术语。

### 4.3 时代背景的合适位置

技术进步使更长上下文更可行，为研究频率表的质量问题提供动机。这可以用 Introduction 中一两句话说明。

全文的新知识来自位置结构、构造和实验；“原生窗口越来越长”本身不是创新。也不需要用产品宣称的窗口推断实际训练长度。稀疏与线性架构仅作背景，不引入未经验证的迁移主张。

## 5. 全文按这一条研究链展开

| 步骤 | 读者的问题 | 用哪个现有结果回答 | 读者应带走什么 |
|---|---|---|---|
| 1. 提出设计对象 | RoPE 频率范围确定后，还剩下什么实质问题？ | 频率分解和固定端点结果 | 内部频率分布对应不同的位置基与模型表现 |
| 2. 分析位置结构 | 不同频率为何可能没有提供不同的位置方向？ | 完整对 overlap、慢频子空间、实际网格与反例 | 频率间隔与有限窗位置结构之间有具体可计算的关系 |
| 3. 明确运行条件 | 一个看起来更分散的位置基，能否直接装进模型？ | 范围策略与 weights-by-table crossing | 学习新表与冻结安装具有不同的使用条件 |
| 4. 给出主要构造 | 怎样设计冻结扩展的内部过渡？ | TailSpline 目标、正增量与有限网格闭式解 | 一个由明确设计偏好产生、可以直接安装的频率表 |
| 5. 展示核心贡献 | 这个构造实际带来什么？ | clean 16K/32K、Native 代价、classic 迁移 | 同一张静态表在已测中间与目标长度提高任务质量 |
| 6. 展示辅助广度 | 研究是否只依赖 TailSpline 这一种形式？ | Cosh 外推搬运及配对学习结果；必要的 BM 支持 | 这条频率设计研究已有其他有用实例 |

兼容性在第 3 步服务方法理解，正文不扩展成新的 head-level 机制主线。Cosh 的构造、学习结果和历史细节合计控制在约 20% 以内，不将它改写成 native 改进或覆盖所有运行设置的统一解。

## 6. 理论与构造的具体优化

### 6.1 把“频率不同”和“位置方向不同”讲给读者看

从一对位置函数开始：

\[
V_\omega=\operatorname{span}\{\cos(\omega\Delta),\sin(\omega\Delta)\}.
\]

直觉是：频率表上的两个数不同，并不意味着它们在实际观察的 Δ 区间里形成容易区分的函数方向。

然后再给 canonical overlap 与有效秩恒等式。正文使用识别网格的已完成数值：b=256、K=32、L=256，最慢 8 对提供 16 个坐标，其 block-whitened 有效秩约 2.11。读者因此能把数学对象与真实实验设置联系起来。

R06 已把 normalized directions 的含义放在公式附近，R07 又将实际网格的完整对几何图移入正文。二者均应保留，不再要求重复增加图或限制段落；下一步是在 Introduction 中让这个结构发现承担明确的问题定义。

### 6.2 两个已有反例各完成一个论证任务

- **cosine-only 与完整对几何排序相反：**说明为什么理论要使用完整 rotary pair。
- **同一组表的几何排序随窗口改变：**说明分析必须声明上下文区间，不能把一个窗口的指标直接移用于全部长度。

这些是位置函数几何的结果。当前摘要中的 “allocation rankings” 建议具体化为 “geometric distinctions missed by cosine-only overlap”，避免被读成已预测任务排名。

现有 C_cos 反例针对特定 overlap 定义，不能直接称它推翻了 MrRoPE 使用的 B_theta 求和界；两者需按各自假设评价。

### 6.3 TailSpline 的理由应落在实际过渡对象上

采用以下顺序：

1. 冻结扩展保留高频外带，低频尾部按 s 缩放。
2. 中间相邻频率的额外 log gap 为 ε_q log s，单位总量分配给整个过渡带。
3. 完全插值尾部的额外 gap 为零，因此尾端接缝是一个明确边界。
4. 对带内 gap 变化与尾端接缝建立离散平方目标。
5. 给出 TailSpline 的唯一正解和静态安装方式。

R06 已正确分开“高频外带固定”与“第一处额外 gap 未受惩罚”，保持这一解释。TailSpline 允许更大的入口变化，换取更平滑的尾部衔接；n=17 时，尾端增量相对 MrPro 为 3/(2n+1)。

这个目标是有明确对象的设计选择。现有完整对几何提供分析视角，native 外带给出安装条件；不将其写成完整对 rank 最大化的必然解。

### 6.4 Cosh 与其他理论的篇幅

Cosh 保留为外推搬运：一句动机、逆 CDF 公式或简洁公式指针、三 seed 的代表结果。完整变分推导和强度分析继续保留附录。

离散核等价 corollary 用来说明不同频谱与固定坐标变换的区别，可缩为主文短段加附录证明。它不承担全稿的主要原创性。

BM 的对称边界与 TailSpline 属于可比较的构造选择。正文按需要用一句话说明；其自然 QA 正结果用于证明研究已有另一实际实例，不另立第三条方法主线。

## 7. 可直接用于下一次改稿的英文文案

这些是建议替换文本，尚未写入 .tex。

### 7.1 研究问题

> How should RoPE's internal frequency distribution be designed when distinct frequencies can provide overlapping positional directions within a finite context, and pretrained models have already learned to use a particular positional basis?

### 7.2 摘要草案，无数字结果

> RoPE represents relative position with a finite set of rotary frequencies, yet different frequencies can supply strongly overlapping positional directions within a finite context. Changing their placement also changes the positional basis used by pretrained weights. We study internal frequency allocation through full sine–cosine geometry, controlled interventions, and explicit constructions. Our analysis characterizes positional overlap, while paired experiments isolate improvements from interior placement at fixed frequency endpoints. For frozen extension, we formulate an objective for connecting native frequencies to an extended low-frequency tail. Its closed-form solution, TailSpline, retains the standard rotary operator and requires neither weight updates nor calibration. A single static table substantially outperforms MrRoPE-Pro on RULER at both intermediate and target extension lengths, with a small observed native-window trade-off. Cosh supplies a complementary extrapolation transport supported by paired learning experiments. Together, these results connect the structure of RoPE's frequency distribution to constructive design and improved quality within a chosen context window.

正式编辑时按第一页版面压缩，保留作者无数字摘要、第一页无图的要求。

### 7.3 Introduction 的前两段

> RoPE encodes relative position through a finite collection of sine–cosine pairs. Its frequencies determine the positional functions available to attention, but frequencies that are distinct on a logarithmic grid can provide nearly overlapping directions over a finite context. The structure of this positional basis depends on the internal frequency distribution as well as its range. Understanding this relationship is important for designing frequency tables that models can use effectively.
>
> Existing scaling methods prescribe how frequencies change when extending a context window. YaRN and MrRoPE provide concrete policies for the intermediate band, with MrRoPE organizing rescaling through mixed-radix conversion. We study the internal distribution as a positional-basis design problem while retaining the standard rotary operator. This requires distinguishing the structure supplied by a frequency table from the way a model has learned to use it. We combine controlled frequency interventions with full sine–cosine analysis, then develop an explicit construction for frozen extension.

这里应配上已在文中使用的 RoPE、YaRN、MrRoPE、Round and Round 等对应引用。不要把前人概括成“只看最大长度”或“只改一个 base”。

第三段：给出固定端点识别与完整对结构的关键发现。第四段：介绍 TailSpline 及 +3.39/+11.72pp 的 clean 结果，用一句话连接 Cosh 外推支持。时代背景最多一两句，不能压过设计问题。

### 7.4 三项贡献

R07 已把贡献改成 range/allocation interaction、positional structure versus learned use、analytic-allocation utility 三项发现，这比之前的 “identify the value” 更明确。可以保留这三项事实，并用同一问题统领。以下是需要进一步统一研究主线时的候选文案，不要求仅为换词再重做一次贡献列表：

1. **Structure and controlled identification of frequency allocation.** We characterize finite-window positional overlap using complete rotary pairs and establish the effect of interior placement through paired interventions at fixed frequency endpoints.
2. **An explicit construction for frozen extension.** We formulate a discrete transition objective for the extended low-frequency tail and derive TailSpline, a positive, closed-form allocation that retains standard rotary computation without weight updates or calibration.
3. **Improved task quality with a single static table.** TailSpline improves clean RULER over MrRoPE-Pro at both intermediate and target extension lengths. Supporting Cosh experiments demonstrate complementary extrapolation gains from another frequency transport.

第一项说明新知道了什么，第二项说明怎样构造，第三项呈现作者强调的核心质量贡献。它们不是三条互相独立的论文方向。

### 7.5 结论应回到问题的回答

> RoPE's frequency range does not determine the finite-window structure of its positional basis. Our analysis and controlled experiments show why internal frequency placement deserves explicit design, while the TailSpline construction demonstrates a practical benefit of doing so: improved task performance at both intermediate and target extension lengths under a single static table.

随后简要总结 Cosh 的辅助外推证据与已测任务取舍。新 z 和 native 改善暂留后续研究，不用它们补足当前贡献列表。

## 8. 已经改好的内容应保留

| 内容 | 核实状态 | 本计划的处置 |
|---|---|---|
| clean 16K/32K 同时进入主文 | R03 后已经完成 | 保留，不再列为待补结果 |
| 主图统一为 clean 两长度 | R05 已完成，已目视检查 | 保留；classic 曲线继续在附录，正文保留迁移结果 |
| TailSpline 在方法和实验中居主要位置 | 已完成 | 继续巩固；不恢复 Cosh/TailSpline 双主线 |
| M4 写明 short-training 与 8.39M tokens | R05 已完成 | 保留，不重跑或继续堆限定 |
| 自然 QA 的接近观测分数进入结论 | 已完成 | 保留对应证据，不加摘要负面清单 |
| coordinate interventions 不再被写成构表算法 | R06 已完成 | 保留 |
| effective rank 的 normalized-basis 含义 | R06 已完成 | 保留 |
| 高频外带与入口 gap 的区别 | R06 已完成 | 保留 |
| 完整对几何可视化进入正文 | R07 已完成，已目视检查 | 保留独立几何图，不再搬进首图重画 |
| 总位移属于 z 的统计量 | R07 已写入定义与方法 | 保留，不再把等位移视为所有配置比较的前提 |
| 详细控制分类移入附录 | R07 已完成 | 保留简洁主文定义与后面的协议表 |
| 三项贡献改成具体发现 | R07 已完成 | 保留事实，补足统领它们的科学问题 |
| 750M continuation 的具体收益恢复正文 | R07 已完成 | 保留预算/监督身份和原生代价，仍属辅助学习证据 |
| 构造图移除同等篇幅的 Cosh 面板 | R07 最终稿已完成 | 保留 TailSpline 主位 |
| 历史 MLA 记录缺失的无依据判断 | 已撤回 | 不恢复，不作为新增实验理由 |

当前仍需改变的是**问题定义及论证顺序**：R07 已有清楚的事实、几何图和正面成果，但引言仍从窗口质量起笔，然后并列介绍这些发现。下一步应将它们组织成对内部频率分布设计问题的回答。

## 9. 逐文件修改方案与九页安排

### 9.1 文件级动作

| 文件 | 下一步具体修改 | 完成标准 |
|---|---|---|
| [摘要](../sections/00_abstract.tex) | 从位置基的结构问题起笔；保留 TailSpline 主要结果；具体化 geometric ranking；Cosh 一句 | 读者能指出研究问题、构造和收益，而不只读到 allocation 有用 |
| [Introduction](../sections/01_intro.tex) | 按第 7 节重写问题与发现之间的逻辑；保留 R07 已准确写出的贡献事实 | 前两段建立具体科学问题，末段呈现实际成果 |
| [变量定义](../sections/02_exponents.tex) | 保留 R07 的简洁定义、D 属于 z 的说明与附录指针；明确上下文窗口和频率支持是不同对象 | 不大范围改记号，不再重复移动已压缩的推导 |
| [受控发现](../sections/03_findings.tex) | 固定端点识别保留；范围策略与兼容性用简洁段落说明使用条件 | 识别与方法动机连起来，兼容性不抢占主要篇幅 |
| [理论](../sections/03_theory.tex) | 保留 R07 新增主文几何图；明确实际网格与两个反例各自回答的设计问题；按需要压缩核等价 | 理论回答“表提供怎样的位置结构”，而不是再增加一轮公式 |
| [构造总节](../sections/04_construction.tex) | TailSpline 的过渡问题是主体；Cosh 压缩为辅助外推搬运 | 主文 Cosh 总体占比不超过约 20% |
| [TailSpline 方法](../sections/04_mature.tex) | 按 gap 对象、尾端边界、目标、解、安装五步展开 | 边界偏好说得清楚，完整方法效果与细形状归因分清 |
| [实验](../sections/04_experiments.tex) | 按研究问题组织；保留 clean 主图与 R07 已恢复的 750M 数据；必要时加入 BM 的简短支持 | 主要结果、取舍与辅助实例有清楚层级 |
| [相关工作](../sections/02_related.tex) | 增加一个简洁的对象级比较段，说明本稿保持什么、设计什么、新回答什么 | 无需靠“更全面”“理论更多”表达创新性 |
| [结论](../sections/05_discussion.tex) | 回答内部频率设计问题，再总结窗口内质量收益 | 与摘要和贡献一致，不重复 practical dimension 作为全部结论 |

正式实施上述改稿时，同步当前 [主张映射](EXPONENT_CLAIM_EVIDENCE_MAP_20260909.md) 与必要导航；本次计划修订不提前改写这些来源。

### 9.2 九页预算

| 部分 | 目标篇幅 |
|---|---:|
| 摘要、Introduction、贡献 | 1.0 页 |
| 频率分解与受控识别 | 1.5 页 |
| 完整对有限窗口几何 | 1.2 页 |
| TailSpline 主要构造 | 1.3 页 |
| 主要部署结果、原生取舍、任务范围 | 2.3 页 |
| Cosh 辅助构造与外推结果 | 0.8 页 |
| 相关工作与结论 | 0.9 页 |

合计约 9 页。Cosh 在受控识别中的具体说明也计入它的整体篇幅考量；不是只统计独立小节，再把大量 Cosh 历史分散到正文。

### 9.3 最终稿五张主图的职责与最小修改

1. **Figure 1：固定范围识别与学得使用。** 保留现有三个面板。crossing 的横轴可将 Geo/Cosh table 明确写成 Geo/Cosh-derived runtime table，让读者做行内比较；不将其误读成左侧固定训练 support 优势的重复实验。
2. **Figure 2：完整对几何。** R07 已新增，保留。它让“频率不同不等于位置方向分开”有了直观的实际网格证据。新增问题定义应在此前引导读者看这张图。
3. **Figure 3：TailSpline 过渡构造。** R07 已只展示 TailSpline/MrPro 的额外 gap 与尾部连接，保留。必要时微调边界标注，无需为版式再恢复 Cosh 同等面板。
4. **Figure 4：clean 2L/4L 任务收益。** 保留 R05 之后的设计；最多做任务族分组与标签微调。不混入 classic 点，完整负格保留。
5. **Figure 5：Cosh 三 seed 外推支持。** 保持紧凑，完整轨迹和配方继续在附录；不将它升级为另一条同等方法主线。

R07 已把前部详细控制分类移入附录，主文仍有一个协议表，不再重复做“合并两张控制表”的旧任务。当前 clean 表给绝对分数和区间，任务主图展示整体增益与分布，两者职责不同，可以同时保留。

## 10. 实验按它能增强哪一条贡献排序

### 10.1 现有证据已经可以支撑的完整稿

| 科学或方法结论 | 已有证据 |
|---|---|
| 内部配置有可识别的模型作用 | 151.9M 三配对 seed、相同实际端点；对应冻结干预 |
| 有限窗口位置结构不能由频率端点概括 | 完整对几何、实际网格与显式反例 |
| TailSpline 是明确且可安装的构造 | 有限网格证明、现有 CPU 核验与静态实现 |
| 同一表在中间与目标长度改善任务质量 | clean 16K +3.39pp、32K +11.72pp |
| 还有其他实际有效的外推搬运 | Cosh 配对学习结果；BM 等独立实例 |

因此，第一批深度改稿不等待新增 GPU 工作。

### 10.2 最值得新增的直接比较：clean YaRN

它增强主要方法的相对位置，回答 TailSpline 对常用静态策略的竞争力。

建议复用当前 clean 16K 的 50/task 规格，预先选定现有 32K 每任务前 50 条，形成两个长度的三臂共同面板。只新增 YaRN，共 1,300 次生成；T/P 使用完全相同 row IDs 重聚合。原 T/P 32K 的 2,600 对确认结果继续保留。

若预算已经覆盖完整确认，则只新增 YaRN 的 16K 650 条与 32K 2,600 条。两种工作包在观察 YaRN 结果前选定，不按显著性临时扩样本。

复核公开 YaRN 公式和实际安装约定；不能为了同端点而静默改变基线。若联合宣称 TailSpline 优于 MrPro 与 YaRN，使用两个比较对应的同时推断，保留原 T/P 主估计的身份。

### 10.3 clean Native：增强核心质量贡献的取舍说明

如果要收紧同一 S4 表的原生窗口代价，可做 clean 8K、13×50 的 T–原始 Native 比较，共 1,300 次生成。Native 保留原始频率与 gain；T 保留同一 S4 表。

这是扩展部署的原生代价评价，不是新 z 或 s=1 的能力改进实验。样本量本身不保证某个非劣界限；当前没有必要额外制造一个损失阈值作为论文门槛。

### 10.4 第二模型与 128K：分别检验迁移和尺度

- 第二模型 clean 确认：优先复用已有 OLMo 权重，S4、16K、13×50；T/P 共 1,300 次生成，加 YaRN 为 1,950 次。这是已有模型上的协议确认，不称完全盲测新模型。
- S16、128K：复用已准备的两臂 RULER-13×10 和 ProofPile10。它回答与 MrRoPE 公开 Llama 尺度对标的问题；130 条/臂仅是粗效应与可行性证据。
- S16 若进入完整区间评价，必须测同一 S16 表的中间长度，不能借用 S4 的对应点。
- 当前正在进行的任务和资源安排保持其 owner；本计划不因新的排序推断自动打断它们。

### 10.5 自然任务与 E1：按要增强的主张选择

**自然任务：**当前 TailSpline Natural-QA631 主差为 +0.20pp，区间跨零，实际输入约 3.7–16.3K。若要增强目标端的应用贡献，选择一个实际覆盖接近 32K 的自然 QA/RAG 类别；不能把同一短上下文池扩基线误认为补上了该长度。数据、任务与评分须在输出前固定。

**BM 的已有自然结果：**OLMo 长输入 631 题的五任务宏 F1 为 21.62%→25.44%，+3.82pp。正文可用一两句作为另一构造的成功实例，保留它的模型、输入和原统计身份；不借给 TailSpline，也不以此恢复 BM 主线。

**E1：**若加强 TailSpline 专属形状解释，按现有方案补 C 的 batch-1 390 次生成，复用符合原合同的 T。当前 −0.41pp、跨零区间及 runtime 资格如实保留。它决定细形状归因的强度，不决定完整 T/P 方法结果是否存在。

无需同时把所有增强项列为“到 7 分必须完成”的清单。推荐先完成科学叙事改稿和一个直接基线比较，再按当前资源选择 Native 或迁移/尺度证据。新 z、新的大规模训练、边界搜索和机制平台本轮不展开。

## 11. 审稿意见怎样影响这份计划

### 11.1 当前评分能够说明什么

R05 两个模型的四视角均为 7/7/6/7、AC=7；R06 Astra 为 7/6/6/7、Sol 为 7/7/6/7，两个 AC 均为 7。

R07 已完成：Astra 为 7/7/6/7、AC=7；Sol 为 8/8/7/8、AC=8。分数针对 R07 输入，不是最终整合后的新一轮评价。

这些视角分别在同一模型上下文中生成，且 AC 综合分不等于四审算术均分；不能据此说已经证实“四位真实审稿人均分稳定 7”。它们有用之处是定位仍影响判断的理由。

参照 MrRoPE 的单次 Markdown 评价为 4 分，与实际 Oral 结果不一致，且与我们的 PDF 输入格式不同。它反对把模拟分数当作会议等级预测，不证明所有具体意见都错，也不应被拿来设置强制高分提示词。

### 11.2 具体裁决

| 意见 | 本轮判断 | 动作 |
|---|---|---|
| 创新性仍像“频率可修改” | R07 已强化发现和坐标等价说明，科学问题仍可更集中 | 用第 1、3、4 节把新增事实组织成对设计问题的回答 |
| 主图 clean/classic 混用 | R05 已解决 | 保留完成状态 |
| 几何与 TailSpline 目标关系不够清楚 | 有价值，但不是发现数学错误 | 明确结构分析、设计条件、声明目标及任务验证的职责 |
| normalized rank 被理解为信号强度 | R06 已就地说明 | 不重复追加限制 |
| 要求摘要列自然 QA、Native 和 E1 的全部不足 | 主摘要已明确 RULER，结果处已有数据 | 不采纳重复负面清单 |
| 只因未匹配 D 就否定内部配置收益 | 混淆完整配置与单因素机制主张；R07 已纠正 | 保留第 4.2 节的归因层级 |
| 历史 MLA 记录缺失、OLMo EOS 数量倒读 | 已撤回或已判为误读 | 不作为重跑理由 |
| R05 Sol 把 Cosh 强度写作 sqrt(d_head/L_train) | 与当前正文和附录不一致；实际参考是 d_head/sqrt(L_train) | 标明公式误读；其“参考点不等于任务最优”的一般提醒已在稿中说明 |

最后一项已核对 [Cosh 强度说明](../appendix/a1_proofs.tex) 与 [构造主文](../sections/04_construction.tex)。不因评审写成“不支持的主张”就把错误公式或更强假设反写进论文。

后续独立审稿继续使用中性标准；不追加无意义的轮数。论文新颖性需要作者侧结合实际文献核对，PDF-only 审稿主要检验稿件能否独立传达并支撑自己的论证。

## 12. 实施顺序与完成标准

### 第一批：研究问题与文案

同时对齐摘要、Introduction、贡献与结论，确保它们共享一个科学问题。保留 R07 已准确写出的发现，重点重写连接它们的问题陈述。保持 TailSpline 主位、Cosh 辅助外推身份以及新 z 暂缓。

### 第二批：结构与图表

保留 R07 已压缩的控制说明及五张主图，重组理论到方法的衔接，做必要的标注微调。逐段检查每个理论结果是否服务于当前问题；不为“深度优化”重做已经合理的图表。

### 第三批：结果与支持材料

主要实验按识别、构造效果、长度响应与实际取舍呈现。保留有利和不利结果的实际身份，用少量 BM/Cosh 支持说明研究广度。追加结果只有在实际完成、核实后才写入正文。

### 第四批：完成现有构建与核验

沿用现有编译、作图、源码打包与文档检查流程。对改动后的主文逐页检查；数学对象、图注与方法公式对齐，正文不超过九页，作者要求的首页与摘要形式保持。既有来源快照差异不能靠重写 hash 掩盖。

### 完成后的稿件应能清楚回答

1. **问题是什么？** RoPE 内部频率分布如何形成并改进有限窗口的位置表示。
2. **新知道了什么？** 完整对位置结构、固定范围的内部效应，以及模型使用条件的区别。
3. **提出了什么？** 一个目标明确、公式可解、可直接安装的主要频率构造。
4. **带来什么贡献？** 同一静态表在已测中间与目标长度的任务质量收益，并有其他外推搬运实例支持。
5. **哪些仍是下一阶段？** 新 z、原生能力改进与其他注意力架构的位置机制。

**最终取向：用“RoPE 内部频率分布的结构性设计”统领论文，用“目标窗口内的质量提升”呈现它的主要价值。** 这样既能承接作者的研究思想，也能让现有理论、构造和实验分别发挥作用。
