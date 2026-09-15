# RoPE 与长度外推顶会论文：写作、理论、实验与附录综合审读

日期：2026-09-14  
面向稿件：`Beyond the Base: Exponent Allocation in RoPE`

## 1. 审读范围与结论

本轮覆盖作者提供的本地 RoPE 文献集中全部 33 份 Markdown，并用同批配对 PDF 的原始文本交叉核对双栏错序、公式和表格语境。材料共 961 页：其中 32 篇外部论文约 931 页，另 1 篇是本项目已经被当前稿替代的旧稿 `RoPE Has a Spectral Budget`。审读覆盖每篇的摘要、引言、贡献声明、主要理论与方法、主实验、结论/局限以及附录组织；对长证明重点核对命题、假设、证明入口和正文用途，而不是重新逐行证明全部引理。

总判断如下。

1. **当前稿件的科学主线是成立且有竞争力的**：内部指数分配 `z` 是有用的 RoPE 设计变量；固定支持实验识别它在不改变端点时的独立作用；完整方法则可以同时改变支持、band、gain 或其他变量，完整方法收益应归于完整配置。
2. **最强组合不是某一个新曲线名字，而是三层闭环**：受控识别 `z` 的价值 → 完整 sine--cosine 有限窗几何解释“分配改变了什么” → Cosh 与 TailSpline/BM 把该自由度变成可执行构造，并分别在学习和冻结部署中验证。
3. **零训练冻结部署已经是核心成果，不应被写成应用补充或未来方向**。TailSpline 在 Llama 与 OLMo 的统一两臂结果，以及 BM 的自然 QA 结果，应该和 Cosh 学习实验共同承担论文主结果。
4. **当前最需要优化的是认知结构，不是继续增加材料**：第一页同时出现 `z`、fixed support、full-pair geometry、Cosh、TailSpline、BM、MLA、RULER、NIAH、PPL，容易形成“方法名很多、中心对象不够突出”的第一印象。
5. **窗口内改善仍应是未来方向或初步证据**。当前证据足以支持外推、冻结任务与自然 QA 的具体结论，但不足以将“普遍改善原生窗口质量”提升为成熟主张。

一句话定位建议：

> **The internal exponent allocation \(z\), not only the sampled frequency support, is a useful RoPE design variable. We isolate its effect at fixed support, characterize the positional structure it changes, and derive explicit allocations for learning and zero-training deployment.**

这句话应控制摘要首句、引言首段、贡献条目、理论落点和实验分组。不要把 `z` 写弱为一个实现细节，也不要把完整方法的收益误写成“纯 `z` 因果效应”。

## 2. 32 篇外部论文形成的共同写作规律

### 2.1 顶会论文最常用的强叙事，不是“背景—方法—实验”

最有效的论文通常采用下列一种或多种因果叙事。

| 叙事 | 代表论文 | 为什么有效 | 本稿可用方式 |
|---|---|---|---|
| 推翻既有解释 | Round and Round；Circular Argument | 先展示一个广泛相信的解释不完整，再用机制与干预替代它 | “频率范围并不是完整设计对象；固定范围内的内部位置也有可识别价值” |
| 三个问题逐层回答 | HoPE | 每一节对应一个问题、一个结论，读者几乎不会迷路 | Q1 `z` 是否有独立价值；Q2 它改变什么位置结构；Q3 能否形成学习与冻结方法 |
| 设计准则到模块 | VideoRoPE；STRING | 先列必须满足的性质，再让每个模块对应一个性质 | 把 fixed support、full-pair、weight-independent、finite-grid 分别对应识别、几何、部署约束 |
| 理论—旋钮—实验 | Role of Sparsity；Decoupling | 理论给出可控因素，实验直接扫描这个因素，再形成方法 | 用 fixed-support 多形状实验承担“旋钮”，不要让几何直接冒充 loss 预测器 |
| 诊断—构建 | Rope-to-NoPE；LongRoPE2 | 先用小而有辨识力的诊断找到失败，再在大模型构造中使用 | 先给 fixed-support 识别，再引出 Cosh/TailSpline，而不是先介绍方法名 |
| 精确保证—经验价值分离 | PINE；Formal Framework | 定理只保证它真正能保证的性质，任务收益由实验承担 | Cosh 的唯一性只相对于声明的变分目标；TailSpline 的唯一性只相对于有限网格目标 |

本稿最适合采用 **HoPE 的问题式骨架 + PINE 的保证边界 + Sparsity 的“理论—旋钮—验证”闭环**，而不是把全文写成三个方法的集合。

### 2.2 摘要的共同结构

强摘要通常只完成五件事：

1. 一句话指出被忽略的变量或失败模式；
2. 一句话说明怎样受控识别；
3. 一句话给出理论或构造，但不展开全部术语；
4. 一到两句给出跨设置的关键数字；
5. 最后一句上升到可复用认识。

当前摘要已经有这个骨架，但名词密度仍高。建议让 `z` 成为唯一一级对象，Cosh 与 TailSpline 是两条实现路径，BM 只在正文或摘要最后结果句中出现。不要再向摘要加入更多历史构造、控制名或附录发现。

可直接采用的前两句：

> The internal exponent allocation \(z\) is a useful RoPE design variable: redistributing interior frequencies improves extrapolation even when the sampled support is fixed. We isolate this effect with fixed-support interventions, characterize the finite-window positional structure that allocation changes, and derive explicit allocations for learning and zero-training deployment.

后续再保留最有辨识力的三组数字：三 seed 固定支持；Cosh 的 16K PPL；TailSpline 的 Llama/OLMo frozen AUC。BM 自然 QA 可以留到引言或实验，不必和所有名字争夺摘要注意力。

### 2.3 贡献条目应按“认识职责”组织

强论文的 contribution 不是“我们做了 A/B/C”，而是三个不同认识职责：

- **Identification**：此前无法区分的变量，被什么控制实验单独识别；
- **Characterization and construction**：该变量改变的数学对象是什么，怎样形成可执行设计；
- **Validation and scope**：在什么训练状态、模型和任务上成立，哪些反例保留。

因此，本稿建议把当前三条进一步平行化：

1. **Allocation is independently useful.** 固定初始化、训练和频率端点，只移动 30 个内部频率，三 seed 的 2/4/8× 外推全部同向；冻结模型的 matched-support 干预补充这一识别。
2. **Allocation changes finite-window positional geometry and admits explicit designs.** 完整旋转对的 canonical overlap 与有效秩刻画供给的位置方向；Cosh 与 TailSpline 分别实现训练期与冻结期构造。
3. **The designs work under both learning and zero-training deployment.** Cosh 的多 seed/续训结果与 TailSpline 的跨模型冻结结果共同形成实践结论，并明确保留局部 PPL、QA 和 Qwen BM 反例。

这样的条目比“理论、方法、实验”更能让审稿人看见缺口、解决和证据范围。

## 3. 理论写法：本稿应该学什么

### 3.1 先声明数学对象，再声明保证

Formal Framework、Non-Asymptotic Length Generalization、STRING、Emergence of Position Bias 和 GRAPE 都有一个共同优点：正文很早就说清楚“对象是什么、假设是什么、定理保证什么”。本稿当前对
`V_omega = span{cos(omega Delta), sin(omega Delta)}` 的定义是正确的，也是相对于只看 cosine 曲线的重要优势。

建议每个理论块保持同一四句结构：

1. **对象**：完整旋转对在声明的 separation measure 下供给二维位置函数空间；
2. **命题**：canonical overlap 与 Rényi-2 effective rank 的精确恒等式；慢频子空间的有限窗极限；
3. **直接含义**：不同频率可以占用高度重合的位置方向；
4. **非含义**：这不是任务损失中介，也不预测任意 checkpoint 对任意频率表的偏好。

Round and Round、FoPE、MrRoPE 等论文的概念比喻很吸引人，但也展示了风险：wavelet、DSP、radix 或“认知”语言一旦先于严格对象，很容易让类比承担超出证明的因果结论。本稿应坚持现在的 `supplied basis` 边界，不把“重合”写成“模型必然浪费这些维度”。

### 3.2 将“定理保证”与“为什么选择这个目标”分开

Cosh 与 TailSpline 都有两层不同命题：

- 给定目标后，解的唯一性、正性、闭式形式或终端 jump 比例是数学结论；
- 为什么该目标适合模型，是设计先验和经验问题。

这两层必须像 PINE 和 Formal Framework 那样明确分离。推荐正文使用稳定句式：

> The theorem establishes the exact optimizer of the stated design objective; it does not assert that this objective is task-optimal. We test its task value below.

TailSpline 尤其要保留“one-sided boundary prior”表述。CPU/KKT 验证不能选择任务最优边界，Llama/OLMo 模型实验才支持当前具体配置的价值。

### 3.3 加一条明确的 theory-to-task contract

Role of Sparsity 和 Decoupling 的理论之所以可信，是因为理论变量在实验里有直接对应旋钮。建议在几何节末增加一段很短的契约：

> Geometry identifies what allocation changes in the supplied positional basis. Fixed-support experiments test whether this variable matters; model evaluations test whether the proposed allocations are useful. We do not assume that overlap alone determines task loss.

这能主动解除审稿人最容易提出的质疑：为什么一个 basis overlap 指标能推出 Cosh 或 TailSpline 的任务结果。答案是它没有直接推出；它定义结构、激发设计，任务结论来自受控实验。

### 3.4 用小表把理论主张和证据角色对齐

Emergence of Position Bias、Non-Asymptotic Length Generalization 和 VideoRoPE 都善用一张小表概括“现象/性质—理论—实验”。本稿适合在附录前部放置下表；版面充足时可压缩进主文。

| 主张 | 数学或实验对象 | 能推出 | 不能推出 |
|---|---|---|---|
| fixed-support `z` 识别 | 相同端点、不同内点 | 当前协议下内部分配有独立作用 | 所有完整方法的收益都只来自 `z` |
| full-pair geometry | supplied sine--cosine subspaces | 有限窗位置方向可能重合 | overlap 单独决定 LM loss |
| Cosh 唯一解 | 声明的密度目标 | 解析、weight-independent 构造 | 任务全局最优 |
| TailSpline 唯一解 | 声明的单侧差分目标 | 精确 finite-grid profile 与 jump 保证 | 任意模型、倍率或窗口都获益 |
| frozen model results | 完整 table × band × gain 配置 | 当前模型/协议下完整方法更好 | 纯 `z` 因果效应或通用 SOTA |

## 4. 实验叙述：从“结果很多”变成“每个实验只回答一个问题”

### 4.1 主文按三个实验问题分组

建议让主文结果标题或开头明确出现三个问题：

1. **Does interior allocation matter beyond support?** 只放 fixed-support 三 seed、多形状、冻结 matched-support；
2. **Can a new allocation be learned?** 放 Cosh 的 432M MLA、训练进程与 750M continuation；
3. **Can an explicit allocation improve frozen deployment?** 放 TailSpline Llama/OLMo，再用 BM natural QA 补充不同边界目标。

现在的实验内容已经基本具备，只需把“证据职责”写得比模型列表更醒目。

### 4.2 每个实验段落采用同一描述顺序

LongRoPE2、PINE、VideoRoPE、RePo 和 MrRoPE 的清晰实验段落通常按以下顺序展开：

1. research question；
2. intervention 与唯一变化；
3. shared controls；
4. model/data/sample unit；
5. metric 与 aggregation；
6. result 和区间；
7. local reversal / limitation；
8. 这组结果支持哪一级主张。

本稿 TailSpline 段已经很接近这一标准。Cosh、BM 和附录历史结果也应尽量统一同一顺序。不要让同一段先报 AUC、再倒回讲 gain、后补样本数，或者把来自不同 adapter/decoder 的结果写成同一趋势。

### 4.3 用一个 protocol glossary 控制异质证据

当前稿横跨 from-scratch、continued training、matched adaptation、frozen table replacement、自然 QA 和视频。广度是优势，但容易被审稿人误解为 pooling。建议在附录 guide 后增加一张一页以内的 protocol 表：

| 标签 | 权重状态 | table 如何得到 | 主要对照 | 结论角色 |
|---|---|---|---|---|
| Fixed-support identification | from scratch / frozen intervention | 端点固定，只动内点 | Geo 或 matched support | 识别 `z` |
| Cosh learning | from scratch / continuation | 解析密度，训练中使用 | Geo/Cosh matched recipe | 学习路径 |
| TailSpline frozen | 无训练 | 公开 grid、window、`s` 的闭式表 | MrPro 同 band/gain/input | 零训练主结果 |
| BM frozen QA | 无训练 | 对称边界目标 | matched complete configurations | 自然 QA 支持 |
| Adaptation evidence | 有明确 continuation | table/position exposure | shared start/recipe | 与 zero-training 分列 |

这一张表会显著减少正文反复解释，也能保护证据边界。

### 4.4 负结果不要删，写成“作用域”

Positional Attention、Selective RoPE、PINE、AdaGroPE 和 Deconstructing 都通过明确局限或失败格增加可信度。本稿应继续保留：

- Llama 8K/16K PPL 略差，PPL AUC 的小收益主要来自 32K；
- TailSpline 的 QA family AUC 下降 1.25 点；
- OLMo 大量生成触顶，官方分数不等同完整答案加 EOS；
- 小 Qwen 128K 上 MrPro 胜 BM；
- native-window 普遍改善尚未成立；
- 不同 checkpoint 对 frozen table 的响应可以相反。

推荐把这些写成“where the current construction helps or does not”，不要使用道歉式语言，也不要藏到只有 reviewer 才能找到的附录末尾。

### 4.5 图注先写结论，再写协议

VideoRoPE、Circular Argument、Massive Values 的高价值图都能仅凭 caption 看懂要回答的问题。本稿当前 Fig. 2/3 caption 已较好。统一采用：

> **结论句。** 面板含义；匹配变量；样本/区间；明确不支持的跨面板推断。

尤其保持现有“两个 panel 协议不同，不能将 overlap 解释为 task-loss mediator”这一句，它是很成熟的边界说明。

## 5. 附录：当前已经很强，重点是导航和去重复

32 篇论文的附录大致分成六类：完整证明、额外实验、实现/超参数、提示词或任务定义、局限/失败案例、reproducibility/checklist。最好的组织有三个特点：

1. **Formal Framework** 在长附录前提供目录/FAQ，并将设计选择、实验细节、完整证明分开；
2. **PINE** 将完整证明、统计、全表、实现和定性例子依次排列；
3. **Deconstructing、GRAPE、Selective RoPE** 先给 notation 或 mathematical derivation，再放实验细节和支撑结果。

本稿已有 `Guide to the Supplementary Material`，并按“识别—构造—完整结果—边界控制”给出入口。这是可以保留并强化的做法。建议：

- guide 表后补一张 protocol glossary，而不是再写一段历史叙述；
- 每个证明开头都写“used by main-text claim X”，并列 assumptions；
- 所有完整表保留 sample unit、aggregation、decoder 和 cap 定义；
- 同一数字不要在多个附录章节重复成不同精度或不同命名；
- 旧实验按证据职责组织，不按项目时间线组织；
- appendix 可以长，但 reader path 必须短；不要为了页数删掉 negative cells、完整分项或来源身份。

当前 55 页成稿约 9 页主文，附录体量本身并不异常；真正风险是历史材料重复和构造名字跨章节漂移。

## 6. 逐篇审读：可以直接借鉴什么

以下每一项都记录该论文最有效的叙事/写法、本稿可以直接学习的做法，以及不应照搬的部分。

### ICLR 2025

1. **Round and Round We Go! What Makes Rotary Positional Encodings Useful?** 以“距离衰减解释并不成立”开场，随后区分高频 positional heads 与低频 semantic bands，最后提出 p-RoPE。可学：先推翻不完整解释，再给机制和构造；讨论中区分证据与猜想。可直接用于本稿的句法是“range-only view is incomplete”。不要学：用少量 head pattern 将低频用途普遍化；本稿的 full-pair geometry 必须继续称 supplied basis。
2. **Wavelet-Based Positional Representation for Long Context.** 把 RoPE/ALiBi 重新解释为受限 wavelet，再以窗口尺度多样性导出 Ricker wavelet 方法；短上下文和长上下文实验分节。可学：概念桥梁后马上落到具体设计维度；附录系统比较 wavelet type、scale/shift 和 LongBench。不要学：让 wavelet 类比代替模型机制证明。
3. **PINE: Eliminating Position Bias of Language Models.** 从文档顺序偏置出发，先证明 invariance，再给 training-free 重排，并在 judge、RAG、分子、数学任务上验证；明确 2×/8× 开销。可学：定理保证与性能结果严格分开；正文主动写成本和适用边界。不要学：把可证明 invariance 自动等同于任务质量。
4. **A Formal Framework for Understanding Length Generalization in Transformers.** 定义 limit transformer 与 C-RASP 可表达性，给非正式主定理、任务预测和实验，再在超长附录完整形式化。可学：正文给 theorem map，附录给 FAQ、设计选择和形式化定义；明确理想化假设。不要学：将理想化模型结论无条件外推到真实 LLM。
5. **Scaling Instruction-Tuned LLMs to Million-Token Contexts via Hierarchical Synthetic Data Generation.** 先展示数据生成流水线，再从 180K/350K/650K 分阶段证据上升到 1M headline，最后做广 benchmark 和能力保留。可学：昂贵 headline 之前先给分阶段、可诊断证据；消融可在较低成本窗口完成。不要学：仅凭最大上下文长度组织贡献；本稿核心不是“更长”。
6. **Linear Transformer Topological Masking with Graph Random Features.** 一个中心问题贯穿：怎样学习拓扑邻接并保持线性复杂度；贡献依次是 learnable mask、GRF 近似/浓缩保证、图像与 30K 点云结果。可学：方法、复杂度、理论和任务一一对应；附录把 proofs、related work、further experiments 和 asymmetric extension 分开。不要学：为显得理论完整而引入与主因果链无关的额外结构。

### ICML 2025

7. **Positional Attention.** contribution 按 expressivity、learnability、OOD 分层，实验也按 ID、OOD、mixed-type 排列，并公开 induction failure。可学：贡献标题和实验标题使用同一类别；失败任务直接限定作用域。不要学：把理论可表达性写成训练中必然获得。
8. **Non-Asymptotic Length Generalization.** 定义 length complexity、最优训练长度和 decidability 关系，用一张总表汇总 DFA/CFG/C-RASP 的定理，正文给 proof sketch，附录承载大量证明。可学：为多个定理制作“对象—结论—条件”总表。不要学：让大证明量挤掉问题直觉和经验落点。
9. **The Role of Sparsity in Length Generalization.** 明确列出 T1/T2 理论 takeaway，合成实验直接控制 sparsity，再用 PPC 和自然语言远 token masking 落地。可学：理论旋钮、受控实验和实用方法形成闭环；这是本稿 fixed-support `z` 最值得学习的范式。不要学：把一个控制变量在合成任务的结论称为真实语言全部原因。
10. **Universal Length Generalization with Turing Programs.** 以统一的 scratchpad/Turing 程序构造覆盖加法、乘法、SGD 和随机图灵机，证明和 RASP 实现高度集中。可学：一个方法贯穿多个任务比多个近义方法更易记。不要学：本稿不需要为统一感把 Cosh、TailSpline、BM 强行写成同一个目标的特例。
11. **Fourier Position Embedding.** 从 DSP 的 spectrum damage 诊断出发，提出 Fourier series 与 undertrained-component 处理，实验按 pretraining、continued training、fine-tuning 分开，并有机制可视化和消融。可学：严格区分训练阶段；将 mechanism validation 单列。不要学：将频域语言本身当作任务因果证明。
12. **STRING: Generalized RoPE for Learning Equivariant Representations.** 先列 separability 与 translation invariance 两个 desiderata，再给一般形式、快速实现和复杂度表，最后跨 vision/robotics 验证。可学：先列设计公理，再说明构造满足什么；复杂度与普通 RoPE 并列。不要学：本稿保持标准 rotary operator，不要因相关性扩大到 group-action 总框架。
13. **LieRE.** 用 dense skew generators 将 RoPE 推到 2D/3D，结合 synthetic spatial task、分辨率扩展和匹配 recipe；附录集中超参数。可学：新位置机制要有对称的 synthetic probe 和真实任务；报告区间。不要学：将 operator 改变与 frequency allocation 混为直接对手。
14. **TAPE: Contextualized Equivariant Positional Encoding.** 以 addressing framing 引出 layerwise contextual PE，用 equivariance/expressivity 理论支撑，再覆盖 scratch、PEFT、算法、SCROLLS 和 passkey。可学：一个高层概念统一多种实验，但每组仍说明训练状态。不要学：context-aware operator 改动远大于本稿变量，不应用它扩大 novelty 口径。
15. **Emergence of Position Bias in Transformers.** 用图动力学分析 causal mask 与 PE 衰减，把经验观察、定理和章节做成映射表，再用合成实验验证。可学：一张 claim-to-theorem-to-experiment 表；把 mask 效应与 PE 效应拆开。不要学：从简化 dynamics 直接下真实语言头的普遍结论。
16. **VideoRoPE.** 先给四项设计性质和 failure matrix，再让 LTA、DL、ATS 模块逐项解决，新增 V-NIAH-D，并报告完整长度与模块消融。可学：设计要求—模块—实验严格对应；新 benchmark 必须针对旧 benchmark 分不出的失败。不要学：本稿现有任务已经足以支撑主张，不需要为了新颖再造 benchmark。
17. **Massive Values in RoPE.** 以编号 finding 组织全文，先定位 Q/K massive dimensions，再用 disruption、quantization 和时间分析区分 contextual 与 parametric knowledge。可学：证据按相关—干预—替代解释排布；因果干预先于解释语言。不要学：仅靠 attention norm 或可视化做机制定论。

### NeurIPS 2025

18. **Cameras as Relative Positional Encoding.** 贡献明确分 Survey、Method、Evaluation、Task generalization；比较 relative/absolute camera encoding、intrinsics/extrinsics、OOD sequence/intrinsics 和任务迁移。可学：贡献用功能标签；把互补因素用 factorial 对照分开。不要学：多模态相机变量与本稿频率分配只属写法参考，不是近邻 novelty。
19. **A Circular Argument: The Case for Spherical RoPE.** 先证明 learned/Mixed RoPE 的一般性，再列 property matrix，通过 Spherical RoPE、均匀频率和 matched seeds 做性质级因果控制。可学：不要只比较方法名，要比较 equivariance、direction diversity、frequency distribution 等具体性质。不要学：vision 结论不能直接迁移为语言长上下文结论。
20. **RoPECraft.** training-free motion transfer 通过 warped RoPE、phase constraint 和优化完成，同时引入 FTD，配定量、定性、用户研究和 runtime。可学：冻结操作必须把预处理、运行时、失败案例和 qualitative evidence 写全。不要学：文本分配论文不需要引入视觉展示型指标。
21. **From RoPE to NoPE.** 先用约 750B-token 诊断模型分析 attention mass 和层间 division of labour，再构建局部 RoPE/全局 NoPE 架构并扩展到 5T 训练。可学：诊断规模与最终验证规模分开；架构结论前有小型可辨识证据。不要学：attention mass 的相关模式不能单独承担因果解释。
22. **HoPE: Hybrid of Position Embedding for Long Context VLMs.** 全文围绕三个显式问题和三个 conclusion；理论指出固定 temporal frequency 最终违反语义偏好，再给 HFA/DTS，覆盖 2B/7B、8–64K 和消融。可学：本稿最适合直接学习的问题式章节和小结句。不要学：不要把 TailSpline/BM 增加成三四个平级模块，核心对象始终是 `z`。
23. **Dynamical Properties of Positional Encoding.** 从连续时间 token dynamics 给出收敛/发散条件，分析 APE/RoPE 效应，再用实验验证 pathology 并给结构修复。可学：定理—病理—修复的顺序；每个修复对应一个被观测问题。不要学：连续时间近似的结论必须保持假设范围。

### ACL / Long-context 方法

24. **AdaGroPE.** training-free progressive relative-position reuse，给动态长度映射、notation table 和算法，再跨模型覆盖 PG19、passkey、LongBench、L-Eval，消融 `P`、`r` 与任务复杂度；limitations 明说缺乏理论。可学：部署算法、符号表、参数指南和复杂度写得可复现；没有理论就明确说。不要学：方法广测不能替代对核心变量的受控归因。
25. **LongRoPE2.** 以 undertrained high-frequency dimensions 假说引出 needle-PPL evolutionary search 与 mixed-window training，随后同时测 RULER、needle、真实任务和短上下文保持；关键维度、搜索和 mixed training 都有消融。可学：长窗口提升与短窗口保留必须共同报告；每个组件有单独验证。不要学：本稿 weight-independent 构造不应转成搜索/校准故事。

### 2026 邻近工作

26. **PPE: Positional Preservation Encoding for Multimodal Token Compression.** 在压缩 token 时保留多个位置并级联压缩，同时报告 55%/94% reduction、任务质量、SFT/training-free、分辨率和失败案例。可学：效率和质量共同报告；附录按 ratio、stage、backbone、size、task、failure 分层。不要学：该问题是位置保留而非内部频率分配。
27. **RePo.** 从 hidden states 学连续、非线性 position assignment，在 OLMo 1B/7B 上覆盖 noisy/structured/long tasks，同时检查普通短任务、attention mass、位置模式和效率。可学：清楚写“预期在哪类任务帮助、在哪类任务不应帮助”。不要学：RePo 改变位置本身，本稿不要将其描述为同一变量上的直接比较。
28. **GRAPE.** 用 group action 统一 multiplicative SO(d) 和 additive unipotent GL，使 RoPE、ALiBi、FoX 成为特例，并给 rank-2 快速算法、streaming 和谱分析。可学：notation、special-case map、复杂度与伪代码的附录组织。不要学：通用框架的宏大口径；本稿贡献是现有 operator 内的分配自由度。
29. **Decoupling Positional and Symbolic Attention Behavior.** 给 positional/symbolic head 的精确定义、排斥定理和评分，在 Gemma/Qwen/Llama 验证，再用 Index/Retrieval/MIX canonical tasks 与频率扫描得到 U/倒 U 曲线。可学：概念必须有 canonical task；机制分析用专门指标而非通用 benchmark 替代。不要学：canonical probe 的结论不能越过模型/头/任务范围。
30. **Selective Rotary Position Embedding.** 用 rotation + decay 统一线性 attention 视角，提出 input-dependent rotation，先做 synthetic recall/state tracking，再做 370M/1.3B LM；坦诚 PPL/accuracy 混合结果和未验证 length extrapolation。可学：负结果和“尚未测试什么”可以直接增强可信度。不要学：未完成的 future work 不能当现有贡献。
31. **Deconstructing Positional Information: From Attention Logits to Training Biases.** 用 Toeplitz additive/multiplicative 框架，两个 synthetic tasks、single-head deposit pattern、四项干预和后续定理形成紧密链条。可学：先观察、再 ablate、最后形式化；每个 ablation 排除一个替代解释。不要学：lens/hypothesis 应保持为解释工具，不要写成完整机制定论。
32. **MrRoPE.** 以 mixed-radix framing 组织 Uni/Pro 构造，广测 PPL、RULER、NIAH、InfiniteBench，并给注意力和边界分析；limitations 保留 Uni 的不足。可学：它是 TailSpline 最直接的完整方法基线，必须按相同 band、gain、输入和评分写清。不要学：radix 隐喻不是任务最优证明；也不能把 TailSpline 的低频 junction 设计称为首次非均匀频率设计。

### 本项目旧稿

33. **RoPE Has a Spectral Budget（旧稿）.** 旧稿的优点是 “spectral budget” 单线叙事和 weight-table crossing 的可记忆性；不足是早于 TailSpline 统一两臂结果，当前标题、结果角色和证据边界均已被 `Beyond the Base` 更新。可回收的是少量解释句和 crossing 兼容性材料，不能把它当外部论文、当前 owner 或待恢复主线。

## 7. 当前稿的具体诊断

### 7.1 已经做对的部分

- 标题 `Beyond the Base` 简洁，直接对应“范围之外还有分配”；不建议恢复旧标题。
- 摘要首句已经把固定支持和内部频率放在一起，且数字密度合理。
- 引言第二段明确“完整方法可以组合 allocation、range、band、amplitude”，这条归因边界必须保留。
- 理论使用完整 sine--cosine pair，而非单 cosine；canonical overlap/effective-rank 身份清楚。
- Cosh 明确称 geometry-inspired、weight-independent prior，没有冒充 task-optimal theorem。
- TailSpline 的 one-sided objective、精确 finite-grid 解、终端 jump ratio 与 frozen 安装规则已经足够方法化。
- Llama/OLMo frozen 评测写明 band、gain、样本数、AUC、paired interval、cap 和局部反转，达到顶会可审计标准。
- BM natural QA、Qwen 反例和不同 checkpoint 兼容性边界都被保留。
- 附录 guide 按读者问题导航，比按项目历史顺序更成熟。

### 7.2 当前主要风险

| 风险 | 审稿人可能产生的误解 | 修复 |
|---|---|---|
| 一级对象不够突出 | 这是 Cosh + TailSpline + BM 的方法合集 | 在摘要、引言、贡献和结果标题反复固定唯一一级对象 `z` |
| theory-to-task 跳跃 | effective rank 被当成 loss 预测器 | 增加 theory-to-task contract，保留 supplied-basis 边界 |
| 完整方法归因过强 | TailSpline 胜利证明了纯 `z` 单因素 | 明写完整配置收益；固定支持实验才承担独立识别 |
| protocol 太多 | 不同训练/适配/frozen 结果被误读为同一估计量 | 增加 protocol glossary；正文按问题分组 |
| zero-training 被稀释 | frozen deployment 只是补充应用 | 将 TailSpline 放入摘要、引言第四段和第三条贡献的中心位置 |
| “in-window”过度外推 | 个别 M4 双改善被写成通用能力 | 保持 future/initial evidence 口径 |
| novelty 口径过宽 | 声称首次 nonuniform/smooth/learned frequencies | 只主张 fixed-support identification、full-pair geometry、具体 explicit objectives 与验证 |

## 8. 可直接执行的改稿方案

### P0：第一页必须完成

1. 摘要首句显式写 `internal exponent allocation z`，并把它作为唯一主语。
2. 引言首段改成两个设计决定：**sampled support** 与 **finite allocation within it**；现有方法常同时改变二者，所以需要先隔离，再构造。
3. 第二段只讲 fixed-support 识别和边界；第三段只讲 geometry → Cosh/TailSpline；第四段只讲 learning + frozen 主结果。
4. 三条贡献统一为 Identification / Characterization and construction / Learning and zero-training validation。
5. 不在第一页增加更多方法名、旧分支、研究历史或新的数字。

建议引言开头：

> A RoPE table makes two distinct design choices: the sampled frequency support and how a finite set of rotary pairs is allocated within it. Existing long-context methods often change both. We isolate allocation and show that it improves extrapolation at fixed support, then derive explicit allocations for learning and frozen deployment.

### P1：主文结构与过渡

1. 在 fixed-support 结果节首句写 Q1；在 theory 首句写 Q2；在 experiments 首句写 Q3。
2. 在 geometry 末增加三句 theory-to-task contract。
3. Cosh 开头说明“new model can learn a supplied allocation”；TailSpline 开头说明“frozen model already has a native table”，让两条路线不是平级名字堆叠。
4. TailSpline 结果中继续先写匹配控制，再写 AUC 和 interval；BM 明确称 complementary symmetric construction。
5. 讨论段集中总结 failure map：native PPL、QA family、generation cap、小 Qwen、checkpoint dependence。

### P1：相关工作

当前 related work 的三轴已经正确：

- support/range：base、FMRoPE、YaRN；
- interior allocation / learned frequencies：MrRoPE、LeRoPE、AdaRoPE、Data Shapes；
- operator/position changes：FoPE、GRAPE、Selective RoPE、RePo。

不建议扩成长 survey。建议在附录加一张 method-coordinate 表，列：support 是否变、内部 allocation 是否变、position assignment 是否变、operator 是否变、是否读 weights/activations、是否训练。它比正文逐篇 novelty 声明更有效。

### P2：附录和复现

1. 在 appendix guide 后加入 protocol glossary 与 claim/guarantee boundary 表。
2. 证明入口统一列 assumptions、statement、main-text use。
3. 完整实验表统一列 sample unit、decoder、aggregation、CI、cap、source identity。
4. 检查 Geo/Native/FMRoPE/MrPro/Cosh/TailSpline/BM 的命名在图、表、caption、附录是否一一一致。
5. 保留全部负格，不为压页删除证据边界；只删重复叙述和项目时间线。

## 9. 我们可以进一步做、但当前不应自动启动的研究

这些方向可增强论文，但不是当前改稿完成的前置条件，也不构成 GPU、下载或远程执行授权。

1. **Fixed-support `z × dependency width` 控制任务。** 直接检验不同 dependency width 对内部 allocation 的偏好。需要注意 Data Shapes 已讨论 `theta ~ 1/W`，新意必须是“有限离散 allocation、固定 support、匹配训练”的交互，而不是再次发现频率与依赖宽度相关。
2. **TailSpline gain 的 head/function decomposition。** 用 matched interventions 判断收益来自 retrieval、tracking、aggregation 的哪些 heads/bands。它是机制解释，不能让 weight-independent 方法反过来依赖 activations 拟合。
3. **Allocation 的 canonical positional/symbolic probes。** 借鉴 Decoupling，构造固定支持下 Index/Retrieval/MIX 或相位分辨任务，验证频率分配的功能取舍。
4. **Native-window retention 的专门确认。** 只有未来要主张广义部署或原生窗普遍改善时才需要；当前已有外推和冻结任务证据不应被它阻塞。

## 10. 不应借鉴或直接照搬的做法

- 不把 wavelet、radix、spectral budget、cognitive load 等类比写成理论证明；
- 不声称“首次非均匀频率设计”或没有完整检索支持的 universal/SOTA；
- 不把 from-scratch、continued training、adaptation 和 zero-training 结果合成同一个效应；
- 不用几何 rank、attention pattern 或 activation entropy 单独证明任务机制；
- 不隐藏负结果、generation cap、单 seed、report-backed/raw-backed 区别；
- 不把 TailSpline 完整配置收益写成纯 `z` 单因素收益；
- 不让新机制实验改变 TailSpline 的无权重、无激活、无 calibration 定义；
- 不为“丰富”摘要继续增加构造名；
- 不恢复旧稿作为当前论文主线。

## 11. 外部检索补充与 novelty 边界

本地 32 篇外部论文足以完成写作模式比较，但不能单独证明 2026 年 9 月的完整 novelty。补充语义检索覆盖 61 条候选结果（不是 61 篇去重论文），确认几篇直接相关工作未作为本地文件出现：

- [LeRoPE](https://arxiv.org/abs/2607.10134)：学习 per-frequency scales；
- [AdaRoPE](https://arxiv.org/abs/2607.19363)：head-specific learned frequencies 与 scaling；
- [How Data Shapes the Learning of RoPE Frequencies](https://arxiv.org/abs/2607.07678)：数据依赖宽度与 learned frequency use；
- [DoPE](https://arxiv.org/abs/2511.09146)：用表示熵诊断和修改 PE；
- [FMRoPE](https://openreview.net/forum?id=PR1PPxvG9Q)：frequency range/base 设计。

当前 `sections/02_related.tex` 已经覆盖这些最近工作，并正确区分 support、interior allocation、learned use 与 operator/position change。最稳妥的 novelty 口径是：

1. 在固定 sampled support 下受控识别内部 `z` 的价值；
2. 用完整 sine--cosine subspaces 刻画 finite-window allocation geometry；
3. 给出 Cosh 与 TailSpline 的具体显式目标、闭式/精确 finite-grid 解；
4. 在 matched learning 与 zero-training frozen protocols 下验证具体构造，并保留完整作用域。

## 12. 给后续改稿任务的执行合同

后续改稿应直接在当前 `paper-2027` 稿件上完成最小、完整修改，并遵守：

1. 当前 owner 是 `docs/research/next_stage_20260912/index.md`，旧稿和历史目录不控制当前主线；
2. 核心结论必须写强：`z` 有用，固定支持实验识别它；
3. 完整方法可以改变支持、band、endpoint、gain 或其他变量，其收益归于完整方法；
4. Cosh learning、TailSpline frozen Llama/OLMo、BM frozen natural QA 都是核心现有证据；
5. zero-training frozen deployment 是现有成功，不是 future work；
6. in-window 普遍改善仍是未来方向/初步证据；
7. 不新增 GPU 实验、下载、远程变更或发布动作；
8. 保留工作区已有修改和证据身份，不覆盖不相关 dirty work；
9. 修改后核对 claim map、引用、主文/附录命名，并编译检查 PDF。

最终验收标准不是“文笔更顺”，而是一个第一次阅读者可以准确复述：

> 这篇论文发现并受控验证了 RoPE 内部指数分配 `z` 的独立价值，解释了它改变的有限窗位置结构，给出了学习期与冻结期的显式构造，并以匹配协议验证这些构造，同时没有把完整方法收益错误归因成单因素定律。
