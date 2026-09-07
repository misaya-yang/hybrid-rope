# 从 MrRoPE 强基线推进频率分配：当前研究计划

- **状态 / 日期：** 2026-09-07；方法设计沿革。后续 Qwen 构造与开发评测已经执行，结果见实际 owner；下文较早的矩阵不是当前队列。
- **问题：** 在可靠的 MrRoPE 零训练基线上，结合 CoPE 和本方已有结果，分别改进中段分配、尾频缩放和区域衔接，能否以少量比较取得实际收益？
- **依据：** MrRoPE 原文及本地全文、CoPE 论文与官方代码、Qwen 官方配置、仓库 Z owner 和 E0/E1 原始结果核对。文献及代码版本见末节。
- **证据边界：** 恒等式属于 Derived result；改法与结果预测属于 Working hypothesis；论文原结果属于外部报告。本文没有证明新方法最优，也没有启动实验。
- **替代关系：** 撤销本文件此前“先闭合通用分配理论才能试验”的执行顺序。另存的[通用分配推导](ROPE_GENERAL_ALLOCATION_DERIVATION_20260907.md)保留数学检查，退出当前方法路线。Pro 提示词已交给作者使用，本轮不修改。

## 当前使用方式

当前优先本方方法、阶段内自主推进，复用公开对手结果，不默认执行下方历史五表矩阵。[实际协议与结果](ROPE_SCALE_TRANSPORT_PILOT_20260907.md)已记录 Qwen 统计、问答诊断和组合 RULER 子集；当前未解决的是 UUID 检索与变量追踪的失败归因，恢复工作只看 [HANDOFF](../../paper-2027/HANDOFF.md)。

下文用于解释方法关系和设计选择。涉及“未下载”“未实现”“首轮拟比较”等措辞，均指其形成时的状态；当前实施情况由实际 owner 覆盖，不作为重新下载或重跑基线的理由。

## 作者临睡前的新顺序（2026-09-07，覆盖下方早期执行建议）

1. 先零训练：执行作者提供的[低频去载波](ROPE_CARRIER_REMOVAL_PILOT_20260907.md)，
   只运行本方固定候选，优先对照MrPro已公布的同模型/任务结果；必要开发比较
   复用已保存Mr输出。原Carrier20条开发行未显示收益，额外50条数字检索也仅32分；剩余12项已停止。
   [逐槽原生相位约束候选](ROPE_NATIVE_SECTOR_CARRIER_20260907.md)作为实验3执行中，
   不能提前宣称第一阶段通过。禁止根据这些答案扫描c、改中段或把统计下降当成功。
2. 零训练得到有效结果后，研究LoRA的增益、真实长序列学习和遗忘；并不永久
   排除训练。旧Qwen最长物理16K低于其Native32K，小数据结果不能代表64K/128K
   适应的上限。CPU先分析数据/成本和已有有效训练材料；GPU阶段仍按顺序执行。
3. 最后研究适配稀疏/压缩/混合注意力的位置编码，允许超出RoPE。先核查已公开
   架构、可区分的问题及可实现机制；不在这张GPU上默认下载/训练旗舰模型。
   三合一长期目标仍保留；oral/solid accept是作者目标，不是当前证据或承诺。

### LoRA：核实后的事实与待解决量

[YaRN §4.1–4.2](https://arxiv.org/html/2309.00071v2)对Llama2的正式训练使用
64K PG19文本、global batch64；先400步，再在更高缩放下200步，后阶段仍是
64K物理长度、测试到128K。按65536 token/段计算，累计约25.17亿token；
不能写成“YaRN一定直接训练128K”，也不能与本方128个小语义组等同。
官方训练为全参数；[MrRoPE §4.1与Appendix E](https://arxiv.org/html/2601.22181v1)
明确无微调实验，没有公开的MrPro LoRA指标可直接借用。

本方已有两种必须区分的结果：
[OLMo Q/K-only](../../paper-2027/research/attention-aware-retrofit/results/adaptation-coadaptation/LOG_P2_QK_LORA_GAIN_MATCHED_RESULT_20260904.md)
改善NLL而不改善生成；[Qwen1.5B全线性LoRA](../../paper-2027/research/attention-aware-retrofit/results/SINGLE_TABLE_FFN_SERVER_EXECUTION_20260904.md)
把受控单证据16K远端从3/32提高到24/32，但分项遗忘仍存在。后者确实有能力
学习证据，不能因前者失败而排除LoRA；也不能把16K结果称作超过Qwen Native。

后续具体训练前需要的是可执行的物理64K完整更新成本、实际不同文本/答案token
总量、固定最终checkpoint，以及同部署表的前后生成和短能力保留。复用分块LM
head与梯度检查点实现，避免64K×vocabulary激活；不把虚拟位置跨度称为物理
长上下文。训练目标需要覆盖上下文而不只盯末尾少量答案；Native replay/KL只
是保留约束，短任务的丢失/获得才是直接功能证据。冻结基座权重不保证功能不忘。

作者允许必要时清理不用且可重新下载的权重；保留当前模型、有效adapter、原始
证据和回执。存储与运行状态只写HANDOFF，此处不是新队列或自动清理名单。

### 稀疏注意力：先纠正架构前提

“少用全注意力”不等于“不用RoPE”。Qwen3.8-Max公开基座的
[模型卡](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B/blob/207bd685a7e3696cfaff12ded7c6a7ea0f88c996/README.md)
列出每四层三层Gated DeltaNet、一层Gated Attention，后者仍用64维RoPE；
不能把稀疏MoE专家与稀疏attention混写。
[Qwen3.8-Next论文](https://arxiv.org/html/2608.30320v1)中的QSA则在长度256K的
继续训练引入压缩索引器：先池化4个token的key，再以块起点作partial RoPE，
选中的块展开为真实token交给core attention。这让“索引器如何给一个块编码
位置”和“core如何给一个token编码位置”成为不同问题。

[DeepSeek-V4官方说明](https://deepseek.com/en/news/v4-preview/)采用token压缩与
DSA；[GLM-5.3-Flash模型卡](https://huggingface.co/zai-org/GLM-5.3-Flash)描述稀疏与
线性混合。具体位置算子仍须读其固定版本代码，不能从宣传名称推断已弃RoPE。
Qwen3.8-Next还报告NoPE预训练相近、后训练后更易不终止：位置方法不能只按
语言模型损失或几何稳定判胜。本节是已核查入口，尚无本方稀疏编码突破或新实验。

## 0. 收到尺度搬运提案后的当前修订

已读作者提供的 [Pro 原文](../../paper-2027/research/external-reviews/ROPE_SCALE_TRANSPORT_METHOD_AND_CODEX_20260907.md)。[独立评议](ROPE_SCALE_TRANSPORT_REVIEW_20260907.md)保留其响应尺度估计和干扰诊断，补充可见窗口裁切控制；不采纳对手微调和改回 OLMo 地板终点的安排。公式可提供候选依据，尚无真实模型验证。

**首轮收敛为三个参考加两个方向：** Native、YaRN、MrPro、MrPro+CoPE-style tail，以及一张以尺度统计生成的候选。后者替代原手设 `/2` 的 T↑；不同时铺开旧 Z 中段、桥接和全套 Pro 原矩阵。校准器尚未实现时不阻塞强基线/CoPE 的直接比较。原有限扰动公式作为历史方向解释保留，不能误当当前开跑清单。

## 1. 研究起点与现有资产

从有效方法的可改进部分出发：MrRoPE 提供冻结权重的强基线和累计 radix 构造；CoPE 提供低频软裁剪的机制与实现；本方提供 Z 的频率重分配经验、Cosh 的收益与反转，以及可复用的实验代码。已有结果不等于各方法已经达到最优，但也不能先把它们重做成一个抽象的“通用最优”问题。

本轮限定三个问题：**中段怎么分；尾频是否应全部除以 s；怎样衔接这两部分。** 理论用于确定改动位置、方向、可计算规则和失效条件。无需等到完整最优性证明，但也不把任意新曲线包装成推导。

长期“三合一”仍指本方方法的从零训练、轻量适配和冻结部署。先在可信冻结基线上找到有效改进，再沿同一构造验证其他阶段；不要求给 MrRoPE/YaRN 微调。现有 Z 适配任务保持其独立授权与结果范围，本文件不改 Luna 队列。

## 2. 对齐实际操作，而不是方法名字

令频率 ω_j 按槽位由高到低排列，j=0,…,K−1；s 为扩展倍数。

| 方法 | 实际操作 | 可借鉴的部分与边界 |
| --- | --- | --- |
| MrPro | ω′_j=ω_j s^(−m_j)；高段 m=0，低段 m=1，中段 m=t(t+1)/[n(n+1)] | 累计边增量和两端保护；等差边指数是设计假设，不是 LM 最优性结论 |
| CoPE 官方实现 | 最后 20 槽乘 c_j=[1+cos(πu_j)]/2，u 从 0 到 1 等距取样，末项为零 | 让深尾部更慢并保留连续变化；不是增加旋转，也不是删除 Q/K 维度 |
| 本方 Z | legacy-u 导出的全槽 log-p2 位移；另有历史选择的 gain | 可提取中段分配作受控比较；不能把整张 Z 与 Mr 互换后归因于中段 |

MrRoPE 的尾频除以 s 是位置插值，不是统一修改 scalar base；尾段自身 λ=1 时，前面累计的 radix 乘积仍为 s。其高频频率不变，不代表增加全局 gain 后注意力贡献不变。[MrRoPE §3 / Appendix B](https://arxiv.org/html/2601.22181v1)

CoPE 有两个必须明确的实现事实：论文 Eq.10 写的是按**频率值**变化的窗口，官方代码实际按**槽位索引**等距取余弦；首轮采用代码定义并标明版本。官方推理代码先取得 RoPE/YaRN 频率，再乘尾部窗口。CoPE 论文的主要结果来自长上下文训练后的模型，不能算原始 checkpoint 零训练证据。把同一操作施加到 MrPro 上，应称 **MrPro + CoPE-style tail**，不冒称复现了整篇 CoPE 的训练结果。[论文](https://arxiv.org/html/2602.05258v1)、[官方实现](https://github.com/hrlics/CoPE/blob/f8957a1c7e4891f31266ca79299437a672175a1a/modeling_cope.py#L90-L128)

## 3. 首轮选择 Qwen2.5-3B-Instruct

采用 **Qwen/Qwen2.5-3B-Instruct**，这是 MrRoPE 原文测试的模型，不用现有 Qwen 1.5B 或 OLMo 代替。官方配置：Native 32768，base=1000000，36 层，16 个 query heads、2 个 KV heads，head_dim=128，即 K=64。该配置下 Mr 的 32/1 圈边界为 l=23、h=40；中段跨度 n=17，尾段 j≥40。CoPE 最后 20 槽从 j=44 开始，恰好落在这个尾段内，适合分离中段与深尾部的效果。

固定 s=4，目标 128K，所有候选使用一张静态表，不按每条样本长度重算。先在 Native 与 64K 开发样本筛方向；进入确认时使用未参与选择的 128K 和自然任务样本。32K 内也必须使用实际部署表测保持性。

选择理由是原论文可比性、小权重和 GQA，而非模型越新越好。BF16 标准 KV 在 batch=1、128K 时按配置计算约 4.5 GiB；这不包含权重、工作区、logits 与预填充峰值，不能据此承诺 32GB 整体显存或速度。模型 revision 已查到，权重未下载，实际长推理尚未验证。[官方模型与配置](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/tree/aa8e72537993ba99e69dfaafa59ed015b17504d1)

先保留 Native、faithful YaRN、MrPro 三个参考。使用原论文/RULER 官方任务和评分约定取得可解释基线，另记 EOS 与完整输出，不用我们 E1 的 paired-world strict 分数直接对照论文百分比。加入一个 HELMET 自然任务类别，避免只在简单 needle 上筛选。具体数据版本、开发/确认划分和成本随正式运行计划冻结。

E1 中 OLMo 16K 远端 single-evidence 的 MrPro 为 0/32、Z 为 1/32，其他长任务也存在地板效应。这支持换一个已知有较强基线的模型/测量协议，不支持判定 MrRoPE 无效或 Z 优胜。结果范围见 [ROI 评议](ROPE_FREQUENCY_LUNA_ROI_20260907.md)。

## 4. 分辨尾频方向：保留 CoPE，统计候选替代手设倍率

MrPro 在目标窗口 sL0 上，使尾频累计相位与原窗口 L0 一致。这是一种保守选择，不是必须满足的物理定律。真正需要判别的是：深尾部应更稳定，还是需要保留更多位置变化？

**T↓：更强稳定。** 在 MrPro 之后施加官方 CoPE 的 20 槽窗口，得到 ω↓_j=ωMr_j c_j。其余槽位不动，不改变 gain。这是已有机制的直接组合，必须作为后续新组合的强对照，不能把它本身包装为已确立的新颖性。

**原 T↑：固定少压缩对照，已撤出首轮。** 只对 Mr 尾段令 u=(j−h)/(K−1−h)，取 m↑_j=1−u/2，ω↑_j=ω_j s^(−m↑_j)。它在尾段入口仍接到 Mr 的 1/s，在最慢端变为 1/√s；s=4 即最末频率从 /4 变为 /2。高频和中段保持。这里的 1/2 是一次固定、可解释的方向性扰动，**不是理论推出的最优系数**；不继续扫描多个尾指数。

**当前统计候选：** 在可见窗口控制与独立文档检查之后，用响应距离增长估计 β，生成 ωs^(−β)；先保持 Mr 高频，将统计用于中低频。频率接口、顺序处理及重放细节见评议，尚不是已生成的数组。β=1/2 具有上面 `/2` 的相位含义，但不再人为为所有尾槽指定它。

CoPE 与统计候选分别检验更强稳定和放松 PI 的方向；当前 β∈[0,1] 不能产生 CoPE 裁剪，不能说一条公式已经统一两者。结果决定保留哪种机制。统计不可辨或最终 λ=0 时，不宣称新方法有效，也不因此关闭现有 Mr/CoPE 比较。

### 理论如何帮助这里的选择

对单个旋转对、距离 Δ，有精确关系：

\[
\|R(\omega'\Delta)-R(\omega\Delta)\|_2
=2|\sin((\omega'-\omega)\Delta/2)|
\le |\omega'-\omega|\,|\Delta|.
\]

因此极慢尾频在 Native 窗口里的绝对变化可以很小，同时在长窗口逐步产生区别。Qwen 最末频率在 Native 窗口约转 0.00647 圈；在 s=4 的目标窗口，Mr、T↑、T↓ 最末项分别约为 0.00647、0.01294、0 圈。这给出了选择深尾部作有限干预的依据，没有保证模型输出一定保留。

设单块 logit 为 a cos(ωΔ)+b sin(ωΔ)，当 |ωΔ| 小时约为 a+bωΔ−a(ωΔ)²/2。降频趋向保留距离不变的 a，升频保留更多位置变化；系数的符号和内容决定哪种有利。CoPE 将 ω 置零仍保留 a，不等于把该槽位信息抹掉。这解释两种方向为什么都值得一次测试，也解释为何不能从“转得更多”直接推出更好。

## 5. 中段借用本方已有规则，过渡最后处理

**当前状态：** 本节 Z 中段与固定桥接是收益出现后的归因备选，退出首轮。优先对实际有效的统计候选做仅中段/仅尾段消融；不同时引入另一张 Z 表以扩大矩阵。

**中段 M：** 从对应模型配置按现有 Z 流程计算位移 z_j，仅在 l…h 内使用

\[
m^{M}_j=(z_j-z_l)/(z_h-z_l).
\]

高段固定 0、尾段固定 1。与 MrPro 比较时端点、gain、尾频及权重相同，只有中段边增量发生变化。必须重新按 Qwen 配置生成，不能移植 OLMo 数组。若 z_h−z_l 接近零、非单调或数值不稳，CPU 阶段就拒绝这项构造，保留 Mr 中段；不靠静默排序补救。Z 的历史 p2 是现有经验来源，不重标成新推导。

Mr 的中段边增量为 2t/[n(n+1)]；M 的边增量为 (z_{l+t}−z_{l+t−1})/(z_h−z_l)，两者总和同为 1。比较直接回答“相同累计缩放，分给哪些中段槽位更好”。无需再提出另一族 cosh/sigmoid。

当尾频方向确定后，仅补 **M** 和 **M+选定尾频** 两臂，与已测的 Mr、仅尾频构成 2×2。预先看组合相互作用：若组合不超过单独尾频，最终方法就不强行加入 M；若 M 自身有效而组合损伤，保留中段改进。中段与尾部独立有效、组合更好也只是经验结论，不能把同一开发集上的提升当独立确认。

**过渡 S：** 仅在有收益的配置上，补一次固定局部桥接对照。每个实际连接点两侧各两槽，在 log-frequency 位移上作单调三次 Hermite 桥接；端点值固定，端点斜率取外侧差分并按单调条件限制。范围外逐元素不变，保存实际改动槽位。若该带宽无法满足端点/有序约束，停止此候选，不扩展为插值器搜索。

离散槽位上的曲线更光滑不等于 attention 随位置更平滑。S 要检验的是突变的相邻 radix 分配是否造成可测功能代价；若无独立增益，就不把桥接加入最终方法。单纯余弦窗口、二次分配或拼接光滑不承担核心 novelty。

## 6. 少量比较的顺序与退出条件

| 波次 | 对象 | 结果改变什么 |
| --- | --- | --- |
| 基线资格 | Native、YaRN、MrPro | 先取得可信、可分辨的比较；异常先查实现/模板/评分 |
| 已有机制组合 | MrPro+CoPE-style tail | 判断深尾稳定是否在原始冻结权重上有收益，不等校准器才开始研究 |
| 一版统计构造 | Native 相对长度响应与可见窗口控制；一张最终候选 | 替代手设倍率；若数据不支持或退回 Mr，不重跑相同表 |
| 效果比较 | 上述最多五张不同表 | 比较实际 Native 与长任务；重放分数不承担能力选择 |
| 收益归因 | 仅中段、仅尾段，已有 Mr 和联合表复用 | 有联合收益才追加；过渡仅在有具体问题时另测一次 |
| 独立确认 | 一个候选与强参考 | 未见 128K/自然任务和 Native 保持，之后才进入本方适配及其他阶段 |

不把五张表铺满所有模型与长度。原计划八张表的预设矩阵由上表替换。构造阶段若用 Pro 的固定五点 λ，须记录校准选择成本；它不是五个独立新方法或任务分数搜索。所有 Mr 派生臂固定其 gain，旧 Z 整体结果保留原 gain 身份。

现行 Native 损伤预算约 0.12、严格 retention 0.88 保留各自指标定义，不能套到不同分母的 benchmark 总分。开发集领先必须经未见数据确认。

新运行包、模型 hook 和预算尚未验证。Qwen 的 32 文档相对长度校准名义输入为 1835008 tokens，不能复用 Pro 的 OLMo token 量或据此承诺小时数。现有 `cross_audit/tables.py` 不接受零尾项、等频或交叉频率；后续需显式区分主方法约束与算子合法性，不能静默排序或加 epsilon。保持当前 Z 任务的冻结代码和授权。

## 7. 论文贡献与三阶段连接

有价值的目标是：在强零训练基线之上，找到有用的中段/尾段分工，给出简单、可复现的构造，并用消融解释组合何时有效。若只是 MrPro 加现成 CoPE 就胜出，这本身是值得保留的结果，同时提高后续“我们方法”的比较标准；不能为求新意删除这个对照。

本方方法随后用同一构造做轻适配前/后比较；从零训练再检验其学习价值，复用现有有效历史结果但不拿旧 Cosh 权重替代新表训练。三阶段不要求数值数组完全一致，却需要共同机制和明确的阶段输入。未来是否达到 SOTA、能支持何种会议评价，取决于冻结确认和公平比较，而非预先承诺。

## 8. 来源、身份和本轮未执行项

- [MrRoPE 原文](https://arxiv.org/html/2601.22181v1)：§3、§4.1、Appendix B；主实验为零训练，B.1 已测试两条边界，不能声称其未研究边界。原论文数字不能与本项目不同评分终点直接相减。
- [CoPE 原文](https://arxiv.org/html/2602.05258v1)；[官方代码](https://github.com/hrlics/CoPE/blob/f8957a1c7e4891f31266ca79299437a672175a1a/modeling_cope.py)：固定 commit `f8957a1c7e4891f31266ca79299437a672175a1a`，已读推理实现。未核验训练实现及作者训练结果原始输出。
- [Qwen 官方配置](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/aa8e72537993ba99e69dfaafa59ed015b17504d1/config.json)：revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`；本轮只读配置与元数据。
- [Z 定义与历史选择](../../paper-2027/research/attention-aware-retrofit/theory/SINGLE_STATIC_LOG_P2_SELECTION_AND_LORA_20260903.md)、[E1 与 ROI](ROPE_FREQUENCY_LUNA_ROI_20260907.md)、[固定 support 三种子结果](../../paper-2027/research/evidence/EXACT_RANGE_151M_3SEED_RESULT_20260820.md)。不恢复 seed42 权重。
- [指定 9/6 cross-audit](../../paper-2027/research/external-reviews/ROPE_ICLR2027_CROSS_AUDIT_20260906.md)：保留强基线和证据核对建议，阶段优先级按作者最新指令修正；不采用其他实验的 v5。

本轮只做来源核查、标准库参数计算和文档修改。未下载模型、未训练/推理、未更改 Luna 任务、未修改 Pro 提示词或 TeX/PDF。实时状态见 [HANDOFF](../../paper-2027/HANDOFF.md)。


## 稀疏接口的代码核查补充（GPU仍优先零训练）

已读DeepSeek-V4-Pro官方
[固定版本inference/model.py](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/inference/model.py)，
文件SHA `ce962f1face79d4f633d36436576214057a7e11443c9789935e1deb5c6cd1d71`。
Compressor先做逐通道learned gated pooling和RMSNorm，再在末64维应用RoPE；
ratio4还合并重叠窗口，所赋位置仍为当前块起点。Indexer使用独立压缩器、
RoPE、量化旋转和ReLU分数；core同样保留RoPE。纯滑窗层使用另一base并禁用
YaRN，压缩层使用压缩路径的base与YaRN参数。不能给所有路径盲目套一张“统一
长上下文频率表”，也不能说该模型已弃用RoPE。这里只下载代码/配置，没有模型权重。

因此第三阶段至少要分开三件事：位置变化影响索引器是否选到证据；压缩位置
如何代表多个真实token；证据已选中时core的距离编码是否正确。QSA块选择后
展开回token，和V4直接让core使用压缩KV并不等价。下一项推导应针对其中一个
明确算子，避免从全attention下的频率几何直接跳到稀疏系统能力结论。


LoRA文献的另一项限定：[LongLoRA Table2与§3.3](https://arxiv.org/html/2309.12307v2)
发现扩大attention-only LoRA的rank并不能追上全参，开放embedding与norm后
PPL差距显著缩小；这来自Llama2、特定数据和S²-Attn训练，不能直接认定Qwen3B
同样需要全embedding训练。本方已成功学习过的Qwen all-linear LoRA包含FFN，
并不等于该文的attention-only对照。64K实际更新和Native保持需要在本方真实
训练范围下验证，不盲目增rank，也不把shifted sparse训练当稀疏推理编码已解决。


## 真实64K训练资产的CPU准备

候选训练源采用官方[PG19](https://github.com/google-deepmind/pg19)的train分区，
先冻结目录前1000项中至少500000字节的前128篇不同书籍，合计110304242字节。
全部按云目录声明的size和MD5核验，下载截断按Range补齐，原失败回执保留。
没有使用PG19 validation/test，也没有按模型分数选择文本。

[准备器](../../scripts/experiments/scale_transport/prepare_pg19_long.py)对每篇书选取
由seed与source key确定的连续65537-token窗口（65536真实输入+下一token标签），
不拼短文、不拉伸位置。预期一遍为8388608个预测token，明显低于YaRN论文
训练量；此阶段仅准备资产，不承诺这点数据已经足够，不启动模型训练。

当前频率候选最末槽为0，改变了频率support；不能把它的后续收益直接写成
旧论文固定support下pure-z的因果证据。若有收益，应作为明确的冻结部署方法
报告，再分别验证适配效果与原有从零训练证据之间的联系。


## 后续LoRA的单一预案（尚未启动，先看零训练完整结果）

预案在完整RULER分数揭晓前确定，避免按失败任务临时改训练配方：固定当前
频率表/gain，复用已有有效Qwen全线性LoRA的七类线性层范围，r16/alpha16、
无dropout/bias、不开放embedding/norm；AdamW lr2e-5、betas(.9,.95)、无weight
decay、5% warmup与cosine。只保存固定最终128步adapter，不按中间生成挑权重。

每步一篇不同书的真实64K上下文，全部65536个下一token位置做CE；再加权重1
的Native全词表forward KL（先按词表求和，再按预测位置取平均）。Native replay
使用既有源隔离pool的128个train行，四类各32；其tokenizer.json与当前Qwen3B
逐字节相同。非文本行的teacher将来自原始Qwen3B的真实greedy轨迹，不能复用
旧Qwen1.5B logits或用学生adapter冒充原始teacher。teacher执行时关闭adapter，
使用原生频率/gain1；必须在学生建图前恢复部署表。

[Native行准备器](../../scripts/experiments/scale_transport/prepare_native_replay.py)
已完成CPU运行，另冻结128个validation行，source IDs与train不交叉；这仍是
历史开发pool，不声称新盲确认。文本NLL与完整字符串/EOS分开报告，按source
组处理不确定性，不把平均KL当作无遗忘证明。

[训练实现](../../scripts/experiments/scale_transport/long_lora.py)复用已有分块
LM-head CE/KL，避免64K×词表完整激活；新的teacher隔离逻辑已用独立原始小模型
通过CPU集成对照。GPU在本方零训练完整比较过关后才安排两次完整64K更新的
内存/耗时smoke。它不评价能力、不留下候选adapter；其成本计入本夜。
实际smoke若内存不够，保留64K物理长度修复内存问题，不偷偷改回原生长度。

正式训练只有在128步及后续生成验证能落入同一本夜预算时启动；同表训练前后
在预先固定的RULER每项前10条做配对生成，Native128行也实际生成回读。它们是
能力判别，训练loss、数组和smoke不决定晋级。当前一遍8.39M预测token远少于
YaRN论文，不将这项小预算实验升级为整个LoRA类别的上限；暂不新增rank/lr扫描。

适配器推理采用独立BF16低秩权重，不并入BF16基座，以免合并时将微小更新舍去；
这与训练AMP中的BF16 GEMM对应，但仍需真实128K运行核验内存及数值。加载器
检查原模型revision、最终128步、频率表、adapter配置/权重/部署文件SHA。
[CPU集成检查](../../tests/test_long_lora_native_teacher.py)已有3项通过：独立原始
teacher概率、真实greedy前缀、保存/重载后BF16低秩更新一致性。最初一次收集失败
来自独立代码根漏拷贝既有`cross_audit/training.py`，补齐依赖后通过；不是GPU实验。

[Native评测器](../../scripts/experiments/scale_transport/native_lora_eval.py)预备在同一
128行验证集比较原始Native、本表未训练父模型和固定最终adapter。非文本只删除
末尾EOS后解码，保留其他特殊token，检验完整字符串与EOS；文本单独报告全部
下一token位置NLL。该评测器仅通过CPU导入，尚未运行真实模型，不能称为已验证无遗忘。
