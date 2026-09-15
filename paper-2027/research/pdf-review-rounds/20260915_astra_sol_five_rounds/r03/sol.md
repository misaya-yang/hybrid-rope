# ICLR 2027 风格独立评审

## 1. 我理解的中心贡献

本文把 RoPE 的频率设计拆成两个坐标：频率覆盖的对数区间（support/range）与该区间内部的频率分配（allocation）。核心论点是：即使最高、最低频率完全相同，内部频率的位置仍会显著改变模型的长度外推行为；而且训练时可适配的频率表与冻结模型可直接替换的频率表属于两个不同的设计问题。

论文为这一观点给出三层支撑。第一，固定端点的配对训练和冻结模型干预识别内部 allocation 的独立作用。第二，以完整 sine/cosine 二维子空间而非单个相位函数刻画有限窗口内不同频率的重叠，并给出有效秩恒等式与频谱不可由固定 Q/K 换基消除的判据。第三，分别提出用于训练的 Cosh 分配和用于冻结部署的 TailSpline；前者用显式密度目标重分配学习时的频率基，后者用闭式有限网格解平滑连接原生高频带与压缩后的低频尾部。

## 2. 最强贡献与证据

1. **问题分解清楚，固定 support 的识别实验很有价值。** 第 3.1 节（PDF 第 2–3 页）用 151.9M 模型、三组配对种子保持初始化、token 顺序、优化器、预算和两个频率端点一致，只改变 30 个内部坐标。保留训练 support 时，Cosh 在 2×/4×/8× 长度的三组种子中全部改善；把两种 allocation 都改为目标长度 support 后，排序又全部反转。附录 B.1（第 25–26 页）给出频率公式、数据、窗口和逐种子数值。这是本文最干净、最有说服力的因果证据：range 与 allocation 确实是不同设计变量，二者还会与训练后权重发生交互。

2. **理论对象选择正确，数学陈述总体严谨。** 第 4 节（第 4 页）使用完整旋转对的子空间 `span{cos(ωΔ), sin(ωΔ)}`，避免任意内容相位造成的偏置；block whitening 后的 canonical correlation 与 Rényi-2 有效秩恒等式直接说明有限窗口内的频率冗余。附录 A.1–A.2（第 15–18 页）给出交叉 Gram、慢频极限展开以及识别实验实际网格上的有限窗口计算。Corollary 2 的频谱论证也在其明确条件下成立，附录 A.6/I.3 说明了整数混叠、频率 multiset 与 slot assignment 的边界。没有发现影响主结论的明确数学错误。

3. **TailSpline 是简单、闭式、易部署的构造。** 第 5.2 节（第 6 页）把额外 log-span 写成 transition increments，并最小化相邻额外 gap 的变化及低频端接缝；Theorem 3 给出正增量的唯一闭式解。附录 H.1（第 50 页）用正定二次型和 KKT stationarity 完成证明。该方法不读取权重、激活或校准输出，不增加训练参数，并保持标准 rotary 运算，工程价值明确。

4. **冻结部署的主结果规模较大且报告透明。** 第 6.2 节（第 7–8 页）及附录 H.7（第 53–54 页）在 Llama-3-8B-Instruct 的 clean source-order RULER-200 上比较 2,600 个配对输入，TailSpline 相对 MrRoPE-Pro 提升 11.72 个百分点，配对区间为 [10.32, 13.11]，12/13 个任务均值获胜，leave-one-task-out 仍为正。作者保留 capped/empty 输出并报告 EOS、cap-hit、empty 数量，未通过结果筛选美化指标。附录 H.4 还给出 OLMo 上预先冻结方法和数据后的完整长度曲线，方向很强。

5. **论文主动呈现负结果和适用边界。** 自然 QA 上 TailSpline 与 MrPro 基本持平（第 8 页；附录 H.9），native-window 任务差异区间跨零而 LM PPL 有小幅但可检测的退化（附录 H.8），Cosh 在 target-matched support 下反而变差，Qwen 上旧的 BM 比较也会输。这样的结果没有削弱核心“allocation matters”结论，反而使实际主张更可信。

## 3. 决策相关的弱点

### 弱点 1：TailSpline 相对 MrPro 的主增益尚不能归因于其平滑边界目标

- **位置：** 第 5.2、6.2 节（PDF 第 6–8 页）；附录 H.2、H.8（第 50、55 页）。
- **具体证据：** 主比较保持权重、outer bands、端点、gain、输入和 decoder 一致，但 TailSpline 与 MrPro 同时改变 transition shape 和总 log-frequency displacement。作者构造了精确 equal-dose 控制 C；然而 T–C 的 Full-13 AUC 为 −0.41 pp，区间 [−2.63, 1.82]，且 T 用 batch 1、C 用 batch 2并重排 batch，缺少完整匹配的 runtime provenance。因此该对照既未显示 TailSpline shape 的优势，也不能作等价性结论。
- **附录中的反证/缓解：** 附录 G.8（第 49 页）在 OLMo 上用 BM–Uni 证明“总位移相同而 shape 不同”可以产生 19.20 pp 差异；这支持 shape 一般而言会重要，但不是 TailSpline 的 T–C 特异验证。论文也明确承认当前归因未决。
- **对实际主张的后果：** “TailSpline 作为一个完整静态频率表优于 MrPro”仍由实验直接支持；但“因为其特定的低频接缝平滑目标而优越”尚未建立。理论目标是一个可解释的边界先验，不是经实验验证的机制。
- **最小修复：** 在相同 batch、backend、revision 与输入顺序下完成 C arm，并把 T–C 设为明确的 shape 诊断；若仍无显著差异，收窄机制措辞，将 TailSpline 的价值表述为闭式完整 profile 的经验效果。

### 弱点 2：Cosh 的理论目标与实际任务收益之间仍主要是设计启发，而非推导链

- **位置：** 第 4.2、5.1、6.1 节（第 4–7 页）；附录 A.9–A.14、B.3（第 21–29 页）。
- **具体证据：** 有效秩分析使用真实 sine/cosine 子空间重叠，而 Cosh 的目标采用 `αρ² + βρρ min(ϕ,ψ)` 的连续密度先验；正文未从 canonical-overlap 指标推导该 kernel，也未证明最小化该目标会改善模型损失。τ 的参考规则依赖 diffuse attention、full-RoPE MHA、小 τ 等显式建模假设；附录 A.13 还指出在工作 τ 处局部近似会明显偏离精确 stiffness。固定范围 50.9M factorial 中，公式点仅在 7/12 配置改善，1.25× 为 10/12；matched exponential 与 Cosh 几乎相同（差 0.00074 NLL，区间跨零）。
- **附录中的反证/缓解：** 闭式密度最优解本身证明完整且严格；151.9M 三种子固定 support 和 432M MLA 三种子均给出一致的长度外收益，说明 Cosh 是有效候选，而非纯理论构造。
- **对实际主张的后果：** 证据支持“非均匀 allocation 和 Cosh 这个实例有用”，但尚不足以把 Cosh 的具体 cosh 形状或 τ 规则视为由 full-pair geometry 唯一或近似最优地导出。论文的理论解释力强于预测力。
- **最小修复：** 在正文明确标注从 overlap 分析到 density objective 是设计假设；把 matched exponential 的近似持平提前到主文，并用一张小表报告 geometry 指标、目标值与任务结果是否跨配置同向，无需提出普适最优性定理。

### 弱点 3：冻结部署的强主结果主要来自合成长上下文任务，跨自然任务的优势未出现

- **位置：** 第 6.2 节（第 7–8 页）；附录 H.3–H.9（第 51–56 页）。
- **具体证据：** 最大的 +11.72 pp 结果来自单个 Llama checkpoint、单一 32K source-order RULER 面板。自然 QA 的 631 个配对问题上，T–P 仅 +0.20 pp，区间 [−1.53, 1.89]；>8K 子集为 −0.94 pp，区间同样跨零。clean RULER 中 empty 输出也很多（TailSpline 240、MrPro 310），表明方法差异有一部分可能体现在生成稳定性/终止行为，而非所有任务能力都同步提高。
- **附录中的反证/缓解：** clean 面板覆盖 13 个任务、四个任务家族，12/13 任务均值为正；OLMo 的 classic 面板上增益更大且所有 task AUC 为正，PPL 也改善。所有 empty/capped 输出均计分，因此这不是排除失败样本造成的偏差。
- **对实际主张的后果：** “TailSpline 可显著改善冻结模型的 RULER 长上下文表现”证据很强；“提供广泛实用的长上下文改善”则应保持任务限定，因为自然 QA 没有显示优势。
- **最小修复：** 在摘要和结论中把实际收益明确限定为 RULER/检索型长上下文；将自然 QA 的空结果与 RULER 的大增益并列呈现。若增加实验，最有价值的是另一个自然长文生成或问答集合，而不是更多 RULER 变体。

### 弱点 4：学习侧的跨配置效应小且异质，较大的 Cosh 数值收益没有同时满足固定 support 控制

- **位置：** 第 3.1、6.1 节（第 2–3、6–7 页）；附录 B.1、B.3、E.1（第 25–29、35–36 页）。
- **具体证据：** 最干净的 151.9M 固定-support 实验在 native length 有 +0.026 NLL 成本，并在 retargeted support 下全面反转。50.9M factorial 的平均收益约 0.01 NLL，方向随配置和 strength 变化。较醒目的 432M MLA 结果（16K PPL 138.8→95.6）使用 unanchored midpoint tables，因此 allocation 与实际 sampled endpoints 都变化；它证明完整 Cosh recipe 的效用，但不能单独归因于内部 allocation。
- **附录中的反证/缓解：** 三种子固定-support 结果在所有外推长度方向一致，且 factorial 覆盖 12 个结构配置；论文的 Table 1 已明确区分 identification 与 utility 两类协议，没有把它们错误合并。
- **对实际主张的后果：** “allocation 是独立变量”成立；“Cosh 通常能带来大幅学习外推收益”仍依赖具体 support policy、架构和 strength。贡献显著性更像可靠的新设计维度与一个强实例，而不是稳定的通用增益配方。
- **最小修复：** 在主文增加一行按协议区分效应量：固定-support identification、完整 Cosh recipe、retargeted failure；并将结论中的 Cosh 表述保持为条件性收益。

## 4. 可选建议

- 主文已经较清楚，但 47 页附录混合当前主证据、历史探索和多个旧 profile。可在补充材料开头再加一张“主结论所需的最短证据路径”，并把历史探索明确标为 archival；现有 Table 3 已接近这个目标。
- Figure 4 的三个 panel 使用不同模型、输入选择与长度协议，视觉上容易形成统一 benchmark 的印象。可在各 panel 标题直接写入 `clean source-order`、`classic padded` 及模型名。
- 报告 native trade-off 时，可优先给出绝对差、配对区间和 PPL 差；“2.33% relative reduction”容易让一个区间跨零的点估计显得比其不确定性更确定。
- 可公开一个最小构造伪代码框，直接从 native frequencies、native window 和 s 输出 TailSpline table，帮助读者在不穿越长附录的情况下复现公式。

## 5. 总体判断

**推荐：接受（偏强的 weak accept / accept）**  
**内部评分：7/10**  
**置信度：4/5**

决定性理由是：论文提出了一个清楚且此前常被 base/range 掩盖的设计变量，并用非常干净的固定端点配对实验建立其独立作用；完整 sine/cosine 子空间分析和换基边界在数学上扎实；TailSpline 是闭式、零训练、易部署的实际方法，且在大规模配对 RULER 面板上有强而透明的结果。作者还系统报告了反转、native cost、自然 QA 持平和旧 profile 失败，论证边界可信。

保留意见主要影响主张范围，而非推翻核心贡献：TailSpline 对 MrPro 的优势尚不能归因于其特定平滑目标，Cosh 的 density objective 与 full-pair overlap 之间是启发式桥梁，且冻结部署在自然 QA 上没有显示收益。只要论文把机制与跨任务泛化措辞限定在证据所支持的范围内，我认为其新颖性、技术完整性和实验价值足以接受。

## 实际检查材料

我阅读了 PDF 的完整主文（第 1–10 页，包括摘要、方法、实验、相关工作、结论及 reproducibility/AI-use statements），检查了参考文献与补充材料导读，并重点逐段核查附录 A（完整旋转对几何、有效秩、频谱/换基判据、Cosh 变分解与 τ 假设）、B（固定 support 配对训练和 factorial）、D/E/F（训练与成熟模型协议）、G/H（冻结 profile、TailSpline 证明、完整 RULER 曲线、equal-dose/native 对照、自然 QA）、I（range/table/slot 干预）和 J（部署区间与探索性诊断）。我将全部 61 页渲染后逐页检查了图、表、公式和版式；未发现明显裁切、重叠、乱码或不可读图表。
