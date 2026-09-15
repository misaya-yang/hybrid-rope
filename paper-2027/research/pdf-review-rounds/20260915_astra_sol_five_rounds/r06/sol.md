# ICLR 2027 模拟评审组：Beyond the Base: Frequency Allocation in RoPE

## 审阅范围与材料

本评审只使用冻结稿 `input.pdf`。我通读了主文第 1–14 页（摘要、§1–§8、伦理/复现声明及参考文献），并核查了与主要论断直接相关的附录：A（完整正余弦对子空间、有限窗口秩、核等价边界）、B（固定端点训练、weights-by-table crossing、50.9M factorial）、D/E（Cosh 构造与 432M MLA 实验）、F（冻结固定 support、适配实验）、G（BM、自然 QA、等总位移控制）、H（TailSpline 证明、等位移控制、Llama/OLMo、clean RULER、原生窗口与自然 QA）、I（频谱与 slot assignment）、J（部署区间目标及探索性诊断）。另外渲染并目视检查了主文含图表和公式的关键页，以及附录 B、H 的关键实验/证明页。未读取仓库文件、外部文献、其他评审或作者对话，也未浏览网络。

版式总体成熟：正文、公式、图表和附录交叉引用清晰；图 1–4、表 1–2 在打印尺寸下可读，未见裁切、重叠、乱码或公式排版错误。附录非常长（全文 62 页），但导航表 3–4 有效降低了查证成本。

---

## R1：新颖性与意义

### 核心贡献（我的理解）

论文把 RoPE 的“support（频率端点/范围）”与“allocation（固定端点内的内部频率位置）”明确分离，并通过固定端点的训练与冻结干预说明 allocation 本身可以改变模型质量。随后给出不读取权重、激活或校准输出的闭式冻结构造 TailSpline，以及用于训练研究的 Cosh transport。这个组合贡献的价值在于：它把常被 base/scaling 掩盖的一个可控维度变成了可分析、可安装、可比较的设计对象。

### 最强证据

最强的识别证据是 §3.1 / 附录 B.1 的三 seed、151.9M 配对训练：训练 support 固定时，Cosh 在 2×/4×/8× 的三个 seed 全部优于 Geo，而将两者 support 改为 evaluation-length-matched 后排序在三个 seed 全部反转。这直接说明“内部位置”与“范围”是不同且相互作用的变量。最强的方法证据是 §6.1 / 附录 H.7、H.10：冻结 Llama-3-8B-Instruct 上，单一静态 s=4 表在 clean RULER Full-13 的 16K 和 32K 分别提高 3.39 和 11.72 个百分点，配对区间均为正；32K 有 2600 对提示、12/13 个任务均值获胜。附录 H.4 的 OLMo 迁移结果（Full-13 AUC +49.23 点）进一步显示公式并非只在单一 checkpoint 上有效。

### 重要关切

1. **TailSpline 的方法级收益与“平滑低频尾连接”这一具体机制尚未完全分离。**
   - 位置：PDF 第 5–7 页，§5.1、§6.1；附录 H.2、H.8（第 50、55 页）。
   - 受影响的实际论断：TailSpline 通过平滑过渡到低频尾部而带来所报告的任务收益。
   - 证据：TailSpline–MrPro 保持外带、端点、gain、输入和 decoder 一致，但同时改变总位移和 transition shape。等总位移的 T–C 诊断在 Full-13 AUC 上是 −0.41 点，95% 区间 [−2.63, 1.82]，且 T/C 分别用 batch 1/2、批次顺序不同，论文自己称残余 shape 排序未解决。
   - 反证/附录已有回答：这不削弱“allocation 整体有用”；附录 G.7/G.8 还给出了其他 profile 在等总位移下可产生较大差异的证据。作者也在第 5 页明确说 T–P 同时改变 dose 与 shape。
   - 最小修复：把摘要/引言中可能让人读成机制归因的“smooths ... tail”与性能句进一步分开；明确 TailSpline 是由该正则目标导出的有效整体 allocation，而当前数据未确认收益由 tail-smoothing 单独介导。
   - 分类：**解释性问题**。

2. **从 PDF 单独无法可靠判断相对既有工作的全球新颖性。**
   - 位置：PDF 第 8 页 §7，尤其与 MrRoPE、YaRN、LeRoPE、LongRoPE/2、STRING/GRAPE 的关系。
   - 受影响的实际论断：固定端点识别、完整 pair 几何及闭式 tail transition 的组合是新的。
   - 证据：论文清楚描述了自己的差异：MrRoPE-Pro 是最近的冻结 comparator；LeRoPE 学习频率；STRING/GRAPE 讨论群结构/生成元；本文强调 actual-endpoint controls、phase-invariant full-pair examples 与闭式 transition。
   - 反证/附录已有回答：相关工作覆盖面较广，且论文没有提出 SOTA 或普适最优性。我的限制是不能查阅被引文献来验证优先权和重叠程度。
   - 最小修复：在 related work 中加入一个紧凑对照表，逐项标明既有方法是否固定实际端点、是否闭式、是否零训练、是否分析完整 sin/cos pair；这能让新颖性主张在稿内自足。
   - 分类：**解释性问题**。

3. **实用意义目前最可靠地成立于“强 synthetic RULER 改善 + 很小/不确定的 natural-QA 差异”，而不是广泛下游任务提升。**
   - 位置：PDF 第 7–9 页，§6.1、§8；附录 H.9（第 55–56 页）。
   - 受影响的实际论断：TailSpline 是提高目标窗口内“model quality”的实用设计。
   - 证据：clean RULER 增益大且稳定；但 631 个自然 QA 的 task-equal F1 为 41.08/40.88，差 +0.20，区间 [−1.53, 1.89]；>8K 子集点估计反而为 −0.94，区间跨零。
   - 反证/附录已有回答：论文准确报告“close observed F1”，没有把它写成自然 QA 胜利；原生 8K 任务差 −2.14 点的区间也跨零，PPL 只升 0.37%。因此这是外部效度边界，不是推翻核心结果。
   - 最小修复：在摘要最后一句和结论中把 practical value 明确限定为“synthetic long-context task quality and allocation as a deployable knob”，同时把 natural QA 的无明显差异放在与 RULER 增益同一段的醒目位置。
   - 分类：**解释性问题**。

### 推荐

**Weak Accept。内部评分：7/10；置信度：4/5。**

决定性理由是：固定端点识别是干净且有概念价值的，TailSpline 是简单可部署的闭式方法，Llama clean RULER 的两长度结果和 OLMo transfer 都很强。保留意见主要限制机制归因和任务外推，不否定“allocation 是独立而实用的 RoPE 设计维度”。新颖性评分受限于不能查外部文献。

---

## R2：理论与方法

### 核心贡献（我的理解）

理论部分给出完整 rotary pair 的 phase-invariant overlap：以两个二维子空间的 canonical correlations 定义冗余，并把平均 pairwise overlap 与 block-whitened Gram 的 Rényi-2 effective rank 精确联系起来。它证明慢频率在有限窗口内趋向共享的 `span{1, Δ}`，给出仅看 cosine 会反转 allocation 排序的反例，并用旋转谱说明真正改变频率 multiset 不能被固定 Q/K 基变换吸收。方法上，TailSpline 是一个有限网格上的严格凸二次问题，其唯一正增量闭式解平滑低频尾部 junction。

### 最强证据

§4 / 附录 A 的数学链条较完整：cross-Gram 有显式积分式；block whitening 保证 pair 内基变换不变性；有效秩恒等式由 block Gram 的二阶迹直接推出；慢频极限在附录给出四阶展开和显式系数。§5.1 / 附录 H.1 对 TailSpline 给出正定性、KKT/站立条件、正性和闭式累积 profile，且清楚说明高频入口未惩罚是有意的边界选择。Corollary 2 的 d=0,1 论证也在其条件（频率位于 (0,π)、对所有 content vector 的精确核）下成立。

### 重要关切

1. **几何量是结构描述，不是任务损失的充分预测器；主文有时把二者连接得过快。**
   - 位置：PDF 第 4–5 页，§4.2 的 “Design implication” 与 §5.2；附录 A.1、A.3、J.4–J.6。
   - 受影响的实际论断：降低 full-pair overlap / 更有效地“spend positional budget”解释有用 allocation。
   - 证据：effective rank 明确依赖声明的 separation measure，并经过 block whitening；它忽略 raw feature scale、内容系数、softmax key competition 与模型读出。附录 A.1 已说 raw energy、coefficient magnitude 和 numerical precision 仍决定方向贡献；附录 J.6 还给出 band-count 增加却使连续文本 NLL 全部恶化的反例。
   - 反证/附录已有回答：论文没有把几何指标称为任务最优目标，§4.2 也说模型学习如何与 content 组合，§5.1 明确“task value is tested below”。
   - 最小修复：在 §4.2 的 design implication 加一句明确否定充分性：该秩只刻画给定 separation prior 下的可区分方向，不能单独排序 checkpoint/task 上的 allocation；并将 J.6 的反例前置引用。
   - 分类：**解释性问题**。

2. **TailSpline 的边界目标有良好闭式性质，但为什么应只惩罚低频 junction 而不惩罚高频入口，仍是设计先验而非推导出的任务原则。**
   - 位置：PDF 第 5 页 §5.1，Eq. (7)；附录 H.1（第 50 页）。
   - 受影响的实际论断：该目标是适合冻结原生表的 principled transition。
   - 证据：目标省略入口惩罚，所以 TailSpline 明确接受更大的 entry jump 换取更小 terminal jump；对称地加入入口惩罚就得到 BM。数学只证明各自优化所声明的目标，不证明低频边界优先适合所有 checkpoint。
   - 反证/附录已有回答：作者表述谨慎，称“a useful boundary prior need not minimize both”，并在两个不同模型上取得正面任务结果；因此方法是有动机的先验，不是理论错误。
   - 最小修复：在主文明确将“preserving high-frequency outer band”与“入口不惩罚”区分开；前者保持外带，后者仍产生较大首个额外 gap。补一句说明这是经实验检验的 asymmetric prior。
   - 分类：**解释性问题**。

3. **完整 pair overlap 与最终 TailSpline objective 之间没有直接优化桥梁。**
   - 位置：PDF 第 4–6 页，§4 到 §5 的过渡。
   - 受影响的实际论断：几何分析“guides”显式 TailSpline 构造。
   - 证据：TailSpline 优化的是 adjacent log-gap variation 与 terminal boundary penalty，而不是 §4 的 canonical overlap、effective rank 或 softmax-aware geometry；Cosh 的 density objective 与慢频集中更直接相关。
   - 反证/附录已有回答：论文实际将 TailSpline 描述为 native-relative frozen construction，把 Cosh 称为由 finite-window overlap 动机驱动的 supporting transport；并未声称 TailSpline 是 effective-rank 最优解。
   - 最小修复：在 §5 开头明确分成两条设计路径：full-pair geometry 直接启发 Cosh 的频率密度，而 TailSpline 由 frozen-checkpoint continuity 需求导出。避免让统一叙事暗示不存在的定理链。
   - 分类：**解释性问题**。

### 推荐

**Weak Accept。内部评分：7/10；置信度：4/5。**

决定性理由是数学对象定义得当、边界和反例交代充分，闭式构造证明完整；没有发现已成立的数学错误。主要不足是“结构量 → 方法目标 → 任务收益”的桥梁更多是动机和经验验证，而不是一个统一的预测理论。论文若更明确地区分刻画、先验和实证，会显著增强可信度。

---

## R3：实验与实践价值

### 核心贡献（我的理解）

实验贡献不是单一 benchmark 分数，而是一组不同识别合同：固定 support 的 scratch training 识别 z；冻结 mature checkpoints 证明相同端点下 allocation 仍重要；crossed tables 和 slot permutation 证明学习到的 coordinate association；TailSpline–MrPro 则测试一个无需训练的可安装方法。论文反复区分 paired rows、seed、configuration、task 和 document 作为统计单位，这一点优于许多长上下文工作。

### 最强证据

最强实证是 clean RULER 的预处理与计分合同：32K 使用上游 source-order 前 200 行/任务，无内容 padding、无深度选择，实际输入 28,270–32,606 tokens，2600 对提示，official scorer，task-equal primary endpoint；+11.72pp 区间 [10.32,13.11]，leave-one-task-out 仍为正。16K 独立面板的 +3.39pp 区间 [1.53,5.34] 说明不是只在 horizon 有效。附录 H.8 同时诚实报告原生 PPL 的 0.37% 上升与原生 RULER 点估计 −2.14pp。432M、三 seed、500M token/arm 的 Cosh MLA 结果也很有分量：16K PPL 138.8→95.6，三个 seed 同向。

### 重要关切

1. **TailSpline 的主清洁比较只覆盖一个 8B checkpoint 和一个最近邻 baseline；对更广泛冻结部署的相对优势仍有限。**
   - 位置：PDF 第 6–8 页 §6.1；附录 H.4、H.7、H.10。
   - 受影响的实际论断：TailSpline 是普遍有竞争力的 frozen extension allocation。
   - 证据：Llama clean 16K/32K 的直接 comparator 是 MrPro。OLMo 上 TailSpline 极强，但仍主要是同一 comparator；YaRN/Uni 在附录 G.8 的六任务小 follow-up 上出现，而没有出现在 Llama clean Full-13 主面板。论文也未在 clean panel 比较 LongRoPE2 等需训练/搜索的方法，这些合同本来不同。
   - 反证/附录已有回答：论文没有声称 SOTA，也解释了为何 MrRoPE-Pro 是共享 outer bands 和 gain 的最近冻结 comparator；OLMo formula transfer 是实质性的第二模型支持。
   - 最小修复：无需扩大为大网格；在现有 Llama clean 16K/32K 输入上补一个 widely used frozen baseline（例如稿内已有 official YaRN 实现）即可显著校准实际效应大小。若无法补实验，则把“strong paired gain over nearest comparator”保持为主要范围。
   - 分类：**可选扩展**。

2. **自然 QA 没有显示 TailSpline 相对 MrPro 的实际提升，限制了 RULER 结果的应用外推。**
   - 位置：PDF 第 7 页 §6.1；附录 H.9。
   - 受影响的实际论断：synthetic retrieval/QA 的提升会转化为自然长上下文问答。
   - 证据：整体 +0.20pp、CI 跨两种排序；>8K 为 −0.94pp；五个任务中只有 2Wiki 和 MultiFieldQA 点估计为正。document-equal sensitivity 几乎完全相等（−0.023pp）。
   - 反证/附录已有回答：稿件没有提出上述转化主张，反而明确写出 natural QA close；全部输出健康、cluster bootstrap 合理。BM 的不同 OLMo 自然 QA 实验是另一构造，不能用来替 TailSpline 声称成功。
   - 最小修复：主文把自然 QA 表 41 的三行（总体、≤8K、>8K）浓缩成一张小表，而不只给总体；这会让边界更透明。将后续扩大自然任务覆盖列为未来工作即可。
   - 分类：**解释性问题**。

3. **若把 equal-dose T/C 用作 residual shape 的实验证据，运行时不匹配使它不足以完成归因。**
   - 位置：PDF 第 5 页 §5.1；附录 H.8。
   - 受影响的实际论断：TailSpline 相对等总位移 control 的剩余 shape 效应。
   - 证据：点估计 −0.41pp 且区间跨零；batch size、批序和 provenance 不同；E0 replay 即使 official score 不变，仍有 token sequence/decoded output 改变。作者明确说需要完整 matched C arm 才能加强归因。
   - 反证/附录已有回答：当前稿件实际上称排序 unresolved，并未据此宣称 shape 胜出；更广泛的等总位移 BM–Uni 和 A/B profile 实验只证明“shape 可重要”，不是 TailSpline 特定 shape 胜利。
   - 最小修复：将主文这一结果标成 robustness diagnostic，而非 “test the remaining shape difference” 后可能被误读为已完成检验；若资源允许，唯一高价值补跑就是在同 batch/backend 下重跑 C。
   - 分类：**解释性问题**。

### 推荐

**Weak Accept。内部评分：6/10；置信度：4/5。**

决定性理由是主 clean RULER 结果规模大、配对严谨、两长度一致，并有 OLMo transfer 和多条训练证据；实验报告对负结果和 protocol 差异非常透明。扣分来自自然 QA 无提升、主方法 comparator 范围较窄，以及 TailSpline 特定 shape 的等 dose 归因未解决。它仍建立了实践上值得知道的结论：在零训练、同端点、同 gain 下，仅换内部频率表就能显著改变长上下文任务表现。

---

## R4：整体论证与叙事

### 核心贡献（我的理解）

整篇论文试图建立三层论证：第一，allocation 是独立变量；第二，它改变有限窗口中的 positional subspace 与 learned coordinate compatibility；第三，一个由部署边界条件导出的静态 allocation 可以带来实际收益。论文的真正中心不是 Cosh 或 TailSpline 单独取胜，而是把 allocation 从 base/scaling 的附属细节提升为可识别、可解释、可部署的设计维度。

### 最强证据

论证最强之处是不同实验合同没有混算：表 1、表 3、表 4 明确区分 fixed support、learning utility、frozen deployment、same-dose diagnostic、slot swap 和 Native comparison。正文也保留了关键反向结果：support retarget 后 Cosh 排序反转；TailSpline natural QA 基本持平；native window 有小成本；equal-dose 不能定序。这些反例没有破坏中心论点，反而使“allocation 与 operating setting 相互作用”的叙事更可信。

### 重要关切

1. **论文同时承载“识别科学”“几何理论”“Cosh 训练”“TailSpline 部署”四条线，主贡献层级在阅读中偶尔漂移。**
   - 位置：PDF 第 1 页贡献列表、§3–§6、§8。
   - 受影响的实际论断：读者应把哪项视为首要贡献，哪些是 supporting evidence。
   - 证据：摘要先从 allocation identification 转向 full-pair geometry，再称 learned-coordinate effects “guide” 修改 pretrained table，随后引入 TailSpline 和 Cosh；但 TailSpline 并非由 learned-coordinate crossing 直接推导，Cosh 也不是主部署方法。
   - 反证/附录已有回答：§5 开头称 TailSpline 为 main frozen construction、Cosh 为 complementary transport；表 3–4 的附录导航非常好。
   - 最小修复：在引言末加一句显式层级：“中心科学结论是 allocation 的独立作用；TailSpline 是主要部署实例；Cosh 是训练侧的独立佐证；几何给出结构刻画而非性能预测器。”相应压缩 §6.2 的枝节。
   - 分类：**解释性问题**。

2. **“learned-coordinate effects guide how a pretrained model’s table can be modified” 的措辞强于所展示的逻辑。**
   - 位置：PDF 第 1 页摘要、第 3 页 §3.2、第 4 页 §4.3、第 57–58 页附录 I。
   - 受影响的实际论断：crossing/slot permutation 为 TailSpline 的具体修改规则提供指导。
   - 证据：这些实验有力证明 frozen model 对 frequency-to-coordinate association 敏感；但它们只说明不能把训练所得 allocation 当成可互换表，并未选择 TailSpline 的 asymmetric spline profile 或 32/1-turn band。
   - 反证/附录已有回答：附录 I 正确把 spectrum、table crossing、slot assignment 分成三个不同干预；其结论是 frozen extension 必须尊重已有 association，这对 native-relative 设计确有一般性约束。
   - 最小修复：把 “guide how ... modified” 改为 “constrain post-hoc modification by showing that learned coordinate assignment matters”；这更精确且不削弱贡献。
   - 分类：**不受支持的表述**。

3. **附录极其完整，但主文对最关键限制的显著性仍可改善。**
   - 位置：PDF 第 5 页等 dose 一段、第 7 页 natural QA / native trade-off、第 8–9 页结论。
   - 受影响的实际论断：读者能否在不深挖 48 页附录的情况下正确限定贡献。
   - 证据：正文确实报告所有关键数值，但自然 QA 的 >8K 分层、T/C runtime mismatch，以及 clean Llama 尚未对比 YaRN，只在附录给出或明示。
   - 反证/附录已有回答：正文已经比常见稿件透明，未隐藏 null 结果；表 1 也明确 “T–C: dose too”。
   - 最小修复：在 §8 增加三句 limitations：任务迁移未证实、TailSpline 特定 shape 机制未隔离、最强 clean 结果只相对 MrPro。无需新增整节。
   - 分类：**解释性问题**。

### 推荐

**Accept。内部评分：7/10；置信度：4/5。**

决定性理由是论证总体自洽，并且主动呈现反向与无效结果，避免了把不同合同拼成一个过度结论。主线稍拥挤、个别连接词暗示了比证据更强的因果指导，但通过小幅重写即可修复，不需要改变核心方法或结果。

---

## AC Meta-review

### 分歧调和与事实复核

四个视角在核心事实上一致：论文可靠建立了“固定频率端点不固定模型行为，内部 allocation 是独立设计变量”；TailSpline 在冻结 Llama clean RULER 的 2L/4L 上分别有 +3.39/+11.72pp 的显著配对收益，并在 OLMo 上有强 transfer；同时 TailSpline 在冻结 natural QA 上没有可辨别优势。

我针对潜在争议重新核对 PDF 后，排除以下误读：

- **不是数学错误：** Corollary 2 只讨论对所有 content vector 的精确 bilinear kernel，并限定频率在 (0,π)；附录 A 处理 aliasing/连续区间边界。它不能被批评为声称“任意学习网络都不能适应频率变化”。
- **不是隐藏的自然 QA 失败：** 正文和附录都明确给出 +0.20pp、区间跨零，并称两者 close；论文没有声称自然 QA 获胜。
- **不是已经完成的 TailSpline shape 因果证明：** 正文说 T–C 排序 unresolved，附录 H.8 明示 runtime mismatch。因此应修正的是叙事归因，而不是判定结果造假或统计错误。
- **不是 SOTA 要求：** 论文的实际主张是相对最近的 frozen MrPro，在固定 support/gain/inputs 下证明 allocation 与一个有效构造；缺少所有 long-context baseline 是可选扩展，不构成已成立错误。
- **不是 universal optimum：** support-retarget 后排序反转、不同模型/任务的 trade-off 以及 J.6 的反例都被主动报告；论文的结论是 allocation consequential and setting-dependent。

### 决策相关问题

决策相关的限制有两项。第一，TailSpline 的主方法收益是“整体 profile 对 MrPro 的收益”，尚不能归因于其低频 junction smoothing 本身；摘要和过渡段应避免机制性暗示。第二，实践价值的外部效度目前主要来自 synthetic RULER 与 PPL，natural QA 对 TailSpline 是持平结果。它们限制结论范围，但不推翻中心贡献。

其余问题属于改进项：在同一 clean Llama 面板补一个 YaRN 等常用 frozen baseline 会增强校准，但不是接受前提；扩大模型、任务或更长长度也属于后续工作。基于 PDF 单独不能验证相关工作中的优先权，因此 novelty 判断仅限稿件自述及稿内比较，不能断言其对全部外部文献绝对新颖。

### 论文建立的新知识与实践价值

论文建立了三点可迁移的新知识：

1. 在实际 sampled endpoints 和 log-span 相同的情况下，内部频率位置仍能系统改变 learned extrapolation 与 frozen checkpoint 的质量；因此只报告 base/range 不足以描述 RoPE 设计。
2. 完整 sin/cos pair 的二维子空间是比 cosine-only proxy 更稳健的有限窗口分析对象；慢频 pair 可以占用许多 nominal coordinates 却只提供接近二维的 block-whitened effective rank。
3. 一个 O(K)、无训练、无校准、保持标准 rotary operator 的静态表，可以在两个冻结模型上显著改变目标窗口任务质量；这使 allocation 成为实际工程旋钮，而不只是理论自由度。

实践价值最强的形式是：对无法或不愿重训模型的用户，TailSpline 给出可直接安装、可复现的候选表，并展示了与 MrPro 的大幅 clean RULER 改善。其已证实价值不包括普遍自然 QA 提升、所有 checkpoint 最优或平滑机制的单独因果证明。

### 总体建议

**总体推荐：Weak Accept。AC 内部评分：7/10；置信度：4/5。**

这不是四个分数的算术平均。决定性依据是中心识别问题重要、控制设计扎实、理论刻画有独立价值、闭式方法简单且有两模型实证；作者对 null、反转和 protocol 边界也异常透明。主要缺陷可由收窄机制与任务迁移措辞来修复，而无需补做决定性实验。若会场对纯方法新颖性的门槛很高，外部文献核验可能改变 novelty 判断；在只看 PDF 的条件下，我倾向接收。

### 最高价值的稿件修改

1. 在摘要、§5–§6 过渡和结论中明确区分：allocation 整体有效；TailSpline 由 asymmetric smoothing objective 导出；当前 equal-dose 数据未证明收益由该 smoothing 单独介导。
2. 将自然 QA 的总体与 >8K 分层结果前置到主文，并将 practical claim 限定为已测的 synthetic long-context quality / PPL / frozen deployability。
3. 重写 “learned-coordinate effects guide...” 为“coordinate association constrains post-hoc modification”，并明确 crossing/slot experiments 不导出 TailSpline 公式。
4. 用一张很小的稿内相关工作属性表支持 novelty 自足性；若只允许一个新实验，优先在现有 clean Llama 输入上跑一个 common frozen baseline，而不是扩展大规模网格。

### 评分汇总

| 视角 | 推荐 | 内部评分 | 置信度 | 决定性理由 |
|---|---|---:|---:|---|
| R1 新颖性与意义 | Weak Accept | 7/10 | 4/5 | 固定端点识别与闭式冻结方法有意义；外部优先权无法仅凭 PDF 确认 |
| R2 理论与方法 | Weak Accept | 7/10 | 4/5 | 数学对象和证明扎实；几何到任务与 TailSpline 目标的桥梁主要是动机 |
| R3 实验与实践 | Weak Accept | 6/10 | 4/5 | clean RULER 强且透明；自然 QA 持平、主 comparator 较窄 |
| R4 整体论证 | Accept | 7/10 | 4/5 | 论证自洽并主动呈现反例；个别连接措辞过强 |
| AC | Weak Accept | 7/10 | 4/5 | 核心知识增量成立，限制主要影响范围和机制归因，不推翻贡献 |
