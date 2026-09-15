# ICLR 2027 模拟评审组意见：Beyond the Base: Frequency Allocation in RoPE

## 审阅范围与证据边界

本评审唯一的实质来源是冻结文件 `input.pdf`。我通读了 62 页全文：主文第 1–14 页（含参考文献起始部分），补充材料第 15–62 页；逐项核对了主文公式 (1)–(9)、定理/命题、表 1–2、图 1–4，并查阅了与关键主张直接相关的附录 A（完整正弦–余弦子空间、慢频塌缩、核等价、Cosh 目标）、B（固定支撑配对训练及 factorial）、E–F（学习与成熟模型协议）、G（冻结 profile、自然 QA、等总位移控制）、H（TailSpline 证明、等 dose 控制、clean RULER、原生窗口和自然 QA）、I（range/table/slot 干预）及 J（部署区间与探索性诊断）。我还渲染并视觉检查了全文页面，重点放大检查了首页、主文图表与公式密集页、补充材料导航页、几何图、clean RULER、equal-dose、Natural-QA 和末页。没有读取仓库文件、外部文献、其他评审或作者对话，也没有联网。因此，下述“新颖性”只评价论文在 PDF 内建立的区别与定位，无法独立确认对全部同期工作的优先权。

---

## R1：新颖性与意义

### 中心贡献（我的理解）

论文把 RoPE 的“频率区间”与“区间内频率如何分配”明确拆成两个设计自由度，并通过固定端点的配对训练和冻结模型干预，证明后者会独立影响模型表现。它进一步给出完整 sine–cosine pair 的有限窗口重叠度量，以及无需训练、无需校准的闭式冻结方案 TailSpline。对我而言，最有价值的新知识不是“又一种 scaling 曲线”，而是：相同频率端点不等于相同位置基底；allocation 本身既可被训练适配，也会受预训练坐标–频率关联约束。

### 最强证据

最强识别证据是第 2–3 页 §3.1 / 附录 B.1：151.9M 模型三组配对种子保持端点、初始化、token 顺序、优化器和约 5 亿 token 预算一致，仅改变 30 个内部坐标；固定训练支撑时，Cosh 在所有种子的所有外推长度上均改善，而把两者支撑一起 retarget 后排序整体反转。这既表明 allocation 有效，也诚实地显示其效果依赖支撑策略。实际价值方面，第 7 页 §6.1 / 附录 H.7、H.10 的 clean RULER 对照最有说服力：Llama-3-8B-Instruct 上同一静态表、同权重、同端点、同增益和同输入，TailSpline 在 16K/32K 分别提高 3.39/11.72 个百分点，区间均不跨零；32K 有 2,600 对提示并覆盖 13 个任务。

### 重要关切

1. **新颖性定位仍可更尖锐（解释问题）**  
   **位置：**第 1 页 §1、第 8–9 页 §7，以及第 15 页补充材料导航。  
   **受影响主张：**“识别 allocation 的独立价值”及其相对于 MrRoPE、YaRN、LeRoPE、LongRoPE/2、p-RoPE 的概念增量。  
   **具体证据：**论文清楚说明最近冻结 comparator 是 MrRoPE-Pro，也指出既有工作已有 scaling、搜索和学习频率；但主文没有用一个紧凑的属性表说明哪些既有方法改变 support、interior allocation、坐标赋值、需要训练/搜索/校准，以及本稿第一次严格识别了哪一格。  
   **反证/已有回答：**§7 已逐类讨论相关工作，表 1 和补充表 4 也很好地区分本文内部协议；因此不是缺少引用，更不是已证实的新颖性错误。  
   **最小修复：**在 §7 增加一张小表，按“固定端点识别 / frozen closed-form / calibration-free / learned”四列定位最接近工作，并把新颖性句子限定为 PDF 实际证明的层级。

2. **TailSpline 的正面结果与“allocation 普遍重要”的正面意义强，但两者容易在摘要中被合并阅读（解释问题）**  
   **位置：**第 1 页摘要与贡献列表，第 5–8 页 §5–6。  
   **受影响主张：**TailSpline 的设计价值是否来自其特定 spline 目标，还是更一般的 frequency redistribution。  
   **具体证据：**TailSpline–MrPro 主对照同时改变总位移和 transition shape（第 5 页 §5.1 明说）；等 dose 的 T–C 在附录 H.8 为 −0.41 pp，区间 [−2.63, 1.82]，而且 T/C 分别使用 batch 1/2。  
   **反证/已有回答：**作者没有声称该对照证实 TailSpline shape 胜出；主文第 5 页和附录 H.8 已明确称其 unresolved。另有 BM–Uni 与构造 A/B 在固定总位移下显示 profile shape 可以影响得分（附录 G.7–G.8），支持一般性主张。  
   **最小修复：**在摘要中把“TailSpline beats MrPro”与“形状超越 dose 的一般证据”拆成两句，并明确 TailSpline 特定形状的增益归因尚未分离。

3. **从两类模型和有限任务形成了有意义的正结果，但“practical dimension”应保持条件化（可选扩展）**  
   **位置：**第 8–9 页 §6.1、§8，附录 H.4、H.9。  
   **受影响主张：**静态 allocation 对广泛实际长上下文任务的可迁移价值。  
   **具体证据：**clean 主确认只在 Llama 的 16K/32K RULER；OLMo 是 classic padded/depth-selected 面板；Llama Natural-QA 总体 +0.20 pp，区间跨越两种排序，>8K 子集点估计 −0.94 pp。  
   **反证/已有回答：**论文明确报告自然 QA 基本持平，没有宣称普遍改进；OLMo、Qwen 及学习实验也提供跨设置佐证。  
   **最小修复：**若篇幅允许，增加第二个模型上的 clean source-order 主面板；否则维持当前条件化措辞即可。

### 推荐

**推荐：Weak Accept。内部评分：7/10；信心：4/5。** 关键的新知识被控制实验可靠识别，且冻结部署有规模足够的正结果。主要限制是 TailSpline 的特定形状归因未闭合、实际收益在自然 QA 上不明显，但论文对此大体诚实，未把可选扩展冒充为接受前提。

---

## R2：理论与方法

### 中心贡献（我的理解）

论文给 allocation 一个干净参数化：端点/跨度是 support，归一化内部坐标是 allocation。理论部分用每个 rotary pair 的二维函数空间、canonical correlations 和 block-whitened Gram 的 Rényi-2 effective rank 描述有限窗口冗余；证明慢频 pair 趋于共享 `span{1, Δ}`，并用旋转谱说明不同频率 multiset 不能被一个固定的 Q/K 基变换普遍吸收。TailSpline 则是一个明确的有限网格二次优化，其闭式解平滑低频尾部连接。

### 最强证据

第 4 页 §4 与附录 A.1–A.6 的数学链条最强：完整 pair 避免 cosine-only 代理的排序反转，式 (5) 的 effective-rank identity 是直接代数恒等式，命题 1 给出慢频子空间的四阶收敛，Corollary 2 的假设和 aliasing 边界也被明确限定。TailSpline 方面，第 5 页定理 3 与附录 H.1 通过正定三对角二次型给出唯一正增量解，闭式公式和边界行为清楚且可复算。

### 重要关切

1. **几何指标是结构描述，不是已验证的任务预测器（解释问题）**  
   **位置：**第 4 页 §4.1–4.2，附录 A.1–A.4、J.4–J.6。  
   **受影响主张：**“full sine–cosine geometry characterizes the positional overlap changed by allocation”可以成立；若读成“该 rank 解释或预测 TailSpline 的任务增益”则证据不足。  
   **具体证据：**block whitening 刻意去除了 slow pair 的 within-pair scale；附录 A.1 同时承认 raw feature energy、系数大小和数值精度决定其能否影响模型。J.6 的探索性 band-count 在同一 12 profiles 上拟合，后来增加 count 的连续文本修改反而全部恶化 NLL。  
   **反证/已有回答：**作者明确称 J.6 为 descriptive fit，并明确拒绝“count 保证降低自然文本 loss”；H.6 也说没有完成机制干预。因此这不是理论错误。  
   **最小修复：**在 §4 的 design implication 末尾加一句：该指标刻画可用位置方向的重叠，但不单独预测任务效用；把与 TailSpline 的关系称为设计动机而非机制解释。

2. **TailSpline 目标的边界选择是可解释先验，但尚非由前述重叠理论推导（解释问题）**  
   **位置：**第 5 页 §5.1 式 (7)，附录 H.1、H.6。  
   **受影响主张：**TailSpline 是否是由 full-pair geometry 原理必然导出的方案。  
   **具体证据：**式 (7) 平滑 transition 内部及低频端的 log-gap variation，但故意不惩罚高频入口；这使入口 jump 更大、尾端 jump 更小。前述 canonical-overlap 理论没有推出这一非对称边界条件，也没有给出任务最优性。  
   **反证/已有回答：**论文准确称之为 declared objective / boundary prior，并明确“task value is tested below”；它只主张该目标的唯一解，不主张全局任务最优。  
   **最小修复：**在 §5.1 直接标注“design prior”，并用一两句说明为何冻结模型的 native association 使低频 junction 比高频 entry 更值得平滑。

3. **Cosh 的连续密度推导与实际有限强度规则之间仍有理论间隙（不支持的主张，仅影响强公式动机）**  
   **位置：**第 5–6 页 §5.2，附录 A.9–A.14。  
   **受影响主张：**`τ ∝ sqrt(d_head/L_train)` 作为有理论支撑的 operating rule。  
   **具体证据：**附录 A.13 的 scaling calculation 依赖 diffuse softmax、full-RoPE MHA、小 τ 等四个模型假设；在实际 τ 值处，局部 quartic 近似可高估 exact stiffness 约 1.52–3.52 倍。factorial 中相邻强度的排序也随配置变化。  
   **反证/已有回答：**作者已明确写出这些假设，说明实验使用 exact quantiles，且不宣称 finite-τ 精度保证；主论文主要贡献也不依赖 Cosh 强度最优。  
   **最小修复：**将 §5.2 的 “reference” 更醒目标成 heuristic reference rule，并把理论保证限制为密度目标的 minimizer，而非 τ 的任务最优选择。

### 推荐

**推荐：Weak Accept。内部评分：7/10；信心：4/5。** 数学对象、假设和证明大多严谨，尤其完整 pair 的处理优于常见单相位代理。理论与经验之间的桥主要是解释性而非预测性，但 PDF 已多处诚实限定；修辞再收紧即可。

---

## R3：实验与实际价值

### 中心贡献（我的理解）

实验部分证明两件不同的事：（1）固定支撑时，内部 frequency placement 本身会改变学习与冻结模型表现；（2）一个 O(K)、零训练、零校准的 TailSpline 表，在 Llama-3-8B-Instruct 的 2×/4× clean RULER 上明显优于 MrRoPE-Pro，并在 OLMo classic 面板上有很大迁移增益。论文还主动测量原生窗口成本、PPL、终止健康和自然 QA。

### 最强证据

第 7 页表 2 / 附录 H.7、H.10 是最强实用证据：共享 prompt IDs、decoder、gain、weights、端点和静态安装方式；32K 使用 200/任务，提升分布在 retrieval/tracking/aggregation/QA，leave-one-task-out 仍为正；16K 独立面板也显著为正。附录 H.8 给出 Native 对照：8K task 点估计下降 2.14 pp 且区间跨零，whole-prefix PPL 上升 0.37%；这使收益–成本边界可判断。附录 H.9 的自然 QA 零结果同样增加可信度，因为作者没有筛掉它。

### 重要关切

1. **clean 主结果的 baseline 覆盖较窄（可选扩展）**  
   **位置：**第 6–8 页 §6.1，附录 H.7。  
   **受影响主张：**TailSpline 的实际竞争力，而不是其相对 MrRoPE-Pro 的有效性。  
   **具体证据：**2,600-row clean 32K 和 650-row clean 16K 只比较 TailSpline 与 MrPro；附录 H.7 明说尚未在该面板比较 YaRN。论文其他 YaRN、Native、Uni、BM 对照来自不同模型、输入或历史协议，不能合并成同一 clean 排名。  
   **反证/已有回答：**MrRoPE-Pro 是共享外带、端点、gain 的最近构造 comparator，因而作为归因 baseline 合理；SOTA 或所有 baseline 不是证明本文核心命题的必要条件。  
   **最小修复：**最有价值的额外实验是在同一 clean 16K/32K prompt 上加入官方 YaRN 和原生/常用 scaling；若无法完成，应把主结论明确保持为 “over MrRoPE-Pro”。

2. **TailSpline 特定 transition shape 的经验归因尚未完成（不支持的主张，若主张被读成 shape 优越）**  
   **位置：**第 5 页 §5.1，第 7 页 §6.1，附录 H.2、H.8。  
   **受影响主张：**主 RULER 增益能否归于平滑低频 junction，而非不同的 total displacement。  
   **具体证据：**T–MrPro 同时改变 shape 与 displacement；精确 equal-dose C 的 Full-13 AUC 比 T 高 0.41 pp，CI 跨零，且 C 使用 batch 2/reordered batches，缺完整 runtime provenance。  
   **反证/已有回答：**论文已经明确称 finer ordering unresolved，并没有作等价性或 tail-only mediation 声明；G.7/G.8 支持“shape 一般会重要”，但不能替代 TailSpline 专属归因。  
   **最小修复：**用 batch 1、相同顺序和完整 runtime receipt 跑完 C；在此前，把“smooth transition 导致收益”改为“由该先验产生、经任务验证的表带来收益”。

3. **实际任务迁移显示清晰边界，应在结论中更居中（解释问题）**  
   **位置：**第 8 页 Natural QA 段、附录 H.9。  
   **受影响主张：**“improve model quality within a chosen context range”的任务广度。  
   **具体证据：**Natural-QA631 总体 T/P 为 41.08/40.88，CI [−1.53, 1.89]；>8K 层点估计反而 −0.94，CI [−4.15, 1.95]。这表明强 synthetic RULER 增益没有转化为该自然 QA pool 的明显 F1 增益。  
   **反证/已有回答：**论文完整报告并正确称两者接近，也没有声称自然 QA 显著提高；BM 在 OLMo 的自然 QA 有正结果，但属于另一构造和协议。  
   **最小修复：**在摘要或结论中加一句“在 Natural-QA pool 上与 MrPro 持平”，使读者无需进入附录即可看到 practical boundary。

### 推荐

**推荐：Weak Accept。内部评分：6/10；信心：5/5。** 主对照设计和样本规模足以支持相对 MrPro 的冻结部署价值，且报告非常透明。降分来自 baseline 广度、特定 shape 归因未闭合，以及自然 QA 未呈现清晰收益；这些限制缩小适用范围，但不推翻核心实证贡献。

---

## R4：整体论证与叙事

### 中心贡献（我的理解）

全文形成一条三段论证：先用固定端点识别 allocation 是独立变量；再用完整 rotary pair 的有限窗口几何说明 allocation 改变什么结构；最后提出两个明确构造，并以 TailSpline 的冻结部署结果为主、Cosh 的学习结果为辅。论文尤其重视区分不同 protocol 能回答的问题，这种证据分层本身是优点。

### 最强证据

第 15 页补充表 3–4 是论证组织最成功之处：它主动声明各协议不可汇总，并把 geometry theorem、learning utility、frozen deployment、natural QA 和 control 分开。主文第 2 页“what each control fixes”、第 6 页表 1、附录 H.7–H.9 的限定性措辞，使读者能区分“方法在某任务上有效”与“机制/归因已建立”。版式上，主文图 1–4 清晰、字体可读、公式无明显裁切或错位；62 页附录虽然很长，但表格和章节层次一致。

### 重要关切

1. **核心故事被大量历史/探索协议稀释（解释问题）**  
   **位置：**第 15–62 页补充材料，尤其 E–G、J。  
   **受影响主张：**读者能否迅速判断接受所需的最短证据链。  
   **具体证据：**补充材料包含 99-run staged study、多个 adaptation、BM、FullLagP2、C42V24、slot permutation 和探索性 response fit；部分结果使用不同 checkpoint、gain、数据选择、batch/runtime，且不能与主结果合并。  
   **反证/已有回答：**第 15 页已明确给出最短路径 B.1 → H.7 → H.9，表 4 也警告不跨行 pooling；因此这是可读性问题，不是证据混用错误。  
   **最小修复：**将附录首张表再加一列“decision-relevant / supporting / historical exploratory”，或把历史协议集中到一个 archival subsection。

2. **“几何 → 构造 → 性能”的箭头视觉上强于实际因果链（解释问题）**  
   **位置：**第 1 页摘要、第 4–5 页 §4–5、第 9 页 §8。  
   **受影响主张：**full-pair overlap 是否解释了 TailSpline 为何赢。  
   **具体证据：**几何结果刻画 overlap，TailSpline 则最小化 native-relative log-gap variation；PDF 没有证明前者推出后者，也没有机制干预。  
   **反证/已有回答：**H.6 明确说未完成机制实验，§5.1 也说 task value 单独测试。  
   **最小修复：**在主文加入一句桥接边界：“geometry establishes why placement can matter; TailSpline is a separate native-compatible design prior whose value is empirical.”

3. **主文可以更直接呈现零结果的意义（解释问题）**  
   **位置：**第 8 页 §6.1、第 9 页结论。  
   **受影响主张：**实际价值边界和论文可信度。  
   **具体证据：**Natural QA 的 +0.20 pp 及跨零区间已经在主文出现，但结论只说它“further describes task response”，没有直说“未观察到清晰优势”。  
   **反证/已有回答：**数字和限定都完整，没有隐藏负结果。  
   **最小修复：**结论用一句直白表述：TailSpline 对 RULER 有明显优势，而在该自然 QA pool 上与 MrPro 基本持平。

### 推荐

**推荐：Accept。内部评分：7/10；信心：4/5。** 论证主体完整、证据边界罕见地自觉，主文九页能承载复杂贡献。叙事仍可通过弱化“理论导出方法”的暗示、突出零结果和压缩历史附录来显著提升。

---

## AC Meta-review

### 分歧核对与事实裁决

四位视角的分歧主要不是事实，而是对“实用充分性”和“理论–方法桥”的权重不同。重新核对 PDF 后，我作如下裁决：

- **应保留的核心事实：**固定支撑的三种子配对训练确实只改变内部 allocation；排序在 retarget support 后反转，说明 allocation 有独立影响且与 support 交互。冻结模型的 fixed-support 实验也支持这一点。
- **应保留的方法事实：**TailSpline 确实是闭式、O(K)、不读权重/激活/校准结果的静态表；相对 MrRoPE-Pro 的 clean Llama 16K/32K 增益均由匹配输入和非跨零配对区间支持，OLMo classic transfer 也很强。
- **必须丢弃的误读：**论文没有声称 TailSpline 在自然 QA 上显著更好；没有声称 full-pair rank 单独预测任务效果；没有声称 equal-dose T/C 已证明 TailSpline 特定 shape 优越；也没有声称全任务、全模型或 SOTA 普适最优。附录已明确否定这些更强读法。
- **决定相关问题：**主结果只能直接支持 “TailSpline over MrPro under tested protocols”；TailSpline 特定 spline shape 相对 total displacement 的归因尚未完成；Natural-QA 迁移基本持平。这些问题限定结论，但不推翻 allocation 的独立价值或方法的已测收益。
- **可选改进：**在同一 clean panel 加 YaRN/其他常用 baseline、增加第二模型 clean source-order panel、扩大自然任务覆盖。这些会提高实用比较的广度，但不是接受 allocation 核心贡献的必要前提。

### 新知识与实际价值

本论文建立的新知识有三层。第一，RoPE 的 endpoint/range 不能代表整个 frequency table；内部 allocation 是可独立干预、可被学习适配、且在冻结模型中受坐标关联约束的设计变量。第二，完整 sine–cosine pair 的 canonical-overlap/effective-rank 框架提供了比 cosine-only proxy 更正确的有限窗口结构描述，并严格刻画慢频 pair 的方向塌缩。第三，TailSpline 展示了无需训练或校准、保持标准 rotary operator 的实际部署路径，在测试的 Llama clean RULER 和 OLMo classic RULER/PPL 上具有显著价值，同时代价和边界被量化。

这不是“已证明的任务最优 RoPE”，也不是自然长上下文 QA 的普遍改进。它仍然值得接收，因为核心识别问题重要、方法简单、正结果规模足够、失败和协议边界报告透明。

### 总体推荐

**总体推荐：Weak Accept。AC 内部评分：7/10；信心：4/5。** 这不是四个分数的算术平均。决定性理由是固定支撑识别与 clean frozen deployment 两条证据链相互独立且都成立；理论提供了真实的新结构视角，虽未形成性能预测定理。未闭合的 shape attribution 和 Natural-QA 零收益降低了主张宽度，而非核心正确性。

### 最高价值的稿件修订

1. 在摘要、§5.1 和结论中明确分开三件事：allocation 独立有效；TailSpline 相对 MrPro 有任务收益；TailSpline 特定 shape 超越 equal-dose alternatives 尚未确定。
2. 在 §4→§5 的过渡处明确 full-pair geometry 是结构动机，不是 TailSpline 性能的因果或最优性证明。
3. 在主文结论直接写出 Natural-QA 与 MrPro 基本持平，并把 practical claim 限定到已测 protocol。
4. 若能补实验，优先在同一 clean 16K/32K panel 加官方 YaRN，并以 batch/runtime 完全匹配的 C 完成 equal-dose 对照；这两项比新增更多探索性 profile 更有信息量。
5. 用一张简短 related-work 属性表强化新颖性边界，并在补充材料目录标出核心、支持和历史探索证据。

### 评分汇总

| 视角 | 推荐 | 内部评分（1–10） | 信心（1–5） | 决定性理由 |
|---|---:|---:|---:|---|
| R1 新颖性与意义 | Weak Accept | 7 | 4 | 独立设计变量的识别清楚，闭式冻结方法有价值；优先权仅能据 PDF 判断 |
| R2 理论与方法 | Weak Accept | 7 | 4 | 数学对象与证明严谨；理论主要解释结构，未预测任务收益 |
| R3 实验与实用 | Weak Accept | 6 | 5 | clean RULER 强，但 baseline 较窄、shape 归因未闭合、Natural QA 持平 |
| R4 论证与叙事 | Accept | 7 | 4 | 证据分层透明、主线成立；附录过密且理论到方法的连接需降调 |
| AC | **Weak Accept** | **7** | **4** | 两条核心证据链成立，限制缩小适用范围但不否定贡献 |

### 新颖性评估的限制

由于本评审被限定为只使用冻结 PDF，我只能评价作者在文中如何区别和引用既有工作，无法检索外部文献、核对同期论文内容或确认“首次”优先权。我的正面新颖性判断因此是条件性的：就 PDF 呈现的文献版图而言，固定端点识别 allocation、完整 pair 的有限窗口几何以及 calibration-free TailSpline 的组合具有清晰增量。
