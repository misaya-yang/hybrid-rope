# ICLR 2027 风格独立审稿报告

## 1. 中心贡献

本文把 RoPE 频率表分解为频率支持区间（端点）与区间内部的频率分配，并主张后者是一个独立且有实际价值的设计变量。论文用固定端点的配对训练和冻结模型干预来识别内部频率分配的作用；用完整 sine-cosine 二维子空间的典型相关与 Rényi-2 有效秩描述有限窗口内的频率冗余；再提出无需训练或校准的闭式方法 TailSpline，在保持标准旋转算子、频率端点、外侧频带和增益不变的条件下，重新安排过渡带中的额外 log-frequency span。作为另一条证据线，Cosh transport 用于配对训练，说明“改变内部 allocation”并不限于 TailSpline 这一种构造。

就 PDF 所呈现的相关工作而言，论文的新意不在于再次证明频率缩放有用，而在于较干净地分离 support 与 allocation、给出完整旋转对的几何刻画，并把这一视角落实为一个可直接替换冻结模型 RoPE 表的闭式构造。

## 2. 最强贡献与证据

1. **固定支持区间的识别实验设计清楚。** 第 3.1 节及附录 B.1 在三个配对 seed 上固定架构、初始化、token 顺序、优化器、训练预算和实际频率端点，只改变 30 个内部归一化坐标。保留训练 support 时，Cosh 相对 Geo 在 2×/4×/8× 长度的 mean tail NLL 分别改善 0.281/0.176/0.146，三个 seed 在每个扩展长度方向一致。把同一批权重和 allocation 重新映射到目标长度 support 后排序反转（附录 I.1），有力地说明 support 与 allocation 是两个不同的设计坐标，而不是简单把“更慢频率”重新命名。

2. **冻结部署的主要结果规模大、配对严格、报告透明。** 第 6.1 节与附录 H.7/H.10 的 Llama-3-8B-Instruct clean RULER 结果使用同一 checkpoint、静态表、增益、输入和解码器。16K（650 对输入）TailSpline 对 MrRoPE-Pro 为 +3.39 pp，配对区间 [1.53, 5.34]；32K（2,600 对输入）为 +11.72 pp，[10.32, 13.11]，且 13 个任务中 12 个任务均改善，leave-one-task-out 仍为正。附录 H.7 保留 cap-hit、empty 和 EOS 统计，没有按生成健康度删样本。OLMo-2-1B 的 classic Full-13 AUC 也从 17.36% 提升到 66.60%，所有 task-AUC 差异为正。跨两个不同模型家族的方向一致性增强了方法结果的可信度。

3. **论文对证据边界相当克制。** 第 5.1 节明确说 TailSpline 与 MrPro 同时改变总位移和过渡形状；附录 H.8 明确承认等剂量 T/C 结果跨 batch runtime，区间跨零，不能用于 equivalence 或 residual-shape attribution。Natural-QA631 的总体差异仅 +0.20 pp、区间 [-1.53, 1.89]，正文据实表述为两者接近，没有把 synthetic RULER 增益外推成通用 QA 改善。原生窗口任务差异的区间也跨零，论文只称其为 observed trade-off。

4. **数学对象定义得完整，核心结论与其假设匹配。** 第 4 节把单个频率视为完整的 `span{cos(ωΔ), sin(ωΔ)}`，通过 block whitening 得到基不变的典型相关，并给出平均 pairwise overlap 与 Rényi-2 effective rank 的精确恒等式。附录 A.1 给出 cross-Gram、恒等式证明和慢频极限展开。第 4.3 节的频谱等价判据明确限于对所有内容向量成立的精确双线性 kernel，并在附录 A.6 处理 aliasing；正文没有把它夸大为训练后功能等价定理。TailSpline 的有限网格凸二次目标及闭式唯一解在附录 H.1 中也可直接核验。

5. **复现实验契约的文字信息丰富。** 附录区分 scratch training、continuation、adaptation、frozen deployment 等不同证据角色，给出表构造、频带、增益、样本数、选择规则、bootstrap 单元和多个完整任务表。尤其 clean RULER 明确采用 source-order、无内容 padding、无 depth balancing，并说明 allocation 在这些输入产生之前已经固定。

## 3. 决策相关弱点

### W1. TailSpline 的“平滑尾部连接”尚未被实验单独确认为胜因

- **位置：** 第 5.1 节，PDF 第 5–6 页；附录 H.2/H.8，PDF 第 50、55 页。
- **具体证据：** 主比较中的 TailSpline 与 MrRoPE-Pro 虽共享外侧频带、端点和增益，但同时改变总 log-frequency displacement 与过渡形状。论文构造了精确等剂量控制 C；然而 T–C 的 Full-13 AUC 为 -0.41 pp，区间 [-2.63, 1.82]，且 T/C 使用不同 batch runtime。作者自己判定该结果不能支持 equivalence 或 residual-shape attribution。
- **附录中的反向证据：** 附录 G.7/G.8 的 BM–Uni 在固定总位移下出现较大任务差异，说明“高阶形状可以重要”；但这不是 TailSpline 平滑目标相对 MrPro 的直接识别，也不能确定 TailSpline 的具体边界先验是关键原因。
- **对实际主张的后果：** 这不推翻“完整 TailSpline 方法优于 MrPro”的实证结论，也不影响闭式优化定理；它限制的是机制归因。当前证据不能说优势来自 Eq. (7) 所强调的 terminal smoothing，而不是其不同的位移剂量或两者组合。
- **最小修复：** 在同一 runtime/batch 下重跑预先指定的 T–C 等剂量配对比较，并在摘要或结论中继续把主要结果称为完整构造效果；只有该对照明确后再把收益归因于 tail smoothing。

### W2. 几何分析解释“allocation 会改变什么”，但尚未形成可验证的任务性能预测

- **位置：** 第 4 节，PDF 第 4–5 页；附录 A.1–A.4，PDF 第 15–19 页；附录 J.5–J.6，PDF 第 61–62 页。
- **具体证据：** 完整旋转对子空间重叠和有效秩是严格、清楚的描述量，但正文从“慢频方向重叠”过渡到 TailSpline/Cosh 设计时，缺少一个在未观察任务结果前即可判断哪种 allocation 更优的映射。附录 J.6 的 12-profile band-count/logit fit 明确是用同一组 profiles 选择阈值和拟合系数的探索性描述，不能解析组内排序；随后同一类 profile 的自然文本 NLL 还显示，增加该 count 并不保证更低 loss。
- **附录中的反向证据：** 附录 A 的数学推导确实证明重分配会改变有限窗表示冗余；第 3 节的固定支持实验也证明 allocation 对模型结果有因果作用。因此问题不是几何量错误，而是它对具体构造选择的预测力尚未建立。
- **对实际主张的后果：** “allocation 是独立设计变量”和 TailSpline 的经验优势仍成立；较弱的是论文叙事中从几何机制到具体设计的解释性闭环。当前更像合理设计动机加经验验证，而不是由几何理论导出的性能原则。
- **最小修复：** 明确把 overlap/effective-rank 定位为诊断而非性能代理，并增加一个很小的、预先固定候选的 out-of-sample 排序测试：只用几何准则选择候选，然后在未参与选择的任务/模型上检验排序。

### W3. 最强 clean RULER 结论只与 MrRoPE-Pro 作直接比较，限制了相对实践价值的定位

- **位置：** 第 6.1 节，PDF 第 7–8 页；附录 H.7/H.10，PDF 第 54–57 页；附录 G.1/G.4/G.8，PDF 第 44–49 页。
- **具体证据：** 16K/32K clean、source-order、Full-13 大样本结果只有 TailSpline 与 MrRoPE-Pro 两臂。论文因此能可靠支持“TailSpline 优于 MrPro 于这些面板”，但无法从该最强协议判断它相对 YaRN、uniform transition 或其他零训练常用表的优势。
- **附录中的反向证据：** 附录 G 在不同模型、任务子集、样本数或 classic 协议下提供 Index、BM、MrRoPE-Uni 和 YaRN 等比较；这些结果支持 allocation 的广义价值，但不能与 clean Llama 主结果无缝合并。附录 H.7 也主动声明该 clean 结果尚未比较 YaRN。
- **对实际主张的后果：** 论文没有要求 SOTA，且其明示的 comparator claim 是成立的；该缺口主要降低贡献显著性的可判定性，因为 MrPro 可能是一个在当前 s=4 Llama 设置中较弱或不最合适的单一基线。
- **最小修复：** 在已经冻结的 16K/32K clean 输入上加入一个预先指定、公开公式且无训练的强基线（优先选择论文附录已实现的 YaRN 或 MrRoPE-Uni），共享 gain、decoder 和 endpoints 时需清楚说明哪些量能够匹配、哪些不能。

### W4. 部署区间与原生窗口权衡的外部有效性仍主要依赖单 checkpoint、条件化输入区间

- **位置：** 第 6.1 节，PDF 第 7–8 页；附录 H.3–H.5/H.8–H.10，PDF 第 51–57 页。
- **具体证据：** clean 主结果的 bootstrap 区间条件于已选 source rows，不能反映 checkpoint、模型训练或任务分布变化。Llama 的原生任务参考只有 130 对 classic padded 输入，T–Native 为 -2.14 pp 且区间 [-6.14, 1.92]；自然 QA 的主区间跨两种排序。OLMo 提供跨模型支持，但使用 classic depth-selected/prefix-padded 协议，且两臂大量输出未以 EOS 结束（257/248 of 390），与 clean Llama 协议的外部有效性不同。
- **附录中的反向证据：** 论文完整报告这些限制、termination counts、PPL、Natural-QA 和不同长度曲线，并没有把行级 bootstrap 当成训练 seed 不确定性。clean 32K 的效应很大，且 leave-one-task-out 为正，因此不是脆弱的单任务偶然结果。
- **对实际主张的后果：** “在两个具体冻结模型和已测协议中有效”证据充分；“适合作为一般冻结部署规则”的可信度仍低于主结果的视觉冲击，尤其原生窗口成本和自然任务迁移尚不精确。
- **最小修复：** 无需大规模 SOTA 网格；在第二个模型上复用 clean source-order 协议，或在 Llama 上增加一个不同训练来源的 checkpoint，并报告同一静态表在 native/2L/4L 的统一协议曲线即可。

## 4. 可选建议

- 正文可把协议繁多的证据压缩成一张“主结论—主要实验—仅作诊断的实验”图，进一步减少读者把 historical/development panels 与 confirmation panels 混合解读的风险。
- Figure 3(c) 的 OLMo 巨大增益与大量 non-EOS 输出同时出现；可在正文图注中直接标注 termination 比例，帮助读者快速判断该协议的生成行为。
- 对 Eq. (7) 的边界先验增加一句直观解释：为什么部署目标更看重进入完全插值 tail 的连续性，而允许 high-frequency 入口有更大跳变。当前数学目标明确，但选择该不对称先验的任务层理由略短。
- 若版面允许，可在主表中同时给 absolute task-family scores，而不只给 aggregate/differences；附录已有完整数据。

## 5. 总体评价

**建议：接收（介于 weak accept 与 strong accept 之间）**  
**内部评分：7/10**  
**置信度：4/5**

决定性理由是：论文提出了一个清楚且有辨识度的问题分解；固定端点的配对实验确实证明内部 allocation 有独立作用；TailSpline 是简单、闭式、零训练的可部署方法，并在大样本 clean RULER 上对最近比较对象取得强且一致的提升。数学部分对自身对象和假设的表述基本严谨，附录对协议差异、负结果、区间和未解决归因异常坦诚。这些优点足以支持接收。

我没有给到 8 分，主要因为 TailSpline 相对 MrPro 的胜因仍混合了剂量与形状，理论几何尚未成为可检验的候选选择原则，而最强 clean 结果缺少第二个强零训练基线和同协议的跨 checkpoint 验证。这些是贡献定位和机制闭环的限制，不是已建立的数学错误，也不推翻论文实际、较克制的比较主张。

## 实际检查材料

我只使用了提供的 62 页 PDF。逐页提取并通读了正文第 1–10 页（摘要、§1–§8、伦理、复现和 AI use statement），检查了参考文献页；通读/核对了附录导读与协议表、A.1–A.6 的几何及频谱证明、B.1–B.3 的固定支持与 crossed-table 实验、D.2 的 grid/operator 定义、F.2 的冻结 fixed-support 对照、G.1–G.9 的 profile/placement/等位移与 QA 对照、H.1–H.10 的 TailSpline 证明、完整长度曲线、clean RULER、原生窗口和 Natural-QA、I.1–I.4 的 support/table/slot 干预，以及 J 的区间目标与探索性 profile 分析。我还渲染并目视检查了正文关键图页和附录中的主要表/图页；未发现遮挡、裁切、不可读公式、坏字形或明显版式故障。
