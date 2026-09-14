# 从仓库重新理解这项研究

2026-09-14。作者要求重新广泛阅读仓库后形成的研究理解稿。本轮没有修改论文、运行模型、改写历史结果或变更实验队列。以下是对已读证据的综合判断，不替代各结果 owner，也不把历史文档的改稿命令当作本轮指令。

## 1. 这项研究实际要告诉读者什么

**通过重新配置 RoPE 内部频率，我们已经在学习期和零训练冻结部署中改善了长上下文表现。固定支持实验进一步识别了其中不由频率范围变化解释的收益；Cosh、BM及最新TailSpline提供具体构造和实际评测证据。**

论文已有的实质并非“改参数会改结果”。它已经连接了：可实现的解析分配、多个协议中的外推收益、固定范围的归因、有限窗口的位置基分析，以及成熟模型的适配和冻结部署。论文应让读者看到这条链。完整的任务最优性理论尚未建立，不取消已观察到的设计价值。

这里也不是宣称所有历史 EVQ 实验都是纯 z 干预。历史有限表有不同网格和端点处理；严格的纯 z 归因由匹配控制承担，实际用途由各自训练/部署实验承担。

零训练是已经取得实证成功的主线，不是未来应用设想。冻结同支持实验直接隔离内部配置；BM在冻结OLMo上改善五项自然QA；最新精确TailSpline在冻结Llama上改善Full-13与NIAH，PPL AUC也有小幅正向收益。它们不需要借适配实验来证明“能进入实际回答”。适配结果是另一条支持分配价值的证据，不能替代、更不能遮蔽零训练成绩。各方法并非所有模型/任务普遍占优，但这不改变已完成的零训练成功。

## 2. 重新读到的实验证据

下表按科学问题组织，不按赢分大小排列。数字分别来自本轮读过的 JSON、结果报告和实现；本轮重新聚合的条目另行标明。

| 研究问题 | 已做的实验与结果 | 应怎样理解 |
|---|---|---|
| 分配能否改善实际外推？ | 早期 TinyStories 125M 标称模型，τ1.5 在 seeds42/137 的 4/8/16K 均优于 Geo；16K PPL 分别 34.153→27.699、28.502→26.860。[早期 sweep](../../docs/exp/2026-02/2026-02-27_evq_tau_sweep_results.md) | 早期经验发现，report-backed。报告中的“最优定律”“随规模放大”不是这些点已经证明的结论。 |
| 收益是否超出 base/端点变化？ | 151.9M、L256、每臂约500M tokens、三配对 seeds，固定两个实际端点，只改变30个内部频率；2/4/8× tail NLL 差 −0.281/−0.176/−0.146，三个长度均3/3同向，窗内 +0.026。[固定支持 owner](evidence/EXACT_RANGE_151M_3SEED_RESULT_20260820.md) | 是外推改善的归因证据，不应只写成“行为不同”。 |
| 是否只有 Cosh 一条曲线有效？ | 12配置×3 seeds 的固定支持 M4：reference Cosh 7/12、1.25× Cosh 10/12、匹配 Exp 9/12 优于 Geo。本轮从每配置均值重算一致。[记录](../../rebuttal/rebuttal_0723/theory_results/m4_exact_range_factorial_evidence_20260726.json) | 支持设计空间超出单一曲线；128步/臂的短训练承担广度，不替代成熟训练。192条 run 记录不是192个独立结构配置。 |
| 旋转通道少时是否有实际价值？ | 432M MLA、K16、8K训练、500M tokens、三 seeds：16K PPL 138.807→95.588；20/24/28/32K 也均三 seeds 同向。窗内35.445→35.775。[原始值可移植副本](../../data/curated/table18_mla_3seed_aggregate.json) | 不能只拿16K一个终点讲。整个扩展曲线支持学习到的外推差异；约0.9%的窗内PPL代价应同图可见。 |
| 优势是否只在最后训练时刻出现？ | 同一 MLA 16K，50%预算 Geo/Cosh=166.240/117.983；75%=146.223/101.605；100%=138.807/95.588，每阶段3/3同向。本轮从 progression 重算。 | 已有训练轨迹证据值得展示；这些是相关检查点，不能当作九个独立 seeds。 |
| 改分配是否只能从头训练？ | 750M从同一2K Geo checkpoint继续500M tokens到4K；16K PPL45.136→24.407，8K answer-token AR exact 0/40→31/40，4K PPL21.955→22.282。[Geo raw](../../results/core_text/phase15/phase15_geo_seed42_result.json)、[EVQ raw](../../results/core_text/phase15/phase15_evq_r0_seed42_result.json)、[可移植摘要](../../data/curated/phase15_750m_continue_result_20260306.json) | 一对训练展示继续学习可利用新的分配；本轮核对 raw 数值。AR指标不额外含完整EOS要求。 |
| 分配与后续缩放如何组合？ | 454M、2K、三 seeds四臂：8K PPL Geo/EVQ=161.9/150.3；同一历史缩放规则后82.9/70.9，检索分数61%/100%。[四臂记录](../../data/curated/table2_evq_yarn_454m_passkey_10pct.json) | 显示缩放与训练分配的组合效果。历史“YaRN”是仓库fixed ramp，PK是teacher-forced NLL-gap，不能改名为官方YaRN或自由生成。 |
| 组合现象是否也出现在 MLA？ | 同一 wavelength blend、s4，16K Geo/Cosh=117.88/71.13，32K=278.50/236.59；另有s2完整曲线。[MLA记录](../../data/curated/table18_mla_3seed_aggregate.json) | 可以展示“同一规则作用于不同分配”的2×2；不能与454M的另一算子合并成同一个官方YaRN结果。 |
| 更长期、分阶段学习有何现象？ | 454M三阶段512→1024→2048，历史报告16K raw PPL13.17→2.48，48K加历史overlay为14.22→2.63。[Phase17C报告](../../docs/exp/2026-03/2026-03-11_phase17c_2048_continue_results.md) | report-backed单seed支持，轨迹和τ均分阶段改变；不能取代单变量因果实验，也不能仅凭PPL称48K任务能力。 |
| 是否只有语言自回归模型出现？ | 129.6M双向Video DiT、32→128帧，单seed matched protocol远端MSE0.009891→0.006388，约−35.4%。[记录](../../data/curated/video_dit_seed42_head_to_head_20260826.json) | 跨模态广度，指标是denoising MSE，不是文本能力或视频生成SOTA。 |
| 收益能否进入完整回答？ | OLMo匹配routing与answer+EOS续训后，4/8/16K完整答案+EOS Native=95/18/0，EVQ=100/98/60，各100输入。[结果及lineage](../../rebuttal/rebuttal_0723/theory_results/evq_query_gap_realized_eos32_20260728/FINAL_METRICS_AND_LINEAGE.json) | 真正的输出能力证据。本轮核对数值；训练物理≤4K但显式暴露长相位，因此是长度迁移/适配结果。 |
| 是否还有自然任务支持？ | 独立selective-QK适配，200条/长度的2Wiki长度迁移：8K F1 Native/EVQ=0.07/21.48%，16K=0/8.57%。[协议与记录指针](../appendix/a6_mature_scale.tex) | 与上行不是同一个adapter。收益支持可用性，4K及RULER代价按各自协议保留。 |
| 冻结权重后内部配置还有多少作用？ | 相同支持、gain和模型下，OLMo长端uniform/derived/ramp=0.56/60.47/61.04%，Qwen=57.75/66.50/64.00%。[owner](attention-aware-retrofit/results/causal-mechanism/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md) | 强内部配置效应；不是derived独家胜利，也不是Native只动内部端点不变：共同慢端已是Native/4。 |
| 总位移一样，中段细形状还有作用吗？ | C42两表端点、gain、总位移与increment质心相同，350开发输入43.67→54.40%，另16文档NLL2.9417→2.8309。[公式、raw路径和复算入口](../appendix/a9_recovered_design_evidence.tex) | 比单纯“范围相同”更细的控制，支持内部形状不能全部压成一个总量。开发面板身份不能改成独立确认。 |
| 能否构造不读取模型权重的部署表？ | BM五自然QA可用池778条/臂，其中631长输入，五任务宏F1 21.62→25.44%，差+3.82pp，已有区间[1.32,6.29]。[逐行owner](../../docs/research/ROPE_OLMO_BM_FIVE_QA_RESULT_20260908.json) | 本轮从长输入逐行score重算五任务均值一致。公式可执行且有自然输出价值；跨Qwen差异限定推广范围。 |
| 最新研究到底推进到哪里？ | 精确TailSpline在Llama S4已完成Full13、NIAH、PPL两臂比较，AUC差+3.20pp/+3.75pp/−0.00449。[最新结果](../../docs/research/next_stage_20260912/TAILSPLINE_LLAMA_CLASSIC_RESULT_20260914.md) | 当前owner报告已完成，不能再说只有CPU。属于本轮稿件原计划之外的新结果；未在本轮重算raw，也不自动加入稿件。 |

## 3. 理论、构造和实现究竟连接在哪里

设 `x_k=-log ω_k=a+R z_k`。这不是凭换符号创造贡献：它把“整个表变快/变慢”“跨度变化”“内部重新配置”拆开，给实验提供可锁定的变量。对几何表，换base仍保持均匀z；改变内部z才能离开这个几何子族。

**几何部分已经有实质结果。** 单个pair的注意力贡献为 `C cos(ωΔ)+D sin(ωΔ)`；因此比较完整sin/cos子空间，比仅看cosine collision更贴近旋转结构。block-whitened Gram的重叠—有效秩恒等式、慢频共同二维极限、有限参数数值给出“对数频率分开，不保证有限窗位置方向分开”的精确定义。实际base256识别设置的8个最慢pairs，16坐标的有效秩约2.11。

**这解释的是重新分配的结构动机。** 它还没有证明提高该秩就是模型外推改善的唯一中介。block whitening除去了能量尺度；慢频的内容相关系数、原始特征幅度和softmax使用不能从秩里自动读出。这个区别用一句定义与一段机制说明即可交代，不必让整篇稿件围绕“不能预测任务”展开。

**Cosh已经是构造，不是任意候选名称。** 声明密度集中惩罚和累计慢尾能量，变分得到 `ρ''−τ²ρ=0`、对应边界条件与唯一正cosh密度，逆CDF给有限表。固定端点下正τ使内部频率向更快方向移动；所选择的τ不是一个已证明的任务最优常数。实现见 [schedules.py](../../scripts/lib/rope/schedules.py)，证明见 [a1_proofs.tex](../appendix/a1_proofs.tex)。

**BM研究的是另一种构造问题。** 冻结模型用 `ω'_k=ω_k S^(−m_k)`；高频保持，低频到 `/S`，中段改变累计减速。有限边界匹配给出cubic累计式，代码仅需要native公式与公开base/L/S：[boundary_matched.py](../../scripts/lib/rope/boundary_matched.py)。学习期把内部频率调快，部署期把部分频率调慢，二者并不矛盾：前者让模型学习新基，后者延展既有模型的使用范围。不能未经推导说二者都验证“越往某一端搬越好”。

**full-z是更一般的空间，不等于某个已完成的万能方法。** 已有完整正gap参数化：[fixed_support_z.py](../../scripts/lib/rope/fixed_support_z.py)，成熟direct-z pilot及z×QK联合oracle确实运行过，不能说“从未学过z”。二者有特定full/tail权衡，不能顶替新scratch比较。最近2K scratch Geo/Cosh/full-z有[实现和运行审计](../../experiments/fixed_support_joint_151m_20260912/index.md)，本轮本地广搜尚未定位其完整三臂共同端点结果；这一缺口保持明确，不据此宣布它没有做过或失败。

**三段式的后续理论已经有建设性进展。** BM对称边界与TailSpline单侧边界都可给唯一离散解；等剂量YaRN–MrPro对照识别了原比较的总位移混杂。它们缩小了可检验问题，不等于已经从注意力推出唯一最佳band。新构表继续不读权重/激活，历史权重干预用于理解而非选表。[方法合同](../../docs/research/next_stage_20260912/TAILSPLINE_ROPE_METHOD_AND_UNIFIED_EVAL_20260914.md)、[等剂量分析](../../docs/research/next_stage_20260912/MRROPE_YARN_EQUAL_DOSE_PRINCIPLE_AUDIT_20260914.md)。

## 4. 我此前写作中的具体误读

1. **把有方向、有收益的发现缩成“改变行为”。** 固定支持结果是一致的外推改善，MLA与继续训练给实际尺度支撑；引言应该首先告诉读者收益是什么。
2. **把构造实验全部降成“用途”。** Cosh的可执行公式和多协议结果是贡献的一部分，不是科学结论讲完后的附带demo。
3. **把兼容性当成了主角。** 换表交叉和槽位置换解释为什么训练期设计不能无条件硬换到成熟模型。它们增强方法理解，不应压过外推成功本身。
4. **把历史审计中的停止门当成研究价值排序。** 如direct-z pilot held-out均值改善但一条超gate；这是该校准协议没有达到部署要求，不是z没有作用。相反，也不能把所有通过门的结果当同一类外推能力证据。
5. **没有充分利用已完成的训练轨迹与输出实验。** MLA progression、750M AR exact、OLMo完整答案及自然QA都应参与叙事取舍。仅说“还有大量附录结果”不够。
6. **对新2K Cosh与历史EVQ身份反应过度。** 已有源码审计表明锚定版本相对历史midpoint版本约统一加速13.8%，这是明确的构造差异；局部尚未找到完整运行证据证明arm交换或“训练反了”。不能猜测故障原因，更不能抹去旧结果。

## 5. 下一次改稿应遵循的证据逻辑

**先展示外推改善，再做变量归因，再解释如何构造和使用。** 可用中心表述：

> Redistributing RoPE frequencies improves long-context performance both during learning and through training-free deployment. We develop explicit allocations and demonstrate gains in language modeling, retrieval, and natural QA; matched-support interventions isolate the contribution of interior allocation beyond frequency-range changes.

这句话是证据综合，不是新的已验证普适定理。相比当前三个并列“区别”，正文更适合围绕下面的递进关系：

- **观察：** 内部分配能改进外推；以fixed-support和MLA曲线先建立实质。
- **解释与构造：** 哪些位置方向被重复表示，Cosh如何给出一个可计算的分配。
- **验证广度：** 继续学习、缩放组合与实际读出分别验证不同用途。
- **零训练方法与任务收益：** BM的闭式、固定安装及五项自然QA是主文实证支柱；最新TailSpline应按作者的最终稿范围决定是否纳入，不继续误称未完成。兼容性干预服务于理解这些方法，不能取代它们。

“范围重定向后相对排序反转”应作为一个明确的range×allocation交互控制，不再独立承担摘要贡献。它不是训练方向错误；也不是历史所有缩放组合都无收益，因为算子、训练范围和频率表不同。

图表方面，优先利用已有数据增加信息量：MLA训练预算×长度曲线，或Geo/Cosh×无缩放/同缩放的四曲线，比再增加一个泛泛的概念图更能证明价值。独立协议分panel，不混合NLL、PPL、AR exact与F1。现有三图两表的数量是排版方案，不能变成遮蔽关键证据的硬约束。

## 6. 本轮覆盖与仍未确认的内容

已按问题跨读：早期训练报告与原始JSON、curated结果、固定支持记录、完整位置基证明、Cosh/BM/full-z实现、M4多形状、成熟适配、冻结同支持、C2/C42、频段删除与置换、直接z及联合oracle、最新band/TailSpline研究、当前主文及相关附录。旁线稀疏selector研究与本稿RoPE allocation贡献分开，未混作论文证据。

本轮实际复算了MLA三个训练阶段和六长度的seed均值/方向，M4三个主形状的配置胜数，BM631长输入五任务均值；核对750M raw PPL/AR和OLMo完整答案lineage汇总。其余明确按报告或数值owner读取，没有声称重新验证全部模型输出。

尚未完成：新2K full-z训练完整结果追回；历史Geo/midpoint“差异很小”的直接对照owner定位；所有旧overlay逐版本执行快照复核；最新TailSpline raw独立复算。它们不妨碍理解已确认的贡献，但涉及这些具体主张时必须继续追源。本轮未开展外部文献优先权审查，因此不作“首个”或全球SOTA判断。
