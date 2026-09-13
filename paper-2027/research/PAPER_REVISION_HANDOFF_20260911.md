# Beyond the Base：论文资产、科学关系与修订交接

建立于 2026-09-11；当前整理版本 R4：2026-09-13。文件名保持稳定，内容按当前论文维护。

本档回答：**已经获得了什么知识，哪些证据最重要，它们怎样组成一篇论文，下一阶段最值得解决什么问题。** 目录入口为 [index.md](index.md)，逐项来源与可达性见 [证据索引](evidence/index.md)。原 R1 的四阶段详账、术语纠正、反例、历史队列和来源线索完整保存在 [R1 历史快照](history/PAPER_REVISION_HANDOFF_R1_20260912.md)。

本次用户授权整理全仓库，并要求三个子代理分别审查理论、实验和论文以形成下一阶段计划。旧档“只改本文件”“仅限 paper-2027”“某代理只审不写”等特定任务条款不是现行仓库规则；历史 GPU 队列、预算与“今晚”只描述记录时点。下一阶段安排集中于 [计划入口](../../docs/research/next_stage_20260912/index.md)。

> **夜间结果更新：** 当前全局anchored Cosh适配连续负结果已触发[主线与优先级复核](../../docs/research/next_stage_20260912/index.md)。下列A01–A28是既有资产清单，不意味着每条原构造路线仍是后续优先项。新E3/C42形状对照、BM自然QA和Cosh适配负结果需要共同进入下一版证据判断；新增数字目前按执行报告依据登记，尚未在本任务逐行核验。S1实时端点仍待取得，不沿用旧运行状态作结论。

## 2026-09-13 当前目标覆盖

以[固定表全窗口研究方向](../../docs/research/next_stage_20260912/PAPER_INTERVAL_DIRECTION_20260913.md)为准。目标是实际质量：全面胜过强基线，或直接优化一张固定表覆盖短端至SL的质量并与强基线比较。解释YaRN/MrRoPE差异不是论文主线；transition、band、低频是否/s是研究手段。A01–A28继续是有效历史资产，EVQ/LoRA不成功不能在训练审查完成前被升级为整个构造无效。

正文顺序已改为：定义→固定支持/几何→学习兼容→成熟模型固定表部署→学习期构造。新增区间目标、条件深度推导与明确标为开发证据的长度对照；未声称存在已确认的统一全窗口赢家。本次修订与验收见[修订记录](PAPER_INTERVAL_REORIENTATION_20260913.md)。

## 1. 当前论文与中心认识

标题：**Beyond the Base: Exponent Allocation in RoPE**。

> 内部指数分配是 RoPE 中有实质作用的设计自由度：它组织有限的位置基，参与模型表征的学习，并能通过具体构造改善长度泛化与上下文利用。

三个连续问题决定论文结构：

1. **分配改变了什么？** 同一支持区间中的有限频率点组织不同的位置方向。
2. **怎样进入模型行为？** 权重学习使用这些位置基，作用涉及运行范围及频率与槽位的配合。
3. **如何利用这个自由度？** 解析构造、继续学习、适配和成熟模型相对调整提供具体用法。

主文依次为定义、有限位置基、学习与兼容性、成熟模型固定表部署、构造与学习收益。Cosh、BM、几何、固定支持和共适应各自推进其中一个问题。论文价值由这些联系共同形成，不能按模型大小、日期或“赢家”排列成实验清单。

当前科学正文9页，总计52页（2026-09-13）；此前版本的两轮独立 PDF-only 审稿及修订已完成，见 [重构交付](STORY_RESTRUCTURE_20260912.md) 与 [评审索引](pdf-review-rounds/index.md)。20260909 旧系列与 20260912_story 两轮分别计数；旧“十轮计划”不是十轮完成。页数与哈希是交付快照，后续改稿需重新核验。

新增A29（区间开发摘要）和A30（条件深度数学）见[证据索引](evidence/index.md)。A01–A28全部保留；本轮新增状态不是“全窗口方法已成功”。

## 2. 优先级：重要性不等于证据强度

**论证重要性、证据强度、后续工作优先级分开判断。** 附录位置不表示科学价值低；探索性结果可以提出重要问题；大效果不能取代干净识别。

| 层级 | 论证职责 | 当前资产 |
|---|---|---|
| P0 论证骨架 | 定义研究对象、识别作用、刻画位置基和学习关系、提供系统构造 | A01–A08：三seed固定支持、M4多形状、full-pair几何与有限窗、range/crossing/slot、Cosh、成熟同支持 |
| P1 实际后果与设计启发 | 把认识推进到学习、成功读出、自然任务和可迁移构造 | A09–A17：MLA、750M、8B、EOS、QK、BM QA、C2、C42、placement |
| P2 完整展开与范围 | 丰富条件、检验简化解释、保留完整正负结果 | A18–A27：454M、1.485B、五架构、Video DiT、后续8B、FullLagP2、base补偿、计数反例、跨模型和选择效应 |
| P3 研究储备 | 已有方案/代码，或邻接但尚未完成的研究问题 | constrained learnable-z、Llama两实现、未完成同面板交互、稀疏/算子/KKT后续；见下一阶段计划 |

证据状态分别登记：数学推导及数值核验、原始行可复核、带来源摘要、报告依据、开发面板、待执行、无效/被纠正。文件自称“Tier 0”、日期最新或目录归入 evidence 都不能自动提高证据强度。

## 3. P0：各项成果怎样共同成立

### A01 固定支持配对训练：识别自由度

151.9M，B=W=256、K=32、seeds42/137/256，每臂499,974,144 tokens；端点、log-span、初始化、数据顺序、优化器和预算匹配，只动30个内部指数。固定范围 Cosh-minus-FMRoPE/uniform 的 NLL 差在512/1K/2K为 −0.28073/−0.17599/−0.14571，9个seed×OOD格方向一致；256处约+0.026。

**为何重要**：从base/range中单独识别内部铺点，是后面所有分配设计讨论的实证起点。与A02联合回答“这个变量真实且不限于一条曲线”。来源：`evidence/EXACT_RANGE_151M_3SEED_RESULT_20260820.json`。未追回的绝对四格见§8。

### A02 多形状factorial：设计空间的宽度

50.9M，2 bases×2训练长度×3 head dimensions，12配置×3 seeds，各配置严格固定端点。reference Cosh改善7/12，预指定1.25×Cosh改善10/12，deformation-matched exponential改善9/12。**10/12不能安到reference Cosh上。** Cosh与Exp差异按原区间呈现。

**为何重要**：多个非均匀解析形状可以提供有效选择，强化研究对象的设计价值。128步、重复构造语料、约0.01 NLL量级属于短预算多配置证据，与A01较长配对训练互补。来源：`rebuttal/rebuttal_0723/theory_results/m4_exact_range_factorial_evidence_20260726.json`；协议见当前a5。

### A03–A04 完整位置基与有限窗：给出准确的数学对象

每个pair提供 `span{cos(ωΔ),sin(ωΔ)}`。白化cross-Gram的canonical overlap与block Gram的Rényi-2 effective rank满足恒等式 `r2=2K/[1+(K−1)c_bar]`。慢频率极限趋向共同的 `span{1,Δ}`。

A01实际min(ωL)=1.1892，不能直接套ωL≪1。有限窗计算建立连接：b=256、K=32、L=256最慢8对r2=2.11474、最慢4对2.00361；旧b=500K、K=64、L=4096的23对锚点2.00013。解析Gram与独立求积相符。

**为何重要**：不同频率点不等于不同的位置方向，有限旋转预算值得重新组织。几何刻画供给的位置基，后续学习和任务测量说明模型怎样使用它。来源：`foundations/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md`、`evidence/FINITE_WINDOW_SLOW_RANK_RECEIPT_20260911.md`；当前复算在`figs/make_story_figures.py`和`verify_explicit_geometry.py`。

### A05–A06 范围、交叉与槽位：刻画学得使用

A01权重在target-matched range retargeting下，Cosh-minus-uniform在512/1K/2K转为+0.06032/+0.22720/+0.45959，展示分配与运行范围的关系。

50M weights×runtime-table PPL为 `[[7.14,76.20],[23.05,7.16]]`；151.9M在1024的两seed tail NLL为 `[[3.426,5.776],[4.455,3.479]]`。行是训练权重，列是运行表，各自匹配受到偏好。同多重集只置换内部槽位时，OLMo tail NLL 3.10423→6.86493，Qwen64K 0.7000→0；同步变换Q/K的精确补偿关系解释安装方式的作用。

**为何重要**：模型学得了如何使用位置基，静态表分数不能单独决定成熟模型表现；这为native-relative调整提供动机。来源：foundations §5.2、`attention-aware-retrofit/evidence/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULTS_20260823.json:small_model_crossing`、`attention-aware-retrofit/results/coupling-transfer/SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md`。151M crossing是seeds137/256，不是3seed。

### A07 Cosh：可系统构造的实例

显式凸密度目标有唯一Cosh minimizer；inverse CDF、有限网格与端点锚定产生可安装表。定理在写明的functional内完整成立，目标表达设计先验；`tau=d/sqrt(L)`是接受实测检验的参考工作点。

**为何重要**：从定义走向系统构造。A02展示更广形状空间，A09以后展示实际收益。保留定理的完整内容与构造价值；历史错误的KL二阶增益、c_coll循环校准和“下游全局最优”不参与此论证。推导在当前`appendix/a1_proofs.tex`，历史幸存定理和纠正在rebuttal索引。

### A08 成熟同支持干预：识别延伸至已有权重

OLMo held-out 9×20面板geo/ramp/derived为0.56/61.04/60.47%；Qwen为57.75/64.00/66.50%。同checkpoint固定端点、pair数和配对内条件，检验内部点位仍能改变行为。

**为何重要**：与A01形成跨学习阶段呼应，随后BM、C2、placement回答如何利用。来源：`data/curated/frozen_fixed_support_mature_20260823.json`、`attention-aware-retrofit/evidence/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULTS_20260823.json`；构造协议见a6。

## 4. P1：模型收益与可用设计

| ID / 资产 | 已知结果 | 增加的认识与来源 |
|---|---|---|
| A09 / 432M MLA三seed | K=16；8K Geo/Cosh PPL35.4/35.8，16K138.8/95.6；同wavelength blend后117.9/71.1，完整24/32K保留 | 少旋转通道中的学习价值与组合用途；`data/curated/table18_mla_3seed_aggregate.json`，缓存边界见§8 |
| A10 / 750M续训 | 同起点500M tokens；4K22.0/22.3，16K45.1/24.4；8K answer-token exact 0/40→31/40 | 继续学习可利用新分配，生成补充PPL；`data/curated/phase15_750m_continue_result_20260306.json`。无A12的终止EOS条件 |
| A11 / 8B适配及来源使用 | 300步pair，8/16/32K PPL6.82/108.96/991.48→10.07/24.07/127.91；source-removal ΔNLL−0.0095/+1.5055 | 长程来源使用进入成熟大模型；窗口内代价和同adapter任务负结果解释读出阶段；`data/curated/lora_longalpaca_temporal_s42_20260712.json`、`llama8b_causal_source_use_s42_20260714.json`；任务见`rebuttal/EXPERIMENT_THEORY_REVIEW_20260720.md` E5–E7 |
| A12 / 完整答案+终止EOS | 每长度100提示；Native/Cosh在4/8/16K为95/100、18/98、0/60%；匹配+100 query-gap和+32 EOS阶段，单seed | 直接回答成功读出，连接A11来源使用；物理训练≤4K但有长目标相位暴露；`rebuttal/rebuttal_0723/theory_results/evq_query_gap_realized_eos32_20260728/FINAL_METRICS_AND_LINEAGE.json` |
| A13 / selective-QK QA | 2Wiki F1：8K0.07/21.48，16K0/8.57，独立适配器 | 位置相关投影的学习进入自然输出；`rebuttal/rebuttal_0723/theory_results/olmo2_qk_phase_adaptation_20260729/metrics.json` |
| A14 / BM自然QA与强对照 | 778输入中长层631个；五任务task-equal whole-response F1 21.62→25.44，+3.82pp CI[1.32,6.29]；另72提示面板的48长提示BM/Uni/officialYaRN/MrPro为51.32/32.12/6.94/2.78 | 成熟识别转成实际构造；自然QA与六任务各自作为证据；`docs/research/ROPE_OLMO_BM_FIVE_QA_RESULT_20260908.json`、`ROPE_OLMO_BM_RESULT_20260908.json` |
| A15 / C2两参数迁移 | 无Qwen重拟合，64/128K67.75/57.25，对应完整64点67.25/54.50；native窗口82→71.25 | 有效表可有紧凑、可迁移结构；`attention-aware-retrofit/evidence/LOW_DIM_COUPLING_GPU_RECEIPT_20260901.json` |
| A16 / C42受控形状对 | 同support/band/Σm42/增量质心，350个dev提示43.6714/54.4000，+10.7286pp，75胜26负249平；另16文档NLL2.9417/2.8309 | 相同总量仍有内部形状信息，属于开发面板的受控发现；`ds_workspace/recon_20260910/work/jsonl/olmo_c42/`，`code/coverage_theory_20260911.py`，`verdicts/HEADLINE_20260911.md` |
| A17 / placement与完整配置 | GemmaK128 index/direct +6.19pp CI[2.81,9.63]；Qwen0.5B full13对YaRN +6.09pp CI[2.76,9.58] | 连续profile到有限网格的安装选择；后者是table–amplitude联合收益；`attention-aware-retrofit/evidence/K128_COORDINATE_CONFIRMATION_RECEIPT_20260901.json`及`K32_NORMALIZED_INDEX_FULL13_CONFIRMATION_RECEIPT_20260901.json` |

A11的native-endpoint Geo与midpoint EVQ属于conversion对比；A12/A13是后续独立适配，不能拼成一个adapter的全套能力。这些协议在首次定义处说清，后续直接讨论结果，不逐句防御。

## 5. P2：完整展开、反例与其他高价值资产

| ID | 保留什么认识 | 来源与位置 |
|---|---|---|
| A18 | 454M四臂：同repo fixed-ramp下16K157.7→107.5；teacher-forced PK保留独立定义，fixed-ramp不是officialYaRN | a2；`data/curated/table2_evq_yarn_454m_passkey_10pct.json` |
| A19 | 1.485B released-RoPE16K182.73→159.64，126/128文档同向；from-init trainer差异另列 | a6；`rebuttal/rebuttal_0723/theory_results/OLMO2_1B_RELEASED_ROPE_BASELINE_20260725.md` |
| A20 | 五架构GQA/MLA完整表非单调，不能概括“压缩越多收益越大”；提供预算与架构关系线索 | a2；`docs/exp/2026-03/2026-03-20_gqa_mla_125m_compression_ablation.md`，report-backed |
| A21 | VideoDiT129.6M、32→128frames，远端去噪MSE约−35%，单seed；另一模态的后果 | supporting appendix；`evidence/VIDEO_DIT_HEAD_TO_HEAD_SEED42_RESULT_20260826.md` |
| A22 | 独立516步8B RULER-mix：16K0.29/14.03%，8K代价完整保留 | a6；`rebuttal/rebuttal_0723/theory_results/llama8b_matched_ruler_mix_20260726.json` |
| A23 | FullLagP2：完整sin/cos几何引出的另一构造，保留Qwen1.5B小面板和迁移全任务 | a9；`docs/research/ROPE_QWEN15_FULL_LAG_P2_RESULT_20260907.json` |
| A24 | scalar-base geometric fitting部分恢复mismatch：Cosh权重23.05→9.63，原生匹配7.16 | a1；foundations完整谱基报告§6；说明兼容性可有多种调整手段 |
| A25 | 12profile计数拟合与释放反例：几何描述可成立，能力预测仍需任务检验 | a8/a9；`figs/profile_diagnostic_inputs.json`、`ds_workspace/recon_20260910/theory/RELEASE_AXIS_20260911.md` |
| A26 | BM的Qwen3B/7B128K完整观测与小面板区间；不隐去更偏好MR的格 | a3/a7；`docs/research/ROPE_BM_TRANSFER_RESULT_20260908.json`、`ROPE_QWEN7_BM_RESULT_20260908.json` |
| A27 | 具体干预中的NLL/任务分离、开发选择后反转、gain×table条件性 | `ds_workspace/recon_20260910/verdicts/`；不把反向实例称为全任务定律 |
| A28 | 成熟full-z与Q/K联合学习、直接z pilot：已有真实联合学习，phase-shell门未通过；后续dense-LM recovery尾部NLL改善而全序列变差、2Wiki近持平 | 研究背景；`attention-aware-retrofit/evidence/COADAPTIVE_ALLOCATION_ORACLE_RESULTS_20260825.json`及配套report；`scripts/lib/rope/fixed_support_z.py`真实存在。它不是缺失的scratch配对实验，也不能概括为“学z已失败” |

早期Algorithm1盲测、learnable-tau softplus死区、99-run重分析，成熟Native-isotonic的endpoint tradeoff、无效Hotpot stress、Gemma参考长度修正，以及step42/LongBridge样本外反转均在R1详账和分类index保留。具体失败帮助检验解释，不自动否定整个方法家族。

## 6. 理论与实验的连接任务

| 内容 | 当前最有价值的用途 | 关系 |
|---|---|---|
| x=a+Rz、native-relative d、radix累积、gain | 定义干预和可比对象 | 基础语言，贡献由后续推导、识别和收益成立 |
| full-pair、canonical overlap、rank identity | 精确刻画有限位置基 | P0理论核心 |
| Cosh凸目标与inverse CDF | 系统构造的闭式实例 | 与factorial和学习结果相连 |
| slot permutation与Q/K精确补偿 | 解释谱与安装，联系共适应 | P0关系刻画 |
| BM离散粗糙度与Poisson forcing | 展示边界匹配构造，区分Pro/Uni/BM | 构造依据不自动等于任务最优性 |
| 可读窗条带与覆盖计数 | 描述可达结构，提出预测 | 结构保留，能力性接受释放/迁移检验 |
| NLL、Fisher、逐槽梯度 | 具体诊断需要匹配任务测量 | 失败限制相应选型规则，不以旧标题升级结论 |

三个子代理的理论、实验和论文评估与主代理综合，统一归入 `docs/research/next_stage_20260912/`。该计划回答下一阶段新增什么认识，避免只扩benchmark或反复包装已有曲线。

## 7. 后续优先级的初始依据

以下是2026-09-12资产整理的初始依据。当前目标与排序以[下一阶段计划](../../docs/research/next_stage_20260912/index.md)为准。当时记录S1首个block已启动、尚无端点；这不是本轮核验的实时运行状态。

1. **先追回已有信息**：151M绝对四格、MLA/M4缓存身份；找到匹配hash/seed/协议的原件才能补齐。
2. **同面板覆盖检查**：Native/MR(s)/BM(s)、table×gain；已有格复用，缺格显式列出，350/180不能拼成同一交互。
3. **受限learnable-z主比较**：Pro§8.1提出固定端点、单调gaps、配对预算。新审查已找到full-z实现及A28成熟联合学习，但缺少学习期scratch匹配结果。当前推荐实际2K、两个support、Geo/Cosh/full-z三seed充分训练，窗口和初始化作为明确扩展；不因便宜默认只补256臂。旧unconstrained frequencies/learnable-tau不是等价对照。资源和时间由作者协调，排序与合同详见下一阶段计划。
4. **有科学目的的稳健性扩展**：例如MLA精确端点/明确holdout、750M或8B第二seed；检验什么依赖比覆盖多少模型更重要。
5. **研究储备独立保留**：Llama两套60方向、稀疏/压缩注意力、算子改变通过`experiments/index.md`查询；Agent-Range/S6远端开发分支已完成有限收尾并冻结，不进入论文第四主线，local selftest仍不是任务结果。

Pro指导书已原样保存于[外部指导](external-reviews/pro-guidance-20260911/index.md)。它是参考材料；用户确定的“建立成果联系、保留价值、不防御性写作”控制当前叙事。评分预测不作为结论；750M的strict AR按实际scorer定义，不替代EOS。

## 8. 来源强度与尚未追回的记录

- **来源与解释分开**：原始rows/运行记录/实际实现决定测量；摘要报告解释口径；本档组织价值；PDF证明写了什么。冲突沿来源核对，不按“最新日期”机械裁决。
- **151M绝对四格**：权威配对差在，匹配raw绝对值未追回。旧pending汇总不补齐，跨策略绝对排序未由现有摘要确定。
- **MLA原始评价与缓存**：本轮实验代理找到本地`results/eval_3seeds_full_results.json`，SHA为`1e44d30bb880e4b7427ae55bd7034782989152bd2afca9217495f9b8ece30953`，与curated owner相符。已核对注册evaluator，共享5M-token cache、RandomState9999、每长度8窗口、全序列目标。准备代码用FineWeb-Edu sample-10BT train split shuffle99999/buffer10000；旧revision与token-array hash缺失，不能认证文档独立holdout。原始评价JSON可达和数据独立性是两回事。
- **M4缓存**：WikiText2 train/validation分开构造、各自repeat/trim、4个RNG9999 offsets；旧Arrow revision缺失。预算和统计单位见a5。
- **可达性独立登记**：tracked/local-ignored/Git-history/remote-only/missing不是实验成功标签。缺本地raw不表示未运行，报告依据不等于刚复核原始行。
- **历史Git原件**：旧paper与falsification_benchmark在main_0726；被精简curated件与pre_rebuttal报告部分在main。恢复时写实际ref/path/hash；旧a1_proofs行号不对应新稿。
- **指标与操作身份**：NLL/PPL、teacher-forced PK、answer-token exact、完整字符串+终止EOS、RULER部分分、whole-response F1分开；0.074系数、1.1386 cos/sin gain、profile整体倍率与Σm分开。
- **无效证据具体标对象**：buffer-alias的28条hybrid零分、名义16K实6827token旧Geo95%、无原始测量的单槽数值不复用；不由此否定方法族。

## 9. 按任务查询

| 要做什么 | 入口 |
|---|---|
| 看当前成稿、构建、打包 | [paper-2027/index.md](../index.md) |
| 从主张找公式/图表 | [EXPONENT_CLAIM_EVIDENCE_MAP_20260909.md](EXPONENT_CLAIM_EVIDENCE_MAP_20260909.md) |
| 从资产ID找来源、位置、可达性 | [evidence/index.md](evidence/index.md)，`asset_registry.json` |
| 理论与成熟模型依据 | [foundations/index.md](foundations/index.md)、[retrofit/index.md](attention-aware-retrofit/index.md) |
| BM战役、审计、更正、原始行 | [ds_workspace/index.md](../../ds_workspace/index.md) |
| 下一阶段安排 | [next_stage/index.md](../../docs/research/next_stage_20260912/index.md) |
| 完整四阶段历史、术语争议和旧计划 | [R1快照](history/PAPER_REVISION_HANDOFF_R1_20260912.md)、[历史索引](history/index.md) |
| 整个仓库 | [根index.md](../../index.md) |

图表复核在`paper-2027/figs/`：两个`make_*`、`verify_explicit_geometry.py`、`verify_profile_diagnostics.py`、`verify_recovered_assets.py`、`verify_routing_schedule.py`。它们不执行模型，完整协议仍在附录和来源记录。

## 10. 交接完成标准

读者能说明主要成果如何推进中心命题；每个数字能找到来源、统计单位和协议；正结果、代价与反例可查；完成实验、开发选择、数值核验和计划分清。修改论文按实际变更复核数字与编译，修改目录验证链接与迁移清单。原Top15是历史选材快照，不作为排除EOS/C2/C42/完整压缩/FullLagP2的名单。

## R3：Pro领域分析吸收（2026-09-12）

[2026-09-12完善说明](PRO_FIELD_MAP_REFINEMENT_20260912.md)记录引言研究对象地图、tail-energy构造身份、C2门限、完整M4工作点图与未采纳建议。该次交付快照为PDF49页/正文9页；现版见本档顶部及2026-09-13修订记录。首两轮审稿后是一次作者指导下的完善，不计新独立审稿。保留EOS等高价值生成资产；有限排序反转不升级为全称不存在性定理，Möbius边界ladder不简单归类为改rotary算子。当时计划以实际窗口匹配learnable-z与机制/独立确认推进，不因Pro给的1–2天建议缩短训练。
