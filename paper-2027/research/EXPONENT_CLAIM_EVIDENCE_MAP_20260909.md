# Beyond the Base：当前论证与证据安放表

更新：2026-09-13。当前主线：内部指数分配组织有限位置基，参与模型表征学习，并可通过具体构造改善长度泛化与上下文利用。历史映射保留在本文后半部分，仅用于追踪此前位置；当前稿件位置以下表为准。

当前部署目标与设计手段见[研究方向](../../docs/research/next_stage_20260912/PAPER_INTERVAL_DIRECTION_20260913.md)。成熟部署现为§5，学习期构造为§6；旧日期段落保留原有历史位置。

## 当前主张链

| 认识 | 本项发现与作用 | 相邻成果的联系 | 当前稿件位置 |
|---|---|---|---|
| 研究对象 | x=a+Rz 分离端点与内部位置；pi 分离槽指派 | 为纯内部干预和冻结安装定义对象 | 主文 §2 |
| C01 独立作用 | 151.9M 三 seed：只动 30 内点，OOD 三长度全同向 | 给几何分析一个实际需要解释的变量 | §3.1、Fig.1、App. exact-range |
| C02 位置对象 | 完整 sin/cos 子空间、canonical overlap、rank 恒等式 | 把点位差异转成位置方向分配 | §3.2–3.3、Fig.2、App. A |
| C03 有限窗结构 | 核心 b256 网格最慢8对 r2=2.11474 | 使位置基分析落到核心训练参数，非模型loss中介效应证明 | §3.3、Fig.2、App. finite-window-rank |
| C04 学得兼容 | 两尺度 crossing、运行范围反转、同谱置换与补偿恒等式 | 将位置基与权重使用联系，动机转入成熟部署 | §4、Fig.1、完整交叉附图 |
| C05 可构造性 | Cosh 明示密度目标、唯一解、逆CDF、端点锚定 | 从对象和设计偏好到一个可安装实例 | §6.1、App. A |
| C06 多形状价值 | M4：7/12 reference、10/12 preassigned1.25、9/12 Exp | 主变量价值超出一条曲线；Cosh/Exp差异按区间呈现 | §6.2、App. B |
| C07 旋转预算 | 432M MLA三seed，完整8/16/24/32K曲线与共享blend | 学习期构造在少pair架构的实际价值 | §6.3、Table1、App. MLA |
| C08 继续学习 | 750M共享起点续训、完整长度PPL与40-case生成 | 学习已有模型的新分配可影响生成 | §6.3、App. larger-scale |
| C09 适配与读出 | 8B PPL/来源使用、同adapter任务结果、独立516-step后续 | 区分学到长程来源使用与任务转换；保留整组证据 | §6.3、App. Llama、完整曲线 |
| C10 成熟模型内点效应 | OLMo/Qwen同支持冻结干预 | 与学习期C01连接，扩展至固定权重和任务指标 | §5.1、Fig.3a、App. frozen |
| C11 实际部署 | BM构造、同倍率确认、五任务自然QA | 从纯作用到一个有用分配实例 | §5.2–5.4、Fig.3b、App. BM |
| C12 离散安装 | Gemma K128 index/direct gap +6.19 | 连续profile安装到有限网格仍是设计选择 | §5.4、App. placement |
| C13 完整配置 | Qwen0.5B full13 +6.09，32K近等 | 频率表与振幅的实际系统收益 | §5.4、App. index |
| C14 更广学习证据 | 454M、1.485B、selectiveQK、Video DiT | 各自补充组合、规模、适配及跨模态，不混合估计量 | §6末指针；各协议附录完整保留 |
| C15 探索如何指导设计 | 12profile计数拟合及平台释放反例 | 静态描述须经模型及目标任务检验 | 冻结附录，phase几何与profile连续安放 |

## 新增区间设计内容

- C16：固定表全窗口目标与三个设计变量，§5.3；目标不是已解决方法。
- C17：端点与区间条件最优、Native/端点加权解、周期反例，App.H；来源为当前证明与独立数值校验。
- C18：Llama中段差异、OLMo mini权衡，App.H；A29为报告依据开发摘要，A30为数学资产。新增实验未冒称raw重验或独立确认。

## 2026-09-12 二次回查：此前未充分使用的资产

用户要求重新思考交接档中的高价值资产后，回读战役总账、foundations 报告、125M 压缩消融、C2 CPU/GPU owners、FullLagP2 JSON、完整答案EOS lineage、selective-QK metrics、C42 原始行、四方法 controls、gain 析因判决。结论不是复刻当前Top15排序。

| 资产 | 新增价值与采用位置 | 直接来源 |
|---|---|---|
| 完整答案+EOS | 提升主文§5：从来源使用到成功读出，不能由旧750M固定长度探针替代 | rebuttal/rebuttal_0723/theory_results/evq_query_gap_realized_eos32_20260728/FINAL_METRICS_AND_LINEAGE.json |
| selective-QK QA | 提升主文§5：独立适配支撑自然题的长度迁移；详细协议仍在成熟模型附录 | 同目录 olmo2_qk_phase_adaptation_20260729/metrics.json |
| C2两参数 | 提升主文§6：有用分配可以有紧凑描述，跨checkpoint无重拟合保留长端行为 | attention-aware-retrofit/evidence/LOW_DIM_COUPLING_GPU_RECEIPT_20260901.json；CPU_LOW_DIM_COUPLING_LAW与GPU_RESULT报告 |
| C42受控对 | 提升主文§6：相同support、总位移和增量质心仍有形状信息；按dev面板呈现 | ds_workspace/recon_20260910/work/jsonl/olmo_c42/ 两个350行JSONL；code/coverage_theory_20260911.py两构造；HEADLINE §二 NLL报告 |
| 四方法同场对照 | 主文§6增加Uni/officialYaRN；完整短长表入附录 | docs/research/ROPE_OLMO_BM_RESULT_20260908.json:experiments.existing_controls与seed_replication、bm_vs_existing_controls |
| FullLagP2 | 新附录：把全sin/cos几何转成实际表的另一个正面例子，保留长端及迁移全分项 | docs/research/ROPE_QWEN15_FULL_LAG_P2_RESULT_20260907.json及candidate JSON |
| 125M五架构压缩 | 新附录完整表：展示allocation在架构约束中的价值，不把非单调五配置说成压缩越多收益越大 | docs/exp/2026-03/2026-03-20_gqa_mla_125m_compression_ablation.md（report-backed） |
| gain 2×2 | 不采用跨350/180面板拼接；已有Qwen匹配小面板保留 | GAIN_TABLE_2x2_FINAL §五明确说明BM两gain未在180行测，故§二不能作同面板交互证据 |
| base-only补偿 | 保留为共适应的额外解释材料：历史50M的scalar geometric fitting也能恢复部分compatibility；不借此混淆纯z训练 | foundations/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md §6 |
| 算子压缩旁线 | 查阅RECENT_RESULTS与EXPERIMENT_VALUE：改变operator和拟合目标，保留项目资产，不并入本稿内部指数主张 | experiments/rope_operator_family/ |

重算：figs/verify_recovered_assets.py；source摘要和scores在figs/recovered_asset_inputs.json。未运行模型。新source可直接从上表定位，不受旧Top15是否列入约束。

## 数字与来源规则

C01/04/06/07/08/09/10/11/12/13/14 的 owner 沿用 PAPER_REVISION_HANDOFF_20260911.md §8 与下方历史数值索引；逐项追踪真实指标、训练seed或提示单位，不将PPL、官方部分分、完整字符串和EOS视作同一个指标。

- C03 owner：research/evidence/FINITE_WINDOW_SLOW_RANK_RECEIPT_20260911.md；复算脚本 figs/make_story_figures.py，结果 figs/finite_window_geometry.json；独立解析Gram与Gauss-Legendre求积相符。
- C09 配对任务结果：rebuttal/EXPERIMENT_THEORY_REVIEW_20260720.md E5/E6/E7（report-backed）；正面概率与来源使用沿用 curated 两JSON，不把不同adapter合并。
- C15 后续释放结果：ds_workspace/recon_20260910/theory/RELEASE_AXIS_20260911.md §六；连续NLL与原350-row任务拟合分列；不将跨仪器差异当成同指标比较。
- 数字绘图输入：figs/story_figure_inputs.json 保留四个source完整精度与SHA；原五图绘制仍用 figs/figure_inputs.json。
- 核心每项证据均保留或迁移；没有以KB或附录总页数作为删减目标。已知无效证据不作为正面论据恢复。

## 当前章节连接

§2 定义可干预对象 → §3 证明作用并刻画位置基 → §4 说明学得使用和安装关系 → §5 构造并展示学习收益 → §6 在成熟模型中利用同一设计自由度 → §8 汇总正面认识和可实践的设计方式。

## 历史映射（2026-09-09，位置已由上文更新）

# 指数分配稿：主张、公式、图表与引用对应表

2026-09-09。内部审查表；不进入匿名稿件。实验优先级及逐项裁决见 [Top 15](EXPERIMENT_ASSETS_TOP15_20260909.md)，完整文件SHA256见 [source index](EXPONENT_REVISION_SOURCE_INDEX_20260909.json)。新图/表的直接输入及运行检查另见 [figure receipt](../figs/exponent_revision_source_receipt.json)。

## 理论与构造

| 主张/构造 | 公式与位置 | 直接证据与核验 |
|---|---|---|
| 非均匀指数分配可与频率范围分开描述 | §2 `omega=b^(-phi)`；`x=a+Rz`，K≥2、b>1、严格有序正频率表 | 这是定义与控制工具；geometric的z始终等距。形状的行为后果由Top15 #1、#3识别。 |
| 完整位置对象是sin/cos二维子空间 | §4.1 `Q=S_omega^(-1/2) H S_nu^(-1/2)`、`c=||Q||_F^2/2` | `appendix/a1_proofs.tex`给完整trigonometric Gram与相位不变性；foundation report §2保存数值检查。 |
| 平均canonical collision关联有效rank | §4.1 `r2=2K/[1+(K-1)c_bar]` | 同一block-whitened Gram的trace恒等式，完整证明在A1；不是raw entropy rank或LM loss。 |
| 慢频率共享位置子空间 | §4.2 `V_omega→span{1,Delta}`；`2-||Q||²=O(epsilon⁴)` | Uniform[0,L]下证明与展开；标准k/K网格K64,b500K,L4096有23慢pair、r2=2.00013。endpoint-normalized网格的24pair是另一配置。 |
| Cosh是明确变分目标的闭式解 | §4.3平方密度+min(phi,psi) interaction；rho_tau、inverse CDF | alpha>0,beta≥0、正C²、单位积分；严格凸与边界条件导出cosh。正文一次说明surrogate，完整推导与tau规则在A1。 |
| 成熟表的指数位移统一三种操作 | §6.1 `d=log(omega_N/omega')`；频率混合、log-shift、mixed-radix各自公式 | 频率混合是`-log(1-w+w/s)`，不能把w误当log-shift的m；radix乘积取log成为sum。 |
| BM平滑的是radix增量 | 附录A7，epsilon_q=6q(N+1-q)/[N(N+1)(N+2)]，累加得三次m_q | `ROPE_MRPRO_BM_CANDIDATE_20260908.json`保存OLMo N18、Qwen N17、逐项exponents；绘图代码逐项断言相等。 |
| finite-grid profile placement有不同构造 | A7参考K64插值与target local-gap直接计算 | `export_frozen_coupling_transport.py`；K128/K32两份confirmation identity逐字段核对，gain在每一比较内一致。Gemma reference4K是operating reference。 |
| 454M历史scaler的准确身份 | A2 `R_s(omega)=omega/[s^r T(s)^(r/2)]`, T=1+.07log2(s) | `official_yarn.py`中的legacy fixed-index operator；K32、cutoffs6/28、s8、T1.21、cos/sin gain1。与官方YaRN单独命名。 |

## 图表契约与最终表面

使用可复现Matplotlib矢量PDF；新图同时导出PNG预览。论文固定宽度内检查字体、零线、范围、标签、图注和黑白可辨性。基线蓝色/圆点、构造橙色/菱形；co-adaptation矩阵用同一橙色根表达相对匹配格的NLL增量。图源为现有实验，未执行模型。

| 图/表 | 问题与直接主张 | 形式/数据粒度 | 输入与呈现 |
|---|---|---|---|
| `fig_evidence_overview` | 相同端点下改变什么、是否有实证后果？30个interior改变，所有9个seed×OOD length差值为负。 | 有序样点图+四个真实评价长度的paired line；3训练seed，均值不代替单seed | Top15 #1；正文首图。固定range结果不与retargeted结果拼接。 |
| `fig_8b_length_curve` | 在适配窗口之外发生什么？完整8/16/32K PPL曲线显示长程收益与窗口内成本。 | 两条有序长度曲线，log-PPL轴；24 packs/length | Top15 #2；300-step pair，不能与516-step RULER后续pair混用。 |
| `fig_weight_table_crossing` | 为什么第二阶段相对native表调整？权重偏好与其训练表相容的运行表。 | 两个2×2矩阵；左标PPL，右标tail NLL；阴影统一表示diagonal-relative ΔNLL | Top15 #9；50M来自报告§5.2，151M来自`small_model_crossing`，不引用Qwen K32 receipt。 |
| `fig_bm_natural_qa` | 指数调整能否在自然输入获益？五任务均值均增加，macro21.62→25.44%。 | 五任务+task-equal mean的paired dot；样本166/173/119/61/112 | Top15 #6；对778个row_id去重、重算两length strata逐任务均值，再绘631长输入。 |
| `fig_bm_exponent_profiles` | BM与MrPro究竟改了什么？同边界/总量，增量向band中间重分配。 | 两个离散transition width的cumulative profile，N18/N17 | 直接对比recorded exponent arrays；放附录，数值差异不是新的模型实验。 |
| `table_index_full13` / `table_coordinate_confirmation` | 静态表的breadth及不同placement | 分任务官方RULER分数/配对差值CI | macro按task均分；K32两个长度用97.5% CI，K128用95%；旧pilot不pool。 |
| `table_bm_tasks` / `table_bm_qa_all` | 保留模型、任务与生成端点细节 | 三模型六任务分项；自然QA完整输出F1%与EOS计数 | 不把FWE/VT部分分数称exact accuracy；不从generation_config默认值猜实际cap。 |
| `table_evq_ramp` | 训练时分配与后续固定scaler的组合 | 四臂、三训练seed；PPL与teacher-forced PK分列 | 只使用full-sequence summary；不拼接per-document PPL与早期s4单seed数据。 |

三点/四点评价长度是作者明确要求保留的离散模型评价网格，并非时间序列抽样不足；不插值生成新观测。所有caption写实际条件和metric。

## 最近相关工作的实质关系

| 文献 | 本稿比较的实际内容 | 原始来源 |
|---|---|---|
| FMRoPE / Frequency Bands | base选择保持geometric normalized exponents等距；本文固定端点改变内部形状并做配对训练 | [Oka et al.](https://openreview.net/forum?id=PR1PPxvG9Q) |
| YaRN | NTK-by-parts频率混合与cos/sin amplitude；指数位移是混合后的负log | [YaRN](https://arxiv.org/abs/2309.00071) |
| LongRoPE | dimension-wise factors加token-position threshold；位移公式只对应频率因子，不能代替位置阈值机制 | [LongRoPE](https://proceedings.mlr.press/v235/ding24i.html) |
| MrRoPE | mixed-radix累乘转成指数位移累加；BM在同band边界下修改增量形状 | [MrRoPE](https://arxiv.org/abs/2601.22181) |
| LeRoPE | 每pair的log-space scale跨层/头共享，并与weights联合学习；不是因名称不同就与指数分配无关 | [LeRoPE](https://arxiv.org/abs/2607.10134) |
| AdaRoPE | head-specific learned frequencies和attention scaling；本文聚焦几何、显式密度与受控指数问题 | [AdaRoPE](https://arxiv.org/abs/2607.19363) |
| DoPE | 以truncated matrix entropy分析rotated activations低秩并修改PE；本稿比较supplied sin/cos subspaces | [DoPE](https://arxiv.org/abs/2511.09146) |
| Du et al. | 长上下文position/token辨识理论；本稿有限分离区间的basis geometry与学得表征交互是不同对象 | [Du et al.](https://arxiv.org/abs/2605.15514) |
| Gemma | 新加入实验的模型家族来源；具体1.1 checkpoint identity在本地confirmation receipt | [Gemma paper](https://arxiv.org/abs/2403.08295), [官方model card](https://huggingface.co/google/gemma-1.1-2b-it) |

相关工作保持紧凑。核对表为内部工具；正文不复制逐项novelty审查，也不复述一年的试错过程。

## 复现与记录层级

Figure builder重算分任务均值、样本数与部分记录一致性，并绑定原始输入hash；这不等于重新运行训练或生成。历史报告、curated summaries和raw-backed receipts按其实际层级使用。论文源包包含完整TeX、styles、bibliography、所有引用PDF图与成稿；模型checkpoint及原始流单独维护。

## 2026-09-12 Pro领域分析完善

新增Fig. m4-operating-points（附录B）：原M4的12配置×4非均匀主臂联合展示1×差值与weighted OOD差值，每点3seed。源为原factorial owner，派生输入/CSV/脚本在figs/m4_tradeoff*。完整采纳与纠正见[PRO_FIELD_MAP_REFINEMENT](PRO_FIELD_MAP_REFINEMENT_20260912.md)。主文更清楚区分supplied basis与learned usage，并把C2注册保留率门与长端正结果并置；原实验数字与证据身份不变。
