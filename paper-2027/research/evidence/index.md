# 论文核心证据索引

资产身份与来源见下表；2026-09-14的解释纠正见[主张映射](../EXPONENT_CLAIM_EVIDENCE_MAP_20260909.md)。优先级表示论证职责；命题性质、干预控制、开发/留出身份、评测覆盖和来源可用性分别判断。所有源路径相对仓库根；`local-ignored`表示本机材料未随Git分发，索引不替代原始流。

机器可读：[asset_registry.json](asset_registry.json)。完整原始目录保持既有路径，避免破坏脚本和哈希来源。

| ID / 重要性 | 资产与科学问题 | 证据身份 / 论文位置 | 来源 |
|---|---|---|---|
| A01 / P0 | **固定支持三seed识别**：只动内部指数能否改变学习与外推 | 三seed配对；带raw hash摘要；§3.1 / Fig1 / a5 | [EXACT_RANGE_151M_3SEED_RESULT_20260820.json](EXACT_RANGE_151M_3SEED_RESULT_20260820.json) (tracked) |
| A02 / P0 | **M4多形状factorial**：设计价值是否超出一条曲线 | 12配置×3seed；摘要；§3多形状段 / a5 | [m4_exact_range_factorial_evidence_20260726.json](../../../rebuttal/rebuttal_0723/theory_results/m4_exact_range_factorial_evidence_20260726.json) (tracked) |
| A03 / P0 | **完整位置基几何**：有限坐标提供哪些位置方向 | 证明和独立数值核验；§4.1 / a1 | [FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md](../foundations/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md) (tracked)<br>[verify_explicit_geometry.py](../../figs/verify_explicit_geometry.py) (tracked) |
| A04 / P0 | **实际配置有限窗结构**：慢频几何怎样落到核心参数 | 有限矩阵与独立求积；§4.2 / Fig2 | [FINITE_WINDOW_SLOW_RANK_RECEIPT_20260911.md](FINITE_WINDOW_SLOW_RANK_RECEIPT_20260911.md) (local-untracked)<br>[finite_window_geometry.json](../../figs/finite_window_geometry.json) (local-untracked) |
| A05 / P0 | **运行范围与权重交叉**：权重如何学得使用分配 | 50M报告、151M两seed摘要；附录兼容性 / 原Fig3 | [FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md](../foundations/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md) (tracked)<br>[SAME_SUPPORT_FROZEN_CHECKPOINT_RESULTS_20260823.json](../attention-aware-retrofit/evidence/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULTS_20260823.json) (tracked) |
| A06 / P0 | **同谱槽位与精确补偿**：频率集与安装方式是否不同 | 置换定理、冻结干预报告；附录兼容性 / a1 | [SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md](../attention-aware-retrofit/results/coupling-transfer/SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md) (tracked)<br>[03_compatibility.tex](../../sections/03_compatibility.tex) (local-untracked) |
| A07 / P0 | **Cosh解析构造**：怎样系统构造有限分配 | 显式凸目标推导；§5.1 / a1 | [a1_proofs.tex](../../appendix/a1_proofs.tex) (tracked)<br>[04_construction.tex](../../sections/04_construction.tex) (local-untracked) |
| A08 / P0 | **成熟同支持冻结识别**：已有权重下变量是否仍有效 | held-out与curated摘要；§3.2 / a6 | [frozen_fixed_support_mature_20260823.json](../../../data/curated/frozen_fixed_support_mature_20260823.json) (tracked)<br>[SAME_SUPPORT_FROZEN_CHECKPOINT_RESULTS_20260823.json](../attention-aware-retrofit/evidence/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULTS_20260823.json) (tracked) |
| A09 / P1 | **432M MLA三seed**：有限旋转通道中的实用价值 | 三seed曲线摘要；§6.1 / Table1 / a3 | [table18_mla_3seed_aggregate.json](../../../data/curated/table18_mla_3seed_aggregate.json) (tracked)<br>`results/eval_3seeds_full_results.json`（本机材料，不随Git同步） (local-ignored) |
| A10 / P1 | **750M继续学习**：继续学习能否利用新分配 | 单seed配对摘要；§6.1 / Table1 / a2 | [phase15_750m_continue_result_20260306.json](../../../data/curated/phase15_750m_continue_result_20260306.json) (tracked) |
| A11 / P1 | **8B适配及来源使用**：长程来源如何进入表征 | 单适配pair；curated与报告；§6末指针 / a6 | [lora_longalpaca_temporal_s42_20260712.json](../../../data/curated/lora_longalpaca_temporal_s42_20260712.json) (tracked)<br>[llama8b_causal_source_use_s42_20260714.json](../../../data/curated/llama8b_causal_source_use_s42_20260714.json) (tracked)<br>[EXPERIMENT_THEORY_REVIEW_20260720.md](../../../rebuttal/EXPERIMENT_THEORY_REVIEW_20260720.md) (tracked) |
| A12 / P1 | **完整答案加终止EOS**：能否成功读出远处答案 | raw-hash lineage；100提示/长度；§6末指针 / a6 | [FINAL_METRICS_AND_LINEAGE.json](../../../rebuttal/rebuttal_0723/theory_results/evq_query_gap_realized_eos32_20260728/FINAL_METRICS_AND_LINEAGE.json) (tracked)<br>[routing_protocol_receipts.json](../../figs/routing_protocol_receipts.json) (local-untracked) |
| A13 / P1 | **selective-QK自然QA**：位置相关适配能否改善自然输出 | 匹配独立适配摘要；§6末指针 / a6 | [metrics.json](../../../rebuttal/rebuttal_0723/theory_results/olmo2_qk_phase_adaptation_20260729/metrics.json) (tracked) |
| A14 / P1 | **BM自然QA及四方法匹配**：成熟构造的任务价值 | QA逐行重算；匹配摘要；§6.3 / Table2 / a7,a9 | [ROPE_OLMO_BM_FIVE_QA_RESULT_20260908.json](../../../docs/research/ROPE_OLMO_BM_FIVE_QA_RESULT_20260908.json) (tracked)<br>[ROPE_OLMO_BM_RESULT_20260908.json](../../../docs/research/ROPE_OLMO_BM_RESULT_20260908.json) (tracked)<br>[ROPE_MRPRO_BM_CANDIDATE_20260908.json](../../../docs/research/ROPE_MRPRO_BM_CANDIDATE_20260908.json) (tracked) |
| A15 / P1 | **C2紧凑profile迁移**：有效配置能否用低维结构描述 | CPU构造和GPU回执；§6末指针 / a9 | [LOW_DIM_COUPLING_GPU_RECEIPT_20260901.json](../attention-aware-retrofit/evidence/LOW_DIM_COUPLING_GPU_RECEIPT_20260901.json) (tracked) |
| A16 / P1 | **C42同总量形状对**：同总位移下的高阶increment形状 | 350开发raw rows；另16文档NLL；§3末指针 / a9 | [ctl_C42.jsonl](../../../ds_workspace/recon_20260910/work/jsonl/olmo_c42/ctl_C42.jsonl) (tracked)<br>[ctl_C42V24.jsonl](../../../ds_workspace/recon_20260910/work/jsonl/olmo_c42/ctl_C42V24.jsonl) (tracked)<br>[coverage_theory_20260911.py](../../../ds_workspace/recon_20260910/code/coverage_theory_20260911.py) (tracked)<br>[HEADLINE_20260911.md](../../../ds_workspace/recon_20260910/verdicts/HEADLINE_20260911.md) (tracked) |
| A17 / P1 | **有限网格placement**：连续profile如何安装到不同K | 预注册确认回执；§6末指针 / a7 | [K128_COORDINATE_CONFIRMATION_RECEIPT_20260901.json](../attention-aware-retrofit/evidence/K128_COORDINATE_CONFIRMATION_RECEIPT_20260901.json) (tracked)<br>[K32_NORMALIZED_INDEX_FULL13_CONFIRMATION_RECEIPT_20260901.json](../attention-aware-retrofit/evidence/K32_NORMALIZED_INDEX_FULL13_CONFIRMATION_RECEIPT_20260901.json) (tracked) |
| A18 / P2 | **454M四臂组合**：分配与部署算子怎样组合 | 三seed表摘要；a2 | [table2_evq_yarn_454m_passkey_10pct.json](../../../data/curated/table2_evq_yarn_454m_passkey_10pct.json) (tracked) |
| A19 / P2 | **1.485B released与from-init**：更大模型中的分配后果 | 报告与文档配对；a6 | [OLMO2_1B_RELEASED_ROPE_BASELINE_20260725.md](../../../rebuttal/rebuttal_0723/theory_results/OLMO2_1B_RELEASED_ROPE_BASELINE_20260725.md) (tracked) |
| A20 / P2 | **五架构GQA/MLA消融**：架构与旋转预算的关系 | 历史report-backed；a2 | [2026-03-20_gqa_mla_125m_compression_ablation.md](../../../docs/exp/2026-03/2026-03-20_gqa_mla_125m_compression_ablation.md) (tracked) |
| A21 / P2 | **Video DiT跨模态**：分配在另一模态有何后果 | 单seed结果报告；a3 | [VIDEO_DIT_HEAD_TO_HEAD_SEED42_RESULT_20260826.md](VIDEO_DIT_HEAD_TO_HEAD_SEED42_RESULT_20260826.md) (tracked) |
| A22 / P2 | **8B后续516步RULER适配**：进一步学习后的任务输出 | 独立适配pair摘要；a6 | [llama8b_matched_ruler_mix_20260726.json](../../../rebuttal/rebuttal_0723/theory_results/llama8b_matched_ruler_mix_20260726.json) (tracked) |
| A23 / P2 | **FullLagP2与迁移**：full-pair几何能否引出其他构造 | 小面板结果JSON；a9 | [ROPE_QWEN15_FULL_LAG_P2_CANDIDATE_20260907.json](../../../docs/research/ROPE_QWEN15_FULL_LAG_P2_CANDIDATE_20260907.json) (tracked)<br>[ROPE_QWEN15_FULL_LAG_P2_RESULT_20260907.json](../../../docs/research/ROPE_QWEN15_FULL_LAG_P2_RESULT_20260907.json) (tracked) |
| A24 / P2 | **scalar-base近似兼容恢复**：不等谱关系能否近似补偿 | 历史报告；a1 | [FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md](../foundations/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md) (tracked) |
| A25 / P2 | **profile计数及释放反例**：静态结构是否预测能力 | 12表重建与后续报告；a8,a9 | [profile_diagnostic_inputs.json](../../figs/profile_diagnostic_inputs.json) (local-untracked)<br>[RELEASE_AXIS_20260911.md](../../../ds_workspace/recon_20260910/theory/RELEASE_AXIS_20260911.md) (tracked) |
| A26 / P2 | **BM跨模型完整对照**：配置作用怎样依赖checkpoint | 小面板配对摘要；a3,a7 | [ROPE_BM_TRANSFER_RESULT_20260908.json](../../../docs/research/ROPE_BM_TRANSFER_RESULT_20260908.json) (tracked)<br>[ROPE_QWEN7_BM_RESULT_20260908.json](../../../docs/research/ROPE_QWEN7_BM_RESULT_20260908.json) (tracked) |
| A27 / P2 | **NLL任务分离与选择效应**：简化选型解释怎样被检验 | 报告和部分raw；讨论/历史判决 | [NLL_VS_TASK_20260911.md](../../../ds_workspace/recon_20260910/verdicts/NLL_VS_TASK_20260911.md) (tracked)<br>[GAIN_TABLE_2x2_FINAL_20260911.md](../../../ds_workspace/recon_20260910/verdicts/GAIN_TABLE_2x2_FINAL_20260911.md) (tracked)<br>[STEP42_RESULT_20260911.md](../../../ds_workspace/recon_20260910/verdicts/STEP42_RESULT_20260911.md) (tracked) |
| A28 / P2 | **成熟full-z与Q/K联合学习历史**：直接学习分配已有何种证据与反例 | 本地报告/compact回执/实现；后续研究背景，尚未提升当前正文 | [COADAPTIVE_ALLOCATION_ORACLE_RESULTS_20260825.json](../attention-aware-retrofit/evidence/COADAPTIVE_ALLOCATION_ORACLE_RESULTS_20260825.json) (tracked)<br>[COADAPTIVE_ALLOCATION_ORACLE_RESULT_20260825.md](../attention-aware-retrofit/results/adaptation-coadaptation/COADAPTIVE_ALLOCATION_ORACLE_RESULT_20260825.md) (tracked)<br>[DIRECT_Z_FIXED_SUPPORT_PILOT_20260824.json](../attention-aware-retrofit/evidence/DIRECT_Z_FIXED_SUPPORT_PILOT_20260824.json) (tracked)<br>[fixed_support_z.py](../../../scripts/lib/rope/fixed_support_z.py) (tracked) |

## 新增资产

| ID | 内容 | 身份与来源 |
|---|---|---|
| A29 | 固定表区间开发对照：Llama两表端点相同、中段不同；OLMo324行mini权衡 | [记录汇总](../../figs/interval_development_inputs.json)、[结果owner](../../../docs/research/next_stage_20260912/BAND_NEXT_STAGE_RESULT_20260913.md)；report-backed，非本次raw复核；§6.2末指针 / App.H |
| A30 | 区间深度条件解、加权解与周期反例 | [证明](../../sections/05_threeband.tex)、[独立校验](../../figs/verify_interval_design.py)；条件数学，不是已验证模型方法；§6.2末指针 / App.H |
| A31 | Llama S=4固定表同口径区间确认：log-gain中点相对BM/MrPro/C42的AUC与worst提升、Native FWE反例及两条修复路线的收束 | [结果owner](../../../docs/research/next_stage_20260912/LLAMA_S4_RANGE_CONFIRM_AND_INTERFACE_RESULT_20260913.md)；Core-6×8/16/32K×12行/格完全配对；后续full-z CAL和shape×gain哨兵为开发证据；非跨benchmark SOTA；当前正文尚未据此改写 |
| A32 | Llama S=8冻结经验规则迁移：共同32K桥接、64K强对照与Native代价 | [结果owner](../../../docs/research/next_stage_20260912/LLAMA_S8_SCALE_TRANSFER_RESULT_20260913.md)；Core-6低6；64K优于MrPro、与BM持平，Native 8K显著退化；未形成匹配区间AUC |
| A33 | OLMo S=8冻结经验规则迁移：4/16/32K采样区间及两个gain点 | [结果owner](../../../docs/research/next_stage_20260912/OLMO_S8_SCALE_TRANSFER_RESULT_20260913.md)；Core-6低6；完整配置AUC优于BM/MrPro，绝对32K低且Native未测 |
| A34 | Qwen2.5-1.5B S=2冻结确认：开发低6、独立追加12及累计18 | [结果owner](../../../docs/research/next_stage_20260912/QWEN_S2_MIX075_RANGE_RESULT_20260913.md)；Core-6×32/48/64K；累计AUC对BM区间为正、Native 32K正差；对MrPro/C42未确认 |
| A35 | 功能相位区间的条件构造与band内固定坐标倍率迁移 | [推导](../../../docs/research/next_stage_20260912/ROPE_FUNCTIONAL_CONSTRAINTS_AND_Z_TRANSPORT_20260914.md)、[CPU核验](../../../docs/research/next_stage_20260912/rope_design_theorems_cpu_20260913.json)；条件数学与变量控制，不是任务最优表或GPU胜利 |
| A36 | OLMo S=4→8 fixed-u倍率迁移反事实 | [结果owner](../../../docs/research/next_stage_20260912/OLMO_S8_FIXED_U_TRANSPORT_RESULT_20260914.md)；Core-6×4/16/32K×6行/格；同prompt/gain的fixed-u AUC显著低于fixed-m，终止该迁移分支 |
| A37 | TailSpline精确有限网格构造及同位移T/C分解 | [方法与评测合同](../../../docs/research/next_stage_20260912/TAILSPLINE_ROPE_METHOD_AND_UNIFIED_EVAL_20260914.md)、[CPU验证](../../../experiments/fixed_rope_three_interfaces_20260913/tailspline_verification.py)；one-sided唯一解是TailSpline、symmetric唯一解是BM；CPU不选择边界条件 |
| A38 | YaRN–MrPro等剂量单交叉后移对照 | [理论审计](../../../docs/research/next_stage_20260912/MRROPE_YARN_EQUAL_DOSE_PRINCIPLE_AUDIT_20260914.md)、[CPU验证](../../../experiments/fixed_rope_three_interfaces_20260913/dose_matched_yarn_mrpro_verification.py)；`n=17`唯一`S*=7.51324282212058`、等总log位移、单交叉；尚无模型性能结论 |
| A39 | TailSpline–MrPro Llama S4经典两臂判决 | [结果owner](../../../docs/research/next_stage_20260912/TAILSPLINE_LLAMA_CLASSIC_RESULT_20260914.md)；Full-13与PPL46全量配对raw；Full-13 `+3.20pp`、NIAH `+3.75pp`、PPL AUC `−0.00449`，3/3方向通过；跨模型见A40，机制仍待解混 |
| A40 | TailSpline–MrPro OLMo S4跨模型确认 | [结果owner](../../../docs/research/next_stage_20260912/TAILSPLINE_OLMO_CLASSIC_RESULT_20260914.md)；Full-13与PPL46全量配对raw；Full-13 `+49.23pp`、NIAH `+64.45pp`、PPL AUC `−3.853`，13任务差全部为正；机制仍混入总剂量 |

## 使用时的解释边界

- **A01**：matched support下仅改变30内点，三seed在2×/4×/8×均改善；几何对照配置源自FMRoPE，不作为方法排名。绝对四格NLL尚未追回，配对差已确认。
- **A02**：128步；reference7/12，1.25×10/12，Exp9/12。
- **A03**：block whitening对象，不是模型loss预测器。
- **A04**：b256不能直接套ωL≪1。
- **A05**：50M/151M指标不同；range差值不是四格绝对排序。
- **A06**：固定权重置换与同步Q/K补偿分开。
- **A07**：唯一性相对stated functional；τ为参考工作点。
- **A08**：与BM改变范围的应用对比分开。
- **A09**：历史cache不能认证文档独立holdout。
- **A10**：answer-token exact不含终止EOS要求。
- **A11**：native-endpoint vs midpoint conversion；同adapter任务负结果保留。
- **A12**：单seed；物理≤4K而训练暴露长目标相位。
- **A13**：与EOS/8B不同adapter和协议。
- **A14**：自然QA631长层与48长RULER提示不混；F1含触顶响应。
- **A15**：无Qwen重拟合；native窗口代价保留。
- **A16**：开发发现，不等于独立泛化确认；固定band、倍率与单位increment mass时，质心由总位移决定。
- **A17**：K与checkpoint共变；full13为table×amplitude联合收益。
- **A18**：repo fixed-ramp不是officialYaRN；PK为teacher-forced。
- **A19**：from-init trainer差异与released配对分开。
- **A20**：完整五配置非单调。
- **A21**：MSE和文本生成端点不同。
- **A22**：与300步A11不可拼接。
- **A23**：全任务与迁移负格保留。
- **A24**：冻结近似恢复不能代替训练期纯z归因。
- **A25**：拟合与预测验证分开。
- **A26**：区间与样本量，不将不显著格称确定反转。
- **A27**：350/180不可拼接交互；局部反例非普遍定律。
- **A28**：成熟phase-shell oracle非scratch matched比较；门失败与后续recovery分开。
- **A31**：AUC/worst强结果与Native FWE负格同时保留；开发暴露面板不冒充最终独立SOTA。
- **A32**：只支持当前64K端点和8K Native配对；经验band、0.75和gain未由理论推出。
- **A33**：候选与基线是frequency×gain完整配置比较；两个candidate gain点不能证明普遍校准规律。
- **A34**：累计18含已开封低6；独立追加块三项AUC差均跨0，C42的64K/worst点估计更高。
- **A35**：功能区间给定后才可解析求表；当前缺逐行、角色限定signed C输入，Native旋转误差界在实际中频槽已饱和。
- **A36**：每格6行的机制判别块足以否决当前fixed-u扩展，但不是16/24/8最终样本块；不否决S4直接mix075、z自由度或其他构造。
- **A37**：one-sided边界是假说而非全局smoothness定理；CPU只证明各声明目标的唯一解，不能从连续边界族中选择任务最优条件。历史mix075是近似prior，不能代替精确TailSpline的GPU结果；Core-6也不能冒充PPL/passkey/full-13闭环。
- **A38**：等剂量构造排除了YaRN–MrPro原整数倍率比较中的总位移混杂，但CPU没有证明后移更好；只有按冻结两臂合同取得任务结果后，才能评价back-loading原则，且该分支不插队当前TailSpline–MrPro主判决。
- **A39**：支持的是冻结Llama-3-8B S4、canonical `[18,35]`、同gain合同下的整体方法优势。PPL优势很小且由32K驱动，8/16K轻微变差；QA与若干单任务反转保留。当前数据不证明one-sided边界机制、YaRN胜负或跨checkpoint普适性。
- **A40**：与A39共同支持两个模型族对应canonical S4合同上的TailSpline方法优势，但OLMo历史BM/front-loaded方向已经提供先验，不能称完全独立盲测。高cap率保留；巨大差值不识别`sum(m)`、early transport与tail landing各自贡献。

## 添加或更正结果

新结果在同一记录中给出问题、模型/表/数据/代码身份、比较单位、完整长度与指标、原始输出、来源hash和正负结论；登记到asset_registry并更新本index。新文件入evidence不会自动成为论文结论。已有资产编号保持稳定。

## 跨机器使用

来源路径以Git仓库根为基准。`tracked`或`local-untracked`是整理时快照，新增文件需随Git提交；`local-ignored`只在同步原始镜像后可用。MLA原始评价JSON在`results/`，未同步时使用本页已链接的curated摘要核对现有论文，不声称重算了raw。

## A02的联合工作点图（本轮新增展示）

[绘图脚本](../../figs/make_m4_tradeoff.py)、[可移植输入](../../figs/m4_tradeoff_inputs.json)、[48点CSV](../../figs/m4_tradeoff_points.csv)由原M4记录派生；12配置×4主臂、每点3配对seed，含全部正负点。此图不增加模型实验数量，也不拟合跨配置Pareto前沿。

本轮正文已纳入A37构造及A39/A40 TailSpline两模型结果（§5.2、§6.2、Fig5）；BM自然QA仍属A14。兼容性完整内容移入附录，主要证据身份不变。当前版面以[主张映射](../EXPONENT_CLAIM_EVIDENCE_MAP_20260909.md)为准。

A41：历史原生Std-RoPE三seed对照已追回，见[可移植输入](../../figs/allocation_value_inputs.json)的native_grid_historical及[附录](../../appendix/a5_identification.tex)。严格保持短训练、span匹配、报告聚合身份；不替代A01。
