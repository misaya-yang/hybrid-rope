# 既有实验资产 Top 15：主代理裁决

2026-09-09。五个 Luna 分别盘点基础对照、规模与架构、成熟模型适配、冻结调整、机制实验；本表由主代理逐项复核后排序。排序依据是对本文问题的回答力度、比较控制、独立信息及自然任务价值。一个资产可以包含同一科学问题的成组对照；同一训练 run 的多个端点不拆开凑数。数值按各自原始指标保留，不跨模型汇总成绩。

当前完整历史稿件为 `main_0726`（`6bca5abac464b2abcfae63a54dfeabfe2321f919`），另以八月封板 `8e4d567` 校核。当前分支已恢复下列必要的小型结果 JSON；原始流、curated 汇总和报告各保留自己的证据身份。

| 排名 | 实验资产与可成立的主张 | 核心证据、比较单位 | 裁决与稿件位置 |
|---|---|---|---|
| 1 | **151.9M 三种子固定范围训练**：固定频率端点后，内部指数分配改变外推表现。 | 499,974,144 tokens/arm；seeds 42/137/256；固定范围 Cosh−Geo NLL 在 512/1K/2K 为 −0.281/−0.176/−0.146，9/9 seed×length 同向；256 为 +0.026。 | 核心识别证据，首图与正文 §3.1。target-retargeted 反转是同一实验的部署分析，保留完整结果而不另计。 |
| 2 | **Llama-3-8B 匹配适配及来源使用**：设计指数在 8K 匹配 LoRA 后显著改善长文本概率，并增加远端来源依赖。 | 300 steps、一个匹配 pair、24 temporal packs；8/16/32K PPL 6.82/108.96/991.48 → 10.07/24.07/127.91；10 个冻结来源干预案例，gold-block 删除 ΔNLL −0.0095/+1.5055。 | 保留醒目完整曲线和来源干预。二者属于同一适配 run 的不同评价端点，合并一个资产。 |
| 3 | **成熟冻结模型同范围分配对照**：已有权重下，仅改变内部指数即可大幅改变 RULER 表现。 | 相同 endpoints/gain/inputs/decoder；OLMo held-out 9×20 rows：geometric/ramp/residual 0.56/61.04/60.47%；Qwen development 4×20：57.75/64.00/66.50%。 | 正文 §6.1。主要结论是 allocation recovery，不能把 residual 与 ramp 的差异称作已分离的优越性。 |
| 4 | **Qwen-0.5B 单张静态表全13项确认**：静态指数调整同时覆盖原窗口和扩展窗口。 | 新 data seed 202609027，13×20×2=520 generations/arm；64K index/YaRN 51.46/45.37%，差 +6.09 pp，CI [2.76,9.58]；32K 55.92/55.94%。 | 正文主表；全分项在附录。index/YaRN 的幅度分别 1.0513/1.0693，按实际方法比较，不称纯 shape-only 因果。 |
| 5 | **432M MLA 三种子**：有限旋转通道架构中，指数设计改善长程建模。 | K=16，训练8K/500M tokens，seeds42/43/88；PPL8K 35.4/35.8，16K 138.8/95.6，三种子同向。 | 正文训练结果；24K/32K 完整曲线在附录。架构压力测试，不把 rotary channel 数自动等同于 token 稀疏性。 |
| 6 | **OLMo BM 五项自然QA**：中间频段重分配产生实际自然问答收益。 | 778 inputs/arm，其中长输入631；五任务均值21.62→25.44 token-F1%，+3.82 pp CI [1.32,6.29]；每个任务均值均为正。 | 正文独立任务图；完整输出评分、短窗口147条、EOS与长度预算在附录。 |
| 7 | **50.9M M4 固定范围 factorial**：跨 base、训练长度和 head dimension 比较多种非均匀分配。 | 12 configurations×3 seeds，180主臂+12边界臂；预指定1.25×Cosh改善10/12，deformation-matched Exp改善9/12；Cosh-rule 与 Exp 的差异未分离。 | 正文一段、附录完整表。保留多构造结果，tau multipliers/Exp/边界检查属于同一实验族。 |
| 8 | **750M 全参数续训**：从同一 Geo checkpoint 继续学习新指数表，可改善长文本和严格生成。 | 2K→4K，500M continuation tokens，一个匹配 pair；16K PPL45.1→24.4；40个8K passkey trials，AR exact0/40→31/40。 | 正文。明确是共享起点的全参数 continuation；teacher-forced retrieval100/100与AR exact分列。 |
| 9 | **50M及151.9M 权重×运行表交叉**：权重会与训练时的指数基共适应。 | 50M PPL矩阵 [[7.14,76.20],[23.05,7.16]]；151.9M 两训练种子、32 anchors/seed、1K tail-NLL矩阵 [[3.426,5.776],[4.455,3.479]]。 | 从附录提升为正文机制图，连接训练构造与冻结调整。50M报告与151.9M JSON分别溯源，两个尺度作为机制复核成组呈现。 |
| 10 | **454M 指数表×固定 scaler 四臂实验**：训练时的分配效果在共享推理变换后仍然存在。 | 三seed42/123/7，100M tokens/arm，Ltrain2K，10% passkey mix；同 R8 后16K PPL157.7→107.5，8K teacher-forced PK61±3→100±0%。 | 恢复正文短段及完整四臂附录表。实际算子为repo fixed-index smooth-ramp，明确公式，cos/sin gain=1。原始本地镜像未留存不等于结果不存在。 |
| 11 | **BM/MrRoPE 三模型匹配比较**：同样的中间带边界与总缩放并不固定最佳分配形状。 | OLMo独立seed确认16K MrPro/BM2.78/51.32%；Qwen3B128K78.13/70.83%；Qwen7B128K84.44/71.11%。样本数分别24/48、12/24、6/12（短/长，每臂）。 | 正文保留全部模型和长度，支撑 checkpoint-relative 设计结论；Qwen是小样本screen，不能抹掉反向结果。 |
| 12 | **跨有限网格的冻结 profile placement 确认**：profile 的离散放置方式本身有可测影响。 | Gemma K128：新seed，4×80=320 inputs/arm，16K index79.00/direct-gap72.81%，差+6.19 pp CI[2.81,9.63]；K32 Qwen对应确认在64K未分离两者。 | Gemma提升至正文，K32与K128完整对照进附录。两臂内幅度/端点/输入一致；Gemma新输入没有Native/YaRN，不能拼接旧panel假装同场比较。 |
| 13 | **1.485B OLMo 从公开step-0初始化训练**：分配的长程作用延续到约1.5B训练规模。 | 2.097B counted tokens，128 document-disjoint PG-19 docs；16K PPL182.73→159.64，126/128 documents NLL同向；2K/4K有小幅成本。 | 保留附录完整结果，正文用进一步设置指针。Geo用AI2分布式trainer、EVQ用HF单GPUloop；同初始化/recipe不等于bitwise配对轨迹。 |
| 14 | **OLMo selective-Q/K 适配**：相位相关参数的进一步适配可承载自然QA及任务族长度迁移。 | QA与RULER是独立adapters；各300步、显式长相位暴露、物理≤4K。2Wiki200/length：8K F1 0.07→21.48%，16K0→8.57%；RULER8K2.02→31.63%。 | 保留详细附录和独立指标；该成组实验比单一数字检索更接近自然任务。不得把其两个adapters说成同一run的多端点。 |
| 15 | **Llama-8B RULER-family 匹配续训**：已有8K适配继续接受同任务族监督后，指数表影响16K自回归迁移。 | 516 additional steps，13×20 rows/length；8K Native/EVQ94.44/77.60%，16K0.30/14.03%。 | 从附录提升正文。与第2项共享更早parents，但追加训练及评估不同；Native32K仅10/13任务完成，不报完整macro。 |

## 有价值但不占 Top 15 名额的材料

- **129.6M video DiT**：32→128 frames，256 videos、单seed，远端denoising MSE降低35%；保留跨模态附录。它增加breadth，对本文主问题的直接信息少于榜单中的受控文本与成熟模型结果。
- **OLMo最终完整答案+EOS**：matched +100 query-gap/+32 EOS continuation，4/8/16K Native95/18/0%，EVQ100/98/60%；补回附录，明确长相位暴露、同一numeric NIAH family和单训练seed。严格端点值得保留，不与其祖先first-number结果合并。
- **BM跨模型自然NLL**：同16个FineWeb prefixes，OLMo16K ΔNLL−0.8259而Qwen结果接近零或反向；留在证据索引，作为模型依赖性的补充。正文已经有自然QA和三模型匹配任务结果。
- **125M learned-inv-freq/learnable-tau**：保留已有附录。两者来自同一128-token campaign；seed42固定tau5的333.7不能当作三seed均值，learnable-tau的437.9±12.3另行注明。
- **逐pair NoPE/swap、base-only compensation、attention-gradient、dose response**：有机制与反例价值；作为附录或内部解释资料，避免将观察到的梯度或先验代理包装成可用的训练前selector。
- **454M staged continuation、125M GQA/MLA单seed**：保留历史来源。可补充曲线或架构线索，当前已有更充分的训练/架构对照。
- **短序列Phase11/11B模型身份冲突项**：同一组260.2/99.6被标成125M、350M或454M；此项暂不晋升。原因是模型归属/数据身份冲突，区别于第10项来源一致的454M实验。
- **PSR/native sparse相关近期结果**：作为独立研究资产保留，其问题和已验证结果与本稿指数分配主线不同；第三阶段留在discussion。

## 主代理对 Luna 反馈的具体纠正

1. 不采纳“454M完全遗漏”的时间点判断：在其盘点期间，正文段落和正确附录已经恢复。检查最终TeX依赖，而不只看文件是否存在。
2. 将151.9M crossing的错误Qwen K32 receipt改为 `SAME_SUPPORT_FROZEN_CHECKPOINT_RESULTS_20260823.json:small_model_crossing`；逐格核对四个完整精度值和两seed身份。
3. Gemma新确认只含physical/index两臂；从receipt逐字段核对checkpoint、输入hash、gain、decoder及冻参状态。未将旧Native/YaRN rows并入。
4. 8B 32K Native-LoRA未完成全部13任务；保留8K/16K完整比较，32K仅在附录按实际完成范围描述。
5. 454M157.7/107.5采用full-sequence PPL；不混入per-document聚合262.0/237.2或QuALITY full-eval的指标。
6. 理论解析/数值验证是重要资产，另列于公式证据表，不用它替代模型实验计数。

## 数值来源

- 1：`research/evidence/EXACT_RANGE_151M_3SEED_RESULT_20260820.json`。
- 2：`data/curated/llama8b_causal_source_use_s42_20260714.json`；历史 `rebuttal/pre_rebuttal/LORA_LONGALPACA_TEMPORAL_NLL_20260712.md`。
- 3：`data/curated/frozen_fixed_support_mature_20260823.json` 与 `research/attention-aware-retrofit/evidence/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULTS_20260823.json`。
- 4：`research/attention-aware-retrofit/evidence/K32_NORMALIZED_INDEX_FULL13_CONFIRMATION_RECEIPT_20260901.json`。
- 5：`data/curated/table18_mla_3seed_aggregate.json`。
- 6：`docs/research/ROPE_OLMO_BM_FIVE_QA_RESULT_20260908.json`。
- 7：`rebuttal/rebuttal_0723/theory_results/m4_exact_range_factorial_evidence_20260726.json`。
- 8：`data/curated/phase15_750m_continue_result_20260306.json`。
- 9：`research/foundations/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md` §5 与 `SAME_SUPPORT_FROZEN_CHECKPOINT_RESULTS_20260823.json:small_model_crossing`。
- 10：`data/curated/table2_evq_yarn_454m_passkey_10pct.json`；历史 `docs/exp/2026-03/2026-03-03_passkey_mix_results.md` 和 `scripts/lib/rope/official_yarn.py` legacy operator。
- 11：`docs/research/ROPE_OLMO_BM_RESULT_20260908.json`、`ROPE_BM_TRANSFER_RESULT_20260908.json`、`ROPE_QWEN7_BM_RESULT_20260908.json`。
- 12：`research/attention-aware-retrofit/evidence/K128_COORDINATE_CONFIRMATION_RECEIPT_20260901.json` 与 `K32_PAIRED_CROSSING_CONFIRMATION_RECEIPT_20260901.json`。
- 13：`rebuttal/rebuttal_0723/theory_results/OLMO2_1B_RELEASED_ROPE_BASELINE_20260725.md` 给出匹配比较；同目录小写 `.json` 主要保存公开Geo基线。
- 14：`rebuttal/rebuttal_0723/theory_results/olmo2_qk_phase_adaptation_20260729/metrics.json`。
- 15：`rebuttal/rebuttal_0723/theory_results/llama8b_matched_ruler_mix_20260726.json`。

上面 `research/` 路径相对 `paper-2027/`，`data/`、`docs/`、`rebuttal/` 相对仓库根。已有源文件的SHA256和文件身份见同期source index；核心新图还具有逐项图源receipt。
