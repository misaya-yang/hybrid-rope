# 下一阶段实验：把已有资产变成更强证据

审查日期：2026-09-12。范围：本地文件、当前 TeX、Git 历史及结果 owner；未登录远端、未加载模型、未运行 GPU，也未核验今日机器和 checkpoint 的可用性。本文件是研究优先级建议，不是执行授权或已完成实验记录。**用户最新纠正优先：原约6天/14天仅为历史规划背景，不再据此压缩实验或降低判断标准；时间由用户把控，更好机器、更多资源和连续运行均可协调。** 以下按科学价值及实验依赖排序，不作倒排工期。

最值得投入的新训练是**实际2K窗口、严格固定支持、Geo/Cosh/full-z公平联合训练的三seed比较**；具体主合同和资源扩展顺序见§3。原256窗口补臂保留为历史合同桥接，不再因为便宜而默认成为主实验。但它不是从零开发：本地已经有完整 K−2 自由度的参数化，历史上也确实做过成熟 OLMo 的 z 与 Q/K LoRA 联合学习。真正缺失的是匹配学习期主实验的比较，不是“从未学过 z”。应同时优先追回 151.9M 绝对四格和给 MLA 增加独立语料复评；这比增加一个异条件 benchmark 更能提升稿件可信度。

## 1. 本轮实际核查了什么

证据层级按对象标注，不能因为一个 JSON 在本地，就把它引用的全部 raw、权重和数据视为本地已验证。

| 资产 | 本地核查与可用层级 | 当前科学含义及缺口 |
|---|---|---|
| 151.9M 三 seed exact-range | **本地 owner 已查、raw-hash-receipted**：`paper-2027/research/evidence/EXACT_RANGE_151M_3SEED_RESULT_20260820.{md,json}`。每臂 499,974,144 tokens、7,629 steps、seeds 42/137/256、K32、L256；32 anchors、末 128 token NLL。 | 固定范围三 OOD 长度 Δ=−0.28073/−0.17599/−0.14571；target-matched +0.06032/+0.22720/+0.45959。**三 seed 绝对四格仍未追回**；较早 aggregate 与今天 owner 不同，不能补列。绝对四格缺失不抹掉已识别的配对效果和方向反转。 |
| 432M MLA 三 seed | **本地 raw JSON 已查**：`results/eval_3seeds_full_results.json` 本轮 SHA256 实测为 `1e44d30bb880e4b7427ae55bd7034782989152bd2afca9217495f9b8ece30953`，与 curated owner 嵌入 hash 相同；本地 `results/350m_mla32_evq_report.md`、`scripts/core_text_phases/eval_extended_3seeds.py` 也存在。 | 三 seed 16K PPL 138.81→95.59；但旧 5M-token 评价缓存的上游 revision、token-array hash、文档排除清单未追回。现稿 `appendix/a3_supporting_results.tex` 已正确称 shared-corpus stress test。当前 JSON 的真实可达性比“只有 portable summary”的印象更强，**数据独立性仍未证明**。 |
| 50.9M M4 多形状 | **本地 owner/实现已查**：`rebuttal/rebuttal_0723/theory_results/m4_exact_range_factorial_evidence_20260726.json`，`scripts/core_text_phases/phase16_exact_range_factorial_m4.py`；当前主文和附录已采用。 | 12 配置、三 seeds、128 步的多形状证据已经回答“只比较 Geo/Cosh 吗”。WikiText2 train/validation 分离但重复/裁剪；约 0.01 NLL 量级。无需重新发明十个解析形状，也不能以此短预算替代 500M-token 主比较。 |
| 完整 fixed-support 可学习 z | **本地实现已查**：`scripts/lib/rope/fixed_support_z.py:41`，63 gap logits/K64、62 有效自由度，softmax/cumsum、精确原生端点、单位 gain；`project_` 默认 logits bound ±2。**本地结果 owner 已查**：DIRECT_Z pilot、COADAPTIVE oracle（见下文）。 | 实现真实存在；完整 scratch 匹配结果未找到。另 `knot_allocation.py` 只是五个内点 knots 的受限子族，不能当 K−2 full-z。 |
| 成熟模型 z 联合学习 | **本地 report + compact receipt 已查**：`attention-aware-retrofit/results/adaptation-coadaptation/COADAPTIVE_ALLOCATION_ORACLE_RESULT_20260825.md` 与 `evidence/COADAPTIVE_ALLOCATION_ORACLE_RESULTS_20260825.json`。训练代码当前路径缺失，但本轮由 `git show main_0726:rebuttal/rebuttal_0723/experiments/olmo2_allocation_oracle_5090/train.py` 实读确认：导入 full fixed-support 模块、联合 Q/K LoRA。 | 有真实联合学习，物理 4K、phase shells、300 步；registered all-shell gate 失败。随后冻结 learned table、匹配 dense-LM Q/K recovery：尾部 NLL 改善、全序列变差，full-200 2Wiki 基本持平。它是**已完成的受限反例与机制资产**，不是 scratch learned-z comparator，也不能说“学 z 已失败”。 |
| 旧 learnedfreq / learned τ | **历史 runner/报告身份已查**：`rebuttal/rebuttal_playbook.md:56`、`:78`；历史 `8616af4:experiments/run_128tok_pe_quality.py`。`scripts/lib/rope/learnable_evq.py` 当前为单参数 τ、midpoint quantile。 | 旧“125M/DAPE”主比较实际约 151.9M、L128、15M tokens、seed42、learnable shared inv_freq；没有本次要求的端点固定身份。τ 学习只有一维，midpoint 表端点会随 τ 改变。两者均不能自动当 full-z/LeRoPE 的严格匹配实验。不要误用 `phase11b_125m_dape.py`：它含 Kerple/MLP 改算子路径，是另一个实验。 |
| C2 压缩与迁移 | **本地 owner/compact GPU receipt 已查**：`attention-aware-retrofit/results/coupling-transfer/LOW_DIM_COUPLING_GPU_RESULT_20260901.md`、CPU owner、`evidence/LOW_DIM_COUPLING_GPU_RECEIPT_20260901.json`。 | 无 Qwen 重拟合，64/128K 为 67.75/57.25 vs 64-point 67.25/54.50；Qwen 原生窗 82→71.25、OLMo PG19 retention 0.870971<0.875。完整 C2 已进入当前稿。是紧凑长端迁移的正结果；post-gate diagnostics 不能改写旧方法门失败。 |
| C42 / C42V24 | **本地两份 350 行 raw 路径、构造和 owner 已查**：`ds_workspace/recon_20260910/work/jsonl/olmo_c42/`、`code/coverage_theory_20260911.py`、`verdicts/HEADLINE_20260911.md`。当前稿也已恢复。 | 相同支持、band、S、增量质心，开发面板差 10.7286pp；NLL 16 文档 report −0.1109。可反驳低阶摘要充分性，不能证明普适形状方向。**step42 的 heldout 反转不是 C42/C42V24 本身的独立负复现**；却明确降低继续依赖开发面板选优的价值。 |
| EOS / selective QK | **本地 compact owners 已查**：`rebuttal/rebuttal_0723/theory_results/evq_query_gap_realized_eos32_20260728/FINAL_METRICS_AND_LINEAGE.json`；`olmo2_qk_phase_adaptation_20260729/metrics.json`。raw generation/adapter 以 hash 记录，本轮未取回或重新生成。 | EOS：4/8/16K Native/Cosh=95/100、18/98、0/60%，n100/长度、单训练 seed、+100 query-gap/+32 EOS、明确长位置暴露。QK QA 是独立适配器，8/16K F1 0.07/0→21.48/8.57。已提升当前稿主文；不应因已被找到就再跑同一实验，也不能拼成同一模型同时获得两种能力。 |
| table×gain | **本地 OLMo 判决已查**：`GAIN_TABLE_2x2_FINAL_20260911.md`。350-row Native/BM 四格真实完整但 Native 地板；180-row Native 两格完整，BM 两格未测。**Qwen 本地代码已查但不是结果**：`scripts/eval/eval_qwen_k32_table_gain_factorial.py`；`ZERO_TRAINING_TWO_DAY_EXPERIMENT_SUMMARY_20260902.md:182` 明确 raw owner 未导入。 | OLMo 跨 350/180 拼接不能算完整非地板交互。Qwen factorial 是 Native/Index×unit/index-gain，并非 BM/MR×两 gain，也不是 full13 +6.09 的直接消融。Pro 手册“数据接近完整”须按此细分，不能统一标“已有”。 |

上述遗漏中，C2、C42、EOS、QK、FullLagP2、125M 五架构已经由本轮改稿恢复，不能再作为“未来新增证据”计功。125M 压缩表为 report-backed 单 seed：MLA32 的 8K 改善只有 6.3%，比 MHA 14.4% 小，不能恢复“压缩越多收益单调越大”的原报告标题结论。最有价值的新发现是：**完整 learnable-z 工具及成熟共适应反例可以复用，而 MLA 原始 evaluator JSON 今天在本地可直接 hash 核验。**

## 2. 四项建议，按科学影响排序

### P1 — 补齐学习期 constrained full-z 强对照

**科学问题。** 在相同 sampled support 和训练机会下，标准 LM 损失联合学习内部 z 与全部权重，能否得到比 uniform 更好的长度行为；解析 Cosh 与直接优化的差距多大？这直接联系“分配参与学习”和“可构造”的主线。应称 fixed-support learnable-z comparator，不能仅因参数化相似就标作完整 LeRoPE 复现。

**已有证据/反例。** 三 seed 151.9M fixed-range 因果结果、M4 多形状是正锚点；成熟 oracle 表明短 phase proxy 下联合优化可把收益主要分给 Q/K、尾部 NLL 不必变成任务收益。旧 learnedfreq 未固定支持，旧 τ 死区也不是 full-z 失败证据。

**原256窗口桥接设计（已降为可复用的补充，不是当前首选主合同）。** 可沿 151.9M、L256、base256、K32、499,974,144 tokens、seeds42/137/256 的科学合同；固定三臂：uniform/FMRoPE、anchored Cosh τ4、full-z initialized at uniform。前两臂若 checkpoint、数据、初始化、优化合同可重建且逐 seed 对齐，就复用已完成结果；否则在一个恢复好的同合同 trainer 上补缺失配对，不能让 learned 臂使用不同训练机会。full-z 与权重共同训练，只用短窗 LM loss，冻结 support、slot、gain，不加入目标长窗 loss。预先确定 allocation LR multiplier、weight decay 和是否投影，不能看 OOD 后调整。当前模块 ±2 clipping 是成熟 pilot 的旧选择，**不是本问题的必然约束**；移植时应明确保留其受限意义或采用只保证正 gaps 的无额外投影版本，不能隐形继承后宣称覆盖全部 z。K−1 logits 有一个 gauge、K−2 有效自由度。

运行前最小必要检查是：uniform 初始化 logits/NLL parity、端点位级相等、排序、allocation 梯度非零有限、optimizer 确实更新它，训练中频率不能复用脱离计算图的静态 sin/cos cache。此处功能检查通过后直接进入有判断力的完整训练；短 smoke 不承担阴性结论。共享 z across heads/layers，避免额外架构因素。

**baseline / 单位 / 端点。** 训练 seed 是主独立单位；同 seed/anchor 配对，报告 1×/2×/4×/8× 全部末128 NLL 和 whole-sequence NLL，预先固定 OOD 长度等权均值作一个主汇总，不选择最佳长度。原固定-range 端点是 primary；target-matched 作为预先保留的 operating-range 交互附表，不能混为一组。图形/方向 cosine 仅描述，不是任务或最优性证据；小 seed 数报告 effect/方向与宽 CI，不以 anchors 冒充 seeds。

**复用路径。** `scripts/lib/rope/fixed_support_z.py`；历史 oracle trainer 可从 `main_0726` 取回阅读；`scripts/core_text_phases/phase16_exact_range_factorial_m4.py` 提供 anchored shape 构造，但其 WikiText/M4 训练不可冒充 151.9M 合同。151.9M 当前 owner 保存科学 hashes；原完整 trainer/checkpoint 地址需 P2 回收。`scripts/analysis/compare_learned_allocation_directions.py` 可作次级诊断，其 pinned-walk null 警告防止把平滑位移当理论验证。

**必须补的真实资源信息。** 原训练脚本版本、六个 baseline checkpoints/初始状态是否仍可达；token caches 与 validation anchors；可用 GPU 型号/显存、torch/CUDA/attention kernel；可学习表移植后真实 tokens/s、峰值显存、评估吞吐。本轮未核验这些。历史 owner 确实记录 4 个新 baseline 臂在 RTX5090 合计 10,941.55 training seconds，但不能直接当新 autograd/编译路径的工期承诺。

**上述256桥接设计的预算。** 可复用时只新增 3×约500M tokens 的 learned 臂；无法复用会增到三臂×三 seeds。它的便宜不构成替代§3实际窗口主实验的理由。先以一个 seed 做完整合同的工程和方向读数，再完成预先锁定的三 seeds；不按第一个 seed 的胜负反复选超参。

**成功/失败如何改变论文。** learned > Cosh：优化仍有空间，Cosh 是解析低成本 baseline；相近且区间足够窄：支持简单构造具竞争力；learned > uniform 但 < Cosh：显示短窗训练目标和长窗构造偏好可不同；learned≈uniform：仅限这套训练合同，检查训练曲线/梯度后如无实现问题便保留负结果。不要把“点估计相近”写成等价，也不要把真实负结果无期限解释成未收敛。

**停止/升级条件。** identity/梯度故障修复，不作科学结论；完整预算结束即锁表，不追加 OOD-guided sweep。正式完成之前不新增结果承诺。三seed方向、训练状态和代价清楚后锁定这一合同的结论；后续扩展按§3的可判别问题安排，不以日期或第一轮胜负决定是否继续。

### P2 — 追回核心源记录，并在真正独立语料上复评 MLA

**科学问题。** 151.9M 的运行范围反转究竟对应什么绝对损失地形；MLA 大效应能否在未参与训练/选择的文档上复现？这是 provenance 和泛化效度修复，不是把现有正结果默认判无效。

**已有证据/反例。** 151.9M 只有可靠配对差；两个差不能解出四个绝对值。50M/151M weight×table crossing 也不是四格范围实验，禁止补位。MLA raw JSON 今天可核对；但旧缓存来自 FineWeb-Edu sample-10BT train split 的 shuffled stream，seed99999/buffer10000 不构成文档排除证明。每长8起点且可重叠，不能当独立文档。

**最小设计分两步，先做便宜且可复用的一步。**

1. 回收 151M 三 seed 原 evaluation JSON、final checkpoints、32 anchors 与 trainer/scientific contract；与 owner 中三个 hash `801792f0…`、`6a5ab42b…`、`23a0dd06…` 对照。输出同一 seed×length 的 uniform/Cosh×fixed/target-matched 四格绝对 NLL、原差及 range-change 差。若只有 checkpoints，则只重评既有 32 anchors，不重训。回收 MLA 原 token cache、构建日志和训练输入 manifest；对源 revision/文档集合可核验才升级独立性。
2. 无论旧缓存能否证明独立，若六个 MLA final checkpoints 仍在，优先补一次**冻结 checkpoint、锁定新评估集**的对照：GEO/EVQ×三 seeds，8K/16K/32K 全序列 NLL，16K 作为 primary。新数据应从能证明未参与训练的 pinned shard/文档集合构建；不能仅“换 seed”，也不能拿当前版本的 train split 默认为独立。保留 doc IDs、token-array hash、tokenizer revision、截取 offsets、确切 scored targets，统一排除过短文档规则。baseline/EVQ 同文档；自然可达的独立长文档不足时预先使用同样的 document-disjoint packs，统计单位明确为 pack，不冒称单篇文档。

**baseline / 单位 / 端点。** MLA 的权重是六个已训练 final checkpoints；训练 seed 的条件配对与文档/pack 配对分别报告。每 seed 先聚合 NLL，再 exp；均值 PPL 与 exp(mean NLL) 分列。文档 bootstrap 只覆盖评价集不确定性，三训练 seed 的区间另列；至少保留 8/16/32K 全曲线、原生窗代价。先锁 32 个独立 doc/pack 的面板；必要时一次性扩到预先定下的 64 或 128，增样以区间能否区分零和已声明的最小有用效应为依据，不以“终于显著”为终点。PPL 复评不宣称 retrieval 改善。

**复用路径。** 上表两项核心 owner；`scripts/core_text_phases/eval_extended_3seeds.py`、`run_gqa_evq_experiment.py`、`run_350m_mla32_500m.sh`；当前 `appendix/a3_supporting_results.tex:21` 和 `a5_identification.tex` 已写评价合同。EOS/QK 两个 compact owners 的 raw outputs 和 adapters 可与本次源记录回收一起按既有 hash 拉回，属于附带归档核查；不增开训练项目，不用两个不同 adapter 拼一个成功故事。

**资源信息。** 151M 六个 checkpoint、MLA 六个 final checkpoint 的今日地址/尺寸/hash；旧训练 token cache 与文档清单能否取得；新独立长文本是否具备合法可达的固定版本；模型载入/kernel/32K NLL 实测吞吐。历史 `/root/autodl-tmp` 路径只是回收线索，不是当前可达证明。

**预算。** 源记录回收为低预算 CPU/I/O；MLA 冻结复评为低至中等 GPU 推理预算，具体由真实长度吞吐决定。缺六个权重时不默认重训 432M×6；先记录缺失以及现有结论的限定。

**论文变化。** 四格齐备可明确“改善/恶化哪一臂”及绝对最优配置；若未追回，继续只讲可识别的 paired interaction，删除绝对排序主张。MLA 独立复评保持大效应，会显著提高学习期 flagship 的可信度；若缩小/反转，保留 shared-corpus 原结果，主文改成分布依赖而非独立泛化定论。

**停止/升级。** 从 owner 指针、Git 历史、已知归档位置定向回收；一轮穷尽仍无原件就标 unresolved，不漫无目的全盘扫描。hash 不匹配先查 semantic identity/来源，不把 whole-container drift 自动当科学失效。新 holdout 开封后不改表/τ；观察到实现故障可修复，观察到真实反转必须报告。只有本项揭示配方混杂且现有稿无法用限定解决时，才单列新训练成本交由主代理评估。

### P3 — 固定 C2 规则的独立迁移确认，优先于 C42 再选优

**科学问题。** 已有低维规则是一个可迁移的长程分配描述，还是被选定任务和 checkpoint 共同决定的近似？它能否在独立任务集保留长端效益，并量化原生窗代价？这比继续造新的形状更贴近“可构造并改善上下文利用”。

**已有证据/反例。** C2 两参数从 OLMo 到 Qwen 无重拟合的正结果真实，长端 VT 有贡献；原生窗损失和注册门失败也真实。C42/C42V24 证明在开发面板上低阶摘要不充分，但所有后续开发集冠军不自动享有 holdout 地位；step42、b3_lo14 的反转已提醒勿再筛表。现稿已如实使用这些资产。

**最小设计。** 冻结 C2 参数、64-point teacher、gain c=.074、所有源 hashes；先在已有 Qwen2.5-1.5B 上取新的独立 RULER prompts/任务 strata 及一组真实长输入 QA，使用 Native、C2、原 64-point transport 三臂。覆盖 32K 原生窗及64/128K；自然 QA 按真实 token 长度和 evidence 可用性定义端点，不能将截短窗口的名义预算当真实远距任务。不在 Qwen 重拟合、不增加参数、不偷换 routing。若这一确认通过且资源仍足，才预先按既有确定 transport rule 安装到一个新的 checkpoint；若新 K/参考长度下没有已经确定的安装规则，此时跨模型不是“零重拟合确认”，应暂不升级。

**baseline / 单位 / 端点。** 合成检索按 task-equal official score，整串 exact/整行正确分别作为明确 secondary；自然 QA 采用当前完整响应 F1，保存 token IDs、EOS、cap-hit。prompt 在模型之间配对，按 task/length 分层 bootstrap，使用任务均权而非谁样本多谁权重大；单 checkpoint 的 prompt CI 不证明跨模型分布。primary 是锁定的长端 C2-vs-64-point 差与 C2-vs-Native 效应，原生窗代价必须同表报告；“保留效果”的非劣界值需开封前说明，不借用旧0.875当普适科学门。

**复用路径。** C2 CPU/GPU owners 与 compact receipt；`scripts/analysis/compile_low_dim_coupling_law.py`；当前稿的 `figs/recovered_asset_inputs.json` 绑定 C2 原始 owner。EOS/QK 的旧 raw 回收用于验证保留的学习期生成资产，不将它们作为本次 C2 的任务标签。

**资源信息/预算。** 需要冻结三臂表、Qwen权重及新面板缓存的真实地址，128K KV cache 显存/attention kernel/解码吞吐；当前没有这些现场数据。预算为中等至高的长上下文推理，按 prompts×真实长度×生成 cap 计，不编小时数；优先两表长端 paired confirmation，Native 原生窗保留，不为地板 Native 128K 做无信息的重复测量。

**论文变化。** 独立长任务近似保留且无重拟合：从单次 diagnostic 扩展为可信低维迁移结构；如果只有 RULER 保留而自然 QA 无增益，明确任务边界；若新任务反转，C2 降为已有 checkpoint/task 的压缩描述。原生窗仍下降则仍不称通用 one-table deployment law。

**停止/升级。** 一次锁定确认失败便保留负结果，不扫 c/band/第三参数。C42 的独立两表复评只在作者明确要把“高阶形状影响”升为核心主张时替换本项；它不能自动追加成第五项目。若做，应锁原 C42 对、原实验确切 cos/sin gain 与表位移振幅、同S/质心，先新任务/新提示确认，不重新选 winner；C42 的另一候选 step42 已失败不能冒充其结果。

### P4 — 用同一面板识别 table×gain，补足部署归因

**科学问题。** BM 相对强 deployment baseline 的增益，在另一个预先指定 gain 下是否保留；gain 是否与表发生交互？这是两因子的因果问题，不是再做 gain tuning。

**已有证据/反例。** 350-row BM gain effect +38.19pp 真实，但 native 两格全零；180-row native 两格不能与350-row BM拼接。Qwen Native/Index 小面板 factorial 代码真实，raw owner 未回收；它不能分解不同表、不同gain的 full13 +6.09，更不能替代 MR/BM factorial。当前稿把 +6.09 写联合配置收益是正确的。

**最小设计。** 先尝试追回已有 Qwen factorial raw/receipt `5d6f2f2e…`，确认它能回答的局部问题。新增实测优先一个 OLMo checkpoint、一个 untouched 且有1×和4× strata 的面板：BM(4)/MR(4)×g=1 与 g=1+0.1 ln4，共四格；锁定 support/band/operator/decoder/precision/rows。Native×两gain可复用完全同 identity 的行，缺失才补，作为原生窗参照；official YaRN同面板是强实用参照，有效旧行可复用，不必默认扩完整4×2。g=1.10258578 与1.13862944 的细粒度两值比较回答另一问题，不与unit-gain主矩阵混用。若四格在4×全地板且无区分力，报告这一现象并依预先规则使用8K strata，不临时挑最高得分长度。

**baseline / 单位 / 端点。** 主交互逐 prompt 计算 `I=(score_BM,gY−score_MR,gY)−(score_BM,1−score_MR,1)`，再按任务/长度既定权重聚合；同面板配对 bootstrap CI。每长度四格绝对值、BM−MR 各gain差、gain效应分别展示。official task score、完整字符串/整行终点分列；whole-response QA 与 RULER 不拼一个平均数。长度混合不能掩盖native的1×/4×不同地板状态。

**复用路径。** `ds_workspace/recon_20260910/verdicts/GAIN_TABLE_2x2_FINAL_20260911.md` §五为缺口权威；`experiments/llama3_60dir_20260911/phase1.py`/既有 queue 是实现线索，非今日任务状态；`scripts/eval/eval_qwen_k32_table_gain_factorial.py` 有逐样本交互统计结构，但其对象是 Native/Index；四方法 controls 在 `docs/research/ROPE_OLMO_BM_RESULT_20260908.json`。

**资源信息/预算。** 需要确认可复用行的 prompt/token hash、checkpoint/生成参数、gain实际落点（cos/sin gain使QK logits乘g²）；runner能否同时安装外部table与gain；缺臂数量与真实生成吞吐。预算低至中等推理；先完整2×2，只有主文需要比较所有 deployment 构造时才扩4×2。

**论文变化。** 显著交互则将“形状收益”明确写成条件效应；两gain均稳定胜且交互小，增加构造稳健性；共同地板或宽CI则只保留原 matched-gain结论，不宣称gain无作用。负结果不动学习期 pure-z核心。

**停止/升级。** 原 raw 未追回不能借 session 摘要归因；同一固定面板补齐后一次判读，不扫最优gain、不把不同panel边际拼成交互。P1/P2尚未完成时，本项不应吞掉训练与核心复评预算。

## 3. 资源充足时的推荐主合同（取代原6/14天倒排）

**应把主要新增证据推进到实际2K/4K窗口。首选主合同是151.9M、K32、实际2K连续输入、Geo/anchored-Cosh/full-z、三配对训练seeds；增加第二个预先指定support层，而不是先扩大模型和benchmark。** 原256实验已识别出变量，但只在那里补learned臂，仍留下“短窗特例、分配训练与实际长文使用未连接”的疑问。新2K实验直接在同一模型上连接严格内部干预、正常语言建模学习与4/8/16K长度行为。它是新匹配实验，旧256基线数字不能直接作为它的控制臂。

建议主合同如下。数值是待实施的设计选择，不是已验证最优配方；既有完整训练配方和损失曲线恢复后，在读取新OOD结果之前锁定。

| 项 | 推荐合同及其识别价值 |
|---|---|
| 模型、配对 | 保持151.9M架构、K32，seeds42/137/256。每个seed×support内三臂同初始权重、同连续文档流、同优化器/全局token batch/训练目标与预算；只改变固定解析表或是否联合学习z。共享跨层/头的30个有效z自由度，单位gain，固定slot和RoPE算子。 |
| 主窗口 | **实际L=2048**，没有phase-gap或虚拟长位置注入；验证集文档与训练排除。训练目标为全序列next-token LM，不把长端loss混入主合同。实际4K是后续窗口因素确认，而不是通过更多synthetic offsets冒充。 |
| 支持 | 两个预先命名的block：**常用宽支持B=500,000**和**FMRoPE训练尺度支持B=L=2048**。各block都严格锚定同一最高/最低频率与span。前者与实际长窗配方及慢频位置基相接，后者延续原base=L因果识别。两个值均有项目既有路线依据，不做base sweep、不按结果挑一列。两个block分别报告，可直接回答收益是否只依赖一种支持。 |
| 三表 | Geo；端点锚定的Cosh，预先采用现有经验参考τ=64/√2048=√2；full-z从Geo初始化、与全部权重联合学习。τ是操作默认，不称最优或理论必然。同一τ跨两个support，使support交互不混入额外调参。full-z LR/正gap数值保护必须预先锁定，不能继承成熟oracle的±2投影却不披露。 |
| 完整预算与训练成熟度 | 建议**每臂1B tokens的完整预定训练轨迹**，保存500M和1B两个共同检查点，两个点均做匹配评价。理由是`results/PHASE19_TAU1_vs_GEO_REPORT.md:13`记录过1B/4K条件下GEO胜过EVQ；该历史线还存在架构/τ口径问题，不能将跨旧实验差异直接归因于token预算。因此应在本次同合同轨迹中检验“早期优势是否存活”，不能把500M当充分收敛证书。1B也不是收敛保证；读取主结果前依据训练窗独立validation曲线与原配方确定是否采用此完整预算。所有臂的最终预算/调度相同，不能只延长落后臂。固定1B schedule的500M中途点**不是**旧500M终止schedule的严格复现。 |
| 端点 | 各support内2/4/8/16K whole-sequence NLL及末128-token NLL完整向量；预先固定OOD等权NLL主汇总、原生窗代价，按三训练seed报告。长度对应真实连续输入；target-matched运行范围另作冻结推理交互，不能代替fixed-support主终点。新独立文档/pack及其hash保留，不能重用已用于选表的窗口。 |
| 预算单位 | 两support×三方法×三seed=**18条完整训练轨迹**；若采用1B合同则18B总训练tokens，500M中间读数不需另训。这里给token量，不按旧256吞吐承诺小时；需要实测2K autograd、kernel、显存、并行机器数。 |

这比仅在256加一个learned臂更有机会改变论文判断：即使Cosh不胜，仍能确认full-z在正常窗口是否有价值、学习目标是否选择不同分配、支持是否决定效应，以及短预算排序是否持续。三个seed完整重复比“多跑十种形状、各一个seed”更有解释力。

**4K升级应回答窗口问题，不能同时偷偷换掉support、τ和目标。** 推荐在B=500,000 block复做实际L=4096的三表×三seed，保持τ=√2及主训练token/优化合同，评估4/8/16/32K；这是9条配对训练轨迹，直接检验同一构造随训练窗口改变的表现。若使用τ=64/√4096=1，则应明确它检验“随窗口变化的默认构造规则”，不再称纯window因素。两种问题择一预先声明，不需要默认把两种τ都铺满。训练tokens固定意味着覆盖不同数量的长序列；global tokens/update和总steps可保持，microbatch依硬件调整并记录。若为了4K改变了模型/attention实现，则先排除这项实现混杂。

资源扩展的优先顺序是：

1. **真实窗口和完整配对重复。** 先完成上述2K主合同及三seed；不要用更大模型的单seed替代。P2的MLA独立holdout与原件回收可独立推进。
2. **support及训练成熟度。** 两support block与500M/1B同轨迹检查点优先于更多方法；它们能够直接区分范围依赖和训练预算依赖，而不是只是增加覆盖数。
3. **4K窗口确认。** 对同一support/构造做前述配对扩展，判断2K发现是否随着真实训练窗口延伸。仅在2K阴性就取消4K会引入选择；是否执行应由窗口问题是否仍未解决及授权资源决定。
4. **训练目标因素。** 如果full-z在短窗LM下改善窗内却伤害OOD，或Cosh胜出但learned不胜，才追加明示的目标比较：相同2K物理输入和总监督token预算，预定任务/长位置暴露目标对比标准LM；所有三表都获得同等目标和训练机会。若使用长相位暴露，报告max position与暴露分布，不称无长位置暴露外推。若任务能力成为主张，应保存自由生成完整输出与任务适用的EOS指标，不能以NLL代替。不要重跑已失败的两文档direct-z或旧phase-shell oracle来充当这个新比较。

更多资源仍不增加价值的项目：已暴露panel上的形状/τ/gain搜优；只重复显著几何proxy而不触及模型任务；不匹配训练目标的learned与analytic比较；把不同checkpoint的最佳分数拼表；无具体异质性问题的benchmark扩展。EOS/QK既有结果应先追回raw与适配器身份，是否补训练seed由其是否成为核心生成结论决定，不因机器多就把所有历史实验补齐。

**停止依据是结论已可判别，或明确的实现/数据阻塞；不是“第几天到了”。** 对预先锁定的主合同，正负结果都完成三seed，不在读OOD后换support、τ、预算。若宽CI仍容纳有用增益与实质伤害，更多独立seed/文档比新模型、新benchmark更有价值；若结果已经稳定地限定某个构造，则结束该构造的搜优，把算力投到仍未回答的因素。

暂不新开：更多任意曲线、60规则搜索、750M/8B/Video各补seed的宽矩阵、仅提高静态rank/碰撞分数的优化、旧成熟phase oracle加steps/加LR、同已暴露开发面板上的C42变种选优。它们不是永远无价值；只是目前相对上述四项，对中心论证的边际贡献更低。

执行时每项记录主终点、绝对臂值、逐seed/逐prompt差、原生窗代价、数据和checkpoint身份。回收成功、CPU构造正确、runtime smoke通过、正式模型端点完成分别记账。任何远端历史状态，包括“9/12凌晨阻塞解除”，在本审查中均未获今日重验证。
