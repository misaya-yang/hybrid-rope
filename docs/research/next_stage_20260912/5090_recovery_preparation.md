# 5090 无卡准备：能力恢复优先

更新：2026-09-12。此文件记录本 session 的最新用户范围，覆盖下载计划书原先按机器分配 E4/E6 的建议；不更改另一 session 的训练合同。

## 接管时先读：没有全部就绪

截至本次轻量源码复核，**本 5090 session 完成的正式实验为 0 组，已启动 GPU 作业为 0；端到端验证完成的实验组为 0。** 已有的是三组首批实验的代码和部分输入，其中 E2 最接近可运行。除此以外，原计划还包含 GPU 复评/机制计算和 CPU 理论计算，见下方完整计算清单；三组首批不是全部任务。语法检查通过不代表模型训练、评分分析或完整数据管线通过。

新 Sol session 从 [5090_SOL_HANDOFF_20260912.md](5090_SOL_HANDOFF_20260912.md) 接管：包含当前目录、命令、缺口、优先级、恢复和工时重估规则。本次仅更新文档、检查源码与做常数算术；没有新跑模型、测试、分词、下载或服务器负载。

## 一共多少项

计数分三层，避免把配置数当独立实验数：

| 范围 | 科学实验组 | 准备/审计 | 当前含义 |
|---|---:|---:|---|
| 原计划推荐 E0–E8 | 8 组（E1–E8） | E0 一项 | 共 9 个编号工作包；不是本机全承接 |
| 加用户新增 OLMo LoRA | 9 组 | E0 一项 | 共 10 个工作包；不包含默认排除的 E9 和 C2 128K |
| 当前 5090 明确首批 | **3 组** | 共享轻量准备及显卡资格 | OLMo LoRA、E2、E3；E5 等待上游结果，Llama 后置 |

原计划 E0–E9 全编号为 10 项，其中 E9 是额外实际 4K 训练设计，不能因编号存在就计入本轮承诺。

| 当前组 | 正式配置与工作量 | 评价数量 / 状态 |
|---|---|---|
| OLMo LoRA 恢复（Top1） | Native / 旧 EVQ（代码名 Cosh）/ OfficialYaRN，**3 条训练臂**；各首段 33,554,432 CPT 预测 tokens（2048 updates），三臂合计 100,663,296（6144 updates） | 默认开发面板在三表 0-step、三臂 8Mi 中点、三臂 32Mi 端点，共 **9 个评价快照**；每快照含生成与四长度 LM。真实语料尚未成包，具体生成总数待冻结；8Mi 是同一轨迹的观察点，不是新增训练臂 |
| E2 自然 QA | **7 个冻结推理配置**，每配置同 778 个问题 | **5,446 次生成**；631 长输入为五任务等权主统计，147 短输入成本单列 |
| E3 C42 独立确认 | **2 个冻结推理配置**；每配置 700 条 16K + 280 条 4K | **1,960 次生成**；980 是计划样本数，目前未生成完整新面板 |

因此首批为 **3 条训练臂 + 9 个冻结推理配置**，并非 12 项独立科学研究。E2/E3 合计 7,406 次生成；不能把它当成包含 LoRA 评价的总量。LoRA 延续到 64Mi/128Mi 是尚待决定的同轨迹扩展，首段工期不包含这些延续，也没有自动训练队列替用户承诺它们。

## GPU / CPU 计算完整清单（按用户提醒补齐）

本次重新核对了以下仓库说明：

- [当前阶段 index](index.md) 的 **S2 机制主线、S3 既有核心结果复评**，以及 [experiments.md](experiments.md)、[theory.md](theory.md)。这些说明明确要求计算，而非只训练或跑生成 benchmark。
- [双路线实验说明](../../../experiments/twotrack_20260911/EXPERIMENT_PLAN.md) 的 P1/P2/P3：固定16K的 d/b/n 因子干预、有限频谱线性读出诊断和 weights×table crossing。它是2026-09-11的研究设计，须与最新 E5/E7 对照，不直接复活旧排队状态。
- [RUN_READY_20260911](../../../ds_workspace/recon_20260910/RUN_READY_20260911.md)：已有表无关 evaluator/YaRN 构造等可以复用；历史“已传服务器/中断待跑”的条目须重新核验。

没有定位到名为“实验说明”的唯一文件，以上是本次找到且内容直接对应用户所指 GPU 计算的材料。另查到 PC2 十项稀疏选择器计划和 KLD 计划，它们是独立研究线，不能仅因含 GPU 计算就自动计入当前 RoPE allocation 队列。

| 编号 / 计算类型 | 具体计算及完整工作量 | 可复用实现 / 当前缺口 | 原计划资源参考 |
|---|---|---|---:|
| **E0 补档后必要重评** | 旧151M的 Geo/Cosh × fixed/target-matched 绝对四格；有原checkpoint和32 anchors时重评，无需重新训练 | 当前证据 owner 保留配对差；原 absolute evaluator输出/原checkpoint/anchors仍需定向恢复。不能用新S1结果或差值反解补格 | 未单独定 GPU 小时 |
| **E1：MLA 冻结 NLL** | 六个 final checkpoints × 原表/论文 blend 两策略 × 8/16/32K ×64 packs，共 **2,304 个 pack×配置评价**，不是生成任务 | [eval_extended_3seeds.py](../../../scripts/core_text_phases/eval_extended_3seeds.py) 是历史入口；还需六个权重的可达性、独立数据与新合同接入/容量验证。复用原模型，不重训432M | 2.85 GPU·h |
| **E4 的新统一评价** | 18条训练臂 ×500M/1B两个checkpoint ×2/4/8/16K ×128 packs，共 **18,432 个 pack×配置评价** | [S1 evaluate.py](../../../experiments/fixed_support_joint_151m_20260912/evaluate.py) 目前写死旧512文档面板，不能当新128-pack evaluator已完成。4080已有评价与这项新合同需分开登记；新session与原owner明确评价归属后只做一次 | 3.90 GPU·h，已包含在原E4总价内 |
| **E5a：原始训练表3×3 crossing** | 每support×seed固定权重，三个权重 ×三张训练完成表；共54矩阵格。复用18个对角格，新增36个非对角格 ×4长度×128 packs = **18,432 个 pack×配置评价** | 依赖E4完整1B权重、learned原始表和新统一面板；既有crossing只作实现/身份线索，**当前未写好本E5新runner** | 3.90 GPU·h |
| **E5b：同权重 range 干预** | 18个1B checkpoint ×4/8/16K ×128 packs = **6,912 个新增 pack×配置评价**。`B_run=B×L/L0`，保持该权重的z和最快频率 | 复用相同权重/输入，与保持训练range逐行配对；不把不同GPU上的训练差异当运行range效应。当前缺完整新runner | 1.17 GPU·h |
| **E5c：模型实际系数与置换控制** | 提取真实 pre-RoPE Q/K、每pair有符号cos/sin系数、有限相位影响和来源/干扰贡献；频率与Q/K槽同步置换应恢复no-op | 有理论与历史工具；须连接当前模型、Q/K归一化及GQA映射，锁采样协议并验收。单纯算Gram不算完成此GPU部分 | 1.00 GPU·h（采样/控制预留） |
| **E7：来源利用因果矩阵** | 先48个4K资格输入；通过后96基础案例×3距离×2来源值，层组四格与gold/sham遮挡，另192 Native控制；最大 **4,848 次生成**，并记录答案NLL/来源干预 | 依赖OLMo基础读出能力、真实16K面板、attention mask/sham无操作一致性；**当前没有验收过该插桩实现**。它不是LoRA训练后分数更高就自动完成 | 21.35 GPU·h，另原计划估1.5–3人日工程 |
| **E5c / S2 的纯几何部分（CPU）** | 实际表的 raw Gram、block-whitened overlap、去掉span{1,t}后的剩余谱；uniform/causal-triangular/预定来源距离测度 | [verify_explicit_geometry.py](../../../paper-2027/figs/verify_explicit_geometry.py)、[verify_profile_diagnostics.py](../../../paper-2027/figs/verify_profile_diagnostics.py)可作历史核查入口，需接当前表与测度；这些结果不等于PPL/任务能力 | 原计划 CPU 2–4小时，非当前0.5核ETA |
| **S2 理论核查（CPU/数学）** | 锚定密度身份、有限K目标及系数预算界；只有产生独立模型预测才考虑新实验 | [theory.md](theory.md)已有部分代数核查，先复用，不为了“计算队列”重跑同一恒等式或增加无目的曲线 | 未给当前机器定量时长 |

E5三项合计 **6.07 GPU·h**，不能再额外叠加一个“E5整体6.07”重复计价。表中小时均来自原计划的 Pro/指定假设，移至5090前须测容量和速度；**原计划写Pro不等于5090必然不能承担，也不等于已验证能承担。**

从执行管理看，可将本机/可承接后续分为 **7个工作包**：LoRA、E1、E2、E3、E4新统一评价、E5、E7，另配CPU理论/统计整理；其中仅3个首批已有本轮新代码，4个后续GPU包仍有资产/实现/归属依赖。E4评价属于原E4实验的一部分，不因此让“全研究9组科学实验”的计数变成10组。E0旧四格回收后必要重评另列为条件任务，不凭空计作已排作业。

### 双路线说明里另外写到的机制计算

`twotrack/EXPERIMENT_PLAN.md` P1还设计了：OLMo固定16K，远距d（0.2L/0.8L）×局部key-value间隔b（4/64）×干扰记录数n（8/64），16个内容种子×8变体×3表=**384次生成**，并记录完整答案NLL和最危险token margin；随后32个新内容的重采样控制。其用途是分清距离重标定、局部绑定与attention竞争。

这是与E7来源距离/机制解释相关、但合同不同的候选设计。**登记保留，不与E7自动同时加入预算**；要采用它应说明回答哪个E7未覆盖的问题、先核历史是否已完成、冻结对应表及内容。如果已有结果无新增判别力，则复用/结束，不把旧的384次再跑一遍当新进展。P2的线性读出/ridge误差是CPU模型，P3的权重换表与E5部分重叠，也不双算。

## 准备程度和剩余工作

| 组 | 已落实 | 尚未完成；新 session 必须处理 |
|---|---|---|
| LoRA | 模型/词表上轮确认在盘；训练、教师轨迹 KL、数据、恢复、评价和 probe 代码；最终只做静态检查 | 五项原始 sources、处理后数据包、旧 EVQ 精确数组、实际 PEFT/恢复/显存资格；Native lost/gained 与按 source 聚类的配对分析仍需接好 |
| E2 | 原池 778/631 行已组合，上轮 dry-run 通过；七表、逐行恢复、主对比与 gain 分析代码 | 最终代码与服务器 manifest 的身份重新核对；真实 5090 生成路径尚未跑；最终统计须从完整七臂 raw 验证 |
| E3 | 两表精确构造、生成器、去重、运行和逐格统计代码 | 完整 980 行面板；完整旧面板排除 registry 重建；实际运行资格；**主差值与任务分层 paired-bootstrap CI 尚未接入 postprocess** |

本次阅读另确认两处分析接口缺口，不能交接为已验收：

- `e3_postprocess.py` 当前把 lower/whitespace-normalized、命中任一 reference 的结果叫 `complete_string_exact`；这不是 literal 完整多答案合同，须与 normalized/官方部分分分别命名和实现。它目前只给逐格均值，缺主 16K 宏平均 B−A 及配对区间；输出重复行和完整性也需严格验证。
- LoRA `evaluate.py` 的 raw 使用 `eval_id`、`eos_or_eot_terminated`，独立 `data_score.py` 使用 `id`、`eos_terminated`；二者不是直接兼容的下游接口。优先以 evaluator 的已定义 raw 为 owner 接好 paired native lost/gained 与 source-cluster 分析，不直接把文件喂进另一个 scorer。

这些是静态源码发现，尚未执行失败用例；本次不扩大为实现/测试工作。完整可运行状态必须在有资源后核实。

## 预计多久：情景预算，不是实测 ETA

**先为 5090 首批预留约 1–3 天的执行窗口，但这不是上限或完成承诺。** 可计算的训练加 E2/E3 参考工作量约 20–45 小时，另有下述未计量项；数据获取/源码修复受阻、实际吞吐偏低时会更长。不能承诺“一晚全部完成”。

OLMo 新配方没有当前机器实测吞吐。定义 `q_eff` 为**完成 CPT+SFT+replay CE+Native 教师生成/KL 整个更新后，每秒推进的 CPT 预测 tokens**，不是所有监督 tokens/s，更不是纯 CPT kernel tokens/s。默认每更新推进 16,384 CPT tokens。用三个纯规划情景，并加 20% 保存/启动执行余量：

`H_train = 1.20 × 3 × 33,554,432 / (q_eff × 3600)`

| 假设 q_eff | 对应完整 update 时间 | 三臂到共同 32Mi 的训练 | 若三臂都最终延续到 128Mi：累计训练 |
|---:|---:|---:|---:|
| 4,000 CPT tokens/s | 4.096 s | 8.39 GPU·h | 33.55 GPU·h |
| 2,000 CPT tokens/s | 8.192 s | 16.78 GPU·h | 67.11 GPU·h |
| 1,000 CPT tokens/s | 16.384 s | 33.55 GPU·h | 134.22 GPU·h |

这些是假设，**不是三个已测配置，也没有证据保证速度一定落在此范围**。旧 5090 QKVO LoRA 的吞吐不能直接套到新增 FFN、长 SFT 和教师轨迹 KL 的配方。两步显存 probe 使用最长样本，且教师有首次生成/缓存差异，只用于执行资格；ETA 要用正式阶段的完整更新计时，并随实际缓存占比修正。

| 其余部分 | 参考耗时 | 来源与限制 |
|---|---:|---|
| E2 七臂 | 6.59 GPU·h | 原计划 §5 的 **Pro 6000 假设**；作为临时规划参照，不是 5090 实测 |
| E3 两臂 | 5.08 GPU·h | 原计划 §6 的 **Pro 6000 假设**；实际输入数/长度和输出 token 数需从新面板统计 |
| E2+E3 | 11.67 GPU·h | 两项参考相加；不可由“5090更快”自动打折 |
| LoRA 三臂训练 + E2/E3 | 20.06 / 28.45 / 45.22 GPU·h | 分别对应上表 4K/2K/1K 的 q_eff；尚未包含下行项目 |
| 数据下载/处理、源码缺口修复、显卡资格、LoRA 9 次评价快照、后续 test 确认 | **未计量，额外增加** | 缺冻结语料规模和当前运行速度；不能静默按 0 小时计算 |

在单张 5090 串行有效执行时 GPU·h 大致对应占卡墙钟；断连、人工修复和下载等待另计，不把它当成已知准确日历截止时间。LoRA 每个评价快照应按冻结的输入 `I_L` 与实际生成上限/输出 `O_L` 估算：`Σ_L[I_L/prefill_L + O_L/decode_L]/3600 + 加载开销`，再加该评价独立余量；NLL 的四长度输入也必须计入。

原计划的 **E0–E5 89.9 GPU·h（含项目余量107.9）/ E0–E8 165.0 GPU·h（含余量197.9）** 是不同机器全部工作量的历史规划，**不含新增 LoRA，也未扣掉 4080 已完成量**。它们不是本 session 剩余量，更不能再与上述 5090 小计相加（会重复算 E2/E3）。全研究计划的准确剩余量需要各 owner 提供完成回执后重算。

若上述后续GPU计算均由本机接下且原参考速度适用：E1+E4新评价+E5 = **12.82 GPU·h**；再加E7 = **34.17 GPU·h**。与三组首批20.06–45.22的参考小计合并，得到 **54.23–79.39 GPU·h**。这只是“7个工作包的已列计算分子”情景，仍未含 LoRA 评价/真实数据处理/工程修复/E0追回后的重评，也未含等待4080 checkpoint的日历时间。不能将前面的首批1–3天窗口当成这七包全部完成的承诺。

## 当前范围

1. **Top 1：OLMo 长上下文 LoRA 能力恢复。** 先用现有 OLMo-2-0425-1B-Instruct；OLMo 稳定后再考虑 Llama。当前不下载 Llama 权重、不安排 Llama 训练。用户明确重视 FFN、训练数据、训练成熟度和损失设置，主终点必须同时包含长端 PPL、完整生成和原生能力保留。
2. **E2：自然 QA 强基线补齐及 gain 交互。** 使用原 778 条输入、七配置、631 条长输入的五任务等权 F1；是原池匹配重评。
3. **E3：C42/C42V24 独立确认。** 同一新面板运行两表，保留 4K 成本和 16K 全部结果。
4. CPU/GPU 计算和 150M 以下训练是允许的任务类型；本轮没有为占用算力另设训练任务。E5 需要完整 S1 checkpoints 时复用其产物。

**不重复 S1/E4/E6 的 151.9M 训练。** 另一任务“检查GPU服务器实验代码”（task `01a0914a-10f9-7300-971b-095a3edebf44`）在 westc:53405 管理 B500000 seed42 的恢复、Geo/Cosh/full-z、长窗评价及 seed137 串行链。本 session 只读核对其记录，没有改动它的进程、文件或配置。

本机服务器入口为 westd:40403，当前为无卡、**0.5 CPU 核、2 GiB cgroup 内存**。本机也不承担重计算或大文件存储。最新限制下仅做代码/配置阅读、编辑及轻量静态检查；停止两端模型 CPU 测试、分词、批量数据生成、依赖安装、下载与大文件哈希。这里的计划和此前 CPU 检查都不等于完成显存/吞吐资格。

## 原计划总清单

[用户原始计划全文](ICLR2027_ROPE_EXPERIMENT_PLAN_20260912.md)按原字节保存，避免依赖 Downloads。计划书保持原文；本页记录后续用户调整。下表是责任与准备状态，不把所有行写成已可执行。

| 项目 | 当前归属 / 前置条件 | 本 session 状态 |
|---|---|---|
| E0 | 版本、代码、数据及硬件身份 | PDF/计划版本已绑定；0.5 核只做 metadata inventory；GPU 资格待有资源 |
| E1 | 六个 MLA final checkpoints、独立语料；原计划 Pro 评价线 | checkpoint 可达性尚未核实；不在 5090 伪造重评就绪 |
| E2 | OLMo 冻结七配置自然 QA | 原 778/631 输入准备及校验完成；生成、统计代码待完整运行 |
| E3 | C42/C42V24 新 980 基础案例 | 精确表与生成/去重/评分代码已写；完整新面板未生成，留至有资源阶段 |
| E4 | 4080 session 的 S1 主训练线 | 不接管、不复制训练；需要后续 seed/support 扩展由原 owner 调度 |
| E5 | E4 完整 checkpoints、原始训练表、统一面板 | 依赖清单保留；不以零散 checkpoint 做选择性 crossing |
| E6 | 原配对机器上的 Cosh-init 补臂 | 不在本 session 重复启动；纳入范围跟随原训练 owner |
| E7 | 冻结 OLMo 的真实来源反事实与层组/mask 干预 | 原计划保留；不是当前 LoRA 恢复结果，也未完成插桩验收 |
| E8 | Llama 冻结跨家族确认 | 用户要求先稳定 OLMo，Llama 后置 |
| E9 / C2 128K | 原计划非默认承诺 | 不自行启动 |

新增 OLMo LoRA 是 **计划之外 Top1**，不是用它替换 E2/E3，亦不是将全部 E0–E8 宣称准备完成。

## 版本和历史依据

输入计划书 `ICLR2027_RoPE_实验计划_20260912.md` SHA256 为 `df3b6f000ec7a64eff494e97a48e7cfbfb07371c736cad8208b3b3fa49a67c26`。当前 [main.pdf](../../../paper-2027/main.pdf) SHA256 为 `4e5895a43b2040b60225c6b6e5429b9caead8b0e34570639dd30a70596a875d2`，与其 P1 完全一致；初读仓库 HEAD 为 `588d4efe2a17632ee8c63298eb8bf3e11ad139db`。本轮不改论文结论。

| 历史依据 | 已测事实及对新实验的作用 |
|---|---|
| [A11：Llama 8B](../../../data/curated/llama8b_causal_source_use_s42_20260714.json) | 8K、300-step、Q/K/V/O LoRA；长端 PPL、远源删除敏感性改善，但 8K 成本和同 adapter 下的任务负结果必须保留。概率访问不等于完整输出。 |
| [OLMo Q/K 阶段](../../../rebuttal/rebuttal_0723/theory_results/olmo2_qk_phase_adaptation_20260729/metrics.json) | 从 Q/K/V/O parent 继续答案与终止监督，部分自然 QA 与 RULER 改善；物理输入 ≤4K 而有长 position-ID 曝光，不当作真实 16K 训练复现。 |
| [FFN 执行记录](../../../paper-2027/research/attention-aware-retrofit/results/SINGLE_TABLE_FFN_SERVER_EXECUTION_20260904.md) / [对应审计](../../../paper-2027/research/external-reviews/pro-materials-20260908/SINGLE_TABLE_FFN_REPORT_AUDIT_AND_CODEX_GUIDANCE_20260904.md) | 全线性适配可以改变完整生成，但已有数据没有证明 FFN 的必要性；语义、格式、EOS、lost/gained 分开。source-truth CE 和原模型行为分布保留不是同一合同。 |
| [真实长输入恢复实现](../../../experiments/evq_recovery/index.md) | 可复用 dense CPT、完整长指令、短 replay、source split 和分块输出头；历史准备状态须实地检查，不能继承“目录已 READY”的文字快照。 |

## 新恢复合同

这是新的匹配恢复实验，直接从对应原始 Instruct 基座开始；旧 adapter 及旧 LongAlpaca 数据尚未恢复到足以按字节复播，因此旧论文仅作历史锚。

- Native / Cosh / 官方 YaRN 三臂使用相同模型权重、adapter seed、数据顺序、loss 归一化和训练预算。
- 全层 Q/K/V/O 与 gate/up/down 的 LoRA，r32、alpha32、dropout0；FFN 参与此复合配方，不预写 FFN 因果归因。
- 实际连续 16K CPT；首段 32Mi 预测 tokens，8Mi 保存观察点，沿固定 128Mi scheduler 可继续至 64Mi/128Mi。SFT/replay/KL 的曝光与代价另外计量。中途点是同一轨迹，不是假定收敛。
- 各项分别 mean-normalize：dense CPT CE、完整答案与合法对话终止 CE、短 replay CE、短 Native teacher KL，权重 1/1/0.25/0.25。教师使用同一个冻结基座关闭 adapter 并恢复 Native 表。短指令 KL 应覆盖原模型实际 greedy 答案/终止 prefixes，文本 replay 使用已声明的文本位置；两者都不保证覆盖整个生成分布。此新路径尚未运行验证。
- OLMo 首比较优先固定旧失败链中的 EVQ FP32 表，SHA `917a52426b4ac986545c8ec73b115daae3c6515d6b9047f09d30c972ea1a4607`；精确数组仍需追回。另有新 fixed-support anchored τ=2 入口，但不拿它冒充旧 EVQ 的修复。官方 YaRN 使用 s4，为实用对照。Llama 的旧构造入口暂时保留代码，不进入当前队列。
- 长文本 NLL、自然 QA 完整 F1/答案准确率/终止、原生能力逐实例 lost/gained 分别评价。仅 PPL 改善不能结案，格式改善也不自动解释为取证能力恢复。

有资源后先完成真实数据处理及资产校验，第一模型 GPU 动作是丢弃式真实输入显存/梯度检查：BF16、Flash SDPA、激活重计算、分块输出头；校验 Q/K/V/O 和 FFN 更新、teacher 状态恢复、finite loss、峰值显存与吞吐。训练前分别保存 Native/EVQ/YaRN 的 0-step 基线，以区分直接换表损坏与训练遗忘。无卡期间不宣称 8B 已能在 5090 完成训练，也不默认换成 4bit。

## 资产与清理记录

用户最初允许清理旧实验资产；当时本 session 错将任务缩到 E2/E3，并据此删除了 westd 副本上的旧 Llama/Qwen 权重和实验目录。该操作完成后用户将 Llama LoRA 提升为 Top1。已说明此范围判断失误；4080 原机未被清理，Llama 两套权重已在原机只读确认仍存在。

清理前验证保留了 30 项模型、数据和输入文件身份，数据盘空闲由 28,643,053,568 bytes 增至 87,953,694,720 bytes。原始回执位于服务器 `rope_fast_5090_20260912/receipts/cleanup.json`（数据盘相对路径）；属于运行材料，不随 Git 交付。

Llama-Instruct 的 SSH 回传尝试速度过慢，已停止；几 MiB 的不完整下载位于隔离 `model_incoming`，不能当成模型。用户最新决定是继续准备代码，缺失权重在实验执行时下载。后续保留模型权重，只在明确范围内清理无用输出与缓存。

保留资产：OLMo 1.485B 完整 BF16 权重及 tokenizer、三个原 QA 输入批次、RULER 上游及旧输入排除面板、FineWeb-Edu 冻结 token 数据与源 parquet、GPT-NeoX tokenizer、NLTK/PG19 小型语料。新 LoRA 所需完整 LongAlign/QASPER/replay 来源需由准备器检查；缺源状态不升级为 READY。

## 代码与当前验收

[恢复代码](../../../experiments/olmo_recovery_20260912/index.md)；[E2/E3 代码](../../../experiments/rope_fast_5090_20260912/index.md)。当前未启动 GPU、LoRA 或额外 S1 训练。

资源限制澄清前，E2/E3 七项轻量测试通过；E2 原 778 行及 631 长行在服务器组合完毕、dry-run 校验通过。E3 远端首次调用因 Python 包路径失败，未生成面板；现已修正包导入，未在资源限制下重跑。E3 QA offset 改为 3000/4000，避开原面板 offset1024；来源去重检查代码需随完整生成一起验证。

最新代码含 OLMo 训练/评价/恢复、真实 PEFT 测试入口和 metadata-only `readiness.py`。后续修改目前只做静态检查，不能沿用此前小测试称最终完整路径通过。**总体为 CODE_PREPARED_WITH_PENDING_ASSETS_AND_RUNTIME_CHECKS；不建议把当前状态称为“全部验证无问题”。**

最终轻量检查：20 个源码文件 AST 语法通过；文档检查覆盖 85 个管理文档、1281 个本地链接、28 个资产，零错误。约 200KB 的纯源码包已同步至服务器 `rope_fast_5090_20260912/prepared_code/`（数据盘相对路径），随后同步最终两份小文件修正；没有传模型或大数据。

服务器 metadata-only 回执确认 OLMo 模型和 tokenizer 文件仍在，模型文件大小 2,969,854,224 bytes；已有 torch 2.8.0+cu128、transformers 5.15.1、peft 0.20.0。新 LoRA sources/data 目录尚未形成可用数据包，五项源清单和全部处理后文件仍标 missing。此回执仅检查配置、文件存在/大小和包版本，不加载 torch 或权重。GPU 始终未启动。
