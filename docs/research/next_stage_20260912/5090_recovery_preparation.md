# 5090：实验准备、数据与工时

更新：2026-09-12。最新执行原则：**使用现有 OLMo 小步快跑，全部在 5090 上推进，不等待 4080 主训练完成。** 本页是当前执行说明；[原计划全文](ICLR2027_ROPE_EXPERIMENT_PLAN_20260912.md)保留原文，新 Sol session 读[接管说明](5090_SOL_HANDOFF_20260912.md)。

## 当前入口与门禁（用户最新约定）

当前入口：`ssh -p 24904 root@connect.westd.seetacloud.com`。原40403是准备时的旧实例；实例克隆可以改变SSH端口，后续以用户最新告知为准，不追回旧端口、不据此重做准备。用户保证克隆模型、数据和代码资产准确，**后续不做SHA校验，也不因历史代码/manifest/权重SHA不一致阻塞启动、恢复、评价或统计。** 不反复读取大文件计算哈希，不为重新绑定字节身份重建同一数据包。

保留的检查只有直接影响执行的轻量项目：必要文件存在、模型配置、实际8K/16K输入及完整答案、arm/频率值与预算、恢复位置、解码设置与样本/输出对应、CUDA显存和有限loss。历史receipt的SHA仅是历史记录，无需重新验证；样本分组/去重的短ID不是资产门禁。

2026-09-12已只读连入24904：RTX5090，显存32607MiB，E2 PID2197运行时GPU100%；LoRA数据PID1838和E3准备PID2457同时运行。另一任务“5090实验负责agent-5.6sol”负责实际队列，已得到用户开跑授权；其用户另说明可用约25核/90GB，不再套用旧无卡0.5核限制。已有健康进程继续；本任务不重复启动、不重跑已完成G1。

## 本轮做四条线

| 顺位 | 实验 | 工作量 | 当前情况 |
|---|---|---|---|
| Top1 | **OLMo LoRA 恢复**：同时看长端PPL、完整回答和原生能力保留 | Native / Cosh两臂，各32Mi CPT预测tokens，8Mi中间点；8K/16K实际长输入各占一半CPT tokens，Q/K/V/O+FFN LoRA，CPT+长指令+短replay+原生教师KL | Cosh正在训练；其后只接Native。YaRN训练对照已取消，旧零训练结果保留 |
| E2 | **自然QA强基线＋gain交互** | 七配置×778条＝5,446次生成；631长输入主统计，147短输入成本 | 原778条输入已在服务器；七表/运行/统计代码已写 |
| E3 | **C42/C42V24独立确认** | 两表×(700条16K+280条4K)＝1,960次生成 | 公式、生成器、运行和配对统计代码已写；RULER原始数据已在盘，开卡后生成新案例 |
| G1 | **实际Q/K系数和相位重放的GPU计算** | 现有五任务各前2条长输入，共10条；Native/BM两次观测、4层，20次backbone forward、80条层级记录 | 新增 `qk_diagnostic.py`；复用E2输入和模型，不等新checkpoint。输出实际系数、同gain BM/MrPro相位差及距离/频段统计 |

**四条线不是四条训练。** LoRA当前只有Native/Cosh两条训练臂；E2/E3是冻结推理配置，G1是GPU诊断。已完成的YaRN冻结推理结果不升级为LoRA训练对照。

科学优先级仍是LoRA。执行上，先就绪的E2/G1可以先跑，同时在有卡机器的CPU准备LoRA/E3数据；不让整张卡等待一个旧文件或4080。数据准备不能挤掉健康GPU作业。

## 已追加的三组快实验

不重复4080训练，不新增权重下载；CPU准备已在24904完成，GPU尚未由本任务启动。入口统一为`experiments.rope_fast_5090_20260912.supplement_queue`，默认只列队列，执行session在空闲GPU时加`--job p1|source|layers --execute`选择一项。已有健康工作和LoRA仍优先。

| 队列ID | 科学问题 | 固定工作量 | 准备目录（数据盘相对路径） |
|---|---|---:|---|
| p1 | 固定16K，距离×KV局部间隔×干扰数量分别怎样影响BM/MrPro/YaRN？ | 16种内容×8变体×3表＝384次生成 | `rope_fast_5090_20260912/supplement_p1_factorial_prepared/` |
| source | 只改来源值，完整答案是否随之正确改变，同时保留4K能力？ | 4任务×16seed×4K/16K×双world×3表＝768次生成 | `rope_fast_5090_20260912/supplement_source_counterfactual_prepared/` |
| layers | BM/MrPro效果主要由前、中、后哪组层贡献？ | 五任务各12条，共60；6种分层替换＝360次新生成，复用E2的两条全层基线；无法复用时另补120次 | `rope_fast_5090_20260912/supplement_layer_group_probe_prepared/` |

新增共1512次生成（不复用时1632）；它们回答三个机制问题，不是又搜一批曲线。p1来自双路线P1；source/layers分别覆盖原E7的来源追随与分层干预部分，未包含完整gold-mask/sham矩阵，不能称E7全部完成。层组使用E2旧问题，是机制子集，不是新独立确认。合成面板与LoRA训练seed/key/value独立，不能拿这些评价案例训练。

预留约2–6 GPU小时只是初步安排，按运行首段的真实prefill/decode重估；不据此压缩案例。新增LoRA改造见[LoRA v2设计](OLMO_LORA_V2_20260912.md)，其数据扩容与当前GPU实验并行；活动主线为Native与两个Cosh工作点，不再包含YaRN训练。

## 从论文选择这些工作的理由

重新读了当前摘要、§4学习兼容性、§5学习收益、§6成熟模型和讨论：

- LoRA直接补“PPL/来源概率改善没有变成完整任务和原生能力保留”的缺口。
- E2补论文五任务QA结论的强对手与gain条件，不重做曲线搜索。
- E3把350条开发面板上的同总位移结构差异推进到新案例。
- G1把纯几何量连接到模型实际Q/K系数；它是解释性诊断，不能用它代替任务分数或宣称完成全因果机制。

仓库的[current-stage S2/S3](index.md)、[实验审查](experiments.md)、[理论审查](theory.md)和[双路线说明](../../../experiments/twotrack_20260911/EXPERIMENT_PLAN.md)都包含GPU/CPU计算。G1采用其中“实际系数/有限相位”这条无需新训练资产的工作。双路线P1的384次距离/局部间隔/干扰数实验保留为后续候选，不和E7重叠重复开跑。

原E5的**新S1权重×训练表crossing**确实需要那些checkpoint，所以留作产物到达后的补充；这只影响E5原合同，不阻塞上述四条线。MLA复评有权重就接，缺权重不重训432M，也不挡当前工作。E7复杂插桩、Llama/E8、E9和C2 128K后置。

## 不再等待旧EVQ字节文件

本轮LoRA默认使用代码中明确构造的 **OLMo fixed-support anchored Cosh τ=2**，只与Native同配方比较；直接检验“解析分配的能力能否通过更合适适配恢复”。YaRN官方依赖全参数长窗SFT，当前LoRA套壳不具备清晰对照意义，因此不再训练。这是当前构造的恢复实验，不冒充旧adapter的字节复播。

旧表SHA `917a…4607` 的精确复播入口仍保留，但不是启动条件。用户已明确要求小步快跑，不再把追回它作为整机前置。Native/Cosh使用对应数据与预算，训练前各记一次0-step；完成输出、PPL、原生lost/gained分别看即可，不新增长审批链。

## 长程训练数据必须本身是2×/4×

用户补充的核心修正：OLMo原生窗口4K，长程能力训练使用真实8K和16K输入，不能只在4K训练后期待外推。`Native`训练臂指原始频率表，**不代表只喂原生长度**；Native/Cosh都训练8K/16K数据。

- **连续文本CPT**：PG19真实连续片段，预测跨度8192/16384；按`8K、8K、16K`轮转，两档各占50%CPT tokens。相邻16K原文块交替分配给两档，8K由其专属块拆分，不把同一段监督重复计入两档。
- **长指令SFT**：使用完整LongAlign长文任务，8K池要求prompt至少7K、总序列不超过8K；16K池要求prompt至少14K、总序列不超过16K。对应CPT长度取对应SFT池，保留完整答案和EOS，不以拼接无关短指令或padding充当长任务。
- **短能力保留**：Dolly等4K内样本只用于单独低权重replay/原生教师KL；不计为长程数据，不以其替换长指令。
- **任务依赖**：输入够长只是必要条件。LongAlign包含长文指令，但现有标签并不逐条提供证据跨度；不能声称筛过长度就证明使用了远端证据。开卡后抽看实际长文问答/摘要，并用未用于训练的RULER/NIAH深度与多信息检索、QASPER完整回答验证长程能力。

首段仍为每臂32Mi CPT tokens，混合长度后每臂3072updates；两臂共6144updates。Blackwell混合compile路径实测约27.5--28.2GiB，计算阶段达到97--100% GPU；LR、恢复与8Mi/32Mi观察点按真实累计tokens推进。显存调整不把长输入改成4K。

## 原始数据下载

仅在服务器数据盘下载，不占本机磁盘；原始数据下载已获用户明确授权。**不在0.5核无卡模式下分词、生成长案例或加载模型。**

目录：服务器数据盘相对路径 `olmo_recovery_20260912/sources/`。

| 用途 | 来源 / 文件 | 处理方式 |
|---|---|---|
| 长文本CPT与LM评价 | PG19：128本官方train、validation/test各24本；`pg19_books.json`、`pg19/{split}/` | 下载原文，保留官方split；8K/16K训练、32K评价窗口开卡后构建 |
| 长指令SFT | LongAlign-10k，固定revision `12f17c4baff1001f0d44c4f8feab09ee2ee8c6dc`；`longalign.jsonl` | 公开长指令，保留合成答案身份；不在无卡时tokenize |
| 短能力replay | Databricks Dolly-15k；`dolly.jsonl`，下载时固定revision写入receipt | 人写指令/回复；按来源分train/dev/test，每类别最多64/16/16。它是新的公开replay，不是旧Native池复播；开放答案F1有局限，另看原模型行为保留 |
| 自然长文QA | QASPER v0.3，train/dev与test两个archive | 评价使用官方dev/test；完整原文与LongAlign来源排除在开卡后处理 |
| RULER/NIAH | 已有上游revision `c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a`，PaulGrahamEssays、english_words、SQuAD、HotpotQA | 原始代码和四个数据JSON复用，不重复下载；NIAH是生成型任务，新980条实例开卡后生成 |
| E2 | 原三批自然QA的778条tokenized输入 | 已存在，复用原问题，不另下整套LongBench |
| 小模型储备 | 已有FineWeb-Edu两个parquet及冻结token流、GPT-NeoX tokenizer | 保留供已有/后续小模型工作；不拿NeoX token IDs直接训练OLMo |

下载器：`experiments/olmo_recovery_20260912/download_sources.py`，2个网络worker，支持中断继续；`sources/acquisition.json`记录每个文件、revision、大小、哈希及失败。只读此小回执即可跟进，不反复遍历/哈希全目录。

**下载完成：`RAW_SOURCES_READY`，180/180文件，777,255,900 bytes，0失败；逐文件大小与回执一致。** 其中PG19 176本共94,409,995 bytes；LongAlign 655,059,649；Dolly 13,085,339；QASPER两个archive共14,700,917 bytes。RULER四个源文件另外复用77,109,835 bytes。

Dolly固定revision `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`。每个文件的SHA在服务器 `sources/acquisition.json`；最终回执SHA256为 `06a804284c4ae0f58de6d9e67ee737991a10c3c9e25f316c1dfa8ea4a579b0e8`。下载进程已正常结束；没有待继续的下载任务。模型/词表文件也仍在盘。

**准备交付时尚未分词或生成新包；目前执行任务已在24904并行构建**8K/16K数据和E3新案例。此次只完成原始文件下载、源码同步、24个Python文件的AST检查、纯stdlib端点/恢复计数检查和文档检查；没有训练结果。

## 简单工时估算

首批四条线暂按 **1–3天执行窗口**安排，实际开卡后用首段更新速度重估；数据下载/预处理和LoRA评价仍额外计时，不承诺一晚全完。

- LoRA当前两臂到共同32Mi：按Blackwell warm-cache探针与真实新teacher阶段，Cosh+Native含开发评价暂估约 **3.5--4.5 GPU小时**；首臂完成后以完整回执更新。
- E2原计划参考 **6.59 GPU小时**；E3参考 **5.08 GPU小时**。两数来自原计划Pro6000吞吐假设，先作为规划参照，不冒充5090实测。
- G1只有20次backbone forward，先留 **0.5–2 GPU小时**的加载/诊断安排；这是预留，不是测试结果。
- 上述计算小计约 **21–47小时**，未含未定的数据处理、LoRA完整评价和工程修复。LoRA128Mi是后续同轨迹延续，首段不自动承诺跑满。
- 不能拿纯CPT吞吐或probe的所有监督tokens/s套进CPT工时分子。第一条健康训练跑起来后，直接按完成的CPT tokens/实际时间更新ETA。

原计划推荐E1–E8是8组科学实验，加此次LoRA为9组，E0另为准备；它们不是都在本轮同时运行。原89.9/165 GPU小时是跨机器历史总预算，不是当前剩余时间，且不含新增LoRA。当前四条线不需要等待其余项目。

## 机器、资产和验收

- 5090入口：`ssh -p 24904 root@connect.westd.seetacloud.com`。
- 后续无SHA门禁源码：数据盘相对路径 `rope_fast_5090_20260912/prepared_code_no_sha_20260912/`。当前健康进程保留原`prepared_code/`，不因更新重启；新启动切到新源码根，数据/输出路径沿用。
- 模型：`olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct/`，1,484,916,736参数；上轮确认权重/词表在盘。
- 另一session westc:53405管理4080的S1。不得重复启动、迁移或干预其训练。
- 保留全部现有模型权重。之前5090克隆中的Llama/Qwen清理已向用户说明，原4080未动；不再进行清理，不下载Llama。
- 轻量语法/文档检查已做；最终新增教师轨迹/G1/评分代码未跑GPU，开卡后用最短必要检查发现并修复问题，随后正式跑，不反复建检查轮次。
- 当前已写好E3主差值/paired bootstrap；normalized-any-reference与单答案literal/multi-answer诊断已分开。LoRA配对native lost/gained收尾仍需接上，但不会阻塞原始完整输出保存。

## 公开数据扩容与移交

原始UltraChat三个train_sft分片＋LongAlpaca已下载1,230,343,245 bytes；LongCite与PG19扩容由执行Sol接续。完整LongAlign的新训练池已为8K 989条、16K 754条、17,978,985个样例输入tokens；相比R0的358/70，取消抽样上限并按完整来源筛选已有明显扩容。其他来源实际筛选数量以各`data_expanded_*`的manifest为准。

用户要求不持续监控下载，已交给任务“5090实验负责agent-5.6sol”。交接时估剩余下载10–20分钟、分词筛选20–40分钟；这些是安排参考，不能据此宣称完成。v2代码已补齐、独立部署，真实GPU数值验证仍由Sol做最短必要检查。
