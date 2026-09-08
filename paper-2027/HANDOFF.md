# Active owner update — 2026-09-08, native selector diagnostics

The active goal is `docs/research/ACTIVE_RESEARCH_GOAL.md`; continue research and
real experiments until a supported core conclusion and corresponding paper exist.
Do not treat the diagnostics below or the provisional budget draft as completion.
User clarified that avoiding waste includes avoiding long GPU idle periods:
reasoning and useful experiments must progress together, without blind scans.

Read `experiments/native_sparse_position/CORE_DIAGNOSIS.md` first. PSR real-Q/K
checks finished on Qwen3.5-0.8B and Qwen2.5-3B; no robust advantage over count-matched
contiguous/random partitions. Exact RoPE/NoPE selection oracles then generated all
16 development event answers from a question-blind prefix. All 48 EOS; R/N raw
outputs identical in 15/16 pairs, one NoPE content correction. Strict prescribed
answer format fails throughout; manual content annotation is a separate endpoint.
These are local mechanism tests, not a deployment or positional-necessity theorem.

Currently running `natural_qwen35_01` (PID21638 when launched) at
`/root/autodl-tmp/position_observability_20260908`, SSH port27741. Fixed inputs
`natural_inputs_01` contain 24 untruncated natural QA examples, 8 per task, jointly
8K–16K in both tokenizers. Native Qwen3.5 six full-attention layers use Dense,
RoPEOracle, NoPEOracle from the first question token; recurrent layers remain
unchanged. All-keys-gather bitwise native-logit parity passed. Check actual status,
log and PID; then run the identical prepared inputs on existing Qwen2.5-3B and
analyze complete outputs. Never restart on an old PID or a shell timeout alone.

Oracle scores all keys and reads 32 remote blocks per query head, B64, local2048,
sink64. It is explicitly not a fast selector. NoPE removes only the selector's
explicit rotary transform; hidden states and the native reader still encode order.
Full natural outputs carry EOS, trimmed exact, normalized exact and whole-answer
F1; no substring or first-line extraction. Custom prompts/development subset are
not official full LongBench scores. Source code, pinned models and input hashes
are recorded. The old rotary training assets remain prepared, not scheduled.

Pro's one-time follow-up was completed; new source is
`experiments/native_sparse_position/DESIGN_SOURCE.md`. Do not poll that task again
or send messages to other tasks. The original ZIP was not acquired; current
reference implementation was independently written from the downloaded Markdown.
No commit, push, deletion or server shutdown is currently requested.

Paper remains a provisional budget-focused draft (previous compile main5/total33
pages) with old empirical sections in supporting material. Its final central claim
must be chosen from actual new evidence. See the following historical entries only
for provenance; their old goals, running PIDs and claimed next steps are superseded.

---

# Active owner update — 2026-09-08, core experiment first

User now prioritizes a high-value modern positional-encoding question and one solid
core experiment over completing six arms. The two Pro documents are starting points,
not mandatory plans. Do not launch the old zero-training search or rotary six-arm
queue automatically. No messages to the independent audit conversation.

Read `docs/research/POSITION_AFTER_SPARSE_RESEARCH_20260908.md` for current research
judgment, `experiments/rotary_budget/REPORT.md` for prepared execution evidence,
and `experiments/kld_v2/RESEARCH_PLAN.md` for the separate follow-up.
The official recent Qwen3.8-Next NoPE/termination observation and Kimi K3 KDA+NoPE
architecture motivate a conditional positional-necessity problem. Their cross-model
difference is not a causal explanation.

Real-data preparation is complete; short training probe17018 is COMPLETE. No formal
long training is running. Remote root `/root/autodl-tmp/rotary_budget_20260908`, ssh
port27741. Server has the restored pinned train/validation sources and exact-stream
hash verification. Probe throughput74.65k tokens/s on the actual4080SUPER. Preserve
these assets. Latest heartbeat follows core-first priorities.

Paper: budget-focused draft compiles, main5/total33pages, zero undefined refs and
overflows; provisional because new neural results and final central claim are open.
Main finite-bound proofs added; incorrect inference from K^-2 distortion to actual
budget benefit removed; prior multi-regime results demoted to supporting material.
No commit/push made.

---

# 当前交接

- **用户最新协调约束：** 不再向此前独立复核对话发任何消息，由本任务独立研究；必要时可开一个Astra子代理。已开一个只读机制候选子代理，禁止其GPU或跨任务通信。当前7B筛查PID9227运行，OLMo阶段干预只准备未排入并发GPU。

- **覆盖上行7B PID：** run_screen_01首128K OOM后保留6条短MrPro；run_screen_02/PID9764用expandable_segments恢复并校验复用这6条。128K现已成功，实际51–60秒/条。MrPro完成、BM在跑。顺序队列PID10306等待7B成功和GPU空闲后启动OLMo `stage_run_01`，固定single_1_16384_1与qa_2_16384_13；见阶段协议，非并发GPU。若用户叫停须同时停止等待队列。

- **最新实际状态，覆盖下方旧PID/空闲描述：** OLMo自然三任务567条/臂完成，扩展长度F1宏均值BM26.01%对Mr21.37%；新增7项RULER各50条/臂完成，16K宏均值41.67%对7.09%，156胜9负。NLL16篇自然前缀完成：4K Native2.835/Mr3.211/BM2.955；8K Mr3.257/BM2.955；16K Mr3.688/BM2.862。NLL仅最后512下一token，不是生成证明。对应RESULT文件已写。正在准备NarrativeQA/MultiFieldQA的全可用未截断池；7B官方下载metadata匹配的镜像权重仍下载中。单GPU所有者仍本任务。

- **深线最新否证：** 双正交投影残差BM虽精确保留公共方向，却不是正交相位算子；实际均值产生方向增益最高约1.65，不能忽略。若进一步要求同维正交群在所有lag完整保留循环公共向量轨道，通常迫使退回MrPro。均值核统计亦不支持BM普遍增大背景。故该路线未上GPU，不根据均值代理选表；继续深入解释已观测的跨任务收益与模型差异。

- **作者最新判断规则：** 看跨模型、NLL、passkey、RULER、QA的整体证据，不要求每个benchmark子集全胜，不因单个分项下降自动否决；保留负项与代价，不能混合量纲掩盖。理论逻辑优先：为什么比MrPro好、何处有效、还能否更好。整体有价值后深入，而非继续换曲线。仍按快慢并行安排GPU与深入思考。

- **快慢并行正在执行：** 作者强调轻量GPU验证期间同步深入思考，不要串行等理论。现运行既有OLMo BM正结果的[自然QA迁移](../docs/research/ROPE_OLMO_BM_NATURAL_TRANSFER_20260908.md)：Hotpot/2Wiki/Qasper各60条，MrPro自然匹配基线此前缺失故补一次归档，再跑原BM；不是新猜测曲线。远端OLMo根`prepared_natural_01`/`run_natural_01`/`code_natural_01`，PID4395。与此同时继续3B机制分析，勿并发第二GPU作业。

- **最新完成与纠正：** [Gap-capped完整负结果](../docs/research/ROPE_GAP_CAPPED_RESULT_20260908.md)：32K84.44%对Mr87.22%，128K62.15%对78.13%，0胜7负29平。只跑本方36条、复用原Mr基线，899.87秒；原始数据与执行快照已取回核对，PID2991已退出，GPU空闲。作者警告不要“改一个破坏一个、十分钟一个猜想”循环，因此CausalGain代码虽已通过2项CPU测试但暂停未运行，K/V亦暂停；不扫描cap。下一阶段先建立能同时分析原生关系保留与长距区分、对选法有判别力的机制依据，再针对验证；不能把另一几何最优解自动上GPU。

- **当前授权与主线：** 作者授权今夜自主推进README零训练收益目标，BM只是候选之一。从2026-09-08 14:15 UTC开始，约10小时不是硬截止；未解决持续到用户睡醒回来，只有跨模型全量可比验证支持彻底完成才授权关服务器。见[本夜研究owner](../docs/research/ROPE_OVERNIGHT_RESEARCH_20260908.md)。作者强调方法不能猜：18/18混层因缺乏构造依据已撤回，STOP已生效，仅两次uniform资格回放，没有混合候选结果，当前GPU空闲。方法需有机制/推导依据且能在论文讲清楚。已有可比MrPro基线复用，零训练胜出后换模型/任务；缺基线才补跑一次存档。K/V诊断代码已准备未运行，先补结果到方法构造的决策映射，避免无目的下钻。自动接续`rope`每30分钟跟进直到用户回来/目标完成。不要因上一小诊断完成而停止主目标。

- **最新接续结果：** [3B前缀形成×读取表交叉诊断](../docs/research/ROPE_BM_CROSS_CACHE_20260908.md)已完成。固定两反例均跟随前缀来源：MrPro前缀用BM读取仍保持多键正确/VT召回100%；BM前缀用MrPro读取仍误绑定/VT20%提前EOS。全部前缀可见，直接捕获pre-RoPE K后重旋转，V保留来源状态。VT/MrPro缓存query与完整O仅第30token有差异，因此另立匹配cached基线、同表重建逐token核对，不能宣称原输出完全一致或100% EOS exact。16次生成含资格与一次停止后的VT接续；两例诊断完成，GPU空闲。下一步可定位有限层组的预填充状态来源；未运行层组矩阵。远端资产`bm_transfer_20260908/cross_cache_run_01/02`及`code_cross_cache_01/02`，本地已取回两轮JSON和对应运行器。旧轮次状态保留如下。

- **目标与授权：** 作者要求从MrPro自主实现零训练改进，运行、及时分析并实施后续方法；不设置例行确认门槛。
- **OLMo轮次已完成：** [完整结果](../docs/research/ROPE_OLMO_BM_RESULT_20260908.md)。7个GPU阶段、780次完整生成；全部正常结束，逐行重算与回执核对完成。
- **保留成果：** BM在OLMo-2-0425-1B-Instruct、静态S4的独立72条六任务复核中，16K为51.32%，同输入MrPro2.78%、MrUni32.12%、官方YaRN6.94%；4K BM81.81%。这是局部能力收益，不是完整RULER、跨模型或SOTA结论。
- **已实现：** [可复用BM函数](../scripts/lib/rope/boundary_matched.py)，与实际FP32实验数组逐位一致；29项相关测试通过。标准S4 BM作为当前已验证配置；新幅度分配两项均未超过它。
- **适用边界：** 静态S8在32K BM6.94%、MrPro0.69%，检索/追踪地板；同32K使用S4 BM为0%，S8频率+S4gain为5.56%，两个恢复方案均不晋级。Native短端也有逐任务取舍。不要重启这些固定负结果，或把短端恢复当32K收益。
- **服务器：** `ssh -p 27741 root@connect.westc.seetacloud.com`，RTX4080SUPER32GB；Python `/root/miniconda3/bin/python`。OLMo作业已完成；Qwen3B冒烟也已完成，无运行中的本轮作业。机器未关机。
- **资产：** 远端根目录`/root/autodl-tmp/olmo_fast_screen_20260908/`；本地`results/olmo_fast_screen_20260908/`保留完整输入、逐行生成、manifest及各阶段源码快照。归档快照与最新源码分开，旧manifest应配其对应快照。
- **后续研究边界：** 可以从已验证S4 BM继续做其他模型/自然任务的匹配检验，或提出能针对16K密集绑定/追踪残余失败的新干预；现有结果不支持继续调失败gain系数、盲增scale或无条件扩大32K矩阵。

- **最新任务已完成：** 作者最终指定MrRoPE论文的Qwen2.5-3B-Instruct，先做冒烟再考虑全量。[3B结果](../docs/research/ROPE_BM_TRANSFER_RESULT_20260908.md)：32K BM91.67%对MrPro87.22%；128K BM70.83%对MrPro78.13%，长端4胜4负16平、均分下降7.29pp，未晋级全量。72次生成全部正常完成，约29.44分钟；1.5B按用户要求在MrPro19/36条时中止，不作方法结论。
- **跨模型资产：** 远端`/root/autodl-tmp/bm_transfer_20260908/`；本地`results/bm_transfer_20260908/`，含原始输入/输出、metadata和执行代码快照。最新运行器支持分片权重与各模型原生短端长度；32项相关测试通过。不要重新启动被用户中止的1.5B或无条件展开3B全量。

- **用户最新要求已完成：** 研究128K错误并尽快收尾、提交推送，回家PC接续。[最小诊断](../docs/research/ROPE_BM_128K_DIAGNOSIS_20260908.md)两例已完成14次生成：C/P均满分；保留完整预填充KV的L多键仍0%、VT仅60%，O与原冒烟逐token一致。说明原长背景形成的状态参与退化，不能仅归因于128K坐标或单一竞争key；层/头/槽机制尚未识别。原始数据远端`bm_transfer_20260908/diagnosis_run_02`，紧凑证据随Git保存。全部GPU任务已结束，无自动队列。
