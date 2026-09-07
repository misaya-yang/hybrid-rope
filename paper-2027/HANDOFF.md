# Hybrid-RoPE 当前交接

- **更新：** 2026-09-07；作者已回家，明确由家用PC的Codex接替；本机整理并通过Git同步当前工作。
- **状态：** 作者要求持续推进方法研究，当前禁止启动新实验；研究阶段尚未完成。EVQ 最后核验为已关机，`rope` 监控已暂停，无待续跑的旧进程。恢复时重新核对，不能把旧 PID 当活进程。
- **职责：** 这里只保留当前状态、授权、资产定位和下一步；规则在 [AGENTS](../AGENTS.md)，结果在对应 owner，项目介绍在 [README](../README.md)。

## 回家后的继续入口

从本文件继续，不需要作者重新说明背景。作者现要求结束“猜想—失败”循环，已形成 [研究复盘](../docs/research/ROPE_RESEARCH_FAILURE_REVIEW_20260907.md) 和 [交给 Pro 的最终问题](../docs/research/ROPE_PRO_DECISION_REQUEST_20260907.md)，目标是下一工作日取得有依据的方法改进与可解释结果。作者已自行发送最终提示词和复盘，Pro 无法访问仓库；本任务尚未收到本次回复；不重复改写早先已发送的提示词。先结合 Pro 新回复（若已到）选择主方案；没有回复时可继续基于现有材料研究，不把等待回复变成全局停工理由。再打开 [本轮协议、结果及解释修正](../docs/research/ROPE_SCALE_TRANSPORT_PILOT_20260907.md)，重点读末尾的组合 RULER 结果，再接着处理失败原因。

本阶段目标仍是改进本方RoPE方法；尚未解出有成功证据的新构造，不能把分析工具完成当作课题完成。家用PC接手后按下面的实际进度继续，不重新开始资料大扫查：

1. **先读复盘末节及 [已恢复的旧Qwen有效表](../docs/research/ROPE_RECOVERED_QWEN_P2_20260907.json)。** 三个数组哈希已核对；旧结果属于Qwen1.5B，频率几何与当前3B相同。与MrPro的主要分歧是中段后半程，低频同样接近`/4`；不要再把今天失败提案的低频当作已验证本方优势。
2. **复用正确的共享频率响应入口。** [shared_frequency_response.py](../scripts/analysis/shared_frequency_response.py) 在NumPy构造数组上验证完整GQA、跨key/head抵消与零频率导数；尚未接上真实QKV缓存。SSH拒绝连接，AutoDL要求重新登录，已向作者请求无卡访问。只读缓存，不需要新模型前向。缓存分析仍不等于下游能力优化器；历史18样本/64维行为梯度路线已失败，勿原样重启。
3. **方法先收敛，再考虑最多3次实验。** 有效Mr与旧p2是起点；解释其共有段/分歧段，结合真实响应决定中段、尾频、衔接怎么改。低频统一加carrier的恒等式只是未选定线索，不是待执行候选。结合Pro回复时核查它是否真正解决这些问题，不盲从，也不因回复未到停止独立研究。
4. **保留现有生成诊断。** UUID短控也有错，不能把长格0分全归频率；VT短输出遗漏了一个变量后进入解释，不能认定只需增加token预算。旧长VT为未测。必要协议修复另存记录，暂不启动计算。

作者已将最终提示词和复盘发给Pro，对方不能访问仓库。本机未收到这次Pro回复。当前禁止新实验；先持续研究及已有数据分析，不逐个小发现请求“继续”。本机没有正在运行的后台计算需要家用PC接管。

## 当前方法与结果位置

- 已完成 Qwen2.5-3B-Instruct 冻结试验，未进行新的权重训练。当前组合为 MrPro 中段、本方尾频与 CoPE 风格末 20 槽衰减；不是 Cosh，也不是微调 MrRoPE。
- [主结果 owner](../docs/research/ROPE_SCALE_TRANSPORT_PILOT_20260907.md)：构造、两项自然问答诊断、组合 RULER 结果、逐行解释和原始回执哈希。普通数字检索在 128K 档通过，UUID 检索失败，VT 长格未测；尚无 SOTA 证据。
- [数组及 CPU 分析](../docs/research/ROPE_SCALE_TRANSPORT_FOLLOWUP_20260907.json)保留构造时状态；后续已执行情况由主结果 owner 更新。组合 tensor SHA：`a63fe2714b37a30b4387fff8af500c270b62770a5b90677302083569af9c2eb2`。
- [研究主线](../docs/research/ROPE_FREQUENCY_UNIFIED_PLAN_20260907.md)解释 MrRoPE / CoPE / 本方工作的关系；[Pro 方案评议](../docs/research/ROPE_SCALE_TRANSPORT_REVIEW_20260907.md)区分采用部分与未验证假设。
- [OLMo E0/E1 owner](../docs/research/ROPE_FREQUENCY_LUNA_ROI_20260907.md)是已完成历史证据；旧 Z-only 1528-step 训练从未启动，不属于当前自动执行队列。禁止恢复 seed42 权重或自动回到旧三臂微调。

## 执行资产与预算

| 对象 | 已知位置/状态 |
| --- | --- |
| 服务器工作区 | 已有 EVQ 实例中的 `rope_qwen_baseline_20260907`；SSH/绝对路径沿用当前任务或私有回执，不更换实例；GRPO 是另一台机器 |
| 模型 | `model/`、`model_ready.json`；Qwen 官方 revision `aa8e72537993ba99e69dfaafa59ed015b17504d1` 已下载且核验 |
| 自然文本数据 | `prepared_v2/`；首版因聊天模板返回 BatchEncoding 而失败，修正版已运行 |
| RULER 数据 | `ruler_prepared_01/`；30 条冻结行及源版本、哈希均在 manifest |
| 已完成输出 | `runs/pilot_01`、`runs/unguarded_01`、`runs/tail_01`、`runs/combined_ruler_01` |
| 执行代码 | [scale_transport](../scripts/experiments/scale_transport/)；监督器 [cross_audit/jobs.py](../scripts/experiments/cross_audit/jobs.py)；原执行代码版本与旧 plan 保留在服务器，不覆盖其身份 |
| 本地紧凑回执 | 私有维护目录 `qwen_baseline_20260907/result_readback` 与 `combined_ruler_readback`；两份原始压缩包已校验 |

作者后续要求最多用 **3 个实验** 完成方法验证及必要修正；不得用一个实验名隐藏多候选搜索或不断追加。这是新的实验次数约束，不是三次必成的已验证保证，也不重置原有GPU时间额度。当前新增的是复盘中的缓存统计与频率映射分析，尚未启动这三次实验。

此前总额度为两小时；已记录作业耗时约 **875.55 秒**，不等于云端占卡/账单时间。旧 RULER plan 的绝对截止为 2026-09-07 13:15:15 UTC，属于已结束运行，不能直接复用。恢复时计入已有消耗、计算剩余额度并冻结新的运行截止，不因新会话/重启再自动获得两小时；需要扩大预算时才向作者提出明确缺口。

已有验证：四个工作机 CPU 测试、真实 Qwen hook 与 Flash/BF16 路径、静态数组身份及回读评分。新诊断入口尚未实施，不能把已完成脚本原样重新启动当作继续研究。GPU 开启前尽可能完成 CPU 准备，已有运行期间并行准备下一步。

## Local Git and manuscript identity

- 工作分支 `main_0726_09_06`；本轮代码、规则、协议和结果说明通过 Git 提交保存。提交身份用 `git log -1` 核对，不在提交内容中硬编码自身 SHA。
- 换电脑继续使用该分支的最新提交，先读本文件，再读主结果 owner；不需要另行制作或携带私有交接包。并发任务已同步过先前提交；最新可用状态以 `git status -sb` 与实际推送结果为准。
- 作者原有八份 solver/attack 草稿已由其他并发 Git 操作归档（`07c7978`），本研究任务未修改其内容。模型与大型原始回执留在原服务器/私有存储，仓库保留结果、来源和哈希；需要复核时按对应 owner 定位。
- 活动 PDF SHA：`37aa6402a65d68b21909b0b3479c4e8edd811079e3922c2c1be915ddeab167e4`；TeX/PDF 未改、未重新编译，`main_0726` 历史分支未操作。

历史瞬时 PID、下载百分比、开关机过程及旧执行指令不再堆叠在本文件。对应科学结果、清理记录和外部原文仍由 INDEX 路由；整理前文档另有私有备份。
