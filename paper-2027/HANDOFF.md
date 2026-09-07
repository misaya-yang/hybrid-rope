# Hybrid-RoPE 当前交接

- **更新：** 2026-09-07；作者要求先整理文档，回家后继续研究。
- **状态：** 作者主动暂停；研究阶段尚未完成。EVQ 最后核验为已关机，`rope` 监控已暂停，无待续跑的旧进程。恢复时重新核对，不能把旧 PID 当活进程。
- **职责：** 这里只保留当前状态、授权、资产定位和下一步；规则在 [AGENTS](../AGENTS.md)，结果在对应 owner，项目介绍在 [README](../README.md)。

## 回家后的继续入口

从本文件继续，不需要作者重新说明背景。作者现要求结束“猜想—失败”循环，已形成 [研究复盘](../docs/research/ROPE_RESEARCH_FAILURE_REVIEW_20260907.md) 和 [交给 Pro 的最终问题](../docs/research/ROPE_PRO_DECISION_REQUEST_20260907.md)，目标是下一工作日取得有依据的方法改进与可解释结果。提示词已准备，未代作者发送、尚未收到本次回复；不重复改写早先已发送的提示词。先结合 Pro 新回复（若已到）选择主方案；没有回复时可继续基于现有材料研究，不把等待回复变成全局停工理由。再打开 [本轮协议、结果及解释修正](../docs/research/ROPE_SCALE_TRANSPORT_PILOT_20260907.md)，重点读末尾的组合 RULER 结果，再接着处理失败原因。

本阶段目标仍是改进本方 RoPE 方法，寻找有实际价值的中段、尾频与衔接方案。下一段连续工作是：

1. **UUID 检索：** 区分短程本来就难、模板/答案抽取问题与频率造成的额外损伤。复用现有错例，只有能改变判断时才补最小必要参照，不重跑对手论文。
2. **变量追踪：** 两条短控制均得 80%，随后在 30-token 预算处结束。分清答案预算、回答格式和遗漏变量；官方分数、完整答案和 EOS 分别记录。原规则跳过的 8 条长样本是未测，不是零分。若需修正诊断协议，保留原结果并用新 run ID。
3. **继续推进方法：** 据上述结果选择下一项必要改动/测量，准备、运行、分析连续进行；小实验完成不是再向作者索要“继续”的交接点。不在同一小面板上反复挑表后宣称独立确认。

这不是待作者逐项批准的菜单。作者恢复本阶段后，在已有资源与预算范围内由执行者补齐诊断入口和必要配置并推进；当前暂停要求优先，本次整理不启动计算。

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

此前总额度为两小时；已记录作业耗时约 **875.55 秒**，不等于云端占卡/账单时间。旧 RULER plan 的绝对截止为 2026-09-07 13:15:15 UTC，属于已结束运行，不能直接复用。恢复时计入已有消耗、计算剩余额度并冻结新的运行截止，不因新会话/重启再自动获得两小时；需要扩大预算时才向作者提出明确缺口。

已有验证：四个工作机 CPU 测试、真实 Qwen hook 与 Flash/BF16 路径、静态数组身份及回读评分。新诊断入口尚未实施，不能把已完成脚本原样重新启动当作继续研究。GPU 开启前尽可能完成 CPU 准备，已有运行期间并行准备下一步。

## Local Git and manuscript identity

- 工作分支 `main_0726_09_06`；本轮代码、规则、协议和结果说明通过 Git 提交保存。提交身份用 `git log -1` 核对，不在提交内容中硬编码自身 SHA。
- 换电脑继续使用该分支的最新提交，先读本文件，再读主结果 owner；不需要另行制作或携带私有交接包。本次仅提交，尚未推送；远端可用状态必须以实际推送结果为准。
- 作者原有八份 solver/attack 草稿不属于本轮提交，保持不动。模型与大型原始回执留在原服务器/私有存储，仓库保留结果、来源和哈希；需要复核时按对应 owner 定位。
- 活动 PDF SHA：`37aa6402a65d68b21909b0b3479c4e8edd811079e3922c2c1be915ddeab167e4`；TeX/PDF 未改、未重新编译，`main_0726` 历史分支未操作。

历史瞬时 PID、下载百分比、开关机过程及旧执行指令不再堆叠在本文件。对应科学结果、清理记录和外部原文仍由 INDEX 路由；整理前文档另有私有备份。
