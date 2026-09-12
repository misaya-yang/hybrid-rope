# 今晚实验守则（Plan B 执行授权版）

更新：2026-09-11（America/New_York）
状态：**只冻结守则；等待 Claude Code 完成代码后再审计、上卡和启动。**

## 0. 权威顺序与目标

1. 科学协议以 `/Users/yang/Downloads/RoPE_Integrated_Experiment_Guide_Codex_20260911.md`
   （Plan B，sha256 `ad63f0b62eb387215302fd907949109ad4fff445ad0a82fd69353696960f8f1e`）为准。
2. Plan B 锁定的原 60 配置来源是
   `LLAMA3_ROPE_20_DIRECTIONS_60_CONFIGS_REVIEW_PLAN_20260911.md`
   （sha256 `7af9c9eca79996b30603c6d173b7fcd5d42b2497b7c55216eee35a308d2cb8d7`）。
   其他同名或相近的 `60_METHODS_PLAN` 不能静默替换候选身份。
3. 本守则补充执行授权、监控和成功后的泛化要求；与 Plan B 冲突时，科学定义、数据隔离、
   统计判据和证据等级服从 Plan B。
4. 目标仍是：在冻结权重条件下找到真实任务成绩优于 MrRoPE-Pro 的可部署方法；
   CPU 恒等式、NLL、coverage、attention 或代码数量不能替代该目标。
5. **今晚主实验只运行 Meta-Llama-3-8B-Instruct。** Plan B §5 的 OLMo/Qwen 历史闭环
   不进入今晚 GPU 队列；它们只作为既有背景和以后单独核查的资产。Qwen 与 Llama-2
   仅在 Llama-3 得到成功锁定规则后，按 §7 作为泛化验证运行。

## 1. 上卡前的唯一启动门

Claude Code 宣布代码准备完成后，由执行者做一次最终只读审计和最短必要 CPU/parity 检查：

- 候选 ID、公式和 scope 与 Plan B 逐项一致；
- stock/native、MrRoPE-Pro、官方 YaRN、BM、UNI 及新增控制走同一真实 forward/scorer；
- P/S/V/H 与 semantic/source cluster 隔离，旧 350/180/fresh_72 不冒充新 H；
- Q/K pair layout、GQA 映射、gain 约定、KV cache、真实 token 长度与生成输出合同正确；
- 串行队列、断点续跑、原子落盘、失败状态和读数器可用；
- 至少一个真实模型的最短端到端 probe 通过，且无 GPU 时不会误报 READY。

未通过者先修；不得为赶开卡把实现失败记作候选 0 分。通过后立即进入真实实验，
不在付费 GPU 上重复无信息的准备检查。

## 2. 执行者的修复授权与边界

代码基本准备完成后，执行者负责连续执行，并可自行修复：

- 普通运行错误、依赖/API 漂移、路径、参数、恢复和日志问题；
- 与 Plan B 不一致的候选身份、控制、数据隔离、统计或评分逻辑；
- 已知错误继续传播的问题，例如 gain 重复、部分面板读数、符号读反、跨模型硬抄槽号、
  无效 dry-run、缺臂却自动判决等。

修复必须保持可审计：保留旧产物；记录代码/配置 hash 和修复原因；若修复改变了科学合同，
已受影响的行标 `INVALID_IMPLEMENTATION`，只重跑受影响的完整比较单元，不与旧结果混算。
不能因为某个结果难看而改变 scorer、样本、阈值或公式。

## 3. 串行队列与持续运行

1. 大量实验使用一个可恢复的 **Llama-3-only 串行队列**，按 Plan B 的依赖与完整对照单元排序；
   一次只让一项真实实验占卡，完成/失败后自动接下一项。
2. 队列中每项预先写明 `requires / parent / statistic / decision / on_fail / next`；
   提交或排队不算正在实验。
3. 进入付费 GPU 阶段后，真实 GPU 不得因 CPU 分析、重复校验或人工等待连续空转超过两分钟。
   下一项有效实验必须提前准备；15 分钟提醒不是允许空转 15 分钟。
4. 明确的工程故障由看门狗即时发现并按同条件最多重试一次；仍失败则记录、修复或转下一
   READY 单元，不能让整个夜晚卡在一项上。
5. 不为占卡运行虚假负载、重复无信息实验或已被 Plan B 关闭且没有新机制依据的方向。

## 4. 15 分钟跟踪任务

串行队列真实启动后，立即创建当前任务的 **15 分钟周期 heartbeat**。它每次应：

1. 检查 SSH、GPU、主进程、队列状态、最近日志、结果文件完整性和剩余磁盘；
2. 识别完成、崩溃、OOM、停滞、部分数据、NaN、身份漂移或 GPU 空转；
3. 对新完成的完整比较按预注册判据判断 `SUCCESS / NEAR_MR / HARD_FAIL /
   UNDERPOWERED / INVALID / BLOCKED`；
4. 需要时执行已授权的常规修复，或推进到预先登记的下一实验；
5. 用真实结果更新理论判断和下一项准备，但不从 H 结果临时发明公式；
6. 状态无实质变化时保持安静；只在成功、失败、修复、分支变化、需要用户动作或最终完成时通知。

heartbeat 负责研究级复盘；进程级连续性由本地串行控制器/看门狗负责，不能依赖 15 分钟轮询。

## 5. MrRoPE-Pro 基线只准备一次

- 对每个严格匹配的 `model + checkpoint/tokenizer + dataset/split + relative lengths + scorer +
  generation config + operator/gain convention`，MrRoPE-Pro 只运行一次，保存原始输出、manifest、
  operator hash 和 row identity，之后所有候选复用该配对基线。
- 不为每个候选重跑基线。换到 Qwen 或 Llama-2 时，因为模型与窗口不同，各自只建立一次
  对应的 MrRoPE-Pro 基线。
- 若后来修复使原基线科学合同失效，不得假装仍可比；把受影响比较标无效并明确处理，
  不用不同合同的旧基线拼接结果。

## 6. 结果分流

### 6.1 明显失败

按 Plan B 已冻结的 hard-fail/guard/validity 规则判定。明显失败仍保存最小回执、原始输出、
效应量和失败类型，然后直接推进队列；不为它设计优化版，不写长复盘，也不把单一失败扩大为
整个方法族不可能。

实现故障、地板/天花板和统计无功效不属于“明眼失败”，分别记 `INVALID_IMPLEMENTATION`、
`INSTRUMENT_LIMITED`、`UNDERPOWERED`，不能当方法负结果略过。

### 6.2 接近 MrRoPE-Pro

每个 batch 读数前冻结该 batch 的灰区。若无更具体的预注册，默认 `NEAR_MR` 为：结果有效、
无 hard guard 失败、尚未达到确认成功，并且相对 MrRoPE-Pro 的主任务点估计落在 ±5pp 内。
区间很宽但点估计不在该范围的结果记 `UNDERPOWERED`，不冒充“只差一点”。

每个 `NEAR_MR` 候选必须新增一份短 MD，至少写：

- 方法/父方法/实际 operator hash、数据和基线回执；
- 主效应、逐长度效应、CI/SE、strict/QA/EOS/format/native guard；
- 哪些逐样本得失造成差距，差距是 reach、resolution、assignment、gain、decode 还是测量问题；
- 一个机制明确、可证伪的优化版本，以及为什么它不是读结果后任意扫参数；
- 对应对照、预计效应、MDE、成本、解锁条件和放弃条件。

优化只能在仍属开发的 S/C-dev/V 或全新数据上进行；H 不回流选公式。每个近失候选默认只允许
一个最有依据的优化版本进入队列，除非新证据明确解锁 Plan B 已登记的其他分支。

### 6.3 成功

“成功”至少要求在有效的新数据上优于全部登记强基线，满足 Plan B 的区间、逐长度、native、
strict、QA 和实现 parity 要求。只胜 MrRoPE-Pro、只胜弱基线或只改善 NLL 不算最终成功。

成功后先锁定完整规则和 hash，不在目标模型上继续调参；随后进入跨模型泛化。

## 7. 成功后的 Qwen 与 Llama-2 泛化

对锁定规则至少验证 Qwen 和 Llama-2，不能硬抄 Llama-3 槽号：

1. 从各 checkpoint 实际 config/module 读取 `W, theta, head_dim, K, rope_scaling`，按同一物理规则
   重新构造边界、profile、amplitude 和 gain；公式本身不得用目标结果重调。
2. 为各模型分别生成与其原生窗口相适应、语义任务结构相同的独立数据：native guard 在 `W`，
   长程至少覆盖相对 `2W/4W`；预留输出 token，使用目标 tokenizer 后的真实长度和 evidence 位置。
3. 同一模型上一次性准备并复用 `native / MrRoPE-Pro / official YaRN / BM / locked method`，
   保持任务、scorer、生成配置和 row identity 配对。
4. 保存完整自由生成输出；NLL 只作为独立诊断，不替代检索/QA/strict/EOS 结果。
5. 报告各模型相对其本地强基线的效应和区间，不直接比较不同模型 raw score。

Llama-3、Qwen、Llama-2 三家族同向且通过各自 guard，可支持“在这些 checkpoint、相对长度和
任务域中跨家族泛化”；仍不得写成对所有模型的普适定理。某家族失败时保留有限外部边界，
不得回头用其结果调整锁定公式后仍称零调参迁移。

## 8. 今晚结束条件

- 队列未完成且仍有 READY、能改变决定的实验：继续运行并保持下一项已准备。
- 队列完成、真实阻塞或已有结果使后续预登记臂不再增加信息：安全落盘、汇总状态和剩余问题。
- 关机只使用用户届时明确授权且已验证的现有平台 hook；不同步结果、不确认其他进程时不关机。
- 最终报告首先回答是否找到优于强基线的方法，再回答机制与泛化；不以“跑了多少配置”代替结果。
