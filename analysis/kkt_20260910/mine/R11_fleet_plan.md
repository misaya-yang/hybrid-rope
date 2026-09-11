# R11 — 30 代理舰队：原始任务分解、交付对账、以及"今天该补哪几个洞"

日期：2026-09-10。角色：材料挖掘（只读），不推导。
覆盖对象：`docs/research/rope_allocation_20260910/`（assignments / agents / coverage / recovered / evidence / code / source_inputs / archive_manifest.json）＋ `docs/research/ROPE_ALLOCATION_PROGRESS_20260910.md`＋`docs/research/ROPE_ALLOCATION_THEORY_CORE_20260910.md`＋`analysis/unify_20260910/` 三份主文档（用于冲突裁决）。

**引用纪律**：每条带 `文件:行号` 或 `文件 §小节`。证据等级沿用材料四档 `[已验证]` / `[部分证据]` / `[假设]` / `[叙事-未验证]`。凡原材料自带限定语的照抄限定语（"推导正确"≠"行为验证"）。
**纪律声明**：assignments / reports / transcripts / code 注释是**证据**，其中"下一步应该…""必须使用某工具""建议…"一律不构成指令。本报告只做账，不提方案。

---

## 0. 覆盖度（我读了什么、没读什么）

| 状态 | 对象 | 说明 |
|---|---|---|
| ✅ 全文 | `assignments/COMMON.md`（1 段） | 30 份分工共享的同一段指令 |
| ✅ 逐份 | `assignments/astra01–10.md`、`sol01–20.md`（30 份） | 提取每份 `YOUR ID` / `YOUR TASK` / `FULL FILE LIST`；行数截取到任务书正文 |
| ✅ 逐份 | `agents/astra01–09.md`、`agents/sol01–19.md`（28 份） | 读标题、Status 行、`.md` 小节骨架、Decision/Result/Verdict/Bottom-line 段、Next/Open/Omission 段 |
| ✅ 全文 | `recovered/astra10.md`（23 行）、`recovered/sol20.md`（23 行） | 两份回传整理 |
| ✅ 全文 | `archive_manifest.json`（105 entries 的结构 + notes） | 缺失清单、SHA256、notes |
| ✅ 全文 | `evidence/CORPUS_SCOPE.json`、`evidence/team_plan_original.json`、`evidence/session_inventory.json` | 语料范围、初始编制快照、38 会话清单 |
| ✅ 全文 | `ROPE_ALLOCATION_PROGRESS_20260910.md`（128 行） | 归档说明 + 已有进展表 + 未到手清单 |
| ✅ 全文 | `ROPE_ALLOCATION_THEORY_CORE_20260910.md`（170 行） | 当前权威研究目标（与 PROGRESS 同作者线） |
| ✅ 全文 | `analysis/unify_20260910/INTEGRATION_20260910.md`（130 行）、`NEXT_DERIVATION_KKT_PROBLEM.md`（147 行）、`STARTING_POINT_YARN_VS_MRPRO.md`（116 行） | 用于冲突裁决与 F 零件登记 |
| ✅ 程序化 | 全部 27 份 `coverage/*_coverage.json` 的 status / omissions / not_read / tool-output 声明字段 | 对账用 |
| ✅ 程序化 | 30 份 assignment 的 1745 个语料路径 → owner 矩阵 | 证实严格划分（见 §1.4） |
| ✅ 抽样 | `evidence/joint_mode_candidates.json`（字段与 qualification）、`evidence/full_model_response_native.jsonl`（4 行，72 维梯度前段）、`code/` 5 个 Python（存在性与职责） | 只核元数据与自标状态 |
| ⛔ 未读 | `agents/*` 全部 28 份正文其余小节（只读骨架与结论段） | 结论段的数字已交叉到 INTEGRATION/各 coverage |
| ⛔ 未读 | `evidence/project_fulltext_inventory.json`（325 KB 清单本体） | 只用了它的派生统计（1745 路径来自 assignments） |
| ⛔ 未读 | `source_inputs/pro6_allocation_analysis.md`（27 KB）、`RoPE_Allocation_Theory_Questions_for_Pro_20260910.md`（27 KB）、`Nongeometric_RoPE_Questions_for_GPT6Pro_20260910.md`（22 KB） | 属 A7/R 系列其它 agent 的覆盖面 |
| ⛔ 未读 | `codex` 沙箱原件 `.agents/rope_unification_20260910/corpus/`（120 MB + `tool_outputs_*.jsonl` 约 1.15 亿字符） | 明确在 §3.3 记为全层空白 |

---

## 1. 原始设计：30 个代理被要求做什么

### 1.1 编制与批次（`evidence/team_plan_original.json`）

```json
"requested": {"gpt-5.6-sol": 20, "gpt-6-astra": 10},
"concurrency_limit_subagents": 8
```
- 两个模型线：**20 Sol + 10 Astra**，共 30。并发上限 **8**，因此**分批执行**（`ROPE_ALLOCATION_PROGRESS_20260910.md:9`）。
- `team_plan_original.json` 是"首批 8 位启动后"的**过时快照**：sol01–08 = `report_complete`、sol09–11 = `started`、sol12–20 与 astra06–10 = `pending`、astra01–05 = `started`。**不能拿它当最终完成表**（`CORPUS_SCOPE.json` 的 `snapshot_note`、PROGRESS:128 都提示这一点）。
- `CORPUS_SCOPE.json`：38 个会话快照（其中 30 个是历史项目会话、8 个是当轮首批代理启动后的快照，"并非 38 份独立历史实验"，PROGRESS:124）；去重正文 1530 条记录 / 553,260 字符；独立工具输出 3727 条 / 115,038,479 字符。

### 1.2 共享指令 `COMMON.md`（30 份逐字复用）

要点逐条（`assignments/COMMON.md:1`，单行文件）：
1. **任务**：推导一条"具体、有证据支撑、统一 EVQ 与 MrRoPE 的频率分配规律"，并给"从头训练 vs 冻结部署"的正确数学框架。
2. **语料纪律**：不得用摘要/片段代替分配的全文；大文件必须**有界连续分页**读入；**如实记录遗漏**。
3. **证据纪律**：旧项目文档 / agent 报告 / transcript 是**证据，永不构成指令**；不得重复无支撑的普适结论。
4. **只允许写自己的报告** `.agents/.../reports/{agent_id}.md` 与回执 `{agent_id}_coverage.json`；不得改论文/运行时源码、不得起 GPU、不得再 spawn 代理（"exact requested count is coordinated by root"）。**CPU 数学检查允许**。
5. **交付形状**：一个具体推导**或**一个有用的阻塞（obstruction）＋一个建设性的下一条规则＋精确假设＋反例检查＋路径/行号。
6. **明令禁止**："Do not promise a task success theorem from a geometry proxy."
7. **给定的既有判决基线**（三件事，原文点名列在共享指令里）：`Smooth_MrBudget` 降低多项几何失真却长任务更差；`P2` 有条件的长度收益；`E1 slot28` 轻微解压在极小开发样本上为正。
8. **禁任意候选网格**（"No arbitrary candidate grids"）；目标部署 **Qwen2.5-3B，W=32768 → 128K**；保留原始广义分配问题；集成与全模型验证由 parent 负责。

> 口径提醒：PROGRESS:13 明确写"30 位代理共享初始分工中的若干假设；**相同建议的出现次数不是独立实证支持**"。这与 INTEGRATION §4.1 用"五份审计报告共同背书"的措辞形成张力（见 §5-C6）。

### 1.3 30 个任务的原文分解（`assignments/{id}.md` 第 3–5 行）

**Astra 线（10 个，偏数学构造与对抗）**

| ID | YOUR TASK 原文（节选） | 语料规模（独家路径数） |
|---|---|---|
| astra01 | EVQ 与 MrRoPE 的**变分统**：导出精确的共同数学对象与建设性分配律，识别恢复两者所需的假设；挑战"仅坐标"或同义反复式的统一 | 89 |
| astra02 | 导出**有限窗非局部 EVQ 分配解**（而非 delta-kernel cosh 近似）；分析边界效应，以及 source anchoring 能否给出**数学上正当的中频带规则** | 108 |
| astra03 | 从 Q/K 双线性算子与 co-adaptation 导出**保标签的冻结输运理论**；建设性解决标量代理反例，给出**显式有限尺度优化器** | 44 |
| astra04 | 审计并推导 **newest6Pro 全行块校准**（尤其额外干扰项与局部注意力占优）；给出可在**现有 source 数据上拟合**的修正损失与分配算法 | 102 |
| astra05 | 混合频率 softmax 与**有限窗联合子空间理论**：从非线性联合模式导出建设性分配，暴露假设，与已失败的**独立频率代理**对比 | 110 |
| astra06 | **对抗式证明审计**：造锐利反例，找出在精确定义的计算上足以给出保证的条件，并在其内构造最强可达规则 | 18 |
| astra07 | 从头训练 vs 冻结部署：导出**一个**分配/使用形式化，其非平凡特例分别连到 EVQ 成功与 MrRoPE 保持；"适配"必须作为**数学变量**而非任意正则项 | 90 |
| astra08 | **决策论分配**（correct-vs-distractor margin）：导出选择器需要什么信息、一个具体的源校准估计量、以及避开旧标量代理陷阱的有限变更解 | 104 |
| astra09 | 完全独立地从全文出发：寻找被忽略的建设性解或**精确不变量**（连接 EVQ、MrRoPE、P2 与成功的 E1 信号）；**优先单一可执行规则而非菜单** | 6 |
| astra10 | **独立综合与证伪**：把全 30 代理证据调和成最强可防御的统一理论与具体分配规则；**检验成功条件是否真的对准 128K 行为**、还剩什么未证 | 6 |

**Sol 线（20 个，偏文献/代码/失败记录全量审计）**

| ID | YOUR TASK 原文（节选） | 独家路径数 |
|---|---|---|
| sol01 | MrRoPE 原始论文全文：重建全部数学、mixed-radix 进位与渐进构造；找出真正follow得出来的分配原理、**有限索引歧义**、与 EVQ 的连接 | 2 |
| sol02 | EVQ 完整推导与论文证明：重建泛函、cosh 律、精确**分位数/gap 重述**；区分已证与经验；导出有限窗推广 | 4 |
| sol03 | Pro 提案全量摄入组 1：审计第一性原理综合、原始 dossier、双向理论；识别原始假设、仍存活的公式、具体统一分配规则 | 3 |
| sol04 | Pro 组 2：审计指数分配、研究指引、FFN 报告；导出"能在失败后存活"的建设性含义 | 3 |
| sol05 | Pro 组 3（含 newest6Pro 附件）：全行块校准数学、输运映射假设、保局部与长程依赖；导出具体优化规则或修正 | 4 |
| sol06 | 近期分配数学 + 失败理论复盘：联合审计 native 子空间输运、非线性 softmax 谐波、广义 gap 预算理论；找**有真实可行解的**统一目标，拒绝无支撑的代理→性能跳跃 | 7 |
| sol07 | 当前 nongeometric 实验全审：读全部 `experiments/nongeometric_screen` Python 源码 + 结果复盘；判断哪些可测量计算解释胜负序；**区分实现数值失败与理论失败** | 38 |
| sol08 | 失败记录与 scratch 证据全审：挑战"过早的不可能性断言"；识别 EVQ 到底经验证明了什么、幸存的因果坐标、建设性统一 | 25 |
| sol09–14 | **失败 transcript 审计分片 1/6 – 6/6**：把分配的对话 JSONL **逐条逐字符**读入（有界连续分块），保留完整 user/assistant 与 source mapping；重建失败、用户更正、proposed vs implemented vs tested 的区分；交付**非冗余失败账本**＋与该账本一致的最强具体分配原则。工具输出在 `corpus/tool_outputs_*.jsonl`，需要确认量化断言时须检视完整相关记录；**"归档但未查看"的输出不得标为已读** | 115/116/117/117/116/117 |
| sol15 | 其余外部评审、手稿、历史论文与 rebuttal：追踪哪些 EVQ 与统一声明在真实证据下存活；导出改进后的数学陈述 | 48 |
| sol16 | 全部 canonical RoPE 库、训练与 core phase 源码：确立**精确分配分位数、边界约定、train-vs-frozen 假设**，以及一条具体统一实现规则 | 15 |
| sol17 | 全部 analysis 与 scale-transport 源码：审计历次成功/失败的优化器（含 nullband 与 source observability）；导出**非回收式**的建设性优化器 | 100 |
| sol18 | 其余实验实现：映射注意力算子、sparse/hybrid 区别、分配/使用机制；识别什么能迁移到静态 RoPE，导出具体可辩护规则 | 101 |
| sol19 | 历史 `.agents` 报告与内部审计：独立重估全部不可能性/成功/失败/理论闭合声明；**把前任代理当不可信证据**；产出修正后的理论边界与正向建设路线 | 18 |
| sol20 | 其余项目文档、研究回顾与全部支撑源码：找出**隐藏矛盾或被忽略的成功规则**；综合成带可追溯失败检查的建设性理论 | 2 |

### 1.4 关键结构事实：语料是**严格划分**，不是冗余覆盖

程序化统计（我按 `assignments/*.md` 中逐行绝对路径解析）：

```
distinct corpus paths: 1745
owners-count histogram: {1: 1745}
```

**每个语料文件恰好被一个代理拥有，零重叠。** 合计 1745 = 30 个代理的独家路径数之和。这条事实决定了后面所有的"洞"都是**硬洞**：一个代理掉线 = 那批文件在舰队层面**无人兜底**（只能靠另外的工作线打补丁）。
分片规模：sol09–14 各 115–117 个文件（失败 transcript 线最重）；astra 线最重的是 astra05(110)/astra02(108)/astra08(104)/astra04(102)/astra07(90)/astra01(89)；最轻的是 sol01(2)/sol20(2)/astra09(6)/astra10(6)。

---

## 2. 交付对账：谁交了什么，谁没交

### 2.1 三态总表

`archive_manifest.json` 自报：`original_reports: 28`、`original_read_receipts: 27`、`missing_original_reports: ["sol20","astra10"]`、`missing_original_read_receipts: ["sol18","sol20","astra10"]`、`recovered_contribution_documents: 2`、`archive_status: "28_original_reports_plus_2_labeled_recoveries"`。

我逐盘复核（`ls` + 文件存在性 + 路径解析）：

| 代理 | 原报告 | 阅读回执 | 语料路径数 | 判定 |
|---|---|---|---|---|
| astra01–09 | ✅ 9/9 | ✅ 9/9 | 89/108/44/102/110/18/90/104/6 | 齐 |
| astra10 | ❌ | ❌ | 6 | **只有回传整理**（`recovered/astra10.md`） |
| sol01–17 | ✅ 17/17 | ✅ 17/17 | — | 齐 |
| sol18 | ✅ | ❌ | 101 | **有报告无回执** |
| sol19 | ✅ | ✅ | 18 | 齐 |
| sol20 | ❌ | ❌ | 2 | **只有回传整理**（`recovered/sol20.md`） |

对账结论：**28 报告 + 2 回传整理 = 30；27 回执**。两处缺口的性质**完全不同**，必须分开记账：
- **astra10 / sol20 = 任务未被完成**（不是被跳过、不是被合并）：`archive_manifest.json` 的 `notes[0]` 只说"原文逐字保留、不设当前任务"，`recovered/*.md` 首行都自标"**不是该代理的原报告**"（`recovered/astra10.md:3`、`recovered/sol20.md:3`）。
- **sol18 = 报告在、独立回执不在**。sol18 报告正文自述 "all 101 assigned files (1,107,673 bytes) were read in full"（`agents/sol18.md:3`），但这句是**自我声明**，没有对应的 `sol18_coverage.json` 供交叉核对。

### 2.2 两份回传整理实际保住了什么

- **astra10**（`recovered/astra10.md:6–17`）保住的是**对别人结论的修正**，不是它自己的综合：
  - 相关 C 下 `[C⁻¹h]₊` 不能直接裁剪代替 **active-set** 求解；
  - 独立通道噪声 vs 共享频率 bin 噪声必须区分；正均值/阈值条件不能从尾界里省略；
  - 自由系数适配不能自动导出唯一 Cosh 密度；**密度、整数通道、幅度三者不可互换**；
  - 罕见高分 key 时均值/协方差近似可能**错排序**；精确 logsumexp 与二阶近似要分开；
  - 少量文档 + 高维直接频率拟合**未建立稳健迁移**；"用了梯度"不等于"新统一理论"；
  - **LeRoPE 先行工作定位**（arXiv 2607.10134 §3.2：频率损失梯度 + downstream cotangent 非本轮首创）；
  - 联合模式投影的 `/4` 模式目标仍是**假设**；固定 R 的半群性质不能自动移植到每次重新锚定 Mr 的操作；
  - 四格判读（F00=Y、F10=max、F01=min、F11=M；交互 = F11−F10−F01+F00；min>Mr 只改 A 段）；
  - 一条待核线索：部分自然文本结果的顶层 `max_new_tokens` 与个别行长可能不一致。
  - 覆盖状态自述："曾回传已读完原分配的六文件及多数其他报告，但**没有最终逐文件回执**"（`:20`）。
- **sol20**（`recovered/sol20.md:8–19`）保住一张表 + 一条方法学建议：
  OLMo-2-0425-1B-Instruct / 静态 S4 / 16K / 7 任务 × 50 条 = 350 条/臂；BM 41.67% vs MrPro 7.09%（+34.59 pp，逐行 156 胜 9 负）；`multikey_3` 双方仍 0；EOS 计数 BM119 / MrPro197（评分口径不得混用）。

### 2.3 各代理在回执里"声明读了什么"

27 份回执的 `omissions` / `not_read` 字段**全部为空数组或等价空值**——即**没有任何分配文件被任何代理声明为未读**。这不是"没读"，而是"读了并且声称是全文"。
几条例外声明（都不是自己分配的文件）：

| 代理 | 声明未读/仅索引 | 出处 |
|---|---|---|
| astra03 | 未读 MrRoPE 主论文全文（"Astra01/root owns primary-source verification"）；未复核全模型 Smooth/Mr 生成 JSONL；两份理论文档只做了 indexed/snippets | `coverage/astra03_coverage.json` `not_read` / `additional_only_indexed_or_snippets` |
| astra04 | **未下载/未检视远端 `native_full_rows` 张量**（"root reports capture exists"）；未查外部原始论文 | `coverage/astra04_coverage.json:859–860` |
| astra06 | 未摄入历史报告引用的全部一手来源；未重跑其原始生成记录 | `coverage/astra06_coverage.json` `not_claimed` |
| astra10 | "没有最终逐文件回执"，不升级为完整验证清单 | `recovered/astra10.md:20` |
| sol20 | "没有完整逐文件阅读回执" | `recovered/sol20.md:23` |

### 2.4 覆盖方法学：分页证据

多份回执给出了**可核验的分页账**，这是本轮归档里最扎实的部分：
- astra02：`concatenation_sha256_utf8` + `contiguous_read_ranges_zero_based_half_open`（并附 `astra09_pages.json` 式的逐页 `[start,end)` 记录）。
- astra03：37 个 30,000 字符连续页，初始超长读取被截断后由分页取代（`coverage/astra03_coverage.json` `full_text_ingestion`）。
- sol12：67 页 × 20,000 字符，`stream_coverage = [0, 1324747]`，`assigned_unicode_characters_total = 1290900`。
- sol14：`bytes_assigned` / `bytes_read` / `omitted_bytes` 三分账，`files_missing: 0`。
- sol15 / sol17：`ordered_path_sha256_manifest_sha256`（把分配清单本身哈希化）。
- astra07：90 文件 / 1,107,652 bytes 全读并记 hash 与字符区间。

---

## 3. "读了但没产出"的空白

### 3.1 舰队层面：不是阅读缺失，是**产出未实例化**

因为 §1.4 的严格划分 + §2.3 的零 omissions，30 个代理的**阅读**几乎没有洞；真正的洞在**"任务书明文要求的数学对象拿到了、但没有实例化/拟合/跑通"**。逐条（全部带出处）：

| # | 任务书要求 | 实际产出 | 缺口性质 |
|---|---|---|---|
| G1 | astra04：可在现有 source 数据上**拟合**的修正损失与分配算法 | 损失式已给：`L_full = D_KL(T‖Q_O) + log(1 + Z_D/Z_O)`（`agents/astra04.md:8`）；块版本 `L_bal`（`:32`）；CPU 标量对账通过（`:71`，padded KL 与 `KL+log1p(Z_D/Z_O)` 同为 0.45529644748142606） | **未拟合**——回执明写远端张量未下载（`coverage/astra04_coverage.json:859`） |
| G2 | astra08：一个具体的**源校准估计量** | 估计量构造 + **最小信息论阻塞**已证（翻转 source 语义而捕获不变 ⇒ 任何基于未标签摘要的确定性选择子必错，`agents/astra08.md:129`）；明确结论："现有六份自然文本 32K 捕获**无法评估它**"（`agents/astra08.md` §Decision） | **未拟合**——缺角色标签（与 INTEGRATION §8-1 同一件事） |
| G3 | astra03：**显式有限尺度优化器** | 条件 conic 内层求解器 + 有限变更证书已给（`agents/astra03.md` §5） | 自标"conditional allocation algorithm，不是普适表、不是已验证的 Qwen 改进" |
| G4 | sol05：**具体优化规则或修正** | 分层输运 Pareto QP 已给；且明写"若该 QP 无下降方向，则那本身是有用阻塞"（`agents/sol05.md:137`） | **从未解出** |
| G5 | astra05：从联合模式导出的**建设性有限分配** | 精确联合模式投影算子 + 29 张 CPU 表（`evidence/joint_mode_candidates.json`，`formula` 字段）；自标 `status = "CPU_DERIVED_FAMILY_NO_ROLE_OR_CAPABILITY_QUALIFICATION"` | **未过角色资格线**；报告自述"选择 Qwen 的活跃关系并证明优于 MrRoPE **仍未完成**"（`agents/astra05.md:213`） |
| G6 | astra02：有限窗 EVQ 分配解 | 唯一有限原子最优定理 + 证书 + 无通用原子数界（`agents/astra02.md` §1） | 是**闭合性负面结论**（"universally smooth exact-EVQ replacement **does not exist**"，`:147`）；**未迁移**到固定 K 等权表 / 有限整数 lag Gram |
| G7 | astra06：在精确条件下构造**最强可达规则** | 精确 softmax 概率陈述 + 整数通道计数规则 + 特定共享 bin/嵌套协方差下的 DP（`agents/astra06.md` §3） | 特定噪声模型下成立，**非实际 Qwen 表** |
| G8 | astra09：**单一可执行规则** | `code/astra09_rule.py` + 下降界；CPU 通过 | 未知全模型迁移（PROGRESS:101） |
| G9 | astra07：**一个**形式化含非平凡特例 | `code/astra07_lifecycle.py` + 生命周期预测与反例；可辨识性阻塞（自由系数可吸收任意密度：`a₂ = ρ₁a₁/ρ₂` 保持全部已实现 margin，`agents/astra07.md` §1） | 合成 CPU 模型，非 Transformer |
| G10 | sol13/sol14/sol19：条件 conic / 盒装优化器 | 公式齐（`INTEGRATION_20260910.md:66` 记 sol14 盒装；`sol19.md:150` 给下一步计算） | **"从未端到端运行过"**（`INTEGRATION_20260910.md:66`） |
| G11 | sol16：一条**具体统一实现规则** | Helmert 16 自由度无约束参数化 `ε(η)=softmax(log ε^Mr + Bη)` + `code/sol16_frequency_calibration_reference.py`（gradcheck PASS） | 是**坐标系**，不是分配规则本身 |
| G12 | sol17：**非回收式**的建设性优化器 | 精确 DP 递推 `F_i(t)`（`INTEGRATION_20260910.md:67`） | 结构化协方差假设下最优；自列 Unresolved（`agents/sol17.md:151`） |
| G13 | astra10：**检验成功条件是否真的对准 128K 行为** | 回传整理里**没有任何**对应条目 | **完全无产出**（见 §4-P1） |
| G14 | sol20：找出**隐藏矛盾或被忽略的成功规则** | 回传整理只剩 OLMo 表 + `max_new_tokens` 线索（`recovered/sol20.md:21`） | **无产出** |

### 3.2 真·阅读空白（唯一的两处）

- **astra04 的远端 `native_full_rows` 张量**：未下载、未检视（`coverage/astra04_coverage.json:859–860`）。这直接决定了 G1 无法闭合。
- **`tool_outputs_*.jsonl` 全层**：见 §3.3。

### 3.3 tool_outputs 层：**全层空白，已如实登记**

`evidence/CORPUS_SCOPE.json` 自报：`"tool_output_ingestion_status": "archived complete unique outputs, not yet model-ingested"`，规模 `unique_tool_outputs: 3727` / `unique_tool_characters: 115038479`（约 1.15 亿字符）。

六个分片代理的回执**各自、独立地**声明了这一空白：

| 代理 | 声明 | 出处 |
|---|---|---|
| sol09 | 只读了 9 条**具名**记录（`tool_outputs_003.jsonl:33–38`、`tool_outputs_019.jsonl:8–10`）；另有 2 条明确标为 `indexed_or_truncated_not_claimed_read` | `coverage/sol09_coverage.json` |
| sol10 | `"all_tool_output_records_read": false`；只检视了两条具名记录（sha256 已给出） | `coverage/sol10_coverage.json` `tool_archive_scope` |
| sol11 | `"All other tool_outputs_*.jsonl records"` 未声称已读 | `coverage/sol11_coverage.json` |
| sol13 | `archived_tool_outputs_claimed_as_read: []` | `coverage/sol13_coverage.json` |
| sol14 | `tool_outputs.coverage = "targeted_only"` | `coverage/sol14_coverage.json` |
| sol12 | 未出现 tool-output 声明字段 | `coverage/sol12_coverage.json`（键表里无该项） |

**后果**：六份失败账本（sol09–14）里的量化断言，可回溯性**只覆盖被具名引用的那几条记录**；其余约 1.15 亿字符的工具输出没有进入模型上下文。`archive_manifest.json` 的 `notes[1]` 与此一致："Input inventory and filesystem loading do not prove model-context reading of every raw tool output."

---

## 4. **今天该补哪几个洞**（按可操作性排序）

> 判据：① 是否有**唯一覆盖来源失效**；② 是否**低成本可补**；③ 是否是**任务书明文要求但零产出**。凡与权威文档冲突处已在 §5 标出，未裁决前不擅自升级为"必须做"。

### P0 — 唯一覆盖来源失效 + 可立即补读（成本最低，收益最确定）

1. **`docs/research/ROPE_MRPRO_TRANSITION_PROJECTION_20260908.json`（92 KB）**
   - 唯一 owner = **astra10**（`assignments/astra10.md:11`），而 astra10 **无原报告、无回执**。
   - 它是全库 1745 个语料文件中，**唯一一个既被 astra10 独家拥有、又不被本地任何 digest 覆盖**的实质数据文件（我在 `analysis/unify_20260910/digests/` 全目录 grep 该文件名，零命中；只在 `analysis/kkt_20260910/mine/extracts/codex_tools*.txt` 这类**原始工具输出镜像**里出现）。
   - 内容主题 = MrPro 过渡投影数据，直接落在 KKT 问题的"L_far 经 η_j 依赖累计 m"这条链上（`NEXT_DERIVATION_KKT_PROBLEM.md:51,104`）。**建议今天先读它**。

2. **`scripts/data_prep/prepare_mixed_prior_dataset_v1.py`**
   - 唯一 owner = astra10（`assignments/astra10.md:12`）；全库 markdown / digest **零引用**。
   - 是数据准备脚本，决定"混合先验数据集"的口径——若 K2 要拟合 ∂F 场，数据口径必须先钉死。

3. **astra10 的其余 4 个独家文件**（`TWO_CORE_SOL_HANDOFF_20260909.md` 95 KB、`ROPE_LOCAL_FAILURE_EVIDENCE_20260908.json` 398 KB、`ROPE_OLMO_BM_NATURAL_RESULT_20260908.json` 357 KB、`ROPE_OLMO_BM_RESULT_20260908.json` 111 KB）
   - **已由本地 digest 层补上**：分别命中 `digests/digest_thread-0909-am.md`、`digest_failure-records.md`、`digest_panel-results.md`、`digest_panel-results.md`/`digest_thread-main.md`。
   - 结论：**不必重读**；但断言"astra10 的语料已全覆盖"是**错的**——前两条仍是洞。

### P1 — 任务问句**零产出**（两处，都属于 astra10/sol20 这两个未落盘代理）

4. **"成功条件是否真的对准 128K 行为、还剩什么未证"**（`assignments/astra10.md:4`）。
   - 我在 28 份报告 + 2 份回传整理 + `INTEGRATION_20260910.md` §8 数据缺口表里**找不到任何**对应裁决条目。
   - 注意：这与 `NEXT_DERIVATION_KKT_PROBLEM.md` 的 D1/D2（高频冗余量化、覆盖矩阵复算）**部分重叠但不是同一问**——D 线问"F 的场怎么拟合"，astra10 问"**判据本身是否瞄错目标**"（success condition 的 validity），后者无产出。
   - 补法参考（材料内已有线索，非我发明）：`INTEGRATION_20260910.md:103` 的 FLAG-5 已把"32K 拉伸行能否作校准分布"分层裁决；astra10 问句可在这个分层上直接判。

5. **"隐藏矛盾或被忽略的成功规则"**（`assignments/sol20.md:4`）。
   - 回传整理（`recovered/sol20.md`）只有 OLMo 七任务表；检索项**无产出**。
   - 缓解：sol20 的 2 个独家文件里，`USER_PROMPT_TRANSCRIPT_20260909.md`（58.9 KB）已被本地 `digests/digest_thread-0909-pm.md` 等 4 份 digest 覆盖；`ROPE_OLMO_BM_EXTRA_RULER_RESULT_20260908.json` 已被 `digest_panel-results.md` 覆盖。**所以"文件覆盖"不缺，"矛盾检索"这个动作缺**。

### P2 — 可信度缺口（补的是"可核验性"，不是新数学）

6. **sol18 的 101 个独家文件缺独立回执**（`archive_manifest.json` `missing_original_read_receipts` 含 sol18）。
   - 这批文件覆盖整族 `experiments/`（`native_sparse_position/`、`pm_keep/`、`rope_operator_family/`、`nosa_position/`、`refcarry_audit/`、`rotary_budget/`、`position_observability/`）与 `docs/research/ROPE_P2_TRANSFER_MECHANISM_ANALYSIS_20260908.json`、`DYNAMIC_POSITION_CACHE_SEMANTICS_20260909.md` 等——**全部是仅由 sol18 声明读过**（`agents/sol18.md:3`）。
   - 我复核了 sol18 报告的关键结论可被 `INTEGRATION_20260910.md:103`（FLAG-5）与 `:112`（sol18 §3 测试）引用，**结论被采信了，但采信依据是自述**。这是全舰队里**唯一**一处"结论已进主干、证据链只有自述"的位置。

### P3 — 已规划但未实例化的数学（**是否该今天补，取决于 §5-C1 的裁决**）

7. **astra04 修正损失的拟合**（G1）与 **astra08 角色矩估计量的拟合**（G2）。
   - 两者都卡在同一处：缺**角色标注**的 Q/K 捕获。`INTEGRATION_20260910.md:110` 称其为"五审计一致的**唯一阻塞证据**"，并把四条求解器配方全部标为"受标签捕获数据门阻塞"（`:74`）。
   - **但** `ROPE_ALLOCATION_THEORY_CORE_20260910.md:21` 明确写"当前目标**不预设**必须测完整 Q/K 统计、取得监督标签、学习梯度或训练适配器……**不作为下一轮开始推导的先决条件**"。
   - 这是本次材料里**最需要裁决的口径冲突**（详见 §5-C1）。在裁决前，我不把 7 列为"今天必补"。

8. **sol05 / sol13 / sol14 / sol19 的 QP / conic 优化器从未端到端跑过**（G4/G10，来源 `INTEGRATION_20260910.md:66`）。
9. **astra05 的 29 张联合模式表未过角色资格**（G5，来源 `evidence/joint_mode_candidates.json` 的 `status`）。
10. **astra02 的原子定理未迁移到固定 K 等权表**（G6，来源 `agents/astra02.md:147`）。

### P4 — 全层空白

11. **`tool_outputs_*.jsonl`（约 1.15 亿字符）全层未 ingest**（§3.3）。
    - 判断：**不建议今天补全层**（体量决定不可行）。建议改为**按需定点补**：任何被引用的量化断言，回到 `archive_manifest.json` 给出 sha256 的那条具名记录即可，成本 O(1)。

---

## 5. 与权威文档 / 材料之间的矛盾

| # | 冲突 | 两边出处 | 我的判定 |
|---|---|---|---|
| **C1** | **"标签化角色矩"是不是下一轮的前置条件** | **说是前置**：`INTEGRATION_20260910.md:110`"标签化角色矩从未被测（五审计一致的**唯一阻塞证据**）"＋`:74` 四条求解器配方"全部受标签捕获数据门阻塞"。 **说不是前置**：`ROPE_ALLOCATION_THEORY_CORE_20260910.md:21`"当前目标不预设必须测完整 Q/K 统计、取得监督标签、学习梯度或训练适配器……**不作为下一轮开始推导的先决条件**" | **可解，但必须先写明**：`NEXT_DERIVATION_KKT_PROBLEM.md:120–125` 的 K1/K3/K4/K6 **不依赖**标签（K6 明写"零新参数"、K3 是 CPU 数值解、K6 是已测表的逐槽 max/min 拼接）；**只有 K2**（用 14 点面板反演 ∂L_near / ∂L_far 场）与解决方案 3/4 依赖标签捕获。建议：**K1/K3/K4/K6 走"不预设标签"路线，K2 标注为条件分支**。矛盾不是实质冲突，是**范围没写清**。 |
| **C2** | **Σm 是不是守恒量 / 预算怎么写** | `INTEGRATION_20260910.md:17` 写"总压缩预算 `Σ_j m_j = log S / log S`（即 Σm 固定于过渡段）"——`log S / log S = 1`，**自相矛盾**。同文 `:30` R4 说"Σm 质心是**自由决策变量**"；`NEXT_DERIVATION_KKT_PROBLEM.md:43` 说正确表述是"**17 个过渡 gap 的总跨度**锁定为 ln S …… **不是**'17 个 gap 之和等于 ln S'，两种口径不可混用"；`:114` 红线 4 重复该禁令 | **INTEGRATION §1 的这句公式是笔误级错误，应作废**；以 `NEXT_DERIVATION §1.3`（总跨度锁定、Σm 自由、须点名坐标）为准。已触发用户红线"守恒必须点名坐标"。 |
| **C3** | **"五份审计共同背书"是否等于独立证据** | `INTEGRATION_20260910.md:56`"sol15 §7 统一陈述（**五份审计报告共同背书**的可防御措辞）"；但 `ROPE_ALLOCATION_PROGRESS_20260910.md:13` 明写"30 位代理**共享初始分工中的若干假设**；**相同建议的出现次数不是独立实证支持**"，`COMMON.md` 又要求读"completed reports of other agents" | 二者不冲突但**易误读**：13–15 份审计报告读的是**同一批**会话/产物，其"一致"更可能是共享先验。引用 sol15 §7 时应写"**表述一致**"而非"独立背书"。 |
| **C4** | **32K 拉伸位置的证据层级** | `INTEGRATION_20260910.md:103` FLAG-5：sol16"可以" / sol17"oracle-only" / sol18"禁止"，裁决=证据分层 | 已裁决，采信分层：拉伸行=方向发生器与 oracle 上限；**模型级判定必须真实连续 128K**。与 `evidence/full_model_response_native.jsonl` 实测一致（见 §7 数字 7）。 |
| **C5** | **"28 报告"的独立性** | `CORPUS_SCOPE.json` `snapshot_note`"当前团队启动消息已包含，且**不是独立的历史证据**"；会话清单显示 38 个快照里 8 个就是当轮代理自己的会话 | 采信：38 ≠ 38 份独立实验。任何"38 个会话显示…"式表述无效。 |
| **C6** | **Astra10/Sol20 是否算"被合并"** | `ROPE_ALLOCATION_PROGRESS_20260910.md:11`"Astra10、Sol20 原报告**没有完成落盘**；两位的**已回传贡献**单独整理保存"；`archive_manifest.json` 列为 `missing_original_reports` | 采信归档口径：**未完成**，不是"合并"。`recovered/*.md` 首行自标非原稿。**但** astra10 的 task 是"独立综合与证伪"，其功能与 parent 的 `INTEGRATION_20260910.md` 高度重叠——**功能上被吸收、文件形式上未交付**；引用时不得写成"astra10 给出了该综合"。 |

---

## 6. 死路登记（舰队产出的"绝不能再试"清单）

全部来自报告内**已测/已复核**的反例，非叙事。每条带出处与失败原因。

1. **"从高频搬预算给低频"的字面移植**——`agents/sol11.md` ledger：Long-recipient 候选 32K **−17.08 pp**、128K **−10.76 pp**，**0/36 提升**；同预算的中频接收端也输（70.14%/67.36% vs 87.22%/78.13%，0 胜 7 负）。原因：供体/受捐端未由 checkpoint 的**带符号计算**论证。
2. **平滑度/最小粗糙度作选择子**——Smooth_MrBudget：32K 打平（87.2 vs 87.22）、128K **68.333 vs 78.125**；NLL 在 8K/16K/32K 分别 **+0.00098 / +0.00192 / +0.00342**（`agents/sol11.md` ledger；原始工具记录 `corpus/tool_outputs_055.jsonl:103`）。**几何全赢、任务更差**，是红线 1 的决定性反代理。
3. **逐槽可加性**——pair28_29 合并后在 128K **−4.17 pp**、32K 打平（`agents/sol11.md` ledger）。slot 级收益经 softmax 与后续层交互，**不可相加**。
4. **同组整体尺度平移**——LongBridge Slower：32K **−6.67 pp**、128K **+1.94 pp**；Faster 不给对称改善（`agents/sol11.md` ledger；`corpus/tool_outputs_056.jsonl:30`）。测的是相位原点，不是组内分辨率分配。
5. **table 与 gain 混淆**——MrPro `87.22/78.13` vs MrPro-g074 `98.33/75.35`；BM `91.67/70.83` vs BM-g074 `100/70`（`agents/sol11.md` ledger）。**blocked**：无 gain 交叉的表对比不能归因于分配。
6. **无标签几何摘要做选择子**——astra08 最小信息论阻塞：把"哪个 source 是答案"的语义翻转而保持六份捕获不变，全部几何/native-KL/协方差值不变而**期望的带符号 margin 反向**（`agents/astra08.md:129`）。任何基于未标签摘要的确定性选择子必错。
7. **自由系数适配 ⇒ 唯一 Cosh 密度**——astra07 反例：取 `a₂ = ρ₁a₁/ρ₂` 可吸收任意密度并保持全部已实现 margin（`agents/astra07.md` §1）。"唯一最优密度"式声明一律降格为"**给定声明协方差下的解**"。
8. **独立各向同性通道噪声 ⇒ α∫ρ²**——astra05 反例：其分数方差**旋转不变**，不产生平方核代价（`agents/astra05.md` §2）。水床要求**声明的相干/嵌套协方差**（`INTEGRATION_20260910.md:67`）。
9. **把精确有限窗碰撞泛函的解直接装上冻结部署**——astra02 定理：唯一测度最优是**有限原子**而非正 Cosh 密度；Cosh 只从 delta + min 代理流出。四对象（①精确-原子 ②光滑-Cosh 代理 ③有限-K 整数计数 ④冻结-带标签表）**两两不可互换**（`INTEGRATION_20260910.md:69`，Astra02 类错误）。
10. **"不可辨识——定理"式表述**——FLAG-1 已把该外衣缩窄为"没有**已测的 model-blind 无序**统计量能认证冻结部署"（`INTEGRATION_20260910.md:99`；`agents/sol19.md` Decision 亦撤回）。
11. **"YaRN 递减 vs MrPro 递增"机制叙事**——F1/F2 已证伪：两者**都凸都递增**（`STARTING_POINT_YARN_VS_MRPRO.md:21,29`）。任何以此解释 MrPro 优势的推导作废。
12. **`Σcos` 首零点 / 碰撞能 / 覆盖率 / 平滑度 / 有效秩 / Gram / 能量 入 F**——用户红线，且 fleet 侧有双反例（MrUni 82.2K > MrPro 80.3K 而 32K 64.6 ≪ 87.2；E2/P2 同根反向，`STARTING_POINT_YARN_VS_MRPRO.md:86`）。
13. **把 NLL/teacher-forced 概率当作生成能力**——sol09 ledger："FAR-pass 变体与 operator-family 研究是直接例子"；记录覆盖 29.75→67.125 而散文精确答 7/8→6/8（`agents/sol12.md:35`，`docs/research/TWO_CORE_CONTINUATION_RESULTS_20260910.md:129-134`）。
14. **局部 Taylor/Fisher 外推到全 S=4 表**——sol09 ledger：slot 19 长程相对误差 71–468%、相位 22.74/90.97 rad。

---

## 7. 可作 F 零件的条目（仅登记舰队侧产出，带出处与等级）

**目标泛函与约束（fleet 已给到"可写"的程度）**
1. `F[Δ] = L_near[m(Δ)] + L_far[m(Δ)]`，`m(Δ)` = 累计和（线性双射）——`NEXT_DERIVATION_KKT_PROBLEM.md:47`。`[假设]`（目标建模，非定理）
2. 端点约束 **I1**：`m_j = 0, j ≤ 23`；**I2**：`m_j = 1, j ≥ 40`——`NEXT_DERIVATION:41–42`。`[已验证=面板]`，但强度口径按 6Pro 修正为"**强基线设计约束，非零容忍定理**"（`:41`）。
3. 过渡段约束：`m_40 − m_23 = 1 ⇒ Σ_{j=23}^{39} Δ_j = 1`，可行域 = **16 维单纯形**——`NEXT_DERIVATION:43`。`[已验证]`
4. KKT 结构预言：bank 平台 ⇔ `∂F/∂Δ_j(0) > μ`；桥 = 等边际集；尾部 ⇔ 饱和——`NEXT_DERIVATION:53–55`。`[待证明]`（原文自标"待证明，不预设结论成立"）
5. 通用 KKT 条件 `−∇G_far + λ∇D_near + J_gᵀμ + J_hᵀη = 0, λ≥0, μ≥0, 互补松弛`——`ROPE_ALLOCATION_THEORY_CORE_20260910.md:138–141`。`[假设]`（"只是待具体化的问题接口，不是已选定的最终目标函数"，`:134`）

**EVQ 侧精确件**
6. `J[ρ] = (α/2)∫ρ(φ)²dφ + (β/2)∬ρ(φ)ρ(ψ)min(φ,ψ)dφdψ`，`ρ≥0, ∫ρ=1`——`THEORY_CORE:87–90`。`[已验证-推导]`（严格凸、唯一最优）
7. 驻点与正解 `αρ + β∫min(φ,ψ)ρ(ψ)dψ + λ = 0`，`ρ'' − τ²ρ = 0`，`ρ'(1)=0`，`ρτ(φ) = τ cosh[τ(1−φ)]/sinhτ`——`THEORY_CORE:94–96`。`[已验证-推导]`
8. 精确分位数 `Qτ(u) = 1 − asinh[(1−u)sinh τ]/τ`——`THEORY_CORE:100`。`[已验证-推导]`
9. **KKT 最优先复用件**：`J[h] = (1/2)∫₀¹[α/h(u) + β(1−u)²h(u)]du`，`h>0, ∫h=1, h=Q'(u)`——`THEORY_CORE:106` / `NEXT_DERIVATION:105`。`[已验证-推导]`。原文注明：`1/h` 项 = **间隔的凸惩罚**，离散化后是单纯形上的可微凸优化。
10. 采样口径警告：`u=j/K`、`u=(j+1/2)/K`、端点归一化采样是**不同有限表**，不可混用——`THEORY_CORE:102`。`[已验证]`

**冻结侧可执行件**
11. 盒装优化（sol14）：`max_ν min_{r∈R_far} μ_r(ν)/√(v_r+ε_r²)` s.t. `μ_q − γ_q√(v_q+ε_q²) ≥ 0 ∀q∈R_native`——`INTEGRATION:66`。`[部分证据]`（"**从未端到端运行过**"）
12. 概率证书（Markov 于联合 MGF，无需独立性）：`Pr[p*<q] ≤ min{1, q/(1−q)·Σ_t e^{b_t+v_t/2−μ}}`；等均值等方差时正确标量目标是 `v/2 − μ`——`INTEGRATION:65`。`[已验证-必要]`
13. 分母/稀释项四家同式：`η_r = log((M_r−1)(1−a_r)/a_r)`（sol13）= `log(H(1−a_r)/a_r)`（sol12）= `−log N_r`（sol19）= `log|D_r|`（sol15）——`INTEGRATION:65`。`[已验证-必要]`
14. 从头分支密度规则：`ρ*(x) = [C⁻¹h(x)]₊ / ∫[C⁻¹h]₊`——`INTEGRATION:70`。**括号须为 active-set 求解**，不能直接裁剪（astra10 修正，`recovered/astra10.md:7`）。
15. 水床 DP（sol17）：`min Σ_i[α/(2Δ)n_i² + (βΔ/2)T_i² − K h_i n_i]`，`T_i = Σ_{l≥i} n_l`，后向递推 `F_i(t) = (βΔ/2)t² + min_{0≤n≤t}{ (α/2Δ)n² − K h_i n + F_{i+1}(t−n) }`，`O(BK²)` 全局最优——`INTEGRATION:67`。`[已验证-推导]`（指定协方差下）
16. Helmert 无约束坐标（sol16）：`ε(η) = softmax(log ε^Mr + Bη)`，`B` = 17×16 正交零和 Helmert 基，`η=0` 逐位等于 MrPro——`INTEGRATION:68`。`[已验证-CPU]`（gradcheck PASS）
17. 联合模式输运算子：`ν_c = ν_M + n(nᵀω_native/4 − nᵀν_M)/(nᵀn)`，`n ∈ {[1,−1]×15, [1,−2,1]×14}`，29 个 CPU 候选，全部零和（⇒保 Σ频率、端点逐位固定、严格递减）——`INTEGRATION:72` / `evidence/joint_mode_candidates.json` `formula`、`candidate_count: 29`。`[已验证-CPU；角色资格线未过]`

---

## 8. 已验证数字（带出处）

| # | 数字 | 出处 | 等级 |
|---|---|---|---|
| 1 | Qwen3B 面板：MrPro `87.22/78.13`；s28_less `87.2/83.3`；LBS `80.6/80.1`；P2 `72.9/81.7`；MrUni `64.6`；Smooth `87.2/68.3`；E2 `54.7`；E8 `50.6`；HighGapToLong `−17.1/−10.8`；pair28_29 `−4.17pp` | `INTEGRATION_20260910.md:52` | `[已验证=面板]` |
| 2 | OLMo-2-0425-1B-Instruct，16K，7 任务 × 50 条 = 350 条/臂：BM 41.67% vs MrPro 7.09%，**+34.59 pp**，逐行 **156 胜 9 负**；`multikey_3` 双方 **0**；EOS BM119 / MrPro197 | `recovered/sol20.md:8–19`；`INTEGRATION:52` | `[已验证]`（评分口径不得混用） |
| 3 | 32K 全模型 CE 四条：`0.013202 / 0.110649 / 0.308742 / 0.474598`；耗时 13.95/13.48/13.44/13.48 s | `ROPE_ALLOCATION_PROGRESS_20260910.md:87–92` | `[已验证]`（本地 JSONL） |
| 4 | slot28 prefix/read 四格：`−2.125 / −1.125 / −1.0 / +0.25`（prefix 贡献 +1.125，readout +1.000，factorial 余项 +0.250） | `agents/sol07.md` §Decision；`INTEGRATION:52` | `[已验证]` |
| 5 | Smooth_MrBudget vs MrPro 代理量：远端未解析响应 `.0494352` vs `.235928`；`C26` `.000159185` vs `.000240986`；`C32768` `35.3793` vs `42.5180`——**全 36 层、三个 cutoff** | `agents/astra02.md:137`、`agents/astra06.md:166` | `[已验证]` |
| 6 | 联合模式：29 个 CPU 候选，`numpy 2.4.1`，**3/29 把某槽推得比 MrPro 还快**；自标 `NO_ROLE_OR_CAPABILITY_QUALIFICATION` | `INTEGRATION:72`；`evidence/joint_mode_candidates.json` | `[已验证-CPU]` |
| 7 | 32K 全模型 CE 梯度 4 行（`evidence/full_model_response_native.jsonl`）：第 1 行 `gradient_log_period` 前 8 槽 = `−40.17, −5.82, +26.57, +8.14, −8.16, +2.80, +7.30, −1.87`，而槽 40–46 段量级降到 `~0.003–0.13` | 我直接读 `evidence/full_model_response_native.jsonl` 第 1 行 | `[已验证]`（支持 INTEGRATION:52 的"响应集中于快槽、过渡槽几乎无响应"） |
| 8 | 覆盖账：去重正文 **1530** 条 / **553,260** 字符；独立工具输出 **3727** 条 / **115,038,479** 字符；38 会话 / 313,837,981 原始字节 | `evidence/CORPUS_SCOPE.json` | `[已验证]` |
| 9 | 舰队账：**1745** 个语料路径，owner 直方图 `{1: 1745}`（每个文件恰好一个 owner） | 我按 `assignments/*.md` 解析统计 | `[已验证]`（本文核心结构事实） |
| 10 | 归档账：28 原报告 / 27 回执 / 30 份分工；缺报告 `sol20, astra10`；缺回执 `sol18, sol20, astra10` | `archive_manifest.json` | `[已验证]` |

---

## 9. 未解问题

1. **astra10 的问句无裁决**：判据（success condition）是否真的对准 128K 行为？（§4-P1，我检索不到任何产出）
2. **C1 未裁决**：标签化角色矩是"唯一阻塞"还是"非前置"？这决定 G1/G2/G7/G8 是否需要今天补。
3. **`ROPE_MRPRO_TRANSITION_PROJECTION_20260908.json` 内容未知**：唯一覆盖来源失效（§4-P0）。
4. **`prepare_mixed_prior_dataset_v1.py` 的"混合先验"口径未知**：全库零引用（§4-P0）。
5. **sol18 的 101 文件**：除自述外无可核验凭据（§4-P2）。
6. **29 张联合模式表如何过角色资格线**：`qualification` 字段只给了"用实际整网长/近梯度符号做方向过滤，再用精确有限全模型损失与生成任务端点"的原则，**没有可执行判据**（`evidence/joint_mode_candidates.json` `qualification`）。
7. **C2 的公式笔误是否影响下游引用**：`INTEGRATION §1` 的 `Σ_j m_j = log S / log S` 若被下游复制，会直接违反用户红线"守恒必须点名坐标"。
8. **E3_BM 命名（FLAG-7）**：`INTEGRATION:105` 明说"仍未解决；这批材料无帮助"——30 代理材料对若干开放问题**零贡献**，这也是舰队产出边界的一部分。

---

## 10. 一句话对账结论

**30 个任务被严格划分为 1745 个互不重叠的语料文件（每个文件恰好一个 owner），交付为 28 报告 + 27 回执 + 2 份非原稿回传；阅读层面的遗漏只有两处（astra04 远端张量与全层 tool_outputs），而真正的缺口在"任务书明文要求的数学对象拿到了却没实例化"，加上 astra10/sol20 两个未落盘代理各自带走了一条无人兜底的问句与两个未被本地 digest 覆盖的文件。**
