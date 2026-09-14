# digest thread-0909-pm

覆盖 2026-09-09 晚（本地 EDT）至 2026-09-10 上午的两个 codex 子代理线程：F1 = `/root/current_manuscript_accept_review`（论文接收评审），F2 = `/root/parallel_plan_audit`（并行计划审计＋两份"给 Pro"研究附件的制作）。两线程同属主会话 `01a0806f-3df5-74b1-bc56-bf00d89d238e`（/root，模型 gpt-6-astra，context window 570,000）。

时间基准：本文所有 `…Z` 时间为 rollout 内 UTC 时间戳；本机 timezone 为 America/New_York（EDT = UTC−4，见 F2 `#5` environment_context 与文件名对照）。项目内文档（如 USER_PROMPT_TRANSCRIPT）使用 UTC 日历日，个别处与本地日相差一天，引用时以时间戳为准。

## 1. 来源清单

| 项 | 路径 | 大小/行数 | 说明 |
|---|---|---|---|
| F1 原始 rollout | `/Users/[REDACTED_AUTHOR].yanghejazfs.com.au/.codex/sessions/2026/09/09/rollout-2026-09-09T21-44-01-01a088fc-7675-7932-b30d-232cae01ee86.jsonl` | 1,790,828 B / 73 行 | 会话 id `01a088fc…`，parent=主会话 `01a0806f…`，`source=subagent`，`agent_path=/root/current_manuscript_accept_review`（旧版 raw 提取显示其 nickname=Euler）。仅 1 user + 1 assistant 明文消息 |
| F2 原始 rollout | `/Users/[REDACTED_AUTHOR].yanghejazfs.com.au/.codex/sessions/2026/09/10/rollout-2026-09-10T01-59-00-01a089e5-e681-75e1-819d-b3ffa379898c.jsonl` | 2,292,959 B / 653 行 | 会话 id `01a089e5…`，`forked_from=01a0806f…`，`agent_path=/root/parallel_plan_audit`；含 5 个 task_started、4 个 task_complete、1 条父会话 compacted 快照、6 条继承的 user 消息、4 条 assistant 明文消息、22 条 agent_message（正文加密） |
| 本地工作副本 | `analysis/unify_20260910/raw/rollout-0909-2144.jsonl`、`raw/rollout-0910-0159.jsonl` | 同上 | **访问事故记录**：提取期间对 `~/.codex` 的直接读取间歇性返回 ENOENT（同一命令内 `head` 成功、随后 `open()` 失败，8 次探测全失败，`find` 又可见文件）。最终用"find 定位＋立即 cp"于一次调用内原子完成复制（首试即成）。对 `~/.codex` 做批量提取时建议直接采用该原子模式 |
| 提取件 | `analysis/unify_20260910/raw_thread-0909-pm.txt` | 59,893 B / 657 行 | user/assistant 明文 + task 边界 + agent_message 框架行（正文以 `[ENCRYPTED_CONTENT nB]` 标注）+ exec 命令行（截断 400 字符） |
| 工具索引 | `analysis/unify_20260910/raw_thread-0909-pm_tools.txt` | 100,241 B | 全部 custom_tool_call 完整输入与 function_call 头部（send_message 仅记加密长度）；含两份 apply_patch 的完整新文件正文 |
| 产物 1 | `docs/research/PARALLEL_NONGEOMETRIC_20X10_AUDIT_20260910.md` | 17,596 B / 127 行 | F2 轮 1 创建（06:05:12Z apply_patch #118）；git 状态为未跟踪 |
| 产物 2 | `/Users/[REDACTED_AUTHOR].yanghejazfs.com.au/Desktop/Nongeometric_RoPE_Questions_for_GPT6Pro_20260910.md` | 22,427 B / 227 行 | F2 轮 2 创建（06:12:19Z）、轮 3 修 §4D（06:14Z）。**注意 mtime = 02:28 EDT（06:28Z），晚于本线程末次编辑 06:14Z，说明此后另有写者（父会话/兄弟线程）改动过当前文件内容；引用具体句子时以本 digest 内嵌版本或 git 版为准** |
| 产物 3 | `/Users/[REDACTED_AUTHOR].yanghejazfs.com.au/Desktop/RoPE_Allocation_Theory_Questions_for_Pro_20260910.md` | 26,703 B / 247 行 | F2 轮 4 定稿（文件头自述"证据更新至 2026-09-10 08:42 UTC"，mtime 04:43 EDT = 08:43Z，与 task_complete 一致） |
| 关联（非本线程） | `docs/research/USER_PROMPT_TRANSCRIPT_20260909.md`（T2-P28，2026-09-09T00:45:05.060Z："你的好实验报告提交并推送代码吧，别找了"）、`USER_INTENT_GUIDE_20260909.md:50` | — | 任务简报所引"提交并推送代码吧别找了"**不在本二文件任何明文记录中**（对两份 jsonl grep `提交并推送`/`别找了` 均 0 命中）；该指令记录于主会话 T2 线程转录文档，本线程活动属其"之后"的活动 |

**rate_limit 核查（任务简报疑点，如实记录）**：F2 文件 grep `rate_limit` 得 180 次、F1 得 20 次，但这是**同一 key 双计**：每条 `event_msg/token_count` 记录含 `"rate_limits"` 与 `"rate_limit_reached_type"` 两个字段名（F2 90 条 token_count × 2 = 180；F1 10 × 2 = 20）。逐条解析结果：两文件 `rate_limit_reached_type` **全部为 null**，weekly（window_minutes=10080）`used_percent` 从 F2 开始 30% 线性升至结束 39%（F1 恒 17%），credits 无、plan_type=pro。**结论：[已验证] 本二线程内未发生任何 API 级 rate-limit 拒绝或受阻重试；"180 次 rate_limit 疑似长时间受阻"为 grep 假阳性。** F2 真正的时间空洞是 06:14:25Z→08:06:34Z（112.1 分钟）的**空闲等待**：该区间两条记录之间无任何事件，线程未丢失进行中工作（轮 3 已完整交付；轮 4 的 NEW_TASK 08:06:34Z 到达后线程全部完成）。丢失/延迟的是**时间窗而非成果**；等待期间主会话在做什么本文件无证据。

## 2. 任务时间线

### 2.0 前置语境（非本二文件内容，仅作锚点，出处注明）
09_09 分支收尾：git log 有 `Workspace snapshot 2026-09-09 20:54/21:07/21:37/22:37` 快照提交（分支 09_09）；主会话 `raw_thread-main.txt:336-375` 记录 09-08 的同类指令（"我准备回去了……提交并推送代码吧"→"已提交并推送：分支 main_0726_09_06，最新提交 0177e6d"）。"提交并推送代码吧，别找了"精确出处为 `USER_PROMPT_TRANSCRIPT_20260909.md` T2-P28（2026-09-09T00:45Z），`USER_INTENT_GUIDE_20260909.md:50` 解读为"不能凭旧 goal 恢复那一夜的研究队列"的停止指令。本 digest 的 F1（21:44 EDT 09-09 本地）即在该指令之后约 25 小时。

### 2.1 F1（/root/current_manuscript_accept_review）——paper 接收评审 [成功，单轮 2 分 26 秒]
- **目标**（NEW_TASK 正文加密 2616 B，任务名可辨）：对 `/tmp/exponent-accept-review-20260910/submission.pdf` 做独立"是否接收"评审（评审契约来自 skill `review-conference-paper`）。
- **方案**：SHA256 校验 + PyMuPDF 全文提取（分段 sed 1–1100、1100–2700 行）+ 定向读页（14、21–27、28–37 页）+ 渲染 4 页图（p4/14/16/35）目检公式图表；全程只读，不访问代码/日志/历史/外部文献。
- **结果**：明文交付（assistant #69 = task_complete last_agent_message）——**"建议接收"**。依据 ICLR 2027 官方审稿准则、不另设数值分。核心优点 3 条（151.9M 固定端点配对训练归因；几何分析区分位置方向与能力；OLMo 自然 QA 631 长输入宏平均 F1 +3.82 点、配对区间 [1.32, 6.29]）。不足 3 条（中等级别）：①50.9M 因子实验仅 128 步重复 WikiText-2、Cosh vs Geo 区间含零、§C.3 split 隔离未说明；②理论只给描述/构造不给性能预测（式 6 慢尾惩罚是指定偏好非推导；§A.8 依赖局部假设）；③成熟模型证据需条件化（Llama 8K 退化、Qwen-7B 128K 每项仅 2 输入）。2 个可改判问题：§C.3 OOD split 与训练流重叠与否；§F.1 幅度系数 0.074 的选择模型/任务及确认集隔离。PDF 实际 37 页、SHA256 与指定一致。

### 2.2 F2 轮 1（05:59:10Z–06:06:05Z，约 7 分钟）——审计"20×10 非几何并行计划" [成功]
- **目标**：NEW_TASK（加密 2212 B）+ 继承的用户语境（§6 引用）——用子代理独立审计 `docs/research/PARALLEL_NONGEOMETRIC_20X10_PLAN_20260910.md`（qwen3.8-max 并行研究计划），结合实验实际。
- **方案**：读计划与 skill；`rg` 定位实现；SSH `connect.westc.seetacloud.com:27741` 读 `/root/autodl-tmp/nongeometric_screen_20260909`（development_summary.json、selection/E7_precision_breakdown.json、long_nll/E1_s28/s29 摘要、各方法 results/<method>/{summary.json,ruler.jsonl}、causal_cases/*.json）；CPU 重算平方弦距（从 results/E5_layer21/contract.json 的 MrPro 表与各候选 contract）；随后 apply_patch 写出审计文档（"No GPU jobs were launched or changed"）。
- **结果**：`PARALLEL_NONGEOMETRIC_20X10_AUDIT_20260910.md` + 汇报（#136）。关键纠正见 §3/§4；开发面板关键数字：MrPro 32K 87.222%/128K 78.125% 绝对；s28_less 128K→83.333%（+5.208pp，两条改善无退化）；s29_more 32K +8.333pp（全来自 1 条 QA）/128K −0.208pp；BM×gain074 32K +12.778/128K −8.125pp；E7 +2.778/−9.514；E8 −13.889（首 12 条）。保留项：扩大独立评测、table×gain 因子、shape×budget 分解、优先补 **MrPro×gain074** 一臂成 2×2。

### 2.3 F2 轮 2（06:06:53Z–06:13:08Z，约 6 分钟）——制作给 GPT-6 Pro 的自包含附件 [成功]
- **目标**：NEW_TASK（加密 2508 B）：单文件、不依赖仓库/transcript、可直接上传给 6 Pro。
- **方案**：查 BM/FullLagP2 协议文档与实现（ROPE_MRPRO_BM_PROTOCOL_20260908.md、audit_qwen_p2_full_lag.py 等）、远程交叉四格数据；apply_patch 写 `Desktop/Nongeometric_RoPE_Questions_for_GPT6Pro_20260910.md`（自称约 20KB）；python 校验 10 题编号连续、`\[`/`\]` 配对、含 '7.93668e−8'。
- **结果**：文件含精确公式（MrPro/BM/E1/P2 定义）、开发＋独立结果表、机制纠正（§4 A–D）、按优先级 10 个研究问题、交付要求；"未发送、未改 GPU 队列"。（当前磁盘版本 mtime 06:28Z 有线程外后续改动——见 §1。）

### 2.4 F2 轮 3（06:13:59Z–06:14:25Z，26 秒）——修 §4D 记号 [成功]
E9 距离核双时钟改用**小写 \(w=26\)**（来自校准数据 primitive 键值记录最大 token 跨度），与 row-wise 的模型窗口 \(W=32768\) 区分；仅改该附件并回读确认（#224）。

### 2.5 F2 空窗（06:14:25Z–08:06:34Z，112.1 分钟）
线程完全空闲（#227→#228 之间零记录）。非 API 受阻（见 §1 rate_limit 核查）。

### 2.6 F2 轮 4（08:06:34Z–08:43:26Z，约 37 分钟）——定稿第二份 Pro 附件 [成功]
- **目标**：NEW_TASK（加密 5004 B，最大的一次）：整合当局新证据制作 `Desktop/RoPE_Allocation_Theory_Questions_for_Pro_20260910.md`（"问题总结、理论分析与下一条规则"）。
- **过程**：读 `NONGEOMETRIC_MECHANISM_TRANSFER_20260910.md`、`ROPE_EXTRAPOLATION_FAILURE_AND_LIMITS_20260910.md`、`Downloads/Nongeometric_RoPE_Theory_and_Experiments_20260910.md`（qwen 侧分析）、`.codex/attachments/f295d4fc-c4c2-445a-801b-cf40dd9bf4f4/pasted-text.txt`（用户转来的外部分析）、本地 results（qwen7b/olmo1b transfer summary、reference_tables.json 的 Native/MrPro/MrProBM 逐槽周期、MrUni、LongBridgeSlower/Faster summary）；web 检索 YaRN(2309.00071v2) 与 2410.06205v1（"Round and Round We Go"）；期间收到 10+ 条加密 MESSAGE 增量指令（524–1272 B），每条触发一次文本替换修订；公式对齐修正（`(q_p^ν)ᵀ R_ν(t−p) k_t^ν`，#515）；恒等式 CPU 双向六距离核对误差 8.9e−16 写入 §7。
- **结果**：247 行定稿：6 个核心问题、LongBridge 两臂完整 36 条 + 48 文档长度单元 NLL、slot 周期事实表（23/28/29/36–40/51）、waterbed 约束推导、Smooth(MrBudget) 反例、冻结迁移与 P2 历史表、E7 精度四分解、MK2 margin 序列、QA cached 交叉失效、BM_ScaleTaper/FullLagP2_Transfer3B/G1-G3 标注"已排/进行中，无结果"。"已定稿……旧附件保留"（#649）。

## 3. 理论主张表

（主张 | 证据等级 | 出处[本线程内] | 后续是否被纠正/推翻）

| # | 主张 | 证据等级 | 出处 | 纠正状态 |
|---|---|---|---|---|
| C1 | E7 的约 161 倍 NMSE 差异发生在 BF16 实现层，不是"一阶理论低估 160 倍"：linear 7.9366844e−8、理想有限改动 7.9364028e−8、FP32 7.9325473e−8、BF16 1.2807392e−5 | [已验证]（同冻结状态/局部支持四路对照，远程 selection/E7_precision_breakdown.json + 本地重算） | AUDIT §Corrections.1；F2#136；GPT6Pro 附件 §4B；RoPE_Allocation §6 | 纠正的是 qwen 计划的表述；三命题（导数/实现/下游）分离的框架被两附件沿用 |
| C2 | gain 只乘 Q/K 幅度（logits ×g²），不改频率/相位跨度；BM×gain074 对 MrPro 的差混杂表效应，不能作 arc restoration 证据 | [已验证]（实现审查 operators.py/select.py） | AUDIT §2；两附件 | 同上；派生行动：补 MrPro×gain074 臂（未在本线程执行） |
| C3 | 静态表下平方弦距 \(4\sum w_j\sin^2(\Delta\nu_j/2)\) 不依赖绝对位置 p、对频率不单调；s28_less 在 5 个测距中 3 个降低分离（+.1623/−.3651/+.5320/−1.5277/−.4050 @1–16K），s29_more 为 −.1988/−.0022/+.0087/+.0340/+.1255 | [已验证]（CPU 从真实 contract 表计算；但单位权重示例≠实测学习权重） | AUDIT §3；RoPE_Allocation §6 | "解压→分离更好"的普适解释被否；计划中 min over p≈90–128K 项被指无作用 |
| C4 | MK2 成功的机制级解释：正确 6 vs 错误 9 的 logit margin 四格 (−2.125,−1.125,−1.000,+0.250)，prefix +1.125、read +1.000 近加和跨 argmax，交互余项 +0.250；二元四格 (0,0,0,1) 像 AND 但不证强非加性 | [部分证据]（单案例、BF16 logits；交互项非零） | AUDIT §4；两附件 | 收紧了 qwen 的"非加性交互"说法；后续诊断（数字互换两法皆错 6624365/6624369、位置 ID+1 令 E1 成功翻败）进一步限定"绑定恢复"解读（RoPE_Allocation §6） |
| C5 | s29_more 的 32K +8.333pp 全部来自一条 QA；其 cached 交叉四格全 0、未保留原现象 | [已验证]（面板逐条 + causal_cases） | AUDIT §Verified/§4 | "普遍短端机制"宣称被否；同时提示收益对数值执行路径敏感≠收益不存在 |
| C6 | "零训练+静态+非几何无人占据"的创新边界过宽：MrPro 本身已是非全局等比（radix 递增 ε=2i/[N(N+1)]），YaRN 亦分段 | [已验证]（arxiv 2601.22181v1 §3.1–3.2 式 10、14–16 直查） | AUDIT "Prior-art boundary"；F2#136 | 创新定位改为"具体分配规则+可泛化预测"；两附件均改口径（"非几何只是研究简称"） |
| C7 | BM 在 OLMo-2-1B（实 1.485B）有独立大收益：独立 seed 72 条 BM 4K 81.81%/16K 51.32% vs MrPro 37.85%/2.78%，44 胜 0 负 28 平；BM 优于等预算 MrUni（76.88/32.12）与官方 YaRN（54.38/6.94） | [已验证]（本项目面板内独立复核集；跨模型通用性未立） | GPT6Pro §3；RoPE_Allocation §5 | 保留为"必须解释的真实成功"；S8@32K 双地板（MrPro .69/BM 6.94）限定其尺度 |
| C8 | BM 相对 MrPro 在 Qwen3B 长端为负（−7.29pp @g.1；−8.125 @g.074），模型/尺度反转必须解释；匹配 gain 后 BM 表净效应短 +1.6667/长 −5.3472pp | [已验证]（同输入 36 条面板） | AUDIT；RoPE_Allocation §5 表 | 反转本身是开放问题（10 问之 1、6 问之 1），非被推翻对象 |
| C9 | Smooth(MrBudget)：凸问题（固定 B=16/3 最小 roughness，KKT 独立求解验证，约束误差<1e−15，roughness .0130719→.0048864）解对了，但 128K 比 MrPro 低 9.7917pp | [已验证]（3B 开发面板 36 条） | RoPE_Allocation §4/§5 | 否证"再平滑一点"作为任务主线（对 6 Pro 固定预算建议的反证）；不否定全部分配自由度 |
| C10 | E1 冻结迁移规则（目标 MrPro transition 5/17 处取最近槽换前一槽 exponent；Qwen7B→slot28、OLMo→slot19）跨模型未完成：Qwen7B 长 −1.6667pp（18 条），OLMo 短 −3.3333/长 +3.8194pp（开发 36 条） | [已验证]（冻结、未按目标调参；小面板） | GPT6Pro §3 后续更新；RoPE_Allocation §5 | "统一最优表/统计等价"不成立；"有条件迁移"继续（标假设） |
| C11 | LongBridge：slot36–39 共同绝对频移 ±1/131072（32K ±0.25rad、128K ±1rad）：Slower 32K 80.5556%（−6.6667pp）/128K 80.0694%（**+1.9444pp**）；Faster 32K 持平/128K 73.9583%（−4.1667pp）；带内恒等式 A_δ=cos(dδ)A+sin(dδ)Q 数值误差约 8.9e−16 | [已验证]（36 条 + 48 文档长度单元 NLL 完整；开发面板） | RoPE_Allocation §7（轮 4 现场读取 summary.json #482/#636/#643） | 定位为"起点诊断"：Slower 长端正号保留为开发信号；不构成"低频越慢越好"法则；±1rad 是显式探索幅度非理论最优（#577 处替换文本原话） |
| C12 | "名义中频里已有长程时钟"：MrPro 部署下 slot36–40 周期 33,983–141,332 tokens、slot51 达 1,518,763；s28/s29 原生 32K 圈数 12.367/9.966（并非未转满一圈）；保持 40–63 不动≠保持全部长程相关频率（BM 在 slot38 造成 −1.676rad 变化）；MK2 实测证据距离 88,725（prompt 末 130816 − value 首 42091） | [已验证]（FP32 参考表直算） | RoPE_Allocation §3 | 推翻"单槽 arc-OOD/不足一圈"式解释；s28_less 未取模相位变化 +3.144rad@32K、+12.575@128K、+8.512@证据距离 |
| C13 | 更多 distractors 本身构成难度：一真 key + (n−1) 同分干扰下保持 mass 需 logit 优势 ≈ln(n−1)，四倍上下文约需 +ln4；attention margin ≠ 输出 token margin | [假设]（简化模型的推导，明示非整网充分条件） | RoPE_Allocation §3 | 无纠正；标注为数学可能性 |
| C14 | 论文（paper-2027 "Exponent"）达到接收水准：固定端点配对训练支持内部指数分配的行为效应；三范围内同时报告排序反转防过度解释 | 评审意见 [部分证据]（37 页 PDF 内证据；未核验代码/实验执行真实性——F1 原文声明"未访问代码、日志、项目历史或外部文献"） | F1 #69/#72 | 两个可改判问题（§C.3 split；0.074 系数来源）待作者答复 |
| C15 | ψ(x)=x(1−x)(2x−1) 对称零面积函数拟合 P2−Mr：仅解释 13.18% 几何差能量、slot29–31 方向拟反 | [已验证]（CPU 拟合，未作为模型结果） | RoPE_Allocation §4 | 关闭的是该构造，不是"所有几何设计"（AUDIT §6 反过度关闭原则） |

## 4. 失败机制清单

**A. 科研叙事层面（qwen 并行计划中被审计纠正的过度解读）**
1. 把数值实现误差（BF16）误读成一阶理论失效（"低估 160 倍"）→ 复发模式：**混层归因**：局部导数正确 / 部署算术 / 下游预测是三命题，必须分离陈述（AUDIT §1）。
2. 把 gain 混杂进表效应当机制证据 → **缺因果臂先下结论**；解法是补最小 2×2（MrPro×gain074）再谈。
3. 把"解压→相位分离增大"当单调规律 → **几何代理无方向定理**；实测 3/5 距离反而降低。
4. 把二元 (0,0,0,1) 四格当强非加性交互 → **阈值放大 ≠ 内部非加性**；连续 margin 显示近加和。
5. 用五个已知结果调一个代理再称"独立确认" → **回顾校准冒充预注册**。
6. 把 300–500 点搜索失败外推为"静态表族无 headroom"、把"输出持平"叫"维度无用"（E8 慢带 ν=0 保留内容通道）、把 NLL 可加性与任务分数对立当成对逐槽可加性的否证 → **超范围关闭方法族**（AUDIT §6 逐条列举）。
7. s29"短端恢复"其实一条 QA；OLMo 结果"开发 36 条"与"独立 72 条"两面板禁止混算；六任务小面板 78.125 不可对标论文 RULER-13 的 53.2 → **面板/口径混用**是本项目反复出现的失败通道。
8. 计划中的成功率与 runtime 估计被审计点名"是规划猜测，不得决定晋级" → **未测数字不得进结论**（对应铁律"不得把未测写成否证/确证"）。

**B. 候选构造层面（已实测失败/受限，出处 AUDIT 表 + RoPE_Allocation §5–§7）**
E2 slow-tail_more（128K −9.722pp，12 条初筛）；E8 zero51（−13.889pp；固定态代理极好而生成退化 → 否证"attention mass 足够"）；E7 投影（−9.514pp → 局部保护不保整网）；Smooth(MrBudget)（−9.79pp）；s28+s29 组合（128K 73.958% 低于两者单用 → 正方向不自动复合）；Faster（−4.167pp）；E9/E10 仅 12 条全部持平 → 审计与附件均拒绝写成"失败"（如实：未达完整评估）。QA cached 交叉丢失原现象 → **干预执行路径本身可毁掉被解释对象**；位置 ID+1 使 E1 成功翻败但首 digit margin 改善留存 → 胜利脆弱。

**C. 工程/流程层面（本二文件内）**
- rate_limit"180 次受阻"为 grep 假阳性（§1）：真实情况是 112 分钟**无事件空闲等待**；线程用 `wait_agent`（F2 出现 24 次）挂起等新指令。教训：**统计受阻要解析 `rate_limit_reached_type` 与 used_percent 序列，不能数关键词**。
- 对 `~/.codex` 的批量读取受沙箱/迁移影响间歇 ENOENT（本次实测；曾致一个 heredoc python `open()` 失败而同路径 `find`/`getsize` 成功）→ 用 find+cp 原子化。
- F2 fork 自带父会话 `compacted` 快照（#05:59:00.504Z replacement_history），继承的 6 条 user 消息时间戳全部等于 fork 时刻，**不代表用户真实发送时间**；引用需回父转录查原始时间（T2-P28 即是一例：真实记录在 USER_PROMPT_TRANSCRIPT_20260909.md）。
- 子线程与 /root 的编排消息全部 `encrypted_content`（F1 1 条、F2 21 条 MESSAGE + 4 NEW_TASK），**任务级指令内容不可从本二文件恢复**——digest 对轮 4 增量指令只能按"每次触发一次修订"的行为模式重建，不能引用原文。

## 5. 频率表/方法定义清单（含 32K/128K 得分，如本线程记录）

模型/环境：Qwen2.5-3B-Instruct（rev aa8e725…），36 层、16Q/2KV heads、head 128→64 slots、W=32768、b=10⁶、S=4、ν_j=ω_j·S^{−m_j}、g=1+c·ln4；BF16 权重/旋转输出、FP32 相位、SDPA、greedy、rep penalty=1。开发面板 36 条（32K×12 + 128K×24，官方六任务）；**均为开发证据，非独立泛化**。

| 名称 | 构造规则（摘要） | 32K | 128K | 记录出处 |
|---|---|---:|---:|---|
| MrPro | 中段 l=23,h=40,N=17：m=q(q+1)/[N(N+1)]，ε_i=2i/[N(N+1)]；g=.1 | 87.2222%（基线） | 78.1250% | GPT6Pro §2；RoPE_Allocation §2 |
| BM | ε_i=6i(N+1−i)/[N(N+1)(N+2)]，唯一最小化 Σ(Δε)²（端点零、Σε=1）；B=8 | 91.6667% | 70.8333% | 同上；定理[已验证]为几何最优非任务最优 |
| BM/gain.074 | BM 表 + g=.074 | 100.0000% | 70.0000% | 匹配 gain 后表净效应短 +1.6667/长 −5.3472pp |
| MrPro/.074 | MrPro 表 + g=.074 | 98.3333% | 75.3472% | 短端大涨非 BM 独有 |
| BM/gain=1 | BM 表、g=1 | 89.5833% | 58.8194% | NLL 改善、长端任务损失 |
| MrUni | m=q/N（B=8 与 BM 同预算） | 64.5833% | 73.3333% | 排除"总预算足够"说 |
| Smooth(MrBudget) | 固定 B=16/3 最小 roughness 凸解 | 87.2222% | 68.3333%（−9.79pp） | 轮 4 读取 summary.json（#308） |
| E1 s28_less | 仅 m₂₈:30/306→20/306；ν₂₈ .00207001995→.00216595642 | 87.2222%（0） | **83.3333%（+5.208）** | 集中于 MK2 一条 0→1、MQ 一条 .75→1 |
| E1 s29_more | 仅 m₂₉:42/306→56/306；ν₂₉ .00157984428→.00148275390 | 95.5556%（+8.333） | 77.9167%（−0.208） | 短端增益全在一条 QA |
| s28+s29 组合 | 两改动并置 | 87.2222% | 73.9583% | 不自动组合 |
| E2 slow-tail_more | j≥40 额外除 10^{6/64}（首筛 12 条） | 0 | −9.722pp | MQ 重复绑定/FWE 退化 |
| E7 local projection | BM 方向经局部输出约束投影 | +2.778 | −9.514pp | NMSE 四分解见 C1 |
| E8 zero51 | ν₅₁=0 保留内容通道（首筛 12 条） | 0 | −13.889pp | mass 代理反例 |
| E9/E10 | 距离双时钟 φ_j(Δ)=ω_jΔ(Δ≤**w=26**)后接 ω_jw+ν_j^Mr(Δ−w)；双频核 | 12 条全持平 | 同 | E9 距离核 w 记号即轮 3 修正处 |
| FullLagP2 | native 网格因果 lag 列对最小二乘残差 u_j→m_j=(1−ū_j)²，全 64 槽，历史 gain .074 | Qwen1.5B/64K（MK2/VT/FWE）：P2 37.5/87.5/70.83 vs Mr 12.5/82.5/45.83（同 gain Mr 25/77.5/45.83）；128K MK2 双方 0 | Qwen3B/64K：75/90/66.67 vs 50/95/75（1 胜 9 平 2 负，缺同 gain 控制） | 历史面板 rep penalty=1.1 不可混算 |
| LongBridgeSlower/Faster | slot36–39 ν±1/131072，其余不动 | Slower 80.5556（−6.6667）/Faster 87.2222（0） | Slower **80.0694（+1.9444）**/Faster 73.9583（−4.1667） | NLL 8/16/32K：Slower +.0010876/−.0002314/−.0002280；Faster +.0002666/−.0007077/−.0008891 |
| BM_ScaleTaper | T_j=2π/ν_j^Mr，w_j=clip[ln(W/T_j)/lnS,0,1] 几何插值 BM↔Mr（24–31 BM、32–35 渐退、36–63 Mr） | 已实现/排队 | — | [未测]（附件明示无结果） |
| FullLagP2_Transfer3B | 历史 64 频率数组原封 + gain.074 于当前 3B 面板 | 排队 | — | 源 MrPro 数组与当前 3B 基线逐位核对相同（#592 替换文本） |
| G1/G2/G3 | (m₂₈,m₂₉) (24,48)/(36,36)/(36,48)/306 | 排队 | — | 来自"用户转来的 5.6 Pro 分析"；G1/G2 各增 roughness 20(6/306)² 且动外侧 gap |
| 冻结迁移规则 | j=l+round[(h−l)·5/17] 槽换前一槽 exponent | Qwen7B 83.3333（0）/OLMo −3.3333 | Qwen7B 82.7778（−1.6667）/OLMo 18.7500（+3.8194） | 见 C10 |
| OLMo 历史 | OLMo-2-0425-1B（1.485B），W4096/base500000/边界14/32/S4/g.1 | 4K：MrPro 37.22、BM 79.44、独立72条 BM 81.81 vs MrPro 37.85、MrUni 76.88、YaRN 54.38 | 16K：14.93 / 49.03 / **51.32** / 2.78 / 32.12 / 6.94；S8@32K 双方近地板 | GPT6Pro §3 |
| F1 论文对象（paper-2027） | Cosh 凸目标指数分配 vs Geo 等；151.9M 三种子三长度；OLMo QA 631 输入宏 F1 +3.82 [1.32,6.29] | 论文内 | 同 | F1 #69；评审=建议接收 |

自然文本 NLL（尾 512，FineWeb-Edu 16 篇@8/16/32K）：s28 vs MrPro +.000154/−.000604/−.000882 nats；s29 +.000958/−.000211/−.001363；Smooth +.000975/+.001925/+.003423；MrUni +.0000408/+.0005863/−.0025646。远程根：`/root/autodl-tmp/nongeometric_screen_20260909`（以各方法 summary.json/ruler.jsonl 为准，根 development_summary.json 已过期）。

## 6. 用户指令与纠正（原文引用）

继承自父会话的明文 user 消息（F2 #4–#14，时间戳为 fork 时刻 2026-09-10T05:59:00.504Z，真实发送时间见父转录）：
> "有卡了，你去实验吧：ssh -p 27741 [REDACTED_EMAIL]"

> "现在情况如何？"

> "你这个是纯粹的猜，没有任何理论依据啊，"

> "是的，而且你有成功案例，你要夸模型，跨任务也试试，你继续研究为什么吧，我们超过MrRoPE是目的，但是也要尽可能通用和泛化，理论要能解释"

> "这是我让qwen3.8-max 和你同时进行的研究，还阅读了你的transcript，你用一个子代理去分析，结合实验实际，看看有什么结果docs/research/PARALLEL_NONGEOMETRIC_20X10_PLAN_20260910.md"

（第一条"有卡了"读作"有（新）卡了"——GPU 就绪提示；[假设]，无更多上下文。）

项目文档中定位的停止/收尾指令（非本二文件明文，出处注明）：
- `docs/research/USER_PROMPT_TRANSCRIPT_20260909.md` T2-P28（2026-09-09T00:45:05.060Z）：**"你的好实验报告提交并推送代码吧，别找了"**；`USER_INTENT_GUIDE_20260909.md:50`："这条历史停止指令意味着不能凭旧 goal 恢复那一夜的研究队列"。
- 主会话 09-08 同类指令（`raw_thread-main.txt:336`）："怎么说，诊断了吗，我准备回去了，没法监督你，我会在家里PC继续工作，你尽快收尾，提交并推送代码吧"；执行记录（:365）：分支 main_0726_09_06 @ 0177e6d 已推送 + HANDOFF.md。

编排器（/root→子线程）指令：F1 1 条 + F2 25 条均为 `encrypted_content`（524–5004 B），**原文不可得**；其存在由 iacm/agent_message 框架行与"收到消息→执行一次 python 替换修订"的时间耦合证实。

交付文档内嵌的用户指令性口径（轮 2 文件自述，反映用户目标）："请独立判断，可以指出双方都错"；"不要把本文件中'尚未完成'写成结果"；"我们需要你推进方法，而不只是增加审查限制"（GPT6Pro §1/§6）。

## 7. 未决问题

1. **轮 4 与轮 2 的 NEW_TASK/MESSAGE 明文指令内容不可恢复（加密）**——两份 Pro 附件的完整需求清单只能从产物反推；父会话 `01a0806f` 转录是唯一可能来源。
2. **112 分钟空窗内父会话在做什么**无本线程证据；"长时间受阻"若属实应在父会话或 GPU 队列侧找记录。
3. **GPT6Pro 附件当前磁盘版本（mtime 06:28Z）含线程外改动**，具体差异未 diff——以轮 3 末态为基准的修订清单不完整。
4. 排队/进行中、尚无结果的项（定稿附件 §5/§7 明示）：Qwen3B 独立新 seed 128 条（MK2/MQ/VT/QA×32K/128K×16）；MrPro×gain074 与 shape×budget 2×2 完整臂；G1/G2/G3；BM_ScaleTaper；FullLagP2_Transfer3B；LongBridgeSlower 冻结原表/gain 的定向确认 48 次生成（VT128K 16 条 + MK2 32K/128K 各 16 条，复用同 cohort MrPro 结果）；E9/E10 完整面板。
5. 理论开放核心（两附件问题组）：能提前预测方向的统一机制（Q1）；signed margin→可泛化选表规则（Q2）；s28/s29 是过渡结构还是脆弱偶然点（Q3）；BM 最小粗糙度的任务意义或无（Q4）；P2 条件残差抓住了什么（Q5）；prefix/readout 分解的解释力上限（Q6）；gain×长度×(S,g) 规则（Q7）；精度改变测量还是对象（Q8）；跨模型该运输什么坐标（Q9）；"超过 MrPro 且通用"的判定证据与转动态核条件（Q10）。轮 4 收敛为 6 问（"寻找更好的整体分配规则"为中心）。
6. 外部 Pro（5.6 Pro / 6 Pro / GPT-6 Pro / qwen3.8-max）回复是否到达、如何被消化——本线程只见"用户转来的分析"入件，无回件记录。
7. 接收评审（F1）提出的两个可改判问题（§C.3 split 隔离；§F.1 0.074 选择来源）在论文侧如何处置——本线程无后续。
8. 慢频"训练时 Wω≪1 → 长端陌生对齐"论证与"已转一圈≠联合模式学过"的张力，仅有推导与反例，无实验臂（附件 §3 标为设计选择）。
9. 数值路径敏感性问题（QA cached 交叉毁现象、位置+1 翻败）：需要"对部署算术稳健的 margin"定义——10 问之 8 未答。
