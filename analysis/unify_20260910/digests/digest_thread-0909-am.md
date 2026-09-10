# digest thread-0909-am（2026-09-09 凌晨会话簇：00:50–07:42 本地时间，7 个 rollout）

提取方法：codex rollout JSONL → 仅保留 `response_item` 中 role∈{user,assistant} 的 `message` 与 `agent_message` 负载（任务指令正文为 `encrypted_content`，不可读，以任务名+助手自述代替），工具调用命令单独抽取用于时间线重建。压缩纯文本位于
`/Users/misaya.yanghejazfs.com.au/paper_project/hybrid-rope/analysis/unify_20260910/raw/raw_s{1..7}-*.txt`（7 个文件合计约 79KB，已全部通读，无抽样）。
工具调用清单（54.5KB）持久化于 Claude 侧
`/Users/misaya.yanghejazfs.com.au/.claude/projects/-Users-misaya-yanghejazfs-com-au-paper-project-hybrid-rope/7cf3ec49-b384-45ca-99f1-edab805c575a/tool-results/b4kxtt5u7.txt`。
时间戳说明：消息 timestamp 为 UTC（Z 后缀）；文件名与用户口语时间为美东 EDT（UTC−4）。00:50–02:25 本地 = 04:50–06:25 UTC；07:38 本地 = 11:38 UTC。

## 0. 范围核对（重要：与派发任务描述的偏差）

派发简报称本簇是"失败对话"最密集的一段（两核路线评估、10 候选计划执行）。逐文件全文检索结果与此不符，必须先记录：

- 在全部 7 个 rollout 的**明文**（消息+工具命令）中检索 `两核/双核/两个核心/十候选/十个候选/TEN_CANDIDATE/ten_candidate`：**全部零命中**（检索命令与结果见本 digest 工作记录；`候选` 仅命中 s6 中"三个候选旧 skill"与近邻 01-53-14 运维会话，均非研究候选）。子代理任务指令正文加密，无法排除其中提及，但助手自述与最终报告一致表明任务不是这两项。
- "两核"在项目内的真实出处是 **当晚（09-09 晚→09-10 晨）** 的 `docs/research/TWO_CORE_SOL_HANDOFF_20260909.md`（mtime 9月9日 21:14；"今晚围绕这两个核心（PC2 + PMKeep）持续改进……用户早上自行停止"，E01–E12 十二项实验）——不在本簇 7 个会话内。[已验证：文件内容+时间戳]
- "10 候选"最接近的两处：① 本簇实际参与的 **十轮独立 PDF 审稿流水线**（`paper-2027/research/pdf-review-rounds/20260909/README.md`："Ten independent PDF review rounds"，本簇覆盖 r02–r05）；② **十个非几何候选计划** `docs/research/NONGEOMETRIC_TEN_CANDIDATE_PLAN_20260909.md`（文首自述"2026-09-09。状态：研究设计完成，尚未实现或测得这些新候选的模型结果"；mtime 为 9月10日 07:50，其 §2 引用了本簇 s7 产出的 `FOLLOWUP_RESULTS_20260909.md`）。两核 E01–E12 的"10/12 项执行"与十轮审稿均为 09-09 日间之后或晚间工作，不在本簇。[已验证]
- 本簇 7 个会话的**实际主题**：paper-2027 论文的 4 轮盲审子代理（r02、r03、r04、r05，其中 r05 中断）+ 七篇 RoPE 论文的"取舍写法"风格调研（r? 名 `seven_paper_tradeoff_style`）+ 学术 skill 软删除后的 4 个替代 skill 创建（`research_skills_sol`）+ KV 压缩检索失败的子代理归因（`retrieval_failure_analysis`，含用户关于三项后续方法与"检索题是不是概率问题"的真实指令）。近邻（未列入任务清单、但同目录、同时段、id 同前缀）的 `rollout-2026-09-09T01-53-14-01a084ba-42b2-...jsonl`（11MB，source=vscode，用户直接交互会话）包含：MarkItDown 工具溯源、三套学术 skill 软删除、两台 AutoDL 间模型迁移与下载、DeepSeek-V4-mini-300M-init scaffold 准备、两台关机——一并记入 §2.8 供统一理论参考。[已验证]

## 1. 来源清单

全部父线程 = `01a0806f-3df5-74b1-bc56-bf00d89d238e`（2026-09-08T09:52:49Z 起，source=vscode 的主编排会话，其 rollout 位于 `/Users/misaya.yanghejazfs.com.au/.codex/sessions/2026/09/08/rollout-2026-09-08T05-52-49-01a0806f-3df5-74b1-bc56-bf00d89d238e.jsonl`）。工作目录均为 hybrid-rope 仓库，分支 `codex/exponent-allocation-manuscript`（s7 父链标注 `main_0726_09_06`）。

| key | 文件（.codex/sessions/2026/09/09/） | 大小 | JSONL 行数 | 提取消息数 | 会话 id | 子代理名/昵称 | 本地起时 |
|---|---|---|---|---|---|---|---|
| s1 | rollout-…T00-50-47-01a08481…jsonl | 16,032,697 B | 188 | 5 | 01a08481-1480-7a81-832b-cff00b932107 | /root/pdf_review_r02 "Carson" | 00:50 |
| s2 | rollout-…T01-08-08-01a08490…jsonl | 13,488,047 B | ~200 | 5 | 01a08490-f75e-7500-ab91-0241913d0594 | /root/pdf_review_r03 "Feynman" | 01:08 |
| s3 | rollout-…T01-25-25-01a084a0…jsonl | 10,730,440 B | ~210 | 5 | 01a084a0-ca7e-7132-92f8-b668b6c7b4a7 | /root/pdf_review_r04 "Schrodinger" | 01:25 |
| s4 | rollout-…T01-53-17-01a084ba-4d5d…jsonl | 1,738,612 B | ~160 | 5 | 01a084ba-4d5d-7483-9652-884ca9248ad2 | /root/seven_paper_tradeoff_style "Kuhn" | 01:53 |
| s5 | rollout-…T02-01-16-01a084c1…jsonl | 2,354,737 B | 95 | 4 | 01a084c1-9be2-7d11-9c5a-8a7d27585969 | /root/pdf_review_r05 "Euclid" | 02:01 |
| s6 | rollout-…T02-16-08-01a084cf…jsonl | 1,315,674 B | ~70 | 9 | 01a084cf-3bcf-7272-881a-94b8ffe4e7cc | /root/research_skills_sol "Bohr" | 02:16 |
| s7 | rollout-…T07-38-14-01a085f6…jsonl | 803,498 B | ~70 | 11 | 01a085f6-1cd5-7452-a902-6397b82ff5e4（内嵌父 meta 01a0806f） | /root/retrieval_failure_analysis "Linnaeus" | 07:38 |

16MB/13MB 大会话"重"的原因：`custom_tool_call_output` 单行最大 7.26MB / 3.2MB / 2.5MB / 1.9MB——全部是 PDF 页面渲染的 base64 图像与整篇 pdftotext 输出，对话文本本身极少。[已验证：逐行体积统计]

近邻补充：`rollout-…T01-53-14-01a084ba-42b2…jsonl` 11,497,395 B（09-09 01:53–03:19 本地，交互式主会话，62 条消息）——非清单内，见 §2.8。

## 2. 任务时间线（逐会话）

### 2.1 s1 = pdf_review_r02（00:50–01:00 本地 | 04:51–05:00 UTC）— 成功
- 目标：对不可变 PDF 快照 `paper-2027/research/pdf-review-rounds/20260909/r02/paper.pdf`（sha256 `0021523e…a89c`，identity.json）做全文+逐页渲染盲审，只读该 PDF 及临时导出物。
- 方案：`pdfinfo`+`pdftotext -layout` 分 6 块通读（~3060 行）→ `pdftoppm -r130` 38 页全渲染 → montage 缩略 + `view_image` 分批查看约 32 页 → `rg` 关键词（causal/establish/prove/isolate/optimal/bootstrap/selected…）做措辞审计 → send_message 回传 /root（正文加密）。
- 结果：**成功交付**。评审全文存于 `pdf-review-rounds/20260909/r02/review.md`（与会话末 assistant 长文一致）。判定："边界，偏接受，信心中高"。
- 关键意见（数字证据）：① Table 15 中 Cosh 与位移幅度匹配的 exponential ΔNLL=+0.00074、区间跨零→非均匀分配被识别、Cosh 函数形式未被识别；② 成熟模型结果为方法级比较（Llama Native 与 midpoint-Cosh 连有限网格端点都不同；Qwen g=1.0513 vs YaRN 1.0693；Video DiT sweep 后 best τ=1.5 单种子；1.485B 训练器栈不同）；③ 991.5→127.9 仅一个 matched trained pair，bootstrap 不能替代训练变异；④ Eq.(6) cumulative-tail penalty 是指定设计偏好、非从几何推导，A.6 只验证 cosine slice，pp.16–17 有反例表明静态秩排序不保证跨长度行为排序；⑤ pp.5,33–34 十例 passkey 屏蔽 gold block 的 NLL 变化 +1.5055 / −0.0095 未说明聚合方式；⑥ 可复现性陈述强于证据（p.38 §G.1 未打包 checkpoint）。

### 2.2 s2 = pdf_review_r03（01:08–01:18 本地）— 成功
- 同一流程审 r03 快照。判定："边界，信心中高"。存于 r03/review.md。
- 关键意见：① 摘要级 Llama-3-8B PPL（991.5→127.9）缺协议——"三个域的 24 个 frozen temporal text packs"无来源/构造/嵌套性/计分掩码/聚合公式，要求补协议表与 24 pack 配对 NLL 差；② Theorem 2 的 α∫ρ²+β∫Sρ² 凸泛函**不是** canonical correlation / r₂(Γ) 的连续极限，Table 6 用 cosine-only collision 支持 Cosh，而 §A.3 自己构造了 cosine-only 排序与 full-subspace 排序相反的反例→几何与 Cosh 链条未闭合，建议明确写"design construction, not an optimizer"；③ Qwen 64K +6.09 是 bundled table+gain 比较；④ 慢频 S_ω^{-1/2} 白化把病态方向放大 ~1/ω，r₂≈2 不等于两个可用自由度；⑤ 附录空页/Table 11 版面。

### 2.3 s3 = pdf_review_r04（01:25–01:39 本地）— 成功
- 唯一显式走 skill 的一轮：先读 `~/.codex/skills/academic-research-suite/SKILL.md` 与 `ars/academic-paper-reviewer/WORKFLOW.md`、跑 `ars/scripts/pdf_read_preflight.py`（此刻 ARS 尚未被软删，删发生在 02:05–02:06 本地，见 §2.8）。判定："**接受倾向**，信心中高"。存于 r04/review.md。
- 关键意见：① RULER 官方 macro 与严格指标分歧：Llama-3-8B@16K 官方 macro 0.295%/14.03% 而 normalized exact 0/1.54%；@8K 官方 macro EVQ-Cosh 更差 94.44/77.60、normalized exact 却略好 17.69/21.54→结论依赖 scorer；② 核心固定范围实验只报差值不报两臂绝对 NLL（Table 12）；③ Proposition 1 在 ω=0 退化为一维，证明实际用了 sin(ωΔ)/ω 重标度+Grassmannian limit，命题应限定 ω>0；④ 23 vs 24 对慢频来自 k/K 与 k/(K-1) 两种网格（46 vs 48 nominal dims）；⑤ τ 来源混合（zero-search convention / 固定 recipe / 经验预设 / sweep 最优）应逐结果标注；⑥ EVQ 缩写从未展开；Table 1 "38K" 视觉粘连；pdffonts 检查字体嵌入。

### 2.4 s4 = seven_paper_tradeoff_style（01:53–02:01 本地）— 成功
- 目标：核对 7 篇 RoPE 论文（MrRoPE、Decoupling、Deconstructing、Group Representational、PPE、Selective RoPE、RePo，位于 `~/Downloads/RoPE_Papers/Markdown/`，MarkItDown 转换件）中"取舍/适用条件/局限"的实际写法，给主代理组织论文负结果用。
- 方法：对 7 个 md 逐篇 `rg '^#'` + `nl -ba | sed` 定点抽取（含 limitation/trade-off/failed 等词审计）。
- 结果（会话末原文）："这些论文保留实际非最优结果，并按其意义组织正文——机制分化、成本收益、适用条件、独立后续问题各有位置。Selective 明写模型相关 trade-off；RePo 用 comparable 概括含下降的通用任务；PPE 直接并列压缩收益与精度损失。可迁移的是这种论证组织，而不是隐藏下降或反复自我否定。" [已验证：s4 raw 126–129 行]

### 2.5 s5 = pdf_review_r05（02:01–02:03 本地）— **中断，未交付**
- 流程正常推进（mktemp、pdfinfo、pdftotext layout+raw、06:02–06:03 已通读 3060 行、渲染 37 页 jpeg、view_image 到第 4–7 页），06:03:52Z 最后一条 assistant message 之后事件流出现 **`turn_aborted`**；全文件无 `send_message`、无 `task_complete`；`pdf-review-rounds/20260909/r05/` 目录只有 paper.pdf+identity.json（created_utc 2026-09-09T06:00:48Z），**没有 review.md**。
- 结论：十轮审稿计划在 r05 处被中断（与 s6 skill 重建、s4 调研穿插发生在同一 40 分钟窗口，主会话当时正做大改组；中断原因明文不可证，[假设] 与 skill 软删除/上下文切换有关）。r06–r10 在 20260909 目录中不存在。

### 2.6 s6 = research_skills_sol（02:16–02:25 本地）— 成功
- 背景：近邻交互会话（§2.8）在 02:05–02:06 本地把 `academic-research-suite`、`write-scientific-prose`、`interactive-study-guide` 软删除到 `~/.codex/skill-backups/2026-09-09-academic-research-writing/` 并清理 deslop 过期引用。
- 本任务：按 `.system/skill-creator` 规范新建 4 个独立 skill：`reason-research-theory`、`design-research-experiments`、`analyze-research-literature`、`review-conference-paper`（`init_skill.py` 生成 → apply_patch 重写 SKILL.md → 中途一次整批 Delete+重建 → `quick_validate.py` 四项 exit 通过 → 追加修订 design-research-experiments 的"estimand/unit of analysis"条目）。
- 交付原则（会话末原文摘要）："吸收了备份中的一手来源验证、证据锚点、盲审隔离、失败诊断和渐进披露；舍弃旧 ARS 总路由、多代理面板、固定阈值与轮数、复杂 schema/checker、Material Passport、跨模型传输和机械评分规则。未依赖任何已删除 skill，也未修改仓库、论文、AGENTS.md、记忆、配置或 narrative skill。" [已验证：s6 raw 158 行]

### 2.7 s7 = retrieval_failure_analysis（07:38–07:42 本地 | 11:38–11:42 UTC）— 成功交付归因
文件头部内嵌了父会话（01a0806f）09-09 早晨被回放的完整用户回合（各消息同刻 11:38:14.131Z 出现即回放标志），内容链：
1. 用户："已经开机了，ssh -p 57109 root@connect.westc.seetacloud.com"；父代理上一轮已完成 rope_operator_family 首次 GPU 配对实验并误报"GPU 已空闲，可以关机"。配对结果（REPORT.md 数字）：原模型留出 NLL 2.651387 / 共同初始化 5.300757 / score+value 拟合 9.989853 / output+value 蒸馏 3.942717；两边均 50% KV（256 实数 vs 原生 512），均未恢复冻结 8K 检索题（8189 chat tokens、412 条随机记录、答案 `f83a9144`）。
2. 用户："**没有其他方法了吗？**" 助手自我纠正："我上一条'可以关机'的表达太早，容易让人以为这条路线已经结束"——列出三个方向：① BKV 平衡初始化（TransMLA 有原文依据：未平衡联合 PCA 偏向幅度大的 K）；② attention 分布/输出约束拟合（KL 替代原始 score MSE）；③ 学生实际输入逐层恢复。明确"第二、三项仍是待验证假设，不能提前承诺有效"。
3. 用户："**那你试，试完这三个看看**" → 父会话在 GPU 上执行（执行本身不在本簇 7 文件内，产物 `experiments/rope_operator_family/FOLLOWUP_RESULTS_20260909.md`）：BKV+KD 4.992231（劣于原 KD）、attention KL+output+value **3.745052**（7/8 篇改善，最优）、逐层 progressive 4.567863；三者答案 NLL 4.466099/3.296716/3.747326，**全部未恢复检索**。关键细节：BKV 只初始化不训练 NLL 4.150590 优于旧初始化 5.300757，但同 Adam 配方后反而 4.992231——"初始化改善不自动转化为同一优化器下的最终改善，不能简单总结成 BKV 没有作用"。attention_kd 输出的是被询问的 key `5c4cb02e` 而非 value（内容错误，非格式差异）。
4. 用户："**检索题是不是概率问题，对模型太难了，原始模型也不一定此次答对，有没有可能？**"与"**为什么破坏了检索呢，你让一个子代理分析原因**" → spawn 本 s7 子代理（Linnaeus）。
5. 子代理交付（11:42:25Z 最终消息，[已验证，全部为只读 CPU 复测+代码依据]）：
   - 概率问题裁定：原模型同一输入**两次贪心均答对**且答案 NLL 0.000241、不压缩的精确 CompactAttention 接口也答对（NLL 0.000283，KV 字节与原生一致，排除接口本身不可答）→"**这条冻结题的失败不能归为随机没答对**"；但单题不能估计总体准确率。
   - 单层替换答案区 attention mass（0-based，目标行 token 1033–1051、答案字符 1044–1051，query 8188）：layer20/head5 native 0.500751 → init 0.459702 → operator(score) 0.352697 → output KD **0.203389** → attention KL+KD 0.211080；layer26/head1 native 0.0483004 → output KD **0.000048458**（几乎清零）。
   - attention/value 双通道局部分解（进 o_proj 前跨 12 头 NMSE）：layer20 仅换 attention 0.183092 / 仅换 value 0.239501 / 全换 0.378011；layer26 对应 0.466273 / 0.071682 / 0.446137 → layer20 瓶颈在 value 重建、layer26 在 attention 路径；三列不可相加（交叉项）。
   - score 目标错配的数值反例：layer0 score 因子原始 relative score MSE 仅 0.00001305，去行均值后 0.48446，attention KL 7.196 → softmax 对行常数平移不敏感，小 relative error 不能解释成竞争关系已保存。
   - 结构性事实：原生 2×KV-head×128 维 K+同宽 V=512 实数被压为共享 64 维 rotary + 192 维 content=256；`operator.py:171–181` `score = qP·([k,v]C) + rotate(qA)·rotate(kB)`、`value=[k,v]CU`（代码依据 operator.py:107–113,133–151,171–181）；一般不能精确保存全部原生响应，但**不能据此断言该预算必然无法保留检索**。
   - 协议差异：`study.py:82` 每篇仅 64 个等间隔 query、K/V 来自 2048-token PG19 前向；`study.py:123–138,221` 位置放大只改 RoPE 相位、不重生成真实长上下文 hidden states、不增加竞争 key 数——"能训练距离响应，不能等同真实 8K 412 条随机记录精确复制"。[已验证协议事实]；是否为主要瓶颈 [假设，待干预]。
   - 频率漂移反事实：output KD 的 learned phase 换回 init phase，layer20 mass 0.203389→0.321101（仍远低于 0.500751）、layer26 0.000048458→0.000081950 → 频率学习有局部影响但不足以解释全部损伤。
   - 逐层恢复：`core_progressive_kd.json` 仍错（答案 NLL 3.747326，输出 `value 19/000000000000000000000000000`）→"只说明本次逐层恢复方案不充分，不能否定状态漂移存在或所有逐层方法"。
   - 最小下一步：在 output KD 整体模型中恢复少量有证据的原生层、测同题 NLL/生成（模型级因果干预）；真实 8K 校准与更宽表示是两条竞争解释。
   - 数字告警：新增测量"仅在工具输出中，未保存文件"（后被主代理复现并存入 FOLLOWUP_RESULTS_20260909.md——两处表一致，含 bkv_kd mass 0.18048660/0.00001128 与 progressive_kd 0.01997839/0.00110177 两行，为父代理补测）。

### 2.8 近邻交互会话 01-53-14（01:53–03:19 本地，11MB，不在清单但同簇同时段）— 全部完成
- MarkItDown 溯源：用户问 `RoPE_Papers/Markdown` 用什么转换；助手初查 `base` 环境报"未安装"被用户纠正（"你刚才只查了当前安装命令/包，范围不够"），确认在 `aidemo` 环境 v0.1.5。
- 学术 skill 软删除（02:05–02:06 本地）：ARS/write-scientific-prose/interactive-study-guide → `.codex/skill-backups/2026-09-09-academic-research-writing/`，清 deslop 残留引用；直接触发 s6 的 4 skill 重建。
- AutoDL 迁移（02:09–07:19 本地）：老机→新机 rsync 实测 ~1MB/s 放弃；并行 5 组 ModelScope 下载被用户制止（"这个下载任务不该继续占着机器"——助手承认"刚才的做法过头了……实际上没有下载任何论文"，SHA-256 仅为迁移校验）；改单命令 ModelScope 公共直链 ~29MB/s 顺序下载 Qwen2.5-7B(4片)/3B(2片)/1.5B/OLMo-2-1B 完成（日志 `ALL_MODELSCOPE_DOWNLOADS_COMPLETE`，磁盘 50G 曾打满、清残片后剩 19G）；再按实验方案只补缺的 `kshitijthakkar/deepseek-v4-mini-300M-init`（~317M 随机初始化 V4 scaffold，含 CSA/HCA、sliding、partial RoPE、MTP；1,278,268,920 B）两台机各一份；用户指令"transformers 不装、无卡模式不做加载测试"被执行；最后两台关机。该 scaffold 服务于"你贴出的 CC-RoPE 机制实验"（09-09 晚两核/候选路线的基础设施）。[已验证：该会话 assistant 明文]

## 3. 理论主张表（本簇会话中出现的、与统一理论相关的主张）

| 主张 | 证据等级 | 出处 | 后续是否被纠正/推翻 |
|---|---|---|---|
| 固定端点三种子实验（151.9M，只改 30 个内部频率）因果识别的是"内部指数分配形状"，这是论文最硬的证据 | [已验证]（三评审独立一致 + 论文 §3.1/Table 12） | s1 05:00、s2 05:18、s3 05:39 | 未推翻；r02 要求叙事区分"分配有效"与"Cosh 特异" |
| Cosh 的函数形式优势**未**被经验识别（vs 位移匹配 exponential ΔNLL=+0.00074 跨零） | [已验证（Table 15 层面）] | s1 r02/review.md | 与 MEMORY 中"cos-only kernel = half the story"审计一致 |
| Theorem 2 的变分泛函 α∫ρ²+β∫Sρ² 是指定设计构造，非 r₂/c 的连续极限；cosine-only 排序与 full-subspace 排序可相反（§A.3 自带反例） | [已验证（论文内反例）] | s2 r03/review.md | 未被推翻；直接支持"静态碰撞≠外推机制"的复盘结论 |
| Qwen normalized-index vs YaRN（64K +6.09）是 table+gain 捆绑比较（g 1.0513 vs 1.0693），纯 placement 只支持 Table 23 同增益对照 | [已验证（表格定义）] | s2 r03、s3 r04 | 未推翻，措辞级修正 |
| RULER 官方 macro 与 normalized exact 分歧大（16K：14.03% vs 1.54%；8K 方向甚至相反）→ 能力结论依赖 scorer | [已验证（论文 Table 16/§E.3 数据）] | s3 r04/review.md | 未推翻；对"代理指标≠能力"铁律是直接实例 |
| 慢频子空间 r₂≈2 是白化后的方向重合度，S_ω^{-1/2} 放大 ~1/ω，不等于有限精度下两个可用自由度 | [已验证（解析）] | s2 r03 | 未推翻 |
| KV 联合压缩到 50% 预算下，四种拟合（init/score/output KD/attention KL+KD）与逐层恢复均未恢复冻结 8K 检索；失败非采样运气（原生两次贪心+无压缩同接口均答对） | [已验证（单题）] | s7；REPORT.md、FOLLOWUP_RESULTS_20260909.md | 界限定了"本配置/本求解"，不否证压缩族或 3B 频率分配（十候选文档 §2 明确此边界） |
| 检索损伤至少部分由"关键内容 attention 权重与 value 传回同时受损"造成，层间瓶颈不同（L20 value、L26 attention） | [部分证据（单层局部替换，非全模型因果归因）] | s7 最终消息 | 模型级最小干预（恢复原生层）09-09 未执行 |
| score MSE 目标可掩盖 softmax 竞争关系（行常数平移不变；0.000013 vs 去均值 0.484 反例） | [已验证（layer0 具体数值）] | s7 第 5 点 | 推广到其他 score 目标是 [假设] |
| BKV 初始化更优（4.15 vs 5.30）但同优化器下最终更差（4.99）——初始化改善不自动传导 | [已验证（单配方）] | FOLLOWUP_RESULTS §实际结果 | "BKV 无用"是越界读法，被文档自身禁止 |
| 2K 自然文本相位重放校准 ≠ 真实 8K 分布（不增竞争 key、不重生成 hidden states） | [已验证（协议事实）]；其是否为主要瓶颈 [假设] | s7 第 4 点 | 待真实 8K 校准对照干预 |
| "论文可保留非最优结果，按机制分化/成本收益/适用条件组织正文；可迁移的是论证组织而非隐藏下降" | [已验证（7 篇原文核查）] | s4 最终消息 | 成为 r 之后论文写作的风格依据 |

## 4. 失败机制清单

1. **子代理审稿轮 r05 被 turn_aborted 中断且无交付**（s5，02:03:52Z；无 send_message、无 review.md）。复发模式警告：多轮流水线穿插在主会话大改组（skill 软删除）时，单轮评审没有落盘 checkpoint，中断即整轮作废且不留负结果记录；r06–r10 未再运行，"十轮"实际只完成 4 轮——引用审稿证据时**只能说 r01–r04 有 review.md，不得把"十轮计划"写成"十轮执行"**。
2. **"可以关机"式过早收束**（s7 父链）：只测了同一压缩结构的两种拟合目标就向用户宣告路线结束；用户一句"没有其他方法了吗"即暴露收束过早。助手自认"表达太早"。警告：对"路线失败"的宣告必须绑定方法族边界，否则会把未穷尽写成否证。
3. **score MSE 代理指标失真**：原始 relative score MSE 1.3e-5 看似几乎无损，行去均值后 0.48、KL 7.2——正是铁律"代理指标≠能力"的实例；同族失误历史见 OVERNIGHT_FAILURE_POSTMORTEM_20260909.md（09-08 夜）。
4. **初始化收益与优化收益混淆**（BKV）：换了初态又换参数坐标尺度，未搜索优化适配就下结论会错判两向。
5. **下载/迁移任务膨胀**（§2.8）：并行 5 组整套权重下载把"迁移"变成了"下不完"，且磁盘 50G 被 `.partial` 打满、HF 镜像 404（用 HF commit SHA 当 ModelScope 路径）与 0.27MB/s 慢速连环踩坑；用户两次纠正后收敛为单命令顺序下载。警告：迁移=校验搬运，不得顺手扩成模型获取任务。
6. **环境假设过窄**（§2.8）：只查 base conda env 就断言"未安装 MarkItDown"，被用户指出应查历史与所有环境。
7. **子代理测量不落盘**：s7 明文自述"新增数值仅在工具输出中，未保存文件"——幸而父代理复现入档；复发风险：若父不复现，关键证据随 rollout 沉没。

## 5. 频率表/方法定义清单（本簇出现的构造）

论文侧（审稿对象，paper-2027 main.pdf r02–r05 快照；得分引自评审转述的 PDF 表格）：
- **Cosh / EVQ-Cosh**：Theorem 2 变分解 ρ∝cosh 族，τ 为外部选择（zero-search convention/预设/sweep 混合）；固定端点三种子优于 FMRoPE，但 vs 位移匹配 exponential 无差异（ΔNLL +0.00074）。
- **FMRoPE（Geo/线性指数）**：基线族；151.9M 三种子中劣于非均匀族。
- **BM / MrRoPE-Pro**：论文中作为"完整表干预"比较对象（1.485B、Qwen 等）；Qwen 64K Index-vs-YaRN +6.09（bundled）。
- **midpoint-Cosh（Llama-8B）**：LoRA 适配方法级比较，PPL 991.5→127.9（单训练对，24 packs 协议缺失被 r03 点名）。
- 注意：以上"32K/128K 得分"细节存于 PDF/论文表格，本簇会话未逐值转录，引用需回 r 轮 review.md 或 main.pdf。

实验侧（rope_operator_family，Qwen2.5-1.5B-Instruct 32K checkpoint，50% KV 压缩，content rank 192 + rotary 64，28 层各 500 步 Adam，seed 42）：
- **initialization（共同初始化）**：NLL 5.300757；恢复参照。
- **score+value 拟合**：NLL 9.989853、答案 NLL 5.992948、错误。
- **output+value 蒸馏（KD）**：NLL 3.942717、答案 NLL 4.567184、错误；8/8 篇优于 score。
- **BKV+output KD**：NLL 4.992231、答案 NLL 4.466099、错误（BKV-only 不训 4.150590）。
- **attention KL+output+value**：NLL 3.745052（7/8 改善）、答案 NLL 3.296716、错误（输出 key 5c4cb02e）。
- **progressive（学生输入逐层蒸馏）**：NLL 4.567863、答案 NLL 3.747326、错误（`value 19/0…`）。
- 冻结检索题：8189 tokens chat、412 随机记录、`key=5c4cb02e, value=f83a9144`；原生 2.651387 / 0.000241 正确。全部原文件 `experiments/rope_operator_family/results/20260909_gpu/REPORT.md` 与 `FOLLOWUP_RESULTS_20260909.md`（mtime 09-09）。
- 术语：本项目"两核"= 09-09 **晚**交接的 PC2（pair covariance selector，NOSA/16K 四任务）+ PM/PMKeep（相位范围平均保留策略，Qwen3B 8K 自然文档），E01–E12 排程——**本簇无其执行记录**；"十候选"= E1–E10 非几何分配候选（`NONGEOMETRIC_TEN_CANDIDATE_PLAN_20260909.md`，本簇无执行、文档自述未测）。

## 6. 用户指令与纠正（原文引用）

- s7 链内（父会话 09-09 晨，11:38:14Z 回放）："已经开机了，ssh -p 57109 root@connect.westc.seetacloud.com"；"**没有其他方法了吗？**"；"**那你试，试完这三个看看，**"；"**检索题是不是概率问题，对模型太难了，原始模型也不一定此次答对，有没有可能？**"；"**为什么破坏了检索呢，你让一个子代理分析原因**"。
- s7 助手据此自纠："我上一条'可以关机'的表达太早，容易让人以为这条路线已经结束。"
- §2.8 近邻会话内："你说得对，既然那个目录里已有批量转换结果，我刚才只查了'当前安装命令/包'，范围不够。"（用户纠正环境检索范围后助手语）；"你说得对，这个下载任务不该继续占着机器。"；"你说得对，刚才的做法过头了。……实际上没有下载任何论文。"；"明白，transformers 不装，我只负责把模型文件下载好；无卡模式不做加载测试。"
- 铁律语境：09-09 当日仓库 AGENTS.md（每会话头部均含）第 3/5/7 条——"Distinguish hypotheses and proxy improvements from demonstrated outcomes"、"Do not repackage failed assumptions or generalize a specific failure beyond its evidence"、后加 "**Test the claim directly.** keep both positive and negative conclusions within what the experiment actually tested"（s7 头部首次出现第 7 条，说明 AGENTS.md 在 09-09 中午前后被更新）。[已验证：s1 与 s7 头部对比]

## 7. 未决问题

1. r05 中断原因（父会话为何 abort）明文不可证；r06–r10 从未运行——统一理论若要引用"审稿共识"，边界是 r01–r04（r01 在 09-08 深夜会话）。
2. 子代理任务指令全部加密（`encrypted_content`）；r02–r05 的验收标准只能从助手自述反推。若父代理简报口径（两核/十候选）确曾写入任务指令，则无法从明文核实——应视为 [假设] 并以本 digest §0 的明文检索结果为准。
3. s7 指认的模型级最小干预（恢复少量原生层测同题 NLL/生成）09-09 未执行——"局部 mass 损伤是否因果决定最终答案"仍开放；真实 8K 校准、更宽 latent 两条竞争解释未隔离。
4. attention_kd 的 NLL 改善（4.57→3.30）与仍答错并存——"答案 NLL"作检索能力代理的效度未单独审计。
5. 论文侧遗留：Table 12 两臂绝对 NLL、24-pack 协议表、Eq.(6) 与几何的桥、full-subspace c/r₂ 对 Table 6 的解析重算——均为审稿要求、09-09 晨无执行痕迹。
6. DeepSeek-V4-mini-300M-init scaffold（两台 AutoDL）与十候选计划/两核路线的关系（CC-RoPE 机制实验）要到 09-09 晚 `TWO_CORE_SOL_HANDOFF_20260909.md` 与 09-10 `PARALLEL_NONGEOMETRIC_20X10_*` 会话中取证——建议并行 digest 任务覆盖 `~/.codex/sessions/2026/09/09/rollout-…T21-44-01…` 与 `2026/09/10/` 全目录。

### 证据文件路径索引
- 原始转录：`/Users/misaya.yanghejazfs.com.au/.codex/sessions/2026/09/09/rollout-2026-09-09T{00-50-47,01-08-08,01-25-25,01-53-17,02-01-16,02-16-08,07-38-14}-*.jsonl`（近邻 `01-53-14`）
- 压缩文本：`analysis/unify_20260910/raw/raw_s{1..7}-*.txt`
- 审稿交付：`paper-2027/research/pdf-review-rounds/20260909/r0{1,2,3,4}/review.md`（r05 无）
- 压缩实验：`experiments/rope_operator_family/results/20260909_gpu/REPORT.md`、`FOLLOWUP_RESULTS_20260909.md`、`FOLLOWUPS_20260909.md`
- 远端工件：`/root/autodl-tmp/operator_family_prepare_20260909/work/`（s7 命令引用；机器 07:19 本地已关机）
- 计划文档：`docs/research/TWO_CORE_SOL_HANDOFF_20260909.md`（mtime 09-09 21:14）、`docs/research/NONGEOMETRIC_TEN_CANDIDATE_PLAN_20260909.md`（mtime 09-10 07:50）
- 新建 skills：`~/.codex/skills/{reason-research-theory,design-research-experiments,analyze-research-literature,review-conference-paper}/`
