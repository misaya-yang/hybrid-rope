# digest thread-0907 — 09-07/09-08 监控与小会话簇（LUNA 实验执行监控、拉取代码、引用/数值只读核查子代理）

提取时间：2026-09-10。提取方法：codex rollout JSONL → user/assistant 纯文本（跳过 tool 输出/reasoning/token_count），全文通读，无抽样。所有转录引用给出 UTC 时间戳；文件名时间为 America/New_York 本地时间（UTC = 本地 +4h）。

证据等级标注约定：本 digest 中 [已验证] = 转录内有落盘 receipts/SHA/行数级证据支撑；[部分证据] = agent 报告了结果但未在转录中展示原始文件全文；[假设] = 推断或未测主张。监控转录只记录执行事实，不等于科学结论（LUNA 交接原话："不依据 KL 或训练 loss 宣布能力提升"）。

---

## 1. 来源清单

| key | 文件（绝对路径） | 大小 | 行数 | user/assistant 消息数 | 活跃时段（UTC） | 性质 |
|---|---|---|---|---|---|---|
| 0819 | /Users/[REDACTED_AUTHOR].yanghejazfs.com.au/.codex/sessions/2026/08/19/rollout-2026-08-19T22-42-50-01a01d0c-bfa9-73d3-aedc-7e82f11f65d7.jsonl | 22,694 B | 4 | 0 | 2026-08-20T02:42:50Z 起 482 ms | 启动即被打断的空会话（无 user/assistant 内容） |
| 0907-monitor | /Users/[REDACTED_AUTHOR].yanghejazfs.com.au/.codex/sessions/2026/09/07/rollout-2026-09-07T02-42-41-01a07a9a-cf12-79d2-94e3-75b94a92b7e0.jsonl | 4,252,461 B | 1006 | 62 | 2026-09-07T06:42:49Z → 08:37:29Z（约 1h55m） | LUNA 执行交接监控主会话（GPU 实验队列 E0–E2） |
| 0908a | /Users/[REDACTED_AUTHOR].yanghejazfs.com.au/.codex/sessions/2026/09/08/rollout-2026-09-08T21-11-41-01a083b8-809c-7713-b51d-95bbf8e66c6e.jsonl | 266,768 B | 41 | 4 | 2026-09-09T01:11:54Z → 01:12:16Z | "拉取最新代码" 小程序会话 |
| 0908b | /Users/[REDACTED_AUTHOR].yanghejazfs.com.au/.codex/sessions/2026/09/08/rollout-2026-09-08T22-48-51-01a08411-7461-7b52-acd6-6a1b272197e6.jsonl | 1,176,122 B | 154 | 4（另 4 条 developer、1 条 agent_message） | 2026-09-09T02:49:05Z → 02:56:58Z | 多代理子会话 `/root/citation_fact_check`（引用机制核查，只读） |
| 0908c | /Users/[REDACTED_AUTHOR].yanghejazfs.com.au/.codex/sessions/2026/09/08/rollout-2026-09-08T22-49-13-01a08411-c962-70e0-88b0-f543cf955956.jsonl | 8,150,906 B | 426 | 7（另 4 条 developer、3 条 agent_message，2 个 task） | 2026-09-09T02:49:26Z → 03:14:49Z | 多代理子会话 `/root/numeric_evidence_check`（数值/证据核查 + 454M 协议恢复，只读） |

- 提取后的纯文本：`/Users/[REDACTED_AUTHOR].yanghejazfs.com.au/paper_project/hybrid-rope/analysis/unify_20260910/raw_0819.txt`、`raw_0907-monitor.txt`、`raw_0908a.txt`、`raw_0908b.txt`、`raw_0908c.txt`。
- 0908b/0908c 的 NEW_TASK 任务正文（Payload）是 `encrypted_content`，本地不可解密；任务范围由子代理自己的开场白与最终报告还原。两者的 `<context_window>` developer 消息给出代理名：`/root/citation_fact_check`（context window id 01a08411-7461-…）、`/root/numeric_evidence_check`（01a08411-c962-…）。两者 environment 里的 visualization root 都是 `01a0806f-3df5-74b1-bc56-bf00d89d238e`，即父会话 = 09-08 晚间主研究线程（该线程同时在跑十轮 Sol PDF 审稿与 asset_* 子代理簇：`analysis/unify_20260910/raw_thread-main.txt:1070-1082` 的 subagents 名单，numeric_evidence_check 代号 Hooke）。
- 交叉引用：本地交接文件 `/Users/[REDACTED_AUTHOR].yanghejazfs.com.au/.local/share/hybrid-rope-maintenance/cross-audit-20260907/LUNA_EXECUTION.md`（3,933 B，2026-09-07 02:35）及配套 `jobs.json`（SHA256 `718fa8a950c0141a880bde093c0f1af7fe16b88e69265e3782044a1c1d664937`）、`input_preflight_final.json`、`protected.json`、`yarn_995db5b.py` 等均在盘上（本次 digest 只引用其头部条款）。
- 仓库内 grep `jobs_e2_z_only|E0_report_1788764478|cross_audit_20260907|citation_fact_check|numeric_evidence_check` 在 docs/、paper-2027/ 下无命中（仅本次提取文件与主线程 roster 行）→ 这批改动的落盘文档不在 09-07/09-08 本地仓库快照里，E0/E1 结果文档归属主研究会话后续处理（[已验证]：grep 空结果）。

## 2. 任务时间线

### 2.0 会话 0819（2026-08-19，空会话）

- 目标：未知（无任何 user 消息）。session_meta：cwd=hybrid-rope，originator=codex-tui，cli_version=0.142.0，thread_source=user。
- 结果：**中断**。task_started 后 482 ms 即 `turn_aborted reason=interrupted`，developer 注入 `<turn_aborted>`（提示后台 unified exec 进程可能仍在运行）。无用户内容可提取。[已验证：JSONL 仅 4 条记录]

### 2.1 会话 0907-monitor：LUNA cross-audit GPU 队列执行（本次簇的核心）

授权链：用户 06:42:49 指令（§6 原文）"严格按照交接迅速执行实验，监控实验即可"；交接文件 LUNA_EXECUTION.md 的 GPU 合约是 RTX 5090 ≥31 GiB，脚本化串行、失败即停、不自动重试。

1. **预检（06:43–06:45）— 成功**。远端 `ssh -p 27741 [REDACTED_EMAIL]`：在线但 `nvidia-smi: No devices were found`，PyTorch `cuda_available=False`；执行包、`jobs.json`、计划 SHA（718fa8…664937）在位且匹配；E0/E1 全部 `NOT_STARTED`，E2/E3 按交接保持 `BLOCKED`；无运行中进程。未安装、未改文件、未启动作业。[已验证，06:45:33 汇报]
2. **硬件降级（06:50）**：5090 被占，用户改授权"本机执行…都是32Gb，只不过慢点"→ 实际机器为 **RTX 4080 SUPER，32760 MiB**，空闲无进程。原始 `jobs.json` 保持不动（交接把 GPU 型号写死为 5090，执行器会拒绝 4080），另建 **4080 override plan，SHA `df062bc5c13ff96abb5c6699c13615f1b7e533bbe40904abcb13c35851ebe6fb`**，只改硬件元数据、不改代码/数据/预算/停止规则。运行时 BF16+Flash SDPA 探针通过（06:53:11）。[已验证]
3. **E0 执行（06:53 → 07:02，约 9 分钟）— 8/8 COMPLETE**：
   - `E0_native_controls`（exit 0，约 70 s）：binding / double_evidence / single_evidence 三项 `resolved=true`；deleted 控制原答案命中均为 0。
   - 教师缓存（exit 0，约 30 s）。
   - `E0_YaRN_controls`、`E0_MrPro_controls`、`E0_Z_controls` 完成。
   - 3 个四步 full cost probe（丢弃式，不算训练结果）：**YaRN 79.3 s / MrPro 79.2 s / Z 80.9 s**；每臂 146,892 input tokens、49,184 prediction tokens；**峰值 reserved ≈30.3 GB（allocated ≈29.5 GB）/32.76 GB**，无 OOM、无非有限 loss、无超时。
   - 报告目录：`/root/autodl-tmp/cross_audit_20260907/E0_report_1788764478/`。[已验证（agent 读盘汇报 receipts）]
4. **GPU 利用率诊断（06:55–06:56）— 非问题**：用户问"为啥 GPU 还没吃满"。单点 `nvidia-smi` 落在逐条生成的 kernel 间隙显示 0%；连续 5 s `nvidia-smi dmon`：SM 95–100%、235–252 W。显存 6.35/32.76 GiB 是 batch=1 逐样本评测的正常值；不改批量（会破坏冻结协议）。[已验证]
5. **E1 执行（07:02 → 07:53，约 51 分钟）— 11/11 COMPLETE**：E1_scratch → E1_Native_long（512/512 行约 530 s）→ E1_Native_native → E1_YaRN long/native → E1_MrUni long/native（07:23–07:36 心跳跟踪）→ E1_MrPro long/native → E1_Z long/native。报告 `E1_report_1788767307` + **312 KB execution_receipts**。监控自动化 `cross-audit-e0-e1-4080`（5 分钟心跳，NOTIFY/DONT_NOTIFY 决策）在 E1 完成后删除。[已验证（心跳序列）] **注意：转录中只出现执行状态，没有出现 E1 任何一臂的科学分数**（LUNA 条款：科学解释交回主研究会话）。
6. **E2 启动失败链（07:58–08:37）— 未执行，经历三轮纠正**：
   - 用户 07:58"启动啊"→ agent 拒绝绕过 BLOCKED，先跑无 GPU 预算计算：probe 推得 **共同 full_steps=1528**、单臂 5 h 上限；`/root/autodl-tmp` 可用 **22.23 GiB**；`train.py` checkpoint guard 要求单臂 **30.66 GiB**，三臂保留 >90 GiB。生成 `budget_proposal_4080.json`（SHA `a80ad9da7a6380caba405bb21b7c1eb458f8e87477d103134f12fa7e6c683828`）。
   - 用户两轮质疑（08:00"怎么可能要这么多，你搞笑吧"；08:03"1.485B FP32 不是已经有了？…最终 optimizer/resume 状态；你脑子呢"）→ agent 复核代码（`scripts/experiments/cross_audit/train.py:74` guard、`:119` retention 列表）后**承认把实现层的保守保留 guard 误说成实验硬需求**：30.66 GiB = 5.53 GiB 权重 ×（step_382/764/1528 三个 milestone + 最终 resume_state(Adam/RNG)）+ 3 GiB 保留，并非科学必需；final-only 即可。期间 SSH 一度 `connection refused`（用户切无卡模式）。
   - 用户追问"我们训练什么？"→ agent 复述配方（§5.4），确认训练对象是 **已有 OLMo-2-0425-1B-Instruct（1.485B）的 full adaptation**，AdamW 只是优化器。
   - 用户令 agent 把抱怨转发给 **Astra**（codex://threads/01a07996-6df5-7a42-a906-aa1616b7361c）并"你看着改"（08:09）。Astra 于 08:12/08:18 收敛出 **final-only 三臂 plan `jobs_e2_final_only_1528.json`（SHA `8b44d608b7837645b790e8eff7180ff5f6b32a72ead1f0de48840d2b0ac39380`，9 job）**：单臂 guard 降至 **8.53 GiB**、三臂 19.60 GiB < 22.22 GiB 可用；27 个代码 hash 匹配；全部 NOT_STARTED。
   - 08:29 范围再被纠正（经 Astra 侧）：**YaRN/MrPro 是零训练对照（复用 E1 frozen 结果），不微调；微调只保留本方路线 Z**。最终 plan：**`jobs_e2_z_only_1528.json`（SHA `433f6a006354101030df59bdcc20e72a2fe0623913e8099f397099def6b348d8`，3 job：Z full、最终 long、最终 native）**，代码 `code_own_adaptation_20260907`；旧三臂 E2 plan 标记 `BLOCKED/SUPERSEDED`。
   - 会话结束态（08:37:29）：服务器无 GPU（无卡模式）、无进程、3 个新任务 `NOT_STARTED`，等待用户开卡。**E2/E3 在本簇内从未执行。**[已验证：每步有 SHA/路径回读]
7. **速度/并行讨论（07:29–07:31）— 结论未执行**：用户两次提出吃不满/可并行。agent 判断：E1 评测逐样本是设计使然（显存 3–6 GB）；E0/E2 训练显存已近上限（29.5/30.3 reserved，16K step 加 batch 很可能 OOM）；真正瓶颈是 **checkpoint/save：每 4-step probe 计算约 33 s，保存约 44–45 s**；多进程抢单卡不建议（supervisor `gpu.lock` 设计为串行）；可行提速 = 单进程内按长度分桶 batch prefill/generation + 优化保存频率/异步写盘，但须先小对照验证、不能热改冻结中的 E1。计划"E1 完成后、E2 冻结前做吞吐探针"——本簇转录中无该探针执行记录。[部分证据：数字为 agent 读日志汇报；"提速方案有效"从未验证]

### 2.2 会话 0908a：拉取代码（2026-09-08 21:11 本地 / 09-09 01:11 UTC）

- 目标（用户）："拉取最新代码"。
- 结果——成功：分支 `main_0726_09_06`，更新 `0177e6d → 3dc3527`，工作区干净，最新提交 `research: archive RefCarry audit and CPU checks`。全程 22 秒。[已验证]

### 2.3 会话 0908b：`/root/citation_fact_check` 子代理（引用机制只读核查）

- 目标（任务体加密，由开场白还原）：按 academic-research-suite 引用核查流程，只读核查四个指定文件（`paper-2027/sections/02_related.tex`、`04_mature.tex`、`02_exponents.tex`、`refs/references.bib`），以原论文/arXiv/OpenReview 官方页为准，覆盖 8 项外部工作（FMRoPE、YaRN、LongRoPE、MrRoPE、LeRoPE、AdaRoPE、DoPE、Du et al.），只报会改变定位或公式表述的问题。
- 结果——成功（02:49 → 02:56:58，未改文件）：**无作者/年份/题名/公式硬错误，八项无遗漏**；1 处建议实改（LongRoPE 定位）+ 5 处可选精确化 + 一批确认正确项（详见 §3 C 组、§5.3）。[部分证据：agent 给出 arXiv 编号级出处，转录未含页面原文]

### 2.4 会话 0908c：`/root/numeric_evidence_check` 子代理（两个任务）

- **任务 1（02:49:26 → 03:02:18）**：按"实验身份 → 指标定义 → 数字复算"三层只读核验正文数字与落盘 evidence。结果——成功：**1 个实质正文问题（454M 比较身份错标 "YaRN-style"）+ 2 个标注/溯源问题**；其余 151.9M/MLA/750M/Llama-8B/BM/natural-QA 数字逐项核对一致（明细见 §3 D 组、§5.5）。[部分证据：逐槽数值为 agent 复算汇报]
- **任务 2（03:09:24 → 03:14:49，第二条 NEW_TASK）**：把 454M 结果按"已验证摘要层/可复现算子层"拆开恢复协议——从 `main_0726` 精确恢复算子公式、三种子训练/评测协议，输出可压成 TeX 的版本。结果——成功：给出 R_8 算子完整定义、训练配置、建议表（Geo/Geo+R8/EVQ/EVQ+R8 四臂 PK@8K/12K/16K、PPL@8K/16K）、六条精确来源路径、两处真实历史文本冲突（70.9>70.7 写反；98% vs 62% 属 seed42 s=4 早期）。[部分证据]
- 与父线程关系：09-09 04:39:28Z 主线程消息显示其吸收该核查线后另发现"750M 续训端点网格误写"（`raw_thread-main.txt` [2026-09-09T04:39:28Z]），属簇外后续。

## 3. 理论主张表

A 组 = 0907 监控执行主张；B 组 = 0907 中被纠正的主张；C 组 = 0908b 引用核查；D 组 = 0908c 数值核查。

| # | 主张 | 证据等级 | 出处 | 后续是否被纠正/推翻 |
|---|---|---|---|---|
| A1 | E0 8/8 完成、控制全部 resolved=true、probe 无 OOM/非有限 loss；峰值 reserved ≈30.3 GB | [已验证] receipts 级（E0_report_1788764478） | raw_0907-monitor [07:03:45Z] | 未被纠正 |
| A2 | E0/E1 成功只是运行资格+成本探针证据，不代表 YaRN/MrPro/Z 科学优胜 | [已验证]（agent 主动声明边界） | raw_0907-monitor [07:03:45Z] | 无 |
| A3 | E1 11/11 完成（五臂 long/native + scratch），312 KB receipts | [部分证据]（转录未见分数） | raw_0907-monitor [07:53:29Z] | 无（分数去向主线程） |
| A4 | GPU"没吃满"是采样假象：dmon SM 95–100%、235–252 W | [已验证] | raw_0907-monitor [06:56:11Z] | 无 |
| A5 | 评测低显存（3–6 GB）是 batch=1 设计使然；训练臂显存已近上限，加 batch 会 OOM；保存（44–45 s）> 计算（33 s）是 probe 的主要可优化瓶颈 | [部分证据]（probe 读数；"会 OOM"是外推） | raw_0907-monitor [07:30:46Z] | 提速方案未执行、未验证 |
| A6 | 单进程按长度分桶 batching 是合法提速路径，多进程抢单卡不合法（gpu.lock 串行） | [假设]（未做 benchmark） | raw_0907-monitor [07:31:13Z] | 计划"E2 冻结前吞吐探针"未见执行记录 |
| B1 | ~~E2 需要把数据盘扩到 ~100 GiB 或授权 offload 才能启动~~ | 已被推翻 | raw_0907-monitor [08:00:09Z] | 用户 08:00/08:03 两轮质疑后，agent 08:02–08:04 承认把 train.py 实现层保守 guard（milestone×3 + Adam/RNG resume + 3 GiB margin）误报成实验硬需求；final-only 后单臂仅需 8.53 GiB |
| B2 | ~~base 模型权重需要计入磁盘预算~~（"1.485B FP32 ≈5.53 GiB 还要这个干嘛"） | 已被推翻 | 同 B1 | base 只读加载不复制；预算只算训练后产物 |
| B3 | E2 三臂（YaRN/MrPro/Z）都要 full adaptation | 被范围纠正推翻 | 旧 plan 8b44d608 | 08:29 起：YaRN/MrPro = 零训练 frozen 对照（复用 E1），只微调 Z → Z-only plan 433f6a00 |
| C1 | LongRoPE ≠ 纯静态 exponent table：其维度缩放部分满足 d_k=log λ_k，但完整方法有 token-position threshold n̂（前保持原 RoPE） | [已验证]（arXiv 2402.13753 Eq.3） | raw_0908b [02:56:58Z] | 建议实改 02_related.tex:13-15 |
| C2 | YaRN NTK-by-parts ω'_k=ω_k[(1−γ_k)/s+γ_k] 与稿中 d_k=−log(1−w_k+w_k/s)（w_k=1−γ_k）完全等价；完整 YaRN 另含 attention-temperature scaling（稿中表 g 已体现） | [已验证]（arXiv 2309.00071 Eq.13/Def.2） | raw_0908b | 建议措辞"uses the first construction" |
| C3 | LeRoPE 学的是"每 rotary pair 一个 log-space scale，跨层跨头共享"（θ̂_m=e^{α_m}θ_m）；shared 粒度优于 per-layer/per-head；fixed-frequency ablation 保留约 63% 增益 → 稿中"table 与 joint training 都有贡献"正确 | [已验证]（arXiv 2607.10134） | raw_0908b | 可选精确化 |
| C4 | DoPE 不是静态 exponent-table 调整：低频成分↔over-aligned low-rank attention，truncated matrix entropy 选头、选择性抑制位置编码、可用 isotropic Gaussian 重参数化 | [已验证]（arXiv 2511.09146） | raw_0908b | 稿句"不假但过于模糊" |
| C5 | MrRoPE 映射稿中完全正确：ω'_k=ω_k/∏_{j<k}λ_j ⇒ d_k=∑_{j<k}log λ_j | [已验证]（arXiv 2601.22181v1 Eq.10） | raw_0908b | 无 |
| C6 | 02_exponents.tex:7 ω_k=b^{−k/K} 与标准 RoPE b^{−2k/d_rot} 一致；固定 base 几何表归一化后 z_k=k/(K−1)；FMRoPE 改 base 只改 log-frequency 范围，归一化 spacing 仍均匀 → fixed-range interior-allocation 区分成立；AdaRoPE 学 per-head/per-block 频率+head-specific length-aware scaling 属实；references.bib 八项元数据与官方一致（DoPE 用 2025 首提年、AdaRoPE 标 ICML 2026） | [已验证] | raw_0908b | 无 |
| D1 | **454M 正文比较身份错误**：04_experiments.tex:80 写 "YaRN-style scaling" 实为仓库自定义 fixed-index s=8 range operator，非 HF 官方 YaRN；157.7/107.5 数值与三种子身份正确；a2_experiment_details.tex:83 承诺的协议不存在、appendix 指针不成立 | [已验证]（official_yarn.py:331-373 明示 mscale=1.0/gain=1.0/禁称 official YaRN） | raw_0908c [03:02:18Z] | 任务 2 给出保留方案（改名+恢复协议），非删除 |
| D2 | 454M 结果可以保留：证据层级 = validated historical report + tracked curated aggregate + tracked operator implementation；本地缺 raw mirror 只限制逐样本重算，不构成 summary 矛盾 | [部分证据]（远端原始路径未保留是历史事实） | raw_0908c [03:14:49Z] | 无 |
| D3 | make_exponent_revision_figures.py 硬编码 BM 公式与 N=18/17，source receipt 未纳入定义 owner `docs/research/ROPE_MRPRO_BM_CANDIDATE_20260908.json`；数值核验正确（OLMo (l,h,N)=(14,32,18)、Qwen (23,40,17)、两臂 gain 均 1.1386294361） | [已验证]（逐槽复算） | raw_0908c | 建议 SOURCES 加 owner |
| D4 | 指标称呼：部分得分（FWE/VT）不能读作 "task accuracy"，应写 official RULER task score (%)；natural-QA 表 F1×100 列名应标 F1 (%)；正文用 "RULER macro score" 统一 | [已验证]（脚本读数） | raw_0908c | 措辞级修正 |
| D5 | 151.9M 三训练种子 +0.026/−0.281/−0.176/−0.146；target-retargeted +0.060/+0.227/+0.460；MLA 三种子 138.8→95.6、8K 35.4/35.8；750M=seed42、40 passkey trials、77.5% 为 greedy strict AR exact match（teacher-forced retrieval 本身 100%/100%）；Llama-8B 24 packs、hit@16 18.75%→64.06%、gold-block deletion −0.0095/+1.5055；g=1.0513（normalized-index）≠1.0693（YaRN）；BM 样本数 24/48、12/24、6/12；natural-QA n=631、task-equal F1 21.62%→25.44%、CI [1.32,6.29]、五任务方向均正 | [部分证据]（agent 逐项复算汇报，未附原始 JSON） | raw_0908c [03:02:18Z] | 无（均判为一致） |
| D6 | full-sequence PPL 253.2/229.5 与 per-document PPL 262.0/237.2 不是冲突，是聚合口径不同；报告 "70.9 < 70.7" 写反（应为 70.9>70.7，差 0.2）；"98% vs 62%" 是 seed42、s=4 早期结果，不能混进 s=8 三种子主比较 | [已验证]（对照 2026-03-03 报告行号） | raw_0908c [03:14:49Z] | 属"已验证摘要层"内部的更正 |

## 4. 失败机制清单

1. **实现层 guard 被复述成科学硬需求（0907，最重要）**：E2 磁盘判定把 `train.py` 的保守保留策略（milestone checkpoint×3 + Adam/RNG resume_state + 3 GiB margin → 30.66 GiB/臂、">90 GiB"）说成实验本身必需，触发用户"你搞笑吧/你脑子呢"两级反弹才回查代码修正。复盘要点：**引用需求数字前先分清"代码 guard"与"实验必需"两层，并给出参数化算式**（5.53 GiB 权重、3 milestone、resume、margin）。复发风险：任何 retention/预算/超时数字被当作不可协商约束转述时同型。
2. **过度发散的配置沉积（Astra 侧，0907）**：交接方案层层加码（保留这个那个权重、发散设定）导致"实验基本无法做"（用户原话），需用户人工下令收敛。final-only → Z-only 两轮裁剪才落地。复发模式：计划 agent 与执行 agent 分层时，计划侧的保守默认会吃掉执行窗口。
3. **单点 nvidia-smi 采样误导（0907，已解决）**：kernel 间隙采样读出 0% util，被当成"没吃满"；连续 dmon 5 s 才见 95–100%。教训：**GPU 占用判断必须用连续采样**，单点读数不作证据。
4. **比较身份错标（0908c）**：仓库自定义算子在正文被冠名 "YaRN-style"，且 appendix 指针指向不存在的协议。与项目铁律同族（代理指标/身份冒充）。修正方式已给出：改名 "repository fixed-index range operator" 并恢复协议。复发风险：任何 legacy 算子复用官方方法名。
5. **代理指标命名漂移（0908c）**：部分得分被图表标成 "task accuracy"、F1×100 不带 % ——把聚合口径读成逐例 exact 的风险，正是"不得把代理指标说成能力结果"铁律的文档面。
6. **外部资源中断（0907，非实验失败）**：5090 被占用 → 现场降级 4080 SUPER（等显存、更慢），交接 GPU 合约写死 5090 导致执行器直接拒绝，靠 override plan（只改硬件元数据、SHA 留痕）合规绕过；SSH 在切无卡模式时 connection refused。队列零任务失败（E0/E1 无一臂 retry）。
7. **未闭环的提速探针（0907）**："E1 后、E2 冻结前做 batching/save 吞吐探针"被明确规划，本簇无执行记录 → 属"计划未执行"，不得写成已否证或未验证结论。
8. **任务体加密的审计盲区（0908b/c，记录性）**：codex 多代理 NEW_TASK payload 为 encrypted_content，子会话的任务边界只能从其开场白/报告反推——统一理论重建时间线时，凡引用这两个子会话的"任务要求"必须标注为还原而非原文。

## 5. 频率表/方法定义清单

### 5.1 cross-audit 阶段与臂（0907）

| 名称 | 构造/规则 | 执行状态 | 得分 |
|---|---|---|---|
| E0 | 串行：Native 控制 → 教师缓存 → YaRN/MrPro/Z 控制 → 3×四步 full cost probe；硬超时上限 3.25 GPU-hour | 8/8 COMPLETE（约 9 min） | probe 79.3/79.2/80.9 s；146,892 in / 49,184 pred tokens/臂；reserved ~30.3 GB |
| E1 | 151M 新叠加 + 五臂（Native/YaRN/MrUni/MrPro/Z）OLMo-1.485B 冻结长生成 + 分层 native；上限 8.5 GPU-hour | 11/11 COMPLETE（约 51 min） | **转录未报告任何科学分数** |
| E2 | Z full adaptation seed137 **1528 steps**（final-only retention） | NOT_STARTED（无卡模式，等待开卡） | — |
| E3 | 匹配 all-linear LoRA r16 桥接 | BLOCKED（未冻结） | — |
| E4/E5 | 条件阶段 | 不自动运行 | — |

### 5.2 执行侧计划文件（0907，均 [已验证] 有 SHA 回读）

| 文件 | SHA256 | 命运 |
|---|---|---|
| jobs.json（原始，5090 合约） | 718fa8a9…664937 | 未改动，E0/E1 以其为基 |
| 4080 override plan | df062bc5c13ff96abb5c6699c13615f1b7e533bbe40904abcb13c35851ebe6fb | E0/E1 实际执行用 |
| budget_proposal_4080.json | a80ad9da7a6380caba405bb21b7c1eb458f8e87477d103134f12fa7e6c683828 | 1528 steps 预算计算 |
| jobs_e2_final_only_1528.json（三臂 9 job） | 8b44d608b7837645b790e8eff7180ff5f6b32a72ead1f0de48840d2b0ac39380 | SUPERSEDED |
| jobs_e2_z_only_1528.json（3 job） | 433f6a006354101030df59bdcc20e72a2fe0623913e8099f397099def6b348d8 | 最终冻结方案，未启动 |

### 5.3 外部方法的频率/指数映射定义（0908b 核查确认）

- YaRN：ω'_k = ω_k[(1−γ_k)/s + γ_k] ≡ d_k = −log(1−w_k+w_k/s)，w_k=1−γ_k（+温度 scaling）。
- MrRoPE：ω'_k = ω_k/∏_{j<k}λ_j ⇒ d_k = ∑_{j<k}log λ_j。
- LongRoPE：维度缩放 d_k=log λ_k **且** token-position threshold n̂（阈值后生效）→ 非单一静态 exponent table。
- LeRoPE：θ̂_m = e^{α_m}θ_m，每 rotary pair 一个 log-scale，跨层跨头共享。
- 标准 RoPE：ω_k = b^{−k/K}（= b^{−2k/d_rot}）；固定 base 归一化 z_k=k/(K−1)；FMRoPE 改 base 仅平移 log-frequency 范围。

### 5.4 E2 训练配方（0907 转录原文核对，OLMo full adaptation）

- 对象：已有 **OLMo-2-0425-1B-Instruct（≈1.485B）** 微调（非 from-scratch，非"训练 AdamW"）。
- 优化器 AdamW：lr=2e-5，betas=(0.9, 0.95)，weight_decay=0；BF16 计算 + FP32 master weights。
- 每步 = 1 条 8K/16K 交替 CPT prefix（CPT 语料 33.55M tokens，重复暴露）+ 1 组 near/far SFT pair + 1 条原始 Native replay；损失 = CPT CE + SFT CE + Native KL，权重 1/1/1。
- retention（final-only 后）：steps.jsonl + 最终 step_1528 权重 + tokenizer + manifest；不保留中间 checkpoint、不存 optimizer/RNG resume、不复制 base、不删旧资产。

### 5.5 仓库自定义 R_8 算子与 454M 表（0908c 任务 2 恢复，[已验证] 自 main_0726 精确出处）

```
a=⌊0.20K⌋, b=⌊0.90K⌋, u_k=clip((k−a)/(b−a),0,1), r_k=u_k²(3−2u_k)
R_s(ω_k) = ω_k / ( s^{r_k} · T(s)^{r_k/2} ),  T(s)=1+0.07·log₂ s
454M 比较：K=32, s=8 → (a,b,T)=(6,28,1.21)；g=1、无 attention mscale —— "YaRN-style construction，非官方 YaRN"
```
- 基座：24 层、width 1024、16 heads、d_head 64、base=500,000；Geo(τ=0) 与 EVQ-Cosh(τ=1.5) 用 midpoint 频率约定，from-scratch 100M tokens/臂、L_train=2048、FineWeb-Edu+10% synthetic passkey、seeds 42/123/7；R_8 推理期施加、不再训练。
- 表（三种子均值；PK@8K 附 std；PK = teacher-forced NLL-gap retrieval；PPL = full-sequence scoring）：

| Method | PK@8K | PK@12K | PK@16K | PPL@8K | PPL@16K |
|---|---|---|---|---|---|
| Geo | 41±5% | 57% | 51% | 161.9 | 253.2 |
| Geo+R_8 | 61±3% | 59% | 51% | 82.9 | 157.7 |
| EVQ | 53±8% | 63% | 50% | 150.3 | 229.5 |
| EVQ+R_8 | **100±0%** | 79% | 68% | 70.9 | 107.5 |

- 来源：`main_0726:data/curated/table2_evq_yarn_454m_passkey_10pct.json`（summary owner）、`main_0726:docs/exp/2026-03/2026-03-03_passkey_mix_results.md`（第 1–8/80–98/187–207/247–255 行；本地 raw mirror 未保留）、`main_0726:scripts/core_text_phases/eval_pe_baselines.py:66-94`、`main_0726:scripts/lib/rope/official_yarn.py:331-373`、`main_0726:paper-2027/tables/table_evq_ramp.tex`；`phase14c_multiscale_evq_yarn.py`（50M/125M、5% mix）只是 supporting reproduction；`quality_454m_full_eval.json` 不能支持 157.7/107.5。
- **本簇转录中没有 32K/128K RULER 得分**：09-07 监控只报执行状态，E1 分数与任何 32K/128K 数字不在这些会话里。

## 6. 用户指令与纠正（原文引用）

0907-monitor（raw_0907-monitor.txt）：
- [06:42:49Z] "阅读执行交接：…LUNA_EXECUTION.md 以及服务器基本信息：ssh -p 27741 [REDACTED_EMAIL]， 后续我会升级卡为5090，你严格按照交接迅速执行实验，监控实验即可"
- [06:43:43Z] "目前还没升级卡，你先了解清楚，准备好执行后，我会升级卡"
- [06:48:07Z] "你不用确认，我还会给你假货吗，你稍等，马上开机"（→ agent 停止重复资格确认，直接启动）
- [06:50:17Z] "有点尴尬，5090被用了，你就在本机执行吧，我开卡了，一样的，都是32Gb，只不过慢点"（4080 SUPER 授权）
- [06:55:17Z] "为啥GPU还没吃满？"
- [07:02:30Z] "e0结果如何？"
- [07:11:19Z] "预计时间要多久？"
- [07:17:22Z] "后续还有其他实验吧，整体下来要执行多久呢？"
- [07:29:34Z] "我感觉目前实验配置没有吃满GPU以及显存，后续试验可能可以提速很多"
- [07:30:48Z] "那你可以并行啊，1如果评测的话，是不是会更快一点的"
- [07:58:35Z] "启动啊"
- [08:00:33Z] "怎么可能要这么多，你搞笑吧"
- [08:03:52Z] "你在搞笑吗？ 你训练什么模型？1.485B 个 FP32 参数 ≈ 5.53 GiB； 这个不是已经有了？ 你要这个干嘛？最终 optimizer/resume 状态； 你脑子呢/"
- [08:07:41Z] "我真服了，astra写的太多了，我们不是AdamW吗？我们这个训练是训练什么？"
- [08:09:07Z] "你把问题发给astra：codex://threads/01a07996-6df5-7a42-a906-aa1616b7361c 告诉它别那么蠢，不要保留这个那个的权重，过度发散这些逆天的设定和行为导致实验基本无法做 我先无卡模式开机了ssh -p 27741 [REDACTED_EMAIL] 你看着改"

0908a：
- [01:11:55Z] "拉取最新代码"

0908b/0908c：无直接 user 消息——指令由父会话以加密 NEW_TASK 下发（见 §1/§4-8）。

治理上下文（两条 user environment 消息中的 AGENTS.md 差异，[已验证]）：09-07 生效的项目 AGENTS.md 是"Plan before experiments / 授权边界"长版；09-08 起改为六条版，第 2 条 "**Act without unnecessary gates** … do not invent time limits, approval steps, or workflow constraints"、第 5 条 "Use history as evidence"——正对治 09-07 观察到的过度 gating 与保守 guard 沉积。

## 7. 未决问题

1. **E1 五臂科学分数从未出现在本簇任何转录**（07:53 后只留 receipts 指针 `E1_report_1788767307`，在远端 /root/autodl-tmp，本机无镜像）——E1 的 long/native 各臂表现、E0/E1 是否改变 YaRN/MrPro/Z 定位，需回主研究线程或远端取证。
2. **Z-only E2（1528 steps，seed137）在本簇结束时仍未执行**；后续是否开卡启动、`jobs_e2_z_only_1528.json` 是否被再改，本簇无记录（仓库 grep 无命中）。seed256/E3（LoRA 桥接）始终未启动。
3. **吞吐优化探针计划（评测 batching / CPU→GPU staging / 异步保存）从未执行**——"E1 低显存评测可批量化提速、保存占 probe 时间 57%"两条都停留在 [假设]/[部分证据]。
4. **E2 预算外推未验证**：full_steps=1528 由 4-step probe 线性外推 + 5 h/臂上限，16K 长步的真实显存/时间行为（29.5→是否 OOM）未测——"加 batch 很可能 OOM"是外推。
5. **454M raw 逐 seed 重算不可行**（远端 raw mirror 未保留）——157.7/107.5 只能停留在"validated summary + curated aggregate + tracked implementation"层级；若审稿要求逐样本 reproduction，此为已知硬缺口。
6. **0908b/c 的核查结论是否已全部落进 TeX**（LongRoPE 阈值句、"YaRN-style"改名、BM SOURCES、指标命名、R_8 附录表）——子会话只读不改，修复动作发生在父线程；本簇只能确认发现，不能确认合入。
7. **0819 空会话被打断的原因不明**（可能人为 Ctrl-C；与当日 full-rope collision audit 的关联无法从转录证实）。
8. **Astra（01a07996）侧的收敛过程与决策依据不在本簇转录内**：final-only、Z-only 两轮裁剪是 Astra 先出方案、monitor 回读核验——统一理论时间线需要 Astra 线程补全"谁先提出 Z-only 范围纠正"（0907 [08:29:26Z] 只说"范围刚被进一步纠正"，未指明发起者）。
9. 监控自动化（heartbeat `cross-audit-e0-e1-4080`）删除后 E2 阶段无监控方案；`--poweroff-after` 从未被执行验证（LUNA 原文即声明）。
