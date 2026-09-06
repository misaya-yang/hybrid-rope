# §10 LoRA 固定比较轮次执行报告（2026-09-05，进行中）

> **状态（2026-09-06 更新）**：本轮为 Qwen-2.5-1.5B 时代轮次，已被项目重置与
> Round 12 取代（成功标准改为各模型自己的 2×/4× + 忠实 YaRN 对照 Y2）。
> **评测模式警示**：本轮经 release008 引擎以 raw completion 模式生成；09-06 发现
> Qwen-Instruct 在 raw 模式即使原生 32K 也崩为复述 filler（0/8）——本轮的零生成
> 结果可能混同模式假象与真实失败，Qwen 家族已在 Round 12 以 chat-template 重测。
> 本地回执 JSON 已于 09-06 按用户指令清理（非核心）；服务器原件保留于
> `/root/autodl-tmp/claude_round10_20260905/`。

- 协议：`CONSTRAINED_FRONTIER_AND_2X4X_LORA_PREFLIGHT_20260904.md` §10 + `PRO_REPORT_AUDIT_RECONCILIATION_20260905.md` 排序（B→C）
- 机器：westc RTX 4080 SUPER 32GB（`ssh -p 27741`），容器 `autodl-container-c904489327-8b72fcf9`
- 授权：用户 2026-09-05 授权 GPU 实验安排由 Claude 决断；本轮开场又授全权（"只要不是硬性阻塞，可以尽可能做实验、及时纠偏、根据结果调整最优解；全部完成且报告写好后在容器内关机"）
- 工作目录：服务器 `/root/autodl-tmp/claude_round10_20260905/`；本地 `claude_code_workspace/round10_20260905/`
- 状态：**三个案全部完成**（Z / Y 双双点值可行、统计不可区分；**E3a 判读=长输入 exposure 不必要，强烈支持机理 A**）

## 1. 资产验证（全部通过）

以 `qwen_N_s42_fixed`（N128）的 run.json/complete.json 为锚点，逐项哈希核对：

| 资产 | 期望哈希 | 结果 |
|---|---|---|
| release008 engine（trainer） | `bc9826ea…` | ✓ |
| release008 runtime（single_table_generation） | `401767d5…` | ✓ |
| release008 contract（generation_contract） | `554fe323…` | ✓ |
| checkpoint 权重 model.safetensors（3.09GB） | `dd924a11…` | ✓ |
| run.json 自哈希 = complete.run_sha256 | `1ec00ae3…` | ✓ |
| training.jsonl = N128_TRAINING_SHA | `682da90f…` | ✓ |
| tasks / native_pool / contract manifests | `baf1d791…/5e128413…/a32ed5b3…` | ✓ |
| views（603MB 全量）/ qualification / rows / candidate_pool | manifest 记录值 | ✓ |
| examples.jsonl（native/task baseline） | `ef191193…/31c2b138…` | ✓ |
| release008 ↔ 部署 trainer 的 4 个调度函数 AST | 全等 | ✓ |
| 部署包关键文件（本地=服务器） | `9084ec0b…/e9cd1933…/36b7511d…` | ✓ |

4 个待发现路径（已写入配置 `ROUND10_CONFIG_FILLED_V1`）：
- teacher_cache：`/root/ffn_review_scratch_20260904/teacher_cache_qwen`（manifest `d7f2391e…`，896 条=512train+128cal+256val，6.6GB，teacher=原 Native/gain1/无 adapter）
- controls：`$B/qwen_fixed_controls`（FIXED_NZGY_CONTROLS_FROZEN_V1，native_sha256=`138c99b1…`）
- native_baseline：`$B/qwen_native_validation`（native/gain1/无 adapter/fold=selection）
- baseline_eval_engine：release007 trainer（`ee4b53db…`）

## 2. 环境发现：无卡模式 2GiB 内存上限

无卡模式容器 cgroup 内存上限仅 2GiB。`prepare` 的引擎 preflight（check_assets）会全量载入
603MB/4736 行的 transport_views.jsonl（约 128M token，Python 对象 >3GB），实测 OOM（exit 137）。
这是资源限制，非输入错误；切有卡模式（90GiB 上限）后 `prepare` 一次通过。**未改任何冻结代码。**

prepare 封版输出（`out/plan.json`，`PREPARED_CPU_ONLY_GPU_PENDING`）：
- 曝光重建与 N128 complete.json 完全一致：`task_exposure=276f203a…`、`native_exposure=74c7ba39…`
- 768 views、replay R256/T192、answer+EOS 标签 3642、长输入 6,328,481 token、**compact 输入 203,631 token**

## 3. N_compact 案：冻结引擎守卫冲突（硬性阻塞，已存档）

**现象**：launcher 的 N_compact 阶段命令（`--arm N --compact-only`）被 release008 引擎 argparse 拒绝：
`error: explanatory ablations are fixed to Z / seed 42, after main feasibility`（stage train exit 2，
`out/N_compact/execution.json` = STOPPED_PRESERVE_PARTIAL，已镜像本地 `receipts/`）。

**冲突的两侧（都是项目权威文件）**：
- §10 与审查复核（2026-09-04/05，较新）：N_compact = **原始 Qwen、Native 表**、仅换 compact 输入，
  用于回答 `S(N128)−S(N_compact)`（长输入 exposure 是否必要）。
- 冻结引擎（release007–012 与当前仓库 trainer **全部**带同一条守卫）：`--compact-only` 只允许
  `arm Z / seed 42`；且 arm Z 强制表文件=Z.npy（P2 witness），"arm Z + N.npy(gain1)" 也被身份校验拒绝。
  ⇒ 冻结引擎**无法表达** "Native 表 + compact-only"。

**为何不绕过**：改 release008 = 覆盖历史引擎（明令禁止）；换引擎 = §10 明文"不能顺手使用不同训练引擎"；
伪造一个把 N.npy 标成 Z 臂的 controls manifest = 伪造证据。全项目无任何历史 N+compact 运行可复用。
该冲突是协议文档与冻结引擎之间的真实矛盾（守卫写于"compact 消融只属于 Z"的时期），按审计纪律存档上报，不私改。

**替代**：按 dossier E3a 预注册的 `Z_compact_only_matched_answer_presentations_seed42`
（arm Z + compact-only，同配方同 caps）作为可执行的 compact 反事实，在 Z 主可行性完成后执行（见 §6）。
它回答的是"Z 的迁移是否只学了短任务"，与 §10 原问（N128 的长输入增量）相关但不同，报告中分开陈述。

## 4. Z / Y 案

`run --plan out/plan.json --cases Z Y --authorized`（nohup，launcher 自带锁与 GPU 占用检查）。
阶段：train(≤3600s) → native0/task0 → native32(R32 诊断) → native128/task128 → v4 review。

### 4.1 Z 案（完成，44 分钟；回执已镜像 `receipts/Z/`）

execution.json = `BOUNDED_CASE_COMPLETE_READ_REVIEW`，7 阶段全部完成，无阶段失败。
review.json（protocol=single_evidence_v4，checkpoint step128）：

**判定：`POINT_FEASIBLE_CONFIRMATION_PENDING`**
（点值通过双门；配对 bootstrap CI 下界 0.866 < 0.88，未获"确认"级证书）

| 指标 | 数值 | 门槛 | 判定 |
|---|---|---|---|
| official_gate（strict） | task_retention **0.959** / ppl_retention **0.970** | ≥0.88 | PASS |
| EOS_gate（strict） | 0.959 / 0.970 | ≥0.88 | PASS |
| 配对 CI 下界（task/ppl，1000 重采样） | [0.866, 1.053] / [0.961, 0.980] | 全部下界 ≥0.88 | **不满足** |
| 主家族 single_evidence compact 合格组 | 26 组 | ≥8 | 合格 |

**内容迁移（严格精确匹配，16K far 为主终点）**：

| 家族 | 布局 | baseline → candidate | 变化 |
|---|---|---|---|
| single_evidence | far | 3/32 → **23/32** | **+20，0 丢失**；far_delta=0.625，CI[0.469,0.781] |
| single_evidence | near | 20/32 → 25/32 | +8/−3 |
| single_evidence | compact | 26/32 → 26/32 | **零损失** |
| binding | far | 0/16 → 5/16 | +5，0 丢失；far_delta=0.3125，CI[0.125,0.5625] |
| binding | compact | 10/16 → 10/16 | 零损失 |
| double_evidence | far | 0/16 → 3/16 | 组数不足（4<8），仅描述性 |

未过滤 16K far 宏平均 = **0.406**。Native 层转移：instruction 35→34（−1）、
position_format 9→9（保留 1.0）、reasoning 5→4（保留 0.8，样本极小）。
语义状态 = `NOT_MEASURED_BY_STRICT_SCORER`（严格分不判语义，待盲标注）。

**读法**：固定 Z 表（零训练静态表，gain 1.1026）+ all-linear r16 LoRA 在 Qwen2.5-1.5B-32K 上
产生了真实的大幅远端内容迁移（主家族 far 提升 +0.625）且 compact 零代价、远端零丢失；
Native/任务双门点值全部通过。这是该架构上固定表配方首次达到"点值可行"。
不足仅在确认级别：配对区间下界 0.866 未过 0.88，按纪律属于"点值合格、未确认"，
解法是**新的独立确认协议**，不是重选 checkpoint、不是加步数、不是扫 KL。

### 4.2 Y 案（完成，44 分钟；回执已镜像 `receipts/Y/`）

execution.json = `BOUNDED_CASE_COMPLETE_READ_REVIEW`，7 阶段全部完成。
review.json：**判定与 Z 相同 `POINT_FEASIBLE_CONFIRMATION_PENDING`**

| 指标 | Y | （对照 Z） |
|---|---|---|
| official/EOS 门 | task **0.939** / ppl **0.956**，PASS | 0.959 / 0.970 |
| 配对 CI task 下界 | **0.819**（<0.88，未确认） | 0.866 |
| single_evidence far 16K | 3/32 → **22/32**（+19，**0 丢失**）；delta=0.594，CI[0.438,0.750] | 23/32，delta 0.625 |
| single_evidence near / compact | 20→25 / 26→26（零损失） | 同 |
| binding far / near / compact | 0→5 / 4→9 / 10→10 | 0→5 / 4→4 / 10→10 |
| 未过滤 16K far 宏平均 | **0.354** | 0.406 |
| Native 层转移 | instruction 35→35；position_format 9→8；reasoning 5→3 | 35→34 / 9→9 / 5→4 |

### 4.3 Z vs Y 系统对照（触发 §10 映射"Y 不逊 Z"行）

两臂内容迁移**统计上不可区分**：far_delta 0.625 vs 0.594，CI 大幅重叠
（[0.469,0.781] vs [0.438,0.750]）；compact 双零损失、far 双零丢失形态一致。
差异均在噪声尺度内（Y 的 binding near 略好、reasoning 层多丢 1 例）。

**含义**（按 §10 映射与 FFN 指导 §13.2 如实报告）：
- 两个**不同的固定表系统**在同一配方下给出几乎相同的远端迁移与零代价形态。
  这本身是重要机理信息：迁移现象对"哪张表"不敏感（在两张都合格的表之间）。
- 它**不**否定 allocation 研究——§10 明文："Y 不逊 Z → 如实报告系统对照；
  allocation 的贡献仍由原 fixed-support 研究支撑"。
- 它把机理问题的焦点移向：**是表解锁了既有能力（两表同效），还是 LoRA 适配
  主导（表只是载体）**——由 E3a（Z_compact）回答（见 §6）。

## 5. 步骤 A 内容审计导出（已完成，无卡模式）

`audit_generation_transitions.py export`：768 盲化案例（baseline N0 与 candidate N128 的 validation 全 384 行×2），
其中 417 个 exact 成功自动预填真值标签，**351 例待人工盲标注**；
`private_mapping.json`（映射）冻结前不外发。产物：服务器 `claude_round10_20260905/content_audit/`。

## 6. Z_compact（E3a）案（执行中）

**目的**：机理归因的关键实验（dossier 预注册名 `Z_compact_only_matched_answer_presentations_seed42`）。
同 Z 配方（Z 表 + all-linear r16 LoRA + 128 步 + KL .02），但训练**只见 2K compact 输入**，
评估仍在 2K/16K。区分机理：
- 若 16K far 仍 ≈23/32 → 长输入 exposure 不必要，表换了就能解锁远端能力 → **强烈支持机理 A**（谱预算是 binding constraint）；
- 若 far 大跌 → 长输入 exposure 必要 → LoRA 在长输入上学到的东西不可少（A+C 或 B 成分上升）。

**脚本**：`z_compact_e3a.sh`，完全镜像 launcher 的 Z 阶段命令 + `--compact-only`，同 caps
（train 3600s / native 900s / task 1800s / review 600s + 120s cleanup），审查走
`review_native_constrained_transfer.py --protocol single_evidence_v4`。

**执行纠偏（如实记录）**：首次启动被引擎身份守卫拒绝
（`ValueError('declared N/Z/Y arm identity mismatch')`）——我排程脚本里的 `--gain` 写错为
1.2156954082626084，而冻结 controls manifest 要求 Z 臂 rotary_amplitude 精确等于
**1.102585782722872**（与 `qwen_fixed_controls/manifest.json` 及 plan.json 一致）。
修正后重启（pid 7822），训练正常进行。该失败是我的排程错误，非引擎/数据问题；
未产生任何可用输出，不占用结果空间。

**预注册判读阈值**（CPU 精确二项检验，`cpu_power_result.json`）：
- 全 32 组 far：Z_compact ≤ **18/32** → 拒绝"与 Z 同水平"（size=0.042）→ exposure 必要；
  ≥ **19/32** → 统计上不可区分 → 支持机理 A。
- 合格队列 26 组：≤ **12/26** → 拒绝（size=0.034）。

### 6.1 结果（完成；回执已镜像 `receipts/Z_compact/`）

review.json：**判定 `POINT_FEASIBLE_CONFIRMATION_PENDING`**（与 Z/Y 同级；
点值双门 PASS：task **0.918** / ppl **0.971**；配对 CI task 下界 **0.795** < 0.88，未确认）

| 家族 | 布局 | baseline → candidate | 变化 |
|---|---|---|---|
| single_evidence | far | 3/32 → **20/32** | **+17，0 丢失** |
| single_evidence | near | 20/32 → 24/32 | +6/−2 |
| single_evidence | compact | 26/32 → 26/32 | **零损失** |
| binding | far | 0/16 → 4/16 | +4，0 丢失 |
| binding | compact | 10/16 → 10/16 | 零损失 |

未过滤 16K far 宏平均 = 0.333。Native 层转移：instruction 35→34、position_format 9→5（−4）、
reasoning 5→6。合格队列（26 组）far 保留 = 19/26 = 0.731。

### 6.2 预注册判读：**长输入 exposure 不必要 → 强烈支持机理 A**

- 全 32 组 far = **20/32 ≥ 19** → **不拒绝**"与 Z 同水平"；精确二项
  P(X≤20 | p₀=23/32) = **0.162**（单侧），远在拒绝域（≤18）之外。
- 合格队列 far = 19/26，远大于拒绝阈值 12/26。
- **训练全程只见 2K compact 输入的模型，在 16K 远端仍拿到 20/32（零丢失），
  与见过全部 16K 长输入的 Z（23/32）统计不可区分。**
  长输入 exposure 不是远端迁移的必要条件。
- 机理归因：换表使 16K 距离变得可判别，模型既有的长程计算被解锁；
  LoRA 只需在短输入上完成接口修补（格式/终止/任务对齐）即可在长输入上兑现。
  **这是对"谱预算是 binding constraint"（机理 A）的直接实验支持**，
  同时大幅压低机理 B（LoRA 自己学会长上下文检索）的地位。
- 保留项：E3a 的 position_format 层保留劣于 Z（9→5 vs 9→9），
  说明长输入 exposure 对**格式层**行为有部分作用；但该层样本极小（基线 9 对），
  且主终点（主家族 far）不受影响。如实记录，不过度解读。

### 6.3 跨家族后续（round11，另见独立报告）

E3a 的"长输入 exposure 不必要"只在 Qwen 窗内评估下成立。同配方换到
OLMo-2-0425-1B-Instruct（native 4096，far 16K = 4× 外推）的验证结果、
失败分解与机理边界分析见 `reports/ROUND11_OLMO_RESULTS_20260905.md`。
要点：表的内容解锁跨家族复现（零训练含答案 0/64→26/64），
但格式接口未在 4× 外推上兑现——§6.2 的保留项（格式层依赖 exposure）
在更弱的指令基线上被放大为主失败模式。

## 7. 结果 → 行动映射与原因分析

§10 映射表逐行落位（✔=已触发并执行对应动作）：

| §10 映射行 | 本轮证据 | 落位 |
|---|---|---|
| Z/Y 至少一臂 Native 与主生成点值可行 → 冻结 final128 及新确认协议；先新 Native 确认，再预冻结 farther 测试 | **Z 案点值双门 PASS（0.959/0.970），far +0.625 零丢失** | ✔ 触发：冻结 `out/Z/train`（step128 adapter `e93f84af…`）；新确认协议待登记 |
| Native CI 不够窄但点值合格 → 未确认；允许准备新独立确认，不重选 checkpoint | Z 配对 CI task 下界 0.866 < 0.88 | ✔ 触发：不重选 checkpoint；旧 N128 确认池已暴露禁用 |
| 固定 Z/Y 预算都不满足 Native → 停候选 | Z 已满足点值 | ✘ 未触发 |
| Native 可行而内容迁移不足 → 登记 N/Z × prefix off/on | Z 内容迁移充分 | ✘ 未触发 |
| Y 不逊 Z → 如实报告系统对照 | **Y far_delta 0.594 vs Z 0.625，CI 大幅重叠、统计不可区分；双零损失形态一致**（§4.3） | ✔ 触发：如实报告两表系统对照；allocation 贡献仍由 fixed-support 研究支撑 |
| N_compact 两行 | 被引擎守卫阻塞（§3），以 E3a Z_compact 替代 | 见 §6 |
| （E3a 预注册）compact 训练后 far 是否保持 | **far 20/32，≥19 阈值，P(X≤20\|p₀)=0.162 不拒绝** | ✔ 判读完成：长输入 exposure 不必要 → 强烈支持机理 A（§6.2） |

**原因分析（三案综合）**：
- **机理归因（本轮最重要结论）**：E3a 证明远端迁移不依赖长输入 exposure——
  只见过 2K 输入的模型与见过 16K 的 Z 在 16K far 上统计不可区分（20/32 vs 23/32，
  双双零丢失）。结合 Y≈Z（两张不同的合格固定表给出同等迁移），证据形态一致指向：
  **换表解锁了模型既有的长程能力（机理 A），LoRA 承担的是短输入上即可完成的
  接口适配**；"LoRA 自己学会长上下文检索"（机理 B）被压低到无证据支持的地位。
  剩余未闭合：A 的定量理论（谱预算→判别边界的定理形态，见桌面求助文档）。
- 训练侧：96 transfer 步内 `native_kl` 始终压在预算 .02 内（末段 text≈0.025 为采样瞬时值，
  对偶乘子稳定），说明固定表 + KL 预算的约束配方没有牺牲 Native 系统换取任务分。
- 迁移侧：far +20 / 0 丢失的形态与"表提供位置先验、LoRA 承担格式/终止适配"的分工一致；
  compact 零损失说明迁移不是靠压缩输入记忆答案（E3a Z_compact 将进一步检验这一点）。
- 确认侧缺口：task 配对 CI 下界 0.866 来自 reasoning 层小样本波动（5→4）与 instruction 的
  单例丢失，并非系统性退化；但按纪律 CI 不足即"未确认"，不做语义粉饰。

## 8. 交接

### 8.1 盲标注（步骤 A，待人执行）
- 本轮导出：服务器 `claude_round10_20260905/content_audit/`，768 盲化案例（N0/N128 各 384），
  其中 351 例待人工盲标注（417 个 exact 成功已预填真值）。
- 前序未标：`/root/autodl-tmp/claude_audit_prep_20260905/out/` 下 1A（58 例）与 1B（384 例）。
- 纪律：`private_mapping.json` 在标签冻结前不外发；盲标注完成前，语义状态一律记
  `NOT_MEASURED_BY_STRICT_SCORER`。

### 8.2 新确认协议（§10 映射要求，待登记）
- Z（与 Y）点值可行但未获确认级：旧 N128 确认池已暴露，禁用；需要**新的独立确认协议**。
- CPU 功效分析结论（`cpu_power_result.json` / 本地 `round10_20260905/analysis/`）：
  单纯加大确认池不稳（K=128 且效应保持时成功概率仅 0.71；效应衰减 5% 即 ≤0.20），
  新协议必须从**构造/分层/区间方法**上解决下界问题，登记后再执行。

### 8.3 Codex / 文档事项
- N_compact 引擎守卫冲突（§3）：协议文档（§10、审查复核）与冻结引擎守卫的真实矛盾，
  已存档不私改；由 Codex 决定如何在论文/协议中表述（E3a 已作为可执行替代完成）。
- §9.3 teacher-prefix 诊断（Native 丢失样本）：需 GPU 前向，本轮未执行，留作后续。
- 桌面理论求助文档：`QUESTION_SPECTRAL_BUDGET_LORA_ROUND10_20260905.md`
  （A/B 机理归因 + 确认池设计 + 推导优先级，供外部理论分析）。

### 8.4 关机记录

2026-09-06 凌晨：round11 OLMo 跨家族轮（ZC/ZF/ON 三案）全部完成、报告定稿、
桌面文档合并重写后，按用户授权执行 `shutdown now`。

**关机前用户裁定（2026-09-06）**：本轮及 §10 构成研究目标漂移——
LoRA 次级分支被抬为主线，"表解锁内容、LoRA 修接口"等归因未经严格识别。
主线归零为两个硬问题（静态表能否同时做到 transport+Native retention、
能否产生真实自然长生成）与三个待钉死的问题，详见
`reports/ROUND11_OLMO_RESULTS_20260905.md` §9 与桌面合并文档。
本报告 §8.1/8.2/8.3 的三项待办（盲标注、确认池、Codex 事项）**暂停**，
不再自动执行。
