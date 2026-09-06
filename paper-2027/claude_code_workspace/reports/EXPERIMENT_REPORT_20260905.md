# 两方向审计实验执行报告（2026-09-05）

> **状态更新（2026-09-05 晚）**：主线 §10 LoRA 轮次已在有卡模式执行完毕（Z / Y 双双
> "点值可行、确认待定"，统计不可区分；N_compact 被冻结引擎守卫阻塞，以预注册的
> E3a = Z_compact 替代，训练中）。完整结果、原因分析与交接见
> `ROUND10_LORA_RESULTS_20260905.md`。本报告下文"只等有卡模式跑 prepare+run"等
> 表述为该更新之前的状态，保留原样不改写。

指导文件：`HYBRID_ROPE_TWO_DIRECTION_THEORY_AUDIT_20260905.md`；机器：westc 主机（`ssh -p 27741`，RTX 4080 SUPER 32GB）。
本报告覆盖 Claude 工作目录（`claude_code_workspace/`）的三个审计实验（辅助审计）以及主线 §10 LoRA 轮次
的准备状态。审计实验产物在服务器 `/root/autodl-tmp/claude_audit_prep_20260905/`（本地镜像
`results_20260905/` 已于 2026-09-06 按用户指令清理，服务器原件保留）；§10 轮次开机包在本地
`round10_20260905/`。

## 0. 主线定位（2026-09-05 修正）

按 `AGENTS.md` / `INDEX.md` / `README.md` / `HANDOFF.md`：项目当前只有两条活跃路线——
**Z（零训练静态表）** 与 **F（轻量适配 = LoRA 微调）**。本报告的三个审计实验是**辅助审计**，不是主线：

- **方向一（1A 首分歧诊断 + 1B 要求交叉）** ↔ §10 的"步骤 A 内容审计"与 N128 格式/索引退化
  （77.05%）的机理诊断，服务于 LoRA 轮次的结果解读。已完成，待人工盲标注。
- **方向二（support×shape 析因）** ↔ 对已冻结的 fmrope-125M 三 seed 结论的因果分解
  （属于固定 support 研究的历史收尾），与 EVQ、与 Qwen LoRA 轮次无关。已遇本机身份墙，待 Codex 决策。

**主线 = preflight §10 的固定比较轮次（LoRA）**：`N_compact`（是否复制 N128 的长输入 exposure 增量）→
`Z`、`Y`（同一 Qwen 上固定表配方能否产生 Native 可行的生成工作点）。固定参数：all-linear r16/alpha16、
seed42、KL 预算 .02、32R+96T、最终 step128、prefix off；引擎为原 N128 的 `code_release_008`（hash 双重锁定）。
该轮次已由作者预先授权（`HANDOFF.md`："The next session runs the prepared bounded comparison"），
caps 与停止规则冻结在 §10。

**§10 轮次准备状态（2026-09-05 无卡模式全部完成，只等有卡模式跑 prepare+run）**：
- 用户当日以**无卡模式**开机（容器 `autodl-container-c904489327-8b72fcf9`，cgroup 内存上限 2GiB，
  无 GPU，/root/autodl-tmp 余 20G）。无卡模式下完成：
  1. 全部锚点哈希复核（release008 engine/runtime/contract `bc9826ea…/401767d5…/554fe323…`；
     权重 `dd924a11…`；run.json/training.jsonl 自哈希 `1ec00ae3…/682da90f…`；tasks/native/contract
     manifests；603MB views 与 qualification/rows/candidate_pool 全量哈希）——与 N128 receipts 逐项一致。
  2. **4 个待定路径发现并验证**：teacher_cache=`/root/ffn_review_scratch_20260904/teacher_cache_qwen`
     （manifest `d7f2391e…`，896 条，6.6GB）；controls=`qwen_fixed_controls`（FROZEN_V1）；
     native_baseline=`qwen_native_validation`（native/gain1/无 adapter/fold=selection）；
     baseline_eval_engine=release007 trainer（`ee4b53db…`=native receipt 的 evaluation_engine_sha256）。
  3. 部署新代码目录 `/root/autodl-tmp/claude_round10_20260905/code_round10/`（388 文件，关键文件哈希本地=服务器；
     未触碰 code_release_008）；配置 `ROUND10_CONFIG_FILLED_V1` 已填好上传。
  4. 服务器单元测试 **9/9 OK**。
  5. prepare 预检（2GiB 内可跑部分）全过：release008 与部署 trainer 的 4 个调度函数 **AST 全等**、
     `--compact-only` 存在、数据/资格/对照/缓存身份字段一致。
  6. **步骤 A 内容审计导出完成**（CPU/tokenizer-only）：`/root/autodl-tmp/claude_round10_20260905/content_audit/`
     768 盲化案例（N0 165/384 strict，N128 252/384 strict；417 预填、**351 待盲标注**）。
- **唯一留给有卡模式的步骤**：`prepare` 封版 + 三个 case 的 GPU 执行。原因：冻结引擎的 `preflight`
  （check_assets）会把 603MB/4736 行的 transport_views.jsonl 全量载入内存（约 128M token，>3GB 对象），
  无卡容器 2GiB 上限下 OOM（exit 137，日志 `claude_round10_20260905/prepare_host.log`）。这是资源限制，
  不是输入错误；N128 全流程曾在有卡模式完成，说明有卡模式内存足够。**不改任何冻结代码**：
  launcher/引擎/数据全部按原样使用，prepare 在有卡模式作为第一条命令执行，随后
  `run --cases N_compact --authorized` → 读结果 → `run --cases Z Y --authorized`（详见更新后的
  `RUNBOOK_ROUND10.md`）。
- 指纹漂移（方向二）**不影响本主线**：§10 轮次的全部输入（release008 引擎、Qwen checkpoint、
  tasks/native_pool、teacher cache、controls、baseline receipts）由 prepare/run 逐项 hash 强校验，
  与 fmrope-125M 的 checkpoint 完全无关。

## 直接回答：两个方向的问题解决了吗？

- **方向一（N128 相对 N0 的退化）— 机理层面已解决。** 核心疑问"29 个原正确→错误是真实退化还是数值假象"已有
  明确答案：**是真实的 argmax 翻转**（29/29 在首分歧点 winner 改变，KL 屏障从未给出'不变'证书）。1B 的生成也已完成；
  唯一未闭合的是**语义归类**（属于内容错误/格式/终止/合法改写中的哪类），这需要人工盲标注，不是 GPU 任务。
- **方向二（把既有结论分解为 support 效应 vs shape 效应）— 部分解决，遇本机身份硬墙。** 可复用回执复现了**对角政策
  对比及其跨 support 反号**（seed137：train −0.191 / target +0.156；seed256：−0.132 / +0.344）。但把 support 与 shape
  真正分离的**非对角格被 `_load_checkpoint` 指纹校验挡住**：checkpoint 记录的协议/代码指纹与当前代码不符（且无 git
  历史可回退、全机器无匹配副本）。按审计纪律不能绕过、不能重训。**这一项需要 Codex 决策**：找回训练这些 checkpoint 的
  原始代码修订重跑非对角格，或接受'对角对比+反号'的部分结论并相应收紧论文措辞。

**GPU 侧已无可推进的合规工作**：两个方向的 GPU 步骤（1A 诊断、1B 生成、方向二补缺）均已跑完或因身份墙明确停在前述
位置。剩下的是 (a) 人工盲标注、(b) Codex 对方向二身份漂移的处置。故本轮到此收尾，交接给 Codex。

## 审计实验结论速览（三实验）

| 实验 | 状态 | 关键数字 |
|---|---|---|
| 方向一 1A 首分歧 KL 诊断 | **完成** | 58 例 / 29 分歧；`winner_unchanged_at_divergence=0`、`barrier_sufficient_at_divergence=0` |
| 方向一 1B 输出要求交叉 | **生成完成，语义待盲标注** | 384 行，新增生成恰 192（命中硬上限） |
| 方向二 固定权重 support×shape 析因 | **部分完成（本机极限）** | 8 REUSED / 8 BLOCKED_IDENTITY_DRIFT / 8 MISSING_ASSET |

两条不变边界均守住：未覆盖任何历史 run 目录；一次一个 GPU 进程；诊断未产生新参数选择权，方向二无格升格为部署。

---

## 1. 方向一 1A：首分歧 KL 诊断（完成）

**设计**：29 个原正确→错误案例（15 position_format + 5 instruction + 9 reasoning）+ 29 个规则匹配保留对照；
在冻结前缀 {0, t\*//2, t\*} 上，对 N0（baseline，logp）与 N128（candidate+adapter，logq）做 full-vocab KL、
KL 屏障 B(p)、q-margin。身份锁与 `code_release_008` 引擎哈希（`401767d5…`）逐键核对通过。

**结果**（`out/1a_diagnose/summary.json`）：
- cases=58，prefix_rows=118，divergence_cases=29，nonfinite=0。
- `teacher_argmax_not_reproduced=29`；`barrier_sufficient=2`（仅出现在非分歧前缀）。
- **`winner_unchanged_at_divergence=0`**、**`barrier_sufficient_at_divergence=0`**。

**读法**：在每一个首分歧位置，candidate 的 argmax 确实改变（winner 变了），且 KL 屏障从未给出"argmax 不变"的
充分证书。即这 29 个退化是真实的 argmax 翻转，不是小 KL 的数值假象。按审计纪律：`kl<B` 只是充分条件，
B=0/tie 无证书；本诊断只针对已暴露案例，不估计总体发生率、不给表排名。

**产物**：盲标注已导出（58 例，`out/1a_labels_export/`，`private_mapping.json` 冻结前不外发；
rubric 版本 `FIRST_DIVERGENCE_LABEL_V1`），等待 `divergence_class / student_semantic / reason` 标注。

---

## 2. 方向一 1B：输出要求交叉（生成完成，语义待标注）

**设计**：16 组（前 8 single_evidence + 前 8 binding，按语义 id 升序，**未按 N128 表现过滤**）× 3 布局
（compact/near/far）× 2 世界 × 2 部署（N0、N128）× 2 要求（原裸答、新短句）= 384 次生成；原要求输出按
prompt 精确哈希复用，新增生成硬上限 192。

**关键实现发现（tokenizer）**：Qwen 把任意长度的连续换行合并为**单个** token（`\n\n\n\n`→1 token），因此
"按换行填充精确保持 token 数"在该 tokenizer 下不可达。改为文本空间不变式：替换只发生在要求句内部，句外逐字节
往返一致；相对原要求的头部长度差 `delta` 被**记录而非隐藏**（compact 恒 +2；near −8…−19；far −8…−18）。
同一要求内 N0 与 N128 收到字节一致的 prompt，故 N0-vs-N128 比较在位置上仍是干净的。

**结果**（`out/1b_generate/cross.json`）：rows_total=384，new=192（恰命中硬上限，未超）。原要求 192 行复用；
短句要求 192 行新生成。

**可解性分层（审计纪律 #3，先报 N0 两格式可解性）**——以 EOS 正常结束为生成纪律代理：
- candidate N128：短句要求 **96/96 全部 EOS**，平均生成 7.8 token（max 38）。
- baseline N0：短句要求 **90/96 EOS**，6 例未 EOS（near 2、far 4），平均 21.3 token（max 64，部分逼近预算）。
- 即短句要求下，N128 终止更干净、更短；N0 在 far/near 有未收尾样本。语义正确性**不以终止判定**，待盲标注。

**产物**：盲标注已导出（384 例，`out/1b_labels_export/`，rubric 版本
`NATURAL_ASSERTION_FORMAT_EOS_V1_REQUIREMENT_CROSS`），等待 `semantic_correct / format_compliant / reason` 标注。

---

## 3. 方向二：固定权重 support×shape 析因（部分完成 — 本机身份极限）

**设计**：24 格 = 3 seeds(42,137,256) × 2 权重(G=fmrope_base256, C=anchored_cosh_tau4) ×
2 supports(S_train, S_target@1024) × 2 shapes(z_G, z_C)。identity-exact 回执复用，只补缺格；不重训。

**结果**（`out/2_ledger.json` → `out/2_report.json`）：
- **8 REUSED**：两 seed 的对角格（各自权重配自身训练 shape）。修复了一处验收 bug——归档回执的
  `tail_target_sha256` 是对**尾部 128 token** 的哈希，核对须用 `targets[i,-TAIL:]` 而非全窗；修复后 8 对角格
  全部通过尾目标哈希核验并复用。
- **8 BLOCKED_IDENTITY_DRIFT**：非对角格需重载 checkpoint，但 `_load_checkpoint` 的协议/代码指纹校验失败——
  checkpoint 记录的 `protocol_sha256`（seed137=`8135dc4d…`，seed256=`2b776fd1…`）与当前 repo 快照
  `SPEC.fingerprint()`（`d92da1b8…`）及 `code_fingerprint()` 均不符；且两 seed 记录值彼此不同，说明它们由**不同
  代码修订**训练。本机无 git 仓库、无匹配代码快照可恢复。按审计明文："指纹校验失败即停止该增量、报告身份漂移、
  不猜表"——故标记为 BLOCKED，不加载、不伪造。
- **8 MISSING_ASSET**：seed42 checkpoints 不在本机；按审计 §4.3 收紧措辞、不重训。

**仍可从 REUSED 对角格读出的量**（对角政策对比，`report.diagonal_policy_contrasts`）：
- seed137：D_S_train=**−0.191**（train support 上 C 对角更优），D_S_target=**+0.156**（target support 上 C 对角更劣）。
- seed256：D_S_train=**−0.132**，D_S_target=**+0.344**。
- 两个 seed 均出现**跨 support 的对角效应反号**（train 支持 C、target 支持 G）。注意这是"对角"对比
  （各权重配自身训练 shape，support 为按臂构造），非析因交互。

**不可计算量**：allocation 对比（固定 W、S 下 z_C−z_G）与运行时交互 `I_W` 都依赖非对角格，因身份漂移为 `None`。
`I_W≠0` 是交互、不是 crossover 的判读纪律保持不变。

**方向二结论**：本机可复用的证据复现了对角政策对比及其跨 support 反号；但把 support 与 shape 分离的真正析因
（非对角格）在当前机器上被身份漂移阻断。这是对既有结论的**部分**因果分解，需在拥有匹配代码修订与全部三 seed
checkpoint 的环境补齐，或由 Codex 决定如何处理该漂移。

---

## 4. 交接给 Codex 的待办（按优先级）

1. **盲标注（人工，非 GPU）**：1A（58 例）与 1B（384 例）已导出 `cases.jsonl / prompts.json / rubric.json` 于
   `out/1a_labels_export/`、`out/1b_labels_export/`。标注者只拿这三样；`private_mapping.json` 在标签冻结前不外发。
   标注完成后解冻并出语义结论（设 `PY=/root/miniconda3/bin/python`、`WORK=/root/autodl-tmp/claude_audit_prep_20260905`、
   `BASE=/root/autodl-tmp/ffn_review_execution_20260904`、`export HYBRID_ROPE_ROOT=$BASE/code_release_008`）：
   ```bash
   $PY $WORK/code/first_divergence_kl_diagnosis.py summarize-labels \
     --export $WORK/out/1a_labels_export --annotations /path/to/1a_annotations.jsonl \
     --output $WORK/out/1a_labels_result
   $PY $WORK/code/output_requirement_cross.py summarize \
     --export $WORK/out/1b_labels_export --annotations /path/to/1b_annotations.jsonl \
     --output $WORK/out/1b_result
   ```
2. **方向二身份漂移处置（需决策）**：非对角 8 格被 `_load_checkpoint` 指纹校验挡住。已核实：训练用代码目录就是
   `/root/autodl-tmp/hybrid-rope`（见 `iclr_exact_range_multiseed/logs/prepare.out`），但当前该目录
   `SPEC.fingerprint()=d92da1b8…` 与 seed137 记录 `8135dc4d…`、seed256 记录 `2b776fd1…` 均不符，两 seed 亦互不相同；
   本机无 git 历史、无任何匹配副本。选项：(a) 从别处找回训练这些 checkpoint 的原始代码修订与三 seed checkpoint，
   重跑非对角格；或 (b) 接受"对角对比 + 跨 support 反号"作为部分结论，按审计 §4.3 收紧论文措辞。
   **禁止**绕过指纹校验或以重训填格。
3. **§10 LoRA 轮次（主线，Claude 执行）**：已于 2026-09-05 执行。N_compact 被冻结引擎守卫
   阻塞（协议文档与引擎的真实矛盾，已存档），以预注册 E3a（Z_compact）替代；Z / Y 完成、
   双双点值可行。结果按 §10"结果到后续工作的映射"表落位于 `ROUND10_LORA_RESULTS_20260905.md`
   （其 §7 映射表与 §8 交接），不再追加到本报告。

## 5. 复现与产物索引

- 服务器根：`/root/autodl-tmp/claude_audit_prep_20260905/{code,configs,runbooks,out,logs}`；日志在 `logs/`。
- 本地镜像：`results_20260905/`（1a/1b/2 的 manifest、summary、ledger、report、cross、盲标注 manifest+rubric）。
- 脚本：`code/first_divergence_kl_diagnosis.py`、`code/output_requirement_cross.py`、`code/fixed_weight_support_shape_cross.py`。
- 机器私有路径映射：`configs/asset_paths_westc_20260905.json`（勿提交共享仓库）。

**纪律复述**：这是回顾性盲化诊断，不是独立确认；ambiguous 保留、不用 substring 判语义；诊断不产生新参数选择权；
方向二任何格都不升格为部署；失败案例不回灌训练。
