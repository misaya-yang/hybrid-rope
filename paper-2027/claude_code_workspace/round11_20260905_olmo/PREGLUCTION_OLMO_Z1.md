# OLMO_Z1 预注册：跨家族 LoRA 固定表轮次（OLMo-2-0425-1B-Instruct）

日期：2026-09-05。在 §10（Qwen2.5-1.5B-32K，Z/Y/E3a 三案完成）之后，用户指令：同一配方换模型家族验证（"直接1.4那个"）。本文件在任何 OLMo GPU 运行之前冻结。

## 1. 引擎与代码

- 引擎 = `code_release_008`，逐字不改（训练器/运行时/数据脚本全部原样执行；已核对：引擎守卫、长度校验、审查器均无模型特定硬编码阻挡 OLMo）。
- `code_round11` = release008 的逐字副本（仅用于承载审查器脚本，round10 先例）。release008 只读。
- OLMo 是 P2/log_s4 witness 的本家模型：Z 表 float32_sha256 = 56ddfae2… = P2_SHA，引擎守卫零修改通过（Qwen 当时需要 profile_source 绕行）。

## 2. 资产（全部已存在，本轮不新建任何冻结资产）

| 资产 | 路径 | 关键哈希/值 |
|---|---|---|
| 检查点 | olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct | weight sha 36d044c7…；1.48B，16 层，rope_theta 500000，**max_position_embeddings 4096** |
| 合同 | ffn_review_execution_20260904/olmo1485_contract.json | FROZEN_CHECKPOINT_CONTRACT_V1，native_context_length 4096 |
| 固定控制表 | ffn_review_execution_20260904/fixed_controls/ | status FIXED_NZGY_CONTROLS_FROZEN_V1；Z gain **1.102585782722872**，file ee968fce…，float32 56ddfae2…(=P2_SHA) |
| Native 池 | ffn_review_execution_20260904/native_pool_v3/manifest.json | NATIVE_REPLAY_POOL_V1，OLMo tokenizer，rows sha 282e5740…，池 sha a060c665… |
| 教师缓存 | /root/ffn_review_scratch_20260904/teacher_cache_olmo/ | ORIGINAL_NATIVE_FULL_VOCAB_CACHE_V1，pool_sha256 a060c665…（与池清单匹配），896 条 = 全部非 test 行 |
| 源语料 | /root/ffn_review_scratch_20260904/data_ids_april7.zip | sha 95df2bf5… = Qwen 候选清单记录的 two_wiki_zip_sha256（同源） |
| SQuAD/GSM/FineWeb | ffn_review_scratch_20260904/hybrid_rope_{squad,gsm}* + autodl-tmp/fineweb_edu/sample/10BT | 与 Qwen 轮相同输入文件 |

## 3. 与 Qwen 轮的差异（全部预先声明）

1. **布局相同、语义不同**：训练器硬编码 length_cap ∈ {2048,8192,16384}、验证 {2048,16384}。对 Qwen（native 32K）这些都是 within-native；对 OLMo（native 4096），near 8192/far 16384 是 **2×/4× 外推**。因此本案的 far 16K 是真正的超窗口外推测试，比 Qwen 轮（16K=0.5×native）更难。这是跨家族配方的更强检验，不是同等条件复现。
2. **候选重建**：候选文件内嵌 tokenize 结果（compact_prompt_ids/target_ids），Qwen 版候选是 Qwen tokenizer 的。用同一 2wiki zip + 同一脚本 + OLMo tokenizer 重新生成（`candidates` 动作，CPU，零参数改动）。语义任务分布同源，条目集不同（合格集本来就按模型重筛）。
3. **基线自建**：native 基线 = 本轮 step0 native-evaluate（native0），task 基线 = task0（native 表、无适配器）。二者与候选同引擎同命令族产生，审查器身份核对自洽。fresh_native_validation 留作旁证。
4. 比较是**配方级**的（跨模型、跨条目集），不做条目级配对；与 Qwen 数字只做定性对照。

## 4. 案与配方（一次一块 GPU，顺序执行）

| 案 | 内容 | 目的 |
|---|---|---|
| D0 候选重建 | candidates 动作，OLMo tokenizer，CPU | 前置数据 |
| D1 资格筛选 | qualify（GPU，OLMo native 贪心，双世界全对+EOS+正 margin，配额 64/32/32，≤2000 筛） | 风险闸门：已知 OLMo 零训练有 EOS 坍缩史 |
| D2 视图固化 | materialize（CPU）：train 2048/8192/16384，val 2048/16384，test 16384/32768/65536 | 前置数据 |
| OLMO-T0 | evaluate，native 表、无适配器，--lengths 2048 16384 | task 基线 |
| OLMO-Z0 | evaluate，Z 表 gain 1.102585782722872、无适配器 | 零训练表解锁量（Qwen 对应 far 3/32） |
| OLMO-ZC（主案） | train：arm Z，seed42，all_linear r16/α16，128 步（32R+96T），KL .02，**--compact-only**，gain 同上，教师缓存 teacher_cache_olmo | 与 Qwen E3a 同构：训练只见 2K compact（≤OLMo native），评估到 16K 外推 |
| 评估 | native32/native128 + task128（--lengths 2048 16384） | 终点 |
| 审查 | review_native_constrained_transfer.py --protocol single_evidence_v4，基线/候选引擎源均 release008 | 冻结判读 |

训练输入全部 ≤2048 token（compact-only），Native KL 池行 ≤4096：训练全程在 OLMo native 窗口内；只有评估的 16384 是外推。

## 5. 预注册判读（先看数据前冻结）

- 主终点：single_evidence far 16K 严格成功数（完整答案+EOS），32 组。
- 门值：Native 保留点值 task≥0.88 且 ppl≥0.88（v4 审查器）；紧凑 2K 零损失（相对 task0）。
- 信号分级（一次运行、按原样读，不重选）：
  - **far ≥12/32 且门过**：强跨家族信号——固定表+约束 LoRA 能让 4K-native 模型在 4× 外推上精确作答，强烈支持机理 A（表给位置判别预算，LoRA 只做接口修补，且完全不需要长样例）。
  - **6–11/32 且门过**：点值可行的跨家族信号（远优于零；弱于 Qwen 幅度可归因于 4× 外推难度）。
  - **<6/32 或门不过**：配方未按原样迁移。诊断顺序：Z0 vs T0 差（表本身是否零训练有解锁）、EOS/格式失败占比、Native 保留掉在哪层。结果如实登记为负/部分结果。
- 硬性阻塞分支：若 D1 = BLOCKED_INSUFFICIENT_NATIVE_COMPACT_QUALIFICATION → 登记"OLMo native 紧凑作答无法通过双世界筛选，配方前置条件不满足"，轮次终止，关机。
- 禁止事项（自律）：不改冻结常数来"救"本回合；不重选检查点；Z0/T0 暴露后不得据其调参。

## 6. 次要探索（主案完成后、若 GPU 时间允许）

- arm N 全暴露训练（引擎允许：N 无 compact-only 限制）→ 跨家族版"长输入训练无换表"对照。只在主案判读完成后启动，结果单独登记。
