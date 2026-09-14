# §10 固定比较轮次（N_compact / Z / Y）开机运行手册

> **状态（2026-09-06）**：轮次已执行完毕（三案全完成）；本手册为历史文档。
> 结果与后续更正（含 Qwen raw 模式崩溃警示）见
> `../reports/ROUND10_LORA_RESULTS_20260905.md` 顶部状态栏。本地回执 JSON
> 已于 09-06 清理，服务器原件保留。

- **日期：** 2026-09-05（无卡模式准备已全部完成；等待用户切有卡模式）
- **协议：** `CONSTRAINED_FRONTIER_AND_2X4X_LORA_PREFLIGHT_20260904.md` §10（2026-09-05 准备版）
- **授权：** 用户 2026-09-05 明确授权 GPU 实验安排由 Claude 决断；§10 为作者预先冻结的有界轮次
  （固定 caps：train 3600s / native 900s / task 1800s / review 600s；seed42；step128 固定；无 resume/
  prefix/test/扫参）。本手册只执行该冻结轮次，不新增任何臂。
- **目标机：** westc（`ssh -p 27741 [REDACTED_EMAIL]`；开机后端口可能变化，需用户确认）。
  无卡模式容器 cgroup 内存上限仅 **2 GiB**；有卡模式内存充足（N128 全流程曾在有卡模式完成）。
- **本机包：** `claude_code_workspace/round10_20260905/package/`；`config/round10_config_westc.json`（已填好）。

## 无卡模式已完成（2026-09-05，勿重做）

1. **机器与 bundle 核对**：容器 `autodl-container-c904489327-8b72fcf9`；`nvidia-smi` 无设备；
   `/root/autodl-tmp` 余 20G ≥ 8GiB；GPU 无残留进程。
2. **全部锚点哈希复核**（对照 `$B/qwen_N_s42_fixed/run.json` + `complete.json`，B=`/root/autodl-tmp/ffn_review_execution_20260904`）：
   - release008 engine/runtime/contract = `bc9826ea…` / `401767d5…` / `554fe323…` ✓
   - checkpoint 权重 `model.safetensors` = `dd924a11…` ✓；contract weight_sha256 一致 ✓
   - `sha256(run.json)` = `1ec00ae3…` = complete.run_sha256 ✓；`sha256(training.jsonl)` = `682da90f…` = N128_TRAINING_SHA ✓
   - tasks / native_pool / contract manifests = `baf1d791…` / `5e128413…` / `a32ed5b3…` ✓
   - views / qualification / rows / candidate_pool 数据哈希 ✓（603MB views 全量流式核对）
3. **4 个待定路径全部发现并哈希验证**：
   - teacher_cache = `/root/ffn_review_scratch_20260904/teacher_cache_qwen`（manifest `d7f2391e…` ✓，
     status ORIGINAL_NATIVE_FULL_VOCAB_CACHE_V1，896 条=512 train+128 cal+256 val，teacher=原 Native/gain1/无 adapter/权重哈希一致，6.6GB）
   - controls = `$B/qwen_fixed_controls`（FIXED_NZGY_CONTROLS_FROZEN_V1，checkpoint_config `98d2ff8c…`，
     native_sha256=table `138c99b1…`，32K；Z/Y 表文件 file_sha256 ✓）
   - native_baseline = `$B/qwen_native_validation`（table_is_native=true，gain=1，adapter=null，fold=selection，examples.jsonl `ef191193…` ✓）
   - task_baseline = `$B/qwen_task_baseline`（split=validation，table_is_native=true，gain=1，adapter=null，examples.jsonl `31c2b138…` ✓）
   - baseline_eval_engine = `$B/code_release_007/scripts/train/train_single_table_native_constrained.py`
     （sha `ee4b53db…` = native_validation receipt 的 evaluation_engine_sha256 ✓；task_baseline 用 `bc9826ea…`=release008 trainer，
     与 reviewer 的双引擎集合校验一致）
4. **部署新代码目录**：`/root/autodl-tmp/claude_round10_20260905/code_round10/{scripts,tests}`（388 文件，6.5MB；
   三个关键文件本地/服务器 sha 一致：`9084ec0b…` / `e9cd1933…` / `36b7511d…`）；配置在
   `/root/autodl-tmp/claude_round10_20260905/round10_config_westc.json`（ROUND10_CONFIG_FILLED_V1）。
5. **服务器单元测试**：`python -m unittest tests.test_matched_transfer_round -v` → **9/9 OK**。
6. **prepare 预检（低内存可执行部分全部通过）**：release008 与部署 trainer 的 4 个调度函数 **AST 全等**；
   `--compact-only` 在冻结引擎中存在；数据/资格/对照/缓存身份字段全过。
7. **步骤 A 内容审计导出已完成（CPU/tokenizer-only）**：`/root/autodl-tmp/claude_round10_20260905/content_audit/`
   — 768 盲化案例（baseline 165 strict/219 非；candidate N128 252 strict/132 非），417 个 exact 成功自动预填，
   **351 例待人工盲标注**；`private_mapping.json` 冻结前不外发。

## 有卡模式待执行（用户切卡后按序执行）

### 阶段 4：CPU prepare（必须在有卡模式：冻结引擎的 preflight 要全量读 603MB views+tokenizer，>2GiB）

```bash
PY=/root/miniconda3/bin/python
R=/root/autodl-tmp/claude_round10_20260905/code_round10
cd $R
nvidia-smi --query-compute-apps=pid --format=csv,noheader   # 必须为空
df -h /root/autodl-tmp                                       # output_root 需 >=8GiB
$PY scripts/experiments/matched_transfer_round.py prepare \
  --config /root/autodl-tmp/claude_round10_20260905/round10_config_westc.json
# 期望输出 {"status": "PREPARED_CPU_ONLY_GPU_PENDING", ...}；记下 plan 路径
export ROUND_PLAN=/root/autodl-tmp/claude_round10_20260905/out/plan.json
```

prepare 失败时：保留失败目录与日志作诊断，换新 output_root 重试；不改旧证据。
（无卡模式曾试跑 prepare：容器 2GiB 上限 → 冻结引擎 check_assets 全量载入 4736 views 时 OOM exit 137；
日志 `/root/autodl-tmp/claude_round10_20260905/prepare_host.log`。此为资源限制，非输入错误——全部输入哈希已在无卡模式逐项预检通过。）

### 阶段 5：GPU 执行（先 N_compact，读完结果再 Z Y）

```bash
$PY scripts/experiments/matched_transfer_round.py run \
  --plan $ROUND_PLAN --cases N_compact --authorized
# 读 out/N_compact/review.json 与 execution.json，写入报告；然后：
$PY scripts/experiments/matched_transfer_round.py run \
  --plan $ROUND_PLAN --cases Z Y --authorized
```

- 一次一个 GPU 进程；launcher 自带锁与 GPU 占用检查。
- 训练完成后 launcher 强制核对 complete.json（step128、final reload exact、曝光哈希）。
- 任何阶段失败 → `STOPPED_PRESERVE_PARTIAL`，读日志与 execution.json；不自动 resume。
- 全轮结束后进程自行退出；**不关闭主机**。
- 参考：N128 原训练 1721.9s（≈29 分钟）；单 case 全阶段（含 eval/review）预计 ~1 小时内。

## 阶段 7：结果 → 行动映射（§10 原文）

| 结果 | 下一步 |
| --- | --- |
| N_compact 复现大部分严格提升、内容差分无明确增量 | 接受低成本任务适配解释；收缩长训练机制叙事 |
| N128 有额外 far 内容增量且 compact 接近 | 保留长输入适配主张；不归因 FFN/RoPE 必要性 |
| Z/Y 至少一臂 Native 与主生成点值可行 | 冻结 final128 及新确认协议；先新 Native 确认，再预冻结 farther 测试 |
| Native CI 不够窄但点值合格 | 未确认；允许准备新独立确认，不重选 checkpoint |
| 固定 Z/Y 预算都不满足 Native | 停本配方候选；设计真实 teacher 决策轨迹覆盖变化，不加步/扫 KL |
| Native 可行而内容迁移不足 | 再登记 N/Z × prefix off/on；只加固定辅助 LM 目标 |
| Y 不逊 Z | 如实报告系统对照；allocation 贡献仍由原 fixed-support 研究支撑 |

## 不变边界

- 不覆盖 `code_release_008` 与任何历史运行目录；不另下权重、不另建训练集。
- 确认集（已暴露的 N128 confirmation）不用于调参或确认新变体。
- 严格成功 = 完整答案 + EOS；不用 substring 提分。
- 一次一个 GPU 进程；结束后不自动关机。
- 机器私有路径只留在 `claude_code_workspace/`（未跟踪），不进共享仓库。
