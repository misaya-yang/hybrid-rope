# 开机运行手册 — 两方向审计实验（2026-09-05）

> **状态（2026-09-06）**：已作废——属两方向审计时代（项目重置前）。后续开机
> 手册见 `../round12_20260906/RUNBOOK_ROUND12.md`；本轮资产映射配置
> （configs/asset_paths_westc_20260905.json，标注"不得提交"）已于 09-06 删除。

依据：`HYBRID_ROPE_TWO_DIRECTION_THEORY_AUDIT_20260905.md` + Codex `HANDOFF.md`/preflight §10。
本手册只覆盖 Claude 工作目录的三个诊断脚本；Codex 的 N_compact/Z/Y 固定比较轮次按其
preflight §10 原文执行，不在此重复。

## 不变边界

- 不覆盖 `code_release_008` 与任何历史 run 目录；新产物全部进 `$WORK` 新目录。
- 一次只跑一个 GPU 进程；进程自己退出，不自动关闭主机。
- 严格成功 = 完整答案 + EOS；不做 substring 评分。
- 已暴露确认集只用于诊断，不用于调参或回灌训练。
- 诊断不产生新参数选择权；方向二任何好格都不升格为部署表。

## 路径（已在无卡模式核对）

```bash
PY=/root/miniconda3/bin/python
BASE=/root/autodl-tmp/ffn_review_execution_20260904
REL=$BASE/code_release_008                 # 方向一引擎根（code_sha256=401767d5… 与冻结回执一致）
MROOT=/root/autodl-tmp/hybrid-rope         # 方向二根（含 rebuttal/）
MULTI=/root/autodl-tmp/iclr_exact_range_multiseed
CKPT=/root/autodl-tmp/qwen25_1p5b_32k
WORK=/root/autodl-tmp/claude_audit_prep_20260905   # 本次上传的新目录
```

开机先核对：`nvidia-smi`、`df -h`（$WORK 所在盘 ≥8GiB 空闲）、`$PY -V`。

## 阶段 0：CPU 准备（不开 GPU，几分钟）

```bash
export HYBRID_ROPE_ROOT=$REL
# 1A prepare：定位 29 个原正确→错误案例 + 规则匹配保留对照，冻结前缀清单
$PY $WORK/code/first_divergence_kl_diagnosis.py prepare \
  --baseline $BASE/qwen_native_confirmation_N0 \
  --candidate $BASE/qwen_native_confirmation_N128 \
  --baseline-lock $BASE/native_confirmation_N0_lock.json \
  --candidate-lock $BASE/native_confirmation_N128_lock.json \
  --native-pool $BASE/qwen_native_pool_clean/manifest.json \
  --checkpoint $CKPT --output $WORK/out/1a_prepare

# 1B lock-groups + prepare：锁定 16 组、生成短句要求面板（token 级拼接校验）
$PY $WORK/code/output_requirement_cross.py lock-groups \
  --tasks $BASE/qwen_tasks/manifest.json \
  --baseline-task-run $BASE/qwen_task_baseline --output $WORK/out/1b_lock
$PY $WORK/code/output_requirement_cross.py prepare \
  --tasks $BASE/qwen_tasks/manifest.json --locked $WORK/out/1b_lock \
  --checkpoint $CKPT --output $WORK/out/1b_prepare

# 方向二 audit-cells：枚举 24 格，匹配已有回执，列出缺失格
export HYBRID_ROPE_ROOT=$MROOT
$PY $WORK/code/fixed_weight_support_shape_cross.py audit-cells \
  --root $MULTI --output $WORK/out/2_ledger.json
```

任一 prepare 失败：保留输出目录做诊断，停下报告，不绕过身份检查。

## 阶段 1：Codex 固定轮次（按其 preflight §10）

`matched_transfer_round.py template → 填 config → prepare → 用户开机授权后 run --cases N_compact`。
本手册不复述；只强调与下面诊断的时序：N_compact 训练期间 GPU 被占用，诊断 GPU 步骤排队等待。

## 阶段 2：方向一 GPU 诊断（小预算，排在 N_compact 之后、Z/Y 之前）

```bash
export HYBRID_ROPE_ROOT=$REL
# 1A diagnose：N0/N128 双臂在声明前缀上的 full-vocab KL、B(p)、q-margin
$PY $WORK/code/first_divergence_kl_diagnosis.py diagnose \
  --baseline-lock $BASE/native_confirmation_N0_lock.json \
  --candidate-lock $BASE/native_confirmation_N128_lock.json \
  --native-pool $BASE/qwen_native_pool_clean/manifest.json \
  --checkpoint $CKPT --checkpoint-contract $BASE/qwen15_contract.json \
  --candidate-adapter $BASE/qwen_N_s42_fixed/step_128 \
  --prepare $WORK/out/1a_prepare --output $WORK/out/1a_diagnose --authorized

# 1B generate：复用原格式输出，新增生成硬上限 192（两臂各 96 条短句要求）
$PY $WORK/code/output_requirement_cross.py generate \
  --prepare $WORK/out/1b_prepare \
  --baseline-task-run $BASE/qwen_task_baseline \
  --candidate-task-run $BASE/qwen_N_s42_fixed_task128 \
  --baseline-lock $BASE/native_confirmation_N0_lock.json \
  --candidate-lock $BASE/native_confirmation_N128_lock.json \
  --checkpoint $CKPT --checkpoint-contract $BASE/qwen15_contract.json \
  --candidate-adapter $BASE/qwen_N_s42_fixed/step_128 \
  --output $WORK/out/1b_generate --authorized
```

## 阶段 3：方向二补缺格（可与阶段 2 交错，仍是一次一个 GPU 进程）

```bash
export HYBRID_ROPE_ROOT=$MROOT
$PY $WORK/code/fixed_weight_support_shape_cross.py run-missing \
  --root $MULTI --ledger $WORK/out/2_ledger.json \
  --output $WORK/out/2_runs --authorized
$PY $WORK/code/fixed_weight_support_shape_cross.py report \
  --ledger $WORK/out/2_ledger.json --run-dir $WORK/out/2_runs \
  --output $WORK/out/2_report.json
```

seed42 checkpoints 不在本机 → 相应格为 MISSING_ASSET，按审计 §4.3 收紧措辞，不重训。
若 `_load_checkpoint` 指纹校验失败：停止该增量，报告身份漂移，不猜表。

## 阶段 4：盲标注（CPU，可与 GPU 并行）

```bash
export HYBRID_ROPE_ROOT=$REL
$PY $WORK/code/first_divergence_kl_diagnosis.py export-labels \
  --native-pool $BASE/qwen_native_pool_clean/manifest.json \
  --checkpoint $CKPT --prepare $WORK/out/1a_prepare --output $WORK/out/1a_labels_export
$PY $WORK/code/output_requirement_cross.py export \
  --prepare $WORK/out/1b_prepare --generate $WORK/out/1b_generate \
  --checkpoint $CKPT --output $WORK/out/1b_labels_export
```

标注者只拿 `rubric.json / prompts.json / cases.jsonl`；`private_mapping.json` 在标签冻结前不外发。
填完 `divergence_class/student_semantic/reason`（1A）或 `semantic_correct/format_compliant/reason`（1B）后：

```bash
$PY $WORK/code/first_divergence_kl_diagnosis.py summarize-labels \
  --export $WORK/out/1a_labels_export --annotations /path/to/1a_annotations.jsonl \
  --output $WORK/out/1a_labels_result
$PY $WORK/code/output_requirement_cross.py summarize \
  --export $WORK/out/1b_labels_export --annotations /path/to/1b_annotations.jsonl \
  --output $WORK/out/1b_result
```

这是回顾性盲化诊断，不是独立确认；ambiguous 保留，不用 substring 判语义。

## 读结果的三条纪律

1. `kl < B(p)` 只是充分条件；argmax 未变不能反推 KL 小。B=0/tie 无证书。
2. 方向二 `I_W≠0` 是交互，不是 crossover；两个 support 下 allocation 效应变号才是反转。
3. 1B 短句格式与裸答格式难度不假设相同；先报 N0 在两种格式下的可解性分层。
