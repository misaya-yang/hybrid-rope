# Round 12 执行手册 v2（2026-09-06，服务器 westc 27741，4080 SUPER **32GB**）

> **状态（2026-09-06 晚）**：Round 12 已执行（Track A 全矩阵 + Y2 + Qwen 对照 +
> E1 + 审计），用户指令中止全部实验并关机。结果与恢复清单见
> `REPORT_ROUND12_20260906.md`。
> **更正**：开机实测显存 32760MiB（32GB），v1/v2 中"16GB"预期作废——
> 7B bf16 载入 14.6GB、16K 前向 peak 16.4GB 均通过（见 `probe4080/probe_result.json`）。

**v2 变更（相对 v1）**：按 Pro 第一性原理综合文档（2026-09-06）+ 用户 09-06 裁决
（"不收窄、重排优先级；7B/更多数据用 4080 试；三者结合"）重排优先级：
新增 Phase 0（Pro §11 两个只读诊断 + 4080/7B 能力探测）置顶；
Track A 保持；Track B 降级为条件执行（需当日用户确认）；7B 由探测结果门控。
**v1 的冻结资产与配方全部保持不变**（表/任务/视图/CPT 数据/评分口径）。

约定：单 GPU 进程；每步完成先核对 manifest 再进下一步；失败保留现场不续跑。
路径根：`B12=/root/autodl-tmp/claude_round12_20260906`
R11 根：`R11=/root/autodl-tmp/claude_round11_olmo_20260905`
1B 模型：`/root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct`
7B 模型：`/root/autodl-tmp/models/OLMo-2-1124-7B-Instruct`
PY=`/root/miniconda3/bin/python`

## 0. 开机检查（切卡模式后）
```
nvidia-smi                      # 必须有卡（实测 4080 SUPER 32GB）
df -h /root/autodl-tmp          # 剩余空间（09-06 收尾时 ~10G 空闲，注意别写满）
ls $B12/tables/manifest_round12.json $B12/tasks/round12_tasks.jsonl \
   $B12/data/sft/views.jsonl $B12/data/cpt/manifest.json
sha256sum $B12/data/cpt/train_2048x16385.npy $B12/data/cpt/validation.npy
# train 期望 7ab6f784…243938；val 期望 b7da15e1…059cb3f13d
```

## Phase 0 — P0 诊断与能力探测（新置顶，预算 ≤2.5 GPU-h）

依据：Pro 综合文档 §11/§12（只读、不新增表/训练/候选）。已登记在
`PREGLUCTION_ROUND12_V2_ADDENDUM.md`。所有输出进 `$B12/diag/`。

### 0.1 P0a — 内容分叉 E/B（§11.2），5 个系统各一跑（单进程串行）
```
cd $B12/code
# T0（无逐行 far receipt → registered 记 null，聚合值引用 review.json）
$PY diag_fork_eb.py --model <1B> --tables $B12/tables --system-id T0 --arm N \
  --views $R11/olmo_tasks/transport_views.jsonl \
  --family single_evidence --split validation --layouts compact near far \
  --out $B12/diag/fork_eb_T0
# Z0
$PY diag_fork_eb.py ... --system-id Z0 --arm Z \
  --receipts $R11/out/z0/examples.jsonl --out $B12/diag/fork_eb_Z0
# ZC
$PY diag_fork_eb.py ... --system-id ZC --arm Z --adapter $R11/out/train/step_128 \
  --receipts $R11/out/task128/examples.jsonl --out $B12/diag/fork_eb_ZC
# ZF
$PY diag_fork_eb.py ... --system-id ZF --arm Z --adapter $R11/out_zf/train/step_128 \
  --receipts $R11/out_zf/task128/examples.jsonl --out $B12/diag/fork_eb_ZF
# ON
$PY diag_fork_eb.py ... --system-id ON --arm N --adapter $R11/out_on/train/step_128 \
  --receipts $R11/out_on/task128/examples.jsonl --out $B12/diag/fork_eb_ON
```
核对：每个目录 `manifest.json` status=DIAG_FORK_EB_COMPLETE；
`ce_identity_residual` 应 ≤1e-6（§6.2 恒等式精确性自检）。

### 0.2 P0b — attention-write / logit 分解（§11.1），T0 必须先跑
```
# 两个预注册实例（规则与锁定依据见 PREGLUCTION_ROUND12_V2_ADDENDUM.md §3）
INST="0ee492a7351f230cc7aac34a6970df2e00f7aa838fb5569e3fbd8cc9208a9a2d 50b88ec1b6454878c549aa1e5cd39cb8ebdde7979aeddb98d9bbd97350b7afa0"
$PY diag_write_decomp.py --mode system --system-id T0 --arm N --model <1B> \
  --tables $B12/tables --views $R11/olmo_tasks/transport_views.jsonl \
  --proofs $R11/olmo_tasks/source_proofs.jsonl --instances $INST \
  --out $B12/diag/wd
for S in Z0 ZF ON; do
  case $S in Z0) ARM=Z; AD="";; ZF) ARM=Z; AD="--adapter $R11/out_zf/train/step_128";;
             ON) ARM=N; AD="--adapter $R11/out_on/train/step_128";; esac
  $PY diag_write_decomp.py --mode system --system-id $S --arm $ARM $AD \
    --model <1B> --tables $B12/tables \
    --views $R11/olmo_tasks/transport_views.jsonl \
    --proofs $R11/olmo_tasks/source_proofs.jsonl --instances $INST \
    --ref-dump $B12/diag/wd/T0/ref_dump.npz --out $B12/diag/wd
done
$PY diag_write_decomp.py --mode report --out $B12/diag/wd
```
核对：每系统 `outer_recon_rel_far` ≤5e-3（hook/对齐保真度）；
`identity_rel_residual` ≤1e-4（§5.2 代数恒等式）；§2.3 `rel_residual_p99` ≤1e-3。
任一失败 = 停，保留现场，报告（不得调容差救场）。

### 0.3 4080/7B 能力探测（预算 ≤0.5 GPU-h，失败即结果）
```
$PY probe_capability.py --model-1b <1B> --model-7b <7B模型路径> --out $B12/probe4080
```
预期（**已作废**，实测卡为 32GB）：stage1 通过；stage2 7B bf16（~15.2GB 权重）在 16GB 卡上加载即 OOM。
实际：stage2 载入 14.6GB、stage3 16K 前向 peak 16.4GB，全过 → 7B 矩阵放行。
**OOM 边界本身就是结果**，记录于 `probe_result.json`，不强行量化绕过
（量化改变数值口径，属方法变更，需另行决策）。

## Phase 1 — Track A 1B 静态矩阵（保持，≤3 GPU-h）
round12_tasks 的 single_evidence 行是 split=train，与 round-11 receipt（validation）
不逐位相同 → 本轮全部重跑，不复用旧 receipt。
```
for A in N Z Y M; do
  $PY track_a_eval.py --model <1B> --model-id olmo1b --tables $B12/tables \
    --arm $A --tasks $B12/tasks/round12_tasks.jsonl --output $B12/track_a/olmo1b_$A
done
```
顺序 N 先（吞吐基线 + Native 逐项披露基线）。每格核对 `manifest.json` + examples 行数。

## Phase 2 — Track A 7B（由 0.3 探测门控）
- 若 `probe_result.json` stage2 = FAIL_OOM：**跳过 7B 矩阵**，把边界写进当日记录
  （升级路径：≥24GB 卡；或另行决策量化口径）。不启动任何 7B 前向。
- 若 7B bf16 可载入且 16K 前向通过：先资格
  `$PY track_a_eval.py --model <7B> --model-id olmo7b --arm N --tables $B12/tables \
     --tasks $B12/tasks/round12_tasks.jsonl --families single_evidence --lengths 2048 \
     --output $B12/track_a/olmo7b_N_qualif`（可先加 `--smoke` 跑 2 条做运行时检查，
  再 `--max-rows-per-cell 64` 跑资格），通过后再 {N,Z,Y,M} 全量。
- 任一长度 OOM：记录最大可行长度，停止更长格，不得静默改口径。

## Phase 3 — Track B 训练（降级、条件执行，≤12 GPU-h）
**启动前必须当日用户明确确认**。背景如实标注：Pro 综合文档 §0/§14 明确
"当前不应执行 7B / 33M-token CPT / LoRA+norm+embedding 组合矩阵"；
用户 09-06 裁决保留本路线但降级。若 Phase 0 诊断已给出改变前提的信息
（如 §11.3 决策表显示训练信号不修复内容方向），先向用户汇报再决定。
配方不变：R12_DUAL_ARM_CPT_SFT_V1（CPT_DATA_FROZEN_V1，512 步 ×4×16K，
token 上限 33,554,432 硬断言；不加数据——"更多数据"=配方变更，需另行决策）。
```
$PY track_b_train.py --arm Y --model <1B> --tables $B12/tables \
  --cpt-data $B12/data/cpt --sft-views $B12/data/sft/views.jsonl \
  --replay-manifest /root/autodl-tmp/ffn_review_execution_20260904/native_pool_v3/manifest.json \
  --out-root $B12/track_b/Y_CPT_SFT --phase all
# Z 臂同构（--arm Z --out-root $B12/track_b/Z_CPT_SFT）
```
检查点：CPT128/256/512 + SFT64；候选= CPT512/SFT64，Native-first，暴露后不重选。

## 预算闸门（4080 SUPER 32GB，实测）
Phase 0 ≤2.5h；Phase 1 ≤3h；Phase 2 ≤0.5h（探测）+ 条件 ≤6h；Phase 3 ≤12h。
当日总 ≤20 GPU-h；任何一步预计超支 → 停并记录，不得静默超支。
单 GPU 进程；失败现场保留不自动续跑。

## Phase 0 之后（汇报义务）
P0 完成后先产出中文小结：E>|B| 与否（分系统/布局）、§5.2 三项相对大小与方向、
§2.3 直接项/继承项占比（S vs D），对照 Pro §11.3 决策表逐条标注落到哪一行。
**只汇报与解读，不据此自动启动任何新训练/新候选**（§11.3：研究责任，不是配方）。
