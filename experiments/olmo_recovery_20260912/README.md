# OLMo 能力恢复：先准备，再验证

当前只做 OLMo，Llama 在 OLMo 路线稳定后再考虑。入口和历史依据见 [index.md](index.md) 与 [运行记录](../../docs/research/next_stage_20260912/5090_recovery_preparation.md)。

## 当前资源限制

无卡实例仅 0.5 核 / 2 GiB，本机也不承担重计算或大文件存储。只运行下面的 metadata inventory；不运行模型 CPU 测试、分词、数据生成、下载或大文件哈希。

从仓库根运行：

```bash
python3 -m experiments.olmo_recovery_20260912.readiness \
  --model /root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct \
  --sources /root/autodl-tmp/olmo_recovery_20260912/sources \
  --data /root/autodl-tmp/olmo_recovery_20260912/data
```

`METADATA_INVENTORY_ONLY` 不代表模型/数据已验证。`make_plan`、`train` 的 dry-run 及 `evaluate --dry-run` 会做完整文件校验，**也必须等有资源后再运行**。

## 原问题与第一比较

先把直接换表损坏、适配过程中的遗忘、答案/终止未学会分开。记录原始 Native 和未适配 EVQ/YaRN 在同一开发面板上的 0-step 行为，再对 Native、EVQ 使用同配方训练。官方 YaRN 保留强参照并使用相同预算完成匹配比较。32Mi CPT 是预设首段，8Mi 仅是同一轨迹的中间观察点，不能将中间阴性升级为方法失败。

优先使用旧 OLMo EVQ 表，实际 FP32 SHA 必须为 `917a52426b4ac986545c8ec73b115daae3c6515d6b9047f09d30c972ea1a4607`。原表文件当前仍需追回；可以重建并核对完全相同字节，但不得把另一张 τ=2 anchored 表当成旧结果的恢复。新的 anchored 合同单独命名，暂不进入第一比较。

所有训练臂使用 Q/K/V/O、gate/up/down LoRA；长 dense CPT、完整长指令答案/EOS、短 replay CE 与 Native teacher KL 分别归一化和记录。复合配方成功只能支持其联合恢复效果，不能自动归因 FFN 必要性。

## 有资源后的顺序

1. 在服务器准备真实 sources：PG19 原文及官方 split、固定 revision LongAlign、QASPER、来源分组的 Native replay。`experiments.evq_recovery.acquire` 提供公开语料下载实现；PG19 选书清单及历史 replay 必须先恢复并核对，不生成假占位数据。
2. 运行 `data_prepare`（需要 `--sources --model --source-tokenizer --output`），再运行 `data_audit`（`--data --model`）。保留原 tokenizer 身份，禁止跨词表直接复用 token IDs。
3. 运行 `make_plan --model MODEL --data-manifest DATA/data_manifest.json --out RUN --cosh-contract olmo_old_evq_exact --cosh-table ARRAY.npy`；冻结模型/代码/表/数据身份及共同预算。
4. 运行 `cpu_checks` 的真实 PEFT 小模型测试，再运行 `probe --root RUN --arm Cosh --execute` 做两步丢弃式真实 16K 显存检查。也核 Native/YaRN 的执行路径。显存/算子错误修复后才训练。
5. 用 `evaluate` 保存未适配表的 0-step 结果；按固定训练与验证计划运行 `train --root RUN --arm ARM --execute`，保存滚动断点及共同观察点。32Mi 主端点必须包含完整匹配比较。
6. 用 `evaluate` 的开发集闭合长端与原生能力问题；候选、表、评分及统计规则冻结后才打开 test。完整响应和 EOS、PPL、原正确样本 lost/gained 分列。

具体参数以各入口源码为准。本轮没有执行上述资源密集步骤，没有 GPU 或方法结果。旧 adapter 的 hash/逻辑名不等于可用文件，也不是新 r32 全线性配方的可恢复 checkpoint。
