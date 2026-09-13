# OLMo 能力恢复：先准备，再验证

当前只做 OLMo，Llama 在 OLMo 路线稳定后再考虑。入口和历史依据见 [index.md](index.md) 与 [运行记录](../../docs/research/next_stage_20260912/5090_recovery_preparation.md)。

## 当前资源限制

无卡实例仅 0.5 核 / 2 GiB，本机也不承担重计算或大文件存储。用户已另行授权原始数据提前下载到服务器：可以运行 `download_sources`，但不运行模型 CPU 测试、分词或案例生成。

从仓库根运行：

```bash
python3 -m experiments.olmo_recovery_20260912.readiness \
  --model /root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct \
  --sources /root/autodl-tmp/olmo_recovery_20260912/sources \
  --data /root/autodl-tmp/olmo_recovery_20260912/data
```

`METADATA_INVENTORY_ONLY` 仅为文件/config清单。用户确认可信克隆，当前5090入口为westd:24904；后续不做SHA校验，不因代码/manifest字节变化阻塞实验。启动仅做存在性、模型配置、8K/16K长度、arm/预算/恢复位置等轻量检查。

## 原问题与第一比较

先把直接换表损坏、适配过程中的遗忘、答案/终止未学会分开。活动训练只比较Native与EVQ/Cosh。官方YaRN依赖不同的全参数长窗SFT协议，当前LoRA套壳不再作为匹配训练对照；已经产生的YaRN零训练结果仅保留历史参考。32Mi CPT是预设首段，8Mi仅是同一轨迹的中间观察点。

当前默认使用可直接构造的 OLMo fixed-support anchored Cosh τ=2。它回答当前解析分配的恢复能力，不宣称复播旧 adapter。旧 EVQ exact 模式保留为可选，不等待旧字节文件，不等待4080训练完成。

所有训练臂使用 Q/K/V/O、gate/up/down LoRA；长 dense CPT、完整长指令答案/EOS、短 replay CE 与 Native teacher KL 分别归一化和记录。复合配方成功只能支持其联合恢复效果，不能自动归因 FFN 必要性。

实际长程训练按8K、8K、16K轮转（OLMo原生4K），两种长度各占一半CPT tokens。长指令对应prompt至少7K/14K，总序列不超过8K/16K，保留完整答案和EOS。Native表示原始频率表，也使用这套长输入；4K以内样本仅做replay/KL。32Mi端点每臂3072次更新，计数与恢复按真实累计tokens。

## 有资源后的顺序

1. `download_sources --root SOURCES --ruler-upstream RULER --execute` 下载PG19、LongAlign、Dolly短replay和QASPER，复用已有RULER/NIAH原始数据。无需等旧Native池。
2. 开卡有资源后运行 `data_prepare --sources SOURCES --model MODEL --output DATA`，再运行 `data_audit --data DATA --model MODEL`。使用旧native_rows才加 `--source-tokenizer`；默认Dolly直接用目标模板。
3. 运行 `make_plan --model MODEL --data-manifest DATA/data_manifest.json --out RUN`；默认生成Native/Cosh两臂合同。
4. 复用已有PEFT/恢复检查结果；新8K/16K配置仅做一次必要的 `probe --root RUN --arm Cosh --execute`，确认显存和有限loss后进入训练，不重复检查或扫描SHA。
5. 用 `evaluate` 保存未适配表的 0-step 结果；按固定训练与验证计划运行 `train --root RUN --arm ARM --execute`，保存滚动断点及共同观察点。32Mi 主端点必须包含完整匹配比较。
6. 用 `evaluate` 的开发集闭合长端与原生能力问题；候选、表、评分及统计规则冻结后才打开 test。完整响应和 EOS、PPL、原正确样本 lost/gained 分列。

具体参数以各入口源码为准。本轮没有执行上述资源密集步骤，没有 GPU 或方法结果。旧 adapter 的 hash/逻辑名不等于可用文件，也不是新 r32 全线性配方的可恢复 checkpoint。
