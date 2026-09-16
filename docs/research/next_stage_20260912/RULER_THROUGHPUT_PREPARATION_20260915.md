# RULER 完整生成吞吐：证据、实现与下一次 GPU canary

日期：2026-09-15。状态：**CPU 分析和计划代码完成；没有启动 GPU benchmark，没有修改现行 evaluator 或运行队列。**

本文件负责回答“2600 条 Full-13 RULER 为什么慢、可以如何更快保持同一科学问题”。研究方法与口头报告准备见本目录[索引](index.md)。实现为 [throughput.py](../../../experiments/native_enhancement_oral_20260915/throughput.py)，测试为 [test_native_enhancement_throughput.py](../../../tests/test_native_enhancement_throughput.py)。

## 1. 结论和实施顺序

1. **先测试等长 batch=2 的完整生成。** 它保留无 padding 路径，有真实面板配对空间；当前 clean32K 的 2600 个调用可以组成 876 个双样本 batch 加 848 个单样本，共 1724 个调用。这是调度数，尚不是加速比。
2. **进一步收益需要合法的变长 Flash attention 路径。** CPU 上按输入长度排序可以组成 1300 个双样本 batch，额外 prefill 平方工作量仅约 0.0096%。但当前 masked left-pad 路径有已完成的 kernel 失败证据，不能直接打开参数便承诺可用。
3. **不要再把 chunk 调参、减少 max_new_tokens 或优化官方 score 当作主要提速项目。** 已有 chunk 工程结果差异很小；真实输出很短，且存在用满预算的样本；官方 score CPU 成本可以忽略。
4. **方法探索与最终证据采用不同预算。** 用已定义的机制实验选择唯一冻结候选；最终保留完整任务、完整上下文和既定样本数。减少没有解释价值的新 arm 会直接减少计算，但不能把缩小后的面板冒充 Full-13 确认证据。

## 2. 本轮实测了什么

只读取了服务器已完成的 Llama clean32K TailSpline/MrPro 原始输出、输入面板、日志和既有运行合同。CPU 进程 `nice=19`、单 CPU affinity、`OMP_NUM_THREADS=1`，不加载权重、不调用 CUDA、不重复 hash 资产。

服务器证据根目录为 `/root/autodl-tmp/today_rope_plan_20260914`；以下路径均相对此根，属于服务器原始资产，不是 Git 分发文件：

- `tailspline_llama_s4_32k_ruler200_clean/assets/inputs.jsonl`
- `tailspline_llama_s4_32k_ruler200_clean/runs/{tailspline,mrpro}/generations.jsonl`
- 同 run 下 `contract.json`、`runtime_batching.json`、`status.json`
- `tailspline_llama_s4_32k_ruler200_clean/logs/clean_supervisor.log`
- `prefill_chunk_benchmarks/llama32k_32gb_v2.json`
- `tailspline_olmo_s4_naturalqa631/runtime/batch_canary/selection.json` 与 `logs/canary_batch4.log`

### 完整两臂的真实工作量

| 项目 | TailSpline | MrPro |
| --- | ---: | ---: |
| Full-13 样本数 | 2600 | 2600 |
| 真实完整输入 token 总数 | 83,489,402 | 83,489,402 |
| 输入长度 min / median / max | 28,270 / 32,370 / 32,606 | 相同 |
| 实际生成 token 总数（含 EOS） | 54,330 | 58,894 |
| 每条输出 median / mean / p90 | 7 / 20.90 / 52 | 7 / 22.65 / 64 |
| 以 EOS 结束 | 2573 | 2517 |
| 没有 EOS 且用满预算 | 27 | 83 |
| 完整 arm 日志时间 | 6h12m36s | 6h15m52s |
| 每条摊销 wall time | 8.598s | 8.674s |
| 2600 条官方 score 重算 CPU 时间 | 0.00624s | 0.00580s |
| 重算分数不一致 | 0 | 0 |

原面板单 CPU 流式解析耗时 6.76 秒；这不是每条重复解析的成本。官方 score 计时只包括 [ruler_bench.py](../../../scripts/experiments/olmo_fast_screen/ruler_bench.py) 的逐条匹配，不包括 JSON I/O、其他 F1 指标或配对 bootstrap。生成日志没有逐样本 prefill/decode 分项计时，因此**无法从本轮 CPU 数据精确分解 GPU 时间比例**。83.49M 输入、输出很短及既有单样本约 8.3 秒的结果共同支持先检查 prefill 与 batch 的判断。

max_new_tokens 的原合同分布是 30×200、32×400、50×200、120×200、128×1600。HF/custom greedy 已在 EOS 后结束，所以“预算 128，但平均只生成 21”并不表示每条浪费了 107 次 decode。T 的 27 个 hit-cap 全部位于 multiquery/multivalue；P 的 83 个分散在多项任务。降低预算会改变输出与任务成绩，不能作为保持合同的执行优化。

### 已有 chunk 实测应如何读

既有 receipt 在 `NVIDIA vGPU-32GB`、capability `(8,9)`、Torch `2.8.0+cu128` 上完成；不能把仓库 RTX 5090 **训练**的 1.9 倍提升当作本次生成的预期。GPU 细节来自该 receipt，未来开工时以实际设备为准。

| generation strategy | 单完整输出耗时 | peak allocated | token/score parity |
| --- | ---: | ---: | --- |
| direct，首次 | 8.7660s | 24.275GB | reference |
| chunk8192 | 8.3357s | 21.339GB | 完全相同 |
| chunk16384 | 8.3540s | 22.323GB | 完全相同 |
| chunk32768（此输入实际走 direct） | 8.3596s | 24.275GB | 完全相同 |

这里 GB 是十进制。8192 相对首次 direct 的差异混有首次运行开销；与后一次 direct 的时间只差约 0.3%。**该单行证据支持 chunk8192 的内存优势，不支持大幅加速承诺。** 生成和 LM 的建议不同：同 receipt 的 LM 推荐 direct，不能混用。

## 3. 现有代码里的实际限制

代码依据：[recovery_v2_eval.py](../../../experiments/olmo_recovery_20260912/recovery_v2_eval.py) 与 [runtime.py](../../../experiments/olmo_recovery_20260912/runtime.py)。

- 单样本 `greedy_tokens` 在输入长于 chunk 时走 DynamicCache + lower-right causal attention，并保持全左上下文。chunk 不减少理论上的完整 causal attention 工作量。
- `batched_greedy_tokens` 走 stock `model.generate`；当前代码**不会把 `prefill_chunk_size` 传到多样本 batch**。因此 `--batch-size 2 --prefill-chunk-size 8192` 不是已经存在的“batch2 chunk8192”实现。混合单/双 batch 还会混用 prefill 路径，未来 receipt 必须逐 batch 记录实际执行策略。
- chunked custom attention 明确限制 `batch==1` 且无 padding。仅去掉 assert 不足以支持变长 batch：padding mask、真实 position_ids、cache 对齐和 lower-right causal 对齐都要成立。
- stock masked left-pad 的 OLMo batch4 canary 实际在 `scaled_dot_product_attention` 处报 `RuntimeError: No available kernel. Aborting execution.`，selector 回落为 batch1。这是 kernel 可用性失败，不能报告为任务退化。
- Llama clean32K 目录另有 `tailspline_failed_leftpad_20260914T223301`，合同 batch2、left-pad=true，输出 0 行；本轮未找到其独立异常日志，因此不推断具体原因。
- 此 evaluator 的生成循环没有逐 batch `empty_cache()` 或 `gc.collect()`；“删除无条件清缓存”在这里没有现成可赚的收益。
- 当前 `run_clean_c.py` 要求与已完成 T/P 的 batch/prefill 合同相同。**不能把提速改动插入在跑的 X4。** 新执行策略通过 canary 后，在新的独立 run 目录使用；旧结果按科学身份复用，不能续写不同排序/执行身份的旧输出文件。

## 4. CPU 调度器已经准备的内容

`throughput.py` 只依赖标准库；可选官方 score 重算仅导入已有轻量评分函数。它流式读取输入并丢弃 token 数组，保留 row_id、task、长度和预算；检查完整输出匹配、重复/漏行、真实长度与结束标记。

调度只看**输入元数据**，不会用候选 arm 的答案、分数或观察到的输出长度决定样本分组。输出长度只用于事后估算同步 decode 的无效工作。

| B=2 计划 | batch 数 | prefill 平方工作相对逐条 | T/P 同步 decode token 工作相对逐条 |
| --- | ---: | ---: | ---: |
| 无 padding，按相同长度分组 | 1724 | 1.000000 | 1.1517 / 1.1590 |
| 按长度分组，masked left-pad | 1300 | 1.000096 | 1.2190 / 1.2431 |
| 再按 task 分组，masked left-pad | 1301 | 1.000206 | 1.1970 / 1.2073 |

CPU 还验证了 **cap → budget → task → length** 的 left-pad 分组，避免把长答案 multiquery 和短答案 single-key 混在一个同步 batch。它仍保留每条原预算；新的候选输出长度可能不同，历史浪费比例不是未来准确预测。该方案在 1% padding 占比限制下形成 1299 个双样本 batch 和 2 个单样本 batch。

`prefill_square_cost_ratio = Σ_b B_b Pmax_b² / Σ_i P_i²` 只是工作代理。同步 decode ratio 用历史完成输出计算 `Σ_b B_b max_i G_i / Σ_i G_i`；它包括 EOS token，近似描述短样本等待长样本，不是精确的 GPU instruction 数。减少 batch 调用可提高吞吐，也可能因已饱和的计算、额外 padding 和 EOS 等待而无收益，必须测。

Llama 配置为 32 layers、8 KV heads、head_dim128、BF16，KV tensor 为 **131,072 bytes/token**。32K 单样本约 4GiB，batch2 约 8GiB，权重本身约 16.1GB。CPU ceiling `65536` KV tokens 只是候选预算，不证明峰值显存可行：还需要模型激活、attention 临时张量与 allocator 空间。OLMo/Qwen 从自己的 config 计算，不按 Llama 数值猜。

可复现的 CPU 命令（从仓库根执行，输入文件必须已经完成）：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -m experiments.native_enhancement_oral_20260915.throughput \
  --panel /root/autodl-tmp/today_rope_plan_20260914/tailspline_llama_s4_32k_ruler200_clean/assets/inputs.jsonl \
  --generations /root/autodl-tmp/today_rope_plan_20260914/tailspline_llama_s4_32k_ruler200_clean/runs/tailspline/generations.jsonl \
  --model-config /root/autodl-tmp/models/Meta-Llama-3-8B-Instruct/config.json \
  --batch-size 2 --max-kv-tokens 65536 --elapsed-seconds 22356 \
  --rescore-official --out /tmp/ruler32k_tailspline_cpu_plan.json
```

该命令不提供 GPU execute 开关。测试命令：`python3 -m unittest tests.test_native_enhancement_throughput`，本轮 **9 项通过**。测试覆盖：不变的样本/预算/长度合同、无 padding 的等长要求、task 分组、KV ceiling、错误输出身份拒绝、生成计数和 GQA KV 内存公式；这些是 CPU 软件证据。

## 5. 下一次 GPU 工程实验：具体规格

### E1：等长 batch2，优先做

目的：验证 stock generate 的相同长度 batch 能否降低完成时间，并保留相同 greedy 输出。

1. 从 CPU plan 中选已有等长配对，覆盖 NIAH 单答案、multiquery/multivalue 长答案、VT、能形成同长度组的 QA。先用最长一组做 kernel/OOM 检查；失败即保留现行路径，不对整个面板试错。
2. 使用同模型、同冻结表、原完整 prompt 和原预算，比较 reference batch1+既有 prefill 与 candidate batch2 direct；保留少量单样本 direct 对照，区分 prefill 变化与 batching 的影响。
3. 首次通过后，对 13 个 task 各取长度两端的工程样本，另补已有 EOS=1-token 与 hit-cap 边界样本。不能只用一个很容易的 needle 就宣布所有输出等价。
4. 逐样本比较 `generated_ids`、`ended_eos`、`hit_cap`、官方分数、完整输出文字及其他被使用的指标；要求 token 级一致作为第一次采用的简单明确标准。若 token 不同但分数相同，记录差异，不自动合并新旧输出，应扩大针对差异的检查或维持 reference。
5. 单独记录 warmup 与 steady state；对同一工程面板重复 reference/candidate 的顺序做平衡。比较**整个面板 wall time**，并记录 CUDA prefill time、decode time、生成 tokens/s、max allocated/reserved、实际 attention backend、实际 prefill strategy、torch/transformers 版本。不把 cold-start 或一次最快样本当收益。

E1 的 GPU 代码需新建独立 canary runner；本次没有执行此阶段，也没有修改 active evaluator。先验证等长，避免把 kernel 修复和吞吐问题绑在同一实验里。

### E2：变长 batch，解决已知 kernel 不兼容后才测

有两条合法工程路径：支持真实 padding mask 的兼容 Flash backend；或可对变长序列正确 unpad/repad 的 varlen Flash attention。无论选哪条，都必须保证每条序列的独立 causal visibility、真实 position_ids、KV/cache_position、GQA 和当前静态 RoPE 表注入语义。

不能通过关闭 mask 让 padding 可见、把 pad 当作 prompt 内容，或偷偷打开会物化 32K² 矩阵的 math attention 来“使它运行”。也不能把 prompt 截到桶边界。native 4K/8K 可先测更小模型的路径正确性，再测目标模型/32K 的显存和吞吐；短窗口成功不等于长窗口可用。

现有 CPU plan 使用相同 budget/cap 分组，并限制 padding 占比；建议比较同 task 分组与跨 task 分组的真实 wall time。若任务输出长短差异占主导，再考虑 continuous batching、已结束行退出或 paged/static KV cache。vLLM 等新引擎只有确认本项目静态表、gain、position IDs、greedy 与完整输出合同兼容后才可进入此比较，本文件没有引擎兼容性或加速实测。

### E3：值得准备，但不应抢在 E1 前的优化

- **Decode compile/static cache**：可减少 kernel launch 与 cache 管理成本，但当前输出普遍较短，收益上限受 prefill 占比限制。先分项计时再投入，不把训练 compile 的收益搬过来。
- **同一模型内复用加载**：可以在每个完整 arm 之间安装静态表，免重复权重加载；32K 当前加载只约数秒，无法解释小时级耗时。跨 RoPE 表不可以复用 KV，因为中间隐藏状态及后续层的 K/V 已改变。
- **同一表、相同上下文、不同 query 的 prefix cache**：对本次 native 机制成组实验可能有价值；只有逐 token 相同前缀与因果模型才能复用，且每个分支 cache 必须独立。当前 RULER 各样本大多独立，不预设存在可观的公共 prefix。
- **更快设备/更大显存**：后续 Qwen 的卡型升级能提供 batch 与 backend 空间；必须以目标模型的短工程面板证明成本降低。没有硬件测量时，不给预计倍速。

## 6. 后续 agent 的交接边界

- 本次交付可直接用于 CPU 成本核算、选取工程样本、生成完整调度清单；不会启动模型。
- GPU 阶段尚未授权执行；按用户后续授权与当时真实队列安排。不要因本文件改动 X4 或抢占已有 GPU 进程。
- 新 run 要在自己的目录记录执行身份；不能把新排序继续追加到旧 `generations.jsonl` prefix。
- 工程 canary 不参与方法选择或扩大主张；冻结算法的科学确认仍按研究合同完整执行。
