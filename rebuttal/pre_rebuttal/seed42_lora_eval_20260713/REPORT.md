# Seed-42 LongAlpaca LoRA 长上下文评测报告（2026-07-13）

## 结论

本轮证据不能表述成“EVQ + LoRA 完全没有 NIAH 能力”。更准确的结论是：

1. EVQ + LoRA 在 8K 仍有明显的检索信号；在当前未混入检索监督的 LongAlpaca LoRA 配方下，16K 的 NIAH/Passkey 能力接近失效，32K 基本失效。
2. EVQ + LoRA 的 2026 temporal holdout 前缀 PPL 在 16K 为 24.068，显著低于 Geo + LoRA 的 108.958；但低 PPL 没有转化成稳定的长上下文精确检索。这两个指标测量的是不同能力。
3. 官方 YaRN 推理扩展没有在 pilot 中恢复 Passkey/NoLiMa：Geo + LoRA 和 EVQ + LoRA 的 YaRN x2/x4 均为 0。
4. 因而，把 NIAH/Passkey 或其他明确的长程检索监督混入微调数据，是下一步最直接、最可检验的方案；当前结果支持这个实验动机，但尚不能证明它是唯一可行方案。

本报告只记录本轮真实生成并已拉回的原始结果，不修改论文中的任何实验数字，也不把单 seed、pilot 或 supporting 结果升级为主结论。

## 训练与制品身份

| 项目 | Geo + LoRA | EVQ-Cosh + LoRA |
|---|---:|---:|
| 基座 | Meta-Llama-3-8B-Instruct | Meta-Llama-3-8B-Instruct |
| seed | 42 | 42 |
| 训练长度 | 8,192 | 8,192 |
| 数据 | Yukang/LongAlpaca-12k | Yukang/LongAlpaca-12k |
| 训练样本 | 7,476 | 7,476 |
| 步数 | 300 | 300 |
| LoRA | r=64, alpha=128；q/k/v/o | r=64, alpha=128；q/k/v/o |
| 频率 | native geometric RoPE | EVQ-Cosh, tau=1.414 |
| 最终训练 loss | 2.4817 | 3.1090 |
| 训练时间 | 2.284 h | 2.281 h |
| adapter SHA-256 | `0e7efa6e...ebec9a` | `8ea04234...0c780` |
| frequency SHA-256 | `09fab0f4...bb794` | `49b20592...bacf` |

两臂共享训练数据 manifest、模型 manifest、训练代码哈希和除频率 substrate 外的训练协议。训练状态均为 `complete`、`global_step=300`。

## 语言建模：2026 temporal holdout

这是三臂 matched-pack teacher-forced NLL/PPL 评测；三个域为 arXiv 2026、Federal Register 2026 和 Stack Overflow 2026。每臂、每 pack 只做一次 32K backbone forward，8K/16K 是严格因果前缀聚合。

| arm | 8K NLL / PPL | 16K NLL / PPL | 32K NLL / PPL |
|---|---:|---:|---:|
| Geo base | 2.0730 / 7.948 | 5.0143 / 150.545 | 7.3085 / 1492.915 |
| Geo + LoRA | 1.9195 / 6.817 | 4.6910 / 108.958 | 6.8992 / 991.475 |
| EVQ + LoRA | 2.3093 / 10.068 | 3.1809 / 24.068 | 4.8513 / 127.911 |

EVQ + LoRA 在 8K PPL 略差，但在 16K/32K 明显优于两个 Geo 臂。该结果证明语言建模损失随长度增长得更慢，不证明模型能从长上下文中精确取回一个离散 needle。

EVQ + LoRA 分域 PPL：

| 域 | 8K | 16K | 32K |
|---|---:|---:|---:|
| arXiv 2026 | 20.403 | 40.694 | 154.046 |
| Federal Register 2026 | 5.456 | 14.333 | 75.970 |
| Stack Overflow 2026 | 9.166 | 23.903 | 178.827 |

## NIAH/RULER-style log-prob 结果

下表来自当前仓库已有的 raw/synthetic RULER-style log-prob evaluator，每个单元 10 trials。它不是 NVIDIA 官方 RULER 生成器的正式分数，因此只用于诊断能力随长度的变化。

| 任务 | 8K Top-1 | 16K Top-1 | 32K Top-1 |
|---|---:|---:|---:|
| S-NIAH | 56.67% | 3.33% | 0.00% |
| MK-NIAH | 63.33% | 0.00% | 0.00% |
| KV-Retr | 72.22% | 19.84% | 3.47% |
| VT | 36.67% | 0.00% | 0.00% |

对应趋势很清楚：8K 并非没有能力；16K 只在 S-NIAH 和 KV-Retr 留下弱信号；32K 除 KV-Retr 的 3.47% 外基本为零。

## 额外 PE probes（exploratory）

| probe | 8K | 16K | 32K |
|---|---:|---:|---:|
| multi-depth passkey exact match | 100% | 0% | 0% |
| multi-needle recall | 20% | 0% | 0% |
| KV association | 0% | 0% | 0% |
| positional ordering Kendall tau | -0.10 | 0.00 | 0.00 |

该 quick evaluator 的 generation 路径没有显式传入全 1 attention mask，因此这里只把它作为探索性佐证；不能用来替代最终 capability evaluator。

## 官方 YaRN capability pilot

本轮“官方 YaRN”指固定到 `jquesnelle/yarn@995db5b` 的 correction range、linear ramp 和 `mscale = 1 + 0.1 ln(scale)`。Geo 直接在 native geometric 坐标上应用；EVQ 在保持 EVQ substrate 的前提下，通过 virtual coordinate 应用同一官方 YaRN 算子。x2/x4 分别对应 16K/32K 扩展配置。

冻结 capability manifest 共 1,603 条：

| 任务族 | 数量 |
|---|---:|
| deterministic Passkey | 300 |
| NoLiMa-Hard | 500 |
| MCQA：MMLU/ARC/HellaSwag/OpenBookQA/WinoGrande | 500 |
| LongBench：NarrativeQA/Qasper | 303 |

Pilot 不是完整 1,603 条全量评测：每个 task/length/depth cell 只取一个固定样本，用来快速覆盖所有任务族并验证完整执行链。

### Geo + LoRA + 官方 YaRN

| 项目 | YaRN x2 | YaRN x4 |
|---|---:|---:|
| Passkey 8K，5 depths | 0/5 | 0/5 |
| Passkey 16K，5 depths | 0/5 | 0/5 |
| Passkey 32K，5 depths | 0/5 | 0/5 |
| NoLiMa-Hard 16K，5 depths | 0/5 | 0/5 |
| NoLiMa-Hard 32K，5 depths | 0/5 | 0/5 |
| NarrativeQA F1，1 example | 0.0714 | 0.0896 |
| Qasper F1，1 example | 0.1579 | 0.0426 |
| MCQA，5 tasks 各 1 example | 1/5 | 1/5 |

MCQA 的唯一命中都是 ARC Challenge。由于每个 MCQA/LongBench 任务只有一个样本，这些数值只能确认执行链，不能做模型优劣结论。

### EVQ + LoRA + 官方 YaRN

| 项目 | YaRN x2 | YaRN x4 |
|---|---:|---:|
| Passkey 8K，5 depths | 0/5；NLL 4.985 | 0/5；NLL 4.593 |
| Passkey 16K，5 depths | 0/5；NLL 7.662 | 0/5；NLL 4.921 |
| Passkey 32K，5 depths | 0/5；NLL 10.660 | 0/5；NLL 5.714 |
| NoLiMa-Hard 16K，5 depths | 0/5；NLL 13.200 | 0/5；NLL 12.092 |
| NoLiMa-Hard 32K，5 depths | 0/5；NLL 13.630 | 0/5；NLL 11.787 |
| NarrativeQA F1，1 example | 0.0274 | 0.0580 |
| Qasper F1，1 example | 0.1471 | 0.1481 |
| MCQA，5 tasks 各 1 example | 1/5 | 2/5 |

x2 的唯一 MCQA 命中是 MMLU；x4 命中 MMLU 和 ARC Challenge。YaRN x4 明显降低了当前 Passkey gold-answer 的 teacher-forced NLL，但 autoregressive exact match 仍为 0。这再次表明“答案 token 概率改善”和“稳定生成正确答案”不能混为一谈。完整 pilot 用时 369.62 秒，峰值 CUDA 显存 24.49 GB。

## 解释：为什么 PPL 没崩但 NIAH 崩了

PPL 是对整段自然文本 next-token 分布的平均损失。模型可以利用局部语法、主题一致性和高频模式得到较低 PPL，而无需把远处的一个 key/value 精确绑定并在回答位置复制出来。

NIAH/Passkey 要求的是更窄、更苛刻的计算链：识别 needle、跨长距离保存键值绑定、抵抗大量干扰、在特定提示格式下精确生成答案。EVQ 改善频率 substrate 和长距离语言建模稳定性，并不自动给 LoRA 注入这项行为监督。当前 LongAlpaca 配方主要训练长指令/自然文本建模，因此出现“16K PPL 24.1，但 NIAH 接近零”并不矛盾。

## 有效性边界与未完成项

- 这是 seed 42 的 supporting/evidence-discovery 结果，不能升级为多 seed 主结论。
- 官方 YaRN capability 是 pilot cell coverage，不是全 1,603 条样本结果。
- 官方 RULER 数据生成器因服务器环境缺少 `tenacity` 未执行；没有在付费 GPU 窗口临时安装依赖。报告中的 RULER 表明确标为仓库现有的 RULER-style log-prob evaluator。
- 一次旧 MCQA 命令因数据加载失败退化到 synthetic fallback；该进程已停止，其输出被判无效，未写入任何结果表。
- NoLiMa/LongBench/MCQA 的冻结数据来源和 SHA-256 已记录在 `raw/capability_manifest.json`。
- 报告未引入或修改任何论文实验数字。

## 原始证据索引

- `raw/legacy_results/temporal_three_arm_2026.json`：三臂 temporal holdout NLL/PPL。
- `raw/evq_lora_s42_ruler_logprob_full/ruler_logprob_evq.json`：RULER-style log-prob 全表。
- `raw/evq_lora_s42_pe_probes_quick/pe_probes_evq.json`：exploratory PE probes。
- `raw/geo_lora_s42_official_yarn_all_pilot.json`：Geo + LoRA + 官方 YaRN 全任务族 pilot。
- `raw/evq_lora_s42_official_yarn_all_pilot.json`：EVQ + LoRA + 官方 YaRN 全任务族 pilot。
- `raw/{geo,evq}_lora_s42_official_yarn_pilot.json`：两臂先行 pilot。
- `raw/capability_manifest.json`：冻结 capability 数据 manifest。
- `raw/training_metadata/{geo,evq}/`：训练协议、trainer state 和 artifact provenance。
- `raw/legacy_results/*.log`、`raw/*official_yarn*.log`：运行日志和失败证据。
