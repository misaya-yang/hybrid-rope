# 8B LoRA 实验分析 (8B LoRA Experiment Analysis)

> 最后更新：2026-02-25
> 本文档记录 8B 规模实验的历史探索、失败分析与最新的公平重构方案。

## 1. TL;DR

- **旧版实验协议不公平**：早期的 8B Hybrid 实验由于使用了 forward monkey patch 而基线使用了 `rope_scaling`，导致底层核不一致，测试结果不可比且性能崩塌 (PPL@16K 达 11.8+)。
- **参数未调优饿死中频**：旧版缺乏 `rigid_j0` 高频锚定保护，导致中频信息分配不足，NIAH 等针刺任务失败。
- **已启动公平校验链**：全部重构为基于 `inv_freq.copy_()` 的统一注入法。当前 overnight 脚本正在运行 4 种方法（Baseline, PI, YaRN, Anchored Hybrid）的公平对比。

## 2. Claims (论文可用主张)

- **Claim 1 (Methodology)**: 评价频率缩减方法必须在相同的底层实现机制下进行，否则会引入框架误差。
  - *置信度*: High
  - *适用范围*: 扩展所有长文本 Llama 架构
  - *证据指针*: `scripts/debug_sigmoid_rope.py` (探针修复)
- **Claim 2 (Failure Mode)**: 极端高频直接缩放会导致局部注意力模板遗忘，引发零射击能力丧失和长程 PPL 劣化。
  - *置信度*: High
  - *适用范围*: 8B 及以上指令微调模型
  - *证据指针*: [-> BAD_8B_LORA_OLD] 历史运行记录与模型崩溃日志。

## 3. Evidence Map

| 实验 | 实验路径 / 数据路径 | 核心产物 | 可复现命令 / 脚本 |
|------|-------------------|----------|------------------|
| **旧版不公平实验 (废弃)** | `knowledge_base/08_8b_experiment_analysis.md` (旧版留存) | PPL劣化日志 | *(Deprecated)* |
| **新版公平流水线 (进行中)** | `results/overnight_8h/summary/` | 统一的 loss 与 NIAH | `bash scripts/run_fair_comparison.sh` |
| **验证探针与数值测试** | `archives/2026-02-22/scripts/_test_inject.py` | 验证 `inv_freq` 注入成功 | `python archives/2026-02-22/scripts/_test_inject.py` |

## 4. Failure & Fix (失败与修复)

- **Failure**: 旧版 Hybrid 方法 PPL@16K 为 11.875，远逊于 PI 的 6.136。
  - *原因*: 实现层面的不一致造成了 flash attention 核的行为差异；缺乏高频保护导致局部指令连贯性受损。
- **Fix**: 
  - 弃用 forward patch，统一通过 `.copy_()` 将四种策略的具体频率张量注入 `inv_freq` buffer。
  - 强制设定 `rope_scaling=None`。
  - 引入了 `rigid_j0=12` 的位级对齐保护核心高频。

## 5. What to write in paper (论文段落建议)

**段落：Implementation Mirages in Length Extension**
> "When scaling position encodings, seemingly innocuous implementation details—such as whether frequencies are overridden via buffer mutation or scaling APIs—can confound comparative evaluations. In our early 8B LoRA experiments, we observed that mixing arbitrary forward patch mechanisms with native scaling heuristics exaggerated the gap between methods, rendering ablation studies unreliable. To establish a rigorous testbed, we unify all interpolation techniques under a strict `inv_freq` injection protocol..."

## 6. Open Questions

- 新的严格公平协议下，`anchored_hybrid` 能否在 NIAH 任务中反超 YaRN？
- 如果 8B LoRA 的效果差距收窄，是否意味着 Llama-3 原生的注意力机制掩盖了部分频率重塑的带来的 PPL 增益？(取决于 overnight 数据)
