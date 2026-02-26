# Qwen Fast400 双 Seed 评测分析研判报告（2026-02-26）

## 1. 结论摘要（可直接汇报）

- 本次“双 seed”最终可核验结果为 `seed=42` 与 `seed=1337`（未检索到 `seed=1432` 对应目录或产物）。
- 在 `LongBench-21` 全量评测下，`Anchored-Sigmoid` 相对 `Baseline` 为**稳定负向差值**：
  - seed42: `44.4355 -> 44.0830`（`-0.3525` pct）
  - seed1337: `44.4701 -> 44.0522`（`-0.4179` pct）
  - 双 seed 均值差：`-0.3852` pct
- 统计上（联合 2 seeds、per-sample）：
  - diff(raw)=`-0.004387`（约 `-0.4387` pct）
  - 95% CI(raw)=`[-0.007776, -0.000971]`
  - `p_raw=0.0102`, `p_fdr_bh=0.0102`，结论为 `no_improvement`。
- 结构性现象是“**检索涨、复杂多文档推理和部分摘要/代码掉**”，典型 trade-off，而非全局提升。

## 2. 数据来源与核验范围

- 评测目录（服务器）：
  - `artifacts/reviewer_2026-02-25/h2_qwen_fast400_seed42/downstream_eval_fast400/longbench/`
  - `artifacts/reviewer_2026-02-25/h2_qwen_fast400_seed1337/downstream_eval_fast400/longbench/`
- 统计汇总（联合双 seed）：
  - `artifacts/reviewer_2026-02-25/h2_qwen_fast400_seed42/significance_full21_fdr_qwen.json`
  - `artifacts/reviewer_2026-02-25/h2_qwen_fast400_seed42/significance_full21_fdr_qwen.csv`
  - `artifacts/reviewer_2026-02-25/h2_qwen_fast400_seed42/claim_policy_report.md`
- 评测协议关键项（从结果 meta 读取）：
  - base model: `Qwen2.5-7B-Instruct`
  - task set: `lb21`（全量）
  - `max_samples_per_task=0`（全样本）
  - `max_input_tokens=16384`
  - `prompt_source=official`, `chat_template=auto`, `truncate_mode=middle`
  - `score_scale=pct`, `strict_parity_check=true`

## 3. 全局指标对比（LongBench-21 宏平均）

| Seed | Baseline | Anchored | Delta (Anchored-Baseline) |
|---|---:|---:|---:|
| 42 | 44.4355 | 44.0830 | -0.3525 |
| 1337 | 44.4701 | 44.0522 | -0.4179 |
| Mean | 44.4528 | 44.0676 | -0.3852 |

附：每个 seed 的任务级胜负统计：
- seed42：`8` 胜 / `11` 负 / `2` 平
- seed1337：`9` 胜 / `12` 负 / `0` 平

## 4. 任务层面的“涨/跌”分解

### 4.1 平均涨幅 Top（两 seed 平均）

| Task | Mean Delta (pct) | seed42 | seed1337 |
|---|---:|---:|---:|
| passage_retrieval_en | +2.5000 | +2.0000 | +3.0000 |
| narrativeqa | +1.3640 | +1.2341 | +1.4940 |
| triviaqa | +0.8877 | +0.8301 | +0.9452 |
| lsht | +0.5000 | +0.0000 | +1.0000 |
| multifieldqa_en | +0.2685 | +0.2886 | +0.2483 |

### 4.2 平均跌幅 Top（两 seed 平均）

| Task | Mean Delta (pct) | seed42 | seed1337 |
|---|---:|---:|---:|
| musique | -3.2538 | -2.8648 | -3.6428 |
| 2wikimqa | -2.6386 | -3.0177 | -2.2595 |
| hotpotqa | -2.1799 | -1.9463 | -2.4135 |
| repobench-p | -1.4184 | -1.2037 | -1.6332 |
| dureader | -1.1228 | -0.7492 | -1.4965 |

## 5. 任务族群（机制层面）解读

按 LongBench 常见任务族群聚合（两 seed 平均）：

| 族群 | 平均 Delta (pct) | 结论 |
|---|---:|---|
| single_doc_qa | +0.5226 | 小幅正向 |
| multi_doc_qa | -2.2988 | 明显负向（核心短板） |
| summarization | -0.2542 | 轻微负向 |
| few_shot | +0.2214 | 轻微正向 |
| synthetic | +0.2185 | 结构性分化（EN 检索涨，ZH/count 掉） |
| code | -0.7545 | 负向 |

研判：
- 当前 Anchored 配置更像在增强“定位/检索型信号”（尤其 `passage_retrieval_en`），
- 但牺牲了“跨段聚合+多跳推理”（`musique/2wikimqa/hotpotqa`）与代码任务稳定性。

## 6. 统计显著性与可宣称边界

来自联合双 seed 的 `significance_full21_fdr_qwen.json`：

- `Anc-Sig vs Baseline | per_sample`
  - diff(raw)=`-0.004387`
  - 95% CI(raw)=`[-0.007776, -0.000971]`
  - `p_raw=0.0102`, `p_fdr_bh=0.0102`, `p_fdr_by=0.0102`
  - `claim_grade=no_improvement`

- `Anc-Sig vs Baseline | per_task`
  - FDR-BH 显著的负向任务：`musique`, `qmsum`
  - 正向任务里 `passage_retrieval_en` 虽有提升，但 `p_fdr_bh=0.06636`，未过 0.05。

结论：当前证据不支持“整体提升”叙述，只支持“机制性 trade-off”叙述。

## 7. 评测质量与管线健康

四组结果（2 seeds × 2 methods）均满足：
- `num_selected = num_scored = 4750`
- `generation_error = 0`
- `parse_fail = 0`
- `empty_output = 0`
- `truncation_at_question = 0`

模板泄漏率：
- seed42: baseline `0.526%` vs anchored `0.842%`
- seed1337: baseline `0.505%` vs anchored `0.695%`

说明：评测管线稳定，结果可用；差异主要来自模型行为，不是评测崩溃。

## 8. 对“seed=1432”的核验说明

- 在服务器 `artifacts/` 全量目录检索 `1432/seed1432/s1432` 未发现匹配路径。
- 本轮统计与报告基于可复现且完整的双 seed：`42 + 1337`。
- 若你确认“1432”是另一组未归档 run，需要先提供具体产物目录后再并入复算。

## 9. 下一步实验建议（针对当前结果最小增益路径）

1. 固定当前管线不变，先做“长程推理修复”单变量实验：仅扫 `anchor_factor/slope/center`，目标缩小 `multi_doc_qa` 负差。
2. 增加语言对齐检查：`passage_retrieval_zh` 与 `dureader` 同时掉分，优先排查中文子任务的 prompt/解码细节与模板泄漏。
3. 若主张“理论指导有效”，当前稿件建议改为：
   - 主结论：`directional trade-off consistent with mechanism`
   - 避免表述为“overall improvement”。
4. 补做 sign-test 风格条件计数时，不要混合不同任务族群；建议按 `retrieval` 与 `multi-hop reasoning` 分别报告。

