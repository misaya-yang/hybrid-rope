# Qwen Fast400 无提升诊断报告（2026-02-26）

## 1. 结论先行

截至 2026-02-26 07:41 CST，已完成结果显示：
- `seed42` 上 `anchored_sigmoid` 相比 `baseline` 的 LongBench-21 平均分为 **-0.3525 pct**（44.0830 vs 44.4355）。
- 这是“有涨有跌但整体略负”的形态，不是全线崩溃。
- `seed1337` 的 `anchored_sigmoid` 仍在运行，当前不能下最终结论。

当前最合理判断：
- **暂不支持“anchored_sigmoid 在本轮设置下稳定优于 baseline”**。
- 更接近“收益与副作用并存，且在当前预算/数据下净收益不足”。

## 2. 证据范围与数据来源

分析基于以下服务器产物（只读提取）：
- `artifacts/reviewer_2026-02-25/h2_qwen_fast400_seed42/downstream_eval_fast400/longbench/baseline.json`
- `artifacts/reviewer_2026-02-25/h2_qwen_fast400_seed42/downstream_eval_fast400/longbench/anchored_sigmoid.json`
- `artifacts/reviewer_2026-02-25/h2_qwen_fast400_seed1337/downstream_eval_fast400/longbench/baseline.json`
- `artifacts/reviewer_2026-02-25/h1_baseline_gold_seed42/longbench_lb21_baseline_gold_seed42.json`
- `artifacts/cross_model_fast_tuned_b1_gc/*/artifacts/summary.json`
- `artifacts/reviewer_2026-02-25/h2_qwen_fast400_seed1337/downstream_eval_fast400/longbench/eval_longbench.log`

## 3. 关键观察

### 3.1 协议一致性（先排除“比较不公平”）

`seed42 baseline` 与 `seed42 anchored` 的关键评测开关一致：
- `task_set=lb21`
- `tasks` 完全一致（21任务）
- `max_samples_per_task=0`
- `max_input_tokens=16384`
- `prompt_source=official`
- `chat_template=auto`
- `truncate_mode=middle`
- `max_new_tokens_policy=official`
- `score_scale=pct`
- `strict_parity_check=True`
- `manifest_json` 路径一致

结论：本次 `seed42` 对比具备公平性，不是协议漂移导致的假差异。

### 3.2 总分与任务计数

- `seed42 baseline_avg = 44.4355`
- `seed42 anchored_avg = 44.0830`
- `delta = -0.3525`
- 任务胜负：`up/down/eq = 8/11/2`

### 3.3 提升/下降任务结构

涨幅前 5：
1. `passage_retrieval_en +2.0000`
2. `narrativeqa +1.2341`
3. `triviaqa +0.8301`
4. `multifieldqa_zh +0.8224`
5. `lcc +0.6471`

跌幅前 5：
1. `2wikimqa -3.0177`
2. `musique -2.8648`
3. `hotpotqa -1.9463`
4. `passage_count -1.3375`
5. `repobench-p -1.2037`

解释：
- 单跳检索/部分生成有提升，
- 但多跳问答与部分摘要/代码任务回落更明显，拉低了整体均值。

### 3.4 按 metric 的平均 delta（anchored - baseline）

- `retrieval_en`: `+2.0000`（n=1）
- `qa_f1_zh`: `+0.8224`（n=1）
- `classification`: `+0.0000`（n=2）
- `code_sim`: `-0.2783`（n=2）
- `rouge_l_f1`: `-0.4149`（n=4）
- `rouge_l_zh`: `-0.3929`（n=2）
- `qa_f1`: `-0.7692`（n=7）
- `count`: `-1.3375`（n=1）

重点：`qa_f1`（多任务）平均负向，且任务数最多，是总分下行主因。

### 3.5 失败类型审计（pipeline 健康）

总体上 `empty_output_rate / parse_fail_rate / truncation_at_question_rate` 两方法差异很小；
明显差异仅见：
- `narrativeqa` 的 `template_leakage_rate` anchored 相比 baseline 增加约 `+0.09`。

解释：
- 没有证据表明是“输出空/解析失败/截断”导致总体负向。
- 更像是“能力取舍 + 训练配置”问题，而非评测脚本崩坏。

### 3.6 训练侧信号（同预算下 anchored 更难收敛）

四个 run 的 `summary.json` 显示：
- 训练预算一致：`max_steps=400, batch=2, grad_acc=1, seq=16384, lr=2e-5`
- 训练损失：anchored 在两个 seed 都高于 baseline（约 `+0.006~0.007`）
  - `baseline_42: 1.84597`
  - `anchored_42: 1.85287`
  - `baseline_1337: 1.84174`
  - `anchored_1337: 1.84829`

解释：
- 在当前 400 steps 下，anchored 可能处于“未充分补偿注入扰动”的欠收敛态。

### 3.7 baseline 稳定性与主链可信度

- `baseline seed42 vs seed1337` 平均分差仅 `+0.0347`，非常稳定。
- `baseline_gold_seed42`（未微调）平均 `38.5137`，而 `fast400 baseline` 到 `44.4355`，说明训练评测链条在工作，非整体异常。

## 4. 可能问题（按优先级）

### P0（最可能主因）

1. **数据配方不匹配主任务**
- 当前快跑训练默认 WikiText-only，和 LongBench 的指令/QA/摘要分布不一致。
- 表现为：检索类可涨，但多跳 QA 与摘要类净损失更大。

2. **固定 400 steps 对 anchored 不够**
- anchored 在两 seed 训练 loss 持续高于 baseline。
- 可能需要更长训练或更适配的 warmup/学习率策略，才能把“形状改动”转化为最终收益。

### P1（高概率副因）

3. **anchored 参数在 Qwen + 16k 组合下存在迁移误差**
- 当前锁定参数 `anchor=4, slope_raw=20, center=0.70` 并非为本组合专门搜索。
- 从任务形态看，可能强化了某段频域而损失多跳推理信息。

4. **收益结构偏窄（检索收益不足以覆盖 QA 损失）**
- `passage_retrieval_en` 明显增益，但 `qa_f1` 多任务平均负向，导致总分不升反降。

### P2（次要但需跟踪）

5. **模板泄漏在个别任务上抬升**
- `narrativeqa` 出现更高 `template_leakage_rate`。
- 当前不构成主因，但建议做样本抽检，避免在最终稿中被审稿人质疑 generation hygiene。

## 5. 在跑任务状态（截至 2026-02-26 07:41 CST）

`seed1337 anchored` 仍在跑：
- 进程：`eval_longbench.py` PID `97597`
- 最新日志：正在 `multifieldqa_zh`，进度 `100/200`，运行中
- 结果文件尚未落盘：
  - `artifacts/reviewer_2026-02-25/h2_qwen_fast400_seed1337/downstream_eval_fast400/longbench/anchored_sigmoid.json`

## 6. 下一步建议（执行顺序）

1. **先收齐 seed1337 anchored，再做双 seed 统计**
- 跑 `significance_test.py`（paired + FDR）后再决定 claim wording。

2. **立即做“WikiText vs 混合长指令”对照（同预算、同协议）**
- 使用 `scripts/prepare_long_instruction_mix.py` 生成混合训练集。
- 矩阵最小版：`baseline/anchored × seed42`，`lb6 + 高压切片` 先看方向。

3. **anchored 参数微调（小网格）**
- 在公平设置下探索：
  - `anchor_factor`: `3/4/5`
  - `slope_raw`: `16/20`
  - `center_ratio`: `0.60/0.70`
- 先 `lb6` 过滤，再上 `lb21`。

4. **保留当前负向证据并如实写入限制项**
- 若双 seed 后仍无净增益，不应继续正向 claim。
- 论文表述应降级为“任务相关 trade-off + 数据配方依赖”。

## 7. 对论文陈述的即时建议

在 `seed1337 anchored` 完成前，建议使用如下保守表述：
- “在 Qwen Fast400 + WikiText-only 设置下，anchored-sigmoid 未显示稳定总体提升；收益主要集中在部分检索任务，而多跳 QA 存在下降。”

避免使用：
- “显著提升”“一致提升”“全任务稳健提升”。
