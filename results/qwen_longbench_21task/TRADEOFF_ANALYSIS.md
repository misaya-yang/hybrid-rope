# Qwen Fast400 Tradeoff 详细分析（2026-02-26）

## 1. 数据范围

- 使用已回收本地产物（`artifacts/_server_sync/2026-02-26_tradeoff`）：
  - `seed42 baseline` 与 `seed42 anchored_sigmoid` 的完整 `lb21` 对照
  - `seed1337 baseline`（用于稳定性参考）
- `seed1337 anchored_sigmoid` 仍在服务器运行中，尚未落盘最终 `anchored_sigmoid.json`。

## 2. 全局结论（当前可确认）

- `seed42`：
  - baseline 平均：`44.4355`
  - anchored 平均：`44.0830`
  - 平均差值（anchored - baseline）：`-0.3525 pct`
  - 任务胜负：`8 升 / 11 降 / 2 平`
- baseline 跨 seed 稳定性：
  - `seed1337 baseline` 平均：`44.4701`
  - 相对 `seed42 baseline`：`+0.0347`

当前形态是典型 tradeoff：不是崩溃，但净收益不足。

## 3. 任务级变化（seed42，anchored - baseline）

### 3.1 主要上涨任务

- `passage_retrieval_en`：`+2.0000`
- `narrativeqa`：`+1.2341`
- `triviaqa`：`+0.8301`
- `multifieldqa_zh`：`+0.8224`
- `lcc`：`+0.6471`

### 3.2 主要下跌任务

- `2wikimqa`：`-3.0177`
- `musique`：`-2.8648`
- `hotpotqa`：`-1.9463`
- `passage_count`：`-1.3375`
- `repobench-p`：`-1.2037`

## 4. 长程任务拆分（按任务平均输入长度）

定义：`mean_len >= 12000` 视为“长程高压任务”。

### 4.1 长程上涨

- `passage_retrieval_en`（mean_len≈12944）：`+2.0000`
- `narrativeqa`（mean_len≈14763）：`+1.2341`

### 4.2 长程持平

- `lsht`（mean_len≈12831）：`+0.0000`

### 4.3 长程下跌

- `qmsum`（mean_len≈12506）：`-0.6111`
- `passage_count`（mean_len≈13887）：`-1.3375`
- `hotpotqa`（mean_len≈13296）：`-1.9463`
- `musique`（mean_len≈15908）：`-2.8648`

长程子集平均差值：`-0.5036`（7 个任务）。

## 5. 分桶分析（样本级，按输入长度）

按 `input_tokens_after_trunc` 分桶，统计样本级差值（anchored - baseline）：

- `<=4k`：`mean_delta=+0.1604`
- `4k-8k`：`mean_delta=-0.7373`
- `8k-12k`：`mean_delta=-1.1362`
- `12k-16k`：`mean_delta=+0.0723`

解释：收益并非“越长越稳增”，而是出现结构性波动；中长段（4k-12k）负向更明显。

## 6. 按 metric 聚合（任务级）

- 正向：
  - `retrieval_en`：`+2.0000`（n=1）
  - `qa_f1_zh`：`+0.8224`（n=1）
- 近零：
  - `classification`：`+0.0000`（n=2）
- 负向：
  - `qa_f1`：`-0.7692`（n=7）
  - `rouge_l_f1`：`-0.4149`（n=4）
  - `rouge_l_zh`：`-0.3929`（n=2）
  - `code_sim`：`-0.2783`（n=2）
  - `count`：`-1.3375`（n=1）

## 7. 审计健康（anchored - baseline，任务均值）

- `empty_output_rate`：`+0.000000`
- `parse_fail_rate`：`+0.000000`
- `truncation_at_question_rate`：`+0.000000`
- `template_leakage_rate`：`+0.003571`

说明本轮不是“脚本挂了/输出空了”导致，而更像是方法与数据配方的真实 tradeoff。

## 8. 下一步（等待 seed1337 anchored 完结后）

1. 补齐 `seed1337 anchored` 并做双 seed 合并结论。
2. 跑 paired 显著性（bootstrap + permutation/sign-flip + effect size + FDR）。
3. 如双 seed 仍净负，结论应维持 tradeoff 表述，不升级“稳定提升”。

