# 1.5B / 2xH100 执行 Runbook

## 1. 开跑前 10 分钟检查

1. 环境与 GPU
```bash
bash /Users/misaya.yanghejazfs.com.au/dfrope/h100_advanced_experiments/scripts/bootstrap_h100_env.sh
```

2. 确认目录
```bash
cd /opt/dfrope
mkdir -p results/raw results/processed logs
```

3. 固定实验版本（建议）
- 记录 `git rev-parse HEAD`
- 记录 `pip freeze | rg "torch|transformers|datasets|accelerate|flash"`
- 保存到 `results/raw/env_snapshot.txt`

## 2. 标准训练顺序（推荐）

1. `geo_500k`（基线）
2. `hybrid_a0.2_t100k`（主对照）
3. `anchpoly_p3.9_omf0.3_t500k`（替代方案）

每个配置跑完后立刻执行：
- 保存 `results_<config>.json`
- 写入 `results/raw/merged_results.json`（增量更新）

## 3. 统一结果格式（建议）

```json
{
  "meta": {
    "model": "1.5B",
    "seed": 42,
    "train_tokens": 20000000000
  },
  "experiments": {
    "geo_500k": {
      "2048": {"ppl": 8.91, "std": 0.12, "n": 10},
      "16384": {"ppl": 23.74, "std": 0.91, "n": 10}
    }
  }
}
```

## 4. 画图与汇总

```bash
cd /Users/misaya.yanghejazfs.com.au/dfrope/h100_advanced_experiments
python scripts/plot_h100_results.py \
  --input-dir results/raw \
  --output-dir results/processed \
  --title "1.5B RoPE Frequency Comparison"
```

输出：
- `results/processed/summary_by_length.csv`
- `results/processed/summary_table.md`
- `results/processed/figures/ppl_vs_length.png`
- `results/processed/figures/ppl_16k_bar.png`

## 5. 判断规则（快速口径）

- 短程不过关：`PPL@2048` 比最佳差 > 10%  
- 长程有效：`PPL@16384` 优于 `geo_500k` 且 3-seed 至少 2 次胜出  
- 稳定有效：`PPL@32768` 仍优于 `geo_500k`，同时短程退化不超过 5%

## 6. 常见中断恢复

1. 断网但进程在跑：  
   - `tmux ls` -> `tmux attach -t <session>`
2. 进程挂掉：  
   - 从最近 `results_<config>.json` 恢复；
   - 下一个配置继续跑，不回滚已完成配置。
3. OOM：  
   - `micro_batch /= 2`，`grad_accum *= 2`，保持 effective batch 不变。
