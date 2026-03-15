# AI Handoff: DiT τ Sweep 实验

## 当前状态

**服务器**: `ssh -p 29382 root@connect.westd.seetacloud.com`
**GPU**: RTX 5090 32GB
**进程**: `nohup` 后台运行，日志在 `/tmp/dit_tau.log`
**工作目录**: `results/video_dit/20260316_tau_sweep`

### 正在运行的实验

129.6M模型（medium profile）上的τ sweep，3个值顺序执行：

| τ | 状态 | 预计耗时 |
|---|------|----------|
| 0.30 | 训练中 (step ~1200/15000, ETA ~51min) | ~56min训练 + ~10min评估 |
| 0.70 | 等待中 | ~56min训练 + ~10min评估 |
| 1.50 | 等待中 | ~56min训练 + ~10min评估 |

**预计总完成时间**: ~3小时（从启动算起，约2026-03-16 09:00左右）

### 启动命令

```bash
cd /root/hybrid-rope
nohup env PYTHONUNBUFFERED=1 python scripts/video_temporal/run_dit_temporal.py \
  --tau 0.3,0.7,1.5 --profile medium --seed 42 \
  --work_dir results/video_dit/20260316_tau_sweep > /tmp/dit_tau.log 2>&1 &
```

### 监控

```bash
tail -20 /tmp/dit_tau.log          # 查看进度
nvidia-smi                          # 确认GPU在用
ps aux | grep run_dit               # 确认进程存活
```

## 为什么做这个实验

τ* = K_t/√T_train 是为AR模型推导的。DiT是双向attention，频率的作用不同：
- AR: 频率决定"因果链能传多远" → 需要更多低频 → 大τ有利
- DiT: 频率决定"位置编码区分度" → 需要均匀覆盖 → 小τ可能更好

之前的结果：
- 38.8M模型: τ=2.83 (EVQ) 赢 τ=0 (GEO) 27-40%
- 129.6M模型: τ=0 (GEO) 赢 τ=2.83 (EVQ) 71%

**假说**: τ=2.83对DiT太大了（中频空洞导致位置ambiguity），最优τ在[0, 1.0]之间。

**预测**: 如果τ=0.3或τ=0.7在129.6M上beat τ=0的far-extrap，说明EVQ对DiT有用，只是需要不同的τ。

## 已有结果位置

| 实验 | 本地路径 | 服务器路径 |
|------|----------|------------|
| 38.8M (τ=0, τ=2.83) | results/video_dit/20260316_002758/ | 同 |
| 129.6M (τ=0, τ=2.83) | results/video_dit/20260316_medium/ | 同 |
| 129.6M τ sweep | 待下载 | results/video_dit/20260316_tau_sweep/ |
| 综合报告 | results/video_dit/REPORT_FINAL.md | 同 |

## 实验完成后要做的事

1. 下载τ sweep结果到本地:
   ```bash
   scp -P 29382 root@connect.westd.seetacloud.com:/root/hybrid-rope/results/video_dit/20260316_tau_sweep/*_results.json \
     /Users/yang/projects/hybrid-rope/results/video_dit/20260316_tau_sweep/
   ```

2. 对比5个τ值的关键指标（已有2个 + sweep的3个）:
   - τ=0.00: results/video_dit/20260316_medium/geo_seed42_results.json
   - τ=0.30: results/video_dit/20260316_tau_sweep/tau0.30_seed42_results.json
   - τ=0.70: results/video_dit/20260316_tau_sweep/tau0.70_seed42_results.json
   - τ=1.50: results/video_dit/20260316_tau_sweep/tau1.50_seed42_results.json
   - τ=2.83: results/video_dit/20260316_medium/evq_seed42_results.json

3. 判断结果:
   - 如果τ=0始终最优 → DiT不需要EVQ，AR结果足够支撑论文
   - 如果某个小τ>0 beat τ=0 → EVQ对DiT有用，但需要DiT-specific的τ公式
   - 写入REPORT_FINAL.md

4. 关机: `shutdown -h now`（仅在所有实验完成后）

## 脚本改动

`scripts/video_temporal/run_dit_temporal.py` 新增:
- `--tau` 参数: 逗号分隔的τ值，覆盖--method
- `run_one_method()` 增加 `tau_override` 参数
- `print_comparison()` 支持任意数量method对比（不再限制geo/evq两列）
- 对比表增加 near/mid/far 分段显示
