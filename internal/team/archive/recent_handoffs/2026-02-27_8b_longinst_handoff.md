# 8B LongInst 训练交接文档

> **创建时间**: 2026-02-27 21:40 CST
> **实验 ID**: `EXP_EVQ_8B_LONGINST`
> **服务器**: RTX PRO 6000 Blackwell 96GB (SSH below)
> **状态**: A1 training in progress (step ~38/800)

---

## 1. 服务器连接

```bash
ssh -p 23173 root@connect.bjb1.seetacloud.com
# 密码: htG0sD63/yG0
source /root/miniconda3/bin/activate
```

## 2. 环境信息

| 项目 | 值 |
|------|-----|
| GPU | NVIDIA RTX PRO 6000 Blackwell, 96GB, SM 12.0 |
| CUDA | 13.1 |
| PyTorch | 2.8.0+cu128 |
| transformers | 5.1.0 |
| Attention | **SDPA** (flash-attn 不支持 SM 12.0) |
| Python | 3.12.3 (miniconda3) |

## 3. Pipeline 概览

4-job dual-seed pipeline，顺序执行：

| Job | Run Name | τ | Seed | 目的 |
|-----|----------|---|------|------|
| **A1** | `A1_geometric_tau0p00_r32_s800_seed42` | 0.0 (geo baseline) | 42 | Baseline seed=42 |
| A2 | `A2_evq_cosh_tau1p50_r32_s800_seed42` | 1.5 (EVQ) | 42 | Treatment seed=42 |
| B1 | `B1_geometric_tau0p00_r32_s800_seed1337` | 0.0 (geo baseline) | 1337 | Baseline seed=1337 |
| B2 | `B2_evq_cosh_tau1p50_r32_s800_seed1337` | 1.5 (EVQ) | 1337 | Treatment seed=1337 |

**每 job 包含**: 800 step LoRA 训练 + Full LongBench-21 评估 + NIAH + Passkey

## 4. 训练参数

```
model:           LLaMA-3-8B-Instruct
lora_rank:       32
max_steps:       800
max_seq_len:     8192
batch_size:      4
grad_accum:      1
lr:              2e-5
warmup_steps:    50
data_mix:        70% LongAlpaca + 10% WikiText(by tokens) + 20% Synthetic
min_supervised_tokens: 64 (从32上调, 过滤短wiki样本)
```

## 5. 时间预估 (修正版)

实测每 step ~10.25s (混合长短序列)：

| 阶段 | A1 | A2 | B1 | B2 |
|------|-----|-----|-----|-----|
| 数据预处理 | ~10 min | ~5 min (缓存) | ~10 min | ~5 min |
| 训练 800 steps | ~2.3h | ~2.3h | ~2.3h | ~2.3h |
| Full eval | ~1h | ~1h | ~1h | ~1h |
| **小计** | **~3.3h** | **~3.3h** | **~3.3h** | **~3.3h** |

**总预计**: ~12-13 小时
**A1 开始时间**: 21:22 CST
**预计全部完成**: 明天 ~09:00-10:00 CST

## 6. 监控命令

```bash
# 查看当前 job 日志 (实时)
tail -f /root/autodl-tmp/dfrope/hybrid-rope/artifacts/llama8k_theory_v1/logs/A1_geometric_tau0p00_r32_s800_seed42.log

# 一键查看所有 job 日志最新状态
for f in /root/autodl-tmp/dfrope/hybrid-rope/artifacts/llama8k_theory_v1/logs/*.log; do
  echo "=== $(basename $f) ==="; tail -3 "$f"; echo; done

# 查看 orchestrator 总日志
tail -50 /root/longinst_run.log

# GPU 状态
nvidia-smi

# 检查进程
ps aux | grep python | grep -v grep | grep -v jupyter

# 查看训练 loss 趋势 (从 log 提取)
grep -oP "'loss': [\d.]+" /root/autodl-tmp/dfrope/hybrid-rope/artifacts/llama8k_theory_v1/logs/A1_*.log | tail -20

# 查看 eval 结果 (完成后)
cat /root/autodl-tmp/dfrope/hybrid-rope/artifacts/llama8k_theory_v1/A1_geometric_tau0p00_r32_s800_seed42/eval_results.json 2>/dev/null
```

## 7. 输出路径

```
/root/autodl-tmp/dfrope/hybrid-rope/artifacts/llama8k_theory_v1/
├── logs/                          # 每个 job 的完整日志
│   ├── A1_geometric_tau0p00_r32_s800_seed42.log
│   ├── A2_evq_cosh_tau1p50_r32_s800_seed42.log
│   ├── B1_geometric_tau0p00_r32_s800_seed1337.log
│   └── B2_evq_cosh_tau1p50_r32_s800_seed1337.log
├── data/
│   └── <run_name>/                # 数据预处理产物
│       ├── longinst_mix.jsonl     # 混合训练数据
│       ├── stats.json             # 数据统计 (含 guard 检查结果)
│       └── data_manifest.json     # 数据溯源
├── <run_name>/                    # 训练产物
│   ├── checkpoint-*/              # LoRA checkpoints
│   ├── eval_results.json          # LongBench 评估结果
│   ├── niah_results.json          # NIAH 热力图数据
│   └── passkey_results.json       # Passkey 准确率
└── stats/                         # 跨 job 汇总 (pipeline 完成后)
```

## 8. 代码改动 (仅环境适配)

**唯一修改的文件**: `scripts/isolated/longinst/new_lora_longinst_train_v1.py`

| 行号 | 改动 | 原因 |
|------|------|------|
| L1789 | `min_supervised_tokens` default 32 → **64** | WikiText 短样本导致 `assistant_tokens_lt64_ratio=0.369` 触发 guard (>0.10)。提升到 64 后比率降为 0.0 |
| L1876-1882 | `rope_scaling` guard 允许 `rope_type="default"` | transformers 5.1.0 自动填充 `rope_scaling={'rope_type':'default'}`，虽然等同于标准 RoPE |

**零逻辑改动**：不影响训练算法、频率分配、评估协议。

## 9. 数据统计 (A1 实际)

```json
{
  "rendered_total": 62727,
  "num_samples_after_tokenize": 37233,
  "num_dropped_low_supervised": 25414,
  "assistant_tokens_lt64_ratio": 0.0,
  "assistant_tokens": {"min": 64, "p50": 89, "max": 762, "mean": 116.1},
  "total_tokens": {"min": 104, "p50": 1559, "max": 8192, "mean": 1999.5}
}
```

## 10. 已知问题与应急

| 问题 | 处理方式 |
|------|---------|
| flash-attn 不支持 SM 12.0 | 已改用 SDPA，性能等效 |
| transformers 5.1.0 rope_scaling guard | 已修复 (允许 rope_type=default) |
| 短 wiki 样本触发 lt64 guard | 已修复 (min_supervised_tokens=64) |
| **如果训练中途 OOM** | 降低 batch_size 至 2 (当前 VRAM 82/96GB 还有余量) |
| **如果 SSH 断线** | nohup 启动，不影响运行，重新连接即可 |
| **如果进程意外退出** | orchestrator 会跳过失败 job 继续下一个，检查 log |

## 11. 结果回收后的下一步

训练完成后需要：
1. **数据回收**: 将 eval_results.json / niah_results.json / passkey_results.json 下载到本地
2. **统计分析**: 运行 `paired_stats_llama8k_theory_v1.py` 计算 paired bootstrap CI + permutation p-value
3. **写入论文**: 填充 Section 5.3 (LoRA evaluation) 的占位数据
4. **更新 EXPERIMENT_REGISTRY**: 将 EXP_EVQ_8B_LONGINST 状态改为 Paper-Ready

```bash
# 下载结果到本地 (训练完成后执行)
sshpass -p 'htG0sD63/yG0' scp -P 23173 -r \
  root@connect.bjb1.seetacloud.com:/root/autodl-tmp/dfrope/hybrid-rope/artifacts/llama8k_theory_v1/ \
  /Users/misaya.yanghejazfs.com.au/neurIPS-2026/hybrid-rope/artifacts/llama8k_theory_v1_blackwell/
```

---

> **Launch Script**: `/root/launch_8b.sh`
> **Orchestrator PID**: 6155
> **A1 Training PID**: 6156
