# Session Log: 2026-02-22

> 时间：01:20 - 09:15
> 目标：审查训练脚本 → 修 bug → 服务器部署 → 启动 overnight 实验

---

## 1. 脚本审查与修复（4 Fatal + 2 Warning）

| Bug | 文件 | 修复 |
|-----|------|------|
| `tail_base` 退化为 baseline | `run_llama8b_fair_suite.py` L157 | `base × scale²` + floor 4x |
| BOS token 截断 | L367 | 保留 `[BOS] + tail` |
| inv_freq 注入可能失效 | L656 | 添加 logit diff 探针 |
| 探针 device mismatch | L656 | `next(model.parameters()).device` |
| gradient_checkpointing 缺 use_reentrant | TrainingArguments | 已加 `use_reentrant=False` |
| evaluation_strategy 弃用 | TrainingArguments | 改为 `eval_strategy` |

---

## 2. Transformers 5.1.0 兼容性问题 ⚠️ 关键

> [!CAUTION]
> **transformers 5.x 改变了 LLaMA config 结构，不修复会导致 rope_base 错误！**

```python
# transformers 4.x
config.rope_theta = 500000.0           # 顶层属性
config.rope_scaling = None

# transformers 5.1.0
config.rope_theta → AttributeError!    # 不存在了!
config.rope_scaling = {
    'rope_theta': 500000.0,
    'rope_type': 'default',
    'original_max_position_embeddings': ...,
    'attention_factor': ...,
}
```

**影响**：
1. `getattr(config, 'rope_theta', None)` 返回 `None` → fallback 到 10000（而非 500000）
2. `rope_scaling is not None` → 被误判为"非默认缩放"

**修复**（`run_llama8b_fair_suite.py`）：
- `infer_model_rope_base()`：增加从 `rope_scaling` 字典提取
- `rope_scaling_is_effectively_default()`：只看 `rope_type`，不检查 key 白名单

---

## 3. 服务器环境

```
torch=2.8.0+cu128
transformers=5.1.0
peft=0.18.1
datasets=3.0.0
GPU: NVIDIA RTX PRO 6000 Blackwell (97,887 MiB)
```

---

## 4. 踩坑记录（按时间顺序）

### 坑 1：Baseline calibration 失败（02:14）

**现象**：4/4 中只有 3/4 通过（baseline 失败）
**原因**：第一次启动时还没修 transformers 5.1.0 兼容性，`rope_scaling_is_effectively_default` 因为白名单检查失败
**修复**：上面第 2 点

### 坑 2：CUDA OOM — 僵尸进程（02:26）

**现象**：所有 4 个方法 OOM，`Process 28631 has 85.46 GiB`
**原因**：kill 第一次 overnight 进程（PID 28311）时只杀了父进程，子训练进程（28631）还活着
**修复**：kill 时必须同时杀子进程

> [!IMPORTANT]
> **Kill 实验时必须用 `pkill -f run_overnight; pkill -f run_llama`，不要只 kill 父进程 PID！**

### 坑 3：Timeout 太短（08:30 发现）

**现象**：第二次实验看起来超时了（实际是坑 2 导致的，但原始 timeout=7200s 确实不够）
**原因**：16K ctx 下每步 ~48 秒，600 步 = ~8 小时 > 7200 秒（2 小时）timeout
**修复**：`TRAIN_STEPS=300`，`timeout=36000`

### 坑 4：CUDA OOM — Python Popen 占 GPU（09:05）

**现象**：用 Python `subprocess.Popen` 启动 overnight 脚本，所有训练 OOM
**原因**：Python 进程 `import torch` 分配 CUDA context (~81 GB)，子进程无法分配
**修复**：必须用纯 shell `nohup` 启动，不能用 Python 脚本包装

> [!CAUTION]
> **永远不要用 Python Popen 启动 overnight 脚本！用 `bash _launch.sh` 或直接 `nohup python -u ... &`**

---

## 5. 正确的启动方式

```bash
# 1. 确保 GPU 清空
pkill -f run_overnight 2>/dev/null
pkill -f run_llama 2>/dev/null
sleep 3
nvidia-smi  # 确认 0 MiB

# 2. 用纯 shell 启动（不要用 Python 包装！）
cd /root/autodl-tmp/dfrope/hybrid-rope
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nohup /root/miniconda3/bin/python -u 2026-02-22/scripts/run_overnight_8h.py \
    > results/overnight_8h/console.log 2>&1 &
echo "PID=$!"

# 3. 确认训练开始
sleep 120
nvidia-smi --query-gpu=memory.used --format=csv,noheader  # 应该 >80 GB
tail -5 results/overnight_8h/experiment.log
```

或者直接用启动脚本：
```bash
bash /root/autodl-tmp/dfrope/hybrid-rope/2026-02-22/scripts/_launch.sh
```

## 6. 监控命令

```bash
# 实验日志
tail -f results/overnight_8h/experiment.log

# GPU 状态
nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv

# 训练 loss
cat results/overnight_8h/train_baseline/logs/train.log

# 进程状态
ps aux | grep run_llama
```

## 7. 当前实验

- **PID**: 44430
- **启动时间**: 09:10
- **配置**: 300 steps × 4 方法 × 16K ctx
- **预计完成**：~19:00
- **结果位置**: `results/overnight_8h/summary/`
