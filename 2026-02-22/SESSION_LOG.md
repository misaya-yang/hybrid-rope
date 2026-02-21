# Session Log: 2026-02-22 凌晨

> 时间：01:20 - 02:30  
> 目标：审查训练脚本 → 修 bug → 设计 overnight 实验 → 服务器部署 → 启动

---

## 1. 脚本审查与修复

对 `run_llama8b_fair_suite.py` 进行红队审查，发现并修复 **4 个致命 Bug + 2 个 Warning**：

| Bug | 严重性 | 修复 |
|-----|--------|------|
| `tail_base = max(base, 500000)` 在 LLaMA-3 下退化为 baseline | Fatal | 改为 `base × scale²`，floor 4x |
| BOS token 被截断 | Fatal | 保留 `[BOS] + tail` |
| inv_freq 注入可能被 forward 忽略 | Fatal | 添加运行时 logit diff 探针 |
| 验证探针 device mismatch（cuda vs cpu） | Fatal | 改用 `next(model.parameters()).device` |
| `gradient_checkpointing` 缺少 `use_reentrant=False` | Warning | 已加 |
| `evaluation_strategy` 已弃用 | Warning | 改为 `eval_strategy` |

## 2. Transformers 5.1.0 兼容性问题 ⚠️

**服务器版本**：torch=2.8.0+cu128, transformers=**5.1.0**, peft=0.18.1

**发现**：transformers 5.1.0 改变了 LLaMA config 结构：

```python
# transformers 4.x
config.rope_theta = 500000.0
config.rope_scaling = None

# transformers 5.1.0
config.rope_theta → AttributeError!
config.rope_scaling = {'rope_theta': 500000.0, 'rope_type': 'default', ...}
```

**影响**：
- `getattr(config, 'rope_theta', None)` 返回 `None` → fallback 到 10000（错误！）
- `rope_scaling is not None` → 被误判为"非默认"

**修复（两个函数）**：
- `infer_model_rope_base()`：增加从 `rope_scaling` 字典提取 `rope_theta` 的逻辑
- `rope_scaling_is_effectively_default()`：只看 `rope_type` 是否为 `default`，不再检查 key 白名单

## 3. 服务器验证结果

| 测试项 | 结果 |
|--------|------|
| `eval_strategy` 存在 | ✅ |
| `evaluation_strategy` 已移除 | ✅ |
| `gradient_checkpointing_kwargs` | ✅ |
| `LlamaRotaryEmbedding.inv_freq` 是 buffer | ✅ shape=(64,) float32 |
| inv_freq.copy_() 注入有效 | ✅ logit_diff=1.47 |
| rope_base 正确推断为 500000 | ✅（修复后） |

## 4. Overnight 实验

- **PID**: 29315
- **启动时间**: 02:22
- **预计完成**: ~09:30
- **GPU**: RTX PRO 6000 Blackwell, 87.5/97.9 GB, 100%, 59°C

**流水线**:
1. Gate 0: 4 方法 calibration（4/4 通过 ✅）
2. Gate 1: 4×600 步训练（baseline/pi/yarn/anchored_hybrid）
3. Gate 2: NIAH 评测（base+4 adapters × [4K,8K,16K,32K]）
4. Gate 3: 自动生成对比表 + 热力图

**结果位置**: `results/overnight_8h/summary/`

## 5. 查看进度

```bash
# SSH 到服务器
plink -P 42581 root@connect.bjb1.seetacloud.com -pw htG0sD63/yG0

# 查看日志
tail -50 /root/autodl-tmp/dfrope/hybrid-rope/results/overnight_8h/experiment.log

# GPU 状态
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv

# 检查进程
ps aux | grep run_overnight
```
