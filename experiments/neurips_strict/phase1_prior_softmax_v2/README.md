# Phase 1 Prior-Softmax Experiment (v2 - Transformers 5.x Compatible)

## 修复说明

此版本修复了 transformers 5.x 的兼容性问题。

### 问题
- **transformers 4.x**: 使用 `_attn` 方法进行 monkey-patching
- **transformers 5.x**: `GPT2Attention` 对象不再有 `_attn` 方法，改为使用模块级别的 `eager_attention_forward` 函数

### 解决方案

将 monkey-patching 从实例方法改为模块级别函数：

```python
# 旧代码 (transformers 4.x) - 会导致 AttributeError
attn_module._attn = make_new_forward(...)

# 新代码 (transformers 5.x) - 正确的方式
from transformers.models.gpt2 import modeling_gpt2 as gpt2_module
gpt2_module.eager_attention_forward = patched_eager_attention_forward
```

## 关键变更

1. **导入 GPT2 模块**：
   ```python
   from transformers.models.gpt2 import modeling_gpt2 as gpt2_module
   ```

2. **全局存储原始函数**：
   ```python
   _original_eager_attention_forward = None
   ```

3. **Patch 函数重写**：
   - 存储原始 `eager_attention_forward`
   - 替换为支持 distance prior 的版本
   - 添加 `unpatch_model_for_prior()` 用于清理

4. **Attention Monitor 调整**：
   - 通过 `model._attention_monitor` 传递 monitor 实例
   - 在 patched 函数中通过 `hasattr` 检查并记录统计信息

## 使用方法

```bash
cd /Users/misaya.yanghejazfs.com.au/neurIPS-2026/hybrid-rope/experiments/neurips_strict/phase1_prior_softmax_v2
python experiment.py
```

## 输出

- `results/phase1_history_*.json` - 详细训练历史
- `results/phase1_summary_*.csv` - 结果汇总表
- `figures/` - 图表输出目录

## 实验组

| 组别 | 描述 | λ值 |
|------|------|-----|
| A | Baseline (Vanilla Softmax) | N/A |
| B | Prior-Softmax | 0.01, 0.05, 0.1 |
| C | Prior-Softmax + LoRA | 0.01, 0.05, 0.1 |

## 停止条件

1. PPL > baseline × 5.0
2. 2000 步无改善
3. 熵崩溃 (< 0.1)
