# 方法说明（中文）

本文档解释：我们到底“改了什么”、以及为什么这仍在 RoPE 的“合法空间”内（不破坏 Translation Invariance, TI）。

## 1. 核心变量：频率集合 `ω_k`

在每个 head 的 RoPE 里（本项目常用 `head_dim=64`），有：

- `K = head_dim / 2 = 32` 个频率通道

相对距离 `d` 的相位差形式仍然是：

- `Δφ_k(d) = ω_k * d`

因此 TI 仍成立：编码只依赖距离 `d`，不依赖绝对位置。

我们做的唯一改变是：重新设计 `ω_k`（或等价的 `inv_freq[k]`）的分布。

## 2. 频率函数（必须逐字一致）

下面三段函数是我们全套实验的“不可变核”，在 50M/350M 实验中保持一致（任何改动都属于 definition drift，会让结果不可对比）。

```python
def geometric_freq(K, theta):
    k = torch.arange(K, dtype=torch.float32)
    return 1.0 / (theta ** (2 * k / (2 * K)))

def anchored_poly_freq(K, theta_base, p=3.9, omf=0.3):
    k = torch.arange(K, dtype=torch.float32)
    geo = geometric_freq(K, theta_base)
    omega_max = geo[0].item()
    omega_min = geo[-1].item() * omf
    t = k / (K - 1)
    log_omega = math.log(omega_max) + (t ** p) * (math.log(omega_min) - math.log(omega_max))
    return torch.exp(log_omega)

def hybrid_freq(freq_a, freq_b, alpha):
    return (1 - alpha) * freq_a + alpha * freq_b
```

解释：

- `geometric_freq`：标准等比频谱（对应 RoPE 传统 `theta` 参数）
- `anchored_poly_freq`：固定端点（`omega_max/omega_min`）后，在 log 空间用 `t^p` 做非线性插值，改变“中频”分配
- `hybrid_freq`：两组频谱的凸组合（不改变 TI），用于验证“较小 theta + 频谱重分配”是否可替代“极大 theta”

## 3. 训练与评测定义（避免 definition drift）

### 3.1 数据

- TinyStories streaming + `gpt-neox-20b` tokenizer
- `encode(add_special_tokens=False)`
- 训练：取前 N 个 tokens（N=50M 或 500M）
- 验证：取前 5M tokens

### 3.2 评测 slicing（非常关键）

对每个评测长度 `L`，取连续的 `EVAL_CHUNKS` 个 chunk：

- chunk 0：`val[0:L]`
- chunk 1：`val[L:2L]`
- ...

好处：

- 给定同一 `val` token 流时完全确定性
- 易于复现与对比

### 3.3 报告指标

- `PPL@2048`：训练长度内质量（必须不崩）
- `PPL@16384`：外推能力主指标
- 多 seed：以 `mean ± std` 为最终结论依据

## 4. 为什么不上传权重 / memmap cache

- 权重：体积大，且本工作目标是对比结论可复现，不是 release 模型。
- memmap cache：500M tokens 级别可能上百 GB，不适合 GitHub；cache 可由脚本用 streaming 重建。

