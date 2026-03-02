# Phase 9: Base=500K 大模型 Scale-Up 验证

## 服务器信息
- SSH: ssh -p 23173 root@connect.bjb1.seetacloud.com
- 密码: htG0sD63/yG0
- 环境: conda base
- GPU: RTX PRO 6000

## 目标

在 base=500K（已验证的主战场）上 scale 到更大模型，验证两个核心假说：
1. **Hybrid 优势随模型规模增大**（方差优势 + 可能出现 retrieval/PPL 胜出）
2. **Pure EVQ 在大模型上开始赢 Geometric**（可学习性瓶颈随 scale 消失）

## 背景：350M 已有结论（Phase 8F, base=500K, 4 seeds）

| Method | retrieval (mean±std) | PPL@16K (mean±std) |
|--------|---------------------|---------------------|
| Geometric | 0.735±0.055 | 175.7±13.6 |
| EVQ τ=1.0 | 0.706±0.014 | 193.9±17.1 |
| Hybrid τ=1.0 r=16 | 0.709±0.007 | 177.0±7.4 |

关键发现：Hybrid 方差是 Geo 的 1/8，PPL 持平，但 retrieval/PPL 均值没有显著赢。

## 实验设计

### 模型配置

```python
# 1B 模型（实际 ~1.7B with SwiGLU，比 350M 大 ~3.8x）
MODEL_CONFIG_1B = dict(
    hidden_size=2048,      # 1024 → 2048
    num_layers=24,         # 不变
    num_heads=32,          # 16 → 32（保持 head_dim=64）
    head_dim=64,           # 不变！这是关键——τ* 不变
    intermediate_size=8192,# 4096 → 8192
    vocab_size=50304,
)
# 参数量：~1713M (1.7B)

ROPE_BASE = 500_000       # 主战场 base
BATCH_SIZE = 8             # RTX PRO 6000 96GB with SDPA
LR = 3e-4                 # 比 350M 低（scaling law）
TRAIN_TOKENS = 500_000_000 # 500M tokens
DATASET = "HuggingFaceFW/fineweb-edu"
SEED = 42
```

**⚠️ head_dim=64 不变**，所以 τ*=d_head/√L=64/√4096=1.0 不变，r*=16 不变。频率配置和 350M 完全相同。

### Token 量选择

```python
# 方案 A：和 350M 同 tokens（50M），看纯模型规模效应
TRAIN_TOKENS_A = 50_000_000

# 方案 B：Chinchilla 比例（tokens ≈ 20 × params），更充分训练
TRAIN_TOKENS_B = 120_000_000  # 12B tokens / 100 ≈ 120M（简化版）

# 建议：先跑方案 A（~40min），确认方向后再考虑方案 B
```

### Phase 9A：六组对比（500M tokens, 1B model）

**Group 1: L_train=4096, τ*=64/√4096=1.0**

| Run ID | Method | τ | r | L_train | 目录 |
|--------|--------|---|---|---------|------|
| 9A-1 | Geometric | — | — | 4096 | `phase9/1b_geo_4k/` |
| 9A-2 | EVQ τ=1.0 | 1.0 | — | 4096 | `phase9/1b_evq1.0_4k/` |
| 9A-3 | Hybrid τ=1.0 r=16 | 1.0 | 16 | 4096 | `phase9/1b_hybrid1.0_r16_4k/` |

**Group 2: L_train=2048, τ*=64/√2048≈1.41→用τ=1.5**

| Run ID | Method | τ | r | L_train | 目录 |
|--------|--------|---|---|---------|------|
| 9A-4 | Geometric | — | — | 2048 | `phase9/1b_geo_2k/` |
| 9A-5 | EVQ τ=1.5 | 1.5 | — | 2048 | `phase9/1b_evq1.5_2k/` |
| 9A-6 | Hybrid τ=1.5 r=16 | 1.5 | 16 | 2048 | `phase9/1b_hybrid1.5_r16_2k/` |

**⚠️ 重要**：Group 2 的 baseline 是 9A-4（Geo L=2048），不能和 Group 1 直接比绝对值。

**预计时间**：每组 ~1.5h，总计 ~9h（顺序跑）。

**历史数据**：L=2048 + τ=1.5 在 50M/125M/350M 三种模型规模上全部赢 Geo 10-19%（retrieval），scaling law τ*=d/√L 预测吻合。

### Phase 9B：扩展（基于 9A 结果）

**如果 9A-Group1 显示 Hybrid/EVQ 趋势改善：**

| Run ID | Method | 说明 |
|--------|--------|------|
| 9B-1 | Geo (seed=137) | 多种子验证 |
| 9B-2 | Hybrid (seed=137) | 多种子验证 |
| 9B-3 | EVQ τ=0.8 | 测试 τ 敏感性 |
| 9B-4 | EVQ τ=1.2 | 测试 τ 敏感性 |

**如果 9A-Group2 确认 L=2048+τ=1.5 在 1B 模型仍赢 Geo：**
- 多种子验证（seeds=[42,137,256,999]）确认统计显著性
- 这将是论文的 Main Result（scale-up confirmation）

**如果 9A 显示和 350M 一样（Hybrid ≈ Geo）：**
- 考虑增加 tokens 到 500M-1B
- 或跑更大模型（2B+）

## EVQ 频率生成

和 350M 完全相同（因为 head_dim=64 不变）：

```python
import torch, math

def evq_cosh_inv_freq(head_dim=64, tau=1.0, base=500000.0):
    K = head_dim // 2
    u = torch.linspace(0.5/K, 1 - 0.5/K, K, dtype=torch.float64)
    phi = 1 - (1/tau) * torch.arcsinh((1 - u) * math.sinh(tau))
    return (base ** (-phi)).float()

def hybrid_evq_inv_freq(head_dim=64, tau=1.0, base=500000.0, r=16):
    K = head_dim // 2
    geo = torch.tensor([1.0/(base**(2*i/head_dim)) for i in range(K)], dtype=torch.float64)
    n_evq = K - r
    theta_max = geo[r].item()
    theta_min = geo[-1].item()
    u = torch.arange(n_evq, dtype=torch.float64) / max(n_evq - 1, 1)
    phi = 1 - (1/tau) * torch.arcsinh((1 - u) * math.sinh(tau))
    evq_part = (theta_min ** phi) * (theta_max ** (1 - phi))
    return torch.cat([geo[:r], evq_part]).float()

inv_freq_geo = torch.tensor([1.0/(500000.0**(2*i/64)) for i in range(32)], dtype=torch.float32)
inv_freq_evq = evq_cosh_inv_freq(64, 1.0, 500000.0)
inv_freq_hybrid = hybrid_evq_inv_freq(64, 1.0, 500000.0, r=16)
```

## 评估

### 基础评估

```python
EVAL_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384]
PK_LENGTHS = [1024, 2048, 4096, 8192]
PK_TRIALS = 100
```

### RULER Benchmark（Multi-Needle Retrieval）

**目的**：单 passkey 只测"找一根针"能力，不足以区分频率分配方法的长程记忆质量。Multi-needle 在上下文中插入 N 个独立 passkey，模型必须**全部**找回，更全面地测试频率分布的信息保持能力。

**实现**：`eval_multi_needle.py`（简化版 RULER multi-needle retrieval）

```python
# 在上下文中插入 5 个独立 passkey，均匀分布在不同深度
# 使用 indexed 格式: <<KEY1:7-4-2-9-1>>, <<KEY2:3-8-5-0-6>>, ...
# 每个 needle 独立通过 NLL-gap 评分

from eval_multi_needle import eval_multi_needle_passkey

MN_LENGTHS = [2048, 4096, 8192, 16384]
MN_NEEDLES = 5     # 每次插入 5 个 passkey
MN_TRIALS  = 20    # 每个长度 20 次试验

results = eval_multi_needle_passkey(
    model, tokenizer, filler_tokens,
    lengths=MN_LENGTHS,
    n_needles=MN_NEEDLES,
    num_trials=MN_TRIALS,
    seed=42,
)
```

**核心指标**：
| 指标 | 含义 |
|------|------|
| `per_needle_retrieval` | 单个 needle 检索成功率 |
| `all_needle_retrieval` | 所有 needle 同时检索成功率（更严格） |
| `by_needle_position` | 按 needle 位置（深度）分解的检索率 |
| `mean_nll_gap` | 平均 NLL gap（正=检索成功） |

**预期**：
- Hybrid 应在 `all_needle_retrieval` 上优于 Geometric（方差优势 → 每个频段都稳定）
- EVQ 的均匀间距在多 needle 场景下可能比 geometric 的指数间距更有优势
- 8K/16K 长度差异最明显（超出训练长度越远，频率分配质量越关键）

**Variable Tracking**（完整 RULER 子任务，如需要后续补充）：
- 在上下文中定义多个变量赋值 `X = 42, Y = 17, ...`
- 模型在末尾回答 `X = ?`
- 测试长距离变量绑定能力
- 实现复杂度较高，Phase 9A 先用 multi-needle passkey 替代

## 判定标准

### 核心指标

| 指标 | 350M 基线差距 | 600M 期望 | spotlight 所需 |
|------|-------------|-----------|---------------|
| Hybrid vs Geo retrieval | -2.6pp (ns) | **缩小或反转** | Hybrid > Geo (p<0.05) |
| Hybrid vs Geo PPL@16K | +1.3 (ns) | **持平或反转** | Hybrid ≤ Geo |
| Hybrid std / Geo std | 1/8 | **维持或更好** | 维持 |
| EVQ vs Geo retrieval | -2.9pp (ns) | **缩小** | EVQ > Geo |

### 决策树

```
Phase 9A 完成后：
│
├─ Hybrid 赢 Geo（retrieval 或 PPL）？
│   ├─ YES → 进 9B 多种子验证 → 论文 Main Result
│   └─ NO but std 更低 → 跑 120M tokens 看趋势
│
├─ EVQ 赢 Geo？
│   ├─ YES → 可学习性假说验证！论文 Main Result
│   └─ NO → 可学习性瓶颈在 600M 仍未消除
│
└─ 全面和 350M 一样？
    → 需要更大 scale（1B）或更多 tokens
```

## 显存估算

```
600M 模型 (fp32):
- 参数: ~600M × 4B = 2.4 GB
- 梯度: 2.4 GB
- 优化器 (AdamW): 4.8 GB
- 激活 (seq=4096, batch=2): ~8-12 GB
- 总计: ~18-22 GB

RTX PRO 6000 (48GB): 充裕
如果用 bf16 混合精度: ~12-15 GB

可能需要调整：
- 如果 OOM: batch_size=1 + gradient_accumulation=2
- 或开启 gradient_checkpointing
```

## 输出

每个 run 保存到 `result.json`，包含：
- ppl（各长度）
- passkey_global（retrieval_rate, mean_nll_gap）
- passkey_by_length
- train_time_sec

**完成后请发回三个 run 的核心指标对比表。**

## ⚠️ 注意事项

1. **head_dim=64 必须不变**：这是频率维度，改了的话 τ* 和 inv_freq 都要重算
2. **num_heads 要和 hidden_size 匹配**：hidden=1536, heads=24 → head_dim=64 ✅
3. **LR 需要调低**：600M 用 4e-4（350M 用 6e-4）。如果训练不稳，降到 3e-4
4. **数据缓存**：FineWeb 应该已在本地。50M tokens ≈ 12207 steps (batch=2, seq=4096)
5. **如果 run_evq_sweep.py 的模型配置是硬编码的**：需要修改或新建配置条目
6. **Geo baseline 必须重跑**：600M 的 Geo 和 350M 不一样，不能复用
7. **seed=42**：和所有之前实验一致
