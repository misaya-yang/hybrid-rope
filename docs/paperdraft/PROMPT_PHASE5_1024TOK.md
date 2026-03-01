# Claude Code 实验指令：Phase 5 — 1024-Token Training + Passkey + SOTA Baselines

> **前置条件**: Phase 0-4 已完成/进行中。请先读 `docs/paperdraft/EXPERIMENT_RESULTS_128TOK.md` 了解已有结果。

---

## 目标

在 5090 (32GB) 上用最少 GPU 时间获取以下证据：
1. **Passkey retrieval**：证明 EVQ 不只降 PPL，还提升实际检索能力
2. **SOTA baseline 对比**：加入 FoPE (ICML 2025) 作为最新竞品
3. **Scaling**：50M + 125M 两个规模点，验证 EVQ 优势随规模的稳定性

---

## 实验 A: 1024-Token 训练 + Passkey（核心，~4h）

### A.1 训练配置

```yaml
Common:
  model_sizes: [50m, 125m]
  train_seq_len: 1024
  train_tokens: 50M  # 比 128-tok 更多，因为序列更长
  dataset: fineweb-edu
  base: 500000.0
  seed: 42
  eval_ppl_lengths: [1024, 2048, 4096, 8192]
  eval_passkey_lengths: [2048, 4096, 8192]  # 都是训练外
```

### A.2 方法矩阵

**125M 模型（必跑，6 runs）**:

| Run | Method | 实现 | 说明 |
|-----|--------|------|------|
| 1 | Geometric RoPE | `--method geometric --tau 0` | baseline |
| 2 | EVQ fixed τ=1.5 | `--method evq_cosh --tau 1.5` | 我们的方法 |
| 3 | EVQ learnable | `--method learnable --tau_init 1.0 --tau_lr_mult 100` | 自适应 |
| 4 | YaRN | 见下方实现 | ICLR 2024 baseline |
| 5 | PI | `inv_freq /= (target_len / train_len)` | 经典 baseline |
| 6 | FoPE | 见下方实现 | **ICML 2025 SOTA** |

**50M 模型（scaling 验证，3 runs）**:

| Run | Method | 说明 |
|-----|--------|------|
| 7 | Geometric | 50M baseline |
| 8 | EVQ fixed τ=1.5 | 50M EVQ |
| 9 | EVQ learnable | 50M learnable |

### A.3 需要实现的方法

#### YaRN (NTK-by-parts)

```python
def yarn_inv_freq(head_dim, base, train_len, target_len, beta_fast=32, beta_slow=1):
    """YaRN: NTK-by-parts interpolation."""
    n = head_dim // 2
    scale = target_len / train_len

    # Standard geometric frequencies
    idx = torch.arange(n, dtype=torch.float64)
    inv_freq = 1.0 / (base ** (idx / n))

    # Compute wavelengths
    wavelengths = 2 * math.pi * base ** (idx / n)

    # Low/high frequency boundaries
    low_freq = 1.0 / (wavelengths / (beta_fast * train_len))
    high_freq = 1.0 / (wavelengths / (beta_slow * train_len))

    # Ramp: 0 for high freq (no scaling), 1 for low freq (full PI scaling)
    ramp = (wavelengths / train_len - beta_slow) / (beta_fast - beta_slow)
    ramp = ramp.clamp(0, 1)

    # Interpolate between no-scaling and PI-scaling
    inv_freq_scaled = inv_freq / scale  # PI scaling
    inv_freq_yarn = inv_freq * (1 - ramp) + inv_freq_scaled * ramp

    return inv_freq_yarn
```

#### FoPE (Fourier Position Embedding, ICML 2025)

FoPE 的核心思路：每个维度不再用单一频率，而是用傅里叶级数（主频 + 谐波）。实现需要：

```python
def fope_inv_freq(head_dim, base, n_harmonics=3):
    """FoPE: Fourier series for each frequency channel.

    参考: arxiv 2412.17739, Section 3.2
    核心: 每个维度 d 的编码从 cos(ω_d * m) 变成 Σ_h a_h * cos(h * ω_d * m)

    简化实现（inference-time, 不需要训练 W_F 矩阵）:
    - 保留 RoPE 的频率结构
    - 对低频分量（频率 < 某阈值）零化（Clip-Floor-to-Zero）
    - 这是 FoPE 的 CF 部分，不需要额外参数
    """
    n = head_dim // 2
    idx = torch.arange(n, dtype=torch.float64)
    inv_freq = 1.0 / (base ** (idx / n))

    # CF (Clip Floor to Zero): 将训练不充分的低频分量置零
    # 阈值: 在训练长度内不足 1 个完整周期的频率
    # 这是 FoPE 论文的核心简化版本
    # 完整 FoPE 需要训练 W_F 矩阵，这里用 CF-only 作为 baseline
    return inv_freq  # CF 部分在 attention 计算时处理
```

**注意**: FoPE 的完整实现需要训练 W_F 矩阵（将单频扩展为傅里叶级数）。在 from-scratch 训练中，可以直接用 FoPE 的 CF (Clip-Floor-to-Zero) 子方法作为 baseline——这是 FoPE 论文中报告的主要 inference-time 组件。如果 CF-only 不好实现，可以跳过 FoPE，论文中只 cite 并讨论差异即可。

### A.4 Passkey 评测实现

```python
def eval_passkey(model, tokenizer, test_lengths=[2048, 4096, 8192], n_trials=20):
    """Passkey retrieval evaluation.

    Protocol:
    1. 生成格式: "The passkey is {5-digit number}. " + padding + "What is the passkey?"
    2. padding 用随机 token 填充到 target_length
    3. passkey 位置在序列中随机放置
    4. 成功标准: 模型输出包含正确的 5 位数字

    Output: accuracy@length for each method
    """
    results = {}
    for length in test_lengths:
        correct = 0
        for trial in range(n_trials):
            passkey = str(random.randint(10000, 99999))
            # 构造输入: passkey 在随机位置
            insert_pos = random.randint(0, length - 200)
            # ... 生成 prompt, 运行 model.generate(), 检查输出
            if passkey in output:
                correct += 1
        results[length] = correct / n_trials
    return results
```

**已有参考代码**: `submission/code/eval_passkey.py` 和 `submission/code/eval_passkey_teacher_forcing.py`。优先复用这些代码。

---

## 实验 B: 128-tok 补充 50M scaling 点（~30min）

用 Phase 1 同样的配置，跑 50M 模型 3 个 run：

| Run | Method | 说明 |
|-----|--------|------|
| 1 | Geometric | 50M baseline |
| 2 | EVQ fixed τ=1.5 | 50M EVQ |
| 3 | Learnable EVQ | 50M learnable |

与已有 125M 结果组合成 scaling 曲线。

---

## 输出要求

### 必须保存的文件

```
/root/autodl-tmp/evq_1024tok/
├── results_summary.json          # 所有 run 的 PPL + passkey 结果
├── 125m_geometric_seed42/
│   ├── model.pt
│   └── eval_results.json         # PPL@各长度 + passkey accuracy
├── 125m_evq_fixed1.5_seed42/
│   ├── model.pt
│   └── eval_results.json
├── 125m_learnable_init1.0_seed42/
│   ├── model.pt
│   ├── tau_trajectory.json       # 必须保存
│   └── eval_results.json
├── 125m_yarn_seed42/
│   └── eval_results.json
├── 125m_pi_seed42/
│   └── eval_results.json
├── 50m_geometric_seed42/
│   └── eval_results.json
├── 50m_evq_fixed1.5_seed42/
│   └── eval_results.json
└── 50m_learnable_init1.0_seed42/
    ├── tau_trajectory.json
    └── eval_results.json
```

### 必须输出的汇总表

```
Phase 5 Results (1024-token training)

Method         | Params | PPL@1K | PPL@2K | PPL@4K | PPL@8K | PK@2K | PK@4K | PK@8K | Δ@8K
---------------|--------|--------|--------|--------|--------|-------|-------|-------|------
Geometric      | 0      | ...    | ...    | ...    | ...    | ...%  | ...%  | ...%  | —
PI             | 0      | ...    | ...    | ...    | ...    | ...%  | ...%  | ...%  | ...
YaRN           | 0      | ...    | ...    | ...    | ...    | ...%  | ...%  | ...%  | ...
EVQ τ=1.5      | 0      | ...    | ...    | ...    | ...    | ...%  | ...%  | ...%  | ...
EVQ learnable  | 1      | ...    | ...    | ...    | ...    | ...%  | ...%  | ...%  | ...
DAPE(从128结果) | 32     | —      | —      | —      | —      | —     | —     | —     | (ref)

PK = Passkey retrieval accuracy
```

---

## 优先级

如果 GPU 时间不够全做：
1. **必做**: 125M 的 Geometric + EVQ τ=1.5 + Learnable + Passkey（4 runs, ~2h）
2. **高优**: 125M YaRN + PI（2 runs, ~1h）
3. **中优**: 50M scaling 点（3 runs, ~1h）
4. **可选**: FoPE baseline（实现复杂，可跳过）

---

## GPU 内存注意

1024 tokens, 125M 模型在 32GB 5090 上应该没问题（batch_size=16-32）。
Passkey 评测在 8K 长度时可能需要降低 batch_size 到 1 并用 torch.no_grad()。

---

*指令创建: 2026-03-01*
