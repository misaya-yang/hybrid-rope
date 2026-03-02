# Phase 10: Supervised Passkey Probing — 消除 Zero-Shot 混淆因素

> **核心问题**: Phase 0-8F 的 passkey 评测全部是 zero-shot（训练数据不含 `<<PASS:...>>` 格式）。350M 模型的 copy head 能力不足，导致 zero-shot passkey ~55-70%，方法间差异被 copy head 噪声淹没。**我们无法区分"EVQ 频率分配确实没帮助"和"模型根本不具备 copy 能力所以看不出差异"。**
>
> **解决方案**: 在训练中混入少量 passkey supervision（0.5%），让模型学会 copy 机制，然后比较 Geo vs EVQ/Hybrid 在 supervised 条件下的 passkey 表现。这消除了 copy head emergence 的混淆因素，直接测试频率分配对 retrieval 的影响。
>
> **硬件**: 5090 32GB（非 R6000，R6000 继续跑 Phase 9F）
> **预计 GPU 时间**: ~4-6h
> **前置**: `eval_passkey_scratch.py` 中的 `MixedDataset` 和 `make_passkey_training_sample` 已实现

---

## 理论动机

### 为什么 PPL 赢但 Passkey 不赢？

| 指标 | 测什么 | 需要什么 | EVQ 提供什么 |
|------|--------|----------|-------------|
| PPL | 全位置平均预测质量 | 好的频率分配 (E_alloc) | ✅ 直接受益 |
| Passkey | 单位置精确 copy | 好的频率 + 学会 copy 机制 | ✅ 频率 + ❌ copy 要靠模型自己学 |

E_total = E_alloc(τ) + E_copy(N_params, N_tokens)

- 小模型 (≤350M, ≤50M tok): E_copy >> E_alloc → passkey 被 copy 能力瓶颈限制
- 大模型 (8B): E_copy ≈ 0 → passkey 直接反映 E_alloc → EVQ 100% vs Geo 80%

### Supervised Passkey = 消除 E_copy

通过 0.5% passkey mixing，我们直接给模型教会 copy 机制（E_copy → 0），然后 passkey 差异就只反映 E_alloc —— 即 EVQ 频率分配的真实效果。

**论文叙事**: "Zero-shot passkey 在小模型上统计等价，因为 copy mechanism 是 emergent capability。为排除这个混淆因素，我们加入 0.5% passkey supervision 后重新比较..."

---

## 实验设计

### 10A: 350M, L=2048, 50M tokens, passkey_ratio=0.5%（主实验）

**为什么选 L=2048**: 这是历史信号最强的 regime（anchored sigmoid 全线赢、5090 PPL@16K -15.4%）。

**为什么 350M 而非 750M**: 5090 跑 350M 快（~1-1.5h/run），可以做更多消融。如果 350M supervised 就能看到差异，说明问题确实是 copy head，不是模型规模。

#### 配置

```python
# === 基本配置 ===
MODEL_TIER = "350m"  # hidden=1024, layers=24, heads=16, head_dim=64
SEQ_LEN = 2048
TOKENS = 50_000_000  # 50M tokens（与 5090 验证实验一致）
ROPE_BASE = 500_000
SEED = 42
LR = 6e-4  # from-scratch 标准 lr
DATASET = "fineweb-edu"

# === Passkey Mixing ===
PASSKEY_RATIO = 0.005  # 0.5%，已在 run_evq_sweep.py 中验证过
# MixedDataset 已实现: eval_passkey_scratch.py

# === 评测 ===
EVAL_LENGTHS = [1024, 2048, 4096, 8192, 16384]
PK_LENGTHS = [1024, 2048, 4096, 8192]
PK_TRIALS = 50  # 比之前 40 略多
PK_DEPTHS = [0.5]  # 固定 depth=0.5 减少 variance
```

#### Runs（共 4 个，~1.5h/run = ~6h 总计）

| Run | Method | τ | r | 预期 |
|-----|--------|---|---|------|
| 10A-1 | Geometric | — | — | Baseline |
| 10A-2 | EVQ | 1.5 | 0 (pure) | τ* for L=2048 |
| 10A-3 | Hybrid | 1.5 | 16 | τ=1.5, r*=16 |
| 10A-4 | Geometric (no passkey) | — | — | 对照：无 passkey mixing |

> **10A-4 是关键对照**：和 10A-1 完全相同但 passkey_ratio=0。对比 10A-1 vs 10A-4 可以量化 passkey mixing 对 PPL 的影响（应 <1%），以及对 passkey 本身的提升（预期从 ~55% 跳到 85%+）。

#### 关键代码改动

Phase 9F 的脚本是 tensor-based（`train_data[perm[idx]]`），不支持 MixedDataset。需要：

```python
# 方案 1: 直接用 run_evq_sweep.py 的框架（推荐）
# run_evq_sweep.py 已经集成了 MixedDataset，只需设置正确的 tier 和 tau

# 方案 2: 修改 phase9f 脚本支持 MixedDataset
# 在 train_model_ga() 中把 data[perm[chunk_idx : ...]] 替换为 MixedDataset 的 __getitem__
```

**推荐方案 1**：直接调用 `run_evq_sweep.py` 的 `run_single_experiment()` 函数，它已经内置了 MixedDataset。只需要：
1. 设置 `tier="350m"`, `seq_len=2048`, `tokens=50_000_000`
2. 设置不同的 `tau` 和 `method` 参数
3. 评测逻辑也已内置

如果 `run_evq_sweep.py` 不支持 Hybrid，则需要小幅改动加入 Hybrid inv_freq 构造。

#### 成功标准

| 条件 | 判定 |
|------|------|
| 10A-1 (Geo+PK) passkey ≥ 80% | Supervised passkey 训练生效 |
| 10A-4 (Geo-no-PK) passkey < 65% | Zero-shot 确实是瓶颈 |
| 10A-1 vs 10A-4 PPL 差异 < 2% | Passkey mixing 不损害 PPL |
| **10A-2 或 10A-3 passkey > 10A-1** | **🎉 EVQ 频率分配在排除 copy 瓶颈后确实帮助 retrieval** |
| 10A-2/3 passkey ≈ 10A-1 | EVQ 帮 PPL 不帮 retrieval → 论文只能讲 PPL story |

---

### 10B: Passkey Ratio 消融（如果 10A 信号明确）

**目的**: 找到最小有效 passkey ratio，确认不是 "暴力记忆" 而是 "copy 机制学习"。

| Run | passkey_ratio | 预期 |
|-----|---------------|------|
| 10B-1 | 0.0% | = zero-shot baseline |
| 10B-2 | 0.1% | 最小剂量 |
| 10B-3 | 0.5% | = 10A 标准 |
| 10B-4 | 2.0% | 高剂量 |

只跑 Geometric baseline，每个 ~1.5h。看 passkey rate vs passkey_ratio 的 saturation curve。

**如果 0.1% 就饱和**：说明模型只需要一点 hint 就能学会 copy → copy mechanism 确实是 emergent，只是需要最小 supervision。
**如果需要 2%+ 才工作**：可能是暴力记忆格式而非真正学会 copy → 需要更谨慎地解读结果。

---

### 10C: Eval Length Extrapolation（如果 10A 信号明确）

**关键测试**: Supervised passkey 只在 L_train=2048 内训练。模型能否 extrapolate 到 4K/8K/16K？

```python
PK_LENGTHS = [1024, 2048, 4096, 8192, 16384]
# 训练只见过 ≤ 2048 的 passkey 样本
# 4K/8K/16K 是 extrapolation 区域
```

**预期**:
- @1K, @2K: Supervised 大幅提升（interpolation 区域，所有方法都 >90%）
- @4K: 部分提升（2x extrapolation，80%+ 如果 copy head 泛化）
- @8K+: 关键战场 —— 如果 EVQ > Geo，说明频率分配确实帮助 long-range copy
- @16K: 可能两边都 drop，但 drop 幅度的差异就是频率分配的贡献

**这是论文最想要的数据**：在 extrapolation 区域（>L_train），supervised passkey 差异 = 纯频率分配效果。

---

## 完整执行流程

```
# 在 5090 上执行（不动 R6000）

# Step 0: 环境
export PATH="/root/miniconda3/bin:$PATH"
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1

# Step 1: 跑 10A（4 runs，~6h）
# 按 10A-1 → 10A-4 顺序执行
# 每跑完一个检查 result.json 确认无报错

# Step 2: 分析 10A 结果
# 关键比较：
#   (1) 10A-1 vs 10A-4：supervised 效果（passkey 提升量）
#   (2) 10A-1 vs 10A-2 vs 10A-3：方法间差异（频率分配效果）
#   (3) 所有 runs 的 PPL 对比（确认 passkey mixing 不损 PPL）

# Step 3: 根据 10A 结果决定是否跑 10B/10C
#   如果 10A 显示 EVQ/Hybrid passkey > Geo → 跑 10C（extrapolation 是论文数据）
#   如果 10A 显示无差异 → 跑 10B（排除 ratio 太低的可能）
#   如果 10A 显示 passkey mixing 没提升 → 检查代码实现是否有 bug
```

---

## Claude Code 启动提示词

```
你是实验执行者。请执行 Phase 10A: Supervised Passkey Probing 实验。

## 任务
在 5090 上从零训练 4 个 350M 模型 (L=2048, 50M tokens, base=500K, seed=42)：
1. Geometric + passkey_ratio=0.5%
2. EVQ τ=1.5 + passkey_ratio=0.5%
3. Hybrid τ=1.5 r=16 + passkey_ratio=0.5%
4. Geometric + passkey_ratio=0 (zero-shot 对照)

## 实现路径
优先复用 `scripts/m4_evq_sweep/run_evq_sweep.py` 的框架（已集成 MixedDataset）。
如果不兼容，基于 phase9f 脚本改写：
- 用 MixedDataset 包装 train_data
- `from eval_passkey_scratch import MixedDataset`
- `mixed = MixedDataset(lm_data=train_data, filler_tokens=val[:50000], tokenizer=tok, passkey_ratio=0.005, seq_len=2048)`

## Hybrid inv_freq 构造（如果框架不支持）
```python
def hybrid_evq_inv_freq(dim=64, base=500000, tau=1.5, r=16):
    n = dim // 2
    geo = torch.tensor([1.0/(base**(2*i/dim)) for i in range(n)], dtype=torch.float64)
    n_evq = n - r
    theta_max, theta_min = geo[r].item(), geo[-1].item()
    u = torch.arange(n_evq, dtype=torch.float64) / max(n_evq - 1, 1)
    phi = 1.0 - (1.0/tau) * torch.arcsinh((1.0 - u) * math.sinh(tau))
    evq_part = (theta_min**phi) * (theta_max**(1.0 - phi))
    return torch.cat([geo[:r], evq_part]).float()
```

## 评测
- PPL: [1024, 2048, 4096, 8192, 16384], 6 chunks
- Passkey: [1024, 2048, 4096, 8192], depth=0.5, 50 trials/length
- 特别关注 @4K 和 @8K（extrapolation 区域）

## 输出
每个 run 保存 result.json 到 `/root/autodl-tmp/evq_phase10/`
跑完后打印 summary 表格：method, PPL@2K, PPL@16K, PK@1K, PK@2K, PK@4K, PK@8K

## 注意事项
1. 先跑 10A-1 (Geo+PK) 和 10A-4 (Geo-no-PK) 作为对照
2. 确认 passkey mixing 生效：打印 `[passkey] Mix ratio check: X/1000`
3. 如果 10A-1 passkey < 70%，说明 supervised 没起作用，检查 MixedDataset 实现
4. 先读 scripts/m4_evq_sweep/eval_passkey_scratch.py 理解 MixedDataset 接口
5. 先读 scripts/m4_evq_sweep/run_evq_sweep.py 看现有集成方式
```

---

## 与论文的关系

| Phase 10 结果 | 论文影响 |
|--------------|---------|
| Supervised 下 EVQ PK > Geo | **最强论据**：频率分配 + copy ability 解耦后 EVQ 确实更好。Figure: "Zero-shot vs Supervised passkey by method" |
| Supervised 下全部 >90% 无差异 | 可接受：说明 copy 饱和后频率不再是瓶颈。论文讲 PPL story + scale-dependent emergence |
| Supervised 没提升 passkey | Bug 或方法问题，需要 debug |
| Extrapolation 区域 (>2K) EVQ 赢 | **杀手级结果**：supervised 只教了 ≤2K 的 copy，EVQ 在 >2K 仍然赢 → 纯频率贡献 |

**最理想的论文 Figure**:

```
x: eval context length (1K, 2K, 4K, 8K)
y: passkey retrieval rate
lines: Geo-zero, Geo-supervised, EVQ-supervised, Hybrid-supervised
预期: Geo-zero ~55%, Geo-supervised ~90%@2K dropping to ~60%@8K
      EVQ-supervised ~90%@2K dropping to ~70%@8K (差距在 extrapolation 区间拉大)
```

这张图同时讲了两个 story：(1) supervised 消除 copy 瓶颈；(2) EVQ 的频率分配在 extrapolation 区间发力。
