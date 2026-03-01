# Phase 8G: Base=10K 对照实验（验证大 base 掩盖 EVQ 优势的假说）

> **目标**: Phase 8 全部用 base=500K，Geometric 自身就很强（θ_min 周期 1.9M tokens，16K eval 只走了 0.8%）。本实验用 base=10K（LLaMA-2 原始设置），验证 EVQ 优势在低 base 下显著放大。
> **硬件**: RTX 5090 32GB
> **预计 GPU 时间**: ~1.5h（3 个 runs × ~25 min）
> **前置**: 8F 完成后执行（或与 8F 并行，如果显存够）

---

## 为什么需要这个实验

### Base 对 Geometric 外推能力的影响

| Base | θ_min 周期 | 16K eval 占周期 | Geometric 压力 |
|------|-----------|----------------|---------------|
| 10,000 | ~38K tokens | **42%** | **高压** |
| 500,000 | ~1.9M tokens | 0.8% | 几乎无压力 |

base=500K 下 Geometric 不崩是因为低频通道根本没被 challenge。base=10K 下，16K eval 已经接近最低频通道的半周期——Geometric 应该会明显退化，而 EVQ 的频率重分配优势会显著放大。

### 预期结果

| 指标 | base=500K（已有 8E） | base=10K（预期） |
|------|---------------------|-----------------|
| Geo PPL@16K | 175.4 | 更差（>200?） |
| EVQ τ=1.0 PPL@16K | 180.1 (+2.7%) | **显著好于 Geo** |
| EVQ vs Geo PPL gap | -6.3% (τ=2.0) | **-15% ~ -25%?** |
| EVQ vs Geo PK gap | +3pp | **更大?** |

---

## 实验设计

### 配置

和 8C/8E **完全相同**，只改 `ROPE_BASE`：

```python
MODEL_CONFIG = dict(hidden=1024, layers=24, heads=16, head_dim=64)  # 350M
TRAIN_SEQ_LEN = 4096
TRAIN_TOKENS = 50_000_000  # 50M tokens
ROPE_BASE = 10_000  # ← 唯一的区别！
BATCH_SIZE = 2
LR = 6e-4
DATASET = "HuggingFaceFW/fineweb-edu-score-2"
SEED = 42  # 和 8C/8E 同 seed，方便 paired 对比
```

### 要跑的 runs

| Run ID | Method | τ | 目录 | 预计时间 |
|--------|--------|---|------|---------|
| G1 | Geometric | — | `base10k/geo_4k/` | ~25 min |
| G2 | EVQ τ=1.0 | 1.0 | `base10k/evq1.0_4k/` | ~25 min |
| G3 | Hybrid τ=1.0 | 1.0 | `base10k/hybrid1.0_4k/` | ~25 min |

**注意**: τ=1.0 是 scaling law 对 L=4096 的预测，和 base 无关（scaling law 推导中 base 被吸收进 θ_min/θ_max 的边界条件，τ* 只依赖 L 和 d_head）。

### EVQ 频率生成

完全复用 `evq_cosh_inv_freq` 和 `hybrid_evq_inv_freq`，只把 `base=10000` 传进去：

```python
# base=10K 时的频率范围
theta_max = 1.0
theta_min = 1.0 / (10000 ** (62/64))  # ≈ 1.65e-4（vs base=500K 的 3.3e-6）
# θ_min 大了 50 倍 → 低频端频率更高 → 外推能力更差（对 Geometric 来说）
# EVQ 可以在这个更窄的频率范围内做更优分配
```

---

## 评估

```python
EVAL_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384]
PASSKEY_LENGTHS = [1024, 2048, 4096, 8192]
TRIALS_PER_LENGTH = 100
```

---

## 关键对比

### 1. Base=10K 内部对比（G1 vs G2 vs G3）

期望 EVQ 大幅赢 Geometric：
- PPL@16K: EVQ 预期 -15% ~ -25% vs Geo（base=500K 下只有 -6.3%）
- Passkey: EVQ 预期 +5pp ~ +10pp vs Geo

### 2. 跨 Base 对比（8E vs 8G，同 seed=42）

| | base=500K (8E) | base=10K (8G) | 解读 |
|---|---------------|--------------|------|
| Geo PPL@16K | 175.4 | ? (更差) | 大 base 帮了 Geo |
| EVQ PPL@16K | 180.1 | ? | EVQ 受 base 影响小? |
| EVQ-Geo gap | +2.7% | ? (更大负值) | **EVQ 优势被 base 掩盖的程度** |

### 3. 论文 narrative

如果 8G 确认 base=10K 下 EVQ 大幅赢 Geo：
> "The benefit of EVQ-cosh frequency allocation is inversely related to the RoPE base frequency. With base=10K (LLaMA-2), EVQ achieves −XX% PPL and +Ypp passkey vs Geometric. With base=500K (LLaMA-3), the advantage narrows to −6.3% PPL and +3pp passkey, as the larger base already provides sufficient frequency spread."

这个 narrative 比"EVQ 只赢一点点"强得多——说明 EVQ 的优势是 real 的，只是被现代大 base 部分抵消了。

---

## 目录结构

```
/root/autodl-tmp/evq_phase8/
├── ...（已有 8A-8F）
└── base10k/                        # 8G
    ├── geo_4k/
    ├── evq1.0_4k/
    └── hybrid1.0_4k/
```

## 汇总 JSON（追加到 results_phase8.json）

```json
{
  "8G_base10k_comparison": {
    "purpose": "Verify EVQ advantage amplifies with smaller RoPE base",
    "rope_base": 10000,
    "model": "350M, from-scratch 4K, 50M tokens, seed=42",
    "results": {
      "geometric": {"ppl": {}, "passkey_global": null, "passkey_by_length": {}},
      "evq_1.0": {"ppl": {}, "passkey_global": null, "passkey_by_length": {}},
      "hybrid_1.0": {"ppl": {}, "passkey_global": null, "passkey_by_length": {}}
    },
    "cross_base_comparison": {
      "geo_gap_500k": "+2.7%",
      "geo_gap_10k": null,
      "note": "Larger gap at base=10K confirms base masks EVQ advantage"
    }
  }
}
```

## 注意事项

1. **只改 ROPE_BASE，其他全部不动**：模型结构、训练 tokens、lr、seed 都和 8E 一致
2. **base=10K 的 PPL 绝对值会不同**：不要拿 base=10K 和 base=500K 的绝对 PPL 比较，只比较 EVQ vs Geo 的**相对 gap**
3. **τ=1.0 仍然适用**：scaling law τ*=d_head/√L 不依赖 base（base 只影响频率范围的端点，不影响最优分配形状）
4. **如果 Geo PPL@16K 在 base=10K 下直接爆炸（>500）**：说明 base=10K 对 4K 训练 + 16K eval 已经太弱了，可以只看 PPL@8K
5. **这个实验也是 reviewer 会问的**："你试过不同 base 吗？" 提前做了就不用 rebuttal 补
