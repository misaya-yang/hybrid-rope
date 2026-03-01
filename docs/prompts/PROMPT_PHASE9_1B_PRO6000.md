# Phase 9: 1B 模型 Pro 6000 终极验证实验

> **目标**: 在 1B 模型 + 32K 上下文上验证 EVQ-cosh 的 scalability，产出论文最核心的大模型结果
> **硬件**: RTX Pro 6000 96GB（Blackwell 架构）
> **预计 GPU 时间**: 14-18h
> **前置**: Phase 8 全部完成 + Phase 8D scaling law 验证通过
> **启动条件**: 仅在 Phase 8 结果正面（EVQ PPL ≤ Geometric + 5%）且 scaling law R² > 0.9 时启动

---

## 为什么需要这个实验

Phase 6-8 全部在 125M/350M 上做的。NeurIPS reviewer 最可能的质疑：
1. "这些 observation 能 scale 到真实模型吗？"
2. "350M 太小，frequency allocation 可能不 matter"

1B + 32K 是 96GB 单卡能做到的最大规模，而且正好在 YaRN (7B+128K)、DAPE (125M) 之间，是可信的规模。

---

## 模型配置

```python
MODEL_CONFIG = dict(
    hidden_size=2048,
    num_layers=16,
    num_heads=32,
    head_dim=64,  # 保持 head_dim=64，和 125M/350M 一致，确保 scaling law 可验证
    # 参数量: ~1.0B (embedding excluded)
    # 实际参数: 2048*32000 (embed) + 16*(4*2048^2 + 2048*5461*2) ≈ 1.1B
)
ROPE_BASE = 500_000
VOCAB_SIZE = 32000  # 或 tokenizer 实际大小
```

**为什么 head_dim=64 而不是 128**：
- 和之前所有实验（125M/350M）保持一致，确保 τ* scaling law 的 C=64 可直接对比
- head_dim=128 会让 scaling law 预测 τ* 变为 128/√L，是一个独立的验证，但不如先确认 64 的结论

### 显存估算

```
参数 (fp32): ~4.2 GB
激活 (4K, batch=4, gradient checkpointing): ~12-16 GB
优化器 (AdamW states): ~8.4 GB
训练峰值: ~28-32 GB → Pro 6000 96GB 非常安全

激活 (32K eval, batch=1): ~20-25 GB
Eval 峰值: ~30-35 GB → 安全
```

---

## 阶段 A: 预训练（4K seq_len）

**目的**: 建立 1B geometric baseline checkpoint
**预计时间**: ~6-8h

```python
TRAIN_SEQ_LEN = 4096
TRAIN_TOKENS = 500_000_000  # 500M tokens（~120K steps @ batch=4）
BATCH_SIZE = 4  # 4K * 4 = 16K tokens/step
LR = 3e-4  # 1B 模型的标准 lr
WARMUP_STEPS = 2000
LR_SCHEDULE = "cosine"
DATASET = "HuggingFaceFW/fineweb-edu-score-2"
```

**注意**:
- 只训练 Geometric（τ=0）版本，作为所有 extension 实验的共享 checkpoint
- 500M tokens 对 1B 模型是欠拟合的（Chinchilla 推荐 20B），但对 PE 研究足够——我们关心的是位置编码的相对表现，不是绝对 PPL
- 如果 6h 训不完，可以降到 300M tokens（只要 PPL 收敛到稳定区间即可）
- **每 50K steps 保存 checkpoint**，防止中断

### 预训练评估

训练完后评估 baseline PPL:

```python
PRETRAIN_EVAL_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384, 32768]
```

---

## 阶段 B: Context Extension（4K → 32K，8x 扩展比）

**目的**: 在 1B 上对比 4 种频率方案的 context extension 能力
**预计时间**: ~6h（4 个 run × 1.5h）

### 配置

```python
PRETRAIN_CKPT = "/path/to/phase9/pretrain_4k/checkpoint.pt"  # 阶段 A 产出

TRAIN_SEQ_LEN = 32768  # 32K，8x expansion from 4K
TRAIN_TOKENS = 100_000_000  # 100M tokens（预训练的 20%）— Phase 8B 证明续训量对 passkey 恢复至关重要
BATCH_SIZE = 1  # 32K 需要 batch=1
GRADIENT_ACCUMULATION = 4  # 有效 batch=4
LR = 3e-5  # 续训 lr
DATASET = "HuggingFaceFW/fineweb-edu-score-2"
```

### 显存估算（32K 训练）

```
参数 (fp32): ~4.2 GB
激活 (32K, batch=1, gradient checkpointing): ~35-40 GB
优化器: ~8.4 GB
训练峰值: ~50-55 GB → Pro 6000 96GB 安全（余量 ~40GB）
```

如果 OOM，降 gradient checkpointing 粒度或用 bf16 混合精度：
```python
# 备选：bf16 混合精度
USE_AMP = True
DTYPE = torch.bfloat16  # Pro 6000 Blackwell 原生支持 bf16
# bf16 下参数 ~2.1GB，激活减半，总峰值 ~30-35GB
```

### 要跑的 runs

| Run ID | Method | 频率方案 | τ 值 | 预计时间 |
|--------|--------|---------|------|---------|
| B1 | Geometric | 原始频率不变 | — | ~1.5h |
| B2 | EVQ | cosh 分配，scaling law 最优 | **1.0** | ~1.5h |
| B3 | EVQ | cosh 分配，对照 | 1.5 | ~1.5h |
| B4 | Hybrid EVQ | 高频 8ch Geo + 低频 24ch EVQ | **τ=1.0** | ~1.5h |

**τ 值选择说明（基于 Phase 8C 更新）**：
- **τ=1.0 是主力**：scaling law 预测 τ*(4096) = 1.0。Phase 8C 证明 τ=2.0 对 L=4096 过大（passkey 输 3pp），必须用更小的 τ
- τ=1.5：对照组，看 τ=1.0 vs 1.5 的 passkey 差异
- **Hybrid 也用 τ=1.0**：Phase 8A Hybrid τ=2.0 passkey 不够好，降 τ 应改善
- **不跑 PI/YaRN**：Phase 7F/8A 已证明它们在 8x 扩展比下崩溃

**如果 Phase 8D 的 L=4096 实测 τ* ≠ 1.0**：
- 用实测值替换 B2 和 B4 的 τ，B3 保持 1.5 作为对照

### EVQ 频率生成

直接复用 Phase 8 的 `evq_cosh_inv_freq` 和 `hybrid_evq_inv_freq` 函数，dim=64 不变。

---

## 阶段 C: 评估

**目的**: 全面评估 PPL + Passkey
**预计时间**: ~2-3h

### PPL 评估

```python
EVAL_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
# 512-4096: 预训练窗口内（不应退化）
# 32768: 续训窗口（应该最好）
# 65536: 2x 外推（关键战场）
EVAL_BATCH = 1  # 长序列用 batch=1
```

**65536 显存估算**:
```
Eval (65K, batch=1, no grad): ~45-50 GB → Pro 6000 96GB 安全
```

如果 65K OOM，可以用 sliding window eval 或跳过。

### Passkey 评估

```python
PASSKEY_LENGTHS = [4096, 8192, 16384, 32768]
TRIALS_PER_LENGTH = 100
PASSKEY_POSITION = 0.5  # 中间位置

# 额外：多位置测试（仅在时间充裕时）
PASSKEY_POSITIONS = [0.1, 0.3, 0.5, 0.7, 0.9]  # 5 个位置
TRIALS_PER_POSITION = 50  # 每个位置 50 次
```

---

## 优先级排序

| 优先级 | 阶段 | 时间 | 核心产出 |
|--------|------|------|---------|
| ★★★ P0 | A: 预训练 1B@4K | ~6-8h | Geometric baseline checkpoint |
| ★★★ P0 | B: 4 个 extension runs | ~6h | Geo vs EVQ vs Hybrid PPL |
| ★★★ P0 | C: PPL 评估 (4K-64K) | ~1.5h | 论文核心 PPL 表 |
| ★★ P1 | C: Passkey 评估 (4K-32K) | ~1h | Passkey retrieval rate |
| ★ P2 | C: 多位置 Passkey | ~1h | 锦上添花 |

**推荐执行顺序**: A → B1-B4 顺序跑 → C 评估全部 → 汇总

---

## 目录结构

```
/root/autodl-tmp/evq_phase9/
├── pretrain_4k/                    # 阶段 A
│   ├── checkpoint.pt
│   ├── checkpoint_50k.pt           # 中间 checkpoint
│   ├── checkpoint_100k.pt
│   └── training_log.json
├── ext_32k/                        # 阶段 B
│   ├── extend_geo/
│   ├── extend_evq_1.0/             # 或实测 τ*
│   ├── extend_evq_1.5/
│   └── extend_hybrid_1.5/
├── eval/                           # 阶段 C
│   ├── ppl_results.json
│   └── passkey_results.json
├── results_phase9.json             # 汇总
└── phase9_report.md
```

---

## 汇总 JSON 格式

```json
{
  "phase": 9,
  "date": "2026-03-XX",
  "hardware": "RTX Pro 6000 96GB (Blackwell)",
  "model": "1B (hidden=2048, layers=16, heads=32, head_dim=64)",
  "experiments": {
    "9A_pretrain": {
      "seq_len": 4096,
      "tokens": "500M",
      "train_time_h": null,
      "ppl_baseline": {
        "512": null, "1024": null, "2048": null, "4096": null,
        "8192": null, "16384": null, "32768": null
      }
    },
    "9B_extension": {
      "expansion_ratio": "8x (4K->32K)",
      "continuation_tokens": "50M",
      "methods": {
        "geometric": {
          "ppl": {"512": null, "4096": null, "8192": null, "16384": null, "32768": null, "65536": null},
          "passkey": {"4096": null, "8192": null, "16384": null, "32768": null},
          "train_time_h": null
        },
        "evq_1.0": {
          "ppl": {}, "passkey": {}, "train_time_h": null
        },
        "evq_1.5": {
          "ppl": {}, "passkey": {}, "train_time_h": null
        },
        "hybrid_1.5": {
          "ppl": {}, "passkey": {}, "train_time_h": null
        }
      }
    }
  },
  "scaling_law_check": {
    "L_train": 4096,
    "predicted_tau_star": 1.0,
    "observed_best_tau": null,
    "note": "Compare B2 (tau=1.0) vs B3 (tau=1.5) to check scaling law at 1B scale"
  }
}
```

---

## 结果输出格式

phase9_report.md 必须包含：

1. **预训练 baseline PPL 表**: 1B@4K 在各 eval 长度的 PPL
2. **Extension 结果表**: 4 个方法 × 8 个 eval 长度的 PPL
3. **PPL@eval_length 曲线图**: 4 条线，标注 4K 训练窗口和 32K 续训窗口
4. **与 Phase 8A（350M）的对比**: 相同扩展比（8x）下，1B vs 350M 的方法排名是否一致
5. **Passkey 结果表**: 4 个方法 × 4 个长度的 retrieval rate
6. **Scaling law 验证**: B2 (τ=1.0) vs B3 (τ=1.5) 在 PPL@32K/64K 上的表现，判断 τ*(4096) 更接近 1.0 还是 1.5
7. **关键结论**: EVQ/Hybrid 是否在 1B 上保持了 350M 上观察到的趋势

---

## 注意事项

1. **预训练是最耗时的部分（~6-8h）**: 如果中断，从最近的 checkpoint 恢复。每 50K steps 自动保存
2. **32K 训练用 gradient accumulation**: batch_size=1, grad_accum=4。如果 OOM，开 bf16
3. **不跑 PI 和 YaRN**: Phase 7F/8A 已经证明它们在 8x 扩展比下远不如 Geo/EVQ，1B 实验不需要重复验证已知失败的方法
4. **EVQ τ 的最终值取决于 Phase 8D**: 如果 Phase 8D 证实 τ*(4096)=1.0，则 B2 用 1.0；如果不是，用实测值
5. **65536 eval 可能 OOM**: 如果 OOM，降为 49152 或跳过。论文只需要 eval 到 2×续训长度（64K）即可
6. **Passkey 代码完全复用 Phase 7F/8A**: 不修改评估逻辑
7. **bf16 混合精度是首选**: Blackwell 架构的 bf16 性能远优于 fp32，且 Pro 6000 的 tensor core 针对 bf16 优化。建议全程 bf16 训练
8. **如果预训练超时**: 可以用 300M tokens 的 checkpoint（只要 training loss 已收敛到平台期）。论文中注明 tokens 数即可
9. **保存完整的 training curve**: loss vs step，用于判断是否收敛
10. **实验数据保存到 `/root/autodl-tmp/evq_phase9/`**: 和 phase7/8 平级
