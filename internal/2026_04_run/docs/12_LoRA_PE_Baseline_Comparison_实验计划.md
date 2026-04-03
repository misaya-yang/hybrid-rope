# LoRA PE Baseline Comparison 实验计划

> **日期**: 2026-04-03
> **目标**: 同一 base model + 同一 LoRA config，只换 PE 频率分配方法，多 seed 对照
> **核心指标**: Positional PPL 分区间 (0-4K / 4K-8K / 8K-12K / 12K-16K)
> **预算**: ~18h GPU (3 methods × 3 seeds × 300 steps × ~2h)

---

## 1. 实验矩阵

| 方法 | 频率分配 | inv_freq 实现 | 需要训练 |
|------|---------|--------------|---------|
| **Geometric (控制组)** | base^{-2k/d}，标准等比 | 不改 inv_freq（τ=0） | ✅ 300 steps × 3 seeds |
| **EVQ-cosh** | φ_k = 1-(1/τ)arcsinh((1-u_k)sinh(τ)) | inject EVQ inv_freq, τ=1.414 | ✅ 已有 seed42，补 seed43/44 |
| **YaRN** | NTK-aware 分段缩放 + attn temperature | rope_scaling config | ✅ 300 steps × 3 seeds |

### 可选第四组
| **LongRoPE2** | 进化搜索每维度 rescale factor | 需先搜索再训练 | 搜索 ~5h + 训练 ~6h |

---

## 2. 固定变量（所有组共享）

| 参数 | 值 |
|------|-----|
| Base model | LLaMA-3-8B-Instruct |
| LoRA | r=64, α=128, dropout=0.05, targets=qkvo |
| 精度 | bf16 full precision |
| 训练步数 | 300 |
| Batch | bs=2, grad_accum=4 (effective bs=8) |
| LR | 1e-4, cosine, warmup=60, wd=0.01 |
| 数据 | LongAlpaca-12k, seq_len=8192 |
| Seeds | 42, 43, 44 |

---

## 3. 各方法实现细节

### 3.1 Geometric (τ=0)

不改 inv_freq。等价于标准 LoRA fine-tuning。
```bash
python train_evq_lora.py --tau 0 --seed 42 --output_dir ./checkpoints/geo_s42
python train_evq_lora.py --tau 0 --seed 43 --output_dir ./checkpoints/geo_s43
python train_evq_lora.py --tau 0 --seed 44 --output_dir ./checkpoints/geo_s44
```

### 3.2 EVQ-cosh (τ=1.414)

已有 seed=42 的 checkpoint。补跑 seed=43, 44。
```bash
# seed42 已有: ./checkpoints/evq_r64_tau1414/
python train_evq_lora.py --tau 1.414 --seed 43 --output_dir ./checkpoints/evq_s43
python train_evq_lora.py --tau 1.414 --seed 44 --output_dir ./checkpoints/evq_s44
```

### 3.3 YaRN

在模型加载时设置 `rope_scaling`，训练时模型自动使用 YaRN 频率。
```bash
python train_yarn_lora.py --yarn_factor 2.0 --seed 42 --output_dir ./checkpoints/yarn_s42
python train_yarn_lora.py --yarn_factor 2.0 --seed 43 --output_dir ./checkpoints/yarn_s43
python train_yarn_lora.py --yarn_factor 2.0 --seed 44 --output_dir ./checkpoints/yarn_s44
```

### 3.4 LongRoPE2（可选）

1. 先用进化搜索找到 LLaMA-3-8B 在 16K 目标下的最优 per-dimension rescale factors
2. 将 rescale factors 应用为 inv_freq 修正
3. 训练 LoRA 300 steps × 3 seeds

实现复杂度较高，优先级 P2。

---

## 4. 评测方案

### 4.1 主指标：Positional PPL 分区间

对每个 checkpoint 跑一次 16K WikiText PPL，按位置分桶：
- 0-4K（训练区内，应该都好）
- 4K-8K（训练区边界）
- 8K-12K（外推区间 1，关键对比区）
- 12K-16K（外推区间 2）

### 4.2 辅助指标

| 指标 | 说明 |
|------|------|
| Gold-Answer NLL | LongBench QA 的正确答案 NLL（本地 JSONL） |
| LongPPL | 只算 key token 的 PPL（ICLR 2025 方法） |
| 训练 loss 曲线 | 收敛速度和最终 loss 对比 |

---

## 5. 预期结果与论文叙事

| 对比 | 预期 | 论文价值 |
|------|------|---------|
| EVQ vs Geo | EVQ 在 8K-16K PPL 显著更低 | 核心论点：变分最优频率分配 > 等比分配 |
| EVQ vs YaRN | 待验证。如果 EVQ ≥ YaRN → EVQ 是理论最优的无搜索方案 | |
| EVQ vs LongRoPE2 | 待验证。LongRoPE2 用搜索可能更好，但 EVQ 是 closed-form | |
| 多 seed 误差线 | 证明结果稳定，不是随机 | 统计显著性 |

### 关键叙事

"EVQ-cosh 提供了 closed-form 的最优频率分配。在 LoRA 微调场景下，EVQ 无需搜索即可获得与 YaRN/LongRoPE2 搜索方案相当甚至更优的外推 PPL 改善。"

---

## 6. 执行顺序

1. **准备代码**：train_yarn_lora.py（YaRN 版训练脚本）+ eval_positional_ppl.py（统一评测脚本）
2. **第一轮**：Geo × 3 seeds + EVQ × 2 seeds（补跑）= 5 次训练，~10h
3. **第二轮**：YaRN × 3 seeds = 3 次训练，~6h
4. **评测**：9 个 checkpoint × positional PPL，~1h
5. **可选**：LongRoPE2 搜索 + 训练，~11h

总计约 **18-28h GPU**。

---

## 7. 服务器路径

```
MODEL=/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct
DATA=/root/autodl-tmp/data/longalign_10k/longalign_10k.jsonl
WIKI=/root/autodl-tmp/data/wikitext2/wikitext2_test.txt
BASE_DIR=/root/autodl-tmp/lora_evq_v2/checkpoints/
```
