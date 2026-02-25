# NeurIPS 严格实验设计

## 核心假设

**Zero-shot sparse attention 失败的原因**：attention operator 与预训练权重不匹配，而非 prior 本身的问题。

**验证策略**：
1. Phase 1: 证明 prior-guided softmax 是"可控正则化器"
2. Phase 2: 通过渐进式 entmax annealing 实现稀疏化

---

## Phase 1: Prior-Softmax 适配 (运行中)

### 实验组

| Group | 配置 | 训练参数 | 预期结果 |
|-------|------|----------|----------|
| A | Baseline (vanilla softmax) | 全参数微调 3k steps | PPL ≈ 26 (baseline) |
| B1 | Prior-Softmax, λ=0.01 | 全参数微调 3k steps | PPL ≤ 27.3 (+5%) |
| B2 | Prior-Softmax, λ=0.05 | 全参数微调 3k steps | PPL ≤ 27.3 (+5%) |
| B3 | Prior-Softmax, λ=0.10 | 全参数微调 3k steps | PPL ≤ 27.3 (+5%) |
| C1 | Prior-Softmax + LoRA, λ=0.01 | LoRA(r=16) 3k steps | PPL ≤ 27.3 (+5%) |
| C2 | Prior-Softmax + LoRA, λ=0.05 | LoRA(r=16) 3k steps | PPL ≤ 27.3 (+5%) |
| C3 | Prior-Softmax + LoRA, λ=0.10 | LoRA(r=16) 3k steps | PPL ≤ 27.3 (+5%) |

### 停止条件

- PPL > baseline × 5.0 (爆炸检测)
- 2k steps 无改善 (早停)
- Entropy < 0.1 (坍缩检测)

### 验收标准

```
若 ∃ λ ≤ 0.1 使得 PPL ≤ baseline × 1.05:
    → Prior 是"可控正则化器"
否则:
    → Prior 方向需调整
```

---

## Phase 2: Entmax Annealing 稀疏化 (待运行)

### 核心创新

不使用直接 sparsemax，改用渐进式 entmax(α)：
- α=1.0: softmax (dense)
- α=1.5: 中等稀疏
- α=2.0: sparsemax (sparse)

### 训练策略

| Phase | Steps | α | Sparsity Target |
|-------|-------|---|-----------------|
| 1 | 0-1k | 1.0 | 0% (baseline) |
| 2 | 1k-2k | 1.2 | ~30% |
| 3 | 2k-3k | 1.4 | ~50% |
| 4 | 3k-4k | 1.6 | ~65% |
| 5 | 4k-5k | 1.8 | ~75% |

### 蒸馏稳定项

```
L = CE(sparse_output) + β × KL(p_softmax || p_entmax)
```

测试 β ∈ {0.1, 0.5}

### 只训练参数

- Attention LoRA (q/k/v/o_proj)
- 不训练全部参数

### 验收标准

```
若 ∃ phase 使得:
    sparsity ≥ 0.70
    AND
    PPL ≤ baseline × 1.05
→ Pareto sweet spot 存在
```

---

## 文件结构

```
neurips_strict/
├── phase1_prior_softmax/
│   ├── experiment.py      # Phase 1 主实验
│   ├── experiment.log     # 运行日志
│   └── run_experiment.sh  # 启动脚本
├── phase2_entmax_annealing/
│   └── experiment.py      # Phase 2 主实验
├── results/
│   ├── phase1_summary_*.csv   # Phase 1 结果
│   ├── phase2_summary_*.csv   # Phase 2 结果
│   └── *_history_*.json       # 详细历史
├── figures/
│   ├── phase1_ppl_curves.png
│   ├── phase2_alpha_vs_ppl.png
│   └── phase2_pareto.png
├── monitor.sh             # 监控脚本
└── README.md              # 本文档
```

---

## 监控命令

```bash
# 查看实验进度
cd /Users/misaya.yanghejazfs.com.au/neurIPS-2026/hybrid-rope/experiments/neurips_strict
./monitor.sh

# 实时查看日志
tail -f phase1_prior_softmax/experiment.log

# 查看已保存结果
ls -la results/
```

---

## 论文结论模板 (预期)

1. **Zero-shot sparse attention fails catastrophically** (+950% PPL on GPT-2)
2. **Prior-guided softmax is controllable** under light adaptation
3. **Direct sparsemax is incompatible** with pretrained weights
4. **Gradual annealing (entmax) enables sparse attention** with minimal PPL degradation
5. **Pareto sweet spot exists** at ~70% sparsity with ≤5% PPL increase
6. **Distillation stabilization is necessary** for sparse attention training
7. **Lightweight LoRA adaptation is sufficient** for attention pattern adjustment

---

## 当前状态

- [x] Phase 1 代码完成
- [x] Phase 2 代码完成
- [ ] Phase 1 运行中 (预计 3-4 小时)
- [ ] Phase 2 待运行
- [ ] 结果可视化待生成
- [ ] 论文结论待撰写
