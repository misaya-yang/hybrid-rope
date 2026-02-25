# Prior-guided Variational Sparse Attention 实验

验证在 M4 Max 上，将距离先验注入 LLM 注意力机制的可行性与收益。

## 实验目的

1. **证明结构性稀疏**: 距离先验 D(Δ) 注入注意力权重可以产生大量精确 0
2. **证明距离分布**: 稀疏的距离分布符合理论预期（距离越远越容易被截断）
3. **证明 PPL 不崩**: 语言建模 PPL ≤ baseline softmax 的 5% 相对劣化
4. **证明远距离保留**: Needle-in-a-Haystack 任务中远距离关键 token 仍可被保留

## 三种注意力变体

| 变体 | 名称 | 公式 |
|-----|------|------|
| **A** | Baseline Softmax | `softmax(QK^T / √d)` |
| **B** | Prior-Biased Softmax | `softmax(QK^T / √d + λ log D(Δ))` |
| **C** | Prior-Guided Sparse | `sparsemax((QK^T / √d + λ log D(Δ)) / γ)` |

距离先验: `D(Δ) ∝ (Δ + δ₀)^(-α)`

## 快速开始

### 1. 环境准备

```bash
# 激活conda环境
conda activate aidemo

# 安装依赖
pip install torch torchvision torchaudio transformers datasets entmax numpy matplotlib

# 验证安装
python -c "from entmax import sparsemax; import torch; print(sparsemax(torch.tensor([1.,2.,0.5])))"
# 应输出: tensor([0., 1., 0.])
```

### 2. 运行完整实验

```bash
# 默认参数运行（gpt2, α=1.0, λ=2.0, γ=1.0）
python scripts/run_variational_attention_experiment.py

# 自定义参数
python scripts/run_variational_attention_experiment.py \
    --model-name gpt2 \
    --alpha 1.5 \
    --lam 3.0 \
    --gamma 0.5 \
    --n-tokens 50000 \
    --batch-size 2

# 使用 entmax1.5 替代 sparsemax
python scripts/run_variational_attention_experiment.py \
    --use-entmax15 \
    --gamma 1.0
```

### 3. 输出目录结构

```
outputs/var_attn/{timestamp}/gpt2/
├── env.json              # 环境配置与参数
├── ppl_results.json      # PPL 结果
├── needle_results.json   # Needle 测试结果
├── final_report.json     # 完整报告
├── ppl_comparison.png    # PPL 对比图
├── sparsity_comparison.png  # 稀疏性分析图
└── needle_attention_mass.png  # Needle 注意力质量图
```

## 参数说明

### 距离先验参数

| 参数 | 说明 | 建议值 |
|-----|------|-------|
| `--alpha` | 幂律指数 α | 0.5, 1.0, 1.5 |
| `--delta0` | 距离偏移 δ₀ | 1.0 (固定) |
| `--lam` | 先验权重 λ | 1.0 - 5.0 |
| `--gamma` | sparsemax 温度 γ | 0.5 - 2.0 |

### 实验控制参数

| 参数 | 说明 | 默认值 |
|-----|------|-------|
| `--model-name` | 模型名称 | gpt2 |
| `--n-tokens` | 评估token数 | 200000 |
| `--seq-len` | 序列长度 | 1024 |
| `--batch-size` | 批次大小 | 4 |
| `--max-batches` | 最大批次（调试用） | None |
| `--seed` | 随机种子 | 42 |

### Needle测试参数

| 参数 | 说明 | 默认值 |
|-----|------|-------|
| `--needle-seq-len` | Needle测试序列长度 | 4096 |
| `--needle-positions` | Needle位置列表 | [100,500,1000,2000,3000] |

## 结果解读

### 预期成功标准

1. **稀疏性**: C组应达到 >10% 的精确稀疏率
2. **PPL**: C组相对A组劣化 ≤ 5%
3. **Needle**: 远距离 needle 仍保持非零注意力质量

### 示例输出

```
FINAL REPORT
============================================================

1. Perplexity Results:
   Variant A: PPL = 25.43 (+0.00%)
   Variant B: PPL = 24.89 (-2.12%)
   Variant C: PPL = 26.12 (+2.71%)

2. Sparsity Analysis:
   Variant C:
     - Average sparsity: 35.20%
     - Average nonzero weights: 664.3

3. Needle Test:
   Variant A: 6/6 positions retain needle
   Variant B: 6/6 positions retain needle
   Variant C: 6/6 positions retain needle

4. Conclusion:
   ✅ PASS: Prior-guided sparse attention is VIABLE.
      - PPL degradation: 2.71% (threshold: 5%)
      - Achieved sparsity: 35.20%
      - Structural sparsity with distance prior works without breaking LM performance.
```

## 调参指南

### 增加稀疏性
- 增大 `--lam` (先验权重)
- 减小 `--gamma` (sparsemax 温度)
- 增大 `--alpha` (幂律指数)

### 减少PPL劣化
- 减小 `--lam`
- 增大 `--gamma`
- 使用 `--use-entmax15` (比 sparsemax 更平滑)

### 平衡策略
建议网格搜索: `--lam` ∈ {1,2,3}, `--gamma` ∈ {0.5,1,2}

## 代码结构

```
scripts/
├── run_variational_attention_experiment.py  # 主实验脚本
├── needle_data.py                           # Needle数据生成工具
└── README.md                                # 本文档
```

### 核心类说明

| 类/函数 | 用途 |
|--------|------|
| `DistancePrior` | 计算距离先验矩阵 D(Δ) |
| `PatchedGPT2Attention` | Monkey patch GPT2 attention |
| `baseline_softmax_attention` | A组: 原始softmax |
| `prior_biased_softmax_attention` | B组: 先验偏置softmax |
| `prior_guided_sparse_attention` | C组: 稀疏注意力 |
| `NeedleInHaystackGenerator` | 合成needle测试数据 |

## 技术细节

### 为什么用 Monkey Patch?

为了保持与原始模型的最大兼容性，我们选择 monkey patch 而非重写模型:
1. 保留预训练权重加载
2. 保留所有其他模型组件不变
3. 只修改 attention weights 计算处的 softmax

### 距离先验的因果性处理

```python
# 距离矩阵: Δ[i,j] = i - j (query位置 - key位置)
# 因果mask: 只考虑 j <= i 的位置
# 未来位置: log_prior = -inf
```

### Sparsemax 算法

使用 `entmax` 包提供的 `sparsemax` 函数，该实现:
- 保证输出在 simplex 上 (∈[0,1], sum=1)
- 产生精确零值
- 支持反向传播

## 故障排除

### MPS 内存不足
```bash
# 减小批次大小
--batch-size 1
# 减小评估token数
--n-tokens 50000
```

### PPL 劣化严重
```bash
# 减小先验权重
--lam 1.0
# 增大温度
--gamma 2.0
# 使用entmax15
--use-entmax15
```

### 稀疏性不足
```bash
# 增大先验权重
--lam 3.0
# 减小温度
--gamma 0.5
# 增大幂律指数
--alpha 1.5
```

## 引用

```bibtex
@article{peters2019sparse,
  title={Sparse sequence-to-sequence models},
  author={Peters, Ben and Niculae, Vlad and Martins, Andr{\'e} FT},
  journal={arXiv preprint arXiv:1905.05702},
  year={2019}
}

@inproceedings{martins2016softmax,
  title={From softmax to sparsemax: A sparse model of attention and multi-label classification},
  author={Martins, Andr{\'e} FT and Astudillo, Ram{\'o}n Fernandez},
  booktitle={International conference on machine learning},
  pages={1614--1623},
  year={2016}
}
```

## 作者

研究工程实现 for NeurIPS 2026
