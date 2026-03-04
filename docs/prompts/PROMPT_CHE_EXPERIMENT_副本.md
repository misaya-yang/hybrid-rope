# Prompt: CHE (Chomsky Hierarchy Evaluation) 实验执行

> 用途：给 Claude/Gemini 执行 CHE benchmark 集成与实验

---

## 任务

你需要在我们的 RoPE 频率优化研究中集成并运行 **CHE (Chomsky Hierarchy Evaluation)** benchmark，对比 Geometric RoPE vs EVQ-Cosh 频率分配的 length generalization 能力。

## 背景

我们研究 RoPE 位置编码的频率分配优化。核心方法 EVQ-Cosh 用单参数 τ 控制频率分配：
```
φ_k(τ) = 1 - (1/τ) · arcsinh((1 - u_k) · sinh(τ))
```
其中 u_k = (2k-1)/(d_head) 是均匀网格。τ=0 退化为 Geometric (标准 RoPE)。

**CHE** 来自 Delétang et al. (ICLR 2023)，包含 15 个序列任务，按 Chomsky 层级分类。DAPE (NeurIPS 2024) 用 CHE 评测 PE 方法，是 PE 论文的标准 benchmark。

## 关键参考

- DAPE 代码: https://github.com/chuanyang-Zheng/DAPE
- CHE 原始代码: https://github.com/google-deepmind/neural_networks_chomsky_hierarchy
- 我们的训练代码: `/path/to/train.py`
- EVQ 实现: 参考 `docs/paperdraft/CORE_THEORY.md` §3

## DAPE 的 CHE 实验配置

| 项 | 值 |
|----|-----|
| Model | 5 层, 8 heads, d_model=256 |
| 训练长度 | 40 |
| 训练步数 | 200K |
| 评估长度 | 41 → 500 |
| 指标 | Accuracy |
| 随机基线 | 50%（多数任务）, 20%（Modular Arithmetic, Cycle Navigation, Bucket Sort, Solve Equation） |

## 需要执行的步骤

### Step 1: 环境与代码准备

1. Clone DAPE repo, 找到 CHE 相关代码（data loader, training script, eval script）
2. 理解 DAPE 如何加载 15 个 CHE 任务
3. 确认 PyTorch 实现可用（DAPE repo 应该有）
4. 确认每个任务的输入/输出格式

### Step 2: 集成 EVQ 频率

在 DAPE 的 PE 模块中添加 EVQ 选项：
```python
import torch
import math

def evq_cosh_frequencies(d_head: int, tau: float, base: float = 10000.0) -> torch.Tensor:
    """EVQ-Cosh frequency allocation."""
    if abs(tau) < 1e-8:
        # τ→0: geometric (standard RoPE)
        k = torch.arange(0, d_head, 2, dtype=torch.float32)
        return 1.0 / (base ** (k / d_head))

    d = d_head
    u = torch.linspace(1/(d), (d-1)/d, d//2)  # uniform grid
    phi = 1.0 - (1.0/tau) * torch.arcsinh((1.0 - u) * math.sinh(tau))

    # Map φ ∈ [0,1] to frequencies via base
    freqs = 1.0 / (base ** phi)
    return freqs
```

### Step 3: 训练矩阵

**Phase 1 — CHE-Small (与 DAPE 对标)**：

```
Models: CHE-Small (5L/8H/d256, d_head=32)
Methods: {Geo, EVQ(τ=5.0)}
Seeds: {42, 123, 7}
Tasks: all 15 CHE tasks
Train: 200K steps, L_train=40
Eval: L_eval ∈ {50, 100, 200, 300, 500}

Total runs: 2 methods × 3 seeds × 15 tasks = 90 runs
```

**Phase 2 — Scale-up**：

```
Models: 125M (12L/12H/d768), 350M (24L/16H/d1024)
Methods: {Geo, EVQ(τ=10.0)}
Seeds: {42, 123, 7}
Tasks: top-5 most discriminative tasks from Phase 1
Train: 200K steps, L_train=40
Eval: same as Phase 1

Total runs: 2 models × 2 methods × 3 seeds × 5 tasks = 60 runs
```

### Step 4: 评估与输出

对每个 (task, method, seed, eval_length) 输出 accuracy。
汇总为以下表格格式：

```
| Task | Level | Geo (3-seed) | EVQ (3-seed) | Delta |
|------|-------|-------------|-------------|-------|
| Even Pairs | R | xx.x ± y.y | xx.x ± y.y | +z.z |
| Parity Check ‡ | R | ... | ... | ... |
| ... | ... | ... | ... | ... |
```

另外输出 length generalization curve：
```
| Task | Method | Acc@50 | Acc@100 | Acc@200 | Acc@300 | Acc@500 |
```

### Step 5: 关键分析

1. **Permutation-variant vs invariant**：EVQ 应该在 permutation-variant 任务上更有优势（PE 对这些任务重要）
2. **长程衰减率**：计算 Acc@500 / Acc@50 的衰减比，EVQ 是否衰减更慢？
3. **Chomsky 层级分组**：Regular avg / DCF avg / CS avg，哪个层级 EVQ 优势最大？

## 重要注意事项

1. **τ 值选择**：L_train=40 是极短序列，τ*=d_head/√40 可能过大。如果 Phase 1 效果差，尝试 τ ∈ {1, 2, 3, 5} 的小范围 sweep
2. **不要用我们的 350M/125M 模型配置直接跑**——先用 DAPE 的 5L/8H/d256 配置保证可比性
3. **每个任务独立训练**——CHE 的每个任务都是独立的分类/预测问题，不是共享模型
4. **先跑一个 task pilot**（如 Even Pairs）确认 pipeline 通了再批量跑
