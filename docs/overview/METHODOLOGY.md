# 方法论与公平评测协议 (Methodology & Fair Evaluation Protocol)

> 最后更新：2026-02-22
> 供学术发表与审稿复现参考的底层机制说明。

## 1. 核心数学重塑 (Core Mathematical Reformulation)

在每个注意力头（典型 `head_dim=64`）中，我们拥有 $K = \text{head\_dim} / 2 = 32$ 个维度的旋转频率 $\omega_k$。距离为 $d$ 时的相位差由 $\Delta\phi_k(d) = \omega_k \cdot d$ 决定。这确保了严谨的平移不变性 (Translation Invariance, TI)。

我们的方法不同于通过标量拉伸 $\theta$，而是通过构造**形状映射**来重分配频带资源的策略。

### 1.1 频率分布函数实现

所有代码逻辑必须基于以下不可突变的基础函数进行计算：

```python
# 基线: 数学原生的几何级数 (Standard RoPE)
def geometric_freq(K, theta):
    k = torch.arange(K, dtype=torch.float32)
    return 1.0 / (theta ** (2 * k / (2 * K)))

# 方案: 高频绝对锚定+中低频平滑凸组合 (Anchored Hybrid)
# 本项目提倡以 "rigid core" 保护高频段以维持局部结构完整性
def hybrid_freq(freq_a, freq_b, alpha):
    return (1 - alpha) * freq_a + alpha * freq_b
```

## 2. 严密的公平比较原则 (Strict Fairness Protocols)

当我们评价现有方法（PI, YaRN）与形状调节方法时的底层规范：

### 2.1 频率注入的唯一出口：`inv_freq` Buffer Mutation
**[Critical Requirement]** 过去文献（和早期的 HF 实现）通过定义前向 `rope_scaling` 甚至猴子补丁 (monkey patch) 修改注意力传播路径，这在框架计算层面引发了不平等的计算误差。
为此，项目中强制移除所有的挂载式代码，所有策略的对比必须体现为 `model.model.rotary_emb.inv_freq.copy_(target_omega)` 的静态显存替换。
- `model.config.rope_scaling = None`

### 2.2 评估协议：数据确定性 (Deterministic Evaluation)
- **Token 流切割**: 使用固定验证集上的滑动窗口计算（即 Chunk 0 为 `val[0:L]`, Chunk 1 为 `val[L:2L]`...）。在同等 Token 容量下排除了 PPL 计算的偶然波动。

## 3. 指标定义口径

- **Perplexity (PPL)**: 利用验证集的 Causal Language Modeling 极大似然底数，主要观测长片段下的稳定度（如 `PPL@16K`, `PPL@32K`）。
- **Needle In A Haystack (NIAH)**: 必须采取 10 级背景长度和 11-级深度的综合热力图探针测试，评价文档找回。
- **LongBench**: 综合代码补全、长篇阅读理解和多文档检索的 F1 / ROUGE-L 指标。
