# L_eff 假设：τ* 在大 L 下的失效机制与修正方案

> **Created**: 2026-04-20
> **Author**: internal working doc
> **Status**: 理论假设 + 待验证实验协议
> **Related**: `21_weekend_sweep_and_phase16_joint_analysis_0420.md` (TinyStories 50M 异常)

---

## TL;DR

论文公式 **τ\* = d_head/√L_train** 在 L_train ≥ 16K 的现代训练设定下会让 EVQ 退化到 Geo（特别是 MLA 这类压缩通道架构在 L=32K 时正好撞 τ_floor）。这不是数值近似误差，是变分导出中 "diffuse softmax baseline p₀=1/L" 假设的**硬性后果**。

**诊断**：真实 LLM 的 attention 是稀疏的，每个 query 只集中在 O(1K–4K) 个关键 key 上。有效 softmax baseline 应是 p₀ = 1/**L\***，其中 L\* 是 attention 的自然集中窗口，**由架构决定，与 L_train 无关上界**。

**修正公式**：

$$\tau^* = \frac{d_{\text{head}}}{\sqrt{\min(L_{\text{train}},\; L^*)}}$$

- L_train ≤ L\*：退化到论文原公式 τ* = d/√L_train（覆盖 27 primary configs）
- L_train > L\*：饱和到 τ* = d/√L\*（大 L 下 EVQ 保持有效）
- 欠训情形：L_eff < L_train 也可套同一公式，解释 TinyStories 50M 的 ν ≥ 1.7

**待验实验**（<1h GPU）：对已训 checkpoint 跑 per-head attention 熵，测 L_eff = exp(H)，看是否在 1K–4K 区间饱和。饱和则假设成立。

---

## 1. 问题：τ* 在大 L 下的预测值

d_head=64, 128 和 MLA d_rope=32 在各 L 下的 τ*（论文公式）：

| L_train | d=64 | d=128 | d_rope=32 (MLA) |
|---|---|---|---|
| 256 | 4.00 | 8.00 | 2.00 |
| 1K | 2.00 | 4.00 | 1.00 |
| 4K | 1.00 | 2.00 | 0.50 |
| 8K | 0.71 | 1.41 | **0.35** ← 论文 MLA 实验 |
| 16K | 0.50 | 1.00 | 0.25 |
| 32K | 0.35 | 0.71 | **0.18** ← τ_floor |
| 64K | 0.25 | 0.50 | 0.13 |
| 128K | 0.18 | 0.35 | 0.09 |

τ_floor（Proposition 2）≈ 0.18·τ*(L)·K=32 在 d_rope=32 对应 τ ≈ 0.18。所以 MLA 在 L=32K 训练时 EVQ **数学上**就变成 Geo 了。

MHA d_head=128 在 L=32K 还有 τ=0.71 可用，但 d=64 的模型在 L=32K 时 τ=0.35 也已经接近 marginal 区。

---

## 2. 诊断：Q₁ 饱和不是主因

变分公式 τ*² = 45·λ·Q₁(L,b)·d²/L 里：

$$Q_1(L,b) = \int_0^1 \eta(\phi)\, q(Lb^{-\phi})\, d\phi,\quad \eta(\phi)=\tfrac{(1-\phi)^2}{2}-\tfrac{1}{6}$$

Q₁ 随 L 下降（粗算 b=500K）：

| L | Q₁ | 45·Q₁ |
|---|---|---|
| 256 | 0.032 | 1.44 |
| 8K | 0.024 | 1.08 |
| 32K | 0.018 | 0.81 |
| 128K | 0.013 | 0.59 |

Q₁ 只下降约 60%（256→128K），但 d²/L 下降约 500×。**τ* 的塌缩主要来自 1/√L，不是 Q₁ 的数值变化**。

这说明 **L^{-1/2} scaling 是硬性结构性结论**，不是数值微调能救的。要让 τ* 在大 L 下不塌，必须修改 L 进入公式的方式本身。

---

## 3. 物理修正：L_eff 假设

### 3.1 论文推导的基线假设

Proposition `softmax-transport` 的导出关键一步是：softmax 在 diffuse baseline **p₀ = 1/L** 处的 Jacobian 本征值为 1/L。这直接给出 τ*² ∝ d²/L。

### 3.2 实际 attention 是稀疏的

经验上现代 LLM 的 attention 不是 diffuse 的。多个独立观测一致：
- Longformer、StreamingLLM 等稀疏模式工作显示每个 query 的有效 key 集中在几百到几千的范围
- Attention sink 现象：大量概率集中在 sequence 开头少数几个 token
- 长上下文模型的 needle-in-a-haystack 测试：只有当目标位置被具体任务"激活"时才会有权重

所以**真实基线 p₀ ≠ 1/L_train**，而是 **p₀ ≈ 1/L\***，其中 L\* 是 attention 的自然有效集中窗口。

### 3.3 同一 proposition，换基线

把 p₀ = 1/L\* 代进同一个推导：softmax Jacobian 本征值 = 1/L\*，utility integral 变成

$$U_{\text{eff}}(\tau, L_{\text{train}}, L^*) = \frac{d}{L^*}\int q(L^* b^{-\phi}) \rho_\tau(\phi)\, d\phi$$

最小化 F = S_χ² − λ·U_eff 给出：

$$\boxed{\;\tau^{*2} = 45\cdot\lambda\cdot Q_1(L^*, b)\cdot\frac{d_{\text{head}}^2}{L^*}\;}$$

**L_train 完全不进入 τ* 的表达式**。L\* 才是真正的标度。

### 3.4 L_train ≤ L\* 时的退化

当 L_train ≤ L\* 时，attention 没有足够 key 来"施展" diffuse-attention 的统计平均，有效基线降为 L_train 本身。这时 L\* 应被 L_train 替换：

$$L_{\text{eff}} = \min(L_{\text{train}}, L^*)$$

给出**统一公式**：

$$\tau^* = \frac{d_{\text{head}}}{\sqrt{L_{\text{eff}}}},\quad L_{\text{eff}} = \min(L_{\text{train}}, L^*)$$

---

## 4. 一致性检验：能否同时覆盖所有实验？

| 实验 | L_train | 预期 L_eff | 预测 τ* | 与观测一致？ |
|---|---|---|---|---|
| 27 primary configs | 32–8K | L_train (全部 ≤ L\*) | d/√L_train (不变) | ✓ |
| Phase 16 WikiText 125M | ≤ 8K | L_train | d/√L_train | ✓ |
| Collision oracle | 代理目标 | 纯 K_app，不涉及 | ν₀=1.082 | ✓ |
| **MLA 432M L=8K** | 8K | L_train (=L\* 或略小) | d_rope/√8K=0.35 | ✓ (-31.1% PPL 符合 τ=0.35 有效区) |
| Video DiT bidir | 中等 | L_train × 1/2 (双向) | 几何修正 1/√2 | ✓ (残差 0.75 另解) |
| **TinyStories 50M undertrained** | 256–1024 | **< L_train**（欠训 L_eff 小） | > d/√L_train | ✓ (ν ≥ 1.7 对应 L_eff ≈ 0.35 L_train) |
| **MLA L=32K (预测)** | 32K | L\* (≤ 8K) | d_rope/√L\* ≈ 0.5 (若 L\*=4K) | **待验证** |

**关键点**：L_eff 假设**同时**解释了两个曾经看起来矛盾的现象：
- TinyStories 50M 欠训 → L_eff 小 → τ 需要更大（周末观察到的 ν ≥ 1.7）
- 大 L 训练 → L_eff 被 L\* 封顶 → τ 不再继续变小（解决用户提出的大 L 失效问题）

两者是**同一机制**：**L_eff 是 attention 实际能有效利用的 key 数，上有 L\* 封顶，下会因欠训而收缩**。

---

## 5. 实验协议：如何测 L\*

### 5.1 观测量：attention 熵

对一个训练好的 checkpoint，用长上下文输入（≥ 8K tokens）前向一次，对每层每 head 计算：

$$L_{\text{eff}}^{(l,h)} = \exp\left(H^{(l,h)}\right),\quad H^{(l,h)} = -\sum_{i,j} p^{(l,h)}_{ij}\log p^{(l,h)}_{ij}$$

其中 p^{(l,h)}_{ij} 是 layer l, head h, query i 对 key j 的 attention 权重。

L_eff^{(l,h)} 是该 head 在该 query 上"有效参与"的 key 数量（perplexity-based effective support size）。

### 5.2 汇总量

- **L\* 候选 1**：L_eff 在所有 (l,h) 上的中位数
- **L\* 候选 2**：L_eff 的 75 百分位（上界）
- **L\* 候选 3**：按层聚合后最大层的 L_eff

论文意义下的 L\* 应该是 **L_eff 分布的上界**（因为 τ* 是被最需要 positional resolution 的 head 决定的）。

### 5.3 饱和检验

对同一 checkpoint 在多个输入长度（L=1K, 2K, 4K, 8K, 16K, 32K）下分别测 L_eff(L)：

- 若 L_eff(L) 在 L 增大时**饱和**到某个 L\*，假设成立
- 若 L_eff(L) 继续跟随 L 增大（L_eff ≈ L），则 attention 不是稀疏的，L_eff 假设**失败**

### 5.4 GPU 预算

| 实验 | 模型 | 上下文 | 时间 |
|---|---|---|---|
| 快速验证 | 任意已训 750M checkpoint | 8K | <10min |
| 饱和测试 | 同上 | 1K/2K/4K/8K/16K/32K | <1h |
| 跨模型验证 | LLaMA-3-8B + DeepSeek-V2（若可访问） | 8K/32K | <3h |
| 稀有架构 | 已训 MLA 432M | 8K | <30min |

**总共 < 5h GPU**，远低于用户 20h 预算。

---

## 6. 决策树：实验结果如何解读

### 6.1 理想情形：L\* 清晰存在且 ≈ 2K–4K

- **假设验证**
- 论文 Limitations 可以加一句：τ* 应使用 min(L_train, L\*)，在 L_train 超出 attention 自然集中窗口 L\* 时饱和
- 这把用户担心的 "大 L 下 EVQ 失效" 转化为可预测的 scope statement
- Rebuttal 里对 "L=32K 时 EVQ 还有用吗？" 有完整答案

### 6.2 次优情形：L\* 存在但跨层差异大（例如前几层 L_eff≈全局，深层 L_eff≈小）

- 假设部分成立：不同 head 应该用不同 τ
- 这指向 **per-layer 或 per-head EVQ**，是论文 "future work" 里提的方向之一
- 短期可以按 "最敏感 head 决定 τ*" 操作

### 6.3 假设失败：L_eff 持续跟随 L_train，没有饱和

- attention 不是稀疏的，diffuse baseline 假设近似成立
- 这意味着 τ* 在大 L 下**真的**塌陷 → EVQ 在大 L MHA/MLA 里**确实**接近 Geo
- 论文需要更诚实的 scope 声明：EVQ 的显著收益在 L_train ≤ 8K + 压缩通道架构 + extrapolation
- 不是灾难，只是缩窄适用域

---

## 7. 实现脚本（建议新增 `results/L_eff_probe/`）

```python
# results/L_eff_probe/measure_L_eff.py
"""
Measure attention entropy → L_eff = exp(H) on a trained checkpoint.

Usage:
    python measure_L_eff.py --model_path <path> --context_lengths 1024,2048,4096,8192,16384,32768
"""
import torch
import numpy as np
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

def measure_L_eff(model, tokenizer, text, context_len, device='cuda'):
    """Returns L_eff per (layer, head) averaged over queries."""
    inputs = tokenizer(text, return_tensors='pt', max_length=context_len,
                       truncation=True).to(device)
    with torch.no_grad():
        out = model(**inputs, output_attentions=True, use_cache=False)
    # attentions: tuple of (batch, heads, seq, seq) per layer
    L_eff_per_layer = []
    for layer_attn in out.attentions:
        # layer_attn: (1, H, L, L)
        p = layer_attn.squeeze(0)  # (H, L, L)
        # entropy along key dim
        H = -(p * torch.log(p.clamp(min=1e-12))).sum(dim=-1)  # (H, L)
        # average over queries (skip first few tokens to avoid boundary)
        H_mean = H[:, 32:].mean(dim=-1)  # (H,)
        L_eff = torch.exp(H_mean)  # (H,)
        L_eff_per_layer.append(L_eff.cpu().numpy())
    return np.stack(L_eff_per_layer)  # (num_layers, num_heads)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--context_lengths', default='1024,2048,4096,8192,16384,32768')
    parser.add_argument('--text_file', default=None,
                       help='Long text file for probe; defaults to repeated Wikipedia sample')
    parser.add_argument('--output', default='L_eff_results.npz')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map='auto',
        attn_implementation='eager',  # required for output_attentions
    )
    model.eval()

    if args.text_file:
        with open(args.text_file) as f:
            text = f.read()
    else:
        # fallback: repeat enough text to fill longest context
        text = "The quick brown fox jumps over the lazy dog. " * 10000

    results = {}
    for L in map(int, args.context_lengths.split(',')):
        print(f'Context {L}...')
        L_eff = measure_L_eff(model, tokenizer, text, L)
        results[f'L_{L}'] = L_eff
        print(f'  median L_eff = {np.median(L_eff):.0f}, '
              f'p75 = {np.percentile(L_eff, 75):.0f}, '
              f'p95 = {np.percentile(L_eff, 95):.0f}')

    np.savez(args.output, **results)
    print(f'Saved to {args.output}')

if __name__ == '__main__':
    main()
```

### 7.1 结果汇总（建议脚本输出）

```
Context 1024:   median L_eff =   287, p75 =   512, p95 =   820
Context 2048:   median L_eff =   430, p75 =   890, p95 =  1650
Context 4096:   median L_eff =   580, p75 =  1420, p95 =  2800
Context 8192:   median L_eff =   720, p75 =  1920, p95 =  3400  ← 饱和？
Context 16384:  median L_eff =   750, p75 =  2010, p95 =  3450  ← 饱和
Context 32768:  median L_eff =   770, p75 =  2050, p95 =  3500  ← 饱和
```

上面是**假设成立时**预期的数字形态。若 L_eff 跟随 context 线性增长到 32K，则假设失败。

---

## 8. 对论文的影响

### 8.1 若 L\* 清晰（~2K–4K）

**正文不改**。在 Limitations 加一句（< 25 词）：

> The $\tau^*$ rule uses $L_{\mathrm{train}}$ as a proxy for effective attention range; at $L_{\mathrm{train}}$ beyond the attention's natural concentration window $L^*$, the rule should use $\min(L_{\mathrm{train}}, L^*)$.

并在 rebuttal 阶段把 L_eff 测量结果作为补充证据放 Appendix（需要一张小图）。

### 8.2 若 L\* 不存在

诚实承认：正文不改，但 Limitations 里 **Scale and evaluation** 段加一句强化现有 disclosure：

> EVQ's magnitude shrinks with $L_{\mathrm{train}}$ as $\tau^* \propto 1/\sqrt{L_{\mathrm{train}}}$; at $L_{\mathrm{train}} \geq 32$K with compressed channels (e.g., MLA with $d_{\mathrm{rope}} \leq 32$), the variational correction approaches the geometric baseline, and gains concentrate in post-hoc extrapolation via EVQ+YaRN rather than training-time allocation.

这 statement 本身也不是新 claim，只是把现有隐含 scope 写明。

### 8.3 rebuttal 弹药

无论哪种结果都会有 review 问到这个 —— 审稿人只要扫一眼 τ*=d/√L 就会算到 L=32K 的情况。**有 L_eff 测量数据在手** 总比只能回答 "future work" 强。

---

## 9. 下一步行动

**立即可做**（< 1h GPU）：
1. 选一个已训 checkpoint（建议 LLaMA-3-8B 或你手上的 750M）
2. 跑上面脚本，output 一张 `L_eff vs context_length` 曲线
3. 判断 L\* 是否存在及其量级

**若 L\* ≈ 2K–4K**：
4. 在 MLA 432M checkpoint 上再跑一次（验证跨架构一致）
5. 写 1 页 Appendix 子节 "Attention concentration window"，纳入下版论文

**若 L\* 不存在**：
4. 修 Limitations 一句话，诚实缩窄 scope
5. 把 GPU 预算转去别的验证（例如 Phase 16 某个 primary config 补 seed）

---

## 10. 附：为什么这个假设是 "最简修正"

有三个候选修正都能解释大 L 下 EVQ 失效：

| 方案 | 新参数 | 理论改动 | 可测性 |
|---|---|---|---|
| A. 改 L^{-α} 的 α (α < 1/2) | 1 | 需重新导 softmax 展开 | 难直接测 |
| B. 加 log(L) 修正项 | 0–1 | 需重新导中间-τ 展开 | 难直接测 |
| C. L_eff = min(L_train, L\*) | **0（L\* 可测**） | **同一 proposition 换基线** | **直接测 attention 熵** |

C 是唯一一个**不改导出、不引新参数、可直接测量**的方案。所以从 Occam's razor 上就应该先测 C。

---

*若 GPU 可用，请在 `results/L_eff_probe/` 下运行上面脚本，把结果回传后我整合到 Limitations 或 Appendix。*
