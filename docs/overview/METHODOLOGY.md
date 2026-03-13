# Methodology & Fair Evaluation Protocol

> 最后更新: 2026-03-13
> 供学术发表与审稿复现参考的底层机制说明。

## 1. Core Mathematical Framework

在每个注意力头 (head_dim=64) 中，有 K = head_dim/2 = 32 个旋转频率 ω_k。标准 RoPE (geometric) 使用:

```
ω_k = base^(-2k/d), k = 0, ..., K-1
```

**EVQ-Cosh** 通过变分方法推导出闭式频率重分配:

```
φ_k(τ) = 1 - (1/τ) × arcsinh((1 - u_k) × sinh(τ))
ω_k(τ) = base^(-φ_k(τ))

其中 u_k = (k + 0.5) / K (midpoint quantization)
```

τ 是唯一控制参数，τ=0 时退化为 geometric RoPE (Theorem 2)。最优 τ* = d_head / √L_train。

### 1.1 参考实现

规范版本: `scripts/core_text_phases/run_evq_sweep.py` 第 141-157 行
RoPE 库: `scripts/lib/rope/schedules.py`

## 2. Fair Comparison Protocols

### 2.1 频率注入方式

**强制要求**: 所有 PE 方法的比较必须通过 `inv_freq` buffer 替换实现:

```python
model.rotary_emb.inv_freq.copy_(target_inv_freq)
```

禁止使用 HuggingFace `rope_scaling` API 或 monkey patch，避免底层 kernel (Flash Attention 等) 差异导致不公平比较。

### 2.2 YaRN Extension Protocol

本项目使用 **Progressive YaRN** (per-channel smoothstep ramp)，不使用 NTK-aware YaRN (uniform scaling)。

Progressive YaRN 保护高频通道、仅缩放低频通道，与 EVQ 的频率结构兼容。NTK-aware 对所有通道施加相同 factor，会破坏 EVQ 精心设计的频率分布。

参考实现: `scripts/core_text_phases/phase14c_multiscale_evq_yarn.py`

### 2.3 Evaluation Protocol

**PPL 计算:**
- 数据集: FineWeb-Edu validation split (streaming)
- Tokenizer: `EleutherAI/gpt-neox-20b`, `add_special_tokens=False`
- 滑动窗口: 连续 chunk (Chunk 0: [0:L], Chunk 1: [L:2L], ...)
- EVAL_CHUNKS: 8-10 (配置中指定)

**Passkey Retrieval:**
- 5-digit passkey embedded in random text
- Depths: [0.1, 0.2, 0.5, 0.8, 0.9]
- Lengths: [2K, 4K, 8K, 12K, 16K]
- Trials: 10 per (depth, length) pair
- Metric: Exact match accuracy

**Gold NLL (Downstream):**
- 数据集: QuALITY (n=2086)
- 计算: 仅对 gold answer token 的 NLL
- 用于替代准确率 (454M 处于容量地板)

### 2.4 Seed Robustness

核心比较指标必须基于至少 3 个 seed 的 mean ± std。单 seed 结果标注为 supporting evidence。

Seeds: 42, 123, 7 (标准三件套)

## 3. Do-Not-Cite List

| 实验 | 废弃原因 | 替代 |
|------|---------|------|
| 2026-02-13 8B LoRA (monkey patch) | 不符合公平协议 (rope_scaling vs patch) | `EXP_8B_FAIR_LORA` (2026-02-22) |
| 50M Base=300K Sigmoid | Base 选型错误 | 作为 Limitations 反例 |
| Zero-shot inv_freq 直接替换 | 无训练适配，attention 崩塌 | 作为 Limitations |
| `phase21b_quality_eval.py` | 使用 NTK-aware YaRN (bug) | 用 `phase21b_quality_eval_clean.py` |
