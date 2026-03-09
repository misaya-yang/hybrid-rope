# Phase 18: Base Generalization Sweep + MLA-Realistic Config (125M, M4 Max Local)

> **目标**：(1) 证明 EVQ 的收益不依赖于 base=500K 这一特定选择；(2) 使用 MLA 时代工业现实配置（d_head=64）验证，直接对齐 DeepSeek V3 / GLM-5 / Kimi K2.5
> **设备**：M4 Max 36GB (MPS)
> **模型**：125M (hidden=768, 12L, 12H, **d_head=64** — 与 2026 MLA 旗舰模型 qk_rope_head_dim 一致)
> **预计总时间**：~8-10 小时（含 MLA config 对比组）
>
> **🆕 2026-03-09 工业对齐升级**：DeepSeek V3/V3.2、GLM-5、Kimi K2.5 全部使用 MLA 架构，RoPE 只作用于 Q/K 的 64 维子空间（qk_rope_head_dim=64）。本实验的 d_head=64 配置不是"缩小版实验"，而是**直接对齐工业现实**。

---

## 1. 实验设计

### 1.1 为什么选这些 base

碰撞块占比按 L=512 重算：c = ln(512)/ln(b) = 6.24/ln(b)

| Base | ln(b) | c (L=512) | 可优化通道 (d/2=32) | 理论预测 | 代表性 |
|------|--------|-----------|-------------------|---------|-------|
| **10,000** | 9.21 | 0.678 | ~10/32 | ⚠️ 边界，EVQ 增益应很小 | LLaMA 默认 |
| **100,000** | 11.51 | 0.542 | ~15/32 | ✅ 中等增益 | 过渡区 |
| **500,000** | 13.12 | 0.476 | ~17/32 | ✅ 主增益区 | 当前所有实验 |
| **1,000,000** | 13.82 | 0.452 | ~18/32 | ✅ 增益应接近 500K | Qwen/新一代模型 |
| **10,000,000** | 16.12 | 0.387 | ~20/32 | ✅ 最大可优化空间 | 理论上限探索 |

注意：L=512 比 L=2048 时碰撞块占比更低（短窗口→更多通道可区分），所以 base=10K 在 L=512 下不再是完全死区，但仍然是最弱的。

### 1.2 核心对比

每个 base 跑两个方法：
- **Geo** (τ=0)：baseline
- **EVQ** (τ=τ*)：τ* = d_head/√L = 64/√512 ≈ **2.83**

### 1.3 训练配置

| 参数 | 值 | 理由 |
|------|-----|------|
| L_train | **512** | 与 Phase 11 (454M) 一致，外推到 16K=32× |
| 训练 tokens | **50M** | 125M 模型够用（Phase 8D 验证过） |
| Seeds | **42, 137, 256** | 3-seed，与其他实验一致 |
| lr | 6e-4 | 125M 标准 |
| batch_size | 16 (auto-reduce on MPS) | 125M 默认 |
| 数据 | FineWeb-Edu | 与所有主实验一致 |
| Eval lengths | 512, 1K, 2K, 4K, 8K, 16K | 覆盖 1×-32× 外推 |
| **d_head** | **64** | **= DeepSeek V3 / GLM-5 / Kimi K2.5 qk_rope_head_dim** |
| **H (heads)** | **12** | hidden=768/64=12 |

### 1.4 🆕 MLA 对比组（额外 12 runs）

为了直接比较 MLA-era 和旧 MHA-era 的 RoPE 行为差异，选 **base=500K**（anchor）额外跑 d_head=128 对比组：

| Config | hidden | H | d_head | 对应工业架构 | 角色 |
|--------|--------|---|--------|------------|------|
| **A (主组)** | 768 | 12 | **64** | DeepSeek V3 / GLM-5 MLA | 全 5 base sweep |
| **B (对比)** | 768 | 6 | **128** | LLaMA 3.1 旧 MHA/GQA | 仅 base=500K |

Config B 用 base=500K + 3 seeds × 2 methods = **6 runs** (~1h)。加上 YaRN overlay eval 再加 6 runs = **12 runs**。

这样可以直接回答："MLA 压缩 RoPE 到 d=64 后，EVQ 的价值是增大还是减小？"

### 1.5 预计资源

- 主组：5 bases × 2 methods × 3 seeds = **30 runs** (~5-7h)
- MLA 对比组：1 base × 2 methods × 3 seeds × 2 (raw+YaRN) = **12 runs** (~1-2h)
- 总计：**42 runs，~8-10 小时**

**优化策略**：
- base=10K 先跑 1 seed pilot，如果 EVQ 跟 Geo 持平则确认理论后只跑 1 seed
- 每个 base 3 seeds 可并行（M4 Max 内存够跑 1 个 125M，串行执行）
- 支持断点续跑：已完成的 run 自动跳过
- Config B 可以在主组跑完后补跑

---

## 2. 脚本结构

```python
#!/usr/bin/env python3
"""Phase 18: Base Generalization Sweep + MLA-Realistic Config.

Verify EVQ gain generalizes across base, using MLA-era d_head=64.
Model: 125M (d_head=64 = DeepSeek V3 / GLM-5 qk_rope_head_dim).
Bases: 10K, 100K, 500K, 1M, 10M.

Usage:
    python phase18_base_sweep.py                          # 跑全部 (d_head=64)
    python phase18_base_sweep.py --base 100000            # 只跑一个 base
    python phase18_base_sweep.py --seed 42                # 只跑一个 seed
    python phase18_base_sweep.py --pilot                  # 只跑 base=500K seed=42 确认环境
    python phase18_base_sweep.py --mla-compare             # 跑 MLA vs MHA 对比 (d64 vs d128, base=500K)
"""

# === 主组: MLA-realistic d_head=64 ===
BASES = [10_000, 100_000, 500_000, 1_000_000, 10_000_000]
SEEDS = [42, 137, 256]
D_HEAD_MLA = 64        # DeepSeek V3 / GLM-5 / Kimi K2.5 qk_rope_head_dim
D_HEAD_MHA = 128       # LLaMA 3.1 旧架构（对比组）
TAU_STAR_MLA = D_HEAD_MLA / (512 ** 0.5)   # ≈ 2.828
TAU_STAR_MHA = D_HEAD_MHA / (512 ** 0.5)   # ≈ 5.657
SEQ_LEN = 512
TRAIN_TOKENS = 50_000_000
EVAL_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384]

# === MLA vs MHA 对比配置 ===
# Config A: hidden=768, H=12, d_head=64 (MLA-era)
# Config B: hidden=768, H=6,  d_head=128 (MHA-era)
# 仅 base=500K，3-seed, Geo + EVQ + YaRN overlay
```

### 2.1 关键逻辑

```python
# === 主组: 5 bases × d_head=64 (MLA-realistic) ===
for base in BASES:
    for seed in SEEDS:
        for tau in [0.0, TAU_STAR_MLA]:  # Geo vs EVQ
            name = f"mla64_base{base}_tau{tau:.2f}_seed{seed}"
            if result_exists(name): skip

            inv_freq = evq_cosh_inv_freq(d_head=64, tau=tau, base=base)
            model = GPT(cfg_125m_h12_d64, inv_freq)
            train(model, data_L512, 50M tokens)
            ppl = eval(model, val, EVAL_LENGTHS)
            save(name, ppl)

# === 对比组: base=500K, d_head=128 (旧 MHA) ===
if args.mla_compare:
    for seed in SEEDS:
        for tau in [0.0, TAU_STAR_MHA]:
            name = f"mha128_base500k_tau{tau:.2f}_seed{seed}"
            inv_freq = evq_cosh_inv_freq(d_head=128, tau=tau, base=500_000)
            model = GPT(cfg_125m_h6_d128, inv_freq)
            train(model, data_L512, 50M tokens)
            ppl = eval(model, val, EVAL_LENGTHS)
            # + YaRN overlay eval
            ppl_yarn = eval_yarn(model, val, EVAL_LENGTHS, scale=4)
            save(name, ppl, ppl_yarn)
```

### 2.2 输出格式

每个 run 输出到 `results/core_text/phase18_base_sweep/{name}/result.json`：

```json
{
  "base": 500000,
  "tau": 2.83,
  "seed": 42,
  "ppl": {"512": 30.1, "1024": 42.3, "2048": 78.5, "4096": 145.2, "8192": 267.8, "16384": 445.1},
  "train_time_sec": 720,
  "model_params_M": 124.5
}
```

跑完后自动生成 `phase18_summary.json`，包含所有 run 的 3-seed 均值和标准差。

---

## 3. 预期结果与论文叙事

### 3.1 预期 pattern

| Base | EVQ vs Geo @4K (8×) | EVQ vs Geo @16K (32×) | 理论解释 |
|------|---------------------|----------------------|---------|
| 10K | -2% ~ -5% | -5% ~ -10% | 碰撞块 c=0.68，增益有限但不为零 |
| 100K | -8% ~ -15% | -15% ~ -25% | 中等可优化空间 |
| 500K | -10% ~ -20% | -20% ~ -35% | 已知 anchor |
| 1M | -10% ~ -20% | -20% ~ -35% | 应接近 500K |
| 10M | -12% ~ -25% | -25% ~ -40% | 最大可优化空间 |

### 3.2 最理想结论

> EVQ gain generalizes across base ∈ {10K, 100K, 500K, 1M, 10M} under the MLA-realistic d_head=64 configuration matching DeepSeek V3 and GLM-5. Improvement monotonically increases with base, consistent with collision-block theory: larger base → lower collision fraction c → more channels benefit from frequency reallocation. Furthermore, EVQ's relative gain is larger at d_head=64 than d_head=128, confirming that frequency allocation optimization becomes more critical as MLA compresses the RoPE subspace.

### 3.3 MLA vs MHA 对比的预期结论

> With MLA compressing RoPE to d_head=64 (32 frequency pairs), the per-channel allocation has higher leverage: each misallocated channel costs proportionally more. EVQ's relative improvement at d_head=64 should be ≥ that at d_head=128, confirming that **MLA makes frequency optimization more important, not less**.

### 3.4 论文中的位置

- **正文 Table**："EVQ Δ% @16K across base (d_head=64, MLA-realistic)" — 五个数字说明 base 泛化
- **正文 1 行**：d_head=64 vs d_head=128 EVQ gain 对比，说明 MLA 趋势
- 或 **Appendix 完整表** + 正文一句：*"EVQ gain generalizes across base (Table X); crucially, the gain is amplified at d_head=64 — the RoPE dimension used by state-of-the-art MLA architectures (DeepSeek V3, GLM-5, Kimi K2.5)."*

---

## 4. 执行检查清单

```
Step 0: 环境检查
  - [ ] 确认 run_evq_sweep.py 在 MPS 上能跑 125M @ L=512
  - [ ] 确认 FineWeb-Edu 数据可下载/已缓存
  - [ ] 确认 cfg_125m_h12_d64 (H=12, d=64) 和 cfg_125m_h6_d128 (H=6, d=128) 两种配置可用

Step 1: Pilot — MLA config (10 min)
  - [ ] base=500K, d_head=64, seed=42, τ=0 和 τ=2.83 各跑一次
  - [ ] 确认 PPL 范围合理、无 NaN、训练时间 ~10-15 min

Step 2: base=10K pilot (10 min)
  - [ ] d_head=64, seed=42 only, τ=0 和 τ=2.83
  - [ ] 确认碰撞块预测：EVQ 增益应显著弱于 500K

Step 3: 全量 base sweep — MLA config (~5-7h)
  - [ ] 5 bases × 2 methods × 3 seeds = 30 runs (d_head=64)
  - [ ] 断点续跑，已完成自动跳过

Step 4: MLA vs MHA 对比组 (~1-2h)
  - [ ] base=500K, d_head=128 (H=6), 3 seeds × 2 methods = 6 runs
  - [ ] + YaRN overlay eval on all 6 runs
  - [ ] 比较 d64 vs d128 的 EVQ gain 差异

Step 5: 汇总
  - [ ] 生成 summary table (3-seed mean ± std)
  - [ ] 检查 EVQ gain vs base 单调性
  - [ ] 比较 d64 vs d128 EVQ gain (MLA vs MHA era)
  - [ ] 画 "EVQ gain vs (1-c)/ln(b)" 散点图（optional bonus）
```

---

## 5. 碰撞块理论定量验证 (Bonus)

如果 5 个 base 的 EVQ gain 数据足够干净，可以画：

**X 轴**：理论净增益 `(1-c)/ln(b)`
**Y 轴**：实测 EVQ PPL improvement @16K (3-seed mean)

如果呈正相关（接近线性），就是碰撞块理论的**最直接定量验证**。这个图价值极高：
- 可以直接进 Figure 4（mechanism 图）
- 回答 "为什么 base=10K 不行"
- 回答 "base 选什么最好"
- 把 §9 碰撞块分析从纯理论升级为理论+实验双验证
