# Spotlight 冲刺实验计划

> 目标: 从 Poster (63-75%) 推到 Spotlight (80%+)
> 预计总 GPU 时间: ~50-70 小时 (RTX 5090)
> 优先级: 实验 1 > 实验 3 > 实验 2

---

## 实验 1: Progressive Stage 2-3 多种子 [最高优先级]

### 动机
GPT 给 Experimental Thoroughness 只打了 6.5/10, 主因就是 "progressive stages 2-3 single-seed"。补上 2 个种子可以直接把这项拉到 8+。

### 前置条件
- Stage 1 (L=512) seeds 43/44 的 checkpoint 应该已存在于 AutoDL 服务器
  - 论文已报告 seeds 43/44 的 Stage 1 结果 (PPL@4K: -16.5%)
  - 确认路径: `/root/autodl-tmp/evq_phase17/` 下是否有 seed43/seed44 目录
  - **如果不存在**: 需要先跑 Stage 1 (每 seed ~9h, 1B tokens, L=512)

### Step 1: 确认 Stage 1 checkpoints

```bash
# 在 AutoDL 服务器上检查
ls /root/autodl-tmp/evq_phase17/*seed43* 2>/dev/null
ls /root/autodl-tmp/evq_phase17/*seed44* 2>/dev/null
# 也可能在子目录中
find /root/autodl-tmp/evq_phase17 -name "*.pt" | grep -E "seed.*(43|44)"
```

如果不存在，先跑 Stage 1:
```bash
# 跑 seed 43
PHASE17_SEED=43 python scripts/core_text_phases/run_evq_sweep.py \
    --mode single --tau 2.83 --base 500000 --tokens 1000000000 --seq_len 512
# 跑 seed 44
PHASE17_SEED=44 python scripts/core_text_phases/run_evq_sweep.py \
    --mode single --tau 2.83 --base 500000 --tokens 1000000000 --seq_len 512
```

### Step 2: Stage 2 (L=1024) × seeds 43, 44

```bash
# === Seed 43 ===
export PHASE17B_SEED=43
export PHASE17B_TAU=2.0                    # d_head/√1024 = 64/32 = 2.0
export PHASE17B_SEQ_LEN=1024
export PHASE17B_TOKENS=1000000000          # 1B tokens
export PHASE17B_GEO_INIT_CKPT="/root/autodl-tmp/evq_phase17/<geo_seed43_ckpt_path>"
export PHASE17B_EVQ_INIT_CKPT="/root/autodl-tmp/evq_phase17/<evq_seed43_ckpt_path>"
export PHASE17B_WORK="/root/autodl-tmp/evq_phase17b_seed43"
python scripts/core_text_phases/phase17b_454m_512_to_1024_continue_ckpt_eval.py

# === Seed 44 ===
export PHASE17B_SEED=44
export PHASE17B_GEO_INIT_CKPT="/root/autodl-tmp/evq_phase17/<geo_seed44_ckpt_path>"
export PHASE17B_EVQ_INIT_CKPT="/root/autodl-tmp/evq_phase17/<evq_seed44_ckpt_path>"
export PHASE17B_WORK="/root/autodl-tmp/evq_phase17b_seed44"
python scripts/core_text_phases/phase17b_454m_512_to_1024_continue_ckpt_eval.py
```

**预计时间**: ~10h/seed × 2 forks (Geo+EVQ) × 2 seeds = ~40h
**可并行**: 如果有 2 张 GPU, Geo 和 EVQ 可以各占一张

### Step 3: Stage 3 (L=2048) × seeds 43, 44

```bash
# === Seed 43 ===
export PHASE17C_SEED=43
export PHASE17C_TAU=1.414                  # d_head/√2048 = 64/√2048 ≈ 1.414
export PHASE17C_SEQ_LEN=2048
export PHASE17C_TOKENS=500000000           # 500M tokens
export PHASE17C_GEO_INIT_CKPT="/root/autodl-tmp/evq_phase17b_seed43/454m_geo_seed43_continue1024/model.pt"
export PHASE17C_EVQ_INIT_CKPT="/root/autodl-tmp/evq_phase17b_seed43/454m_evq_seed43_continue1024/model.pt"
export PHASE17C_WORK="/root/autodl-tmp/evq_phase17c_seed43"
export PHASE17C_MICRO_BATCH_SIZE=5
export PHASE17C_GRAD_ACCUM=4
python scripts/core_text_phases/phase17c_454m_1024_to_2048_continue.py

# === Seed 44 (同上，改 seed) ===
```

**预计时间**: ~5h/seed × 2 forks × 2 seeds = ~20h (500M tokens, L=2048 更慢但 token 数减半)

### 收集的关键数据

对每个 seed, 记录:
- PPL@16K (raw, no YaRN): 用于计算 3-seed 平均的 -34.6% / -52.0% / -81.2%
- PPL@{2K,4K,8K,16K,32K,48K} (raw + YaRN): 用于更新 appendix 表格
- Passkey retrieval @{1K,2K,4K,8K}: 确认 100% 是否跨 seed 稳定

### 论文更新

更新目标文件:
1. `sections/05_experiments.tex` §5.3:
   - "primary seed" → "3-seed average (seeds 42/43/44)"
   - 加入 ±std 或 min-max range
2. `appendix/a3_supporting_results.tex`:
   - Stage-by-stage 表格扩展为 3-seed 版本
   - 添加 "mean ± std" 列
3. `main.tex` abstract:
   - "primary seed; Stage 1 verified..." → "3-seed average"

**影响**: 这一个改动预计将 GPT 的 Experimental Thoroughness 从 6.5 提升到 8.0+

---

## 实验 2: Same-Codebase DAPE 复现 [中优先级]

### 动机
GPT 指出 "DAPE comparison relies on numbers from the original paper rather than a same-codebase rerun"。当前论文的应对是 "verify that our own Geo baseline matches their reported Geo numbers within <2%"，但同 codebase 复现更有说服力。

### 方案: 基于现有 Phase 11B 框架

Phase 11B 已经有完整的 DAPE 实现 (`phase11b_125m_dape.py`)，但是在 L=256 设置下。需要调整为 L=128。

### 修改 Config

```python
# 在 phase11b_125m_dape.py 基础上修改:
SEQ_LEN = 128          # 原来是 256
TOKENS = 15_000_000    # 15M tokens (与 EXPERIMENT_RESULTS_128TOK.md 一致)

CFG_125M = dict(
    vocab_size=50304,
    hidden_size=768,
    num_layers=12,
    num_heads=12,
    head_dim=64,
    intermediate_size=3072,
    max_position_embeddings=128,
    seq_len=128,
    train_tokens=15_000_000,
    lr=6e-4,
    batch_size=256,
    micro_batch_size=64,
    grad_accum=4,
)

EVAL_LENGTHS = [128, 512, 2048, 4096, 8192]
```

### 实验矩阵

| Method | Extra Params | τ | Seeds | 说明 |
|--------|:---:|:---:|:---:|------|
| Geo | 0 | 0 | 42,137,256 | baseline |
| EVQ Fixed | 0 | 1.5 | 42,137,256 | 论文主要结果 |
| DAPE (Kerple+MLP) | 32 | N/A | 42,137,256 | same-codebase DAPE |
| EVQ + DAPE | 32 | 1.5 | 42,137,256 | 新: 两者结合是否进一步改善? |

### 运行命令

```bash
# 新建脚本或修改参数
export PHASE11B_SEQ_LEN=128
export PHASE11B_TOKENS=15000000
python scripts/core_text_phases/phase11b_125m_dape.py
```

**预计时间**: ~25 分钟 × 4 methods × 3 seeds = ~5h (之前的 128-tok 实验 25 分钟/run)

### 论文更新

如果 same-codebase DAPE 结果与原论文一致 (PPL@8K ~455):
1. `sections/05_experiments.tex` §5.4: 删除 "For comparability, the DAPE numbers are taken from the original paper" 注释，换成 "We reproduce DAPE in our codebase under identical protocol"
2. Table 4: 如有变化，更新数据

如果 EVQ+DAPE 组合进一步改善:
- 这是加分项 — 证明 EVQ 是 orthogonal 基础设施

**风险**: 低。128-tok DAPE 实验已经跑过，只是不是以 "same-codebase comparison" 的形式呈现。

---

## 实验 3: Full-ODE (P≠0) Ablation [中高优先级]

### 动机
GPT: "there is no empirical ablation showing that the full solution is actually inferior, negligible, or unstable in practice. For a paper built around closed-form derivation, that missing control is a conspicuous hole."

### 理论背景

完整 ODE 解: ρ*(φ) = C₁cosh(τφ) + C₂sinh(τφ) + P·b^{-2φ}
其中 P = μ(2ln b)² / (α((2ln b)² - τ²))

Pure-tether (当前 EVQ): μ=0, P=0
Full-ODE: μ>0, P>0, 额外 forcing term

### 实现方案

在 `run_evq_sweep.py` 的 `evq_cosh_inv_freq` 基础上新增:

```python
def evq_full_ode_inv_freq(
    head_dim: int, tau: float, mu_alpha: float, base: float = 500000.0
) -> torch.Tensor:
    """Full ODE solution with P≠0 forcing term.

    ρ(φ) = C·cosh(τ(1-φ)) + P·b^{-2φ}

    P = mu_alpha * (2*ln(b))^2 / ((2*ln(b))^2 - τ^2)

    mu_alpha is the ratio μ/α, a free parameter controlling forcing strength.
    We normalize ρ to integrate to 1, then invert the CDF numerically.

    Args:
        head_dim: head dimension (K = head_dim // 2 channels)
        tau: temperature parameter
        mu_alpha: forcing ratio μ/α (0 = pure tether, typical range: 0.01 ~ 1.0)
        base: RoPE base frequency
    """
    K = head_dim // 2
    ln_b = math.log(base)
    ln_b2 = (2 * ln_b) ** 2

    # P coefficient
    if abs(ln_b2 - tau**2) < 1e-10:
        # degenerate case: resonance
        P = 0.0
    else:
        P = mu_alpha * ln_b2 / (ln_b2 - tau**2)

    # Build density ρ(φ) on fine grid
    N = 10000
    phi_grid = torch.linspace(0, 1, N, dtype=torch.float64)

    # Homogeneous part: cosh(τ(1-φ))
    rho_hom = torch.cosh(tau * (1 - phi_grid))

    # Particular part: P * b^{-2φ}
    rho_part = P * torch.pow(torch.tensor(base, dtype=torch.float64), -2 * phi_grid)

    rho = rho_hom + rho_part
    rho = torch.clamp(rho, min=1e-10)  # safety: ensure positive

    # Normalize to CDF
    cdf = torch.cumsum(rho, dim=0)
    cdf = cdf / cdf[-1]

    # Inverse CDF at midpoint quantiles
    u = (torch.arange(K, dtype=torch.float64) + 0.5) / K
    phi_k = torch.zeros(K, dtype=torch.float64)
    for i in range(K):
        idx = torch.searchsorted(cdf, u[i])
        idx = min(idx.item(), N - 1)
        phi_k[i] = phi_grid[idx]

    inv_freq = torch.pow(torch.tensor(base, dtype=torch.float64), -phi_k)
    return inv_freq.float()
```

### 实验矩阵

在 350M, L=512, b=500K, 3 seeds 设置下测试:

| Method | τ | μ/α | Description |
|--------|:---:|:---:|------------|
| Geo | 0 | 0 | baseline |
| EVQ (P=0) | 2.83 | 0 | 当前论文方法 |
| Full-ODE (weak) | 2.83 | 0.01 | 弱 forcing |
| Full-ODE (medium) | 2.83 | 0.1 | 中等 forcing |
| Full-ODE (strong) | 2.83 | 1.0 | 强 forcing |
| Full-ODE (oracle) | 2.83 | grid search | 扫描最优 μ/α |

### 运行命令

```bash
# 基于 Phase 16 sweep 框架，修改 inv_freq builder
python scripts/core_text_phases/run_evq_sweep.py \
    --mode single \
    --method full_ode \
    --tau 2.83 \
    --mu_alpha 0.1 \
    --base 500000 \
    --tokens 1000000000 \
    --seq_len 512 \
    --seed 42
```

需要修改 `run_evq_sweep.py`:
1. 添加 `evq_full_ode_inv_freq` 函数
2. 在 `--method` 参数中添加 `full_ode` 选项
3. 添加 `--mu_alpha` 参数

**预计时间**: ~9h/run × 5 configs × 1 seed = ~45h (单 seed 先验证方向)
**快速版**: 只跑 P=0 vs P(μ/α=0.1) × 1 seed = ~18h

### 预期结果

根据理论分析:
- **最可能**: Full-ODE 与 Pure-tether 差异 <0.5% PPL — 验证 P=0 近似合理
- **次可能**: Full-ODE 在某个 μ/α 下略好 — 但改善幅度 <1%, 不影响核心结论
- **不太可能**: Full-ODE 显著优于 Pure-tether — 这会是新发现，但要小心: 如果 P 依赖 b, 则失去 base-agnostic 优势

### 论文更新

无论结果如何，都有价值:
- 如果 P=0 ≈ Full-ODE: 在 §3.4 加一句 "empirical ablation confirms that the pure-tether branch matches the full solution within 0.X% (Appendix C.X)"
- 如果 Full-ODE 略好: 讨论 trade-off (微小 PPL 改善 vs 失去 base-agnostic 封闭形式)

---

## 执行时间线 (建议)

### Day 1 (明天)

| 时间 | 任务 | GPU 占用 |
|------|------|----------|
| 上午 | 确认 Stage 1 seed 43/44 checkpoints 是否存在 | 0 |
| 上午 | 如果不存在: 启动 Stage 1 seed 43 (Geo+EVQ) | 1 GPU |
| 上午 | 同时启动 DAPE same-codebase 实验 (128-tok, 快速) | 1 GPU |
| 下午 | DAPE 实验完成 (~5h), 收集数据 | 0 |
| 下午 | 实现 `evq_full_ode_inv_freq`, 启动 quick ablation | 1 GPU |
| 晚上 | Stage 1 seed 43 训练中 / Full-ODE ablation 训练中 | 1-2 GPU |

### Day 2

| 时间 | 任务 | GPU 占用 |
|------|------|----------|
| 上午 | Stage 1 seed 43 完成, 启动 seed 44 | 1 GPU |
| 上午 | 启动 Stage 2 seed 43 (如 Stage 1 已有 checkpoint) | 1 GPU |
| 下午 | Full-ODE ablation 完成, 分析结果 | 0 |
| 晚上 | Stage 2 训练中 | 2 GPU |

### Day 3

| 时间 | 任务 | GPU 占用 |
|------|------|----------|
| 全天 | Stage 2 完成 → Stage 3 启动 | 2 GPU |

### Day 4

| 时间 | 任务 | GPU 占用 |
|------|------|----------|
| 上午 | Stage 3 完成, 收集所有数据 | 0 |
| 下午 | 更新论文, 重新编译 | 0 |
| 晚上 | 再跑一轮 AI review (Version D), 预计 80%+ | 0 |

---

## 快速通道 (如果时间紧张)

如果只有 1 天时间, 按 ROI 排序:

1. **DAPE same-codebase** (~5h) — 低成本高回报, 堵住 "not same-codebase" 的口
2. **Full-ODE quick ablation** (~18h, 后台跑) — P=0 vs P(0.1), 1 seed, 验证理论
3. **Progressive 多种子** — 如果 Stage 1 checkpoints 已存在, 直接进 Stage 2

---

## 数据收集模板

### 实验 1 数据模板

```
# Stage-by-stage PPL@16K (3-seed)
| Stage | L_train | Geo (mean±std) | EVQ (mean±std) | Δ (mean) |
|-------|---------|----------------|----------------|----------|
| 1     | 512     | ?±?            | ?±?            | ?%       |
| 2     | 1024    | ?±?            | ?±?            | ?%       |
| 3     | 2048    | ?±?            | ?±?            | ?%       |
```

### 实验 2 数据模板

```
# Same-codebase 128→8K
| Method        | Params | PPL@128 | PPL@8K | Δ vs Geo |
|---------------|--------|---------|--------|----------|
| Geo           | 0      | ?       | ?      | —        |
| DAPE          | 32     | ?       | ?      | ?%       |
| EVQ τ=1.5     | 0      | ?       | ?      | ?%       |
| EVQ+DAPE      | 32     | ?       | ?      | ?%       |
```

### 实验 3 数据模板

```
# Full-ODE ablation (350M, L=512, seed 42)
| Method          | μ/α  | PPL@2K | PPL@4K | PPL@8K | PPL@16K |
|-----------------|------|--------|--------|--------|---------|
| Geo             | —    | ?      | ?      | ?      | ?       |
| EVQ (P=0)       | 0    | ?      | ?      | ?      | ?       |
| Full-ODE (0.01) | 0.01 | ?      | ?      | ?      | ?       |
| Full-ODE (0.1)  | 0.1  | ?      | ?      | ?      | ?       |
| Full-ODE (1.0)  | 1.0  | ?      | ?      | ?      | ?       |
```
