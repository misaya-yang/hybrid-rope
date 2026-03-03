# Phase 12: r-sweep — 验证 Hybrid Warp 边界公式 r*

> **核心目标**: 扫描 Hybrid 的 warp 通道数 r，验证 r* = (d/(2·ln b))·ln(L/(2π)) 公式，找到 PPL-Passkey Pareto 最优点
> **硬件**: 新开 5090 32GB
> **预计 GPU 时间**: ~8-10h
> **论文价值**: **冲 spotlight/oral 的关键**——验证 r* 后论文可以 claim "both τ and r are analytically derived, truly zero hyperparameters"

---

## 0. 背景与动机

### 0.1 当前理论有两个公式

| 公式 | 含义 | 验证状态 |
|------|------|----------|
| τ*(L) = d_head/√L | EVQ warp 强度 | ✅ Phase 8D 5 点验证 |
| r* = (d/(2·ln b))·ln(L/(2π)) | Hybrid warp 边界（保持 Geometric 的高频通道数） | ❌ **从未验证** |

### 0.2 存在关键矛盾

| 实验 | 方法 | OOD PPL vs Geo | 说明 |
|------|------|----------------|------|
| 350M 3-seed | Full EVQ (r=0, 全 32 通道 warp) | **-13.3%** ✅ | 全通道 warp，PPL 赢 |
| 750M Phase9F | Hybrid (r=16, 只 warp 低频 16 通道) | **+5.7%** ❌ | 半通道 warp，PPL 输 |

两个实验方向**相反**。不做 r-sweep 无法判断是 r 的问题还是模型规模的问题。

### 0.3 r* 具体预测

d=64, L=2048:

| base | ln(b) | r* = (64/(2·ln b))·ln(2048/6.28) | 实际 |
|------|-------|----------------------------------|------|
| 500K | 13.12 | **14.1** | 用 r=16 |
| 10K | 9.21 | 20.1 | 用 r=16 ❌ |

### 0.4 Hybrid 代码实现（已有，在 phase9 脚本中）

```python
def hybrid_evq_inv_freq(dim=64, base=500000, tau=1.5, r=16):
    """r = 保持 Geometric 的高频通道数。n_evq = d/2 - r 个低频通道做 EVQ warp。"""
    n = dim // 2  # 32
    geo = torch.tensor([1.0 / (base ** (2 * i / dim)) for i in range(n)])
    n_evq = n - r
    if n_evq <= 0:
        return geo.float()  # r >= 32 = 纯 Geometric
    theta_max = geo[r].item()  # EVQ 区域的最高频率
    theta_min = geo[-1].item()  # EVQ 区域的最低频率
    u = torch.arange(n_evq, dtype=torch.float64) / max(n_evq - 1, 1)
    if abs(tau) < 1e-8:
        phi = 1.0 - u
    else:
        phi = 1.0 - (1.0 / tau) * torch.arcsinh((1.0 - u) * math.sinh(tau))
    evq_part = (theta_min ** phi) * (theta_max ** (1.0 - phi))
    return torch.cat([geo[:r], evq_part]).float()
```

**通道布局**（d_head=64, K=32 通道）:
```
通道 index:  0  1  2  ...  r-1 | r  r+1  ...  31
频率:        高 ←──────────── | ──────────→ 低
方法:        Geometric (锁定)  | EVQ warp (重分配)
```

- r=0: Full EVQ（全 32 通道 warp）= `evq_cosh_inv_freq(tau=1.5)`
- r=16: 当前默认 Hybrid（16 Geo + 16 EVQ）
- r=32: 纯 Geometric（等效 τ=0）

---

## 1. 服务器信息

- SSH: `ssh -p <PORT> root@<HOST>`（开机后填写）
- 密码:（开机后填写）
- GPU: 5090 32GB
- 代码: 从本地同步 `hybrid-rope/`

## 2. 环境初始化

```bash
export PATH="/root/miniconda3/bin:$PATH"
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1
```

若 miniconda 不存在：
```bash
pip install torch transformers datasets tokenizers --break-system-packages
```

---

## 3. 代码改动

### 3.1 核心改动：在 `run_evq_sweep.py` 中添加 Hybrid/r-sweep 支持

#### 3.1.1 添加 `hybrid_evq_inv_freq` 函数

在 `evq_cosh_inv_freq` 函数之后，添加：

```python
def hybrid_evq_inv_freq(
    head_dim: int, tau: float, r: int, base: float = 500000.0
) -> torch.Tensor:
    """Hybrid frequency allocation: first r channels stay Geometric, rest use EVQ warp.

    Args:
        head_dim: head dimension (e.g. 64)
        tau: EVQ warp temperature
        r: number of high-frequency channels to keep as Geometric
           r=0 → full EVQ, r=K → full Geometric
        base: RoPE base frequency

    Returns:
        inv_freq of shape (head_dim // 2,) in float32
    """
    K = head_dim // 2
    if r <= 0:
        return evq_cosh_inv_freq(head_dim, tau, base)
    if r >= K:
        return evq_cosh_inv_freq(head_dim, 0.0, base)  # geometric

    # Geometric part: channels 0..r-1 (high-frequency)
    geo_full = evq_cosh_inv_freq(head_dim, 0.0, base)
    geo_part = geo_full[:r]

    # EVQ part: channels r..K-1 (low-frequency), warp within [theta_r, theta_{K-1}]
    n_evq = K - r
    theta_max = geo_full[r].item()      # highest freq in EVQ region
    theta_min = geo_full[K - 1].item()  # lowest freq in EVQ region

    idx = torch.arange(n_evq, dtype=torch.float64)
    u = idx / max(n_evq - 1, 1)  # 0..1 within EVQ region

    if abs(tau) < 1e-8:
        phi = 1.0 - u
    else:
        sinh_tau = math.sinh(tau)
        phi = 1.0 - (1.0 / tau) * torch.arcsinh((1.0 - u) * sinh_tau)

    evq_part = (theta_min ** phi) * (theta_max ** (1.0 - phi))

    return torch.cat([geo_part, evq_part.float()])
```

#### 3.1.2 添加 CLI 参数

在 `argparse` 区域添加：

```python
parser.add_argument("--r_values", type=str, default=None,
                    help="Comma-separated Hybrid r values to sweep (e.g. '0,8,14,16,24,32'). "
                         "If provided, runs r-sweep instead of tau-sweep. "
                         "r=0 means full EVQ, r=32 means full Geometric.")
parser.add_argument("--fixed_tau", type=float, default=None,
                    help="Fixed tau for r-sweep (required when --r_values is set)")
```

#### 3.1.3 修改 main() 逻辑

在 `main()` 中处理 r-sweep 模式。在 `taus = ...` 之后添加：

```python
r_sweep_mode = args.r_values is not None
if r_sweep_mode:
    r_values = [int(r) for r in args.r_values.split(",")]
    fixed_tau = args.fixed_tau
    if fixed_tau is None:
        # 用理论预测的 tau*
        fixed_tau = cfg["head_dim"] / math.sqrt(cfg["seq_len"])
        print(f"  [r-sweep] No --fixed_tau provided, using tau*={fixed_tau:.2f}")
    print(f"  [r-sweep] r_values={r_values}, fixed_tau={fixed_tau:.2f}")
```

#### 3.1.4 修改 `run_single_experiment` 或在 main loop 中切换 inv_freq

**方案 A**（推荐，最小改动）：在 main 的 tau loop 中，当 r_sweep_mode 时改为 r loop：

```python
if r_sweep_mode:
    for seed in seeds:
        for r_val in r_values:
            # Build inv_freq for this r value
            inv_freq = hybrid_evq_inv_freq(cfg["head_dim"], fixed_tau, r_val, base)
            run_id = f"{args.tier}_r{r_val}_tau{fixed_tau:.2f}_seed{seed}"

            # 复用 run_single_experiment 的逻辑，但传入自定义 inv_freq
            # 需要修改 run_single_experiment 接受 optional inv_freq 参数
            result = run_single_experiment(
                tier=args.tier, tau=fixed_tau, seed=seed, base=base,
                cfg=cfg, work_dir=work_dir, dry_run=args.dry_run,
                tokenizer=tokenizer, eval_16k=args.eval_16k,
                override_inv_freq=inv_freq,    # 新增参数
                override_run_id=run_id,         # 新增参数
            )
            all_results[run_id] = result
else:
    # 原有 tau-sweep 逻辑不变
    for seed in seeds:
        for tau in taus:
            ...
```

#### 3.1.5 修改 `run_single_experiment` 函数签名

添加两个可选参数：

```python
def run_single_experiment(
    tier: str, tau: float, seed: int, base: float,
    cfg: dict, work_dir: Path,
    dry_run: bool = False,
    tokenizer=None,
    eval_16k: bool = False,
    override_inv_freq: Optional[torch.Tensor] = None,  # 新增
    override_run_id: Optional[str] = None,               # 新增
) -> RunResult:
    run_id = override_run_id or f"{tier}_tau{tau:.2f}_seed{seed}"
    ...
    # Build inv_freq
    if override_inv_freq is not None:
        inv_freq = override_inv_freq
    else:
        inv_freq = evq_cosh_inv_freq(cfg["head_dim"], tau, base)
    ...
```

### 3.2 验证改动

改完后先 dry_run 验证：

```bash
python run_evq_sweep.py \
    --tier 350m \
    --r_values 0,14,16,32 \
    --fixed_tau 1.5 \
    --seeds 42 \
    --base 500000 \
    --work_dir /tmp/test_r_sweep \
    --dry_run
```

应输出 4 个 run_id 及其 inv_freq hash：
- `350m_r0_tau1.50_seed42` — Full EVQ, inv_freq hash 应与 `evq_cosh_inv_freq(64, 1.5)` 一致
- `350m_r14_tau1.50_seed42` — r* 理论最优
- `350m_r16_tau1.50_seed42` — 当前默认
- `350m_r32_tau1.50_seed42` — 纯 Geometric, inv_freq hash 应与 tau=0 一致

---

## 4. 实验设计

### 4.1 核心实验：r-sweep（固定 τ=1.5）

**条件**：350M, L=2048, base=500K, τ=1.5, 50M tokens, FineWeb-Edu + 10% passkey

| Run | r | 含义 | EVQ 通道数 | seed | 预计时间 |
|-----|---|------|-----------|------|----------|
| 1 | 0 | Full EVQ | 32/32 | 42 | ~50min |
| 2 | 4 | 极重 warp | 28/32 | 42 | ~50min |
| 3 | 8 | 重 warp | 24/32 | 42 | ~50min |
| 4 | 12 | 中等 | 20/32 | 42 | ~50min |
| 5 | **14** | **r* 理论预测** | **18/32** | 42 | ~50min |
| 6 | 16 | 当前默认 | 16/32 | 42 | ~50min |
| 7 | 20 | 轻 warp | 12/32 | 42 | ~50min |
| 8 | 24 | 极轻 warp | 8/32 | 42 | ~50min |
| 9 | 32 | Geometric baseline | 0/32 | 42 | ~50min |

**seed=42 pilot 完成后**（确认趋势合理），补 seed=137：

| Run | r | seed | 预计时间 |
|-----|---|------|----------|
| 10-18 | 同上 0,4,8,12,14,16,20,24,32 | 137 | ~50min×9 |

**总计**: 18 runs × ~50min = **~15h**。如时间紧，seed=137 只跑关键 r 值（0,14,16,32）= 4 runs 额外。

### 4.2 加分实验：τ-r 联合验证（如有余力）

在 r=14（r* 理论值）处加一个 τ-sweep：

| Run | r | τ | seed | 预计时间 |
|-----|---|---|------|----------|
| 19 | 14 | 0.5 | 42 | ~50min |
| 20 | 14 | 1.0 | 42 | ~50min |
| 21 | 14 | 2.0 | 42 | ~50min |
| 22 | 14 | 2.5 | 42 | ~50min |

（τ=1.5 已在 Run 5 中跑过）

这给出二维最优点 (r*, τ*)，直接证明两个公式联合正确。

---

## 5. 执行命令

### Step 1: 同步代码并做改动

```bash
# 同步到服务器
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='*.pt' \
    /path/to/hybrid-rope/ root@<HOST>:<PORT>:/root/autodl-tmp/dfrope/hybrid-rope/

# SSH 到服务器后做 §3 的代码改动
```

### Step 2: Pilot（seed=42, 关键 r 值）

```bash
cd /root/autodl-tmp/dfrope/hybrid-rope/scripts/m4_evq_sweep
mkdir -p /root/autodl-tmp/evq_phase12_r_sweep

# 先跑 4 个关键 r 值验证趋势
nohup python -u run_evq_sweep.py \
    --tier 350m \
    --r_values 0,14,16,32 \
    --fixed_tau 1.5 \
    --seeds 42 \
    --base 500000 \
    --passkey_mix_ratio 0.10 \
    --eval_16k \
    --work_dir /root/autodl-tmp/evq_phase12_r_sweep \
    > /root/autodl-tmp/evq_phase12_r_sweep/pilot_seed42.log 2>&1 &

# 监控
tail -f /root/autodl-tmp/evq_phase12_r_sweep/pilot_seed42.log
```

### Step 3: Pilot 完成后检查趋势

读取 4 个 result.json，输出快速对比：
```
r=0  (Full EVQ):  PPL@2K=?  PPL@16K=?  PK@4K=?  PK@8K=?
r=14 (r*):        PPL@2K=?  PPL@16K=?  PK@4K=?  PK@8K=?
r=16 (current):   PPL@2K=?  PPL@16K=?  PK@4K=?  PK@8K=?
r=32 (Geometric): PPL@2K=?  PPL@16K=?  PK@4K=?  PK@8K=?
```

**判断标准**：
- 如果 r=0 的 PPL@16K 最优但 PPL@2K 退化 >1% → Hybrid 有意义，继续细粒度扫描
- 如果 r=14 和 r=16 在 PPL@2K 上接近 r=32 且 PPL@16K 接近 r=0 → r* 公式正确
- 如果 r=0 Pareto 支配所有其他 r → Hybrid 理论需修正，Full EVQ 更好

### Step 4: 完整 r-sweep（seed=42 全 9 值）

```bash
nohup python -u run_evq_sweep.py \
    --tier 350m \
    --r_values 0,4,8,12,14,16,20,24,32 \
    --fixed_tau 1.5 \
    --seeds 42 \
    --base 500000 \
    --passkey_mix_ratio 0.10 \
    --eval_16k \
    --resume \
    --work_dir /root/autodl-tmp/evq_phase12_r_sweep \
    > /root/autodl-tmp/evq_phase12_r_sweep/full_seed42.log 2>&1 &
```

### Step 5: 补 seed=137（关键 r 值）

```bash
nohup python -u run_evq_sweep.py \
    --tier 350m \
    --r_values 0,14,16,32 \
    --fixed_tau 1.5 \
    --seeds 137 \
    --base 500000 \
    --passkey_mix_ratio 0.10 \
    --eval_16k \
    --resume \
    --work_dir /root/autodl-tmp/evq_phase12_r_sweep \
    > /root/autodl-tmp/evq_phase12_r_sweep/seed137.log 2>&1 &
```

### Step 6（可选）: τ-r 联合验证

```bash
# 在 r=14 处扫描 τ
nohup python -u run_evq_sweep.py \
    --tier 350m \
    --r_values 14 \
    --fixed_tau 0.5 \
    --seeds 42 \
    --base 500000 \
    --passkey_mix_ratio 0.10 \
    --eval_16k \
    --resume \
    --work_dir /root/autodl-tmp/evq_phase12_r_sweep \
    > /root/autodl-tmp/evq_phase12_r_sweep/tau_sweep_r14.log 2>&1 &

# 注意：--fixed_tau 每次只能传一个值
# 需要分别跑 0.5, 1.0, 2.0, 2.5（1.5 已有）
# 或者修改代码支持 --fixed_tau 接受逗号分隔列表
```

---

## 6. 分析模板

### 6.1 Table A: PPL vs r（seed=42）

```
| r  | 含义          | PPL@2K | PPL@4K | PPL@8K | PPL@16K | Δ@2K   | Δ@16K  |
|----|---------------|--------|--------|--------|---------|--------|--------|
| 0  | Full EVQ      | xxx    | xxx    | xxx    | xxx     | +x.x%  | -x.x%  |
| 4  | 极重 warp     | xxx    | xxx    | xxx    | xxx     | +x.x%  | -x.x%  |
| 8  | 重 warp       | xxx    | xxx    | xxx    | xxx     | +x.x%  | -x.x%  |
| 12 | 中等          | xxx    | xxx    | xxx    | xxx     | +x.x%  | -x.x%  |
| 14 | r* (理论)     | xxx    | xxx    | xxx    | xxx     | +x.x%  | -x.x%  |
| 16 | 当前默认      | xxx    | xxx    | xxx    | xxx     | +x.x%  | -x.x%  |
| 20 | 轻 warp       | xxx    | xxx    | xxx    | xxx     | +x.x%  | -x.x%  |
| 24 | 极轻 warp     | xxx    | xxx    | xxx    | xxx     | +x.x%  | -x.x%  |
| 32 | Geometric     | xxx    | xxx    | xxx    | xxx     | —      | —      |
```

Δ 列以 r=32 (Geometric) 为基线。

### 6.2 Table B: Passkey Retrieval vs r（seed=42）

```
| r  | PK@2K | PK@4K | PK@8K | PK Global | AR exact |
|----|-------|-------|-------|-----------|----------|
| 0  | xxx%  | xxx%  | xxx%  | xxx%      | xxx%     |
| ...                                                |
| 32 | xxx%  | xxx%  | xxx%  | xxx%      | xxx%     |
```

### 6.3 Figure: Pareto 曲线

```
x-axis: PPL@2K (短程代价, 越低越好)
y-axis: PPL@16K (长程改善, 越低越好)
点标签: r=0, r=4, ..., r=32
标注: r* 理论预测点
```

### 6.4 理论验证检查清单

| 检查项 | 预期 | 实际 | 通过？ |
|--------|------|------|--------|
| r=0 PPL@16K 最优 | 是 | ? | |
| r=0 PPL@2K 退化 >1% | 是 | ? | |
| r=14 PPL@2K ≈ r=32 (Δ<0.5%) | 是 | ? | |
| r=14 PPL@16K 接近 r=0 (差距 <3%) | 是 | ? | |
| r=14 是 Pareto 前沿拐点 | 是 | ? | |
| r=14 vs r=16 差距 <2% | 是 | ? | |
| r→32 单调回到 Geometric | 是 | ? | |
| passkey 对 r 的敏感度 < PPL 对 r 的敏感度 | 猜测 | ? | |

---

## 7. 成功标准与决策树

### 场景 A: r* 公式验证通过（最佳结果）

条件：r=14 在 Pareto 前沿拐点，PPL@2K ≈ Geometric 且 PPL@16K 接近 Full EVQ

→ **论文 claim**: "Both τ* = d_head/√L and r* = (d/(2·ln b))·ln(L/(2π)) are analytically derived. The entire method is one line of code with zero hyperparameters."

→ **新增 Table**: r-sweep 表 + Pareto 图
→ **新增 Contribution**: "We derive both the warp intensity (τ*) and warp boundary (r*) from variational principles"

### 场景 B: Full EVQ (r=0) Pareto 支配（需要调整）

条件：r=0 在 PPL@2K 和 PPL@16K 上都不逊于任何 Hybrid

→ **论文方向调整**: 放弃 Hybrid 故事线，推荐 Full EVQ
→ **好消息**: 更简单（一行 EVQ 代码，不需要 r 参数）
→ **坏消息**: 与 750M Phase9F 使用 Hybrid 的选择冲突，需解释

### 场景 C: 最优 r 远离 r*（理论修正）

条件：最优在 r=8 或 r=24，而非 r=14

→ **论文影响**: r* 公式需要修正因子
→ **行动**: 分析偏差模式，提出修正公式
→ 仍可声称"理论指导的 r 选择"，只是精度需要改进

---

## 8. 风险与备选

| 风险 | 对策 |
|------|------|
| 350M + 50M tokens 下 r 差异太小（所有 r 都差不多） | 改用 125M（更敏感）或增加到 100M tokens |
| Passkey@4K 在 10% mix 下全部 >80%（天花板效应） | 看 passkey@8K 或 AR exact match |
| 5090 OOM at eval_16k | 代码已有 OOM 保护，自动跳过 16K |
| Hybrid boundary 处 inv_freq 不连续（跳变） | 检查 inv_freq 值：geo[-1] 与 evq[0] 处应平滑 |

---

## 9. 论文用途

### 9.1 新增 Figure（高优先级）

**Figure: r-sweep Pareto 曲线**
```
双 y 轴图：
左 y-axis: PPL@16K (实线)
右 y-axis: PPL@2K (虚线)
x-axis: r (warp 边界)
竖线标注: r* = 14.1 (理论预测)
```

或者 **Pareto 散点图**：
```
x: PPL@2K（短程代价）
y: PPL@16K（长程收益）
每个点标注 r 值
r* 点高亮
```

### 9.2 新增 Table

```latex
\begin{table}[t]
\centering
\caption{Warp boundary sweep (350M, $L{=}2048$, base=500K, $\tau{=}1.5$).
$r$ controls the number of high-frequency channels kept as Geometric.
$r^*{=}14$ (theory-predicted) achieves near-zero short-context cost with
near-maximal long-context improvement.}
\begin{tabular}{lcccccc}
\toprule
$r$ & EVQ channels & $\Delta$@2K & $\Delta$@16K & PK@4K & PK@8K \\
\midrule
0 (Full EVQ) & 32/32 & ... & ... & ...& ... \\
8             & 24/32 & ... & ... & ...& ... \\
\textbf{14 ($r^*$)} & \textbf{18/32} & ... & ... & ...& ... \\
16 (default)  & 16/32 & ... & ... & ...& ... \\
24            &  8/32 & ... & ... & ...& ... \\
32 (Geometric)&  0/32 & ... & ... & ...& ... \\
\bottomrule
\end{tabular}
\end{table}
```

### 9.3 新增 Contribution 文字

> "We derive both the optimal warp intensity $\tau^*(L) = d_{\mathrm{head}}/\sqrt{L}$ (validated across five context lengths) and the optimal warp boundary $r^* = \frac{d}{2\ln b}\ln\frac{L}{2\pi}$ (validated via 9-point $r$-sweep). The entire method reduces to replacing one line of \texttt{inv\_freq} initialization with analytically determined parameters—truly zero hyperparameters beyond the architecture."

---

## 10. ⚠️ Claude Code 执行注意事项

1. **先做代码改动**（§3）：添加 `hybrid_evq_inv_freq` 函数 + `--r_values` / `--fixed_tau` CLI 参数 + 修改 `run_single_experiment` 签名
2. **先 dry_run 验证**：确认 r=0 的 inv_freq hash 与 `evq_cosh_inv_freq(64, 1.5)` 一致；r=32 与 tau=0 一致
3. **Pilot first**：先跑 r=0,14,16,32 四个关键值（~3.5h），看趋势再决定是否补全
4. **passkey_mix_ratio=0.10**：与现有 passkey mix 实验一致，同时拿 PPL + passkey
5. **eval_16k**：必须加，因为 PPL@16K 是核心指标
6. **不要动 R6000**
7. **结果保存到** `/root/autodl-tmp/evq_phase12_r_sweep/`
8. **注意 inv_freq 连续性**：hybrid_evq_inv_freq 在 r 处有一个 "接缝"，geo[r-1] 和 evq_part[0] 之间可能有跳变。打印出来检查。跳变不影响功能（模型可以学），但如果太大（>10×），可能影响训练稳定性
