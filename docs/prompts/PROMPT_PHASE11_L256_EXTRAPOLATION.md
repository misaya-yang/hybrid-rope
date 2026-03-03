# Phase 11: L=256 Extrapolation Ratio Sweep — 2×到32×外推曲线

> **核心目标**: 训练 L=256，测试 512/1K/2K/4K/8K（2×~32×外推），画出"EVQ vs Geo 外推衰减曲线"
> **硬件**: 新开 5090 32GB（不动 R6000）
> **预计 GPU 时间**: ~4-6h
> **论文价值**: 补全"仅测了 8× 外推"攻击面 + 验证 τ* scaling law 在短 L 的预测

---

## 服务器信息

- SSH: `ssh -p <PORT> root@<HOST>`（开机后填写）
- 密码: （开机后填写）
- GPU: 5090 32GB
- 代码: 需要从本地同步 `hybrid-rope/` 到服务器

## 环境初始化（每次 SSH 必执行）

```bash
export PATH="/root/miniconda3/bin:$PATH"
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1
```

若 miniconda 不存在，需先装环境：
```bash
pip install torch transformers datasets tokenizers --break-system-packages
```

---

## 实验设计

### 核心参数

| 参数 | 值 | 原因 |
|------|-----|------|
| 模型 | 350M (hidden=1024, layers=24, heads=16, head_dim=64) | 与 5090 已有实验一致 |
| L_train | **256** | 极短训练 → 大外推倍率 |
| base | 500,000 | 与所有现有实验一致 |
| τ (Geo) | 0.0 | baseline |
| τ (EVQ) | **4.0** | 由 τ*=64/√256=4.0 预测 |
| τ (EVQ-low) | **2.0** | 消融：τ*预测偏差时的鲁棒性 |
| 数据 | 90% FineWeb-Edu + 10% passkey | 同时测 PPL + retrieval |
| tokens | 50M | 与 5090 验证实验一致 |
| seeds | 42, 137, 256 | 3-seed 验证 |

### 测试长度（6 个外推倍率）

| 长度 | 倍率 | PPL | Passkey |
|------|------|-----|---------|
| 256 | 1× (in-dist) | ✅ | ✅ |
| 512 | 2× | ✅ | ✅ |
| 1024 | 4× | ✅ | ✅ |
| 2048 | 8× | ✅ | ✅ |
| 4096 | 16× | ✅ | ✅（可能 OOM，加保护） |
| 8192 | 32× | ✅（可能 OOM） | ✅（可能 OOM） |

### Runs 总览（9 runs）

| Run | Method | τ | Seed | 预计时间 |
|-----|--------|---|------|----------|
| 1 | Geometric | 0.0 | 42 | ~20min |
| 2 | EVQ | 4.0 | 42 | ~20min |
| 3 | EVQ | 2.0 | 42 | ~20min |
| 4 | Geometric | 0.0 | 137 | ~20min |
| 5 | EVQ | 4.0 | 137 | ~20min |
| 6 | EVQ | 2.0 | 137 | ~20min |
| 7 | Geometric | 0.0 | 256 | ~20min |
| 8 | EVQ | 4.0 | 256 | ~20min |
| 9 | EVQ | 2.0 | 256 | ~20min |

L=256 训练 50M tokens = ~195K steps（batch_size=1, seq_len=256）。350M 在 5090 上 ~20min/run。

**总计**: 9 runs × ~20min = ~3h。加上数据下载和 eval 时间，总计 ~4-6h。

---

## 代码改动

### 核心: 在 `run_evq_sweep.py` 中添加 `--seq_len` 参数

需要在 `run_evq_sweep.py` 的 `main()` 函数中做以下改动：

#### 1. 添加 CLI 参数（在 argparse 区域）

```python
parser.add_argument("--seq_len", type=int, default=None,
                    help="Override training sequence length (default: from tier config)")
parser.add_argument("--train_tokens", type=int, default=None,
                    help="Override training token count (default: from tier config)")
```

#### 2. 在 main() 中应用 override

在 `cfg = TIER_CONFIGS[args.tier].copy()` 之后加：

```python
if args.seq_len is not None:
    cfg["seq_len"] = args.seq_len
    cfg["max_position_embeddings"] = args.seq_len
    # L=256 时 batch 可以开更大
    if args.seq_len <= 512:
        cfg["batch_size"] = min(32, cfg["batch_size"] * (2048 // args.seq_len))
    # eval lengths: 从 1× 到 32× 外推
    max_eval = min(args.seq_len * 32, 16384)  # cap at 16K 防 OOM
    cfg["eval_lengths"] = []
    L = args.seq_len
    while L <= max_eval:
        cfg["eval_lengths"].append(L)
        L *= 2
    print(f"  [seq_len override] L_train={cfg['seq_len']}, batch={cfg['batch_size']}, "
          f"eval={cfg['eval_lengths']}")

if args.train_tokens is not None:
    cfg["train_tokens"] = args.train_tokens
```

#### 3. passkey eval lengths 也要跟着改

在 eval_passkey_scratch.py 的调用处，确保 passkey eval 长度也使用同样的 eval_lengths，而不是硬编码。

当前代码（检查 `run_single_experiment` 或等效函数）可能有：
```python
pk_lengths = [2048, 4096, 8192]  # 硬编码
```

改为：
```python
pk_lengths = [L for L in cfg["eval_lengths"] if L >= cfg["seq_len"]]
```

---

## 执行命令

### Step 1: 同步代码到服务器

```bash
# 从本地
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='*.pt' \
    /path/to/hybrid-rope/ root@<HOST>:<PORT>:/root/autodl-tmp/dfrope/hybrid-rope/
```

### Step 2: 做上述代码改动

SSH 到服务器后修改 `run_evq_sweep.py`（或者在本地改好再 rsync）。

### Step 3: 执行实验

```bash
cd /root/autodl-tmp/dfrope/hybrid-rope/scripts/m4_evq_sweep

# 先跑 seed=42 pilot（确认一切正常）
mkdir -p /root/autodl-tmp/evq_phase11_L256

nohup python -u run_evq_sweep.py \
    --tier 350m \
    --taus 0.0,2.0,4.0 \
    --seeds 42 \
    --base 500000 \
    --seq_len 256 \
    --train_tokens 50000000 \
    --passkey_mix_ratio 0.10 \
    --work_dir /root/autodl-tmp/evq_phase11_L256 \
    > /root/autodl-tmp/evq_phase11_L256/run_seed42.log 2>&1 &

# 监控
tail -f /root/autodl-tmp/evq_phase11_L256/run_seed42.log
```

seed=42 三个 τ 跑完确认结果合理后，继续 seed=137,256：

```bash
nohup python -u run_evq_sweep.py \
    --tier 350m \
    --taus 0.0,2.0,4.0 \
    --seeds 137,256 \
    --base 500000 \
    --seq_len 256 \
    --train_tokens 50000000 \
    --passkey_mix_ratio 0.10 \
    --resume \
    --work_dir /root/autodl-tmp/evq_phase11_L256 \
    > /root/autodl-tmp/evq_phase11_L256/run_seed137_256.log 2>&1 &
```

### Step 4: 分析结果

跑完后读取所有 result.json，输出两张关键表：

#### 表 A: PPL 外推曲线（3-seed mean）

```
| 倍率 | 长度  | Geo    | EVQ τ=2.0 | EVQ τ=4.0 | Δ(τ=4.0) |
|------|-------|--------|-----------|-----------|----------|
| 1×   | 256   | xxx    | xxx       | xxx       | +x.x%   |
| 2×   | 512   | xxx    | xxx       | xxx       | -x.x%   |
| 4×   | 1024  | xxx    | xxx       | xxx       | -x.x%   |
| 8×   | 2048  | xxx    | xxx       | xxx       | -x.x%   |
| 16×  | 4096  | xxx    | xxx       | xxx       | -x.x%   |
| 32×  | 8192  | xxx    | xxx       | xxx       | -x.x%   |
```

#### 表 B: Passkey Retrieval 外推曲线（3-seed mean）

```
| 倍率 | 长度  | Geo   | EVQ τ=2.0 | EVQ τ=4.0 | Δ(τ=4.0) |
|------|-------|-------|-----------|-----------|----------|
| 1×   | 256   | xxx%  | xxx%      | xxx%      | +xpp    |
| 2×   | 512   | xxx%  | xxx%      | xxx%      | +xpp    |
| 4×   | 1024  | xxx%  | xxx%      | xxx%      | +xpp    |
| 8×   | 2048  | xxx%  | xxx%      | xxx%      | +xpp    |
| 16×  | 4096  | xxx%  | xxx%      | xxx%      | +xpp    |
| 32×  | 8192  | xxx%  | xxx%      | xxx%      | +xpp    |
```

### Step 5: 保存报告

```bash
# 保存完整报告
cat > /root/autodl-tmp/evq_phase11_L256/analysis_report.md << 'REPORT'
# Phase 11: L=256 Extrapolation Results
... (生成的表格和分析)
REPORT
```

---

## 成功标准

| 条件 | 判定 |
|------|------|
| PPL@256 (1×): Geo ≈ EVQ (差距 <2%) | ✅ 训练内等价，符合理论 |
| PPL@512+ (2×+): EVQ < Geo，且差距随倍率增大 | ✅ 核心假设验证 |
| PPL@8192 (32×): EVQ 改善 >15% | 🎉 强外推信号 |
| τ=4.0 优于 τ=2.0 | ✅ τ* scaling law 在 L=256 的验证 |
| Passkey@256: 两边都 100% | ✅ 训练内都学会 |
| Passkey@512-2048: EVQ > Geo | ✅ 外推区域 EVQ 优势 |
| 3-seed 方向一致 | ✅ 统计稳定 |

## 风险与备选

| 风险 | 对策 |
|------|------|
| L=256 FineWeb-Edu 训练效果差（文本太短） | 改用 TinyStories（本身短文本） |
| τ=4.0 太大导致训练崩溃 | 有 τ=2.0 备选 |
| 8192 eval OOM | 代码已有 OOM 保护，自动跳过 |
| Passkey 在 L=256 内训练不出来 | 10% mix + 256 token 够放一个短 passkey（key 只需 ~20 token） |
| Batch size 计算不合理 | L=256 时 350M 可以开 batch=16-32（显存 256/2048 = 1/8） |

## 论文用途

### 成功则新增的 Figure（最重要）

**Figure: Extrapolation Ratio Curve**
```
x-axis: Extrapolation ratio (1×, 2×, 4×, 8×, 16×, 32×)
y-axis: Relative PPL improvement (%) OR Passkey retrieval (%)
lines: Geometric, EVQ τ=2.0, EVQ τ=4.0 (τ*-predicted)
shading: 3-seed std
```

这张图同时验证：
1. EVQ 优势随外推倍率增大（核心价值）
2. τ* = 64/√256 = 4.0 是否确实最优
3. 2×到32× 完整覆盖（reviewer 无法挑刺"只测了 8×"）

### 论文新增 Table

```latex
\begin{table}[t]
\centering
\caption{Extrapolation ratio sweep (350M, L\_train=256, base=500K, 3-seed).
EVQ uses theory-predicted τ*=4.0. Improvement grows monotonically with ratio.}
\begin{tabular}{lccccccc}
\toprule
& 1× & 2× & 4× & 8× & 16× & 32× \\
\midrule
Geo PPL & ... \\
EVQ PPL & ... \\
Δ PPL & ... \\
\midrule
Geo PK & ... \\
EVQ PK & ... \\
Δ PK & ... \\
\bottomrule
\end{tabular}
\end{table}
```

### 论文 Contribution 补充

> "Across extrapolation ratios from 2× to 32× (L\_train=256), EVQ improvement grows monotonically, confirming that the frequency redistribution benefit scales with context extension demand."

---

## ⚠️ Claude Code 执行注意事项

1. **先做代码改动**：添加 `--seq_len` 和 `--train_tokens` 参数到 `run_evq_sweep.py`。这是 Phase 11 的前提
2. **先跑 pilot**：seed=42, τ=0.0 和 4.0 两个就够。确认 L=256 训练正常、eval 正常后再补全
3. **passkey 注意**：L=256 时 passkey sample 很短（~256 token），需要确认 `make_passkey_training_sample` 能处理这么短的序列。如果不行，降低 filler 长度
4. **eval passkey 长度**：eval 时 passkey 长度应该从 256 开始（1×），不是从 2048 开始
5. **不要动 R6000**：R6000 正在跑 Phase 9F，不要 SSH 也不要干扰
6. **结果保存**：所有 result.json 保存到 `/root/autodl-tmp/evq_phase11_L256/`，跑完生成 `analysis_report.md`
