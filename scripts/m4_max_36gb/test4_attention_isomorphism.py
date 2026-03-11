#!/usr/bin/env python3
"""
Test 4: Attention Bias 同构验证 — EVQ tradeoff 是否在 attention 距离维度上也成立？

=== 核心假说 ===
EVQ 在 RoPE 频率维度上的最优分配是 cosh 型密度（压缩高频冗余，扩大低频间距）。
如果 attention 的距离维度存在同构 tradeoff，那么 GPT-2 学到的 effective attention bias
也应该呈现类似结构：近距离（"高频"）被压缩/稀疏，远距离（"低频"）获得更高分辨率。

=== 实验设计（4 层，零训练成本） ===

Layer 1: Effective bias 提取
  - 从 GPT-2 attention D(Δ) 提取 effective bias B(Δ) = log A(Δ)
  - 在 log-distance 坐标下观察 B 的形状

Layer 2: 函数族拟合竞赛
  - 拟合 5 种模型：linear (ALiBi), quadratic, power-law, exp-decay, cosh
  - 如果 cosh 或 power-law 显著赢 linear，说明非线性结构存在

Layer 3: Per-head 分析
  - 按 head 类型（local/mixed/global）分组
  - 检查不同 head 类型的 effective bias 是否有不同的 "τ_attn"

Layer 4: 同构映射
  - 将 attention bias 参数映射到 EVQ 参数空间
  - 计算 correlation：τ_attn vs τ_EVQ 是否有结构性关系

=== 依赖 ===
- numpy, scipy, matplotlib
- 已有数据: results/m4_max_36gb/D_attention_per_head.npy (GPT-2 test3 输出)
- 如果没有数据，会自动调用 GPT-2 提取（需要 transformers, datasets）

=== 运行 ===
    python test4_attention_isomorphism.py                 # 全部
    python test4_attention_isomorphism.py --no-plot        # 无图（纯 CI 模式）
    python test4_attention_isomorphism.py --extract-fresh   # 重新从 GPT-2 提取

输出:
    results/m4_max_36gb/test4_attention_isomorphism_results.json
    results/m4_max_36gb/test4_*.png (4 张图)
"""

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "m4_max_36gb"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ============================================================
# 数据加载 / 提取
# ============================================================

def load_or_extract_attention(extract_fresh: bool = False) -> dict:
    """加载已有的 GPT-2 attention D(Δ) 数据，或从头提取。"""
    per_head_path = RESULTS_DIR / "D_attention_per_head.npy"
    global_path = RESULTS_DIR / "D_attention_global.npy"
    per_layer_path = RESULTS_DIR / "D_attention_per_layer.npy"

    if not extract_fresh and per_head_path.exists():
        print(f"Loading cached attention data from {RESULTS_DIR}")
        D_per_head = np.load(per_head_path)    # (12, 12, max_delta)
        D_global = np.load(global_path)         # (max_delta,)
        D_per_layer = np.load(per_layer_path)   # (12, max_delta)
        print(f"  D_per_head: {D_per_head.shape}")
        print(f"  D_global:   {D_global.shape}")
        return {
            "D_per_head": D_per_head,
            "D_global": D_global,
            "D_per_layer": D_per_layer,
        }

    # 没有缓存，从 GPT-2 提取
    print("No cached data found. Extracting from GPT-2 (requires transformers, datasets)...")
    import torch
    from transformers import GPT2LMHeadModel, AutoTokenizer
    from datasets import load_dataset

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager")
    model.eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    texts = [t for t in ds["text"] if len(t.strip()) > 200]

    n_seqs, seq_len = 100, 1024
    n_layers, n_heads = 12, 12
    max_delta = seq_len - 1

    # Tokenize
    all_ids = []
    buf = []
    for t in texts:
        buf.append(t)
        if len(buf) >= 50:
            enc = tokenizer(buf, truncation=False, add_special_tokens=False)
            for ids in enc["input_ids"]:
                all_ids.extend(ids)
            buf = []
            if len(all_ids) >= n_seqs * seq_len * 2:
                break
    if buf:
        enc = tokenizer(buf, truncation=False, add_special_tokens=False)
        for ids in enc["input_ids"]:
            all_ids.extend(ids)

    seqs = []
    for i in range(0, len(all_ids) - seq_len, seq_len):
        seqs.append(all_ids[i:i+seq_len])
        if len(seqs) >= n_seqs:
            break

    D_per_head = np.zeros((n_layers, n_heads, max_delta), dtype=np.float64)

    print(f"Running {len(seqs)} sequences through GPT-2...")
    with torch.no_grad():
        for si, seq in enumerate(seqs):
            input_ids = torch.tensor([seq], device=device)
            out = model(input_ids, output_attentions=True)
            for layer_idx, attn in enumerate(out.attentions):
                # attn: (1, n_heads, seq_len, seq_len)
                A = attn[0].float().cpu().numpy()  # (n_heads, seq_len, seq_len)
                for h in range(n_heads):
                    for delta in range(1, max_delta + 1):
                        # 对角线平均: A[i, i-delta] for i >= delta
                        diag = np.diagonal(A[h], offset=-delta)
                        D_per_head[layer_idx, h, delta - 1] += diag.sum()
            if (si + 1) % 20 == 0:
                print(f"  {si+1}/{len(seqs)} sequences done")

    # Normalize
    for l in range(n_layers):
        for h in range(n_heads):
            total = D_per_head[l, h].sum()
            if total > 0:
                D_per_head[l, h] /= total

    D_per_layer = D_per_head.mean(axis=1)
    D_global = D_per_head.mean(axis=(0, 1))
    D_global /= D_global.sum()

    np.save(per_head_path, D_per_head)
    np.save(global_path, D_global)
    np.save(per_layer_path, D_per_layer)
    print(f"Saved to {RESULTS_DIR}")

    return {"D_per_head": D_per_head, "D_global": D_global, "D_per_layer": D_per_layer}


# ============================================================
# Layer 1: Effective bias 提取与可视化
# ============================================================

def layer1_effective_bias(D_global: np.ndarray) -> dict:
    """
    从 D(Δ) 提取 effective attention bias B(Δ) = log D(Δ)。
    在 log-Δ 坐标下观察形状。
    """
    print("\n" + "=" * 70)
    print("LAYER 1: Effective Attention Bias B(Δ) = log D(Δ)")
    print("=" * 70)

    max_delta = len(D_global)
    deltas = np.arange(1, max_delta + 1, dtype=np.float64)

    # Avoid log(0)
    D_safe = np.maximum(D_global, 1e-20)
    B = np.log(D_safe)

    # 在 log-Δ 坐标下的斜率变化
    log_deltas = np.log(deltas)

    # 分段斜率
    segments = [
        ("short",  1, 10),
        ("mid",    10, 100),
        ("long",   100, 500),
        ("tail",   500, max_delta),
    ]

    slopes = {}
    for name, lo, hi in segments:
        mask = (deltas >= lo) & (deltas <= hi)
        if mask.sum() < 3:
            continue
        ld = log_deltas[mask]
        lb = B[mask]
        valid = np.isfinite(lb)
        if valid.sum() < 3:
            slopes[name] = {"slope": float("nan"), "n_points": 0}
            continue
        p = np.polyfit(ld[valid], lb[valid], 1)
        slopes[name] = {"slope": round(float(p[0]), 4), "n_points": int(valid.sum())}

    print("\n  Segment slopes (d log D / d log Δ):")
    for name, info in slopes.items():
        print(f"    {name:8s}: slope = {info['slope']:+.4f}  (n={info['n_points']})")

    # 关键指标: 短距离 vs 长距离的 D 比值（衡量压缩程度）
    D_short = D_global[:10].mean()
    D_long = D_global[100:500].mean() if max_delta > 500 else D_global[100:].mean()
    compression_ratio = D_short / D_long if D_long > 0 else float("inf")

    print(f"\n  D_short (Δ=1-10) mean:    {D_short:.6e}")
    print(f"  D_long  (Δ=100-500) mean: {D_long:.6e}")
    print(f"  Compression ratio:        {compression_ratio:.1f}×")
    print(f"  → GPT-2 关注近距离 {compression_ratio:.0f}× 多于远距离")

    return {
        "slopes": slopes,
        "D_short": float(D_short),
        "D_long": float(D_long),
        "compression_ratio": float(compression_ratio),
        "B_values": B.tolist(),
    }


# ============================================================
# Layer 2: 函数族拟合竞赛
# ============================================================

def _fit_linear(deltas, B):
    """B(Δ) = a - b*Δ  (ALiBi 风格)"""
    valid = np.isfinite(B)
    d, b = deltas[valid], B[valid]
    A = np.column_stack([np.ones_like(d), -d])
    try:
        coeffs, res, _, _ = np.linalg.lstsq(A, b, rcond=None)
        pred = A @ coeffs
        ss_res = np.sum((b - pred) ** 2)
        ss_tot = np.sum((b - b.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return {"name": "linear (ALiBi)", "r2": r2, "params": {"a": coeffs[0], "b": coeffs[1]},
                "pred_full": coeffs[0] - coeffs[1] * deltas}
    except Exception:
        return {"name": "linear (ALiBi)", "r2": 0, "params": {}, "pred_full": np.zeros_like(deltas)}


def _fit_quadratic(deltas, B):
    """B(Δ) = a - b*Δ - c*Δ²"""
    valid = np.isfinite(B)
    d, b = deltas[valid], B[valid]
    A = np.column_stack([np.ones_like(d), -d, -d**2])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        pred = A @ coeffs
        ss_res = np.sum((b - pred) ** 2)
        ss_tot = np.sum((b - b.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return {"name": "quadratic", "r2": r2,
                "params": {"a": coeffs[0], "b": coeffs[1], "c": coeffs[2]},
                "pred_full": coeffs[0] - coeffs[1]*deltas - coeffs[2]*deltas**2}
    except Exception:
        return {"name": "quadratic", "r2": 0, "params": {}, "pred_full": np.zeros_like(deltas)}


def _fit_power_law(deltas, B):
    """B(Δ) = a - b*log(Δ)  即 D(Δ) ∝ Δ^{-b}"""
    valid = np.isfinite(B)
    d, b = deltas[valid], B[valid]
    log_d = np.log(d)
    A = np.column_stack([np.ones_like(log_d), -log_d])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        pred = A @ coeffs
        ss_res = np.sum((b - pred) ** 2)
        ss_tot = np.sum((b - b.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return {"name": "power-law", "r2": r2,
                "params": {"a": coeffs[0], "alpha": coeffs[1]},
                "pred_full": coeffs[0] - coeffs[1]*np.log(deltas)}
    except Exception:
        return {"name": "power-law", "r2": 0, "params": {}, "pred_full": np.zeros_like(deltas)}


def _fit_exp_decay(deltas, B):
    """B(Δ) = a - b*Δ^c  with c ≈ 0.5-1.0 (stretched exponential feel)
    简化: B(Δ) = a + b*exp(-Δ/λ), 用 log-space grid search λ"""
    valid = np.isfinite(B)
    d, bv = deltas[valid], B[valid]
    best_r2, best_params, best_pred = -1, {}, np.zeros_like(deltas)

    for lam in [5, 10, 20, 50, 100, 200, 500]:
        exp_d = np.exp(-d / lam)
        A = np.column_stack([np.ones_like(d), exp_d])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, bv, rcond=None)
            pred = A @ coeffs
            ss_res = np.sum((bv - pred) ** 2)
            ss_tot = np.sum((bv - bv.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            if r2 > best_r2:
                best_r2 = r2
                best_params = {"a": coeffs[0], "b": coeffs[1], "lambda": lam}
                best_pred = coeffs[0] + coeffs[1] * np.exp(-deltas / lam)
        except Exception:
            pass

    return {"name": "exp-decay", "r2": best_r2, "params": best_params, "pred_full": best_pred}


def _fit_cosh(deltas, B):
    """
    B(Δ) = a - b*cosh(τ * Δ/L)  — EVQ 同构族

    cosh 在小 τ 时退化为 quadratic，大 τ 时退化为 exponential。
    如果 cosh 赢 linear 和 exp，说明 EVQ 的 tradeoff 结构存在。

    用 grid search τ (因为非线性, lstsq 不能直接解)
    """
    valid = np.isfinite(B)
    d, bv = deltas[valid], B[valid]
    L = float(deltas[-1])  # max distance
    best_r2, best_params, best_pred = -1, {}, np.zeros_like(deltas)

    for tau in np.concatenate([
        np.linspace(0.1, 2.0, 20),
        np.linspace(2.0, 10.0, 20),
        np.linspace(10.0, 50.0, 10),
    ]):
        cosh_d = np.cosh(tau * d / L)
        A = np.column_stack([np.ones_like(d), -cosh_d])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, bv, rcond=None)
            pred = A @ coeffs
            ss_res = np.sum((bv - pred) ** 2)
            ss_tot = np.sum((bv - bv.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            if r2 > best_r2:
                best_r2 = r2
                best_params = {"a": coeffs[0], "b": coeffs[1], "tau": tau}
                best_pred = coeffs[0] - coeffs[1] * np.cosh(tau * deltas / L)
        except Exception:
            pass

    return {"name": "cosh (EVQ-iso)", "r2": best_r2, "params": best_params, "pred_full": best_pred}


def _fit_log_cosh(deltas, B):
    """
    B(Δ) = a - b*log(cosh(τ * Δ/L))  — 更柔和的 cosh 变体

    log(cosh(x)) ≈ x²/2 for small x, ≈ |x| for large x
    这是 Huber loss 的光滑版本，可能更好地捕捉 attention 的 soft decay
    """
    valid = np.isfinite(B)
    d, bv = deltas[valid], B[valid]
    L = float(deltas[-1])
    best_r2, best_params, best_pred = -1, {}, np.zeros_like(deltas)

    for tau in np.concatenate([
        np.linspace(0.5, 5.0, 20),
        np.linspace(5.0, 30.0, 15),
    ]):
        lcosh_d = np.log(np.cosh(tau * d / L) + 1e-30)
        A = np.column_stack([np.ones_like(d), -lcosh_d])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, bv, rcond=None)
            pred = A @ coeffs
            ss_res = np.sum((bv - pred) ** 2)
            ss_tot = np.sum((bv - bv.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            if r2 > best_r2:
                best_r2 = r2
                best_params = {"a": coeffs[0], "b": coeffs[1], "tau": tau}
                best_pred = coeffs[0] - coeffs[1] * np.log(np.cosh(tau * deltas / L) + 1e-30)
        except Exception:
            pass

    return {"name": "log-cosh", "r2": best_r2, "params": best_params, "pred_full": best_pred}


def layer2_fitting_contest(D_global: np.ndarray) -> dict:
    """拟合 6 种函数族到 effective bias，比较 R²。"""
    print("\n" + "=" * 70)
    print("LAYER 2: Function Family Fitting Contest")
    print("=" * 70)

    max_delta = len(D_global)
    deltas = np.arange(1, max_delta + 1, dtype=np.float64)
    D_safe = np.maximum(D_global, 1e-20)
    B = np.log(D_safe)

    fitters = [_fit_linear, _fit_quadratic, _fit_power_law, _fit_exp_decay, _fit_cosh, _fit_log_cosh]
    results = []

    for fitter in fitters:
        res = fitter(deltas, B)
        results.append(res)
        params_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                              for k, v in res["params"].items())
        print(f"  {res['name']:20s}  R² = {res['r2']:.6f}  ({params_str})")

    # 排名
    results.sort(key=lambda x: x["r2"], reverse=True)
    print(f"\n  🏆 Winner: {results[0]['name']} (R² = {results[0]['r2']:.6f})")
    print(f"     vs ALiBi linear: ΔR² = {results[0]['r2'] - [r for r in results if 'linear' in r['name']][0]['r2']:+.6f}")

    # 关键判断
    cosh_r2 = [r for r in results if "cosh" in r["name"] and "log" not in r["name"]][0]["r2"]
    linear_r2 = [r for r in results if "linear" in r["name"]][0]["r2"]
    power_r2 = [r for r in results if "power" in r["name"]][0]["r2"]

    print(f"\n  === 同构判断 ===")
    print(f"  cosh R²   = {cosh_r2:.6f}")
    print(f"  linear R² = {linear_r2:.6f}")
    print(f"  power R²  = {power_r2:.6f}")

    if cosh_r2 > linear_r2 + 0.01:
        print(f"  ✅ cosh 显著优于 linear (Δ={cosh_r2-linear_r2:+.4f})")
        print(f"     → 支持 EVQ 同构假说：attention bias 具有非线性 cosh 结构")
    elif cosh_r2 > linear_r2:
        print(f"  ⚠️ cosh 略优于 linear (Δ={cosh_r2-linear_r2:+.4f})，证据不够强")
    else:
        print(f"  ❌ cosh 不优于 linear (Δ={cosh_r2-linear_r2:+.4f})，同构假说不成立")

    return {
        "rankings": [{"name": r["name"], "r2": round(r["r2"], 6),
                       "params": {k: round(v, 6) if isinstance(v, float) else v
                                  for k, v in r["params"].items()}}
                      for r in results],
        "cosh_wins_linear": cosh_r2 > linear_r2,
        "delta_r2_cosh_vs_linear": round(cosh_r2 - linear_r2, 6),
    }


# ============================================================
# Layer 3: Per-head analysis
# ============================================================

def layer3_per_head(D_per_head: np.ndarray) -> dict:
    """
    按 head 分析 effective bias 的 cosh τ 和 power-law α。
    分类 head 类型，检查 τ_attn 的分布。
    """
    print("\n" + "=" * 70)
    print("LAYER 3: Per-Head τ_attn Distribution")
    print("=" * 70)

    n_layers, n_heads, max_delta = D_per_head.shape
    deltas = np.arange(1, max_delta + 1, dtype=np.float64)

    head_results = []

    for l in range(n_layers):
        for h in range(n_heads):
            D = D_per_head[l, h]
            D_safe = np.maximum(D, 1e-20)
            B = np.log(D_safe)

            # Power-law fit
            pl = _fit_power_law(deltas, B)
            alpha = pl["params"].get("alpha", 0)

            # Cosh fit
            ch = _fit_cosh(deltas, B)
            tau = ch["params"].get("tau", 0)

            # Log-cosh fit
            lch = _fit_log_cosh(deltas, B)
            tau_lc = lch["params"].get("tau", 0)

            # Linear fit
            lin = _fit_linear(deltas, B)

            # Head type classification
            if alpha > 1.5:
                htype = "very_local"
            elif alpha > 0.8:
                htype = "local"
            elif alpha > 0.3:
                htype = "mixed"
            else:
                htype = "global"

            head_results.append({
                "layer": l, "head": h,
                "alpha": round(alpha, 4),
                "tau_cosh": round(tau, 4),
                "tau_logcosh": round(tau_lc, 4),
                "r2_power": round(pl["r2"], 4),
                "r2_cosh": round(ch["r2"], 4),
                "r2_logcosh": round(lch["r2"], 4),
                "r2_linear": round(lin["r2"], 4),
                "type": htype,
            })

    # 统计
    alphas = np.array([h["alpha"] for h in head_results])
    taus = np.array([h["tau_cosh"] for h in head_results])
    taus_lc = np.array([h["tau_logcosh"] for h in head_results])
    r2_cosh_all = np.array([h["r2_cosh"] for h in head_results])
    r2_lin_all = np.array([h["r2_linear"] for h in head_results])

    # 多少 head 的 cosh > linear
    n_cosh_wins = int(np.sum(r2_cosh_all > r2_lin_all))
    n_total = len(head_results)

    print(f"\n  Per-head statistics (n={n_total}):")
    print(f"    α (power-law):   mean={alphas.mean():.3f}, median={np.median(alphas):.3f}, std={alphas.std():.3f}")
    print(f"    τ_cosh:          mean={taus.mean():.3f}, median={np.median(taus):.3f}, std={taus.std():.3f}")
    print(f"    τ_log-cosh:      mean={taus_lc.mean():.3f}, median={np.median(taus_lc):.3f}, std={taus_lc.std():.3f}")
    print(f"    R²_cosh > R²_linear: {n_cosh_wins}/{n_total} heads ({100*n_cosh_wins/n_total:.1f}%)")

    # 按 type 分组
    types = {}
    for h in head_results:
        t = h["type"]
        if t not in types:
            types[t] = []
        types[t].append(h)

    print(f"\n  Head type breakdown:")
    for t in ["very_local", "local", "mixed", "global"]:
        if t not in types:
            continue
        hs = types[t]
        ts = np.array([h["tau_cosh"] for h in hs])
        als = np.array([h["alpha"] for h in hs])
        r2c = np.array([h["r2_cosh"] for h in hs])
        r2l = np.array([h["r2_linear"] for h in hs])
        wins = int(np.sum(r2c > r2l))
        print(f"    {t:12s}: n={len(hs):3d}, α_mean={als.mean():.3f}, "
              f"τ_mean={ts.mean():.2f}, cosh_wins={wins}/{len(hs)}")

    # Layer-wise τ 趋势
    print(f"\n  Layer-wise τ_cosh trend:")
    for l in range(n_layers):
        layer_taus = [h["tau_cosh"] for h in head_results if h["layer"] == l]
        layer_r2 = [h["r2_cosh"] - h["r2_linear"] for h in head_results if h["layer"] == l]
        print(f"    Layer {l:2d}: τ_mean={np.mean(layer_taus):.2f}, "
              f"Δ(R²_cosh-R²_lin)_mean={np.mean(layer_r2):+.4f}")

    return {
        "n_heads": n_total,
        "n_cosh_wins": n_cosh_wins,
        "pct_cosh_wins": round(100 * n_cosh_wins / n_total, 1),
        "alpha_stats": {
            "mean": round(float(alphas.mean()), 4),
            "median": round(float(np.median(alphas)), 4),
            "std": round(float(alphas.std()), 4),
        },
        "tau_cosh_stats": {
            "mean": round(float(taus.mean()), 4),
            "median": round(float(np.median(taus)), 4),
            "std": round(float(taus.std()), 4),
        },
        "type_counts": {t: len(hs) for t, hs in types.items()},
        "per_head": head_results,
    }


# ============================================================
# Layer 4: 同构映射分析
# ============================================================

def layer4_isomorphism(layer2_res: dict, layer3_res: dict) -> dict:
    """
    将 attention 维度的发现映射回 EVQ 框架，量化同构程度。

    EVQ 的 waterbed：高频通道间距压缩 → 低频通道间距扩大
    Attention 的 waterbed：近距离 attention 密度高但冗余 → 远距离 attention 稀疏但关键

    如果同构成立：
    1. effective bias 的形状应该是 cosh-like（而非 linear）
    2. τ_attn 应该和 head 的"工作距离"正相关
    3. 不同 layer 的 τ_attn 可能有趋势（浅层 local → 深层 global）
    """
    print("\n" + "=" * 70)
    print("LAYER 4: Isomorphism Mapping Analysis")
    print("=" * 70)

    heads = layer3_res["per_head"]

    # 4.1 τ_attn vs α 的相关性
    alphas = np.array([h["alpha"] for h in heads])
    taus = np.array([h["tau_cosh"] for h in heads])

    # Pearson correlation
    valid = np.isfinite(alphas) & np.isfinite(taus)
    if valid.sum() > 5:
        corr = np.corrcoef(alphas[valid], taus[valid])[0, 1]
    else:
        corr = 0

    print(f"\n  4.1 τ_attn vs α correlation: r = {corr:.4f}")
    if abs(corr) > 0.5:
        print(f"      ✅ Strong correlation → α 和 τ 在同一结构上")
    elif abs(corr) > 0.3:
        print(f"      ⚠️ Moderate correlation")
    else:
        print(f"      ❌ Weak correlation")

    # 4.2 Layer depth vs τ 趋势
    n_layers = max(h["layer"] for h in heads) + 1
    layer_tau_means = []
    for l in range(n_layers):
        lt = [h["tau_cosh"] for h in heads if h["layer"] == l]
        layer_tau_means.append(np.mean(lt))

    # Linear trend in τ vs layer
    layers = np.arange(n_layers, dtype=np.float64)
    tau_arr = np.array(layer_tau_means)
    if len(layers) > 2:
        p = np.polyfit(layers, tau_arr, 1)
        trend_slope = p[0]
    else:
        trend_slope = 0

    print(f"\n  4.2 Layer depth → τ_attn trend: slope = {trend_slope:+.4f}")
    if trend_slope > 0.1:
        print(f"      ✅ τ increases with depth → deeper layers attend more globally")
        print(f"         这与 EVQ 的预测一致：深层需要更多长程信息 → 更大的 τ")
    elif trend_slope < -0.1:
        print(f"      ⚠️ τ decreases with depth → unexpected, deeper layers more local")
    else:
        print(f"      ≈ No clear depth trend in τ")

    # 4.3 Waterbed-like tradeoff
    # 如果把每个 head 的 D(Δ) 看作一个"资源分配"，
    # 短距离的 attention 占比 vs 长距离的 attention 占比应该呈负相关
    # （就像 EVQ 中高频压缩 → 低频扩大）
    print(f"\n  4.3 Waterbed tradeoff check:")
    print(f"      cosh 拟合赢 linear 的 head 比例: {layer3_res['pct_cosh_wins']}%")
    if layer3_res["pct_cosh_wins"] > 60:
        print(f"      ✅ 多数 head 的 attention bias 呈 cosh 型 → waterbed tradeoff 存在")
    elif layer3_res["pct_cosh_wins"] > 40:
        print(f"      ⚠️ 约一半 head 呈 cosh 型 → 部分 head 存在 waterbed")
    else:
        print(f"      ❌ 少数 head 呈 cosh 型 → waterbed 不普遍")

    # 4.4 综合同构评分
    scores = {
        "cosh_wins_pct": layer3_res["pct_cosh_wins"],
        "tau_alpha_corr": abs(corr) * 100,
        "layer_trend": min(abs(trend_slope) * 50, 100),  # normalize
        "r2_improvement": (layer2_res["rankings"][0]["r2"] -
                          [r for r in layer2_res["rankings"] if "linear" in r["name"]][0]["r2"]) * 100,
    }

    iso_score = (
        0.4 * min(scores["cosh_wins_pct"], 100) +
        0.2 * scores["tau_alpha_corr"] +
        0.2 * scores["layer_trend"] +
        0.2 * scores["r2_improvement"]
    )

    print(f"\n  === 同构综合评分 ===")
    print(f"  cosh 优势 (40%):     {scores['cosh_wins_pct']:.1f}%")
    print(f"  τ-α 相关 (20%):     {scores['tau_alpha_corr']:.1f}%")
    print(f"  Layer 趋势 (20%):   {scores['layer_trend']:.1f}%")
    print(f"  R² 提升 (20%):      {scores['r2_improvement']:.1f}%")
    print(f"  ────────────────────")
    print(f"  综合得分:            {iso_score:.1f}/100")

    if iso_score > 60:
        verdict = "STRONG_SUPPORT"
        print(f"\n  🟢 结论: 强支持同构假说 — attention bias 具有 EVQ-like cosh 结构")
    elif iso_score > 35:
        verdict = "MODERATE_SUPPORT"
        print(f"\n  🟡 结论: 中等支持 — 部分结构存在但不普遍")
    else:
        verdict = "WEAK_SUPPORT"
        print(f"\n  🔴 结论: 弱支持 — 同构假说证据不足")

    return {
        "tau_alpha_correlation": round(float(corr), 4),
        "layer_trend_slope": round(float(trend_slope), 4),
        "layer_tau_means": [round(float(t), 4) for t in layer_tau_means],
        "scores": {k: round(v, 2) for k, v in scores.items()},
        "iso_score": round(iso_score, 1),
        "verdict": verdict,
    }


# ============================================================
# Plotting
# ============================================================

def plot_results(D_global, D_per_head, layer1_res, layer2_res, layer3_res, layer4_res):
    """生成 4 张诊断图。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    max_delta = len(D_global)
    deltas = np.arange(1, max_delta + 1, dtype=np.float64)
    D_safe = np.maximum(D_global, 1e-20)
    B = np.log(D_safe)

    # --- Fig 1: Effective bias with fits ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(deltas[:500], B[:500], "k-", alpha=0.6, lw=0.8, label="B(Δ) = log D(Δ)")
    for rank_info in layer2_res["rankings"][:4]:
        name = rank_info["name"]
        # Reconstruct predictions
        if "linear" in name:
            p = rank_info["params"]
            pred = p["a"] - p["b"] * deltas[:500]
        elif "power" in name:
            p = rank_info["params"]
            pred = p["a"] - p["alpha"] * np.log(deltas[:500])
        elif "cosh" in name and "log" not in name:
            p = rank_info["params"]
            pred = p["a"] - p["b"] * np.cosh(p["tau"] * deltas[:500] / max_delta)
        elif "log-cosh" in name:
            p = rank_info["params"]
            pred = p["a"] - p["b"] * np.log(np.cosh(p["tau"] * deltas[:500] / max_delta) + 1e-30)
        else:
            continue
        ax.plot(deltas[:500], pred, "--", lw=1.2,
                label=f"{name} (R²={rank_info['r2']:.4f})")

    ax.set_xlabel("Distance Δ")
    ax.set_ylabel("B(Δ) = log D(Δ)")
    ax.set_title("Effective Attention Bias — Linear Scale")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(np.log10(deltas), B, "k-", alpha=0.6, lw=0.8, label="B(Δ)")
    ax.set_xlabel("log₁₀(Δ)")
    ax.set_ylabel("B(Δ)")
    ax.set_title("Effective Attention Bias — Log Distance Scale")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "test4_fig1_effective_bias.png", dpi=150)
    plt.close()
    print(f"  Saved: test4_fig1_effective_bias.png")

    # --- Fig 2: Per-head τ distribution ---
    heads = layer3_res["per_head"]
    taus = [h["tau_cosh"] for h in heads]
    alphas = [h["alpha"] for h in heads]
    layers_arr = [h["layer"] for h in heads]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax = axes[0]
    ax.hist(taus, bins=30, edgecolor="black", alpha=0.7)
    ax.set_xlabel("τ_cosh")
    ax.set_ylabel("Count")
    ax.set_title(f"Per-Head τ_cosh Distribution (n={len(taus)})")
    ax.axvline(np.median(taus), color="red", ls="--", label=f"median={np.median(taus):.2f}")
    ax.legend()

    ax = axes[1]
    colors = {"very_local": "red", "local": "orange", "mixed": "blue", "global": "green"}
    for h in heads:
        ax.scatter(h["alpha"], h["tau_cosh"], c=colors.get(h["type"], "gray"),
                   alpha=0.5, s=20, edgecolors="none")
    ax.set_xlabel("α (power-law)")
    ax.set_ylabel("τ_cosh")
    ax.set_title(f"α vs τ_cosh (r={layer4_res['tau_alpha_correlation']:.3f})")
    # Legend
    for t, c in colors.items():
        ax.scatter([], [], c=c, s=20, label=t)
    ax.legend(fontsize=8)

    ax = axes[2]
    tau_means = layer4_res["layer_tau_means"]
    ax.bar(range(len(tau_means)), tau_means, color="steelblue", edgecolor="black")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean τ_cosh")
    ax.set_title(f"Layer-wise τ (trend={layer4_res['layer_trend_slope']:+.3f})")

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "test4_fig2_per_head_tau.png", dpi=150)
    plt.close()
    print(f"  Saved: test4_fig2_per_head_tau.png")

    # --- Fig 3: R² comparison per head (cosh vs linear) ---
    fig, ax = plt.subplots(figsize=(8, 6))
    r2_cosh = [h["r2_cosh"] for h in heads]
    r2_lin = [h["r2_linear"] for h in heads]
    ax.scatter(r2_lin, r2_cosh, c=[colors.get(h["type"], "gray") for h in heads],
               alpha=0.6, s=25, edgecolors="none")
    lims = [min(min(r2_lin), min(r2_cosh)) - 0.02, 1.0]
    ax.plot(lims, lims, "k--", alpha=0.3, label="y=x")
    ax.set_xlabel("R² (linear / ALiBi)")
    ax.set_ylabel("R² (cosh / EVQ-iso)")
    ax.set_title(f"Per-Head: cosh vs linear (cosh wins {layer3_res['pct_cosh_wins']}%)")
    for t, c in colors.items():
        ax.scatter([], [], c=c, s=25, label=t)
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "test4_fig3_cosh_vs_linear.png", dpi=150)
    plt.close()
    print(f"  Saved: test4_fig3_cosh_vs_linear.png")

    # --- Fig 4: Isomorphism summary card ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")

    score = layer4_res["iso_score"]
    verdict = layer4_res["verdict"]
    color = {"STRONG_SUPPORT": "#2E7D32", "MODERATE_SUPPORT": "#F57F17",
             "WEAK_SUPPORT": "#C62828"}.get(verdict, "gray")

    text = f"""
EVQ ↔ Attention Isomorphism Test
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Model: GPT-2 125M (12L × 12H, d_head=64)
Data:  WikiText-103, 100 seqs × 1024 tokens

━━━ Fitting Contest (Global D(Δ)) ━━━
"""
    for r in layer2_res["rankings"]:
        marker = "🏆" if r == layer2_res["rankings"][0] else "  "
        text += f"  {marker} {r['name']:20s} R² = {r['r2']:.6f}\n"

    text += f"""
━━━ Per-Head Analysis ━━━
  Heads where cosh > linear: {layer3_res['pct_cosh_wins']}%
  τ-α correlation:           r = {layer4_res['tau_alpha_correlation']:.3f}
  Layer depth trend:          slope = {layer4_res['layer_trend_slope']:+.3f}

━━━ Isomorphism Score: {score:.1f}/100 ━━━
  Verdict: {verdict.replace('_', ' ')}
"""

    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontfamily="monospace",
            fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "test4_fig4_summary.png", dpi=150)
    plt.close()
    print(f"  Saved: test4_fig4_summary.png")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Test 4: Attention Bias Isomorphism")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting")
    parser.add_argument("--extract-fresh", action="store_true", help="Re-extract from GPT-2")
    args = parser.parse_args()

    print("=" * 70)
    print("TEST 4: EVQ ↔ Attention Isomorphism Verification")
    print("=" * 70)
    print("""
    核心假说: EVQ 在频率维度上的 cosh 型 tradeoff 在 attention 距离维度上同构。
    如果成立，GPT-2 学到的 effective attention bias 应该呈 cosh-like 衰减，
    而非 ALiBi 的线性衰减。
    """)

    # Load data
    data = load_or_extract_attention(extract_fresh=args.extract_fresh)

    # Layer 1
    l1 = layer1_effective_bias(data["D_global"])

    # Layer 2
    l2 = layer2_fitting_contest(data["D_global"])

    # Layer 3
    l3 = layer3_per_head(data["D_per_head"])

    # Layer 4
    l4 = layer4_isomorphism(l2, l3)

    # Plot
    if not args.no_plot:
        print("\n" + "=" * 70)
        print("PLOTTING")
        print("=" * 70)
        plot_results(data["D_global"], data["D_per_head"], l1, l2, l3, l4)

    # Save results
    results = {
        "test": "test4_attention_isomorphism",
        "date": "2026-03-11",
        "model": "GPT-2 125M",
        "layer1_effective_bias": {k: v for k, v in l1.items() if k != "B_values"},
        "layer2_fitting_contest": l2,
        "layer3_per_head_summary": {k: v for k, v in l3.items() if k != "per_head"},
        "layer3_per_head_detail": l3["per_head"],
        "layer4_isomorphism": l4,
    }

    save_path = RESULTS_DIR / "test4_attention_isomorphism_results.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved → {save_path}")

    # Final verdict
    print("\n" + "=" * 70)
    print(f"  FINAL VERDICT: {l4['verdict'].replace('_', ' ')}")
    print(f"  Isomorphism Score: {l4['iso_score']:.1f}/100")
    print("=" * 70)

    # 论文建议
    print("""
  === 论文使用建议 ===

  如果 score > 60 (STRONG):
    → Discussion 或 §5 加一段 "Attention as a Variational Resource"
    → 展示 Fig 1 + Fig 4 作为 "preliminary evidence for unified framework"
    → Future Work: "The cosh structure in learned attention suggests the
       EVQ variational principle may extend beyond PE to attention itself"

  如果 score 35-60 (MODERATE):
    → Future Work 一句话提及即可
    → "We observe suggestive structural similarities between EVQ's
       frequency tradeoff and learned attention patterns (Appendix X)"

  如果 score < 35 (WEAK):
    → 不放入论文。同构假说需要更多验证。
""")


if __name__ == "__main__":
    main()
