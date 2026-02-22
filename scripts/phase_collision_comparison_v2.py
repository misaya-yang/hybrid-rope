#!/usr/bin/env python
"""
Phase Collision Comparison V2: Fixed Design
=============================================
修正版：统一base + 正确的Power-law方向

Key fixes:
1. 统一base=10000，所有方法只用这个base
2. Power-law短距离优先（模拟真实LLM注意力）
3. 增加两种Power-law方向对比
"""

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

# ============ 频率生成函数（统一base=10000） ============

def geometric_freq(d: int, base: float = 10000.0) -> torch.Tensor:
    """Standard geometric RoPE: θ_i = base^(-2i/d)"""
    i = torch.arange(d // 2, dtype=torch.float64)
    freqs = base ** (-2.0 * i / float(d))
    return freqs


def ntk_aware_freq(d: int, base: float = 10000.0, scale: float = 4.0) -> torch.Tensor:
    """NTK-aware: 放大base来延长上下文（统一base=10000）"""
    # NTK-aware: effective_base = base * (scale ^ (d/(d-2)))
    scaled_base = base * (scale ** (d / (d - 2)))
    return geometric_freq(d, base=scaled_base)


def yarn_freq(d: int, base: float = 10000.0, scale: float = 4.0, orig_ctx: int = 4096) -> torch.Tensor:
    """YaRN: rope scaling factor（统一base=10000）"""
    freqs = geometric_freq(d, base)
    # YaRN使用linear interpolation，scale控制拉伸程度
    return freqs / scale


def longrope_freq(d: int, base: float = 10000.0, scale: float = 4.0) -> torch.Tensor:
    """LongRoPE: 非均匀缩放（统一base=10000）"""
    freqs = geometric_freq(d, base)
    # LongRoPE使用非线性缩放：高维压缩更多
    i = torch.arange(d // 2, dtype=torch.float64)
    scale_factor = 1.0 + (scale - 1.0) * (i / (d // 2)) ** 2
    return freqs / scale_factor


def hybrid_freq(d: int, base: float = 10000.0, theta_geo: float = 100000, split: float = 0.5) -> torch.Tensor:
    """Hybrid: 前半部分geometric, 后半部分更大theta（统一base=10000）"""
    n = d // 2
    split_idx = int(n * split)
    
    # 前半部分用原始base
    i_low = torch.arange(split_idx, dtype=torch.float64)
    freqs_low = base ** (-2.0 * i_low / float(d))
    
    # 后半部分用更大theta的base
    i_high = torch.arange(split_idx, n, dtype=torch.float64)
    freqs_high = theta_geo ** (-2.0 * i_high / float(d))
    
    return torch.cat([freqs_low, freqs_high])


def sigmoid_freq(d: int, base: float = 10000.0, k: float = 0.125, x0: float = 32.0) -> torch.Tensor:
    """Sigmoid: S形分布压缩高频（统一base=10000）"""
    n = d // 2
    i = torch.arange(n, dtype=torch.float64)
    k_t = torch.tensor(k, dtype=torch.float64)
    x0_t = torch.tensor(x0, dtype=torch.float64)
    
    def sigma(z):
        if not isinstance(z, torch.Tensor):
            z = torch.tensor(z, dtype=torch.float64)
        return 1.0 / (1.0 + torch.exp(-z))
    
    raw = sigma(k_t * (i - x0_t))
    raw_min = sigma(k_t * (torch.tensor(0.0, dtype=torch.float64) - x0_t))
    raw_max = sigma(k_t * (torch.tensor(float(n - 1), dtype=torch.float64) - x0_t))
    denom = raw_max - raw_min
    
    if torch.abs(denom).item() < 1e-18:
        return geometric_freq(d, base)
    
    s_tilde = (raw - raw_min) / denom
    return base ** (-s_tilde)


def compute_inv_freq_anchored(head_dim, theta=100000, anchor_factor=20.0, anchor_dim=16, slope=0.5, base=10000.0):
    """Anchored Sigmoid（统一base=10000）"""
    n = head_dim // 2
    
    # 基础几何频率
    i = torch.arange(n, dtype=torch.float64)
    geo_base = base ** (-2.0 * i / head_dim)
    geo_extended = (theta * anchor_factor) ** (-2.0 * i / head_dim)
    
    # sigmoid权重
    k = slope * 2.0 / n
    j0 = float(anchor_dim)
    sigmoid_weight = 1.0 / (1.0 + torch.exp(-k * (i - j0)))
    
    # 锚定低维度
    if anchor_dim > 0:
        sigmoid_weight[:anchor_dim] = 0.0
    
    # 混合
    freqs = geo_base * (1.0 - sigmoid_weight) + geo_extended * sigmoid_weight
    return freqs


# ============ Phase Collision 计算（修正权重） ============

def _sample_log_distances(L: int, num_samples: int, device: torch.device) -> torch.Tensor:
    # 减少采样点数以加快速度
    d = torch.unique(torch.logspace(0, math.log10(float(L)), min(num_samples, 1000), device=device).long())
    d = d[d > 0]
    return d


def phase_collision_score_v2(
    freqs: torch.Tensor,
    L: int,
    num_samples: int = 5000,
    device: torch.device | None = None,
) -> Tuple[float, Dict[str, float]]:
    """计算Phase Collision Score - 修正版权重"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    distances = _sample_log_distances(L=L, num_samples=num_samples, device=device)
    freqs_f32 = freqs.to(device=device, dtype=torch.float32)
    angles = distances.to(dtype=torch.float32).unsqueeze(-1) * freqs_f32.unsqueeze(0)
    inner = torch.cos(angles).mean(dim=-1)
    
    short_mask = distances <= 100
    mid_mask = (distances > 100) & (distances <= 10000)
    long_mask = distances > 10000
    
    score_short = inner[short_mask].abs().mean().item() if short_mask.any() else 0.0
    score_mid = inner[mid_mask].abs().mean().item() if mid_mask.any() else 0.0
    score_long = inner[long_mask].abs().mean().item() if long_mask.any() else 0.0
    
    # ============ 修正版权重 ============
    # Uniform: 等权重
    score_uniform = (score_short + score_mid + score_long) / 3.0
    
    # Power-law Short（短距离优先，模拟真实LLM注意力）: 短0.7 + 中0.2 + 长0.1
    score_powerlaw_short = 0.7 * score_short + 0.2 * score_mid + 0.1 * score_long
    
    # Power-law Long（长距离优先，模拟长上下文需求）: 短0.1 + 中0.2 + 长0.7
    score_powerlaw_long = 0.1 * score_short + 0.2 * score_mid + 0.7 * score_long
    
    # Bimodal: 短0.4 + 中0.2 + 长0.4
    score_bimodal = 0.4 * score_short + 0.2 * score_mid + 0.4 * score_long
    
    total = 0.2 * score_short + 0.3 * score_mid + 0.5 * score_long
    
    return float(total), {
        "short": float(score_short),
        "mid": float(score_mid),
        "long": float(score_long),
        "total": float(total),
        "uniform": float(score_uniform),
        "powerlaw_short": float(score_powerlaw_short),
        "powerlaw_long": float(score_powerlaw_long),
        "bimodal": float(score_bimodal),
    }


def phase_collision_curve(
    freqs: torch.Tensor,
    L: int,
    num_points: int = 2000,
    device: torch.device | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """计算Phase Collision曲线"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    distances = torch.unique(torch.logspace(math.log10(1.0), math.log10(float(L)), num_points, device=device).long())
    distances = distances[distances > 0]
    freqs_f32 = freqs.to(device=device, dtype=torch.float32)
    angles = distances.to(torch.float32).unsqueeze(-1) * freqs_f32.unsqueeze(0)
    collisions = torch.cos(angles).mean(dim=-1)
    
    return distances.detach().cpu().numpy(), collisions.detach().cpu().numpy()


# ============ 主实验 ============

def run_comparison_v2():
    """运行修正后的Phase Collision比较"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 配置
    d = 128  # 头维度 (LLaMA-3 8B用的是128)
    L = 16384  # 最大长度
    base = 10000.0  # 统一base！
    scale = 4.0  # 统一scale（对标L/4096）
    
    # 定义所有方法（统一base=10000）
    methods = {
        # 基线 - 标准几何
        "geo_base": lambda: geometric_freq(d, base=base),
        
        # NTK-aware variants（统一base+scale）
        "ntk_4x": lambda: ntk_aware_freq(d, base, scale=scale),
        
        # YaRN（统一base+scale）
        "yarn_4x": lambda: yarn_freq(d, base, scale=scale),
        
        # LongRoPE（统一base+scale）
        "longrope_4x": lambda: longrope_freq(d, base, scale=scale),
        
        # Hybrid variants
        "hybrid_50_100k": lambda: hybrid_freq(d, base, theta_geo=100000, split=0.5),
        "hybrid_50_500k": lambda: hybrid_freq(d, base, theta_geo=500000, split=0.5),
        
        # Sigmoid variants
        "sigmoid_v2": lambda: sigmoid_freq(d, base, k=0.125, x0=32.0),
        "sigmoid_k016": lambda: sigmoid_freq(d, base, k=0.16, x0=32.0),
        
        # Anchored Sigmoid
        "anchored_x10": lambda: compute_inv_freq_anchored(d, theta=100000, anchor_factor=10.0, anchor_dim=16, slope=0.5, base=base),
        "anchored_x20": lambda: compute_inv_freq_anchored(d, theta=100000, anchor_factor=20.0, anchor_dim=16, slope=0.5, base=base),
        "anchored_x20_dim0": lambda: compute_inv_freq_anchored(d, theta=100000, anchor_factor=20.0, anchor_dim=0, slope=0.5, base=base),
    }
    
    # 计算所有方法的Phase Collision Score
    results = {}
    print("\n" + "="*80)
    print("Phase Collision Score Comparison V2 (Fixed Design)")
    print("="*80)
    print(f"Config: d={d}, L={L}, base={base}, scale={scale}")
    print("="*80)
    
    for name, freq_fn in methods.items():
        freqs = freq_fn().to(device)
        score, breakdown = phase_collision_score_v2(freqs, L, device=device)
        results[name] = {
            "score": score,
            "breakdown": breakdown,
            "freqs": freqs.cpu(),
        }
        print(f"{name:20s}: total={score:.4f} (short={breakdown['short']:.3f}, mid={breakdown['mid']:.3f}, long={breakdown['long']:.3f})")
    
    # 不同D假设下的排名
    print("\n" + "="*80)
    print("Ranking under Different Distance Distribution D(Δ)")
    print("="*80)
    
    weight_schemes = ["uniform", "powerlaw_short", "powerlaw_long", "bimodal"]
    weight_names = [
        "Uniform (equal: 0.33/0.33/0.33)", 
        "Power-law Short (LLM attention: 0.7/0.2/0.1)",
        "Power-law Long (long-context: 0.1/0.2/0.7)",
        "Bimodal (local+global: 0.4/0.2/0.4)"
    ]
    
    for scheme, name in zip(weight_schemes, weight_names):
        print(f"\n{'='*60}")
        print(f"{name}")
        print(f"{'='*60}")
        sorted_methods = sorted(results.items(), key=lambda x: x[1]["breakdown"][scheme])
        for i, (mname, data) in enumerate(sorted_methods, 1):
            print(f"  {i:2d}. {mname:20s}: {data['breakdown'][scheme]:.4f}")
    
    # 保存结果
    output_dir = Path("results/phase_collision_comparison_v2")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存数值结果
    import json
    json_results = {}
    for name, data in results.items():
        json_results[name] = {
            "score": data["score"],
            "breakdown": data["breakdown"],
        }
    
    with open(output_dir / "scores.json", "w") as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to {output_dir / 'scores.json'}")
    
    return results


def plot_comparison(results: Dict):
    """绘制对比图"""
    try:
        import matplotlib.pyplot as plt
        
        output_dir = Path("results/phase_collision_comparison_v2")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 选择主要方法绘图
        main_methods = ["geo_base", "ntk_4x", "yarn_4x", "sigmoid_v2", "anchored_x20", "hybrid_50_100k"]
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 绘制Phase Collision曲线
        L = 16384
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for name in main_methods:
            if name in results:
                freqs = results[name]["freqs"].to(device)
                distances, collisions = phase_collision_curve(freqs, L, device=device)
                ax.plot(distances, collisions, label=name, alpha=0.8)
        
        ax.set_xlabel("Distance Δ")
        ax.set_ylabel("Phase Correlation (cos)")
        ax.set_title("Phase Collision Curves - Different RoPE Variants (Fixed Design)")
        ax.legend()
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "phase_collision_curves.png", dpi=150)
        print(f"Saved to {output_dir / 'phase_collision_curves.png'}")
        
        # 绘制柱状图 - 四种D假设
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        weight_schemes = ["uniform", "powerlaw_short", "powerlaw_long", "bimodal"]
        weight_names = ["Uniform", "Power-law (Short)", "Power-law (Long)", "Bimodal"]
        
        for ax, scheme, name in zip(axes.flat, weight_schemes, weight_names):
            scores_scheme = [(m, results[m]["breakdown"][scheme]) for m in results.keys()]
            scores_scheme.sort(key=lambda x: x[1])
            
            m_names = [x[0] for x in scores_scheme]
            m_scores = [x[1] for x in scores_scheme]
            
            colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(m_names)))
            ax.bar(range(len(m_names)), m_scores, color=colors)
            ax.set_xticks(range(len(m_names)))
            ax.set_xticklabels(m_names, rotation=45, ha="right", fontsize=9)
            ax.set_ylabel("Score (lower is better)")
            ax.set_title(f"D(Δ): {name}")
            ax.grid(True, alpha=0.3, axis="y")
            
            # 标注最优
            ax.axhline(y=m_scores[0], color='green', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(output_dir / "phase_collision_by_D_v2.png", dpi=150)
        print(f"Saved to {output_dir / 'phase_collision_by_D_v2.png'}")
        
    except ImportError:
        print("matplotlib not available, skipping plots")


if __name__ == "__main__":
    results = run_comparison_v2()
    plot_comparison(results)
    print("\nDone!")
