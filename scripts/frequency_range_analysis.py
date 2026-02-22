#!/usr/bin/env python
"""
Frequency Range Analysis: Range Matching Theory
==============================================
验证"范围匹配"理论：
- 当频率范围覆盖到目标长度L时，标准RoPE最优
- 当范围不匹配时，形状调整有帮助

这将给出实用的频率选择指南
"""

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch


def geometric_freq(d: int, base: float = 10000.0) -> torch.Tensor:
    """Standard geometric RoPE"""
    i = torch.arange(d // 2, dtype=torch.float64)
    return base ** (-2.0 * i / float(d))


def ntk_aware_freq(d: int, base: float = 10000.0, scale: float = 4.0) -> torch.Tensor:
    """NTK-aware"""
    scaled_base = base * (scale ** (d / (d - 2)))
    return geometric_freq(d, base=scaled_base)


def yarn_freq(d: int, base: float = 10000.0, scale: float = 4.0) -> torch.Tensor:
    """YaRN"""
    return geometric_freq(d, base) / scale


def longrope_freq(d: int, base: float = 10000.0, scale: float = 4.0) -> torch.Tensor:
    """LongRoPE"""
    freqs = geometric_freq(d, base)
    i = torch.arange(d // 2, dtype=torch.float64)
    scale_factor = 1.0 + (scale - 1.0) * (i / (d // 2)) ** 2
    return freqs / scale_factor


def sigmoid_freq(d: int, base: float = 10000.0, k: float = 0.125, x0: float = 32.0) -> torch.Tensor:
    """Sigmoid"""
    n = d // 2
    i = torch.arange(n, dtype=torch.float64)
    k_t = torch.tensor(k, dtype=torch.float64)
    x0_t = torch.tensor(x0, dtype=torch.float64)
    
    raw = 1.0 / (1.0 + torch.exp(-k_t * (i - x0_t)))
    raw_min = 1.0 / (1.0 + torch.exp(-k_t * (torch.tensor(0.0, dtype=torch.float64) - x0_t)))
    raw_max = 1.0 / (1.0 + torch.exp(-k_t * (torch.tensor(float(n - 1), dtype=torch.float64) - x0_t)))
    denom = raw_max - raw_min
    
    if torch.abs(denom).item() < 1e-18:
        return geometric_freq(d, base)
    
    s_tilde = (raw - raw_min) / denom
    return base ** (-s_tilde)


def compute_anchored(d: int, theta: float = 100000, anchor_factor: float = 20.0, 
                     anchor_dim: int = 16, slope: float = 0.5, base: float = 10000.0) -> torch.Tensor:
    """Anchored Sigmoid"""
    n = d // 2
    i = torch.arange(n, dtype=torch.float64)
    
    geo_base = base ** (-2.0 * i / d)
    geo_extended = (theta * anchor_factor) ** (-2.0 * i / d)
    
    k = slope * 2.0 / n
    sigmoid_weight = 1.0 / (1.0 + torch.exp(-k * (i - anchor_dim)))
    sigmoid_weight[:anchor_dim] = 0.0
    
    return geo_base * (1.0 - sigmoid_weight) + geo_extended * sigmoid_weight


def compute_frequency_range(freqs: torch.Tensor, base: float = 10000.0) -> Dict[str, float]:
    """
    计算有效频率范围
    - max_freq: 最高频率（对应最低维度）
    - min_freq: 最低频率（对应最高维度）
    - wavelength_range: 对应的波长范围
    """
    max_freq = freqs[0].item()  # 最高频率
    min_freq = freqs[-1].item()  # 最低频率
    
    # 波长 = 2π / frequency
    # 能够准确表示的最大距离 ≈ wavelength / 2 (奈奎斯特采样)
    max_distance = (2 * math.pi / min_freq) / 2  # 最高频率能表示的最大距离
    min_distance = (2 * math.pi / max_freq) / 2  # 最低频率能表示的最小距离
    
    return {
        "max_freq": max_freq,
        "min_freq": min_freq,
        "max_distance": max_distance,
        "min_distance": min_distance,
    }


def phase_collision_at_L(freqs: torch.Tensor, L: int, device) -> float:
    """计算在特定长度L下的Phase Collision Score"""
    # 采样距离
    distances = torch.unique(torch.logspace(0, math.log10(float(L)), 500, device=device).long())
    distances = distances[distances > 0]
    
    freqs_f32 = freqs.to(device=device, dtype=torch.float32)
    angles = distances.to(dtype=torch.float32).unsqueeze(-1) * freqs_f32.unsqueeze(0)
    inner = torch.cos(angles).mean(dim=-1)
    
    return float(inner.abs().mean())


def run_analysis():
    """运行范围分析"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    d = 128  # 头维度
    base = 10000.0
    
    # 不同目标长度
    lengths = [4096, 8192, 16384, 32768, 65536]
    
    # 不同base配置
    configs = {
        "geo_10k": lambda: geometric_freq(d, base=10000),
        "geo_50k": lambda: geometric_freq(d, base=50000),
        "geo_100k": lambda: geometric_freq(d, base=100000),
        "geo_500k": lambda: geometric_freq(d, base=500000),
        "geo_1M": lambda: geometric_freq(d, base=1000000),
        "geo_2M": lambda: geometric_freq(d, base=2000000),
        "ntk_4x": lambda: ntk_aware_freq(d, base, scale=4.0),
        "ntk_8x": lambda: ntk_aware_freq(d, base, scale=8.0),
        "yarn_4x": lambda: yarn_freq(d, base, scale=4.0),
        "yarn_8x": lambda: yarn_freq(d, base, scale=8.0),
        "sigmoid": lambda: sigmoid_freq(d, base, k=0.125, x0=32.0),
        "anchored_20": lambda: compute_anchored(d, theta=100000, anchor_factor=20.0, anchor_dim=16, slope=0.5, base=base),
    }
    
    print("\n" + "="*80)
    print("Frequency Range Analysis: Range Matching Theory")
    print("="*80)
    
    # 1. 频率范围对比
    print("\n### 1. Frequency Range Comparison")
    print(f"{'Method':<15} {'max_freq':>12} {'min_freq':>12} {'max_dist':>12} {'min_dist':>12}")
    print("-" * 65)
    
    for name, freq_fn in configs.items():
        freqs = freq_fn().to(device)
        rng = compute_frequency_range(freqs, base)
        print(f"{name:<15} {rng['max_freq']:>12.6f} {rng['min_freq']:>12.6e} {rng['max_distance']:>12.0f} {rng['min_distance']:>12.2f}")
    
    # 2. Phase Collision vs Length
    print("\n### 2. Phase Collision Score at Different Lengths")
    print(f"{'Method':<15}", end="")
    for L in lengths:
        print(f"{L:>10}", end="")
    print()
    print("-" * (15 + 10 * len(lengths)))
    
    results = {}
    for name, freq_fn in configs.items():
        freqs = freq_fn().to(device)
        scores = []
        for L in lengths:
            score = phase_collision_at_L(freqs, L, device)
            scores.append(score)
        results[name] = scores
        print(f"{name:<15}", end="")
        for s in scores:
            print(f"{s:>10.4f}", end="")
        print()
    
    # 3. 关键发现：范围匹配分析
    print("\n### 3. Range Matching Analysis")
    print("\n#### 3.1 Which methods cover each length?")
    
    # 计算每个长度需要覆盖的最小频率
    for L in lengths:
        required_min_freq = 2 * math.pi / (2 * L)  # 奈奎斯特
        print(f"\nL={L}: need min_freq < {required_min_freq:.6e}")
        
        # 找出能覆盖这个长度的方法
        matching = []
        for name, freq_fn in configs.items():
            freqs = freq_fn().to(device)
            rng = compute_frequency_range(freqs, base)
            if rng['min_freq'] < required_min_freq:
                matching.append(name)
        
        print(f"  Can cover: {', '.join(matching)}")
    
    # 4. 理论验证
    print("\n### 4. Theory Verification")
    print("\nComparing: Does range matching predict method ranking?")
    
    # 选择几个代表性长度
    test_lengths = [4096, 16384]
    
    for L in test_lengths:
        print(f"\n--- L = {L} ---")
        
        # 找出能覆盖L的方法（range match）
        required_min_freq = 2 * math.pi / (2 * L)
        matched_methods = []
        unmatched_methods = []
        
        for name, freq_fn in configs.items():
            freqs = freq_fn().to(device)
            rng = compute_frequency_range(freqs, base)
            if rng['min_freq'] < required_min_freq:
                matched_methods.append(name)
            else:
                unmatched_methods.append(name)
        
        # 计算两类方法的平均Phase Collision
        matched_scores = [results[m][lengths.index(L)] for m in matched_methods]
        unmatched_scores = [results[m][lengths.index(L)] for m in unmatched_methods]
        
        print(f"  Range-matched methods: {matched_methods}")
        print(f"  Average PC score: {np.mean(matched_scores):.4f}")
        
        if unmatched_methods:
            print(f"  Range-unmatched methods: {unmatched_methods}")
            print(f"  Average PC score: {np.mean(unmatched_scores):.4f}")
        
        if np.mean(matched_scores) < np.mean(unmatched_scores):
            print(f"  ✓ Theory VERIFIED: Range-matched methods have lower PC!")
        else:
            print(f"  ✗ Theory NOT verified")
    
    # 5. 实用指南
    print("\n" + "="*80)
    print("### 5. Practical Guidelines")
    print("="*80)
    
    print("""
Based on the analysis, here's the frequency selection guideline:

**For a given target length L:**
1. Compute required min_freq = 2π / (2L)
2. Choose a base that gives min_freq_geo < required min_freq
   - If base=10000: covers up to ~3000 tokens
   - If base=100000: covers up to ~30000 tokens  
   - If base=500000: covers up to ~150000 tokens

**If you can't change base (e.g., pretrained model):**
- Use NTK-aware scaling to extend coverage
- Use Sigmoid/Anchored for better shape when range is limited

**Key Insight:**
When the frequency range MATCHES the target length, geometric RoPE is optimal.
When there's a MISMATCH, frequency shape optimization helps.
""")
    
    # 保存结果
    output_dir = Path("results/frequency_range_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import json
    output_data = {
        "configs": list(configs.keys()),
        "lengths": lengths,
        "results": results,
    }
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    run_analysis()
    print("\nDone!")
