#!/usr/bin/env python
"""
Optimal Base Selection Analysis
==============================
找出每个目标长度L的最优base值

关键发现：geo_10k在所有长度上都最好？
这可能意味着需要找出"最优base"的规律
"""

import math
from pathlib import Path
import numpy as np
import torch


def geometric_freq(d: int, base: float) -> torch.Tensor:
    """Standard geometric RoPE"""
    i = torch.arange(d // 2, dtype=torch.float64)
    return base ** (-2.0 * i / float(d))


def phase_collision_at_L(freqs: torch.Tensor, L: int, device) -> float:
    """计算Phase Collision Score"""
    distances = torch.unique(torch.logspace(0, math.log10(float(L)), 500, device=device).long())
    distances = distances[distances > 0]
    
    freqs_f32 = freqs.to(device=device, dtype=torch.float32)
    angles = distances.to(dtype=torch.float32).unsqueeze(-1) * freqs_f32.unsqueeze(0)
    inner = torch.cos(angles).mean(dim=-1)
    
    return float(inner.abs().mean())


def run_optimal_base_search():
    """搜索每个长度的最优base"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    d = 128
    lengths = [2048, 4096, 8192, 16384, 32768, 65536]
    
    # 搜索base值（从1k到10M）
    bases = np.logspace(3, 7, 50)  # 1k to 10M
    
    print("\n" + "="*80)
    print("Optimal Base Selection Analysis")
    print("="*80)
    print(f"Searching bases from {bases[0]:.0f} to {bases[-1]:.0f}")
    
    # 对每个长度，找最优base
    optimal_bases = {}
    results_all = {}
    
    for L in lengths:
        print(f"\n### Length L = {L}")
        
        scores = []
        for base in bases:
            freqs = geometric_freq(d, base).to(device)
            score = phase_collision_at_L(freqs, L, device)
            scores.append(score)
        
        # 找最优
        best_idx = np.argmin(scores)
        best_base = bases[best_idx]
        best_score = scores[best_idx]
        
        optimal_bases[L] = {
            "base": best_base,
            "score": best_score,
        }
        
        results_all[L] = {
            "bases": bases.tolist(),
            "scores": scores,
        }
        
        print(f"  Best base: {best_base:.0f} (score: {best_score:.4f})")
        
        # 打印top 5
        sorted_idx = np.argsort(scores)[:5]
        print(f"  Top 5 bases:")
        for i, idx in enumerate(sorted_idx):
            print(f"    {i+1}. base={bases[idx]:.0f}, score={scores[idx]:.4f}")
    
    # 分析规律
    print("\n" + "="*80)
    print("Summary: Optimal Base vs Length")
    print("="*80)
    
    print(f"{'L':<10} {'Optimal Base':<15} {'Optimal Score':<15} {'Ratio L/base':<15}")
    print("-" * 60)
    
    for L, data in optimal_bases.items():
        ratio = L / data['base']
        print(f"{L:<10} {data['base']:<15.0f} {data['score']:<15.4f} {ratio:<15.4f}")
    
    # 关键洞察
    print("\n" + "="*80)
    print("Key Insight")
    print("="*80)
    
    # 计算L / optimal_base的规律
    ratios = [L / optimal_bases[L]['base'] for L in lengths]
    avg_ratio = np.mean(ratios)
    
    print(f"\nAverage L / optimal_base ratio: {avg_ratio:.2f}")
    print(f"This suggests: optimal_base ≈ L / {avg_ratio:.1f}")
    
    # 验证这个规律
    print("\nVerification:")
    for L in lengths:
        predicted_base = L / avg_ratio
        actual_base = optimal_bases[L]['base']
        error = abs(predicted_base - actual_base) / actual_base * 100
        print(f"  L={L}: predicted={predicted_base:.0f}, actual={actual_base:.0f}, error={error:.1f}%")
    
    # 保存结果
    output_dir = Path("results/optimal_base_search")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import json
    output_data = {
        "optimal_bases": {str(k): v for k, v in optimal_bases.items()},
        "avg_ratio": float(avg_ratio),
    }
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    run_optimal_base_search()
    print("\nDone!")
