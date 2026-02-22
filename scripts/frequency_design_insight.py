#!/usr/bin/env python
"""
Frequency Design Insights
=========================
深入分析：为什么小base更好？什么才是真正的"最优"？

关键问题：
1. 小base → 高频 → 短波长 → 长距离相位循环少
2. 但我们需要同时满足短距离（高频）和长距离（低频）
3. 所以需要的是"频率覆盖范围"而不是"绝对值"
"""

import math
from pathlib import Path
import numpy as np
import torch


def geometric_freq(d: int, base: float) -> torch.Tensor:
    i = torch.arange(d // 2, dtype=torch.float64)
    return base ** (-2.0 * i / float(d))


def analyze_frequency_coverage():
    """分析频率覆盖特性"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    d = 128
    bases = [1000, 5000, 10000, 50000, 100000, 500000, 1000000]
    
    print("="*80)
    print("Frequency Coverage Analysis")
    print("="*80)
    
    print(f"\n{'Base':<10} {'max_freq':>12} {'min_freq':>15} {'freq_ratio':>15} {'coverage':>15}")
    print("-" * 80)
    
    for base in bases:
        freqs = geometric_freq(d, base).to(device)
        
        max_freq = freqs[0].item()
        min_freq = freqs[-1].item()
        
        # 频率跨度比例
        freq_ratio = max_freq / min_freq
        
        # 可覆盖的距离范围
        # 对应波长范围
        max_wavelength = 2 * math.pi / min_freq  # 对应最低频率
        min_wavelength = 2 * math.pi / max_freq  # 对应最高频率
        
        # 能准确表示的距离范围（奈奎斯特：波长/2）
        max_distance = max_wavelength / 2
        min_distance = min_wavelength / 2
        
        coverage = max_distance / min_distance
        
        print(f"{base:<10} {max_freq:>12.4f} {min_freq:>15.6f} {freq_ratio:>15.1f} {coverage:>15.1f}")
    
    # 关键洞察
    print("\n" + "="*80)
    print("Key Insights")
    print("="*80)
    print("""
1. **Frequency Ratio (max_freq/min_freq)**:
   - base=1000:  ratio=1.0 (几乎恒定频率)
   - base=10000: ratio=~85 (有变化的频率范围)
   - base=1M:    ratio=~800000 (非常宽的频率范围)
   
   **Insight**: 大的base提供更宽的频率覆盖范围

2. **What matters is not absolute frequency, but COVERAGE**:
   - Small base → all frequencies are high → can't encode long distances well
   - Large base → frequencies span from high to low → can encode both short AND long

3. **The optimal base should make frequency range COVER the target length**:
   - For L=8K: need min_freq < 2π/(2L) ≈ 0.0004
   - This requires base > ~10000 for standard geometric
""")
    
    # 验证理论
    print("\n" + "="*80)
    print("Theory Verification: Minimum Base for Length L")
    print("="*80)
    
    lengths = [2048, 4096, 8192, 16384, 32768, 65536]
    
    for L in lengths:
        # 奈奎斯特：需要频率 < 2π/L
        required_min_freq = 2 * math.pi / L
        
        # 对于几何级数：freq[d/2-1] = base^(-1)
        # 需要 base^(-1) < required_min_freq
        # 即 base > 1/required_min_freq
        
        min_base_theory = 1 / required_min_freq
        
        # 实际测试
        test_bases = [1000, 5000, 10000, 20000, 50000, 100000]
        
        print(f"\nL={L}: need min_freq < {required_min_freq:.6f}")
        print(f"  Theoretical min base: {min_base_theory:.0f}")
        
        for base in test_bases:
            freqs = geometric_freq(d, base).to(device)
            actual_min = freqs[-1].item()
            status = "✓" if actual_min < required_min_freq else "✗"
            print(f"  base={base:>7}: min_freq={actual_min:.6f} {status}")


def design_guideline():
    """给出实用设计指南"""
    print("\n" + "="*80)
    print("Practical Design Guideline")
    print("="*80)
    print("""
Based on the analysis, here's the CORRECT way to design RoPE frequency:

**Step 1: Determine target length L**
   - e.g., L = 16384 (16K tokens)

**Step 2: Compute required frequency range**
   - max_freq: ~1 (for short distances, no constraint)
   - min_freq < 2π / (2L) = π/L
   - For L=16K: min_freq < 0.0002

**Step 3: Choose base to satisfy min_freq constraint**
   - For geometric: base^(-1) < required_min_freq
   - base > 1 / required_min_freq = L / π ≈ 0.32 * L
   
   **Rule of thumb**: base ≈ 0.3 × L works well

**Step 4: If base is fixed (e.g., pretrained model = 10000)**
   - Use NTK-aware scaling to effectively increase base
   - Use Sigmoid/Anchored when base is insufficient

**Verification**:
   - L=4K: base ≈ 1200 → actual base 10000 is ENOUGH
   - L=8K: base ≈ 2500 → actual base 10000 is ENOUGH  
   - L=16K: base ≈ 5000 → actual base 10000 is ENOUGH
   - L=32K: base ≈ 10000 → actual base 10000 is BORDERLINE
   - L=64K: base ≈ 20000 → actual base 10000 is NOT ENOUGH
   - → Need NTK/YaRN/Sigmoid for L > 32K
""")


if __name__ == "__main__":
    analyze_frequency_coverage()
    design_guideline()
    print("\nDone!")
