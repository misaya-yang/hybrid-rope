# Phase Collision Analysis with Different Distance Priors D(Δ)

**Date**: 2026-02-21  
**Author**: AI Assistant  
**Purpose**: Experiment A for paper narrative validation

---

## 1. Experiment Motivation

The goal is to verify the paper's core theoretical claim:
> "Different distance distribution assumptions D(Δ) lead to different method rankings - there is no universal optimal frequency allocation"

Two experimental versions:
- **V1**: Initial design (mixed base values)
- **V2**: Corrected design (unified base=10000)

---

## 2. Experimental Design V2 (Final)

### 2.1 Configuration
- Head dimension: d = 128
- Maximum length: L = 16384
- **Unified base = 10000** (critical fix!)
- Scale factor: 4.0 (for NTK/YaRN/LongRoPE)

### 2.2 Methods Tested
| Method | Description |
|--------|-------------|
| geo_base | Standard geometric RoPE (base=10000) |
| ntk_4x | NTK-aware scaling (scale=4x) |
| yarn_4x | YaRN (scale=4x) |
| longrope_4x | LongRoPE (scale=4x) |
| hybrid_50_100k | Hybrid: 50% geo + 100k theta |
| hybrid_50_500k | Hybrid: 50% geo + 500k theta |
| sigmoid_v2 | Sigmoid (k=0.125, x0=32) |
| sigmoid_k016 | Sigmoid (k=0.16, x0=32) |
| anchored_x10 | Anchored Sigmoid (factor=10) |
| anchored_x20 | Anchored Sigmoid (factor=20) |
| anchored_x20_dim0 | Anchored Sigmoid (dim=0, factor=20) |

### 2.3 Distance Distribution D(Δ) Variants

| D(Δ) Type | Weights (Short/Mid/Long) | Interpretation |
|------------|--------------------------|----------------|
| Uniform | 0.33 / 0.33 / 0.33 | All distances equally important |
| Power-law Short | **0.7 / 0.2 / 0.1** | Real LLM attention pattern (short-focused) |
| Power-law Long | 0.1 / 0.2 / **0.7** | Long-context requirement focus |
| Bimodal | 0.4 / 0.2 / 0.4 | Local + Global both important |

---

## 3. Results

### 3.1 Raw Scores (Total = 0.2×Short + 0.3×Mid + 0.5×Long)

| Method | Short | Mid | Long | Total |
|--------|-------|-----|------|-------:|
| geo_base | 0.534 | 0.196 | 0.070 | 0.2005 |
| ntk_4x | 0.596 | 0.287 | 0.057 | 0.2339 |
| yarn_4x | 0.687 | 0.329 | 0.075 | 0.2734 |
| longrope_4x | 0.579 | 0.281 | 0.078 | 0.2390 |
| hybrid_50_100k | 0.544 | 0.340 | 0.122 | 0.2719 |
| hybrid_50_500k | 0.545 | 0.407 | 0.224 | 0.3432 |
| sigmoid_v2 | 0.512 | 0.295 | 0.078 | 0.2298 |
| sigmoid_k016 | 0.507 | 0.324 | 0.069 | 0.2331 |
| anchored_x10 | 0.597 | 0.277 | 0.064 | 0.2347 |
| anchored_x20 | 0.600 | 0.280 | 0.059 | 0.2335 |
| anchored_x20_dim0 | 0.614 | 0.293 | 0.067 | 0.2441 |

### 3.2 Rankings under Different D(Δ)

#### Uniform (0.33/0.33/0.33)
| Rank | Method | Score |
|:---:|--------|-------:|
| 1 | **geo_base** | **0.2667** |
| 2 | sigmoid_v2 | 0.2950 |
| 3 | sigmoid_k016 | 0.3000 |
| 4 | longrope_4x | 0.3126 |
| 5 | anchored_x10 | 0.3128 |
| 6 | anchored_x20 | 0.3130 |
| 7 | ntk_4x | 0.3134 |
| 8 | anchored_x20_dim0 | 0.3246 |
| 9 | hybrid_50_100k | 0.3354 |
| 10 | yarn_4x | 0.3634 |
| 11 | hybrid_50_500k | 0.3921 |

#### Power-law Short (0.7/0.2/0.1) - Real LLM Attention
| Rank | Method | Score |
|:---:|--------|-------:|
| 1 | **geo_base** | **0.4203** |
| 2 | sigmoid_v2 | 0.4252 |
| 3 | sigmoid_k016 | 0.4266 |
| 4 | hybrid_50_100k | 0.4609 |
| 5 | longrope_4x | 0.4692 |
| 6 | anchored_x10 | 0.4798 |
| 7 | ntk_4x | 0.4803 |
| 8 | anchored_x20 | 0.4822 |
| 9 | hybrid_50_500k | 0.4851 |
| 10 | anchored_x20_dim0 | 0.4951 |
| 11 | yarn_4x | 0.5539 |

#### Power-law Long (0.1/0.2/0.7)
| Rank | Method | Score |
|:---:|--------|-------:|
| 1 | **geo_base** | **0.1413** |
| 2 | ntk_4x | 0.1570 |
| 3 | anchored_x20 | 0.1573 |
| 4 | anchored_x10 | 0.1601 |
| 5 | sigmoid_k016 | 0.1639 |
| 6 | sigmoid_v2 | 0.1646 |
| 7 | anchored_x20_dim0 | 0.1667 |
| 8 | longrope_4x | 0.1686 |
| 9 | yarn_4x | 0.1867 |
| 10 | hybrid_50_100k | 0.2080 |
| 11 | hybrid_50_500k | 0.2929 |

#### Bimodal (0.4/0.2/0.4)
| Rank | Method | Score |
|:---:|--------|-------:|
| 1 | **geo_base** | **0.2808** |
| 2 | sigmoid_v2 | 0.2949 |
| 3 | sigmoid_k016 | 0.2953 |
| 4 | ntk_4x | 0.3186 |
| 5 | longrope_4x | 0.3189 |
| 6 | anchored_x20 | 0.3197 |
| 7 | anchored_x10 | 0.3200 |
| 8 | anchored_x20_dim0 | 0.3309 |
| 9 | hybrid_50_100k | 0.3345 |
| 10 | yarn_4x | 0.3703 |
| 11 | hybrid_50_500k | 0.3890 |

---

## 4. Key Findings

### 4.1 Theorem Verification

| Question | Expected | Actual Result |
|----------|----------|----------------|
| 1. Uniform D, Standard optimal? | Yes (Theorem 1) | ✅ **Yes** - geo_base ranks #1 |
| 2. Power-law Short, Hybrid rises? | Yes (Hybrid ~ Standard) | ⚠️ **Partially** - geo_base still #1, Hybrid at #4 |
| 3. Bimodal, Sigmoid optimal? | Yes (Theorem 3) | ❌ **No** - geo_base still #1 |

### 4.2 Critical Discovery

**After unifying base=10000, standard geometric RoPE is optimal under ALL D(Δ) assumptions!**

This is a significant finding that requires theoretical reconsideration:
- Previous V1 results showed ranking shifts because different methods used different base values
- V2 (controlled) shows that when base is controlled, there's no advantage to frequency redistribution
- The "ranking shift" observed in V1 was due to confounded base values, not the shape of frequency allocation

### 4.3 Implications

1. **For Theorem 1**: The prediction holds - geometric is optimal under Uniform
2. **For the paper narrative**: The "D(Δ) determines optimal method" claim needs stronger evidence
3. **For future work**: Need to reconsider whether frequency shape (vs base) is the right design dimension

---

## 5. Comparison: V1 vs V2

### V1 (Mixed Bases)
- geo_500k: base=500000
- ntk_8x: base scaled by 8x
- Sigmoid:有自己的base

**Result**: geo_500k ranked #13 due to inappropriate base for L=16384

### V2 (Unified Base)
- All methods: base=10000
- Only frequency distribution shape differs

**Result**: geo_base ranks #1 in ALL scenarios

---

## 6. Output Files

- `results/phase_collision_comparison_v2/scores.json` - Full numerical results
- `results/phase_collision_comparison_v2/phase_collision_curves.png` - Curves
- `results/phase_collision_comparison_v2/phase_collision_by_D_v2.png` - Bar charts

---

## 7. Discussion Points for Advisor

1. **Is the "frequency shape" design dimension meaningful?**
   - V2 suggests base selection matters more than shape
   
2. **Should we reconsider the theoretical framework?**
   - Perhaps the optimization target should be different
   
3. **What's the path forward?**
   - Option A: Keep exploring frequency shapes with different formulations
   - Option B: Accept that geometric is near-optimal, focus on other aspects
   - Option C: Theoretically justify why different bases are needed

---

*Last updated: 2026-02-21*
