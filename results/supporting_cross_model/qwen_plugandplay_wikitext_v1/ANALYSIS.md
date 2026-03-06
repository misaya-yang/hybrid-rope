# Qwen Plug-and-Play RoPE PPL Evaluation Analysis

**Date:** 2026-02-14
**Model:** Qwen2.5-7B-Instruct
**Dataset:** wikitext-103-raw-v1 validation
**Quantization:** 4-bit

## Summary Table

| Config | PPL@2048 | PPL@4096 | PPL@8192 | PPL@16384 | PPL@32768 | Collapse |
|--------|----------|----------|----------|-----------|-----------|----------|
| qwen_orig | **8.75±0.09** | **7.59±0.34** | **7.20±0.15** | **7.22±0.15** | **6.80±0.12** | 0.78 |
| qwen_yarn8 | N/A | N/A | N/A | N/A | N/A | N/A |
| qwen_geo_100k | 8.86±0.09 | 7.71±0.35 | 7.33±0.13 | 7.38±0.15 | 10.33±0.11 | 1.17 |
| qwen_sigmoid_best_t100k | 12.07±0.21 | 10.81±0.49 | 10.53±0.24 | 12.92±0.21 | 11.88±0.27 | 0.98 |
| qwen_hybrid_a0.2_t100k | 26.94±1.86 | 40.96±3.34 | 93.69±5.02 | 252.24±4.55 | 473.91±16.69 | 17.59 |
| qwen_random_control | 3157.60±214.76 | 3011.59±124.93 | 3286.35±120.07 | 3587.70±152.86 | 4311.14±87.99 | 1.37 |

## PPL Ratio vs Baseline (qwen_orig)

| Config | 2048 | 4096 | 8192 | 16384 | 32768 |
|--------|------|------|------|-------|-------|
| qwen_geo_100k | +1.2% | +1.6% | +1.7% | +2.3% | **+51.8%** |
| qwen_sigmoid_best_t100k | +37.9% | +42.4% | +46.1% | +79.1% | +74.6% |
| qwen_hybrid_a0.2_t100k | +208% | +440% | +1200% | +3395% | **+6865%** |
| qwen_random_control | +36067% | +39691% | +45616% | +49708% | +63356% |

## Key Findings

### 1. Baseline (qwen_orig) Characteristics
- **Optimal length:** PPL decreases with longer context (6.80 at 32k vs 8.75 at 2k)
- **Collapse ratio:** 0.78 (healthy, no length extrapolation issues within training range)
- This suggests Qwen2.5-7B has good native length generalization

### 2. Geometric Scaling (qwen_geo_100k)
- **Short sequences (≤16k):** Near-identical to baseline (+1-2%)
- **Long sequences (32k):** Significant degradation (+52%)
- **Interpretation:** Geometric frequency scaling works well within training range but fails at extrapolation
- **Root cause:** Geometric scaling may push frequencies outside the model's learned distribution

### 3. Sigmoid Frequency (qwen_sigmoid_best_t100k)
- **Overall:** Consistent degradation across all lengths (+38-79%)
- **Collapse ratio:** 0.98 (stable, no catastrophic failure)
- **Interpretation:** Sigmoid modulation disrupts the learned frequency distribution
- **Issue:** The sigmoid parameters (steepness=8.0, midpoint=0.5, omf=0.3) may not be optimal for Qwen

### 4. Hybrid Method (qwen_hybrid_a0.2_t100k)
- **Severe failure:** PPL explodes exponentially with length
- **Collapse ratio:** 17.59 (catastrophic)
- **Interpretation:** The hybrid polynomial-sigmoid combination is incompatible with Qwen's RoPE implementation
- **Key difference:** This method worked on Llama but fails dramatically on Qwen

### 5. Random Control (qwen_random_control)
- **Expected failure:** PPL ~3000-4000 confirms position encoding importance
- **Collapse ratio:** 1.37 (relatively stable at high PPL)
- **Validates:** The evaluation protocol correctly detects broken position encodings

### 6. YaRN (qwen_yarn8)
- **Status:** Failed to load
- **Error:** `unsupported operand type(s) for ** or pow(): 'NoneType' and 'Tensor'`
- **Cause:** YaRN implementation incompatible with 4-bit quantization in bitsandbytes

## Critical Analysis

### Why did methods that work on Llama fail on Qwen?

1. **Different RoPE implementations:**
   - Llama uses standard RoPE with `rotary_emb.inv_freq` shape [head_dim/2]
   - Qwen may have different frequency initialization or application

2. **Qwen's native Yarn:**
   - Qwen2.5 already includes YaRN-style scaling for long contexts
   - Adding external frequency modifications conflicts with built-in mechanisms

3. **Frequency distribution mismatch:**
   - Our methods were tuned for Llama's frequency distribution
   - Qwen may have different base frequency or scaling

### Recommendations

1. **For Qwen models:**
   - Use native RoPE without modification for contexts ≤16k
   - Avoid sigmoid/hybrid methods without re-tuning
   - Geometric scaling is acceptable for short sequences

2. **For future work:**
   - Analyze Qwen's native `inv_freq` distribution
   - Tune sigmoid/hybrid parameters specifically for Qwen
   - Consider Qwen's built-in YaRN before adding external modifications

3. **Cross-model generalization:**
   - RoPE modifications are NOT universally transferable
   - Each model architecture needs specific parameter tuning
   - Always validate on target model before deployment

## Conclusion

This experiment demonstrates that **RoPE modifications are model-specific**. Methods that show significant improvements on Llama models (hybrid, sigmoid) can actually degrade performance on Qwen models. The baseline Qwen2.5-7B-Instruct performs optimally with its native position encoding for contexts up to 32k tokens.

**Bottom line:** For Qwen2.5-7B, stick with the original RoPE implementation. The model's native length generalization is already well-tuned.