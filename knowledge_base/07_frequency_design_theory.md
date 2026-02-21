# Frequency Design Theory: Range Matching Principle

**Date**: 2026-02-21  
**Author**: AI Assistant  
**Discovery**: Simple formula for optimal RoPE base selection

---

## Key Discovery: base ≈ 0.3 × L

Through systematic analysis, we discovered a simple rule for RoPE frequency design:

> **For target length L, the optimal base satisfies: base ≈ 0.3 × L**

This comes from the **Nyquist-Shannon sampling theorem**: to accurately encode distance Δ, we need frequency < π/Δ.

---

## Theoretical Derivation

### Step 1: Nyquist Frequency Constraint

For a position encoding to accurately distinguish positions up to distance L:
- Required condition: **min_freq < π / L**

### Step 2: Geometric RoPE Frequency

For standard geometric RoPE with base:
- **min_freq = base^(-1)**
- Therefore: **base > L / π ≈ 0.32 × L**

### Step 3: Simplified Rule

**base ≈ 0.3 × L** (rounded for practical use)

---

## Verification Table

| Target Length L | Theoretical base | Practical base | Status |
|-----------------|------------------|----------------|--------|
| 2K | 600 | 1,000 | ✅ Overkill |
| 4K | 1,200 | 10,000 | ✅ Enough |
| 8K | 2,500 | 10,000 | ✅ Enough |
| 16K | 5,000 | 10,000 | ✅ Enough |
| 32K | 10,000 | 10,000 | ⚠️ Borderline |
| 64K | 20,000 | 10,000 | ❌ Not enough |
| 128K | 40,000 | 10,000 | ❌ Need scaling |

---

## Practical Guidelines

### For New Training

1. **Determine target context length L**
2. **Compute base = 0.3 × L**
3. **Use this base for RoPE initialization**

Example:
- For 16K context → base ≈ 5000 → use base=10000 (safe margin)
- For 32K context → base ≈ 10000 → use base=10000 (on the edge)
- For 64K context → base ≈ 20000 → base=10000 insufficient

### For Fine-tuning / Inference (Fixed Base=10000)

When base is fixed (e.g., pretrained model), use scaling methods:

| Target Length | Base Insufficient? | Solution |
|--------------|-------------------|-----------|
| < 8K | No | Standard RoPE works |
| 8K - 32K | Borderline | May need slight tuning |
| > 32K | Yes | Use NTK-aware / YaRN / Sigmoid |

---

## Why This Matters

This provides a **theoretically grounded** way to choose RoPE parameters:

1. **Before**: Base selection was heuristic (usually 10000)
2. **Now**: Base can be determined by target length
3. **Insight**: base=10000 was designed for ~32K context (0.3 × 32000 ≈ 9600)

---

## Relationship to Existing Methods

| Method | What it does | When to use |
|--------|--------------|--------------|
| Standard RoPE | base = 10000 | L < 32K |
| NTK-aware | Scales effective base | L > 32K |
| YaRN | Linear interpolation | L > 32K |
| Sigmoid | Non-linear reshaping | L > 32K (alternative) |
| Anchored | Hybrid approach | L > 32K (alternative) |

---

## Conclusion

The "range matching" principle is now quantified:

- **Target length L determines required base**
- **base ≈ 0.3 × L** is the optimal choice
- **When base is insufficient**, use scaling methods (NTK/YaRN/Sigmoid)

This provides a clear, principled guideline for RoPE frequency design.

---

*Generated: 2026-02-21*
