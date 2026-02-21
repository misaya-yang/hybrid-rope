# 8B LoRA Experiment Analysis

**Date**: 2026-02-21  
**Purpose**: Analyze why the 8B experiment failed and identify design issues

---

## Current Results

| Method | PPL@16k | PPL@32k | Status |
|--------|----------|----------|--------|
| YaRN | 6.057 | 6.270 | ✅ Completed |
| PI (linear) | 6.137 | 6.310 | ✅ Completed |
| Hybrid | - | - | ❌ Poor results |

---

## Issues Identified

### Issue 1: Training Hyperparameters Mismatch

**In `train_llama8b_lora_variant.py` (variant script defaults):**
```python
--per_device_train_batch_size: 1
--gradient_accumulation_steps: 8
# Effective batch size = 1 * 8 = 8
```

**In `run_llama8b_fair_suite.py` (suite runner overrides):**
```python
--per_device_train_batch_size: 4
--gradient_accumulation_steps: 1
# Effective batch size = 4 * 1 = 4
```

The suite passes these as arguments, so they should be consistent. But the defaults in the variant script are different, which could cause issues if not passed correctly.

---

### Issue 2: Attention Implementation Variation

The code tries multiple attention implementations in order:
```python
def attn_candidates(mode: str) -> List[Optional[str]]:
    if mode == "auto":
        return ["flash_attention_2", "sdpa", None]
```

This means different runs might use different attention backends, introducing variability.

**Found in results:**
- YaRN: `"attn_used": "sdpa"`
- PI: `"attn_used": "sdpa"`

But there's no guarantee this is consistent across runs.

---

### Issue 3: Hybrid Uses Completely Different Implementation

**YaRN/PI**: Use HuggingFace's built-in `rope_scaling`:
```python
{"rope_type": "yarn", "factor": 8.0, "rope_theta": 500000.0}
{"rope_type": "linear", "factor": 8.0, "rope_theta": 500000.0}
```

**Hybrid**: Uses custom monkey patching:
```python
def compute_hybrid_inv_freq(
    head_dim: int,
    theta_base: float = 500000.0,
    split_ratio: float = 0.5,
    alpha: float = 0.2,
    p: float = 3.9,
    min_freq_scale: float = 4.0,
) -> torch.Tensor
```

This is NOT a fair comparison - the baseline methods use native HF implementation while Hybrid uses custom code.

---

### Issue 4: Hybrid Has Multiple Unoptimized Hyperparameters

```python
hybrid_split_ratio: 0.5     # Not tuned
hybrid_alpha: 0.2           # Not tuned  
hybrid_p: 3.9               # Not tuned
hybrid_min_freq_scale: 4.0   # Not tuned
```

These values were not grid-searched and may not be optimal.

---

### Issue 5: RoPE Theta Interpretation

From YaRN result:
```json
"rope_theta": 500000.0,
"original_max_position_embeddings": 8192
```

But this is the **transformers internal** representation. The actual effective theta is:
- YaRN: rope_theta × factor = 500000 × 8 = 4M effective
- PI: rope_theta × factor = 500000 × 8 = 4M effective

However, the Hybrid uses:
- `theta_base: 500000.0` directly in the custom function

This might not be equivalent!

---

### Issue 6:rope_scaling vs Custom Patch Incompatibility

YaRN and PI use `cfg.rope_scaling` which transforms the model's position embeddings internally.

Hybrid uses `patch_hybrid_rope()` which directly modifies `module.inv_freq` after model loading.

These two approaches might interact differently with the LoRA training.

---

## Recommendations for Re-design

### 1. Fix Hyperparameters
Ensure ALL variants use identical:
- Batch size: 4 × 1 = 4
- Learning rate: 2e-4
- Steps: 600
- Attention implementation: force sdpa for all

### 2. Make Implementation Consistent
**Option A**: Use custom implementation for ALL methods (like Hybrid)
**Option B**: Use native HF rope_scaling for ALL methods

Currently mixing:
- YaRN: native
- PI: native
- Hybrid: custom

### 3. Grid Search Hybrid Parameters
Before comparing, tune:
- `split_ratio`: [0.3, 0.5, 0.7]
- `alpha`: [0.1, 0.2, 0.3]
- `p`: [2.0, 3.9, 5.0]

### 4. Use Same Base Theta
Ensure all methods start from the same theta and only change the frequency distribution.

### 5. Add Baseline Comparison
Run a "no-scaling" baseline (standard RoPE at 8K) to understand the baseline PPL.

---

## Proposed Fair Comparison Design

| Parameter | All Variants |
|-----------|---------------|
| Base Model | LLaMA-3-8B-Instruct |
| Training Steps | 600 |
| Seq Len | 8192 |
| Batch Size | 4 |
| LR | 2e-4 |
| Attention | sdpa (forced) |
| Base Theta | 500000 (same for all) |
| Factor | 8.0 (for scaling methods) |

**Methods to compare:**
1. **Baseline**: No scaling (standard RoPE at 8K)
2. **PI (linear)**: factor=8.0
3. **YaRN**: factor=8.0
4. **Anchored-Sigmoid**: Custom implementation with optimal params (anchor_factor=20)

Each uses the same implementation approach (either all native or all custom).

---

*Analysis completed: 2026-02-21*
