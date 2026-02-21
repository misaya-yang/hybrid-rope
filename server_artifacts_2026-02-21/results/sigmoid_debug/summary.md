# Sigmoid RoPE Debug Summary

## Bug Identified
The sigmoid parameterization had a critical bug:
- `positions * 100` created values 0, 100, 200, ... 6400
- `mid = T/2 = 50000`
- `sigmoid((positions*100) - 50000)` ≈ sigmoid(-50000 to -43600) ≈ 0 for ALL positions
- Result: ALL inv_freq ≈ inv_freq_min = 1e-6 (way too small!)

## Fix Applied
New sigmoid implementation (v2):
- Uses proper scaling to match geometric at low dimensions
- Applies sigmoid-based modulation only at higher dimensions
- Smooth transition from geometric to faster decay

## Results

| Config | PPL@2048 | PPL@8192 | Ratio to geo@2k |
|--------|----------|----------|-----------------|
| geo_100k | 8.802 | 593.629 | 1.00x |
| sigmoid_buggy | 43915.210 | 46107.567 | 4989.36x |
| sigmoid_fixed | 8.802 | 593.429 | 1.00x |

## Gate Evaluation
- Threshold: PPL@2048 <= 1.5x geo_100k
- sigmoid_fixed ratio: 1.00x
- **GATE: ✅ PASSED**

## Next Steps
- Gate passed! Can proceed to 16k/24k/32k boundary scan.
