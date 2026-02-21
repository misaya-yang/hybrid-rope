# 实验1：稳健性复评

## Setup
- Configs: geo_500k, anchored_x10
- Lengths: [2048, 8192, 16384]
- Slicing: random_start, sequential
- Seeds: [42, 123, 777]

## Results

### PPL by Config×Length×Slicing (mean±std)

#### geo_500k
| Length | random_start | sequential |
|--------|--------------|------------|
| 2048 | 10.05±0.76 | 5.96±0.00 |
| 8192 | 8.33±0.84 | 5.95±0.00 |
| 16384 | 194.96±48.35 | 105.92±0.00 |

**Collapse Ratios:**
- random_start: 19.401x
- sequential: 17.769x

#### anchored_x10
| Length | random_start | sequential |
|--------|--------------|------------|
| 2048 | 10.01±0.77 | 5.85±0.00 |
| 8192 | 8.23±0.84 | 5.88±0.00 |
| 16384 | 19.65±4.33 | 13.74±0.00 |

**Collapse Ratios:**
- random_start: 1.964x
- sequential: 2.349x

## 结论
- geo_500k collapse_ratio: 19.401x
- anchored_x10 collapse_ratio: 1.964x
- **anchored_x10 比 geo_500k 稳定 9.9x** ✅
