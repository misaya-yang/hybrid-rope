# 实验2：θ替代强度

## Setup
- anchor_factors: [5, 10, 20]
- 固定: theta=100k, anchor_dim=16, slope=0.5
- Length: 16384
- Slicing: random_start
- Seed: 42

## Results
| anchor_factor | PPL@16k |
|---------------|---------|
| x5 | 246.26 |
| x10 | 25.11 |
| x20 | 9.28 |

## 结论
- 最佳 anchor_factor: x20 (PPL=9.28)
