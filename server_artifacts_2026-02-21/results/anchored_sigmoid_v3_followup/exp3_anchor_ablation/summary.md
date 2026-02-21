# 实验3：锚定消融

## Setup
- anchor_dim: [16, 0]
- 固定: theta=100k, anchor_factor=10, slope=0.5
- Lengths: [2048, 16384]
- Slicing: random_start
- Seed: 42

## Results
| Config | PPL@2k | PPL@16k | Collapse |
|--------|--------|---------|----------|
| anchor_dim=16 | 11.09 | 25.11 | 2.27x |
| anchor_dim=0 | 11.10 | 25.44 | 2.29x |

## 结论
- **低维锚定是关键**: anchor_dim=16 比 anchor_dim=0 稳定 1.0x ✅
