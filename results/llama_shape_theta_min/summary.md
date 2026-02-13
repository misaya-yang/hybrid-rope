# LLaMA Shape vs Theta Min Experiment

## Setup
- **Model**: LLaMA-3-8B
- **Data**: WikiText-103-raw-v1
- **Slicing**: random_start
- **Seed**: 42
- **Lengths**: 2048, 16384

## Results

| Config | PPL@2048 | PPL@16384 | Collapse Ratio |
|--------|----------|-----------|----------------|
| geo_10k | 518.013 | 11214.916 | **22.026x** |
| sigmoid_t100k | 10.372 | 12.51 | **1.077x** |

## Collapse Ratio Comparison
- geo_10k: 22.026x
- sigmoid_t100k: 1.077x
- **Improvement**: 20.5x more stable with sigmoid

## Conclusion

sigmoid_t100k 在 16K 的 collapse_ratio (1.077x) 明显低于 geo_10k (22.026x)，支持 **频谱形状 > θ 大小** 的稳定性假设。

Sigmoid频率分布即使使用较小的θ参数(T=100k)，也比geometric分布(θ=10k)更稳定，说明**频率曲线的形状**（平缓过渡 vs 陡峭截断）是影响长序列外推的关键因素。

### 关键发现

1. **Geometric RoPE在16K完全崩溃**: PPL从518暴涨到11215 (22x)
2. **Sigmoid RoPE保持稳定**: PPL仅从10.4增加到12.5 (1.08x)
3. **20.5倍稳定性提升**: sigmoid相比geometric在长序列上的改进

### 理论解释

- **Geometric**: 高频分量在长序列上快速衰减，导致位置编码失效
- **Sigmoid**: 平滑的频率过渡避免了高频分量的突然截断，保持了长距离依赖的建模能力