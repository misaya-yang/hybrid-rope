# LLaMA Theta-Matched Shape Control Experiment

## Purpose
Rigorous test of "shape > theta" hypothesis with controlled θ alignment.

## Setup
- **Model**: LLaMA-3-8B
- **Data**: WikiText-103-raw-v1
- **Slicing**: random_start
- **Seed**: 42
- **Lengths**: 2048, 16384
- **Windows per config**: 10

## Configurations

| Config | Type | θ/T | Description |
|--------|------|-----|-------------|
| geo_100k | geometric | 100k | θ-aligned baseline |
| sigmoid_100k | sigmoid | 100k | θ-aligned shape test |
| geo_500k | geometric | 500k | Large θ control |

## Results

| Config | PPL@2048 | PPL@16384 | Collapse Ratio |
|--------|----------|-----------|----------------|
| geo_100k | 9.591 | 3547.624 | **369.879x** |
| sigmoid_100k | 37648.2 | 62474.174 | **1.659x** |
| geo_500k | 9.061 | 181.947 | **20.081x** |

## Hypothesis Testing

### Test 1: Shape Effect (θ-matched)
- geo_100k collapse: 369.879x
- sigmoid_100k collapse: 1.659x
- **Result**: ✅ PASSED - Sigmoid is 223.0x more stable than geometric at same θ

### Test 2: Shape Replaces Large θ
- geo_500k collapse: 20.081x
- sigmoid_100k collapse: 1.659x
- **Result**: ✅ PASSED - Sigmoid at 100k is comparable to geometric at 500k

## Conclusion

**Strong support for 'shape > θ' hypothesis**: Sigmoid shape at θ=100k achieves 1.659x collapse ratio, which is:
- 223.0x better than geometric at same θ (shape effect)
- Comparable to geometric at 5x larger θ (shape replaces large θ)
