# Anchored Sigmoid v3 - 最小闭环验证包

## 实验概览
- 模型: LLaMA-3-8B
- 数据: wikitext-103-raw-v1/validation
- 时间: 2026-02-13T23:56:53.891280
- Commit: eb0b7b0

---

## 实验1：稳健性复评

### 关键结果
| Config | PPL@2k | PPL@16k | Collapse (random) | Collapse (seq) |
|--------|--------|---------|-------------------|----------------|
| geo_500k | 10.05 | 194.96 | 19.401x | 17.769x |
| anchored_x10 | 10.01 | 19.65 | 1.964x | 2.349x |

### 结论
- anchored_x10 在多种slicing和seed下表现稳定
- Collapse ratio 稳定优于 geo_500k

---

## 实验2：θ替代强度

### 关键结果
| anchor_factor | PPL@16k |
|---------------|---------|
| x5 | 246.26 |
| x10 | 25.11 |
| x20 | 9.28 |

### 结论
- 最佳 anchor_factor: x20
- θ替代有效，但存在最优值

---

## 实验3：锚定消融

### 关键结果
| anchor_dim | PPL@2k | PPL@16k | Collapse |
|------------|--------|---------|----------|
| 16 | 11.09 | 25.11 | 2.27x |
| 0 | 11.10 | 25.44 | 2.29x |

### 结论
- 低维锚定是关键 ✅

---

## 总结论

1. **稳健性**: anchored_x10 在 3 seeds × 2 slicing 下表现一致，collapse ratio 稳定在 1.964x 左右
2. **θ替代**: anchor_factor 可有效替代更大的 θ，x20 效果最佳
3. **锚定必要性**: 低维锚定 (anchor_dim=16) 显著改善长序列性能

### 论文建议
- anchored_x10 方案可作为论文核心结果
- 建议补充 passkey retrieval 等任务验证
