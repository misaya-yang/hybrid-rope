# Phase16: Formula Optimality Sweep (`tau*=d_head/sqrt(L)`)

> 日期: 2026-03-09
> 状态: **COMPLETE**
> 设备: `M4 Max 36GB / mps`
> 环境: `aidemo`
> 主报告: `results/theory/phase16_formula_optimality_sweep_local_m4_wikitext/reports/report.md`

---

## 0. 实验目的

这轮不是再证明 `EVQ > Geo`，而是专门回答一个更硬的问题：

**`tau*=d_head/sqrt(L_train)` 到底是不是一个稳健的近似最优公式，还是只是少数设置下碰巧有效？**

---

## 1. Sweep 设置

| 项 | 值 |
|----|----|
| 模型 tier | `50.9M` |
| 数据 | `local_wikitext` |
| 训练长度 `L` | `256 / 512 / 1024` |
| 头数 `H` | `4 / 8 / 16` |
| `d_head` | `128 / 64 / 32` |
| 配置数 | `9` |
| pilot | `45` runs |
| confirm | `54` runs |
| 总 runs | **`99`** |
| confirm seeds | `42 / 137 / 256` |
| 对照 | `tau=0` (Geometric), `tau=theory`, theory 附近邻点 |

---

## 2. 最重要结论

### 2.1 这条公式没有被打脸

在 `9` 个 `(L, H, d_head)` 配置里：

- 理论值是**精确最优**：`3/9`
- 理论值进入**前二**：`6/9`
- 理论值进入**前三**：`8/9`
- 所有配置的最佳 `tau` 都落在理论值的 **`1.5x`** 以内
- 最佳值与理论值的平均比例约为 **`1.20x`**

这说明 `tau*=d_head/sqrt(L)` 不是一个脆弱 heuristic，而是一个**稳定的近似最优中心线**。

### 2.2 它更像“近似最优定律”，不是“精确最优定律”

这轮 sweep 最清楚暴露出的 pattern 是：

- 最优点通常就在理论值附近
- 但经常比理论值**略偏大**
- 偏移不是乱跳，而是常见在 `1.2x-1.25x` 附近

所以 paper-safe 的说法应该是：

> `tau*=d_head/sqrt(L)` 给出了一个稳健的 near-optimal scaling law；有限容量、有限训练和离线小语料 setting 下，最优值常表现出轻微右偏。

### 2.3 相比 Geometric，理论值大多数配置都更好

按 confirm 后的 mean selection score 看：

- 理论 `tau` 优于 `tau=0`：`7/9` configs
- 仅 `H=16` 且 `L>=512` 的两组里，理论值略输给 `tau=0`
- 即便在这些 case，最佳值也仍然只是在理论值右侧不远处，而不是退回 `tau=0`

这说明问题不在“公式完全错了”，而在**理论值略偏保守**。

---

## 3. 分配置结果

| Config | Theory tau | Best tau | Theory rank | 结论 |
|---|---:|---:|---:|---|
| `L256_H4_Dh128` | `8.00` | `10.00` | `2` | theory 很接近最优 |
| `L256_H8_Dh64` | `4.00` | `4.00` | `1` | theory 精确命中 |
| `L256_H16_Dh32` | `2.00` | `2.00` | `1` | theory 精确命中 |
| `L512_H4_Dh128` | `5.66` | `5.66` | `1` | theory 精确命中 |
| `L512_H8_Dh64` | `2.83` | `4.24` | `3` | theory 偏保守，最优右移 |
| `L512_H16_Dh32` | `1.41` | `1.77` | `5` | theory 明显偏保守 |
| `L1024_H4_Dh128` | `4.00` | `5.00` | `2` | theory 很接近最优 |
| `L1024_H8_Dh64` | `2.00` | `2.50` | `2` | theory 很接近最优 |
| `L1024_H16_Dh32` | `1.00` | `1.25` | `3` | theory 偏保守但仍在近邻 |

---

## 4. 应该如何写进论文

### 4.1 可以直接写的结论

- `tau*=d_head/sqrt(L)` 在多 `L`、多头数、多 seed sweep 中稳定落在最优带附近
- 它是一个**强默认值**，不是只在单一配置上成立的拍脑袋公式
- 最优值常位于理论值右侧的轻微偏移带，表明理论式抓住了主导 scaling，但有限容量 setting 仍有小的 correction

### 4.2 不应该写的结论

- 不要写成 “Phase16 证明这是精确全局最优解”
- 不要写成 “理论值在所有配置都优于任何邻点”
- 不要写成 “所有 head count 都严格 obey 公式而无偏差”

---

## 5. 对核心叙事的真正价值

这轮实验把 `tau*` 的地位从：

- “有几个点看起来挺对”

推进到了：

- “在系统 sweep 下，它稳定给出 near-optimal 中心线；真正的最优值只是做小幅右偏修正”

因此主叙事应升级为：

> EVQ-cosh 的关键不是引入了一个要大调特调的新超参数，而是给出了一个几乎免调的解析默认值。`tau*=d_head/sqrt(L)` 已经足够把搜索空间压缩到一个非常小的近最优带内。

---

## 6. 推荐正文措辞

可直接复用这句：

> A 99-run sweep across 9 `(L, H, d_head)` configurations and 3 seeds shows that the theoretical scaling `tau*=d_head/sqrt(L)` is a robust near-optimal prior rather than a fragile heuristic: it is exactly best in 3/9 settings, top-2 in 6/9, top-3 in 8/9, and every empirical optimum stays within `1.5x` of the theoretical value.

更保守版本：

> The theory captures the dominant scaling correctly, while finite-capacity experiments exhibit a mild right-shift of the optimum relative to the closed-form prediction.
