# Phase17: 454M L=512 Continue-to-1B + YaRN Overlay Results

> 日期: 2026-03-09
> 状态: **COMPLETE**
> 服务器: `autodl-container-16864f9d88-db19646c`
> 训练目录: `REMOTE_RUN_ROOT/evq_phase17`
> 评测目录:
> - raw long eval: `REMOTE_RUN_ROOT/evq_phase17/eval_ckpt25_50_long_20260309_135414`
> - raw + YaRN overlay: `REMOTE_RUN_ROOT/evq_phase17/eval_yarn_overlay_20260309_140844`

---

## 0. 实验设置

| 参数 | 值 |
|------|-----|
| 模型 | `454M` (`24L x 1024H x 16 heads`, `d_head=64`) |
| 训练长度 | `L_train = 512` |
| 训练 token | `25% ≈ 0.5B`, `50% ≈ 1.0B` |
| 对比 | `Geo` vs `EVQ tau=2.8` |
| 评测长度 | `4K, 8K, 16K, 24K, 32K` |
| passkey | `depth=0.5`, lengths=`4K, 8K, 16K`, `10 trials` |
| YaRN 方式 | **推理时 overlay**，每个长度使用 `scale = L / 512` 动态替换频率 |

**本文采用的权威口径**

- 所有 `raw` 与 `+YaRN` 数值都来自同一套 harness
- raw 直接使用 checkpoint 自带的 `inv_freq.npy`
- `+YaRN` 在同一 checkpoint 上按长度动态 overlay
- 因此本文中的 raw / +YaRN 对比是自洽的，应优先于之前用解析式重建频率的 quick check

---

## 1. Executive Summary

### 1.1 继续用短程目标训练，raw 长程外推会继续变差

从 `25% -> 50%`，`Geo` 和 `EVQ` 的 raw 长程 PPL 都上升了，也就是都变差了。

- `Geo raw` 在 `4K-32K` 平均恶化 `+24.4%`
- `EVQ raw` 在 `4K-32K` 平均恶化 `+62.2%`

这说明：

- 固定 `L=512` 继续训练，确实会让模型更贴近短程分布
- 但这种训练不会自动改善长程 extrapolation
- 相反，raw 长程能力会进一步退化

### 1.2 但这不意味着 EVQ 在现实 setting 下更弱

关键发现是：

- `YaRN` 对 `Geo` 有帮助
- 但 `YaRN` 对 `EVQ` 的帮助**大得多**

平均 `4K-32K` 的 `raw -> +YaRN` PPL 改善：

| 方法 | 平均改善 |
|------|----------|
| `Geo 25%` | `+35.4%` |
| `EVQ 25%` | `+81.2%` |
| `Geo 50%` | `+35.6%` |
| `EVQ 50%` | `+84.9%` |

### 1.3 最强 practical recipe 不是 Geo+YaRN，而是 EVQ+YaRN

同一 checkpoint 下，`EVQ+YaRN` 相对 `Geo+YaRN` 的 `4K-32K` 平均优势：

- `25%`: `+87.4%`
- `50%`: `+86.3%`

也就是说，**EVQ 不是被 YaRN 替代，而是给 YaRN 提供了更强的底座**。

---

## 2. Raw Results

### 2.1 Raw PPL

| Checkpoint | Method | 4K | 8K | 16K | 24K | 32K |
|------------|--------|----|----|-----|-----|-----|
| `25%` | `Geo` | `31.171` | `75.236` | `141.864` | `193.345` | `215.803` |
| `25%` | `EVQ` | `8.604` | `28.037` | `71.439` | `116.383` | `135.266` |
| `50%` | `Geo` | `34.467` | `100.013` | `181.889` | `228.093` | `285.620` |
| `50%` | `EVQ` | `13.602` | `50.610` | `118.956` | `170.887` | `215.182` |

### 2.2 Raw 25% -> 50% 变化

| 方法 | 4K | 8K | 16K | 24K | 32K | 4K-32K 平均 |
|------|----|----|-----|-----|-----|-------------|
| `Geo raw` | `+10.6%` | `+32.9%` | `+28.2%` | `+18.0%` | `+32.4%` | `+24.4%` |
| `EVQ raw` | `+58.1%` | `+80.5%` | `+66.5%` | `+46.8%` | `+59.1%` | `+62.2%` |

**结论**

- 在 raw setting 下，`50%` 比 `25%` 更差
- 这条结论对 `Geo` 和 `EVQ` 都成立
- 而且 `EVQ raw` 的退化更明显

这支持下面这句话：

> 继续用短程目标训练会巩固短程拟合，但不会保住 raw 长程 extrapolation。

---

## 3. YaRN Overlay Results

### 3.1 Geo + YaRN

| Checkpoint | 4K | 8K | 16K | 24K | 32K |
|------------|----|----|-----|-----|-----|
| `Geo 25% + YaRN` | `17.517` | `55.250` | `74.735` | `115.637` | `174.503` |
| `Geo 50% + YaRN` | `19.946` | `63.749` | `102.889` | `149.163` | `224.743` |

### 3.2 EVQ + YaRN

| Checkpoint | 4K | 8K | 16K | 24K | 32K |
|------------|----|----|-----|-----|-----|
| `EVQ 25% + YaRN` | `2.696` | `5.436` | `7.829` | `12.976` | `28.393` |
| `EVQ 50% + YaRN` | `2.742` | `6.224` | `11.567` | `19.515` | `46.666` |

### 3.3 Raw -> +YaRN 改善幅度

| 方法 | 4K | 8K | 16K | 24K | 32K | 平均 |
|------|----|----|-----|-----|-----|------|
| `Geo 25%` | `+43.8%` | `+26.6%` | `+47.3%` | `+40.2%` | `+19.1%` | `+35.4%` |
| `EVQ 25%` | `+68.7%` | `+80.6%` | `+89.0%` | `+88.9%` | `+79.0%` | `+81.2%` |
| `Geo 50%` | `+42.1%` | `+36.3%` | `+43.4%` | `+34.6%` | `+21.3%` | `+35.6%` |
| `EVQ 50%` | `+79.8%` | `+87.7%` | `+90.3%` | `+88.6%` | `+78.3%` | `+84.9%` |

**最重要的观察**

- `Geo + YaRN` 当然更好
- 但 `EVQ + YaRN` 的收益远远更大
- 这不是加一点点，而是从 `32K` 的三位数 PPL 直接压到几十甚至个位数

例如：

- `EVQ 25% @16K`: `71.439 -> 7.829`
- `EVQ 50% @16K`: `118.956 -> 11.567`

---

## 4. EVQ+YaRN vs Geo+YaRN

| Checkpoint | 4K | 8K | 16K | 24K | 32K | 平均 |
|------------|----|----|-----|-----|-----|------|
| `25%` EVQ+YaRN over Geo+YaRN | `+84.6%` | `+90.2%` | `+89.5%` | `+88.8%` | `+83.7%` | `+87.4%` |
| `50%` EVQ+YaRN over Geo+YaRN | `+86.3%` | `+90.2%` | `+88.8%` | `+86.9%` | `+79.2%` | `+86.3%` |

这个表已经足够支撑 practical claim：

> 在短程训练、长程部署的现实设置里，最佳组合是 `EVQ + YaRN`，而不是 `Geo + YaRN`。

---

## 5. Passkey: Supporting Only

当前这轮 passkey 只做了：

- `depth=0.5`
- `10 trials`
- `4K / 8K / 16K`

所以它更适合做 supporting signal，不适合做 headline。

### 5.1 比较稳定的点

- `Geo 25% + YaRN @16K`: retrieval `40%`
- `EVQ 25% + YaRN @16K`: retrieval `70%`

- `Geo 50% + YaRN @16K`: retrieval `30%`
- `EVQ 50% + YaRN @16K`: retrieval `60%`

### 5.2 不稳定的点

- `EVQ 50% + YaRN` 在 `4K/8K` 上没有像 PPL 一样稳定拉开
- 这更像是低 trial 数和 single-seed 下的方差问题

因此，这轮主结论应以 **PPL** 为主，passkey 只作补充说明。

---

## 6. 最终解释

这轮结果最适合写成下面这段逻辑：

1. **继续用短程目标训练**  
   会让模型更贴合短程分布，因此 raw 长程 extrapolation 会变差。

2. **这不等于 EVQ 在现实 setting 下没用**  
   因为实际部署不是只能用 raw，而是可以叠加 inference-time extrapolation 方法。

3. **YaRN 并不会替代 EVQ，而是放大 EVQ 的优势**  
   同样的 YaRN overlay，`EVQ` 拿到的收益远大于 `Geo`。

4. **所以 practical recipe 是 EVQ+YaRN**  
   如果现实世界训练长度就是短的，那么最合理的结论不是“EVQ 更吃亏”，而是：
   **EVQ provides the better substrate for YaRN.**

---

## 7. 当前可支撑的主叙事

不建议写成：

- “更多短程训练会让 EVQ 越来越强”

因为这轮不支持这个命题。

建议写成：

- **short-only continued training hurts raw long-context extrapolation for both Geo and EVQ**
- **but EVQ unlocks much larger YaRN gains than Geo**
- **therefore EVQ+YaRN, not Geo+YaRN, is the strongest practical recipe in short-train / long-test settings**

更短的 paper 句式可以直接写：

> Even when continued short-context training weakens raw long-range extrapolation, EVQ remains the better substrate for inference-time extrapolation: YaRN yields much larger gains on EVQ than on Geo, making EVQ+YaRN the strongest practical combination.

---

## 8. 对 DAPE 叙事的意义

如果 `DAPE` 类工作依赖大量训练 token 来“硬学”长程结构，那么这轮结果说明：

- 只靠继续喂短程数据，并不能自然解决长程问题
- raw 长程能力甚至可能继续恶化
- 真正有效的路线可能是：
  - 训练时做更合理的频率分配（`EVQ`）
  - 推理时做更合理的外推校正（`YaRN`）

也就是把“shape optimization”和“inference-time scaling”拆开，而不是把所有希望都压在继续短程训练上。
