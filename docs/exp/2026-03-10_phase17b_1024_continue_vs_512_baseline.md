# Phase17B: 454M `512 -> 1024` Continue vs Yesterday's `L_train=512` Baseline

> 日期: 2026-03-10
> 状态: **COMPLETE**
> 基线报告: `docs/exp/2026-03-09_phase17_evq_yarn_overlay_results.md`
> 当前 raw eval 本地缓存（gitignored）: `results/core_text/phase17b_1024_continue/final_eval_summary.json`
> 远端原始产物: `REMOTE_RUN_ROOT/evq_phase17b/final_eval_summary.json`

---

## 0. 比较口径

这份对照报告回答的是：

> 从昨天 `phase17` 的 `L_train=512`、`50% checkpoint (~1.0B tokens)` 出发，再额外做一段 `L=1024`、`1B` continuation，关键指标相对昨天那份 `512` 报告到底变了什么。

因此它是一个 **staged continuation before/after** 对照，不是严格的等预算比较。

- 昨天的基线点:
  - 来自 [`docs/exp/2026-03-09_phase17_evq_yarn_overlay_results.md`](REPO_ROOT/docs/exp/2026-03-09_phase17_evq_yarn_overlay_results.md)
  - 使用 `phase17` 的 `50% checkpoint`
  - 训练长度保持 `L_train=512`
  - raw PPL 评到 `4K / 8K / 16K / 24K / 32K`
- 今天的新结果:
  - 从昨天同一个 `50% checkpoint` 起步
  - 追加 `1B` token、把 continuation 长度切到 `L=1024`
  - raw PPL 评到 `512 / 1K / 2K / 4K / 8K / 16K`

可直接同口径比较的只有 raw PPL 的重叠长度:

- `4K`
- `8K`
- `16K`

`24K / 32K` 这轮还没测，所以本报告不对那两档做结论。

passkey 也不做昨天-今天直比，因为 protocol 已变：

- 昨天报告主要是 `depth=0.5`, `10 trials`, 且重点写了 `+YaRN`
- 今天这轮是 raw 模型、`5 depths x 20 trials`

所以这份报告的 headline metric 以 **raw PPL** 为准。

---

## 1. Executive Summary

### 1.1 最关键指标: `EVQ raw PPL@16K`

昨天 `L_train=512` 基线里，`EVQ` 在 `16K` 的 raw PPL 是 `118.956`。  
今天做完 `512 -> 1024` continuation 之后，`EVQ` 在 `16K` 的 raw PPL 变成 `57.635`。

这意味着：

- `EVQ raw PPL@16K`: `118.956 -> 57.635`
- 相对下降 `-51.5%`

这是当前最值得盯的单点指标，因为它同时满足：

- 是昨天和今天都测了的最长重叠长度
- 直接反映 raw long-context collapse 是否被明显缓解
- 不依赖 YaRN overlay，也不受额外 inference trick 影响

### 1.2 Geo 也改善，但 EVQ 在最远重叠长度上的终点更强

- `Geo raw PPL@16K`: `181.889 -> 120.084`，下降 `-34.0%`
- `EVQ raw PPL@16K`: `118.956 -> 57.635`，下降 `-51.5%`

所以：

- 切到 `L=1024` continuation 之后，**两条线都明显变好**
- 但在最关键的 `16K raw` 终点上，**EVQ 仍然更强**
- 而且 `EVQ over Geo` 的 `16K` 优势从昨天的 `34.6%` 扩大到了今天的 `52.0%`

### 1.3 这轮结果的核心结论

昨天的 `512` 报告说明：

- 持续做短程训练，会让 raw 长程 extrapolation 继续恶化

今天的 `1024 continue` 结果说明：

- 一旦把 continuation 长度从 `512` 提到 `1024`，这个恶化趋势会被明显扭转
- raw 长程 PPL 会显著回落
- EVQ 在长程端仍然保持优势，尤其是 `16K`

最稳妥的叙事应写成：

> relative to the `L_train=512` baseline, the `512 -> 1024` continuation materially improves raw long-context PPL for both Geo and EVQ, and EVQ remains the stronger long-range endpoint, especially at `16K`.

---

## 2. Raw PPL: Yesterday `512` vs Today `512 -> 1024`

### 2.1 Geo

| Length | Yesterday `L_train=512` | Today `512 -> 1024` | Relative change |
|--------|-------------------------|---------------------|-----------------|
| `4K` | `34.467` | `9.208` | `-73.3%` |
| `8K` | `100.013` | `44.807` | `-55.2%` |
| `16K` | `181.889` | `120.084` | `-34.0%` |

`4K-16K` 平均 PPL:

- yesterday: `105.456`
- today: `58.033`
- relative: `-45.0%`

### 2.2 EVQ

| Length | Yesterday `L_train=512` | Today `512 -> 1024` | Relative change |
|--------|-------------------------|---------------------|-----------------|
| `4K` | `13.602` | `6.607` | `-51.4%` |
| `8K` | `50.610` | `27.782` | `-45.1%` |
| `16K` | `118.956` | `57.635` | `-51.5%` |

`4K-16K` 平均 PPL:

- yesterday: `61.056`
- today: `30.675`
- relative: `-49.8%`

### 2.3 这一组表真正说明什么

- `Geo` 在重叠长度上全部改善
- `EVQ` 在重叠长度上也全部改善
- `EVQ` 的平均改善幅度略大于 `Geo`
- 最远重叠长度 `16K` 上，`EVQ` 的改善尤其明显

换句话说，昨天那条“继续短程训练会继续伤 raw 长程”的趋势，在把 continuation 长度升到 `1024` 后被反向修正了。

---

## 3. EVQ vs Geo Gap: Yesterday vs Today

为了避免只看“都变好了”，还需要看 `EVQ` 相对 `Geo` 的差距有没有保住。

这里用:

`EVQ advantage = 1 - PPL_EVQ / PPL_GEO`

数值越大，表示 EVQ 相对 Geo 越强。

| Length | Yesterday `L_train=512` | Today `512 -> 1024` | Change |
|--------|-------------------------|---------------------|--------|
| `4K` | `60.5%` | `28.2%` | `-32.3pp` |
| `8K` | `49.4%` | `38.0%` | `-11.4pp` |
| `16K` | `34.6%` | `52.0%` | `+17.4pp` |

解释要分开看：

- 在 `4K/8K`，两条线都因为 `1024` continuation 而一起下来了，所以 EVQ 对 Geo 的相对优势变窄
- 但到了 `16K`，EVQ 的下降更大，所以最终反而把相对优势拉宽了

因此更稳妥的说法不是：

- “`1024` continuation 让 EVQ 在所有长度上都更占优”

而是：

- “`1024` continuation 让两条线都明显改善，而 EVQ 在最远重叠长度 `16K` 上保住并扩大了优势”

---

## 4. Passkey: 当前只能做当日内部 supporting signal

这轮 raw passkey 的全局统计是：

| Method | Global retrieval | Mean NLL gap | AR exact match |
|--------|------------------|--------------|----------------|
| `Geo` | `0.674` | `1.4597` | `0.4` |
| `EVQ` | `0.692` | `1.8211` | `0.4` |

可以支持的最保守结论是：

- 在今天这轮 raw `1024` continuation 评测里，`EVQ` 的 global retrieval 略高于 `Geo`
- `EVQ` 的 mean NLL gap 也更高

但不建议把它直接和昨天的 passkey 结果写成 day-over-day headline，因为 protocol 已经不一致。

所以这轮报告里：

- **headline 用 raw PPL**
- passkey 只保留为 supporting signal

---

## 5. Report-Ready Claim

如果要把昨天 `512` 报告和今天 `1024 continue` 报告串起来，建议用下面这段：

> In the `L_train=512` baseline, continued short-context training degraded raw long-range extrapolation. Starting from that same checkpoint, switching the continuation stage to `L=1024` materially improved raw long-context PPL for both Geo and EVQ. The strongest single overlap metric is `EVQ raw PPL@16K`, which dropped from `118.956` to `57.635` (`-51.5%`). EVQ also remained the better long-range endpoint, with its `16K` advantage over Geo expanding from `34.6%` to `52.0%`.

更短的中文版可以直接写：

> 相对昨天的 `L_train=512` 基线，从同一个 `50% checkpoint` 切到 `L=1024` 做 continuation 后，raw 长程 PPL 明显回落；其中最关键的单点指标是 `EVQ raw PPL@16K: 118.956 -> 57.635 (-51.5%)`。Geo 也改善，但 EVQ 在 `16K` 仍然是更强的终点。

---

## 6. 当前边界与下一步

这份报告还不能回答：

- `24K / 32K` 在 raw 下会不会延续同样趋势
- `1024 continue + YaRN` 相对昨天 `512 + YaRN` 的 practical recipe 是否继续扩大优势

所以这份报告的边界要写清楚：

- **已确认**: `4K / 8K / 16K` raw 明显改善
- **已确认**: `16K` 上 EVQ 终点更强
- **未确认**: `24K / 32K` raw 以及 `+YaRN` 后的同口径对照

如果后面要继续跑 `32K`，这份报告就作为当前 canonical bridge note：

- 昨天的 `512` 报告回答 “短程继续训练会怎样”
- 今天这份 bridge 报告回答 “从同一个 512 checkpoint 切到 1024 continuation 后，关键指标怎么变”
