# 2026-03-14 Staged Continuation Diagnostic Report

> 日期: 2026-03-14
> 状态: **COMPLETE**
> 范围: 记录今日围绕 “为何 fixed-length EVQ 稳定成立，而 staged continuation 频繁出现异常” 的诊断实验、协议修正与综合判断
> 结论级别: **diagnostic / protocol audit**, 不是新的 paper-ready 主结果

---

## 0. 执行摘要

今天的诊断把问题压到了一个更清晰的位置。

1. **fixed-length 主结论没有坏。**
   在干净的 `125M, L=1024, 400M tokens, pure LM, FineWeb-Edu` 设定下，EVQ 仍然表现为：
   - 短程几乎不亏
   - 长程 (`4K/8K/16K`) 稳定优于 Geo

2. **之前那次 “PPL 爆炸” 不能归因于 `tau` retarget。**
   在四阶段 progressive probe (`256→512→1024→2048`) 中，
   `evq_dynamic` 在所有评测长度都优于 `evq_frozen`，
   说明“按长度改 `tau` 增加学习负担”这一说法不成立，至少不是主因。

3. **staged 的异常仍然真实存在。**
   虽然 `evq_dynamic > evq_frozen`，但在今天的 staged mixed-data 诊断里，
   `geo` 仍优于两条 EVQ 线。
   这说明 staged 的问题更像是：
   - continuation path dependence
   - mixed objective interference
   - 小模型 / 小预算下的容量与训练充分性不足
   而不是单独的 `tau surgery`。

4. **今天最严重的负结果来自错误评测协议，而不是模型本身。**
   一次误用 `proof-pile-2` 的 OOD eval 把绝对 PPL 拉到了 `600+`，造成了错误的负面印象。
   在改回严格 `FineWeb-Edu` 之后，结果恢复到合理区间。

5. **标准 YaRN 再次放大了 EVQ 的 fixed-length 长程优势。**
   用仓库标准 progressive YaRN（不是旧版假 YaRN / NTK-aware）后，
   `EVQ+YaRN` 在 `4K/8K/16K` 相比 `Geo+YaRN` 分别达到
   `-45.4% / -53.7% / -56.0%` 的优势。

一句话总结：

> **fixed-length EVQ 结论依然干净；staged continuation 的问题不在 `tau` 本身，而在 staged protocol。**

---

## 1. 今日实验清单

| ID | 实验 | 目的 | 结论 |
|----|------|------|------|
| A | `stage1 @512` 原始 checkpoint 审计 (`seed43/44`) | 确认基础 checkpoint 健康 | EVQ 在 `1K/2K/4K/8K` 全胜 |
| B | `from-scratch @2048, 100M, frozen tau` vs Geo | 快速看“不改 tau”是否明显伤害 PPL | 没有；EVQ PPL 优于 Geo，但 budget 太小，不作主结论 |
| C | `256→512→1024→2048, 4×25M` 三臂 probe (`geo / evq_dynamic / evq_frozen`) | 隔离 `tau retarget` 是否增加学习负担 | `evq_dynamic > evq_frozen`，否掉“改 tau 是主因” |
| D | `125M, 256→512, 200M + 200M` (`geo` vs `evq_dynamic`) | 看小模型 staged 是否出现带通行为 | 是；EVQ 只在中程赢，近程和更远程输 |
| E | `125M, fixed L=1024, 400M, pure LM` (`geo` vs `evq`) | 重新夯实 fixed-length 主结论 | EVQ 在长程恢复稳定优势 |
| F | `Phase17H strict FineWeb re-eval` | 修复错误的 OOD eval | `proof-pile-2` 结果作废，严格 FineWeb 结果有效 |
| G | `Phase17H + standard YaRN` | 用仓库标准 YaRN 检查 `EVQ+YaRN` | EVQ 长程优势被进一步显著放大 |

说明：

- 今日部分 staged 目录在远端因磁盘压力被删除，但关键数值已在本报告中保留。
- `NIAH` 在小预算 from-scratch probe 中未学成，因此今天不作为 headline 判断依据。

---

## 2. 协议与实现修正

### 2.1 一次错误的 OOD eval：`proof-pile-2`

今天最重要的负面教训，是一次把 fixed-length clean anchor 误评成了 OOD probe。

错误做法：

- 训练数据: `FineWeb-Edu`
- 评测数据: `proof-pile-2`

这会产生两个问题：

1. 绝对 PPL 会显著变差
2. EVQ 相对 Geo 的优势会被压小

因此，这一版结果不能用于判断 fixed-length 的主结论。

### 2.2 `load_val("fineweb-edu")` 的静默 fallback 风险

仓库里的 `run_evq_sweep.load_val(dataset="fineweb-edu")` 存在 fallback candidates：

- `HuggingFaceFW/fineweb-edu`
- `cerebras/SlimPajama-627B`
- `roneneldan/TinyStories`

如果 FineWeb 加载失败，代码可能静默回退到别的数据源，再把 tensor 仍保存成 `val_fineweb-edu_5000000.pt`。

这对 clean eval 是不可接受的。

因此，今天专门生成了严格的 FineWeb-Edu 验证 tensor，并显式推到远端：

- 本地: `data/fineweb_val_cache/val_fineweb-edu_5000000_strict.pt`
- 远端: `REMOTE_RUN_ROOT/evq_phase17h_125m_L1024_fixed/data_cache/val_fineweb-edu_5000000.pt`

### 2.3 标准 YaRN 的统一口径

今天最终采用的 YaRN 口径来自：

- `scripts/core_text_phases/phase11_yarn_eval.py`
- `scripts/core_text_phases/eval_pe_baselines.py`

关键规则：

- 使用 **真正的 progressive YaRN**
- `scale = L_eval / L_train`
- `20%-90%` channel smoothstep ramp
- attention temperature correction

这和旧版某些脚本里的“全通道同因子缩放”不是一回事。后者更接近 NTK-aware，不应再称为 YaRN。

---

## 3. 结果

### 3.1 实验 A: `stage1 @512` 原始 checkpoint 审计

目的：确认基础 `L=512` checkpoint 本身没有问题。

直接读取原始 `results.json`：

- `seed43`

| Model | PPL@1K | PPL@2K | PPL@4K | PPL@8K |
|---|---:|---:|---:|---:|
| Geo | `48.342` | `112.958` | `227.645` | `338.396` |
| EVQ | `41.742` | `82.918` | `180.143` | `286.112` |

- `seed44`

| Model | PPL@1K | PPL@2K | PPL@4K | PPL@8K |
|---|---:|---:|---:|---:|
| Geo | `48.977` | `113.364` | `236.176` | `360.116` |
| EVQ | `42.460` | `86.113` | `190.943` | `310.853` |

结论：

- `stage1 @512` 的原始模型是健康的
- EVQ 在两个 seed 上、四个长度上全部优于 Geo
- 所以 staged 问题不是 “初始 512 checkpoint 就坏了”

---

### 3.2 实验 B: `from-scratch @2048, 100M, frozen tau` vs Geo

设定：

- from-scratch
- `L=2048`
- `100M` tokens
- `5%` single-passkey mix
- EVQ `tau=2.828` 固定，不改到 `1.414`

最终 PPL：

| Length | EVQ frozen-tau | Geo | EVQ vs Geo |
|---|---:|---:|---:|
| `1K` | `1.5859` | `1.6156` | `-1.8%` |
| `2K` | `1.2029` | `1.2098` | `-0.6%` |
| `4K` | `1.1029` | `1.1368` | `-3.0%` |
| `8K` | `73.4120` | `83.9726` | `-12.6%` |
| `16K` | `153.2902` | `176.1073` | `-13.0%` |

备注：

- 这是一个 **小预算 probe**，目的是看“不改 tau”会不会直接把 PPL 搞坏
- 结果并没有支持这种说法
- 但 `NIAH` 基本没学起来，因此这组不作 headline 结论

结论：

- 在小预算 from-scratch `@2048` 下，冻结 `tau` 并不会让 EVQ 的 PPL 直接输给 Geo
- 这更像是一个排错 probe，而不是最终答案

---

### 3.3 实验 C: `256→512→1024→2048, 4×25M` 三臂 staged probe

目的：隔离“按长度改 `tau` 是否增加学习负担”。

三条线：

- `geo`
- `evq_dynamic`: 每段按当前长度 retarget `tau`
- `evq_frozen`: `tau` 固定在初始段值

最终 PPL@`1K/2K/4K/8K`：

| Arm | 1K | 2K | 4K | 8K |
|---|---:|---:|---:|---:|
| `geo` | `64.833` | `78.943` | `35.767` | `73.962` |
| `evq_dynamic` | `67.075` | `86.752` | `39.064` | `78.381` |
| `evq_frozen` | `72.024` | `95.380` | `44.415` | `86.752` |

关键比较：

- `evq_dynamic` 相对 `evq_frozen`
  - `1K`: `-6.9%`
  - `2K`: `-9.0%`
  - `4K`: `-12.0%`
  - `8K`: `-9.7%`

结论：

- `evq_dynamic` 在所有长度都优于 `evq_frozen`
- 所以“改 `tau` 会增加学习负担，导致 PPL 爆炸”这一归因被今天这组结果基本否掉
- 但 `geo` 仍然是三者中最优

因此，这组支持的是：

> **`tau retarget` 不是 staged 失败的主因；真正的问题在 staged protocol 其他部分。**

---

### 3.4 实验 D: `125M, 256→512, 200M + 200M` 两阶段 probe

目的：看更小模型、更短 staged setting 下是否会出现带通型交叉。

最终 PPL：

| Length | Geo | EVQ dynamic | EVQ vs Geo |
|---|---:|---:|---:|
| `512` | `17.526` | `18.114` | `+3.36%` |
| `1024` | `20.225` | `17.151` | `-15.20%` |
| `2048` | `43.907` | `42.706` | `-2.74%` |
| `4096` | `65.467` | `74.277` | `+13.46%` |
| `8192` | `106.902` | `115.917` | `+8.43%` |

结论：

- EVQ 不是全长度单调更好
- 它在 `1K/2K` 能赢，但 `512` 和 `4K/8K` 又输回去
- 这说明在小模型 staged regime 里，EVQ 更像是 **带通增益**，而不是固定长度里那种单调长程优势

这也解释了为什么今天对 staged 的感受是“互有胜负、很怪”。

---

### 3.5 实验 E/F: `125M, fixed L=1024, 400M, pure LM` clean anchor

#### 3.5.1 错误版本：`proof-pile-2` OOD eval

最初误用了 `proof-pile-2`，得到：

| Length | Geo | EVQ |
|---|---:|---:|
| `512` | `79.397` | `80.303` |
| `1024` | `68.599` | `71.188` |
| `2048` | `114.251` | `119.032` |
| `4096` | `270.272` | `263.761` |
| `8192` | `492.678` | `478.303` |
| `16384` | `642.726` | `590.711` |

这组数值现在应被视为：

- **无效的 fixed-length 主结果**
- 仅能作为 OOD 诊断痕迹保留

#### 3.5.2 修正版本：严格 `FineWeb-Edu` multi-length eval

在修正为严格 `FineWeb-Edu` 后，结果变为：

| Length | Geo | EVQ | EVQ vs Geo |
|---|---:|---:|---:|
| `512` | `37.224` | `37.172` | `-0.14%` |
| `1024` | `36.607` | `36.523` | `-0.23%` |
| `2048` | `58.954` | `59.728` | `+1.31%` |
| `4096` | `131.242` | `127.638` | `-2.75%` |
| `8192` | `271.824` | `248.246` | `-8.67%` |
| `16384` | `414.457` | `356.826` | `-13.91%` |

结论：

- fixed-length 主结论重新成立
- `512/1024` 基本持平
- `2048` 轻微落后
- `4K/8K/16K` EVQ 稳定反超，且优势随长度增加

这与我们此前大量 fixed-length 结果是同方向的，只是幅度比最 PE-dominant 的短训练长度实验要小。

---

### 3.6 实验 G: `Phase17H + standard YaRN`

使用仓库标准 YaRN 的单长度评测协议后，得到：

#### raw（单长度评测口径）

| Length | Geo raw | EVQ raw | EVQ vs Geo |
|---|---:|---:|---:|
| `512` | `37.224` | `37.172` | `-0.14%` |
| `1024` | `34.236` | `34.362` | `+0.37%` |
| `2048` | `53.878` | `55.227` | `+2.50%` |
| `4096` | `139.789` | `133.285` | `-4.65%` |
| `8192` | `255.975` | `230.401` | `-9.99%` |
| `16384` | `403.482` | `338.977` | `-15.99%` |

#### +YaRN（`scale = L_eval / 1024`）

| Length | Geo+YaRN | EVQ+YaRN | EVQ vs Geo |
|---|---:|---:|---:|
| `512` | `37.224` | `37.172` | `-0.14%` |
| `1024` | `34.236` | `34.362` | `+0.37%` |
| `2048` | `37.768` | `32.204` | `-14.73%` |
| `4096` | `79.996` | `43.668` | `-45.41%` |
| `8192` | `143.464` | `66.414` | `-53.71%` |
| `16384` | `220.581` | `97.054` | `-56.00%` |

#### YaRN 对各自 raw 的改善

| Length | Geo `+YaRN vs raw` | EVQ `+YaRN vs raw` |
|---|---:|---:|
| `2048` | `-29.9%` | `-41.7%` |
| `4096` | `-42.8%` | `-67.2%` |
| `8192` | `-44.0%` | `-71.2%` |
| `16384` | `-45.3%` | `-71.4%` |

结论：

- 标准 YaRN 对两条线都有效
- 但对 EVQ 的增益显著更大
- `EVQ+YaRN` 在 `4K+` 的优势不再是“小赢”，而是大幅领先

因此，今天 fixed-length 线的最强结论是：

> **在标准 YaRN 下，EVQ 的长程优势会被进一步放大。**

备注：

- 这里的 raw 是 **单长度评测协议**，和上一节 multi-length raw 不应直接逐项混表
- 但它们给出的定性方向是一致的：短程接近，长程 EVQ 更强

---

## 4. 综合判断

### 4.1 今天可以被视为“已确认”的结论

1. **fixed-length EVQ 结论仍然成立。**
   今天最干净的 `125M, L=1024, 400M, pure LM, strict FineWeb` 结果已经重新证明：
   EVQ 的优势在短程不明显，但在长程稳定显现。

2. **标准 YaRN 会进一步放大 EVQ 的 fixed-length 长程优势。**
   `EVQ+YaRN` 相对 `Geo+YaRN` 在 `4K/8K/16K` 的优势达到
   `45%-56%`。

3. **staged 的异常不是 `tau retarget` 单独导致的。**
   因为 `evq_dynamic > evq_frozen` 已经直接否掉了这个解释。

### 4.2 今天仍未完全钉死的结论

1. **为什么 staged 在小模型上难以复现 750M 那种“纯赢”结果？**
   今天更合理的候选原因是：
   - 小模型容量不足
   - 总有效训练量不足
   - continuation path dependence
   - mixed objective / passkey mix 对 staged 的干扰更大

2. **为什么 750M `2048→4096` 能成功，而今天 staged probe 不稳定？**
   当前最可能的解释是：
   - `750M` 更大
   - 总训练量更高
   - 目标混合更稳
   - staged path 的不利因素被更高容量 / 更高 token budget 抵消

### 4.3 今天最重要的叙事修正

今天最大的叙事修正不是“EVQ 失败了”或“Geo 更稳”，而是：

> **fixed-length theorem 没问题；问题出在 staged continuation 不是 fixed-length theorem 的直接 corollary。**

换句话说：

- fixed-length from-scratch 学到的是一个一致的新解
- staged continuation 学到的是一个带历史路径依赖的解

这两者不应该被混成同一个命题。

---

## 5. 建议的下一步

如果目标是 **最快把 staged 的根因钉死**，最有性价比的是：

1. **做一个更干净的 staged pure-LM 对照**
   - 去掉 passkey / mixed objective
   - 只测 `512→1024`
   - 严格匹配 fixed-length 的数据分布和 eval 口径

2. **做一个更高预算的 fixed-length 对照**
   - `125M @512` 或 `125M @1024`
   - 拉高到 `800M` 或 `1B` tokens
   - 看 EVQ 优势是否随训练充分性继续扩大

3. **如果资源允许，复制 750M 的干净条件**
   - 更大模型
   - 更高总 token budget
   - 尽量避免额外 objective 干扰

如果目标是 **最快夯实论文主叙事**，则更建议：

1. 继续扩 fixed-length clean anchor
2. 补多 seed
3. 用标准 YaRN 统一 overlay 评测

---

## 6. 可直接引用的结论句

### 6.1 Fixed-length

> In a clean 125M fixed-length run at `L=1024` with `400M` pure language-modeling tokens and strict FineWeb-Edu evaluation, EVQ remains roughly tied with geometric RoPE near the training length but regains a clear advantage at longer contexts, reaching `-13.9%` lower PPL at `16K`.

### 6.2 YaRN

> Under the repository-standard progressive YaRN overlay, the EVQ advantage becomes substantially larger: compared with Geo+YaRN, EVQ+YaRN achieves `-45.4%`, `-53.7%`, and `-56.0%` lower PPL at `4K`, `8K`, and `16K`, respectively.

### 6.3 Staged diagnosis

> The staged failures observed this week cannot be attributed primarily to `tau` retargeting: in a controlled progressive probe, dynamically retargeted EVQ consistently outperformed frozen-`tau` EVQ. The remaining instability is therefore more likely due to staged continuation path dependence, limited model capacity, limited token budget, or mixed-objective interference.
