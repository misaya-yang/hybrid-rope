# EVQ-Cosh τ 的真实作用与 operating rule 审计

Date: 2026-07-25
Status: `analysis_only / no_new_training / working_tree_audit`
Repository snapshot: `main@498a43e3c3df22bcb3eaadf7be8954469ef9ca96`

## 0. 结论先行

### 0.1 最重要的结论

1. **[事实] 当前证据不支持任何已找到的通用 τ 修正式。**
   `τ=d_head/sqrt(L_train)` 在 Phase16 的 99-run 小模型网格中是一个有用但不稳定的
   operating prior；它相对 midpoint-Geo 的方向通常有利，却不能可靠预测相邻网格的
   最佳点。把它统一乘以 `1.25` 能明显改善该网格内的回顾性 regret，但在整组
   `L_train=1024` 留出时不再被选中，并被独立的 `L_train=128,d_head=64` 扫参反例
   否定为通用修正。

2. **[事实] raw τ 不是跨配置可比较的“变形强度”。**
   在当前 midpoint 实现中，τ 同时改变：

   - 指数密度的内部形状；
   - 有限 `K` 下采样到的最高、最低指数及实际 log-frequency span；
   - 乘上 `ln B` 后的实际 log-frequency 位移；
   - 乘上相对距离 `Δ` 后的相位位移。

   例如在 Phase16 的 `L=256,d_head=128,K=64,B=500K` 配置中，把 τ 从 6 增到
   8、10，会把采样指数 span 分别压到 midpoint-Geo span 的 `80.5%`、`61.5%`、
   `49.2%`。这不是“只改变 Cosh shape”的比较。

3. **[事实] τ 的真实直接作用点是频率向量，不是上下文长度本身。**
   更大的正 τ 把所有采样指数 `φ_k` 向 0 移动，因而把频率
   `ω_k=B^{-φ_k}` 整体推向更高频；它同时把更多通道密度集中到高频端。
   “τ 越大就是更多低频/更长程容量”这一解释在当前公式下方向相反。

4. **[推断] 训练后模型真正优化的不是一个仅由 `(d_head,L_train)` 决定的静态
   谱几何量，而是任务、数据、head/layer、训练动力学和部署长度共同决定的风险。**
   仓库中的静态 collision、phase、Q/K norm selector 都没有预测真实 NLL；
   频率与已训练通道之间还存在强烈绑定。因此，任何不读取任务信号却声称给出全局
   最优 τ 的公式都缺少识别基础。

5. **[推断] 目前最接近“理论与方法 SOTA”的方向不是另一个 scalar τ 公式，而是
   task-conditioned frequency learning / selection。**
   2026 年的 LeRoPE 直接学习每个频带的 log-scale；AdaRoPE 进一步学习
   per-head、per-band 频率和 head-specific temperature。它们提供了比共享 scalar τ
   更接近训练目标的自由度和梯度信号，但仍不证明通用 OOD 最优：LeRoPE 的原生
   extrapolation 反而更差，需要再配 YaRN；AdaRoPE 同时改变频率、head sharing 和
   temperature，不能被当成 τ 理论的单变量验证。

6. **[待验] 对 EVQ 最有希望的下一步不是继续拟合 raw τ，而是：**

   - 用 endpoint/span-anchored Cosh 隔离“内部 allocation shape”；
   - 用 `θ=τ²` 或实际 phase/log-frequency deformation 代替 raw τ 作为搜索坐标；
   - 把最终选择写成带 ID 约束、跨 seed 与部署长度风险的 bilevel problem；
   - 在确认共享 shape 有增益后，再决定是否值得升级到 per-head 或 per-band 学习。

### 0.2 对三个实际问题的直接回答

| 问题 | 审计答案 | 证据等级 |
| --- | --- | --- |
| 是否已有比旧公式更好的简单修正？ | Phase16 内 `1.25×` 最好，但外部已有反例；不能作为通用新公式 | [事实] |
| τ 真正控制什么？ | Cosh quantile warp 经有限采样后造成的频率与相位变形；它不是独立的“长程强度” | [事实] |
| 理论上应优化什么？ | 训练后、任务条件化、部署分布下的风险；scalar τ 只是受限参数族中的一个坐标 | [推断] |

---

## 1. 审计范围、证据标签与 provenance 边界

本文统一使用：

- **[事实]**：可由当前代码、配置、tracked/retained artifact 或一手论文直接复核；
- **[推断]**：多个事实共同支持，但仓库没有完成决定性验证；
- **[待验]**：明确可证伪、尚无足够证据的假设。

本次没有启动训练，没有修改训练代码、论文、playbook 或已有实验结果。新建的只有
本审计文件；所有数值重算均由临时只读脚本完成。

### 1.1 “99raw” 当前到底是否存在

**[事实] 当前工作树里找不到 Phase16 的 99 个 `runs/*/result.json`、spec、日志、
checkpoint 或逐步训练曲线。** 当前可定位的 Phase16 证据是：

- `data/curated/phase16_99run_manifest.csv`
- `data/curated/phase16_99run_manifest.meta.json:3-20`
- `results/theory/phase16_formula_optimality_sweep_local_m4_wikitext/reports/report.md`
- `rebuttal/rebuttal_0723/theory_results/PHASE16_99RUN_RAW_REANALYSIS_20260724.md`
- runner `scripts/core_text_phases/phase16_formula_optimality_sweep.py`

metadata 明确写明 `source_bundle.available_in_current_checkout=false`，预期 raw 输入是
`pilot_plan.json`、`confirm_plan.json` 和 `runs/*/result.json`
（`data/curated/phase16_99run_manifest.meta.json:12-20`）。

现有 reanalysis 报告又写道，当时那台 workstation 上 ignored raw tree 曾存在并能
重建 manifest（
`rebuttal/rebuttal_0723/theory_results/PHASE16_99RUN_RAW_REANALYSIS_20260724.md:41-61`
）。这两个陈述并非同一时间点的矛盾：前者描述可移植 checkout，后者描述一次先前的
本地恢复。但在**本次审计时点**，那个 source bundle 已不可访问。因此：

- 本文可以复算 99 行 final metrics、配置字段和 `inv_freq_hash`；
- 本文不能再次审计逐 step 曲线、optimizer state、checkpoint 内容或 raw 日志；
- 不能把先前报告中的“raw present”升级为当前 raw-backed 声明。

### 1.2 工作树边界

`main`、`HEAD` 与 `origin/main` 在审计开始时均指向
`498a43e3c3df22bcb3eaadf7be8954469ef9ca96`。工作树在本次审计前已经有用户修改与
未跟踪结果文件；本文没有覆盖或整理它们。特别是
`rebuttal/rebuttal_0723/theory_results/EXPERIMENT_REPORT_20260724.md` 在当前工作树中
是已修改文件，因此本文引用它时均视为“当前工作树证据”，不冒充干净 HEAD artifact。

---

## 2. τ 的精确数学作用

### 2.1 当前实现

当前 canonical API 使用

\[
u_k=\frac{k+1/2}{K},\qquad
\phi_\tau(u)
=1-\frac{1}{\tau}\operatorname{asinh}\!\bigl((1-u)\sinh\tau\bigr),
\qquad
\omega_k=B^{-\phi_\tau(u_k)}.
\]

实现位置是 `scripts/lib/rope/schedules.py:94-140`。当 `τ→0` 时，
`φ_τ(u)→u`，恢复的是 **midpoint-discretized Geo**，不是 native standard RoPE 的
endpoint grid。native Geo 在同一文件 `:86-91` 使用 `u_k=k/K`。

Phase16 所有 τ arms 都调用 `evq_cosh_inv_freq`；其零点 baseline 也是同一 midpoint
API，所以 Phase16 内部的 τ 比较没有 midpoint/endpoint 混杂
（`scripts/core_text_phases/phase16_formula_optimality_sweep.py:938-945`）。
但跨报告比较 “Geo” 时仍必须检查它到底是 midpoint Geo 还是 native endpoint Geo。

### 2.2 连续分布解释

由反函数可得 Cosh warp 在指数坐标上的通道密度

\[
\rho_\tau(\phi)
=\frac{du}{d\phi}
=\frac{\tau\cosh(\tau(1-\phi))}{\sinh\tau},
\qquad \phi\in[0,1].
\]

因此：

- **[事实]** `τ=0` 给出均匀 exponent density；
- **[事实]** 正 τ 越大，密度越集中在 `φ≈0`，即较高频端；
- **[事实]** 对固定 `u∈(0,1)`，`τ→∞` 时
  `φ_τ(u)≈-log(1-u)/τ→0`，所以 `ω→1`。

τ 对单个指数的精确导数是

\[
\frac{\partial \phi_\tau}{\partial\tau}
=
\frac{\operatorname{asinh}(A\sinh\tau)}{\tau^2}
-
\frac{A\cosh\tau}
{\tau\sqrt{1+A^2\sinh^2\tau}},
\qquad A=1-u.
\]

实际 log-frequency 灵敏度为

\[
\frac{\partial\log\omega_k}{\partial\tau}
=-\log B\,
\frac{\partial\phi_\tau(u_k)}{\partial\tau}.
\]

这已经说明 raw τ 不能完全独立于 base：即便 exponent warp 相同，实际
log-frequency 位移仍按 `ln B` 放大。

### 2.3 小 τ 下真正自然的局部坐标是 `θ=τ²`

Taylor 展开为

\[
\phi_\tau(u)
=u-\frac{\tau^2}{6}u(1-u)(2-u)+O(\tau^4).
\]

所以：

- `∂φ/∂τ=O(τ)`；
- 任一足够光滑的 task risk 在 τ=0 附近首先感受到的是 `θ=τ²`；
- `+τ` 与 `-τ` 给出同一 schedule，raw τ 的符号本身不可识别。

这也暴露了 learnable 实现的一处实际问题。`scripts/lib/rope/learnable_evq.py:76-93`
用 `τ=softplus(raw_tau)`；当初始化 τ 很小时，

\[
\frac{d\theta}{d\,raw_\tau}
=2\tau\,\sigma(raw_\tau)
\approx 2\tau^2,
\]

梯度会二次消失。文件 `:95-108` 的注释称 Taylor fallback “never traps τ near 0”，
但 retained evidence 恰好记录 `τ_init=0.01` 落入 softplus dead zone
（`data/curated/learnable_tau_128tok_evidence.json:34-35`）。该注释不是实验事实。

同一 learnable 文件 `:12-16,44-47` 还声称 “boundary anchoring/endpoints don't
move”。这对连续端点 `u=0,1` 成立，却不适用于实际 midpoint 样本
`u_0=1/(2K)`、`u_{K-1}=1-1/(2K)`。有限 grid 的两端会随 τ 移动。

### 2.4 τ 进入训练后模型的五层作用链

τ 的可解释作用不能停在 `ρ_τ`：

1. **连续 surrogate 层**：`τ=sqrt(β/α)` 只表示特定 convex surrogate 中两项权重比；
2. **有限采样层**：`K` 个 midpoint quantiles 同时改变端点、span 与内部 spacing；
3. **物理频率层**：指数位移乘上 `ln B` 才成为 log-frequency 位移；
4. **相位层**：部署距离 `Δ` 把它转成 `Δω_k`，并经周期函数折叠；
5. **训练后功能层**：Q/K norm、相对角、softmax、value、downstream gradient 和
   channel semantics 决定该频率是否有用。

对一个已训练 attention head，单个频带对 logit 的形式是

\[
\ell_k(\Delta)
=a_k(x)\cos(\omega_k\Delta)+b_k(x)\sin(\omega_k\Delta).
\]

因此完全相同的 `ω_k` 几何，在不同内容、head、layer 和训练权重下可以有相反的风险
导数。τ 不是模型任务的 sufficient statistic。

---

## 3. 有限 K 下 raw τ 混合了 shape 与 range

为了量化混杂，本文直接从 canonical `φ_τ` 重算：

\[
r_{\rm span}(\tau)
=
\frac{\phi_\tau(u_{K-1})-\phi_\tau(u_0)}
{u_{K-1}-u_0},
\]

\[
\eta_{\log\omega}(\tau)
=
\left[\frac1K\sum_k
\bigl(\log\omega_k(\tau)-\log\omega_k(0)\bigr)^2
\right]^{1/2}.
\]

Phase16 中两个角落的结果：

| 配置 | multiplier | τ | sampled span / midpoint-Geo | RMS `Δφ` | RMS `Δlogω`, `B=500K` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `L256,d128,K64` | 0.75 | 6 | 0.8053 | 0.3727 | 4.8910 |
|  | 1.00 | 8 | 0.6149 | 0.4211 | 5.5264 |
|  | 1.25 | 10 | 0.4921 | 0.4514 | 5.9228 |
| `L1024,d32,K16` | 0.75 | 0.75 | 1.0017 | 0.0245 | 0.3215 |
|  | 1.00 | 1.00 | 1.0019 | 0.0418 | 0.5489 |
|  | 1.25 | 1.25 | 1.0011 | 0.0622 | 0.8161 |

**[事实]** 同一个 `0.75/1/1.25 × d/sqrt(L)` multiplier 在两个角落对应的实际
频率干预量相差近一个数量级，而且 span 的变化方向和幅度强烈依赖 `K` 与 τ 所在区间。

**[推断]** 截图中观察到 `d_head=128` 时 `τ=8` “偏大”，即便最终在它的新 protocol
上成立，也可能是 finite-grid span collapse，而不是 `d_head` 指数本身错误。Phase16
同一名义配置的结果反而支持 τ 继续增至 10，说明 protocol/task 变量同样重要。

---

## 4. 99-run 重新计算：能修正什么，不能修正什么

### 4.1 设计与不可识别变量

Phase16 runner 的 profile 与 manifest 给出：

- `L_train∈{256,512,1024}`；
- head count `H∈{4,8,16}`；
- 固定 hidden size 下 `d_head∈{128,64,32}`；
- `K=d_head/2`；
- `B=500000`；
- local WikiText、约 50M model tier；
- 每 run 训练 `8,388,608` tokens；
- seed 42 的 45 个 pilot arms；
- seed 137/256 的 54 个 confirmation arms。

runner 在
`scripts/core_text_phases/phase16_formula_optimality_sweep.py:655-703`
由 `hidden_size/H` 计算 `d_head`，再围绕预测 τ 生成 multiplier grid。

这意味着：

- `d_head`、`K` 与 `H` 完全耦合；
- `B` 没有变化；
- 模型、数据、token budget 只有一个小型 regime；
- 只有三个 `L_train` 水平。

**[事实]** 99-run 无法分别估计 `d_head`、channel count `K`、head count `H` 和 base
的独立作用。因此不能从它推出一个系统依赖 `(B,K,L_train)` 的可识别公式，更不能把
拟合的 `d_head` exponent 解释为纯频谱预算定律。

### 4.2 重算指标

本文只使用每一行都有的 PPL，按历史 reanalysis 的共同指标计算

\[
R
=
\frac{\sum_{r\in\{2,4,8\}}\log_2(r+1)\log PPL_{rL}}
{\sum_{r\in\{2,4,8\}}\log_2(r+1)}.
\]

manifest 有 99 行，SHA-256 为
`39ce676ca26967434c0091e09d36824cd16d1a1a204ad464dad0a33aef7b18d5`。

### 4.3 seed-42 pilot 的真实最佳 multiplier

| 配置 | formula τ | pilot 最佳 multiplier | pilot 最佳 τ |
| --- | ---: | ---: | ---: |
| `L256,d32` | 2.000 | 1.00 | 2.000 |
| `L256,d64` | 4.000 | 1.00 | 4.000 |
| `L256,d128` | 8.000 | 1.25 | 10.000 |
| `L512,d32` | 1.414 | 1.25 | 1.768 |
| `L512,d64` | 2.828 | 1.50 | 4.243 |
| `L512,d128` | 5.657 | 0.75 | 4.243 |
| `L1024,d32` | 1.000 | 1.25 | 1.250 |
| `L1024,d64` | 2.000 | 1.25 | 2.500 |
| `L1024,d128` | 4.000 | 1.25 | 5.000 |

公式正好最佳仅 `2/9`。最佳 multiplier 包含 `.75,1,1.25,1.5`，不是一个单调的
`d_head` 或 `L_train` 修正。

### 4.4 简单规则的回顾性 regret

以每配置 pilot 最佳 arm 为 0 regret：

| 规则 | 9 配置 mean pilot NLL regret |
| --- | ---: |
| midpoint-Geo (`0×`) | 0.07723 |
| 固定 `0.75×formula` | 0.04371 |
| 原公式 (`1.00×`) | 0.04501 |
| 固定 `1.25×formula` | **0.01990** |
| 固定 `1.50×formula` | 0.05219 |
| 旧 `max(formula,≈1.4)` floor | 0.04285 |
| static softmax proxy `c_pred×formula` | **0.01990** |

`c_pred` 在这九个配置约为 `1.19`，映射到离散 grid 后等同 `1.25×`。数值实现见
`scripts/analysis/verify_c_coll.py:19-76`。但该 proxy 是连续静态 transport curvature，
不是训练后 NLL objective。

**[事实]** 旧 floor 修正几乎没有改善原公式；Phase16 内最有竞争力的简单候选是
`1.25×`，不是 floor。

**[事实]** 逐配置 leave-one-out 仍每次选择 `1.25×`，mean regret 仍为 `0.01990`。
但这九个点共享同一 corpus、model tier 与 harness，不是九个独立 deployment domains。

### 4.5 更严格的 group holdout 暴露外推失败

| 完整留出组 | 用其余 L 选择的 multiplier | 留出组 mean regret |
| --- | ---: | ---: |
| `L_train=256` | 1.25 | 0.01266 |
| `L_train=512` | 1.25 | 0.04705 |
| `L_train=1024` | **1.00** | **0.07965** |

按 `d_head` 留出时三组都选 1.25，但该方向与 `K`、`H` 完全共线，不能识别为
head-dimension law。

**[事实]** 一旦把整个未见长度组当作 domain，`1.25×` 不再稳定。这比随机留一个
高度相关配置更接近公式应满足的预测任务。

### 4.6 confirmation seeds 对 `1.25×` 的有限支持

只有 7 个配置在 confirmation 阶段同时保留了 formula 和 `1.25×`，共 14 个
seed-matched pairs：

- `1.25×` 赢 `6/7` 个 configuration means；
- 赢 `9/14` 个 pairs；
- `formula - 1.25×` 的 pair mean 为 `+0.03438 NLL`。

配置均值差：

| 配置 | formula − `1.25×` NLL |
| --- | ---: |
| `L256,d32` | -0.00599 |
| `L256,d64` | +0.00385 |
| `L256,d128` | +0.02704 |
| `L512,d32` | +0.13758 |
| `L1024,d32` | +0.01115 |
| `L1024,d64` | +0.00597 |
| `L1024,d128` | +0.06104 |

负值才是 formula 更好。该结果说明 `1.25×` 值得作为 **Phase16-local candidate**，
但 confirmation arms 的可用性源于 pilot 后的 staged selection，且缺少
`L512,d64/d128` 的 `1.25×` confirmation；不能当作预注册的全网格验证。

### 4.7 回顾性 power law 没有解决问题

直接用九个 pilot optimum 拟合

\[
\tau\approx 0.783\,
d_{\rm head}^{0.931}L^{-0.393}
\]

得到 configuration leave-one-out mean regret `0.02971`，比固定 `1.25×` 更差。
它还用 9 个受离散 grid、单 seed 和 `H/K/d` 共线影响的标签拟合 3 个参数。

**[事实]** 这个 power law 只能作为 post-hoc 描述，不能替代旧规则。

---

## 5. 截图中的 `τ=6` 主张：当前状态是“新 protocol 待验”

截图称：

- `d_head=128` 时公式 τ=8 太大；
- `0.75τ=6` 在 `1×–8×` 都优于 Geo；
- 当前仅完成 `2/12` 新结构配置。

本次仓库搜索未定位到这 `2/12` run 的 spec、日志、result JSON 或 manifest，因此不能
复核它的 model、`L_train`、base、token budget、seed、数据和 eval metric。

在当前可复算的 Phase16 **同名配置**
`L_train=256,d_head=128,B=500K,seed=42` 中：

| τ | weighted extrapolation NLL |
| ---: | ---: |
| 6 (`0.75×`) | 6.17866 |
| 8 (`1.00×`) | 6.11433 |
| 10 (`1.25×`) | **6.08370** |

并且 `τ6 − τ8` 在 `2×/4×/8×` 分别为
`+0.07398/+0.05871/+0.06362 NLL`，即 τ=6 在三个点都更差。

这不证明截图错误；它证明截图若成立，必然依赖至少一个尚未披露的 protocol 变量。
因此：

- **[事实]** 截图结论不能与 Phase16 合并成同一数据集；
- **[推断]** 它更像 task/protocol dependence 的新证据，而不是 `.75×` 新公式证据；
- **[待验]** 等 12 个配置完整且 artifacts 可用后，必须做 configuration-level
  holdout，不能先用 2 个点更新公式。

---

## 6. 仓库内其他 τ 证据：共同支持“有 basin”，不支持统一闭式

### 6.1 直接 τ calibration 与反例

| 证据族 | 原始位置 | 观察 | 对公式的含义 |
| --- | --- | --- | --- |
| 独立 selection/test anchors，`L=128,d=64,B=500K` | `rebuttal/rebuttal_0723/theory_results/EXPERIMENT_REPORT_20260724.md:59-80` | τ=5 selection 最佳；formula=5.657 很接近；τ=6、7 明显恶化 | 支持 bounded basin；反对 `1.25×` 通用化 |
| 早期 Phase6，`L=128/1024` | `docs/exp/2026-02/2026-02-26_full_experiment_report.md:150-220` | `L=128` 扫到 τ=5 仍单调改善、未 bracket；`L=1024` 只比较 0/2/2.5，τ=2 略好 | 提供长度依赖信号；标签不是精确 optimum |
| Phase8D，`L=256/512` | `docs/exp/2026-02/2026-02-26_full_experiment_report.md:408-433` | 两条曲线在测试上界 τ=5/4 仍改善，报告明确写 “no peak” | 不能验证预测的 τ=4/2.83；只能给单侧下界 |
| Phase11，`L=256` 三 seed | `docs/exp/2026-03/2026-03-04_phase11_L256_results.md:10-35` | 测试 τ=2 与 4，τ=4 更好，但没有两侧 bracket | 支持公式方向；不能定最优 |
| 早期 `L=2048` sweep | `results/legacy/paper_ready/evq_tau_sweep/evq_sweep_paper_table.csv:1-14` | 50M seed42 中 τ=1.5 是测试最佳；125M 仅稀疏 τ=0/0.2/1.5 | 与 formula≈1.414 相容；不能证明 `-1/2` exponent |
| staged dynamic retarget | `docs/exp/2026-03/2026-03-14_staged_diagnostic_report.md:150-217` | frozen τ=2.828 小 probe 优于 Geo；dynamic retarget 优于 frozen，但 Geo 又优于两者 | τ 与训练/部署变更交互，不是单向长程旋钮 |
| Phase19 MLA，τ=1 对照 | `results/PHASE19_TAU1_vs_GEO_REPORT.md:34-70` | τ=1 的 500M run 远差于 τ=1.414 的 500M run；但与 Geo headline 混用了 1B budget | 是异常/优化失败证据；不能推出 universal floor |
| MLA Phase22/23 | `results/PHASE22_23_MLA_TAU_SWEEP_REPORT.md` | 不同 K/base/model/budget 下曲线 jagged；Phase23 最佳测试 τ=2.5 | 单 seed、跨 protocol，不能拟合统一 rule |
| Video DiT | `results/video_dit/TAU_SWEEP_REPORT.md:14-95` | τ=.7 灾难，τ=1.5 为测试最佳但上侧未 bracket；τ=0 重复有明显波动 | 跨模态说明 τ 敏感；不提供语言模型 scaling law |

特别地，`L=128,d=64` 上 `1.25×formula≈7.07`，而 retained selection NLL 从
formula 附近的 `6.0017` 恶化到 τ=7 的 `6.2328`。这是当前最直接的“1.25 修正”
外部反例。

#### 历史 τ 叙述中的可定位错误

**[事实] Phase8D 没有验证旧公式。** `L=256` 的预测点 τ=4 后，τ=5 继续改善；
`L=512` 的预测点 τ=2.83 后，τ=3.5、4 继续改善。原报告自己在
`docs/exp/2026-02/2026-02-26_full_experiment_report.md:433` 写明两条曲线都没有 peak，却又在
`:511-536` 把这些 censored labels 放进 `C/sqrt(L)` fit。把测试上界或未 bracket 点当
作 observed optimum 会人为强化 scaling law。

**[事实] Phase19 的机制解释方向错误。**
`results/PHASE19_TAU1_vs_GEO_REPORT.md:63-68` 称 τ=1 会把频率聚到 “very low
values/near DC” 并丢失高频；canonical 公式却表明任何正 τ 都相对 τ=0 把指数下移、
把频率推高。τ=1 相对 τ=1.414 确实是**较弱的高频迁移**，但不是相对 Geo 丢掉高频。
因此那个 run 的灾难性 PPL 不能由报告给出的频率方向解释，仍可能包含训练失败或其他
protocol 因素。

**[事实] Phase22/23 汇总存在 arm 标签错误。**
`results/PHASE22_23_MLA_TAU_SWEEP_REPORT.md:17-24` 的 Phase22 并没有 τ=1.414 arm，
但 `:96-101` 把 `+16.3%/-24.1%` 标成旧架构 τ=1.414；这两个数实际来自 τ=2.2。
因此不能用该表声称同一个 τ 在 K/base 变化前后发生 pattern reversal。

**[事实] `TAU_UNIFIED_THEORY.md` 不能作为修正式依据。**

- `docs/tau_algor/TAU_UNIFIED_THEORY.md:34` 称 99-run 使用 base 10K；实际 Phase16
  manifest/runner 使用 base 500K；
- `:267-296` 的 “18 组实际 τ” 混合了测试值、选中值、不同 seed 和不同 protocol，
  其中多条曲线没有 bracket；这些不是 18 个独立真实 optima；
- `:310-314` 的 `R²>0.99` 与当前 common-metric held-out reanalysis 不一致；
- `:322-326` 把小于一个 exponent grid spacing 的连续位移称为“等价恒等”并据此构造
  `4/sqrt(K)` floor，但 RoPE 频率没有量化到 grid：任何非零连续位移都改变 phase。
  这是一条 heuristic detectability threshold，不是离散可行性约束。

本文对该旧 `max(d/sqrt(L),1.4)` 的直接复算也只把 mean pilot regret 从 `0.04501`
降到 `0.04285`，远弱于 Phase16-local `1.25×` 的 `0.01990`。这进一步否定把 floor
当作理论修正。

### 6.2 learnable scalar τ 并未学到 OOD optimum

`data/curated/learnable_tau_128tok_evidence.json:12-35` 记录：

- 125M、`L_train=128`、`B=500K`、`d_head=64`、15M tokens；
- 三 seed 的 learned τ 最终为 `1.1391/1.1445/1.1383`；
- learned τ 的 8K PPL mean `437.9`；
- fixed τ=5 的三 seed 8K PPL mean `335.7`；
- raw 日志/checkpoint 未进入 portable bundle。

**[事实]** scalar τ 可重复收敛，并不等于收敛到 extrapolation objective 的最优点。
训练只看 in-range LM loss，而该 loss 在小 τ 区域近乎平坦；固定 τ=5 的 OOD 结果更好。

**[推断]** 若未来学习 τ，outer objective 必须显式包含 deployment lengths/tasks，
或使用与 deployment 对齐的 validation hypergradient；普通训练 loss 不足以识别它。

### 6.3 base 与 scarce-channel 证据没有校准 τ

- `data/curated/text_base_10k_500k_pilot.json:22-75` 在 base 10K 与 500K 都测试同一个
  τ=2.828；它说明方向不只存在于 base 500K，但没有各自 tuning，不能证明 τ 与 B 无关。
- `data/curated/table18_mla_3seed_aggregate.json:1-10` 明确把 τ=1.414 标成
  `d_eff=128` 的经验 convention，不是不同 `d_eff` 定义的直接比较。

### 6.4 已失败的静态 selector

当前工作树报告记录：

- exact finite-K Gram selector 给出 `τ*=13.13–13.34`；
- Phase16 nearest retained arms 上 mean PPL regret `5.44%`；
- 旧公式为 `4.69%`；
- selector 仅有 `1/9` top-2，旧公式 `5/9`。

位置：
`rebuttal/rebuttal_0723/theory_results/EXPERIMENT_REPORT_20260724.md:755-774`。

同一报告还发现：

- Q/K pair norm 与 causal delta 的 median Spearman 为 `-0.167`；
- phase-covariance utility 为 `-0.045`；
- 保持频率 multiset 不变、只交换 band assignment，就有 `22/24` cells 的
  NLL 改变量至少 0.05。

位置：同文件 `:629-661`。

**[事实]** collision、phase 和 norm 都不是训练后 task risk 的可靠 scalar proxy。

**[推断]** 失败原因不是 selector 数值不够精致，而是它省略了训练后
channel-frequency binding、softmax 和 downstream task weights。

### 6.5 shape 证据反而削弱“τ 唯一正确”

当前工作树的三 seed matched-shape 结果中，matched exponential 的均值在所有报告点
都优于 Cosh，但多处置信区间仍含 0
（`EXPERIMENT_REPORT_20260724.md:82-107`）。

native endpoint、同 span、同 RMS deformation 的比较又显示：

- Cosh 在短到中等 extrapolation 很有竞争力；
- exponential 在 8K 比 Cosh 低 `0.0769 NLL`；
- attention-prior two-band shape 在 8K 比 Cosh 低 `0.211 NLL`；
- Cosh 与 attention shape 在 `φ` 空间并不接近。

位置：同文件 `:210-270`。

**[事实]** Cosh 是有效低参数 shape，但不是已识别的训练后 optimum。

### 6.6 application-only 结果不能反向校准 τ

LoRA、QuALITY、750M continuation、YaRN composition 和部分 staged runs 都只复用了
一个预先指定的 τ。例如：

- `experiments/lora_evq_v2/dryrun_validate.py:138-165` 直接检查输入 τ 是否匹配旧公式；
- `experiments/lora_evq_v2/train_stage2_retrieval.py:295` 固定 τ=1.414；
- `scripts/core_text_phases/run_quality_454m.sh:74` 固定 τ=1.41421；
- `scripts/core_text_phases/run_750m_full_eval.sh:23-24` 只比较 τ=1 与 τ=0 checkpoints。

这些结果可以检验“某个已选 allocation 是否迁移/组合”，却没有 within-protocol τ
bracket，因而不能证明该 τ 是最优，也不能用于回归新公式。把所有应用结果都当成
τ calibration points 会产生严重的 survivor/selection bias。

---

## 7. 为什么旧理论能给出尺度结构，却不能给出真实最优 τ

### 7.1 surrogate optimum 与 model optimum 不是同一个命题

在固定 surrogate 中，若目标写成

\[
\mathcal J[\rho]
=\alpha\,\mathcal I[\rho]
+\beta\,\mathcal R[\rho],
\]

则 Euler–Lagrange 解可以给出 Cosh family，并把
`τ=sqrt(β/α)` 解释成 surrogate 权重比。这只证明：

> 给定这两个泛函、边界条件和连续密度假设，Cosh 是该 surrogate 的解。

它没有证明：

- `α,β` 等于语言模型训练后 loss 的局部 Hessian；
- `β/α` 只由 `d_head,L_train` 决定；
- 连续 density 结论在 midpoint finite-K 下保持端点/span；
- 同一个 τ 适用于不同 base、数据、head/layer 或 deployment task。

仓库早期理论审计已经把 ordinary KL 的小 τ 变化定为 `O(τ^4)`，并指出 task-local
optimum 应由真实风险的一阶、二阶系数决定
（`rebuttal/pre_rebuttal/THEORY_FREQUENCY_OPTIMALITY_AND_TAU_20260716.md:347-411`）。
这与上面的 `θ=τ²` 展开一致。

### 7.2 训练后局部 optimum 的正确形式

令 `θ=τ²`，对给定训练协议和部署风险作局部展开：

\[
R_{\mathcal T}(\theta)
=R_{\mathcal T}(0)+g_{\mathcal T}\theta
+\frac12h_{\mathcal T}\theta^2+o(\theta^2).
\]

若 `h_T>0`，局部最优是

\[
\theta^*_{\mathcal T}
=\left[-\frac{g_{\mathcal T}}{h_{\mathcal T}}\right]_+.
\]

这里 `g_T,h_T` 依赖：

- 数据中的相对距离与依赖结构；
- model/head/layer 的 feature usage；
- seed 与训练动力学；
- base、K 和 endpoint/span convention；
- 部署长度及其权重；
- NLL、retrieval 或其他下游 objective。

因此不存在理由让它恒等于 `d_head²/L_train`。

### 7.3 retraining 后必须包含 implicit response

若模型参数为 `W`，频率参数为 `η`，训练得到

\[
W^*(\eta)
=\arg\min_W \mathcal L_{\rm train}(W,\eta),
\]

真实 outer risk 为 `R_val(W*(η),η)`，其 ideal implicit gradient 是

\[
\frac{dR}{d\eta}
=R_\eta
-R_W H_{\rm train}^{-1}
\mathcal L_{W\eta}.
\]

只优化 frozen-frequency 几何量相当于只保留 `R_η` 的一个代理，忽略模型重新分配
通道功能的 response term。仓库的 band-swap 与 learnable/fixed gap 正好说明该项不能
默认忽略。

---

## 8. 一手文献审计：谁真正优化了哪个变量

### 8.1 不独立优化 training-time exponent allocation 的工作

| 一手工作 | 实际变量与目标 | 为什么不是 τ 的答案 |
| --- | --- | --- |
| [RoFormer](https://arxiv.org/abs/2104.09864) | 定义 rotary operator 与固定 geometric frequencies | 没有把 `u_k` 作为独立设计变量 |
| [YaRN](https://arxiv.org/html/2309.00071) | 对 pretrained RoPE 做 inference/fine-tuning range interpolation，并拟合 attention temperature | 改的是 extension mapping 与 logits，不是从头训练时固定 B 下的 exponent shape |
| [LongRoPE](https://arxiv.org/html/2402.13753) | 以 target-length next-token loss 搜索 per-dimension rescale 与 position threshold | 优化 pretrained context extension；不是固定训练 operator/base/range 的 `u_k` |
| [LongRoPE2](https://arxiv.org/html/2502.20082) | 以 synthetic needle-driven PPL 搜索 rescaling factors | 直接证明普通 PPL objective 可能错过 retrieval，但仍是 rescaling search |
| [Base of RoPE](https://arxiv.org/abs/2405.14591) | 研究 base 与可支持 context length 的关系 | 优化/约束的是 B，不是独立 shape |
| [Resonance RoPE](https://arxiv.org/abs/2403.00071) | 把波长改到与训练长度共振以改善 OOD position recognition | 改变频率周期条件，不提供共享 scalar τ 的 training optimum |
| [FoPE](https://arxiv.org/abs/2412.17739) | 修改 Fourier/periodic positional representation | operator/representation 变化，不是固定 RoPE 下的 exponent allocation |

这些工作不能被列成 “EVQ τ 已被别人理论求解”，但它们共同说明：最优变量随目标
（in-range PPL、target PPL、needle retrieval、周期稳定性）变化。

### 8.2 与真实 action point 最接近的工作

下列 2026 工作在本次检索时均为近期 arXiv preprint；本文把它们作为最新一手方法证据，
不把尚未完成同行评审的结果升级为定论。

#### LeRoPE

[LeRoPE: Learnable RoPE Frequencies Improve Language Modeling](https://arxiv.org/html/2607.10134)
直接对每个频带学习一个 log-frequency scale。其一手结果包括：

- 52M–2.5B 的从头训练 ladder；
- 每 band 梯度由相对 offset、Q/K norm、QK angle、post-softmax weight 和 downstream
  gradient 加权的 Fourier signal 决定；
- independent run 学到的 frozen frequencies 能保留 full LeRoPE gain 的 `63.6%`；
- learned spectrum 在 seed/scale 间有重复结构；
- naive extrapolation 比 RoPE 更差，但配 YaRN 后能恢复并改善。

**[事实]** 这给出了比 Cosh surrogate 更接近真实训练目标的梯度 action point。

**[限制]** 它主要优化 in-distribution LM objective，所有 layer/head 共享一组频率；
其 extrapolation 需要额外 range method。它证明“频率值得学”，不证明 learned
in-range spectrum 是通用 OOD optimum。

#### AdaRoPE

[AdaRoPE: Not All Attention Heads Should Rotate and Scale Equally](https://arxiv.org/html/2607.19363)
学习 per-head、per-block log frequencies，并同时学习 head-specific、length-aware
attention temperature。一手 ablation 报告：

- 强制所有 heads 共享一个 learnable schedule 会退化；
- learned base 比 head-wise frequency selection 更差；
- frozen-backbone 少样本 extension 和 joint LoRA/frequency training 都有收益；
- full method 同时依赖 adaptive frequency 与 adaptive scaling。

**[推断]** 对真实模型而言，共享 scalar τ 很可能欠参数化；head heterogeneity 是比
`d_head` 单一缩放更重要的遗漏变量。

**[限制]** AdaRoPE 同时改变 frequency granularity、head sharing 与 temperature，
不能用来断言单独放开 τ 就会得到同样收益。

#### Data-induced dependency theory

[How Data Shapes RoPE Frequency Usage](https://arxiv.org/html/2607.07678)
把任务相关相对距离写成 dependency kernel。对单一 width 的受控任务，它证明最高
admissible frequency 随 dependency width 反比变化；论文同时明确指出自然语言是多尺度
依赖混合，不应预测一个单频最优。它还给出 position interpolation 有效所需的
self-similar dilation 条件。

**[推断]** 这是目前解释 τ 为什么不可能只依赖 `L_train` 的最直接理论：
sequence length 只是可观察窗口，真正控制有用频率的是数据/任务的 dependency
distribution。相同 `L_train` 的两个 corpus 或 task 可以有不同 optimum。

### 8.3 其他重要边界

[Round and Round We Go!](https://arxiv.org/abs/2410.06205) 在已训练 Gemma 中观察到
最高频形成 positional patterns，而最低频被大量用于 semantic transport。这与“通道
均匀贡献”假设冲突，并与本仓库的 channel-frequency binding 诊断方向一致。

[Bilevel Programming for Hyperparameter Optimization and Meta-Learning](https://proceedings.mlr.press/v80/franceschi18a.html)
提供了把训练动态放入 inner problem、validation objective 放入 outer problem 的一般
框架；它不是 RoPE 定理，但恰好给出 τ 作为 training hyperparameter 时应使用的数学
问题形式。

[Random Fourier Features for Kernel Ridge Regression](https://proceedings.mlr.press/v70/avron17a.html)
说明有限 Fourier feature 的最优采样依赖 kernel、data 与 regularization。这里只能作
类比：RoPE attention 不是 kernel ridge regression，不能把 ridge-leverage 结论直接
移植为 RoPE theorem。

### 8.4 文献结论

**[事实]** 截至本次检索，没有一手工作在以下全部约束下给出通用最优解：

- 固定 RoPE operator；
- 固定 B 与 sampled frequency range；
- 固定有限 K；
- 独立优化 exponent positions `u_k`；
- 目标是训练后模型在声明 deployment distribution 上的风险；
- 同时对 unseen architecture/data/seed 有预测保证。

**[推断]** 这不是文献遗漏了一个容易求的 closed form，而是一般问题本身由 task 与
trained weights 决定。若不指定这些量，“通用最优 exponent allocation”并未定义。

---

## 9. 重新定义一个有意义且可求解的最优问题

### 9.1 先把 shape 与 sampled range 分开

给定 midpoint grid `u_k`，原 Cosh 样本为 `φ_τ(u_k)`。定义 endpoint/span-anchored
Cosh：

\[
\bar\phi_{\tau,k}
=a+(b-a)
\frac{\phi_\tau(u_k)-\phi_\tau(u_0)}
{\phi_\tau(u_{K-1})-\phi_\tau(u_0)},
\]

其中 `a,b` 固定为 reference schedule 的采样首尾指数。则对所有 τ：

\[
\bar\phi_{\tau,0}=a,\qquad
\bar\phi_{\tau,K-1}=b.
\]

这使 τ 只改变 interior allocation。Cosh、exponential、power 或 monotone spline
都应先用同一 `(a,b)`，再比较 shape。

### 9.2 用真实 deformation，而不是 raw τ，定义搜索半径

最简单的无量纲 shape 坐标：

\[
\eta_\phi(\tau;K)
=
\left[
\frac1K\sum_k
\bigl(\bar\phi_{\tau,k}-\phi^{\rm ref}_k\bigr)^2
\right]^{1/2}.
\]

若需要比较不同 base，使用实际 log-frequency deformation：

\[
\eta_{B,K}(\tau)
=
\left[
\frac1K\sum_k
\bigl(\log\omega_k(\tau)-\log\omega_k(0)\bigr)^2
\right]^{1/2}
=\log B\,\eta_\phi.
\]

若部署距离分布 `q(Δ)` 已声明，更直接的是 phase-chord deformation：

\[
\eta_q^2
=\frac1K\sum_k
\mathbb E_{\Delta\sim q}
\left|e^{i\Delta\omega_k}
-e^{i\Delta\omega^{\rm ref}_k}\right|^2.
\]

这些量只能规范化 intervention strength，**不是性能代理**。它们的价值是让不同
`B,K,shape` 的 arm 在同一实际扰动尺度上比较。

### 9.3 真正的 task-conditioned robust bilevel objective

令 `s` 表示 training seed，`ℓ` 表示 deployment length/task，`η` 表示 anchored
shape 参数或完整 frequency vector：

\[
W^*_{s}(\eta)
=
\operatorname{Train}
\bigl(W_{0,s};\eta,\mathcal D_{\rm train}\bigr).
\]

定义：

\[
\min_{\eta\in\mathcal C}
\quad
\mathbb E_{s,\ell\sim P_{\rm deploy}}
\left[
\operatorname{NLL}_{\ell}
\bigl(W_s^*(\eta),\eta\bigr)
\right]
+\lambda\,\operatorname{CVaR}_q(\Delta R_{s,\ell}),
\]

subject to

\[
\mathbb E_s[
R_{\rm ID}(W_s^*(\eta),\eta)
-R_{\rm ID}^{\rm ref}
]\le\epsilon,
\]

以及固定 operator、B、K、sampled endpoints/span、parameter/compute budget 等
约束集合 `C`。

若目标是 retrieval，则把 answer-token NLL 或 autoregressive exact match 明确加入
`P_deploy`；普通平均 PPL 不能替代它。LongRoPE2 的 needle-guided search 正是这一
objective mismatch 的一手反例。

### 9.4 三个层级的可求解版本

1. **零学习参数版**：在 anchored Cosh/Exp/Power family 与离散 `η` grid 中做
   nested selection；这是当前论文框架最兼容的版本。
2. **低参数版**：每 head 一个 `η_h`，加 shrinkage
   `γΣ_h(η_h-\barη)^2`，用 held-out deployment objective 选择；检验 scalar sharing。
3. **高表达版**：直接学习 monotone ordered `log ω_{h,k}`，并用 range/endpoints、
   smoothness、ID risk 约束；这更接近 LeRoPE/AdaRoPE，但已经是后续方法，不应包装成
   当前 Cosh 定理的自然延伸。

---

## 10. 可证伪预测

### P1：raw τ 的跨配置失稳主要来自 intervention-strength 失配

**[待验]** 把 arms 改为相同 `η_{B,K}` 或 `η_q` 后，不同 `K,B` 配置的 optimum
离散 index 会比 raw multiplier 更稳定。

反证条件：matched-η 后 configuration-level rank 一样不稳定，或 raw rule 在真正
held-out domains 上更好。

### P2：span-anchoring 会改变高 τ arm 的排序

**[待验]** 在 `L=256,d=128,K=64` 这类高 τ 配置中，raw Cosh 的收益有相当部分来自
sampled range collapse；固定 endpoints/span 后，τ=8/10 的优势会缩小或改变。

反证条件：anchored Cosh 在相同 deformation 下仍稳定复现 raw Cosh 排序和效应大小。

### P3：最优频谱取决于 dependency profile，而非仅 `L_train`

**[待验]** 在相同 model、B、K、L、token budget 下，改变长程依赖分布但保持 token
边际统计近似不变，会移动最佳 `η` 或 shape。

反证条件：多个明确不同 dependency profiles 仍给出相同 held-out optimum，且不只是
宽 basin 所致。

### P4：共享 scalar τ 欠参数化

**[待验]** 在相同 frequency/range budget 与相同 parameter penalty 下，
per-head shrinkage model 会稳定优于 shared scalar，且 learned `η_h` 与 head 的
dependency-distance profile 对齐。

反证条件：跨 seed 的 `η_h` 无结构、收益消失，或 shared scalar 在 held-out objective
上不劣。

### P5：普通 in-range loss 不足以学到 OOD τ

**[事实支持、仍待前瞻验证]** 仅用 in-range gradient 会复现小 τ operating point；
显式 outer OOD objective 会选择更大/不同 η，并改善注册 deployment risk，同时可能
付出受约束的 ID cost。

反证条件：严格 matched optimizer 下两种目标收敛到同一 basin，或 OOD hypergradient
无稳定收益。

---

## 11. 最小、具有识别力的验证方案（本次未运行）

### 11.1 实验合同

1. **Reviewer/AC concern addressed**：`R27bE.1` 的 exact/small-τ/finite-τ 边界，
   以及 `R27bE.4` 对 independently tuned τ 和其他 shape 的要求。
2. **Existing evidence**：99-run 说明旧公式相对 midpoint-Geo 有方向性但不是可靠
   optimum；matched-shape、native-range 与 static-selector 结果说明 shape 有效但
   Cosh/静态 proxy 不唯一。
3. **Smallest missing evidence**：在固定 sampled range 与实际 deformation 下，
   τ/shape 是否仍有可复现收益，以及一个 calibration rule 能否预测未见配置。
4. **Smallest executable plan**：一个 calibration cell 做 shape×deformation bracket，
   冻结选择后只在两个结构性 held-out cells 上跑最小 confirmation。
5. **Stop condition**：若 anchored Cosh 在 calibration 三 seed 上不能优于同 range
   Geo，或 held-out 两 cell 的方向不一致，则停止 τ 公式研究，把 τ 降级为需调
   robustness factor；不再拟合新闭式。

### 11.2 Stage A：不训练的 geometry registry

对每个 cell 先生成并 hash：

- native/midpoint reference；
- raw Cosh；
- endpoint/span-anchored Cosh；
- endpoint/span-anchored exponential；
- 每 arm 的 `φ_k,ω_k`、sampled endpoints/span、`η_φ,η_B,η_q`。

这一步只验证 arm identity，不预测性能。

### 11.3 Stage B：一个 calibration cell

建议保留当前证据最完整的 `B=500K,L=128,d_head=64,K=32` protocol，使用完全相同
model、data/token order、token budget、optimizer 和三 seeds。

训练 arms：

1. fixed-range Geo reference；
2. raw Cosh 的三个注册 deformation levels；
3. anchored Cosh 的同三个 `η_B` levels；
4. anchored exponential 的同三个 `η_B` levels。

selection anchors 只用于选择每个 family 的 `η`；test anchors 只评估一次。报告 ID、
1K/2K/4K/8K NLL，并预先声明权重。

这 10 arms 是区分 `raw range effect / Cosh shape / alternative shape / training noise`
所需的最小完整 bracket；删掉任何一类都会失去一个识别问题。

### 11.4 Stage C：两个真正的 held-out cells

冻结 Stage B 的选择规则，不重新调：

- **held-out K/architecture cell**：固定 `L,B`，改变 rotary channel budget，同时保持
  总 model capacity 与 operator contract 尽可能 matched；
- **held-out L/base cell**：改变 `L` 或 B，但用 `η_B/η_q` 映射，不复用 raw τ。

每个 cell 只跑：

1. Geo；
2. frozen raw-Cosh rule；
3. frozen anchored-Cosh rule；
4. frozen anchored-exp rule；

三 seeds。若必须研究 K，优先在固定 `d_head` 下改变 rotary subset `d_rot`，避免再把
`K,d_head,H` 完全耦合。

### 11.5 决策门

新 operating rule 只有同时满足以下条件才能成立：

- calibration/test anchors 分离；
- 三 seed paired direction；
- 两个 held-out cells 都不重新调；
- 优于旧 formula 的 mean regret，且 worst-cell regret 不恶化；
- 对 B/K/L 的依赖来自独立变化，不是共线拟合；
- anchored Cosh 的收益不能由 endpoints/span 解释；
- 预注册 objective 同时报告 ID cost 与 deployment gain。

否则正确结论不是“再加一个修正项”，而是：

> τ 是一个 configuration/task-conditioned robustness factor，必须在声明的部署分布上
> 校准；`d_head/sqrt(L_train)` 只提供 basin initialization。

---

## 12. 对当前论文、rebuttal 和后续研究的含义

### 12.1 当前论文贡献

**保留：**

- finite spectral budget / training-time allocation 是独立设计轴；
- Cosh 是一个闭式、零学习参数、有效的 allocation family；
- 多个 matched-range 结果支持 interior allocation 会影响训练后性能。

**不能声称：**

- `τ=d_eff/sqrt(L)` 全局最优或 near-optimal；
- 99-run 验证了 `-1/2` exponent；
- τ 与 B/K/data/task 无关；
- Cosh 等于训练后 attention optimum；
- learnable τ 的失败证明固定公式更正确。

**[推断]** τ 不应承载论文最核心 novelty。更稳的贡献是“exponent allocation 是可控
设计轴”；当前 rule 只是这个轴上的默认 operating point。

### 12.2 rebuttal 表述

建议的安全表述：

> The closed-form rule is an operating prior, not an optimum theorem. In the
> staged 99-run study it improved over midpoint-Geo in 7/9 configuration means,
> but it was best among Geo, the rule, and a pilot-selected neighbor in only
> 3/9 held-out comparisons. An independent tau sweep found a bounded basin near
> the rule rather than an exact optimum. We therefore separate the analytic
> Cosh family from empirical tau calibration.

还应主动披露：

- raw τ 同时改变 finite-grid range 与 interior shape；
- Phase16 不能独立识别 `d_head,K,H,B`；
- 截图中的新 `2/12` 结果在 artifacts 齐全前不能用于 rebuttal；
- `1.25×` 是 Phase16-local observation，不是修订公式。

### 12.3 后续方法研究

优先级应为：

1. anchored shape + deformation-coordinate 的前瞻验证；
2. task-conditioned bilevel selection；
3. shared scalar 与 per-head shrinkage 的直接 ablation；
4. 最后才是自由 per-band frequency learning。

如果第 1 步失败，就不应继续为 Cosh τ 寻找解析常数；应把研究转向 target-aware
frequency learning 或更一般的 deployment-conditioned spectrum。

---

## 13. 本次复算与核验记录

临时脚本：

- `/tmp/evq_tau_operating_rule_audit_20260725.py`
- SHA-256:
  `635ddacb406c80f5fccd04982d89cbdae499d56dd6503ea255770f0cdaf48fe3`

临时输出：

- `/tmp/evq_tau_operating_rule_audit_20260725.json`
- SHA-256:
  `2b3ba731e2658a3d81c6196fc11d9828ee88177f6eb2f8098d540cac098e9402`

检查边界：

- PASS：99-row manifest count/stage count/hash；
- PASS：pilot optimum、fixed multiplier regret、configuration LOO/group holdout；
- PASS：available confirmation pairs 上 formula vs `1.25×`；
- PASS：finite-grid endpoint/span/deformation 重算；
- PASS：截图同名 Phase16 配置交叉检查；
- PASS：`python3 scripts/analysis/verify_c_coll.py`；
- SKIPPED：任何新训练、GPU、checkpoint 加载；
- UNVERIFIED：当前不可访问的 Phase16 raw logs/checkpoints/curves；
- UNVERIFIED：截图所述新 `2/12` runs；
- FAILED（环境命令，不影响分析）：系统没有 `python` 命令，改用 `python3`；
- FAILED 后修复：临时脚本初版字段名不匹配触发 `KeyError`，修正后完整运行通过。

---

## 14. 最终判断

**[事实]** 仓库没有证明旧 τ 公式最优，也没有支持一个可直接替换它的通用修正。

**[推断]** 旧公式能工作的原因更可能是：它把部分配置放进一个宽的有效变形 basin，
而不是准确恢复一个由 `d_head` 与 `L_train` 唯一决定的物理常数。

**[事实]** raw τ 把连续 Cosh concentration、finite-K sampled range、base-scaled
frequency displacement 和 deployment phase displacement混在一个数里。

**[推断]** 理论上更正确的对象是“受 range/ID/compute 约束的、训练后
deployment-risk 最小化”，而不是“寻找 τ 的普适闭式”。

**[待验]** anchored allocation + deformation-coordinate + robust bilevel selection
可能比 EVQ 当前 operating rule 更稳定；只有前瞻 held-out 实验能决定它是否真正更好。

---

## 15. 核心主张逐项核验矩阵

| ID | 待核验主张 | 结论 | 最强支持 / 反证 | 可安全表述 |
|---|---|---|---|---|
| C01 | 当前代码与实验一致实现 `τ=d_eff/sqrt(L_train)` | **部分通过** | canonical schedule 与 Phase16 runner 一致；但不同实验对 `d_eff`、midpoint grid、fixed τ 与 tuned τ 的使用并不完全统一 | 这是主要 operating default，不是所有实验共享的物理定律 |
| C02 | 截图中 `d_head=128,L=256` 的 `τ=6` 优于 `τ=8` | **当前未验证；同名 Phase16 证据反向** | Phase16 seed-42 同 cell 中 `τ=8` 在 2×/4×/8× 均优于 `τ=6`，且 `τ=10` 又优于 `τ=8`；截图所述新 protocol 的 2/12 artifacts 未找到 | 只能作为待导入的新实验假设，不能写入 rebuttal 事实 |
| C03 | 99-run 足以推出依赖 `d_head,K,L,B` 的修正规则 | **不通过** | B 固定，`K=d_head/2`，且 `d_head` 与 head count 共线；只有三个 L，配置也非独立样本 | 99-run 只能检验该九格设计内的局部排序 |
| C04 | τ 的理论作用是一个纯 allocation-shape 参数 | **不通过** | raw τ 同时改变连续密度、有限 K 的 sampled endpoints/span、以 `ln B` 缩放的 log-frequency displacement、以及依赖部署距离的 phase displacement | τ 是 intervention knob；须先固定 range 并报告实际 deformation |
| C05 | surrogate、collision/phase 指标、learnable τ 与训练后目标等价 | **不通过** | surrogate 没有 learned coefficients/data/optimizer response；learnable τ 的 in-range objective 也未复现 OOD optimum | 这些量只能作机制 proxy 或初始化，不能替代 retrained deployment risk |
| C06 | 存在已被数据支持、优于旧公式的通用修正规则 | **不通过** | `1.25×` 在 Phase16 内最强，但 leave-L1024-out 会选择 `1.0×`；回顾性 power law 更差，并有独立 L128 反例 | `1.25×` 是 Phase16-local candidate，不是新公式 |
| C07 | 一手文献已经独立求解固定 operator/base/range 下的指数分配最优 | **未发现** | 相邻工作主要优化 base、range rescaling、operator、per-dimension pretrained extension 或 learnable/head-conditioned frequencies | 文献支持“目标应 task/head/data-conditioned”，不支持一个通用 scalar τ |
| C08 | 有意义且可求解的替代问题存在 | **通过（定义层面），性能待验** | anchored exponent family 可隔离 shape；deformation coordinate 可统一 intervention strength；bilevel risk 明确包含 retraining response | 可前瞻求解受 ID/range/robustness 约束的 deployment-risk optimum |

核验判据是：代码与原始/curated artifact 优先于报告文字；同 protocol 的 held-out
结果优先于 pilot；独立变化优先于共线回归；训练后 task loss 优先于几何 proxy。

---

## 16. 检索方法、覆盖边界与披露

- 检索截止：2026-07-25。
- 核心来源：13 篇一手论文或正式会议页面；核心方法主张均回到 arXiv/PMLR
  原文核对，不依赖博客或二手综述。
- 检索概念不局限于 RoPE 名称，还覆盖 learnable Fourier frequencies、random
  features、frequency utilization、head-conditioned rotation、bilevel
  hyperparameter optimization、context-extension search objective。
- 纳入标准：工作必须实际改变或优化 operator、base、frequency/rescaling vector、
  exponent allocation、head-wise frequency use 或其训练/部署 objective；仅标题相似或
  related-work 转述不计作证据。
- 局限：2026 年三篇最相关工作均为近期预印本；它们提供重要反例和方法方向，但不等同于
  已稳定复现的共识。本审计也不声称检索穷尽所有 positional encoding 文献。
- AI 披露：本报告由 Codex 辅助完成代码审计、数值复算、文献检索和综合。所有拟用于
  正式 rebuttal 的新增结论仍需作者对原始 artifact 与引文逐项复核。
