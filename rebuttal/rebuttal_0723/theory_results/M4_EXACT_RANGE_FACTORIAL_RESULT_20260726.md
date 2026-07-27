# M4 exact-range pure-shape factorial：完整结果与结论

日期：2026-07-26

证据等级：**POST_SUB_RAW_HASH_BACKED；supporting / mechanistic，50.9M，三 seeds**

论文边界：不修改主表，不升级为大模型或下游能力证据

## 技术摘要

本实验回答一个此前始终混杂的问题：

> EVQ-Cosh 的历史收益究竟来自有限 \(K\) 下隐式改变了频率端点 /
> span，还是在频率 range 完全不变时，内部通道 allocation shape
> 本身仍然有作用？

实验严格固定每个 schedule 的最高频率、最低频率和 log-frequency
span，只允许中间 \(K-2\) 个频率的位置发生变化。主实验完成
`180/180` 次训练，额外完成 `12/12` 个边界 \(\tau\) 臂；全部
`192` 个结果状态为 `completed`，没有失败训练。

结论可以压缩为四点：

1. **allocation shape 的作用在 exact-range 控制下仍然存在。**
   公式 Cosh 相对 Geo 的平均加权 OOD NLL 为 `-0.009879`；
   预注册的 `1.25×` Cosh 和 deformation-matched exponential
   分别为 `-0.012100` 和 `-0.010619`。因此历史 EVQ 收益不能全部
   归结为隐式 base/range 修改。
2. **旧公式是一个不稳定的 basin prior，不是 pure-shape optimum。**
   在 12 个结构配置中，`1.0×` 公式点只赢 4 个，`1.25×` 赢 6 个，
   `0.75×` 赢 2 个；公式点相对每个配置最佳预注册 Cosh 点的平均
   regret 为 `0.011440 NLL`。
3. **Cosh 不是唯一有效 shape。** 公式 Cosh 相对 matched
   exponential 的平均差仅 `+0.000740 NLL`，95% configuration
   bootstrap 区间跨零，精确 sign-flip `p=0.836`。这支持
   “allocation shape 是独立设计轴”，但不支持“Cosh 在所有经验目标
   上唯一最优”。
4. **边界臂显示旧规则给出的 \(\tau\) 范围过宽。** 当公式
   \(\tau=1\) 时，`1.5×` 更好；当公式 \(\tau=8\) 时，`0.75×`
   更好。最优点均向公式范围内部移动，但还不足以推出一个新的静态
   闭式规则。

最诚实、也最有利于 rebuttal 的表述是：

> exact-range 对照确认了 finite-channel interior allocation 的独立
> 因果作用；Cosh 是一个闭式、无学习参数的实例，但旧
> \(d_{\rm head}/\sqrt{L_{\rm train}}\) 规则只能作为 operating
> default，不能声称为普遍最优点。

## 1. 完成状态

| 部分 | 计划 | 完成 | 状态 |
|---|---:|---:|---|
| 主 factorial | 180 | 180 | 完成 |
| 边界 \(\tau\) follow-up | 12 | 12 | 完成 |
| 总训练 | 192 | 192 | 完成，0 failed |
| final-checkpoint fixed-range PPL | 180 主臂 + 12 边界臂 | 完成 | 本报告主体 |
| runtime target-range / cross-swap | 已预注册 | 0 | 未执行；权重已在证据固化后清理 |
| milestone 25/50/75% dynamics | 已预注册 | 0 | replay gate 停止；权重已清理 |

主 factorial 从 `2026-07-24 20:31` 运行到 `2026-07-26 11:23`，
累计训练时间约 `38.87 h`。12 个边界臂从 `2026-07-26 11:42`
运行到 `14:25`，累计约 `2.99 h`。

## 2. 核心结果：range 固定后，内部 allocation 仍然改变结果

### 2.1 头条指标

头条指标为每个 checkpoint 在 \(2L/4L/8L\) 上的加权
natural-text NLL：

\[
\bar N_{\mathrm{OOD}}
=
\frac{\sum_{r\in\{2,4,8\}}\log_2(r+1)\,N_{rL}}
{\sum_{r\in\{2,4,8\}}\log_2(r+1)}.
\]

先在相同结构配置和 seed 内配对，再在 seed 内求均值，最后对 12 个
结构配置等权汇总。下表中的“等效 PPL”是
\(\exp(\operatorname{mean NLL})\)，不是对原始 PPL 做算术平均。

| 训练 schedule | 平均加权 OOD NLL | 等效 PPL | 相对 Geo NLL | 优于 Geo 的结构配置 |
|---|---:|---:|---:|---:|
| Native Geo | 6.211973 | 498.684 | 0 | — |
| Anchored Cosh `0.75×` | 6.202858 | 494.159 | **-0.009115** | 8/12 |
| Anchored Cosh `1.00×` rule | 6.202094 | 493.782 | **-0.009879** | 7/12 |
| Anchored Cosh `1.25×` | **6.199873** | **492.686** | **-0.012100** | 10/12 |
| Anchored exponential, rule-RMS matched | 6.201354 | 493.417 | **-0.010619** | 9/12 |

这张表最重要的不是哪一行最低，而是所有非 uniform 臂与 Geo 使用
完全相同的采样端点和 span。只移动内部频率，训练后 NLL 仍然发生了
系统变化，因此：

\[
\boxed{
\text{fixed range}
+
\text{different interior allocation}
\Longrightarrow
\text{different trained-model behavior}
}
\]

### 2.2 不确定性与配对检验

统计单位是 12 个结构配置。每个配置先平均三个 paired seeds，再做
10,000 次 configuration-level bootstrap；精确 sign-flip 检验枚举
全部 \(2^{12}\) 种符号翻转。负值表示左侧 schedule 更好。

| 对比 | 平均 NLL 差 | 95% config-bootstrap CI | 负向配置 | 双侧 exact sign-flip |
|---|---:|---:|---:|---:|
| Cosh `0.75×` − Geo | -0.009115 | [-0.017780, -0.001006] | 8/12 | 0.0815 |
| Cosh rule − Geo | -0.009879 | [-0.021040, 0.001540] | 7/12 | 0.1250 |
| Cosh `1.25×` − Geo | **-0.012100** | **[-0.020828, -0.002945]** | **10/12** | **0.0273** |
| Exponential − Geo | -0.010619 | [-0.020808, -0.000696] | 9/12 | 0.0708 |
| Cosh rule − Exponential | +0.000740 | [-0.005484, 0.007487] | 7/12 | 0.8364 |

解释边界：

- `1.25×` Cosh 是结果出现前就注册的 arm，不是事后新扫出的点；
  但表中同时比较了多个 arm，因此 `p=0.0273` 是未做 family-wise
  multiplicity 校正的辅助证据，不能单独包装成新的 primary claim。
- 公式 Cosh 的均值方向为正面，但 bootstrap CI 跨零且 sign-flip
  不显著；因此不能说旧公式在所有配置上稳定优于 Geo。
- Cosh 与 exponential 没有可分辨差异。这不是 allocation 轴的
  反证，而是对 “Cosh 唯一性” 的负边界。

## 3. 改善随评测距离存在，但不是每个距离都稳定显著

下表仍以 12 个 seed-averaged 结构配置为统计单位。每个长度使用四个
冻结 natural-text offsets。

| 长度比 | Cosh rule − Geo | 95% CI | Exponential − Geo | 95% CI | Cosh rule − Exp |
|---:|---:|---:|---:|---:|---:|
| \(1\times\) | -0.010480 | [-0.021905, 0.000749] | **-0.015686** | **[-0.026315, -0.005773]** | +0.005206 |
| \(2\times\) | -0.007484 | [-0.020118, 0.004659] | **-0.010831** | **[-0.022382, -0.000949]** | +0.003347 |
| \(4\times\) | -0.007961 | [-0.017492, 0.001231] | **-0.009717** | **[-0.019210, -0.000844]** | +0.001756 |
| \(8\times\) | -0.012482 | [-0.024630, 0.000445] | -0.011174 | [-0.021849, 0.000081] | -0.001308 |

公式 Cosh 的最大平均改善出现在 \(8\times\)，但区间略微跨零；
exponential 在 \(1\times/2\times/4\times\) 更稳定。由此不能把
Cosh 解释成只在某一个长度工作的简单 range trick，也不能声称它
在每个距离上都比 matched analytic shape 更强。

## 4. 旧 \(\tau\) 公式找到一个 basin，但没有找到稳定最优点

### 4.1 12 个结构配置的完整配对结果

每行先对三个 seeds 求均值。最后一列只在三个预注册 Cosh 点
`0.75×/1.00×/1.25×` 中选择最低者。

| Base | \(L_{\rm train}\) | \(d_{\rm head}\) | Rule Cosh − Geo | Exp − Geo | Rule Cosh − Exp | 最佳 Cosh multiplier |
|---:|---:|---:|---:|---:|---:|---:|
| 500K | 256 | 32 | -0.027831 | -0.023435 | -0.004396 | 1.25 |
| 500K | 256 | 64 | -0.015597 | -0.018764 | +0.003166 | 1.00 |
| 500K | 256 | 128 | +0.005171 | +0.010029 | -0.004858 | 0.75 |
| 500K | 1024 | 32 | -0.013360 | -0.011036 | -0.002324 | 1.25 |
| 500K | 1024 | 64 | +0.003574 | -0.009173 | +0.012747 | 1.25 |
| 500K | 1024 | 128 | -0.038789 | -0.032389 | -0.006399 | 1.00 |
| 1M | 256 | 32 | +0.001338 | -0.003480 | +0.004818 | 1.25 |
| 1M | 256 | 64 | -0.021688 | -0.014854 | -0.006834 | 1.00 |
| 1M | 256 | 128 | +0.032014 | +0.003868 | +0.028146 | 0.75 |
| 1M | 1024 | 32 | +0.001374 | +0.021887 | -0.020513 | 1.25 |
| 1M | 1024 | 64 | -0.004756 | -0.003578 | -0.001178 | 1.25 |
| 1M | 1024 | 128 | -0.040001 | -0.046503 | +0.006502 | 1.00 |

最佳 Cosh multiplier 的计数为：

- `0.75×`：2/12；
- `1.00×` rule：4/12；
- `1.25×`：6/12。

公式点相对每个配置最佳预注册 Cosh 点的平均 regret 是
`0.011440 NLL`，95% configuration-bootstrap 区间为
`[0.007119, 0.016755]`。因此不能继续把公式描述为 near-optimal
的普遍定理；合理身份是 **无需搜索的 operating default / basin
selector**。

### 4.2 两个边界配置进一步否定了“公式点就是 optimum”

额外 12 个训练只覆盖公式 \(\tau\) 的最小和最大边界，每个新增
`0.5×/1.5×` 两臂、三个 seeds。表中是 seed-mean 加权 OOD NLL。

| 配置 | 公式 \(\tau\) | `0.5×` | `0.75×` | `1.0×` | `1.25×` | `1.5×` | 最优 |
|---|---:|---:|---:|---:|---:|---:|---:|
| \(L=1024,d_h=32,B=500K\) | 1 | 6.306895 | 6.279150 | 6.277147 | 6.276875 | **6.261556** | 1.5× |
| \(L=256,d_h=128,B=500K\) | 8 | 6.139793 | **6.135611** | 6.151910 | 6.152388 | 6.154125 | 0.75× |

当公式给出很小的 \(\tau=1\) 时，更大的 \(\tau=1.5\) 更好；当公式
给出很大的 \(\tau=8\) 时，更小的 \(\tau=6\) 更好。两个方向都把
有效点向公式范围内部推。这支持“旧缩放结构过宽”的诊断，但两个点
不足以拟合新的经验公式；当前不应再用一个新的回归式掩盖缺口。

## 5. Cosh 与非 Cosh schedule 的关系

matched exponential 与 rule-Cosh 具有：

- 相同的最高和最低采样频率；
- 相同的 log-frequency span；
- 相同的相对 uniform-grid RMS deformation；
- 不同的 interior analytic shape。

其平均结果比 Geo 好 `0.010619 NLL`，与 rule-Cosh 的差只有
`0.000740 NLL`，且区间明显跨零。由此得到的准确贡献不是：

> Cosh 是所有任务上的唯一最优 allocation。

而是：

> 在有限 rotary channels 下，内部频率 allocation 是 base/range
> 之外可独立控制的设计轴；Cosh 提供闭式、参数免费、理论驱动的
> 一个实例，其他 matched analytic shapes 也可以竞争。

这正面回应了 reviewer 对 non-Cosh schedule 的要求，同时不牺牲
论文的核心 identity。

## 6. 结构切片：效应依赖 base、训练长度和通道结构

这些切片每组只有 4 或 6 个结构配置，只作为诊断，不做独立显著性
声明。

| 切片 | 配置数 | Rule Cosh − Geo | Exp − Geo | Rule Cosh − Exp |
|---|---:|---:|---:|---:|
| Base 500K | 6 | -0.014472 | -0.014128 | -0.000344 |
| Base 1M | 6 | -0.005287 | -0.007110 | +0.001824 |
| \(L_{\rm train}=256\) | 6 | -0.004432 | -0.007773 | +0.003340 |
| \(L_{\rm train}=1024\) | 6 | -0.015326 | -0.013465 | -0.001861 |
| \(d_{\rm head}=32\) | 4 | -0.009620 | -0.004016 | -0.005604 |
| \(d_{\rm head}=64\) | 4 | -0.009617 | -0.011592 | +0.001975 |
| \(d_{\rm head}=128\) | 4 | -0.010401 | -0.016249 | +0.005848 |

平均方向在两个 base、两个训练长度和三个 head dimension 上都不是由
单一切片完全驱动，但幅度明显异质。特别是 rule-Cosh 在
`B=1M, L=256, d_head=128` 上退化 `+0.032014 NLL`，说明静态
\(\tau=d_h/\sqrt L\) 没有充分吸收 base 与 finite-\(K\) 的共同作用。

## 7. 实验设计与匹配控制

### 7.1 模型与数据

- 模型：50.9M decoder-only Transformer；
- 层数 / hidden / MLP：`6 / 512 / 2048`；
- vocabulary：`50,304`；
- head 数：`4/8/16`，对应
  \(d_{\rm head}=128/64/32\)；
- 训练长度：`256/1024`；
- nominal RoPE base：`500K/1M`；
- seeds：`42/137/256`；
- 每次训练：`8,388,608` tokens，`128` optimizer steps；
- effective batch：每 step `65,536` tokens；
- 数据：本地 WikiText-2 raw train stream，确定性截取 / 重复到注册
  token 数；
- 训练 mix：所有臂相同的 1% deterministic supervised passkey
  samples；
- 设备：Apple M4 Max MPS，float32。

### 7.2 Exact-range schedule 定义

令 \(K=d_{\rm head}/2\)，native normalized grid 为

\[
u_k=\frac{k}{K-1},\qquad k=0,\ldots,K-1.
\]

native sampled log-span 为

\[
R=\frac{K-1}{K}\log B.
\]

对 Cosh，先在论文实际 midpoint grid 上计算

\[
\phi_\tau\!\left(\frac{k+\tfrac12}{K}\right),
\]

再做端点归一化：

\[
s_k^{\rm Cosh}
=
\frac{\phi_{\tau,k}-\phi_{\tau,0}}
{\phi_{\tau,K-1}-\phi_{\tau,0}}.
\]

所有 schedule 最终使用

\[
\omega_k=\exp(-R s_k).
\]

因此每个 arm 都严格满足：

\[
\omega_0^{\rm arm}=\omega_0^{\rm Geo},\qquad
\omega_{K-1}^{\rm arm}=\omega_{K-1}^{\rm Geo},
\]

且 log-span 完全一致。频率 audit 覆盖全部 12 个结构配置，所有 tensor
严格单调；exponential 的 RMS deformation matching 最大误差为
`5.56e-17`。

### 7.3 训练匹配

- 每个 arm 在模型构造前显式重置相同 seed；
- 相同结构和 seed 的 trainable initialization 相同；
- batch index 只由 seed 与 global micro-step 决定；
- passkey / LM row 的选择和内容只由 row index 决定；
- optimizer、LR schedule、token budget、effective batch 和评测 offsets
  完全一致；
- 唯一注册变量是 immutable training-time `inv_freq`。

### 7.4 评测

- natural-text validation NLL / PPL；
- 每个长度四个确定性 offsets；
- 长度为 \(L,2L,4L,8L\)；
- 主指标只使用 \(2L/4L/8L\)；
- PPL 是完整 teacher-forced next-token NLL 的指数；
- 没有 synthetic fallback。

## 8. 稳健性、限制与尚未完成的分析

### 8.1 当前结果能证明什么

当前结果支持：

- fixed range 下，interior allocation 能改变训练后 NLL；
- 该效应跨两个 base、两个训练长度、三个 head dimension 和三个 seeds
  出现；
- non-Cosh analytic schedule 也有竞争力；
- 旧公式不是稳定最优点。

当前结果不能证明：

- Cosh 普遍优于 Geo、FMRoPE 或 YaRN；
- 旧 \(\tau\) 公式是理论最优；
- 该效应已经迁移到 1B/8B 或真实下游 QA；
- PPL 改善必然转化为 retrieval / generation 能力；
- allocation 与 target-aware range scaling 存在可加 synergy。

### 8.2 数据与规模限制

这是 50.9M、短训练预算、WikiText-2 的 supporting/mechanistic
实验。训练文本为本地语料确定性重复，不代表大规模预训练分布。
1% passkey mix 在所有臂完全匹配，但本实验没有把 passkey 能力指标
作为头条结果。因此报告只能用于机制识别与 reviewer control，不能
替代较大模型和真实 benchmark。

### 8.3 MPS replay gate

预注册 follow-up 希望通过 deterministic replay 补回早期未保留的
25/50/75% checkpoints，并要求 replay 最终权重 hash 与原始权重
bitwise 相同。第一个 replay：

`B500K_L256_H4_Dh128_native_geo_seed42`

没有通过 bitwise hash gate，因此流程按 fail-closed 停止。诊断显示
差异很小：

- relative parameter \(L_2\)：`4.97e-5`；
- maximum absolute parameter difference：`2.52e-4`；
- 原始 / replay PPL：
  `355.7972/355.8038`,
  `404.9163/404.9241`,
  `513.0040/513.0117`,
  `460.5651/460.5692`。

这更符合 MPS kernel 的非 bitwise deterministic 漂移，而不是训练
协议接错；它不否定清理前已经得到的 fixed-range 指标，但意味着：

- 不能把 replay 产生的里程碑 checkpoint 当成与原件完全相同；
- 25/50/75% dynamics 暂不报告；
- final-checkpoint target-range、runtime-shape cross-swap 未执行；
- 自动重试服务已停止，避免继续无效循环。

在 reviewer-grade JSON、per-run `spec.json` / `result.json` 和本报告完成
固化后，用户于 2026-07-27 明确授权清理本实验权重。因此未完成的
cross-swap 和 milestone dynamics 现在不能从原 checkpoint 补跑；这不
改变已经保存的 fixed-range 数值，但缩小了后续可追溯范围。

## 9. 对 rebuttal 的直接意义

### 9.1 可以使用的结论

1. **与 FMRoPE 的技术区别获得了受控实验支持。** range/support
   完全相同时，仅改变内部 allocation 仍改变训练后表现。
2. **论文贡献应收窄到独立设计轴，而不是 Cosh 全局最优。**
3. **旧 \(\tau\) 公式应继续标注为 operating default / basin
   selector。**
4. **matched exponential 是诚实且有利的 non-Cosh 结果。** 它说明
   现象不是某个 Cosh 参数化独占，而是 finite allocation 的一般问题。

### 9.2 不应使用的结论

- 不说 “Cosh 在 12/12 配置都更好”；
- 不说 “公式点 near-optimal across all settings”；
- 不说 “Cosh 显著优于 exponential”；
- 不把 50.9M 三-seed机制实验升级成大模型 SOTA；
- 不把尚未完成的 target-range / cross-swap 写成已获得结果。

### 9.3 可直接放入英文 rebuttal 的精简表述

> To isolate allocation from scalar range, we completed a preregistered
> three-seed factorial over two bases, two training lengths, and three head
> dimensions. All schedules had exactly identical sampled frequency extrema
> and log-span; only the interior finite-\(K\) spacing differed. The
> formula-Cosh arm reduced weighted OOD NLL by 0.0099 on average, while the
> preregistered \(1.25\times\) Cosh and deformation-matched exponential arms
> improved by 0.0121 and 0.0106, respectively. This isolates interior
> allocation as a genuine design variable rather than a scalar base/range
> rewrite. The best Cosh multiplier varied across configurations and Cosh was
> statistically indistinguishable from the matched exponential control, so we
> treat \(d_{\rm head}/\sqrt{L_{\rm train}}\) as an operating default and do
> not claim universal Cosh optimality.

## 10. 下一步

按收益 / 成本排序：

1. **不补跑 cross-swap 或 milestone dynamics。** 原权重已按授权
   清理；只有 reviewer 明确要求该归因时，才重训一个预注册的最小
   子集，不能把新权重伪装成原 checkpoint。
2. **不要继续扩大静态 \(\tau\) sweep。** 当前证据已经足够判断旧公式
   不是稳定 optimum；下一步应研究依赖 \(B,K,L_{\rm train}\) 和目标
   距离分布的 training-free calibration，而不是拟合新的经验回归式。
3. **将较大模型实验用于 scale transfer。** 不再让一个大模型实验
   同时承担 allocation、range、\(\tau\) 与 downstream conversion
   四个问题。

## 11. 结果来源

- Reviewer-grade curated evidence：
  `rebuttal/rebuttal_0723/theory_results/m4_exact_range_factorial_evidence_20260726.json`
  （192 个 per-run spec/PPL、源 JSON hashes、完整 frequency receipts、
  configuration-level statistics；SHA256
  `858710ff52488a0b1d1e0a9a35af9f7658ed974d1b59ff59cc9c89fa2ce9abda`）
- Raw per-run `spec.json` / `result.json`（final checkpoints 已按授权清理）：
  `results/theory/phase16_exact_range_factorial_m4_20260724/runs/`
- 主实验自动汇总：
  `results/theory/phase16_exact_range_factorial_m4_20260724/reports/summary.json`
- 频率 audit：
  `results/theory/phase16_exact_range_factorial_m4_20260724/analysis/frequency_audit.json`
- 边界臂计划：
  `results/theory/phase16_exact_range_factorial_m4_20260724/analysis/extreme_plan.json`
- 主训练 runner：
  `scripts/core_text_phases/phase16_exact_range_factorial_m4.py`
- inference follow-up runner：
  `scripts/core_text_phases/phase16_exact_range_followup_m4.py`
- 预注册：
  `rebuttal/rebuttal_0723/theory_results/M4_EXACT_RANGE_FACTORIAL_PREREG_20260725.md`

## 12. 权重清理回执

2026-07-27，在确认 curated evidence 可解析、包含 `192/192` 个结果且
SHA256 与上文一致后，清理了本实验、对应 milestone/smoke，以及上次
Phase16 99-run/smoke 的模型权重：

- 删除权重文件：`583`；
- 删除逻辑字节：`356,326,225,622`；
- APFS 可用空间：`185 GiB` → `517 GiB`，实际增加约 `332 GiB`；
- 本实验正式 runs 保留：`192 result.json + 192 spec.json`；
- 上次 99-run 保留：`99 result.json + 99 spec.json`；
- 保留全部报告、日志、频率 audit、curated evidence 和源代码；
- 未触碰其他实验目录或数据缓存。

清理范围仅限：

- `results/theory/phase16_exact_range_factorial_m4_20260724/{runs,milestone_replays}/`
- `results/theory/phase16_exact_range_factorial_m4_20260724_smoke/runs/`
- `results/theory/phase16_formula_optimality_sweep_local_m4_wikitext/runs/`
- `results/theory/phase16_formula_optimality_sweep_smoke/runs/`
