# EVQ：8K 内学习到 8K 外能力的机制重审与实验计划

## Material Passport

- Origin Skill: `academic-research-suite / experiment-agent`
- Origin Mode: `plan + theory audit`
- Origin Date: `2026-07-13`
- Verification Status: `UNVERIFIED`（实验尚未运行；文中的 fresh 数值仅为 CPU、model-free 频率诊断）
- Version Label: `evq_8k_only_capability_transfer_v1`
- Status: `internal rebuttal plan / no new model result / do not quote as evidence`
- Scope: LLaMA-3-8B-Instruct，训练物理长度和 position id 均不超过 8192，评测到 32K

> 本文件是当前最高优先级的能力转化计划。它不修改论文数值，也不把尚未运行的实验写成结果。
> 它取代“先做 16K continued training”作为第一科学检验，但不删除已有实现。

---

## 0. 结论先行

当前 8B 结果不是弱信号。300 步、8K LongAlpaca、q/k/v/o LoRA 后，EVQ 相对匹配训练流水线的 Geo+LoRA 在 8K 付出 `+0.390` NLL，却在 16K/32K 获得 `-1.510/-2.048` NLL；长程方向覆盖 24/24 packs 和 3/3 domains。这个结果已经说明：

1. 成熟 LLaMA-3-8B 并非无法适应 EVQ；直接替换后，短 LoRA 已足以得到稳定且显著改善的超窗 NLL，而不是只留下灾难性相位扰动。
2. EVQ 产生的主要变化发生在训练窗口之外，不能只解释成训练集拟合。
3. 但它目前证明的是 **long-position language-model stability**，不是 source-dependent retrieval，更不是完整下游能力。

作者当天观测到 NIAH 没有随 NLL 一起改善；该观测的原始 artifact 尚未在仓库完成审计，因此在本文件中记为 `author-reported / evidence pending`。RULER 当前是工程链路未跑通，不是科学负结果。

我现在认为最值得检验的机制不是“EVQ 直接优化了长文本任务”，而是：

> **EVQ 把一部分在 8K 内几乎不发生相位变化的 dormant rotary modes 移入可观测频带，使更多相位方向在 8K 内可被梯度区分和潜在校准，并降低这些通道在 16–32K 第一次发生大相位变化的程度。它因此可能改善平均语言建模稳定性；但只有训练数据提供明确的 source-dependent dependency gradient 时，这种相位稳定性才有机会转化为检索、组合和真实 QA 能力。**

这是一个可证伪机制假设，不是现有变分目标的推论。

本计划的主问题是：

> 在训练物理序列和所有 position id 都不超过 8191 的条件下，使用完全相同的 source-dependent 任务训练，EVQ 是否比 native Geo 和 same-grid midpoint Geo 更能把 8K 内学到的信息路由电路迁移到 16K/32K？

若答案为是，EVQ 的价值就不是“替代长序列训练”，而是 **提高有限训练窗口对频率基底的覆盖，使在窗口内学到的内容路由规则更容易跨位置尺度继续工作**。若答案为否，则当前 NLL 结果应被解释为 substrate stability，而不能升级为长上下文能力。

---

## 1. 证据总账：已经知道什么，不知道什么

| 事实或判断 | 状态 | 来源/边界 |
| --- | --- | --- |
| 8B EVQ+LoRA 在 16K/32K temporal NLL 上显著优于 Geo+LoRA | 仓库证据，单 seed supporting | `LORA_LONGALPACA_TEMPORAL_NLL_20260712.md`；native endpoint Geo 与 midpoint EVQ，不是纯 shape control |
| 8K 内代价小于 16K/32K 的 NLL 收益，且长程方向为 24/24 packs | 仓库证据 | 同上；应报告 NLL，不能用 PPL 百分比夸大 8K trade-off |
| temporal 32K packs 由较短文档拼接 | 仓库证据 | 测到的是长绝对位置上的 token NLL；多数预测不必读取几十 K 之外的唯一证据，因此不能等同于 long-dependency capability |
| 原训练 loss 监督所有 non-padding tokens | 仓库代码证据 | `legacy_lora_protocol.py` 注册为 `all_non_padding_input_tokens`；绝大多数梯度是普通局部/中程 LM token，而非远端 source-dependent answer |
| NIAH 没有转化 | 作者当天观测，artifact 待审计 | 不能在 rebuttal 中先写成正式负结果；先排除 chat template、抽取、EOS 和 evaluator 问题 |
| RULER 失败 | 工程状态 | 不能解释为模型能力失败 |
| 151.9M local diagnostic 出现 long NLL 改善、retrieval 反向 | local supporting diagnostic | 说明 NLL 与 retrieval 可解耦；不能外推为 8B 定律 |
| cosh 是 stated convex surrogate 的唯一 minimizer | 精确定理 | 只证明 surrogate 下 allocation shape，不证明 LM/task objective |
| `tau=d_eff/sqrt(L)` 是 trained-task optimum | 不成立 | ordinary KL 推导错误；只能保留为 proxy-motivated、empirically calibrated operating point |
| EVQ 的 8K phase coverage 是否使 LoRA 更易校准超窗行为 | 新条件性假设 | 本文件给出精确代数、model-free sanity check 与失败条件；尚无 task-weighted Jacobian 证据 |

### 1.1 对已有两条“检索修复”路线的裁决

`experiments/lora_evq_v2/train_stage2_retrieval.py` 不能作为当前主实验：

- 只有 EVQ continuation，没有 matched Geo control；
- 所有 token 都进入 loss，远端答案梯度仍被大量 filler/local-LM 梯度稀释；
- 旧 generator 的 filler、模板、split 和 token-distance contract 不足以证明跨任务能力；
- 即使 32K 提升，也无法分离普通检索 SFT 与 EVQ 的独立贡献。

`rebuttal/pre_rebuttal/evq_seed42_retrieval_repair/` 修复了 chat template、answer-only loss、counterfactual triplet 与 provenance，工程质量更高，但它也不是当前核心检验：

- 它只有 EVQ arm，没有相同数据/预算下的 Geo control；
- R16 明确使用 16K physical training 和 factor-2 YaRN-derived operator；
- 因而它回答“能否用长训练修复 EVQ”，不回答“8K 内是否能学到 8K 外能力”。

结论：复用它的 exact-token construction、answer masking、triplet evaluator 和 artifact contract；暂停 R16 作为第一实验，不沿用 EVQ-only 的因果解释。

---

## 2. 为什么 8K LoRA 可能改善 32K NLL：底层机制假设

### 2.1 RoPE 的精确代数

对第 \(k\) 个二维 rotary plane，RoPE 满足

\[
R_{\omega_k}(m)^\top R_{\omega_k}(n)
=R_{\omega_k}(n-m).
\]

令相对距离 \(\Delta=n-m\)。对固定 token pair，其该平面的 attention logit 可精确写成

\[
q_{m,k}^{\top}R_{\omega_k}(\Delta)k_{n,k}
=a_{mnk}\cos(\omega_k\Delta)
+b_{mnk}\sin(\omega_k\Delta),
\]

其中

\[
a_{mnk}=q_{1}k_{1}+q_{2}k_{2},
\qquad
b_{mnk}=q_{2}k_{1}-q_{1}k_{2}.
\]

因此频率表提供了一组 phase features：

\[
\psi_{\Omega}(\Delta)
=\left[\cos(\omega_1\Delta),\sin(\omega_1\Delta),\ldots,
\cos(\omega_K\Delta),\sin(\omega_K\Delta)\right].
\]

这部分是精确恒等式。需要立刻强调：\(a_{mnk},b_{mnk}\) 会随内容、层、head 和 token pair 改变，所以完整 Transformer 不是一个共享系数的 Fourier regression。

### 2.2 dormant mode 与 phase coverage

若 \(\omega_kL_{\mathrm{train}}\ll1\)，则该通道在训练窗口内近似

\[
\cos(\omega_k\Delta)\approx1,
\qquad
\sin(\omega_k\Delta)\approx\omega_k\Delta.
\]

多个极低频通道在 8K 内都近似常数/线性，彼此高度共线。训练只能看到它们很小的一段相位弧；到 16K/32K 后，一部分通道才首次发生明显旋转。此时预训练 Q/K projection 从未在训练分布中校准过这些相位状态，attention logits 可能出现 OOD drift。

EVQ-Cosh 在当前 \(\tau>0\) 下把 interior quantiles 推向更小的 \(\phi\)、即更高的 \(\omega\)。这不是数值印象。令 \(x=1-u\in(0,1)\)，由 \(\sinh\) 的严格凸性和 \(\sinh(0)=0\) 有

\[
\sinh(x\tau)<x\sinh(\tau),
\]

所以

\[
\operatorname{asinh}(x\sinh\tau)>x\tau,
\qquad
\phi_\tau(u)
=1-\frac{\operatorname{asinh}(x\sinh\tau)}{\tau}<u.
\]

因此同一个 midpoint quantile index 下，EVQ 对所有 interior channels 都严格提高频率。它不是把所有旋转“变慢”，而是让更多通道在 8K 内已经走过可辨认的相位弧。较高频通道在训练内经历单位圆上更多相位状态，外推时不再第一次离开近恒等区域。

对 \(\Delta\sim U[0,L]\)，单通道 phase 的一阶矩还有闭式：

\[
\mathbb E[\cos(\omega\Delta)]
=\frac{\sin(\omega L)}{\omega L},
\qquad
\mathbb E[\sin(\omega\Delta)]
=\frac{1-\cos(\omega L)}{\omega L}.
\]

当 \(\omega L\) 很大时，均值按 \(O((\omega L)^{-1})\) 衰减，二阶矩趋向 \(I/2\)；当 \(\omega L\ll1\) 时，phase 仍集中在单位圆的一小段。这个结论精确解释了为什么“训练内 phase exposure”可以用 moment shift 诊断，但它仍只针对指定 distance prior 和单通道统计。

这给出 NLL 改善的一个具体解释：

1. 直接 EVQ 替换扰动了 pretrained phase basis；
2. 机制假设认为，300 步 q/k/v/o LoRA 利用 8K 内可见的相位方向完成了部分重新校准；该项仍需 task-weighted gradient/Jacobian 诊断验证；
3. EVQ 比 Geo 留下更少“8K 内近恒等、32K 才开始转动”的 modes；
4. 所以长位置的平均 attention-logit distribution 可能更接近训练时见过的分布，减少 catastrophic token-NLL tails。

### 2.3 为什么这不自动产生检索

该机制同时解释了“长 NLL 大幅改善而 NIAH 不改善”为什么并不矛盾：

- **平均 NLL 的监督结构不同。** 当前 LongAlpaca protocol 对全部 non-padding tokens 做 CE，绝大多数 token 可由局部语法、短依赖和既有知识预测；模型不必读取一个远端唯一 source 才能降低 loss。
- **phase coverage 不等于 distance injectivity。** 更高频更容易在 8K 内被观测，但也更周期化；不同长距离可能出现相近相位码。可观测性提高不保证全局无混叠。
- **检索需要 Q/K/V/O 的完整因果链。** Q/K 要选中 source，V/O 要运输内容，上层还要保持、组合并按 instruction 输出。仅让 logits 数值更稳定，不代表这条电路被训练。
- **32K 还有训练时不存在的竞争。** key 数量、softmax denominator、极值 distractor、KV cache 内容分布和多层递归状态都改变。8K phase feature 的性质不能消除这些问题。
- **成熟模型存在频带共适应。** 预训练形成的高能 Q/K frequency band 可能承担 sink、局部匹配或 induction 功能。全表替换即使能恢复 NLL，也可能没有恢复 source-specific routing。

因此，真正需要的不是“再做一次普通长文本 SFT”，而是 **在不超过 8K 的窗口内给出高密度、明确、可反事实验证的 dependency gradient**，然后观察该电路是否只在 EVQ 基底上更好地跨到 16K/32K。

---

## 3. 对新“频谱可观测性理论”的反向审计

### 3.1 可以作为 exact statement 的部分

1. 上述 RoPE relative-rotation identity 与每个固定 token pair 的 sin/cos 展开是精确的。
2. 给定 frequency set 和明确 distance prior，phase-feature Gram matrix

   \[
   G_{\Omega}(D)=\mathbb E_{\Delta\sim D}
   [\widetilde\psi_{\Omega}(\Delta)\widetilde\psi_{\Omega}(\Delta)^\top]
   \]

   是定义良好的；\(\widetilde\psi\) 可取 raw feature 或减去 prior mean 的 centered feature，但两者不能混报。
3. \(\omega_kL\) 是该通道在窗口内覆盖的相位弧长度；用固定阈值报告 dormant count 是精确、可复现的 schedule diagnostic。
4. 对指定 prior，phase feature 的一、二阶矩及其 train→test shift 可精确计算。

### 3.2 只有在强条件下才成立的部分

对 LoRA 参数 θ 在初始化附近线性化：

\[
\delta z_i \approx J_i\,\delta\theta.
\]

真实训练曲率更接近

\[
H_D=\mathbb E_i[J_i^\top S_iJ_i],
\]

其中 \(S_i\) 包含 softmax/LM loss curvature、labels、content activations、layer/head routing。只有额外假设：

- q/k content coefficients 对距离近似独立；
- rotary pairs 的幅值近似各向同性或可分离；
- task 对各 phase directions 提供足够 excitation；
- 层间非线性与 softmax 可在局部线性化中控制；

才可能把 \(H_D\) 的 phase 部分近似成 unweighted \(G_\Omega(D)\)。在这些条件下，更高的 Gram effective rank/更小的 condition degeneration 才能解释“更多 LoRA 方向可从 8K 数据识别”。

完整模型中这些条件都尚未验证。因此安全表述只能是：

> EVQ 在指定 distance prior 下提高了 model-free phase-feature coverage；这提出一个关于 task-weighted LoRA Jacobian/Fisher coverage 的可证伪预测，而不是证明 EVQ 更容易学会下游任务。

### 3.3 不能从该理论推出的内容

- 不能推出 lower NLL、retrieval accuracy 或 LongBench gain 的符号与幅度；
- 不能推出 `tau=1.414` 或 `d/sqrt(L)`；
- 不能把 unweighted Gram rank 当作 trained attention rank；
- 不能把 ridge leverage 当作完整 Transformer 的误差界；
- 不能说“更高频越多越好”；dependency width 较大时，低频 coarse field 可能更重要；
- 不能忽略 aliasing、attention sinks、candidate-count 和 value transport；
- 不能把一次 8K task SFT 后的 32K 成功自动归因于 EVQ，除非 matched Geo 和 midpoint-Geo controls 同时存在。

更强的反例是：若只最大化 unweighted phase rank 或只最小化有限阶 train→test moment shift，把所有 \(\omega_k\) 推到足够高通常会让训练区间内的 phase 看起来更“充分”、一二阶统计更接近稳定分布，但同时丢失 coarse field、增加周期混叠并与长依赖的尺度失配。因此 observability 本身是一个不完整、甚至单独优化会退化的目标；它不能替代 collision surrogate，不能选择 allocation，更不能选择 \(\tau\)。

同样，一二阶 moment 接近不表示完整 phase distribution 接近，更不表示经过 content-dependent Q/K weighting 和 softmax 后的 attention distribution 接近。Section 4 的 moment table 只检验一个必要方向，不构成充分条件。

### 3.4 leverage 只保留为诊断，不作为新 τ 定理

可以定义

\[
\Lambda_{\Omega,\lambda}(\Delta')
=\psi(\Delta')^\top(G_{\Omega}+\lambda I)^{-1}\psi(\Delta'),
\]

但它只在指定线性回归/RKHS、noise、regularization 和 feature-span 假设下与 prediction uncertainty 相关。它对 centering、λ、feature normalization 和 distance prior 敏感。当前不应用它重新选择 τ，也不应写进 rebuttal 作为理论闭环。

### 3.5 这套理论与旧变分理论的关系

两者不是同一个目标：

- 旧 ρ-Cosh 理论：给定 broadband collision surrogate，求 allocation density 的 exact optimizer；
- 新 observability hypothesis：给定实际训练距离分布，问有限 frequency table 在窗口内能暴露多少独立 phase directions；
- 下游任务：还要再乘上 model/data-dependent content coefficients、loss curvature 与多层 routing。

所以新理论不能“修好”旧 KL 证明，也不能由旧 surrogate 自动推出。它更接近当前 8B NLL 现象，但必须作为新机制假设单独验证。

---

## 4. Fresh model-free sanity check（不是模型实验结果）

本轮用 LLaMA-3 geometry `K=64, base=500000, L_train=8192` 比较；EVQ 使用实际计划值 `tau=1.414`：

- native endpoint Geo：φ_k=k/K；
- midpoint Geo：φ_k=(k+0.5)/K；
- midpoint EVQ-Cosh：\(\tau=1.414\)。

### 4.1 dormant channels

| Schedule | ωL < 0.3 | ωL < 1 | ωL < π | 在 8K 时 ωL<1、到 32K 时跨过 1 的 channels |
| --- | ---: | ---: | ---: | ---: |
| native Geo | 14 | 20 | 25 | 7 |
| midpoint Geo | 14 | 20 | 26 | 7 |
| EVQ τ=1.414 | 11 | 15 | 20 | 5 |

这支持“EVQ 减少 8K dormant modes”的方向，但阈值本身是 diagnostic convention，不是自然常数。

### 4.2 phase-feature entropy effective rank

使用 128 列 `[cos, sin]` features；centered 表示按该 distance prior 去均值。

| Distance prior / preprocessing | native Geo | midpoint Geo | EVQ τ=1.414 |
| --- | ---: | ---: | ---: |
| uniform 0–8K, raw | 23.60 | 22.71 | 36.23 |
| uniform 0–8K, centered | 54.96 | 54.00 | 68.98 |
| causal triangular, centered | 54.15 | 53.20 | 67.89 |
| boundary-heavy, centered | 48.38 | 47.45 | 61.98 |

EVQ 的方向对三种先验和 raw/centered 处理均保持，但数值明显依赖先验和处理方式，因此只能当 schedule-level evidence。

### 4.3 8K → 16K/32K phase-moment shift

定义每个通道的二维 phase vector \(x_k=[\cos(\omega_k\Delta),\sin(\omega_k\Delta)]\)，并计算

\[
D_{mom}(L_0,L_1)=\frac1K\sum_k
\left(\|\mu_k(L_1)-\mu_k(L_0)\|_2^2
+\|\Sigma_k(L_1)-\Sigma_k(L_0)\|_F^2\right).
\]

| Prior | Target | native Geo | midpoint Geo | EVQ τ=1.414 |
| --- | ---: | ---: | ---: | ---: |
| uniform | 16K | .07393 | .07393 | .06147 |
| uniform | 32K | .17071 | .17060 | .13883 |
| causal triangular | 16K | .04564 | .04565 | .03821 |
| causal triangular | 32K | .13061 | .13063 | .10728 |

EVQ 的 unweighted phase moments 从 8K 到更长区间移动得更少，和 long-position NLL stability 的方向一致。但最大单通道 shift 并未明显改善，且真实模型会以 Q/K energy、head、layer 和 content 非均匀加权；因此这仍不是 trained-model 解释的完成态。

### 4.4 必须补的 task-weighted 验证

真正有决策价值的诊断不是继续堆 model-free 指标，而是从训练中记录：

1. 每层/head/rotary pair 的 q/k activation energy；
2. 每 pair 的 LoRA gradient/update energy；
3. source token 对 answer logits 的 causal effect；
4. task-weighted phase Gram 或 Jacobian sketch；
5. 上述量是否预测 16K/32K 的 source-dependent success，而不仅预测 NLL。

如果 unweighted rank 提升但 task-weighted Jacobian 仍集中在少数原生 frequency band，新理论就被否定或需要大幅修改。

---

## 5. 可证伪假设

### H1：phase-coverage hypothesis

在同样的 8K source-dependent training 下，EVQ 会让 task gradient 覆盖更多 rotary pairs，且训练到的 routing rule 在 16K/32K 保持得比 same-grid midpoint Geo 更好。

预测：EVQ 的 task-weighted pair coverage、更长距离 pair consistency 和 source-removal causal effect同时上升。

### H2：NLL stability 只是 task-independent phase stabilization

EVQ 主要减少长位置的平均 logit/NLL drift，但没有改善 source-specific routing。

预测：temporal NLL 继续显著改善，controlled retrieval、counterfactual source dependence 和真实 QA 均不优于 Geo。

这是完全可能、并与当前观测相容的结果。

### H3：完整 EVQ 替换损伤了 pretrained frequency band

EVQ 能改善 dormant-mode coverage，但同时移动了成熟模型已经重度使用的 frequency band，导致 retrieval circuit 被破坏或无法在短预算内恢复。

预测：EVQ adapter 在 temporal NLL 上好，但高-energy q/k pairs 的 source attention/causal effect弱于 Geo；adapter×frequency cross-swap 显示明显 co-adaptation failure。

若 H3 得到支持，下一 venue 的合理方法是 **band-preserving/partial EVQ**，而不是继续增加全表替换训练步数。

### H0：8K 内 task learning 本身不能跨到 16K/32K

即使 Geo 和 EVQ 都在 8K 学会 source-dependent task，两者在 16K/32K 都失败，且差异不稳定。此时严格 zero-phase-exposure 外推不成立；再做 PoSE 可以回答“短物理窗口、长 position exposure”是否足够，但不能挽救纯 8K 外推 claim。

---

## 6. Stage 0：不训练的最低成本诊断

在任何新 8B LoRA 前，先使用已有 base、Geo+LoRA 和 EVQ+LoRA artifacts，修通一个严格的 source-dependent evaluator。

### 6.1 评测矩阵

| Adapter | Runtime frequency | 目的 |
| --- | --- | --- |
| none | native Geo | 原始能力 |
| none | midpoint Geo | 只看 quantizer shift |
| none | EVQ | 只看 direct frequency replacement |
| Geo+LoRA | native Geo | matched Geo parent |
| Geo+LoRA | EVQ | cross-swap：通用 adapter 还是 frequency-specific co-adaptation |
| EVQ+LoRA | native Geo | cross-swap：EVQ adapter 是否只是在做普通 SFT |
| EVQ+LoRA | EVQ | 当前 positive NLL arm |

Cross-swap 是机制诊断，不是公平性能 arm；其失败不能单独证明任何 schedule 更差。

### 6.2 长度与指标

长度固定为 4K/8K/12K/16K/24K/32K。每组都使用：

- 完整 LLaMA-3 chat template，context budget 包含 template 和 answer；
- original / source-swapped / source-removed 等长 triplet；
- extracted exact match、strict generation exact match、gold containment；
- answer-token NLL；
- pair consistency；
- source-removal ΔNLL；
- EOS 与输出长度；
- KV、last-write、two-hop/variable-tracking 分项。

只有这一 evaluator 能在 8K 对 base 和现有 adapter 给出合理结果，才允许训练。若 NIAH 失败只是输出格式/EOS 问题，必须与真正的 source dependence failure 分开报告。

### 6.3 Stage 0 决策门

- 若 EVQ+LoRA 在 16K/32K 已有显著 source dependence，只是 strict exact 失败：先修 decoding/output contract，不训练。
- 若 EVQ+LoRA 与 Geo+LoRA 都无 source dependence：继续 Stage 1，检验 task-directed 8K gradient。
- 若 base/Geo 在 8K 也无法通过 evaluator：停止，先修任务或 evaluator。

---

## 7. Stage 1：rebuttal-fast matched continuation（只训练 8K）

这是 8 天内最快能改变结论的实验。它从已经完成的两份 seed-42 parent adapters 出发：

- native Geo + LongAlpaca LoRA parent；
- EVQ τ=1.414 + LongAlpaca LoRA parent。

两臂继续训练完全相同的 8K dependency curriculum。它回答“现有 NLL-stable adapter 能否在不见 >8K 的情况下被转化成能力”，但仍将 EVQ shape 与 midpoint quantizer change 作为组合干预；不能称 pure density-shape proof。

### 7.1 不变协议

- Model: 同一 manifested LLaMA-3-8B-Instruct bytes；
- physical seq len: `8192`；
- `position_ids.max() <= 8191`，训练器必须 fail closed；
- LoRA: 延续 q/k/v/o，`r=64, alpha=128, dropout=0.05`；
- backbone、LM head、frequency tensor 全部冻结；
- 每臂同一 parent step、数据 tensor、row order、seed、optimizer、LR 和 token budget；
- 不使用 YaRN/NTK/PI/PoSE；
- 不使用 teacher、student、hidden-state matching 或 logits distillation；
- 不在看到 EVQ 结果后调 rank、τ、数据比例或 decoding。

### 7.2 训练数据：在 8K 内教“依赖规则”，不是教 benchmark 答案

每个 semantic instance 在多个 source-query gap 上重排，但答案规则不变。所有 task family 的 train/eval nonce、模板和 filler 文档严格分离。

建议 mix：

| 比例 | Task | 必须学到什么 |
| ---: | --- | --- |
| 30% | nonce key–value / source swap | 内容绑定与远端寻址 |
| 20% | last-write-wins | 抑制旧 source，选择最新证据 |
| 20% | two-hop / variable tracing | 多 source 组合与状态更新 |
| 15% | natural-evidence extractive QA | 把电路连接回自然文本，而非只适配模板 |
| 15% | held-out-style instruction replay | 保持 LLaMA-3 instruction/output behavior |

距离 prior 对所有 arms 完全相同：

- 20%: 256–2,048；
- 30%: 2,048–5,120；
- 50%: 5,120–7,680。

一半 algorithmic examples 使用高 distractor density，在 8K 内尽量接近 32K 的 candidate-count pressure；这仍不能完全复制 32K softmax competition，必须作为限制报告。

不得直接训练 official RULER/LongBench test templates。训练任务教内容路由原语，真实 benchmark 检查 task/format generalization。

### 7.3 Loss

Primary loss 仅为 assistant answer tokens 的 causal CE：

\[
\mathcal L=\frac1{|A|}\sum_{t\in A}
-\log p_\theta(y_t\mid x,y_{<t}).
\]

Prompt、filler、source 和 query labels 全部为 `-100`。Original 与 swapped 各自作为有正确答案的训练样本；source-removed 只用于评测，不把一个任意答案强加给无 source 的输入。

这确实是一种任务微调，但关键区别是：

- 训练仍严格限制在 8K；
- Geo 与 EVQ 接受完全相同的 task signal；
- 训练 task family 与 16K/32K 真实评测 family 分离；
- 只有 EVQ 相对 Geo 的超窗 interaction 才能归因于 frequency substrate。

### 7.4 Budget

每个 optimizer step 固定 32,768 physical tokens：`seq_len=8192, microbatch=1, grad_accum=4`。

| Segment | Steps | Tokens / arm | 用途 |
| --- | ---: | ---: | --- |
| S1 | 32 | 1,048,576 | time-to-signal |
| S2 | +32 | +1,048,576 | 仅当两臂都在 8K 学到 task 且仍在改善 |

两臂最小总预算 2,097,152 tokens，绝对上限 4,194,304 tokens。每 32 步保存 checkpoint；不得在看到 16K/32K test 后决定是否续训。续训只看 8K validation learning curve 和预注册 guardrail。

### 7.5 Stage 1 成功条件

先过 in-range learning gate：

- 两臂都达到 8K validation pair consistency ≥ .80；若只有一臂通过，则这是 in-range trainability 差异，不是同等已学能力的超窗迁移，停止 transfer claim；
- source-removal positive fraction ≥ .75；
- KV、update、two-hop 均无空 subgroup；
- 相对各自 parent 的 8K temporal NLL 恶化 ≤ .10；
- short-core benchmark macro 下降不超过 2 points。

再看 zero-shot 16K/32K。对 higher-is-better metric \(M\)，先计算

\[
I_L=
\bigl(M^{\mathrm{EVQ}}_{\mathrm{post},L}-M^{\mathrm{EVQ}}_{\mathrm{pre},L}\bigr)
-\bigl(M^{\mathrm{Geo}}_{\mathrm{post},L}-M^{\mathrm{Geo}}_{\mathrm{pre},L}\bigr),
\]

避免把两个 parent adapters 的初始能力差异误当作新 curriculum 的 EVQ effect。最终同时报告 endpoint difference 和该 difference-in-differences：

- primary：pair consistency 与 source-causal success；
- EVQ-vs-Geo endpoint difference 与 \(I_L\) 的 paired bootstrap 95% CI 下界都 > 0；
- effect 至少在 16K/32K 平均达到 +5 points，才算 mechanism-positive；
- 若两长度均 ≥ +10 points，且至少两个 unseen task families 同方向，才够资格称 strong supporting evidence；
- temporal NLL 优势必须与 task signal 同时存在，不能单独过关。

Stage 1 是 single-seed continuation。即使强阳性，也只能进入 rebuttal supporting evidence；不能称 multi-seed general law。

---

## 8. Stage 2：same-base 三臂 confirmatory（主科学方案）

只有 Stage 1 出现真实的 source-dependent EVQ advantage，才从同一 base checkpoint 和同一 LoRA initialization 运行三臂：

1. `G_endpoint`: native endpoint Geo；
2. `G_midpoint`: midpoint Geo，τ=0；
3. `EVQ_midpoint`: midpoint EVQ-Cosh，τ=1.414。

三臂都在 step 0 直接使用最终 frequency tensor，不做 homotopy。理由是现有 300-step 8B 结果已经证明 direct replacement 对 NLL 是可学习的；在没有 in-range optimization failure 的证据前，加入 homotopy 只会增加变量并削弱归因。

### 8.1 模型与参数

- LLaMA-3-8B-Instruct，同一 byte manifest；
- q/k/v/o LoRA，全部 32 层，`r=64, alpha=128, dropout=.05`；
- 不解冻 MLP、embedding、LM head 或 layer norm；
- frequency fixed，不学习 τ、不学习 per-head gate；
- BF16、同一 attention backend、optimizer 和 sample order。

### 8.2 Budget

`128 steps × 32,768 tokens = 4,194,304 tokens/arm`；三臂总计 12,582,912 tokens。

Checkpoint 固定在 step 0/32/64/128。训练选择只依据 8K validation，不打开 16K/32K final test。

### 8.3 因果分解

在每个 endpoint 计算：

\[
\Delta_{quantizer}=G_{midpoint}-G_{endpoint},
\]

\[
\Delta_{shape}=EVQ_{midpoint}-G_{midpoint}.
\]

只有第二项能支持 cosh allocation shape 的独立贡献。`EVQ - native Geo` 仍只是 end-to-end schedule intervention。

### 8.4 Confirmatory success

- `EVQ_midpoint` 在 16K 和 32K 的 controlled source-causal score 均优于 `G_midpoint`；
- paired bootstrap 95% CI 下界 > 0，macro gain ≥ 5 points；
- 至少两个真实 long-context task datasets 同方向，且 aggregate official metric 改善 ≥ 3 points；
- 4K/8K task、temporal NLL 和 short-core benchmark 不发生超过预注册阈值的退化；
- task-weighted rotary-pair diagnostics 与能力 gain 有一致方向；
- `G_endpoint`/`G_midpoint` 结果允许明确量化 quantizer confound。

若三臂只在 NLL 上分离、能力不分离，当前“能力转化”主张失败。

---

## 9. 评测：必须区分 phase、competition、retrieval 和真实任务

### 9.1 In-range 保持

- 4K/8K controlled tasks；
- 8K temporal holdout NLL；
- 小型 fixed short-core suite：MMLU/ARC-style knowledge reasoning、GSM-style reasoning、instruction-following各一个固定子集；
- 输出格式、EOS 和 generation length。

这些只做 catastrophic-forgetting guardrail，不用来宣称普遍能力提升。

### 9.2 Long-position language modeling

- 冻结的 temporal 8K/16K/32K prefixes；
- position buckets：0–4K、4–8K、8–12K、12–16K、16–24K、24–32K；
- 同时报告 mean NLL、tail quantiles 和 per-pack deltas，检验 EVQ 是否主要消除 catastrophic tails。

### 9.3 Controlled source-dependent capability

- single KV；
- multi-key / multi-value；
- last-write-wins；
- variable tracking；
- two-hop composition；
- original/swapped/removed triplets；
- 4K/8K/12K/16K/24K/32K。

Primary 是 autoregressive extracted exact / pair consistency；teacher-forced NLL 只作诊断。

### 9.4 RULER/NIAH

RULER 必须先修成可复现的小型 official subset，至少覆盖：

- S-NIAH；
- MK/MV-NIAH；
- variable tracking；
- aggregation/common-word extraction。

RULER 本身强调 vanilla NIAH 只测很浅的检索，因此不能只用一个 needle 宣称下游能力。若 full harness 在 rebuttal 时间内仍不可靠，宁可报告经过测试的 bounded subset，也不报告不可复现的总分。

### 9.5 真实下游长上下文任务

优先固定三类 held-out datasets，并按实际 tokenizer 长度筛选 8–32K 样本：

- single-document evidence QA（如 Qasper/MultiFieldQA 类）；
- multi-document multi-hop QA（如 2Wiki/Hotpot/MuSiQue 类）；
- long-document generation/summary（如 NarrativeQA/GovReport 类）。

不得从这些 test examples 生成训练模板。使用 official metric，并同时给 length bucket 和 answer-source distance。若只能完成两个 dataset，优先 QA 与 multi-hop，而不是用 summary ROUGE 掩盖 retrieval failure。

### 9.6 phase 与 key-count 分解

对同一 controlled semantic instance 构造三个 evaluation-only 版本：

1. `dense-8K`：所有 position id 与 physical tokens 都在 8K；
2. `phase-only-32K`：最多 8K retained tokens，但 position ids 稀疏跨到 32K；
3. `full-32K`：真实 32K tokens、连续 positions 和完整 distractor competition。

解释：

- 2 失败、1 成功：主要是 phase extrapolation；
- 2 成功、3 失败：主要是 key-count/softmax competition 或多层长序列状态；
- 2/3 都成功：才说明能力真正跨到长位置和长序列；
- 三者都失败：任务本身未学会或 evaluator 有问题。

`phase-only-32K` 只用于诊断，不能等同于完整 32K capability。

---

## 10. 训练动态与机制诊断

每个 checkpoint 必须保存：

- layer/head/rotary-pair q/k activation energy；
- q/k LoRA-B gradient energy 与 update energy；
- v/o update norm；
- answer predictor 对 source span 的 attention mass；
- source removal、source swap 对 gold logit 的 paired causal effect；
- high-energy pretrained frequency band 在训练前后的利用变化；
- exact runtime inv_freq hash；
- position-id max、distance histogram、data/sample hashes。

重点检验以下预测：

1. EVQ 的 unweighted phase rank 提升，是否真的变成更广的 task-weighted pair gradient coverage；
2. 哪些 layer/head 把额外 coverage 变成 source attention；
3. NLL gain 是否只来自 local/sink heads，而 retrieval heads 没有改善；
4. capability gain 是否在 adapter×frequency cross-swap 后消失，从而证明 schedule-specific co-adaptation。

只有第 1→2→3 的链条成立，才有资格说“模型学会并利用了新的频率分配”。

---

## 11. 若严格 8K 外推失败：PoSE 只作为第二问题

若模型在 8K 内已稳定学会 source-dependent task，但 16K/32K 两个 schedule 都失败，可运行一个 short-physical / long-virtual-position 对照：

- physical tokens ≤ 8192；
- chunks 使用跳跃 position ids，virtual span 到 32K；
- Geo/midpoint-Geo/EVQ 三臂保持相同 gaps、tokens 和预算；
- 仍只训练 answer CE；
- 不训练 16K/32K physical sequence。

这类训练已有 [PoSE](https://arxiv.org/abs/2309.10400) 先例：它用固定物理窗口和跳跃位置暴露目标位置，曾以 2K 训练窗口扩展到 128K。因此该实验能证明“无需 full-length token/memory 也能适应长 phase”，但不能证明“模型从未见过 >8K position”。

EVQ 的独立贡献仍必须是同一 PoSE protocol 下 EVQ − Geo；若两者等幅改善，贡献属于 position exposure，而不是 EVQ。

---

## 12. 若 NLL 继续好、retrieval 继续差：下一 venue 的方法改造

此时不应继续增加普通 SFT 步数。优先测试 H3：完整替换是否移动了 pretrained high-energy frequency band。

最小新方法是 `band-preserving EVQ`：

1. 在冻结 base 上用 8K calibration set 测 q/k rotary-pair energy；
2. 保留 top-energy pretrained band 的 native frequencies；
3. 只把 low-energy/dormant pairs 重分配到 EVQ quantiles；
4. 用 energy-matched random preserved subset 作对照；
5. 同样只做 8K dependency training，评 16K/32K；
6. 不先加入 per-head learnable gates，避免新参数和选择偏差。

若它恢复 retrieval 且保留 NLL stability，说明成熟模型的正确干预不是全局换基底，而是 **保留已共适应频带、重用未充分利用的 spectral budget**。这会改变方法定义，适合下一 venue，不适合在 rebuttal 中静默替换当前 EVQ。

只有 fixed partial schedule 成立后，才值得进一步测试 layer/head-specific gating 或 constrained learnable frequencies。否则 learnable routing 很容易只拟合 8K training loss，无法证明长程贡献。

---

## 13. 对当前理论、τ 和直接替换的明确裁决

| 对象 | 裁决 | 原因 |
| --- | --- | --- |
| cosh convex-surrogate theorem | 保留 | 数学成立，但只给 conditional allocation shape |
| collision/transport proxy →真实 LM/task | 大幅降级 | 尚无 task-weighted bridge；ordinary KL 不能支撑旧叙事 |
| 新 phase-observability mechanism | 保留为可证伪假设 | 与 schedule 数值和 8B NLL 现象一致，但未证明 LoRA Fisher/downstream link |
| `tau=d_eff/sqrt(L)` | 仅保留为 empirical operating default | 新理论也没有识别 τ；本次固定 1.414，不 sweep、不称 optimal |
| 8B direct EVQ replacement | 保留为 primary intervention | 现有 300-step 结果已经反驳“完全不可学习”；不先加 homotopy |
| homotopy / progressive interpolation | 暂不运行 | 只有 direct arm 在 8K task optimization 明显失败时才触发 |
| q/k-only adaptation | 放弃作为主路线 | 寻址之外还需要 value transport/readout；保留 q/k/v/o |
| teacher/student distillation | 放弃 | 已失败且目标错位；本计划不用 teacher、hidden MSE 或 logits imitation |
| 16K physical continued training | 不作为核心实验 | 会直接放弃“8K 内学到 8K 外”的最强问题 |
| learnable frequencies/gating | reviewer 后/下一 venue | attribution 更弱；先验证 fixed schedule 能否转化能力 |

---

## 14. 8 天执行优先级

### 必须做

1. **修通 evaluator，不训练。** 用 chat-template、triplet、extracted exact、containment、NLL、EOS 分解作者当天 NIAH failure。
2. **跑已有 artifacts 的 7-cell adapter×frequency matrix。** 先确定 NLL gain 是否有任何 source-dependent footprint。
3. **冻结 8K dependency curriculum。** 保存 exact tensors、position max、distance histogram、template/nonce/filler split 和 hashes。
4. **Stage 1 两臂 32-step continuation。** 只看 8K validation gate 决定是否再 32 步。
5. **16K/32K zero-shot controlled capability + temporal NLL。** 必须同时报告正负结果。
6. **至少两个真实 task families。** 优先 evidence QA 与 multi-hop QA；不以 RULER 工程失败替代。
7. **写清 interaction。** 报 EVQ-vs-Geo、before-vs-after 与 difference-in-differences，不把普通 task SFT 收益归因于 EVQ。

### Stage 1 强阳性后必须做

8. same-base midpoint-Geo confirmatory arm；若资源允许完成三臂 128-step protocol。
9. task-weighted rotary-pair/causal diagnostics，证明模型利用频率而不是只记模板。
10. paired bootstrap 与 artifact/protocol audit。

### Reviewer 明确要求或下一 venue 再做

11. PoSE short-physical/long-position exposure；
12. native-long model、YaRN/NTK/LongRoPE-adapted starting points；
13. layer/head/partial-frequency ablations；
14. band-preserving EVQ；
15. τ sweep、learnable frequency/gating；
16. 多 seed 完整三臂与更多真实 benchmark。

---

## 15. 结果→结论矩阵

| 结果 | 允许的结论 | 不允许的结论 |
| --- | --- | --- |
| EVQ NLL 好，task 与 Geo 无差异 | EVQ 改善 long-position LM stability | EVQ 提升长上下文能力 |
| Geo/EVQ task 同幅提升 | 8K task SFT 可部分外推 | EVQ 有独立贡献 |
| EVQ 在 controlled 16/32K 好，真实任务不变 | EVQ 改善 source routing primitives | EVQ 改善一般下游能力 |
| EVQ 在 controlled 与真实任务都优于 midpoint Geo | EVQ frequency shape 提供独立、短窗训练后的超窗能力增益 | universal SOTA 或 τ optimal |
| direct EVQ 8K 内也学不会，Geo 能学会 | full replacement 损伤成熟 frequency band | EVQ 原理整体错误 |
| PoSE 后两者同幅恢复 | 长 phase exposure 是关键 | EVQ 是关键 |
| band-preserving EVQ 成功、full EVQ 失败 | 保留共适应频带、重用 dormant budget 更合理 | 当前 fixed full-EVQ 已被验证 |

---

## 16. Rebuttal 与下一 venue 的措辞边界

当前即可诚实说明：

> A matched-training-pipeline 300-step, 8K-only LoRA comparison on LLaMA-3-8B shows a pronounced position-dependent interaction: EVQ incurs a much smaller in-window NLL cost than its 16K/32K NLL gains, consistently across all held-out packs. Geo uses the native endpoint grid whereas EVQ uses midpoint quantization, so this is an end-to-end schedule comparison rather than a pure density-shape contrast. The result demonstrates rapid adaptation and long-position language-model stability, but not yet retrieval or downstream capability.

只有 Stage 1/2 通过后才能补：

> Under identical source-dependent training confined to 8K tokens and positions, EVQ transfers the learned routing behavior to unseen 16K/32K contexts better than the matched Geo control.

不能写：

- “EVQ automatically gives retrieval from perplexity”；
- “8K training is equivalent to 32K training”；
- “phase Gram rank proves downstream capability”；
- “τ is theoretically identified”；
- “PoSE-style virtual positions are zero-exposure extrapolation”；
- “task fine-tuning gain belongs to EVQ”而没有 matched Geo；
- “RULER failed”而实际是 evaluator 未跑通。

---

## 17. 外部工作对本计划的影响（内部使用）

- [PoSE](https://arxiv.org/abs/2309.10400) 已证明短物理窗口可通过跳跃 position ids 暴露长相位；因此它是高效 adaptation baseline，不是纯 EVQ 外推证据。
- [RULER](https://arxiv.org/abs/2404.06654) 明确指出 vanilla NIAH 只覆盖浅层检索，并包含 multi-hop、tracking 和 aggregation；因此评测不能止于单 needle。
- ICLR 2026 的 [How Base Frequency Shapes RoPE](https://openreview.net/pdf/aac47f50d94dd6b2ae1151f8fcfb8f822304638e.pdf) 报告低频 dimensions 可能弱利用、frequency band 早期形成且在 context extension 后持续；这支持“dormant modes + pretrained band”作为诊断方向，但不证明 EVQ-Cosh。
- ICLR 2026 的 [Frayed RoPE and Long Inputs](https://openreview.net/pdf/cc6255d9b5a3c2354b3a1a29efdee1c6474cafa5.pdf) 从 Q/K geometry 与 attention sinks 解释 OOD rotation；这提醒我们必须测 sink/source head，而不能只测 unweighted phase features。
- 2026-07-08 发布的 [How Data Shapes RoPE Frequency Usage](https://arxiv.org/abs/2607.07678) 认为 learned frequency usage 与训练数据 dependency width 匹配。它是 post-submission 新工作，但直接支持本计划的核心警告：频率表只有在任务数据真正激发相应依赖时才会变成能力，而且“更多高频”不是普遍最优。

这些工作使本项目更应该把贡献边界收紧为：**closed-form finite-channel allocation + short-window phase coverage hypothesis + matched capability conversion test**，而不是声称已经拥有完整的 trained-task theory。

---

## 18. 实验输出与可复现性合同

建议新运行使用外部显式目录 `EVQ_8K_TRANSFER_WORK_DIR`，至少写出：

| Output | Format | 必须包含 |
| --- | --- | --- |
| `data_manifest.json` | JSON | tokenizer/model hashes、split、task mix、positions、distances、tensor hashes |
| `protocol.json` | JSON | arm、frequency hash、parent hash、LoRA/optimizer、steps、token budget |
| `checkpoint-{32,64,128}/` | adapter + JSON | immutable parent、runtime frequency、trainer state、diagnostics |
| `controlled_per_example.jsonl` | JSONL | task、length、distance、counterfactual group、generation、NLL、exact |
| `temporal_per_pack.jsonl` | JSONL | domain、pack、prefix、NLL sum/token count |
| `real_task_per_example.jsonl` | JSONL | dataset、length、metric inputs、prediction |
| `mechanism_diagnostics.npz/json` | arrays + manifest | layer/head/pair activation、gradient、update、source causal effects |
| `summary.json` | JSON | preregistered metrics、CIs、all arms、all failures |

GPU 前必须完成 data/model hashes、CPU tests、shell syntax、dry run、output non-overwrite 和 automatic evaluation。任何 arm 的数据、labels、position ids、sample order 或 token budget变化都构成新 protocol。

---

## 19. 最终理论判断

这次重审后，我不会把“phase observability”写成新定理来替换旧代理。更准确的层次是：

1. **Exact algebra**：RoPE 把相对距离映射成有限 sin/cos phase features；cosh 是 stated surrogate 的 exact optimizer。
2. **Model-free schedule fact**：在 LLaMA-3 8K geometry 下，EVQ 减少 dormant modes、提高多个 distance priors 下的 phase-feature effective rank，并降低 unweighted 8K→16/32K phase-moment shift。
3. **Conditional mechanism hypothesis**：若 task-weighted LoRA Jacobian 能继承这些性质，短窗梯度可校准更多频率方向，解释快速适应和长位置 NLL 稳定。
4. **Unresolved bridge**：source-specific retrieval、multi-hop 和真实 QA 还需要 task excitation、coarse addressability、candidate competition、value transport 与 pretrained frequency-band preservation。

所以这篇论文当前最本质、最可守的科学命题不是“一个闭式频率表自动赋予长上下文能力”，而是：

> **有限训练长度不仅限制可见 token 数，也限制固定 RoPE 频率表中哪些 phase directions 实际被训练。频率 allocation 改变了这种训练可见性。EVQ 已显示它能显著改变成熟 8B 模型的超窗语言建模稳定性；下一项决定性证据，是在严格相同的 8K dependency training 下证明这种可见性差异能否转化为 16K/32K 的 source-dependent 与真实任务能力。**

这个命题足够强，也足够容易被实验否定。
