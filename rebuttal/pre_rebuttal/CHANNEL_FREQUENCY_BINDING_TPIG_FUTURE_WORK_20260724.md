# Channel-frequency binding and T-PIG — internal future-work memo

Status: **internal / post-rebuttal research only**
Recorded: 2026-07-24

This memo preserves a user-supplied theoretical analysis of the 2026-07-24
causal-spectrum and held-out-pruning results. It is not a current rebuttal
claim or an approved experiment plan. T-PIG changes phase participation and is
a new operator, not EVQ-Cosh. For the current rebuttal, only the established
frequency-allocation positioning may be reused, subject to the concern-driven
rules in `../rebuttal_0723/theory_results/EVQ_COSH_REBUTTAL_PRINCIPLES.md`.

---

以下用 **【数学结论】、【经验事实】、【强假设】、【待验证推论】** 标记认识状态。对新增实验事实，按题述视为已确认；涉及具体 hook、gate 粒度和算力的部分均属于实现假设。

## 1. 对统一机制假设的判决

**判决：部分通过，而且是当前最好的工作假设；但尚未完成因果识别。**

* **【经验事实】已较强成立的是两件事：**

  1. **channel–frequency binding 存在。** 固定 frequency multiset、只改变其与通道的对应关系即可显著改变 NLL，因此训练后通道不是匿名槽位。
  2. **target-dependent OOD interference 存在。** 单频对效用会跨长度翻转，独立选择并移除 OOD 有害频对可以改善 Geo/EVQ，且 FMR 中翻转和可剪余量都较少。

* **【待验证推论】尚未被唯一证明的是：这种 binding 是否主要由训练期的 phase co-adaptation 造成。** 静态的通道范数差异、语义分工和 softmax 竞争，也能使“同一频率放到不同通道”产生不同结果。

因此，建议把原假设从较强的：

> 通道进入训练未覆盖的相位区域后发生功能翻转

改成更可证伪的：

> **每个通道学习的是相对于其 RoPE 基函数的有符号响应；目标长度改变 phase–content–gradient 的联合测度，使部分通道的 signed utility 变负。频率修正通过减小这种联合测度失配而减少干扰。**

这里“完全未见过的相位 support”不是必要条件；**相位权重重新分布**也足以造成符号翻转。

这能同时解释：raw EVQ 有益、Cosh 不唯一、优化 target base 后静态 residual 消失、FMR 减少翻转、norm band 不等于 causal band，以及 8B 中 routing 改善但 readout 失败。论文原理论本来也只严格覆盖所写 surrogate，而不覆盖训练后注意力目标。

---

## 2. 最小数学模型、因果量与低成本 proxy

对一个 layer/head，令

[
\alpha_{ij,k}
=q^C_{i,k}\overline{k^C_{j,k}}
=A_{ij,k}+iB_{ij,k},
\qquad
\theta_{ij,k}(T)=\Delta_{ij}\omega_k(T).
]

则第 (k) 个二维频对对 attention logit 的贡献为

[
z_{ij,k}(T)
===========

## A_{ij,k}\cos\theta_{ij,k}(T)

B_{ij,k}\sin\theta_{ij,k}(T).
]

这是 RoPE logit 的精确复数形式，而不是 collision surrogate。

**【数学结论】通道—相位共同适应在结构上必然“可以发生”：**

[
\frac{\partial z_{ij,k}}{\partial A_{ij,k}}=\cos\theta_{ij,k},
\qquad
\frac{\partial z_{ij,k}}{\partial B_{ij,k}}=-\sin\theta_{ij,k}.
]

因此，分配给通道的 (\omega_k) 会直接改变 Q/K 参数在训练中接收的梯度权重；但这只证明可能性，不证明它是主要机制。

### NoPE–RoPE phase gate

定义

[
e^{i\theta}\longmapsto 1+g_k(e^{i\theta}-1),\qquad 0\le g_k\le1.
]

于是 (g_k=1) 为完整 RoPE，(g_k=0) 为该频对的 NoPE，同时保留其内容点积。令
(\delta_{ij}=\partial\mathcal L/\partial z_{ij})，则

[
G_k(T)
:=
\left.\frac{\partial\mathcal L_T}{\partial g_k}\right|_{g_k=1}
==============================================================

\mathbb E!\left[
\delta_{ij}
\left(
A_{ij,k}(\cos\theta-1)-B_{ij,k}\sin\theta
\right)
\right].
]

**【数学结论】** 一个 backward 即可同时得到全部 (G_k(T))。若 (G_k>0)，小幅降低该频对的 phase participation 会降低 loss。

对于题目定义的硬 NoPE ablation，

[
U_k(T)=\mathcal L_T(g_k=0)-\mathcal L_T(g_k=1)
=-\int_0^1
\frac{\partial\mathcal L_T}{\partial g_k}(t),dt.
]

所以：

* (-G_k(T)) 是 (U_k(T)) 的一阶近似；
* 3–5 个 gate 节点的 integrated gradient 是远比逐频对硬删除便宜的近似；
* 所有频对可在相同几次 forward/backward 中估计，而不需要 (K) 次完整评估。

### 可证伪的 phase-mismatch 分解

令 (\mu_{k,E}) 是环境 (E) 下相位 (\theta\bmod2\pi) 的分布，令

[
f_{k,E}(\theta)
===============

\mathbb E[
\delta{A(\cos\theta-1)-B\sin\theta}
\mid\theta,E].
]

则 (G_{k,E}=\int f_{k,E},d\mu_{k,E})，并有精确分解

[
\boxed{
G_{k,T}-G_{k,L}
===============

\underbrace{\langle f_{k,L},\mu_{k,T}-\mu_{k,L}\rangle}
*{\text{phase-measure transport}}
+
\underbrace{\langle f*{k,T}-f_{k,L},\mu_{k,T}\rangle}
_{\text{content/softmax/readout response shift}}
}
]

* **【待验证推论】** 若第一项预测大部分效用翻转，则支持 phase-induced interference。
* 若第二项占主导，则更像普通的长序列 softmax、内容分布或下游非线性变化。

### (U_k) 的边界

**【数学结论】** (U_k(T)) 是“该 checkpoint、该数据、该 intervention”下的合法因果效应，但不是频对的内禀 Shapley value。它会受到：

[
I_{k\ell}(T)
============

U_{{k,\ell}}(T)-U_k(T)-U_\ell(T)
]

所刻画的交互影响，还包括 softmax 重新归一化、频对冗余、残差流/MLP 非线性和大步 ablation 的 off-manifold 效应。因此，最可靠的组合是：

[
\text{gate gradient}
;\rightarrow;
\text{integrated gradient}
;\rightarrow;
\text{少量硬 ablation 与 pairwise interaction 验证}.
]

---

## 3. 至少两个有竞争力的替代解释

1. **【强假设：effective-range + 静态通道各向异性】**
   EVQ/FMR 主要改变有效波长范围；频率置换之所以有害，只是因为 Q/K 通道本来就具有不同范数或语义角色，强通道与不同波长配对自然产生不同 NLL。它不要求训练期间真正形成 phase-specific response function。

2. **【强假设：普通 softmax/序列长度干扰】**
   键数量增加会改变 attention entropy、softmax Jacobian 和 (\delta_{ij})。即使相位不变，同一频对也可能从帮助目标 key 变成增强竞争 key。此时翻转主要来自上式第二项，而非 phase-measure transport。

3. **【强假设：欠训练与冗余剪枝】**
   Geo/EVQ 的有害频对可能是尚未充分协调的冗余特征；FMR 只是改善优化条件。若随着训练成熟所有方法的翻转都趋近于零，或硬删除收益不能被 infinitesimal gate influence 预测，那么“phase co-adaptation”就不是必要解释。

高 norm band 删除并不显示相应因果重要性，已经排除了最简单的“activation 大＝utility 大”解释。FMRoPE 原工作本身主要用 norm band 和 base–training-length 关系描述频率利用；这类量不能替代 signed causal utility。

---

## 4. 最有希望的方法：Target-conditioned Phase-Interference Gating

我会优先研究 **T-PIG：目标长度条件化的 NoPE↔RoPE 相位门控**，而不是继续设计静态频率表。

在已经优化的 FMR/YaRN 类 target-aware frequency correction
(\widetilde\omega_k(T)) 上，使用

[
e^{i\Delta\widetilde\omega_k(T)}
\longmapsto
1+
g_k(T)
\left(
e^{i\Delta\widetilde\omega_k(T)}-1
\right).
]

这增加的是 **phase participation**，而不是另一个 base 或频率缩放：

* (g_k(T)=1)：target-corrected RoPE；
* (g_k(T)=0)：保留内容点积、删除该频对的相位作用；
* 中间值：Native/NoPE 两条 logit 路径的连续混合。

**【数学结论】** 除端点外，这通常不能写成任何单一静态频率 (\omega'_k)，因此不会被重新优化 base 后简单吸收。

最小参数化可以是

[
g_k(T)=
\sigma!\left(
a_k+b_k\log\frac{T}{L_{\rm train}}
\right),
]

共享 layer/head 时仅 (2K\le128) 个参数；零参数版本则直接由独立 selection split 上的 (\widehat U_k(T)) 做软阈值。具体是否需要 layer/head 粒度属于 **【实现假设】**。

**为何它比静态 allocation 更可能超过 FMR/YaRN：**

* FMR/YaRN 决定“相位基函数放在哪里”；T-PIG 决定“已学习通道在该目标长度是否应表达相位”。
* 它直接利用了独立数据上有害频对可剪的因果事实。
* 它允许不同长度出现 gate crossover，不要求一个 schedule 在所有 (T) 上最优。
* 静态 Cosh/two-band residual 在重调 target base 后选择 (\lambda=0)，并不否定这一新自由度。
* **【风险判断】** FMR 剩余可剪空间较小，因此第一代收益可能是增量式的；若 gate 全部收敛到 1，应立即判定该方向没有额外 headroom。

### 8B readout failure

T-PIG 只能改善 routing/interference，**不能被假定为自动修复 top-1 readout**。建议另设一个明确的 readout adapter：冻结 Q/K 与 RoPE，只对最后 2–4 层和 LM head 做小 LoRA，并加入

[
\mathcal L_{\rm out}
====================

\mathcal L_{\rm CE}
+
\lambda
\left[
m-\ell_y+
\log\sum_{v\in H}e^{\ell_v}
\right]_+
+
\beta
\left[
m_s-
\big(
\ell_y^{\rm source}
-------------------

\ell_y^{\rm shuffled}
\big)
\right]_+ ,
]

其中 (H) 是当前 top-(k) hard negatives。

**【待验证推论】** 第一项把正确 token 从 top-thousands 推入 top-1；第二项强迫答案 logit 真正依赖远程 source。成功标准必须是 answer logit margin、正确 token rank 和 EM，而不是 PPL 或 attention routing。

---

## 5. 最有区分力的低成本机制实验

### Frozen phase-only binding factorial

使用 Geo、EVQ、FMR 的现有 early/final checkpoints，在完全相同的 token、key 数量和内容下，将 position IDs 从 (n) 改成 (sn)，其中 (s\in{1,4,8})。这只扩大相位，不引入更多 key；再与真实长序列评估对照。

对每个 (s) 做 (2\times2)：

|                                  | 原通道—频率对应 | 置换对应、multiset 不变 |
| -------------------------------- | -------: | ---------------: |
| Native frequency/range           |        A |                B |
| target-corrected frequency/range |        C |                D |

同时测：

* 全部频对的一阶 (G_k(T))；
* top/bottom/random 少量频对的 3-point IG 与硬 (U_k(T))；
* top-4 频对的 pairwise interaction；
* phase-measure 项与 response-shift 项。

**【实现假设】** 推理框架允许自定义 position IDs、频率表及 NoPE gate；不假设代码中已经存在这些 hook。

**建议预注册的成功标准：**

* target correction 在 phase-only arm 中将效用翻转率降低至少 30%；
* 频率置换重新带回至少一半的翻转或产生至少 (0.03) NLL 的 correction×binding interaction；
* (-\mathrm{IG}_k) 与 held-out 硬 (U_k) 的 Spearman 相关至少 0.5；
* 上述效应在成熟 checkpoint 中仍存在，而不是随训练消失。

这将支持“phase mismatch + learned binding”。

**否定标准：**

* correction 只有主效应，而置换交互低于 (0.02) NLL；
* phase-only dilation 不产生翻转，只有增加真实 key 数量才产生；
* response-shift 项远大于 phase-measure 项；
* IG 无法预测硬删除，且 pairwise interaction 与 singleton utility 同量级。

这些结果分别支持 pure range、普通局部 NLL 干扰或冗余补偿。

**停止标准：**

先用一个 checkpoint、独立的 500–1000 条序列筛选；若 binding interaction (<0.02) NLL 且翻转变化 (<10%)，不扩展到更多 seed。硬 ablation 只验证约 16–32 个预选频对，不再逐频全扫。

---

## 6. 极少参数训练实验

在现有约 152M 的三 seed held-out checkpoint 上：

* Baseline：每个 (T) 独立优化的 target-aware frequency correction；
* Method：同一 correction + T-PIG；
* 冻结全部模型权重，只训练 (g_k(T))；
* selection 长度：2K/4K/8K；
* 独立 test：1K/2K/4K/8K/16K；
* 对 baseline 和 gate arm 使用相同 base/range 优化预算，并允许 gate arm 重新 profile correction，避免 gate 只是在补偿未调好的 base。

建议目标：

[
\min_{g,\psi_T}
\mathcal L_T(g,\psi_T)
+\lambda|1-g|*1
+\gamma\sum_k
\left|
\partial*{\log T}g_k(T)
\right|^2.
]

**成功标准：**

* 重调 correction 后，在至少两个 OOD 长度仍有 (\ge0.05) NLL 改善；
* 三 seed 同方向；
* in-range 退化不超过 (0.02) NLL；
* held-out utility-flip rate下降至少 30%；
* gate 在未参与拟合的 16K 或中间长度上可插值，而非只记忆 selection 长度。

**否定标准：**

* 最优 (g_k(T)\approx1)；
* test 增益低于 (0.03) NLL；
* 重新优化 base/range 后增益消失；
* gate pattern 跨 seed 无相关性，或只在同一 selection 数据上有效。

**停止标准：**

先跑 seed 1，最多约 1–2M gate-adaptation tokens，或连续三次评估无改善即停；若 8K test 增益未达到 (0.03) NLL，不运行 seed 2、3。因为只更新极少 gate 参数，**【实现假设】** 该实验应明显低于一次完整小模型训练，并适合单张 RTX 5090。

---

## 7. OLMo-2 1B 最应回答的问题

不要把主问题写成“1B 上 EVQ 是否赢得更多”，而应写成：

> **在匹配训练成熟度、距离分布和 RoPE 通道预算后，binding×phase-transport interaction 与跨长度 utility flip 是否仍出现，并是否被 frequency correction 系统性压低？**

应在多个 early checkpoints 跟踪：

[
\text{permutation sensitivity},\quad
\text{flip rate},\quad
\mathrm{corr}(-IG,U),\quad
\text{correction×binding interaction}.
]

**【待验证推论】变量优先级应是：**

1. train/test 的距离—相位联合分布及 target correction；
2. 训练成熟度和 Q/K 可适应自由度，包括 full training 与 LoRA；
3. (K=d_{\rm rot}/2)，而不是笼统的总参数量；
4. partial RoPE、MLA/GQA、局部/全局 attention 和 head sharing；
5. (d_{\rm head})、attention scale 与 base；
6. 总参数规模只作为这些变量和能力的间接代理。

若 1B 与 152M 的 (d_{\rm rot})、attention 架构或训练距离分布不同，就不能把差异单独归因于模型规模。FMR 工作也表明 band 位置由 base、训练长度和 head dimension 共同决定，并且可在训练早期形成，因此 time-course 比最终单点更有信息。

---

## 8. 当前 rebuttal 可保留的贡献边界

### 可以保留

* **【经验事实】** 在受控小模型、native Std-RoPE grid、多个 seed 和 held-out base/head 配置中，仅改变固定 frequency allocation 就会稳定改变 raw extrapolation。
* **【经验事实】** matched-span 不能完全解释收益，因此 allocation shape 是实质设计变量；但 exponential/two-band 说明 Cosh 没有经验特权。
* **【数学结论】** Cosh 是所写凸 surrogate 的闭式唯一解。
* **【经验事实】** (\tau=d_{\rm head}/\sqrt L) 只能称为可用 basin prior。
* **贡献定位：** EVQ-Cosh 是“显式提出并受控研究 training-time frequency allocation、同时给出一个闭式零参数实例”，不是 universal long-context recipe。

held-out base (1)M、(d_{\rm head}=128) 的结果支持 allocation effect 不局限于原始配置，但仍不能替代更大规模能力验证。

### 只能作为后续工作

* channel–phase co-adaptation 的因果主张；
* OOD phase-measure decomposition；
* T-PIG、phase augmentation 或双路径方法；
* OLMo-2 scale transfer；
* 8B 的 readout-margin 修复。

### 必须收窄

* 不再声称 Cosh 对真实 LM 目标最优；
* 不再声称静态 EVQ 与 FMR/YaRN 普遍正交或稳定互补；
* 不把 raw EVQ 描述为优于 target-aware correction；
* 不把 Q/K norm、phase variance 或 predictor band 当成 causal utility；
* 不把 8B PPL、source dependence 或 routing 改善写成 QA capability 改善。

最稳固的 rebuttal 核心应是：

> **固定 RoPE 频率表不是中性实现细节；非几何 allocation 在受控设置中具有可重复的训练后果。EVQ-Cosh 是这一设计轴的闭式零参数实例，而非被理论或实验唯一选中的最优 schedule。**

这与评审要求的 matched schedules、独立调参和 held-out 配置直接对应；新机制和新 gate 必须明确作为 post-submission research，而不能反向冒充原投稿方法。
