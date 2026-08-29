# GPT-5.6 Sol Pro research memo

> **Archived snapshot boundary.** This file is frozen external-model output
> retained only for audit provenance. Every `VERIFIED`, `DEFECT`, `mandatory`,
> `decision`, and `priority` label below is bundle-local: it is not project
> evidence, an instruction, a current priority, or experiment/edit authorization.

- **Date received:** 2026-08-25
- **Status:** external, unverified research analysis
- **Scope:** next-generation RoPE allocation and mature-checkpoint migration
- **Authority:** analysis input only; canonical facts, results, priorities, and
  experiment authorization remain with the repository owners indexed by
  `../../README.md`, `../../../HANDOFF.md`, and `../../../../INDEX.md`.

The text below is preserved from the user-supplied model response. Its
citations, equations, interpretations, and proposals have not been promoted to
project evidence or manuscript claims.

## Preserved response

# 独立裁决

这项工作的下一步不应再围绕“再找一个更好的单调 density”或“把 EVQ-Cosh 的 (\tau) 调得更准”展开。

论文已经完成了两个关键任务：

1. **识别了 allocation 这一独立设计轴。** 固定频率端点与 log-span，移动内部频率仍改变训练结果；support 与 allocation 不是同一个变量。
2. **证明了频率表不是可在推理时无痛替换的外部配置。** 静态有效秩从 (4.57) 提高到 (12.54) 时，冻结模型的 PPL 反而从 (7.14) 崩到 (76.20)；不同频率 multiset 也不存在一般的、位置无关的可逆 Q/K 线性变换来精确保持全部相对位置 logits。 

结合你在论文之后的新结果，我的核心判断是：

> **此前尚未正确表述的科学问题，不是“什么频率密度的静态几何最好”，而是：如何设计一个有限的、训练可观测的、由实际 attention 使用方式加权的低歧义位置码；以及如何把一个已经与旧位置码共适应的模型，沿低功能冲击路径迁移到新的、非共轭位置表示。**

因此应正式拆成两条方法线：

* **从头训练：有限频谱码与 head/subspace 使用方式的鲁棒共设计。**
* **成熟模型：两个非等价位置坐标系之间的功能保持型迁移。**

二者共享数学对象，但优化问题不同，不应继续用同一种 analytic table 同时承担两项任务。

下面区分：

* **[论文事实]**：PDF 或原始文献已建立；
* **[现有结果推断]**：由你当前结果支持，但尚未成为一般结论；
* **[新假设]**：建议下一步验证的机制。

---

# 一、最核心的科学问题

## 1. RoPE 表本身不是完整的功能对象

对某个 head，使用复数形式可把 RoPE logit 写成

[
z_{hij}(\Omega)
===============

\operatorname{Re}
\sum_{k=1}^{K}
a_{hijk},e^{i\omega_{hk}\Delta_{ij}},
\qquad
a_{hijk}
========

\frac{q^{\mathbb C}*{hik}\overline{k^{\mathbb C}*{hjk}}}{\sqrt{d_h}}.
]

论文也明确指出，每个频率对应的是完整的

[
V_\omega=\operatorname{span}{\cos(\omega\Delta),\sin(\omega\Delta)}
]

而不是固定相位的一条 cosine feature。

这里有三个不同对象：

1. **编码字典**：(\Omega={\omega_k})；
2. **内容依赖系数场**：(a_{hijk})，由 Q/K 学出；
3. **路由后果**：softmax 改变了哪些 value，被 (W_O) 写入 residual stream。

当前大部分 structural metric 只研究第一个对象，最多假设第二个对象各向同性，完全省略第三个对象。

真正相关的一阶输出扰动是：

[
\delta y_{hi}
=============

W_{O,h}V_{hi}^{\top}
J_{\rm sm}(p_{hi}),g_{hi},
]

其中 (g_{hi}) 是频率表变化造成的 logit 扰动，且

[
J_{\rm sm}(p)=\operatorname{diag}(p)-pp^\top .
]

若再引入 residual/output 方向的局部 Fisher 或任务度量 (F_i)，则功能冲击为

[
|\delta y_{hi}|_{F_i}^{2}
=========================

g_{hi}^{\top}
\underbrace{
J_{\rm sm}
V
W_O^\top F_iW_O
V^\top
J_{\rm sm}
}*{Q*{hi}^{\rm consequence}}
g_{hi}.
\tag{A}
]

论文附录 A.12 已经到达了 (g^\top J_{\rm sm}g) 这一 attention-conditioned Rayleigh quotient，但尚未纳入 V/O 与下游后果。

**这一步是目前最缺失的数学对象。**

它解释了为什么：

* Q/K 决定“读哪里”；
* V 决定“从那里取出什么”；
* O 决定“这个 head 把结果写到哪里”；
* 同样大小的 attention 变化，功能后果可以相差几个数量级。

近期 RoVE 甚至把“标准 RoPE 只改变 Q/K 路由、value pathway 本身对距离不敏感”作为出发点，这进一步说明 routing geometry 与 message consequence 必须分开。([arXiv][1])

---

## 2. 更准确的问题表述

### 从头训练问题

给定：

* 有限 (K)；
* 训练窗口与训练分布 (P_{\rm train})；
* support 约束；
* 一类未知的未来 dependency-scale 与 head-usage 分布 (\mathcal U)；

选择的对象不应只是一个共享 density (\rho)，而应允许一个有限频谱组合

[
\mathcal O={(\Omega_j,\mathcal H_j)}_{j=1}^{J},
]

其中 (\Omega_j) 是第 (j) 个 codebook，(\mathcal H_j) 是使用它的 heads 或 subspaces。

优化目标应是：

> 在训练可观测性和窗口内性能约束下，最小化对未知 dependency scales 和 head-use metrics 的最坏或高分位 attention-output 歧义，而不是最小化一个 uniform-distance、uniform-coefficient 的静态 Gram 指标。

### 成熟模型问题

给定预训练模型 ((\theta_0,\Omega_0)) 和目标表 (\Omega_1)，寻找路径

[
(\theta(t),\Omega(t)),\qquad t\in[0,1],
]

使得：

[
\min
\int_0^1
\underbrace{
D_{\rm native}
\bigl(f_{\theta(t),\Omega(t)},f_{\theta_0,\Omega_0}\bigr)
}*{\text{坐标迁移冲击}}
,dt
+
\lambda
\underbrace{
R*{\rm long}
\bigl(f_{\theta(1),\Omega_1}\bigr)
}*{\text{长程能力风险}}
+
\gamma
\underbrace{
R*{\rm forget}(\theta(t))
}_{\text{遗忘}}.
\tag{B}
]

这不是“再找一张表”，而是一个**受约束的函数运输问题**。

---

# 二、“target-free”究竟怎样才数学上成立

## 1. 无限长度上的无碰撞目标不成立

定义归一化 joint phase code：

[
c_\Omega(d)
===========

\frac{1}{\sqrt K}
\bigl(e^{i\omega_1d},\ldots,e^{i\omega_Kd}\bigr),
]

与原点之间的平均 chord distance 为

[
D_\Omega(d)
===========

\frac1K
\sum_{k=1}^{K}
\left[1-\cos(\omega_kd)\right].
\tag{C}
]

对任意有限频率表和任意 (\varepsilon>0)，总存在足够大的整数 (d)，使所有 (\omega_kd) 同时接近 (2\pi\mathbb Z)，从而 (D_\Omega(d)<\varepsilon)。

一个直接的 pigeonhole 证明是：令 (\alpha_k=\omega_k/(2\pi))，把

[
0\alpha,1\alpha,\ldots,N^K\alpha \pmod 1
]

放入 (N^K) 个边长 (1/N) 的 (K) 维小盒；两个点必在同一盒，其差 (1\le q\le N^K) 使所有 (q\alpha_k) 同时接近整数。

因此：

* 有理相关频率会出现精确或低阶 recurrence；
* 一般的有理独立频率也会出现任意精确的近 recurrence；
* 对有理独立频率，Kronecker–Weyl 理论还意味着轨道最终在相应 torus 上均匀分布。([arXiv][2])

由此得到一个重要结论：

> **“在所有未限制的未来长度上控制最坏碰撞”对所有有限表都会退化；而 generic 表的无限期 phase-only 平均风险也会趋向相同的 torus 平均。真正可优化的是有限时间 recurrence、shell-wise sidelobe、训练可观测性和部署长度风险。**

---

## 2. 三种不同的 target-free

| 定义                                                 | 是否良定义 | 含义               |
| -------------------------------------------------- | ----: | ---------------- |
| 对所有 (d\to\infty) 保证无碰撞                             |     否 | 有限表必然 recurrence |
| 不指定单个目标长度，报告完整 anytime/Pareto 曲线                   |     是 | 不产生唯一标量最优表       |
| 在预先声明的 log-length 风险类上最小化 Bayes 风险或 minimax regret |     是 | 最实用，也最适合方法 claim |

建议采用 dyadic shells：

[
S_m=
[2^mL_{\rm train},2^{m+1}L_{\rm train}),
\qquad m=0,1,\ldots
]

并定义：

[
R_\pi(\Omega)
=============

\sum_{m=0}^{\infty}
\pi_m,
\operatorname{CVaR}*{d\in S_m}
A*\Omega(d),
\tag{D}
]

其中 (A_\Omega(d)) 是 near-collision 或 usage-conditioned ambiguity，(\pi_m) 是在研究开始前固定的、具有无限 support 的 log-scale prior。

若不愿指定唯一 prior，可以声明一个 prior 集合 (\Pi)，优化：

[
\min_\Omega
\sup_{\pi\in\Pi}
\left[
R_\pi(\Omega)
-------------

\inf_{\Omega'}R_\pi(\Omega')
\right].
\tag{E}
]

这是真正的 **target-free minimax regret**：

* 不输入单个 (L_{\rm target})；
* 不读取某个测试长度的 LM loss；
* 但明确承认任何设计都隐含一个 scale-risk preference。

还应报告 recurrence survival curve：

[
T_\varepsilon(\Omega)
=====================

\min{d>L_{\rm train}:D_\Omega(d)\le \varepsilon}.
\tag{F}
]

不要把不同的 (T_{0.1},T_{0.05},T_{0.01}) 强行压成一个数字。

---

## 3. “无 target”不等于“无 dependency prior”

近期理论表明：对于宽度为 (W) 的 dependency kernel，满足无歧义场约束时，单频率的最优量级是

[
\omega^\star=\frac{\pi}{W},
]

即频率尺度随 dependency width 成反比；Position Interpolation 只有在长程 dependency 是训练期 dependency 的 dilation 时才自然保持这种效用。([arXiv][3])

这意味着：

* 不同 dependency-width 类不可能共享一个无条件单频最优解；
* geometric RoPE 的“每个 log-frequency 区间等量采样”，等价于隐含地假设 dependency scales 在 log-space 上大致均匀；
* EVQ-Cosh 相当于在这个 prior 上叠加“慢端训练不可观测/冗余”的修正；
* phase-chord 更像是从实际 attention-distance 中估计 dependency-scale prior。

所以“target-free”可以做到，但“assumption-free universal optimum”做不到。最合理的目标是：

> **覆盖一类 dependency-scale 分布，而不是预测一个部署长度。**

---

# 三、把 collision、phase OOD、分辨率、可观测性和使用方式放进同一框架

令

[
\phi_\Omega(d)=
[
\cos(\omega_1d),\sin(\omega_1d),\ldots,
\cos(\omega_Kd),\sin(\omega_Kd)
]^\top.
]

对某个距离分布 (P)，定义 feature covariance：

[
G_P(\Omega)
===========

\mathbb E_{d\sim P}
[
\phi_\Omega(d)\phi_\Omega(d)^\top
].
\tag{G}
]

它有两个对偶解释：

* column Gram：频率方向是否冗余、训练是否能区分；
* row kernel：不同位置 codeword 是否相似、是否碰撞。

在同一个距离权重和 metric 下，它们来自同一个 feature operator。当前理论的裂缝并不是“collision 与 redundancy 是两件毫无关系的事”，而是：

> **使用了错误的距离分布和错误的系数/后果度量。**

可以分别定义：

### 1. 训练可观测性

[
O_{\rm train}(\Omega)
=====================

\lambda_{\min}
\left(
G_{P_{\rm train}}(\Omega)
\big|*{\mathcal S*{\rm used}}
\right).
]

更实际的版本是频率参数 Fisher：

[
O_k
===

\mathbb E_{\rm train}
\left[
g_k^\top
Q^{\rm consequence}
g_k
\right],
\qquad
g_k=\frac{\partial z}{\partial\log\omega_k}.
\tag{H}
]

若 (O_k\approx0)，该频率在训练期即使数学上存在，也没有足够信号让 Q/K 与它稳定共适应。

### 2. phase-OOD amplification

对某个部署 shell (S_m)：

[
A_m^{\rm OOD}(\Omega)
=====================

\lambda_{\max}
\left[
\bigl(G_{\rm train}+\epsilon I\bigr)^{-1/2}
G_{S_m}
\bigl(G_{\rm train}+\epsilon I\bigr)^{-1/2}
\right].
\tag{I}
]

它测量：是否存在某个 coefficient direction，在训练窗口几乎不可见，却在部署 shell 上被大幅放大。

### 3. 局部分辨率

近 (d=0)：

[
|\phi_\Omega(d)-\phi_\Omega(0)|^2
=================================

d^2\sum_k\omega_k^2+O(d^4).
]

因此

[
I_{\rm local}(\Omega;w)
=======================

\sum_kw_k\omega_k^2
\tag{J}
]

是 usage-weighted local resolution。频率全部变慢可以延长 field，却必然损失局部分辨率。

### 4. joint recurrence / collision

若某个 head 对频率的实际使用权重为 (w_k)，则其 ambiguity function 为

[
A_{\Omega,w}(d)
===============

\left|
\sum_kw_ke^{i\omega_kd}
\right|^2.
\tag{K}
]

展开得到

[
A_{\Omega,w}(d)
===============

\sum_kw_k^2+
2\sum_{k<\ell}
w_kw_\ell
\cos((\omega_k-\omega_\ell)d).
\tag{L}
]

这说明 practical sidelobe 不只取决于频率位置，还取决于：

* frequency-difference multiset；
* head 实际激活了多少个频率；
* 是否有多个 difference 在某个长度附近相干叠加。

全表 (K=64) 的 recurrence 可能极远，但若一个 retrieval head 只强烈使用 3–6 个频段，它的**有效 recurrence 维数**会小得多。

这也是为什么只优化 uniform-weight full-table collision 可能严重误判。

---

# 四、当前矛盾应如何解释

## 1. 为什么 structural pair geometry 更好，却不能稳定预测 LM

### [论文事实]

论文已经给出了最直接的反例：静态 (r_2) 显著提高，但 frozen PPL 灾难性恶化。论文也明确限定 collision、effective rank 与 logdet 描述的是“没有 learned coefficients 的 table”。

### [现有结果推断]

phase-isotropy 与 min-eigenvalue 继续失败，不是偶然噪声，而是暴露了四个系统性缺项：

1. **uniform-distance prior 错位。**
   LM 不均匀使用所有相对距离。

2. **block whitening 去掉了真实 band utilization。**
   一个在数学上冗余的慢频 pair，可能是模型故意使用的近-NoPE semantic channel。

3. **最坏方向不等于被训练使用的方向。**
   min-eigenvalue 会花预算改善模型根本不用的 coefficient directions。

4. **没有 V/O consequence。**
   attention pattern 变化不等于 residual 功能变化。

Barbero 等发现高频常被用于强位置模式，而低频大量承载近似 semantic matching；后续工作也发现 positional/symbolic head 行为与频率使用有强对应关系。([arXiv][4])
LeRoPE 学到的慢频通常被进一步推向近零，而不是简单删除，说明“慢频结构冗余”不等于“这些维度功能无用”。([arXiv][5])

因此更准确的结论是：

> structural geometry 是一个必要的 code diagnostic，但它不是 LM objective 的一致 surrogate。

---

## 2. 为什么 phase-chord 最接近 win-win

你的 phase-chord 对单个频率使用

[
m(\omega)
=========

\mathbb E_{D_{\rm att}}
[1-\cos(\omega D)].
]

令距离分布的 characteristic function 为

[
\varphi_D(\omega)
=================

\mathbb E[e^{i\omega D}],
]

则

[
m(\omega)
=========

1-\operatorname{Re}\varphi_D(\omega),
\tag{M}
]

也即

[
2m(\omega)
==========

\mathbb E
\left|
e^{i\omega D}-1
\right|^2.
]

它不是抽象的 Gram rank，而是：

> **某个频率在模型实际使用的距离上产生了多少 phase separation。**

这有三个优势。

### 第一，它同时惩罚两个极端

* 太慢：在实际 dependency distances 上几乎不转，缺乏位置辨别；
* 太快：在相关距离上多次 wrap，平均 contrast 可能下降或发生 alias。

### 第二，它使用了经验 dependency prior

Wu 等近期工作的结论正是：有用频率由数据诱导的 dependency width 决定，而不是只由 (L_{\rm train}) 或 support 决定。([arXiv][3])

### 第三，它更接近真实 logit 的第一变分

它仍然忽略 Q/K coefficient covariance 和 V/O consequence，但比 uniform pair Gram 更接近“实际相位变化在被使用距离上的效用”。

因此，我目前的机制排序是：

1. **最可能：phase-chord 成功主要来自 dependency-scale matching。**
2. **其次：chord kernel 本身比 static subspace rank 更接近 attention operator 的局部效用。**
3. **尚未排除：参与 profile 构造的 seed 带来的方法选择偏差。**

一个独立 seed 也在所有 OOD 长度改善，使“完全是同 seed 过拟合”不太可信；但两 seed 还不能区分“跨模型稳定 prior”与“该 corpus/architecture 的偶然 prior”。

---

## 3. phase-chord 仍缺什么

当前 (m(\omega)) 是一个**单频 diagonal utility**。它没有处理：

[
\mathbb E
\left[
e^{i\omega D}
\overline{e^{i\nu D}}
\right]
=======

\varphi_D(\omega-\nu),
]

即不同频率之间在实际 dependency distribution 上的冗余；也没有处理式 (A) 中的 V/O 后果。

因此下一步不应继续修改 (m^{1/3}) 的指数或平滑细节，而应把它升级为：

> **usage-conditioned、output-consequence-weighted 的离散 frame/codebook 设计。**

---

## 4. 1024/2048 或 4×/8× 非单调反转意味着什么

这类反转不是反常现象，而是有限 trigonometric code 的基本性质。

由式 (L)，某个 shell 上的表现可能因为若干

[
(\omega_k-\omega_\ell)d
]

同时接近 (2\pi\mathbb Z) 而出现高 sidelobe；更长后这些项又可能重新 dephase。因此：

* 2× 好；
* 4× 差；
* 8× 再好；

在纯 operator 层面完全可能。

但 LM 长度变化还同时改变：

* 可见 key 数量和 softmax denominator；
* attention entropy；
* retrieval/streaming head 激活；
* 文档中的实际 dependency-distance 分布；
* 多层路由组合。

因此 4× reversal 有三个竞争解释：

| 机制                        | 可证伪预测                                                         |
| ------------------------- | ------------------------------------------------------------- |
| operator resonance        | 在训练前的 usage-weighted ambiguity curve 上即可看到相同长度峰值；换语料后峰值位置基本不动 |
| dependency-prior mismatch | 更换 dependency-width 分布后，反转位置随之移动                              |
| learned co-adaptation     | 训练早期没有反转，随后与 band norm、head specialization 一同出现               |

只在 (1×/2×/4×/8×) 四个点读 NLL 无法区分。需要的是**密集长度曲线与 mechanism trace**，不是再随机扫几个 (\tau)。

---

# 五、最有希望的从头训练方向

# 方向 A：Usage-Conditioned Discrete Spectral Portfolio

中文可称为：

> **使用条件化的离散频谱组合。**

核心不是一张共享的单调 density，而是 (J=2\sim4) 个固定小 codebook，覆盖不同 dependency/functional roles。

## 1. 为什么是 portfolio，而不是 unrestricted per-head table

现有文献并不支持一个简单结论：

* LeRoPE 在其从头训练设置中发现全层全 head 共享频率优于更细粒度的 per-head/per-layer learning。([arXiv][5])
* AdaRoPE 在长上下文适配设置中则发现强制共享频率表会退化，并观察到少量关键 heads 需要不同的频段和 attention scaling。([arXiv][6])
* retrieval-head 工作表明长程能力集中在稀疏、因果关键的 heads；DuoAttention 也观察到 retrieval 与 streaming heads 的上下文需求不同。([arXiv][7])

这组证据更支持一个中间方案：

* 不用所有 head 共享一个表；
* 也不让每个 head 自由学习 (K) 个连续频率；
* 而是学习或预设少量 **spectral roles/codebooks**。

这能避免 per-head 高维过拟合，同时允许功能分化。

---

## 2. 构造 dependency measure

对 head (h)、query (i)、key (j)，attention output 对 logit (z_{hij}) 的导数为：

[
\frac{\partial o_{hi}}{\partial z_{hij}}
========================================

p_{hij}
(v_{hj}-o_{hi}).
]

可用其输出后果定义低成本 diagonal weight：

[
c_{hij}
=======

p_{hij}^2
\left|
F_i^{1/2}
W_{O,h}
(v_{hj}-o_{hi})
\right|^2.
\tag{N}
]

然后构造 consequence-weighted dependency measure：

[
P_h^{\rm use}(d)
\propto
\sum_{i,j:,i-j=d}
c_{hij}.
\tag{O}
]

至少应比较三个 prior：

1. raw attention-distance (P_{\rm att})；
2. output-consequence (P_{\rm out})；
3. causal/gradient intervention 得到的 (P_{\rm causal})。

profile 必须来自独立 pilot/checkpoint/文档，不从候选表的测试 loss 反向选择。

---

## 3. 直接设计有限离散表

在 dense candidate grid (\mathcal C) 上直接选 (K) 个频率，而不是先求连续 density 再 midpoint quantize。

对 codebook (j)，可以优化：

[
\begin{aligned}
\max_{\Omega_j\subset\mathcal C,\ |\Omega_j|=K_j}
\quad&
\underbrace{
\log\det
\left(
G_{P_j^{\rm use}}(\Omega_j)+\epsilon I
\right)
}*{\text{usage-conditioned observability / nonredundancy}}
\
&+
\alpha
\underbrace{
\sum*{\omega\in\Omega_j}
\mathbb E_{d\sim P_j^{\rm use}}
[1-\cos(\omega d)]
}*{\text{phase-chord utility}}
\
&-
\beta
\underbrace{
\sum_m\pi_m
\operatorname{CVaR}*{d\in S_m}
A_{\Omega_j,w}(d)
}*{\text{anytime recurrence / sidelobe risk}}
\
&-
\gamma
\underbrace{
R*{\rm cross}(\Omega_j,\Omega_{-j})
}_{\text{portfolio diversity}}.
\end{aligned}
\tag{P}
]

约束包括：

[
O_{\rm train}(\Omega_j)\ge \eta,
\qquad
I_{\rm local}(\Omega_j)\in[I_{\min},I_{\max}].
]

实现上不需要复杂连续优化：

* 先在 512–4096 个 log-frequency candidates 上计算 kernel；
* 用 greedy logdet、pivoted Cholesky 或 swap-search 选点；
* 不读取候选模型的 LM loss；
* 完全可以得到非单调、多峰、带空洞的频率表。

论文附录已经指出连续 density 到有限 (K) midpoint table 存在显式的 (K^{-1}) 或 (K^{-2}) transport/distortion gap；在 (K=16/32) 时直接离散设计尤其有意义。

---

## 4. portfolio 中应允许一个 semantic/NoPE role

慢频 pair 的 structural collapse 不等于功能浪费。文献与现有实验共同提示，部分慢频可能被模型当成近似不旋转的 semantic matching subspace。([arXiv][4])

因此 portfolio 可以包含：

1. **local-resolution codebook**：偏高频，负责精细顺序；
2. **dependency-matched codebook**：phase-chord 主导；
3. **long-range low-sidelobe codebook**：强调 recurrence survival；
4. **semantic codebook**：小块 NoPE 或近零频率 subspace。

这不是预设 partial RoPE 一定正确。需要把以下两者作为竞争条件：

* 全部正频率、完全 rotary；
* 显式保留少量 unrotated/near-zero subspace。

RoPE–NoPE hybrid、CoPE、RoPE-ID 和 DroPE 等近期结果都说明“所有 head、所有维度永久使用同一 rotary law”并非不可挑战的默认前提。([arXiv][8])

---

## 5. coding/frame 理论怎样真正帮上忙

相邻数学领域提供的是设计语言，不是 LM 结论：

* Welch bound 和 frame potential说明有限维中不能同时让所有 codeword 近似正交，必须选择平均、最大或加权 coherence 风险。([arXiv][9])
* minimum-redundancy arrays / Golomb-type constructions通过减少重复 difference 来压制 sidelobes。([ADS 旨在促进天文学与物理学的开放获取。][10])
* multi-frequency ranging 中，非均匀或随机频率集合能够延长有限时间的 unambiguous range，而不必仅靠减小统一频率步长。([arXiv][11])

对 RoPE 最直接的移植是优化式 (L) 中的 frequency-difference multiset，但必须加上：

* training observability；
* dependency prior；
* head usage；
* semantic subspace；

否则会重演 phase-isotropy 的失败。

---

# 六、成熟模型最有希望的方向

# 方向 B：Dual-Basis Spectral Bridge + Tangent Transport

直接 morph 已经活跃的频率，会立即让大量长距离 phase 发生 (d,\delta\omega) 级变化。即使 (\delta\omega) 很小，长 (d) 下也不是小扰动。

更稳妥的办法不是移动旧坐标，而是**临时引入新坐标，再逐步转移 learned coefficients**。

## 1. 双基底桥

适配期间使用：

[
z_t(d)
======

\operatorname{Re}
\sum_k
\left[
a_k^{(0)}(t)e^{i\omega_k^{(0)}d}
+
a_k^{(1)}(t)e^{i\omega_k^{(1)}d}
\right].
\tag{Q}
]

初始化：

[
a_k^{(0)}(0)=a_k^{\rm pretrained},
\qquad
a_k^{(1)}(0)=0.
]

结束：

[
a_k^{(0)}(1)=0,
\qquad
a_k^{(1)}(1)=a_k^{\rm migrated}.
]

实现上：

* 临时增加一个 new-table Q/K branch；
* native branch 初始 gate 为 1，new branch 为 0；
* V/O 先共享；
* 逐渐降低 native gate、提高 new gate；
* 最后删除 native branch，恢复标准单表 architecture。

这在 (t=0) 精确保持原功能，不会一开始就产生 hard-swap shock。它不违反 transplant obstruction，因为：

* theorem 禁止的是固定权重下的精确单步线性吸收；
* 这里临时扩大表示字典，并实际训练权重；
* 只要求在有限训练分布上近似运输，而非对所有连续 (\Delta) 精确相等。

这与 Net2Net 的 function-preserving expansion 思想相近，但这里最终的非共轭频率表示仍需通过训练完成迁移。([arXiv][12])

---

## 2. tangent-space 解释

每一步可写成：

[
\dot\theta(t)
=============

\arg\min_v
\mathbb E_{\rm native}
\left[
\left|
\frac{\partial f}{\partial t}
+
J_\theta f,v
\right|_F^2
\right]
+
\lambda|v|^2.
\tag{R}
]

其中 (\partial f/\partial t) 是频率/gate 变化造成的功能漂移，(J_\theta f,v) 是 Q/K/O 等参数更新对它的局部补偿。

SGD 不必显式求 pseudoinverse；关键是用正确的 trust-region loss：

[
\begin{aligned}
\mathcal L_{\rm bridge}
=&
\lambda_{\rm LM}
D_{\rm KL}
\bigl(
f_{\rm native}\Vert f_t
\bigr)
\
&+
\lambda_{\rm att}
D_{\rm KL}
\bigl(
p_{\rm native}\Vert p_t
\bigr)
\
&+
\lambda_{\rm out}
\left|
W_OV^\top(p_t-p_{\rm native})
\right|^2
\
&+
\lambda_{\rm step}
D_{\rm KL}
\bigl(
f_{t-\delta t}\Vert f_t
\bigr)
\
&+
\lambda_{\rm long}
\mathcal L_{\rm sparse\ phase}.
\end{aligned}
\tag{S}
]

gate 或频率步长不按固定 schedule 前进，而是在 native attention-output KL 低于阈值时才推进。

---

## 3. Q/K、O、V 的合理解冻顺序

### [论文事实]

论文自己的 OLMo 协议表明，只继续 Q/K 已经能在 4K 基本保持 QA，同时把 8K/16K 能力迁移到 EVQ-Cosh arm。

### [新假设]

建议顺序是：

1. **先 Q/K**：直接补偿 rotary logit geometry；
2. **再 O**：若 routing 已接近 teacher，但 residual 写入仍失配，O 可以重新组合 head outputs；
3. **最后才考虑 V**：V 改变被搬运的信息内容，遗忘风险最大。

但这不是先验真理，应通过 tangent residual 判断：

[
r_{\mathcal M}
==============

\min_{v\in\mathcal M}
\mathbb E
\left[
|
\partial_tf+J_{\mathcal M}f,v
|^2
\right],
]

依次比较

[
\mathcal M\in
{QK,\ QKO,\ QKVO}.
]

若 Q/K 的 residual 已很小，就没有理由先动 V；若只有 QKVO 才能压低 residual，则“RoPE 只作用于 Q/K，所以只需 Q/K”这一假设被否定。

---

## 4. sparse long-phase exposure，而不是只做 native distillation

只保持 native 功能可能得到一个“在新表下复制旧模型”的学生，却没有学会新 phase states。

可以在短物理序列中加入：

* skipped/offset position IDs；
* suffix phase perturbation；
* terminal anchoring；
* dyadic shell 采样的相对位置。

已有工作表明，RoPE-index perturbation 配合 self-distillation 能降低位置脆弱性；EndPrompt 也显示短物理序列可以提供远距离 phase supervision。([arXiv][13])

在 short-context restoration 方面，LongReD 将退化归因于 hidden/attention drift 和 continued-pretraining forgetting；LinearARD 则直接蒸馏 Q/Q、K/K、V/V relation 来修复 RoPE scaling 造成的 attention drift。([ACL Anthology][14])

这些工作支持 distillation，但尚未解决：

* 频率坐标的渐进路径；
* 双基底过渡；
* 同一最终 table 下，hard swap 与 smooth transport 的因果区别。

这正是 spectral bridge 的新贡献空间。

---

# 七、最值得先做的方向与最小判别实验

## 优先级裁决

**科学优先级最高的是方向 A：usage-conditioned discrete portfolio。**

原因不是它最复杂，而是它直接回答当前所有核心矛盾：

* structural geometry 为什么失效；
* phase-chord 为什么成功；
* profile 是否只是 selection bias；
* head heterogeneity 是否是真正需要的自由度；
* 非单调长度响应能否被 operator ambiguity 预测。

在开始新训练前，先做一个几乎零成本的判别实验。

---

## 实验 0：现有 artefact 上的 metric shootout

不训练新模型，不搜索新表。对当前已有的：

* FMRoPE；
* phase-isotropy；
* min-eigen；
* phase-chord；

计算以下四类预测量：

1. uniform structural Gram；
2. raw-attention phase-chord；
3. consequence-weighted chord，式 (N)–(O)；
4. full output-Fisher shell risk，式 (A)、(I)、(K)。

要求：

* profile、metric 构造文档与验证文档完全分离；
* 对参与 profile 构造的 seed 不计独立证据；
* 在 (1×) 到 (8×) 间做密集长度曲线，而非只看四个点；
* **事前预测**各表在每个长度的相对排序与符号；
* 特别要求预测 phase-isotropy 的 4× 退化和 8× 恢复。

判决规则：

| 结果                                    | 科学结论                                        |
| ------------------------------------- | ------------------------------------------- |
| uniform Gram 已能预测                     | 之前 metric 实现或 weighting 有问题，可暂缓复杂化          |
| attention-chord 能预测，V/O weighting 无增益 | 成功主要来自 dependency prior + phase operator    |
| 只有 consequence metric 能预测             | V/O 与 downstream consequence 是缺失主因          |
| 所有 frozen/pilot metric 均失败            | 静态表设计不足，必须研究训练动态或 joint learned frequencies |
| 只在参与 profile 的 seed 上预测成功             | phase-chord 路线按 selection bias 关闭           |

这个实验比再训练三五个 seed 的信息量高得多。

---

## 实验 1：profile-source × dependency-distribution 交叉实验

构造两个 token marginals、entropy、训练长度完全一致，但 dependency widths 明显不同的训练任务：

[
P_A:\ W\approx W_A,
\qquad
P_B:\ W\approx W_B,
\qquad
W_B\gg W_A.
]

从独立 pilot 得到 profile (D_A,D_B)，构造表 (\Omega_A,\Omega_B)。

然后做完整的 (2\times2)：

| 训练 dependency |        表 A |        表 B |
| ------------- | ---------: | ---------: |
| (P_A)         |    matched | mismatched |
| (P_B)         | mismatched |    matched |

再增加一个两 codebook portfolio ({\Omega_A,\Omega_B})，在 (P_A/P_B) mixture 上训练。

判别：

* **两个对角 matched 均胜出**：phase-chord 主要来自可推广的 dependency prior；
* **同一张表在 A/B 都胜出**：存在更强的 operator-universal 结构；
* **portfolio 只在 mixture 上胜出，并出现 head-role 分化**：单共享表不是正确对象；
* **cross-fitted matched 不胜，仅 in-source 胜**：方法选择偏差，关闭 phase-chord；
* **表差异训练后消失**：模型优化会吸收该 profile，allocation 只影响优化速度而非最终解。

这不是“再跑参数”，而是用互斥预测区分三个机制。

---

## 实验 2：mature retrofit 的最小路径实验

固定同一张目标表，不再搜索 allocation。比较：

1. hard swap + Q/K adaptation；
2. direct frequency homotopy + Q/K adaptation；
3. dual-basis bridge + Q/K adaptation；
4. dual-basis bridge + attention-output distillation。

所有条件保持：

* 同样数据；
* 同样更新步数；
* 同样 trainable rank/参数量；
* 同样 long-phase exposure。

主要判据不是单独的 long NLL，而是二维 Pareto：

[
(\Delta {\rm native\ capability},
\ \Delta {\rm long\ capability}).
]

结果解释：

* bridge 在相同 long gain 下显著保住 1×：现有 trade-off 主要是 coordinate shock；
* 所有路径最终落到同一 Pareto 点：目标表存在内在窗口内代价；
* Q/K 路径失败、加 O 后恢复：head-output recomposition 是瓶颈；
* 必须加 V 才恢复：frequency migration 改变了 message semantics，不只是 routing；
* DroPE 或 native/long routing 明显支配全部单表路径：单一最终表本身可能是错误 retrofit 目标。

---

# 八、哪些现有路线应该停止

## 应停止作为主线

### 1. 继续证明 allocation 是第三轴

已经由三种子 exact-range、crossing、transplant obstruction 完成。论文的固定 support 三种子结果也明确显示所有 OOD 长度同向。

### 2. 把 uniform structural pair Gram、effective rank 或 min-eigenvalue 当作 LM optimizer

保留为诊断工具，不再作为方法生成器。它们缺少 usage 与 V/O consequence，且已有反例。

### 3. 在成熟模型上对少量文档直接优化高维 (z)

它同时具有：

* checkpoint-specific；
* document-specific；
* high-dimensional；
* non-convex；
* coordinate-shock-confounded；

五个问题。当前 held-out 失败足以关闭，除非未来引入强结构化低维先验和真正独立 meta-training。

### 4. 继续雕刻 fine-grained profile

论文的 matched exponential 与 Cosh 在 factorial 中几乎无法区分。
成熟 checkpoint 上 derived profile 与粗 ramp 的差异也未被识别。

在粗结构尚未识别前，微调局部曲率、平滑度和单点 density 是低收益工作。

### 5. 用 frozen hard swap 的 1× 损失证明 allocation 存在必然 waterbed trade-off

这是不成立的归因。hard swap 混合了：

* table 本身的功能差异；
* 权重—表失配；
* 未适配的新 phase states。

只有在相同最终表下比较不同 migration paths，才能识别 intrinsic trade-off。

### 6. 把单调一参数 density 当作最终设计空间

EVQ-Cosh 应保留为：

* 解析 baseline；
* 初始化；
* causal control；

但不应继续被当作下一阶段理论中心。现有结果更支持非单调、多峰、head/subspace portfolio。

### 7. 在 target-free 主线中混入 target-aware support retargeting

LongRoPE、LongRoPE2 的核心是面向给定目标长度的非均匀 scaling/search，并配合短窗恢复或 mixed-window training；LongRoPE2 甚至使用 target-length、needle-driven perplexity 引导搜索。([Proceedings of Machine Learning Research][15])

这些是有效的部署技术，但应单独标为：

[
\text{target-aware transport},
]

而不是 target-free table design。

---

## 不应停止

* phase-chord：当前最高优先级，但必须 cross-fit；
* coarse ramp：作为低复杂度 control；
* native/long session routing：作为工程 Pareto upper bound；
* EVQ-Cosh：作为 allocation-axis analytic baseline；
* fixed-support control：继续用于因果识别，但不应成为最终设计的永久约束。

固定 support 是发现变量时的实验控制，不是最优部署表必须遵守的自然定律。

---

# 九、如果所有候选都失败，仍能学到什么

## 1. 可能不存在有意义的 universal single table

若 dependency-profile crossover 显示不同数据稳定偏好不同表，且 portfolio 也不能共享，那么可以形成一个更强的负结论：

> 对足够宽的 dependency class，不存在既 target-free、又全 head 共享、又统一 Pareto 改善的有限 RoPE 表。

此时合理解是：

* per-role tables；
* runtime routing；
* dynamic positional operator；
* 或去除部分 RoPE。

## 2. static encoding 可能不是主要瓶颈

若所有 usage-conditioned code metric 都不能预测训练结果，说明瓶颈主要位于：

* learned coefficient dynamics；
* head formation；
* layer composition；
* attention-to-output circuit；

而不在编码本身。

这会把研究问题从“frequency table design”升级为“frequency table × optimization dynamics co-design”。

## 3. mature model 可能无法收敛到单一新坐标系

若 spectral bridge 也无法在保持 1× 的同时迁移到新表，可能意味着旧坐标已被深度分布式地写入多层 circuit。

此时 persistent dual coordinates、session routing，甚至 DroPE 式 recalibration，可能不是工程妥协，而是更正确的表示形式。

## 4. 仍可得到明确的理论结果

即使没有新方法胜出，仍可建立：

* 有限 RoPE 的 recurrence/no-uniform-optimum 定理；
* structural metric 不足以一致预测 LM 的反例族；
* dependency prior 与最优频率之间的 no-free-lunch；
* mature table migration 的 path-dependence；
* target-free 风险的正式定义与 Pareto frontier。

这些比继续扩展 Cosh surrogate 的附属定理更可能改变领域理解。

---

# 十、极简研究导航

建议只维护四个文件：

1. `FACTS.md`：已建立事实；
2. `CLOSED.md`：已关闭路线及失败模式；
3. `OPEN.md`：互斥机制与可证伪预测；
4. `RUNS.csv`：每个实验改变了哪个判断。

每个 claim 只允许以下格式：

```yaml
id: A-017
object: allocation | support | usage | exposure | assignment | migration
claim: 一句话陈述
status: established | supported | falsified | open
scope: 模型、K、support、训练长度、数据、是否成熟模型
evidence: artefact / table / commit / seeds
competing_mechanism: 最强替代解释
discriminator: 哪个最小实验能区分
aliases: 过去使用过的其他名称
reopen_if: 什么新证据足以重新开放
```

当前可以直接初始化为：

| ID    | 状态                      | 内容                                               |
| ----- | ----------------------- | ------------------------------------------------ |
| F-001 | established             | fixed support 下 allocation 独立影响训练结果              |
| F-002 | established             | 权重与训练期频率表共适应                                     |
| F-003 | established             | unequal frequency multisets 不能被固定可逆 Q/K map 精确吸收 |
| C-001 | falsified as optimizer  | uniform structural pair geometry 稳定预测 LM         |
| C-002 | falsified               | 少量文档上直接优化 mature high-dimensional (z) 可泛化        |
| C-003 | unresolved/low priority | fine derived profile 优于 coarse ramp              |
| O-001 | open, highest priority  | cross-fitted phase-chord 是否泛化                    |
| O-002 | open                    | output-consequence metric 是否解释 4× reversal       |
| O-003 | open                    | 小型 spectral portfolio 是否优于共享表                    |
| O-004 | open                    | spectral bridge 是否消除 hard-swap 1× shock          |

再增加一条硬规则：

> **任何新方法在立项前，必须说明它改变的是 support、allocation、usage metric、phase exposure、head/subspace assignment 还是 migration path；若只是给已关闭对象换名，不进入实验队列。**

---

# 最终路线裁决

最值得先做的不是第三张 analytic density，而是：

1. **用现有 artefact 验证 output-consequence-weighted phase metric 能否预测当前的长度特异性反转；**
2. **若通过，做 dependency-profile crossover，并把 phase-chord 升级为离散、多峰、小型 head-codebook portfolio；**
3. **成熟模型另开 spectral bridge 线，用同一最终表直接判定 hard-swap shock 与 intrinsic trade-off。**

一句话概括：

> **EVQ-Cosh 发现了频谱预算；下一步应研究的不是预算曲线本身，而是“谁在花这笔预算、训练是否看见了它、它在未知尺度上何时发生歧义，以及如何把已经按旧预算记账的模型迁移过来”。**

[1]: https://arxiv.org/abs/2606.11275 "https://arxiv.org/abs/2606.11275"
[2]: https://arxiv.org/pdf/2007.05763 "https://arxiv.org/pdf/2007.05763"
[3]: https://arxiv.org/html/2607.07678v1 "https://arxiv.org/html/2607.07678v1"
[4]: https://arxiv.org/abs/2410.06205 "https://arxiv.org/abs/2410.06205"
[5]: https://arxiv.org/html/2607.10134 "LeRoPE: Learnable RoPE Frequencies Improve Language Modeling"
[6]: https://arxiv.org/html/2607.19363v1 "https://arxiv.org/html/2607.19363v1"
[7]: https://arxiv.org/abs/2404.15574 "https://arxiv.org/abs/2404.15574"
[8]: https://arxiv.org/abs/2501.18795 "https://arxiv.org/abs/2501.18795"
[9]: https://arxiv.org/pdf/1801.04548 "https://arxiv.org/pdf/1801.04548"
[10]: https://ui.adsabs.harvard.edu/abs/1968ITAP...16..172M/abstract "https://ui.adsabs.harvard.edu/abs/1968ITAP...16..172M/abstract"
[11]: https://arxiv.org/pdf/1702.05616 "https://arxiv.org/pdf/1702.05616"
[12]: https://arxiv.org/abs/1511.05641 "https://arxiv.org/abs/1511.05641"
[13]: https://arxiv.org/abs/2604.14339 "https://arxiv.org/abs/2604.14339"
[14]: https://aclanthology.org/2025.acl-long.524/ "https://aclanthology.org/2025.acl-long.524/"
[15]: https://proceedings.mlr.press/v235/ding24i.html "LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens"


# 独立裁决

当前路线的瓶颈已经不再是“怎样把有限频率排得更均匀”，也不是再寻找一个更漂亮的单调密度。**真正缺失的对象是：模型实际使用的、带有 Q/K 内容系数和 softmax 观测度量的有限相位代码。**

更准确地说，下一阶段的核心科学问题应改写为：

> **在有限旋转维度下，怎样让模型实际使用的 positional subspace 在未知部署尺度上保持稳定的低尾分离能力，同时显式保留模型需要的 position-invariant semantic subspace；对于成熟模型，又怎样沿一条低功能扰动路径，把旧的坐标系运输到新的相位代码？**

这与“优化一张全局共享、单调、全 RoPE 频率表”不是同一个问题。后者可能只是前者的一个受限特例，而且很可能已经接近方法上限。

论文已经足够清楚地建立了 allocation 轴，以及表—权重共适应。尤其是 PDF 第 6 页 Table 1：静态有效秩从 4.57 提高到 12.54 时，交叉表 PPL 却从 7.14 恶化到 76.20。这已经排除了“更好的静态位置子空间几何必然带来更好的 LM”这一解释。 同页及其后的定理还证明，不同频率多重集不能由固定、位置无关的可逆 Q/K 变换精确补偿。

我的总体判断是：

1. **从头训练最值得推进的是 usage-conditioned、role-conditioned 的有限相位代码，而不是继续调一个全局密度。**
2. **phase-chord 很可能抓住了正确对象的一阶近似，但目前无法区分“精确圆周算子结构”“attention-distance prior”与“profile 选择偏差”。**
3. **成熟模型 retrofit 必须成为独立方法线。它优化的不是终点表，而是从原生坐标到目标坐标的可达路径。**
4. **严格意义上、不声明任何长度风险分布的 target-free 全局最优表不存在一个良定义的标量目标。只能讨论预先声明风险类下的 minimax/Pareto 设计。**

---

# 一、此前尚未正确表述的核心问题

## 1. 名义频谱预算不等于模型实际使用的频谱预算

对一个 attention head，把每个二维 pair 写成复数形式，RoPE logit 可以写成

[
s_{hij}(\Omega)
===============

\frac{1}{\sqrt d}
\operatorname{Re}
\sum_{k=1}^{K}
a_{hij,k},e^{i\omega_{h,k}\Delta_{ij}},
]

其中 (a_{hij,k}) 由未旋转的 Q/K 内容、相对方向和幅值共同决定。

频率表本身只给出相位轨迹

[
\Psi_{\Omega}(\Delta)
=====================

\big(
e^{i\omega_1\Delta},\ldots,e^{i\omega_K\Delta}
\big)
\in \mathbb T^K.
]

但模型并不是用欧氏度量直接读取 (\Psi_\Omega)。它通过以下对象观察这个代码：

* Q/K pair 的幅值与相位；
* 某个 query 实际竞争的 key 集合；
* softmax Jacobian，它消除 query-wise 常数方向；
* V/O 和下游梯度，它们决定一次注意力重排是否真正影响 loss；
* head 与 layer 的功能角色。

因此，真正有意义的“有效位置维度”不是论文中的静态 block-whitened rank，而应是一个 **usage-weighted effective rank**。

论文的 block whitening 本来就有意去除了各通道的能量和条件数，只保留子空间重合关系。 这适合证明“几何冗余”，但恰好丢掉了 LM 最关心的训练信号幅度。一个经过 whitening 后很独立的慢频方向，原始相位运动仍可能只有 (O((\omega L)^2))，训练几乎看不见。

## 2. “慢频冗余”包含两种完全不同的东西

当前叙事容易把以下两类通道混在一起：

1. **未充分观测的 positional channel**
   模型想用它编码距离，但训练窗口只覆盖一小段相位弧，系数估计不稳定，OOD 后发生符号翻转或未见相位。

2. **有意近似 position-invariant 的 semantic channel**
   模型并不希望它对距离敏感；慢旋转只是近似 NoPE，用于内容匹配。

第一类应该减少或移动；第二类未必是浪费。将两者都视为“低频坍缩”并重新分配，会在静态几何上变好，却破坏语义 QK 通道。

在 Gemma 7B 上的 mechanistic 分析发现，高频更常用于稳健的位置模式，而模型整体偏好使用低频，作者推测低频承载语义信息。([arXiv][1]) Retrieval-head 研究进一步表明，真正承担长程检索的 head 很稀疏，通常少于 5%，而且这些 head 在短上下文预训练模型中已经存在。([arXiv][2])

因此，最核心的未决变量不是 (\rho(\phi))，而是：

[
\mathbb R^{d_h}
===============

S_h^{\rm invariant}
\oplus
S_h^{\rm positional},
]

以及 (S_h^{\rm positional}) 内部应该使用什么有限相位代码。

这也是为什么 **collision 并不总是坏事**：对 positional/retrieval head，碰撞是坏的；对 symbolic head，距离不变性本来就是功能要求。

---

# 二、“target-free”怎样才数学上良定义

## 1. 无穷长度上的统一非碰撞保证不可能

令

[
\alpha_k=\frac{\omega_k}{2\pi},
\qquad
\Phi_\Omega(q)
==============

\left(
e^{2\pi iq\alpha_1},\ldots,e^{2\pi iq\alpha_K}
\right).
]

Dirichlet 同时逼近定理给出：对任意整数 (Q)，存在

[
1\le q\le Q^K
]

使得

[
\max_k|q\alpha_k|_{\mathbb R/\mathbb Z}
\le \frac1Q.
]

该形式可直接见相应数学文献中的式 (1.2)。([arXiv][3]) 因而

[
\left|
\Phi_\Omega(q)-\Phi_\Omega(0)
\right|_2
\le
\frac{2\pi\sqrt K}{Q}.
]

令 (T\approx Q^K)，得到一个直接推论：

[
\min_{1\le q\le T}
\left|
\Phi_\Omega(q)-\Phi_\Omega(0)
\right|_2
\lesssim
2\pi\sqrt K,T^{-1/K}.
]

所以：

* 频率有理相关时会出现精确周期；
* 一般频率也必然出现任意近的联合 recurrence；
* 任何有限表在无穷长度上都不存在正的统一分离 margin。

这只否定 **位置代码的全局非碰撞保证**，不等价于证明某模型的 LM loss 不可能在所有测试点更好。但它足以说明：“对所有未知长度都稳定优于几何表”不能用全局 worst-case separation 来定义。

而且，若模型实际只使用 (r\ll K) 个频率组合，实际 recurrence 更接近一个 (r) 维问题。这个结论目前是推断，不是已证明定理，但它与 dominant-band 现象及成熟模型的低维 profile 可辨识性高度一致。

## 2. 无穷 log-length 上也不存在均匀概率分布

所谓“所有长度同等重要”同样不成立。对

[
m=\log_2(L/L_{\rm train})
]

不存在可归一化的无穷区间均匀分布。因此 target-free 只能有三种诚实定义：

### A. Operator-only target-free

[
\Omega=f(K,\omega_{\min},\omega_{\max},L_{\rm train})
]

完全不使用数据、模型或 target 信息。

这是最强但也最不现实的定义。最新工作表明，模型学习使用的 RoPE 频率随数据依赖宽度变化，长程泛化还取决于训练依赖在更长尺度上怎样延伸。([arXiv][4]) 因而只依赖 (K) 和 (L_{\rm train}) 的表不可能对任意数据分布都最优。

### B. Training-prior-conditioned、deployment-target-agnostic

允许使用：

* 训练语料；
* 训练窗口内的 attention/QK/Fisher 统计；
* 独立 donor 模型；
* 预先声明的未知长度风险分布。

禁止使用：

* 具体 (L_{\rm target})；
* 测试长度 LM loss；
* 根据远端测试结果选择 profile。

phase-chord 当前属于这一类，而不是严格的 operator-only analytic table。

### C. Target-aware

直接使用某个 (L_{\rm target})、伸长倍率或远端 loss。PI、YaRN、LongRoPE 类方法大多属于这一类问题，不应与 target-free allocation 混称。论文自己的 exact-range 实验也已经显示，改变 sampled support 以匹配 target 后，allocation 排名会反转。

## 3. 推荐的 target-free 风险定义

令 dyadic annulus 为

[
\mathcal A_m
============

[2^mL_{\rm train},,2^{m+1}L_{\rm train}),
\qquad m=0,1,\ldots
]

用一个可归一化的尺度 hazard：

[
\nu_m(\beta)
============

(1-2^{-\beta})2^{-\beta m},
\qquad \beta>0.
]

不要固定一个 (\beta)，而是声明一个不确定性集合

[
\beta\in[\beta_-,\beta_+].
]

同时，对长程 dependency profile 使用 ambiguity set：一部分依赖随尺度 dilation，另一部分继续保持局部距离，不预设所有自然语言关系严格自相似。

目标不应优化平均 collision，而应优化 **lower-tail separation**，因为 4× 局部崩坏正是平均指标遗漏的对象：

[
\mathcal R_{\rm out}(\Omega)
============================

\sup_{\nu\in\mathcal V}
\sum_{m=0}^{\infty}
\nu_m
\operatorname{CVaR}*{\alpha}
\left[
-D*\Omega(q)
,:,
q\in\mathcal A_m
\right].
]

再加上训练窗口风险、局部分辨率和 phase-OOD 风险，最终只能谈

[
\big(
\mathcal R_{\rm in},
\mathcal R_{\rm collision},
\mathcal R_{\rm phaseOOD},
-\mathcal R_{\rm local}
\big)
]

的 Pareto frontier，或者在一个明确的 (\mathcal R_{\rm in}\le\epsilon) 约束下做 minimax。不存在不声明这些权重或约束的“普适最优表”。

---

# 三、把五类机制放进同一个可证伪框架

## 1. Usage-weighted phase kernel

定义

[
v_\Omega(q)
===========

\left(
e^{i\omega_1q}-1,\ldots,e^{i\omega_Kq}-1
\right).
]

令

[
\Sigma_g
========

\mathbb E[
a_{g}a_{g}^{*}
]
]

表示某个 head group 的 Q/K complex coefficient 协方差，也可以进一步用 attention、softmax Fisher 或下游梯度加权。则一个实际使用的相位分离度为

[
D_g(q;\Omega)
=============

v_\Omega(q)^*
\Sigma_g
v_\Omega(q).
]

它有几个退化版本：

* (\Sigma=I)、均匀距离：接近纯结构几何；
* (\Sigma) 取对角且只使用 attention-distance prior：接近 phase-chord；
* 完整 (\Sigma) 和 softmax/downstream 权重：接近真实 task-conditioned operator。

这个对象直接包含 cross-frequency cancellation。一个 pairwise Gram 指标可以整体改善，但 (D_g(1024)) 的几个高权重 band 仍可能相消，而 (D_g(2048)) 又重新展开。

## 2. 精确的 coordinate-shock 局部度量

令

[
x_k=-\log\omega_k.
]

单个 pair 的 logit 项为

[
f_k(\Delta)
===========

C_k\cos(\omega_k\Delta)
+
D_k\sin(\omega_k\Delta).
]

则

[
\frac{\partial f_k}{\partial x_k}
=================================

\omega_k\Delta
\left[
C_k\sin(\omega_k\Delta)
-----------------------

D_k\cos(\omega_k\Delta)
\right].
]

对一个 query，把所有 key-logit 对 (x) 的 Jacobian 记为 (B_i)，则

[
\mathrm{KL}
\big(
p_i(x),|,p_i(x+\delta x)
\big)
=====

\frac12
\delta x^\top
B_i^\top
J_{\rm sm}(p_i)
B_i
\delta x
+
O(|\delta x|^3).
]

因此定义

[
F_x
===

\mathbb E_i
\left[
B_i^\top
J_{\rm sm}(p_i)
B_i
\right].
]

这是一个统一对象：

* (F_x) 的特征值给出成熟模型实际可辨识的 table modes；
* (\delta x^\top F_x\delta x) 预测 hard-swap 的短窗口 attention shock；
* 两个 profile 若在 (F_x) 顶部子空间上的投影近似相同，LM 行为也应近似相同；
* 高维 (z) 优化若主要沿低特征值方向移动，就会对有限文档噪声过拟合。

论文 Appendix A.12 实际上已经给出了这一方向的一维版本：先推导 logit transport，再用 softmax Jacobian Rayleigh quotient度量 attention map 扰动。  下一步应把它提升为完整 interior-(z) 的 (K\times K) Fisher，而不是继续构造新的静态 Gram。

LeRoPE 的精确频率梯度也表明，真实频率信号由 Q/K 范数、未旋转 QK 角度、post-softmax attention、((v_t-o_s)) 以及 downstream gradient 共同决定。([arXiv][5]) 这正是 structural geometry 所缺失的部分。

## 3. Training-frame 与 phase OOD

设

[
\psi_\Omega(d)
==============

[
\cos\omega_1d,\sin\omega_1d,\ldots,
\cos\omega_Kd,\sin\omega_Kd
]^\top,
]

使用 head-specific metric (M_g\succeq0) 后

[
\phi_g(d)=M_g^{1/2}\psi_\Omega(d),
]

训练期 frame operator 为

[
G_g^{\rm train}
===============

\mathbb E_{d\sim\pi_{g,\rm train}}
[
\phi_g(d)\phi_g(d)^\top
].
]

这样五个概念可以放在同一套对象中：

| 机制             | 可检验对象                                                                |
| -------------- | -------------------------------------------------------------------- |
| 训练期可观测性        | (G_g^{\rm train}) 的谱与有效秩                                             |
| 局部分辨率          | (|\phi_g(d+1)-\phi_g(d)|^2)                                          |
| 远程碰撞           | (|\phi_g(d)-\phi_g(d')|^2) 的低分位数                                     |
| phase OOD      | (\phi_g(d)^\top(G_g^{\rm train}+\lambda I)^{-1}\phi_g(d)) 的 leverage |
| attention 实际使用 | (M_g,\Sigma_g,F_x)                                                   |

Vandermonde conditioning 和非谐 Fourier 理论能够告诉我们：有限窗口内频率分离不足会导致指数系统病态；Moitra 给出了频率间距与 Vandermonde 条件数之间的 sharp transition。([arXiv][6]) Slepian 的离散 prolate 理论则说明，有限时间窗口与有限频带共同决定真正可观测的自由度。([Wiley Online Library][7])

但这些只能提供 **必要的 observability/frame constraints**。它们不包含 (M_g)，不能单独预测 LM。你们的 phase-isotropy/min-eigen 失败正好给出了这一负面证据。

---

# 四、现有矛盾的解释

## 1. 为什么 structural pair geometry 更好，却不能稳定预测 LM

### 原因一：whitening 抹掉了训练信号幅度

对 (\omega L\ll1)，

[
1-\cos(\omega d)
\approx
\frac12(\omega d)^2.
]

慢频即使在标准化后张成独立方向，其原始相位运动、频率梯度和 signal-to-noise 仍可能很小。结构指标通过 whitening 消除了这个差异。

### 原因二：距离测度错了

均匀遍历 ([0,L]) 与模型实际使用的 attention-distance distribution 不是同一对象。自然语言依赖宽度会改变最合适的频率尺度，且长上下文泛化取决于依赖模式怎样跨尺度延伸。([arXiv][4])

### 原因三：结构目标把所有不变性当成冗余

低频或 NoPE-like 通道可能是 semantic content match 的载体。把它们重新分配到更强旋转频率，会提高 positional rank，但损害内容匹配。

### 原因四：pairwise geometry 不控制 joint lower tail

LM 在某个长度的退化可能来自几个高权重 band 的联合相消。平均 canonical correlation、平均 kernel distance、甚至全窗口 min-eigen 都不保证 dyadic annulus 内不存在窄的 resonance valley。

### 原因五：attention 不是网络终点

FoPE 的分析明确指出，RoPE 在 attention 中的周期结构还会被后续线性层、激活和训练期频谱截断破坏。([arXiv][8]) 所以 attention-only geometry 最多是必要条件，不是端到端 LM 目标。

## 2. 为什么 phase-chord 目前最接近 win-win

令某个 pair 的内容系数为 (a=re^{i\varphi})。它在相对距离 (d) 的贡献可写成

[
r\cos(\varphi+\omega d).
]

对内容相位 (\varphi) 做各向同性平均：

[
\mathbb E_{\varphi}
\left[
r\cos(\varphi+\omega d)-r\cos\varphi
\right]^2
=========

r^2\big(1-\cos(\omega d)\big).
]

而

[
|e^{i\omega d}-1|^2
===================

2\big(1-\cos(\omega d)\big).
]

因此 (1-\cos(\omega d)) 不是任意 heuristic；它是：

* 圆周上精确 chord distance 的一半；
* isotropic content phase 下的预期 pair-logit 变化；
* 小相位下 log-frequency Fisher 的有限差分版本；
* 大相位下自然饱和并保留周期 wrapping 的函数。

再把它对真实 attention-distance prior 平均，就同时保留了：

* 相位运动幅度；
* 训练期间实际使用的距离；
* 圆周周期结构。

这比 whitened pair Gram 更接近 LM 的观测过程。

而 (\rho\propto w^{1/3}) 本身并不是新疑点。论文已经推导了高分辨率有限采样失真

[
D_K[\rho]
\approx
\frac{1}{12K^2}
\int \frac{w(\phi)}{\rho(\phi)^2},d\phi,
]

其变分最优解自然是 (\rho\propto w^{1/3})。

**所以 phase-chord 的成功真正支持的是 weight (w) 更接近正确对象，而不是再次支持 cube-root calculus。**

## 3. phase-chord 的成功来自什么

我的当前裁决是：

* **算子结构：有实质贡献。** chord 是圆周相位差和局部 Fisher 的正确有限差分。
* **attention prior：很可能是 win-win 的主要来源。** 它避免把预算分给训练期几乎不使用的相位区域。
* **选择偏差：仍然是实质性威胁。** 一个 recipient seed 参与 profile 构造，当前结果不能证明可迁移性。

最关键的是，当前 phase-chord 还没有正确处理 softmax centering。

对一个 query 的 attention 分布 (p_j)，更精确的单频率、label-free profile 是

[
w_{\rm sc}(\omega)
==================

\mathbb E_i
\left[
\operatorname{Var}*{j\sim p_i}
\left(
e^{i\omega\Delta*{ij}}
\right)
\right],
]

即

[
w_{\rm sc}(\omega)
==================

\mathbb E_i
\left[
1-
\left|
\sum_j p_{ij}e^{i\omega\Delta_{ij}}
\right|^2
\right].
]

等价地，

[
w_{\rm sc}(\omega)
==================

\mathbb E_i\mathbb E_{j,j'\sim p_i}
\left[
1-\cos\big(\omega(\Delta_{ij}-\Delta_{ij'})\big)
\right].
]

这比当前从零偏移出发的

[
\mathbb E_\Delta[1-\cos(\omega\Delta)]
]

更原则化，因为：

* 它与 softmax 的 (J_{\rm sm}) 完全一致；
* 自动消除 query-wise 常数方向；
* 测量的是 key 竞争集合内部的相位分辨率；
* 不需要 LM label 或 target-length loss。

完整版本就是前述

[
F_{kk'}
=======

\mathbb E_i
[
b_{i,k}^\top J_{\rm sm}(p_i)b_{i,k'}
].
]

**这是我认为当前最值得尝试的新数学对象。**

## 4. 1024/2048 非单调响应反映什么

单频率或多频率相位分离

[
D(q)
====

\sum_k w_k\big(1-\cos(\omega_kq)\big)
]

本来就是 almost-periodic，而不是随 (q) 单调增加。因此 4× 退化、8× 恢复完全可能来自相位 beating，而不需要假设训练噪声。

LeRoPE 的分析已经观察到，一个 dominant band 在训练窗口中落在约 (2.2L_{\rm train}) 周期的负半周期，超过训练窗口后贡献翻正并引发 naive extrapolation 失效。([arXiv][5])

但你们的 4×/8× 反转仍至少有三种竞争解释：

1. **relative-offset resonance**：由 (\omega_kd) 决定；
2. **total-length dilution/routing**：上下文更长后，softmax 竞争 key 数和活跃 head 发生变化；
3. **evaluation content composition**：不同长度尾部并非同一目标 token 或同一远程依赖。

最小判别实验不是再评 16 个长度，而是做一个二维控制：

* **固定总长度 (N)，移动同一个 source 的相对位置 (d)**，或只修改显式 position IDs；
* **固定 source-query 距离 (d)，增加无关 distractor 改变总长度 (N)**。

若 valley 跟着 (d) 移动，是相位 resonance；若跟着 (N) 移动，是 dilution/head routing；若二者都不稳定，优先检查数据构成。随后冻结激活，对 head×band 做 logit contribution sweep，即可定位是不是少数 dominant bands 的符号翻转。

---

# 五、真正有希望的从头训练方向

## 方向 A：role-conditioned、usage-weighted robust phase code

这是优先级最高的方法方向。

### 1. 显式区分 invariant 与 positional subspace

对 head group (g)，使用

[
R_g(\Delta)
===========

I_{2r_g}
\oplus
\bigoplus_{k=1}^{K-r_g}
R(\omega_{g,k}\Delta).
]

其中：

* (I_{2r_g}) 是 exact NoPE/invariant pairs；
* 其余 pair 构成 positional phase code。

选择 (r_g) 时，不按频率索引手工截断，而使用 training-only donor statistics，例如

[
\text{role score}_{h,k}
=======================

\frac{
\text{positional Fisher}*{h,k}
}{
\text{QK content energy}*{h,k}+\epsilon
}.
]

低 role score、但内容能量高的维度倾向保留为 invariant；高 positional/retrieval Fisher 的维度进入 rotary code。

这条路线有一个很强的第一性原理理由：**任何非零频率最终都会旋转。** 若模型确实需要跨任意长度保持稳定的 semantic channel，重新排列正频率永远不能提供严格不变性；必须引入 (\omega=0)、NoPE subspace 或双坐标算子。

这一步有意超出论文当前“固定正支撑内部 allocation”的轴。若研究目标是突破方法上限，这是合理的；若必须严格保持当前轴，则设 (r_g=0)，其余设计仍然适用。

### 2. 用 softmax-centered/Fisher profile 取代静态 profile

由独立 donor 模型收集：

* query-conditioned attention 分布；
* Q/K pair norms 和相位；
* 可选的 value/downstream 因子；
* 不同语料和不同种子的 profile。

构造 robust profile

[
\bar w_g(\phi)
==============

\operatorname{RobustAgg}*{r\in\mathcal D*{\rm donor}}
w_{g,r}(\phi),
]

例如 donor 间的 lower quantile 或 mean-minus-variance，而不是对单一 seed 拟合。

当 (K) 足够大时，使用

[
\rho_g(\phi)
\propto
\big(
\lambda+(1-\lambda)\bar w_g(\phi)
\big)^{1/3}.
]

当每组只有 8–16 个 rotary pairs 时，不要继续假设高分辨率近似。直接求解有限原子设计：

[
\min_{\phi_{g,1:K_g}}
\int
\bar w_g(\phi)
\min_k|\phi-\phi_{g,k}|^2d\phi
+
\lambda_{\rm rec},
\mathcal R_{\rm lower-tail}(\Omega_g).
]

这允许 profile 自然形成多峰，而不是强制 Cosh、ramp 或单调密度。

### 3. 只使用少数 head groups，而不是完全 per-head 自由

相关文献并未给出“每个 head 都应该有独立表”的一致结论：

* AdaRoPE 在其预训练与 context-extension 设置中发现，不同功能 head 需要不同频率范围和 scaling。([arXiv][9])
* LeRoPE 的对应 ablation 却发现，全 layer/head 共享一组频率最稳定，per-head/per-layer 参数化并未更好。([arXiv][5])
* 但 LeRoPE 从独立训练 run 学到并冻结的表仍保留了 63.6% 的 gain，说明某种跨 run 的 frequency prior 是可迁移的。([arXiv][5])

因此合理的中间方案是：

* (G=2) 到 (4) 个 regularized head groups；
* 组内共享表；
* 分组依据 local/semantic/retrieval/Fisher profile；
* 不开放每个 head 的高维自由频率。

候选组可以是：

1. local/positional；
2. semantic/invariant；
3. sparse retrieval/global；
4. 必要时一个 mixed group。

这比“单表”更能匹配真实功能，又比 AdaRoPE 式完全自由的 per-head 设计更容易识别、迁移和解释。

### 4. 在平均 phase utility 外加入 lower-tail anti-collision

phase-chord 目前更接近平均 utility。要解决 4× valley，目标中必须增加

[
\operatorname{CVaR}_\alpha
\left[
-D_g(q)
\right],
\qquad
q\sim\mathcal A_m,
]

而不是只最大化平均 chord 或平均 frame rank。

最新的 ATFlash 也从另一个角度强调了每个 RoPE wavelength 的有效判别范围不同，并按 wavelength 设置 attention 范围。([arXiv][10]) 它不是 allocation 方法，但支持“每个 band 有不同 operational scale”这一建模假设。

---

# 六、条件性高风险方向：Fisher 约束的 anti-resonance micro-jitter

只有在二维 offset/length 实验证明 4× valley 主要跟随相对 offset 后，才值得推进这一支线。

构造

[
\omega_k'
=========

\omega_k+\delta\omega_k,
]

约束

[
\max_k|\delta\omega_k|L_{\rm train}\le\epsilon
]

以及

[
\delta x^\top F_{\rm train}\delta x
\le \epsilon_F^2,
]

但最大化 dyadic annuli 中使用加权 code distance 的低分位数：

[
\max_{\delta x}
\inf_{\nu\in\mathcal V}
\sum_m\nu_m
Q_\alpha
\left[
D_{\Sigma}(q;\Omega+\delta\Omega)
\right].
]

其关键性质是：

* 在训练窗口，(\delta\omega L_{\rm train}) 很小；
* 在远端，(q\delta\omega) 可以明显破坏相干 recurrence；
* 不改变端点和粗 profile；
* 可作为 phase-chord 或几何表的微扰层。

但这条路线有明显风险：若模型只实际使用少数 band，优化全 (K) 维 torus recurrence 会再次在“未使用维度”上取得虚假改善。必须使用 (\Sigma_g) 或 (F_x) 加权，而不能做纯数论表设计。

---

# 七、成熟模型 retrofit 必须单独立项

## 1. 从头训练与 retrofit 是两个不同优化问题

从头训练是

[
\min_{\theta,\Omega}
\mathcal L(\theta,\Omega),
]

模型可以从第一步起共同选择 Q/K/VO 与频率坐标。

retrofit 是

[
\min_{\theta(t),\Omega(t)}
\mathcal R_{\rm final}
]

满足

[
(\theta(0),\Omega(0))
=====================

(\theta_{\rm native},\Omega_{\rm native}),
]

并受到：

* 可训练 token 数；
* 可更新参数；
* native forgetting；
* 路径上的 loss barrier；
* 最终推理复杂度

等约束。

所以从头训练最优的 (\Omega^\star)，可能不是成熟模型在低预算下最可达的终点。**前者优化终点，后者优化带路径约束的可达终点。**

论文的 crossing 和 fixed-map obstruction 已经提供了这一区分的理论基础。成熟模型的 hard swap 不能用来推断该表从头训练时存在 intrinsic 1×–2× trade-off。

## 2. 先做一个无需训练的高收益诊断

在现有 1.485B checkpoint 上计算完整 (F_x)，然后做三件事。

### A. 检查 spectrum

若前 1–3 个特征方向解释绝大部分 trace，则说明成熟模型只识别少数粗 schedule modes。高维 (z) 优化失败将获得直接解释。

### B. 比较 derived profile 与 ramp

令二者位移为 (\delta x_{\rm derived}) 和 (\delta x_{\rm ramp})。计算

[
d_F^2
=====

(\delta x_{\rm derived}-\delta x_{\rm ramp})^\top
F_x
(\delta x_{\rm derived}-\delta x_{\rm ramp}).
]

PDF 第 29 页 Table 13 中 derived 与 ramp 的差值置信区间均包含零，说明二者尚不可辨识。

如果 (d_F) 也很小，则“细 profile 细节未被模型读取”得到机制解释；如果 (d_F) 很大但 LM 仍无差异，说明 attention-KL 仍不足，需要加入 V/O/downstream task-Fisher。

### C. 验证 shock 预测

对已有 frozen swaps，检查

[
\delta x^\top F_x\delta x
]

是否预测 1× attention KL、NLL 和 head-level disruption。若不预测，暂时不要以 Fisher 设计 retrofit path。

## 3. 最有希望的 retrofit：Fisher-geodesic sparse-head morph

### 第一步：选择低维 schedule modes

不要再优化原始高维 (z)。构造：

* (F_{\rm short})：native 短窗口 shock；
* (G_{\rm long})：预声明 scale distribution 下的远程区分/phase utility。

使用 utility-to-shock 比例选择方向，例如广义特征问题：

[
G_{\rm long}u
=============

\lambda
(F_{\rm short}+\epsilon I)u.
]

仅保留 1–3 个跨校准文档稳定的 modes：

[
\delta x=U_rc.
]

这比手工 ramp 或高维 z 更可识别。

### 第二步：按 head 的 shock/utility 决定迁移范围

每个 head 计算：

[
s_h
===

\mathbb E_{\rm native}
\mathrm{KL}
(A_h^{\rm native}|A_h^{\rm target}),
]

以及远端 utility (u_h)，例如：

* remote-source attention recovery；
* causal source deletion drop；
* random-scale phase exposure 下的 robust code gain。

然后：

* 高 shock、低 utility：保留 native；
* 低 shock、高 utility：直接 morph；
* 高 shock、高 utility：使用 dual-coordinate bridge；
* 低 positional utility、高 semantic energy：考虑 NoPE/native anchor。

这不是预设“只改 retrieval heads”，而是让数据决定。但 retrieval-head 文献表明，稀疏选择在机制上是合理候选。([arXiv][2])

### 第三步：沿 Fisher 控制的连续路径迁移

使用 log-frequency path

[
x(t)
====

(1-t)x_0+tx_1
]

只是初始化。实际步长应满足

[
\delta x_t^\top F_x(t)\delta x_t
\le\epsilon_{\rm step}^2.
]

即每一步保持近似固定的 attention-KL，而不是固定 (t) 增量。

## 4. 临时 dual-coordinate bridge

最稳妥的路径不是直接替换，而是在训练阶段临时保留两套 logit：

[
z_h(t)
======

(1-g_h(t))z_h^{\rm native}
+
g_h(t)z_h^{\rm target}.
]

训练流程：

1. (g_h=0)：模型函数严格等于 native；
2. 冻结 native branch；
3. target branch 只使用 Q/K LoRA 或小型 pair-mixer；
4. 在 native 数据上蒸馏 attention 和 token distribution；
5. 在随机 log-scale position-ID augmentation 上训练远程 phase；
6. 只有当 1× KL 低于阈值时才增加 (g_h)；
7. 最终尝试蒸馏为 target-only branch。

这与 function-preserving network morphism 的原则一致：先扩展一个初始贡献为零的新分支，再逐渐转移函数，而不是一开始破坏旧函数。Net2Net 是这种思想的经典实例。([arXiv][11])

若 target-only 蒸馏始终失败，而 dual branch 持续有效，则结论应是：**成熟模型需要永久双坐标 chart，而不是更长的单表优化。**

## 5. Q/K 与 V/O 的角色

Q/K 是第一优先级，因为 RoPE 直接作用在 Q/K 并决定 addressing。但“只改 Q/K”不能成为默认真理：

* Q/K 决定关注谁；
* V 决定从该位置读取什么；
* O 决定该 head 的信息怎样写回 residual stream；
* 下游梯度决定这种重定向是否有价值。

LeRoPE 的频率梯度中明确出现

[
\alpha_{st}
|q_s||k_t|
(v_t-o_s)^\top g_s,
]

说明频率 utility 天然依赖 V 和 downstream。([arXiv][5])

论文中的 Q/K-only adaptation 在 2Wiki 1× 上接近保留，但 RULER 4K 仍从 72.25 降至 42.50，因此“Q/K-only 已解决 coordinate shock”并未建立。

最小诊断是 cross-patching：

* native attention (A_0) + adapted (V/O)；
* adapted attention (A_1) + native (V/O)。

若恢复 (A_0) 后输出能力恢复，问题主要在 Q/K addressing；若 attention KL 已恢复但输出仍差，先解冻 O，再考虑 V。不要一开始就全参适配。

---

# 八、最值得先做的判别实验

## 1. 第一优先：phase-chord 机制的 cross-fitted factorial

使用 151.9M exact-range，而不是再用高噪声 50M 作为最终判据。所有 profile 必须由独立 donor 模型和独立文档构造，构造完成后冻结。

建议五个 arm：

| Arm | Prior                       | Kernel/profile                    |
| --- | --------------------------- | --------------------------------- |
| A   | 独立 donor attention prior    | exact (1-\cos x)                  |
| B   | 预声明 generic log-scale prior | exact (1-\cos x)                  |
| C   | 独立 donor attention prior    | matched nonperiodic saturation    |
| D   | 无细节 prior                   | 与 A 匹配端点、RMS 位移和快慢质量的 coarse ramp |
| E   | —                           | FMRoPE                            |

非周期 matched kernel 可取

[
g(x)
====

2\left(1-e^{-x^2/4}\right),
]

它与 chord 有相同的小角度二次行为和相似饱和尺度，但没有周期 recurrence。

所有 intervention 应匹配：

* 固定 endpoints；
* 相同 RMS node displacement；
* 相同总体 fast/slow mass shift；
* 相同训练协议。

至少使用两个完全未参与 profile 构造的 paired seeds；不是为了“多 seed 补证据”，而是为了排除 donor leakage。

判别逻辑：

* **A 优于 B**：attention prior 是主要因素；
* **A 优于 C**：精确圆周 chord/periodicity 有额外贡献；
* **A≈D**：只识别了粗质量重分配，phase profile 细节没有被识别；
* **A 不再转移**：当前结果主要是 seed/profile 选择偏差；
* **A、B 都赢且 C 差**：较强的 operator-universal 证据；
* **只有 A 赢**：方法是 training-prior-conditioned，不应声称 universal analytic table。

若 (w_{\rm sc}) 与当前 phase-chord profile 实际差异足够大，再增加一个 softmax-centered arm；若两张表几乎相同，不值得额外训练。

## 2. 评价方式

不要只报告 1×/2×/4×/8× 四点平均。使用：

* nested documents；
* 相同 target tokens；
* dense virtual-offset sweep；
* 每个 dyadic annulus 的 worst/lower-quantile；
* 预声明 1× 容忍阈值；
* remote-source causal deletion。

论文现有 source-deletion 已经证明远端增益确实使用了远程内容，这类机制验证应继续保留。

## 3. Retrofit 最小四臂实验

保持相同 target table、相同更新参数量、相同训练 tokens：

1. hard swap + Q/K LoRA；
2. one-shot target + native attention/output distillation；
3. Fisher-adaptive homotopy + 同样 Q/K 预算；
4. dual-coordinate bridge + 同样 Q/K 预算。

结果解释：

* 3/4 显著改善 1×、远端相同：coordinate path 是主要瓶颈；
* 四者收敛到同一 Pareto：最终 table 本身存在 trade-off；
* attention KL 恢复、输出能力仍差：V/O 或 residual coadaptation；
* dual branch 有效、target-only 蒸馏失败：永久双 chart 更合理。

你们现有 native/long session routing 应作为 **oracle Pareto envelope**，而不是和单表争夺同一个 claim。Jet-Long 独立采用 local native-faithful window 加 dynamic remote window，并在短输入精确恢复原模型，说明 bifocal operator 是一个有原则的竞争解，而不只是工程 workaround。([arXiv][12]) DroPE 则提供了更激进的证据：训练时需要的位置归纳偏置与部署时最终使用的 positional operator 可以不同。([arXiv][13])

---

# 九、应停止或降级的现有路线

| 路线                                              | 裁决    | 原因                                         |
| ----------------------------------------------- | ----- | ------------------------------------------ |
| 继续证明 allocation 是独立轴                            | 停止    | 已由 exact-range 多种子和 crossing 建立            |
| phase-isotropy / pair Gram 作为主优化目标              | 停止    | 已出现稳定性失败；缺失 usage、softmax、semantic role    |
| min-eigen 作为单一目标                                | 停止    | 只是 E-optimal static frame，当前表现为远端收益换 1× 代价 |
| 高维 mature (z) + 少量文档                            | 停止    | 识别维度不足，优化噪声方向                              |
| frozen hard swap 推断 intrinsic trade-off         | 停止    | 它主要测 coordinate shock                      |
| mature fine profile 继续精修                        | 暂停    | derived≈ramp；先计算 (F_x) 有效秩                 |
| EVQ-Cosh 继续扫 (\tau)/base                        | 降级为基线 | 已不再是最接近突破的对象                               |
| target-aware support retargeting 冒充 target-free | 禁止    | 解决的是另一个问题                                  |
| unrestricted per-head full table                | 不作为首轮 | 自由度太高，文献证据相互矛盾，难以归因                        |
| 直接跳 GRAPE/Selective RoPE                        | 暂缓    | 会同时改变旋转平面、频率和内容门控，无法回答当前机制                 |

静态几何仍应保留，但只作为：

* observability constraint；
* collision diagnostic；
* 理论 lower bound；
* 失败定位工具。

不能再作为 LM surrogate 的主要证据。

---

# 十、所有候选都失败时仍能得到什么

1. **若 cross-fitted phase-chord 失败**
   当前 win-win 主要来自 donor/seed/data-specific selection；不存在已被证明可迁移的 operator-only allocation recipe。

2. **若 usage-weighted、grouped code 也失败，而 routing 稳定成功**
   单张静态共享表面临真正的 scale/role Pareto 冲突。正确对象可能是双坐标或 dynamic operator，而非更精细的表。

3. **若 NoPE anchor 失败**
   低频通道的作用不能简单归结为 semantic invariance，或者 semantic/positional role 会在训练中重新分配，静态 donor 分类不可迁移。

4. **若 anti-resonance 改善所有 code proxy 但 LM 不变**
   实际瓶颈不是 torus recurrence，而是 attention routing、QK content coefficient 或 downstream computation。

5. **若 homotopy 和 dual bridge 都失败**
   hard swap 的短损失不是单纯 path barrier；目标表本身与 native function 存在低预算下不可消除的冲突。

6. **若 (F_x) 有效秩极低**
   成熟模型只能识别一两个粗 schedule modes。高维 allocation 细节在预训练后已经成为不可访问自由度；这本身是一个重要 negative result。

最终可能得到的最强理论结论不是“找到了 universal table”，而是：

> 固定正频率、有限维、静态共享的 RoPE 代码具有不可避免的 recurrence；其 LM 效果由训练数据和 head-role-conditioned usage metric 决定；普适 target-free dominance 不能由 table geometry 单独获得。

这比再给出一张 marginally better analytic table 更有价值。

---

# 十一、极简研究导航协议

每个结论只记录以下字段：

```yaml
id:
object:        # 数学对象，不用方法昵称
status:        # ESTABLISHED | CLOSED_NEGATIVE | PROMISING | OPEN | REOPENED
claim:
scope:
evidence:
selection_boundary:
falsifier:
aliases:
supersedes:
```

初始化 ledger：

| ID                      | Status                  | 内容                                                 |
| ----------------------- | ----------------------- | -------------------------------------------------- |
| `F-ALLOC-001`           | ESTABLISHED             | 固定 support/endpoints 时，interior allocation 改变训练结果  |
| `F-COADAPT-001`         | ESTABLISHED             | weights 与 table 共适应；固定线性 Q/K map 不能精确 transplant   |
| `N-STATIC-GEOM-001`     | CLOSED_NEGATIVE         | 静态 pair Gram/rank 不能单独预测 LM                        |
| `H-PHASE-CHORD-001`     | PROMISING               | 两种子近 win-win；存在 donor-seed 参与构造问题                  |
| `N-HIGH-D-Z-001`        | CLOSED_NEGATIVE         | mature checkpoint 上少量文档直接优化高维 (z) 不稳定              |
| `N-FINE-PROFILE-001`    | CLOSED_NEGATIVE         | mature 模型下 derived profile 尚不可区别于 coarse ramp      |
| `O-TARGETFREE-RISK-001` | OPEN                    | dyadic scale ambiguity 下的 minimax/Pareto 风险        |
| `O-USAGE-CODE-001`      | OPEN                    | softmax-centered/Fisher-weighted finite phase code |
| `O-ROLE-SPLIT-001`      | OPEN                    | invariant NoPE subspace + positional rotary code   |
| `O-MORPH-001`           | OPEN                    | Fisher-controlled homotopy与 dual-coordinate bridge |
| `E-ROUTING-001`         | ESTABLISHED_ENGINEERING | native/long session routing 有效，但不是单表理论解            |

导航规则只需要四条：

1. **按数学对象命名，不按新方法名命名。**
2. **每条结果明确写 selection data 与 evaluation data。**
3. **CLOSED_NEGATIVE 只能因新假设、新作用域或新证据重开。**
4. **新方法提交前必须说明它与 ledger 中哪个 object 不同，不能只换 kernel、loss 或名字。**

---

# 最终优先级

**P0，无训练成本：** 在 1.485B 上计算完整 (F_x) 谱、derived-vs-ramp 的 Fisher 距离，并完成 relative-offset × total-length 二维诊断。

**P1，第一项决定性训练实验：** 独立 donor 构造的 empirical-chord / generic-chord / matched-nonperiodic / matched-ramp factorial，在 151.9M exact-range 上用两个完全独立 recipient seeds。它直接裁决 phase-chord 的成功来源。

**P2，真正的方法突破：** softmax-centered、usage-weighted、少数组 head 的 robust phase code；同时显式测试 NoPE invariant anchors。不要再强制单调密度。

**P3，成熟模型单独推进：** 低维 generalized schedule modes + Fisher-adaptive homotopy + temporary dual-coordinate bridge。当前 native/long routing作为可达到的 Pareto 上界。

最重要的转向是：**停止寻找“更好的全局频率密度”，开始设计“模型真正使用的、角色分化的有限相位代码”，并把成熟模型转换明确建模为坐标运输问题。**

[1]: https://arxiv.org/abs/2410.06205 "https://arxiv.org/abs/2410.06205"
[2]: https://arxiv.org/abs/2404.15574 "https://arxiv.org/abs/2404.15574"
[3]: https://www.arxiv.org/pdf/1809.05570v3 "https://www.arxiv.org/pdf/1809.05570v3"
[4]: https://arxiv.org/abs/2607.07678 "https://arxiv.org/abs/2607.07678"
[5]: https://arxiv.org/html/2607.10134v1 "https://arxiv.org/html/2607.10134v1"
[6]: https://arxiv.org/abs/1408.1681 "https://arxiv.org/abs/1408.1681"
[7]: https://onlinelibrary.wiley.com/doi/abs/10.1002/j.1538-7305.1978.tb02104.x "https://onlinelibrary.wiley.com/doi/abs/10.1002/j.1538-7305.1978.tb02104.x"
[8]: https://arxiv.org/abs/2412.17739 "https://arxiv.org/abs/2412.17739"
[9]: https://arxiv.org/abs/2607.19363 "https://arxiv.org/abs/2607.19363"
[10]: https://arxiv.org/abs/2608.02947 "https://arxiv.org/abs/2608.02947"
[11]: https://arxiv.org/abs/1511.05641 "https://arxiv.org/abs/1511.05641"
[12]: https://arxiv.org/abs/2607.07740 "https://arxiv.org/abs/2607.07740"
[13]: https://arxiv.org/abs/2512.12167 "https://arxiv.org/abs/2512.12167"
