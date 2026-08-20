# Full-RoPE 有限谱基、Attention 使用与训练共适应

- **日期：** 2026-08-19
- **状态：** 理论与 CPU 诊断已完成；新总 claim 尚未写入论文正文
- **用途：** ICLR 2027 核心理论重构、证据路由和跨上下文续作

## 技术摘要

本轮工作的结论不再是“EVQ-Cosh 是否最优”，而是：

> **有限 RoPE frequency table 是一组谱基。其 phase-invariant 子空间几何决定静态可辨识性；模型在训练中学习如何使用这组基，因此 frequency table 与 Q/K 权重发生强共适应。**

已完成的理论和数值结果支持以下四点：

1. 当前 cosine-only collision kernel 遗漏了每个 RoPE pair 的完整二维
   \(\operatorname{span}\{\cos(\omega\Delta),\sin(\omega\Delta)\}\)。白化
   cross-Gram 的 canonical correlations 给出 pair 内 phase rotation 不变的
   冗余度量，并与 block-whitened stable rank 存在严格恒等式。
2. geometric RoPE 的极慢 bands 存在真实 spectral collapse。普通度量下它们
   收敛到 \(\operatorname{span}\{1,\Delta\}\)；attention softmax 度量消除常数
   方向后，极限变为 centered
   \(\operatorname{span}\{\Delta-\mathbb E_p\Delta,\Delta^2-\mathbb E_p\Delta^2\}\)。
3. 最大化静态 rank/logdet 会产生近 Fourier harmonic comb，而不是自然的
   多尺度 RoPE 表。已经构造出 cosine collision 与 full rank 排序相反、以及
   \(L\) 与 \(2L/4L\) collision 排序反转的直接反例。因此 collision reduction
   不蕴含 extrapolation improvement。
4. 50M 的 \(2\times2\) weights-by-table counterfactual 显示，LM loss 主要由
   table × weights interaction 决定，而不是 table 或 weights 的独立主效应。
   单纯调整 base 能解释一大部分频谱移动，但不能解释全部 non-geometric
   allocation 与共适应。

因此，下一版论文可建立在“finite spectral basis + training co-adaptation”上；
EVQ-Cosh 应降为闭式、零学习参数的 constructive instance，而不是通用最优解。

---

## 1. 研究问题与结论边界

### 1.1 本轮回答的问题

给定有限 \(K\) 个 RoPE frequency pairs：

- 怎样无相位偏置地度量两个 frequency subspaces 的冗余？
- geometric 表的低频 bands 是否真的丢失有效维度？
- 静态 collision/rank 能否推出更好的长程行为？
- attention softmax geometry 与 task-sensitive LM gradient 是否一致？
- frequency table 的作用能否与 trained weights 的共适应分开？
- EVQ 的作用有多少可以由单纯调整 scalar base 解释？

### 1.2 本轮不支持的说法

- Cosh 是 full-RoPE、attention 或 LM loss 的全局/近全局最优。
- 任一静态 collision、effective-rank 或 logdet objective 可以直接预测外推。
- EVQ 与 LeRoPE 学到相同或相近的 frequency table。
- LeRoPE 验证了 EVQ 的外推机制。
- seed-42 的 post-hoc retrofit 结果等于多 seed、从头训练的因果结论。
- 当前 CPU probe 证明了 base-only 从头训练可以替代 non-geometric allocation。

---

## 2. Full-RoPE 子空间几何

### 2.1 Cosine-only kernel 漏掉的对象

真实一个 RoPE pair 对 relative-position attention logit 的贡献为

\[
f_\omega(\Delta)=C\cos(\omega\Delta)+D\sin(\omega\Delta).
\]

所以 frequency \(\omega\) 的自然对象是二维子空间

\[
V_\omega=\operatorname{span}\{\cos(\omega\Delta),\sin(\omega\Delta)\}.
\]

当前论文的 exact kernel

\[
K_{\cos}(\omega,\nu)
=\mathbb E_D[\cos(\omega\Delta)\cos(\nu\Delta)]
\]

只观察 \(D=0\) 的单一内容相位，遗漏 sin–sin、cos–sin 和 sin–cos 三个
Gram 分量，并且不对 pair 内 phase rotation 保持不变。

### 2.2 Self/cross Gram 与 canonical collision

令

\[
x_\omega(\Delta)=
\begin{bmatrix}\cos(\omega\Delta)&\sin(\omega\Delta)\end{bmatrix},
\]

\[
S_\omega=\mathbb E[x_\omega^\top x_\omega],\qquad
H_{\omega\nu}=\mathbb E[x_\omega^\top x_\nu].
\]

对均匀 \(\Delta\in[0,L]\)，记

\[
d=(\omega-\nu)L,\quad s=(\omega+\nu)L,\quad
a(t)=\frac{\sin t}{t},\quad b(t)=\frac{1-\cos t}{t},
\]

则

\[
H_{\omega\nu}=\frac12
\begin{bmatrix}
a(d)+a(s) & b(s)-b(d)\\
b(s)+b(d) & a(d)-a(s)
\end{bmatrix}.
\]

定义 whitened cross-Gram

\[
Q_{\omega\nu}=S_\omega^{-1/2}H_{\omega\nu}S_\nu^{-1/2}.
\]

其奇异值 \(\sigma_1,\sigma_2\) 是两个子空间的 canonical correlations。
推荐的 pair collision 是

\[
c_{\omega\nu}
=\frac12\lVert Q_{\omega\nu}\rVert_F^2
=\frac{\sigma_1^2+\sigma_2^2}{2}\in[0,1].
\]

该量不仅对 pair 内 phase rotation 不变，而且对同一二维子空间的可逆基变换
不变。

### 2.3 与 positional effective rank 的严格关系

对所有 pairs 做 block whitening，得到全局 correlation Gram \(R\)。其对角块
均为 \(I_2\)。若 \(\bar c\) 是所有 \(c_{ij}\) 的平均，则

\[
\operatorname{tr}(R)=2K,
\]

\[
\operatorname{tr}(R^2)=2K\bigl[1+(K-1)\bar c\bigr],
\]

从而 Rényi-2/stable effective rank 精确满足

\[
r_2(R)
=\frac{(\operatorname{tr}R)^2}{\operatorname{tr}(R^2)}
=\frac{2K}{1+(K-1)\bar c}.
\]

这是本轮最强的已完成静态定理。它不意味着 pairwise collision 可以唯一决定
Shannon effective rank 或 logdet；后两者仍包含高阶多子空间依赖。

---

## 3. 低频 spectral collapse

### 3.1 普通 \(L_2\) 度量

令 \(t=\Delta/L\)，\(x=\omega L\)。当 \(x\to0\)：

\[
\cos(xt)=1-\frac{x^2t^2}{2}+O(x^4),
\]

\[
\frac{\sin(xt)}{x}=t-\frac{x^2t^3}{6}+O(x^4).
\]

因此

\[
V_\omega\longrightarrow\operatorname{span}\{1,\Delta\}.
\]

对两个低频 \(x,y\)，symbolic projection 给出

\[
2-\lVert Q_{x,y}\rVert_F^2
=\frac{19}{12600}(x^2-y^2)^2+O(\epsilon^6).
\]

数值验证在 \(x=0.05,y=0.10\) 时，exact/leading 比为 `1.00058`。

### 3.2 Geometric RoPE 的低频维度损失

设置：\(L=4096\)、\(b=500000\)、物理端点 \([1/b,1]\)。只观察
\(\omega L\le1\) 的 geometric bands：

| \(K\) | 低频 pairs | 名义维数 | block-whitened \(r_2\) | raw entropy rank | 稳定维数损失 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 6 | 12 | 2.0001 | 1.056 | 83.33% |
| 32 | 12 | 24 | 2.0001 | 1.072 | 91.67% |
| 64 | 24 | 48 | 2.0002 | 1.079 | 95.83% |

解释：把每个 pair 当抽象子空间并白化后，许多低频 pairs 共同只提供约二维；
保留原始特征能量时，sine 能量还随 \(\omega^2\) 消失，所以 raw effective rank
进一步接近一维。

### 3.3 Softmax metric 下的 centered polynomial 极限

attention categorical Fisher 为

\[
F_{\rm sm}=\operatorname{diag}(p)-pp^\top,
\qquad F_{\rm sm}\mathbf1=0.
\]

令 \(\bar f=f-\mathbb E_p f\)。则

\[
\frac{\overline{\sin(\omega\Delta)}}{\omega}
\to\Delta-\mathbb E_p\Delta,
\]

\[
-\frac{2\,\overline{\cos(\omega\Delta)}}{\omega^2}
\to\Delta^2-\mathbb E_p\Delta^2.
\]

因此，只要 \(p\) 在至少三个不同距离上有非退化支持，softmax quotient geometry
收敛到 centered

\[
\operatorname{span}\{\Delta-\mathbb E_p\Delta,
\Delta^2-\mathbb E_p\Delta^2\}.
\]

在 50M 的四个 weights/table cells、1,920 个 head-query observations 上，
canonical chordal deficit 的 log-log slope 为 `4.009–4.011`，与
\(O((\omega\Delta_{\max})^4)\) 一致；未发现反例。

---

## 4. Static finite-\(K\) objectives 的失败边界

### 4.1 CPU-only 对照协议

- \(L=4096\)，\(b=500000\)；
- \(K\in\{16,32,64\}\)；
- 所有表强制共享物理 frequency endpoints \([2\times10^{-6},1]\)；
- uniform continuous \(\Delta\in[0,L]\)；
- 比较 geometric、endpoint-anchored EVQ-Cosh、cosine-collision grid optimum、
  full-subspace collision grid optimum、block-whitened logdet optimum；
- 数值 optimum 是 `1025` 个 log-frequency candidates 加解析 harmonics 上的
  greedy/exchange 解，不冒充连续全局最优；`2049` 网格复核了 pairwise 解。

### 4.2 结果

以下单元格为 `full-subspace collision / block-whitened stable rank`：

| \(K\) | geometric | EVQ-Cosh | cosine-opt | full-subspace-opt | logdet-opt |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | .2250 / 7.31 | .2163 / 7.54 | .0555 / 17.46 | 1.98e-7 / 32.00 | 1.98e-7 / 32.00 |
| 32 | .2383 / 7.63 | .1870 / 9.41 | .0314 / 32.42 | 1.02e-7 / 64.00 | 1.02e-7 / 64.00 |
| 64 | .2432 / 7.84 | .1034 / 17.03 | .0156 / 64.48 | 5.17e-8 / 128.00 | 5.17e-8 / 128.00 |

full-subspace/logdet 目标几乎达到满秩，但 \(K=64\) 的 pairwise optimum 把
62 个 interior frequencies 全部压到 \(\phi\in[0.0169,0.0264]\) 的近 Fourier
harmonic comb，仅保留两个强制端点。它优化了静态正交性，却放弃多尺度覆盖。

### 4.3 两个直接反例

**Cosine metric 选错表。** \(K=64\)：

\[
C_{\cos}(A)=2.62\times10^{-9}
<1.44\times10^{-8}=C_{\cos}(B),
\]

但 full stable rank 为

\[
r_2(A)=64.48<128.00=r_2(B),
\]

raw full entropy rank 为 `93.44 < 120.03`。

**长度排序反转。** 固定端点的两个 \(K=16\) 表：

\[
C_L(A)=.45837<.61695=C_L(B),
\]

\[
C_{2L}(A)=.45834>.41132,
\qquad
C_{4L}(A)=.45830>.24763.
\]

离散 \(\Delta=0,\ldots,L-1\) 复算保持相同反转。

此外，解析 Fourier table 可在 \([0,L]\) 上正交，却满足

\[
\Phi(\Delta+L)=\Phi(\Delta),
\]

从而在 \(L\) 外发生精确 positional aliasing。这严格否定“静态 collision 越低
就必然越会外推”。

---

## 5. Attention structural Fisher 与 LM empirical Fisher

### 5.1 必须分开的对象

若 \(J\) 是某组 positional/frequency 变量对 attention logits \(z\) 的 Jacobian：

\[
M_J^{\rm sm}=J^\top F_{\rm sm}J,
\qquad
F_{\rm sm}=\operatorname{diag}(p)-pp^\top.
\]

这是 attention categorical/softmax Fisher，也等于 softmax log-partition 的局部
Hessian。它不是 LM task Fisher。

对真实平均 LM loss，令

\[
g_z=\frac{\partial L_{\rm LM}}{\partial z}.
\]

本轮定义观测到的 empirical-gradient outer product：

\[
M_J^{\rm EF}
=(J^\top g_z)(J^\top g_z)^\top.
\]

它使用真实 task gradient，但仍是局部诊断，不是训练前预测器，也不是总体
Fisher expectation。

三种 Jacobian：

- `bare`：\(J=\Phi=[\cos(\omega_k\Delta),\sin(\omega_k\Delta)]\)；
- `content`：每个 pair 对实际 Q/K attention logit 的贡献；
- `frequency`：对 \(\log\omega_k\) 的真实局部导数，含 \(\omega_k\Delta\) 尺度。

### 5.2 50M \(2\times2\) counterfactual

协议：TinyStories validation、\(L=512\)、base 500K、seed-42；同一 8 个 windows、
query positions `63/127/255/383/511`、6 layers、8 heads，共 1,920 个
head-query observations。参数全部冻结；未训练，未使用 GPU。

| Weights | Runtime table | LM loss | PPL | bare geometry \(r_2\) | bare \(M^{sm}\) \(r_2\) | content \(M^{EF}\) `r2 / trace` | frequency \(M^{EF}\) `r2 / trace` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Geo | Geo | 1.9659 | 7.14 | 4.57 | 14.86 | 10.20 / 1.99e-8 | 1.79 / 8.80e-7 |
| Geo | EVQ | 4.3333 | 76.20 | 12.54 | 25.51 | 1.48 / 2.93e-6 | 2.21 / 3.95e-3 |
| EVQ | Geo | 3.1378 | 23.05 | 4.57 | 16.94 | 8.34 / 1.12e-7 | 1.04 / 5.63e-5 |
| EVQ | EVQ | 1.9685 | 7.16 | 12.54 | 21.91 | 12.65 / 1.69e-8 | 4.99 / 2.80e-7 |

两个自洽系统的训练长度 PPL 基本相同；两个 post-hoc table mismatch 明显崩溃。

### 5.3 Factorial decomposition

对任意 cell metric \(y\)，记 weights/table 为 `GG/GE/EG/EE`：

\[
E_T=\frac{y_{GE}+y_{EE}-y_{GG}-y_{EG}}{2},
\]

\[
E_W=\frac{y_{EG}+y_{EE}-y_{GG}-y_{GE}}{2},
\]

\[
I_{T\times W}=y_{EE}-y_{EG}-y_{GE}+y_{GG}.
\]

对 LM loss：

\[
E_T=+0.5991,\qquad E_W=-0.5965,\qquad I_{T\times W}=-3.5367.
\]

40 个 sampled-query groups、500 次配对 bootstrap：

| Effect | 95% CI |
| --- | ---: |
| table main effect | `[+0.331, +0.956]` |
| weights main effect | `[-0.896, -0.175]` |
| table × weights interaction | `[-5.165, -3.039]` |

interaction 绝对值约为两个 main effects 的 5.9 倍。当前最直接的解释是：
frequency table 是训练时的坐标系，权重学习如何使用它；训练完成后换表不是纯
geometry intervention，而是破坏已形成的共适应。

### 5.4 Metric validity verdict

- bare geometry 和 bare softmax rank 在最坏的 `Geo weights + EVQ table` cell
  上都显著“改善”，但 PPL 从 `7.14` 恶化到 `76.20`。这是 structural metric
  与 task loss 方向相反的直接反例。
- content softmax stable rank 的最大值出现在 `EVQ weights + Geo table`
  (`20.08`)，该 cell PPL 却为 `23.05`；自洽 EVQ cell rank 仅 `12.58`，PPL
  为 `7.16`。
- 四 cells 上，`log10(EF trace)` 与 LM loss 的描述性 Pearson 分别为 bare
  `0.991`、content `0.989`、frequency `0.994`。这说明 task-gradient sensitivity
  能识别 mismatch，但它使用了已经观察到的 loss gradient，不能被包装成
  training-free 选择器。
- content-weighted Jacobian 在机制上比 bare \(\Phi\) 更接近真实 pair 使用，
  但当前四 cells 没有证明它在预测上明显优于 bare empirical trace。
- frequency Jacobian 对 mismatch 极敏感：固定 Geo weights 换 EVQ 表后，
  frequency-EF trace 放大约 4,494 倍；但其量级含 \(\omega\Delta\) 参数尺度，
  代表局部脆弱性，不代表表更优。

---

## 6. Base-only controls：scale 与 interior allocation 部分重叠

本轮没有按 loss 搜 base。三个 geometric controls 均由历史 EVQ 表解析确定：

- `base=331.6K`：匹配 EVQ 最慢 sampled frequency；
- `base=377.7K`：匹配 EVQ log-frequency span；
- `base=8.06K`：在纯 geometric-base family 内最小化与 EVQ 表的 log-frequency
  RMS。

| Weights | Runtime table | Base | LM loss | PPL | Static \(r_2\) |
| --- | --- | ---: | ---: | ---: | ---: |
| Geo | original geometric | 500K | 1.9659 | 7.14 | 4.57 |
| Geo | span-match geometric | 377.7K | 1.9687 | 7.16 | 4.66 |
| Geo | endpoint-match geometric | 331.6K | 1.9884 | 7.30 | 4.71 |
| Geo | LS-fit geometric | 8.06K | 3.8206 | 45.63 | 7.53 |
| Geo | full historical EVQ | 500K+Cosh | 4.3333 | 76.20 | 12.54 |
| EVQ | original geometric | 500K | 3.1378 | 23.05 | 4.57 |
| EVQ | span-match geometric | 377.7K | 3.0975 | 22.14 | 4.66 |
| EVQ | endpoint-match geometric | 331.6K | 3.0769 | 21.69 | 4.71 |
| EVQ | LS-fit geometric | 8.06K | 2.2644 | 9.63 | 7.53 |
| EVQ | full historical EVQ | 500K+Cosh | 1.9685 | 7.16 | 12.54 |

结论：

- 对 Geo weights，轻量 span-match `500K→377.7K` 基本无损，但几乎不改变
  static rank。
- 对 EVQ weights，`base=8.06K` 将 Geo-table PPL `23.05→9.63`，追回从
  Geo table 到 EVQ table loss gap 的 `74.7%`。
- 因此，EVQ 的一大部分作用与“整体抬高慢频率”这一 base axis 同向；剩余
  `9.63→7.16` 才可能来自 non-geometric interior allocation、完整表差异和
  共适应。
- 当前只是 frozen-checkpoint counterfactual。要证明 base-only training 能否
  达到相同结果，仍需要 matched from-scratch base arm。

---

## 7. LeRoPE 的安全定位

本地已核验的 LeRoPE 事实：

- 每个 frequency band 学一个 scalar，共 32 个，层和头共享；
- 52M–2.5B 阶梯上学出跨 seed/规模一致的非几何 profile；
- 217M 上，使用另一轮 LeRoPE 学出的频率、从头以固定表训练，保留完整
  LeRoPE validation-PPL gain 的 `63.6%`；p-RoPE 仅保留 `10.4%`；
- 原文据此认为 final table 与 joint training dynamics 都有贡献。

这与本轮结果互补：

- Fixed-LeRoPE 说明“一张从别处得到的好表”具有可迁移价值；
- 它仍让新模型从训练开始就接触该表，因此权重可以共适应；
- 本轮 post-hoc swap 说明已经训练完成的权重不能安全地硬换坐标系。

下一版不能声称“首次学习/优化 per-band frequencies”。可主张的不同贡献是：

1. phase-invariant full-RoPE finite-basis geometry；
2. low-frequency collapse 及 softmax-centered 极限；
3. static identifiability 与 extrapolation 不等价的反例；
4. fixed-range interior-allocation identification；
5. table quality 与 frozen-retrofit compatibility 的分离；
6. 一个无需 learned run 的闭式 fixed-table 实例 EVQ-Cosh。

禁止复用旧内部笔记里已撤回的“LeRoPE 与 EVQ 是同一机制的两端”、
“\(2.205L\) 确证 EVQ 机制”等叙事。安全表述仅是：LeRoPE 独立证明 frequency
table 是值得设计的变量，并用 Fixed-LeRoPE 证明固定表携带显著价值。

---

## 8. 可投稿的 claim architecture

### 8.1 推荐中心 claim

> **A finite RoPE table is a spectral basis, not merely a scalar base or range
> hyperparameter. Its phase-invariant subspace geometry bounds positional
> identifiability, while the coefficients that exploit this basis co-adapt with
> the table during training.**

这个 claim 的每个分句都有独立证据：

| Claim component | Theory/evidence owner | 当前状态 |
| --- | --- | --- |
| finite table 是二维谱基集合 | full self/cross Gram | 已完成 |
| phase-invariant redundancy 可严格定义 | canonical collision | 已完成 |
| redundancy 与 effective dimension 有严格关系 | stable-rank identity | 已完成 |
| 低频重复导致维度损失 | static + softmax collapse | 已完成 |
| static rank 不等于 extrapolation | Fourier 与排序反例 | 已完成 |
| interior allocation 是真实训练变量 | exact-range + M4 | 已有论文证据 |
| table 与 weights 共适应 | 50M \(2\times2\) | 已完成；额外 seed 仅为可选稳健性 |
| fixed table 可跨训练复用 | Fixed-LeRoPE 63.6% | 外部一手证据 |
| 非平凡换表不能由冻结 Q/K 精确吸收 | operator theorem owner | **exact invertible case 已证明** |

### 8.2 Retrofit obstruction theorem：exact case 已证明

定理陈述：设 \(R_\Omega(\Delta)\) 为 frequency multiset \(\Omega\) 的
block-rotation operator。若存在与位置无关的可逆线性变换 \(A,B\)，使

\[
A^\top R_{\Omega'}(\Delta)B=R_\Omega(\Delta)
\]

对某个含开区间的 \(\Delta\) 集合恒成立，则 \(\Omega'\) 与 \(\Omega\) 必须
拥有相同 frequency multiset，仅允许符号与排列。频率重复时，相似变换可在整个
等频不变子空间内混合，而不必保留原始的二维 pair 分块。

证明已记录在
`rebuttal/rebuttal_0723/theory_results/OLMO2_POSTHOC_FREQUENCY_TRANSPLANT_OBSTRUCTION_20260726.md`：

1. \(\Delta=0\) 给出 \(A^\top B=I\)；
2. 因而 \(A^\top R_{\Omega'}(\Delta)A^{-\top}=R_\Omega(\Delta)\)；
3. 在零点求导，两个 block generator 相似；
4. 相似性保持特征值，而 generator 的谱为
   \(\{\pm i\omega_k\}\)，所以 frequency multisets 必须相同。

**状态：PROVED for exact, position-independent, invertible compensation.**
整数位置只额外允许符号与 \(2\pi\) alias。该定理不排除有限数据上的近似重训、
非可逆或非线性映射、V/O/残差路径及新算子。principal-angle approximate
residual bound 仍是可选开放问题，不是 exact theorem 的完成门槛。

### 8.3 EVQ-Cosh 的正确位置

EVQ-Cosh 保留三个角色：

1. stated convex surrogate 的闭式 stationary family；
2. 零 learned positional parameters、无需额外 learned run 的 fixed-table
   construction；
3. 用于识别 frequency allocation axis 的可复现实验干预。

它不再承担 full theory 的“正确答案”。旧的 \(\tau=d_{\rm eff}/\sqrt L\)
只保留为 fallible operating prior，不升级为静态或 task-optimal selector。

---

## 9. 与现有论文证据的连接

本轮理论不替换以下已验证实验，而是重新解释它们：

- **Exact-range identification：** 固定最高频、最低频和 log-span，只移动
  30 个 interior frequencies；seed-42 owner 报告的 tail-NLL 差为
  `-0.47750/-0.20499/-0.11284`，三 seed aggregate 的原始 per-seed 文件仍缺失，
  所以不使用未拥有的置信区间或显著性语言。
- **M4 factorial：** 12 structural configurations × 3 seeds；1.25× Cosh 在
  `10/12`、matched exponential 在 `9/12` 配置优于 uniform；formula Cosh
  与 Exp 差 `+0.000740 NLL`、`p=.836`。支持 non-geometric allocation axis，不支持
  Cosh shape 唯一性。
- **1.485B phase-matched adaptation：** Native 与 EVQ 使用相同 Q/K adaptation
  和 phase exposure；8K RULER `2.02 vs 31.63`。支持训练期 table 约束不能由
  long-phase exposure 单独替代。
- **8B matched adaptation：** 16K RULER `0.295 vs 14.03`。支持频率 substrate
  差异在成熟模型中持续存在；不单独证明 full theory。

论文中必须继续分开：

- exact-range study 拥有纯 allocation 因果识别；
- 50M \(2\times2\) 拥有 frozen-retrofit/co-adaptation 诊断；
- mature-scale studies 拥有 persistence 与 capability endpoint；
- LeRoPE 拥有外部 learned-table 与 Fixed-LeRoPE 证据。

---

## 10. 方法、复现与验证收据

### 10.1 独立 analysis scripts

| Script | 作用 | SHA-256 |
| --- | --- | --- |
| `scripts/analysis/full_rope_collision_audit.py` | full Gram、canonical collision、finite-\(K\) optimum、反例 | `7544a355380aa4c012181f1eba6ffa37c597fdd184e7962965ce0b89ba296ee4` |
| `scripts/analysis/attention_fisher_50m_probe.py` | \(2\times2\) LM loss、softmax/EF、layer-head、bootstrap、低频 softmax 极限 | `9e133a830c8622349c3ee19cf79e5fedf1193fbbef61ba469ea0a5be602c8a78` |
| `scripts/analysis/base_only_50m_control.py` | 解析 base-only controls 与 frozen-checkpoint PPL | `c588c77d0540e54af74fd91db9b0b85c8be4def909f194782c3cb75a4d186a24` |

### 10.2 本轮输出指纹

完整 JSON 默认写到 `/tmp`，未复制进论文包，避免机器路径和 checkpoint 身份
误入提交材料：

| Output | SHA-256 |
| --- | --- |
| `full_rope_collision_audit_20260819.json` | `5af8c92197136e5e0dd5bca5c0fde16883404a4cb95a28199721ca3cb82cd749` |
| `attention_fisher_50m_probe_20260819.json` | `32770ef134a457de8f4878f6d669f499defedafd079f96da625503b2660bc2c9` |
| `base_only_50m_control_20260819.json` | `dc3493b21ff5ba012b2161aa6fa98f1964b9529e6f88854266d1b069bc2c803f` |

### 10.3 验证通过

- full analytic Gram vs 200,001-point quadrature：max abs `1.51e-6`；
- phase-rotation canonical-correlation invariant：max abs `7.44e-15`；
- low-frequency symbolic leading coefficient：`19/12600`；
- 50M manual attention vs SDPA：max abs `1.25e-6`；
- manual vs fused full LM loss：max abs `4.77e-7`；
- log-frequency Jacobian finite difference：max abs `2.64e-6`；
- 8-window/500-bootstrap probe 确定性复跑：除 runtime seconds 外完全一致；
- base-only controls 确定性复跑：完全一致；
- 论文正文、训练代码和历史 `results/`：未修改；
- GPU/training：均未启动。

### 10.4 复现命令

```bash
python3 scripts/analysis/full_rope_collision_audit.py

CUDA_VISIBLE_DEVICES='' conda run --no-capture-output -n aidemo \
  python scripts/analysis/attention_fisher_50m_probe.py \
  --windows 8 --bootstrap-samples 500

CUDA_VISIBLE_DEVICES='' conda run --no-capture-output -n aidemo \
  python scripts/analysis/base_only_50m_control.py --windows 8
```

---

## 11. 限制与稳健性边界

1. 50M task-sensitive probe 目前只有 seed-42；其强 interaction 需要 seed-43/44
   复核。
2. 历史 Geo/EVQ tables 同时改变 sampled endpoints、span 和 interior shape；
   50M retrofit 不拥有 pure-shape identification。
3. empirical-gradient outer product 使用 window-mean LM loss 的 \(g_z\)；逐 token
   empirical Fisher 可能减少梯度抵消，尚未执行。
4. static full-RoPE 数值优化使用 uniform distance prior；不同 task/attention prior
   会改变 optimum，正是不存在 distribution-free static selector 的原因之一。
5. base-only control 是 frozen-checkpoint 推理诊断，不等于从头以该 base 训练。
6. LeRoPE 是并行工作；不能暗示我们的表优于其 learned/fixed tables，因为没有
   matched direct comparison。
7. ICLR 2027 官方模板已逐文件核验；旧 ICML 模板仅保留在
   `paper-2027/venue_icml_fallback/`，不参与构建。

---

## 12. 推荐下一步

按中稿杠杆排序：

1. **把已证明的 exact retrofit obstruction 写入论文。** 统一符号、补入正式
   theorem/proof，并保持 approximate-retraining 边界。
2. **重排论文骨架。** full-RoPE geometry → collapse/retrofit theory → exact-range
   training identification → mature persistence → EVQ constructive instance。
3. **做逐 token empirical-Fisher robustness。** 仅验证诊断排序，禁止据此优化
   新 frequency schedule；它不是正文重构的前置条件。
4. **CPU seed-43/44 的 \(2\times2\) 仅作可选稳健性。** 当前效应量已足以支撑
   诊断，不得把额外 seed 变成无必要的投稿阻塞项。
5. **若以后授权新训练，最小训练证据是 matched Geo-base500K / Geo-base8K /
   fixed-range non-geometric 三臂。** 未授权前不得启动。

进一步问题：

- principal-angle retrofit bound 能否在真实 content-weighted operator 上保持可计算？
- Fixed-LeRoPE 学出表与解析 base-only/EVQ 表的主要差异来自 scale 还是 interior
  placement？
- 训练期间何时形成 table-specific Q/K usage，早期轨迹能否预测最终共适应？

---

## 13. Agent continuation packet

### NON_NEGOTIABLE_CONSTRAINTS

- `paper/` 是不可修改的 NeurIPS 2026 baseline。
- `paper-2027/` 是唯一活跃论文目录。
- 禁止捏造数据、实验、统计、证明状态或引用。
- 未经用户明确授权，不启动训练、推理型 GPU 任务或付费机器。
- 内部审计必须记录反例和负结果；论文正文只使用准确且决策相关的边界。
- 不把 static collision/rank 称为 extrapolation 或 LM-quality 指标。
- 不把 softmax Fisher 与 LM empirical-gradient outer product 混称为同一 Fisher。
- 不把 EVQ-Cosh 称为 universal optimum；Cosh uniqueness 只属于 stated convex
  surrogate。
- LeRoPE 必须引用；不得声称 EVQ≈LeRoPE、LeRoPE 验证 EVQ 外推或我们优于它。

### ARCHITECTURE_DECISIONS

- 中心理论对象：full sin/cos frequency subspaces 与 canonical correlations。
- 中心论文 framing：finite spectral basis + training co-adaptation。
- EVQ-Cosh：constructive fixed-table instance，而不是总理论答案。
- exact-range experiment：pure allocation identification owner。
- 50M \(2\times2\)：frozen retrofit/co-adaptation diagnostic owner。
- mature OLMo/LLaMA：scale/persistence owner。
- durable paper-facing research notes 放 `paper-2027/research/`；可执行分析放
  `scripts/analysis/`。

### FILE_LEDGER

| Path | Role | State |
| --- | --- | --- |
| `paper-2027/research/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md` | canonical internal report | created |
| `scripts/analysis/full_rope_collision_audit.py` | static theory/numerics | created, untracked at report time |
| `scripts/analysis/attention_fisher_50m_probe.py` | task-sensitive \(2\times2\) | created, untracked at report time |
| `scripts/analysis/base_only_50m_control.py` | base-only controls | created, untracked at report time |
| `AGENTS.md` | active-workspace routing | modified |
| `paper-2027/README.md` | research entrypoint | modified |
| `paper-2027/main.tex` and manuscript sources | active manuscript | unchanged by this audit |

### REJECTED_APPROACHES

- 用 cosine-only collision 代表完整 RoPE pair geometry。
- 由 collision reduction 推出 extrapolation gain。
- 直接优化静态 full-rank/logdet 得到新 schedule；其解退化为 harmonic comb。
- 从 50M post-hoc table swap 推出 intrinsic EVQ PPL cost。
- 把 base 与 interior allocation 宣称为完全正交。
- 根据本轮 Fisher 直接发明 unified loss 或新 frequency optimizer。
- 使用已撤回的 LeRoPE `2.205L` 机制等价叙事。

### RISKY_REGIONS

- `paper-2027/sections/03_theory.tex`：当前仍以 \(\mathcal C_{\rm app}\)/Cosh 为
  主骨架；大改时应替换而不是叠加，并准确限定已证明 exact retrofit theorem。
- `paper-2027/sections/02_related.tex`：LeRoPE 定位当前安全；不要从旧内部 note
  复制撤回内容。
- `paper-2027/sections/04_experiments.tex`：不要把 exact-range、mature adaptation
  和 50M retrofit 合并成一个因果层级。
- `scripts/analysis/`：新脚本当前是内部诊断；提交前需清除机器路径、checkpoint
  identity 和非匿名输出。
- Page budget：正文已用满 8 页；新理论必须替换旧低杠杆内容，而不是叠加。

### VERIFICATION_CHECKS FOR NEXT AGENT

1. 读本报告后再提新 claim；列出它映射到哪个 theorem/evidence owner。
2. 区分 `PROVED / NUMERICALLY VERIFIED / EMPIRICAL / HYPOTHESIS / OPEN`。
3. 修改正文前确认 `paper/` tracked diff 为空。
4. 修改数字时从 owner/JSON 重算，不从本报告手抄后直接宣称完成。
5. 编译仅用于格式验证，不能替代证据验证。
6. 任何新训练先写 concern、最小缺口、命令、预算、owner 和 stop condition，
   并取得用户明确授权。
