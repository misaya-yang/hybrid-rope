# RoPE 频谱范围、分配形状与训练—推理映射

日期：2026-07-24

用途：面向 rebuttal 的定向理论结论与单张 RTX 5090 最小验证方案

状态：理论与已有结果已核验；下述新实验尚未运行

## 0. 核心判断

最重要的结论不是“EVQ 应该与更大的 base 叠加”，而是：

> RoPE 实际只看到离散 log-frequency 向量
> \(x_k=-\log\omega_k=(\log b)\phi_k\)。如果不先固定其端点和跨度，
> `base` 与 exponent/quantile warp 在参数上不可识别。当前 raw
> EVQ-Cosh 的 \(\tau\) 同时改变了频谱端点、log-span 和内部非均匀
> spacing；已有 `EVQ+FMRoPE` 因此不是干净的 “shape + range” 组合实验。

由此得到三个层级不同的判断：

1. **严格成立。** 单个 base 只能缩放几何表的线性 log-frequency
   坐标，不能复现非几何表的归一化间距；但这只证明两张表不等价，
   不证明非几何表在语言模型上更好。
2. **已有经验支持。** 在相同 Std-RoPE span 下，native EVQ、
   exponential 和 attention-derived two-band 均能在部分长度优于
   Std-Geo；这已经说明 interior allocation 不是纯粹的端点效应。
   但现有实验没有与“每个 shape 都重新调过的最佳 geometric
   base/range envelope”比较，因此尚不能声称 base/range 已被排除。
3. **当前最合理、可证伪的联合设计。** 用端点锚定的 normalized shape
   \(s_\eta\) 表示通道内分配，用独立的 \(a_T,R_T\) 表示目标长度的
   log-frequency offset/span；训练与推理保持同一个 \(s_\eta\)，只将
   \(a_T,R_T\) 随目标长度变换。若需要 YaRN，则另加同一组、按通道固定的
   correction coefficients，而不能把非几何表上的 YaRN-derived
   operator 称为 official YaRN。

这意味着当前 rebuttal 最稳妥的表述应是：

> 现有结果支持 “frequency allocation shape 是一个真实设计轴”，但不支持
> “raw EVQ 优于 target-aware range scaling” 或 “Cosh 是最优 shape”。
> 是否能在强 range baseline 上提供独立增益，必须由下述 matched-range
> 实验决定。

## 1. 实现与证据边界

### 1.1 真实频率身份

令 \(K=d_{\rm rot}/2\)，\(B=\log b\)，\(x_k=-\log\omega_k\)。
当前仓库中的真实定义是：

| 方法 | 真实公式 | 说明 |
| --- | --- | --- |
| Std-Geo | \(u_k=k/K,\ x_k=Bu_k\) | native endpoint RoPE |
| Paper-Geo | \(u_k=(k+\tfrac12)/K,\ x_k=Bu_k\) | 投稿主链路的 midpoint Geo |
| raw EVQ-Cosh | \(u_k=(k+\tfrac12)/K,\ x_k=B\Phi_\tau(u_k)\) | 与 Paper-Geo 共享 midpoint quantizer |
| Cosh warp | \(\Phi_\tau(u)=1-\tau^{-1}\operatorname{asinh}((1-u)\sinh\tau)\) | \(\tau\to0\) 返回 Paper-Geo |

代码锚点为 `scripts/lib/rope/schedules.py:86-121` 和
`rebuttal/rebuttal_0723/experiments/geo_rope_contract.py:45-120`。本次还直接从历史
Git 对象读取了两份实际保存的 `inv_freq.npy`，不是根据论文文字反推：

| 历史 artifact | 前四通道 | 后四通道 | float32 tensor SHA-256 |
| --- | --- | --- | --- |
| Paper-Geo, \(b=500K,d=64\) | 0.8146172, 0.5405810, 0.3587302, 0.2380538 | 8.4015e-6, 5.5752e-6, 3.6997e-6, 2.4551e-6 | `88654f1fe2a414d38b1cc7e5a1c0e119e168eaa1118934d865b5b5004b858139` |
| EVQ, \(\tau=5,b=500K,d=64\) | 0.9595150, 0.8816264, 0.8077777, 0.7378857 | 3.0333e-3, 1.2662e-3, 3.4255e-4, 2.6861e-5 | `1749296c29fa0662b0fae9ecabcb9ace39146fef611556f550e7ac260d9eaff6` |

这些值和
`rebuttal/rebuttal_0723/theory_results/FREQUENCY_DEFINITION_MANIFEST.json:6-143`
完全一致。Primary-I 的历史训练代码也使用同一 midpoint builder，但其
checkpoint bytes 不在仓库中；该组只能依赖训练 lineage 和结果记录，不能声称
做过当前 checkpoint 的逐字节复核。

Paper-Geo 与 Std-Geo 的关系尤其简单：

\[
\omega_k^{\rm Paper}
=b^{-(k+1/2)/K}
=b^{-1/d}\omega_k^{\rm Std}.
\]

它们有相同的均匀 log-spacing；Paper-Geo 只是将所有频率统一乘以
\(b^{-1/d}\)，即所有波长统一乘以 \(b^{1/d}\)。它不是另一种 allocation
shape，也不存在一个 Std-Geo base 能同时复现它，因为 Std-Geo 的
\(\omega_0\) 对任意 base 都恒为 1。

### 1.2 一手文献所定义的不同操作

原始 RoPE 将每个二维通道对变成一个频率，并使 attention logit 只依赖相对
位置；标准表为几何频率
([RoFormer](https://arxiv.org/abs/2104.09864))。相关长上下文方法在
\(x=-\log\omega\) 坐标中的作用并不相同：

| 操作 | log-frequency 作用 | 是否保持 normalized shape |
| --- | --- | --- |
| 改 base \(b\to b'\) | \(x_k\to(\log b'/\log b)x_k\) | 是 |
| Position Interpolation，scale \(s\) | \(\omega_k\to\omega_k/s\)，即 \(x_k\to x_k+\log s\) | 是 |
| FMRoPE | Std-Geo 训练 \(b=L_{\rm tr}\)，推理将 \(b\) 调到目标长度 | 是；本地实现见 `../experiments/fmrope_125m_l256/protocol.py:152-158,235-252` |
| YaRN | \(\omega'_k=\omega_k[\mu_k+(1-\mu_k)/s]\)，另有 attention `mscale` | 否；是频率依赖的 by-parts operator |

PI 的位置缩放见
[Chen et al.](https://arxiv.org/abs/2306.15595)；FMRoPE 的
\(b=L_{\rm tr}\) 和推理 target-base 调整见
[Oka et al., ICLR 2026](https://openreview.net/forum?id=PR1PPxvG9Q)。
截至本轮核查，没有找到作者公开的 FMRoPE 实现；因此下文把仓库版本严格
称为 **paper-faithful local protocol**，不称为 official reproduction。
YaRN 明确保留高频局部信息、对低频做更强插值，并另做 attention
temperature scaling
([paper](https://arxiv.org/abs/2309.00071),
[official code](https://github.com/jquesnelle/yarn))。仓库已将 native
endpoint grid 上的 official equations 与任意非几何表上的
YaRN-derived/generalized control 分开
(`scripts/lib/rope/official_yarn.py:1-18,95-154,167-237`)。
投稿 Primary-I 中标作 “YaRN” 的历史行实际是 repository-defined
fixed-index smooth-ramp scaler；它不能作为 official YaRN interaction
证据。本文以下提到的 official YaRN 均指上述独立核验的 equations 路径。

LongRoPE/LongRoPE2 对每个维度搜索不同缩放因子，是外部证据表明 scalar
base 不是唯一有用的自由度；它们是 post-hoc/continuation methods，不能直接
证明 EVQ 的训练期 shape
([LongRoPE](https://arxiv.org/abs/2402.13753),
[LongRoPE2](https://arxiv.org/abs/2502.20082))。关于 base 对可达长度的
作用，可参见
[Base of RoPE Bounds Context Length](https://arxiv.org/abs/2405.14591)。

## 2. 统一模型与必要推导

### 2.1 RoPE 真正优化的是一组有限 Fourier basis

对相对距离 \(\Delta=m-n\)，一层一头的 RoPE logit 可写为

\[
\ell(\Delta)
=\sum_{k=0}^{K-1}
\left[A_k\cos(\Delta\omega_k)+C_k\sin(\Delta\omega_k)\right],
\]

其中 \(A_k,C_k\) 由 token、layer/head 以及训练后的 Q/K 权重决定。因此：

- \(b,\phi_k\) 只通过乘积 \(x_k=(\log b)\phi_k\) 进入频率表；
- 最优表依赖训练后的通道幅值、任务和 distance prior，不存在
  objective-free 的 universal optimum；
- 训练期改变频率表会改变模型学习的 \(A_k,C_k\)，推理期大幅更换 shape
  不是无害的重参数化。

FMRoPE 观察到 frequency band 在训练早期形成并在后续 context extension
中持续存在，这与上述 channel co-adaptation 一致；它不等价于一个
shape-optimality 定理。

### 2.2 base 与 shape 的可识别分解

对任意单调离散表，定义

\[
a=x_0,\qquad R=x_{K-1}-x_0,\qquad
s_k=\frac{x_k-x_0}{R}.
\]

则

\[
\boxed{x_k=a+R\,s_k,\qquad s_0=0,\ s_{K-1}=1.}
\]

- \(a,R\) 是实际 sampled spectrum 的 offset 与 span；
- \(s=(s_0,\ldots,s_{K-1})\) 才是消除 affine range 后的 allocation
  shape；
- 对所有 geometric base，\(s_k=k/(K-1)\)，相邻 normalized gaps
  恒相等，离散二阶差分恒为零；
- 任何二阶差分非零的 \(s\) 都不可能由单个 base 复现。

这给出“base 不能包办 shape”的**数学判据**。但 metric 上某个 base
仍可能与非几何表打平或更好，所以性能判据必须与最佳 base envelope 比较。

### 2.3 当前 \(\tau\) 确实混合了 range 与 shape

对 \(u\in(0,1),\tau>0\)，`sinh` 的严格凸性给出

\[
\sinh((1-u)\tau)<(1-u)\sinh\tau
\Longrightarrow
\operatorname{asinh}((1-u)\sinh\tau)>(1-u)\tau,
\]

因此

\[
\Phi_\tau(u)<u.
\]

所以 raw EVQ 相对同一 midpoint Geo 将每个 sampled frequency 都推向
更高频；同时

\[
a_\tau=B\Phi_\tau(u_0),\quad
R_\tau=B[\Phi_\tau(u_{K-1})-\Phi_\tau(u_0)],\quad
s_{\tau,k}=
\frac{\Phi_\tau(u_k)-\Phi_\tau(u_0)}
{\Phi_\tau(u_{K-1})-\Phi_\tau(u_0)}
\]

三者都随 \(\tau\) 变化。\(\tau\) 因而不是 pure-shape 参数。

理论上还存在两个不同的 \(\tau\) 对象：给定论文凸 surrogate 的
\(\tau_{\rm surr}=\sqrt{\beta/\alpha}\) 决定其条件最优 Cosh density；
部署式 \(d/\sqrt L\) 则是在该 family 上选 operating point 的经验规则。
前者只对写明的 surrogate 严格，后者不是 exact attention/LM optimum，
二者不能因为使用同一个符号就视为同一推导。

已有 \(L_{\rm tr}=128\) sweep 中，\(\tau=5\) 的 selection NLL 为
5.9898，经验规则 \(64/\sqrt{128}=5.657\) 为 6.0017，差 0.0119
NLL；这支持它是 Cosh family 内的 basin selector，但不消除上述
range/shape 混合，也不把它升级为 target-aware 或全局最优公式
(`EXPERIMENT_REPORT_20260724.md:56-77`)。

在实际 \(K=32,b=500K,\tau=4\) 表上：

| | Paper-Geo | raw EVQ \(\tau=4\) |
| --- | ---: | ---: |
| \(x_0\) | 0.2050 | 0.0516 |
| \(x_{31}\) | 12.9173 | 11.7628 |
| realized log-span | 12.7123 | 11.7111 |
| 最高 sampled \(\omega\) | 0.8146 | 0.9497 |
| 最低 sampled \(\omega\) | 2.455e-6 | 7.789e-6 |

也就是说，raw EVQ 在改变内部 spacing 的同时，将 log-span 缩短约
7.9%，最低 sampled frequency 提高约 \(3.17\times\)。

### 2.4 为什么已有 EVQ+FMRoPE 没有协同

已有组合的训练/推理公式是
\(\omega_k=b^{-\Phi_4(u_k)}\)，并在推理时将
\(b=256\) 换成 \(b=L_{\rm eval}\)
(`../experiments/fmrope_evq_combo_l256/protocol.py:136-215`)。它与纯 FMRoPE 的区别不只
是 shape：

- 纯 FMRoPE 使用 Std endpoint grid；组合使用 midpoint EVQ；
- 在 \(b=256\) 时，纯 FMR 的 log-span 为 5.3719，组合只有 4.9488；
- 组合的最慢频率比纯 FMR 高 \(1.49\times\)；到
  \(b=2048\) 时该比值增至 \(1.74\times\)；
- \(\tau=4\) 是 raw \(b=500K,L_{\rm tr}=256\) 的 operating default，
  没有在 \(b=256\) 的 anchored range 下重新定义。

因此这次组合实际测试的是“同时改变 quantizer、range 和 shape 的一个点”，
不是在 FMR 最佳 range 上加入独立 shape。结果在 512/1K 相对纯 FMR
分别差 `+0.075/+0.042 NLL`，2K 只好 `-0.037`
(`EXPERIMENT_REPORT_20260724.md:127-145`)；最合理解释是收益重叠加上
参数混杂，而不是已经证明 allocation 与 range 不可组合。

另一个限制来自目标距离：相同 span/RMS 下，Cosh 在较短距离有竞争力，
exponential 在部分中距离更好，attention-derived two-band 在 4K/8K
最好 (`EXPERIMENT_REPORT_20260724.md:207-267`)。单个固定 Cosh shape
没有理由在所有 target length 上支配。

## 3. 建议的联合参数化

### 3.1 Range-anchored shape

令 \(r_k=k/(K-1)\)。对 Cosh 定义端点锚定 shape：

\[
s_{\tau,k}^{\rm cosh}
=\frac{\Phi_\tau((k+1/2)/K)-\Phi_\tau(1/(2K))}
{\Phi_\tau(1-1/(2K))-\Phi_\tau(1/(2K))}.
\]

于是 \(s_{\tau,0}=0,s_{\tau,K-1}=1\)，且 \(\tau\to0\) 精确返回
\(r_k\)。再定义

\[
\boxed{
\omega_k(L_{\rm tr},T,\eta)
=\exp[-a_T-R_Ts_{\eta,k}].
}
\]

这里有意采用 Std-Geo 的 endpoint 坐标：它与本地 FMR protocol 和
official YaRN 的 reference grid 对齐。Paper-Geo 仍是投稿事实基线，但不再
作为检验独立 shape effect 的坐标系。

最小、可解释的选择是：

\[
a_T=0,\qquad
R_T=\frac{K-1}{K}\log b_T,\qquad
b_{\rm tr}=L_{\rm tr},\quad b_T=cT.
\]

- \(s_\eta\) 在训练和推理保持不变；
- \(L_{\rm tr}\) 决定训练 range；
- 预先声明的目标集合决定 shape 的选择标准；
- \(T\) 只通过 \(b_T\) 改变推理 range；
- \(c=1\) 是本地 paper-faithful FMR mapping，\(c\) 的小网格用于给
  geometric baseline 一个公平的 best-base envelope。

这不是“新方法已成立”，而是当前问题最小的 identifiable
parameterization。若目标长度确实偏好不同 shape，正确做法是**在训练前**
针对目标集合选择 \(s_\eta\)，而不是对一个已训练 checkpoint 在每个目标长度
随意更换 shape。

在这个参数化中，\(\tau=d/\sqrt{L_{\rm tr}}\) 最多保留为一个预先固定的
shape candidate；它不再负责 range。当前也没有理论或数据足以写出可信的
\(\tau(L_{\rm tr},T)\) 闭式公式。若预先知道目标长度集合，应按该集合的
selection aggregate 选择一个训练期 shape；若目标未知，则只能选择一个
robust fixed shape，不能在 rebuttal 中宣称 target-specific optimum。

### 3.2 与 YaRN 的受控组合

设 extension ratio \(\rho=T/L_{\rm tr}\)，official YaRN 在 native grid
上给出固定通道系数 \(\mu_k(\rho)\)，则

\[
x_{T,k}
=a_{\rm tr}+R_{\rm tr}s_{\eta,k}
-\log[\mu_k(\rho)+(1-\mu_k(\rho))/\rho].
\]

归因实验必须对所有 shape 使用同一组 \(\mu_k\) 和同一 `mscale`。Geo
一臂可称 official YaRN equations；非几何臂只能称
**shared-index YaRN-component control**。若改为按每张表的 virtual
wavelength 重算 ramp，operator 本身也变了，不能再用于 pure interaction
归因。

## 4. 单张 RTX 5090：两组最高信息量实验

### 实验 A：matched range × allocation shape（唯一新训练组）

**问题。** 在训练 range、推理 target-range、初始化、数据和预算完全一致时，
非几何 interior spacing 能否超过最佳 geometric base envelope？

**模型与数据。** 直接复用现有 `fmrope_125m_l256` 的 151,898,880 参数模型、
FineWeb-Edu tensors、训练顺序、held-out shard、tail-128 NLL 和 anchor
构造逻辑。保留已有 256–2K endpoints；4K/8K endpoints 必须在看结果前从
同一 held-out shard 预注册，并与 selection endpoints 分离。
\(L_{\rm tr}=256,K=32\)，每臂 100M tokens。

**三个训练臂。**

| arm | normalized shape \(s_k\) | 训练 range |
| --- | --- | --- |
| A0 Geo | \(k/(K-1)\) | \(a=0,\ R=\frac{31}{32}\log256\) |
| A1 Anchored-Cosh | 上式，固定 \(\tau=4\) | 与 A0 完全相同 |
| A2 Anchored-Exp | \(s(q)=(e^{\gamma q}-1)/(e^\gamma-1)\) | 与 A0 完全相同 |

A1 是为隔离机制而定义的 range-anchored diagnostic，不是投稿中 raw
EVQ-Cosh 的逐字复现；其正结果支持 allocation-shape mechanism，不能静默
替换成“submitted implementation 已在该 baseline 上获胜”。

A2 的 \(\gamma\) 只用解析 RMS matching 决定，使
\(\operatorname{RMS}(s_{\rm Exp}-r)=
\operatorname{RMS}(s_{\rm Cosh}-r)\)。在 \(K=32,\tau=4\) 下，
该值约为 \(\gamma=3.45449\)；CPU preflight 必须用 float64 重算并写入
schedule receipt，而不是在训练结果后调整。

**推理条件。**

1. fixed train range；
2. literal FMR：\(b_T=T\)；
3. adversarial best-base envelope：
   \(b_T=cT,\ c\in\{0.5,1,2,4\}\)。每个 arm 只能在 16 个 selection
   anchors 上选择一个跨所有外推长度共用的 \(c\)，随后冻结；
4. secondary：A0 使用 official YaRN；A1/A2 使用完全相同 index mask 和
   `mscale` 的 shared-index control。

256 in-domain 指标始终使用 checkpoint 的原始训练表，不参与 \(c\) 的
选择或 retarget。

**指标。**

- 长度：256/512/1K/2K/4K/8K；
- primary：32 个未见 test anchors 上的 paired tail-128 NLL；
- 主 estimand：
  \[
  \bar\Delta_\eta
  =\frac15\sum_{T\in\{512,1K,2K,4K,8K\}}
  [\mathrm{NLL}_{\eta,\mathrm{best}\ c}(T)
  -\mathrm{NLL}_{\rm Geo,\mathrm{best}\ c}(T)];
  \]
- 同时保留每个长度、256 in-domain cost、win count；不得用单个振荡长度
  覆盖 aggregate。

**预算和停止条件。**

- seed 42：3 臂，共 300M tokens；
- 只有当某个非几何臂满足
  \(\bar\Delta\le-0.05\)、至少 4/5 个外推长度同向、任何长度不劣于
  `+0.05 NLL`、且 256 cost 不超过 `+0.02 NLL`，才扩展；
- 扩展时只运行 A0 与 seed-42 胜出的一个非几何臂，seeds 137/256，
  追加 400M tokens；总上限 700M tokens；
- 若 A1/A2 均未过 seed-42 gate，立即停止，不增加 token budget；
- 按 `docs/overview/RTX5090_BLACKWELL_PROFILE.md:37-74` 做 BF16/Flash-only/
  `torch.compile` 丢弃式 probe。不同模型不能套用 MLA throughput；
  第一臂实测 ETA 若使总预算超过 4 GPU-hours，则先停止。合理预期为
  1–3 小时，但以 probe 为准。

**成功标准。**

三 seed 的 \(\bar\Delta\) 全部为负、均值不高于 `-0.05 NLL`、seed-level
paired 95% CI 上界低于 0，且上述 in-domain/per-length gates 保持成立。
此外，它必须击败 **best-base Geo envelope**；只击败 \(c=1\) 不足以证明
base 不能解释收益。

上述标准只足以支持“在 FMR-style target-base mapping 上有独立增益”。
若要在 rebuttal 中进一步说它 survives the strongest tested range
component，还要求同一三 seed方向在 shared-index YaRN control 下保持为负；
否则必须把结论限制在 FMR/base-range，而不能外推到 YaRN。这里依然只是
inference-only equations/component control，不等于 YaRN 论文的完整
continuation-training protocol。

**否定标准。**

- best-base Geo 与所有非几何臂打平或更好；
- 优势在 exact span matching 后消失；
- 仅某个振荡长度为正，aggregate 或 seeds 不稳定；
- “优势”来自某个 range control 的灾难性崩溃。

若只有 A2 成功而 A1 失败，结论是“shape axis 成立但 Cosh family 不是合适
实例”；不得把 A2 的结果写成 EVQ-Cosh 胜利。
即使三 seed gate 全部通过，这仍是 151.9M/100M-token 的机制诊断，不是
production-scale 或 universal long-context claim。

### 实验 B：训练 shape × 推理 shape 的 2×2 映射（不新增训练）

**问题。** 目标长度改变时，应只移动 range，还是可以在现有 checkpoint 上
直接更换 allocation shape？

复用实验 A 的 Geo 与胜出非几何 checkpoints；主分析固定 literal FMR
\(c=1\)，从而两个 checkpoint 在每个长度使用完全相同的
\(a_T,R_T\)，再交叉评估：

| checkpoint training shape | runtime Geo | runtime matched non-Geo |
| --- | ---: | ---: |
| Geo | matched | shape swap |
| non-Geo | shape swap | matched |

长度先用 256/1K/4K/8K 和 16 个 selection anchors；若结论不清楚再读取
32 个 test anchors。主量为

\[
D_{\rm swap}(W_\eta,T)
=\mathrm{NLL}(W_\eta,s_{\eta'\ne\eta},T)
-\mathrm{NLL}(W_\eta,s_\eta,T).
\]

预算为 evaluation-only，目标上限 30 分钟；若 256 上任一 swap 已造成
`>0.10 NLL` 损失，只需再验证一个外推长度即可停止。

解释规则：

- 两个 checkpoint 都以 diagonal/matched shape 最好：说明 training-time
  channel co-adaptation 真实存在，正确映射是“保持 shape、只变 range”；
- 同一个 runtime shape 无论训练 shape 都最好：说明收益更接近
  inference-time schedule，训练期 substrate novelty 变弱；
- 所有差异绝对值 `<0.02 NLL`：在强 range 下 shape 实际无关；
- swap 偶然在单长度获益但不跨长度稳定：不能据此提出 target-dependent
  runtime shape。

该组本身不证明 shape 有增益；它只在实验 A 成功后解释训练—推理映射。

## 5. 哪些结论能进入当前 rebuttal

### 现在即可使用

1. 实际 `inv_freq` 证明 Paper-Geo 是 midpoint uniform log-spacing，raw
   EVQ 的 \(\tau\) 同时改变 range 与 shape。
2. 已有 native matched-span 三 seed结果证明 interior allocation 的 LM
   effect 不是 Paper-Geo midpoint 或端点跨度的假象：
   native EVQ 在 512–8K 均优于 Std-Geo，但 exponential/two-band 在部分
   长度更好。
3. FMRoPE 和本轮 official-YaRN-equations diagnostic 的 target-aware
   scaling 明显强于 raw EVQ；已有 EVQ+FMRoPE 组合不支持协同或性能优越性。
4. “base 无法数学复现非均匀 spacing”可以严格陈述；“非均匀优于最佳
   base”目前不能陈述。

### 只有实验 A 通过后才能使用

> 在相同 train range、相同 target-range operator 且给 geometric
> baseline 重新优化 base multiplier 后，非几何 shape 仍提供稳定增益。

若 shared-index YaRN secondary 也同向，可进一步说 shape benefit survives
a fixed strong range-extension component；不能说这是 non-Geo 上的
official YaRN。

### 失败时应直接收窄

若实验 A 未过 gate，当前方向的可辩护结论应停在：

> frequency allocation 是可区分的训练期设计自由度，并改善 raw
> extrapolation；但我们尚未证明它能在调优的 target-aware base/range
> scaling 之上提供独立实用增益。

这不会否定已有 raw 结果，却会否定“第三轴在强 baseline 上已有增量价值”的
更强版本。

### 仅适合后续研究

- 从 target-specific attention prior 学习 \(s_\eta(L_{\rm tr},T)\)；
- 在 sparse/local attention 下重推 distance prior 和 \(\tau\)；
- 与完整 YaRN continuation、LongRoPE2 或大模型 RULER 联合优化；
- universal optimal allocation、Cosh 的 exact-attention/LM optimality。

这些问题都超出当前两组实验能够识别的范围，不应进入本轮 rebuttal 主张。
