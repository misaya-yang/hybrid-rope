# 《Beyond the Base》ICLR 2027 论文重构与 Codex 执行手册

**版本日期：2026-09-11**  
**适用对象：当前论文仓库、主稿 `main(20260911-210014).pdf` 对应源码、后续补充实验与 ICLR 2027 投稿**  
**核心目标：把已经被多条实验线和防御性写作打散的稿件，重构成一篇问题定义干净、因果分离明确、理论与实证层级清楚的“RoPE exponent allocation”论文。**

---

## 0. 结论先行

### 0.1 这次对问题的理解没有错

截至 2026-09-11，对 RoPE 频率设计、长上下文扩展、频率使用、可学习频率、非几何频率表和位置编码理论工作的系统检索，没有发现一篇已有论文同时完成以下四件事：

1. 将有限 RoPE 频率表明确分解为  
   \[
   x_k=-\log \omega_k=a+Rz_k,
   \]
   其中 \((a,R)\) 表示采样的 log-frequency support，\(z\) 表示归一化的内部指数分配；
2. 在 **最高频、最低频、log-span、rotary pair 数量全部严格相同** 的条件下，只改变排序后的 \(K-2\) 个内部指数；
3. 用配对训练、跨表冻结评估等实验，识别 interior allocation 对学习与外推行为的独立因果影响；
4. 在此问题上给出零额外参数、可审计的解析构造，并研究它与训练权重、运行时范围和 slot assignment 的关系。

因此，正确、可守的核心 novelty 不是：

> 我们首次改变了 RoPE 频率或指数。

而是：

> **我们首次系统地将 normalized interior exponent allocation 从 sampled frequency support、slot assignment 和 extension factor 中分离出来，并把它作为一个独立的 RoPE 设计变量进行定义、因果识别、几何分析和构造。**

这一定义避开了 LongRoPE、YaRN、MrRoPE、LeRoPE、AdaRoPE 等工作的已有贡献，同时准确覆盖当前论文最强的理论与实验资产。

### 0.2 没有 universal SOTA 不是结构性问题

ICLR 2027 官方 Reviewer Guide 明确指出，缺少 SOTA 结果本身不是拒稿理由；审稿人应判断论文是否带来新的、相关的、有影响力的知识。本文的价值不依赖“一个固定 schedule 在所有模型、长度和任务上统治全部方法”，而依赖以下闭环：

\[
\boxed{
\text{clean variable}
\rightarrow
\text{causal identification}
\rightarrow
\text{mechanism}
\rightarrow
\text{principled construction}
\rightarrow
\text{consequential model effects}
}
\]

当前资产已经覆盖这个闭环。剩余任务主要是**重排证据、正确定义 prior-work 边界、删除错误的 SOTA 焦虑和拼接感**，而不是继续无边界搜索新曲线。

### 0.3 当前稿件达到 ICLR 6/10 是合理目标

上一版 EVQ 的三位评分为 4、2、3，平均 3.0；主要否定点是：

- 与 FMRoPE 的区别没有证明干净；
- 模型规模偏小，缺乏 1B–7B 证据；
- allocation shape 与 range、operator capacity、调参效应没有充分分离；
- 缺少更强任务和直接匹配对照。

当前版本已经新增或强化了：

- 严格固定 support、只改 30 个 interior exponents 的三 seed 配对实验；
- 12 configuration × 3 seed 的 fixed-support factorial；
- 50M 和 151.9M 的 weight–table crossing；
- 432M MLA 三 seed；
- 750M full-parameter continuation；
- 1.485B from-initialization / adaptation 证据；
- Llama-3-8B matched LoRA；
- OLMo/Qwen frozen deployment；
- RULER、LongBench natural QA、PPL、complete generation 和 causal source-use；
- full sine–cosine subspace、slow-frequency collapse、exact rank identity；
- 明确区分 exact geometry、surrogate、operating rule 和 downstream measurement。

因此，当前论文已经不是旧版本的“一个 Cosh 启发式 + 小模型诊断”。它已经具备 ICLR borderline-accept / accept 论文应有的工作量和证据形态。

更准确的评分判断是：

- **当前未重构、叙事较乱的 PDF：**容易得到 5，审稿分布可能跨 4–6；
- **完成本手册的重构、正面对比 LeRoPE 等最近工作、清理证据层级后：**6 应成为最合理的中心评分，7 有现实可能；
- 不能把 6 当作数学保证，但现在真正威胁 6 的不是“没有 universal SOTA”，而是**稿件组织、claim 边界和证据呈现失误**。

---

## 1. 论文要研究的对象：锁死定义，不再漂移

### 1.1 标准 RoPE 的隐藏设计选择

设一个 attention head 中有 \(K=d_{\mathrm{rot}}/2\) 个 rotary pairs。标准 RoPE 通常写成

\[
\omega_k=b^{-\phi_k},\qquad \phi_k=\frac{k}{K}.
\]

对任意严格下降的正频率表，写成

\[
x_k=-\log\omega_k=a+Rz_k,
\]

其中

\[
a=x_0,\qquad R=x_{K-1}-x_0,\qquad
z_k=\frac{x_k-x_0}{x_{K-1}-x_0},
\]

并有

\[
z_0=0,\qquad z_{K-1}=1.
\]

于是一个有限频率表至少包含三个彼此不同的对象：

1. **Support / range：**\((a,R)\)，决定最高频、最低频和 log-span；
2. **Interior allocation：**排序后的 \(z_1,\ldots,z_{K-2}\)，决定有限 rotary budget 如何占据 support；
3. **Slot assignment：**置换 \(\pi\)，决定这组频率被安装到哪些已标号 Q/K coordinate blocks。

对任意 geometric table，归一化后总有

\[
z_k=\frac{k}{K-1},
\]

与 base 的具体取值无关。因此，改变 base 改的是 support；在固定 \((a,R,K)\) 下改变 \(z\)，才是纯粹的 interior exponent allocation intervention。

### 1.2 本文的中心问题

全文必须围绕一句话展开：

> **When frequency support and rotary-pair count are fixed, how does the placement of the interior exponents change the positional basis, the weights learned on that basis, and length behavior?**

中文含义：

> 在频率端点、范围和 rotary pair 数量完全固定时，内部指数如何分配，会怎样改变模型获得的位置基、模型在该基上学到的表示，以及长度泛化行为？

### 1.3 四个必须持续分离的变量

整篇论文不能再把下面四件事混写：

\[
\boxed{
\text{support}
\neq
\text{allocation}
\neq
\text{assignment}
\neq
\text{compatibility}
}
\]

- **Support：**频率表覆盖哪些尺度；
- **Allocation：**有限频率点在这个 support 内如何铺设；
- **Assignment：**这些频率放入哪些已有 Q/K blocks；
- **Compatibility：**冻结权重是否与新安装的频率表协调。

上下文扩展还增加第五个外部量：

\[
s=\frac{L_{\mathrm{target}}}{L_{\mathrm{native}}}.
\]

任何 \(s\)-conditioned 方法都必须写出倍率，例如：

\[
\operatorname{YaRN}(s),\qquad MR(s),\qquad BM(s).
\]

特别地：

\[
MR(1)=\text{native RoPE}.
\]

禁止把 “MrRoPE-Pro” 当作一个不带倍率的固定频率表。`MR(4)` 和 `MR(16)` 是两张不同的部署表，回答不同的 target-length 问题。

---

## 2. 相关工作审计：谁真正与我们重合

### 2.1 判定重合的操作化标准

判断 prior work 是否与本文核心 novelty 重合，不看它“最后有没有改变频率”，而看它是否满足以下列：

| 判据 | 本文核心研究 |
|---|---|
| 是否显式区分 support 与 normalized interior allocation | 是 |
| 是否固定最高/最低频率和 log-span | 是 |
| 是否只改变排序后的 interior exponents | 是 |
| 是否把 allocation 当成待研究的独立变量 | 是 |
| 是否进行 paired/co-controlled training identification | 是 |
| 是否区分 allocation 与 slot assignment | 是 |
| 是否研究 weight–table co-adaptation | 是 |
| 是否提供零参数解析 allocation | 是 |
| 是否要求给定 extension factor \(s\) | 学习期主线不要求；部署应用要求 |
| 是否以 universal SOTA 为主 claim | 否 |

只要一篇方法改变了 \(\omega_k\)，就总能在事后改写成某个 exponent displacement；这不代表它提出了本文的问题。

### 2.2 核心文献矩阵

| 工作 | 真正研究的问题 / 干预 | 是否固定 support 只改内部 \(z\) | 与本文的关系 |
|---|---|---:|---|
| RoFormer / RoPE | 定义旋转位置算子和 geometric frequency grid | 否 | 基础对象 |
| Position Interpolation | 将 position index 除以 \(s\) | 否 | 全频统一 context extension |
| YaRN | 给定 \(s\)，按 high/mid/low bands 混合 extrapolation 与 interpolation，并改变 attention scale | 否 | extension baseline；不是 learning-time allocation identification |
| Scaling Laws of RoPE Extrapolation | base、训练长度和临界维度如何影响外推 | 否 | support/range 理论背景 |
| Base of RoPE Bounds Context Length | base 与理论 context bound | 否 | support/range，不研究固定端点内部铺点 |
| LongRoPE | 搜索 per-dimension scaling factors 和 nonuniform position interpolation | 否 | 说明非均匀频率变换有用；目标是给定长度扩展，且搜索会改 support |
| Resonance RoPE | 将 wavelength 调整到整数/共振结构，常与 YaRN 组合 | 否 | 离散周期设计，不是 fixed-support exponent allocation |
| Round and Round We Go / p-RoPE | 分析高低频作用，移除/停止旋转部分最低频率 | 否 | 频段功能和 partial RoPE；改变 operator/rotary fraction |
| LongRoPE2 | needle-PPL 引导的搜索 + mixed-window training，追求 near-lossless extension | 否 | 特定 target length 的强工程扩展 |
| FoPE | 每个维度使用 Fourier components，处理周期外推 | 否 | 改变 positional operator |
| HoPE | 将部分慢频旋转替换为 position-independent components | 否 | 改变 operator |
| DoPE | 用 entropy / denoising 指标选择性衰减或替换位置分量 | 否 | feature/head 选择，不是排序后的固定 support allocation |
| FMRoPE / Frequency Bands in RoPE | 用训练长度选择 base / learned band；normalized exponents 仍等距 | **否** | 最适合作为 support 与 allocation 分离的对照 |
| Frequency Entropy | 测量模型实际使用哪些频率，进行 frequency-level probing | 否 | “learned usage”背景，不是“supplied allocation” |
| MrRoPE | 用 mixed-radix 解释 extension；给定 \(s\) 改中频累计 shift | 否 | 与 frozen BM application 接壤，不与 fixed-support learning-time core 重合 |
| CoPE | soft-clip 低频，联合 OOD 与 semantic modeling | 否 | 改 operator / spectral taper |
| LeRoPE | 每个频率学习一个 scalar，52M–2.5B scratch training | 否；support 和内部位置共同学习 | **最接近的 conceptual neighbor**，必须正面比较 |
| AdaRoPE | 每个 head 学习 frequencies 和 attention scaling | 否；head-specific、learned、extension-oriented | 说明统一 schedule/scale 不足；与 assignment / compatibility 邻近 |
| How Data Shapes RoPE Frequency Usage | 解释训练数据 dependency scale 如何决定已训练模型的 frequency usage | 否 | 与本文互补：本文研究 supplied basis，它研究 learned use |
| Anti-Periodic / Möbius RoPE | 在部分 heads 使用 half-integer harmonic anti-periodic ladder，研究边界条件和 retrieval reliability | 否 | 一个明确的非几何固定频率表；因此不能声称“首个非几何 schedule”，但其科学问题不同 |
| Deconstructing Positional Information | 分析 positional mechanism 如何进入 logits 和 training bias | 否 | 理论背景 |
| RoPE Distinguishes Neither Positions Nor Tokens... | 证明长上下文中位置/内容辨识的内在局限 | 否 | 理论动机 |
| VideoRoPE / multimodal RoPE variants | 在视频、多轴输入中分配时间/空间频率 | 通常否 | 跨模态旁证，主文不必展开 |

### 2.3 最接近的三个威胁及正确应对

#### A. LeRoPE：最需要认真处理

LeRoPE 明确指出 geometric schedule 不应被固定，并为每个频率学习一个 scalar。它验证了 52M–2.5B 模型，属于最接近的 conceptual overlap。

本文不能声称：

- 首次认为 RoPE frequency schedule 可优化；
- 首次使用非几何频率；
- 首次证明 learned/flexible frequency 比 geometric 好。

但本文可以并且应当强调：

1. LeRoPE 直接对各频率做 task-loss-driven learning，support、interior spacing 和可能的频率次序/范围共同变化；
2. 本文先定义并识别 **normalized interior allocation under exact fixed support**；
3. 本文的 paired experiment 不依赖频率学习优化是否成功，直接识别这个变量的因果作用；
4. 本文给出零额外参数的 analytic table；
5. 本文发现 weight–table co-adaptation，并通过 crossing 将“table quality”与“weights learned on the table”分开；
6. 本文的 full-pair geometry 是 supplied positional basis 的结构分析，而不是只观察学习后频率。

**建议补一个 constrained learnable-\(z\) baseline，见第 8 节。** 这会把 LeRoPE 从 novelty 威胁变成支持本文问题定义的 strongest comparator。

#### B. Anti-Periodic / Möbius RoPE：阻止过宽 claim

该方法确实构造了一种非 geometric 的固定频率 ladder，并在 matched controls 中分析频带和边界条件。它说明“固定的非几何频率表能够改善特定行为”早已有直接实例。

但它：

- 研究的是 anti-periodic boundary condition / holonomy；
- 在部分 heads 安装 harmonic ladder；
- 不以相同 geometric endpoints 下的 normalized interior allocation 为研究对象；
- 不建立一般的 support–allocation 分解；
- 不做本文类型的 sorted interior fixed-support paired identification。

因此它不击穿本文，但要求本文删除所有“first non-geometric RoPE table”式表述。

#### C. MrRoPE：只与部署应用局部重合

MrRoPE 的准确对象是 \(MR(s)\)：

\[
\omega_k'=\omega_k s^{-m_k},
\]

其中 \(s\) 是预先指定的 extension factor，\(m_k\) 是中频 transition 的累计 shift。论文沿用 YaRN 的 band boundaries 和 attention rescaling，并明确将其相对 YaRN 的区别放在 intermediate-dimension extrapolation strategy 上。

因此：

- \(MR(4)\) 回答 4× extension；
- \(MR(16)\) 回答 16× extension；
- \(MR(1)\) 是 native identity；
- 其论文在 Llama3 8K→128K 上比较的是同为 \(s=16\) 的方法，没有给出 native \(s=1\) 的 8K 对照，因此不能据此证明 native preservation；
- 它没有识别 fixed-support learning-time interior allocation。

本文的 BM 是对 \(MR(s)\) 中 transition profile 的一个 deployment-side improvement，因此要清楚归为**应用与后果**，不能让它抢占论文主线。

### 2.4 最终 novelty 判词

建议在 Introduction / Related Work 使用下面这种强度：

> **A RoPE table makes two independent choices: which log-frequency interval is sampled and how a finite number of rotary pairs are allocated within that interval. Prior work has changed bases, learned frequencies, selected bands, or transformed frequencies for a target context length. We instead isolate normalized interior allocation by holding the sampled endpoints and pair count exactly fixed, and study its effect on the supplied positional basis, learned weights, and length behavior.**

建议使用：

> **To our knowledge, this is the first paired training study in which the sampled RoPE frequency endpoints and pair count are held fixed while only the sorted interior exponents are changed.**

不建议使用：

> We are the first to modify/optimize/learn RoPE frequencies.

也不建议使用：

> Previous work only changes the base.

---

## 3. 当前论文已经拥有的核心资产

### 3.1 资产一：最干净的因果识别

151.9M paired training：

- \(K=32\)；
- \(L_{\text{train}}=256\)；
- 三个独立 training seeds；
- architecture、initialization、token order、optimizer、schedule、499,974,144-token budget 全匹配；
- 最高频、最低频、log-span 完全相同；
- 唯一变量是 30 个 interior exponents。

固定训练 support 时，Cosh-minus-Geo tail NLL：

- 1×：+0.026；
- 2×：−0.281；
- 4×：−0.176；
- 8×：−0.146。

三个 seeds 在全部 OOD lengths 上同方向。

当每个 evaluation length 都重新 retarget support 后，排序反转：

- 2×：+0.060；
- 4×：+0.227；
- 8×：+0.460。

这个实验同时证明：

1. interior allocation 是独立变量；
2. allocation 不能脱离 operating range 谈“绝对好坏”；
3. “不存在 universally optimal table”不是失败，而是机制发现。

### 3.2 资产二：跨配置 shape 证据

50.9M factorial：

- 2 bases × 2 training lengths × 3 head dimensions；
- 12 configurations × 3 seeds；
- 全部保持 sampled endpoints 和 log-span；
- Geo、三个 Cosh strength、matched exponential。

结果：

- EVQ 0.75×：8/12 configs 改善；
- formula EVQ：7/12；
- EVQ 1.25×：10/12；
- matched exponential：9/12；
- formula Cosh 与 matched exponential 的差异未决。

它支持的不是“Cosh 函数唯一正确”，而是：

> **多个不同的 nonuniform allocation 在固定 support 下都可以系统性改变甚至改善 OOD behavior；具体最优形状依赖 configuration。**

这是对问题本身非常有价值的证据，应该进入主文，而不是被当成“EVQ 没有全胜”的负面结果。

### 3.3 资产三：full sine–cosine subspace geometry

每个 rotary pair 对一个固定 content pair 提供的 positional object 是

\[
V_\omega=\operatorname{span}\{\cos(\omega\Delta),\sin(\omega\Delta)\}.
\]

使用 whitened cross-Gram 的 canonical correlations，可以 phase-invariant 地比较两个完整二维子空间。本文进一步给出：

- mean pairwise overlap 与 block-whitened Rényi-2 effective rank 的 exact identity；
- 当 \(\omega L\to0\) 时，
  \[
  V_\omega\to\operatorname{span}\{1,\Delta\};
  \]
- 对 \(b=500K,K=64,L=4096\)，23 个 slow pairs、46 个坐标维度的 block-whitened effective rank 约为 2。

这不是 downstream accuracy bound，而是一个精确的 supplied-basis geometry。它解释了为什么 geometric log-spacing 并不等价于均匀分配 positional directions。

### 3.4 资产四：weight–table co-adaptation

50M crossing：

| Trained weights | Geo runtime | Cosh runtime |
|---|---:|---:|
| Geo | 7.14 | 76.20 |
| Cosh | 23.05 | 7.16 |

151.9M crossing 也显示，每组 weights 都偏爱从自己的 training allocation 导出的 factor-four runtime table。

这建立了一个重要结论：

> 一个静态 table metric 不能独立决定 mature model performance；模型权重会与训练时 frequency basis 共适应。

这也是本文区别于纯静态频率搜索论文的关键贡献。

### 3.5 资产五：EVQ-Cosh 是解析构造，而不是全部 claim

EVQ-Cosh 从明确的 convex density design criterion 得到唯一 minimizer：

\[
\rho_\tau(\phi)
=
\frac{\tau\cosh[\tau(1-\phi)]}{\sinh\tau}.
\]

其 inverse-CDF quantiles 产生有限频率表。正确表述是：

- Cosh 是这个 surrogate functional 的 unique optimum；
- surrogate 是显式 design prior，不是 LM loss；
- 具体 \(\tau\) 和 finite-grid convention 是 operating choice；
- downstream 效果由实验测量，不从 surrogate optimality 自动推出。

这四层必须与当前 Table 29 一样持续分开。

### 3.6 资产六：足以证明实际意义的模型结果

#### 432M MLA，三 seeds，只有 16 rotary pairs

| Method | PPL@8K | PPL@16K |
|---|---:|---:|
| Geo | 35.4 | 138.8 |
| EVQ-Cosh | 35.8 | 95.6 |
| Geo + same wavelength blend | 35.5 | 117.9 |
| EVQ-Cosh + same blend | 35.8 | 71.1 |

这是主文最漂亮的 downstream result：

- training-length 几乎不变；
- 16K 明显改善；
- 三 seed 一致；
- 在现代 MLA / scarce rotary budget 架构中意义明确；
- 与同一个 inference-time operator 可组合。

#### 750M full-parameter continuation

- 4K PPL：22.0 vs 22.3；
- 16K PPL：45.1 vs 24.4；
- 8K strict autoregressive passkey：0% vs 77.5%。

#### Llama-3-8B matched LoRA

- 8K：6.82 vs 10.07；
- 16K：108.96 vs 24.07；
- 32K：991.48 vs 127.91。

它是明显的 length-transfer trade-off，不要包装成 near-lossless；作为“allocation 会改变 adaptation frontier”的证据即可。

#### Frozen deployment

- OLMo natural QA：BM(s=4) 相对 MR(4)，task-equal F1 21.62→25.44，五个任务都提升；
- OLMo 16K six-task：2.78→51.32；
- 但 Qwen 3B/7B 在 128K 上更偏好 MR(4)，必须保留，作为 checkpoint/length dependence；
- Qwen 0.5B 64K 相对 YaRN 的 +6.09 是 **table × amplitude joint configuration**，不能当纯 allocation gain。

---

## 4. 当前稿件为什么显得乱

当前主文把以下内容几乎平级地压在前 8 页：

1. fixed-support causal identification；
2. full-pair geometry；
3. EVQ variational derivation；
4. scratch / continuation / MLA / 8B adaptation；
5. frozen OLMo/Qwen deployment；
6. BM vs MrRoPE；
7. table placement；
8. data-dependent profile diagnostic；
9. related work。

结果不是工作量少，而是**每个结果都在抢论文中心**。

### 4.1 当前最严重的组织问题

#### 问题 A：学习期主线和冻结部署主线被写成“两篇并列论文”

这会让审稿人问：

- EVQ-Cosh 和 BM 到底是不是同一个方法？
- 论文是在研究 pretraining allocation，还是 test-time extension？
- geometry 到底服务哪一个结果？
- 为什么 abstract 同时列 432M、BM、Qwen+YaRN 三组不同协议？

正确关系应当是：

\[
\text{fixed-support allocation science}
\Rightarrow
\begin{cases}
\text{learning-time analytic construction}\\
\text{frozen-deployment consequence}
\end{cases}
\]

学习期是主线，部署是“同一分解在 mature checkpoint 中的后果与应用”。

#### 问题 B：最强证据没有形成明确等级

当前证据应分成：

1. **Identification：**fixed support + paired seeds；
2. **Mechanism：**subspace geometry + crossing；
3. **Construction：**EVQ-Cosh；
4. **Consequences：**MLA / continuation / 8B / frozen deployment。

不能把每个数据集都写成一个新贡献。

#### 问题 C：过多协议细节进入主文

以下内容应主要放 appendix：

- 12-profile exploratory fit；
- profile tensor hashes；
- placement 的全部公式和每 task 数值；
- 1.485B trainer implementation mismatch 的细节；
- Video DiT 完整实验；
- Qwen 小 panel 的全部 gain combinations；
- BM 离散 Euler 推导细节；
- 全量 protocol map。

主文只留审稿人决定 accept/reject 必须知道的内容。

#### 问题 D：防御性限定仍然过多

应保留真实 caveat，但不要每句话都预防性认错。原则是：

- 在 claim 第一次出现时准确限定；
- 在 limitations 统一列出范围；
- 不在每个结果后重复“not universal / not guaranteed / only under this protocol”。

---

## 5. 新的论文故事

### 5.1 一句话故事

> **RoPE’s base determines where a frequency table lies, but not how a finite rotary budget is allocated inside that range. By isolating this hidden design coordinate, we show that interior exponent allocation changes the supplied positional basis, the weights learned on it, and length behavior; we then derive and evaluate an analytic allocation and show how the same distinction informs frozen deployment.**

### 5.2 三段逻辑

#### 第一段：发现一个被混在 base 里的独立变量

传统讨论经常把 RoPE table 等同于 base。本文指出，一个 finite table 还包含 interior allocation：

\[
(a,R,z,\pi).
\]

然后用 exact fixed-support paired training 证明 \(z\) 不是无关参数。

#### 第二段：解释为什么这个变量会产生作用

每个频率提供完整 sin/cos subspace。慢频率在有限区间中趋向共同的 \(\operatorname{span}\{1,\Delta\}\)，因此 geometric log-spacing 可能把大量 nominal dimensions 花在高度重叠的 positional directions 上。

同时，crossing 说明模型不是被动使用频率表，而是与这个 basis 共适应。

#### 第三段：把这个认识变成构造和实际收益

- EVQ-Cosh：学习期、解析、零参数的一个 principled construction；
- MLA / continuation / 8B：证明 allocation 能产生 consequential effects；
- BM / frozen profiles：证明在 mature model 中，固定 total extension budget 后，intermediate allocation 仍然影响表现，但 optimum 会随 checkpoint 和 length 变化。

### 5.3 论文不需要讲的故事

禁止再讲：

> 我们要找到一个击败所有 YaRN/MrRoPE/LongRoPE 的 universal zero-training context-extension schedule。

这不是本文的核心科学问题，也会让所有 mixed results 变成负担。

禁止再讲：

> EVQ-Cosh 是理论推导出的下游最优 frequency table。

理论只保证 surrogate optimum。

禁止再讲：

> 低频冗余，所以把所有低频向高频搬就一定更好。

range retargeting reversal 和 co-adaptation 已经否定这种简单规律。

---

## 6. 建议的 9 页主文结构

ICLR 2027 投稿主文必须不超过 9 页，references 不计；appendix 可无限，但 reviewer 不必阅读。以下按正文约 8.6–8.9 页设计。

### 第 1 页：Introduction + Hero Figure

必须在前半页完成：

1. 标准 RoPE 被通常描述为 base-controlled geometric table；
2. finite table 还有独立的 interior allocation；
3. 现有工作改变 base、learn frequencies、做 target-\(s\) extension，但没有用 exact fixed endpoints 识别 normalized interior allocation；
4. 本文做什么；
5. Hero Figure。

Hero Figure 保留当前四 panel 结构：

- (a) same endpoints, different allocation；
- (b) fixed-support 三 seed improvement；
- (c) retargeted-range reversal；
- (d) weight–table crossing。

图题直接说出主发现：

> Support, interior allocation, and learned compatibility are distinct.

### 第 2 页：Problem Formulation and Experimental Identification

#### 2.1 Table decomposition

\[
x_k=a+Rz_k
\]

再引入 slot assignment \(\pi\)。

#### 2.2 Exact fixed-support paired intervention

给出最重要的控制变量和三 seed 结果。

#### 2.3 Factorial breadth

把 12-config factorial 压成一个小表或一段：

- nonuniform shapes 多数配置改善；
- Cosh vs exponential 未决；
- 结论是 allocation matters，不是 Cosh universal optimum。

### 第 3–4 页：Positional-Basis Geometry and Co-adaptation

#### 3.1 Full-pair subspace

\[
V_\omega=\operatorname{span}\{\cos(\omega\Delta),\sin(\omega\Delta)\}.
\]

#### 3.2 Canonical overlap and rank identity

只给 main theorem，证明进 appendix。

#### 3.3 Slow-frequency limit

\[
V_\omega\to\operatorname{span}\{1,\Delta\}.
\]

放当前 slow-band collapse / effective rank figure。

#### 3.4 Allocation vs slot assignment vs learned compatibility

- permutation equivalence 的核心式；
- 50M crossing；
- 151.9M crossing 可放 appendix，main text 一句 replication。

这一节结尾：

> Static geometry characterizes the supplied basis; model behavior additionally depends on how learned projections use and co-adapt to that basis.

### 第 4–5 页：An Analytic Allocation

给出：

- convex density functional；
- unique Cosh minimizer；
- inverse-CDF quantiles；
- endpoint anchoring；
- epistemic boundary。

不要把 \(\tau=d_{\mathrm{head}}/\sqrt{L}\) 写成普适定律。写成 pre-specified reference rule / tested operating choice。

### 第 5–7 页：Learning-time Model Evidence

这一节只围绕“analytic allocation has consequential effects”组织。

#### 5.1 Scarce rotary budget：主 Hero Result

432M MLA 三 seed必须是核心表：

- Geo；
- EVQ；
- Geo + same blend；
- EVQ + same blend。

#### 5.2 Full-parameter continuation

750M：

- 4K near-equal；
- 16K PPL；
- 8K strict AR retrieval。

#### 5.3 Matched adaptation and scale

Llama3-8B 只保留 compact curve / table：

- 明确 8K trade-off；
- 16K/32K transfer gain；
- 不称 lossless。

1.485B from-init 由于 trainer implementations 不完全相同，放 appendix；主文可一句 “supporting scale evidence” 并明确协议差异。

### 第 7–8 页：Frozen Deployment as a Consequence

标题不要写成第二个主要方法。建议：

> **Implication for Frozen Context Extension**

先统一写：

\[
d_k=\log\frac{\omega_k^N}{\omega_k'},
\qquad
\omega_k'=\omega_k^N e^{-d_k}.
\]

然后：

- 解释 YaRN(s)、MR(s)、BM(s) 是同一 displacement coordinate 下的 deployment profiles；
- 明确 native = \(s=1\)；
- 只保留 OLMo natural QA 的 BM vs MR(4) 主结果；
- 同一句承认 Qwen 128K preference reverses，说明 checkpoint/length dependence；
- Qwen +6.09 joint table-amplitude 结果移 appendix，或者在主文明确标成 complete configuration，不作为 allocation-only evidence。

### 第 8–9 页：Related Work、Discussion、Limitations、Conclusion

Related Work 按四类写：

1. support / base / target-length scaling；
2. learned/head-specific frequencies；
3. frequency-use diagnostics；
4. operator-changing methods。

Conclusion 回到一句话：

> interior allocation is a distinct design coordinate whose effect depends on range and learned compatibility.

---

## 7. 主文应该摆哪些图表

### Figure 1：Allocation, range, and compatibility

沿用当前 Figure 1，稍微重画：

- panel (a) 用括号直接标 \(a\)、\(R\)、\(z\)；
- panel (b)/(c) 共用 y-axis；
- panel (d) crossing；
- 图内不要写 “Cosh is better”，写 fixed-support / retargeted。

### Figure 2：Where the finite rotary budget goes

组合：

- slow-frequency collapse；
- full-table effective rank vs K；
- 不需要塞过多公式。

### Table 1：Controlled identification and breadth

建议：

| Study | What is fixed | What changes | Unit | Result |
|---|---|---|---|---|
| 151.9M paired | support, init, data, optimizer | 30 interiors | 3 seeds | all OOD lengths same direction |
| 50.9M factorial | endpoints within config | allocation family/strength | 12 configs × 3 seeds | 1.25× 10/12; exponential 9/12 |
| Crossing | weights or table | runtime allocation | seeds / windows | matched table preferred |

### Table 2：Learning-time effects

| Model / regime | Seeds | Train length | Eval | Geo | EVQ |
|---|---:|---:|---:|---:|---:|
| 432M MLA | 3 | 8K | 8K | 35.4 | 35.8 |
|  |  |  | 16K | 138.8 | 95.6 |
| + same blend | 3 | 8K | 16K | 117.9 | 71.1 |
| 750M continuation | 1 | 4K | 4K | 22.0 | 22.3 |
|  |  |  | 16K | 45.1 | 24.4 |
| Llama3-8B LoRA | 1 pair | 8K | 8K | 6.82 | 10.07 |
|  |  |  | 16K | 108.96 | 24.07 |
|  |  |  | 32K | 991.48 | 127.91 |

表题必须提醒不同 rows 属于不同 matched protocols，不能横向当 leaderboard。

### Figure / Table 3：Frozen implication

主文只放：

- `Native (s=1) / MR(4) / BM(4)` 的定义；
- OLMo 五项 natural QA slope chart；
- 一行 Qwen counterexample / preference reversal。

不要让 deployment evidence 的页数超过 learning-time evidence。

---

## 8. 还需要补充什么实验

### 8.1 唯一建议的新训练实验：constrained fixed-support learnable-\(z\)

这是最高 ROI，也是正面对比 LeRoPE 的实验。

#### 问题

在 support 完全固定时，直接学习 interior allocation，是否能超过 Geo？EVQ 作为解析零参数构造，与 task-loss-learned allocation 的关系是什么？

#### 方法

固定：

\[
z_0=0,\qquad z_{K-1}=1.
\]

将内部 gaps 参数化为正数：

\[
g_i=\operatorname{softplus}(u_i)+\epsilon,
\]

归一化后累积：

\[
z_k=\frac{\sum_{i<k}g_i}{\sum_i g_i}.
\]

或者使用 softmax gaps。这样：

- endpoints 永远固定；
- 序列严格单调；
- 只学习 \(K-2\) 个 allocation degrees of freedom；
- 不允许 support 漂移。

#### 对照

至少：

1. Geo；
2. anchored EVQ-Cosh；
3. constrained learnable-\(z\)。

可选：

4. matched exponential；
5. one-parameter learnable \(\tau\)。

#### 协议

优先复用当前 125M learned-frequency comparator 的代码、数据和评估：

- 三 seeds；
- 相同 init/data/order/optimizer/token budget；
- 为 \(u_i\) 预先固定 learning-rate multiplier；
- 不在 eval 后调参；
- 评估 1×/2×/4×/8×；
- 同时报 training-window NLL 和 weighted OOD NLL。

#### 无论结果如何都能解释

- learnable-\(z\) > EVQ：证明 allocation optimization 有空间；EVQ 是 analytic/no-search baseline；
- EVQ ≈ learnable-\(z\)：说明简单解析构造已捕捉主要收益；
- learnable-\(z\) < EVQ：说明 task-loss optimization 在短预算下不容易找到好 allocation，不能据此声称 analytic universally better；
- learnable-\(z\) 和 Geo 都差不多：与现有 fixed-support paired evidence 冲突时，优先检查 protocol、budget 和 gradient scale，不应直接推翻主结论。

### 8.2 必须整理但大概率不需要新训练的实验

#### A. Native / MR(s) / BM(s) 三臂表

在 OLMo 上统一列：

- Native \(s=1\)；
- MR(4)；
- BM(4)；
- 同一 checkpoint；
- MR/BM 同一 gain；
- 4K / 8K / 16K；
- 同一 task rows / decoder / precision。

这张表的目的不是证明 BM universal SOTA，而是消除“MrRoPE-Pro 是固定 baseline”这一叙事错误，并区分：

- native preservation；
- same-\(s\) extension comparison。

已有 outputs 能覆盖的优先重算汇总，不要重复跑模型。

#### B. Table × amplitude 的 2×2

至少一个 Qwen panel：

| Table | \(c=0.1\) | \(c=0.074\) |
|---|---:|---:|
| MR(s) |  |  |
| BM(s) |  |  |

当前数据已接近完整。主文如果继续使用 +6.09，必须叫：

> static table–amplitude configuration gain

而不是 pure exponent-allocation gain。

#### C. Alternative analytic shape 进入主文

50.9M factorial 已经有 matched exponential。把它提升到主文，直接回答旧 reviewer 的“其他 analytic schedule”问题，不需要再发明十条曲线。

### 8.3 可选，不应阻塞投稿

- 750M 第二 seed；
- 8B 第二 adaptation seed；
- 432M anchored exact-endpoint 复现；
- Video DiT 第二 seed；
- 更多 128K benchmark。

这些都不如 constrained learnable-\(z\) 对 novelty 和 reviewer confidence 的边际价值高。

### 8.4 明确停止的实验

从现在起不再做：

1. 寻找一个所有 checkpoint、所有长度、所有任务都超过 YaRN/MrRoPE 的静态表；
2. 没有 pre-specified hypothesis 的 profile curve sweep；
3. 看到某个 Qwen/OLMo 失败就再发明一个三段函数；
4. 用 PPL、RULER、QA 中最有利的一项 post-hoc 选择方法；
5. 为了“看起来 SOTA”加入不匹配 gain、prompt、decoder 或 evaluation rows 的比较；
6. 在主问题已被识别后继续扩展 benchmark zoo；
7. 把 failed universal dominance 当作论文主命题失败。

---

## 9. Claim 规则

### 9.1 可以强讲的 claim

1. **Definition**
   > A finite RoPE table has an interior allocation degree of freedom after its sampled endpoints and pair count are fixed.

2. **Causal identification**
   > Changing only the 30 interior exponents changes learned extrapolation behavior across all three paired seeds.

3. **Range interaction**
   > Retargeting the support reverses the ordering, so allocation quality is operating-range dependent.

4. **Co-adaptation**
   > Frozen weights prefer runtime tables derived from their training allocation.

5. **Exact geometry**
   > Slow full sine–cosine subspaces converge to a shared two-dimensional limit.

6. **Constructive value**
   > EVQ-Cosh is the unique density minimizer of the stated convex design criterion and yields a zero-parameter finite table.

7. **Practical consequence**
   > Under matched protocols, exponent allocation produces consequential changes in MLA training, continuation, adaptation, and frozen deployment.

### 9.2 必须限定的 claim

- “best” 只能限定到明确表格和 protocol；
- “improves” 必须写 model、length、metric；
- “near-lossless” 不用于 Llama3-8B；
- “SOTA” 除非有同协议完整基线，否则删除；
- “theoretical optimum” 必须写成 “optimum of the stated surrogate”；
- BM 只能说 improves OLMo panel，不说 dominates MrRoPE；
- Qwen +6.09 必须标 joint table–amplitude configuration；
- 1.485B from-init 必须披露 trainer implementation difference。

### 9.3 禁止词句

Codex 应全仓搜索并人工审计：

- `universal optimum`
- `globally optimal`
- `first non-geometric`
- `first to modify RoPE frequencies`
- `lossless` / `near-lossless`
- `state-of-the-art` / `SOTA`
- 不带 `(s)` 的 `MrRoPE-Pro`，当它被作为部署表比较时
- `proves downstream performance`
- `theory guarantees`
- `all models`
- `across architectures`，若实际上只覆盖特定协议

并不是全部删除，而是每处必须检查是否有直接证据。

---

## 10. 预期 ICLR reviewer 会怎样读

### 10.1 重构后的理想 review summary

> This paper identifies normalized interior exponent allocation as a distinct design coordinate of finite RoPE tables, separate from their sampled frequency support and slot assignment. A tightly controlled paired training study changes only the interior exponents and finds consistent effects on out-of-distribution language modeling. The paper develops a phase-invariant full sine–cosine subspace analysis, proves a shared slow-frequency limit and an effective-rank identity, and shows strong weight–table co-adaptation. It then derives an analytic zero-parameter allocation and demonstrates consequential effects in MLA-style pretraining, full-parameter continuation, matched 8B adaptation, and frozen deployment. The paper does not establish a universal optimal schedule, but the controlled identification and breadth of evidence support exponent allocation as an important design variable.

这段 summary 对应 6 的逻辑非常自然。

### 10.2 最可能的四个 reviewer concerns

1. **LeRoPE/AdaRoPE 已经学习 frequencies，novelty 是否足够？**  
   回答：精确固定 support 的 causal isolation、analytic/no-search construction、crossed compatibility 和 full-pair geometry。

2. **Cosh 为什么是下游正确目标？**  
   回答：不是下游 theorem；它是一个明示 prior 的 analytic construction，downstream value 由 matched experiments 测量。

3. **协议太多，是否 cherry-picking？**  
   回答：主文按 identification → mechanism → construction → consequences 排列；每个 protocol 有独立单位和完整 ledger；保留负结果与 reversal。

4. **为什么不是 SOTA？**  
   回答：论文目标是识别新的设计变量和机制；ICLR 官方标准不要求每项工作建立 leaderboard SOTA。

---

## 11. Codex 仓库执行方案

### 11.1 总原则

Codex 不能拿当前 PDF 直接“凭感觉重写”。它必须先建立仓库事实地图，再编辑。

顺序固定：

\[
\text{repo inventory}
\rightarrow
\text{claims ledger}
\rightarrow
\text{related-work matrix}
\rightarrow
\text{structure patch}
\rightarrow
\text{prose patch}
\rightarrow
\text{figures/tables}
\rightarrow
\text{adversarial audit}
\]

### 11.2 在仓库根目录创建 `AGENTS.md`

```markdown
# AGENTS.md — ICLR 2027 Paper Repository

## Primary objective

Restructure the paper around one scientific thesis:

A finite RoPE table contains a normalized interior exponent-allocation
degree of freedom that is distinct from sampled frequency support, slot
assignment, target extension factor, and compatibility with learned weights.

Do not shrink this objective into a paper about one Cosh curve or one
training-free context-extension profile.

## Scientific invariants

1. Use the decomposition x_k = a + R z_k.
   - (a, R): sampled log-frequency support.
   - z: sorted normalized interior allocation.
   - pi: slot assignment.
   - compatibility: interaction with learned weights.
   These are distinct objects.

2. For context-extension methods, always retain the extension factor:
   YaRN(s), MR(s), BM(s). Native RoPE is the s=1 identity.
   Never treat MrRoPE-Pro as an unparameterized fixed baseline.

3. The central causal evidence is the paired fixed-support experiment:
   identical endpoints, pair count, initialization, data order, optimizer,
   schedule, and token budget; only K-2 interior exponents change.

4. EVQ-Cosh is the unique optimum of the stated convex surrogate.
   Do not claim it is a universal downstream optimum.

5. Do not claim first use of non-geometric, learned, or modified RoPE
   frequencies. The novelty claim is exact fixed-support isolation,
   geometry, co-adaptation, and analytic allocation.

6. Preserve negative and conditional evidence:
   - target-range retargeting reverses the fixed-support ordering;
   - BM does not dominate MR(s) on every Qwen checkpoint/length;
   - the Qwen +6.09 result is a joint table-amplitude configuration;
   - the Llama-3-8B result has an 8K/long-length trade-off;
   - the 1.485B from-init comparison has trainer-implementation differences.

## Evidence integrity

- Never invent, round, move, or combine a number without locating its
  repository source.
- Every main-text numeric claim must have an entry in CLAIMS_LEDGER.yaml.
- Prefer generated tables/figures from scripts over hand-copied values.
- Do not use the compiled PDF as the sole source of truth.
- Preserve the experimental unit: training seed, configuration, task,
  prompt, or document, as specified by the protocol.
- Distinguish exact theorem, surrogate assumption, operating heuristic,
  exploratory analysis, and measured downstream result.

## Editing behavior

- Inspect the repository before editing.
- Make the smallest coherent patch for the current phase.
- Do not refactor experiment code while rewriting prose unless required
  to reproduce a claimed result.
- Do not delete evidence; move secondary material to the appendix.
- Continue through non-blocking issues and record them in OPEN_ISSUES.md.
- Ask only when a genuinely blocking ambiguity cannot be resolved from
  the repository.
- Compile after every structural or LaTeX change using the repository's
  existing build command.
- Keep ICLR submission main text at or below 9 pages excluding references.
- Preserve double-blind anonymity.

## Required validation

Before declaring completion:

1. Compile without unresolved references or missing figures.
2. Verify main-text page count.
3. Run the claim-ledger verification script.
4. Search for banned or risky phrases listed in PAPER_RESTRUCTURE_SPEC.md.
5. Report changed files, moved evidence, unresolved conflicts, and exact
   build commands/results.
```

Codex 官方会在开始任务前读取 `AGENTS.md`，并允许根目录规则与子目录 override 分层。论文仓库的关键科学约束应放根目录，而不是只写在一次性 prompt 里。

### 11.3 第一阶段：只读仓库审计

让 Codex 先生成，不改论文：

#### `PAPER_MAP.md`

包含：

- main TeX entry；
- section files；
- bibliography；
- current page boundaries；
- figures and generation scripts；
- tables and numeric sources；
- experiment manifests；
- checkpoints / output JSON / CSV；
- compile command；
- stale/dead files；
- 每个主文 claim 对应的 source location。

#### `CLAIMS_LEDGER.yaml`

模板：

```yaml
- id: C01
  claim: "Interior allocation changes learned extrapolation at fixed support."
  tier: causal_measurement
  manuscript_location: "Sec. 3.1 / Fig. 1"
  protocol:
    model: "151.9M scratch"
    intervention: "30 interior exponents"
    fixed:
      - endpoints
      - pair_count
      - initialization
      - token_order
      - optimizer
      - schedule
      - token_budget
    unit: training_seed
  result_source:
    path: "REPLACE_WITH_REPO_PATH"
    script: "REPLACE_WITH_REPRODUCTION_SCRIPT"
  displayed_values:
    - "2x: -0.281 NLL"
    - "4x: -0.176 NLL"
    - "8x: -0.146 NLL"
  caveat: "Ordering reverses after target-range retargeting."
  status: verified
```

Tier 必须只用：

- `exact_geometry`
- `proved_surrogate_result`
- `operating_rule`
- `causal_measurement`
- `matched_empirical_result`
- `exploratory_result`
- `descriptive_support`

#### `RELATED_WORK_MATRIX.md`

字段：

| paper | scientific question | intervention | support fixed? | allocation explicit? | learned parameters? | target-s conditioned? | head-specific? | changes operator? | exact overlap | required wording |

### 11.4 第二阶段：结构先行，不改数字

Codex 在独立 worktree 中：

1. 新建结构 skeleton；
2. 将现有段落映射到新章节；
3. 将 secondary material 移 appendix；
4. 不重写句子，不改数字；
5. 编译并报告页数；
6. 输出 `STRUCTURE_DIFF.md`。

推荐 worktree：

```bash
git status
git tag pre-iclr-restructure-20260911
git worktree add ../evq-paper-story -b paper/story
git worktree add ../evq-evidence-audit -b audit/evidence
git worktree add ../evq-related-work -b audit/related-work
```

只有工作区干净时才打 tag。并行 worktree 适合读审计和彼此不重合的改动；禁止两个 agent 同时写同一个 TeX section。

### 11.5 第三阶段：按顺序重写

重写顺序固定：

1. title；
2. one-sentence thesis；
3. abstract；
4. introduction；
5. contributions；
6. problem formulation；
7. related work；
8. section transitions；
9. conclusion；
10. 最后才压缩 method / experiment prose。

原因：如果 abstract 和 introduction 没锁定，Codex 会继续把 frozen deployment、EVQ-Cosh 和 geometry 写成三篇论文。

### 11.6 第四阶段：图表重构

要求 Codex：

- 查找生成当前 Figure 1、geometry figure、MLA table、natural-QA figure 的脚本；
- 不手工复制数值；
- 统一术语和颜色/marker；
- 每张图只回答一个 reviewer question；
- 主图 caption 自包含；
- figure sources 和 output hash 写入 ledger。

### 11.7 第五阶段：并行只读审计 agent

#### Agent A：Novelty auditor

```text
Read the paper and bibliography as a hostile ICLR reviewer.
Do not edit files. Identify every sentence that could be contradicted by
LeRoPE, AdaRoPE, LongRoPE, MrRoPE, FMRoPE, Möbius RoPE, frequency-usage
papers, or operator-changing RoPE variants. For each sentence, classify it
as safe, too broad, unsupported, or incorrectly framed. Propose the smallest
replacement wording that preserves the paper's central contribution:
exact fixed-support isolation of normalized interior exponent allocation.
Return a table with file, line, issue, prior work, and replacement.
```

#### Agent B：Evidence auditor

```text
Do not edit the manuscript. For every number in the abstract, introduction,
main figures, and main tables, locate the repository source, reproduction
script, experimental unit, and caveat. Flag copied numbers, inconsistent
rounding, mismatched rows, hidden amplitude differences, and claims whose
source is only the compiled PDF. Update no data; return an audit report.
```

#### Agent C：Skeptical ICLR reviewer

```text
Review the current paper under the ICLR 2027 criteria. Focus only on issues
that could change accept/reject. State the paper's single main question,
whether the literature placement is correct, whether each central claim is
supported, and whether the significance is sufficient without universal
SOTA. Give a 1–10 rating and list at most three decision-critical concerns.
Do not request a benchmark merely because it exists.
```

#### Agent D：Page-budget editor

```text
Find prose, tables, and protocol details that can move from the nine-page
main text to the appendix without weakening the accept/reject case. Preserve
the fixed-support experiment, full-pair geometry, co-adaptation, EVQ-Cosh
construction, 432M MLA result, 750M result, 8B trade-off, and one frozen
deployment consequence. Return a proposed page budget and relocation map.
```

#### Agent E：LaTeX / consistency auditor

```text
Compile the paper, inspect warnings, cross-references, bibliography entries,
figure readability, anonymity, page count, terminology, and equation symbol
consistency. Search for unparameterized MrRoPE-Pro comparisons and replace
only after confirming context. Do not alter scientific claims.
```

### 11.8 给 Codex 的主任务 prompt

```text
You are restructuring an ICLR 2027 paper repository, not inventing a new
method and not chasing a universal context-extension leaderboard.

Read AGENTS.md and PAPER_RESTRUCTURE_SPEC.md first. Then inspect the full
repository before editing. Identify the main TeX entry point, section files,
figure/table generation scripts, experiment manifests, numeric result files,
bibliography, build command, and current nine-page boundary.

The paper's locked scientific thesis is:

A finite RoPE frequency table contains a normalized interior exponent-
allocation degree of freedom that remains after the sampled log-frequency
support and rotary-pair count are fixed. This allocation changes the supplied
positional basis, the weights learned on that basis, and length behavior.
Its effect interacts with operating range and learned compatibility.

The paper must distinguish:
1. support (a, R);
2. sorted normalized allocation z;
3. slot assignment pi;
4. learned weight-table compatibility;
5. target extension factor s.

MrRoPE-Pro must be represented as MR(s). MR(1) is native RoPE. MrRoPE is a
target-s training-free extension method and is not the core novelty collision.
LeRoPE is the closest learned-frequency comparator and must be discussed
directly. Do not claim first modification, learning, or non-geometric use of
RoPE frequencies.

Execute in phases:

Phase 1 — no manuscript edits:
- create PAPER_MAP.md;
- create CLAIMS_LEDGER.yaml;
- create RELATED_WORK_MATRIX.md;
- report the current main-text page budget and source-of-truth files.

Phase 2 — structure only:
- reorganize the paper into:
  Introduction;
  Exponent Allocation;
  Controlled Identification;
  Positional-Basis Geometry and Co-adaptation;
  Analytic Allocation;
  Learning-time Model Evidence;
  Implication for Frozen Context Extension;
  Related Work;
  Discussion/Conclusion.
- move secondary protocol details and exploratory analyses to appendices;
- preserve every result and compile.

Phase 3 — rewrite:
- rewrite abstract, introduction, contributions, related work, transitions,
  and conclusion around the locked thesis;
- remove defensive writing and unsupported broad claims;
- do not alter any number unless the claims ledger identifies its source.

Phase 4 — figures/tables:
- retain the four-panel support/allocation/retargeting/crossing hero figure;
- foreground the 432M MLA three-seed result;
- present the 750M and matched 8B results as distinct matched protocols;
- make frozen deployment a consequence, not a co-equal second paper;
- label the Qwen +6.09 result as a joint table-amplitude configuration;
- preserve Qwen cases where BM loses to MR(s).

Phase 5 — adversarial validation:
- compile;
- verify <=9 main-text pages excluding references;
- run all available claim/figure reproduction checks;
- search for risky claims and unparameterized MR comparisons;
- produce FINAL_AUDIT.md with changed files, moved evidence, verified claims,
  unresolved issues, compile command, warnings, and page count.

Do not stop at a plan. Continue through all non-blocking phases. Do not ask
for optional preferences. Ask only if a required scientific fact cannot be
resolved from the repository. Use minimal, reviewable commits for each phase.
```

---

## 12. 建议的新 Abstract 草稿

> Rotary position embedding (RoPE) tables are commonly characterized by a base, which determines the sampled range of positional scales. A finite table, however, makes another choice: how its rotary pairs are allocated within that range. We formalize this distinction by decomposing log frequencies into fixed endpoints and normalized interior exponents. In paired 151.9M-parameter training runs with identical endpoints, initialization, data order, optimization, and token budget, changing only 30 interior exponents improves language-modeling loss at 2×, 4×, and 8× the training length in all three seeds. Retargeting the frequency range reverses this ordering, while crossed frozen evaluations show that weights strongly prefer tables derived from their training allocation. We analyze the supplied positional basis through full sine–cosine subspaces, deriving an exact effective-rank identity and a shared two-dimensional limit for slow frequencies. An explicit convex allocation criterion yields EVQ-Cosh, a zero-parameter analytic table. Under matched protocols, it reduces 16K perplexity from 138.8 to 95.6 in three-seed 432M MLA-style models and from 45.1 to 24.4 after 750M-parameter continuation, while matched 8B adaptation exhibits a clear short–long transfer trade-off. These results establish interior exponent allocation as a distinct RoPE design coordinate whose effect depends on sampled range and learned compatibility.

这个版本有意不在 abstract 中塞 BM、Qwen +6.09 和 Video DiT。它只保留一个中心命题、一套机制和两组最强 learning-time evidence。

---

## 13. 建议的 Contributions

> **Our contributions are:**
>
> 1. **A controlled allocation variable.** We decompose a finite RoPE table into sampled log-frequency support, normalized interior allocation, and slot assignment. Paired training with exact shared endpoints isolates a causal effect of the interior exponents, while range retargeting shows that the preferred allocation depends on the operating span.
>
> 2. **Positional-basis geometry and learned compatibility.** We compare complete sine–cosine subspaces through canonical correlations, derive an exact effective-rank identity and the shared slow-frequency limit, and use crossed evaluations to show that learned weights co-adapt strongly to their training allocation.
>
> 3. **Analytic and deployment constructions.** A stated convex design criterion yields the zero-parameter EVQ-Cosh allocation, which produces consequential gains in matched MLA-style training, continuation, and adaptation. Expressing frozen extensions as native-relative exponent displacements further shows that fixed extension budgets remain sensitive to their interior profile.

不要再把 “BM” 单独列成与前两项平级的第四贡献。

---

## 14. 立即执行时间表

### 9 月 11 日晚—9 月 12 日

- 冻结本文定义与 claim boundary；
- 将本手册加入仓库；
- 创建 `AGENTS.md`；
- Codex 完成 `PAPER_MAP.md`、`CLAIMS_LEDGER.yaml`、`RELATED_WORK_MATRIX.md`；
- 打干净的 pre-restructure tag。

### 9 月 13–14 日

- 完成结构迁移；
- 编译到 9 页以内；
- Abstract / Intro / Contributions / Related Work 第一版；
- 生成 Native / MR(s) / BM(s) 汇总表。

### 9 月 15–17 日

- 启动 constrained fixed-support learnable-\(z\)；
- 同时完成 evidence audit、figure audit、bibliography；
- 用已有 factorial 将 alternative analytic schedule 直接拉入主文。

### 9 月 18 日

ICLR abstract deadline。提交真实、稳定、与最终主线一致的 abstract。标题建议继续使用：

> **Beyond the Base: Exponent Allocation in RoPE**

### 9 月 19–21 日

- 合并 learnable-\(z\) 结果；
- 完成图表；
- skeptical reviewer pass；
- 删掉 protocol zoo 感。

### 9 月 22–23 日

- 全文 adversarial novelty audit；
- 数字、CI、seed、unit、gain、表格逐项核对；
- appendix navigation 和 reproducibility 完成。

### 9 月 24 日

- 冻结内容；
- double-blind / author metadata / PDF fonts / links / page count / supplementary archive；
- 在新环境完整编译一次。

### 9 月 25 日

不要卡 11:59 PM AOE，提前提交最终 PDF 和 supplement。

---

## 15. 最终验收清单

### 科学问题

- [ ] 审稿人看完第一页能复述：support 不等于 allocation；
- [ ] 主命题不依赖 EVQ-Cosh universal optimum；
- [ ] learning-time 是主线，frozen deployment 是 consequence；
- [ ] MrRoPE 全部按 \(MR(s)\) 解释；
- [ ] LeRoPE、AdaRoPE、Möbius RoPE 被正面引用和区分。

### 实验

- [ ] fixed-support 三 seed 是第一主实验；
- [ ] 12-config factorial 进入主文；
- [ ] matched exponential 被用于证明不是只比较 Geo 与 Cosh；
- [ ] 432M MLA 是 Hero Result；
- [ ] 750M 与 8B 保持各自 protocol；
- [ ] frozen BM 同时报告 OLMo positive 和 Qwen reversal；
- [ ] +6.09 明确是 table–amplitude joint configuration；
- [ ] 所有主文数字都有 ledger source。

### 理论

- [ ] full sin/cos subspace，而不是 cosine-only proxy；
- [ ] exact rank identity 与 downstream prediction 分开；
- [ ] slow-frequency limit 的假设与 measure 写清楚；
- [ ] surrogate optimum 不写成 LM optimum；
- [ ] \(\tau\) rule 写成 reference operating choice；
- [ ] finite quantile grid、anchoring 和 midpoint/inclusive convention 说明清楚。

### 写作与格式

- [ ] 9 页以内；
- [ ] abstract 只讲一条主线；
- [ ] contributions 不超过三项；
- [ ] main text 没有实验日志式叙述；
- [ ] appendix 中每个主文结果可快速定位；
- [ ] 无 unresolved refs、missing figures、匿名泄露；
- [ ] AI use statement、reproducibility statement 符合 ICLR 2027 要求。

---

## 16. 主要参考文献与检索入口

### 最接近的 2026 工作

- **LeRoPE: Learnable RoPE Frequencies Improve Language Modeling**  
  arXiv:2607.10134  
  https://arxiv.org/abs/2607.10134

- **AdaRoPE: Not All Attention Heads Should Rotate and Scale Equally**  
  arXiv:2607.19363  
  https://arxiv.org/abs/2607.19363

- **How Data Shapes RoPE Frequency Usage: From Positional Scale Matching to Length Generalization**  
  arXiv:2607.07678  
  https://arxiv.org/abs/2607.07678

- **Anti-Periodic Positional Encoding: Möbius Boundary Conditions Make In-Context Retrieval Reliable**  
  arXiv:2607.21405  
  https://arxiv.org/abs/2607.21405

- **MrRoPE: Mixed-radix Rotary Position Embedding**  
  ICLR 2026 / arXiv:2601.22181  
  https://arxiv.org/abs/2601.22181

- **Frequency Bands in RoPE: Base Frequency and Context Length Shape the Interpolation–Extrapolation Trade-off**  
  ICLR 2026  
  https://openreview.net/forum?id=PR1PPxvG9Q

- **Probing Rotary Position Embeddings through Frequency Entropy**  
  ICLR 2026  
  https://openreview.net/forum?id=1JZuEDq62N

- **CoPE: Clipped RoPE as a Scalable Free Lunch for Long Context LLMs**  
  arXiv:2602.05258  
  https://arxiv.org/abs/2602.05258

### Context extension 与频率变换

- **YaRN: Efficient Context Window Extension of Large Language Models**  
  ICLR 2024  
  https://openreview.net/forum?id=wHBfxhZu1u

- **LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens**  
  ICML 2024  
  https://proceedings.mlr.press/v235/ding24i.html

- **LongRoPE2: Near-Lossless LLM Context Window Scaling**  
  ICML 2025  
  https://proceedings.mlr.press/v267/shang25a.html

- **Resonance RoPE: Improving Context Length Generalization of Large Language Models**  
  arXiv:2403.00071  
  https://arxiv.org/abs/2403.00071

- **Scaling Laws of RoPE-based Extrapolation**  
  ICLR 2024 / arXiv:2310.05209  
  https://arxiv.org/abs/2310.05209

- **Base of RoPE Bounds Context Length**  
  NeurIPS 2024 / arXiv:2405.14591  
  https://arxiv.org/abs/2405.14591

### 频段作用与 operator-changing 工作

- **Round and Round We Go! What Makes Rotary Positional Encodings Useful?**  
  ICLR 2025 / arXiv:2410.06205  
  https://arxiv.org/abs/2410.06205

- **Fourier Position Embedding: Enhancing Attention’s Periodic Extension for Length Generalization**  
  ICML 2025 / arXiv:2412.17739  
  https://arxiv.org/abs/2412.17739

- **DoPE: Denoising Rotary Position Embedding**  
  arXiv:2511.09146  
  https://arxiv.org/abs/2511.09146

- **RoPE Distinguishes Neither Positions Nor Tokens in Long Contexts, Provably**  
  arXiv:2605.15514  
  https://arxiv.org/abs/2605.15514

### 投稿与 Codex

- ICLR 2027 Reviewer Guide  
  https://iclr.cc/Conferences/2027/ReviewerGuidelines

- ICLR 2027 Author Guidelines  
  https://iclr.cc/Conferences/2027/AuthorGuidelines

- Codex `AGENTS.md` 官方说明  
  https://developers.openai.com/codex/guides/agents-md

- Codex Worktrees 官方说明  
  https://developers.openai.com/codex/app/worktrees

---

## 17. 最核心的执行原则

最后只保留五条：

1. **守住问题，不再逃离 exponent allocation。**
2. **把 fixed-support causal identification 放在一切方法结果之前。**
3. **把 EVQ-Cosh 写成 principled construction，不写成 universal answer。**
4. **把 mixed results 写成 range × allocation × learned compatibility 的发现，不写成失败。**
5. **从现在起，每个新实验都必须回答一个已命名的 reviewer question；否则不做。**
