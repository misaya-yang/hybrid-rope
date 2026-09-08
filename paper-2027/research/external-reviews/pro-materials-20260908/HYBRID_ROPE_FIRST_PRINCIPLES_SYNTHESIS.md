# Hybrid-RoPE：从位置基到真实计算的第一性原理分析

**用途：**整合已有论文和失败记录，先确定两个核心问题的计算结构，再决定是否需要补观测。本文不授权新训练、换模型、曲线搜索或扩大 benchmark。

**研究对象：**权重冻结、所有长短请求使用同一张静态 RoPE 表时，能否保留 Native 并实现真实长生成；允许小规模适配后，增加的自由度究竟需要学习什么。按最新归零备忘录，静态方案是主线，LoRA 是放宽权重约束后的补充分支。

**证据约定：**`[观察]`只指附件报告实际记载的结果；`[推导]`表示本文给出的数学结论及其明确假设；`[假设]`表示尚未由同一 checkpoint、同一批样本的内部观测识别的机制。文末列出可移植来源。本文没有读取服务器原始 JSON、模型激活或 live repository，没有运行模型/GPU实验。

---

## 0. 核心判断

之前路线的问题，不是少试了一条频率曲线，而是三次跨越了尚未建立的理论连接：

1. 从“位置函数相似/冗余”，跳到“这些 rotary slots 对成熟模型便宜、可重新分配”。
2. 从“远端证据改变输出/答案字符串出现”，跳到“正确来源已被绑定、其内容已成为可用答案”。
3. 从“训练读过很多长 tokens 或增加很多监督位置”，跳到“足以学到可迁移的长程计算”。

这三个跳跃都不由 RoPE、attention 或常规优化理论自动保证。它们也不意味着目标无法实现。正确的求解对象是：

$$
\boxed{\text{内容条件化的相对位置核}
\;\longrightarrow\;
\text{归一化的信息写入}
\;\longrightarrow\;
\text{正确的生成决策}}
$$

原论文的谱预算理论给出了第一个对象的“位置函数部分”，并由实验证明 allocation 有独立行为后果。它没有给出完整成熟模型的效用最优化。需要补的是这条连接，而不是用新的几何代理填上空白。

**当前不应执行上一版建议的 7B / 33M-token CPT / LoRA+norm+embedding 组合矩阵。**那些是参考公开经验提出的候选，并非已由本项目失败唯一推出的解决方案。特别是开放 input embedding 会显著改变参数预算，不能当作“几乎无成本的 LoRA 修正”。

---

## 1. 先固定事实，不把不同阶段拼成同一条机制链

### 1.1 最新 OLMo 表给出的实际事实

[I01] 记载的是 Native 4K 的 OLMo 在真实 16K 输入上的结果，严格成功要求完整答案、终止和两个证据世界均正确。

| 系统 | near 严格，组/32 | far 严格，组/32 | far 含答案表面，行/64 |
|---|---:|---:|---:|
| T0：原生表、无训练 | 0 | 0 | 0 |
| Z0：静态 Z、无训练 | 3 | 1 | 26 |
| ZC：Z+compact-only 适配 | 4 | 1 | 27 |
| ZF：Z+长输入适配 | 16 | 4 | 29 |
| ON：原生表+长输入适配 | 0 | 0 | 0 |

ZF 的错误分类包含 32/64“无正确答案但正常终止”；ON 为 64/64 同类。**EOS 已经不是解释全部剩余错误的充分原因。**

但 26/64 是输出字符串层面的观察，不是内部 attention、source binding 或可解码表示的直接测量。原论文的 8B source-block deletion 证明发生在另一套模型/适配/样本中，不能移植成这 26 行的机制证书。

同样，compact 的 27→25 属于 ZC/ZF 的适配后结果；当前摘要没有给出足以将这两例全部归因于 Z0 的配对轨迹。不能把适配后的回退全部计为纯换表损伤。

### 1.2 Native 失败、Native 保留与总体/分项要分开

[I02] 的 OLMo 历史固定部署结果为：

| 部署 | PPL retention | 普通任务 retention | EOS-weighted retention |
|---|---:|---:|---:|
| Z 原幅度 | 88.50% | 80.85% | 70.30% |
| Z 单位幅度 | 71.06% | 90.76% | 81.59% |
| Y 原幅度 | 66.32% | 87.28% | 84.06% |
| Y 单位幅度 | 41.08% | 75.96% | 72.37% |

结论是这四个固定系统没有通过该组联合门槛，不是全静态表空间的不可行证明。

Qwen N128 的独立 Native 确认总体为 98.06%，区间下界超过原声明的 88%；格式/索引分项为 77.05%。总体通过应当保留，分项回退也应当保留。不能事后改成“每项都必须超过 88%”，也不能把总体通过改写成没有任何遗忘。

[I01] 中 Qwen Z/Y 适配的 0.959/0.970 点值及 0.866 下界，不是 N128 的同一项确认结果。这些名称和样本域不能合并。

### 1.3 论文与当前候选不是同一“方法”

[I03] 至少包含三个不同对象：

- EVQ-Cosh：给定 surrogate 的解析 allocation，用于训练或共适应。
- 成熟模型的 derived/coarse allocation：独立构造，不是把 EVQ-Cosh 原样硬换进去。
- Native/long 会话路由：短请求调用原表，长请求调用一张固定表。

最新目标禁止通过第三项获取 Native 保留，因此比原稿部署方案更强。

原稿 p.29 的 pure-z controls 固定的是**候选之间相同的目标 support**：最快频率为 Native 最快，最慢为 Native 最慢除以四。它不表示候选的 support 与 Native 原 support 完全相同。

**方法定义：**当前严格静态候选是固定的有序张量 $\Omega_Z$ 加已声明固定幅度，全部层/heads 按声明共享，权重、decoder 和 mask 不变，所有长短请求均使用该配置，KV 生命周期中不换表。其科学承诺是一个待证实的有限域联合工作点，不是任务无关全局最优或通用长度等变性。

---

## 2. 理解 RoPE：它改变的是内容相关的相对关系核

### 2.1 从真实计算写起

在一层一个 head 中，令 $q_i,k_j$ 是经过本模型实际 Q/K normalization 后、施加 RoPE 前的向量，$v_j$ 是 value。使用 $\Delta=j-i$：

$$
s_{ij}=\frac{a}{\sqrt{d_h}}q_i^\top R_\Omega(\Delta)k_j,
\quad \alpha_{ij}=\frac{e^{s_{ij}}}{\sum_{r\le i}e^{s_{ir}}},
\quad o_i=\sum_{j\le i}\alpha_{ij}v_j.
$$

$a$ 是直接 logit multiplier。若 Q/K 各乘 $c$，则 $a=c^2$；不能与 amplitude 混用。[R01]

对第 $k$ 个二维 pair，令 $J=\begin{bmatrix}0&-1\\1&0\end{bmatrix}$，则

$$
R(\omega_k\Delta)=I\cos(\omega_k\Delta)+J\sin(\omega_k\Delta),
$$

$$
s_{ij}=\frac a{\sqrt{d_h}}\sum_k
\left[A_{ijk}\cos(\omega_k\Delta)+B_{ijk}\sin(\omega_k\Delta)\right],
$$

其中 $A_{ijk}=q_{ik}^\top k_{jk}$，$B_{ijk}=q_{ik}^\top Jk_{jk}$。

**关键：$A_{ijk},B_{ijk}$ 随内容、层、head、前缀和 checkpoint 改变。**不是一组与文本无关的 Fourier 系数。RoPE 提供一个相对位置作用，模型学习如何把内容放入这个作用的各个子空间。

若某个分析子模型中 $q=Qh,k=Kh$ 为线性投影，则可以写成矩阵值核：

$$
\mathcal K_{\theta,\Omega}(\Delta)
=Q^\top R_\Omega(\Delta)K
=\sum_k[C_k\cos(\omega_k\Delta)+D_k\sin(\omega_k\Delta)].
$$

$C_k=Q_k^\top K_k,D_k=Q_k^\top JK_k$。实际模型存在额外 Q/K norm 时，不能把它悄悄省略并使用全局固定 $C_k,D_k$；应回到真实 activation 或局部线性化。

### 2.2 为什么“RoPE 让距离越大、相关性越低”不足以指导设计

单个项的符号和相位由内容系数决定。对于不同 $q,k$，增加距离可以减小，也可以增大 logit，甚至把原本不相关的 key 变成最大竞争者。

[R02] 在受控分析和真实模型中明确质疑把 RoPE 的作用概括为普遍距离衰减。其观察是某些高频参与稳定的局部位置模式，部分低频承担语义匹配；这些不是所有模型上的固定 head 标签。

因此不能用一个距离标量解释“长上下文计算损坏”，也不能把所有低频统一归类为闲置资源。

### 2.3 为什么改表会改变后层的内容系数

对两个配置，固定同一已对齐 token pair，令 $M=aR_\Omega(\Delta)$、$M'=a'R_{\Omega'}(\Delta')$，以及 $q'=q+\delta q,k'=k+\delta k$。有精确恒等式：

$$
\sqrt{d_h}(s'-s)
=q^\top(M'-M)k
+\delta q^\top M'k
+q^\top M'\delta k
+\delta q^\top M'\delta k.
$$

第一项是**同一内容表示下的直接位置/幅度效应**；后三项是上游表示已改变的效应。

把 Native activations 固定后只算第一项，可以研究直接机制，但不等于整个网络的换表损伤。特别是 causal prefill 中，后层证据 key/value 是在它之前的整个前缀上计算出来的；把证据搬到另一位置，可能连“被读取的内容表示”本身都改变。

---

## 3. 原理论最需要补上的一条连接：位置冗余不等于内容冗余

### 3.1 原论文的结论应原样保留

[I03, p.4] 的定理描述 full sin/cos 子空间在给定距离分布下的 block-whitened Gram，并精确给出有效维数；slow-collapse 命题描述 $\omega L\to0$ 时位置子空间趋向 $\operatorname{span}\{1,\Delta\}$。

这些结论没有错误地宣称所有对应内容通道无用。原文也注明这些 channels 仍可携带内容。

真正的缺口在从这一步转向**成熟 checkpoint retrofit 规则**时：几何冗余不能直接给出可移动预算。

### 3.2 一个精确反例

[推导] 令 $K$ 个二维 pairs 使用同一个正频率 $\omega$。所有 pair 的位置函数张成空间完全相同，因此合并的位置函数空间仍只有两维。

取内容投影 $Q=K=I_{2K}$。此时

$$
\mathcal K(\Delta)=\operatorname{diag}(R(\omega\Delta),\ldots,R(\omega\Delta))
$$

对每个 $\Delta$ 都是满秩 $2K$ 的内容双线性算子。

于是：

$$
\boxed{\text{位置函数维数}=2\quad\text{但可实现的内容核秩}=2K.}
$$

这不是在说模型的信息容量等于某个矩阵秩；它只是严格否定“位置基重复，所以这些通道功能上可以无代价移动”。近似相同的正频率给出相应连续近似。

低频极限下同样容易理解：

$$
q_{ik}^\top R(\omega_k\Delta)k_{jk}
=q_{ik}^\top k_{jk}+\omega_k\Delta\,q_{ik}^\top Jk_{jk}+O((\omega_k\Delta)^2).
$$

第一项近似不依赖距离，但随候选内容 $j$ 改变，完全可以承担重要的语义区分。

### 3.3 一个容易误用的 softmax 性质

Softmax 消去的是整行相同常数：

$$\operatorname{softmax}(s+c\mathbf1)=\operatorname{softmax}(s).$$

它**不消去**一组“几乎不随距离旋转，但对不同内容不同”的数值 $q_i^\top k_j$。

因此，论文 p.15 的“在固定概率度量下中心化常数方向”不能升级成“真实 attention 会删掉慢频语义贡献”。位置函数中的常数，与实际 key 维度上的常数，不是同一个对象。

### 3.4 对零训练的直接含义

修改空间必须以真实模型使用的内容方向来判断。某 pair 在位置 Gram 中高度冗余，可能在 Native 中承担不可替代的内容匹配；另一个几何独特的 pair，却可能几乎未被权重使用。

这为过去“有效 rank 上升、PPL 反而崩”“相同 multiset 换槽位严重退化”“极小 movement RMSE 不保 Native”提供了共同的结构解释。它不是对每次实验的唯一已识别原因。

---

## 4. 静态表为什么会发生近远冲突：实际长输入不是统一坐标放大

### 4.1 RoPE 是平移的表示，不是自由距离函数

标准静态 RoPE 满足

$$R_\Omega(\Delta_1+\Delta_2)=R_\Omega(\Delta_1)R_\Omega(\Delta_2).$$

若要求一个位置映射 $f$ 的相对位移只依赖 $i-j$，即

$$f(i)-f(j)=g(i-j),$$

则整数位置上的相邻增量必须相同：$f(i+1)-f(i)=g(1)$，从而 $f(i)=ci+b$。

所以，一个保持标准相对平移结构的单坐标映射，不能同时对所有近邻使用斜率 1、对所有远距离使用另一个斜率。非线性分段的绝对位置映射可能连续，却已经改变了这个结构。

这解释旧 boundary-slope 构造为什么不等于“免费保局部再压远端”，但不否定所有不同 attention 算子。

### 4.2 PI 的精确身份只覆盖一个特殊变换

统一放大全部 token 间距时：

$$R_{\Omega/s}(s\Delta)=R_\Omega(\Delta).$$

但向两个证据之间插入文档时，原 token $i$ 被映到

$$e(i)=i+g_i,$$

于是

$$e(i)-e(j)=(i-j)+(g_i-g_j).$$

句内相邻距离可能不变，证据到问题的距离增长，问题到回答开头距离仍很短，还多出许多新的 keys。一个统一 $s$ 不能描述整个关系图。

[R04] 的数据依赖分析同样把频率缩放的有效性与任务依赖结构是否近似 dilation 联系起来。它的结果有明确建模假设，不能直接作为本项目的万能最优规律。

### 4.3 同一 pair 承担两种关系时的局部冲突条件

[推导；限制为无相位绕回的 lifted-phase 容差] 假设某个活跃 pair 同时要：

- 在局部关系 $d_N$ 上保持原相位：$d_N|\omega'-\omega|\le\eta_N$；
- 在被拉伸 $s$ 倍的关系 $d_L$ 上复现原相位：$d_L|s\omega'-\omega|\le\eta_L$。

两个频率区间有交集，当且仅当

$$
\boxed{(s-1)|\omega|\le\frac{s\eta_N}{d_N}+\frac{\eta_L}{d_L}.}
$$

这是两个闭区间相交的直接代数结果。

如果不满足，则在这个 pair、这两项精确相位目标和这些容差下，没有一个静态频率同时实现目标。多训练一个 scalar gain 也不改变这两个相位条件。

**边界：**真实功能不要求每个 pair 都复现原相位；模型可以依靠别的通道、抵消或大 margin。有限位置也可能出现绕回。因此这不是静态表整体不可能定理，更不是 Native 88% 不可能证明。

### 4.4 有利条件是什么，而不是“哪里都不能动”

静态方案有机会的结构包括：关键局部计算主要依赖可保持的方向；被移动方向在 Native 的有效作用小或有决策裕度；这些方向在远端的改变确实改善正确的内容区分；新增干扰的写入不破坏原计算。

例如，一个语义 pair 的相似度为 $\cos(\omega\Delta)$，$\omega L=\pi/8$。原生范围内它与目标 key 保持正对齐，$4L$ 时到 $\pi/2$。降低该频率可让远端重新正对齐；若精细局部顺序由别的未动方向承担，就不存在逻辑上的全局矛盾。

这仅是有限尺度可行性的构造性例子，不是候选曲线，也不声称实际 OLMo 的 heads 恰好这样分工。

**共同条件不等于手工 head-selective。**共享表也可能利用权重已有的功能分离；headwise 表也可能失败。是否有可用共同方向取决于真实内容系数和 downstream 使用，不由 sensitivity 排名自动确定。

---

## 5. 把“功能运输”写成一个可检查的有限恒等式

这一节不把 attention KL 或 activation RMSE 当作新效用函数。目标是精确拆开“长输入究竟在哪一类计算上偏离 compact”。

### 5.1 比较对象

选择一个合法的 compact 输入及保持原有 token 顺序的长输入嵌入 $e$。原问题、证据和答案关系不变，新增内容不应改变真值。多证据的顺序敏感任务不能任意 shuffle。

对一个已对齐的 query，在一层一个 head 中：

- $S$：原 compact 中可见的 tokens；
- $D$：新插入的可见 tokens；
- $\alpha^0$：compact attention；
- $\widetilde\alpha$：long attention；
- $u_j=W_O^{(h)}v_j$：已投影到 residual 空间的 value contribution。

定义

$$
\beta=\sum_{j\in D}\widetilde\alpha_j,
\quad
\bar\alpha_j=\frac{\widetilde\alpha_j}{1-\beta}\;(j\in S),
$$

$$
\bar u_S=\sum_{j\in S}\bar\alpha_j\widetilde u_j,
\quad
\bar u_D=\frac1\beta\sum_{j\in D}\widetilde\alpha_j\widetilde u_j.
$$

$\beta=0$ 时最后的乘积项约定为零。

### 5.2 精确分解

[推导，无一阶近似]

$$
\boxed{
\widetilde o-o^0
=
\underbrace{\sum_{j\in S}\alpha^0_j(\widetilde u_j-u^0_j)}_{\text{已有内容的写入表示改变}}
+
\underbrace{\sum_{j\in S}(\bar\alpha_j-\alpha^0_j)\widetilde u_j}_{\text{原 tokens 之间的路由改变}}
+
\underbrace{\beta(\bar u_D-\bar u_S)}_{\text{新增 tokens 的竞争与写入}}.
}
$$

这是对两个真实 attention 输出相减后加减中间项。三项依赖选定参考，不是一个无参考的唯一“因果份额”；但对该参考，恒等式是精确的。

它解释三个不能跳过的事实：

1. 增强证据 attention 不够；value 可能已在长前缀中变了。
2. 改变 attention 不一定有害；若对应 value 写入一样，路由差异可能不影响功能。
3. 干扰 attention mass 大小本身不够；真正进入 residual 的是 $\beta(\bar u_D-\bar u_S)$。

尤其是“空操作”也依赖 attention：softmax 总质量为 1，但某些 heads 需要近似不写入有效内容。sink/no-op 机制若被相位变化破坏，可能在没有取证需求的地方不断写入噪声。[R03] 给出了这样的模型实例；当前 OLMo 是否如此，尚无相应观测，不能把它直接定为原因。

### 5.3 从分解得到充分误差界

记 $\operatorname{diam}(\widetilde U_S)=\max_{j,r\in S}\|\widetilde u_j-\widetilde u_r\|$，则

$$
\|\widetilde o-o^0\|
\le
\sum_{j\in S}\alpha_j^0\|\widetilde u_j-u_j^0\|
+
\operatorname{diam}(\widetilde U_S)\,\mathrm{TV}(\bar\alpha,\alpha^0)
+
\beta\|\bar u_D-\bar u_S\|.
$$

若 $r_j=\widetilde s_j-s_j^0$ 是原 tokens 上的 logit 扰动，则

$$
\mathrm{TV}(\bar\alpha,\alpha^0)
\le\tanh\left(\frac{\max_j r_j-\min_j r_j}{4}\right).
$$

这说明真正影响 softmax 的是 logit 的相对变化，整行平移不重要。界的简证见附录 A。

### 5.4 多层传播与最终生成

把一个完整 Transformer block 表示为映射。令 $e_\ell$ 为该层已对齐状态的差异，$d_\ell$ 为在相同已对齐输入下，由位置/新增 tokens/已声明权重变化引起的 block defect；若参考 block 在所比较区域 Lipschitz 常数为 $\Lambda_\ell$，则

$$e_{\ell+1}\le\Lambda_\ell e_\ell+d_\ell,$$

$$e_L\le e_0\prod_{\ell=0}^{L-1}\Lambda_\ell
+\sum_{r=0}^{L-1}d_r\prod_{\ell=r+1}^{L-1}\Lambda_\ell.$$

该界对有限差异成立，但依赖相应区域的 Lipschitz 假设；用最坏权重谱范数估计往往很松。不能算一个很松的上界就宣传实际生成证书。

令最终 norm 后状态为 $h$，LM head 冻结。原正确 token $y$ 相对竞争 token $v$ 的 margin 为

$$m^0_{yv}=(w_y-w_v)^\top h^0+(b_y-b_v).$$

新 margin 精确为

$$m'_{yv}=m^0_{yv}+(w_y-w_v)^\top(h'-h^0).$$

因此，只要每个所需决策都满足

$$\|h'-h^0\|<\min_{v\ne y}\frac{m^0_{yv}}{\|w_y-w_v\|_*},$$

该决策保持。对整条正确输出及实际终止 token 均成立，则 greedy 完整输出保持。

这是“位置扰动→写入差异→生成”之间的一个完整充分条件链。它不要求每层注意力完全相同，也不保证当前候选满足条件。

**重要区别：**compact-correct 是研究“运输已有能力”的定义条件，不是任何 frozen positional intervention 都绝不可能改善 compact 错题的全局定理。改变路由也可能启用原有权重中的其它计算。

---

## 6. 为什么 signal 增强却答不对：正确方向与共同偏置必须分开

### 6.1 两世界的第一个内容分叉

对于一组合法 counterfactual 世界 $b=0,1$，两答案在第一个不同 token 前共享相同目标前缀。该分叉处的正确 tokens 分别为 $a_0,a_1$。

取真实完整词表 logits，定义

$$d_0=z^{(0)}_{a_0}-z^{(0)}_{a_1},\qquad d_1=z^{(1)}_{a_0}-z^{(1)}_{a_1},$$

$$B=\frac{d_0+d_1}{2},\qquad E=\frac{d_0-d_1}{2}.$$

则 $d_0=B+E,d_1=B-E$。有精确等价：

$$
\boxed{\text{两个世界各自把本世界答案排在另一世界答案之前}\iff E>|B|.}
$$

$E$ 是相对答案方向上随证据变化的分量；$B$ 是共同偏向某个答案的分量。即使 $E>0$，两个世界仍可能总是偏向同一个答案。例如 $E=2,B=3$ 时，world 0 相对顺序正确，world 1 仍错误。

这比“远端信息进入 logits”多了一项必要要求：证据差异必须足够大且方向正确，压过共同偏置。

**仍不等于完整生成：**其它词表 token 可能超过两个答案候选；后续 token/EOS 也可能失败。因此这只是一个内容分叉诊断，不替代严格 scorer，不引入候选 reranking，不用 gold 候选限制解码。

### 6.2 答案 CE 本身也可以精确拆开

令 $P_b=p_b(a_0)+p_b(a_1)$，则这一个分叉上的配对 CE 为

$$
\frac{-\log p_0(a_0)-\log p_1(a_1)}2
=
\underbrace{\frac{\operatorname{softplus}(-B-E)+\operatorname{softplus}(B-E)}2}_{\text{区分两种证据的条件损失}}
-
\underbrace{\frac{\log P_0+\log P_1}2}_{\text{将概率质量分配给这两个内容候选}}.
$$

因此，CE 下降可能来自候选总质量增加，也可能来自真正区分世界变好。全序列 CE 还包括共同前缀和终止位置。

这个恒等式解释为什么 NLL、EOS、格式、相对内容决策可以不同步。它不是再发明一种损失：现有合法两世界 CE 已在优化这些因素，问题在于参数和数据是否允许同时优化。

### 6.3 对当前结果的约束

- ON 的正常终止但无答案，说明它可以学会某种停止行为，而内容分叉或其它竞争尚未正确。
- ZF 严格优于 ZC，说明此次长输入训练改变了最终行为；但没有 $E,B$、其它词表竞争和逐样本内容关系，就不能仅由计数拆净内容/表达贡献。
- 26/64→29/64 不能解释成表决定了 29/64 的内容上限。
- 正确源的 likelihood 变化也可能来自一个不足以跨越 $|B|$ 的证据效应。更不能据此把未完成部分预设为 FFN-only。

所有这些解释均保持原严格成功定义，没有把表面含答案、二候选排序或 EOS 单项改成成功。

---

## 7. Native 保留为什么不能只靠平均 KL

### 7.1 NLL 不是 KL

$I02$ 的文本 NLL 近乎持平，不等于对实际 Native 生成路径的完整输出 KL 很小。前者在真实标签上平均，可以抵消；后者在原模型输出分布下平均。

$$\Delta\mathrm{NLL}=\mathbb E_{\text{data}}\log\frac{p_0(y|c)}{p_\phi(y|c)},$$

$$D_{KL}(p_0\|p_\phi)=\mathbb E_{y\sim p_0}\log\frac{p_0(y|c)}{p_\phi(y|c)}.$$

### 7.2 决策风险依赖原 margin

对唯一赢家概率 $a$ 和 runner-up 概率 $b$，使原赢家不再唯一最大的最小 forward KL 为

$$\mathcal B(p)=a\log\frac{2a}{a+b}+b\log\frac{2b}{a+b}.$$

故 $D_{KL}(p\|q)<\mathcal B(p)$ 保证原赢家保持。反过来不成立。

小 margin 时，$\mathcal B(p)\approx(a-b)^2/[2(a+b)]$。例如二项概率 $(0.5001,0.4999)$ 对调，KL 约 $8\times10^{-8}$，决策仍翻转。

因此固定平均阈值不能脱离 margin 分布和轨迹覆盖，直接变成 88% 完整生成的证书。

### 7.3 当前优化应该保护什么

应区分：换表起点造成的功能损伤、适配新增的损伤、适配恢复的功能。Native student 必须使用实际部署表。

约束形式仍然是

$$\min_\phi L_{\text{content/long}}(\phi;\Omega_Z)
\quad\mathrm{s.t.}\quad D_N(\theta_\phi,\Omega_Z;\theta_0,\Omega_0)\le\epsilon,$$

但 $D_N$ 的分布必须明确，最终仍由声明的 Native 指标决定。平均 replay KL 是优化工具，不是完整生成等价条件。

本文不要求立刻重启被暂停的 teacher-prefix 大面板。这里的理论作用是阻止把平均数当成所有决策的保证，也阻止看到回退就任意加大 KL 系数。

---

## 8. LoRA 究竟要学什么：数据可识别性与参数可达性是两个不同问题

### 8.1 它不能被理解成一组静态反向旋转

原稿 Theorem 5 已给出精确限制：位置无关、可逆 Q/K 映射不能把不同非零频谱在一段连续距离上精确共轭为同一个 RoPE 算子。

但真实多层 LoRA 可以改变内容如何投影到各频段、证据如何被编码和写入、后层怎样组合，所以不属于那个受限补偿类。原定理不能推出“LoRA 救不了换表”，也不能推出“仅 Q/K 必然足够”。

### 8.2 学会某长度的答案，不等于学会外推规则

设 $C$ 为局部上下文，$E$ 为远端信息，$Y$ 为要预测的内容。在真实分布及最优预测器下：

$$H(Y|C)-H(Y|C,E)=I(Y;E|C).$$

没有额外条件信息的远段，对最优 log-loss 没有信息收益。把同样局部预测任务放到 16K，不自动生成远程监督。

反过来，一条带独立随机值、真值依赖远证据的答案监督，也可能包含很强长程信息。**不能由 3,642 个输出标签这一计数，证明监督本质不足；也不能由几百万输入 token，证明适配已经充分。**

Dense LM 可以提供广泛分布适配，但大量目标也可能主要依赖近邻。Paired QA 能约束内容，但可能只约束很窄的关系族。两种目标解决的问题不一样；没有当前观测支持唯一的最优混合比例。

### 8.3 训练信号与频率的关系

在简化的矩阵值核中，对 cosine/sine 系数的梯度包含

$$\nabla C_k L\propto\mathbb E[G_{ij}\,h_i h_j^\top\cos(\omega_k\Delta_{ij})],$$

$$\nabla D_k L\propto\mathbb E[G_{ij}\,h_i h_j^\top\sin(\omega_k\Delta_{ij})],$$

其中 $G_{ij}=\partial L/\partial s_{ij}$，包含 softmax、value 与 downstream 误差信号。

所以真正起作用的是**距离—内容—监督梯度的联合结构**，不是距离直方图或标签数量本身。只有在额外的平滑/独立等假设下，才可以用高频相消解释某一平均梯度弱；不能直接引用 Riemann–Lebesgue 引理宣布普通 LM 必然失败。

### 8.4 最小参数空间的局部必要且充分条件

[推导；固定有限观测、固定工作点和一阶近似]

设允许的可训练参数坐标为 $\delta\phi$，$A_N,A_L$ 为声明的 Native 与 long 功能残差的 Jacobian。目标修正分别为 $b_N,b_L$。固定表起点已损伤 Native 时，通常 $b_N\ne0$。

先看精确局部方程：

$$A_N\delta\phi=b_N,\qquad A_L\delta\phi=b_L.$$

若 $b_N\in\operatorname{range}(A_N)$，取

$$\delta_0=A_N^\dagger b_N,\qquad P=I-A_N^\dagger A_N.$$

则同时有解当且仅当

$$
\boxed{b_L-A_L\delta_0\in\operatorname{range}(A_LP).}
$$

令

$$K_{L|N}=A_LPA_L^\top,$$

它描述在保持这些 Native 线性观测不变的剩余方向中，哪些 long 残差可被改变。

这不是新的大模型 Fisher 工程项目。它回答的是概念问题：

- 残差不在像空间：该工作点、该观测集上的局部参数空间不够。
- 在像空间但对应奇异值很小：可能可学，但该局部训练几何下很慢。
- 训练残差解决、未见样本失败：可能是泛化/覆盖，不是训练空间必然不足。

真实目标是近似约束和分类不等式，容许 88% 保留，不是所有残差精确为零，因此上面的精确条件是分析工具，不是实际门槛。LoRA 零输出初始化处存在切空间退化，不能在该点算到某个秩就冒充整个 rank-r 网络的表达上限。

### 8.5 为什么“多训一些”不能成为无条件答案

对局部二次目标 $\frac12\|A\delta-b\|^2$，固定 Jacobian、欧氏梯度流下有

$$b-A\delta(t)=(I-P_A)b+\sum_r e^{-\sigma_r^2t}u_ru_r^\top b.$$

第一项是该线性空间不能拟合的分量；第二项是可以学但快慢不同的分量。

增加训练时间只会衰减第二项。增加模型/模块可以改变空间和奇异值，但没有告诉我们数据是否识别了所需方向。Adam 的动态预条件、非线性特征变化和有限步长不由这条固定-Jacobian 公式覆盖。

### 8.6 “16K 训练，32K/64K 泛化”的局部可识别条件

令 $A_T$ 是训练功能观测矩阵，$A_E$ 是待泛化功能观测矩阵。训练上不可区分的更新在测试上也不可区分，当且仅当

$$\ker A_T\subseteq\ker A_E,$$

等价于存在 $R$ 使 $A_E=RA_T$。

这还不够：目标规律也要一致，$b_E=Rb_T$。若两者成立，则

$$\|A_E\delta-b_E\|\le\|R\|\,\|A_T\delta-b_T\|.$$

近似情况下：

$$
\|A_E\delta-b_E\|
\le\|R\|\,\|A_T\delta-b_T\|
+\|A_E-RA_T\|\,\|\delta\|
+\|b_E-Rb_T\|.
$$

这给出三个明确问题：训练是否覆盖所需功能方向；目标关系是否真随尺度保持；该运输是否稳定而非大幅放大误差。

长度、token 数、RoPE 相位是否见过，都只能约束这三个问题的一部分。低频的函数列在短窗口中几乎重合时，即使数学上能唯一识别，其外推条件数仍可能很大。

[R05] YaRN 和 [R06] LongLoRA 给出特定模型、目标配置、数据和预算下的建设性实例；不提供上述条件在当前 OLMo 上自动成立的证明。它们也没有证明全局单表零训练保留、极少监督和任意更远长度可以同时获得。

---

## 9. 当前失败的统一解释：哪些已确定，哪些仍缺观察

| 现有现象 | 由计算结构能确定什么 | 仍不能确定什么 |
|---|---|---|
| 同 multiset 换 rotary slots 崩溃 | 无序位置谱不是完整的成熟模型描述；内容系数绑定重要 | 哪一组 slots 是最便宜且远端有益的改动 |
| rank/coverage/小 movement 预测失败 | 位置几何没有包含内容核、value、归一化与决策 | 某个更复杂几何分数一定会成功 |
| Z/Y/gain=1 都不过 Native | 四个候选的联合工作点失败；幅度不是简单万能补偿 | 所有静态表不可行 |
| Z0 far 字符串 0→26/64 | 当前干预改变了实际输出，提供有用但不完整的信号 | 同一批失败样本已完成正确源绑定 |
| ZF near16/32、far4/32 | 此训练在同样总长度下对两种布局作用不同 | 唯一差别是最后一步 QK 相位；源 K/V 没变 |
| ON EOS1.0、内容0/64 | 终止可被学到，而内容仍失败 | LoRA 没有更新；小模型永远不行 |
| Qwen compact-only 改善16K | 原生窗口内部分行为可通过短输入适配迁移 | OLMo 4× 也不需要长输入训练 |
| Native总体98.06%、格式/索引77.05% | 总体声明成立，局部损失真实存在 | 总体失效或所有能力无遗忘 |
| all-linear 比原模型生成好 | 该复合适配有效改变功能 | FFN 必要、rank16 最小充分 |
| YaRN从头训练差 | 该从头训练设定的观察 | 成熟模型 YaRN 持续训练被否定 |
| learnable tau无信号 | 可能涉及参数化/梯度/任务信息，需原日志 | 所有allocation learning不可能 |
| Power-Shift/Wan2.1失败 | 相关模态/算子的负例应保留 | 直接解释当前文本模型的同一原因 |

最后三项在 [I01] 只有结果摘要。上一轮关于 softplus 梯度死区或 Power-Shift 方向的具体说法，除非能从原 owner 重新核对，不能在当前证据层级升为已确认原因。数学上可以说明相应可能性，不能借助先前模型的复述完成事实认证。

**目前最可信的结构性诊断：**Z 改变了部分关系计算，但还没有使完整的内容编码、候选竞争与答案生成在长输入中稳定共存；既有训练可以修其中一些方向，却没有证据显示所需残差已在当前数据与参数空间内被充分识别。具体主导项仍不能由结果汇总唯一反演。

---

## 10. 静态零训练还能优化到哪里？

### 10.1 合法的效用不能缺失

有了指定的任务关系、长输入嵌入分布和严格 decoder，才有

$$\max_{\Omega'\in\mathcal C} U_{\text{strict}}(\theta_0,\Omega')
\quad\mathrm{s.t.}\quad S_N(\theta_0,\Omega')\ge0.88S_N(\theta_0,\Omega_0).$$

从 Native weights/activations 能估计原功能对某个改动的响应，但不能单靠目标长度确定所有长任务的正确内容目标。这不是因为问题久未解决，而是效用没有由那些输入唯一指定。

可以将任务限定为“保持真值的证据搬移/干扰插入”，从而明确期望保持的输出关系；但这种结构假设应公开声明。不能把一个位置秩、相位覆盖或 attention entropy 默认为真实效用。

### 10.2 正确的局部解结构，及其缺失量

若真实可微 long utility 已给定，Native输出KL局部为 $\frac12\delta x^\top H_N\delta x$，且 $H_N\succ0$，则线性效用约束问题给出

$$\delta x^*\propto H_N^{-1}g_L.$$

但 $g_L$ 必须来自已声明任务效用，不会由 $H_N$ 自动生成；$H_N$ 奇异时的零空间分量需要单独约束；有限大改动还需非线性验证。

这不能用于“先在 long 测试上求梯度，再称 Native-only zero-training”。研究性的 long 响应测量，不得倒灌成那种部署选择器。

### 10.3 最有价值的理论方向是什么

不是再发明曲线，而是判断是否存在：

$$\boxed{\text{对 Native 关键内容关系弱作用、对目标长关系强且方向正确的合法静态改动。}}$$

这个条件允许取消、冗余和 margin，不要求每个 head 不变；也不预设 head mask。第3–6节指出了判断它所需的真实对象。

一旦这些对象显示有可用空间，再从明确 Native 校准权限和预先声明的结构假设构造一次方法；现在尚不能从现有汇总唯一推出那张表。继续列候选只会重启搜索。

---

## 11. 实验设计：只补两个有理论区分力的问题，不开新训练矩阵

这里的“先理论”不等于拒绝任何可证伪观测，而是先明确若某机制成立，哪些内部量必定呈现什么关系。现有分数已经足够否定多条过强叙事；再加 benchmark 分数不足以定位余下原因。

### 11.1 验证问题一：当前换表/远移的误差主要发生在被读取的内容，还是原 tokens 之间的匹配？

**计算对象：**第2.3节的直接/上游 logit 分解，以及第5节三个 attention-write 项。

**现有资产优先：**从已有固定 C/N/F 输入、已保存 Q/K/V 或可恢复中间量出发。只使用当前 N/Z/ZF 等已完成产物，不新增表、gain、head selector，不修改正常解码。没有完整中间量就明确列出缺失字段，而不是假称已完成机制验证。

**补前向时的范围：**只读已有、事先固定的诊断实例；使用本模型 Q/K norm 后、RoPE 前的 activations，按原 compact token 的真实映射对齐。全 heads 保留，禁止看哪个 head 表现好后只展示它。可以只保存所需 query rows，不能创建整段 N×N attention 常驻显存。

**可证伪预测：**

- “只差远端寻址，源/查询表示基本完好”要求：直接相位项解释相关 score 变化，写入表示差项相对可忽略，原 tokens 的条件路由差突出。
- 若即使把相位项替回参考值，已有 Q/K/V 表示差异仍产生大误差，该简化解释被否定。
- “只是新干扰数更多”要求条件原-token路由基本保持，新增写入项足以解释差异；若条件路由本身已严重改变，这个单因解释被否定。

**边界：**三项范数大不等于它就是最终答案错误的唯一原因，误差可以抵消。要声称模块因果性，还需要事先定义的有限 restoration intervention；本文不安排 layer/head sweep，也不自动推出新训练模块。

这不是新的成绩试验，是对一个精确恒等式及两个旧解释的检查。

### 11.2 验证问题二：实际新增的训练信号有没有修复证据的内容方向？

**计算对象：**第6节的 $E,B$、候选外最大竞争者，以及实际已声明的严格输出；优先重算已保存的目标分叉 logits。不重新启动被暂停的351案例盲标或大型teacher-prefix面板。

**配对要求：**使用已经合法验证的两个证据世界，选择答案的第一个不同 token 前的共享目标前缀。保留全部世界，不挑变化最大的样本。该读取是 teacher-forced诊断，不叫真实生成成功。

**可证伪预测：**

- 若 ZF 主要降低共同偏置而证据区分量基本未变，就不能把 strict 增益全称作新检索能力。
- 若它稳定增大证据区分量并跨过 $E>|B|$，内容条件化确实被增强；但其它词表竞争和完整终止还须正常解码验证。
- 若训练与未见样本差异很大，优先考虑数据/关系覆盖；若训练集也不满足目标，才进一步区分优化与参数空间。

**参数空间检验只作条件性下一步：**可以在已有梯度可用时检查第8节局部可行关系；若只能获得几个保存点的更新跨度，必须标注为“已观察更新子空间”，不能声称整个 LoRA 空间不够。禁止为此新建大规模全词表 Jacobian 或任意 damping/Fisher 搜索。

### 11.3 两项验证之后，允许的决策

| 观测 | 下一项研究责任，而非立即启动的配方 |
|---|---|
| 主要是直接相位造成的 Native/long 活跃关系冲突 | 明确有无共享表的合法功能余量，不以继续调gain掩盖 |
| 源/查询表示在长前缀中已改变 | 方法须处理分布式计算，不再只在最终readout处补丁 |
| 主要是新增背景写入 | 数据和模型须学会正确筛选/no-op，不把虚拟position exposure当充分训练 |
| 证据对比信号弱，训练分布没有相应约束 | 先设计真值依赖远源的任务关系，而不是简单增加标签数 |
| 训练目标可满足、未见关系失败 | 把重点放在关系泛化，不任意增加参数 |
| 局部参数空间可行但慢 | 有依据讨论优化预算；不把所有失败解释为“多训一些” |

结果只限定相应机制和作用域。任何进一步方法都必须重新写清它解决上述哪一个已被观测支持的问题；不从这个表自动派生十种候选。

---

## 12. 给 Codex 的理论分析伪代码

```python
# Read-only analysis. No new frequencies, no optimizer, no launch of training.
# Names denote required interfaces, not existing library APIs.

sources = load_current_reports_and_frozen_manifests()
assert_not_mix_model_arm_metric_or_native_length(sources)

for record in existing_declared_diagnostic_records:
    compact, extended, token_map = record.compact, record.extended, record.map
    assert_order_preserving_for_original_causal_tokens(token_map)
    assert_same_truth_relation_and_declared_counterfactual(record)

    if not has_required_qkv_and_write_outputs(record):
        record_missing_observation(record, fields="post-qk-norm/pre-rope QK; V; W_O; map")
        continue

    for layer, head, query in declared_observation_locations(record):
        q0, k0, u0, a0 = compact_state(record, layer, head, query)
        q1, k1, u1, a1 = extended_state(record, layer, head, query)
        # M includes actual phase and direct logit gain.
        M0, M1 = actual_relative_operators(record, layer, head, query)
        direct = bilinear(q0, M1 - M0, k0)
        inherited = (bilinear(q1-q0, M1, k0)
                     + bilinear(q0, M1, k1-k0)
                     + bilinear(q1-q0, M1, k1-k0))
        assert_exact_reconstruction(direct + inherited, actual_logit_difference(record))

        S, D = original_visible_tokens_and_inserted_tokens(token_map, query)
        beta = a1[D].sum()
        abar = a1[S] / (1-beta)
        value_change = weighted_sum(a0, u1[S]-u0)
        routing_change = weighted_sum(abar-a0, u1[S])
        added_write = beta * (conditional_mean(a1[D], u1[D])
                              - weighted_sum(abar, u1[S]))
        assert_exact_reconstruction(value_change + routing_change + added_write,
                                    actual_attention_write_difference(record))
        save_signed_terms_not_just_norms(record)

for pair in existing_legal_counterfactual_pairs:
    if not has_content_fork_logits(pair):
        record_missing_observation(pair, fields="shared-prefix full-vocab logits")
        continue
    d0, d1 = two_world_content_log_odds(pair)
    E, B = (d0-d1)/2, (d0+d1)/2
    save(E=E, B=B, pair_order_correct=(E > abs(B)),
         full_vocab_margin=actual_gold_margin(pair),
         unchanged_strict_generation_score=pair.registered_score)

write_theory_observation_map(
    preserve_all_registered_negative_results=True,
    no_claim_that_large_component_norm_proves_causality=True,
    no_method_or_checkpoint_selection=True,
    no_new_training=True,
)
```

**实施边界：**已有日志缺中间量时，不把伪代码变成一个无限扩张的数据平台。只列出对应的有限缺失观测及其为什么会改变结论；实际补 GPU 前向仍需按当前暂停状态获得明确安排。

---

## 13. 原论文应怎样完善，而不偏离位置编码主线

### 13.1 保留三个已经成立的结果

原稿 fixed-support 多 seed 因果干预、full sin/cos 谱预算恒等式、EVQ-Cosh 在所声明 surrogate 内的解，继续保留。不是因为成熟单表尚未成功，就把这些结果全部否定。

### 13.2 补一条重要的理论边界

将第3节“位置函数维数与内容核秩不同”的反例和解释放入理论部分，明确为何 static rank 不能给 mature retrofit 排序。这不是旁支 FFN 理论，而是 RoPE 内容—位置耦合的直接后果。

### 13.3 给行为连接一个可检验的计算表达

第5节的有限 attention-write 分解和最终 margin 条件，可以把“geometry≠capability”从口头限制变成数学解释。它本身主要由标准代数与稳定性工具构成，不能单凭写出公式就声称新颖性；科学价值取决于它是否解释本项目真实反例并带来明确设计约束。

### 13.4 收紧尚未识别的强结论

p.25 的 Appendix 已说明 target-retargeting 排序反转没有识别完整 support×allocation interaction law。主文不宜把这件事进一步写成通用交互定律。

同理，frozen zero-training、Cosh训练、会话routing和LoRA应各自有方法身份与评价合同。当前仍需争取一个真实工作点，但不应为了它成功而改变论文已建立的科学对象。

### 13.5 不能把经验参照变成必要性定理

YaRN 的长LM训练、LongLoRA的norm/embedding经验，可以是未来设计依据；并不证明本项目的失败唯一源于数据量、norm冻结或模型大小。下一轮应把这些选择建立在具体缺失功能上，而不是“某论文这样做成功了”。

---

## 14. 最终回答

**静态表为何还没有同时保住 Native 与正确长生成？**

因为它修改的是同一套内容相关关系核在全部距离上的作用；局部语义/顺序、远程取证、背景筛选和停止行为共用这套计算。现有规则主要依据位置几何分配移动量，尚未证明这些改动落在“Native可容忍且long方向正确”的功能空间。四个候选失败并不关闭该空间。

**为什么信息似乎到达输出，生成仍失败？**

因为“输出对源有变化”只说明依赖存在，不说明正确源绑定、value写入方向、竞争偏置和全轨迹margin满足条件。两个世界在一个内容分叉上都正确，至少需要 $E>|B|$；还要压过其它词表候选，随后正确继续并终止。

**LoRA怎样才算学会外推？**

必须学到对合法位置/背景变换稳定的内容关系，而不是只拟合某个长度上的答案形式。它需要数据确实约束这些关系、允许参数能实现所需修正、Native约束保留相应决策。所需功能不在当前训练可识别空间中时，多读tokens不自动补上；在可达空间但慢时，才有理由增加优化预算。

**现在做什么？**

先完成本文件中的理论连接与既有观测对照；不启动新的训练/模型/曲线矩阵。必要时仅补第11节两个问题所缺的有限内部观测。目标不撤回，但不再用下一张benchmark表代替原因。

---

# 附录 A. 两个界与分解的简证

## A.1 attention-write 恒等式

长输出为 $(1-\beta)\bar u_S+\beta\bar u_D$，短输出为 $\sum\alpha^0u^0$。加减 $\bar u_S$ 与 $\sum\alpha^0\widetilde u$ 即得到第5.2节。范数界来自三角不等式以及两个分布期望的差不超过 value 集合直径乘 TV。

## A.2 exponential tilt 的 TV 界

设 $q_j=p_je^{r_j}/Z$，令 $t_j=e^{r_j}\in[a,b]$，$Z=\mathbb E_p t$。则

$$\mathrm{TV}(p,q)=\frac{\mathbb E_p|t-Z|}{2Z}.$$

对给定均值 $Z$，凸函数 $|t-Z|$ 的期望不超过只在端点取值的分布所得值，因此

$$\mathrm{TV}\le\frac{(b-Z)(Z-a)}{Z(b-a)}.$$

右侧在 $Z=\sqrt{ab}$ 处最大，值为

$$\frac{\sqrt b-\sqrt a}{\sqrt b+\sqrt a}
=\tanh\frac{\log b-\log a}{4}.$$

这也显示整行 $r_j$ 同时加一个常数不影响界。

## A.3 KL 决策边界

对固定竞争者 $v$，最小化 $D_{KL}(p\|q)$ 且要求 $q_v\ge q_y$。最优点位于 $q_v=q_y$；拉格朗日条件给出这两项各为 $(p_y+p_v)/2$，其余为原概率。对竞争者取最小值由 runner-up 达成，于是得到第7.2节。等号仅到达tie边界，严格保留使用严格小于。

---

# 附录 B. 原论文慢频解释的额外限定

谱预算 $r_2$ 是声明的距离分布下的基性质，而原稿强 fixed-support训练对照位于 $L=256,\mathrm{base}=256,K=32$。最慢频率的 $\omega_{\min}L=256^{1/32}\approx1.189$，不是所有频带都位于 $\omega L\ll1$ 的深慢频极限。

因此，不能说 slow-collapse 命题已经解释全部强训练收益。更准确的表述是：它解释一个可出现的冗余机制，而独立因果实验说明 allocation 的行为作用还覆盖其它区间。这一限定不削弱因果结果。

---

# 附录 C. 本轮计算检查，不是模型实验

本文在 CPU 随机小数组上做了代数 sanity checks：

- 1,000 组 attention-write 分解，最大绝对重构误差约 $8.9\times10^{-16}$。
- 1,000 组 softmax tilt，未发现 TV 界数值违反。
- 1,000 组配对CE分解，最大绝对误差约 $8.9\times10^{-16}$。
- 一个相容局部线性系统，Native/long联立构造残差约 $1.5\times10^{-15}$ / $5.7\times10^{-15}$。

这些只帮助检查代数与符号，不是理论证明的替代，不是 OLMo/Qwen 上的结果，不产生任何方法成功率估计。

---

# 参考与证据索引

## 内部来源

**[I01]** `SPECTRAL_BUDGET_LORA_20260905.md`，文内标题日期2026-09-06。主线归零、code_release_008、T0/Z0/ZC/ZF/ON、停止事项。是摘要，不等于本轮已核验原始逐行产物。

**[I02]** `SINGLE_TABLE_FFN_SERVER_EXECUTION_20260904(1).md`，最终Native确认版本。包含固定Z/Y与单位幅度、Qwen N128训练与确认、source-only对照、已完成及未完成状态。旧版同名报告仍有历史状态，不覆盖更新版。

**[I03]** `main(20260905-052024).pdf`，31页。重点：p.4谱预算/慢频/surrogate；p.5共适应与transplant；p.6–7方法身份与成熟实验；p.25固定support与retargeting反转及限定；p.29–30 pure-z block和routing的区别。

**[I04]** `HYBRID_ROPE_NEXT_DAY_PLAN_20260906.md`。仅作为上一轮建议记录；其训练矩阵不是已执行结果，当前第一性原理分析不将其自动延续为启动指令。

## 本轮核对的外部原始研究

**[R01]** Su et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding*. arXiv:2104.09864. 用于RoPE标准算子；具体真实模型行为不能由原始距离衰减叙事直接推定。

**[R02]** Barbero et al. *Round and Round We Go! What makes Rotary Positional Encodings useful?* ICLR 2025; arXiv:2410.06205v3. 低频语义使用、位置head与距离衰减的机制边界。

**[R03]** Wertheimer et al. *Frayed RoPE and Long Inputs: A Geometric Perspective*. arXiv:2603.18017. Q/K几何及sink机制的模型实例，不是当前OLMo已经被证明的原因。

**[R04]** Wu, Liu, Jadbabaie. *How Data Shapes RoPE Frequency Usage: From Positional Scale Matching to Length Generalization*. arXiv:2607.07678. 任务依赖尺度与频率使用；理论结论依赖其明确模型。

**[R05]** Peng et al. *YaRN: Efficient Context Window Extension of Large Language Models*. ICLR 2024; arXiv:2309.00071. 目标位置配置与训练尺度泛化的建设性实例，不是当前预算的成功保证。

**[R06]** Chen et al. *LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models*. ICLR 2024; arXiv:2309.12307v3. 特定设置下norm/embedding与低秩适配经验，不能推导本项目模块必要性。

**[R07]** Chiang, Yogatama. *The Rotary Position Embedding May Cause Dimension Inefficiency in Attention Heads for Long-Distance Retrieval*. Findings of ACL 2025; arXiv:2502.11276. 某些旋转维度的利用效率证据，与“所有低频无用”不是一个结论。

**新颖性声明：**本文的矩阵反例、softmax分解、margin边界和局部线性代数主要使用标准工具。本文不声称它们分别构成新定理贡献；其用途是建立对当前失败可核验的理论连接，并筛除无依据的新配方。是否足以成为论文新增贡献，需要对相关文献和实际解释力进一步审查。
