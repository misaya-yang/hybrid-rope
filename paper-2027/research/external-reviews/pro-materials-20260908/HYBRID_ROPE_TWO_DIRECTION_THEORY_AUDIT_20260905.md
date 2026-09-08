# Hybrid-RoPE：两个高 ROI 方向与论文范围审查

日期：2026-09-05  
依据：最新执行报告 `SINGLE_TABLE_FFN_SERVER_EXECUTION_20260904(1).md` 与上传的 31 页 `main(20260905-052024).pdf`。  
用途：给研究负责人和 Codex 的分析与决策文件，不是已执行的新实验报告。本文没有访问服务器上的模型、梯度或原始 JSONL，也没有运行模型实验。报告中的模型结果视为报告陈述；本文明确标识推导与假设。

## 0. 最终判断

作为当前论文的提交前置条件，两个新目标设得过大；作为后续独立研究目标，它们仍然合理。原稿研究“support 与 allocation 是否是独立且有行为后果的变量”，新目标却要求“在严格部署约束下找到可行的生成系统”。前者不蕴含后者，也不需要以后者成功才能成立。

现在不是“已知只差 FFN/prefix-LM 这一块”。至少还有三种未分开的解释：输出策略改变、真正的内容能力变化、换表后学习计算与新坐标的兼容性。用另一个大训练配方同时处理它们，会再次失去解释力。

仅保留两个新方向：

1. **以生成决策边界和轨迹覆盖解释功能保留。** 将上一轮的点态 KL 边界具体化为可检验的 Native 首分歧机制；同时用固定输出要求的交叉控制，区分内容收益与输出策略偏移。首先只用已有模型，不新增训练方法。
2. **把 support–allocation 效应分解为即时坐标效应与安装历史效应。** 复用原稿已训练的小模型，必要时补齐冻结权重下的 support × allocation 交叉，而不是继续寻找新频率曲线。这一方向主要服务原论文的因果识别，不冒充解决全局单表方法。

第一项新增训练仍然是原计划的 N_compact，没有证据证明 prefix-LM 或新的模块配置优于它。第二方向可以并行复用已有资产，但不能阻止 N_compact 运行。Qwen Z/Y 是否继续投入，取决于需要验证的论文主张，不取决于必须完成整张大矩阵的惯性。

---

## 1. 最新证据：哪些结论必须更新

### 1.1 零训练：四个候选均未通过，幅度拆分已经结束

最新 OLMo 结果 [E1, Completed unit-amplitude decomposition]：

| 部署 | PPL retention | 普通任务 retention | EOS-weighted retention |
|---|---:|---:|---:|
| Z，原幅度 | 88.50% | 80.85% | 70.30% |
| Z，单位幅度 | 71.06% | 90.76% | 81.59% |
| Y，原幅度 | 66.32% | 87.28% | 84.06% |
| Y，单位幅度 | 41.08% | 75.96% | 72.37% |

不能再把去幅度作为下一项待检验方案。Z 的幅度在语言建模与任务端点之间有不同作用，Y 的原幅度在这三项上均优于单位幅度。结果否定这四个部署点的联合可行性，不否定所有静态表。

这些是 OLMo 上、既定验证行与指标下的结果。它们不能作为 Qwen 同名表的结果，也不能直接覆盖原稿使用不同公式、数据及 routing 的历史数字。

### 1.2 N128：承认总体通过，不能把分项损失抵消掉

独立确认集结果 [E1, Final fixed Native confirmation]：

- 总体生成分数保留 98.0556%，95% source-group bootstrap 区间 [94.3452%, 101.6479%]，按原先声明的 aggregate 88% 门槛通过。
- PPL retention 99.9736%。
- 格式/索引由 61/500 变为 47/500，15 个原正确实例丢失、1 个新增；净分数比 77.05%。原正确实例保留为 46/61=75.41%。
- 三组累计原正确 360 个，丢失 29 个、新增 22 个。净分数比 353/360=98.06%，原正确实例保留为 331/360=91.94%。后一个数是附加描述，不替代注册指标。

不能事后要求每组也达到 88%，然后撤销原 aggregate pass；也不能把这个 pass 写成逐项无遗忘。固定模型的案例区间不是训练 seed 区间。

### 1.3 新报告没有证明最终 Native 输出 KL 接近零

报告给出了 NLL 几乎不变，以及在指定 source-truth prefixes 上的训练 KL。它没有给出整个 Native 确认分布、原模型实际生成轨迹上的完整 KL。

对数据真值的变化为

$$
\Delta\mathrm{NLL}=\mathbb E_{(c,y)\sim\mu}\log\frac{p_0(y\mid c)}{p_\phi(y\mid c)}.
$$

这是可以正负抵消的量。只有在同一个 prefix 分布下，标签按原模型分布抽样时，相应期望才等于 forward KL。不能由平均 NLL 持平推导最终 KL 很小。

### 1.4 内容、输出格式、安装坐标仍未分开

NIAH 的 0→100% strict 改善主要是原来输出正确数字的句子、现在只输出数字；两者都正常终止。自然 QA 有拒答/错误实体修复的迹象，但 gold surface 计数不是语义判定。N128 与 Native 的同宽 source-only 对照保留了单证据 near/far 16/3→26/24 的改善，但仍没有排除输出策略解释。[E1, Completed simple tasks and NIAH; Latest completed source-only guard]

Qwen 的 16K 在其 configured Native 32K 内；32K 是适配长度之外，不是模型配置 Native 之外；64K 才是 2×配置 Native。不能借 OLMo 4K 分母声称 Qwen 的 4×Native 外推。

---

## 2. 静态表联合可行性的最小条件

### 2.1 真正的约束是函数交集，不是频谱覆盖

设 x 是合法的单表及已声明静态 attention 参数，所有短长请求使用同一 x。设 $S_N$ 是明确的 Native 指标，$U_L$ 是指定任务、布局、decoder 上的长程完整生成效用。问题是

$$
\mathcal F=\mathcal C_{\rm single}\cap
\{x:S_N(x)\ge0.88S_N(x_0)\}\cap
\{x:U_L(x)\ge u_{\rm required}\}.
$$

涉及多个 Native 端点时，按原协议分别约束；aggregate 与分项不偷换。目标长度本身没有定义 $U_L$ 或 $u_{\rm required}$。现有信息既没有证明 $\mathcal F$ 非空，也没有证明为空。

“静态表能在 Native 功能允许的范围内改变远端决策”是需要证明/验证的性质；不是只要有足够慢频率、phase coverage 或较高 effective rank 就自动满足的条件。

### 2.2 内容耦合和决策裕度是不可省略的对象

单层给定输入时，

$$
s_{ij}=\frac{a}{\sqrt d}\sum_k[A_{ijk}\cos(\omega_k\Delta)+B_{ijk}\sin(\omega_k\Delta)],\quad
\alpha_{ij}=\operatorname{softmax}_j(s_{ij}),\quad o_i=\sum_j\alpha_{ij}v_j.
$$

如果 m 是后续某一正确答案相对竞争答案的 margin，对这个 attention 节点的直接导数是

$$
\frac{\partial m}{\partial s_{ij}}
=\alpha_{ij}\langle\nabla_{o_i}m,v_j-o_i\rangle.
$$

这是链式法则与 softmax 导数的直接结果；全网影响需继续反向传播。它说明“增加证据 attention”是否改善决策，还取决于证据 value 所携带的信息和后处理如何利用它。高 attention、低 sensitivity、低频冗余均不能代替右侧的有符号决策效应。

最小的功能要求包括：任务相关差异仍可区分；与既有 Q/K 内容绑定兼容；有用信息到达可用的 value/residual 方向；正确答案与终止动作仍具有足够决策裕度。它们可以由不同实现满足，不要求所有 heads 都保形，也不要求严格复现整个 attention 图。

### 2.3 局部可行条件：存在共同方向，而不是只有低 Native 曲率

对固定目标 token 与竞争 token 的 margin，局部写成

$$
m_j(x+\delta x)=m_j(x)+g_j^\top\delta x+O(\|\delta x\|^2).
$$

一个局部修复方向至少要同时满足适当保留集合上的 Native 不越界，以及 long 待修复决策过界：

$$
J_N\delta x\succeq -m_N+\text{保留余量},\qquad
J_L\delta x\succeq \tau_L-m_L,
$$

再加有限步长与合法表约束。若保护所有原正确决策，条件强于 aggregate 88% 的要求。若只看当前竞争 token，还必须在实际前向中检查其它竞争 token；局部可行不等于有限改动后可行。

低 Native Fisher 方向不一定有正向 long 效用；二次代价中的零空间也不一定在有限步长下安全。对于 4×以上有限变换，不能把局部不可行写成全局不存在，也不能把局部可行写成生成必然成功。

**当前缺少的观测：**不是新的频率统计，而是任务条件下的完整决策裕度、相关决策对同一合法位置改动的联合响应，以及冻结权重是否已经能执行所需长程内容计算。已有几个表的失败只测到了有限个点。

### 2.4 不能要求不必要的严格算子同一性

原稿的 transplant obstruction 在固定、可逆、位置无关 Q/K 补偿下证明了全相对位置算子精确一致的频谱约束。这不是“Native 保留 88% 不可能”的定理。近似保留行为可以容忍内部变化，只要未跨越重要决策边界。[E2, §3.4, Appendix A.4]

数据依赖研究也表明，频率缩放的收益与长程任务依赖是否像训练依赖的尺度拉伸有关；其定理有明确的 positional utility 和场域假设，不能升级成真实生成的普遍保证。[R1]

---

## 3. 新方向一：决策边界与轨迹覆盖，而不是再缩小平均 KL

### 3.1 要检验的机制假设

**H1：**N128 的一部分“long strict 收益 + Native 格式/索引损失”源自回答策略的偏移，以及实际决策轨迹没有被 replay 充分约束；并不全部是新增检索/推理能力或低秩容量不足。

假设不是说“所有遗忘都是格式”。报告中格式/索引含真实索引错误，内容必须独立判定；也不能由有一批短答案训练样本便推断策略偏移一定发生。

选择 H1 的原因是同一个 N128 在 long 中更服从某种短回答格式，却在 Native 格式/索引上丢失原正确案例。这组方向相反的变化有可检验的共同解释。隐式任务推断、蒸馏轨迹分布错配在已有文献中都有前例；不能把这些概念本身称作新颖性。[R2,R4]

### 3.2 推导一：一个决策跨界需要多少 KL？

固定一个真实解码 prefix。原模型分布 p 的唯一赢家为 y，第二大概率 token 为 r，概率为 a=p_y、b=p_r。定义

$$
B(p)=a\log\frac{2a}{a+b}+b\log\frac{2b}{a+b}.
$$

则

$$
\inf_{q:\exists v\ne y,\;q_v\ge q_y}D_{KL}(p\|q)=B(p).
$$

所以

$$
D_{KL}(p\|q)<B(p)\Rightarrow \operatorname{argmax}q=y.
$$

**证明概要。** 固定竞争者 v，最便宜的边界将 q_y=q_v=(p_y+p_v)/2，其余概率保持 p。对 v 取并集，p_v 越大，代价越低，因此 runner-up 最便宜。完整证明见附录 A。

当概率差很小时，

$$
B(p)=\frac{(a-b)^2}{2(a+b)}+O\!\left(\frac{(a-b)^4}{(a+b)^3}\right).
$$

例如二分类 p=(0.5001,0.4999)，交换两个概率后 KL 约为 $8\times10^{-8}$，却已翻转 argmax。因此不存在不依赖原决策裕度的“平均 KL 足够小就保住所有 greedy 决策”阈值。

这是 KL 投影和分类校准的基础性质，**不是声称新发现一个未发表的普遍数学原理**。[R3] 对本项目的价值是它把“保留失败”转成可观测的 prefix 级约束，而不是要求新增一个魔法 KL 权重。

### 3.3 推导二：完整 greedy 保留只需保护原路径，平均 replay 则未必覆盖它

对一个输入，原模型生成 $y^0_{1:T}$，包含正常终止 token。只要新模型在所有原前缀 $(c,y^0_{<t})$ 上仍选择 $y^0_t$，就逐步归纳得到完全相同的 greedy 输出。

因此：

- 保护原模型固定 greedy 轨迹，不在逻辑上要求先做 RL 或 student on-policy rollout。
- 原轨迹上的首次决策变化，不能归因于已经进入错误的 student 轨迹；那是在首次分歧之后的放大机制。
- token 改变不一定是语义遗忘；同义输出或满足要求的另一种答案必须另行判断。
- 多种 stop IDs、logits processors、tie-breaking 都要纳入相同的实际 decoder；原始模型 logits 的 KL 不能直接证明另一个处理后 decoder 的同一性。

GKD 研究 student 生成分布与训练轨迹的失配，这与这里“保住一个确定 teacher 路径”的充分条件有关，但不是同一个实验目标。[R4]

### 3.4 推导三：平均 KL 要转换成生成保留，需要覆盖与 margin 两座桥

设 $e(c)=D_{KL}(p_0(\cdot|c)\|p_\phi(\cdot|c))$；$\mu$ 是 replay prefix 分布；$\nu$ 是 Native 输入上原模型完整 greedy 路径的长度归一化占用分布：

$$
\mathbb E_\nu f=\frac{\mathbb E_X\sum_{t=1}^{T(X)}f(c_t)}{\mathbb E_X T(X)}.
$$

若 $\nu\ll\mu$ 且 $d\nu/d\mu\le C$，$\mathbb E_\mu e\le\epsilon$，则任意 $\eta>0$ 下

$$
\Pr(\text{greedy 路径改变})
\le
\Pr\left(\min_t B(p_{0,t})\le\eta\right)
+
\frac{C\,\mathbb E T\,\epsilon}{\eta}.
$$

证明：排除低 margin 轨迹后，发生分歧意味着至少一个原路径 prefix 有 KL≥η，因而整条路径的 KL 总和≥η；应用 Markov 与分布覆盖即可。

**边界：**这是一条有假设的充分界，可能很松。当前有限 replay 数据无法认证总体覆盖常数 C，source-truth 前缀也不自动覆盖 teacher 生成前缀。因此不能将它填入数字，伪造 88% 的保留证书。

它准确暴露三个可能缺口：原决策脆弱、prefix 分布不覆盖、每条生成包含多个机会。降低平均 KL 只能直接处理其中一个因素。

### 3.5 最便宜的区分实验：不更新模型

只使用现有 N0/N128；不调表、gain、decoder。两部分合并为一个有明确上限的诊断，先复用已有输出，不重做整个评估工具链。

**A. Native 首分歧诊断。** 用现有确认输出定位 29 个原正确→错误案例，并取预先规则匹配的保留案例。共同前缀上的首个不同 token 可从已保存 token IDs 得到。只在所需 prefix 上计算原模型和 N128 的完整 next-token 分布，记录 p 的 top-2 概率、B(p)、KL、q 对原 token 的 margin，以及分歧属于内容、格式、终止或合法改写。

这是对已暴露数据的机制分析，不是新泛化证据，也不允许将失败案例回灌训练。需要估计整体轨迹风险时，另在未用于选择方法的 Native calibration 轨迹上做固定位置采样；不能由失败条件样本估计总体发生率。

**B. 内容 × 输出要求交叉。** 取已锁定的 16 个语义组（8 单证据、8 binding；不按 N128 表现筛选），沿用 compact/near/far、两种合法证据世界，增加一个预先声明的合法输出要求，例如只给答案 vs 固定短句回答；不提示答案，不追加推理示范。N0/N128 同样运行。总计 16×3×2×2×2=384 个生成，其中原格式已有输出可复用，最多新增 192 个。

不假设两个格式难度相同。检查原模型 compact 在每种格式下的可解性，分层报告；不能看到新格式结果后再选第三种。读数并列为内容断言正确、格式符合、正常结束、双世界都正确。若合法自然回答不能可靠判定，保留 ambiguous，不以答案字符串出现代替语义判定。

此小面板是区分机制的 development 诊断，不是高置信确认，不产生新的参数选择权限。

### 3.6 结果如何解释，什么会否定 H1

| 结果 | 支持什么 | 后续 |
|---|---|---|
| N128 的 strict 增益主要随输出要求变化，语义收益很小；Native 损失集中于未覆盖的格式/终止决策 | 输出策略偏移 + 决策覆盖具有解释力 | N_compact 是更关键对照；再决定是否做一项 trajectory replay 比较 |
| N128 在两种要求下都修复真实错误实体/关系，双世界响应正确，near/far 语义差距缩小 | 存在不能由原先格式解释完全覆盖的内容收益 | 保留 N_compact，然后 Z/Y 才有“位置适配”意义 |
| 被保护的 training 轨迹也丢失，且这些位置的 KL 很高/边界被穿越 | 不是单纯 held-out 轨迹漏覆盖 | 检查约束执行、优化和权衡；不能用增加 teacher 轨迹数量直接解释 |
| 大部分损失是跨格式稳定的真实索引/关系错误，输出策略操纵没有改变差异 | 否定“主要是表面策略偏移”的强版本 | 内容冲突或表示改变需另证；不立即宣布 FFN 容量不足 |

若 H1 获支持，唯一允许的后续训练比较：在完全相同 prompts、参数、task loss、updates、replay 组别和 KL 位置总预算下，比较 source-truth prefixes 与 teacher 实际生成 prefixes。数据只来自训练/校准源，新确认集独立。含自然文本的保留部分不必因为指令轨迹实验而全部替换。

这项比较不叠加 prefix-LM、不增加 rank、不增加 replay 总预算，不用新确认失败样本选训练 prefix。若覆盖确实改善，但同预算下决策损失与独立分项保留没有可信改善，则关闭“轨迹覆盖足以解决本轮遗忘”的假设。弱检验只有宽区间时，应写 inconclusive，不能当作反证。

### 3.7 区分内容学习、轨迹覆盖、参数不足的判别层级

1. **内容学习是否发生：**先看语义双世界、输出要求交叉、N_compact，不看 strict 或 gold-surface 单项。
2. **保留状态是否覆盖：**看 teacher 实际路径、first-divergence 与相应 KL/margin，而不是只看 replay 均值。
3. **覆盖后的目标能否拟合：**训练决策约束已满足、held-out 失败，优先指向泛化/覆盖；连训练目标也不满足，可能是优化、约束权衡或容量，不能直接归因某模块。
4. **参数空间不足要额外证据：**局部可用 $J_N\delta\phi=0,\ J_C\delta\phi=b$ 检查可达性；满足严格等式时等价于 $b\in\mathrm{range}(J_C P_{\ker J_N})$。但这是局部线性化的强保留约束，失败不代表允许 88% 行为保留的非线性问题无解。不要把 LoRA 初始化处退化切空间误作全局 rank 下界，也不要为了这个记账构造全参数 Hessian。

FFN/attention 参数预算相同的比较只能识别预算分配价值，不证明 FFN 必要；优化失败也不是表达能力不足的证书。最近 replay/容量研究提供了这一区分的实证背景，但不能由它推断当前 Qwen 已饱和。[R5]

---

## 4. 新方向二：即时坐标效应与安装历史效应的因果分解

### 4.1 为什么它比再发明一张表更适合这篇论文

原稿的最强识别实验是相同 support 下改变 z 的三 seed 训练对照；而主图还展示 target-retargeted support 使排序反转。[E2, Fig.1、§2.1、Table9]

这不是无效证据。它识别了**训练 allocation 与部署 support 策略的端到端交互**。但它没有自动识别“同一组冻结权重中，运行时 support 与 runtime allocation 的直接交互”。主文的 interaction 语言与附录 C.1 最后“不是 interaction law”的限制应保持一致。

### 4.2 精确分解：旧对角比较还混有安装历史

记 $F(W,S,z)$ 为一个完全指定的评价损失；S=(a,R)，G/C 为已有 geometric/Cosh 形状，$W_G,W_C$ 为分别在其训练表上得到的权重。

原来的端到端差为

$$
D(S)=F(W_C,S,z_C)-F(W_G,S,z_G).
$$

在同一 $W_G$ 上插入中间项，有精确分解

$$
D(S)=\underbrace{F(W_G,S,z_C)-F(W_G,S,z_G)}_{\text{该权重上的 runtime allocation 效应}}
+
\underbrace{F(W_C,S,z_C)-F(W_G,S,z_C)}_{\text{固定 runtime allocation 下的安装历史效应}}.
$$

所以 $D(S_1)-D(S_0)$ 的反转可能来自第一项、第二项或两者共同变化。另一种参照权重给出另一种合法分解；不能把某一分解称作不依赖参照的唯一“贡献比例”。

要直接识别 runtime 交互，对固定 W 测

$$
I_W=[F(W,S_1,z_C)-F(W,S_1,z_G)]-[F(W,S_0,z_C)-F(W,S_0,z_G)].
$$

$I_W$ 本身的符号是一项交互；两个 support 下 allocation 效应异号才是“排序反转”。不能把 $I_W\ne0$ 等同于发生 crossover。

**H2：**已有 reversal 至少部分是在固定成熟权重内仍存在的 runtime support–allocation 交互，而非只来自不同训练历史对 support 的不同耐受。

H2 为假也有科学意义：论文应强调安装历史条件化的 allocation，而不是暗示一个脱离权重的普遍最优形状。

### 4.3 最便宜的区分实验

不训练任何新模型，不优化任何表。复用 151.9M 三对已训练 checkpoints；主长度固定1024，support仅使用已有 training support 与已有该长度 target-matched support，shape仅使用已有 z_G/z_C。

固定每个 W，完整交叉这两 support、两 shape。总计 3 seeds ×2 weights ×2 supports ×2 shapes=24 个条件，每条件原有32 anchors；已有且身份完全一致的条件直接复用。先只跑缺失格，不为这个问题加入更多长度或形状。

读取原稿 C.2 的已有 weights×derived-table crossing，检查它的实际频率字节能否填入本 factorial。已有 derived tables 不一定等于这里严格固定 shape 的 $(S,z)$ 组合；不能仅凭名称复用。如果 checkpoints 缺失，不为补图重训1.5B或151.9M；收紧论文措辞，方向停止在现有证据边界。

C.2 已有两训练 seeds 的强 co-adaptation 交互，故这项增量不是再证明“换表会崩”。应只回答：**把 W 固定后，support 改变是否仍改变同一 allocation 对比的方向/幅度。** 若 off-diagonal 全部进入近乎失效的区域，不能把其巨大 NLL 差异包装成有用的最优规律。

该试验可以使用 long NLL，因为它是已经固定因素的解释性检验，不是选取新 zero-training 方法。不会因得到某个较好格就将其选为部署表；不使用其结果声称生成能力。

### 4.4 判决与否定条件

| 结果 | 合理结论 |
|---|---|
| 多 seed 中，固定 W 的 allocation 效应随 S 稳定改变，并在两个 W 下有一致结构 | 支持 runtime 坐标交互；可加强原稿，但仍不是普遍 interaction law |
| 对角策略反转存在，固定 W 内无对应反转，或 I_W 很小 | 反转主要不是所声称的直接 runtime crossover；加强安装历史解释，收紧标题/正文语言 |
| 两个 W 得到相反结构，或 off-diagonal 全失效 | 效用强依赖已学系数；不能推出统一频率曲线 |
| seed 间不稳定、区间宽 | 机制未识别；原三 seed 固定 support 因果效应仍保留，不补训练 sweep |

以上不以胜过 YaRN 为成功标准，也不以找到更好的表为输出。这是对已有科学主张的因果分解，不是第三个方法项目。

---

## 5. 原稿究竟缺哪一块，不必等哪一块

### 5.1 原稿已有证据与新目标不是同一个证明责任

原稿明确区分 pure-z frozen 对照、带 Native/long routing 的系统、matched adaptation 和 from-training。旧系统的 Native 精确保留来自 routing，原稿并未把它归于那张长表自身。[E2, §4.1–4.3、Appendix E.2]

现在要求全球一张表同时服务所有长度，增加了一个旧结论并不需要满足的约束。旧 arithmetic 表、旧gain、旧scorer及旧routing结果也不等于最新 log-profile 的四个冻结候选。最新失败不是对旧结论自动反证；必须先匹配物理系统和指标。

原稿的 1.485B 与 8B 旧适配分别包含 task-family transfer 和 source-NLL 因果效应，也不等于新目标“完全未见任务上的通用生成、≤16K训练且真实超Native”。应继续保留这些层次，而不是把全部实验打成一场未完成的系统竞赛。

### 5.2 三处最值得完善的理论与表述

**第一，几何解释不等于成熟模型选择器。** trace/effective-rank 恒等式严格，但 block whitening 消除了能量和条件数；它没有模型 coefficients 或决策 margin。Cosh 最优是相对于声明 surrogate 的最优，不是 function-space retrofit 最优。这些限定应留在主文显眼位置，而不是依赖附录救场。

**第二，slow-collapse 不能独自解释所有强结果。** 主三 seed 训练对照取 base=Ltrain=256、K32。按原稿公式最慢频率有 $\omega_{min}L=256^{1/32}\approx1.189$，不在 $\omega L\ll1$ 的深慢频极限。因此该实验支持“allocation 在深慢频极限之外仍有后果”，而不是单独验证“清掉 dead frequencies 就获得全部收益”。这不是否定 Proposition2，而是准确区分定理适用域与效应 owner。

**第三，零训练与无 long-outcome 方法开发不相同。** 附录 E.2 记录 core-four 用于 method selection，之后 OLMo unseen-nine 确认；Qwen core-four 属 development families。可以说无参数训练、公式构造不读 task labels、具有后续确认，但不能把整个历史开发过程改写成从未使用 long benchmark。新的 Native-only 方法设计目标比原稿已证明内容更强。

### 5.3 新结果纳入论文的最低要求

- 最近 N128 的 aggregate retention pass 可以作为声明分布下的结果；格式/索引损失必须在同一处披露。
- N128 使用 Native Qwen 表，其成功本身不为 Hybrid-RoPE allocation 提供增量证据；至多是适配可运行的正对照，直到同 checkpoint Z/Y 对比完成。
- 重新检查原稿最强生成 headline，例如750M的严格0→77.5%：使用已有 raw outputs 分解语义、格式和终止。不能因新Qwen存在格式混杂就断言旧结果也一样；但这是风险很高、成本很低的必要解释核对。
- 不增加不相关视频、更多规模或大量后处理。现有原稿已很宽，增量应修补主张而非扩大目录。
- 对“理论给出唯一/最优/无法恢复”的词逐条检查：已有数学不足以支持整个模型类的 global claims。选择一个可检验反转/条件结论比新增多个不闭合理论框架更有价值。

### 5.4 投稿目标建议

核心主张保持：**固定 support 下 allocation 对行为有独立作用；它的效用依赖部署 support 与权重安装历史；几何改善并不自动跨越生成决策边界。**

将“单表≥4×且Native≥88%”列为未完成的独立工程目标，而不是正文成立的先决条件。将 tiny adaptation 的要求缩为：在一个有明确 compact 可解性、自然来源、合法反事实的任务族上，确认真正语义收益及声明范围 Native 保留，再报告更远长度边界。无跨任务实证则不宣称跨任务通用能力。

这会减少不必要证明责任，但不消除 novelty 风险：仅重新参数化频率并不够。论文价值依赖因果控制、清晰反例和实际行为后果；两个新方向本身也不是“必然中稿”的保证。

---

## 6. 给 Codex 的实际顺序与预算边界

### 6.1 本轮不变项

不改变已完成 run 的身份、指标、aggregate 88% 判定。冻结 OLMo Z/Y及两单位幅度失败状态。无新gain、表曲线、headmask、rank或prefix权重搜索。N_compact仍使用原计划；不因本文临时改成一个含多项新增loss的配方。

### 6.2 顺序

**先做无训练的信息回收。** 从已有输出完成语义/格式/停止三分；运行方向一的小型分歧/输出要求诊断。工具已能读取有效原始产物时，不重构评估平台。新增GPU诊断预算不超过当前一个N训练run量级；历史1721.87秒仅为报告中的参考，不是本次运行时间保证。

**第一项新增训练：N_compact。** 同样语义曝光、答案标签、updates、LoRA、Native replay；替换长输入为对应compact。报告输入token与FLOPs不同，不为了伪配平加入长噪声。若全部收益主要由输出模式适配解释，不扩大为长程内容学习。

**并行的小模型论文收尾：方向二。** 优先CPU整理已有factorial格，只执行缺失推理；资产缺失直接收紧表述，不重训。

**随后才完成需要的同模型Z/Y。** 若主张要保留“位置底座辅助内容迁移”，这些比较不能省。使用N128旧配方，别同时加入prefix或teacher轨迹新方案。Z26无法可靠恢复时，从已验证引擎和原checkpoint重跑，不掩盖resume问题。double精度不足是族内限制，不再阻断single实验。

**条件性新增训练：仅一项trajectory replay比较。** 只有方向一支持相关解释时才启动。它替换prefix覆盖方式，不同时更改参数空间/任务数据/steps；之后使用新确认池。prefix-LM继续保持未证实状态，不因已有代码就获得科学优先级。

### 6.3 缺少观测时的明确结论

| 想回答的问题 | 当前真正缺的观测 |
|---|---|
| 是否有全局单表联合可行点 | 未被结果挑选的合法候选在联合完整生成/Native端点上的证据，或严格限定类的不可行证书 |
| 平均KL为什么没保住某些决策 | 原模型实际prefix上的KL、top-2裕度与首分歧，不是NLL均值 |
| N128是否学到了内容 | 语义双世界判定、输出要求交叉与N_compact |
| 是否必须训练FFN | 在明确定义预算/目标下的可达性或对照；现有全模块更新不回答必要性 |
| 是否是参数空间不够 | 充分覆盖后训练约束的可行性，以及排除优化失效的证据；现有报告没有 |
| support反转是不是runtime交互 | 固定W的support×shape缺失格，而不是不同W的对角策略差 |
| 新方法是否提高ICLR说服力 | 对原论文主张新增的独立语义/因果证据，而不是单独一项strict分数 |

---

## 附录 A：决策 KL 边界的证明与实现要点

令 p 位于有限概率单纯形内部，y是唯一最大项。对固定v≠y，求最小 $D_{KL}(p\|q)$ 且 $q_v\ge q_y$。无约束极小值q=p违反约束，因此最优落在 $q_y=q_v=t$。

拉格朗日驻点给出其它 $q_k=p_k/\lambda$，$t=(p_y+p_v)/(2\lambda)$。归一化给λ=1，因而最小值

$$
B_v(p)=p_y\log\frac{2p_y}{p_y+p_v}+p_v\log\frac{2p_v}{p_y+p_v}.
$$

对固定a=p_y，$\partial B_v/\partial p_v=\log[2p_v/(a+p_v)]<0$。因此最大竞争概率r给最小值。对所有v的半空间取并集，得到B(p)。若要求严格超过而非相等，最小值变成同一个下确界。

B=0或解码存在tie时无正裕度证书，不能除以0或调整阈值来宣布保留。若分布计算含fp下溢，使用log-softmax与稳定log-sum，不将full-vocab KL替换成top-k近似后沿用保证。

稳定的等价写法：s=a+b，d=(a-b)/s，

$$
B=\frac{s}{2}[(1+d)\log(1+d)+(1-d)\log(1-d)].
$$

本文对3、7、13类随机分布进行了CPU约束优化数值自检，数值最小值与闭式差异在约$10^{-16}$内；这只是实现自检，不替代上述证明，更不是模型实验证据。

### 诊断伪代码

```python
# No optimization, no table selection, no on-the-fly decoder tuning.
for case in locked_diagnostic_cases:
    teacher_ids = archived_or_fixed_teacher_greedy(case)
    student_ids = archived_or_fixed_student_greedy(case)
    first_difference = first_token_difference(teacher_ids, student_ids)
    prefixes = declared_prefixes(teacher_ids, first_difference)

    for prefix in prefixes:
        p = actual_decoder_distribution(teacher, case, prefix)
        q = actual_decoder_distribution(student, case, prefix)
        y, r = top_two(p)
        barrier = exact_top_change_kl_barrier(p[y], p[r])
        kl = full_vocabulary_kl(p, q)
        winner_unchanged = argmax(q) == y
        # kl < barrier implies unchanged; the converse is NOT asserted.
        save(prefix, barrier, kl, winner_unchanged,
             student_margin_for_teacher_token(q, y))

    save_separate_content_format_termination_labels(case)
```

不会用B或KL/B给新表排序，不以该指标取代正式完整生成指标。它只解释已固定系统中的原决策失守。

## 附录 B：固定权重交叉的伪代码

```python
# Existing support values, existing shapes, existing checkpoints only.
L_eval = 1024
for seed in [42, 137, 256]:
    for training_shape in ["G", "C"]:
        W = resolve_existing_checkpoint(seed, training_shape)
        for support in ["training_support", "existing_target_support_1024"]:
            for runtime_shape in ["G", "C"]:
                table = exact_existing_coordinate_embedding(support, runtime_shape)
                assert_fixed_gain_operator_tokenizer_and_anchors()
                receipt = find_exact_matching_receipt(W, table, L_eval)
                if receipt is None:
                    receipt = evaluate_fixed_tail_nll(W, table, L_eval,
                                                      existing_32_anchors)
                preserve_raw_losses_and_hashes(receipt)

# Report all cell values, allocation contrasts within fixed W/S,
# runtime interactions I_W, and the original diagonal policy contrasts.
# Never promote the best cell to a new deployed model.
```

若缺失权重、表定义不明确或旧receipt不能匹配，停止相应增量，不能猜表或编造缺失格。以训练seed为重复单位，anchors是seed内配对；三个seed不是大量独立样本。

## 来源索引

**[E1]** `SINGLE_TABLE_FFN_SERVER_EXECUTION_20260904(1).md`。重点：Completed unit-amplitude decomposition；Final fixed Native confirmation；Paired natural validation；Completed simple tasks and NIAH；Latest completed source-only guard；2026-09-05 priority amendment。报告内部包含运行hash，本稿未逐个下载或验证服务器raw数据。

**[E2]** `main(20260905-052024).pdf`，*RoPE Has a Spectral Budget*，31页。重点：第1–5页核心问题与定理；第6–9页冻结/routing/适配与讨论；第25页Table9与其限定；第26页C.2；第29–30页method-selection与routing边界。

**[R1]** Wu, Liu & Jadbabaie. *How Data Shapes RoPE Frequency Usage: From Positional Scale Matching to Length Generalization*. arXiv:2607.07678v1, 2026。数据依赖、条件尺度匹配；其positional utility不是实际生成效用。

**[R2]** Kotha, Springer & Raghunathan. *Understanding Catastrophic Forgetting in Language Models via Implicit Inference*. arXiv:2309.10105; ICLR2024。为输出/任务条件化解释提供已有背景，不直接证明当前N128机制。

**[R3]** Ávila Pires & Szepesvári. *Multiclass Classification Calibration Functions*. arXiv:1609.06385, 2016。surrogate risk与分类决策风险转换的基础文献。本文的点态KL投影公式在附录独立推导，不声称该论文逐字给出同一公式。

**[R4]** Agarwal et al. *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes*. arXiv:2306.13649v3; ICLR2024。训练与student rollout分布错配；不与固定teacher greedy路径保留混为一个命题。

**[R5]** Marek et al. *Forgetting in Language Models: Capacity, Optimization, and Self-Generated Replay*. arXiv:2605.26097v1, 2026。容量、优化与replay的实证区分；不能据此推断当前小模型饱和。

**[R6]** *LongReD: Mitigating Short-Text Degradation of Long-Context Large Language Models via Restoration Distillation*. arXiv:2502.07365v3。RoPE变化后的短程分布漂移与训练遗忘已有先例，恢复蒸馏不是未被占据的方法概念。

**[R7]** Wertheimer et al. *Frayed RoPE and Long Inputs: A Geometric Perspective*. arXiv:2603.18017v1, 2026。Q/K聚簇、attention sink与避免不必要信息混合提示“取到证据”之外还有attention功能；本稿未据此新增headmask或频率设计。

---

**执行结论：**把工程野心与论文证明责任分开。立即的信息增量来自现有输出的决策机制解释；第一项新增训练仍是N_compact。原论文的高ROI理论补强来自安装历史条件化的坐标因果分解，不来自再承诺一条同时解决所有checkpoint、长度和任务的静态频率定律。
