# RoPE ICLR 2027：交叉审核、理论推导与 Codex 执行方案

日期：2026-09-06。目标：用有限计算建立相对强基线的实际增量、能够检验的非几何分配理论，以及可靠的 Native 与长生成结果。

本报告交叉核对《大修与实验执行清单.md》、上一份《RoPE_ICLR2027_Major_Revision_20260906.md》、本会话的 31 页稿件及最新实验综述，并重新检查 MrRoPE、LeRoPE、YaRN、AdaRoPE 与数据尺度论文。本文的模型分数来自附件；新运行只包含随附 Python 的 CPU 数学核验，没有新增语言模型训练或评测。底层服务器日志、训练源码和检查点尚需 Codex 对照本地资产核准。

## 0. 本轮决定

核心研究问题改为：

> 在指定支撑、训练暴露、部署变换和适配预算下，怎样选择有限的 RoPE 指数分配，使实际 Native 与长上下文表现更好？

保留 `frequency = base ** exponent`，但参数化、固定端点、三种训练阶段均不单独承担首要新颖性。

优先执行两件工作：先用已有 151.9M 检查点检验 Cosh 的训练收益是否在 MrRoPE-Pro/YaRN 延展后保留；同时在 OLMo 1.485B 上完成可靠 YaRN、MrRoPE-Pro、当前 Z 的冻结对照及同配方适配。有限步训练与部署风险的推导负责解释分配何时有价值；实际任务与 Native 生成负责确认收益。

我调整上一份方案的三点：

1. 把“重新训练 MrRoPE 表作为 scratch 初始化”降为扩展研究。已有检查点叠加强推理变换，ROI 更高，也更符合 MrRoPE 的原始用途。
2. 不把所有候选表、所有参数体制、所有模型做笛卡尔积。零训练和已有 scratch 评测先覆盖候选，训练只保留可靠强基线及本方主要候选。
3. 保留最新报告已转向的 1.485B 全参数适配作为成熟模型主比较；LoRA 做同表、同数据、相同 token 里程碑的桥接比较。附件已经多次执行 all-linear LoRA 与完整答案监督，不能仅以 LoRA 便宜为由再把同一配方重跑一遍。

另一份方案的补充也需要修正：LeRoPE 已经包含“独立运行学得频率冻结后再训练”，并研究 LeRoPE 与 YaRN 的推理组合。因此，“训练分配的收益经过强延展仍存在”是必须补的比较，尚不足以独自构成新的主张。[R3 §6–7, App. D]

### 0.1 三项交付及其最低证据

| 交付 | 实际要回答的问题 | 主证据 |
|---|---|---|
| 方法价值 | 在同样的 Native 要求和计算预算下，本方分配是否比强基线有用？ | MrRoPE-Pro/可靠 YaRN/本方的匹配比较；至少一个独立确认 |
| 理论增量 | 何时分配改善学习，何时换表代价或部署变换抵消收益？ | 训练表、部署表、有限步数共同进入风险；至少一个未参与拟合条件下的预测 |
| 真实能力 | 改善是否落在完整生成，且保留原模型已经会的能力？ | 实际长任务、固定长表 Native、输出与终止诊断、训练重复 |

三项相互支持。一般线性代数定理、CPU 恒等式通过、以及代理指标改善，都不能替代模型结果。录用概率无法由这三个检查项机械换算。

---

## 1. 交叉审核：哪些结论已经可以采用

### 1.1 MrRoPE 重合范围

MrRoPE 用

\[
\omega'_j=\omega_j/\prod_{d<j}\lambda_d.
\]

令 \(x_j=-\log\omega_j\)、\(h_j=x'_j-x_j\)，则

\[
h_j=\sum_{d<j}\log\lambda_d,\qquad
\log\lambda_j=h_{j+1}-h_j.
\]

因此，在首频率固定的情况下，log-frequency 位移和 radix 因子之间是一一对应的累积/差分转换。其具体 Uni/Pro 在正常存在高低频保护段时，共享高频端点与低频端点除以扩展倍率，区别就在中间频率。[R1 §3]

本方固定端点实验依然有因果控制价值。其新增内容要体现在定量规律、学习结果及更强对照上。允许某些频率变快的 Cosh 分配，和 MrRoPE-Pro 的具体变换并非相同数组；这项区别需要经过实验证明其价值。

### 1.2 LeRoPE 是必须纳入的另一条边界

LeRoPE 的 §6 用独立运行学习的频率固定训练新模型；§7 和 App. D 用实际频率的训练旋转圈数定义 NTK-by-parts/YaRN。它也比较了不同 geometric base。[R3]

因此以下表述应撤出首要贡献：“非几何表可以用于从零训练”“频率可以冻结而保留收益”“好的训练频率与 YaRN 互补”。它们仍可作为本方结果的一部分，但需要进一步给出解析分配的成本优势、固定预算下的可预测边界，或实际更好的 Native—long 取舍。

AdaRoPE 已经研究 head-specific 频率与缩放，LongReD 已经研究原生能力恢复蒸馏，数据尺度论文已研究依赖尺度与频率使用。[R4–R6] 本方新的全局共享静态表、有限学习分析与严密控制可以与这些方向区分；仅把已有组件组合起来仍然不够。

### 1.3 需要立即更新的实验判断

| 问题 | 本轮采用的判断 |
|---|---|
| Y2 是 faithful YaRN | 按 MD 的构造描述不成立。smoothstep 与 `sqrt(1+0.1 ln s)` 的 Q/K gain 均需要纠正 |
| 2K vs 16K 证明目标物理长度必要 | 不成立。相位、竞争项和数据暴露一起变化，且统计单位也有问题 |
| Qwen 29/32 证明 OLMo 500M 可达到约 91% | 不成立。它是该 checkpoint/评测的可行性证据，训练历史与预算不能移植 |
| rank≈2 意味着只有两对频率有用 | 不成立。它是特定测度下的位置函数子空间有效秩，内容通道仍可能有用 |
| E1 loss/EOS 改善证明仅欠训练 | 尚未确立。它说明训练尚未完成目标；继续增加 tokens 的充分性未知 |
| Cosh 在 fixed support 获胜证明总体最佳 | 不成立。已有 support retargeting 结果发生反转，强基线扩展尚需补齐 |
| 原生聚合 NLL 基本不变代表生成保持 | 不成立。最新 Qwen 和 OLMo 报告已有具体格式、EOS 和决策损失 |

这些判断分别来自 P1/P2 的协议与 R1/R2 的实现核对。MrRoPE 审稿材料中作者做过 8K 微调；其补充长文档 PPL 使用 8K 滑窗。那项结果不能替代完整 32K–128K 可见上下文能力证据。[P4]

### 1.4 已有失败配方必须保留身份

历史执行方案已包含真实长输入、完整答案与 EOS、最差 margin、all-linear LoRA、Native 教师 KL，以及恢复阶段。新报告不能把同样的组件重新排列后称为解决方案。[P5]

本轮改变的主要变量是：可靠对手、训练/部署表的明确身份、全参数与 LoRA 的匹配比较，以及有限训练过程的可检验分析。后面的 Native 决策约束仅在“长任务已经改善、Native 仍失败”时作为一个单独增量使用。

---

## 2. 统一数学对象：训练用什么表，部署用什么表

用三个数组区分阶段：

\[
\Omega_P\quad\text{预训练表},\qquad
\Omega_A\quad\text{适配表},\qquad
\Omega_D\quad\text{部署表}.
\]

同时记录 \(W_0\)、适配参数集合 \(\mathcal A\)、输入/目标分布、优化器和预算。

从零训练后的强扩展通常是：

\[
W_P=\operatorname{Train}(W_{\rm init},\Omega_P),\qquad
\Omega_D=T_s(\Omega_P).
\]

成熟模型换表适配通常是：

\[
W_A=\operatorname{Adapt}(W_0,\Omega_A,\mathcal A,B),\qquad
\Omega_D=\Omega_A.
\]

零训练是后一个流程中 \(B=0\)。LoRA 与全参数改变允许更新的权重和实际学习动态。不同阶段不必强制使用同一张 Cosh 表，但不能把不同表的收益都写成 Cosh 的收益。

定义实际研究量：

\[
V(\Omega_P,\Omega_A,\Omega_D,\mathcal A,B)
=\left(\mathcal L_N,\mathcal L_L,\mathcal S_N,\mathcal S_L,C\right),
\]

其中 \(\mathcal L\) 为声明的概率损失，\(\mathcal S\) 为实际生成指标，\(C\) 为选择、训练和评测成本。论文要改善的是这组实际量的取舍。

### 2.1 一个能直接输出数组的共同构造

保留当前解析实例。令

\[
u_k=(k+1/2)/K,\quad
q_k(\tau)=1-\frac{\operatorname{asinh}((1-u_k)\sinh\tau)}{\tau},
\]

\[
z_k(\tau)=\frac{q_k-q_0}{q_{K-1}-q_0},\qquad
x^P_k=a+Rz_k(\tau).
\]

\(\tau=0\) 取连续极限 \(z_k=k/(K-1)\)。当前 151.9M 主协议保留已经测试的 \(\tau=4\)，不在这组既有检查点上虚构强度优化。

MrRoPE-Pro 用明确的边索引表示。设高频边界顶点为 \(l\)，低频边界顶点为 \(h\)，共有 \(n=h-l\) 条转换边。令 \(t=\operatorname{clip}(k-l,0,n)\)，则

\[
m^{\rm Pro}_k=\frac{t(t+1)}{n(n+1)},\qquad
m^{\rm Uni}_k=t/n.
\]

于是共同参考变换是

\[
\boxed{x^D_k=x^P_k+\log(s)m^{\rm Pro}_k,\qquad
\omega^D_k=\exp(-x^D_k).}
\]

其边因子恰为

\[
\lambda_i=s^{2i/[n(n+1)]},\quad i=1,\ldots,n,
\]

乘积为 \(s\)，两端共享。不依赖任务标签即可输出完整数组。

主配对先由共同几何参考表确定 \(l,h,m\)，再将同一个 \(m\) 施加到 Geo/Cosh 训练表。这样

\[
x^D_C-x^D_G=x^P_C-x^P_G
\]

精确成立。第二个敏感性版本用各自实际频率的旋转圈数重定边界。两种问题不同，应分别命名为“共同参考变换”和“按自身频率分段的推广”。前者提供清楚控制，后者检验实际扩展策略；都不能假装是 MrRoPE 官方代码原样支持非几何表。

端点或阈值落在数组之外时，严格记录边界退化。尤其 151M 的小训练窗口和小 base 可能没有完整保护段；不能偷偷移动阈值以制造正结果。官方几何输入复现与本方推广用不同配置字段。

### 2.2 对叠加矩阵的正确解释

对每个训练种子，令

\[
\Delta_T=L(W_C,T\Omega_C)-L(W_G,T\Omega_G),
\]

\[
I_T=\Delta_T-\Delta_{\rm identity}.
\]

\(\Delta_T<0\) 表示训练分配的收益在该强扩展后仍然存在；\(I_T<0\) 才表示相对原始差距进一步扩大。实际价值不要求一定有正协同。相同终点表在不同训练历史下的比较则测共适应，不应和这组部署组合混成同一估计量。

---

## 3. 理论主线：有限学习预算和部署风险的完整推导

### 3.1 模型、共同任务与适用范围

定义有序的完整 sin/cos 特征：

\[
\Phi_\Omega(\Delta)
=[\cos(\omega_1\Delta),\sin(\omega_1\Delta),\ldots,
\cos(\omega_K\Delta),\sin(\omega_K\Delta)]^\top.
\]

考虑固定内容条件下的关系函数

\[
f_{\Omega,c}(\Delta)=\Phi_\Omega(\Delta)^\top c.
\]

所有候选面对同一个目标函数 \(g\)，或同一个提前定义的任务分布。训练、Native 评价、长评价可以有不同距离分布。不能为每张表选择一个更容易表示的新目标。

这是一个可精确求解的 RoPE 位置关系学习模型。对真实 Transformer，其内容系数、深层特征、normalization 和优化器会变化；下述结论不直接等于完整 LLM 的训练定理。新增贡献必须来自其可验证预测和实际使用价值。

### 3.2 训练矩阵与受限更新

在适配表 \(\Omega_A\) 下，定义

\[
G_A=\mathbb E_A[\Phi_A\Phi_A^\top],\qquad
b_A=\mathbb E_A[\Phi_A g_A].
\]

令 \(c=c_0+Bu\)，\(B\) 为固定允许更新方向。优化

\[
J(u)=\frac12\mathbb E_A[(\Phi_A^\top(c_0+Bu)-g_A)^2]
+\frac\lambda2\|u\|^2.
\]

直接求导得到

\[
\nabla J(u)=Hu-d,
\]

\[
H=B^\top G_A B+\lambda I,\qquad
d=B^\top(b_A-G_Ac_0).
\]

固定步长 \(\eta\) 的梯度下降，从 \(u_0=0\) 出发：

\[
u_{t+1}=(I-\eta H)u_t+\eta d.
\]

归纳展开可得

\[
\boxed{
 u_n=\eta\sum_{t=0}^{n-1}(I-\eta H)^t d,
 \qquad c_n=c_0+Bu_n.
}
\tag{1}
\]

当 \(0<\eta\lambda_{\max}(H)<2\) 时，有界正特征方向稳定。零特征方向用有限和直接处理。

对 \(H=Q\operatorname{diag}(\lambda_i)Q^\top\)，式 (1) 等价于

\[
u_n=Q\operatorname{diag}(f_{n,\eta}(\lambda_i))Q^\top d,
\]

\[
f_{n,\eta}(\lambda)=
\begin{cases}
[1-(1-\eta\lambda)^n]/\lambda,&\lambda>0,\\
n\eta,&\lambda=0.
\end{cases}
\]

实现采用矩阵仿射递推的二进制合成，不做病态 Gram 的伪逆，也不随意裁掉小特征值。理论因此覆盖有限相位、奇异 Gram 和有限步数。

### 3.3 部署表必须独立进入误差

对 \(\Omega_D\) 和评价分布，定义

\[
G_D=\mathbb E_D[\Phi_D\Phi_D^\top],\qquad
b_D=\mathbb E_D[\Phi_Dg_D],\qquad
q_D=\mathbb E_D[g_D^2].
\]

代入 \(c_n\)，有精确风险

\[
\boxed{
R_D(n)=q_D-2b_D^\top c_n+c_n^\top G_Dc_n.
}
\tag{2}
\]

证明只需展开平方并取期望。计算时直接评价残差平方，避免三项接近相消时损失精度。

(1)–(2) 同时保留：目标是什么、训练暴露在哪里、训练表怎样分配、初值与槽位怎样耦合、允许更新哪些系数，以及推理时又怎样改了表。上一份报告只强调同表训练/评价，尚未把强推理变换充分写进核心公式；这一版补上了 \(\Omega_A\neq\Omega_D\) 的情况。

### 3.4 四种训练阶段如何落入同一式子

| 阶段 | 固定特征模型的对应 |
|---|---|
| 零训练替换 | \(n=0\)，保留既有 \(c_0\)，只改部署特征 |
| 受限适配 | 固定 \(B\)，由 (1) 求有限步更新 |
| 全系数适配 | \(B=I\) |
| 从零学习后再延展 | 用训练表从共同初值求 \(c_P\)，再以 \(\Omega_D=T_s(\Omega_P)\) 代入 (2) |

真正的 LoRA 的 \(B\) 不是只由 rank 决定的常数矩阵。LoRA 的两个因子随训练变化，功能切空间也会变化；实际 Adam 的步长与预条件器同样不同。所以这里统一的是明确的学习对象和受限模型，LM 层面的统一证据仍由同表同预算比较建立。

### 3.5 噪声、学习速度与终点表现

若递推还包含独立、零均值梯度噪声 \(\xi_t\)，协方差为 \(\Sigma\)，则令 \(F=I-\eta H\)：

\[
C_n=\eta^2\sum_{t=0}^{n-1}F^t\Sigma(F^t)^\top.
\]

因 \(c_n=c_0+Bu_n\)，期望部署风险增加

\[
\operatorname{tr}(B^\top G_D B\,C_n).
\tag{3}
\]

这给出表示误差、有限步偏差和噪声三种明确来源。它不把真实 SGD 噪声假装成固定 \(\Sigma\)，但可以检验“更大 rank 一定更好”“训练更久一定改善长任务”为什么没有一般保证。

即使最简单的一个模式，若学习轨迹是 \(c_n=c_\infty(1-r^n)\)，部署目标为 \(y_D\)，风险为 \((a c_n-y_D)^2\)。当 \(0<y_D/(ac_\infty)<1\) 时，风险会先下降后上升，转折点由

\[
r^{n_*}=1-y_D/(ac_\infty)
\]

确定。因此，早期 Cosh 优势和充分训练后 Geo 反超可以同时出现。这个存在性推导不能反向证明附件中的反转已经由此机制导致。

### 3.6 怎样形成具体的分配选择

在有限候选集合 \(\mathcal Z\) 上，提前固定共同任务、训练步数、学习率和 Native 平方误差上限，计算

\[
\boxed{
z^*=\arg\min_{z\in\mathcal Z}R_L(z,n)
\quad\text{s.t.}\quad R_N(z,n)\le\varepsilon_N.
}
\tag{4}
\]

每个候选通过 §2.1 生成实际频率，(1)–(2) 给出分数。有限集合枚举可以精确求出该受限问题的最优候选；若无候选可行，输出“此集合在此代理问题中无可行点”。

随附 `select_allocation.py` 实现了这一接口，必须提供共同目标和初始化定义。它不接受“仅输入 K、L，输出最佳 LM 表”这种缺失任务的调用。目标与数据来自开发探针时，选择成本应报告；不能继续称为零搜索默认规则。最终 benchmark 不参与选择。

这一构造目前是固定特征问题的完整算法。把它升级为论文方法需要证明它能在独立条件下选择更好的真实模型分配。本轮不预填某个 τ 或某个分配必胜的数字。

### 3.7 分配梯度：为何不能只看 rank

对 \(x_j=-\log\omega_j\)，

\[
\frac{\partial\cos(\omega_j\Delta)}{\partial x_j}
=\omega_j\Delta\sin(\omega_j\Delta),
\qquad
\frac{\partial\sin(\omega_j\Delta)}{\partial x_j}
=-\omega_j\Delta\cos(\omega_j\Delta).
\]

对固定 \(B,c_0\)，令 \(V_{t,j}=\partial u_t/\partial x_j\)，微分递推为

\[
V_{t+1,j}=F V_{t,j}-\eta(\partial_jH)u_t+\eta\partial_jd,
\qquad V_{0,j}=0.
\]

再微分部署风险，得到

\[
\partial_jR_D=
2\mathbb E_D\left[(\Phi_D^\top c_n-g_D)
\left((\partial_j\Phi_D)^\top c_n+\Phi_D^\top B V_{n,j}\right)\right].
\tag{5}
\]

这包含了“频率移动改变当前函数”和“频率移动改变学习过程”两部分。仅对部署 Gram 的 rank 求导，遗漏目标、初值与学习路径；只优化 Native Fisher，又没有给出长任务希望的方向。若初值也由待优化表训练得到，应连同前一训练阶段一起微分。

(5) 是可选理论优化接口，不授权对实际 LLM 继续大规模扫频。有限既有表的强基线比较先执行，理论探针与它并行。

---

## 4. 非几何分配理论应该保留什么、补足什么

### 4.1 几何分配的精确最优性范围

固定端点 \(x_0=a,x_{K-1}=a+R\)，相邻间隔 \(d_k=x_{k+1}-x_k\) 满足 \(\sum d_k=R\)，所以

\[
\max_k d_k\ge R/(K-1).
\]

等号当且仅当所有间隔相等。几何表最小化最大的对数尺度空隙。这是一个明确、简单的覆盖准则，不是对语言建模最优性的证明。

非几何收益必须来自不均匀任务需求、学习信号或成熟模型兼容性。论文应当证明具体哪个条件使“均匀覆盖”不再是最好选择。

### 4.2 为什么现有主实验不能由慢频渐近式直接解释

主因果实验 \(K=32,b=L=256\)，最慢频率满足

\[
L\omega_{\min}=256^{1/32}=1.189207115\ldots.
\]

它没有处在 \(\omega L\ll1\) 的慢频极限。原稿第 2 页把慢频塌缩和该实验的收益放在同一张总览图，不足以建立两者之间的解释连接。第 25 页 Figure 6 的支撑重定向反转，也要求把部署变换纳入理论。[P1]

主理论使用 §3 的有限相位计算，慢频极限作为特殊情形。

### 4.3 慢频条件数与学习代价

设 \(m\) 个互异正数 \(r_j\) 固定，\(\omega_jL=\epsilon r_j\)。对 \(t=\Delta/L\) 的 sin/cos 特征逐列去均值，假设中心化多项式 \(t,t^2,\ldots,t^{2m}\) 的 Gram 正定。

则物理坐标 Gram 的特征值按降序有

\[
\lambda_j(G_\epsilon)=\Theta(\epsilon^{2j}),\quad j=1,\ldots,2m,
\]

\[
\lambda_{\min}=\Theta(\epsilon^{4m}),\qquad
\kappa(G_\epsilon)=\Theta(\epsilon^{-(4m-2)}).
\tag{6}
\]

证明：中心化 sin 的幂级数由奇数次幂组成，cos 由正偶数次幂组成。利用 \(r_j^2\) 的 Vandermonde 矩阵做与 \(\epsilon\) 无关的可逆列变换，逐次消去低阶项，可得到 \(\epsilon^j[p_j(t)+O(\epsilon^2)]\)。中心化多项式 Gram 的正定性使变换后矩阵与 \(\operatorname{diag}(\epsilon^2,\ldots,\epsilon^{4m})\) 以常数因子可比。固定可逆变换保持这些幂次阶。

物理弱方向在固定学习率下学习慢；block whitening 会改变这一优化度量。Adam、QK norm 或可训练预条件改变此结论的实际时间含义。Fourier/Vandermonde 聚簇病态性属于已有数学领域，Batenkov 等已研究聚簇数目与最小奇异值的幂次界。[R10] 不把 (6) 单独包装为首次发现。

### 4.4 Cosh 的完整条件推导

给定现稿 surrogate

\[
J[\rho]=\frac\alpha2\int\rho^2+
\frac\beta2\iint\rho(\phi)\rho(\psi)\min(\phi,\psi),
\]

在非负、单位积分的 \(L^2[0,1]\) 函数上，第一项在 \(\alpha>0\) 时严格凸；第二项半正定，因为

\[
\iint f(\phi)f(\psi)\min(\phi,\psi)
=\int_0^1\left(\int_s^1 f(u)du\right)^2ds\ge0.
\]

令 \(g(\phi)=\int\rho(\psi)\min(\phi,\psi)d\psi\)，则 \(g''=-\rho\)、\(g(0)=0\)、\(g'(1)=0\)。带质量约束的驻点满足

\[
\alpha\rho+\beta g+\nu=0.
\]

微分两次得到

\[
\rho''-\tau^2\rho=0,\qquad \tau^2=\beta/\alpha.
\]

边界和质量条件给出 \(\rho'(0)=-\tau^2,\rho'(1)=0\)，故

\[
\rho_\tau(\phi)=\frac{\tau\cosh(\tau(1-\phi))}{\sinh\tau}.
\]

其处处为正、积分为 1，严格凸性保证唯一。积分并反解即为 §2.1 的 quantile 构造。

数学上这一条已经闭合。尚未闭合的是 surrogate 系数如何对应真实目标与学习过程。\(\tau=d_{head}/\sqrt L\) 的常数不是普适结论。另一种高分辨量化模型导出的 \(\rho\propto w^{1/3}\) 同样需要外部给定 \(w\)；不能设 \(w=\rho_{Cosh}^3\) 再宣称推导出 Cosh。

论文可以保留 Cosh 作为低成本解析实例，同时由 §3 的风险和真实实验说明适用条件。单一曲线不必承担所有阶段的最佳分配。

---

## 5. 原生保持的具体解决方案：从平均 KL 到真实决策门槛

### 5.1 为什么平均 KL 很小仍然会翻转答案

取一个 Native 前缀，教师分布为 \(p\)，最大概率 token 为 \(i\)，次大为 \(j\)。记 \(a=p_i>b=p_j\)。学生分布为 \(q\)，使用相同确定性解码变换、同一有效 token 集合。

有如下精确界：

\[
\boxed{
\inf_{q:\operatorname{argmax}q\ne i}\operatorname{KL}(p\|q)
=\kappa(p)
=a\log\frac{2a}{a+b}+b\log\frac{2b}{a+b}.
}
\tag{7}
\]

严格的错误 argmax 区域是开集，边界值以 infimum 取得；允许与 \(i\) 并列的闭集则在边界达到最小值。

**证明。** 固定竞争 token \(j\)，约束 \(q_j\ge q_i\) 是凸的。无约束极小点 \(q=p\) 不可行。KKT 条件迫使最优点在 \(q_i=q_j\) 的边界，其余坐标保持 \(q_l=p_l\)，两者都等于 \((a+b)/2\)。代回 KL 得到右式。对固定 \(a\)，该式对 \(b\) 的导数为 \(\log(2b/(a+b))<0\)，因此最大的竞争概率给出最小门槛。最后对全部竞争者取并集，即得 (7)。

令 \(m=a-b,s=a+b\)，小间隔时

\[
\kappa(p)=\frac{m^2}{2s}+O(m^4/s^3).
\]

本轮 CPU 例子 \(p=(0.5001,0.4999)\)、\(q=(0.4999,0.5001)\)：KL 约 \(8\times10^{-8}\)，却已经反转 argmax。这直接说明固定一个平均 KL 门槛无法保护所有原生决策。

### 5.2 序列级充分条件

在教师真实 greedy 轨迹每个前缀 \(t\) 上，若

\[
D_t=\operatorname{KL}(p_t\|q_t)<\kappa(p_t),
\]

则学生完整轨迹与教师一致，包括真实终止 token。

证明用归纳：第一步 argmax 一致；若此前输出一致，第 \(t\) 步输入前缀相同，(7) 再保证该步一致。必须保持模板、logit processor、position IDs、表和缓存计算一致。

对一条教师轨迹定义 \(r_t=D_t/\kappa_t\)，若教师每步均为唯一 argmax，则还有确定性上界

\[
\mathbf1\{\text{学生轨迹改变}\}
\le \min\{1,\max_t r_t\}.
\tag{8}
\]

因为真实首个分歧必然发生在某个教师前缀，并使该位置 \(r_t\ge1\)。对 prompts 求均值，可得到这一测试集合上的轨迹变化率上界。

只测采样前缀，就只能给这些前缀的结论。用人工 gold 前缀代替教师实际轨迹，也不能直接得到保持教师完整生成的证书。模型存在多个正确答案时，保持教师轨迹只是充分条件，要求比任务正确更强。

### 5.3 由推导得到一个明确的修正

当前 Native KL 的统一预算，与每个真实决策的脆弱程度不匹配。先利用已经存在的教师缓存，计算 \(D_t,\kappa_t\) 和实际翻转；补测仅限缺失的关键前缀，不自动重新启动已暂停的大规模教师面板。

当主要训练已经改善 long、Native 却失败时，预先固定 \(\gamma=1/2\)，对开发 replay 的受保护前缀使用

\[
L_{guard}=\frac1N\sum_i
\max_{t\in\mathcal T_i}
\frac{[D_{it}-\gamma\kappa_{it}]_+}
{\max(\kappa_{it},\kappa_{floor})}.
\tag{9}
\]

分母下限只用于数值稳定，不提高允许的 KL 阈值。若每一项为零且 \(\kappa>0\)，所有被保护前缀满足正确决策的充分条件。\(\gamma=1/2\) 是明确的安全余量选择，没有声称它是最佳训练超参数。

执行原则：保持原任务 loss、模型、表、rank/全参数范围、token 预算不变；对本方与最强对手同时使用相同 guard；复用已冻结的 Native 正则权重及调度，报告新目标的尺度。\(\kappa_{floor}\) 可由教师概率计算的数值精度预算确定并先冻结；最终判据仍使用真实 \(\kappa\)，不能用 floor 代替证书。

这是一个新的、具体的 Native 约束候选。它不增加长证据监督，也不证明联合可行点必定存在。若它保住 Native 却消灭 long 增益，说明当前候选/更新路径仍有实际冲突，不再以平均 NLL 过门宣称成功。一般 KL 决策几何本身不作为 RoPE 特有的新定理申报。

---

## 6. LoRA 与全参数：可证明的限制及实际判定

### 6.1 局部剩余误差

在一个固定网络点附近，频率扰动造成的白化输出误差为 \(A\delta z\)，允许适配方向的白化 Jacobian 为 \(B\)。最小二乘的精确投影关系是

\[
\min_u\|A\delta z+Bu\|^2
=\delta z^\top A^\top(I-BB^\dagger)A\delta z.
\tag{10}
\]

它说明“换表损伤是否容易被当前适配方向抵消”取决于误差方向与更新空间的关系，不能由频率 MAE 或名义 LoRA rank 判断。该公式已在历史方案中出现，不能算本轮新发现。大幅换表和长训练会超出局部近似。

### 6.2 一个有限变化的单层补充结果

为明确全局频率差别的作用，考虑固定输入、线性 Q/K、固定 normalization 的单头双线性算子：

\[
M_0=W_Q^\top R_0(\Delta)W_K.
\]

令 Q/K 更新 \(U_Q,U_K\) 的 rank 分别不超过 \(r_Q,r_K\)，目标变成

\[
M_1=(W_Q+U_Q)^\top R_1(\Delta)(W_K+U_K).
\]

定义原始换表缺陷

\[
E=W_Q^\top(R_0-R_1)W_K.
\]

更新产生的修正可写成

\[
C=U_Q^\top R_1(W_K+U_K)+W_Q^\top R_1U_K,
\]

因此 \(\operatorname{rank}C\le r_Q+r_K\)。Eckart–Young 定理给出

\[
\boxed{
\|M_1-M_0\|_F^2\ge
\sum_{j>r_Q+r_K}\sigma_j(E)^2.
}
\tag{11}
\]

它覆盖有限相位变化。若 \(W_Q,W_K\) 的行正交，\(E\) 的非零奇异值由各旋转差决定：

\[
2\left|\sin\frac{(\omega'_k-\omega_k)\Delta}{2}\right|,
\]

每对重复两次。一般情况下，可由对 \(W_Q^\top,W_K^\top\) 的 thin QR 将奇异值计算降到 head dimension 量级。

此式只给该单层问题的必要下界。对多个 \(\Delta\) 分别求下界仍没有证明存在一个共享适配器同时达到它们。深层 all-linear LoRA、V/O 与残差改动、随输入变化的 QK norm 都会改变问题；OLMo2 的实际 normalization 必须检查。不得拿 (11) 宣称 OLMo r16 不可能、或全参数必能解决。[R8]

### 6.3 主实验采用全参数，LoRA 如何公平桥接

最新附件已经有约 10M-token LoRA 训练远端未充分拟合的记录。全参数匹配对照的价值在于直接检验这项限制；模型档位仍保持 OLMo 1.485B，避免同时换模型和任务。[P2]

全参数主比较使用相同表、数据与预算；随后对最强对手和本方表完成相同 token 里程碑的 LoRA。LoRA 默认复用已登记的 all-linear r16 配置与缩放约定；不得悄悄改成 Q/K-only、改变 alpha 或使用不同父适配器。

| 完整结果 | 可以形成的判断 |
|---|---|
| 同表全参数学会，LoRA 没学会 | 此 LoRA 协议存在优化/更新空间限制；不是所有 rank 的普遍不可能 |
| 两者都学会，本方更快且 Native 相当 | 可以写分配降低适配需求 |
| 两者都未学会 | 此预算/数据/主体尚未实现目标，不能断言只缺更多 tokens |
| 对手同预算学会，本方没学会 | 对本方当前分配的直接负证据 |
| long 学会但 Native 严重下降 | 检验 §5 的约束修正，不能用概率代理覆盖生成损失 |

完整对比先完成，不能因为某臂早期看起来弱就停掉它而保留另一臂的完整预算。失败或不稳定臂的退出条件只用已登记的数值故障规则。

---

## 7. 长输入训练：给出需要的监督，避免再次误判

### 7.1 总长度和证据距离的数学区别

对单个证据分数 \(s_*\) 与干扰分数 \(s_j\)，有

\[
p_*=\sigma\left(s_*-\log\sum_j e^{s_j}\right).
\]

若干扰分数近似共同值 \(\mu\)，要保持 \(p_*\ge q\)，需要

\[
s_*-\mu\ge\log N+\log\frac{q}{1-q}.
\tag{12}
\]

式 (12) 解释了证据距离不变、干扰项增加仍会使任务变难。实际 LM 中干扰分数不是常数，内容和深层表示也改变；不能把所有长上下文失败都归因于精确的 \(\log s\) 缺口。

1.8%→22.1% 的证据质量改善，说明换表有效改变注意力。它不足以证明选词所需信息已经被完整读取、传播和使用。背景 token 数、内容竞争、局部绑定和终止都需要真实任务检验。

### 7.2 本轮主训练长度选择

主适配优先使用真实 8K 物理序列、连续位置、固定 factor-4 部署表，测试 4K/8K/16K。理由：已有 Z 在 8K 有明确生成信号；相对 16K 训练成本更低；16K 测试超过本轮相位暴露。它是一个待检验的训练选择，没有宣称 8K 必然足够。

如果这一配方在 8K 已学会、16K 未迁移，再用预留预算做“同表继续 16K”的两臂比较。届时 16K 改善应写为原生 4K 模型的上下文延展；未见长度目标改为 32K。不能把见过的 16K 继续称为适配后的未见长度。

若真实吞吐或已整理训练数据使 16K 主配方更划算，也可以在首次训练前固定 16K，但所有比较臂一起更改，并据此重新定义测试目标。

### 7.3 数据和目标不再多线扩展

使用已经准备好的自然文本、远端依赖任务和 Native replay，但必须在 tokenization 后证明：答案对应真实远端证据、局部提示没有答案泄漏、干扰项具有适当相似性、完整答案及真实 EOS 都在监督中。

保留既有任务损失，分开按自然文本 token、每条答案、Native 预测位置归一化。训练比例在 launch manifest 固定。若现有比例尚无明确 owner，Codex 先提交缺失字段；不得从本报告猜一个“20% 或 50%”填入科学配方。

任务监督已经存在，新增工作首先是可靠的对手、表和参数体制对比。完整答案 CE、margin、KL 不再被当作本轮新发明。

### 7.4 少量机制对照的准确范围

在同一任务生成过程下，保留三个暴露条件：physical-4K/ordinary positions；相同 4K 内容/sparse positions 到 8K；真实 physical-8K/continuous positions。

前两项固定可见 token 集合，可以较清楚测相位改变。第三项额外引入内容和竞争，不能只靠匹配最大 position ID 就声称纯识别竞争负荷。匹配监督目标数，另外报告 input tokens 与时间，不强行让两者同时相等。

这组对照在主比较完成后才占新训练预算。新窗口上的完整能力结果优先于又一轮大规模 attention heatmap。

---

## 8. 理论预测如何进入执行，而不成为拖延实验的新门槛

### 8.1 可信比较直接运行

MrRoPE/YaRN 资格核查、已有 checkpoint 叠加矩阵和主要配对训练，本身就在回答论文是否有新增价值，可以直接做。理论上的预测要求只限制额外新曲线与昂贵新分支，不要求先证明实际赢家才能运行关键对照。

### 8.2 最值得预注册的三个预测对象

**对象一：训练预算造成的排序变化。** 在固定表、相同 optimizer 下，比较训练中间里程碑和终点的 Native/long 差距。理论使用训练特征而非目标 rank。已有“充分训练 Geo 反超”是发现线索；独立支持范围/新的预定预算负责确认。

**对象二：推理变换能否抵消训练分配的收益。** 对同一组检查点比较 identity、共同参考 MrPro、按实际频率推广的 YaRN，以及已有 FMRoPE retarget。预测必须由未查看的条件确定；不能观察四张表后重写原因。

**对象三：Native 决策风险。** 在未参与方法选择的 Native 前缀上，检验 \(D_t/\kappa_t\) 是否比平均 KL 更能识别实际 argmax 翻转。(7) 是单向充分界，超过阈值不要求一定翻转；因此检验应报告证书覆盖率、被覆盖前缀的实际一致性、未覆盖部分的风险分布。

### 8.3 理论预测的资格

固定特征模型的目标应来自同一个预先说明的控制任务，或一套不含最终测试结果的开发探针。多尺度窗口、局部关系或证据判别可以作为明确任务，不能为每个模型临时挑使 Cosh 获胜的目标。

先在控制问题核验 (1)–(5) 的数值及排序；真正有论文价值的是它对完整 Transformer 的独立预测。在这里尚未验证前，正文只能称“可检验模型”，不能称“已解释全部实验”。

只由 \(K,L\) 或 Gram rank 推断最优表的替代方案不进入执行。现在缺少的实际任务权重，不应再次被任意 utility 伪装成已知量。

---

## 9. ROI 排序与 100 GPU 小时安排

以下是额度，不是已测运行时间。所有训练预算由实际吞吐反推，优先完成一组完整匹配对比。CPU 数学检查独立执行，不占这些 GPU 额度。

| 阶段 | 核心工作 | GPU 小时上限 |
|---|---|---:|
| E0 | 实际表、gain、评价与训练峰值的新边界核查 | 6 |
| E1 | 151M 已有检查点强变换矩阵；OLMo 冻结强对照 | 12 |
| E2 | OLMo 全参数主比较与主要配对重复 | 38 |
| E3 | 同表同 token 里程碑的 LoRA 桥接 | 12 |
| E4 | 一个补强的 scratch/独立支持范围比较 | 14 |
| E5 | 原 MrRoPE 模型上的独立确认 | 12 |
| 预留 | 序列边界、必要延长或最终复核 | 6 |
| 合计 |  | 100 |

### E0：只核查新改动和有疑问的结果身份

输出 `table_manifest.json`、`baseline_equivalence.json`、`eval_manifest.json` 和 `throughput.json`。

优先核查：Y2 到可靠 YaRN 的两项变化；MrRoPE 累乘边界；Z 的实际公式与 gain；配置中的 rotary pair 数；prefill/decode 的同表行为；模型 template；KL reduction。过去已经通过、且本轮未触及的缓存/merge/reload 检查复用原回执。

HF v4.44.2 的数学约定用于参考，不要求用旧版本运行 OLMo2。实际 runtime 必须支持目标模型并保存 commit/version。

对 Z 的 uniqueness 投影检查计算精度与伪逆容差敏感性。已有结果使用存档频率 tensor；重新生成的数组有不同哈希时视为新变体。旧稿约 60% RULER 和最新严格成功率的对账，应先在同一批已保存输出上运行两套 scorer，并逐项核对任务、模板、表和 gain。

### E1：第一项决定论文主张的结果

已有 151M：训练表 `{Geo,Cosh}` × 推理 `{identity,YaRN,MrPro}` × `3 seeds`；长度 1×/2×/4×/8×。保留固定 s4 部署表跨长度和按目标 s 的策略区别，不把二者混成一条曲线。已有 FMRoPE retarget 原结果完整并列。

先复现 paper-owned 32 个 anchors，再增加独立文档评测。小模型上下文短，优先把样本和完整比较做扎实。普通 NLL 的评价可用 base completion 模式；不能对未指令训练的小模型强制用指令任务失败充当 RoPE 否证。

OLMo 冻结：Native、可靠 YaRN、MrUni、MrPro、既有 Z。各方法完整原配方作为主要方法比较；另设共同 gain 的频率因果对照。只在需要归因的主对比上做 amplitude 消融。

### E2：成熟模型完整配对

主模型保持 OLMo-2-0425-1B-Instruct。第一种子至少三臂：可靠 YaRN、MrPro、当前 Z。原始 Native 保留冻结参考，不优先购买一个明显超窗失效的 500M Native 训练臂。Z 的精确定义从实际 tensor manifest 读取；旧 paper Cosh 和 Z 不互相替换。

总额度中先规划三臂完整第一种子，再预留对最强可靠对手与 Z 的第二个训练种子。每个首种子运行使用相同实际 tokens、数据顺序、LR 选择预算与评测次数。预算里程碑采用固定总预算的 1/4、1/2、1，而不是先承诺三臂各 500M。

设三臂有效吞吐 \(v_i\)，训练时间额度 \(H\)，相同每臂 token 预算的上界是

\[
B\le\frac{3600H-T_{eval}-T_{save}}{\sum_i1/v_i}.
\]

完整 step 的计时必须包含任务、Native replay、checkpointing 和优化器；不能只测无 teacher/replay 的 forward。

在看结果前确定配对预算和重复预算。主要终点不通过 validation 从四个里程碑中事后挑最好的一臂；如采用 checkpoint selection，两臂共享同一提前固定规则，最终确认在独立集合进行。

### E3：LoRA 桥接

只做最强对手与 Z 的匹配 pair，采用已经有 owner 的 all-linear r16 配方。选择 E2 中双方都保存的共同 token 里程碑，同时报告 GPU 时间。它回答受限适配是否保留分配收益；不能拿 LoRA 一个小预算与全参数大预算比较后宣称表达上限。

若 LoRA 也学会并更省时，剩余外部确认优先 LoRA。若全参数明显更可行，外部确认优先证明固定方法的收益，不追求所有体制都成功。

### E4：从零训练补强

新增训练优先一个能较充分训练的小模型，保持 trainer、support 和 token order 配对。条件由 E1 与理论预测共同确定，例如一个新的训练长度或支撑。候选先保留 Geo、Cosh 与最必要的对照，不继续做几十个弱预算分配。

实践基线须包含合理 geometric base。固定 support 的因果控制和按开发集选择的 geometric base 是两项不同比较，两者均保留。

如果 LeRoPE 是主张所需的直接对手，就做其真实频率学习协议和相同调参额度；旧的低预算 32-scalar 学习表不直接称为充分复现。无法在额度内完成时，应在限制和相关工作写明，不能据旧弱对照声称全面优于 LeRoPE。

E1 若显示收益完全被强扩展消除，E4 先用于验证独立条件下是否存在可预测收益，而不是默认原主张仍成立。

### E5：独立确认

优先确切的 Llama-3-8B-Instruct，与 MrRoPE 模型身份一致；Llama-3.1 不作同名替换。若已有下载、许可或后端不可用，记录原因并使用另一个确切的原文模型。先确认 frozen 主对比；只有主结论依赖适配时，才购买匹配适配确认。

外部模型至少使用新任务实例，并使其结果不再进入分配、gain 或 loss 的选择。Qwen 旧 64K core-four 为 development 证据，不能重新命名成确认集。

---

## 10. 评价、统计与 Native 的实际判据

### 10.1 固定部署规则

主结论使用一张表、一个固定 gain，prefill 到 decode 不变；Native 评价也运行这张最终表。有 routing 的完整系统可以作为另一个策略，但所有基线必须有同等 routing 权限，且不能用于证明静态表原生保持。

成熟 Instruct 模型的主评价采用各自官方 chat template，并在所有方法间保持一致；旧 raw 结果作为模式审计保留。未指令训练的 scratch 模型使用相应 completion/NLL 协议。

每行保存 `L_native,L_physical,L_phase,L_test,evidence_distance,distractor_count`。训练、实际预训练历史和模型卡配置长度不能互相代替。

### 10.2 评价集合

主开发阶段用覆盖单证据、多个竞争项、组合关系的少量固定任务；最终确认覆盖完整 RULER 任务家族及至少两类真实文档 QA，并单列自然文本 NLL。预算不足时优先主要对比的样本量，缩减无关模型数。

训练已包含某个 RULER generator family 时，该 family 的新实例结果称为任务家族内长度迁移。最终真实文档 QA 使用未参与任务构造和选择的文档，并记录来源重叠核查。

保留官方任务分数；同时记录完整答案正确、正常终止、格式和联合严格成功。对本身允许多个合法回答的任务，不把唯一字符串 exact match 作为唯一语义标准。

Native 必须包括自然文本、模型原本确实能完成的短 QA/指令，以及终止敏感生成。每类分别报告，避免宏平均掩盖某类回归。原实验 0.88 的保持阈值保留历史身份；新确认的非劣边界在启动前固定，不悄悄改成 0.90 或放宽。

PPL retention 为

\[
\frac{PPL_N}{PPL_C}=e^{NLL_N-NLL_C}.
\]

例如 0.90 对应 \(\Delta NLL\le0.1053605\)，只是单位换算，不构成统一科学门槛。

### 10.3 组级检验

当前 1/32 与 4/32 组成功，相同 prompts 配对时，共同成功数只能为 0 或 1。两种 discordant 计数分别为 \((1,4)\)、\((0,3)\)。双侧精确 McNemar p 为 0.375、0.25。因此原行级 Fisher p=0.0076 不能支持组级可靠性提升的显著结论。

文档、prompt 或双答案世界组作为重采样单位；训练种子单独报告。三个训练种子全部同向的双侧符号检验 p 仍为 0.25，不得用数千 token bootstrap 替代训练重复来写训练显著性。

对配对二元差异 \(D_i\in\{-1,0,1\}\)，设 discordance rate 为 \(q\)，均值为 \(\delta\)，则

\[
\operatorname{Var}(\bar D)=(q-\delta^2)/n.
\]

若希望检测 5 个百分点、\(q\approx0.2\)，普通正态近似的 80% power 规划约需

\[
n\approx (1.96+0.84)^2\,0.2/0.05^2\approx627
\]

个独立配对单位。具体任务应使用自己的方差/分层结构。每格 20–32 例适合预检与大效应，不适合保证 3–5 点小增益可靠。

### 10.4 搜索成本与选择偏差

候选表、gain、LR、数据比例和 checkpoint 选择都计入成本。最终报告主要对比的确认差值及 CI，不能报告探索中最好的一条均值作为无偏效果。理论选择若使用校准任务，就明确标为校准；“模型权重未更新”和“完全没有任务选择成本”分开记账。

---

## 11. Codex 的执行接口与输出

### 11.1 先生成资产与协议清单

本报告不会替 Codex 猜缺失的仓库路径。先在现有研究分支中定位 P2 的原始回执、实际表、151M 检查点、已登记 LoRA/FT 配方与 evaluator；保留路径和 SHA256。附件中引用但找不到的资产写为缺失，不从摘要重造训练历史。

最低 run manifest：

```yaml
run_id: unique_id
model_id: exact_id
model_revision: exact_revision
weight_sha256: recorded
runtime_commit: recorded
train_table_sha256: recorded
deploy_table_sha256: recorded
frequency_parameterization: exact_formula_or_stored_array
rotary_pair_layout: exact_layout
gain_location: recorded
q_gain: numeric
k_gain: numeric
logit_multiplier: numeric
training_regime: frozen_or_lora_or_full
trainable_modules: explicit
lora_rank: null_or_numeric
lora_alpha: null_or_numeric
native_length: numeric
physical_train_length: numeric
max_relative_train_position: numeric
input_tokens_budget: numeric
unique_tokens: numeric
answer_supervision_tokens: numeric
native_prediction_positions: numeric
loss_normalization: explicit
optimizer_and_lr: frozen_config
teacher_cache_sha256: recorded_or_null
data_split_hashes: explicit
evaluation_template_and_decoder: frozen_config
routing: disabled_or_explicit_policy
eval_selection_role: development_or_confirmation
```

实际表为优先事实。`Z`、`Y`、`Cosh` 等名称只作为索引。

### 11.2 CPU 入口

```bash
cd rope_codex_revision
OPENBLAS_NUM_THREADS=1 python verify_theory.py
```

产物 `verification_results.json` 已随包提供。本轮核验包括：Cosh 端点、MrPro 的边因子和复合、YaRN ramp 方向、有限步和显式迭代一致、奇异 Gram、噪声协方差、KL 决策边界、有限秩修复、组级精确统计。

固定特征候选选择：

```bash
python select_allocation.py --spec problem.json --data data.npz --out result.json
```

`problem.json` 必须说明共同任务、初始化、预算、候选训练/部署数组和 Native 代理风险上限。NPZ 字段与接口见脚本。该程序输出有限候选的实际 `inv_freq` 和风险；不会输出伪造的 LM 分数。

### 11.3 服务器侧按次序产出

1. `assets_and_missing.json`、`protocol_delta.md`：只列本轮改变的实现与科学协议。
2. `baseline_equivalence.json`：可信 YaRN/MrPro、Z 实际 tensor、gain、template 与已有 scorer 对账。
3. `existing_checkpoint_overlay.csv`：E1 所有种子、训练表、推理策略、长度和逐文档数据。
4. `paired_training_curves.csv`：E2/E3 的相同 token 里程碑与真实 GPU 时间。
5. `native_decision_budget.jsonl`：已有可用教师前缀的 \(p_1,p_2,\kappa,D\)、实际 argmax、样本身份；未覆盖位置明确为空。
6. `confirmation_results.json`：冻结后独立确认与聚类统计。
7. `claim_to_evidence.md`：每项主张指向具体表、run、统计单位及限制。

每个阶段完成后保留失败臂。未经授权不做服务器文件删除、模型仓库上传或长期任务重启。

### 11.4 关键工程细节

Native KL 应先沿 vocabulary 求和，再按真正受监督预测位置和声明的组权重归一化。对 `[B,L,V]` 直接 `batchmean` 可能只除 B，使有效约束随 L 改变。

教师缓存必须明确存储精度；接近 (7) 边界时需要以更高精度复核或直接检查实际 margin。不能把 BF16 teacher logits 带来的不确定性当成理论反例。

答案 CE/Native KL 可只对相应预测位置的 hidden states 应用 LM head，从而避免整个 `[L,V]` 张量。位置必须是预测下一 token 的 hidden state，和标准短序列实现做 parity。

训练峰值包含优化器与反向；长 attention backend 禁止静默回退成不可承受的路径。数据盘和恢复检查点预算按真实全参数状态计算，不能继续采用 LoRA 小检查点估算。

---

## 12. 主文重构及投稿判断

### 12.1 建议主文结构

问题和主要结果：直接展示相对 MrRoPE-Pro 的固定表 Native—long 表现与适配成本。

定义与最近工作：承认 radix/exponent 参数化关系，以及 LeRoPE 已有的训练频率和 YaRN 组合。明确本文研究的预算、支撑和共享表范围。

理论与构造：保留完整 pair 几何基础，将 §3 的训练/部署风险作为核心解释；Cosh 给出一个明确解析实例。一般 KL 决策界可放 Native 分析或附录；有限秩 Q/K 下界放附录，避免方法主文偏离非几何分配。

主要实验：既有训练表叠加强变换；OLMo 主适配曲线；LoRA 桥接；独立模型与任务确认。

分析与限制：真实排序变化、Native 代价、理论不适用区间、未通过结果。跨模态和弱预算 factorial 根据主线需要保留在附录，不继续扩展。

### 12.2 三种可能形成的论文

**A. 固定表方法收益成立。** Z 或后续明确构造在不路由下，比强对手更好保留 Native，并改善真实长生成。主文突出方法与解析解释。

**B. 主要收益来自学习阶段。** 冻结表优势有限，但非几何训练/适配在强扩展后有稳定收益、达到同质量所需成本更低，且理论能预测至少一个边界。主文突出预算条件下的学习规律。

**C. 强基线补齐后没有 matched benefit。** 保留已经成立的因果事实与边界分析，但现有实验尚不足以按 A/B 的贡献强度投稿。继续投入应针对一项明确新增预测与构造，不以堆规模、加更多 surrogate 或扩大措辞替代。

A 与 B 都是有价值的目标。当前材料尚未确定哪一个已经成立；本轮 E1/E2 正是成本最低的判别办法。

### 12.3 截止前安排

9/6–9/8 完成 E0 与主要 E1；9/9–9/14 完成成熟模型配对曲线，并行处理已有 scratch 强变换；9/15–9/17 冻结摘要所依赖的主要主张；9/18–9/21 做独立确认和关键重复；9/22–9/24 完成图表、引用和复现核对。

ICLR 官方摘要截止为 9/18 23:59 AoE，全文为 9/25 23:59 AoE。[R9] 9/17 作为内部摘要冻结日。

第一笔新增训练预算应购买一组可信的匹配对比。先取得它，再决定 Native guard、更长训练或新分配构造中哪一项值得继续。

---

## 附录 A：YaRN 的 radix 单调性完整核查

在几何表上令相邻频率比例 \(q\in(0,1)\)，旋转圈数 \(r_j=r_0q^j\)。原式 rotation-count 线性 ramp 的严格中间段满足

\[
f_j=\frac{\omega'_j}{\omega_j}
=\frac{\beta-s\alpha+(s-1)r_j}{s(\beta-\alpha)}.
\]

记 \(c=\beta-s\alpha\)、\(v=(s-1)r_j>0\)，相邻 radix 为

\[
\lambda_j=\frac{c+v}{c+qv},\qquad
\lambda_{j+1}=\frac{c+qv}{c+q^2v}.
\]

相减后分子为 \(cv(1-q)^2\)，严格中段分母为正。因此 \(c>0\) 递减、\(c=0\) 恒定、\(c<0\) 递增。

HF 的维度线性 ramp 在严格中间三点上为 \(f_j=A-Bj,B>0\)，有

\[
\lambda_{j+1}-\lambda_j
=\frac{B^2}{f_{j+1}f_{j+2}}>0.
\]

这一核查限定 MrRoPE 将 YaRN 一概称为 regressive 的解释范围。它不否定其已报告性能，也不能独自证明本方方法更好。实现版本和 ramp 坐标必须随结果记录。[R1,R2,R7]

## 附录 B：本轮 CPU 回执

本轮 `verify_theory.py` 实际得到：

- MrPro 边指数最大误差约 `5.55e-17`；端点与 s=1 一致。
- 原式 YaRN 在 s=4/32/64 的严格中段 radix 趋势为 `[-1,0,+1]`；HF 维度 ramp 严格中段递增。
- s4 的正确 Q/K amplitude 平方为 `1.2964769927807063`。
- 有限步均值与逐步递推最大误差约 `3.22e-15`；部署风险误差约 `5.00e-16`；奇异 Gram 误差约 `1.80e-16`。
- KL 边界闭式与约束凸优化一致；随机错误 argmax 未违反必要 KL 门槛。
- 单层有限秩修复的下界检查通过；小矩阵与原矩阵奇异值尾和差约 `4.55e-13`。
- 组级 1/32 对 4/32 的可行 McNemar p 为 `[0.25,0.375]`。

这些数值核验算法与推导一致，不构成语言模型或新方法收益。

## 来源与材料定位

### 项目资料

[P1] `main(20260906-144853).pdf`，31 页《ROPE HAS A SPECTRAL BUDGET》。重点：第 3–5 页理论与共适应；第 6–7 页冻结/LoRA；第 24–27 页 fixed support 与弱预算 factorial；第 27–31 页实际 phase exposure、routing 与部署表。另一份 Pro 使用 15:12 文件名，两个版本未经字节比对，不假设完全相同。

[P2] `THEORY_EXPERIMENT_SYNTHESIS(2).md`，2026-09-06 重组版。重点：§1.2–1.5 新的真实生成、Native、数据长度与 E1；§2 全参数 500M 建议；§3 未决问题。

[P3] `大修与实验执行清单.md`；以及 `RoPE_ICLR2027_Major_Revision_20260906.md`。两份是研究建议，不能作为新增实验的独立来源。

[P4] File Library 中 `5551_MrRoPE_Mixed_radix_Rotary.pdf` 及用户提供的完整审稿 `粘贴的 markdown (1)。md`，2026-09-06 上传版本。审稿中的作者回复含 8K 微调与 sliding-window PPL 说明。

[P5] 历史 `hybrid_rope_iclr2027_theory_experiment_dossier_20260904.md`、`SINGLE_TABLE_FFN_REPORT_AUDIT_AND_CODEX_GUIDANCE_20260904.md`、`RoPE_Exponent_Allocation_Unified_Plan_20260905.md`。用于确认已经尝试的损失、LoRA 与 Native 协议，不将其中假设当新实验事实。

### 外部原始来源（本轮检索）

[R1] Tian et al. *MrRoPE: Mixed-radix Rotary Position Embedding*. ICLR 2026. `https://arxiv.org/html/2601.22181v1`

[R2] Peng et al. *YaRN: Efficient Context Window Extension of Large Language Models*. ICLR 2024. `https://arxiv.org/html/2309.00071v3`

[R3] Karypis et al. *LeRoPE: Learnable RoPE Frequencies Improve Language Modeling*. §6–7 与 App. D. `https://arxiv.org/html/2607.10134v1`

[R4] Wang et al. *AdaRoPE: Not All Attention Heads Should Rotate and Scale Equally*. `https://arxiv.org/html/2607.19363v1`

[R5] Wu et al. *How Data Shapes RoPE Frequency Usage: From Positional Scale Matching to Length Generalization*. `https://arxiv.org/html/2607.07678v1`

[R6] *LongReD: Mitigating Short-Text Degradation of Long-Context Large Language Models via Restoration Distillation*. ACL 2025. `https://aclanthology.org/2025.acl-long.524/`

[R7] Hugging Face Transformers v4.44.2, `modeling_rope_utils.py`. `https://raw.githubusercontent.com/huggingface/transformers/v4.44.2/src/transformers/modeling_rope_utils.py`

[R8] Hugging Face OLMo2 配置与实现。配置：`https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct/raw/main/config.json`；实现：`https://github.com/huggingface/transformers/blob/main/src/transformers/models/olmo2/modeling_olmo2.py`。运行前保存实际版本。

[R9] ICLR 2027 CFP / Dates. `https://iclr.cc/Conferences/2027/CallForPapers`；`https://iclr.cc/Conferences/2027/Dates`。

[R10] Batenkov, Demanet, Goldman, Yomdin. *Stability of partial Fourier matrices with clustered nodes*. `https://math.mit.edu/icg/papers/vandermonde_clusters.pdf`。本文 (6) 给出特定中心化实 sin/cos 模型的直接推导，不声称其完整陈述与该文定理相同。
