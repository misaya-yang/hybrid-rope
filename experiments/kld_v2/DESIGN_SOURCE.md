# KLD v2：关键审查、修正推导与最小真实模型实验

日期：2026-09-08

## 0. 投入结论

原 KLD 不宜直接作为主投稿方向重投入。正交地址下的历史队列、状态非扩张和共享三角分解，没有证明真实混合模型存在对应瓶颈，更没有证明同状态预算下超过强基线。合成任务跑通本身不足以支撑强录用判断。

本版保留有用的算子，但撤回通用“实体内部序位”的解释。把它改为一个使用原 checkpoint 的 keys、values、写入门和衰减的**擦除敏感性读出**。先在真实 checkpoint 上比较同预算的读出适配，不先从头训练模型去期待它学出理想地址。

这不是通过改名字获得新颖性。能否成为论文，取决于它能否超过额外普通记忆、多擦除率记忆及相关强递归基线，并且在完整模型中改善实际答案。

## 1. 原版哪些正确，哪些推断越界

对于单位 key、beta∈[0,1]，原两槽更新为：

\[
S_t=(I-\beta_tP_t)S_{t-1}+\beta_tk_tv_t^\top,
\quad
H_t=(I-\beta_tP_t)H_{t-1}+\beta_tP_tS_{t-1},
\quad P_t=k_tk_t^\top.
\]

以下算子事实成立：正交地址、完整写入时的两槽历史；固定驱动序列下的联合状态非扩张；使用更新前读数的共享三角系统。它们不推出：真实 key 对应稳定的实体属性；gate 对应语义事件；SGD 会学会这套协议；保留旧值比其他用途更划算；decoder 会用它；全局路径没有已经解决同一问题。

原先的局部 writer 实验可以隔离计算，但不应作为主效果实验。切断跨记录上下文可能同时切掉普通模型的合法解决路径；其胜利不能外推为完整混合架构的胜利。

### 1.1 独立干扰下，KLD 会出现虚假的序位推进

只追踪一个既有写入的线性影响，其他写入的 value 为零。每个干扰 key 独立、单位范数、各向同性，满足 E[P_t]=I/d；gate 固定为 b。令 p=b/d、a=1-p。对两槽的影响矩阵，有精确期望递推：

\[
\mathbb E\begin{bmatrix}\delta S_n\\\delta H_n\end{bmatrix}
=
\begin{bmatrix}aI&0\\pI&aI\end{bmatrix}^{n}
\begin{bmatrix}\delta S_0\\\delta H_0\end{bmatrix}.
\]

若初始影响仅在当前槽：

\[
\mathbb E\delta S_n=a^n\delta S_0,
\qquad
\mathbb E\delta H_n=np\,a^{n-1}\delta S_0.
\]

因此，无关输入不仅衰减旧影响，还将其推入历史槽。d=128、b=1 时，128 次独立干扰后的当前槽系数为 0.3664，历史槽为 0.3693；512 次后分别为 0.01803、0.07269。

这是该随机模型下的精确期望，不是实际 checkpoint 的数值预测；也不是完整状态已无信息的结论。它否定的是“只要单个地址相似度很小，就无需担心长序列推进”的推理。不能假设训练后的 gates、key 分布符合这些条件。

上一版代码中，固定同一干扰 key、持续写入同一个值 100 次的例子，在第 2 次后已饱和。该例子只能证明非正交串扰，不能证明长时间累计退化。

### 1.2 一次 token 写入不等于一次语义更新

一条事实通常由多个 token 构成。实体名、属性名、值与结束标志可能触发不同 key/gate；同一事实重复提及也可能再次写入。原版需要模型额外学出稳定寻址、事务边界与重复识别，但没有给出学习这些协议的依据。

即使地址精确，M 次非更新误触发，每次 gate=epsilon，未被推进的系数包含 (1-epsilon)^M。这是需要真实轨迹检验的条件，不是用“小门值”自动解决的问题。

## 2. 最重要的等价：KLD 是擦除强度方向上的 Taylor 系数

固定进入该层的全部投影与门控，定义一个参数化记忆族：

\[
S_t(z)=\big[I-(1-z)\beta_tP_t\big]S_{t-1}(z)+\beta_tk_tv_t^\top.
\]

这里 z 只改变擦除，不改变写入。令 S_t(0)=S_t。对 z 求导：

\[
\left.\partial_zS_t(z)\right|_0
=(I-\beta_tP_t)\left.\partial_zS_{t-1}(z)\right|_0
+\beta_tP_tS_{t-1}.
\]

这恰好是 H_t。更一般，若 S_t(z)=sum_r z^r S_t^{(r)}，其前 R 个系数恰好服从原 R 槽递推；S_t^{(r)}=(1/r!)partial_z^r S_t(z)|_0。

结论：任意真实 key 下，该状态的精确含义是“对减弱擦除的敏感性”；只有理想写入条件下，它才退化为“上一条语义记录”。这也说明必须加入不同擦除率的记忆作为强对照，因为：

\[
H_t=\lim_{\epsilon\to0}\frac{S_t(\epsilon)-S_t(0)}{\epsilon}.
\]

导数表示可能有更好的数值条件或共享计算，但并没有凭空创造一个普通多状态动力学无法逼近的函数族。

## 3. 修订方法：保留原记忆，加可读的擦除轨迹

### 3.1 原 checkpoint 完整保留

设原 GDN/KDA 更新为：

\[
\bar S_{t-1}=D_tS_{t-1},
\quad
S_t=(I-\beta_tP_t)\bar S_{t-1}+\beta_tk_tv_t^\top,
\]

其中 D_t 是原模型实际使用的逐头标量或逐通道对角衰减。新增：

\[
\bar H_{t-1}=D_tH_{t-1},
\qquad
\boxed{H_t=(I-\beta_tP_t)\bar H_{t-1}+\beta_tP_t\bar S_{t-1}.}
\]

等价于：

\[
H_t=\left.\partial_z S_t(z)\right|_0,
\quad
S_t(z)=[I-(1-z)\beta_tP_t]D_tS_{t-1}(z)+\beta_tk_tv_t^\top.
\]

这是固定驱动序列下的层内偏导，不是整网参数改变后的总导数，更不是“少忘一点必然改善答案”的结论。

新增读出使用 q_t^T H_t；为公平比较，当前读出 q_t^T S_t 也可输入相同规模的残差读出。保持原头的 normalization、output gate 和 output projection 不变，将新增残差加到原模块输出后：

\[
y_t^{new}=y_t^{base}+U\,\phi(q_t^TS_t,q_t^TH_t),\qquad U_{init}=0.
\]

phi 在所有对照中相同。不要让 baseline 缺少 current-read 通道，否则它无法公平做两轨迹差分。若使用低秩 U，零初始化其输出因子，不要同时把两个因子都置零。

U=0 且原路径不改时，整个模型的输出应恢复原 checkpoint；这只保证初始化一致，不保证训练后的泛化。

第一轮固定原 q/k/v/beta/D，只训练读出；不需要原模型先学出新的实体地址或事件 gate。学习 reader 是否足够，直接由真实输出验证。

### 3.2 稳定性

先对两槽共同施加 D_t，再施加原两槽算子。若 norm(D_t)≤1、norm(k_t)=1、beta_t∈[0,1]，固定驱动下联合状态差异非扩张。

此结论没有梯度下界，不能保证旧信息不消失；也不自动适用于任意非对称 GDN2 擦除算子。移植至其他更新规则前需重新检查其实际形式。

### 3.3 共享对角衰减下的精确 chunk 形式

在 chunk 内定义 D_{a:b}=D_b...D_a，空乘积为 I，初始状态为 S_in,H_in。令：

\[
X_t=k_t^TD_{1:t},
\qquad
L_{ts}=\mathbf1[s<t]\beta_s k_t^TD_{s+1:t}k_s,
\quad A=I+L.
\]

残差满足：

\[
AE^S=V-XS_{in},
\qquad
AE^H=X(S_{in}-H_{in})+LE^S.
\]

两次求解共享 A。最终状态为：

\[
S_{out}=D_{1:C}S_{in}+\sum_s\beta_sD_{s+1:C}k_s(E_s^S)^T,
\]

H_out 同理替换初始状态与残差。输出可用相同带衰减的 causal QK 矩阵计算。

实现不得将全序列累积衰减直接相除；应使用原 kernel 的 chunk-local 稳定处理或直接局部乘积。参考实现刻意用直接局部乘积，只检查代数，未实现融合 GPU 内核。

## 4. 理论如何产生可量化的实验预测

### 4.1 用真实驱动计算完整的源影响，不能只看目标信号范数

固定 K、D、beta 后，记 A_t=(I-beta_t P_t)D_t。源 i 对时刻 t 当前读出的系数为：

\[
c^S_{ti}=\beta_iq_t^TA_t...A_{i+1}k_i.
\]

历史读出对应其擦除方向导数：

\[
c^H_{ti}=\beta_i\sum_{j=i+1}^{t}q_t^TA_t...A_{j+1}
(\beta_jP_jD_j)A_{j-1}...A_{i+1}k_i.
\]

这些项可能变号、相消。H 大不能说明它保存了正确内容，也不能说明 reader 已学会利用。

### 4.2 可恢复性和干扰必须同时进入预测

一个严格可解的诊断是：冻结真实驱动，将被追踪的单维 payload 设为独立标准高斯；把可见状态特征写成 x=Cv+eta，其中 eta 与 v 独立、均值为零、协方差为 Sigma。对目标 v_i，最优线性均方误差为：

\[
E_i^*=1-c_i^T(CC^T+\Sigma)^\dagger c_i.
\]

若所有量联合高斯，这也是最优预测器的 Bayes MSE；一般独立零均值条件下只是最优线性 MSE。它比较目标源信息和所有竞争源，而不是只比较目标源幅度。

对于固定同一特征映射 C，还有：

\[
\sum_i(1-E_i^*)=\operatorname{tr}[(CC^T+\Sigma)^\dagger CC^T]\le\operatorname{rank}(C)\le\dim(x).
\]

求和时 C 必须固定，不能把每个目标都换一个 query 后的不同 C 混在一起使用。这个记忆容量观点属于经典线性读出/动力系统理论，不作为新的论文定理。

比较候选 H 与相同状态量的额外普通记忆 X，而不是候选双状态与单状态基线。测量时必须检查实际状态精度、协方差条件数、SVD/pseudoinverse 截断稳定性，防止把病态线性反演包装成记忆收益。

这仍是层内因果诊断，不是自然文本能力证明。主效果必须使用原始文本、完整计算路径与最终生成。

### 4.3 预期结果的区分

- 目标过去确实影响当前层写入、随后主要被 delta 擦除：H 可能提供 S 不再易读的成分。
- 目标主要被 D 衰减、从未形成有用 value，或者 archive 中竞争源更强：H 不应被期待修复。
- 额外慢擦除记忆已提供同等可读信息：H 的特定贡献不成立，即使二者都超过原模型。
- 层内信息改善、完整模型答案不改善：reader/其他路径/任务需求链条尚未闭合；不能把 probe 作为主结论。

## 5. 实验顺序：先真实 checkpoint，再决定是否学习新架构

### 5.1 首个模型和改动范围

Qwen3.5-0.8B 官方配置为 24 层、18 个 linear-attention 层与 6 个 full-attention 层；linear key/value head dim 均为 128、各 16 heads，模型配置包含 float32 recurrent state。以实际加载版本为准记录 gate、conv、norm 与 cache 实现。

先固定最后一个 recurrent block 为入口，不做大范围 head/layer 搜索。低成本负结果只否定这个迁移入口，不扩大成所有层和所有训练方式不可能。

保留全注意力、卷积、残差和 FFN。不得为增加候选收益屏蔽原模型合法路径。只有独立机制诊断使用局部隔离。

### 5.2 必要对照

1. 原 checkpoint + 相同规模的 current-read 输出适配，用于隔离额外训练的收益。
2. 原 checkpoint + 同状态量的额外普通 delta 记忆/可学习擦除率轨迹 X。其读出同时看到 S 与 X，保持与 S+H 相同参数预算。写入与擦除可独立缩放，使有限差分竞争解释不被人为排除。
3. 原 checkpoint + H 擦除敏感性状态与相同读出。

对照2不能只固定一个故意接近重复状态的配置；允许在独立开发集上学习擦除率，并给候选与对照相同调参预算。若原 checkpoint + 简单 gate 适配就匹配候选，它同样否定复杂状态的投入价值。

若进入架构论文阶段，加入与论点最相关的 MDN/GDN2/EDA 强基线；不要凭本文小适配实验宣称超越这些从头预训练的模型。

### 5.3 数据与输出

主数据使用完整文本，包含旧版本查询、当前状态查询、重复提及、跨属性更新、无关干扰与缺失答案。随机化值以降低先验猜测；按照基础文档/实体配置划分训练、开发、测试，整个反事实家族必须处于同一 split。

训练与主测试覆盖相同的 4K–32K 范围；这是建议的首轮规模，不是声称该范围覆盖全部原生长上下文。

除答案 token 的 teacher-forced NLL 外，必须测自由生成语义正确率、终止与格式，并分开报告。报告当前值任务的退化，防止历史残留让模型更容易回答过时信息。

外部验证可用 LongMemEval cleaned v1 的时间推理、知识更新以及匹配的非时间任务。使用原始历史；不能按 has_answer 选择性裁剪后称为官方完整分数。资源所需的短上下文改编必须单独命名。不要在这个公开测试集上选择层、gate 或训练 reader。

同时使用独立普通文本检查 LM NLL，使用短距离同任务检查模型是否理解输出要求。

### 5.4 精确的因果干预

局部机制实验固定真实驱动序列，只在旧记录对应的 value 通道注入两个不同 payload，分别传播得到 H^(a)、H^(b)。查询时仅替换 archive state；S、query、其他层 state/cache、残差流保持受体版本。对照包括 current-state 替换、archive 随机置换或匹配强度干扰。

若输出随 archive payload 变化，说明这个读出路径有因果作用。它不意味着自然语言旧记录已经被正确编码：后者由完整文本的配对反事实评估判断。

不能拿来自两段不同原文的整个 hidden state 或整层 cache 做替换，再将效果全部归因于 archive；这会同时带入其他答案通道。

### 5.5 后续投入条件与停止解释

第一轮只训练 reader；不同时改 q/k、gate、衰减、位置频率、层比率。目标是让结果具有明确解释。

若 H 的真实生成收益不超过额外普通记忆，停止将 KLD 作为新的位置机制推进。若 H 的层内信号没有新增信息，不用更长训练 reader 弥补。若有稳健新增信息但读出未使用，只围绕该已观测的缺口进行适配，而不是重新猜一套架构。

只有真实输出在同成本基线前有优势，才值得扩大适配或安排小规模从头训练。报告配对差异和不确定性、状态字节、prefill/decode 吞吐、峰值显存；“多状态胜单状态”仅是预算扩张，不能称为效率提升。

## 6. 新颖性与论文中心

SFDA 的 partial-permutation 扩展已有有限栈构造；默认 phase-only SFDA 不等于该扩展。MDN 已有动量双状态与分块算法。GDN2、EDA 分别丰富 gate 与地址层面的擦除/写入控制。这些是必须正面处理的近邻，不是仅在相关工作里列名。

修订版可争取的中心是：**在保持当前记忆动态不变的情况下，让被擦除轨迹成为 query 可读的额外表示；证明何时它具有普通同预算记忆没有的有效信息，并在完整混合模型中实现质量—成本收益。**

目前只有构造、精确解释和可区分实验，尚无最后的实效证据。若只有人工上一值任务胜利，不是足够强的中心；若通过重新命名导数状态、堆稳定性定理来包装，仍然没有解决新颖性问题。

## 7. 本轮实际完成的检查

`kld_audit_v2.py` 检查并生成 `kld_audit_v2_results.json`：

- 原 KLD/共享衰减修订版与擦除方向偏导等价；PyTorch 双精度自动微分最大误差 1.11e-16。
- 共享对角衰减 chunk 与递推一致，最大误差 2.22e-16。
- 原固定干扰反例在第2次后饱和。
- 独立各向同性干扰的期望公式与蒙特卡洛一致。
- 随机 payload 的线性可恢复性公式与蒙特卡洛一致。

以上均非训练后的 LLM 效果、GPU 内核性能或论文录用证据。

## 主要原始资料（本轮核对）

[1] Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention. arXiv:2605.22791；NVlabs/GatedDeltaNet-2 官方仓库。

[2] MDN: Parallelizing Stepwise Momentum for Delta Linear Attention. arXiv:2605.05838。

[3] Semidirect Fourier Delta Attention: Phase-Controlled Delta Memory with Constructive Chunk-WY Kernels. arXiv:2607.11897，重点区分 phase-only 与 partial-permutation 扩展。

[4] Erase-then-Delta Attention: Decoupling Erase and Write Addresses in Delta-Rule Linear Attention. arXiv:2606.26560。

[5] Sparse Delta Memory: Scaling the State of Linear RNNs through Sparsity. arXiv:2607.07386。

[6] Qwen/Qwen3.5-0.8B 官方模型卡及 config.json。

[7] Dambre et al. Information Processing Capacity of Dynamical Systems. Scientific Reports 2, 514 (2012)。

[8] Ballarin, Grigoryeva, Ortega. Memory of recurrent networks: Do we compute it right? JMLR 25(243), 2024。

[9] LongMemEval 官方仓库 xiaowu0162/longmemeval；本计划明确使用 cleaned v1 而不是声称最新 V2。
