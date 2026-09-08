# 原生稀疏注意力中的位置兼容摘要
## 面向 Codex 的研究计划：理论、最近邻比较、真实模型因果诊断与主实验

**日期：2026-09-08**  
**工作题目：When the Router Forgets Position: Position-Compatible Summaries for Native Sparse Attention**  
**方法暂名：Phase-Stratified Routing（PSR，相位分层路由）**  
**状态：研究方案＋已执行的 CPU 算子检查；不是已经取得模型提升的论文。**

---

## 0. 决策摘要：本轮到底换了什么

### 0.1 不再以旧预算实验作为论文中心

上一版把核心放在“少一些 rotary pairs 时，EVQ 是否更划算”。这个问题可以研究，但相对于现有频率选择、partial RoPE、MHA2MLA 和学习频率工作，它的独立重要性不够明确；即使得到一个 K×allocation 交互，也未必构成有竞争力的主结果。

本轮改成一个直接影响现代长上下文计算的问题：

> **模型已经支持当前上下文长度，但索引器在筛选候选时压缩了位置敏感的 key。如何在不读取全部原始 KV 的情况下，保留后续精确读取需要的位置关系？**

关键不是扩大频率表的外推范围，而是**避免位置关系在进入精确读取之前被错误摘要抹掉**。

### 0.2 一篇论文只争取这条主结论

> **对使用位置敏感读取的模型，索引阶段“保留 RoPE 后直接池化”和“干脆去掉 RoPE”之间的取舍，部分来自摘要方式，而不一定来自位置本身无用。通过与实际旋转相位相容的块内摘要，可在原生长度内改善稀疏读取的质量—成本关系。**

这句话现在是待检验的研究假设。不能将“部分来自摘要方式”提前改写成“所有现代模型的共同瓶颈”。

### 0.3 唯一主方法

不改模型权重、不改原始频率、不旋转新 value、不引入 writer→reader bridge，不新增全局检索层。

保留物理连续 KV 块。在一个块内部，根据原生频率的相位距离，将 token offsets 分成少量组；每组保存**真实 post-RoPE key 的均值与组大小**。块选择用这些均值的加权 log-sum-exp；选中后仍读取原始 K/V。

默认：物理块 `B=64`，摘要数 `R=4`。这里的 R 是**索引摘要数**，不是旧稿的频谱 span；新代码中使用 `n_representatives`，避免符号混淆。

### 0.4 第一篇论文的部署范围

优先研究**cached-prefix QA / streaming-query decode**：长文档先作为不含问题的前缀缓存，问题到达后，所有问题 token 和答案 token 都经过被测稀疏路径。

这样既能复用成熟模型，又不会让“先用 dense attention 看过问题并计算好答案，再测试 sparse decode”掩盖路由错误。完整 sparse prefill 是后续系统扩展，不把尚未验证的 prefill 加速写进主张。

### 0.5 不是新的口号：本轮要识别的核心量

对同一真实reader，NoPE selector的误差可以精确拆成

\[
F^{\rm RoPE}-\widehat F^{\rm NoPE}
=\underbrace{(F^{\rm RoPE}-F^{\rm NoPE})}_{\text{位置目标的变化}}
+\underbrace{(F^{\rm NoPE}-\widehat F^{\rm NoPE})}_{\text{摘要误差}}.
\]

**NoPE赢了，不自动说明位置无用；RoPE均值输了，也不自动说明应该改频率。** 第4.7A节给出margin形式和真实激活归因方式。这是新问题与旧“预算/rank→能力”叙事的区别。

### 0.6 最重要的贡献门槛

只有满足以下组合，这条路线才有值得投稿的中心：

1. 在真实、原生长度、未对测试任务 SFT 的模型中，定位到**有实际回答代价的位置相关索引损失**；
2. PSR 的改善不能被**同摘要数量、同组大小的连续／随机分组**解释；
3. 相对于官方 NoPE/PE selector、Quest，以及 Prism/COBS 的相关强实现，得到可靠的**回答质量—实际总读取成本或延迟**优势；
4. 至少一个官方稀疏实现和一个不同的 hybrid 骨干支持该结论。

本计划不以“多一个正确恒等式”“attention mass 更高”或“构造任务成功”为论文级终点。也不预估录用概率。

---

## 1. 截至 2026-09 的事实：位置没有消失，而是分散在不同接口

### 1.1 需要区分的实际架构

| 对象 | 本轮从一手资料核对的事实 | 对研究问题的约束 |
|---|---|---|
| DeepSeek-V4 | CSA 先压缩 KV 再稀疏选择；HCA 对更重压缩的条目进行全局读取；另有局部路径，且 Q、共享 KV 与输出涉及 partial rotary 处理。[R1] | 直接压缩的主 KV 不等于“索引压缩、答案读取原始 KV”。只改索引不能补回丢失的 value。 |
| Kimi K3 | 全局 MLA 使用 NoPE；KDA 提供顺序敏感、recency-aware 的状态混合。报告仍有长上下文课程与数据训练。[R2] | “MLA 无显式 RoPE”不等于整个网络没有位置信息。不能证明所有全局层都需要 RoPE。 |
| Qwen3.8-Flash-Next | GDN 与 QSA 混合；索引和原始 token 读取是不同路径。模型卡区分原生 262,144 与通过扩展配置达到的 1M。[R3] | 本轮接受“研究重心转向原生窗口”，但不能假定所有模型已经以同样方式原生训练 1M。 |
| MiniCPM4.1-8B | 官方提供 dense / InfLLM-v2 路径，以及只用于块选择的 `use_nope` 开关；原生支持长度为 65,536。[R4–R6] | 适合直接检验“selector 的位置处理”而不重训模型。 |
| Qwen3.5-2B | 配置为 24 层，18 个 linear-attention 层、6 个 full-attention 层；full head 为 256 维，partial rotary factor 0.25；支持 262,144。[R7] | 可承担成本的 hybrid 骨干；对其 full 层加稀疏选择属于 post-training sparse conversion，不能称其原生训练了我们的 selector。 |

**本文不把“稀疏”“线性状态”“KV 压缩”当成同一种机制。** 也不把巨型旗舰作为在单卡上能够直接实施的实验对象。

### 1.2 “还需要位置编码吗？”应该如何回答

需要的是完成任务所需的顺序／距离信息，而不是某一种固定公式。因果 mask、局部窗口、递归状态、内容特征和显式 PE 都可能承担一部分作用。

RoPE 在稀疏 mask 下仍满足其旋转组合律；稀疏化本身不会使

\[
(R(p_i)q_i)^\top R(p_j)k_j=q_i^\top R(p_j-p_i)k_j
\]

变成错误。改变的是**哪些候选能够进入这个运算，以及被读条目代表一个 token 还是一个集合**。

因此，真正要研究的不是“2026 年应该统一删除或保留 RoPE”，而是三个接口：

- **状态更新**：系统已经怎样保留顺序？
- **候选索引**：在低成本摘要中必须保留哪些关系，才能不漏掉需要精确读的候选？
- **精确或压缩读取**：读到的对象是什么，其位置标签和内容来源是否匹配？

本轮选择第二个接口。它比“统一替换所有位置编码”更容易形成完整的理论—方法—能力—成本证据链，也比仅减少 rotary dimensions 更直接触及有效长上下文。

### 1.3 为什么这不是偷偷退回外推

两个模型的所有主结果都位于各自已支持的长度内。不得通过增大 `max_position_embeddings`、加新 YaRN factor 或自定义 position remapping，把超出范围的数据混入主结果。

已经训练到长窗口，仍不保证一个低成本索引摘要能保留读取所需的区分；但模型也可能学会补偿。因此必须先检查**真实激活与真实错误**，不能只用固定内容的三角函数例子宣布架构有缺陷。

---

## 2. 最近邻已经占据了什么：本轮不能再犯的 novelty 错误

| 最近邻 | 已覆盖的关键点 | PSR 必须额外证明什么 |
|---|---|---|
| **Prism** [R8] | RoPE 后均值池化的相消／低通效应；分频段评分和能量温度校准。 | 不是重述相消；需要证明**池化之前的分组方式**保留了单均值后处理无法恢复的关系，并在等成本下胜过其强实现。 |
| **COBS** [R9] | 用块内矩估计 softmax mass；均值＋低秩协方差、query subspace、量化；并使用 NoPE 压缩／选择。 | 不能把二阶项或 block mass 当新发现。需要检验多相位分组在真实 RoPE 读取、多模态 score 分布下的增量，以及与量化 COBS 的公平成本比较。 |
| **SAAP** [R10] | 不旋转的 key 上做聚类、学习 asymmetric query→cluster 索引，再精确读取。 | 不是首次聚类稀疏注意力。区别是**不重排全局 KV、不训练查询分类器、在物理块内用静态相位分组**；是否值得仍由成本与效果决定。 |
| **FASA** [R11] | 利用频率通道的差异进行轻量稀疏候选选择。 | 不是首次 frequency-aware selection。我们改变的是摘要聚合，而非少数通道直接替全头排序。 |
| **Quest** [R12] | 页级 key 范围信息与 query-aware 稀疏读取。 | 必须比较成熟 decode 方法，不能只打一个均值基线。 |
| **RNoPE-SWA / HyPE** [R13–R14] | 全局 NoPE 与局部／线性位置处理的分工。 | 不能把分支分工或“全局层去 RoPE”当新意。 |
| **TAPE / RoVE / PPE** [R15–R17] | 上下文化位置、value 侧旋转、压缩条目的来源位置处理。 | 本轮不再提出 RRPE 或 Transport-then-Pool；PSR 不承担其地址交接或压缩 value 解码问题。 |

本轮检索没有确认与“原生 rotary orbit 的块内静态分组＋多个真实 key 均值＋不改变物理 KV 读取”的完整方案完全相同的实现。**这不是穷尽性查新证明，也不是仅凭不同名字宣布新颖。**

### 2.1 研究价值究竟在哪里

最有说服力的论文不是：

> 我们又发明了一个 pooling heuristic。

而是：

> **某些被解释为“全局位置无用”的稀疏读取退化，实际上与位置如何进入可缓存摘要有关。把摘要做成位置兼容后，不需要改预训练权重或重新选择频率，也能在原生窗口内保留更多有用关系，并降低精确读取预算。**

这项主张需要同时有两个结果：**机制反事实**说明不是多存几个向量就行；**真实模型 frontier**说明不是只有数学可行性。假如只能完成其中一个，不把它包装成已经足够的主贡献。

### 2.2 COBS 对本轮尤其重要

COBS 的 limitations 明确交代了其长程结果涉及 RULER-style SFT、位置配置差异和 KV 读量不等于运行时间。[R9] 我们应避免同样的混淆：

- 主模型不做 RULER/关系任务微调；
- 保持精确 reader 的位置编码不变，只改 selector；
- 正面测试原生自然数据和流式问题；
- 总摘要读量、原始 KV 读量、缓存构建和真实延迟同时报告。

这不是批评其结果无效，而是明确本项目能够新增的证据。

---

## 3. 科学对象：一个块不是一个“新 token 位置”

令一个注意力层的**实际 post-RoPE query 除以实际 attention 分母**为 \(\bar q\)。设原始块 \(b\) 内的实际 post-RoPE keys 为 \(y_j\)，values 为 \(v_j\)。

\[
s_j=\bar q^\top y_j,\qquad
M_b(\bar q)=\sum_{j\in b}e^{s_j},\qquad
F_b(\bar q)=\log M_b(\bar q).
\tag{1}
\]

后续精确读取面对的是整个块的候选。单个均值 \(\mu_b\) 或一个“块中心位置”，一般不代表 \(F_b\) 对所有 query 的响应。

从集合角度，块携带一个经验测度：

\[
\nu_b=\sum_{j\in b}\delta_{(y_j,v_j)}.
\tag{2}
\]

**选择原始 KV 的索引器**只需要近似有用的块选择指标；本轮用 \(M_b\) 作为与原 reader 对齐的、value-agnostic 指标。**直接压缩主 KV**还需要近似

\[
N_b(\bar q)=\sum_{j\in b}e^{s_j}v_j.
\tag{3}
\]

PSR 只解决前者，不把 \(M_b\) 的近似当成 \(N_b\) 的解码能力。

块 mass 不是任务奖励，也不一定给出实际回答最优的候选集合。它是一个可以精确测量、能与读取误差建立联系的中间目标；COBS 已明确使用过这个目标。[R9]

---

## 4. 可立住的理论：从信息损失到可缓存近似，而非从 rank 推到 PPL

本节将标准数学事实、针对该接口的推导、待验证的模型假设分开。**不将 Jensen、Hoeffding、k-center 或 softmax 的标准性质冒充新的定理。** 论文的理论贡献应是这些性质如何精确约束位置兼容摘要，以及实际识别出的主导误差，而不是定理数量。

### 4.1 单个线性摘要何时可能精确

**命题 A：** 若在含开集的 query 域上存在 \(c,\mu\)，使

\[
F_b(\bar q)=c+\bar q^\top\mu,
\]

则块内所有 \(y_j\) 必须相同。反向显然成立，此时 \(c=\log |b|\)。

**证明。** 令 \(p_j(\bar q)=e^{s_j}/M_b\)，则

\[
\nabla F_b=\mathbb E_{p}y,
\qquad
\nabla^2F_b=\operatorname{Cov}_{p}(y).
\tag{4}
\]

若 \(F_b\) 在开集上仿射，协方差为零。每个有限 logit 的 \(p_j\) 都严格正，故全部 key 相同。若 query 只在一个子空间变化，结论相应只约束 keys 在该子空间上的投影。

**含义。** 给整个块一个“校正后向量＋位置”并不能一般性地精确恢复其 softmax response。**不意味着任何有限样本上的单均值都表现不好**；实际 query 可能恰好位于近似线性的区域。

### 4.2 均值摘要丢掉的量：精确 KL，不依赖小相位展开

记 \(n=|b|\)，\(\mu_b=n^{-1}\sum_jy_j\)，\(U_b\) 是块内均匀分布。则

\[
\boxed{
F_b(\bar q)-\log n-\bar q^\top\mu_b
=D_{\mathrm{KL}}\big(U_b\Vert p_b(\bar q)\big)\ge0.
}
\tag{5}
\]

**证明。** 将 \(\log p_j=s_j-F_b\) 代入 \(n^{-1}\sum_j\log[(1/n)/p_j]\) 即得。

这说明均值评分忽略的是**块内注意力分布相对均匀分布的偏离**。它既可能来自内容差异，也可能来自位置旋转；不能只看 gap 大就归因 RoPE。

更一般地，在固定参考 query \(\bar q_0\) 的 softmax 分布 \(p_0\) 下，\(\mu_0=\mathbb E_{p_0}y\)，有

\[
F_b(\bar q)=\bar q^\top\mu_0+H(p_0)
+D_{\mathrm{KL}}(p_0\Vert p_b(\bar q)).
\tag{6}
\]

所以即便用某个参考 query 的“最佳加权中心”，换 query 后仍有一般非零的余项。本文不把这个参考 query 做成新增模块。

### 4.3 多摘要的非渐近区间

把块划成非空分组 \(\mathcal P_b=\{C_1,\ldots,C_R\}\)，\(n_r=|C_r|\)，\(\mu_r=n_r^{-1}\sum_{j\in C_r}y_j\)。定义

\[
\widehat M_b(\bar q)=\sum_{r=1}^R n_r e^{\bar q^\top\mu_r},
\quad
\widehat F_b=\operatorname{LSE}_{r}(\log n_r+\bar q^\top\mu_r).
\tag{7}
\]

每组有精确 Jensen gap

\[
J_r=\log\left[\frac1{n_r}\sum_{j\in C_r}
 e^{\bar q^\top(y_j-\mu_r)}\right].
\]

令 \(\widehat\pi_r=n_re^{\bar q^\top\mu_r}/\widehat M_b\)，则

\[
\boxed{F_b-\widehat F_b=
\log\sum_r\widehat\pi_re^{J_r}.}
\tag{8}
\]

若组内 score 范围为 \(\Delta s_r=\max s_j-\min s_j\)，标准 Hoeffding lemma 给出 [R21]

\[
0\le J_r\le \frac{(\Delta s_r)^2}{8}.
\tag{9}
\]

这不要求随机 token 独立；这里是把有限组的经验均匀分布当作一个有界随机变量，不是对训练样本作独立假设。

可缓存的保守版本令

\[
\rho_r=\max_{j\in C_r}\|y_j-\mu_r\|_2,
\quad a_r=\|\bar q\|_2\rho_r,
\quad u_r=\min(a_r^2/2,a_r).
\]

由于 score 偏差在 \([-a_r,a_r]\)，且其均值为零，得到

\[
\boxed{
\widehat F_b\le F_b
\le \operatorname{LSE}_r(\log n_r+\bar q^\top\mu_r+u_r).
}
\tag{10}
\]

\(\rho_r\) 只用于诊断和误差上界，不是第一版部署必须增加的自适应精排流程。高维情况下该上界可能松；不能用它宣称已经获得实用的无漏选证书。BF16/FP4 舍入也需要额外误差包络。

**精细化性质。** 若对一个固定分组继续拆分，\(\widehat M_b\) 不减；单元素组时完全精确。但“各块下界更紧”不保证 top-k 排名更好：不同块可以以不同速度逼近真实值。附带测试给出了真实分组构造的反例。

### 4.4 如何把误差分成“内容”和“相位”

在正交 rotary 部分，将真实 key 写为

\[
y_s=R_\Omega(p_s)u_s.
\]

固定的原生幅度缩放可以吸收到 \(u_s\) 中；非旋转维用恒等块。则

\[
y_s-y_t=
R(p_s)(u_s-u_t)+[R(p_s)-R(p_t)]u_t.
\tag{11}
\]

第二项的平方范数精确为

\[
\boxed{
\|[R(p_s)-R(p_t)]u_t\|^2
=4\sum_k\|u_t^{(k)}\|^2
\sin^2\!\left(\frac{\omega_k(p_s-p_t)}2\right).
}
\tag{12}
\]

这保留完整二维 pair，不只看 cos 分量，也不做长距离 Taylor 截断。

定义不依赖内容的相位距离

\[
d_\Omega(s,t)^2=
4\sum_{k=1}^{K}\sin^2\!\left(\frac{\omega_k(s-t)}2\right).
\tag{13}
\]

它是 orbit embedding

\[
\Phi_\Omega(s)=[\cos\omega_1s,\sin\omega_1s,\ldots]
\]

上的欧氏距离，可能是伪度量。共同平移不改变距离。

对一组 C，令 \(D_u(C)=\max_{s,t\in C}\|u_s-u_t\|\)，\(U(C)=\max_{t\in C}\|u_t\|\)，\(D_\Omega(C)=\max_{s,t\in C}d_\Omega(s,t)\)。由三角不等式和 operator norm 不超过相位 Frobenius 上界：

\[
\Delta s_C\le\|\bar q\|
\left[D_u(C)+U(C)D_\Omega(C)\right].
\tag{14}
\]

代入式 (9)：

\[
J_C\le\frac{\|\bar q\|^2}{8}
\left[D_u(C)+U(C)D_\Omega(C)\right]^2.
\tag{15}
\]

**这就是方法的推导落点。** 摘要数量有限时，让组内相位更接近，可以控制此误差界中的**位置项**；但内容项可能抵消该收益。PSR 不假定真实 keys 在块内恒定，更不声称最小化式 (13) 就最小化任务损失。

### 4.5 从相位度量到一个确定的分组算法

对固定物理块 offsets \(0,\ldots,B-1\)，解近似 k-center：用 R 个 offsets 作中心，降低最大相位覆盖半径。

采用经典 farthest-first [R20]：首中心固定为0，此后选离现有中心集合最远的 offset，tie 取最小 index，最后分配给最近中心。

经典 2-approx 证明在这里仍成立：取 R 个已选中心与下一最远点，R+1 个点的两两距离至少是 greedy 的最终覆盖半径；任何 R 个最优覆盖球必有一个同时包含其中两点，其距离至多两倍最优半径。因此 \(r_{\rm greedy}\le2r_*\)。每组相位直径至多 \(2r_{\rm greedy}\)。

这只是**相位覆盖**近似保证，不是 keys 聚类、attention mass 或 LM accuracy 的 2-approx。数学工具本身不新。

第一版不学习频率、不按 benchmark 搜权重、不逐层搜索相位 codebook。采用运行时实际频率，所有 pair 的权重为1。若数据表明内容项主导，不用不停调权重来维护方案。

### 4.6 一阶／二阶摘要确实可能丢掉不同的东西

设单频率 \(\omega=\pi/4\)，B=64，原始内容 key 都为 \((1,0)\)。块内共有8个循环相位。

- 完整均值为0；多个覆盖完整周期的连续组，其均值也为0。
- 按8个相位类分组，每组内部 key 相同，式 (7) 对每个 query 都精确。
- 按4个近相位组分组不精确，但保留部分被连续均值消去的响应。

这不是“给接近零的 pooled vector 重新放大”。**已被平均消去的方向不能仅靠后续范数归一化重建。** Prism 自身也讨论了 dead-zone 的限制，不能说它宣称能无条件恢复所有零信号。[R8]

二阶摘要也不是一般充分统计量。例如均匀分布在四个轴向单位向量与四个对角单位向量上的两个 key 集合，均值都为0、协方差都为 \(I/2\)，但其沿横轴 query 的 softmax mass 分别含

\[
\tfrac12(\cosh t+1),\qquad \cosh(t/\sqrt2),
\]

并不相同。大 \(t\) 下二阶 cumulant 的二次增长也不同于有限 key 集的线性 log-mass 渐近。这说明多峰响应可能需要不止均值／协方差；**不说明 R=4 的 PSR 一定优于实际 COBS**。

### 4.7 NoPE 的精确边界，不作整个模型的不可能性证明

若 selector 只接收未旋转的固定 Q/K，而两个合法接口实例拥有相同的这些输入，但实际 position IDs 使 post-RoPE block ranking 相反，则同一个位置盲 selector 不可能同时给出两个正确 top-1。

这只是一个**冻结接口的不可识别性**结论。完整网络的 Q/K 可能已经携带顺序信息；KDA 等状态层还会主动编码顺序。因此该结论不反驳 Kimi K3，也不证明 NoPE 整体架构比 RoPE 差。

实证对应物应该是：在同一模型中，仅切换 selector 的位置处理，测量哪些任务出现或消除错误，而不是跨两种训练架构直接比总分。

### 4.7A 更关键的理论中心：把“NoPE 的收益”拆成两个可识别项

仅证明均值会相消仍太接近 Prism。更有区分力的问题是：**去掉位置为什么有时反而提高 selector？它减少的是摘要误差，还是确实去掉了 reader 不需要的信息？**

对同一份冻结的 pre-RoPE Q/K、相同 norm/幅度/attention denominator，分别计算：

\[
F_b^R=\log\sum_j e^{s_j^R},\qquad
F_b^N=\log\sum_j e^{s_j^N}.
\]

上标R是实际RoPE reader的score，N是只撤去显式旋转的反事实。令 \(\widehat F_b^R,\widehat F_b^N\) 为各自完整均值的估计，并定义

\[
\Delta_b^{\rm pos}=F_b^R-F_b^N,
\quad J_b^R=F_b^R-\widehat F_b^R,
\quad J_b^N=F_b^N-\widehat F_b^N.
\]

则对真实reader目标，有精确分解

\[
\boxed{
F_b^R-\widehat F_b^N
=\underbrace{\Delta_b^{\rm pos}}_{\text{撤去显式位置造成的目标变化}}
+\underbrace{J_b^N}_{\text{NoPE摘要误差}},
\quad
F_b^R-\widehat F_b^R=J_b^R.
}
\tag{19}
\]

\(J_b^R,J_b^N\ge0\)，而 \(\Delta_b^{\rm pos}\) **有正有负**。因此NoPE胜过RoPE均值，有可能是显式位置确实不影响当前选择，也可能只是 \(J^R\) 太大，甚至是两个误差偶然抵消；不能仅靠最终总分区分这几种解释。

对两个候选块A、B，实际margin为 \(\Gamma^R=F_A^R-F_B^R\)。NoPE均值的margin恰好是

\[
\widehat\Gamma^N=
\Gamma^R-(\Delta_A^{\rm pos}-\Delta_B^{\rm pos})-(J_A^N-J_B^N),
\tag{20}
\]

RoPE均值则是

\[
\widehat\Gamma^R=\Gamma^R-(J_A^R-J_B^R).
\tag{21}
\]

PSR不撤去位置，只把式(21)中的完整均值gap替换成式(8)的分组gap。这明确了它应改善哪一类错误：**原始RoPE判别有意义，但均值gap差异把margin翻转的候选对**。若主要错误来自内容摘要而非相位分组，PSR就没有理论上的优先性。

另一个安全边界来自log-sum-exp的 \(\ell_\infty\) Lipschitz性质。若块内 \(\max_j|s_j^R-s_j^N|\le\delta_b\)，则

\[
|F_b^R-F_b^N|\le\delta_b.
\tag{22}
\]

所以精确NoPE的A/B margin大于 \(\delta_A+\delta_B\) 时，撤去显式旋转不会翻转该对的真实排名。这是有条件的兼容性判据，不是说必须给所有selector保留RoPE。

**实证如何使用：** P1用同一批Q/K精确计算这些项，在错误候选对上报告“位置目标变化”“池化误差”及二者交互，而不是仅画频率能量图。然后P2检验这些错误类别是否对应真实回答恢复。

这里的分解是直接恒等式，不靠新定理名字建立新颖性；有价值的是对真实native模型的可重复归因及由此导出的有效方法。若官方NoPE开关同时撤去原生RoPE幅度，需要额外的amplitude-matched反事实，不能把温度变化算作纯位置作用。

### 4.8 选择误差怎样接到读取，而不是假装等于正确答案

对固定 Q/K/V，若丢弃的 dense attention mass 为 \(\epsilon\)，所有 \(\|v_j\|\le V\)，稀疏读取在保留集合内重新归一化，则

\[
\|o-o_S\|\le2V\epsilon.
\tag{16}
\]

证明可写成 \(o=(1-\epsilon)o_S+\epsilon o_{\bar S}\)，再用两者范数上界。该 value-agnostic 联系已有相关工作，不能独占其新颖性。[R9]

若各块有质量区间 \(L_b\le M_b\le U_b\)，S 是下界的 top-m，O 是真实质量的 top-m，则

\[
\sum_{b\in O}M_b-\sum_{b\in S}M_b
\le \sum_{b\in O}(U_b-L_b).
\tag{17}
\]

因为 \(\sum_OL_b\le\sum_SL_b\le\sum_SM_b\)。这只比较 mass selection，不是任务最优 oracle。

一层输出误差更小，仍可能不改变答案；dense 输出本身也可能错。因此主结果必须是真实生成，并与 mass、value-output error、最终 logits 的诊断分开。

---

## 5. 单一方法 PSR：精确定义及退化情况

### 5.1 写缓存

1. 用模型原生函数得到 Q/K 的 norm、RoPE 和幅度；不能根据 `rope_theta` 重新猜一张频率表。
2. 物理 KV 块保持连续，默认 B=64。
3. 按该层实际 rotary frequencies 预计算 offsets 的 R=4 分组。相同表的层共享 codebook；不根据测试输出重新生成。
4. 每个完成块、每个 KV head，只对其**实际 post-RoPE key**按组累计 FP32 sum，再除以 count。部署可存 BF16 mean。
5. 组内 mean 的成员可能不相邻，但都位于同一个物理块；原始 KV cache 不做全局重排。

完整块的 counts 由静态 codebook 决定，可共享存储。部分完成块不进入远程摘要选择，留在原始 local path；chunked prefill 必须跨 chunk 保留 partial-block accumulator。

### 5.2 来一个 query 时

\[
\widehat F_b=\operatorname{LSE}_{r=1}^{R}
\left(\log n_{br}+\bar q^\top\mu_{br}\right).
\tag{18}
\]

排序后取固定 remote top-m 物理块，合并固定的局部与起始块，去重，再调用原精确读取 kernel。

代码接口的 \(\bar q\) 已经除以原始 denominator；不得在 `summary_log_mass` 内再次除 \(\sqrt d\)。也不能把维度 d 换成 rotary width。

### 5.3 GQA 与共享选择

若原 kernel 按 query head 单独选择，则各 query head 使用自己的分数；若按 KV group 共享选择，必须保持原来的选择粒度和总块数，不能每个 query head 先选 m 块再取并集，从而偷偷放大预算。

共享 group 的可比控制使用同一个 mass-based 聚合，再求 group 内质量和。该类聚合已存在于 COBS 的推导中，不作为新贡献。[R9] 不能将每个 head 的 remote-only 分布单独归一化后直接相加，否则原本几乎只读local的head会获得不合理的远端权重。对需要自定义共享group选择的adapter，采用

\[
\widehat p_{hb}=\frac{e^{\widehat F_{hb}}}{M_{h,\mathrm{mandatory}}+\sum_{c\in\mathrm{remote}} e^{\widehat F_{hc}}},
\qquad G_b=\sum_{h\in\mathrm{group}}\widehat p_{hb}.
\]

这里mandatory集合为去重后的local与sink，其质量由本来必须读取的Q/K计算；若其score在selector之前额外计算了一遍，必须计入实测成本，不能写成免费。该规则对所有同接口估计器一致。原生selector保留官方聚合并单列；若采用其现有共享规则，则所有移植臂一起采用，不偷偷改变head预算。P1的remote-conditional mass仅是专门检查远端候选的诊断量，不等于这里的跨head部署权重。

正式 adapter 应先 probe 原生 `topk_idx` 的维度和 kernel contract，再决定采用逐 head 还是共享 group；所有移植 estimator 使用同一规则。官方未修改配置仍单列结果。

### 5.4 不可省的等资源对照

PSR 的分组大小往往不均匀。因此两个主要对照必须不仅同 R，而且**同 counts**：

- `contiguous_same_counts`：将 PSR 的 counts 按顺序铺成连续组；
- `random_same_counts`：随机打乱上述 labels，固定三个分组种子，不挑最好或最差者。

另保留普通等大小连续 R 分组，防止仅击败一种不利的组大小安排。

它们使用同样的 post-RoPE keys、同样的式 (18)、同样的精确 reader、同样字节预算。若 PSR 与这些对照相当，则只能说多摘要有用，不能说相位分组是新机制。

### 5.5 关键退化与反例

- R=1：退化成完整块均值；R=B：单点表示，block mass 精确，但成本不再像摘要。
- 所有 rotary frequencies 为0：相位距离为0，明确使用同大小连续分组，并标记 `phase_active=false`；此时不应有相位特有收益。
- 若 \(u_j=R(-p_j)c\)，post-RoPE keys 全部等于 c，均值已精确；相位跨度大也没有损害。
- 若内容沿连续段恒定，而相位组跨段混合，连续分组可能优于 PSR。附带 CPU 检查已经出现这种反例。
- 共同平移且块成员保持不变时，phase metric 与真实 Q/K 对齐具有平移兼容性。**不保证改变物理块边界之后输出不变。**
- 动态频率、非仿射 position IDs、multimodal 多轴位置不能默默套用文本 offsets 的共享 codebook。本轮只做纯文本、实际配置固定的路径。

---

## 6. 理论如何事前指导实际结果

本方案没有从静态相位半径直接预测“某模型会涨多少分”。它给出三个可先用同一批真实激活判别的量：

\[
\text{真实 gap }J_C,
\quad \text{相位项与内容项的大小},
\quad \text{真实读取中漏掉的 remote mass 与 value 差异}.
\]

### 6.1 有利条件

某些真实候选块的均值发生明显相消；其位置相关 score 在 reader 中确实参与区分；PSR 降低真实组内 score range，并且收益没有被内容混合抵消；改善发生在 baseline 漏选、对读出有影响的块，而非 sink/local 已经覆盖的部分。

这是需要模型数据确认的机制条件，不是“原生长上下文”自动提供的性质。

### 6.2 无效或有害条件

若 query 几乎只使用非旋转内容方向、远端 positional logits 很弱、局部窗口已覆盖全部相位相关注意力，或者语义变化与相位分组强烈冲突，PSR 很可能不如 NoPE、连续摘要或协方差摘要。

**最需要主动排除的一种情况：Prism 关注的高频收益主要来自局部 slash，而 local path 已经完整保留这些 token。** 因此本轮所有路由诊断都必须将 local/sink 从可竞争 remote 集合剔除。不能用包含 local 的高 attention recall 冒充远端读取改善。

### 6.3 四个事前预测

| 预测 | 观测方式 | 什么结果会否定当前机制解释 |
|---|---|---|
| 相位一致分组的增量主要出现在真实 score 相消可观测的 remote 块 | 同一真实 Q/K，按实际 gap、相位/内容项分层 | 只在完全不同的内容统计下赢，相位项与增益无关系 |
| 将相位 labels 打乱而保持 counts/字节数应破坏该类增量 | matched-size random control | 随机分组同样好，说明不是相位匹配 |
| 索引位置处理的选择会依赖任务关系，而不只是距离长短 | 唯一 KV 查找 vs 重复内容、先后/版本选择 | 所谓关系增益只来自更容易的任务或更多读取 |
| 相位摘要的改进应能影响实际问题到达后的生成 | query-blind prefix cache，全部 question/answer 走 sparse | 只有重放 dense 激活的 mass 改善，live 模型不改善 |

不同频率 multiset 不能任意置换 learned pair。PSR 只用它们定义分组，并读取原来的 keys；主研究不再重复整表替换所造成的权重失配。

---

## 7. 主实验：一个原生缓存读取实验，两个阶段完成

### 7.1 主骨干与第二骨干

**主骨干：`openbmb/MiniCPM4.1-8B`。** 使用官方支持的 sparse path。原生 cap=65,536；主输入档位 nominal 16K、32K、64K，但最终 `prompt_tokens + max_new_tokens <= 65,536`。64K 档预留答案 tokens，不能真的填满后再超窗生成。[R4–R6]

该模型配置包含训练后既有的 LongRoPE factors；“本轮不做外推”是指**不新增扩展、不改其既有运行规则**，不是说 checkpoint 从来没采用过频率扩展设计。

**第二骨干：`Qwen/Qwen3.5-2B`。** 只对六个 full-attention 层增加同样的 selected-original-KV 路径；18个 linear/state 层保持原样。只用文本。16/32/64K 都不超过其支持范围。[R7]

这验证跨 attention 组成的外部效度，不能据此宣布在 DeepSeek-V4 HCA 上验证成功。0.8B 可以 debug，但不替代第二骨干的主能力评价。

### 7.2 P0：最小接入核验

在实际环境中运行官方 dense 和 sparse 样例，固定下载 revision，记录实际 RoPE、norm、attention scale、query/KV head 关系和 `topk_idx` 形状。

只做与正确性直接相关的检查：

- adapter 关闭时，原 query、key、attention 输出、完整 logits 与 cache decode 对齐；容差参考原 kernel 的重复运行差异；
- 部分 RoPE、split-half/interleaved、原有幅度、output gate 顺序不变；
- 物理 block ID 来自原始 token index，而非选中项排名；
- 一次 prefix build 与多个 chunk build 的摘要一致到声明精度；不在 chunk 边界重新起块；
- future token 不进入当前可选摘要；padding、packed-example boundary 正确；
- 稀疏新读取集合与原读取 kernel 契约完全一致；不意外新增 local、sink 或重复 block。

不要先写复杂 kernel，也不要为了核验无限扩张日志或 hash 流程。原项目发生过 wrapper parity 失败，因此这一步必须有，但通过后立即做真实模型诊断。[I2]

### 7.3 P1：同一批真实激活的只读诊断

**数据：** 从未用于最后报告的自然长文本和问题中固定24个 prefix，长度覆盖16/32/64K。用未修改 dense 骨干产生问题到达后的真实 Q/K；不把 gold answer 作为 query 的输入。另收集少量 live sparse 路径的 query，以评估 dense replay 到 sparse 轨迹的偏移。

**采样：** 每个模型固定3个相对深度层；Qwen 取第一个、中间、最后一个 full 层。每个选中层覆盖全部 KV groups及其 query heads，问题尾部与前4个自由生成 token。不是挑最有利的 head。

**记录：** 原始/未旋转与 post-RoPE Q/K、实际频率和位置、同一物理块划分、每个 query 的 remote eligibility mask。分层分片保存，不保留 NxN attention 或全模型所有层 cache dump。

**同一输入上比较：**

1. 官方 native PE 与 native NoPE selector；
2. 一个完整块均值；
3. PSR-R4；
4. matched-count continuous、matched-count random、equal-count continuous；
5. Prism 的相关评分实现；
6. COBS 的相关 estimator：至少包括其 NoPE 版本及使用真实 RoPE keys 的对应版本；
7. Quest，和 exact block-mass 诊断上限。

COBS full-covariance 只作离线强诊断；不能把未压缩的大协方差当作部署同成本。部署比较需要 low-rank/query-subspace/precision 等实际表示的 byte accounting。附带脚本中的 `full_second_cumulant_*_diagnostic` **不是 COBS 完整复现**。

**输出的四个指标：** remote retained mass；精确 reader 的 value-output error；真实 `F_b` 的估计误差／排序；position/content 项及真实组内 score range。每个指标按 prefix 聚合，不把百万个 head-query 对当独立样本。

**进入 P2 的条件：** PSR 至少在明确、非偶发的 remote 条件下优于等资源连续/随机对照，且相对于强 selector 存在可用余量。最好还能在同问题的单层替换中改善下游 answer log-probability；这仍只是诊断，不计为回答提升。

若全协方差诊断都无余量、PSR 不优于等资源分组、或者差异仅来自 local/sink，不启动长 benchmark。交付实际反例与失败层，不重新抛出十种曲线。

### 7.4 P2：真正的主结果——问题到达后的稀疏读取

**协议：**

1. 先构造一个只含系统信息与长文档的 token prefix，**不含测试问题和答案**；按同一官方 dense path 建立缓存。
2. 固定这个 prefix cache，为每种 selector 和每个问题 fork 独立 continuation；不混入上一题答案。
3. 从问题第一个 token 开始，所有被测 global/sparse layers 都采用该方法；答案完整自由生成到 EOS 或相同 token cap。
4. 生成的 history 会随方法变化，后续 Q/K 自然随之变化。这才是主 end-to-end 对比；不能全程用 dense query replay。
5. cache build、fork/copy 与 continuation 的时间分别记录；长前缀缓存复用 M 次时再报告摊销总成本。不能把 prefix cost 直接消失掉。

**原生cache接入注意：** 官方InfLLM-v2的多tokenprefill分支不能仅凭函数名假定支持“已有很长cache、再追加一段问题”的所有mask情形。第一版question按token递增提交，使用已有decode路径；该设置对全部方法相同，测量中包括question ingest。确认带prefix的chunked query mask正确后才能加速成多token调用。dense prefix产生的raw cache需要显式构建各selector所需metadata（包括NoPE keys），不能把空的InfLLM附加cache直接接上去。

每次只保持一个文档的prefix，依次评估全部方法/问题，再释放；不要把数千个全模型KV cache全部写盘。

**基准协议标记：** 对上下文本就在问题之前的模板保留原顺序。若官方模板把具体问题放在文档之前，为满足query-blind前缀，需要显式改成“context→question”，对全部方法一致，并标记为该benchmark的cached-prefix adaptation；不直接冒充官方排行榜原协议分数。评分类别、答案、完整原文不改，原模板dense分数另作检查。

完整 chat template 先 tokenize，再确定 prefix 切点；分开 tokenize 再拼字符串可能改变 BPE 边界。所有方法使用同一 token prefix。

**为什么不从答案 decode 才开 sparse：** dense question-prefill 可能已经完成远端检索，把答案放进最后 token 的 hidden state；这种测试无法有效归因 selector 的能力。

### 7.5 主评价数据与冻结划分

**自然数据 A：LongBench v1 的 Qasper、MultiFieldQA-en、NarrativeQA。** 保留完整原始上下文，仅按实际 tokenizer 长度分组，主要报告8K以上样本。使用官方任务 scorer。不能把填充后的短文本称作原生长自然数据。[R18]

**自然数据 B：LongBench v2。** 使用全部适合原生 cap 的样本，报告 `_id`、实际 token length 与类别。其题型为选择题，按官方答案提取规则评分，不用大模型 judge。`split="train"` 是数据发布命名，不代表允许拿这些题训练。[R18]

先以独立 dev 集确认各 backbone 的 full-context dense 表现不是接近随机水平；若某个骨干在 LBv2 无有效区分，不把该骨干在此基准的零差异当作 selector 结论，但仍公开全集分数。第二骨干可以以自然 v1 与下面控制任务承担能力复现。

**机制数据：**

- RULER 全部可运行的13类，16/32/64K，每类每长度固定100个 evaluation examples；完全不用于 SFT。先跑dev小样本，再冻结正式集。[R19]
- 关系对照：固定生成器产生重复记录、版本更新和 latest/before/after 查询；对照是唯一 key-value 查找。答案随机、实体和值与 dev 不重合。它们是合成机制任务，不冒充自然任务。
- 关系任务同时发布 compact 版本和完整版本。主报告不按 PSR 是否答对筛样本；`dense-compact-correct` 只作事先冻结的辅助能力子集。

机制任务不宣称强制模型执行 `find anchor → handoff → read target`；causal hidden state 可能预先整合答案，本轮也不依赖这种未经控制的两跳叙事。

**划分建议：** 自然集按公开 ID hash 分成10% dev、90% test；同一原文/来源的多个问题归入同一 split，避免 prefix 泄漏。最后报告全部test，并按来源bootstrap。P1 的prefix来自独立calibration语料，或只使用dev，不读取test激活调方法。

### 7.6 主要比较臂与预算

PSR及同接口归因对照固定 B=64、local window=2048、一个起始块；remote budget `m∈{16,32,64}`。主要固定资源点为m=32，其余两点描绘frontier。

**确认性主臂：** dense reference、官方InfLLM-v2默认与NoPE、PSR-R4、matched-size continuous、matched-size random、Quest，以及在P1/dev确定的最强Prism/COBS相关实现。

不能让Prism与COBS只出现在related work里。Prism主要针对prefill；在本轮decode/query-stream条件下应明确标记 `Prism-estimator port (Bq=1)`，不声称复现其完整prefill系统或击败其论文中的TTFT。COBS的训练骨干不同，同样需区分 `COBS-estimator port` 与作者完整系统。[R8–R9]

若可用作者实现不能适配，需要实现其文中核心估计器并通过数学对照验证；不能删掉关键校准/协方差后仍使用论文名字。SAAP/FASA可作为补充比较，需明确其calibration/训练/重排成本；不拿随意简化版本当弱对手。

Quest等完整系统可以保留作者推荐的page size；按其实际读到的原始tokens、metadata与字节进行预算换算。若移植到统一B=64 reader，标记为controlled estimator comparison，并保留作者原生page配置的系统检查；不能通过强行改page size削弱对手。

**资源的两个视角都报告：**

- 相同真实远端KV块数：直接比较回答与selector overhead；
- 相同实测总字节/延迟：比较整体frontier。PSR多摘要与COBS更高rank/低精度应在真实字节预算下竞争。

R=2/8只在dev作为预声明的表示预算曲线；R=4是主配置。不根据test挑R、频率、层集合或query norm修正项。

### 7.7 稀疏 prefill 的范围限制

第一版PSR使用逐query评分。其prefill score工作量仍是

\[
O(N_q\cdot (N/B)\cdot R\cdot d),
\]

而不是linear-time，也不是Prism的双侧块级复杂度。它可以比逐token dense score少一个B/R因子，但不能据此宣布快于成熟prefill kernel。

因此论文主系统终点是**原生缓存前缀上的问题/解码读取**。完整sparse prefill只有在另做有效的因果mask、query处理与实测优化后才能作为扩展结果。不要为覆盖所有部署阶段再加一个query pooling模块。

---

## 8. 让机制归因真的成立：不是只有几张 attention 图

### 8.1 两级 intervention

**局部因果：** 在固定Q/K/V的同一层，只替换selector，随后使用同一个精确reader；检验新集合是否真的改变value-output与下游logits。固定内容只重算PE的反事实，仅说明显式旋转对这个接口的作用，不等于自然数据世界。

**完整因果：** P2从问题到达开始采用不同selector，使所有后续计算自然变化，以真实回答为终点。这才是任务贡献。

### 8.2 不能遗漏的消融

- matched-size continuous/random：排除更多摘要与组大小分布；
- 原始query/key/noPE副本不动，仅换分组：排除改模型或改frequency；
- local和sink单独统计：排除重新拾取已经强制读到的token；
- 与COBS/Prism对照：排除已知统计量/校准机制；
- input同时含多条相似记录与唯一内容对照：区分内容选择与关系选择；
- phase-inactive数值控制：没有显式旋转时不宣称同样的phase特有收益。

不需要把每一项都扩成新的训练矩阵。本方案默认训练步数为0。

### 8.3 事先写下的结论分支

| 最终观察 | 支持的结论 | 不能写的结论 |
|---|---|---|
| PSR在同预算下超过最强selector，且优于等组数对照；真实关系/自然QA均有收益 | 位置兼容摘要可改善原生稀疏读取，具有方法价值 | 所有NoPE都错误；已解决HCA压缩value |
| PSR与随机/连续R组相当，都超过单均值 | 多摘要有价值 | phase grouping是关键创新 |
| PSR只改善mass，不改善回答或总成本 | 算子近似更好 | 已提高有效上下文/已获得accept级结果 |
| NoPE或COBS全面更好 | 当前相位分组不值得作为新主方法 | 因为样本不够所以继续任意扫曲线 |
| 只在关系控制任务有效，自然任务无优势 | 局部位置敏感失败机制 | 实用通用收益已经成立 |
| 原生caller几乎没有索引损失 | 当前骨干不是合适的瓶颈实例 | 应故意破坏baseline来造提升 |

---

## 9. 什么结果有机会支撑一篇强论文

这里是内部选题与实验证据标准，不是会议录用承诺。

### 9.1 核心图应长什么样

**横轴：实测总KV/摘要读取成本或每个问题的continuation latency。纵轴：实际生成质量。**

至少展示MiniCPM4.1与Qwen3.5 hybrid的独立panel；不能把不同模型、不同指标做一个不透明平均分。除了主曲线，再给一个同R、同counts消融，明确真正多出的是什么。

比“少K时曲线有变化”更有价值的结果是：

> **在原生长文档缓存上，用显著更少的远端读取保留同样的回答质量，或在同成本下恢复之前漏掉的关系答案，且不必重新训练模型。**

### 9.2 预声明的实用目标

任一以下结果可作为值得认真组织投稿的主效果目标，同时满足机制归因与跨骨干复现：

- 同总读取成本下，主要自然/关系指标相对最强对手有至少约5个百分点的稳定增量；或
- 自然指标在1个百分点非劣界内，实测总读取字节至少减少25%，并且主要continuation latency确有改善。

这些数字是本项目选择的实用阈值，不是统计定理。还必须报告dense质量差距、短query/长query、不同remote预算以及全部预声明测试。

若大量成本来自固定2048 local window，remote减半也可能达不到25%总成本改善。这是结果的一部分，不把remote相对比例偷换成总系统比例。

### 9.3 统计与选择偏差

- 确认性比较事先指定PSR-R4与dev上选出的最强对手；所有test臂都报告。
- 自然QA按来源文档cluster bootstrap，RULER按任务族与样本结构分层；报告paired差值与区间。
- 冻结权重没有“训练seed重复”。随机分组的seed只反映该control的分组随机性。
- 不将tokens、head-query或bootstrap次数当独立模型复现。
- 不用上万token的NLL显著性替代实际任务效果大小。
- 首轮100–200题的很小差值可能没有足够power；基于dev估计paired方差后一次性确定test样本量，不逐步窥视test直到显著。
- `EOS`失败、答案格式失败、截断与OOM均记录；所有臂相同generation settings。

---

## 10. 系统预算：把“省计算”算到真实字节和路径

### 10.1 逻辑缓存量

设每层原始缓存包含N个tokens，Hkv个KV heads，key/value宽度d，dtype为b字节。原始K/V为

\[
\mathrm{bytes}_{KV}=2NH_{kv}db.
\]

PSR摘要为

\[
\mathrm{bytes}_{summary}\approx(N/B)H_{kv}Rdb,
\]

加上实际metadata、partial accumulator和可选radius。完整块counts可共享。

以官方MiniCPM配置的N=65,536、Hkv=2、d=128、BF16为例，每层raw KV为64MiB；R=4/B=64的means为2MiB，即raw KV的3.125%。32层的means合计约64MiB。**这是逻辑存储计算，不是实测峰值或读取流量。**[R6]

官方实现同时有kernel/stride为32/16与更粗的128/64摘要路径；粗略分别对应每64 tokens约4个和1个summary。PSR-R4并不必然比原生索引存更多，但必须实际统计两者完整cache与读量。[R5]

### 10.2 必须记录的成本

- prefix建cache时间、summary构建、增量更新；
- 每问题的question ingest、first answer token、后续decode的median/p95；
- 所有summary reads、实际原始KV reads、GQA重复或共享方式；
- 排序、索引、scatter/gather、dtype scales和padding；
- 峰值显存与常驻cache；
- 前缀复用M=1、4、16时的摊销总时间。

NoPE实现若额外保留unrotated key cache，照实计入，但不要把一个可优化的重复缓存当作“原理上必须如此”的优势。COBS的FP4压缩也按真实实现计算，不假定BF16 rank3已经是其最强等字节配置。

### 10.3 GPU执行预算建议

以下是本次研究的**建议上限**，不是对用户余额的读取，也不是已经测得的耗时：

| 阶段 | 建议累计上限 | 产物 |
|---|---:|---|
| P0＋第一骨干P1 | 4 GPU小时 | parity、实际激活诊断、是否存在值得扩大验证的效应 |
| 两骨干dev与核心baseline接入 | 12 GPU小时 | 固定方法与最强比较臂、吞吐测量、test样本量预算 |
| 确认性真实生成＋性能 | 总计不超过80 GPU小时 | 主frontier、配对统计、归因消融 |

所有大规模运行先用20个dev问题测真实秒数，以 `样本数×方法数×实测时间` 估算。超预算时减少次要数据面和非主预算点，不删核心强baseline、不把质量指标换成CPU proxy。

核验通过后持续完成已授权阶段；不要用任意“5分钟门禁”或过度审计阻塞正常工作。若遇到真实无法解决的kernel问题，交付具体错误与可运行的reference，不假装已经实现GPU加速。

---

## 11. Codex实施：已核对的源码位置、需要编写的文件

### 11.1 MiniCPM的可行接入点

本轮读取了官方 `modeling_minicpm.py`：[R5]

- `MiniCPMInfLLMv2Attention`：原生sparse入口；
- `CompressK`：原生均值压缩；
- `compressed_attention`：生成物理块选择结果；
- `sparse_forward`：将选择集合交给 `infllmv2_attn_varlen_func`；
- 原实现的NoPE副本用于selector，而真实reader仍接收其原post-RoPE Q/K/V。

**最小改动位置：** 添加PSR summary cache与替代selector，继续返回原kernel需要的合法physical `topk_idx`。不重写整个attention module，不修改Q/K投影。

默认配置中remote `topk`与local blocks在内部有组合。必须数清实际集合，并保持相同local/sink约定，不能只比较配置文件里的同一个整数。

### 11.2 Qwen3.5的接入范围

使用当前官方Transformers实现，并固定实际revision；读取text_config，而不是vision_config。保留原生norm、partial rotary、output gate、GQA和全部state层。

先在六个full层用可核对的selected-raw-KV reference实现，再接适合其head_dim=256和GQA的kernel。不把额外gather复制/重复KV隐藏掉。该adapter尚未随本包实现。

### 11.3 预期目录

```text
native_sparse_position/
  configs/main.json
  psr_reference.py               # 本包已实现，CPU NumPy
  run_checks.py                  # 本包已实现
  offline_compare.py             # 本包已实现，读取真实激活NPZ
  adapters/minicpm_psr.py         # Codex待实现
  adapters/qwen35_psr.py          # Codex待实现
  kernels/psr_summary.py          # 正确性确认后再优化
  kernels/psr_scores.py
  extract_activations.py          # Codex待实现
  eval_cached_prefix.py           # Codex待实现
  benchmark_runtime.py            # Codex待实现
  analyze_paired_results.py       # Codex待实现
  outputs/operator_checks.json
  outputs/activation_metrics.json
  outputs/predictions.jsonl
  outputs/runtime.jsonl
  REPORT.md
```

这是文件职责约定，不声称用户仓库已有上述路径。

### 11.4 激活文件contract

一个NPZ对应一个model/layer/KV group。`queries_scaled [Q,D]` 为真实post-RoPE query除以真实denominator；`keys_rotated [Nblocks,B,D]` 为完成的原始物理块；`omega [K]` 为实际运行频率；`eligible [Q,Nblocks]` 排除future/local/sink。

可附 `queries_nope_scaled`、`keys_nope` 以对照NoPE selector。若声称式(19)中的纯位置变化，必须在旁文件确认这两项保留相同norm/原生幅度，并设置NPZ标记 `nope_is_amplitude_matched=true`；官方开关原样运行的结果仍单独保存。仅用于诊断的数据还需旁文件记录query positions、block starts、prompt IDs、native rotary amplitude/layout、是否来自dense或live sparse轨迹。

`offline_compare.py`只报告remote mass fidelity，不完成GQA group优化、不比较完整Prism/COBS/Quest、不输出模型回答。

### 11.5 现成可运行命令

```bash
python -m pip install -r requirements_reference.txt
python run_checks.py --out operator_checks.json
python offline_compare.py actual_activations.npz \
  --representatives 4 --topk 32 --out activation_metrics.json
```

第三条需要Codex先实际提取符合contract的数据。本包没有伪造一个“真实模型activations.npz”。

### 11.6 明天首先完成的三件事

**先定位接口，而不是启动训练。** 运行官方MiniCPM dense/sparse样例，确认其实际freq、cache和selector共享粒度；把关闭adapter的parity做完。

**然后得到真实诊断，而不是继续猜新方法。** 提取P1规定的少量dev激活，跑PSR和强比较，首先看remote而非local，分别记录有利和不利prefix，特别是式(19)–(21)的误差来源。

**最后只在有机制和效果余量时扩成完整回答实验。** 固定R和预算，跑query-blind prefix协议；优先补强对照，不增加LoRA、EVQ或新位置模块。

---

## 12. 本轮已经执行的检查与真实边界

运行 `run_checks.py` 得到23项通过的CPU算子检查，包括：

- partial-RoPE、共同平移、phase/content分解；
- NoPE目标变化/摘要误差分解、log-mass Lipschitz界和位置盲接口反例；
- KL gap恒等式、200组有限范围误差界；
- nested refinement、singleton精确性；
- 小规模枚举对照的k-center覆盖性质；
- NoPE退化、相消构造、相同前二阶矩但不同mass；
- 真实分组的排名反例；
- 100组读取误差/质量区间regret检查；
- 内容结构反对相位分组的负例。

**实际数值：** 在B=64、单频率π/4、相同raw key、query norm=4的构造里，R=4相位分组的平均log-mass gap约0.4646，同counts连续分组约2.4250；R=8相位组达到浮点精度内精确。另一个内容按连续段变化的反例中，连续摘要精确，而PSR平均gap约0.2299。

这些正负数值只是算子检查，不来自预训练语言模型。没有新GPU训练、没有真实模型回答提升、没有速度提升测量。

它们证明代码实现了声明的运算，并主动展示了方法的失败条件。**它们不构成论文主结果，也不证明R=4是实际模型最优值。**

---

## 13. 如何把主线写成论文，而不是继续写项目总结

### 13.1 建议的正文顺序

1. **问题与真实失败。** 原生cached-prefix稀疏读取中，selector位置处理与reader需求不一致；用一个真实模型例子和总体数据建立重要性。
2. **理论对象。** 一个候选块的query response不是一个点；Jensen gap和phase/content分解说明为什么单均值校准与NoPE具有不同边界。
3. **方法。** 固定物理块内的phase strata，多均值log-mixture，原始reader不变；说明cache和代价。
4. **核心实验。** 自然回答质量—总读取成本frontier，MiniCPM主结果与Qwen hybrid复现。
5. **归因。** 同R/同counts对照、NoPE/Prism/COBS/Quest、query-blind协议、负例。

不把全部标准引理堆到主文。保留一个关键误差分解和一个可执行算法，其余证明放附录。

### 13.2 旧EVQ材料如何处理

保留其已经建立的两个研究经验：**有限频率在不同尺度上有不同分辨作用；learned pair与频率有序配对不能随意拆散。** 这支持本轮用完整pair和实际freq构造相位metric。

但PSR不由Cosh surrogate推出，也没有使用EVQ频率。不能为了保留沉没成本，宣称它是EVQ的自然定理结论。

151.9M、750M、MLA、LoRA和视频结果不再被全部塞进新主文。新主线若成立，可以作为独立论文或实质重构后的论文；旧稿是否单独提交，是另一项决策，不由本方法的命名解决。

### 13.3 论文可以争取的三个贡献，而不是六条宽泛claim

**机制贡献：** 在原生缓存读取中识别位置敏感路由损失，区分position information、summary approximation和value-read后果。

**方法贡献：** 不改预训练位置表、无需模型训练、保持physical KV locality的位置兼容摘要；相对最近邻有可复核的算法差异。

**效果贡献：** 在真实自然任务和至少两类骨干上得到可用的质量—读取成本优势，且不是靠任务SFT、额外读取或弱比较得到。

这是“足够值得accept”的研究目标对应的证据结构。只有看到真实结果后，才判断论文是否达到了这个目标。

---

## 14. 来源与证据状态

### 内部来源

**[I1]** `main(20260908-154044).pdf`，用户提供的封板稿。重点：§3完整sin/cos几何；§C固定support与weight-table交叉；§E冻结表/LoRA的边界；§F.1 MLA。本文不将它的已有结果当作PSR结果。

**[I2]** `read_referenced_position_research_20260908.md`，本轮从用户Library读取。重点：实际attention接口区别、RRPE/TAPE边界、旧wrapper parity失败、RULER与自然QA结果可能不同。另检索了其他对话的项目上下文：RRPE的两读诊断存在泄漏风险，不能默认地址交接就是主瓶颈。这里只把它们作为项目历史，不当作公开论文证据。

**[I3]** `rotary_budget_theory_and_experiment_20260908.md`，上一版计划。其有限预算推导和CPU检查不是新增model evidence；本轮不沿用“budget scaling是主线”的决定。

### 本轮核对的一手公开来源（访问：2026-09-08）

**[R1] DeepSeek-V4 技术报告，§2.3。**
https://arxiv.org/html/2606.19348v1

**[R2] Kimi K3 技术报告，§2.2/MLA与§3.4。**
https://arxiv.org/html/2607.24653v1

**[R3] Qwen3.8-Flash-Next 官方模型卡。**
https://huggingface.co/Qwen/Qwen3.8-Flash-Next

**[R4] MiniCPM4.1-8B 官方模型卡：native长度、InfLLM-v2与NoPE配置。**
https://huggingface.co/openbmb/MiniCPM4.1-8B

**[R5] MiniCPM4.1 官方实现：实际selector/reader路径与cache。**
https://huggingface.co/openbmb/MiniCPM4.1-8B/raw/main/modeling_minicpm.py
https://github.com/OpenBMB/infllmv2_cuda_impl

**[R6] MiniCPM4.1 官方config：65536、32层、32Q/2KV heads、既有LongRoPE factors。**
https://huggingface.co/openbmb/MiniCPM4.1-8B/blob/main/config.json

**[R7] Qwen3.5-2B 官方config。**
https://huggingface.co/Qwen/Qwen3.5-2B/raw/main/config.json

**[R8] Prism: Spectral-Aware Block-Sparse Attention，v2，§3与dead-zone讨论。**
https://arxiv.org/html/2602.08426v2
https://github.com/xinghaow99/prism

**[R9] COBS: Cumulant Order Block Sparse Attention，§3–5、§9。**
https://arxiv.org/html/2607.09052v1

**[R10] Inference-time Sparse Attention with Asymmetric Indexing（SAAP）。**
https://arxiv.org/html/2502.08246v1

**[R11] FASA: Frequency-Aware Sparse Attention。**
https://arxiv.org/html/2602.03152

**[R12] Quest 官方项目页与作者实现。**
https://hanlab.mit.edu/projects/quest
https://github.com/mit-han-lab/Quest

**[R13] Rope to Nope and Back Again: A New Hybrid Attention Strategy。**
https://arxiv.org/html/2501.18795v2

**[R14] Hybrid Linear Attention Done Right: Efficient Distillation and Effective Architectures for Extremely Long Contexts。**
https://arxiv.org/html/2601.22156v1

**[R15] Rethinking Addressing in Language Models via Contextualized Equivariant Positional Encoding（TAPE）。**
https://arxiv.org/html/2501.00712v1

**[R16] RoVE: Rotary Value Embeddings Attention for Relative Position-dependent Value Pathways。**
https://arxiv.org/html/2606.11275v1

**[R17] PPE: Positional Preservation Embedding for Token Compression in Multimodal Large Language Models。**
https://arxiv.org/abs/2510.22936

**[R18] LongBench / LongBench-v2 官方仓库与数据schema。**
https://github.com/THUDM/LongBench
https://huggingface.co/datasets/THUDM/LongBench-v2

**[R19] RULER 官方仓库。**
https://github.com/NVIDIA/RULER

**[R20] Gonzalez, Clustering to minimize the maximum intercluster distance, 1985。**
https://www.sciencedirect.com/science/article/pii/0304397585902245

**[R21] Hoeffding, Probability Inequalities for Sums of Bounded Random Variables, 1963。**
https://www.tandfonline.com/doi/abs/10.1080/01621459.1963.10500830

main分支源码会变化。Codex实施时固定实际模型revision与依赖版本，并记录与本计划所核对接口的差异；不要把文中的访问日期当作commit hash。
