# TailSpline 的可迁移设计原则：依赖伸缩、有限相位响应与一个可判别比较

**用途：** 给 Codex 的聚焦理论补充与定点入稿说明。保留当前标题、z 主线、TailSpline/NCP/Cosh 的不同操作条件；不增加候选方法族，不把模型状态用于构造拟合。

**核对基线：** `misaya-yang/hybrid-rope@c2abf72ae208d0e351e67062b8a47e3d63e43f4b`（2026-09-18）。接续 `ROPE_THEORY_SYNTHESIS_FOR_CODEX_20260917.md`，不重复其有限网格最优解与单通道伸缩阈值推导。

**本次实际完成：** 核对最新提交、Kanana 报告、活动构造和实验章节，复读先前理论报告，核对相关原始文献；完成下述解析推导及独立数学检查。没有运行模型、读取服务器原始生成、修改远端仓库或启动 GPU。数值检查不是机制验证。

---

## 0. 今天最值得落笔的结论

值得提出的原则是：

> 在冻结扩窗中，优先把伸缩分配给随上下文扩展而拉远的内容依赖；保留承担未被拉伸局部操作的距离响应。尾侧连接是实现这种选择的一种公共参数构造，而不是“频谱越平滑，模型必然越好”。

这是关于**伸缩放在哪里**的原则，不是另建任务风险统一框架。它允许检索依赖、固定偏移操作和不同检查点有不同响应。

目前能够诚实写入方法动机的是：以上原则及已有频率使用研究，给出了尾侧优先的独立理由；TailSpline 是其可迁移的具体实现，其整体价值已有跨模型和等位移证据。尚不能写成：真实模型中的 terminal-gap 已被证明是全部收益的因果中介。

本轮最有价值的增量有两个：

1. 将已有相位误差排序升级为**带非零内容相位误差的有限响应条件**，说明何时更充分伸缩真正增加目标响应，以及何时不会。
2. 利用已存在的 T/C，提出**固定总输入长度、只交换证据远近位置**的差分比较。它不引入新频率曲线，能复用原有生成，并检验收益是否确实依赖被拉伸的内容距离。

---

## 1. 三个独立于 TailSpline 成绩的假说

### 1.1 输入中的依赖并非整体均匀拉伸

考虑由可交换记录组成的检索输入：增加无关记录、改变记录与查询之间的间隔，可以增大跨记录依赖距离，但不改变记录内部的 key–value 对应及局部 token 顺序。这个事实可以直接通过输入构造检验，不需要先看 T 的分数。

将一条明确依赖的参考距离记为 d0，实际距离为 d=αd0。α 来自该依赖的几何变换，不是自动等于输入长度 H/L，也不是自动等于部署表倍率 s。

因此，常见工作负载可以同时包含 α≈1 的局部操作和 α>1 的跨记录读出。这是本报告的操作条件，不是关于全部自然语言的普遍定律。

### 1.2 频率使用具有与依赖尺度有关的结构

需要检验的模型假说是：一些未改动的高频外带支持局部操作；尾端邻近的部分过渡通道参与较宽距离范围内的内容匹配。这里特别强调**尾端邻近的过渡通道**：T/P 的完全插值尾带完全相同，仅仅观察“最慢频率重要”不能解释二者的差异。

Barbero 等关于高频位置头、低频使用的观察提供动机，但没有证明所有模型有相同语义/位置分区。[W1]
Wu 等将频率使用联系到数据依赖尺度，并区分可随距离伸缩与固定偏移的任务，提供了更直接的参照。[W2]
本报告不照搬其最优频率结论，也不采用其模型/数据拟合建议。

两个重要限制：

- 高 Q/K 范数只表示贡献潜力，不等于正确证据的正向贡献；必须看相位和目标–干扰关系。
- 固定高频外带只保证相应 rotary blocks 的直接作用不变，不能保证整个多层网络的局部能力无损。

### 1.3 学到的内容匹配在一定相位扰动下保留相干性

需要检验的不只是“低频能量大”，而是：相关通道在一个独立定义的 Native 参考读上，确实对正确内容有相干的正响应，并且对非均匀距离扩展没有立即发生巨大上游表示漂移。

下面给出这一假说的明确数学版本。它不使用 T 的解或成绩定义真假，可以先在 Native 参考输入上测量，再对候选表给出预测。

---

## 2. 从残余相位到真实响应：一个有限、非 Taylor 的条件命题

### 2.1 设定

对一个已声明的注意力头、正确证据 key 和旋转 pair q，将共同 gain、attention normalization 吸收入复数系数 Cq。该通道的 logit 贡献为

\[
f_q(\nu_q,d)=\Re\{C_q e^{id\nu_q}\}.
\]

以 Native 频率 ωq 和独立定义的参考距离 d0 对齐坐标，写

\[
C_q e^{i\omega_q d_0}=a_q+i b_q.
\]

这里 a 是参考读上的同相分量，b 是正交相位分量。假设

\[
a_q>0,\qquad |b_q|\le\kappa_q a_q.
\tag{1}
\]

它表示参考匹配位于半角 arctan(κq) 的相干扇区，不要求 b=0。a、b 必须来自 Native/事先定义的参考读；不得根据 T−C 或 T−P 最终输赢挑通道。

方法 M 使用 νq=ωq s^(-mq^M)。实际依赖变为 d=αd0 时，定义

\[
\delta_q^M=\omega_qd_0(\alpha s^{-m_q^M}-1).
\tag{2}
\]

在暂时保持该内容系数的参考分析中，精确有

\[
f_q^M=a_q\cos\delta_q^M-b_q\sin\delta_q^M.
\tag{3}
\]

这不是小位移近似。相位使用声明参照下的未绕回残差；不能任意选不同的 2π 分支来制造满足条件。

### 2.2 有限相干响应命题

对任意两个配置 A、B，假设残差同号，且

\[
0\le r_q=|\delta_q^A|<t_q=|\delta_q^B|\le\pi.
\]

令

\[
\mu_q=(r_q+t_q)/2,\qquad h_q=(t_q-r_q)/2.
\]

则

\[
\boxed{
 f_q^A-f_q^B\ge
 2a_q\sin h_q
 \big[\sin\mu_q-\kappa_q|\cos\mu_q|\big].
}
\tag{4}
\]

若右侧为正，残差更小的 A 具有严格更大的该项内容响应。

**证明。** 式 (3) 相减。由于残差同号，cos 的差等于 cos r−cos t，sin 差的绝对值等于 |sin r−sin t|。再用 |b|≤κa：

\[
f^A-f^B\ge a(\cos r-\cos t)-\kappa a|\sin r-\sin t|.
\]

利用

\[
\cos r-\cos t=2\sin\mu\sin h,\qquad
|\sin r-\sin t|=2|\cos\mu|\sin h
\]

即得式 (4)。不需要对频率变化或相位变化做 Taylor 展开。□

**性质身份。** 式 (4) 是从有限旋转算子得到的条件性下界，不是关于真实模型已经满足条件的实证结论。三角恒等式不是数学新工具；新增用途是把公共表的残差改善接到可独立检查的 Native 内容相位条件。

### 2.3 比旧残差排序多了什么

旧结论只说明哪张表更接近一个相位参照。式 (4) 还说明：

- 为什么同相、有用的内容信号可以保留这项排序；
- Native 匹配允许有多大的非零相位偏差；
- 为什么高范数、低 residual ratio 都不足以保证收益；
- 为什么各检查点无需使用相同的精确幅度 aq：在一个符合条件的通道族中，任何非负幅度混合均保留排序。

最后一点是**可迁移性的一种条件依据**：公共表可以保护一类相干响应，而不需要逐模型拟合其精确权重。

### 2.4 两个必须保留的失败例

若 a=1、b=−1、δA=0.1、δB=0.3，则 A 的残差更小，但

\[
f^A-f^B\approx-0.156019<0.
\]

因此，不满足相干扇区条件时，残差更小完全可能更差。若残差跨过单调半周期，cos 的排序也可能反转。不得用“所有低频都承载语义”删掉这两个问题。

---

## 3. 从通道收益到目标–干扰竞争

### 3.1 为什么不能停在目标 logit

正确证据 g 的 log-odds 定义为

\[
\mathcal O^M=\ell_g^M-\log\sum_{j\ne g}e^{\ell_j^M}.
\]

令 R 为事前声明的相关通道集合，Gq 为式 (4) 的右侧。对于未纳入 R 的目标贡献、真实前向中的表示漂移，以及干扰组 partition function 的变化，分别给出上界 ξrest、ξstate、ξcomp，则

\[
\boxed{
\mathcal O^A-\mathcal O^B
\ge\sum_{q\in R}G_q
-\xi_{\rm rest}-\xi_{\rm state}-\xi_{\rm comp}.
}
\tag{5}
\]

**证明。** 将正确 logit 分成 R 与剩余通道；R 用式 (4)，剩余项用绝对值界。对真实前向系数与参考系数的差，用

\[
|\Re\{(C_q^M-C_q^0)e^{id\nu_q^M}\}|\le|C_q^M-C_q^0|
\]

分别对 A/B 求和。最后扣除干扰组 log-partition 的最大增量。□

这是有限改变的充分条件，不是把原有 log-odds 导数再做一遍数值自洽。它明确要求收益不能被错误 keys 的同步抬升抵消。

### 3.2 这些界如何不变成循环论证

不得把“ξcomp 很小”定义成“观察到 T 赢”。可以采用以下独立量：

- 对固定参考系数，所有竞争 keys 的有限 logit 变化可由其通道系数和安装表直接计算；log-sum-exp 的 1-Lipschitz 性给出其变化不超过最大的 logit 变化绝对值。
- 参考内容系数的漂移直接比较实际前向和事前 Native 参考读；如果漂移很大，就不能借参考模型断言实际机制成立。
- 一个可分析的理想情况是干扰 codes 的联合分布对逐 pair 旋转不变。此时干扰 log-partition 的**期望**与频率表无关，期望比较中的 ξcomp 可为零。独立各向同性高斯 codes 是例子，但这不是对真实 distractors 的默认假设。

式 (5) 可能保守到无法给出符号。那说明该证书尚未覆盖该读，不说明模型一定没有收益。也不得只留下通过证书且 T 赢的读来代表全模型。

### 3.3 入口变化何时可接受

对固定参考内容系数，两个配置的直接旋转差满足

\[
|f_q^A(d)-f_q^B(d)|
\le2|C_q|\left|\sin\frac{d(\nu_q^A-\nu_q^B)}2\right|
\le |C_q|\,d|\nu_q^A-\nu_q^B|.
\tag{6}
\]

这是全局成立的界，不是 Taylor 近似。若实际被使用的入口侧局部依赖满足 d≤ℓ，则相应直接影响由 ℓ 而不是目标窗口 sL 放大。主要局部操作若由共同保留的高频外带承担，改变入口附近少数过渡通道可能仍在局部 margin 容忍范围内。

但要同时满足：入口侧在远距竞争中的贡献不大，或已被 ξrest/ξcomp 覆盖；不能因为正确 key 很近，就忽略远处干扰 keys 的变化。

因此，尾侧优先具有价值的真实条件是：**伸长依赖的相干收益，超过入口及其他读的实际损失和竞争漂移。**这个条件由依赖类型、Native 内容相位和有限响应给出，不由 JT 的最小值给出。

---

## 4. 这怎样具体解释 T/P 与现有 T/C

### 4.1 T/P：尾侧收益与入口风险同时存在

对所有内部过渡位置有 Tq>Pq。对真正按 s 拉伸的依赖，α=s：

\[
0\le\delta_q^T\le\delta_q^P.
\]

在式 (4) 的条件内，T 改善相干响应。对保持绝对距离的依赖，α=1：

\[
|\delta_q^T|\ge|\delta_q^P|.
\]

此时，在同样的相干条件内 P 反而更接近原响应。

这意味着：**T/P 的成功并不是“更慢的表在所有用途上更好”，而是冻结模型的实际使用足以让伸缩收益超过相应成本。**保留外带、无新增权重、统一表的部署条件，使这个取舍可由同一公共规则实现。[R1,R2]

### 4.2 T/C：已经存在的方向性配置对照

当前精确等位移控制满足

\[
\boxed{
T_q-C_q=\frac{q(n-q)(2q-n)}{2n(n+1)(2n+1)}.
}
\tag{7}
\]

因此：

- q<n/2：T<C，T 在入口半段比 C **少伸缩**；
- q>n/2：T>C，T 在尾侧半段比 C **多伸缩**；
- 两端和总位移相同。

若独立 Native 使用分析显示，局部、未拉伸读更多使用前者，而跨记录、拉伸读更多使用后者，则式 (4) 对两类读都可给出同方向的 T 优势。相比 T/P，这更直接解释了为什么**同一总移动量的不同分配**仍然重要。

这里没有假设真实模型必须按 n/2 精确分工。n/2 是 T/C 的数学交叉点；实际使用是否与这一粗分配相符，是待检验假说。不得按 T/C 成绩后验移动分界。

### 4.3 尾端相位残差比例不能独自解释收益大小

式 (4) 显示收益不仅取决于残差比，也取决于残差所在的相位区间。即使 residual ratio 极小，当两个残差本来都接近零时，绝对 cos 收益也可能很小。

在 Llama 公共数学网格 n=17、s=4、参考距离 d0=L/4=2048、伸长距离 d=sd0=L=8192 上，令 a=1,b=0：

| q | δT | δC | 伸长后的单位相干信号 T−C | 未伸长 d0 上 T−C |
|---|---:|---:|---:|---:|
| 12 | 0.873890 | 1.018235 | +0.116979 | −0.003771 |
| 16 | 0.025544 | 0.056031 | +0.001243 | −0.007548 |

这些是有限旋转响应的计算例子，**不是任何 checkpoint 的测量**。两行的伸长响应量级差约 94 倍：更接近最后一个通道，不意味着绝对收益更大。

所以应把物理解释放在**尾端邻近的一段有用中频如何维持内容匹配**，而不是把最后一个 gap 当成信号传播的“接口”。

### 4.4 相邻频率没有自动的“边界反射”机制

RoPE 的旋转算子按 pair 块对角作用。相邻 q 并不因为索引相邻就具有波动方程、传输线或阻抗匹配中的直接耦合。JT 等价于 log-distance 通道曲率，是精确结构陈述；要把它解释成注意力损失，仍需要模型怎样组合这些通道的条件。

因此，下列写法不可用：smoother tail prevents spectral reflections；the spline is a Nyquist-optimal sampling grid；the tail seam directly transports information to neighboring channels。

当前可以用的写法是：**the tail-facing allocation preserves a useful class of dilated distance responses; the spline is a regularized, analytic way to implement that allocation preference.**

---

## 5. 为什么这一偏好可能跨模型复用

### 5.1 公共规则作用于无量纲 native-turn 坐标

有

\[
d\nu_q=\frac dL(L\omega_q^N)s^{-m_q}.
\tag{8}
\]

对于几何 Native 网格、c=log(b)/K、未发生边界裁剪的 32/1-turn band，令 u=q/n，则当前附录已有

\[
L\omega_{l+q}^N
=64\pi\,32^{-u}\exp((1-u)\eta_+-u\eta_-),
\qquad 0<\eta_\pm\le c.
\tag{9}
\]

所以公共规则在不同模型上处理的是相近的 Native 相位区间，而不是硬编码第几个绝对频率。这解释为什么大 base 本身不必压掉配置效应；剩余差异包括离散采样、band 宽度、训练使用方式和操作条件。[R1]

### 5.2 “保护一个响应类”比“拟合一套权重”更有迁移意义

如果多个模型在该无量纲频带中都使用相干的拉伸依赖读，那么式 (4) 的符号可以对不同的 aq 保持成立。它们不需要具有相同 Q/K、相同 head 编号或相同幅度分布。

这是一个可检验的公共规则迁移解释：共同的尺度化频带，加上一类相似的已学习用途。跨模型结果支持这种解释值得研究，但尚未测量出五个模型都满足同一相干条件。

### 5.3 该原则不唯一推出 JT

依赖非均匀伸缩、局部操作保留与尾侧响应相干，可以支持“尾侧伸缩/连接值得优先”的设计偏好；它们不会唯一决定 ε 的二次型系数、β=32/1 边界、或 TailSpline 三次累计曲线。

当前闭式 JT 的角色仍然是：把已声明的偏好落实为无校准、单表、可核验的构造。整条规则和高阶配置差异的实际价值由 T/P、T/C 与迁移实验支持。无需把理论写成对所有架构都求出了同一个任务风险最优解。

NCP 继续承担原生固定支持/LM 证据；Cosh 的冻结安装、训练和适配继续承担其他操作条件下的配置价值。不以本节重分配它们的论文地位。

---

## 6. 现有证据实际区分了什么

| 证据 | 已经区分的解释 | 没有单独识别的量 |
|---|---|---|
| 匹配 T/P | 同 checkpoint、s、外带、端点、gain 下，完整内部配置会改变表现 | 总位移、尾侧分配、入口变化各自的份额 |
| clean T/C 32K +2.10pp，区间 [1.11,3.08] | “全部收益仅来自总位移”不足以解释；高阶形状有任务价值 | 尾侧连接的独立因果中介；16K/32K 是否同内容依赖拉伸 |
| BM/Uni | 同总位移下形状效应并非只存在于一组 T/C 构造 | TailSpline 特有性或尾端优先的唯一性 |
| 固定支持、谱保持坐标干预 | support 和无序频谱均不能独自决定冻结行为；learned assignment 有作用 | 哪些实际通道携带正确证据，以及为何偏向尾侧 |
| 多家族与 70B 配置复用 | 同规则的有效性不是只在一个小模型成立；无逐 checkpoint 搜索的部署价值成立 | 共同机制是否已被直接观测；不能把同 benchmark 的模型面板当独立同分布抽样 |
| 检索、LM 与自然 QA | 不只一个合成终点存在收益 | 每个模型、长度和任务都一致，或 PPL 足以替代 QA |

这里最关键的剩余问题不是“TailSpline 到底有没有用”，而是：

> 在相同总位移下，T 的额外价值是否确实来自对**被拉伸的正确内容依赖**更合适的响应分配，而不只是某个不依赖证据位置的曲线评分、任务混合或输出行为差异？

这是值得补充的一步。孤立 terminal penalty 的中介份额不是本轮前置要求。

### Kanana 的使用边界

本次可读取 HEAD 的 RESULT 已落盘 T/Y=72.65/65.64 的 Full-13，并记录官方 runtime 配方为 factor4.4、beta64/2；P 在该文件仍为旧 pilot 状态。[R3]
用户本轮提供 P=70.33 的完整面板更新，本报告将它作为**用户提供的新截点**使用，不称已从 Git 重算。它不削弱当前研究方向，也不需要重做已有实验。

T/P 同 S2 等条件的比较支持内部配置效应；T/Y 比较包含 s、band 与可能的幅度规则差异，支持的是与发布方运行时方案的部署比较，不能独自隔离 z。尚在运行的 128K QA 不纳入已完成结论，也不把运行时 YaRN 自动称作 YaRN 续训历史。

---

## 7. 唯一建议的新增比较：T/C × 证据远近，固定总长度

### 7.1 为什么选择它

T/C 已匹配总位移并已有真实生成。再加曲线主要改变新的几何指标，容易继续停留在“哪个 profile 好”。本比较直接改变物理假说中的**正确依赖距离**，其他全局部署量不动。

**只做一个比较，不安排新方法家族：**在 Llama 同一 S4、同一 32K 输入长度下，比较 T/C 在同源远证据和近证据输入上的差分。

### 7.2 输入与复用

1. 从已有 clean T/C 32K 的结构化 key–value 检索行读取原始 prompt、官方答案和记录位置。只选择答案不依赖记录顺序、且能明确标注正确记录的任务；不迁移 tracking、聚合或自然 QA 段落。
2. 纳入条件只能由输入决定：有合法远位置、存在可交换且 token 长度相同的近端干扰记录，交换不改变 query、答案、任务含义或因果可见性。冻结 ID 清单后再看新输出。不因为原 T/C 输赢排除行。
3. 原输入作为 far 条件，尽量使对应 d_far>L。将完整正确记录与近端等 token 长度干扰记录交换，构造 near 条件。近端位置优先满足 d_near≈d_far/s；在真实记录边界无法精确做到时记录实际 α，不冒充严格比例。
4. 每个 source 内保留总 input_ids 长度、所有记录内容、query、答案、记录内 token 顺序、解码器、cap、表、gain。只交换两个完整记录，不重写内容、不增添 padding、不删任务失败输出。
5. T_far/C_far 可复用已有完整运行；仅生成 T_near/C_near。原运行的合同或 row 身份不匹配时不得硬复用。新 GPU 执行需要作者授权。

这个干预也改变了被交换干扰记录的距离，且完整前向的中间表示可能改变；这是完整模型的输入因果干预，不是“所有非目标 logits 固定”的理想实验。解释性读数必须保留竞争项。

### 7.3 唯一主终点

以 y 为任务的官方逐样本正确性/检索分数，定义

\[
I=\mathbb E_i\left[
(y_{T,f,i}-y_{C,f,i})-(y_{T,n,i}-y_{C,n,i})
\right].
\tag{10}
\]

任务固定等权、任务内 source 等权。以 source 为簇重采样，四个对应单元一起抽样；重复问法或多个记录来自同一 source 时不拆成独立样本。保留所有 empty/cap/unparseable 结果在主分母，并用原官方规则计分。真实纳入量由冻结 manifest 给出，不预先虚报能复用全部 2600 行。

另报告两个简单差 G_far 与 G_near，不能只报 I。如果研究样本是机制子集，其统计对象是该机制子集，不替换原 Full-13 主分数，也不合并当新增独立重复。

### 7.4 各解释的预测

| 解释 | 本比较的预测或区分点 |
|---|---|
| 仅总位移决定表现 | T/C 总位移相同，无法解释系统性差异；原 T/C 已反对这一充分解释 |
| 完全由固定 H 下的 J 或位置 Gram 排名决定，且不依赖内容用途 | 表和 H 不变；不能给出正确证据位置引起的 T/C 排名反转 |
| 尾侧通道支持拉伸后的内容匹配 | 对符合相干条件的尾侧敏感读，G_far 应更正，近端应减弱；更强签名是 G_far>0 且 G_near<0 |
| 主要是通用输出格式改善 | 不特别预测绑定纠正随正确证据距离变化；若差异只来自空输出/cap而非正确内容匹配，则不支持本机制 |

只得到 I>0 而没有反转，也可以提供依赖距离的支持，但比反转弱；不能声称已排除一切内容依赖的平滑解释。允许更复杂的 competing model 时，它也可能预测相同结果。

**不能把任何泛称“平滑解释”强行写成 I=0。**只有明确的、内容无关的表级充分指标才被反转直接反驳。

### 7.5 在同一个比较中增加解释性读数，不追加选表

如执行环境允许，在 source-ID/hash 事前确定的样本上，记录同一 query 位置与正确/干扰 keys 的有限前向贡献：

- 固定公开 band 分组，检查 Native 参考读的 a、b 与相干扇区；不要只看 Q/K 范数。
- 检查正确组相对干扰组的有限 log-odds 变化，而非导数有限差分是否一致。
- 检查参考系数到实际两次前向的漂移，以及尾侧/入口侧贡献；所有预定 heads/layers 都报告，不选择最符合 T 优势的头作为总体。

原始生成并不保证已经保存 Q/K；如需额外 forward，应在同一实验合同中记清计算量，不能称这些状态已存在。状态只用于支持/反驳假说，不调整 band、gain 或表。

### 7.6 结果如何改变论文

- **I 正、远端优势且 near 减弱/反转，并看到预期证据竞争变化：**可增加“等位移形状收益依赖正确依赖的伸缩”这一机制结果。
- **I 近零或反向，且前提诊断满足：**反对当前机制预测；保留原有公共构造和性能结果，但不写该机制为已证实。
- **I 正，但内容相位/竞争变化不符：**只认定距离条件交互，不认定相干尾侧机制。
- **Native 相干条件或表示稳定性明显不成立：**说明这份条件模型解释不了这些行；不删除失败行，不把它转写成“方法必须调整”。

该实验不能单独识别 εn² 的因果份额，也不能证明 TailSpline 是满足原则的唯一曲线。其作用是为可迁移原则补上一个有区分力的完整模型预测，不是重新验证已成立的全部收益。

---

## 8. 可直接进入现稿的文本

### 8.1 English method motivation

> Extending a context does not dilate every dependency equally: distances within a record may remain local while the same record must be accessed from a more distant query. This motivates allocating dilation according to the distance responses a frozen model needs to preserve. Prior studies associate different RoPE frequencies with different dependency scales. If tail-adjacent transition channels support coherent long-range content matching, while local operations remain supported by the unchanged high-frequency band, reducing their mismatch to the interpolated reference can be more valuable than minimizing the entry jump. TailSpline implements this tail-facing preference with a smooth allocation of additional log-frequency span, preserving coordinate assignment and the two outer bands. The resulting spline is optimal for the stated boundary objective; its task value is established empirically. At equal total displacement, TailSpline stretches the entry half less and the tail half more than control C, making their comparison informative about where dilation is useful. Cross-model transfer supports this allocation preference as a reusable design choice, rather than a checkpoint-fitted correction.

建议在 prior studies 句后使用现有 `barbero2025round`、`wu2026datashapes` 引用。该段不声称已完成本报告第 7 节机制实验；不要把 if 改成无条件的实际模型结论。

### 8.2 中文“理论—构造—证据”的组织说明

> 先解释冻结扩窗面对的不是把全部依赖统一拉长：局部读和跨记录读对距离伸缩的需求不同，z 决定各频率通道如何承担这种变化。理论部分用有限相位响应说明，在 Native 内容匹配相干且竞争项没有抵消收益的条件下，接近伸长参照可以改善正确证据响应；入口变化是否值得接受，应由局部使用距离和内容 margin 判断，而不是由曲线平滑度判断。随后介绍 TailSpline 怎样以公共参数保留高频外带、增加过渡段伸缩并接入完全插值尾带，其闭式解负责把这个设计偏好落实为可部署规则。最后让 T/P 的工作区间与跨模型结果证明整条配置有效，让 T/C、BM/Uni 证明作用不止总位移；新增的远近证据比较若支持预测，再补上“这种配置优势随真实依赖伸缩而变化”的解释，不把尚未观测的 tail mechanism 写成现成事实。

### 8.3 Codex 定点执行

**现在即可完成，无需等待新 GPU：**

- `paper-2027/sections/04_mature.tex`：在 JT 之前补入压缩后的方法动机。避免重复整段现有高/低频外带定义。保留现有有限网格定理。
- `paper-2027/appendix/compact_b_constructions.tex`：在参照响应小节加入式 (1)–(4) 及短证明，并在 T/C 小节解释式 (7) 的“前半少、后半多”含义。清楚标注条件性分析。
- `paper-2027/appendix/compact_a_geometry.tex`：在既有 content-competition 之后补入式 (5) 的简短竞争/表示漂移边界；不要将它命名成另一种最优任务风险。
- `paper-2027/sections/04_experiments.tex`：T/C 段增加一句当前发现的设计意义，不添加未完成距离干预的结果。
- 保持标题、章节主线、现有数字、NCP 已成立内容和 Cosh 的多操作条件身份。

**研究准备而非默认运行：**为第 7 节创建一个输入 manifest 和对照合同。只允许这一项新比较，不自动启动全模型网格、A/B 边界曲线族或 checkpoint 校准。

**交付验收：**新命题有假设、证明和失败例；已有事实与假说分开；正文没有把 JT、有限残差或数学自检写成模型性能证明；更新后的 TeX 编译及 PDF 检查由 Codex 实际执行并记录，不沿用旧检查的通过状态。

---

## 9. 来源与核查

### 9.1 仓库与前序报告

- [R1] 当前构造与等位移身份：`paper-2027/appendix/compact_b_constructions.tex`，特别是 `sec:tailspline-dose-control`、`sec:allocation-reference-intervals`、`sec:turn-normalization`。
  https://github.com/misaya-yang/hybrid-rope/blob/c2abf72ae208d0e351e67062b8a47e3d63e43f4b/paper-2027/appendix/compact_b_constructions.tex
- [R2] 当前主实验与 T/C：`paper-2027/sections/04_experiments.tex`。
  https://github.com/misaya-yang/hybrid-rope/blob/c2abf72ae208d0e351e67062b8a47e3d63e43f4b/paper-2027/sections/04_experiments.tex
- [R3] Kanana 当前 Git 结果截点：`experiments/kanana_yarn_tailspline_64k_20260918/RESULT.md`。
  https://github.com/misaya-yang/hybrid-rope/blob/c2abf72ae208d0e351e67062b8a47e3d63e43f4b/experiments/kanana_yarn_tailspline_64k_20260918/RESULT.md
- [R4] 前序用户文件：`ROPE_THEORY_SYNTHESIS_FOR_CODEX_20260917.md`，本次复核相关坐标、残差和阈值章节。已有 α* 和尾端 residual ratio 不再列作本次新发现。
- [U1] 用户本轮补充的 Kanana P=70.33 及 128K QA 运行状态。与 [R3] 的落盘截点区别见第 6 节。

### 9.2 外部原始研究

- [W1] Barbero et al. *Round and Round We Go! What makes Rotary Positional Encodings useful?* ICLR 2025，arXiv v3。用作不同频率使用方式的独立动机，不当作所有模型的固定分工定理。
  https://arxiv.org/html/2410.06205v3
- [W2] Wu, Liu, Jadbabaie. *How Data Shapes RoPE Frequency Usage: From Positional Scale Matching to Length Generalization.* arXiv:2607.07678v1，2026-07-08。用作依赖尺度和伸缩条件的相关理论；本报告不声称首次提出尺度匹配。
  https://arxiv.org/html/2607.07678v1
- [W3] Tian et al. *MrRoPE: Mixed-radix Rotary Position Embedding.* arXiv:2601.22181v1。用于核对前驱构造定位，不用其单一几何上界代替本研究的内容响应条件。
  https://arxiv.org/html/2601.22181v1

### 9.3 本次数学验证（不是模型证据）

配套 `check_finite_phase_claims.py` / `finite_phase_checks.json` 实际完成：n=1…128 的 T/C 有理数身份与分配方向检查；非零正交相位下有限下界的随机数学检查；相干条件失效的明确反例；第 4.3 节公共网格数值复算。

它们检查证明的代数实现，不验证真实 Native 相位条件、任务收益或机制。构造器、模型权重及原运行表均未更改。

**最终决策：今天落笔的是“依赖尺度不同，因此伸缩应有所侧重”的方法动机，以及有条件的有限响应说明；下一步只检验同一 T/C 的证据远近交互。不要以此重写整篇论文，也不要把解释升级为已经完成的机制识别。**
