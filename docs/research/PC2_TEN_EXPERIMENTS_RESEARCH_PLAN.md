# PC2：十个不同方向的高价值实验与统一研究计划

**版本：v1；对应项目材料版本 2026-09-10。**  
**对象：作者与 Codex。**  
**本轮性质：从 PC2 的实际计算对象出发设计实验，不依据某个历史失败给整个方向设禁区。本文没有新增真实模型实验结果。**

## 0. 核心决策

PC2 要解决的不是“让协方差近似看起来更漂亮”，而是：

> **在保留模型实际位置编码和原始 K/V 的条件下，以更低代价完成有用的块选择与读取。**

十项实验分别改变：**数值分辨率、GQA 选择目标、时间复用、分布表示、块内多模态、随机估计、资源分配、value 读取、位置对称性、选择器学习**。它们不是十个 rank/horizon 变体，也不是十篇已经成立的新论文。

**首跑 E02**：原 exact 已经产生所需分数，只替换 GQA 的集合决策，最快获得一个具有方法含义的答案。**主要效率候选 E01**：用有界近似决定哪些块需要精算，保护 exact 的选择语义。**最直接的位置贡献候选 E09**：检验共享压缩子空间是否需要随块的位置坐标共同变换。

其他七项同样具有明确的可部署版本，不是失败后随口补出的解释。每项下方列出六种没有入选的替代构造及其具体缺陷；这些是设计审查结论，不代表运行了六组实验。

### 0.1 成功概率的排序含义

排序是**在当前有限资源下得到可复用质量／成本改善的主观优先判断**，不是数学推出的概率，也不是 ICLR 录用率。依据是：收益链有多少未知条件、是否使用当前真实 Q、能否约束错误、实施需要多少新接口。不给没有校准依据的百分数。

| 成功优先序 | 实验 | 不同的研究角度 | 理论支持最直接的部分 | 最大剩余未知 | 独立成文新意 |
|---:|---|---|---|---|---|
| 1 | E02 非线性 GQA 集合选择 | 决策目标 | 每次接受的交换严格降低既定非线性风险 | 风险下降是否改善实际回答 | 中；须超出标准公平分配解释 |
| 2 | E01 有界低精度索引＋精确补算 | 数值分辨率／可验证决策 | 区间分离时保持 baseline 的选择 | 精算比例与 kernel 成本 | 中；证书与量化本身不新 |
| 3 | E04 投影后的经验指数分布 | 表示／高阶响应 | 子空间内保留全部经验 log-partition，而非只二阶 | 有用子空间是否足够小 | 中高；须正面对比 COBS/Loki |
| 4 | E08 被遗漏块的 value 响应补偿 | 读取目标 | 可准确写出补偿后的剩余误差 | 块内 value 是否可用低成本代表 | 中；不能宣称首次补 residual |
| 5 | E03 时间上的有界响应复用 | 动态计算 | 当前 Q 变化可给出合法更新区间 | 真实 query 漂移和状态开销 | 中；须区别 ReTopK/GVR |
| 6 | E07 固定总预算的跨 KV-group 分配 | 资源分配 | 对指定可加目标，预算分配有明确解 | kernel 是否兑现 ragged 预算、任务是否受益 | 较低；主要是高价值目标诊断 |
| 7 | E05 两成分条件 PC2 | 多模态／局部展开 | 成分间分离不再被压成单一二阶项 | 成分内余项与描述符成本 | 中；不是重新命名聚类 |
| 8 | E10 cutoff-regret 小型残差选择器 | 学习／信息充分性 | 监督对象直接对应当前可见 Q 的选择差值 | 描述符可识别性和跨材料泛化 | 中；DSA/SAAP 是强近邻 |
| 9 | E09 块局部坐标下的共享压缩基 | 位置群作用／共享表示 | 原点等变性条件与不变的反例边界 | 对齐后能否实际降低共享表示成本 | 相对最高，但尚无模型收益证据 |
| 10 | E06 PC2 控制变量的残差抽样 | 统计估计 | partition 估计无偏且无 Gaussian 前提 | 指数尾部使所需样本数过大 | 中；控制变量本身不是贡献 |

E09 排得较后，不是因为位置不重要，而是它还要求“块间共享结构可以被位置对齐”成立。E06 排得最后，是因为指数尾部可能让低样本估计失去成本优势。E01 的选择保持保证，不等于速度有保证。上述排序不把这三种不同保证混成同一种“成功率”。

---

# 1. PC2 的理论到底是什么

## 1.1 只使用材料支持的定义

当前材料给出的精确目标是 [U1, Q04]：

\[
F_b(a)=\log Z_{b,\mathrm{CIS}}+a^\top\mu_b
+\log\mathbb E_{w_b}\exp\{a^\top(k-\mu_b)\},\qquad a=q/\sqrt d.
\]

pair-PC2 使用

\[
\widehat F_b^{\mathrm{PC2}}(a)=\log Z_{b,\mathrm{CIS}}+a^\top\mu_b
+\frac12a^\top D_ba,
\qquad D_b=\operatorname{pairdiag}(\Sigma_b).
\]

其中 k 是实际 selector 合同中的 key，不能把 pre-RoPE 与 post-RoPE 混用。最终 reader 的 Q/K/V、RoPE、幅度和物理位置保持原样，除非某项实验明确声明改变的正是 reader。

附件没有提供 CIS 的完整源码和缩写定义，本文不补造其含义。若现有精确分数可写为

\[
F_b(a)=\log\sum_{j\in b}\exp\{\beta_{bj}+a^\top k_{bj}\},
\]

则取 \(c_j=e^{\beta_j}, Z_{\mathrm{CIS}}=\sum c_j,w_j=c_j/Z_{\mathrm{CIS}}\)。以下矩公式在固定这组权重时成立。若 CIS 权重随当前 query 改变，**逐 query 的公式仍可成立，但不能把这些 query-dependent 矩当作可一次构建、永久复用的缓存**。E03 必须额外处理权重变化；其他实验按现有实际 exact scorer 执行，不静默改成均匀权重。

### selector mass 与 reader mass 不能因都叫 exact 就混成一个对象

记 B0 的精确评分质量为 \(M^{sel}\)，实际 reader 的 softmax 质量为 \(M^{read}\)。源码若表明二者相同，后文可共用 M；若 CIS 只是 selector 的额外权重，二者并不自动相同。

E01/E03/E04/E05/E06/E09/E10以 B0 的精确评分合同为拟合对象。E02关于读取风险、E08关于输出恢复的等式，必须使用 \(M^{read}\)。E07优化的则是 B0 已声明的可加分数。两类 mass 的核对只是复用现有源码和一小份真实 logits 的对应关系，不另建漫长审计。

若 \(M^{sel}\ne M^{read}\)，E02增加一个“实际reader mass＋原additive规则”的内部对照，隔离改目标与改非线性两件事；主baseline仍是B0。E08始终保留B0集合，但用真实reader mass计算tail权重，所增加的计算如实计价。不能把selector的CIS权重静默植入原reader再宣称恢复了原输出。

实际 attention scale 从B0继承；已缩放的Q不再次除 \(\sqrt d\)，d也不能替换成rotary维数。

## 1.2 精确的误差分解

令 \(x=a^\top(k-\mu_b)\)，\(\psi(t)=\log\mathbb E_w e^{tx}\)。有限 key 集上，实数 t 附近导数存在：

\[
\psi'(0)=0,\quad\psi''(0)=a^\top\Sigma_ba,
\quad\psi'''(t)=\mathbb E_{w_t}[(x-\mathbb E_{w_t}x)^3].
\]

Taylor 的积分余项给出

\[
\boxed{F_b-\widehat F_b^{\mathrm{PC2}}
=\tfrac12a^\top(\Sigma_b-D_b)a+
\tfrac12\int_0^1(1-t)^2\psi'''(t)\,dt.}
\]

这是本轮使用的数学中心：**跨 pair 的二阶项与沿真实查询方向的高阶响应，是不同误差。** “完整协方差”只消去第一项。Gaussian 分布下的二阶精确性，不能自动赋给有限经验 key 集。

若 \(|x|\le A\)，一个直接但常常很松的余项界是 \(|R_3|\le4A^3/3\)：因为 tilted 分布下 \(|x-E_tx|\le2A\)，将三阶中心矩的绝对值界积分即可。它说明适用范围，**不拿这个松界当实际 cutoff 证书**。

## 1.3 排序看的是差，而不是单块误差

记 \(e_b=\widehat F_b-F_b\)，则

\[
\widehat F_b-\widehat F_c=(F_b-F_c)+(e_b-e_c).
\]

对单 head，若 \(F_b-F_c>|e_b|+|e_c|\)，次序保持。对 GQA，实际排序通常还包括每个 query head 自己的分母，因此不能只在一个 head 的 logmass 排名上宣布“选择已正确”。

若同一 KV group 内 G 个 Q heads 共用集合 S，令

\[
p_{gb}=M_{gb}/Z_g,\qquad
Z_g=Z_{g,\mathrm{mandatory}}+\sum_{b\in\mathrm{remote}}M_{gb}.
\]

固定 baseline 沿用现有聚合；在采用可加 mass 聚合的配置中，其分数是 \(G_b=\sum_gp_{gb}\)。**mandatory 包含去重后的 local/sink，不做 remote-only 归一化。** [U2, §5.3; W1]

## 1.4 选择正确与回答正确之间，仍然有一个需要实验承担的接口

固定一次实际 Q/K/V，令保留质量为 \(r_g=1-\tau_g\)，则

\[
o_g-o_{g,S}=\tau_g(o_{g,D}-o_{g,S}).
\]

当 value 范数有界时可以上界输出差；但向量抵消、head 的有效贡献、后续网络与答案决策都没有由此确定。COBS 的 mass 目标本来就包含 value-agnostic 与 GQA 线性化假设；本计划把这些假设变成 E02/E07/E08 的可改变对象，而不是认为 exact mass 就是最优答案策略。[W1,W10]

## 1.5 成本必须满足的算术

对于以更低成本逼近 exact 的候选，实际有用的必要条件是

\[
T_{\mathrm{descriptor}}/Q_{\mathrm{reuse}}
+T_{\mathrm{approx}}+fT_{\mathrm{exact\_score}}
+T_{\mathrm{routing\_extra}}
<T_{\mathrm{exact\_score}},
\]

这里 f 是需要精算的真实工作比例，不是只数最后保留的块；还要考虑重算粒度和 kernel 的非线性时间。两边共同 reader 的成本不凭空消失，额外 reader 工作另加。

当近似 kernel 本身已经慢于 exact scorer，继续减少数学 FLOPs 没有直接用途；当构建成本大，则必须报告单次使用与共享前缀复用的不同场景。**PC2 保留原始 K/V，是稀疏读取研究，不把 descriptor 节约写成删除全部 KV。**

---

# 2. 一个固定 baseline，一套数据，一次接入

## 2.1 固定 B0

**B0 = 当前已经可运行的 exact 块评分＋原 CIS＋原 GQA 聚合＋原名额＋原 reader。**

从它的 manifest 取得模型、真实 revision、block size、protected 集合、remote 名额、原生长度、tokenizer、生成设置及数值后端。本文不猜当前 checkpoint 路径，也不要求下载一个新模型。

所有十项的主比较都是 `Ei vs B0`。E02 改集合优化，E07 改组间名额，E08 改 tail 的读取处理，须在自己的行里标明；其他候选不顺便改变这些项。E02/E07 也不是换了一个更弱 baseline，而是在**同一 B0 之上测一个明确的目标／预算干预**。

文中的内部数学对照只承担归因，不轮流成为主 baseline。最后进入论文级评测时，再加入固定版本的完整强方法；不能用本地 estimator port 冒称复现其完整系统。

## 2.2 共用实验材料

- **校准材料**：最多 24 个独立来源文档，仅供 E04/E09/E10 的共享基或小选择器拟合；不含确认集。没有校准需求的候选不因此获得额外监督。
- **首轮共同 DEV**：48 个独立材料，四类各 12 个：多 key／多 query 检索、重复 occurrence／先后查询、HotpotQA 类自然多跳、另一项已有自然长文 QA。沿用项目现有生成器、完整原文和原 scorer。
- 48 个材料覆盖两种已支持长度，优先约 8K 与 16K；若 B0 窗口或真实数据不支持，沿用 B0 的两个合法长度，而非增加 RoPE 扩展。自然原文不裁掉证据，不为凑长度塞大量重复噪声。
- 重复背景与自然背景必须一开始同时出现；但同一个材料做两种背景变体仍是**一个聚类单位**，不变成两个独立样本。
- **确认集**：先预留新的独立来源。仅对有质量／成本信号的少数候选使用；样本量依据 DEV 的成对方差决定，不承诺 48 或 128 个例子足以确认 1–3pp。

共享前缀的构建可复用，但每种方法从问题第一个 token 起用自己的 live sparse 路径，各自形成后续历史。不能让 dense 先处理完整问题再只压缩答案 decode，也不能把 B0 的未来 Q 注入候选主运行。

## 2.3 每项最少输出同一张小表

`task score / paired difference / uncertainty / summary bytes / raw K reads / raw V reads / build / scoring / routing / reader / continuation wall time`。

干净计时不含全量observer、shadow精算或保存激活的传输；诊断另记，不混入速度主表。

主指标沿用原任务定义：RULER 答案项 recall 不改称整句 exact；自然 QA 用原 F1/EM；完整输出、截断与 EOS 均保留。Full 不是答题正确性的绝对上界，不先按候选正确与否筛题。

采样实验 E06 的随机种子是算法随机性，不叫训练种子。统计按来源文档配对／聚类，不把 head-query 数量作为独立问答样本量。

## 2.4 本计划的实用成功标准

两种不同成功均有方法价值：

**Q 型**：同实际稀疏资源下，独立新材料中出现约 3pp 以上有意义的生成收益，且系统代价没有抵消收益；若有额外成本，给出质量—成本曲线，而非只报质量。

**E 型**：保持 B0 质量，在同协议下 continuation 时间至少降低约 15%，并报告构建计入后单次使用和复用场景。严格保持全部 keep-set 的实验，可直接用其选择等价性解释质量保持；非等价方法不能以小样本“没显著下降”冒充非劣。

这些是工作目标，不是定理或审稿人的固定线。质量保持的非劣界可先取 1pp；确认样本不足则记录效应区间，不换指标制造达标。即使没有过该实用阈值，实验仍可回答机制问题，但**机制被识别不等于已经交付一个有竞争力方法**。

---

# 3. 同领域论文给本计划的具体参照

以下只使用核对过的一手论文／官方来源。新颖性判断是针对本方案的比较，不是穷尽性查新。

| 工作 | 已经解决或实验过什么 | 我们吸收的研究方法 | 不能重复包装的内容 |
|---|---|---|---|
| COBS [W1] | 从块 mass 推导矩摘要，提供 exact 参照，研究 subspace/rank/量化；其训练与位置协议有限制 | 将可控算子误差与真实生成、成本分开；直接检验被使用的 relaxation | 二阶项、query 子空间、低秩、FP4 不新 |
| Quest [W2] | query-aware 页选择与 cached key 范围 | 用真实 key 的便宜范围信息服务读取成本 | min/max 页描述符、query-aware 筛选不新 |
| Prism [W3] | RoPE pooling 的频谱影响，面向 block-sparse attention | 位置主张必须有纯内容／相同资源的对照 | 相位相消和“频率重要”不新 |
| Loki [W4] | 用低维 keys 近似 sparse ranking | 对投影误差单独审计，不把低维表示自动当足够 | 低秩 key 索引本身不新 |
| SparQ [W5] | query-aware 稀疏读取与遗漏质量的 value 均值重分配 | 把真实输出而非单个 score 当终点 | 补回 tail value 或 residual 本身不新 |
| AB-Sparse [W6] | 不同 head 的 block 粒度与 kernel 联合设计 | 数学预算必须兑现为硬件读量／时间 | adaptive block size 不新；E05 保持物理块不变 |
| ReTopK / GVR [W7,W8] | 分别研究 query/support 复用和 exact top-k 的时间相关优化 | 复用必须验证；区分少算分数与更快排已有分数 | 首次时间复用、首次 exact top-k 不新 |
| Certified Top-k theory [W10] | 截断质量与输出界、边界信息及认证 | 不把标准界当原创，实验测其是否可用 | 首次 attention 证书不新 |
| Uncertainty-gated selection [W11] | 在不确定的 cutoff 扩大保留集合 | 不确定性值得作为分配计算的量 | E01 必须固定最终 reader 名额，只多做评分，不能靠双倍读取取胜 |
| DSA / SAAP [W13,W14] | main-attention 监督的 indexer／学习式非对称索引 | 真查询监督可学，但必须控制校准与在线偏移 | 首次学 selector、首次 KL 蒸馏不新 |
| Ada-KV [W12] | head 级自适应 eviction 预算 | 区分固定预算是否用对与 scorer 是否估对 | 首次按 head 分预算不新；E07 是原 KV 稀疏读取合同 |
| Efficient Attention via Control Variates [W15] | 用控制变量理解和改进近似 attention | 无偏估计、方差和实际成本必须分开 | 控制变量数学不是 E06 原创 |

HPC-Ops Top-K 的新预印本 [W9] 进一步提醒：**对已经 materialize 的 score row 做 exact top-k，和避免读取原 K 来形成那些 scores，不是同一个问题。**E01/E03 若只加速排序而没有省掉评分，不能声称完成了后者。

---

# 4. 十个实验的完整规格

## E01｜有界低精度索引，只对不确定块读取原 K

**研究问题。** 真的需要全精度读取所有 K 才能得到 B0 的块集合吗，还是多数块只需便宜的上下界就能排除？

### 入选构造

初版构建**有误差包络的 INT8 原 scoring-space key 索引**，原始 K/V 不删除。若B0使用post-RoPE K，就原样使用；不自行撤去或增加旋转。优先使用现有量化 score 路径；否则仅实现一个 tiled scoring kernel，不重写 reader。每个小组记录量化 scale 和保守重构误差。先用 INT8 而非 INT4，是为了减少边界处的不确定宽度；不是先假设最激进位宽一定更好。

令 \(|k_{ji}-\tilde k_{ji}|\le e_{bi}\)。对当前实际 a：

\[
|a^Tk_j-a^T\tilde k_j|\le\delta_b(a)=\sum_i|a_i|e_{bi}.
\]

log-sum-exp 的无穷范数 Lipschitz 性质给出

\[
\tilde F_b-\delta_b\le F_b\le\tilde F_b+\delta_b.
\]

PC2 可用于精算的优先排序；**PC2 本身不是天然的 lower/upper bound**。若已有 PC2 排序增加成本而无收益，关掉该优先项不改变证书。

### GQA 的证书必须真的覆盖分母

设 \(L_{gb}\le F_{gb}\le U_{gb}\)，mandatory mass 已知。则

\[
\underline p_{gb}=\frac{e^{L_{gb}}}{Z_{g,mand}+e^{L_{gb}}+\sum_{c\ne b}e^{U_{gc}}},
\quad
\overline p_{gb}=\frac{e^{U_{gb}}}{Z_{g,mand}+e^{U_{gb}}+\sum_{c\ne b}e^{L_{gc}}}.
\]

对 g 求和得到 \(\underline G_b,\overline G_b\)。若大小为 m 的候选集合 S 满足

\[
\min_{b\in S}\underline G_b>\max_{c\notin S}\overline G_c,
\]

便已得到 B0 的真实集合。否则，对仍可能跨边界的块分批读取原 K、算 exact、缩小区间。**不能只精算近似 top-m 内部而忽略外部可能的高分块。**必要时回退到全部 exact，最终 m 不变。

证书是实数运算结论。实际量化、乘积、求和与 exp/log 的误差都要并入包络；未实现受控浮点包络时，报告“实数证书＋数值 keep-set 核验”，不声称严格 bitwise 证明。tie 使用 B0 原规则。

### 最小实验

B0 结果复用。先在共同 DEV 的真实 current Q 上记录区间、精算比例 f、每步 keep-set；随后完整生成。只做 INT8 初版。记录平均／p95 的精算块数，以及算法和 kernel 两种原因导致的耗时。

低精度必须在 tile 内完成 score，不得先把全部索引反量化成一份完整 BF16 K 再遍历一遍，并声称节省了 KV 带宽。

### 有效、无效、受损条件

- **有效**：边界 gap 足以超过多数误差包络；INT8 扫描与少量 exact 合计便宜。
- **质量应相同但无加速**：大量块区间重叠，或 kernel 开销过大。
- **出现质量差异**：先检查证书／数值／reader 是否违反等价条件；这不是本法计划中的质量 trade-off。

### 六种排除的替代设计

| 未选设计 | 排除原因 |
|---|---|
| 固定取 approximate top-2m 再精排 | 没有覆盖集合外的潜在真高分块 |
| 用 pair-PC2 ± 一个经验常数作证书 | 常数未覆盖跨 pair、高阶和 OOD Q 误差 |
| 一开始用 INT4 追最大压缩率 | 误差加宽可能令全部候选都需 BF16 补算 |
| 只认证每个 head 的独立 top-m | 不认证实际共享 GQA 聚合与分母 |
| 难 query 直接把最终 m 翻倍 | 改了 reader 预算，混入另一种改进 |
| 把全量反量化临时张量当免费 | 隐藏带宽和 workspace，理论压缩无实际省时 |

### 失败后得到什么

用 f 与 break-even 公式判断：是**信息分辨率不足**还是**实现代价过高**。若原始索引误差小但 GQA 区间因分母耦合很宽，得到明确的“归一化联合界过保守”结论，下一步应改认证方式而非加协方差 rank。若原始误差本身就跨越大多数真实 margin，则该位宽下的保守筛选没有余量；不能宣称所有近似排序不可能。

### 成功后深化

只深化一次：同一配置拓展第二合法长度／第二模型，加入论文级 Quest、完整 COBS 与当前 exact-top-k 系统对照。再研究位宽的渐进分配和 kernel 融合；先有 INT8 的真实收益，不先做 INT4/FP4 大工程。

**可能贡献：**在保持 position-sensitive、normalized GQA 选择的条件下，将精度花在实际决策边界。量化和认证单独不构成 novelty。

---

## E02｜恢复 GQA 目标中的非线性，避免共享集合饿死少数 head

**研究问题。** B0 的质量损失是否有一部分并非块 mass 估错，而是精确 mass 被一个过于线性的共享集合目标使用了？

### 入选构造

沿用 B0 的 key dot-products、相同 m、相同 reader。风险中的概率使用真实 reader mass；它与 CIS score mass 的区别按§1.1处理。对集合 S 定义

\[
r_g(S)=p_{g,mand}+\sum_{b\in S}p_{gb},\qquad
\mathcal R(S)=\sum_g\frac{1-r_g(S)}{r_g(S)}.
\]

这是恢复既有 value-agnostic 推导中的非线性惩罚，不新增“它就是答案 loss”的假设。[W1]

从 B0 的集合开始，枚举一入一出的交换。每轮选真正降低 \(\mathcal R\) 最大的交换，最多接受两轮；若没有正改进，保留 B0。候选外部先取线性分数最高的 m 个，加上每个 head 自己最强的两个未选块，去重后作有限候选池。该池限制只影响能找到多少改进，不破坏“接受的每次交换均降低 R”的性质。

\[
\Delta\mathcal R_{i\to j}
=\sum_g\left[\frac1{r_g-p_{gi}+p_{gj}}-\frac1{r_g}\right].
\]

两轮交换不是全局最优算法；没有如此主张。

### 明确事前预测

- G=1 时与 mass top-m 等价，应无改进空间。
- 当各 head 的 r 都接近 1，线性化差异小，应近乎无效。
- 当同组 head 有互补需求，某个 r 远低于其他 head，非线性选择更可能改变集合。
- 如果被救助的 head 对任务没有有用贡献，R 可以下降而答案不升，甚至下降。

### 最小实验

直接在现有 score tensor 上加小函数，完成共同 DEV 的生成；不新建 descriptor、不训练、不重新提取所有历史激活。主表给出质量、两轮交换成本、各 head 的 retained-mass 分布、R 的变化。对固定少量输入同时测同状态下的真实 attention output 差，区分“bound 改善”与“实际读取改善”。

### 六种排除的替代设计

| 未选设计 | 排除原因 |
|---|---|
| 各 head 独立 top-m 再取并集 | 通常偷偷增加实际读取块数 |
| 按 remote-only softmax 做 head 平均 | 把只需要 local 的 head 强行当作有同等远程需求 |
| 永远优先最弱 head 的 max-min 规则 | 可能牺牲其他有用 head，且不是上述可审计目标 |
| 扫多个 softmax temperature | 同时改变分布和目标，归因不清 |
| 直接做全组合最优搜索 | 在线代价指数级，不符合实验 ROI |
| 用 head norm 当任务重要性 | norm 不是已识别的输出／答案贡献 |

### 失败后得到什么

若 R 不变，说明这组数据的共享需求没有暴露线性化缺口；若 R 下降但真实 output 不改善，说明 value-agnostic/head 系数的 relaxation 是当前剩余缺口；若 output 改善但回答不改善，则在本实验范围内该层面 fidelity 不是任务瓶颈。不能把三者统一写成“GQA 没用”。

### 成功后深化

先在第二模型的不同 GQA group size 复现，验证 G=1 null 和异质 retained mass 的效应交互。然后才把低成本 scorer 接到同一非线性目标上；不在第一轮同时更换 scorer。强论文需要“为什么标准 additive rule 不足＋可用的低开销替代＋真实任务”，不是只有 R 降低。

---

## E03｜对真实当前 Q 做有界时间复用，不重新预测未来 Q

**研究问题。** decode 的多数旧块，是否可以从最近一次精确响应安全更新，而不用每步重读原 K？

### 入选构造

对每个缓存完整块，保存最近一次精算的实际 query \(a_0\)、\(F_b(a_0)\)、tilted mean

\[
\mu_{b,0}=\nabla F_b(a_0)=\mathbb E_{p_{a_0}}k.
\]

同组多个 heads 各有自己的 F 与 tilted mean；其存储与更新成本必须计入。缓存原始 key 坐标范围。当前 query 已真实产生，令 \(\delta=a-a_0\)，不使用未来问题或 synthetic Q。

固定权重时，由 convexity 与 Hoeffding 引理，若 \(W_b(\delta)\) 是 \(\delta^Tk_j\) 的合法 range 上界：

\[
F_b(a_0)+\delta^T\mu_{b,0}
\le F_b(a)
\le F_b(a_0)+\delta^T\mu_{b,0}+W_b(\delta)^2/8.
\]

取坐标范围可得到
\(W_b(\delta)\le\sum_i|\delta_i|(k_{b,i}^{max}-k_{b,i}^{min})\)。保存半径还可给另一界，使用二者较小值。用 E01 的同一 GQA 区间决策，只有不确定块刷新 exact 及梯度。新块、未完成块和 local/sink 按原路径处理。

若 CIS logweights 改变，须对 \(\Delta\beta_j\) 的 min/max 加到区间两端；无法便宜获得变化界时，这些步走 exact，不声称时间复用定理适用。

### 位置在这里怎样进入

\[
a_t-a_{t-1}=R(p_t)(u_t-u_{t-1})+[R(p_t)-R(p_{t-1})]u_{t-1}.
\]

可以分别记录内容变化和已知位置变化，但**不能因为 RoPE 变化可计算，就把它对 block response 的影响当作零**。q 的 cosine 相似度也不替代上述响应界。

### 最小实验

B0 不变；E03 第一轮 exact 初始化状态，之后在线刷新。报告按 question token、answer token 划分的 exact 刷新率、状态 bytes、每次梯度刷新成本和实际时间；包括答案很短时的总成本，不只挑长 decode 摊销。

### 六种排除的替代设计

| 未选设计 | 排除原因 |
|---|---|
| 每隔固定16步刷新一次 | 内容突变可发生在任何一步 |
| q cosine 高就直接复用集合 | 小范数方向变化仍可能跨关键 margin |
| 上一步 top-k 与 recent 的并集即可 | 可能漏掉历史候选之外的新需求，且预算易增加 |
| 只按 token 位置差旋转旧分数 | score 不是一个可直接旋转的二维向量 |
| 用 prefill 平均 Q 作为永远的 anchor | 不能约束当前真实请求变化 |
| 只复用 top-k 排序，不区分 score 计算 | 可能没有省掉本项目最贵的 K 读取 |

### 失败后得到什么

分别判断：真实 delta 太大、区间太松、梯度状态太贵、decode 太短，还是实现没有减少实际 K fetch。若只有 query 相似却无响应可复用，否定的是“相似度足够”的实现，不是所有 temporal 结构。若 interval 上界松，不能据此断言真实 scores 不稳定。

### 成功后深化

跨不同 query 长度／生成阶段确认刷新规律，与 ReTopK 的近邻复用、GVR 的已有 score top-k 加速严格区分 [W7,W8]。更强结果是**减少 score 形成所需的全 K 扫描，且保持 B0 集合**，而不是又一个 temporal cache heuristic。

---

## E04｜存投影后的经验分布，绕开“二阶足够”的假设

**研究问题。** 与其存一个低维 Gaussian 近似，能否在相同数量级的 descriptor 内保留低维空间中的整个有限 key 分布？

### 入选构造

从独立校准材料的真实 Q 确定共享正交基 \(U\in\mathbb R^{d\times s}\)，第一版 \(s=\min(32,\lfloor d/2\rfloor)\)。构建每块 \(\mu_b\) 与每个原 token 的

\[
y_{bj}=U^T(k_{bj}-\mu_b).
\]

读时只计算

\[
\widehat F_b=\log Z_{b,CIS}+a^T\mu_b+
\log\sum_jw_{bj}\exp\{(U^Ta)^Ty_{bj}\}.
\]

它没有删除原 K/V，也没有改 reader。描述符约为 \(d+Bs\) 个数外加权重和 scale，取代 \(Bd\) 的原 K 评分读取。构建和指数求值仍有成本，不说它免费。

### 理论与区别

在 \(a\in\operatorname{span}(U)\) 时，这个目标精确；对一般 a，令
\(r_j=(I-UU^T)(k_j-\mu)\)，有

\[
|F_b-\widehat F_b|\le
\|(I-UU^T)a\|\max_j\|r_j\|.
\]

这只剩投影误差；**子空间内的三阶、四阶、稀有高匹配全部仍在经验 log-sum-exp 中**。COBS 是子空间二阶统计，Loki 已经使用低维 key 索引；我们不能宣称首次低秩检索。[W1,W4]

### 最小实验

主比较 Ei vs B0。内部归因增加一个无需新基底的 `same-U second-cumulant`：相同 U，比较经验指数响应与投影协方差响应。两者 descriptor bytes 不一定相同，额外给出匹配 descriptor 成本的结果，不能以“同 s”冒充同成本。

在48 DEV 的 live query上同时观察真实 cutoff 与投影 residual；自然 QA 和检索一起跑。若 s=32 的构造信息不够，第一轮得出的就是该预算下的结论，不立刻扫 8/16/32/64/96。

### 六种排除的替代设计

| 未选设计 | 排除原因 |
|---|---|
| 直接增加 PC2 rank | 即使协方差恢复，仍保留高阶截断 |
| 用随机一小撮原 K 代替全部 projected K | 将投影问题混入尾部抽样漏检 |
| 只存 projected mean 与方差 | 回到了需要检验的二阶假设 |
| 对每个块现算一个 query-specific PCA | 在线读回原 K，可能消耗全部省下的成本 |
| 所有模型固定极低 s=4/8 | 没有实际 query 方向覆盖依据，易人为制造失败 |
| 只看解释方差百分比 | 不直接约束 cutoff 或真实分数差 |

### 失败后得到什么

`投影后 exact` 与 `同 U 二阶` 的差异区分**高阶不足**；`投影后 exact` 与 B0 的差异区分**子空间不足**。若两个误差都小而成本无优势，结论是表示可行但压缩比／算子代价不合算；若投影 exact 自身错排，不应继续给同一窄子空间添加高阶项。

### 成功后深化

只扩一项：跨位置／跨域固定 U 的泛化，然后与匹配 bytes 的完整 COBS、Loki 类索引对照。若位置迁移破坏 U，再让 E09 回答是否需要位置共变基，而不是事后随意给每个位置拟合一个 U。

---

## E05｜块内两个条件成分，比全块一个二阶展开更合适吗

**研究问题。** 阻塞 PC2 的是“需要更高阶”，还是“本来就不应围绕一个混合中心展开”？

### 入选构造

物理 KV 块及最终 m 完全不变。每个完整块在 actual key 空间使用确定性的两中心分配：从固定起点做两次 farthest 选择得到中心，按距离分成两组；空组退化成单组。不是按相位或目标标签分组。

每组保存条件权重 \(W_r\)、\(\mu_r\)、pair 协方差 \(D_r\)，评分

\[
\widehat F_b=\log Z_{CIS}+
\operatorname{LSE}_{r=1,2}\{\log W_r+a^T\mu_r+\tfrac12a^TD_ra\}.
\]

两成分间的均值分离以 log-sum-exp 保留，不再全部压成整体 covariance。若每组成分内 key 恒定，方法对任意 query 精确。一般情况下，成分内跨 pair 与高阶误差仍存在；**不会宣称 mixture 二阶在无限 query norm 下拥有有限-support 的精确渐近**。

### 最小实验

同48 DEV生成；逐 query 分解整体真实 F、两组成分真实 logmass、两组成分 PC2，明确改善来自哪一项。内部对照是同 descriptor bytes 的单成分 covariance 压缩，以及同组大小的连续分组。不是比较2个摘要对1个摘要然后宣称分组机制成立。

### 六种排除的替代设计

| 未选设计 | 排除原因 |
|---|---|
| 三阶／四阶完整张量 | 存储与构建高，局部展开依然有范围问题 |
| 抽一个最大半径 singleton | 半径不决定实际 query 的分布成分 |
| 随机把块切两组 | 不能系统降低条件内的响应离散 |
| 只按时间或 phase 分组 | 可能把同一内容簇拆散，不能预设相位是主因 |
| 在线按当前 query 对所有原 K 重聚类 | 失去 cached summary 的成本目的 |
| 任意增加到8/16成分 | 先消耗存储解释所有误差，无法检验两成分机制 |

### 失败后得到什么

若条件内真实响应范围不减，分组没有捕获可压缩模式；若范围减而组内 PC2 仍错，瓶颈是条件内高阶／跨 pair；若评分明显改善而生成无益，则多中心 fidelity 不足以改变当前任务。若连续分组同样好，保留“多中心有效”，不声称内容分簇或相位是独特机制。

### 成功后深化

研究在固定 descriptor bytes 下按可观测条件响应复杂度分配成分数，但先只跨 block size 与第二模型验证2成分。AB-Sparse 改物理 block granularity，E05 改固定物理块内部描述；二者要分别计价。[W6]

---

## E06｜用 PC2 作控制变量，直接估计被它丢掉的真实指数残差

**研究问题。** 能否利用已有 pair 矩，把少量真实 K 读取集中用于估计遗漏项，而不是猜一个新的确定性 tail 修正？

### 入选构造与无偏性

令 \(x_j=a^T(k_j-\mu)=\sum_p x_{jp}\)，p 遍历实际二维 pair／其余固定分组。取

\[
g_j=1+x_j+\tfrac12\sum_p x_{jp}^2,
\quad\mathbb E_w g=1+\tfrac12a^TD a.
\]

从已知 \(w\) 抽 m 个 token，初版 m=8：

\[
\widehat A=1+\tfrac12a^TD a+
\frac1m\sum_{r=1}^m(e^{x_{J_r}}-g_{J_r}),
\quad
\widehat M=Z_{CIS}e^{a^T\mu}\widehat A.
\]

\(E\widehat M=M\)，且
\(\operatorname{Var}(\widehat A)=\operatorname{Var}_w(e^x-g)/m\)。这是对**partition** 的无偏性，不是对 logmass、归一化 GQA、top-k 或答案的无偏性。这里使用的是有已知期望的多项式控制变量，不把 \(e^{v/2}\) 误当作该多项式的精确和。

GQA中尽量共享一次原K抽样：若各head的w相同，直接共享J；若w不同，可从 \(\pi_j=G^{-1}\sum_gw_{gj}\) 抽样，用 \(w_{gJ}/\pi_J\) 修正每个head的残差项。每个head仍无偏，并且该混合proposal下权重比不超过G。所有head共享样本会产生相关误差，因此按实际联合差值估计排序方差，不能假定独立；真实K读取量按样本并集统计。

估计可能非正；非正时回退该块 exact，不裁成一个小正数后宣称仍无偏。回退后的整体混合算法也不自动继承原估计的无偏性，需另按实际结果评价。

### 最小实验

当前真实 Q 在线采样；保持种子策略固定。48 DEV主种子运行；另选事先固定的8个输入用两额外种子测算法噪声。对少量共同状态读全部K求出残差真实方差，估计在该有限分布下 m=8 的可行性，而非假设 Hoeffding 界已经足够实用。

### 六种排除的替代设计

| 未选设计 | 排除原因 |
|---|---|
| 直接均匀采样少量 logits 取 LSE | 尾部方差高，也没有利用已有矩信息 |
| 只采 query 无关的大半径 key | 无法覆盖多个真实 query 方向 |
| 没有全支持的 importance proposal | 真正重要 key 可能采样概率为零 |
| 负估计强行 clip 并声称无偏 | 改变估计性质，并可能系统性错排 |
| 对随机 logmass 用 Gaussian 标准误 | log与归一化非线性，尾部可严重偏斜 |
| 持续加样本直到候选赢分 | 数据依赖停止和实际读取成本被隐藏 |

### 失败后得到什么

可算出的残差方差给出需要多少样本才可能稳定区分真实 margin。若所需 m 接近 B，说明**当前控制变量没有消去昂贵的指数尾部**，应换控制函数／表示，而不是继续增加随机种子。对稀有质量为 w* 的必要事件，m次采样漏掉它的概率为 \((1-w_*)^m\)，能明确暴露样本预算与尾部之间的冲突。

这不证明所有随机 attention 方法无效，也不能把一次运气较好的生成当真实收益。

### 成功后深化

比较“相同 raw-K读取量的直接抽样”与控制变量，验证方差缩小而非更多读取；再做第二模型和尾部压力分层。控制变量已有相关注意力研究 [W15]，独立贡献要落在**cached PC2 残差、当前查询决策和可用的方差—带宽 trade-off**。

---

## E07｜固定总读取预算，允许同层 KV groups 分配不同名额

**研究问题。** 即使每个块的分数准确，uniform group quota 是否仍把稀疏资源花错了地方？

### 入选构造

保留 B0 的分数和每个 group 内的原排序。只在**同一层**分配总名额 \(M=\sum_hm_h^{B0}\)，不跨未来未知层拿 oracle 分数。

若各 group 的块成本相同，对原可加目标，把各 group 的候选边际分数合并，选择总共 M 个；保留预先声明的每组最小名额1以及原 mandatory 集合。各组最终取其排序前缀。它是相应单调可加预算目标的精确分配，而非“最优回答分配”。若实际成本不同，按明确的离散成本做预算优化，不能用分数／成本排序冒充一般 knapsack 的精确解。

### 最小实验

48 DEV。记录各组实际 \(m_h\)、真实 gather/reader bytes、padding和kernel时间。主结果比较相同总物理读取；若 backend 被最大 m padding 成统一大小，则按 padding 后的真实开销报告，不能写已等资源。

### 六种排除的替代设计

| 未选设计 | 排除原因 |
|---|---|
| 各组按固定百分比增加 m | 总预算增加，无法识别分配作用 |
| query head 各自分配再展开 KV | 破坏 GQA 共享与真实缓存成本 |
| 依据未来题目标签分配 | 使用不可部署的信息 |
| 跨层先看全部 exact scores再排全局预算 | 需要尚未产生的后续层状态 |
| 忽略 local/sink的head归一化 | 对原本不需远程读取的组高估需求 |
| 只数理论块数，忽略 ragged padding | 可能完全没有实际预算变化 |

### 失败后得到什么

若最优可加分配仍接近uniform，说明这个工作点的预算异质性不足；若可加保留质量升而生成降，与 E02 一起检验“线性 mass 值得怎样跨head使用”；若质量升而 kernel无收益，结论是分配有效但当前硬件接口不能兑现，不能再用统计摘要解释问题。

### 成功后深化

沿固定总预算只增加一个更低预算点，检验是否真正移动质量—成本曲线。对比 Ada-KV 等预算工作时明确：这里是原 K/V 常驻的稀疏读取，而不是相同生命周期的 eviction。[W12] 若只得到通用 adaptive quota，作为系统组件或机制结果，不独占一篇位置论文的 novelty。

---

## E08｜准确 mass 之外，遗漏 value 的条件响应能否便宜补回

**研究问题。** 仅提高选择精度不够时，是否可以在不增加 raw-V读取的情况下减少 sparse renormalization 的输出偏移？

### 入选构造

使用 B0 的**同一个 exact 选择集合**，并使用真实 reader 的 exact 块 mass（§1.1说明其与CIS score mass可能的区别）。构建每个原物理块的 \(\bar v_b=E_{w_b}v\)。选中块的 numerator/denominator 原样精确；遗漏块用自己的真实 mass 加权 value均值：

\[
\hat o=\frac{N_S+\sum_{b\notin S}M_b\bar v_b}
{Z_S+\sum_{b\notin S}M_b}.
\]

这里不是直接加一个没有质量权重的 residual，也不是把全局 meanV 复制到每个 query。它是一个明确、凸组合形式的替代 reader。不得把其质量差归给 selector。

### 精确的误差对象

令真实条件 value 为 \(v_b(a)=E_{p_a(\cdot|b)}v\)，则

\[
\boxed{\hat o-o=\sum_{b\notin S}p_b\,[\bar v_b-v_b(a)].}
\]

若各被遗漏块内 V 恒定，本法对固定 Q/K/V 精确恢复 dense attention output。一般情况下，剩余误差由 key–value条件相关性决定，不能只看 V 的整体范数小。

实际整个模型运行时，候选的历史会自然变化；上面的恒等式只对一次固定状态成立，不承诺回到完整 dense 生成轨迹。

### 最小实验

48 DEV；同时保存少量固定状态的 `B0输出 / 本法输出 / dense真实输出`，记录块内条件均值差。新 meanV 每块仅 d_v 个数，额外读取和加权GEMM计价。构建meanV若由CIS权重决定，也必须处理其是否可缓存。

### 六种排除的替代设计

| 未选设计 | 排除原因 |
|---|---|
| 全局 meanV 无区分补全部tail | 无法检验块级条件结构，且 SparQ已有近邻 |
| 不估 omitted mass直接加向量 | 改变输出幅度，没有归一化合同 |
| 只恢复 denominator、不补 numerator | 会不成比例地缩小现有输出 |
| 同时换 selector和value补偿 | 无法区分哪个接口贡献收益 |
| 用正确答案定位要补的块 | oracle而非当前实际部署 |
| 直接用不受控线性cross-moment外推V | 可能远离value凸包并放大读出；初版先用有界代表 |

### 失败后得到什么

若精确 mass 下补偿仍差，直接测 \(\bar v_b-v_b(a)\)：它大则说明 tail 需要 query-conditioned value 信息，不是继续加 key covariance就足够；它小而生成仍差，则输出 fidelity 与答案目标之间仍有余量／交互，不能将局部误差收益冒充任务收益。

### 成功后深化

只在成功后尝试每块两个具有 key–value对应关系的代表，和全局 meanV的匹配成本对照；再将 B0 exact mass 换成候选低成本 mass，分开测两项误差。SparQ和value-aware attention已有相关思路 [W5,W16]，主贡献必须是PC2选择合同中被遗漏的条件响应及可复用的质量—成本收益。

---

## E09｜共享压缩基应否随块位置变换：真正的 position-specific 实验

**研究问题。** 对共享子空间的统计压缩，固定全局坐标基是否把本来相同的相对位置结构浪费成多个方向？

### 先排除一个必然无效的“方法”

单纯把同一块的 Q/K 都旋转到局部坐标，不改变精确内积，也不改变完整协方差的特征值。更强地，若 R 是相同pair上的旋转：

\[
\operatorname{pairdiag}(R^T\Sigma R)
=R^T\operatorname{pairdiag}(\Sigma)R.
\]

因此**只换局部坐标、仍用相同pair-PC2，分数严格不变**。这一版本不进GPU候选。局部旋转也不能降低每个块自己的最小精确秩。

### 入选构造：变的是跨块共享基，不是单块坐标名字

设块参考位置为 \(c_b\)，\(k'_j=R(-c_b)k_j\)，\(a'_b=R(-c_b)a\)。用校准材料拟合一个共享低维基 U。每块存局部框架下的投影协方差 \(C'_b=U^T\Sigma'_bU\)，分数

\[
\hat F_b=\log Z_b+a^T\mu_b+
\tfrac12(U^Ta'_b)^TC'_b(U^Ta'_b).
\]

原始位置差 \(p_j-c_b\) 仍然存在，不是把每个 key独立撤掉RoPE。作为对照，固定全局共享 U 并存 \(U^T\Sigma_bU\)，同维数、相同来源、匹配总descriptor bytes。

### 理论贡献的准确边界

普通共享投影 P 的双线性近似在共同位置平移t下对所有a,k等变，当且仅当

\[
R(t)^TPR(t)=P.
\]

任意混合频率的低秩共享基通常不满足该条件。对块使用
\(P_b=R(c_b)PR(-c_b)\)，则在共同平移并相应更新c_b时自然共变。这是群作用的标准结论在共享索引压缩上的应用，不宣称数学工具首次出现。

但共同原点平移时，**公平基线也可以同时旋转它的P**；不能故意保持基线基不动制造“位置鲁棒性提升”。真正待检验的是：在不同c_b同时存在的自然材料里，局部frame是否让一个共享低维表示更经济。

### 最小实验

使用E04同一批独立校准材料。比较 local共享基、global共享基、每块独立同rank压缩的诊断。记录共享谱随局部对齐是否集中，以及同descriptor预算下真实选择和生成。

第一版s同E04；每块query旋转与投影需要约O(N_blocks d s)，不能把全球一次projection的成本偷换过来。普通完整pair-PC2的local/global分数不变，作为最短零效应单元测试。

### 六种排除的替代设计

| 未选设计 | 排除原因 |
|---|---|
| 只把每块Q/K换局部坐标仍用原pair-PC2 | 数学上完全同一分数，必无新方法效应 |
| 每个key分别逆旋转到自己的位置0再pool | 改变真实reader需要的相对位置关系 |
| 认为旋转会降低单块covariance rank | 正交相似变换不改变特征值 |
| U只由完整不混频pair构成却宣称共变修复 | 该P已经与R交换，原点不匹配未出现 |
| 只给候选变换坐标、不给基线基共变 | 不公平的gauge对照，制造虚假收益 |
| 忽略每块query投影的计算 | 可能省descriptor却花更多在线时间 |

### 失败后得到什么

如果局部对齐后共享谱没有变集中，说明这组实际keys没有该共享统计结构，不能再靠旋转对齐期待更低rank；若共享谱集中但真实Q误差仍大，说明key压缩方向没有覆盖查询需求；若选择正确但成本无益，需要的是结构化可快速变换的基，而不是声称PE已改善任务。

### 成功后深化

冻结同一个基跨原生位置区间、长度和模型族验证；报告纯几何等变性质与自然生成分开。可形成的中心是：**位置作用应穿过共享压缩表示，而不是在压缩前被不兼容的全局子空间丢掉。**必须胜过已有frequency-aware/NoPE/compression近邻，不能仅重复Prism的pooling相消。[W3]

---

## E10｜直接学习决策边界的微型残差选择器，而非再次拟合平均矩误差

**研究问题。** 现有PC2描述符中是否已有足够信息，只是固定二次读出不足？或者描述符确实丢了任何小读出器都无法恢复的信息？

### 入选构造

骨干权重完全冻结。令z_b为已存在的PC2均值、pair矩、权重总量及可缓存范围统计。缓存一个小映射 \(v_\theta(z_b)\in\mathbb R^8\)，当前query只过小映射 \(u_\theta(a)\in\mathbb R^8\)：

\[
\hat F_{\theta,b}=F_{\mathrm{PC2},b}+u_\theta(a)^Tv_\theta(z_b).
\]

初版使用最多一层64宽隐藏层的映射；descriptor输出8维，仍使用实际归一化GQA规则。不添加新训练任务或答案监督。

teacher取校准材料上B0实际当前Q的exact块分数／GQA分数。每个样本的训练pair覆盖teacher cutoff两边，并加入固定比例的全局随机未选块，以免只看一个自我满足的shortlist。

设真实margin \(\gamma_{ij}=G_i^*-G_j^*>0\)，采用直接对应这些margin的hinge，例如

\[
\mathcal L=\mathbb E_{(i,j)}[\gamma_{ij}-(\hat G_i-\hat G_j)]_+,
\]

可按校准集预先固定的尺度归一化。不是以答案标签挑“重要块”。只跑一份预先固定训练预算（例如500更新步），独立文档验证；若需检查轨迹偏移，最多增加一次预声明的student轨迹采集／teacher标分，不反复用DEV调到赢。

### 理论能说什么

在给定(a,z)的平方预测合同中，任何读出器的不可约风险是
\(E\operatorname{Var}(F\mid a,z)\)。相同mean/covariance可以对应不同logmass，所以不存在一般的“足够大MLP一定恢复所有块”定理。margin监督只把训练目标拉近决策，不保证训练可达、泛化或真实答案。

### 六种排除的替代设计

| 未选设计 | 排除原因 |
|---|---|
| 直接LoRA/SFT整个骨干 | 混入额外能力学习，破坏小改动和成本对照 |
| 平均logmass MSE覆盖全部块 | 大量无关块可主导目标而忽略cutoff |
| 只用teacher选中块和一个固定shortlist | 不能识别集合外潜在真高分块 |
| 用gold source/答案训练router后称data-free | 改变监督合同且有标签依赖 |
| 只报告teacher-forced训练轨迹 | 掩盖student自身query的偏移 |
| 不断扩网络直到dev赢 | 无法区分信息不足、泛化与选择偏差 |

### 失败后得到什么

训练也拟合不了时，先区分优化未收敛与(a,z)缺信息，不能武断归因之一；训练拟合、独立来源失败，得到明确的校准迁移问题；held-out cutoff改善而live生成无益，说明低误差selector还没有兑现任务／成本。它可以帮助判断“应该增加描述符信息，还是改变读出函数”，而不是再无目的调rank。

### 成功后深化

冻结backbone与校准规模跨域／长度确认；比较同参数的全块MSE或KL教师和cutoff-regret教师，验证不是单纯多训练了一个indexer。DSA/SAAP已经学selector [W13,W14]，新意必须是PC2可缓存描述符的决策充分性、有限校准和可用成本，不是“我们也训练router”。

---

# 5. 即便十项都失败，怎样得出真实有用的结论

“都没涨分”不是足够详细的实验输出。每项的中间量对应一个**有限、可行动**的分支：

| 观察组合 | 能得出的实际结论 | 下一项构造应变化什么 | 不能由此宣称 |
|---|---|---|---|
| E01区间窄、精算少，但仍慢 | 信息足够；实现／调度／数据布局不划算 | kernel融合或索引布局，而非更多统计量 | PC2理论无效 |
| E01区间广；E04 projected exact也误差大 | 当前压缩分辨率不足以维护该query集合的边界 | 提高表示能力或更换压缩对象 | 所有小摘要不可能 |
| E04经验指数明显胜同U二阶 | 高阶在该子空间确实有信息价值 | 保留有限分布／条件成分，不继续只修covariance | 已证明回答收益 |
| E04投影无益；E05两成分也不能压小余项 | 两类紧凑表示没有覆盖该工作点 | 用真实方向风险或改变预算／读取目标 | 高阶或位置整体无关 |
| E02降低非线性R且output也改善，但答案无益 | 所测读取fidelity不是充分任务目标 | 更接近任务的层/路径或信息目标；保留原评分结果 | 模型不能被改进 |
| E07质量变差但E02变好 | additive预算价值与head平衡有冲突证据 | 联合的非线性资源目标 | 已唯一找到神经机制 |
| E08条件value均值误差大 | mass不是唯一缺口，遗漏响应需要条件信息 | joint K/V代表而非单纯更多key二阶 | 全部value压缩无效 |
| E03刷新率高且真实delta大 | 此协议缺少足够的短期响应复用 | 不再用时间复用作为主要省算来源 | 所有模型时间相关性低 |
| E06所需m接近B | 该控制变量没有消去指数尾部的采样成本 | 换控制函数／离散表示 | 所有随机方法不行 |
| E09局部共享谱不集中 | 位置信息对齐未提供该共享压缩结构 | 不把“局部坐标”当成本来源 | RoPE不重要 |
| E10训练好、跨域差 | 描述符读出在当前校准下不迁移 | 校准覆盖／结构约束，而非无限加训练步数 | 无法学到有用selector |

**若十项都完成且都没有实用收益**，结论不是“位置领域没有机会”，也不是继续发明十个名字。要按观测分别判断：

1. **表示有信息但系统不省**：问题归到表示—硬件实现，利用已经足够的表示，不再做数学微调。
2. **选择目标的上界改善不转移到输出**：应改变value/多头目标，不再将mass视为充分代理。
3. **实际输出改进不转移到答案**：这个模型／任务／预算下，当前接口不是主要可用增益来源。
4. **多个紧凑表示都不足、但原K exact仍有用**：得到的是明确预算下表示与当前Q分布的困难，不是普遍不可压缩定理；需要新信息表示，而非原目标上更多同质变体。
5. **结果区间宽或实现未完成**：只有不确定／工程未交付，不能算成十个科学假设已否定。

负结果的价值在于使下一次构造改变真正相关的对象。它不自动凑成一篇可以发表的负结果论文。

---

# 6. 实际执行顺序：不重新制造十个大型实验工程

## 6.0 实施成本与算法读取量：先算清 ROI

记完整prefix有N个token，n=N/B个物理块，d为head宽度，G为一组Q-head数，s为投影宽度。下表是算法计数，不是GPU实测预测；矩阵布局、共享与padding会改变真实时间。

| 实验 | 新增实现范围 | 一次构建／状态 | 单query主要新工作 | 最关键成本风险 |
|---|---|---|---|---|
| E02 | 现有score tensor上的集合函数 | 无新大缓存 | 最多两轮有限候选交换，约O(G m(m+2G)) | 大m下候选枚举；可批量tensor化 |
| E01 | 有界量化索引、tiled scorer、refine调度 | O(Nd)构建；INT8索引约原BF16 K的一半，另加包络 | 低精度全索引扫描＋f比例原K精算 | 全量反量化workspace、过多回退 |
| E04 | 共享基与projected empirical scorer | O(Nds)投影；约Ns+nd个descriptor数 | O(ds+nd+Ns)，仍有N项exp | s不够小、exp与descriptor构建成本 |
| E08 | 每块meanV＋一个tail加权项 | O(Nd_v)构建；nd_v个数 | O(nGd_v)，mass复用或显式计算 | 额外GEMM抵消读取收益 |
| E03 | block-response cache与在线刷新 | 最坏约O(nGd)梯度加O(nGd)参考Q；需真实计数 | O(nGd)更新界＋f比例exact/梯度刷新 | 状态接近原K大小；query快速漂移 |
| E07 | 同层预算分配、合法ragged读取 | 小型配额元数据 | 已有分数上global allocation | padding按最大配额执行 |
| E05 | 物理块内两成分构建和scorer | 两次farthest约O(Nd)；约2套pair-PC2描述符 | 约两套PC2＋2项LSE | 多存一倍摘要而非机制更优 |
| E10 | 小型残差读出训练／缓存 | 固定校准；每块8维额外向量 | 每query小MLP＋每块8维dot，另有原PC2 | 校准成本、特征不可识别 |
| E09 | 局部框架共享基和projected covariance | 约O(Nds)构建；每块均值＋s²统计 | O(nds+ns²)，不能冒称只投影一次 | 每块旋转／投影过贵 |
| E06 | 分组共享抽样、控制变量残差 | 原PC2＋合法抽样权重 | 约O(nmdG)，真实fetch按group union计数 | 随机gather和tail方差 |

先通过同一B0的已有scorer确认哪部分时间可被这些操作替换。不能拿总体decode时间直接乘数学压缩比预测提速。对于E02/E08这类目标／reader实验，即使第一轮没有省算，只要有足够的同资源质量提升，也有继续做低成本实现的理由。

## 6.1 先做便宜且能改变判断的两项

**E02 → E08。** 二者复用B0现有exact分数与reader相关缓存；首先分别检验集合目标和tail读取。不要先训练E10，也不要先写全套量化kernel。它们虽然可以与B0同前缀比较，live continuation仍独立运行。

随后开展 **E01 → E04**，分别测试保守精算分配和更完整的低维响应。这四项覆盖目标、读取、数值与表示四个不同瓶颈，不是四个同方向的rank。

后续 **E03、E07、E05、E09、E10、E06** 使用相同DEV与B0资产按适配成本安排。每项都有自己的单变量版本，不把前项的新组件强制作为后项前提；例如E08仍使用B0exact质量，不等E04成功。

## 6.2 全部十项如何控制成本

每项先用共同48 DEV完成一次候选真实比较。昂贵的新路径可先跑每任务2个输入（8个），确认能完成生成和计时后继续剩余40个；这8个计入48个，不另起一份统计。没有依赖关系的CPU构建和结果分析可并行，GPU只由已有负责人调度，不杀既有任务。

不要因为某个实验尚未写出加速kernel就阻塞其他可运行实验。正确性reference完成、性能实现未完成的项，分开记这两个事实。

第一批8个输入测得实际秒数t_i后，使用

\[
T_{plan}=T_{shared\ baseline}+\sum_i48t_i+T_{build/calibration}+T_{confirmation}
\]

估算剩余成本，而不是提前承诺“十个实验一小时一定跑完”。用户实际授权预算优先；本文不授权新租GPU、不扩大付款上限。预算不足时保留十项设计，但只运行可完成的高ROI项，不能把没跑的项记为失败。

## 6.3 实验数量与选择偏差

10个候选共用DEV，意味着DEV同时承担选择方法的功能。因此获胜者必须在**新的文档来源**确认。不能从十个候选中挑最高DEV分后沿用该DEV置信区间声称显著。

对成对分数差D，确认样本量可用开发集估计的方差作初步规划：

\[
n\approx(z_{1-\alpha/2}+z_{1-\beta})^2\widehat{\mathrm{Var}}(D)/\delta^2.
\]

这是规划近似；二元、多来源、聚类任务用相应设计与bootstrap。确认最多先选2个，分别事前指定比较，或控制相应多重检验。若样本不足以证明1pp非劣，不把“不显著”改称“保持质量”。

---

# 7. Codex 的最小实施合同

## 7.1 不动的东西

模型权重、原生位置表、tokenizer/chat template、问题可见性、原始K/V和数据划分。E02/E07/E08明确标记的单变量例外之外，原GQA、名额和reader不动。不同工作线不互相替换baseline。

不要改全局AGENTS.md，不搭第二套评测框架，不重写已有论文。只新增小型candidate函数及必要缓存字段。

## 7.2 接口建议，不假装这是现有仓库路径

```python
class Candidate:
    def build(self, prefix_state, baseline_contract):
        """Only prefix-visible data. Return candidate metadata; never mutate raw KV."""

    def select(self, current_query, raw_cache, metadata, baseline_contract):
        """Return legal physical block IDs and actual extra scoring cost."""

    def read(self, current_query, selected, raw_cache, metadata, baseline_contract):
        """Default: existing reader. E08 explicitly supplies a different tail response."""
```

默认沿用仓库已有等价接口；不为遵守这段伪代码做重构。

最小配置字段：

```yaml
experiment_family: pc2_ten_directions_v1
baseline: existing_exact_manifest
model: inherit_baseline
rope: inherit_baseline_unchanged
block_size: inherit_baseline
mandatory_policy: inherit_baseline
main_remote_budget: inherit_baseline
live_question_ingest: sparse_from_first_question_token
main_comparison: candidate_vs_same_B0
candidate_ids: [E02, E08, E01, E04, E03, E07, E05, E09, E10, E06]
calibration_independent_documents: 24
dev_independent_documents: 48
confirmation: fresh_sources_after_candidate_lock
weight_updates_backbone: false
selector_training: E10_only
new_gpu_rental: false
budget: inherit_existing_user_authorization
```

## 7.3 只验证变化的路径

只改score／set目标，复用reader parity；E08新reader需全可见退化检查；E01/E03需interval和same-set检查；E09有数学上必须为零的pair-PC2坐标变换控制。源码小字段／报告变动不使B0全部失效。

CLI类型、row ID文件格式、必需baseline路径在模型加载前检查。B0缺哪几行补哪几行，不每个候选重跑完整baseline。用同保留集合验证算子等价，不只看最终是否答对。

## 7.4 输出只需要三类文件

1. `per_example.jsonl`：候选、文档ID、完整输出、原指标、实际配置、耗时和资源。
2. `summary.csv`：十项同一个B0的质量／成本／机制关键读数。
3. `DECISION.md`：逐项实际做了什么、是否过实用门槛、机制结论、未排除的解释、下一步构造必须改变的对象。

候选使用额外supervision、cache、样本或exact补算，全部显式入账。算法“运行结束”不算研究成功；一个正确恒等式也不算方法已提升模型。

---

# 8. 什么结果能形成有竞争力的论文

最终只围绕一个被数据支持的中心，不把十项全塞进Introduction。

### 路线A：决策保持的低成本位置敏感块选择

E01/E03/E04中至少一项在多个模型／长度实现真实质量—成本前沿；E09或相应严格对照证明保留位置作用是不可替代的一部分。可以主张：**不需要统一提高所有块的统计阶数，只需使位置敏感的实际选择决策在低成本表示下可靠。**

需与Quest、完整COBS、Prism及相关精确／近似索引对照；exact保真加速本身可以有系统价值，但标准量化＋rerank不足以自动获得强novelty。

### 路线B：PC2以外的瓶颈是集合目标／条件响应

E02/E07/E08显示，在exact mass已知时，仍可因GQA或value响应处理获得真实改善，再用一个低成本可部署评分器保留增益。核心可以是：**选择目标不是可加mass的简单堆叠，有限稀疏资源必须考虑head与条件输出的结构。**

这首先是稀疏attention论文；除非有独立位置因果证据，不硬写成“新的RoPE方法”。这不是降低贡献，而是不把题目与实际贡献错配。

### 路线C：共享位置共变表示

E09若在同字节下明显降低实际所需表示维数，并改善真实任务／成本，可独立发展成位置编码与索引压缩的强结合。必须展示：pair-PC2换坐标不变的null、共享基共变条件、相同预算的全局/局部/独立基、跨位置泛化与真实生成。

**最低完整证据结构**：一个非平凡方法或机制、一个可复现的实际前沿结果、两个不同模型/结构、自然与受控任务、真实总成本，以及最近邻无法解释全部增益的对照。单纯十项实验做完、或某个toy例子成功，都不等于已到accept。

---

# 9. 本轮附带的数学检查及其边界

`math_checks.py` 已在本轮CPU运行，`math_checks.json` 保存结果。检查包含：量化logmass与归一化GQA区间、非线性GQA交换、时间更新界、经验投影界、两点mixture精确性、控制变量期望、固定预算可加分配、块内constant-value补偿、pair-PC2坐标不变性／共享投影共变性、同矩不同logmass反例。

**这些只验证所写代数与参考实现，不是checkpoint实验，不用于给上表成功率赋值，不支持任何速度或任务提升。**实际实验应直接复用已有模型栈，不要求先扩成一份新的CPU审计工程。

---

# 10. 来源与边界

## 项目来源

- **[U1]** `PC2_PM_FAILURE_QUESTION_AUDIT_20260910.md`，尤其Q04的PC2定义、Q09的稀疏读取合同、Q22/Q24的指标与baseline身份。本文不以其历史失败禁止新构造。
- **[U2]** `native_sparse_position_research_plan_20260908.md`，尤其§3–5的原始KV块mass、GQA与位置接口。它是PSR前身计划，不当作当前PC2完整源码或已运行配置。

## 一手公开来源

- **[W1]** COBS: Cumulant Order Block Sparse Attention，§3–5、§7、§9。 https://arxiv.org/html/2607.09052v1
- **[W2]** Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference，ICML2024。 https://proceedings.mlr.press/v235/tang24l.html
- **[W3]** Prism: Spectral-Aware Block-Sparse Attention。 https://arxiv.org/html/2602.08426v2
- **[W4]** Loki: Low-rank Keys for Efficient Sparse Attention。 https://arxiv.org/abs/2406.02542
- **[W5]** SparQ Attention: Bandwidth-Efficient LLM Inference，尤其mean value reallocation。 https://arxiv.org/html/2312.04985v6
- **[W6]** AB-Sparse: Sparse Attention with Adaptive Block Size for Accurate and Efficient Long-Context Inference。 https://arxiv.org/abs/2605.12110
- **[W7]** Recall Before You Rank: Similarity-Guided Top-K Reuse for Efficient Long-Context Attention。 https://arxiv.org/abs/2607.27692
- **[W8]** Guess-Verify-Refine: Data-Aware Top-K for Sparse-Attention Decoding on Blackwell via Temporal Correlation。 https://arxiv.org/abs/2604.22312
- **[W9]** Sample-Guided Exact Top-K Selection for Long-Context Sparse Attention。 https://arxiv.org/abs/2609.08450
- **[W10]** A Mathematical Theory of Top-k Sparse Attention via Total Variation Distance。 https://arxiv.org/html/2512.07647v1
- **[W11]** Uncertainty-gated selection for block-sparse attention。 https://arxiv.org/html/2607.07724v1
- **[W12]** Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation for Efficient LLM Inference。 https://arxiv.org/abs/2407.11550
- **[W13]** DeepSeek-V3.2 technical report。 https://arxiv.org/html/2512.02556v1
- **[W14]** Inference-time Sparse Attention with Asymmetric Indexing。 https://arxiv.org/abs/2502.08246
- **[W15]** Efficient Attention via Control Variates。 https://arxiv.org/abs/2302.04542
- **[W16]** Value-aware Approximate Attention。 https://arxiv.org/abs/2103.09857

未核验正式接收状态的2026工作按预印本／技术报告处理。本文的拟议新构造与实验结果预测不是上述作者报告的事实；没有由论文标题补造实现，也没有声明查新已穷尽。
