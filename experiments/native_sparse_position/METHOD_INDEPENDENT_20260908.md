# 第二轮独立判断：旋转二维平面上的协变极值摘要

更新时间：2026-09-08 22:40 UTC。**本节取代下方初稿的主推荐；初稿保留为推导历史。**
当前落点是主任务正在进行的同预算源记录块修复 oracle，不再建议直接铺 150 条自然格式 F1 矩阵。

## 单一路径

我建议从泛 key 聚类转向 **Rotary-Pair Extremal Envelope（RPEE，暂名）**：每个物理块、每个真实 rotary pair，保存一个随实际 key 分布旋转的二维包围矩形；用当前真实 query 对该矩形的支持函数给块评分。原 K/V、频率、物理块、local/sink、top-k 不变。它是对 Quest 极值摘要的结构性修正，不是再发明 key clustering。

具体研究假说是：**均值把稀有证据峰值抹掉；普通逐坐标极值又可能把不同 token 的 x/y 极值拼成并不存在的高分点。RoPE 改变这些坐标极值的拼接误差，而真实 attention 对共同旋转保持不变。以原生 rotary pair 为最小几何单位，可以保留极值，同时去除这类坐标系依赖的误选。**

这条假说只有在正在验证的可修复错误中成立，才值得做 GPU 方法臂。它不因以下 CPU 性质自动获得额度。

## 真实的一手近邻核对结果

- **TriAttention §4.1–4.3，Eq.6–13** 使用离线校准的 pre-RoPE **query center**，但式6使用每个缓存 key 的**实际表示**，并非实际方法把 Q/K 都替成均值；随后加频段 norm correction、平均未来 offsets，在 GQA 中做 headwise z-score 后取 max，每128生成token进行 pruning。其对象是未知未来 query 下的 token 保留。RPEE 使用到达的实际 query、块级缓存、保留原reader，不需要 query-center 校准或未来位置平均。因此“非零中心＋三角关系”已经有明确近邻，不能作为新贡献。来源：https://arxiv.org/html/2604.04921v1 ，作者代码链接由正文给出为 https://github.com/WeianMao/triattention 。
- **Quest §3 与 Algorithm 1** 保存每页每坐标的 min/max，评分是各坐标最大可能乘积之和。RPEE 必须正面胜过它，不能只打均值。来源：https://arxiv.org/html/2406.10774v2 。
- **COBS §5.1–5.2、§6.4** 采用 NoPE compression/selection，缓存协方差用于二阶 cumulant mass 估计，并明确分析 Quest 的轴对齐 box。RPEE 的协方差只确定二维坐标轴，缓存的是**实际 extrema**；不把 qᵀΣq 当作 mass，也不作 Gaussian tail 假设。来源：https://arxiv.org/html/2607.09052v1 。
- **Prism §3–4** 已覆盖 post-RoPE mean pooling 衰减和 spectral calibration。RPEE 既不重放大均值，也不重建虚拟 key。来源：https://arxiv.org/html/2602.08426v2 。
- **SAAP §3.3–3.4** 不仅已有 key clustering，还明确指出 Q/K 的分布不一致使最近中心访问失效，并用 de-RoPE 降低时间偏移。这个事实进一步削弱我初稿把 key 欧氏 k-center 作为独立主方法的说服力。RPEE 不靠 query 与 key 同分布，而对实际 query 给真实 key 的分数上界。来源：https://arxiv.org/html/2502.08246v1 。

以上五篇均在本轮下载并直接阅读正文。没有查到这些文中的同一“原生二维 pair 协变 extrema”构造；**这不是穷尽查新结论，更不等于已经具有足够新意。**

## 构造：到可以直接编码的程度

对每个物理 B=64 块、每个 KV head、每个真实二维 rotary pair，令 `x_j∈R²` 为实际 post-RoPE key pair。Qwen 的 split-half 配对必须是 `(i,i+d_rot/2)`，不能误用邻接维。

1. FP32 算 `mu=mean(x)` 和二维中心化协方差 `C=mean[(x-mu)(x-mu)^T]`。
2. 若两个特征值不同，令 `u` 为最大特征值的单位方向，`v=J u` 为垂直方向。分别对所有真实点的 `u·(x_j-mu)`、`v·(x_j-mu)` 求 min/max。
3. 把这个有向矩形写成 `c + [-a,a]u + [-b,b]v`。缓存 `c(2),u(2),a,b`，共6个数/pair，即全 rotary 下 **3D**。`v=(-u_y,u_x)` 不另存。
4. 若协方差严格各向同性，采用以 `mu` 为中心、`r=max_j||x_j-mu||` 的圆盘。实际代码对相对 eigengap 小于固定浮点稳定阈值的情况也使用圆盘；例如 FP32 下 `1e-6` 是数值约定，不作为任务参数搜索。零方差也走该分支，r=0。这个分支避免退化 PCA 轴任意性破坏旋转协变。
5. 当前实际、含 scaling 的 query pair `q` 对该矩形的上界为

   `U_pair(q)=q·c + a|q·u| + b|q·v|`。

   圆盘分支为 `q·mu + r||q||`。非 rotary 维保留普通 min/max。
6. 全块评分 `U_b(q)=sum_pairs U_pair(q) + nonrotary_terms`。选择最大 U_b 的 remote 物理块，再按原 reader 完整读取。它是 **max-logit upper bound**，不是 mass 的无偏估计；对等长块加 logB 不改变排序。

构建 O(BD)，只包含2×2协方差、解析 eigensystem/小 eigh 与投影 extrema；不做 B×B 聚类、不需 query 校准。每块完成后只构建一次，最近/未满块由既有 local 路径覆盖。查询成本 O(D)，无需恢复 B 个虚拟 keys，也不全扫原始 K。

缓存为 `2D+d_rot` 个数/块，全 rotary 是3D；少于 PSR 的4D，多于 Quest 的2D。必须把这个差别计入实际成本。对照除了原始 Quest，还应考虑两段 contiguous Quest（4D），不能把多50%的摘要内存藏起来。FP16/BF16 存储若要声称上界，必须处理矩形端点/轴舍入的 outward error；第一版 FP32 selector 的真实 byte 数应如实记录。

## 三个可证结构，以及不能越界的地方

**(A) 当前 query 的有效上界。** 每个真实 `x_j` 在其包围矩形/圆盘内，因此 `q·x_j<=U_pair(q)`，从而 `max_j q·y_j<=U_b(q)`。不依赖 query 与 keys 同分布，也不依赖 query center 近似。

**(B) 精确 reader 的共同旋转对称性得到保留。** 对任意共同位置平移Delta，每个 native pair 分别乘 `R(omega Delta)`。均值、协方差主轴和矩形中心一起旋转，半宽不变；圆盘同样协变。q也共同旋转时，U_b严格不变。原始 Quest 的轴对齐 box 不具有该性质。原生 partial RoPE 的恒等维不受影响。

**(C) 二维共线 keys 的支持函数精确。** 如果某个 pair 的 keys 落在一条线上，b=0，其 per-pair 支持函数完全精确，不会把同一 pair 内两个 token 的 x/y 极值拼出虚假角点。**这不是整个D维 block max必然精确**：不同 pair 的最大值仍可能来自不同token，跨pair的依赖仍被丢弃。

这一结构把“位置”落实为可检验的 **rotary pair 闭包与共同位置平移的对称性**，而不是给一般 clustering 加一个 RoPE 名字。

## CPU 实际检查与强反例

一个64-token、单pair的精确例子：块A由32个 `(1,1)` 与32个 `(-1,-1)` 构成；块B的64个key均为 `(.5,-.5)`；query为 `(1,-1)`。真实 max logits 永远为 A=0、B=1。

| 对 q/K 共同旋转 | Quest A/B | RPEE A/B |
|---|---|---|
| 0 | 2 / 1，错误选A | 0 / 1，选B |
| pi/4 | 约0 / 1，选B | 约0 / 1，选B |
| pi/7 | 1.2469796 / 1，错误选A | 约0 / 1，选B |

真实 attention logits 在这些共同旋转下完全相同；Quest 排名的改变来自摘要坐标系。这是位置规范选择导致的索引伪差异，**不是已经观察到的模型失败**。

另外在1000个FP64随机二维块上，已实际检查：最大上界违反仅 `8.88e-16`，共同旋转后的最大评分差 `7.11e-15`。

也验证了反面：keys为 `(0,0),(2,0),(0,1)`，q为 `(1,0)`，Quest上界2完全精确，而PCA矩形上界约2.1094。因此RPEE**不逐query支配Quest**；PCA方向也不是最小面积矩形的保证。不能把旋转等变性当成性能保证。

更重要的失效类：跨pair关联才是主因时，两种 product envelopes 都可能很松；大量各向同性 keys 的圆盘也可能产生太多false positives；max-logit排序可能偏离真正有用的 block mass。这些都要在当前可修复错误中判定，不能再改几种阈值维护构造。

## 必须与正在进行的真实修复 oracle 对接

本方法要解释的是“源记录块确实缺失，并且同预算补回能恢复完整正确回答”的实例，不是一般自然F1波动。

**先取得主任务本来就在做的 paired 证据：** Dense正确、RoPEMean失败、source-block oracle恢复、同距离错误记录块oracle不恢复。源记录根据问题key定位，不能用gold数值参与method selection。多key/multiquery必须完整输出全部要求的答案及EOS。

在这些实例里，从**实际 RoPEMean 失败轨迹**保存 query/key，并定位源块首次需要进入候选但被漏掉的层/问题token。不要只取Dense轨迹query：失败方法的隐藏状态可能已经漂移。

然后在同一冻结接口、同一实际query上计算：

- 源块的真实 max score、真实 logmass、RoPEMean score；
- Quest和RPEE给源块的排名及阈值margin；
- 实际挤占其预算的false-positive blocks，它们的 `U_b-max_j score_j` 分别有多大；
- RPEE是否通过**减少错误竞争块的角点膨胀**让源块入选，而非只是把所有分数整体抬高。

这是可修复结果的机制核对，不能单独当成能力结果。如果源块的真实max本身也远低于阈值，或PCE/RPEE没有修复其实际排名，这条extremal路线没有得到当前失败支持，不能因CPU性质启动整组GPU臂。

### 必需的位置归因控制：相同算法，相同byte，错误的pair

将 rotary 维做一个固定seed、与答案无关的随机配对，仍执行完全相同的二维PCA矩形/圆盘与评分。真实pair与随机pair都缓存3D、都保留实际keys的有效上界、都无query calibration。只有真实pair在 native RoPE 作用下逐组闭合，因此有上述严格共同位置平移性质。

这比“RPEE胜Quest”更有区分力：如果真实pair与随机pair性能相当，则收益可能只是一般二维range建模，不能声称修复了位置兼容性。应预先固定随机配对，不搜索最好/最坏的pairing。

### 通过机制核对之后的唯一GPU方法比较

在已经确认的失败及配套未失败/错误记录控制样本上，关闭所有oracle注入，直接运行 **Quest、真实pair RPEE、随机pair RPEE**，从首question token开始同预算、原reader完整生成。复用已完成Dense/RoPEMean/oracle结果。关键判断是RPEE是否**自主找回源记录并恢复完整答案**，且未把原本正确的控制改错。

若Quest已经修复全部而RPEE没有增量，最多说明extrema优于均值，不构成独立方法贡献。若RPEE只改善冻结排名、不恢复输出，也不能称成功。后续迁移必须复用同一冻结构造，而非新模型换方向或格式F1矩阵。

共同旋转测试可以先完全在冻结selector接口完成：同时旋转q和K只是同一attention目标的坐标变换，无需超出原生窗口、改position表或重新跑模型。只有在真实失败上确实出现有后果的选择差异后，才有理由把该结构写成论文主张。

## 交付边界

本轮实际完成了五篇一手正文核对、单一方法修订、算法与上界/对称性推导，以及上述正反CPU性质检查。没有启动GPU、下载模型、训练、改reader或宣称真实回答改善。相关网页临时副本在 `/tmp/{triattention_2604.04921v1,quest_2406.10774v2,cobs_2607.09052v1,prism_2602.08426v2,saap_2502.08246v1}.{html,txt}`。

---

# 以下为第一轮推导历史，主推荐已由上方第二轮替代

# 独立方法判断：用正权重、真实 key 的块内混合摘要替代静态相位分组

日期：2026-09-08。主任务硬期限：2026-09-09 00:13:46 UTC。
状态：独立推导和 CPU 反例；**没有新的真实模型结果，没有录用或成功保证**。

## 结论与推荐

我最建议立刻实施的单一构造是 **Reader-Metric Block Coreset（RMBC，暂名）**：在每个物理 B=64 块、每个 KV head 内，对**实际 post-RoPE keys**做确定性 R=4 farthest-first 分组；缓存各组实际 key 的均值与实际计数，以 count-weighted logsumexp 评分，最后读取原始物理块 K/V。只改变 PSR 的分组度量，保留其正混合评分、缓存界面和 reader。

这不是认为 PSR 全路线失败。它针对当前已经测到的具体问题：Qwen3.5 样本的 phase variance=.023、content residual=2.929；Qwen2.5 的 phase=1.937、residual=2.963，PSR 与连续组的剩余方差几乎相同。纯 offset 相位度量没有利用决定实际组内 score 分布的大量信息。RMBC 把 Pro 推导中真正需要小的 `||y_s-y_t||` 作为距离，而非只控制其一个三角上界项。

还有一个直接源码/结果观察：已保存 Qwen2.5 的 PSR 标签是连续区间（时间顺序组大小 7,13,20,24），现有 count-matched Contiguous 的时间顺序是 7,24,20,13。这里不是“相位组能神奇跨越周期”的实验条件，而主要是两个连续切分边界。这个观察进一步解释了局部结果相近，但不能推广到其他模型/长度/频率。

**贡献诚实边界：** farthest-first、key clustering 和 Jensen 界都不是新算法。若它只胜 PSR、不胜同成本连续组或 pre-RoPE metric 控制，不能称独立位置编码贡献。如果它在原生 reader 上持续胜过这些控制及强方法、同时真正减少 KV 读取，则可形成“摘要度量必须与 reader 的内容—位置联合几何一致”的方法及系统证据。只换名字不足以支撑论文。

## 数学与可实现细节

令一个块中实际 reader keys 为 `y_j`，包括已有 norm、partial RoPE、原生幅度与频率；`qbar` 包括实际 attention scaling。距离直接取

`d(s,t)^2 = ||y_s-y_t||_2^2`。

对每个 KV head 单独执行：

1. FP32 求块均值 `mu`；首中心取离 `mu` 最远的真实 key，ties 取最小 offset。
2. 维护每个 token 到已选中心的最小平方距离。每轮选择当前最远 token，共最多 R=4 个中心。
3. 将 token 指派给最近中心，ties 取最早中心。若全部剩余距离为零，提前结束；这是重复 keys 的合法退化，不通过复制空中心凑数。
4. 每组保存 `mu_r = mean(y_j, j in C_r)`、整数 `n_r`。**均值必须由真实 post-RoPE keys 得到；不能把未旋转均值旋到一个伪中心位置。**
5. 评分 `Fhat_b(qbar) = logsumexp_r(log(n_r)+qbar dot mu_r)`。空 padding 的 log count 为负无穷。
6. 按既有物理 B=64 块预算选择，精确读取原 K/V；sink/local/causal mask 原样保留。

不做 Lloyd 迭代、query calibration、频率权重搜索或训练。构造始终 question-blind。分组 labels 只在构建时使用；reader 不重排 token。对 GQA 只为 KV heads 建摘要，不重复为 query heads 存储。

流式实现：块一旦满 64 token 就构建一次；未满块及最新 2048 local token 由原始 local 路径读取。在既有 local>=B 的契约下，任何参与 remote 排序的块都已完成，不需要每个 decode token 重建全前缀。摘要构建为 O(N R D)；缓存为每块 R*D 个数加 R 个 counts；query 评分约为全 key dot products 的 R/B=1/16，外加小 LSE。以相同 dtype 计，R=4 是 K cache 的约 6.25%，或完整 K+V 的约 3.125%，不含 counts 与临时构建内存。**构建时间和实际延迟仍须测量**。

它保留共同平移等变性：原生固定旋转表下，共同平移 Delta 使所有 `y_j`、`qbar` 乘同一正交 `R(Delta)`；距离、分组、counts 不变，均值一起旋转，所有评分不变。这里不涵盖动态改变频率表的情形。

## 为什么正混合比“低 key MSE”更可靠

每组均值是条件期望。因而对任何 query，

`sum_r n_r exp(qbar dot mu_r) <= sum_j exp(qbar dot y_j)`。

它不会凭空制造超过真实 block mass 的指数质量；R 个独立真实 key 模式时它可以完全精确，且不限于恒定内容或单个旋转轨道。静态相位组若把这些模式混合则一般不精确。

若 `rho_r=max_j ||y_j-mu_r||`，则每组 gap 满足

`0 <= J_r <= min(||qbar||^2 rho_r^2/2, ||qbar|| rho_r)`。

Farthest-first 在实际 key 欧氏距离下给离散 k-center 半径的标准 2-approx；中心替换成组均值后，半径至多再乘 2。因此它控制的是实际 post-RoPE key 的最坏误差，而 PSR 的 2-approx 只作用于单位相位 orbit。这个界可能很松，**不是 block top-k 或 QA 的 2-approx**。

两块的低估差异仍可翻转排名。方法没有“下界更紧必然排名更好”保证，必须测漏选块 margin 和完整生成。

## 对主任务轨道回归候选的独立检查

主任务给出的复数逐 pair 构造为 `phi=(z-mean(z))/sigma`、`mu=mean(y)`、`beta=mean(conj(phi)*y)`、`yhat=mu+phi*beta`。我认为这有可解释的结构，但不应只检查 Frobenius residual。

### 一个有用的等价式

设块中心坐标中 `z_j=exp(i omega t_j)`，`m=mean(z)`。则

`beta = [mean(conj(z)*y) - conj(m)*mu] / sqrt(1-|m|^2)`。

在理想原生旋转 `y_j=z_j*u_j` 下，`mean(conj(z)*y)=mean(u)`。因此它本质上由**同一块的 RoPE mean 与中心坐标下 NoPE mean 这两个统计量**确定，是 span{1,z} 上的最小二乘重建。这比“Prism 对单均值重新放大”多保留了一份独立统计量；但是否已有同类双均值或 Fourier sketch 工作，需要一手查新，不能据此宣布首次。

### 确凿的反例：投影制造 ghost peaks

正交回归的 token-space 投影矩阵含负/复权重，并非条件期望。它可在降低均方误差的同时，把原 key 范围外的虚拟峰值送进 exp。

CPU/NumPy 检验：B=64，`z_j=exp(2*pi*i*j/B)`，令

`a_j=1+conj(z_j)`，`y_j=conj(a_j)/|a_j|`，在 j=32 的零分母处设 `y_j=1`。

所有原始 keys 的模长为 1，沿实轴最大值为 1。对主任务的完整回归式，虚拟 key 沿实轴最大值为 **1.272983871**。均方中心化能量从 **0.574743502** 降到残差 **0.189267751**，确实改善约 67%。但是取实轴 query `qbar=20`：

| 块 | 真 log mass | 回归 log mass |
|---|---:|---:|
| A：上述 64 个 keys | 22.525078826 | 27.450586177 |
| B：64 个恒定 key 1.1 | 26.158883083 | 26.158883083 |

真实 top-1 是 B，回归 top-1 是 A。qbar=50/100 时同样错，差距更大。这是算子反例，**不表明自然激活一定发生或路线不可行**。

对应真实激活必须额外记录：`Fhat-Fexact` 的正尾、虚拟 score 是否超过原 block max、正误差引发的 top-k false positives。只看平均 residual 降低或平均 log-mass error 容易遗漏这个失败通道。

不建议用任意温度系数修复 ghost peaks，也不建议立刻把 clipping 作为新主方法。clipping 对实际 keys 的最大 query score 若要精确计算，本身可能重新扫描全 keys；只夹 key norm 不能保真 block mass。RMBC 的正混合从构造上排除了这个具体问题，同时能处理当前主导的内容残差。

## 必须面对的强反例与必要比较

1. **无低复杂度 key 模式。** 64 个近正交 keys 且 query 命中某个未孤立方向，R=4 必然丢掉尾部；Quest 的范围上界可能更适合这类异常 key 检索。
2. **query 看不见的大方差方向。** 欧氏 k-center 可能将代表预算花在 query 实际不使用的方向。不能仅以 key MSE 判定成功；实测 native query margin 和回答决定是否有效。
3. **高 mass 与关键答案不等价。** 现有 exact RoPE oracle 的改善没有稳定转化为真实答案；即便 RMBC 接近该 oracle，也可能没有 QA 收益。
4. **位置贡献控制。** 对 pre-RoPE keys 做完全相同分组，但仍以各组的真实 post-RoPE means 评分、原 reader 读取。它与 RMBC 只差分组 metric 是否保留最终显式旋转。若两者相当，则主要收益属于内容自适应摘要，不能冠以位置必要性。
5. **同容量控制。** 连续 R=4 与随机 R=4 均使用同样均值、counts 和 reader。对 RMBC 的每块实际 counts，分别构造 count-matched 连续/随机 labels，避免不等组大小造成解释混淆。PSR 原始 R=4 也应保留。
6. **最近邻。** 按已读计划，SAAP 已有 NoPE 上的全局聚类和学习索引，COBS 已有均值/协方差 mass 近似，Prism 已有 post-RoPE pooling 频谱校准，Quest 已有页级范围检索。因此只能声称物理块内、无需训练、原生 reader metric 上的正 coreset 构造与实证增量；不声称“首次 key clustering/位置感知/softmax mass sketch”。主论文至少需要 Quest 与 COBS/Prism 中最近且可复核的强实现，不能把未来比较写成已完成。

## 两小时内的决定性验证：实现方法并完整生成，不再用几例诊断替代

现有 mean 两臂的已归档总运行时是 Qwen3.5 48 个输出 **151.4 秒**、Qwen2.5 48 个输出 **236.8 秒**。这只是当前参考程序的已完成运行时间；新缓存构建开销要测，但数据说明两小时的主要瓶颈不应是再跑 24 个完整自然回答本身。

建议主任务将已经计划的真实激活 CPU 检验限制在选定构造的诊断作用，同时直接实现可缓存 RMBC，完成以下单一推进链：

1. 现有 24 个自然问题上运行 RMBC、count-matched 连续控制、pre-RoPE metric 控制；Dense/exact oracle/单均值已完成结果复用。所有问题从第一个 token 稀疏，独立前缀 cache，原生频率，完整 token+EOS/cap384。这里是开发比较，hotpotqa_77 只作预先识别的修复例，不能据它报告泛化率。
2. 验证的不只是 F1：逐例对照完整答案，分开实体/事实修复、错误引入和纯措辞差异。若排序 fidelity 改善而真实语义回答不变，不能把中间量写成能力结果；继续用已经存在但尚未用于开发的自然数据检验，而不更换模型。
3. 现有 archive 的联合长度筛选共有 **103 HotpotQA、10 Qasper、61 MultiFieldQA**，已用各 8，共留下 **150 个未用于当前开发的样本**。保持同一 source SHA 顺序、同一长度契约与模板，无需新下载。把方法冻结后在这 150 个上做完整生成；Qwen2.5 主测试可先做 Dense、连续 R=4、RMBC 三臂，RMBC 的 pre-RoPE metric 控制是位置归因必需项，应在时间允许时一起完成而非先宣称位置主张。方法不用扫描 R/topk。
4. Qwen3.5 或已准备的 MiniCPM 是已有资产上的迁移，不是下一个“换模型找胜例”。MiniCPM 直接调用原生精确 sparse reader，替换返回的物理 block indices；必须保持真实 shared-GQA budget。RMBC 和匹配控制使用相同 shared-head aggregation 和 normalizer，不把“改摘要”和“改头权重”绑定成不可归因的一臂。官方 native baseline 独立保留；参考 selector 的速度不是 fused kernel 速度。
5. 实际总读量包括摘要和原 K/V，报告 cache build 与 decode 分项耗时。若只能在固定读取预算上证明回答增益，就只写质量结果；不能先写低延迟 frontier。

这条链在剩余时间内能直接判定“真实 key 几何上的正摘要是否修复完整回答，并迁移到已有骨干”。它没有预先保证方法赢，也没有把必要结果推给下一轮下载。若轨道回归在真实激活上明显更好，也可把它作为同成本竞争构造；上面的 ghost-peak 检验应随之保留，不用理论反例越级淘汰它。

## 本次实际完成与证据定位

实际完成：只读方案、CORE_DIAGNOSIS、ACTIVE_RESEARCH_GOAL、groups.py、activation_audit.py、mean_native.py、MiniCPM 官方本地镜像与相关已有结果；推导上述单一推荐方法；运行回归 ghost-peak NumPy 反例；核对可复用样本数和既有运行时间。没有 GPU、模型下载、远端任务、真实能力实验或新查新的完成声明。

证据文件：
- `results/position_observability_20260908/psr_qwen25_activation_01/result.json`：Qwen2.5 PSR 实际 labels。
- `results/position_observability_20260908/psr_activation_01/phase_content.json` 与 Qwen2.5 对应文件、`CORE_DIAGNOSIS.md`：有限 phase/content 证据。
- `results/position_observability_20260908/mean_qwen35_01/status.json`、`mean_qwen25_01/status.json`：151.4/236.8 秒、48 完整输出与预算。
- `results/position_observability_20260908/natural_inputs_01/manifest.json`：174 eligible、24 used、150 remaining、原 archive 路径和 SHA。
