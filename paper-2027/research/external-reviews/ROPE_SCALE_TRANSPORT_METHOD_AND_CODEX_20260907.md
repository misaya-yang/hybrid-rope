# RoPE 频率分配新方案：由作用距离的增长确定缩放，并约束干扰放大

日期：2026-09-07。用途：研究判断与 Codex 实现输入。

**状态：这是一个新的、可否定的方法提案。本次仅运行附带的 CPU 数学检查，没有运行预训练模型，没有得到新的任务分数，也没有确认 E2 已经开训。**

## 1. 本轮决定

主方案是：从原生窗口内的真实模型前向，测量各频率槽位的相位变化在哪些距离影响注意力输出；估计这些作用距离随可见长度增长的倍率；据此构造逐槽频率，再用完整 sin/cos 的缓存重放限制原生输出损伤与无关 key 的指数分数增长。

简称“尺度搬运分配”。它保持标准 RoPE、固定 K、固定槽位，没有新增推理网络；最终只有一张 inv_freq 和一个 gain。尾段、过渡和边界由同一组统计决定。没有预设尾段一定更快，也没有优化 Cosh。

主要性能目标是相同训练或适配预算下的 Native 与真实长任务质量。零训练部署、轻量适配、从随机初始化训练均使用同一构造。此处零训练分支需要前向校准和五点缓存选择，0 优化器更新、0 任务损失反向传播；必须报告校准成本，不能称免数据、免校准或免搜索。

**不建议把旧 Z 的 1528 步 full training 作为下一笔主要研究支出。**现有 E1 没有证明它具备 4× 实际可靠性，新目标也不再以 Z 为共同构造。Z 的冻结结果与代码继续作为已有对照。四步 full probe 不能当成正式学习曲线。

## 2. 最新反馈能支持什么

本轮以用户直接提供的 2026-09-07 数据为准。旧 THEORY_EXPERIMENT_SYNTHESIS 与上轮报告中已被更新的数字、方法身份和阶段优先级不继续沿用。

相同 E1 开发面板中，Z 的 single-evidence near@16K 为 6/32，far@16K 为 1/32；MrUni 为 3/32、1/32。每组要求两个答案世界都正确且 EOS。它提示某些频率重分配值得继续研究，但不足以识别尾段过度减慢，更不能确认方法胜负。double-evidence、binding 和部分 Native reasoning 的基准能力太低，不能让这些格子的全零决定方法选择。

已有 scratch overlays 的固定 s4 与按目标 s8 排序相反，是两种部署策略下的有效结果。它们没有证明 Cosh 与强扩展全面互补，也没有否定所有非几何分配。新方法的收益需自己建立；这些旧结果不能移作新方法的证据。

可直接复用：E1 的原始答案、组级身份、官方分数、完整答案与 EOS、Native 分类结果、频率数组及 gain、模型模板、推理路径、已有 teacher cache 和 full/LoRA 执行探针。teacher 的最终 logits 缓存不能反推出本方法所需的 Q/K/V，需增加一次带 hook 的前向统计。

## 3. 对作者尾段直觉的判断

### 3.1 现有工作的边界应写准确

MrRoPE §3 保留高频，尾段统一除以 s，通过不同累计 radix 改中段；但 Appendix B.1 已经分别改变两条边界并做了实测。新稿不能声称它完全没有研究边界。[R1]

LeRoPE §6–7 已研究学得频率的冻结复用以及与 YaRN 的组合。[R2] 因而，三阶段都有频率数组、或训练分配可以叠加强扩展，不单独构成这次新方法的增量。

### 3.2 应决定的是作用距离如何增长

设某槽位在源窗口 L0 上作用于宽度 W_j 的关系；目标窗口变长后，其实际作用距离扩展为

\[
W_j(sL_0)=s^{\beta_j}W_j(L_0).
\]

在这一假设下，保持对应相位需要

\[
\nu_j=\omega_j s^{-\beta_j}.
\]

beta=0 对应固定局部尺度；beta=1 对应与总长度同比扩展的尺度；中间值对应部分扩展。这是条件化的相位匹配，并非从“尾段”标签推导出 beta<1。

频率与依赖宽度的反比关系、依赖形状随长度拉伸时插值可保持对应相位，已经出现在 Wu 等的研究中。[R3] 本提案不把这条原则当成新定理。候选贡献在于：用可测的完整输出响应估计每个既有槽位的增长率，并处理原本受抑制的干扰项在窗外被激活的问题。

### 3.3 分辨率收益与相位代价必须同时给出

完整二维位置特征的差为

\[
\|\phi_\omega(d_2)-\phi_\omega(d_1)\|^2
=4\sin^2\left(\frac{\omega(d_2-d_1)}2\right).
\]

如果距离差增长 s^beta，采用 omega/s^beta 可保持上式；统一 /s 在小相位区把这一差异能量缩为约 s^{2(beta-1)}。例如 beta=1/2、s=4 时，是原来约 1/4。这说明“统一 /s 可能损失分辨率”有一个可检验条件。

同时，全目标窗口上的总相位仍然变为

\[
\nu_j sL_0=s^{1-\beta_j}\omega_jL_0.
\]

beta<1 必然允许超出源窗口的全局相位暴露。保持相应作用距离的相位，不能保证所有干扰 key 的相位都留在训练范围内。

对同一 Native 距离 Delta，旋转变化满足

\[
\|R(\nu_j\Delta)-R(\omega_j\Delta)\|_2
=2\left|\sin\frac{(\nu_j-\omega_j)\Delta}{2}\right|
\le\min\{2,|\nu_j-\omega_j|\,|\Delta|\}.
\]

较少减速会减小未缠绕的频率位移及此上界，但不能由此断言真实网络损伤单调减小。

以历史 OLMo 几何参数 b=500000、K=64、L0=4096 举例：最慢频率在原生窗口累积约 0.0100563 rad。s=4 且 beta=1/2 时，目标窗口约 0.0201125 rad。相对增加一倍，绝对相位仍很小。因此“最慢尾段多转几圈”并不是这个配置中准确的物理描述。新构造必须读取实际 inv_freq，不使用该示意参数替代模型数组。

## 4. 可直接计算的校准统计

### 4.1 测量单个相位扰动对残差输出的影响

取一个注意力头、一个查询 t。Q/K 是实际模型经过归一化、进入 RoPE 之前的值，不能省略 OLMo 的 QK normalization。令 Delta=t-i，采用约定

\[
z_i=\frac{g^2}{\sqrt d}\sum_j
[C_{ij}\cos(\omega_j\Delta_i)+D_{ij}\sin(\omega_j\Delta_i)],
\]

\[
C_{ij}=q_{j,1}k_{i,j,1}+q_{j,2}k_{i,j,2},\qquad
D_{ij}=q_{j,1}k_{i,j,2}-q_{j,2}k_{i,j,1}.
\]

令 p=softmax(z)、o=sum_i p_i v_i、y=W_{O,h}o。只对一个 key、一个 pair 的相对相位 phi_ij 求导：

\[
a_{ij}=\frac{g^2}{\sqrt d}
[-C_{ij}\sin\phi_{ij}+D_{ij}\cos\phi_{ij}],
\]

\[
\boxed{\frac{\partial y}{\partial\phi_{ij}}
=p_i a_{ij} W_{O,h}(v_i-o).}
\]

推导：softmax 导数为 dp_k/dz_i=p_k(1[k=i]-p_i)，代入 dy/dz_i 即得到 p_i W_O(v_i-o)，再乘 dz_i/dphi_ij。

定义非负量

\[
\chi_{ij}=p_i^2a_{ij}^2\|W_{O,h}(v_i-o)\|^2.
\]

若每个被考察的 key/pair 相位获得独立、零均值、方差 epsilon² 的微扰，则该头输出变化的二阶期望为 epsilon² sum chi + o(epsilon²)。因此它具有明确的局部输出敏感度含义，并包含值向量与输出投影；不能直接把 attention mass 当成位置作用。

这不是 log-frequency 导数。此处不能再乘 Delta²，否则会先验偏向远距离。Delta² 在后续“频率变化引起相位变化”的目标中才进入。

这个量也不是“正确答案效用”。没有下游任务梯度，它不能识别某个响应对答案是有益还是有害。把响应距离视作应保持的功能尺度，是本方法的第一条待检验假设。

### 4.2 从响应形成距离分布

对固定采样设计下的文档、层、头、查询汇总：

\[
H_{j,L}(d)=\sum_{\ell,h,t,i:\Delta_i=d}\chi_{\ell h tij},\qquad
S_{j,L}=\sum_{d>0}H_{j,L}(d),
\]

\[
P_{j,L}(d)=H_{j,L}(d)/S_{j,L},\quad d>0.
\]

Delta=0 不用于估计频率缩放，因为实际换频率对相位零点没有影响。完全零响应的槽位标记 inactive，原始提案保持其源频率，不凭空设一个 least-squares cutoff。

汇总不引入人工任务权重。它使用残差输出同一物理单位下的平方响应，但跨层能量不能替代最终 logits 的真实敏感度。记录分层/分头离散度，防止将异质性误解为一条严格共同规律。

### 4.3 数据、数量和实现开销

首次 OLMo 校准使用 32 篇真实文档，来自允许的训练/校准语料，与 E1 任务选择样本及最终确认文档分开。固定 16 篇为构造集 C，16 篇为尺度预测检查集 V。

在同一文档的同一结尾锚点构造嵌套上下文，模板后可见长度为 1K、2K、4K。三种长度保留相同末尾查询 token 与局部文本，新增前面的上下文。每段预先固定 8 个尾部查询位置；对选中的查询保留全部可见 key 计算 softmax，不能仅在抽样 key 上归一化。记录实际模板、位置 IDs 与查询数。

名义前向输入量为 32×(1024+2048+4096)=229376 tokens；模板与边界按实测增加。这是输入量，不是时间预测。scratch 用 Ltrain/4、Ltrain/2、Ltrain，外部模型用自己的相对长度。

流式 hook 每层计算并归约 H；不保存所有层全部 Q/K/V。后续候选重放需要再做构造集的 Native 前向，可逐层同时算五个候选，然后释放临时张量。跨文档无关 key 使用有界 reservoir 或成对文档前向。W_O 能量可在 head 维度用 W_O^T W_O 计算。

参考实现使用 NumPy float64，只实现数学内核。Codex 需接入已有模型的 PyTorch hook，核对旋转布局、QK norm、GQA head 映射、bias、缩放和 mask。测量采集、重放、真实推理各自耗时及峰值显存；没有预设吞吐或小时承诺。

## 5. 从统计到 inv_freq：完整构造

### 5.1 用离散分位数求所需缩放

令 a=Lb/La>1，Qa、Qb 是同一槽位两档距离分布的逆 CDF。求

\[
\min_{r\in[1/a,1]}\int_0^1(rQ_b(u)-Q_a(u))^2du.
\]

展开为 r² E Qb² −2r E QaQb+E Qa²，求导得到

\[
\boxed{
 r_j=\operatorname{clip}_{[1/a,1]}
 \frac{\int Q_aQ_b}{\int Q_b^2},\qquad
 \beta_j=-\frac{\log r_j}{\log a}.}
\]

经验分布的积分通过两份累积质量序列的合并精确计算，无需伪逆、核带宽或任意分位数采样网格。区间 beta∈[0,1] 是本版主动限制：不加快到源频率以上，也不压到 full PI 以下；不是普适物理定理。报告未截断 r、截断数量与每槽残差。

形状残差为

\[
\epsilon_j^2=
\frac{\int(r_jQ_b-Q_a)^2}{\int Q_a^2}.
\]

若 Qb=a^beta Qa，r=a^-beta，且对应分位点上的完整复相位相等；一般情况下有

\[
\int|e^{i\omega_jrQ_b}-e^{i\omega_jQ_a}|^2du
\le\omega_j^2\int(rQ_b-Q_a)^2du.
\]

这由 |e^{ix}-e^{iy}|≤|x-y| 直接得到，对有限相位成立。相位缠绕可能使上界变松，它没有把真实 LM 风险变成这个平方距离。

目标倍率 s 的原始提案为

\[
\widetilde\nu_j=\omega_j s^{-\beta_j}.
\]

构造时使用 C 集的 2K→4K 统计估计 beta；独立预测使用更早的 1K→2K，见第 9 节。不要用目标 16K 的任务分数回拟合 beta。

### 5.2 共同约束有限频率顺序

不同槽位的 beta 不必单调，直接换频率可能交叉。保持槽位身份，解

\[
\boxed{
\nu^*=\arg\min_\nu\sum_j A_j(\nu_j-\widetilde\nu_j)^2,
\quad\omega_j/s\le\nu_j\le\omega_j,
\quad\nu_j\ge\nu_{j+1}.}
\]

权重由同一统计给出：

\[
A_j=S_{j,L_0}s^{2\beta_j}\int Q_{j,L_0}(u)^2du.
\]

来源：在预测距离 s^beta Q 上，频率 nu 相对保持原相位的目标误差为

\[
S_j\int(\nu_js^{\beta_j}Q-\omega_jQ)^2du
=A_j(\nu_j-\omega_js^{-\beta_j})^2.
\]

这是指定独立相位响应模型下的有限表投影；真实共享频率扰动存在跨 key、跨头和跨层的交叉项，此二次目标不是完整模型 Hessian。下一节重放保留完整相位与同层跨头加和，检查该近似的实际缺陷。

有界加权 isotonic regression 可计算这个解。它没有加曲线光滑项、没有规定两条边界，也没有规定尾段必须 m=1。允许原始 m 在相邻频段中回落，但不置换已有 rotary slots。

若解产生相同相邻频率，应如实记录。K 个内容子空间仍存在，独立频率数可能减少；不能宣称本方法自动增加 Gram rank。

### 5.3 为什么还必须检查被抑制的干扰

设证据分数 z_e，干扰分数 z_i，则

\[
\log\frac{p_e}{1-p_e}=z_e-\log\sum_{i\ne e}e^{z_i}.
\]

恢复证据位置的相位并不控制分母。LeRoPE §7 直接展示了某个主频段在训练窗内贡献负分、窗外变正的情况；只对该频段插值能避免其快速 PPL 爆炸。[R2]

因此 p 很低不能解释为“这个槽位无用”。一种负分作用恰好使 p 低，而加速尾段可能解除这种抑制。仅用 H 会遗漏它，这是必须公开处理的机制缺口。

### 5.4 两个完整三角函数重放约束

取 E1 已核验的 MrPro 实际数组作为参考 nu_R。它只作为保守比较锚点，原始提案由上面的统计产生。

第一项：Native 缓存残差输出变化

\[
D_N(\nu,g)=\operatorname{mean}_{\ell,t}
\left\|\sum_h W_{O,\ell h}
\{o_{\ell h t}(\nu,g)-o_{\ell h t}(\omega,g_0)\}\right\|^2.
\]

每次使用完整 visible keys、完整 sin/cos 和 softmax；先将同一层同一查询的所有头输出相加，再取范数，保留同时换表的跨头作用。所有缓存的输入 hidden states 固定为源模型状态；它没有覆盖深层重新前向引起的累积漂移。

第二项：无关 key 的指数分数

从其他校准文档抽取 key，与当前 query 错配，在目标区间内独立分配相对距离。保留整条频率表的分数和，而不是逐频能量相加。每查询/头估计

\[
B_-(\nu,g)=\operatorname{mean}_q
\left[\log N_q+\log\widehat{\mathbb E}_{k\perp q,\Delta\le sL_0}
\exp z_{\nu,g}(q,k,\Delta)\right].
\]

这是一个明确可计算的“无关内容”分布，cross-document 并不等于绝对语义无关；需保留这个限制。目标距离均匀或按已声明的目标 query 位置抽样；若使用分层抽样，权重只用于恢复该已知抽样分布。

对这个指定无关分布，Jensen 给出

\[
\mathbb E\log\sum_{i=1}^{N}e^{Z_i}
\le\log\mathbb E\sum_{i=1}^{N}e^{Z_i}
=\log N+\log\mathbb E e^Z
\]

（最后等式要求相同边际，独立性并非必要）。经验估计并非真实长文本的认证上界。它检查的是源 Q/K 内容统计被放到长距离时，是否产生更大的背景指数分数；真实长 hidden states 的变化仍然未知。

不把该量替换成方差或归一化后的 attention histogram：两者都可能掩盖少量很大的正分干扰。

### 5.5 最终静态表

使用唯一预先声明的五点集合：

\[
\nu(\lambda)=(1-\lambda)\nu_R+\lambda\nu^*,\qquad
\lambda\in\{0,1/4,1/2,3/4,1\}.
\]

选择满足下面两式的最大 lambda：

\[
D_N(\nu(\lambda),g)\le D_N(\nu_R,g),\qquad
B_-(\nu(\lambda),g)\le B_-(\nu_R,g).
\]

只允许浮点舍入容差。没有根据 16K 成功率逐渐放宽阈值的步骤。所有候选都在原频率槽位逐点混合；正频率、顺序及每槽上下界保持。实际最终指数为

\[
m_j^{\mathrm{final}}=-\log(\nu_j^{\mathrm{final}}/\omega_j)/\log s,
\]

不能把原始 beta 误当成最终 m。

lambda=0 要输出 REFERENCE_ONLY，表示本构造在这一校准与五点集合上没有产生新的可接受表。它不是“新方法自动获得基线成绩”的正结果，也不是普遍不存在更好表的证明。

这里保的是“两个缓存代理不劣于 MrPro”，没有证明实际 Native 已达到用户的保持要求。正式 Native 生成与最终长任务决定是否有实际增量。

### 5.6 Gain

成熟冻结与适配采用共同的 YaRN 原式 amplitude

\[
g(s)=1+0.1\log s.
\]

完整 RoPE 中，logits 倍率为 g²；s=4 时 g=1.1386294361，g²=1.2964769928。[R4] 本轮不为新方法另调温度，因果消融全部使用同一个 g。旧 Z 使用其已经声明的历史 gain，作为整体方案保留身份。

高频即使 nu=omega，其贡献仍会被 g² 改变。Native replay 包含这个效应，不能声称高频 m=0 就是功能不变。

## 6. 三阶段如何使用同一个方法

| 阶段 | 统计来源 | 构造时机 | 训练表/gain | 部署表/gain |
|---|---|---|---|---|
| 零训练部署 | 当前成熟 checkpoint 的 Native 前向 | 部署前一次 | 无模型、adapter 或频率优化器更新 | final nu，g(s)，Native/prefill/decode 全程相同 |
| 轻量适配 | 同一成熟 checkpoint，同一份校准 | 适配开始前一次 | final nu，g(s)，pure all-linear r16 | 同一 nu、同一 g，不再额外选表 |
| 从随机初始化训练 | 本次 run 自身 warmup 结束时的前向 | 只进行一次自举构造 | warmup:几何 omega、g=1；剩余:final nu、g=1 | 同一 final nu，使用共同声明的 g(s)；Native 也使用此部署设置 |

scratch 的 warmup 使用原先已规定的学习率 warmup 截止点，属于相同总预算的一部分。无需成熟外部 teacher。频率切换在优化器步边界进行，表本身不加入 Adam 状态；仍训练普通模型权重。

这是“从随机初始化出发、带一次自举分配的完整算法”，不是“第零步就有解析最优频率”的方法。warmup、校准和剩余训练的全部成本都计入。若 warmup 后的响应还不稳定、不能预测下一个 Native 长度，便是本版 scratch 构造的失败，不能偷偷延后到看起来最好的检查点。

scratch 的 train gain=1、deploy gain=g(s) 是与 Geo+扩展基线对齐的训练/推理约定，不是新增温度贡献。若实验改成训练部署同 gain，那是一项新的协议，必须对所有比较臂共同修改，不能只改本方。

零训练和 LoRA 首轮故意使用完全相同最终数组。三阶段不同的是统计所依赖的源权重与可训练参数范围，目标和求表步骤相同。新方法不再用 Z/Cosh 分别占据不同阶段。

## 7. 相对最近工作的实际改变

| 工作 | 已有决策 | 本提案新增的待验证决策 |
|---|---|---|
| YaRN / MrRoPE | 根据频率、训练窗口与边界规定分区缩放；MrPro 给出中段累计位移 | 每个既有槽位的缩放来自其测得的作用距离增长；允许尾段 m<1、边界移动或局部非单调 m；完整 key 指数分数约束可能否决加速 |
| LeRoPE | 用任务训练梯度学习频率，并分析冻结复用与 YaRN 组合 | 用前向响应的跨长度统计直接构表；对不相关 key 在目标相位上的激活作显式限制；scratch 仅用本次自举统计 |
| Wu 等尺度匹配理论 | 数据依赖宽度、频率反比、形状拉伸与 PI 的关系 | 将其原则变成逐槽有限表估计，同时公开“输出响应是否能代表应保持的依赖”的可否定假设 |

相关恒等式、softmax 导数、分位数最小二乘、isotonic projection 和 Jensen 均不是数学新发现。候选方法价值取决于新的可测决策是否在未参与构造的模型/文本上改善最终质量。仅得到漂亮 beta 图、较高 rank、较小缓存误差，不能承担论文主贡献。

本次公开原文核对未发现上述完整构造就是已有三段方法，但检索不可能证明独占新颖性。实际论文必须根据可复现效果及后续直接相关文献检查确定最终贡献措辞。

## 8. 三阶段最小实验矩阵

### 8.1 冻结部署

主体 OLMo-2-0425-1B-Instruct。复用 E1 Native、YaRN、MrUni、MrPro、Z 输出，只增加本方法。数据、模板、decoder、固定 s4、输出预算、EOS 与评分不变。

主终点为官方任务分数、完整合法答案与正常终止，同时单列 Native instruction、position/format、reasoning、text NLL。既有低基线 reasoning 保留数字但不承担保持结论。最终扩展到未参与构造/选择的真实文档与 Qwen 外部模型；外部模型使用自身 L0 和相对倍率，不能用 OLMo 的 16K 冒充其窗外。

### 8.2 轻量适配

首个成对实验为新表与 MrPro，均 pure all-linear r16、同参数范围、同初始权重、同数据顺序、相同 token 数。形成论文主表时补齐可靠 YaRN 的同预算适配；MrUni 如在有区分能力开发任务上明显强于 MrPro，也需成为主对手，不能利用冻结全零回避它。

沿用当前 8K/16K 交替 CPT，就明确 16K 已见。16K 仍是 4× Native 延展能力端点；要论证超过适配暴露长度的能力，测 32K，并声明是否仍用固定 s4。不要临时改 s8 后把结果混入固定 s4 曲线。

三条方法臂应享有相同 replay、teacher loss 和 SFT 权限。先不更改监督方案与 norm/embedding 训练范围，避免无法识别频率变化。主目标是预算匹配的最终能力；学习速度作为补充，不取代它。

不再做 full×LoRA×所有频率的笛卡尔积。full 只在新表/强对手的匹配 LoRA 都未学会时，作为一个明确的参数范围诊断。

### 8.3 从随机初始化训练

利用已有 151.9M Geo seeds137/256 的同协议结果与部署管线。若沿用原 500M tokens 预算，新增本方法相同 seeds 两臂；LeRoPE 需要一个忠实的同规模同预算训练及其 YaRN 推理比较，先做主要 seed，再根据主要比较补重复。Geo+MrPro、Geo+YaRN 可复用现有源权重评测。

所有部署表主比较采用固定 s4 跨 Native/2×/4×/8×，目标匹配策略单列。主终点为独立自然文档上的 Native 与 OOD NLL；不能赋予这个 151M 基础模型不存在的指令能力要求，也不能用其 NLL宣称真实多步问答能力。

新方法早期 warmup 与校准必须计入预算。若新总预算不同，只有匹配预算的 Geo/LeRoPE 才是主对照；不能拿短跑新模型和历史 500M 结果声称训练效率或质量胜负。不要寻找 seed42 权重，不重训 Cosh。

### 8.4 作者尾段假设的最小消融

用 E1 MrPro 实际 m=1 的槽位定义 T；剩余被新表改变的槽位为 B。B 可能包含边界外移涉及的原高频槽位，应如实标注，不能都称作原始中频。

只需要四个冻结表：MrPro、仅换 T、仅换 B、同时换 T+B。都使用相同 gain、相同源模型、相同样本。主结果是最终静态表，T/B 消融只用于解释作用来源。

逐槽替换不能排序或再次投影，否则会改变其他槽位、污染消融。即便拼接数组局部非单调，RoPE 本身仍然有效，槽位没有被置换；记录频率交叉。主方法的单调约束是一项兼容性设计选择，不是算子合法性的必要条件。

若最终表在 T 完全没有变化，就没有尾段消融；若只有 T 变化，就没有 B 消融。若所有表在当前任务都接近零，先用有区分能力的 8K single-evidence，不在 floor 上把四个消融跑满。只有最终联合表出现值得解释的行为差异，才追加两项单独消融。

## 9. 一项独立、可能失败的预测

预测对象是“响应距离是否按所估计的倍率增长”，不把优化过的缓存目标下降当成验证。

1. 只用 C 文档的 1K→2K 统计估计 beta12。
2. 对独立 V 文档，只用它们 2K 的距离分布，预测 4K 分位数为 2^beta12 Q_V,2K。
3. 用 V 的真实 4K 前向测得分布检验预测；4K 这一长度步没有参与 beta12 拟合。
4. 比较不增长、完整 2× 增长，以及 MrPro 中 m 对应的倍率。后者是供本检验使用的物理解释基线，不声称 MrRoPE 原文提出了这个响应分布预测。

每文档计算预先加权的 W2² 误差：

\[
E=\frac{\sum_j S_{j,2K}\int(
\widehat Q_{j,4K}-Q_{j,4K})^2du}
{\sum_j S_{j,2K}\int Q_{j,2K}^2du}.
\]

所有方法使用同一 2K 权重和归一化。按文档配对比较，不能把 K 个槽位当成独立训练重复。报告各基线误差、配对差与 bootstrap 区间、预言释放的尾段单独误差、截断与形状残差。

**明确的否定情形：**尺度搬运在这个未见长度步上的预测误差被简单基线稳定击败，尤其拟释放的尾段实际仍接近完整同比增长。此结果否定当前“原生响应增长能外推并指导逐槽缩放”的操作性机制，停止本方法的昂贵 scratch/LoRA/full 分支，不继续通过改 ramp、增加强度参数或换掉验证集来保住它。

区间很宽是证据不足，不等于机制被证明错误。ROI 上可以停止追加昂贵投入，同时如实区分证据不足与反证。

此外，lambda=0 是当前约束构造未找到新表；正 lambda 但 Native/真实长任务无收益，则缓存统计未转化为所需能力。即使第一项预测成立，也不能凭它越过行为确认。

## 10. 给执行者的下一次短判别实验

### 10.1 先做什么

在当前可用 OLMo 上，执行 32 文档三长度的前向统计、独立 Native 长度预测、构造集缓存重放，得到唯一 final inv_freq/gain。优先完成真实前向响应与重放的数学实现，不再重复已经通过的 full training 显存探针。

随后只增加一个新冻结臂，复用 E1 的 single-evidence compact/near/far 和有意义的 Native 分项原始样本。原有 double/binding 与低基线 reasoning 保留为已有观察，不用它们选择新表。

为补齐区分能力，加一个 8K single-evidence 桥接格：沿用 E1 的 32 个答案世界组与生成逻辑，只改变长度；比较新表、MrPro、MrUni。这个新增格仍然只是开发诊断，后续真实文档与完整官方测试另留。不能只报告选出的 Native 成功子集；可用该子集分解“本来会做的是否保住”，同时报告完整无条件面板。

若构造退回 lambda=0，无需为相同表重新跑 E1。若尺度预测明确失败，不启动本构造的新增训练。若机制预测成立但冻结 Native 有损伤，先检查同表的短 LoRA 成对比较是否能改善，不能因为零训练没有立刻胜出就宣布整个三阶段目标失败；也不能用这一理由无期限购买旧 Z full training。

### 10.2 输出字段

```text
model_id, source_checkpoint_id, template_id, rotary_layout, source_inv_freq_sha
L0, target_length, extension, physical_length, phase_span, parameter_scope
calibration_doc_ids/split, actual_tokens, sampled_queries, sampled_layers/heads
beta_12, beta_24, raw_r, clipped_slots, inactive_slots, shape_residuals
heldout_profile_errors_by_doc_and_baseline
proposal_inv_freq, final_inv_freq, actual_m, gain, gain_squared
projection_displacement, equal_frequency_count, tail_changed_slots
selected_lambda, five_replay_rows, heldout_replay_rows, status
native_category_scores, text_token_weighted_NLL
long_official_score, valid_complete_answer, termination, both_worlds_group_success
collector_wall_seconds, replay_wall_seconds, generation_wall_seconds, peak_memory
training_started_receipt_or_null
```

数组保存 NPY，逐样本答案保存现有 JSONL 即可。恢复 optimizer 与保存多个中间全参数状态不是完成这次科学判别的前置条件。E2 没有正式回执时填 null，不将成本探针写成训练已发生。

## 11. CPU 参考实现范围与已跑检查

`scale_transport.py`：单头完整 phase-response、精确离散分位数积分、beta 构造、有界顺序投影；CLI 只输出带 UNGUARDED_PROPOSAL 标记的提案。

`guard_replay.py`：完整 trig/softmax Native 重放（可按 residual_group_id 汇总所有头）、无关 key 指数分数、五点最终选择与静态表导出。

`test_math.py` 与 `math_check.json`：实际 CPU 检查，无模型结果。

已运行：相位输出导数与有限差分一致；0、1/2、1 的尺度增长指数恢复；s=1 恒等；full dilation 输出 /s；100 个有界投影与独立约束二次求解器比对；有限 phasor 上界；一个“训练内负分、窗外变正”的合成反例会被干扰约束拒绝。

模型级 hook、Flash 路径及任务改善均尚未验证。代码没有通过存放一张示意频率数组来冒充 OLMo 新表；只有真实统计输入后才能得到真实候选。

## 12. 本方案目前最脆弱的假设

第一，前向输出相位敏感度能够代表值得保持的关系尺度。第二，原生范围内测得的增长规律能延续到更长上下文。第三，在原生 hidden states 上进行完整相位重放，能预测一部分真实深层退化。第四，scratch warmup 时的统计已稳定到足以构表。

这些都没有被上述恒等式证明。选择本方案的理由是：它给出了尾段与中段共同决策的可测依据，把负分干扰被重新激活纳入构造，并让最关键假设可以在长训练之前接受一次独立检验。

若得到相对可靠 YaRN/MrPro/LeRoPE 的最终质量增量，这些机制能够支撑方法贡献；如果只有统计图和缓存分数改善，就仍未完成论文要求。

## 原始来源

[R1] Tian et al. MrRoPE: Mixed-radix Rotary Position Embedding. arXiv:2601.22181v1，§3、Appendix B.1。`https://arxiv.org/html/2601.22181v1`

[R2] Karypis et al. LeRoPE: Learnable RoPE Frequencies Improve Language Modeling. arXiv:2607.10134v1，§3、§6、§7。`https://arxiv.org/html/2607.10134v1`

[R3] Wu et al. How Data Shapes RoPE Frequency Usage: From Positional Scale Matching to Length Generalization. arXiv:2607.07678v1，尺度匹配与 self-similarity 的相关结论。`https://arxiv.org/html/2607.07678v1`

[R4] Peng et al. YaRN: Efficient Context Window Extension of Large Language Models. arXiv:2309.00071v3，NTK-by-parts 与 attention temperature。`https://arxiv.org/html/2309.00071v3`

[R5] LongReD. ACL 2025 long.524。已有 short-context restoration 工作，Native 蒸馏本身不作为本方案贡献。`https://aclanthology.org/2025.acl-long.524/`

[P1] 用户 2026-09-07 当前消息中的 E0/E1、scratch overlays 与执行状态：本轮最新事实依据。

[P2] main(20260906-144853).pdf、THEORY_EXPERIMENT_SYNTHESIS(2).md、大修与实验执行清单.md：历史协议与证据。

[P3] RoPE_ICLR2027_Cross_Audit_Theory_and_Codex_Plan_20260906.md：保留强基线/证据建议；其 Cosh 路线和适配优先次序已被当前任务取代。
