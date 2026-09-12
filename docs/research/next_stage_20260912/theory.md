# Beyond the Base：下一阶段理论研究与机制验证

日期：2026-09-12。范围：独立审查理论原件并提出下一阶段理论与机制研究。**最新用户指导：时间由用户把控，可协调更好机器、更多资源和通宵执行；不以预设资源限制科学主问题。** 文中原“6 天摘要 / 14 天正文”、D2/D6 等节点仅保留为前版背景与工作依赖参考，均不再作为自动降级、停止或缩减有效实验的条件。实际优先级由问题判别力、训练是否充分和已有证据决定。本文没有运行模型、访问远端、提交或推送；只写本文件。旧 GPU 状态不代表当前可执行性。

## 最重要的判断

现稿已经有一条成立的理论主线：**内部指数分配改变有限位置基提供的方向，学习把这些方向与内容表示耦合，具体构造能利用这个自由度改善长度行为。** 最有价值的下一步是把“方向如何供给”与“模型如何用它区分证据”连起来。无需用一个静态几何泛函包办语言模型损失，也不应把已经成立的几何和构造理论退成一串免责声明。

我的排序为：

1. **学得来源区分的有限相位机制**：用精确的局部 logit/attention odds 公式，配合一个内容匹配的相位干预，判断已有收益究竟经由哪些内容相关方向实现。对论文的“参与表征学习”最直接，数学成本低，模型测量成本中。
2. **带系数预算的有限基可用性**：把现有 block-whitened overlap 补成方向可表示性与系数代价的条件命题，检验核心网格上的实际尺度。理论成本中，CPU 成本低；它回答为什么同样的坐标数不等于同样容易使用的位置方向。
3. **连续 Cosh 到有限、有序分配的构造连接**：给出端点锚定的准确密度身份及有限网格目标，并把 constrained learnable-z 作为同一设计空间里的强比较。理论成本低至中，新增训练成本中。**若只能新增一组训练，优先把预算交给 constrained learnable-z；方向 1、2 尽量复用该组及已有 checkpoint。**

三个方向都是补强现有主线，不以取得全模型、全任务最优曲线为成功条件。摘要截止前只收已完成的命题和实测结论；新实验未出结果不阻塞已有核心主张。

## 已查原件与证据等级

本轮先读 `AGENTS.md`。下表区分“直接检查当前数学/构造文本”与“阅读历史结果报告”，避免把汇总档当成重跑证据。

| 来源（相对仓库根） | 本轮直接核查内容 | 证据身份 |
|---|---|---|
| `paper-2027/main.tex`；`sections/00_abstract.tex`、`02_exponents.tex`、`03_findings.tex`、`03_theory.tex`、`03_compatibility.tex`、`04_construction.tex` | 实际 include 链、中心主张、full-pair Gram、slow limit、crossing、构造 | 当前论文原件；`03_theory` 由 `03_findings` 引入，不能按旧稿行号判断是否使用 |
| `paper-2027/appendix/a1_proofs.tex` | Gram、基变换不变性、rank 恒等式、slow 展开、raw eigenvalues、parity lattice、跨长度反例、transplant、有限相位界、Cosh 存在唯一性 | 当前证明原件；本轮手工检查逻辑，未重跑已有绘图脚本 |
| `paper-2027/research/evidence/EXACT_RANGE_151M_3SEED_RESULT_20260820.json` | 三 seed、30 内点、499,974,144 tokens/臂、固定范围与 retargeted 差值及 provenance 字段 | 直接读 owner JSON；未取回原始训练流或重做评价 |
| `paper-2027/research/foundations/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md`（重点 §2–6）、`ATTENTION_AWARE_ALLOCATION_THEORY_STATE_20260826.md` | 完整位置基、crossing 与近似补偿、历史机制问题与终止方向 | 数学原始报告/历史结果报告；50M crossing 的库内 owner 是报告，不假称 raw-backed 重算 |
| `paper-2027/research/foundations/ROPE_OPTIMALITY_IDENTIFIABILITY_AND_CONDITIONAL_EQUATIONS_20260903.md` §5 | signed adjoint、KKT、有限训练响应、隐式微分 | 直接读理论原件；其中公式并不提供未知任务导数 |
| `ds_workspace/recon_20260910/theory/WHY_THE_FISHER_ROUTE_DIED_20260911.md`、`RELEASE_AXIS_20260911.md` §6 | Fisher 距离与真实目标分离、单槽梯度低 SNR、N 计数释放反例 | 已查判决原文；没有重新取得其模型测量，因此引用为 report-backed；文档内部历史解释有被后节修正的情况 |
| `docs/research/ROPE_QWEN15_FULL_LAG_P2_CANDIDATE_20260907.json`、`ROPE_QWEN15_FULL_LAG_P2_RESULT_20260907.json` | full-lag 构造、matched-gain 控制、结果身份和部分分项 | 直接读构造 JSON 与结果 owner，未重算全 raw；正结果不能被“静态理论都无价值”的口号抹除 |
| `paper-2027/research/external-reviews/pro-guidance-20260911/EVQ_ICLR2027_RECONSTRUCTION_GUIDE.md`（重点 §0–3、§8–10） | 原始 Pro 重构建议、constrained learnable-z 参数化与比较 | 已查 Pro 原件；它的 novelty 检索与评分预测仍是外部报告判断，本轮未联网核文献或复核评分 |
| `paper-2027/research/STORY_RESTRUCTURE_20260912.md`、`EXPONENT_CLAIM_EVIDENCE_MAP_20260909.md`、`history/PAPER_REVISION_HANDOFF_R1_20260912.md`、`PAPER_REVISION_HANDOFF_20260911.md` | 恢复资产、当前位置、来源索引、旧理论失败记录 | 导航/报告依据；读取时当前 handoff 仍标 R1，不以其旧 GPU 记录作当前状态 |

本轮只把直接检查的数学式与明确 owner 支持的事实作为出发点。C2、C42、EOS、selective-QK 的数字在本报告只用于指出证据连接，依据当前映射/重构报告；未宣称本轮审计过对应原始输出。

## 已稳固、值得更正面使用的理论

| 理论资产 | 正面叙事价值 | 还缺什么 |
|---|---|---|
| `x=a+Rz` 与固定端点干预 | 给出可以设计、识别、优化的有限表坐标；不是换一种 base 写法 | 主缺口是强比较和机制实测，非新的存在性定理 |
| 完整 sin/cos 子空间及 canonical overlap | content phase 不应决定几何度量；同样的旋转坐标数可以供给高度重叠的位置方向 | block whitening 移除了方向的尺度，不能单独判断学习/使用代价 |
| `r2=2K/[1+(K−1)c̄]` | 精确解释有限坐标如何集中成较少有效方向 | 这是同一 Gram 的谱恒等式；新价值应来自实际网格、不同 separation measure 和可用方向的测量 |
| 慢频共同二维极限及核心网格有限计算 | 使“有限旋转预算”成为具体结构：b=256、K=32、L=256 最慢八对的 rank 约 2.11，即使其 ωL 全大于 1 仍强重叠 | 极限不能冒充此有限网格的近似公式；核心数值应继续用准确 Gram |
| slot 置换补偿恒等式 + crossing | 描述基与表征的共同坐标系统；频率集和 Q/K 槽位的联合变换是精确对称性，冻结换表破坏已学对应 | crossing 已证明兼容性有作用，但不能单独定位具体层、内容系数或来源使用路径 |
| 不同谱不能被固定可逆 Q/K 变换完全吸收 | 分配是在改变可供学习的函数族，因此可能值得重新学习 | 该定理并不排除行为近似恢复；历史 geometric fit 将 Cosh 权重 PPL 从 23.05 降至 9.63，而原配表为 7.16，恰好是有用的近似机制问题 |
| Cosh 严格凸密度目标、唯一正解、逆 CDF、锚定后内点向快端移动 | 给出清楚设计偏好到可安装方法的完整构造；模型正结果验证其用途 | 真正可补的数学连接是连续密度与有限端点表，而非把 prior 重新包装为 loss 推导 |
| 最大 rank 的 parity lattice 同时有周期/反周期 | 正面展示方向数、跨尺度放置、远距复现是三个不同设计特征，提示为何需要多尺度分配 | 不宜把这一事实仅写成“我们的理论不能做什么”；也不宜声称 parity class 容量是所有表的全局上界 |

保留固定范围三 seed OOD 改善、432M MLA、750M 续训、8B 适配、成功读出与成熟部署收益，各自沿原协议。retargeting 反转不是把正结果打折，而是“学到的位置方向在什么尺度使用”这一发现的重要半边。

## 方向 1：学得来源区分的有限相位机制（首选）

### 机制与可交付命题

固定某层某 head 的 pre-RoPE Q/K 激活、causal mask 和 gain。以 a 表示含 attention scaling 的固定 logit 比例，单个 key 的 logit 为

\[
s_j(\Omega)=a\sum_k[C_{jk}\cos(\omega_k\Delta_j)+D_{jk}\sin(\omega_k\Delta_j)].
\]

换表到 Ω′ 时，**有限变化**可逐槽精确重放：

\[
\delta s_j=a\sum_k\{C_{jk}[\cos(\omega'_k\Delta_j)-\cos(\omega_k\Delta_j)]
+D_{jk}[\sin(\omega'_k\Delta_j)-\sin(\omega_k\Delta_j)]\}.
\]

给定正确证据 key 集 E 和其余有效 key 集 D，定义该 attention head 的 evidence odds

\[
M=\log\sum_{j\in E}e^{s_j}-\log\sum_{j\in D}e^{s_j}.
\]

令 p_E、p_D 分别是在各集合内归一化的 baseline softmax，则恒等式为

\[
M(\Omega')-M(\Omega)=\log\mathbb E_{p_E}e^{\delta s_j}
-\log\mathbb E_{p_D}e^{\delta s_j}.
\]

这是**可直接证明且可数值逐项验证的局部来源区分公式**：不仅区分“证据增强”和“干扰增强”，还保留跨槽相位干涉。它无需二阶小扰动、Fisher 最优假设或均匀内容权重。槽贡献对 δs 可加，但 odds 经 log-sum-exp 后不可任意线性分摊；若需要 band 归因，报告预定 band 的完整干预或明示的顺序分解，不拿单槽和替代非线性交互。

这条恒等式的价值是定义正确可测对象，不应包装成全新 softmax 定理。论文的新增知识必须来自它对实际模型响应的解释与干预验证。

### 已有证据、反例与最小实验

- 支持：50M/151.9M crossings、同谱槽位置换、成功读出/自然 QA 资产说明“供给基 → 权重使用 → 行为”值得直接测；C42/C42V24 相同支持/总位移/增量质心仍有差异，支持不能只看无符号总量（后者为开发面板，不是泛化确认）。
- 反例：静态 rank 不排序任务；N 增加的六表没有改善所测连续 NLL；Fisher 大不等于真实损失代价大；单槽 finite-difference gradient 在已有小文档面板低 SNR。因此不重开全槽梯度搜索，也不把少数 attention 热图当机制完成。
- 最小模型检验：复用一个已有、能做基本检索的 checkpoint/adapter，两个已确定的表；同一内容/答案/顺序/解码条件，比较 contiguous 与 virtual-gap 两种 position map。表若不能固定 support，明确它是部署机制研究，不能升级为纯 z 识别。gap 改变相对距离，不能只做对所有 token 加同一个 offset。
- 先用独立开发样本确定读取位置/少量 head 或层，再在保留样本验证；可用数十个已存在、短条件能成功的提示作机制子集，但要同时列出原面板筛选数和短条件失败数。最终测量单位是提示，不是 head 数。
- 在答案首 token 前记录 exact replay 的 evidence odds 和证据/干扰 contributions，随后跑同样提示的真实端到端表干预，保存完整答案与实际终止行为。受检任务要求 exact+EOS 时沿用原合同。首-token odds 只是局部机制量；长答案检索不能用它替代最终输出。
- 若 replay 与真实收益有稳定联系，再做一个预定 band 或小层组的受控干预以检验路径；先检查有序性和支持，避免历史 abrupt restore 生成非法表的故障。不新增任意 layer split 搜索。

### 结论如何分叉

1. 同内容 gap 降低 Native 来源 odds 和任务成功，候选同时恢复，且受控干预改变该恢复：可以给出“分配通过已学内容系数改善长距离来源区分”的实测机制；统计关联与干预证据分别报告。
2. 候选任务好、局部 odds 不变：保留收益，结束该层/该来源机制归因；可能是 value/readout 或其他层路径，不立即无限扩遥测。
3. odds 好、任务不改善：局部路由恢复不够完成任务，不能升级为任务机制闭环。
4. 短条件地板或真实位置无改变：实验没有触发可识别机制；修正该具体问题后才判模型，不作方法负结论。

### 依赖、成本、停止与稿件影响

依赖：checkpoint 可用性、已知表/position-ID 安装路径、证据 token 位置、可读取 pre-RoPE Q/K。当前未核机器，因此不承诺 GPU 小时；预计一套小面板推理 + 少量重放，工程成本 1–3 天。D2 前若拿不到有效激活/索引路径，降为已有 crossing 解释，优先完成方向 3 的训练。

D6 目标：精确公式及一个保留样本、内容匹配的机制判决。D14 目标：补一个干预或第二种证据位置分布确认。最多两轮针对已定位问题的修正；无稳定可检验连接就停止此机制归因，保留既有分配收益。

摘要收益：只有完整机制证据成立，才加一句分配改善 learned source discrimination；否则保持现有“weights co-adapt”。正文收益：用一幅“完整相位变化 → evidence/distractor odds → 完整输出”的图，替代泛泛的几何至性能箭头。

## 方向 2：带系数预算的有限位置基（理论主项）

### 问题与命题

block whitening 擅长比较方向，却把慢 sine 的小幅值消掉。模型要实现某一方向需要多大 Q/K 内容系数、不同方向多容易学习，是另一个可定义的问题。当前附录已有单 pair 原始 Gram 特征值

\[
\lambda_\pm=(1\pm|\operatorname{sinc}(\omega L)|)/2,
\quad\lambda_-\sim(\omega L)^2/12.
\]

应把这个已有结果提升为可操作描述，并补一个**有明确系数条件的 cluster 命题**。令 t∈[0,1]、ω_kL=εr_k，固定有限 r_k；Φ_ε(t) 拼接所有 raw sin/cos；P_⊥ 投影掉 span{1,t}。由 Taylor 展开，cos 的残差为 O(ε²r_k²)，sin 的残差为 O(ε³r_k³)，故

\[
\|P_\perp\Phi_\varepsilon c\|_{L^2}\le
\|c\|_2\,\|P_\perp\Phi_\varepsilon\|_{\mathrm{HS}}
\le B C_r\varepsilon^2\quad(\|c\|_2\le B).
\]

对任意单位范数且与 span{1,t} 正交的指定目标 h，有

\[
\inf_{\|c\|\le B}\|\Phi_\varepsilon c-h\|_{L^2}
\ge \max\{0,1-B C_r\varepsilon^2\}.
\]

交付完整常数版本及一般 separation measure 的条件：测度的低阶 moment Gram 非退化，支撑在有界区间。该命题解释**固定系数预算下，慢簇在二维主空间外能提供的信号受限**。这是供给基的条件可表达性，不是对 transformer 的全局容量或下游 loss 下界；真实 Q/K 系数是 content-dependent，跨层结构也不等同于一个固定 c。

可以进一步在明确的 fixed-feature 平方损失模型中给出谱模态学习率：GD 对某 eigenmode 的收敛因子由 λ_i 决定。该条件模型只是把 coefficient scale 与 learning dynamics 连起来，采用标准线性学习结论，不声称发现了 transformer 训练定律。

### 数值与模型交付

1. 在真正用于 151.9M 的 Geo/anchored Cosh/已有 Exp 表上，同时计算 raw Gram、block-whitened Gram、P_⊥ residual spectrum；评价固定 L_train 与原 2×/4×/8× 网格。
2. Uniform、causal triangular、一个事先定义的证据距离分布三种 measure 分列。不能挑排序最漂亮的测度；若读模型 attention 作为 measure，标为 learned、table-dependent 测度，不能再称外生几何。
3. 核心网格最慢 ωL=1.19 及以上，直接算有限矩阵，不套 ε≪1 的下界解释现有数字。慢极限只用于精确渐近检验和概念说明。
4. 在方向 1 或 constrained learnable-z 的已有模型激活上测 raw coefficient norm、目标证据与干扰 contributions，判断 Cosh 是否真以可比较的系数代价使用新增方向。无需单独训练随机玩具 transformer；若做 ridge 数值例子，明确是条件线性例子。

### 正结果、竞争解释和停止条件

支持：现有 full-pair 理论、核心八 pair rank、125M 压缩消融提供问题背景。竞争解释：Cosh 收益可能来自改变关键尺度/优化轨迹，未必来自 raw conditioning；高有效 rank 的周期表有远距复现，已有静态尺度反转说明 rank 单指标不够。宽度/压缩五配置也不支持“pair 越少收益必然越大”。

D3 前完成常数界/有限计算；成本以 CPU 和 1–2 天理论工作为主。D6 前必须能说清哪个指标在实际网格变了、哪个没有变；D14 只在有模型连接时把“更容易使用”写成实测结论。

停止条件：若有限网格变化弱、跨 measure 不稳、或者系数观测不支持预测，保留条件命题和准确几何作为附录，不再搜索第 N 个 scalar score 来重现模型排序。对摘要最多补“allocation controls directional overlap and coefficient cost under a finite-window model”；没有实际模型支持时不写 “explains the loss gains”。正文可用一个合并图同时表达供给方向与使用尺度，比再增一条有效 rank 曲线更有价值。

## 方向 3：有限分配构造与 constrained learnable-z

### 真正的推导缺口

Cosh 的连续目标、存在唯一性和 inverse CDF 在当前附录已齐；再次证明 Euler–Lagrange 没有边际价值。真正未完全说透的是：连续 prior 的 optimum，经 midpoint 取样和端点锚定，是否仍是原 prior 的精确最优对象。应直接给出安装后的对象，借此增加构造清晰度。

**交付 A：锚定的精确密度身份。** 设 Q 是任一正密度 ρ 的 quantile，u₀=1/(2K)、u₁=1−1/(2K)，q₀=Q(u₀)、q₁=Q(u₁)、d=q₁−q₀。锚定中点表

\[
z_k=[Q((k+1/2)/K)-q_0]/d
\]

恰是以下密度在 inclusive 节点 k/(K−1) 上的 quantile：

\[
\widetilde\rho(z)=\frac{d}{1-1/K}\rho(q_0+dz),\qquad z\in[0,1].
\]

证明来自先将 ρ 条件化到 [q₀,q₁] 再仿射变换。这是**截取后的连续分配与有限端点表的精确连接**，可解释 midpoint/anchored 的区别，不把锚定后的表未经证明称作原密度目标的离散最优。

**交付 B：相同设计偏好的一个严格有限目标。** n=K−1，h_i=z_i−z_{i−1}>0、Σh_i=1；每个区间分配 1/n 质量，形成分段常数密度。把它代入当前 Capp，得到完全有限的目标

\[
J(h)=\frac{\alpha}{2n^2}\sum_{i=1}^{n}\frac1{h_i}
+\frac{\beta}{6n^2}\sum_{i=1}^{n}h_i
[(n-i+1)^2+(n-i+1)(n-i)+(n-i)^2].
\]

第一项严格凸且边界发散，第二项线性，故在 simplex 内有唯一解。写第二项系数为 b_i，则

\[
h_i=\sqrt{\frac{\alpha/(2n^2)}{\lambda+b_i}},
\qquad\sum_i h_i=1,
\]

λ 由一个单调标量方程确定，λ>−min_i b_i。β=0 精确恢复等距。b_i 随 i 下降，最优 gap 向慢端增大，与 Cosh 的分配方向一致。这里“每 cell 等质量”的有限密度模型是明示选择，不是声称原经验点质量可直接代入 ∫ρ²。

这提供一种可独立验证的 finite-budget construction；先比较它与现有 anchored Cosh 的偏差、Capp 及有效方向。**不要仅因为多得到一张表就追加训练臂。** 若不同 K/τ 上几乎一致，其主要价值是说明现有实现；若差异显著且预测明确，正文截止前才考虑它与已有 Cosh 的一次匹配检验，不替代强 comparator。

### 与直接学习的比较

Pro 原件 §8.1 建议的 constrained learnable-z 是最值得新增的训练比较，因为它在相同端点、pair 数、排序空间里直接优化任务，而不是让最近邻文献的全部范围变化混入主变量。

- 建议 softmax gaps：p_i=softmax(v)_i、z_k=Σ_{i≤k}p_i，i=1…K−1。其 intrinsic dimension 为 K−2；固定一个 v 或减去均值消除平移 gauge，不把 K−1 个存储标量冒充 K−1 个独立 allocation 参数。
- 明确可检验 Jacobian：∂z_k/∂v_i=p_i(1_{i≤k}−z_k)。这便于检查 endpoint exactness、梯度链和真实位移；过小 gap 导致梯度饥饿是实现问题，不能靠“训练完了”判定学习比较有效。
- 最少三臂：fixed Geo、fixed anchored Cosh、Geo-init learnable-z；复用可比旧基线，只新增学习臂。若与原训练合同不兼容就承认必须补对应基线，不能拼接其他预算的结果。Exp 尽量用现有证据，不增曲线 sweep。
- 记录中间 z(t)、training/held-out NLL 和原离散 OOD 长度网格；evaluation 长度不反向调 LR 或选择 epoch。loss-driven learned-z 优于/劣于 Cosh 都是对该训练合同的可解释结果，训练窗口变好而外推不变尤其能区分“优化窗内”与“设计长度行为”。
- Geo-init 单臂不足以严格区分 Cosh 的初始化先验与最终表效应。如主问题需要，可在共享 checkpoint/预算下追加一个 Cosh-init learned-z，或只作两表 frozen crossing；不要把没有这组因子的比较称作中介分解。
- 新训练可提供方向 1、2 所需 coefficient/phase 观测；不另开一套理论玩具模型。

### 依赖、成本、反例与稿件影响

交付 A/B 的证明与 CPU 数值约 1–2 天；constrained learnable-z 需已验证训练器、梯度实现、可比 baseline 与足够训练预算，成本为一组学习臂而非 128-step smoke。现有 151.9M owner 是每臂约 500M tokens/7629 steps；这些是科学合同参照，不是当前硬件耗时承诺。三 seed 对照的完成时间由主代理实测安排。

历史 handoff 指向早期 learnable-τ softplus 死区和 Algorithm 1 离散化伪影：二者要求核查梯度/网格，但本轮未读原始 2 月日志，不能当成当前 learnable-z 实现已失败。Cosh 的正训练结果和 matched Exp 结果继续保留；反面约束是 τ 经验规则不等于从 loss 推导最优，finite-grid 泛函改善也不等于模型任务改善。

D2 交付有限目标/公式核验；D3 前落实训练合同，D6 若结果未完整则摘要只陈述已有解析构造。D10 前完成主比较，D11–14 整合与复核。停止条件：梯度路径故障先修具体错误；完整训练后若 learnable-z 与 Geo 不分辨，报告该合同下的结果与位移，不无限加 LR/形状扫描直到赢；若有限目标表与 anchored Cosh 近等，则不新增其模型臂。

摘要影响：若 learned-z 或有限表真有匹配模型收益，可把“可构造”提升为“解析与直接学习在受控分配空间的比较”；否则不增承诺。正文影响：一个准确的安装命题、一组强 comparator、可读的 z 轨迹，把方法从孤立 Cosh 曲线变成同一有限设计空间里的研究。

## 前版 6 / 14 天依赖顺序（历史参考，不作为执行约束）

| 时间 | 理论交付 | 与模型工作的最小连接 | 决策 |
|---|---|---|---|
| D0–2 | 方向 1 exact replay/odds；方向 3 anchor identity/finite objective；方向 2 明确命题条件 | 找可复用 checkpoint/基线，完成 learnable-z 梯度和 endpoint 检查 | 不以 CPU 公式通过宣称模型成功；确定只有一套新增训练主合同 |
| D3–5 | 实际网格 raw/whitened/residual 三类计算；有限构造差异 | 已有模型小面板 phase intervention；新训练继续 | 选能改变主线认识的机制证据，不增加表族 |
| D6 摘要 | 已成立定理 + 已完成实测 | 新训练未完成则不依赖它 | 用现有正结果锁定主张，保留修订空间 |
| D7–10 | 机制保留样本判决；学习轨迹对照 | 完成主 comparator，必要时单项机制干预 | 只有明确机制分叉才扩一项；不铺全模型矩阵 |
| D11–14 正文 | proof/figure/claim 对应、来源分层、写作整合 | 仅补直接影响结论的核验 | 推导闭环与任务闭环分别验收 |

优先舍弃：重新寻找 universal scalar loss functional；全槽 Fisher/KKT 扫描；从覆盖 N 倒推能力；宣称静态理论已经被“穷尽”；为 θ、K、L 的经验 τ 规则补虚假的最优性证明；为了提高理论数量新增架构/算子旁线。以上不否定已成功的 Cosh、BM、FullLagP2 或 C2 构造，而是把其正结果用于约束真正需要解释的机制。

最终成功标准：至少一条新连接让读者更具体地知道**内部指数改变了什么、学习怎样使用它、何种构造在何种条件下有效**。公式、CPU 计算、编译通过只是交付条件；没有模型证据的条件定理，保持其精确理论身份。

## 资源充分时：共享训练合同的判别性机制实验（最新补充）

资源优先用于**配对重复、训练充分、机制干预和独立确认**。最值得回答的问题是：不同分配是否让模型学到不同的内容—位置耦合，而且这种差异是否因果地承载长距离来源使用。单纯增加 profile 数不能回答它。

### 共用一个主合同

核心三臂是 paired fixed Geo、fixed anchored Cosh、Geo-init constrained learnable-z；共享每个 seed 的权重初始化、数据顺序、优化器、充分 token 预算、精确 sampled support、pair 数与评价内容。资源充足时，优先补 **Cosh-init constrained learnable-z**，形成“初始分配 Geo/Cosh × 分配冻结/可学习”的 2×2。它区分初始构造、学习自由度与训练路径，较多加一种手工曲线更有信息。保存事先指定的早/中/末训练 checkpoint，避免只在最终赢家上讲机制。

同一 checkpoint 上做 runtime-table crossing；这是兼容性干预。再对同一内容、答案和 distractors 做预定证据位置/virtual-gap 干预；这是相对相位干预。两者的角色分开：真实长输入检验长度泛化，内容不变的 virtual gap 定位相位因素。训练或适配数据若含长目标相位暴露，需单独标注；不能把“物理训练窗口短”自动写成“从未见过长相位”。任务优先复用已有能被该量级模型学会、存在准确证据位置的生成任务。若 scratch LM 没有基本任务能力，就把自然文本 NLL 和另一个匹配适配合同分别呈现，不借成熟模型的能力替它完成主合同。

### 从 odds 探索走向因果路径

1. **开发阶段只定位候选路径。** 在预先划分的开发样本上，计算完整有限相位贡献，找少量能解释表差异的 band/层组；记录同一证据 token 集、全部有效 keys、系数尺度和正确/干扰 odds。按机制解释选组并冻结规则，随后在保留样本及配对 seeds 上验证。层/head 数不当独立样本数。
2. **双向移除与恢复。** 在同一个权重模型里，用 exact band replay 将候选运行表在已选路径的 rotary contribution 替换为基线表贡献，检验候选的长距离收益是否减少；反向在基线运行表中仅安装该路径的候选贡献，检验是否恢复部分收益。必须对该路径的全部有效 keys 施加同一频率变换，不能只抬高已知正确证据 logit，也不能利用答案标签决定 patch 幅度。证据标注只用于事后计分和 odds 分解。真正施加的是可定义的 rotary 计算替换，不是“把 attention 调到正确答案”的 oracle。
3. **必要对照。** 同表重放作为 sham；按事先规则选择的其他 band/层组作路径特异性对照；分别报告短距离与长距离影响，并在共享训练合同的另一配对 seed 上复现。用有序的完整 runtime 表做部署对照；混合层/band patch 是机制干预，身份单独标注，不能混称为纯全表 z 方法。每次干预重新构建对应 KV cache，生成过程中保持该干预，避免只改 prefilling 的一帧就解释整段答案。
4. **主终点是原任务输出。** 比较 intact、移除、恢复、sham 的配对完整生成成功与该任务要求的终止合同，另列答案 NLL、来源 odds。若研究自然文本，使用原位置分层 NLL；不能由检索 accuracy 代替语言建模收益。结果单位以提示/文档及训练 seed 分层。提前确定有意义的效果幅度，使用原面板变异规划保留样本量；不预设少到必然没功率的 tiny panel，也不无依据要求整个 benchmark zoo。

最有判别力的结果是：**候选收益在指定路径被移除时下降、在反向恢复时部分重现，sham 无效，短距/远距模式与相位预测一致，并在独立样本或 seed 保持。** 这支持该路径因果参与，而不是“某个 rank 与分数相关”。若效应只在所有运行表下普遍破坏模型，应解释为通用重要性；若 odds 恢复而答案不恢复，说明局部路由不足以解释任务成功。即使双向结果成立，也不自动报告自然间接效应百分比：跨层交互和激活分布变化仍需额外识别假设。对不同训练臂重复这种同权重路径干预，才可具体讨论 learned compatibility 的差别；只做 frozen swap 不能独自证明训练过程中的中介因果链。

### 新理论的知识增量，按价值而非数量安放

| 候选命题 | 本身增加什么认识 | 主文 / 附录决策 |
|---|---|---|
| 系数预算下慢簇离开 span{1,t} 的信号界 | 区分“whiten 后有方向”与“有限系数能实现多强方向”；给出可证伪的使用代价预测 | 有价值的条件理论，但 Taylor+Cauchy–Schwarz 不应冒充独立重大定理。只有实际有限网格及模型 coefficient/use 证据支持其解释力，才在主文突出；否则作为附录支撑现有位置基理论 |
| 锚定密度的精确身份 | 精确说明原 continuous prior 如何变为实际 fixed-endpoint table，消除 continuum/finite 混淆 | 直接且有用，主要是方法/附录命题；无需用模型实验“验证恒等式”，不独立占一项 contribution |
| 有限 equal-mass cell 目标与唯一解 | 在明示离散化下给出真正有限 K 的构造和一个标量求解，可能揭示 K 依赖及现有 anchored Cosh 的近似关系 | 若只重新导出几乎同一表，作为附录，价值是实现解释；若预测有限 K 下系统差别且匹配模型实验证实，再作为正文设计发现。只降低 J 不构成更好 PE |
| finite replay / evidence-odds 恒等式 | 提供正确的来源—干扰响应测量对象，避免无符号 sensitivity 归因 | 恒等式本身是工具，正文新知识来自双向因果干预的实测结论；不可用一条漂亮公式代替验证 |

资源充分也不改变终止逻辑：停止的是被有效实验反驳的具体预测、已没有设计判别力的诊断或重复盲扫；不因预设天数到期缩短有效训练。训练故障、任务地板、干预未触发路径属于待修具体问题，不把它们当成理论失败。既有训练与部署正结果继续独立成立。

## 本轮完成的 CPU 恒等式核验

为避免计划中的公式自身有误，使用标准库 Python 做了独立内存计算（未写实验数据、未运行模型）：固定 seed=20260912，K=8，α=1、β=4，随机正 gaps；分段常数目标解析式与每 cell 10,000 点中点积分绝对差 `3.40099e-11`；τ=2 的锚定条件 CDF 恒等式最大误差 `1.38778e-16`；9 个随机 logits、3 个证据 key 的有限 evidence-odds 恒等式误差 `5.55112e-17`。这些结果确认这里提出的有限目标和重放公式的代数一致性，不是模型机制证据。完整证明、实际 K=32 网格及模型判决仍是下一阶段交付。`git diff --check -- docs/research/next_stage_20260912/theory.md` 通过。
