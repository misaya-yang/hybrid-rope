# 2026-08-24 GPU 实验复盘与问题 2 路线

- **状态：** 内部决策报告；不是新的实验结果或论文 claim
- **作用：** 统一解释 2026-08-24/25 GPU 窗口、恢复此前理论主线，并定义问题 2 的最短解决路径
- **权威边界：** 数字仍由各 result/evidence owner 持有；本文只拥有跨实验解释、错误复盘和下一步研究决策
- **取代：** 取代“候选静态表失败意味着单表存在内外取舍”以及“session 路由是理论答案”这两种跨实验解释

核验本报告时按以下 owner 读取，不要从本报告复制数字后脱离协议：

1. [`../../ROPE_CAUSAL_VARIABLES_AND_ZERO_TRAINING_RETROFIT_20260823.md`](../../ROPE_CAUSAL_VARIABLES_AND_ZERO_TRAINING_RETROFIT_20260823.md) — 当前变量、阶段和 zero-training 语义；
2. [`../../FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md`](../../FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md) — full sin/cos geometry、反例和 crossings；
3. [`../results/EXPERIMENT_REPORT_20260821.md`](../results/EXPERIMENT_REPORT_20260821.md) — phase-chord 两种子内部 Pareto owner；
4. [`../results/DIRECT_Z_FIXED_SUPPORT_PILOT_RESULT_20260824.md`](../results/DIRECT_Z_FIXED_SUPPORT_PILOT_RESULT_20260824.md)、[`../results/ZERO_PARAMETER_SINGLE_TABLE_RESULT_20260824.md`](../results/ZERO_PARAMETER_SINGLE_TABLE_RESULT_20260824.md) 和 [`../results/FRESH_FINEWEB_S4_GENERALIZATION_RESULT_20260824.md`](../results/FRESH_FINEWEB_S4_GENERALIZATION_RESULT_20260824.md) — 本次 GPU 窗口的三个实验结论。

## 0. 结论先行

2026-08-24/25 的实验本身有效，但实验选择偏离了最重要的未解问题。

1. direct-`z` 四文档实验只否定了一个严重欠定的冻结权重校准协议；
2. anchored/protected-band 只否定了两个手选解析候选；
3. fresh FineWeb 只确认了已有 Native/s4 session policy 的跨样本持久性，并分离了几个具体 `z`、profile detail 和 routing 对照。

它们都没有解决问题 2：

> 如何在第三轴 `z` 上直接找到同一张窗口内良好、多个 OOD 尺度也良好的表，并解释它与 RoPE 相位/碰撞的关系；随后，如何把该表以零参数或最小适配方式带到成熟大模型，而不把 co-adaptation shock 错当成 allocation 的内禀代价。

“窗口内好 + 外推好”只是必要的性能门槛，不是问题 2 的完整答案。完整答案还需要：

- 一个对 OOD/碰撞本身成立的 operator-level 目标；
- `z` 与该目标之间的可检验机制；
- 静态几何、共适应训练行为和成熟模型 retrofit 三层证据；
- 不依赖单个目标长度的构造或 frontier；
- 大模型零参数/最小适配的单独闭环。

## 1. 三个问题不能再合并

| 层级 | 科学问题 | 当前状态 | 不能偷换成什么 |
| --- | --- | --- | --- |
| P1：第三轴 | 固定 sampled support `(a,R)` 时，interior allocation `z` 是否真实影响几何和训练行为？ | 已完成：exact-range、M4、weights×table crossing；成熟模型有同 support corollary | 不能写成某个具体曲线唯一、非几何普遍更好，或 support 不重要 |
| P2：joint table | 哪个 `z` 能在共适应训练后同时保持窗口内并改善多个 OOD 尺度，为什么？ | 未完成；phase-chord 只有内部可行性证据，稳定方法与 collision→LM bridge 均缺 | 不能把 collision proxy、两个长度 NLL 或某个 prior 单独当成答案 |
| P3：大模型迁移 | 好的训练期表如何进入成熟 checkpoint：零参数硬换表、session policy，还是最小适配？ | current session-s4 是已验证 fallback；硬换表有 exact obstruction 和 candidate negatives | 不能把 route 写成 P2 的理论答案，也不能把 frozen swap 代价写成 allocation 内禀代价 |

论文主线的中心仍是 P1。下一方法增量必须先解决 P2，然后才有资格升级 P3。

## 2. 此前证据已经说明什么

### 2.1 第三轴真实，但不提供好坏标签

对任意有序非退化表，写

\[
x_k=-\log\omega_k=a+Rz_k,
\qquad z_0=0,\quad z_{K-1}=1.
\]

Exact-range 三训练种子在固定 `(a,R)` 下只移动 30 个内部频率，已经识别出 `z` 是真实训练期变量。M4 支持“分配方向可识别、特定曲线未被识别为唯一”。成熟 OLMo/Qwen same-support controls 又说明具体 `z` 在冻结 checkpoint 中仍然 consequential。

这些结果都不支持以下二分类：

- geometric = 差，non-geometric = 好；
- in-window 好 = 外推好；
- 静态 collision/rank 好 = LM/OOD 好。

Fresh FineWeb 的 8K/16K 排序反转反而说明：在同 support、gain、route、checkpoint 和 rows 下，一个 `z` 可以在较近长度更好、在更远长度崩溃。被识别的是**具体 `z × 长度` 的作用**，不是方法类别。

### 2.2 “兼得”已有内部可行性，但不完整

Phase-chord 两个训练种子的均值相对 FMRoPE 为

\[
+0.00070/-0.16051/-0.15577/-0.20522
\]

（`1x/2x/4x/8x` tail NLL）。这说明一张固定表可以做到训练窗口近似平价，同时改善所有测试 OOD 长度；因此不存在由现有证据支持的“单表必然取舍”。

但该结果不完整：

- 只有两个训练种子，且一个种子参与方法选择；
- 151.9M from-scratch 不能直接回答成熟大模型；
- endpoint 是固定长度 teacher-forced NLL，不是广义 OOD 或能力；
- phase-chord 依赖测得的 attention-distance distribution；
- chord demand 仍是 surrogate，不是被识别的 LM-risk derivative；
- 它没有建立 collision 改善导致 OOD 改善的定量桥梁；
- 它没有给出真正与单个 `L_target` 无关的最优性或稳健性结论。

所以它拥有“可行性”，不拥有“问题 2 已解决”。

### 2.3 共适应失配与 allocation 内禀代价不同

窗口内变化应分解为

\[
\Delta\mathcal L_{\rm in}
=\Delta\mathcal L_{\rm alloc}
+\Delta\mathcal L_{\rm adapt}.
\]

- `alloc` 是模型从头或继续训练后与新表共适应时的内禀变化；
- `adapt` 是把新坐标系硬装到 Native 权重上的失配。

50M 自洽 Geo/Geo 与 EVQ/EVQ 的训练长度 PPL 是 `7.14/7.16`，而交叉硬换表为 `76.20/23.05`。这说明冻结换表的巨大 1x 代价主要不能被解释成“好的 allocation 必须牺牲窗口内”。

## 3. 2026-08-24/25 实验到底回答了什么

| 实验 | 真正回答的问题 | 有效结论 | 没有回答的问题 |
| --- | --- | --- | --- |
| learned direct-`z` | 两个 design rows 能否稳定校准 62-effective-degree frozen table？ | 不能；一个 held-out 2x row 超过 robustness gate | 不否定直接优化 `z`、joint Pareto、collision-aware objective 或更充分识别的数据协议 |
| anchored/protected static tables | 两个预先指定的解析候选能否冻结权重通过 1x/2x gate？ | 不能；二者 2x 方向好但 1x 失败 | 不否定其他单表、共同训练、约束变分或成熟模型最小适配 |
| fresh FineWeb Native/session | 已冻结的 session-s4 policy 是否在新 shard rows 上保持？ | 是；具体同-support `z` 在 16K 出现巨大差异，routing 单独拥有 4K parity | 不建立 collision→OOD 因果，不寻找新的 `z`，不证明 route 必要，不解决 Qwen/模型总体泛化 |
| 1B tokenization | 新数据是否可供后续训练？ | 数据准备完成 | 不是训练、方法或结果证据 |

### 为什么实验选择会走偏

1. 按“5090 上现成代码能跑什么”排序，而不是按“哪个缺口能完成 P2”排序；
2. 从 P3 的冻结大模型 fallback 开始，而不是先冻结 P2 的 operator objective；
3. 把自然文本 NLL 当成机制端点，没有直接测 collision/recurrence；
4. 在四文档欠定优化失败后，转向两个手选解析候选，而不是回到约束目标；
5. 把 candidate-specific negative 提升成 single-table class negative；
6. 把 session routing 的工程成功提升成理论必要性；
7. 忘记 phase-chord 已经否定“兼得不可能”，也忘记它本身仍缺 collision/OOD bridge。

## 4. 问题 2 的主方法：直接优化“兼得”

### 4.1 先做硬约束，再优化最差 OOD

加权和

\[
\mathcal L_{\rm in}+\lambda\mathcal L_{\rm out}
\]

会让结果依赖一个人为 `lambda`，也允许外推收益掩盖窗口内退化。正确形式是词典序/约束优化。令 `W^*(z)` 表示在固定 recipe 下与表 `z` 共适应后的权重，定义相对 matched geometric/FMRoPE baseline 的长度风险

\[
r_s(z)=
\mathcal L_s(W^*(z),z)-
\mathcal L_s(W^*_{\rm base},z_{\rm base}),
\qquad s\in\{1,2,4,8\}.
\]

先定义 in-window 可行域

\[
\mathcal F_\varepsilon
=\{z:r_1(z)\le\varepsilon,\ 0=z_0<\cdots<z_{K-1}=1\},
\]

再解

\[
z^*=\arg\min_{z\in\mathcal F_\varepsilon}
\operatorname{RobustAgg}_{\text{seed}}
\left[\max_{s\in\mathcal S_{\rm OOD}}r_s(z)\right].
\]

`S_OOD` 是预注册的多尺度集合，而不是为每个 endpoint 重新选表。最低要求是最差 `2x` 外推仍良好；更远的 `4x/8x` 用来检查 ranking reversal 和鲁棒性。`epsilon` 必须根据 matched baseline variability 在看结果前冻结。

这才直接回答“怎样兼得”：窗口内不是 soft penalty，而是晋级资格；在资格内优化最差 OOD，而不是优化平均分。

### 4.2 不能在冻结权重上寻找训练期 Pareto

`W^*(z)` 是问题定义的一部分。冻结 Native 权重只测 `z` 与旧坐标系是否兼容，不能估计共适应后的 joint frontier。可承担发现阶段的最小实现是：

1. 50M/151.9M 低成本 inner training，使每个候选与自身权重共适应；
2. 所有候选使用 matched initialization、row order、optimizer 和 token budget；
3. 独立 1x/2x/4x/8x validation rows 计算 outer constrained objective；
4. discovery seed 只产生候选，候选冻结后用独立三训练种子确认；
5. 通过 weights×table crossing 单独估计 co-adaptation。

这比四文档 frozen direct-`z` 更贵，但它解决的是正确问题。

### 4.3 从 phase-chord 起步，但不把 prior 当答案

Phase-chord 已经找到接近 joint Pareto 的方向，因此不是应该丢弃的旧实验，而是 warm start：

- 用其 realised `z` 初始化，而不是重新从 Native 随机搜索 62 个自由度；
- 先测沿 `z_phase - z_base` 方向和其正交补的局部 Pareto 几何；
- 将 attention-distance prior 从最终构造中移除，检查 joint objective 是否仍选择相近方向；
- 若多个 seed/base/model 的 joint-optimal `z` 具有稳定结构，再蒸馏为只读取 `(K,L_train,a,R)` 或 Native profile 的固定解析规则；
- learned/search oracle 只负责发现结构，最终 fixed table 必须在独立训练中冻结使用，并如实保留其搜索 provenance。

Phase-chord 的正确下一步不是补一个普通第三种子后直接 promotion，而是先把“为什么接近 Pareto”变成可复现、无 prior 的优化规律。

### 4.4 可承担的搜索实现

不能再次直接开放 62 个 gap 自由度。先建立一个低维、平滑、单调的 joint-search space：

\[
z(\alpha)=\Pi_{\rm mono}
\left[z_{\rm phase}+\sum_{j=1}^{d}\alpha_jB_j\right],
\qquad d\ll K,
\]

其中 `B_j` 包括 Geo↔phase、phase↔EVQ/ramp 的已知方向和少量低频 spline modes；`Π_mono` 固定端点并投影回严格单调 simplex。第一轮 `d` 应保持在 `2–6`，不是再次做 62 维文档级拟合。

实际搜索使用 multi-fidelity successive promotion：

1. 50M/短 token budget 淘汰违反 1x hard constraint 的方向；
2. 对可行候选计算 worst-2x/4x/8x outer score；
3. 只将 Pareto 前沿的少数候选提升到完整 151.9M budget；
4. discovery seed 选出唯一表后锁死，三独立种子只做确认；
5. 在不同 base、`K` 和模型规模上观察稳定结构；
6. 若稳定，再将 learned/search `z` 蒸馏成不读取 OOD labels、attention prior 或 request `L_target` 的固定规则。

这样 learned oracle 回答“什么表能兼得”，蒸馏后的 fixed rule 回答“能否形成零运行时训练参数的可复用方法”。搜索蒸馏不会自动变成 EVQ-Cosh 意义上的 closed-form zero-search construction；只有独立理论推导出的规则才能使用后一个称呼。发现与确认也不能使用同一组 validation rows。

## 5. OOD/碰撞是解释与约束，不替代 joint objective

### 5.1 OOD 与碰撞不是一个 NLL 表格

“collision”必须拆成两个对象：

1. **band/subspace redundancy：** 不同频率在一个窗口内张成近似相同的 full sin/cos 子空间；现有 canonical-correlation/stable-rank theory 拥有这一层；
2. **distance recurrence：** 两个不同相对距离产生近似相同的整体或局部 rotary phase code；这是外推时的位置别名问题。

二者都受 `z` 影响，但当前没有定理说明改善一个必然改善另一个，更没有定理直接推出 LM loss。问题 2 需要同时报告，而不是继续共用一个模糊的 collision scalar。

RoPE 把整数相对距离 `n` 映射为相位算子

\[
R_\Omega(n)=\operatorname{diag}
\bigl(R_{\omega_0}(n),\ldots,R_{\omega_{K-1}}(n)\bigr).
\]

任意两个位置差 `p-q=n` 的 phase-code collision 可化为 `R_Ω(n)` 接近恒等算子。一个直接、phase-invariant 的归一化距离是

\[
d_\Omega(n)
=\frac{\|R_\Omega(n)-I\|_F^2}{4K}
=\frac1K\sum_{k=0}^{K-1}\bigl(1-\cos(\omega_k n)\bigr).
\]

`d_Ω(n)` 接近零表示相位编码在距离 `n` 发生近复现。这是 positional-code collision，不是 LM loss；它必须保持这个 claim ceiling。

全表平均仍可能掩盖局部风险：如果内容映射主要使用一个频带子集，其他频带不能替它“平均掉”碰撞。为避免先猜 attention prior，同时定义 fixed-cardinality subset-robust distance

\[
d_\Omega^{(m)}(n)
=\min_{S\subset\{0,\ldots,K-1\},\ |S|=m}
\frac1m\sum_{k\in S}\bigl(1-\cos(\omega_kn)\bigr).
\]

它等价于取 `K` 个 pair distances 中最小的 `m` 个求平均。报告多个 `m`，而不是观察结果后选择一个“重要频带数”。`m=K` 恢复全表距离，小 `m` 测试对未知内容支持的鲁棒性。

### 5.2 不依赖单个 `L_target` 的辅助诊断

固定训练窗口 `L_train`、阈值 `ε` 和预注册子集大小 `m`，定义首次 OOD 近碰撞

\[
T_{\varepsilon,m}(\Omega)
=\min\{n>L_{\rm train}:d_\Omega^{(m)}(n)\le\varepsilon\}.
\]

再报告完整多阈值 frontier

\[
\mathcal T(\Omega)
=\{T_{\varepsilon_i,m_j}:i=1,\ldots,r;\ j=1,\ldots,s\},
\]

而不是只在一个 `L_target` 上最小化某个 kernel。等价的数值视图是每个 dyadic annulus 上的最坏近碰撞：

\[
C_j(\Omega)
=\min_{2^jL_{\rm train}<n\le2^{j+1}L_{\rm train}}d_\Omega^{(m)}(n),
\qquad j=0,1,2,\ldots
\]

该辅助方向的优点：

- 不读取任务标签或 attention-distance prior；
- 不把某个请求的 `L_target` 输入构造；
- 直接针对相位近复现，而不是把“外推好”当定义；
- `z` 在固定 `(a,R)` 下直接改变相位向量的共振/复现结构；
- 可以同时看到 2x 好、4x 坏之类的 ranking reversal。

边界必须同时写清：有限维准周期编码在无限距离上会出现任意近复现；因此不能声称“永不碰撞”。`T_{ε,m}`/`C_j` 是一个待验证的 horizon-free frontier proposal，不是现有定理，也不是已证明的 LM-risk objective。

### 5.3 collision frontier 不能覆盖主目标

问题不能退化为单独最大化一个 `T_{ε,m}`。它可以作为 joint optimization 的结构 regularizer 或 tie-breaker：

\[
\min_{z\in\mathcal F_\varepsilon}
\max_{s\in\mathcal S_{\rm OOD}}r_s(z)
\quad\text{with}\quad
\mathcal T(a+Rz)\ \text{reported or regularized}.
\]

真正的晋级门槛是共适应训练后的 1x/OOD LM 与 capability。Full sin/cos window Gram、最小频率间距、phase-code separation 和 recurrence frontier 只能解释、regularize 或淘汰明显病态候选，不能预测最终 LM 排序。

这与旧的 Cosh surrogate 不同：Cosh 仍是声明清楚的凸代理下的闭式 construction；新目标研究的是整数相对距离上的多尺度近复现。二者是否一致是实验问题，不能预设。

## 6. 从 joint Pareto 到 collision/LM 机制的缺失桥梁

找到 joint Pareto 点是第一目标；随后必须验证以下链条，才能把性能结果提升为问题 2 的机制答案：

\[
z
\longrightarrow
\text{phase recurrence / subspace coverage}
\longrightarrow
\text{co-adapted positional distinguishability}
\longrightarrow
\text{OOD LM/capability}.
\]

需要三种互不替代的证据：

1. **operator evidence：** `T_ε/C_j`、full sin/cos canonical correlations、stable-rank/conditioning；
2. **co-adapted causal evidence：** 相同 `(a,R)`、初始化、recipe、rows 下训练不同 `z`；
3. **task evidence：** 1x 与多个 OOD 长度的自然 NLL，随后才是严格生成和 capability。

必须主动保留两个反例：

- bare static rank 改善而 frozen LM loss 崩溃；
- 同一个 fixed-support geometric `z` 在 fresh FineWeb 8K 略好、16K 崩溃。

新 collision 指标若不能解释或至少兼容这些反例，就不能成为方法选择器。

## 7. 第三轴怎样帮助问题 2

第三轴不是答案，而是把答案变成可识别优化问题的坐标系。

1. 固定 `(a,R)` 后，`z` 排除了“只是扩大 range/base”的解释；
2. `z` 的单调 simplex 允许搜索、闭式构造和 learned table 使用同一物理对象比较；
3. phase recurrence 由整个频率向量决定，恰好主要受内部样本位置控制；
4. in-window/OOD joint objective 可以直接写成 `z` 上的约束多目标问题；
5. 权重×table crossing 可以检验候选收益来自 table 还是 co-adaptation；
6. 对成熟模型，`z` 可用于寻找离 Native 更近、collision frontier 更好的最小移动表，降低但不消除 coordinate shock。

因此第三轴的正确方法化不是“再发明一个指数”，而是：

> 在固定或显式控制 support 的条件下，先用共适应训练的硬 1x 约束和 worst-OOD objective 选择 `z`，再用多尺度相位复现与 full-sin/cos geometry 解释该 Pareto 改善。

## 8. 大模型：零参数与最小适配是两个门

### 8.1 零参数硬换表

Exact transplant theorem 只阻止固定、位置无关、可逆 Q/K 映射对不同频率 multiset 的**精确**补偿。它不排除：

- 一个与 Native 足够接近、冻结权重下近似可用的更好 `z`；
- session-static Native/long routing；
- 新的 position/operator map；
- 模型本身对部分频带不敏感。

所以零参数第一门应比较：

1. Native；
2. 新 collision-frontier table 的 hard swap；
3. 当前 Native/s4 session fallback；
4. 相同 support/gain 的几何和 profile-detail controls。

Hard swap 若同时通过 1x 和 OOD gate，路由不再必要；若失败，只能说明该候选与该 checkpoint 不兼容。

### 8.2 最小适配

如果好表在共适应训练中通过、冻结 hard swap 失败，问题变成修复 `ΔL_adapt`，而不是重新否定 `z`。优先级是：

1. 保留完整 Native attention path；
2. 添加初始化为零的 position-dependent/additive residual；
3. 只在 OOD 请求或被识别的 layer/head 上开启；
4. 用 short Native teacher 约束 attention/context retention；
5. 用 natural long batches 学远端 source use；
6. 最后才考虑普通 Q/K LoRA 或更大适配范围。

静态 Q/K LoRA 不能被描述为能精确吸收换表；加性 operator route 是对 exact obstruction 更诚实的函数类。

### 8.3 `L_target` 边界

需要区分：

- **request-target-free：** serving 不接收实验者声明的未来长度；
- **construction-target-free：** table 不针对一个单独 `L_target` 优化；
- **unbounded guarantee：** 对任意长度永不近碰撞。

前两者可以追求；第三个在有限准周期编码上不能直接声称。`T_{ε,m}/C_j` proposal 旨在用一条多尺度 frontier 替代单点 `L_target`，但仍需证明、数值审计和 LM 验证。

## 9. 下一实验阶梯

### J0：CPU/no-GPU joint-objective 冻结

第一项工作不是重新优化 collision，而是把“兼得”变成可执行协议：

- 复算 phase-chord、FMRoPE、EVQ-Cosh 的完整 realised `z` 与已有 `1x/2x/4x/8x` rows；
- 冻结 `F_ε`、worst-OOD aggregator、seed aggregation 和 baseline variability；
- 冻结 monotone-gap `z` 参数化以及 discovery/confirmation seed 隔离；
- 规定任何 candidate 只有通过 1x hard constraint 才能比较 OOD；
- 规定同一张表覆盖所有长度，不做 endpoint-specific selection；
- 输出新 `z` 的 no-GPU preflight 和 realised-tensor identity，而不是结果。

并行的 collision diagnostics 比较 Native/Geo、FMRoPE、EVQ-Cosh、phase-chord、derived/ramp 和 counterexample grids，报告 `T_{ε,m}`、`C_j`、full sin/cos geometry、in-window separation 与 ranking reversal。它们只用于解释/病态淘汰，不单独 gate joint training，也不能把候选排成“必然更好”。

### J1：151.9M from-scratch joint-Pareto discovery 与三种子确认

只在 J0 协议冻结后，从 phase-chord warm start 搜索一个新 `z`，与 matched geometric/FMRoPE、EVQ-Cosh 和已完成 phase-chord owner 对照：

- 相同 `(a,R)`、初始化、row order、optimizer、token budget；
- 1x/2x/4x/8x natural tail NLL；
- 不按任一评测长度重新选表；
- 报三训练种子与每种子的完整长度向量；
- 同时保存 realised tensor 和 collision-frontier receipt。

Discovery seed 只能选择一次 candidate；选择后 table/code/data identity 冻结，再用三独立训练种子确认。Gate：候选必须通过预注册的 1x retention 与至少最差 2x OOD 条件；具体阈值由 matched baseline variability 冻结，不从结果倒推。

### J2：小规模 weights×table crossing

训练通过后交叉运行 Native/new weights 与 Native/new tables，分离：

- table main effect；
- weights main effect；
- co-adaptation interaction。

这一步决定成熟 hard swap 失败应归因于 table 还是 adaptation shock。

### J3：1.485B same-initialisation screen

沿仓库已验证的 1.485B scientific recipe 做一轮 matched screen；只有同时保持 in-window natural NLL/capability 并改善最差 2x OOD endpoint，才进入多种子或 retrofit。

### J4：成熟 OLMo 零参数/最小适配矩阵

在同一个新表被冻结后比较 hard swap、session fallback、additive residual/minimal adaptation。必须同时报告：

- 1x natural NLL、2Wiki、RULER family；
- 2x natural NLL 与严格 source-use/capability；
- 至少一个更远长度的诊断，但不能用它替换 2x gate；
- route/table/gain/adaptation 的独立身份；
- 完整 short-retention pass。

### J5：8B 延后

只有 T3/T4 通过，且高显存硬件和预算重新授权后，才运行 8B。不得把 8B 多种子作为默认下一步。

## 10. 代码与数据复用边界

可复用：

- `scripts/analysis/full_rope_collision_audit.py`：full sin/cos geometry 与 counterexample 框架；
- `scripts/lib/rope/fixed_support_z.py`：固定 support 的 realised-table 安装；
- `scripts/eval/optimize_olmo_fixed_support_z.py`：梯度/receipt scaffold，不复用四文档 objective；
- `scripts/analysis/attention_phase_demand.py`：phase-chord historical comparator，只能标为 attention-prior surrogate；
- 新 FineWeb-Edu 1B token corpus：训练数据资产，不是方法证据。

缺失：

- joint-Pareto objective/preflight 与 phase-chord warm-start analyser；
- horizon-free recurrence/frontier 辅助诊断器；
- 与现有 table owner 对齐的 realised-tensor manifest；
- T1 的三种子 matched wrapper；
- mature additive positional residual 的最小、Flash-compatible implementation。

实例已关机，原始 1B corpus 仍只在该实例系统盘。任何训练前先决定持久化 owner，不能把 tracked receipt 当 raw corpus。

## 11. 立即决策与停止项

1. 不再把 session routing 写成理论答案；它是 current deployment fallback。
2. 不再从两个解析候选失败推出 single-table trade-off。
3. 不再把 phase-chord 的 Pareto 点写成问题 2 已完成；它只证明可行性。
4. 不再用 attention-distance prior、static rank 或单长度 NLL 单独选择方法。
5. 不重复四文档 direct-`z`、`tau`/protected-band sweep、fresh shard002 confirmation 或现有 RULER cells。
6. 下一项代码工作只能是 J0 的 joint-objective/preflight；collision frontier 是并行诊断，不再取代主目标。
7. 下一项 GPU 工作只能由 J1 通过明确授权后启动；不得直接跳到 LoRA/8B。

## 12. 一句话回答问题 2

> 用 `x=a+Rz` 把 support 与内部配置分开，先在共适应训练下把 1x retention 设为硬约束，再优化同一张表在多个 OOD 尺度中的最差表现；collision/full-sin-cos 指标只解释和 regularize 这个 joint Pareto，而不替代它。小模型确认兼得后，再把大模型问题拆成 zero-parameter hard swap 与 co-adaptation repair 两个门。
