# EVQ-Cosh Rebuttal Playbook — 2026-07-22

> **2026-07-16 fact gate:** 任何 reviewer response 都必须先经过 `FULL_PAPER_INTEGRITY_AUDIT_20260713.md` 与 `FIRST_PRINCIPLES_REBUTTAL_REASSESSMENT_20260716.md`。旧 DAPE、official-YaRN、native-Geo、ordinary-KL、`c_coll`、Phase16-27-config、LoRA-rank、MLA-`d_eff` 或“pure-shape control”口径不得从历史材料恢复。

最后核对：2026-07-16

- 工作模式：`triage-only`
- 决策状态：`unclear / high-risk trust repair`
- Response package：`needs_real_reviews + needs_author_input`
- 当前发送状态：**NOT READY**。真实 NeurIPS reviews 尚未收到。

本文件是 rebuttal 的唯一操作入口。它只做 claim disposition、score-driving risk、证据准备、条件分支和作者决策门；不伪写逐条回复。旧 master ledger 只作历史风险索引，不再裁决当前事实或发送范围。

## 0. Rebuttal 的硬原则

1. **Reviewer-response-first，但不隐瞒 material error。** 普通回答只绑定真实 reviewer / AC 原话。唯一例外是：若已确认的 comparator、theory 或 reporting 错误会使 accepted record 保留实质性错误，即使无人点名，也准备一条合并、克制的 AC integrity disclosure；是否发送由作者确认。
2. **先纠错，再保留贡献。** 固定顺序为：`direct answer/correction -> unchanged fact -> withdrawn interpretation -> surviving narrower claim -> boundary -> author-approved future correction`。不能先用结果数量淡化错误。
3. **submitted、current 与 future 必须分开。** 不能用当前源码或计划中的 camera-ready 修订，写成 reviewer 看到的提交件已经正确。
4. **不把 rebuttal 当二次投稿。** 不引入新主张、新机制或新的证据层级。NeurIPS 2026 不允许上传论文/补充材料修订；每份 review 最多 10,000 characters；response 中不得放链接。官方规则以 [Main Track Handbook V2026.3](https://neurips.cc/Conferences/2026/MainTrackHandbook) 为准。
5. **实验必须由问题触发。** 有意义且高成功率的实验并不被禁止，但必须直接区分一个会改变回答的假设，具有匹配控制、任务端点、停止条件和完整 provenance。新结果可以在文字中报告，但原投稿仍是评分依据；实验不能修复方法身份或数学错误。
6. **无法关闭就明确让步。** 不用 supporting LoRA、video、750M、progressive training、单 seed 或相邻协议替代缺失的主控制。
7. **只保住窄而真实的贡献。** RoPE 的有限频率表也是 finite spectral budget；EVQ-Cosh 把 training-time frequency allocation 作为 operator design 与 inference-time range scaling 之外的第三个 PE 设计轴。

证据标签：

- `[数学事实]`：可从当前条件推出；
- `[实验事实]`：由 paper、raw artifact 或实际代码直接支持；
- `[文献事实]`：由公开一手来源支持；
- `[判断]`：内部风险判断，不是真实 reviewer 意见。

统一行动标签：`CLARIFY_EXISTING`、`SOFTEN_CLAIM`、`PARTIAL`、`ACCEPT_TEXT`、`ACCEPT_ANALYSIS`、`ACCEPT_EXPERIMENT`、`ACCEPT_FIGURE`、`ADD_CITATION`、`DISAGREE`、`OUT_OF_SCOPE`、`AUTHOR_INPUT_NEEDED`、`BLOCKING`。当前没有真实评论，因此没有条目可以升级成最终 `READY`。

## 1. 权威来源与核心 claim

### 1.1 Source of truth

| 需要判断什么 | 当前权威 | 不能替代它的材料 |
| --- | --- | --- |
| 方法身份、数学正确性、协议事实 | `FULL_PAPER_INTEGRITY_AUDIT_20260713.md` | paper wording、旧类名、旧 rebuttal 草稿 |
| 最优性、\(\tau\)、PDF可见性与三门机制 | `FIRST_PRINCIPLES_REBUTTAL_REASSESSMENT_20260716.md` | model-free collision、单次PPL或LoRA正向切片 |
| 频率最优性与 \(\tau\) 详细边界 | `THEORY_FREQUENCY_OPTIMALITY_AND_TAU_20260716.md`，与 first-principles reassessment 互相校验 | 旧 ordinary-KL 推导、异质 sweep fit 或 model-free task claim |
| 其他详细数学推导 | `THEORY_REBUTTAL_MATHEMATICAL_AUDIT_20260711.md`，受 7 月 13/16 日 fact gate 约束 | Phase16、simulation 或经验 sweep |
| 数字与 provenance | `docs/overview/RESULT_PROVENANCE_MANIFEST.md` 及其 raw-backed artifacts | 汇总表、历史 trace、服务器口头记录 |
| 政策与 venue 决策 | `REBUTTAL_VIABILITY_AND_VENUE_PLAN_20260713.md` | 旧年份 rebuttal 规则 |
| 真实评论的映射与发送 QA | `REVIEWER_TRIAGE_PLAYBOOK.md` | simulated reviews、master ledger |

### 1.2 核心 claim 与证据索引

| Claim | 直接证据 | 最窄可辩护表述 | 不可升级为 |
| --- | --- | --- | --- |
| Frequency allocation 是第三 PE 轴 | `paper/sections/01_intro.tex`；`paper/sections/02_related.tex`；canonical schedule `scripts/lib/rope/schedules.py` | EVQ 在训练前改变有限 RoPE frequency table，不改变 rotation operator | universal long-context SOTA、range-scaler替代品或容量定理 |
| Cosh allocation family | `paper/sections/03_theory.tex`；`paper/appendix/a1_proofs.tex`；full audit §3 | 给定 stated convex surrogate，cosh density 是唯一正的归一化最小解 | exact RoPE kernel、attention、LM objective 或 PPL 的闭式最优解 |
| Allocation optimality | first-principles reassessment §2 | 不存在task-independent optimum；geometric在uniform log-frequency coverage/quantization等对称目标下严格最优，cosh只在stated surrogate内严格最优 | “geometric generally suboptimal”或“cosh is the RoPE/task optimum” |
| Deployed \(\tau\) | first-principles reassessment §3；theory audit；现有 sweep | 真实task-local optimum依赖\(A_{task}/B_{task}\)；\(d/\sqrt L\)只是在额外diffuse/channel/dimension假设下的conditional proxy与经验basin selector | ordinary-KL theorem、global optimum、exact-collision optimum或trained-task theorem |
| Primary I | `paper/tables/table2_evq_yarn_main.tex`；`data/curated/primary1_evq_yarn_10pct_raw.json` | 454M、3-seed、midpoint-Geo/EVQ × repo fixed-ramp 的 2×2 factorial contrast与 differential leverage | official/tuned YaRN complementarity或formal interaction |
| Primary II | `paper/tables/table4_pe_dominant.tex`；历史 runner锚点 `8616af4`；curated artifacts | 约151.9M、seed-42 的 midpoint-Geo / EVQ / learnable-shared-frequency diagnostic | faithful DAPE comparison、125M protocol或全表多seed |
| Primary III | `paper/sections/05_experiments.tex`；`paper/appendix/a3_supporting_results.tex`；full audit §4 | actual head_dim=64、d_rope=32下，empirical \(\tau=1.414\) 的 scarce-channel方向性结果；三seed协议异质 | \(d_{\mathrm{eff}}=128\) theorem、strict same-protocol replication或production-identical MLA |

Supporting LoRA、video DiT、750M continuation、QuALITY、progressive training 不承担上述核心 claim。2026-07-15 registered QA 已否证当前 EVQ-LoRA 的 practical capability claim，不能用16K/32K PPL稳定性覆盖。

### 1.3 Claim disposition

| Disposition | 内容 | Rebuttal 动作 |
| --- | --- | --- |
| **DEFEND** | finite spectral budget / training-time frequency allocation 这一设计视角；给定 convex surrogate 的 exact cosh optimizer；closed-form、zero-learned-parameter schedule | `CLARIFY_EXISTING`：给清楚对象、假设和边界；不暗示task-independent optimality |
| **DEFEND, NARROWLY** | Primary I 的 local 2×2 contrast；Primary II seed-42 diagnostic；Primary III heterogeneous scarce-channel result | `PARTIAL + SOFTEN_CLAIM`：只守实际 protocol 与 evidence tier |
| **RELABEL** | YaRN → repo-defined fixed-ramp scaler；DAPE (32p) → learnable shared inv_freq (32p)；Geo → Midpoint-Geo；PK → teacher-forced NLL-gap；Primary II 125M → about 151.9M | `ACCEPT_TEXT + CLARIFY_EXISTING`：数值未变，但旧方法/指标解释撤回 |
| **WITHDRAW** | official-YaRN complementarity、official-DAPE comparison、native/standard-RoPE dominance、fixed-extrema/fixed-span pure-shape interpretation、ordinary-KL对deployed \(\tau\)的推导、`c_coll=1.171`、Phase16 `27 configs/all <1%`、LoRA rank/channel theorem、MLA \(d_{\mathrm{eff}}=128\) theorem | `SOFTEN_CLAIM + BLOCKING`：不得用新增实验或换名恢复 |
| **SUPPORTING ONLY** | registered 303-example negative QA、fresh LongAlpaca single-seed NLL、retrieval/readout causal diagnostics、old LoRA、QuALITY、video、750M、progressive | 只有被真实 review 直接触发且同时披露负边界时使用；否则 `OUT_OF_SCOPE` |
| **DEFER** | faithful DAPE training、native-endpoint + official-YaRN full comparison、broad tuned-scaler grid、新模型族、完整 task-kernel theorem | 不是 7 月 22 日前默认任务；reviewer明确设为score-changing criterion时再评估 |

### 1.4 Reporting / protocol correction index

这些不是新的实验任务，而是 reviewer 一旦触发 trust/reproducibility 时必须准确使用的 factual corrections：

| 项目 | 已确认事实 | 当前动作 |
| --- | --- | --- |
| Primary I Table 3 | 两列都使用连续-token CE；16K差异来自evaluation-length list改变后共享RNG的sampled offsets变化，不是per-document vs full-sequence | relabel为sampled-offset sensitivity；承认eval chunks有限 |
| Primary II | 约151.9M；FineWeb-Edu；\(L_{\mathrm{train}}=128\)；15M tokens；base LR \(3\times10^{-4}\)；effective batch 64；shared-frequency PE LR 0.03；headline seed 42 | 撤回`125M / LR 6e-4 / batch 16 / DAPE`复现口径 |
| Primary III | seed 42 batch 6，seeds 43/88 batch 5；fixed token budget下optimizer steps和schedule不同 | 只称heterogeneous three-seed replication with within-seed paired comparisons |
| Phase16 | 99 runs、9 configurations；45 seed-42 pilots + 54 confirmations；共同三seedweighted extrapolation log-PPL下formula为7/9胜、2/9负 | 撤回27-config、all<1%、near-optimal与旧rank claims |
| Figure 3 | generator读取`yarn_auto`；8K/256时scale是32，但图标成fixed s=8 | 若被问则直接纠正label；不把99.6/260.2称fixed-s8 |
| Checklist / reproduction / compute | claims/proof/details/open-reproduction/statistics的若干`Yes`不成立；tracked reports与A100/H100 compute叙述冲突 | 不作无条件reproducibility声明；只沿manifest与可核验artifact回答 |
| Supporting families | 750M有update/schedule confound；video provenance未闭合；QuALITY pipeline与公开命令不匹配；旧LoRA不隔离EVQ | 从core defense移除；真实trigger时逐项给边界 |

## 2. 理论主轴一：\(\tau\) 是 exact、conditional 还是 heuristic

### 2.1 Exact：shape family

`[数学事实]` 对固定 surrogate

\[
\mathcal C_{\mathrm{app}}[\rho]
=\frac{\alpha}{2}\int\rho^2
+\frac{\beta}{2}\iint\rho(\phi)\rho(\psi)\min(\phi,\psi),
\]

在 \(\alpha>0,\beta\ge0\)、\(\rho\ge0,\int\rho=1\) 下，唯一最小解（\(\beta=0\) 取 \(\tau\to0\) 极限）为

\[
\rho_\tau(\phi)=\frac{\tau\cosh(\tau(1-\phi))}{\sinh\tau},
\qquad \tau^2=\beta/\alpha.
\]

存在性、唯一性、正性、边界条件、CDF、inverse CDF 与 \(\tau\to0\) geometric limit 在 stated problem 下成立。这一结果决定 cosh allocation family；它没有决定 trained task 的最佳 finite \(\tau\)。

### 2.2 Strict：真实 task-local optimum 的形式

令 \(\theta=\tau^2\)。对固定模型和指定task risk，Geo附近可写成

\[
R(\theta)=R_0+A_{\mathrm{task}}\theta
+\tfrac12B_{\mathrm{task}}\theta^2+o(\theta^2).
\]

若 \(B_{\mathrm{task}}>0\)，约束 \(\theta\ge0\) 下

\[
\tau_*^2=[-A_{\mathrm{task}}/B_{\mathrm{task}}]_+.
\]

这里的梯度与曲率依赖数据、trained Q/K幅度、distance distribution、attention routing和task loss；允许权重重训时，曲率还包含weight-adaptation Schur complement。因此不知道这些对象时，不存在只由 \(d,L\) 决定的task-universal \(\tau\)。这不是新的closed-form recipe，而是当前理论缺失量的最窄正确表达。

### 2.3 Conditional：scaling structure

在 diffuse softmax、固定或各向同性 channel amplitude、channel additivity、small \(\theta=\tau^2\)、tested-grid \(Q_1>0\) 等假设下，phase-variance transport proxy

\[
U_{\mathrm{tr}}(\rho;L)
=\frac{M}{L}\int q(Lb^{-\phi})\rho(\phi)\,d\phi
\]

对 \(\theta\) 有非零一阶 variation。若stiffness明确取 \(\theta^2/(90d_S)\)，则该proxy的局部stationary point为

\[
\tau_*^2=45\lambda Q_1(L,b)M d_S/L.
\]

只有再令 \(M\propto d\) 且 \(d_S\propto d\)，才能给出 \(\tau\propto d/\sqrt L\)。这里的 \(q(x)\) 是 uniform-distance cosine phase variance，不是 attention loss、ordinary KL 或 LM loss；一致的per-channel归一化还可能消去线性dimension factor。稀疏/局部attention下应有条件地用有效支持或directional softmax-Jacobian quantity替代nominal \(L\)。

### 2.4 Empirical：deployed operating point

理论没有唯一决定：

- unit prefactor；
- practical finite \(\tau\)，尤其超出small-\(\tau\)区间的设置；
- Pearson \(\chi^2\) stiffness这一 modeling choice；
- trained attention 中 directional \(c^\top J(p)^2c\) 能否安全简化；
- MLA 中将 \(d_{\mathrm{eff}}\) 设为某个 architecture dimension 的约定。

因此唯一安全口径是：

> The cosh allocation family is theory-derived; the \(d/\sqrt L\) dependence is conditionally proxy-motivated; the deployed unit-prefactor \(\tau\) is an empirically supported basin selector.

现有三个常用经验点也不能写成clean scaling-law verification：\(L=128\) 的最佳测试点 \(\tau=5\) 位于sweep边界；\(L=1024\) 只有粗网格；\(L=2048\) 的峰值定位相对较强但仍非完整多seed sweep。它们混合模型、数据、预算和外推倍数，只支持“useful \(\tau\) 随训练长度增加总体下降”的方向。small-\(\tau\) stiffness在 \(\tau=2/4\) 的审计相对误差约为97%/252%，不能定量解释practical regime。

### 2.5 Ordinary-KL correction

若 \(z_\theta=z_0+\theta g+O(\theta^2)\)，则

\[
D_{KL}(p_0\|p_\theta)
=\tfrac12\theta^2g^T J_{\mathrm{sm}}(p_0)g+O(\theta^3)
=O(\tau^4).
\]

ordinary baseline-to-perturbed KL 的一阶 variation 为零。提交稿把 \(O(\tau^2)\) term 称为 ordinary post-softmax KL gain 是 order/identity error；两个 \(O(\tau^4)\) 项不能导出小而非零的 optimum。安全动作是承认错误，把 \(q/L\) 限定为 probability-displacement / Fisher-transport proxy，而不是用 sweep 证明旧推导。

### 2.6 Triggered answer kernel

> We agree that the deployed \(\tau\) is not an end-to-end theoretical optimum. The exact result is the cosh density for our stated convex surrogate. For a specified trained task, the local optimum depends on task-specific loss derivatives, \(\tau_*^2=[-A_{task}/B_{task}]_+\), which we have not estimated. Separately, under a diffuse-softmax, channel-additive phase-transport proxy and additional dimension identifications, the leading balance motivates the \(d/\sqrt L\) scaling. We also identified that calling its linear term an ordinary KL gain was incorrect: ordinary baseline-to-perturbed KL starts at \(O(\tau^4)\). We therefore treat \(\tau=d_{\mathrm{eff}}/\sqrt L\) as a proxy-motivated, empirically supported basin selector, not a global optimum or trained-task theorem.

只有真实 review 触发 \(\tau\)、KL、prefactor或optimality时，才按原话裁剪使用；不能整段预填到最终 response。

## 3. 理论主轴二：surrogate 与 RoPE phase kernel 的实际关系

### 3.1 Exact kernel 测量什么

论文定义

\[
K(\phi,\psi)
=\mathbb E_{\Delta\sim D}
[\cos(\omega(\phi)\Delta)\cos(\omega(\psi)\Delta)],
\qquad \omega(\phi)=b^{-\phi}.
\]

`[数学事实]` 这是指定 distance prior 下 cosine phase responses 的 Gram kernel。若两个频率在训练可见距离上产生高度相关的 cosine pattern，\(K(\phi,\psi)\) 较大，表示有限 channel budget 中的 phase-basis redundancy。它与 RoPE 的实际联系是：一个 RoPE pair 对 relative-position logit 的贡献由 cosine/sine phase线性组合构成。

但它不是“原始 RoPE loss”：

- content-independent，没有 trained \(Q/K\) amplitude、head/layer分布和 task gradient；
- 当前 kernel 只取 cosine-coordinate Gram，未完整表达二维 sin/cos pair；
- 依赖 distance prior \(D\)，uniform prior不能代表所有实际 attention distances；
- 因而它是 pre-training phase-redundancy proxy，不是 attention或LM objective。

### 3.2 Surrogate 接上 kernel 的强度

论文使用

\[
K_{\mathrm{app}}(\phi,\psi)
=\alpha\delta(\phi-\psi)+\beta\min(\phi,\psi)
\]

并最小化 \(\frac12\langle\rho,K_{\mathrm{app}}\rho\rangle\)。这一连接只支持三点：

1. \(\delta\) 项惩罚 frequency-density concentration，\(\min\) 项惩罚 cumulative low-frequency overlap；两者平衡产生 cosh family；
2. oscillatory exact kernel 与 smooth surrogate 不是 pointwise 或 global operator approximation；
3. 现有 12-config结果是 fixed-allocation directional validation：deployed EVQ相对Midpoint-Geo降低已有 normalized collision diagnostic并提高effective rank。

它不支持 shared minimizer。surrogate最小化线性 quadratic form，而 appendix 报告的是 squared、normalized、off-diagonal statistic；当前没有 theorem 证明两者 minimizer相同，更没有证明任一目标最小化PPL。更关键的是，appendix把 \(\alpha,\beta\) 直接拟合到 exact kernel 后得到的经验 scaling约为 \(\sqrt d\,L^{-0.11}\)，并没有推出 deployed \(dL^{-1/2}\)。因此 cosh family、deployed scale 与 exact-kernel diagnostic 必须保持三段式证据链，不能写成一步推导。

### 3.3 `c_coll=1.171` 为什么不能用

`scripts/analysis/verify_c_coll.py` 没有优化 collision score，而是读入预置 `tau_coll` 后重算比例。独立审计找到更优可行点，因此：

- 不引用 `c_coll=1.171` 作为 exact-kernel optimum；
- 不说 exact kernel闭合unit prefactor或 \(L^{-1/2}\)；
- 只保留固定 deployed allocation 相对 Midpoint-Geo 的 directional diagnostic。

### 3.4 EVQ 本质上优化什么

| 层次 | 对象 | 证据强度 |
| --- | --- | --- |
| 严格数学 | stated convex surrogate中，density concentration与cumulative low-frequency overlap的平衡 | exact theorem |
| 信号处理解释 | 将部分near-static / redundant low-frequency channels迁向训练窗口内phase variation更充分的区域 | exact construction + proxy interpretation |
| 现有 diagnostic | listed configs中的cosine-Gram normalized collision/effective rank | empirical directional validation |
| 训练结果 | 列明模型、任务、seed和protocol中的PPL/retrieval | empirical only |

不得说 EVQ closed-form minimizes exact RoPE kernel、full attention loss、LM objective、PPL，或所有真实 distance priors 下的 collision。

### 3.5 Finite-grid shape / extrema / span boundary

对实际inverse-CDF map

\[
\phi_\tau(u)=1-\tau^{-1}\operatorname{asinh}((1-u)\sinh\tau),
\]

当 \(u\in(0,1),\tau>0\) 时严格有 \(\phi_\tau(u)<u\)。因此midpoint EVQ不只改变density shape，也把所有离散频率向高频移动并改变finite-\(K\) extrema与realized span。代表性设置 \(K=32,b=500\mathrm K,\tau=4\) 中，最高/最低采样频率约变为midpoint-Geo的1.17/3.17倍，natural-log span约从12.71缩到11.71。

所以submitted control只能称“shared midpoint-quantile convention下的schedule intervention”，不能称fixed-extrema/fixed-span pure-shape causal isolation。若reviewer问“是否只是range/base effect”，最小判别是加入与EVQ realized extrema匹配的uniform log-frequency control，再将 `Geo-original -> Geo-span-matched` 与 `Geo-span-matched -> EVQ` 分开；不能用same quantile grid代替这个控制。

### 3.6 Triggered answer kernel

> Our exact kernel is a content-independent Gram kernel of RoPE cosine phase responses under a specified distance prior, so it measures redundancy among finite phase channels rather than the full attention or language-model objective. The \(\delta+\min\) model is a tractable surrogate for a diagonal concentration penalty and cumulative low-frequency overlap; the cosh density is the exact minimizer of that surrogate, not of the oscillatory kernel itself. Existing diagnostics show that the resulting fixed allocation reduces normalized channel collision, but this is directional validation, not objective equivalence. The finite midpoint implementation also changes realized extrema and span, so the submitted comparison establishes a schedule intervention, not fixed-span pure-shape causality.

若 reviewer 问“理论贡献还剩什么”，只答：**一个明确、可解的 frequency-allocation surrogate及其closed-form optimizer，加上对phase-redundancy和trained behavior的分层验证**。不声称kernel-to-task闭环。

## 4. Score-driving 风险与准备状态

Likelihood 是 reviewer 实际提出的相对可能性；Impact 是回答失败对评分或 AC trust 的影响；Prepare 表示 7 月 22 日前能否形成诚实、可核验回答。没有真实评论前，这些都是内部判断。

### P0-A — 方法身份与 scientific-integrity disclosure

- **Reviewer 可能问**：DAPE、YaRN 和 Geo control 是否忠实？为什么代码、表格和复现说明不一致？
- **Likelihood / Impact / Prepare**：`H / BLOCKING / high`。
- **真实风险**：这是三项已确认的 identity error，不是 tuning 分歧。数值可保留，但 official-method comparison和standard-RoPE dominance不能保留。
- **证据**：full audit §2、§4、§7；实际 forward path、官方定义和raw-backed结果。
- **当前材料**：事实与correction map充分；是否主动向AC合并披露、如何跨reviews去重，仍为 `AUTHOR_INPUT_NEEDED`。
- **策略**：`ACCEPT_TEXT + SOFTEN_CLAIM + CLARIFY_EXISTING`。错误 → unchanged numerics → withdrawn interpretation → narrow local result。
- **现在准备**：冻结一条不含链接、无辩解语气的120–180词integrity kernel；不跑新DAPE/YaRN实验来淡化错误。
- **启动信号**：任一 reviewer / AC点名fidelity、code、reproduction或trust；若无人点名，由作者决定是否向AC作一条合并披露。这是研究诚信建议，不是Handbook明文规定的专用流程。

### P0-B — \(\tau\)、ordinary KL、exact kernel与“到底优化什么”

- **Reviewer 可能问**：\(\tau=d/\sqrt L\) 是theorem还是heuristic？surrogate和RoPE kernel有何关系？是否真的优化attention/LM loss？
- **Likelihood / Impact / Prepare**：`H / BLOCKING / high with concession`。
- **真实风险**：ordinary KL一阶为零；`c_coll`未被优化；surrogate functional与exact diagnostic不同；finite-\(\tau\)不是exact-kernel optimum。
- **证据**：本文件§2–§3；full audit §3；theory audit。
- **当前材料**：长推导充分，短答已有；需按真实问题压缩，状态 `PARTIAL`。
- **策略**：`ACCEPT_ANALYSIS + SOFTEN_CLAIM`。exact shape / conditional scaling / empirical operating point三层；撤回ordinary-KL与`c_coll`。
- **现在准备**：保留“optimizes / does not optimize”对照和两个short kernels；无需GPU实验，不补造task theorem。
- **启动信号**：reviewer点名\(\tau\)、KL、prefactor、optimality、collision、distance prior、surrogate validity、mechanism或task relation；若无人点名但accepted record保留KL/`c_coll`错误，则纳入合并integrity disclosure。

### P0-C — 协议、统计身份与整体 trust

- **Reviewer 可能问**：Primary II是125M还是152M、是否多seed？Primary III是否严格同协议？Phase16与checklist/compute是否可复现？
- **Likelihood / Impact / Prepare**：`H / MAJOR-to-BLOCKING / medium`。
- **真实风险**：Primary II实际约151.9M且headline是seed-42；Primary III三seed batch不一致；Phase16是99 runs/9 configs/selected-confirmation；继续复述旧协议会把局部错误升级为整体失信。
- **证据**：full audit §4；provenance manifest及raw-backed artifacts。
- **当前材料**：核心数字链可用，但exact runner/checkpoint closure与checklist/compute并非全部闭合，状态 `PARTIAL + BLOCKING`。
- **策略**：只报exact model/seed/batch/metric/artifact；明示single-seed、heterogeneous replication与selected-confirmation。缺失runner不写成fully reproducible。
- **现在准备**：从现有audit/manifest抽取每个真实trigger需要的最小provenance行；不再建第二份大总表，不预防性复跑supporting families。
- **启动信号**：reviewer提到seed、variance、model size、budget、compute、checklist、artifact、figure/table inconsistency或reproducibility。

### P0-D — 纠错之后还剩什么贡献

- **Reviewer 可能问**：撤回official DAPE/YaRN、global \(\tau\)和native-RoPE口径后，还有足够novelty/significance吗？
- **Likelihood / Impact / Prepare**：`M-H / BLOCKING / high`。
- **真实风险**：继续依赖已撤回部分会让AC判断核心坍塌；只罗列缺点又会丢失仍成立的mechanism contribution。
- **证据**：full audit §5；本文件§1.3；Primary I–III真实evidence tier。
- **当前材料**：survivor set清楚，状态 `READY_WITH_CONCESSION`，最终stance需作者批准。
- **策略**：`CLARIFY_EXISTING + SOFTEN_CLAIM + AUTHOR_INPUT_NEEDED`。只守finite spectral budget、training-time allocation、exact surrogate optimizer、zero-parameter construction和三个受限empirical signals。
- **现在准备**：冻结一段100–140词contribution kernel，与所有correction使用同一窄口径。
- **启动信号**：AC/meta-review问remaining contribution、novelty、significance，或多位reviewer共同指向“论文还剩什么”。

### P1 — 真实风险，但只在明确 trigger 后展开

| 风险 | Reviewer 可能如何问 | 当前最稳回答 | 材料状态 / 何时启动额外工作 |
| --- | --- | --- | --- |
| PK / capability | 100% PK是否就是生成式exact retrieval或通用长上下文能力？ | PK只定义为teacher-forced NLL-gap；AR exact单列，同时报告8K seed spread与4K reversal | `CLARIFY_EXISTING`；无需新benchmark，reviewer明确要求某endpoint后再评估 |
| Primary II single seed | 为什么把seed-42 diagnostic作为primary？ | 承认整张comparator table未多seed；额外fixed-EVQ seeds不能升级全表 | `SOFTEN_CLAIM`；不机械补seed，除非reviewer把matched replication列为明确升分条件 |
| Official YaRN / native endpoint | faithful YaRN或standard RoPE下方向是否仍在？ | 已跑 single-seed component ablation（`EVQ_YARN_COMPONENT_ABLATION_20260714.md`）：abundant MHA 下 official-YaRN 的**频率校正**（非 mscale）抹平 EVQ gap，证实 complementarity 撤回并给出机制；scarce MLA 激进外推处（scale 8/16）**未抹平**（EVQ+full 8K 71.6<85.5）。只支持机制方向，不恢复 complementarity；P2/P3 已否证 | `PARTIAL + SUPPORTING`；deploy-on-trigger；边界见 ablation 文档（single-seed、L=512、τ 未重推、PPL-only、modest 外推仍被抹平） |
| MLA convention | 为什么用 \(d_{\mathrm{eff}}=128\)，是否验证公式？ | actual head_dim=64、d_rope=32；\(\tau=1.414\)只作ad-hoc empirical setting | `SOFTEN_CLAIM`；不补dimension ablation，不称theorem |
| Undertraining / scale | 短训练预算是否制造优势，能否泛化到重预训练模型？ | 报exact budget与负/反向边界；现有8B结果显示强training-time coadaptation与readout bottleneck，不能把small-model PPL外推成mature-model capability | `PARTIAL`；先用已有inference-only causal evidence；不再默认启动8B训练 |
| LoRA / downstream | 频率重分配在LLaMA-3-8B下是否真的可学、是否改善任务？ | registered 303-example QA为negative：EVQ-LoRA task-macro F1 0.1126 vs Native-LoRA 0.2110，差值-0.0984、95% CI [-0.1297,-0.0697]；缺口集中\(\le8\)K，\(>8\)K各臂接近floor；同时16/32K temporal PPL更稳 | 默认 `OUT_OF_SCOPE`；reviewer点名时必须把task negative、8K harm、long-PPL positive与single-seed/two-task边界一起报告，不能称capability gain或pure-shape control |
| Novelty / related work | 是否只是调base、插值、搜索或既有scaler变体？ | 用intervention stage、optimized object与自由度区分；不声称替代或数学正交 | `CLARIFY_EXISTING + ADD_CITATION`；只回应reviewer点名的最近工作 |

### P2 — 当前明确排除

- broad tuned-base / scaler / model-family zoo；
- faithful DAPE重训练、production-scale pretraining或新下游benchmark；
- video、750M、progressive、QuALITY或旧LoRA的预防性复跑；
- 新Bessel/forcing/global exact-kernel theorem或 \(L_{\mathrm{eff}}^J\) 测量；
- 重复失败的8K/50-step retrieval recipe、继续扫sparse budget/block/selector、temperature/beam/top-p或宽\(\tau\)网格；
- 无matched span/Geo control、无任务梯度或只做隐藏态蒸馏的LoRA；
- 没有真实reviewer trigger的预制author response。

这些工作不是被绝对禁止，而是当前不能直接改变一个已触发的评分问题，或无法在 rebuttal 窗口达到 reviewer-grade。若真实 review 给出明确升分判据，再按 `question -> discriminating result -> stop rule -> provenance` 评估。

## 5. 现有 rebuttal 材料审计

| 材料 | 当前状态 | 使用规则 |
| --- | --- | --- |
| `FULL_PAPER_INTEGRITY_AUDIT_20260713.md` | **充分 / canonical fact gate** | 所有method identity、theory、protocol与survivor-set判断以此为准 |
| `REBUTTAL_VIABILITY_AND_VENUE_PLAN_20260713.md` | **充分 / decision context** | 使用政策边界与trust-repair判断，不复制venue扩展内容到response |
| `THEORY_REBUTTAL_MATHEMATICAL_AUDIT_20260711.md` | **内容充分但不能直接发送** | 只提取与真实问题对应的最短推导；受7月13日审计覆盖 |
| `REVIEWER_TRIAGE_PLAYBOOK.md` | **流程入口** | 真实reviews到达后保存verbatim trigger、分配稳定ID、记录action/readiness |
| `REBUTTAL_MASTER_QUESTION_LEDGER_20260711.md` | **archival / 部分过时** | 只用于找历史攻击面；不得复制answer kernel、实验优先级或旧方法身份 |
| `FIRST_PRINCIPLES_REBUTTAL_REASSESSMENT_20260716.md` | **充分 / canonical reasoning gate** | optimality、\(\tau\)、PDF可见性、shape/span与representation-routing-readout判断以此为准；不是可直接发送的response |
| `THEORY_FREQUENCY_OPTIMALITY_AND_TAU_20260716.md` | **充分 / focused theory audit** | 按目标区分 geometric/cosh 最优性，给出ordinary-KL、task-local \(\tau\)、normalization、finite-grid 与异质 sweep 边界；只按真review trigger裁剪 |
| `LORA_GEO_CONTROL_RESULT_AUDIT_20260711.md`、`LORA_LONGALPACA_TEMPORAL_NLL_20260712.md`、`../docs/exp/2026-07-14_lora_retrieval_conversion_probe.md`、`../docs/exp/2026-07-15_lora_qa16k_three_arm_results.md` | **supporting-only / capability claim negative** | reviewer点名LoRA时才启用；必须同时披露quantizer差异、single seed、\(\le8\)K task harm、\(>8\)K floor及PPL/能力分离 |
| `EVQ_YARN_COMPONENT_ABLATION_20260714.md`（源：`data/curated/native_rope_evq_150m_s42_500m_20260713.json`、`results/mla_yarn_short_s42_20260714/`、`docs/exp/2026-07-14_mla_k16_short_context_yarn_ablation.md`） | **主动实验 / supporting-only** | official-YaRN 被 trigger 时启用；单seed机制证据；确认 complementarity 撤回并支持 scarce-channel 方向；不升级 primary、不恢复 complementarity |
| `frequency_adaptation_8b/` | **paused / archival mechanism implementation** | registered QA与readout diagnostics已给出negative gate；不再作为默认实验队列，也不修复paper-lineage comparator或theory错误 |
| `simulated_reviews/` | **内部压力测试** | 不是真实opinion、score或trigger，不进入最终response |

### 5.1 主动实验状态（proactive component experiments）

以下实验由内部 audit 触发（对应 P1 "Official YaRN / native endpoint" 与 P0-D remaining-contribution），把之前标为"未知"的 official-YaRN 方向变成有边界的机制证据。全部 single-seed / supporting、deploy-on-trigger，不升级 primary、不恢复已撤回的 complementarity。

| 实验 | 状态 | 结论（含边界） | Disposition |
| --- | --- | --- | --- |
| MHA K=32 four-arm 分解（500M 六格 + 同 checkpoint 四算子） | 完成 | abundant 通道下 official-YaRN 的**频率校正**抹平 EVQ gap（mscale 不抹）→ 证实 complementarity 撤回并给机制 | 确认 WITHDRAW 并附机制；trigger 时用 |
| MLA K=16 four-arm 分解（`mla_yarn_short_s42_20260714`） | 完成 | scarce + 激进外推（scale 8/16）处频率校正**未抹平**，EVQ+full>Native+full；但 P2/P3 被否证、modest 外推被抹平、PPL-only、L=512、τ 未重推、单seed | 支持 scarce-channel 机制方向；trigger 时用；不升级 Primary III |
| 8B retrieval/readout causal decomposition | 完成 | EVQ提高remote block hit/rank，gold-block removal对EVQ有因果影响；但forced-gold、sparse selector和8K/50-step tune均未转化为EM，首token仍约rank 2K | supporting-only；定位representation/routing signal与readout failure；停止该recipe |
| Sparse-attention 转换实验 | 完成 / negative stop | score sparsity帮助Geo更多，EVQ/Geo均0% EM；oracle gold inclusion仅给EVQ约0.034 NLL改善，dense dilution不是当前主瓶颈 | 不扩100-case、不扫budget/block/selector；reviewer点名sparse时报告negative boundary |
| Registered 303-example 16K-max QA | 完成 / negative gate | EVQ-LoRA显著差于Native-LoRA，aggregate deficit集中\(\le8\)K；\(>8\)K三臂均接近floor；long PPL稳定未转成QA | capability claim false；仅reviewer点名8B/downstream时完整部署 |
| Span-matched schedule + frozen task-risk decomposition | ready-to-spec / not run | 分开`Geo-original -> Geo-span-matched`的range/span效应与`Geo-span-matched -> EVQ`的shape效应，并估计真实task loss对\(\theta=\tau^2\)的局部方向 | 只有reviewer把pure-shape或surrogate-to-task relation设为score-changing criterion时启动；正负都必须报告 |

不再把 \(\tau=5.66\) MLA复跑当作默认硬化项：当前公式本身不是task theorem，重推一个公式值不能关闭method-identity或optimality问题。只有reviewer明确要求该历史setting的sensitivity时才评估。

当前真正缺失的不是另一份大文档，而是：真实reviews、作者对integrity disclosure的决定、每个实际trigger对应的短答案，以及未闭合provenance项的诚实边界。

## 6. 7 月 22 日前最小行动清单

1. **作者确认 survivor stance。** 统一采用本文件§1.3，不再在不同回复中恢复official DAPE/YaRN、native Geo或global-\(\tau\)口径。
2. **作者决定 integrity disclosure。** 若reviews未触发已确认的material errors，是否向AC发一条合并说明；建议最小覆盖comparator identity、Geo discretization、ordinary-KL、`c_coll`与Phase16 reporting correction；若Primary II/III仍作为core，再加入真实model/optimizer/head/batch/`d_eff`事实。
3. **冻结三个短核。** comparator-identity correction 120–180词；\(\tau\)/kernel correction 120–180词；remaining-contribution 100–140词。它们只是组件，真实评论前不组装final response。
4. **冻结一页事实索引。** 直接复用full audit、first-principles reassessment、manifest和本文件，不再新建平行ledger。每项只保留submitted location、actual method/protocol、unchanged number、withdrawn interpretation和surviving claim。
5. **7月22日先做verbatim mapping。** 读完所有reviews和meta-review后，最多选择3–5个score-driving concerns；逐条分配稳定ID，再决定篇幅和实验。
6. **冻结实验队列而不是启动GPU campaign。** 第一候选是span-matched schedule/frozen task-risk分析；第二候选是现有adapter×runtime-frequency的paired QA cross；只有真实reviewer给出score-changing criterion且§8 gate满足时启动。
7. **停止已否证路线。** 不重复8K/50-step tune、不扩sparse pilot、不做宽8B LoRA/LongBench sweep。只有Stage 0/1定位出明确readout bottleneck后，true-16K source-dependent matched supervision才是新的研究问题，不能回证submitted zero-shot claim。

### 6.1 已冻结的内部组件，不是 final response

Comparator-identity kernel（只在P0-A trigger或作者批准的integrity disclosure中裁剪使用）：

> During our post-submission audit, we identified three method-identity corrections. The row labeled “DAPE” learned a layer-shared inverse-frequency vector and did not implement the cited data-adaptive attention-score operator. The arm labeled “YaRN” used our fixed-index smooth-ramp scaler rather than the official rotation-derived correction range and attention mscale. Our matched geometric control also used midpoint rather than native endpoint discretization. The reported numerical values are unchanged, but we withdraw the DAPE-specific, official-YaRN, and standard-RoPE interpretations. The surviving evidence is a midpoint-grid allocation comparison, a learnable shared-frequency control, and a local 2×2 contrast with the repository-defined fixed-ramp scaler.

Remaining-contribution kernel（只在P0-D trigger中裁剪使用）：

> After these corrections, the contribution is narrower but still concrete: RoPE exposes a finite frequency-allocation budget that can be designed at training time, separately from the rotation operator and inference-time range transforms. EVQ-Cosh provides a closed-form, zero-learned-parameter allocation whose shape is the exact optimizer of a stated convex surrogate. Its empirical support is limited to the audited midpoint-grid, fixed-ramp, shared-frequency, and scarce-channel protocols; we do not claim faithful DAPE/YaRN comparison, native-RoPE dominance, or an end-to-end optimality theorem.

理论组件见§2.6与§3.6。真实评论到达前不得把这些段落拼成通用“全错说明”；必须按trigger删除无关内容并满足10,000-character budget。

## 7. 真实 reviews 到来后的工作流

1. local-only保存reviewer与AC原话，不改写、不把模拟review混入。
2. 按 `R1.1`、`R1.2`、`R2.1`、`AC.1` 分配稳定ID；排序变化不重编号。
3. 为每个ID记录：category、severity、likelihood、impact、action、readiness、evidence、boundary、author decision。
4. 先判断它是 **review-triggered response**，还是 **untriggered material-integrity disclosure candidate**。两者不能混写。
5. 只选择3–5个能改变score的concerns；minor wording、broad suggestions和future work不抢字符。
6. 每段按以下结构组装：

```text
Direct answer or correction:
Unchanged fact/evidence:
Withdrawn interpretation:
Surviving narrower claim:
Boundary or limitation:
Future manuscript action (only if author-approved; never claim already revised):
```

7. 需要新实验时，先执行§8 gate；失败、负结果和边界必须与正结果一起进入决策。
8. 每份review单独执行10,000-character gate、no-link gate、double-blind gate和OpenReview readers检查。

## 8. 新实验的 rebuttal gate

实验不是禁区，但必须同时满足：

- **Trigger**：绑定真实 reviewer/AC逐字问题或明确升分标准；
- **Discrimination**：无论正负都能区分两个会改变回答的解释；
- **Control**：同数据/order/tokens/optimizer/steps/evaluator，并隔离schedule quantizer与range operator；
- **Endpoint**：使用问题真正要求的task endpoint，不能用PPL或hidden-state match替代能力；
- **Stop rule**：预注册seed-42或小规模gate；无material signal立即停止，不机械扩seed；
- **Provenance**：config、commit、data hash、checkpoint identity、raw result与失败日志齐全；
- **Scope**：只能补充被问证据，不能创造新主claim。

按当前状态：

- ordinary-KL、`c_coll`、DAPE/YaRN/Geo identity：**实验不能修复，只能纠正**；
- pure-shape / range confound：首选同\(K\)、同realized extrema的uniform Geo control；shared midpoint quantile grid本身不隔离span；
- surrogate-to-task relation：首选现有checkpoint上的paired finite-\(\theta\) task-risk/rank/generation测量，不先训练新模型；
- official YaRN / native endpoint：现有single-seed component result只作triggered mechanism evidence，不能称faithful submitted comparison或恢复complementarity；
- 8B frequency adaptation：registered QA已经给出negative practical gate；新训练只回答新的capability-learning问题，不能证明paper主结果、exact-kernel theorem或standard-RoPE superiority；
- sparse attention：现有matched pilot与oracle inclusion已否定当前dense-dilution解释，停止参数搜索；
- broad baseline、new downstream、video/750M rerun：默认 `OUT_OF_SCOPE`。

## 9. 条件分支

| Trigger | 回答路径 | 必须停止的位置 |
| --- | --- | --- |
| \(\tau\) / KL / optimality | §2 + P0-B | basin selector，不扩成trained-task theorem |
| kernel / surrogate / mechanism | §3 + P0-B | directional validation，不声称objective equivalence |
| pure shape / base / spectral range | §3.5 + first-principles reassessment §2.4 | submitted是schedule intervention；没有span-matched control就不声称pure-shape causality |
| DAPE / YaRN / Geo fidelity | §1.3 + P0-A | relabel与withdraw，不用近似新实验补洞 |
| seed / protocol / reproducibility | P0-C + manifest | exact artifact边界，不声称full reproduction |
| PK / downstream capability | P1 metric row + 2026-07-14/15 reports | TF/AR与negative QA boundary；PPL/routing不替代readout/generation |
| novelty / significance | P0-D + P1 novelty row | finite-budget mechanism，不恢复撤回部分 |
| reviewer未触发material errors | 作者决策门 | 最多一条合并AC disclosure，不向每份review发散 |
| reviewer要求大规模新实验 | §8 | 若无法在窗口形成matched reviewer-grade result，明确scope并defer |

## 10. 外部真实评审的校准

- [The Impact of Positional Encoding on Length Generalization, NeurIPS 2023](https://openreview.net/forum?id=Drrl2gcjzl)：真实review集中于scope、LM/task外推、规模和novelty；限定scope比扩大主张更有效。
- [Scaling Laws of RoPE-based Extrapolation, ICLR 2024](https://openreview.net/forum?id=JO7k0SJ5V6)：reviewer关注YaRN novelty、PPL-only evaluation和实际任务；直接回答endpoint有用，但模型族限制仍会限制评分。
- [Probing RoPE through Frequency Entropy, ICLR 2026](https://openreview.net/forum?id=1JZuEDq62N)：reviewer抓causal confound和practical utility；matched control能提供信息，“潜在应用”不能替代直接证据。

迁移到本论文的结论只有一个：**回答真实 reviewer 指出的最近因果/理论缺口，必要时诚实纠错和收窄；不要用额外但不直接相关的材料制造breadth。**

## 11. 当前无法完全解决的真实风险

1. surrogate 与 exact phase kernel没有pointwise、operator-norm或shared-minimizer theorem；只有结构动机与directional validation。
2. phase kernel不是full RoPE attention/task objective；trained amplitudes、sin/cos pair、distance prior和optimization dynamics均未闭合。
3. midpoint EVQ同时改变shape、finite-grid extrema、span与active-channel behavior；submitted control没有隔离pure-shape effect。
4. \(d/\sqrt L\) unit prefactor与practical finite \(\tau\)是经验的；task-local \(A_{task}/B_{task}\)、effective support和dimension identity均未测。
5. official DAPE/YaRN与native endpoint baseline没有在原提交协议中实现；rebuttal内不能靠relabeled近似恢复。
6. Primary II single-seed、Primary III heterogeneous replication、Phase16 selected-confirmation与reproduction closure会限制trust。
7. official YaRN在abundant MHA中抹平gap；8B registered QA为negative；当前证据没有真实下游能力提升。
8. reviewer可能认为纠错后的理论—任务链过松或empirical scope过窄。这是应接受的评分风险，不能靠发散回答解决。

## 12. 中文核对：作者必须明确的决定

- [ ] 同意 survivor stance：主张收缩到finite spectral budget、exact surrogate family和受限empirical signals。
- [ ] 决定若reviewer未点名，是否仍向AC合并披露material comparator/theory errors。
- [ ] 接受submitted midpoint comparison不是fixed-span pure-shape control，并决定是否只在reviewer trigger后启动span-matched诊断。
- [ ] 确认任何disclosure不把current/future revision写成submitted paper已修复。
- [ ] 确认registered 8B QA practical claim为negative；不再把新8B训练当默认rebuttal任务。
- [ ] 真实reviews与meta-review到达后，逐字材料已local-only冻结并完成稳定ID映射。
- [ ] 所有 `AUTHOR_INPUT_NEEDED` 已由论文作者确认，才允许package从NOT READY升级。

## 13. 最终发送门

- [ ] 每个普通回答绑定真实、逐字保存的reviewer/AC trigger；唯一例外是作者批准的合并integrity disclosure。
- [ ] 每份review不超过10,000 characters，不含链接，OpenReview readers正确，双盲信息检查通过。
- [ ] direct answer/correction出现在段首，不把错误归因于reviewer误解。
- [ ] submitted、current source、raw-backed fact与future correction时态分开。
- [ ] 所有数字来自最新provenance manifest或其raw-backed artifact。
- [ ] \(\tau\)的exact / conditional / empirical三层未混写；ordinary KL一阶为零。
- [ ] 未把geometric写成generally suboptimal；cosh optimality只绑定stated surrogate；task-local optimum依赖\(A_{task}/B_{task}\)。
- [ ] exact kernel只称phase-redundancy proxy；surrogate与exact diagnostic未写成同一objective。
- [ ] midpoint EVQ的extrema/span变化已披露；没有span-matched control时未声称pure-shape causality。
- [ ] 未引用`c_coll=1.171`、27 configs/all<1%、LoRA-rank或MLA-`d_eff`旧理论。
- [ ] DAPE、YaRN、Midpoint-Geo、PK、Primary II model size、seed/batch/protocol身份准确。
- [ ] supporting LoRA若出现，同时披露single seed、quantizer差异、teacher-forced NLL、registered QA negative、\(\le8\)K harm与\(>8\)K floor。
- [ ] 没有universal SOTA、global optimum、tuned dominance、production readiness或end-to-end theory closure。
- [ ] 没有声称已上传或已完成NeurIPS不允许的paper/supplement revision。
