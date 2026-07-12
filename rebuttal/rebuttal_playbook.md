# EVQ-Cosh Rebuttal Playbook — 2026-07-22

最后核对：2026-07-12

状态：`pre-review / reviewer-response-first / targeted-preparation`

用途：真实 reviews 到来后的统一入口。本文优先准备如何回答最可能影响评分的问题，不预写完整 rebuttal，不无边界扩张论文。允许提前开展少量、高信息价值、具有明确停止条件的内部实验；未经完整核验且未被 reviewer 实际触发的新结果不得自动进入回复。

## 0. Rebuttal 的硬原则

1. **只回应真实 reviewer。** 模拟 review、内部审计和外部案例只用于预判，不得伪装成 reviewer trigger。
2. **实验服从问题，不服从禁令。** 可以做能直接澄清核心机制、因果对照或现有实验失败原因的高价值实验；不做无明确问题、无匹配控制、无停止条件的训练或 baseline zoo。新结果在完成 provenance、匹配协议和负结果边界核验前只属于内部研究材料。
3. **不把 rebuttal 当二次投稿。** 不引入新主张、新机制或新的证据层级。
4. **回答顺序固定为：直接回答 → 现有证据 → 适用边界。** reviewer 没问到的内部问题不主动发散。
5. **无法由现有证据关闭的问题，明确让步。** 不用 supporting LoRA、视频、progressive training 或单 seed 结果替代缺失的主控制。
6. **优先保住窄而真实的贡献。** RoPE 的有限频率表也是 finite spectral budget；EVQ-Cosh 把 training-time frequency allocation 作为 operator design 与 inference-time range scaling 之外的第三个 PE 设计轴。

证据标签：`[数学事实]` 表示可从当前条件推出；`[实验事实]` 表示由已有 paper/raw artifact/代码直接支持；`[文献事实]` 表示公开一手来源；`[判断]` 表示风险判断，不是已发生的 reviewer 意见。

## 1. 核心 claim 与现有证据

| Claim | 现有证据 | 最窄可辩护表述 | 不可升级为 |
| --- | --- | --- | --- |
| Frequency allocation 是第三 PE 轴 | `paper/sections/01_intro.tex:3-20`; `paper/sections/02_related.tex:3-17` | EVQ 在训练前改变标准 RoPE 的 frequency table，不改 rotation/operator | universal long-context SOTA 或 range scaler 替代品 |
| Cosh density / inverse-CDF warp | `paper/sections/03_theory.tex:23-65`; `paper/appendix/a1_proofs.tex:4-50` | 给定所写 convex surrogate，cosh density 是唯一正的归一化最小解 | full trained-transformer 或 exact RoPE kernel 的闭式最优解 |
| \(\tau=d_{\mathrm{eff}}/\sqrt L\) | `paper/sections/03_theory.tex:90-117`; theory audit `:177-199`; existing sweep report | conditional proxy 提供 scaling 动机，经验 flat basin 支持 operating default | 全局最优、参数无关定理或 exact-kernel minimizer |
| Primary I | `paper/tables/table2_evq_yarn_main.tex`; `data/curated/primary1_evq_yarn_10pct_raw.json` | 454M、3-seed、固定 scale 的 repository-defined progressive range overlay 上，EVQ substrate 获得更高 leverage | tuned/published YaRN dominance |
| Primary II | `paper/tables/table4_pe_dominant.tex`; `data/curated/fig3_extreme_128.json`; fixed-EVQ 3-seed JSON | seed-42 \(128\to8192\) diagnostic 中，EVQ 优于 Geo 与 32-parameter learnable-frequency control | faithful DAPE comparison 或全表 3-seed结论 |
| Primary III | `paper/sections/05_experiments.tex:49-52`; `paper/appendix/a3_supporting_results.tex:8-29` | stated \(d_{\mathrm{eff}}\) convention 下的 3-seed scarce-channel sensitivity | optimal MLA rule 或 production-identical DeepSeek result |

Supporting LoRA、video DiT、750M continuation、progressive training 不承担上述核心 claim；evidence tier 见 `paper/tables/table_evidence_tier.tex:12-20`。

## 2. 理论主轴一：\(\tau\) 到底是理论还是启发式

### 2.1 必须拆开的三个层次

#### A. 理论严格给出的：shape family

`[数学事实]` 对固定的 surrogate

\[
\mathcal C_{\mathrm{app}}[\rho]
=\frac{\alpha}{2}\int\rho^2
+\frac{\beta}{2}\iint\rho(\phi)\rho(\psi)\min(\phi,\psi),
\]

在 \(\rho\ge0,\int\rho=1\) 下，唯一最小解为

\[
\rho_\tau(\phi)=\frac{\tau\cosh(\tau(1-\phi))}{\sinh\tau},
\qquad \tau^2=\beta/\alpha.
\]

存在性、唯一性、正性、边界条件、CDF、inverse CDF 和 \(\tau\to0\) geometric 极限都成立：`paper/appendix/a1_proofs.tex:4-50`。这部分是论文最坚固的理论。

#### B. 有条件理论动机：scaling structure

`[数学事实]` 在 diffuse softmax、固定/各向同性 channel amplitude、channel additivity、small \(\theta=\tau^2\) 且 tested-grid \(Q_1>0\) 等假设下，phase-variance transport proxy

\[
U_{\mathrm{tr}}(\rho;L)
=\frac{M}{L}\int q(Lb^{-\phi})\rho(\phi)\,d\phi
\]

对 \(\theta\) 有非零一阶 variation。与 \(O(\tau^4)\) 的 allocation stiffness 平衡，可给出 \(\tau\propto M/\sqrt L\) 的 leading-order structure。`q(x)` 是 uniform distance 下 cosine phase variance，不是 task loss：`THEORY_REBUTTAL_MATHEMATICAL_AUDIT_20260711.md:78-100,177-187,297-360`。

这只是一条 **conditional proxy theorem**，不是 ordinary KL theorem，也不是 trained-attention theorem。

#### C. 启发式/经验部分：deployed operating point

以下部分没有被理论唯一决定：

- 单位 prefactor 取 1；
- practical finite \(\tau\)（例如 \(\tau=4\)）超出 small-\(\tau\) 控制范围；
- Pearson \(\chi^2\) stiffness 是有动机的 modeling choice，不是唯一选择；
- trained attention 的 \(L_{\mathrm{eff}}^J\) 没有被直接测量；
- MLA 中 \(d_{\mathrm{eff}}=d_{\mathrm{head}}\) 是 architecture-specific convention。

因此最准确的一句话是：

> **The cosh allocation family is theory-derived; the \(d/\sqrt L\) dependence is conditionally proxy-motivated; the deployed unit-prefactor \(\tau\) is an empirically supported basin selector.**

### 2.2 ordinary KL 错误必须怎么处理

`[数学事实]` 令 \(z_\theta=z_0+\theta g+O(\theta^2)\)，则

\[
D_{KL}(p_0\|p_\theta)
=\tfrac12\theta^2g^T J_{\mathrm{sm}}(p_0)g+O(\theta^3)
=O(\tau^4),
\]

一阶为零。当前 `paper/sections/03_theory.tex:93-108` 把 \(O(\tau^2)\) term 称为 ordinary post-softmax KL gain，是错误命名；两个 \(O(\tau^4)\) 项不能产生小而非零的 optimum。完整核查见 theory audit `:217-293`。

安全处理是承认 ordinary-KL order error，同时把 \(q/L\) 降格并精确定义为 probability-displacement / per-position Fisher transport proxy。不能用 99-run sweep 证明 KL 推导正确。

### 2.3 Reviewer-facing 回答核

> We agree that the deployed \(\tau\) is not an end-to-end theoretical optimum. The exact result is the cosh density for our stated convex surrogate. Separately, under a diffuse-softmax, channel-additive phase-transport proxy, the leading balance motivates the \(d/\sqrt L\) scaling. We also identified that calling its linear term an ordinary KL gain was incorrect: ordinary baseline-to-perturbed KL starts at \(O(\tau^4)\). We therefore treat \(\tau=d_{\mathrm{eff}}/\sqrt L\) as a proxy-motivated, empirically supported basin selector, not a global optimum or trained-task theorem.

只有 reviewer 实际询问 \(\tau\)、KL 或 optimality 时才使用；否则不要主动把 rebuttal 变成理论勘误长文。

## 3. 理论主轴二：surrogate 与原始 RoPE phase-collision kernel 的关系

### 3.1 “Exact kernel” 实际测量什么

论文定义

\[
K(\phi,\psi)
=\mathbb E_{\Delta\sim D}
[\cos(\omega(\phi)\Delta)\cos(\omega(\psi)\Delta)],
\qquad \omega(\phi)=b^{-\phi}.
\]

`[数学事实]` 这是所选 distance prior 下 **cosine phase-response 的 Gram kernel**。若两个频率在训练可见距离上产生高度相关的 cosine pattern，它们的 \(K(\phi,\psi)\) 较大，表示有限 channel budget 中的 phase-basis redundancy。

它与 RoPE 有实际关系，因为单个 RoPE pair 对 relative position 的 logit contribution 正是 cosine/sine phase 的线性组合；appendix 的固定 activation logit derivative明确包含两者：`paper/appendix/a1_proofs.tex:457-475`。

但它不是完整 attention 或语言模型 kernel：

- 它是 content-independent 的，未包含训练后 \(Q/K\) amplitudes、head/layer分布和 task gradient；
- 当前定义只取 cosine-coordinate Gram。完整二维 phase pair \((\cos,\sin)\) 的 cross-frequency overlap会涉及 \(\cos((\omega_i-\omega_j)\Delta)\)，而 cosine-only kernel还含 \(\omega_i+\omega_j\) 项；
- 它依赖 distance prior \(D\)。current functional validation主要使用 uniform prior；更换真实 attention-distance prior会改变 kernel；
- 因此它是 **pre-training phase redundancy proxy**，不是 canonical “original RoPE loss”。

### 3.2 Surrogate 怎样与 exact kernel 接上

论文用

\[
K_{\mathrm{app}}(\phi,\psi)
=\alpha\delta(\phi-\psi)+\beta\min(\phi,\psi)
\]

表示 discrete diagonal ridge + smooth cumulative off-diagonal covariance，并最小化

\[
\mathcal C_{\mathrm{app}}[\rho]
=\tfrac12\langle\rho,K_{\mathrm{app}}\rho\rangle.
\]

这个连接有三层不同强度：

1. **结构动机存在。** `min` 是 mixed-boundary Laplacian 的 Green kernel；\(\delta\) 项惩罚 density concentration，\(\min\) 项惩罚低频端的 cumulative spectral mass。两者平衡自然产生 cosh density。
2. **不是 pointwise 或 global operator approximation。** 论文自己在 `paper/sections/03_theory.tex:23-32` 和 `paper/appendix/a1_proofs.tex:108-124` 承认 exact kernel 是 oscillatory，smooth two-term surrogate不能逐点逼近。
3. **现有支持是 directional functional validation。** 在 12 个已有配置中，deployed EVQ allocation 在 exact cosine Gram 上降低 normalized off-diagonal collision 24–92%并提高 effective rank：`paper/appendix/a1_proofs.tex:119-158`。这说明 surrogate 给出的方向与该 exact-kernel diagnostic 一致，但没有证明两个 objective 有相同 minimizer。

### 3.3 当前必须明确承认的 objective gap

`[数学事实]` surrogate theorem最小化的是 \(\langle\rho,K_{\mathrm{app}}\rho\rangle\)；appendix 的 exact-kernel validation报告的是

\[
C_{\mathrm{norm}}
=\sum_{i<j}\frac{K_{ij}^2}{K_{ii}K_{jj}}.
\]

一个是线性 quadratic form，另一个是 squared、normalized、off-diagonal statistic。当前仓库没有 theorem 证明前者的 minimizer也是后者的 minimizer，更没有证明它最小化 PPL。

此外，`paper/appendix/a1_proofs.tex:306-329` 已承认：把 \(\alpha,\beta\) 拟合到 exact kernel 后，surrogate 自己给出的 scaling约为 \(\sqrt d\,L^{-0.11}\)，并不能推出 deployed \(dL^{-1/2}\)。所以 deployed EVQ 的准确身份是：

1. surrogate 决定 **可解释的 cosh family / redistribution direction**；
2. separate transport proxy 与已有 sweep 决定 **family 中的 operating point**；
3. exact-kernel diagnostic 与 trained results做 **a posteriori directional validation**。

它不是“从 exact RoPE kernel 一步推导出 deployed schedule”。

### 3.4 `c_coll=1.171` 不应作为 rebuttal 防线

当前 `paper/tables/table_lambda_cv.tex` 把一组数称为 exact-kernel collision-score minimizer，并据此声称与 \(c_{\mathrm{pred}}\) 在 2% 内一致。但当前唯一对应脚本 `scripts/analysis/verify_c_coll.py:1-8,39-65` **没有计算 collision minimizer**；它把表中预置的 `tau_coll` 再读入，只重算 \(Q_1\) 和比值。

仓库较早的独立诊断还明确记录：static collision optimum显著大于 deployed \(\tau\)，且没有 tested static allocation objective复现 \(L^{-1/2}\) exponent：`docs/tau_algor/TAU_SCALING_DERIVATION.md:194-226`。因此在 provenance 未闭合前：

- 不引用 `c_coll=1.171` 作为 exact-kernel optimum；
- 不说 exact kernel闭合了 unit prefactor或 \(L^{-1/2}\)；
- 只使用 12-config 的 fixed-allocation directional result：EVQ@deployed \(\tau\) 比 Geo 有更低的已有 collision diagnostic。

这不是要求补实验；它是对现有理论证据边界的纠正。

### 3.5 EVQ 本质上优化了什么

最准确的分层回答：

| 层次 | EVQ 优化/改善的对象 | 证据强度 |
| --- | --- | --- |
| 严格数学 | 对给定 \(\beta/\alpha=\tau^2\) 的 convex surrogate，平衡 density concentration \(\int\rho^2\) 与 cumulative low-frequency overlap \(\int(\int_s^1\rho)^2ds\) | exact theorem |
| 信号处理解释 | 把一部分 near-static / highly redundant low-frequency channels迁向训练长度内有更充分 phase variation 的区域，提高有限 channel set 的 phase diversity | exact definition + proxy interpretation |
| 已有 diagnostic | 在论文列出的 12 个配置中降低 exact cosine-Gram 的 normalized off-diagonal collision并提高 effective rank | empirical directional validation |
| 训练结果 | 在列明的任务/模型/seed protocol中改善 PPL或 retrieval | empirical only |

明确不应说：EVQ closed-form minimizes the exact RoPE kernel、full attention loss、LM objective、PPL，或所有真实 distance priors下的 channel collision。

### 3.6 Reviewer-facing 回答核

> Our exact kernel is a content-independent Gram kernel of RoPE cosine phase responses under a specified distance prior, so it measures redundancy among the finite phase channels rather than the full attention or language-model objective. The \(\delta+\min\) model is a tractable surrogate for its discrete ridge and cumulative off-diagonal structure; the cosh density is the exact minimizer of that surrogate, not of the oscillatory kernel itself. Existing exact-kernel diagnostics show that the resulting fixed allocation reduces normalized channel collision, but this is directional validation, not an equality of objectives. Operationally, EVQ reallocates near-static low-frequency channels toward frequencies that exhibit more phase variation over the training window, thereby improving finite-channel phase diversity.

若 reviewer 继续追问“那理论贡献还剩什么”，回答：**一个明确的、可解的 frequency-allocation surrogate及其 closed-form optimizer，加上对 exact phase-redundancy和 trained behavior的分层验证**；不要声称从 exact kernel 到 task loss 的闭环理论。

## 4. 按 rebuttal 价值分级

### P0：必须准备，但只在 reviewer 触发时回答

#### P0-A — \(\tau\) 的 theory/heuristic 混合

- **可能提问**：\(\tau=d/\sqrt L\) 到底是 theorem、fit 还是 heuristic？
- **真实风险**：正文把 shape theorem、transport proxy 与 empirical basin写得过近；ordinary-KL命名还有真实错误。
- **现有证据**：本文件 §2；theory audit；Phase 16 只作 basin support。
- **当前材料**：数学长文充分，短回答尚需压缩。
- **策略**：明确三层身份；承认 KL order error；不守 global optimality。
- **需要做的事**：整理 120–180 词 answer kernel与公式指针；不为掩盖理论边界而追加无关推导或实验。
- **触发信号**：reviewer 点名 \(\tau\)、KL、prefactor、optimality、small-\(tau\) 或 MLA dimension。

#### P0-B — exact kernel、surrogate 与真实优化对象

- **可能提问**：为什么 \(\delta+\min\) 与原始 RoPE相关？EVQ到底优化了什么？
- **真实风险**：pointwise approximation不成立；surrogate functional与 exact diagnostic不同；deployed \(\tau\) 也不是 fitted surrogate optimum。
- **现有证据**：本文件 §3；`paper/appendix/a1_proofs.tex:108-158,306-329,420-451`。
- **当前材料**：paper有部分 caveat，但 `c_coll` 表述过强，旧 rebuttal材料没有把 objective gap讲清。
- **策略**：把 exact kernel定位为 phase-redundancy Gram；把 cosh定位为 surrogate optimizer；只守 directional exact-kernel validation。
- **需要做的事**：冻结一张 “optimizes / does not optimize” 对照和短回答；独立的 8B 机制实验只检验模型能否适应频率重分配，不得冒充 exact-kernel theorem。
- **触发信号**：reviewer 问 collision kernel、distance prior、surrogate validity、mechanism或 task relation。

#### P0-C — baseline / provenance 触发后的 trust repair

- **可能提问**：DAPE/YaRN是否忠实、Primary II 是否复现、PK是否 exact？
- **真实风险**：旧 Table 4 “DAPE” 实际是 32-parameter learnable frequency control；Primary I 使用 repo-defined smoothstep overlay；复现 docs有协议错配。
- **现有证据**：历史 runner `8616af4`；`scripts/text_eval/eval_454m_multilength.py:123-142`; curated raw artifacts。
- **当前材料**：结果值可追溯，方法 identity 与 end-to-end runner不完整。
- **策略**：只纠正和收窄：旧 DAPE行 relabel；YaRN称 repo-defined progressive overlay；TF/AR分开；不声称 tuned dominance或 full reproduction。
- **需要做的事**：准备 factual errata map；只在能改变因果解释且协议可严格匹配时补最近控制，不启动 broad baseline grid。
- **触发信号**：reviewer 实际质疑 baseline、seed、metric、code或 reproducibility。

### P1：有真实风险，但等 reviewer 原话再展开

| 风险 | 可能问题 | 现有回答 | 暂不做什么 |
| --- | --- | --- | --- |
| Primary II single seed | 为什么 seed-42 是 primary？ | 明确是 PE-dominant diagnostic；fixed EVQ额外 seeds不能升级整张表 | 不补 seeds，不把 \(L=256\) 当 replication |
| tuned scaler / scale | tuned YaRN会否消除优势？是否规模太小？ | 只守 fixed-scale overlay；承认 frontier-scale / tuned baseline缺失 | 不跑无触发的 scale grid；8B 机制实验不用于宣称 tuned-scaler dominance |
| metric / capability | 100% PK是否 exact generation？ | PK=TF NLL-gap；同时给已有 8K AR和4K reversal | 不新增 benchmark，不隐藏 seed spread |
| novelty | 是否只是调 base、插值或 search？ | 用 stage/object/DOF 区分；保持可组合性口径 | 不做 broad related-work rebuttal或组合 zoo |
| midpoint Geo | 是否非标准 RoPE baseline？ | 承认 matched midpoint control，用于隔离 shape | 不补 native endpoint训练 |
| MLA \(d_{\mathrm{eff}}\) | 为什么用 \(d_{\mathrm{head}}\)？ | calibrated convention，只支持 stated setting | 不补 ablation，不称 theorem |
| undertraining | 短 token预算是否制造效应？ | 报 exact budgets、negative/reversal boundary、证据 tier | 不用 supporting LoRA/1B声称已关闭 |

### P2：当前明确排除

- 无匹配 Geo 控制、无任务梯度或只做隐藏态蒸馏的新 LoRA 实验；
- video、progressive、750M、1B/4K、LongRoPE2/FIRE/CARoPE 组合扩张；
- 新模型族、production-scale pretraining，以及在 seed-42 未给出方向前机械增加 seed；
- 新 Bessel/forcing/global exact-kernel theorem或 \(L_{\mathrm{eff}}^J\) 测量；
- broad tuned-base / scaler zoo；
- 没有真实 reviewer trigger的预制 author response。

排除原因统一为：这些工作当前不能以足够低的成本直接改变核心机制或因果判断。它们不是被“rebuttal 禁止”，而是没有达到当前的 information-gain gate。

## 5. 现有 rebuttal 材料怎么用

| 材料 | 状态 | 用法 |
| --- | --- | --- |
| `THEORY_REBUTTAL_MATHEMATICAL_AUDIT_20260711.md` | **充分** | 理论事实权威；压缩，不再扩写 |
| `REVIEWER_TRIAGE_PLAYBOOK.md` | **流程充分** | 保留真实 review ID、correction/concession/evidence/boundary模板 |
| `REBUTTAL_MASTER_QUESTION_LEDGER_20260711.md` | **太广、部分过时** | 只作索引；LoRA优先级、DAPE/YaRN与 kernel链由本文覆盖 |
| `LORA_GEO_CONTROL_RESULT_AUDIT_20260711.md` | **supporting-only** | 仅 reviewer点名LoRA时启用 |
| 历史 paper-local rebuttal playbook（仅 Git 历史） | **退役** | 含过强理论与 comparator 陈述，不再复制或恢复 |
| `simulated_reviews/` | **内部压力测试** | 不作为真实 opinion或 score依据 |
| curated Primary I/II/III artifacts | **数字可用、provenance分层** | 只沿 manifest边界引用，不反推出缺失 runner/checkpoint |

## 6. 7 月 22 日前的最小行动清单

回答准备仍是主线，同时只保留一个机制实验队列：

1. **冻结两个理论短答。**
   - \(\tau\)：exact shape / conditional scaling / empirical basin。
   - kernel：phase-redundancy Gram / surrogate optimizer / directional validation。
2. **冻结禁止使用的理论证据。**
   - ordinary KL \(O(\tau^2)\) gain；
   - exact-kernel global minimizer `c_coll=1.171`；
   - exact kernel、PPL或 task loss的 closed-form optimum；
   - waterbed inequality证明 PPL trade-off。
3. **运行有 gate 的 8B 频率适应实验。** 协议见 `frequency_adaptation_8b/SPEC.md`：先在原生 Geo 下确认 answer-only 检索梯度足以让 q/k/v/o LoRA 学会任务，再从同一 checkpoint 分叉 Geo 与 EVQ，通过连续频率路径完成适应；seed-42 无明确能力信号就停止，不把训练启动或 PPL 变化写成成功。这是独立机制协议，不替代 paper-lineage LongAlpaca clean pair。
4. **整理现有证据索引。** 每个 Primary claim只保留 model、length、tokens、seeds、metric、artifact与边界；新实验数字必须单列 provenance 与 evidence tier。
5. **准备 factual correction map。** DAPE label、repo-defined YaRN-style overlay、TF/AR、Primary II exact-runner缺口；只有 reviewer问到才使用。
6. **预演 response budget。** 真实 reviews到来后只选 3–5 个 score-driving concerns；其余写 `not triggered`，不进入回复。

## 7. 真实 reviews 到来后的条件分支

| Trigger | 回答路径 | 必须停止的位置 |
| --- | --- | --- |
| \(\tau\)/KL/optimality | §2 + P0-A | 到 basin selector为止，不扩成 trained-task theorem |
| exact kernel/surrogate/mechanism | §3 + P0-B | 到 directional validation为止，不声称 objective等价 |
| DAPE/YaRN/baseline | P0-C factual correction | 不承诺临时结果，不用近似实现补洞；已完成的匹配控制须过 provenance gate |
| seed/scale/undertraining | P1相应边界 | 现有 evidence tier之外一律让步 |
| PK/downstream capability | TF/AR + negative boundary | 不新增 benchmark，不泛化到 production |
| novelty | stage/object/DOF对照 | 不用“orthogonal”回避具体重叠 |
| MLA | stated convention | 不把 \(d_{\mathrm{eff}}\) 升为 theorem |

## 8. 外部真实评审的校准

- [The Impact of Positional Encoding on Length Generalization, NeurIPS 2023](https://openreview.net/forum?id=Drrl2gcjzl)：真实 review集中在 scope、LM/task外推、规模与 novelty；清楚限定 scope比继续扩大主张更有效。
- [Scaling Laws of RoPE-based Extrapolation, ICLR 2024](https://openreview.net/forum?id=JO7k0SJ5V6)：reviewer关注 YaRN novelty、PPL-only evaluation和实际任务；直接回答被问 endpoint有效，但模型族限制仍保留为评分上限。
- [Probing RoPE through Frequency Entropy, ICLR 2026](https://openreview.net/forum?id=1JZuEDq62N)：reviewer抓 causal confound和 practical utility；matched control能说服，而“潜在应用”不能替代直接证据。

对本论文的唯一迁移结论是：**回答 reviewer指出的最近因果/理论缺口，诚实收窄；不要用额外但不直接相关的材料制造 breadth。**

## 9. 当前无法完全解决的真实风险

1. surrogate 与 exact phase kernel之间没有 pointwise、operator-norm或 shared-minimizer theorem；只有结构动机与已有 directional validation。
2. phase kernel不是 full RoPE attention/task objective；真实 \(Q/K\) amplitudes、sin/cos pair、distance prior和训练动力学都在外部。
3. \(d/\sqrt L\) 的 unit prefactor和 practical finite \(\tau\)是经验的；conditional proxy不能消除这一点。
4. Primary I/II 的 comparator identity、短训练预算与 exact runner缺口只能纠正/让步，不能在 rebuttal中补齐。
5. reviewer可能认为理论—任务链仍过松或 empirical scope过窄。这是应接受的评分风险，不能靠发散回答解决。

## 10. 最终发送门

- [ ] 每段绑定真实 reviewer原话。
- [ ] 任何新实验或新数字都由 reviewer 原话直接触发，并已通过匹配协议、provenance、负结果与 evidence-tier 核验；否则不进入 response。
- [ ] \(\tau\) 的 exact / conditional / empirical 三层未混写。
- [ ] ordinary KL一阶为零已正确处理。
- [ ] exact kernel只称 phase-redundancy proxy；surrogate与 exact diagnostic未写成同一 objective。
- [ ] 未引用 `c_coll=1.171` 作为 global exact-kernel optimum。
- [ ] DAPE/YaRN/PK/seed/protocol identity按现有事实写，未强行补洞。
- [ ] supporting LoRA/video/progressive未升级。
- [ ] 没有 universal SOTA、global optimum、tuned dominance或 end-to-end theory closure。
