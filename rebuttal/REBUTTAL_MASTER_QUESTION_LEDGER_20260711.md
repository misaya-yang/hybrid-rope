# EVQ-Cosh Rebuttal Master Question Ledger

日期：2026-07-11

Document status：**living internal document**

Preparation：**triage_ready**

Response package readiness：**needs_author_input**

Current mode：**triage-only**；截至 2026-07-12，实际 NeurIPS reviews 尚未收到。

用途：把最新数学审计、Fable5/GPT Pro 模拟审稿、现有 claim/evidence ledger、报告一致性审计，以及 2026-07-11 的 LoRA Geo-control 结果合并为一份可持续更新的 rebuttal 问题总账。它不是最终 author response，也不是第二篇论文；真实 reviews 到来前只做 triage，真实 reviews 到来后只抽取被实际触发的条目进入回复。

---

## 0. 使用方式与证据优先级

### 0.1 本文件覆盖的四类问题

1. **理论正确性**：什么是精确定理，什么只是 conditional proxy，什么只是经验 operating rule；
2. **因果识别与 baseline fairness**：EVQ 相对 Geo、YaRN、DAPE、MLA 和 LoRA 的比较到底隔离了什么；
3. **报告与 provenance 信任**：图表、数据集、seed、artifact 和提交版/修订版之间是否一致；
4. **应用与主张边界**：指标是否代表真实能力，何时能称工业相关，何时只能称机制诊断。

### 0.2 最新状态覆盖旧材料

不存在一个把 paper、数学与实验混在一起的线性“最高权威”。若旧文档与当前状态冲突，必须先按问题分轨：

| 轨道 | 当前权威 | 覆盖规则 |
| --- | --- | --- |
| Submitted/current wording | submitted artifact 与 `paper/` 当前源码，使用时明确版本 | 只证明该版本“写了什么”；不能覆盖理论数学审计，也不能单独升级实验 provenance |
| Experiment numbers/provenance | `docs/overview/RESULT_PROVENANCE_MANIFEST.md` 及其指向的 raw-backed artifact | 这是最新实验 provenance；已移出根目录的 2026-06-14 snapshot 只可在 ignored local snapshot 中追溯，不能覆盖它 |
| Theory correctness | `rebuttal/THEORY_REBUTTAL_MATHEMATICAL_AUDIT_20260711.md` | ordinary KL、transport proxy、finite-$\tau$ 边界以独立数学审计为准；paper source 不具有覆盖权 |
| LoRA identity/causality | `rebuttal/LORA_GEO_CONTROL_RESULT_AUDIT_20260711.md` | LongAlign/LongAlpaca identity 与 clean-pair gate 不得被历史 aggregate 覆盖 |
| Rebuttal decision | 本总账；快速分流用 `rebuttal/REVIEWER_TRIAGE_PLAYBOOK.md` | 旧 response 草稿与 action board 只能提供历史上下文，不能自动进入最终回复 |
| Simulation | `rebuttal/simulated_reviews/` 的字节保真原文 | 只证明内部压力测试问过什么；不是真实 review，也不裁决科学正确性 |

历史内部报告只用于 provenance 追溯，不能覆盖对应轨道的当前权威。

两项重要 supersession：

- 旧理论草稿中“ordinary baseline-to-perturbed KL 提供 $O(\tau^2)$ gain”的段落已经失效。新的正确身份是：**exact surrogate shape theorem + conditional diffuse probability-transport proxy + empirical finite-$\tau$ deployment rule**。
- 旧 Path A/B 文档把 LoRA 阻塞描述为“缺 Geo+LoRA exact numbers”。今天 fresh native-Geo 数字已经完整；contemporaneous records 指向历史 EVQ 使用 LongAlpaca-12k，但 exact historical corpus identity 尚未通过 raw hash/runtime 核验，而 fresh Geo 已确认使用 pinned official LongAlign-10k。准确状态是：**Geo result complete; current artifacts do not establish a strict matched pair; fresh matched EVQ-42 完成前不得提出 reviewer-facing causal claim**。

### 0.3 标签

Likelihood 不是统计概率，而是根据两个独立模拟 reviewer、论文可见性和攻击面的自然程度做的相对排序：

- **VH**：极高概率；多个 reviewer 独立提出，或一眼可见的 trust/causal issue；
- **H**：高概率；至少一类 reviewer 很可能追问；
- **M**：中等概率；通常由理论 reviewer 或深挖 appendix 的 reviewer 提出；
- **L**：低概率；只在 reviewer 明确进入该支线时回答。

Impact：

- **Blocking**：处理不当可能直接维持 reject；
- **Major**：可以改变一档评分或 AC 信任；
- **Moderate**：影响说服力和 scope，但通常不是单独否决项；
- **Minor**：修订级问题。

Status 只允许：

- **READY**
- **READY_WITH_CONCESSION**
- **PARTIAL**
- **EXPERIMENT_PENDING**
- **THEORY_CORRECTION_PENDING**
- **PROVENANCE_BLOCKED**
- **DROPPED**

### 0.4 证据层级

- **Primary**：Primary I–III，严格保留论文声明的 seed/protocol 范围；
- **Robustness**：同一主协议的多 seed 或明确 robustness row；
- **Supporting**：单 seed、小模型、相邻协议、LoRA、video、progressive 等；
- **Internal-only**：尚未进入匿名 result bundle、仅历史报告或服务器文件；
- **Negative boundary**：1B reversal、small-base Geo win、NTK anti-composition、QuALITY near chance、RULER non-improvement 等，必须与正证据同时保留。

### 0.5 本次合并的主要输入

| Source | 在本总账中的作用 |
| --- | --- |
| rebuttal/THEORY_REBUTTAL_MATHEMATICAL_AUDIT_20260711.md | 最新数学事实、KL纠错、三层理论身份和理论send gate |
| rebuttal/LORA_GEO_CONTROL_RESULT_AUDIT_20260711.md | 今天fresh Geo结果、LongAlign/LongAlpaca mismatch、clean-pair decision gate |
| rebuttal/simulated_reviews/2026-07-10_fable5_committee_output.md | Fable5内部模拟原文；只用于压力测试触发面 |
| rebuttal/simulated_reviews/2026-07-10_gpt_pro_committee_full_v2.md | GPT Pro三类模拟reviewer与AC原文；只用于压力测试触发面 |
| Legacy response / claim / action / figure audits | 有效问题映射、禁句、stop rules 与 errata 已吸收进本总账；原文件删除后只可从 Git 历史追溯，不再作为入口或独立权威 |
| paper/sections/01_intro.tex、03_theory.tex、05_experiments.tex、06_limitations.tex及proof appendix | 当前论文实际主张、数字和限制 |

Fable5/GPT Pro 文件是内部模拟与压力测试，不是真实 NeurIPS reviews。它们用于提前覆盖高概率攻击面；7月22日真实reviews到来后，必须重新映射，不能把模拟问题当作reviewer原话。

在真实 reviews 到来前，本总账只允许 triage、证据核验和作者决策准备；不得据模拟问题创建最终 author response。

---

## 1. 当前总判断

### 1.1 论文还能稳健保住什么

最稳健的中心主张仍是：

> RoPE is not only a positional operator or a range-scaling target; its finite frequency table is also a spectral budget. EVQ-Cosh treats training-time frequency allocation as a third PE design axis, complementary to operator design and inference-time range scaling.

可以坚定保留：

- 对论文明确写出的强凸 surrogate，cosh density 是唯一、严格正的 mass-one minimizer；
- closed-form CDF、inverse CDF、$\tau\to0$ geometric limit 与 zero-learned-parameter initializer；
- Primary I 的 3-seed、matched-scale EVQ $\times$ YaRN substrate/range interaction；
- Primary II 在明确 seed-42、$128\to8$K 协议下的 PE-dominant diagnostic；
- Primary III 的 3-seed、432M/500M-token MLA scarce-channel stress result；
- 99-run 对 operating basin 的经验支持；
- PK 与 AR exact 的严格区分，以及已恢复的 Primary-I AR exact 数据。

必须主动收缩或纠正：

- ordinary-KL scale derivation；
- shape theorem 与 deployed scale 的统一最优解释；
- waterbed 到 PPL 的理论跨越；
- practical $\tau=4$ 的 small-$\tau$ 误差保证；
- MLA $d_{\mathrm{eff}}=d_{\mathrm{head}}$ 的“理论唯一性”；
- tuned-base、tuned-YaRN、Primary-II matched seeds、真实 distance prior 和 trained $L_{\mathrm{eff}}^J$；
- LoRA 历史 EVQ 与 fresh Geo 的因果比较；
- QuALITY/Figure 9 的提交版 provenance 错误；
- universal downstream、production、latency/FLOP 或 scaler-agnostic claim。

### 1.2 当前 response package 为什么仍是 needs_author_input

真实 reviews 尚未收到，因此 package 先天需要作者输入；下面的 blocker 进一步表示“某类强 claim 不得进入 response”，不是“没有这些实验就不能提交 rebuttal”。

1. **理论文字 blocker**：所有旧 response 中的 ordinary-KL $O(\tau^2)$ 解释必须替换；
2. **LoRA causal blocker**：fresh EVQ-42 尚未在同一 frozen LongAlign manifest 上完成；
3. **LoRA provenance blocker**：fresh result JSON、manifest 和 evaluator receipt 尚未进入匿名 tracked bundle；
4. **trust blocker**：最终回复必须同时披露 Figure 8/Table 21 与 Figure 9/Table 20 的真实来源错误；
5. **scope blocker**：Primary II、tuned baselines、MLA convention 和 1B row 必须保持当前真实范围。

### 1.3 最可能改变评分的 18 个问题

| Rank | ID | Canonical reviewer concern | Likelihood | Impact | Status | 最小正确动作 |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | P-01 | 多处图表/provenance 错误后，为什么还应信任其余结果？ | VH | Blocking | PARTIAL | 三项 errata、source of truth、保留/暂停的 claim 一次说清 |
| 2 | E-08 | fresh Geo 与历史 EVQ 数据不同，LoRA 因果比较是否失效？ | VH | Blocking | PROVENANCE_BLOCKED | fresh EVQ-42 on identical LongAlign；此前不报 EVQ delta |
| 3 | T-01 | ordinary KL 一阶为零，$L^{-1/2}$ 推导是否错误？ | H–VH | Blocking | THEORY_CORRECTION_PENDING | 承认 order error，重定义 transport proxy |
| 4 | E-01 | EVQ 是否只是在修复过大的 Geo base？ | VH | Major | PARTIAL | tuned-base 缺口承认；base-10K pilot 只作 supporting |
| 5 | E-02 | Geo/EVQ 分别调 YaRN 后，Geo 是否追平？ | H–VH | Major | READY_WITH_CONCESSION | 只保 matched-scale interaction；不写 tuned dominance |
| 6 | E-03 | Primary II 只有 seed 42，为什么称 primary？ | VH | Major | EXPERIMENT_PENDING | 补 exact protocol seeds，或保持 seed-scoped diagnostic |
| 7 | E-04 | DAPE-style baseline 是否忠实、充分调参？ | H | Major | PARTIAL | 单独给实现与 tuning boundary；不可由 EVQ extra seeds替代 |
| 8 | T-02 | cosh shape 与 deployed $\tau$ 来自不同模型，是否 post-hoc？ | VH | Major | READY_WITH_CONCESSION | theorem/proxy/calibration 三层拆开 |
| 9 | T-04 | constant-$\alpha$ + min kernel 是否只为得到 cosh？ | H | Major | PARTIAL | defend conditional theorem + finite-grid functional check |
| 10 | E-06 | 1B raw reversal 是否说明 500M 增益只是训练暂态？ | H–VH | Major | READY_WITH_CONCESSION | 定位为 schedule sensitivity，不作 durability claim |
| 11 | E-07 | MLA 为何用 $d_{\mathrm{head}}$，channel scarcity 是否隔离？ | H | Major | EXPERIMENT_PENDING | convention 与 $K$ 分开；direct-$\tau$ screen |
| 12 | E-05 | 100% PK 是否只是 teacher-forced 指标？ | VH | Major | READY | 同列 TF 与 AR exact、seed spread 和 4K boundary |
| 13 | T-05 | practical $\tau=4$ 时 small-$\tau$/pure-tether 为什么可信？ | H | Major | PARTIAL | exact implementation；无 non-asymptotic/forcing guarantee |
| 14 | P-02 | Figure 8/Table 21 的真实 endpoint 和样本数是什么？ | VH | Major | READY | submitted error + n=2086 source + 26.6→24.6 erratum |
| 15 | P-03 | Figure 9 的 -81.2% 与 Table 20 的 -13.3% 哪个正确？ | H | Major | READY | -81.2 属 progressive；454M row 为 -13.3 |
| 16 | P-04 | 历史 raw/checkpoint 不完整，Primary I–III 能否复现？ | H | Major | PARTIAL | 区分 reproducibility 与 exact historical provenance |
| 17 | A-01 | 除 positional diagnostics 外，有何真实长上下文能力？ | H | Major | READY_WITH_CONCESSION | 8K AR exact；同时保留 QuALITY/RULER 边界 |
| 18 | AC-01 | 纠正这些问题后，贡献还剩什么，为什么值得接收？ | H | Blocking at AC | READY_WITH_CONCESSION | 精确核心 + 三个 empirical anchors + bounded novelty |

---

## 2. 2026-07-11 两项关键新事实

### 2.1 理论审计快照

设 $\theta=\tau^2$ 且

$$
z_\theta=z_0+\theta g+O(\theta^2).
$$

ordinary baseline-to-perturbed KL 的一阶变分严格为零：

$$
D_{\mathrm{KL}}(p_0\|p_\theta)
=\frac{\theta^2}{2}g^\top J_{\mathrm{sm}}(p_0)g+O(\theta^3)
=O(\tau^4).
$$

因此，不能再用 “ordinary KL 的 $O(\tau^2)$ gain 与 Pearson $O(\tau^4)$ stiffness 平衡” 得出小而非零的 operating point。

可以保留的是另一个明确对象。在 diffuse uniform softmax 下，单通道 phase pattern $c_\omega$ 满足

$$
\|J(p_0)c_\omega\|_2^2\simeq \frac{q(\omega L)}{L}.
$$

在 channel amplitude 固定、可加且 cross terms 可忽略等条件下，

$$
U_{\mathrm{tr}}(\rho;L)=\frac{M}{L}\int q(Lb^{-\phi})\rho(\phi)\,d\phi
$$

沿 $\rho_\tau=1+\tau^2\eta+O(\tau^4)$ 有非零一阶 allocation-score variation。它可以条件性给出 $\tau\propto M/\sqrt L$，但它不是 ordinary KL、task loss 或 trained-attention theorem。

当前最准确的三层身份：

1. exact cosh optimizer under the stated surrogate；
2. conditional diffuse probability-transport proxy；
3. empirically calibrated finite-$\tau$ basin selector。

### 2.2 今天 fresh Geo+LoRA seed-42

协议：LLaMA-3-8B-Instruct，native geometric RoPE，pinned official LongAlign-10k，300 steps，q/k/v/o LoRA，rank 64，alpha 128，BF16，seed 42。

| Context | Base NLL / PPL | Geo+LoRA NLL / PPL | NLL delta |
| --- | ---: | ---: | ---: |
| 8K | 2.004237 / 7.4204 | 1.868129 / 6.4762 | -0.136108 |
| 16K | 5.170501 / 176.0030 | 4.757525 / 116.4573 | -0.412976 |
| 32K | 7.572270 / 1943.5477 | 7.094225 / 1204.9877 | -0.478046 |

五个 disjoint chunks × 三个长度的 15 个 paired NLL deltas 全部 favor Geo+LoRA。这证明：

- 普通 long-sequence LoRA adaptation 本身就能改善这个协议下的 extrapolation PPL；
- Base→EVQ-LoRA 的全部变化不能自动归因于 EVQ；
- chunk consistency 是 evaluator-stability check，不是五个 seed。

但历史 EVQ seed-42 的 contemporaneous record 指向 LongAlpaca-12k，fresh Geo 使用 official LongAlign-10k；旧 downloader 还可能把两种内容都写成 longalign_10k.jsonl。因此历史 9.63/21.5/104.3 与 fresh Geo 6.48/116.5/1205 的差不能作为 matched EVQ effect。

当前唯一干净的下一步是：在**同一 frozen LongAlign manifest、同一 seed 42、同一 token order、同一 LoRA/optimizer/evaluator**上 fresh run EVQ-Cosh $\tau=1.414$。在该 pair 出来前：

- 不写“matched EVQ cost is 49%”；
- 不写“LoRA cannot explain the gain”；
- 不写“PPL 104 proves usable 32K”；
- 不运行 EVQ seeds 43/44 来替代缺失的 matched seed-42 pair。

---

## 3. 理论问题详账

### T-01 — Ordinary KL 阶数与 scale law

**Canonical reviewer question**

> Your schedule perturbation is $O(\tau^2)$, but KL is stationary at the baseline. Should the first nonzero KL term be $O(\tau^4)$, invalidating the claimed $L^{-1/2}$ optimum?

- Likelihood / Impact：H–VH / Blocking
- Claim at risk：论文是否真正“derive”了 deployed scale；
- Current truth：reviewer 对 ordinary KL 是对的；旧解释必须撤回。conditional transport proxy 可保留；
- Answer kernel：先承认 zero first variation，再定义 $U_{\mathrm{tr}}$，最后把 practical rule 定位为 empirical basin selector；
- Positive evidence：exact cosh theorem、99-run basin；
- Counterevidence：practical values 远离 local regime，trained task bridge 未测；
- Unsafe wording：ordinary KL supplies the $O(\tau^2)$ gain；the sweep validates the derivation；
- Closure：纯文本/数学修正即可，不需新 GPU 实验；
- Evidence：THEORY_REBUTTAL_MATHEMATICAL_AUDIT_20260711.md §P0.1–P0.3；paper/sections/03_theory.tex；paper/appendix/a1_proofs.tex。

**English triage component（not send-ready）**

仅当真实 review 的逐字 trigger 到达、完成 comment mapping 且作者批准后，才可从本组件组装回复；不得直接复制发送。

> We identified an order error in our KL interpretation. Writing $\theta=\tau^2$, ordinary baseline-to-perturbed KL has zero first variation and begins at $O(\theta^2)=O(\tau^4)$, so it cannot by itself derive a nonzero operating point. The $O(\tau^2/L)$ term used by the scale model is instead an explicitly defined diffuse probability-transport allocation proxy. Under its channel-additivity and small-$\tau$ assumptions it conditionally motivates $\tau\propto M/\sqrt L$; practical finite-$\tau$ values remain empirically calibrated.

### T-02 — Shape、proxy 与 deployment 是否拼接

**Canonical reviewer question**

> The surrogate gives $\tau_{\mathrm{surr}}=\sqrt{\beta/\alpha}$, while the deployed rule scales as $d_{\mathrm{eff}}L^{-1/2}$. Are the shape and scale simply stitched together?

- Likelihood / Impact：VH / Major
- Current truth：必须部分承认。surrogate fit 近似给出 $\sqrt{d_{\mathrm{rot}}}L^{-0.11}$，与部署指数不同，不能用 $O(1)$ prefactor 吸收；
- Safe answer：surrogate 选 analytic shape family；transport proxy 只给 local scale motivation；sweep 选择 practical basin；
- What survives：closed-form family、inverse CDF、empirical utility；
- What does not：一个 full-attention objective 同时推出 shape 和 deployed scale；
- Closure：所有 response 和 epistemic map 使用同一三层叙事；
- Unsafe wording：both stages are one theorem；the exponent difference is only a prefactor。

### T-03 — $Q_1$、$q/L$ 与 task relevance

**Canonical reviewer question**

> Is $Q_1$ a KL curvature, a task gradient, or merely an allocation score, and why should channels add independently?

- Likelihood / Impact：H / Major
- Current truth：$Q_1$ 是 diffuse probability-transport/per-position Fisher allocation score 对 density 的一阶变分；
- Assumptions：uniform baseline、固定 amplitude、channel additivity、cross terms 平均消失、small $\tau$、tested-grid $Q_1>0$；
- Not established：真实 Q/K activation、task-gradient sign、non-diffuse generalization、universal positivity；
- Closure：理论回复必须定义对象，不能只改名为“utility”；
- Unsafe wording：$Q_1$ is Geo-to-EVQ KL curvature；$Q_1$ is downstream task gradient。

### T-04 — Surrogate 与 exact RoPE kernel

**Canonical reviewer question**

> Was the constant-diagonal plus min-kernel surrogate selected mainly because it yields a cosh solution, and do collision-score reductions prove that the exact optimizer is close?

- Likelihood / Impact：H / Major
- Current truth：它是 tractable surrogate，不是 exact oscillatory kernel 的 pointwise 或 continuum operator-norm approximation；
- Defensible evidence：cosh 是 stated surrogate 的 exact optimizer；tested finite grids 上 exact-kernel collision diagnostic 降低 24–92%；
- Missing：finite-dimensional optimizer residual、realistic-prior robustness、trained objective equivalence；
- Safe action：defend functional test，不把强凸性外推为 optimizer closeness；
- Unsafe wording：strong convexity proves the exact-kernel optimizer is close to cosh。

### T-05 — Small-$\tau$、practical $\tau=4$ 与 pure-tether forcing

**Canonical reviewer question**

> What non-asymptotic control justifies using a small-$\tau$ theory and dropping the forcing branch at practical $\tau=4$ or larger?

- Likelihood / Impact：H / Major
- Current truth：没有 practical-$\tau$ remainder guarantee。$\tau=4$ 时 local stiffness leading term 对 exact value 高估约 252%；
- Safe answer：training uses exact cosh frequencies；Taylor argument only motivates local structure；finite-$\tau$ selection is empirical；
- Forcing boundary：pure tether 是 tractable, exactly invertible design family；trained forcing amplitude、BC 和 quantile error 未测；
- Closure：rebuttal 中降级，不承诺临时补“统一非渐近 theorem”；
- Unsafe wording：the forcing residual is controlled/negligible at $\tau=4$。

### T-06 — Waterbed 是否推出任务 trade-off

**Canonical reviewer question**

> Does the waterbed inequality prove an in-range versus long-range PPL Pareto frontier?

- Likelihood / Impact：H / Major
- Current truth：只证明 allocation-space divergence；不推出 PPL；
- 必须纠正：$\mathcal C_{\mathrm{app}}[1]=\alpha/2+\beta/6\ne0$，不能写它与 waterbed 都在 uniform 处消失；
- Safe answer：PPL trade-off is empirical and consistent with, not implied by, the bound；
- Learnable $\tau$：可说与 in-range objective mismatch 一致，不能说 waterbed 必然驱动 $\tau\to0$；
- Closure：文本/公式修正；
- Unsafe wording：waterbed predicts the observed PPL penalty。

### T-07 — Pearson、0.465、0.500 与 $\lambda=1$

**Canonical reviewer question**

> Why Pearson stiffness, why is 0.465 treated as 0.500, and what independently fixes $\lambda=1$?

- Likelihood / Impact：H / Major
- Current truth：Pearson 是在 stated load-variance axioms 下的 motivated choice，不是唯一 attention theorem；
- 0.465：finite-range numerical effective exponent；0.500：local structural target，不能写成相等；
- $\lambda$：在当前 normalization 下只有 $\lambda Q_1$ 可识别，$\lambda=1$ 是 calibration convention；
- Positive evidence：不同规则仍落在宽 basin 的部分证据；
- Missing：直接 trained comparison of 0.465 schedule、独立 physical calibration；
- Unsafe wording：units derive $\lambda=1$。

### T-08 — MLA $d_{\mathrm{eff}}$ 与 finite-channel bounds

**Canonical reviewer question**

> Only $d_{\mathrm{rot}}$ dimensions are rotated. Why use $d_{\mathrm{head}}$, and do the $K^{-1}/K^{-2}$ bounds predict the MLA PPL gain?

- Likelihood / Impact：H / Major
- Current truth：$K=d_{\mathrm{rot}}/2$ 确定 quantile pairs；$d_{\mathrm{eff}}=d_{\mathrm{head}}$ 是 tested architecture 的 empirical convention；
- Bounds：只说明 scarce channels 放大 allocation/discretization sensitivity，不决定下游 PPL 的方向或量级；
- Missing：direct $\tau$ ablation、projection energy/effective rank/Fisher/task-gradient measurement；
- Safe claim：allocation sensitivity under the reported convention；
- Unsafe wording：MLA convention is uniquely derived；the bound predicts -31.1%。

### T-09 — $L_{\mathrm{eff}}^J$ 是否闭合 trained-attention theory

**Canonical reviewer question**

> Why can $L$ simply be replaced by $L_{\mathrm{eff}}^J$ when the derivation switches from channel patterns to the schedule derivative?

- Likelihood / Impact：M–H / Major
- Current truth：$L_{\mathrm{eff}}^J$ 是方向依赖的 quadratic Fisher diagnostic，不是一般 support length；
- 关键缺口：channel pattern $c_\omega$ 与 schedule derivative $g_\theta$ 被混用；nonuniform 下 $g^\top J^2g$ 与 $g^\top Jg$ 不再等价；
- Closure：把 measurement 保留为 falsification diagnostic，不说它自动修复 first-order theorem；
- GPU：通常是 checkpoint analysis，不应排在 acceptance-critical training 前。

### T-10 — Stationary phase、Bessel branch 与 positivity

**Canonical reviewer question**

> Are the stationary-phase normalization and the claim that the Bessel branch can become negative correct?

- Likelihood / Impact：M / Major if raised
- Current truth：causal/signed prior 有 factor-of-two convention；现有 Bessel negativity 理由未证明且被独立复算质疑；
- Main claim impact：主 cosh conditional theorem 不依赖该防御；
- Safe answer：constant $\alpha$ 的理由是 tractability、closed form 和 finite-grid functional behavior；
- Closure：revision 重推 normalization/BC；短 rebuttal 不用“Bessel 非法”作主防御。

### T-11 — Learnable $\tau$ 的负结果说明什么

**Canonical reviewer question**

> Does learned $\tau\approx1.14$ validate the fixed rule, or reveal that the fixed operating point is artificial?

- Likelihood / Impact：M–H / Moderate
- Current evidence：三个 seed 只有 final endpoints 1.1391/1.1445/1.1383，per-step trajectory 未恢复；
- Safe interpretation：与 in-range objective 看不见 OOD utility、flat basin 和 coupling 一致；
- Not established：optimizer-independent convergence、系统性向零漂移、waterbed causation；
- Closure：没有 trajectory 就不写 trajectory-level claim。

### T-12 — Cosh 是否优于所有其他非几何 allocation

**Canonical reviewer question**

> Do the experiments establish that the cosh form is specifically responsible, rather than merely showing that allocation matters?

- Likelihood / Impact：H / Major
- Current truth：证据更强地支持“allocation matters”和“EVQ-Cosh is a useful closed-form instance”，不支持所有 monotone/non-geometric family 中最优；
- Safe novelty：third axis + explicit variational surrogate + exact inverse-CDF implementation + tested results；
- Missing controls：linear、power-law、random monotone、numerically optimized schedules；
- Action：除非真实 reviewer 点名，不在 rebuttal 窗口启动大规模 shape zoo；
- Unsafe wording：cosh is the universally optimal allocation。

---

## 4. 实验与因果识别问题详账

### E-01 — Tuned geometric base

**Canonical reviewer question**

> Is EVQ only correcting an unnecessarily large geometric base, and would a validation-tuned Geo base recover the same active spectrum?

- Likelihood / Impact：VH / Major
- Current evidence：151.9M、$L=512$、seed-42 pilot 在 base 10K 和 500K 都保留方向；
- Counterevidence：video/small-base regime 中 Geo 可胜；$L=4096$ bare-rule negative boundary；
- Current answer：pilot 排除“方向只存在于 500K”，但不排除 best-tuned Geo；
- Closure：matched text Geo base grid，训练/评测协议固定，并报告 in-range 与 extrapolation；
- Decision gate：若 tuned Geo 追平，主张保留为 high-base/dead-channel regime 的机制；不得继续写 general dominance；
- Unsafe wording：EVQ beats the tuned geometric base。

### E-02 — Separately tuned YaRN

**Canonical reviewer question**

> Does the EVQ advantage persist when Geo and EVQ receive separately tuned YaRN scales and parameters?

- Likelihood / Impact：H–VH / Major
- What is already strong：same-scale 2×2 factorial，8K paired TF interaction +38/+42/+36 pp；
- Correct claim：under the same tested scale, YaRN has higher leverage on the EVQ-trained substrate；
- Missing：best-tuned Geo+YaRN leaderboard；
- Negative boundary：NTK-aware composition can reverse；
- Closure：eval-only scale sweep if checkpoints are available；
- Decision gate：若 best Geo catches up，保留 matched-scale interaction，不写 practical superiority；
- Unsafe wording：EVQ beats tuned YaRN；EVQ composes with every scaler。

### E-03 — Primary II matched seeds

**Canonical reviewer question**

> Why is a seed-42 Geo/DAPE/EVQ contrast treated as a primary anchor?

- Likelihood / Impact：VH / Major
- Current truth：printed Geo/DAPE/fixed-EVQ contrast是 seed 42；learnable $\tau$ 是 3-seed；fixed EVQ 的额外 seeds 不能升级整个 baseline comparison；
- Protocol trap：$L=256$/100M multi-seed 不能冒充 $L=128$/15M Table 4；
- Correct answer：seed-42 PE-dominant diagnostic, not broad DAPE dominance；
- Closure：Geo/DAPE/EVQ seeds 137/256 under exact same protocol；
- Decision gate：补齐则报 paired mean/std；未补齐则 re-tier/supporting；
- Unsafe wording：Primary II is fully replicated。

### E-04 — DAPE fidelity 与 baseline tuning

**Canonical reviewer question**

> Is “DAPE-style” a faithful and reasonably tuned reproduction of the strongest DAPE configuration?

- Likelihood / Impact：H / Major
- Current state：与 seed 问题不同，是 baseline implementation/fidelity 问题；
- Required answer：exact DAPE module、position operator、optimizer、token/length protocol、tuned/fixed hyperparameters和 source；
- Missing evidence：当前总账没有足够证据支持“best DAPE”；
- Safe claim：competitiveness in the retained DAPE-style protocol；
- Closure：先文档化 implementation；只有 reviewer 指名且 entrypoint可靠时再补 tuning；
- Unsafe wording：EVQ comprehensively dominates DAPE。

### E-05 — Teacher-forced PK 与 AR exact

**Canonical reviewer question**

> Is the headline 100% passkey number actual generation accuracy?

- Likelihood / Impact：VH / Major
- Status：READY，无需重跑；
- 8K：Geo+YaRN TF 61.3%，AR exact 0/0/0；EVQ+YaRN TF 100%，AR exact 58/18/98%，mean 58%；
- Negative boundary：4K AR exact Geo+YaRN 100%，EVQ+YaRN 77.3%；
- Correct answer：PK 始终叫 teacher-forced NLL-gap retrieval；AR exact单列；
- Statistical boundary：n=3，不作无限定 significance claim；必须披露 18–98% seed range；
- Closure：把 raw payload、trial count 和 evaluator rule 打包；
- Unsafe wording：PK is retrieval accuracy；只报有利 8K 不报 4K。

### E-06 — 1B reversal 与 undertraining

**Canonical reviewer question**

> Does the 1B-token reversal show that EVQ only helps undertrained models?

- Likelihood / Impact：H–VH / Major
- Current truth：1B row 改变了 $L_{\mathrm{train}}$、data/schedule 且 single-seed，不是 token-only continuation；
- Correct conclusion：schedule-sensitivity negative boundary；既不能证明消失，也不能证明 durability；
- What remains：500M/8K MLA 3-seed primary anchor；
- LoRA relevance：heavy-pretrained checkpoint adaptation 可回应最简单 undertraining story，但 clean EVQ incremental effect 尚未闭合；
- Closure：只有 fixed-length、fixed-data/schedule replicated continuation 才能回答 saturation；
- Unsafe wording：9B tokens is overtraining；1B proves robustness；-2.5% composed row rescues the claim。

### E-07 — MLA convention 与 channel scarcity isolation

**Canonical reviewer question**

> Is the MLA effect caused by rotary-channel scarcity, or by other architectural/training differences?

- Likelihood / Impact：H / Major
- Current primary：432M/500M-token/3-seed，16K PPL 138.8→95.6 (-31.1%)，8K cost +1.1%；
- Supporting：within-MLA $d_{\mathrm{rope}}=32/16$ pilot only gives qualitative direction and has weak-baseline caveat；
- Missing：direct $\tau=d_{\mathrm{rope}}/\sqrt L$ and intermediate convention；
- Closure：one-seed screen 0.354/0.707/1.414, then replicate relevant contender；
- Decision gate：alternate wins则改写 operating convention，不撤回“allocation matters under tested convention”；
- Unsafe wording：production-identical DeepSeek；channel pilot proves a $1/K$ PPL law。

### E-08 — LoRA matched-control validity

**Canonical reviewer question**

> Is the fresh Geo+LoRA run actually matched to the historical EVQ-LoRA run?

- Likelihood / Impact：VH / Blocking
- Status：PROVENANCE_BLOCKED；
- Fresh Geo：native endpoints、official LongAlign、seed 42、hash-pinned、300 steps，结果内部一致；
- Historical EVQ：contemporaneous docs 指向 LongAlpaca-12k；exact historical raw hash/runtime 未恢复；
- Consequence：9.63 vs 6.48、21.5 vs 116.5、104.3 vs 1205 不能隔离 EVQ；
- Closure：fresh EVQ-42 on identical frozen LongAlign，改变唯一变量 frequency method；
- Decision gate：只有 clean pair 才可写 Geo→EVQ incremental effect；
- Unsafe wording：matched EVQ trade-off is 49%；historical EVQ used verified LongAlign。

### E-09 — LoRA 的 8K cost、32K PPL 与实际能力

**Canonical reviewer question**

> Is the apparent in-range cost acceptable, and does PPL near 104 mean the model can use 32K context?

- Likelihood / Impact：VH if LoRA used / Major
- Current truth：49% PPL ratio 是 cross-corpus，不能因果解释；PPL 是 exponential，优先报 NLL；
- Historical bucket：0–4K约 +3%，4–8K约 +48%，8–12K和12–16K明显少崩坏；
- Capability boundary：historical RULER/NIAH 未对应改善；PPL 104 只说明 token-level collapse 较少；
- Closure：matched per-position NLL buckets + Gold-answer NLL或 pre-registered retrieval/AR endpoint；
- Decision gate：若只有 PPL gain 无 task gain，LoRA 保持 stability/supporting row；
- Unsafe wording：cost is modest；EVQ-LoRA solves 32K。

### E-10 — QuALITY 是否是正 downstream evidence

**Canonical reviewer question**

> Is QuALITY a real downstream gain when accuracy is near random?

- Likelihood / Impact：H / Major
- Current truth：accuracy deltas +0.7/+2.2/+0.1/-0.4pp 无稳定方向；
- Retained evidence：n=2086 Gold-answer NLL，8K raw 3.202→2.239 (-30.1%) 等 probability-space diagnostic；
- Correct claim：supporting/non-regression and probability-space evidence，不是 accuracy win；
- Negative boundary：capacity floor；
- Closure：不需要新 benchmark fishing；
- Unsafe wording：QuALITY proves downstream improvement。

### E-11 — Statistics、paired deltas 与 pseudoreplication

**Canonical reviewer question**

> Are the reported effects statistically credible, and are evaluation chunks being treated as independent runs?

- Likelihood / Impact：H / Major
- Primary I/III：报告 paired seed directions、range、mean/std；n=3 不作 formal significance overclaim；
- LoRA：15 chunk deltas只证明 evaluator/corpus consistency，不是15次训练 replication；
- Primary II：不能从相邻协议借 variance；
- If only EVQ 43/44 are added：这只估计 EVQ side variability；若 Geo 仍仅 seed 42，不能称 3-seed paired treatment comparison；
- Closure：只有升级主张时才补 matched training seeds；
- Unsafe wording：five chunks are five runs；EVQ-only extra seeds make the pair fully multi-seed。

### E-12 — 99-run basin accounting

**Canonical reviewer question**

> What exactly are the “99 runs across 27 settings,” and do they validate the theory?

- Likelihood / Impact：M–H / Moderate
- Fixed wording：99 sanitized run rows = 45 pilot + 54 confirmation；9 configuration families；27 seed-level validation settings；
- Result：exact best 3/9，top-2 6/9，top-3 8/9，all optima within 1.5×；
- Boundary：manifest 不是 checkpoint/log archive；rank result只支持 empirical basin；
- Unsafe wording：99 runs prove the KL derivation or unique global optimum；
- Closure：flat row-level manifest + field definitions。

### E-13 — Alternative scalers 与 NTK negative result

**Canonical reviewer question**

> Is EVQ generally complementary to inference-time scaling?

- Likelihood / Impact：H / Major
- Current truth：positive claim只对 matched-scale YaRN；
- Negative boundary：$L=256$, 32× 时 EVQ4+NTK 331.4，Geo+NTK 198.1；
- Safe answer：composition is scaler-specific；EVQ changes substrate, not universally improves every rewarp；
- Closure：不需 broad scaler zoo；真实 reviewer 点名某 baseline 时再精确回答；
- Unsafe wording：EVQ replaces or always composes with YaRN/NTK/LongRoPE。

### E-14 — Training budget 与“工业模型学不到”

**Canonical reviewer question**

> Are the effects merely artifacts of small models or insufficient training relative to heavily pretrained industrial models?

- Likelihood / Impact：H / Major
- Current evidence：Primary I/III 的 token/seed provenance、progression、750M supporting row，以及 fresh Geo on pretrained 8B；
- What fresh Geo proves：普通 LoRA 本身就会改变 extrapolation PPL，因此 EVQ 必须测 incremental effect；
- What it does not prove：fresh Geo 不否定 EVQ；历史 EVQ不构成 clean industrial control；
- Safe answer：反对最简单 undertraining-only story，但不声称 trillion-token from-scratch durability；
- Unsafe wording：Chinchilla budget is an overtraining threshold；industrial validation complete。

---

## 5. Trust、provenance 与复现问题详账

### P-01 — 多处 source-of-truth 错误后的整体信任

**Canonical reviewer question**

> With stale figures and a mislabeled LoRA corpus, why should we trust the remaining result pipeline?

- Likelihood / Impact：VH / Blocking
- 必须主动列出：
  1. Figure 8 的旧 n=200 accuracy panel 被放在 NLL caption 下；
  2. Figure 9 把 progressive -81.2% 当成 454M -13.3%；
  3. historical LoRA row 的训练 corpus 很可能是 LongAlpaca，而 paper label 是 LongAlign；
- Correct posture：不说 reviewer misunderstanding，不说“数值没变所以不重要”；
- Trust repair：每项给 source of truth、affected claim、unaffected claim、修订动作、artifact hash；
- Current limit：LoRA historical provenance 未闭环，必须暂停 matched attribution；
- Closure：建立一页 errata/provenance map；最终 response 先讲 trust repair。

### P-02 — Figure 8 / Table 21

**Canonical reviewer question**

> Which QuALITY result is authoritative: the n=200 accuracy plot or the n=2086 Gold-NLL table?

- Likelihood / Impact：VH / Major
- Source of truth：n=2086 full aggregate；
- Submitted issue：旧 panel 是 n=200 accuracy pilot，caption 却写 Gold-NLL，且含不属于 retained table 的 32K point；
- Erratum：8K-raw Geo accuracy 513/2086=24.59%，应为24.6%，不是26.6%；Gold-NLL不变；
- Interpretation：accuracy near random且无稳定方向；NLL仅 supporting；
- Revision tense：区分 submitted artifact 与 local corrected revision，不写成 reviewer 看到的版本本来正确。

### P-03 — Figure 9 / Table 20

**Canonical reviewer question**

> Is the 454M long-range change -81.2% or -13.3%?

- Likelihood / Impact：H / Major
- Correct provenance：-81.2% 属 single-seed progressive training；454M FineWeb-Edu 3-seed row 是 -13.3%；
- Interpretation：Table 20 是不同 dataset/training regime 的 heterogeneous supporting rows，不是 controlled scaling law；
- Closure：重画并做 source-to-render value gate；
- Unsafe wording：figure/table are equivalent；-81.2% is the 454M aggregate。

### P-04 — Reproducibility 与 historical provenance

**Canonical reviewer question**

> Can the primary results be reproduced if some historical checkpoints or raw logs are missing?

- Likelihood / Impact：H / Major
- 两个命题同时成立：
  - manuscript-level specification 曾不够完整；
  - repo/supplement 可以提供 canonical schedule、public-data prep、locked env、runners、expected aggregates 和 tests；
- Correct distinction：reproducibility from public inputs $\ne$ bitwise historical provenance；
- Historical caveat：不可说所有 checkpoints/raw logs 都保留；
- LoRA caveat：fresh hashes已存在，但 result-bearing JSON 需复制进 tracked anonymous bundle；
- Closure：Primary I–III exact command/config index + expected gates + source map。

### P-05 — Seed、aggregate、chunk 和协议不可互换

**Canonical reviewer question**

> Are aggregates, per-seed results and per-chunk evaluations being mixed across protocols?

- Likelihood / Impact：H / Major
- Rules：
  - aggregate不能反推出 per-seed；
  - chunks不能当 training seeds；
  - $L=256$/100M不能升级$L=128$/15M；
  - current LongAlign不能与 historical LongAlpaca做 matched delta；
  - supporting pilot不能升级 primary claim；
- Closure：每个新数字绑定 dataset identity、seed、protocol、artifact、evidence tier。

### P-06 — Submitted、working revision 与 future revision

**Canonical reviewer question**

> Were the errors already fixed in the submitted artifact, or only after review?

- Likelihood / Impact：M–H / Moderate
- Required tense：
  - submitted artifact: contained the stale/mislabeled item；
  - audit/current working revision: corrected source exists；
  - author response: we will correct / have prepared the correction for revision；
- 不得混用：the paper now shows，若 reviewer 实际看到的是旧 submission。

---

## 6. 应用、系统与 novelty 问题

### A-01 — 真实 long-context benefit

**Canonical reviewer question**

> What evidence supports practical long-context ability beyond positional diagnostics?

- Likelihood / Impact：H / Major
- Positive：Primary-I 8K AR exact 58% mean vs Geo+YaRN 0%，且每个 EVQ seed非零；
- Required caveat：18–98% seed spread；4K Geo更好；PK仍是 TF diagnostic；
- Negative：QuALITY accuracy near chance；historical LoRA RULER/NIAH未改善；
- Safe claim：generation-level separation in one matched-scale stress protocol；
- Unsafe wording：universal downstream improvement or usable 32K assistant。

### A-02 — Zero parameters 是否等于系统提速

**Canonical reviewer question**

> Does zero learned parameters mean lower FLOPs, latency, memory or KV-cache cost?

- Likelihood / Impact：H / Moderate
- Safe answer：one-time inverse-frequency initializer change；no added learned parameters or inference-time operator；
- Not measured：training/inference latency、FLOPs、memory、energy；
- Unchanged：attention complexity与KV-cache complexity；
- Unsafe wording：free speedup；compute saving；production efficiency proved。

### A-03 — Production/industrial scope

**Canonical reviewer question**

> Do MLA and LLaMA-3-8B LoRA establish production readiness?

- Likelihood / Impact：H / Major
- Safe answer：MLA 是 production-relevant scarce-channel stress test；LoRA 是 industrial-checkpoint adaptation anchor；
- Not established：production-identical DeepSeek、from-scratch trillion-token durability、deployment reliability；
- Current LoRA caveat：incremental EVQ effect尚未 matched；
- Unsafe wording：industrial validation complete；production-ready。

### A-04 — 何时应启用 EVQ

**Canonical reviewer question**

> What practical diagnostic tells a user when EVQ is likely to help?

- Likelihood / Impact：H / Moderate
- Candidate favorable regime：high base、dead/near-static low-frequency channels、scarce rotary channels、需要与 YaRN matched-scale composition；
- Stop/validate conditions：small base/all channels alive、NTK rewarp、longer schedule shift、alternative $\tau$ in MLA；
- Safe guidance：diagnose spectral headroom and validate a local basin；
- Unsafe wording：enable EVQ by default for every RoPE model。

### A-05 — Novelty 相对 alternative allocation/per-head methods

**Canonical reviewer question**

> Is “a third axis” substantive novelty, or only an empty taxonomy cell that simpler/learned per-head schedules already cover?

- Likelihood / Impact：H / Major
- Safe novelty：explicit finite-budget framing；training-time closed-form allocation from a stated surrogate；exact inverse CDF；zero learned parameters；complementarity with inference scaling；
- Boundary：不声称 learned per-head methods、DAPE、FIRE、CARoPE、LongRoPE 被替代；
- Global vs per-head：shared initialization substrate与 learned specialization stage distinct，但 composition未测试；
- Missing：broad alternative-density trained controls；
- Unsafe wording：EVQ is the unique or universally best allocation。

### A-06 — Video/LoRA phenomenology

**Canonical reviewer question**

> Are the video 0.53 correction and LoRA rank threshold predictive laws or post-hoc fits?

- Likelihood / Impact：M / Moderate
- Correct status：directional/calibrated phenomenology；
- Video transfer evidence来自 head-to-head/base sweep，不依赖0.53的 universal derivation；
- LoRA $r\approx d_{\mathrm{head}}/2$ 是单模型 hypothesis；
- Action：真实 reviewer未问时不占主 rebuttal 篇幅。

---

## 7. AC 级问题

### AC-01 — 修正后贡献还剩什么

**Canonical reviewer question**

> After correcting the theory and provenance issues, is there still a coherent contribution worth accepting?

**Answer kernel**

1. exact contribution：conditional convex-surrogate cosh optimizer + closed-form inverse CDF；
2. mechanism contribution：finite spectral budget / training-time allocation as a distinct axis；
3. empirical contribution：3-seed matched-scale EVQ×YaRN、seed-scoped PE-dominant diagnostic、3-seed MLA scarce-channel stress test；
4. practical contribution：zero-parameter initializer compatible with existing attention computation；
5. honest boundary：not a unified trained-attention optimum, tuned-baseline leaderboard, universal downstream SOTA or production recipe。

### AC-02 — 为什么不因 reporting errors 直接 reject

**Canonical reviewer question**

> Are the reporting inconsistencies signs that the empirical conclusions are unreliable?

**Answer kernel**

- 不淡化错误；按 source-of-truth audit 明确哪些数字属于错误图、哪些 table/raw aggregate 保留；
- Figure 8/9 是 visualization provenance errors；LoRA 是 dataset-provenance/causal attribution blocker；
- Primary I 与 III 的 curated aggregates和 seed scope不由这些错误改变；
- 把受影响的 LoRA/tables claim 暂停或降级，而不是用未受影响结果掩盖。

### AC-03 — 最邻近的未完成控制是什么

按 decision value 排序：

1. fresh matched EVQ-42 on current frozen LongAlign；
2. exact Primary-II Geo/DAPE/EVQ matched seeds；
3. Geo/EVQ separately tuned YaRN scale；
4. MLA alternative-$\tau$ sanity screen；
5. tuned Geo base grid。

没有完成的控制必须以 limitation/claim scope 回答，不能把计划写成结果。

### AC-04 — 为什么 rebuttal 不应变成第二篇论文

- 真实 review 只映射到被触发的 3–5 个原子问题；
- 每个问题用 correction/concession/evidence/remaining-boundary 四段式；
- supporting experiments只用于回答点名质疑；
- 不承诺新的统一 theorem、大规模 benchmark或 method zoo；
- word budget 优先 trust、causal identification、nearest baseline、核心贡献。

---

## 8. 实验与证据更新看板

### 8.1 P0：LoRA clean pair

| Step | Action | Why | Completion gate | Stop rule |
| --- | --- | --- | --- | --- |
| 1 | 冻结 fresh Geo 的 LongAlign/token/eval manifests 与 hashes | 确保 comparator不再漂移 | anonymous bundle 可读、无私有路径 | hash不一致立即停 |
| 2 | fresh EVQ-42，唯一变化为 EVQ-Cosh $\tau=1.414$ | 建立 seed-42 causal pair | 训练、频率注入、final adapter、eval receipts完整 | 任何协议差异先修，不拿历史数补 |
| 3 | 同一 frozen WikiText token tensor做 aggregate + position-bucket NLL | 区分0–4K、4–8K和OOD段 | per-chunk/per-bucket NLL均存档 | 不只报PPL比例 |
| 4 | 加一个预注册 task-sensitive endpoint | 防止PPL=能力的误读 | evaluator、trial数、decision rule固定 | endpoint失败也必须报告 |
| 5 | 根据 seed-42 决定 seeds 43/44 | 节省GPU并先看因果方向 | decision gate below | seed-42无增量价值则停止 |

Decision gate：

- **EVQ 在 matched 16K/32K 明显优于 Geo，且8K代价可量化**：可运行 EVQ 43/44 估计 EVQ-side variability；若要称完整 multi-seed matched comparison，仍需 paired Geo seeds；
- **EVQ≈Geo**：结论是 LoRA adaptation解释大部分变化；LoRA row不能支持 EVQ incremental claim；
- **EVQ worse**：LoRA family降为 negative/supporting boundary；
- **PPL改善但 task endpoint无改善**：只写 reduced token-level collapse，不写 usable context。

### 8.2 其他实验优先级

| Priority | Action | Compute type | Claim upgraded | If negative |
| --- | --- | --- | --- | --- |
| P0 | Primary-II exact-protocol seeds 137/256 | training | matched baseline replication | re-tier Primary II |
| P1 | Geo/EVQ YaRN scale sweep | eval-only if checkpoints survive | tuned-scale robustness | retain matched-scale only |
| P1 | MLA $\tau$ screen 0.354/0.707/1.414 | training/eval | operating convention | revise convention |
| P1 | LoRA position buckets + task endpoint | eval-only | interpretability/capability | keep stability-only |
| P2 | tuned Geo base grid | training | nearest one-knob baseline | scope to high-base regime |
| P2 | trained $L_{\mathrm{eff}}^J$ / forcing diagnostics | checkpoint analysis | theory falsification | further soften mechanism |
| P3 | alternative density zoo / broad downstream suite | expensive | secondary novelty | do not lead in rebuttal |

### 8.3 不能用来“补齐”的替代

- EVQ-only extra seeds不能把 Geo-42/EVQ multi-seed写成 paired multi-seed comparison；
- $L=256$ Phase11B不能替代$L=128$ Table 4；
- five chunks不能替代training seeds；
- current LongAlign不能替代historical LongAlpaca；
- 99-run basin不能证明KL推导；
- PPL不能替代retrieval/AR；
- code存在、job启动或checkpoint目录存在都不等于 completed result。

### 8.4 Deferred / do-not-cite gates

- **Phase17B/C**：在缺少能够明确区分 `raw` 与 `yarn` mode 的 result JSON 时，不得把报告中的 34.6/52.0/81.2 等 raw-vs-raw 数值 relabel 为 YaRN、matched-scale 或 reviewer-grade evidence。
- **Phase22/23 MLA**：旧 pattern row 把 `+16.3%/-24.1%` 标为 $\tau=1.414$，而同一报告的 Phase22 table 把它们归到 $\tau=2.2$。原始 per-$\tau$ JSON 闭环前不得引用该行、拼接结论或据此改写 Primary III。
- **2B/4B**：存在历史 script、checkpoint、job 或 long-train trace，不等于 reviewer-grade completion。不得说“从未运行”，也不得把历史 trace 写成 completed 2B/4B evidence；只有完整结果 bundle、协议与 provenance gate 通过后才能升级。

---

## 9. 真实 reviews 到来后的映射顺序

截至 2026-07-12 真实 reviews 尚未收到，因此当前只执行 triage，不预写 reviewer 立场。真实评论到来后先分配稳定 concern ID、填 `correction / concession / evidence / boundary`，再创建唯一 `AUTHOR_RESPONSE_20260722.md`；不得恢复多路径 response package。

### 9.1 Reviewer 1 / Theory

1. 先承认 T-01 order error；
2. 保住 exact surrogate theorem 与 inverse CDF；
3. 定义 T-03 transport proxy；
4. 用 T-02/T-05 分开 local motivation 与 finite-$\tau$ deployment；
5. 主动限定 waterbed、MLA convention；
6. T-07–T-12 只在 reviewer追问时展开。

### 9.2 Reviewer 2 / Empirics

1. 先做 P-01 trust repair；
2. 给 Primary I/III 最强 traceable anchors；
3. 承认 tuned base、tuned YaRN、Primary-II seeds和DAPE fidelity；
4. 把TF/AR、1B、LoRA provenance讲清；
5. 用最小完成实验，不用计划替代结果。

### 9.3 Reviewer 3 / Applications

1. zero-parameter initializer与无新增inference operator；
2. 8K AR exact stress result；
3. MLA scarce-channel relevance；
4. 同时披露QuALITY/RULER、small-base和1B边界；
5. 不承诺production efficiency或universal task gain。

### 9.4 Area Chair

推荐顺序：

1. corrections/provenance；
2. one-sentence bounded claim；
3. Primary I + III；
4. seed/baseline limitations；
5. exact theory core + corrected scale identity；
6. remaining gaps和明确修订。

---

## 10. 长尾问题索引

这些问题需要保留，但只有真实 reviewer 明确触发时才展开。

| ID | Reviewer question | Likelihood | Impact | Current disposition |
| --- | --- | --- | --- | --- |
| LT-01 | Does a realistic heavy-tailed/empirical distance prior retain the cosh optimizer? | M | Major | uniform functional validation only；realistic prior open |
| LT-02 | Are stationary-phase constants and causal/signed normalization correct? | M | Major | revision-level rederivation |
| LT-03 | Is the Bessel alternative positive and therefore equally valid? | M | Moderate | do not use negativity defense |
| LT-04 | Can forcing errors be small in mass but large in quantiles? | M | Major | yes；amplitude/quantile bound unmeasured |
| LT-05 | Is $\lambda=1$ a physical law? | M | Moderate | no；normalization/calibration convention |
| LT-06 | Does $L_{\mathrm{eff}}^J$ behave like entropy support? | M | Moderate | no；directional curvature diagnostic |
| LT-07 | Can scarce-channel transport bounds rank task PPL? | M | Moderate | no；sensitivity only |
| LT-08 | Why not learn a separate schedule per head? | M | Moderate | shared initializer scope；composition open |
| LT-09 | Does learnable $\tau$ truly converge toward zero? | M | Moderate | no trajectory evidence |
| LT-10 | Are midpoint and endpoint quantization range-matched? | M | Moderate | disclose convention；do not conflate shape/endpoints |
| LT-11 | Is Table 20 a scaling law? | H | Moderate | no；heterogeneous supporting evidence |
| LT-12 | Are the 99 rows complete historical logs? | M | Moderate | no；sanitized run manifest only |
| LT-13 | Does the base-10K pilot settle tuned base? | H | Major | no；single-seed supporting pilot |
| LT-14 | Does fixed EVQ multi-seed settle DAPE variance? | H | Major | no；matched methods/seeds missing |
| LT-15 | Does LoRA rank $r\approx d_{\mathrm{head}}/2$ generalize? | M | Moderate | single-model hypothesis |
| LT-16 | Is video factor 0.53 predicted? | L–M | Moderate | post-hoc directional decomposition |
| LT-17 | Does no new inference operator imply no deployment cost? | H | Moderate | not measured |
| LT-18 | Can EVQ replace LongRoPE/LongRoPE2/FIRE/DAPE? | H | Major | no；complementary axis only |
| LT-19 | Does AR exact at 8K prove general retrieval? | H | Major | one matched stress protocol only |
| LT-20 | Are negative results cherry-picked or hidden? | H | Major | list NTK, small-base, 1B, QuALITY, RULER |

---

## 11. 更新协议

每个新结果必须同时更新以下字段：

1. ID；
2. canonical reviewer question；
3. trigger/source；
4. likelihood 与 impact；
5. claim at risk；
6. status；
7. current answer kernel；
8. positive evidence；
9. counterevidence / negative boundary；
10. evidence tier + artifact/hash；
11. dataset、seed、protocol scope；
12. unsafe wording；
13. closure action；
14. decision gate；
15. response inclusion priority；
16. owner；
17. last verified date/commit；
18. supersedes/change log。

硬规则：

- 一个 ID 只回答一个原子质疑；compound concern 必须拆开；
- 脚本、job、日志片段和checkpoint目录都不算完成结果；
- 每个数字绑定 artifact、dataset identity、seed、protocol 和 evidence tier；
- 跨数据/长度/schedule只作定性边界，不作 causal delta；
- aggregate、per-seed、chunk不可互换；
- 每项保留最强 counterevidence；
- 状态只按预注册 decision gate升级；
- submitted、working revision、future revision分开写；
- 新实验要同步更新 answer、claim tier、unsafe wording和stop rule；
- 不覆盖旧判断，用 supersedes/date保留变化；
- 真实 reviews 到来后，只提升被实际触发条目的 inclusion priority。

---

## 12. Rebuttal send gate

### 12.1 数学

- [ ] 是否明确 ordinary KL first variation 为零？
- [ ] 是否撤回 ordinary-KL $O(\tau^2)$ gain？
- [ ] 是否保留并准确限定 exact surrogate theorem？
- [ ] 是否定义 probability-transport proxy及其假设？
- [ ] 是否把 practical finite-$\tau$写成 empirical basin？
- [ ] 是否把 waterbed限定为allocation divergence？
- [ ] 是否把MLA $d_{\mathrm{eff}}$写成convention？

### 12.2 实验

- [ ] 每个主结果是否同时写seed/protocol/evidence tier？
- [ ] Primary I是否写matched-scale而非tuned dominance？
- [ ] Primary II是否明确seed 42或给出真正matched seeds？
- [ ] 1B是否只写schedule sensitivity？
- [ ] PK是否明确teacher-forced，AR exact是否分开？
- [ ] chunks是否未被当作seeds？

### 12.3 Trust 与 provenance

- [ ] Figure 8/Table 21、Figure 9/Table 20是否都主动披露？
- [ ] 26.6→24.6 erratum与NLL unchanged是否同时写？
- [ ] LoRA LongAlpaca/LongAlign provenance是否没有被掩盖？
- [ ] fresh result JSON是否已进入匿名bundle后才被引用？
- [ ] submitted与working revision时态是否正确？

### 12.4 主张边界

- [ ] 是否没有写universal SOTA、production-ready或measured compute saving？
- [ ] 是否没有说EVQ替代YaRN/LongRoPE/DAPE/FIRE？
- [ ] 是否没有把PPL当task capability？
- [ ] 是否保留negative boundaries？
- [ ] 每一段是否直接回答真实reviewer，而非扩写第二篇论文？

任何关键项失败，该段不得进入最终 response。

---

## 13. 当前推荐的一句话战略

> Lead with trust repair and the real KL correction; preserve the exact surrogate theorem and the three scoped empirical anchors; treat tuned baselines, Primary-II replication, MLA scale, and LoRA attribution as explicit decision gates rather than completed claims.

这会牺牲“统一第一性原理 + 工业级 LoRA 已闭环”的宣传力度，但能最大化 reviewer trust，保住真正可靠的数学核心和主要实证结论，并让后续每个新实验都有明确的 claim-upgrade 条件。
