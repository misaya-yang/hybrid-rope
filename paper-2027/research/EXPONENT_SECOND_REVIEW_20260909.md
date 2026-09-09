# 指数分布稿独立二审

日期：2026-09-09  
审查对象：`paper-2027/main.pdf`（正文 9 页，总 36 页）及当前 active TeX、Top 15 裁决、claim--evidence map、source index、真实 NeurIPS 2026 评审与实际 rebuttal。  
审查边界：只做二审；未修改 TeX，未运行模型，未下载资产。

## 总判断

现稿最强、也最能独立成立的论文，不是“EVQ-Cosh 是由 RoPE 几何必然推出的最优方法”，而是：**RoPE 指数分布包含可与 sampled range 分开的 interior-allocation 变量；这个变量确实改变训练后的长程行为，但其排序会随 range policy 反转，并与已学习的 Q/K 权重共适应。完整 sine--cosine subspace geometry 给出为何分配值得研究的静态机制；Cosh 是一个对所选凸 surrogate 有精确闭式解、且在若干训练/适配协议中有效的 analytic construction。对成熟 checkpoint，合理对象是相对 native table 的 displacement，最佳 profile 随 checkpoint、长度和安装方式变化。**

新批评因此命中了决定性的推论和结构问题，但它若进一步声称 Cosh 没有论文价值、所有结果只是坐标改写，或者只有 fixed-support 单一实验可成立，就过度了。Cosh 的条件性定理、可审计闭式构造和受控行为结果是真贡献；它缺少的是从 slow-subspace collapse 到 LM-optimal allocation 的必然性。论文不需要证明一个更强的 A* 才能成立，必须停止让结构暗示已经证明了 A*。

当前局部措辞比旧稿谨慎，附录尤其诚实；但全篇阅读路径仍是“slow collapse → 自选 surrogate → Cosh theorem → 一串 Cosh 大数 → frozen-model 新 profiles”。因此，两份方向相反的外部评语都把它读成 EVQ-Cosh 方法论文，不是偶然误读。文章必须通过主张顺序和视觉权重改变读者的默认解释，不能继续只在附录加限定语。

我的当前评分是：

- 以旧 NeurIPS 评审标签映射：**3/6，Borderline Reject，接近 4/6**。
- 以常见 ICLR 十分制表达：**5/10，marginally below the acceptance threshold**。
- 审稿信心：**4/5**。理论与实验推论层级在能力范围内；没有重跑模型，原始训练可复现性只按现有 receipt/provenance 审查。
- 接收风险：**高方差、偏 weak reject**。支持者会看到清晰的新研究轴和很强的控制；反对者会把 M4/non-Cosh、retarget reversal、co-adaptation 和跨模型反转读成对 EVQ 方法主线的自我削弱。完成下述结构修订后，现有证据足以形成 **4/6 或 6/10 的 weak-accept 论文**；“Top 10--15% / Strong Accept”没有依据。

这不是数据正确性层面的拒稿判断。已核对的 fixed-range、M4、crossing、Gemma、454M、BM 和 8B 关键数字没有发现抄写硬错。最严重问题是：文章用一个更一般、条件性更强的证据集合，讲了一个更窄且更像唯一方法优越性的故事。

## 对作者最新批评逐项裁定

| 批评 | 裁定 | 依据与边界 |
|---|---|---|
| “真实中心是 fixed-support interior allocation 是因果变量，以及 weights co-adapt。” | **支持，但不应写成唯一中心。** | `sections/02_identification.tex:4-16` 的三种子配对设计固定 architecture、initialization、token order、optimizer、budget 和 endpoints，只改变 30 个 interior exponents；这是当前最干净的 allocation estimand。`appendix/a5_identification.tex:119-155` 的两种子 crossing 和 50M crossing 共同支持 weight--table interaction。它们不能推出任意模型、range 或 profile 的一般规律。 |
| “slow collapse → 自己选 convex surrogate → Cosh optimum，没有必然导出 LM 目标。” | **支持，是当前最大理论叙事风险。** | Proposition 1 只证明在指定 separation measure 下慢频率 subspaces 收敛；`sections/03_theory.tex:62-67` 也只说“suggests”。`sections/03_theory.tex:72-87` 明确是“we choose”一个 density penalty 和 `min(phi,psi)` interaction。Theorem 2 只在该 `C_app` 内唯一。附录 `a1_proofs.tex:505-554` 又明确 operating strength 不是 behavioural selector。因此数学本身可以正确，slow-collapse → surrogate → LM loss 的桥仍不存在。 |
| “M4 formula 只有 7/12；1.25× 是 10/12；matched Exp 也改善；p=.836。” | **支持。** | `tables/table_m4.tex:20-26` 完整显示 formula-Cosh vs Geo 为 7/12、p=.125；1.25× 为 10/12、p=.027；Exp 为 9/12；formula-Cosh vs Exp 为 p=.836。M4 支持“非均匀 interior allocation 的作用不限于 Cosh”，不支持 formula rule 的跨配置稳定最优。 |
| “p=.836 表明 Cosh 与 Exp 一样。” | **过度推断。** | p=.836 只表示该 12-configuration sign-flip 设计未分离二者；没有预设 equivalence margin、equivalence test 或足够 power，不能称 statistical equivalence。应写 “the experiment does not resolve a difference”，不能写 “the two schedules are equivalent”。 |
| “fixed-range 9/9 赢，但 target-retarget 9/9 反转。” | **完全支持，而且这是贡献边界，不是反证。** | `appendix/a5_identification.tex:82-89, 94-117` 报告两个方向完全相反。fixed-support 列识别 interior allocation；retarget 列证明 deployment ranking 对 range policy 条件化。它们不能单独识别完整 support×allocation interaction law，但合起来比只报正结果更有科学价值。 |
| “derived 和 coarse ramp indistinguishable。” | **支持，但现稿附录已经诚实承认。** | `appendix/a6_mature_scale.tex:175-180` 明写 derived-minus-coarse intervals 在两模型都含零，profile-specific ordering unresolved。正文 `sections/04_mature.tex:53-62` 只声称两种 structured profiles 都大幅优于 uniform，这一窄主张成立。若将 residual-guided profile 写成被机制独特验证的方法，则不成立。 |
| “BM 在 OLMo 赢，在 Qwen 输。” | **支持。** | `sections/04_mature.tex:131-156` 和 Table 4 透明报告：OLMo 长端 2.78→51.32；Qwen-3B 128K 78.13→70.83；Qwen-7B 128K 84.44→71.11。正确推论是 checkpoint/length dependence。不能将失败归因于 Qwen 的 base=1e6，因为 model family、checkpoint、native window、length ratio、task rows 等同时变化。 |
| “因此证据不唯一支持 Cosh 方法论文。” | **支持。** | exact-range 证明一个变量；M4 明示非-Cosh 同方向；crossing 和 BM 明示 learned-state dependence。将它们全部用来替一个单一 Cosh method 背书，会丢失更强的一般发现。 |
| “应改成 allocation/range/co-adaptation 解耦，Cosh 只作 analytic construction。” | **支持，但‘解耦’必须写成 estimand separation，而非 independence。** | `(a,R,z)` 将 support 与 normalized allocation 参数化分开；fixed-support 识别 z；retarget 和 crossing 又显示其结果不是独立于 support/weights。推荐措辞是 “separate coordinates with interacting outcomes” 或 “controlled separation”，不是三个互不相关的因素。 |
| “不再堆 750M/1.5B/8B/video 来救方法。” | **大体支持，作为编辑决策而非否认数据。** | 这些结果回答旧评审的 scale/breadth 问题，但不能修复 surrogate-to-LM inference。应保留少数最有独立信息的 case study，其余进 appendix。8B source intervention 很有价值；1.485B 有 trainer confound；video 是单 seed scope check；750M 是单 matched pair。 |

## 最严重的实际文本与结构问题

### 1. 摘要把“研究对象”写回了“英雄方法”

`sections/00_abstract.tex:3-11` 的阅读链是：phase-invariant analysis reveals slow convergence；“This motivates” convex surrogate；closed-form solution EVQ-Cosh；随后 fixed-range 和 8B 巨幅 PPL。虽然每句单独都可防守，连续排列让读者自然推断 geometry 是 method 的理论根据，8B 是该推导的验证。

`sections/00_abstract.tex:12-16` 又只挑 OLMo BM 的正结果，不说同一 construction 在 Qwen long setting 反向。这样第二阶段同样像“又一个有效新方法”，而不是 checkpoint-relative response study。

摘要应该先报三个受控发现：fixed support 结果、retarget reversal、weight--table crossing；再说 geometry 提供 representation account，Cosh 是一个 analytic construction；最后用 OLMo/Qwen 相反结果概括 frozen stage 的条件性。8B 数字可留作 construction consequence，但不应占据摘要中唯一的 scale 位置。

### 2. Introduction 的贡献段仍把 Cosh 当作实证主轴

`sections/01_intro.tex:40-48` 再次按 slow limit → surrogate → EVQ → MLA/8B 的顺序组织。`sections/01_intro.tex:61-66` 的第 (ii) 项写 “derive a closed-form allocation and establish its effects”，而第 (iii) 又是 “derive a boundary-matched adjustment”。读者得到的是两个方法贡献：EVQ 和 BM；allocation/range/co-adaptation 的一般结论反而没有成为贡献条目。

建议贡献顺序改成：

1. 定义 support `(a,R)` 与 normalized allocation `z`，用 fixed-support 训练和 retarget 对照识别它们的作用与条件边界；
2. 给出 phase-invariant basis geometry，并提供 Cosh 作为一个对明确 surrogate 有闭式解、在模型中可检验的 construction；
3. 用 weights×table crossing 建立 co-adaptation，再用 native-relative displacement 组织 frozen-checkpoint 的正、负结果。

### 3. 理论段落的局部限定正确，但章节位置仍制造强推论

`sections/03_theory.tex:44-67` 的 slow-subspace proposition 是合理的静态结果；`a1_proofs.tex:106-126` 也明确固定 measure 不识别模型实际使用哪些 slow channels。问题是紧接着 `sections/03_theory.tex:69-99` 就进入所选 surrogate 和 “Cosh allocation” theorem，正文没有一句足够醒目的话说明：

> Proposition 1 does not determine an optimal allocation. Equation (6) is one tractable design surrogate, not a Transformer-loss objective or an approximation bound for it. Theorem 2 is exact conditional on that surrogate.

附录已经有这些边界，但主文读者不应到 `a1_proofs.tex:369` 和 `:505-554` 才知道。Theorem 标题也可从 “Cosh allocation” 改为 “Cosh optimum for the surrogate”，降低无条件最优的语义。

### 4. 最强识别证据出现得太晚，且主图只画正半边

正文先用两页定义和 theory，至 PDF p.4 才出现 fixed-range experiment。当前 Fig.1 只展示 fixed-support 9/9；retarget 9/9 反转只在 `sections/02_identification.tex:18-24` 的文字和附录图出现。这不是数据隐藏，但视觉层级仍选择了对 EVQ 最有利的半边。

应把现有 `fig_exact_range_control` 的 fixed/retarget 两面提升为首图，最好再保留 allocation schematic，形成三面板：same support；fixed-support outcomes；target-retarget reversal。首图结论应是 “allocation and support are distinct controls with interacting rankings”，而非 “Cosh wins”。

### 5. 8B 图注与论文自己的 allocation estimand 冲突

`sections/04_experiments.tex:64-67` 把 Native-LoRA/EVQ-LoRA 曲线称为 “The effect of exponent allocation grows beyond the adaptation window”。但 `appendix/a6_mature_scale.tex:259-262` 明确承认这个比较不隔离 pure interior shape，因为 Native endpoint grid 与 midpoint-Cosh grid 的 finite-grid embedding 也不同。§2 已把 allocation 定义为固定 `(a,R)` 后的 `z`；这里不能又把整张 frequency-table/substrate intervention 称为同一个纯 allocation effect。

最小修复是将图注改成 “The effect of the installed frequency table under matched adaptation ...”，并把 8B 放到 analytic-construction case study，而不是 fixed-support identification 证据层。

### 6. §4 已经成为资产陈列，而非一个可追踪的论证

`sections/04_experiments.tex:15-100` 在约两页中连续放入 432M MLA、750M continuation、8B NLL、8B source intervention、另一条 516-step RULER continuation、454M scaler composition、1.485B training 和 video pointer。它们的 statistical unit、intervention、support matching、metric 和训练阶段都不同。

这些结果并非虚假，但读者无法判断每一项是在验证：allocation axis、Cosh construction、scale persistence、source use，还是 range composition。尤其：

- 432M 是三种子 scarce-rotary case，适合作为 Cosh construction 最强的 architecture stress test；
- 8B 是一个 matched adapter pair，source-block intervention 有独立机制价值，但不是 training-seed replication，也不是 pure interior-shape control；
- 750M 是单 matched continuation pair；
- 454M 只证明一个指定 fixed-index scaler 下仍保留排序/组合收益，不证明与 scaling “正交”或普遍可叠加；
- 1.485B 在 `a6_mature_scale.tex:100-108` 明确有 AI2 distributed trainer 与 HF single-GPU loop 的 residual confound；正文 `04_experiments.tex:94-100` 没有提示；
- video 是单 seed、跨模态 scope check。

正文建议只保留 432M 与 8B/source-use 两类互补证据；750M、454M、1.485B、video 作为有清楚 tier label 的 supporting appendix。若必须保留 454M，措辞只能是“under this shared operator”，不能说 orthogonality/complementarity in general。

### 7. co-adaptation 被放在 §5 开头，实际应成为前半篇的第三个核心发现

`sections/04_mature.tex:7-25` 和新 `fig_weight_table_crossing` 是目前最有解释力的机制桥：同一 runtime table 对不同 trained weights 的损失方向反转；151.9M 又在两 training seeds 重现正 crossing interaction。它直接解释为什么 frozen adjustment 必须相对 native table，而不是任意移植“更高 rank”的表。

这张图真正补强中心，应在 fixed-support 与 M4 后立即出现。它同时防止把静态 effective rank 当作 LM predictor。50M panel 是单 seed，两模型；151.9M panel 才提供两 training-seed mechanism replication。两个 panel 指标不同（PPL vs tail NLL），应保持现在的分面与标签，不能合成尺度趋势。

### 8. frozen stage 同时介绍太多 profile，中心再次分裂

`sections/04_mature.tex:27-49` 的 displacement coordinate 很适合作为统一描述，但它本身只是 exact reparameterization，不足以构成主要 novelty。后面的科学内容应是“预训练权重如何对 profile 响应”：

- `:53-81` 的 fixed-support OLMo/Qwen 结果识别 structured reallocation recovery；derived 与 ramp 未分离；
- `:83-101` 的 normalized-index 是一个 practical static-table result；
- `:121-156` 的 BM/MrPro 跨模型反转识别 checkpoint/length dependence。

当前又把 Gemma placement、Qwen static table、BM construction、natural QA 和三模型 screen 都放进正文，像三套并列方法。应将 Gemma placement 降回 appendix：79.00 vs 72.81 是干净的两-placement confirmation，支持“finite-grid placement matters”；它没有 Native/YaRN 同场基线，不能支持方法整体优越或跨模型 generality。

BM natural QA 有较高保留价值，因为它回答旧评审“真实输入”的质疑；但必须与 Qwen 反向结果同段呈现。摘要若提 OLMo 正收益，至少应同时说 ranking does not transfer to Qwen at the longest setting。

### 9. Discussion 的 “explains” 超过现有机制链

`sections/05_discussion.tex:4-9` 写 subspace analysis “explains how that reshaping changes the distribution of positional directions”，字面上只解释 basis change，尚可；但下一句立即用 EVQ 跨模型 gains 收束，仍让读者将 basis rank 视为 performance explanation。`a1_proofs.tex:220-281` 自己给了 static proxy/order reversal counterexamples，更严格的结论应是 geometry characterizes what changes, not which table minimizes LM loss。

Discussion 首段应直接总结四个边界：allocation 有因果效应；range policy 可逆转排序；不同 nonuniform profiles 未被 M4 分离；weights 与 table 共适应。Cosh 的条件性构造和行为结果放第二段。这样 negative results 变成论文的科学发现，而不是方法失败注脚。

## 与真实 NeurIPS 2026 初审及 actual rebuttal 的对照

这里以 `git show main_0726:rebuttal/rebuttal_0723/00_REVIEWER_SCORES_AND_AC_METAREVIEW.md` 为 official review owner，不把旧 review 当真理，只看它是否准确预见当前风险。

### 真实 3 分 Reviewer 27bE

27bE 的 Summary 已把旧稿读成：surrogate stationary point → EVQ-Cosh → practical tau rule → 三类方法实验。这与当前读者再次形成的观感高度一致。其最关键意见不是泛泛要求更多数据，而是：`C_app`、pure-tether choice、small-tau operating argument 和 finite tau experiments 是不同层，要求 independently tuned tau 与 matched non-Cosh schedules。

当前稿的实质进展：

- M4 确实加入了同 operator、fixed endpoints 的 Cosh multipliers 和 deformation-matched exponential；
- `a1_proofs.tex:505-569` 明确 rule 是 fallible convention，不是 global optimum；
- 错标 DAPE 的 row 已在 `a5_identification.tex:210-229` 改成 layer-shared learnable inverse-frequency comparator，并撤回 DAPE-specific 解读；
- 1M base、head dimensions 与 larger-model evidence 已存在。

仍未解决的是叙事层：M4 的完整证据只在 appendix table，主文仅一句；而 432M/750M/8B 的正结果占更大版面。27bE 若只看主文，仍有理由认为 matched alternatives 是补充消融，Cosh 才是核心。这说明“回答了 review checklist”与“论文主张已重构”不是一回事。

### AC metareview

AC.1 的 FMRoPE novelty 与 matched comparison 已被显著补强：现稿引用 Oka，明确 base/range 与 interior allocation，151.9M fixed-support 三种子是比旧 rebuttal seed-42 数字更强的直接对照。这个问题不应再用“first method”去硬争；最佳新意是可识别的研究变量和控制。

AC.2 的 small/diagnostic-heavy 问题也实质改善：RULER、natural QA、1.485B/8B 和 source intervention 都比旧稿强。它们支持 scope 与 consequence，不验证 surrogate。

AC.3 的 surrogate-to-Cosh-to-operating-rule 问题只**部分关闭**。形式层已诚实，M4 也说明多个非均匀 schedule 有效；恰恰因为如此，论文更应把 general allocation result 放在 Cosh 之前。继续写成 EVQ method paper 会让新证据看似削弱方法；重写成 exponent-distribution paper 则让它成为支持。

### 其他官方评审

- Dz6s 的“诊断太多、真实任务不足”现在有较好回应，但其另一要求正是把 surrogate proof、kernel diagnostic、trained-model result 分层。当前附录分层比主文好。
- zWsa 的 FMRoPE/novelty 批评不能被当成最终事实；fixed-support 设计确实证明 exponent allocation 不是 base selection 的同义词。与此同时，dead-frequency intuition 本身不宜声称首次，当前 related work 的处理比旧稿正确。

### actual rebuttal 的可继承与不可继承部分

可继承的是：承认 FMRoPE citation omission；把 fixed-range 与 retarget 分开；把 scale evidence 分层；承认没有 universal Cosh/tau optimality。

不应继承的是 rebuttal 的竞技式证据堆叠。`paste/AC_PUBLIC.md` 以 “first closed-form” 开场，并把 scale、RULER、real tasks、bases、architectures 全部串成对 EVQ 的防御；`paste/final_send/FINAL_AC.md` 又按 AC 三个条件压缩大数。这在 rebuttal 字数和说服场景中合理，但不应成为新论文的章节逻辑。

此外，旧 AC response 中 “Cosh follows exactly from the surrogate” 只是在条件 objective 内成立；“surrogate coefficients are fitted from exact kernel” 不应在没有当前明确 derivation/receipt 的情况下恢复。当前稿没有依赖这一句，是正确选择。

## Top 15 与三项新增/恢复资产的二审

Top 15 文档的逐项数值和来源层级总体审慎：它没有把同 run 的端点拆成多个实验，没有把 Gemma 与旧 Native/YaRN 拼表，没有把 8B 不完整 32K macro 伪装成完整，也明确 1.485B trainer confound 和 Qwen BM 失败。问题不在盘点本身，而在“排名”容易被误用成正文纳入清单。主文九页目前实际触及 Top 15 中约十三项，已经超过读者能维持的因果层级。

更合适的分组不是 Top 15，而是 evidence role：

### A. 主文必须保留：决定论文能否成立

1. **#1 fixed-support + retarget 两面**：主识别证据；首图必须同时出现正向与反转。
2. **#7 M4 multi-shape factorial**：证明结果不是单一 Cosh curve；完整的 7/12、10/12、9/12 和 unresolved Cosh-vs-Exp 应进正文紧凑表。
3. **#9 weights×table crossing**：解释 co-adaptation，并为 frozen stage 提供机制桥；新图应提前。
4. **#3 mature fixed-support comparison**：第二阶段最干净的 pure interior-z 证据；必须同时说 derived≈ramp unresolved。
5. **#11 BM/MrPro cross-model outcomes**：不是 BM 胜利表，而是 checkpoint/length dependence 表；OLMo 与 Qwen 都保留。

### B. 主文可选一个或两个：construction consequence 与任务价值

6. **#5 432M MLA 三种子**：Cosh 在 scarce rotary channels 下最稳的 architecture stress test，优先保留。
7. **#2 8B matched adaptation/source use**：规模与 remote-source dependence 有独立价值，保留一个紧凑结果；清楚标为 full-table intervention、one trained pair。
8. **#6 OLMo natural QA**：如果主文必须有 natural task，优先于更多 synthetic RULER 表；与 Qwen BM reversal 同段。

### C. Appendix supporting：不应再占正文主叙事

- **#4 Qwen static index vs YaRN**：完整 13-task、CI 正确，但 gain `g` 不同，属于 practical method comparison，不是 pure-shape causal test。
- **#8 750M continuation**：有 AR exact，但单 matched pair，信息被 432M/8B 覆盖。
- **#10 454M scaler composition**：恢复是正确的 provenance 修复；它说明指定 `R_8` 下 Geo/EVQ 排序保持且差距扩大，不证明正交性或普遍 additivity。
- **#12 Gemma 79.00 vs 72.81**：只支持 finite-grid placement，四个固定 tasks 内的 paired interval；没有 Native/YaRN，不能成为 scale/generalization 标题结果。
- **#13 1.485B from-init**：126/128 文档方向强，但 trainer implementation confound 和单 trajectory 使它只适合 scale-supporting tier。
- **#14 OLMo selective Q/K adapters** 与 **#15 8B RULER continuation**：任务族适配与独立 adapter 有价值，但与更早 parent/constructive evidence 共享来源，正文继续加入会使 protocol genealogy 难以读懂。
- video-DiT：单 seed scope check，留 appendix 即可。

### 新 crossing 图

**真正补强中心。** 数字 owner 正确：151.9M 使用 `SAME_SUPPORT_FROZEN_CHECKPOINT_RESULTS_20260823.json:small_model_crossing`，不是 Qwen K32 receipt。50M 矩阵 `[[7.14,76.20],[23.05,7.16]]` 与 151.9M 两种子均值 `[[3.426,5.776],[4.455,3.479]]` 都呈 diagonal preference；两 training seeds 的 interaction 均为正。它是从 allocation identification 过渡到 mature adjustment 的最好证据。

边界：50M 只有 seed 42；151.9M 是两 training seeds；两 panel 指标、runtime-table construction 不同。可称 replication of crossing direction，不能称已得到通用 co-adaptation law。

### 新 Gemma 79.00 vs 72.81

**窄补强，非中心补强。** receipt 显示两臂同 checkpoint、inputs、gain `1.1025858`、K=128、`L_ref=4K`、s=4，比较 normalized-index 与 direct-gap placement；320 inputs、四个 tasks，CI 全正。它证明在这个 finite grid 上 placement choice 可测。

它不比较 Native/YaRN，不证明 K=128 本身导致差异，不是自然文本，不支持 normalized-index across-model universal superiority。正文六行的收益小于叙事成本，建议回 appendix。

### 恢复的 454M

**provenance 修复正确，中心信息有限。** 当前 `a2_experiment_details.tex:83-111` 已给 legacy fixed-index `R_s` 的精确公式；Table `table_evq_ramp` 使用正确 full-sequence PPL 157.7/107.5，不混入 per-document 262.0/237.2。原本地 raw mirror 缺失会降低独立复跑层级，不等于结果幻觉；curated JSON、历史报告和代码 owner 足以支持“已报告比较”的存在。

它能支持的最强话是：在这一个共享 scaler 下，EVQ-trained table 仍优于 Geo-trained table。它不能支持“scaler 与 allocation 完全正交”“任意 range method 可叠加”或“改善是 additive”。

## 可直接实施的新论证与章节顺序

### 推荐定位

一句话定位：

> **An empirical-mechanistic study of RoPE exponent distributions: separating frequency support from interior allocation, characterizing the positional basis, and showing that learned weights make post-training adjustments checkpoint-relative. Cosh is one analytic construction within this study.**

当前标题 `Beyond the Base: Exponent Allocation in RoPE` 可以保留。若要更明确，可用 `Beyond the Base: Exponent Allocation, Range, and Co-Adaptation in RoPE`；不建议标题重新突出 EVQ-Cosh。

### 九页主文顺序

1. **Introduction**：问题是 exponent distribution，不先推方法。开头提出三个问题：固定 support 时 allocation 是否有行为效应；range policy 是否改变排序；weights 学过 table 后能否直接移植。
2. **Exponent Tables as Controlled Objects**：保留 Eq. (1)--(2)，定义 `(a,R,z)`；同时定义 pretraining-time allocation 与 mature-table displacement 的不同 estimand。只用一两句预告 Cosh、Exp、ramp 是 intervention instances。
3. **Controlled Findings on Allocation, Range, and Learned Weights**：
   - 3.1 fixed-support 151.9M 与 target-retarget reversal，首图双面；
   - 3.2 M4 multi-shape factorial，正文表完整给 formula/1.25×/Exp；
   - 3.3 weight×table crossing，新图提前；
   - 小结只下三条结论：allocation matters；ranking is range-conditional；weights co-adapt。
4. **Positional-Basis Geometry and One Analytic Construction**：
   - 4.1 full sine--cosine subspace、canonical collision、effective-rank identity；
   - 4.2 slow-frequency limit及其明确边界；
   - 4.3 chosen convex surrogate 与 conditional Cosh theorem；
   - 4.4 只保留 432M 与 8B/source-use 作为 construction case studies。operating tau 的 assumption-bound derivation继续放 appendix。
5. **Adjusting Exponents after Pretraining**：从 crossing 自然引出 native-relative `d_k`；先报 mature fixed-support uniform/ramp/residual，明确 ramp≈derived；再报 BM 在 OLMo/Qwen 的相反排序和 natural QA。Qwen static-index、Gemma placement、zero-training routing放 appendix。
6. **Related Work**：围绕 range methods、learned tables、operator changes分组；避免“首次”竞赛。
7. **Discussion and Limits**：将 fixed-support、retarget、M4、crossing、frozen transfer的边界作为结论，不把负结果埋在方法之后。sparse attention 仍作为未来方向，一句话即可。

### 主图/主表重排

- **Fig. 1**：same support allocation schematic + fixed-support 9/9 + retarget 9/9 reversal。
- **Table 1**：M4 的四个关键 contrasts；视觉上不要只 bold 1.25× Cosh，可强调 “multiple nonuniform shapes; formula-vs-Exp unresolved”。
- **Fig. 2**：weights×table crossing。
- **Table 2**：两项 Cosh construction consequence（432M 三种子、8B one-pair full-table adaptation），每行明确 estimand/tier。
- **Table 3**：mature fixed-support uniform/ramp/residual。
- **Table 4 或 Fig. 3**：BM OLMo/Qwen 全方向；natural QA 可作为 OLMo 的 task-level supporting panel。
- 8B length curve、Qwen static index、Gemma placement、454M、1.485B、video 全部可在 appendix 保留完整，不丢资产。

## 可直接采用的关键段落

### Abstract 核心骨架

> RoPE frequency tables vary along two distinct coordinates: their sampled log-frequency support and the allocation of rotary pairs within that support. We study how these coordinates affect positional representation before and after the model weights have adapted to a table. With support fixed, changing only 30 interior exponents improves all nine seed-by-length extrapolation comparisons; retargeting the support reverses all nine, showing that allocation has a causal effect but its ranking is range-dependent. A 12-configuration factorial extends the allocation effect to multiple non-uniform shapes, while crossed table swaps show that learned weights prefer the table on which they were trained. We characterize the resulting sine--cosine subspaces and give Cosh as one closed-form allocation that is exact for a stated convex surrogate and useful in trained and adapted models. For frozen checkpoints, a common exponent-displacement coordinate exposes model- and length-dependent responses, including gains on OLMo and reversals on Qwen. These results establish exponent distribution as a controlled design variable while separating analytic constructions from claims of universal language-model optimality.

其中最后一句可再压缩；关键是先出现受控发现，再出现 Cosh。

### Theory bridge

> Proposition 1 identifies redundancy in the positional basis under a declared separation measure; it does not select an allocation for Transformer loss. We therefore introduce Eq. (6) as one tractable design surrogate that penalizes concentration and slow-end co-location. Theorem 2 is an exact uniqueness result conditional on this surrogate. Whether its Cosh solution is useful for a learned model is evaluated separately by the controlled experiments below.

### Controlled-findings 小结

> The three controls answer different questions. Fixed support identifies an interior-allocation effect. Support retargeting shows that this effect does not determine a deployment ranking independently of range. The weights-by-table crossing shows that a table cannot be judged independently of the coefficients learned around it. We therefore treat Cosh and the frozen displacement profiles as testable constructions within a support- and checkpoint-conditional design space.

### Contributions 段落

> We contribute (i) a support--allocation decomposition and paired experiments that identify an interior-allocation effect together with its range-dependent reversal; (ii) a phase-invariant account of the finite sine--cosine basis and a closed-form Cosh construction for one explicit convex surrogate, evaluated separately from the theorem; and (iii) direct evidence of weight--table co-adaptation, plus a native-relative displacement framework and matched frozen-checkpoint studies that expose when exponent adjustments transfer and when their ranking reverses.

### Discussion 首段

> The results support a conditional design principle rather than a universal profile. Interior allocation changes learned length generalization when frequency support is held fixed, but retargeting that support can reverse the ordering. More than one non-uniform shape follows the favorable direction in the factorial, and frozen table swaps reveal strong co-adaptation with the learned weights. The subspace analysis characterizes what a table changes in the positional basis; it does not by itself predict which allocation minimizes language-model loss.

## 逐维度审稿判断

| 维度 | 判断 | 理由 |
|---|---|---|
| Originality | **MEETS / 边界偏强** | fixed-support interior-allocation estimand、range reversal、phase-invariant full-pair geometry和co-adaptation组合有清楚新意；单独的“改 frequency table”或 slow channels 不是首次。 |
| Methodological rigor | **PARTLY MEETS** | 核心三种子 fixed-support 与 M4 控制强；大量 scale/frozen 结果是单训练 pair、固定 task panels或不同 trainer，不能共享同一因果强度。 |
| Evidence sufficiency | **MEETS for ‘allocation matters’; PARTLY MEETS for any profile-general claim** | 主轴证据充分；Cosh uniqueness、BM transfer、normalized-index transfer都只能在各自条件内成立。 |
| Argument coherence | **DOES NOT MEET 当前接收线** | 章节与视觉顺序仍把一般发现包装成两个方法故事；这是最主要的 decision-bearing weakness。 |
| Writing quality | **PARTLY MEETS** | 句级准确、附录限定很好；整体信息密度和实验 genealogy 过载，让读者难以区分 estimand。 |
| Literature integration | **MEETS** | 已补 FMRoPE、learned-frequency 和 operator-changing work；不要恢复未经证实的“首次”或对 AdaRoPE/MLA compatibility 的推断。 |
| Significance | **MEETS if reframed; PARTLY MEETS 当前叙事** | exponent distribution 作为受控设计轴有广泛价值；把价值压成 EVQ/BM 两个方法会让 mixed outcomes 看似降低意义。 |

## 必改项与可选项

### 接收前必须改

1. 用 fixed-support / retarget / M4 / crossing 重建前半篇主线，Cosh 后置为 construction。
2. 首图同时呈现 retarget reversal；不能继续只把 9/9 正结果可视化。
3. 在主文写明 surrogate 不是 LM objective，Cosh 只在该 surrogate 内唯一。
4. 修正 8B 图注的 pure-allocation 归因冲突。
5. contributions 与 Discussion 改写为一般发现，不再分别包装 EVQ 和 BM 方法。
6. 主文削减资产数量，给每个保留结果明确 evidence role 和 statistical unit。
7. BM 正负迁移、derived≈ramp unresolved 必须在主文保持同等可见。

### 可选但有收益

- 将 Gemma placement、1.485B from-init、454M composition全部移至 appendix；
- 将 M4 full contrast table 提升到主文，减少一个纯正向 capability figure；
- 在 evidence map 中用 “identification / construction consequence / checkpoint response / supporting scope” 替代 ranking；
- 对 bootstrap CI 加一句：它们条件于固定 checkpoint/trained pair 和固定 task set，量化 row uncertainty，不是训练 seed 或 task-population uncertainty。

## 最终建议

**Major structural revision, no new model experiment required.** 现有数据已经足以写成一篇有辨识度的 exponent-distribution 论文；继续增加规模、模型或 modality 不会解决当前接收风险。需要做的是牺牲“Cosh 获得唯一理论背书”的表面完整性，换取更真实也更强的科学结论：allocation 可识别，range 可反转，weights 会共适应，analytic profiles 的价值必须在这些条件下判断。

如果按最小方案执行，论文不会降格成坐标重命名或反例目录。相反，Cosh 的理论和实际效果会获得更准确的位置：它是一般研究对象下一个漂亮、可复现、实证有效但非唯一的 construction；第二阶段也不再是多个 post-hoc 方法拼盘，而是对 learned checkpoint 如何响应 exponent displacement 的系统研究。
