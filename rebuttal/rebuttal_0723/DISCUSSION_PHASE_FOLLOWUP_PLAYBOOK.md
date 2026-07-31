# 讨论期追问应答手册 (NeurIPS 2026 #11628)

Date: 2026-07-31 · 讨论期至 **2026-08-03**
状态:已发五份不动。本文件只用于**被追问时**的回复。

---

## 0. 使用规则(每次回复前重读)

1. **问什么答什么。** 不补充没被问到的边界、不主动削弱自己。
2. **不主动披露。** 只有被直接问到才答;被直接问到则如实答,不闪避。
3. **先给结论,再给数字,最后给边界。** 追问回复应比 rebuttal 更短 —— 目标 800–2,000 字符,不是 10,000。
4. **所有修订承诺用将来时。** 讨论期不能传 PDF。
5. **写"X 更差 / 没提升"之前,先核对两臂身份。**
6. **不编造。** FMRoPE 无官方实现;我们实现的是已发表的 §6.1 规则。
7. **语气**:感谢 → 直接回答 → 停。不要再写"What your review changed"这类收尾,那是 rebuttal 的体裁,追问阶段显得冗长。
8. **区分两个 in-window 代价**,禁止混用:
   - 从头训 4K NLL **+0.0381**(真·分配代价,小)
   - 成熟模型换表 RULER 4K **82.16→37.51**(适配失配)

---

## 1. Reviewer Dz6s(4 分 / 信心 3)—— 友好票,目标是守住并争取 +1

他的三条关切都已答。追问多半是澄清型,不是攻击型。

### D-1 「RULER 4K 从 82.16 掉到 37.51,这个方法在窗口内还能用吗?」

**最可能被问的一条。**

> The two numbers measure different things and we should separate them. The 4K drop is not a property of the frequency table by itself: it is what happens when a table is swapped into a checkpoint that was pretrained with a different one. In the from-scratch comparison, where no swap occurs, the same allocation costs **+0.0381 NLL at 4K** on OLMo-2 1.485B while gaining −0.0437 at 8K and −0.1351 at 16K. The large RULER figure comes from the retrofit path, and narrowing the adaptation to the tensors RoPE acts on already recovers part of it (37.51 → **42.44%**). On the generation and multi-hop QA endpoints under that restriction the two arms are level in-window (2Wiki exact **22.0% vs 21.5%** at 4K). The revision will report these as two separate quantities.

**陷阱**:不要说"in-window 代价很小"而不区分两者 —— 82.16→37.51 就在同一张表里,会被立刻抓住。

### D-2 「第 1 行 8K 是 98/100,第 2 行 8K RULER 只有 21.29%。为什么差这么多?」

> They are different tasks and different scorers. Row 1 is a single numeric-retrieval family with a strict endpoint — literal equality of the complete answer string plus a terminal EOS token, read from raw token IDs. Row 2 is a macro over 13 heterogeneous families including variable tracking, common/frequent word extraction and multi-hop QA, several of which the model does not solve at any length. A high score on one does not predict the other, which is why we report both rather than the more favourable one.

### D-3 「OLMo-2 从头训只跑了 2.097B token,是不是训练严重不足?」

> It is an early-training comparison and we label it that way. It is one trajectory at step 1,000 from the public step-0 initialization, matched on recipe, data-order prefix and evaluation rows with the released geometric checkpoint. What it supports is the direction and its monotonicity in the extrapolation ratio (**+0.0724 / +0.0381 / −0.0437 / −0.1351** at 2K/4K/8K/16K, paired document-level bootstrap intervals excluding zero at every length). It does not support a converged-model claim, and we do not make one.

### D-4 「这些新结果有几个种子?」

> They differ by result and we state each. Three seeds: the fixed-schedule ladder, the exact-range factorial, the held-out base/head suite, and the 432M MLA study. One trajectory: the 1.485B from-scratch run. One continuation seed per arm: the 1.485B and 8B RULER and QA results, with a second independently trained EVQ seed available on the strict retrieval endpoint (**69/100 and 67/100** at 8K against Native 0/100). The mechanistic attribution claims rest on the three-seed studies; the mature-model results are matched single-seed comparisons and the revision will label them so.

### D-5 「video-DiT 和 MLA 是不是在凑广度?」

> They answer a specific question rather than add breadth: whether the effect depends on multi-head attention with a full rotary budget. The MLA study has **16 rotary channels** and three seeds (PPL@16K **138.8±5.5 → 95.6±4.1**); the video-DiT runs test a non-language modality. We would not offer either as scale evidence, and the scale claim rests on the 1.485B and 8B results.

---

## 2. Reviewer 27bE(3 分 / 信心 4)—— **摇摆票,最值得投入**

他最锋利,而且他的追问最可能是真诚的技术追问。

### 27-1 ⚠️ 「你们撤掉的那个 32 参数可学习基线,在论文里原本标的是什么?」

**这是全场最危险的一问。** 若被**直接**问到标签,必须如实答,但只答被问的部分。

> It was labelled as a DAPE-style comparison in the submitted table. That label was inaccurate: the arm we ran is a 32-parameter learnable inverse-frequency baseline, layer-shared, initialized from the geometric grid, with the standard attention operator — not the DAPE method. The correction will appear in the revision. This is a further reason the row cannot carry the allocation-shape claim, and it is why we moved that claim to the fixed zero-parameter schedule ladder.

**规则**:
- 只有被问到**标签/名称**才用上文。
- 若他只问"那一行是什么方法",答内容(32 个可学习 inv_freq,layer-shared,从几何初始化),**不主动说标签错了**。
- **绝不**重新搬出 455.3 / 477.7 的调参辩护 —— 那会把讨论重新锚定在这一行上。
- 不要写"we apologize"式的长段自责。一句陈述 + 修订承诺,停。

### 27-2 「匹配的指数 schedule 与 Cosh 统计上无法区分(p=0.836)。那理论买到了什么?」

> It buys a closed-form operating point rather than a search. The claim the ladder supports is that the allocation axis is identifiable — pre-specified 1.25× Cosh and the matched exponential improve weighted OOD NLL over Geo by **0.0121 and 0.0106**, favouring non-geometric allocation in 10/12 and 9/12 configurations — not that Cosh is the unique optimum. We state in the response and will state in §6 that Cosh's role is the closed-form inverse CDF of the stated surrogate, not a universal optimum. What would be lost without the derivation is not accuracy at the optimum but the absence of a search: the exponential arm was constructed to match Cosh's RMS deformation, which required knowing Cosh first.

### 27-3 「τ 落在 0.75×–1.5× —— 那是 2 倍区间。是不是说明任何 τ 都行?」

> No, but it is a weaker statement than a point optimum and we present it as such. The window is bounded on both sides across 21 configurations in two independent studies, and midpoint-Geo (which is τ → 0) is outside it: against midpoint-Geo the formula wins 7/9 configuration means and 18/27 paired runs. Against the pilot-selected neighbour inside the window the formula wins only 3/9, which is exactly why we call the rule a basin default rather than an optimum. The scaling form τ ∝ d_eff/√L_train is what the derivation supplies; the constant is a zero-search convention.

### 27-4 「C_app 的 α 和 β 是拟合的。拟合在什么上面?会不会只是过拟合到你们的配置?」

> They are fitted from the exact oscillatory kernel K on the deployed discrete channel grid — that is, from the quantity the surrogate approximates, not from any trained-model outcome. No NLL, PPL or task score enters the fit. The validation is functional rather than a goodness-of-fit statistic: allocations derived from the surrogate reduce the **exact-kernel** collision score by **24–92% across 12 configurations**, and a collision-only oracle search performed directly on the exact kernel reaches comparable reductions. The surrogate is therefore doing the work of the search rather than fitting to the evaluation.

### 27-5 「50.9M / 151.9M 的机制控制,凭什么说能迁移?」

> We do not infer transfer from them; we test it separately. The mechanistic ladder isolates the variable under controls that are only affordable at that size (identical parameter count, tuning budget, reference grid, deformation magnitude and spectral range across arms). Whether the effect survives is a separate question answered by the held-out base-1M/d_head=128 suite (untuned, three seeds, **−0.802/−0.433/−0.212** at 1K/4K/16K), the 1.485B from-scratch run, and the 8B retrofit. The size of the intervention does not grow with the model: the frequency table has d_head/2 entries, so the 50.9M controls, OLMo-2 and LLaMA-3-8B all act on the same 64 rotary channel pairs.

### 27-6 「你们说 Q/K-only 定位更好,但 16K 反而从 6.13 掉到 5.03。」

**必须承认,不要辩。**

> Correct, and we should not have implied otherwise if we did. Restricting the continuation to Q/K improves 4K (37.51 → **42.44%**) and 8K (21.29 → **31.63%**) and leaves 16K essentially unchanged or slightly lower (6.13 → **5.03%**). The supported statement is that adaptation scope affects the in-window/8K trade-off materially without accounting for all of it, and it does not improve the 4× endpoint. The revision will state it in those terms.

### 27-7 「四个 link 分开之后,定理还剩下什么?」

> Link (i) stands as an exact statement: Cosh is the stationary density of C_app. What the separation removes is the implication that this determines the trained-model optimum. The chain is (i) exact given the surrogate; (ii) the quadratic surrogate, the discrete grid and the pure-tether branch are modelling choices; (iii) the analysis yields the scaling form; (iv) the constant is a convention. The empirical claim the paper rests on after this separation is narrower and is the one the fixed-schedule ladder tests directly.

---

## 3. Reviewer zWsa(2 分 / 信心 5)—— 大概率不回复;若回复,只面向 AC 写

**关键判断:信心 5 且已给 2 分的审稿人极少改分。所有回复的真实读者是 AC。**
**因此:回复必须简短、事实性、可核验,不要出现任何可被读成情绪的措辞。**

### z-1 「我仍然认为与 FMRoPE 重叠。」

> We take the disagreement as narrow and checkable. FMRoPE's own abstract describes its effect as shifting the band toward lower frequencies, and its §6.1 defines the rule as setting the RoPE base equal to the training context length. Under that rule the exponents remain uniform in i and the normalized geometric order is preserved. Our experiment fixes exactly what a base choice sets — both sampled extrema and the log span — and varies only the 30 interior frequencies; OOD NLL improves by **0.478/0.205/0.113** at 512/1K/2K, winning **32/32, 27/32 and 22/32** anchors. If the mechanisms were the same, that control would show no effect.

### z-2 ⚠️ 「没有官方实现,你怎么保证 FMRoPE 实现正确?」

> There is no official implementation available to us, and we implemented the rule as published in §6.1: base set to L_train at training and L_target at inference. Two things support that the implementation is faithful. First, under retargeting our FMRoPE arm **wins** — NLL is better by 0.061/0.182/0.279 — which is the behaviour the paper reports for a target-aware range method, and an incorrectly weakened implementation would not produce it. Second, the fixed-range comparison pins the extrema and log span by construction, so it does not depend on the base value the FMRoPE rule selects.

**这一条同时是最好的防守**:我们**报告了自己输的那个方向**,这是实现忠实性的最强证据。

### z-3 「retarget 之后 FMRoPE 赢,所以你们的方法更差。」

> Those are two different comparisons and we report both. Under retargeting a target-aware range method is stronger, which is what it is for; we state this as a boundary on the claim. Under a fixed range the ordering is the other way, which is the condition that isolates interior allocation. They also compose rather than compete: applying the target-range rule on top of an EVQ grid improves that model by **0.098/0.529/0.638** NLL at 512/1K/2K. Our claim is that allocation is a distinct axis, not that it replaces range selection.

### z-4 「8B 只是在已发布 checkpoint 上做 LoRA,不算真正的规模验证。」

> The 8B result is a retrofit and we label it as one. The scale evidence has two modes and we report them separately: EVQ as the frequency table when training from scratch (OLMo-2 1.485B, from the public step-0 initialization, matched recipe), and EVQ swapped into a released checkpoint and recovered with a short LoRA fine-tune (LLaMA-3-8B, **176.3 → 21.5** PPL at 16K; 16K RULER macro **0.295 → 14.03%**). Neither is a from-scratch 8B pretraining run, which we did not do and do not claim.

### z-5 「dead channel 那条你们还是在用。」

> We do not claim that observation. §2 of the submission credits channel inequality to prior work, citing Barbero et al. (2025) and Resonance RoPE. Oka et al. credit it to the same source: their §3 opens by investigating "the frequency band identified by Barbero et al. (2025)".

---

## 4. AC(XLtL)—— 决策者,最可能在最后 24 小时提问

### AC-1 「这些新实验是已经在论文里,还是承诺加进去?」

**必答准确,这关系诚信。**

> Both, and we distinguish them. Already in the submitted paper: the 8B LoRA evaluation (Appendix D, Table 23), the 750M continuation (Table 12), the three-seed 432M MLA study (Table 18), the video-DiT results (Table 14), and the YaRN substrate comparison (Table 3). Run after submission and reported in the response as revision commitments: the exact-range control and three-seed factorial, the three-level fixed-schedule ladder, the held-out base/head suite, the 1.485B from-scratch and retrofit results, and the Q/K-only adaptation-scope ablation. We cannot upload a revised PDF during the discussion period, so each of these is stated as a commitment with the numbers given in full.

### AC-2 「三位审稿人分歧很大。单看最强的一条证据是什么?」

> The exact-range control. With both sampled extrema, the log span, initialization, token order, optimizer, budget and all 32 evaluation anchors held fixed, and only the 30 interior frequencies varied, OOD NLL improves by **0.478/0.205/0.113** at 512/1K/2K, winning **32/32, 27/32 and 22/32** anchors; a three-seed factorial reproduces the direction across bases 500K/1M and head dimensions 32/64/128. This is the experiment that separates our variable from the one FMRoPE sets, and it was built specifically for the question the metareview poses.

### AC-3 「窗口内退步很明显。这个贡献净值为正吗?」

> We think the question turns on which of two quantities is meant. Training from scratch, the in-window cost is **+0.0381 NLL** at 4K against −0.0437 at 8K and −0.1351 at 16K — small, and monotone in the extrapolation ratio. Retrofitting a mature checkpoint costs much more, and we report that openly (13-family RULER macro 82.16 → 37.51% at 4K, improving to 42.44% when adaptation is restricted to Q and K). Oka et al. report the same ordering for base selection, with FMRoPE underperforming conventional RoPE in short contexts, so the trade-off appears to be structural to this design space rather than specific to our rule. We would not claim the retrofit path is currently a net win in-window; we would claim the allocation axis is real, identifiable under controls, and reaches 8B.

### AC-4 「漏引是疏忽,还是新颖性本身有问题?」

> The omission is real and it is ours; Oka et al. should have been cited and directly compared. Whether it also defeats novelty is the question the exact-range control was built to answer, because that control fixes precisely what the FMRoPE rule sets and varies only what we claim. We would ask that the determination rest on that experiment rather than on the citation gap alone.

### AC-5 「作者是不是在给委员会施压?」(若 confidential comment 被质疑)

> Not our intention, and we withdraw the framing if it read that way. The confidential note lists four premises that can be checked against the submission and the source the review cites, with locators for each. We are not asking that any review be weighted differently; we are asking that the checkable parts be checked.

---

## 5. 跨审稿人共同追问

### X-1 「为什么不跑更多种子 / 更大模型?」

> Compute. The controls that isolate the variable are the ones we could afford to repeat across seeds, and we prioritized those over adding scale points with single seeds. The revision will state seed counts inline for every result rather than in aggregate.

### X-2 「训练和评测数据有重叠吗?」

> Training and evaluation rows are disjoint in every result we report, with zero exact-row overlap and, for 2Wiki, zero QA-identity overlap. We note the boundary explicitly: the RULER results are task-family adaptation on held-out rows, not clean unseen-task transfer, and we describe them that way.

### X-3 「32K 表现如何?」

**必须如实答 —— 这是负结果,已在台账。**

> At 32K the picture is negative on task endpoints and we say so: after matched 13-family adaptation, both EVQ and untouched Native score zero across all 13 RULER tasks, and the Native-LoRA arm completed only 10 of 13 cells with all ten at zero. The 32K evidence we have is on the probability side only (8B NLL 6.899 → 4.851). We do not claim usable 32K capability.

### X-4 「跟 LeRoPE(arXiv 2607.10134)什么关系?」

**若被问到才答。这是并行工作 —— 我们投稿 2026-05-01,它 2026-07-11。**

> It is concurrent work, posted after our submission and after these reviews, so it does not bear on novelty at submission time; the revision will cite and discuss it. Two of its findings are relevant here. It trains a 52M–2.5B ladder with the RoPE frequencies as free parameters and finds the learned profile converges to a non-geometric one, consistently across seeds and scales — independent evidence that the geometric grid is a design choice rather than a neutral default. And on the confound Reviewer 27bE identified: training with those frequencies **frozen**, with zero learned positional parameters, captures **63.6%** of the gain, while p-RoPE captures 10.4%. On scope, LeRoPE optimizes in-distribution loss and its authors report it degrades more sharply than RoPE under naive extrapolation, which is the regime we address; we cite it as evidence that the table is a design axis worth optimizing, not as evidence for our particular allocation.

---

## 6. ⚠️ 危险问题 —— 只有被**直接**问到才答,答完即止

| 触发 | 回答范围 | 绝不做 |
|---|---|---|
| 「那 32 参数基线在论文里叫什么?」 | 见 **27-1**:承认 DAPE 标签不准确,陈述实际是什么,承诺修订 | 不重提 455.3/477.7 的调参辩护;不写长段自责 |
| 「那一行是什么方法?」(只问内容) | 32 个可学习 inv_freq,layer-shared,从几何初始化,标准注意力算子 | **不主动提标签问题** |
| 「你们实现的是哪个版本的 YaRN?」 | 如实描述实际实现的组件,并说明 Table 3 的结论是 substrate dependence(同一固定变换作用于两个基底),不依赖 YaRN 实现的最优性 | 不声称实现了完整官方 YaRN;不主动展开 |
| 「Geo+YaRN 调参后会不会赢?」 | 已在 Dz6s §3 答过:我们同意固定 scale 无法排除更好调参的 Geo+YaRN,我们没跑那个联合 sweep | 不再列举 s/β_fast/β_slow/mscale 这些组件名 |
| 「exact-range control 是多大模型?」 | 50.9M–151.9M,是机制控制,规模由 1.485B/8B 单独回答 | 不含糊其辞 |
| 「niah_single_2 有退步吗?」 | 是,4K 保持矩阵上从 0.70 降到 0.55,局部退步 | 不隐瞒 |
| 「held-out 任务上 EVQ 表现如何?」 | 冻结 4K 屏幕上 UUID 干扰检索与变量追踪均为 0%;这是我们不宣称 unseen-task transfer 的原因 | 不美化 |

---

## 7. 数字速查(引用前核对本表)

**分配轴(机制控制)**
- exact-range 单种子:**0.478/0.205/0.113** @512/1K/2K;**32/32, 27/32, 22/32** anchors
- exact-range 三种子:−0.3159/−0.1949/−0.1674;3/3 种子
- retarget 反转:**+0.061/+0.182/+0.279**(FMRoPE 赢)
- 组合:EVQ + target-range 改善 **0.098/0.529/0.638**
- 固定 schedule 三种子:**−0.256/−0.305/−0.223/−0.238** @1K/2K/4K/8K
- Level 2 native-endpoint:−0.113/−0.149/−0.207/−0.190/−0.099 @512/1K/2K/4K/8K
- 匹配指数:+0.0007 NLL,p=0.836
- C_app:exact-kernel 碰撞分数降 **24–92%**,12 配置

**τ**
- 21 配置内选中 τ 始终在 **0.75×–1.5×**
- vs midpoint-Geo:7/9 配置均值,18/27 配对
- vs pilot 邻居:**3/9**(必须一起报)

**泛化**
- held-out base 1M / d_head 128:**−0.802/−0.664/−0.433/−0.287/−0.212** @1K/2K/4K/8K/16K;512 处 **+0.069**

**规模 — 从头训**
- OLMo-2 1.485B:PPL 161.19/167.45 @4K,163.88/156.87 @8K,182.73/159.64 @16K
- NLL Δ:**+0.0724/+0.0381/−0.0437/−0.1351** @2K/4K/8K/16K;122/128、126/128 文档

**规模 — 换表 retrofit**
- OLMo-2 全串+EOS exact:95→100 / **18→98** / **0→60** @4K/8K/16K;超 gap 子集 48/50 @8K,**31/66** @16K
- RULER Q/K/V/O:82.16/37.51 · 0.08/21.29 · 0/6.13
- RULER Q/K only:72.19/**42.44** · 2.02/**31.63** · 0.38/5.03
- 2Wiki exact:22.0/21.5 · 0/**17.5** · 0/**4.0**;F1 25.99/24.84 · 0.07/21.48 · 0/8.57
- 8B:16K macro **0.295→14.03%**;8K official 94.44/77.60,normalized exact 17.69/**21.54**
- 8B NLL:1.919/2.309 @8K,4.691/**3.181** @16K,6.899/**4.851** @32K
- 8B 提交版 PPL:**176.3→21.5** @16K,1942.5→104.3 @32K
- 独立复现:两个 EVQ 种子 69/100、67/100 @8K;fresh long-gap 49/100、48/100;Native 均 0/100

**架构**
- MLA 432M 三种子:PPL@16K **138.8±5.5 → 95.6±4.1**(16 rotary channels)

**负结果(被问必答)**
- 32K RULER:两臂全零;Native-LoRA 仅完成 10/13 且均零
- held-out 任务(UUID / VT):0%
- niah_single_2:0.70 → 0.55
- 干净 LongAlign+Tulu 全 RULER:9.74/4.01/2.05%
- 七臂非 RULER 搜索:最好 4K 0.1167,对照 0.5700

---

## 8. 绝对不能说

- ❌ EVQ 取代或普遍优于 target-aware range scaling(retarget 下 FMRoPE 赢)
- ❌ Cosh 在 schedule 中经验最优(匹配指数与二带 schedule 在某些长度更强)
- ❌ τ = d/√L 是普适或精确最优
- ❌ MLA 的 71.1 是纯 EVQ(纯 EVQ 是 **95.6**,71.1 需额外 s=4 变换)
- ❌ 把 98/100(全串+EOS)与 69/100(严格首数)当同一实验
- ❌ 32K 可用
- ❌ unseen-task transfer / 无灾难性遗忘
- ❌ LeRoPE 验证了 EVQ 的外推(它 naive 外推更差)
- ❌ EVQ 的表 ≈ LeRoPE 学出来的表(从未对比)
- ❌ 任何暗示 AC 应当降低某审稿人权重的措辞
