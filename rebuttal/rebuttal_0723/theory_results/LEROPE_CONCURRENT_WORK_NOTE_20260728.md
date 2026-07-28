# LeRoPE (arXiv 2607.10134v1) — concurrent-work note

日期:2026-07-28 · 状态:`EXTERNAL_SOURCE_VERIFIED` · PDF 存于 `rebuttal_0723/LeRoPE_2607.10134v1.pdf`

**LeRoPE: Learnable RoPE Frequencies Improve Language Modeling**
Petros Karypis, Sean O'Brien, Shreyas Kadekodi, Rui Zhu, Julian McAuley (UC San Diego)
arXiv:2607.10134v1 [cs.LG] · **11 Jul 2026**

---

## 0. 时间关系(先确定这个)

我们投稿 **2026-05-01**,评审 **2026-06**,metareview **2026-07-22**。
LeRoPE **2026-07-11** —— **在投稿之后、在所有评审之后。**

→ **并行工作。不能用于质疑我们的新颖性,我们也没有回应义务。**
→ 我们 AC_PUBLIC 里 "To our knowledge … the first closed-form, zero-parameter training-time rule" 的 "to our knowledge" 锚定投稿时,**仍然成立**;但 camera-ready / 下一版必须处理它。

---

## 1. 方法是什么

> "LeRoPE adds **one learned scalar per frequency band, 32 parameters in total** for each of our models."

- 每个频率带一个可学习标量,**所有层和头共享**
- 初始化 α_m = 0,**初始时等价于标准 RoPE**
- LeRoPE 参数不加 weight decay,单独 LR 网格搜索(η ∈ {2^(−i/2)})
- 模型阶梯 **52M → 2.5B**,Chinchilla token 预算

**→ 这和我们论文里那个 `free_inv_freq`(32 参数 layer-shared learnable inv_freq,被错标为 DAPE 的那一行)是同一类方法。**

---

## 2. 三个对我们有利的发现(可引用,已核原文)

### ① 学出来的频率是非几何的,跨 seed / 跨规模一致

> "low-wavelength bands closely track RoPE (y = x), mid-wavelength bands begin to diverge, and **high-wavelength bands are pushed to a much larger, near-constant wavelength — i.e. their frequency is driven toward zero**. … This pattern is **consistent across all model scales and seeds we test**."

并且**分歧点随训练长度移动**:

> "models trained on longer contexts diverge from the RoPE frequencies at lower frequencies than their short-context counterparts"

**对我们的意义**:一个被放开自由选择频率表的网络,系统性、可复现地离开几何分配 —— 独立支持"几何分配不是中性实现细节"。L_train 依赖也和我们 τ ∝ d_eff/√L_train 的结构方向一致(**注意:只是方向一致,不是同一个规则,不可声称等价**)。

### ② Fixed LeRoPE —— **最有价值的一条,正中 27bE 的混淆质疑**

Table 2(217M 模型,validation PPL):

| Method | PPL | % of LeRoPE gain |
|---|---:|---:|
| RoPE | 18.2377 | 0% |
| p-RoPE | 18.2269 | 10.4% |
| **Fixed LeRoPE**(冻结另一次独立训练学到的频率) | **18.1714** | **63.6%** |
| LeRoPE(全学) | 18.1335 | 100% |

> "Training with fixed LeRoPE frequencies yields 63% of the improvement of full LeRoPE, suggesting that joint training dynamics and the final frequency set are both partially responsible."

**对我们的意义**:**一张固定的非几何表、训练期零位置参数,拿走 63.6% 的收益。** 这是外部独立证据,证明"收益主要在表本身,不在学习过程" —— 而这正是 27bE 问的 allocation shape vs parameterization/optimization effort。

### ③ Substrate dependence 被独立复现

> "LeRoPE with NTK-by-parts scaling and YaRN's attention temperature **outperforms the same recipe applied to RoPE and p-RoPE** … the best configuration for extrapolation overall is LeRoPE with YaRN"

NIAH(2.52B,YaRN,到 32k):"LeRoPE outperforms both RoPE and p-RoPE at every evaluated length"

**对我们的意义**:同一个推理期变换、不同的训练期表 → 恢复出的能力不同。**这就是我们提交稿 Table 3 报的现象(Geo+YaRN 61±3% vs EVQ+YaRN 100±0%),由第二组人独立测到。**

---

### ④ 【最重要的一条,2026-07-28 补】方向吻合 —— 而且这验证的是 surrogate,不是外推

**我们的目标函数是碰撞,不是外推。** C_app 是 broadband phase-collision 泛函,推导里没有长度泛化这个概念;外推只是我们选的读数。

**闭式解(Theorem 1)**:

    ρ_τ(φ) = τ·cosh(τ(1−φ)) / sinh(τ)

在 φ∈[0,1] 上**单调递减**。φ=0 → u=0 → ω=b⁰ = **最高频**(密度最大);φ=1 → **最低频**(密度最小)。
**→ cosh 把通道密度往高频端挪,弱化低频端。**

**LeRoPE 学出来的**:低频(高波长)带被推向 0,高频带略微加快。
**→ 同一方向:压低频、保高频。**

**意义**:LeRoPE **只优化分布内 loss**,却独立收敛到我们碰撞分析开出的同一方向。**这是机制在另一个读数上显形,而且去掉了长度泛化这个混淆因素。**

**这正对 AC metareview 里那句**:"the connection between **the surrogate objective, the cosh allocation**, and the recommended operating rule remains **only partially validated**" —— LeRoPE 提供的是这一段的外部独立验证(52M–2.5B,从零训练,跨 seed 一致)。

**⚠️ 机制不同,必须同时说清楚:**

| | 对低频带做什么 |
|---|---|
| EVQ-Cosh | **重分配** —— 所有通道保留,密度往高频挪 |
| LeRoPE | **压制** —— 低频带频率推向 0,等于移除 |

LeRoPE 自己承认终点接近 partial-RoPE:"p-RoPE can be interpreted as a coarse approximation of final LeRoPE frequencies, preserving faster frequencies while suppressing slower ones"。

**→ 可说"同一方向",不可说"同一个解"。我们没做过形状对比。**

---

## 3. 必须讲清楚的差异(否则会被反问)

### ① regime 不同 —— 这是最关键的分界

> "With unmodified positions, **LeRoPE degrades more sharply than RoPE and p-RoPE**" (§7, Naive Extrapolation)

**LeRoPE 优化分布内,naive 外推下比 RoPE 掉得更狠。我们做的是外推。**

他们的解释:dominant band 的 logit 贡献是周期 ≈ 2.2·L_train 的负半周正弦,**训练窗口外变成正的**。需要 dominant-band interpolation 或 YaRN 才能救回来。

### ② 成本不同

LeRoPE:32 个学习参数 + 无 weight decay + 单独 LR 搜索 + **一次训练跑才能发现那张表**。
EVQ-Cosh:闭式初始化,零参数,不需要发现过程。

**Fixed LeRoPE 的 63.6% 恰好说明"一张好的固定表能拿走大部分价值" —— 而"怎么不靠搜索得到一张好的固定表"正是我们在回答的问题。**

### ③ 🚫 绝对不能说的

- **不能说 EVQ 的表 ≈ LeRoPE 学出来的表** —— 我们没有做过任何对比
- **不能说 LeRoPE 验证了 EVQ 的外推收益** —— 它的 naive 外推更差
- **不能暗示我们比 LeRoPE 好** —— 没有共同基准

---

## 4. 其他可能有用的细节

- **Dominant positional band**:λ ≈ 2.205·L_train,across 6 runs(4 个规模 @ L_train=2048 + 217M @ 4096/8192)。Fixed-frequency RoPE 模型**没有** dominant band。
- **Leave-one-out**:zero 掉 band 17(dominant)增加 loss **0.762 nats**;第二大的 band 26 只有 **0.069 nats**。
- p-RoPE 被他们解读为 "a coarse approximation of final LeRoPE frequencies, preserving faster frequencies while suppressing slower ones"。
- 他们引用了 **Oka et al. [2026a] / [2026b]** —— 即 FMRoPE 那条线,说明这个方向在并行推进。
- Seed 控制:217M 上重复实验,**同 seed 内比较**(共享初始化和数据顺序)。

---

## 5. 当前决策(2026-07-28)

**结论:净利好。最有价值的单条是 Fixed LeRoPE 63.6% vs p-RoPE 10.4%。**

**建议动作**:讨论期发一条**短的独立 Official Comment**,理由是它是外部独立证据且正中 27bE(摇摆票)的核心质疑。草稿见下。**发不发由作者决定。**

**不做的事**:
- 不改已发的五份(Revisions 留痕,且内容无需修改)
- 不跑 EVQ vs LeRoPE 对照(六天内出坏结果代价灾难性,而且没人要求)

### 备用草稿 v2(英文,约 1,150 字符)—— 以 surrogate 验证为主线,强于 v1

> **Concurrent work bearing on the surrogate and the allocation axis.** After our submission and after these reviews, LeRoPE (arXiv 2607.10134, 11 July 2026) trained a 52M–2.5B ladder with the RoPE frequencies as free parameters. Three of its findings bear on questions raised here.
>
> First, the learned frequencies converge to a **non-geometric** profile that is consistent across seeds and scales, and the point of departure from the geometric grid tracks the training length.
>
> Second, the direction of that departure is the one our collision surrogate prescribes: the low-frequency end is de-emphasised while the fast bands are preserved. Our objective is phase collision, not length generalisation — extrapolation is a readout we chose, not the target — so an in-distribution result of this kind speaks to the surrogate directly. We note the mechanisms differ: EVQ-Cosh redistributes a fixed channel budget, whereas LeRoPE drives the slowest bands toward zero, which the authors themselves relate to p-RoPE. We claim directional agreement, not a matching allocation; we have not compared the two profiles.
>
> Third, and directly on the confound Reviewer 27bE identified: training with those frequencies **frozen** — zero learned positional parameters — captures **63.6%** of the gain, while p-RoPE captures 10.4%. Most of the benefit is in the table, not in learning it.
>
> On scope: LeRoPE targets in-distribution loss and degrades more sharply than RoPE under naive extrapolation, which is the regime EVQ-Cosh addresses. We cite this as independent evidence that the frequency table is a design axis worth optimising, not as evidence for our particular allocation.

**v1(较短,仅讲设计轴)已弃用 —— v2 把主线换成 surrogate 验证,正对 AC metareview 的 "surrogate objective … only partially validated"。**

---

## 6. 对下一版论文的影响(不管接受还是重投)

1. **必须引用并讨论 LeRoPE**,和 FMRoPE 放在同一张 related-work 对照表里。
2. **三轴分类需要加一行**:learned/searched frequency(LeRoPE、LongRoPE 的 per-channel search)是第四类,或者放在第三轴内部区分"学出来的 vs 解析构造的"。
3. **我们的 `free_inv_freq` 对照获得了新意义** —— 它是 LeRoPE 类方法在**外推 regime** 下的表现。LeRoPE 自己说 naive 外推更差,和我们 455.3 vs EVQ 333.7 的方向**一致**,不矛盾。这条在下一版可以正面写。
4. **最有价值的潜在实验**:EVQ 的闭式表 vs Fixed-LeRoPE 的学出表,在同一 OOD 基准上对比。若接近 → "闭式解逼近搜索解";若不同 → 两者优化不同目标。**现在不做,下一轮做。**
