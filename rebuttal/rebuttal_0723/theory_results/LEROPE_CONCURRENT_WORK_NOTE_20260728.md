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

### ④ 【2026-07-28 修订 — 原版本错误,已重写】同一权衡的两个相反方向

**⚠️ 先撤回一个错误判断。** 初稿写过"方向吻合,同一机制在不同读数上显形"。**错。** 两篇的模式是**相反**的:LeRoPE 窗口内赢、naive 外推更差;我们窗口内付代价、窗口外得收益。相反的模式不可能是同一机制的两个读数。

**EVQ 实际做的(τ=4,b=500K,已数值验算 φ_k = 1 − arcsinh((1−u)sinh τ)/τ):**

| u(几何位置) | 几何波长 | EVQ 波长 |
|---|---:|---:|
| 0.5 | 707 | **9.7** |
| 0.75 | 4,000 | **76** |
| 0.9 | 34,000 | **1,720** |
| 0.99 | 380,000 | **206,000** |

**EVQ 把预算挤向短波长(高频)端,长波长区只留稀疏覆盖(端点仍被 pin 住)。**

**机制(能同时解释外推收益和窗口内代价):**

以"该通道在训练中跑过几个完整周期"划分 —
- **波长 ≪ L_train**:训练中转过很多圈 → 超出 L 后进入**已走过的相位区域**,行为是训练 territory 的重复 → **外推稳**
- **波长 ≫ L_train**:训练中连一圈都没转完 → 超出 L 后相位进入**从未见过的区域** → **外推崩溃点**

→ EVQ 把预算从"从未被完整激励"的通道搬到"被激励很多次"的通道 → **外推稳**
→ 同时,中慢速通道正是窗口内最细的单调位置分辨率来源,抽稀它们 → **窗口内代价**

这与 C_app 的 Green 核 min(φ,ψ) 一致:该核惩罚**两个都慢**的通道对 —— 波长 10 万与 20 万的通道在任何真实上下文里都近似常数,**是重复品**。几何分配把预算浪费在这堆重复品上。

**LeRoPE 的反向选择:** 优化分布内 → 学出 **λ ≈ 2.205·L_train** 的 dominant band,波长卡在训练窗口上,窗口内单调位置信号最干净。他们自己说该 band 的 logit 贡献是周期 2.2·L_train 的负半周正弦,**"on relative distances beyond those seen during training, the contribution becomes positive"** → 超出训练长度翻号 → naive 外推崩得比 RoPE 更狠。

**结论:LeRoPE 把赌注压在恰好 = L_train 的尺度;EVQ 把赌注从那个尺度撤走。同一权衡,相反方向。**

**这个框架对 rebuttal 的价值:** 它把我们的 in-window cost 从"缺陷"变成"该权衡的另一端",并且有一篇独立论文从相反方向验证了这个权衡是结构性的、不是我们方法的毛病。

**🚫 仍然不能说:** EVQ 表 ≈ LeRoPE 表(方向就不同,更别说形状);LeRoPE 验证了 EVQ 的外推。

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
> Second, the two methods traverse the same trade-off from opposite ends, which we think clarifies rather than complicates the picture. LeRoPE optimises in-distribution loss and converges on a dominant band at a wavelength of roughly 2.2 times the training length; the authors report that this band's logit contribution changes sign beyond trained distances, and that LeRoPE degrades more sharply than RoPE under naive extrapolation. EVQ-Cosh moves budget away from that scale, which costs in-window resolution and is why we report an in-window trade-off. The trade-off between in-window resolution and behaviour beyond the training window therefore appears to be structural, and is now visible from two directions in independent work.
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

---

## 7. 【2026-07-28】由 LeRoPE 引出的下一篇研究议程

**状态:假设,未验证。以下所有数值均已复算(τ* = d_head/√L_train,来源 `scripts/analysis/unification_plot*.py`、`verify_softmax_transport.py`;b=500K;λ = 2π·b^φ)。**

### 7.0 先撤回两个说法(2026-07-28 同日自查)

**撤回 A:「EVQ 抽稀了 L 尺度频带,这就是 in-window 代价的来源」** —— 数值不支持。

EVQ 相对几何的通道密度 ρ(φ) = τ·cosh(τ(1−φ))/sinh(τ):

| 波长 | τ=2(OLMo-2 实际值) | τ=4 |
|---:|---:|---:|
| 23 | 1.71× | 2.68× |
| 322 | 1.19× | 1.21× |
| 2,305 | 0.92× | 0.67× |
| **8,563 ≈ 2.2·L(L=4096)** | **0.79×** | 0.46× |
| 61,300 | 0.65× | 0.27× |
| 3,141,593 | 0.55× | 0.15× |

**EVQ 的削减集中在 λ ≫ 任何部署长度的区域** —— 即在真实上下文中近似常数、彼此高度重复的通道。**这正是 C_app 的 Green 核 min(φ,ψ) 所惩罚的对象(两个都慢的通道对),机制自洽。** L 尺度频带在 τ=2 下只削到 0.79×,很轻。

**撤回 B:「窗口内代价」是一个东西** —— 是两个,不能混:

| 观测 | 量级 | 归因 |
|---|---|---|
| OLMo-2 1.485B 从头训 @4K NLL | **+0.0381** | 真·分配代价。与 0.79× 轻度削减量级一致 |
| LoRA 换表后 RULER macro 82.16→37.51 | 巨大 | **换表适配失配**(模型由几何预训练),非分配内禀属性;Q/K-only 探的正是这个 |

**→ 论文层面这是好消息:分配本身的窗口内代价很小。下一版应把这两者明确分开陈述。**

### 7.1 保留的核心二分(这可能是下一版理论章节的骨架)

在长距离 d 处,波长 λ 的通道 cos(2πd/λ):

| | 长距离状态 | 
|---|---|
| **λ ≪ d** | **行为良好但有歧义** —— 相位值训练中见过无数次,模型响应定义良好;但多个 d 映到同一相位 |
| **λ ≫ d** | **无歧义但未受训** —— 近似单调,但相位进入训练中从未到达的区间 |

**外推失效有两个独立来源:歧义(aliasing)和未受训(OOD phase)。现有工作没有把它们分开度量。**

- EVQ 押注「行为良好但有歧义」侧,同时删掉冗余的极慢通道
- p-RoPE 删的是慢端(减少未受训项)
- LeRoPE 押注恰好 λ≈2.205·L_train(窗口内无歧义且受训),超出即翻号

**这个二分能同时安置四种方法,比现在论文里的单一碰撞叙事解释力强。**

### 7.2 实验 A(零成本,最高优先)—— τ 摆动是否是「对准 λ*」

**动机:** LeRoPE leave-one-out:zero 掉 dominant band(λ≈2.205·L_train)增加 loss **0.762 nats**,第二名 band 26 只有 **0.069 nats** —— **11 倍**。若窗口内位置信号如此集中于单一波长,则**通道落点精度**比该区域密度更重要。

我们的表**没有任何机制把通道钉在 λ\* 上**,落点是副产品:

| L_train | λ*=2.205L | EVQ 最近通道/λ* | 几何最近通道/λ* |
|---:|---:|---:|---:|
| 512 | 1,129 | **0.86×** | 0.94× |
| 1,024 | 2,258 | 1.16× | 1.06× |
| 2,048 | 4,516 | 1.04× | 1.09× |
| 4,096 | 9,032 | 1.02× | 0.91× |
| 8,192 | 18,063 | 1.11× | 1.03× |

(d_head=128, K=64, b=500K, τ=d_head/√L)

**假设 H1:** 21 个配置中经验选出的 τ,与「哪个 τ 使某通道最接近 2.205·L_train」相关。

**做法:** 纯 post-hoc,数据已在仓库。对每个配置,在 τ ∈ {0.75,1.0,1.25,1.5}× 上算 min_k |log(λ_k/λ*)|,与实际选中的倍数做秩相关。

**若成立** → **那个 0.75×–1.5× 的摆动不是噪声,是 schedule 在对准 λ\*。这直接补上 27bE 咬得最狠的洞(推导给出标度形式但给不出常数)。**
**若不成立** → 干净地排除一个解释,成本为零。

⚠️ **本轮 rebuttal 不用。** 即使成立也是新论证,讨论期引入新机制主张风险高于收益。

### 7.3 实验 B —— 钉住 λ* 的保留带(zero-parameter 变体)

**设计:** K 个通道中,把一个通道解析地钉在 λ = c·L_train(c 由 LeRoPE 报的 2.205 起步,或从我们自己的数据标定),其余 K−1 个按 cosh 分配。仍然闭式、零参数、无 target。

**预期:** 若 H1 成立,应消掉大部分 +0.0381 的从头训 in-window 代价而不动外推收益。
**若 H1 不成立,本实验多半也无效 —— 先做 A 再做 B。**

**注意:** 这里的 c 依赖 L_train 而非 L_target,**不会引入 FMRoPE 式的 target 依赖**(我们在 AC_PUBLIC / AC_CONFIDENTIAL 中把「不需要 L_target」作为区分点,该性质必须保住)。

### 7.4 实验 C —— 把「歧义」与「未受训」分开度量

按 λ 相对部署距离 d 分组,分别消融:
- 组 1(λ ≪ d):制造 aliasing
- 组 2(λ ≳ L_train):制造未受训相位

测各组对长距离 NLL 的边际贡献。**若两条曲线随 d 的走向不同 → 二分成立,可作为下一版的核心图。**

### 7.5 ❌ 已否决的想法(留档,避免重走)

**「外推时低频更重要,把高频置零(→NoPE)」** —— 方向反了。高频通道在长距离处相位是训练中反复见过的,模型响应定义良好,只是有歧义;把它们关掉后只剩**未受训**的慢通道,而那正是外推崩溃的来源。p-RoPE(删慢端)与 LeRoPE 对 p-RoPE 的解读("preserving faster frequencies while suppressing slower ones")都指向相反方向。

### 7.6 优先级

**A(零成本,先做)→ 视 A 结果决定 B → C 作为下一版理论图。全部在本轮讨论期之外。**
