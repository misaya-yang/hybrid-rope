# ICLR 2027 理论架构：从 attention 第一性原理到有限谱预算

- **状态**：历史理论设计探索；当前 claim 与写作边界以
  `ROPE_CAUSAL_VARIABLES_AND_ZERO_TRAINING_RETROFIT_20260823.md`、
  `ICLR2027_RESEARCH_SYNTHESIS_20260819.md` 和
  `FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md` 为准
- **日期**：2026-08-19
- **回答的是用户提出的第 3、4 号问题**：
  - Q3 非几何分配的理论没完善——理论上应能做到 in-window 不输 LeRoPE、外推强于原始 RoPE；
  - Q4 cosh collision kernel 与原始 EVQ 是「瞎猫碰上死耗子」，外推只是表现而非设计目标，需要从第一性原理重新推 RoPE↔attention 的关系。
- **纪律**：每条 CLAIM 标注 STATUS（exact proved / proved-under-assumption /
  empirically supported / conjecture）与 owner。不得把 conjecture 写成定理。
  所有数字来自命名 owner，未做任何再推导。

---

## 0. 一句话结论

旧稿的因果顺序是**反的**。旧稿写的是「我们造了一个 collision surrogate → 它的
解是 cosh → cosh 外推更好」。真实的因果链是：

> **一个有限 RoPE 表就是一组二维谱基。它的 phase-invariant 子空间几何决定了模型
> 在训练期能把哪些距离区分开（positional identifiability），而权重在训练中学会
> 使用这组基。因此 (i) 内部分配是一个真实的训练期自由度，(ii) 分配的目标是
> 「让有限的 K 个二维子空间尽可能少地互相冗余地覆盖需要被区分的距离」，
> (iii) 外推收益是这个目标的推论——不是目标本身。**

cosh 不是「答案」，是这个目标在一个**声明清楚的凸代理**下的闭式驻点。这不是弱化，
反而是能扛住对抗审稿的唯一写法：我们拥有的是**轴**（interior allocation），
cosh 是这条轴上零参数、有几何极限的代表点。

---

## 1. 第一性原理链：attention 到底拿 rotary phase 做什么

### L1. 一个频率是一个二维子空间，不是一个余弦特征

**CLAIM**：单个 RoPE pair 对 relative-position attention logit 的贡献是
\[
f_\omega(\Delta)=C\cos(\omega\Delta)+D\sin(\omega\Delta),
\]
其中 \(C,D\) 由该 pair 上的 Q/K 内容决定。因此频率 \(\omega\) 的自然对象是
\[
V_\omega=\operatorname{span}\{\cos(\omega\Delta),\sin(\omega\Delta)\},
\]
一个二维子空间。

**STATUS**：exact（代数恒等式）。
**OWNER**：`FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md` §2.1。

**这一步为什么是全文的起点**：论文现行的 exact kernel
\(K_{\cos}(\omega,\nu)=\mathbb E_D[\cos(\omega\Delta)\cos(\nu\Delta)]\)
只是 \(D=0\) 的一个内容相位切片，漏掉 sin–sin、cos–sin、sin–cos 三个 Gram 分量，
**并且不对 pair 内的 phase rotation 不变**。也就是说：旧 kernel 的数值会随着一个
物理上无意义的相位选择而变。这是旧理论「瞎猫碰死耗子」感的技术根源——它优化的
量不是一个良定义的几何量。

### L2. 冗余的正确定义是 canonical correlation（相位不变）

**CLAIM**：令 \(x_\omega(\Delta)=[\cos(\omega\Delta)\ \ \sin(\omega\Delta)]\)，
\(S_\omega=\mathbb E[x_\omega^\top x_\omega]\)，\(H_{\omega\nu}=\mathbb E[x_\omega^\top x_\nu]\)，
whitened cross-Gram \(Q_{\omega\nu}=S_\omega^{-1/2}H_{\omega\nu}S_\nu^{-1/2}\)。
其奇异值 \(\sigma_1,\sigma_2\) 是两个子空间的 canonical correlations，且
\[
c_{\omega\nu}=\tfrac12\lVert Q_{\omega\nu}\rVert_F^2=\tfrac{\sigma_1^2+\sigma_2^2}{2}\in[0,1]
\]
对 pair 内相位旋转与任意可逆基变换不变。对 \(\Delta\sim\mathrm{Unif}[0,L]\) 有闭式
\[
H_{\omega\nu}=\tfrac12\begin{bmatrix}a(d)+a(s) & b(s)-b(d)\\ b(s)+b(d) & a(d)-a(s)\end{bmatrix},
\quad d=(\omega-\nu)L,\ s=(\omega+\nu)L,
\]
\(a(t)=\sin t/t,\ b(t)=(1-\cos t)/t\)。

**STATUS**：exact proved。
**OWNER**：同上 §2.2。

### L3. 「谱预算」是一个可以精确写出来的维数守恒律

**CLAIM（本轮最强的静态定理）**：对所有 pair 做 block whitening 得到全局
correlation Gram \(R\)（对角块为 \(I_2\)）。若 \(\bar c\) 是全部 \(c_{ij}\) 的平均，则
\[
\operatorname{tr}(R)=2K,\qquad
\operatorname{tr}(R^2)=2K\bigl[1+(K-1)\bar c\bigr],
\]
从而 Rényi-2 / stable effective rank **精确**满足
\[
\boxed{\,r_2(R)=\frac{(\operatorname{tr}R)^2}{\operatorname{tr}(R^2)}=\frac{2K}{1+(K-1)\bar c}\,}
\]

**STATUS**：exact proved。
**OWNER**：同上 §2.3。

**这就是论文标题里的 “spectral budget” 的严格版本**，而且它把标题从一个比喻升级
成一个恒等式：名义维数是 \(2K\)，实际可用维数被平均冗余 \(\bar c\) 压缩。
分配（allocation）唯一能动的东西就是 \(\bar c\)——**在 \(K\) 与端点都被钉死时**。
这正是 exact-range 实验的理论意义：它在实验上钉死了 \(K\) 和端点，只动 \(\bar c\)。

> **写作要点**：这条恒等式必须进正文，它把「预算」从叙事变成定理，
> 并且直接给了 exact-range 实验一个 pre-registered 的理论动机。

### L4. 慢通道不是「死」，是「重复」——而且退化极限是可以写出来的

**CLAIM A（\(L_2\) 度量）**：令 \(x=\omega L\)。当 \(x\to0\)，
\(\cos(xt)=1-x^2t^2/2+O(x^4)\)、\(\sin(xt)/x=t-x^2t^3/6+O(x^4)\)，因此
\(V_\omega\to\operatorname{span}\{1,\Delta\}\)。两个低频之间
\[
2-\lVert Q_{x,y}\rVert_F^2=\tfrac{19}{12600}(x^2-y^2)^2+O(\epsilon^6).
\]
数值核对：\(x=0.05,y=0.10\) 时 exact/leading \(=1.00058\)。

**CLAIM B（softmax 度量）**：attention categorical Fisher
\(F_{\rm sm}=\operatorname{diag}(p)-pp^\top\) 满足 \(F_{\rm sm}\mathbf 1=0\)，
即**常数方向被 softmax 消掉**。中心化后
\(\overline{\sin(\omega\Delta)}/\omega\to\Delta-\mathbb E_p\Delta\)，
\(-2\overline{\cos(\omega\Delta)}/\omega^2\to\Delta^2-\mathbb E_p\Delta^2\)，
所以 softmax 商几何收敛到
\(\operatorname{span}\{\Delta-\mathbb E_p\Delta,\ \Delta^2-\mathbb E_p\Delta^2\}\)。
50M 上 4 个 cells、1,920 个 head-query 观测的 canonical chordal deficit
log-log 斜率 `4.009–4.011`，与 \(O((\omega\Delta_{\max})^4)\) 一致，无反例。

**STATUS**：A、B 均 exact proved + 数值验证。
**OWNER**：同上 §3.1–§3.3。

**几何 RoPE 的实际损失**（\(L=4096,b=5\times10^5\)，只看 \(\omega L\le1\) 的 band）：

| \(K\) | 低频 pairs | 名义维数 | block-whitened \(r_2\) | 稳定维数损失 |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 6 | 12 | 2.0001 | 83.33% |
| 32 | 12 | 24 | 2.0001 | 91.67% |
| 64 | 24 | 48 | 2.0002 | 95.83% |

> **这张表是全文最有说服力的「为什么要重新分配」的一句话证据**：在 \(K=64\) 的
> 标准配置里，24 个 pair（48 个名义维度）合起来只提供约 2 维。这不是
> 「慢通道没用」（禁语），而是「慢通道彼此高度重复」。
> **禁语**：不得说 dead / unused / freely reclaimable。安全表述是
> *redundant, not necessarily unused*。

### L5. 旧 cosh surrogate 在这条链上的准确位置（Q4 的正面回答）

现在可以精确说出旧推导做对了什么、做错了什么。

**做对的**：\(\Capp\) 的两项——\(\tfrac\alpha2\int\rho^2\)（channel-load 集中惩罚）
与 \(\tfrac\beta2\iint\rho\rho\min(\phi,\psi)\)（**两个都慢**的 pair 的惩罚）——
在方向上正是 L4 的内容：慢–慢配对的 canonical redundancy 在
\(\omega L\to 0\) 时以 \((x^2-y^2)^2\) 的速率趋于满冗余，因此在 log-frequency 坐标下
「两者都慢」必须被惩罚。\(\min(\phi,\psi)\) 恰好是 \(-\partial_\phi^2\) 在 \([0,1]\) 上的
Green 核，这使一阶变分退化成 \(\rho''-\tau^2\rho=0\)，闭式解为 cosh。

**做错的 / 讲过头的**：
1. \(K_{\cos}\) 是 \(D=0\) 的切片，不是相位不变量（L1/L2）；
2. \(\Capp\) 是**对该切片的凸代理**，不是 canonical redundancy \(\bar c\) 的
   代理的推导结论——两者之间目前只有方向一致，没有定量控制；
3. 「cosh 最优 ⟹ 外推更好」这一步在数学上从来不成立，L6 有直接反例。

**因此 Q4 的诚实答案是**：cosh 不是猜的，但它的**证明力**比旧稿写的弱，
而它的**科学地位**比旧稿写的更清楚：

> \(\Capp\) 的唯一约束极小元是 cosh 密度；其 inverse CDF 给出零学习参数的闭式表，
> 且 \(\tau\to0\) 精确退化为几何 RoPE。这是一个 **construction theorem**，
> 不是 full-RoPE 最优性定理，也不是 task-loss 最优性定理。

**并且这恰好被实验支持而不是被实验削弱**：M4 factorial 里
deformation-matched exponential 与 rule-cosh **没有被分开**
（cosh − exponential \(=+0.00074\) NLL，bootstrap \([-0.0055,0.0075]\)，
exact sign-flip \(p=0.836\)；owner `M4_EXACT_RANGE_FACTORIAL_RESULT_20260726.md`）。

> **这是写作上的关键转向**：不要把它当成「我们的形状没被验证」的负结果。
> 它是**轴命题的正面证据**：两族形状完全不同、但都把预算从慢端搬向快端的
> 固定 schedule，预注册的 `1.25x` cosh 与 exponential 分别打赢 uniform
>（10/12 与 9/12）；公式 cosh 是 7/12，且与 exponential 打平。
> 能被识别的是**分配方向**，不是某条特定曲线。cosh 的独特之处不是它更优，
> 而是它是这条轴上**唯一有闭式、零参数、且以几何 RoPE 为极限**的代表点。
> 一句话正文写法见 §6。

### L6. 静态几何 ≠ 任务表现（必须自己先说出来）

**CLAIM**：static collision / rank / logdet 的改善**不蕴含**更好的外推或更低的
LM loss。这有直接反例，而且反例来自我们自己的 2×2。

**OWNER**：同上 §5.2、§5.4。50M、TinyStories val、\(L=512\)、base 500K、seed-42，
全部参数冻结、无训练、无 GPU：

| Weights | Runtime table | LM loss | PPL | bare geometry \(r_2\) |
| --- | --- | ---: | ---: | ---: |
| Geo | Geo | 1.9659 | **7.14** | 4.57 |
| Geo | EVQ | 4.3333 | **76.20** | **12.54** |
| EVQ | Geo | 3.1378 | 23.05 | 4.57 |
| EVQ | EVQ | 1.9685 | **7.16** | 12.54 |

最差的 cell（Geo weights + EVQ table）**静态 rank 从 4.57 涨到 12.54**，
PPL 却从 7.14 崩到 76.20。

**STATUS**：empirically established counterexample（CPU-only，可复现）。

> **为什么这必须写进正文**：这是 `RDz6s.3`（理论—实践链条要分层）最强的正面回答。
> 一篇自己给出「我的静态指标不能预测任务表现」的反例的论文，比一篇被审稿人发现
> 这一点的论文强得多。它同时杀掉了整类「最大化 rank 就行」的 naive 设计。

### L7. 表是坐标系，权重学的是坐标（co-adaptation）

**CLAIM**：对同一 2×2 做 factorial 分解，LM loss 上
\[
E_T=+0.5991,\quad E_W=-0.5965,\quad I_{T\times W}=-3.5367,
\]
交互项绝对值约为两个主效应的 **5.9 倍**。40 个 sampled-query group、500 次配对
bootstrap 的 95% CI：table `[+0.331,+0.956]`、weights `[-0.896,-0.175]`、
interaction `[-5.165,-3.039]`。

**STATUS**：empirically established（CI 不跨零）。
**OWNER**：同上 §5.3。

**解释（可写进正文）**：两个**自洽**系统 Geo/Geo 与 EVQ/EVQ 的训练长度 PPL 几乎相同
（7.14 vs 7.16）；两个 **post-hoc 换表** 的 cell 崩溃。所以频率表在训练期是一个
坐标系，权重学的是在这个坐标系里的系数；训练后换表不是纯几何干预，而是破坏
已形成的共适应。

### L8. 换表代价不是调参失败，是结构障碍（exact 定理）

**THEOREM（post-hoc frequency transplant obstruction）**：设 \(R_\Omega(\Delta)\)
为频率 multiset \(\Omega\) 的 block-rotation operator。若存在与位置无关的可逆线性
\(A,B\) 使
\[
A^\top R_{\Omega'}(\Delta)B=R_\Omega(\Delta)
\]
在某个含开区间的 \(\Delta\) 集上恒成立，则 \(\Omega'\) 与 \(\Omega\) 必须有相同的
frequency multiset（仅允许符号与排列；若频率重复，相似变换可在整个等频不变
子空间内混合；整数位置额外允许 \(2\pi\) alias）。

证明：\(\Delta=0\) 给出 \(A^\top B=I\)；于是
\(A^\top R_{\Omega'}(\Delta)A^{-\top}=R_\Omega(\Delta)\)；在零点求导得两个 block
generator 相似；相似保谱，而 generator 谱为 \(\{\pm i\omega_k\}\)。∎

**STATUS**：**PROVED**（exact, position-independent, invertible compensation）。
**OWNER**：`rebuttal/rebuttal_0723/theory_results/OLMO2_POSTHOC_FREQUENCY_TRANSPLANT_OBSTRUCTION_20260726.md`；
复述见 canonical report §8.2。

**边界（必须同时写）**：该定理**不**排除有限数据上的近似重训、非可逆或非线性映射、
V/O/残差路径、以及新算子。approximate residual 的定量下界仍是开放问题，
**但那不是 exact 定理的完成门槛**。

> **这条定理是本轮最大的未使用资产**：零 GPU 成本、exact、而且它把
> 「换表后 in-window 掉分」从一个尴尬的负结果，变成一个**被自己理论预测到的结果**。
> 审稿人看到的不再是「你的方法在成熟模型上有代价」，而是
> 「他们证明了这个代价必然存在，并给出了它的来源」。

---

## 2. Q4 的正面回答：为什么外推只是症状

把 L1–L8 串起来，可以给出一个**不依赖外推**的设计目标：

> **目标**：在 \(K\) 与端点固定时，让这 \(K\) 个二维子空间在模型**必须区分的距离集合**
> 上尽可能少地互相冗余，即压低 \(\bar c\)、抬高 \(r_2(R)\)。

外推收益是这个目标的**推论**，不是目标：

1. **相位复现论证（phase recurrence）**。\(1/\omega\ll\Ltr\) 的通道在训练中完成
   多个周期，因此 \(|\Delta|>\Ltr\) 的位置落在**已经被训练过的相位区域**；
   \(1/\omega\gg\Ltr\) 的通道在训练中连一个周期都没走完，超出 \(\Ltr\) 后进入
   **未训练相位**。
2. **L4 说这些慢通道彼此高度冗余**（\(K=64\) 时 24 个 pair 只值 ~2 维）。
3. 所以「把预算从慢端搬向快端」同时做到两件事：**降低 \(\bar c\)**（因为被砍掉的是
   重复子空间）与**增加落在已训练相位内的通道数**。外推变好是第二件事的直接后果。

**这就是 Q4 的答案**：原始 EVQ 观察到的外推增益，是「压低冗余」这个目标在
「训练相位支撑」这个附带结构上的**副产物**。旧稿把副产物写成了主结论，
所以整条链看起来像运气。新写法把主结论放回 \(\bar c\) / \(r_2\)，外推作为推论，
并且**推论有独立的可证伪预测**（见 §4 的 budget-crossing lemma 与
multi-key vs single-needle 预测）。

**同时必须自己说清楚的边界**（否则会被 L6 反例打脸）：
压低 \(\bar c\) 是**必要方向而非充分条件**——静态改善不蕴含任务改善（L6），
真正的收益必须经过训练期共适应（L7）才能兑现。这一句话恰好解释了为什么
from-scratch 有效而 post-hoc retrofit 昂贵（L8）。

---

## 3. Q3 的正面回答：in-window 与外推能否兼得

### 3.1 成本二分：in-window 的代价其实是两个不同的东西

\[
\Delta\mathcal L_{\rm in}(\rho')
=\underbrace{\Delta\mathcal L_{\rm alloc}(\rho')}_{\text{分配内禀，小}}
+\underbrace{\Delta\mathcal L_{\rm adapt}(\rho',\theta_{\rm pre})}_{\text{共适应失配，大}}
\]

**OWNER**：`LEROPE_CONCURRENT_WORK_NOTE_20260728.md` §7.0；
`research_notes/iclr2027/03_INWINDOW_EXTRAPOLATION_THEORY.md` §2。

**\(\Delta\mathcal L_{\rm alloc}\)（from-scratch，小）**：

| 证据 | 数字 | Owner |
| --- | --- | --- |
| 50M 2×2 自洽 cell 的训练长度 PPL | Geo/Geo **7.14** vs EVQ/EVQ **7.16**（\(L=512\)） | canonical report §5.2 |
| OLMo-2 1.485B same-init step-0→1000，2.097B tokens | \(\Delta\)NLL \(+0.0724\) @2K、\(+0.0381\) @4K（in-window 代价）；\(-0.0437\) @8K、\(-0.1351\) @16K | `OLMO2_1B_RELEASED_ROPE_BASELINE_20260725.md` |
| MLA 432M 3 seeds | 训练长度 \(+0.9\%\) PPL，\(2\times\) 外推 \(-31.1\%\) | `mla_scarcity_seed42_result_20260724.json` |

**\(\Delta\mathcal L_{\rm adapt}\)（retrofit，大）**：50M 2×2 的 post-hoc 换表
PPL \(7.14\to76.20\)；成熟模型上 OLMo-2 full-EVQ 换表后 4K RULER macro
\(82.16\to37.51\)。

### 3.2 现在能claim什么、不能claim什么（逐句可用措辞）

**可以claim（有 owner，逐句为真）**：

- ✅ 「从头训练时，非几何分配的 in-window 代价很小：在 50M 完全共适应的系统上，
  训练长度 PPL 为 7.14 与 7.16；在 1.485B 上，同初始化轨迹的 in-window NLL 代价
  为 \(+0.072\)@2K 与 \(+0.038\)@4K，同时 8K/16K 分别改善 \(-0.044\)/\(-0.135\)。」
- ✅ 「in-window 的**任务级**代价是 task-dependent 而非一致退化：phase-matched
  OLMo 上 4K 2Wiki exact 为 22.0 vs 21.5（近似打平），而同一对模型的 4K RULER
  macro 为 72.19 vs 42.44。」
- ✅ 「retrofit 的 in-window 代价与 allocation 的内禀代价数量级不同，且前者被
  L8 的 exact obstruction 定理预测。」

**不能claim（会被打）**：

- ❌ 「EVQ in-window 无损」——4K RULER 72.19→42.44 直接证伪，`R27bE` 类审稿人一定会查。
- ❌ 「EVQ in-window 优于几何」——我们没有任何一条 in-window **增益** 的证据；
  LeRoPE 有，我们没有。
- ❌ 「EVQ ≈ LeRoPE 的闭式版本」/「LeRoPE 验证了 EVQ」——canonical report §7 明令禁止，
  且无共同 benchmark。

**结论（Q3 的诚实答案）**：
> **目前仓库能支持的最强 in-window 表述是「近似平价（parity）」，不是「增益」。
> 而且这个平价只在 from-scratch / 完全共适应的体制下成立；retrofit 体制下有明确
> 代价，且该代价被我们自己的定理预测。**

### 3.3 与 LeRoPE 的关系：分工表（可直接改写成 §2 段落）

已核验的 LeRoPE 事实（owner：`LEROPE_CONCURRENT_WORK_NOTE_20260728.md`、
`rebuttal/rebuttal_0723/LeRoPE_2607.10134v1.pdf`）：每个 frequency band 学一个
scalar（共 32 个，层/头共享）；52M–2.5B 阶梯上学出一致的非几何 profile；
217M 上用另一轮学出的表从头固定训练，保留完整 LeRoPE validation-PPL gain 的
**63.6%**，p-RoPE 只保留 **10.4%**。

| LeRoPE 拥有 | 本文拥有 |
| --- | --- |
| 学出的 per-band 表；**in-window PPL 增益** | phase-invariant full-RoPE 子空间几何与 \(r_2\) 恒等式 |
| 「固定表可跨训练迁移价值」的外部证据（63.6%） | **fixed-range interior-allocation 识别**（端点/log-span 钉死） |
| joint table/weight 优化 | **exact frozen-retrofit obstruction** 与 2×2 co-adaptation 诊断 |
| 需要一次额外的 learned run | 闭式、零学习参数、以几何 RoPE 为极限 |

**可写的定位句**（安全且强）：
> LeRoPE 独立证明了 frequency table 是值得设计的变量，并用 Fixed-LeRoPE 证明
> 一张固定的好表本身携带显著价值（63.6% vs p-RoPE 的 10.4%）。这与本文互补：
> 它说明表值得学，我们说明**在范围被钉死后表内部的位置就已经是一个可识别的轴**，
> 并给出一个不需要 learned run 的闭式点。

**禁止**：说我们更好、说我们近似他们、说他们验证了我们。

### 3.4 「兼得」的理论程序（诚实标注为未完成）

要真正 claim「in-window 不输 + 外推更强」，需要把问题写成**约束变分**：
\[
\min_{\rho}\ \mathcal C_{\rm ext}(\rho;[\Ltr,L_{\rm tgt}])
\quad\text{s.t.}\quad
\mathcal C_{\rm in}(\rho;[0,\Ltr])\le(1+\varepsilon)\,\mathcal C_{\rm in}(\rho_{\rm geo}).
\]
两个零参数闭式候选（均只依赖 \(\Ltr\)，不引入 FMRoPE 式 \(L_{\rm tgt}\) 依赖）：
**pinned-\(\lambda^*\)**（一个通道解析钉在 \(\lambda=c\Ltr\)，其余 cosh）与
**protected-band two-regime**（保留 \([\lambda_1,\lambda_2]\ni O(\Ltr)\) 的几何通道，
补集上做 sub-budget cosh 分位构造）。\(\varepsilon\to\infty\) 退化为 EVQ-Cosh，
保护集取全体退化为 geometric——EVQ 与 native 成为同一族的两个端点。

**STATUS：CONJECTURE / 未跑**。owner：`research_notes/iclr2027/03_INWINDOW_EXTRAPOLATION_THEORY.md` §3 T2。

**本轮处理方式**：**写进 Discussion 作为最有信息量的下一步，不写进 Theory 作为结果。**
理由：(a) 它是审稿人会自己想到的问题，主动点名比被问强；(b) 它给论文一个明确的
后续方向，符合 `AC.4`「什么能改变推荐」的思路；(c) 没跑就写成结果会直接触碰红线。

**与 waterbed 的关系（必须写对）**：waterbed 约束的是 surrogate 全局，
纯重分配不可能处处占优。约束分配方案**不违反**它——它赚的是 waterbed 之外的
\(\Delta\mathcal L_{\rm adapt}\)，以及补集里那些 in-window 边际贡献 ≈ 0 的
重复慢通道（L4 + Green 核论证）。

---

## 4. 已有但正文没用的可证伪预测

### 4.1 Single-crossing budget shift（已在正文，保留）

对每个 \(\tau>0\)，\(\rho_\tau\) 与均匀密度恰好交叉一次，交点
\(\phi_c(\tau)=1-\tfrac1\tau\operatorname{arcosh}(\sinh\tau/\tau)\le1-1/\sqrt3\approx0.4226\)。
OLMo-2 配置（\(b=5\times10^5,L=4096,M=32768\)）下 \(\phi_L=0.494\)、\(\phi_M=0.652\)
都在这个上界右侧；\(\tau=2\) 时三区质量从 \(0.494/0.158/0.348\) 变为
\(0.671/0.122/0.208\)，64-pair 表从 \(32/10/22\) 变为 \(43/8/13\) 通道。

**这是一个 exact budget 陈述**，并给出可证伪预测：
**thinning 中间波长区的分配，应该在「同时多参考解析」上付出比「单参考检索」更大的代价**。
现有证据方向一致：named NIAH 家族的 post-hoc 拆分中，三个 single-needle 家族
4K 平均掉 \(-21.7\) 个点，五个 multi-key/value/query 家族掉 \(-52.3\) 个点。
（标注为 post-hoc split + 方向性一致，**不是** 预注册检验。）

### 4.2 τ 是 basin selector（现有写法自伤，须改）

现行 `03_theory.tex:216` 与 `04_experiments.tex` 都以「规则只在 4/12 里最好」开场。
这是**把自己最弱的读数放在最前面**。owner 里存在更强且同样为真的containment 陈述：
21 个配置上被选中的 \(\tau\) **从未离开 \(0.75\times\)–\(1.5\times\) 区间**
（owner：`PHASE16_99RUN_RAW_REANALYSIS_20260724.md`；该陈述已在 rebuttal 中发给 27bE）。

**改写方向**：先陈述 containment（规则总落在一个窄 basin 内），再给
「不是点最优（4/12）」作为精确边界。两句都为真，顺序决定读者印象。

---

## 5. 9 页正文的定理清单（建议）

主文最多放 4 条形式陈述，每条必须有一个明确的实验或概念后果。

| # | 陈述 | 非形式化一句话 | 后果 | 状态 |
| --- | --- | --- | --- | --- |
| **T1** | \(r_2(R)=2K/[1+(K-1)\bar c]\)（L3） | 有限表的可用位置维数由平均冗余精确决定 | 给 exact-range 实验（钉死 \(K\)、端点，只动 \(\bar c\)）一个先验动机 | exact proved |
| **T2** | 低频塌缩：\(V_\omega\to\operatorname{span}\{1,\Delta\}\)，softmax 下 \(\to\{\Delta,\Delta^2\}\) 中心化（L4） | 慢通道彼此重复，不是没用 | 说明预算该往哪搬；\(K=64\) 时 24 pairs 只值 ~2 维 | exact proved + 数值 |
| **T3** | Post-hoc transplant obstruction（L8） | 冻结的静态 Q/K 线性映射无法吸收非平凡换表 | 把 retrofit 的 in-window 代价变成被预测的结论 | exact proved |
| **T4** | \(\Capp\) 的唯一约束极小元为 cosh 密度，\(\tau\to0\) 退化为几何（现行 Thm 3.1） | 一个零参数闭式代表点 | 给出可复现的干预手段 | exact under stated surrogate |

**移入附录**：softmax-transport 的 \(\deff/\sqrt{\Ltr}\) proposition（保留结论一句话
+ 指针）、single-crossing lemma 的证明、\(\Capp\) 系数拟合、\(\mathcal C_{\rm ext}\) 的
双项结构讨论。

**正文必须新增的非形式化内容（不占定理位）**：
L6 的反例表（静态 rank 涨而 PPL 崩）与 L7 的交互项 \(-3.5367\)。
这两条是全文最能建立可信度的东西，且都是零 GPU 成本的已完成结果。

---

## 6. 建议的正文措辞（可直接用）

**关于 cosh 的地位**（替换现行「Cosh is a derived instance」段）：
> The empirical result identifies an axis, not a curve. Two structurally
> different pre-specified schedules — the \(1.25\times\) cosh member and a
> deformation-matched exponential — beat the uniform grid (10/12 and 9/12)
> configurations. Separately, formula-cosh and the matched exponential are not
> separated (\(+0.00074\) NLL, \(p=0.836\)). What the
> data identify is the direction in which a finite budget is moved. Cosh is the
> member of that direction that is available in closed form, adds no learned
> parameter, and degenerates exactly to geometric RoPE as \(\tau\to0\).

**关于静态几何与任务的分离**（新增，回应 `RDz6s.3`）：
> Static geometry constrains what a table can express; it does not predict what
> a trained model does with it. In a frozen 50M \(2\times2\), replacing the
> geometric table with \evq{} at inference *raises* the block-whitened stable
> rank from 4.57 to 12.54 while perplexity degrades from 7.14 to 76.20. We
> therefore treat identifiability as a necessary condition on the substrate and
> keep every task claim on trained models.

**关于 co-adaptation 与 retrofit 代价**（新增，回应 in-window 质疑）：
> The same factorial separates the two costs. Table and weight main effects on
> LM loss are \(+0.599\) and \(-0.597\); their interaction is \(-3.537\), about
> \(5.9\times\) larger. The two self-consistent systems are essentially tied
> in-window (7.14 vs 7.16); only the mismatched cells collapse. This is not a
> tuning failure: Theorem T3 shows no fixed, position-independent invertible
> Q/K map can absorb a nontrivial table change exactly.

---

## 7. 必须杀掉的说法

1. static collision / rank / logdet 改善 ⟹ 更好外推或更低 LM loss。（L6 反例）
2. cosh 是 full-RoPE 或 task-loss 最优。（只在 \(\Capp\) 下唯一）
3. 慢通道是 dead / unused / freely reclaimable。（只能说 redundant）
4. EVQ in-window 无损或有增益。（4K RULER 72.19→42.44）
5. EVQ ≈ LeRoPE 的闭式版；LeRoPE 验证了 EVQ；任一方 dominate。（无共同 benchmark）
6. multi-source RULER 拆分证明了排他机制。（post-hoc split）
7. \(\tau=\deff/\sqrt{\Ltr}\) 是最优选择器。（只是 basin selector）
8. 单一共享表在数学上不可能同时服务两个长度体制。（未证明）
9. 「2.205L 常数」被验证。（\(c\) 仍是待标定量）
10. NLL 改善 = 能力。（8B 32K PPL 3.71 vs 5.35 而 RULER 0/20 的 dissociation 先例）

---

## 8. 与 §3 现稿的差异（实施清单）

| 现稿 | 问题 | 动作 |
| --- | --- | --- |
| §3.2 以 \(K_{\cos}\) 开场 | 非相位不变，是 \(D=0\) 切片 | 前置 L1/L2，把 \(K_{\cos}\) 降为「本文分析的那个切片」 |
| Thm 3.1 是唯一定理 | 只有 construction theorem，撑不起 9 页理论 | 补 T1（\(r_2\) 恒等式）、T2（塌缩）、T3（obstruction） |
| 无 L6 反例 | 审稿人会自己发现 | 新增一段 + 4-cell 小表 |
| 无 co-adaptation | in-window 质疑无法回答 | 新增交互项段落 |
| §3.7 以「4/12」开场 | 自伤 | 改为 containment 优先 |
| §3.5 已移入附录 | ✅ 已在本轮完成 | — |

**页面预算现实**：正文已在 9 页硬顶（`compile.sh` 已按 ICLR 9 页门禁）。
以上每一条新增都必须有命名的替换对象。建议的替换来源：
§4.4 的 MLA/RAMP 段落进一步压缩（已移表入附录）、§2 的 context-range 段落再压两行、
§4.3 的 8B 段落合并。
