# Strong-model theory verdict on Q1–Q8 (EVQ-Cosh)

日期：2026-07-20
状态：`independent_adjudication`（对 `CORE_THEORY_QUESTIONS_FOR_STRONG_MODEL_20260720.md` 的逐条裁决）
关系：不重复 `FULL_PAPER_INTEGRITY_AUDIT_20260713.md` / `THEORY_REBUTTAL_MATHEMATICAL_AUDIT_20260711.md` / `EXPERIMENT_THEORY_REVIEW_20260720.md` 已承认的错误（T-01…T-07、P0.1…P0.7、c_coll、Phase16 重述等），只裁决 Q1–Q8 的残余问题。凡与提问文档预设冲突处，以本文的独立复算为准。
复算脚本：`rebuttal/strong_model_verdict_numerics_20260720.py`（纯 NumPy，全部数字可在 <1 min 内重现；kernel、fit、collision score、EVQ warp 均按论文自身定义重新实现，未 import 仓库代码）。

裁决用语：**成立** = 问题指认的缺陷属实且需要论文层面动作；**部分成立** = 核心指认属实但问题的某个前提/数字/推论被推翻；**不成立** = 指认不属实。

---

## 0. 总裁决表

| Q | 一句话裁决 | 判定 |
| --- | --- | --- |
| Q1 | 定理本身是真数学（存在唯一性+闭式），但作为 RoPE 论断是空的：surrogate 的 Green-kernel 结构使**任何** (α,β) 拟合都只能输出 cosh，拟合环节不可证伪；独立陈述的 exact-kernel 二次型的最小元是截断型密度而非 cosh | **成立**（认识论层面；定理须重标为 representation/solvability result） |
| Q2 | τ_surr=√(β/α)≈6.24/5.70 vs 部署 1.41/1.00（4.4×/5.7×，随 L 增大）；‖ρ_surr−ρ_dep‖₁=0.90/0.93；不存在能把 √(β/α) 变成 d/√L 的 gauge——两个指数（d 与 L）都不同，规范只能吸收 O(1)；24–92% 表确认是在**部署 τ** 上算的 | **成立** |
| Q3 | 推导内部 a=1 确是 normalization pairing 的产物（三种可辩护 pairing 给出 d⁰/d^½/d¹）；但问题问的 normalization-independent observable **存在**：Phase16 本身就是 fixed-L 的 d-sweep，其窗口内粗略偏向 d¹、排除 √d | **部分成立** |
| Q4 | balance 只有当 U 是 ρ 的线性泛函（O(τ²)）才成立；KL 读法下两项同为 O(τ⁴)，stationarity 退化。正文 03_theory.tex:93/108 的 "KL gain" 命名仍在稿中。且诚实定义下 1/L 因子本身也是 proxy 读法选择（total-variance 读法下 L 依赖完全消失） | **成立** |
| Q5 | (a) E_off 无符号、无距离分辨，仓库自身数据证明 collision 最优点（τ≈13, c≈4.6）远离 PPL basin；(b) w≡1 时 "shaping 降低 ∫w/ρ²" 为假（Jensen；∫1/ρ²=1.004→11.6→113 随 τ 严格上升）——但问题 (ii) 所要的非循环权重**存在**：w=q(Lb^{−φ}) 时 cosh 在全部部署配置降低 ∫w/ρ²（0.32–0.87），只是该降低同样被一般单调密度共享 | **(a) 成立 (b) 部分成立** |
| Q6 | "三个不一致 kernel" 的前提有误：uniform prior 下 cos-product 积分**解析等于** sum+diff sinc 闭式，(i)≡(ii)，验证表确按含 sum 项的 kernel 重现；真实残余问题只有 content weights（并入 Q7），外加新发现：验证表 video 行在 K=16 网格上用 τ=1.5=0.53·16/√32（d_eff≠2K，表内未披露的 convention 混用） | **部分成立**（前提部分不成立） |
| Q7 | η_F 是唯一 model-dependent 项且被弃置、从未测量——属实；"controlled residual" 措辞不可保留；但问题的放大因子算错：sinh(4)/4=6.82（非 ≈13.6）；部署主档 τ=1.41 处仅 1.37，τ=5.66/8 处 25.3/186（该处 L¹→warp bound 空洞） | **部分成立** |
| Q8 | 同网格拟合+同 12 配置验证+非线性 C≠二次型+无 held-out——属实；控制实验杀伤力更强：最优指数单调密度在 12/12 配置达到 ≈100% 降低（≥EVQ），线性 tilt 达 43–92%，EVQ@τ_surr 达 97–100%>部署点；24–92% 只证明"把质量移出死通道会降低该冗余统计量"，对 cosh 形状与部署 τ 均无鉴别力 | **成立** |

---

## 1. Q1 — "cosh minimizes C_app" 的内容量

**判定：成立**（对认识论指认；"tautology" 一词需精确化）。

**推导。** 对任意 α>0, β≥0，C_app 的约束一阶变分是 αρ+βg+ν=0，g''=−ρ，两次求导必然给出 ρ''−(β/α)ρ=0；配合 mass 约束产生的边界条件 ρ'(0)=−τ²、ρ'(1)=0，正质量解族就是 {ρ_τ}。因此 **surrogate 族 {C_app(α,β)} 的最小元集合 = cosh 单参族本身**：无论把 (α,β) 拟合到什么 kernel、什么配置，输出必是某个 cosh。"我们拟合了 exact kernel，最小元是 cosh" 这一验证步骤的证据量为零——它不可能输出别的。定理的真实内容是：该二次型+Green-kernel 结构**可解**且解闭式、唯一、严格正（这是真数学，不是同义反复），但其中没有任何 RoPE 物理。

**实跑数字（独立目标下 cosh 是否再现）。** 把"独立陈述的目标"取为 exact kernel 的二次型 min_{ρ≥0, ∫ρ=1} ρᵀKρ（K = uniform-prior cos-product Gram，d=64 网格），projected-gradient 全局解（凸问题）：

- L=512：最优 ρ* 把 32 通道中 **13 个置零**，活跃带（φ≤log_b L=0.48）质量 0.93，形状为"活跃带近平台 ≈2.3–2.6 + 截断"，非单调；对 cosh 族最佳 L2 拟合误差 0.499（部署 τ 的 cosh 误差 0.538）。
- L=2048：**15 个通道置零**，活跃带质量 1.00，平台 ≈1.9；最佳 cosh 拟合误差 0.433。

即：exact 二次型自己选出的是**截断/均衡型**配置，不是 cosh。配合论文自证（a1_proofs.tex:111–117：把常数 α 换成 stationary-phase 的 φ-依赖对角，最小元变 Bessel 而非 cosh），结论明确：**选出 cosh 的是 surrogate 的 constant-diagonal + min-kernel 代数（为闭式可逆 CDF 而选），不是 RoPE。**

**"什么（如果有）从 RoPE 物理选出 cosh"：** 现有材料中没有任何东西。exact 二次型选截断；exact 非线性 collision score 选 τ≈13 的极端配置（§8）；变系数对角选 Bessel。cosh 的辩护只能是工程性的（闭式、可逆、正性、单参、经验 basin），不能是变分性的。

**论文动作。**
- `paper/sections/03_theory.tex:39`：定理名 "Exact stationary allocation under the broadband surrogate" 可留，但正文引用处不得再作为 "exact tier" 的 RoPE 论据；`paper/tables/table_epistemic_map.tex:11` 行 2 "Theoretical core" 须改。
- 最窄诚实替代（epistemic map 行 2 Role 栏）：
  > "Closed-form solvability/representation result: the Green-kernel surrogate admits a unique positive minimizer in closed form for every (α,β). Because every fit of (α,β) necessarily returns a member of this cosh family, agreement of the fitted family with cosh carries no evidential weight about RoPE; all RoPE-specific support must come from tests outside the surrogate."
- `03_theory.tex:32` 及 `a1_proofs.tex:117(iii)` 中"functionally validated by the 24–92% reduction"作为 cosh 形状的支持必须撤（见 Q8：该测试无形状鉴别力）。

---

## 2. Q2 — τ_surr 与 τ_deploy：convention 还是断裂

**判定：成立。**

**实跑数字（复算脚本 §A/§B，拟合协议 = 论文协议：α=mean(K_ii)·Δφ，β=off-diag 对 min 的 LS）：**

| 量 | L=2048 (d=64,b=500K) | L=4096 |
| --- | --- | --- |
| α（拟合） | 0.0218（∝1/d 精确成立：α·d=1.40 于 d∈{32,64,128}；对 "α≈1/d_rot" 的字面值偏 1.35–1.61 且随 L 漂移） | 0.0211 |
| β（拟合） | 0.849（全域拟合 β∼L^−0.221） | 0.685 |
| τ_surr=√(β/α) | **6.244** | **5.704** |
| τ_deploy=d/√L | 1.414 | 1.000 |
| 因子 | **4.42×** | **5.70×** |
| ρ(0)/ρ(1)=cosh τ | 257:1 vs 2.18:1 | 150:1 vs 1.54:1 |
| ‖ρ_surr−ρ_deploy‖₁ | **0.896** | **0.930** |

因子随 L 单调增大（d=64：1.36→5.70 across L=128→4096），在主实验档 L=2048–4096 最大；d=128,L=128 时比值≈0.96（曲线交叉），进一步说明两者只是偶然相交的两条不同标度律。直接拟合 τ_surr∼√d·L^−0.085（β 的 −0.22 折半后 −0.11，与直拟同量级）、τ_surr∼d^{0.50–0.52}（fixed L）。

**gauge 问题的答案：不存在这样的 convention。** K_app=αδ+βmin 在 (α,β)→(cα,cβ) 下最小元不变，物理量只有比值 β/α；该比值由"exact kernel 在网格上的对角自能 vs 累积重叠能之比"这一**可测泛函**钉死（对 δ 的离散化约定只影响 O(1) 前因子）。convention 能改前因子，不能改指数；而 √(β/α)∼d^{0.50}L^{−0.09..−0.11} 与 d¹L^{−0.5} 在**两个指数**上都不同。P0.4 已认指数不匹配；本文补上此前缺的三件事：(i) 因子 4.4–5.7× 且随 L 增大；(ii) 密度近乎不相交（L¹≈0.9，上限 2）；(iii) 部署点在 surrogate 自身目标下也远非最优——同一 collision 指标上 EVQ@τ_surr 的降低是 97–100%，部署 τ 只有 24–93%（§8 表）。

**"validation 表用的哪个 τ"：已确证是部署 τ。** 复算以 τ=d/√L（d=2K）逐行重现正文 8 个 text 配置（C_geo 229.9/196.3/164.3/134.5/108.9/86.0/33.3/78.4 vs 论文 226.3/192.6/163.2/133.9/109.0/87.0/34.8/79.4；C_evq 同精度），video 4 行在 τ≈1.5=0.53·16/√32 处逐行重现（14.0/28.3/41.2/48.5 vs 论文 14.2/28.0/39.8/47.2）。**因此 24–92% 表验证的是部署规则的 allocation，不是定理的最优点**；τ_surr 处降幅更大这一事实同时说明该表也不能反过来为部署 τ 背书。

**论文动作。**
- `a1_proofs.tex:315–323`（"What the surrogate predicts"）：保留已有让步，但须把 τ_surr 的数值断裂写实。替代表述：
  > "The fitted surrogate's own optimum is τ_surr=√(β/α)≈6.2 (L=2048) and 5.7 (L=4096) at d=64, b=500K, versus the deployed τ=1.41 and 1.00 — a factor 4.4–5.7× that grows with L, with ‖ρ_surr−ρ_deploy‖₁≈0.90–0.93. The deployed allocation is therefore not the surrogate optimum, and no rescaling convention connects the two: the exponent pairs (d^{1/2}, L^{−0.11}) and (d¹, L^{−1/2}) differ in both variables. Theorem 1 supplies the shape family only."
- `a1_proofs.tex:126–127`（表 caption）与 `03_theory.tex:15/32`：注明验证 τ 为部署值（text 行 τ=d_eff/√L，video 行 τ=0.53·16/√32），并撤去任何"validates the theorem/surrogate optimum"读法。

---

## 3. Q3 — d_head 因子是不是 convention

**判定：部分成立。** 推导侧指认成立；"不存在 normalization-independent observable"这一预设被推翻。

**推导侧（成立）。** τ*∝d^{(a+b)/2}L^{−b/2} 无误。a 的取值由 pairing 决定，且至少三种 pairing 都有故事：

| pairing | S 归一 | U 归一 | τ* 的 d 指数 |
| --- | --- | --- | --- |
| 论文（Prop.） | S=τ⁴/(45d)（把拟合 α∝1/d 移植进 stiffness；a1_proofs.tex:561 自认 "the d_head factor enters via the surrogate diagonal"） | U 通道可加 ∝(d/L) | **d¹** |
| surrogate 自身 | α∝1/d, β 无 d 依赖 | —（同一泛函内部） | **d^{1/2}**（复算 §A：d^0.50） |
| "双 extensive" | S 亦按通道求和 ∝d·τ⁴ | U∝(d/L) | **d⁰** |

同一篇论文在两个 tier 用了两种 pairing（√d 与 d¹），故"结构性 d_head 因子"在推导内部确是 convention；`a1_proofs.tex:320` "This correctly captures the d_head dependence"（指 √d！）与 Prop 的 d¹ 在正文里并存而未标注冲突。

**观测侧（推翻问题预设）。** Phase16 manifest 本身就是问题所要求的"controlled d-sweep at fixed L"：d∈{32,64,128}×L∈{256,512,1024}，每配置的 τ 网格 = {0, 0.75, 1, 1.25, 1.5}×(d/√L)。复算（extrapolation-length mean log-PPL，seeds 取均值）得最优倍数 c：

```
L=256 : d=32 c=1.00  d=64 c=1.00  d=128 c=1.25
L=512 : d=32 c=1.25  d=64 c=0.75  d=128 c=1.00
L=1024: d=32 c=1.25  d=64 c=1.25  d=128 c=1.25
naive d-exponent: 1.16 / 0.84 / 1.00 (per L)；naive L-exponent: −0.34 / −0.34 / −0.50 (per d)
```

若真值是 τ∝√d（a=0），锚定 d=64 后应有 c(32)≈1.41、c(128)≈0.71——即 d=128 应被压在网格下缘 0.75。观测相反：**三个 L 行里 c(128)≥c(32) 无一例外**。故在窗口 [0.75,1.5] 与本 metric 的限度内，d¹ 被粗略支持、√d 被排除。必须带的 caveat：网格按假设本身缩放（只能在 2 倍窗口内鉴别）、n=1 pilot 行与 n=3 行混合、log-PPL 差仅 0.01–0.05、metric 是本文简化版而非 runner 复合分。另外 collision 侧完全帮不上忙：argmin_τ C 在 d=32/64/128 处为 13.5/13.1/10.8（近乎无 d 依赖，c=9.6/4.6/1.9）——exact-collision tier 既不支持 d¹ 也不支持 √d。

**论文动作。**
- `03_theory.tex:113`："\emph{derives the structural} d_head factor and L^{−1/2} exponent" 必须改写。最窄诚实替代：
  > "Within the stated proxy normalization (per-channel-extensive utility, α-normalized stiffness), the balance yields τ*∝d_head/√L; the d-exponent depends on this normalization pairing (alternative defensible pairings give d⁰ or d^{1/2}), while the L^{−1/2} exponent follows from the diffuse-softmax displacement normalization. Empirically, the fixed-L d-sweep (d∈{32,64,128}) is consistent with the d¹ pairing and inconsistent with d^{1/2} within its 0.75–1.5× search window."
- `a1_proofs.tex:320`：在 "correctly captures the d_head dependence" 处加注两 tier 的 d 指数互相矛盾（√d vs d¹），或删去 "correctly"。

---

## 4. Q4 — U 的身份与 balance 的内部一致性

**判定：成立。**（T-02 认了 order/mislabel；此处裁决的是残余结构依赖，并确认稿件文本未改。）

**现状确认。** `03_theory.tex:93` 仍写 "U(τ,L) the per-channel post-softmax KL gain"；`03_theory.tex:108` 仍写 "balances an O(τ²) utility gain (RoPE logit perturbation transported through a diffuse softmax with 1/L Jacobian curvature)"——括号里描述的对象（logit 扰动经 softmax Jacobian 曲率）正是 O(τ⁴) 的 KL 曲率量，与句首 O(τ²) 自相矛盾。

**结构依赖（推导）。** 设 θ=τ²。(i) 若 U=KL：D_KL(p₀‖p_θ)=½θ²gᵀJg+O(θ³)（∇A(z₀)=p₀ ⇒ 一阶变分恒零）。此时 F=θ²[1/(90d)−λκ/L]+O(θ³)，stationarity 只区分 θ=0 稳定/失稳，**不产生内点 τ*²∝d²/L**。(ii) 只有 U 为 ρ 的线性泛函 U=(M/L)∫qρ（沿 ρ_τ=1+θη+O(θ²) 一阶变化 (M/L)Q₁θ ≠0）时，τ⁴-vs-τ² 平衡才给出 τ*²=45λQ₁d²/L。数值确证（脚本 §F/§Part2-1）：schedule-KL 斜率 θ^1.84（→O(τ^3.7)，即 O(τ⁴) 加高阶污染）；线性 allocation score 斜率 θ^0.99（→O(τ²)）；Q₁(512,500K)=0.03192 与论文一致，Q₁ 全表 {128:0.0301, 512:0.0319, 2048:0.0305, 8192:0.0265}(b=500K)、b=10K:{2048:0.0195, 4096:0.0141, 8192:0.0083}、video b=100/L=32:0.0273——Prop 声称的 Q₁∈[0.008,0.032] **确证**（低端即 b=10K,L=8192 的 0.0083），Q₁>0 在测试网格成立但非 L-常数。

**对 (a)(b) 的回答。**
- (a) U 的绝对归一（b 指数）**不被诚实定义钉死**：同一 pattern c_ω 有三种自然读法——total variance Var_{p₀}[c_ω]=q（O(1)，**无 1/L**）、Euclidean displacement ‖J(p₀)c_ω‖²=q/L、per-position Fisher (1/L)c_ωᵀJc_ω=q/L。取 total-variance 读法则 U∝M·Q₁θ，balance 变 τ*²∝d^{a+1}·L⁰——**L 依赖整体消失**。故不仅 d 因子（Q3），连 (d/L) 的 1/L 也属于 proxy 读法选择；它有一个"自然"辩护（概率位移能量/逐位置归一），但不是唯一。
- (b) 1/L 的诚实读法：**diffuse softmax 的 Jacobian 特征值经由位移-能量归一化进入一次**（J(p₀)=P/L，‖Jc‖²=q/L），而非"KL Taylor 曲率的 ½ε² 因子"。`03_theory.tex:93` "convention ½ factors from the KL Taylor expansion are absorbed into λ" 这半句必须删——λ 吸收的是位移归一的因子，与 KL 展开无关。
- (c) 重写后的 balance（最窄诚实版，替换 `03_theory.tex:95–109` Prop 叙述）：
  > "Define the diffuse transport score U_tr(ρ;L)=(M/L)∫₀¹q(Lb^{−φ})ρ(φ)dφ, the channel-sum of squared probability displacements ‖J(p₀)c_ω‖²=q(ωL)/L at the uniform baseline. U_tr is linear in ρ, so along ρ_τ=1+τ²η+O(τ⁴) it has the nonzero first-order change (M/L)Q₁τ². Balancing this against the O(τ⁴) Pearson stiffness yields τ*²=45λQ₁M²/L. The interior optimum exists only because U_tr is linear in ρ; ordinary baseline-to-perturbed KL is O(τ⁴) and yields no such point. The 1/L factor is the displacement-energy normalization of the diffuse softmax Jacobian, not a KL curvature; under the alternative total-variance normalization the L-dependence disappears, so the L^{−1/2} exponent is conditional on this normalization choice."

---

## 5. Q5 — collision 机制的符号问题与 ∫w/ρ² 断言

**判定：(a) 成立；(b) 部分成立**（论文句子为假，但问题 (ii) 所要的非循环权重存在——结论方向与提问预设相反）。

**(a) 成立。** E_off=Σ_{i<j}K_ij²/(K_iiK_jj) 对所有位置对一致惩罚相干性，无符号、无距离分辨。这不只是概念批评，仓库数据可直接证明它与训练目标脱钩：argmin_τ E_off≈13.1（c≈4.63，E_off=0.042 vs 部署点 41.8，d=64/L=512），而 PPL basin 在 c≈1（Phase16）；τ_surr 处 collision 降幅 97–100% 仍无人部署。"从 LM loss 导出符号/距离权重"在仓库内不可得：一阶项 E[∂L/∂z_ij·δz_ij] 需要 trained model 的梯度测量（与 THEORY_REBUTTAL §11.2-1 同一未做实验）。诚实定位：E_off 只是 redundancy diagnostic（EXPERIMENT_THEORY_REVIEW §1.3(2) 的重framing 是正确方向），不得写成"LM-favorable 的机制"。

**(b) 部分成立。**
- 论文句为假（w 未指明时按 w≡1 读）：Jensen/Cauchy–Schwarz 给 ∫1/ρ²≥(∫1/ρ)²≥1=∫w·1²，等号 iff ρ≡1。闭式 ∫1/ρ_τ²=sinh²τ·tanhτ/τ³ 复算：τ=0.5/1/1.414/2/4/5.657 → **1.0039 / 1.0518 / 1.1760 / 1.5851 / 11.63 / 113.2**（问题引用的 1.004/1.05/1.59/11.6 全部确证）。shaping 严格**增加**未加权量化失真。
- 但问题 (ii)"是否存在非循环的 RoPE-derived w 使 cosh 降低 ∫w/ρ²"答案是**有**：取 w(φ)=q(Lb^{−φ})（相位方差权重：只依赖 (L,b)，不依赖 ρ，也不是对 cosh 拟合出来的——它正是 transport proxy 已有的权重）。复算（脚本 §E）：

| 配置 | ∫q/ρ² | ∫q（uniform） | 比值 |
| --- | --- | --- | --- |
| L=2048, b=500K, τ=1.414 | 0.2009 | 0.2627 | **0.765** |
| L=4096, b=500K, τ=1.0 | 0.2510 | 0.2891 | **0.868** |
| L=512, b=500K, τ=2.828 | 0.1043 | 0.2099 | **0.497** |
| L=128, b=500K, τ=5.657 | 0.0499 | 0.1571 | **0.318** |
| L=32, b=10K, τ=1.4 | 0.0871 | 0.1486 | **0.586** |

  全部部署配置下降。**限定**：该下降不是 cosh 专属——线性 tilt ρ=1+1.9(½−φ) 在 L=2048 得 0.1427（比 cosh 更低）；且 w=q 继承 diffuse-uniform 位置先验（非循环于 ρ，但 proxy-laden）。

**论文动作。**
- `a1_proofs.tex:626`："a shaped ρ reduces the weighted inverse-density load ∫w/ρ²" 必须改写。最窄诚实替代：
  > "For the unweighted load (w≡1), shaping strictly increases ∫1/ρ² (Jensen; equality iff ρ≡1): the values are 1.004/1.05/1.59/11.6 at τ=0.5/1/2/4. For the phase-variance weight w(φ)=q(Lb^{−φ}) — fixed by (L,b) and independent of ρ — the cosh density does reduce ∫w/ρ² below uniform at all deployed configurations (ratios 0.32–0.87), but so do generic monotone reallocations; the reduction reflects moving cells toward phase-resolving channels, not a cosh-specific property, and inherits the uniform-position-prior assumption."
- `a1_proofs.tex:425–451`（mechanism isolation）与所有引用 E_off 处：加"unsigned redundancy diagnostic; its minimizer (τ≈13, c≈4.6) lies far outside the trained-PPL basin"一句（可直接采用 07-20 review §1.3(2) 的表述）。

---

## 6. Q6 — 三个 "exact kernel" 是否互相矛盾

**判定：部分成立（关键前提不成立）。**

**前提修正。** (i) 与 (ii) 不是两个 kernel：对 uniform prior D=1/L on [0,L]，∫D(Δ)cos(ω₁Δ)cos(ω₂Δ)dΔ = (1/2L)[sin((ω₁−ω₂)L)/(ω₁−ω₂) + sin((ω₁+ω₂)L)/(ω₁+ω₂)] 是**同一对象的展开**，sum-frequency 项自动包含。复算用含 sum 项的闭式逐行重现验证表 text 配置（§2），证明验证与拟合用的是同一 kernel。"fit 与 validation 的 kernel 不一致"不成立，无需 reconcile。（重现精度：C_geo 偏差 ≤4%，C_evq 偏差 ≤8%（多数 ≤4%）；残差量级与网格/离散化约定差异（midpoint vs endpoint、离散 Δ 求和 vs 连续积分）一致，不影响 τ 与 kernel 的判定——若表用 τ_surr，config 0 的 C_evq 应为 ≈4.6 而非 17.7。）

**真实残余（成立部分）。**
1. **Content weights 缺席**：(i)/(ii) 都是 content-free 的。真 RoPE Gram z_ij=ReΣ_k α_{ij,k}e^{ir ω_k} 的权重 |α_{ij,k}|² 只有当 η_F(φ)≈const 时才从 allocation 问题中因子化掉；而论文自己的 Fisher forcing 项 γb^{−2φ}η_F(φ)（a1_proofs.tex:511–521）就是"它不因子化"的形式化——content weights 蕴含的 allocation 修正**正是被丢掉的 forced branch**。此问题并入 Q7 处理，不构成独立缺陷。
2. **新发现（本文复算得出，此前审计未记录）**：验证表 video 4 行（K=16, L=32）不是在 τ=d_rot/√L=5.657 或 0.53×(2K)/√L=3.0 处计算，而是在 **τ≈1.5=0.53·16/√32** 处逐行重现（14.0/28.3/41.2/48.5 vs 论文 14.2/28.0/39.8/47.2）。即同一张表内 text 行用 d_eff=2K、video 行用 d_eff=K（=d_head=16，配 K=16 网格），d_eff 与通道数的搭配在表内不一致且未披露——T-06/P0.7 的 d_eff ambiguity 在纯数值验证表里再次出现。

**论文动作。**
- `a1_proofs.tex:124`（"For a uniform distance prior..."段）与表 caption：披露每行使用的 τ（text: d_eff=2K 的部署规则；video: 0.53·d_head/√L 且 d_head=16≠2K）。
- content-weight 空缺随 Q7 的重定位一并披露，不必新增第三条 kernel 叙述。

---

## 7. Q7 — Fisher forcing：唯一的模型耦合项被弃置

**判定：部分成立。**

**成立部分。** (i) η_F（a1_proofs.tex:511–515）确是理论中唯一进入 allocation 的 trained-model 统计量；homogeneous 分支只依赖 kernel 几何拟合的 τ，模型无关。"exact minimizer 与它所部署的模型无关，恰因唯一 model-dependent 项被丢弃"——结构描述正确。(ii) η_F 从未在任何 checkpoint 上测量：仓库内无脚本、无 artifact（`scripts/analysis/compute_eta_vp.py` 是 video-DiT 噪声调度的 η，与 η_F 无关）；因此 λ_F·η̄_F/α 未知，R_F 与 warp 修正在部署 τ 处**无法评估**，"controlled residual" 无从谈起（与 P1.3 一致，且正文 `03_theory.tex:48`、`a1_proofs.tex:72` 的 "controlled but nonzero residual" 措辞仍在稿中）。(iii) `a1_proofs.tex:122` 把 24–92% 表称为 "operational forced-branch residual diagnostic" 不成立：该表从未评估 forced branch，且 §8 显示任何单调 reallocation 都能通过该测试——它对 forcing 残差无诊断力。

**推翻部分（数字）。** 问题断言 "at τ=4 this amplification is ≈13.6"：错。放大因子是 1/inf ρ_τ = sinhτ/τ（a1_proofs.tex:536–541），复算：

| τ | 1.0 | 1.414 | 2.828 | 4.0 | 5.657 | 8.0 |
| --- | --- | --- | --- | --- | --- | --- |
| sinhτ/τ | 1.18 | **1.37** | 2.98 | **6.82** | 25.3 | 186 |

13.6=sinh(4)/2 是误算。修正后的定量结论反而更有层次：主实验档（τ=1.0–1.41）放大仅 1.2–1.4，L¹ bound 在那里并不空洞；τ=4（PE-dominant）6.8×；Phase16 pilot 的 τ=5.66/8 处 25×/186×——bound 完全失效。共振 τ=2log b 从未接近（τ/(2ln500K)≤0.31）。

**(b) 重定位——支持。** 诚实版本恰好更简洁：EVQ-Cosh 是 model-free 的闭式 allocation 族；Fisher forcing 是一个**从未测量**的弃置修正，不是 justification。η_F 测量协议（checkpoint 上按 eq:eta-F 求 E[w|α|²]，代入 γ(b)、R_F）应列为 future measurement，非本轮承诺。

**论文动作。**
- `03_theory.tex:48` 与 `a1_proofs.tex:72`：删 "controlled"。替代：
  > "…the forced branch is an unmeasured residual: its amplitude depends on the activation-conditioned coefficient η_F, which we do not measure; the L¹/CDF bound controls mass transport but is amplified by sinh τ/τ (1.4 at τ=1.41, 6.8 at τ=4, 186 at τ=8) under inverse-CDF inversion, so no pointwise control is claimed at deployed τ."
- `a1_proofs.tex:122` 最后一句（"also serves as the operational forced-branch residual diagnostic…"）整句删除。
- `table_epistemic_map.tex:12` 行 3：“forcing branch is CDF/L¹-suppressed at typical bases” 后补 "amplitude unmeasured (η_F never estimated)"。

---

## 8. Q8 — 24–92% 验证的鉴别力

**判定：成立。**

**确证的事实链（脚本 §C/§C2/§D）。**
1. 协议同网格：α,β 拟合与验证共用同一 12 配置、同一离散网格、同一 kernel（§6）；被优化的二次型 ⟨ρ,K_app ρ⟩ 与被报告的 C=ΣK_ij²/(K_iiK_jj) 是不同泛函（后者非线性）。
2. τ=部署值（§2 已证），表内未披露。
3. **控制实验（本文新增，杀伤性最强）**：在完全相同的 12 配置、相同 exact kernel 上——

| 配置(前8=text) | EVQ@部署τ | EVQ@τ_surr | 最优线性tilt | 最优指数密度 |
| --- | --- | --- | --- | --- |
| K32 L128 | 93% | 98% | 51% | **99%** |
| K32 L256 | 87% | 99% | 58% | **100%** |
| K32 L512 | 75% | 99% | 65% | **100%** |
| K32 L1024 | 57% | 100% | 71% | **100%** |
| K32 L2048 | 39% | 100% | 76% | **100%** |
| K32 L4096 | 24% | 100% | 82% | **100%** |
| K32 L2048 b10K | 46% | 97% | 92% | **100%** |
| K32 L2048 b100K | 42% | 100% | 82% | **100%** |
| video 4 行 | 81–90% | 82–89% | 43–63% | 82–94% |

   单参指数密度族（非 cosh）在 12/12 配置匹配或超过 EVQ；EVQ 在 τ_surr 处全面高于部署点；collision-only argmin（c=4.6–9.6）更远超两者。
4. Held-out prior 抽查（指数距离先验，均值 L/4）：EVQ@部署τ 仍降低 22–87%——方向可迁移，但迁移的是"把质量移出死通道"这一泛性质，不是 cosh 形状。

**结论。** 24–92% 所确立的全部内容是：**在拟合所用的同一 kernel 与配置上，部署的 EVQ allocation 相对 midpoint-Geo 降低了一个无符号冗余统计量；该性质被几乎任意的低频减载 reallocation 共享，且在 cosh 族内部也偏好远离部署点的 τ**。它不鉴别 cosh vs 控制族、不鉴别部署 τ vs τ_surr、不验证 surrogate 拟合值、更不连接 trained objective（collision 最优点远在 PPL basin 外）。作为"surrogate quality: functional validation"（a1_proofs.tex:119 标题）它名不副实。

**论文动作。**
- `a1_proofs.tex:119–158` 小节改名并限缩（如 "Directional collision diagnostic at the deployed allocation"）；caption 加：
  > "This diagnostic is computed at the deployed τ on the same kernel and configurations used to fit (α,β). It is not shape-discriminating: a one-parameter exponential reallocation matches or exceeds these reductions on all 12 configurations, EVQ at the surrogate's own τ_surr achieves 97–100%, and the collision-only optimum lies at c≈4.6–9.6. We report it only as evidence that the deployed allocation moves in the redundancy-reducing direction, not as validation of the cosh shape, the fitted surrogate, or the deployed τ."
- `03_theory.tex:15/32`、`a1_proofs.tex:117(iii)/106` 所有以 24–92% 作为"functional validation of the surrogate/cosh"的引用同步降级为 "directional diagnostic (not shape- or τ-discriminating; see App. …)"。

---

## 9. 全局结论

### 9.1 撤回/改写后理论还剩什么

三层结构仍在，但每层的身份都比提交稿窄一档：

1. **精确层（保留，重标）**：给定 C_app 的凸性、唯一正最小元、闭式 CDF/逆 CDF、τ→0 geometric 极限、self-consistency 恒等式、waterbed 不等式（allocation-divergence 层面）、K^{-1}/K^{-2} 量化界。身份：**关于一个为可解性而选的 surrogate 的 representation results**；对 RoPE 的鉴别力为零（Q1：任何拟合只能输出 cosh；exact 二次型选截断型；变系数对角选 Bessel）。
2. **条件 proxy 层（保留，条件全部显式）**：U_tr=(M/L)∫qρ 线性 ⇒ 一阶变化非零 ⇒ τ*²=45λQ₁M²/L。三个 normalization 旋钮全部要披露：stiffness 的 1/d（a=1 pairing）、utility 的通道可加 M∝d、位移-能量归一带来的 1/L（换 total-variance 读法则 L 依赖消失）。它是一个 **proxy-objective 下的自洽 stationarity 计算**，不是 KL 定理、不是 task 定理。
3. **经验层（这次审计后反而更清晰）**：Phase16 的 9 配置窗口内——τ=0 在 9/9 配置都不是最优（geo 被支配）；最优倍数 c 全部落在 [0.75,1.5]；fixed-L d-sweep 粗略支持 d¹、排除 √d（窗口内）；L 指数 −0.34…−0.50。加上先前审计保留的 matched midpoint contrasts 与 MLA scarce-channel 方向性结果。

被本轮进一步钉死、不得再写的：cosh 的 RoPE 变分辩护；τ_surr 与部署 τ 的任何"O(1) convention"弥合；"structural d_head factor"的无条件表述；U 的 KL 命名与 "½ factors absorbed into λ"；24–92% 作为 surrogate/shape/τ 的验证；"controlled forcing residual" 与"validation 表即 residual diagnostic"；∫w/ρ² 无指明 w 的降低断言。

### 9.2 τ=d/√L 最强能辩护到什么程度

最强的诚实表述（一段话版本）：

> τ=d_head/√L is an operating-point selector with three independent supports and three disclosed limitations. Supports: (i) a conditional stationarity calculation — linear diffuse-transport score against Pearson stiffness — yields τ*∝d/√L under explicitly stated normalization choices; (ii) in the 9-configuration trained sweep the rule's multiple lies in [0.75,1.5] everywhere, τ=0 is never optimal, and the fixed-L d-sweep is consistent with the d¹ pairing while rejecting d^{1/2} within the search window; (iii) the L-exponent is empirically −0.3…−0.5 on the tested grid. Limitations: (i) the exact tier does not support it — the fitted surrogate's own optimum is √d·L^{−0.11}, a 4.4–5.7× larger τ with near-disjoint density, and the exact-kernel collision optimum (c≈4.6–9.6) has no d/√L structure at all; (ii) both the d-factor and the 1/L factor are normalization-dependent within the proxy; (iii) all trained evidence lives in a ≤2× search window at ≤1024 context, with the L≥16K falsification test unrun.

即：**可以辩护为 "proxy-motivated, coarsely d-and-L-consistent empirical default"；不能辩护为任何层面的 derived optimum。** 相对 07-11/07-13 审计的净变化：经验侧略强（d-sweep 窗口证据此前未被利用），理论侧更弱（validation 表被证明无鉴别力、τ_surr 断裂被定量化、1/L 的 convention 依赖被显式化）。

### 9.3 复算记录

脚本：`rebuttal/strong_model_verdict_numerics_20260720.py`（Part 1 §A–J + Part 2 follow-ups）。关键输出：α·d=1.40（∝1/d 精确）、β∼L^−0.221、τ_surr=6.244/5.704、因子 4.42/5.70、L¹=0.896/0.930、验证表 text 行在部署 τ 重现（偏差 ≤8%，多数 ≤4%）、video 行在 τ=1.5 重现、指数密度控制 12/12 ≥EVQ、EVQ@τ_surr 97–100%、held-out prior 22–87%、argmin C=13.5/13.1/10.8/13.5（d32/64/128@L512, d64@L2048）、∫1/ρ²=1.0039/1.0518/1.1760/1.5851/11.63/113.2、∫q/ρ² 比值 0.32–0.87、KL 斜率 θ^1.84、线性 score 斜率 θ^0.99、Q₁ 表、Phase16 c-矩阵与 naive 指数、sinhτ/τ=1.18/1.37/2.98/6.82/25.3/186、exact-QF 最小元 13–15 零通道/活跃带质量 0.93–1.00/最佳 cosh 拟合 L2 err 0.43–0.50、cosh 族内 exact-QF argmin τ=5.3/4.5 vs τ_surr=7.0/6.2 vs 部署 2.8/1.4。

对提问文档的四处修正：Q3 的 observable 存在（Phase16 d-sweep）；Q5(ii) 的非循环 w 存在（w=q，但非 cosh 专属）；Q6 的 kernel (i)≡(ii)（前提错误）；Q7 的 13.6 应为 6.82。
