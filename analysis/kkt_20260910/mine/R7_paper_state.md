# R7 — 论文现状（paper-2027）与 KKT 推导的落点

日期：2026-09-10。角色：挖掘与整理（只读）。本文件只描述**已读到的文本事实**，每条带 `文件:行号` 与证据等级。
论文侧一律**只读**，未改动任何 `.tex`；未写入 `~/.codex`。

证据等级：[已验证] = 本轮直接读到源文件行；[部分证据] = 有读数但范围/复现受限；[假设] = 机制解释或未完成计划；[叙事-未验证] = 文档自述、无独立支撑。

---

## 0. 覆盖度

**完整读过**（Read 全文）：`paper-2027/README.md`、`REVISION_BRIEF.md`、`NARRATIVE_GUIDE.md`、`HANDOFF.md`、`CHANGES_FROM_NEURIPS2026.md`、`main.tex`、`SUBMISSION_CHECKLIST.md`；`sections/00_abstract.tex`、`01_intro.tex`、`02_exponents.tex`、`02_identification.tex`、`03_findings.tex`、`03_theory.tex`、`04_experiments.tex`、`04_mature.tex`、`02_related.tex`、`05_discussion.tex`；`sections/budget_{abstract,intro,theory,method}.tex`；`appendix/a1_proofs.tex`、`a7_exponent_adjustments.tex`、`budget_proofs.tex`；`tables/table_allocation_protocols.tex`；`research/EXPONENT_CLAIM_EVIDENCE_MAP_20260909.md`、`EXPONENT_FOLLOWUP_QUESTIONS_20260909.md`、`EXPONENT_REVISION_REPORT_20260909.md`、`pdf-review-rounds/20260909/{README.md,r04/review.md,package_verification.json,r0*/identity.json}`。
**部分读过**：`appendix/a6_mature_scale.tex`(1–240 行，余下 240–326 未读)、`DOCUMENT_TEXT_MAP.md`（只 grep 结构，未逐行）。
**只做结构扫描**：`sections/*` 与 `appendix/*` 的 `\section/\subsection/定理环境` grep；`scripts/`、`figs/*.py` 未读。
**权威文档（对照用）**：`analysis/unify_20260910/{INTEGRATION_20260910.md, NEXT_DERIVATION_KKT_PROBLEM.md, STARTING_POINT_YARN_VS_MRPRO.md}` 全文；`digests/digest_paper-state.md`（前序 digest，284 行，全文）。
**未读/跳过**：`main.pdf` 全文（只取 pdfinfo + pdftotext 关键页）、`exponent-allocation-source.zip`、`rope-spectral-budget-iclr2027-supplement.zip`、`research/` 下其余约 15 份 md、`venue_icml_fallback/`、`refs/references.bib` 逐条。

---

## 1. 论文当前核心叙事

标题 **Beyond the Base: Exponent Allocation in RoPE**（`main.tex:74`；`README.md:3`）。

一句话叙事（`README.md:3-13`、`REVISION_BRIEF.md:6-13`）：
> 研究 **指数分布** `omega = b^(-phi)`（而非 base 标量）如何改变有限位置表示与长上下文建模；用**固定范围配对训练**、**范围重设**、**权重×运行表交叉** 三个受控发现开路，再给完整 sin/cos 子空间几何，再给 Cosh 显式构造，最后给冻结模型的 native-relative 位移调整。

三问（`sections/01_intro.tex:12-16`）：(i) 重分配指数时位置基变了什么；(ii) 频率范围固定时该重分配是否影响习得行为；(iii) 已训练模型如何响应其 native 指数的调整。

贡献三条（`sections/01_intro.tex:58-63`）：(i) 定义 exponent allocation 并分析 full-subspace 位置几何；(ii) 用 fixed-range / cross-shape / weights-by-table 控制识别其效应，并在真实模型评估闭式构造；(iii) 发展并测试冻结模型的 native-relative 调整，含 boundary-matched intermediate-band 规则。

叙事纪律（`NARRATIVE_GUIDE.md:9-14, 24-25`）：`x = a + R z` 只作**定义与控制**，"改坐标本身不是研究结果"；长推导放附录；"用数学操作/训练阶段/发现来比较相关工作"。

> **关键观察 [已验证]**：当前活跃稿**完全不含** KKT / 水床（waterbed）/ 守恒 / 三段（three-band）框架。对 `sections/` + `appendix/` + `main.tex` 全文 grep：`waterbed` 0 命中、`conserv` 0 命中、`守恒` 0 命中、`three-band` 0 命中；`KKT` 仅 1 处命中且是**顺带一提**（`appendix/a1_proofs.tex:356`："the unconstrained Euler--Lagrange solution coincides with the KKT-constrained solution"）；`Lagrange` 仅 `a1_proofs.tex:322`；`optimality` 仅出现在"不主张最优"的否定句（`sections/budget_method.tex:15`、`sections/budget_intro.tex:32`）与一个脚本名（`a1_proofs.tex:510`）。→ **KKT 推导在论文里是 100% 空白。**

---

## 2. Claim 清单（正文实际声明）

### 2.1 摘要（`sections/00_abstract.tex:1-16`）

| # | Claim | 等级 | 出处 |
|---|---|---|---|
| A1 | 固定端点配对训练，只改内部指数，在 3 seed × 所有 3 个测试外推长度上全改善 | [已验证]（源：`02_identification.tex:11-16`） | `00_abstract.tex:4-5` |
| A2 | 重设范围会改变排序 | [已验证] | `00_abstract.tex:6`；`02_identification.tex:18-24` |
| A3 | 冻结权重偏好与其训练分配相容的 runtime 表 | [已验证] | `00_abstract.tex:6-8`；`03_findings.tex:14-20` |
| A4 | full sine–cosine 子空间分析刻画指数分配如何改变位置基，含慢频二维共享极限 | [已验证]（数学） | `00_abstract.tex:8-9`；`03_theory.tex:46-52` |
| A5 | 从**指定凸准则**推出解析 Cosh 族并在训练/适配中评估 | [已验证]（数学，范围限定） | `00_abstract.tex:9-11` |
| A6 | "After matched Llama-3-8B adaptation at 8K, 32K perplexity falls from **991.5 to 127.9**" | [已验证]（读数=表值） | `00_abstract.tex:12`；`04_experiments.tex:54-55` |
| A7 | "A boundary-matched adjustment improves OLMo's mean token F1 over MrRoPE-Pro from **21.62% to 25.44%** across five natural-QA tasks" | [已验证] | `00_abstract.tex:14`；`04_mature.tex:120-121` |

摘要**不含任何 128K / Qwen-3B 数字** [已验证，逐字核对 `00_abstract.tex`]，也无任何最优性/普适性措辞。

### 2.2 受控发现（`sections/02_identification.tex`）

- 151.9M 配对、L=256、K=32、499,974,144 tokens、Geo 臂 `b = L_train`（FMRoPE 规则）、Cosh 臂 τ=4 端点锚定、**仅 30 个内部指数不同**（`:4-9`）。Cosh−Geo NLL @256/512/1024/2048 = **+0.026 / −0.281 / −0.176 / −0.146**（`:11-16`）[已验证]。
- 重设范围后反转：Cosh−Geo @512/1K/2K = **+0.060 / +0.227 / +0.460**（`:18-24`）[已验证]。
- 50.9M factorial：参考强度 7/12 改善、1.25× 强度 10/12、matched exponential 9/12；Cosh−exp 均值差 **+0.00074**，区间 **[−0.006, 0.008]**（`:28-38`）[已验证]。措辞已明确"参数间不确定性"。

### 2.3 权重×表交叉（`sections/03_findings.tex:12-20`）

50M：Geo 权重换 Cosh 表 PPL **7.14→76.20**；反向 **7.16→23.05**。151.9M 两 seed 复现偏好反转。

### 2.4 模型效应（`sections/04_experiments.tex`）

| 设置 | 数字 | 出处 |
|---|---|---|
| 432M MLA（16 对，500M tok，8K 训，3 seed） | 16K PPL **138.8→95.6**；8K **35.4/35.8** | `:14-19`，表 `:41-42` |
| 750M 续训（4K 训） | 16K PPL **45.1→24.4**；4K 22.0/22.3；8K passkey AR exact **0→77.5%**（40 trials） | `:22-27`，表 `:43-44` |
| Llama-3-8B matched LoRA（8K/300 步/r64） | PPL Native/EVQ = 6.82/10.07 @8K、**108.96/24.07** @16K、**991.48/127.91** @32K | `:49-56` |
| 8B source 干预 | 10 例 true-16K passkey median hit@16 **18.75%→64.06%**；去远程 gold 注意力 pooled NLL **+1.5055**（EVQ） vs **−0.0095**（Native） | `:68-73` |
| 8B RULER 续训（+516 步，13 家族） | 16K official macro **0.30%→14.03%**；8K **94.44/77.60**；16K whole-response exact **0/1.54%** | `:76-82` |
| 454M 组合固定 smooth-ramp | 16K PPL Geo **157.7** / EVQ **107.5** | `:84-90` |

### 2.5 冻结模型（`sections/04_mature.tex`）

- 坐标定义（`:15-20`）：`d_k = log(omega_k^N / omega'_k)`，`omega'_k = omega_k^N e^{-d_k}`，`phi'_k − phi_k^N = d_k/log b`。
- 三种构造的位移 profile（`:22-30`）：frequency blend `−log(1−w_k+w_k/s)`；log-frequency shift `m_k log s`；mixed-radix `Σ_{j<k} log λ_j`。`:32-33` 明说 "YaRN's NTK-by-parts frequency map uses the first construction; MrRoPE uses the third"。
- 固定范围 RULER（表 `:57-65` + `appendix/a6_mature_scale.tex:117-199`）：OLMo-1B unseen-nine 16K：Geo **0.56** / Coarse ramp **61.04** / Derived **60.47**；Qwen-1.5B core-four 64K：57.75 / 64.00 / **66.50**。参考行：Native 0.00/54.50、official Transformers YaRN(s=4) **7.94/60.25**（`a6_mature_scale.tex:146-156`，表 `tab:frozen-fixed-support`）。
- 静态表（表 `:96-105`）：Qwen-0.5B RULER-13 macro — Native 54.78/22.05、YaRN-2 55.94/45.37、Normalized-index-2 55.92/**51.46**；64K 差 **+6.09pp [2.76, 9.58]**，32K 差 −0.0256 [−3.26, 3.22]。
- Gemma-1.1-2B（K=128，16K）：index **79.00** vs direct-gap **72.81**，+6.19 [2.81, 9.63]（`:68-78`）。
- BM vs MrRoPE-Pro（表 `:151-160`）：OLMo-1B 4/16K 短/长 — MrPro **37.85/2.78**、BM **81.81/51.32**；Qwen-3B 32/128K — MrPro **87.22/78.13**、BM **91.67/70.83**；Qwen-7B — MrPro **83.33/84.44**、BM **80.00/71.11**。`a7:140-141` 补：MrRoPE-Uni 76.88/32.12，official YaRN 54.38/6.94（OLMo 16K）。
- 正文结论句（`:136-141`）："BM gives the higher OLMo scores and Qwen-3B's higher 32K score; MrRoPE-Pro gives the higher 128K scores on both Qwen checkpoints… these differences connect the preferred exponent shape to the checkpoint and operating length." → **论文自己承认 128K 上目前没有赢家**。

### 2.6 相关工作定位（`sections/02_related.tex`）

FMRoPE（base↔训练长度）、YaRN（band blending + amplitude）、LongRoPE（dimension-wise + position threshold）、MrRoPE（mixed-radix Uni/Pro）、LeRoPE/AdaRoPE（学习频率表项）、DoPE（truncated matrix entropy）、FoPE/HoPE/Clipped、GRAPE/Selective RoPE/RePo、Du et al.（长上下文 position/token 辨识）。差异句："Our contribution lies in their allocation geometry, explicit density construction, and controlled separation from range."（`:27-28`）。

---

## 3. 理论部分：已经写了什么、还空着什么

### 3.1 正文 §4 `Positional Geometry of Exponent Distributions`（`sections/03_theory.tex`）

| 小节 | 已写内容 | 行号 |
|---|---|---|
| §4.1 What different exponents represent | 每个旋转对贡献 `C cos(ωΔ)+D sin(ωΔ)`；位置对象 = 子空间 `V_ω = span{cos(ωΔ), sin(ωΔ)}` | `:8-15` |
| | canonical correlation：`S_ω=E[x_ω^T x_ω]`、`H_ων`、`Q_ων=S_ω^{-1/2}H_ων S_ν^{-1/2}`；`c_ων = ½‖Q_ων‖_F² ∈[0,1]`，相位不变 | `:17-28` |
| | 谱预算恒等式：`r2(Γ) = (trΓ)²/tr(Γ²) = 2K / [1+(K−1) c̄]` | `:30-38` |
| | 数值锚点：b=500,000、K=64、L=4096 → **23 个标准网格 slow pair（ωL≤1）、46 名义维、r2 = 2.00013** | `:58-62` |
| §4.2 How slow exponents share positional directions | **Proposition（Shared slow-frequency subspace）**：`ωL→0` 时 `V_ω → span{1, Δ}`；`2−‖Q_ων‖_F² = O(ε⁴)` | `:46-52` |
| §4.3 An analytic allocation construction | 密度 `ρ∈C²([0,1])`、`∫ρ=1`；慢尾质量 `S_ρ(t)=∫_t^1 ρ`；**凸准则** `C_app[ρ] = (α/2)∫ρ² + (β/2)∫S_ρ²`，α>0, β≥0 | `:74-90` |
| | **Theorem（Cosh allocation）**：唯一 minimizer `ρ_τ(φ)= τ cosh(τ(1−φ))/sinh τ`，`τ=√(β/α)` | `:92-102` |
| | Euler–Lagrange 归约为 `ρ''=τ²ρ`，`ρ'(1)=0`；反 CDF 得有限表 `φ_k(τ) = 1 − (1/τ) arcsinh((1−u_k) sinh τ)`，`u_k=(k+½)/K`，称 **EVQ-Cosh** | `:104-113` |
| | 安装：`ω_k = b^{−φ_k}`，训练/适配中固定不变，注意力算子与可训练参数量不变 | `:117-125` |

### 3.2 附录 A1 `Proof Details`（`appendix/a1_proofs.tex`，561 行）

已经全部写实的内容（按小节）：

1. **闭式 cross-Gram**（`:8-26`）：`H_ων = ½[[a(d)+a(s), b⋆(s)−b⋆(d)],[b⋆(s)+b⋆(d), a(d)−a(s)]]`，`d=(ω−ν)L`、`s=(ω+ν)L`、`a(t)=sin t/t`、`b⋆(t)=(1−cos t)/t`；basis invariance 的显式正交因子证明（`:28-37`）。
2. **Theorem（Spectral budget identity）**（`:39-56`）：`trΓ=2K`、`tr(Γ²)=2K[1+(K−1)c̄]`，完整证明。
3. **Prop collapse 的完整证明**（`:62-121`）：θ=ω²L² 展开；`Π_0^⊥` 投影多项式 `p_2=t²−t+1/6`、`p_3=t³−(9/10)t+1/5`；`G_0`、`M` 显式；**`2−‖Q‖_F² = (x²−y²)² tr(G_0^{-1}M) + O(ε⁶) = (19/12600)(x²−y²)² + O(ε⁶)`**（`:96-101`）；数值比 1.00058 @x=.05,y=.10（`:102-105`）。**softmax 度量段**（`:107-120`）：`F = diag(p) − pp^T`，`F·1=0`，常数方向在 whitening 前被湮灭；中心化极限 `sin̄(ωΔ)/ω → Δ − E_pΔ`、`−2cos̄(ωΔ)/ω² → Δ² − E_pΔ²`。
4. **方向重叠 vs 原始特征尺度**（`:123-139`）：`λ_±(S_ω) = (1 ± |sinc x|)/2`，`λ_− ~ x²/12`，`κ(S_ω) ~ 12/x²`。标准网格 23 个慢对；endpoint-inclusive 网格 24 个。
5. **Proposition（Parity-lattice orthogonality and recurrence）**（`:179-199`）：`ω_k=πa_k/L` 同奇偶 ⇒ `Γ=I_{2K}`，`r2=2K`；容量 `K_par(L)=⌈⌊L/π⌋/2⌉`；偶数 a 时 `Φ(Δ+L)=Φ(Δ)`，奇数时反周期。
6. **两个静态排序反例**（`:228-276`）：cos-only 可选更低秩表（`C_cos(A)=0 < 1.38725e−5 = C_cos(B)`，而 `r2(A)=6.29478 < 7.99971 = r2(B)`）；full-subspace 碰撞跨长度反转（`C_L(A)=0.538996 < 0.631721=C_L(B)`，`C_{2L}(A)=0.384445 > 0.204995`，`C_{4L}(A)=0.345188 > 0.033506`）。数值经 128 点 Gauss–Legendre 校验，最大偏差 < 4e−14（`:271-276`）。
7. **Theorem（Post-hoc transplant obstruction）**（`:284-312`）：`A^T R_{Ω'}(Δ) B = R_Ω(Δ)` 于 0 的开邻域 ⇒ 频率多重集相同（可差符号/置换）；含整数位置的单步谱论证。
8. **Cosh 变分最优的完整证明**（`:315-367`）：核 `K_app(φ,ψ)=αδ(φ−ψ)+β min(φ,ψ)`；带 Lagrange 乘子的约束一阶变分；`L²` 上存在性/唯一性；`g''=−ρ` ⇒ `ρ''−τ²ρ=0`；**边界条件 `ρ'(0)=−τ²`、`ρ'(1)=0`**；解 `<式 eq:rho-tau-closed>`；`:356` 明确 **"the unconstrained Euler--Lagrange solution coincides with the KKT-constrained solution"**（正性约束非活跃）；`:358-363` PSD/凸性（Green 核恒等式 `∬ f f min = ∫(∫_s^1 f)² ≥ 0`）。
9. **Lemma（Single-crossing budget shift）**（`:373-398`）：`ρ_τ` 与均匀密度恰好交叉一次于 `φ_c(τ) = 1 − τ^{-1} arcosh(sinh τ / τ) ≤ 1 − 1/√3`。
10. **Theorem（Surrogate self-consistency）**（`:406-442`）：`τ²T_2(τ) + T_1(τ) = τ coth τ`；Corollary 闭式 `T_2(τ) = (sinh 2τ − 2τ)/(4τ sinh²τ)`。
11. **Allocation strength and a reference rule**（`:447-512`）：`τ = c·d_head/√L_train`，`u_k=(k+½)/K`，`ω_k^{EVQ}=b^{−φ_k(τ)}`，取 c=1（`:454-461`）；四假设局部标度计算 `S_{χ²}(τ)=τ⁴/(45d_head)+O(τ⁶)`、`U(τ,L)=(d_head/L)[Q_0+τ²Q_1+…]`、`q(x)=½+sin2x/(4x)−(sin x/x)²`；平衡给 `τ_*² = 45λQ_1 d_head²/L`、`c_loc=√(45λQ_1)`（`:463-499`）；99 runs 的参考/邻点结果（`:500-512`）。
12. **Discrete-channel transport gap**（`:515-561`）：`W_∞ ≤ 1/(2Km)`、`W_1 ≤ 1/(4Km)`、`‖ρ_K−ρ‖_1 ≤ B/(Km)`；Cosh 代入 `m_τ=τ/sinh τ`、`B_τ=τ²` ⇒ `W_1 ≤ sinh τ/(4Kτ)`、`‖ρ_K,τ−ρ_τ‖_1 ≤ τ²/K + O(τ⁴/K)`；核积分误差 `L_K/(2Km)`；高分辨率 Bennett 积分 `D_K[ρ] = (1/12K²)∫ w/ρ² + O(K^{-3})`。

### 3.3 理论部分**空着**的东西（对照 KKT 目标）

[已验证 —— 以上文 grep 与逐节阅读为依据]

1. **没有任何以"部署表最优性"为名的定理**：不存在极小化"in-window 损失 + 外推能力"的泛函；`C_app` 是**训练期密度准则**（"specified design preferences" / "tractable surrogate"，`03_theory.tex:89-90`），不是 in-window vs extrapolation 的权衡泛函。
2. **没有端点约束形式化**：主文没有任何 `m_j=0 (j≤l)` / `m_j=1 (j≥h)` / `ΣΔ=1` 的约束记号。最接近的是 `04_mature.tex:22-30` 的 `d_k`/`m_k` 坐标与 `a7:92-94` 的 `m_0=0, m_N=1`。
3. **没有"三段结构"陈述**：全文没有 high/mid/low 三段的定理化表述；`intermediate` 只作定语（`04_mature.tex:107,110,129,136`；`01_intro.tex:51,55,63`）。
4. **没有 YaRN/NTK/MrRoPE/BM 的同一变分问题统一陈述**：`04_mature.tex:22-30` 只给了三条位移公式的**并列**，没有共同的目标泛函或共同可行域。
5. **没有水床/守恒不等式**。`NEXT_DERIVATION_KKT_PROBLEM.md:33` 称"水床不等式（EVQ tex 原形 `∫ln E ≥ ln b − ln c`，取等 iff 均匀）"——**该式在当前活跃 tex 中查无此句**（grep `ln E`/`waterbed` 0 命中；`main_0726` 历史树未核）。这是一个待核对的引证缺口。
6. **`C_app` 与 BM 变分的关系没写**：`a7:106-114` 已解 `min Σ(ε_{q+1}−ε_q)²` s.t. `ε_0=ε_{N+1}=0`、`Σε=1` ⇒ `ε_q = 6q(N+1−q)/[N(N+1)(N+2)]` ⇒ `m_q^{BM} = q(q+1)(3N+2−2q)/[N(N+1)(N+2)]`。这是**论文里现成的"单纯形 + 端点约束 + 严格凸 ⇒ 唯一解"实例**，但正文/附录**没有**把它与 `C_app` 并列成同一方法论。→ 这是 KKT 插入的**最短桥**（[部分证据]，见 §5）。

### 3.4 未被 `main.tex` 引用的孤段（**潜在插入载体**）

`main.tex:89-97` 只输入 `00_abstract, 01_intro, 02_exponents, 03_findings, 03_theory, 04_experiments, 04_mature, 02_related, 05_discussion`（正文）与 `a1,a2,a5,a6,a7,a3`（附录）。**未被引用**的 tex [已验证]：

- `sections/budget_intro.tex`、`budget_method.tex`、`budget_theory.tex`、`budget_experiments.tex`、`budget_related.tex`、`budget_abstract.tex`、`budget_discussion.tex`（"固定旋转预算"框架残段）
- `appendix/budget_proofs.tex`、`appendix/a4_supporting_experiments.tex`
- `sections/02_identification.tex` **被** `03_findings.tex:9` 输入（不是孤儿）

其中 `budget_theory.tex:33-50` 的 **Proposition（Finite budget and bounded-coefficient response）** 是最接近"带约束最优/上界"的现成数学：慢对残差 `e(x)=‖(I−P_0)Q_x‖_HS²`、`η=Σ_{x_k≤ε} e(x_k)`、`r_0=2(K−m+1)` ⇒ `Σ_{r>r_0} λ_r ≤ η`；`‖(I−P_U)A_Ω c‖ ≤ B√η`；`inf‖b−A_Ω c‖² ≥ (‖(I−P_U)b‖ − B√η)_+²`；softmax 桥 `D_KL(p_b‖p_f) ≥ (e^{-2M}/2)‖b−f‖²`（`budget_proofs.tex:1-55` 给条件与证明）。`:81-87` 已明说这是**条件性基结果、不是 LM loss 界**。

---

## 4. 投稿目标与时间线约束

- 目标 venue：**ICLR 2027**（`main.tex:8`、`SUBMISSION_CHECKLIST.md:17-20`）。
- 里程碑（`SUBMISSION_CHECKLIST.md:24-27`；`main.tex:8`）：**2026-09-17** 内部标题/摘要/作者冻结 → **2026-09-18 23:59 AoE** 官方摘要截止 → **2026-09-25** 全文截止。
- 硬格式（`main.tex:13-18`、`SUBMISSION_CHECKLIST.md:84`、`compile.sh:13`）：正文 **≤9 页**（rebuttal/CR 才 10），references 与 appendix 不计；AI use statement 必需且不计页。
- **页数实测 [已验证]**：`pdfinfo main.pdf` = **37 页**；pdftotext 页脚定位 Related Work 结束于 **p8**、Discussion 在 **p9**，Ethics/Reproducibility 与 Discussion 共处 p9（豁免）。→ **正文正处在 9/9 页满额，零余量**。
  - **口径冲突**：`HANDOFF.md:26` 说 "9 body pages and 38 total pages"（总页数对不上，37 vs 38）；`main.aux:1` 的 `\label{page:bodyend}` = **7**（aux 生成于 09-08 17:58，早于 sections 的 09-09 23:28 mtime，**已陈旧**）。以 PDF 页脚为准 = 9 页满。
- 审稿进度 [已验证]：`research/pdf-review-rounds/20260909/` 下有 `r01`–`r05`，**r01–r04 有 `review.md`，r05 只有 `identity.json` + `paper.pdf`**。而 `REVISION_BRIEF.md:23-24` 说"Rounds 1 and 2 have completed; round 3 is reviewing"、`HANDOFF.md:23-27` 同口径、`EXPONENT_REVISION_REPORT_20260909.md:32` 说"三轮完成、第四轮进行中"。→ 三份文档互相矛盾且都落后于目录实况（实况 = 4 轮完成，第 5 轮快照已建无评审）。
- 最高一轮结论（`r04/review.md:3`）："Verdict: accept leaning, medium-high confidence… without finding an error overturning them"；且 `:17` 记录了**作者指令**："after this review, the author reaffirmed the exponent-distribution research question and **prohibited defensive writing**… removed repeated inventories of unclaimed universal optima, missing theory targets, evidence ceilings, and capability caveats." ← **这条直接约束 KKT 的写法**（见 §5.4）。
- r02 处置 4（`r02/review.md:12`）："No inferred **LM-optimality** statement added."

---

## 5. KKT 推导应当插到哪里、以什么形式

### 5.1 结论（推荐）

**分三处插入，主文一处、附录两处**：

| # | 落点 | 形式 | 理由（含出处） |
|---|---|---|---|
| **I-1** | `sections/04_mature.tex` §6.1 之后（现 `:30` `eq:adjustment-families` 之后，新建 §6.x），**或** `sections/03_theory.tex` §4.3 之后（现 `:125` 之后） | 一条 **Proposition + 一个 Remark 式三段结构句**（约 8–14 行 TeX） | §6.1 是论文唯一定义 `m_k`/`d_k` 与三种位移族的地方（`04_mature.tex:15-33`），KKT 的三段结构正是关于 `m_k` 形状的命题。若放 §4.3 之后，则与 `C_app`→Cosh 形成"同一变分模板、两种约束"的并列，方法论更闭合（`03_theory.tex:74-113` + `a1_proofs.tex:315-367`）。**两者都要避开把几何量当目标**（见 §6 冲突 C2）。 |
| **I-2** | `appendix/a1_proofs.tex` 新增一小节，放在 `sec:lambda-cv`（`:512`）之后、`sec:discrete-continuous-gap`（`:515`）之前 | 完整证明（KKT 条件 + 互补松弛 + 离散凸程序 + 与 `C_app`/`J[h]` 的接口） | 与现有证明体例同构：`a1_proofs.tex:315-367` 已经是"凸泛函 + 约束 + 一阶变分 + 约束活跃性讨论"，`:356` 已写 KKT 术语。零体例摩擦。 |
| **I-3** | `appendix/a7_exponent_adjustments.tex` §`sec:bm-construction`（`:85-124`）就地扩写 | 一小段把 `min Σ(ε_{q+1}−ε_q)²` s.t. `ε_0=ε_{N+1}=0, Σε=1` 明确写成**单纯形上带端点约束的凸程序**，作为 KKT 命题的离散实例 | 论文**已经有**这个解（`a7:106-113`），只差一句"这是同一族的实例"。零新结果、零新数学风险。 |

### 5.2 表格形式（强烈建议，成本最低）

新增一张**小表**（3–6 行），把现有方法映射到同一个单纯形上的点：列 = 方法 / 位移公式 / `Δ` 形状 / 端点约定。素材现成：
- `04_mature.tex:22-30`（YaRN=blend、log-shift、MrRoPE=radix）
- `a7:96-99`（`m_q^Uni=q/N`、`m_q^Pro=q(q+1)/[N(N+1)]`、`m_q^BM=q(q+1)(3N+2−2q)/[N(N+1)(N+2)]`）
- `tables/table_allocation_protocols.tex:13-24`（各实验 τ/网格/设定约定）

**不要**直接抄 `NEXT_DERIVATION_KKT_PROBLEM.md:80-95` 的 14 行面板表（含 `HighGapToLong`、`E2`、`E8`、`LBS`、`P2`、`s28_less` 等**论文未出现过的臂名与未发表的 36 行面板数字**）——那会引入新 claim 与新证据层（违反 `SUBMISSION_CHECKLIST.md:99-119` Gate D 的"每个数字必须可追溯到 canonical owner"）。

### 5.3 页预算（这是硬约束）

正文 **9/9 页已满**（§4），所以：
- **首选**：I-2 + I-3 全在附录（**不占页数**，`main.tex:15`、`compile.sh:39-46`），主文只加 **一句话指引**（"the displacement families admit a common optimality condition; see App. A.x"）。零页成本，零风险。
- **次选**：I-1 进主文，则必须腾页。前序 digest 已给可压缩对象 `digest_paper-state.md:279`（§6.2 Gemma 段保数字删散文；Qwen-0.5B 段重复协议句）。**但压缩正文属于 codex/作者的编辑决定，不是本次挖掘的结论。**
- 任何改文后必须跑 `bash paper-2027/compile.sh` 过 9 页闸，并重生成 source zip + `package_verification`（`digest_paper-state.md:283`；上次验证记录 `pdf-review-rounds/20260909/package_verification.json`：五图像素一致、四表一致、几何偏差 3.55e-14）。

### 5.4 写法约束（来自权威文档 + 作者指令，逐条有据）

1. **不得当"防御性写作"写**（`r04/review.md:17`）：作者明确禁止"未主张的普适最优性清单 / 缺失理论目标 / 证据上限 / 能力保留"的堆叠。
2. **不得声称 universal optimality**（`r02/review.md:12`；`REVISION_BRIEF.md:10`；`CHANGES_FROM_NEURIPS2026.md:17`："the paper's empirical allocation finding does not depend on Cosh being the uniquely best nonuniform family"）。→ KKT 命题必须写成"**在声明目标类内**的驻点/结构形式"，与 `INTEGRATION_20260910.md:58` 的可防御措辞一致（该段本身就是为这种限定写的）。
3. **不得把静态几何量当 F 的分项或选择子**（红线；`INTEGRATION_20260910.md:28,84`）。→ 若 I-1 放在 §4.3 之后，必须显式说明 KKT 目标**不是** `c_ων`/`r2`/`C_cos`，否则会被读成把 `a1_proofs.tex:236-267` 的反例自己踩了。
4. **必须点名坐标**（`INTEGRATION_20260910.md:30` R4：Σm 不是守恒量）。→ 若写"守恒"，只能写"过渡段总跨度锁定"（`NEXT_DERIVATION_KKT_PROBLEM.md:43`），不得写"17 个 gap 之和 = ln S"（`:114` 明令禁止）。
5. **变量清单要逐项声明**（`NEXT_DERIVATION_KKT_PROBLEM.md:107`：频率/间隔/密度/整数通道数、固定端点还是固定 A、允许重复频率否、整数距离 vs 连续近似）。
6. **gain 独立记账**（`INTEGRATION_20260910.md:107` 附带；论文 `a7:5-6` 已声明 amplitude `g` 对 logit 贡献 `g²`）——F 不含 gain 自由度（`NEXT_DERIVATION_KKT_PROBLEM.md:115`）。

### 5.5 不能直接搬进论文的 KKT 素材（本轮判定）

- `NEXT_DERIVATION_KKT_PROBLEM.md:45-57` 的 F = `L_near + μ L_far` 与其 KKT 三段预言：`:57` 自己标注"**待证明，不预设结论成立**"，且 `:49-51` 的 `L_far` 机制仍是 [部分证据]，`:122`（K3 数值解）、`:124`（K5）未跑。→ 只能作为**命题的假设清单**或 Discussion 的 hypothesis 从句。
- `:104-107` 的 `J[h]=(1/2)∫[α/h + β(1−u)²h]` 与"EVQ=Cosh 的 E-L 解、MrPro=常边际近似"：前者标 [已验证-推导]，后者是 K4 **待证**。→ `J[h]` 可作 KKT 与 `C_app` 的接口件引用；两个"极限关系"必须写成 conjecture。
- `INTEGRATION_20260910.md:66` 的 sol14 盒装优化（`max_ν min_r μ_r/√(v_r+ε_r²) s.t. …`）：标 [部分证据]，且 "**从未端到端运行过**"。→ 不得进论文。
- `INTEGRATION_20260910.md:68` 的 sol16 softmax-Helmert 参数化 `ε(η)=softmax(log ε^Mr + Bη)`：CPU gradcheck PASS，可作附录里"如何求解"的一段；但它是 MrPro 面的坐标，不是论文对象。

---

## 6. 与权威文档 / 其他材料的冲突与口径不一致

| # | 冲突点 | 论文侧出处 | 权威/他处出处 | 判定 |
|---|---|---|---|---|
| **C1** | **τ 规则** `τ = c·d_head/√L_train`（c=1）被论文当"reference rule"写实，并用于设定 151.9M(τ=4=64/√256)、Llama-3-8B(1.414≈128/√8192)、OLMo-2(τ=2=128/√4096) | `appendix/a1_proofs.tex:454-461`；`tables/table_allocation_protocols.tex:13,16,24` | `INTEGRATION_20260910.md:87` 把 "τ≈d_head/√L" 列入**叙事-过度类**，附 "PASS"=脚本约定、**9.6% 均值 / 33.3% 最大锚误差**；`digests/digest_paper-state.md:145`（P15）补：COSH_REDESIGN §2 给"公式−邻点 3/9、8/18、+0.0221"与"解析缺口 4.4×/5.7×" | **实质冲突，需作者裁决**。论文已用 "reference used"、`order varies by configuration`（`a1:505-507`）限定，但表格里 `64/√256`、`128/√8192`、`128/√4096` 的呈现读起来像定律。若 KKT 命题要引用 τ，必须先解决这条。 |
| **C2** | 论文 §4 的 `c_ων`、`r2`、`C_cos` 是**诊断量** | `03_theory.tex:23-38`；`a1_proofs.tex:236-267`（反例） | `INTEGRATION_20260910.md:28`（R2）、`:84`（几何-无符号类全灭） | **不是直接矛盾**（论文未把它们当选择子），但**KKT 命题一旦与 §4 相邻就会被误读**。插入时必须写"geometry is diagnostic, not the objective"的**一句**（不得堆叠，见 §5.4-1）。 |
| **C3** | YaRN/MrRoPE 的算子映射 | `04_mature.tex:32-33`（YaRN=first construction、MrRoPE=third）；`04_mature.tex:110-113`（"MrRoPE-Pro progressively distributes…"） | `STARTING_POINT_YARN_VS_MRPRO.md:9`："任何后续推导不得再使用 'YaRN 递减 vs MrPro 递增' 作为机制解释（F1/F2 已证伪）"；`:21,25-27`：标准 YaRN 的 `m_Y` 也凸也递增；MrPro 的二次形式是**设计假设**不是任何目标的解 | **论文未声称单调性差异**，故非硬矛盾；但 `:110` 的 "progressively" 与 `:136-141` 的"preferred shape"叙述在权威口径下必须降为**设计选择**（`STARTING_POINT:27`）。 |
| **C4** | Cosh 的"唯一 minimizer"范围 | `03_theory.tex:92-102`（Thm）、`a1_proofs.tex:315-367`（证明）；限定语在 `03_theory.tex:89-90`（"tractable surrogate"） | `INTEGRATION_20260910.md:69`（astra02/sol15）：精确有限窗碰撞泛函的唯一测度最优是**有限原子**，不是正 Cosh 密度；四对象（①精确-原子 ②光滑-Cosh 代理 ③有限-K 整数计数 ④冻结-带标签表）**两两不可互换** | **论文缺一句"四对象不可互换"的范围限定**。若 KKT 命题引入 `J[h]`/`C_app` 的连续→离散桥，这条会变成审稿人抓手。 |
| **C5** | 审稿轮次状态 | `REVISION_BRIEF.md:23-24`、`HANDOFF.md:23-27`、`EXPONENT_REVISION_REPORT_20260909.md:32` 三份互不一致 | 目录实况：`research/pdf-review-rounds/20260909/r01..r05`，r05 无 `review.md` | **文档口径不一致**（见 §4）。任何理论插入都会让"不可变 PDF"审稿链失效，必须写回 `HANDOFF.md`。 |
| **C6** | 页数 | `HANDOFF.md:26` "9 body / 38 total" | 实测：`pdfinfo` = 37 页；页脚定位正文 = 9 页；`main.aux:1` label=7（陈旧） | 以 PDF 为准：正文 9/9 满额。 |
| **C7** | OLMo 16K BM vs MrPro 面板 | `04_mature.tex:156` 表 = MrPro **2.78** / BM **51.32**（48 行长 prompt 分层） | `INTEGRATION_20260910.md:52` 记录 350 条：**BM 41.67% vs MrPro 7.09%**（156W/9L） | **不是矛盾，是不同面板**（48 行 vs 350 条），但两处数字若同段引用必须标注口径。`a7:141` 的 YaRN 6.94% 与 `STARTING_POINT:75` 的 6.94% 一致，交叉印证 2.78 那条。 |
| **C8** | "水床不等式 EVQ tex 原形 `∫ln E ≥ ln b − ln c`" | 活跃 tex **查无此式** [已验证 grep] | `NEXT_DERIVATION_KKT_PROBLEM.md:33` 称其为"EVQ tex 原形" | **引证缺口**：该式可能只在 `main_0726` 历史树或旧稿里。若 KKT 稿要复用它，必须先定位原式（本轮未在活跃 tex 找到）。 |

---

## 7. 死路登记（**绝不能再试**，含失败原因与出处）

全部来自权威文档的"死亡机制登记册"与红线；论文侧仅作对照。这些是**给推导/写作的禁令**，不是给代理的指令。

1. **Σcos 首零点 / 根排序作为 F 或选择子**：根排序与能力排序显著失序（MrUni 82.2K > MrPro 80.3K 而 32K 64.6 ≪ 87.2；E2/P2 同根一崩一 81.7；s28 修复 +5.2pp 时根几乎不动）。出处 `INTEGRATION_20260910.md:27,84`；`STARTING_POINT_YARN_VS_MRPRO.md:86`。论文已有两个静态反例同族（`a1_proofs.tex:236-269`）。
2. **静态几何代理作选择子**：碰撞能 / 覆盖率 / 平滑度 / 有效秩 / 能量 / Gram / MAE / 轨道计数。决定性反例 Smooth：几何全赢而 far 端 68.3 vs MrPro 78.13（near 打平）。出处 `INTEGRATION_20260910.md:28`（R2）、`:84`。
3. **"YaRN 递减 vs MrPro 递增"叙事**：F1/F2 已证伪（两者都凸、都递增）。出处 `STARTING_POINT_YARN_VS_MRPRO.md:9,21`。
4. **"17 个 log-gap 之和 = ln S" 的口径**：错误；正确 = 总跨度锁定（5.0560 nats）+ 原生部分分开计。出处 `NEXT_DERIVATION_KKT_PROBLEM.md:43,114`。
5. **Σm（质心）当守恒量**：Σm 是**自由决策变量**；零和频移实测 ΔΣm ≠ 0（−0.01826…+0.00185）。**任何守恒论证必须先点名坐标**。出处 `INTEGRATION_20260910.md:30`（R4）。
6. **逐槽可加性**：pair(28+29) 相对 MrPro **−4.17pp**；跨 key 相干与共享 head/W_O 对消。逐槽列表法先天失效。出处 `INTEGRATION_20260910.md:52,86`；`NEXT_DERIVATION_KKT_PROBLEM.md:87`。
7. **冻结 checkpoint 上用密度/多重集/排序参数化**：同多重集置换 NLL 3.104→6.865、Qwen core-4 0.70→0；联合置换频率+学习系数槽才是恒等。出处 `INTEGRATION_20260910.md:85`。
8. **训练期密度直觉移植到冻结表**：Geo↔Cosh 运行时互换 PPL 7.14↔76.20 / 7.16↔23.05。论文**已把它当 finding 写**（`03_findings.tex:12-20`）——即：这条在论文里是**正面发现**，但**不能**反过来当"冷冻态可自由换密度"的依据。出处 `INTEGRATION_20260910.md:85`。
9. **Taylor/Jacobian 局部分数跨全表**：相对误差 71–468%、相位 22.74/90.97 rad。出处 `INTEGRATION_20260910.md:86`。
10. **universal 不可辨识定理 / 六观测量充要 / VICTORY CONFIRMED 类闭合证书 / universal 1×–2× 交换率**：出处 `INTEGRATION_20260910.md:87`。注意 `:99`（FLAG-1）对 R1 的裁决：**面板判决保留（经验事实），但"不可辨识——定理"式表述作废**，措辞缩窄为"没有**已测的 model-blind 无序**统计量能认证冻结部署"。
11. **把 τ 规则升级为 "near-optimal law"**：`digest_paper-state.md:145`（P15）+ `INTEGRATION_20260910.md:87`（C1）。论文现有 `order varies by configuration` 的限定不得删。
12. **直接优化两条死路**：64 维行为梯度未开 holdout 即败；direct-z 定支撑 pilot 挂门——**优化器必须在可辩护统计对象下游**。出处 `INTEGRATION_20260910.md:88`。
13. **一阶/局部代理算子（J_r、校准-KL、E7 160×）入 F 主链**：只配假设生成。出处 `NEXT_DERIVATION_KKT_PROBLEM.md:115`。
14. **重复频率 / 与 `main_0726` 以外的"论文已证"叙事**：`a1_proofs.tex:69`（原子性）与 `INTEGRATION_20260910.md:69` 提示的"类错误"——解一个对象装上另一个对象。

---

## 8. 未解问题

1. **KKT 的 F 还不可计算**：`NEXT_DERIVATION_KKT_PROBLEM.md:51` 明确 `L_far` "不能按单槽弧语言定义，须按联合谱覆盖/风险带定义——这是下一次推导要闭合的核心建模步骤（Q9）"；`:122`（K3 数值解）、`:124`（K5）未跑。→ 论文里只能写**结构定理（KKT 条件的形态）**，不能写"最优表"。
2. **插入位置的裁决权在作者/codex**：正文 9/9 满额（§4），主文插入必须腾页；本轮不替作者决定压缩哪段。
3. **Cosh 的四对象范围限定是否补**（冲突 C4）：`r04/review.md:17` 的"禁止防御性写作"与"补一句范围限定"之间存在张力。`digest_paper-state.md:243`（§7-5）已把同类问题标记为"留给作者决定"。
4. **`budget_*.tex` 与 `a4_supporting_experiments.tex` 的去留未决**：未输入、未删除。`digest_paper-state.md:245`（§7-6）同判。`budget_theory.tex:33-50` 的有限预算命题是否作 A1 remark 一行，无结论。
5. **审稿十轮断在 r04/r05 之间**（`digest_paper-state.md:235`；§4 实况）：补完 r05–r10 还是带 4 轮记录进冻结门，未决。
6. **C1 的 τ 口径**如何与论文已有表格共存，未决。
7. **`∫ln E ≥ ln b − ln c` 的原式在哪**（冲突 C8）——若 KKT 稿要引"水床"必须先定位，否则只能引 `ΣΔ=1` 形式的"过渡段总跨度锁定"。
8. **KKT 命题的验收标准**：`NEXT_DERIVATION_KKT_PROBLEM.md:97` 给出可证伪版本（"若面板回归给出的 ∂F 场在单纯形上的极小点 ≠ 已测赢家的 (j_b, j*) 坐标邻域，则目标建模错误，回炉"）——但该验收依赖尚未解冻的 GPU 队列（`:52`、`digest_paper-state.md:236`）。论文命名的假设清单里是否写入这一验收，未决。

---

## 9. 一句话给下游

论文的**几何篇（§4 + A1）已经把"凸泛函 + 约束 + 一阶变分 + 约束活跃性"这套机器全部装好**（`a1_proofs.tex:315-367`，`:356` 甚至已出现 KKT 术语），**冻结篇（§6 + A7）已经把 `m_k`/`Δ_k` 坐标系与"单纯形 + 端点约束 ⇒ 唯一解"的离散实例装好**（`a7:106-113`）。KKT 推导**不需要新机器，只需要把两处接起来**，并把"报告/transcript 里的 14 点面板"**挡在论文之外**。硬约束只有三条：**正文 9/9 页满**、**不得写普适最优性**、**不得把 `c_ων`/`r2` 当目标**。
