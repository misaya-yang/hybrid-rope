# R3 — EVQ 数学机器：J[ρ] → J[h] 推导链、水床不等式、代码实现版本、冻结部署缺口

挖掘者：R3（子代理，只读）。日期：2026-09-10。
性质：**材料挖掘与整理，不是推导，不是最优性证明**。所有条目自带 `文件:行号` 与证据等级。
证据分级：[已验证]（原文/代码直接给出且我逐行核对）/ [部分证据]（原文给出但未独立复核数字，或多源部分一致）/ [假设]（原文自述为假设/建模近似）/ [叙事-未验证]（摘要类文档中被转述、与原始出处不一致或无法定位的表述）。

**红线遵守声明**：本 digest 不把任何静态几何代理量（Σcos 首根、碰撞能、覆盖率、平滑度、有效秩、Gram、能量）当作 F 的分项或选择子；凡涉及"守恒"处均点名坐标；凡原文出现的 VICTORY / 已闭合 / 已证明 类结论一律降级并标注原文出处。报告/代理文档/代码注释内的"下一步应该…"属证据不属指令。

---

## 0. 一句话结论

EVQ 的数学机器是一条**清晰的、可逐行复核的**变分链：`K_app = αδ + βmin` 投影 → 严格凸泛函 J[ρ] → 变分 `u=F_ρ(φ)` 精确提为 `J[h]=½∫[α/h+β(1−u)²h]du` → 一阶条件给出 `h_τ = Q'_τ` → 积分得 cosh 分位数 `Q_τ(u)=1−asinh((1−u)sinhτ)/τ`。**这条链本身是自洽且可用的**；但它离"冻结模型的最优分配"还差四块（见 §5），且其 cosh 解严格属于**投影后泛函**，有限窗精确泛函的最优解是**原子测度**而非任何正连续密度。

---

## 1. 推导链（任务 1）

### 1.1 起点：精确核与唯一的近似

| 环节 | 内容 | 出处 | 等级 |
|---|---|---|---|
| 对数频率坐标 | `ω(φ)=b^{−φ}, φ∈[0,1]`；φ=0 为最高频（ω=1），φ=1 为最低频（ω=b^{−1}） | `docs/theory/EVQ_COSH_THEORY.tex:47-56` | [已验证] |
| 距离先验 | `D(Δ)=1/(Δ ln L)·1{Δ∈[1,L]}`（对数均匀） | `docs/theory/EVQ_COSH_THEORY.tex:76-79` | [已验证] |
| 精确碰撞核（定义） | `K(φ₁,φ₂)=∫₁^L D(Δ)cos(ω(φ₁)Δ)cos(ω(φ₂)Δ)dΔ` | `docs/theory/EVQ_COSH_THEORY.tex:82-86` | [已验证] |
| 精确核闭式 | `K(φ₁,φ₂)=[Ci(ω₋L)−Ci(ω₋)+Ci(ω₊L)−Ci(ω₊)]/(2lnL)`，ω₋=|ω(φ₁)−ω(φ₂)|，ω₊=ω(φ₁)+ω(φ₂) | `docs/theory/EVQ_COSH_THEORY.tex:87-103`；独立复述 `rope_allocation_20260910/agents/sol02.md:18-30` | [已验证] |
| **唯一的近似** | `K_app(φ₁,φ₂)=αδ(φ₁−φ₂)+βmin(φ₁,φ₂)`，α,β>0 | `docs/theory/EVQ_COSH_THEORY.tex:107-115`（原文自述"the only approximation in the EVQ-Cosh derivation chain"，107 行小节标题即"Broadband projection (the only approximation)"） | [已验证-原文自述] |

**注意（重要限定）**：`docs/theory/EVQ_COSH_THEORY.tex:116-125` 的 Remark 明确写了**不主张**逐点残差在大 base（如 b≈5×10⁵）下渐近小；只主张中频结构经验上 R²>0.9（L=512 时 ≈0.96，L=16K 降到 ≈0.87）。同一 Remark 用了 "Hilbert–Schmidt projection" 措辞 —— 该措辞**在数学上不成立**（见 §4 冲突 C4）。

### 1.2 J[ρ]：原始密度泛函

```
J[ρ] = (α/2)∫₀¹ ρ(φ)² dφ + (β/2)∬ ρ(φ)ρ(ψ) min(φ,ψ) dφdψ,   ρ≥0, ∫ρ=1, α>0, β≥0
```
另一种常用写法用累积量 `S_ρ(t)=∫_t^1 ρ(φ)dφ`：
`C_app[ρ] = (α/2)∫ρ² + (β/2)∫S_ρ(t)² dt`。

- 出处：`docs/research/rope_allocation_20260910/source_inputs/pro6_allocation_analysis.md`（"目标为 J[ρ]=…，其中 S_ρ(t)=∫_t^1ρ(φ)dφ"）；`docs/research/ROPE_ALLOCATION_THEORY_CORE_20260910.md:88-90`；`paper-2027/sections/03_theory.tex:79-85`（eq:Capp）；`paper-2027/appendix/a1_proofs.tex:319-325`。
- 等级：[已验证]（四处一致）。
- 两个二次型都 PSD；Green 核 PSD 恒等式 `∬f(φ)f(ψ)min(φ,ψ)=∫₀¹(∫_s¹f)²ds≥0`，`paper-2027/appendix/a1_proofs.tex:360`（eq:min-kernel-psd）。

### 1.3 变量替换：从密度 φ 到分位 u（**任务问的"在哪一步"**）

原文原话（`pro6_allocation_analysis.md`，二.1 节）：
> 设 `u=F_ρ(φ)`，`φ=φ(u)`，并定义 `h(u)=φ'(u)>0`。`h(u)` 就是连续版本的"每个频率槽位获得多少对数跨度"。因为端点固定：`∫₀¹h(u)du=1`。利用 `ρ(φ(u))=1/h(u)`，`S_ρ(φ(u))=1−u`，原目标**精确等价于** `J[h]=½∫₀¹[α/h(u)+β(1−u)²h(u)]du, ∫₀¹h(u)du=1`。

我逐项复核了替换（两者都通）：
- **α 项**：`(α/2)∫ρ(φ)²dφ`，令 `φ=φ(u)`, `dφ=h(u)du`, `ρ(φ(u))=1/h(u)` ⇒ `(α/2)∫(1/h)²·h du = (α/2)∫du/h`。
- **β 项（用 S_ρ 写法最快）**：中间量 `S_ρ(φ(u))=∫_{φ(u)}^1ρdφ = 1−F_ρ(φ(u)) = 1−u`；β 项 `=(β/2)∫₀¹S_ρ(t)²dt`，令 `t=φ(u)`，`dt=h(u)du` ⇒ `(β/2)∫₀¹(1−u)²h(u)du`。
  （等价的 min 核路径：`∬ρ(φ)ρ(ψ)min(φ,ψ)=∫₀¹(∫_s¹ρ)²ds`，令 `s=φ(u)` 得同一结果。）
- 结论 `J[h]=½∫[α/h(u)+β(1−u)²h(u)]du` 与 `docs/research/ROPE_ALLOCATION_THEORY_CORE_20260910.md:104-107` 完全一致；该核心文档把这条列为本轮**最优先复用件**（同处 109 行）。

**三个位置约束的读法**（原文 `pro6_allocation_analysis.md` 二.1 节逐条给出，[已验证-原文]）：
- `α/h(u)`：避免某些位置的间隔被挤得过小（间隔 `h→0` 时罚 ∞）。
- `β(1−u)²h(u)`：**按位置的间隔代价**，`u≈0`（高频侧）代价更高、`u≈1`（低频侧）代价 →0。
- `∫h=1`：总预算守恒 —— **这里"守恒"的坐标是"分位参数 u 上的总跨度"，即 ∫h du=1，不是 Σm**（对齐红线 R4）。

### 1.4 一阶条件 ⇒ cosh 解

原文一阶条件（`pro6_allocation_analysis.md` 二.2 节）：
```
−α/(2h(u)²) + (β/2)(1−u)² + λ = 0   ⇒   h(u) = √(α / (β(1−u)² + 2λ))
代入归一化 ⇒ h_τ(u) = sinhτ / (τ√(1+(1−u)²sinh²τ)),  τ=√(β/α)
积分后恰好恢复 EVQ 分位点公式。
```
- 等级：[已验证]（原文给出且与 `Q'_τ` 逐项一致，见下）。
- **踩坑提示（我复核时发现）**：若只做逐点 AM-GM 而不带约束，`h∝1/(1−u)` 在 u=1 处**不可积**，`∫h=1` 无法满足。是**归一化约束（乘子 λ）提供了 sinh²τ 这个正则化**，把不可积的幂律改成可积的 cosh 分位。所以 J[h] **不是**自由逐点优化问题 —— 这点在原文与核心文档里都未被点破，属我的整理性备注，标 [假设-整理性观察]。

**连续泛函侧的独立证明（另一条路，结论相同）**：`paper-2027/appendix/a1_proofs.tex:315-368` 从 J[ρ] 直接做：
- 变分：`αρ(φ)+βg(φ)+ν=0`, `g(φ)=∫ρ(ψ)min(φ,ψ)dψ`（a1_proofs.tex:324-326）；
- `g'(φ)=∫_φ¹ρ`, `g''(φ)=−ρ(φ)`（a1_proofs.tex:335-339）；
- 二阶级分消去常数 ν ⇒ `ρ''−τ²ρ=0`, `τ=√(β/α)`（a1_proofs.tex:341-345，eq:homogeneous-ode）；
- **边界条件**：`ρ'(0)=−τ², ρ'(1)=0`（a1_proofs.tex:347，eq:bcs）。原文给出完整推导：由 `αρ(1)+βg(1)+ν=0` 求导 + `g'(1)=0` 得 `ρ'(1)=0`；在 φ=0 得 `αρ'(0)+βg'(0)=0`，且 `g'(0)=∫₀¹ρ=1` ⇒ `ρ'(0)=−β/α=−τ²`（a1_proofs.tex:348-352）。
  - **注意**：质量归一化**是通过边界条件 g'(0)=∫ρ=1 进入的**。
- 通解 `ρ=Acosh(τ(1−φ))+Bsinh(τ(1−φ))`，`ρ'(1)=0` 逼 B=0，归一化定 `A=τ/sinhτ`（a1_proofs.tex:352-356；另见 `EVQ_COSH_THEORY.tex:243-247` 的同一证明）。
- 正性：`ρ_τ(1)=τ/sinhτ>0`，`ρ_τ(0)=τcothτ>0` ⇒ 非负 KKT 约束**处处不激活**，无约束 E-L 解 = KKT 解（a1_proofs.tex:357-359）。
- β=0 退化 ⇒ `ρ≡1`（几何），也是 τ→0 极限（a1_proofs.tex:359-361）。

### 1.5 cosh 密度 / CDF / 分位数（公式表）

| 量 | 公式 | 出处 | 等级 |
|---|---|---|---|
| 密度 | `ρ_τ(φ)=τ cosh(τ(1−φ))/sinhτ` | `EVQ_COSH_THEORY.tex:225-245`（eq:rho_cosh_density）；`a1_proofs.tex:352-356`；`ROPE_ALLOCATION_THEORY_CORE_20260910.md:96`；`03_theory.tex:97-101`；`sol02.md:54-58` | [已验证] |
| CDF | `F_τ(φ)=1 − sinh(τ(1−φ))/sinhτ` | `EVQ_COSH_THEORY.tex:249-269`（eq:cosh_cdf） | [已验证] |
| 分位（warp） | `Q_τ(u)=1 − asinh((1−u)sinhτ)/τ` | `EVQ_COSH_THEORY.tex:255-260`（eq:evq_warp）；`ROPE_ALLOCATION_THEORY_CORE_20260910.md:100`；`03_theory.tex:107-110` | [已验证] |
| 分位导数 | `Q'_τ(u)=sinhτ/(τ√(1+(1−u)²sinh²τ)) = 1/ρ_τ(Q_τ(u))`，**严格递增** | `sol02.md:79-86` | [已验证] |
| τ→0 极限 | `φ_k(τ)→u_k`（EVQ 退化为几何） | `EVQ_COSH_THEORY.tex:271-286` | [已验证] |
| 局部展开 | `φ(u;τ)=u − u(1−u)(2−u)τ²/6 + O(τ⁴)`，τ>0 时 φ<u（整体向高频端平移），**低频间隔变大、高频间隔变小** | `EVQ_COSH_THEORY.tex:296-305`（eq:phi_taylor）；`sol02.md:96-102` | [已验证] |
| 端点锚定 | `φ(0;τ)=0`, `φ(1;τ)=1`（asinh(sinhτ)=τ） | `EVQ_COSH_THEORY.tex:288-294` | [已验证] |
| 相邻间隔（精确） | `g_k=Q_τ(u_{k+1})−Q_τ(u_k) = (1/τ)[asinh((1−u_k)sinhτ) − asinh((1−u_{k+1})sinhτ)]`，等距 u 格上 `g_{k+1}>g_k` | `sol02.md:86-96` | [已验证] |

**单交叉（预算搬运的可证结构）**：`φ_c(τ)=1−τ^{−1}arcosh(sinhτ/τ) ≤ 1−1/√3`，`paper-2027/appendix/a1_proofs.tex:373-398`。[已验证-原文]

### 1.6 离散采样约定（**任务点名项，四种约定并存**）

| 约定 | 使用者 | 出处 | 等级 |
|---|---|---|---|
| `u_k = k/N`（端点含 u=0，不含 u=1） | 理论 note 的主体 | `docs/theory/EVQ_COSH_THEORY.tex:63`（"standard RoPE indexing"）；`:71`（Remark 明确该格不含 u=1）；`:272`, `:353` | [已验证] |
| **`u_k=(k+1/2)/K`（midpoint，两端都不锚定）** | **全部生产代码 + 大多数论文节** | `scripts/lib/rope/learnable_evq.py:82-84`（注释"Fixed: midpoint quantization u_k=(k+0.5)/N [matches paper eq. 9]"）；`scripts/lib/rope/schedules.py:160-183`（docstring 明确 midpoint=True 匹配论文）；`paper-2027/sections/03_theory.tex:110`；`a1_proofs.tex:454-456`（eq:evq-practical）；`paper-2027/sections/budget_method.tex:8-10`；`sol02.md:107` | [已验证] |
| `u_k=(k−1/2)/K`（**同文件内与上一行冲突**） | 同一附录的有限通道传输小节 | `paper-2027/appendix/a1_proofs.tex:518` | [已验证] |
| `u_k=k/(K−1)`（端点全含，anchored inclusive） | 750M continuation 臂 | `paper-2027/appendix/a2_experiment_details.tex:129,133`；`paper-2027/tables/table_allocation_protocols.tex:17` | [已验证] |
| `u_k=k/K`（standard，端点含 u=0） | OLMo-2 EVQ 全部臂 | `paper-2027/appendix/a6_mature_scale.tex:10` | [已验证] |

**锚定（anchoring）操作**（只在 evq_recovery 实验与 750M 臂出现，不是 EVQ 理论本体）：
`z_k^E = (q_k−q_0)/(q_{K−1}−q_0)`，`paper-2027/sections/budget_method.tex:8-12`；代码 `experiments/evq_recovery/tables.py:13-24`（`anchored_cosh`：`u=(arange(k)+.5)/k`，τ→0 时 q=u，最后做 `(q−q[0])/(q[−1]−q[0])`）。

**核心文档的警告（可直接引用）**：`u=j/K`、`u=(j+1/2)/K`、端点归一化采样是**不同有限表，不能混用**（`docs/research/ROPE_ALLOCATION_THEORY_CORE_20260910.md:102`）。更强的量化警告：换约定会改变**每一个槽**，最大可达**半个分位单元**，且端点锚定命题**不描述部署的 midpoint 表**（`sol02.md:104-109`）。逐实验的 τ/grid/约定对照表：`paper-2027/tables/table_allocation_protocols.tex:1-20`（151.9M τ=4 anchored midpoint；50M crossing τ=2.83 midpoint；432M MLA τ=1.414 midpoint；454M composition τ=1.5 midpoint；750M τ=1.5 anchored inclusive；Llama-3-8B τ=1.414 midpoint；OLMo-2 τ=2 standard；129.6M DiT τ=1.5 midpoint）。

---

## 2. 水床不等式（任务 2）

**命题原文**（`docs/theory/EVQ_COSH_THEORY.tex:307-330`，eq:waterbed）：
> 定义局部 Fisher 信息 `I(φ)=c ρ(φ) b^{−2φ}`（c>0），并假设局部误差代理满足 `E(φ) ≥ 1/I(φ)`。则 `∫₀¹ ln E(φ)dφ ≥ ln b − ln c`。进一步，若 `E(φ)=1/I(φ)` 且 ρ 除归一化外无约束，取等**当且仅当 ρ 为常数（几何）**。

**证明步骤（原文给出，我逐行复核）**：
1. `E≥1/I` ⇒ `lnE ≥ −lnI`；
2. `−lnI = −lnc − lnρ + 2φ lnb`；对 φ∈[0,1] 积分，`∫₀¹φdφ=1/2` ⇒ `∫lnE ≥ −lnc − ∫lnρ + lnb`；
3. **Jensen**：`−lnx` 凸 ⇒ `−∫₀¹lnρ ≥ −ln(∫₀¹ρ)=0`，取等 iff ρ 常数。

**假设清单**（要复用必须先复述这四条）：
- (A1) 分辨率模型 `I(φ)=cρ(φ)b^{−2φ}` 形式成立（[假设]，`EVQ_COSH_THEORY.tex:308-310`）；
- (A2) 局部误差代理下界 `E(φ)≥1/I(φ)`（[假设]，同处）；
- (A3) ρ 归一化 `∫ρ=1` 且非负（[已验证]，命题自带）；
- (A4) **ρ 除归一化外无约束**（这是取等的关键前提；若 ρ 被额外约束到某个子族，"ρ 常数"的取等结论不再成立）——原文 `:314-315` 明写 "ρ is unconstrained beyond normalization"。

**守恒坐标点名**（对齐红线 R4）：这里被界住的是 **∫₀¹ln I(φ)dφ 的上界**，取到它的密度是均匀/几何密度。即：**非均匀再分配只能降低平均对数分辨率，不能提高**。这与 `Σm` 不是同一件事 —— 核心文档 `ROPE_ALLOCATION_THEORY_CORE_20260910.md:113-117` 明确划定水床的适用范围，并**自述它没有宣称"任何长任务收益必然伴随短任务分数损失"**。

**与部署规则的对撞（重要，材料内部矛盾）**：`paper-2027/research/three_completions/evq_three_completions.tex:388-391` 把水床尺度 `W(ρ_{τ_*})=O(d_head⁴/L_eff²)` 列为 App. A.6 的产物，并称其"inherits the full 0.28 error"（即继承同一模型 0.28 的误差）。这是"水床"在自洽 τ 体系里被当成**可计算量**的一处，与 `EVQ_COSH_THEORY.tex` 的纯 Jensen 界不是同一对象，**同名不同物**。见 §4 冲突 C5。

---

## 3. 代码实际实现的是哪一版（任务 3）

### 3.1 生产实现 = midpoint + softplus(τ) + Taylor 兜底

`scripts/lib/rope/learnable_evq.py`：
- `:82-84` 注释与代码：`# Fixed: midpoint quantization u_k = (k + 0.5) / N  [matches paper eq. 9]`，`u = (arange(n_freqs)+0.5)/n_freqs` —— **midpoint**。
- `:88-104` `_compute_phi`：`tau.item()<1e-4` 时用二阶 Taylor `phi = u − (τ²/6)·A·(1−A²)`（A=1−u），否则 `phi = 1 − (1/τ)·arcsinh(A·sinh(τ))` —— 与 §1.5 的 `Q_τ` **逐字一致**，且 Taylor 与 eq:phi_taylor 一致（注意代码写 `1−A²`，公式写 `A(1−A²)`，`A=1−u`，逐项吻合）。
- τ 参数化：`raw_tau` 经 `softplus` 得 τ（`:75-79`, `:91-93`）。
- `:224-307` `estimate_tau_from_distance_prior` = Algorithm 1：两步拟合，step1 用**非对角**元素回归 `K_ij≈c₀+β·min(φ_i,φ_j)` 得 (c₀,β)（`:275-286`），step2 用**对角** `K_ii≈c₀+βφ_i+α/Δφ` 得 `alpha=(residual_diag.mean())·dphi`（`:288-291`），`τ*=√(β/α)`，非物理时回退 `1.0`（`:293-297`）。
- 文件头 docstring 自述引用 "§4 of 'RoPE Scaling as a Variational Inverse Problem'"。

`scripts/lib/rope/schedules.py:161-183` `evq_cosh_phi(n_freqs, tau, midpoint=True)`：midpoint 分支 `u=(idx+0.5)/n_freqs`，否则 `u=idx/n_freqs`；docstring 原文："`midpoint=True` matches the grid used in the paper experiments: u_k=(k+0.5)/K. With tau=0 this recovers the midpoint-discretized geometric schedule, while `geometric_inv_freq` above keeps the standard RoPE endpoint grid u_k = k/K."

`experiments/evq_recovery/tables.py:13-24`：`anchored_cosh` 在 midpoint 之上再做 `(q−q[0])/(q[−1]−q[0])` 锚定；同文件另有 `exponential`、`hybrid`（保留 16 个高频对，split=k//4）、`match_deformation`（二分匹配 RMS 形变 0.129239）、`construct`。

### 3.2 代码 vs tex 的一致性判定

| 项 | 判定 | 依据 |
|---|---|---|
| warp 公式 `Q_τ` | **一致** | 代码 `learnable_evq.py:107` / `schedules.py:182` 与 `03_theory.tex:110`、`EVQ_COSH_THEORY.tex:255-260` 逐字相同 |
| 离散格 | **不一致（分三类）** | 理论 note 用 k/N；全部生产代码用 (k+1/2)/K；750M 臂用 k/(K−1)；OLMo-2 用 k/K |
| 附录内部 | **自相矛盾** | `a1_proofs.tex:456` 用 (k+1/2)/K，`:518` 用 (k−1/2)/K（同一文件） |
| τ 物理含义 | **一致** | 代码 softplus(raw_tau) 与 tex 的 τ=√(β/α) 均是单一正参数；但代码的 **τ 估计器**（Algorithm 1）与 `a1_proofs.tex:453` 的**参考律** `τ=c·d_head/√L_train` 是两条不同的取 τ 途径 |
| 锚定 | **只在实验代码出现，不在理论本体** | `tables.py` vs `EVQ_COSH_THEORY.tex:288-294`（理论版端点锚定是 φ 端点，不是重标定 q） |

### 3.3 已发现的代码侧问题（转述，非我自行复现）

- `scripts/analysis/tau_static_vs_dynamic_experiment.py:71-83` 的 α 拟合约定**被判定为错**：用 `mean(diag)·mean_spacing` 且**未减去拟合出的 min-kernel 对角项**，log-distance 求积也**不是精确 Ci**（出处：`docs/research/rope_allocation_20260910/agents/astra02.md` 的相应段落，[部分证据-转述]）。
- 有限 K 直方图遗憾有独立 CPU 证书：`K²(J[ρ_K]−J[ρ_*]) → (α/24)(τ²−τ·tanhτ)`，`scripts/analysis/finite_k_cosh_regret_audit.py:1-9,100`（函数 `cosh_quantile(mass,tau)=1−asinh((1−mass)sinhτ)/τ` 在 `:28`，解析 `continuous_components` 在 `:40`）；测试断言常数相对误差 <3e-4、log-log 斜率 −2±0.01，`tests/test_finite_k_cosh_regret_audit.py`。默认格状态 PASS，**作用域明写 "not r2"**。[已验证-代码+测试]

---

## 4. 冲突与不一致登记

| 编号 | 冲突 | 两边出处 | 处理 |
|---|---|---|---|
| **C1** | 有限表约定在同一附录内自相矛盾：`u_k=(k+1/2)/K` vs `u_k=(k−1/2)/K` | `paper-2027/appendix/a1_proofs.tex:456`（eq:evq-practical） vs `:518` | 以 **midpoint (k+1/2)/K** 为准（与全部生产代码、`03_theory.tex`、`budget_method.tex`、sol02 一致）；`:518` 记为待改的文档缺陷 |
| **C2** | 理论 note 用 `u_k=k/N`，代码与论文节用 midpoint | `docs/theory/EVQ_COSH_THEORY.tex:63,71,272,353` vs `scripts/lib/rope/learnable_evq.py:83`、`schedules.py:179-183`、`03_theory.tex:110` | 两者都对（不同对象），但**任何跨文档抄公式必须显式声明用哪一格**；sol02.md:104-109 已给量化警告 |
| **C3** | 摘要类 digest 把 E-L 解写成 `h*=C/(α+β(1−u²)^{1/2})` | `analysis/unify_20260910/digests/digest_theory-0910.md:158` vs 一级出处 `pro6_allocation_analysis.md` 二.2 节与 `ROPE_ALLOCATION_THEORY_CORE_20260910.md:104-107` | 该 digest 表达式**错误**：α 位置错、且 `(1−u²)` 应为 `(1−u)²`。正确：`h=√α/√(β(1−u)²+2λ)`。标 [叙事-未验证]，**不得引用** |
| **C4** | "Hilbert–Schmidt projection" 措辞在数学上不成立 | `EVQ_COSH_THEORY.tex:120` vs `docs/research/rope_allocation_20260910/agents/astra02.md` 的 HS 不可能性小节（L²[0,1] 上 I 不是 Hilbert–Schmidt ⇒ 不存在带非零 δ 系数的字面 HS 投影；Galerkin 修补给出 α=O(1/n)、β→6⟨K_L,G⟩_HS、**τ=√(β/α)~√n（固定 L），不是 n/√L**） | 若保留投影语言，必须声明有限维分辨率与归一化（核心文档 `:125` 已给同一提醒） |
| **C5** | "水床"同名不同物 | `EVQ_COSH_THEORY.tex:307-330`（Jensen 界，被界对象 = ∫ln I 的上界，取等 = 几何密度） vs `paper-2027/research/three_completions/evq_three_completions.tex:388-391`（`W(ρ_{τ_*})=O(d_head⁴/L_eff²)`，一个**可计算的尺度量**） | 引用时必须点名哪一个；两者不能互相代入 |
| **C6** | τ 的"操作点"律本身有争议 | 论文参考律 `τ=c·d_head/√L_train`，`a1_proofs.tex:453`（c=1）；自洽体系给出 `τ^sc≈1.43 = 0.51×` 部署律，且 `appendix` 的 failure 例（如 deployed τ=4 超出模型最优 2.32–2.43 的 1.6–1.7×，`evq_three_completions.tex` 的 `Prop failure`） | τ 律**未定**；任何 KKT 推导不得默认 c=1 就是最优 |
| **C7** | 跨文档"τ≈d_head/√L（'PASS'=脚本约定，9.6% 均值 / 33.3% 最大锚误差）"被列为叙事过量 | `analysis/unify_20260910/INTEGRATION_20260910.md` §5 dead-mechanism register | 该律只作为**参考约定**，不得作能力定理 |
| **C8** | 4.1 节 `J[ρ]` 的水床/稀疏性口径在不同文档里的"最优解"称谓不同 | `ROPE_ALLOCATION_THEORY_CORE_20260910.md:98`（"归一化 Cosh 是此问题的唯一最优解"） vs `E≠K−K_app` 有限窗下最优解是**原子测度** | 前者只在**投影后泛函**内为真；对精确 `E_L` 不成立 |

---

## 5. 这套机器要变成"冻结模型的最优分配"还缺哪一块（任务 4）

按材料给的四条缺口（每条带出处）：

**缺口 1 — 目标函数错位：cosh 只是投影泛函的解，精确有限窗的最优是原子测度。**
- 精确核 `K_L` 保留时，`sol02.md:126` 明确："There is no cosh ODE for this exact finite-window functional."（[已验证-原文]）
- `astra02.md` 的原子均衡定理：唯一极小元**有限支撑**；**没有任何正连续密度（含 Cosh）**最小化精确目标；KKT 势 `V_μ(ω)=∫K(ω,ν)dμ − qω²`，`V≥c` 于 I，`=c` 于支撑。CPU 实例：I=[0.1,1], b=10, L=8, q=0，支撑 1/2/3 时 E=.183907229/.058949724/.058639327，对偶间隙 .458261952/.004252824/4.2e-17；几何 ≈.13323829 vs Cosh τ=2 ≈.09518227（[部分证据-转述自 agent 文档]）。
- 精确有限表目标与**精确间隔梯度**（可直接用作 KKT 的手）：
  `E_{L,K}(φ)=(1/2K²)ΣΣK_L(φ_i,φ_j) − (μ_F/K)Σe^{−2cφ_i}`；
  `∂E_{L,K}/∂g_r = Σ_{m≥r}[ (1/K²)Σ_j∂₁K_L(φ_m,φ_j) + (2cμ_F/K)e^{−2cφ_m} ]`（`sol02.md:130-146`，boxed）。核导数可避开对 Ci 求导：`∂₁K_L(φ,ψ)= (cω(φ)/lnL)∫₁^L sin(ω(φ)t)cos(ω(ψ)t)dt`（`sol02.md:150-155`）。
  **这块是"可判决、无任意候选格"的现成替换目标**，也是把 J[h] 升级为 KKT 的最短路径。

**缺口 2 — 密度不可辨识（gauge）。**
- 内容系数自由时，`(ρ,a)` 是 gauge 对：`a₂=ρ₁a₁/ρ₂` 保所有已实现 margin 不变（`astra07` 的恒等式，转述自 `analysis/unify_20260910/digests_codex/digest_astra-evq-finite.md`）。故任何**无条件的**"唯一最优密度"论断须降级。
- EVQ 只在**等加载 w∝ρ + 共享相干噪声 + 常数被保护信号**三条同时成立时才可恢复；独立通道噪声给出的是 Neyman 律 `ρ*(x)=σ(x)|w(x)|/∫σ|w|`，**不是 EVQ**（同 digest；亦见 `ROPE_ALLOCATION_THEORY_CORE_20260910.md:161`："独立通道噪声不能自动推出 α∫ρ²"）。[部分证据-转述]

**缺口 3 — 冻结部署需要槽标签与内容系数，纯几何选不出槽。**
- `sol02.md:201-212` 给出正确的冻结形式：**位移预算有限窗步**，`min_φ E_{L_target,K}(φ) + ½Σ_i w_i(φ_i−φ_i^{(0)})²`，单调间隔 + 关键槽硬固定；
  **权重 w_i 必须从冻结模型测**（留出损失曲率，或受控槽扰动），**不得由余弦几何推断**（`sol02.md:212`，[已验证-原文]）。
- 原因：`E_q` 对槽标签是**置换不变**的（`digest_astra-evq-finite.md` 的 lineage 判决第 3 条），纯几何无法定位槽。正确目标须用**有符号的** source−distractor 矩（astra03/07/09）。
- 严格障碍：位置无关的可逆 Q/K 映射**不能**在开区间上精确改变冻结模型的旋转频率多重集（冻结移植不可能性，`sol02.md:188` 与 `:222`，引 `a1_proofs.tex:288-312`）。

**缺口 4 — 端点不变量必须作为硬约束 + 活跃集结构进入，而不是被推导出来。**
- I1（m_j=0, j≤23）/ I2（m_j=1, j≥40）是**端点不变量**，`analysis/unify_20260910/NEXT_DERIVATION_KKT_PROBLEM.md` §1.3；`docs/research/UNIFIED_BUDGET_ALLOCATION_THEORY_20260910.md` §8.3.2 明确**拒绝自由端点优化**：I1/I2 必须作为硬约束进入任何搜索。
- 另：`β(1−u)²` 的角色需要重新声明 —— 在**重学习**体制它是 representation prior（合法），而在**冻结部署**体制间隔代价是 compatibility cost，其符号与位置来自 bank/arc 二分，**不是** `(1−u)²`（`UNIFIED_BUDGET_ALLOCATION_THEORY_20260910.md` §8.1.2）。

**缺口小结（我的整理，标 [假设-整理性]）**：把 J[h] 变成冻结最优分配，缺的是一个**"目标替换 + 权重测量 + 槽标签"**的三角：目标换为 `E_{L,K}`（缺口 1），代价项换为实测位移权重 `w_i`（缺口 3），可行域加 I1/I2 硬约束（缺口 4），并且全程承认密度只在 gauge 意义下可辨（缺口 2）。J[h] 与 `Q_τ` 的价值在于它是这条链的**解析骨架**（唯一可闭式积分的特例），而不是终点。

---

## 6. 关键数字（复核过，带出处）

| 数字 | 值 | 出处 | 等级 |
|---|---|---|---|
| 有限 K 直方图遗憾常数 | `K²(J[ρ_K]−J[ρ_*]) → (α/24)(τ²−τ·tanhτ)`，斜率 −2 | `scripts/analysis/finite_k_cosh_regret_audit.py:9,100,117-137`；`tests/test_finite_k_cosh_regret_audit.py` | [已验证-代码+测试] |
| 传输界 | `W_∞≤1/(2Km)`，`W₁≤1/(4Km)`，`‖ρ_K−ρ‖₁≤B/(Km)`；EVQ 代入 `m_τ=τ/sinhτ`, `B_τ=τ²` ⇒ `W₁≤sinhτ/(4Kτ)`，`‖ρ_K−ρ_τ‖₁≤τsinhτ/K=τ²/K+O(τ⁴/K)` | `paper-2027/appendix/a1_proofs.tex:515-561`；`sol05/sol02.md:88-93` | [已验证-原文] |
| 高分辨率（Bennett）失真 | `D_K[ρ]=(1/12K²)∫w/ρ² + O(K^{−3})` | `paper-2027/appendix/a1_proofs.tex`（传输小节） | [部分证据] |
| 单交叉上界 | `φ_c(τ) ≤ 1−1/√3` | `paper-2027/appendix/a1_proofs.tex:373-398` | [已验证] |
| 自洽 τ 小 τ 展开 | `S_χ²(τ)=τ⁴/(45d_head)+O(τ⁶)`；`τ*²=45λQ₁(L,b)d_head²/L` | `paper-2027/appendix/a1_proofs.tex:485-495`；`evq_three_completions.tex` Part I | [部分证据] |
| 常数自由规则 | `λ=4/(15g_max)=0.6928`，`g_max=2/(3√3)=0.38490`，`c(Π)≤1` | `evq_three_completions.tex` 的 `Cor zeroconst` | [部分证据] |
| τ^sc 定点 | 64/√2152=1.380 (Geo substrate)、64/√1868=1.481 (EVQ)、定点 τ^sc≈1.43 = 0.51× 部署律 | `evq_three_completions.tex` 的 `Cor fixedpoint` | [部分证据] |
| 部署 τ 表 | 50M pred 2.828 vs 部署 2.830；MLA pred 0.322 vs 0.354；750M pred 0.949 vs 1.000；OLMo-2 pred 1.898 vs 2.000；Llama3-8B pred 1.287 vs 1.414；151.9M b=256 pred 2.318 vs 部署 4.000 | `evq_three_completions.tex` Part I τ 表 | [部分证据] |
| EVQ 恢复五臂 | Cosh τ=2 midpoint、Exponential λ≈1.47547、Hybrid r16 τ≈2.83714、官方 YaRN scale=4 gain 1.138629、共享 RMS 形变 ≈0.129239 | `analysis/unify_20260910/digests/digest_evq-code.md` §5.2；`experiments/evq_recovery/tables.py:35` | [部分证据] |
| 13 次完成的运行 + c=1 参考 | 99 次完成运行；c=1 参考在 7/9 配置均值、18/27 配对种子上胜过 midpoint 离散化 Geo 基线 | `paper-2027/appendix/a1_proofs.tex:451-512` | [部分证据] |
| MLA 431M 关键结果 | 16K PPL 138.8→95.6（−31.1%） | `paper-2027/sections/04_experiments.tex:16`；`paper-2027/appendix/a3_supporting_results.tex:39-48` | [已验证-论文] |
| 750M | 45.1→24.4 | `paper-2027/sections/04_experiments.tex:23` | [已验证-论文] |
| 部署坐标 | `ν_j=ω_j·S^{−m_j}`，`λ_j=S^{Δ_j}`，`Δ_j=m_{j+1}−m_j`，`Σ_{j=23}^{39}Δ_j=1`，`Σ gap_j=ln S=1.3863 nats` | `analysis/unify_20260910/NEXT_DERIVATION_KKT_PROBLEM.md` §1.3；`digests/digest_mrrope-evq.md` §3.5 | [已验证-权威文档] |

---

## 7. 死路（不得再试，含失败原因）

| 机制 | 失败原因 | 出处 |
|---|---|---|
| 把 **Σcos 首根 / 根理论** 当 F 或其选择子 | 根与能力排序失序（MrUni 82.2K>MrPro 80.3K 而 32K 64.6≪87.2；E2/P2 同根反向）；根 = 诊断量，禁入 F | `STARTING_POINT_YARN_VS_MRPRO.md:86`，红线 R1 |
| 把 **Σm** 当守恒量 | Σm 是自由决策变量；守恒只在指定坐标内成立，水床的守恒坐标是 ∫₀¹ln I 的上界 / ∫h du=1，不是 Σm | `INTEGRATION_20260910.md` §1 R4；`NEXT_DERIVATION_KKT_PROBLEM.md` §1.3；`ROPE_ALLOCATION_THEORY_CORE_20260910.md:47` |
| **"YaRN 递减 vs MrPro 递增"** 作机制解释 | 两者都凸都递增（m_Y′>0, m_Y″>0）；推断自 F1/F2 的叙事作废 | `STARTING_POINT_YARN_VS_MRPRO.md:9,21` |
| **hs 字面 Hilbert–Schmidt 投影** 导出 τ~K/√L | L²[0,1] 上 I 不是 HS；无带非零 δ 系数的字面 HS 投影；Galerkin 修补给出 α=O(1/n)、τ~√n（固定 L） | `EVQ_COSH_THEORY.tex:120`；`sol02.md:188`;`astra02.md` HS 小节 |
| **几何平滑度/覆盖率/有效秩/Gram/能量** 作 F 分项或选择子 | 静态几何代理，红线 R1；Smooth 87.2/68.3 vs MrPro 87.22/78.13 是反例 | `INTEGRATION_20260910.md` §5；`STARTING_POINT_YARN_VS_MRPRO.md:80` |
| **局部旋转扰动更小 ⇒ 短任务分更高** | OLMo 反例：MrPro 局部扰动降到 ~47.8%，但独立 72 条长端 MrPro 2.78% < YaRN 6.94% ≪ BM 51.32%；F3 材料自带"不得升级为能力定理" | `STARTING_POINT_YARN_VS_MRPRO.md:42,75` |
| **自由端点优化** | I1/I2 是端点不变量，任何搜索必须当硬约束 | `UNIFIED_BUDGET_ALLOCATION_THEORY_20260910.md` §8.3.2 |
| **用静态余弦几何推断 w_i**（冻结位移权重） | 权重必须从冻结模型测（留出损失曲率 / 受控槽扰动）；`E_q` 对槽标签置换不变，纯几何选不出槽 | `sol02.md:212`；`digest_astra-evq-finite.md` lineage 第 3 条 |
| **ALGORITHM 1（距离先验拟合 τ）作可靠取 τ 途径** | 项目记录 Algorithm 1 失败（离散化伪影），被 mini-sweep 替代 | MEMORY.md Phase 1-5 条目（项目记忆，[部分证据]） |
| **逐点 AM-GM 求 J[h] 的无约束最优** | `h∝1/(1−u)` 在 u=1 不可积，∫h=1 无法满足；必须带归一化乘子 | 我的复核（[假设-整理性观察]），依据 `pro6_allocation_analysis.md` 二.2 节的一阶条件形式 |
| **把水床当作"任何长任务收益必然伴随短任务损失"** | 原文明确不主张此意 | `ROPE_ALLOCATION_THEORY_CORE_20260910.md:117` |
| **保留截断导数泛函 `⟨ρ,h_c*ρ⟩≈A₀‖ρ‖₂²−(π⁴/720c³)‖ρ′‖₂²` 去优化** | 截断后在高端波数**无下界**，且正有限区间密度有端点项 | `sol02.md:173-180` |

---

## 8. 未决问题

1. **哪一个有限表才是"EVQ"？** 理论 note 的 k/N、代码/论文节的 (k+1/2)/K、750M 的 k/(K−1)、OLMo-2 的 k/K 四种并存；`table_allocation_protocols.tex` 如实记录但**没有统一裁决**。KKT 推导必须先冻结一格（`ROPE_ALLOCATION_THEORY_CORE_20260910.md:102` 已点名该问题）。
2. **τ 的操作律未定**：c=1 参考律 vs 自洽 τ^sc≈0.51× 部署律 vs Algorithm 1 拟合 —— 三者共存，`a1_proofs.tex:453` 只用参考律。
3. **精确 `E_{L,K}` 的 KKT 解**：需要的 μ_F、c、以及 `w_i` 从哪个冻结模型怎么测，材料只给了方法名（`sol02.md:212`），**没有给出一次完成的测量**。
4. **`E_{L,K}` 与下游任务的连接**：`sol02.md:216` 自述"does not prove downstream improvement"；哪些槽的关联是"critical"（需硬固定）**没有任何已测清单**。
5. **HS/有限维投影的确切声明**：若要保留 α、β 与 τ 的投影解释，需要声明维度 n 与归一化；astra02 的 Galerkin 结果（τ~√n）与论文的 τ~d_head/√L **不是同一个 n** —— 两者的 n 分别指什么，材料未澄清。
6. **C_app 与精确 K_L 的数值间隙量级**：`EVQ_COSH_THEORY.tex:116-125` 只给了 R² 经验值（0.96@512 → 0.87@16K），**没有给 α、β 的最优拟合值表**，也没有给"残差主导于边界与对角脊"的量化证据。
7. **水床在自洽体系中的 `W(ρ_{τ_*})` 到底是哪个对象**（C5）：`EVQ_COSH_THEORY.tex` 的 Jensen 界与 `evq_three_completions.tex` 的尺度量同名不同物，未见消歧。

---

## 9. 覆盖（读了什么 / 跳过了什么）

**逐行读过**：
- `docs/theory/EVQ_COSH_THEORY.tex:47-135, 220-335`（核、投影、Green 函数、cosh、CDF/分位、锚定、Taylor、水床、τ 律）；`:1-46` 与 `:336-378` 读过摘要/公式表。
- `docs/research/rope_allocation_20260910/source_inputs/pro6_allocation_analysis.md`（"二、EVQ 可以精确改写成间距预算优化"整节，1. 从密度改写到间隔 / 2. 最优条件直接推出 Cosh 分配）。
- `docs/research/ROPE_ALLOCATION_THEORY_CORE_20260910.md`（全文 171 行）。
- `docs/research/rope_allocation_20260910/agents/sol02.md`（全文 225 行）。
- `paper-2027/appendix/a1_proofs.tex:313-370, 447-470, 512-525`。
- `paper-2027/sections/budget_method.tex:1-30`。
- `scripts/lib/rope/learnable_evq.py:75-120, 224-310`；`scripts/lib/rope/schedules.py:158-200`。
- `experiments/evq_recovery/tables.py`（函数级）。
- `scripts/analysis/finite_k_cosh_regret_audit.py`（函数级 + 常数）。
- `paper-2027/tables/table_allocation_protocols.tex`、`paper-2027/appendix/a2_experiment_details.tex:129-133`、`paper-2027/appendix/a6_mature_scale.tex:10`。
- `analysis/unify_20260910/STARTING_POINT_YARN_VS_MRPRO.md`（全文 115 行）。

**读摘要/摘录而未逐行读原文**（在文中已标 [部分证据] 或 [叙事-未验证]）：
- `docs/research/rope_allocation_20260910/agents/astra02.md`（HS 不可能性、原子均衡 CPU 数值、τ~√n 结论 —— 读的是转述与摘要，未逐行复核其推导）。
- `paper-2027/research/three_completions/evq_three_completions.tex`（Part I 的自洽 τ 体系、τ 表、`Prop failure`、`Cor fixedpoint`、`Prop sign` —— 读的是摘录与行号定位，未逐行读 1006 行全文）。
- `analysis/unify_20260910/digests_codex/digest_astra-evq-finite.md`、`analysis/unify_20260910/digests/digest_evq-code.md`、`digest_mrrope-evq.md`（digest 层，二手）。
- `analysis/unify_20260910/NEXT_DERIVATION_KKT_PROBLEM.md`、`INTEGRATION_20260910.md`（权威文档，凭前序会话已读并引用其 §/行）。
- `docs/research/UNIFIED_BUDGET_ALLOCATION_THEORY_20260910.md`（§8.1.2、§8.3.2 两处）。

**明确未读/跳过**：
- `paper-2027/research/three_completions/` 下 `optimization_notes.md`、`verify_optimizations.py`、`verify_three_completions.py`。若 kernel 解析推导要延伸，建议下一轮补 15 分钟。
- `paper-2027/appendix/budget_proofs.tex`（只经由 sol02 的 source anchors 知道其存在与作用域）。
- `docs/research/EVQ_NONLOCAL_KERNEL_CORRECTION_20260910.md`（sol02 §3 的 `K_L=(c·min(φ,ψ)−γ+h_c(φ−ψ))/lnL+E_L` 分解、`ĥ_c(k)` 乘子、以及"截断导数泛函无下界"的原文，我读的是 sol02 的转述）。
- `~/.codex/**` —— **严格未写入、未读取**（遵守只读纪律）。
- 除本文件外，**未修改仓库任何文件**。

---

## 10. 给下一轮的一句话接口

`J[h]=½∫[α/h+β(1−u)²h]du, ∫h=1` 是**可用的解析骨架**（唯一可闭式积分 ⇒ cosh），其 KKT 手是"预算乘子 λ + 归一化"；但要接上"冻结模型最优分配"，最短路径是**把目标换成 `E_{L,K}(φ)` 的精确有限表形式 + 精确间隔梯度 `∂E/∂g_r`**（sol02.md:130-146），把 I1/I2 当硬约束，把位移权重 `w_i` 从冻结模型**测**出来 —— 这条路径上的三个参数（μ_F、c、w_i）目前**一个都没有已测量的值**。
