# A1 — astra01–astra05 数学内容挖掘

挖掘对象：`docs/research/rope_allocation_20260910/agents/astra01.md` … `astra05.md`（codex 30 代理归档；原文在 `.agents/rope_unification_20260910/`，本地已不存在）。
权威对照：`analysis/unify_20260910/{INTEGRATION_20260910.md, NEXT_DERIVATION_KKT_PROBLEM.md, STARTING_POINT_YARN_VS_MRPRO.md}`。
本文件只做提取，不做推导。证据等级：[已验证]=本轮独立复算通过；[已验证-源文件]=核对过被引源文件的对应行；[部分证据]；[假设]；[叙事-未验证]。

---

## 0. 覆盖度（读了什么、没读什么）

| 项 | 状态 |
|---|---|
| astra01–05 全文 | ✅ 逐行读完（283/236/247/92/260 行） |
| 权威文档 3 份 | ✅ NEXT_DERIVATION、STARTING_POINT 全文；INTEGRATION §4–§9 重点段 |
| 已有 codex digest | 仅用于交叉索引，未作为证据（`digests_codex/digest_{astra-margin-lineage,astra-evq-finite,calibration}.md`） |
| 本轮独立复算 | ✅ astra01 三频反例、astra02 原子均衡 + §7 Galerkin 表 + 光滑对照能量、astra03 slot28 反例、astra04 padded-KL 恒等式 + 映射计数 + 块分解、astra05 投影表 |
| 源文件核对 | ✅ `docs/theory/EVQ_COSH_THEORY.tex`、`scripts/analysis/tau_static_vs_dynamic_experiment.py`、`experiments/nongeometric_screen/pro_block_calibration.py`、`docs/exp/2026-02/2026-02-27_evq_tau_sweep_results.md`、`docs/research/ROPE_ALLOCATION_SUBSPACE_DERIVATION_20260910.md`、`docs/research/EVQ_NONLOCAL_KERNEL_CORRECTION_20260910.md`、`docs/research/ROPE_CARRIER_REMOVAL_PILOT_20260907.md` |
| **未读 / 不可达** | astra06–09、sol01–19（不在本次任务范围）；`.agents/rope_unification_20260910/` 全部（目录已删，`joint_mode_candidates.py/.json`、coverage 回执、`full_model_response_native.jsonl` 均不可核）；`results/nongeometric_screen_20260909/` 下 `development_summary.md`、`planned_controls/*.json`、`queue/0440_Smooth_MrBudget.json`、`planned_controls/p2_gap_comparison.json` 在本地**均找不到**；Capon 1969 原文；MrRoPE 论文 Markdown（只在 astra01 引文内出现） |

**覆盖度警告**：astra02/03/04/05 的大量关键数字来自 `results/nongeometric_screen_20260909/` 与 `.agents/rope_unification_20260910/`，这两处在当前工作树中**不存在**（仓库 09-06 瘦身只留 `paper-2027/ docs/ scripts/ tests/`；`.agents/` 已归档删除）。所以下面凡标 [源不可达] 的条目，只能作为"报告自称"接受，不能上升为已验证事实。

---

## 1. astra01 — margin 谱系（信号 vs 干扰）

### 1.1 中心命题

**单一无符号余弦能量不能作普适分配目标；正确的共同对象是"角色条件化的带符号 source-versus-distractor margin 及其协方差"，分配律是广义匹配滤波 ρ*∝C⁻¹h。**（astra01:20–35, 74–115）

作者自我定位：这不是新优化器，是经典结构（Capon 1969 匹配滤波谱估计）；本报告的实质贡献是"指认正确的角色条件化 h,C 与使分配正当化的假设"（astra01:107）。

### 1.2 精确定义与假设

| # | 内容 | 出处 | 等级 |
|---|---|---|---|
| D1 | EVQ 精确余弦核 `K(x,y)=E_{Δ~D}cos(ω(x)Δ)cos(ω(y)Δ)`，`ω(x)=b^{−x}`；二次能量 `⟨ρ,Kρ⟩=E_D B_ρ(Δ)²`，`B_ρ(Δ)=∫cos(ω(x)Δ)ρ(x)dx` | astra01:9–18 | [已验证-源文件]（与 `EVQ_COSH_THEORY.tex:89–100` 的 `K(φ1,φ2)=∫_1^L D(Δ)cos(ω(φ1)Δ)cos(ω(φ2)Δ)dΔ` 一致） |
| D2 | 角色场 `Z_e(x)=Re{A_{+,e}(x)e^{iω(x)Δ_{+,e}} − A_{−,e}(x)e^{iω(x)Δ_{−,e}}}`；`h(x)=E Z_e(x)`，`C(x,y)=Cov(Z_e(x),Z_e(y))` | astra01:41–52 | [假设]——h,C 由声明的响应场规定，非从模型导出 |
| D3 | 测度 η 下：`M_η=∫Z_e dη`，`μ_η=∫h dη`，`v_η=∬C dηdη`（式(1)） | astra01:53–60 | [已验证-推导]（对该规定模型精确；C 由构造 PSD，含跨频带符号对消） |
| D4 | 无量纲量 `t=μ/√v`：Gaussian margin ⇒ `Pr(M≤0)=Φ(−t)` 精确；无分布假设 ⇒ Cantelli `Pr(M≤0)≤v/(v+μ²)=1/(1+t²)` | astra01:64–70 | [已验证-推导] |
| D5 | 阈值 a 可吸收为 h→h−a（因 η(I)=1）；常数 K 归一不改排序（记账，非增益干预） | astra01:62 | [已验证-推导] |
| D6 | 多 distractor：union bound 求和各 pairwise 界；"对单个随机 key 正均值"不足以对抗 128K 竞争者 | astra01:70 | [已验证-推导] |

### 1.3 分配律（式(2)–(5)）

```
max_{ρ≥0, ∫ρ=1, ⟨h,ρ⟩>0}  ⟨h,ρ⟩ / √⟨ρ,Cρ⟩            (2)
  ⇔ (齐次化) min_{w≥0, ⟨h,w⟩=1} ½⟨w,Cw⟩,  ρ=w/∫w      (3)
C⁻¹h ≥ 0 且积分正 ⇒   ρ*(x) = [C⁻¹h](x) / ∫[C⁻¹h]     (4)
positivity 绑定 ⇒  active-set KKT:
     Cw ≥ λh,  w ≥ 0,  w(Cw−λh)=0,  ⟨h,w⟩=1           (5)
t*² = ⟨h,C⁻¹h⟩   （Cauchy–Schwarz in C-内积）
```
出处 astra01:78–105；等级 [已验证-推导]。

**关键限定（astra01 自己写明）**：`C⁻¹h` 有负分量时**直接裁剪一般是错的**（:99）——必须解 active-set。这与 INTEGRATION §4.2 的 astra10 修正条"不能直接裁剪代替 active-set 求解"一致，**astra01 已先行给出正确形式**（[已验证-源文件]，INTEGRATION:70）。

**多角色版**（:109–115）：最大化 `min_r t_r`；固定 t 时约束 `‖C_r^{1/2}ρ‖ ≤ ⟨h_r,ρ⟩/t` 与质量/非负同为凸集 ⇒ 可行性二分给全球最优。受保护频段贡献固定均值/协方差交叉项，剩余测度上仍仿射。

### 1.4 EVQ 与 Cosh 的回收（含失效条件）

- **EVQ 回收**：nuisance 模型 `Z_e(x)=h₀+ξ_e cos(ω(x)Δ_e)`，`Eξ=0, Eξ²=1`，ξ⊥Δ ⇒ `h=h₀`（常数）、`C=K_exact`；此时"最大化标准化 margin"**恰好等价于最小化 EVQ 的二次能量**（:118–126）。等级 [已验证-推导，条件式]。
  作者限定：**"this is a restricted model of a protected target against nuisance sidelobes, not a general model of a remote content source"**（:126）。
- **Cosh 回收**：再加宽带协方差假设 `C_app=αI+βG, G(x,y)=min(x,y)` ⇒ `αρ+βGρ=λ` ⇒ `ρ''−τ²ρ=0, ρ'(1)=0` ⇒
  `ρ*(x)=τ cosh(τ(1−x))/sinh τ`，`τ²=β/α`；尾形 `v=α∫(T')²+β∫T²`，`T(x)=sinh(τ(1−x))/sinh τ`（:128–153）。等级 [已验证-推导]。
- **非恒定信号**：`αρ''−βρ=λh''`，`αρ'(1)=λh'(1)` + free-boundary ⇒ **不再是 Cosh**。作者原话：系数与 forcing 必须来自声明的角色总体，"inventing them to recover a desired profile is circular"（:154–161）。

### 1.5 astra01 的明确边界（作者自陈，三条 + 若干）

1. 精确振荡核 K **不自动等于** Brownian+identity C；历史独立计算发现精确核最优带截断密度与训练域外的碰撞强度（:165，引 `rebuttal/STRONG_MODEL_THEORY_VERDICT_20260720.md:27–40,50–75,194–223`，**该文件本轮未核**）。"This report does not restore those withdrawn equivalences."
2. `τ=√(β/α)` 是本模型内的协方差比，**不是**经验 `d/√L` 律，"unless that ratio is separately demonstrated"（:166）。→ 与 INTEGRATION §5 死亡登记"τ≈d_head/√L"一致 [已验证-源文件]。
3. 保护性常数信号假设、频率平稳协方差假设**都不从泛型 RoPE 推出**；律的可检验性恰恰在于改变 target lag/role 会改 h 并可能反转预测（:167）。
4. §7：native-window 重相位**不能**从 native-only 观测辨识长上下文 h,C——两个响应族可以在所有 lag ≤ W 上一致而在 W 之外有用源系数反号（:256）。`tests/test_native_windows.py:17–33` 证明的是"缓存干预在该特殊构造下保全后续窗口形成"，**不是**与完整连续长 prefill 等价（:258）。
5. §8：把 MrPro 变成 (4) 的解只能靠反向构造 `h=Cρ_Pro`——"inverse-optimal-control identity with no predictive content; this report explicitly rejects it"（:264）。结论：MrPro 是同一 margin 问题里的**启发式成员**，不是另一个精确优化器。

### 1.6 astra01 的 CPU 解例（本轮全部独立复算通过）

- **三频同支撑反例**（Δ=2π，A=(1,½,¼)、B=(1,0.9,¼)，均正、严格有序、共端点）：
  - `B_A=0`，`B_B=+1.8090169943749475`；EVQ 二次能量 A=0 vs B=3.2725424859373686。
  - 错误 key 模型（margin=B，Gaussian σ=√3）pairwise error：A **0.5** vs B **0.1481417537951759**。
  - 位置模型（margin=3−B）pairwise error：A **0.041632258331775196** vs B **0.24584783176990438**。
  - **结论**：同一 lag 的语义角色一变，正确排序反转 ⇒ 单一无符号余弦能量不能做普适目标。
  - 等级：**本轮 [已验证]（四位有效数字全中）**；但作者注明"exact Gaussian score-model calculations, not LM outcomes"（:35）。
- 800 点中点离散化解 `I+4G`，与解析 τ=2 Cosh 密度最大误差 2.986e−7（:280）。
- 光滑协方差 `C(x,y)=0.3e^{−|x−y|/0.15}+min(x,y)`、h=1+0.4x、Cosh 密度：K=16/64 等分位对应均值误差 3.37e−4 / 2.55e−4，方差误差 1.333e−3 / 5.03e−4；解析界 1.25e−2 / 3.125e−3 与 0.1875 / 0.046875（界正确但松）（:281）。作者自注"validate formulas, not model quality"。

### 1.7 astra01 §5 白噪声的正名（对 F 建模有直接约束）

- `αI` 必须解释为**频率场白噪声**：`Var(∫ρ dW)=α∫ρ²`（:174）。
- **K 个独立通道噪声做不到这件事**：其平均贡献 `K⁻²Σ_k v(x_k) ≃ K⁻¹∫v(x)ρ(x)dx`，v 常数时为 v₀/K（线性于密度，:181–184）。
- 可实现替代：PSD 三角核 `k_ε(x−y)=ε⁻¹(1−|x−y|/ε)_+`，`C_ε=αk_ε+βmin`；K 个等原子时对角贡献 `α/(Kε)` ⇒ **ε→0 在 K 固定下发散，有限通道与白噪声极限不可交换**（:192）。
- 等级 [已验证-推导]；与 INTEGRATION §5"iid 噪声→α∫ρ²"死亡条一致。

### 1.8 astra01 §6 有限-K 分位实现界

中点分位取原子 `x_k=F⁻¹((k+½)/K)` ⇒ `W₁(η_K,ρdx) ≤ R/(2K)`（:198–213，无需 ρ 正下界）；端点等权版 `W₁≤R/K`。
h 为 L_h-Lipschitz、C 逐参 L_C-Lipschitz、d=W₁ ⇒ `|μ_K−μ|≤L_h d=:e_μ`，`|v_K−v|≤2L_C d=:e_v`；
μ>e_μ 时 `t_K ≥ (μ−e_μ)/√(v+e_v)`（式(8)），并给出有限-K Cantelli 界（:215–237）。
**作者自评：在 K=64 与 128K 可能很弱**，因为相位导数随 lag 增长、窄脊协方差 `L_C ~ α/ε²+β`；"No small error is claimed without evaluating the actual constants"（:239）。

---

## 2. astra02 — 连续原子解与有限窗

### 2.1 核心定理（本批 5 份报告中最硬的数学）

**定理（唯一有限原子均衡）**：取 `I=[a,b]`，`0<a<b<∞`，`1<L<∞`，`p(t)=1/(t log L)` 于 `[1,L]`，`q≥0`：

```
f_μ(t) = ∫_I cos(ωt) dμ(ω)
K(ω,ν) = ∫_1^L p(t) cos(ωt) cos(νt) dt
E_q(μ) = ½ ∫_1^L p(t) f_μ(t)² dt − q ∫_I ω² dμ(ω)
```
则 `E_q` 有唯一极小解 `μ*`，**其支撑有限**。特别地，**没有任何正连续密度（含 Cosh）能极小化精确目标**。
出处 astra02:11–47；等级 [已验证-推导] + **CPU 均衡点本轮独立复现**。

**证明思路（四段）**：
1. **存在性**：紧区间上概率测度弱紧，K 与 ω² 在 I、I² 上连续有界 ⇒ E_q 弱连续。
2. **严格凸性**：任一非零有限带号测度 η，`∬K dηdη = ∫_1^L p(t) f_η(t)² dt > 0`。若为零 ⇒ 连续性+p>0 ⇒ `f_η≡0` on [1,L] ⇒ 紧支撑使 `f_η(z)` 为复平面整函数 ⇒ 唯一性定理 ⇒ `f_η≡0` ⇒ 偶化后 Fourier 唯一性 ⇒ η_even=0 ⇒（I 与 −I 不交）η=0，矛盾。**这就是必须声明正频率支撑的理由**（cos ω/−ω 退化）；线性 Fisher 项不破坏严格凸。
3. **KKT 势**：`V_μ(ω)=∫K(ω,ν)dμ(ν) − qω²`，`c_μ=∫V_μ dμ`。朝点质量 δ_ω 的方向导数为 `V_μ(ω)−c_μ`；最优满足
   ```
   V_μ*(ω) ≥ c_μ*   ∀ω∈I
   V_μ*(ω) = c_μ*   on supp(μ*)
   ```
   先 μ*-a.e.，由连续性提升到拓扑支撑；凸性使该不等式组充分（:36–41）。
4. **有限性**：`V_μ(z)` 对复频率整（t 区间有界）。若 supp 无限 ⇒ 紧性给 I 内聚点（端点也是复解析域内点）⇒ 唯一性定理 ⇒ `V_μ*≡c`。Riemann–Lebesgue 给 `F_μ(ω)→0`（ω→∞）。q>0 时 `V_μ` 不能为常数；q=0 时常数必须为 0 ⇒ `p f_μ` 余弦变换为 0 ⇒ `f_μ(0)=μ(I)=1` 与 `f_μ(0)=0` 矛盾。故 `V_μ−c` 是**非零整函数，在 I 上只有有限零点**，supp 有限。∎

**作者自陈**："This is exact mathematics of the declared collision model, **not** a discovery that a language model ought to collapse its channel table."（:47）

### 2.2 原子数随 L 无一致界（:49–57）

对固定 `0<a<b`、q=0，最小必要原子数随 L→∞ 无界。记 `A_L(z)=∫_1^L cos(zt)/t dt`；由乘积恒等式得 `K_L(ω,ν) ≥ −C/log L` 一致、`K_L(ω,ω)=1/2+O(1/log L)` 一致。≤M 原子测度有 `E_0(μ) ≥ 1/(4M) − C'/log L`（用 `Σweights²≥1/M`）；而 M+1 个等质量不同频率的能量趋于 `1/[4(M+1)]`。故 L 足够大时击败一切 ≤M 原子测度。**"no L-independent atom-count bound exists"**。等级 [已验证-推导，渐近论证]；作者注明"constants can depend on the fixed interval; this is not a joint a→0 limit"。

### 2.3 构造性 exchange 算法 + 证书（:59–77）

1. 维护有限频率与非负质量和为 1；
2. 精确解受限严格凸权重 QP；
3. 找连续势 `V_μ` 在 I 上的全局最小点 ω_new；
4. 若 `g(μ)=c_μ−min_I V_μ` 小则停 —— 凸性给 `0 ≤ E_q(μ)−E_q(μ*) ≤ g(μ)`；
5. 否则加入 ω_new、重优化质量、可选细化位置。

固定支撑 `x_i` 与正质量时解线性 KKT 系统：
```
[ K_AA  −1 ] [ p_A ]   [ q x_A² ]
[ 1ᵀ      0 ] [  c  ] = [   1   ]
```
位置细化满足内点 `V'(x_i)=0`、端点单边不等式。**端点原子允许；不存在光滑密度的 Neumann 条件**（:77）。等级 [已验证-推导] + **本轮数值复现**。

### 2.4 CPU 解例（本轮独立复现，用 SLSQP 而非报告的 mask 枚举）

声明：`I=[0.1,1]`，base=10，`L=8`，`q=0`；96 点 Gauss–Legendre 于 [1,8]。从单原子 0.55 出发。

| 支撑数 | 报告 E | 报告对偶 gap | 本轮复现 E |
|---:|---:|---:|---:|
| 1 | .183907229150896 | .458261951556866 | .183907229150897 |
| 2 | .058949724094609 | .004252824362431 | .0589497240946106 |
| 3 | .058639327401539 | 4.2e−17 | .0586393274015407 |

最终频率 `[0.1, 0.4066617223165344, 1.0]`（本轮 `[0.1, 0.406661736582, 1.0]`，差 1.4e−8）；
质量 `[0.0987464879705222, 0.4002375621647298, 0.5010159498647481]`（本轮前 8 位一致）；
全局 gap 上界 1.53377e−8（本轮同值 1.533765e−8），本轮 10001 网格上 `c − min V = 9.7e−16`，且三个原子处 `V = 0.11727865480308142` **完全相等** ⇒ KKT 支撑等式条件成立。
对照：log-均匀密度 E≈.13323829017761（本轮 .1332382901776104）；Cosh τ=2 E≈.09518226679009（本轮 .09518226679008959）。
**结论：精确有限窗碰撞泛函的测度最优是原子，定性区别于两种光滑分配。** 等级 **本轮 [已验证]**。
作者限定："No relevance to Qwen task ranking is inferred from those lower energies."（:94）

有限 K 是**不同约束**：`μ_K=(1/K)Σδ_{ω_j}`，原子质量须是 1/K 的整数倍。K=8 中点量化给多重数 (1,3,4)，能量 .05892945173956（本轮 .05892945147079193，一致到 9 位），**高于**测度最优 .05863932740154，但"one feasible construction, not a globally solved K=8 optimum"（:98）。严格互异使域开、可失 attainment；显式最小间距使其紧但引入新架构约束（:100）。作者定论：测度解是有限-K 问题的**下界与初始化器，不是它的解**（:102）。

### 2.5 astra02 §3 源锚定推不出普适中频带

- `E_q` 只依赖**无标签**频率测度；冻结 checkpoint 依赖槽位与带号 Q/K 内容、值、后续层。交换槽位关联不改 E_q 但可改计算 ⇒ **单靠 E_q 推不出普适的槽位特异中频规则**（:106–107）。
- **两时钟特例（精确）**：`f_r(d)=Σ_j[A_rj cos(ω_j d)+B_rj sin(ω_j d)]`。局部族须精确保留 ⇒ `f_r^dep(d)=f_r(d)` on 开区间；长程族须精确重定时 ⇒ `f_r^dep(Sd)=f_r(d)`。不同指数频率的线性无关性迫使：**排他性局部分量用原生 ω_j、排他性重定时分量用 ω_j/S；被两种不相容时钟共享的分量使联合需求不可行**。退化系数签名允许置换，须排除（:108–112）。
- 结论句：**"a native block plus common-PI block has a mathematical justification when useful computations separate that way. The middle is the set of conflicting or coupled active computations, not automatically an interval determined by one rotation count."**（:114）等级 [已验证-推导]。
  → 这条对 K1 是硬约束：三段结构的**中段位置不可能由单一"圈数"公式决定**，只能由耦合计算集决定。与 STARTING_POINT F5（交点随 S 移动 Qwen38→Llama25）兼容但要求 S-依赖来自计算耦合，而非公式。

### 2.6 astra02 §4 冻结 QCQP（带真实任务后果的构造）

对一条 attention row 冻结条件 Q/K 系数，取有用 key k*：`M_r(x)=ℓ_{k*}(x)−ℓ_k(x)`；位移 v 下每个槽项 `C cos(z)+D sin(z)`，`z=ν_j^r e^{−v_j}d`；`∂/∂v_j = −z[−C sin z + D cos z]`，`|二阶导| ≤ √(C²+D²)(|z|+z²)`。在 `|v_j|≤r_j` 上求和给显式 `M_rj≥0`，Taylor 展开给**严格的**下界
```
margin_r(x⁰+v) ≥ margin_r(x⁰) + g_rᵀv − ½ Σ_j M_rj v_j²
```
最大化最小认证 long-margin 增益 η，约束含已用源的阈值、long-margin 不等式、盒界、实际 log-频率序/端点 ⇒ **凸 QCQP**（凹二次的超水平集 + 线性目标）。无任意相邻光滑罚、无自造通道效用、无新范数当能力选择子（:118–129）。

**实际后果**：若 N−1 个竞争 margin 都 ≥m>0，则有用 key 是唯一 argmax 且 attention 概率 `≥1/[1+(N−1)e^{−m}]`；对以该 argmax 为输出的声明式 pointer task，正确性成立（:131）。**作者限定**：全模型还改上游状态并做 value mixing/readout，所以"this conditional theorem cannot certify its generation"。

### 2.7 astra02 §7 有限分辨率投影（对 τ 标度律的关键否证）

**问题**：`EVQ_COSH_THEORY.tex:120` 的"Hilbert–Schmidt projection"在无穷维 L2[0,1] 上不成立——I 不是 HS 算子（`Σ_n 1=∞`），故对任意 α≠0，`‖K_L−αI−βG‖_HS=∞`，**不存在带非零 δ 系数的字面连续 HS 最小二乘投影**。等级 [已验证-推导]；[已验证-源文件]（该措辞确在 tex 的 "Broadband projection" remark 中）。

**建设性修补**：先声明分辨率。把 [0,1] 分 n 等份，基 `e_i=√n·1_{C_i}`，
```
A_ij = n ∬_{C_i×C_j} K_L(φ,ψ)      B_ij = n ∬_{C_i×C_j} min(φ,ψ)
min_{α,β} ‖A−αI_n−βB‖_F²  ⇒  正规方程
   nα + β tr(B) = tr(A)
   α tr(B) + β‖B‖_F² = <A,B>_F
```
（α,β≥0 时须用非负最小二乘/KKT；负 α 不得代入 Cosh 泛函。）bin 常数密度 p_i 的系数是 `√n p_i` ⇒ 拟合能量 `(αn/2)Σp_i² + (βn/2)pᵀBp`。**因子 n 必须记账：离散质量空间对角系数 d 对应 α=d/n，不是 α=d**。B 的精确迹 `1/2−1/(6n)`。固定 L、base，n→∞ 时 `tr(A)→∫_0^1K_L(φ,φ)dφ ≤1`、`tr(B)→1/2`、`‖B‖_F²→1/6`、`<A,B>_F→<K_L,G>_HS`。
⇒ `β→6<K_L,G>_HS`，而内部拟合为正时 `α=[tr(A)−βtr(B)]/n = O(1/n)` ⇒ **`τ=√(β/α)` 在固定 L 下按 √n 发散，不是自动的 n/√L**；若 positivity 绑定则 α 可为 0、Cosh 强制凸性假设直接失效。投影分辨率 n 与实际通道数 K 是**两个独立选择**，取 n=K 是另一个声明性建模决定（:151–176）。

**数值表（base=10, L=8）——本轮独立复现**：

| n | 报告 α | 报告 β | 报告 n·α | 报告 τ | 本轮 α | 本轮 β | 本轮 n·α | 本轮 τ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | .00709041638147 | .858718721249 | .113446662104 | 11.004988569 | .007088727197952 | .858718721249 | .113419635167 | 11.006299688 |
| 32 | .00329366373151 | .870290110287 | .105397239408 | 16.255202923 | .0032932357451201 | .870290110287 | .105383543844 | 16.256259145 |
| 64 | .00157961736196 | .875477427278 | .101095511166 | 23.542171940 | .001579509727617 | .875477427278 | .101088622567 | 23.542974057 |

β 12 位全同；α 相对差 ~1e−4（我的 bin 内积分点数较少）；**n·α 缓降趋常、τ∝√n 的定性结论成立**（τ/√n = 2.75→2.87→2.94）。等级 **本轮 [已验证]**。
作者自评："This is numerical refinement of one declared projection, not a search of task candidates."（:236）

**astra02 对项目脚本的批评（[已验证-源文件]）**：`scripts/analysis/tau_static_vs_dynamic_experiment.py:71–83` 的 `fit_broadband` 用 `alpha = np.mean(np.diag(K_mat)) * dphi`——**未减去拟合 min 核的对角贡献**（diag(K)=α+β·φ，故高估 α），且其有限 log-距离求积不是精确 Ci 求值。我核到该行确为 `alpha = np.mean(np.diag(K_mat)) * dphi`。作者推论："The old script's final dichotomy between a static exponent and training dynamics is broader than what that numerical experiment can prove."（:236）

### 2.8 astra02 的自我限定（逐条）

- :3 无 GPU/模型执行、无部署表、"does not assert downstream superiority"。
- :47 是**声明碰撞模型的**精确数学，不是 LM 应塌缩通道表的发现。
- :89–94 原子最优是"one modest interval, not a deployment search"；低能量**不推断** Qwen 任务排序。
- :98 有限 K=8 构造"one feasible construction, not a globally solved K=8 optimum"。
- :102 测度解**不是**有限-K 解。
- :114 中频带不是由圈数决定的区间。
- :131 条件 QCQP 是"a mechanistic allocation proposal with an explicit missing assumption, not a whole-model theorem"。
- :133 "The rule is not yet computed for Qwen: the required content-conditioned useful contrasts are not supplied by geometry or raw projection norms."
- :137 原子最优**不得**因那些几何理由成为另一个冻结赢家提案。

---

## 3. astra03 — softmax 信息投影 / 保标签有限尺度优化器

### 3.1 结果陈述

给定 checkpoint 的**带号源-对-干扰系数分布**，存在具体冻结分配准则：在真实有限目标 lag 上最大化标准化 source margin 的**下界**，把 native 操作概率要求作为**硬约束**。变量是既有带标签频率槽；均值/协方差**保槽身份**。**无几何效用奖励、无二次 native 系绳、无任意混合系数**（astra03:5）。

作者限定（:7）："This is a conditional allocation algorithm, not a universal table or a demonstrated Qwen improvement... The empirical coefficients needed to emit a defensible new Qwen table have not been measured in this subtask. The prior Q/K projection-matrix Gram is not that measurement."

**核心机制**：**信号与干扰的内容条件不同**（:9）——"EVQ's collision quadratic is obtainable as an interference covariance under exchangeable co-adaptation assumptions. MrRoPE's positive matched-content response concerns the signal mean. **It is invalid to minimize one aggregate response in one regime and maximize it in another without specifying which population generates it.**"

### 3.2 精确带标签双线性对象（:25–47）

```
R(θ) = [[cosθ, −sinθ],[sinθ, cosθ]]
A_j = q_{j0}k_{j0} + q_{j1}k_{j1}
B_j = q_{j1}k_{j0} − q_{j0}k_{j1}
q_jᵀ R(ν_j d) k_j = A_j cos(ν_j d) + B_j sin(ν_j d)
```
尺度：g 同时乘 cos/sin 表时 head logit 尺度 `γ=g²/√d_head`；非旋转贡献可作额外常数特征。
```
c_r      = γ (A⁺_{r1}, B⁺_{r1}, A⁻_{r1}, B⁻_{r1}, …)
f_r(x)   = (cos u⁺_{r1}, sin u⁺_{r1}, −cos u⁻_{r1}, −sin u⁻_{r1}, …),  u^±_{rj} = d_r^± ω_j e^{−x_j}
Z_r(x)   = c_rᵀ f_r(x)      ← logit contrast 精确
```
`x_j=log(ω_j/ν_j)` 是带标签位移。**只置换表**改 f 而不置换 c ⇒ 一般改 Z；**联合置换完整槽**（f 与 c 同置）才不改 Z。这是正确的"保标签"性质，**不只是保持频率序或多重集**（:45）。等级 [已验证-推导]；与 INTEGRATION §5"联合置换频率+学习系数槽才是恒等"一致。
作者限定：完整 transformer 中 `c_r` 本身依赖已装表（经前层与前缀），"Freezing c_r for every x is an assumption, not an identity"（:47）。

### 3.3 生成式假设与非平凡保证（:49–91）

**条件矩包络假设**（对每个操作类型与 lag 格 r，在声明区域内的所有可容许表下）：
- `‖E c_r − μ_r‖₂ ≤ ε_r`；
- 协方差 PSD 上界 `Σ_r=L_rL_rᵀ`；
- 期望源标签在声明的内容/布局输运后仍指正确证据。

作者限定：这些**不是**"native 样本能建立长上下文不变性"的主张；允许相关槽、相关 sin/cos 系数、源/干扰相关；单靠 source-native 系数样本**不**建立该包络（:57）。

```
M̲_r(x) = μ_rᵀf_r(x) − ε_r‖f_r(x)‖₂
S_r(x)  = ‖L_rᵀf_r(x)‖₂
z_r(x)  = (M̲_r(x) − η_r) / S_r(x)
```
η_r 是**操作化 logit-margin 要求**（非偏好权重）：硬源排序取 η_r=0；M 个 key 中软源质量 ≥1−a 时充分 pairwise margin
```
η_r = log[ (M−1)(1−a) / a ]        ← 与 INTEGRATION §4.2 的"稀释项四家同式"一致
```
保证：`Pr[Z_r≤η_r] ≤ 1/(1+z_r²)`（Cantelli）；更强 sub-Gaussian MGF 包络下改进为 `exp(−z_r²/2)`；精确 Gaussian 时 `Φ(−z_r)`（ε_r=0）。M−1 个干扰用 union bound：z≥6 且 M−1=1000 时概率 ≥1−1000e^{−18}>0.9999847 ⇒ "the guarantee is not formally true but numerically vacuous by construction"（:89）。
完整答案：若额外假设"条件在正确前缀上，每个必需 routing 事件写入的 value 其下游 decoder 在 margin 阈值成立时保留正确 next token"，则 T 个事件的联合成功 ≥1−Σ_t b_t。**作者明确称这是受限 copying/readout 定理，泛型 LM value/MLP/EOS 不自动满足**（:91）。

### 3.4 有限分配规则 (F) 与凸内层求解器 (I)

```
(F)   max_{x∈F} min_{r∈R_L} z_r(x)   s.t.  z_r(x) ≥ z̄_r  (r∈R_N)
```
- 可行集 F：固定采样端点 + 保序 `ω_j e^{−x_j} ≥ ω_{j+1} e^{−x_{j+1}}`（log 变量下线性）；可选 `x_j∈[0,log S]` 排除频率加速。
- **现用 Qwen 面**：`x_j=0 (j≤23)`、`x_j=log4 (j≥40)`，**16 个内部槽 j=24…39 为变量**（:105）。作者自注："Keeping that face is a scoped continuation of MrPro evidence, not a universal 32-turn/one-turn theorem."
- 在相邻混合基坐标 `e_j=x_{j+1}−x_j` 下总跨度 `Σe_j=log4`；均匀与渐进 MrRoPE 是两种特例预算分配。**(F) 不推导 MrPro 的二次累计剖面**："Such an exact-profile derivation requires additional assumptions on the means/covariances; reverse-engineering them to make MrPro optimal would be circular."（:107）
- 目标无混合系数，输出**可以不可行**，从而暴露声明需求间的不相容（:103）。

**凸内层（:111–156）**：在参考 `x⁰` 的盒 `|v_j|≤r_j` 上，
```
u = d ω e^{−x};  一阶导 (u sin u, −u cos u);  二阶导欧氏范数 √(u²+u⁴)
U^±_{rj} = |d_r^±| ω_j exp(−x_j⁰+r_j)
B_{rj} = √( (U⁺)²+(U⁺)⁴ + (U⁻)²+(U⁻)⁴ ),   Q_r(v) = ½ Σ_j B_{rj} v_j²
‖f_r(x⁰+v) − f̂_r(v)‖₂ ≤ Q_r(v)                 （Taylor 积分余项 + 三角不等式，盒上全局）
```
认证约束（固定候选可靠度 t≥0，每个 target 格）：
```
μ_rᵀf̂_r(v) − η_r  ≥  ε_r‖f̂_r(v)‖₂ + t‖L_rᵀf̂_r(v)‖₂ + (‖μ_r‖₂+ε_r+t‖L_r‖₂) Q_r(v)      (I)
```
左边对固定 t 仿射、右边凸 ⇒ 凸二次约束锥可行性问题（epigraph 可表为二阶锥）；对 t 二分求局部内近似下的最大认证可靠度。**关键**：第一余项下界精确均值、ε 项覆盖估计/输运不确定、末项 `t‖L‖Q` 经三角不等式上界精确标准差 ⇒ **(I) 蕴含 `M̲_r(x⁰+v)−η_r ≥ t S_r(x⁰+v)`（用精确相位）**，即使 S=4 也是**有限步**结论而非无穷小替代（:146）。

**算法（:148–156）**：从记录的 MrPro 表出发、幅度固定 → 由当前正 margin 所需的余项分辨率选半径（非候选网格）→ 二分解 (I)，取最小欧氏范数可行 v **仅用于打破求解器平局**（词典序，非效用正则）→ 在 `x⁰+v` 上评估精确三角 margin，记录精确与认证 t → 以正认证改进为条件迭代。**性质**：若 v=0 在某个 t 可行，则每阶段有可行 incumbent，认证 target 可靠度非降且所有声明 native 界保持。**作者限定**：这只建立第一个单调性；"Local convergence, global optimality, and actual language-model task improvement are distinct claims."（:156）

**一维特例（精确）**：学习操作均值 `A cos(νD−θ)`、常数 nuisance 标准差 σ、阈值 η 时，达到可靠度 t 要求
```
ν ∈ ⋃_{n∈ℤ} [ (θ+2πn−arccos((η+tσ)/A))/D , (θ+2πn+arccos((η+tσ)/A))/D ]
```
（arccos 参数在 [−1,1] 内时）。与正频率区间、native 角色区间求交后二分 t。作者点题："This exact finite-scale allocation illustrates why **individual first-zero horizons are not universal**: a slot's correct content phase θ and admissible phase branch matter."（:166）——直接否证"首零视界"作为普适选择子，与红线 1 一致。

### 3.5 astra03 §7 用真实 Smooth/Mr 频率的算术反例（本轮全部复现）

**这是逻辑反例，不是事后解释，也不是新部署候选**（:184）。参数：
```
Qwen native slot28: ω = 10^(−6·28/64) = 0.0023713737056616554        ← 本轮 [已验证] 精确一致
MrPro slot28 周期 3035.3259630259545 ⇒ ν_M = 0.002070019952952862    ← 本轮 [已验证] 精确一致
Smooth ν_S = 0.0022761875297874212                                    ← [源不可达]（queue/0440_Smooth_MrBudget.json 本地缺失）
d₀ = 26590, D = 4d₀ = 106360；均值系数 (cos(ωd₀), sin(ωd₀)) ⇒ 期望 target 响应 cos(νD − ωd₀)
```
| 表 | Native-template matched target mean | 本轮复现 |
|---|---:|---:|
| MrPro | +0.9994600706262113 | ✅ 完全一致 |
| Smooth | −0.9995409168024664 | ✅ 完全一致 |

加独立各向同性系数噪声 σ=0.1（投影后标准差仍 0.1，因 cos²+sin²=1）⇒ Mr 标准化 margin ≈ **+9.995**、Smooth ≈ **−9.995**。Gaussian 下两者排序概率近 1 / 近 0；仅用矩时 Cantelli 认证 Mr 成功 >99%，对 Smooth 的负均值**无有用界**。**同一批固定全表仍保持公开发表的无符号 distortion 序（Smooth 更优）** ⇒ 观察到的无符号序与"标签保持的计算序反转"相容，且在真实 32K→128K 有限尺度内（:195–197）。
lag 由 `d₀∈[8192,32767]` 内穷举整数算术选出以展示最清晰反例；**"It was not a frozen experiment prediction."**

一槽最优（邻近相位分支 n=25）：`ν* = (ωd₀+2π·25)/106360 = 0.0020697109769935414`（本轮精确一致），`m* = 0.09814685885` vs MrPro `m≈0.09803918082`；target mean = 1，native lag26 template mean 0.9999692420。作者限定：**"It is not evidence that this tiny adjustment helps Qwen, nor an endorsement of its direction over E1."**（:206）

**退化检查（:209–214）**——四条"什么时候定理空转"：
1. 若所有操作都是纯长程内容匹配、无局部/序要求，零频率或全 PI 可赢——**局部角色的缺席是实质性的，不是优化器 bug**。
2. 若所有系数均值为 0，仅靠降协方差推不出正 margin 定理。
3. **若噪声协方差各向同性且跨所有 sin/cos 对独立，频率改变在固定 lag 下不降其方差**——碰撞下降必须有相干跨槽或 lag 条件 nuisance 结构，**不能从独立各向同性干扰推断**。
4. 若某个置换使所有 μ 与 Σ 不变，则该模型下槽标签真的可交换；此时不能从该模型主张冻结标签敏感性。
5. 若实际表改动上游激活到包络之外，概率陈述不再适用，更小的固定系数残差救不回来。

### 3.6 astra03 §9 softmax 信息投影恒等（"root 的 identity"复核）

固定频率表与特征 f_j，`p_c(j)=exp(cᵀf_j+b_j−A(c))`。若有限无约束 c* 极小化 `KL(p‖p_c)`，则 `E_p f = E_{p*} f`，且
```
KL(p‖p_{c₀}) = KL(p‖p_{c*}) + KL(p_{c*}‖p_{c₀})
```
（把右式从左式减去剩 `(c*−c₀)ᵀ(E_p f − E_p* f)=0`）。**不需要特征线性无关，但需要有限极小解存在、偏移/支撑固定**。c 上有凸约束且 c₀ 可行时由一阶最优性变成对应的 ≥（Pythagorean 不等式）；可分情形下确界可能只在无穷系数处达到，须极限论证。（:233–241）等级 [已验证-推导]。
**作者限定**：这只把"表征拟合"与"固定系数相容性"在 softmax 指数族内精确分离，是比任意平方-logit 拟合更好的分解；**它本身不选冻结频率表**——目标 p 必须编码合法源选择，且特征与上游状态随表改变。"native 同位置 p 在长长度下不自动是正确的输运目标。它只能作解释或拟合诊断，不能替代 (F) 的带符号 target。"（:243）

### 3.7 astra03 的证据边界与 CPU 检查

- 引用四条外部证据（:15–21）：`paper-2027/DOCUMENT_TEXT_MAP.md:1795–1864` 的 Geo-trained/Geo-runtime vs Geo-trained/Cosh-runtime PPL 7.14 vs 76.20（positional rank 4.57→12.54），反向 weights-by-table crossing 也失败；`ROPE_ALLOCATION_SUBSPACE_DERIVATION_20260910.md:145–203` 的 Smooth/MrPro 远程弱子空间能量 0.049435/0.235928、native-range unweighted distortion 35.3793/42.5180（**本轮 [已验证-源文件] 数字全中，见 §6.2**），以及 :215–278 的逐 lag Q/K operator MSE 与逐层加权弱响应（**前提是独立单位二阶矩输入向量**——"these are projection-weight calculations, not the checkpoint's content-conditioned activation distribution"）；`results/nongeometric_screen_20260909/development_summary.md:1–31` 的 Smooth −9.79pp、E1 slot28 +5.21、P2 +3.54 long / −14.31 short（**[源不可达]**，本地该目录不存在）。
- "A head-routing theorem must not be renamed an answer-generation theorem"（:21）——引 750M 比较：8K 上两法 NLL-gap retrieval 均 100%，而 strict autoregressive exact match 0%/77.5%。
- **CPU 检查**（:247）：Taylor 余项用 1000 组随机 16-槽扰动，带号 lag −106360/−70000，半径 0.01，seed 7303，**最大 精确余项/界 比 = 0.8346442448**（界有效但不算紧）。等级 [报告自称，不可复现——代码未落盘]。

---

## 4. astra04 — 校准映射与全行校准

### 4.1 主结果：6Pro 目标的精确修正

`6Pro` 是有效的固定态位置拉伸蒸馏损失，但**不含额外 key 竞争**。在显式 distractor-不变性假设下，精确修正是
```
L_full = D_KL(T ‖ Q_O) + log(1 + Z_D/Z_O)
```
其中 T 为原生 key O 上的 teacher、`Q_O` 为候选在 O 上的 softmax 条件、`Z_D, Z_O` 为候选指数 logit 和。**它等于"teacher 以零填充"与增广 student 的 KL**（:7–12）。等级 [已验证-推导] + **本轮数值验证**（T=[.7,.2,.1]、O logits [1,−.5,.2]、D logits [.4,−.7]：`KL(T‖Q_O)=0.09247652027729411`，`log1p(Z_D/Z_O)=0.3628199272041319`，和 = **0.45529644748142606**，与 padded KL 完全一致）。
**作者限定（:12）**："It is not a theorem that unrelated natural-text blocks deserve zero teacher mass."

### 4.2 映射修正（**对仓库脚本的实质修正**）

事实（`astra04:15–17`，[已验证-源文件]，我核过 `experiments/nongeometric_screen/pro_block_calibration.py:22–30` 确有 `fit_docs 0:4 / validation 4:6 / M=4096 / S=4`、`block_size=4096`、`length=32768`、`selected_heads=[index%4+4*i for i in range(4)]`、`qpos=length−1`、key-minus-query 带号分离、精确 sin/cos、score multiplier `module.scaling*common_gain**2`，末尾跑 `original(...)` 即**存的是 native/common-gain 态，不是候选重算态**）：

- 原写映射 `f(bM+r) = S·bM + r`，`W=BM=32768`、`B=8`、`M=4096`、`S=4`，**最大索引 118783，不是 131071**。
- 修正：`f(bM+r) = (S−1)M + S·bM + r`，插入**前导 12288 槽 + 七个块间 12288 槽 = 98304 个干扰槽**，末端正好 131071（总长 131072）。
- **本轮 [已验证]**：naive max = 118783；corrected max = 131071；八个 gap 全为 12288，和为 98304。
- 共同平移保每个原 query-key 位移不变。作者限定（:18）：**"This correction does not turn stitched raw states into an actual 128K forward pass."**

### 4.3 精确分解与局部支配

把原 key 分到 8 个 source block，`p_b=Σ_{k∈b}T_k`、`q_b=Σ_{k∈b}Q_{O,k}`、`T_b,Q_b` 为归一化条件分布，则
```
D_KL(T‖Q_O) = D_KL(p‖q) + Σ_b p_b D_KL(T_b‖Q_b)
```
（**本轮 [已验证]**：toy 2 块实例 LHS=RHS 至 7e−18。）
⇒ `p_local=0.999` 的 teacher **允许任意差的远块条件保持**而平均损失可忽略。**"Equalizing conditional block terms is an explicit scale-priority choice, not a correction to the native probability distribution."**（:27）

具体无答案目标（:29–35）：
```
L_bal = D_KL(p‖q) + B⁻¹ Σ_b D_KL(T_b‖Q_b) + log(1 + Z_D/Z_O)
```
每个条件由 block log-softmax 直接算（含小质量块以免下溢）；等块权是"均匀 source-block 评估先验"的声明；保留 native block-mass KL（全局压平 T 会主动破坏局部注意力结构）。零当且仅当原 key 概率匹配 T 且干扰质量消失（有限 logit 达不到后者）。**要求分别报告普通 full KL、每个 p_b、条件 block KL、干扰质量**——"balanced-loss improvement alone cannot establish useful far relations"。作者限定："Nearly uniform low-mass block teachers might just encode noise; this is a real limitation"（:35）。
**恒等映射 guard**（:37）：在 `f(p)=p, D=∅` 上算同一 block-balanced 原支撑损失，要求其拟合聚合不差于 MrPro 的值 ⇒ 具体地保一个已有参考，而非引入短/长损失间的自由权重。注意 source native teacher gain 是 **common gain，不是 operational Native gain1**。

### 4.4 额外干扰的构造（可不加 GPU 捕获）

每篇 fit 文档的 Q 配**自己**的 32K K 作 O；98304 缺口用**其余三篇 fit 文档**的 K 填，逐层、逐正确 KV-head 独立，内部 token 序保持连续块；对三种循环 donor 序取平均以免 donor 身份与距离混淆（小规模精确对称平均，非 mapping grid）；**不得用 validation 文档当 fit donor**（:41）。holdout 只有 2 篇文档 ⇒ 同一非重复三 donor 构造不可能，诚实选项：(a) 用 docs4,5 验原支撑位置拉伸、另报重复块增广（显式为**改变的 donor 分布**）；(b) 用已冻结 fit donor bank 配 holdout query，披露只有 source query/原文档是 held-out（**不是**全新文档干扰 holdout）。**不得把复用 donor 块悄悄算作独立观测**（:43）。

**两条语义边界**（:45）：
- donor K 都来自 native RoPE 下的不同前缀 ⇒ query/donor 相容性与前层状态不同于真实插入；零填充 teacher 假定 insert 应被忽略。**反例**：与某个期望 key 完全相同的 donor 必然参与竞争，若其 V 也相同则 attention 输出可不变而 penalty 为正 ⇒ **该 penalty 测的是"原 key 身份保持"，比"功能保持"更严**。
- 反之，把原 K/V 复制四份时，重复 logit 相同时 marginal softmax 恰好等于 T ⇒ **此时没有理由收 factor-4 干扰罚**。增广语义决定正确 teacher。

### 4.5 具体约束分配器

参数化 `x_j = −log(ν_j/ν_ref)`，`x_0` 与 `x_63` 固定为 MrPro 端点；线性不等式 `x_{j+1}−x_j ≥ ε`（ε 是数值序容差，非频率先验）；因果比较时另固定 `Σ_j x_j = Σ_j x_j^Mr` ⇒ **剩 61 维**；无预设拐点、无带号逐槽标签、无几何粗糙度目标（:49）。
从 MrPro 出发，极小化平均 fit `L_bal`（对声明的 block-stretch 干预 + 干扰填充），服从上述恒等映射 guard；文档等权 → 层等权 → 保存 head 等权；流式处理完整 key 与 logsumexp，不为所有文档存逐对系数。用**精确有限相位目标 + 约束优化器（如 SLSQP 带解析一阶导）**，**不是一次性 Taylor 步**。冻一张表后只评 docs4,5。**不得用任务答案搜优化器检查点**。若求解器在满足约束下不能改进，**保留 MrPro 并报告"声明的校准问题未给出方向"**（:51）。

梯度（:53–65）：
```
z = σ Σ_j ( C_jk cos(d'_k ν_j) + D_jk sin(d'_k ν_j) )
∂z_k/∂x_j = σ d'_k ν_j [ C_jk sin(d'_k ν_j) − D_jk cos(d'_k ν_j) ]
平衡原支撑项对原 z_k 的导数:  (q_b − p_b) Q_b(k) + B⁻¹( Q_b(k) − T_b(k) )
干扰罚对 O: Q_aug(k) − Q_O(k);  对 D: Q_aug(k)
```
求和后与有限相位导数缩并 ⇒ **保留经完整求和 logit 与 softmax 的跨对耦合，无 pair-additive 损失假设**。
冻结后用 `v = x_fit − x_Mr`；需要反向对照时把 `+v` 与 `−v` 同因子 ≤1 缩放以保序；两向都保端点与累计压缩。用**完全相同 gain** 在新生成的独立全模型长任务上评三档。

### 4.6 astra04 的数学检查与阻碍

- **CPU 标量检查**（:71）：padded KL 对 `KL(T‖Q_O)+log1p(Z_D/Z_O)` 差 0；映射端点与 98304 槽计数亦核对。**本轮全部 [已验证]**。
- **解析梯度检查**（:83）：在 seeded 3-block/21-key/5-frequency CPU fixture 上与中心差分比，最大绝对误差 **2.3480405640652346e−10**。等级 [报告自称，脚本未落盘]。
- **阻碍清单**（:73）：单查询/文档的捕获**不能**辨识跨 query 位置的行为、也不能推广 query-local vs block-local；层/块等权表达的是测试优先级、**不提供因果任务重要性标签**；native/common-gain 态可能已不同于 native/gain1；完整 prefill 候选态须单独评。历史 `ROPE_BM_CROSS_CACHE_RESULT_20260908.json` 记录 source-cache 依赖的 MK2 结果与 VT 原/缓存续写不一致（**[源不可达]**）。
- 附加源限定（:85–90）：`ROUND11_OLMO_RESULTS_20260905.md` §9 明确把早期机制叙事降为假设（终止改善与含答案子串不足以精确生成；"LoRA 只修格式"与严格答案改善冲突）⇒ "Apply the same restraint to calibrated attention KL"；`experiments/rope_operator_family/results/20260909_gpu/REPORT.md` 记录 output-KD 自然 NLL 3.942717 vs score-fit 9.989853 而 score fitting 的位置响应误差更小，两者都错过冻结检索答案（**客观迁移警告**）；`experiments/rotary_budget/DESIGN_SOURCE.md` 附录 B 的 48 条件 softmax fit 中 E16 在窄响应赢 G16、在 4096 宽响应输，U16 在声明的等宽混合中赢 E16（**声明的校准先验必须显式**）；跨文档 donor 含人工 native 文档起始/sink 态，在缺口重复可造出真实连续 128K 中不存在的额外 sink 竞争 ⇒ 增广拟合异常时须报告 donor sink 行为，**不得在观察到损失后悄悄删难 donor key**。

### 4.7 astra04 的统一性范围（作者自陈）

"分配坐标把 EVQ 与 MrRoPE 精确统一为**表坐标**。本损失是一个**提议的冻结相容性估计器**。它**不**从真实任务损失推导 EVQ 的解析间距代价，也**不**证明从头训练最优。构造性规则可证伪：它在查看目标答案之前预测一个分配。在冻结 stitch 行成功后于 dense-128K 失败，具体地**证伪这个估计器的可迁移性，而不是频率分配自由度本身**。"（:77）

---

## 5. astra05 — 联合模式输运算子（role-conditioned nonlinear modes）

### 5.1 结果

EVQ 的平方余弦目标与 MrRoPE 的正余弦目标可以是**同一个"信号 vs 干扰 log-partition margin"的不同项**，而不是同一核在同一 lag 上的两个普适指令。相干匹配受益于正均值分数；零均值相干 nuisance 招致指数矩代价，二阶上由**其平方核**支配。**关键反例：独立各向同性干扰坐标不产生该平方核代价——它们的分数方差旋转不变。内容假设是统一的一部分，不是可省去的细节。**（astra05:5–7）

构造性推论：精确分配
```
ν(S) = (I − P_R) ω + S⁻¹ P_R ω,      P_R = Rᵀ(RRᵀ)†R
```
R 的行是"已测量、经角色资格"的整数频率关系。它精确拉伸选定的联合模式、精确保留每个正交关系，并服从尺度复合律（:9–15）。

### 5.2 §1 先于谱近似的精确对象

固定 query/head/layer、pre-RoPE Q/K 固定，`d_t` 为 key t 的带号相对位置：
```
z_t(ν) = b_t + Σ_j { a_tj cos(ν_j d_t) + b_tj sin(ν_j d_t) }      （系数已含 attention scale 与固定幅度）
Z_A(ν) = Σ_{t∈A} e^{z_t(ν)},   M(ν) = log Z_S(ν) − log Z_D(ν),   p(S|S∪D) = σ(M)
```
**这些恒等式不需要 Fourier 平稳、独立、小幅、小表改变假设。** 但若 S∪D 遗漏其他 key，这是**条件质量**；全局 p(S) 还依赖所有被省略 key——对全局注意力权重极小的模式，这个区分很重要（:34）。等级 [已验证-推导]。
作者限定：LM 中"有用源"不自动是注意力最高的 key；其身份须来自任务的 source relation 或证明源使用的干预。"Even then M is an attention observable, not a theorem about emitted answers: V、W_O、其他 head、后续层、decoder margins 仍在。"（:35）

### 5.3 §2 EVQ/Mr 之桥及其必要限制

令 `Kν(d)=K⁻¹Σ_j cos(ν_j d)`。风格化检索行：一个确定性信号分数 `μKν(d_s)`（μ>0），distractor 距离 iid 自 p_D，其分数 `ξ_t Kν(d_t)`，`ξ_t~N(0,σ²)` 跨 key 独立 —— **这是"每个 key 内部的共享相干 nuisance"：同一标量乘整个频率和**（:41）。
```
E e^{z_D} = E_{d~p_D} exp{ ½σ² K_ν(d)² }
M̄(ν) = μ K_ν(d_s) − log N − log E_{p_D} exp{ ½σ² K_ν(d)² }
−M̄ = log N − μ K_ν(d_s) + ½σ² E_D K_ν(d)² + O(σ⁴)      （|K|≤1 弱相干噪声时）
```
这是**log 信号质量减 log 期望干扰质量**，**不是 E log odds**。独立有限矩干扰下经验 partition 随 N 趋近期望；Jensen 给 `E log Z_D ≤ log E Z_D` ⇒ 确定性信号下 `E M ≥ M̄`，但**该期望不等式不保证某个实现或精度**（:56）。等级 [已验证-推导]。
解读（:65–69）：
- **Mr 型项**：最大化信号距离上的正匹配内容余弦均值。Mr 的根诊断在其 §4.4（markdown 782–795）；其实际渐进规则假定 arithmetic radix increments（§3.2.2, line 420），**正余弦论证并不唯一导出那些增量**。
- **EVQ 型项**：在 lag 测度下压制相干干扰碰撞。它**在频率求和之后**平方。lag 测度、内容相干性与信号约束必须声明。
- `−log N` 项是干扰多重数；**仅重定时相位并不消掉它**；gain 只在有利的信号/干扰 margin 下有用，且会放大错误极大值。
作者限定（:71）："This derivation does not turn the current Cosh surrogate into the exact optimizer of M̄." 并指出 `paper-2027/sections/03_theory.tex:74–110` 显式选取密度判据 `α∫ρ²/2+β∫S_ρ²/2` 并导出 Cosh minimizer，其 α,β **不被本桥辨识**。

### 5.4 §2 反例：对"泛型噪声解释"的否证

若 distractor 系数是**独立 Gaussian 正交分量** `A_j,B_j ~ iid N(0,σ²/K)`，则
```
Var z_D(d) = σ²,   E exp z_D(d) = exp(σ²/2)     对 ν 与 d 都无关
```
**EVQ 式平方和消失。** 同理独立各向同性 Q/K 向量给旋转不变的分数分布。⇒ **"random distractors produce squared cosine collisions" 为假，除非声明相干协方差结构。**（:75–82）一般 Gaussian 系数 u（均值 m、协方差 C）：`log E e^{uᵀx_ν(d)} = mᵀx_ν(d) + ½ x_ν(d)ᵀC x_ν(d)`：相干秩一协方差给 K²；各向同性 C 给常数；中间与学到的 C 产生实际联合和/差（含 sin 相位与跨槽协方差）。**"Q/K projection Frobenius norms alone do not determine C under real activations."**（:90）等级 [已验证-推导]；与 INTEGRATION §5"iid 噪声→α∫ρ²（旋转不变，无碰撞项）"及 astra01 §5 一致。

### 5.5 §3 非线性联合模式（保角色、保归一化）

对小旋转块 B，逐 key 贡献写作 `Σ_{j∈B} r_tj cos(ν_j d_t−φ_tj)`，留下精确剩余分 `z_{t,−B}`：
```
e^{z_t} = e^{z_{t,−B}} Σ_{n∈ℤ^|B|} c_{t,n} e^{i(nᵀν_B)d_t},   c_{t,n} = Π_{j∈B} I_{n_j}(r_tj) e^{−i n_j φ_tj}
H_{A,n}(κ) = Σ_{t∈A} e^{z_{t,−B}} c_{t,n} e^{iκ d_t},   Z_A = Σ_n H_{A,n}(nᵀν_B)
```
**位置可变系数也有效**：这是每个 key 各自的展开，不是自然 prompt 的平稳 Fourier 模型。精确模式对 M 的频率导数：
```
∇_{ν_B} M = Re Σ_n n { H'_{S,n}(nᵀν_B)/Z_S − H'_{D,n}(nᵀν_B)/Z_D }
```
**这是敏感度，不是有限变化预测**；对提议的有限输运须直接重算 Z_S、Z_D。**关键是：同一谐波可因带号角色对比、内容相位与占据位置而增大或减小 margin。光靠 |c_n| 或长周期不能选择保全/拉伸/压制它。**（:117）等级 [已验证-推导]。
平稳特例 `H_{A,n}=c_{A,n}Σ_{t∈A}e^{iκd_t}`：对区间是精确 Dirichlet 核（含混叠）。若某慢谐波在快速平均后支配角色差异，则决定其功能的是**低维角色划分而非原始模式幅度**。截断误差控制：`|error|≤η_A Z_A`（η_A<1）时 log-margin 误差至多 `−log(1−η_S)−log(1−η_D)` ⇒ 数值截断规则，**不是学习到的能力阈值**（:119）。

### 5.6 §4 有限构造性分配（投影输运算子）

**最小位移解**：
```
min_ν ‖ν−ω‖₂²   s.t.  Rν = Rω/S
P_R = Rᵀ(RRᵀ)†R,   ν = (I − P_R)ω + P_R ω/S
```
性质：行空间内每个 n 满足 `nᵀν = nᵀω/S`；每个与行空间正交的 q 满足 `qᵀν = qᵀω`；**`T_R(S)T_R(U) = T_R(SU)` ⇒ 同一固定 R 的重复施加无标度路径歧义**；改 R 或内容分布则破坏该复合解释（:134–138）。
作者限定："这声明了什么被保留、什么被拉伸，而不给每个频率分配自由的目标尺度或权重。**它不主张欧氏频率位移就是模型损伤**；它只是选出实现指定模式变换的唯一最小干预。Native 功能代价仍须精确评估。"（:140）

**两频包络输运**：`n=(1,−1)`，`c=(ω₁+ω₂)/2`，`g=ω₁−ω₂` ⇒
```
ν₁ = c + g/(2S),   ν₂ = c − g/(2S)
```
拍频被拉伸而和/载波被保留，**其中一个频率向上移动**。**纯压缩盒 `ν_j≤ω_j` 排除这个精确保载波解**；该盒下最小位移解变成 `ν₂=ω₂, ν₁=ω₂+g/S`（ω₁>ω₂）——**拉伸拍频但移动载波**。⇒ "a blanket ban on any acceleration is a substantive mechanism restriction, not merely a harmless implementation convention."（:150）等级 [已验证-推导]。与 INTEGRATION §4.2"carrier-preserving 形式打破 ν_j≤ω_j 的压缩-only 盒"一致。

**三频曲率输运**：`n=(1,−2,1)`，`κ=ω_j−2ω_{j+1}+ω_{j+2}` ⇒
```
ν_B = ω_B − (1−S⁻¹) κ (1,−2,1)/6
```
块均值与一阶线性斜率不变，曲率/二阶差分频率被除以 S ⇒ 中段分配的变化**交替变号**，"It is not generic smoothing and need not be well represented by a monotone movement-exponent ramp."（:160）

**CPU float64 公式网格检查**（`b=10⁶, K=64, S=4`；**不是存储的 FP32 初始化器**）—— **本轮独立复现全部一致**：

| 关系 | 旧周期（报告 / 本轮） | 新周期（报告 / 本轮） | 相对槽变化（报告 / 本轮） | 范数比（报告 / 本轮） |
|---|---|---|---|---|
| slots 36−37 | 76740.5661 / 76740.5661 | 306962.2644 / 306962.2644 | −7.2809%, +9.0352% / −7.28092%, +9.03517% | .10690 / .10690 |
| slots 28−2×29+30 | 70286.2105 / 70286.2105 | 281144.8420 / 281144.8420 | −.471216%, +1.169499%, −.725638% / −.47122%, +1.16950%, −.72564% | .01069 / .01069 |

作者限定：两个块在本检查中仍严格递减；但"**Adjacent untouched slots, positivity, and every finite functional cost must be checked for any full table; no general order guarantee follows from a projector.**"（:169）存储的 Qwen 周期 70285.94 与公式网格有小的初始化器舍入差（源 owner 在 187–190 行已记录，**[源不可达]**）。

**统一性**：R 满秩时全 PI 被恢复；R 张成某个坐标块时块 PI 被恢复 ⇒ 相干块压缩与稀疏关系输运是**同一精确有限族的成员**。若 R 满秩，保载波的自由度消失 ⇒ 跨全空间的显著交互**可迫使全局 PI，暴露真实的局部-长程冲突**（:183）。

### 5.7 §5 如何在不发明关系权重的前提下辨识 R

最小观测契约：固定 pre-RoPE Q/K、显式 source relation S、完整竞争 key 集 D、source-to-target 粗坐标映射。合成 source-use assay 上这些由生成规则已知；自然文本上需可辩护的源标注或反事实。`scripts/experiments/source_only_generation_guard.py` 的 source-only 条件有用，**注意力幅度本身不是**（:187）。
关系资格三条件（缺一不可）：(a) 纳入整行剩余划分后仍有实质归一化贡献；(b) 其符号/相位在 source-only 配对世界中支持源-干扰分离；(c) 预期的坐标扩张确实要求它被重定时。**"A relation merely present in Q/K biases fails this contract."**（:189）
若**没有**关系满足契约 ⇒ 结果是 **"no justified mixed-mode intervention"**，不是"因为周期好看而选一个任意三元组"。**"Do not pick a harmonic, cutoff, sign, or S by maximizing the already exposed 128K answers."**（:191）
判决充分性机制比较：**一个经角色资格的输运 vs 一个等规模正交干预**，先看精确归一化角色 odds，再看实际生成答案。正交扰动保 `nᵀν` 而改载波；秩一干预以最小位移改该关系 ⇒ 区分"所选拍频重要"与"任意局部表扰动都有帮助"。**native/局部对照必需**，因为精确谐波重定时并不保所有原始 logit。作者明确：**本轮未跑任何模型干预**（:193）。

### 5.8 §6 现有失败对构造的约束

- `ROPE_ALLOCATION_SUBSPACE_DERIVATION_20260910.md:151–204`：Smooth 相对 MrPro 改善 source-weak 暴露与 unweighted distortion 却 128K 开发输出更差；:208–281 进一步确立投影算子加权与加权 source-weak 能量**都不能修复排序**（Smooth 在三个 cutoff 的每一层都更优）。⇒ **排除"靠降低 source-weak 总能量 / 算子距离 / 频率范数来选 R 或表"**（:197）。
- source weak projector N 仍可作**约束诊断**：输运后评 `tr(NG_far(ν))` 与精确局部 feature/logit 行为；**它不是"极小化即有用"的目标**。"A small source eigenvalue does not authorize independently moving participating channels."（:199）
- bias-only 审计 `results/nongeometric_screen_20260909/planned_controls/bias_harmonic_head27_8_audit.json:2–34`（**[源不可达]**）：按最大 native (1,−2,1) bias 系数选 layer27/head8，但 **Native 有效质量全在第一个 target 四分位**；MrPro 超出该四分位的约 **2.576e−8**；Smooth 约 **2.055e−7**。⇒ **"Large coefficients inside a separately normalized block therefore do not establish global far attention."** 该作者限定：该 head 的局部 KL 差异可能仍重要，但审计**不**指认实际远证据通路（:201）。
- **载波剔除 pilot**（`docs/research/ROPE_CARRIER_REMOVAL_PILOT_20260907.md`，全文已读）：背景导数目标下降 **73.85%** 同时伴随严重长 VT/UUID 退化 ⇒ "Preserving frequency differences while altering the carrier was not enough there."（:203）**本轮 [已验证-源文件]：该文件第 117 行确写"能量下降73.85%，与能力退化同时存在，明确证明该构造目标改善不等于功能改善。"**
- 明确不采纳的历史主张（:205）：assigned handoffs 中的普适闭合或"all vetoes watertight"；"concentrated residual 不证明与实测主导 loss-Hessian 特征向量对齐"（挑战文本断言该因果对齐却未测量）；**单圈边际相位覆盖永不证明联合相位安全**；`rope_transport/nullband.py` 开头"被包裹通道无论移到哪里都无条件安全"的断言被混合模式与该文件后续联合轨迹段明确否证。

### 5.9 §8 续作：全部相邻最低阶过渡候选（29 个）

实现 `.agents/rope_unification_20260910/code/joint_mode_candidates.py`，产出 `joint_mode_candidates.json`（**[源不可达]**）。族定义：slots 24–39 内**完全支持的 15 个相邻差分 + 14 个相邻二阶差分**（明确是所请求的族，非额外任意网格）。每个候选从**已验证的实际 FP32 MrPro 表**出发，目标是对应实际 Native 时钟除以 4：
```
nu_c = nu_M + n * (nᵀ ω_native / 4 − nᵀ nu_M) / (nᵀ n)
```
作者列出的冻结元数据：Native tensor SHA `138c99b1…80f6e`；Mr tensor SHA `33cbe3a4…016f`；Gain 1.138629436111989 不变；`Mr sum(m) = 29.333333268998935`；29 个候选全部有限、正、严格递减、逐位保留所有外部槽与两端点、过 FP64 投影恒等式与显式 FP32 舍入界；独立标准库精确有理算术与生成器理想频率差 0；最大部署相对关系时钟误差 1.0494258591124796e−05。等级 [报告自称，JSON 不可达]。

**"对朴素时钟推理的重要更正"**：**这些 Mr 关系中没有一个是已经把 Native 重定时为 4 的**。中段靠前的相邻一阶差分**比 Native 快**，尽管每个原始频率都被减慢；二阶差分周期大多也变短。⇒ 该族**确实改变实际联合时钟**，不是既有 Mr 斜坡的伪装复现。"That observation establishes the intervention, not its utility."（:224）**本轮 [已验证]**：由报告给出的 iid 周期比列，比值 <1（即比 Native 快）的三个恰是 `d1_s24_25 (0.939106)`、`d1_s25_26 (0.924643)`、`d1_s26_27 (0.919962)`，与报告"three early pair candidates accelerate one slot beyond Native: 25, 26, 27 respectively"一致。

报告表格（逐候选 `Mr/native period ratio`、`Δsum(m)`、128K 最大绝对相位变化、是否在 `0≤m≤1` 盒内）共 29 行，astra05:226–256。若干要点：
- d1 系列的 `Δsum(m)` 全为负（−0.0093 … −0.0183），d2 系列全为正（+0.00037 … +0.00185）。
- 只有前三个 d1 候选（s24_25/s25_26/s26_27）**出压缩-only 盒**；全部 d2 三元组都在盒内。
- **零和关系保的是原始频率和，不是 log-压缩指数之和** ⇒ "Therefore these candidates are not same-compression-budget controls; every delta is reported rather than silently corrected with an additional slot."（:258）等级 [已验证-推导]。
- 每个 JSON 条目给出 `delta_log_period = −log(nu_c/nu_M)` 供与父级 `gradient_log_period` 缩并；**"The resulting first-order predicted loss is only a direction filter."** 对变化在 128K 达 58.305（pair）/6.907（triple）弧度，"the gradient cannot forecast their finite loss reliably"；**必须用精确全模型有限答案 CE 与实际生成裁决**；要作后期因果谐波归因还需对独立正交方向重复干预。"No GPU was used in this continuation."（:260）

---

## 6. 交叉对比、复算与矛盾

### 6.1 五份报告的共识对象

| 对象 | astra01 | astra02 | astra03 | astra04 | astra05 |
|---|---|---|---|---|---|
| 角色条件 margin `M_r` | ✔ 式(1) | （§3 用） | ✔ `Z_r=c_rᵀf_r` | （校准 KL 版） | ✔ `log Z_S−log Z_D` |
| 带号、按槽标签、带内容系数 | ✔ | ✔ | ✔ 显式 A/B | ✔ C_jk/D_jk | ✔ a_tj/b_tj |
| 稀释项 `log((M−1)(1−a)/a)` | union bound | — | ✔ 显式 | Z_D/Z_O | `−log N` |
| Cantelli | ✔ `1/(1+t²)` | — | ✔ | — | （指数矩） |
| 密度参数化仅限从头 | ✔ §7 | ✔ §3 | ✔ 显式区分为 (F) 与冻结 | ✔ 表坐标 | ✔ 显式 (I−P_R)ω |
| 无符号几何不得入 F | ✔（反例） | ✔（:137） | ✔ §7 反例 | ✔ 恒等映射 guard | ✔ §6 |

**与 INTEGRATION §4.1（五份审计共同背书措辞）一致**：astra01–05 都落在"带符号、按槽标签、带内容系数——三缺一即死"这一句之内，且都未用首零根排序。

### 6.2 本轮独立复算结果汇总（全部通过）

| 复算对象 | 报告值 | 本轮值 | 判定 |
|---|---|---|---|
| astra01 三频反例 `B_A`,`B_B` | 0, 1.8090169944 | 6.1e−17, 1.8090169943749475 | ✅ |
| astra01 EVQ 二次 A/B | 0, 3.2725424859 | 3.7e−33, 3.2725424859373686 | ✅ |
| astra01 wrong-key err A/B | 0.5, 0.1481417538 | 0.5, 0.1481417537951759 | ✅ |
| astra01 positional err A/B | 0.0416322583, 0.2458478318 | 0.0416322583317752, 0.2458478317699044 | ✅ |
| astra02 原子解 1/2/3 原子能量 | .183907229150896 / .058949724094609 / .058639327401539 | .183907229150897 / .0589497240946106 / .0586393274015407 | ✅ |
| astra02 原子频率/质量 | [0.1,.4066617223,1.0] | [0.1,.4066617366,1.0]；质量 8 位同 | ✅ |
| astra02 KKT 支撑等式 `V=c` | — | 三原子 V 全等 0.11727865480308142 | ✅ |
| astra02 全局 gap 上界 | 1.53377e−8 | 1.533765e−8 | ✅ |
| astra02 geomean / Cosh τ=2 对照 E | .13323829017761 / .09518226679009 | .1332382901776104 / .09518226679008959 | ✅ |
| astra02 K=8 量化多重数 (1,3,4)、E | .05892945173956 | (1,3,4), .05892945147079193 | ✅ |
| astra02 §7 β（n=16/32/64） | .858718721249 / .870290110287 / .875477427278 | 全同 | ✅ |
| astra02 §7 α 与 τ | .00709041638147/.00329366373151/.00157961736196；τ 11.005/16.255/23.542 | 相对差 ~1e−4；τ 11.006/16.256/23.543 | ✅（定性同） |
| astra03 slot28 `ω`、`ν_M` | 0.0023713737056616554 / 0.002070019952952862 | 全同 | ✅ |
| astra03 Mr/Smooth target mean | +0.9994600706262113 / −0.9995409168024664 | 全同 | ✅ |
| astra03 分支最优 `ν*`(n=25) | 0.0020697109769935414 | 全同 | ✅ |
| astra04 padded-KL 恒等式 | 0.45529644748142606 | 0.45529644748142606 | ✅ |
| astra04 映射 max/KL 计数 | 118783 → 131071、98304 槽 | 全同（8×12288） | ✅ |
| astra04 块分解恒等式 | — | LHS=RHS（差 7e−18） | ✅ |
| astra05 两关系周期/相对变化/范数比 | 76740.5661→306962.2644 等 | 全同到所示位数 | ✅ |

### 6.3 与权威文档的冲突 / 口径不一致（逐条裁决）

| # | 冲突点 | 两边出处 | 判定 |
|---|---|---|---|
| C1 | **"碰撞二次型 ⇒ cosh"** | `NEXT_DERIVATION_KKT_PROBLEM.md:101` 把"碰撞二次型 ⟨ρ,Kρ⟩ 的欧拉–拉格朗日 ⇒ cosh 轮廓"列为 EVQ 极限关系的 [部分证据] 表述；astra02 定理说**精确有限窗碰撞泛函的唯一最优是原子**，Cosh 只能从 `αδ+βmin` 代理流出（astra02:19–47） | **精度冲突已裁决**：INTEGRATION §4.2（:69）已写"Cosh 只从 delta-plus-min(αI+βG) 代理流出"。NEXT_DERIVATION 的措辞应补上"经代理"三字。**取 astra02 的精确陈述**。 |
| C2 | **αI+βG 的合法性来源** | `EVQ_COSH_THEORY.tex:114–127` 写 `K_app=αδ(φ1−φ2)+βmin(φ1,φ2)` 并称"e.g. via Hilbert–Schmidt projection / finite-N calibration"；astra02 §7 说无穷维 L2 上 I 非 HS，**字面连续 HS 投影不存在**（对任意 α≠0 范数为 ∞） | **astra02 修正 tex 措辞，不与权威文档冲突**（NEXT_DERIVATION §4 只提"非局部修正…[已验证=CPU 数学]"，未主张 HS 投影）。tex 的 τ* 标度律本身已自标 "Status: Conjecture"（我核到 :333 附近 "we therefore record the scaling as a conjecture"）⇒ astra02 的批评**有据**。 |
| C3 | **τ≈d_head/√L** | astra01:166 说 τ=√(β/α) 不是 d/√L 除非另行论证；astra02 §7 给 τ∝√n（固定 L）；INTEGRATION §5 把"τ≈d_head/√L"列入死亡登记册 | **三方一致，无冲突**。astra02 额外给出**为什么**该式在固定 L 投影路线下是假象。 |
| C4 | **astra04 的校准 KL 在 F 中的地位** | astra04:51 把 `L_bal` 当可优化目标、给完整约束分配器；`NEXT_DERIVATION` §5 红线 5："线性读出/固定态代理算子（J_r、校准-KL、E7 160×）**只配假设生成，不入 F 主链**" | **口径冲突，权威已部分裁决**：INTEGRATION §4.3（:80）把 astra04 归为"校准件"而非求解器主链。astra04 自身也自标"frozen-compatibility estimator"、不称 F。⇒ **按权威文档：astra04 是校准/诊断件，F 不得以它为主项。** |
| C5 | **astra05 的 R 辨识状态** | astra05:213 明说"Selecting its active relations for Qwen and proving any advantage over MrRoPE remain uncompleted"；INTEGRATION §4.2 记该文件自标 `NO_ROLE_OR_CAPABILITY_QUALIFICATION` | **一致，无冲突**。但注意 INTEGRATION §4.2 把候选的加速表述为"比 **MrPro** 还快"，astra05 表述为"beyond **Native**"；我复算证明这两者在 s24_25/s25_26/s26_27 上**同时成立**（周期比 <1 ⇒ 比 Native 快；ν_c25=0.00485 > ν_M25=0.00441 ⇒ 也比 MrPro 快）⇒ **仅措辞强弱不同，无实质矛盾。** |
| C6 | **astra02 §3 的两时钟特例 vs NEXT_DERIVATION 的 KKT 三段预言** | NEXT_DERIVATION:55 预言"最优解形态自动是三段：Δ=0 的 bank 平台、Δ>0 的过渡桥、尾部平台"；astra02:114 说中段"is the set of conflicting or coupled active computations, **not automatically an interval determined by one rotation count**" | **不是矛盾，是约束**：KKT 三段说的是"结构形式"；astra02 说的是"中段**位置**不能由单一圈数公式定"。⇒ 给 K1 加一条硬要求：三段**边界位置**的 S-依赖必须从计算耦合导出，不能从 r_j 公式导出。 |
| C7 | **astra05 §8 的 29 候选 vs INTEGRATION §5"null band 假设被联合模式否定"** | astra05:222 "none of these Mr relations already retimes Native by4"；INTEGRATION:72 "MrPro 对 29 个相邻关系时钟的 0/29 做了 ×4 重定时" | **一致（0/29 与 none 同义）**。 |
| C8 | astra05 引 `docs/theory/EVQ_COSH_THEORY.tex` 非局部修正说"h_c ridge 乘子随波数降" | `EVQ_NONLOCAL_KERNEL_CORRECTION_20260910.md` 给出 `K_L = [c·min(φ,ψ) − γ + h_c(φ−ψ)]/log L + E_L`，`h_c(x) = −½log(1−e^{−2c|x|})` | **本轮 [已验证-源文件]**。**这是重要 F 零件**：精确核里 `c·min(φ,ψ)/log L` 是**精确领头项**（不是近似），而 `αδ` 才是错位的替代；非局部部分是**对数脊**而非 δ。修正注自身注明该展开"deliberately not a uniform approximation at the diagonal"。 |
| C9 | astra01:22 引 MrRoPE §4.4 与 Appendix B 的 Qwen 边界 (23,40) | STARTING_POINT F1/F2 已判"YaRN 递减 vs MrPro 递增"叙事作废；NEXT_DERIVATION §3 表把 YaRN 记作"≈MrPro 近邻、Qwen S=4 无表格载体差异" | **astra01 未使用被作废叙事**（它只说 MrPro 的等差数列是 :420 处的**显式假设**，"not derived from the first-zero analysis"）⇒ 与 F1/F2 相容。 |
| C10 | astra02:141 引 `docs/exp/2026-02/2026-02-27_evq_tau_sweep_results.md` 称"τ=1.5 改进每个 50M 长度"这句话"on its face false"，因 4K 给 6.667 vs geometric 6.183 | 我核到该文件 :14 行 τ=0.00 的 PPL@4096 = **6.183**，:20 行 τ=1.50 = **6.667**（同一 50M、seed42 表，:117/:118 重复确认） | **本轮 [已验证-源文件]，astra02 的指控成立**。该句作为"普适改进"被其自身表格证伪（4K 上变差 7.8%）。 |

### 6.4 材料内部的不一致（同一报告内）

- astra02 §7 的 α 数值与本轮有 ~1e−4 相对差：**我用 32 点/bins、报告未声明 bin 内积点数**（可能影响 α 的高阶项），β 完全一致。⇒ 报告未给 bin 内求积阶数，**是可复现性缺口，不是错误**。
- astra04 的报告标题/正文混用 `W=BM=32768` 与 `block_size=4096`（`B=8`）：确认 `M=4096` 与 `W=32768` 自洽（`B·M=W`），但正文从未显式写 `M=4096`，只有源码里有 `'block_size':4096`。⇒ **口径需在引用时补齐 M=4096**。

---

## 7. 可作为 F 零件的清单（按可用性排序）

> 红线复查：以下**没有**一条是静态几何代理（Σcos 首零、碰撞能、覆盖率、平滑度、有效秩、Gram、能量）作 F 分项或选择子。§7.1 的 astra02 条目被明确标注为**诊断/下界**，不是 F 分项。

**F-1 角色条件化带符号 margin 与协方差（astra01 式(1)–(5)；astra03 `Z_r=c_rᵀf_r` + (F)/(I)；astra05 §1）** [已验证-推导]
```
μ_η = ∫h dη,  v_η = ∬C dηdη,  t = μ/√v
Pr(M≤0) ≤ 1/(1+t²)                     （Cantelli，无分布假设）
ρ*(x) = [C⁻¹h](x)/∫[C⁻¹h]              （C⁻¹h≥0 时）
Cw ≥ λh, w ≥ 0, w(Cw−λh)=0, ⟨h,w⟩=1     （active-set，逆有负分量时）
```
多角色：`max min_r t_r`，或 `max_{x∈F} min_{r∈R_L} z_r(x) s.t. z_r(x)≥z̄_r (r∈R_N)`（硬 native 约束，无混合系数）。

**F-2 稀释项 η_r（四家同式，astra03 显式给形）** [已验证-推导，INTEGRATION §4.2 交叉确认]
```
η_r = log[ (M−1)(1−a) / a ]        （M 个 key 中软源质量 ≥1−a 的充分 pairwise margin）
硬源排序：η_r = 0
```

**F-3 有限盒 Taylor 认证约束 (I)（astra03）与 margin QCQP（astra02 §4）** [已验证-推导]
```
(I):  μ_rᵀf̂_r(v) − η_r ≥ ε_r‖f̂_r(v)‖₂ + t‖L_rᵀf̂_r(v)‖₂ + (‖μ_r‖+ε_r+t‖L_r‖₂)Q_r(v)
      Q_r(v)=½Σ_j B_{rj}v_j²,  B_{rj}=√((U⁺)²+(U⁺)⁴+(U⁻)²+(U⁻)⁴)
      U^±_{rj}=|d_r^±|ω_j exp(−x_j⁰+r_j)
astra02 §4:  margin_r(x⁰+v) ≥ margin_r(x⁰) + g_rᵀv − ½Σ_j M_rj v_j²,  M_rj 显式
```
两者都把"有限步"变成可认证对象（不是无穷小替代）。

**F-4 softmax log-partition margin 的两项分解（astra05 §2）** [已验证-推导，前提=共享相干 nuisance]
```
−M̄ = log N − μ K_ν(d_s) + ½σ² E_{p_D} K_ν(d)² + O(σ⁴)
      ↑Mr 型：正匹配余弦均值       ↑EVQ 型：求和后平方的相干碰撞
```
**这是把 EVQ 与 MrRoPE 放进同一个目标的最直接零件**（astra05:5–7）。**前提必须显式**：distractor 分数 `ξ_t Kν(d_t)`、`ξ_t~N(0,σ²)` 跨 key 独立（每 key 内部共享相干标量）。**若改成独立各向同性正交分量，平方项消失**（astra05 §2 反例；astra01 §5；INTEGRATION §5 同判）。

**F-5 精确核的领头结构（`EVQ_NONLOCAL_KERNEL_CORRECTION_20260910` + astra01 §5 + astra02 §1/§7）** [已验证-源文件]
```
K_L(φ,ψ) = [ c·min(φ,ψ) − γ + h_c(φ−ψ) ] / log L + E_L,   h_c(x)=−½log(1−e^{−2c|x|})
|E_L| ≤ (1/log L)[ 1/(Lδ) + 1/(Lσ) + (δ²+σ²)/8 ]
```
含义：`min` 项**精确**、非局部项是**对数脊**而非 δ；EVQ 的 `αδ` 是错位替代。**可作核对 EVQ 代理合法性的诊断件**，若要用在 F 中必须声明它替换了哪一项。

**F-6 联合模式输运算子（astra05 §4）** [已验证-推导 + 本轮数值复现]
```
ν(S) = (I − P_R)ω + S⁻¹ P_R ω,   P_R = Rᵀ(RRᵀ)†R
T_R(S)T_R(U) = T_R(SU)   （同一 R 的尺度复合律）
两频: ν₁=c+g/(2S), ν₂=c−g/(2S)；三频: ν_B = ω_B − (1−S⁻¹)κ(1,−2,1)/6
```
它是"选择保留/重定时哪些关系"的**精确有限输运**，不是一阶 QP；打破压缩-only 盒。

**F-7 保标签恒等性（astra03 §2）** [已验证-推导]
```
q_jᵀR(ν_j d)k_j = A_j cos(ν_j d) + B_j sin(ν_j d)
A_j = q_{j0}k_{j0}+q_{j1}k_{j1},  B_j = q_{j1}k_{j0}−q_{j0}k_{j1}
```
**只置换表改 Z；联合置换完整槽才不变** —— 这是"槽标签约束"的精确依据。

**F-8 校准侧的精确恒等式（astra04）** [已验证-推导 + 本轮数值复现]
```
L_full = D_KL(T‖Q_O) + log(1+Z_D/Z_O)  =  teacher 零填充后与增广 student 的 KL
D_KL(T‖Q_O) = D_KL(p‖q) + Σ_b p_b D_KL(T_b‖Q_b)
L_bal = D_KL(p‖q) + B⁻¹Σ_b D_KL(T_b‖Q_b) + log(1+Z_D/Z_O)
∂z_k/∂x_j = σ d'_k ν_j [ C_jk sin(d'_kν_j) − D_jk cos(d'_kν_j) ]
```
**地位受限**（C4）：按 NEXT_DERIVATION §5 红线 5 与 INTEGRATION §4.3，这是**校准件**，不入 F 主链。

**F-9 软投影恒等（astra03 §9）** [已验证-推导]
```
KL(p‖p_{c₀}) = KL(p‖p_{c*}) + KL(p_{c*}‖p_{c₀})   （c* 极小化 KL(p‖p_c)）
```
把"表征拟合"与"固定系数相容性"在 softmax 指数族内精确分离。**需要有限极小解存在 + 偏移/支撑固定。**

**F-10 有限-K 分位实现界（astra01 §6）** [已验证-推导]
```
W₁(η_K, ρdx) ≤ R/(2K)  （中点分位）；≤ R/K（端点等权）
|μ_K−μ| ≤ L_h d,  |v_K−v| ≤ 2L_C d,   t_K ≥ (μ−e_μ)/√(v+e_v)
```
作者自评 K=64/128K 可能很弱。

---

## 8. 死路登记（绝不能再试的机制，含失败原因）

| # | 机制 | 出处 | 失败原因 |
|---|---|---|---|
| DE-1 | **单一无符号余弦能量 / EVQ 二次能量作普适分配目标** | astra01:20–35, 274 | 同 lag 的语义角色一变，正确排序反转（三频同支撑反例：A 在 EVQ 二次下"更优"，但在位置判别下 pairwise error 0.0416 vs 0.2458）。[已验证] |
| DE-2 | **"随机/独立干扰产生平方余弦碰撞"⇒ 由此推出 α∫ρ²** | astra05:73–82; astra01:169–194; INTEGRATION §5 | 独立各向同性 Gaussian 正交分量给 `Var z_D=σ²`、`E e^{z_D}=e^{σ²/2}`，与 ν、d **无关**，平方和消失；独立各向同性 Q/K 给旋转不变分数分布。[已验证-推导] |
| DE-3 | **K 个独立通道噪声实现 α∫ρ²** | astra01:181–183 | 其平均贡献 `K⁻¹∫v(x)ρ(x)dx` 对密度是**线性**的，v 常数时是 v₀/K；不可能生成 `α∫ρ²`。白噪声是**频率场**积分，在原子点求值无定义。[已验证-推导] |
| DE-4 | **δ+min 代理当"无害近似"、把它当作精确核的一阶修正** | astra02:7; `EVQ_NONLOCAL_KERNEL_CORRECTION` | δ 项引入精确有限窗目标**不具备**的 L2 反集中代价；精确核在**对角有限**而 h_c 脊在对角发散；只恢复奇异脊仍不等于恢复精确核。⇒ 会**过度惩罚窄变化**，窄桥合法性判断被扭曲。[已验证-源文件] |
| DE-5 | **把原子/连续最优解直接当部署表（对象类错误）** | astra02:96–102, 143; INTEGRATION §4.2/§5 | 四对象两两不可互换：①精确-原子最优 ②光滑-Cosh 代理 ③有限-K 整数计数 ④冻结-带标签表。测度最优**不**在"质量为 1/K 整数倍"的可行类里（K=8 时能量 .05893 > .05864）。[已验证] |
| DE-6 | **"裁剪 C⁻¹h 的负分量"代替 active-set** | astra01:99; INTEGRATION:70（astra10 修正） | 逆有负分量时裁剪一般错误；正确形式是 active-set KKT (5)。[已验证-推导] |
| DE-7 | **从 native-only 观测辨识长上下文 h,C / 用 native 缓存重相位替代长 prefill** | astra01:256–258; astra02:176; astra03:176; INTEGRATION FLAG-5 | 两个响应族可在所有 lag ≤ W 上一致而在 W 之外有用源系数反号 ⇒ 一切 native-only 选择子看到同样输入却给出不同长最优分配；缓存重相位只对**该缓存**的局部旋转收缩精确，不重现前层在 128K 历史下的内容形成。[已验证-论证] |
| DE-8 | **首零视界 / 根排序作选择子** | astra03:166; astra05:227（明说 first-zero surrogate 弃掉槽特异系数均值、求积相位、硬干扰结构、下游值）；STARTING_POINT F7/F8；红线 1 | 一维精确解要求 `ν ∈ ⋃_n [θ+2πn±arccos((η+tσ)/A)]/D`——**槽的正确内容相位 θ 与可行相位分支**才是决定因素，"首零地平线"不普适。[已验证-推导] |
| DE-9 | **用无符号几何量（source-weak 总能量、投影算子距离、频率范数、unweighted distortion）选 R 或表** | astra02:137; astra03:18; astra05:197–199 | Smooth 在三个 cutoff 每一层都更优却 128K 更差；"A small source eigenvalue does not authorize independently moving participating channels." [已验证-源文件] |
| DE-10 | **重复 donor 块当独立观测；零填充 teacher 当"无关块应被忽略"的定理** | astra04:43, 45 | 与期望 key 完全相同的 donor 必然竞争（V 同则输出可不变而 penalty 为正）⇒ penalty 测的是身份保持而非功能保持；反之复制四份时 marginal softmax 恰为 T，**没有 factor-4 罚的理由**。donor sink 态会造出真实 128K 不存在的额外竞争。[已验证-论证] |
| DE-11 | **把模型干预选在"周期好看/相位覆盖大"的谐波上；用已暴露的 128K 答案挑 R、cutoff、符号或 S** | astra05:117, 189–191 | 同一谐波可因**带号角色对比、内容相位、占据位置**增大或减小 margin；|c_n| 与长周期都不能决定保全/拉伸/压制。无关系满足三条件契约时唯一合法输出是 **"no justified mixed-mode intervention"**。[已验证-推导] |
| DE-12 | **靠降低 source-weak 能量/算子距离/频率范数来"修复"排序；把校准 KL 当 F 主项** | astra04:77 自陈；astra05 §6；NEXT_DERIVATION §5 红线 5 | 校准件地位受限（C4）；Smooth/MrPro 反例已把无符号路径封死。[已验证] |
| DE-13 | **"集中残差 = 与主导 loss-Hessian 特征向量对齐"；"单圈相位覆盖 ⇒ 联合相位安全"；"被包裹通道移到哪都无条件安全"** | astra05:205 | 前者断言因果对齐却未测量；中者不成立；后者被混合模式与同文件后续联合轨迹段明确否证。[部分证据] |
| DE-14 | **用被自身表格证伪的"普适改进"叙述** | astra02:141；`docs/exp/2026-02/2026-02-27_evq_tau_sweep_results.md:14,20,117,118` | 该文档的"τ=1.5 改进每个 50M 长度"与其自身 4K 表（6.667 vs 6.183）冲突。[已验证-源文件] |

---

## 9. 未解问题 / 材料间的开放缺口

**Q1（阻塞级，五份报告共同指向）**：**带标签的角色矩从未被测**。需要 pre-RoPE Q/K 上按角色（native/far × source/hard-distractor）拆分的**带符号均值/协方差/单位对数 MGF**，含跨槽协方差，保 layer/head/relation/lag 标签。（astra03:170–180 给测量合同；astra01:246–256 给 native 测量的可实现部分与不可实现部分；INTEGRATION §8-1 列为"唯一阻塞证据"。）预注册门槛：这些统计量必须**先正确否决 Smooth**（slot-28 符号反转 +0.99946→−0.99954，astra03 §7 已给解析模板）并**暴露 P2 的 +long/−short 权衡**，否则停并报告缺失因果层。

**Q2**：`h,C` 的**长上下文可辨识性**。astra01:256 证明 native-only 观测不足；需要显式平稳/输运假设、独立目标域证据，或对结果偏差的界。**这条不解决，任何 F 的 L_far 侧都没有测量基础。**

**Q3**：**中段边界位置的 S-依赖**（K1）。astra02:114 排除了"由单一圈数公式决定"；F5 要求 Qwen 38→Llama 25 的移动可从 KKT 条件的 S-依赖重 derive。**这是三段结构预言能否成立的核心。**

**Q4**：**有限-K 整数分配问题未被解**。astra02 给了下界与初始化器（:102）、给了交换算法与对偶证书（:59–77），但明确"the measure solution is not its solution"。astra03 的 (F) 是同一单纯形上的另一目标。**需要 K=64 的真实整数解或明确的不可行性/vacuous 报告。**

**Q5**：**`−M̄ = logN − μKν(d_s) + ½σ²E_D Kν²` 里的内容假设是否在 Qwen 上成立**（astra05:41 的"每 key 共享相干标量"）。astra05 §2 反例说明换成独立各向同性就全没了。**这是一个可判定的实证问题，且决定 F-4 是否为 F。**

**Q6**：**γ（gain）的地位**。astra03 显式含 `γ=g²/√d_head`；NEXT_DERIVATION §5 红线 6 说"gain×相位不正交，F 不含 gain 自由度"。⇒ **两口径未对齐**：F 里能不能出现 γ？（当前权威说不能。）

**Q7**：**astra04 的校准路线在 Qwen 上的可迁移性**。作者自陈"Failure at dense128K after success on frozen stitched rows specifically falsifies transfer of this estimator"（:77）——**该证伪实验未做**。另：holdout 只有 2 篇文档使三 donor 构造不可能（:43），**如何在不造假独立性的前提下做 holdout，未解**。

**Q8**：**astra04 的捕获只有 4 个 query head、单 query 位置**（`pro_block_calibration.py` 的 `selected_heads=[index%4+4*i for i in range(4)]`，`qpos=length−1`）⇒ 不能辨识跨 query 位置行为，也不能推广 query-local vs block-local（:73）。**需要扩捕获还是接受这一限制，未决。**

**Q9**：**astra05 的 29 个候选的判决**：报告明确"必须用精确全模型有限答案 CE 与实际生成裁决"，且"要作因果谐波归因还需对独立正交方向重复干预"（:260）。**GPU 未跑，判决未出。** 另外 29 个候选里有 3 个出压缩-only 盒（比 Native 还快），⇒ **"是否允许加速"是一个必须先声明的机制自由度**（astra05:150），当前权威文档把它记为"实质性机制限制"而非实现约定。

**Q10**：**astra02 §7 的 α 复现差**（~1e−4 相对）：报告未声明 bin 内求积阶数，**是可复现性缺口**。

**Q11**：**源不可达导致的一批数字无法核**：`results/nongeometric_screen_20260909/development_summary.md`（Smooth −9.79pp、E1 +5.21、P2 +3.54/−14.31）、`planned_controls/{bias_harmonic_head27_8_audit.json, p2_gap_comparison.json}`、`queue/0440_Smooth_MrBudget.json`（ν_S）、`ROPE_BM_CROSS_CACHE_RESULT_20260908.json`、`.agents/rope_unification_20260910/*`（29 候选 JSON、coverage 回执、`full_model_response_native.jsonl`）。**这些构成 astra03/04/05 的关键证据链，建议在归档中找回或明确标注"证据不可复核"。**

**Q12**：astra01 §3 引 Capon 1969 作为匹配滤波谱估计的谱系来源，**原文未读**（作者自注 "full Capon paper not ingested"）⇒ 谱系命名需另核。

---

## 10. 给下游推导的一句话索引

- **想要 F 的信号项** → astra05 §2 的 `μKν(d_s)`（Mr 型）+ astra01 的 `μ_η=∫h dη`。
- **想要 F 的干扰项** → astra05 §2 的 `½σ²E_D Kν(d)²`（**必须声明相干 nuisance；独立各向同性则此项不存在**）；astra01 的 `v_η=∬C dηdη`。
- **想要 KKT 约束形式** → astra01 式(5) 的 active-set（**不是裁剪**）；astra03 (F) 的 `z_r(x) ≥ z̄_r` 硬 native 约束 + (I) 的有限盒认证。
- **想要 dilution / log N 项** → astra03 的 `η_r = log[(M−1)(1−a)/a]`。
- **想要"为什么中段不能由公式定"** → astra02:108–114 的两时钟特例与不可行条件。
- **想要"为什么不能解一个装另一个"** → astra02 §1/§4/§7 的四对象分离 + 原子定理。
- **想要"哪些方向已被证伪"** → 本文 §8 的 14 条死路。
