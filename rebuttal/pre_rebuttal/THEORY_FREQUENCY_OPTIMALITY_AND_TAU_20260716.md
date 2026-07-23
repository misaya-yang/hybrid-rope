# 频率分配最优性与 \(\tau\)：复核后的理论分层

日期：2026-07-16

状态：`internal_verified`

范围：纯理论。不修改代码、不修改论文实验数字、不引入新实验。

权威关系：方法身份、Phase16、`c_coll`、DAPE/YaRN 等事实以 `FULL_PAPER_INTEGRITY_AUDIT_20260713.md` 为准；ordinary-KL / transport proxy 长推导见 `THEORY_REBUTTAL_MATHEMATICAL_AUDIT_20260711.md`；策略与机制全景见 `FIRST_PRINCIPLES_REBUTTAL_REASSESSMENT_20260716.md`。本文在上述材料之上，**重梳并修补逻辑链**，专答两问：

1. 均匀（geometric）分配是否“就是最优”？cosh 非均匀在何种意义上最优？
2. \(\tau = d/\sqrt{L}\) 能否从第一性原理推出？与经验轨迹是否同构？

证据标签：

- `[严格]`：在写明假设下可证明或可严格否证；
- `[有条件]`：在额外建模假设下自洽，假设缺失则降级；
- `[经验]`：数值或训练结果支持，不是定理；
- `[开放]`：当前条件推不出。

回应规则：本文的“必撤 / 禁止外推”表示该论据不得在内部策略或被触发的回答中继续使用；**不等于**在 reviewer 未询问时自行发散。是否将未被点名的 material error 合并向 AC 披露，仍服从 `rebuttal_playbook.md` 的 author-decision gate。

仓库证据锚点：

| 对象 | 位置 |
| --- | --- |
| Surrogate 定义、严格凸、边界条件与 cosh 解 | `paper/sections/03_theory.tex:21-46`；`paper/appendix/a1_proofs.tex:4-50` |
| Inverse-CDF 与 midpoint 实现 | `paper/sections/03_theory.tex:50-65`；`scripts/lib/rope/schedules.py:94-140` |
| 提交稿 \(\tau\) / KL / stiffness 原叙事 | `paper/sections/03_theory.tex:90-117`；`paper/appendix/a1_proofs.tex:297-403` |
| 方法身份、`c_coll`、Phase16 与 provenance 纠错 | `rebuttal/pre_rebuttal/FULL_PAPER_INTEGRITY_AUDIT_20260713.md:119-206,280-296` |
| 早期 \(L=128/1024/2048\) sweep | `docs/exp/2026-02-26_full_experiment_report.md:150-220,511-552`；`docs/exp/2026-02-27_evq_tau_sweep_results.md:9-54` |

---

## 0. 结论先行（复核后）

| 主张 | 判定 |
| --- | --- |
| Geometric RoPE 是 RoPE/LM 的普适最优频率表 | **否** `[严格]`：无此定理；它是 log-uniform 设计点 |
| Geometric 在 uniform log-frequency 量化/覆盖或最大熵目标下严格最优 | **是** `[严格]`：但目标与边界条件必须写明（§3.3） |
| Geometric 是 \(\mathcal C_{\mathrm{app}}\) 在 \(\beta=0\) 时的唯一最优 | **是** `[严格]` |
| Cosh 是 \(\mathcal C_{\mathrm{app}}\)（\(\alpha>0,\beta\ge0\)）的唯一最优 shape | **是** `[严格]`（仅对该 surrogate） |
| Cosh 是 exact oscillatory kernel 或 LM loss 的闭式最优 | **否/未证** `[开放]` |
| 非均匀在任意目标下都优于均匀 | **否** `[严格]`：见 §3 目标表 |
| Submitted midpoint Geo→EVQ 是 fixed-extrema / fixed-span 的 pure-shape control | **否** `[严格]`：finite \(K\) 下还移动频谱极值，且没有保持 realized span；\(K=32,b=500\mathrm K,\tau=4\) 时它缩短（§3.6） |
| Ordinary KL 导出非零 \(\tau_*\propto L^{-1/2}\) | **否** `[严格]`（否证）：一阶变分为零，从 \(O(\tau^4)\) 起 |
| Diffuse transport proxy 下 \(\tau\propto\sqrt{d_S d_U/L}\) | **有条件成立** `[有条件]` |
| 部署式 \(\tau=d/\sqrt{L}\) 是全局任务最优 | **否** `[严格]`：最多是 proxy 结构 + 经验 basin |
| 指定 trained task 的局部 \(\tau_*\) 只由 \(d,L\) 决定 | **否** `[严格]`：它取决于 task-risk 的一阶/二阶导数（§4.0） |
| Surrogate 拟合尺度 \(\tau_{\mathrm{surr}}=\sqrt{\beta/\alpha}\) 与部署 \(\tau_{\mathrm{deploy}}\) 是同一最优解 | **否** `[严格]`：幂次不同（§4.4） |

**原有的 surrogate → exact kernel → ordinary KL → deployed \(\tau\) → task improvement 端到端链条已断裂。** 尚未失效的是一个更窄的 formal core：

> 有限 RoPE 通道构成 finite spectral budget；在明确写出的凸 surrogate 上 allocation shape 有唯一闭式最优（cosh 族）；scale 必须另定对象。部署 \(\tau\) 是 conditional proxy 提供结构动机后的经验 operating default，不是 trained-task theorem。

还必须补一句离散边界：当前 midpoint implementation 是一个 **schedule intervention**，并未在 fixed extrema / fixed span 下单独隔离 density shape。

---

## 1. 逻辑链总图（先修缺陷）

上一轮叙述中有三处容易把层次粘在一起。本文强制拆开：

```text
[层 A]  设计空间
        有限 K 个 rotary pairs；ρ 经 inverse-CDF 量化
        geometric ⇔ ρ ≡ 1

[层 A'] Finite-K 实现
        midpoint / endpoint grid；realized extrema 与 span
        ※ 连续 support 同为 [0,1]，不代表离散表端点/span 已匹配

[层 B]  Shape 目标  C_app[ρ]  (或 forced / variable-α 变体)
        β=0 → ρ=1
        β>0 → ρ_τ cosh，τ_surr = √(β/α)
        ※ 此 τ 由 surrogate 系数比决定，不是部署公式

[层 C]  一参数族上的 scale 目标
        限制在 pure-tether 族 {ρ_τ : τ≥0} 上
        平衡 stiffness S(τ) 与 某 utility U(·,L)
        ※ U 若取 ordinary KL → 推不出非零小 τ*
        ※ U 若取 transport proxy U_tr → 条件性 tau ~ 1/sqrt(L)

[层 D]  部署坐标
        τ_deploy = d_eff / √L_train  (+ 单位约定)
        落入经验 PPL basin 的 selector，不是层 B/C 的同一解

[层 E]  Task
        指定 checkpoint：R(θ)=R₀+A_task θ+½B_task θ²+...
        重训练：曲率再含 weight-adaptation Schur complement
        ※ 形式可写，A_task/B_task 当前未测
```

**关键修补 1（族限制 vs 全空间最优）**
\(\mathcal C_{\mathrm{app}}\) 的最优是 cosh；但 **单独最大化**
\(U_{\mathrm{tr}}(\rho;L)=\frac{M}{L}\int q(Lb^{-\phi})\rho(\phi)\,d\phi\)
时，因目标对 \(\rho\) **线性**，在概率测度空间的最优解支撑在 \(\arg\max_\phi q(Lb^{-\phi})\) 上，一般可取 Dirac，但 argmax 未必是端点。若只允许绝对连续 / \(L^2\) 密度，且 argmax 集零测，则通常只有向该集集中的 supremum，没有可行 Dirac 解。
因此：**cosh 不是 transport 的自由最优**；它是“surrogate 系数比变化产生 cosh 一参数族，再用另一个对象选择部署坐标”的结果。
任何“cosh = transport 最优 shape”的说法都过强。

**关键修补 2（两个 \(\tau\) 不是同一对象）**
论文附录对 exact kernel 的离散拟合给出量级
\(\alpha\sim d_{\mathrm{rot}}^{-1}\)，\(\beta\sim L^{-0.22}\)，故

\[
\tau_{\mathrm{surr}}=\sqrt{\beta/\alpha}\sim\sqrt{d}\,L^{-0.11},
\]

而部署

\[
\tau_{\mathrm{deploy}}\sim d\,L^{-1/2}.
\]

\(L\) 指数 \(-0.11\) vs \(-0.5\)、维度 \(\sqrt{d}\) vs \(d\) **不能**用 \(O(1)\) prefactor 吸收。
更精确地说：给定一组 \(\alpha,\beta\)，层 B 会固定 \(\tau_{\mathrm{surr}}\)；当系数比变化时得到 analytic family。论文拟合出的层-B坐标与层-C/D部署坐标不是同一条 scaling law。

**关键修补 3（utility 方向 ≠ schedule-KL 方向）**
- **Channel pattern** \(c_\omega(j)=\cos(\omega j)\)：定义每通道 phase-variance / transport score；allocation 改 \(\rho\) 后总 score 可有对 \(\theta=\tau^2\) 的 **一阶** 变化。
- **Schedule derivative** \(g_\theta=\partial_\theta z\)：从 Geo 滑到 EVQ 的 ordinary KL 在 \(\theta=0\) **一阶为零**，从 \(O(\theta^2)=O(\tau^4)\) 起。
正文曾把二者混称为 “post-softmax KL gain \(O(\tau^2)\)”，逻辑不成立。

**关键修补 4（连续 shape ≠ finite-grid pure-shape control）**
连续密度都定义在 \([0,1]\)，但 midpoint inverse-CDF 不取 support 端点。对任意 \(u\in(0,1),\tau>0\)，严格有 \(\phi_\tau(u)<u\)，所以 practical EVQ 把每个采样频率都向高频移动，并改变 finite-\(K\) 的最小/最大频率与 realized span。same quantile convention 只能控制采样规则，不能控制离散频谱范围。

**关键修补 5（task optimum 必须带 task 导数）**
若只问一个指定 checkpoint / task，第一性原理答案不是另一个 model-free closed form，而是 \(\theta=\tau^2\) 下的局部 risk expansion。只有 task gradient 为负且曲率为正时，非零 \(\tau\) 才是局部改进；重训练还会改变曲率。这是连接 LLaMA co-adaptation、PPL 与下游失败时不能省略的层。

---

## 2. 符号与对象

- \(\phi\in[0,1]\)：log-frequency；\(\omega(\phi)=b^{-\phi}\)（\(\phi=0\) 最高频，\(\phi=1\) 最低频）。
- \(K=d_{\mathrm{rot}}/2\)：rotary pair 数；标准 MHA 常取 \(d_{\mathrm{rot}}=d_{\mathrm{head}}\)。
- \(\rho\ge0\)，\(\int_0^1\rho=1\)：连续 allocation 密度；离散通道 \(\phi_k=F_\rho^{-1}(u_k)\)。
- 连续 Geometric：\(\rho\equiv1\)。实际表还需指定离散网格：核心实验用 \(u_k=(k+\tfrac12)/K\) 的 midpoint-Geo，native RoPE 常用 \(u_k=k/K\)；两者不是同一 finite table。
- Pure-tether 族：

\[
\rho_\tau(\phi)=\frac{\tau\cosh\bigl(\tau(1-\phi)\bigr)}{\sinh\tau}
\quad(\tau>0),\qquad
\rho_0\equiv1.
\]

  故 \(\rho_\tau(0)=\tau\coth\tau>1\)，\(\rho_\tau(1)=\tau/\sinh\tau<1\)：相对均匀，**质量更偏向高频端**。这是公式事实，不是“低频更多”的口语叙事。
- Pearson load stiffness（未归一）：

\[
S_0(\tau)
:=\int_0^1\frac{1}{\rho_\tau(\phi)}\,d\phi-1
=\frac{\sinh\tau\cdot\arctan(\sinh\tau)}{\tau^2}-1.
\]

  小 \(\tau\)：\(S_0(\tau)=\tau^4/45-2\tau^6/315+O(\tau^8)\)。
  级数在 \(0\) 处收敛半径 \(\pi/2\)，**不能**控制 \(\tau=4\)。
- Phase variance：

\[
q(x)=\mathrm{Var}_{t\sim U[0,1]}[\cos(xt)]
=\frac12+\frac{\sin(2x)}{4x}-\Big(\frac{\sin x}{x}\Big)^2,
\quad
q(x)=\frac{x^4}{45}+O(x^6).
\]

- 完整 RoPE pair 的 exact relative-position logit 可写为

\[
\ell_k(\Delta)
=a_k\cos(\omega_k\Delta)+b_k\sin(\omega_k\Delta),
\]

  其中 \(a_k,b_k\) 由 token content、layer/head 与已训练 Q/K 权重决定。因此频率表是 finite Fourier dictionary；不给 task kernel、distance prior 与 trained amplitudes，不存在 objective-free 的通用最优表。

---

## 3. 问题一：均匀是否最优？非均匀何时更优？

### 3.1 必须相对目标回答

| 目标 \(J[\rho]\) | 均匀是否最优 | 非均匀是否更优 | 层级 |
| --- | --- | --- | --- |
| 固定 \([0,1]\) 上 uniform log-frequency 的 \(K\)-点平方量化 / 最坏覆盖 | 是，midpoint 等间距网格唯一（不计排列） | 否 | `[严格]` |
| 固定 support 上最大熵，或 \(D_{\mathrm{KL}}(\rho\|1)\) | 是，唯一 | 否 | `[严格]` |
| \(\frac12\langle\rho,T\rho\rangle\)，\(T1\) 为常数且 \(T\) 在零均值子空间严格正定 | 是，唯一 | 否 | `[严格]` |
| \(\mathcal C_{\mathrm{app}}\)，\(\beta=0\) | 是，唯一 | 否 | `[严格]` |
| \(\mathcal C_{\mathrm{app}}\)，\(\beta>0\) | 否 | 是：唯一最优为 \(\rho_{\tau}\)，\(\tau=\sqrt{\beta/\alpha}\) | `[严格]` |
| \(-\,U_{\mathrm{tr}}\)（无 stiffness，全空间） | 否 | 是，但最优是 \(q\) 的 argmax 集中，**不是 cosh** | `[严格]`（线性目标） |
| \(\tfrac12 S(\rho_\tau)-\lambda U_{\mathrm{tr}}(\rho_\tau)\) 沿 pure-tether 族 | \(\tau=0\) 当 \(Q_1\le0\) 或 \(\lambda=0\) | \(\tau_*>0\) 当 \(Q_1>0\) 且 small-\(\tau\) 平衡成立 | `[有条件]` |
| \(D_{\mathrm{KL}}(p_0\|p_\tau)+\text{Pearson stiffness}\) | 是（两项非负，\(\tau=0\) 同时为零） | 否 | `[严格]` |
| \(\tfrac12S-\lambda D_{\mathrm{KL}}\) 的 small-\(\tau\) 局部式 | 由同为 \(O(\tau^4)\) 的二次-\(\theta\) 系数决定稳定性 | 四阶 balance 本身不选出 finite small \(\tau_*\) | `[严格]` |
| 指定 checkpoint / task 的 \(R(\theta)\) | 取决于 \(A_{\mathrm{task}},B_{\mathrm{task}}\) | 仅当 task 方向导数支持 | `[有条件]` / `[开放]` |
| Exact kernel 碰撞二次型 | 未证 | cosh 未证为全局最优；仅有 functional 诊断叙事 | `[开放]` / `[经验]` |
| 训练后 LM / 检索任务 | 无定理 | 无定理 | `[经验]` 或 `[开放]` |

### 3.2 Shape 定理（可坚定保留）

对

\[
\mathcal C_{\mathrm{app}}[\rho]
=\frac{\alpha}{2}\int_0^1\rho^2\,d\phi
+\frac{\beta}{2}\iint\rho(\phi)\rho(\psi)\min(\phi,\psi)\,d\phi\,d\psi,
\]

在 \(\alpha>0,\beta\ge0\)，\(\rho\in L^2\)，\(\rho\ge0\)，\(\int\rho=1\) 上：

1. 两二次项均 PSD（\(\min\) 核有 \(\iint f\min f=\int(\int_s^1 f)^2\ge0\)），故 \(\mathcal C_{\mathrm{app}}\) **凸**；\(\alpha>0\) 使 \(\frac\alpha2\|\rho\|_2^2\) 本身严格凸，故在可行集上 **严格凸** → 唯一最小。
2. 内点 Euler–Lagrange + 边界条件 \(\rho'(0)=-\tau^2\)，\(\rho'(1)=0\)，\(\tau=\sqrt{\beta/\alpha}\)，得
   \(\rho_\tau(\phi)=\tau\cosh(\tau(1-\phi))/\sinh\tau\)。
3. \(\rho_\tau\ge\tau/\sinh\tau>0\)，不等式约束不活跃。
4. \(\beta=0\Rightarrow\rho\equiv1\)；亦为 \(\tau\to0\) 极限。

**这只证明：给定该 surrogate，shape 最优是 cosh。**
不证明：surrogate = exact kernel，或 = attention，或 = LM。

**注（正文易错）**：\(\mathcal C_{\mathrm{app}}[1]=\alpha/2+\beta/6\neq0\)。
“两者在 uniform 处都消失”对 \(\mathcal C_{\mathrm{app}}\) 不成立。若需在 uniform 归零的代价，应使用 Bregman 中心化

\[
D_{\mathcal C}(\rho,1)
=\frac{\alpha}{2}\|\rho-1\|_2^2
+\frac{\beta}{2}\langle\rho-1,T_{\min}(\rho-1)\rangle\ge0.
\]

### 3.3 为何 geometric 不是“RoPE 数学最优”

Su 等 geometric 表是 **log-uniform 设计**，不是从 LM 变分导出的唯一解。但“不是普适最优”也不等于“没有严格最优性”：

1. **Uniform log-frequency 量化** `[严格]`：在 \([0,1]\) 的 uniform \(\phi\) prior 下，

   \[
   D(\{\phi_k\})=\int_0^1\min_k|\phi-\phi_k|^2\,d\phi
   \]

   的最优 Voronoi cell 中心是 midpoint，cell length \(\ell_k\) 的代价为 \(\ell_k^3/12\)。由 \(\sum_k\ell_k=1\) 与严格凸性，唯一解（不计点的排列）是等长 cell 与 midpoint geometric grid。同一网格也唯一最小化最坏 log-frequency 覆盖半径。
2. **最大熵 / 无信息 prior** `[严格]`：固定 support 时，\(\rho\equiv1\) 唯一最大化 \(-\int\rho\log\rho\)，等价地唯一最小化 \(D_{\mathrm{KL}}(\rho\|1)\)。
3. **对称正二次目标** `[严格]`：对 \(J=\frac12\langle\rho,T\rho\rangle\)，若 \(T1\) 为常数，且 \(T\) 在零均值子空间严格正定，则 uniform 是归一仿射集上唯一最小。
4. **本文 surrogate** `[严格]`：\(\beta=0\) 时 uniform 唯一；\(\beta>0\) 时 \(T_{\min}1\) 非常数，uniform 不是 stationary，cosh 严格更优。

反过来，如果给定 task-weighted log-frequency distortion \(w(\phi)\)，高分辨率量化近似为

\[
D_K[\rho]\simeq\frac{1}{12K^2}
\int_0^1\frac{w(\phi)}{\rho(\phi)^2}\,d\phi,
\qquad
\rho^*(\phi)\propto w(\phi)^{1/3}.
\]

这是 high-rate 条件结论：仅当 \(w\) 为常数时恢复 uniform。它说明非均匀能有严格理由，但仓库尚未从 trained LM 中识别这个 \(w\)。

因此，有限 \(K\) 只使 **allocation 问题良定**（finite spectral budget）；不单独给出最优 \(\rho\)。最优性必须与 prior / kernel / risk 绑定。

### 3.4 非均匀优于均匀的**可证明**窗口

**窗口 S（surrogate）** `[严格]`
\(\beta>0\Rightarrow\mathcal C_{\mathrm{app}}[\rho_\tau]<\mathcal C_{\mathrm{app}}[1]\)（唯一最小且 \(\rho_\tau\neq1\)）。

**窗口 T（transport，沿 pure-tether）** `[有条件]`
展开 \(\rho_\tau=1+\theta\eta+O(\theta^2)\)，\(\theta=\tau^2\)，

\[
\eta(\phi)=\frac12(1-\phi)^2-\frac16,\qquad\int_0^1\eta=0.
\]

则

\[
U_{\mathrm{tr}}(\rho_\tau;L)
=U_{\mathrm{tr}}(1;L)+\frac{M}{L}Q_1(L,b)\,\theta+O\!\Big(\frac{M\theta^2}{L}\Big),
\]

\[
Q_1(L,b)=\int_0^1\eta(\phi)\,q(Lb^{-\phi})\,d\phi.
\]

独立数值（\(b=5\cdot10^5\)，与既有审计一致）：

| \(L\) | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| \(Q_1\) | 0.0301 | 0.0314 | 0.0319 | 0.0316 | 0.0305 | 0.0288 | 0.0265 |

在 **tested grid** 上 \(Q_1>0\)：相对均匀，pure-tether 方向对 \(U_{\mathrm{tr}}\) 有一阶正增益。
**不是**全域 \(b,L\) 的普遍证明；小 \(b\)、大 \(L\) 需另验符号。

**窗口 K（exact kernel）** `[经验]/`[开放]`
论文 functional test：在固定离散配置上，由 cosh 导出的表相对 geometric 降低所定义 collision score。
这支持“方向性诊断”，**不是** “\(\rho_\tau=\arg\min\) exact kernel”。
`c_coll=1.171` 闭环 **已撤回**（验证脚本未做真优化；存在远更优可行点）。不得再用其证明 surrogate–kernel 定量一致。

### 3.5 均匀不输或反超的窗口

| 条件 | 机制 |
| --- | --- |
| \(\beta=0\) | 无累积重叠惩罚 |
| Utility = ordinary KL | 一阶为零，与 \(O(\tau^4)\) stiffness 同阶 |
| \(Q_1\le0\) | pure-tether 对 \(U_{\mathrm{tr}}\) 无一阶正增益 |
| 注意力极度 peaky | \(p_0=1/L\) 与 \(J\propto P/L\) 失效；需保留 directional \(c^\top J(p)^2c\) |
| Task gradient 与 \(\eta\) 反号 | proxy 与任务可分离（`[开放]`） |
| 只比 free \(U_{\mathrm{tr}}\) 最优 | 最优非 cosh，几何也不是；比较对象变了 |

### 3.6 Finite-\(K\) 实现：当前对照不是 fixed-span pure shape

midpoint inverse-CDF 的 practical map 是

\[
u_k=\frac{k+1/2}{K},
\qquad
\phi_\tau(u)=1-\frac{\operatorname{asinh}((1-u)\sinh\tau)}{\tau}.
\]

对 \(u\in(0,1),\tau>0\)，由 `sinh` 在正半轴严格凸且过原点，

\[
\sinh((1-u)\tau)< (1-u)\sinh\tau
\quad\Longrightarrow\quad
\phi_\tau(u)<u.
\]

所以每个 sampled \(\omega_k=b^{-\phi_k}\) 都向高频移动。这是 `[严格]` 的 finite-grid 事实，不是训练结果。例如 \(K=32,b=500\mathrm K,\tau=4\) 的公式重算给出：

| 量 | EVQ / midpoint-Geo |
| --- | ---: |
| 最高 sampled frequency | \(1.166\times\) |
| 最低 sampled frequency | \(3.173\times\) |
| realized natural-log span | \(12.712\to11.711\) |

因此 submitted Geo→EVQ 对照严格说是 **shared midpoint-quantile convention 下的 schedule intervention**：它不是简单换 base，但也没有将 density shape 与 realized extrema/span、active-channel behavior 分离。若 reviewer 追问 pure-shape causality，需要的是同 \(K\)、同 realized extrema 的 affine-rescaled EVQ / uniform 诊断；这会定义新 control，不能声称现有结果已完成隔离。

### 3.7 Waterbed（只守 allocation 命题）

在适当条件下 allocation 发散不等式（Pearson / Burg 型）可对 \(\rho\not\equiv1\) 给出严格代价。
**不能**推出：长程 PPL 必升、短程 PPL 必降、二者构成 task Pareto、cosh 为 task 最优。
安全措辞：经验 trade-off **consistent with** allocation waterbed，而非 **implied by**。

### 3.8 问题一的收口句

> Geometric 不是 RoPE 的普适最优；它在 uniform log-frequency 量化/覆盖、最大熵和一类对称正二次目标下有严格最优性，并等于 \(\mathcal C_{\mathrm{app}}\) 在 \(\beta=0\)（及 pure-tether \(\tau\to0\)）时的最优。
> Cosh 是 **stated convex collision surrogate** 的唯一最优 shape，不是 exact kernel / attention / LM 的闭式最优，也不是 transport 线性目标在全空间的最优。
> 非均匀可优于均匀的可证窗口：\(\beta>0\) 的 \(\mathcal C_{\mathrm{app}}\)，或 \(Q_1>0\) 时 pure-tether 上的 \(U_{\mathrm{tr}}\)。
> 当前 midpoint 实现还同时改变 finite-grid extrema/span，因而实验只能支持 schedule-level effect，不能声称已识别 pure density-shape causality。
> 理论根基应收缩为上述分层，而不是“均匀才是真理”或“cosh 全局最优”。

---

## 4. 问题二：\(\tau\) 的第一性原理

### 4.0 先给定 task：真实局部最优的一般式

设 \(\theta=\tau^2\ge0\)。由 inverse-CDF 在 Geo 附近的展开，

\[
\phi_\tau(u)
=u-\theta\frac{u(1-u)(2-u)}{6}+O(\theta^2),
\]

因而完整 RoPE logits 可写为 \(z(\theta)=z_0+\theta g+O(\theta^2)\)，但 \(g\) 含 trained \(a_k,b_k\) 与任务数据。假设对固定 checkpoint \(W_0\) 的明确任务 risk 在 \(\theta=0\) 附近二次可微，

\[
R(W_0,\theta)
=R_0+A_{\mathrm{task}}\theta
+\frac12B_{\mathrm{task}}\theta^2+o(\theta^2).
\]

其中 \(A_{\mathrm{task}}=\partial_\theta R(W_0,0)\)，\(B_{\mathrm{task}}=\partial_{\theta\theta}R(W_0,0)\)。

当 \(B_{\mathrm{task}}>0\)，且所得 candidate 仍落在 remainder 受控的局部邻域时，局部二次模型的约束最优是

\[
\boxed{\tau_*^2=\theta_*=\left[-\frac{A_{\mathrm{task}}}{B_{\mathrm{task}}}\right]_+}.
\]

- \(A_{\mathrm{task}}\ge0\)：Geo 是该方向的局部最优；
- \(A_{\mathrm{task}}<0,B_{\mathrm{task}}>0\)：小的非零 warp 可能改善任务；
- \(B_{\mathrm{task}}\le0\)：二阶局部式不能选出 finite 最优，需更高阶或直接测量。

若对每个 \(\theta\) 都允许权重重训练，\(R^*(\theta)=\min_W R(W,\theta)\)。在 \(W_0\) 为内点局部最优且 \(H_W=R_{WW}\succ0\) 时，envelope / implicit-function 计算给出

\[
\frac{dR^*}{d\theta}(0)=R_\theta,
\qquad
\frac{d^2R^*}{d\theta^2}(0)
=R_{\theta\theta}-R_{\theta W}H_W^{-1}R_{W\theta}.
\]

这个 Schur complement 表明 pretrained co-adaptation 一般不能被 \(d,L\) 两个标量代替。仓库尚未估计 \(A_{\mathrm{task}},B_{\mathrm{task}}\) 或该 cross-Hessian，所以 trained-task \(\tau_*\) 是 `[开放]`，不能从下面的 model-free proxy 直接代入。

### 4.1 严格否证：ordinary KL 路径

设 \(\theta=\tau^2\)，\(z_\theta=z_0+\theta g+O(\theta^2)\)，\(p_\theta=\mathrm{softmax}(z_\theta)\)，\(A(z)=\log\sum_i e^{z_i}\)。则

\[
D_{\mathrm{KL}}(p_0\|p_\theta)
=A(z_\theta)-A(z_0)-p_0^\top(z_\theta-z_0).
\]

因 \(\nabla A(z_0)=p_0\)，

\[
\partial_\theta D_{\mathrm{KL}}(p_0\|p_\theta)\big|_{\theta=0}=0,
\]

\[
D_{\mathrm{KL}}(p_0\|p_\theta)
=\frac{\theta^2}{2}g^\top J_{\mathrm{sm}}(p_0)g+O(\theta^3)
=O(\tau^4).
\]

Pearson stiffness 局部亦 \(O(\tau^4)\)。若两者相加，\(\tau=0\) 因两项非负而最小；若写成 \(\frac12S-\lambda D_{\mathrm{KL}}\)，同阶系数只判断 \(\theta=0\) 稳定/失稳，**不能**在局部四阶 balance 中选出 \(\theta_*\propto L^{-1}\) 从而 \(\tau_*\propto L^{-1/2}\)。

正文 Proposition 将 \(U\) 写成含 \(O(\tau^2)\) 的 “KL gain”，与附录 KL 二阶展开 **自相矛盾**。
**判定：ordinary-KL 导出部署尺度 = 不成立。** 任何被触发的理论回答都必须纠正该命名；未被点名时是否作合并 integrity disclosure，由 playbook 的作者决策门裁决。

### 4.2 条件路径：transport proxy 的定义

Diffuse baseline \(p_0=1/L\)，\(J_{\mathrm{sm}}(p_0)=L^{-1}P\)，\(P=I-\mathbf{1}\mathbf{1}^\top/L\)。
对单通道 \(c_\omega(j)=\cos(\omega j)\)，

\[
\|J_{\mathrm{sm}}(p_0)c_\omega\|_2^2
=\frac{1}{L^2}\|P c_\omega\|_2^2
\;\xrightarrow{\text{连续化}}\;
\frac{q(\omega L)}{L}.
\]

（因 \(J^2=L^{-1}J\)，同量也等于 per-position Fisher 形式 \(L^{-1}c^\top J c\)。）

在 \(M\) 个可加、幅值归一、交叉项可忽略的通道上定义 **proxy**（不是 KL，不是 task loss）：

\[
U_{\mathrm{tr}}(\rho;L)
=\frac{M}{L}\int_0^1 q(Lb^{-\phi})\,\rho(\phi)\,d\phi.
\]

这一式只对 cosine pattern 做了幅值归一的方差诊断。完整 pair 还有 sine pattern、cos/sin covariance 与 content-dependent \(a_k,b_k\)；不同通道的 trained amplitudes 与 cross-term 一般不能严格“吸收进一个 \(\lambda\)”。因而 \(U_{\mathrm{tr}}\) 是 deliberately normalized 的 cosine-only proxy，不是 full RoPE-pair objective。

### 4.3 沿 pure-tether 族的 leading balance

**Stiffness 约定（必须写清，否则 prefactor 混乱）**

论文正文常用 \(d_{\mathrm{head}}\)-归一：

\[
S_{\chi^2}(\tau)=\frac{1}{d_{\mathrm{head}}}S_0(\tau)
=\frac{\tau^4}{45\,d_{\mathrm{head}}}+O(\tau^6).
\]

审计笔记常用 \(M=d_{\mathrm{head}}/2\) 归一。二者只差常数因子，**指数结构相同**，数值 \(\tau_*\) 不同。

取 \(M\)-归一写法以便与 \(U_{\mathrm{tr}}\) 的 \(M\) 对齐：

\[
S(\theta)=\frac{\theta^2}{45M}+O(\theta^3/M),
\qquad
U_{\mathrm{tr}}=U_0+\frac{M}{L}Q_1\theta+O(M\theta^2/L).
\]

令

\[
F(\theta)=\frac12 S(\theta)-\lambda U_{\mathrm{tr}}(\theta,L).
\]

Leading stationarity：

\[
\boxed{
\theta_*=45\lambda Q_1\frac{M^2}{L},
\qquad
\tau_*=\sqrt{45\lambda Q_1}\,\frac{M}{\sqrt{L}}
}
\quad\text{在 }Q_1>0\text{ 时}.
\]

一般维数（stiffness 维 \(d_S\)，utility 通道数 \(d_U=M\)）：

\[
\tau_*^2\sim 45\lambda Q_1\frac{d_S\,d_U}{L}.
\]

**维度幂次不是 normalization-invariant。** 上式同时采用了 stiffness 的 \(1/d_S\) 归一与 utility 的 \(d_U\) 求和。若两项都按通道求和，或都按通道平均，共同的 \(M\) 可以抵消；不同 normalization 也可得 \(\sqrt d\) 而非 \(d\) 的依赖。在 \(\lambda\) 没有与实际 task risk 标定之前，不得将 \(d_Sd_U\) 当作第一性原理预测。

**额外假设** \(d_S\propto d\) 且 \(d_U\propto d\) 之后，才有

\[
\tau_*\propto\frac{d}{\sqrt{L}}.
\]

MLA 中 \(d_{\mathrm{eff}}=d_{\mathrm{head}}\) **不是**该推导的必然结果，是 architecture-specific convention（且 Primary III 实际 \(d_{\mathrm{rope}}\) 与 ad-hoc \(d_{\mathrm{eff}}\) 身份以 full audit 为准）。

**假设清单（缺一则降级）**

1. Utility 定义为 \(U_{\mathrm{tr}}\)，不是 ordinary KL / task loss；
2. Diffuse \(p_0=1/L\)；
3. 通道可加、幅值固定、cross-term 可忽略；
4. 忽略完整 cos/sin pair 的方向差异，并假设 Q/K 能量与 cross-term 可用统一常数近似；
5. \(Q_1(L,b)>0\) 且在讨论范围内缓变；
6. small \(\theta\)（practical \(\tau\sim4\) **不在** remainder 控制内）；
7. 变分限制在 pure-tether 族（非全空间、非 forced/Bessel 分支）；
8. 若要写成 \(d/\sqrt{L}\)，需 \(d_S,d_U\propto d\)，且需固定上述非对称的求和/平均 convention；
9. Diffuse support 覆盖全部 \(L\) 个位置。对只在 \(m\) 个位置上均匀的 local/sparse attention，\(J=m^{-1}P_m\)，Jacobian dilution 是 \(1/m\) 而非 \(1/L\)；距离相位还取决于被选 offsets。一般 trained attention 应保留 directional 量 \(c^\top J(p)^2c\)，不能用单一标量 \(L_{\mathrm{eff}}\) 无损替代。

### 4.4 Surrogate 尺度 ≠ 部署尺度（逻辑硬分隔）

| | Surrogate 内 \(\tau_{\mathrm{surr}}=\sqrt{\beta/\alpha}\) | 部署 / proxy \(\tau\) |
| --- | --- | --- |
| 来源 | 拟合 \(\alpha,\beta\) 或设定重叠比 | \(S\) vs \(U_{\mathrm{tr}}\) 或经验 basin |
| 典型 \(L\) 依赖 | \(\sim L^{-0.11}\)（附录拟合量级） | \(\sim L^{-1/2}\) |
| 典型 \(d\) 依赖 | \(\sim\sqrt{d}\) | 当前非对称 normalization 下 \(\sim d\) 或 \(\sim M\)；换 convention 可改变或消去该幂次 |
| 决定什么 | shape 族坐标（若坚持用 \(\mathcal C_{\mathrm{app}}\) 系数） | operating point on the family |

**不得**写：“由 \(\mathcal C_{\mathrm{app}}\) 唯一推出部署 \(\tau=d/\sqrt{L}\)”。
正确：**surrogate → family；proxy/sweep → coordinate on the family。**

### 4.5 Small-\(\tau\) 与 practical \(\tau\)

| \(\tau\) | exact \(S_0\) | \(\tau^4/45\) | lead/exact |
| ---: | ---: | ---: | ---: |
| 0.5 | 0.001297 | 0.001389 | 1.07 |
| 1.0 | 0.01745 | 0.02222 | 1.27 |
| 1.414 | 0.05827 | 0.08884 | 1.52 |
| 2.0 | 0.1803 | 0.3556 | 1.97 |
| 4.0 | 1.617 | 5.689 | 3.52 |

- Exact cosh warp 在任意 \(\tau\) 可算（实现层）。
- Leading balance **只**在 diffuse full-support convention 下激励 \(L^{-1/2}\) 结构。
- \(\tau=4,5.66,8\) 等部署点由 **经验 basin** 承担，不是 Taylor 定理覆盖。

### 4.6 与经验轨迹的对照（对齐，不升级）

仓库中可追溯的三组早期证据不是同一 scaling protocol：

| \(L_{\mathrm{train}}\) | 实际观察 | 它能支持什么 | 不能支持什么 |
| ---: | --- | --- | --- |
| 128 | 125M 早期报告在 eval 8K 的最佳 tested point 是 \(\tau=5\)，且仍在 sweep 上边界 | 截至 \(5\) 未见峰；该区间内更大 warp 继续有利 | 不能写真实 \(\tau_*>5\)，更不能定位有限最优 |
| 1024 | 同报告的 eval 8K 仅有 Geo、\(2.0\)、\(2.5\) 与一个 learnable 点；\(2.0\) 略好于 \(2.5\) | \(\tau\approx2\) 是可用邻域 | 稀疏网格与无方差不能给 precise optimum |
| 2048 | 50M TinyStories eval 16K 的 8 点 dense sweep，seed 42 上 \(1.5\) 最好；125M 的两 seed 只验证 selected \(1.5\) 对 Geo（seed 42 另有 \(0.2\)） | \(1.5\) 是这三组中最强的 selected operating-point evidence | dense optimum 本身仍是 single-seed；两 seed 方向验证不等于两 seed 全谱选择 |

来源：`docs/exp/2026-02-26_full_experiment_report.md:150-220,511-552`；`docs/exp/2026-02-27_evq_tau_sweep_results.md:9-54`。这些组还改变了 dataset、model size、training budget 与 extrapolation ratio（128→8K 是 \(64\times\)，1024→8K 与 2048→16K 是 \(8\times\)）。因而它们支持的最窄结论是：

> 在这些不同 protocol 中，有用的 \(\tau\) 随 \(L_{\mathrm{train}}\) 增大而下降；它们与 \(64/\sqrt L=(5.66,2.00,1.41)\) 在数值上 compatible，但没有验证指数 \(-1/2\)、线性 \(d\) 依赖或单位 prefactor。

**两个不得升级的 fit**

1. 历史报告的 forced-origin \(67.84/\sqrt L\) fit 只有 \(R^2=0.76\)，且混入 right-censored / indirect 点。它不是独立验证。
2. `scripts/analysis/verify_stiffness_and_regime.py` 在特定 cosine proxy、Pearson stiffness、\(L\in[128,4096]\) 和数值优化约定下得到有限区间指数 \(0.465\)。这是 **proxy sensitivity diagnostic**，不是 observed-sweep fit；同脚本调制 stiffness 形状 \(p\approx0.80\) 即可得 \(0.498\)，反而说明它不能作为新导出。

Phase16 也只能作为 fallible basin prior：full audit 为 99 runs / 9 configs / selected-confirmation，共同三 seed 的 formula-vs-Geo 比较是 7/9 胜、2/9 负；不得恢复“27 configs / all <1% / near-optimal”。

### 4.7 \(\tau\) 理论上“应该”取什么？

| 问题 | 答案 | 层级 |
| --- | --- | --- |
| 无额外目标的唯一 \(\tau^*\)？ | 不存在 | `[严格]` |
| 仅给定 \(\mathcal C_{\mathrm{app}}\)？ | \(\tau=\sqrt{\beta/\alpha}\) 定 shape；\(\beta/\alpha\) 需另定 | `[严格]` |
| 给定 \(U_{\mathrm{tr}}+S\)、small-\(\tau\)、族限制？ | \(\tau_*=\sqrt{45\lambda Q_1}\,M/\sqrt{L}\)（或 \(d_S,d_U\) 版） | `[有条件]` |
| 给定 fixed checkpoint + task？ | 局部式为 \(\tau_*^2=[-A_{\mathrm{task}}/B_{\mathrm{task}}]_+\)（\(B>0\)；系数未测） | `[严格]`（形式）+ `[开放]`（数值） |
| 给定 retrained LM？ | 还需 weight-adaptation Schur complement；不能只用 \(d,L\) | `[有条件]` / `[开放]` |
| local/sparse attention？ | diffuse \(1/L\) 需改为 support- 和 direction-dependent Jacobian 量 | `[严格]`（diffuse 反例边界）/ `[开放]` |
| 部署实践？ | \(\tau=d_{\mathrm{eff}}/\sqrt{L}\) 作为 basin selector | `[经验]` + 条件动机 |

### 4.8 Forced 分支与 pure-tether（不夸大）

加 Fisher 型源后 \(\rho''-\tau^2\rho=\gamma b^{-2\phi}\)，全解 = 齐次 cosh + 特解。
EVQ 取 \(\gamma=0\) 是 **闭式可逆** 的设计选择。
\(L^1\)/CDF 级 residual 叙事与 quantile 放大 \(\sim\sinh\tau/\tau\) 表明：大 \(\tau\) 时 forced 分支不宜用 “可忽略” 一笔带过。
**不得**写：practical residual 已由训练系统测得并受控。

### 4.9 问题二的收口句

> Ordinary KL 不能导出非零 \(d/\sqrt{L}\)。
> 在 diffuse probability-transport / phase-variance proxy、通道可加、pure-tether 族限制、small-\(\tau\) 与写明的求和/平均 convention 下，leading balance 给出 \(\tau\propto\sqrt{d_S d_U/L}\)；再经维度认同才得到 \(d/\sqrt{L}\) 形态。该维度幂次不是 normalization-invariant，sparse/local attention 也会把 \(1/L\) 改成 support- 和 direction-dependent 量。
> 对指定 task，真实局部式是 \(\tau_*^2=[-A_{\mathrm{task}}/B_{\mathrm{task}}]_+\)，而这两个系数尚未测。部署公式因此只是结构动机加单位约定与有限 \(\tau\) 经验 basin，不是全局任务最优，也不是 \(\mathcal C_{\mathrm{app}}\) 拟合尺度的同一解。

---

## 5. 统一可守 / 必撤 / 待定

### 5.1 可守 `[严格]` 或干净的 `[有条件]`

| ID | 内容 |
| --- | --- |
| R1 | Finite \(K\) ⇒ allocation 是合法设计轴（finite spectral budget 视角） |
| R2 | Geometric 在 uniform log-frequency 量化/覆盖、最大熵和写明的对称正二次目标下严格最优 |
| R3 | \(\mathcal C_{\mathrm{app}}\) 严格凸；cosh 为唯一最小；geometric = \(\beta=0\) / \(\tau\to0\) |
| R4 | CDF / inverse-CDF / pure-tether warp 闭式；finite midpoint 表对 extrema/span 的改变可精确计算 |
| R5 | Ordinary KL 从 \(O(\tau^4)\) 起；不能 alone 导出 \(L^{-1/2}\) |
| R6 | \(q(x)\) 的 variance 恒等式与 \(q\sim x^4/45\) |
| R7 | \(S_0(\tau)\) 闭式与 \(\tau^4/45\) leading |
| R8 | 在 §4.3 假设与 normalization 约定下，proxy balance \(\Rightarrow\tau\propto\sqrt{d_S d_U/L}\) |
| R9 | 指定 fixed task 的局部最优形式取决于 \(A_{\mathrm{task}},B_{\mathrm{task}}\)；数值未测 |
| R10 | Allocation waterbed 不等式本身（不作 task 解释） |

### 5.2 必撤或禁止外推

| ID | 内容 |
| --- | --- |
| X1 | “\(O(\tau^2)\) ordinary KL gain” 导出 \(\tau^*\) |
| X2 | \(\tau=d/\sqrt{L}\) 全局最优 / trained-task theorem |
| X3 | \(\tau_{\mathrm{surr}}\) 与 \(\tau_{\mathrm{deploy}}\) 同一最优解 |
| X4 | cosh = exact kernel 或 LM 最优 |
| X5 | cosh = 自由 \(U_{\mathrm{tr}}\) 最优 shape |
| X6 | `c_coll=1.171` 与 2% agreement 闭环 |
| X7 | Phase16：27 configs、all <1%、near-optimal 全表 |
| X8 | Waterbed ⇒ PPL Pareto |
| X9 | MLA \(d_{\mathrm{eff}}=d_{\mathrm{head}}\) 由理论唯一决定 |
| X10 | small-\(\tau\) remainder 覆盖 \(\tau=4\) |
| X11 | submitted midpoint EVQ = fixed-extrema / fixed-span pure-shape control |
| X12 | 三个异质 sweep 点已验证 \(-1/2\) 指数、linear-\(d\) 或 unit prefactor |
| X13 | Pearson proxy 数值指数 \(0.465\) 是 empirical law 或 independent derivation |

### 5.3 仅经验 / 诊断

| ID | 内容 |
| --- | --- |
| E1 | Tested grid 上 \(Q_1>0\) |
| E2 | 部署规则与部分 sweep 落在同一 PPL basin（口径以 full audit 重算为准） |
| E3 | Exact-kernel collision score 的方向性降低（非 minimizer） |
| E4 | 异质 protocol 中的 best-tested / selected useful \(\tau\) 随 \(L\) 下降，数值上 compatible with \(c\,d/\sqrt L\)；不是 exponent fit |

---

## 6. 与论文表述的对应修正（理论层，不改 tex）

| 论文易读法 | 复核后应读法 |
| --- | --- |
| Cosh 优化 “the” collision problem | Cosh 优化 **stated** \(\mathcal C_{\mathrm{app}}\) |
| \(U\) = post-softmax KL gain \(O(\tau^2)\) | \(U_{\mathrm{tr}}\) = diffuse transport / phase-variance score；KL 为 \(O(\tau^4)\) |
| \(\tau^*=d/\sqrt{L}\) 由 softmax transport 定理给出全局点 | 条件 proxy 在特定 normalization 下给 **结构**；单位与 finite \(\tau\) 为 **basin selector** |
| 三个 sweep 点验证 \(L^{-1/2}\) law | 它们只支持“useful \(\tau\) 随 \(L\) 下降”；protocol 异质且有 right-censoring / sparse-grid / single-seed selection |
| Surrogate 是 sole approximation | 至少还有：pure-tether、cosine-only proxy、diffuse、proxy≠task、Pearson 选择、求和/平均 convention、small-\(\tau\)、MLA convention |
| \(\mathcal C_{\mathrm{app}}\) 与 \(\mathcal W\) 在 uniform 归零 | \(\mathcal C_{\mathrm{app}}[1]\neq0\)；需 Bregman 中心化 |
| Geometric 是 \(\tau\to0\) 的 “RoPE 极限” | 仅 pure-tether 子族内；forced 分支 \(\tau\to0\) 不回到同一对象 |
| Submitted EVQ vs Geo 是 pure shape isolation | 是 shared midpoint-quantile schedule contrast，但 finite extrema/span 也变；没有 fixed-span causal isolation |
| Proxy 等于 trained task | 真实 task 局部取决于 \(A_{\mathrm{task}},B_{\mathrm{task}}\)；retraining 再加 Schur complement |

---

## 7. 内部可复用英文短句（非发送稿；须经真实 review 与作者批准）

**Shape**

> Under the stated convex broadband surrogate, the cosh density is the unique positive minimizer. Geometric RoPE is recovered as the vanishing-overlap-penalty (\(\tau\to0\)) limit of that family, not as a universal optimum of RoPE or of the LM objective.

**Scale / KL**

> Ordinary KL between the baseline and schedule-perturbed attention distributions has zero first variation and begins at \(O(\tau^4)\). It therefore cannot, by itself, be balanced against the local \(O(\tau^4)\) Pearson stiffness to produce a nonzero \(L^{-1/2}\) operating point.

**Proxy**

> The quantity that does admit a first-order allocation gain is a normalized cosine phase-variance score \(U_{\mathrm{tr}}=(M/L)\int q\rho\). Restricted to diffuse attention and the pure-tether family, and under the stated additivity and normalization conventions, balancing this proxy against Pearson stiffness yields \(\tau\propto\sqrt{d_S d_U/L}\). Its dimension dependence is convention-sensitive, and we do not identify it with the full RoPE pair or task loss.

**Two scales**

> The surrogate fixes an analytic shape family; a separate proxy motivates local scale structure; empirical sweeps select the practical basin. The fitted surrogate scale and the deployed \(d/\sqrt{L}\) rule are distinct layers and do not share the same power-law identity.

**Uniform vs non-uniform**

> Uniform allocation is strictly optimal for uniform log-frequency quantization/coverage, maximum entropy, and the pure diagonal (\(\beta=0\)) surrogate. Non-uniform cosh is optimal for the written surrogate whenever \(\beta>0\), and is a first-order improvement of \(U_{\mathrm{tr}}\) along the pure-tether direction when \(Q_1>0\). Neither statement implies task-level dominance, nor that cosh maximizes \(U_{\mathrm{tr}}\) over all densities.

**Finite grid / task**

> Our midpoint comparison uses the same quantile convention, but the finite EVQ table also shifts both sampled extrema and does not hold the realized log-frequency span fixed (it narrows in the representative \(K=32,b=500\mathrm K,\tau=4\) setting); it is therefore a schedule-level contrast rather than a fixed-span pure-shape isolation. For a specified checkpoint and task, the local optimum depends on task derivatives, \(\tau_*^2=[-A_{\mathrm{task}}/B_{\mathrm{task}}]_+\) when \(B_{\mathrm{task}}>0\), which we have not estimated.

---

## 8. 开放问题（诚实边界）

1. \(Q_1(L,b)\) 的普遍符号与渐近，超出 tested grid。
2. Peaky / trained attention 下 directional \(c^\top J(p)^2c\) 的可证简化；当前没有单一 \(L_{\mathrm{eff}}\) 的无损定理。
3. Forced / Bessel / cosh 谁更接近 exact-kernel 全局最优。
4. \(d_S\) 与 \(d_U\) 的 normalization 是否有 task-grounded 规范；当前 \(d\) 幂次不唯一。
5. Task loss 对 pure-tether 方向的 \(A_{\mathrm{task}}\) 符号与 \(B_{\mathrm{task}}\) 曲率 —— **完全开放**，属实验而非本笔记。
6. Fixed-span 对照下，density shape 是否仍有独立的 kernel / task 优势。
7. 将层 B 与层 C 并入单一 variational principle 且保持闭式可逆 —— 未完成。

---

## 9. 独立复核清单（本文写作时已做）

| 检查项 | 结果 |
| --- | --- |
| \(\rho_\tau=1+\tau^2\eta+O(\tau^4)\)，\(\eta=\frac12(1-\phi)^2-\frac16\)，\(\int\eta=0\) | 通过；\(\tau=1\) 时 \((\rho-1)/\tau^2\) 对 \(\eta\) 的偏差已 \(O(10^{-2})\)，提示 small-\(\tau\) 边界 |
| \(q(x)\) 级数与数值 | 通过 |
| \(Q_1(L)\) 表 | 与 2026-07-11 审计一致 |
| \(S_0=\sinh\tau\arctan(\sinh\tau)/\tau^2-1\) | 数值与闭式一致 |
| Ordinary KL 一阶为零 | 分析确认 |
| \(U_{\mathrm{tr}}\) 对 \(\rho\) 线性 ⇒ 自由最优非 cosh | 分析确认；\(q\) 的 argmax 随 \(L\) 在 \(\phi\in(0,1)\) 内移动 |
| Uniform midpoint 量化 / 覆盖最优性 | cell-distortion 与严格凸性复核通过 |
| Finite-grid \(K=32,b=500\mathrm K,\tau=4\) 的 extrema/span | 按 `scripts/lib/rope/schedules.py:94-140` 重算：\(1.166\times\), \(3.173\times\), \(12.712\to11.711\) |
| Task-local \(A/B\) 式与 retraining Schur complement | envelope / implicit-function 展开复核通过；仓库未测数值 |
| \(\tau_{\mathrm{surr}}\) vs \(\tau_{\mathrm{deploy}}\) 幂次 | 附录拟合量级 vs \(L^{-1/2}\) 冲突，必须分层 |
| 早期 \(L=128/1024/2048\) sweep 口径 | 回查两份 `docs/exp/` 报告；right-censoring、sparse grid 与 single-seed selection 已标出 |
| Pearson proxy 指数 | 本轮 fresh 运行 `scripts/analysis/verify_stiffness_and_regime.py`：\(0.4652\)；\(p=0.80\) 为 \(0.4982\)，只作 sensitivity |
| `c_coll` / Phase16 口径 | 不恢复；服从 full audit |

未在本笔记中重新训练模型、未改代码、未改论文表格数字。

---

## 10. 一页纸总答

**均匀是否最优？**
不是普遍最优，但在 uniform log-frequency 量化/覆盖、最大熵、对称正二次目标及 \(\beta=0\) 的 \(\mathcal C_{\mathrm{app}}\) 下有严格最优性。当本文 surrogate 的 \(\beta>0\) 时，唯一最优 shape 是 cosh。Transport 上，仅当限制在 pure-tether 且 \(Q_1>0\) 时，非均匀有可证的一阶优势；自由最优则不是 cosh。这些都不是 trained-task theorem。

**\(\tau\) 应由什么决定？**
对指定 task，局部答案是 \(\tau_*^2=[-A_{\mathrm{task}}/B_{\mathrm{task}}]_+\)（\(B>0\)；这些系数取决于已训练权重与数据，目前未测。对 model-free proxy，在 pure-tether、diffuse、small-\(\tau\)与指定 normalization 下，平衡 Pearson stiffness 与 \(U_{\mathrm{tr}}\) 得 \(\tau\propto\sqrt{d_S d_U/L}\)。部署 \(d/\sqrt{L}\) 是该结构加 dimension convention 与经验 basin；它与 surrogate 内部的 \(\sqrt{\beta/\alpha}\) **不是**同一条定律。

**理论还剩什么？**
Exact：有限谱预算视角 + stated surrogate 的闭式 shape + finite-grid 范围改变。
Conditional：cosine transport proxy 在写明假设/归一下的 \(L^{-1/2}\) 结构。
Empirical：prefactor、dimension law、finite \(\tau\)、任务表现；当前只有异质 PPL basin 信号。
撤回：KL 阶数错误、尺度混层、全局最优与错误校准叙事。
