# `optimization_notes.md` — 对 `evq_three_completions.tex` 的审查与优化

> **生命周期状态（2026-08-29）**：历史/支持性推导，不是当前方法路线或实验队列。
> O1/O3 的条件推导与 O5 的负结果可保留；O2 的经验机制没有获得支持；O4 仅是
> 指定 surrogate 下的数学构造；O6--O8 是非优先级开放问题。本文不授权计算。

面向 *ROPE HAS A SPECTRAL BUDGET* + 推演笔记三部分。每条注明 **Part** 与状态。
所有数值来自 `verify_three_completions.py` 与本文附的 `verify_optimizations.py`（纯 NumPy/SciPy，CPU ~60 s）。

**历史一句话结论**：这些笔记在指定假设与 surrogate 内整理了 $L_{\rm rng}$
和对角/交叉项，并为 re-adaptation 只给出线性化替代。arcsine 预测被数值
**证伪**；分辨率跃变是 surrogate 内的一个可行修正，不是已验证的方法结论。

---

## O1 — $L_{\rm rng}$ 不是自由参数（Part I）· **已完成**

**问题**：笔记 Part I 引入两个长度 $L_{\rm eff}^J$（key 计数）与 $L_{\rm rng}$（相位量程），只给了前者的定义，后者停在文字层面。$\varphi_*$ 依赖 $L_{\rm rng}$，因此整条公式悬空。

**推导**。附录 A.10 的 $q(x)=\tfrac12+\tfrac{\sin 2x}{4x}-(\tfrac{\sin x}{x})^2$ 被写成 $x=\omega L$ 的函数，是**假设 $\Delta\sim\mathrm{Unif}[0,L]$ 的产物**。对任意分离先验 $\mu$，单通道相位方差是

$$q_\mu(\omega)\;=\;\operatorname{Var}_{\Delta\sim\mu}[\cos(\omega\Delta)]\;=\;\tfrac12+\tfrac12 A_\mu(2\omega)-A_\mu(\omega)^2,\qquad A_\mu=\operatorname{Re}\psi_\mu .$$

代入 $\mu=\mathrm{Unif}[0,L]$，$A_\mu(t)=\sin(tL)/(tL)$，逐项还原论文的 $q(\omega L)$。**数值核对**（$L{=}4096$，$\omega\in[10^{-4},10^{-1}]$）：$q_\mu/q_{\rm paper}\in[0.9997,1.0005]$。

于是 $L_{\rm rng}$ 由 $\mu$ **导出**：把笔记里的对数矩平衡改写成先验版本，

$$\int_0^\infty\Big[q_\mu(\omega)-\tfrac12\mathbf 1\{\omega>\omega_0\}\Big]\frac{d\omega}{\omega}=0,
\qquad \boxed{\;L_{\rm rng}:=x_0/\omega_0(\mu),\quad \varphi_*=\min\Big\{1,\tfrac{\log(1/\omega_0)}{\log b}\Big\}.}$$

**命题 O1.** 对 $\mu=\mathrm{Unif}[1,L]$ 有 $\omega_0 L=2.0772$，与笔记的普适常数 $x_0=2.0743$ 相差 $0.14\%$；一般先验下 $L_{\rm rng}$ 是 $\mu$ 的泛函，不等于窗口长度、也不等于 $\mu$ 的任何单一矩。

**数值**（$\mu_\alpha\propto\Delta^{-\alpha}$ on $[1,4096]$）：

| $\alpha$ | 0.0 | 0.5 | 1.0 | 1.5 | 2.0 | 2.5 |
|---|---|---|---|---|---|---|
| $L_{\rm rng}$ | 4090 | 4959 | 1227 | 8.8 | 0.3 | 0.0 |
| $L_{\rm rng}/L$ | 1.00 | 1.21 | 0.30 | 0.0022 | $10^{-4}$ | — |
| $\mathrm{sd}_\mu$ | 1182 | 1220 | 855 | 256 | 50 | 9.5 |

**推论（可证伪）**：$\alpha\ge1.5$ 时 $\varphi_*\to0$，$Q_1\to0$，模型给出 $\tau_*=0$ —— 强局部先验下**不该做任何重分配**。这与 Part III 中 $\alpha\ge2$ 时 EVQ 优势降到 $1.1\times$ 完全一致，两条独立路径互相印证。

**验证方案**：`verify_optimizations.py --O1`；把 $\mu$ 换成从 checkpoint 实测的注意力加权距离分布即可直接得到部署用的 $\varphi_*$。

---

## O2 — $L_{\rm eff}^J$ 与 $\mu$ 的条件关系（Part I）· **代数条件保留；经验机制未获支持**

**问题**：Part I 把 $L_{\rm eff}^J$ 和 $\mu$ 当成两个独立输入，理论因此不自洽（同一份注意力被计了两次、口径不同）。

**推导**。若 $g_i(j)\approx G(r_{ij})$（A.12 的传输方向在固定层/头上主要是距离的函数），则 $\operatorname{Var}_{p_i}[g_i]=\operatorname{Var}_{\mu_i}[G]$ 而 $\|P_ig_i\|^2=n\operatorname{Var}_{U_n}[G]$，故

$$\boxed{\;L_{\rm eff}^J\;=\;n\cdot\frac{\operatorname{Var}_{U_n}[G]}{\operatorname{Var}_{\mu}[G]}\;\xrightarrow[\;G(r)\simeq\gamma r\;]{}\;n\cdot\frac{\operatorname{Var}_{U_n}[\Delta]}{\operatorname{Var}_{\mu}[\Delta]}\;}$$

**命题 O2.** 由 $g_i(j)=O(r_{ij})$，注意力越局部 $\operatorname{Var}_\mu[\Delta]$ 越小，$L_{\rm eff}^J$ 越**大**。这解析地解释了 audit 的实测反直觉结果。

**生命周期修正。** 上式依赖 $g_i(j)\approx G(r_{ij})$ 的距离主导假设。后续仓库内
实测注意力距离分布没有支持下面反演出的局部尺度，因此 O2 不能继续标为已完成的
机制解释；它只保留为带显式假设的代数关系。该负向校准不否定恒等式在满足假设的
其他协议中成立。

**反演出的可证伪预测**。把 audit 的 $\kappa_{\rm att}=4.6476\times10^{-4}$ 代回（probe 的 query 位置 $63,127,255,383,511$，$n=64\ldots512$）：

$$\text{隐含 }\ \mathrm{sd}_\mu\approx 46\ \text{tokens（跨 query 位置均匀聚合）},\quad 72\ \text{tokens（}n{=}512\text{ 处）}.$$

对照均匀注意力的 $\mathrm{sd}_{U_{512}}=148$。**直接用 probe 已存的 $p$ 张量重算注意力加权距离的标准差即可判定**：若落在 $40$–$80$ tokens，恒等式成立、$L_{\rm eff}^J$ 有了机制解释；若接近 $148$，说明 $g$ 的内容依赖不可忽略，恒等式失效。

**不动点收缩性**（补 Part I Cor. 1.11 缺的严格性）。由两点割线 $dL_{\rm eff}/d\tau\approx-100$，映射 $T(\tau)=c\,d_{\rm eff}/\sqrt{L_{\rm eff}(\tau)}$ 的 Lipschitz 模

$$|T'|=\tfrac12 c\,d_{\rm eff}L_{\rm eff}^{-3/2}\big|\tfrac{dL_{\rm eff}}{d\tau}\big|\approx 0.036\ll 1 .$$

Banach 不动点唯一，一次迭代即达 $3.6\%$ 精度 —— 所以"测一次、重建一次"就够，不需要迭代训练。

---

## O3 — $\gamma_{\rm eff}$：A.11 的 stiffness 扫描在拟合一个模型并不预测的函数形式（Part I）· **已完成**

**问题**：A.11 用 $\tau_*\propto L^{-\gamma}$ 拟合，报 $\chi^2$ 给 $\gamma=0.465$（目标 $0.5$），并由此引入 $p\approx0.85$ 的"指数匹配"诊断。

**推导**。有了闭式 $Q_1=\tfrac1{12}g(\varphi_*)$，$g(\varphi)=\varphi(1-\varphi)(2-\varphi)$，$\varphi_*=\log(L/x_0)/\log b$：

$$\boxed{\;\gamma_{\rm eff}(L,b)=-\frac{d\log\tau_*}{d\log L}=\frac12-\frac{g'(\varphi_*)}{2\,g(\varphi_*)\,\log b},\qquad g'(\varphi)=2-6\varphi+3\varphi^2 .}$$

**命题 O3.** $\tau_*(L)$ 不是幂律。$\gamma_{\rm eff}$ 在 $\varphi_*<1-1/\sqrt3$ 时 $<1/2$、之后 $>1/2$，并在 A.11 的扫描区间内漂移约 $0.1$。

**数值**（$d_{\rm eff}=64$，$L\in[128,4096]$，$\chi^2$ stiffness）：

| $b$ | $\gamma_{\rm eff}$ 逐点 | OLS 单指数（一阶） | OLS 单指数（精确解） |
|---|---|---|---|
| $10^4$ | 0.512 → 0.788 | 0.614 | 0.511 |
| $10^5$ | 0.473 → 0.594 | 0.532 | 0.434 |
| $5\times10^5$ | 0.457 → 0.551 | 0.507 | 0.415 |

A.11 报的 $0.465$ 正落在精确解的 $b$ 依赖带内。**结论**：$\gamma\ne0.5$ 不是 $\chi^2$ 选错了，而是"单一幂律"这个拟合形式本身错。$p\approx0.85$ 的诊断可以撤掉，$\chi^2$ 的公理化理由（A.14 channel-load）反而更稳。

**验证方案**：在三个不同 $b$ 上重跑 A.11 的数值优化，检查 $\gamma$ 是否按 $\gamma_{\rm eff}$ 移动。若移动 → 诊断被解释掉；若不动 → 闭式 $Q_1$ 有问题。

---

## O4 — 用受迫 ODE 合并对角效用与交叉项（Part III → Part I）· **surrogate 推导完成；方法路线已退役**

**问题（笔记 Prop 3.9 的诊断）**：A.9 的 collision 泛函 $\mathcal C_{\rm app}$ 只选**形状**，A.10 的 $\mathcal F$ 只选**尺度**，两者从未合并；A.10 的效用在通道指标上是对角的，看不见交叉冗余。而且 A.9 自己承认存在"两个都叫 $\tau$"的量（$\tau_{\rm surr}\sim\sqrt{d}L^{-0.11}$ vs 部署的 $\tau_*\propto d/\sqrt L$），这本身就是不自洽。

**方案**：合并成单一泛函，让形状与尺度同时由它决定：

$$\mathcal J[\rho]=\underbrace{\tfrac{\alpha}{2}\!\int\!\rho^2}_{\text{channel load}}+\underbrace{\tfrac{\beta}{2}\!\iint\!\rho\rho\min(\phi,\psi)}_{\text{redundancy}}-\underbrace{\lambda\!\int\!\rho\,q_\mu}_{\text{resolution}},\quad \int_0^1\!\rho=1,\ \rho>0 .$$

这正是 A.13 的 "Fisher forcing" 分支，只是把未指定的 $V_b$ 换成**已由 O1 定出的 $q_\mu$**。用 $(G\rho)''=-\rho$ 两次求导消去乘子：

$$\rho''-\tau^2\rho=\frac{\lambda}{\alpha}q_\mu'' ,\qquad
\rho'(0)=-\tau^2+\tfrac{\lambda}{\alpha}q_\mu'(0),\qquad \rho'(1)=\tfrac{\lambda}{\alpha}q_\mu'(1).$$

在锐阈近似 $q_\mu(\phi)=\tfrac12\mathbf 1\{\phi<\varphi_*\}$ 下，源项是 $-\kappa\delta'(\phi-\varphi_*)$，$\kappa:=\lambda/(2\alpha)$，跳跃条件为 $[\rho]=-\kappa$、$[\rho']=0$。解出：

> **定理 O4（EVQ-Cosh-R：分辨率感知的闭式分配）**
> $$\rho_{\tau,\varphi_*,\kappa}(\phi)=\frac{\tau\cosh\!\big(\tau(1-\phi)\big)}{\sinh\tau}
> +\frac{\kappa}{\sinh\tau}\begin{cases}\ \ \sinh\!\big(\tau(1-\varphi_*)\big)\cosh(\tau\phi), & \phi<\varphi_*\\[2pt]
> -\sinh(\tau\varphi_*)\,\cosh\!\big(\tau(1-\phi)\big), & \phi>\varphi_*\end{cases}$$
> 满足：**(i)** $\int_0^1\rho=1$ 对**任意** $(\tau,\varphi_*,\kappa)$ 自动成立；**(ii)** 在 $\varphi_*$ 处下跳恰为 $\kappa$；**(iii)** $\kappa=0$ 时精确退化为论文的 $\rho_\tau$；**(iv)** 正性充分条件 $\kappa<\tau/\sinh(\tau\varphi_*)$；**(v)** 逆 CDF 仍是闭式（分段 arcsinh），零搜索性质保留：
> $$\Phi(u)=\begin{cases}\dfrac{1}{\tau}\Big[\delta+\operatorname{arcsinh}\dfrac{u-1}{R}\Big], & u\le u_*\\[8pt]
> 1-\dfrac{1}{\tau}\operatorname{arcsinh}\dfrac{(1-u)\,\tau\sinh\tau}{\tau-\kappa\sinh(\tau\varphi_*)}, & u\ge u_*\end{cases}$$
> 其中 $P=\coth\tau+\dfrac{\kappa\sinh(\tau(1-\varphi_*))}{\tau\sinh\tau}$，$R=\sqrt{P^2-1}$，$\delta=\operatorname{arctanh}(1/P)$，$u_*=1+R\sinh(\tau\varphi_*-\delta)$。

**数值验证**（已跑）：$\int\rho-1$ 的量级 $10^{-7}$（梯形误差）；跳跃与 $-\kappa$ 逐位相符；$\kappa=0$ 时解析逆 CDF 与论文 Eq.(warp) 的最大偏差 $4.4\times10^{-16}$；一般 $\kappa$ 与数值反演偏差 $\le1.3\times10^{-6}$。

**效果**（$K{=}32$，$b{=}10^4$，$L{=}1024$，$\tau{=}2$，$\varphi_*{=}0.673$；块白化 $r_2$）：

| $\alpha$ | cosh | $\kappa{=}0.2$ | $0.4$ | $0.6$ | $0.9$ | $1.2$ |
|---|---|---|---|---|---|---|
| 0.0 | 20.23 | 22.86 | 26.06 | 29.32 | 34.40 | **38.91** |
| 0.5 | 18.29 | 20.34 | 22.84 | 25.31 | 29.07 | **32.38** |
| 1.0 | 9.47 | 9.92 | 10.37 | 10.77 | 11.28 | **11.66** |
| 1.5 | 4.85 | 4.92 | 4.97 | **5.01** | **5.04** | **5.04** |
| 2.0 | 3.67 | 3.68 | **3.69** | **3.69** | 3.67 | 3.65 |

**在这些已测试的静态 surrogate 设置下，$\kappa>0$ 都不劣于纯 cosh**，且
$\kappa^\*$ 随 $\alpha$ 增大而回落（与 Part III 的过载图像一致）。这给出一个
**surrogate-level 闭式构造**，同时让 $\tau$ 在该泛函内只剩一个含义
（$\sqrt{\beta/\alpha}$）。它不是部署级构造、LM 选择器或当前方法路线。

**历史开放点（非当前优先级）**：$\kappa=\lambda/(2\alpha)$ 仍含 $\lambda$。A.13
给出可测代理 $\lambda\!\leftrightarrow\!\lambda_F\bar\eta_F$；这不构成新的实验建议
或行动队列。

---

## O5 — arcsine 猜想被证伪（Part III）· **已完成（结论为否）**

笔记 §3.7 预言重尾下最优分配应为 arcsine 型（两端堆积）。**在等 stiffness 约束下直接数值最优化 $\bar c$，此预言不成立。**

设置：$K{=}16$，$L{=}512$，$b{=}10^3$，密度用 16 格分段常数参数化，约束 $S_{\chi^2}[\rho]\le S_{\chi^2}[\rho_{\tau=2}]=0.1803$，Powell + 3 次随机重启。

| $\alpha$ | geo | cosh($\tau{=}2$) | **cosh-R**($\kappa^\*{=}1.3$) | 自由密度 best-found（16 格，Powell，3 restarts） |
|---|---|---|---|---|
| 0.0 | 0.1157 | 0.0366 | **0.0115** | 0.0325 |
| 1.0 | 0.2399 | 0.1548 | **0.1358** | 0.1516 |

best-found 自由密度形状（16 格，$\alpha{=}1$，Powell，3 restarts）：`3.01 1.57 1.14 1.17 1.07 0.92 0.82 0.80 0.80 0.77 0.71 0.67 0.65 0.62 0.65 0.63` —— **单调下降 + 快端尖峰 + 尾部平台**，不是 U 型。argmax 在第 0 格；两端>中间为 False。

**结论**：
1. **arcsine 预言证伪**。原因是我把 $c_{\omega\nu}$ 的对数核当成了 log-Riesz 能量，但把它写回 $\phi$ 坐标后领头项是 $\big(\tfrac{\log b}{\log L}\big)^2\min(\phi,\psi)^2$，而 $\min^2$ 并不是 $\partial_\phi^4$ 的 Green 函数（$\partial_\phi^4(K\rho)=-6\rho'-2\phi\rho''$，不闭合），所以"对数位势 ⇒ 平衡测度 ⇒ arcsine"这一步不成立。
2. **O4 的分辨率跃变是一个可行修正**：cosh-R 在两个先验下都优于该次
   16 格、Powell、3-restart 搜索得到的 best-found 自由密度。这只说明该数值搜索
   没有给出上界；不建立连续最优、近最优或全局最优。
3. 未闭合的仍是理论求解：$\alpha_0\rho+\beta_0 K_\mu\rho-\lambda q_\mu+\nu=0$ 是**第二类 Fredholm 方程**，核 $K_\mu(\phi,\psi)=|\psi_\mu(\omega(\phi)-\omega(\psi))|^2$ 正定但非 Green 型。**建议放弃闭式，改用 Nyström**：在 $K\le64$ 的真实通道网格上离散成 $K\times K$ 线性方程组解一次即可 —— 依然零训练搜索。

**历史开放工具（非当前优先级）**：正定非 Green 核上的约束二次规划（Nyström +
单纯形投影）。关于 $\varphi_*$ 附近跃变和快端二次尖峰的判断保持猜想级，不是当前
实验建议。

---

## O6 — re-adaptation 秩界：线性化替代（Part II）· **非优先级开放问题**

**问题**：Part II 的所有秩界约束的是 **emulation**，而 LoRA 做的是 **re-adaptation**，两者不可互推。

**可做的推导（线性化/NTK 一步近似）**。设 $G_Q=\nabla_{W_Q}\mathcal L$、$G_K=\nabla_{W_K}\mathcal L$ 在移植后的 checkpoint 处求值。在 $\|\Delta W\|_F\le\epsilon$ 约束下，由 Eckart–Young，秩 $r$ 更新能取得的一阶下降量

$$|\delta\mathcal L|\;\le\;\epsilon\Big(\sqrt{\textstyle\sum_{j\le r}\sigma_j(G_Q)^2}+\sqrt{\textstyle\sum_{j\le r}\sigma_j(G_K)^2}\Big).$$

> **命题 O6（线性化 re-adaptation 地板）** 秩 $r$ LoRA 至多保留满秩更新一阶下降量的比例
> $$\rho_r=\frac{\sum_{j\le r}\sigma_j(G)^2}{\sum_j\sigma_j(G)^2},$$
> 等号当且仅当更新对齐 $G$ 的前 $r$ 个奇异子空间。

这与 Part II 的 emulation 界**形式同构**（都是"谱前 $m$ 项能量占比"），因此给出一条可判定的桥：

> **待验猜想 O6′**：$\rho_r\approx$ Part II 表 4 中 $s_k$ 谱的能量占比。若成立，emulation 界就是 re-adaptation 界的良好代理，Theorem 3 的定量化就能直接搬到实验上。

**验证方案（一次反向传播，不训练）**：在 8B 移植后的 checkpoint 上取 $\nabla_{W_Q}\mathcal L,\nabla_{W_K}\mathcal L$，做 SVD，画 $\rho_r$ 曲线，与 $s_k$ 曲线叠图。Part II 预测 knee 在 $r\approx20$；若 $\rho_r$ 的 knee 也在那里 → O6′ 成立。

**一般 $W$ 的修正（Part II 的一处不严谨）**。冻结子空间定理在 $W_Q,W_K\ne I$ 时仍成立（$\mathcal N=\ker\Delta Q\cap\ker\Delta K$ 不变），但下界要付条件数代价：

$$\|E(\Delta)\|_2\;\ge\;\sigma_{\min}(W_Q)\,\sigma_{\min}(W_K)\,\big\|P_{W_Q\mathcal N}\,D(\Delta)\,P_{W_K\mathcal N}\big\|_2 .$$

**这是 Part II 秩界的真实弱点**：预训练 Q/K 远非正交，$\sigma_{\min}$ 可能很小，界会松。报秩界时必须同时报 $\sigma_{\min}(W_Q)\sigma_{\min}(W_K)$。

**所需新工具（若要真正闭合）**：秩受限 Fisher/NTK 容量界 —— 把 $\min_{\operatorname{rank}\Delta W\le r}\mathcal L(W+\Delta W)$ 从下方界住，需要 Fisher 信息在秩 $r$ 切流形上的迹，以及损失景观的 PL/局部凸性假设。定性预测：容量界会给出 $r_{\min}\asymp$（移植引入的 Fisher 增量的有效秩），与 $s_k$ 的有效秩同阶 —— 即 O6′ 的理论版本。

---

## O7 — 统一先验口径（Part I/II/III）· **规范完成；异质性推论未验证且非优先级**

目前 A.1（Gram）、A.10（效用）、Part II（秩界）各自默认
$\Delta\sim\mathrm{Unif}$。O1/O7 给出统一记号；O2 行仅在
$g_i(j)\approx G(r_{ij})$ 的显式假设下成立：

| 对象 | 定义 | 出现处 |
|---|---|---|
| $\psi_\mu$ | $\mathbb E_\mu e^{it\Delta}$ | A.1 交叉 Gram（Part III Prop 3.1） |
| $q_\mu(\omega)=\tfrac12+\tfrac12A_\mu(2\omega)-A_\mu(\omega)^2$ | 单通道相位方差 | A.10 效用 → $\varphi_*$（O1） |
| $c_{\omega\nu}\to|\psi_\mu(\omega-\nu)|^2$ | 快频冗余 | Part III Prop 3.3 |
| $s_k=2(1-\operatorname{Re}\psi_\mu(\delta_k))$ | 移植失配 | Part II 秩界 |
| $L_{\rm eff}^J=n\operatorname{Var}_{U_n}[G]/\operatorname{Var}_\mu[G]$ | 有效窗口 | A.12（O2） |

**历史规范假设**：曾建议只测一次 $\mu$ 并由它导出全部下游量；O2 的后续负向
校准说明该统一不能作为当前论文规则或部署处方。

**头间异质性（未验证、非当前优先级）**：真实 $\mu=\sum_h\pi_h\mu_h$ 混合。
$q_\mu$ 对 $\mu$ 非线性，但效用按头聚合时用的是
$\bar q=\mathbb E_h[q_{\mu_h}]$（线性），可能抹平阈值并减小 $Q_1$。这只提出
“共享表可能对异质 $\mu_h$ 次优”的假设；它不是已证明的 Jensen 结果、方法选择
定理或 per-head/grouped allocation 的执行依据。

---

## O8 — 剩余的正则性缺口（Part III）· **非优先级开放问题**

- **$\Lambda(\mu)=\operatorname{tr}(G_0^{-1}M)$ 要求 $m_6<\infty$**。$\mu_\alpha$ 在 $[1,L]$ 上恒有限，但 $L\to\infty$ 时 $\alpha<7$ 即发散，展开半径按 $m_6^{-1/6}$ 收缩。**正确表述**：$2-\|Q\|_F^2=\Lambda(\mu)(\omega^2-\nu^2)^2+O(\epsilon^6)$ 的有效范围是 $\max(\omega,\nu)\ll m_6^{-1/6}$，而非 $\ll 1/L$。
- **先验侧天花板**：笔记 Lemma 3.8 给的 $(\sum M_{ii})^2/\sum M_{ii}^2$ 是有效上界（$\alpha{=}2$ 时 6.8–8.4 vs 实测 $r_2$ 3.3–3.8），但朴素参与比 $1/\sum\mu_i^2$ **不是**上界（$\alpha{=}2$ 给 2.5 < 3.8）。已在笔记中标注，此处重申：不要引用后者。

---

## 总结

**这些笔记保留下来的数学整理**：

1. **统一先验记号**。$L_{\rm rng}$、$\varphi_*$、$c_{\omega\nu}$ 与 $s_k$
   可写成 $\mu$ 的泛函；$L_{\rm eff}^J$ 的同一化额外依赖 O2 的距离主导假设，
   其经验机制未获支持。
2. **surrogate 内合并形状与尺度**。受迫 ODE 在指定泛函中给出闭式
   **EVQ-Cosh-R**；这保留为数学推导，不是部署表或当前方法候选。
3. **A.11 的指数异常被解释掉**。$\gamma_{\rm eff}$ 的闭式说明 $\tau_*(L)$ 不是幂律，$p\approx0.85$ 诊断是拟合形式错误的产物，$\chi^2$ 的公理化选择反而更稳（O3）。
4. **诚实的负结果**：arcsine 猜想证伪（O5）；re-adaptation 秩界在现有框架内不可得，只能给线性化替代 + 可测桥（O6）；一般 $W$ 下秩界要付 $\sigma_{\min}$ 代价（O6）。

**历史预测清单（不构成当前实验队列；P1 已有负向校准）**：

| # | 预测 | 成本 | 判定 |
|---|---|---|---|
| P1 | 50M probe 的注意力加权距离 $\mathrm{sd}_\mu\approx46$–$72$ tokens（非 148） | 读已存的 $p$ 张量 | O2 恒等式成立/失效 |
| P2 | $\tau$ 的最优点在 $0.5\times$ 现规则处（现有 sweep 最低仅到 $0.75\times$） | 加一个训练臂 | Part I 自洽不动点 |
| P3 | 8B LoRA 秩 knee 在 $r\approx20$，$r\le8$ 塌 | 一轮 rank sweep | Part II 秩地板 |
| P4 | $\nabla_{W_{Q,K}}\mathcal L$ 的 $\rho_r$ 曲线与 $s_k$ 能量占比曲线重合 | 一次反向传播 | O6′ 桥 |
| P5 | EVQ-Cosh-R（$\kappa\approx0.5$–$1.2$）优于纯 cosh | 一组小规模训练 | O4 定理的训练级验证 |
| P6 | A.11 的 $\gamma$ 随 $b$ 按 $\gamma_{\rm eff}$ 移动 | 纯数值，$<1$ min | O3 |

**历史工具需求（非当前优先级）**：Nyström 离散化、秩受限 Fisher/NTK
容量界，以及混合先验下的 per-head 阈值分析。
