# EVQ-Cosh 理论问题 Rebuttal：数学审计、主张降级与候选回复组件

日期：2026-07-11

状态：`internal_verified_working_note`

回复状态：`triage-only / needs_author_input`；候选英文仅是内部组件。只有真实 review 的逐字 trigger 到达、完成 comment mapping 且作者批准后，才可组装进入 author response；不得直接复制发送。

用途：作者内部理论 rebuttal 总文档；不是补充材料，不应原样公开提交

范围：只处理理论正确性、理论—实证边界和 reviewer-facing 表述；不修改任何实验数值

---

## 0. 结论先行

这次外部数学审计抓住了一个真实的 P0 问题，但它不是可以照单全收的最终结论。独立重推后的判断是：

- 它对 **ordinary baseline-to-perturbed KL 的阶数**判断正确；
- 它对 waterbed、finite-\(\tau\)、forcing、MLA dimension 和 task-level overclaim 的多数警告正确；
- 但它把 \(L^{-1/2}\) 理论收得过窄：仓库中的 \(q(x)\) 并非凭空假设的 task loss，而是 uniform-softmax 下单通道 phase pattern 的 probability-transport / per-position Fisher proxy。对这个**明确的 proxy objective**，\(O(\tau^2/L)\) 的一阶 gain 与 \(L^{-1/2}\) 局部 scaling 可以条件性严格成立；错误的是把它称为 ordinary KL gain，并进一步当成 trained-task utility。

最重要的结论不是“EVQ-Cosh 理论被整体推翻”，而是必须把理论拆成三个严格不同的层次：

1. **可以坚定保留的精确定理**：给定论文明确写出的凸 surrogate
   \(\mathcal C_{\mathrm{app}}\)，cosh 密度是唯一、严格正的归一化最小解；其边界条件、CDF、逆 CDF 和 \(\tau\to0\) geometric 极限均成立。
2. **可以保留的条件性 proxy theorem**：在 diffuse baseline、固定/各向同性通道幅值、通道可加、以 probability displacement energy 或 per-position Fisher sensitivity 作为 utility proxy 等假设下，仓库里的
   \[
   U_{\mathrm{tr}}(\rho;L)=\frac{M}{L}\int q(Lb^{-\phi})\rho(\phi)\,d\phi
   \]
   对 \(\theta=\tau^2\) 确实有非零一阶项，并给出 \(\tau\propto M/\sqrt L\)。这是 proxy-objective theorem，不是 ordinary KL theorem，也不是 trained-task theorem。
3. **部署规则的正确身份**：\(\tau=d_{\mathrm{eff}}/\sqrt L\) 可以称为“由 conditional transport proxy 提供结构动机、由 sweep 校准和验证的 operating default / basin selector”。单位常数、trained-attention extension 与 MLA \(d_{\mathrm{eff}}\) 仍是经验部分。

真正的 P0 数学错误是：当前论文把 \(O(\tau^2)\) 的 RoPE/logit 扰动与 ordinary post-softmax KL 的量级混为一谈。设 \(\theta=\tau^2\)，若

\[
z_\theta=z_0+\theta g+O(\theta^2),
\]

则 baseline 与 perturbed attention 之间的 ordinary KL 在基线处一阶变分为零：

\[
D_{\mathrm{KL}}(p_0\|p_\theta)
=\frac{\theta^2}{2}g^\top J_{\mathrm{sm}}(p_0)g+O(\theta^3)
=O(\tau^4),
\]

而不是 \(O(\theta)=O(\tau^2)\)。因此，不能再说“baseline-to-perturbed ordinary KL 的 \(O(\tau^2)\) gain 与 \(O(\tau^4)\) stiffness 平衡导出 \(L^{-1/2}\)”。

这项修正**不改变任何已训练模型、频率表或实验数值**。它改变的是理论对象的命名和外推边界：从“ordinary KL / trained utility 的结构定律”改为“精确 surrogate shape + conditional probability-transport proxy theorem + empirical basin rule”。

推荐的 rebuttal 总姿态是：

> 主动认下 ordinary-KL 解释错误；不要把 proxy theorem 一并撤掉。应明确重定义 utility 为 diffuse-softmax 下的 probability-transport / phase-variance proxy，给出其假设，再把实际部署规则定位为 proxy-motivated、empirically calibrated basin selector。与此同时，严格限定 waterbed、MLA \(d_{\mathrm{eff}}\)、pure-tether 和 finite-channel bounds。

---

## 1. 本文核查了什么

本审计对照了以下层次，而不是只复述外部分析：

- 当前正文理论：`paper/sections/03_theory.tex`；
- 当前证明附录：`paper/appendix/a1_proofs.tex`；
- 当前 epistemic map 与 \(\lambda\) 表：`paper/tables/table_epistemic_map.tex`、`paper/tables/table_lambda_cv.tex`；
- 当前 limitations：`paper/sections/06_limitations.tex`；
- 当前 rebuttal 控制室、claim ledger、response draft 与 reviewer 原文；
- 仓库内较早的数学审计：`docs/theory/THEORY_MATH_VALIDATION.md`；
- 后来声称“softmax transport gap 已闭合”的文档：`docs/theory/THEORY_IRONCLAD.md`；
- 99-run operating-basin 报告及其 curated manifest；
- 外部 `Mathematical Audit of EVQ-Cosh` 全文。

仓库内部其实已经留下了一个重要冲突：

- `docs/theory/THEORY_MATH_VALIDATION.md:126-131` 曾把 \(\tau=d_{\mathrm{head}}/\sqrt L\) 明确列为 conjecture；
- `docs/theory/THEORY_IRONCLAD.md:293-313` 后来声称 softmax transport 已经闭合该 gap；
- 新的数学审计说明，后一个“闭合”不能以 ordinary KL 名义成立；独立重推进一步表明，它可以降格并重写成一个明确的 probability-transport proxy theorem，而不必完全退回“没有任何理论”的状态。

### 1.1 独立复算结果

本轮没有把外部审计当作 ground truth，而是单独完成了以下检查：

1. 用 log-partition Taylor expansion 重推 baseline-to-perturbed KL，确认一阶项为零。
2. 直接验证
   \[
   q(x)=\frac12+\frac{\sin2x}{4x}-\left(\frac{\sin x}{x}\right)^2
   =\operatorname{Var}_{t\sim U[0,1]}[\cos(xt)],
   \]
   且
   \[
   q(x)=\frac{x^4}{45}-\frac{x^6}{315}+\frac{x^8}{4725}+O(x^{10}).
   \]
3. 从 uniform softmax Jacobian 独立推出 \(q/L\) 的 probability-transport energy，见 Section 3.2。
4. 数值积分复核 \(b=500\mathrm K\) 时 \(Q_1\) 的符号和 \(L\)-dependence：

   | \(L\) | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
   | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
   | \(Q_1\) | .03009 | .03145 | .03192 | .03159 | .03051 | .02878 | .02646 |

   这确认了 tested grid 上 \(Q_1>0\)，但也确认它不是严格的 \(L\)-independent constant。
5. 复核 exact Pearson stiffness 与 \(\tau^4/45\) 的误差，确认 practical \(\tau\) 不受局部级数控制。
6. 对 variable-coefficient Bessel solution 复核正确 variational boundary conditions。由 modified-Bessel Wronskian 可得 \(I_1(x)/K_1(x)\) 严格递增，因此归一化系数与解均为正；多组 \((a_0,\beta,b)\) 数值检查也满足 mass one、\(y'(0)=-\beta,y'(1)=0\)。这支持外部审计对当前 appendix “Bessel may violate positivity” 的反驳。
7. 代回 ODE、边界和 mass constraint 验证 external audit 给出的 nonresonant forced correction；残差达到浮点精度。

### 1.2 外部审计本身不够准确的地方

外部审计不是无误的，至少有以下需要修正或加限定之处：

1. **它对 scale theory 的最终结论过于保守。** 它正确否定 ordinary KL，但没有充分利用仓库里已经明确给出的 phase-variance transport proxy。该 proxy 可在 diffuse baseline 下独立推出 \(q/L\)，不是纯粹“未来也许存在的 signed utility”。
2. **“\(Q_1\) 不是 softmax curvature”说得过于绝对。** 更精确的说法是：\(Q_1\) 不是 schedule perturbation 的 quadratic KL curvature \(g_\theta^T Jg_\theta\)，也不是 task gradient；但它可以是“每通道 Fisher/transport curvature score 对 allocation density 的一阶变分”。
3. **collision objective 的二阶展开写得不完整。** 若 \(\rho_\theta=1+\theta\eta+\theta^2\xi+\cdots\)，二阶系数还包含 \(\langle\xi,T1\rangle\)。审计写出 \(\frac12\theta^2\langle\eta,T\eta\rangle+O(\theta^2)\) 不能被当作完整二阶公式；它只足以支持一阶项可能非零。
4. **conditional scaling 的 remainder 写得过精确。** \(O(L^{-2})\) 需要固定维度、\(Q_1\) 的规则性以及更具体的高阶系数控制。Rebuttal 中只应写 leading-order relation，不应照抄统一的 remainder。
5. **stationary-phase 常数依赖 Fourier、one-sided/signed 与边界 convention。** 审计已经部分区分 causal 与 signed 情形，但这些系数不应在 rebuttal 中脱离 convention 当作无条件常数。
6. **部分“错误”在当前论文里已被主动披露。** 例如 Pearson 非唯一、near-one-hot 时 \(L_{\mathrm{eff}}^J\) 与 entropy proxy 分离、MLA dimension 是 convention；这些不是新发现。真正的新问题是 ordinary-KL 命名和从 schedule-derivative curvature 到 linear allocation score 的逻辑混用。

因此，本文件后续采用的不是“接受/拒绝整份审计”，而是逐项判定。

### 1.3 对外部审计 34 项 error list 的逐项判定

| # | 外部审计项 | 独立判定 | 对当前 rebuttal 的意义 |
| ---: | --- | --- | --- |
| 1 | ordinary KL order error | **确认**，前提是比较同一路径在 \(\theta=0\) 相等的 baseline/perturbed distributions | 必须纠正 KL 命名 |
| 2 | baseline KL 被叫 gain | **确认**；该 KL 在 baseline 处最小 | 改为 transport-capacity proxy |
| 3 | 两个 \(O(\tau^4)\) 项不能给小非零 optimum | **确认，针对 ordinary KL**；**不适用于**线性 allocation transport score | 不能用 KL balance，但可保留 proxy balance |
| 4 | task gradient 被省略 | **确认，若声称 task loss** | 当前未建立 task theorem |
| 5 | attention target 不明确 | **确认** | 不把 LM label 当 attention-position target |
| 6 | \(Q_1\) 与 curvature 混淆 | **部分确认**；它不是 schedule-KL curvature，但可为 per-channel curvature score 的 allocation first variation | 精确重命名，不应删除 \(Q_1\) |
| 7 | \(Q_1\) sign 未保证 | **确认一般情形**；独立数值确认 tested \(b=500K,L=128\ldots8192\) 为正 | 只报告 tested grid，不写 universal positivity |
| 8 | \(Q_1\) 可能依赖 \(L\) | **确认**；复算从 .03192 降到 .02646 | 只能说 slowly varying on tested range |
| 9 | 常数 \(45\lambda Q_1\) 被吸收 | **确认** | prefactor 是 convention/calibration |
| 10 | surrogate-fit scale 与 deployed scale 混层 | **确认** | 必须保留 shape/proxy/deployment 三层 |
| 11 | small-\(\tau\) 外推到 \(\tau=4\) | **确认不受控** | exact warp 可用，但 Taylor theorem 不覆盖 |
| 12 | 0.465 不等于 0.500 | **确认** | 写 finite-range effective exponent |
| 13 | \(\alpha\delta+\beta\min\) 不是 pointwise exact-kernel approximation | **确认，但当前正文已披露** | 不需要重复自我攻击 |
| 14 | continuum operator-norm obstruction | **确认于 regular continuum kernel**；不适用于已明确的 finite-grid comparison | rebuttal 只 defend finite-grid functional test |
| 15 | strong convexity 不自动给 robustness | **确认** | 需要 residual/norm 才能说 optimizer close |
| 16 | 不同 distance priors 不给同一 cosh optimizer | **确认** | cosh exact only for stated surrogate |
| 17 | stationary-phase normalization ambiguity | **确认 notation 风险** | camera-ready 修正，短 rebuttal 不展开 |
| 18 | causal/signed factor-of-two | **确认 convention 风险** | 不在 rebuttal 报无条件常数 |
| 19 | stationary-phase domain 条件缺失 | **确认** | 需要 boundary 与 \(L/b\) 条件 |
| 20 | Bessel exact scope | **确认** | 只对 variable-local-coefficient surrogate |
| 21 | forced branch 沿用 homogeneous BC 不一致 | **确认**；已代回 ODE/mass 数值验证 | 需要从 source functional 重建 BC |
| 22 | forcing source ambiguity | **确认** | ODE 单独不足以定 BC |
| 23 | \(O(1/\log b)\) 依赖 amplitude scaling | **确认** | 不能无条件说 forcing negligible |
| 24 | mass-small 不等于 pointwise-small | **确认** | density/CDF/quantile 分开报告 |
| 25 | quantile error 有 \(\sinh\tau/\tau\) 放大 | **确认** | practical large \(\tau\) 更需谨慎 |
| 26 | \(\mathcal C_{\mathrm{app}}[1]\neq0\) | **确认**，等于 \(\alpha/2+\beta/6\) | 修正文中 “both vanish” |
| 27 | simple centering 不保证非负 | **确认**；Bregman centering 才自然非负 | waterbed 与 surrogate 不混写 |
| 28 | waterbed 不推出 PPL trade-off | **确认** | 只能写 “consistent with” |
| 29 | Pearson 非唯一 | **确认，但当前 paper 已称 modeling choice** | 保留现有克制语气 |
| 30 | \(\lambda=1\) 非 units 定理 | **确认**；当前 paper 已部分称 convention，但 table 的 “implied curvature” 过强 | 只做 a posteriori alignment |
| 31 | \(L_{\mathrm{eff}}^J\) 非 entropy support | **确认，但当前 appendix 已明确 near-one-hot 分离** | 不是新 P0；避免简写成 support length |
| 32 | \(L_{\mathrm{eff}}^J\) 自动替换 first-order utility | **确认逻辑缺口**，且涉及 channel pattern / schedule derivative 混用 | nonuniform extension 需重做 |
| 33 | MLA dimension 不唯一 | **确认** | \(d_{\mathrm{eff}}=d_h\) 仍是 convention |
| 34 | quantization bound 不推出 PPL gain | **确认** | 只能说 scarce channels 增大 allocation sensitivity |

总体上，34 项中多数数学警告成立；真正需要反驳/修正外部审计的是第 3、6 项的过度收缩，以及它没有把仓库现有 \(q/L\) 识别为可独立定义的 transport proxy。

---

## 2. 理论主张总账

### 2.1 可以坚定保留：数学上已建立

| 主张 | 状态 | Rebuttal 中可以怎么说 |
| --- | --- | --- |
| \(\mathcal C_{\mathrm{app}}\) 在 mass-one 可行集上强凸 | 精确，条件是 \(\alpha>0,\beta\ge0\) | “The stated surrogate is strongly convex and has a unique minimizer.” |
| cosh 密度是唯一最小解 | 精确，条件是所写 surrogate | “The cosh density is the exact optimizer of the stated convex surrogate.” |
| 解严格为正，非负约束不活跃 | 精确 | 可以直接保留 |
| \(\rho'(0)=-\tau^2,\rho'(1)=0\) | 精确 | 可以直接保留 |
| CDF 与 inverse CDF | 精确 | 可以直接保留 |
| \(\tau\to0\) 时回到 geometric | 精确，仅指 pure-tether family | 可以直接保留 |
| Pearson stiffness 的闭式积分 | 精确 | 可以给公式，但不要把它升级成 task-risk theorem |
| ordinary baseline KL 从 \(O(\tau^4)\) 开始 | 精确局部结论 | 必须用来纠正当前 Proposition 解释 |
| Burg/Pearson waterbed 不等式 | 精确，针对 allocation divergence | 可以保留数学不等式本身 |
| 一维 inverse-CDF quantization 的 \(K^{-1}\) transport bound | 精确，在密度正下界等条件下 | 只能说明离散化误差尺度 |

### 2.2 可以保留，但必须写清条件

| 主张 | 必要条件 | 安全定位 |
| --- | --- | --- |
| diffuse-softmax transport proxy 下的 \(L^{-1/2}\) scaling | utility 明确定义为 probability-transport / per-position Fisher proxy；uniform baseline；通道可加/幅值假设；小 \(\theta\)；\(Q_1>0\) 且不改变指数 | conditional proxy theorem |
| signed task utility 下的 \(L^{-1/2}\) scaling | 另需非零 task-loss directional derivative、\(d_U/L\) prefactor 与 sign | future conditional extension，当前未建立 |
| exact-kernel perturbation stability | 必须实际界定有限维 norm、quadratic-form residual 或 solution residual | conditional stability theorem |
| stationary-phase local diagonal coefficient | 距离先验归一化、causal/signed convention、远离频域边界、\(L/b\gg1\) 等 | local asymptotic only |
| modified-Bessel solution | 只对 variable-local-coefficient + min-kernel surrogate 精确 | alternative surrogate solution |
| forced pure-tether correction bounds | 必须指定产生 forcing 的 functional、边界条件及 forcing amplitude | conditional perturbation analysis |
| Pearson 作为 channel-load variance | 必须接受 inverse-load、\(P_\rho\) sampling、二阶中心矩等建模公理 | motivated modeling choice |

### 2.3 只能称为经验校准

| 主张 | 当前正确身份 |
| --- | --- |
| \(\tau=d_{\mathrm{eff}}/\sqrt L\) 的 reviewer-facing 部署含义 | proxy-motivated, empirically calibrated operating convention / basin selector |
| 单位 prefactor | empirical normalization inside the observed basin |
| \(\lambda=1\) | gauge / unit convention after calibration |
| MLA 中 \(d_{\mathrm{eff}}=d_{\mathrm{head}}\) | architecture-specific empirical convention |
| practical \(\tau\approx4\) | empirical operating point，非 small-\(\tau\) theorem 覆盖范围 |
| finite-range exponent 0.465 | numerical effective exponent，不能写成等于 0.5 |
| 忽略 forcing branch | tractability choice with unmeasured residual |

### 2.4 不应再写

- ordinary baseline KL 有 \(O(\tau^2)\) gain；
- \(Q_1\) 就是 schedule perturbation 的 KL curvature 或 task gradient；
- diffuse softmax 在没有明确定义 transport proxy 与通道假设时自动导出了部署 \(L^{-1/2}\) law；
- \(L_{\mathrm{eff}}^J\) 替换自动修复了 scale 推导；
- waterbed inequality 证明了 long-range gain 与 in-range PPL degradation；
- \(\mathcal C_{\mathrm{app}}\) 与 exact kernel 在 continuum global operator norm 下很接近；
- Bessel alternative 可能失去正性，因此 cosh 在数学上更合法；
- transport / quadrature bound 预测 EVQ 一定获得 PPL 增益；
- MLA 的 \(d_{\mathrm{eff}}=d_{\mathrm{head}}\) 由理论唯一决定。

---

## 3. P0：必须在 rebuttal 里修正的核心问题

## P0.1 Ordinary KL 的阶数错误

### 当前问题

`paper/sections/03_theory.tex:93-108` 把 utility 描述为“per-channel post-softmax KL gain”，并写成

\[
U(\tau,L)=\frac{d_{\mathrm{head}}}{L}
\left[Q_0+\tau^2Q_1+O(\tau^4)\right].
\]

同一篇附录 `paper/appendix/a1_proofs.tex:501` 又正确写出了

\[
D_{\mathrm{KL}}(p\|p_\epsilon)
=\frac{1}{2}\epsilon^2g^\top J_{\mathrm{sm}}(p)g+R_3.
\]

而附录前文已经定义 frequency / logit perturbation 对 \(\theta=\tau^2\) 是一阶的。因此若 \(\epsilon=\theta\)，KL 必须从 \(\tau^4\) 开始。正文和附录在这里自相矛盾。

### 为什么一阶项必为零

令 \(p_0=\mathrm{softmax}(z_0)\)，\(p_\theta=\mathrm{softmax}(z_0+\theta g+O(\theta^2))\)，则

\[
D_{\mathrm{KL}}(p_0\|p_\theta)
=A(z_\theta)-A(z_0)-p_0^\top(z_\theta-z_0),
\]

其中 \(A(z)=\log\sum_i e^{z_i}\)。由于 \(\nabla A(z_0)=p_0\)，一阶项严格抵消：

\[
\left.\frac{d}{d\theta}D_{\mathrm{KL}}(p_0\|p_\theta)
\right|_{\theta=0}=0.
\]

所以

\[
D_{\mathrm{KL}}(p_0\|p_\theta)
=\frac{\theta^2}{2}g^\top J_{\mathrm{sm}}(p_0)g+O(\theta^3).
\]

reverse KL 也有相同的二阶主项。

### 对 scale balance 的后果

Pearson stiffness 局部也是 \(O(\tau^4)=O(\theta^2)\)。如果 utility 也只是 ordinary KL 的 \(O(\theta^2)\) 项，那么局部目标为

\[
F(\theta)-F(0)
=\theta^2\left[
\frac{1}{90d_S}-\lambda\frac{d_UC_2}{L}
\right]+O(\theta^3).
\]

这只给出：

- \(\theta=0\) 稳定；或
- \(\theta=0\) 失稳并离开局部区间；或
- 临界点由更高阶项决定。

它不会选出一个小而非零的 \(\theta\propto1/L\)，因此不会导出 \(\tau\propto L^{-1/2}\)。

### Rebuttal 必须怎么说

必须直接承认“KL interpretation”错误，不能只说 shape/scale 是两个 epistemic layers。

候选英文组件（受顶部 author-approval gate 约束）：

> We thank the reviewer for prompting us to re-examine the local transport argument. We identified an order error in our KL interpretation. Writing \(\theta=\tau^2\), the EVQ logit perturbation is \(z_\theta=z_0+\theta g+O(\theta^2)\), but ordinary KL between the baseline and perturbed attention distributions has zero first variation and begins at \(O(\theta^2)=O(\tau^4)\). It therefore cannot, by itself, be balanced against the \(O(\tau^4)\) Pearson stiffness to derive a nonzero \(L^{-1/2}\) operating point. We have accordingly withdrawn that interpretation.

单独写下面这句仍然不够，因为没有定义 proxy：

> The diffuse-softmax assumptions still derive the structural \(L^{-1/2}\) exponent.

如果改成 Section P0.2 中明确的 probability-transport score，并列出 channel additivity / normalization 假设，则可以保留为 conditional proxy theorem。

---

## P0.2 \(Q_1\) 的准确身份：transport-curvature score 的一阶变分

外部审计把 \(Q_1\) 与 Fisher curvature 完全切开，这个判断过强。独立推导显示，仓库中的 \(q(x)\) 确实可以从 uniform softmax 下的 probability transport 得到。

令长度为 \(L\) 的单通道 logit pattern 为

\[
c_\omega(j)=\cos(\omega j),
\]

令 \(P=I-\frac1L\mathbf1\mathbf1^\top\)，diffuse baseline 为 \(p_0=\frac1L\mathbf1\)，则

\[
J_{\mathrm{sm}}(p_0)=\frac1L P,
\qquad
J_{\mathrm{sm}}(p_0)^2=\frac1LJ_{\mathrm{sm}}(p_0).
\]

对一个小的 channel-amplitude perturbation，欧氏 probability displacement energy 为

\[
\|J_{\mathrm{sm}}(p_0)c_\omega\|_2^2
=\frac{1}{L^2}\|Pc_\omega\|_2^2.
\]

在 continuum / dense-grid limit，\(L^{-1}\|Pc_\omega\|_2^2\) 收敛到

\[
q(\omega L)
=\operatorname{Var}_{t\sim U[0,1]}[\cos(\omega Lt)],
\]

所以

\[
\boxed{
\|J_{\mathrm{sm}}(p_0)c_\omega\|_2^2
\simeq\frac{q(\omega L)}{L}.}
\]

由于 \(J^2=(1/L)J\)，同一个量也等于 per-position Fisher curvature \(L^{-1}c_\omega^T Jc_\omega\)。因此 \(q/L\) 不是凭空添加的因子；它是**特定 diffuse proxy** 下可推导的 transport score。

若有 \(M\) 个可加、幅值归一化且 cross-channel terms 在期望下消失的 frequency patterns，则

\[
U_{\mathrm{tr}}(\rho;L)
=\frac{M}{L}\int_0^1q(Lb^{-\phi})\rho(\phi)\,d\phi.
\]

令 \(\rho_\theta=1+\theta\eta+O(\theta^2)\)，则

\[
U_{\mathrm{tr}}(\rho_\theta;L)
=U_{\mathrm{tr}}(1;L)
+\frac{M}{L}Q_1(L,b)\theta
+O(M\theta^2/L),
\]

其中

\[
Q_1(L,b)=\int_0^1\eta(\phi)q(Lb^{-\phi})\,d\phi.
\]

所以 \(Q_1\) 的准确身份是：

> the first variation of a per-channel phase-variance / diffuse probability-transport score with respect to allocation density.

它**不是**：

- baseline geometric attention 与 EVQ attention 之间 ordinary KL 的一阶项；
- schedule derivative \(g_\theta=\partial_\theta z\) 的 quadratic curvature \(g_\theta^TJg_\theta\)；
- downstream task-loss gradient；
- 任意 trained, non-diffuse attention 下自动成立的 utility。

这里存在两个容易混淆、但数学上不同的方向：

1. **channel pattern direction** \(c_\omega\)：用来定义每个已分配频率的 transport capacity，改变 allocation 后总 score 可有 \(O(\theta)\) 变化；
2. **schedule derivative direction** \(g_\theta=\partial_\theta z\)：用来计算从 Geo schedule 移到 EVQ schedule 的 ordinary KL，该 KL 是 \(O(\theta^2)\)。

当前论文把这两种方向混在了一起。修正方式不是删除 \(Q_1\)，而是给 proxy 一个准确名称并删除 ordinary-KL 等同。

`paper/tables/table_lambda_cv.tex` 中 \(c_{\mathrm{pred}}=\sqrt{45Q_1}\) 与 collision optimum 的数值接近，仍只能作为 a posteriori diagnostic alignment；它不能证明同一系数控制 trained-task PPL。

---

## P0.3 \(L^{-1/2}\) 可以保留为 conditional proxy theorem，部署仍是经验 basin rule

### 条件性 proxy theorem

在标准 MHA 下，令 \(M=K=d_{\mathrm{head}}/2\) 为 rotary pair 数。采用 \(M\)-normalized Pearson stiffness：

\[
S(\theta)=\frac{\theta^2}{45M}+O(\theta^3/M),
\]

以及上节推导的

\[
U_{\mathrm{tr}}(\theta,L)
=U_0+\frac{M}{L}\left[Q_1\theta+O(\theta^2)\right],
\qquad Q_1>0,
\]

则对

\[
F(\theta)=\frac12S(\theta)-\lambda U_{\mathrm{tr}}(\theta,L)
\]

的 leading-order stationarity 给出

\[
\boxed{\theta_*=45\lambda Q_1\frac{M^2}{L}},
\qquad
\boxed{\tau_*=\sqrt{45\lambda Q_1}\frac{M}{\sqrt L}}.
\]

由于 \(M=d_{\mathrm{head}}/2\)，把 factor 2 吸收到 normalization 后可以写成 \(d_{\mathrm{head}}/\sqrt L\) 形式。

这条推导在其 proxy assumptions 下是数学自洽的，不应被外部审计一并撤销。但必须列出假设：

- diffuse uniform attention baseline；
- utility 是 probability-transport / per-position Fisher proxy，不是 task loss；
- channel amplitudes 的 normalization 固定；
- cross-channel terms 可忽略或在期望下消失；
- cos 与 sin pair、Q/K activation energy 的影响可吸收到常数；
- \(Q_1>0\) 且在所讨论 \(L\) 范围内缓慢变化；
- small \(\theta\)；
- standard MHA 中 \(M\propto d_{\mathrm{head}}\)。

对于一般 \(d_S,d_U\)，只可写 leading order

\[
\tau_*^2\sim45\lambda Q_1\frac{d_Sd_U}{L};
\]

不要照抄外部审计的统一 \(O(L^{-2})\) remainder，除非补齐固定维度和高阶 regularity 条件。

### 经验部署层

99-run sweep 仍然是独立的经验资产：

- 9 个 \((L,H,d_{\mathrm{head}})\) 配置；
- theory value exact-best 3/9；
- top-2 6/9；
- top-3 8/9；
- observed optimum 都在 rule 的 1.5x 内。

这些结果支持：

> The proxy-motivated rule is an empirically useful default or basin center on the tested grid.

它们不支持：

> The sweep proves that ordinary KL or trained-task loss has the assumed form.

尤其 Phase16 中部分 rule 为 \(\tau=4,5.66,8\)，远离 small-\(\tau\) 区间。因此实际部署值必须由 empirical basin 承担，而不是由局部 remainder 承担。

### 候选英文组件（受顶部 author-approval gate 约束）

> We correct the terminology and scope of the scale argument. Ordinary baseline-to-perturbed KL begins at \(O(\tau^4)\) and is not the \(O(\tau^2)\) utility in our balance. The quantity actually used by the formula is a diffuse-softmax probability-transport proxy: for a channel pattern \(c_\omega\) at \(p_0=1/L\), \(\|J(p_0)c_\omega\|_2^2=q(\omega L)/L\). Summing this score over the allocated channels gives \(U_{\mathrm{tr}}=(M/L)\int q\rho\), whose first variation along \(\rho_\tau=1+\tau^2\eta+O(\tau^4)\) is \((M/L)Q_1\tau^2\). Balancing this explicitly defined proxy against the local Pearson stiffness conditionally yields \(\tau\propto M/\sqrt L\). We do not identify this proxy with task loss or ordinary KL, and practical constants and finite-\(\tau\) values remain empirically calibrated inside the observed basin.

---

## P0.4 必须把 surrogate scale 与 deployed scale 分开

论文自己报告的 surrogate coefficient fit 近似为

\[
\alpha\sim d_{\mathrm{rot}}^{-1},
\qquad
\beta\sim L^{-0.22}.
\]

因此 surrogate 内部的参数

\[
\tau_{\mathrm{surr}}=\sqrt{\beta/\alpha}
\sim\sqrt{d_{\mathrm{rot}}}\,L^{-0.11}.
\]

这与部署规则

\[
\tau_{\mathrm{deploy}}\sim d_{\mathrm{eff}}L^{-1/2}
\]

在 \(L\) 指数和维度指数上都不同。二者不能通过“\(O(1)\) prefactor”吸收，因为幂次不同。

Rebuttal 应该明确：

- surrogate 选择的是一个 tractable one-parameter **shape family**；
- deployed \(\tau\) 是由另一 transport proxy 提供局部动机、再经验校准的 **operating coordinate**；
- 当前理论没有把两者统一成一个 full-attention objective 的同一最优解。

安全句：

> The surrogate determines the analytic family, while a separate diffuse-transport proxy motivates the local scale dependence and empirical sweeps select the practical operating basin; the fitted surrogate scale and the deployed rule are distinct theoretical layers.

---

## P0.5 small-\(\tau\) 不能覆盖 practical \(\tau=4\)

Pearson stiffness 的精确式为

\[
S_{\chi^2}(\tau)
=\frac{1}{d_S}\left[
\frac{\sinh\tau\,\arctan(\sinh\tau)}{\tau^2}-1
\right].
\]

局部展开为

\[
d_SS_{\chi^2}(\tau)
=\frac{\tau^4}{45}
-\frac{2\tau^6}{315}
+O(\tau^8).
\]

该 Taylor series 在零点的收敛半径为 \(\pi/2\)，因此不能在 \(\tau=4\) 收敛。数值误差也已经很大：

| \(\tau\) | exact \(d_SS_{\chi^2}\) | \(\tau^4/45\) | leading-term overestimate |
| ---: | ---: | ---: | ---: |
| 0.5 | 0.001297 | 0.001389 | 7.1% |
| 1.0 | 0.017453 | 0.022222 | 27.3% |
| 2.0 | 0.180326 | 0.355556 | 97.2% |
| 4.0 | 1.616709 | 5.688889 | 251.9% |

正确的 rebuttal 边界是：

- exact cosh implementation 在任何 practical \(\tau\) 都可计算；
- empirical result 不依赖 Taylor 截断来生成频率；
- 但 small-\(\tau\) scaling argument 不能为 \(\tau=4\) 提供受控误差保证。

候选英文组件（受顶部 author-approval gate 约束）：

> The small-\(\tau\) calculation is only a local motivation. Practical values such as \(\tau=4\) use the exact cosh warp and are selected empirically; they are not covered by the Taylor remainder used in the local scaling argument.

---

## P0.6 Waterbed 只能证明 allocation-divergence cost

当前正文 `paper/sections/03_theory.tex:88` 有两个不同层次的问题。

第一，\(\mathcal C_{\mathrm{app}}\) 在 uniform density 处并不为零：

\[
\mathcal C_{\mathrm{app}}[1]
=\frac{\alpha}{2}+\frac{\beta}{6}.
\]

因此“\(\mathcal W\) 与 \(\mathcal C_{\mathrm{app}}\) both vanish at uniform”是错误的。若需要非负、在 uniform 处为零的 surrogate-centered object，应使用 Bregman centering：

\[
D_{\mathcal C_{\mathrm{app}}}(\rho,1)
=\frac{\alpha}{2}\|\rho-1\|_2^2
+\frac{\beta}{2}
\langle\rho-1,T_{\min}(\rho-1)\rangle\ge0.
\]

第二，精确成立的 inequality 是 allocation-space 命题：

\[
\chi^2(U\|P_\rho)
\ge e^{D_B(\rho)}-1.
\]

它说明 nonuniform allocation 具有正的 divergence cost，但不推出：

- long-context PPL 会改善；
- in-range PPL 必然恶化；
- 二者构成 task-level Pareto frontier；
- cosh 是 task risk 的最优 allocation。

安全句：

> The waterbed inequality is an exact allocation-divergence statement. The observed in-range/long-range PPL trade-off is empirically consistent with that statement, but is not implied by it.

因此当前 rebuttal 中关于 learnable \(\tau\) 的“训练目标必然看见 immediate waterbed cost，所以 gradient 系统性指向 \(\tau\to0\)”也不能写成 theorem。最多可说：

> The observed learnable-\(\tau\) behavior is consistent with an objective mismatch: in-range training loss need not reward out-of-range utility. The waterbed bound motivates this interpretation but does not prove the optimization trajectory.

---

## P0.7 MLA 的 \(d_{\mathrm{eff}}\) 不能由通道数唯一决定

MLA 中必须区分：

- \(d_h\)：完整 attention head dimension；
- \(d_r\)：实际 rotated subspace dimension；
- \(K=d_r/2\)：inverse-CDF quantization 的 rotary pair 数；
- Q/K latent projection energy；
- softmax curvature；
- signed task-gradient sensitivity。

频率扰动只经过 rotated subspace。一般形式为

\[
g_j=\frac{1}{\sqrt{d_h}}
q_R^\top\dot R_jk_{j,R}.
\]

其尺度取决于 projection singular values、Q/K covariance、phase derivatives、softmax Jacobian 和 task-gradient alignment。仅凭 \(d_h\) 或 \(d_r\) 不能确定唯一的 \(d_{\mathrm{eff}}\)。

在抽象的条件性 law

\[
\tau^2\propto\frac{d_Sd_U}{L}
\]

中，不同假设会给出不同维度：

- \(d_S=d_U=d_h\)：\(d_{\mathrm{eff}}=d_h\)；
- \(d_S=d_U=d_r\)：\(d_{\mathrm{eff}}=d_r\)；
- \(d_S=d_h,d_U=d_r\)：\(d_{\mathrm{eff}}=\sqrt{d_hd_r}\)；
- anisotropic projection：可能应使用 projection energy 或 effective rank。

所以当前最安全的说法仍是：

> In MLA, \(K=d_{\mathrm{rot}}/2\) is fixed by the number of rotary pairs, whereas \(d_{\mathrm{eff}}=d_{\mathrm{head}}\) is an empirical operating convention for the tested latent-attention path, not a theorem.

MLA 的 primary empirical result不因这一理论降级而消失；但不能把该结果说成验证了 \(d_{\mathrm{eff}}=d_h\) 的一般 law。

---

## 4. P1：重要但不应在短 rebuttal 中全部展开的问题

## P1.1 Exact kernel 与 surrogate 的关系

`paper/sections/03_theory.tex:23-32` 已经正确避免了 pointwise approximation 的说法，这是应保留的优点。

但还需要更严格地区分：

- exact kernel 的 discrete functional score 在测试配置上降低；
- surrogate operator 与 exact operator 在某个 norm 下接近；
- surrogate minimizer 与 exact minimizer 接近。

这三句话不是同一件事。

对于 continuum \(L^2\)，regular bounded kernel 诱导 compact operator；\(T_{\min}\) 也是 compact，而 \(\alpha I\) 不是 compact。因此

\[
T_D-(\alpha I+\beta T_{\min})
=-\alpha I+\text{compact},
\]

其 essential norm 至少为 \(\alpha\)。所以不能普遍声称 global operator-norm error 小于 \(\alpha\)，也不能直接套用 \(\|E\|<\alpha\) 的强凸 perturbation theorem。

可以保留的说法：

> On the deployed finite channel grid, the cosh allocation reduces the evaluated exact-kernel collision diagnostic across the tested configurations.

不要升级为：

> Strong convexity proves that the exact-kernel optimizer is close to cosh.

除非实际给出 finite-dimensional residual 或 quadratic-form bound。

---

## P1.2 Constant \(\alpha\)、stationary phase 与 Bessel branch

constant \(\alpha\) 的安全理由是：

- 它定义了 tractable discrete surrogate；
- 给出 elementary inverse CDF；
- 在所测离散 collision diagnostic 上表现良好。

不能再把“Bessel solution 可能失去正性”作为选择 cosh 的数学理由。对于正的 variable coefficient 与正确的 variational boundary conditions，外部审计给出的 modified-Bessel 解为正；当前 `paper/appendix/a1_proofs.tex:111-117` 关于 Bessel 可能在右端失去正性的表述需要重新证明，否则应删除。

stationary-phase coefficient 还必须区分两种约定：

1. 一侧 causal distance prior \(D_L^+(\Delta)=L^{-1}d(\Delta/L)\)：

   \[
   \alpha_{\mathrm{sp}}(\phi)
   =\frac{\pi d(0)}{2L\log b\,b^{-\phi}}.
   \]

2. signed even normalized prior：

   \[
   \alpha_{\mathrm{sp}}(\phi)
   =\frac{\pi d(0)}{L\log b\,b^{-\phi}}.
   \]

若直接用 physical density \(D_L(0)=d(0)/L\)，则不能再额外乘一个 \(1/L\)。当前 appendix 的 \(D_0\) 定义、\(1/L\) 和 factor-of-two convention 有混淆风险。

短 rebuttal 不需要展开完整 stationary-phase 修正；只需避免用它做过强防御。

---

## P1.3 Pure-tether forcing 不是已控制残差

非齐次 ODE 本身不能决定边界条件；必须从产生 forcing 的 functional 出发。

若 functional 为

\[
\mathcal C_{\mathrm{forced}}[\rho]
=\mathcal C_{\mathrm{app}}[\rho]
-\mu\int v_0e^{-c\phi}\rho(\phi)d\phi,
\]

则

\[
\rho''-\tau^2\rho=\gamma e^{-c\phi}
\]

对应的边界条件是

\[
\rho'(0)=-\tau^2-\frac{\gamma}{c},
\qquad
\rho'(1)=-\frac{\gamma}{c}e^{-c}.
\]

沿用 homogeneous boundary conditions 同时强制 mass one 会不相容。

更重要的是，\(O(1/\log b)\) 的 mass/CDF 误差依赖 forcing amplitude 的尺度选择；pointwise correction 可仍是 \(O(1)\)，而 inverse-CDF error 会被 \(\sinh\tau/\tau\) 放大。没有测量 forcing coefficient，就不能声称 practical \(\tau\) 下 residual 已受控。

安全句：

> The homogeneous branch is retained as a tractable, exactly invertible design family. The omitted forcing residual has not been quantitatively bounded for trained attention at practical \(\tau\).

---

## P1.4 Pearson stiffness 是建模选择，不是 attention 唯一推出

如果显式假设：

- local load 为 \(1/\rho\)；
- channel 按 \(P_\rho=\rho d\phi\) 抽样；
- stiffness 是 centered quadratic moment；

则

\[
\operatorname{Var}_{P_\rho}(1/\rho)
=\int\frac{(1-\rho)^2}{\rho}
\]

确实就是 Pearson divergence。

但 convexity、permutation invariance、uniform 零代价等更弱条件允许很多 \(f\)-divergence。因此应该说“canonical under the stated load-variance axioms”或“motivated choice”，不能说 uniquely derived from attention。

finite-range \(p=1\) 数值 exponent 0.465 与 0.500 也必须按不同数值报告。不能用“接近”替代数学相等。

---

## P1.5 \(\lambda=1\) 不是由单位自动决定

stiffness 与 utility 的相对归一化独立选择。可以把 \(\lambda\) 吸收到 utility 中，但这会改变 \(Q_1\) 的数值。因此没有独立标定时，真正可识别的是 \(\lambda Q_1\)，而不是 \(\lambda\) 单独等于 1。

`table_lambda_cv` 可以保留为 empirical consistency check，但应改为：

> Under the chosen normalization, the calibrated prefactor aligns numerically with the tested collision optimum.

不要说：

> The units derive \(\lambda=1\).

---

## P1.6 \(L_{\mathrm{eff}}^J\) 的真正问题是方向混用，不是 near-one-hot 现象未披露

定义

\[
\frac{1}{L_{\mathrm{eff}}^J}
=\frac{\mathbb E[g^\top J(p)g]}
{\mathbb E[\|Pg\|_2^2]}
\]

是一个关于 quadratic Fisher curvature 的 directional ratio。uniform attention 时，它等于 sequence length；near-one-hot attention 时 \(J(p)\to0\)，该量趋于无穷，而 entropy support size 趋于 1。当前 paper 在 `a1_proofs.tex:493-499` 已经明确披露了后一点，所以这不是外部审计新抓出的隐藏错误。

所以：

- 它不是一般意义上的 entropy-like effective context length；
- 它依赖 \(g\) 的方向，不只依赖 \(p\)；
- 它控制 quadratic KL/Fisher term；
- 它不能未经新假设替换 first-order task utility 中的 \(L\)。

真正的逻辑问题是当前 appendix 对 \(g\) 的含义发生了切换：

- 推导 \(q/L\) 时应使用单通道 phase pattern \(c_\omega\)，并对所有已分配通道的 curvature/transport score 求和；
- `a1_proofs.tex:464-475` 定义的 \(g\) 却是 schedule derivative \(\partial_\theta z\)；
- 对后者，\(g^TJg\) 是 Geo-to-EVQ schedule KL 的二阶系数，不能直接成为 \(U_{\mathrm{tr}}\) 的线性一阶项。

另外，若 nonuniform extension 要延续**欧氏 probability displacement**解释，正确 quadratic form 是 \(g^TJ(p)^2g\)；若要延续 Fisher/per-position curvature 解释，则可以用 \(g^TJ(p)g\)，但必须保留明确的 per-position normalization。uniform baseline 下 \(J^2=(1/L)J\) 使两者相合，nonuniform 时则不再相同。

因此 `paper/appendix/a1_proofs.tex:501` 不能在未固定 utility definition 和 direction 的情况下直接“re-run”出同一 scale balance。

测量 \(L_{\mathrm{eff}}^J\) 仍可作为有价值的 curvature diagnostic，但它不是修复 P0.1 的决定性实验。若要推广 proxy theorem，应重新定义 channel-pattern-level nonuniform transport score，而不是把 schedule derivative 直接代入。

---

## P1.7 Finite-channel bounds 不能推出 PPL 排序

对 \(K=d_{\mathrm{rot}}/2\) 个 midpoint quantiles，generic bounds 如

\[
W_1\le\frac{\sinh\tau}{4\tau K}
\]

是有意义的。它们说明 quantization/transport sensitivity 在通道更少时变大。

但这不推出：

- EVQ 的 transport error 比 Geo 小；
- EVQ 的 task risk 比 Geo 小；
- MLA 的 PPL 增益应按 \(K^{-1}\) 或 \(K^{-2}\) 定量变化。

安全定位：

> Scarce rotary channels amplify sensitivity to allocation and discretization; the direction and magnitude of downstream PPL changes remain empirical.

---

## 5. 对当前 rebuttal 控制面的直接影响

2026-07-12 consolidation 已删除采用旧 ordinary-KL / waterbed 解释的多路径 response、Path A/B、issue-audit 与模拟 crosswalk 文件。它们只可从 Git 历史追溯，不得恢复为活跃入口，也不得复制旧段落到真实 author response。真实 reviews 到来后，唯一 `AUTHOR_RESPONSE_20260722.md` 必须从本文件、Master Ledger 与最新 provenance 重新生成。

当前 rebuttal 可保留的内容是：

- finite spectral budget / third design axis；
- exact cosh optimizer under the stated surrogate；
- exact inverse CDF / zero learned parameters；
- matched-scale EVQ x YaRN empirical anchor；
- Primary II seed-scope；
- MLA scarce-channel empirical stress test；
- PK 与 AR exact 区分；
- 1B schedule-sensitivity limitation；
- tuned-scaler 与 production-scale 边界。

---

## 6. Reviewer 问题分流与逐点策略

## T1. “你们的 \(L^{-1/2}\) 是否真的由 post-softmax theory 推出？”

**判断**：reviewer 是对的；当前 ordinary-KL 推导有阶数错误。

**动作**：`ACCEPT_CORRECTION + SOFTEN_CLAIM`。

**必须回答**：

1. 承认 \(\theta=\tau^2\) 后 ordinary KL 从 \(\theta^2\) 开始；
2. 明确撤回 baseline-KL derivation；
3. 把 ordinary-KL interpretation 改为明确的 probability-transport proxy；
4. 把 local \(L^{-1/2}\) 写成 conditional proxy theorem，把 practical rule 写成 empirical basin selector；
5. 说明实验数据与 exact warp 不受影响。

**不要做**：用 99-run sweep 证明数学推导正确。sweep 证明经验可用性，不证明推导。

---

## T2. “shape 与 scale 是否只是两个拼接的 heuristic？”

**判断**：需要部分承认，但不能撤回 exact surrogate theorem。

**动作**：`PARTIAL + CLARIFY_EXISTING`。

安全回答结构：

- shape：exact under stated surrogate；
- scale：由另一 diffuse probability-transport proxy 条件性动机化，并由 sweep 经验校准；
- 两者不是 full-attention unified optimum；
- 方法贡献仍是一个闭式、可审计、零参数的 allocation family，以及其在测试 regime 的经验效用。

不要说“这两个层次来自一个统一 full-attention objective”。更准确的说法是：shape theorem、transport-proxy theorem 与 empirical calibration 是三个不同证据层。

---

## T3. “constant \(\alpha\) / min kernel 是否只是为了得到 cosh？”

**判断**：这是有效质疑。

**动作**：`CONCEDE_TRACTABILITY + DEFEND_FUNCTIONAL_TEST`。

安全回答：

- constant diagonal + min kernel 是明确选择的 tractable surrogate；
- 不是 exact oscillatory kernel 的 pointwise 或 global operator approximation；
- cosh 是该 surrogate 的 exact optimizer；
- discrete exact-kernel collision diagnostic 与 trained outcomes 是 surrogate 之外的经验检验；
- 不声称 Bessel alternative 数学上不合法。

---

## T4. “waterbed 是否证明了 task trade-off？”

**判断**：reviewer 是对的。

**动作**：`SOFTEN_INTERPRETATION`。

安全回答：

> The theorem is an allocation-divergence bound. The PPL trade-off is observed empirically and is only consistent with the bound.

不要争论“结构一致就等于理论预测”。

---

## T5. “MLA 为什么用 \(d_{\mathrm{head}}\) 而不是 \(d_{\mathrm{rot}}\)？”

**判断**：当前没有唯一理论答案。

**动作**：`CONCEDE_CALIBRATION`。

安全回答：

- \(K=d_{\mathrm{rot}}/2\) 决定实际 quantization；
- \(d_{\mathrm{eff}}=d_{\mathrm{head}}\) 是测试架构的 operating convention；
- 该 convention 尚未通过 direct tau ablation 或 projection/sensitivity measurement 从第一性原理决定；
- empirical MLA result 仍是测试 protocol 下的有效 stress test。

---

## T6. “learnable \(\tau\) 为什么更差？”

**判断**：可以提出机制解释，但不能写成已证明的结构必然。

**动作**：`EMPIRICAL_INTERPRETATION`。

安全回答：

> The three-seed learnable-\(\tau\) result is consistent with objective mismatch: training loss is observed only within \(L_{\mathrm{train}}\), whereas the target extrapolation utility is evaluated out of range. The flat basin and frequency/weight coupling may further weaken or destabilize the training signal. We treat this as an empirical interpretation, not as a theorem implied by the waterbed bound.

若没有完整 trajectory，不要说它“系统性向零漂移”或“震荡”。

---

## 7. Rebuttal 的推荐叙事顺序

理论回复不应从“我们其实已经很诚实”开始。新的数学错误被发现后，最有效的顺序是：

1. **先纠错**：ordinary KL order error；
2. **再说明不受影响的精确核心**：convex surrogate theorem + inverse CDF；
3. **重建准确 proxy**：从 uniform softmax 推导 \(\|Jc_\omega\|^2=q/L\)；
4. **分开 local theorem 与 deployment**：proxy 下得到 local \(L^{-1/2}\)，practical rule 由 empirical basin 校准；
5. **明确其他边界**：waterbed、finite \(\tau\)、MLA dimension；
6. **最后才提 sweep**：说明 rule 在测试 grid 上仍然是有用 default。

这种顺序比“paper 已经通过 epistemic map 披露了 shape/scale 分层”更有说服力。后者无法回答 KL 阶数本身写错的问题。

---

## 8. 候选英文理论回复组件（非 send-ready）

以下文本只用于真实 reviewer trigger 到达后的内部组装。必须先完成逐字 comment mapping，再经作者批准；不得因其措辞完整而预填或直接发送。

### 8.1 推荐标准版

> We thank the reviewer for prompting a closer examination of the scale argument. We identified an order error in our terminology and interpretation of post-softmax KL. Writing \(\theta=\tau^2\), the EVQ schedule perturbation is \(z_\theta=z_0+\theta g+O(\theta^2)\), while ordinary KL between the baseline and perturbed attention distributions has zero first variation and begins at \(O(\theta^2)=O(\tau^4)\). We therefore withdraw the statement that ordinary baseline-to-perturbed KL supplies the \(O(\tau^2)\) utility.
>
> This correction does not affect the exact variational result used to define EVQ-Cosh. For the stated strongly convex surrogate, the unique mass-one minimizer remains \(\rho_\tau(\phi)=\tau\cosh(\tau(1-\phi))/\sinh\tau\), with the stated closed-form CDF, inverse CDF, and geometric \(\tau\to0\) limit. The theorem is conditional on that surrogate and does not claim to solve the exact oscillatory kernel or the trained-transformer objective.
>
> The \(O(\tau^2)\) term in our formula instead comes from an explicitly defined diffuse probability-transport proxy. For a channel pattern \(c_\omega\) at the uniform baseline, \(J(p_0)=P/L\) gives \(\|J(p_0)c_\omega\|_2^2=q(\omega L)/L\). Under channel-additivity and normalization assumptions, summing this score over an allocation gives \(U_{\mathrm{tr}}=(M/L)\int q\rho\); its first variation along \(\rho_\tau=1+\tau^2\eta+O(\tau^4)\) is \((M/L)Q_1\tau^2\). Balancing this proxy against the local Pearson stiffness conditionally yields \(\tau\propto M/\sqrt L\). We do not identify this proxy with downstream task loss or ordinary KL. Practical constants and finite-\(\tau\) values are calibrated by the observed basin, and the MLA choice \(d_{\mathrm{eff}}=d_{\mathrm{head}}\) remains an empirical convention. We also clarify that the waterbed result is an allocation-divergence bound, while the PPL trade-off is empirical.

### 8.2 字数紧张版

> We correct an order error in our KL terminology: ordinary baseline-to-perturbed KL begins at \(O(\tau^4)\), not \(O(\tau^2)\). The \(O(\tau^2/L)\) term actually used by the scale model is a different, explicit diffuse probability-transport proxy, since \(\|J(p_0)c_\omega\|_2^2=q(\omega L)/L\). Under channel-additivity and small-\(\tau\) assumptions this proxy conditionally gives \(\tau\propto M/\sqrt L\); it is not a task-loss or ordinary-KL theorem. The exact cosh minimizer and inverse CDF under the convex surrogate are unchanged, while practical constants, finite-\(\tau\) use, and MLA \(d_{\mathrm{eff}}\) remain empirically calibrated.

### 8.3 如果 reviewer 只问 shape/scale 分离

> We agree that the original presentation connected the layers too strongly. The cosh shape is the exact optimizer of the stated convex surrogate. A separate diffuse probability-transport proxy conditionally motivates the local \(M/\sqrt L\) dependence, while the practical constant and finite-\(\tau\) operating point are selected empirically. This is not a unified first-principles optimum of trained attention.

### 8.4 中文核对

- 必须出现“order error / zero first variation / withdraw baseline-KL interpretation”。
- 必须明确 theorem 保留的是 surrogate theorem，不是 full-attention theorem。
- 必须把 99-run 写成 empirical support，不写成 mathematical validation。
- 必须把 probability-transport proxy 的定义和假设写出来；不要把它改名成 task utility。
- 不要声称实际 \(\tau=4\) 被 small-\(\tau\) remainder 控制。
- 不要用 waterbed 证明 learnable \(\tau\) 的训练轨迹。

---

## 9. 不要写的高风险句子

以下句子会被数学 reviewer 直接反击：

- “The diffuse-softmax Jacobian alone derives the trained-task \(L^{-1/2}\) law.”
- “Ordinary KL provides the \(O(\tau^2)\) gain.”
- “\(Q_1\) is the Geo-to-EVQ KL curvature or downstream task gradient.”
- “The 99-run sweep validates the theoretical derivation.”
- “The exponent 0.465 is the same as 0.5.”
- “The small-\(\tau\) theory explains \(\tau=4\) quantitatively.”
- “\(\lambda=1\) follows from units.”
- “\(L_{\mathrm{eff}}^J\) is the trained model's effective support length.”
- “Measuring \(L_{\mathrm{eff}}^J\) will close the first-order scale theorem.”
- “The exact-kernel optimizer is close to cosh by strong convexity.”
- “Bessel allocations can become negative, so cosh is the valid solution.”
- “The forcing residual is \(O(1/\log b)\) and therefore negligible at \(\tau=4\).”
- “The waterbed theorem predicts the observed PPL penalty.”
- “Scarce-channel quantization bounds prove EVQ's MLA PPL gain.”
- “MLA \(d_{\mathrm{eff}}=d_{\mathrm{head}}\) is derived.”
- “Learnable \(\tau\) must converge to zero by the waterbed theorem.”

---

## 10. 后续 manuscript 修订地图

本轮用户要求的是 rebuttal 文档，不是修改已提交论文。若后续进入 revision/camera-ready，应按下表处理。

| 文件 | 位置 | 必要修订 |
| --- | --- | --- |
| `paper/sections/03_theory.tex` | 15 | 把 Proposition 明确为 conditional probability-transport proxy + empirical rule |
| `paper/sections/03_theory.tex` | 48 | 删除“controlled residual”暗示；改为 unmeasured forcing residual |
| `paper/sections/03_theory.tex` | 88 | 修正 \(\mathcal C_{\mathrm{app}}[1]\neq0\)；删除 waterbed-to-PPL theorem 推断 |
| `paper/sections/03_theory.tex` | 93-109 | 撤回 ordinary KL \(O(\tau^2)\) 解释；以 \(\|Jc_\omega\|^2=q/L\) 重写 conditional transport-proxy Proposition |
| `paper/sections/03_theory.tex` | 111-117 | 把 structural exponent 限定在 proxy assumptions 下；把 deployed rule 改为 proxy-motivated empirical basin selector |
| `paper/appendix/a1_proofs.tex` | 100-117 | 重做 forcing 与 stationary-phase conventions；删除未证明的 Bessel negativity |
| `paper/appendix/a1_proofs.tex` | 178-217 | 保留 waterbed inequality；删除 task-level PPL implication；修正 centered functional |
| `paper/appendix/a1_proofs.tex` | 262-335 | 把 collision/prefactor alignment 降为 a posteriori diagnostic |
| `paper/appendix/a1_proofs.tex` | 368-404 | 明确 0.465 是 finite-range numerical exponent；Pearson 是 modeling choice |
| `paper/appendix/a1_proofs.tex` | 453-504 | 区分 channel pattern \(c_\omega\) 与 schedule derivative \(g_\theta\)；重新定义 nonuniform transport extension，删除自动替换 |
| `paper/appendix/a1_proofs.tex` | 506-590 | 采用来源一致的 forced BC；明确 amplitude dependence；把 \(\lambda\) 降为 calibration |
| `paper/appendix/a1_proofs.tex` | 592-630 | 保留 quantization bounds；删除到 PPL ordering 的理论跨越 |
| `paper/tables/table_lambda_cv.tex` | caption | 删除“curvature agreement”强表述；改为 numerical alignment under chosen score/normalization |
| `paper/tables/table_epistemic_map.tex` | row 4 | 明确 conditional proxy theorem + empirical operating convention；不再写 ordinary-KL derivation |
| `paper/sections/06_limitations.tex` | 4 | 加入 baseline-KL interpretation withdrawn、proxy-to-task bridge unmeasured、finite-\(\tau\) uncontrolled |

任何 manuscript 修订都应保持：

- 不改实验数值；
- 不把新的 conditional theorem 写成已验证机制；
- 不声称这次 rebuttal 新增了实验；
- 不把 internal audit、私有路径或模型名称写进匿名材料。

---

## 11. 是否需要补新的理论实验

### 11.1 Rebuttal 期不建议做的事

- 不要再设计一套新的“统一 trained-task \(\tau\) theorem”来抢救当前 Proposition；应先把已有 proxy 定义写准确；
- 不要只测 \(L_{\mathrm{eff}}^J\) 就宣称 scale gap 闭合；
- 不要用更多 \(f\)-divergence sweep 选择一个更接近 0.5 的 exponent；
- 不要把 collision optimum 的数值拟合当作 task-gradient measurement；
- 不要在没有 forcing amplitude 的情况下补一张 forced/cosh 曲线并声称 residual controlled。

### 11.2 真正能升级理论、但属于未来工作的测量

1. 直接测 signed task derivative

   \[
   -\mathbb E[\nabla_z\mathcal L(z_0)^\top g]
   \]

   并检查其 sign、\(L\)-dependence 与 architecture scaling。

2. 对现有 \(U_{\mathrm{tr}}\) 测试 channel additivity、cos/sin pair、activation amplitude 与 cross-channel terms，确认 \(M/L\) proxy 在真实 activations 上的误差。

3. 对 external attention target 或明确 collision score 定义 task-facing \(Q_1\)，不再混称 ordinary KL。

4. 测 MLA rotated-subspace projection energy、Fisher curvature 与 task-gradient sensitivity，从而区分 \(d_h,d_r,\sqrt{d_hd_r}\) 或 effective-rank scaling。

5. 在 finite channel grid 上计算 exact-vs-surrogate solution residual，而不是追求不可能的 continuum global norm closeness。

6. 对 forced branch 测量实际 amplitude，并报告 density、CDF、quantile 三种不同误差。

这些工作可能形成后续理论强化，但不应成为当前 rebuttal 的承诺或阻塞项。

---

## 12. 最终 send gate

在理论段进入最终 rebuttal 前，逐项检查：

- [ ] 是否明确承认 ordinary KL first variation 为零？
- [ ] 是否撤回由 ordinary KL 导出非零 \(L^{-1/2}\) optimum 的说法？
- [ ] 是否保住并准确限定 surrogate cosh theorem？
- [ ] 是否把 deployed \(\tau\) 称为 proxy-motivated empirical default / basin selector？
- [ ] 若提 conditional scaling，是否定义 \(U_{\mathrm{tr}}\)、\(q/L\)、channel assumptions 与 nonzero first variation？
- [ ] 是否区分 \(Q_1\) 的 allocation-score first variation、schedule KL curvature 与 task gradient？
- [ ] 是否说明 practical \(\tau\) 不受 small-\(\tau\) remainder 控制？
- [ ] 是否把 waterbed 限定为 allocation-divergence statement？
- [ ] 是否把 MLA \(d_{\mathrm{eff}}\) 写成 convention？
- [ ] 是否没有用 quantization bounds 推出 PPL ordering？
- [ ] 是否没有用 99-run sweep 证明数学推导？
- [ ] 是否说明所有实验数值与 exact implemented warp 均未改变？

只要上述任一项失败，理论段就不应发送。

---

## 13. 最终战略判断

理论 rebuttal 最危险的做法有两个极端：一是把这次审计降格成“reviewer 对 epistemic map 的误读”，继续把 proxy 叫 ordinary KL；二是未经独立判断就把整个 scale theory 全部撤回。正确做法是修正理论对象。

但也没有必要把论文说成“理论完全无效”。最稳健、最诚实、也最容易恢复 reviewer trust 的结论是：

> EVQ-Cosh 的第一层精确贡献，是强凸 surrogate 下的闭式 spectral-allocation optimizer、inverse CDF 与 geometric limit。第二层是一个明确但条件性的 diffuse probability-transport proxy：\(\|Jc_\omega\|^2=q(\omega L)/L\)，在通道可加与 small-\(\tau\) 假设下给出 \(\tau\propto M/\sqrt L\)。第三层才是经验部署：prefactor、finite-\(\tau\) basin 与 MLA \(d_{\mathrm{eff}}\)。ordinary baseline-to-perturbed KL 的一阶 gain 必须撤回，但 transport-proxy theorem 不应被误撤。

这一重分层会牺牲“统一 trained-attention 第一性原理推导”的宣传力度，但比简单退回纯 empirical rule 更准确，也能保住真正可靠的数学核心、条件性 transport mechanism 和 empirical evidence。
