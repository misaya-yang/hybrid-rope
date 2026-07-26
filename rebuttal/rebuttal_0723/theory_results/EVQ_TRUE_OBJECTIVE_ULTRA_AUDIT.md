# RoPE 指数分配“真实最优”独立审计

Date: 2026-07-25
Status: `independent_ultra_audit`
Scope: repository, paper, reviews, rebuttal evidence, raw and retained results,
implementation, mathematical re-derivation, and primary literature.
Execution boundary: analysis only. No new model training was started. No paper,
training code, playbook, or existing result was modified.

## 0. 结论先行

### 0.1 最终判断

**在允许任务分布变化的合法 teacher-attention 任务族中，不存在脱离任务、
训练算法、训练预算、部署长度分布、评价指标和 head/layer 共享约束的通用
最优 RoPE 指数分配。** 这不仅是经验上的“尚未找到”，而且可由两个互斥
任务的唯一最优和有限训练算法的排序反转严格证明。它不证明某个固定自然
语言分布和固定训练协议下不存在 instance-specific optimum。

**EVQ-Cosh 不是当前证据下的训练后“真实最优”。** Cosh 是指定指数坐标、
Lebesgue 参考测度和指定二次 surrogate 下的唯一连续密度解；这个定理本身
成立，但不能推出：

1. 有限 \(K\) 离散 grid 的最优；
2. 完整 sin/cos RoPE operator 的最优；
3. 训练完成后的语言模型风险最优；
4. 部署规则 \(\tau=d_{\rm eff}/\sqrt L\)；
5. 跨长度、base、architecture、seed 或 inference scaling 的最优。

**存在比当前 Cosh surrogate 更接近训练后目标的方法类，但没有一个方法在
matched、受控比较中被证明相对 EVQ-Cosh 更稳定或更通用。** 最直接的现有
替代是：

- 从训练损失直接学习频率的 LeRoPE；
- 允许 head-specific 频率与 scale 的 AdaRoPE；
- 以有限训练轨迹为内层、独立 held-out workload 为外层的受控 bilevel
  frequency-shape 优化。

前两者证明了默认几何表不是某些训练协议下的经验最优，却没有固定频率端点
和 span，因而没有识别纯 shape；第三者是本审计提出的可证伪 operational
objective，尚未在本仓库训练验证。外部已有 anti-periodic frequency
ladder 报告跨 seed retrieval variance 降低，但没有相对 EVQ-Cosh 的 matched
direct comparison，也不是 pure shape optimization。

### 0.2 证据分级后的核心新知识

| 结论 | 级别 | 最短解释 |
| --- | --- | --- |
| task-independent 通用最优不存在 | **在合法 teacher-attention 任务族中的严格证明** | 两个任务可分别让任意两个可行的自由指数值成为互斥的唯一最优 |
| raw EVQ 不是纯 shape intervention | **仓库事实** | midpoint Cosh 同时移动 sampled extrema、log-span 和 interior spacing |
| Cosh 不支配固定-span alternatives | **含逐 seed 值的机器可读 aggregate；上游 raw/checkpoint 缺失** | Cosh 只在 128/512 最优；Exp 在 256/1K，attention-derived two-band 在 2K/4K/8K 最优 |
| 最优随 workload 反转 | **机器可读 aggregate + 代数** | Cosh 与 two-band 的 512/8K plug-in 排序在长度权重约 \(0.8008\) 处翻转 |
| 闭式 \(\tau\) 未被验证为稳定 near-optimum selector | **sanitized run-manifest 重分析** | 对 held-out neighbor 只赢 3/9 configuration means、8/18 paired runs |
| 静态 Gram selector 不是训练后目标 | **已执行的历史否证** | selector 返回约 13.2，PPL regret 5.44%，劣于旧公式 4.69% |
| 论文 collision-prefactor 表的 claimed minimizer 不最小化明示目标 | **本次独立数值反证** | 代表点 \(\tau=3.31\) 的 score derivative 为 \(-17.63\)，可行 \(\tau\approx13.05\) 的 score 更低 |
| surrogate-validation 表不直接验证 deployed grid | **本次数值指纹** | 六对 rounded Geo/EVQ 值全部匹配 inclusive \(k/(K-1)\)，与 documented native/midpoint grids 不一致 |
| LeRoPE 最接近直接训练期频率优化 | **一手文献事实** | 每 band 学一个 frequency scalar；在 217M C4 单次 ablation 中，frozen learned frequencies 保留 63.6% 完整收益 |
| 全局共享表可能是错误模型类 | **支持性外部证据** | AdaRoPE 的 per-head schedule 优于 shared-frequency ablation，但还混入 per-head temperature |
| 有意义的最优只能是实例条件的 Pareto/稳健最优 | **推论与提案** | 必须预先声明训练算法、workload、metric、seed risk、range 和 inference contract |

### 0.3 最诚实的一句话

> 当前证据支持“训练期频率分配是一个真实、可影响训练后行为的设计变量”，
> 但不支持“Cosh 是这个变量的真实最优”；可求解的对象不是 universal
> schedule，而是一个明确训练算法与部署分布下的有限训练、固定 range、
> 多目标 operational optimum。

---

## 1. 审计问题、边界与方法

### 1.1 对应的正式 concern

本审计直接回答：

- `R27bE.1`：surrogate、finite-\(\tau\)、operating rule 是否形成同一证明链；
- `R27bE.2`：base、head dimension、architecture 和 scale 的外推边界；
- `R27bE.3`：DAPE 与 learned frequency 的方法身份和优化预算混杂；
- `R27bE.4`：Cosh、非 Cosh shape 与独立 \(\tau\) 的识别；
- `R27bE.5`：held-out configuration 和更强证据的缺口；
- `AC.1`--`AC.4`：novelty、scale、surrogate chain 和 controlled evidence。

正式来源：
`rebuttal/rebuttal_0723/00_REVIEWER_SCORES_AND_AC_METAREVIEW.md:34-42,62-77,103-131`。
其中 AC 文本是 author-supplied、当前未独立验证的来源；不能与 payload-hashed
reviewer source 混为一谈。

### 1.2 每个未来实验必须先回答的五行

1. **Reviewer/AC concern:** Cosh shape 是否在固定真实频率范围后仍优于替代
   shape，并能转移到 held-out training trajectories？
2. **Existing evidence:** allocation 有效，但 Cosh、Exp、two-band 的排序随
   长度交叉；raw Cosh 还混入 range。
3. **Smallest missing evidence:** endpoint/span-matched、matched-seed、完整
   重训练的 Cosh versus target-driven shape。
4. **Smallest executable plan:** 先做无训练 identity/proxy gate，再做一个
   小模型 selection/held-out nested protocol。
5. **Stop condition:** 若收益在固定端点、best-range envelope 或 held-out
   seeds 下消失，停止 shape-optimum claim。

### 1.3 审计材料与相互反证

本次并行研究拆成以下独立产品，再进行交叉质疑：

- canonical implementation 与 schedule identity；
- continuum/discrete theorem 重推；
- ordinary-KL、pairwise-KL 与 full RoPE phase 的对象审计；
- raw result、training curve、manifest 与 provenance 分级；
- universal-optimum counterexamples；
- finite-training bilevel objective；
- RoPE/long-context 一手文献；
- learned-frequency 一手文献；
- Fourier/kernel/optimal-design 一手文献；
- 最小可证伪验证设计；
- 独立 source/claim verifier。

综合前，理论代理被要求反驳 bilevel 是“唯一真实目标”，实验代理被要求反驳
attention-derived schedule 是新 universal optimum，文献代理被要求反驳
LeRoPE 已识别 pure shape。三条强说法均未通过反证。

### 1.4 Provenance 层级

本报告严格区分：

1. **raw/per-seed machine-readable**；
2. **sanitized run-manifest**：可复算指标，但不是 raw/checkpoint bundle；
3. **machine-readable aggregate**：可复算表格，但上游 raw/checkpoint 缺失；
4. **report/hash-backed**：当前只能验证文档与历史 hash；
5. **author-confirmed only**：不能称 raw-backed；
6. **本次 CPU/代数复现**：只验证所声明数学对象，不替代模型训练证据。

当前 checkout 中没有可用的 `checkpoint*.pt` 或 `*.safetensors`；Phase16 文档
所述历史 checkpoint tree 当前也不在工作树中。冻结 checkpoint 分析因此是
未来 gate，不是本次已完成证据。

---

## 2. 先重建真正的设计变量

### 2.1 \(B,u\) 不是可识别坐标

标准写法

\[
\omega_k=B^{-u_k}
\]

有 gauge：

\[
B'=B^c,\qquad u'_k=u_k/c
\quad\Longrightarrow\quad
{B'}^{-u'_k}=B^{-u_k}.
\]

因此物理决策变量应先写为

\[
x_k=-\log\omega_k=(\log B)u_k=a+Rz_k,
\]

其中

\[
a=x_{\min},\qquad
R=x_{\max}-x_{\min},\qquad
0=z_1\le\cdots\le z_K=1.
\]

\(a,R\) 是真实频率 offset/range；\(z\) 才是 pure normalized shape。
“固定 nominal base \(B\)”并不自动固定 sampled endpoints 或 span。

### 2.2 当前仓库实际有三种 geometric grid

1. Native/Std Geo：

\[
u_k=k/K,
\]

见 `scripts/lib/rope/schedules.py:86-92` 和
`rebuttal/rebuttal_0723/experiments/geo_rope_contract.py:45-57`。

2. Paper midpoint Geo：

\[
u_k=(k+1/2)/K,
\]

见 `rebuttal/rebuttal_0723/experiments/geo_rope_contract.py:60-74`。

3. Surrogate-validation inclusive grid：

\[
u_k=k/(K-1).
\]

这个身份不是由注释得出，而是本次由表值数值指纹重建；见 §3.7。

Paper midpoint 与 Std Geo 的所有频率相差同一倍数

\[
\omega_k^{\rm mid}=B^{-1/d_{\rm rot}}\omega_k^{\rm std}.
\]

在 \(B=500000,d_{\rm rot}=64\) 时该倍数为 `0.8146172339`，等价于
position scale `1.227570...`。两者不是同一个 native table。

### 2.3 deployed EVQ 的精确公式

仓库实现和论文 inverse CDF 一致：

\[
\rho_\tau(\phi)=
\frac{\tau\cosh(\tau(1-\phi))}{\sinh\tau},
\]

\[
F_\tau(\phi)=
1-\frac{\sinh(\tau(1-\phi))}{\sinh\tau},
\]

\[
Q_\tau(u)=
1-\frac{\operatorname{asinh}((1-u)\sinh\tau)}{\tau},
\qquad
u_k=\frac{k+1/2}{K}.
\]

实现：`scripts/lib/rope/schedules.py:94-140`。
论文：`paper/sections/03_theory.tex:39-64`。

注意：\(\sinh(\tau u)/\sinh\tau\) 是反射方向的 CDF 形式，不是仓库实际
指数 warp。

### 2.4 raw EVQ 同时改变 offset、span 和 shape

对 \(K=32,B=500000,\tau=4\) 的 float64 复算：

| Schedule | first exponent | last exponent | sampled exponent span | sampled log-frequency span |
| --- | ---: | ---: | ---: | ---: |
| Std Geo \(k/K\) | 0 | 0.968750 | 0.968750 | 12.712290 |
| midpoint Geo | 0.015625 | 0.984375 | 0.968750 | 12.712290 |
| raw EVQ-Cosh | 0.003934 | 0.896390 | 0.892456 | 11.711131 |

raw EVQ 的最后一个频率是 midpoint Geo 的 `3.17265×`。所以 raw
EVQ-versus-Geo 不能被解释为“固定 base/range 只变 interior shape”。

### 2.5 两个 “endpoint-normalized Cosh” 也不是同一个 schedule

仓库至少有两种固定端点实现：

- midpoint warp 后 affine normalize：
  `rebuttal/rebuttal_0723/experiments/fmrope_125m_l256_500m/protocol.py:158-172`；
- endpoint grid warp 后按最后 exponent rescale：
  `rebuttal/rebuttal_0723/experiments/reviewer27be_shape_base/derive_real_rope_shapes.py:231-240`。

在 \(K=32,\tau=4\) 时，两者最大 exponent 差 `0.02972`、RMS
`0.01726`。用第二种重新拟合 \(\tau\) 仍有非零 residual。因此未来报告必须
明确写出公式，不能把它们都叫 “normalized Cosh”。

### 2.6 方法身份纠正

- 仓库旧 sweep 中的 “DAPE” 是 layer-shared、无单调约束的 learnable
  inverse-frequency vector；一个历史向量有 7 次顺序反转。它不是正式
  DAPE。
- 正式 DAPE 学的是 context-dependent additive attention bias，而非
  \(u_k\)。一手来源：[DAPE, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/2f050fa9f0d898e3f265d515f50ae8f9-Abstract-Conference.html)。
- 仓库 `LearnableEVQRoPE` 只学习一个 scalar \(\tau\)，不是自由
  \(K\)-dimensional exponent allocation；也未发现它在正式训练路径中的
  调用。
- 仓库 legacy `"yarn"` helper 不是官方 YaRN；正式比较必须使用
  `scripts/lib/rope/official_yarn.py` 并记录 attention `mscale`。

---

## 3. Cosh 理论究竟证明了什么

### 3.1 成立的定理很窄，但确实成立

论文定义

\[
\mathcal C_{\rm app}[\rho]
=
\frac{\alpha}{2}\int_0^1\rho(\phi)^2\,d\phi+
\frac{\beta}{2}\iint\rho(\phi)\rho(\psi)\min(\phi,\psi)\,d\phi d\psi .
\]

在闭可行集

\[
\mathcal A=
\{\rho\in L^2([0,1]):\rho\ge0\ {\rm a.e.},\ \int_0^1\rho=1\}
\]

上，若 \(\alpha>0,\beta\ge0\)，coercivity、弱下半连续性和严格凸性保证
唯一 minimizer。若 \(\beta>0\)，它是

\[
\rho_\tau(\phi)=
\frac{\tau\cosh(\tau(1-\phi))}{\sinh\tau},
\qquad \tau=\sqrt{\beta/\alpha}.
\]

且事后严格为正。若 \(\beta=0\)，唯一解是 \(\rho\equiv1\)，即上式的
\(\tau\to0\) 极限。这个结论可从 Euler equation 和 Green kernel 直接重推，
未发现代数错误。
论文也明确承认没有求解 full trained-transformer objective：
`paper/sections/03_theory.tex:15,23-45`。

### 3.2 唯一性不具坐标不变性

若换非线性坐标 \(y=h(\phi)\)，push-forward density 为

\[
\rho_y(y)=\rho_\phi(\phi)/h'(\phi).
\]

原对角项变为

\[
\int \rho_y(y)^2h'(h^{-1}y)\,dy,
\]

而 \(\min(\phi,\psi)\) 变成

\[
\min(h^{-1}y,h^{-1}z).
\]

除非 \(h\) 仿射，常系数 `delta + min` 结构和 Cosh 解都不保持。于是
“Cosh unique”必须完整读成：

> 在 \(\phi=-\log_B\omega\) 坐标、Lebesgue 参考测度和指定
> \(\mathcal C_{\rm app}\) 下 unique。

这不是坐标自由的物理频谱最优。

### 3.3 continuum theorem 不推出 \(K=16/32\) 的离散最优

附录自身给出的 inverse-CDF discretization bound：

\[
W_1\le\frac{\sinh\tau}{4K\tau},
\qquad
\|\rho-\rho_K\|_1\le\frac{\tau\sinh\tau}{K}.
\]

在 \(\tau=4,K=32\) 时，第二个上界约 `3.41`，比概率密度有意义的
\(L_1\) 最大距离 2 还大；\(K=16\) 时约 `6.82`。这说明 deployed
finite-\(K\)、large-\(\tau\) 恰处于该离散界失效的区域。midpoint
inverse-CDF 是一种构造，不是有限节点最优化定理。

### 3.4 “exact kernel” 不是完整 RoPE feature

论文的 exact collision kernel 是

\[
K_{\cos}(\omega_1,\omega_2)
=\int D(\Delta)
\cos(\omega_1\Delta)\cos(\omega_2\Delta)\,d\Delta,
\]

见 `paper/sections/03_theory.tex:23-32` 和
`paper/appendix/a1_proofs.tex:306-313`。

真实一个 RoPE pair 对 logit 的贡献却是

\[
f(\Delta)=C\cos(\omega\Delta)+D\sin(\omega\Delta),
\]

见 `paper/appendix/a1_proofs.tex:464-475`。其 small-phase variance 为

\[
\operatorname{Var}f
=
\frac{D^2x^2}{12}
-\frac{CDx^3}{12}
+\left(\frac{C^2}{45}-\frac{D^2}{40}\right)x^4
+O(x^5).
\]

论文的 cosine-only carrier

\[
q_{\cos}(x)=x^4/45+O(x^6)
\]

是 \(D=0\) 的特殊内容相位。只要 sine carrier 非零，leading order
可以是 \(O(x^2)\)。只有把 phase basis 本身按等权内积，或额外假设
content coefficient covariance 与二维单位阵成比例且 cross terms
消失时，完整 sin/cos feature Gram 的核心才是

\[
\cos((\omega_1-\omega_2)\Delta),
\]

而不是含 sum-frequency 项的 cosine-only product。一般训练后的 \(C,D\)
不满足这些条件。因此 “dead-frequency order”、\(Q_1\) 符号和 stiffness
balance 都依赖训练出的 content phase；静态 cosine geometry 不能自动代表
完整 RoPE。

### 3.5 finite \(\tau\) 不受 small-\(\tau\) 展开保障

Pearson exact expression 的首项 \(S\sim\tau^4/45\) 只在小 \(\tau\)
有效。exact/leading 比较：

| \(\tau\) | exact | \(\tau^4/45\) | leading / exact |
| ---: | ---: | ---: | ---: |
| \(\sqrt2\) | 0.05830 | 0.08889 | 1.52× |
| 2 | 0.18033 | 0.35556 | 1.97× |
| \(\sqrt8\) | 0.53078 | 1.42222 | 2.68× |
| 4 | 1.61671 | 5.68889 | 3.52× |

其 complex Taylor 最近 singularity 位于 \(\tau=i\pi/2\)，因此
\(\tau=2,2.83,4\) 已超出该级数的收敛半径。更直观地，\(\tau=4\) 时
exact \(\rho(1)=0.1466\)，而一阶
\(\rho\approx1+\tau^2\eta\) 给出 \(-1.667\)，甚至不再是可行密度。

对 static cosine utility，在 \(B=500000,\tau=4,L=256\)：

\[
\Delta U_{\rm exact}=0.20098,\qquad
\tau^2Q_1=0.50315.
\]

在 \(L=128\ldots8192\) 上，truncation 高估约 2.41×--2.90×。
所以 exact Cosh warp 在 \(\tau=4\) 仍合法，但 small-\(\tau\) balance
不能给它 theorem-level 保证。

### 3.6 \(\tau=d/\sqrt L\) 不是 Cosh variational theorem 的预测

对给定 fitted \(\alpha,\beta\)，surrogate minimizer 的参数是

\[
\tau_{\rm surr}=\sqrt{\beta/\alpha}.
\]

新鲜复跑 `python3 scripts/analysis/tau_scaling_analysis.py`：

- \(d=64,B=500000,L=128\ldots4096\) 时，
  \(\tau_{\rm surr}=7.688,7.403,7.038,6.696,6.244,5.704\)；
- deployed \(d/\sqrt L=5.657,4,2.828,2,1.414,1\)；
- 脚本拟合约 \(\tau_{\rm surr}\propto L^{-0.085}\)，而部署为
  \(L^{-0.5}\)；
- 固定 \(L=2048\)，surrogate 随 \(d\) 约 \(\sqrt d\)，部署随 \(d\)
  线性。

任意 \(\tau\) 都可通过重新定义 \(\beta/\alpha=\tau^2\) 成为某个
\(\mathcal C_{\rm app}\) 的 minimizer。因此 Cosh family theorem 选择
shape family，不选择 deployed operating point。

### 3.7 新发现：surrogate-validation 表值强烈指向第三种 grid

`paper/appendix/a1_proofs.tex:137-143` 的六组
Geo/EVQ collision values：

\[
(226.3,17.7), (192.6,27.6), (163.2,43.3),
(133.9,59.0), (109.0,67.5), (87.0,66.3)
\]

在一位小数精度下全部与以下设置一致：

\[
u_k=\operatorname{linspace}(0,1,K)=k/(K-1),
\quad
K=32,\ B=500000,\ \tau=64/\sqrt L.
\]

同一公式改用仓库 documented native \(k/K\) 或 deployed midpoint
\((k+1/2)/K\) 均不一致。例如 \(L=512\)：

| Grid | Geo score | EVQ score |
| --- | ---: | ---: |
| inclusive \(k/(K-1)\) | **163.2** | **43.3** |
| native \(k/K\) | 154.7 | 37.0 |
| deployed midpoint | 164.3 | 41.8 |

六对值共同构成很强的 numerical fingerprint，但在缺少原始生成代码
provenance 时，不把它升级为 inclusive grid 的唯一来源。它不否定该数值
grid 上 collision 降低，却不能作为 deployed midpoint/native grid 的直接
验证。

### 3.8 新发现：collision-prefactor 表中的 claimed minimizers 不最小化其明示 objective

论文定义 normalized off-diagonal score

\[
C(\tau)=\sum_{i<j}\frac{K_{ij}(\tau)^2}
{K_{ii}(\tau)K_{jj}(\tau)}.
\]

对 `paper/tables/table_lambda_cv.tex` 的代表配置
\(K=32,L=512,B=500000\)，按 uniform
\(\Delta\in[0,L]\) 和上式 float64 重算。midpoint grid 上：

| \(\tau\) | \(C(\tau)\) |
| ---: | ---: |
| 0 | 164.28698 |
| deployed \(2.82843\) | 41.76306 |
| table-claimed “collision optimum” 3.31 | 28.17075 |
| 一个更低的可行 scan/refine 点 13.05190 | 0.04207 |

三种仓库相关 grid 都反驳 claimed minimizer：

| grid | \(C(3.31)\) | central \(C'(3.31)\) | 更低的可行 scan/refine 点 |
| --- | ---: | ---: | ---: |
| midpoint | 28.17075 | -17.62780 | \(\tau=13.05190,\ C=0.04207\) |
| native \(k/K\) | 25.08349 | -23.81255 | \(\tau=13.43440,\ C=0.03315\) |
| inclusive \(k/(K-1)\) | 29.78246 | -22.55833 | \(\tau=14.01858,\ C=0.06154\) |

负导数说明 \(\tau=3.31\) 不是任何一个 grid 上的 interior stationary
point。把 midpoint \(\tau\) 改为
\(0.8\times/1.2\times\) 后，score 分别变化约 `+70.1%/-42.4%`，
也与“附近极平”不符。改用以下合理近邻定义仍在约 13--13.6 找到更低点：

- normalized \(i<j\) 或 \(i\ne j\)；
- raw squared 或 absolute off-diagonal；
- trace normalization；
- full sin/cos continuous phase；
- discrete lags \(0,\ldots,L-1\) 或 \(1,\ldots,L\)；
- endpoint 或 midpoint grid。

本次脚本：
`/tmp/evq_true_objective_collision_audit_20260725.py`，SHA-256
`b9bc349f41ec659057f2ee9194f62df74b6537177ab029fa6cdb95b87abaa330`。
复现协议是 \([0,30]\) 上 30,001 点 scan 后 bounded scalar refinement；
central difference \(h=10^{-5}\)；
`np.sinc(x)=sin(pi*x)/(pi*x)`。这些只是明确更低的可行点，不是
certified global minima。结论只需要负导数：表中 \(\tau\) 不最小化所写
objective；不能把约 13 的扫描点升级成语言模型最优。

provenance 审计进一步发现：

- commit `0d399c4` 首次加入同一组数值时列名为
  `tau*_emp`；
- commit `74ea465` 在数值不变的情况下把它们改称
  `tau*_coll`；
- `scripts/analysis/verify_c_coll.py:39-64` 把表值作为输入再计算 ratio，
  并不优化 collision objective。

因此当前 provenance 不能建立 `c_coll=1.171` 的 collision-calibration
身份。这里不推断改名动机，只报告可复现的对象与历史事实。

### 3.9 两类 KL 被混在了一条 scaling chain 中

必须区分：

1. **schedule KL:** 固定 activation/query，比
   \(p_0=\operatorname{softmax}z(0)\) 与
   \(p_\tau=\operatorname{softmax}z(\tau)\)。若
   \(z(\tau)=z_0+\tau^2g+O(\tau^4)\)，则

   \[
   D_{\rm KL}(p_0\|p_\tau)
   =\frac{\tau^4}{2}g^\top J_{\rm sm}(p_0)g+O(\tau^6).
   \]

   附录 `paper/appendix/a1_proofs.tex:464-501` 实际也给出了这条二阶
   curvature。

2. **pairwise positional-discrimination KL:** 固定 schedule，比两个
   distance/phase 诱导的 attention distributions。它可在 Geo 处有
   \(Q_0\)，改变 allocation 后有 \(\tau^2Q_1\)。该对象不是
   \(D_{\rm KL}(p_0\|p_\tau)\)，而是某个 absolute positional-
   discrimination functional；它必须独立定义 position pair、content
   distribution 和 carrier direction。

主文 `paper/sections/03_theory.tex:93-108` 称 “post-softmax KL gain”，
却给 \(Q_0+\tau^2Q_1\)；附录先定义 linear cosine phase-variance
score，又切换到 schedule derivative \(g\) 的 ordinary KL curvature。
如果指第一类，\(O(\tau^2)\) KL gain 的阶数错误；如果指第二类，
\(Q_0+\tau^2Q_1\) 可以成立，但必须另立完整随机变量和方向定义，不能用
第一类 \(g\) 的 Rayleigh quotient 直接证明。问题核心是 **direction
mismatch**，而不只是常数 convention。

### 3.10 其他理论链断点

- 主文称 \(\mathcal C_{\rm app}\) 与 waterbed \(\mathcal W\) 在
  \(\rho=1\) 都为零；实际
  \(\mathcal C_{\rm app}[1]=\alpha/2+\beta/6\)。
- 对固定 \(\alpha\) 和 \(\beta>0\)，
  \(\mathcal C_{\rm app}[\rho_\tau]-\mathcal C_{\rm app}[1]\)
  含 \(O(\tau^2)\) 项。若把 \(\beta\) 与路径参数 \(\tau^2\) 同时绑定，
  可以得到 quartic 路径变化，但那是在跨不同 objectives 比较；独立定义
  的 centered waterbed functional 也可以从 quartic 开始。
- Fisher-forced ODE 的 particular solution 不是自动可忽略；当前
  \(O(1/\log B)\) residual bound 仍依赖未测的 activation-conditioned
  coefficient。
- 主文引入 smooth \(D\in W^{m,1}(\mathbb R)\)，实际 validation 使用
  extended uniform indicator \(1_{[0,L]}/L\)，其边界 jump 不在
  \(W^{1,1}(\mathbb R)\)。uniform sinc 计算本身没错，但不是同一个
  regularity assumption。

---

## 4. 原始结果重新回答了什么

### 4.1 fixed-span native-shape 数据直接排除 Cosh dominance

含逐 seed 值的 machine-readable aggregate（上游 raw/checkpoint 当前缺失）：
`rebuttal/rebuttal_0723/theory_results/native_attention_shape_l128_results_20260724.json`。
每格为三个 training seeds 的 mean tail NLL：

| schedule | 128 | 256 | 512 | 1K | 2K | 4K | 8K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cosh | **5.4645** | 5.4473 | **5.5434** | 5.6966 | 5.8457 | 6.0664 | 6.2935 |
| Exp | 5.4698 | **5.4464** | 5.5450 | **5.6883** | 5.8423 | 6.0342 | 6.2165 |
| attention two-band | 5.4826 | 5.5099 | 5.5959 | 5.7212 | **5.8254** | **5.9504** | **6.0823** |
| exact uniform | 5.4759 | 5.4514 | 5.7355 | 5.8295 | 5.9365 | 6.0724 | 6.1727 |
| native Std Geo | 5.5046 | 5.5026 | 5.6563 | 5.8456 | 6.0530 | 6.2566 | 6.3928 |

直接事实：

- Cosh 只在 128、512 最优；
- Exp 在 256、1K 最优；
- attention-derived two-band 在 2K、4K、8K 最优；
- 8K 上 two-band-minus-Cosh 为 `-0.2111` NLL，三个 paired seeds 的
  95% t interval `[-0.357,-0.065]`；
- 但仅三个 seeds 的 exact sign test 很弱，且 28 个探索性 Cosh
  comparisons 做双侧 Holm correction 后，只有
  `Cosh < StdGeo @ 2K` 存活（raw \(p=0.00142\)，Holm
  \(p\approx0.0397\)）；没有 alternative-versus-Cosh 结论存活。

所以安全结论不是 “two-band universally better”，而是：

> Cosh 不支配；schedule 排名随 deployment length 改变。

attention-derived arm 还使用 seed-42 selection prior 和 30 个 interior
nodes 的 SLSQP，调参信息预算高于闭式 Cosh，不能当作公平的 zero-tuning
替代。

### 4.2 deployment weight 可严格反转 Cosh 与 two-band

在 512：

\[
\Delta_{\rm Cosh-Attn}=5.5434-5.5959=-0.0525;
\]

在 8K：

\[
\Delta_{\rm Cosh-Attn}=6.2935-6.0823=+0.2112.
\]

若 workload 为

\[
J_p=pR_{512}+(1-p)R_{8K},
\]

则

\[
\Delta J_p=0.2112-0.2637p.
\]

用完整 JSON 精度，排序在

\[
p\approx0.8008
\]

处翻转。这个 crossover 是 plug-in descriptive value，不带可靠 CI；
它只证明观察到的 mean plug-in scalarization 会随 \(p\) 反转，不证明
population ranking。更一般地，未声明 deployment weights 时 scalar
optimum 在定义上就不存在；这点不依赖该数值 crossover。

### 4.3 Phase16 99-run 重分析不支持 uniformly near-optimal

`data/curated/phase16_99run_manifest.csv` 是 99-row sanitized
run-manifest；当前原始 runs/checkpoints 不在 checkout。统一用
runner-defined weighted extrapolation NLL 重算：

- formula vs midpoint Geo：7/9 configuration means、18/27 paired runs
  更好，equal-configuration mean `-0.01331` NLL；
- formula vs pilot-selected neighbor，仅 held-out seeds 137/256：
  3/9 configuration means、8/18 pairs 更好，但 equal-configuration
  mean `+0.02210` NLL，即 formula 平均更差；
- 在 `{Geo, Formula, Neighbor}` 中恰好 3 次第一、3 次第二、3 次第三。

证据：
`rebuttal/rebuttal_0723/theory_results/PHASE16_99RUN_RAW_REANALYSIS_20260724.md:15-39,97-125,150-185`。
manifest provenance 见
`data/curated/phase16_99run_manifest.meta.json:3-20`：这是 sanitized
run-manifest，source bundle 当前 checkout 不可用。

此外 Phase16 的 raw \(\tau\) sweep 同时改变 endpoints、span 和 shape，
不能验证纯 Cosh shape。它不支持“公式稳定优于 held-out neighbor”或
“已经验证 uniformly near-optimal”；只能支持“formula 是一个有时有用、
但会失败的 operating prior”。

### 4.4 静态 proxy 与训练后 channel usage 不一致

`EXPERIMENT_REPORT_20260724.md:576-701` 的 retained evidence：

- Q/K norm 与 causal delta 的 median Spearman：`-0.167`；
- phase utility 与 causal delta：`-0.045`；
- 从 256 到 8K，Geo 有 21/32、EVQ 有 23/32 pairs 的 deletion effect
  反号；
- 只交换 frequency-pair assignment、保持 multiset 不变，24 个
  checkpoint-length cells 中 22 个 NLL 改变至少 0.05。

该 retained report 直接支持训练后存在 channel-frequency co-adaptation；
但当前上游 checkpoint/raw trace 已不在 checkout，因此不把它升级为当前
可从 raw 重新构造的结论。频率 multiset、索引 assignment 和训练路径不是
可随意互换的三个变量。

### 4.5 training-free selector 已被实际否证

`TRAINING_FREE_TAU_SELECTOR_20260724.md:145-190`：

- static full sin/cos finite-\(K\) Gram selector 在九个配置上返回
  \(\tau^\star=13.131\)--13.344；
- mean relative PPL regret 5.44%，旧 formula 为 4.69%；
- top-2 basin entry 1/9，旧 formula 为 5/9；
- gate 明确失败，因此没有启动新 GPU run。

它和本次 collision-prefactor 复算共同说明：**一个 deterministic
geometry score 可以被优化得很好，同时更差地预测 trained-model risk。**

### 4.6 训练损失也不是 OOD objective

retained learnable-\(\tau\) 结果：

- learned \(\tau=1.1406\)（三 seed mean）；
- 8K PPL retained display `437.9`，逐 seed 重算 mean `437.97`；
- fixed \(\tau=5\) PPL `335.7`（三 seed mean）。

train-range loss 近似持平时，learned in-training scalar 明显选错 OOD
目标。这不是“学习频率无效”的证明；它说明 **joint training objective、
parameterization 和 deployment objective 必须一致**。

机器可读来源：
`data/curated/learnable_tau_128tok_evidence.json:1-30` 和
`data/curated/primary2_l128_fixed_tau5_3seed.json:1-44`。

### 4.7 range 与 shape 不是经验正交项

历史 exact-range seed-42 report-backed 结果：

- fixed range 下 Cosh-minus-uniform at 512/1K/2K：
  `-0.4775/-0.2050/-0.1128`；
- 两者都 retarget 到 evaluation length 后：
  `+0.0611/+0.1818/+0.2786`。

见
`MATCHED_RANGE_COSH_500M_S42_20260724.md:47-84,102-119`。
方向翻转说明 shape 与 target-aware range transport 不是简单可加项。

未跟踪的三 seed aggregate
`rebuttal/rebuttal_0723/theory_results/matched_range_cosh_500m_3seed_result_20260724.json:1-74`
给 fixed-range means
`-0.3159/-0.1949/-0.1674` 且 3/3 seeds 同向，但明确写有：

- `local_raw_aggregate_present=false`；
- `local_per_seed_values_present=false`；
- `confidence_intervals_present=false`。

它只能称 `AUTHOR_CONFIRMED_AGGREGATE_PENDING_LOCAL_RAW_PROMOTION`，不能
作为 raw-backed reviewer evidence。

### 4.8 其他负面边界

- historical MLA raw
  `data/curated/eval_3seeds_full_results.json` 中 EVQ-minus-Geo at
  8K/16K/20K/24K/28K/32K 为
  `+0.0093/-0.3731/-0.3006/-0.1660/-0.1098/-0.1059` NLL：16K 达峰，
  20K 仍大，随后衰减；16K--24K intervals 排除 0，28K/32K 已包含
  0。对 legacy Geo+s4 在约 28K crossover，不是“越长越强”。
- \(K=8\) MLA scarcity 单 seed 8K 上 EVQ 比 native 差 `+0.6522` NLL；
  不能说 scarce channels 单调强化 Cosh。机器工件：
  `rebuttal/rebuttal_0723/theory_results/mla_scarcity_seed42_result_20260724.json:53-175`。
- QuALITY 16K 上 gold-answer NLL 改善没有转成 accuracy：
  EVQ accuracy 约低 `0.39pp`。NLL proxy 与 downstream capability
  不同。raw-JSON-backed aggregate：
  `data/curated/quality_454m_full_eval.json:1-95`。
- shape×legacy-YaRN interaction 随长度变号；不能仅凭两个主效应声称
  orthogonality。逐 seed machine-readable 来源：
  `data/curated/eval_3seeds_full_results.json`。按
  \([\log PPL({\rm EVQ+s4})-\log PPL({\rm EVQ})]
  -[\log PPL({\rm Geo+s4})-\log PPL({\rm Geo})]\) 重算，8K 为
  `+0.00043`，16K 以后为负；这里的 scaler 不是 official YaRN。

---

## 5. 为什么通用最优在数学上不存在

### 5.1 任务依赖的严格唯一最优反例

固定 \(B>1\)、单个 RoPE pair、两个 key 的相对距离 \(0,1\)，固定内容向量。
指数 \(u\) 产生 logits

\[
z_u=(1,\cos(B^{-u})),
\qquad p_u=\operatorname{softmax}(z_u).
\]

对任意 \(v\)，令 teacher target 为 \(p_v\)。交叉熵

\[
H(p_v,p_u)=H(p_v)+D_{\rm KL}(p_v\|p_u).
\]

令 \(u,v\in[0,1]\)、\(B>1\)，则

\[
\omega=B^{-u}\in[B^{-1},1]\subset(0,\pi),
\]

且 \(u\mapsto\cos(B^{-u})\) 严格单调，所以 \(p_u=p_v\) 当且仅当
\(u=v\)。故唯一最优为 \(u=v\)。取两个不同的可行自由值
\(v_1,v_2\)，得到互斥的两个唯一最优。多通道时把其他 pairs 的 Q/K
内容置零即可嵌入。更强地，在 \(K\ge3\) 时可把两个 frequency endpoints
固定且设为内容不活跃，只让同一个 interior pair 活跃；两个 schedules
只改变该 interior exponent。于是反例仍完全位于 “fixed endpoints/span、
pure interior shape” 的受控空间内。

因此即使 operator、base、\(K\) 全部固定，也不存在 task-independent
schedule。

### 5.2 有限训练算法可反转排序

同一数据和模型

\[
f_{a,\omega}=a\cos\omega,\quad
L=\frac12(a\cos\omega-1)^2,\quad a_0=0.
\]

一步 gradient descent 后

\[
R_\eta(\omega)=\frac12(\eta\cos^2\omega-1)^2.
\]

取 \(\omega_A=0.2,\omega_B=0.8\)：

- \(\eta=1/\cos^2(0.2)\) 时，A 风险 0，B 约 0.1223；
- \(\eta=1/\cos^2(0.8)\) 时，B 风险 0，A 约 0.4791；
- 训练至收敛时两者都可达到 0。

所以 optimizer/LR/budget 可以反转 schedule 排名，也可以让 schedule
完全不可识别。

### 5.3 其他不可识别因素

1. **base–exponent gauge:** 已见 §2.1。
2. **pair permutation:** 只有同步置换 frequency、Q/K blocks、
   initialization 和 optimizer state 才函数等价；固定 seed 下只换
   frequency assignment 不等价。
3. **distance-lattice alias:** 若所有相对距离在 \(g\mathbb Z\)，则
   \(\omega\) 与 \(\omega+2\pi n/g\) 等价。
4. **finite-window resolution:** 频率差小于约 \(1/L\) 时难以稳定识别。
5. **feature-space equivalence:** 在 finite lag set \(\mathcal D\) 上，

   \[
   \Phi_\Omega(\Delta)=
   [\cos(\Delta\omega_1),\sin(\Delta\omega_1),\ldots]
   \]

   只通过中心化 column space 起作用。若该显式 phase-feature matrix
   满足 \(2K\ge|\mathcal D|\) 且 full row rank，则线性 readout 可在有限
   \(\mathcal D\) 上插值任意 target；差异来自 conditioning、
   regularization、optimization、noise 或 OOD。这个结论只属于该线性
   feature model，不是完整 transformer 的函数等价定理。

这解释了为什么“表示能力最好”与“有限训练后最好”不是同一个问题。

---

## 6. 一手文献：实际优化变量，而不是标题相似度

### 6.1 RoPE 与 inference range 方法

| Work | 实际变量/目标 | 对 fixed-range shape 问题的含义 |
| --- | --- | --- |
| [RoFormer](https://arxiv.org/abs/2104.09864) | 预定义 geometric \(\theta_i\)，分析 relative rotation | 没有在 \(u_i\) 上定义风险或优化 |
| [Position Interpolation](https://arxiv.org/abs/2306.15595) | 统一缩放 position/frequency range 后微调 | shape 不变 |
| [YaRN, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/874a4d89f2d04b4bcf9a2c19545cf040-Paper-Conference.pdf) | wavelength-dependent interpolation ramp + attention temperature | 逐维改变频率，但目标是 inference context transport，不是 fixed-range pretraining shape |
| [CLEX, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/3df38ca67befaed9c03b95ffee07d9f8-Abstract-Conference.html) | neural ODE 学 length-conditioned scaling dynamics | 改 target-length transport，且加入 learned dynamics |
| [LongRoPE, ICML 2024](https://proceedings.mlr.press/v235/ding24i.html) | evolutionary search per-dimension \(\lambda_i\) 与 threshold，以目标长度 PPL 选择 | 直接搜有效 \(u_i\)，但 range、shape、stage 与少量 validation samples 耦合 |
| [LongRoPE2, ICML 2025](https://proceedings.mlr.press/v267/shang25a.html) | needle answer-token PPL 搜 per-band scaling，再 mixed training | ablation 实证显示普通 PG19-PPL 与 needle-PPL 搜索指标会改变所选解/下游结果；不是形式证明或通用目标 |
| [Resonance RoPE, ACL Findings 2024](https://aclanthology.org/2024.findings-acl.32/) | 把各 wavelength round 到整数周期 | 是逐维 analytic deformation；objective 是 periodic alignment，不是 trained LM risk |
| [FMRoPE, ICLR 2026](https://openreview.net/forum?id=PR1PPxvG9Q) | 研究 base 与 training length 如何决定 learned high-norm band；按长度改 base | 主要是 scalar range/base，反而显示 interpolation/extrapolation trade-off |

结论：不能说过去没有改变或搜索逐维频率；能说的窄空白是“固定 operator、
物理 endpoints/span、共享表和 \(K\)，只优化 interior shape，并对每个
候选完整训练后用 held-out risk 选择”。

### 6.2 operator/model-class alternatives

| Work | 变量 | 关键边界 |
| --- | --- | --- |
| [DAPE, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/2f050fa9f0d898e3f265d515f50ae8f9-Abstract-Conference.html) | context-dependent additive bias MLP | 不是 inverse-frequency learning |
| [FoPE, ICML 2025](https://proceedings.mlr.press/v267/hua25b.html) | 每个 block 变 Fourier series，under-trained slow components 置零 | 改 operator/basis，不是标准 RoPE 内的 \(u_k\) |
| [Round and Round, ICLR 2025](https://openreview.net/forum?id=GtvuNrk58a) | 诊断 slow bands，把部分 bands 变 NoPE | 表明候选空间可能需要 \(\omega=0\) atoms |
| [HoPE, ACL 2025](https://aclanthology.org/2025.acl-long.1123/) | 低频块替换为 NoPE | 离开严格正频率、固定 \(K\) 的 shape space |
| [Ms-PoE, NeurIPS 2024](https://arxiv.org/abs/2403.04797) | head-specific inference scaling | 全局 shared schedule 不是唯一合理模型类 |
| [Anti-Periodic PE, 2026-07-23 v1](https://arxiv.org/abs/2607.21405) | 25% heads 使用固定 half-integer harmonic ladder，以 in-window NIAH 跨 seed variance 为核心 endpoint；160M 还有同 \(K\)、同 band/endpoints 的 incommensurate control | 160M \(n=6\)、410M \(n=4\) 提供直接 stability evidence；但未与 EVQ-Cosh 比、25% special-head model class 不同、未以 held-out risk 选择 shape、同-band control 只在 160M 复验，且多重校正后显著性减弱；单作者未审稿 preprint，只声称 training-window retrieval |

这些方法提示 omitted variable 不只是一条更好的 monotone curve；零频 atoms、
head specialization、attention temperature 和 operator basis 都可能比
global shape 更重要。

### 6.3 最接近训练后“真实目标”的外部方法

#### LeRoPE

[LeRoPE](https://arxiv.org/abs/2607.10134) 从 RoPE 初始化，学习

\[
\widehat\omega_k=e^{\alpha_k}\omega_k,
\]

所有 layer/head 共享一组 \(\alpha_k\)，直接通过 LM training loss 更新。
它是本次文献中最直接的 training-time frequency allocation。

一手事实：

- 52M--2.5B C4 scaling ladder 上优于 RoPE/partial-RoPE；
- 217M 三 seed 复验支持 method gap；
- 在 217M C4 的单次 fixed-frequency ablation 中，把独立 run 学到的
  frequencies 冻结再训练，保留完整 gain 的 63.6%；这不是跨 scale 或
  multi-seed 的固定比例结论；
- 未缩放 LeRoPE 在 OOD length 上比 RoPE 更快恶化；NTK-by-parts+YaRN
  可以恢复并优于对应 baselines，但不是已证明的唯一 remedy；
- 大规模多为 single seed，且当前是 2026-07-11 v1 preprint。

它证明：

> 默认 geometric table 不是该 training recipe 下的经验最优，final
> frequency set 与 joint dynamics 都有贡献。

它没有证明：

- endpoints、span、base-equivalent range 固定后的 shape 因果效应；
- global optimum 或 stationarity；
- C4 以外、不同 architecture 或 OOD workload 的通用性。

#### AdaRoPE

[AdaRoPE](https://arxiv.org/abs/2607.19363)（论文标注 accepted at ICML
2026）学
head/group-specific、dimension-wise log frequencies，并同时学习
head-specific length-aware temperature。其 shared-frequency ablation
弱于 per-head version。

它最重要的反证是：

> 如果 head 的功能尺度不同，单个 global \(u_k\) 可能本身就是一个过强、
> 仅为硬件便利而保留的约束类。

但 AdaFreq 与 AdaScale 同时变化，理论只覆盖简化 retrieval/global-head
setting；因此它也没有证明 head-wise frequency 是普适真理。

#### How Data Shapes RoPE

[How Data Shapes RoPE](https://arxiv.org/abs/2607.07678)（2026-07-08 v1
preprint）定义单频率
contrast proxy，并在 \(\theta W\le\pi\) 的 hard field constraint 下得到
\(\theta^\star=\pi/W\)。这不是多频有限 \(K\)、softmax、Q/K co-adaptation
或 LM loss 的最优；其真正贡献是支持 frequency utility 随 dependency
scale/data 改变。

### 6.4 跨领域 spectral design 的正确类比

关键区分：

1. **给定目标谱后的有限节点离散化**；
2. **为训练后任务风险选择目标谱本身**。

RFF/QMC/quadrature 多数解决第一类，RoPE exponent design 属于第二类。

- [Rahimi & Recht 2007](https://proceedings.neurips.cc/paper_files/paper/2007/file/013a006f03dbc5392effeb8f18fda755-Paper.pdf)：
  对给定 shift-invariant kernel 的谱做随机近似，不选择任务最优谱。
- [Bach 2017](https://jmlr.org/papers/volume18/15-178/15-178.pdf) 与
  [Avron et al. 2017](https://proceedings.mlr.press/v70/avron17a.html)：
  leverage sampling 依赖输入分布、kernel 和 regularization。
- [Li et al. 2019](https://proceedings.mlr.press/v97/li19k.html)：
  kernel approximation error 与 expected learning risk 必须区分。
- [Lázaro-Gredilla et al. 2010](https://www.jmlr.org/papers/volume11/lazaro-gredilla10a/lazaro-gredilla10a.pdf)：
  直接学习 sparse spectral points 会出现初始化依赖与过度自信。
- [Wilson & Adams 2013](https://proceedings.mlr.press/v28/wilson13.html)：
  data likelihood 可学 spectral mixture，但 sampling alias 也会产生伪频率。
- [Tancik et al. 2020](https://proceedings.neurips.cc/paper/2020/hash/55053683268957697aa39fba6f231c68-Abstract.html)：
  在其 controlled tasks 中，匹配 bandwidth 后多种 shape 的曲线接近，
  提示 scale 可能压过 shape。
- [Moitra 2015](https://arxiv.org/abs/1408.1681)：
  finite-window Vandermonde conditioning 受 frequency separation 控制，
  schedule 收益可能来自优化稳定性而非表示优越性。

这些理论都没有推出 Cosh 或另一条 universal schedule；它们共同指出最优
sampling measure 必须条件化在 data/operator/regularization/objective 上。

### 6.5 Bilevel 也不是本体论“真实目标”

[Franceschi et al. 2018](https://proceedings.mlr.press/v80/franceschi18a.html)
给出了 validation outer objective、training inner dynamics 的规范形式。
但其 convergence 需要 compact hyperparameter set、unique inner minimizer、
uniform convergence 等深网通常不满足的条件；论文也观察到 validation
overfitting。

因此能成立的说法是：

> 明确算法的 held-out finite-training bilevel risk，比 static collision
> proxy 更接近一个声明清楚的 deployment objective。

不能升级为：

> bilevel risk 是唯一真实目标，或它找到 universal optimum。

---

## 7. 重新定义一个有意义且可求解的最优问题

### 7.1 问题实例必须完整索引

定义

\[
\kappa=
(\text{architecture/operator},K,a,R,
\mathcal A,T,P_{\rm train},
\{Q_j,\ell_j,O_j\}_{j=1}^J,
\Xi,\text{checkpoint rule}).
\]

其中：

- \(a,R\)：physical log-frequency endpoints/span；
- \(\mathcal A,T\)：optimizer、LR schedule、precision、regularization、
  token budget 和 finite checkpoint；
- \(Q_j,\ell_j\)：预注册 deployment task/length distributions 和 metrics；
- \(O_j\)：native、YaRN、FMR 或其他 inference operator contract；
- \(\Xi\)：initialization、data order、dropout 与 execution randomness。

shape space 取

\[
\mathcal Z_\delta=
\{z\in[0,1]^K:
z_1=0,\ z_K=1,\ z_{k+1}-z_k\ge\delta\}.
\]

这里假设 \(K\ge2\) 且 \(0\le\delta\le1/(K-1)\)，否则可行集为空。
\(\delta=0\) 保留 repeated nodes/zero-gap atoms 或最低频端重复原子的
closure；固定有限 endpoints/span 时它不包含 \(\omega=0\) atom。若一开始
强制严格单调和平滑，仍可能排除 HoPE/LeRoPE 提示的边界型候选；要允许
真正的 \(\omega=0\)，必须另行扩展可行集。

### 7.2 finite training，而不是抽象全局 ERM

\[
\theta_T(z;S,\xi)=
\mathcal A_\kappa^T(z,S,\xi)
\]

显式表示实际有限训练轨迹。对 deployment endpoint \(j\)：

\[
R_j(z;S,\xi)=
\mathbb E_{(q,y)\sim Q_j}
\ell_j(f_{\theta_T(z;S,\xi),O_j(z)}(q),y).
\]

用与预注册 baseline \(z_0\) 相同 data/seed stream 的 paired regret：

\[
D_j(z;S,\xi)=
\frac{R_j(z;S,\xi)-R_j(z_0;S,\xi)}{c_j}.
\]

\(c_j\) 必须在看结果前固定，不能用观察后的 sample SD 调整。

### 7.3 正确输出是 Pareto set，不一定是一点

若没有可信 workload weights，定义

\[
\operatorname{ParetoMin}_{z\in\mathcal Z_\delta/G_\kappa}
\bigl(G_1(z),\ldots,G_J(z)\bigr),
\]

其中 \(G_\kappa\) 只商掉完整实验真正保持的 permutation/alias
symmetries。

若有足够 seeds，可令

\[
G_j(z)=
\sup_{P\in\mathcal U_j}
\operatorname{CVaR}_{\alpha,P}[D_j(z;S,\xi)].
\]

当前 \(n=3\) 不足以估计 0.8/0.9 CVaR；此时应报告全部 paired seed
values 和均值，不伪造 tail estimate。anchors 是 seed 内 repeated
measure，不是独立训练单位。

若项目必须选一个 schedule，使用预注册 no-harm constrained Chebyshev：

\[
\min_z\max_{j\in{\rm OOD}}
\frac{G_j(z)-b_j}{s_j}
\quad\text{s.t.}\quad
G_j(z)\le\epsilon_j,\ j\in{\rm ID}.
\]

这个对象称为：

> **算法/分布条件化、有限训练、固定 range 的 operational optimum。**

不要称 universal true optimum。

### 7.4 存在性与可计算性

若 \(\mathcal Z_\delta/G_\kappa\) 非空紧致，且每个最终使用的
extended-real objective \(G_j\) proper、lower-semicontinuous，并存在一个
所有 \(G_j\) 都有限的共同可行点，则任意严格正权重的标量化都达到最小值，
其 minimizer 是 Pareto point。Failed run 记为 \(+\infty\) 只有在该扩展
仍 lower-semicontinuous 时才保持结论；CVaR/ambiguity supremum 也必须
另行验证这些性质。

这不保证：

- unique solution；
- global computability；
- hypergradient 不受 nonconvex path 影响；
- inference kernel/compile 离散切换时的连续性；
- solution 转移到另一个 \(\kappa\)。

### 7.5 比 Cosh 更有前景、但尚待验证的算法

不是再发明一个 schedule family，而是：

1. 用 endpoint-preserving positive gaps 参数化 \(z\)；
2. Cosh 只作为 initialization/prior；
3. 对有限训练轨迹反传 outer workload：

   \[
   \nabla_zR=
   \partial_zR+
   \partial_\theta R\,
   \frac{\partial\theta_T}{\partial z};
   \]

4. 在 full sin/cos phase metric 下做 projected trust-region step；
5. 用 independent design seeds 选择；
6. 冻结 candidate 后在 held-out training seeds、held-out documents、
   best-range envelope 和 held-out configuration 上验证。

与它同时比较一个 **exact-span LeRoPE control**：

- 学 \(K-2\) interior logits/gaps；
- 每步投影保持相同 physical endpoints/span；
- 不学习 attention temperature；
- 其训练 loss control 与 bilevel outer control 共享相同自由度和优化预算。

前者测试 held-out post-training objective，后者测试 ordinary in-training
frequency learning。若后者赢，说明额外 bilevel complexity 没有必要；
若两者都不转移，结论应是该实例下 shape 不可稳定识别。

### 7.6 一个廉价但不冒充终极目标的桥梁

从多个 schedule/checkpoint 抽取真实 positional-response targets
\(H\)，定义 full feature matrix

\[
\Phi_\Omega(\Delta)=
[\cos(\Delta\omega_k),\sin(\Delta\omega_k)]_{k=1}^K.
\]

对 lag weights \(W\)，ridge readout

\[
A^\star=
(\Phi^\top W\Phi+\lambda I)^{-1}\Phi^\top WH
\]

和 held-out linearized error

\[
J(\Omega)=
\|W^{1/2}(H-\Phi_\Omega A^\star)\|_F^2
\]

可作为 no-training gate。它必须：

- cross-fit across schedules/checkpoints；
- 在未参与构造的 targets 上评估；
- 同时匹配 span、bandwidth、low moments 和 conditioning；
- 明确只是一阶 bridge，不是 transformer risk theorem。

---

## 8. 候选方法的可证伪预测

若“operational bilevel shape”比 Cosh 更接近训练后目标，应同时预测：

1. endpoint/span 固定后仍给出非零、跨 design seeds 同向的 shape
   hypergradient；
2. short-unroll hypergradient 与 finite difference 的 sign/cosine 一致；
3. candidate 在 calibration workload 上优于 endpoint-normalized Cosh；
4. candidate 在未见 training seeds 上保持方向；
5. candidate 经过 best-range Geo/FMR envelope 后仍保留 shape gain；
6. candidate 在 held-out base/head/length 联合配置上不需重调；
7. ID no-harm 约束成立；
8. 与 direct exact-span LeRoPE 比较后，outer objective 的额外收益仍在；
9. 若 attention two-band 是真实 target structure，而不是 selection overfit，
   candidate 会向相似的多带/非 Cosh 结构移动；
10. 若 bandwidth/conditioning 才是主因，匹配 moments/condition number 后
    Cosh 与 alternatives 的差异会显著收缩。

任何一条失败都应缩窄 claim，而不是继续增加 schedule 参数化。

---

## 9. 最小验证方案（本次未运行训练）

### G-1：无训练身份 gate

- 冻结并 hash \(a,R,z,\omega\)、operator、grid convention；
- 验证 base–\(u\) gauge forward parity；
- 验证同步 pair permutation parity；
- 验证 sparse-lag alias 与加入破别 lag 后的消失；
- 对每个 arm 报 endpoints、span、moments、minimum spacing、phase
  condition number；
- 预注册 workload weights、metrics、seeds 和 stop rules。

### G0：冻结 checkpoint bridge

当前 checkout 无 checkpoint，因此暂时 **blocked by missing artifact**。
只有恢复且 hash 核验后才可：

- 在 checkpoint 上做 endpoint-preserving symmetric finite differences；
- cross-fit \(\partial{\rm NLL}/\partial\log\omega_k\)；
- 用 document-disjoint anchors 检验 direction。

它只能 falsify local proxy，不能证明 retraining optimum。

### G1：short-unroll gradient gate

- 两个 design seeds；
- 两种 unroll horizons；
- 一个 projected Cosh→candidate direction；
- finite-difference sign 与 hypergradient cosine 必须稳定；
- 任一不一致立即停止。

### G2：最小完整训练 selection

固定 50M-tier、同一 model initialization/token order/RNG/optimizer/global
batch/token budget，arms：

1. native Std Geo；
2. raw-range uniform；
3. raw Cosh；
4. endpoint/span-normalized Cosh；
5. endpoint/span-matched Exp；
6. frozen attention two-band；
7. exact-span direct-learned frequency；
8. bilevel candidate；
9. best-range Geo envelope。

Primary workload 在实验前固定，不允许看表后选 length weights。建议把
2×/4×/8× tail NLL 的加权和作为唯一 primary，ID NLL 和 retrieval
作为 co-primary/no-harm，不用一个 proxy 替代另一个。

进入 held-out gate 的最低阈值：

- candidate vs endpoint-Cosh primary delta \(\le-0.05\) NLL；
- 至少 2/3 deployment ratios 同向；
- 任一 ratio 不差于 `+0.05`；
- ID cost \(\le+0.02\)；
- best-range envelope 后仍保留方向。

这些阈值是 future preregistration proposal，不是已观察显著性。

### G3：冻结 candidate，独立训练 seeds

- 至少 6 个未见 paired training seeds；
- candidate、weights、range、test documents 全冻结；
- anchors 只描述 seed 内 measurement error；
- sequential testing 或 max-\(T\) simultaneous intervals 控制多重比较；
- 报全部 seed values，不只报均值。

### G4：held-out configuration

把同一个 candidate 无重调迁移到至少一个 held-out
`base/head_dim/Ltrain` configuration。当前
`base=1M,d_head=128,L=512` 同时改变三个因素且保持
\(128/\sqrt{512}=64/\sqrt{128}\)，只能验证联合新配置，不能分别识别
base/head/length scaling。

### 总停止条件

任一成立即停止“新最优 shape”路线：

- finite difference 与 hypergradient 不一致；
- endpoint/span 固定后 gain 消失；
- best-range envelope 后 gain 消失；
- held-out seeds 不一致；
- ID no-harm 失败；
- 只在一个长度有效；
- direct exact-span learning 与 bilevel 都不转移；
- 所有 meaningful contrasts 小于 0.02 NLL；
- candidate 的优越性只来自更多 selection/tuning budget。

---

## 10. 对当前论文、rebuttal 与后续研究的含义

### 10.1 当前论文贡献

仍可保留：

- RoPE 具有有限 spectral budget；
- training-time frequency allocation 是独立、可控、会影响训练后行为的
  设计轴；
- Cosh 是一个 closed-form、zero-learned-parameter、smooth analytic
  prior；
- range transport 与 training substrate 是不同决策阶段。

必须降级或纠正：

- Cosh unique 只能修饰 \(\mathcal C_{\rm app}\)，不能修饰真实训练目标；
- raw EVQ 不是 fixed-range pure shape；
- collision calibration、deployed-grid validation 和 KL scaling chain
  当前存在可复现的不一致；
- \(\tau=d/\sqrt L\) 是 empirical operating prior，不是由同一个
  surrogate 拟合预测出来的参数；
- “first allocation” 太宽。LongRoPE、Resonance、
  [RoPE-Mixed](https://arxiv.org/abs/2403.13298)（ECCV 2024，2D
  vision、layer/head-specific mixed-direction frequencies；未解决本文 1D
  shape 问题）、
  LeRoPE/AdaRoPE 都已直接改变、搜索或学习 frequencies。能防守的只剩：

  > 在已审计来源中，EVQ-Cosh 可能是首个针对 standard RoPE 的
  > closed-form、zero-learned-parameter、variational inverse-CDF
  > training-grid construction。

这个窄 novelty 仍需正式 related-work completeness 审核，不能把本报告
当作 novelty proof。

本审计没有改论文，也没有改变任何实验数字或重生成 PDF。

### 10.2 Rebuttal 表述

建议主动承认：

1. fixed range 下 allocation effect 已被支持；
2. Cosh 不是跨 length 的 empirical optimum；
3. formula 不是可靠近最优 selector；
4. 当前 author-confirmed aggregate 只报告 retargeted uniform 在每个 OOD
   length 的均值方向更好；强度只有 seed-42 report-backed 数值可定量；
5. DAPE 方法身份需要纠正；
6. exact-range 三 seed raw 尚未本地 promotion；
7. collision prefactor 与 grid/KL chain 不能继续作为强 calibration
   证据，除非先重算并解释。

安全 wording：

> Our evidence supports frequency allocation as a real training-time variable,
> not Cosh as a universal or trained-objective optimum. Under controlled span,
> different analytic shapes trade off across lengths; Cosh is a smooth
> parameter-free prior. The author-reported three-seed aggregate says
> retargeted uniform wins in mean direction at every tested OOD length, while
> only the seed-42 report currently quantifies that gap. The three-seed
> exact-range aggregate still awaits local raw promotion.

不安全 wording：

- “Cosh is the true attention shape”；
- “collision minimization validates \(c=1.171\)”；
- “formula is within 1% of optimum across 27 independent configs”；
- “shape and range are orthogonal/additive”；
- “DAPE learns the same object”；
- “no prior work optimized per-dimension RoPE frequencies”。

### 10.3 后续方法研究

最有前景的研究问题不是再找一个漂亮闭式 density，而是：

> 在固定 physical range、真实 full RoPE operator 和预注册 deployment
> workload 下，训练算法能否产生一个跨 seed 可转移的 frequency-shape
> hypergradient；若能，它是否优于 direct exact-span frequency learning？

Cosh 在这里的合理位置是 initialization、regularizer 或 low-complexity
baseline。若 operational objective flat/noisy，正确的新知识是：

> 在该 \(\kappa\) 下 pure shape 不可稳定识别。

而不是强行制造新 schedule。

---

## 11. 冻结主张的二次核验 ledger

| ID | 主张 | 最终状态 | 依据 |
| --- | --- | --- | --- |
| C01 | 存在 base–指数 gauge | **Verified** | 代数恒等式 |
| C02 | raw EVQ 混合 shape/endpoints/span | **Verified** | `scripts/lib/rope/schedules.py:86-140` + float64 重算 |
| C03 | Cosh 只在指定 surrogate/坐标下 unique | **Verified** | `paper/sections/03_theory.tex:23-45` + 坐标 push-forward |
| C04 | paper exact kernel 是 cosine-only，而真实 pair 有 sin/cos | **Verified** | `paper/sections/03_theory.tex:23`; `paper/appendix/a1_proofs.tex:464-475` |
| C05 | 两类 KL 在 scaling chain 中未被清楚区分 | **Verified inconsistency** | `paper/sections/03_theory.tex:93-108`; `paper/appendix/a1_proofs.tex:277-288,464-501` |
| C06 | collision-prefactor 表不是其明示 objective 的 minimizer | **Verified contradiction** | 独立脚本、负导数、commit provenance |
| C07 | surrogate table 使用 inclusive grid | **Verified numerical fingerprint** | 六个表值逐项 exact reproduction |
| C08 | fixed-span 后 Cosh 不支配，排名随 length 变 | **Verified from per-seed machine-readable aggregate; upstream raw/checkpoint absent** | `native_attention_shape_l128_results_20260724.json` |
| C09a | static selector 不能选 trained optimum | **Verified by CPU recomputation plus sanitized manifest** | selector failed gate |
| C09b | Q/K/phase proxies 对 causal utility 低相关 | **Report/hash-backed post-hoc support** | upstream raw/summary absent |
| C10 | 文献已直接搜索/学习 frequencies，但未解 controlled shape-only held-out problem | **Supported after primary-source audit** | LongRoPE/2、LeRoPE、AdaRoPE、Resonance 等 |
| C11 | task-independent universal optimum 不存在 | **Proved within a legal teacher-attention task family** | 不排除固定自然语言 \(\kappa\) 的 instance optimum |
| C12 | operational optimum 应条件化为 finite-training Pareto problem | **Well-defined proposal, not empirically validated** | 数学定义；bilevel literature boundary |
| C13 | 当前 provenance 不足以宣布新方法或三 seed exact-range closure | **Verified** | raw/checkpoint absence；author-confirmed fields |

---

## 12. 本次检查收据与未验证项

### Passed

- checkout identity：`main == origin/main == 498a43e3...`；
- paper/review/playbook/experiment files：read-only；
- deployed schedule/CDF identity：实现与解析式一致；
- raw EVQ endpoint/span 数值重算；
- native-shape JSON 逐 seed/逐 length 聚合复算；
- Phase16 sanitized manifest 关键统计与 standalone reanalysis 一致；
- training-free selector retained metrics 一致；
- collision representative configuration 独立重算；
- inclusive-grid surrogate table 六行精确复现；
- primary-source variable/target audit；
- universal-optimum counterexamples 代数复核。

### Failed or contradicted

- `c_coll` table 值作为明示 collision objective minimizers；
- surrogate-validation 表作为 deployed midpoint/native grid 的直接验证；
- formula 作为 stable near-optimum selector；
- Cosh 作为 cross-length dominating shape；
- static geometry 作为 trained risk selector；
- raw EVQ 作为 pure fixed-range shape intervention。

### Skipped by design

- 新模型训练；
- GPU run；
- checkpoint probing（当前工件缺失）；
- paper compile（没有 paper 修改，也受 repository prohibition 约束）；
- 修改 paper/playbook/已有 report；
- 任何 commit、push 或 deployment。

### 仍未验证

- exact-range 三 seed per-seed raw/CI；
- candidate bilevel hypergradient 的可重复性；
- exact-span LeRoPE vs Cosh 的 matched training；
- 更大模型、真实 downstream 和 held-out architecture 的转移；
- AdaRoPE/LeRoPE 结论在本仓库数据、模型和 workload 上的复现；
- operational Pareto frontier 的稳定性与 search cost。

---

## 13. 最终答案

对“是否存在比 EVQ-Cosh 更正确、更稳定或更接近训练后真实目标的方法”的
回答是：

1. **更接近声明清楚的训练后目标：有。** 直接 learnable frequencies 和明确 outer
   workload 的 finite-training bilevel objective，比 static Cosh/collision
   surrogate 更接近声明清楚的训练后风险。
2. **相对 EVQ-Cosh 已经更稳定：没有 matched direct 证据。**
   Anti-Periodic PE 已报告特定 in-window retrieval 的跨 seed variance
   降低，但它改变 head-wise frequency class、没有与 EVQ-Cosh 直接比较；
   LeRoPE/AdaRoPE 也改了更大的变量集合，现有 bilevel candidate 尚未训练
   验证。
3. **通用更优：在允许任务/训练协议变化时不存在。** 合法
   teacher-attention 任务族和有限训练算法的严格反例排除了
   task-independent universal schedule；不排除固定自然语言实例有自己的
   optimum。
4. **当前最有意义的可求解问题：** 固定 physical endpoints/span 与完整
   operator，预注册训练算法、deployment workload 和 seed risk，求
   held-out finite-training Pareto optimum；Cosh 只作 prior/baseline。

这一定义既能容纳负结果，也能把 shape、range、operator、training noise
和 tuning budget 真正分开；如果它找不到跨 seed 稳定方向，最终结论应是
“该实例下 shape 不可识别”，而不是继续寻找形式上的闭式最优。
