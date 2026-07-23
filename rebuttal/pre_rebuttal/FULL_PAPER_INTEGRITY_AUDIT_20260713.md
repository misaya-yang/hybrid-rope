# EVQ-Cosh Full Paper Integrity Audit

日期：2026-07-13

状态：`canonical_internal_audit`

用途：汇总提交稿的实现身份、数学、实验协议、统计与 provenance 问题，作为 2026-07-22 rebuttal triage 的最新事实入口。本文不修改任何实验数值，也不是可直接提交的 author response。

若本文与 2026-07-11 及更早的 rebuttal 文档冲突，以本文的事实边界为准；长篇数学推导仍可查阅 `THEORY_REBUTTAL_MATHEMATICAL_AUDIT_20260711.md`。

---

## 0. Executive verdict

这次审计不是发现“所有实验都无效”，而是发现论文把若干**仍然有效的本地方法与数值**赋予了错误的官方方法身份或过强理论解释。

最重要的结论如下：

1. **Primary I 数值可以保留，但不是官方 YaRN 结果。** 实际方法是一个固定通道边界的 YaRN-inspired smoothstep scaler。它与 YaRN 共享“高频保持、低频拉伸”的核心思想，因此降低 PPL 并不反常；但它缺少官方 YaRN 的 rotation-derived correction range 和 attention `mscale`，不能作为 faithful YaRN comparison。
2. **Primary II 的 “DAPE (32p)” 不是 DAPE。** 它只学习 32 个全层共享 inverse-frequency 参数。数值仍可作为 `Learnable shared inv_freq (32p)` control，但所有 DAPE head-to-head 解释必须撤回。
3. **核心 Geo 是 midpoint-discretized geometric schedule，不是标准 native RoPE endpoint grid。** EVQ 与这个 midpoint-Geo 的 shape contrast 仍是同网格干净对照；但没有 native endpoint baseline 时，不能声称击败标准 RoPE。
4. **cosh surrogate minimizer 定理成立；ordinary-KL 的 \(\tau=d/\sqrt L\) 推导不成立。** Ordinary baseline-to-perturbed KL 从 \(O(\tau^4)\) 开始。尺度规则最多是一个显式 phase-transport proxy 下的条件性结构，再加经验校准。
5. **`c_coll=1.171` 的 exact-kernel calibration 不成立。** 对应脚本没有求最优值，只复读了预置数值；独立优化找到远优于论文点的可行解。
6. **Phase 16 的 “27 configurations / all <1% PPL” 不成立。** Raw manifest 是 99 runs、9 个配置、先 pilot 再 selected-confirmation 的设计。用共同的 runner-defined weighted extrapolation log-PPL，只比较都有三 seed 的 formula tau 与 midpoint-Geo，formula 为 7/9 胜、2/9 负；这不是 27 个配置全部小于 1%。

这些问题**显著削弱 comparator fidelity、理论闭环和 reviewer trust**，但不抹掉以下信号：有限频率预算视角、给定 surrogate 下的 closed-form cosh family、EVQ 对 midpoint-Geo 的若干 matched empirical gains，以及 2x2 factorial contrast中自定义 fixed-ramp scaler 对两种 substrate的 differential leverage。当前没有 formal interaction test，不使用统计学意义上的 interaction claim。

本轮审计没有改动任何已报告 metric value。

---

## 1. 审计方法与来源优先级

### 1.1 方法身份必须从官方定义核对

- YaRN：官方论文 [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071)；官方仓库 [`jquesnelle/yarn`](https://github.com/jquesnelle/yarn)，审计锚点 commit `995db5b575e75230b3384d658f8b944c9662f775`。
- DAPE：官方论文 [DAPE: Data-Adaptive Positional Encoding for Length Extrapolation](https://proceedings.neurips.cc/paper_files/paper/2024/file/2f050fa9f0d898e3f265d515f50ae8f9-Paper-Conference.pdf)；官方仓库 [`chuanyang-Zheng/DAPE`](https://github.com/chuanyang-Zheng/DAPE)，审计锚点 commit `bde344a844f2bd1f498b2bac70240dcda41c50c1`。
- 本文方法身份不是由 class 名、注释、引用或旧实验报告决定，而是由实际 forward path 与参数化决定。

### 1.2 仓库证据优先级

1. Raw/sanitized result artifact 和历史 commit 中实际执行的 runner；
2. 当前 canonical implementation；
3. 论文表格和正文；
4. 实验报告、模拟审稿和旧 rebuttal 草稿。

Primary II 的历史锚点为 commit `8616af4`；Phase 16 的公开锚点为 `data/curated/phase16_99run_manifest.csv` 及其 meta 文件。

### 1.3 判定标签

- `VALID_NUMERIC`：数值仍可按实际协议报告；
- `RELABEL_REQUIRED`：数值可留，方法身份必须改；
- `INTERPRETATION_WITHDRAWN`：数值可留，但论文级结论不能由它支持；
- `PROVENANCE_LIMITED`：方向性结果存在，但 raw/runner/seed 闭环不足；
- `REMOVE_FROM_DEFENSE`：不得作为 rebuttal 防线。

---

## 2. 方法身份审计

| 提交稿名称 | 实际实现 | 与官方/标准定义的差异 | 数值状态 | 必须采取的动作 |
| --- | --- | --- | --- | --- |
| YaRN / Geo+YaRN / EVQ+YaRN | 固定 20%--90% 通道边界、smoothstep ramp、把 `temperature` 继续乘入 inverse-frequency divisor | 官方 YaRN 用 rotation count、`beta_fast=32`、`beta_slow=1` 得到 correction range，线性混合 extrapolation/interpolation，并用 `mscale` 调 attention 幅度 | `VALID_NUMERIC` + `RELABEL_REQUIRED` | 改称 `fixed-ramp scaler` 或 `YaRN-inspired fixed-ramp scaler`；不再声称官方 YaRN complementarity |
| Geo（Primary I--III） | \(u_k=(k+1/2)/K\) 的 midpoint geometric grid | 标准 native RoPE 使用 \(u_k=k/K\) endpoint grid | `VALID_NUMERIC` + `RELABEL_REQUIRED` | 改称 `Midpoint-Geo`；没有 native baseline 时不声称击败标准 RoPE |
| DAPE (32p) | 32 个共享 `log_inv_freq`，初始化为 midpoint Geo | 官方 DAPE 是作用于 attention score/PE signal 的 learned data-adaptive operator，不是可学习共享频率表 | `VALID_NUMERIC` + `RELABEL_REQUIRED` | 改称 `Learnable shared inv_freq (32p)`；撤回 DAPE-specific comparison |
| 当前 `phase11b_125m_dape.py` | repo-local Kerple/MLP-inspired 变体 | 也不是官方 DAPE 的 faithful reproduction | `PROVENANCE_LIMITED` | 不得用它补救旧 DAPE 标签；如未来比较必须重新做 parity audit |
| MLA “YaRN” supporting evaluator | wavelength-threshold 变体 | 与 Primary I 的 fixed-index scaler和官方 YaRN 都不同 | `RELABEL_REQUIRED` | 不把不同 scaler 汇总成同一方法族结果 |

### 2.1 为什么自定义 scaler 仍然能降低 PPL

当前实现并非随机扰动。对按 index 从高频到低频排列的 RoPE 通道，它保留前约 20% 高频，在 20%--90% 区间平滑缩频，并让低频尾部接近完整 context-scale 拉伸。这与 YaRN 的频率侧核心动机一致：

- 高频保持，保护局部相位分辨率；
- 低频拉长，降低长距离相位变化速度；
- 中间频段渐变，避免全频段 Position Interpolation 的短程损伤。

因此，当前三 seed PPL/PK 增益可以是真实的 local algorithm result。错误不在于“它一定无效”，而在于把一个有效的本地变体标成官方 YaRN，并据此声称与文献方法的互补性。

### 2.2 官方 YaRN 与当前实现的可复核差异

官方实现的核心步骤是：

1. 用原训练长度、base、dimension 与 rotation thresholds 计算 correction range；
2. 构造 `inv_freq_extrapolation = inv_freq` 与 `inv_freq_interpolation = inv_freq / scale`；
3. 用 linear ramp 混合两者；
4. 用 `mscale = 1 + 0.1 ln(scale)`（再乘 attention factor）缩放 sine/cosine amplitude。

当前仓库 `scripts/lib/rope/schedules.py` 与 `scripts/core_text_phases/eval_pe_baselines.py` 则：

1. 固定 `start=int(.20*K)`、`end=int(.90*K)`；
2. 使用 smoothstep；
3. 以 `scale**ramp * temperature**(.5*ramp)` 除 inverse frequencies；
4. 没有 attention-amplitude `mscale`。

以 \(d=64,b=500K,L_{train}=2048,s=8\) 为例，官方 rotation-derived 过渡区约在 channel 5--15，而当前实现约在 6--28，覆盖范围明显不同。

### 2.3 Midpoint-Geo 不是 native Geo，但仍是 geometric family

Midpoint grid 与 endpoint grid 的相邻频率比相同，因此它仍是几何序列；区别是所有频率统一乘上

\[
b^{-1/(2K)}.
\]

在 \(b=500K\) 时：

| \(K\) | 全局频率因子 | 等效 context stretch |
| ---: | ---: | ---: |
| 16 | 0.6636 | 1.5069x |
| 32 | 0.8146 | 1.2276x |
| 64 | 0.9026 | 1.1080x |

因此 midpoint 不是无关紧要的 index convention。Primary I/II/III 的 EVQ-vs-midpoint-Geo comparison 仍然隔离了同一 quantizer 下的 density-shape effect；但它没有隔离 EVQ 相对 native endpoint RoPE 的全部效果。

---

## 3. 理论审计

### T-01：cosh surrogate minimizer 成立

对论文明确写出的

\[
\mathcal C_{app}[\rho]
=\frac{\alpha}{2}\int_0^1\rho^2
+\frac{\beta}{2}\iint_{[0,1]^2}\rho(\phi)\rho(\psi)\min(\phi,\psi),
\]

在 \(\rho\ge0,\int\rho=1,\alpha>0,\beta\ge0\) 下，一阶变分和边界条件给出

\[
\rho''-\frac\beta\alpha\rho=0,\quad
\rho'(0)=-\tau^2,\quad \rho'(1)=0,
\]

其唯一解为

\[
\rho_\tau(\phi)=\frac{\tau\cosh(\tau(1-\phi))}{\sinh\tau},
\qquad \tau^2=\beta/\alpha.
\]

当 \(\beta>0\) 时使用上面的 cosh closed form；当 \(\beta=0\) 时按 \(\tau\to0\) 极限定义 \(\rho_0\equiv1\)，避免把闭式中的 `0/0` 当成直接取值。严格凸性来自 \(\alpha\int\rho^2\) 与 `min` kernel 的 positive semidefiniteness。这是可以保留的 exact theorem，但它只说明“给定 surrogate 后的最优 shape”，不说明 surrogate 等于真实 attention/LM objective，也不推出 practical \(\tau\)。

### T-02：ordinary KL 推导错误

令 \(\theta=\tau^2\)，若

\[
z_\theta=z_0+\theta g+O(\theta^2),
\]

则

\[
D_{KL}(p_0\Vert p_\theta)
=\frac{\theta^2}{2}g^T J_{sm}(p_0)g+O(\theta^3)
=O(\tau^4).
\]

KL 在相同分布处一阶变分为零。因而“\(O(\tau^2)\) ordinary-KL utility 与 \(O(\tau^4)\) stiffness 平衡，推出非零 \(d/\sqrt L\)”不成立。当前正文与附录中正确的 KL Taylor expansion 自相矛盾。

安全保留方式是另行定义 phase-variance / probability-transport proxy：

\[
U_{tr}(\rho;L)=\frac ML\int_0^1 \rho(\phi)q(Lb^{-\phi})d\phi,
\]

其中 \(M\) 是 utility proxy 实际相加的相位/频率通道数。沿 \(\rho_\tau=1+\tau^2\eta+O(\tau^4)\)，该**线性 allocation score**可以有 \(O(M\tau^2/L)\) 一阶变化。若 Pearson stiffness 另以维度 \(d_S\) 归一化，则所选 convention 下的 balance 形如

\[
\tau^2=45\lambda Q_1\frac{M d_S}{L}.
\]

只有再假设 \(M\) 与 \(d_S\) 都随同一个 architecture dimension \(d\) 成比例，才得到 \(\tau\propto d/\sqrt L\)。这条额外的 dimension-identification 不是 ordinary KL、task loss或 trained-attention theorem；prefactor、normalization、实际 \(M/d_S\) 身份与 finite \(\tau\) 均需经验校准。

### T-03：`c_coll=1.171` calibration 是错误证据

`scripts/analysis/verify_c_coll.py` 没有优化 collision score，而是硬编码表中 `tau_coll` 后重算比例。对论文定义的 representative case \(d=64,L=512,b=500K\)，独立搜索得到：

| 点 | \(\tau\) | collision score |
| --- | ---: | ---: |
| 论文 `c=1.170` | 3.3093 | 28.18385 |
| 更优可行点 | 13.0519 | 0.04207 |

只需存在这个更优可行点，就足以否定论文点是 exact-kernel minimizer。对应的 `c=4.6145` 也与 1.171 不接近。

因此以下内容全部进入 `REMOVE_FROM_DEFENSE`：

- `c_coll=1.171`；
- surrogate 与 exact-kernel 2% agreement；
- 0.28% CV、leave-one-out 误差；
- `lambda_infty=0.96`；
- exact-kernel 对单位 prefactor的独立闭环。

仍可保留的是固定点 directional diagnostic：在已列配置上，deployed EVQ allocation 相对 midpoint-Geo 降低所定义的 collision statistic。不得称 minimizer 或 task-level proof。

### T-04：waterbed 解释越界

论文写 \(\mathcal C_{app}[1]=0\) 不成立；实际为

\[
\mathcal C_{app}[1]=\alpha/2+\beta/6.
\]

在 \(\rho>0\)、相关积分有限、\(\alpha>0,\beta\ge0\) 的条件下，allocation-divergence waterbed inequality 本身成立；但它不能证明 PPL trade-off、长程增益或任务因果。安全措辞只能是“empirical behavior is consistent with an allocation trade-off”。

### T-05：LoRA rank 不是 frequency-channel count

LoRA rank \(r\) 不表示只能改变 \(r\) 个频率通道。一个低秩矩阵仍可影响全部输出坐标。因此 `1-r/K`、`r≈K` phase transition 与“48 个 frozen frequency channels”的推导没有数学依据；当 \(r>K\) 时 `1-r/K` 甚至为负，使所谓 stiffness失去合理性。论文只有训练过的 `r=64` row；所谓 rank sweep 是用同一观测校准后的 phenomenological landscape，不是多个 rank 的训练实验。对文中 \(R(x)=1/\sqrt{1+\Lambda_0e^{-x}}\)，同一 \(\Lambda_0\) 下 70% 与 95% recovery 的 \(x\) 只相差约 2.265，而不是声称的 5；`70%@5` 与 `95%@10` 分别隐含约 154 与 2380 的不同 \(\Lambda_0\)。这一整段不能保留为理论或经验 rank-sweep evidence。

### T-06：MLA dimension 与 practical \(\tau\)

- 实际 Primary III `head_dim=64`、`d_rope=32`、`d_nope=32`，所以每头 Q/K 的总 width 与 SDPA scale 对应 64，而 rotary 子空间只有 \(K=16\) 个 frequencies；
- \(\tau=1.414\) 不是 \(64/\sqrt{8192}\)；论文为了得到该值引入了代码中不存在的 ad-hoc doubled `d_eff=128`；
- `d_eff=128=2×actual head_dim` 最多是 ad-hoc empirical operating convention；写成 `d_eff=d_head` 是事实错误；
- `kv_lora_rank=256` 是层级共享 latent rank，不是每头额外 256 个 content dimensions，也不能用于把 64 改写成 128；
- practical \(\tau=1.414,4,5.66,8\) 都没有 small-\(\tau\) remainder guarantee；在 \(\tau=1.414\) 时，Pearson stiffness 的 \(\tau^4/45\) leading term相对 exact stiffness已约高估 52%。

Primary III 数值可以作为 empirical scarce-channel sensitivity，不再作为公式验证。

### T-07：其他必须明确的近似

“broadband surrogate 是 sole approximation”不成立。至少还存在：pure-tether omission、diffuse-softmax、proxy/task bridge、channel additivity、Pearson choice、dimension normalization、small-\(\tau\) 和 MLA `d_eff` convention。

此外必须保留下列边界：

- pure-tether forcing coefficient没有由 trained system测量，不能称 practical-\(\tau\) residual已受控；
- variable-\(\alpha\) Bessel branch只对另一 surrogate精确，但没有证据表明它会失去正性，不能用该说法为选择 cosh辩护；
- `lambda=1` 是相对 normalization/gauge choice，不是 units 自动决定；删除错误 `c_coll` 后没有独立 prefactor闭环；
- nonuniform attention下 probability displacement涉及 \(J^2\)，Fisher/KL curvature涉及 \(J\)，不能用同一个 `L_eff^J` 无证明地同时替换二者；
- generic transport/quantization bounds不预测 EVQ 相对 Geo 的 PPL符号或幅度；finite-channel bounds必须保留 density lower/upper bounds与 smooth-kernel等条件。

理论应固定为三层：

1. `Exact`：给定 convex surrogate 后的 cosh density、CDF、inverse CDF 与 quantization identities；
2. `Conditional proxy`：列明假设后的 phase-transport scaling structure；
3. `Empirical`：prefactor、finite \(\tau\)、MLA convention、PPL basin 与 task gains。

---

## 4. 实验与 provenance 审计

### E-01：Primary I 实际支持什么

可保留：

- 454M、FineWeb-Edu + 10% synthetic passkey、\(L_{train}=2048\)、3 seeds；
- midpoint-Geo 与 EVQ substrate 在同一个 repo-defined fixed-ramp scaler 下的 2x2 factorial contrast与 differential scaler leverage；
- 当前 raw-backed PPL 与 PK 数值。

不可保留：

- official YaRN identity；
- “EVQ 与 YaRN 的正交缺陷”这一数学表述；
- tuned YaRN dominance；
- 没有 faithful official implementation 时的社区部署结论。

Table 3 的 “per-document vs full-sequence” 解释也不成立。两者均使用连续 token CE evaluator；16K 差异来自 evaluation-length list 插入 12K 后，共享 `RandomState(9999)` 消耗顺序改变了 sampled offsets。表可保留，但 caption 应改为 sampled-offset sensitivity，并承认 eval chunk 数有限。

### E-02：Primary II 的真实协议

- 架构：hidden 768、12 layers、12 heads、head_dim 64、intermediate 3072、vocab 50304、tied embedding；实际参数量约 151.9M，而不是 125M；
- 数据：FineWeb-Edu；\(L_{train}=128\)，15M tokens；
- base LR `3e-4`，AdamW `(0.9,0.95)`，weight decay `.1`，2% warmup + cosine；
- effective batch 为 64 sequences（从长序列 tier 的 batch 16 按短序列放大），约 1831 optimizer steps；
- shared-frequency control 的 PE LR multiplier 为 100，即 PE LR `0.03`；
- Geo、shared-frequency control 与 fixed EVQ headline 是 seed 42；learnable-\(\tau\) row 是 3 seeds。

正确身份：`152M PE-dominant midpoint-grid diagnostic`。错误的 `125M / LR 6e-4 / batch 16 / DAPE` 组合不能用于复现说明。

### E-03：Primary III 不是严格 same-protocol 3-seed

- 模型是 repo-local MLA-style architecture，不是 production-identical DeepSeek；
- seed 42 用 batch 6；seeds 43/88 用 batch 5；
- 在固定 token budget 下，这改变 optimizer step count 与 cosine schedule；
- 所以 per-seed paired Geo/EVQ direction 仍可报告，但 std 不是纯 seed variation。

安全标签：`heterogeneous three-seed replication with paired comparisons within each seed`。

### E-04：Phase 16 过度汇总

Raw manifest 的真实结构：

- 99 runs；
- 9 个 `(L, num_heads, head_dim)` 配置；
- 45 个 seed-42 pilot runs（每配置 5 个 tau）；
- 54 个 confirmation runs：对 midpoint-Geo、formula tau 与 pilot-selected alternative 增加 seeds 137/256；
- 每 run 8,388,608 tokens；
- pilot 与 confirmation 的评测预算和 score组成不同：pilot使用8次 passkey、DSR关闭；confirmation使用16次 passkey与8次 DSR。`compute_selection_score`在有结果时把0.1 passkey和0.1 DSR加到 weighted log-PPL上，因此两阶段的 composite score并不直接可比。

使用 runner 自己的 weighted extrapolation log-PPL 定义，仅比较都有三 seed 的 formula tau 与 midpoint-Geo，formula 在 7/9 配置获胜，在 2/9 配置失败。因此：

- 可以说：`formula is a useful but fallible prior in this small 9-configuration study`；
- 不可以说：`27 configurations`、`all <1% PPL`、`near-optimal across the grid`；
- 旧 `exact best 3/9, top-2 6/9, top-3 8/9` 把 n=1 pilot-only arms 与 n=3 confirmed arms混排，且两阶段 score组成不同；在用共同 metric、共同 seed count重算前，这组 rank必须撤回，不能作为 formula optimality evidence。

### E-05：750M continuation 有 confound

Geo 与 EVQ 都从同一个 Geo-2K checkpoint retrofit；Geo effective batch 16、7629 updates，EVQ batch 14、8719 updates，EVQ 多约 14.3% optimizer updates，LR schedule也随 steps变化。此外，该 runner的 Geo是 native endpoint，而 EVQ使用 \(u=k/(K-1)\) 的 endpoint inverse-CDF，不是 Primary I--III 的 canonical midpoint EVQ。该 row同时有 update与 schedule-identity confound，不应进入 multi-scale trend，只能作为 `confounded exploratory continuation`。

### E-06：Video DiT provenance 未闭环

Runner 为每个 method 重新构造 model 与 AdamW；同 seed重置可得到相同初始化与 data order，但它们不共享训练中的 weights 或 optimizer state。论文 seed-42 报告的 Geo/EVQ `train=.01350/.01069, all=.00670/.00573, far=.00774/.00502`，而 tracked head-to-head summary为 Geo `.009113/.007244/.009891`、EVQ `.007204/.006064/.006388`；seed-137 exact raw artifact也未闭环。Video Geo还是 canonical tau-0 midpoint，不是 native endpoint。方向性可能存在，但在 provenance 修复前应从 rebuttal defense移除。

### E-07：旧 LoRA 表不能归因于 EVQ

旧表比较 unadapted Base-Geo 与 EVQ-frequency + LoRA + long-data training，同时改变三项；对应 aggregate 数值缺 tracked raw JSON，旧 evaluator只用 5 个 WikiText chunks。它不能隔离 EVQ effect。

2026-07-12 的 fresh LongAlpaca seed-42 artifact在 model、data/order、LoRA、optimizer和 evaluator上配对，但 schedule quantization并不相同：Geo是 native endpoint，EVQ是 midpoint。Domain-macro `EVQ+LoRA - Geo+LoRA` NLL为 8K `+0.38986`（更差；PPL `10.068 vs 6.817`，约1.477x）、16K `-1.51008`、32K `-2.04786`。因此它是单 seed、特定数据、teacher-forced NLL下的 matched training-pipeline comparison，不是纯 density-shape control；必须同时报告 8K degradation，且不得静默替换旧表。

### E-08：Figure 3 / scaler 标注

Figure 3 generator 读取 `yarn_auto`，但标注 fixed `s=8`。在 8K/256 条件，`yarn_auto=L_eval/L_train=32`；因此图中 `99.6/260.2` 是 auto-s32，不是 fixed-s8。Panel (a)已有并会使用 `data/curated/fig3_extreme_128.json` fallback；只有 panels (b,c)仍依赖缺失的 phase11 result路径，尚未接入已 tracked 的 `data/curated/phase11_l256_3seed_recovered.json`。图与 caption在未来修订稿中必须同步修正。

### E-09：QuALITY 与 supporting rows

QuALITY 是从 2K checkpoint 直接做 4K task finetune，不是先做独立 4K LM continuation再 finetune。其 `+YaRN` 仍是20%--90% fixed-ramp/no-mscale自定义 scaler；Geo evaluator使用 native endpoint而EVQ使用 midpoint，非同 quantizer。表内 n=2086 数值有 curated JSON，但公开 reproduction命令默认 `eval_samples=200, scoring_mode=gold_answer_nll`，没有传表格所需的2086/options-NLL，也不包含 task-finetune阶段，不能复现完整表。Progressive、QuALITY、video、LoRA与 750M均不得升级为 primary evidence。

### E-10：checklist / reproducibility

提交稿 checklist 中以下 `Yes` 需要重审：

- claims correctness：justification仍依赖27 configs、YaRN/DAPE身份与 \(\tau\) 推导，不能给无条件 Yes；
- theory proof correctness：只有 cosh surrogate theorem可给 Yes；scale proposition与 calibration不能；
- experimental setting/details：Primary II参数、Primary III batch、Phase16 design和多个 evaluator配置写错或缺失；
- full/open-access reproduction：Primary I/II/III 的 checkpoint/data hash/exact launcher closure不完整；supplement allowlist还缺 Primary I exact evaluator、Primary II历史 shared-frequency runner、Primary III extended evaluator与QuALITY finetune runner；
- compute：提交稿称 A100/H100 internal cluster与“few hundred accelerator-hours”，但 tracked Primary I/II reports明确记录 RTX 5090/external worker；这是正面冲突，不只是缺记录；
- statistical reporting：Primary II single-seed、Primary III heterogeneous batch、Phase16 selected-confirmation必须明示。

---

## 5. Reviewer-safe survivor set

### 5.1 可以继续防守

1. RoPE 的有限 frequency table 可以作为一个可设计的 finite spectral budget；这是机制视角，不是 information-theoretic capacity theorem。
2. 给定明确 convex surrogate，cosh allocation 是唯一 closed-form minimizer。
3. EVQ-Cosh 是 zero-learned-parameter 的 training-time frequency allocation family。
4. Primary I 的 3-seed结果支持 `EVQ substrate × repo-defined fixed-ramp scaler` 的 2x2 factorial contrast与 differential leverage；尚无 formal interaction test。
5. Primary II 支持 seed-42 的 `EVQ vs midpoint-Geo vs learnable-shared-frequency` PE-dominant diagnostic。
6. Primary III 支持 empirical \(\tau=1.414\) 在 scarce rotary-channel setting 中相对 midpoint-Geo 的方向性结果，并需披露 heterogeneous replication。
7. Fresh LongAlpaca LoRA temporal-NLL result可作为单 seed supporting evidence，不能承担主结论。

### 5.2 必须撤回或重标

- official DAPE comparison；
- official YaRN complementarity；
- improvement over native/standard RoPE；
- ordinary KL derivation of deployed \(\tau\)；
- `c_coll=1.171` exact-kernel validation；
- Phase16 `all <1% across 27 configurations`；
- LoRA rank/channel phase-transition theorem；
- MLA `d_eff=128` theorem；
- video shared-weight/shared-optimizer claim；
- 750M controlled scaling evidence；
- old Base-vs-EVQ LoRA causal interpretation。

---

## 6. YaRN follow-up：低成本算子诊断与完整方法实验分开

官方 YaRN 是一个使用 scaled RoPE 继续训练/微调的 context-extension method。只在已有 checkpoint 上替换推理期算子，不足以称 faithful YaRN method comparison。后续分两阶段，不能混写。

### 6.1 Stage A：official-formula-anchored zero-shot diagnostic（低成本）

前提是恢复 Primary I 的 3-seed midpoint-Geo/EVQ checkpoints。保留原 checkpoint，不声称它们是 native-endpoint或 YaRN-trained models，评测：

1. 原 midpoint-Geo / EVQ；
2. midpoint-Geo + `YaRN-derived operator transform`；
3. EVQ + `YaRN-derived operator transform`；
4. 现有 fixed-ramp arms作为 repo-local ablation。

固定同一 `scale=8`、原训练长度、`beta_fast=32`、`beta_slow=1`、`attn_factor` 与 evaluator，报告相同 PPL、teacher-forced PK 和已有 AR exact。该阶段只回答“官方公式结构替换现有 fixed ramp后，zero-shot方向是否仍在”，不能恢复 official-YaRN method complementarity。

### 6.2 Stage B：matched training comparison（需要训练）

Native endpoint baseline不能从 midpoint-trained checkpoint评测时换频率得到。先按完全相同的 base recipe得到 native endpoint Geo与 endpoint-quantized EVQ两个 substrate checkpoints，再从每个 checkpoint各分两臂做**等量 continuation/fine-tuning**：

1. native endpoint Geo + identity/no-scaler continuation；
2. native endpoint Geo + official YaRN continuation；
3. endpoint EVQ + identity/no-scaler continuation；
4. endpoint EVQ + 明确定义的 `YaRN-derived generalization` continuation。

四个 continuation arms必须使用相同 data/order/tokens/optimizer/steps；不能把未continued base checkpoint与YaRN-continued checkpoint直接比较。

先做 seed 42 cost gate；只有差异达到预注册的 material threshold再扩展到相同 seeds。Native Geo + official YaRN 可以称 faithful official baseline；YaRN在EVQ substrate上没有官方定义，始终必须称 `YaRN-derived generalization`。

### 6.3 实现 gate

- 在 native endpoint Geo 上，inverse frequencies、correction mask、cos/sin amplitude与 pinned official implementation逐元素 parity；parity不能从 native Geo自动外推到 midpoint或EVQ；
- 官方 `mscale` 必须作用于 sine/cosine amplitude，不能再次写进 phase divisor；
- frequency-aware generalization先把实际频率 \(\omega\) 映射到能在 native grid复现官方 dimension ramp的 virtual coordinate
  \[
  j_v=-\frac{d\log\omega}{2\log b},
  \]
  再使用官方 `find_correction_dim` 的 floor/ceil边界和 linear ramp；必须先证明在 native grid逐点退化为官方 mask；
- 同时做 `matched official Geo mask on EVQ` ablation，以分开 substrate effect 与 schedule-dependent mask effect；
- paper label -> artifact -> runner -> module -> pinned official source 的 trace必须自动保存；
- 无法达到 native parity时，所有相关 arm只能标 `YaRN-inspired`。

### 6.4 如何解释结果

- 若 Stage A方向保持：只说明 YaRN-equation operator diagnostic支持继续投入 Stage B；
- 若 Stage B显示EVQ对该generalization有更大的 differential scaler leverage：可以说“EVQ empirically complements a YaRN-derived range transform in this matched protocol”，不能在没有formal test时称 statistical interaction，也不能说数学正交或“official YaRN on EVQ”；
- 若增益缩小或消失：现有 fixed-ramp result仍有效，但应用结论收缩为该 repo-local scaler；
- 若 native Geo + official YaRN胜过EVQ generalization：必须如实报告，不能只保留 midpoint或 fixed-ramp版本。

---

## 7. Rebuttal correction map

若真实 reviewer 触发这些问题，回答顺序固定为：`correction -> unchanged fact -> withdrawn interpretation -> surviving narrower claim -> future fix`。

### 7.1 Baseline identity disclosure

> During our post-submission audit, we identified two comparator-identity errors. The row labeled “DAPE” actually learned a layer-shared inverse-frequency vector and did not implement the data-adaptive attention-score operator of Zheng et al.; we therefore relabel it as a learnable shared-frequency control and withdraw all DAPE-specific comparisons. Likewise, our “YaRN” arm used a repository-defined fixed-index smooth-ramp scaler rather than the official rotation-derived correction range and attention mscale; we therefore describe the existing result only as EVQ composed with this fixed-ramp scaler. The underlying reported values are unchanged, but the official-method interpretations should not be considered supported by the submission.

### 7.2 Geo identity disclosure

> We also clarify that the matched Geo control in the core sweeps uses midpoint discretization, \(u_k=(k+1/2)/K\), so the clean empirical contrast is EVQ versus midpoint-Geo on the same quantization grid. This isolates allocation shape within that family, but it is not a native endpoint-RoPE comparison; we withdraw any broader wording that implied otherwise.

### 7.3 Theory disclosure

> We identified an order error in the scale argument: ordinary KL between the baseline and the schedule-perturbed attention distributions has zero first variation and begins at \(O(\tau^4)\), so it does not derive a nonzero \(d/\sqrt L\) optimum. The exact theoretical result that remains is the cosh optimizer of the stated convex allocation surrogate. The deployed scale should be viewed as a proxy-motivated, empirically calibrated operating rule, not a trained-task or global-optimality theorem. We also withdraw the reported `c_coll` calibration because its verification script did not optimize the stated collision objective.

这些段落是内部组件。NeurIPS 2026 rebuttal 不能上传修订 PDF；最终只回答真实 review 触发的 3--5 个 score-driving concerns，并在必要时向 AC 做一条统一的 integrity disclosure。

---

## 8. 最终判断

当前论文仍有可辩护的核心，但必须换成更窄、更准确的身份：

> EVQ-Cosh is a closed-form, zero-parameter frequency-allocation family whose shape is exact for a stated convex surrogate and whose empirical value is supported by matched midpoint-grid stress tests. Existing results also show compatibility with a repository-defined fixed-ramp range scaler; faithful YaRN and native-endpoint comparisons remain open.

这比提交稿的 DAPE/YaRN/native-Geo/ordinary-KL 叙述弱，但它是真实、可复核且仍有研究价值的结论。
