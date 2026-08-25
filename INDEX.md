# INDEX — 理论、证据、代码与下一步

- **最后更新：** 2026-08-25
- **角色：** 本仓库唯一的**持久索引**。回答「已有什么、谁拥有它、下一步做什么」。
- **不负责：** 硬性规则（见 [`AGENTS.md`](AGENTS.md)）、易变状态（见
  [`paper-2027/HANDOFF.md`](paper-2027/HANDOFF.md)）。
- **内部文件：** 不进匿名 supplement。

三层分工，任何时候只有一个权威：

| 层 | 文件 | 内容 | 变更频率 |
| --- | --- | --- | --- |
| 规则 | `AGENTS.md` | 不可协商的约束、claim 上限、命名、计算与 Git 纪律 | 很少 |
| 索引 | `INDEX.md`（本文件） | 理论/证据/代码地图、目录职责、研究议程 | 每有新 owner |
| 状态 | `paper-2027/HANDOFF.md` | 当前 PDF/哈希/验证回执/Git/机器状态/作者待办 | 每次会话 |

冲突时：**规则 > 索引 > 状态**。状态与索引冲突，回到 canonical owner 核验，
不要就地改索引。

---

## 1. 冷启动读序

1. [`AGENTS.md`](AGENTS.md) — 规则。
2. 本文件 §2–§4 — 已有什么。
3. [`paper-2027/HANDOFF.md`](paper-2027/HANDOFF.md) — 现在在哪。
4. [`paper-2027/main.pdf`](paper-2027/main.pdf) 与其 `sections/`、`appendix/` —
   reviewer 看到的真相。
5. 本文件 §6 — 下一步。

动任何 claim 或数字之前，必须读到 §3 表格里那一行指向的 canonical owner。
**不要**从最新日期、外部评审、preflight 或文件名开始。

---

## 2. 理论索引

### 2.1 现行理论（当前论文使用）

| 对象 | 结论 | Owner |
| --- | --- | --- |
| 有限谱基与稳定秩恒等式 | $r_2(R)=\dfrac{2K}{1+(K-1)\bar c}$，块白化后精确成立 | [`FULL_ROPE_...20260819`](paper-2027/research/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md) §2.3 |
| 相位不变碰撞度量 | $c_{\omega\nu}=\tfrac12\lVert S_\omega^{-1/2}H_{\omega\nu}S_\nu^{-1/2}\rVert_F^2$；对 pair 内旋转与基变换不变 | 同上 §2.2 |
| 精确正交格（uniform 测度） | 自由 support + uniform 距离测度下 $r_2=2K$ **可精确取到**；同宇称格 $\omega_k=\pi a_k/L$ 给出一个构造（$L=256$：$K=32$ 可取全偶 $a_k$，$K=16$ 可取全奇）。同一格在三角权重下不再正交 | 复现 owner：`third_axis_ceiling.py --decompose`；原始推导与容量律 $K^*=\lceil\lfloor L/\pi\rfloor/2\rceil$ 在 [`analysis/full_rope_audit/`](analysis/full_rope_audit/) |
| 低频 spectral collapse | $L_2$ 度量下 $V_\omega\to\mathrm{span}\{1,\Delta\}$；softmax 度量下 $\to$ centered $\mathrm{span}\{\Delta,\Delta^2\}$ | 同上 §3 |
| 因果变量分解 | $x_k=-\log\omega_k=a+Rz_k$；support $(a,R)$ 与内点分配 $z$ 分离 | [`ROPE_CAUSAL_VARIABLES_...20260823`](paper-2027/research/ROPE_CAUSAL_VARIABLES_AND_ZERO_TRAINING_RETROFIT_20260823.md) |
| 冻结 transplant 障碍 | 不等频率 multiset 下精确可逆 Q/K 补偿受阻 | [`OLMO2_POSTHOC_..._20260726`](rebuttal/rebuttal_0723/theory_results/OLMO2_POSTHOC_FREQUENCY_TRANSPLANT_OBSTRUCTION_20260726.md) |
| EVQ-Cosh 构造 | 闭式、零学习参数；**仅**对所述凸 surrogate 唯一 | [`ICLR2027_RESEARCH_SYNTHESIS_20260819`](paper-2027/research/ICLR2027_RESEARCH_SYNTHESIS_20260819.md) |
| claim 架构 | 实现版的 claim 结构与证据层级 | 同上 |

### 2.2 理论优化层（已闭合，尚未进正文）

[`three_completions/optimization_notes.md`](paper-2027/research/three_completions/optimization_notes.md)
把三处各自假设均匀先验的松散结构压成**单一可测先验 $\mu$** 的泛函链。

| 编号 | 内容 | 状态 |
| --- | --- | --- |
| O1 | $L_{\rm rng}$ 由 $\mu$ 导出，不是自由参数；$\omega_0L=2.0772$ | 已完成 |
| O2 | $L_{\rm eff}^J=n\,\mathrm{Var}_{U_n}[G]/\mathrm{Var}_\mu[G]$ | 已完成，**但见 §3.4 的反证** |
| O3 | $\gamma_{\rm eff}$ 闭式；$\tau_*(L)$ **不是幂律**，$p\approx0.85$ 诊断可撤 | 已完成 |
| O4 | 统一泛函 $\mathcal J[\rho]$ 合并 collision（形状）与 resolution（尺度），给出闭式 **EVQ-Cosh-R**（cosh + $\varphi_*$ 处下跳 $\kappa$） | 已完成 |
| O5 | arcsine 猜想**证伪**；剩余为第二类 Fredholm 方程，建议 Nyström | 已完成（结论为否） |
| O6 | re-adaptation 秩界不可得，只有线性化替代 + 可测桥 O6′ | 需进一步研究 |
| O7 | **共享频率表对头间异质 $\mu$ 是 Jensen 次优的**；per-head/分组 $\varphi_*$ 有理论依据 | 需进一步研究 → §6 |
| O8 | $\Lambda(\mu)$ 的正则性缺口；朴素参与比**不是**上界 | 需进一步研究 |

验证脚本：[`verify_optimizations.py`](paper-2027/research/three_completions/verify_optimizations.py)、
[`verify_three_completions.py`](paper-2027/research/three_completions/verify_three_completions.py)（纯 CPU，~60 s）。

### 2.3 历史理论（已被取代，只读）

`docs/tau_algor/`（15 篇，2026-03）与 `docs/archive/`（4 篇）是 τ scaling /
habitable zone / softmax transport 的原始推导。**它们是当前理论的前身，不是当前
权威。** 引用时必须标注为历史推导。

冷启动只需记住两条：$\tau_*$ **不是幂律**（被 O3 解释掉，拟合形式本身错），且
$\tau_*=\max(d_{\rm head}/\sqrt L,\;1.4)$ 是 operating prior 而非普适律（见
`AGENTS.md` claim 上限「Finite tau」）。**逐条取代关系由
[`docs/tau_algor/README.md`](docs/tau_algor/README.md) 与
[`docs/archive/README.md`](docs/archive/README.md) 拥有，本文件不复制。**

`docs/tau_algor/` 由 `tau-theory-assistant` skill 使用，因此保留原位、不做删除。

---

## 3. 证据索引

### 3.1 主因果证据

| 问题 | Canonical owner | 结果 | 最高可用 claim |
| --- | --- | --- | --- |
| 固定 support 下内点分配 $z$ 是否因果有效 | [`EXACT_RANGE_151M_3SEED_RESULT_20260820`](paper-2027/research/EXACT_RANGE_151M_3SEED_RESULT_20260820.md) | $+0.026/-0.281/-0.176/-0.146$，3/3 seed | 固定采样 support 下的纯内点分配识别 |
| target-matched 下同一对比 | 同上 §4 | $+0.026/+0.060/+0.227/+0.460$，0/3 | 部署边界；support 与 $z$ 是不同但相互作用的坐标，本协议不支持可加性结论 |
| weights×table 共适应 | 同上族 + `attention_fisher_50m_probe.py` | 2×2 PPL `7.14 / 76.20 / 23.05 / 7.16` | 诊断；与 exact-range estimand 分开 |
| M4 exact-range 因子实验 | [`M4_EXACT_RANGE_FACTORIAL_RESULT_20260726`](rebuttal/rebuttal_0723/theory_results/M4_EXACT_RANGE_FACTORIAL_RESULT_20260726.md) | — | 与 151.9M 共同拥有内点识别 |

### 3.2 规模与系统证据

| 问题 | Owner | 角色 |
| --- | --- | --- |
| 稀缺通道系统旗舰（432M MLA，3 seed） | [`table18_mla_3seed_aggregate.json`](data/curated/table18_mla_3seed_aggregate.json) | 系统旗舰 |
| 750M 续训持久性 | [`2026-03-06_phase15_750m...`](docs/exp/2026-03-06_phase15_750m_2k_to_4k_continue_results.md) | 训练阶段持久性 |
| 1.485B 同初始化 | [`OLMO2_1B_RELEASED_ROPE_BASELINE_20260725`](rebuttal/rebuttal_0723/theory_results/OLMO2_1B_RELEASED_ROPE_BASELINE_20260725.md) | 预训练规模上限 |
| 8B 适配 | [`EVQ_8B_ADAPTATION_EVIDENCE_20260724`](rebuttal/rebuttal_0723/theory_results/EVQ_8B_ADAPTATION_EVIDENCE_20260724.md) | 适配/能力证据，**不是**规模因果 |
| 成熟相位暴露 | [`OLMO2_1B_SELECTIVE_QK_PHASE_ADAPTATION_20260729`](rebuttal/rebuttal_0723/theory_results/OLMO2_1B_SELECTIVE_QK_PHASE_ADAPTATION_20260729.md) | 协议特定能力证据 |

### 3.3 成熟 checkpoint retrofit

全部 owner 在 [`attention-aware-retrofit/`](paper-2027/research/attention-aware-retrofit/)，
子目录分工见该目录 README。

| 问题 | Owner | 判决 |
| --- | --- | --- |
| 同 support 的 $z$ 控制 | [`SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823`](paper-2027/research/attention-aware-retrofit/results/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md) | 内部因果案例 |
| 零训练 session 策略 | [`SESSION_BINARY_S4_REAL_CONTEXT_RESULT_20260823`](paper-2027/research/attention-aware-retrofit/results/SESSION_BINARY_S4_REAL_CONTEXT_RESULT_20260823.md) | 部署/能力证据 |
| 新数据上的持久性 | [`FRESH_FINEWEB_S4_GENERALIZATION_RESULT_20260824`](paper-2027/research/attention-aware-retrofit/results/FRESH_FINEWEB_S4_GENERALIZATION_RESULT_20260824.md) | 封闭正向确认 |
| 长度条件化 budgeted retrofit | [`LENGTH_CONDITIONED_BUDGETED_RETROFIT_RESULT_20260822`](paper-2027/research/attention-aware-retrofit/results/LENGTH_CONDITIONED_BUDGETED_RETROFIT_RESULT_20260822.md) | **RULER core-4：0.5825@8K / 0.4000@16K，对官方 YaRN 0.5375 / 0.0125；零学习参数** |
| 联合 in-window/外推可行性 | [`EXPERIMENT_REPORT_20260821`](paper-2027/research/attention-aware-retrofit/results/EXPERIMENT_REPORT_20260821.md) | **phase-chord 两 seed Pareto：$+0.0007/-0.161/-0.156/-0.205$**；seed 范围阻止晋升 |
| 跨 owner 决策备忘 | [`POST_GPU_REFLECTION_..._20260824`](paper-2027/research/attention-aware-retrofit/analysis/POST_GPU_REFLECTION_AND_PROBLEM2_ROADMAP_20260824.md) | 决策备忘，**其路线判断见 §6 的修订** |

### 3.4 已证伪 / 已关闭（**不要重做**）

这是本仓库的反重复机制。任何新候选在开工前必须先对照本表。

| # | 被证伪的对象 | 证据 | Owner |
| --- | --- | --- | --- |
| 1 | cosine-only collision kernel | $C_{\cos}(A)<C_{\cos}(B)$ 但 $r_2(A)<r_2(B)$ | 全 RoPE 报告 §4.3 |
| 2 | collision/logdet 最小化作为外推目标 | 退化为 Fourier comb，$\Phi(\Delta+L)=\Phi(\Delta)$ 精确 aliasing | 同上 §4.2–4.3 |
| 3 | $\kappa_{\rm att}$ attention-Fisher 序 | 两种排序矛盾，Branch C | [`KAPPA_..._AUDIT_20260820`](paper-2027/research/audits/KAPPA_ATTENTION_MEASURE_AUDIT_20260820.md) |
| 4 | LeRoPE $w^{1/3}$ 曲率 oracle | 失败 | [`LEROPE_PROFILE_ORACLE_AUDIT_20260820`](paper-2027/research/audits/LEROPE_PROFILE_ORACLE_AUDIT_20260820.md) |
| 5 | arcsine 猜想 | 等 stiffness 下自由最优非 U 型 | optimization_notes O5 |
| 6 | direct attention-distance map（无相位核） | `STOP_DIRECT_DISTANCE_MAP` | [`EXPERIMENT_REPORT_20260821`](paper-2027/research/attention-aware-retrofit/results/EXPERIMENT_REPORT_20260821.md) §4 |
| 7 | $D^*$ 作为 retrofit 设计目标 | Spearman $-0.55$，**符号相反** | [`RETROFIT_AXIS_FALSIFICATION_20260822`](paper-2027/research/attention-aware-retrofit/analysis/RETROFIT_AXIS_FALSIFICATION_20260822.md) |
| 8 | coverage 残差 | Spearman $-0.25$ | 同上 |
| 9 | phase risk（「已 wrap 过的通道安全」） | Spearman $0.000$ + `one_turn_floor_s2` 决定性反例 | 同上 |
| 10 | direct-$z$ 两文档标定 | per-row gate 失败 | [`DIRECT_Z_FIXED_SUPPORT_PILOT_RESULT_20260824`](paper-2027/research/attention-aware-retrofit/results/DIRECT_Z_FIXED_SUPPORT_PILOT_RESULT_20260824.md) |
| 11 | 两个解析静态单表候选 | 改善 2×、损伤 1× | [`ZERO_PARAMETER_SINGLE_TABLE_RESULT_20260824`](paper-2027/research/attention-aware-retrofit/results/ZERO_PARAMETER_SINGLE_TABLE_RESULT_20260824.md) |

**结构性结论：** 第 1–11 项全部是**同一张共享一维表在假定 content model 下的标量
泛函**。而 §3.1 的 2×2 显示 LM 结果由 table×weights **交互**主导（换表后 PPL
`7.14→76.20`）。看不见权重的泛函在预测一个非主导项。**再提第 12 个同类 score
之前，必须先说明它如何逃出这一类。**

### 3.5 未决（**不是**失败）

| 对象 | 现状 | 为什么未决 |
| --- | --- | --- |
| phase-isotropy / pair-volume / min-eigenvalue | [`PHASE_ISOTROPY_50M_M4_RESULT_20260824`](paper-2027/research/attention-aware-retrofit/results/PHASE_ISOTROPY_50M_M4_RESULT_20260824.md)、[`PHASE_ALLOCATION_M4_EXTENDED_RESULT_20260824`](paper-2027/research/attention-aware-retrofit/results/PHASE_ALLOCATION_M4_EXTENDED_RESULT_20260824.md) 均为 `SCREEN_UNRESOLVED` | 50M regime 的 anchored-Cosh 方向与 151.9M owner 不同，触发其预注册 unresolved 分支。跨模型和 token budget 的符号差是 regime evidence，不是 noise-floor 估计或 candidate rejection。详见 §6.2 |
| O2 的 $L_{\rm eff}^J$ 机制解释 | 反演出 $\mathrm{sd}_\mu\approx46$–$72$ | 仓库内唯一实测注意力距离分布（GPT-2 125M / WikiText-103，`results/m4_max_36gb/D_attention_*.npy`）给出 $\mathrm{sd}_\mu=295.5$，对 uniform 的 295.3，方差比 0.9985 → 修正仅 0.15%。**P1 仍是最便宜的判定** |
| per-head / per-layer 分配 | 代码完成、**从未训练** | `heterogeneous_rope_5090`：`OFFLINE_CODE_COMPLETE / GPU_TRAINING_NOT_STARTED` |
| per-head adapter 秩分配 | 同参数量下 0.89 vs 0.09 可修复比例 | 从未运行；被现行 roadmap 禁用，见 §6.3 |

---

## 4. 代码索引

| 需要 | 位置 | 规则 |
| --- | --- | --- |
| **频率表实现权威** | [`scripts/lib/rope/`](scripts/lib/rope/) | `schedules.py`（EVQ-Cosh 分位）、`target_free.py`（target-free 相位延拓）、`fixed_support_z.py`（可微内点 $z$）、`official_yarn.py`、`learnable_evq.py` |
| 主实验链 runner | [`scripts/core_text_phases/`](scripts/core_text_phases/) | canonical runner；新主实验放这里 |
| 可复用 CPU 诊断 | [`scripts/analysis/`](scripts/analysis/) | 不自动成为 paper claim |
| 注意力需求测量 | `scripts/analysis/attention_phase_demand.py` | 含 `layerwise_plan()` → `per_layer_inv_freq` |
| 全 RoPE 碰撞审计 | `scripts/analysis/full_rope_collision_audit.py` | §2.1 的数值 owner |
| **第三轴静态 $r_2$ 搜索诊断** | [`scripts/analysis/third_axis_ceiling.py`](scripts/analysis/third_axis_ceiling.py) | §6.1 数值的可复现脚本；纯 CPU；报告 optimizer 的 best-found value，不是全局或行为上限 |
| 2026-08-19 全 RoPE 审计（有限 $K$、反例、正交格） | [`analysis/full_rope_audit/`](analysis/full_rope_audit/) | §2.1 正交格行与 §3.4 第 1–2 项的原始 owner；`finK_*`、`verify_small_models.py` |
| supporting evaluator | [`scripts/supporting_eval/`](scripts/supporting_eval/) | endpoint 身份必须由 owner 确认 |
| 独立规模实验包 | [`experiments/`](experiments/)、`rebuttal/rebuttal_0723/experiments/` | supporting，除非显式 promotion |
| 回归门禁 | [`tests/`](tests/) | 改 source-of-truth 时同步；导航门禁是 `tests/test_repository_navigation.py` |
| supplement 打包 | `scripts/package_supplement.py` | 从仓库根运行，`--profile iclr2027` |

---

## 5. 目录职责

| 路径 | 职责 | 规则 |
| --- | --- | --- |
| `paper-2027/` | ICLR 2027 唯一活跃稿件、图表、构建、交接 | 所有 claim 先过 owner |
| `paper-2027/research/` | durable internal theory、审计、claim/evidence 决策 | 内部层，不直接复制进正文 |
| `paper-2027/research/attention-aware-retrofit/` | 成熟 retrofit 的 `results` / `evidence` / `analysis` / `preflights` / `theory` | 各子目录职责见其 README |
| `paper-2027/research/audits/` | 内部 theory/manuscript/evidence 审计 | 不建立第二 action queue |
| `paper-2027/research/external-reviews/` | 外部模型独立复核 | untrusted input，必须回 owner 核验 |
| `paper/` | NeurIPS 2026 投稿基线 | **不可修改、不可编译、不可重生成** |
| `rebuttal/rebuttal_0723/` | NeurIPS review、回复历史、成熟实验 owner | 历史证据层，**不是** action queue |
| `scripts/` | 见 §4 | — |
| `data/curated/` | 已清洗、可跟踪的 machine-readable evidence | reviewer-safe 候选层 |
| `docs/overview/` | NeurIPS-era provenance、复现、术语 | 有用但不覆盖当前 ICLR 路由 |
| `docs/exp/` | 历史实验报告 `YYYY-MM-DD_slug.md` | 归档层 |
| `docs/tau_algor/`、`docs/archive/` | 历史 τ 理论推导 | 只读；取代关系见 §2.3 |
| [`analysis/`](analysis/) | 一次性深度审计的独立工作区（当前只有 `full_rope_audit/`） | tracked 历史 owner；新的可复用诊断放 `scripts/analysis/`，不放这里 |
| `paper_experiments/` | manifest 驱动的 experiment-code 视图 | canonical code 仍在 `scripts/` / `experiments/` |
| `tests/` | 实现、协议、证据、supplement 回归门禁 | — |
| `results/` | tracked history + ignored local output | **不能自动升级为证据** |
| [`internal/`](internal/) | NeurIPS-era 工作归档（2026-03/04 记录、旧稿快照、计划、审计、skill 备份） | 只读历史层，不再接收新文件；不公开、不整体打包；见其 README |
| `nonuniform-alloc/` | 私有工作层 | 未经明确请求不修改、不公开 |

### 新文件放哪里

| 类型 | 位置 |
| --- | --- |
| 易变交接状态 | 只更新 `paper-2027/HANDOFF.md` |
| 新理论/证据 owner 上索引 | 本文件 §2 / §3 |
| durable paper-facing research note | `paper-2027/research/` |
| retrofit 完成结果 | `.../attention-aware-retrofit/results/` |
| 机制分析 / 证伪 | `.../attention-aware-retrofit/analysis/` |
| 预注册 / 已撤销协议 | `.../attention-aware-retrofit/preflights/` |
| 紧凑 receipt | 对应 `evidence/` |
| 外部模型评审 | `.../external-reviews/<source-date>/` |
| 可复用分析代码 | `scripts/analysis/` |
| 新主实验 runner | `scripts/core_text_phases/` |

**不要**再创建：第二份根级 Agent、第二份 handoff、第二份索引/REPO_MAP、第二份
provenance manifest、第二个 rebuttal control room、额外的投稿 PDF 入口。

---

## 6. 研究议程与下一步

> 本节是**研究判断**，不是 claim，也不是已授权的计算。任何 GPU 运行仍需
> `AGENTS.md` §4 的显式授权与门禁。

### 6.1 第三轴的静态几何诊断

`r_2` 搜索回答一个受限问题：在声明的距离测度和固定 support 下，数值优化器能把
块白化稳定秩推到哪里。它可以比较表的静态几何，但不能给行为第三轴定标，也不能
回答哪个 $z$ 会获得更好的 LM 结果。

| 级 | 对象 | 定义 | 现状 |
| --- | --- | --- | --- |
| L0 | 代数上界 | $r_2\le 2K$，当且仅当 $\bar c=0$ 取到 | 精确；uniform 距离测度下有显式构造 |
| L1 | 静态搜索参考 | 固定 $(a,R)$ 后优化 $r_2$ | 已运行；只有 best-found value，无全局证书 |
| L2 | 需求加权泛函 | $\mathcal J[z;\mu]$，$\mu$ 为实测注意力距离分布 | 理论对象；既有静态排序失败限制其用途 |
| L3 | 结构异质性 | 共享表与逐层/逐头表的几何差 | 尚无行为结果 |
| L4 | 行为搜索 | 在训练装置上比较冻结候选 $z$ | 未测；搜索本身会引入选择偏差，不是无成本上限 |

**L1 数值参考。** Owner：[`scripts/analysis/third_axis_ceiling.py`](scripts/analysis/third_axis_ceiling.py)。
约定必须随数字一起引用：因果三角测度 $p(d)\propto(L-d)$、端点精确锚定、
$\tau=4$、Adam、固定 restart seeds。表中 `best-found` 是本次 optimizer 的最大观测
值，只是未知 supremum 的下界。

| $K$ | $L$ | base | $\omega_{\min}L$ | 几何 | EVQ-Cosh | best-found $r_2$ | $2K$ | spread | EVQ 占 best-found 改善量 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 256 | 256 | `1.19` | `12.892` | `45.719` | **`54.852`** | 64 | `0.019` | **`78.2%`** |
| 32 | 256 | 10 000 | `3.4e-2` | `5.563` | `23.090` | **`54.846`** | 64 | `0.021` | `35.6%` |
| 32 | 256 | 500 000 | `7.7e-4` | `3.973` | `13.065` | **`54.852`** | 64 | `0.036` | `17.9%` |
| 32 | 1024 | 10 000 | `1.4e-1` | `9.400` | `43.360` | `63.206` | 64 | `0.022` | `63.1%` |
| 32 | 2048 | 256 | `9.51` | `46.077` | `63.634` | `63.806` | 64 | `0.002` | `99.0%` |
| 32 | 2048 | 10 000 | `2.7e-1` | `12.797` | `52.075` | `63.788` | 64 | `0.015` | `77.0%` |
| 16 | 256 | 256 | `1.41` | `12.565` | `28.366` | `30.684` | 32 | `0.025` | `87.2%` |
| 16 | 2048 | 10 000 | `3.6e-1` | `12.230` | `28.559` | `31.975` | 32 | `0.001` | `82.7%` |

第一行是 §3.1 exact-range 151.9M 主因果实验的实际 support（$L=256$、base=256、
$K=32$），$\omega_{\min}L=1.1892$ 与 §6.2 独立一致。

**正确读法：**

1. 在三个已测的宽 support 上，$L=256$ 的 best-found 值都约为 `54.85`；这说明
   这些约束在当前 optimizer 解附近近乎不绑定。它不建立 support invariance。同一
   脚本在 $K=32,L=256$、base `1.1` 和 `2` 上分别只得到约 `7.70` 和 `32.25`，
   因此 support 明确可以改变 best-found $r_2$。
2. 块白化消除了每个 pair 的边际尺度，但 cross-pair collision 仍依赖频率所在的
   support。它会减弱某些慢端幅度效应，不会把 support 的代价普遍“洗掉”。
3. `78.2%/35.6%/17.9%` 是相对各行 geometric baseline 和各自 best-found point 的
   静态改善比例；分母随 base 改变，不能跨行解释成方法利用率或行为 headroom。
   `L=256, base=5e5` 还是一个诊断性 stress regime，不代表现代长上下文模型的完整
   工作点。

4. **静态度量显示几何分配没有充分利用新增 pair。** $L=256$ 下 $K$ 从 16 加到 32：
   几何的 $r_2$ 只从 `12.565` 走到 `12.892`（`+2.6%`），而 best-found 值从 `30.684`
   走到 `54.852`（`+79%`）。$L=2048$、base=10⁴ 同样：几何 `12.230`→`12.797`
   （`+4.6%`），best-found 值 `31.975`→`63.788`（`+99%`）。这是静态 basis
   utilization 的补充诊断；它不证明新增 pair 在训练后的模型中没有价值。

5. **距离测度与 support 的数值比较。** 同一 $K=32$、$L=256$
   （owner：`third_axis_ceiling.py --decompose`）：

   | 条件 | best-found $r_2$ | 观测结构 |
   | --- | --- | --- |
   | uniform 测度 + **自由** support | **`64.0000`** $=2K$，精确 | 落在同宇称格 $\omega_k=\pi a_k/L$，$a_k$ 与整数偏差 `0.0000` |
   | 三角测度 + **自由** support | `54.9281` | 格被破坏，$a_k$ 偏离整数达 `0.4869` |
   | 三角测度 + **钉住** support | `54.8519` | — |

   uniform 条件存在精确正交构造；其余两行只是相应 optimizer 的 best-found 值。
   没有全局证书时，不能把两段数值差精确归因为测度成本和 support 成本。

**与 §3.4 的关系。** 这个脚本不是新的 LM 排序器；它只刻画声明指标的数值景观。
因此它可以留作理论诊断，却不能据此选择候选表、估计剩余行为收益，或改变论文主张。

**事实 / 推断 / 待验证：**

- **事实：** 上表是声明 optimizer 与 restart seeds 的 best-found 数字；uniform
  条件的 parity lattice 精确取到 `2K`；宽 support 上若干值接近。
- **反例：** 窄 support 的 base `1.1/2` 结果否定 support invariance。
- **待验证：** 这些静态数值是否与任何训练后行为同向。既有反例说明不能默认同向。
- **claim 纪律：** `78.2%` 只能写成“相对本次 best-found point 的静态改善比例”，
  不能写成最优比例、频谱利用率、行为 headroom 或方法上限。

### 6.2 对 2026-08-24 M4 窗口的修订判断

三条可核验的发现（表哈希已与 canonical owner 逐字节核对）：

1. **该候选族是近乎零的干预。** base=256 下对 FMRoPE 的 $\max|\Delta z|$：
   phase-isotropy `0.063`、min-eigenvalue `0.044`、pair-volume `0.026`；而
   anchored EVQ-Cosh 是 `0.376`。三个「不同 score」彼此只差 $\le0.038$，
   **比各自与 FMRoPE 的距离还小**——score ablation 在构造上没有分辨力。
   块白化 $r_2$（满值 64，owner 见 §6.1）：FMRoPE `12.89`、三候选 `13.88`–`15.90`、
   EVQ-Cosh `45.72`。这只说明候选在该静态指标上接近 FMRoPE，不能预测训练结果。
2. **两个阶段不是 support×budget 因子实验。** base=256 时
   $\omega_{\min}L=1.19$，isotropy 分数动态范围仅 `12.4×`；base=500 000 时
   $\omega_{\min}L=7.7\times10^{-4}$，动态范围 `2.98e7×`，同一构造的 $\Delta z$
   变成 `0.459`。但 Stage B 同时改变了 base 和每臂 token budget，所以只能视为另一
   个训练 regime，不能从两阶段差异归因 support 或预算。
3. **跨协议符号差不是 noise-floor 估计。** Stage A 的 anchored EVQ-Cosh 与
   151.9M owner 使用相同频率表，却处于不同模型、数据量和训练预算；符号不同说明
   结果具有 regime dependence。没有同协议重复或 null replicate 时，不能从不同
   candidate arm 的摆动估计 harness noise floor。

**结论：标为 `SCREEN_UNRESOLVED`。** 数值是该单 seed、小预算协议的有效结果，但
extended preflight 预先规定 anchored-Cosh reference 在该 regime 失去预期方向时，
不得把 gate failure 解释为 candidate-wide rejection。它既不是成功证据，也不关闭
phase-isotropy 类方法。

### 6.3 下一步（按决策价值排序）

| # | 动作 | 成本 | 为什么 |
| --- | --- | --- | --- |
| **A** | 用已有 owner 对照 phase-chord 与 phase-isotropy 的构造和训练响应，不再发明新静态 score | 0 GPU | 当前唯一近 Pareto 的表是 phase-chord；先解释其差异，避免继续做 score zoo |
| **B** | 冻结现有 phase-chord schedule 后补 **seed 256** | ~47 min（5090，需显式授权） | 最小的独立复现；不再搜索方法或超参数，直接决定能否晋升为三 seed 结果 |
| **C** | 为成熟 checkpoint 设计 partial-head / per-head coordinate migration 的最小 preflight | 0 GPU 设计；运行另行授权 | 现有 hard swap 失败与 per-head 可修复比例共同指向迁移结构，不支持整表静态替换或直接跳 8B |

**方法层立即停止**（这三条是研究议程判断，属持久层；当前投稿的易变队列停止项
在 [`paper-2027/HANDOFF.md`](paper-2027/HANDOFF.md) §6，不在这里重复）：

- 任何新的「共享单表 + 静态 score」候选——§3.4 已九连败，且该类看不见 weights；
- 50M M4 harness 上的 candidate-wide 方法判决；
- 把 2026-08-24 的结果当作对 phase-isotropy 的否决。

### 6.4 对现行 roadmap 的两处异议

1. [`POST_GPU_REFLECTION_..._20260824`](paper-2027/research/attention-aware-retrofit/analysis/POST_GPU_REFLECTION_AND_PROBLEM2_ROADMAP_20260824.md)
   停用了 151.9M harness（「小模型已完成 feasibility 角色」）。但那是仓库里
   **唯一**在 fixed-support 主协议上完成三 seed identification 的装置，也是现有
   phase-chord schedule 最自然的复现层级。冻结 schedule 后补一个 seed 不是新的
   method search；若用户授权，它应保留为最小验证选项。
2. 同一文件把 LoRA/continued adaptation 整体禁为 fallback。禁令的理由成立
   （不能用适配掩盖 hard-swap 失败），但过宽：它同时禁掉了
   [`RETROFIT_AXIS_FALSIFICATION_20260822`](paper-2027/research/attention-aware-retrofit/analysis/RETROFIT_AXIS_FALSIFICATION_20260822.md)
   §5 里唯一带定量余量的存活发现——**同参数量下 per-head 块对角映射达到 0.89 的
   可修复比例，现行 rank-64 共享只有 0.09**。注意 $D^*$ 已被证伪为**排序器**
   （§3.4 第 7 项），所以这是容量陈述而非性能预测。

### 6.5 论文层面（与方法线分开）

现稿核心识别结果成立，按时投。审稿人视角的完整对抗台账在
[`paper-2027/research/README.md`](paper-2027/research/README.md)「Reviewer
objections」，此处只记两条与研究议程耦合的：

- **R2（EVQ-Cosh 在行为轴上处于什么位置）仍然开放。** §6.1 只有静态几何参考，
  不能给行为轴定标，也不应进入正文作为方法 headroom。
- **R3（target-matched 反转）** `+0.460`@8x 且随长度单调放大，目前在
  appendix/discussion 边界。它支持 support 与 allocation 相互作用，不支持把
  allocation 降格成 support 欠配时才存在的修正轴。

## 7. 双机协调（工作电脑 / 家里 PC）

### 7.1 什么在 Git 里，什么不在

| 类别 | 位置 | 跨机可用 |
| --- | --- | --- |
| 稿件、理论、证据 owner、receipt、代码、测试 | Git tracked | ✅ 直接 pull |
| raw GPU rows、checkpoint、缓存、token 语料 | 仓库外 / ignored | ❌ **必须显式搬运** |
| `results/` 下 tracked 的历史 | Git tracked | ✅ |
| `results/` 下本地输出 | ignored | ❌ |
| 1B token FineWeb-Edu 语料 | 已停机实例系统盘 | ❌ 见 HANDOFF §4 |

**规则：** tracked receipt **不能**替代 raw artifact。当前 checkout 缺 artifact
只说明这台机器没有，不能写成实验没跑过。

### 7.2 换机器时的固定动作

```bash
git fetch --all && git status --porcelain && git log --oneline -5
```

先看 branch / upstream / worktree，再动任何东西。所有 Git 变更操作都需要显式
授权，禁令清单由 [`AGENTS.md`](AGENTS.md) §6 拥有，此处不重复。

### 7.3 环境与构建

测试、构建、打包的**可执行命令只有一份**，在 [`README.md`](README.md)
「Build and validate」。这里不重复。约束条款在
[`AGENTS.md`](AGENTS.md) §6。

换机器时只需记住：Python/PyTorch/pytest 一律走 Conda `aidemo`；Blackwell 相关
读 [`RTX5090_BLACKWELL_PROFILE.md`](docs/overview/RTX5090_BLACKWELL_PROFILE.md)。

### 7.4 不要跨机泄漏

tracked 文档里不得出现绝对家目录路径、服务器地址、凭据、私有 checkpoint 路径。
`tests/test_repository_navigation.py` 对根级路由文档强制这一点。
