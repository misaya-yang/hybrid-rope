# INDEX — 理论、证据、代码与下一步

- **最后更新：** 2026-09-02
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

## 0. 当前两天研究时间线

### 2026-09-02 — 当前决定

- **理论对象收敛：** mature checkpoint 中的有效对象是 learned rotary
  subspace 与 frequency/dilation 的**有序配对**，而不是无序频谱
  \(\pi(\omega)\)。OLMo/Qwen 的 same-multiset permutation collapse 建立了
  non-exchangeability；normalized-index 只是当前最有证据的跨 \(K\)
  工程 transport，不是 canonical 或 universal law。
- **zero-training 正证据：** 冻结 pure-\(z\) 静态表已在已测 OLMo/Qwen/Gemma
  协议上改善 long NLL、RULER/NIAH 与远程 source-conditioned answer
  likelihood；table×gain 对照显示自然 NLL 收益主要来自表，不是
  gain-only。这些结果不建立自然生成 QA 或 SOTA。
- **自然文本与 QA 已完成：** Qwen K32 normalized-index 在 32 条 paired
  streams 的 64K NLL 为 `2.630842`，Native 为 `2.754945`，32K PPL
  retention 约 `.9824`；30-row far-evidence QA macro 为
  Native/index/YaRN `.13229/.10174/.11197`，index-minus-Native 区间跨零。
  source-use 改善已进入正确答案 logits，但未稳定转化为 autoregressive
  winner 与正确停止。该链的远端 raw JSON/JSONL 未回收，故只作内部
  决策证据；见
  [`ZERO_TRAINING_TWO_DAY_EXPERIMENT_SUMMARY_20260902`](paper-2027/research/attention-aware-retrofit/results/ZERO_TRAINING_TWO_DAY_EXPERIMENT_SUMMARY_20260902.md)。
- **headwise 最终 scope ladder：** 两标量每头的 allocation+YaRN-range 表在
  Hotpot/2Wiki/Qasper 上得到 `.23576` macro，official YaRN-4 为 `.23817`；
  但其 4K PG-19 PPL retention 只有 `.77138`。从精确 Native 起点学习将
  retention 提到 `1.04624`，却只有 `.02560` Hotpot F1。free head gain
  显著降低训练 loss 但损害完整生成。因此 headwise specialization 是有效轴，
  却未解决 Native--long joint objective；见
  [`HEADWISE_FACTORIZED_Z_AND_SCALE_FLOW_RESULT_20260902`](paper-2027/research/attention-aware-retrofit/results/HEADWISE_FACTORIZED_Z_AND_SCALE_FLOW_RESULT_20260902.md)。
- **理论指导：** continuous scale-flow 可作为 log-frequency 坐标与多尺度问题定义，
  但 ODE/半群本身不缩小搜索空间，当前 s2/s4 也不识别曲率或
  \(M_\theta(x)\)。后续若重开，必须先在 CPU 上识别非平凡的跨尺度/
  off-orbit 预测，或一个跨 task/scale 同向的 short-capability transport 方向；
  不得从新 ramp、cutoff、gain 或 unrestricted per-frequency field 开始。
- **生命周期：** `PURE_Z_LONG_SIGNAL_ESTABLISHED /
  NATURAL_QA_AND_NATIVE_LONG_JOINT_UNSOLVED / NO_SOTA /
  GPU_METHOD_DEVELOPMENT_STOPPED`。当前没有 GPU 方法开发队列；剩余工作是
  证据回收、稿件 claim 取舍、owner-by-owner 核数和 supplement 重建。

### 2026-09-01 — 已完成的前序证据

- **K32 独立 N80 结果：** 新 seed `202609026` 上，physical/index 的
  64K macro 为 `.46625/.46125`，差 `+.0050`、校正区间
  `[-.038125,.048750]`；两者相对 Native 均改善约 `.22`。32K Native
  retention 为 `.859459/.923243`，只有 index 过 `.875` point gate。旧的
  physical long 优势未复现，crossing 判定 `UNRESOLVED`；条件 P3 入口失败，
  不运行 Native-Q/K KL。随后固定 YaRN-s2 在同批 32K/64K 为
  `.53375/.396875`；index 与其 32K 打平，64K 高 `+.064375`，paired 95% CI
  `[.0275,.102516]`。这是 matched baseline completion，不是 untouched SOTA
  holdout。见
  [`K32_PAIRED_CROSSING_CONFIRMATION_RESULT_20260901`](paper-2027/research/attention-aware-retrofit/results/K32_PAIRED_CROSSING_CONFIRMATION_RESULT_20260901.md)。

- **Qwen 同代 s2 panel 完成：** K32 YaRN2 32K/64K 为 `.5625/.4400`；
  K64 C2-s2 为 `.7675/.6325`，YaRN2 为 `.7675/.6950`。两模型 resolver 均
  通过，K64 C2 的 Native point retention `.935976`；K32 physical/index
  paired CI 仍跨零，不升级为坐标优势或 K 因果。见
  [`QWEN_S2_SAME_FAMILY_IDENTIFICATION_RESULT_20260901`](paper-2027/research/attention-aware-retrofit/results/QWEN_S2_SAME_FAMILY_IDENTIFICATION_RESULT_20260901.md)。

- **K128 坐标确认完成：** 冻结 Gemma K128 physical/index s4 表在新 seed
  `202609028` 的 16K N80 为 `.728125/.790000`；index-minus-physical
  `+.061875`，paired 95% CI `[.028109,.096250]`。旧的 index 倾向独立复现，
  因此关闭 physical-`x` 作为跨 K 特权坐标的强主张；index 只作为工程代表进入
  新 seed breadth confirmation，不升级为 universal law。见
  [`K128_COORDINATE_RANKING_CONFIRMATION_RESULT_20260901`](paper-2027/research/attention-aware-retrofit/results/K128_COORDINATE_RANKING_CONFIRMATION_RESULT_20260901.md)。

- **K32 full RULER-13 `CLEAR_ADVANCE`：** seed `202609027` 上，Native/index/
  YaRN 的 32K macro 为 `.547821/.559167/.559423`，64K 为
  `.220513/.514551/.453654`。index 32K retention `1.020711`；64K
  index-minus-YaRN `+.060897`，paired 95% CI `[.027627,.095835]`。QA/VT
  并非统一更优，故推进 work-machine natural NLL/QA，不称 SOTA。见
  [`K32_NORMALIZED_INDEX_FULL13_CONFIRMATION_RESULT_20260901`](paper-2027/research/attention-aware-retrofit/results/K32_NORMALIZED_INDEX_FULL13_CONFIRMATION_RESULT_20260901.md)。

- **历史 work-machine 入口（已完成，不再执行）：** packed-natural 32K/64K
  final-256 三臂 NLL、far-evidence QA、table×gain、source bridge 与两个 readout
  rescue 都已执行；结果由 2026-09-02 两日综合 owner 汇总。原注册见
  [`WORK_MACHINE_NEXT_EXPERIMENT_PLAN_20260901`](paper-2027/research/attention-aware-retrofit/WORK_MACHINE_NEXT_EXPERIMENT_PLAN_20260901.md)。

- **P2 同代识别注册（非结果）：** P1 之后只补 Qwen K32/K64 的固定 YaRN-s2
  与 K64 唯一 C2-s2 缺口；K64 的 physical/index 相同，不能当作 K 趋势证据。
  见 [`QWEN_S2_SAME_FAMILY_IDENTIFICATION_PREFLIGHT_20260901`](paper-2027/research/attention-aware-retrofit/preflights/QWEN_S2_SAME_FAMILY_IDENTIFICATION_PREFLIGHT_20260901.md)。

- **新一轮归因协议：** stock-HF/自定义 Flash 同 token 对照之后，用全新
  自然文本与独立能力双校准识别 Gemma 的 Native 参考长度；校准/确认分离，
  双族不一致则 abstain，禁止由已有 RULER 反推 4K。后续阶段按入口条件推进，
  selector 与 SOTA 扩展仍后置；完成结果见下方 P0/P1 owner，原注册见
  [`COUPLING_NEGATIVE_ATTRIBUTION_PREFLIGHT_20260901`](paper-2027/research/attention-aware-retrofit/preflights/COUPLING_NEGATIVE_ATTRIBUTION_PREFLIGHT_20260901.md)。
- **P0 首轮：** 新鲜同 suffix 自然文本 Native NLL 在 4K/8K 为
  `3.229847/11.427202`；两码 exact 能力探针短端自身未过门，故 v1
  `ABSTAIN`，没有选择 4K 或 2K。只注册一次新鲜单码测量修复，不重评旧分数；见
  [`NATIVE_REFERENCE_LENGTH_CALIBRATION_RESULT_20260901`](paper-2027/research/attention-aware-retrofit/results/NATIVE_REFERENCE_LENGTH_CALIBRATION_RESULT_20260901.md)。
- **P1 条件注册：** P0 独立确认后，以已冻结 reference 直接计算 target/reference
  的 s2，再条件打开 s4；不改 `G`、边界或 `c`，不打开 SOTA sweep。见
  [`REFERENCE_CORRECTED_K128_PREFLIGHT_20260901`](paper-2027/research/attention-aware-retrofit/preflights/REFERENCE_CORRECTED_K128_PREFLIGHT_20260901.md)。
- **P0 独立确认：** 单码替代探针的 128 个新样本在 4K/8K 为 `127/0`；64 篇新
  文档 NLL 为 `3.196320/11.158831`，四个门均通过，冻结本协议 `L_ref=4096`。
  [`confirmed reference receipt`](paper-2027/research/attention-aware-retrofit/evidence/GEMMA_NATIVE_REFERENCE_CONFIRMED_20260901.json)
  只允许按 target/reference 构表，不等于更改训练长度或确认普适性。
- **Reference-correct K128 s2：** 同一 physical 表在新 RULER 4K/8K 为
  `.9650/.8600`，Native 为 `.8700/0`；独立自然 4K 只增加 `.025083` NLL，
  8K 从 `10.548809` 降至 `3.163365`。physical/index 的 8K 差 `.0050`，
  bootstrap 区间跨零；不识别坐标优越性，条件打开同一 frozen law 的 s4。
  见 [`REFERENCE_CORRECTED_K128_RESULT_20260901`](paper-2027/research/attention-aware-retrofit/results/REFERENCE_CORRECTED_K128_RESULT_20260901.md)。
- **Reference-correct K128 s4：** 同一最大 16K physical 表在 4K/8K/16K 为
  `.8775/.7975/.7250`，4K PPL retention `.927966`；同批 16K 的旧 reference
  表仍为零。index 为 `.9450/.8650/.7950`；三长度校正后的区间跨零，不称
  physical 优越、K 因果或 SOTA。支持 joint reference/request-scale 恢复，
  不是 reference-only 因果识别；完整曲线与 raw owners 见同一
  [`P1 result`](paper-2027/research/attention-aware-retrofit/results/REFERENCE_CORRECTED_K128_RESULT_20260901.md)。
- **当前结果：** `s4` scale-consistent exponent table
  `omega'_i=omega_i s^{-m_i}`，配合只由 1x PG-19 retention 选出的 `c=0.074`
  gain，通过 1x PPL/五任务双门，并在同一静态表上改善 2x/4x NLL 与自然任务；见
  [`SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831`](paper-2027/research/attention-aware-retrofit/results/SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md)。
- **因果分叉：** 同一 dilation multiset 的 endpoint-preserving permutation 在
  1x/2x/4x capability 上崩溃；Haar/MaxEnt ordered distribution 在 16K 为零。
  后续保持最终 frequency multiset、端点与 gain 的 slot permutation 令 1x NLL 从
  `3.10423` 升至 `6.86493`，建立 frozen readout 的 slot–frequency
  non-exchangeability；不建立当前 law 的唯一性或最优性。
- **证据边界：** 单 checkpoint、zero training、同表无 routing；4x natural macro
  仍略低于 YaRN，16K core-4 低于 arithmetic `s4`。不称 SOTA、通用最优或跨模型规律。
- **跨 checkpoint：** 同一 construction 无 Qwen 调参地得到 64K/128K core-4
  `0.7000/0.5875`，高于 Native、YaRN 与旧 corrected-derived；这是 long capability
  transfer，尚无 Qwen 1x PPL retention owner。保持 Qwen 最终 frequency multiset
  只换 slot 后 64K 四任务全为零，non-exchangeability 跨 checkpoint 成立。
- **weights×table：** Native-4K Q/K-LoRA 与 original weights 下，log-s4 都满足同权重
  1x retention，并把 2x/4x NLL 从约 7.1--7.2 降到约 2.9--3.1；未观察到
  readout-induced sign reversal，但单一 adapter 不证明 universal independence。
- **低维 coupling：** 已评测的 OLMo→Qwen 64 点 transport 实际在
  `u_i=L(omega_i-omega_{i+1})/(2pi)` 上线性插值；令
  `x_i=ln u_i` 后，仅用 OLMo movement 拟合的 2 参数 clipped-affine `G_4(x)`
  以 MAE `0.001223` 重建 OLMo，并以 MAE `0.002174` 重建该 Qwen transport。
  这是 CPU 几何压缩，不是 LM 结果；见
  [`CPU_LOW_DIM_COUPLING_LAW_20260901`](paper-2027/research/attention-aware-retrofit/results/CPU_LOW_DIM_COUPLING_LAW_20260901.md)。
- **低维 GPU 判决：** 冻结两参数 `G_4(x)` 在 Qwen 64K/128K core-4 得到
  `0.6775/0.5725`，保留或超过 64 点 transport；但 OLMo 1x PPL retention
  为 `0.870971`、Qwen 32K core-4 retention 为 `0.868902`，均略低于
  `0.875`。因此它支持低维 long coupling，不晋级为当前 deployable
  single-table law；见
  [`LOW_DIM_COUPLING_GPU_RESULT_20260901`](paper-2027/research/attention-aware-retrofit/results/LOW_DIM_COUPLING_GPU_RESULT_20260901.md)。
- **K32 finite-grid 判决：** K32 的 transition resolution 降至
  `eta=0.86135`，但 exact cell-average 只改两个 kink cells，不能生成 OLMo
  fast-side shoulder，故不晋级 GPU。K32 64K 上 frozen physical-`x` / index /
  monotone-self / Native 为 `0.4350/0.4275/0.3900/0.2775`，而 physical-`x`
  在 32K retention 仅 `0.7915`。matched s2 后，physical `x` 为
  `.5225/.5050`、index 为 `.5775/.4375` @32K/64K：scale mismatch 不是
  Native 失败的充分原因；但后续独立 N80 未复现 physical 的 long 优势，故这里的
  pilot crossing 只保留为历史点估计。见
  [`K32_FINITE_K_COUPLING_ANALYSIS_20260901`](paper-2027/research/attention-aware-retrofit/results/K32_FINITE_K_COUPLING_ANALYSIS_20260901.md)。
- **s2 zero-refit confirmation：** 同一 K64 `G(x)` 边界从 s4 零重拟合降到
  s2 后，OLMo 1x PPL/five-task retention 为 `0.98399/1.04122`；2x PG-19
  NLL、六任务 macro、core-4 为 `2.97043/0.26034/0.5150`，均保持可用。
  这支持 OLMo 上 s4→s2 的行为尺度一致性，不等于 arbitrary-s 或跨模型定理。
- **K128 holdout：** 两个 hash-bound Gemma-1 K128 instruction artifacts 上，
  Native core-4 在 8K/16K 均为零；Gemma-1.1 Native 4K 为 `.9050`，physical
  table 在无 gain 时恢复 8K 到 `.8350`，证明模型、任务与 table effect 可解析；
  但全部 16K 仍为零，且 physical 与 index 差异小于 `.05`。gain-only 与
  loader-path controls 均被排除。因此记为
  `SCREEN_UNRESOLVED / LONG_NEGATIVE`：不确认 cross-K universality，也不能因
  K 与 checkpoint 共变而归因于 K。P3 CPU 门失败，s8 stretch 未打开；见
  [`FROZEN_2D_COUPLING_TRANSPORT_RESULT_20260901`](paper-2027/research/attention-aware-retrofit/results/FROZEN_2D_COUPLING_TRANSPORT_RESULT_20260901.md)。
- **下一通用性假设（未验证）：** 不强迫单一 coordinate 赢所有 checkpoint；冻结
  `{Native, physical-x, index}`，仅用预注册 Native calibration 离线选择或 abstain，
  最终仍输出一张 static KV-safe table。另须区分 `L_config/L_train/L_eff`；`L_eff`
  只能由 long holdout 前的 natural+capability 双 calibration 冻结，不能按本轮 RULER
  outcome 选择。该条是后续 protocol，不是完成方法或 SOTA claim。

### 2026-08-31 — 已被次日取代

- 当日收敛的 deterministic non-affine `f` 问题保留；“端点必须固定”、frequency learning、
  matched LoRA first 均不再是当前方法规则。更早 W0/F1/F2--F4 portfolio 只保留历史 owner。

当前底层研究目录是
[`paper-2027/research/attention-aware-retrofit/`](paper-2027/research/attention-aware-retrofit/)：
其 README 说明当前问题、完成证据、退役路线与子目录职责。`§2--§5` 保留完整历史索引，
[`paper-2027/HANDOFF.md`](paper-2027/HANDOFF.md) 保留易变状态。

## 1. 历史与证据导航边界

§2--§5 是理论、证据、代码与目录的历史检索地图。

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
| 下一代理论状态 | position-resolved、co-adaptation-aware 的缺口，matched-content phase 2x2 与方法进入条件 | [`ATTENTION_AWARE_ALLOCATION_THEORY_STATE_20260826`](paper-2027/research/ATTENTION_AWARE_ALLOCATION_THEORY_STATE_20260826.md) |

### 2.2 mature-checkpoint 方法理论（内部，尚未全部进入论文）

| 对象 | 当前最强结论 | 边界 / Owner |
| --- | --- | --- |
| ordered rotary coupling | 成熟权重读取的是 learned rotary subspace 与其 frequency/dilation 的有序配对；same-multiset permutation collapse 否定 unordered-spectrum sufficiency | [`SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823`](paper-2027/research/attention-aware-retrofit/results/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md) 与 [`ZERO_TRAINING_TWO_DAY_EXPERIMENT_SUMMARY_20260902`](paper-2027/research/attention-aware-retrofit/results/ZERO_TRAINING_TWO_DAY_EXPERIMENT_SUMMARY_20260902.md)；不建立每个 raw slot 的跨模型固定语义 |
| static pure-\(z\) 能力 | 一张静态表可以显著改善 long NLL、RULER/NIAH 和 source-conditioned answer likelihood | 不稳定改善自然生成 QA；不是统一胜过 YaRN 或 SOTA |
| capability-conversion barrier | 远程 source 已能改变正确答案 likelihood，但未稳定改变 autoregressive top-1、完整答案与 EOS | 是旧 EVQ-LoRA 障碍的 zero-training 复现，不是新的首次发现；9 月 2 日 raw remote owner 未回收 |
| shared-table ceiling | bounded per-head allocation/range 改善完整 Hotpot 与三任务 long Pareto，说明所有 layer/head 共享一表是真实约束 | log-start 仍失败 Native retention，Native-start 仍失败 long QA；[`HEADWISE_FACTORIZED_Z_AND_SCALE_FLOW_RESULT_20260902`](paper-2027/research/attention-aware-retrofit/results/HEADWISE_FACTORIZED_Z_AND_SCALE_FLOW_RESULT_20260902.md) |
| normalized-index transport | 在已测 K32/K64/K128 中是当前最有证据的 frozen engineering rule；physical-\(x\) 没有跨 \(K\) 特权 | 未证明 normalized rank 唯一、canonical、checkpoint-independent 或 K-causal |
| continuous scale-flow | \(x=-\log\omega,\tau=\log S\) 可作为多尺度坐标；当前 `log_s4` 是 constant-flow 特例 | ODE/半群本身不识别方法；s2/s4 不识别曲率，\(M_\theta(x)\) 尚未定义或验证 |

因此当前理论指导不是「再找一条更好的 ramp」，而是：任何后续方法都必须
显式处理 checkpoint-conditioned ordered coupling，并在同一协议中连接 Native-compatible
与 long-capable 两个已观测盆地。新 geometry scalar、boundary、gain 或 movement RMSE
不能单独充当功能距离。

### 2.3 支持性与历史理论推导（非行动队列）

[`three_completions/optimization_notes.md`](paper-2027/research/three_completions/optimization_notes.md)
把三处各自假设均匀先验的松散结构压成**单一可测先验 $\mu$** 的泛函链。
其中部分恒等式和负结果仍有理论价值，但这组推导不拥有 9 月稿件路线或下一实验
优先级。

| 编号 | 内容 | 状态 |
| --- | --- | --- |
| O1 | $L_{\rm rng}$ 由 $\mu$ 导出，不是自由参数；$\omega_0L=2.0772$ | 已验证的支持性推导 |
| O2 | $L_{\rm eff}^J=n\,\mathrm{Var}_{U_n}[G]/\mathrm{Var}_\mu[G]$ | 条件恒等式保留；经验机制解释未成立，见 §3.5 |
| O3 | $\gamma_{\rm eff}$ 闭式；$\tau_*(L)$ **不是幂律**，$p\approx0.85$ 诊断可撤 | 已验证的纠错结论 |
| O4 | 统一 surrogate 泛函 $\mathcal J[\rho]$ 与闭式 EVQ-Cosh-R | surrogate 数学构造；不是部署方法或 LM 排序器 |
| O5 | arcsine 猜想**证伪**；数值自由优化只报告给定 optimizer/restarts 下的 best-found | 负结果保留 |
| O6 | re-adaptation 秩界不可得；线性化单投影幅度比例是前 $r$ 奇异值能量占比的平方根，Q/K 联合预算与 O6′ 仍开放 | 2026-08-30 纠错已写回 owner；非当前优先级 |
| O7 | 头/层异质分配可能有价值 | 2026-09-02 bounded headwise scope ladder 给出正向但 mixed 证据；不建立 Jensen 定理、通用 head law 或 joint-objective 解 |
| O8 | $\Lambda(\mu)$ 的正则性缺口；朴素参与比**不是**上界 | 开放、非当前优先级 |

历史验证脚本保留在
[`three_completions/`](paper-2027/research/three_completions/)；运行环境与输入必须
在使用前现场核验，脚本存在本身不构成当前结果或复现回执。

### 2.4 历史理论（已被取代，只读）

`docs/tau_algor/`（15 篇，2026-03）与 `docs/archive/`（4 篇）是 τ scaling /
habitable zone / softmax transport 的原始推导。**它们是当前理论的前身，不是当前
权威。** 引用时必须标注为历史推导。

关键结论只有两条：$\tau_*$ **不是幂律**（被 O3 解释掉，拟合形式本身错），且
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
| 跨模态 Video-DiT | [`VIDEO_DIT_HEAD_TO_HEAD_SEED42_RESULT_20260826`](paper-2027/research/VIDEO_DIT_HEAD_TO_HEAD_SEED42_RESULT_20260826.md) | 单 seed-42 head-to-head supporting breadth；不提供训练 seed 不确定性 |

### 3.3 成熟 checkpoint retrofit

全部 owner 在 [`attention-aware-retrofit/`](paper-2027/research/attention-aware-retrofit/)，
子目录分工见该目录 README。

| 问题 | Owner | 判决 |
| --- | --- | --- |
| 同 support 的 $z$ 控制 | [`SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823`](paper-2027/research/attention-aware-retrofit/results/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md) | 内部因果案例 |
| 零训练 session / 单静态 profile | [`SESSION_BINARY_S4_REAL_CONTEXT_RESULT_20260823`](paper-2027/research/attention-aware-retrofit/results/SESSION_BINARY_S4_REAL_CONTEXT_RESULT_20260823.md) | 部署/能力证据；2026-08-31 follow-up 记录 cache-safe static-s4 的 fresh RULER-13 长度曲线、D/S/T YaRN-anchored likelihood，以及 local-gap/s8 探索边界 |
| 尺度一致 exponent-space 单表 | [`SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831`](paper-2027/research/attention-aware-retrofit/results/SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md) | 当前零训练实用 owner：s4 同一静态 `omega'=omega*s^(-m)` 表通过 1x 双门并改善 2x/4x；冻结 `G(x)` 的 s2 zero-refit 也通过 1x 并在 2x 可用。单 checkpoint、会改变 sampled support，不替代 fixed-support 因果 owner |
| `G_4(x)` 低维压缩 | [`CPU_LOW_DIM_COUPLING_LAW_20260901`](paper-2027/research/attention-aware-retrofit/results/CPU_LOW_DIM_COUPLING_LAW_20260901.md) | CPU-only：2 参数 clipped-affine 重建 OLMo movement 并 zero-refit 接近 Qwen geometry；GPU 行为由下一独立 owner 承担 |
| `G_4(x)` GPU confirmation | [`LOW_DIM_COUPLING_GPU_RESULT_20260901`](paper-2027/research/attention-aware-retrofit/results/LOW_DIM_COUPLING_GPU_RESULT_20260901.md) | mixed：Qwen 64K/128K long behavior 保留；OLMo PPL 与 Qwen 32K strict Native gate 失败，不是最终 deployable law |
| 2026-09-01—02 两日综合 | [`ZERO_TRAINING_TWO_DAY_EXPERIMENT_SUMMARY_20260902`](paper-2027/research/attention-aware-retrofit/results/ZERO_TRAINING_TWO_DAY_EXPERIMENT_SUMMARY_20260902.md) | 本地 owner 与 9/2 会话回执分层汇总：long NLL/RULER/source-use 正证据；自然生成 QA 重复历史 EVQ-LoRA capability-conversion barrier；9/2 新增 table/gain 归因与 source-to-logit 定位。raw remote owner 尚未导入，不可直接升级为论文证据 |
| bounded headwise factorization | [`HEADWISE_FACTORIZED_Z_AND_SCALE_FLOW_RESULT_20260902`](paper-2027/research/attention-aware-retrofit/results/HEADWISE_FACTORIZED_Z_AND_SCALE_FLOW_RESULT_20260902.md) | 两标量每头改善 long QA Pareto 并接近 YaRN macro，但 log-start 失败 4K retention、Native-start 失败 long QA；free gain 为负，无 SOTA 或 nonlinear-flow 结论 |
| K32 finite-grid / coupling holdout | [`K32_FINITE_K_COUPLING_ANALYSIS_20260901`](paper-2027/research/attention-aware-retrofit/results/K32_FINITE_K_COUPLING_ANALYSIS_20260901.md) | `eta<1` 但 cell-average 假设未获 CPU 支持；matched s2 physical/index 显示 long/Native Pareto crossing，不开启 corrected C2 |
| Frozen 2D matched-s / K128 transport | [`FROZEN_2D_COUPLING_TRANSPORT_RESULT_20260901`](paper-2027/research/attention-aware-retrofit/results/FROZEN_2D_COUPLING_TRANSPORT_RESULT_20260901.md) | K32 scale confound closed；Gemma-1.1 Native 4K `.905`，table-only 8K `.835`，但两个 K128 screen 的16K全零且physical/index未分离。gain/loader controls闭环；P3 rejected，s8 not opened |
| 新数据上的持久性 | [`FRESH_FINEWEB_S4_GENERALIZATION_RESULT_20260824`](paper-2027/research/attention-aware-retrofit/results/FRESH_FINEWEB_S4_GENERALIZATION_RESULT_20260824.md) | 封闭正向确认 |
| 长度条件化 budgeted retrofit | [`LENGTH_CONDITIONED_BUDGETED_RETROFIT_RESULT_20260822`](paper-2027/research/attention-aware-retrofit/results/LENGTH_CONDITIONED_BUDGETED_RETROFIT_RESULT_20260822.md) | **RULER core-4：0.5825@8K / 0.4000@16K，对官方 YaRN 0.5375 / 0.0125；零学习参数** |
| 联合 in-window/外推可行性 | [`EXPERIMENT_REPORT_20260821`](paper-2027/research/attention-aware-retrofit/results/EXPERIMENT_REPORT_20260821.md) | **phase-chord 两 seed Pareto：$+0.0007/-0.161/-0.156/-0.205$**；seed 范围阻止晋升 |
| 跨 owner 决策备忘 | [`POST_GPU_REFLECTION_..._20260824`](paper-2027/research/attention-aware-retrofit/analysis/POST_GPU_REFLECTION_AND_PROBLEM2_ROADMAP_20260824.md) | 决策备忘，**其路线判断见 §6 的修订** |
| 成熟 checkpoint 共适应 oracle | [`COADAPTIVE_ALLOCATION_ORACLE_RESULT_20260825`](paper-2027/research/attention-aware-retrofit/results/COADAPTIVE_ALLOCATION_ORACLE_RESULT_20260825.md) | 内部机制研究；phase 代理协议关闭，matched 自洽下 allocation 边际效应为 4K `+0.00098` / 8K·16K tail `−0.0387`·`−0.0877` |
| 成熟模型零训练机制假设 | [`ZERO_TRAINING_MECHANISM_AND_CEILING_20260826`](paper-2027/research/attention-aware-retrofit/analysis/ZERO_TRAINING_MECHANISM_AND_CEILING_20260826.md) | 内部分析：冻结表是最小干预而非唯一可能形式；相对码与快带保护是设计假设，不是通用定理 |
| 固定 support 剂量响应 | [`ALLOCATION_DOSE_RESPONSE_RESULT_20260826`](paper-2027/research/attention-aware-retrofit/results/ALLOCATION_DOSE_RESPONSE_RESULT_20260826.md) | 128 文档机制结果；解析 Path A 未通过联合 gate，learned direction 显示连续 full/tail 再分配，不是新方法 |
| Native 4K RULER 诊断 | [`NATIVE_4K_RULER_DIAGNOSTIC_RESULT_20260826`](paper-2027/research/attention-aware-retrofit/results/NATIVE_4K_RULER_DIAGNOSTIC_RESULT_20260826.md) | 描述性 core-four；跨长度不同 rows，不能单独判定模型上限或位置失效 |
| Protected-ramp 理论分析与 band-attribution 设计 | [`PROTECTED_RAMP_THEORY_ANALYSIS_20260828`](paper-2027/research/attention-aware-retrofit/analysis/PROTECTED_RAMP_THEORY_ANALYSIS_20260828.md)、[`PROTECTED_RAMP_RIGOROUS_COMPOSITE_20260828`](paper-2027/research/attention-aware-retrofit/analysis/PROTECTED_RAMP_RIGOROUS_COMPOSITE_20260828.md) | 历史内部分析，**未执行**；保护公式扫描已退役。§9 leave-one-band-out 仅是 matched-content bridge 之后的条件式机制设计，不是当前队列、结果或算力授权 |

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
| 12 | target-free continuous-boundary-slope 具体实现 | 同一 harness 的 8K/16K core-4 RULER 均为 `0.0000` | [`ZERO_TRAINING_MECHANISM_AND_CEILING_20260826`](paper-2027/research/attention-aware-retrofit/analysis/ZERO_TRAINING_MECHANISM_AND_CEILING_20260826.md) §2；只关闭该实现，不关闭 target-free 目标 |

**结构性结论：** 第 1–11 项中的静态 selector 路线属于**同一张共享一维表在假定
content model 下的标量泛函**。而 §3.1 的 2×2 显示 LM 结果由 table×weights
**交互**主导（换表后 PPL `7.14→76.20`）。看不见权重的泛函在预测一个非主导项。
再提同类 score 前，必须先说明它如何逃出这一类。第 12 项关闭的是一个具体
position-dependent operator，不把不同 target-free operator 一并判死。

### 3.5 未决（**不是**失败）

| 对象 | 现状 | 为什么未决 |
| --- | --- | --- |
| phase-isotropy / pair-volume / min-eigenvalue | [`PHASE_ISOTROPY_50M_M4_RESULT_20260824`](paper-2027/research/attention-aware-retrofit/results/PHASE_ISOTROPY_50M_M4_RESULT_20260824.md)、[`PHASE_ALLOCATION_M4_EXTENDED_RESULT_20260824`](paper-2027/research/attention-aware-retrofit/results/PHASE_ALLOCATION_M4_EXTENDED_RESULT_20260824.md) 均为 `SCREEN_UNRESOLVED` | 历史 regime evidence；不是当前 candidate rejection 或 action |
| O2 的 $L_{\rm eff}^J$ 机制解释 | 反演出 $\mathrm{sd}_\mu\approx46$–$72$ | 实测注意力距离方差不支持该经验解释；历史未决，不进入当前队列 |
| Native--long basin bridge | bounded headwise/per-layer scope ladder 已完成，log-start 与 Native-start 落在相反两侧 | 尚无一个 arm 同时通过 Native retention 与自然 long QA；这是当前唯一核心方法缺口，但没有已授权 action |
| per-head adapter 秩分配 | 同参数量下 0.89 vs 0.09 可修复比例 | 历史未运行诊断；不是 §6 的 matched low-rank control，也无当前 action |

---

## 4. 代码索引

| 需要 | 位置 | 规则 |
| --- | --- | --- |
| **频率表实现权威** | [`scripts/lib/rope/`](scripts/lib/rope/) | 固定端点是历史 causal control，不是未来方法硬约束；当前没有新方法实现 |
| 主实验链 runner | [`scripts/core_text_phases/`](scripts/core_text_phases/) | canonical runner；新主实验放这里 |
| MaxEnt dilation table | [`scripts/lib/rope/schedules.py`](scripts/lib/rope/schedules.py)、[`scripts/analysis/maxent_dilation_allocation.py`](scripts/analysis/maxent_dilation_allocation.py) | deterministic CPU-verified construction；不是 LM result；GPU 未授权 |
| Native-support `z`-adaptation 历史实现输入 | [`rebuttal/rebuttal_0723/experiments/olmo2_phase_chord_lora_retrofit_5090/`](rebuttal/rebuttal_0723/experiments/olmo2_phase_chord_lora_retrofit_5090/) | 可审计的 Q/K-LoRA、静态表与 receipt building blocks；不是当前 runner，复用前必须按 §6 重新冻结纯-`z` 身份与 matched control |
| **退役 zero-training tournament tooling** | `scripts/analysis/freeze_success_first_portfolio.py`、`scripts/analysis/build_candidate_manifest.py`、`scripts/eval/eval_zero_training_tournament.py`、`scripts/eval/develop_zero_training_family.py`、`scripts/eval/zero_training_selection.py`、`scripts/data/build_success_first_splits.py`、`scripts/eval/run_zero_training_tournament_5090.sh` | 历史 W0/F1/success-first 代码；无当前 action queue、无 README 运行命令、不得作为新实验入口 |
| 可复用 CPU 诊断 | [`scripts/analysis/`](scripts/analysis/) | 不自动成为 paper claim |
| 低维 coupling 冻结与 holdout | [`scripts/analysis/compile_low_dim_coupling_law.py`](scripts/analysis/compile_low_dim_coupling_law.py) | CPU-only；只用 OLMo 拟合，冻结后读取 Qwen geometry，生成候选表/残差/哈希；不运行 LM |
| Frozen cross-K transport | [`scripts/analysis/export_frozen_coupling_transport.py`](scripts/analysis/export_frozen_coupling_transport.py)、[`scripts/eval/run_frozen_coupling_k_transport.sh`](scripts/eval/run_frozen_coupling_k_transport.sh) | 从 runtime Native tensor 导出 physical/index/wrong-c 静态表；launch fail-closed 绑定 config/K/weights/data/Native/table hashes；结果 owner 为 2026-09-01 K128 mixed report |
| Frozen index confirmation | [`scripts/eval/run_k128_coordinate_confirmation.sh`](scripts/eval/run_k128_coordinate_confirmation.sh)、[`scripts/analysis/summarize_gemma_k128_coordinate_confirmation.py`](scripts/analysis/summarize_gemma_k128_coordinate_confirmation.py)、[`scripts/eval/run_k32_normalized_index_full13_confirmation.sh`](scripts/eval/run_k32_normalized_index_full13_confirmation.sh)、[`scripts/analysis/summarize_k32_full13_confirmation.py`](scripts/analysis/summarize_k32_full13_confirmation.py) | K128 coordinate ranking and K32 new-seed full-RULER owners; fixed tables only, no search |
| Work-machine natural confirmation | [`scripts/eval/run_qwen_k32_experiment_tree.sh`](scripts/eval/run_qwen_k32_experiment_tree.sh)、[`scripts/eval/run_qwen_k32_packed_nll_staged.sh`](scripts/eval/run_qwen_k32_packed_nll_staged.sh)、[`scripts/eval/run_qwen_k32_far_evidence_qa.sh`](scripts/eval/run_qwen_k32_far_evidence_qa.sh) | NLL：32 条 paired packed streams、Native/index/YaRN；通过后进入 30-row far-evidence QA；总费用硬门 10 RMB。2026-09-02 remote run 已在会话中报告，raw receipts 尚未导入；两日 summary 只作内部 synthesis，不替代结果 owner |
| 关闭的条件 Native-Q/K 机制资产 | [`scripts/analysis/native_attention_kl.py`](scripts/analysis/native_attention_kl.py)、[`scripts/data/prepare_native_qk_calibration.py`](scripts/data/prepare_native_qk_calibration.py)、[`input receipt`](paper-2027/research/attention-aware-retrofit/evidence/NATIVE_QK_CALIBRATION_DATA_RECEIPT_20260901.json)、[`conditional preflight`](paper-2027/research/attention-aware-retrofit/preflights/NATIVE_QK_FINITE_KL_PREFLIGHT_20260901.md) | 纯 CPU 数学核与已冻结的 8/8 Native-only 输入；K32 `CONFIRMED_CROSSING` 入口失败，故无模型结果。有限 attention KL 不是已验证 predictor、selector 或新 profile |
| 注意力需求测量 | `scripts/analysis/attention_phase_demand.py` | 含 `layerwise_plan()` → `per_layer_inv_freq` |
| 全 RoPE 碰撞审计 | `scripts/analysis/full_rope_collision_audit.py` | §2.1 的数值 owner |
| **第三轴静态 $r_2$ 搜索诊断** | [`scripts/analysis/third_axis_ceiling.py`](scripts/analysis/third_axis_ceiling.py) | §6.1 数值的可复现脚本；纯 CPU；报告 optimizer 的 best-found value，不是全局或行为上限 |
| 有限 $K$ Cosh surrogate-regret 审计 | [`scripts/analysis/finite_k_cosh_regret_audit.py`](scripts/analysis/finite_k_cosh_regret_audit.py) | equal-mass quantile histogram 的纯 CPU 数值证书；只验证所述 surrogate 值的 $K^{-2}$ 系数，不是 $r_2$、LM loss 或 table selector |
| signed-lag / gap / $k$-way identities | [`scripts/analysis/verify_signed_lag_kway_gap.py`](scripts/analysis/verify_signed_lag_kway_gap.py) | 纯 CPU 内部诊断；验证解析恒等式与静态反例，不是 checkpoint 结果或论文 claim owner |
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
| `paper-2027/research/external-reviews/` | 8 月外部模型审计快照 | frozen history；不进当前优先级或 evidence routing |
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

### 历史审计入口

| 需要追溯 | 入口 | 边界 |
| --- | --- | --- |
| NeurIPS 官方 review / AC metareview | [`00_REVIEWER_SCORES_AND_AC_METAREVIEW.md`](rebuttal/rebuttal_0723/00_REVIEWER_SCORES_AND_AC_METAREVIEW.md) | author-supplied official-history capture；不是当前 ICLR 评分预测或行动队列 |
| NeurIPS 历史回复策略 | [`01_REBUTTAL_PLAYBOOK.md`](rebuttal/rebuttal_0723/01_REBUTTAL_PLAYBOOK.md) | historical response artifact；不自动迁移任务 |
| NeurIPS-era 结果 provenance | [`RESULT_PROVENANCE_MANIFEST.md`](docs/overview/RESULT_PROVENANCE_MANIFEST.md) | historical registry；当前 ICLR claim 走 §3 canonical owner |
| paper experiment code view | [`paper_experiments/MANIFEST.json`](paper_experiments/MANIFEST.json) | schema-v1 code/symlink snapshot；不是 run、result 或 readiness receipt |

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
| 可复用分析代码 | `scripts/analysis/` |
| 新主实验 runner | `scripts/core_text_phases/` |

**不要**再创建：第二份根级 Agent、第二份 handoff、第二份索引/REPO_MAP、第二份
provenance manifest、第二个 rebuttal control room、额外的投稿 PDF 入口。

---

## 6. 9 月投稿与后续研究优先级

> 本节只拥有持久的决策顺序，不拥有实时进度、投稿回执或计算授权。9 月里程碑与
> 当前步骤见 [`paper-2027/HANDOFF.md`](paper-2027/HANDOFF.md)；本轮静态执行范围见
> [`paper-2027/REVISION_BRIEF.md`](paper-2027/REVISION_BRIEF.md)。

### 6.1 当前投稿目标

当前稿已完成 fixed-support allocation 识别、full-pair 静态几何、bounded
EVQ-Cosh construction，以及 frozen / matched-adaptation / from-training 三条分离的
证据路线。9 月迭代从当前 TeX/PDF 出发，不从 8 月外部模型评审、旧 verdict、旧
experiment plan 或旧 line number 继承任务。

投稿前只保留三类高杠杆修改：

1. **novelty 可见性：** 让 fixed-support estimand、FMRoPE 的 control 身份和
   support-policy 条件化的排序反转在 reviewer 路径中各出现一次，不恢复方法胜负叙事；
2. **theory-to-evidence 接口：** 分开 exact identity、bounded surrogate、operating
   prior 与 trained-model evidence，不把静态几何写成 LM 排序器；
3. **证据角色：** frozen、adapted、from-training 各由自己的 owner、endpoint 与
   uncertainty unit 承担，不做跨协议 ledger 或 pooled effect。

当前计划不包含新的 submission compute。任何例外都必须先说明它如何在投稿前改变
reviewer ceiling、现有证据缺什么、精确预算/owner/stop condition，并获得该次运行的
显式授权。

### 6.2 第一研究目标：frozen-checkpoint static pure-`z`

当前目标不是继续证明 `z` 存在，也不是把 frequency learning 改名，而是直接回答：

> 一个 frozen checkpoint 能否只通过一张 static non-affine `z` 表，在预先声明的轻微
> `1x` 代价内，同时改善 `2x` 和 `4x` natural-text NLL，并把同一张表交给 untouched
> downstream 验证？

以 checkpoint 的 Native base 作为坐标约定：

\[
\omega_i=b_{\rm native}^{f(-2i/d)}.
\]

`f` 必须在查看目标 LM 结果前由理论给定，且在 sampled Native grid 上不是
`a+cz`。它可以移动 endpoints；单个 endpoint 漂移不等于全局 scaling，慢端是否恰好达到
某个倍率也不是物理公理。模型 weights 与 `f` 均不训练。任意 loss-calibrated、逐频率学习
或 validation-selected table 属于另一研究问题，不进入第一 zero-training 方法。

当前实用 owner 采用已经冻结的 checkpoint-derived coupling `m_i`，但在 exponent
空间而不是 frequency 空间实现：

\[
\omega_i'=\omega_i s^{-m_i},\qquad
z_i'=z_i-m_i\log_b s.
\]

一张表在同一个模型加载与 KV-cache coordinate system 中覆盖所有长度；Native/long
routing 不参与。Gain 是独立 attention interaction：`c=0.074` 只由 1x PG-19 retention
边界选择，然后与表共同冻结。相同 gain 下，arithmetic frequency interpolation 不通过
1x PPL 门，不能把当前结果归因成 gain-only。

### 6.3 已完成门禁与下一证据缺口

Formal manifest
`74022bf36d444a1735baab72bda0312b9867dd38c9f85ece376049b5f35f66f3`
上的两个 1x ratio 分别为 `0.875302`（PG-19 PPL）和 `0.915103`（五任务 macro），
均通过 0.875 门。冻结后 2x/4x PG-19 NLL 为 `3.083278/3.081946`，六任务 macro
为 `0.307614/0.260055`；完整 RULER-13 在 4K/8K/16K 为
`0.71397/0.66705/0.49859`。

同 gain arithmetic control 在全部 2x/4x natural endpoints 较差，但 RULER-13
为 `0.69731/0.65429/0.50481`：log 改善 4K/8K，16K 小幅反转。故不保留
“log law uniformly improves long capability”的 broad claim。

Qwen long-capability construction transfer与 slot non-exchangeability 均已完成。早期两参数
`G_4(x)` s4 判定保留 long behavior，但在 OLMo/Qwen Native point gate 略低于
`.875`；这关闭的是该 operating point，不是以后任意 reference/scale。后续 s2
zero-refit 在 OLMo 双门通过；P0 又在不读 long outcome 的 Native 双族确认中为精确
Gemma-1.1 artifact 冻结 4K operating reference。同一 G 的 reference-correct s2/s4
表均过 4K RULER/PPL point gate，恢复 8K/16K；旧 reference 表在同批 16K 仍为零。
因此原 K128 阴性不能继续作为 G 或 K128 的通用失败；见 §0 的 P0/P1 owner。

K32 matched s2 pilot point estimates为 physical `.5225/.5050`、index
`.5775/.4375` @32K/64K；补齐 YaRN2 后为 `.5625/.4400`。新 seed N80 三臂确认
得到 physical/index `.496875/.533750` @32K 与 `.466250/.461250` @64K；64K
差 `+.0050` 且校正区间跨零。两张表的 long 改善相同量级，index Native cost
更小。同批 YaRN 为 `.533750/.396875`，index 的 64K paired 优势为
`+.064375 [.027500,.102516]`，但三臂结果先于 YaRN 补臂已知。故旧 20-row
physical long 排序未复现，conditional Native-Q/K 入口失败；标准 finite-cell
projection 仍未满足 CPU entrance condition，也不因结果复活。

同代 K64 Qwen 的 C2-s2 为 `.7675/.6325`，YaRN2 为 `.7675/.6950` @32K/64K；
C2 Native point retention `.935976` 通过，但 C2/YaRN 之差未解决。K64 上
physical/index 在精确构造中退化为同一控制，不能提供第二个坐标差观测。因此当前
K128 新 seed N80 又给出 index-minus-physical `+.061875`、95% CI
`[.028109,.096250]`。结合 K32 的相同 long/更低 Native cost 与 K64 的构造退化，
当前结论是 **reference-correct、scale-correct 的 frozen profile family 有多模型
可用性，但 physical coordinate privilege 被证据反驳；normalized-index 只是当前
工程代表，因果 K、universality 与 SOTA 仍未识别**。不得反向调整
table/gain/boundary/width，不得拟合 `G(x;K)` 或开启 s8/hierarchical rescue。

新 seed full RULER-13 又给出 index 32K retention `1.020711`，64K 相对 Native
`+.294038 [.250192,.338271]`、相对 YaRN `+.060897 [.027627,.095835]`。
这把 normalized-index 从 core-4 development signal 推进为 breadth-confirmed
engineering candidate。随后 natural gate 已完成：32 条 paired streams 的 64K
index-minus-Native NLL 为 `-0.1241 [-0.1479,-0.1029]`，而 30-row
far-evidence QA index-minus-Native 为 `-.03055 [-.1168,+.0486]`。因此当前可升级的
是 natural long-position NLL 内部证据，不是 task-universal、natural-QA 或 SOTA
claim；且 9 月 2 日 raw owner 未回收，尚不得进入 reviewer-facing 数字。

2026-09-02 bounded headwise scope ladder进一步放松“所有 layer/head 共享同表与 gain”假设。
两标量每头的 allocation+YaRN-range arm在 Hotpot/2Wiki/Qasper 的 macro F1 为
`.23576`，official YaRN-4 为 `.23817`；差 `-.00241 [-.03372,.02885]`。该 arm 的
4K PG-19 PPL retention 仅 `.77138`。精确 Native 起点把 retention 提到 `1.04624`，
但 Hotpot F1 仅 `.02560`。因此 headwise factorization 是有用的新轴，但没有解决
Native--long joint objective；free head gain 与 unrestricted scale-flow 均不由本结果晋级。
完整 owner 与 raw/hash-backed evidence 见
[`HEADWISE_FACTORIZED_Z_AND_SCALE_FLOW_RESULT_20260902`](paper-2027/research/attention-aware-retrofit/results/HEADWISE_FACTORIZED_Z_AND_SCALE_FLOW_RESULT_20260902.md)。

### 6.4 已退役路线

以下对象只保留历史 owner/代码 provenance，不再出现在 README 命令或 current action
queue：W0/F1 success-first tournament、F2--F4 portfolio、`ABSOLUTE/ANCHORED` 选择模式、
protected-ramp/band restoration、local-gap、s8 arithmetic/log scaling、任意 64D
per-head/per-frequency tensor、free/dynamic gain、unrestricted spectral flow，以及把
dilation distribution 单独当充分对象的 MaxEnt 扫描。bounded one/two-scalar-per-head
screen 已有 2026-09-02 mixed owner，不据此重开更大自由度。
它们的结果仍可支持“frozen shock、gain interaction、失败边界”等已完成判断，但不能
选择下一张 OLMo 表。

重新进入主线的最低条件不是换名，而是满足本节 deterministic non-affine `f` 与同表多长度
身份。用 routing 隐藏 `1x` 损伤或用 LM loss 学/搜 `f` 均不属于第一 zero-training 方法；
endpoint movement 本身不是出线条件。

### 6.5 生命周期与反重复

- **Current:** 当前 TeX/PDF、§2--§3 canonical owners、`s4` exponent-space result owner。
- **Next gate:** no active GPU method-development queue. The 2026-09-02 natural/source-use and
  bounded-headwise owners close the current table/gain/readout/factorization round with a mixed
  result and explicit no-SOTA verdict. The first outstanding work is evidence recovery and paper
  adjudication. If method research is separately reopened, it must first identify on CPU either
  a nontrivial off-orbit scale-flow prediction or one source-disjoint, cross-task/scale direction
  that can bridge the Native-compatible and long-capable basins. Another table, gain, curvature,
  boundary, residual or unrestricted per-frequency sweep is not an entrance condition.
  Native-Q/K P3、physical-x privilege、`G(x;K)` 与 post-outcome rescue 均保持关闭。
- **Historical:** zero-training tournament 与 scale-law owner；保留证据，不保留任务。
- **Closed negative:** §3.4 的 owner-backed 条目；不得通过改名恢复。
- **Unresolved:** M4 phase-isotropy 仍是 `SCREEN_UNRESOLVED`，不自动进入方法设计。

### 6.6 静态诊断的保留边界

[`scripts/analysis/third_axis_ceiling.py`](scripts/analysis/third_axis_ceiling.py)
仍是固定 measure/support/optimizer/restarts 下 best-found `r_2` 的复现 owner。这些值：

- 是未知 supremum 的下界，不是全局或行为上限；
- 明确**不建立 support invariance**；
- 可解释静态 basis utilization，不能选择下一张 LM 表；
- 不能把某个比例写成方法利用率或行为 headroom。

uniform 测度下的精确正交格、三角测度、钉住 support 和训练后行为是不同问题。
详细数值与反例由脚本、[`analysis/full_rope_audit/`](analysis/full_rope_audit/) 与 Git
历史保留。

## 7. 双机协调（工作电脑 / 家里 PC）

### 7.1 什么在 Git 里，什么不在

| 类别 | 位置 | 跨机可用 |
| --- | --- | --- |
| 稿件、理论、证据 owner、receipt、代码、测试 | Git tracked | ✅ 可经授权的 Git 同步获取 |
| raw GPU rows、checkpoint、缓存、token 语料 | 仓库外 / ignored | ❌ **必须显式搬运** |
| `results/` 下 tracked 的历史 | Git tracked | ✅ |
| `results/` 下本地输出 | ignored | ❌ |
| 1B token FineWeb-Edu 语料 | machine-local；可用性必须在未来 preflight 现场复核 | ❌ |

**规则：** tracked receipt **不能**替代 raw artifact。当前 checkout 缺 artifact
只说明这台机器没有，不能写成实验没跑过。

### 7.2 换机器时的固定动作

先用只读命令核对 branch / upstream / worktree / HEAD；当前值写入 handoff。`fetch`、
`pull`、切换、提交或推送都需要对应的显式授权，禁令清单由
[`AGENTS.md`](AGENTS.md) §6 拥有，此处不重复。

### 7.3 环境与构建

测试、构建、打包的**可执行命令只有一份**，在 [`README.md`](README.md)
「Build and validate」。这里不重复。约束条款在
[`AGENTS.md`](AGENTS.md) §6。

机器角色不可互换：

- **工作机**拥有 Conda `aidemo`，负责 Python/PyTorch/pytest、打包和最终跨环境
  release validation；
- **低配置个人 PC profile**可负责阅读、文档、规划、LaTeX/Tectonic 本地编译、PDF
  视觉迭代和轻量 static/stdlib 检查；缺少 Conda 是预期状态，不是 repository failure，
  不在此机补装工作机环境、运行模型计算或用本地构建替代最终打包/跨环境回执；
- Blackwell 规则只在现场确认是对应 GPU 机器时才应用，见
  [`RTX5090_BLACKWELL_PROFILE.md`](docs/overview/RTX5090_BLACKWELL_PROFILE.md)。

### 7.4 不要跨机泄漏

tracked 文档里不得出现绝对家目录路径、服务器地址、凭据、私有 checkpoint 路径。
`tests/test_repository_navigation.py` 对根级路由文档强制这一点。
