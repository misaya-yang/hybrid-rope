# NeurIPS 2026 (#11628) → ICLR 2027：逐条对照

**这份文档不进 PDF。** 它记录每个改动对应哪条审稿意见，以及每个数字的 owner 文件。

> **Historical snapshot (2026-08-19).** 本文件保留迁移决策与当时的差异
> 审计，不承担当前稿件状态、section locator、PDF hash 或 action queue。
> 当前状态以 `HANDOFF.md` 为准，claim/evidence 路由以
> `research/README.md` 为准。

审稿结果：Dz6s **4**(conf 3) / 27bE **3**(conf 4) / zWsa **2**(conf 5) + AC `XLtL` metareview。

---

## 1. 审稿意见 → 本版应对

| ID | 意见 | 本版怎么答 | 位置 |
|---|---|---|---|
| `AC.1` `RzWsa.1` `RzWsa.2` | 与 FMRoPE (Oka et al., ICLR 2026) 重合；未引用；缺直接对照 | ① 引用并单列为 Layer 2；② 三层参数化把「移动频段位置」和「频段内部分配」分开；③ exact-range 对照就是与 uniform FMRoPE 网格的直接受控比较；④ dead-channel 归因回 Barbero et al.（Oka 自己也这么归） | §2 + Table 1 + §4.2 + App. E |
| `AC.2` `RDz6s.1` `RzWsa.3` `RzWsa.4` `R27bE.2` `R27bE.5` | 规模太小、benchmark 太弱、缺 1B–7B、缺 RULER | 以自然证据链呈现：432M scarce-channel MLA、750M full-parameter continuation、1.485B from-initialisation PPL 对比；另列 1.485B/8B matched adaptation、2Wiki、RULER 与 causal source use | §1 + §4.3–4.4 + App. F |
| `AC.3` `RDz6s.3` `R27bE.1` `R27bE.4` | surrogate→cosh→operating rule 链条只部分验证；allocation 未与 tuning/parameterization 解耦；要求独立调 τ 和 matched non-cosh schedule | exact-range 三种子控制 + 12 配置 factorial，含 0.75×/1.0×/1.25× cosh 与 deformation-matched exponential；完整表和边界臂在附录 | §4.2 + App. E |
| `R27bE.3` | DAPE 对照混淆了 allocation shape 与 parameterization/optimization effort | **认下并退役**：该行移入附录、按真实身份重新标注、正文明确声明不承担归因 | §4.6 + App. E |
| `RDz6s.2` | matched YaRN scale 不足以定论，tuned Geo+YaRN 可能追平 | 明说：这是同一算子在两种 substrate 上的杠杆差异，不是对 tuned range method 的支配；并明说没跑 (substrate, s) 联合 sweep | §4.1 §4.4 + App. F |
| `AC.4` | 只有清晰的 novelty + 受控对照 + 更强评测才可能改推荐 | 三样都在正文，且四条边界（cosh 非唯一 / τ 非最优 / 不替代 range transport / task-adapted）写在正文而非附录 | §1 §4.6 §5 |

## 2. 结构改动

| | NeurIPS 2026 | ICLR 2027 |
|---|---|---|
| 标题 | EVQ-Cosh: Variational Frequency Allocation for RoPE | **RoPE Has a Spectral Budget** |
| Theory | Cosh surrogate | **full sin/cos subspace geometry、stable-rank identity、low-frequency collapse、exact transplant obstruction** |
| Causal identification | EVQ×YaRN 454M | **151.9M raw-hash-receipted 3-seed exact-range + 50.9M 12-config × 3-seed factorial** |
| Scale and capability | PE-dominant vs DAPE | **432M MLA、750M full continuation、1.485B from-init PPL；另列 1.485B / 8B adaptation 与下游** |
| Construction | Cosh as general answer | **EVQ-Cosh as one closed-form, zero-learned-parameter instance** |
| 退役 | — | PE-dominant learned-parameter 对照 → App. E |
| Related Work | 三轴（operator / inference / allocation-analysis）| 非互斥 intervention levels + learned tables（LeRoPE / AdaRoPE）|
| 正文页数 | 9（NeurIPS）| 9（ICLR submission 上限）|
| 尾部 | NeurIPS Checklist | **AI use / Ethics / Reproducibility statements** |

### 2.1 Dual-submission distinctness audit（2026-08-19）

对比对象是 NeurIPS baseline `paper/main.pdf`
(`fa41499486e53c982bd2afae26fe4f532e02fe61c1b9b92e64299dff37d94772`) 与当前
ICLR PDF `paper-2027/main.pdf`
(`185af7984cd23fce6e118fb40c67f7e5ac9ee6f4d8d8dc9b135d05c63d9a28e8`)。

- 科学主线：旧稿的主贡献是 EVQ-Cosh 构造、操作规则与三类机制实验；
  新稿的主贡献是 full-sin/cos 有限基几何、exact stable-rank identity、
  low-frequency collapse、exact transplant obstruction、fixed-range
  identification、$50$M $2\times2$ co-adaptation 与 $1.485$B/$8$B 证据链。
- 方法与证据边界：EVQ-Cosh 及部分旧实验仍保留，但前者已明确降为
  新理论轴上的一个构造性实例，后者主要作为附录支持，不再承担新稿的
  中心识别主张。
- 机械文本检查（只是 sanity check，不代替科学内容判定）：精确
  $8$-word shingles 在新稿九页主文中的重合率为 $1.02\%$；含引用与附录的
  全文为 $18.14\%$，后者包含保留的构造、实验细节和支持证据。

结论：当前稿是有共同方法根基但主 claim、主理论与主证据均已更换的独立续作，
不是旧稿的文本或结果重包装。若 NeurIPS 接收，ICLR 稿需以第三人称引用该工作并
明确贡献边界；这是引用与定位要求，不是由时间线产生的撤稿要求。

## 3. 两处必改问题的处理

### 3.1 「DAPE 不是真的 DAPE」

论文 Table 4 里标为 DAPE 的那一行，实现是 `free_inv_freq`——32 参数、layer-shared
可学习 inverse-frequency。DAPE (Zheng et al.) 是学习位置**算子**，两者不是一回事。

本版做法（三步，缺一不可）：

1. **正文不再依赖它。** §4.6 明说：任何带学习参数的对照都无法分离 allocation shape
   与 parameterization/optimization effort，所以「不把任何 allocation-shape 归因
   压在这一行上」。归因整体转到 exact-range/M4 的零参数固定 schedule 对照。
2. **按真实身份重新标注。** 表格里改成 "Learned inv-freq (layer-shared), 32 params"，
   并在 caption 和 App. E 里说明它属于 learned-**table** 家族（LeRoPE 那一支），
   不属于 learned positional-**operator** 家族。
3. **摘要 / intro / related work 里删掉所有「优于 DAPE」的表述。** 原 §2 Axis A
   那句 "attains lower seed-42 extrapolation PPL than the DAPE-style learned
   positional-operator baseline" 已整段移除。

一处主动决定：**没有在 PDF 里写「早先草稿标错了」**。这是新投稿，审稿人没见过旧版；
写进去反而像在暗示 dual submission。如果将来 camera-ready 需要一句更正说明，
在 App. E 加一句即可，正文不用动。

### 3.2 「YaRN 也是弱化版的」

仓库里那个实现按 `AGENTS.md` 的说法是 *repository-defined fixed-index smooth-ramp
scaler*。本版做法：

- 全文（正文 + 附录 + 表格 + 图注）里该支统一写作 `\rs{}`（渲染为
  `YaRN-style`）。正文直接使用其 range-composition 含义；附录一次性说明固定
  index 边界与 reference implementation 的差别。重复且已漂移的
  method-comparison 表已从附录和源包删除。
- §4.1 加了专门一段声明它不是 YaRN 参考实现，作用只是「同一算子、同一 scale、
  两种训练期表」的受控比较；并明说没跑 (substrate, s) 联合 sweep。
- App. F 完整说明作用域，含**三条明确不支持的主张**。
- App. F 已按实现补全：固定 index 边界和 smoothstep 在，wavelength-derived
  boundaries 与独立 attention mscale 不在。

## 4. 正文新增数字与 owner 文件

数字来自当前 research owner 或 `../rebuttal/rebuttal_0723/theory_results/`；图中
PPL 只做 owner 明确允许的同协议比值/转录，不跨协议聚合。

| 正文位置 | 数字 | Owner |
|---|---|---|
| §1/§4.2, Fig.1a | 三种子 fixed-range mean +0.026/−0.281/−0.176/−0.146；每个 OOD 长度 3/3 同方向 | `paper-2027/research/EXACT_RANGE_151M_3SEED_RESULT_20260820.{md,json}` |
| §4.2 boundary | 三种子 target-matched mean +0.060/+0.227/+0.460 @512/1K/2K；0/3 Cosh wins | 同上 |
| §4.2, Table 4 | factorial −0.009115 / −0.009879 / −0.012100 / −0.010619；CI；sign-flip p | `M4_EXACT_RANGE_FACTORIAL_RESULT_20260726.md` |
| §4.6 | 1.0× 只赢 4/12，1.25× 赢 6，0.75× 赢 2；边界臂 | 同上（从 12 条 structural rows 重算） |
| §4.6 | cosh − exponential +0.000740，p=0.836 | 同上 |
| §4.3, Table 5 | OLMo AR exact 0/100 vs 69/100 & 67/100；NLL 2.235/3.735/4.851 → 2.548/2.703/2.925 | `OLMO2_1B_4K_ONLY_ROUTING_CONVERSION_20260726.md` |
| §4.3, App. F | fresh long-gap 8K：0/100 vs 49/100, 48/100 | `OLMO2_FRESH_ALL_LONG_GAP_N100_20260727.md` |
| §4.3, Table 6 | Q/K-only RULER 72.19/2.02/0.38 vs 42.44/31.63/5.03；2Wiki exact 22.0/0/0 vs 21.5/17.5/4.0 | `OLMO2_1B_SELECTIVE_QK_PHASE_ADAPTATION_20260729.md` |
| §4.3, Table 5 | scratch ΔNLL +0.0381/−0.0437/−0.1351；122/128、126/128 | `OLMO2_1B_RELEASED_ROPE_BASELINE_20260725.md` |
| §4.3, Table 5 | 8B ΔNLL +0.390/−1.510/−2.048；hit@16 18.75→64.06；gold-block deletion −0.0095/+1.5055 | `EVQ_8B_ADAPTATION_EVIDENCE_20260724.md` |
| §4.3, Table 6 | 8B RULER 94.44/0.295 vs 77.60/14.03；32K 全零 | `LLAMA8B_MATCHED_RULER_MIX_20260726.md` |
| §4.6 | 独立 sweep 选中 τ=5，规则值 5.657 差 0.0119 | `EXPERIMENT_REPORT_20260724.md` §2 / `PHASE16_99RUN_RAW_REANALYSIS_20260724.md` |
| §2 | LeRoPE：Fixed LeRoPE 63.6% vs p-RoPE 10.4%；52M–2.5B non-geometric profiles | LeRoPE arXiv:2607.10134 |
| §3.5 | τ=4 波长搬运 707→9.7 / 4000→76 / 34000→1720；τ=2 密度 1.71×…0.55× | 同上 §7.0 |
| App. E | 455.3 / 477.7 / 437.9±12.2 / 513.7 / 333.7 | 提交稿 Table 4 + `docs/exp/2026-02-24_128tok_baseline_report.md` §4 |

## 5. 沿用了投稿版的 owner ledger 里那几条禁令

写作时遵守的边界（来自 `AGENTS.md` §1.5 与 `REVIEWER_USABLE_EVIDENCE_LEDGER`）：

- 不称使用了官方 FMRoPE 实现（不存在）；exact-range 对照写作 uniform-in-log 网格对照。
- 不把 task-family adaptation 说成 unseen-task transfer——正文 §4.3 有专门一段。
- scratch 对照写作 same-initialization / same-scientific-recipe，**不写 bitwise paired**。
- exact-range effect size 归 raw-hash-receipted 三训练种子 owner；M4 只负责跨配置
  和替代 analytic shape 的方向，不做跨协议效应量调和。
- full-string+EOS 与 strict first-number 是不同实验，附录里分开写，不并排。
- 未把 video-DiT / progressive / 750M 从 supporting 升级。

## 6. 尚未处理、值得下一轮做的

1. **EVQ 闭式表 vs Fixed-LeRoPE 学出表** 在同一 OOD 基准上的直接对照。§5 已把它写成
   下一步最有信息量的实验，但没跑。
2. `E-EXACT-RANGE-3S` 已由 2026-08-20 raw-hash-receipted owner 取代并进入正文；
   `E-HELDOUT` 仍未完成独立 owner promotion，继续不进入公开 claim。
3. OLMo future-dated filename 已由 owner 解释为 run label；正文使用的是带实际日期
   和 raw hashes 的 owner。
4. M4 curated JSON 的 summary 字段与逐配置明细存在内部漂移：12 条
   `by_structural_config` 重算为 `0.75×/1.0×/1.25× = 2/4/6`，但
   `best_cosh_multiplier_counts` summary 写成 `1/1/1`；逐行 regret 均值重算为
   `0.007366`，Markdown owner 写 `0.011440`。公开稿只使用可逐行重算的 `2/4/6`
   并删除 regret 数字；现有 evidence validator 尚未覆盖这两个 summary 一致性检查。

---

## 7. 独立审计（对抗性）修掉的错误

一个独立 agent 拿正文每个数字去对 `theory_results/` 的 owner 文件。以下是它确认
为**错误**、已修的部分：

| # | 错在哪 | 原文 | 改成 |
|---|---|---|---|
| 1 | **arm identity** | 摘要/intro/discussion 写「τ 规则被 1.25× 在 **10/12** 配置上打败」 | 10/12 是 **1.25× vs Geo**，不是 vs 规则。规则 vs 三个乘数的对比是 **4/12**（1.25× 赢 6、0.75× 赢 2）。三处全改为 "best member of its own family in only 4/12"，并在 §4.6 显式说明这两个计数不是一回事，表头也从 "Wins" 改成 "Beats Geo" |
| 2 | **量级** | 「effect size 比 151.9M 低**两个数量级**」 | 实际 32–48×。改为 "roughly $40\times$ smaller"，与 handover §4 的「40×」一致 |
| 3 | **假陈述** | evidence-tier caption「摘要和 intro 的任何主张都不压在单 seed 行上」 | 假的——摘要的 RULER 21.3%/0.08% 和 AR exact 0/100 都是单 seed。改为「识别结果全程三 seed；成熟模型行按构造是 1–2 seed，引用处都标了 seed 数」 |
| 4 | **不该报的数** | RULER 表里 LLaMA Native 32K 报了 `0` | owner 明写「a 13-task Native-LoRA 32K macro **must not be reported**」（3 个 shard 没启动）。表里改为 `n/r` 并在 caption 说明 |
| 5 | **缺语料声明** | M4 factorial 没写用的什么数据 | owner: WikiText-2 raw stream，确定性重复。已写进 App. E 的 scope limit，并说明与 151.9M 控制、from-scratch 跑的语料**不同** |

以及四条**过度声明**，已改为源文件允许的强度：

| # | 原来的说法 | 问题 | 现在 |
|---|---|---|---|
| A | 「RULER 4K 的差距是 retrofit adaptation mismatch，是两个效应不是一个」 | 唯一支撑是 from-scratch 的 **NLL** 行，其 owner 明写「It is not yet a capability result … no RULER score has been produced」 | 改为：from-scratch 行**只**给出 LM endpoint 上的窗口内代价上界；retrofit 解释标为 hypothesis，并明说该 checkpoint 没有 RULER 分数 |
| B | 窗口内代价只报 4K 的 `+0.038` | 同一张表 2K 是 `+0.0724`，且只有 3/128 文档 favour EVQ | 2K 数字已加进正文、表 5 和 App. F |
| C | 「baseline 是 FMRoPE 的 uniform-in-log grid」 | handover §1.4：不能暗示用了官方实现（不存在） | 改为 "a paper-faithful reimplementation of the rule specified in §6.1"，并在 App. E 明说没有官方实现可用 |
| D | MLA 的 117.9 那一支曾标成 `Geo+RAMP` | owner 对应的是 **MLA wavelength-blend operator** | 统一为 `MLA wavelength-blend operator`，并与全文的 `\rs{}` / `YaRN-style` 身份分开 |

另外统一使用 `pre-specified before results were inspected`，不再使用暗示外部登记的
措辞。

审计确认**没问题**的部分：所有 exact-range / factorial / OLMo / LLaMA / LeRoPE /
波长搬运数字逐个对得上；base 覆盖没有夸大；98/100 与 69/100 两类 endpoint
全文没有被混用或并排呈现为 seed 方差。
