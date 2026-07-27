# Rebuttal 策略审阅与优化建议 — Submission 11628

审阅日期：2026-07-27
审阅对象：`00_` / `01_` / `02_` + `theory_results/` 两个 ledger
定位：**外部视角的策略批评**，不新增证据、不修改任何 owner、不授权新实验

---

## 0. 总体判断

当前 package 的**证据纪律是优秀的**：concern ID 全覆盖、evidence tier 分层、
negative ledger 与 triggered disclosure 规则完整、claim boundary 明确。
没有任何一个 retained concern 处于"静默未答"状态。

但它目前是一份**审计文档**，还不是一份**说服文档**。主要问题是：

1. 叙事顺序与 AC 自己声明的优先级不一致；
2. 字数预算只用了 14–23%，最便宜的增益全部没吃到；
3. 三处"负面边界"其实是同一个可以正面解释的机制信号，现在被拆散成三个弱点；
4. 有两处**内部自相矛盾**会在 discussion 阶段被抓住；
5. 缺 manuscript revision 承诺清单和 discussion 阶段预案。

下面按优先级排列。P0 = 发送前必须改；P1 = 强烈建议；P2 = 有余力再做。

---

## 1. P0 — 必须修正的内部矛盾与风险点

### 1.1 `complementarity` 的对象错了（内部矛盾）

`02_` §3.2 写"Dual result meets **complementarity** criterion"，
但 negative ledger `G-FMR-DEPLOY` 明确禁止："naive EVQ+FMR does not show
stable additivity"，且禁止 empirical orthogonality / additive synergy 主张。

zWsa 的原话是 "clear advantages **or** complementarity over FMRoPE"。
如果在给 zWsa 的回复里用 complementarity 这个词描述 FMRoPE，
一个 confidence-5 的 reviewer 只需追问一句
"那请给出 EVQ+FMRoPE vs FMRoPE 的数字"，整段就崩。

**建议改法：**

- FMRoPE 段落只主张 **advantage under exact-range control**
  （这是我们真有的：只改 30 个 interior frequency，Cosh 赢 32/32、27/32、22/32 anchors）；
- 把 `complementarity` 这个词**全部移到 YaRN 段落**
  （submitted Table 3，fixed \(s=8\)，三 seed，PK 61±3% → 100±0%）——
  那才是我们真正有的 complementarity，而且是 submitted 证据；
- 明确写一句："we do not claim additivity with FMRoPE; under
  target-aware retargeting FMRoPE is stronger."

`02_` §3.2 的 Outcome 行需要同步改掉，否则 playbook 与 ledger 长期冲突。

### 1.2 FMRoPE 复现的 fidelity 完全没有声明（最大单点风险）

`MATCHED_RANGE_COSH_500M_S42` 把 baseline 描述为
"uniform FMRoPE exponent grid, **training base 256**"，在 151.9M / \(L=256\)。

zWsa 的 confidence 是 5，且明确说"checked the math/other details carefully"。
整个 AC.1 / RzWsa.1 / RzWsa.2 的答复都压在这一个对照上。
如果对方判定"你复现的不是 FMRoPE"，novelty 段落**全线失效**。

**发送前必须补的一句话（放在数字之前）：**

> 我们的 FMRoPE 臂是**我们自己的复现**：采用 [具体 variant]，
> base/target 按 [具体规则] 选取，declared context = [具体值]。
> 如果这一配置与 Oka et al. 的意图不符，请指出，我们会按作者建议重跑。

主动交出复现细节 + 邀请纠正，比让对方发现要好得多。
这也是把 confidence-5 reviewer 从"对抗"转成"合作"的唯一现实路径。

### 1.3 两个 exact-range 数字放在一起会互相拆台

- seed-42 exact-range：Cosh − FMR = **−0.478 / −0.205 / −0.113**
- M4 exact-range：formula-Cosh − Geo = **−0.009879**（bootstrap CI 跨零），
  `1.25×` Cosh = −0.0121，Cosh − Exp = +0.00074（sign-flip p=0.836）

playbook §3.2 和 §5.3 把两组数字放在**同一段**里。
认真的 reviewer 会立刻算出 ~40× 的量级差，然后问：
"哪一个才是真实的 allocation 效应？"

**建议改法（三选一，推荐 A）：**

- **A（推荐）**：M4 作为 **robustness / 跨 base·head 的方向一致性证据**，
  只报 "10/12 与 9/12 configuration favor 非均匀 allocation"
  这类**符号统计**，不把 −0.0099 当 headline 数字；
  seed-42 保持为 direct FMRoPE control 的 effect-size owner。
- B：两个都报，但补一句量级差的解释（模型规模、训练长度、
  归一化 convention、OOD 加权方式不同）。
- C：只报 M4，放弃 seed-42 的大数字。**不推荐**，因为 seed-42 才是
  唯一直接对 FMRoPE 的对照。

**另外：绝对不要用 formula-Cosh（−0.009879）当 M4 的 headline**，
它的 bootstrap CI 跨零。要用 `1.25×` Cosh / matched exponential，
并把主张写成"allocation 轴可识别"，不是"Cosh 更好"。

### 1.4 LLaMA 8K 那一行现在是自曝短板的写法

playbook 只写 official macro `94.44% vs 77.60%`（Native 赢）。
但 owner 里同一行还有 normalized exact `17.69% vs 21.54%`（**EVQ 赢**）。

只报前者 = 主动递给 reviewer 一句"EVQ 在 in-window 更差"。
两个一起报，就变成一个**指标定义**问题（partial/substring match vs strict exact），
而且完全诚实，数字本来就在 owner 里。

**必改**：任何提到 8K 94.44/77.60 的地方，同句补 17.69/21.54。

---

## 2. P0 — 叙事顺序：按 AC 自己的优先级重排

AC 在 `AC.4` 里给了明确的排序：

> (i) clear technical novelty over FMRoPE →
> (ii) direct controlled comparison →
> (iii) stronger evaluation

但 playbook §2 "Result-first opening" 是从 **OLMo 1.485B NIAH（= AC.2）** 开头的，
novelty 排在第二段之后。

AC pilot 的机制是"用 metareview 聚焦 author response"。
开头第一句没有命中 AC 排第一的 gate，等于浪费了最贵的位置。

**建议：AC-facing 开场重排为**

1. 一句 novelty 定式（\(\omega_i=b^{-\phi_\tau(u_i)}\) vs \(b(T)^{-u_i}\)）+ 承认漏引；
2. 一句 direct control（exact range，只改 30 个 interior frequency）+ 诚实反向边界；
3. 然后才是 1.485B / 8B / RULER / scratch。

reviewer-specific 回复保持现有顺序（§5.1–5.3 的排序是对的）。

---

## 3. P1 — 三个"负面边界"其实是同一个正面机制（最大的未开采增益）

目前被分别当作 boundary 报告的三条：

| 设置 | in-window | long-range |
| --- | --- | --- |
| OLMo scratch step-1000（1.485B） | 4K NLL **+0.0381**（EVQ 更差） | 8K/16K **−0.0437 / −0.1351** |
| OLMo matched CF LoRA（1.485B） | 4K NLL 2.235 → **2.548**（更差） | 8K/16K 3.735→**2.703**、4.851→**2.925** |
| LLaMA-3-8B matched LM | 8K NLL **+0.390**（更差） | 16K/32K **−1.510 / −2.048** |

这是**同一个签名**：allocation 把有限 channel 预算从 in-window 密度
换成 long-range resolution。而这正是 surrogate 目标预测的行为。
它在 50.9M / 1.485B / 8B 三个独立规模、两个模型家族上重复出现。

现在的写法把它拆成三处 "mandatory boundary"，读起来像三次认错。
合并写成一个 cross-scale 机制观察，同时命中：

- `AC.3` / `RDz6s.3`（theory→practice 链条：这是理论的**可证伪预测**，且被证实）；
- `R27bE.1`（finite-\(\tau\) 操作点的实际含义）；
- `AC.2`（"跨规模一致"比任何单点数字更像 generality 证据）。

**这是目前最高杠杆的一条改动，且不需要任何新实验。**
诚实性不变——每个数字都照报，只是换了组织方式和一句解释。

配套：4K/8K 上的退化不要再写成"cost"，写成
"the predicted in-window density trade"，并说明它随
\(L_{\rm target}/L_{\rm train}\) 增大而反转。

---

## 4. P1 — 字数预算：现在只用了 14–23%

实测 copy-ready 段落字符数（NeurIPS 上限 10,000 chars/review）：

| 对象 | 当前字符数 | 使用率 | 剩余 |
| --- | ---: | ---: | ---: |
| `Dz6s` | ~1,400 | 14% | ~8,600 |
| `27bE` | ~2,300 | 23% | ~7,700 |
| `zWsa` | ~1,700 | 17% | ~8,300 |
| AC `XLtL` | ~780 | 8% | ~9,200 |

**这是最便宜的增益。** 建议用剩余空间补四块（按性价比排序）：

1. **manuscript revision 承诺清单**（见 §5，最高性价比）；
2. **一张紧凑数字表**（reviewer 更信表格；也避免长句里数字被漏读）；
3. **逐条复述 reviewer 自己的 score-move 条件 + met / partially met 判定**
   —— 对 zWsa 尤其重要，他明确写了四个条件，逐条对账会让 AC 很容易裁决；
4. **related-work 定位段**（见 §6）。

注意：不是简单加长。建议同时做**结构重排**：
把 boundary 从"每句话后面挂一个 caveat"改成
**正面主张在前 60%，末尾一个明确标注的 "Scope of the new evidence" 段落**。
诚实性完全不变，但 score-2 reviewer 读到的第一印象从
"作者自己承认很有限"变成"作者给了结果，并且清楚知道边界在哪"。

---

## 5. P1 — 缺 manuscript revision 承诺清单

整个 package 没有一句"我们会在论文里改什么"。
AC 在判断 `AC.4` 时，需要知道 rebuttal 里的东西**会不会真的进论文**。
这是零成本、高回报的一块。

建议在每个回复末尾加 5–7 行，例如：

> In the revision we will: (1) cite and discuss Oka et al. in §2, with a
> parameterization comparison table (\(b(T)^{-u_i}\) vs \(b^{-\phi_\tau(u_i)}\));
> (2) add the exact-range control and the three-seed factorial as Appendix E;
> (3) add the 1.485B and 8B results, with their single-seed/task-family limits,
> as Appendix F; (4) restate \(\tau=c(\Pi)d_{\rm eff}/\sqrt{L_{\rm train}}\)
> as an empirically calibrated operating rule in §3.3 and Table 1;
> (5) extend §6 limitations with the 4×-transfer and unseen-task boundaries.

另外要**先核实**：NeurIPS 2026 rebuttal 期是否允许上传修订版 PDF。
如果允许，"已加入 Oka et al. 引用"从口头承诺变成既成事实，
对 `RzWsa.1`（missing citation）是决定性的差别。这条务必查证。

---

## 6. P1 — "dead channels" 只回答了 FMRoPE，没回答"一类先验工作"

AC 的原话是 "FMRoPE **and prior dead-frequency observations**"（复数）。
zWsa 的指控是"related work 调研不足"。只加一篇 Oka 的引用，
回答的是"漏引一篇"，不是"调研不足"。

但事实上论文 §2 已经引了 Barbero et al. 2025（高/低频专化）、
Wang et al. Resonance RoPE（critical frequencies）、HoPE、FoPE、
CARoPE、Clipped RoPE 等。**这个事实现在完全没有被使用。**

**建议加一段（约 600 字符），大意：**

> 我们从未把 dead/ineffective channel 的观察本身作为 novelty 主张；
> 提交版 §2 已经引用了 Barbero et al. (2025)、Resonance RoPE 等对
> channel 不均等性的先验观察。我们承认漏引了 Oka et al.，会补上并直接比较。
> 我们主张的窄贡献是：把**有限训练期 grid allocation** 写成显式变分对象，
> 并给出闭式、零学习参数的 inverse-CDF realization。

这比只补一条引用**更能反驳"调研不足"**，而且是纯事实陈述，无风险。
（措辞要平，不要有"reviewer 没看仔细"的暗示。）

---

## 7. P1 — scratch run 需要预防性框定，否则会反噬

`R27bE.5` 要的是 "larger-scale **pre-specified training run**"。
我们给的是 1.485B / step-1,000 / 2.097B tokens / PPL ≈ 160。三个风险：

1. **PPL 160 = 远未收敛**，reviewer 可能直接说"这不算 training run"；
2. **trainer stack 不同**（HF single-GPU vs AI2 distributed）——
   playbook 把它定性为 "scientific claim boundary, not provenance conflict"，
   这个内部定性是对的，但 confidence-5 reviewer 不会这么看；
3. **4K 上 EVQ 更差**，孤立看像失败。

**建议：**

- 明确标注为 **"matched-initialization early-training probe"**，
  并主动说"we do not offer this as a converged comparison"；
- 明确说明 **pre-registration 事实**：评测行、budget、endpoint
  在开跑前就固定了 —— 这正面回应了 "pre-specified" 这个词；
- 4K 的退化按 §3 的机制框架解释（predicted in-window trade），
  不要孤立报告；
- trainer stack 差异**主动先说**，并说明哪些是严格匹配的
  （initialization / recipe / data-order prefix / counted tokens / eval rows）。

如果 27bE 仍然不接受这条，损失有限——他的 `.5` 还有 base/head 分支（M4）兜底。
但如果不框定就发，可能连带削弱 M4 的可信度。

---

## 8. P1 — 单 seed 问题可以用零成本统计强化

成熟模型证据几乎全是 single seed，27bE 和 zWsa 都点了这一条。
但有一条不需要 GPU 就能补强：

OLMo 8K strict NIAH `0/100 vs 69/100`，第二个独立 EVQ seed `67/100`。
这是 n=100 的二项结果，直接给出 Fisher exact / 二项 CI
（0/100 vs 69/100 的 p 值会小到可以忽略），
并说明"两个独立训练 seed 落在 67–69，跨 seed 波动远小于组间差"。

把"single seed"从**定性弱点**变成**已量化的不确定性**，
成本是几分钟的计算。同理适用于 M4 的 10/12、9/12 符号统计
（已有 bootstrap 和 sign-flip，直接引用即可）。

---

## 9. P1 — `E-OLMO-LONG-GAP` 被一个文件名日期卡住了，应立刻解锁

`OLMO2_FRESH_ALL_LONG_GAP_N100_20260727.md`：
所有 source→generation gap 都超出 4K 训练支撑，Native `0/100`，
两个 EVQ adapter `49/100` 和 `48/100`。

它目前是 `CONDITIONAL`，**唯一原因是文件名日期晚于 audit date**（元数据问题）。

但它回答的是对 NIAH 结果**最强的预期反驳**：
"你只是在训练见过的 gap 范围内插值"。
`G-OLMO-GAP` guardrail 也正是这条的对偶。

**这是一个 10 分钟的 bookkeeping 任务，换一条高价值正面证据。**
在所有"待办"里性价比最高，建议排第一。
（同理检查 `20260728` / `20260731` 两个 owner，但那两条是 negative，
解锁优先级低。）

---

## 10. P2 — 缺 discussion 阶段预案

NeurIPS 2026 有 AC pilot 下的 author–reviewer discussion。
现在的 package 只准备了"第一发"，没有准备第二发。建议补一页内部预案：

**（a）最可能的 3–4 个追问，各准备 ≤400 字符的答案：**

1. "你复现的 FMRoPE 配置不对" → §1.2 的复现声明 + 邀请指定配置；
2. "M4 效应量只有 0.01 NLL，有意义吗" → 符号统计 + 与 seed-42 的量级差解释；
3. "step-1000 不算 training run" → §7 的框定 + M4 base/head 分支兜底；
4. "8K 上 EVQ 更差，说明方法有害" → normalized exact 反例 + in-window trade 机制。

**（b）zWsa 沉默怎么办**（confidence-5 + score 2 的 reviewer 常常不回复）：
需要一条 AC-facing 的补充 comment，逐条对账他自己写的四个 score-move 条件，
让 AC 可以在 reviewer 不参与的情况下自行裁决。

**（c）让步规则**：预先定义哪些点可以直接承认（4× 未解决、
unseen-task 未证明、Cosh 非普遍最优），哪些必须坚持
（allocation 轴的可识别性、参数化差异）。
避免 discussion 中临时决策导致前后不一致。

**（d）champion 策略**：Dz6s 是唯一给 4 的人，且认可 mechanism framing。
目前 §5.1 的目标写的是 "protect the 4 / soft aim 5"。
建议更进一步：给 Dz6s **一段可以直接在 discussion 里复述的话**
（一句 novelty + 一句 controlled comparison + 一句 scale），
让他有现成弹药替我们说话。AC pilot 下 champion 的作用被放大。

---

## 11. P2 — 交付物工程化

1. **目前没有最终 paste 文件在 repo 内**。`02_` §8 指向
   `~/Desktop/EVQ_COSH_OPTIMIZED_REBUTTAL_DRAFTS.md`，在版本控制之外，
   无法审计。建议在 `rebuttal_0723/` 下生成四个最终回复文件
   （`04_RESPONSE_Dz6s.md` / `_27bE.md` / `_zWsa.md` / `_AC.md`），
   每个文件头部写明字符数。
2. **把 send gate 第 11 条自动化**："verify every posted number once against
   the named standalone owner" 目前是人工步骤。建议写一个小脚本，
   从回复文件里抽取所有数字，对照 owner 文件做存在性检查。
   四份回复 + discussion 追加发言，人工核对很容易出错。
3. **`Dz6s` 的 optimized-Geo+YaRN 问题**（`RDz6s.2`）目前只靠"narrow the claim"。
   发送前确认一下：submitted 的 YaRN 是否做过 scale sweep？
   如果做过（哪怕只有 2–3 个 \(s\) 值），报出来比纯口头收窄强得多。
   如果确实没有，就明说"我们没有跑 tuned Geo+YaRN 搜索"，
   并列出需要搜索的空间——比含糊带过好。

---

## 12. 修改优先级汇总

| # | 项目 | 优先级 | 成本 | 章节 |
| ---: | --- | --- | --- | --- |
| 1 | 解锁 `E-OLMO-LONG-GAP` 元数据 | P0 | 10 分钟 | §9 |
| 2 | FMRoPE 复现 fidelity 声明 | P0 | 30 分钟 | §1.2 |
| 3 | `complementarity` 改挂 YaRN，不挂 FMRoPE | P0 | 30 分钟 | §1.1 |
| 4 | LLaMA 8K 补 normalized exact | P0 | 10 分钟 | §1.4 |
| 5 | 两个 exact-range 数字的量级差处理 | P0 | 1 小时 | §1.3 |
| 6 | AC 开场按 novelty→control→eval 重排 | P0 | 1 小时 | §2 |
| 7 | in-window / long-range trade 合并成机制叙事 | P1 | 2 小时 | §3 |
| 8 | manuscript revision 承诺清单 ×4 | P1 | 1 小时 | §5 |
| 9 | 核实是否可上传修订 PDF | P1 | 查证 | §5 |
| 10 | related-work 定位段（Barbero 等已引） | P1 | 30 分钟 | §6 |
| 11 | scratch run 预防性框定 | P1 | 1 小时 | §7 |
| 12 | NIAH 二项 CI / Fisher exact | P1 | 30 分钟 | §8 |
| 13 | 用满字数 + boundary 结构重排 | P1 | 3 小时 | §4 |
| 14 | discussion 阶段预案 | P2 | 2 小时 | §10 |
| 15 | 最终 paste 文件入库 + 数字校验脚本 | P2 | 2 小时 | §11 |

**不需要任何新 GPU 实验。** 全部为写作、组织与元数据工作。

---

## 13. 一句话结论

证据侧已经 sendable，**风险不在"证据不够"，在"叙事组织"**：
顺序没对齐 AC 的优先级、字数只用了五分之一、
三条同源的机制信号被拆成三次认错、
两处内部措辞矛盾会在 discussion 被抓。
上面 §12 的 P0+P1 全部不需要新实验，
是当前投入产出比最高的一轮修改。
