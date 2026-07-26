# EVQ-Cosh Rebuttal Principles

## 1. 当前任务

当前目标是完成 NeurIPS rebuttal，不是：

- 重新设计整篇论文；
- 开发 EVQ-v2；
- 全面审计仓库；
- 主动寻找新的论文问题；
- 证明 EVQ 是所有 RoPE 方法中的 SOTA。

所有分析、实验和写作必须服务于 AC 明确指出的三个问题：

1. 解释 EVQ 相对 FMRoPE 的技术新颖性；
2. 提供直接、受控的比较和必要消融；
3. 补充更强评测或更大模型证据。

与这三个问题无直接关系的工作，默认不进入 rebuttal。

---

## 2. 核心方法立场

标准几何 RoPE 可写为：

\[
\omega_i=b^{-u_i}.
\]

FMRoPE 的几何网格通过 base 选择或 retarget 改变可用 range：

\[
\omega_i=b(T)^{-u_i}.
\]

EVQ-Cosh 固定 nominal base，改变指数位置：

\[
\omega_i=b^{-\phi_\tau(u_i)}.
\]

NTK-aware、YaRN 与 LongRoPE 类方法还可能按 channel 非均匀地 transport 或
search 既有频谱；不能笼统写成“所有 range 方法只改变一个 scalar”。EVQ 的
区别应按优化对象、作用阶段与构造方式表述：

- 研究动机存在重合；
- 都关注 RoPE 频率利用和长外推；
- FMRoPE/YaRN/LongRoPE 回答 target range selection、transport 或 post-hoc
  adaptation；
- learned-frequency 方法在训练中优化 \(O(K)\) 个频率自由度；
- EVQ 在训练前把有限 grid allocation 写成显式变分对象，并以闭式
  inverse-CDF 固定频率表；
- 因此优化变量、作用阶段、参数空间和构造方法不同。

不得把“FMRoPE 性能更强”推导为“EVQ 没有技术新颖性”。

不得把“高层观察相似”推导为“方法相同”。

EVQ 的技术贡献应限定为：

- 把有限训练期频率 grid allocation 作为明确优化对象；
- 使用变分代理目标得到闭式 inverse-CDF allocation；
- 在固定 nominal base 下产生非几何指数网格；
- 不增加可学习参数。

最窄且最强的新颖性表述是“首次对 standard RoPE 的有限训练期 grid
allocation 给出显式变分构造与闭式、零学习参数的 inverse-CDF
realization”。不得扩大为“首次改变 interior frequency”或“首次闭式非几何
RoPE schedule”。

---

## 3. 理论主张边界

必须严格区分三层证据：

### 3.1 代理目标上的数学结论

Cosh density 的唯一性只针对论文中明确给出的凸代理目标：

\[
C_{\mathrm{app}}.
\]

不得写成：

- Cosh 是所有频率 schedule 的唯一最优；
- Cosh 是真实 Transformer loss 的全局最优；
- Cosh 在所有目标长度上经验最优。

### 3.2 参数选择结论

\[
\tau=\frac{d_{\mathrm{head}}}{\sqrt{L_{\mathrm{train}}}}
\]

是经验平坦盆地中的 deployable default，不是真实模型目标的精确全局最优。

新增 sweep 的作用是界定：

- 在一个直接选择实验中，默认规则落在良好 basin；
- 跨配置证据表明它仍是 fallible prior，不能排除其他 \(\tau\) 或 schedule。

### 3.3 训练模型上的经验结论

训练后 NLL、PPL、retrieval 和其他任务结果是经验验证，不是从代理目标严格推出的定理。

不得把理论、kernel diagnostic 和训练结果混成同一层证明。

---

## 4. 已接受的新增实验事实

后续工作必须接受以下事实，不得反复重新争论：

- 在已登记的多项 raw-schedule 文本协议中，EVQ 的外推 NLL 优于对应未缩放
  Geo；这不是跨模型、跨长度的普遍支配。
- 报告中的 held-out base=1M、\(d_{\mathrm{head}}=128\)、三 seed 结果保持
  外推收益，但专属 raw/per-seed aggregate 仍待 promotion。
- 一个独立 sweep 的最佳 \(\tau\) 接近公式预测；更广重算证明该规则会失败。
- 没有一种静态 schedule 在所有长度上最优。
- EVQ-Cosh 在部分短距离最好。
- matched exponential 在部分中距离最好。
- attention-derived two-band 在 2K–8K 最好，并在 held-out seeds 上保持长距离优势。
- two-band 尚未证明跨模型、跨数据普适。
- target-aware FMRoPE/YaRN 类 range scaling 明显强于 raw EVQ。
- 朴素 EVQ+FMRoPE 没有稳定协同。
- exact-range seeds 42/137/256 已完成：在 sampled extrema、log-span 与
  within-seed training protocol 完全匹配、只改 30 个 interior positions 时，
  fixed-range Cosh-minus-uniform-FMRoPE 三 seed 均值为
  \(-0.3159/-0.1949/-0.1674\)（512/1K/2K），三 seed 全部同方向。
- 两张 exact-range grids 都按目标长度 retarget 后，三个 OOD 长度的三 seed
  均值全部偏向 uniform FMRoPE；这是否定 additive synergy，不是否定
  allocation 的可识别性。
- 上述 exact-range aggregate 当前由作者确认，tracked tree 尚缺完整 retained
  raw aggregate、per-seed contrasts、hashes 与 CI；promotion 前不得增加
  统计显著性措辞。
- EVQ 的适用性证据覆盖 standard MHA、scarce-channel MLA、bidirectional 3D-RoPE
  video DiT、progressive/continued training 与成熟 8B Q/K/V/O LoRA adaptation。
- 8B 证据支持 long-position probability、remote source dependence 与 routing，
  但尚未稳定转化为 top-1 generation、QA accuracy 或整体 LongBench 提升。
- post-submission OLMo-2 1B Instruct（实际 1.485B 参数）的 matched 4K-only
  counterfactual LoRA 已在同一
  8K `niah_single_1` n=100 集上得到 Native 0/100、EVQ 69/100 strict
  autoregressive exact；第二 EVQ training seed 为 67/100。它关闭一个 2x
  NIAH readout endpoint，但训练和测试共享 official generator family，不等于
  完整 RULER、未见任务或广泛 downstream。
- 一条后续 single-seed continuation 明确加入 4K-only
  VT/CWE/FWE/QA generator-family supervision，并保留 NIAH/LongAlign replay；
  它在 disjoint generated rows 上得到完整 13-task RULER
  0.3751/0.2129/0.0613（4K/8K/16K）。这是 task-adapted capability 与
  length transfer，不是 clean unseen-task transfer，也没有 matched Native
  continuation 支持 pure-EVQ attribution。
- matched LLaMA-3-8B physical-8K 13-family continuation 在 16K 的 official
  macro 为 Native-LoRA 0.295%、EVQ-LoRA 14.03%；它是 single-seed
  task-family adaptation。32K EVQ 为 0，Native-LoRA 仅完成 10/13 cells 且均
  为 0。
- 当前没有完成的 LLaMA counterfactual-training arm；LLaMA gold deletion 与
  source/frequency swaps 是 evaluation intervention。
- 这些结果不推翻 Cosh 在 \(C_{\mathrm{app}}\) 下的条件定理。

---

## 5. 当前 rebuttal 的正确叙事

### 对 FMRoPE

核心回答是技术对象不同，而不是声称全面性能领先。

应说明：

- FMRoPE 选择或 retarget 几何 grid 的 base/range；
- YaRN/LongRoPE 可非均匀移动 channel，但其对象是 target transport 或
  post-hoc search/adaptation；
- EVQ 在训练前改变有限 grid 的指数位置和通道分配，并从显式 surrogate
  闭式构造；
- 两者不是同一个参数化；
- exact-range 三 seed 直接表明，在 sampled range 完全相同时只改 interior
  positions 仍改变训练后 NLL；
- target-aware range scaling 更强，但不消除 allocation 作为可识别训练期
  设计变量的技术新颖性。

### 对理论归因

应说明：

- Cosh 唯一性只属于代理目标；
- \(\tau\) 是 operating rule；
- tuned-\(\tau\)、非 Cosh schedules、held-out base/head 实验补充了 finite-\(\tau\) 和方法归因证据；
- 不存在静态 schedule 全长度支配。

### 对规模和评测

按 evidence ladder 回答：

1. frequency allocation；
2. language-model probability；
3. gold-answer probability；
4. causal remote-source use / attention routing；
5. autoregressive readout；
6. downstream accuracy。

已有跨架构、MLA、video DiT、progressive/continued training 和成熟 8B LoRA
证据说明适用范围不局限于小模型从头训练；但不同 tier 不得汇总成 universal
superiority。OLMo-2 1B Instruct 已补上一个 8K NIAH top-1 readout endpoint；
并在明确 task-family adaptation 后补出完整 13-task RULER 的 4K–16K
能力。clean unseen-task transfer、matched schedule attribution 与 broader
downstream accuracy 仍未闭合。

独立的 OLMo-2 1.485B from-scratch 2.1B-token 数值报告已经存在，但当前
Markdown、curated JSON 与本地 raw 状态冲突。完成 provenance reconciliation
前必须标为 conditional，不得与上述成熟 Instruct LoRA 结果混淆，也不得把任一
结果写成方法首次具有广泛适用性。

不得用明显欠训练、无法完成任务的模型强行跑下游，然后将无意义结果作为核心证据。

---

## 6. 实验原则

每个新实验必须先回答：

> 这个实验直接回应哪一条 reviewer 或 AC concern？

无法回答时，不做。

### 必须满足

- 只改变一个关键变量；
- 初始化、数据顺序、训练 token、optimizer 和评测协议严格匹配；
- calibration 与最终 test 分离；
- 报告 seed 和不确定性；
- 先做最低成本的方向筛选；
- 设置明确的停止条件；
- 优先复用已有 checkpoint 和评测数据；
- 先测 RTX 5090 实际吞吐，再估算时间。

### 优先级

1. FMRoPE 直接受控比较；
2. tuned \(\tau\)、held-out base/head、matched schedules；
3. 较大模型 matched Geo vs EVQ；
4. 只有在直接服务 rebuttal 时才进行额外理论归因实验。

### 停止条件

以下情况应停止扩大实验：

- 初始 screening 无稳定方向；
- 增益只出现在单个 seed；
- 增益来自不匹配的 base/span；
- 更强 baseline 重新调优后增益消失；
- 模型本身没有完成评测任务的基本能力；
- 实验已经明显演化为下一篇论文的方法开发。

---

## 7. 新方法研究与 rebuttal 必须分开

以下方向具有研究价值，但默认属于 EVQ-v2 或后续论文：

\[
q_i(T)=a(T)+s(T)u_i+\lambda(T)r_\theta(u_i).
\]

包括：

- affine/residual 分解；
- base/range-orthogonal residual；
- target-conditioned \(\lambda(T)\)；
- Cosh 与 two-band residual basis；
- train-to-eval frequency transport；
- attention-derived target-dependent schedule。

这些内容可以作为内部分析和未来工作。

除非实验直接澄清原投稿方法，否则不得：

- 把新参数化包装成原 EVQ；
- 用新方法成功替代原方法的证据；
- 在 rebuttal 中重新定义论文贡献；
- 因为发现更强的新方法而主动否定原稿。

---

## 8. 仓库审计边界

以实际代码、配置和日志为事实来源。

但本任务不是全面代码审计。

只检查与当前问题直接相关的内容，例如：

- 实际 `inv_freq` 公式；
- midpoint 与 endpoint grid；
- base/range baseline 的具体实现；
- checkpoint 加载后的频率表；
- 新实验是否严格匹配。

不得主动扩展到与 AC concern 无关的实现问题。

若某个代码事实会直接使计划中的 rebuttal 表述不准确：

1. 在内部报告中明确指出；
2. 不依赖该不准确表述；
3. 提供最小、准确的替代表述；
4. 不把它扩展成全面自我审查或新的 rebuttal 主线。

---

## 9. 写作原则

### 必须做到

- 先直接回答 reviewer 的问题；
- 使用公式或受控实验说明区别；
- 区分现有投稿证据与 post-submission 新实验；
- 对合理限制做窄而准确的承认；
- 保留所有有充分证据支持的核心贡献；
- 每个 reviewer 单独回应其主要 concern。

### 禁止

- 写成长篇论文事故审计；
- 主动增加 reviewer 没提出的新问题；
- 用大量内部推演替代直接回答；
- 把性能不支配误写为没有 novelty；
- 把参数空间不同误写为经验上必然互补；
- 声称 universal SOTA；
- 声称 Cosh 是真实训练中的唯一最优；
- 为显得“诚实”而主动撤回并不存在的强主张；
- 因为一个负面消融而重新解释整篇论文。

---

## 10. Codex 输出规范

每次接到任务时，先输出：

1. 本任务对应的 reviewer/AC concern；
2. 当前已有证据；
3. 仍然缺失的最小证据；
4. 最小可执行方案；
5. 停止条件。

除非明确要求，不输出：

- 全面论文评价；
- 新论文架构；
- 长篇研究路线；
- 无关风险清单；
- 主动披露建议；
- 超出 RTX 5090 或 rebuttal 时间范围的实验。

最终原则：

> Rebuttal 的目标是准确解决已有质疑，而不是主动扩大问题空间。

> 技术新颖性由方法变量和构造区别决定，不要求在所有设置中支配最强 baseline。

> 新实验只补决定性证据，不重新发明论文。
