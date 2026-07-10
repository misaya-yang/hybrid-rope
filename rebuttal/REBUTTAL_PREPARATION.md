# EVQ-Cosh Rebuttal Preparation

日期：2026-06-10

用途：把三份模拟审稿材料、后续模型推理材料、用户新增五条 rebuttal 判断，统一整理成一份可执行的 rebuttal 工作文档。

原始材料状态：`rebuttal/raw_sources/00_INDEX.md` 只保留历史索引，五份 verbatim 文件当前不可用，不能把索引当作可复核原文。用户重新提供的 attachment 才是当前可读的一手材料；做决策使用本文件，但不要把摘要当原文。

后续使用规则：之后所有 rebuttal 审查、实验取舍、论文措辞优化，都优先参考本文件的 scope、反噬措辞黑名单和补实验优先级。若新实验结果进入，先更新本文件的证据矩阵，再决定是否改 paper 或 response。

输入材料索引：

| Source | 内容 | 本文处理方式 |
| --- | --- | --- |
| local-only `raw_sources/00_INDEX.md` | 历史文件名、行数、SHA256 记录；对应 verbatim 文件当前缺失 | 只作恢复线索，不作为可复核原文 |
| `fable相关资料原文.md:1-79` | 第一组 reviewer panel + AC；实证优先级清晰 | 提取 P0 实验排序：YaRN sweep、AR exact、Primary II seeds、MLA tau、图表修正 |
| `fable相关资料原文.md:84-158` | 第二组 reviewer panel + AC；更强对抗 | 提取 R1/R2/R3/AC 共识和 veto 点 |
| `fable相关资料原文.md:161-189` | 用户五条新证据和反噬判断 | 作为本文第 1 节逐条 MD 化的主线 |
| `fable相关资料原文.md:192-206` | 分数/录用概率校准 | 加入总判断和 rebuttal 包价值评估 |
| `fable相关资料原文.md:208-278` | 模型推理底稿 | 只提取行动项，不复写个人信息或推理原文 |

核心约束：

- Rebuttal 不是二次提交论文。优先回应 reviewer 已经问到或必然追问的问题。
- 不无限加实验。补实验只服务于关闭具体攻击面。
- 不新增或发明结果。所有数字必须来自已完成实验或新跑完的最小验证。
- 主 claim 保持窄：EVQ-Cosh 是训练期 RoPE frequency allocation 的机制和设计轴，不是通用 long-context SOTA，也不是 YaRN/LongRoPE/LongRoPE2 的替代品。
- PK 必须定义为 teacher-forced NLL-gap retrieval，除非明确标注 autoregressive exact match。

## 0. 总判断

这篇的 rebuttal 胜负不在于把所有实验补全，而在于让 reviewer 相信三件事：

1. 主要贡献是一个新的训练期频率分配轴，而不是“所有长上下文场景都更强”。
2. 关键攻击面已经被最小控制实验或明确 scope 处理：LoRA 混杂、训练量混杂、YaRN 调参、Primary II seed scope、TF PK 指标、MLA tau convention。
3. 对真实 limitation 不强辩：1B raw reversal、MLA tau convention、text base-tuning/tuned scaler 缺口、downstream benchmark 弱信号，都要主动降级。

两份成品评审和一份模型推理底稿给出的分数高度一致：

| 画像 | 预期分数 | 主要卡点 | 翻盘条件 |
| --- | --- | --- | --- |
| R1 理论/PE | 5 | shape 与 scale 两段推导、A.15 未测、learnable tau 负结果、Bessel/constant alpha | 把 tau rule 说成 operating default，给出 learned tau 轨迹或 L_eff^J measurement plan |
| R2 实证严苛 | 4 | 1B raw reversal、Primary II 单 seed、YaRN scale 未调、TF PK 非 AR exact、LoRA 缺控制、图表矛盾 | Geo+LoRA 控制、Geo+YaRN scale sweep、AR exact、Primary II seed/provenance、主动修图 |
| R3 系统/实用 | 6 | 规模、下游 benchmark、MLA tau convention、production mismatch | 强调零参数、MLA scarce-channel、dead-channel audit、controlled LoRA 工业相关性 |
| AC | borderline reject | 无 champion，R2 attack 太集中 | rebuttal 后争取 5/5.5/6.5 或 5/6/6 |

现实策略：不要追求把所有 reviewer 变成 7。目标是把 R2 从 4 拉到 5，把 R1 从 5 拉到 5.5/6，让 AC 有理由说“scope 清楚、主要攻击已回应、贡献真实”。

### 0.1 Fable 原文里的概率校准

`fable相关资料原文.md:192-206` 给了一个很有用的审稿心理模型，必须保留：

1. NeurIPS 审稿本身有高随机性，同一篇论文换一组委员会，接收集合可能大幅变化。因此 “5/6/6” 不是侮辱，oral 级论文重抽 reviewer 也可能掉到 borderline。
2. 本文的优点是头版可见，弱点需要细读交叉核对：100% vs 61%、-31.1%、-35% 在正文显眼；1B reversal、retained seed-42、Figure/Table mismatch 要花更久才会挖出。
3. 不能把硬伤当作普通澄清。训练规模、Primary II single seed、TF PK、图表不一致会叠加成信任折损，把细究型 reviewer 从 5 拉到 4。
4. 估计分布：
   - 全浅读约 33%：可能 5/6/6 或更高，poster 接收概率约 75%。
   - 恰一个细究型约 45%：可能 4/5/6，rebuttal 定生死；带 Geo+LoRA 控制包后接收概率约 45-50%。
   - 两个以上细究型约 22%：可能 4/4/5，基本很难。
   - 加权 poster 接收约 42-47%，拒稿约 50-55%，spotlight 以上约 2-3%。
5. 可控变量不是“写得更乐观”，而是执行 rebuttal 包：Geo+LoRA 控制、多 seed/token 披露、1B 行改标签。原文估计这个包价值约 10-15 个百分点，正好是 45% 与 65% 乐观估计之间的差距。

战略含义：

- 不要把 WA 均分近似等于接收。一个自信负面 reviewer 加无 champion，AC triage 仍可能拒。
- Rebuttal 的核心不是追求全面加实验，而是把“细究型 reviewer 的 4 分锚点”变成“有保留的 5”。
- Champion 诱饵最可能来自 Table 16 dead-channel audit 和 MLA relevance，不是 QuALITY/RULER。
- 反复强调“弱点埋藏”不是让我们侥幸，而是说明主动修 trust 问题很值钱：图表一致性、token 披露、seed scope、1B relabel 都是半档分的东西。

## 1. 用户新增五条的逐条 MD 化

### 1.1 Geo+LoRA 控制组：真正改变局面的新证据

#### 真正改变了什么

Geo+LoRA 控制组是目前最强的新 rebuttal 证据，因为它同时回应两条 R2 质疑：

- “外推改善只是 LoRA/LongAlign 微调带来的，不是 EVQ 频率注入带来的。”
- “EVQ 只在欠训练小模型上有效，工业 checkpoint 上不成立。”

如果新证据成立，逻辑链是：

1. Base 是 15T-token 级工业 checkpoint。
2. Geo+LoRA 使用同样 300 步 LongAlign/LoRA 路径，但不注入 EVQ 频率。
3. Geo+LoRA 没有获得同等外推增益。
4. EVQ-LoRA 的外推改善不能归因于 LoRA 本身，只能归因于频率注入和 LoRA 适配共同作用。

这会直接废掉原 R2 的 LoRA confound attack，也能作为训练充分模型上的远端锚点，反击“欠训练伪影”。

#### Table 23 推荐重构

Table 23 应改成三行，而不是 Base vs EVQ-LoRA 两行：

| Row | 作用 | 必须报告 |
| --- | --- | --- |
| Base | 工业 checkpoint 原点 | 8K/16K/32K PPL，是否有 AR/passkey/RULER |
| Geo+LoRA | LoRA/LongAlign control | 同样 LoRA rank、steps、data、seed；证明微调本身不带来外推增益 |
| EVQ-LoRA | 频率注入 + 同样 LoRA | 同样 metrics；只把 Geo+LoRA 到 EVQ-LoRA 的差额归因给 EVQ |

如果有多 seed，必须在表里标出来。不要只在文字里说“multi-seed confirmed”。

#### +30% in-distribution cost 的拆分

旧表里 EVQ-LoRA 在 8K 有约 +30% in-dist PPL cost。rebuttal 里不能把 +30% 都说成 EVQ 特有成本。

推荐拆法：

- Base -> Geo+LoRA：LoRA/LongAlign 本身的成本。
- Geo+LoRA -> EVQ-LoRA：EVQ 频率注入的增量成本。

如果 Geo+LoRA 也在 8K 退化约 15%，那 EVQ 特有成本约为剩余 13% 左右。这个数字比“EVQ-LoRA +30%”更适合作为 response 里的成本表述。

注意：这里必须用实际结果算，不要在 rebuttal 里写近似口头数字。

#### 还差的最小证据

还需要一行 eval-only baseline：

| 缺口 | 为什么重要 | 成本 | 用途 |
| --- | --- | --- | --- |
| Geo + Dynamic NTK 或 Geo + YaRN，零训练，16K/32K | LLaMA-3 扩上下文默认强对照通常是 training-free scaling，而不是 raw Geo | 纯 eval，低成本 | 阻止 reviewer 换站位说“打赢 raw extrapolation 不算数” |

这条不是为了证明 EVQ 全面打赢 YaRN，而是为了证明在 LoRA 板块里，EVQ-LoRA 的外推收益不是被一个零训练 Geo scaler 轻易抹掉。

#### 可以怎么写

推荐口径：

> We added a matched Geo+LoRA control on the same pretrained checkpoint, data, rank, and 300-step schedule. This isolates the effect of the frequency injection from LoRA capacity and LongAlign exposure. Geo+LoRA accounts for the adaptation cost but does not reproduce the long-context extrapolation gain; the remaining gain is specific to EVQ frequency injection under the same adaptation budget.

#### 会反噬的写法

- “LoRA proves EVQ scales to industrial training.”
  反噬原因：它只是 post-hoc adaptation，不能替代 from-scratch trillion-token validation。
- “EVQ-LoRA solves long-context LLaMA-3.”
  反噬原因：RULER/LongBench/指令能力没有同等提升，且 in-dist cost 存在。
- “+30% cost is modest.”
  反噬原因：对工业模型 PPL +30% 不一定 modest。应拆成 LoRA cost 与 EVQ incremental cost。
- “No need for Geo+YaRN/Dynamic NTK baseline.”
  反噬原因：R2 会把攻击从 LoRA 混杂转到 training-free scaling baseline。

### 1.2 1B raw reversal 与 tau-L mismatch：可信但双刃剑

#### 真正改变了什么

tau-L mismatch 解释能把最危险的表述：

> EVQ 被训练更久后失效。

改成更窄的表述：

> 静态 tau 对 training length schedule 敏感，尤其在 sparse-channel MLA 中，stage 切换会放大 allocation mismatch。

这是有价值的，因为它把“机制无效”降级成“schedule/operating-point limitation”。

#### 双刃剑在哪里

第一，工业训练经常就是 multi-stage length extension。
如果我们只解释原因，不给处理方案，reviewer 会说这个 limitation 正好击中 deployment。

第二，论文内部有一个明显对照：

| 设置 | L 变化 | 现象 | 对 tau-L mismatch 解释的挑战 |
| --- | --- | --- | --- |
| Progressive 512 -> 2048 MHA | 4x | EVQ+YaRN gap 变大 | 没有换 tau，却更强 |
| MLA 2K -> 4K 或 4K/1B row | 2x | raw EVQ reversal | L 变化更小却反转 |

如果 rebuttal 只说“L 变了所以 tau mismatch”，会被 reviewer 用 progressive row 反打。

#### 推荐补的一层解释

必须引入 channel scarcity：

- MHA 有 K=64 左右的频率通道，频率分配冗余更高。
- MLA 只有 K=16，A.19 的 distortion/scarcity bound 里存在 K^{-1}/K^{-2} 放大项。
- 当 tau 与 length schedule 不匹配时，MLA 的少通道预算更容易把错配放大成 raw PPL reversal。
- Progressive MHA 的 K=64 冗余可以吸收一部分 mismatch，因此不构成直接矛盾。

推荐 framing：

> The 1B row is better read as schedule sensitivity in a scarce-channel MLA regime, not as a same-configuration saturation law. The progressive MHA result is not contradictory because its larger rotary budget gives more redundancy; in MLA, the same allocation error is amplified by the K-dependent scarcity terms.

#### 必须改掉的标签

当前 `table_evidence_tier.tex` 里写：

> MLA 1B-token (robustness to training saturation)

这个标签会反噬。它暗示这行是 robustness evidence，但 raw EVQ reversal 正好说明它不是。

推荐改成：

- “MLA 1B-token schedule-sensitivity check”
- “MLA 1B-token single-seed stress check”
- “MLA 1B-token supporting limitation”

#### 最干净的补实验

| 实验 | 目的 | 成本/风险 | Rebuttal 价值 |
| --- | --- | --- | --- |
| 三个 500M primary MLA checkpoint 固定 L 继续训 200-300M tokens，画 gap-vs-tokens | 判断 gap 是否随更多 tokens 消失，避免 L-schedule 混杂 | 中高，需已有 checkpoint | 最干净地回应“训练量抹平效应” |
| 1B fixed-L rerun | 直接解决 saturation 质疑 | 高，窗口内不一定现实 | 如果结果好，杀伤力最大；如果差，风险也最大 |
| stage 切换时 re-warp + 短适应 | 给工业 multi-stage 的处理方案 | 中等 | 即使不能完全跑完，也可作为机制性 mitigation |

#### 会反噬的写法

- “1B row supports robustness to training saturation.”
  反噬原因：raw EVQ 已经 reversal。
- “This is just overtraining.”
  反噬原因：Chinchilla 是 compute-optimal，不是 overtraining；工业模型常远超 Chinchilla token ratio。
- “Progressive result proves schedule mismatch is harmless.”
  反噬原因：MHA K=64 与 MLA K=16 不同，不能直接迁移。
- “EVQ+YaRN -2.5% at 1B solves the issue.”
  反噬原因：单 seed，量级接近 primary MLA std，且 raw reversal 仍在。

### 1.3 训练强度攻防：不要写“9B tokens = 过训练”

#### 真正可用的证据链

不要说“454M 的 Chinchilla-optimal 是 9B，所以 9B 是过训练”。正确表述是：

- Chinchilla 是 compute-optimal reference，不是 overtraining threshold。
- 工业模型常训练到 Chinchilla token ratio 的 10-100 倍。
- 我们不能用“过训练”这个词为小 token budget 辩护。

论文里已有更好的三段证据链：

1. C.1 / MLA training progression：EVQ 16K advantage 随训练进展增长到 -31.1%，in-dist cost 从 +1.4% 降到 +1.1%。
2. 750M continue@4K：in-range PPL 已到 25.9/26.2，但 16K PPL gap 是全文最大之一，约 -45.9%。
3. Geo+LoRA 控制：工业级 pretrained checkpoint 上，微调本身不能解释外推收益，频率注入仍有作用。

这三段可以连成：

> 在我们可观测的训练进展、较大 continuation、工业 checkpoint adaptation 三个层级里，EVQ gap 没有简单随训练强度消失。我们不声称这证明 trillion-token from-scratch durability，但它反驳了“只是欠训练伪影”的简单解释。

#### Primary I/II token budget 必须补报

R2 会问 Primary I/II 总 token 数。当前需要谨慎：

| 项 | 目前可见证据 | 风险 |
| --- | --- | --- |
| Primary I EVQ x YaRN | curated JSON 写 training_tokens = 100M，L_train=2048，454M，10% passkey mix | 可补报，但要确认是否与最终提交表完全一致 |
| Primary II PE-dominant | 128-token historical report 写 Train tokens = 15M；current `phase11b_125m_dape.py` 写 TOKENS = 100M 且 SEQ_LEN = 256；curated Table 4 写 L_train=128 | provenance 需要先核对，不能在 rebuttal 里混报 |

推荐动作：

1. 先做 result-provenance reconciliation：Table 4 数字到底来自 15M/128 historical artifact，还是 100M/256 phase11b script 的后续 curated fallback。
2. Rebuttal 里只报已经能 trace 的 token count。
3. 如果 trace 仍不完整，就诚实写成 “we will add token budgets and seed scope in revision”，不要编造。

#### 可以怎么写

> We agree that total token budgets should be visible. We will add them next to the primary tables. The key point is that the observed EVQ gap does not monotonically disappear with stronger training in the evidence we have: in the MLA 8K/500M progression the 16K gap grows while the in-range cost shrinks; in the 750M continue@4K row the in-range PPL is already low yet the long-range gap is large; and the matched Geo+LoRA control on a heavily pretrained checkpoint separates frequency injection from adaptation alone. We do not claim this closes trillion-token from-scratch validation.

#### 会反噬的写法

- “9B tokens would be overtraining.”
  反噬原因：概念错误。
- “Training longer increases EVQ advantage.”
  反噬原因：1B raw reversal 是反例。只能说“does not simply erase the effect in our scoped evidence”。
- “750M proves scale.”
  反噬原因：single seed supporting only。
- “LoRA proves from-scratch durability.”
  反噬原因：post-hoc adaptation，不是预训练 scaling law。

### 1.4 Shape/scale 怎么选：learnable tau 负结果反而可用

#### 真实问题

Reviewer R1 会问：

> 同一个一参数族里，为什么 learnable tau 反而比闭式规则差？

旧防守如果只说“optimization failed”，会显得方法脆弱。更好的解释是：这正好说明训练损失不是选择外推频率分配的好目标。

#### 机制解释

Learnable tau 输给固定 EVQ 不异常，原因有三层：

1. 训练目标 myopic：训练 loss 只看 L_train 内，外推收益不可见。
2. Waterbed / in-range cost：外推分配可能有正的 in-range cost，梯度会系统性偏向更小 tau。
3. 频率非平稳性：训练中途改变 inv_freq 会和已学 Q/K phase coupling 冲突，A.13 的 LoRA stiffness 机制可以作为类比。

因此 Table 4 的 437.9 不应写成“learnable tau 也有效”，而应写成：

> training-time allocation cannot be reliably learned from the in-range objective alone; this motivates a prior closed-form allocation.

#### 最值得补的轻量证据

| 证据 | 成本 | 用途 |
| --- | --- | --- |
| learned tau trajectory | 读取已有 `tau_trajectory.json` 或训练日志 | 如果 tau 向小值漂移/震荡，直接支持 myopic-loss 解释 |
| in-range PPL vs tau 曲线 | 已有历史报告显示 in-range 近似 flat | 解释训练 loss 对外推 tau 不敏感 |
| fixed tau sweep vs learnable tau | 已有 Table 4/历史报告 | 说明外推 optimum 与训练 objective optimum 不一致 |

注意：历史报告显示 learnable tau 收敛到约 1.14，当前 Table 4 固定 EVQ row 是 tau=5.0/333.7。要核对最终表述，不要把不同阶段的 tau=1.5、tau=5.0 和 formula tau 混在一起。

#### A.15 measure-then-allocate 是理论焊接路径

R1 最强攻击是：

> shape 来自 variational surrogate，scale 来自 softmax transport，中间没有测 trained model 的 L_eff^J。

最强回应不是强辩“已经统一推导”，而是执行或承诺一个 measure-then-allocate 协议：

1. 在 Geo pilot checkpoint 上测 empirical distance distribution D(Delta)。
2. 测 softmax-Jacobian effective length L_eff^J。
3. 用测得的 D(Delta) 拟合 alpha/beta，得到 shape。
4. 用 L_eff^J 定 scale tau。
5. 说明 shape 和 scale 都来自同一个 trained-model measurement pipeline。

这条如果能 eval-only 跑出初步数值，就是 R1 的一刀闭环。

#### 可以怎么写

> The learnable-tau row is not evidence that the EVQ family is unnecessary; it shows why the allocation should not be left to the in-range objective. The extrapolation benefit is invisible during training at L_train, while any in-range waterbed cost is visible. This drives learned tau toward the training-loss basin rather than the extrapolation basin. We will add the tau trajectory and clarify that EVQ is a prior allocation rule, not a learned PE parameter.

#### 会反噬的写法

- “Learnable tau validates the fixed rule.”
  反噬原因：它明显更差。
- “tau=d_eff/sqrt(L) is globally optimal.”
  反噬原因：论文和证据都只支持 operating default / basin selector。
- “Shape and scale are fully derived from one theorem.”
  反噬原因：R1 已经抓住两段推导。
- “A.15 is future work, so irrelevant.”
  反噬原因：它正是理论桥梁缺口。

### 1.5 分数校准、VideoRoPE、叙事重心

#### 分数校准

这几份 simulated review 是最强对抗采样，不是无偏评估。合理预期：

- pre-rebuttal：4/5/6 或 5/4/6，borderline reject。
- 如果 Geo+LoRA 控制、Geo+YaRN sweep、AR exact、Primary II seed/provenance、图表修正做得好：可能变成 5/5.5/6.5 或 5/6/6。
- AC 是否 accept 取决于是否出现 champion。R3 是潜在 champion，R2 是必须安抚的 veto 风险。

#### VideoRoPE 对比怎么用

不要把 rebuttal 写成 “VideoRoPE 也没比我们强”。这会显得 defensive。

应该提炼 venue standard：

- PE/position allocation 方向的接收证据通常不是 trillion-token frontier-scale。
- DAPE、FoPE、Resonance RoPE、HoPE 一类工作都依赖机制诊断、小中规模、受控 benchmark。
- EVQ-Cosh 的证据密度在这个 band 内，但 reviewer culture 更奖励可识别 benchmark。

#### 最优叙事重心

不要去硬凑 benchmark，因为 454M 上 QuALITY/RULER/LongBench 容量贴地板，可能越跑越暴露弱点。

把 rebuttal 主动重心放在三个“别人立刻有用”的点：

| 点 | 为什么能动 reviewer prior |
| --- | --- |
| MLA scarce-channel | 与 DeepSeek 系 compressed RoPE 架构相关，R3 最容易 champion |
| Table 16 dead-channel audit | 即使不采用 EVQ，社区也能用这个审计观察 |
| Controlled LoRA | 工业 checkpoint 上的最小频率注入证据，关闭 LoRA confound |

#### 会反噬的写法

- “我们比 VideoRoPE 更理论。”
  反噬原因：reviewer 不会因为别的 oral 的弱点给你加分。
- “我们不需要 benchmark，因为 diagnostics 更科学。”
  反噬原因：听起来像逃避实用验证。应说“our claim is mechanism-scoped; downstream benchmark at this scale is capacity-limited and reported as non-regression/supporting only”。
- “RULER/LongBench 不重要。”
  反噬原因：对系统 reviewer 很刺耳。应说“not the primary endpoint for this mechanism study; future production-scale validation should include them”。

## 2. 综合 reviewer 问题矩阵

| Issue | Reviewer 画像 | 真实状态 | Rebuttal posture | 最小动作 |
| --- | --- | --- | --- | --- |
| LoRA confound | R2/R3 | 旧 Table 23 缺 Geo+LoRA；用户称新 Geo+LoRA 控制已改变局面 | 认下新证据，重构表，不夸成 from-scratch scale proof | Base/Geo+LoRA/EVQ-LoRA 三行 + zero-training scaler reference |
| 1B raw reversal | R2/AC | raw EVQ reversal；EVQ+YaRN+FT 小幅存活 | 主动承认为 schedule-sensitive limitation | 改 “robustness” 标签；解释 K=16 放大 mismatch |
| Training budget | R2 | Primary I 可见 100M；Primary II provenance 需核对 | 补报 token counts，不能说 overtraining | Token/provenance reconciliation |
| Primary II seed scope | R2 | Geo/DAPE/EVQ seed 42，learnable tau 3-seed | Diagnostic-scoped，不作为唯一主证据 | 如可行补两 seed；否则明确 seed-42 retained |
| YaRN tuned baseline | R2 | Table 2 是 matched scale s=8，不是 tuned Geo+YaRN | 承认 scope，补 Geo+YaRN scale sweep | s in {4,6,8,12,16} 或 {2,4,8,16,32} |
| NTK reverse composition | R1/R2 | Table 5 note: NTK-aware @32x EVQ4 worse than Geo | 说 composition is YaRN-tested, not universal scaler claim | 主动提一句，避免 “general scaler” |
| TF PK vs AR exact | R2 | Primary I headline是 teacher-forced NLL-gap；AR exact not tabled | 主动定义，不当成 generation exact | 用 existing evaluator 报 AR exact 或解释 unavailable |
| Figure/Table mismatch | R2/AC | submitted Figure 8/9 均存在真实矛盾；本 rebuttal pass 不改 PDF | 主动承认，给出 n=2086 aggregate/Table 20 source of truth，并承诺 revision 修正 | 见 `rebuttal/FIGURE_TABLE_AUDIT.md` |
| MLA tau convention | R1/R3 | d_eff=d_head 是 convention；tau=d_rope/sqrt(L) sanity check 缺 | 承认 empirical convention | 单 seed tau=0.354 ablation 或列为 limitation |
| Base-tuning baseline | R2/R3 | 文本主实验 b=500K；text Geo best-b 不清楚 | 承认 practitioner baseline | 如果可行做 Geo base sweep；否则不 claim best-b dominance |
| Internal terms | R1/presentation | `Habitable Zone`, `Class C2` still appear | 修匿名/清洁文本 | 改为 defined technical terms |

## 3. 补实验优先级

### 3.1 Rebuttal 窗口内最高 ROI

| Priority | 实验/核查 | 类型 | 回应谁 | 成功标准 | 风险 |
| --- | --- | --- | --- | --- | --- |
| P0 | Figure 8/9 rebuttal audit and Table 21 erratum | 无 GPU | R2/AC | 已完成审计与安全回复；PDF 修订留到 revision | 不承认会污染所有数字可信度 |
| P0 | Table 23 重构为 Base/Geo+LoRA/EVQ-LoRA | 已有结果整理 | R2/R3 | 同一 checkpoint/data/steps/rank/seeds | 数字必须精确 |
| P0 | Primary I/II token budget/provenance reconciliation | 无 GPU | R2 | token count、seq_len、seed scope 可 trace | Primary II 当前材料有 15M/128 与 100M/256 混线风险 |
| P1 | Geo+YaRN/Dynamic NTK eval-only for LoRA checkpoint | 纯 eval | R2 | 16K/32K PPL reference | 若 scaler 很强，需要如实 scope |
| P1 | Geo+YaRN scale sweep on Primary I | 纯 eval | R2 | best Geo+YaRN 仍不能解释 EVQ+YaRN gap，或 gap 缩小但仍 supports matched-scope | 如果 best Geo 接近 EVQ，需要改 claim |
| P1 | Primary I AR exact | 纯 eval | R2 | 与 TF PK 同表，或说明 AR exact saturates/fails | 可能弱于 TF，但诚实加分 |
| P1 | learned tau trajectory | 日志读取 | R1 | 显示 tau 漂移到 training-loss basin | 若 trajectory 不支持，也可解释为 flat basin/noisy |
| P2 | MLA tau=d_rope/sqrt(L) ablation | 训练 | R1/R3 | 验证 d_eff convention 必要 | 成本中高，结果不确定 |
| P2 | fixed-L continuation gap-vs-tokens | 训练 | R2/AC | 排除 1B schedule confound | 成本高，但最干净 |
| P3 | 1B MLA multi-seed rerun | 训练 | R2/AC | raw reversal 是否稳健 | 高风险高成本，不建议作为第一选择 |

### 3.2 不建议在 rebuttal 窗口主攻

| 方向 | 为什么不主攻 |
| --- | --- |
| 新 LongBench/RULER 大规模补表 | 454M 容量可能贴地板，结果可能增加攻击面 |
| LongRoPE2 full search | 与 EVQ stage 不同，成本和变量太多，容易变成二次提交 |
| Bessel shape trained ablation | 理论上有趣，但不是 R2/AC 的当前核心 veto |
| 1.5-3B 从头复验 | 价值高但超 rebuttal 窗口 |

## 4. Rebuttal 回应草稿骨架

### 4.1 总开头

> We thank the reviewers for recognizing the main mechanism claim: training-time RoPE frequency allocation is a design axis complementary to operator design and inference-time range scaling. We agree that several supporting rows should not be read as production-scale validation. In the response we narrow the claim, add the missing control/provenance information, and separate matched-scale mechanism evidence from tuned-baseline or deployment claims.

### 4.2 对 R2：LoRA control

> We agree that the original LoRA row was confounded. We therefore added a matched Geo+LoRA control with the same pretrained checkpoint, data, LoRA rank, and 300-step schedule. This separates LongAlign/LoRA adaptation from EVQ frequency injection. Geo+LoRA accounts for the adaptation cost but does not reproduce the extrapolation gain; EVQ-LoRA retains the long-context improvement under the same adaptation budget. We will revise Table 23 to show Base, Geo+LoRA, and EVQ-LoRA side by side, with seed scope.

### 4.3 对 R2：1B reversal

> We agree the 1B MLA row should not be described as robustness to training saturation. It is a supporting schedule-sensitivity check, not a same-configuration token-scaling ablation. The row changes the length schedule and operates in a scarce-channel MLA regime where K-dependent allocation errors are amplified. We will relabel this evidence tier and discuss it as a limitation. The primary MLA claim remains the 8K/500M 3-seed scarce-channel stress test; the 1B row motivates fixed-length continuation and re-warp/short-adaptation follow-up rather than serving as support.

### 4.4 对 R2：training amount

> We added total token budgets and seed scope to the primary tables. We also agree that Chinchilla-style token counts should not be described as overtraining. Our narrower claim is that the EVQ signal is not explained by a simple undertraining artifact: within the 8K/500M MLA progression the long-range gap grows while the in-range cost shrinks; the 750M continue@4K row shows a larger long-range gap despite low in-range PPL; and the controlled LoRA experiment shows frequency injection matters on a heavily pretrained checkpoint. These are not trillion-token durability claims, but they rule out the simplest undertraining-only explanation.

### 4.5 对 R1：shape vs scale

> We agree that the cosh shape and the operating scale come from two layers of analysis. The variational surrogate gives the closed-form allocation family; the softmax-transport argument and sweep evidence select an operating basin. We will clarify that tau=d_eff/sqrt(L) is an operating default, not a global optimum theorem. We also add the learned-tau trajectory: because extrapolation benefit is invisible to the in-range training loss while in-range costs are visible, learned tau follows the training-loss basin rather than the extrapolation basin. This is precisely why EVQ is specified as a prior closed-form allocation rather than learned from the training objective.

### 4.6 对 R3：systems/practical relevance

> We agree that production-scale validation remains future work. The practical value of the current paper is not a deployment recipe but a low-cost mechanism: EVQ changes only inverse-frequency initialization. The most deployment-relevant evidence is the compressed-RoPE MLA stress test, the dead-channel audit, and the controlled LoRA result on a heavily pretrained checkpoint. We will avoid claiming downstream SOTA and will present downstream accuracy as a capacity-limited non-regression check.

## 5. 反噬措辞黑名单

不要写：

- EVQ replaces YaRN / LongRoPE / LongRoPE2.
- EVQ is a universal long-context solution.
- The 1B run proves robustness to training saturation.
- 9B tokens is overtraining.
- PK is retrieval accuracy, unless explicitly teacher-forced NLL-gap.
- EVQ+YaRN beats tuned Geo+YaRN, unless scale sweep proves it.
- LoRA proves from-scratch industrial-scale durability.
- Learned tau validates the fixed rule.
- tau=d_eff/sqrt(L) is globally optimal.
- Shape and scale are derived from one unified theorem.
- Downstream benchmarks are irrelevant.
- VideoRoPE is weaker, therefore EVQ should be accepted.

推荐写：

- mechanism study
- training-time frequency substrate
- matched-scale complementarity
- operating default / basin selector
- diagnostic endpoint
- supporting limitation
- schedule sensitivity
- seed-scoped evidence
- controlled LoRA adaptation
- production-scale validation remains future work

## 6. 当前文档和代码中应优先修的文字风险

| 位置 | 当前风险 | 建议 |
| --- | --- | --- |
| `paper/tables/table_evidence_tier.tex` | 旧版 “MLA 1B-token (robustness to training saturation)” | 已改成 schedule-sensitivity check；response 仍按 limitation 写 |
| `paper/sections/05_experiments.tex` Primary II | “retained seed-42” 容易暗示挑 seed | 改成 seed-scoped diagnostic，并说明 provenance |
| `paper/appendix/a1_proofs.tex` | “Habitable Zone”, “Class C2” 未定义 | 删除内部术语或定义成正式 lemma/observation |
| QuALITY figure/table | submitted Figure 8 stale/mislabeled；Table 21 有 26.6%→24.6% erratum | 主动承认；以 n=2086 aggregate 为 source of truth，并说 will correct in revision |
| LoRA appendix | 仍缺 Geo+LoRA exact row；旧 “modest cost/causal attribution” 口径会反噬 | 已先改成 post-hoc observation，并声明 attribution requires matched Geo+LoRA；最终仍需用新表替换旧两行口径 |
| Limitations | 1B raw reversal 说得不够尖锐 | 主动写成 limitation, not support |

## 7. 一页执行清单

### 今天能做

- [ ] 新建 `rebuttal/` 并保存本综合文档。
- [ ] 整理三份 review input 成 reviewer issue matrix。
- [x] 审计 submitted Figure 8/9、Table 20/21 并形成 rebuttal 安全口径；详见 `rebuttal/FIGURE_TABLE_AUDIT.md`。
- [x] 恢复 n=2086 QuALITY aggregate 和 99-run Phase-16 manifest，不修改 PDF。
- [x] 在 response 中把 1B row 限定为 schedule-sensitivity limitation。
- [x] 在 response 中把 LoRA 两行表限定为 post-hoc/supporting，并明确 attribution 需要 matched Geo+LoRA control。
- [x] 核对 Primary I/II token budget provenance，尤其 Table 4 的 15M/128 vs current phase11b 100M/256 差异。
- [ ] 把 Geo+LoRA 新结果整理成 Base / Geo+LoRA / EVQ-LoRA 表格，标 seed 和 metrics。
- [ ] 读取 learned tau trajectory，如果 artifact 存在，抽出 tau-final、轨迹方向、in-range/extrapolation relation。

### Rebuttal 窗口内优先跑

- [ ] Geo + Dynamic NTK 或 Geo + YaRN eval-only at 16K/32K for LoRA checkpoint。
- [ ] Primary I Geo+YaRN scale sweep。
- [ ] Primary I AR exact match。
- [ ] 如预算允许，MLA tau=d_rope/sqrt(L) 单 seed ablation。

### 最后写 response 时的结构

1. 先收缩 claim：mechanism/design-axis，不是 deployment SOTA。
2. 逐条回应 R2 veto：LoRA control、training budget、YaRN sweep、AR exact、1B limitation。
3. 回应 R1：shape/scale separation、learned tau trajectory、A.15 measurement plan。
4. 回应 R3：MLA/dead-channel/LoRA 三个实用点。
5. 结尾承认 production-scale validation remains future work。

## 8. 需要继续补入的数据

这些不能猜，必须从实际结果或日志读取：

| 数据 | 用途 |
| --- | --- |
| Base / Geo+LoRA / EVQ-LoRA 的 8K/16K/32K PPL 和 seed scope | 重构 Table 23 |
| Geo + Dynamic NTK/YaRN eval-only PPL | 关闭 LoRA training-free baseline attack |
| Primary I final token budget and run provenance | 回应 R2 token budget |
| Primary II exact token budget, seq_len, seed history | 回应 R2 single-seed/provenance |
| learned tau trajectory JSON | 把 Table 4 负结果变成方法动机 |
| Figure 8 / Table 21 source and corrected plot/caption | 主动修正已验证的图表矛盾 |
| Primary I AR exact | 防 TF PK 指标攻击 |

## 9. 四份输入材料的完整结构化摘要

这一节的目的不是复刻原文，而是把每份材料拆成后续 rebuttal 能直接使用的结构：reviewer stance、关键证据、核心问题、可回应点、不可强辩点。

### 9.1 Source A：三 reviewer + AC，强对抗版本

#### R1：理论/位置编码 reviewer

总体分数：Soundness 3 / Presentation 3 / Contribution 3 / Rating 5 / Confidence 4。

认可点：

- “RoPE frequency table 是有限谱预算”这个问题 framing 干净。
- operator / range scaling / training-time allocation 三轴区分有贡献。
- Theorem 1 的变分推导在 surrogate 条件下可验算通过。
- Eq. 11 的 min-kernel convex identity 成立。
- Table 1 epistemic map 与 A.9 self-consistency 写得诚实。

核心弱点：

1. Shape 与 scale 来自两套推导。A.11 里 surrogate fit 的 tau_surr 标度和实际部署的 L^{-1/2} 不一致。
2. 常数 alpha 是工程妥协。stationary-phase 给的是 phi-dependent diagonal 和 Bessel solution，cosh 胜出主要是可逆 CDF 与正性。
3. stiffness 选择有后验性。Pearson chi-square 给 gamma=0.465，精确 -0.5 靠 p≈0.80-0.85；A.17 channel-load axiom 只是 motivated choice。
4. trained model bridge 未测。A.15 的 L_eff^J replacement 是测量协议，但没有执行。
5. A.13 LoRA phenomenology 有循环校准：Lambda_0 用 77.1 观测校准再复现同一观测。
6. 编辑残留：Habitable Zone、Class C2 未定义，像内部术语泄漏。

R1 会问：

- 既然有 16K/32K checkpoint，为什么不测 L_eff^J？
- Learnable tau 为什么显著差于闭式规则？
- 能否做 Bessel allocation shape ablation，区分 cosh-specific vs any non-geometric allocation？

Rebuttal 转换：

- 不要强称 “single derivation”。要说 “variational shape + calibrated operating scale”。
- 把 learnable tau 负结果翻成“训练目标看不到外推收益，所以不能靠训练 loss 学 allocation”。
- A.15 是最短闭环路径。即使不能完整跑，也要把它作为 already specified falsifiable measurement protocol。
- 清掉内部术语比补理论更便宜，presentation 收益大。

#### R2：实证严谨 reviewer

总体分数：Soundness 2 / Presentation 2 / Contribution 3 / Rating 4 / Confidence 4。

认可点：

- Evidence tiering 和 seed 标注诚实。
- DiT shared-run head-to-head 控制了 optimizer/CUDA confound。
- Table 16 dead-channel audit 有独立价值。

核心弱点：

1. 训练饱和下反转。1B-token MLA raw EVQ 变 +11.1%，Primary III 的 -31.1% 是 500M tokens。
2. Primary II 是 seed 42，却挂 Primary。
3. YaRN baseline 未调优。固定 s=8 不等于 best Geo+YaRN vs best EVQ+YaRN。
4. Table 19 显示 NTK-aware @32x 下 EVQ4 很差，说明组合不是通用 scaler 互补。
5. TF NLL-gap passkey 不等于检索。750M row 显示 TF 100% 但 AR exact 0% vs 77.5%。
6. 缺 strongest baseline：Geo + tuned base / base sweep。
7. Figure 8 与 Table 21 矛盾：caption NLL，panel accuracy，delta 不一致。
8. LoRA row 缺 Geo+LoRA+LongAlign control。
9. 统计不足：EVQ raw vs Geo raw @8K 方差重叠，无显著性检验。

R2 会问：

- Primary I/II 总 token 数？
- Primary II seed history？
- Geo+YaRN scale sweep？
- Primary I AR exact？
- 1B-token MLA 是否多 seed？

Rebuttal 转换：

- R2 是最危险 reviewer，必须先处理他的问题。
- Geo+LoRA 新证据是 R2(g) 的关键反击。
- 但 R2 会立即转向 training-free scaler baseline，所以 LoRA 板块要补 Geo + Dynamic NTK/YaRN eval-only。
- 1B reversal 不能说成 support，只能说成 limitation/schedule sensitivity。
- Figure/Table mismatch 必须主动修，不然 reviewer 会认为全篇数字不可信。

#### R3：系统/实用 reviewer

总体分数：Soundness 3 / Presentation 3 / Contribution 3 / Rating 6 / Confidence 3。

认可点：

- 方法工程门槛极低：只改 inverse-frequency initialization。
- MLA scarce-channel 对 DeepSeek 系架构有直接相关性。
- K^{-1}/K^{-2} scarcity amplification 与 MLA/MHA 增益比方向一致。
- Table 16 dead-channel audit 对视频社区有独立价值。
- 证据分层与 falsifiable protocols 写作规范。

核心弱点：

1. Primary scale 小，8B 只通过 LoRA 路径且旧表 confounded。
2. LoRA retrofit 有 +30% in-dist cost，可能说明 post-hoc retrofit 不实际。
3. MLA d_eff=d_head 是 calibration convention，缺 tau=d_rope/sqrt(L) ablation。
4. 下游效用弱：QuALITY accuracy 接近随机，RULER 不提升。
5. Production base mismatch：paper MLA 用 base 500K，DeepSeek production 更接近 base 10K，dead-channel severity 不同。

R3 会问：

- 1.5-3B / >=50B tokens 的复验成本？
- 从业者如何联合选 b 和 tau？

Rebuttal 转换：

- R3 是潜在 champion。不要用 benchmark 弱点和 LoRA overclaim 激怒他。
- 强调三件“立刻有用”：MLA scarce-channel、dead-channel audit、controlled LoRA。
- 明确 production validation future work，而不是假装已解决。

#### AC meta-review

共识优点：

- 第三设计轴新颖。
- 零参数闭式方法有工程价值。
- 证据分层诚实。
- MLA scarce-channel 是最强单点。

决定性问题：

- 理论上 shape/scale 未焊接，A.15 未测。
- 实证上训练饱和或 schedule-sensitive 行出现 raw reversal。
- Primary II 单 seed。
- YaRN tuned baseline 缺失。
- TF passkey 可能膨胀表观效果。
- 图表矛盾降低信任。

AC 结论：

- 提交版 borderline reject。
- 如果 rebuttal 能补 Primary II seeds/token budget、Geo+YaRN scale sweep、AR exact、MLA tau sanity 或至少机制解释，存在 5/6/6 接收路径。

### 9.2 Source B：三 reviewer + AC，实证排序更清楚版本

#### R1：变分方法 reviewer

与 Source A 一致，但多强调：

- d_head exponent internal mismatch：A.11 surrogate 自身给 sqrt(d_head)，部署规则是 linear d_head。
- Table 6 collision-score 支持 linear d_head，但 collision-score validation 不等于 PPL validation。
- stiffness 叙事有三套：S_p p=1、load-moment p=2、exponent-matched p≈0.85。
- Prop 1 依赖 diffuse p0=1/L，trained attention 非 diffuse。
- MLA d_eff=d_head 的 latent-projection 论证为何成立，需要 tau=d_rope/sqrt(L) 消融。

新增 rebuttal 含义：

- 对 R1 不要只说 “Appendix explains”。要把三层 epistemic status 说清：
  - theorem：conditional on surrogate。
  - operating rule：semi-analytic + sweep basin。
  - trained model：functionally validated, not fully derived。
- Table 1 epistemic map 是可引用的 defense。它不是 weakness，而是诚实边界。

#### R2：长上下文实证 reviewer

问题排序更明确：

1. 只比 matched-s YaRN，不比 best Geo+YaRN vs best EVQ+YaRN。
2. Primary II DAPE headline 是单种子。
3. 1B token MLA reversal 是最大疑虑。
4. Primary III d_eff convention 缺消融。
5. TF PK 指标偏宽松。
6. 规模与真实任务收益不足。
7. LoRA 缺 Geo+LoRA control。
8. Figure/Table 矛盾。

新增 rebuttal 含义：

- Geo+YaRN scale sweep 是最便宜且最能打的实验之一。
- Primary II 补两 seed 如果窗口内能做，收益很高；但如果 provenance 混乱，至少必须把 seed scope 说清。
- MLA tau=d_rope ablation 成本较高，但保护最强 result。可作为 P2。

#### R3：训练系统 reviewer

新增强调：

- 没有 compute-optimal 预算 evidence，所以 lab 不会仅凭 454M 欠训练数据改初始化。
- in-range cost +0.4-1.2% 不为零。
- Video tau correction 0.53x 是 m=1 后验拟合，和 deployable default 有张力。
- 文本侧应该回答 EVQ@b=500K vs Geo@best-b。

新增 rebuttal 含义：

- 不要把 in-range cost 说成 zero。说 bounded/small in tested regimes。
- base-tuning baseline 是 real practitioner concern。若不补，就不能 claim best practical schedule。

#### AC meta-review

给出明确 P0/P1 实验优先级：

1. YaRN s-sweep，纯推理。
2. Primary I AR exact match，纯推理。
3. Primary II 补两个 seeds。
4. MLA tau=0.354 单 seed。
5. Figure/Table mismatch 主动说明。
6. 1B reversal 若不能多 seed，至少给 checkpoint curve。

### 9.3 Source C：模型推理底稿

注意：这份材料是推理过程，不应原文复写。本文只抽取可以行动的新增点。

新增有用点：

1. 频率密度表述可能引发误读。EVQ-Cosh 的 rho_tau 在 phi=0 处更高，直觉上是高频端；论文若说“more low-frequency resolution”需要非常精确，最好说“compresses ultra-low/dead tail and reallocates into active spectral band”，避免 reviewer 认为图文矛盾。
2. DAPE 128-token setting 是 deliberate diagnostic，但也确实是 engineered stress regime。rebuttal 要承认 diagnostic scope，不要把它说成 typical setting。
3. Text base-tuning baseline 缺口比旧文档更显眼。Video 有 base sweep，不等于 text MHA primary 有 best-b comparison。
4. Table 5 / Table 19 的 NTK reverse composition 必须主动 scope：EVQ 与 YaRN 组合好，不代表与所有 inference scalers 组合好。
5. Batch size / token count transparency 会被实证 reviewer 追问。Primary I/II token budgets 必须补报。
6. `Habitable Zone` 和 `Class C2` 是明显 presentation hygiene 风险，不修很亏。
7. 100±0% 不一定错，但它是 ceiling/right-censoring，rebuttal 不要把它当连续可比较的 effect size。
8. LoRA +30% cost 对 practitioner 不小，必须拆成 LoRA cost 和 EVQ incremental cost。

新增但需小心的点：

- 推理底稿里有关于 “cosh theory dressed as engineering” 的尖锐表述。rebuttal 不能照这个自贬。正确版本是：closed-form tractability is a deliberate design constraint; the theorem is conditional but useful.
- 它提到个人身份信息，不能进匿名材料。
- 它把若干历史 Figure/Table 编号当提交版编号，需对提交 PDF 复核。

### 9.4 Source D：用户五条补充判断

这五条已经在第 1 节逐条 MD 化。其核心优先级如下：

| 用户判断 | 本文采纳方式 |
| --- | --- |
| Geo+LoRA 是改变局面的新证据 | 升为 P0，重构 Table 23 |
| 1B tau-L mismatch 是双刃剑 | 改为 schedule sensitivity + K=16 scarcity amplification |
| 不要说 9B tokens = overtraining | 加入反噬黑名单，改用 progression/750M/LoRA 三段链 |
| learnable tau 负结果可翻成方法动机 | 加入 R1 response draft 和 trajectory action |
| 不要 benchmark 硬凑，重心压 MLA/dead-channel/LoRA | 加入 R3/champion strategy |

## 10. Master Issue Ledger

这一节按 rebuttal 重要性排列。每一项都包含：reviewer attack、真实状态、可说、不可说、最小证据。

### 10.1 P0：LoRA confound

Reviewer attack：

> EVQ-LoRA 的 8x/19x extrapolation gain 只是 LongAlign/LoRA adaptation，不是 EVQ frequency injection。

真实状态：

- 旧 paper 自认缺 matched Geo+LoRA+LongAlign control。
- 用户提供的新证据称 Geo+LoRA control 已经证明 LoRA 本身不带来同等外推增益。
- repo 里 `scripts/2026-04/README.md` 已有 Geo/EVQ LoRA multi-seed pipeline 说明，但最终结果数值需要另行填入。

可说：

- 新控制组使用 same checkpoint/data/rank/steps。
- Geo+LoRA 是 adaptation control。
- EVQ-LoRA - Geo+LoRA 是 frequency-injection-specific gap。
- Base -> Geo+LoRA 是 LoRA cost，Geo+LoRA -> EVQ-LoRA 是 EVQ incremental cost。

不可说：

- LoRA proves from-scratch industrial durability。
- EVQ-LoRA is production-ready。
- +30% cost is negligible。

最小证据：

- Base/Geo+LoRA/EVQ-LoRA table。
- seed scope。
- 16K/32K PPL。
- Geo + Dynamic NTK/YaRN eval-only reference。

### 10.2 P0：Figure/Table consistency

Reviewer attack：

> Figure shows accuracy deltas but caption says NLL; values disagree with table。

真实状态：

- 两份模拟评审都独立指出图表矛盾。
- 当前 `paper/main.pdf` 已验证同一问题：Table 21 是 Gold-NLL 表；Figure 8 使用 `fig5_downstream_qa` 的 accuracy 内容，但 caption/text 写 Gold-answer NLL。
- 当前源码对应 `tab:quality-nll` 和 `fig5_downstream_qa.pdf`；详见 `rebuttal/FIGURE_TABLE_AUDIT.md`。

可说：

- 主动承认 figure/caption stale or mislabeled and will be corrected。
- 说明 Table 21 的 NLL 数字是该段 NLL 论述的 source of truth。

不可说：

- Ignore as appendix。
- Reviewer misunderstood without showing exact source。

最小证据：

- `rebuttal/FIGURE_TABLE_AUDIT.md` 的 PDF/source mapping。
- 修正后的 NLL 图，或改成 accuracy caption/text 的最小 patch。
- source result JSON for QuALITY。

### 10.3 P0：1B raw reversal

Reviewer attack：

> When trained longer, raw EVQ reverses. Main MLA claim may be undertraining artifact。

真实状态：

- 1B row is single seed/supporting。
- It is not same-config longer-training ablation。
- It changes length schedule/data/provenance。
- Current evidence tier label “robustness to training saturation” is harmful。

可说：

- Real limitation。
- schedule-sensitive single-seed stress check。
- scarce-channel MLA amplifies mismatch through K-dependent terms。
- primary 8K/500M MLA remains 3-seed stress test。

不可说：

- 1B proves durability。
- Overtraining。
- EVQ+YaRN -2.5% fully solves。

最小证据：

- relabel in response/revision。
- mention fixed-L continuation as clean follow-up。
- if possible show checkpoint curve, not endpoint only。

### 10.4 P0：Primary I/II token budgets

Reviewer attack：

> Main claims are in undertrained regimes and total tokens are missing。

真实状态：

- Primary I curated JSON says 100M tokens.
- Primary II historical 128-token report says 15M tokens.
- Current `phase11b_125m_dape.py` says 100M tokens and seq_len 256, while Table 4 curated protocol says L_train 128.
- This mismatch may be historical script drift, but rebuttal must not blur it.

可说：

- We will add token budget and seed scope next to tables.
- Primary II is diagnostic, not production scale evidence.
- Undertraining-only explanation is inconsistent with progression/750M/Geo+LoRA controls.

不可说：

- All primary experiments are adequately trained by Chinchilla standards.
- 9B tokens would be overtraining.

最小证据：

- provenance reconciliation note for Table 4.
- exact token count per primary table.

### 10.5 P1：YaRN tuned baseline

Reviewer attack：

> Fixed s=8 may be unfair. Need best Geo+YaRN vs best EVQ+YaRN。

真实状态：

- Paper claim is matched-scale complementarity, not tuned-scaler dominance.
- But +39pp headline invites tuned-baseline attack.
- Table 5 L=256 supports leverage but not Primary I 454M tuned sweep.

可说：

- Matched-scale test is designed to test substrate/range complementarity.
- We do not claim superiority over every tuned-scale YaRN baseline.
- If scale sweep added, report best Geo/EVQ under same grid.

不可说：

- s=8 is universally fair.
- EVQ+YaRN beats tuned Geo+YaRN without sweep.

最小证据：

- Geo+YaRN s-sweep at Primary I checkpoint.
- Ideally EVQ+YaRN same sweep too, but Geo-only already addresses most attack.

### 10.6 P1：TF PK vs AR exact

Reviewer attack：

> Teacher-forced NLL-gap passkey is not retrieval。

真实状态：

- Paper defines PK correctly in §5.
- But headline “100%” can still inflate perceived generation ability.
- `eval_passkey_scratch.py` supports AR exact generation.

可说：

- PK is teacher-forced diagnostic by design.
- Add AR exact as auxiliary metric where feasible.
- Do not use TF PK as downstream generation claim.

不可说：

- PK equals exact retrieval.
- 100% TF means model can generate passkey.

最小证据：

- Primary I AR exact table or a statement that AR exact is pending/not primary.

### 10.7 P1：Primary II seed scope

Reviewer attack：

> PE-dominant DAPE/Geo/EVQ contrast is seed 42 but called Primary。

真实状态：

- Table 4 says Geo/DAPE/EVQ seed 42; learnable tau 3-seed.
- Historical report has learnable tau multi-seed, not necessarily full Geo/DAPE/EVQ multi-seed.
- `RESULT_PROVENANCE_MANIFEST` notes rows are retained seed 42.

可说：

- PE-dominant row is diagnostic and seed-scoped.
- It tests whether allocation shape alone can reduce extreme extrapolation gap.
- It is not sole primary evidence.

不可说：

- Complete multi-seed DAPE dominance.
- DAPE broadly beaten.

最小证据：

- If possible, rerun two seeds for Geo/DAPE/EVQ.
- Otherwise remove “Primary” emphasis or state seed scope in response.

### 10.8 P1：learnable tau

Reviewer attack：

> Why does learnable tau lose to closed-form tau?

真实状态：

- Table 4 learnable tau PPL@8K 437.9±12.2, EVQ fixed 333.7.
- Historical report says learnable tau converged around 1.14; fixed/sweep extrapolation optimum larger.
- This supports myopic training-loss interpretation.

可说：

- Learnable tau optimizes in-range training loss, not extrapolation.
- Extrapolation benefit invisible at L_train.
- This motivates closed-form prior allocation.

不可说：

- learnable tau validates fixed EVQ.
- SGD should have found same basin but just unlucky.

最小证据：

- tau trajectory plot/table.
- in-range PPL flatness vs extrapolation PPL curve.

### 10.9 P1：MLA tau convention

Reviewer attack：

> d_eff=d_head for MLA is empirical convention; tau=d_rope/sqrt(L) sanity check missing。

真实状态：

- Paper already admits convention.
- Primary MLA result is strongest evidence but also most exposed to this attack.

可说：

- d_rot counts quantized rotary channels; d_eff is operating convention for latent-projection attention path.
- Convention validated empirically by chosen setting, but direct ablation remains natural sanity check.

不可说：

- d_eff=d_head is theorem.
- tau=1.414 is derived from d_rope.

最小证据：

- tau=0.354 single-seed ablation, if possible.
- Otherwise add limitation and future work.

### 10.10 P1：base-tuning baseline

Reviewer attack：

> Practitioners can tune RoPE base. Does EVQ@500K beat Geo@best-b?

真实状态：

- Main text uses b=500K.
- Video has base sweep.
- Text primary best-b comparison not clearly tabled.

可说：

- EVQ holds base fixed to isolate allocation shape.
- Base tuning changes spectral span; EVQ changes within-band density.
- They are orthogonal axes.

不可说：

- EVQ beats all tuned-base Geo.

最小证据：

- text Geo base sweep, if cheap.
- Otherwise state as limitation.

## 11. 误解 vs 真实硬伤

### 11.1 Reviewer 误解或过度推断

| Reviewer reading | 为什么不完全成立 | 应对 |
| --- | --- | --- |
| EVQ claims to replace YaRN/LongRoPE | Paper scope says complementary substrate | 开头重申 axis separation |
| TF PK was hidden as AR retrieval | §5 已定义 PK as teacher-forced NLL-gap | 在 response 第一处数字旁重复定义 |
| 1B row is primary claim | Evidence tier marks supporting, but label wording有问题 | 改 label，主动降级 |
| Downstream accuracy weak means PE mechanism invalid | 454M capacity floor makes accuracy weak discriminator | 用 NLL/PPL/diagnostic scope，别说 downstream SOTA |
| Geo+LoRA confound remains after new control | 如果新 control 已有，R2(g) 已被回应 | 给表，不靠口头 |

### 11.2 Reviewer 真实硬伤

| Issue | 为什么是真硬伤 | 必须怎么处理 |
| --- | --- | --- |
| 1B raw reversal | 真实反例，不能被 rhetoric 消除 | limitation + schedule sensitivity + follow-up |
| Primary II seed 42 | Reviewer 对 selection bias 会敏感 | 补 seed 或 seed-scoped |
| tuned YaRN baseline | matched-scale 不等于 best baseline | 补 sweep 或明确不 claim |
| Figure/Table mismatch | 直接伤害数字信任 | 主动修正 |
| MLA tau convention | 最强系统结果缺自提 sanity check | ablation or limitation |
| Primary token budget | training amount attack 的入口 | 补报且 provenance clean |

## 12. 实验 Runbook 草案

这一节不是最终命令清单，而是后续 agent 接手时的执行地图。运行前必须核对数据/checkpoint 路径。

### 12.1 Geo+LoRA table rebuild

目标：把 LoRA appendix 从 confounded supporting row 改成 controlled adaptation table。

文件线索：

- `experiments/lora_evq_v2/train_evq_lora.py`
- `experiments/lora_evq_v2/eval_evq_lora.py`
- `experiments/lora_evq_v2/train_yarn_lora.py`
- `scripts/2026-04/01a_lora_train_geo_s42.sh`
- `scripts/2026-04/01b_lora_train_geo_s43.sh`
- `scripts/2026-04/01c_lora_train_geo_s44.sh`
- `scripts/2026-04/01d_lora_train_evq_s43.sh`
- `scripts/2026-04/01e_lora_train_evq_s44.sh`

需要产出：

- one table：Base / Geo+LoRA / EVQ-LoRA。
- columns：8K in-dist PPL, 16K PPL, 32K PPL, delta vs Base, delta vs Geo+LoRA, seeds。
- note：same checkpoint, same LoRA rank, same steps, same data。
- if available：RULER/AR exact separate，不能混入主 PPL 表。

审查点：

- Geo+LoRA 是否真的同样 rank=64、alpha=128、steps=300？
- EVQ-LoRA 与 Geo+LoRA 是否同 seed set？
- Base 是否用同一 eval script？
- 是否有 Stage2 retrieval 续训混入？如有，另表。

### 12.2 LoRA training-free scaler reference

目标：防止 R2 从 LoRA confound 转向 “default LLaMA context extension is training-free scaling”。

最小设置：

- Base checkpoint + Geo inv_freq。
- Apply Dynamic NTK or YaRN at eval only。
- Eval 16K/32K PPL under same data/eval script。

可接受结果解释：

- 如果 Geo+scaler 仍远差于 EVQ-LoRA：strong defense。
- 如果 Geo+scaler 接近 EVQ-LoRA：LoRA claim 改成 “EVQ-LoRA is competitive with training-free scaling under same eval”，不能说 uniquely solves。
- 如果 Geo+scaler 更好：LoRA row 降为 supporting caution，主 claim 不靠它。

### 12.3 Primary I scale sweep

目标：回应 “s=8 fixed unfair”。

建议 grid：

- minimal：s in {4, 8, 16}
- stronger：s in {2, 4, 8, 16, 32}
- user suggested variant：s in {4, 6, 8, 12, 16}

产出：

- Geo+YaRN best-s PPL/PK。
- EVQ+YaRN same grid if budget allows。
- report matched-s result remains mechanism test even if best-s gap changes。

风险处理：

- 如果 best Geo+YaRN improves a lot，response 要承认 tuned gap smaller。
- 不要隐藏 best Geo row。

### 12.4 Primary I AR exact

目标：回应 “TF PK != retrieval”。

文件线索：

- `scripts/supporting_eval/eval_passkey_scratch.py` already evaluates both NLL-gap and AR exact。

需要产出：

- For Table 2 checkpoints: AR exact @8K, maybe @12K/@16K。
- same depths/trials as TF PK if feasible。
- If AR exact too expensive, sample fewer trials and label auxiliary。

解释：

- 如果 AR exact mirrors TF PK：strong。
- 如果 AR exact weaker：still useful，因为诚实定义 metric，主 claim stays diagnostic。

### 12.5 Primary II provenance reconciliation

目标：确认 Table 4 的 exact protocol，避免 token budget 被 reviewer 打穿。

冲突：

- Historical 128-token report：Train tokens 15M, seq_len 128。
- Current `phase11b_125m_dape.py`：TOKENS 100M, SEQ_LEN 256。
- Curated `fig3_extreme_128.json`：protocol says L_train 128。

需要做：

1. 查最终 Table 4 数字来源。
2. 查 curated fallback 是否来自 historical 15M artifact。
3. 查是否有 100M/256 script later repurposed but not used for Table 4。
4. 在 rebuttal 里只报 confirmed token count。

产出：

- 一个 5-10 行 provenance note。
- 若不确定，写 “we will make token budget and seed scope explicit”，不报错数。

### 12.6 learned tau trajectory

目标：把 R1 的 “learnable tau why worse” 从 weakness 转为 method motivation。

文件线索：

- historical report mentions `tau_trajectory.json` artifacts。
- Current compact repo may not include raw remote artifact。

需要产出：

- final tau per seed。
- trajectory direction。
- in-range loss curve vs extrapolation PPL。

解释模板：

- training loss optimum tau != extrapolation optimum tau。
- fixed EVQ is prior allocation, not learned PE。

### 12.7 MLA tau sanity

目标：保护最强 Primary III。

设置：

- same 432M MLA, d_rope=32, L_train=8192。
- compare tau=1.414 vs tau=d_rope/sqrt(L)=32/sqrt(8192)=0.354。
- ideally include tau=d_head/sqrt(L)=64/sqrt(8192)=0.707 if code-level head_dim is 64。

产出：

- PPL@8K/16K/24K/32K。
- in-dist cost。
- single seed acceptable if labeled sanity check。

解释：

- 如果 tau=1.414 wins：supports d_eff convention。
- 如果 tau=0.354 close：convention less critical，but still okay if EVQ shape robust。
- 如果 tau=0.354 wins：must revise MLA tau story。

## 13. Rebuttal Response Snippet Bank

这些不是最终 response，是真正写 rebuttal 时可拼接的语料。

### 13.1 Scope opening

> We agree that the paper should be read as a positional-encoding mechanism study rather than a production long-context recipe. EVQ-Cosh changes the training-time frequency substrate; it is complementary to range-scaling methods and does not replace tuned YaRN, LongRoPE, or LongRoPE2. We will revise the response and manuscript wording to keep this distinction visible next to the main results.

### 13.2 Evidence tiering

> We separate primary stress tests from supporting evidence. The 454M EVQ x YaRN and 432M MLA rows are multi-seed mechanism anchors. The PE-dominant DAPE contrast is diagnostic and seed-scoped. LoRA, video, progressive training, 750M continuation, and the 1B MLA stress check are supporting rows and will not be used to claim broad deployment validation.

### 13.3 LoRA

> The reviewer is right that the original LoRA table could not isolate frequency injection from LongAlign/LoRA adaptation. We added a matched Geo+LoRA control with the same pretrained checkpoint, training data, rank, and adaptation steps. The control separates adaptation cost from EVQ-specific frequency injection; we will report Base, Geo+LoRA, and EVQ-LoRA side by side and attribute only the Geo+LoRA -> EVQ-LoRA difference to EVQ.

### 13.4 1B reversal

> We agree that the 1B MLA row is not evidence of robustness to training saturation. We will relabel it as a single-seed schedule-sensitivity stress check. It differs from the primary MLA result in training length schedule and provenance, and in a scarce-channel regime where allocation mismatch is amplified. The row is therefore a limitation and a motivation for fixed-length continuation/re-warp follow-up, not primary support.

### 13.5 Training amount

> We added total token budgets to the primary tables. We also avoid describing Chinchilla-style budgets as overtraining. The narrower empirical point is that the EVQ signal does not simply vanish with stronger training in the scoped evidence we have: the 8K/500M MLA training progression grows to the final gap while the in-range cost shrinks, the 750M continue@4K row shows a large long-range gap despite low in-range PPL, and the controlled LoRA experiment tests a heavily pretrained checkpoint.

### 13.6 Primary II seed

> We agree that the DAPE-style PE-dominant contrast should be read as a diagnostic, seed-scoped stress test. It asks whether closed-form allocation shape can reduce an extreme 128-to-8K extrapolation gap without learned positional parameters. We will make the seed scope and token budget explicit and avoid presenting it as comprehensive DAPE dominance.

### 13.7 YaRN tuning

> The fixed-s YaRN experiment is a matched-scale substrate test, not a tuned-scaler benchmark. Its purpose is to ask whether the same range-scaling operation has different leverage on a Geo-trained vs EVQ-trained frequency substrate. We agree that a best-of-grid Geo+YaRN comparison is the right tuned-baseline check and will report it if completed; otherwise we will keep the claim matched-scale.

### 13.8 TF PK / AR exact

> We use teacher-forced NLL-gap passkey as a PE diagnostic and will state this wherever PK is reported. It should not be read as autoregressive exact retrieval. Where feasible we will add AR exact as an auxiliary metric; otherwise we will keep the claim on the NLL-gap diagnostic endpoint.

### 13.9 Shape/scale theory

> The variational surrogate gives the cosh allocation family; the deployed scale is selected by a separate softmax-transport operating rule and empirical basin evidence. We will clarify this epistemic separation. The claim is not that tau=d_eff/sqrt(L) is a global optimum theorem, but that it is a robust zero-parameter default inside the tested basin.

### 13.10 Learnable tau

> The learned-tau result is informative precisely because it is worse than the fixed EVQ rule. The in-range training objective does not observe extrapolation benefit, while any in-range cost is visible, so the learned parameter follows the training-loss basin rather than the extrapolation basin. This supports specifying frequency allocation as a prior closed-form schedule instead of relying on the training objective to discover it.

### 13.11 MLA tau convention

> In MLA, d_rope counts the quantized rotary channels, while d_eff is an operating convention for the latent attention pathway. We agree this is not a theorem from d_rope alone. We will make this convention explicit and either add the tau=d_rope/sqrt(L) sanity ablation or list it as a limitation.

### 13.12 Downstream/benchmark

> Downstream accuracy at 454M is capacity-limited and is not the primary endpoint of this mechanism paper. We report it as a non-regression/supporting check and rely on PE-diagnostic metrics for the main mechanism claim. Production-scale benchmark validation remains future work.

## 14. Manuscript Patch Map for Later

不要现在无脑改 paper。等新证据表完成后再动。但以下位置已经明确是 rebuttal 后续修改点。

| File | Likely change | Why |
| --- | --- | --- |
| `paper/tables/table_evidence_tier.tex` | rename 1B row from robustness to schedule-sensitivity/supporting limitation | prevent R2 reversal attack |
| `paper/appendix/a4_supporting_experiments.tex` | replace LoRA table with Base/Geo+LoRA/EVQ-LoRA | close LoRA confound |
| `paper/sections/05_experiments.tex` | add token budgets or refer to table; tighten Primary II seed wording | answer R2 |
| `paper/appendix/a2_experiment_details.tex` | add exact token budget/provenance for primary anchors | reproducibility |
| `paper/appendix/a1_proofs.tex` | remove/define Habitable Zone and Class C2 | presentation hygiene |
| QuALITY figure/table source | fix caption/data mismatch | trust |
| `docs/overview/PAPER_CLAIMS_MAP.md` | sync claim map after rebuttal evidence updates | consistency |
| `docs/overview/REPRODUCE.md` | add exact reviewer-facing commands or provenance | reproducibility |

## 15. Decision Tree

### 如果只能完成 3 件事

1. Geo+LoRA table with exact numbers。
2. Figure/Table consistency correction。
3. Primary I/II token budget/provenance note。

这三件事主要修 reviewer trust。

### 如果还能完成 3 件事

4. Geo+YaRN scale sweep。
5. Primary I AR exact。
6. learned tau trajectory。

这三件事主要修 R2/R1 核心技术质疑。

### 如果有 GPU 预算

7. Geo + Dynamic NTK/YaRN eval-only for LoRA。
8. MLA tau=0.354 sanity。
9. fixed-L continuation curve。

这三件事主要把 rebuttal 从“防守”变成“可上调分数”。

### 如果出现坏结果

| 坏结果 | 不要做 | 正确处理 |
| --- | --- | --- |
| tuned Geo+YaRN 接近 EVQ+YaRN | hide sweep | claim narrows to matched-scale and substrate sensitivity |
| AR exact 弱 | pretend TF PK is enough | label TF as diagnostic, AR as future/auxiliary |
| Geo+LoRA 也有外推 gain | claim EVQ unique | attribute only incremental gain |
| tau=0.354 MLA wins | defend d_eff convention | revise tau convention story |
| fixed-L continuation gap shrinks | say noise | concede durability limitation |

## 16. 后续轮次使用本文件的方式

每次有新审稿意见、新实验结果或新写法，先问三件事：

1. 它落在哪个 issue ledger 项？
2. 它是 primary evidence、supporting evidence，还是 limitation？
3. 它会不会触发第 5 节的反噬措辞？

更新顺序：

1. 先更新第 8 节“需要继续补入的数据”。
2. 再更新第 10 节 issue ledger 的真实状态。
3. 再改第 13 节 snippet。
4. 最后才决定是否 patch paper。

这样做的目的：防止 rebuttal 过程中不断被新数字带偏，变成二次提交。

## 17. 当前完成度审查

这一节用于防止把“已经整理过”误当成“rebuttal 准备已完全完成”。

### 17.1 已经完成

| 目标要求 | 当前证据 |
| --- | --- |
| 输入材料恢复状态 | `rebuttal/raw_sources/00_INDEX.md` 仅保留历史索引；五份 verbatim 文件当前不可用，不能声称本 checkout 已逐字校验 |
| fable 原文逐条阅读 | 本文第 1 节、第 9 节、第 10 节已经按 reviewer/source/issue 拆解 |
| 先认真正改变局面的新证据 | 本文第 1.1 节把 Geo+LoRA 控制升为 P0；`rebuttal/PAPER_ISSUE_AUDIT.md` 的 I1-I2 对应当前论文 LoRA confound |
| 制定 rebuttal 计划 | 本文第 2-4、7、12、15 节给出 reviewer matrix、实验优先级、runbook、decision tree |
| 识别误解与真实硬伤 | 本文第 11 节和 `rebuttal/PAPER_ISSUE_AUDIT.md` 第 2 节已分开列出 |
| 标出反噬措辞 | 本文第 5 节、第 13 节以及 `rebuttal/PAPER_ISSUE_AUDIT.md` 第 4-6 节已列出 |
| 对照当前论文实际问题 | `rebuttal/PAPER_ISSUE_AUDIT.md` 已把核心攻击面绑定到当前 `paper/`、`data/curated/`、`docs/exp/` 和脚本行号 |
| Figure 8/9 rebuttal trust audit | 已确认 submitted-version errors、26.6%→24.6% erratum 和 safe response；没有修改 PDF |
| 1B response scope | author response 明确其为 single-seed schedule-sensitivity limitation |
| Primary token/provenance reconciliation | `rebuttal/PRIMARY_PROVENANCE_NOTE.md` 已确认 Primary I 100M/2048/3 seeds、Primary II Table 4 15M/128 seed scope，并把 Phase 11B 256/100M 分开 |
| LoRA response scope | author response 把两行 LoRA 表写成 post-hoc/supporting，并声明 attribution requires matched Geo+LoRA |

### 17.2 仍未完成，不能假装完成

| 缺口 | 为什么阻止目标完全完成 | 下一步 |
| --- | --- | --- |
| Geo+LoRA exact numbers 未在当前 workspace 里确认 | 这是用户新证据里最改变局面的点；没有数字就不能写最终 Table 23 或 response | 找到或让用户提供 Base / Geo+LoRA / EVQ-LoRA 的 8K/16K/32K PPL、seed scope、rank/steps/data |
| Primary I tuned Geo+YaRN / AR exact 未跑或未确认 | 这是 R2 会追击的核心补证据 | 若 checkpoint 可用，优先 eval-only；否则 response 里明确 matched-scale diagnostic |
| MLA tau sanity 未补 | 最强 systems result 仍有 convention attack | 预算允许跑 `tau=d_rope/sqrt(L)`；否则作为 limitation |

### 17.3 当前下一步入口

后续写 response 或改 paper 时，先读：

1. `rebuttal/raw_sources/00_INDEX.md`：查原文。
2. `rebuttal/PAPER_ISSUE_AUDIT.md`：查每个攻击面的当前论文证据、posture、最小行动。
3. `rebuttal/FIGURE_TABLE_AUDIT.md`：查 Figure 8/Table 21 的已验证不一致和修正路径。
4. `rebuttal/PRIMARY_PROVENANCE_NOTE.md`：查 primary token/seed/protocol 的可写口径。
5. `rebuttal/REBUTTAL_DRAFT_EVIDENCE_SCOPED.md`：当前可直接改成 response 的证据限定草稿。
6. `rebuttal/AUTHOR_RESPONSE_PACKET.md`：最终 author response 的 Path A/Path B 双路径写作包。
7. `rebuttal/AUTHOR_RESPONSE_PATH_B_READY_DRAFT.md`：当前无 Geo+LoRA 数字时的无占位符可审阅草稿。
8. `rebuttal/AUTHOR_RESPONSE_PATH_B_COMPACT.md`：当前无 Geo+LoRA 数字时的紧凑提交版。
9. `rebuttal/REBUTTAL_ACTION_BOARD.md`：决定哪些证据现在能写、哪些必须等数字、哪些不要主攻。
10. `rebuttal/REBUTTAL_CLAIM_LEDGER.md`：最终 author response 前逐句判断可写/条件可写/禁写。
11. `rebuttal/REVIEWER_RESPONSE_SKELETON.md`：逐 R1/R2/R3/AC 写最终 response 时的可填模板。
12. `rebuttal/TABLE23_LORA_WORKSHEET.md`：接收 Geo+LoRA exact numbers 并重构 Table 23。
13. 本文件第 15 节：决定如果只能做 3 件事，先做哪 3 件。
