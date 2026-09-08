# 固定单表与生成迁移：初步执行报告审查及 Codex 指导

**日期：2026-09-04**  
**依据：** `SINGLE_TABLE_FFN_SERVER_EXECUTION_20260904.md` 的本次上传快照；对照上一份 `hybrid_rope_iclr2027_theory_experiment_dossier_20260904.md`。  
**性质：** 结果解释、协议纠偏与下一轮执行建议；不是新的实验结果。  
**证据边界：** 已阅读执行报告和原方案相关章节；没有读取服务器 raw JSONL、训练源码或更新后的 preflight，也没有获得两项单位幅度实验的结果。报告中的运行事实作为报告证据引用，不冒充独立源码/原始产物复核。下文 `[R §标题]` 均指本次执行报告；`[D §编号]` 指上一份研究档案。

---

## 0. 结论：两个目标没有闭合，但不能把所有内容统称为失败

### 0.1 目前真实的研究状态

| 问题 | 已经知道 | 尚未知道 |
|---|---|---|
| 零训练可信单表 | OLMo 上当前 Z 和静态 YaRN 的联合 Native 标准均失败；不是只有 NLL 问题 | 单位幅度能否改善；其他冻结表是否可行；更远完整生成是否可用 |
| 少量训练恢复生成 | Qwen 原生表上的全线性适配改善了受控单证据、绑定等任务的完整输出 | 增益有多少超过格式适配；长输入训练是否必要；Z/Y 上能否获得同样恢复 |
| Native 保留 | 平均 NLL 接近原模型，有限简单任务与报告摘要检查有正向信号 | 统计非劣尚未成立，且有明确的逐实例损失；更广原生能力保留未知 |
| 长度泛化 | N128 在部分 32K 检索格式任务上有效 | Qwen 32K 尚在其配置 Native 窗口内；64K 盲测未完成 |
| FFN 作用 | FFN 与 attention 适配器实际参与了训练 | FFN 必需、FFN 最优、位置故障由 FFN 导致均未被识别 |

**当前最重要的缺口不是没有更复杂的理论，也不是没有更多 GPU。是：一个有真实信号的 N128，尚未与能够回答原问题的关键对照闭合。**

本轮应当：

1. 收完已经登记的两项单位幅度诊断；
2. 重用现有 raw outputs 分离语义、格式、终止，不改旧评分；
3. 加一个紧凑输入训练对照 `N_compact`；
4. 完成同一旧训练配方的 Qwen `Z_transfer`、`Y_transfer`，不再由 double-evidence 的样本不足阻断整条路线；
5. 之后才在冻结规则下确认 Native 与更远长度。

**暂不把 prefix-LM、8B、FFN/rank 消融作为下一轮主线。** 这不是永久否定它们，而是这些动作目前不能代替上述缺失比较。

---

## 1. 先纠正“两个问题似乎都没有解决”的具体含义

### 1.1 零训练问题：当前候选确实没有通过

OLMo 的同一批 Native 数据上：

| 系统 | PPL retention | 普通任务 retention | EOS-weighted retention |
|---|---:|---:|---:|
| 固定 Z | 88.50% | 80.85% | 70.30% |
| 静态 HF YaRN-s4 | 66.32% | 87.28% | 84.06% |

这些是当前真实负结果。不能用历史另一套行、另一种评分器的较高 retention 覆盖，也不能用相对原模型更好的 long NLL 抵销 Native 失败。`[R §Frozen OLMo Z; §YaRN comparison]`

但负结果只涉及这两个具体冻结系统及当前标准。它不证明“单表不可能”，也不证明所有零训练方法都失败。

### 1.2 生成迁移问题：已出现能力信号，但没有测试完关键系统

Qwen 的 N128 使用的是原生位置表，而不是 Z。单证据验证结果为 compact 26/32→26/32、near16K 20/32→26/32、far16K 3/32→24/32。成功要求两个合法证据世界都完整答对并 EOS。`[R §Paired natural validation]`

这说明训练确实改变了完整任务表现，不能归类为“又一次只有 NLL 变好”。但它不回答：

- 换表后的计算能否被恢复；
- 单表的增量价值；
- 跨出模型配置 Native 窗口后的表现。

尤其不能用“OLMo 零训练 Z 的 Native 失败”与“Qwen 原生表 N128 的训练成功”拼成同一个因果链。模型、表、Native 长度和被测目标不同。

### 1.3 三个长度必须一直分开

对本轮 Qwen：

$$
L_{\mathrm{adapt,max}}=16K,\qquad L_{\mathrm{configured\ Native}}=32K.
$$

因此 32K 是**适配训练长度之外**，不是**配置 Native 窗口之外**；64K 才跨出后一条边界。这里不能进一步声称知道所有预训练阶段的精确位置曝光历史。模型卡及本报告支持的是该 checkpoint 的配置/公布上下文窗口。`[R §Small-model screen; §Paired natural validation] [W3]`

### 1.4 不要把未完成当成负结果

Z 在 restoration 第 26 步停下；Y 未训练；FFN placement 对照未做；64K 未做。它们的状态是 **未完成/未测试**，不是失败。

N128 的 32 个 restoration steps 是零梯度恒等步骤；实际改变参数的是后面的 96 个 transfer steps。披露训练成本时可保留 128-step 日程，但不能让读者以为有 128 个有效学习步骤。`[R §Corrected N seed42]`

---

## 2. 本轮最值得保留的现象：单证据的近远差距明显缩小

最新 source-only companion 把跨世界变化限制在同宽 source region，near/far 使用相同 blocks 的交换。这比最初 C/N/F 多消除了一项源区宽度混杂。其计数：`[R §Latest completed source-only guard]`

| 任务 | Native near→far | N128 near→far | 训练对 near 的增量 | 训练对 far 的增量 | far 相对 near 的额外增量 |
|---|---|---|---:|---:|---:|
| 单证据，32 组 | 16→3 | 26→24 | 10/32 | 21/32 | 11/32 = 34.375 个百分点 |
| 绑定，16 组 | 4→0 | 9→6 | 5/16 | 6/16 | 1/16 = 6.25 个百分点 |
| 双证据，16 组 | 1→0 | 4→3 | 3/16 | 3/16 | 0 |

定义完整输出指标上的描述性交互：

$$
I_{\mathrm{near,far}}
=[S_T(F)-S_T(N)]-[S_0(F)-S_0(N)].
$$

这里 `N` 在括号中表示 near，不是原生表模型臂；实现必须用 `layout_near`，不要混名。

**解读：** 单证据信号不仅是 compact 变好后所有条件一起抬高。训练后，严格输出在 far 布局的损失明显缩小。因此它值得继续，而不是因为发现格式因素就全部丢掉。

**限制：** 上表是从报告边际计数计算的点值。没有 raw group-level join，不能给交互编造置信区间；严格输出交互也可能由位置相关的格式遵从改善产生，不能直接叫作 retrieval mechanism。双证据的改善没有在这个读数上表现出额外的 far 优势，也不能凭 3 个成功样本推断多跳推理恢复。

这正是 `N_compact` 对照应当回答的问题：**只学同一批短任务，是否已经足以产生上述长布局增益？**

---

## 3. 严格输出很重要，但不能继续让它混淆三种能力

### 3.1 保留原指标，并行增加机制标注

每个 world 的输出应拆成：

- `semantic_correct`：实际断言的答案符合该世界真值；
- `format_compliant`：符合原 prompt 要求的回答形式；
- `termination_valid`：按该模型/模板的正常终止配置结束，且不因 token cap 截断。

旧主指标保持原定义。对于可拆成上述三项的任务：

$$
S_{\mathrm{strict}}=C\land F\land E.
$$

不能为了抬高分数，把原始严格指标改成 substring 命中；也不能因为语义上正确的句子多了几个字，就用严格分数下降证明模型没有取到证据。

### 3.2 必须在原始输出上建立转换表

对每个相同 `semantic_group_id × world_id × layout`，配对原模型和 N128：

| 旧状态→新状态 | 可以支持的解释 |
|---|---|
| 语义正确但格式不合规→完整合规 | 格式/指令遵从修复 |
| 语义错误→语义正确且完整合规 | 实质内容答案改善，但内部检索/决策机制仍未分离 |
| 正确内容但被截断/错误终止→正常结束 | 终止行为改善 |
| 原来完整正确→现在错误 | 必须保留的回退 |
| 两边均错误，错误类型改变 | 状态变化，不算成功 |

两个世界联合成功率与单世界转换表都报告；不把两个世界、多个长度或多深度当成独立样本。

### 3.3 为什么仅查 gold surface 不够

报告中 NIAH 的原始长输出都包含正确 passkey，且均正常 EOS，强烈指向格式主导；自然任务的新 strict-success worlds 中也常出现旧输出已经含答案字符串。`[R §Completed simple tasks and NIAH]`

但是“提到了 A”可能是“不是 A”“A 或 B”或列出全部候选。对自然 QA 必须审查实际断言，不能用包含字符串替代语义判分。

**执行方式：** 先隐藏模型臂身份，对全部已存在自然验证输出做固定 rubric 标注。允许确定性解析覆盖明确案例；歧义案例进入盲审。保留原输出、标注理由和 ambiguous 类。不要先看哪一种标注会让 N128 增益最大。

这是已有数据的解释审计，几乎不需要新 GPU。因为 rubric 是看过结果后增加，它在当前验证集上属于 retrospective diagnostic；在新 confirmation 集上事先冻结后，才能作为新的确认性指标。

### 3.4 EOS 的结论不能跨任务搬运

OLMo 固定 Z 的 summary EOS 失败是真实的；Qwen NIAH 的 0→100% strict 改善不是 EOS 恢复，因为对应行原本都正常结束。不能把前者用作后者的训练机制解释。`[R §EOS diagnosis; §Completed simple tasks and NIAH]`

---

## 4. 零训练路线：先完成幅度归因，不继续发明表

### 4.1 当前证据定位到哪里

OLMo 33 个摘要中，Native 在 512 tokens 内终止 29 个，Z 只终止 3 个。4 个 failure-conditioned 样本在原 Native terminal prefixes 上，EOS margins 从正数变为负数；放大诊断预算后在 635–727 tokens 结束。`[R §Frozen OLMo Z; §EOS diagnosis]`

这是具体的决策边界位移，不是“从来不会 EOS”。同时 Qasper 的普通质量也下降，所以不能把所有 Native 损伤归因于停止时机。

改动 attention gain 会改变内部 attention 分配和后续 hidden states，不等价于只给最后 vocabulary logits 乘一个正温度。前者可能改变 argmax，后者单独缩放不会改变 greedy argmax。因此不能从 EOS 变化直接推出“最后一层温度有问题”。这是一条计算结构推论，不是报告已经定位了具体层。

### 4.2 两项单位幅度诊断正好够回答一个有限问题

已有登记：保持 Z/Y 频率字节不变，仅将 rotary amplitude 设为 1。报告当前没有结果，必须先收取其完成状态与 raw receipt。`[R §Execution review; §Latest completed source-only guard]`

对任一固定表，某个观测量 M 的变化可以按指定路径精确分解：

$$
M(\Omega_Z,c_Z)-M(\Omega_N,1)
=[M(\Omega_Z,1)-M(\Omega_N,1)]
+[M(\Omega_Z,c_Z)-M(\Omega_Z,1)].
$$

第一项是单位幅度下的表替换效应；第二项是**在这张表上的**幅度边际效应。这个分解不证明两个因素独立，也不消除表×幅度交互。

报告 Z 的 amplitude 为 1.1025857827，Y 为 1.1386294361。因此原 Z/Y 对比不是纯 allocation 对比。这个差异可能是合法历史配置，不应未经 owner 核对就称实现 bug；但必须在方法身份和归因中明确。`[R §Runtime repair; §YaRN comparison]`

### 4.3 有限决策，不进入 gain sweep

- **单位幅度仍无法满足 Native 标准：** 关闭当前候选的零训练联合工作点主张；不要再试 1.02、1.05 等系数。
- **单位幅度改善 Native，但未达到要求：** 这是幅度贡献诊断，不是联合成功。
- **预登记的单位幅度变体达到 Native 标准：** 按事先规则冻结该变体，在未用于选择的确认数据上重新确认 Native，并验证原定 long 生成 endpoint。仍使用同一张表，不切回 Native、不追加新 scale。
- **两种单位幅度变体都可行：** 分别报告，不用已暴露 long 分数挑胜者；也不把它们归为同一方法。

这些动作仍符合“Native 只用于可行性，long 用于冻结后的验证”。但报告使用的 115 行已历史暴露，不能再称它们是盲测。

### 4.4 什么情况下才继续零训练理论研究

本报告没有识别一个新的 Native-only long utility，也没有提供可直接推出新频率配置的方程。若两个已登记幅度对照也失败，进一步 zero-training 理论可以作为独立问题继续，但不应占用本轮主 GPU 排期。

此时单表已证明的价值是有代价的 positional witness，而不是无损部署点。后续训练仍可能恢复它，二者不能互相否决。

---

## 5. 当前最大协议错误：把一个任务族的不足升级为全局停机

### 5.1 Double-evidence 不足到底阻止什么

验证集中只有 4/16 个 double groups 被原模型 compact 完整解出，未达原定数量；候选在该小群体上的 near 也不足。`[R §Paired natural validation]`

它阻止的是：

> 用这组 double 数据，做有充分精度的“已存在多证据能力被运到远处”的结论。

它不自动阻止：

- 单证据上测试迁移；
- 检查固定 Z 下的 Native restoration；
- 运行同数据同预算的 Z/Y 对照；
- 在全体 double 结果中描述任务表现；
- 用一个合法但有背景困难的任务评价系统效果。

### 5.2 区分“机制实验不够干净”和“方法评估无效”

原模型 C 成功、near-full 失败，意味着远端失败不能单独归因于距离。但只要任务真值、数据完整性、输入格式和评分有效，它仍然可以用于评价“模型能否在长背景中答题”。背景鲁棒性本来就是实际能力的一部分。

纯距离机制需要更严格的原模型 `compact_correct AND near_correct` 子集；方法总体效果不必要求原模型 near 全会。

### 5.3 对上一版方案的明确修正

上一版 `[D §10.5]` 的“compact 可解实例不足则停止迁移训练”写得过于全局。这使科学上合理的资格检查变成了阻断所有学习问题的开关。

应改为：

$$
\boxed{\text{可解性/精度不足限制对应任务族与对应主张，不默认停止其它已可测问题。}}
$$

真正需要全局停机的是：数据真值错误、label leakage、运行/加载错误、冻结参数违反、cache 错误、数值异常、无法区分数据 split 等。

### 5.4 修正不是“删掉难题就算成功”

必须发布新协议版本，保存旧三任务协议的 `UNRESOLVED_CONTROLS` 判定。当前所有 double 原结果仍在报告中。单证据成为新确认研究的主问题，应在打开新的确认输出之前登记；不能追认旧的三任务联合成功。

不需要重新挑一批模型更容易做对的验证样本，也不需要反复改模板直到 compact 通过。条件化结果与未筛选结果同时保留。

---

## 6. 第一项新训练：紧凑输入对照 N_compact

### 6.1 唯一问题

> 同一批合法短任务的适配，是否已经足以解释 N128 的远端完整生成增益？

这个对照比现在就加 prefix-LM 更有辨别力：若短任务适配已恢复长格式，大量所谓长训练收益不需要长输入；若它不能复制 N128 的语义 far 恢复，才有长输入适配的直接增量证据。

### 6.2 固定设计

| 项目 | N128 / N_transfer | 新 N_compact |
|---|---|---|
| 原 checkpoint 与 Native 表 | 相同 | 相同 |
| LoRA modules/rank/alpha/init | 原登记值 | 完全相同 |
| 语义实例、两种世界、目标答案 | 原 128 组 | 完全相同 |
| task 视图次数/顺序 | 768 | 768 |
| 每个视图的输入 | 原 compact/8K/16K | 替换为对应 compact 输入 |
| 答案/EOS label 序列 | 原值 | 完全相同 |
| CE、margin、replay、dual 日程 | 原已完成配方 | 完全相同 |
| transfer optimizer steps | 96 | 96 |
| prefix-LM | 关闭 | 关闭 |
| 检查点选择 | 按新比较预先冻结 | 相同规则 |

同一个 compact 世界因此会重复出现三次，这是刻意匹配语义/目标曝光，不是生成新的独立样本。

**重要：** 它匹配 optimizer updates、语义曝光与输出监督，不匹配输入 token 数和 FLOPs。长背景与距离同时改变，因此该对照识别“长输入 exposure 的增量”，不单独识别相位、干扰数量、文本风格的贡献。少输入带来的低成本正是这个对照的工程意义。

### 6.3 对照有效性

使用 report 中已通过数值修复的固定引擎。不要从 N128 继续训练；从相同原始 checkpoint 重新开始。Native replay 的 source IDs、顺序、位置、组轮转和 dual 规则必须匹配，不临时加 teacher trajectory replay 或更大 Native batch。

N_compact 的 restoration 是同样的恒等状态；可保留原日程，不把这些零更新解释为额外学习。保存并验证相同的 merge/reload 行为。

### 6.4 读取结果

在同一冻结验证集上同时比较：

$$
S_{\mathrm{strict}},\quad S_{\mathrm{semantic}},\quad F,\quad E,\quad I_{\mathrm{near,far}}.
$$

还要比较来源证据变化后两个世界是否各自答对，不能只看某个世界。

- **N_compact 与 N128 效果接近：** 当前长增益可能主要是任务/格式规则学会后自然迁移。差异不显著不等于已经统计等价；应报告配对差和区间。
- **N128 的 far 语义表现优于 N_compact，compact 表现相近：** 长输入适配具有真实增量。仍不是 FFN 或 RoPE 必要性证明。
- **只在严格格式上拉开：** 支持长布局下的输出遵从适配，不支持检索恢复。
- **N_compact Native 保留更好，长效果不逊：** 它是有价值的低成本参照；不能为了维持原叙事藏掉。

不要据这一次比较重新扫训练长度比例或 compact 重复次数。

---

## 7. 然后补齐真正回答单表适配问题的 Z/Y

### 7.1 最小矩阵

已有 `N0`、`N_transfer`，补足：

| 固定位置系统 | 无 adapter | 同一 transfer 配方 |
|---|---|---|
| Native | N0 | N_transfer，已有 |
| 既有 Z | Z0 | Z_transfer，待完整运行 |
| 官方静态 Y | Y0 | Y_transfer，待完整运行 |

所有行必须针对**同一个 Qwen checkpoint**。OLMo 的 Z0/Y0 数值不能填到 Qwen 行里。

原有 `N_compact` 是对训练机制的对照，不是新的位置候选。当前阶段不再加更多表、gain 或 rank。

### 7.2 Z 第 26 步为什么应从头重跑

报告明确说 serialized-resume parity 尚未解决，正式结果依赖 uninterrupted 日程。因此不得把 Z26 直接接着跑并称与旧协议等价。使用固定已验证引擎，重新开始一次完整 R32+T96；保留旧 Z26 为部分诊断产物。`[R §Numerical invalid run]`

该重跑是恢复原定比较，不是看到差分不理想之后追加训练。

### 7.3 先复用旧 loss，不同步加 prefix

Z/Y 使用 N_transfer 同一已完成配方。不要在补 Z/Y 时顺便改 Native replay、训练目标、单位幅度或数据构造，否则 N/Z/Y 的差异同时包含多个训练因素。

单位幅度实验属于 OLMo 零训练诊断；它的结果不能未经新协议就替换 Qwen 主训练臂的 gain。

当前 Qwen 的确切 Z/Y 表、scale、amplitude 必须从该模型 manifest 读出。本报告不足以把 OLMo 的 tensor 或 gain 数字直接转用给 Qwen。

### 7.4 Restoration-only 保存点有实际信息

R32 保存点回答“只用 Native 功能匹配，能修复多少换表损伤”，不需要另开新训练臂。

对 Z0、Z_R32、Z_transfer 分别报告 Native 与相同任务指标；Y 亦然。以最终 absolute Native distance 为准：

$$
D_N = D\big(p_{\theta_0,\Omega_N},p_{\theta_0+\Delta\theta,\Omega_{arm}}\big).
$$

不能只看 `Z_transfer` 相对已损伤的 Z0 没变差。

### 7.5 哪些结果才决定下一步

- **Z_transfer 恢复 Native，且完整长生成相对 Z0 改善：** 单表＋适配的联合工作点出现，进入确认。
- **Y_transfer 同样有效或更好：** 说明主要收益可能来自适配而非特定 Z。它仍是工程正结果，但削弱 Z 的方法创新主张。
- **N_transfer 与 Z_transfer 在 16K 相近：** Qwen 16K 在配置 Native 窗口内，该结果不决定更远收益；冻结后再看 64K。
- **Z/Y 在固定预算都无法恢复 Native：** 当前 adapter/目标/预算的联合路线未通过，不叫 zero-training 类别不可能，也不自动加 steps。
- **任务变好、Native 明显受损：** 记录 trade-off；不称两大目标已同时解决。

局部 Native CI 未足够窄，可以限制确认性结论和大规模扩展，但不应再次阻止这一最小矩阵完成。

---

## 8. Prefix-LM：有合理性，但不是报告已经推出的答案

### 8.1 3,642 个 labels 能说明什么

报告给出 768 views、3,642 个 answer/EOS labels、6,329,249 个 prompt+target tokens。由此：

$$
\text{平均答案标签数}=3642/768\approx4.74,
$$

$$
\text{标签位置占比}=3642/6329249\approx0.0575\%.
$$

这说明监督集中在很短的回答与终止轨迹。它**不说明其它 tokens 没有梯度**：答案 loss 可以通过 causal attention 对先前上下文的计算路径反向传播。

输入 labels 稀疏、参数梯度弱、远程 credit assignment 不足是三个不同命题。现在只有第一个被计数直接支持。

### 8.2 采样 prefix loss 的正确数学解释

对有效 prompt 位置集合 V，大小 L；从中均匀采 M 个位置：

$$
L_{prefix}=\frac1L\sum_{t\in V}\ell_t(\phi),\qquad
\widehat L_{prefix}=\frac1M\sum_{t\in S}\ell_t(\phi).
$$

在固定参数、均匀采样下：

$$
\mathbb E_S\widehat L_{prefix}=L_{prefix},\qquad
\mathbb E_S\nabla\widehat L_{prefix}=\nabla L_{prefix}.
$$

这与报告的限定相容，但不等于有限训练轨迹等于 dense training。若讨论每步条件无偏，还需要采样与当步历史的相应独立性；反复复用已经影响过参数的固定 mask，不能仅凭固定参数恒等式获得整段 SGD 轨迹保证。

view-normalized loss 不等于 corpus token-weighted loss。加入 `0.1 × mean(prefix CE)`，也不意味着 128 个 prefix labels 自动获得相对约 5 个 answer labels 的 25 倍总权重。权重由归一化、系数及梯度决定。

### 8.3 它同时改变了目标，不只是“密度”

从 answer-only 改成 answer+prefix，不只是提高对同一目标的采样精度，而是加入了一个新的预测任务：预测背景文本。

因此正结果应称为“辅助 prefix LM 目标的增量”，不能直接称“证明了监督密度是瓶颈”。若要纯粹识别密度，需要在同一个 prefix 目标下比较采样精度；这不是当前建议新增的项目。

### 8.4 为什么它可能帮忙，也可能不帮

可能的作用：为长背景中更广的位置提供分布式语言建模信号，帮助换表后的内容处理适应。

限制与风险：

- 若失分主要来自 answer-only 格式，预测 haystack 中间的词不直接教会答案格式；
- 大量 prefix targets 可以凭局部上下文预测，不一定需要远程证据；
- 在 N 臂没有换表的情况下，真实 token CE 在原模型处仍可能有非零梯度，不能被当作保持原函数的 KL；
- 随机拼接边界、角色标签、padding 若被监督，会改变“自然语言建模”的含义；
- 如果证据只占很小比例，128 个均匀位置对证据附近直接目标的覆盖可能很少，但这仍不等于答案路径无法提供梯度。

### 8.5 文献只支持这个方向合理，不提供当前协议的充分性

YaRN 的公开结果支持位置扩展结合训练能够奏效，以及存在适配长度以外泛化的情况；不能由此推出“在当前 6M-input-token 小数据 PEFT 上加 0.1 prefix CE 就会成功”。`[W1]`

LongReD 明确区分换位置编码后的分布漂移与持续训练遗忘，并采用 long LM 加恢复蒸馏；其 skip-position 训练也不是严格未见目标位置的设置。这支持问题拆分，不是当前单表协议的保证。`[W2]`

### 8.6 当前决定及后续唯一可检验版本

**当前不立即正式运行 prefix companion。先完成 N_compact 与 Z/Y 的旧配方。**

若之后仍需要判断 prefix 是否修复换表后的背景计算，使用固定的 `N/Z × prefix off/on` 小型比较，off 复用完整旧结果，最多只增加 N+prefix、Z+prefix 两个单 seed run；其余设置不变。不把新模型、更大 rank、更大 replay 或新 loss 同时加入。

检查：

$$
J=[S(Z,+P)-S(Z,-P)]-[S(N,+P)-S(N,-P)].
$$

必须同时看绝对分数、语义/格式拆分和 Native；floor/ceiling 可以影响交互。若只改善 prefix NLL、不改善完整任务与 Native，不能扩成主路线。

这是一项以后可登记的比较，不是本轮自动执行队列。

---

## 9. Native retention：从“大概没忘”改为精确记账

### 9.1 现有结果既不是全面遗忘，也不是无遗忘

报告 Native 三类 generated tasks 合计：

- 原模型正确 35+5+9=49/192；
- N128 正确 36+3+7=46/192；
- 4 个原正确样本丢失，1 个新获得。

因此平均分数比为 46/49=93.88%，但原正确实例保留率为 45/49=91.84%。它们不是一个量。总体下降 3/192=1.5625 个百分点；推理组 5→3 的小计也必须保留。所有比例仍受低原始成功数和来源聚类限制。`[R §What the low Native score means]`

PPL 接近原模型不能消除这些失败；反过来，4 个丢失也不能证明广泛 catastrophic forgetting。

### 9.2 本轮 Native 约束实际接触了什么

N 的 R32 是恒等零更新，所以前 256 个 restoration replay 样本并没有保护已经改变的模型。T 阶段只用了 192 个 Native replay 样本，四组各 48。`[R §Adaptation]`

18,464,768 个可训练参数占该 Qwen 总参数约 1.20%。这不是“18M 自由参数已被充分约束”的证据；实际功能变化还受低秩结构与目标约束影响。计数的用途是提醒覆盖有限，不是宣布必然过拟合。

### 9.3 Source-truth prefixes 与原模型行为轨迹不同

当前 KL 在 source-truth prefixes 上计算。它保留的是这些给定 prefixes 的下一 token 分布，不自动保护原模型实际会走到的所有答案/终止状态。`[R §Adaptation; §Independent subagent review]`

对于有限长度、EOS 后进入吸收态的完整随机输出分布，链式法则给出：

$$
D_{KL}(P_0\|P_\phi)
=\mathbb E_{Y\sim P_0}\sum_t
D_{KL}\big(p_0(\cdot|X,Y_{<t})\|p_\phi(\cdot|X,Y_{<t})\big).
$$

这个式子里的 prefixes 来自 P0，而不是任意 source-truth 序列。即使换成原模型 greedy trajectories，也只是直接保护确定性部署常走的路径，不是整个随机 P0 的无偏覆盖定理。

**下一步首先审查覆盖，不立即改训练：** 对已报告的 4 个 Native 丢失样本，以及已有 summary EOS 失败样本，读取 teacher 原实际终止/决策 prefixes 上的 margin 和 KL；这些结果只用于解释，不能把验证失败样本回灌训练后仍称它们 held-out。

若以后确实需要改 retention，原模型真实轨迹覆盖比盲目增加自然背景 prefix CE 更直接针对“原来会生成的答案/终止是否保住”。但不能在本轮 Z/Y 对照中偷偷替换 replay 来源。

### 9.4 不要由梯度比例推断 KL 阻塞学习

报告的 Native/task gradient norm ratio 约 16–17，cosine 为正。范数大不等于冲突；在简单梯度下降的一阶近似下，正内积甚至意味着沿一个 loss 的下降方向有利于另一个 loss。

实际 Adam 更新还受预条件、系数和历史矩影响，真正相关的是对实际更新 u 的方向导数 `g_task^T u` 与 `g_native^T u`。没有这些证据，不应为了“解放学习”减弱 KL 或提高 LR。

本轮不要新增高成本逐步全梯度日志。已有日志能回答多少就回答多少；如以后需诊断，只在预定少数 steps 记录，不更改正式优化轨迹。

### 9.5 确认的时机

现有 retention CI 下界未达到报告 88% 门槛，只能说未证明非劣。当前验证结果可以供有限 checkpoint 选择，但不能反复打开 confirmation pool 来调方案。

完成当前最小矩阵、冻结候选与评分之后，一次性使用未见 Native confirmation 数据，并包括短输入和该模型较宽 Native 窗口。保留既有阈值；报告还未提供更新 preflight 全文，Codex 应从已登记版本读取精确标准，不用上一稿不同阈值或新的宽松数字代替。

---

## 10. 完整生成与计算瓶颈：哪些理论仍然成立

### 10.1 正 margin 对 greedy 的充分性没有失效

对固定 prompt、固定表、固定 decoder，以及一条合法完整答案轨迹（含正确 end-of-turn token），若每个 gold prefix 下正确下一 token 严格大于其它 token，则 greedy 会按归纳生成该轨迹。

但这个结果不意味着所有 canonical wording 的 token mismatch 都是语义失败。一个正确长句可能偏离“只答数字”的 canonical 轨迹；那是格式约束还是语义错误，需要独立判定。

### 10.2 当前不能把问题定位到 FFN

FFN 与 attention 参数都改变了，任务成功只证明该复合参数空间能找到有用解。报告提出的 attention-only r68 与 all-linear r16 近似等参数量，只能识别预算分配的相对效果，不证明 FFN 是逻辑必需。

因此不应将当前项目叫作“FFN 修复已证实”。更准确的名称是“固定位置系统下、Native 约束的全线性适配”。

### 10.3 真正的算法瓶颈仍需区分

目前可能包括三类：

1. 证据内容没有被正确利用；
2. 已形成合适内容答案，但选择/表述方式违反任务要求；
3. 答案与终止轨迹在长度和背景变化后失稳。

本报告对第 2 类提供了强证据，对部分第 1 类提供了候选案例，对 OLMo 第 3 类提供了具体 EOS-margin 证据。它没有证明三者都由同一模块引起。

不要将“检索头有效”“gain 影响 EOS”“FFN 有非零梯度”拼成一个未经干预验证的因果机制。

---

## 11. 明确的执行顺序与预算边界

### 阶段 A：整理已有产物，不重建整个实验体系

产物：

- `observed_claims.csv`：每条主张对应模型、表、数据、scorer、receipt；
- `paired_outcome_transitions.jsonl`：语义/格式/终止的逐实例变化；
- `native_lost_gained.jsonl`：保留 4 个丢失和全部新增实例；
- `amplitude_component_report.md`：两项已登记运行的真实结果，含未完成状态；
- `protocol_delta_v2.md`：把 double 精度不足从全局停机改成局部范围限制，保留原判决。

不得为了这一步增加十个 canaries。已有加载、cache、KL 一阶梯度与 merge/reload 证据在其测试范围内继续有效，除非新代码触及对应边界。

### 阶段 B：一个 N_compact 新训练

从相同原 checkpoint 开始，原配方、同一 seed、同样输出目标曝光，改变的唯一实验处理是输入由长布局变 compact。训练后不进行新式 decode rescue。

### 阶段 C：两个完整训练 Z_transfer / Y_transfer

完整固定预算，Z 从头重跑，不 resume Z26。完成 seed42 的三位置系统矩阵之后，才决定是否复制第二、第三 seed。

### 阶段 D：冻结后确认

主确认优先一项自然单证据 QA、声明清楚的检索辅助指标，以及可靠 Native 保留。绑定与双证据保持报告，不假装其原研究已通过。

先冻结待比较 checkpoints、表、gain、scorer、样本清单，再运行更远 natural transfer；32K 与 64K 的结论分别写清。没有新目标位置曝光、没有按结果换表。

若一个系统 Native 明显失败，可以不支付其大规模远程确认费用，但仍保留其完成的比较结果。CI 不够窄与数值/方法失败分开。

### 本轮明确不做

不换 8B；不改变 rank；不加 FFN placement 消融；不扫 gain；不重设计 frequency；不追加 EOS-only 或 rerank；不边看 32K 结果边改 64K 配置；不为了得到自然双证据通过率重复改题。

N128 已完成训练耗时约 28.7 分钟，是该已有 run 的测量，不是其它 runs 的时长保证。N_compact 输入更短，但固定开销不同；Z/Y restoration 有真实梯度，时间也可能不同。GPU 预算用各阶段实测 step time，不用 smoke 的平均值假定完整运行的峰值显存和时间。`[R §Corrected N seed42; §Latest completed source-only guard]`

---

## 12. 核心伪代码

### 12.1 在旧输出上分离指标

```python
for group in fixed_validation_groups:
    for world in (0, 1):
        for layout in ("compact", "near", "far"):
            old = load_raw("N0", group, world, layout)
            new = load_raw("N_transfer", group, world, layout)

            # Keep old scorer and raw strings unchanged.
            old_strict = registered_score(old)
            new_strict = registered_score(new)

            # Rubric is frozen, model identity blinded.
            old_tags = blinded_semantic_format_eos_review(old, truth[group, world])
            new_tags = blinded_semantic_format_eos_review(new, truth[group, world])

            save_transition(group, world, layout, old_strict, new_strict,
                            old_tags, new_tags)

# One bootstrap sample draws semantic/source groups, not worlds or length views.
report_unfiltered_and_baseline_qualified_subsets()
report_group_paired_deltas_and_uncertainty()
```

### 12.2 N_compact 对照

```python
student = load_same_native_base_and_zero_output_lora()
install_native_table_and_gain()
assert optimizer_and_kl_identity_contract()

run_registered_identity_restoration_schedule()
reset_optimizer_as_registered()

for step, original_batch in enumerate(original_transfer_schedule):
    compact_batch = [
        compact_view_for(v.semantic_id, v.world_id)
        for v in original_batch
    ]
    assert same_target_token_ids(compact_batch, original_batch)

    task_loss = original_task_objective(compact_batch)
    native_batch = original_replay_batch(step)
    native_constraints = original_native_kl(native_batch)

    update_with_original_primal_dual_rule(task_loss, native_constraints)
    save_only_registered_checkpoints()

# Equal target exposure and updates, not equal FLOPs/input tokens.
verify_merged_generation_and_evaluate_registered_views()
```

### 12.3 补齐 Z/Y，而不是换一种实验

```python
for arm in ("Z", "Y"):
    model = load_same_qwen_original_checkpoint()
    install_exact_qwen_arm_table_from_manifest(arm)
    attach_same_all_linear_lora()
    assert_no_dynamic_table_or_gain_changes()

    # The existing Z26 was stopped; do not resume an unverified optimizer state.
    run_uninterrupted_registered_R32_T96(prefix_lm=False)
    save_and_evaluate_same_registered_checkpoints()

    report_native_distance_to_original_Native()
    report_all_families_with_semantic_format_eos_breakdown()
    report_double_evidence_as_underqualified_where_applicable()

# Do not change main gain using OLMo amplitude-diagnostic outcomes.
```

### 12.4 前瞻停止规则

```python
if data_truth_invalid or leakage or runtime_identity_invalid or nonfinite_loss:
    stop_affected_runs_and_repair_instrument()
elif double_family_underqualified:
    retain_double_outputs_and_limit_double_claim()
    continue_resolving_single_evidence_and_table_comparisons()
elif completed_small_matrix_has_no_native_feasible_candidate:
    report_current_protocol_tradeoff_or_failure()
    do_not_launch_large_confirmation_or_parameter_search()
else:
    freeze_candidate_hashes_scorers_and_confirmation_protocol()
    run_independent_confirmation_once()
```

阈值与预算从最新已批准 preflight 读取，不由这段伪代码创建新的隐含标准。

---

## 13. 对论文的直接影响

### 13.1 现在能写什么

可以写成内部研究观察：同一位置干预下，NLL、语义答案、格式遵从与终止并非同一指标；全线性小数据适配可以改变受控自然来源任务的完整生成；已有位置底座的 Native 兼容性必须独立验证。

不能写：无损单表已解决、FFN 必要、Qwen 16K 超越 Native、严格 NIAH 0→100% 是取证能力从零恢复、官方 2Wiki 指标达到某数。

### 13.2 哪个结果最能提高这篇论文的可信度

不是再增加一个大幅提升但主要由格式造成的 strict 数字，而是同时回答：

- 固定 Z 相对正常强基线的增量在哪里；
- 相同训练配方能否在实际部署表上恢复 Native；
- 长输入训练相对 compact-only 的增量在哪里；
- 在未用于选择的来源和更远长度上，语义正确的完整输出是否仍提升。

如果 N_compact 基本复制 N128，仍然是值得保留的负向识别：之前的“长训练机制”被简化为更低成本的任务适配。若 Y 与 Z 相当，也应收缩 Z 的方法贡献，把核心放回 fixed-support identification 与有序耦合，不把普通适配包装成新位置理论。

### 13.3 论文主线不要被当前任务重新吞掉

本轮是对原有 RoPE allocation 研究的应用闭环，而不是必须另造一篇 FFN 论文。若当前固定 Z 没能给出联合工作点，不应为了故事完整继续烧预算；有边界的负结果可以约束理论，但不能单凭这种负结果替代足够的正向方法或因果发现。

两种 checkpoint 的证据应在最终表格中按各自 Native 长度、操作系统、参数化和训练预算分别显示。跨模型复现不能靠移动分母或只报告成功模型完成。

---

## 14. 给 Codex 的执行摘要

> 本次报告有真实的全输出改善，但两个原目标尚未被关键对照闭合。不要默认把 prefix-LM 加入后再跑全部矩阵，也不要因 double-evidence 只有 4 个 compact 合格组而继续停止所有 Z/Y 工作。
>
> 首先收取已登记的 Z/Y 单位幅度结果；报告当前两张 OLMo 固定表的联合 Native 失败，禁止拼接旧分数。随后重用全部现有 raw outputs，保持原严格评分不变，并盲化标注语义、格式、终止及逐实例 lost/gained。当前新标注只作为 retrospective diagnostic。
>
> 新增的第一项训练仅为 N_compact：与 N128 同 base、表、seed、LoRA、输出目标、768 次语义曝光、96 个 transfer updates 和相同 Native replay；将各长视图换成对应 compact 视图，不加 prefix。它匹配监督与更新，不匹配输入 tokens/FLOPs。
>
> 发布新 protocol delta，把 double 不足限制为该任务族及纯距离机制的证据范围。保留原 UNRESOLVED 判决与全部 double 输出。然后用原已完成配方，完成 Qwen Z/Y 的 seed42；Z26 不在 resume 问题未解决时直接续跑，而从同一原 checkpoint 完整重跑一次。
>
> 比较 N0/N_transfer/N_compact/Z0/Z_transfer/Y0/Y_transfer 的实际 Native 和相同 near/far 生成。N128 在 Qwen Native32K 内的16K成功不叫 Native 外推。更远测试只在表、adapter、scorer和样本冻结之后一次打开，不据32K结果修改64K。
>
> 当前不做8B、rank/head选择、FFN必要性消融、gain sweep、EOS-only、rerank或新曲线。prefix-LM是后续一个独立受控目标增量，而不是本报告已经证明的修复。报告每次必须给出新增比较回答的问题、真实完成状态、absolute Native代价、语义/格式/EOS分解及停止范围。

---

## 来源与使用范围

**R：** `SINGLE_TABLE_FFN_SERVER_EXECUTION_20260904.md`，本次上传，文件 ID `file_00000000a31881f5b1f24dc8ce9e17e5`。重点章节：初步结论、Frozen OLMo Z、Adaptation、Corrected N seed42、Paired natural validation、Native score、simple tasks/NIAH、YaRN comparison、prefix smoke、source-only guard。所有实验计数来自该快照，后续结果需另立版本。

**D：** `hybrid_rope_iclr2027_theory_experiment_dossier_20260904.md`，文件 ID `file_00000000009c81f5b9545eb365925a90`。使用 §8–10 核对训练日程、实际部署表下 retention、compact 资格检查和停止规则。执行报告引用的更新版服务器 preflight 未包含在本次附件中；对其精确阈值不作推测。

**W1：** Peng et al., *YaRN: Efficient Context Window Extension of Large Language Models*, arXiv:2309.00071。用于核对训练式扩展和适配长度外泛化的公开背景；不把其效果推演成当前 prefix companion 的保证。

**W2：** Dong et al., *LongReD: Mitigating Short-Text Degradation of Long-Context Large Language Models via Restoration Distillation*, arXiv:2502.07365v3。用于核对分布漂移/训练遗忘的区分、long LM与蒸馏组合及目标位置采样边界；本文未复制其方法。

**W3：** 官方 `Qwen/Qwen2.5-1.5B-Instruct` 模型卡。用于核对该模型公布的 32,768-token context 与模型身份；不是完整预训练位置曝光史。

**本文计算/推导：** near/far 描述性交互、平均 labels、逐实例保留率、prefix 均匀估计恒等式、序列 KL 链式分解与 argmax 说明。它们的假设和有限用途已分别声明，不属于新增 GPU 证据。
