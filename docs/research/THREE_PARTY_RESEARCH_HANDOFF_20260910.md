# PC2 / PM 综合交接：核心任务与第三方分析

交接日期：2026-09-10 UTC。本文汇总两个任务已经形成的研究结论、数学依据、实际结果、纠正和未完成工作，供单一实验负责人直接接续。它替代继续跨任务追问进度；逐轮原始记录仍保留。

**后续更新：**第6节的问题状态交叉已经完成，随后的问题读取mask及双标记拆分也已完成。最新的理论、实验与失败问题裁决见[32项问题清单](PC2_PM_FAILURE_QUESTION_AUDIT_20260910.md)，不要将下文保留的交接时计划重新排队。

| 角色 | 任务 | 交接状态 |
| --- | --- | --- |
| 核心方法开发 | [核心实验agent](codex://threads/01a08310-17d8-7112-877a-060616b74dee) | 用户报告会话卡住；本次读取为 idle，最后一轮 interrupted。其研究与准备工作合并如下 |
| 第三方分析与核验 | [解决昨晚两个实验问题](codex://threads/01a0887e-696c-78c3-ba30-75ef76e09845) | 完成本交接后结束；不再启动实验或发送后续研究建议 |
| 接续执行与决策 | [监控今晚实验并持续决策](codex://threads/01a0870a-cc6e-7262-ae4c-300394ae57b3) | 唯一 GPU 调度负责人，独立完成已有实验并继续决策；无需回复其他两个任务 |

## 接管者先读：让下一轮累积已有成功

用户补充要求：每轮要回答为什么失败、为什么成功、具体缺什么，并据此改变下一项，避免继续漫无目的地排列变体。当前最缺的是**把已经有答题效果的干预转成可部署构造**。已有事实和下一项之间应按下表衔接。

| 保留下来的成功 | 已失败的简化及其教训 | 现在确切欠缺什么 | 接续决定 |
| --- | --- | --- | --- |
| PC2完整协方差恢复5个输入相对强对照的得分 | radius-tail、Frobenius-tail没有保住收益；小的整体误差也可能伤害关键查询方向与排名 | 同预算小摘要能否保住导致回答恢复的有效选块；并实际省下全K扫描成本 | 从已保存状态核查关键cutoff与成本；不再仅因另一种norm降低就启动生成 |
| KeyDiff在自然16记录8/8，P只有1/8 | key-novel采样、KVzip固定预算适配仍失败；“换Q代理”未接回成功保留行为 | 后续问题真正需要什么，以及哪些保留差异决定回答 | 保留K作强参考，停止继续微调相同P采样家族 |
| K_pre修复重复背景两例 | 同时伤害自然背景一例，而且该例目标value覆盖更高 | 有益信息与竞争信息、问题状态形成的作用，无法由一个原文覆盖率决定 | 保留2胜1负，拒绝按已知背景事后拼接K/K_pre；不推进大确认 |
| 同预算target oracle救回002，随机sham未救回 | 003完整目标已在所有层/head里，仍返回key | 在这两例中，完整目标与问题阶段形成的K/V状态是否需要共同满足 | 下一项只做下文2×2问题状态交叉，最多4次新自由生成 |

**本次新增的明确工作交付：**接管者自行实现并执行第6节的问题状态交叉，不需要等待核心恢复或再索要一个研究方向；执行仍由接管者掌握GPU。K_pre24条已完整结束，现有GPU配置已无待跑项，不能再写“等待当前K_pre完成”。本任务只完成存量结果分析与交接，不启动该新GPU实验。

成功累积在可复用的原始输出、有效控制和被区分的原因上；不要求每个新变体分数天然递增。一次新比较若不改变可部署设计的取舍，或只是重现已知失败，就没有接续价值。接管者先运行这一项，再按结果分支推进，不能同时排入多个新候选。

## 1. 当前结论与真正要解决的问题

**已有可复查的精度修复和因果证据，但尚无新的低成本 SOTA 方法，也没有足以写成成功论文主结果的独立验证。** PC2、PM 和模型原生的顺序读取错误是不同缺口，不能强行归并成一个已经解决的根因。

- **PC2：完整协方差恢复了这批 DEV 中大部分近似损失。** 32 个输入相对本地 COBS 适配、rank1 各有 5 胜、0 负；但直接精确打分在当前实现中仍更快、略更准。两种单 key 摘要和缓存完整协方差尚未兑现更好的质量与成本取舍。
- **PM：前缀代理评分的收益没有稳定转移到真实问题需求。** 类型均衡在重复背景有效，换成自然背景即明显失效；key 几何采样和本次 KVzip 固定预算适配均未修复。已有实际保留集合与真实 Q 证据支持需求错配，不能只靠增加 H/M 解决。
- **保留错误有具体因果作用，但并非充分解释。** 同预算补回完整目标记录救回 002、未救回 003；003 输出问题中的 key，不能称为已恢复检索。Full 自己也存在要求 first 却返回 latest 的错误。

最新中心问题是：**在已经支持的窗口内、固定稀疏记忆与读取预算，位置计算能否更好地保留重复事件的区别、时序关系及后续问题需要的信息？** 稀疏选择、聚合压缩、递归状态是不同接口。被保留 token 对之间的 RoPE 相对旋转恒等式仍成立；需要改善的是完整信息路径。此前的动态长度或密集模型外推候选不能自动替代这个目标。

当前 claim、原始提案与架构边界由 [SPARSE_POSITION_CLAIM_DISCUSSION](SPARSE_POSITION_CLAIM_DISCUSSION_20260909.md) 负责；本次交接不把该文早期的暂停 GPU 状态当作当前运行状态。

## 2. 核心任务已建立的机制框架

### PC2：二阶信息丢失与高阶响应分开

令块内 `w=softmax(CIS)`、`a=q/√d`、加权均值为 μ、协方差为 Σ。准确块 logmass 是

\[
L=\log Z_{\mathrm{CIS}}+a^T\mu+\log\mathbb E_w\exp(a^T(k-\mu)).
\]

pair-PC2 以 `½ aᵀ pairdiag(Σ) a` 代替最后一项，留下两项误差：

\[
\tfrac12a^T[\Sigma-\operatorname{pairdiag}(\Sigma)]a,
\qquad
\log\mathbb E_w e^{a^T(k-\mu)}-\tfrac12a^T\Sigma a.
\]

第一项是跨 pair 协方差缺失，第二项是完整二阶也不能表示的响应。完整协方差真实生成恢复精度，给第一项的任务影响提供了证据；同状态诊断也发现第二项，因此不能宣布一切失败均由 cross-pair 或指数尾部单独造成。

rank1 加 pair 残差没有单调改进保证：被丢弃的负相关可能原本抵消保留下来的项。已有合法两块 CPU 反例中，plain PC2 排名正确而 rank1 错排。添加更多 rank 之前应明确它要保留哪种与实际查询相关的差异。

### PM：平均位置、真实需求与回答是三层问题

对相同内容 query 分布和预算，P 先在每个位置计算 softmax 再平均，C 先平均旋转再计算 softmax：

\[
\pi_j=\mathbb E_{u,t}\operatorname{softmax}_j(KR(t)u/\sqrt d),\quad
\pi^C_j=\mathbb E_u\operatorname{softmax}_j(K\bar Ru/\sqrt d).
\]

这个差异确实可能改变保留集合，但后续结论有三个必要区分。

1. **只有 logit 差会影响选择。** 大量旋转方差可能只是所有 key 的共同平移。令 `Kc=(I−11ᵀ/T)K`，真正相关的二阶量是 `Γ=Kc ΔC Kcᵀ/d`。在该代理分布下 Γ=0 意味着这一旋转变化不改变 attention；Γ 非零仍不保证 top-k 或任务答案改变。
2. **合法旋转不等于真实 future Q。** pre-RoPE 内容向量仍依赖此前历史。U 逐 pair 恢复单位模时可能引入互不兼容的 π 翻转，未必对应任何共同位置；因此 P 胜 U 不能单独证明多位置分布必要。P/C 才是目前最直接的匹配位置处理比较。
3. **优化代理与满足真实需求不同。** 固定共同轨迹，令 α 是旧前缀占实际 attention 的质量、`a₀=Eα`、ρ 是按 α 加权的真实前缀需求，`d=1_SP−1_SC`，则实际质量差准确为 `a₀ πᵀd + a₀(ρ−π)ᵀd`。代理收益可被需求错配反向压过；增加抽样数只减小抽样误差，不能消除分布偏差。有限 M 下也不能把精确代理的 top-k 最优性直接赠给实际集合。

mass 到回答还存在独立缺口。固定 query、删除集合 D、剩余质量 m 时，准确 attention 输出差为

\[
F_S-F=\frac{\sum_{j\in D}a_j(F-v_j)}{m}.
\]

value 方向、抵消、后续投影与生成轨迹都会影响结果。raw V norm 是启发式；给全部 V 加同一向量不改变上述输出差，却可改变 norm 排名。共同 Full 轨迹上的局部量也不是不同自由生成轨迹的完整因果分解。

详细推导、有限反例与旧接口检查见 [核心交接的 Pro 方案分析](TWO_CORE_SOL_HANDOFF_20260909.md)。EA、KeyDiff、KVzip 之间还存在分母、聚合和预算策略差异；方法得分比较不能自动变成纯位置归因。

## 3. 已完成的任务结果

### PC2：32 DEV 与后续 16 DEV 必须分表

以下每任务 8 个输入，沿用 **RULER 官方答案项 recall**；多查询答案项不是独立样本，不是完整字符串加 EOS 成功率。模型为 NOSA-1B；原 reader、CIS、64-token 块、读取预算、保护和 GQA 规则固定。COBS 是本地适配，非作者完整系统。

| 方法 | multikey | multiquery | single | variable tracking |
| --- | ---: | ---: | ---: | ---: |
| 本地 COBS rank2 | .7500 | .6250 | 1.0000 | .3000 |
| rank1 + pair 残差 | .7500 | .59375 | 1.0000 | .3000 |
| 完整协方差 | .8750 | .8750 | 1.0000 | .4000 |
| 精确块打分 | .8750 | .90625 | 1.0000 | .4000 |
| 最大半径单 key + pair bulk | .5000 | .6250 | 1.0000 | .3000 |

完整协方差相对 COBS/rank1 各 **5 胜0负27平**。预定 8 条共同轨迹共 448 个层/时点状态上，选择分歧块的 cross-covariance 误差 MAE 约 .637，高阶误差约 .444；只用于这些观测状态。变量追踪的完整协方差仅 2/8 EOS，不能把 .4 recall 写成四成完整回答成功。

24 条无 observer 的共同输入中，端到端中位数：full covariance 21.87 秒、rank1 39.02 秒、COBS 38.83 秒、exact 17.71 秒。该实现读取全部原始 K，是机制参考；未得到胜过 exact 的质量—时间结果。

缓存完整协方差只验证了两例：完整输出相同，但 1,786,624 个 head-query 中有 56 个保留集合不同，不能称逐块 bitwise 等价。两例时间从 21.54/23.07 秒变为 23.60/24.61 秒，并增加约 .45 GiB 描述符，故不采用该性能方案。

第三方新增 covariance-tail 只运行了事先固定的 **每任务前4条，共16 DEV**：

| 同16输入的方法 | multikey | multiquery | single | variable tracking |
| --- | ---: | ---: | ---: | ---: |
| covariance-tail | .5000 | .4375 | 1.0000 | .2000 |
| 最大半径 tail | .5000 | .5000 | 1.0000 | .2000 |
| 完整协方差 / exact | .7500 | .9375 | 1.0000 | .4000 |

covariance-tail 相对 radius 为1胜1负，相对 rank1 为1胜2负，相对 COBS 为2胜2负，相对 full covariance/exact 均为 **0胜5负11平**。407.00 秒完整结束；逐行 total/prefill 中位数22.647/20.419秒，描述符约25.60MB。它没有恢复质量，不扩测、不扫更多单 key 准则。full covariance 在这16条中的部分行带 observer，不能用其混合时间来计算加速比。

来源：[一小时结果](TWO_CORE_ONE_HOUR_DECISION_20260909.md)、[16条配对结果](../../results/position_overnight_20260909/pc2_covariance_tail_dev_v1/paired_analysis.json)。

### PM：固定三格背景、八个来源家族

每格8条；同一家族派生三种背景，因此共24行但只有8个来源家族。Qwen2.5-3B、25%每层/每KV-head预算、原始位置与 reader 固定。以下全部为 **完整字符串加 terminal EOS 正确数**；自然背景指控制材料的背景，不是公开自然 QA 的完整基准。

| 方法 | 重复背景/16记录 | 自然背景/16记录 | 自然背景/256记录 |
| --- | ---: | ---: | ---: |
| Full | 5/8 | 6/8 | 6/8 |
| KeyDiff 固定预算适配 K | 3/8 | 8/8 | 0/8 |
| Uniform-prefix P | 1/8 | 0/8 | 0/8 |
| 类型均衡 P | 5/8 | 1/8 | 0/8 |
| 类型均衡 C | 5/8 | 1/8 | 0/8 |
| Key-novel P | 4/8 | 0/8 | 0/8 |
| Key-novel C | 3/8 | 0/8 | 0/8 |
| KVzip 重构评分、固定预算适配 R | 5/8 | 1/8 | 0/8 |
| KeyDiff pre-RoPE评分 K_pre（最新完成） | 5/8 | 7/8 | 0/8 |

类型均衡168条 row-arm完整完成，约479秒；uniform P24条约140秒；key-novel P/C48条约263秒；KVzip24条约79.78秒。上述时间分别对应不同工作量，不能直接排列为速度排名。KVzip还处理183,548个重构输入token；其完整评分成本不能省略。

Key-novel 只把前缀 Q 的抽样权重改为关联 key 到均值方向的球面距离，H128/M256/raw V norm等保持固定。它没有继承 KeyDiff **直接选择 K** 在 prose16 上的收益。继续微调这项抽样几何没有现有结果支持。

KVzip调用固定作者版本的 `prepare/score_kvzip`，用已知前缀重构评分，在真实问题到来前冻结集合。它同时改变 Q 来源、max聚合和归一化，并采用本项目固定per-head配额；未复现作者跨层/head全局分配、fake pruning或KVzip+。因此这既不是纯 Q 来源消融，也不是完整原论文系统被否定。R/P总分相同不代表相同错误：prose16为1胜1负6平，8条输出全部不同。

最新K_pre在约44.19秒内完成24条新生成，另复用K/F48条。对K为 **2胜1负21平**，对F为1胜6负17平；密集记录仍0/8。它提供了这份评分位置干预的局部得失，尚不支持稳定优胜或大TEST。原始证据见 [K_pre完整结果](../../results/position_overnight_20260909/pm_canonical_keydiff_dev_v1/paired_analysis.json)。

更早 E12 锁定 TEST：128个输入、896条row-arm，P对C为 **1胜3负124平**。Single的P/C/U/F/K strict exact均为31/32；order P7/32、C/U8/32、Full9/32、K6/32。自然QA存在取舍，未建立PM位置收益。该TEST已被查看，不再用于调参后冒充独立确认；早期16 DEV/64 DEV与这些128 TEST也不可相加成新独立证据。

来源：[持续研究结果](TWO_CORE_CONTINUATION_RESULTS_20260910.md)、[原始完整交接](TWO_CORE_SOL_HANDOFF_20260909.md)。

## 4. 失败原因：已识别到哪一步

### 4.1 代理抽样的背景依赖，以及真实 Q 反向

类型均衡按词表token ID分配概率，不按记录/事实分配概率。8个家族的离线核验显示，每head抽256次时，目标记录的期望样本数在重复16记录约9.8、自然16记录约.8、自然256记录约.35；均匀抽样约.6。正文token种类增多后，“稀有ID”不再集中于记录。这解释了该抽样统计为何变化，但没有证明必须抽到目标原文位置的 Q 才能成功。

9条冻结输入的真实Full后续 Q 已保存：6条检索仅来自2个材料家族，另3条为自然文档。对齐同一raw V权重后，P自己的代理目标9/9高于C，而真实Q上的目标7/9低于C；去掉prefix质量α后仍是同7条反向，未加V质量重建最大误差7.16e−7。**未来分母权重不是这个反向的充分解释。**

六条检索Full轨迹的答案本身全部错误。接近这些Full局部输出不能称为正确上界；9条也不是9个独立家族，更不能代表全部未来时点。这里支持的是特定轨迹和时点的代理—需求错配。

### 4.2 原文保留集合确实不同

第三方用原tokenizer重建冻结前缀，核对实际生成的keep哈希；旧P/C/U/K/F共30个哈希全部一致，新增R两例也一致。prose16的000/001中：

| 输入 | P目标value平均保留 | K目标value平均保留 | R目标value平均保留 | 完整value所在层/head单元：P / K / R |
| --- | ---: | ---: | ---: | ---: |
| 000_prose_16 | 33.10% | 82.41% | 30.56% | 4 / 48 / 2（共72） |
| 001_prose_16 | 20.83% | 70.24% | 33.73% | 1 / 36 / 3（共72） |

两例均是K正确、P/R错误。全部16条字面记录只需284/286 token，预算1911/1871，因而“连全部记录字面token都容不下”不能解释这两例。R全部记录token覆盖约37.45%/36.34%，K约71.75%/70.61%。prose256的全部记录4565/4556 token则大于约1846/1836槽位，不能把低竞争条件的结论直接外推。

源token未保留不等于信息在其他隐藏状态中不存在；源token完整保留也不保证模型正确读取。R答案只接近正确拼写仍算错误，不能改变完整输出指标。

来源：[原文审计汇总](../../results/position_overnight_20260909/three_party_retention_evidence_v1/summary.json)、[R原文审计](../../results/position_overnight_20260909/three_party_retention_evidence_v1/kvzip_summary.json)。

### 4.3 补回目标记录：一条因果恢复，一条明确反例

核心/执行任务选定原顺序最先两条Full正确、P错误的DEV密集记录输入002/003。原问题定位目标记录，补回原始K/V，保持原位置、总预算和reader；随机sham采用同移除集合与同插入数量。没有插入答案token，但未来问题参与定位，故属于特权oracle。

| 输入 | P与随机sham | 目标记录全部补回 | 正确答案 |
| --- | --- | --- | --- |
| 002_prose_256 | 均错，二者逐token相同 | `bdqynkghuvn`，正确且EOS | `bdqynkghuvn` |
| 003_prose_256 | 均错，二者逐token相同 | `bdxxwmetc`，错误且EOS；它是问题中的key | `bdxudqajtgx` |

每条目标记录19 token，在全部72个层/head单元中完整保留；预算1854/1851，平均交换18.03/14.24位置。原P完整合同与实际集合匹配后复用，只新增四次自由生成，约16.05秒。

002支持这项补回干预的因果效果；003说明完整字面记录不充分，尚未区分问题状态形成、key/value角色绑定、其余上下文或attention竞争。它**不证明attention已被排除**：gold进入支持集并不是完美路由。两条经headroom筛选，不能据此估计总体50%可修复率。

来源：[原始四次生成及交换回执](../../results/one_hour_decision_20260909/pm_target_record_oracle_dev_v1/)。

### 4.4 原生顺序读取也需要单独保留

对旧E12 Full的32个order输出做事后错误分类：全部EOS且两个字段；要求的latest-A正确30/32，first-B正确9/32，却有18/32返回未要求的latest-B，16条完整输出恰为latest-A/latest-B。此分析没有重新生成、没有改评分。它显示原生occurrence读取缺口，不能全归因于压缩；也不能把看过TEST的诊断变成对该TEST训练/挑选新方法。

来源：[order错误分类](../../results/position_overnight_20260909/three_party_retention_evidence_v1/order_error_categories.json)。

### 4.5 最新K_pre三条分歧：成功与失败不能共用一个覆盖率解释

第三方只读核查全部三个K_pre/K得分分歧，没有加载模型或新增生成。冻结前缀tokenization、selection回执和实际生成keep哈希均匹配。

| 固定DEV输入 | K → K_pre完整答案结果 | 目标value平均保留率 K → K_pre | 完整value层/head单元 K → K_pre |
| --- | --- | ---: | ---: |
| 004_repeat_16 | 错 → 对 | 87.73% → 97.92% | 56 → 68 / 72 |
| 005_repeat_16 | 错 → 对 | 84.13% → 96.23% | 55 → 66 / 72 |
| 000_prose_16 | 对 → 错 | 82.41% → 85.88% | 48 → 53 / 72 |

前两条与“目标字面信息保得更完整”一致，尚未通过局部交换把原因唯一归给这些token。第三条则是实际生成层面的反例：K_pre连整条record的平均覆盖也从78.94%升至86.19%，完整record单元从18升至33，却输出`bdbnvkygg`；这与同输入Full的错误文本相同，正确值为`bdztcffuxnw`。**不能把平均覆盖率当作下一轮优化器，也不能把更像Full当作成功。**

另外核对冻结records：004的K错误文本`bdgkzrabmqy`确实是另一个key的value；005的K输出比正确`bdvtsqpjsyq`少一个`v`；000的K_pre错误文本则不等于这16条记录中的任何完整value。三例分别表现为错误对应、字符缺失和未命中声明记录值，不能统一写成“都只选中了错误记录”或“都是顺序问题”。

这些是跨层/head汇总，未声称每一个单元的集合都包含旧集合。仍可能有关键局部缺失、竞争上下文变化或状态形成差异；尚未唯一识别。保留三条完整分歧比只报告总分净增1条更能指导接续。

来源：[三分歧审计](../../results/position_overnight_20260909/three_party_retention_evidence_v1/canonical_keydiff_three_disagreements.json)。

## 5. 第三方候选及核心后续审查：保留公式，停止失败版本

第三方实现的 `pc2_covariance_tail1` 从每块选择一个真实key精确计算，其余仍用pair二阶。令 `E=offpair(Σ)`、`rᵢ=kᵢ−μ`、`αᵢ=wᵢ/(1−wᵢ)`，这个混合摘要在零query处遗漏的Hessian满足

\[
E_i=E-\alpha_i\operatorname{offpair}(r_ir_i^T),\qquad
G_i=2\alpha_i r_i^TEr_i-\alpha_i^2\|\operatorname{offpair}(r_ir_i^T)\|_F^2.
\]

选择非负最大gain，保留不提取选项、近singleton数值保护，以及zero-cross时的明确tie规则；用B×B Gram实现，无SVD。7项本地CPU检查及服务器CPU核验通过，另有100块900次exclusion独立恒等式核验。源快照SHA：`9c5a74846e25f538b5c73be587e0db18461d69522bc718461b6259b198bb1663`。这些只验证声明的数学/实现性质；上面的16次真实生成决定其未成功。

核心接着给出更精确的限制：真实二阶均方风险是 `E[(aᵀEa)²]`，一般依赖query原始四阶矩；均值/协方差不足以决定它。零均值各向同性Gaussian且trace(E)=0时，Frobenius才对应这个风险。实际候选的64-token有限反例中，Frobenius平方误差降约39%，一个查询方向的平方误差反升约74%，并发生真实logmass排名反转。

即使换成经验query风险，校准分布也可能不转移；即使方向风险下降，也可能不改变cutoff两侧相对误差和最终名额。核心写出了经验gain与CPU复现，但**没有新的GPU成功结果**。它是可讨论的后续构造，不是应自动启动的下一轮单key扫描。

近邻工作已核对：COBS §5.4–5.6已有query二阶矩子空间、低秩、FP4和Gram构造。不能以query-weighted或Gram本身声称新颖性，也不能只超过本地full-space rank2适配就称领先COBS完整方法。

来源：[候选实现](../../experiments/nosa_position/covariance_tail.py)、[测试](../../experiments/nosa_position/test_covariance_tail.py)、[查询风险审查与CPU反例](../../results/one_hour_decision_20260909/query_weighted_covariance_review.md)、[COBS原文](https://arxiv.org/html/2607.09052v1)。

## 6. 接续任务：已在执行、已准备、仅建议分清

### 已完成的 KeyDiff 位置评分对照：保留局部收益，不自动扩测

`pm_canonical_keydiff_dev_v1/status.json` 已核对为 **COMPLETE、24条**。其5/8、7/8、0/8及全部分歧已纳入上表；接管任务最新回执为GPU无作业、已交付配置全部完成。它等待的是下一项具体研究工作，而非重复执行旧命令。

只把KeyDiff评分输入改为第一次原生 `k_proj` 捕获的pre-RoPE K；最终gather与读取仍用原post-RoPE K/V和原位置。同prefix上的K_post评分必须重现原K的keep哈希，K/F输出直接复用。没有BF16反旋转或第二次K投影，没有层/系数网格。完整配置见 [canonical_keydiff_ready_command.json](../../results/one_hour_decision_20260909/canonical_keydiff_ready_command.json)，输出目录 `runs/pm_canonical_keydiff_dev_v1`。

这个比较可能纠正显式旋转引入的异常度，也可能丢失有用的post-K几何；现在CPU和任务层面都存在得失。KeyDiff原本已有共同正交旋转与全局原点不变性，不能冒称这是K_pre的新性质；pre-RoPE状态也不等于纯语义。禁止按这24条已知答案或背景类别选择K/K_pre混合规则，然后称为独立收益。

### 下一项明确合同：两输入的问题K/V状态交叉

要区分的解释是：**003仅有完整目标还不够，是因为问题阶段在压缩前缀上形成了不足的状态，还是即使补入Full形成的问题状态，当前支持集与reader仍不足以答对？** 002是已救回的正控制，003是未救回的反例。这里不重新诊断已经成立的“P可能丢原文”。

选择矩阵如下。`S_P`/`S_O`分别为已保存的P/完整目标oracle前缀集合；“本分支”表示此前问题token原本就在该集合上形成状态。

| 最终前缀集合 | 此前问题K/V来自本分支 | 此前问题K/V来自Full |
| --- | --- | --- |
| S_P | 复用原P，两例均错 | 每例新增1次生成 |
| S_O | 复用原oracle：002对、003错 | 每例新增1次生成 |

执行者在现有 `BalancedValueSession`、`DecodeState` 和 `target_record_oracle` 上加一个独立诊断入口，建议新模块 `experiments.pm_keep.question_state_cross`、新目录 `runs/pm_question_state_cross_dev_v1`。这是**交付的实验合同，入口尚未实现**；不得把建议路径写成已存在的可运行命令。固定模型、数据、精度、原生位置、预算与greedy/EOS合同，具体步骤为：

1. 只用 `broad_retrieval_dev_002_prose_256`、`broad_retrieval_dev_003_prose_256`。复用原oracle目录中的 `.keep_sets.pt`、原始输出和交换回执；核对当前原前缀、模型及集合身份。
2. 原生Full前缀后，只摄入 `suffix_ids[:-1]`，且沿用原adapter的逐token摄入路径；捕获这些**已观察问题/模板token**的各层K/V。不要摄入最后prompt token，不生成答案。
3. 分别建立原S_P与S_O分支；原前缀K/V只按对应旧索引gather。在每层追加第2步形成的问题K/V，保持其原逻辑位置。物理长度为 `B+len(suffix)-1`，下一逻辑位置为 `T+len(suffix)-1`。
4. 在这个分支上重算 `suffix_ids[-1]`，再自由生成完整答案。不得沿用Full最后问题logits，不允许插入答案token或更改剩余prefix集合。最终回答入口都是 `B+len(suffix)` 个KV。
5. 保存四次新生成的raw token IDs、EOS、完整答案分数、首次预测token、前缀/问题K/V来源、哈希、物理/逻辑长度与Full问题处理成本。只需对新cache拼接做小型自回放检查；无需重跑全24条基线或整个旧测试套件。

原版代码持有Full前缀以便配对分支，`session.branch('F').consume(suffix_ids[:-1])`可形成参考问题状态；追加的仅是物理索引 `T:T+len(suffix)-1` 的新问题K/V，不能把Full前缀整块复制进压缩分支。原始prefix session应保持未被问题修改。

该诊断利用Full读取历史，计算/峰值内存不等于同预算部署。问题K/V可能已携带潜在答案信息，应称**问题状态及内容传递的干预**，不称纯Q方向或新的压缩成绩。它收紧了核心原先“读完全部问题再gather”的计划，避免直接继承Full起步预测；**本轮只做这一种，不同时再跑原宽版本。**

结果决定下一项，而不是只多写一张表：

| 003的实际结果，始终同时报告002是否保住 | 被支持/排除到的范围 | 下一步设计取舍 |
| --- | --- | --- |
| S_O+Full问题状态正确，S_P+Full仍错 | 在此干预中，需要较好前缀证据与问题状态共同满足 | 只改善token保留不足；合法修复需要同时保留形成正确问题状态的证据链 |
| 两个Full问题状态分支都正确 | Full形成的问题状态能在这两个集合上带来足够信息 | 内容可能经问题KV传递；探索实际预算下能形成这种状态的机制，不能把特权状态直接冒充方法 |
| S_O+Full仍错 | 该问题状态替换不足以修复，不能宣布一切问题历史无关 | 停止这项Full状态替换；转向同支持集内的key/value绑定或竞争信息，不追加更多Q来源拼接 |
| 002退化，或两个集合出现相反取舍 | 存在状态/支持集交互，未获得稳定修复 | 保留正例被破坏的事实，拒绝平均分掩盖；按具体干预差异选择下一项 |

若同路径身份检查失败，先定位这个具体接口差异；不把它当科学阴性，也不重跑无关已有效结果。该任务结束后接管者自行做有证据的下一步判断，不等待另一个会话提供方向。

### PC2剩余问题

接管者可复用full covariance、exact及已保存共同状态，检查恢复得分的输入中，摘要错误是否跨过实际cutoff、改变有效名额。首先区分“小摘要忽略了关键方向”与“二阶已够好而高阶/归一化决定边界”；不把大量无关块的平均误差当依据。该只读检查直接服务一个实现选择，不追加模型前向或全诊断矩阵。

核心另在审查“排名无法判定时回退exact”的有限状态方案：GQA每head分母、原名额政策、浮点误差都要在界内；高维界可能过松而全回退。若保存状态上几乎全回退，就停止该具体界的工程化；若少量精确补算已能保住关键名额，再考虑一次小生成比较与实测总成本。**尚无可交付的低成本实现或生成收益。** 经验query权重也要面对校准迁移及已有COBS工作，不同时排多个新方案。

下一次真实生成应围绕一个具体可区分的缺口，复用既有对照。CPU检查与局部诊断为选择提供依据，不应再演变成等候完备理论的新门槛。

## 7. 历史失败中必须继续带走的纠正

历史总表由 [ROPE_LOCAL_FAILURE_SYNTHESIS](ROPE_LOCAL_FAILURE_SYNTHESIS_20260908.md) 和其证据JSON负责；这里保留与当前决策直接相关的结论，不能把不同实验统一判成“位置理论无效”。

| 既有事实/纠正 | 当前应避免的重复 |
| --- | --- |
| 早期PPL崩溃比、首字符评分、被裁成8K的长输入、混合pilot/confirmation曾制造错误结论 | 原始输出和实际输入长度拥有结论；格式、EOS、答案项recall、整段F1不得互相代替 |
| 频谱multiset相同的槽位置换仍可严重退化；旧有效p2又不必严格单调 | 保留数组/槽位与源身份；不能用有序、平滑、相位覆盖充分证明能力，也不能通用排序“修复”已有方法 |
| 官方YaRN幅度、repo变体、原p2/数值修复p2/FullLag不是同一方法 | 先对齐模型、support、增益、读取器、数据、合同；实现名称相同不保证是匹配基线 |
| ALS首轮`inf<=inf`的虚假收敛、LoRA rank除以head数的错误界已经纠正 | 修bug只否定受影响的证据；局部最优残差不是不可修复下界，未启动/中断训练不是完整配方失败 |
| Gram、曲率、mass、target rank和拟合残差已有改善但回答不改善的直接反例 | 当前covariance-tail与PM的局部量必须接回实际生成，不能改名后再宣称已解决 |
| 强制gold块在集合中仍用普通softmax，未移除全部竞争 | 当前003 oracle阴性不能排除attention，也不能证明所有状态无法修复 |
| BM的Qwen3B 128K两例交叉缓存结果跟随前缀来源，不随最后读取表切换 | 仅改晚期旋转未修复这两例；状态形成应与最终读取分开，不能外推为全部模型/长度的统一根因 |
| PSR小样本未胜等量contiguous；selector RoPE/NoPE自然QA结果无稳定优势 | 当前覆盖不足以永久否定稀疏位置主问题；增加摘要数、换接口、人工格式差异不能冒充位置贡献 |

BM交叉的multikey四臂均EOS；VT的MrPro来源是100%答案项召回但预算结束无EOS，必须保留这个差别。[BM交叉缓存原始结果](ROPE_BM_CROSS_CACHE_20260908.md)只支撑两例回溯诊断。PSR/自然QA、原生shared-head归一化等更早结果见 [CORE_DIAGNOSIS](../../experiments/native_sparse_position/CORE_DIAGNOSIS.md)，不能把每个架构都假定成post-RoPE平均抵消。

共同教训是：每次失败应改变下一项具体干预；真实正例、混合结果、后续更正继续保留。它不意味着所有方向必须先有普遍保证，也不意味着一批DEV无收益后可以直接扩大TEST证明方法。

## 8. 文件、运行身份与交接边界

### 文档与原始证据入口

| 内容 | 当前入口 |
| --- | --- |
| 核心持续更新、已准备工作 | [TWO_CORE_CONTINUATION_RESULTS](TWO_CORE_CONTINUATION_RESULTS_20260910.md) |
| 一小时32DEV结果、成本、两个入口修复 | [TWO_CORE_ONE_HOUR_DECISION](TWO_CORE_ONE_HOUR_DECISION_20260909.md) |
| 最全逐轮交接、E01–E12、推导与旧结果 | [TWO_CORE_SOL_HANDOFF](TWO_CORE_SOL_HANDOFF_20260909.md)；后续时间明确的新记录优先于早期计划状态 |
| 第三方P/K/R保留、顺序错误、PC2状态汇总 | [three_party_retention_evidence_v1](../../results/position_overnight_20260909/three_party_retention_evidence_v1/) |
| 第三方16DEV执行、合同、源快照 | [pc2_covariance_tail_dev_v1](../../results/position_overnight_20260909/pc2_covariance_tail_dev_v1/) |
| 核心一小时及后续配对资产 | [one_hour_decision_20260909](../../results/one_hour_decision_20260909/) |
| GPU主结果同步副本 | [position_overnight_20260909](../../results/position_overnight_20260909/) |

第三方新增源码为 [retention_evidence.py](../../experiments/pm_keep/retention_evidence.py) 及其 [3项测试](../../experiments/pm_keep/test_retention_evidence.py)，以及上文covariance-tail与7项测试；另外追加过既有交接记录。核心的full_covariance_probe/hour_bridge/cached_full_covariance/tail_pair、future_query_probe/replay_weighted_probe/key_novel_queries、kvzip_reconstruction、target_record_oracle、canonical_keydiff代码均仍在各实验包内，准备状态以对应合同和源码快照为准。

远端为 `ssh -p 24941 [REDACTED_EMAIL]`，工作根目录 `/root/autodl-tmp/position_overnight_20260909`，Python `/root/miniconda3/bin/python`，KVPress固定提交 `71640b4f9061054a7630c5049bb9ee659a01523c`。共享 `queue.lock`，启动时沿用STOP/实际PID核查；root自己的16DEV进程早已退出。之后key-novel、KVzip、target-record与已完成K_pre均由接管任务统一执行，本任务未再占GPU。

旧hour_finish入口有两个真实执行错误：缓存shadow的`--row-ids`传入CSV而runner要求JSON文件；key-novel缺必需`--baseline-cache`。二者均在模型加载前失败，修复留有备份；缓存shadow/timing与weighted replay后来完成，原一小时队列最终标`PARTIAL_DEADLINE`，被跳过的key-novel已在后续单独完成。不能隐去入口错误，也不能把首次失败当作已发生的方法阴性。

本次开始写综合交接时，已有并发修改涉及continuation结果文档、旧handoff和canonical_keydiff源码；本文不覆盖这些工作，本任务未提交、推送、删除结果或关机。ignored结果目录不能因为不在Git里就当作不存在。尚未完成的研究不写入现有 `paper-2027` 作为SOTA结果；该目录原有稿件的证据身份独立保留。

**执行交接：**K_pre已完成；接管任务按第6节的明确合同准备并执行四次新生成，用其结果决定下一项，继续已有两条主线。第三方任务在发送本文后结束，核心会话的卡住状态不再成为实验前置依赖；无需回复或等待这两个任务。新结果由接管任务更新自己的持续结果文档，不要求不断回写本交接快照。
