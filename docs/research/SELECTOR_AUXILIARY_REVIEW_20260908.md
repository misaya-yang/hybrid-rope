# 稀疏选择器辅助复核：实际生成、可比性与当前代码

本文件属于纠错辅助任务，记录已经核验的事实与具体建议，不提出第二个GPU队列。主任务两小时窗口截至2026-09-09 00:13:46 UTC，起点不随方案修改重置。

## 最新判决：本轮尚无通过晋级的论文方法

32条冻结新输入的完整比较和强基线已经完成，主指标为未strip的整个原始答案串exact加真实终止EOS。

| 方法 | 正确/32 |
|---|---:|
| Dense | 17 |
| RoPEMean | 0 |
| Quest，64-token读取页 | 3 |
| QuestSplit32，两个32-token摘要但仍读64-token页 | 8 |
| PostMetric4 | 10 |
| PreMetric4 | 11 |
| MatchedContiguous4 | 1 |
| QuestFine32，实际读32-token页 | **13** |

7臂原轮224条生成完成，759.493秒；outputs SHA256为`216ea464d40dc0aca0193251c767302d4e92d40b3499340650a8b8ef678ab8db`。独立复核了(row_id,method)唯一、32行全配对，以及同一行expected/prompt/prefix身份一致。QuestFine32追加32条完成、210.494秒，outputs SHA256为`a642b957a04671baf68db790cc8e55049b2bd31f90e9fe493496050570ff58d8`，input SHA与原轮一致。

QuestFine32保持sink64、local2048、1024个远端token预算；缓存真实BF16输入的min/max端点，无损BF16回存及FP32分数逐位一致均在构建时检查。**正式32行试验的4均值实际为FP32缓存，QuestFine32约只需其一半摘要字节。** 例如首行首层分别为2,059,872与1,022,976字节。此前辅助建议沿用了早期BF16均值草稿的假设，现已根据正式build回执更正；未测试的BF16均值不能当作已完成结果。当前原型中FP32 Quest端点的额外成本不能视作基线不可消除的开销。

PostMetric4对QuestFine32是3胜6负，PreMetric4为3胜5负；未校正的双侧配对sign-test p分别约.508与.727。这里不是统计上证明新方法普遍更差，**而是没有证据支持其质量—成本胜出**。对PostMetric4严格匹配组大小的连续控制有9胜0负，说明这项有限assay中联合key分组有实际恢复价值；PreMetric4只与该连续臂同容量，其具体组大小并不逐块匹配。Pre/Post之间为3胜2负，位置metric独有收益未建立。

这32条是一个RULER生成的key/placement开发assay，不是完整RULER、自然文档总体或跨架构证明。结果应保留为有明确边界的实测进展，不能升格为solid-accept论文核心。已通知主任务不在最后阶段更换R、预算或模型寻找另一个赢点，并保持原授权时间边界。

## 已验证的生成证据

1. `evidence_repair_qwen35_01`：4条生成完成，8.787秒；同预算强制证据块137/138后，完整generated_ids与Dense一致。原NoPEMean把Cornwall答成Great Britain；错误块134/135未恢复。全部EOS，但4臂的严格full_exact_and_eos均false，Dense/Repair为normalized_exact=true。原RoPEMean单均值、两种精确oracle也早已正确，故此例不是新方法胜过已有路径的证据。
2. `support_ruler_02`：2026-09-08 22:55 UTC复核status为COMPLETE，PID28229已退出；4题×4臂，共16条输出，77.484秒，全部EOS。Dense/SupportRepair的严格终点均0/4。两条multikey的Dense数值错误；multiquery_0的Dense/Repair给出了全部4个正确数值但用空格分隔，未按规定逗号格式；multiquery_1不完整。应判当前严格assay未合格，不能以该0/4否定selector机制，也不能把数值解析结果升格严格exact。

第二轮原始outputs SHA256：`77481b42f8f10ea36e820c772f1fd71f7dcd86d652c961610d8857445c4473fb`。原始目录位于远端 `/root/autodl-tmp/position_observability_20260908/`。

## 已确认的历史可比性问题

本地 `results/bm_transfer_20260908/run_qwen3_01/MrPro.jsonl` 的相同4题内容均正确，但 [旧协议](ROPE_BM_TRANSFER_20260908.md) 明确32K和128K都使用静态S4及gain=1+.1ln4，不能把32K行当原生算子基线。旧单key输出含前导空格和末尾句号，旧multiquery有解释段；也不是严格完整字符串能力证明。

远端tokenizer解码确认：`support_strict_inputs_02`保留与原题相同文档前缀和证据位置，却删除了原assistant answer_prefix中的重复查询key，新增用户格式指令并改为空assistant前缀。

因此旧正确与新错误至少混有operator和query-tail变化。建议只在同Native算子与同文档prefix上补2条原tail的Dense回放，与已有strict-tail结果比较；不重跑整套四臂或继续猜提示词。源码记录抽取的证据数值与参考答案顺序一致，目前未见label/order错配。

## 覆盖与性能代码检查

- 已发现旧完整remote块＋精确local token实现，在local边界有N mod 64个不可选token，最多63个。独立CPU枚举N=32768..32831确认；N=32769时缺token30720。若support准备器直接与eligible取交集，会静默漏掉边界证据，造成无效oracle。
- 新 `prepare_support_strict.py` 已加入初始和整个continuation区间的必要证据覆盖断言。2026-09-08读取时SHA256为`afe666ac271e66f76e005eda5739502f750c369e6168b45933732b03d8aedd9c`。这是已看到的代码修正，不能被描述为补齐所有历史reader路径。
- `mixture_native.py`缓存摘要分支仍无条件创建全K的FP32副本，却不用该变量；已给出将ks转换移入单均值分支的具体建议。该建议针对草拟的mixture代码，未声称已应用，也不要求为这个无语义影响的优化重跑旧结果。
- m16是新的较低读取预算开发点；同m16的对照可比较，旧m32的oracle恢复量不能作为m16下已经测过的上限。
- BF16均值的实际混合分数不自动具有精确实数均值的Jensen下界。理论需说明舍入误差，不能把正权重等同于有限精度下绝不高估。

## 新增一手近邻核验

- [ClusterKV §IV-B](https://arxiv.org/html/2412.03213v1)明确在QKV投影与RoPE之后进行key聚类。因此post-RoPE clustering不是新意。其读取单位是语义簇中的token，区别于在固定物理页内建少量摘要后读取整页。
- [Multipole Attention §3.2–3.3](https://arxiv.org/html/2506.13059v1)已有count×exp(query·centroid)项和按时间段分块、只更新末块的在线聚类；它还用centroid补偿未精确读取的部分。不能将这些数学／缓存操作重新命名为新贡献。
- [ClusterAttention §1.2](https://arxiv.org/html/2608.26965v1)已经讨论紧聚类在纯丢弃模式下可能输给随机聚类，因为遗漏质量之外，保留／遗漏value输出的差异同样参与误差。其主要实证是双向注意力场景，不能直接当自回归LLM结果，但其理论边界必须正面承认。
- [TriAttention](https://arxiv.org/html/2604.04921v1)已有pre-RoPE中心、三角位置偏好与key重要性估计。它面向长生成KV压缩，不等于当前真实query的物理块极值界。

以上查新用于限制贡献表述，没有要求当前短实验前先移植全部系统。目标任务已转向RPEE条件候选；其PCA二维包围框上界和共同旋转性质仍须由实际Quest失败、同预算答案恢复和随机配对控制支持，不能仅凭CPU例子进入论文主张。

## 本辅助方案的反向审查

已在 [联合路由方案](ORACLE_FIRST_SHARED_ROUTING_PLAN_20260908.md)追加固定Q/K、固定W_O、仅换共享V就使加性与log覆盖哪个正确反转的CPU反例。最差head覆盖也是proxy，不能替代真实value/readout与完整生成干预。因此本辅助任务没有给自己的log覆盖方法排GPU，也不会把当前摘要误差修复当作其证据。

## RPEE的进一步理论核对：对称性不是独有的能力修复

在频率表固定的前提下，以序列起点p0作anchor，将selector收到的实际post-RoPE Q/K都乘R(−p0)，再应用原Quest。reader不变。共同位置平移Δ时，anchor变为p0+Δ，有R(−p0−Δ)R(p+Δ)=R(p−p0)。因此无需PCA矩形或更多descriptor，这个简单canonicalization也保持共同位置平移不变性。p0=0时就是原Quest，不需因此多开一组GPU对照。

CPU实算原二维例：A由±(1,1)构成，B为(.5,−.5)，q=(1,−1)。真max分别0/1；普通Quest随共同旋转改变排名。起点归一化Quest在0、π/4、π/7、2.71的平移角下始终保持原2/1评分，**依然选择错误的A**。所以RPEE若能赢，实际增量必须表现为包围框角点膨胀减少、固定常规起点下的自主答案恢复，不能只报告恢复了某种对称性。

一个可严格使用的局部几何结论：把原生pair排成连续二维基后，RoPE生成元A=diag(ω_k J)。对原坐标轴张成的子空间，其正交投影P满足[P,A]=0，当且仅当该子空间在所有R(Δ)下闭合。非零ω的坐标若被选中，其旋转伙伴也必须在同一子空间；所以完整native pair是这种**局部坐标分块**的最小闭合单位。这不是关于所有可能selector的维度下界，不能排除使用外部anchor或其它全局不变量的方法。

独立4维CPU反例：点集±(1,0,1,0)，q=(1,0,−1,0)。只旋转native plane(0,2)，真max恒0；native分组(0,2)/(1,3)的包络恒0，而跨pair分组(0,1)/(2,3)在角度0、π/4、π/7时分别给2、约0、1.2469796。该例说明随机配对控制能识别局部闭合性质，但不证明真实模型能力、PCA最优性或论文新颖性。以上两项CPU检查均未使用GPU。

## primary短任务的模板边界与原始输出复核

`primary_ruler_01`的STOPPED原因是上游essay生成器最少500词，无法满足要求的512-token输入；没有模型结果。后续改为2048的`primary_ruler_02`已生成8个compact开发例与32个独立long例。实际落地文本含唯一一次完整数字格式指令，未发现指令替换失效。

`primary_compact_01`完成8个Dense输出，程序记录12.574秒。现有`full_exact_and_eos`为4/8，其实现使用`answer.strip()`。独立以原始generated_ids去除最后EOS、保留其它字符和特殊token解码，**未strip的完整输出exact为0/8**：8条均以一个换行开头，另外4条还有漏数或顺序错误。

实际落地prompt末尾是`<|im_start|>assistant`，缺少原生assistant头的末尾换行。已通知执行任务恢复并断言完整原生chat边界，旧数据/输出保持原样。此问题应按输入模板错误处理，不能由strip静默掩盖，也不能解释为RPEE成败。修复该边界不会自动修复另外4条内容错误。

建议未来回执同时保留`trimmed_full_string_and_eos`与`raw_generated_string_and_eos`，准确标记与规定主终点的关系。需要完整固定字段时，应在生成前完成模板和参考串约定；不能在看到结果后删除任意前缀或用首数字/substring替换主指标。

**2026-09-08 23:23 UTC代码修正复核：** 生成器已补回并断言`assistant\n`头；评分器只删除末尾实际EOS，使用`skip_special_tokens=False`解码，并将不strip的`text_exact/full_exact_and_eos`与trimmed字段分开。8条旧输入逐一验证：补回换行后的token IDs恰好等于原input IDs加其第一个generated token198。这定位了缺失结构分隔符，不构成新的模型能力结果；旧输出及其0/8原始字符串判决保留。

同一时段，执行任务报告候选GPU摘要构建遇到CUDA小矩阵eigh错误，未产生方法生成结果。当前`pair_envelope.py`已改为解析对称2×2解：gap=hypot(a−c,2b)，angle=atan2(2b,a−c)/2，principal=(cos(angle),sin(angle))；退化情形仍由圆盘分支处理。该代数替换正确，实际GPU成功与生成收益仍须由后续回执验证，不能把这次构建失败写成方法负结果。

## 首个RPEE真实生成结果及下一项决策

`envelope_causal_02`已完成，5条输出、30.552秒；本地和远端outputs SHA256同为`d1a7cd94aab5d2ba7a50221632e920ea5744821e6c2af873c3b9cd6def511b90`。固定例是之前可由证据块oracle恢复的`niah_multiquery_32768_0`。

- 正确数字序列为2608476、9403234、4105180、3192420。
- RPEE生成`2608476 4102310 3192420 3192420`；Quest生成`26081234 94032010 4105180 3192420`；RandomPair生成`2608476 4105180 3192420 3192420`；QuestSplit32第二个数也错误；RoPEMean复现原错误`2608471`。
- 全部EOS，严格字符串与完整数字内容都未恢复。不能将这次失败仅归因于逗号格式。
- 同一RoPEMean失败轨迹的1200个相关源块诊断cell中，RPEE选中69、Quest55；竞争块上界膨胀均值约26.38对26.80。该局部差异不是独立答案胜场，也没有对应当前例的能力修复。

在扩大矩阵或改轴／阈值前，应先用现成原始Q/K在同m16预算检验**精确block-max目标**与**精确block-logmass目标**的完整生成。如果连精确max都不能恢复，当前例就没有支持继续优化RPEE的max近似；不将一个例子的失败外推为所有任务不可能。

**后续实际核验：** `oracle_target_01`已完成两条，12.159秒，精确block-max和logmass都恢复了该例全部4个正确数字及EOS，但输出为空格分隔，仍未通过旧逗号串的严格指标。`mixture_causal_01`随后16.750秒完成3臂：PostMetric4和PreMetric4同样恢复该完整内容，匹配连续分组仍有错误。这支持从独立pair极值切回保留联合key结构的有限试验，后续32条结果如本文件顶部；不把内容恢复与旧严格字符串通过混淆。

可在同一次必要回放中计算明确的误差分解。令g为真实rotary pair（非旋转坐标作为单维组），U为各组有效包围上界之和，T=Σ_g max_j(q_g·k_jg)，M=max_j Σ_g(q_g·k_jg)。则严格有U−M=(U−T)+(T−M)，两项非负。U−T是组内包围松弛，T−M是不同组的极值来自不同token的错拼。**即便每个pair都使用完全精确的支持函数，后项仍在。** 只对源块及实际竞争块求值即可；不能凭平均上界下降就继续投入。

另已指出汇总器`joined` key遗漏row_id，多个输入的相同(layer,position,head,block)可能覆盖。当前单例无影响；扩为多例前应加入输入身份，已有diagnostics只需重新汇总，不需GPU重跑。

### 为什么跨pair错拼是表示能力限制

取原生pair为(0,2)、(1,3)。块A包含32次(1,1,0,0)与32次(−1,−1,0,0)；块B包含32次(1,−1,0,0)与32次(−1,1,0,0)。**每个pair的完整投影多重集合在A/B完全相同**，所以任何只缓存彼此独立pair统计、丢掉跨pair token对应关系的表示都无法区分两块；增加每个pair内部的统计精度也不解决。

对q=(1,−1,0,0)，A的真max=0、真logmass=log64；B的真max=2、真logmass=log(64 cosh2)。两者的逐pair精确max之和T均为2。上述值与多重集合相等性已CPU直接核对。这严格说明独立pair摘要可以丢失与全维分数有关的信息；**没有证明真实失败由这一项主导**。必须在前述同轨迹oracle中检查其实际占比和最终答案恢复，不能据这个构造再开启更大pair/gate/频率矩阵。
