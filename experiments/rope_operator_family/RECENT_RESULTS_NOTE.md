# 已回看结果的简记

2026-09-09。只记录本轮已经查到的主要结果与应保留的解释，不继续扩展历史审查，也不恢复旧实验队列。来源是既有结果JSON、owner报告和已有原始输出复核回执；本轮没有重新生成这些模型输出。

**这些工作不能统称为失败。** 其中有真实正结果、有清楚的当前方法负结果、有未完成的训练，也有没能回答目标问题的实验。方法定位应保留有用发现，并把失败限定到实际受检的方案。

## 应保留的结果与应修正的解释

| 结果 | 已有事实 | 本轮应采用的判断 |
|---|---|---|
| **BM / OLMo：正结果** | 631条长输入自然QA，五项等权F1从21.62%到25.44%，+3.82pp，配对区间约[+1.32,+6.29]pp；五项均为正差 | 这是实际自然任务收益，值得保留。它不是因为尚无跨模型统一优势就变成失败 |
| **BM / Qwen：迁移负结果** | 3B的128K小面板从78.12%到70.83%；7B从84.44%到71.11% | 固定BM公式没有把OLMo长端收益迁移到这些条件；停止相应全量扩展合理，但OLMo的正结果仍成立 |
| **FullLagP2：有用的小面板信号** | 1.5B、64K：多key37.5对12.5、VT87.5对82.5、FWE70.83对45.83；同gain对照也有相应改善 | 小样本的真实配对收益有定位价值。后来128K和3B的取舍限制适用范围，不抹掉这批实例的改善 |
| **C2：低维描述的正结果与短端代价并存** | 两参数C2在Qwen64K/128K达到67.75%/57.25%，原64点transport为67.25%/54.50%；OLMo的Native PPL门槛差约0.00462 NLL未过 | 支持“精细64点描述并非这些长端能力所必需”。未通过当时部署门槛，不等于低维路线失败 |
| **Sparse TP：测试设置没完成有效比较** | V1两边100%来自marker邻居捷径；V2约10.06%对9.86%，不压缩控制也未建立关键异query能力 | 应暂停这套scratch任务/训练设置；没有有效依据据此否定TP或更新的压缩算子 |
| **缓存聚类：修复均值摘要有效，位置特有收益未建立** | 32题Mean0、Post10、Pre11；更细Quest对照13 | 保存“完整内容摘要能恢复一部分答案”的发现；当前候选没有超过这个简单相关对照，继续维护位置创新叙事缺少依据 |
| **header/评分边界：不是能力从0恢复到4** | 8题旧raw exact+EOS为0、trimmed为4；新版本两者均4 | 内容正确数没有改变，改变的是格式/评分边界。纯格式差异应按研究目标单独处理 |
| **原Carrier：有效负结果** | 背景复响应构造指标降低73.85%，但同输入128K VT从62.5降到0，输出出现实质错误 | 当前闭式构造没有带来所需能力，足以停止这次候选。没有必要先跑完一张大矩阵才承认它无效；也不据此否定所有去载波方法 |
| **59/128步LoRA：未完成，不是质量失败** | 完成59个更新后被SIGTERM收尾，没有最终adapter及训练后生成 | 没有训练后能力结果，不能把它列作完整适配路线的负结果 |

### C2的代价也要完整保留

C2虽然只差约0.00462 NLL未过OLMo设定的门槛，但这不是它相对Native的全部代价。Qwen32K中，原Native为82.0%，静态C2为71.25%，存在实际短端损失。正确结论是：**长端低维描述有效，统一静态部署存在短端取舍。** 既不把它全部判死，也不把代价藏成“仅差一点门槛”。

### 更早的换表反例应保留原意

常被引用的50M结果中，Geo训练模型使用Geo表的PPL为7.14，直接换Cosh表后为76.20；Cosh训练模型使用自己的表为7.16。它有效否定了“冻结权重直接换成几何更好的表就自然改善”的做法，支持内容计算与训练时旋转共适应。它没有否定Cosh训练方案，更没有测试今天的A/B/旋转联合压缩方法。

### 确实有一些实验没有回答原问题

9月9日已有复盘指出：原计划要求分析**真实错误候选对**的摘要/位置误差与最终答案，实际首轮主要保存全体query/块的平均保留率、value误差和log-mass。平均量可以作描述，但它没有完成那个条件性错误归因问题。此时结论应是“原问题尚未测到”，而不是“位置路线被验证失败”。

## 对本轮实验的直接影响

1. 用 `position_response_mse` 直接测相对位置响应，用同一依赖任务的答案/条件NLL测实际价值；不拿混合训练loss代替这两个问题。
2. 一个方法完成后，只选一个简单相关对照。清楚的配对改善就是方法定位的有效信号；不要求先补齐完整论文矩阵。
3. 当结果不好，记录具体失败对象。工程报错、训练中断、无区分度的测试与实际能力退化分别处理，不扩大成整个方向失败。

这些解释不改写旧输出，也不将当时没有通过的部署门槛重命名为通过。修正的是它们在今天的方法选择中应承担的含义。

## 已查看的来源

- [OLMo BM结果](../../docs/research/ROPE_OLMO_BM_RESULT_20260908.md)、[五项自然QA](../../docs/research/ROPE_OLMO_BM_FIVE_QA_RESULT_20260908.md)、[逐题结果JSON](../../docs/research/ROPE_OLMO_BM_FIVE_QA_RESULT_20260908.json)。
- [Qwen3B迁移](../../docs/research/ROPE_BM_TRANSFER_RESULT_20260908.md)、[Qwen7B迁移](../../docs/research/ROPE_QWEN7_BM_TRANSFER_20260908.md)。
- [FullLagP2逐项配对结果](../../docs/research/ROPE_QWEN15_FULL_LAG_P2_RESULT_20260907.json)、[整夜结果及LoRA中断](../../docs/research/ROPE_OVERNIGHT_EXPERIMENT_REVIEW_20260908.md)。
- [C2完整结果](../../paper-2027/research/attention-aware-retrofit/results/coupling-transfer/LOW_DIM_COUPLING_GPU_RESULT_20260901.md)、[结果回执](../../paper-2027/research/attention-aware-retrofit/evidence/LOW_DIM_COUPLING_GPU_RECEIPT_20260901.json)。C2实验本身早于这几天，此处因最近的失败解释再次引用而列入。
- [Sparse TP记录](../../docs/research/SPARSE_MEMORY_INTERFACE_PILOT_20260908.md)、[原始指标回执](../../docs/research/SPARSE_MEMORY_INTERFACE_RESULTS_20260908.json)。
- [缓存实验的保存输出复核](../../docs/research/OVERNIGHT_FAILURE_AUDIT_20260909.json)、[原计划与实际测量的复盘](../../docs/research/OVERNIGHT_FAILURE_POSTMORTEM_20260909.md)。
- [原Carrier配对生成](../../docs/research/ROPE_CARRIER_REMOVAL_PILOT_20260907.md)。
- [换表交叉结果的来源索引](../../paper-2027/research/EXPERIMENT_ASSETS_TOP15_20260909.md)、[此前已作的失败解释修正](../../docs/research/ROPE_LOCAL_FAILURE_SYNTHESIS_20260908.md)。
