# 核心证据时间线

更新：2026-09-18。本页只收纳**当前35页工作稿已经使用的核心证据**，按研究形成时间组织。具体分数、协议和来源只由链接的结果owner与[证据注册表](../../../paper-2027/research/evidence/index.md)维护；这里不复制服务器状态、执行队列或旧实验计划。

## 论文主线

论文研究对象是RoPE内部配置`z`：在实际频率范围与旋转预算给定后，内部节点怎样改变位置结构、模型使用与任务质量。证据链按以下顺序闭合：

1. 固定支持干预证明内部配置是独立变量；
2. 几何与同谱干预区分“提供了什么位置结构”和“权重如何使用它”；
3. TailSpline给出零训练冻结扩展构造，并在多模型、多长度和真实任务上验证；
4. Cosh的训练与适配实验说明频率搬运可以被学习；
5. NCP把同一配置问题推进到原生窗口。

## 2026年7–8月：配置变量、学习与兼容性

| 论文职责 | 核心证据 | Canonical owner |
|---|---|---|
| 固定实际范围后，内部配置仍能改变质量 | A01三seed固定支持、A02多形状factorial、A08成熟冻结干预 | [证据注册表A01–A08](../../../paper-2027/research/evidence/index.md) |
| 权重与频率槽存在学得兼容性 | A05权重×表crossing、A06同谱指派与补偿 | [主张映射](../../../paper-2027/research/EXPONENT_CLAIM_EVIDENCE_MAP_20260909.md) |
| 训练期和适配期能够利用新配置 | A09 432M MLA三seed、A10 750M继续学习、A12完整答案与EOS | [证据注册表A09–A12](../../../paper-2027/research/evidence/index.md) |

## 2026年9月14日：TailSpline构造与首轮冻结验证

| 论文职责 | 核心证据 | Canonical owner |
|---|---|---|
| 公共参数、零训练的解析构造 | A37有限网格TailSpline、BM与等位移控制 | [方法与评测合同](TAILSPLINE_ROPE_METHOD_AND_UNIFIED_EVAL_20260914.md) |
| Llama多长度任务、PPL与原生参照 | A39 classic与clean 32K | [Llama结果owner](TAILSPLINE_LLAMA_CLASSIC_RESULT_20260914.md) |
| OLMo跨模型迁移 | A40 classic曲线与后续clean 16K | [OLMo结果owner](TAILSPLINE_OLMO_CLASSIC_RESULT_20260914.md) |

## 2026年9月15日：大样本、真实任务与控制

| 论文职责 | 核心证据 | Canonical owner |
|---|---|---|
| clean中间长度与大样本确认 | A46 Llama clean16K、A49 OLMo clean16K | [完成实验采用表](../../../paper-2027/research/COMPLETED_EXPERIMENTS_PAPER_VALUE_20260915.md) |
| 自然语言任务，而非只看synthetic retrieval | A45 Llama Natural-QA631、A50 OLMo Natural-QA631、A53 LongBench-v2子集 | [强证据报告索引](../../../experiments/iclr2027_strong_evidence_20260915/reports/README.md) |
| 原生表与等位移形状控制 | A51 matched-dose C、A52 Native8K | [Llama结果owner](TAILSPLINE_LLAMA_CLASSIC_RESULT_20260914.md) |
| 几何供应与任务使用不是同一排序 | A42算子链、A44有限窗口核、A48有效秩与任务排序反例 | [证据注册表A42/A44/A48](../../../paper-2027/research/evidence/index.md) |

## 2026年9月16–17日：直接基线、规模迁移与原生窗口

| 论文职责 | 核心证据 | Canonical owner |
|---|---|---|
| Qwen/GLM在128K对MrPro与官方静态YaRN | A59 Full-13三臂及GLM独立换书 | [Pro6000结果owner](PRO6000_EXTREME_NATURAL_QA_RESULTS_20260916.md) |
| Llama/OLMo对官方静态YaRN的大样本比较 | A60 NIAH、PPL、Full-13与Natural-QA | [双服务器结果owner](DUAL_SERVER_YARN_AND_70B_RESULTS_20260917.md) |
| 方法规模迁移 | A61 Llama-3-70B NF4 S4/32K | [双服务器结果owner](DUAL_SERVER_YARN_AND_70B_RESULTS_20260917.md) |
| 原生窗口内的公开参数配置干预 | A54/A62 OLMo NCP：NLL、Full-13、Natural-QA与机制拆分 | [Native/NCP结果owner](../../../experiments/native_enhancement_oral_20260915/index.md) |

## 2026年9月18日：官方部署对照与理论身份收束

这是核心证据时间线的最新节点。两项工作均已完成、登记并写入当前工作稿。

| 核心证据 | 主要结论 | 入稿状态与Owner |
|---|---|---|
| A65 Kanana官方runtime YaRN直接比较 | 64K Full-13中TailSpline/MrPro/YaRN为`72.65/70.33/65.64%`；128K完整上下文English-QA中TailSpline/YaRN为`19.59/17.85%` | **已入当前稿**；[Kanana结果owner](../../../experiments/kanana_yarn_tailspline_64k_20260918/RESULT.md) |
| Minimum-bending/frame CPU审计 | Llama/OLMo单侧bending energy相对MrPro降低`95.70/95.93%`；严格支持one-sided minimum-bending身份，同时排除Nyquist或frame稳定性最优的过度解释 | **结构结论已入当前稿**；[9月18日修订owner](../../../paper-2027/research/revision_20260918/README.md) |

## 按需历史

NIAH局部反转、Native-Z5、NTS2、CA-NCP、fixed-u、C42、Phi门控和其他失败/开发实验仍保留真实结果，但不进入默认上下文。需要核实特定主张时从[完整证据注册表](../../../paper-2027/research/evidence/index.md)或[历史实验目录](../../../experiments/CATALOG_20260913.md)按ID进入，不从旧计划恢复任务。
