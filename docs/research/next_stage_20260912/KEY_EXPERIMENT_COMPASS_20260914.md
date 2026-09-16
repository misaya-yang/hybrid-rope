# Hybrid-RoPE关键实验罗盘

更新：2026-09-16。本文区分已入稿、完成待整合、运行中与仅准备。
具体数值、任务分解和原始身份由链接的结果owner维护，不另复制一套完整报告。
下一轮编辑与实验取舍见[统一准备](PAPER_NEXT_REVISION_PREPARATION_20260916.md)。

## 当前认识

内部配置z在实际范围与旋转预算给定后仍影响质量。TailSpline提供零训练扩展构造，
NCP已有原生窗口总体增益，Cosh提供配对学习与外推支持。
已完成的模型/任务证据覆盖Llama、OLMo、Qwen与GLM。
几何秩、相位幅度、总位移和参考风险各描述不同对象，不充当任务优劣的通用排序。

## 一、已进入当前9/29页论文

| 证据 | 主要职责 | 唯一结果/核查入口 |
|---|---|---|
| Llama S4 clean8/16/32K Full-13 | 同一静态扩展表的多长度任务质量；Native原始表另列 | [Llama owner](TAILSPLINE_LLAMA_CLASSIC_RESULT_20260914.md) |
| OLMo S4 clean16K Full-13、经典曲线 | 另一模型族的匹配配置收益 | [OLMo owner](TAILSPLINE_OLMO_CLASSIC_RESULT_20260914.md) |
| OLMo Natural-QA631；Llama Natural-QA631/LongBench-v2 | 自然任务的分模型/分指标结果 | [本批入稿分析与源报告](../../../paper-2027/research/COMPLETED_EXPERIMENTS_PAPER_VALUE_20260915.md) |
| Llama clean T/C | 匹配总位移后32K仍有形状差异，16K接近 | [Llama owner](TAILSPLINE_LLAMA_CLASSIC_RESULT_20260914.md) |
| NCP OLMo原生4K、780条 | 公开参数、固定端点、gain1的原生Full-13总体提升；完整任务轮廓保留 | [NCP正式报告](../../../experiments/iclr2027_strong_evidence_20260915/reports/olmo_native_ncp.json) |
| 151.9M固定支持三seed、权重×表crossing | 训练期配置作用、范围交互和学得兼容性 | [A01](../../../paper-2027/research/evidence/EXACT_RANGE_151M_3SEED_RESULT_20260820.md)、[当前主张表](../../../paper-2027/research/EXPONENT_CLAIM_EVIDENCE_MAP_20260909.md) |
| 完整pair几何、同谱指派干预、BM–Uni | 位置结构、已学内容坐标与等位移控制 | [证据索引](../../../paper-2027/research/evidence/index.md) |
|432M MLA、750M续训、OLMo匹配适配 | Cosh学习/外推支持；各自网格、训练和读出协议分开 | [附录F](../../../paper-2027/appendix/compact_f_learning.tex) |

## 二、新完成，等待下一版统一整合

| 证据 | 新增加的认识 | 结果owner |
|---|---|---|
| Qwen S4/128K En.QA | 同S4下自然问答已出现配置收益；不能再概括Qwen尚无优势 | [Pro6000结果](PRO6000_EXTREME_NATURAL_QA_RESULTS_20260916.md) |
| Qwen S8/256K single-NIAH三任务及LongBook PPL5 | 更远检索与平均建模质量分别呈现，不混成一个分数 | [Pro6000结果](PRO6000_EXTREME_NATURAL_QA_RESULTS_20260916.md) |
| Llama S16/128K Full-13 gate、PPL10、En.QA/En.Dia | 任务类型相关的高倍率表现，正负结果均保留 | [Pro6000结果](PRO6000_EXTREME_NATURAL_QA_RESULTS_20260916.md) |
| Qwen/GLM S4 Full-13×10三臂 | 两个32K-native模型在128K对MrPro与官方静态YaRN的直接比较 | [Pro6000结果](PRO6000_EXTREME_NATURAL_QA_RESULTS_20260916.md) |
| GLM独立第二书池En.QA | 77题、15个新来源簇；换书后TailSpline仍为三臂第一 | [Pro6000结果](PRO6000_EXTREME_NATURAL_QA_RESULTS_20260916.md) |
| Llama既有2600条的抽样稳定性分析 | 描述当前固定总体中小样本的变化，非新模型实验 | [抽样报告](../../../experiments/iclr2027_strong_evidence_20260915/reports/pro6000_ruler_sampling_stability.json) |

[Base综合分析](../reviews/QWEN_BASE_SCALE_SYNTHESIS_20260916.md)已包含新Qwen结果与独立公共几何复算。
Qwen不同面板的单针/八任务/Full-13不能混为同一条曲线。

## 三、旧结果仍有效，解释保持协议范围

| 资产 | 当前用途 | 来源 |
|---|---|---|
| Qwen S2 Core-6、S4 NIAH-8 | 旧面板的接近和任务异号仍保留；不覆盖新自然QA/S8检索 | [辅助结果](SECONDARY_GPU_RESULTS_20260915.md) |
| Llama Full20单针与classic Native参照 | 饱和/任务区别及历史参照，不替代clean Full-13/Native | [辅助结果](SECONDARY_GPU_RESULTS_20260915.md)、[Llama owner](TAILSPLINE_LLAMA_CLASSIC_RESULT_20260914.md) |
| Native-Z5 V1、consensus、all50 | 各自校准与确认身份；其后续未通过不否定独立NCP实验 | [Native-Z5 owner](NATIVE_Z5_EXPLORATION_RESULT_20260915.md) |
| C42/C42V24 | 等位移形状开发证据；重心是同一总位移约束的结果，不另算一项控制 | [原始判决](../../../ds_workspace/recon_20260910/verdicts/HEADLINE_20260911.md) |
| C2、fixed-u、proxy选表 | 保留实际失效与反例，不重启曲线搜索挽救旧候选 | [证据索引](../../../paper-2027/research/evidence/index.md)、[fixed-u](OLMO_S8_FIXED_U_TRANSPORT_RESULT_20260914.md) |

“尚未确认/接近/负向”必须带模型、任务、倍率和协议，不能扩大为整条配置轴失效。
同样，多个正结果不升级为所有模型/任务/长度的保证。

## 四、数学与CPU证据

| 资产 | 已完成内容 | 当前作用 |
|---|---|---|
| [TailSpline构造](TAILSPLINE_ROPE_METHOD_AND_UNIFIED_EVAL_20260914.md) | 有限差分目标唯一解、BM及等位移恒等式 | 可安装的公共构造与控制，不声称真实模型最优 |
| [理论算子链](THEORY_DEEPENING_CPU_VERIFICATION_20260915.md) | 相位、logit、key竞争、value读出的条件计算 | 与任务证据连接；不是全模型中介实验 |
| [Base/turn归一化](../reviews/QWEN_BASE_SCALE_SYNTHESIS_20260916.md) | 不同base下的band、采样与相位响应 | 排除简单“大base抹掉干预”解释，不按RMS选赢家 |
| [Pro重评核验](../reviews/PRO_REASSESSMENT_DISPOSITION_20260916.md) | Cosh目标、竞争恒等式、非零和反例、NCP既有分数重聚合 | 概念澄清与下一版取舍，无新模型生成 |

## 五、执行与准备分开

易变状态只认[带时间戳执行owner](../../../experiments/iclr2027_three_track_sprint_20260915/SERVER_TASK_LAYERS.md)。
2026-09-16 Pro6000队列已完成并归档：Qwen/GLM S4 Full-13三臂、自然QA三臂、
GLM第二独立书池均有正式报告。Llama/OLMo的大样本YaRN任务仍属于32GB队列。

NCP新来源确认、Llama clean32K YaRN和288题反事实面板是[下一版准备项](PAPER_NEXT_REVISION_PREPARATION_20260916.md)，
不是本轮已启动任务。当前不因计划文件、公开建议或旧launcher存在自动运行。
