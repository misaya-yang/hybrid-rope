# 当前研究入口

研究对象是RoPE内部配置`z`：固定实际频率范围与旋转预算后，内部节点仍可改变位置结构、模型使用与任务质量。TailSpline承担冻结扩展，Cosh提供冻结／学习／适配证据，NCP研究原生窗口内的固定支持配置干预。

2026-09-18当天已采用实验和改稿已收束，标题／摘要封板，正文9页、全文35页。Pro对[三个问题](../../../paper-2027/research/revision_20260918/PRO_FINAL_WEEK_QUESTIONS.md)的答复已由作者审阅；获批的[修正T-C距离×结构化竞争确认实验](../../../experiments/tc_distance_competition_20260918/README.md)已经完成。Stage 1完整分母与官方scorer对账成立；fresh-64主交互为`+3.125pp`，95%区间`[-4.6875,+10.9375]pp`，并受near天花板／far地板限制，按预注册判为未解决，不改变当前稿件主张。当前没有其他授权GPU队列。

## 按问题选择入口

1. [当前论文索引](../../../paper-2027/index.md)：当前35页工作稿、PDF与章节身份。
2. [核心证据时间线](KEY_EXPERIMENT_COMPASS_20260914.md)：只列已经写入论文的核心实验，按研究形成时间组织。
3. [论文证据注册表](../../../paper-2027/research/evidence/index.md)：按资产ID核对具体数字、协议和来源。
4. [9月18日修订owner](../../../paper-2027/research/revision_20260918/README.md)：Kanana与理论审计怎样进入当前工作稿。

已知具体结果时直接读取对应owner，不再先读计划、服务器交接或完整目录。

新会话接续见[当前handoff](../../../paper-2027/HANDOFF.md)；实验准备与执行见[工作流程](../protocols/EXPERIMENT_WORKFLOW.md)。无需把上面四项都作为开工前置阅读。

## 当前证据收口

| 问题 | Canonical owner |
|---|---|
| Llama/OLMo上的TailSpline构造、经典曲线、clean确认与自然QA | [核心证据时间线](KEY_EXPERIMENT_COMPASS_20260914.md)中的9月14–15日条目 |
| Qwen/GLM 128K、官方静态YaRN与70B迁移 | [9月16–17日条目](KEY_EXPERIMENT_COMPASS_20260914.md) |
| NCP原生窗口NLL、Full-13、Natural-QA和机制边界 | [Native/NCP结果owner](../../../experiments/native_enhancement_oral_20260915/index.md) |
| Kanana官方runtime YaRN的64K/128K新结果 | [Kanana结果owner](../../../experiments/kanana_yarn_tailspline_64k_20260918/RESULT.md)，属于9月18日最新核心证据并已写入当前工作稿 |

## 非默认上下文

- [实验索引](../../../experiments/index.md)只用于找结果owner和可复用代码。
- [历史研究目录](CATALOG_20260914.md)和[历史实验目录](../../../experiments/CATALOG_20260913.md)只在追溯旧结果、失败方法或旁线时使用。
- 旧计划、服务器任务分层、launcher和准备完成状态都不定义当前优先级；它们存在不表示实验待跑。
- Native-Z5、NTS2、CA-NCP、fixed-u、C42、Phi等按需从证据注册表进入，不放入默认启动上下文。
