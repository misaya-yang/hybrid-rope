# 2026-09-18论文增量准备记录

**状态：已应用。** 当前结果见[9月18日修订owner](../revision_20260918/README.md)。本页保留改稿前的筛选与边界，不再作为默认入口或当前PDF状态。

## 时间线最新证据

| 证据 | 完成结论 | 当前状态 |
|---|---|---|
| A65 Kanana官方runtime YaRN比较 | 64K Full-13中TailSpline/MrPro/YaRN为`72.65/70.33/65.64%`；128K完整上下文English-QA中TailSpline/YaRN为`19.59/17.85%` | 已写入当前工作稿；[结果owner](../../../experiments/kanana_yarn_tailspline_64k_20260918/RESULT.md) |
| Minimum-bending/frame CPU审计 | TailSpline严格最小化声明的单侧bending目标；Llama/OLMo能量相对MrPro降低`95.70/95.93%`；frame审计不支持Nyquist或稳定性最优 | 结构结论已写入当前工作稿；[完整数值](minimum_bending_mesh_cpu_audit.json) |

两项均属于[核心证据时间线](../../../docs/research/next_stage_20260912/KEY_EXPERIMENT_COMPASS_20260914.md)的9月18日节点。

## 下一轮改稿边界

- 保留标题、章节结构、固定支持识别图、432M学习曲线和完整逐任务主图。
- TailSpline继续承担冻结扩展与直接基线比较；Cosh保留训练／适配证据；NCP保留原生窗口研究职责。
- Kanana的价值是官方部署配方与完整长书QA，不写成新增注意力架构或已确认的YaRN续训模型。
- Minimum-bending进入方法解释时必须写明单侧边界；不升级为Nyquist、frame稳定性或任务机制最优。
- 不从旧计划、服务器交接或遗留launcher恢复实验。

## 进一步读取

- [当前论文索引](../../index.md)
- [核心证据时间线](../../../docs/research/next_stage_20260912/KEY_EXPERIMENT_COMPASS_20260914.md)
- [证据注册表](../evidence/index.md)
- [完整9月18日专题讨论与作者取舍](README.md)：仅在处理具体理论、篇幅或叙事问题时读取，不是默认上下文。
