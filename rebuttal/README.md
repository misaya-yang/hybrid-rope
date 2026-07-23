# EVQ-Cosh rebuttal workspace

最后更新：2026-07-24

这里现在只承担目录分流，不再保存一份与真实审稿并行的“大总账”。

## 当前状态

- 已收到并逐字归档 Reviewer `27bE` 的正式 review：评分 3、置信度 4。
- 仓库内目前只有这一份可逐字核验的正式 review。0723 实验计划中对
  `zWsa`、`Dz6s` 的概括不是原始 review source，补齐原文前不得当作 reviewer
  原话。
- 尚未形成可发送的 author response。
- `rebuttal_0723/` 内两个实验包目前只有协议和代码；仓库内没有训练结果或
  reviewer-grade result artifact。
- 当前实验协议已把含混的 `Geo` 拆成 `Std-Geo` 与 `Paper-Geo`，主比较固定
  为 `Paper-Geo vs EVQ-Cosh`，并修复 checkpoint `inv_freq` 持久化/加载
  保护。没有修改论文正文或任何实验数字。

## 目录

| 路径 | 作用 | 权限 |
| --- | --- | --- |
| [`rebuttal_0723/`](rebuttal_0723/README.md) | 当前真实审稿周期的唯一入口：原始 review、concern mapping、获批实验及结果状态 | 当前 |
| [`pre_rebuttal/`](pre_rebuttal/README.md) | 真实 reviews 到来前的理论审计、风险清单、模拟审稿、旧实验方案和旧实现 | 历史参考 |

根目录不再放单独的 rebuttal 策略文档。真实问题从
`rebuttal_0723/00_REVIEWER_27BE_OFFICIAL_REVIEW.md` 开始；历史材料只有在
当前 concern 明确触发时才回查。

## 当前执行原则

1. 先回答 reviewer 实际提出的问题，不把 rebuttal 扩成第二次投稿。
2. reviewer 原话、仓库事实、拟议动作、已完成证据必须分栏记录。
3. 方法身份或数学错误不能靠新实验修复。若回答涉及对应内容，必须使用审计后
   的窄口径。
4. 新实验只有在直接区分 score-changing 假设、协议已冻结、运行产物可追溯时
   才可能进入回复。代码存在不等于结果存在，外部机器状态也不等于仓库证据。
5. 模拟审稿和宽实验计划只能提供检索线索，不得冒充正式 review 或已批准任务。
6. `rebuttal/` 整体不得直接打进 reviewer supplement；任何对外 artifact 仍需
   匿名化和 provenance gate。

项目级操作边界见 [`Agent.md`](../Agent.md)，论文与 reviewer-safe 结果的事实
边界见 `docs/overview/RESULT_PROVENANCE_MANIFEST.md`。
