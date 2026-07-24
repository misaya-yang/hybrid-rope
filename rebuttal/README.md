# EVQ-Cosh rebuttal workspace

最后更新：2026-07-24

这里现在只承担目录分流，不再保存一份与真实审稿并行的“大总账”。

## 当前状态

- Reviewer `27bE` 的正式 review 已逐字归档：评分 3、置信度 4。
- AC metareview 已单独归档；其文本由作者提供，但当前工作区没有独立 URL 或
  payload hash，因此不能伪称已做外部来源校验。
- 当前数值总入口为 `rebuttal_0723/EXPERIMENT_REPORT_20260724.md`；Phase16
  99-run 的重新审计单独记录在
  `rebuttal_0723/PHASE16_99RUN_RAW_REANALYSIS_20260724.md`。
- 尚未形成可发送的 author response，也没有修改论文正文或主表数字。
- 仓库内仍没有其他 reviewer 的逐字 source。0723 宽计划中对 `zWsa`、
  `Dz6s` 的概括不得当作 reviewer 原话。

## 目录

| 路径 | 作用 | 权限 |
| --- | --- | --- |
| [`rebuttal_0723/`](rebuttal_0723/README.md) | 当前真实审稿周期的唯一入口：review、AC、concern mapping、实验与结果状态 | 当前 |
| [`pre_rebuttal/`](pre_rebuttal/README.md) | 真实 reviews 到来前的理论审计、风险清单、模拟审稿、旧实验方案和旧实现 | 历史参考 |

根目录不再放单独的 rebuttal 策略文档。真实问题从
`rebuttal_0723/00_REVIEWER_27BE_OFFICIAL_REVIEW.md` 和
`rebuttal_0723/01_AC_METAREVIEW.md` 开始；历史材料只有在当前 concern
明确触发时才回查。

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
