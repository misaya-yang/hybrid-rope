# EVQ-Cosh rebuttal workspace

最后更新：2026-07-26

这里现在只承担目录分流，不再保存一份与真实审稿并行的“大总账”。

## 当前状态

- 正式面板已逐字归档于
  `rebuttal_0723/00_REVIEWER_SCORES_AND_AC_METAREVIEW.md`：
  - AC `XLtL` metareview；
  - Reviewer `Dz6s`：4（Borderline accept），置信度 3；
  - Reviewer `zWsa`：2（Reject），置信度 5；
  - Reviewer `27bE`：3（Borderline reject），置信度 4，并保留 payload hash。
- `Dz6s` / `zWsa` / AC 当前为作者粘贴的 OpenReview 原文；工作区内尚无它们的
  独立 payload hash，不得伪称已做外部哈希校验。
- 当前数值总入口为 `rebuttal_0723/theory_results/EXPERIMENT_REPORT_20260724.md`；Phase16
  99-run 的重新审计单独记录在
  `rebuttal_0723/theory_results/PHASE16_99RUN_RAW_REANALYSIS_20260724.md`。
- 尚未形成可发送的 author response，也没有修改论文正文或主表数字。
- `pre_rebuttal/` 中的模拟 Reviewer 1/2/3 不是 OpenReview 正式源。

## 目录

| 路径 | 作用 | 权限 |
| --- | --- | --- |
| [`rebuttal_0723/`](rebuttal_0723/README.md) | 当前真实审稿周期的唯一入口：review、AC、concern mapping、实验与结果状态 | 当前 |
| [`pre_rebuttal/`](pre_rebuttal/README.md) | 真实 reviews 到来前的理论审计、风险清单、模拟审稿、旧实验方案和旧实现 | 历史参考 |

根目录不再放单独的 rebuttal 策略文档。真实问题从
`rebuttal_0723/00_REVIEWER_SCORES_AND_AC_METAREVIEW.md` 开始，随后由
`rebuttal_0723/01_REBUTTAL_PLAYBOOK.md` 路由精选证据与行动；历史材料只有在当前 concern
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

项目级操作边界见 [`AGENTS.md`](../AGENTS.md)，论文与 reviewer-safe 结果的事实
边界见 `docs/overview/RESULT_PROVENANCE_MANIFEST.md`。
