# Beyond the Base：当前论文入口

**2026-09-18：标题与167词摘要已封板。** 正文9页、全文35页，第一页无图。
PDF、投稿文本与源码包同步；Astra／Sol独立对读已完成。

## 交付与接续

- [论文PDF](main.pdf)、[投稿标题摘要](title_abstract.txt)、[源码包](exponent-allocation-source.zip)。
- [当前handoff](HANDOFF.md)：新会话所需状态、约束、文件分工与风险。
- [作者工作约定](research/AUTHOR_WORKING_CONTRACT.md)：科学定位和历次纠错。
- [跨机器接续](../docs/maintenance/CROSS_MACHINE_CONTINUATION.md)：未提交工作树、仓库技能与PC环境。

## 修改一篇论文，而不是叠加实验清单

研究对象是z；TailSpline承担主构造和冻结扩展，NCP保留原生LM／上下文利用研究，
Cosh保留训练、继续训练和适配证据。固定支持与等位移控制回答归因问题。
当前首图为YaRN／MrRoPE-Pro／TailSpline配置与距离响应概念图；432M曲线和完整逐任务主图仍在正文。

| 需要做什么 | 入口 |
|---|---|
| 了解本轮改变及保留内容 | [修订概览](REVISION_BRIEF.md) |
| 查本轮三个阶段及审稿 | [9月18日修订owner](research/revision_20260918/README.md) |
| 查正文主张对应的证据 | [当前主张映射](research/EXPONENT_CLAIM_EVIDENCE_MAP_20260909.md) |
| 查具体结果、协议与来源 | [证据索引](research/evidence/index.md) |
| 按研究形成过程看成果 | [核心证据时间线](../docs/research/next_stage_20260912/KEY_EXPERIMENT_COMPASS_20260914.md) |
| 修改或验证稿件 | [仓库改稿技能](../.agents/skills/hybrid-rope-paper-editing/SKILL.md) |
| 比较新旧稿、判断是否倒退 | [仓库审稿技能](../.agents/skills/hybrid-rope-regression-review/SKILL.md) |
| 查当前实验／研究任务 | [研究入口](../docs/research/next_stage_20260912/index.md) |
| 查后续一周待Pro判断的问题 | [三个问题](research/revision_20260918/PRO_FINAL_WEEK_QUESTIONS.md) |

## 构建与版本

仓库根运行 `bash paper-2027/compile.sh`，再用
`python3 paper-2027/package_source.py` 打包。跨机器依赖见接续指南；数字生成与运行代码身份见[runtime说明](runtime/README.md)和[补充材料说明](SUPPLEMENT_README.md)。

- 当前概念首图：`python3 paper-2027/figs/make_intro_claim.py`。
- 当前证据表：`python3 paper-2027/figs/make_revision_evidence.py`。
- 既有核心图：`python3 paper-2027/figs/make_allocation_value.py`；不要用旧首图覆盖新概念图。
- 证据索引：`python3 scripts/render_evidence_index.py`；导航检查：`python3 scripts/check_repository_docs.py`。

作者要求正文9页、总稿优先≤35且最多40页。总稿上限是作者编辑约束。
通常新轮先将当时未修改PDF保存为下一history/vN；本轮v4后的连续章节／摘要修改
按作者要求不新增版本。具体比较基线见[便携审稿清单](research/revision_20260918/portable_review_artifacts.json)。

日常决策继续追加[改稿日记](research/PAPER_REVISION_DIARY.md)。旧实验准备、旧审稿分数和旧章节位置从[研究历史入口](research/index.md)按需读取，不作为默认开工清单。
