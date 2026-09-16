# Beyond the Base：当前论文索引

## 下一版准备，暂不改稿

当前等待GLM及本批后续实验结果后统一整合。已更新
[准备清单](../docs/research/next_stage_20260912/PAPER_NEXT_REVISION_PREPARATION_20260916.md)、
[Pro取舍](../docs/research/reviews/PRO_REASSESSMENT_DISPOSITION_20260916.md)及
[十篇审稿经验](../docs/research/reviews/TEN_PAPER_REVIEW_LESSONS_20260916.md)。
Qwen等新完成结果尚未自动写入当前PDF；下方是已交付稿件状态。


当前标题：**Beyond the Base: Frequency Allocation in RoPE**。

[投稿标题与摘要纯文本](title_abstract.txt)。

最新已完成[附录整理与完整性检查](research/appendix_restructure_20260916/README.md)：
主文9页、总计29页（此前69页），摘要158词且无数字，第一页无图。
六个主题附录保留当前主张所需证明、实验协议和分任务结果；历史旁支保留源码归档。
总页数目标不超过35页、硬上限40页；这是作者的编辑约束，ICLR 2027附录本身无页数上限。
[本轮验证](research/appendix_restructure_20260916/validation.json)记录编译、数值和源码包检查。
上一轮[新落盘结果与NCP正文整合](research/COMPLETED_EXPERIMENTS_PAPER_VALUE_20260915.md)的结果全部延续。

研究对象为z：受控干预识别其作用，完整旋转对理论区分位置结构与模型使用；TailSpline改善
外推质量，NCP展示in-window性能增益，Cosh提供学习与外推辅助证据。
[R08独立PDF审稿](research/pdf-review-rounds/20260915_theory_integration_r08/README.md)对应此前65页版本，
本次未启动额外审稿轮次。

- [论文PDF](main.pdf)、[源码包](exponent-allocation-source.zip)、[主源文件](main.tex)。
- [本轮决策与实验安排](research/COMPARATIVE_GAP_AND_DECISION_MAP_20260915.md)：完整主张、结果身份、A1准备及M1条件。
- [主张映射](research/EXPONENT_CLAIM_EVIDENCE_MAP_20260909.md)、[证据索引](research/evidence/index.md)。
- [核心实验一览](../docs/research/next_stage_20260912/KEY_EXPERIMENT_COMPASS_20260914.md)：主结果、支持证据、负结果与执行状态。
- [现状与修改必要性审计](research/audits/STATE_AND_REVISION_NECESSITY_AUDIT_20260915.md)：文档修正、现稿已解决的问题和仍值得考虑的局部优化。
- [补充材料与复现](SUPPLEMENT_README.md)、[运行说明](runtime/README.md)。

构建：`bash paper-2027/compile.sh`；打包：`python3 paper-2027/package_source.py`。
CPU复核：`python3 paper-2027/figs/verify_completed_evidence.py`；既有检查：`python3 paper-2027/figs/verify_field_gap.py`；主图重建：`python3 paper-2027/figs/make_allocation_value.py`。

历史修订与交接记录保留在[REVISION_BRIEF](REVISION_BRIEF.md)、[HANDOFF](HANDOFF.md)和[research索引](research/index.md)；当前验收以本轮附录整理验证为准。

本轮已完成[两轮独立PDF审稿与优化](research/pdf-review-rounds/20260915_two_rounds/README.md)，
完整Natural-QA631已入§6.1段落与附录H.9（表41）；该轮表号为历史位置；当前表1为跨长度/模型clean结果，表2为自然任务结果。
首图复用151.9M可复现的交叉对照；主文保留9页。

已完成[五轮Astra/Sol独立审稿与改稿](research/pdf-review-rounds/20260915_astra_sol_five_rounds/README.md)；R05起包含四视角与AC综合，提示词和采纳判断均保留。
