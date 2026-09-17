# Beyond the Base：当前论文索引

## 当前证据增量版

后续每日改稿参考[改稿总结（日记）](research/PAPER_REVISION_DIARY.md)：吸收历次纠错，形成可说明的净增强，并评估收益、分数变化与可能损失。

标题保持 **Beyond the Base: Frequency Allocation in RoPE**，原章节结构保持。
主文9页、全稿34页；摘要无数字，第一页无图。
新增大样本静态YaRN、四模型Full-13、70B NF4、GLM第二书池及NCP原生固定支持结果。
NCP既有原生内容保留并补充同目标NLL；NTS2留在研究记录，等待更完整QA确认。
432M曲线、逐任务主图和自然QA主表区间继续保留在正文。

- [本轮写作skills、逐章审核和最终修复](research/revision_20260917_final_editorial/README.md)。
- [此前证据升级、Pro取舍和审稿经验回查](research/revision_20260917_evidence_update/README.md)。
- [本轮修改前基线v3](history/v3.pdf)：上一轮交付的main.pdf原样快照，history只保存PDF。
  每轮开始先将当前未改稿存为最大历史版本加一，再作新旧稿双模型PDF对读。
- [投稿标题与摘要](title_abstract.txt)。
- [此前附录整理记录](research/appendix_restructure_20260916/README.md)：对应9/29页旧版，关键材料延续。
- [后续实验准备](../docs/research/next_stage_20260912/PAPER_NEXT_REVISION_PREPARATION_20260916.md)：
  已完成部分已入本版，在跑实验待下一增量。

总页数目标不超过35页、硬上限40页，是作者的编辑约束。
研究对象为z；151.9M固定端点属于50.9M–750M Cosh训练证据链，
TailSpline提供冻结扩展证据，NCP保留固定支持原生归因作用。

- [论文PDF](main.pdf)、[源码包](exponent-allocation-source.zip)、[主源文件](main.tex)。
- [本轮决策与实验安排](research/COMPARATIVE_GAP_AND_DECISION_MAP_20260915.md)：完整主张、结果身份、A1准备及M1条件。
- [主张映射](research/EXPONENT_CLAIM_EVIDENCE_MAP_20260909.md)、[证据索引](research/evidence/index.md)。
- [核心实验一览](../docs/research/next_stage_20260912/KEY_EXPERIMENT_COMPASS_20260914.md)：主结果、支持证据、负结果与执行状态。
- [现状与修改必要性审计](research/audits/STATE_AND_REVISION_NECESSITY_AUDIT_20260915.md)：文档修正、现稿已解决的问题和仍值得考虑的局部优化。
- [补充材料与复现](SUPPLEMENT_README.md)、[运行说明](runtime/README.md)。

构建：`bash paper-2027/compile.sh`；打包：`python3 paper-2027/package_source.py`。
CPU复核：`python3 paper-2027/figs/verify_completed_evidence.py`；既有检查：`python3 paper-2027/figs/verify_field_gap.py`；主图重建：`python3 paper-2027/figs/make_allocation_value.py`。

历史修订与交接记录保留在[REVISION_BRIEF](REVISION_BRIEF.md)、[HANDOFF](HANDOFF.md)和[research索引](research/index.md)；当前验收以最终逐章审核的验证记录为准。

本轮已完成[两轮独立PDF审稿与优化](research/pdf-review-rounds/20260915_two_rounds/README.md)，
完整Natural-QA631已入§6.1段落与附录H.9（表41）；该轮表号为历史位置；当前表1为跨长度/模型clean结果，表2为自然任务结果。
首图复用151.9M可复现的交叉对照；主文保留9页。

已完成[五轮Astra/Sol独立审稿与改稿](research/pdf-review-rounds/20260915_astra_sol_five_rounds/README.md)；R05起包含四视角与AC综合，提示词和采纳判断均保留。
