# Beyond the Base：当前论文索引

当前标题：**Beyond the Base: Frequency Allocation in RoPE**。

[投稿标题与摘要纯文本](title_abstract.txt)。

当前版本以目标上下文范围内的质量为主线：**摘要不放数字**；z为研究主体，TailSpline为主要构造，clean16K/32K共同展示2L/4L任务收益。Cosh作为辅助外推搬运实例。

- [论文PDF](main.pdf)、[源码包](exponent-allocation-source.zip)、[主源文件](main.tex)。
- [本轮决策与实验安排](research/COMPARATIVE_GAP_AND_DECISION_MAP_20260915.md)：完整主张、结果身份、A1准备及M1条件。
- [主张映射](research/EXPONENT_CLAIM_EVIDENCE_MAP_20260909.md)、[证据索引](research/evidence/index.md)。
- [补充材料与复现](SUPPLEMENT_README.md)、[运行说明](runtime/README.md)。

构建：`bash paper-2027/compile.sh`；打包：`python3 paper-2027/package_source.py`。
CPU复核：`python3 paper-2027/figs/verify_field_gap.py`；主图重建：`python3 paper-2027/figs/make_allocation_value.py`。

历史修订与交接记录保留在[REVISION_BRIEF](REVISION_BRIEF.md)、[HANDOFF](HANDOFF.md)和[research索引](research/index.md)；当前验收以本轮决策映射末尾回执为准。

本轮已完成[两轮独立PDF审稿与优化](research/pdf-review-rounds/20260915_two_rounds/README.md)，
完整Natural-QA631已入正文表2，首图复用151.9M可复现的交叉对照；主文保留9页。

当前正在完成[五轮Astra/Sol独立审稿与改稿](research/pdf-review-rounds/20260915_astra_sol_five_rounds/README.md)；R05起包含四视角与AC综合，提示词和采纳判断均保留。
