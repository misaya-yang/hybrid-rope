# Beyond the Base：当前论文索引

当前标题：**Beyond the Base: Frequency Allocation in RoPE**。

[投稿标题与摘要纯文本](title_abstract.txt)。

最新已完成[理论整合与R08独立PDF审稿](research/pdf-review-rounds/20260915_theory_integration_r08/README.md)：
主文9页、总计65页，摘要156词且无数字。新增Appendix K与可独立运行的CPU复算；
Astra/Sol AC均7/10、接收倾向。当前交付回执见
[理论整合验证](research/THEORY_INTEGRATION_VALIDATION_20260915.json)。

现稿围绕三个发现展开：范围与内部配置具有不同且相互作用的效果；位置结构与学得使用可区分；
解析配置带来任务收益。z为研究对象，TailSpline为主要冻结构造，clean16K/32K共同展示
同一静态表在2L/4L的任务收益；Cosh作为辅助外推搬运实例。**摘要不放数字**。

- [论文PDF](main.pdf)、[源码包](exponent-allocation-source.zip)、[主源文件](main.tex)。
- [本轮决策与实验安排](research/COMPARATIVE_GAP_AND_DECISION_MAP_20260915.md)：完整主张、结果身份、A1准备及M1条件。
- [主张映射](research/EXPONENT_CLAIM_EVIDENCE_MAP_20260909.md)、[证据索引](research/evidence/index.md)。
- [核心实验一览](../docs/research/next_stage_20260912/KEY_EXPERIMENT_COMPASS_20260914.md)：主结果、支持证据、负结果与执行状态。
- [现状与修改必要性审计](research/audits/STATE_AND_REVISION_NECESSITY_AUDIT_20260915.md)：文档修正、现稿已解决的问题和仍值得考虑的局部优化。
- [补充材料与复现](SUPPLEMENT_README.md)、[运行说明](runtime/README.md)。

构建：`bash paper-2027/compile.sh`；打包：`python3 paper-2027/package_source.py`。
CPU复核：`python3 paper-2027/figs/verify_field_gap.py`；主图重建：`python3 paper-2027/figs/make_allocation_value.py`。

历史修订与交接记录保留在[REVISION_BRIEF](REVISION_BRIEF.md)、[HANDOFF](HANDOFF.md)和[research索引](research/index.md)；当前验收以本轮决策映射末尾回执为准。

本轮已完成[两轮独立PDF审稿与优化](research/pdf-review-rounds/20260915_two_rounds/README.md)，
完整Natural-QA631已入§6.1段落与附录H.9（表41）；正文表2是clean16K/32K。
首图复用151.9M可复现的交叉对照；主文保留9页。

已完成[五轮Astra/Sol独立审稿与改稿](research/pdf-review-rounds/20260915_astra_sol_five_rounds/README.md)；R05起包含四视角与AC综合，提示词和采纳判断均保留。
