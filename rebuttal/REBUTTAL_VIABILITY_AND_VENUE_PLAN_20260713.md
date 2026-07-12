# Rebuttal Viability and Venue Plan

日期：2026-07-13

状态：`internal_decision_note`

依赖：`FULL_PAPER_INTEGRITY_AUDIT_20260713.md`

---

## 0. 诚实结论

在 DAPE、YaRN、midpoint-Geo、ordinary-KL 与 collision calibration 同时出现问题后，NeurIPS 2026 接收并非不可能，但已经从“补几个实验即可”变成了**显著困难的 trust-repair case**。

不能给一个负责任的数值接收概率。决定性因素不是错误数量本身，而是 reviews 是否把接收建立在以下已失效部分上：

- faithful DAPE comparison；
- official YaRN complementarity；
- standard/native RoPE dominance；
- \(\tau=d/\sqrt L\) 的 ordinary-KL theorem；
- exact-kernel prefactor calibration。

若 reviewers 主要认可 finite-spectral-budget idea、closed-form surrogate theorem 和 matched empirical signal，诚实收缩后仍可能保住一篇 mechanism paper；若他们要求官方 comparator fidelity 或完整理论闭环，这些缺口无法在短 rebuttal 中真正修复，拒稿风险会很高。

---

## 1. NeurIPS 2026 政策边界

官方 [Main Track Handbook](https://neurips.cc/Conferences/2026/MainTrackHandbook) 明确规定：

- rebuttal 期间不能上传修订论文或补充材料；
- 可以在文本回复中报告新结果；
- 原始投稿仍是 reviewer/AC 的 acceptance basis；
- rebuttal 用于澄清问题，不是第二次提交；
- 每条 review 回复最多 10,000 characters；
- 回复中不能放链接，除非 reviewer 请求代码且匿名链接私发 AC；
- 作者对论文、引用和结果正确性承担最终责任。

因此，正确动作不是假装一份 future corrected PDF 已替代提交稿，也不是自动撤稿。正确顺序是：

1. 等 7 月 22 日真实 reviews；
2. 映射 reviewer 是否触发 baseline identity、theory、native control 与 provenance；
3. 对重大身份错误做短而完整的 factual correction；
4. 明确哪些 numerical values unchanged、哪些 interpretation withdrawn；
5. 只用已经完成且 provenance 闭环的新结果；
6. 如错误未被任何 reviewer点名但会使 accepted record保留实质假陈述，建议向 AC 做一条统一 integrity disclosure。

Handbook没有明确规定作者自行发现 comparator-identity error时的专门报告流程；第6项是本文基于研究诚信作出的建议与政策推论，不是NeurIPS明文命令。

---

## 2. 四种最可能情形

| 情形 | 诚实判断 | Rebuttal 动作 |
| --- | --- | --- |
| Reviews整体正面，未点名方法身份 | 仍有机会，但不能因此隐瞒已知错误 | 向 AC 做一条 concise integrity disclosure；强调 unchanged numerics 与 narrower survivor claim |
| 点名 DAPE/YaRN fidelity 或标准 Geo | 风险高，但最可回答 | 主动 relabel；如已完成 faithful YaRN evaluation，报告；不承诺临时 DAPE 结果 |
| 点名 \(\tau\) optimality / KL / exact-kernel | 风险很高 | 承认 ordinary-KL 与 `c_coll` 错误，只保留 exact cosh surrogate theorem与 proxy-motivated empirical rule |
| Reviews已因规模、真实任务、undertraining或复现性偏负 | 新错误会进一步压低评价 | 不堆砌 supporting结果；用最窄机制贡献回答，接受可能需完整修订后转投 |

### 2.1 什么会真正提高 rebuttal 信息量

1. 先用已有 Primary I checkpoints做 official-formula-anchored zero-shot diagnostic；它低成本但不能称 faithful YaRN method comparison；
2. 若 reviewer要求方法级结论，再做 native endpoint Geo + official YaRN continuation，以及 endpoint EVQ + 明示的 YaRN-derived generalization matched training；native baseline不能靠同 checkpoint换频率得到；
3. 一张完全准确的 paper-row-to-code-to-artifact identity table；
4. 把 `DAPE`、`ordinary KL`、`c_coll` 明确撤回，而不是用大量新实验转移问题。

### 2.2 什么不会修复核心问题

- 再跑 shared-frequency row 的更多 seeds；它仍不是 DAPE；
- 把 repo-local Kerple/MLP 变体叫 DAPE；
- 用 Phase16 composite rank 证明理论；
- 用 LoRA/video/750M supporting evidence掩盖 core identity；
- 声称 implementation差异只是工程细节。YaRN的 correction range与 `mscale` 正是算法定义的一部分。

---

## 3. 公开案例能说明什么

没有找到与“baseline 实际不是所声称方法，同时存在理论阶数错误”同等严重且可作为安全先例的公开案例。以下只能说明“主动承认错误不必然自动拒稿”，不能类推本稿一定安全：

| 案例 | 公开更正 | 结果 | 与本稿差异 |
| --- | --- | --- | --- |
| [Context Clues: EHR Long Context](https://openreview.net/forum?id=zg3ec1TdAP) | rebuttal 承认 Table 2 processing bug 和 pooling 描述错误 | ICLR 2025 Poster | 作者称核心结论未变；比本稿 comparator identity 问题轻 |
| [Needle In A Video Haystack](https://openreview.net/forum?id=ZJo6Radbqq) | Figure 8 axis 0--9 实应 0%--90% | ICLR 2025 Poster | 纯图标签错误，远轻于本稿 |
| [LLM4EHR](https://openreview.net/forum?id=pym3JRajmW) | 承认 baseline label错误 | ICLR 2026 Reject | 拒稿还有 novelty/evidence等原因，不能证明由误标单独导致 |

结论：诚实披露是必要条件，不是接收保证。当前不能引用任何公开案例来声称“这种程度的错误通常仍会接收”。

公开 OpenReview案例还存在 publication/opt-in selection bias，不能用这三个样本估计错误论文的接收频率。

---

## 4. 后续 venue 选择

所有日期与状态截至 2026-07-13。NeurIPS 在审期间不得将同一工作并行提交到另一 archival venue。

| Venue | 当前官方状态 | 适配度 | 投稿前最低修复 |
| --- | --- | --- | --- |
| TMLR | [FAQ](https://jmlr.org/tmlr/faq.html)确认 rolling；[Acceptance Criteria](https://jmlr.org/tmlr/acceptance-criteria.html)强调证据支持claim并允许收缩claim；dual规则见[Editorial Policies](https://jmlr.org/tmlr/editorial-policies.html) | **与 correctness-first criteria适配度最高；不代表更高接收概率** | 完成全部 identity修正；native/official baselines；删除错误理论；完整 provenance；不能与 NeurIPS 并行 |
| COLM 2027 | 2027 CFP 尚未公布；[COLM 2026 CFP](https://colmweb.org/cfp.html) cadence只能作历史参考 | **会议中主题最匹配** | faithful YaRN/native Geo；现代 LM 长上下文能力；严格区分 allocation 与 range scaling |
| [ICLR 2027](https://iclr.cc/Conferences/FutureMeetings) | 仅公布 West Coast North America，deadline尚未官方公布 | **高适配、高竞争** | 同上，并补更强 scale/model-family/seed evidence与清晰 mechanistic analysis |
| [ICML 2027](https://icml.cc/Conferences/FutureMeetings) | 官方称 2027 announcement coming in August，deadline未公布 | **高门槛** | 更大模型/数据规模、更广强 baseline、完整统计与复现链 |
| [ACL 2027](https://2027.aclweb.org/) | 2027-08-17--22，日本；ARR/commit deadlines TBA | **有条件适配** | 必须加入真实长上下文 NLP tasks，不可只依赖 PPL/合成 retrieval |
| EMNLP 2027 | 截至当前未官宣 CFP/deadline；[ARR dates](https://aclrollingreview.org/dates)目前仅列到EMNLP 2026 | **有条件适配** | 与ACL相同，需真实NLP downstream与ARR-compliant revision trail |
| [COLING 2027](https://www.aclweb.org/portal/content/32nd-international-conference-computational-linguistics-coling-2027) | ARR deadline 2026-10-12；澳门，2027-05-09--14 | **有条件适配** | 与 ACL 类似，强化语言任务与可解释误差分析 |
| AISTATS 2027 | 官方 CFP尚未公布 | **理论加强后才合适** | 把理论缩成完全正确的 surrogate theorem与可验证统计/优化结论 |
| [JMLR](https://jmlr.org/author-info.html) | rolling | **仅适合大幅扩展版** | 新理论、跨架构规模验证、系统 baseline和完整 reproducibility均需实质增量 |

### 4.1 推荐顺序

1. 先完成 NeurIPS rebuttal 的诚实回应，不做并行 archival 投稿；
2. 若 NeurIPS 未接收：优先 TMLR，适合 correctness-first 的完整修订；
3. 若希望继续冲会议：COLM 2027主题最匹配，ICLR 2027次之；
4. ICML 只在规模与 baseline 大幅补强后考虑；
5. ACL/COLING 只有加入真正的 NLP downstream evaluation 才值得投。

### 4.2 Dual submission 与重投边界

- NeurIPS 2026在整个评审期禁止同一工作并行投另一 archival venue；
- [ARR CFP](https://aclrollingreview.org/cfp)禁止并行评审，ARR内重投需关联上一版本并逐条说明修改；
- ICLR、ICML、AISTATS与COLM 2027规则尚未公布，任何dual-policy判断只能引用各venue最新已发布版本，不能把2026规则写成2027定案；
- [JMLR author policy](https://jmlr.org/author-info.html)禁止同时投稿；conference扩展版需要 substantive delta，拒稿后重投还受AE许可约束。

---

## 5. 完整修订的最低证据包

### P0：不做就不能再次投稿

1. 全文 `DAPE` row 改为真实 shared-frequency identity，或完成官方 DAPE faithful comparison；当前作者决定是不为 rebuttal补 DAPE，因此先 relabel。
2. 全文区分 official YaRN、repo fixed-ramp scaler 与 MLA wavelength-threshold scaler。
3. 增加 native endpoint Geo，并明确 midpoint quantization。
4. 删除 ordinary-KL proposition、`c_coll` calibration、LoRA rank/channel theorem与错误 waterbed inference。
5. Phase16按 99-run/9-config/selected-confirmation真实设计重写。
6. Primary II改成 151.9M、LR `3e-4`、effective batch 64、真实 seed scopes。
7. Primary III披露 batch mismatch与 empirical \(\tau=1.414\)。

### P1：决定下一稿竞争力

1. Native endpoint Geo + official YaRN matched continuation，以及endpoint EVQ +明确标注的 YaRN-derived generalization；已有checkpoint的zero-shot算子诊断单独报告；
2. 至少一个现代、具备任务能力的 LM setting；
3. 真实长上下文 QA/retrieval任务，报告 generation-level accuracy与 NLL；
4. 主实验全 seed raw metrics、confidence intervals与 fixed evaluator manifests；
5. 删除或重做 750M/video/旧 LoRA 等 provenance不闭环的 supporting rows；
6. paper row -> artifact -> runner -> implementation -> official source 的机器可查 manifest。

### P2：有余力再做

- tuned base / scaler grid；
- 多模型族、多 head dimension；
- 更大 token budget与训练曲线；
- attention-distance / phase-collision diagnostic与 task error的相关性，而非把 proxy当因果 theorem。

---

## 6. 7 月 22 日执行原则

1. 先读真实 reviews，不预制一份覆盖所有错误的超长公开回复；
2. 最多选择 3--5 个真正影响 score 的问题；
3. 每个回答先给 correction，再给 surviving evidence；
4. 不使用链接，不上传修订 PDF，不声称 future revision已经修复提交稿；
5. 不用未完成实验、隐藏负结果或不忠实 comparator；
6. 若 reviewer 未触发但错误会污染 accepted record，向 AC 做一条合并的 identity/theory disclosure；
7. Rebuttal 目标是让 reviewer准确评价仍然存在的贡献，而不是证明提交稿没有错误。
