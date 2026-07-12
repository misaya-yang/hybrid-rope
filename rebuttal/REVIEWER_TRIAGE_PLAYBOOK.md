# Reviewer Triage Playbook

最后更新：2026-07-13

> **Fact gate:** 方法身份、理论与协议以 `FULL_PAPER_INTEGRITY_AUDIT_20260713.md` 为准；本文件旧的official-YaRN、DAPE、native-Geo、`d_eff=d_head`或LoRA matched-shape措辞均不得恢复。

- Preparation：`triage_ready`
- Response package：`needs_author_input`
- 当前模式：`triage-only`。实际 NeurIPS reviews 尚未收到。

用途：真实 reviews 到来后，只识别最多五个会改变评分的 concern，并把它们映射到证据、让步和作者决策。不要把模拟审稿当作真实触发器，也不要把 rebuttal 扩写成第二篇论文。

中心答复边界：EVQ-Cosh 研究 **training-time frequency allocation / finite spectral budget** 这一第三轴；它与 operator design、inference-time range scaling 互补，不是 universal SOTA 或任何 scaler / learned PE 的替代。

## Action labels

- `CORRECTION`：submitted wording、数学解释、指标或 provenance 需要纠正。
- `CONCESSION`：明确承认当前证据没有覆盖的范围。
- `EVIDENCE`：只引用已核验且与该问题直接相关的证据。
- `BOUNDARY`：同时给出 negative / reversal / seed / metric 边界。
- `AUTHOR_INPUT`：需要作者选择、真实 review 语境或 response budget 决策。
- `DEFER`：证据未达 reviewer-grade，或真实 review 未触发，不进入回复。

## 1. Trust / provenance

**Trigger**：真实 reviewer 质疑图表来源、版本不一致、复现路径、seed / aggregate 混用，或因一处错误怀疑全部结果。

**Answer kernel**：先承认被点名的具体错误或歧义；分开 submitted artifact、current source、raw-backed result 与 future revision；说明哪些 claim 保留、纠正或暂停。实验数字只沿 `docs/overview/RESULT_PROVENANCE_MANIFEST.md` 的最新链路回答。

**Use**：精确 erratum；artifact / seed / protocol；submitted-versus-current 时态；可匿名核验的 raw-backed provenance。

**Avoid**：说“reviewer misunderstood”；用当前源码假装提交件本来正确；用仓库根目录 2026-06-14 旧 snapshot 覆盖最新 manifest；把 paper source 当数学或结果 provenance 权威。

**Action / readiness**：`CORRECTION + EVIDENCE + BOUNDARY + AUTHOR_INPUT`；`PARTIAL`。事实链可准备，最终取舍必须等真实原话与作者确认。

## 2. KL / shape–scale correctness

**Trigger**：真实 reviewer 指出 ordinary KL 一阶为零、追问 `L^{-1/2}`、`τ` 最优性、cosh surrogate 与 deployed scale 是否拼接，或质疑 finite-`τ` 外推。

**Answer kernel**：直接承认 ordinary baseline-to-perturbed KL 的一阶变分为零，首项是 `O(τ^4)`，撤回旧 `O(τ²)` KL 解释。随后严格分三层：exact cosh surrogate theorem；conditional diffuse probability-transport proxy；empirically calibrated finite-`τ` basin selector。

**Use**：`FULL_PAPER_INTEGRITY_AUDIT_20260713.md`；长推导见`THEORY_REBUTTAL_MATHEMATICAL_AUDIT_20260711.md`；精确定理假设；区分utility通道数`M`与stiffness维度`d_S`。Phase16只能使用共同三seed weighted-PPL的7/9结果，不能作全局basin证明。

**Avoid**：让 submitted/current paper source 覆盖 7/11 数学审计；说 ordinary KL 导出非零 operating point；说 `τ=d_eff/√L` globally optimal；把 proxy 当 trained-task theorem。

**Action / readiness**：`CORRECTION + CONCESSION + EVIDENCE + BOUNDARY`；`THEORY_CORRECTION_PENDING`。无需为这一纠错临时启动 GPU 实验；只有实际 response 使用新三层口径后才能升级状态。

## 3. Baseline fairness / replication

**Trigger**：真实 reviewer 追问 tuned Geo / tuned YaRN、DAPE fidelity、single-seed、MLA convention、LoRA confound，或要求证明 EVQ 胜过最优 scaler。

**Answer kernel**：只守住已隔离的比较。Primary I 是 midpoint-Geo/EVQ 与repo fixed-ramp scaler的3-seed 2x2 factorial contrast，不是official/tuned YaRN；Primary II 的旧DAPE row实际是shared learnable inverse frequencies，retained headline为seed-42 diagnostic；Primary III actual head_dim=64、d_rope=32，tau=1.414为ad-hoc empirical setting且三seed batch不一致。旧LongAlign/LongAlpaca contrast仍不可作因果比较；fresh LongAlpaca pair只匹配training pipeline，native-Geo与midpoint-EVQ非同quantizer。

**Use**：每项的 exact seed / scale / token / metric protocol；matched-scale factorial；Primary III 3-seed结果；LoRA 数据清单与 manifest。

**Avoid**：把额外 EVQ seed 当完整 replication；把shared-frequency row称DAPE；把fixed-ramp写成official/tuned YaRN；用不同语料计算 causal delta；把native-Geo/midpoint-EVQ称纯shape control；把`d_eff=128`写成actual head dimension或theorem。

**Action / readiness**：`CONCESSION + EVIDENCE + BOUNDARY + DEFER`；总体 `PARTIAL`，LoRA attribution 为 `PROVENANCE_BLOCKED`。只有真实 reviewer 点名且 artifact gate 已满足时才升级证据。

## 4. Metric / capability

**Trigger**：真实 reviewer 问 PK 是否为生成准确率、结果是否代表真实 long-context capability、是否存在反向边界，或是否可作 downstream / production claim。

**Answer kernel**：PK 固定定义为 teacher-forced NLL-gap retrieval，AR exact 必须单列。按真实方法重标后，8K AR exact 为 Midpoint-Geo+fixed-ramp 0/0/0、EVQ+fixed-ramp 58/18/98（mean 58%）；同时披露4K的100%对77.3%反向边界。能力结论限定为已测协议，不升级为通用 downstream、production 或 latency/FLOP 结论。

**Use**：TF 与 AR 并列；trial / seed scope；4K reversal；Primary I–III 的原始任务定位。

**Avoid**：把 100% PK 称为 exact generation；隐藏 18–98% seed spread 或 4K reversal；用 QuALITY、LoRA、video 等 supporting row 代替 primary capability 证据。

**Action / readiness**：`CORRECTION + EVIDENCE + BOUNDARY`；`READY`。不得删除不利边界来换取更强标题。

## 5. AC-level remaining contribution

**Trigger**：AC 或 reviewer 问：在理论纠错、baseline 缺口与 reporting 修正后，论文还剩什么可接受贡献？

**Answer kernel**：剩余贡献是 training-time frequency allocation / finite spectral budget 这一第三设计轴，以及一个 closed-form、zero-learned-parameter 的 EVQ-Cosh 实例。证据核心是 Primary I对repo fixed-ramp的3-seed differential leverage、Primary II seed-42 midpoint-grid/shared-frequency diagnostic、Primary III heterogeneous-replication MLA stress test；理论核心只保留 exact surrogate、conditional proxy 与 empirical operating point的分层身份。

**Use**：一个窄机制主张；三层理论身份；Primary I–III 的真实证据层级；明确 errata 与未完成控制。

**Avoid**：universal SOTA；替代 YaRN/LongRoPE/DAPE/FIRE/learned PE；“工业级闭环”；用新 supporting experiment 重写论文身份。

**Action / readiness**：`CONCESSION + EVIDENCE + BOUNDARY + AUTHOR_INPUT`；`READY_WITH_CONCESSION`。最终篇幅和是否主动提某项修正由作者在看到真实 reviews 后决定。

## 真实 reviews 到来后的稳定 ID 工作流

1. local-only 冻结 reviewer / AC 原话；模拟审稿不得进入这一步。
2. 按出现顺序分配 `R1-C01`、`R1-C02`、`R2-C01` 或 `AC-C01`。ID 一经分配永久稳定，之后只改优先级和状态，不重编号。
3. 把每个 ID 映射到上面一个主 concern；跨类问题指定一个 primary concern，其余写入 `related`。
4. 每个 ID 使用同一模板：

```text
ID:
Verbatim trigger:
Primary concern / related:
Correction:
Concession:
Evidence:
Boundary:
Action labels:
Readiness:
Author decision required:
```

5. `correction / concession / evidence / boundary` 不适用时写 `none`，不得省略以制造已闭环假象。
6. 只选择影响评分的 3–5 个 ID 进入唯一 `AUTHOR_RESPONSE_20260722.md`；该文件只能在真实 reviews 到来后创建。

## Send gate

- [ ] 每段都绑定一个真实、逐字保存的 reviewer / AC trigger。
- [ ] 所有数字来自最新 provenance manifest 或其 raw-backed artifact。
- [ ] 理论答复明确撤回 ordinary-KL `O(τ²)` 解释，并保留三层身份。
- [ ] PK 与 AR exact 分开，且 8K seed spread 与 4K reversal 同时出现。
- [ ] LoRA 没有跨 LongAlign / LongAlpaca 作 causal attribution。
- [ ] 没有把模拟原文、deferred run 或历史 trace 写成真实 review / reviewer-grade evidence。
- [ ] 作者已处理所有 `AUTHOR_INPUT`，response package 才能从 `needs_author_input` 升级。
