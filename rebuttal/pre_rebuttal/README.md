# Pre-rebuttal archive and reuse index

最后审计：2026-07-24

本目录收纳 2026-07-23 正式 reviews 到来前形成的材料。它们没有被删除，因为
其中包含重要的数学纠错、方法身份审计、负结果和可复用代码；但它们不再决定
当前问题优先级，也不能直接复制成 author response。

## 仍然有直接复用价值

| 材料 | 可复用内容 | 使用边界 |
| --- | --- | --- |
| `FULL_PAPER_INTEGRITY_AUDIT_20260713.md` | fixed-ramp、learnable shared `inv_freq`、midpoint Geo、ordinary-KL、`c_coll`、Phase16 和 provenance 的事实审计 | 当前 reviewer 触发相关论点时作为事实底稿；不能替代原始 review |
| `FIRST_PRINCIPLES_REBUTTAL_REASSESSMENT_20260716.md` | PDF-only 可发现性、主动/被动披露、surviving claim 和最小回应策略 | 披露范围仍需作者决定 |
| `THEORY_FREQUENCY_OPTIMALITY_AND_TAU_20260716.md` | geometric/cosh 最优性的条件划分、\(\tau\) 的严格/条件/经验三层边界 | 对 `R27bE.1/.4` 很有用；不能声称解决 trained-task optimum |
| `THEORY_REBUTTAL_MATHEMATICAL_AUDIT_20260711.md` | ordinary-KL 与 transport proxy 的长推导 | 细节参考；若与 7/13、7/16 审计冲突，以后者为准 |
| `rebuttal_playbook.md` | 回答结构、实验 gate、禁止措辞和 integrity disclosure 候选 | 它是 pre-review 快照，不再是“唯一操作入口” |
| `REVIEWER_TRIAGE_PLAYBOOK.md` | stable-ID 和 evidence/boundary/action 字段模板 | 当前 concern IDs 以正式 review 文件为准 |

## 只在 reviewer 明确触发时复用

| 材料 | 价值 | 为什么不主动部署 |
| --- | --- | --- |
| `EVQ_YARN_COMPONENT_ABLATION_20260714.md` | official-YaRN 组件拆分的单 seed 机制证据 | 不恢复 submitted YaRN 身份，不支持普遍互补性 |
| `NATIVE_ROPE_EVQ_150M_500M_RESULT_20260713.md` | native endpoint、EVQ 和多种 scaler 的小模型机制诊断 | 151.9M、单 seed、含 passkey-mix，不是规模或下游闭环 |
| `LORA_GEO_CONTROL_RESULT_AUDIT_20260711.md` | 揭示 LongAlign/LongAlpaca 跨协议比较无因果性 | 不能用旧 contrast 证明 EVQ |
| `LORA_LONGALPACA_TEMPORAL_NLL_20260712.md` | matched-training-pipeline 的 temporal NLL 信号 | native-Geo 与 midpoint-EVQ 非同 quantizer；只证明 NLL，不证明能力 |
| `EVQ_8K_ONLY_CAPABILITY_TRANSFER_PLAN_20260713.md` | phase observability、representation/routing/readout 的机制假设 | 后续 registered QA 已给出负结果；不得继续作为默认最高优先级 |
| `REBUTTAL_VIABILITY_AND_VENUE_PLAN_20260713.md` | 诚信披露和 venue 决策背景 | 不是 reviewer-facing 文本，也不是当前政策的自动替代品 |
| `evq_seed42_retrieval_repair/` | evaluator、mask、provenance 和 fail-closed 组件 | 旧 8B repair 路线，不是当前获批实验 |
| `frequency_adaptation_8b/` | 连续迁移、E16 训练与诊断组件 | 过度混合训练体制和频率效应，当前不启动 |
| `real_dape_compare/` | 追踪真实 DAPE 对照所需的协议和成本边界 | submitted row 并非 DAPE；此包没有自动修复旧比较 |

8B 能力的后续真实边界还应联合查阅：

- `docs/exp/2026-07-14_lora_retrieval_conversion_probe.md`
- `docs/exp/2026-07-15_lora_qa16k_three_arm_results.md`

后者的 registered QA negative 优先于本目录早期“可能转化为能力”的计划。

## 仅作档案或检索

| 材料 | 状态 |
| --- | --- |
| `REBUTTAL_MASTER_QUESTION_LEDGER_20260711.md` | 宽攻击面索引；内容过多且部分已过时，不得恢复为当前 action board |
| `simulated_reviews/` | 内部压力测试，不是真实 reviewer 意见 |
| `raw_sources/` | 历史本地输入控制；`*_verbatim.md` 保持 ignored，不提交、不打包 |

## 已失效的旧结论

- pre-review 文件中“尚未收到真实 reviews”的状态已经过时。
- 旧 playbook 的“唯一入口”身份已经失效；当前入口是
  `../rebuttal_0723/README.md`。
- 8B LoRA 的长位置 PPL/NLL 改善没有转化成 registered QA 提升，不能作为
  downstream defense。
- 不能把 fixed-ramp 称为 official YaRN，把 shared learnable `inv_freq`
  称为 DAPE，把 midpoint Geo 称为 native RoPE。
- ordinary baseline-to-perturbed KL 从 \(O(\tau^4)\) 起；
  \(\tau=d_{\mathrm{eff}}/\sqrt L\) 只是有条件结构动机加经验 operating rule。
- `c_coll=1.171` 和旧 Phase16 叙述不能证明端到端理论闭环。

任何复用都应先回答两个问题：它是否直接对应当前 reviewer concern？它的
method identity、seed、metric 和 raw provenance 是否足以支撑准备写下的那一句？
