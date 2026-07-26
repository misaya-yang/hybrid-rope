# Internal negative, diagnostic, and superseded-evidence ledger

Last audited: 2026-07-26
Audience: authors and future agents only
Reviewer-facing role: claim guardrails, not opening evidence

This file consolidates repeated summaries, negative experiments, theory
audits, and design-only routes that should not occupy the main playbook. Full
protocols and hashes remain in the standalone evidence owners; no owner is
deleted by this consolidation.

## 1. Status vocabulary

- `NEGATIVE_GUARDRAIL`: completed evidence that limits a positive claim.
- `INTERNAL_DIAGNOSTIC`: useful for interpretation, not a submitted theorem or
  reviewer-facing performance result.
- `PENDING_PROMOTION`: result direction exists but provenance or status is not
  ready for external use.
- `DESIGN_ONLY`: proposal or implementation without an admitted result.
- `SUPERSEDED`: an older plan or summary replaced by later evidence.
- `INDEX_ONLY`: duplicates other owners and creates no independent claim.
- `INVALID_IDENTITY`: the realized intervention does not match its label.

## 2. Negative and diagnostic registry

| ID | Status | What happened | Claim it forbids | Evidence owner |
| --- | --- | --- | --- | --- |
| `G-FMR-DEPLOY` | `NEGATIVE_GUARDRAIL` | Target-aware FMRoPE/official-YaRN-style range transport is stronger than raw EVQ in the tested small-model deployment; naive EVQ+FMR does not show stable additivity. | “EVQ replaces or consistently improves target-aware range scaling”; empirical orthogonality or additive synergy. | `EXPERIMENT_REPORT_20260724.md` §§1,5,6,12 |
| `G-COSH-NOT-UNIVERSAL` | `NEGATIVE_GUARDRAIL` | Matched exponential and attention-derived two-band schedules beat Cosh at some lengths. | “Cosh is empirically optimal over schedules” or “the surrogate proves the trained-model optimum.” | `EXPERIMENT_REPORT_20260724.md` §§3,7 |
| `G-TAU-FALLIBLE` | `NEGATIVE_GUARDRAIL` | The operating rule often lands in a useful basin, but Phase16 neighbor tests and a repaired Gram selector do not establish near-optimality. | “\(\tau=d/\sqrt L\) is a universal or exact optimum.” | `PHASE16_99RUN_RAW_REANALYSIS_20260724.md`; `TRAINING_FREE_TAU_SELECTOR_20260724.md` |
| `G-MLA-SCARCITY` | `NEGATIVE_GUARDRAIL` | The registered K=8 interaction was driven by collapse of a range control; raw EVQ remained worse than Native at 2×. The terminal decision is not to expand seeds. | A practical scarce-channel advantage or a passed primary MLA gate. | `EXPERIMENT_REPORT_20260724.md` §8 and `mla_scarcity_seed42_result_20260724.json` |
| `G-LLAMA-READOUT` | `NEGATIVE_GUARDRAIL` | Historical matched 8B LoRA improves long NLL, remote-source dependence, and rank, but dense generation remains zero and QA macro F1 is lower than Native. | Probability/routing equals solved generation, QA, or downstream superiority. | `EVQ_8B_ADAPTATION_EVIDENCE_20260724.md` |
| `G-LLAMA-32K` | `NEGATIVE_GUARDRAIL` | After matched 13-family adaptation, EVQ and untouched Native are zero at 32K. Native-LoRA completed only 10/13 cells and those ten are zero. | Usable 32K RULER capability or a complete Native-LoRA 32K macro. | `LLAMA8B_MATCHED_RULER_MIX_20260726.md` |
| `G-OLMO-GAP` | `NEGATIVE_GUARDRAIL` | OLMo counterfactual NIAH accuracy falls sharply when source-to-generation distance exceeds all training gaps. | Uniformly solved 2× retrieval. | `OLMO2_N100_GAP_STRUCTURE_AUDIT_20260726.md` |
| `G-OLMO-HELDOUT-TASK` | `NEGATIVE_GUARDRAIL` | Full EVQ injection and the NIAH routing adapter score zero on held-out UUID distractor retrieval and variable tracking in the frozen 4K screen. | Unseen-task transfer, no catastrophic forgetting, or general downstream conversion from the NIAH arm. | `OLMO2_1B_4K_ONLY_ROUTING_CONVERSION_20260726.md` |
| `G-OLMO-CLEAN-RULER` | `NEGATIVE_GUARDRAIL` + metadata hold | A clean LongAlign+Tulu 4K arm reports `9.74%/4.01%/2.05%` full-RULER macro and 0/39 cells above length-matched controls. | Natural/instruction SFT alone establishes broad RULER transfer. | `OLMO2_1B_CLEAN_4K_TULU_FULL_RULER_20260728.md` |
| `G-OLMO-NON-RULER` | `NEGATIVE_GUARDRAIL` + metadata hold | Seven non-RULER 4K objectives, including progressive morph, KL, answer-only, and natural-span retrieval, fail their screening gate. | Re-proposing those routes as untested high-confidence fixes. | `OLMO2_1B_NON_RULER_ADAPTATION_SEARCH_20260731.md` |
| `D-TRANSPLANT` | `INTERNAL_DIAGNOSTIC` | A static position-independent LoRA cannot in general exactly conjugate one RoPE frequency generator into another at all distances. | Exact Native→EVQ frequency transplantation by ordinary static LoRA. | `OLMO2_POSTHOC_FREQUENCY_TRANSPLANT_OBSTRUCTION_20260726.md` |
| `D-BAND-PROXY` | `INTERNAL_DIAGNOSTIC` | Static norm/phase proxies do not reliably predict per-pair causal importance; selected interference deletion can still improve held-out NLL. | A universal “high-norm band is the useful band” mechanism claim. | `EXPERIMENT_REPORT_20260724.md` §§10–11 |
| `I-HYBRID-ALIAS` | `INVALID_IDENTITY` | Historical hybrid frequency tensors aliased the Native reference before an in-place EVQ patch; labels did not match realized hashes. | Any success/failure conclusion from the affected historical hybrid runs. | `OLMO2_POSTHOC_FREQUENCY_TRANSPLANT_OBSTRUCTION_20260726.md` |
| `D-RESIDUAL/HYBRID/DC` | `DESIGN_ONLY` | Native-protected tail, DC-pair, and residual dual-branch routes have geometry or prototype code but no admitted reviewer-grade result. | Presenting EVQ-v2 ideas as evidence for submitted EVQ. | `EVQ_4X_AND_8K_NO_HARM_THEORY_20260726.md` and residual prototypes |

The `20260728` and `20260731` owners are dated after this audit date. Their
content may remain useful internally, but they are not reviewer-facing until
the real run/transfer timestamps, raw packages, and metadata are reconciled.

## 3. Superseded and duplicate documents

| Document | Classification | Current treatment |
| --- | --- | --- |
| `OLMO2_1B_OVERNIGHT_EXPERIMENT_SUMMARY_20260726.md` | `INDEX_ONLY` | Superseded as the global routing entry by this ledger and the positive-evidence ledger. Retain only as a historical OLMo-specific summary; do not cite it as an independent evidence owner. |
| `EVQ_Cosh_NeurIPS2026_Rebuttal_Experiment_Design.md` | `SUPERSEDED` wide plan | Not an execution queue. It refers to unpromoted reviewer identities and a broader campaign than the retained concerns require. |
| `ROPE_RANGE_SHAPE_MAPPING_THEORY_AND_5090_PLAN_20260724.md` | `SUPERSEDED` plan | The exact-range experiment it proposed has completed at the author-confirmed level. Use the standalone exact-range owner instead. |
| `OLMO2_MATURITY_ADAPTATION_NEXT_EXPERIMENT_20260726.md` | `SUPERSEDED` plan | Progressive morph was later tested and failed the screen; do not reopen it from this document. |
| `EVQ_4X_AND_8K_NO_HARM_THEORY_20260726.md` | `INTERNAL_DIAGNOSTIC / ARCHIVED_DESIGN` | Retain only no-harm/design reasoning. Its empirical status now points to the completed LLaMA matched owner; its later gates are not an action queue. |
| `TAU_TRUE_ROLE_AND_OPERATING_RULE_AUDIT.md` | `INTERNAL_DIAGNOSTIC` | Retain derivations and audit history; use Phase16 and the compact positive/negative ledgers for response routing. |
| `MLA_YARN_OPERATOR_PARITY_5090_PLAN.md` | `DESIGN_ONLY` deferred | Not active rebuttal evidence or GPU authorization. |
| `MATCHED_RANGE_COSH_500M_S42_20260724.md` | retained provenance owner | Keep until the three-seed raw aggregate is promoted. It is not the headline aggregate. |
| `OLMO2_N100_GAP_STRUCTURE_AUDIT_20260726.md` | retained guardrail owner | Keep while the main conversion report and playbook rely on its distance decomposition. |

## 4. Status clarifications and remaining gates

### 4.1 OLMo step-1,000 scratch source hierarchy — resolved

The sibling JSON is explicitly scoped to
`released_native_rope_baselines_only`; its statement that no EVQ result is
included describes that native-only snapshot. The later standalone Markdown
owner records the completed Geo/EVQ comparison and retains hashes for the Geo
raw result, EVQ checkpoint, EVQ raw result, per-token NLL, paired comparison,
and evaluation anchors.

**Resolution:** classify the paired result as
`POST_SUBMISSION_RAW_HASH_BACKED`. It is reviewer-usable as a
single-trajectory, same-initialization/same-scientific-recipe natural-text LM
comparison. The different HF versus AI2 trainer implementations remain an
adjacent claim boundary; they are not a status or promotion blocker. Do not
again interpret the native-only JSON as a contradiction or require it to
duplicate the later paired-result fields.

### 4.2 Exact-range three-seed status

The aggregate values agree across the report, JSON, and prior playbook, but the
owner explicitly records that local raw/per-seed values and confidence
intervals are absent.

**Resolution:** keep the numbers as `AUTHOR_CONFIRMED` internal drafting
material. Do not call them raw-backed or statistically significant, and do not
send them until promotion.

### 4.3 OLMo NIAH `69/67` versus `49/48`

These are different evaluations:

- `69/67`: the original 8K n=100 set, which mixes within- and
  beyond-training-gap rows;
- `49/48`: a fresh set in which every gap exceeds training support.

**Resolution:** use `69/67` only for task-family-matched 8K capability; use
`49/48` only when explicitly claiming beyond-training-gap support.

### 4.4 LLaMA temporal results

Two temporal tables use different adapters and controls:

- 300-step matched LongAlpaca Native-LoRA versus EVQ-LoRA;
- 516-step RULER-family EVQ-LoRA versus untouched Native.

**Resolution:** use the first for a matched LoRA NLL contrast. Do not combine
its control with the second table or call the second matched Native-LoRA
evidence.

### 4.5 “Counterfactual” wording

The OLMo 300-step routing stage and a later fresh EVQ-only LLaMA arm use
pairwise counterfactual loss. The OLMo 13-task continuation and the matched
LLaMA natural-LM/RULER studies do not.

**Resolution:** name the exact arm. Do not describe either matched LLaMA
Native/EVQ protocol as counterfactual-trained, and do not use the fresh
EVQ-only arm as a matched causal comparison.

### 4.6 Submitted exact-kernel corroboration

The playbook historically used Appendices A.6/A.14 as strong submitted
corroboration, while an internal ultra-audit raises unresolved implementation
identity questions.

**Resolution:** retain A.6/A.14 as submitted-paper claims, but do not use them
as the first or sole rebuttal proof until the audit discrepancy is resolved.
The reviewer-facing response should lead with controlled trained-model
evidence whose provenance is independently clear.

## 5. Triggered disclosure rules

- If the response says “Cosh is optimal,” disclose `G-COSH-NOT-UNIVERSAL` and
  rewrite the claim.
- If it implies EVQ replaces range scaling, disclose `G-FMR-DEPLOY`.
- If it equates NLL with capability, disclose `G-LLAMA-READOUT` and the clean
  OLMo RULER boundary.
- If it says unseen-task, zero-shot, or no forgetting, disclose
  `G-OLMO-HELDOUT-TASK` and the shared-generator-family protocol.
- If it says 32K capability, disclose `G-LLAMA-32K`.
- If it treats the operating rule as exact, disclose `G-TAU-FALLIBLE`.
- If it proposes progressive morph or historical hybrid results, disclose
  `G-OLMO-NON-RULER` or `I-HYBRID-ALIAS`.

These limitations must be adjacent when their omission would make a positive
claim misleading. They do not need to occupy the opening response when the
corresponding broad claim is not made.

## 6. Open internal work, not current rebuttal evidence

1. Promote the exact-range three-seed raw aggregate.
2. Correct future-dated OLMo metadata.
3. Retain the proposed matched LLaMA Native/EVQ counterfactual pair as
   `DESIGN_ONLY`. It is not current evidence, is not required by the sendable
   rebuttal core, and must not be run during the current response cycle.
4. Complete the three missing Native-LoRA 32K RULER cells only if a full 32K
   control is necessary; the current ten completed cells and EVQ result are
   already zero.
5. Keep EVQ-v2 residual/hybrid/DC research outside the submitted-method
   rebuttal.
