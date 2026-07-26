# OLMo-2 Maturity Adaptation: Next Executable Experiment

Date: 2026-07-26

Status: `DESIGN_ONLY_NO_GPU_AUTHORIZATION`

Concern mapping: `R27bE.2`, `R27bE.5`, `AC.2`

## Required five-line experiment contract

1. **Reviewer/AC concern.** Demonstrate that a modern pretrained model can
   acquire useful long-context behavior from EVQ without reducing the result
   to PPL or destroying its existing short-context capabilities.
2. **Existing evidence.** Full-transplant EVQ plus 4K LoRA gives reproducible
   aggregate 8K `niah_single_1` strict exact of 69/100 and 67/100, but the
   scores fall to 21/50 and 19/50 on rows beyond every observed
   routing-training gap and to zero on two held-out 4K tasks. Native
   LongAlign-only Stage A also narrows those held-out tasks from 55%/25% to
   30%/8%.
3. **Smallest missing evidence.** A training path that begins exactly at the
   untouched Native model, reaches the final full-EVQ table, preserves
   baseline-positive 4K capabilities, and then improves an autoregressive 8K
   task not used as a retention target.
4. **Smallest executable plan.** One mature-Instruct EVQ arm with gradual
   Native-to-EVQ frequency morphing, Q/K/V/O rank-64 LoRA, natural plus
   instruction/behavior retention, and pre-registered intermediate gates.
5. **Stop condition.** Stop before long evaluation if the final full-EVQ model
   misses any 4K retention gate. Stop the route entirely if it retains 4K but
   fails on a frozen 8K set whose relevant-source gaps all exceed the maximum
   routing-training gap, or fails to improve a disjoint 8K capability task.

This file is a future experiment contract. It does not authorize training and
does not change the submitted EVQ method or the completed rebuttal evidence.

## Why the completed recipe cannot simply be extended

Three independent observations now constrain the next experiment:

1. **Abrupt coordinate change.** Full Native-to-EVQ replacement changes 63/64
   rotary pairs. The maximum 4K unwrapped phase displacement is 1005.43
   radians. A fixed position-independent Q/K map cannot universally undo this
   frequency change.
2. **Generic adaptation forgetting.** On the untouched Native Instruct model,
   UUID retrieval/variable-tracking recall is 55%/25%. After only the 20M-token
   LongAlign Stage A it is 30%/8%; the routing stage partly recovers it to
   40%/15%. Therefore retention is not an EVQ-only issue.
3. **Narrow behavior can be relearned.** EVQ reaches 69% and 67% at 8K on the
   trained numeric single-needle family, but only 42% and 38% on the half whose
   source gap exceeds training support, and remains zero on held-out UUID
   retrieval and variable tracking. More mixed-gap examples from the same
   family would increase fit, not establish distance-robust transfer.

Increasing rank, steps, or single-task data without addressing the first two
items is not an identified experiment.

## Proposed protocol: progressive frequency-morph LoRA

Let \(\omega^{N}\) be the frozen Native endpoint table and \(\omega^{E}\) the
frozen EVQ endpoint table. During adaptation step \(t\), use

\[
\omega_k(t)=(1-g_t)\omega_k^{N}+g_t\omega_k^{E},
\qquad
g_t=\frac{t}{T}.
\]

The model is exactly Native at \(g_0=0\) and exactly the submitted full-EVQ
frequency table at \(g_T=1\). Every intermediate table remains strictly
decreasing because it is a convex combination of two strictly decreasing
tables. Sequence length, position IDs, attention operator, head dimension,
and final inference implementation remain unchanged.

This is a continuation-training protocol, not an inference-time hybrid. It is
intended to remove the instantaneous 1005-radian transplant while LoRA tracks
the changing coordinate system. It does not defeat the exact-compensation
obstruction or guarantee function preservation at \(g=1\).

### Phase and token budget

For linear morphing, the worst 4K phase change per optimizer step is:

| Morph steps \(T\) | Full-token labels | Worst 4K phase increment |
| ---: | ---: | ---: |
| 611 | 20.0M | 1.646 rad |
| 1,000 | 32.8M | 1.005 rad |
| 1,500 | 49.2M | 0.670 rad |
| **2,000** | **65.5M** | **0.503 rad** |
| 4,000 | 131.1M | 0.251 rad |

The first executable candidate is \(T=2000\). It is the shortest listed
schedule below 0.51 rad per step at every 4K relative position. At the observed
29.6K tokens/s Stage-A throughput, 65.5M tokens are about 37 minutes before
retention overhead and evaluation.

## Training objective and data

The final model remains ordinary full-EVQ plus Q/K/V/O LoRA rank 64, alpha 128,
with a frozen LM head. The training mixture must address both modeling and
retention:

1. **Long-context modeling:** LongAlign 4K full-token next-token CE.
2. **Instruction retention:** official OLMo-2 Tulu assistant-only CE.
3. **Behavior retention:** sparse-token output matching to the untouched
   Native teacher on disjoint Tulu/LongAlign rows.

Teacher matching is a retention regularizer, not the mechanism that teaches
long retrieval. It should be evaluated on baseline-positive tasks and must not
be presented as a successful long-context distillation result.

A compute-bounded initial mixture is:

| Optimizer steps | Family |
| ---: | --- |
| 1,000 | LongAlign full-token CE |
| 500 | Tulu assistant-only CE |
| 500 | LongAlign/Tulu CE plus sparse-token Native-output retention |

Family order is deterministic and repeated as
`LongAlign, LongAlign, Tulu, retention`. Retention positions are sampled and
frozen before training. No RULER evaluation row is used for adaptation.

If sparse full-vocabulary KL is too expensive, the permitted approximation is
teacher top-\(k\) logit matching at the frozen token positions plus the
teacher residual mass. Plain hidden-state MSE is not a substitute.

## Intermediate gates

Save adapters and evaluate at \(g\in\{0.25,0.50,0.75,1.00\}\). These are
diagnostic checkpoints, not four separately tuned arms.

At each gate use the same frozen 4K rows:

| Metric | Untouched Native anchor | Continue threshold |
| --- | ---: | ---: |
| `niah_single_1` strict exact, n=20 | 20/20 | at least 18/20 |
| UUID distractor retrieval, n=20 | 55% | at least 45% |
| Variable-tracking mean reference recall, n=20 | 25% | at least 20% |
| Natural-text 4K mean NLL | frozen anchor | no worse than +0.10 |

An intermediate miss pauses morphing for at most 100 additional adaptation
steps at the same \(g\). If the gate remains below threshold, stop. Do not
silently weaken the gate or continue to \(g=1\).

After \(g=1\), repeat all three capability tasks with n=100 before any 8K
claim. The n=20 gates are early stopping instruments, not final statistics.

## Capability-conversion stage

Only a full-EVQ \(g=1\) adapter that passes every 4K retention gate may continue
for 300 counterfactual-routing steps. Retain the completed paired source/value
swap construction, but replace the old `routing, routing, natural` family
pattern with:

`routing, retention, routing, Tulu`.

The stage must preserve the same 4K gates. It then evaluates:

1. same-family `niah_single_1` at 8K on a frozen n=100 set with every
   source-to-generation gap above 3,933 tokens;
2. UUID multi-key retrieval at 8K, n=100;
3. variable tracking at 8K, n=100;
4. natural-text NLL at 4K/8K/16K.

The first item measures distance-robust replication of the existing positive;
the old mixed-gap aggregate is not a success gate. Items 2–3 are the actual
task-transfer test. If only item 1 improves, retain the result as task-specific
conversion and stop.

## Maturity-stage ordering

Do not run a full matrix.

| Model stage | Current evidence | Next action |
| --- | --- | --- |
| Released step 1K/2K | 4K single-needle floor is 0/100 | No capability adaptation; PPL-only stage |
| Released step 5K | Native 4K single needle 95%, multi-key 0%, VT 0.2% | Useful only for single-needle maturity diagnostics |
| Step 20K/30K Base | Weights present; step-30K EVQ Stage A has strong PPL; comparable task floor not frozen | Do not train until a Native 4K competency screen identifies a positive task |
| Mature Instruct | Strong Native 4K task anchors and completed positive/negative LoRA evidence | Run the one proposed arm first, if separately authorized |

Only after the mature Instruct arm passes both 4K retention and disjoint 8K
transfer should the **locked** recipe be applied to another maturity stage.
Do not tune a different morph duration, data mixture, or gate per checkpoint.

## Control and attribution order

The first candidate begins from the original Native checkpoint and ends at
full EVQ; it is the intervention arm. It does not prepay a second Native-LoRA
training run.

If and only if the intervention passes:

1. train Native with the same LoRA rank, data/order, token budget, and
   retention objective while keeping \(g=0\);
2. compare 4K retention and 8K transfer;
3. report EVQ's incremental effect, not the combined EVQ-plus-training gain.

The existing Native 20M+300-step arm is not the final control because it lacks
the new retention objective and has a smaller token budget.

## RTX 5090 execution contract

Keep the completed fastest shape:

- BF16 autocast;
- Flash-only SDPA;
- fused AdamW;
- micro-batch 4, gradient accumulation 2;
- no activation checkpointing;
- persistent TorchInductor cache;
- `max-autotune-no-cudagraphs`;
- 4K physical sequence and ordinary contiguous position IDs.

Updating a fixed-shape `inv_freq` buffer must not trigger recompilation. Before
paid execution, a five-step probe must demonstrate:

1. exactly one compilation;
2. changing `g_t` changes the realized frequency hash without graph breaks;
3. finite loss and gradients;
4. at least 90% of the completed Stage-A steady throughput.

The probe is an execution check, not an experiment arm.

## Implementation-readiness audit

The repository does **not** currently implement this protocol:

- `train_4k_stage_a.py` writes one fixed Native or EVQ table before training;
- `train_4k_stage_b.py` writes full EVQ once and schedules only its existing
  natural/binding families;
- neither trainer implements a step-indexed frequency morph, the locked
  LongAlign/Tulu/retention mixture, sparse Native-teacher targets, or the four
  intermediate retention gates.

Therefore no existing launcher is a valid command for this design, and no
existing READY receipt authorizes it. Before any paid run, a separate
implementation must provide:

1. a cloned Native endpoint and cloned EVQ endpoint, plus a step-indexed
   `inv_freq` update whose realized hash/value is independently checked;
2. a frozen data-family schedule and frozen sparse-retention positions;
3. teacher-target generation and loss accounting that cannot leak RULER rows;
4. resumable gate checkpoints at \(g=0.25,0.50,0.75,1.00\);
5. a no-GPU contract test and the five-step compile/throughput probe above;
6. a fresh READY receipt binding code, model, data, hashes, output path, and
   exact command.

Until all six items exist and pass, the protocol remains design-only.

## Decision outcomes

| Outcome | Interpretation | Action |
| --- | --- | --- |
| 4K retention fails before \(g=1\) | LoRA cannot track the morph under this budget/objective | Stop; do not add routing training |
| 4K retention passes at \(g=1\), 8K same-family improves only | Task-specific conversion | Preserve as supporting result; no general claim |
| 4K retention and disjoint 8K tasks improve | Candidate broad capability conversion | Trigger matched Native-LoRA control |
| Matched Native control ties or wins | Benefit comes from adaptation protocol, not EVQ | Report honestly; do not claim EVQ increment |
| EVQ wins after matched control | Reviewer-relevant mature-model evidence | Replicate one seed, then freeze raw/SHA package |

## Current authorization boundary

No command in this plan is authorized for GPU execution. The corrected hybrid
screens are also not authorized for rerun. The next paid action requires an
explicit user instruction after review of this contract.
