# OLMo-2 1.485B EVQ attention-function restoration plan

Status: `PREPARED_NO_GPU_NOT_EXECUTED`

Concern targets: `R27bE.2`, `R27bE.5`, `AC.2`

This document registers one bounded feasibility experiment. It is not a
completed result, rebuttal evidence, or authorization to improvise additional
training arms.

## 1. Decision to be made

The experiment asks one question:

> With the mature OLMo-2 1.485B backbone frozen and the exact EVQ frequency
> grid active from step 0, can Q/K/V LoRA recover the Native model's 4K
> attention function closely enough to preserve broad 4K capability?

This is the missing prerequisite for using fixed EVQ in a mature pretrained
model without accepting the observed short-context capability loss. It does
not test from-scratch EVQ training, universal EVQ compatibility, or long-task
transfer by itself.

The registered route has three possible outcomes:

1. **4K restoration passes.** Freeze the adapter and evaluate 8K, then 16K.
2. **Optimization succeeds but capability restoration fails.** Stop this
   route; do not infer restoration from the internal loss or NLL.
3. **The GPU/runtime contract fails.** Stop before training; do not change the
   kernel, package versions, rank, loss weights, or batch contract ad hoc.

## 2. Why another CE or frequency-morph run is not justified

The completed non-RULER search already tested seven 4K-only adaptation arms.
The most relevant negatives are:

| Arm | Intervention | Natural NLL 4K / 8K / 16K | RULER screen 4K / 8K |
| --- | --- | --- | --- |
| A1 | Linear Native-to-EVQ frequency morph with token loss | 3.0565 / 3.2559 / 3.4810 | 0.1167 / 0.0667 |
| A2 | A1 plus sparse Native output-logit KL | 3.1331 / 3.3324 / 3.5410 | 0.0833 / 0.0417 |

On the identical RULER screen, untouched Native at 4K scored `0.5700`, while
untouched Native with the official Transformers YaRN factor-2 control at 8K
scored `0.5125`. The other five arms also failed the registered screening gate.
The canonical evidence owner is
`theory_results/OLMO2_1B_NON_RULER_ADAPTATION_SEARCH_20260731.md`.

These results reject the idea that low token loss, gradual frequency morphing,
or sparse output-level retention is enough to recover the inherited routing
function. The new route therefore changes the supervised object, not merely
the optimizer or injection scope:

- no Native-to-EVQ frequency interpolation;
- no token cross-entropy;
- no output-logit KL;
- no RULER, NIAH, or 2Wiki training rows;
- direct matching of the Native attention map and its per-head value context.

## 3. Identifiability of the registered objective

At the captured layer, let the causal attention function be

\[
S = QK^\top/\sqrt{d_h} + M,\qquad
A = \operatorname{softmax}(S),\qquad
C = AV,
\]

where \(M\) is the causal mask. The Native teacher uses Native RoPE and the
student uses the exact EVQ frequency tensor.

Matching only the Q/Q, K/K and V/V self-relations is insufficient. For
orthogonal matrices \(U\) and \(W\),

\[
Q_s=Q_tU,\qquad K_s=K_tW
\]

preserves \(Q_sQ_s^\top=Q_tQ_t^\top\) and
\(K_sK_s^\top=K_tK_t^\top\), yet generally changes
\(Q_sK_s^\top=Q_tUW^\top K_t^\top\) when \(U\ne W\). Likewise, a V/V relation
does not uniquely determine \(AV\). A self-relation-only objective therefore
admits solutions that preserve its loss while changing the actual attention
function.

The registered loss instead is

\[
\begin{aligned}
\mathcal L
={}&
\operatorname{KL}(A_N\|A_E)
+ \frac{\lVert C_E-C_N\rVert_2^2}
       {\operatorname{mean}(C_N^2)+10^{-6}}\\
&+0.25\,
\frac{
  \operatorname{KL}(R^Q_N\|R^Q_E)
 +\operatorname{KL}(R^K_N\|R^K_E)
 +\operatorname{KL}(R^V_N\|R^V_E)}
 {3},
\end{aligned}
\]

where each \(R\) is the corresponding causal softmax relation. Each KL is the
teacher-to-student forward KL, normalized over batch, query heads, and valid
query rows.

The first term directly identifies the causal QK routing distribution. The
second constrains the result of transporting value content through that
routing distribution. The self-relations are retained only as a lower-weight
geometric regularizer.

This construction is inspired by the exact linear-memory relation loss in
LinearARD (arXiv:2604.00004), pinned to upstream commit
`23866f68a8b65da796c75439a06d4cb996bcb7bb`. It is not a LinearARD replication:
the decisive QK attention and `A@V` context targets are specific to this
restoration hypothesis.

## 4. Frozen scientific contract

| Field | Registered value |
| --- | --- |
| Model | OLMo-2 1.485B Instruct |
| Teacher | Frozen Native RoPE, evaluation mode |
| Student backbone | Same frozen checkpoint |
| Student frequencies | Exact full EVQ-Cosh endpoint from step 0 |
| Trainable parameters | Q/K/V LoRA only, all 16 layers |
| LoRA | rank 512, alpha 1024, dropout 0 |
| Captured target | layer 15 post-RoPE Q/K, V, and per-head attention context |
| Training data | Existing hash-bound LongAlign 4K fixed view |
| Position IDs | Contiguous `0..4095` only |
| Sequence length | Exactly 4096 |
| Objective | Section 3 only; no CE or output-logit KL |
| Optimizer | AdamW |
| Peak LR | `2e-5` |
| Schedule | 4 warmup steps, minimum LR ratio 0.9 |
| Gradient clipping | maximum norm 5.0 |
| Budget | 144 optimizer steps |
| Batch | micro-batch 1, accumulation 4 |
| Student exposure | 2,359,296 tokens |
| Teacher exposure | 2,359,296 tokens |
| Compile | disabled |
| Gradient checkpointing | disabled |
| Seed | 20,260,803 |

The rank is fixed before execution at the same 25% hidden-dimension proportion
as the published LinearARD LLaMA recipe (`1024/4096`), scaled to OLMo's 2048
hidden dimension. There is no rank, alpha, target-layer, loss-weight, or LR
sweep.

## 5. Stage G — mandatory discarded GPU smoke

The existing no-GPU prepared receipt establishes only asset and protocol
readiness. When a compatible GPU is attached, run the registered smoke before
the 144-step job.

The smoke must prove:

1. GPU identity, compute capability, PyTorch/CUDA versions, BF16 support, and
   available memory are recorded.
2. The pinned upstream linear-memory KL kernel runs at BF16, head dimension
   128, and matches the dense FP32 reference in both loss and Q/K gradients.
3. The aliased Q/Q call matches the dense reference after both argument
   gradients accumulate into the same tensor.
4. Teacher and student frequency tensors independently reconstruct to the
   registered Native and EVQ hashes.
5. Full teacher plus student loads, with exactly 96 trainable A/B tensors:
   Q/K/V across 16 layers.
6. The first real composite forward/backward has finite component losses and
   finite gradients; all 48 LoRA-B tensors receive a nonzero gradient.
7. Peak allocated/reserved memory and step time are recorded.

The upstream package was authored against different Torch/Triton versions than
the prepared server environment. That is why numerical kernel parity is a hard
gate rather than an assumption.

**Stop rule:** any failure ends the probe. Do not upgrade packages, replace the
kernel, reduce the sequence length, change the LoRA, or weaken the parity
threshold inside paid GPU time.

## 6. Stage R — one registered restoration run

Only a successful Stage G receipt authorizes the single 144-step job.

During training, record per step:

- total and component losses;
- learning rate and gradient norm;
- step duration;
- peak allocated and reserved memory.

At completion, save:

- adapter weights;
- adapter SHA-256;
- base checkpoint composite SHA-256;
- teacher and student frequency hashes;
- training-row index hash;
- immutable protocol and environment records;
- final status `TRAINING_ONLY_NOT_CAPABILITY_EVIDENCE`.

Training is considered operationally complete only when all 144 steps finish
with finite losses and the adapter and receipts are atomically finalized.
Decreasing internal loss, NLL, or PPL is not a capability result.

## 7. Stage E4 — 4K restoration gate

Evaluate three arms on the same frozen examples and decoder contract:

- `N0`: untouched Native checkpoint;
- `E0`: immediate full-EVQ checkpoint with a zero adapter;
- `EAFR`: immediate full EVQ plus the completed restoration adapter.

`N0` and `E0` must be freshly evaluated; historical values can be shown for
context but cannot replace the matched anchors.

### 7.1 Low-cost screen

Run:

- 2WikiMultiHopQA, 50 fixed held-out examples;
- all 13 RULER families, 5 fixed examples per family;
- 64 held-out 4K natural-text rows for NLL;
- one independent QA or MCQA retention slice.

Report:

- 2Wiki token F1, normalized exact match, complete generated answer, terminal
  EOS, and continuation after the answer;
- RULER official per-family scores and macro;
- natural-text NLL separately;
- the independent task's official capability metric.

### 7.2 Full 4K promotion

Only a screen pass expands to:

- 2Wiki: 200 fixed examples;
- RULER: 20 fixed examples for each of 13 families;
- natural-text NLL: 128 held-out rows;
- the predeclared independent retention set.

The route passes 4K only if all conditions hold against freshly evaluated
`N0`:

1. 2Wiki token F1 is no more than 5 percentage points lower.
2. RULER macro is no more than 10 percentage points lower.
3. No family with material Native capability collapses near zero.
4. Natural-text NLL is no more than `N0 + 0.10`.
5. The independent retention slice shows no material capability collapse.

If any condition fails, stop. Do not average a failed capability gate with an
NLL gain, and do not continue to long-context evaluation.

## 8. Stage EL — frozen-adapter length transfer

This stage is permitted only after every 4K gate passes. The restoration
adapter remains frozen; there is no long-task continuation yet.

Evaluate 8K first on identical frozen rows for:

- raw Native;
- immediate full EVQ;
- Native plus the applicable official range-scaling control;
- restored EVQ.

Use capability endpoints, not PPL, to decide whether the adapter transfers.
For retrieval tasks, save full raw generations and require the official metric.
Any NIAH exact claim additionally requires the complete expected string and
terminal EOS; a first number, substring, gold-token rank, or teacher-forced
score cannot substitute.

Run 16K only if restored EVQ has a material 8K absolute capability and improves
over both raw Native and immediate EVQ on the registered matched subset.
At 16K, use the corresponding range factor and retain every negative endpoint.

A passing 4K restoration result does not imply a passing 8K/16K result.

## 9. Conditional task continuation

This stage is allowed only if:

- broad 4K restoration passes; and
- the frozen restored adapter lacks sufficient long-task capability.

It is not part of the first run. If activated, create two separate
descendants:

1. a 2Wiki descendant;
2. a RULER-family descendant.

Do not train one combined specialist and call it broad transfer. Each update
must pair:

- one contiguous-4K restoration/distillation micro-batch; and
- one task micro-batch whose realized position IDs cover the target
  evidence-to-answer distance.

The EVQ tensor remains fixed. The physical training length remains at most 4K.
For each descendant, train a Native matched control with the same data, row
order, steps, optimizer semantics, LoRA scope, and realized position exposure.
Re-run the entire Stage E4 retention gate after continuation.

Only task-family-adapted capability may be claimed. The result is not clean
unseen-task transfer.

**Stop rule:** one predeclared route per task family, no sweep. Stop if the
descendant fails its own held-out task, loses the 4K retention gate, or has no
material 8K advantage over its matched Native control.

## 10. Controls and causal interpretation

`N0` and `E0` are required evaluation controls. They distinguish inherited
Native capability, the immediate frequency-substitution cost, and recovery
after restoration.

A generic CE-only or output-logit-KD training control is not required for the
first feasibility decision because the completed search already shows that
closely related routes fail. Such a new arm is optional only after EAFR passes
and only if a publication-level causal attribution requires it.

Even a complete pass supports only:

> On the tested OLMo-2 checkpoint and evaluation suite, teacher-guided QKV
> attention-function alignment restored the specified 4K capability margins
> under a fixed EVQ frequency grid.

It does not establish exact functional equivalence, no forgetting outside the
tested suite, EVQ-only causality, universal mature-model conversion, or the
behavior of a model pretrained with EVQ from initialization.

## 11. Artifact owners

The execution must create narrow, machine-readable owners for:

- no-GPU prepared receipt;
- GPU-ready smoke receipt;
- training logs and final run manifest;
- adapter and all hashes;
- `N0`, `E0`, and `EAFR` 4K raw generations and metrics;
- 8K/16K raw generations and metrics, only if opened by the gates;
- a final standalone evidence report that labels positive, negative, skipped,
  failed, and unverified stages.

The experiment may enter the reviewer-facing playbook only after raw/hash
promotion and synchronization with a standalone report. Until then its
evidence tier is `DESIGN_ONLY_OR_PENDING`.

## 12. Handoff state

Prepared now:

- implementation and evaluator support;
- pinned upstream dependency identity;
- unit tests and dense-reference checks;
- no-GPU asset/protocol receipt;
- server-side code copy.

Not completed:

- GPU kernel/full-model smoke;
- training;
- any new 4K, 8K, or 16K result;
- any rebuttal claim.

The next operator should attach a compatible GPU, run Stage G exactly once, and
continue to Stage R only if the GPU-ready receipt is produced. No current
evidence justifies skipping that gate.
