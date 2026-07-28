# OLMo-2 1.485B EVQ attention-function restoration

Status: `DESIGN_ONLY_RUNTIME_UNVERIFIED`

This is an offline-prepared research candidate, not rebuttal evidence and not
an authorized GPU run.

## Why this is not another morph run

The repository already rejects linear Native-to-EVQ frequency morphing with
ordinary token loss, with or without sparse output-logit KL. This candidate
does not interpolate frequencies. The student uses the exact full EVQ endpoint
from step 0, while a frozen Native teacher supplies internal attention targets
on identical contiguous 4K token sequences.

The method is inspired by LinearARD (arXiv:2604.00004, upstream commit
`23866f68a8b65da796c75439a06d4cb996bcb7bb`) but is deliberately not described
as a replication. LinearARD's Q/Q, K/K and V/V self-relations do not uniquely
identify the real QK attention map: separate orthogonal transforms can preserve
self-relations while changing QK. The registered objective therefore uses:

1. post-RoPE causal `KL(A_native || A_evq)`, where `A=softmax(QK^T/sqrt(d))`;
2. normalized MSE between the teacher and student per-head `A@V` contexts;
3. Q/Q, K/K and V/V relation KL only as a lower-weight auxiliary.

The frozen student backbone receives LoRA only on Q, K and V in all 16 layers.
The distilled target is layer 15. V is the attention value projection and is
not called post-RoPE.

## Registered first run

- checkpoint: OLMo-2 1.485B Instruct;
- teacher: Native RoPE, frozen, eval mode;
- student: exact full EVQ-Cosh frequency tensor from step 0;
- data: the existing hash-bound LongAlign 4K fixed view;
- position IDs: contiguous `0..4095`; no virtual 8K/16K exposure;
- LoRA: Q/K/V only, every layer, rank 512, alpha 1024, dropout 0;
- optimizer: AdamW, LR `2e-5`, four warmup steps, minimum LR ratio 0.9;
- budget: 144 optimizer steps, micro-batch 1, accumulation 4;
- student tokens: 2,359,296; teacher tokens: 2,359,296;
- model compile: disabled;
- gradient checkpointing: disabled for the first registered run.

The rank is the same 25% hidden-dimension ratio as the published LinearARD
LLaMA recipe (`1024/4096`), scaled to OLMo's 2048 hidden dimension. It is fixed
before the run; this is not a rank or weight sweep.

## Before a GPU run

The no-GPU receipt is not launch authorization. On the active RTX 5090, run the
registered discarded smoke only. It must pass:

- upstream Triton-kernel BF16 D128 loss and gradient parity against the dense
  reference, including aliased Q/Q gradients;
- exact Native-teacher and full-EVQ-student frequency identities;
- exactly 96 trainable tensors (A/B for Q/K/V across 16 layers);
- full teacher-plus-student memory fit;
- finite composite loss and finite, nonzero QKV-LoRA gradients.

Any failure is a stop, not permission to change Triton, rank, loss weights or
runtime ad hoc.

## Scientific gates after training

Training loss, NLL or PPL alone does not establish restoration.

The first screen must use frozen held-out rows and evaluators:

- 2WikiMultiHopQA at 4K;
- all 13 RULER families at 4K;
- held-out natural NLL at 4K;
- one independent QA/MCQA retention slice.

Pre-registered non-inferiority margins:

- 2Wiki token F1 no more than 5 percentage points below the freshly evaluated
  Native anchor;
- RULER13 macro no more than 10 points below Native, with no
  Native-positive family collapsing near zero;
- natural NLL no more than `Native + 0.10`.

Failure of any gate stops the route. Only if every 4K gate passes may the same
frozen adapter be evaluated at 8K and 16K. A later long-task continuation, if
needed, is a separate experiment requiring matched Native and range controls;
it cannot be inferred from the restoration stage.

## Claim boundary

A passing 4K suite would support tested-suite teacher-guided restoration under
fixed EVQ. It would not prove structural equivalence, no forgetting, EVQ alone,
or unseen long-task transfer. An 8K/16K capability claim requires strict
autoregressive task metrics and the corresponding matched controls.
