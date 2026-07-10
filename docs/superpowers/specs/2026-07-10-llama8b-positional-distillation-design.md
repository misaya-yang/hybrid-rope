# LLaMA-3-8B Positional Distillation Pilot Design

## Status

Approved for implementation on 2026-07-10. This pass prepares code and
documentation only. It must not launch a GPU experiment.

## Goal

Test, with seed 42, whether an EVQ frequency transplant can be recalibrated
toward the frozen native-Geo model after removing LongAlign/task supervision
and restricting adaptation to q/k. This is a new diagnostic protocol; it cannot
by itself attribute the old LoRA row's degradation to LongAlign, LM loss, v/o
updates, or learning rate.

The pilot trains no task answers. A frozen native-geometric view of the same
model supplies representation targets, and only q/k LoRA parameters may adapt
the EVQ student back toward the pretrained model's behavior.

## Scientific Question

The existing supporting row combines three changes:

1. native geometric RoPE is replaced by EVQ-Cosh;
2. rank-64 LoRA is applied to q/k/v/o projections;
3. all LongAlign tokens are optimized with causal language-model loss.

Consequently, Base to EVQ-LoRA cannot attribute the 8K PPL increase or the
16K/32K gain to any one change. The pilot separates direct injection shock from
teacher-guided q/k recovery. It does not remove LoRA and does not close the old
LongAlign/LoRA confound without a separate matched original-protocol block.

## Experimental Arms

| Arm | Student schedule | Adapter | Training | Role |
| --- | --- | --- | --- | --- |
| `base_geo` | native geometric | none | none | pretrained reference |
| `base_evq` | EVQ-Cosh, tau 1.414 | none | none | direct injection shock |
| `geo_distill_s42` | native geometric | q/k LoRA | one-step representation null | pipeline/null control |
| `evq_distill_s42` | EVQ-Cosh, tau 1.414 | q/k LoRA | representation distillation | treatment |

Only the last two arms create checkpoints. All four arms use the same base
checkpoint and evaluation data.

## Frozen Data

The default source is `HuggingFaceFW/fineweb-edu`, config `sample-10BT`, split
`train`, revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, shuffled with
seed 42. Validation is packed first, its final document remainder is discarded,
and training starts at the next document, making the split document-disjoint.
The preparation command packs raw text with the LLaMA-3 tokenizer into:

- 2,400 training sequences of exactly 8,192 tokens (19,660,800 tokens);
- 128 disjoint validation sequences of exactly 8,192 tokens (1,048,576 tokens).

The packed tensors use `int32` on disk and are converted to `long` by the
dataset. Preparation writes a manifest containing source identity, tokenizer
identity, sequence counts, document count, and SHA-256 hashes. A local JSONL
fallback is allowed only when each row has a non-empty `text` field; chat
messages and instruction/answer records are deliberately not accepted.

## Model and Adapter

- Base: Meta-Llama-3-8B-Instruct.
- Teacher schedule: exact native geometric endpoint frequencies.
- Treatment schedule: EVQ-Cosh midpoint allocation with tau 1.414.
- LoRA targets: `q_proj,k_proj` only.
- Rank: 64.
- Alpha: 128.
- Dropout: 0.0.
- Base parameters, embeddings, value/output projections, MLP, norms, and LM
  head remain frozen.

RoPE directly acts on q/k. Excluding v/o removes adaptation capacity that is
not needed to recalibrate positional attention and cuts the trainable adapter
size roughly in half relative to the existing q/k/v/o rank-64 setup.

## Single-Model Teacher/Student Flow

One PEFT-wrapped model is kept in memory. For every microbatch:

1. inject native geometric frequencies;
2. disable the LoRA adapter and run the backbone under `torch.no_grad()`;
3. detach the teacher's final normalized hidden states;
4. inject the student's selected frequencies;
5. enable q/k LoRA and run the backbone with gradients;
6. compute the positional representation loss;
7. leave the student schedule installed so gradient-checkpoint recomputation
   uses the correct frequencies.

This avoids a second 8B model in memory. The launcher defaults to batch 2 and
gradient accumulation 4, but permits batch 4/accumulation 2 or batch
8/accumulation 1 after a short measured equivalence and memory gate; effective
batch must remain 8.

## Objective

No token labels, causal-LM loss, task answers, retrieval examples, or synthetic
supervision are used.

For teacher hidden state `t` and student hidden state `s`, calculate a
scale-normalized MSE independently in three position buckets:

```text
B1 = [0, 2048)
B2 = [2048, 4096)
B3 = [4096, 8192)
L_b = mean((s - t)^2 over valid tokens and hidden dimensions)
      / max(mean(t^2 over the same elements), 1e-8)
L = mean(L_b over non-empty buckets)
```

Equal bucket weighting prevents the 4K-8K region from dominating solely
because it contains more tokens. Matching the final normalized hidden state is
sufficiently strict because the frozen LM head maps that state to logits.

## Training Protocol

- Seed: 42.
- Sequence length: 8,192.
- Optimizer steps: EVQ 300; Geo null 1.
- Per-device batch: 2.
- Gradient accumulation: 4.
- Effective batch: 8 sequences.
- Learning rate: 2e-5.
- Warmup: EVQ 30 steps; Geo null 0.
- Scheduler: cosine.
- Weight decay: 0.01.
- Gradient clipping: 1.0.
- Precision: bf16.
- Attention: PyTorch fused SDPA, verified by a runtime Flash-SDPA smoke gate.
- Gradient checkpointing: enabled, non-reentrant.
- Logging: every 5 optimizer steps.
- Optimizer: fused AdamW.
- Compilation: student backbone only, `torch.compile(dynamic=False,
  mode="default")`; the teacher remains eager and compilation can be disabled
  for the required target-GPU A/B.
- Recovery checkpoints: every 100 EVQ optimizer steps; resume automatically.
- Checkpoint used for claims: fixed final step 300; do not select a checkpoint
  using 16K/32K outcomes.

The native-Geo teacher and student are identical at initialization and LoRA is a
zero-output no-op. A 300-step Geo run therefore cannot be interpreted as a
matched optimizer-drift control and mainly burns GPU time. The retained
one-step arm is explicitly a pipeline/null sentinel; its loss and adapter update
norm are recorded, and it must remain behaviorally identical at evaluation.

## Evaluation

The evaluator must support all four arms without silently replacing their
frequency schedules. It reports:

1. PPL and NLL at 8K, 16K, and 32K on the frozen WikiText text file, five
   non-overlapping chunks per length;
2. held-out representation error on all 128 packed validation sequences,
   evaluated in batches of four;
3. recovery fraction relative to the unadapted EVQ injection shock;
4. optional quick RULER/AR diagnostics, clearly separated from the A-stage
   pass/fail decision.

Every adapter evaluation must load and validate `custom_inv_freq.pt` after
PEFT loading. Result files must carry the variant name, schedule method,
canonical frequency hash, checkpoint identity, adapter hash, data-manifest
hash, WikiText hash, tokenizer fingerprint, exact evaluation config, and seed.
The 32K LM head is evaluated in token chunks so the evaluator does not allocate
one full 32K-by-vocabulary logit tensor.

## Decision Gates

The A-stage pilot passes only if all primary gates hold:

- `geo_distill_s42` PPL is within 1% of `base_geo` at 8K, 16K, and 32K;
- `evq_distill_s42` PPL@8K is no more than 5% above `base_geo`;
- `evq_distill_s42` improves PPL@16K by at least 2x versus the matched Geo
  null arm;
- `evq_distill_s42` improves PPL@32K by at least 4x versus the matched Geo
  null arm;
- held-out representation error recovers at least 90% of the `base_evq`
  injection shock.

If EVQ PPL@8K remains more than 10% above Base, do not run more seeds. Inspect
loss convergence and adapter norms before deciding whether to test rank,
learning rate, or adapter scope. Quick RULER is diagnostic only: without an AR
capability gain, the pilot may support clean positional recovery but not an
industrial context-capability claim.

## Runtime and Throughput Gate

No project-specific speedup or wall-time estimate is considered verified before
the target RTX PRO 6000 run. The one-step Geo null removes 299 structurally
uninformative optimizer steps. The remaining EVQ run records PyTorch/CUDA and
GPU identity, compile configuration, tokens/s, peak allocated/reserved memory,
and five-second utilization/power/clock samples. Before the 300-step run, compare
the same frozen microbatches with compile on/off and, if memory permits,
checkpointing on/off or batch 4/accumulation 2. Select by steady-state wall time,
not claimed library-level speedups, and retain explicit VRAM headroom.

The launcher `benchmark` phase performs these comparisons in isolated 12-step
non-claim directories and ranks the ten post-warmup optimizer steps by nominal
tokens/s; only candidates retaining at least 5% (minimum 4 GiB) VRAM headroom
are eligible. Claim resume is allowed only when an immutable run protocol
matches exactly. Final claim readiness additionally requires an append-only
invocation ledger with continuous step coverage and hashed hardware, log, and
telemetry evidence for every fresh or resumed segment.

FP8/FP4, QLoRA, DDP/FSDP, and a second resident teacher are excluded from the
first claim run because they change numerics or add unsupported execution paths.

## Non-Goals

- Do not modify paper metrics or `paper/main.pdf` in this pass.
- Do not run Stage 2 retrieval adaptation.
- Do not run YaRN, LongRoPE, or a rank sweep.
- Do not promote a single-seed result to primary evidence.
- Do not describe this as removing LoRA or as full-model industrial fine-tuning.
- Do not launch the prepared commands as part of implementation validation.
