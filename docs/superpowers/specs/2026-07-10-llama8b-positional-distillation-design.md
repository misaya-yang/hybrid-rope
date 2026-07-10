# LLaMA-3-8B Positional Distillation Pilot Design

## Status

Approved for implementation on 2026-07-10. This pass prepares code and
documentation only. It must not launch a GPU experiment.

## Goal

Test, with seed 42, whether the reported 8K degradation in the existing
LLaMA-3-8B EVQ-LoRA experiment is primarily caused by supervised LongAlign
adaptation and broad q/k/v/o LoRA drift rather than by EVQ frequency injection
itself.

The pilot trains no task answers. A frozen native-geometric view of the same
model supplies representation targets, and only q/k LoRA parameters may adapt
the EVQ student back toward the pretrained model's behavior.

## Scientific Question

The existing supporting row combines three changes:

1. native geometric RoPE is replaced by EVQ-Cosh;
2. rank-64 LoRA is applied to q/k/v/o projections;
3. all LongAlign tokens are optimized with causal language-model loss.

Consequently, Base to EVQ-LoRA cannot attribute the 8K PPL increase or the
16K/32K gain to any one change. The pilot separates injection shock,
optimizer drift, and positional recovery.

## Experimental Arms

| Arm | Student schedule | Adapter | Training | Role |
| --- | --- | --- | --- | --- |
| `base_geo` | native geometric | none | none | pretrained reference |
| `base_evq` | EVQ-Cosh, tau 1.414 | none | none | direct injection shock |
| `geo_distill_s42` | native geometric | q/k LoRA | representation distillation | optimizer and adapter drift control |
| `evq_distill_s42` | EVQ-Cosh, tau 1.414 | q/k LoRA | representation distillation | treatment |

Only the last two arms create checkpoints. All four arms use the same base
checkpoint and evaluation data.

## Frozen Data

The default source is `HuggingFaceFW/fineweb-edu`, config `sample-10BT`, split
`train`, shuffled with seed 42. The preparation command packs raw text with the
LLaMA-3 tokenizer into:

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

This avoids a second 8B model in memory and allows the existing 96GB profile of
batch size 2 and gradient accumulation 4.

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
- Optimizer steps: 300.
- Per-device batch: 2.
- Gradient accumulation: 4.
- Effective batch: 8 sequences.
- Learning rate: 2e-5.
- Warmup: 30 steps.
- Scheduler: cosine.
- Weight decay: 0.01.
- Gradient clipping: 1.0.
- Precision: bf16.
- Attention: PyTorch SDPA.
- Gradient checkpointing: enabled, non-reentrant.
- Logging: every 5 optimizer steps.
- Checkpoint used for claims: fixed final step 300; do not select a checkpoint
  using 16K/32K outcomes.

The Geo control runs the same 300 steps even though its initial teacher/student
loss should be nearly zero. Any drift is therefore directly observable.

## Evaluation

The evaluator must support all four arms without silently replacing their
frequency schedules. It reports:

1. PPL and NLL at 8K, 16K, and 32K on the frozen WikiText text file, five
   non-overlapping chunks per length;
2. held-out representation error on the packed validation tensor;
3. recovery fraction relative to the unadapted EVQ injection shock;
4. optional quick RULER/AR diagnostics, clearly separated from the A-stage
   pass/fail decision.

Every adapter evaluation must load and validate `custom_inv_freq.pt` after
PEFT loading. Result files must carry the variant name, schedule method,
frequency hash, checkpoint identity, data-manifest hash, and seed.

## Decision Gates

The A-stage pilot passes only if all primary gates hold:

- `geo_distill_s42` PPL@8K is within 1% of `base_geo`;
- `evq_distill_s42` PPL@8K is no more than 5% above `base_geo`;
- `evq_distill_s42` improves PPL@16K by at least 2x versus `base_geo`;
- `evq_distill_s42` improves PPL@32K by at least 4x versus `base_geo`;
- held-out representation error recovers at least 90% of the `base_evq`
  injection shock.

If EVQ PPL@8K remains more than 10% above Base, do not run more seeds. Inspect
loss convergence and adapter norms before deciding whether to test rank,
learning rate, or adapter scope. Quick RULER is diagnostic only: without an AR
capability gain, the pilot may support clean positional recovery but not an
industrial context-capability claim.

## Runtime Budget

The repository's historical A800 run processed an equivalent effective sample
budget in about 1.44 hours without a teacher forward. The single-model teacher
adds one no-grad backbone pass per batch. The conservative RTX PRO 6000 budget
is therefore:

- Geo distillation: 2-3 hours;
- EVQ distillation: 2-3 hours;
- four-arm PPL, hidden recovery, and quick diagnostics: 2-3 hours;
- total single-GPU reservation: 6-9 hours.

This is an estimate. The launcher records actual GPU identity, power limit,
peak allocated memory, and wall time so later seeds use measured throughput.

## Non-Goals

- Do not modify paper metrics or `paper/main.pdf` in this pass.
- Do not run Stage 2 retrieval adaptation.
- Do not run YaRN, LongRoPE, or a rank sweep.
- Do not promote a single-seed result to primary evidence.
- Do not launch the prepared commands as part of implementation validation.
