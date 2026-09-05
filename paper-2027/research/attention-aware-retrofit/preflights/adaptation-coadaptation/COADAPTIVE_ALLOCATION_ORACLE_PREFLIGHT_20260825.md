# Mature OLMo fixed-support co-adaptive allocation oracle

- **Date:** 2026-08-25
- **Status:** executed; registered all-shell gate failed; completed attribution,
  matched dense-natural recovery, and capability follow-up are owned by
  [`../../results/adaptation-coadaptation/COADAPTIVE_ALLOCATION_ORACLE_RESULT_20260825.md`](../../results/adaptation-coadaptation/COADAPTIVE_ALLOCATION_ORACLE_RESULT_20260825.md)
- **Role:** internal method-development oracle, not manuscript evidence
- **Code:**
  `rebuttal/rebuttal_0723/experiments/olmo2_allocation_oracle_5090/`

## Question

Can a mature OLMo-2 1.485B checkpoint jointly move the fixed-support interior
RoPE allocation and Q/K coordinates while preserving Native-window language
modeling and improving unseen phase exposure?  This measures an attainable
co-adapted frontier; it is not a proposed final fixed schedule.

## Intervention

The sampled Native fast and slow frequency endpoints remain exact.  The 63
positive normalized gaps parameterize the 62 effective interior-allocation
degrees of freedom, shared by every layer and head.  The model starts bitwise at
the Native table and trains only these gap logits plus rank-64/alpha-128 Q/K
LoRA.  The table remains strictly ordered by construction; attention amplitude,
V/O, routing, support, and the physical 4K token budget do not change.

Two phase steps and one promptless-natural replay step repeat.  Each phase
micro-batch contains one contiguous row and one deterministic offset from each
of three predeclared log shells: `(0,L]`, `(L,3L]`, and `(3L,15L]`, with
`L=4096`.  Only the natural query/answer suffix receives the offset.  The
construction therefore uses a scale-risk prior but no single target or
evaluation length.

The full run is 300 optimizer steps, micro-batch 4, accumulation 2, BF16,
Flash-only SDPA, fused AdamW, Q/K LoRA LR `5e-5`, allocation LR `1e-3`, and a
20-step shared warm-up.  Data are the already-frozen 896/128-row natural-span
phase view, 32-row independent promptless replay, and 32-row raw retention
view.  No RULER, 2Wiki, or final manuscript endpoint is used for training or
selection.

## Decision rule

A useful oracle candidate must differ from Native while satisfying all three:

1. held-out raw-4K sparse-token NLL increase at most `+0.05`;
2. zero-offset held-out phase-task NLL increase at most `+0.05`;
3. held-out phase-task NLL improves over the exact Native initialization in
   every registered nonzero shell.

Failure closes only this shared fixed-support, Q/K-only, 300-step oracle
protocol.  It does not establish an allocation impossibility.  A passing run
still requires downstream capability and forgetting evaluation before any
method or paper claim.

## No-GPU receipt

The CPU preflight bound the released checkpoint owner, phase/replay/retention
manifests, code hashes, output path, and exact command.  It verified exact
Native initialization, fixed endpoints, strict ordering, deterministic shell
sampling, suffix-only position shifts, and the fullgraph-compilable allocation
path.  The current READY receipt SHA-256 is
`42a5713649e23bab48f4096821dd538f1e73521cfbb64c54c27052f315956873`.
The unauthorized run path exits before model loading and produced no output.

The authorized smoke and full run followed this sequence. This document remains
the frozen before-execution contract; result interpretation belongs only to the
linked owner.
