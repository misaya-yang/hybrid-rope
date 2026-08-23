# OLMo-2 finite function-morph preflight

- **Date:** 2026-08-21
- **Status:** `DEPRIORITIZED_READY_NOT_NEXT_GATE`; code and inputs prepared on
  a no-GPU host; no model evaluation or training has run
- **Code:**
  `rebuttal/rebuttal_0723/experiments/olmo2_function_morph_5090/`
- **Role:** internal protocol owner and navigation receipt, not paper evidence

## Reviewer question

The phase-chord construction is derived from measured attention-distance mass,
but that alone does not show that its finite replacement direction is cheap
for a mature model. The smallest direct test is to measure the actual
candidate-specific function change, while comparing against a
non-attention-aware control with the same sampled support and the same RMS
log-frequency displacement.

## Frozen finite audit

The shared-table OLMo-2 audit has three candidate targets:

1. the OLMo 4K R0 phase-chord table with `lambda=0.1`;
2. an endpoint-inclusive exponential coordinate warp, bent in the opposite
   direction and matched to phase-chord on actual R0-Native RMS log-frequency
   displacement; and
3. endpoint-anchored EVQ-Cosh with the zero-search rule
   `tau=head_dim/sqrt(4096)=2`.

Every arm keeps the exact Native fast and slow sampled endpoints. The finite
log-frequency morph grid is `t={0,.05,.25,.5,.75,1}`. The future evaluator
records per-example Native-teacher forward KL and tail-token NLL delta at 4K,
8K, and 16K on four deterministic pure-text rows per length. It evaluates the
final 64 next-token predictions, uses BF16 Flash-only SDPA, creates no
optimizer, enables no gradients, and restores the Native table on exit.

This is a finite path audit. It is not a full Fisher matrix, a Hessian
eigensystem, a continuous basin, or a learned retrofit.

## No-GPU receipt

The no-GPU host bound the exact OLMo checkpoint and R0 owner, generated the
missing deterministic 8K/16K views from the same validation parquet, froze all
18 candidate/morph tensors, and passed four contract tests.

| Item | SHA-256 or result |
| --- | --- |
| OLMo weight file | `36d044c73655bb904f822915e6294ba3dae8e6e1af5e703e9d452f2d6a3a294f` |
| OLMo configuration | `0d15ebb6cb8d998513b46ef337214176a6fd59fe5f16b30387c70d5f87795a9c` |
| R0 collection | `f507469a564b273525896c8ad55fc51568f32663145c3abcc92a947c253641e2` |
| 4K / 8K / 16K token views | `5981658487ff344095ae740b1245c2d4488c034bfa79a8605a680b5796b7af05` / `fee58b60c30af33f32156453ae710508fb915534cc47aa250fc9fd3016c7ef57` / `e47c8296eba3a13d85c4a83df0192817e9a8abc336a9c5ec7114b0e07ca93b07` |
| Target manifest file | `cf03385431df1084508055232853d341a903aa2d8b99fca087bc45d513d34af9` |
| Target manifest canonical content | `384941dfa7dc95daceab87d709fa805028634de58db3baf05bb5c9186f11598e` |
| Dry-run receipt file | `215375650cf4f0244603fd55697fab47d373ddcc0f093a5769c04de63b2639c7` |
| Contract tests | `4/4` passed under Python 3.12 |
| Execution proof | CUDA unavailable and uninitialised; checkpoint not deserialised; model not loaded; no gradients, optimizer, training, download, or network access |

The phase-chord and matched-control RMS log displacements are
`1.4765455885816183` and `1.4765455885816181`; the absolute difference is
floating-point roundoff (`2.22e-16`). Their mean log displacements have
opposite signs. Target float32 hashes are:

- phase-chord: `4d985cce3c47506079119d9d0454d02a49d753238bce86ea4bf0f0b2e398e931`;
- matched control: `875f808c6791b1e2ca03413809b0b25a21a7c3c5f7b168f9d4074c76bca30155`;
- anchored EVQ-Cosh: `9e82c83312b6f5f44c63f6ad4fea8fbdd1256a64b3bc853ae8e52c5538a4d02d`.

## Execution gate and decision

The GPU entry point fails before importing Torch unless both `--authorize`
and `OLMO_FUNCTION_MORPH_GPU_AUTHORIZED=1` are present. The unauthorized-path
test exited nonzero and created no raw result file.

No GPU action is authorized by this preflight. This audit is retained as a
reproducible diagnostic but is not the next paid gate: it still studies
whole-table replacement, whereas the immediate research objective is a
practical LoRA method that preserves a capable Native path by construction.
Run it only if a later residual result creates a specific unresolved
candidate-attribution question.
