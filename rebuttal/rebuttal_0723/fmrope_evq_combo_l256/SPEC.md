# EVQ-Cosh x FMRoPE: 151.9M / L=256 diagnostic

Status: registered single-seed combination arm. It is independent of the
three-arm `fmrope_125m_l256` result and must not overwrite those checkpoints.

## Question

Does FMRoPE-style target-base retargeting remain useful when the training
frequency allocation uses EVQ-Cosh rather than a uniform log grid?

## Matched protocol

- Model: exact tied-embedding parameter count 151,898,880.
- Train: FineWeb-Edu shard 000, 99,942,400 tokens, `L=256`, seed 42.
- Validation: disjoint shard 004; the same 32 fixed anchors and lengths
  `256/512/1024/2048` as the parent three-arm experiment.
- Optimizer, initialization, row order, precision, and training budget are
  inherited unchanged from `fmrope_125m_l256`.
- Training frequency:
  `omega_k = 256^(-phi_k(tau=4))` on the EVQ midpoint quantizer.
- Inference conditions:
  - fixed: `omega_k = 256^(-phi_k)`;
  - target-matched: `omega_k = L_target^(-phi_k)`.
- Primary metric: paired final-128-token NLL. PPL is only
  `exp(mean NLL)` for readability.

The in-domain `L=256` fixed and target-matched conditions must be exactly
identical. Trainable initialization hash must match all three parent arms.

## Claim boundary

This is a new local combination derived from the two schedule formulas. It is
not official FMRoPE code, does not establish a universal optimum, and remains
a 100M-token single-seed diagnostic.
