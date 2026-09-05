# Extended target-free allocation preflight — M4 window (2026-08-24)

Status: `AUTHORIZED_INTERNAL_PRELIMINARY_RUN`; executed — its own decision
rule is applied by the canonical report
[`../results/PHASE_ALLOCATION_M4_EXTENDED_RESULT_20260824.md`](../../results/PHASE_ALLOCATION_M4_EXTENDED_RESULT_20260824.md).

This preflight records the follow-up matrix authorized after the completed
`phase_isotropy_50m_m4_20260824` screen.  It is internal method selection, not
manuscript evidence.  It does not alter `paper/`, TeX, PDF, `AGENTS.md`, or
`HANDOFF.md`.

## Why this matrix

The first screen used a 50.9M model, 8,388,608 tokens, local WikiText,
four validation anchors, and base 256.  Neither phase-isotropy nor its
anchored-Cosh control produced the positive OOD direction seen in the
151.9M/500M-token FineWeb-Edu exact-range owners. The follow-up runs inspect
three questions across two regimes, but do **not** factorially separate them:

1. training budget / co-adaptation;
2. sampled spectral range (`base=256` versus `base=500,000`); and
3. the structural score used to construct the density.

## Theory ablation family

For the causal pair Gram,

\[
  \lambda_\pm(\omega)=\frac{1\pm|\chi_L(2\omega)|}{2}.
\]

The nearest-cell density is always `rho(q) proportional to q^(1/3)` with
endpoint-inclusive inverse-CDF quantiles.  The fixed score choices are:

| Arm | score `q(omega)` | purpose |
|---|---|---|
| FMRoPE | uniform log-frequency density | paper-faithful control |
| anchored EVQ-Cosh | closed-form `tau=4`, same endpoints | established aggressive reference |
| phase-isotropy | `lambda_- / lambda_+` | original target-free candidate |
| pair-volume | `4 lambda_- lambda_+` | conservative area exposure |
| min-eigenvalue | `lambda_-` | absolute weakest-quadrature exposure; removes the ratio denominator |

The last arm is not claimed optimal.  It tests whether the failure is caused by
the condition-ratio denominator rather than by the pair-Gram premise itself.
No target length, loss, activation, attention, or result is used in any table.

## Frozen training contract

- architecture: existing 50M M4 configuration (50.9M parameters), hidden 512,
  6 layers, 8 heads, `d_head=64`, `K=32`;
- `L_train=256`, seed `137`, MPS float32, global batch 256,
  micro-batch 32, gradient accumulation 8;
- deterministic row sampler, stored-token prefix, AdamW and LR schedule remain
  the existing M4 harness contract;
- validation prefix, row order, and frozen anchors are shared within each
  matrix cell; primary endpoint is final-128-token teacher-forced NLL and full
  NLL is retained;
- evaluation lengths remain `256/512/1024/2048` and never enter construction.

## Matrix and budget

| Stage | base | tokens / arm | arms | role |
|---|---:|---:|---|---|
| A | 256 | 50,331,648 | FMRoPE, Cosh, phase-isotropy, pair-volume, min-eigenvalue | budget/co-adaptation and score ablation |
| B | 500,000 | 25,165,824 | Geo (runner key `FMRoPE`), Cosh, phase-isotropy, min-eigenvalue | separate range-and-budget regime; no factor attribution |

The current completed 8,388,608-token base-256 three-arm screen remains the
zero point.  No second training seed, target-aware table, pair-volume training
arm outside this matrix, or larger model is authorized in this window.

The Stage B runner retained the historical key `FMRoPE`; because its realised
table is geometric at base 500,000, all scientific interpretation calls that
arm `Geo`. The raw key remains only for receipt matching.

## Decision rules

- Report each cell and paired deltas; do not pool across bases or budgets.
- Treat a candidate as a *regime-consistent lead* only if its 50M base-256 arm
  improves all three OOD lengths without exceeding `+0.01` at 256, and the
  same direction is not contradicted by its base-500K diagnostic.
- A positive result is still `PRELIMINARY_INTERNAL`; it cannot replace the
  existing exact-range owner without a frozen multi-seed protocol.
- If the anchored-Cosh control remains neutral/negative at 50M base-256, label
  the regime as unresolved rather than attributing the result to the new score.

## Stop conditions

Stop the matrix if MPS OOM cannot be recovered by halving micro-batch while
preserving global batch, if the first real step is non-finite, or if a run loses
its checkpoint/protocol receipt.  Do not tune a score after reading NLL.
