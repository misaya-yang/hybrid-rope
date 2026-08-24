# Zero-parameter single-table mature-checkpoint gate (2026-08-24)

Status: frozen before GPU execution.

## Question

Can the paper's closed-form allocation construction, normalized to the released
OLMo checkpoint's exact Native sampled support, improve 2x natural-text OOD NLL
without materially harming 1x when installed as one static table?

## Candidate and invariants

- Candidate: endpoint-grid anchored EVQ-Cosh, `tau=2`.
- Inputs: only the released Native inverse-frequency tensor and the frozen
  analytic constant `tau=2`.
- Exact Native fast/slow endpoints and log span; 64 pairs; strict monotonicity.
- One static table and attention scaling `1.0` at both 1x and 2x.
- Zero learned parameters, zero optimizer steps, zero model-weight updates.
- No task labels, OOD losses, attention statistics, collision score,
  `L_target`, requested length, or route is read during construction.

This is a frozen-checkpoint test of the analytic table, not another training
seed for EVQ-Cosh and not a claim that frozen transplantation must work.

## Stage 1: PG-19

Use the already frozen target-free token manifest and its first 20 rows at each
of 1x and 2x. Reuse the completed Native rows with identical row hashes; do not
rerun Native.

Promote only if both hold:

1. mean 1x tail-NLL regression versus Native is at most `+0.05`;
2. mean 2x tail NLL is strictly lower than Native.

If either fails, stop this table before RULER or natural-task generation. Do
not tune `tau` after observing the result.

## Stage 2: capability

Only after Stage 1 passes, run the existing core-4 RULER 8K screen with five
rows per task, then the 20-row-per-cell Qasper/2Wiki 1x/2x natural screen. This
is a candidate gate, not manuscript evidence by itself.
