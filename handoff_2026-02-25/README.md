# Handoff 2026-02-25 (Reviewer-Critical Remediation)

This folder is the single entrypoint for the current NeurIPS rescue track.

## Read order (strict)

1. `01_IMPLEMENTED_SCOPE.md`
2. `02_VALIDATION_SNAPSHOT_AIDEMO.md`
3. `03_NEXT_EXECUTION_GATES.md`

## Scope

- Consolidates what has been implemented for:
  - LongBench parity and 21-task support
  - Protocol-lock propagation
  - Statistical rigor (FDR + claim policy)
  - Theory-strengthening scripts
  - Reproducibility export from empirical prior
- Records runtime verification done in `aidemo`.
- Defines execution gates before launching expensive reruns.

## Important

- Current server run (`cross_model_fast_tuned_b1_gc`) is a fast validation line, not the final full-resources pipeline.
- Do not start full expensive reruns until Gate A (pipeline parity) and Gate B (protocol lock) are both green.
