"""Audited plan-first pipeline for the 2026-09-13 RoPE range study."""

from .mechanisms import (
    DELTA_GRID,
    MechanismNotLocallySeparable,
    audit_mechanism_deltas,
    canonical_pair_overlap,
    causal_separation_weights,
    phase_unseen_fraction,
    table_collision,
    table_phase_ood,
)

__all__ = (
    "DELTA_GRID",
    "MechanismNotLocallySeparable",
    "audit_mechanism_deltas",
    "canonical_pair_overlap",
    "causal_separation_weights",
    "phase_unseen_fraction",
    "table_collision",
    "table_phase_ood",
)
