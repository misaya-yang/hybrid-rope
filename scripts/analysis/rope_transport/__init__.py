"""Zero-GPU RoPE table transportability analysis.

Computes feasible residuals for static maps of projected Q/K coordinates under
an isotropic-content surrogate. It does not measure actual checkpoint function
preservation, certify a global optimum, or bound arbitrary Q/K LoRA updates.
"""

from __future__ import annotations

__all__ = ["METHOD_ID"]

METHOD_ID = "rope_transport_v1"
