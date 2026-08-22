"""Zero-GPU RoPE table transportability analysis.

Answers, without any model forward or training, how much of a mature model's
in-window attention function survives a RoPE frequency-table change under the
best possible static Q/K reparameterization -- the exact operator class of the
post-hoc transplant obstruction and the strict superset of any Q/K LoRA.
"""

from __future__ import annotations

__all__ = ["METHOD_ID"]

METHOD_ID = "rope_transport_v1"
