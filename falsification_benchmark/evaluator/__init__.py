"""Deterministic CPU-only scorer for the theory falsification benchmark."""

from .core import BenchmarkError, score_predictions, validate_predictions

__all__ = ["BenchmarkError", "score_predictions", "validate_predictions"]
