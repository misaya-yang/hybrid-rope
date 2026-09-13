"""Checkpoint-conditioned attention replay for one fixed RoPE table."""

from .core import (
    ReplayCapture,
    evaluate_replay_objective,
    finite_rho_grid,
    increments_to_exponents,
    profile_inv_freq,
)
from .solver import project_simplex, solve_increment_qp
from .capture_io import load_capture, save_capture

__all__ = [
    "ReplayCapture",
    "evaluate_replay_objective",
    "finite_rho_grid",
    "increments_to_exponents",
    "profile_inv_freq",
    "project_simplex",
    "solve_increment_qp",
    "load_capture",
    "save_capture",
]
