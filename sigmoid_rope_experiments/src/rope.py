from __future__ import annotations

import math
from typing import Dict, Tuple

import torch


class RoPEFrequencyAllocator:
    """Unified RoPE frequency allocator: standard and sigmoid variants."""

    def __init__(self, d: int, base: float = 10000.0):
        if d % 2 != 0:
            raise ValueError(f"d must be even, got {d}")
        self.d = int(d)
        self.N = self.d // 2
        self.base = float(base)

    def standard(self) -> torch.Tensor:
        """Standard geometric RoPE frequencies in float64."""
        i = torch.arange(self.N, dtype=torch.float64)
        freqs = self.base ** (-2.0 * i / float(self.d))
        return freqs

    def sigmoid(self, k: float, x0: float) -> torch.Tensor:
        """Sigmoid-RoPE frequencies in float64."""
        i = torch.arange(self.N, dtype=torch.float64)
        k_t = torch.tensor(float(k), dtype=torch.float64)
        x0_t = torch.tensor(float(x0), dtype=torch.float64)

        def sigma(z: torch.Tensor) -> torch.Tensor:
            return 1.0 / (1.0 + torch.exp(-z))

        raw = sigma(k_t * (i - x0_t))
        raw_min = sigma(k_t * (torch.tensor(0.0, dtype=torch.float64) - x0_t))
        raw_max = sigma(k_t * (torch.tensor(float(self.N - 1), dtype=torch.float64) - x0_t))
        denom = raw_max - raw_min
        if torch.abs(denom).item() < 1e-18:
            raise ValueError("Sigmoid normalization denominator is too small.")

        s_tilde = (raw - raw_min) / denom
        freqs = self.base ** (-s_tilde)
        return freqs

    def anchored_sigmoid(self, k: float, j0: float, anchor_factor: float = 20.0) -> torch.Tensor:
        """
        Anchored Sigmoid frequencies in float64.

        theta_i^anchored = theta_i^std * [1 + (alpha - 1) * sigma(k(i-j0))]
        where alpha=anchor_factor.
        """
        if anchor_factor <= 0:
            raise ValueError(f"anchor_factor must be > 0, got {anchor_factor}")
        i = torch.arange(self.N, dtype=torch.float64)
        k_t = torch.tensor(float(k), dtype=torch.float64)
        j0_t = torch.tensor(float(j0), dtype=torch.float64)
        alpha_t = torch.tensor(float(anchor_factor), dtype=torch.float64)
        std = self.standard()
        sig = 1.0 / (1.0 + torch.exp(-k_t * (i - j0_t)))
        gain = 1.0 + (alpha_t - 1.0) * sig
        return std * gain

    def sigmoid_analytical_v1(self, L: int) -> Tuple[torch.Tensor, float, float]:
        """Original heuristic (v1): x0=(d-2)/4, k=c1*ln(L/2pi)/d with c1=2ln19."""
        x0 = (self.d - 2.0) / 4.0
        c1 = 2.0 * math.log(19.0)
        k = c1 * math.log(float(L) / (2.0 * math.pi)) / float(self.d)
        return self.sigmoid(k=k, x0=x0), float(k), float(x0)

    @staticmethod
    def _k_formula_v2(L: int, d: int, form: str, params: Dict[str, float]) -> float:
        form = str(form).upper()
        if form == "A":
            c1 = float(params["c1"])
            return c1 * math.log(float(L)) / float(d)
        if form == "B":
            c1 = float(params["c1"])
            c2 = float(params["c2"])
            return c1 * math.log(float(L)) / float(d) + c2
        if form == "C":
            c1 = float(params["c1"])
            c2 = float(params["c2"])
            return c1 * (float(L) ** c2) / float(d)
        if form == "D":
            c1 = float(params["c1"])
            c2 = float(params["c2"])
            return c1 * math.log(float(L) / max(c2, 1e-6)) / float(d)
        if form == "E":
            c1 = float(params["c1"])
            c2 = float(params["c2"])
            return c1 * math.log(float(L)) / (float(d) ** c2)
        raise ValueError(f"Unknown k formula form: {form}")

    @staticmethod
    def _x0_formula_v2(L: int, d: int, form: str, params: Dict[str, float]) -> float:
        form = str(form).upper()
        if form == "A":
            c3 = float(params["c3"])
            return c3 * float(d)
        if form == "B":
            c3 = float(params["c3"])
            c4 = float(params["c4"])
            return c3 * float(d) + c4 * math.log(float(L))
        if form == "C":
            c3 = float(params["c3"])
            return c3 * (float(d) / 2.0 - 1.0)
        raise ValueError(f"Unknown x0 formula form: {form}")

    def sigmoid_analytical_v2(
        self,
        L: int,
        k_form: str = "A",
        k_params: Dict[str, float] | None = None,
        x0_form: str = "A",
        x0_params: Dict[str, float] | None = None,
    ) -> Tuple[torch.Tensor, float, float]:
        """
        Fitted heuristic (v2), configurable by formula form and parameters.
        Defaults are placeholders and should be overridden by phase-2 fitting results.
        """
        if k_params is None:
            k_params = {"c1": 1.5}
        if x0_params is None:
            x0_params = {"c3": 0.245}

        k = self._k_formula_v2(L=L, d=self.d, form=k_form, params=k_params)
        x0 = self._x0_formula_v2(L=L, d=self.d, form=x0_form, params=x0_params)
        x0 = max(0.0, min(float(self.N - 1), float(x0)))
        return self.sigmoid(k=float(k), x0=float(x0)), float(k), float(x0)

    def sigmoid_analytical(self, L: int, version: str = "v2", **kwargs) -> Tuple[torch.Tensor, float, float]:
        """
        Unified analytical entry.
        - version='v1': original heuristic
        - version='v2': fitted formula (recommended in phase-2)
        """
        if version == "v1":
            return self.sigmoid_analytical_v1(L=L)
        if version == "v2":
            return self.sigmoid_analytical_v2(L=L, **kwargs)
        raise ValueError(f"Unknown analytical version: {version}")

    def sigmoid_analytical_simple(self, L: int) -> Tuple[torch.Tensor, float, float]:
        """Simplified legacy heuristic: k ~= 6 ln(L)/d."""
        x0 = (self.d - 2.0) / 4.0
        k = 6.0 * math.log(float(L)) / float(self.d)
        return self.sigmoid(k=k, x0=x0), float(k), float(x0)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """
    Apply RoPE on x.

    Args:
        x: (batch, seq_len, num_heads, d)
        freqs: (N,) float64 frequency vector
        positions: (seq_len,) position indices
    Returns:
        Rotated tensor with same shape as x.
    """
    if x.ndim != 4:
        raise ValueError(f"x must be 4D (B,S,H,D), got shape={tuple(x.shape)}")
    bsz, seq_len, n_heads, d = x.shape
    if d % 2 != 0:
        raise ValueError(f"last dim d must be even, got {d}")
    n = d // 2
    if freqs.numel() != n:
        raise ValueError(f"freqs size mismatch: expected {n}, got {freqs.numel()}")
    if positions.numel() != seq_len:
        raise ValueError(f"positions size mismatch: expected {seq_len}, got {positions.numel()}")

    device = x.device
    angles = positions.to(device=device, dtype=torch.float32).unsqueeze(-1) * freqs.to(
        device=device, dtype=torch.float32
    ).unsqueeze(0)
    cos_angles = torch.cos(angles).view(1, seq_len, 1, n)
    sin_angles = torch.sin(angles).view(1, seq_len, 1, n)

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    x_rot_even = x_even * cos_angles - x_odd * sin_angles
    x_rot_odd = x_even * sin_angles + x_odd * cos_angles
    x_rot = torch.stack([x_rot_even, x_rot_odd], dim=-1).flatten(-2)
    return x_rot
