"""CPU reference for a 16-DOF MrPro-initialized frequency calibrator.

This is a reference/contract, not a GPU runner.  It exposes the exact finite
allocation convention and a differentiable Qwen-compatible rotary forward.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def helmert_zero_sum(rows: int, *, dtype=torch.float64) -> torch.Tensor:
    """Return an orthonormal basis for {x in R**rows: sum(x)=0}."""
    if rows < 2:
        raise ValueError("rows must be at least two")
    out = torch.zeros(rows, rows - 1, dtype=dtype)
    for column in range(rows - 1):
        count = column + 1
        out[:count, column] = 1.0 / math.sqrt(count * (count + 1))
        out[count, column] = -count / math.sqrt(count * (count + 1))
    return out


def mrpro_increments(width: int, *, dtype=torch.float64) -> torch.Tensor:
    """MrRoPE-Pro radix increments epsilon_i, i=1..width."""
    i = torch.arange(1, width + 1, dtype=dtype)
    return 2.0 * i / (width * (width + 1))


class MrProAllocation16(nn.Module):
    """Strictly ordered, fixed-boundary allocation for Qwen K=64.

    Slots are zero based.  ``low=23, high=40`` gives 17 positive transition
    increments and therefore 16 effective degrees of freedom after their sum
    is fixed to one.  ``eta`` uses an orthonormal zero-sum (Helmert) basis, so
    all 16 stored scalars are identifiable.  eta=0 is exactly MrPro in float64.
    """

    def __init__(
        self,
        *,
        pair_count: int = 64,
        head_dim: int = 128,
        rope_theta: float = 1_000_000.0,
        scale: float = 4.0,
        low: int = 23,
        high: int = 40,
    ) -> None:
        super().__init__()
        if pair_count * 2 != head_dim or not (0 <= low < high < pair_count):
            raise ValueError("invalid Qwen rotary geometry")
        self.pair_count, self.head_dim = pair_count, head_dim
        self.rope_theta, self.scale = float(rope_theta), float(scale)
        self.low, self.high = int(low), int(high)
        width = high - low
        reference = mrpro_increments(width)
        self.eta = nn.Parameter(torch.zeros(width - 1, dtype=torch.float64))
        self.register_buffer("log_reference_increments", reference.log())
        self.register_buffer("basis", helmert_zero_sum(width))
        idx = torch.arange(pair_count, dtype=torch.float64)
        self.register_buffer("native_inv_freq", self.rope_theta ** (-2.0 * idx / head_dim))

    def increments(self) -> torch.Tensor:
        logits = self.log_reference_increments + self.basis @ self.eta
        return torch.softmax(logits, dim=0)

    def exponents(self) -> torch.Tensor:
        eps = self.increments()
        cumulative = torch.cumsum(eps, dim=0)
        before = torch.zeros(self.low + 1, dtype=eps.dtype, device=eps.device)
        after = torch.ones(self.pair_count - self.high - 1, dtype=eps.dtype, device=eps.device)
        # cumulative[q-1] belongs to slot low+q, including slot high at one.
        return torch.cat((before, cumulative, after))

    def inv_freq(self) -> torch.Tensor:
        return self.native_inv_freq * torch.exp(-math.log(self.scale) * self.exponents())


class DifferentiableQwenRotaryEmbedding(nn.Module):
    """Drop-in shared rotary module without HF Qwen2's ``@torch.no_grad``.

    Phase arithmetic stays FP32 with autocast disabled, matching Qwen2.  The
    returned cos/sin use the hidden-state dtype.  Dynamic-RoPE mutation is
    intentionally absent: the allocation itself owns the active table.
    """

    def __init__(self, allocation: MrProAllocation16, attention_scaling: float = 1.0):
        super().__init__()
        self.allocation = allocation
        self.attention_scaling = float(attention_scaling)

    @property
    def inv_freq(self) -> torch.Tensor:
        return self.allocation.inv_freq()

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        inv = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        pos = position_ids[:, None, :].float()
        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            phase = (inv.float() @ pos.float()).transpose(1, 2)
            emb = torch.cat((phase, phase), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def self_check() -> None:
    allocation = MrProAllocation16()
    eps = allocation.increments()
    expected = mrpro_increments(17)
    torch.testing.assert_close(eps, expected, rtol=0, atol=2e-16)
    exponent = allocation.exponents()
    assert exponent.shape == (64,) and allocation.eta.numel() == 16
    assert torch.equal(exponent[:24], torch.zeros(24, dtype=torch.float64))
    torch.testing.assert_close(exponent[40:], torch.ones(24, dtype=torch.float64), rtol=0, atol=2e-16)
    table = allocation.inv_freq()
    assert bool(torch.all(table[:-1] > table[1:]))

    # The actual rotary path, including float32 casts, must reach every eta.
    rotary = DifferentiableQwenRotaryEmbedding(allocation)
    x = torch.zeros(1, 8, 128, dtype=torch.float32)
    positions = torch.tensor([[0, 1, 17, 257, 4096, 32767, 65535, 131071]])
    cos, sin = rotary(x, positions)
    loss = (cos.double() * torch.linspace(0.1, 1.0, cos.numel()).reshape_as(cos)).sum()
    loss += (sin.double() * torch.linspace(1.0, 0.1, sin.numel()).reshape_as(sin)).sum()
    loss.backward()
    assert allocation.eta.grad is not None
    assert bool(torch.isfinite(allocation.eta.grad).all())
    assert bool((allocation.eta.grad.abs() > 0).all())

    # Float64 schedule path agrees with finite differences/autograd.
    probe = MrProAllocation16()
    assert torch.autograd.gradcheck(lambda z: torch.softmax(probe.log_reference_increments + probe.basis @ z, 0),
                                    (probe.eta,), eps=1e-6, atol=1e-6, rtol=1e-4)


if __name__ == "__main__":
    self_check()
    print("sol16 frequency calibration reference: PASS")
