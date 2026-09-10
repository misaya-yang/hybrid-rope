"""Prefix-only P/C/U retention operators; no model or future-question access.

Queries are sampled *after* native Q normalization and *before* RoPE. Keys are
the original post-RoPE prefix keys. All arithmetic in the scorer is FP32, and
softmax includes protected keys before the fixed-budget selection is applied.
Only fixed-frequency, unit-amplitude native RoPE is supported in this version.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from numbers import Integral

import torch


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


@dataclass(frozen=True)
class NativeRope:
    inv_freq: torch.Tensor
    rotary_dim: int | None = None
    layout: str = "split_half"
    rotary_start: int = 0
    amplitude: float = 1.0

    def __post_init__(self):
        freq = torch.as_tensor(self.inv_freq).detach().clone()
        if freq.ndim != 1 or not freq.is_floating_point() or not torch.isfinite(freq).all():
            raise ValueError("inv_freq must be a finite one-dimensional floating tensor")
        dim = 2 * freq.numel() if self.rotary_dim is None else self.rotary_dim
        dim = _integer(dim, "rotary_dim")
        if dim != 2 * freq.numel():
            raise ValueError("rotary_dim must equal twice the number of frequencies")
        if self.layout not in ("split_half", "interleaved"):
            raise ValueError("unsupported native rotary layout")
        if not math.isfinite(self.amplitude) or self.amplitude != 1.0:
            raise ValueError("only native amplitude=1 is audited; do not silently rescale")
        _integer(self.rotary_start, "rotary_start")
        object.__setattr__(self, "inv_freq", freq)
        object.__setattr__(self, "rotary_dim", dim)

    def _apply(self, x, cosine, sine):
        if self.rotary_start + self.rotary_dim > x.shape[-1]:
            raise ValueError("native rotary slice exceeds head dimension")
        x = x.float()
        if self.rotary_dim == 0:
            return x
        start, stop = self.rotary_start, self.rotary_start + self.rotary_dim
        body = x[..., start:stop]
        if self.layout == "split_half":
            a, b = body.chunk(2, -1)
            rotated = torch.cat((a * cosine - b * sine, b * cosine + a * sine), -1)
        else:
            a, b = body[..., 0::2], body[..., 1::2]
            rotated = torch.stack((a * cosine - b * sine, b * cosine + a * sine), -1).flatten(-2)
        return torch.cat((x[..., :start], rotated, x[..., stop:]), -1)

    def apply(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Apply one shared native position per query, across all frequencies."""
        positions = torch.as_tensor(positions, device=x.device)
        if positions.is_floating_point() or positions.dtype == torch.bool:
            raise ValueError("positions must be integer logical token positions")
        if (positions < 0).any():
            raise ValueError("positions must be nonnegative")
        try:
            positions = torch.broadcast_to(positions, x.shape[:-1])
        except RuntimeError as exc:
            raise ValueError("positions must broadcast to query rows, not frequencies") from exc
        phase = positions.float().unsqueeze(-1) * self.inv_freq.to(x.device, torch.float32)
        return self._apply(x, phase.cos(), phase.sin())

    def average_phases(self, start: int, horizon: int, *, unit_modulus=False,
                       zero_modulus_tol=1e-7, device=None):
        """Exact finite-integer-horizon formula, evaluated in FP64.

        Unit control uses the midpoint rotation when |mean phase| <= 1e-7;
        its phase is otherwise undefined at zero. This is a documented control,
        not a sampled legal-position PM query. H=1 uses the native FP32 path.
        """
        start = _integer(start, "future_start")
        horizon = _integer(horizon, "horizon", 1)
        if zero_modulus_tol <= 0 or not math.isfinite(zero_modulus_tol):
            raise ValueError("zero_modulus_tol must be finite and positive")
        freq = self.inv_freq.to(device=device, dtype=torch.float64)
        if horizon == 1:
            phase = self.inv_freq.to(device=device, dtype=torch.float32) * float(start)
            return phase.cos(), phase.sin(), 0
        # Reducing frequency modulo 2*pi is exact on integer positions. sinc
        # supplies the removable limit at zero and at integer multiples of 2*pi.
        reduced = torch.remainder(freq + math.pi, 2 * math.pi) - math.pi
        ratio = torch.sinc(horizon * reduced / (2 * math.pi)) / torch.sinc(reduced / (2 * math.pi))
        center = reduced * (start + (horizon - 1) / 2)
        cosine, sine = ratio * center.cos(), ratio * center.sin()
        zero_count = 0
        if unit_modulus:
            modulus = torch.hypot(cosine, sine)
            zero = modulus <= zero_modulus_tol
            zero_count = int(zero.sum().item())
            safe = modulus.clamp_min(zero_modulus_tol)
            native_center = freq * (start + (horizon - 1) / 2)
            cosine = torch.where(zero, native_center.cos(), cosine / safe)
            sine = torch.where(zero, native_center.sin(), sine / safe)
        return cosine.float(), sine.float(), zero_count

    def average_apply(self, x, start, horizon, *, unit_modulus=False):
        c, s, _ = self.average_phases(start, horizon, unit_modulus=unit_modulus, device=x.device)
        return self._apply(x, c, s)


@dataclass(frozen=True)
class SamplingPlan:
    query_indices: torch.Tensor  # [Hq, M], CPU int64, original prefix indices
    future_positions: torch.Tensor  # [Hq, M], CPU int64, one position per row
    prefix_length: int
    horizon: int
    seed: int
    layer_idx: int
    query_start: int


def _seed(seed, layer_idx, head, component):
    raw = f"pm_keep_v1:{seed}:{layer_idx}:{head}:{component}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**63 - 1)


def make_sampling_plan(prefix_length, num_query_heads, samples_per_head=256,
                       horizon=512, seed=20260909, layer_idx=0, sink_tokens=4,
                       query_start=None):
    """Uniform empirical non-sink sampling with replacement, shared by P/C/U.

    CPU RNG streams are separately seeded by layer/head/component. No question,
    answer, labels, or future hidden states are accepted by this interface.
    """
    length = _integer(prefix_length, "prefix_length", 1)
    heads = _integer(num_query_heads, "num_query_heads", 1)
    count = _integer(samples_per_head, "samples_per_head", 1)
    horizon = _integer(horizon, "horizon", 1)
    sink_tokens = _integer(sink_tokens, "sink_tokens")
    seed, layer_idx = _integer(seed, "seed"), _integer(layer_idx, "layer_idx")
    low = sink_tokens if query_start is None else max(sink_tokens, _integer(query_start, "query_start"))
    if low >= length:
        raise ValueError("prefix has no non-sink query available for the empirical proxy")
    indices, positions = [], []
    for head in range(heads):
        gen = torch.Generator(device="cpu").manual_seed(_seed(seed, layer_idx, head, "query"))
        indices.append(torch.randint(low, length, (count,), generator=gen))
        gen.manual_seed(_seed(seed, layer_idx, head, "position"))
        positions.append(torch.randint(length, length + horizon, (count,), generator=gen))
    return SamplingPlan(torch.stack(indices), torch.stack(positions), length, horizon, seed, layer_idx, low)


def sample_prefix_queries(prefix_queries: torch.Tensor, plan: SamplingPlan):
    """Convenience reference gather; adapters may capture only these rows."""
    if prefix_queries.ndim != 3 or prefix_queries.shape[:2] != (plan.query_indices.shape[0], plan.prefix_length):
        raise ValueError("prefix_queries must have shape [Hq, prefix_length, D]")
    take = plan.query_indices.to(prefix_queries.device)
    return prefix_queries.gather(1, take[..., None].expand(-1, -1, prefix_queries.shape[-1]))


def _validate_qk(queries, keys):
    if queries.ndim != 3 or keys.ndim != 3:
        raise ValueError("queries/keys must have shapes [Hq,M,D] and [Hkv,T,D]")
    hq, count, dim = queries.shape
    hkv, length, kd = keys.shape
    if min(hq, count, dim, hkv, length) < 1 or kd != dim or hq % hkv:
        raise ValueError("nonempty matching dimensions and contiguous divisible GQA heads required")
    if queries.device != keys.device or not queries.is_floating_point() or not keys.is_floating_point():
        raise ValueError("queries and keys must be floating tensors on the same device")
    if not torch.isfinite(queries).all() or not torch.isfinite(keys).all():
        raise ValueError("nonfinite query or key")
    return hq, count, dim, hkv, length


@torch.no_grad()
def attention_mean(queries, keys, *, attention_scale=None, query_chunk_size=32,
                   key_chunk_size=2048):
    """Mean of per-query normalized prefix attention, then native GQA mean.

    Two key-tiled passes implement a stable global softmax. Temporary logits
    are at most [query_chunk_size,key_chunk_size], never [Hq,M,T]. Protected
    keys remain in every denominator. This function does not modify K or V.
    """
    hq, count, dim, hkv, length = _validate_qk(queries, keys)
    qm = _integer(query_chunk_size, "query_chunk_size", 1)
    kt = _integer(key_chunk_size, "key_chunk_size", 1)
    scale = dim**-0.5 if attention_scale is None else float(attention_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("attention_scale must be finite and positive")
    # Explicit FP32 inputs alone do not defeat an outer BF16 autocast context.
    with torch.autocast(device_type=keys.device.type, enabled=False):
        return _attention_mean_tiled(queries, keys, hq, count, hkv, length, qm, kt, scale)


def _attention_mean_tiled(queries, keys, hq, count, hkv, length, qm, kt, scale):
    scores = torch.zeros((hkv, length), dtype=torch.float32, device=keys.device)
    group = hq // hkv
    for head in range(hq):
        g = head // group
        for qstart in range(0, count, qm):
            q = queries[head, qstart:qstart + qm].float()
            maximum = torch.full((q.shape[0],), -torch.inf, device=q.device)
            denominator = torch.zeros_like(maximum)
            for kstart in range(0, length, kt):
                logits = (q @ keys[g, kstart:kstart + kt].float().T) * scale
                new_maximum = torch.maximum(maximum, logits.amax(-1))
                denominator = denominator * (maximum - new_maximum).exp() + (logits - new_maximum[:, None]).exp().sum(-1)
                maximum = new_maximum
            for kstart in range(0, length, kt):
                logits = (q @ keys[g, kstart:kstart + kt].float().T) * scale
                probabilities = (logits - maximum[:, None]).exp() / denominator[:, None]
                scores[g, kstart:kstart + kt] += probabilities.sum(0) / (count * group)
    if not torch.isfinite(scores).all():
        raise FloatingPointError("nonfinite FP32 attention scorer output")
    return scores


@dataclass(frozen=True)
class ScoreResult:
    scores: torch.Tensor
    metrics: dict


@torch.no_grad()
def pm_keep_scores(q_samples, k_post, rope: NativeRope, *, arm="pm",
                   future_start=None, horizon=512, future_positions=None,
                   attention_scale=None, query_chunk_size=32, key_chunk_size=2048):
    """P/C/U matched scorer. Pass the same q_samples object to all three arms."""
    hq, count, dim, hkv, length = _validate_qk(q_samples, k_post)
    start = length if future_start is None else _integer(future_start, "future_start")
    if start < length:
        raise ValueError("future_start must follow the complete visible prefix")
    horizon = _integer(horizon, "horizon", 1)
    if arm not in ("pm", "collapse", "unit"):
        raise ValueError("arm must be pm, collapse, or unit")
    zero_count = 0
    if arm == "pm":
        if future_positions is None:
            raise ValueError("PM requires explicit positions from the shared sampling plan")
        pos = torch.as_tensor(future_positions, device=q_samples.device)
        if pos.shape != (hq, count) or pos.is_floating_point() or pos.dtype == torch.bool:
            raise ValueError("future_positions must be integer [Hq,M], not per-frequency samples")
        if (pos < start).any() or (pos >= start + horizon).any():
            raise ValueError("sampled position outside the declared finite future horizon")
        query = rope.apply(q_samples, pos)
    else:
        c, s, zero_count = rope.average_phases(start, horizon, unit_modulus=arm == "unit", device=q_samples.device)
        query = rope._apply(q_samples, c, s)
    scores = attention_mean(query, k_post, attention_scale=attention_scale,
                            query_chunk_size=query_chunk_size, key_chunk_size=key_chunk_size)
    c, s, _ = rope.average_phases(start, horizon, device=q_samples.device)
    modulus = torch.hypot(c, s)
    norms_before = q_samples.float().norm(dim=-1)
    norms_after = query.norm(dim=-1)
    metrics = dict(arm=arm, query_heads=hq, kv_heads=hkv, samples_per_head=count,
        prefix_length=length, future_start=start, horizon=horizon,
        score_sums=scores.sum(-1).tolist(), scores_finite=True,
        score_min=float(scores.min().item()), score_max=float(scores.max().item()),
        query_norm_mean_before=float(norms_before.mean().item()),
        query_norm_mean_after=float(norms_after.mean().item()),
        average_pair_modulus=modulus.tolist(), unit_zero_pair_count=zero_count,
        unit_zero_fallback="midpoint_rotation_when_modulus_le_1e-7",
        max_logits_tile_elements=min(count, query_chunk_size) * min(length, key_chunk_size),
        softmax_dtype="float32", protected_keys_in_denominator=True,
        value_norm_weighting=False)
    return ScoreResult(scores, metrics)


def floor_keep_budget(prefix_length, numerator=1, denominator=4):
    """Exact integer floor(T/4) by default; protection is inside this budget."""
    length = _integer(prefix_length, "prefix_length", 1)
    numerator = _integer(numerator, "numerator", 1)
    denominator = _integer(denominator, "denominator", 1)
    if numerator > denominator:
        raise ValueError("keep fraction cannot exceed one")
    return length * numerator // denominator


@torch.no_grad()
def select_fixed_budget(scores, total_budget, *, sink_tokens=4, recent_tokens=256):
    """Return sorted unique original indices [Hkv,B], ties by original index.

    The union of sinks/recent keys consumes slots inside B. An undersized
    budget raises instead of silently exceeding the requested memory budget.
    """
    if scores.ndim != 2 or min(scores.shape) < 1 or not torch.isfinite(scores).all():
        raise ValueError("scores must be finite [Hkv,T]")
    budget = _integer(total_budget, "total_budget")
    sinks = _integer(sink_tokens, "sink_tokens")
    recent = _integer(recent_tokens, "recent_tokens")
    heads, length = scores.shape
    if budget > length:
        raise ValueError("total_budget exceeds prefix length")
    ids = torch.arange(length, device=scores.device)
    protected = (ids < sinks) | (ids >= max(0, length - recent))
    fixed, candidates = ids[protected], ids[~protected]
    if fixed.numel() > budget:
        raise ValueError(f"protected union needs {fixed.numel()} slots, total budget is {budget}")
    extra = budget - fixed.numel()
    if extra:
        rank = torch.argsort(scores[:, candidates], dim=-1, descending=True, stable=True)
        chosen = candidates[rank[:, :extra]]
    else:
        chosen = torch.empty((heads, 0), dtype=torch.long, device=scores.device)
    return torch.cat((fixed[None].expand(heads, -1), chosen), -1).sort(-1).values
