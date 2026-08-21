#!/usr/bin/env python3
"""R0': stream attention-distance occupancy, then test companding numerics.

This script deliberately calls the measured object attention-distance
occupancy rather than frequency demand. Mapping a distance histogram to a
frequency-density target is a modelling assumption, so the analysis reports
both an endpoint-normalized log-distance map and the physical one-radian map.

No parameter is updated. Collection stores only accumulated distance
histograms, never attention matrices. Analysis can therefore change the
lambda mixture or diagnostics without rerunning the model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_tokens(path: Path) -> torch.Tensor:
    if path.suffix == ".npy":
        payload = torch.from_numpy(np.load(path).copy())
    else:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict):
        for key in ("input_ids", "tokens", "ids"):
            if key in payload:
                payload = payload[key]
                break
    if not isinstance(payload, torch.Tensor) or payload.ndim not in (1, 2):
        raise ValueError("token artifact must be a 1D/2D tensor or contain input_ids/tokens/ids")
    if payload.numel() == 0 or payload.dtype not in (
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint16,
    ):
        raise ValueError("token artifact must contain non-empty integer ids")
    return payload.long()


def window_batch(
    tokens: torch.Tensor,
    length: int,
    windows: int,
) -> tuple[list[torch.Tensor], list[dict[str, int]]]:
    if windows < 1 or length < 8:
        raise ValueError("windows must be positive and length must be at least 8")
    output: list[torch.Tensor] = []
    identity: list[dict[str, int]] = []
    if tokens.ndim == 1:
        if len(tokens) < length:
            raise ValueError("1D token tensor is shorter than requested length")
        stop = len(tokens) - length
        starts = np.linspace(0, stop, windows, dtype=np.int64)
        for start in starts:
            output.append(tokens[int(start) : int(start) + length].unsqueeze(0))
            identity.append({"row": -1, "start": int(start)})
        return output, identity

    rows, width = tokens.shape
    if width < length:
        raise ValueError("2D token rows are shorter than requested length")
    selected = np.linspace(0, rows - 1, min(windows, rows), dtype=np.int64)
    for row in selected:
        start = 0 if width == length else int((int(row) * 104729) % (width - length + 1))
        output.append(tokens[int(row), start : start + length].unsqueeze(0))
        identity.append({"row": int(row), "start": start})
    return output, identity


def query_positions(length: int, count: int) -> np.ndarray:
    if count < 1 or count > length:
        raise ValueError("query count must lie in [1, length]")
    return np.unique(
        np.linspace(length // 2, length - 1, count, dtype=np.int64)
    )


def pack_windows(windows: list[torch.Tensor], batch_size: int) -> list[torch.Tensor]:
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    return [
        torch.cat(windows[start : start + batch_size], dim=0)
        for start in range(0, len(windows), batch_size)
    ]


class ExactDistanceAccumulator:
    def __init__(self, layers: int, heads: int, length: int, head_dim: int) -> None:
        self.mass = np.zeros((layers, heads, length), dtype=np.float64)
        self.opportunities = np.zeros(length, dtype=np.float64)
        self.window_global_mass: list[np.ndarray] = []
        self._batch_window_mass: np.ndarray | None = None
        self.q_pair_l2_sum = np.zeros((layers, heads, head_dim // 2), dtype=np.float64)
        self.k_pair_l2_sum = np.zeros_like(self.q_pair_l2_sum)
        self.pair_norm_count = np.zeros(layers, dtype=np.int64)

    def begin_window(self, batch: int) -> None:
        self._batch_window_mass = np.zeros((batch, self.mass.shape[-1]), dtype=np.float64)

    def end_window(self) -> None:
        if self._batch_window_mass is None:
            raise RuntimeError("begin_window must precede end_window")
        self.window_global_mass.extend(self._batch_window_mass)
        self._batch_window_mass = None

    def add_opportunities(self, positions: np.ndarray, batch: int) -> None:
        for position in positions:
            self.opportunities[: int(position) + 1] += batch

    def add_pair_norms(self, layer: int, query: torch.Tensor, key: torch.Tensor) -> None:
        if query.shape != key.shape or query.ndim != 4 or query.shape[-1] % 2:
            raise ValueError("query/key must share shape [B,H,L,D] with even D")
        batch, heads, length, head_dim = query.shape
        if heads != self.q_pair_l2_sum.shape[1] or head_dim // 2 != self.q_pair_l2_sum.shape[2]:
            raise ValueError("query/key shape does not match pair-norm accumulator")
        q_norm = torch.linalg.vector_norm(query.float().reshape(batch, heads, length, -1, 2), dim=-1)
        k_norm = torch.linalg.vector_norm(key.float().reshape(batch, heads, length, -1, 2), dim=-1)
        self.q_pair_l2_sum[layer] += q_norm.sum(dim=(0, 2)).cpu().numpy()
        self.k_pair_l2_sum[layer] += k_norm.sum(dim=(0, 2)).cpu().numpy()
        self.pair_norm_count[layer] += batch * length

    def add(
        self,
        layer: int,
        probability: torch.Tensor,
        positions: np.ndarray,
    ) -> None:
        if probability.ndim != 4:
            raise ValueError("probability must have shape [B,H,Q,K]")
        batch, heads, queries, keys = probability.shape
        if heads != self.mass.shape[1] or queries != len(positions):
            raise ValueError("attention shape does not match registered heads/queries")
        qpos = torch.as_tensor(positions, device=probability.device)
        kpos = torch.arange(keys, device=probability.device)
        distance = qpos[:, None] - kpos[None, :]
        valid = distance >= 0
        index = distance.clamp(min=0, max=self.mass.shape[-1] - 1).long()
        values = probability.float() * valid[None, None]
        histogram = torch.zeros(
            (heads, self.mass.shape[-1]),
            dtype=torch.float32,
            device=probability.device,
        )
        expanded = index[None, None].expand(batch, heads, -1, -1)
        histogram.scatter_add_(
            1,
            expanded.permute(1, 0, 2, 3).reshape(heads, -1),
            values.permute(1, 0, 2, 3).reshape(heads, -1),
        )
        self.mass[layer] += histogram.cpu().numpy()
        if self._batch_window_mass is None or len(self._batch_window_mass) != batch:
            raise RuntimeError("attention batch does not match begin_window")
        per_sequence = torch.zeros(
            (batch, self.mass.shape[-1]),
            dtype=torch.float32,
            device=probability.device,
        )
        per_sequence.scatter_add_(
            1,
            index[None].expand(batch, -1, -1).reshape(batch, -1),
            values.sum(dim=1).reshape(batch, -1),
        )
        self._batch_window_mass += per_sequence.cpu().numpy()


def _project_state(path: Path, *, mmap: bool = False) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=mmap)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if not isinstance(state, dict):
        raise ValueError("project checkpoint does not contain a state dict")
    return {
        (key[len("_orig_mod.") :] if key.startswith("_orig_mod.") else key): value
        for key, value in state.items()
    }


def collect_project_gpt(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    from scripts.core_text_phases.run_evq_sweep import (
        GPT,
        TIER_CONFIGS,
        apply_rope,
    )

    device = choose_device(args.device)
    tokens = load_tokens(args.tokens)
    windows_raw, windows = window_batch(tokens, args.length, args.windows)
    batches = pack_windows(windows_raw, args.batch_size)
    positions = query_positions(args.length, args.queries)
    inv_freq = torch.from_numpy(np.load(args.inv_freq)).float()
    config = dict(TIER_CONFIGS[args.tier])
    config["seq_len"] = args.length
    config["max_position_embeddings"] = args.length
    with torch.device("meta"):
        model = GPT(config, inv_freq.to("meta"))
    loaded = model.load_state_dict(
        _project_state(args.checkpoint, mmap=True),
        strict=True,
        assign=True,
    )
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError(f"state-dict mismatch: {loaded}")
    model.extend_rope(args.length)
    model = model.to(device).eval().requires_grad_(False)
    layers = len(model.blocks)
    heads = model.blocks[0].attn.nh
    accumulator = ExactDistanceAccumulator(layers, heads, args.length, model.blocks[0].attn.hd)
    handles = []

    def make_hook(layer: int):
        def hook(module, hook_args):
            hidden = hook_args[0]
            batch, length, _ = hidden.shape
            qkv = module.qkv(hidden).view(
                batch, length, 3, module.nh, module.hd
            ).permute(2, 0, 3, 1, 4)
            query, key = qkv[0], qkv[1]
            cosine, sine = module.rope(length)
            query = apply_rope(query, cosine[None, None], sine[None, None])
            key = apply_rope(key, cosine[None, None], sine[None, None])
            accumulator.add_pair_norms(layer, query, key)
            selected = query[:, :, positions]
            score = torch.matmul(selected, key.transpose(-1, -2)) / math.sqrt(module.hd)
            qpos = torch.as_tensor(positions, device=score.device)
            kpos = torch.arange(length, device=score.device)
            score.masked_fill_(kpos[None, :] > qpos[:, None], -torch.inf)
            probability = torch.softmax(score, dim=-1, dtype=torch.float32)
            accumulator.add(layer, probability, positions)
        return hook

    for layer, block in enumerate(model.blocks):
        handles.append(block.attn.register_forward_pre_hook(make_hook(layer)))

    started = time.perf_counter()
    with torch.inference_mode():
        for batch in batches:
            accumulator.begin_window(len(batch))
            accumulator.add_opportunities(positions, len(batch))
            hidden = model.emb(batch.to(device))
            for block in model.blocks:
                hidden = block(hidden)
            accumulator.end_window()
    for handle in handles:
        handle.remove()
    expected_mass = sum(len(batch) for batch in batches) * len(positions)
    mass_error = float(np.max(np.abs(accumulator.mass.sum(axis=-1) - expected_mass)))
    if mass_error > 2e-4 * max(expected_mass, 1):
        raise AssertionError(f"attention mass conservation failed: {mass_error}")

    metadata = {
        "backend": "project_gpt",
        "tier": args.tier,
        "device": str(device),
        "length": args.length,
        "windows_requested": args.windows,
        "windows_realized": len(windows_raw),
        "sequences_realized": len(windows_raw),
        "batches_realized": len(batches),
        "batch_size_requested": args.batch_size,
        "window_identity": windows,
        "query_positions": positions.tolist(),
        "layers": layers,
        "heads": heads,
        "head_dim": model.blocks[0].attn.hd,
        "base": args.base,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "inv_freq": str(args.inv_freq.resolve()),
        "inv_freq_sha256": sha256(args.inv_freq),
        "tokens": str(args.tokens.resolve()),
        "tokens_sha256": sha256(args.tokens),
        "seconds": time.perf_counter() - started,
        "attention_mass_conservation_max_abs": mass_error,
        "training_or_parameter_updates": False,
    }
    arrays = {
        "mass": accumulator.mass,
        "opportunities": accumulator.opportunities,
        "window_global_mass": np.stack(accumulator.window_global_mass),
        "inv_freq": inv_freq.double().numpy(),
        "q_pair_l2_mean": accumulator.q_pair_l2_sum / accumulator.pair_norm_count[:, None, None],
        "k_pair_l2_mean": accumulator.k_pair_l2_sum / accumulator.pair_norm_count[:, None, None],
    }
    return metadata, arrays


def collect_native_151m(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    from experiments.native_rope_evq_150m.model import GPT, apply_rope

    device = choose_device(args.device)
    tokens = load_tokens(args.tokens)
    windows_raw, windows = window_batch(tokens, args.length, args.windows)
    batches = pack_windows(windows_raw, args.batch_size)
    positions = query_positions(args.length, args.queries)
    inv_freq = torch.from_numpy(np.load(args.inv_freq)).float()
    config = {
        "vocab_size": 50_304,
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "head_dim": 64,
        "intermediate_size": 3_072,
        "max_position_embeddings": args.length,
        "seq_len": args.length,
    }
    with torch.device("meta"):
        model = GPT(config, inv_freq.to("meta"))
    loaded = model.load_state_dict(
        _project_state(args.checkpoint, mmap=True),
        strict=True,
        assign=True,
    )
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError(f"state-dict mismatch: {loaded}")
    model.extend_rope(args.length)
    model = model.to(device).eval().requires_grad_(False)
    accumulator = ExactDistanceAccumulator(12, 12, args.length, 64)
    handles = []

    def make_hook(layer: int):
        def hook(module, hook_args):
            hidden = hook_args[0]
            batch, length, _ = hidden.shape
            qkv = module.qkv(hidden).view(
                batch, length, 3, module.num_heads, module.head_dim
            ).permute(2, 0, 3, 1, 4)
            query, key = qkv[0], qkv[1]
            cosine, sine = module.rope(length)
            query = apply_rope(query, cosine[None, None], sine[None, None])
            key = apply_rope(key, cosine[None, None], sine[None, None])
            accumulator.add_pair_norms(layer, query, key)
            selected = query[:, :, positions]
            score = torch.matmul(selected, key.transpose(-1, -2)) / math.sqrt(module.head_dim)
            qpos = torch.as_tensor(positions, device=score.device)
            kpos = torch.arange(length, device=score.device)
            score.masked_fill_(kpos[None, :] > qpos[:, None], -torch.inf)
            accumulator.add(
                layer,
                torch.softmax(score, dim=-1, dtype=torch.float32),
                positions,
            )
        return hook

    for layer, block in enumerate(model.blocks):
        handles.append(
            block.attention.register_forward_pre_hook(make_hook(layer))
        )
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in batches:
            accumulator.begin_window(len(batch))
            accumulator.add_opportunities(positions, len(batch))
            hidden = model.embedding(batch.to(device))
            for block in model.blocks:
                hidden = block(hidden)
            accumulator.end_window()
    for handle in handles:
        handle.remove()
    expected_mass = sum(len(batch) for batch in batches) * len(positions)
    mass_error = float(np.max(np.abs(accumulator.mass.sum(axis=-1) - expected_mass)))
    if mass_error > 2e-4 * max(expected_mass, 1):
        raise AssertionError(f"attention mass conservation failed: {mass_error}")
    metadata = {
        "backend": "native_151m",
        "device": str(device),
        "length": args.length,
        "windows_requested": args.windows,
        "windows_realized": len(windows_raw),
        "sequences_realized": len(windows_raw),
        "batches_realized": len(batches),
        "batch_size_requested": args.batch_size,
        "window_identity": windows,
        "query_positions": positions.tolist(),
        "layers": 12,
        "heads": 12,
        "head_dim": 64,
        "base": args.base,
        "seed": args.seed,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "inv_freq": str(args.inv_freq.resolve()),
        "inv_freq_sha256": sha256(args.inv_freq),
        "tokens": str(args.tokens.resolve()),
        "tokens_sha256": sha256(args.tokens),
        "seconds": time.perf_counter() - started,
        "attention_mass_conservation_max_abs": mass_error,
        "training_or_parameter_updates": False,
    }
    arrays = {
        "mass": accumulator.mass,
        "opportunities": accumulator.opportunities,
        "window_global_mass": np.stack(accumulator.window_global_mass),
        "inv_freq": inv_freq.double().numpy(),
        "q_pair_l2_mean": accumulator.q_pair_l2_sum / accumulator.pair_norm_count[:, None, None],
        "k_pair_l2_mean": accumulator.k_pair_l2_sum / accumulator.pair_norm_count[:, None, None],
    }
    return metadata, arrays


def collect_hf_olmo2(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    from transformers import AutoModelForCausalLM
    from transformers.models.olmo2.modeling_olmo2 import (
        apply_rotary_pos_emb,
        repeat_kv,
    )

    device = choose_device(args.device)
    if device.type != "cuda" and not args.allow_slow_device:
        raise RuntimeError("HF OLMo2 collection requires CUDA unless --allow-slow-device is set")
    tokens = load_tokens(args.tokens)
    windows_raw, windows = window_batch(tokens, args.length, args.windows)
    batches = pack_windows(windows_raw, args.batch_size)
    positions = query_positions(args.length, args.queries)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    causal_lm = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=dtype,
        attn_implementation="sdpa",
    ).to(device).eval().requires_grad_(False)
    backbone = causal_lm.model
    modules = [layer.self_attn for layer in backbone.layers]
    layers = len(modules)
    heads = modules[0].config.num_attention_heads
    accumulator = ExactDistanceAccumulator(layers, heads, args.length, modules[0].head_dim)
    handles = []

    def make_hook(layer: int):
        def hook(module, hook_args, kwargs):
            hidden = kwargs["hidden_states"]
            cosine, sine = kwargs["position_embeddings"]
            batch, length, _ = hidden.shape
            query = module.q_norm(module.q_proj(hidden))
            key = module.k_norm(module.k_proj(hidden))
            query = query.view(batch, length, -1, module.head_dim).transpose(1, 2)
            key = key.view(batch, length, -1, module.head_dim).transpose(1, 2)
            query, key = apply_rotary_pos_emb(query, key, cosine, sine)
            key = repeat_kv(key, module.num_key_value_groups)
            accumulator.add_pair_norms(layer, query, key)
            selected = query[:, :, positions]
            score = torch.matmul(selected, key.transpose(-1, -2)) * module.scaling
            qpos = torch.as_tensor(positions, device=score.device)
            kpos = torch.arange(length, device=score.device)
            score.masked_fill_(kpos[None, :] > qpos[:, None], -torch.inf)
            probability = torch.softmax(score, dim=-1, dtype=torch.float32)
            accumulator.add(layer, probability, positions)
        return hook

    for layer, module in enumerate(modules):
        handles.append(
            module.register_forward_pre_hook(make_hook(layer), with_kwargs=True)
        )

    started = time.perf_counter()
    with torch.inference_mode():
        for batch in batches:
            accumulator.begin_window(len(batch))
            accumulator.add_opportunities(positions, len(batch))
            backbone(input_ids=batch.to(device), use_cache=False, return_dict=True)
            accumulator.end_window()
    for handle in handles:
        handle.remove()
    expected_mass = sum(len(batch) for batch in batches) * len(positions)
    mass_error = float(np.max(np.abs(accumulator.mass.sum(axis=-1) - expected_mass)))
    if mass_error > 2e-4 * max(expected_mass, 1):
        raise AssertionError(f"attention mass conservation failed: {mass_error}")

    realized = backbone.rotary_emb.inv_freq.detach().float().cpu().numpy()
    rope_parameters = getattr(causal_lm.config, "rope_parameters", {}) or {}
    base = getattr(causal_lm.config, "rope_theta", None)
    if base is None:
        base = rope_parameters.get("rope_theta")
    if base is None:
        exponent = 2.0 * (len(realized) - 1) / modules[0].head_dim
        base = math.exp(-math.log(float(realized[-1])) / exponent)
    weight_files = sorted(args.model.glob("*.safetensors"))
    metadata = {
        "backend": "hf_olmo2",
        "device": str(device),
        "length": args.length,
        "windows_requested": args.windows,
        "windows_realized": len(windows_raw),
        "sequences_realized": len(windows_raw),
        "batches_realized": len(batches),
        "batch_size_requested": args.batch_size,
        "window_identity": windows,
        "query_positions": positions.tolist(),
        "layers": layers,
        "heads": heads,
        "head_dim": modules[0].head_dim,
        "base": float(base),
        "model": str(args.model.resolve()),
        "model_revision": args.model_revision,
        "weight_files": [
            {"path": file.name, "bytes": file.stat().st_size, "sha256": sha256(file)}
            for file in weight_files
        ],
        "tokens": str(args.tokens.resolve()),
        "tokens_sha256": sha256(args.tokens),
        "seconds": time.perf_counter() - started,
        "attention_mass_conservation_max_abs": mass_error,
        "training_or_parameter_updates": False,
    }
    arrays = {
        "mass": accumulator.mass,
        "opportunities": accumulator.opportunities,
        "window_global_mass": np.stack(accumulator.window_global_mass),
        "inv_freq": realized.astype(np.float64),
        "q_pair_l2_mean": accumulator.q_pair_l2_sum / accumulator.pair_norm_count[:, None, None],
        "k_pair_l2_mean": accumulator.k_pair_l2_sum / accumulator.pair_norm_count[:, None, None],
    }
    return metadata, arrays


def write_collection(path: Path, metadata: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        **arrays,
    )


def load_collection(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata"]))
        arrays = {key: np.asarray(payload[key]) for key in payload.files if key != "metadata"}
    return metadata, arrays


def distance_phi(distance: np.ndarray, length: int, base: float, mapping: str) -> np.ndarray:
    distance = np.asarray(distance, dtype=np.float64)
    if mapping == "endpoint_log":
        return np.log1p(distance) / math.log(length)
    if mapping == "physical_1rad":
        result = np.zeros_like(distance)
        positive = distance > 0
        result[positive] = np.log(distance[positive]) / math.log(base)
        return np.clip(result, 0.0, 1.0)
    raise ValueError(f"unknown mapping: {mapping}")


def density_from_distance(
    distance_mass: np.ndarray,
    length: int,
    base: float,
    bins: int,
    mapping: str,
    include_self: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mass = np.asarray(distance_mass, dtype=np.float64).copy()
    if len(mass) != length or np.any(mass < 0):
        raise ValueError("distance mass must be a nonnegative length-L vector")
    if not include_self:
        mass[0] = 0.0
    if mass.sum() <= 0:
        raise FloatingPointError("distance mass is zero")
    phi = distance_phi(np.arange(length), length, base, mapping)
    edges = np.linspace(0.0, 1.0, bins + 1)
    probability, _ = np.histogram(phi, bins=edges, weights=mass)
    probability = probability.astype(np.float64)
    probability /= probability.sum()
    width = 1.0 / bins
    density = probability / width
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, density, probability


def headroom(density: np.ndarray) -> tuple[float, float]:
    density = np.asarray(density, dtype=np.float64)
    if np.any(density < 0):
        raise ValueError("density must be nonnegative")
    width = 1.0 / len(density)
    mass = float(np.sum(density) * width)
    if not math.isclose(mass, 1.0, abs_tol=1e-10):
        raise ValueError(f"density does not integrate to one: {mass}")
    z = float(np.sum(np.cbrt(density)) * width)
    d_star = z**3
    value = 1.0 - d_star
    if value < -1e-10 or value >= 1.0 + 1e-10:
        raise FloatingPointError(f"invalid headroom: {value}")
    return max(0.0, value), d_star


def normalize_density(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    width = 1.0 / len(values)
    integral = float(values.sum() * width)
    if integral <= 0:
        raise FloatingPointError("density normalization integral is zero")
    return values / integral


def companded_density(demand: np.ndarray, lam: float) -> np.ndarray:
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lambda must lie in [0,1]")
    mixed = (1.0 - lam) * np.asarray(demand, dtype=np.float64) + lam
    return normalize_density(np.cbrt(mixed))


def cosh_density(centers: np.ndarray, tau: float) -> np.ndarray:
    if abs(tau) < 1e-12:
        return np.ones_like(centers)
    return normalize_density(
        tau * np.cosh(tau * (1.0 - centers)) / math.sinh(tau)
    )


def distortion(demand: np.ndarray, allocation: np.ndarray) -> float:
    demand = np.asarray(demand, dtype=np.float64)
    allocation = np.asarray(allocation, dtype=np.float64)
    if np.any((allocation <= 0) & (demand > 0)):
        return float("inf")
    terms = np.zeros_like(demand)
    positive = allocation > 0
    terms[positive] = demand[positive] / np.square(allocation[positive])
    return float(np.mean(terms))


def midpoint_quantiles(density: np.ndarray, count: int) -> np.ndarray:
    if count < 1:
        raise ValueError("quantile count must be positive")
    density = normalize_density(density)
    edges = np.linspace(0.0, 1.0, len(density) + 1)
    cell_mass = density / len(density)
    cdf = np.concatenate(([0.0], np.cumsum(cell_mass)))
    target = (np.arange(count, dtype=np.float64) + 0.5) / count
    quantiles = np.empty(count, dtype=np.float64)
    for index, value in enumerate(target):
        cell = min(int(np.searchsorted(cdf, value, side="right") - 1), len(density) - 1)
        local = (value - cdf[cell]) / max(cell_mass[cell], 1e-300)
        quantiles[index] = edges[cell] + local * (edges[cell + 1] - edges[cell])
    if np.any(np.diff(quantiles) <= 0) or not np.all((0 < quantiles) & (quantiles < 1)):
        raise FloatingPointError("inverse-CDF quantiles are not strictly interior/monotone")
    return quantiles


def peak_locations(density: np.ndarray) -> list[int]:
    density = np.asarray(density, dtype=np.float64)
    kernel = np.asarray([1, 2, 3, 4, 5, 4, 3, 2, 1], dtype=np.float64)
    smooth = np.convolve(density, kernel / kernel.sum(), mode="same")
    radius = max(2, len(smooth) // 8)
    threshold = 0.10 * float(np.max(smooth) - np.min(smooth))
    candidates = [
        index
        for index in range(1, len(smooth) - 1)
        if smooth[index] > smooth[index - 1]
        and smooth[index] >= smooth[index + 1]
        and (
            smooth[index]
            - max(
                np.min(smooth[max(0, index - radius) : index]),
                np.min(smooth[index + 1 : min(len(smooth), index + radius + 1)]),
            )
            >= threshold
        )
    ]
    selected: list[int] = []
    for index in sorted(candidates, key=lambda item: smooth[item], reverse=True):
        if all(abs(index - other) >= radius for other in selected):
            selected.append(index)
    return sorted(selected)


def phase_regimes(inv_freq: np.ndarray, length: int) -> dict[str, Any]:
    phase = np.asarray(inv_freq, dtype=np.float64) * length
    wrapped = phase > 2.0 * math.pi
    resolving = (phase > 1.0) & ~wrapped
    dead = phase <= 1.0
    return {
        "phase_span": phase.tolist(),
        "wrapped_pairs": int(wrapped.sum()),
        "resolving_pairs": int(resolving.sum()),
        "dead_pairs": int(dead.sum()),
        "dead_indices": np.flatnonzero(dead).tolist(),
    }


def bootstrap_headroom(
    window_mass: np.ndarray,
    length: int,
    base: float,
    bins: int,
    mapping: str,
    include_self: bool,
    samples: int,
) -> dict[str, float | list[float]]:
    rng = np.random.default_rng(20260821)
    values = []
    for _ in range(samples):
        selected = rng.integers(0, len(window_mass), size=len(window_mass))
        mass = window_mass[selected].sum(axis=0)
        _, density, _ = density_from_distance(
            mass, length, base, bins, mapping, include_self
        )
        values.append(headroom(density)[0])
    return {
        "median": float(np.median(values)),
        "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
    }


def analyze_collection(
    metadata: dict[str, Any],
    arrays: dict[str, np.ndarray],
    bins: int,
    lambdas: list[float],
    tau: float,
    bootstrap_samples: int,
) -> dict[str, Any]:
    mass = arrays["mass"]
    opportunities = arrays["opportunities"]
    length = int(metadata["length"])
    base = float(metadata["base"])
    global_mass = mass.sum(axis=(0, 1))
    propensity = global_mass / np.maximum(opportunities, 1.0)
    profiles: dict[str, Any] = {}
    for mapping in ("endpoint_log", "physical_1rad"):
        for source, vector in (("mass", global_mass), ("propensity", propensity)):
            for include_self in (True, False):
                name = f"{mapping}__{source}__{'with_self' if include_self else 'no_self'}"
                centers, demand, probability = density_from_distance(
                    vector, length, base, bins, mapping, include_self
                )
                h_value, d_star = headroom(demand)
                allocations = {}
                for lam in lambdas:
                    rho = companded_density(demand, lam)
                    mixed = (1.0 - lam) * demand + lam
                    allocations[str(lam)] = {
                        "headroom_mixed": headroom(mixed)[0],
                        "rho": rho.tolist(),
                        "quantiles": midpoint_quantiles(rho, len(arrays["inv_freq"])).tolist(),
                        "distortion_original_m": distortion(demand, rho),
                        "distortion_mixed": distortion(mixed, rho),
                        "slow_half_budget": float(0.5 * np.mean(rho[bins // 2 :])),
                    }
                cosh = cosh_density(centers, tau)
                profiles[name] = {
                    "phi_centers": centers.tolist(),
                    "demand_density": demand.tolist(),
                    "probability": probability.tolist(),
                    "H": h_value,
                    "D_star": d_star,
                    "peaks": [float(centers[index]) for index in peak_locations(demand)],
                    "geo_distortion": distortion(demand, np.ones_like(demand)),
                    "cosh_distortion": distortion(demand, cosh),
                    "cosh_slow_half_budget": float(0.5 * np.mean(cosh[bins // 2 :])),
                    "allocations": allocations,
                }

    primary = profiles["endpoint_log__mass__no_self"]
    per_layer_h = []
    for layer in mass:
        _, density, _ = density_from_distance(
            layer.sum(axis=0), length, base, bins, "endpoint_log", False
        )
        per_layer_h.append(headroom(density)[0])
    per_head_h = []
    for layer in mass:
        per_head_h.append([])
        for head in layer:
            _, density, _ = density_from_distance(
                head, length, base, bins, "endpoint_log", False
            )
            per_head_h[-1].append(headroom(density)[0])

    long_mass = float(global_mass[length // 2 :].sum() / global_mass.sum())
    phase = phase_regimes(arrays["inv_freq"], length)
    frequency_band: dict[str, Any] = {"status": "NOT_COLLECTED"}
    if "q_pair_l2_mean" in arrays and "k_pair_l2_mean" in arrays:
        q_pair = np.asarray(arrays["q_pair_l2_mean"], dtype=np.float64)
        k_pair = np.asarray(arrays["k_pair_l2_mean"], dtype=np.float64)
        if q_pair.shape != k_pair.shape or q_pair.ndim != 3:
            raise ValueError("pair-norm arrays must share shape [layers,heads,pairs]")
        q_band = np.argmax(q_pair, axis=-1)
        k_band = np.argmax(k_pair, axis=-1)
        frequency_band = {
            "status": "QK_PAIR_L2_PROFILE_COMPLETE",
            "interpretation": "descriptive high-norm 2D-pair profile; not causal use or a positional/symbolic score",
            "q_pair_l2_global_mean": q_pair.mean(axis=(0, 1)).tolist(),
            "k_pair_l2_global_mean": k_pair.mean(axis=(0, 1)).tolist(),
            "q_band_index_per_layer_head": q_band.tolist(),
            "k_band_index_per_layer_head": k_band.tolist(),
            "q_band_index_mean": float(q_band.mean()),
            "k_band_index_mean": float(k_band.mean()),
            "pair_count": int(q_pair.shape[-1]),
        }
    return {
        "status": "R0_ATTENTION_OCCUPANCY_COMPLETE",
        "interpretation_boundary": (
            "Attention-distance occupancy is observed. Its map to m(phi) is a "
            "model assumption, not an identified per-frequency demand or LM-risk derivative."
        ),
        "measurement": metadata,
        "definition": {
            "endpoint_log": "phi=log(1+Delta)/log(L), same direction as distance",
            "physical_1rad": "phi=log_base(Delta), clipped to [0,1], omega*Delta≈1",
            "mixture": "m_tilde=(1-lambda)m+lambda*U",
            "allocation": "rho proportional to m_tilde^(1/3)",
            "lambdas": lambdas,
            "tau_comparator": tau,
            "bins": bins,
            "peak_rule": (
                "9-bin triangular smoothing; local prominence >=10% dynamic "
                "range; separation >= one eighth of phi span"
            ),
        },
        "global": {
            "distance_mass": (global_mass / global_mass.sum()).tolist(),
            "long_half_attention_mass": long_mass,
            "phase_regimes": phase,
        },
        "profiles": profiles,
        "heterogeneity": {
            "per_layer_H_endpoint_mass_no_self": per_layer_h,
            "per_layer_head_H_endpoint_mass_no_self": per_head_h,
        },
        "frequency_band": frequency_band,
        "bootstrap_primary_H": bootstrap_headroom(
            arrays["window_global_mass"],
            length,
            base,
            bins,
            "endpoint_log",
            False,
            bootstrap_samples,
        ),
        "registered_gate": {
            "H_gt_0.2": bool(primary["H"] > 0.2),
            "at_least_two_peaks": len(primary["peaks"]) >= 2,
            "automatic_R0_shape_gate": bool(
                primary["H"] > 0.2 and len(primary["peaks"]) >= 2
            ),
            "dead_phase_and_long_mass_are_cooccurrence_only": True,
        },
    }


def plot_analysis(result: dict[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    primary = result["profiles"]["endpoint_log__mass__no_self"]
    physical = result["profiles"]["physical_1rad__mass__no_self"]
    centers = np.asarray(primary["phi_centers"])
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.65))

    axes[0].plot(
        np.arange(len(result["global"]["distance_mass"])),
        result["global"]["distance_mass"],
        color="#2457A6",
        lw=1.2,
    )
    axes[0].set_xscale("symlog", linthresh=1)
    axes[0].set_yscale("log")
    axes[0].set_title("(a) Attention-distance occupancy")
    axes[0].set_xlabel("causal distance")
    axes[0].set_ylabel("attention mass")

    axes[1].plot(centers, primary["demand_density"], color="#20242A", label="candidate m")
    for lam, color in (("0.0", "#D06B25"), ("0.1", "#3E8E69"), ("0.3", "#7A55A3")):
        axes[1].plot(
            centers,
            primary["allocations"][lam]["rho"],
            color=color,
            lw=1.2,
            label=rf"$\rho^*_\lambda$, $\lambda={lam}$",
        )
    axes[1].set_title(f"(b) Endpoint-log map; H={primary['H']:.3f}")
    axes[1].set_xlabel(r"candidate frequency coordinate $\phi$")
    axes[1].set_ylabel("density")
    axes[1].legend(frameon=False, fontsize=6)

    axes[2].plot(
        centers,
        primary["demand_density"],
        color="#2457A6",
        lw=1.3,
        label="endpoint-log",
    )
    axes[2].plot(
        centers,
        physical["demand_density"],
        color="#D06B25",
        lw=1.3,
        label="physical 1-rad",
    )
    axes[2].set_title("(c) Mapping sensitivity")
    axes[2].set_xlabel(r"$\phi$")
    axes[2].set_ylabel("candidate demand density")
    axes[2].legend(frameon=False, fontsize=7)

    for axis in axes:
        axis.grid(True, color="#D9DDE3", lw=0.45)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def self_check() -> None:
    uniform = np.ones(64)
    h_uniform, d_uniform = headroom(uniform)
    if not math.isclose(h_uniform, 0.0, abs_tol=1e-12):
        raise AssertionError("uniform demand must have zero headroom")
    if not math.isclose(d_uniform, 1.0, abs_tol=1e-12):
        raise AssertionError("uniform optimum distortion must equal one")
    concentrated = np.zeros(64)
    concentrated[:8] = 8.0
    h_concentrated, _ = headroom(concentrated)
    if not 0.0 < h_concentrated < 1.0:
        raise AssertionError("non-uniform demand must have positive headroom")
    rho = companded_density(concentrated, 0.1)
    quantiles = midpoint_quantiles(rho, 32)
    if np.any(np.diff(quantiles) <= 0):
        raise AssertionError("quantiles must be strictly increasing")
    distance = np.arange(512)
    if np.any(np.diff(distance_phi(distance, 512, 500_000.0, "endpoint_log")) < 0):
        raise AssertionError("distance-to-phi mapping must be same-direction monotone")


def parse_lambdas(raw: str) -> list[float]:
    values = [float(value) for value in raw.split(",")]
    if not values or any(value < 0 or value > 1 for value in values):
        raise argparse.ArgumentTypeError("lambdas must be comma-separated values in [0,1]")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect")
    collect.add_argument(
        "--backend",
        choices=("project_gpt", "native_151m", "hf_olmo2"),
        required=True,
    )
    collect.add_argument("--tokens", type=Path, required=True)
    collect.add_argument("--length", type=int, required=True)
    collect.add_argument("--windows", type=int, default=16)
    collect.add_argument("--queries", type=int, default=32)
    collect.add_argument("--batch-size", type=int, default=1)
    collect.add_argument("--device", default="auto")
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--checkpoint", type=Path)
    collect.add_argument("--inv-freq", type=Path)
    collect.add_argument("--tier", choices=("50m", "125m", "350m", "500m"))
    collect.add_argument("--base", type=float, default=500_000.0)
    collect.add_argument("--seed", type=int, default=42)
    collect.add_argument("--model", type=Path)
    collect.add_argument("--model-revision", default="")
    collect.add_argument("--allow-slow-device", action="store_true")

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--input", type=Path, required=True)
    analyze.add_argument("--bins", type=int, default=64)
    analyze.add_argument("--lambdas", type=parse_lambdas, default=[0.0, 0.1, 0.3])
    analyze.add_argument("--tau", type=float)
    analyze.add_argument("--bootstrap-samples", type=int, default=1000)
    analyze.add_argument("--output-json", type=Path, required=True)
    analyze.add_argument("--output-figure", type=Path, required=True)
    return parser


def main() -> None:
    self_check()
    args = build_parser().parse_args()
    if args.command == "collect":
        if args.backend == "project_gpt":
            if not args.checkpoint or not args.inv_freq or not args.tier:
                raise ValueError("project_gpt requires --checkpoint, --inv-freq, and --tier")
            metadata, arrays = collect_project_gpt(args)
        elif args.backend == "native_151m":
            if not args.checkpoint or not args.inv_freq:
                raise ValueError("native_151m requires --checkpoint and --inv-freq")
            metadata, arrays = collect_native_151m(args)
        else:
            if not args.model:
                raise ValueError("hf_olmo2 requires --model")
            metadata, arrays = collect_hf_olmo2(args)
        write_collection(args.output, metadata, arrays)
        print(json.dumps({
            "status": "COLLECTION_COMPLETE",
            "output": str(args.output.resolve()),
            "sha256": sha256(args.output),
            "metadata": metadata,
        }, indent=2, sort_keys=True))
        return

    metadata, arrays = load_collection(args.input)
    tau = args.tau
    if tau is None:
        tau = float(metadata["head_dim"]) / math.sqrt(float(metadata["length"]))
    result = analyze_collection(
        metadata,
        arrays,
        args.bins,
        args.lambdas,
        tau,
        args.bootstrap_samples,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    plot_analysis(result, args.output_figure)
    result["artifacts"] = {
        "source_npz": str(args.input.resolve()),
        "source_npz_sha256": sha256(args.input),
        "json": str(args.output_json.resolve()),
        "figure": str(args.output_figure.resolve()),
        "figure_sha256": sha256(args.output_figure),
    }
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "gate": result["registered_gate"],
        "output_json": str(args.output_json.resolve()),
        "output_json_sha256": sha256(args.output_json),
        "output_figure": str(args.output_figure.resolve()),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
