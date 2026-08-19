#!/usr/bin/env python3
"""CPU-only attention-Fisher probe for the local L=512 50M RoPE checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.core_text_phases.run_evq_sweep import (
    GPT,
    TIER_CONFIGS,
    apply_rope,
)


RUN_ROOT = ROOT / "results/weekend_sweep/L512"
ARMS = {
    "geometric_tau0_seed42": RUN_ROOT / "50m_tau0.00_seed42",
    "evq_cosh_tau2.83_seed42": RUN_ROOT / "50m_tau2.83_seed42",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _spectrum(matrix: torch.Tensor) -> dict[str, float | int]:
    value = 0.5 * (matrix.double() + matrix.double().T)
    eigenvalues = torch.linalg.eigvalsh(value).clamp_min(0.0)
    trace = float(eigenvalues.sum())
    if trace <= 0.0:
        raise FloatingPointError("zero Fisher trace")
    probability = eigenvalues / trace
    positive = probability[probability > 1e-14]
    stable = float(1.0 / torch.sum(probability.square()))
    entropy = float(torch.exp(-torch.sum(positive * positive.log())))
    descending = torch.flip(eigenvalues, dims=(0,))
    normalized = eigenvalues / (trace / len(eigenvalues))
    return {
        "dimension": int(len(eigenvalues)),
        "trace": trace,
        "stable_rank": stable,
        "entropy_rank": entropy,
        "top1_share": float(descending[0] / trace),
        "top5_share": float(descending[:5].sum() / trace),
        "normalized_logdet_per_dimension": float(
            torch.mean(normalized.clamp_min(1e-14).log())
        ),
        "positive_eigenvalues_gt_1e-10_max": int(
            (eigenvalues > eigenvalues.max() * 1e-10).sum()
        ),
        "min_eigenvalue": float(eigenvalues[0]),
        "max_eigenvalue": float(eigenvalues[-1]),
    }


def _softmax_fisher(jacobian: torch.Tensor, probability: torch.Tensor) -> torch.Tensor:
    """Per-head J^T(diag(p)-pp^T)J."""
    mean = torch.einsum("hn,hnd->hd", probability, jacobian)
    second = torch.einsum(
        "hn,hni,hnj->hij", probability, jacobian, jacobian
    )
    return second - torch.einsum("hi,hj->hij", mean, mean)


def _empirical_fisher(jacobian: torch.Tensor, logit_gradient: torch.Tensor) -> torch.Tensor:
    """Per-head (J^T g_z)(J^T g_z)^T for the observed LM-loss gradient."""
    projected = torch.einsum("hn,hnd->hd", logit_gradient, jacobian)
    return torch.einsum("hi,hj->hij", projected, projected)


def _summary(values: list[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "iqr": float(np.quantile(array, 0.75) - np.quantile(array, 0.25)),
    }


def _centered_subspace_deficit(
    probability: torch.Tensor, distance: torch.Tensor, max_phase: float
) -> float:
    """Chordal deficit to centered span{Delta, Delta^2} under Cov_p."""
    probability = probability.double()
    distance = distance.double()
    omega = float(max_phase) / max(float(distance.max()), 1.0)
    trigonometric = torch.stack((
        torch.sin(omega * distance) / omega,
        2.0 * (1.0 - torch.cos(omega * distance)) / (omega * omega),
    ), dim=1)
    polynomial = torch.stack((distance, distance.square()), dim=1)

    def orthonormal(features: torch.Tensor) -> torch.Tensor:
        centered = features - torch.sum(probability[:, None] * features, dim=0)
        weighted = probability.sqrt()[:, None] * centered
        basis, triangular = torch.linalg.qr(weighted, mode="reduced")
        if float(torch.min(torch.abs(torch.diag(triangular)))) < 1e-12:
            raise FloatingPointError("centered polynomial basis became singular")
        return basis

    left = orthonormal(trigonometric)
    right = orthonormal(polynomial)
    canonical = torch.linalg.svdvals(left.T @ right)
    return max(0.0, 2.0 - float(torch.sum(canonical.square())))


def _static_geometry(inv_freq: torch.Tensor, length: int) -> dict[str, float]:
    distance = torch.arange(length, dtype=torch.float64)
    phase = torch.outer(distance, inv_freq.double())
    interleaved = torch.stack((phase.cos(), phase.sin()), dim=2).reshape(length, -1)
    gram = interleaved.T @ interleaved / length
    pairs = len(inv_freq)
    whitening = torch.zeros_like(gram)
    for pair in range(pairs):
        block = gram[2 * pair : 2 * pair + 2, 2 * pair : 2 * pair + 2]
        values, vectors = torch.linalg.eigh(block)
        inverse_root = vectors @ torch.diag(values.rsqrt()) @ vectors.T
        whitening[2 * pair : 2 * pair + 2, 2 * pair : 2 * pair + 2] = inverse_root
    correlation = whitening @ gram @ whitening
    stable = float(torch.trace(correlation) ** 2 / torch.sum(correlation.square()))
    collision = float((2.0 * pairs / stable - 1.0) / (pairs - 1.0))
    raw_spectrum = _spectrum(gram)
    whitened_spectrum = _spectrum(correlation)
    return {
        "raw_entropy_rank": raw_spectrum["entropy_rank"],
        "raw_normalized_logdet_per_dimension": raw_spectrum[
            "normalized_logdet_per_dimension"
        ],
        "block_whitened_stable_rank": stable,
        "block_whitened_entropy_rank": whitened_spectrum["entropy_rank"],
        "block_whitened_normalized_logdet_per_dimension": whitened_spectrum[
            "normalized_logdet_per_dimension"
        ],
        "full_subspace_collision_mean": collision,
    }


def _band_shares(diagonal: torch.Tensor, inv_freq: torch.Tensor, length: int) -> dict[str, object]:
    phase = inv_freq.double() * length
    bands = {
        "omegaL_le_1": phase <= 1.0,
        "one_lt_omegaL_le_2pi": (phase > 1.0) & (phase <= 2.0 * math.pi),
        "omegaL_gt_2pi": phase > 2.0 * math.pi,
    }
    total = float(diagonal.sum())
    return {
        name: {
            "pairs": int(mask.sum()),
            "information_share": float(diagonal[mask].sum()) / total,
        }
        for name, mask in bands.items()
    }


def _load_arm(
    path: Path,
    length: int,
    runtime_inv_freq: torch.Tensor,
    runtime_table: str,
) -> tuple[GPT, torch.Tensor, dict[str, object]]:
    checkpoint = path / "model.pt"
    inv_path = path / "inv_freq.npy"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    training_inv_freq = torch.from_numpy(np.load(inv_path)).float()
    config = dict(TIER_CONFIGS["50m"])
    config["seq_len"] = length
    config["max_position_embeddings"] = length
    model = GPT(config, training_inv_freq).cpu().eval()
    loaded = model.load_state_dict(state, strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError(f"state-dict mismatch: {loaded}")
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise RuntimeError("probe must remain CPU-only")
    for layer in range(config["num_layers"]):
        realized = state[f"blocks.{layer}.attn.rope.inv_freq"]
        if not torch.equal(realized, training_inv_freq):
            raise RuntimeError(f"layer {layer} frequency tensor mismatch")
    ropes = {id(block.attn.rope): block.attn.rope for block in model.blocks}
    for rope in ropes.values():
        rope.inv_freq.copy_(runtime_inv_freq)
        rope._build(length)
    model.requires_grad_(False)
    return model, runtime_inv_freq, {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "training_inv_freq": training_inv_freq.double().tolist(),
        "training_inv_freq_sha256_float32": hashlib.sha256(
            training_inv_freq.numpy().tobytes()
        ).hexdigest(),
        "runtime_table": runtime_table,
        "runtime_inv_freq": runtime_inv_freq.double().tolist(),
        "runtime_inv_freq_sha256_float32": hashlib.sha256(
            runtime_inv_freq.numpy().tobytes()
        ).hexdigest(),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "strict_state_dict": True,
    }


def _probe_arm(
    path: Path,
    runtime_inv_freq: torch.Tensor,
    runtime_table: str,
    validation: torch.Tensor,
    starts: np.ndarray,
    query_positions: tuple[int, ...],
    length: int,
) -> tuple[dict[str, object], dict[str, object]]:
    model, inv_freq, identity = _load_arm(
        path, length, runtime_inv_freq, runtime_table
    )
    layers = len(model.blocks)
    heads = model.blocks[0].attn.nh
    head_dim = model.blocks[0].attn.hd
    pairs = head_dim // 2
    dimensions = {
        "bare_softmax": 2 * pairs,
        "bare_empirical": 2 * pairs,
        "content_softmax": pairs,
        "content_empirical": pairs,
        "frequency_softmax": pairs,
        "frequency_empirical": pairs,
    }
    accumulators = {
        name: torch.zeros((dimension, dimension), dtype=torch.float64)
        for name, dimension in dimensions.items()
    }
    layer_head = {
        name: [
            [torch.zeros_like(value) for _ in range(heads)]
            for _ in range(layers)
        ]
        for name, value in accumulators.items()
    }
    query_groups = {name: [] for name in accumulators}
    observations = 0
    attention_entropy = []
    attention_max = []
    lm_losses = []
    sampled_query_losses = []
    parity_error = None
    lm_parity_error = None
    frequency_derivative_error = None
    low_frequency_phase = (1.0, 0.5, 0.25, 0.125)
    low_frequency_deficits = {value: [] for value in low_frequency_phase}
    causal_mask = torch.triu(torch.ones((length, length), dtype=torch.bool), diagonal=1)
    started = time.perf_counter()
    for window_index, start in enumerate(starts):
        tokens = validation[int(start) : int(start) + length].view(1, length)
        hidden = model.emb(tokens).detach().requires_grad_(True)
        captures = []
        for layer, block in enumerate(model.blocks):
            normalized = block.ln1(hidden)
            qkv = block.attn.qkv(normalized).view(
                1, length, 3, heads, head_dim
            ).permute(2, 0, 3, 1, 4)
            raw_query, raw_key, value = qkv[0], qkv[1], qkv[2]
            cosine, sine = block.attn.rope(length)
            query = apply_rope(raw_query, cosine[None, None], sine[None, None])
            key = apply_rope(raw_key, cosine[None, None], sine[None, None])
            scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(head_dim)
            logits = scores.masked_fill(causal_mask[None, None], -torch.inf)
            logits.retain_grad()
            probability = logits.softmax(dim=-1)
            attended = torch.matmul(probability, value)
            if parity_error is None and window_index == 0 and layer == 0:
                with torch.no_grad():
                    fused = F.scaled_dot_product_attention(
                        query.detach(), key.detach(), value.detach(), is_causal=True
                    )
                parity_error = float(torch.max(torch.abs(attended.detach() - fused)))
            hidden = hidden + block.attn.o(
                attended.transpose(1, 2).reshape(1, length, -1)
            )
            hidden = hidden + block.mlp(block.ln2(hidden))
            captures.append({
                "raw_query": raw_query,
                "raw_key": raw_key,
                "query": query,
                "key": key,
                "probability": probability,
                "logits": logits,
            })

        targets = validation[int(start) + 1 : int(start) + length + 1]
        normalized = model.ln(hidden)[0]
        token_losses = []
        for offset in range(0, length, 64):
            lm_logits = model.head(normalized[offset : offset + 64])
            token_losses.append(
                F.cross_entropy(
                    lm_logits,
                    targets[offset : offset + 64],
                    reduction="none",
                )
            )
        token_losses = torch.cat(token_losses)
        loss = token_losses.mean()
        loss.backward()
        if lm_parity_error is None and window_index == 0:
            with torch.no_grad():
                fused_hidden = model.emb(tokens)
                for block in model.blocks:
                    fused_hidden = block(fused_hidden)
                fused_normalized = model.ln(fused_hidden)[0]
                fused_losses = []
                for offset in range(0, length, 64):
                    fused_losses.append(F.cross_entropy(
                        model.head(fused_normalized[offset : offset + 64]),
                        targets[offset : offset + 64],
                        reduction="none",
                    ))
                fused_loss = torch.cat(fused_losses).mean()
            lm_parity_error = abs(float(fused_loss) - float(loss.detach()))
        lm_losses.extend(token_losses.detach().double().tolist())

        for query_position in query_positions:
            sampled_query_losses.append(float(token_losses[query_position].detach()))
            group = {name: torch.zeros_like(value) for name, value in accumulators.items()}
            for layer, capture in enumerate(captures):
                count = query_position + 1
                delta = query_position - torch.arange(count, dtype=torch.float64)
                probability = capture["probability"][0, :, query_position, :count].detach().double()
                gradient = capture["logits"].grad[0, :, query_position, :count].detach().double()
                query = capture["query"][0].detach().double()
                key = capture["key"][0].detach().double()

                entropy = torch.exp(
                    -torch.sum(
                        probability * probability.clamp_min(1e-300).log(), dim=-1
                    )
                )
                attention_entropy.extend(entropy.tolist())
                attention_max.extend(probability.max(dim=-1).values.tolist())

                phase = torch.outer(delta, inv_freq.double())
                bare = torch.cat((phase.cos(), phase.sin()), dim=1)
                bare = bare.unsqueeze(0).expand(heads, -1, -1)
                q1 = query[:, query_position, :pairs]
                q2 = query[:, query_position, pairs:]
                k1 = key[:, :count, :pairs]
                k2 = key[:, :count, pairs:]
                content = (
                    q1[:, None] * k1 + q2[:, None] * k2
                ) / math.sqrt(head_dim)
                frequency = (
                    (-q2[:, None] * k1 + q1[:, None] * k2)
                    * delta[None, :, None]
                    * inv_freq.double()[None, None, :]
                    / math.sqrt(head_dim)
                )
                matrices = {
                    "bare_softmax": _softmax_fisher(bare, probability),
                    "bare_empirical": _empirical_fisher(bare, gradient),
                    "content_softmax": _softmax_fisher(content, probability),
                    "content_empirical": _empirical_fisher(content, gradient),
                    "frequency_softmax": _softmax_fisher(frequency, probability),
                    "frequency_empirical": _empirical_fisher(frequency, gradient),
                }
                for name, per_head in matrices.items():
                    accumulators[name] += per_head.sum(dim=0)
                    group[name] += per_head.sum(dim=0)
                    for head in range(heads):
                        layer_head[name][layer][head] += per_head[head]
                observations += heads

                for head in range(heads):
                    for max_phase in low_frequency_phase:
                        low_frequency_deficits[max_phase].append(
                            _centered_subspace_deficit(
                                probability[head], delta, max_phase
                            )
                        )

                if frequency_derivative_error is None and window_index == 0 and layer == 0:
                    epsilon = 1e-5
                    derivative_errors = []
                    key_position = 0
                    raw_query = capture["raw_query"][0].detach()
                    raw_key = capture["raw_key"][0].detach()
                    for pair in (0, pairs // 2, pairs - 1):
                        omega = float(inv_freq[pair])
                        q_pair = raw_query[0, query_position, [pair, pair + pairs]].double()
                        k_pair = raw_key[0, key_position, [pair, pair + pairs]].double()

                        def contribution(log_shift: float) -> torch.Tensor:
                            shifted = omega * math.exp(log_shift)
                            q_angle = shifted * query_position
                            k_angle = shifted * key_position
                            q_rotated = torch.stack((
                                q_pair[0] * math.cos(q_angle) - q_pair[1] * math.sin(q_angle),
                                q_pair[1] * math.cos(q_angle) + q_pair[0] * math.sin(q_angle),
                            ))
                            k_rotated = torch.stack((
                                k_pair[0] * math.cos(k_angle) - k_pair[1] * math.sin(k_angle),
                                k_pair[1] * math.cos(k_angle) + k_pair[0] * math.sin(k_angle),
                            ))
                            return torch.dot(q_rotated, k_rotated) / math.sqrt(head_dim)

                        numerical = (
                            contribution(epsilon) - contribution(-epsilon)
                        ) / (2.0 * epsilon)
                        derivative_errors.append(
                            float(torch.abs(numerical - frequency[0, key_position, pair]))
                        )
                    frequency_derivative_error = max(derivative_errors)
            for name, value in group.items():
                query_groups[name].append(value / (layers * heads))

    if parity_error is None or parity_error > 2e-5:
        raise AssertionError(f"manual attention parity failed: {parity_error}")
    if lm_parity_error is None or lm_parity_error > 2e-5:
        raise AssertionError(f"manual LM-loss parity failed: {lm_parity_error}")
    if frequency_derivative_error is None or frequency_derivative_error > 2e-4:
        raise AssertionError(
            f"log-frequency Jacobian finite-difference check failed: {frequency_derivative_error}"
        )
    for name in accumulators:
        accumulators[name] /= observations
        for layer in range(layers):
            for head in range(heads):
                layer_head[name][layer][head] /= len(starts) * len(query_positions)

    definitions = {
        "bare_softmax": "Phi^T F_sm Phi",
        "bare_empirical": "(Phi^T g_z)(Phi^T g_z)^T",
        "content_softmax": "J_content^T F_sm J_content",
        "content_empirical": "(J_content^T g_z)(J_content^T g_z)^T",
        "frequency_softmax": "J_logomega^T F_sm J_logomega",
        "frequency_empirical": "(J_logomega^T g_z)(J_logomega^T g_z)^T",
    }
    fisher = {}
    for name, matrix in accumulators.items():
        diagonal = torch.diag(matrix)
        if name.startswith("bare_"):
            diagonal = diagonal[:pairs] + diagonal[pairs:]
        records = []
        for layer in range(layers):
            for head in range(heads):
                records.append({
                    "layer": layer,
                    "head": head,
                    "spectrum": _spectrum(layer_head[name][layer][head]),
                })
        fisher[name] = {
            "definition": definitions[name],
            "spectrum": _spectrum(matrix),
            "band_information": _band_shares(diagonal, inv_freq, length),
            "pair_diagonal": diagonal.tolist(),
            "layer_head_spectrum": records,
            "layer_head_summary": {
                field: _summary([row["spectrum"][field] for row in records])
                for field in (
                    "trace",
                    "stable_rank",
                    "entropy_rank",
                    "normalized_logdet_per_dimension",
                )
            },
        }

    mean_deficit = np.asarray([
        np.mean(low_frequency_deficits[value]) for value in low_frequency_phase
    ])
    slope = float(np.polyfit(
        np.log(np.asarray(low_frequency_phase)),
        np.log(np.maximum(mean_deficit, 1e-30)),
        1,
    )[0])
    public = {
        "identity": identity,
        "runtime": {
            "device": "cpu",
            "seconds": time.perf_counter() - started,
            "windows": len(starts),
            "layers": layers,
            "heads": heads,
            "query_positions": list(query_positions),
            "head_query_observations": observations,
            "manual_vs_sdpa_max_abs": parity_error,
            "manual_vs_fused_lm_loss_abs": lm_parity_error,
            "log_frequency_jacobian_finite_difference_max_abs": frequency_derivative_error,
            "attention_effective_keys_mean": float(np.mean(attention_entropy)),
            "attention_effective_keys_median": float(np.median(attention_entropy)),
            "attention_max_probability_mean": float(np.mean(attention_max)),
        },
        "lm": {
            "loss": float(np.mean(lm_losses)),
            "ppl": float(math.exp(np.mean(lm_losses))),
            "tokens": len(lm_losses),
            "sampled_query_loss_mean": float(np.mean(sampled_query_losses)),
            "sampled_queries": len(sampled_query_losses),
        },
        "static_geometry": _static_geometry(inv_freq, length),
        "fisher": fisher,
        "low_frequency_softmax_geometry": {
            "expected_limit": "centered span{Delta, Delta^2}",
            "max_phase": {
                str(value): {
                    "mean_chordal_deficit": float(np.mean(low_frequency_deficits[value])),
                    "median_chordal_deficit": float(np.median(low_frequency_deficits[value])),
                    "max_chordal_deficit": float(np.max(low_frequency_deficits[value])),
                }
                for value in low_frequency_phase
            },
            "log_log_slope_mean_deficit": slope,
        },
    }
    internal = {
        "query_group_matrices": query_groups,
        "sampled_query_losses": np.asarray(sampled_query_losses, dtype=np.float64),
    }
    return public, internal


def _factorial(values: dict[str, float]) -> dict[str, float]:
    gg = values["geo_weights_geo_table"]
    ge = values["geo_weights_evq_table"]
    eg = values["evq_weights_geo_table"]
    ee = values["evq_weights_evq_table"]
    return {
        "grand_mean": (gg + ge + eg + ee) / 4.0,
        "table_main_effect_evq_minus_geo": ((ge + ee) - (gg + eg)) / 2.0,
        "weights_main_effect_evq_minus_geo": ((eg + ee) - (gg + ge)) / 2.0,
        "table_by_weights_interaction": ee - eg - ge + gg,
    }


def _effect_summary(values: np.ndarray) -> dict[str, float]:
    return {
        **_summary(values),
        "mean": float(np.mean(values)),
    }


def run(length: int, windows: int, bootstrap_samples: int = 500) -> dict[str, object]:
    if length != 512 or windows < 2:
        raise ValueError("this frozen probe requires length=512 and at least two windows")
    validation_path = RUN_ROOT / "val_tinystories_5000000.pt"
    validation = torch.load(validation_path, map_location="cpu", weights_only=True)
    if validation.ndim != 1 or len(validation) < length:
        raise ValueError("invalid validation token tensor")
    starts = np.linspace(0, len(validation) - length - 1, windows, dtype=np.int64)
    query_positions = (63, 127, 255, 383, 511)
    tables = {
        "geometric_tau0": torch.from_numpy(
            np.load(ARMS["geometric_tau0_seed42"] / "inv_freq.npy")
        ).float(),
        "evq_cosh_tau2.83": torch.from_numpy(
            np.load(ARMS["evq_cosh_tau2.83_seed42"] / "inv_freq.npy")
        ).float(),
    }
    arms = {}
    internal = {}
    for checkpoint, path in ARMS.items():
        for table, runtime_inv_freq in tables.items():
            key = f"{checkpoint}__runtime_{table}"
            arms[key], internal[key] = _probe_arm(
                path,
                runtime_inv_freq,
                table,
                validation,
                starts,
                query_positions,
                length,
            )
    keys = {
        "geo_weights_geo_table": "geometric_tau0_seed42__runtime_geometric_tau0",
        "geo_weights_evq_table": "geometric_tau0_seed42__runtime_evq_cosh_tau2.83",
        "evq_weights_geo_table": "evq_cosh_tau2.83_seed42__runtime_geometric_tau0",
        "evq_weights_evq_table": "evq_cosh_tau2.83_seed42__runtime_evq_cosh_tau2.83",
    }
    matrix_metrics = tuple(next(iter(arms.values()))["fisher"])
    main_cells = {}
    for alias, key in keys.items():
        arm = arms[key]
        cell = {
            "lm_loss": arm["lm"]["loss"],
            "lm_ppl": arm["lm"]["ppl"],
            "bare_geometry_stable_rank": arm["static_geometry"][
                "block_whitened_stable_rank"
            ],
            "bare_geometry_logdet": arm["static_geometry"][
                "block_whitened_normalized_logdet_per_dimension"
            ],
        }
        for metric in matrix_metrics:
            spectrum = arm["fisher"][metric]["spectrum"]
            for field in (
                "trace",
                "stable_rank",
                "entropy_rank",
                "normalized_logdet_per_dimension",
            ):
                cell[f"{metric}.{field}"] = spectrum[field]
        main_cells[alias] = cell

    aggregate_factorial = {
        metric: _factorial({alias: cell[metric] for alias, cell in main_cells.items()})
        for metric in next(iter(main_cells.values()))
    }

    layer_head_factorial = {}
    for metric in matrix_metrics:
        layer_head_factorial[metric] = {}
        for field in ("stable_rank", "normalized_logdet_per_dimension"):
            cell_values = {
                alias: np.asarray([
                    row["spectrum"][field]
                    for row in arms[key]["fisher"][metric]["layer_head_spectrum"]
                ])
                for alias, key in keys.items()
            }
            effects = {
                "table_main_effect_evq_minus_geo": (
                    (cell_values["geo_weights_evq_table"] + cell_values["evq_weights_evq_table"])
                    - (cell_values["geo_weights_geo_table"] + cell_values["evq_weights_geo_table"])
                ) / 2.0,
                "weights_main_effect_evq_minus_geo": (
                    (cell_values["evq_weights_geo_table"] + cell_values["evq_weights_evq_table"])
                    - (cell_values["geo_weights_geo_table"] + cell_values["geo_weights_evq_table"])
                ) / 2.0,
                "table_by_weights_interaction": (
                    cell_values["evq_weights_evq_table"]
                    - cell_values["evq_weights_geo_table"]
                    - cell_values["geo_weights_evq_table"]
                    + cell_values["geo_weights_geo_table"]
                ),
            }
            layer_head_factorial[metric][field] = {
                name: _effect_summary(value) for name, value in effects.items()
            }

    rng = np.random.default_rng(20260819)
    group_count = windows * len(query_positions)
    stacked = {
        alias: {
            metric: torch.stack(internal[key]["query_group_matrices"][metric])
            for metric in matrix_metrics
        }
        for alias, key in keys.items()
    }
    sampled_losses = {
        alias: internal[key]["sampled_query_losses"] for alias, key in keys.items()
    }
    bootstrap_effects: dict[str, dict[str, list[float]]] = {
        "lm_sampled_query_loss": {
            name: [] for name in (
                "table_main_effect_evq_minus_geo",
                "weights_main_effect_evq_minus_geo",
                "table_by_weights_interaction",
            )
        }
    }
    for metric in matrix_metrics:
        for field in ("trace", "stable_rank", "normalized_logdet_per_dimension"):
            bootstrap_effects[f"{metric}.{field}"] = {
                name: [] for name in bootstrap_effects["lm_sampled_query_loss"]
            }

    for _ in range(bootstrap_samples):
        indices = rng.integers(0, group_count, size=group_count)
        loss_cells = {
            alias: float(np.mean(values[indices]))
            for alias, values in sampled_losses.items()
        }
        for name, value in _factorial(loss_cells).items():
            if name in bootstrap_effects["lm_sampled_query_loss"]:
                bootstrap_effects["lm_sampled_query_loss"][name].append(value)
        for metric in matrix_metrics:
            spectra = {
                alias: _spectrum(value[metric][indices].mean(dim=0))
                for alias, value in stacked.items()
            }
            for field in ("trace", "stable_rank", "normalized_logdet_per_dimension"):
                effects = _factorial({
                    alias: spectrum[field] for alias, spectrum in spectra.items()
                })
                for name in bootstrap_effects[f"{metric}.{field}"]:
                    bootstrap_effects[f"{metric}.{field}"][name].append(effects[name])

    bootstrap = {
        metric: {
            effect: {
                "median": float(np.median(values)),
                "ci95": [
                    float(np.quantile(values, 0.025)),
                    float(np.quantile(values, 0.975)),
                ],
            }
            for effect, values in effects.items()
        }
        for metric, effects in bootstrap_effects.items()
    }
    return {
        "status": "CPU_ONLY_COMPLETE",
        "interpretation_boundary": (
            "*_softmax uses the attention categorical Fisher F_sm=diag(p)-pp^T. "
            "*_empirical uses the observed task gradient outer product from mean LM loss. "
            "Neither is called the other. Frequency-table transplants use the historical "
            "trained tables and therefore mix their sampled endpoints/span with shape."
        ),
        "protocol": {
            "length": length,
            "validation_windows": windows,
            "window_starts": starts.tolist(),
            "query_positions": list(query_positions),
            "validation_path": str(validation_path.resolve()),
            "validation_sha256": _sha256(validation_path),
            "same_tokens_across_arms": True,
            "training_or_parameter_updates": False,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": 20260819,
        },
        "arms": arms,
        "arm_keys": keys,
        "main_cells": main_cells,
        "aggregate_factorial": aggregate_factorial,
        "layer_head_factorial": layer_head_factorial,
        "query_bootstrap_factorial": bootstrap,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/attention_fisher_50m_probe_20260819.json"),
    )
    args = parser.parse_args()
    result = run(args.length, args.windows, args.bootstrap_samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
