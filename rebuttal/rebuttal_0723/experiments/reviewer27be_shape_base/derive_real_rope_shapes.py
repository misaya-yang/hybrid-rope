#!/usr/bin/env python3
"""Derive fixed native-grid schedules from exact RoPE and attention priors.

The script keeps three evidence levels separate:

1. ``measure-attention`` extracts a distance prior from a trained Std-Geo
   checkpoint on the registered selection anchors only.
2. ``derive`` optimizes a content-independent exact-RoPE collision objective
   under either a uniform distance prior or the frozen attention prior.
3. The resulting schedules are fixed artifacts for later matched LM training;
   neither collision score is presented as a language-model objective.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from experiments.native_rope_evq_150m.model import GPT, apply_rope
from rebuttal.rebuttal_0723.experiments.reviewer27be_shape_base.prepare import (
    sha256_file,
    validate_manifest,
)
from rebuttal.rebuttal_0723.experiments.reviewer27be_shape_base.protocol import SPECS


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().float().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value, dtype=np.float64)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _checkpoint_model(
    checkpoint: Path,
    receipt_path: Path,
) -> tuple[GPT, dict[str, Any]]:
    checkpoint = checkpoint.resolve()
    receipt_path = receipt_path.resolve()
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("checkpoint_sha256") != sha256_file(checkpoint):
        raise ValueError("checkpoint SHA-256 differs from train receipt")
    if receipt.get("suite") != "shape_l128":
        raise ValueError("attention prior requires shape_l128 checkpoint")
    if receipt.get("arm") != "std_geo" or int(receipt.get("seed")) != 42:
        raise ValueError("attention prior must come from Std-Geo seed 42")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    metadata = payload.get("metadata")
    state = payload.get("model")
    if not isinstance(metadata, dict) or not isinstance(state, dict):
        raise ValueError("checkpoint payload is incomplete")
    if metadata.get("checkpoint_sha256") not in (None, receipt.get("checkpoint_sha256")):
        raise ValueError("embedded checkpoint receipt mismatch")
    inv_values = [
        tensor.detach().cpu().float().contiguous()
        for name, tensor in state.items()
        if name.endswith(".rope.inv_freq")
    ]
    if not inv_values or any(
        not torch.equal(inv_values[0], value) for value in inv_values[1:]
    ):
        raise ValueError("checkpoint does not contain one consistent inv_freq")
    if _tensor_sha256(inv_values[0]) != metadata.get("inv_freq_sha256"):
        raise ValueError("checkpoint inverse-frequency hash mismatch")
    spec = SPECS["shape_l128"]
    model = GPT(spec.model_config(), inv_values[0])
    model.load_state_dict(state, strict=True)
    return model, metadata


@torch.no_grad()
def measure_attention(
    *,
    checkpoint: Path,
    receipt: Path,
    data_manifest: Path,
    output: Path,
    batch_size: int,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for attention measurement")
    manifest = json.loads(data_manifest.resolve().read_text())
    validate_manifest(manifest, check_hashes=True)
    model, metadata = _checkpoint_model(checkpoint, receipt)
    spec = SPECS["shape_l128"]
    if metadata.get("validation_sha256") != manifest["validation"]["sha256"]:
        raise ValueError("checkpoint and manifest validation tensors differ")
    anchors_record = manifest["selection_anchors"]
    anchors = np.load(anchors_record["path"], allow_pickle=False)
    if sha256_file(anchors_record["path"]) != anchors_record["sha256"]:
        raise ValueError("selection-anchor hash mismatch")
    validation = np.load(
        manifest["validation"]["path"], mmap_mode="r", allow_pickle=False
    ).reshape(-1)
    length = spec.train_length
    windows = np.stack(
        [
            np.asarray(
                validation[int(start) : int(start) + length],
                dtype=np.int64,
            )
            for start in anchors
        ]
    )
    model = model.to("cuda").eval()
    distance_mass = torch.zeros(length, dtype=torch.float64, device="cuda")
    layer_mass = torch.zeros(
        (spec.num_layers, length), dtype=torch.float64, device="cuda"
    )
    for offset in range(0, len(windows), int(batch_size)):
        tokens = torch.from_numpy(
            np.array(windows[offset : offset + batch_size], copy=True)
        ).to("cuda")
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            hidden = model.embedding(tokens)
            hidden = hidden.to(dtype=torch.bfloat16)
            for layer_index, block in enumerate(model.blocks):
                normalized = block.norm1(hidden)
                batch, seq_len, _ = normalized.shape
                qkv = (
                    block.attention.qkv(normalized)
                    .view(
                        batch,
                        seq_len,
                        3,
                        block.attention.num_heads,
                        block.attention.head_dim,
                    )
                    .permute(2, 0, 3, 1, 4)
                )
                query, key, value = qkv[0], qkv[1], qkv[2]
                cos, sin = block.attention.rope(seq_len)
                query = apply_rope(
                    query, cos[None, None], sin[None, None]
                )
                key = apply_rope(key, cos[None, None], sin[None, None])
                scores = torch.matmul(
                    query.float(), key.float().transpose(-1, -2)
                ) / math.sqrt(block.attention.head_dim)
                causal = torch.ones(
                    (seq_len, seq_len),
                    dtype=torch.bool,
                    device=scores.device,
                ).triu(1)
                scores.masked_fill_(causal, float("-inf"))
                probabilities = torch.softmax(scores, dim=-1)
                for distance in range(seq_len):
                    mass = torch.diagonal(
                        probabilities,
                        offset=-distance,
                        dim1=-2,
                        dim2=-1,
                    ).sum(dtype=torch.float64)
                    distance_mass[distance] += mass
                    layer_mass[layer_index, distance] += mass
                attended = F.scaled_dot_product_attention(
                    query, key, value, is_causal=True
                )
                hidden = hidden + block.attention.output(
                    attended.transpose(1, 2).reshape(batch, seq_len, -1)
                )
                hidden = hidden + block.mlp(block.norm2(hidden))
    raw = distance_mass.cpu().numpy()
    if not np.isfinite(raw).all() or np.any(raw < 0):
        raise ValueError("attention-distance mass is invalid")
    positive = raw[1:]
    if float(positive.sum()) <= 0:
        raise ValueError("attention prior has no positive-distance mass")
    prior = positive / positive.sum()
    layer_raw = layer_mass.cpu().numpy()
    layer_prior = layer_raw[:, 1:] / layer_raw[:, 1:].sum(
        axis=1, keepdims=True
    )
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "source": {
            "identity": "trained Std-Geo native endpoint, seed 42",
            "checkpoint_sha256": sha256_file(checkpoint),
            "train_receipt_sha256": sha256_file(receipt),
            "data_manifest_sha256": sha256_file(data_manifest),
            "selection_anchor_sha256": anchors_record["sha256"],
            "validation_sha256": manifest["validation"]["sha256"],
        },
        "contract": {
            "suite": "shape_l128",
            "length": length,
            "anchors": int(len(windows)),
            "layers": spec.num_layers,
            "heads": spec.num_heads,
            "distance_zero_excluded": True,
            "aggregation": (
                "sum of causal attention probability mass by query-key "
                "distance, then normalize over distances 1..L-1"
            ),
            "evidence_boundary": (
                "selection-only diagnostic; not an LM objective and not "
                "evaluated on test anchors"
            ),
        },
        "distances": list(range(1, length)),
        "prior": prior.tolist(),
        "prior_sha256_float64": _array_sha256(prior),
        "layer_priors": layer_prior.tolist(),
    }
    _atomic_json(output.resolve(), payload)
    return payload


def _native_evq_span_matched(
    channels: int, tau: float
) -> tuple[np.ndarray, np.ndarray]:
    geo = np.arange(channels, dtype=np.float64) / float(channels)
    u = geo.copy()
    raw = 1.0 - np.arcsinh((1.0 - u) * np.sinh(tau)) / tau
    if raw[-1] <= 0:
        raise ValueError("EVQ native-grid span is invalid")
    value = raw * (geo[-1] / raw[-1])
    return geo, value


def _exp_span_matched(
    geo: np.ndarray, target_rms: float
) -> tuple[np.ndarray, float]:
    from scipy.optimize import brentq

    t = np.arange(len(geo), dtype=np.float64) / float(len(geo) - 1)
    span = float(geo[-1])

    def candidate(parameter: float) -> np.ndarray:
        warped = np.expm1(parameter * t) / np.expm1(parameter)
        return span * warped

    def residual(parameter: float) -> float:
        value = candidate(parameter)
        return float(np.sqrt(np.mean((value - geo) ** 2)) - target_rms)

    parameter = float(brentq(residual, 1e-8, 128.0))
    return candidate(parameter), parameter


def _kernel_matrix(
    phi: np.ndarray,
    *,
    base: float,
    distances: np.ndarray,
    prior: np.ndarray,
) -> np.ndarray:
    omega = np.power(float(base), -np.asarray(phi, dtype=np.float64))
    features = np.cos(np.outer(distances, omega))
    # SLSQP can briefly probe numerically invalid interior points even with
    # bounds.  Return the matrix under a local floating-point guard; the caller
    # rejects any non-finite candidate with a large objective.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return (features * prior[:, None]).T @ features


def _collision_score(
    phi: np.ndarray,
    *,
    base: float,
    distances: np.ndarray,
    prior: np.ndarray,
) -> float:
    phi = np.asarray(phi, dtype=np.float64)
    if (
        not np.isfinite(phi).all()
        or float(phi.min()) < 0.0
        or float(phi.max()) > 1.0
        or np.any(np.diff(phi) <= 0.0)
    ):
        return 1e30
    matrix = _kernel_matrix(
        phi, base=base, distances=distances, prior=prior
    )
    diagonal = np.diag(matrix)
    if not np.isfinite(matrix).all() or np.any(diagonal <= 0):
        return 1e30
    normalized = matrix / np.sqrt(np.outer(diagonal, diagonal))
    return float(np.sum(np.triu(normalized * normalized, k=1)))


def _effective_rank(
    phi: np.ndarray,
    *,
    base: float,
    distances: np.ndarray,
    prior: np.ndarray,
) -> float:
    matrix = _kernel_matrix(
        phi, base=base, distances=distances, prior=prior
    )
    values = np.linalg.eigvalsh(matrix)
    values = np.clip(values, 0.0, None)
    probabilities = values / values.sum()
    probabilities = probabilities[probabilities > 0]
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def _optimize_fixed_rms(
    *,
    geo: np.ndarray,
    target_rms: float,
    base: float,
    distances: np.ndarray,
    prior: np.ndarray,
    starts: list[tuple[str, np.ndarray]],
) -> tuple[np.ndarray, dict[str, Any]]:
    from scipy.optimize import minimize

    span = float(geo[-1])
    minimum_gap = 1e-4

    def expand(interior: np.ndarray) -> np.ndarray:
        return np.concatenate(([0.0], interior, [span]))

    def objective(interior: np.ndarray) -> float:
        return _collision_score(
            expand(interior),
            base=base,
            distances=distances,
            prior=prior,
        )

    def rms_equality(interior: np.ndarray) -> float:
        value = expand(interior)
        return float(np.mean((value - geo) ** 2) - target_rms**2)

    def gap_constraint(interior: np.ndarray) -> np.ndarray:
        return np.diff(expand(interior)) - minimum_gap

    candidates: list[dict[str, Any]] = []
    for name, start in starts:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Values in x were outside bounds during a minimize step",
                category=RuntimeWarning,
            )
            result = minimize(
                objective,
                np.asarray(start[1:-1], dtype=np.float64),
                method="SLSQP",
                bounds=[(minimum_gap, span - minimum_gap)] * (len(geo) - 2),
                constraints=[
                    {"type": "eq", "fun": rms_equality},
                    {"type": "ineq", "fun": gap_constraint},
                ],
                options={"maxiter": 1_000, "ftol": 1e-11, "disp": False},
            )
        value = expand(result.x)
        valid = (
            bool(result.success)
            and np.isfinite(value).all()
            and float(np.diff(value).min()) >= minimum_gap * 0.95
            and abs(
                float(np.sqrt(np.mean((value - geo) ** 2))) - target_rms
            )
            < 1e-7
        )
        candidates.append(
            {
                "start": name,
                "success": valid,
                "message": str(result.message),
                "iterations": int(result.nit),
                "objective": float(objective(result.x)),
                "schedule": value,
            }
        )
    valid_candidates = [row for row in candidates if row["success"]]
    if not valid_candidates:
        raise RuntimeError("all exact-kernel schedule optimizations failed")
    selected = min(valid_candidates, key=lambda row: row["objective"])
    schedule = np.asarray(selected["schedule"], dtype=np.float64)
    receipt = {
        "selected_start": selected["start"],
        "objective": selected["objective"],
        "iterations": selected["iterations"],
        "minimum_gap": float(np.diff(schedule).min()),
        "deformation_rms": float(
            np.sqrt(np.mean((schedule - geo) ** 2))
        ),
        "candidate_objectives": {
            row["start"]: row["objective"]
            for row in candidates
            if row["success"]
        },
    }
    return schedule, receipt


def derive_schedules(
    *,
    attention_prior_path: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    attention_record = json.loads(attention_prior_path.resolve().read_text())
    if attention_record.get("status") != "PASS":
        raise ValueError("attention-prior receipt is not PASS")
    spec = SPECS["shape_l128"]
    channels = spec.head_dim // 2
    tau = spec.rule_tau
    geo, evq = _native_evq_span_matched(channels, tau)
    target_rms = float(np.sqrt(np.mean((evq - geo) ** 2)))
    exp_schedule, exp_parameter = _exp_span_matched(geo, target_rms)
    uniform_distances = np.arange(1, spec.train_length + 1, dtype=np.float64)
    uniform_prior = np.full(
        len(uniform_distances), 1.0 / len(uniform_distances), dtype=np.float64
    )
    attention_distances = np.asarray(
        attention_record["distances"], dtype=np.float64
    )
    attention_prior = np.asarray(
        attention_record["prior"], dtype=np.float64
    )
    if (
        len(attention_distances) != spec.train_length - 1
        or attention_prior.shape != attention_distances.shape
        or not np.isclose(attention_prior.sum(), 1.0, atol=1e-12)
    ):
        raise ValueError("attention prior has an unexpected shape")
    midpoint_power = geo[-1] * np.power(
        np.arange(channels, dtype=np.float64) / float(channels - 1), 3.0
    )
    starts = [
        ("native_evq", evq),
        ("native_exp", exp_schedule),
        ("power3", midpoint_power),
    ]
    exact_schedule, exact_receipt = _optimize_fixed_rms(
        geo=geo,
        target_rms=target_rms,
        base=spec.rope_base,
        distances=uniform_distances,
        prior=uniform_prior,
        starts=starts,
    )
    attention_schedule, attention_receipt = _optimize_fixed_rms(
        geo=geo,
        target_rms=target_rms,
        base=spec.rope_base,
        distances=attention_distances,
        prior=attention_prior,
        starts=starts,
    )
    schedules = {
        "std_geo": geo,
        "native_evq_span_rule": evq,
        "native_exp_span_matched": exp_schedule,
        "exact_kernel_uniform_span_matched": exact_schedule,
        "attention_kernel_stdgeo42_span_matched": attention_schedule,
    }
    records: dict[str, Any] = {}
    for name, phi in schedules.items():
        inv_freq = np.power(spec.rope_base, -phi).astype(np.float32)
        records[name] = {
            "phi": phi.tolist(),
            "phi_sha256_float64": _array_sha256(phi),
            "inv_freq": inv_freq.tolist(),
            "inv_freq_sha256_float32": hashlib.sha256(
                inv_freq.tobytes()
            ).hexdigest(),
            "minimum_gap": float(np.diff(phi).min()),
            "span": float(phi[-1] - phi[0]),
            "deformation_rms_from_std_geo": float(
                np.sqrt(np.mean((phi - geo) ** 2))
            ),
            "uniform_exact_collision": _collision_score(
                phi,
                base=spec.rope_base,
                distances=uniform_distances,
                prior=uniform_prior,
            ),
            "attention_weighted_collision": _collision_score(
                phi,
                base=spec.rope_base,
                distances=attention_distances,
                prior=attention_prior,
            ),
            "uniform_effective_rank": _effective_rank(
                phi,
                base=spec.rope_base,
                distances=uniform_distances,
                prior=uniform_prior,
            ),
            "attention_effective_rank": _effective_rank(
                phi,
                base=spec.rope_base,
                distances=attention_distances,
                prior=attention_prior,
            ),
        }
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "contract": {
            "suite": "shape_l128",
            "head_dim": spec.head_dim,
            "channels": channels,
            "base": spec.rope_base,
            "train_length": spec.train_length,
            "native_grid": "u=k/K",
            "span": float(geo[-1]),
            "rule_tau": tau,
            "matched_deformation_rms": target_rms,
            "optimization": (
                "SLSQP over monotone channel locations with fixed native "
                "span and EVQ-matched RMS deformation"
            ),
            "minimum_gap": 1e-4,
            "evidence_boundary": (
                "fixed schedule derivation only; collision objectives are "
                "not language-model losses"
            ),
        },
        "attention_prior": {
            "artifact": attention_prior_path.name,
            "sha256": sha256_file(attention_prior_path),
            "prior_sha256_float64": attention_record[
                "prior_sha256_float64"
            ],
        },
        "exp_shape_parameter": exp_parameter,
        "exact_kernel_uniform_receipt": exact_receipt,
        "attention_kernel_receipt": attention_receipt,
        "schedules": records,
    }
    _atomic_json(output.resolve(), payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    measure = sub.add_parser("measure-attention")
    measure.add_argument("--checkpoint", type=Path, required=True)
    measure.add_argument("--receipt", type=Path, required=True)
    measure.add_argument("--data_manifest", type=Path, required=True)
    measure.add_argument("--output", type=Path, required=True)
    measure.add_argument("--batch_size", type=int, default=4)
    derive = sub.add_parser("derive")
    derive.add_argument("--attention_prior", type=Path, required=True)
    derive.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "measure-attention":
        payload = measure_attention(
            checkpoint=args.checkpoint,
            receipt=args.receipt,
            data_manifest=args.data_manifest,
            output=args.output,
            batch_size=args.batch_size,
        )
    else:
        payload = derive_schedules(
            attention_prior_path=args.attention_prior,
            output=args.output,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
