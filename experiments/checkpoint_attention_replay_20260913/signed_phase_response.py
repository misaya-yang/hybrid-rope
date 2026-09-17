#!/usr/bin/env python3
"""Exact signed local phase response for frozen Q/K/V captures.

This is a detached diagnostic.  It evaluates fixed tables only; it does not
optimize a table, update model weights, or infer full-model answer quality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .capture_io import load_capture
from .core import ReplayCapture, validate_capture


STATUS = "SIGNED_PHASE_RESPONSE_COMPLETE_V1"


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    weights = np.exp(shifted)
    return weights / weights.sum()


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return maximum + float(np.log(np.exp(values - maximum).sum()))


def _components(query: np.ndarray, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q_left, q_right = np.split(np.asarray(query, dtype=np.float64), 2)
    k_left, k_right = np.split(np.asarray(keys, dtype=np.float64), 2, axis=-1)
    a = k_left * q_left + k_right * q_right
    b = k_right * q_left - k_left * q_right
    return a, b


def _scores(
    query: np.ndarray,
    keys: np.ndarray,
    *,
    query_position: int,
    inv_freq: np.ndarray,
    gain: float,
    attention_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a, b = _components(query, keys)
    lag = query_position - np.arange(len(keys), dtype=np.float64)
    phase = lag[:, None] * np.asarray(inv_freq, dtype=np.float64)[None, :]
    pair_scores = gain ** 2 * attention_scale * (a * np.cos(phase) + b * np.sin(phase))
    return pair_scores.sum(axis=-1), pair_scores, a, b


def _margin(scores: np.ndarray, evidence: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    mask = np.zeros(len(scores), dtype=bool)
    mask[evidence] = True
    if not mask.any() or mask.all():
        raise ValueError("evidence and competitor key sets must both be nonempty")
    return (
        _logsumexp(scores[mask]) - _logsumexp(scores[~mask]),
        _softmax(scores[mask]),
        _softmax(scores[~mask]),
    )


def _score_jacobian(
    query: np.ndarray,
    keys: np.ndarray,
    *,
    query_position: int,
    inv_freq: np.ndarray,
    gain: float,
    attention_scale: float,
) -> np.ndarray:
    a, b = _components(query, keys)
    lag = query_position - np.arange(len(keys), dtype=np.float64)
    phase = lag[:, None] * inv_freq[None, :]
    # u = log(omega_native / omega_new), so d omega / d u = -omega.
    return (
        gain ** 2 * attention_scale * lag[:, None] * inv_freq[None, :]
        * (a * np.sin(phase) - b * np.cos(phase))
    )


def _fmr_linear_contrast(
    query: np.ndarray,
    query_mean: np.ndarray,
    keys: np.ndarray,
    *,
    query_position: int,
    native_inv: np.ndarray,
    candidate_inv: np.ndarray,
    gain: float,
    attention_scale: float,
    evidence: np.ndarray,
) -> np.ndarray:
    key_mean = keys.mean(axis=0)
    q_parts = (query_mean, query_mean, query - query_mean, query - query_mean)
    k_parts = (
        np.broadcast_to(key_mean, keys.shape), keys - key_mean,
        np.broadcast_to(key_mean, keys.shape), keys - key_mean,
    )
    values = []
    competitor = np.setdiff1d(np.arange(len(keys)), evidence, assume_unique=True)
    for q_part, k_part in zip(q_parts, k_parts):
        native, _, _, _ = _scores(
            q_part, k_part, query_position=query_position, inv_freq=native_inv,
            gain=gain, attention_scale=attention_scale,
        )
        candidate, _, _, _ = _scores(
            q_part, k_part, query_position=query_position, inv_freq=candidate_inv,
            gain=gain, attention_scale=attention_scale,
        )
        delta = candidate - native
        values.append(float(delta[evidence].mean() - delta[competitor].mean()))
    return np.asarray(values, dtype=np.float64)


def analyze_capture(capture: ReplayCapture, candidate_inv_freq: np.ndarray, *, gain: float) -> dict:
    record = validate_capture(capture)
    if record.v is None or not record.evidence_key_positions:
        raise ValueError("signed response requires V and preregistered evidence positions")
    candidate_inv = np.asarray(candidate_inv_freq, dtype=np.float64)
    native_inv = np.asarray(record.native_inv_freq, dtype=np.float64)
    if candidate_inv.shape != native_inv.shape or np.any(candidate_inv <= 0):
        raise ValueError("candidate frequency table has the wrong shape or sign")
    if gain != record.reference_gain:
        raise ValueError("native phase study holds RoPE gain fixed")
    repeat = record.q.shape[0] // record.k.shape[0]
    keys = np.repeat(np.asarray(record.k, dtype=np.float64), repeat, axis=0)
    values = np.repeat(np.asarray(record.v, dtype=np.float64), repeat, axis=0)
    query_means = np.asarray(record.q, dtype=np.float64).mean(axis=1)
    pair_count = len(native_inv)
    observations = []
    b_values, h_values, fmr_values = [], [], []
    for head in range(record.q.shape[0]):
        for query_index, query_position in enumerate(record.query_positions):
            stop = int(query_position) + 1
            evidence = np.asarray(record.evidence_key_positions[query_index], dtype=np.int64)
            native_scores, _, _, _ = _scores(
                record.q[head, query_index], keys[head, :stop],
                query_position=int(query_position), inv_freq=native_inv,
                gain=record.reference_gain, attention_scale=record.attention_scale,
            )
            candidate_scores, _, _, _ = _scores(
                record.q[head, query_index], keys[head, :stop],
                query_position=int(query_position), inv_freq=candidate_inv,
                gain=gain, attention_scale=record.attention_scale,
            )
            native_margin, p_e, p_n = _margin(native_scores, evidence)
            candidate_margin, _, _ = _margin(candidate_scores, evidence)
            jacobian = _score_jacobian(
                record.q[head, query_index], keys[head, :stop],
                query_position=int(query_position), inv_freq=native_inv,
                gain=record.reference_gain, attention_scale=record.attention_scale,
            )
            mask = np.zeros(stop, dtype=bool)
            mask[evidence] = True
            b = p_e @ jacobian[mask] - p_n @ jacobian[~mask]
            probability = _softmax(native_scores)
            centered = jacobian - probability @ jacobian
            hessian = centered.T @ (probability[:, None] * centered)
            native_output = probability @ values[head, :stop]
            candidate_probability = _softmax(candidate_scores)
            candidate_output = candidate_probability @ values[head, :stop]
            delta_output = candidate_output - native_output
            fmr = _fmr_linear_contrast(
                record.q[head, query_index], query_means[head], keys[head, :stop],
                query_position=int(query_position), native_inv=native_inv,
                candidate_inv=candidate_inv, gain=gain,
                attention_scale=record.attention_scale, evidence=evidence,
            )
            b_values.append(b)
            h_values.append(hessian)
            fmr_values.append(fmr)
            observations.append({
                "head": head,
                "query_index": query_index,
                "query_role": record.query_roles[query_index],
                "query_position": int(query_position),
                "visible_keys": stop,
                "evidence_keys": int(len(evidence)),
                "native_margin": native_margin,
                "candidate_margin": candidate_margin,
                "delta_margin": candidate_margin - native_margin,
                "native_evidence_mass": float(probability[evidence].sum()),
                "candidate_evidence_mass": float(candidate_probability[evidence].sum()),
                "delta_output_l2": float(np.linalg.norm(delta_output)),
            })
    b_mean = np.mean(b_values, axis=0)
    h_mean = np.mean(h_values, axis=0)
    fmr_mean = np.mean(fmr_values, axis=0)
    if b_mean.shape != (pair_count,) or h_mean.shape != (pair_count, pair_count):
        raise AssertionError("signed response aggregate shape drift")
    return {
        "row_id": record.row_id,
        "layer": record.layer,
        "group": record.group,
        "observations": observations,
        "b_mean": b_mean,
        "h_mean": h_mean,
        "fmr_linear_contrast_mean": fmr_mean,
    }


def _table(path: Path) -> tuple[np.ndarray, float, dict]:
    payload = json.loads(path.read_text())
    table = payload.get("table", payload)
    values = np.asarray(table["values_float32"], dtype=np.float64)
    gain = float(table["gain"])
    return values, gain, payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-index", type=Path, required=True)
    parser.add_argument("--table", action="append", required=True, help="NAME=TABLE_JSON")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    index = json.loads(args.capture_index.read_text())
    captures = [(item, load_capture(Path(item["path"]), mmap_mode="r"))
                for item in index.get("captures", [])]
    if not captures:
        raise ValueError("capture index is empty")
    tables = {}
    for spec in args.table:
        name, separator, raw_path = spec.partition("=")
        if not separator or not name or name in tables:
            raise ValueError("--table must contain unique NAME=PATH values")
        path = Path(raw_path)
        values, gain, payload = _table(path)
        tables[name] = (path, values, gain, payload)
    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "status": STATUS,
        "capture_index_sha256": _sha256(args.capture_index),
        "capture_count": len(captures),
        "tables": {},
        "scope": (
            "Detached current-trajectory Q/K/V response for fixed tables. Local margin, value "
            "readout and KL curvature are not full-model generation effects and do not select a table."
        ),
    }
    for name, (path, values, gain, _payload) in tables.items():
        analyses = [analyze_capture(capture, values, gain=gain) for _, capture in captures]
        b = np.stack([item.pop("b_mean") for item in analyses])
        h = np.stack([item.pop("h_mean") for item in analyses])
        fmr = np.stack([item.pop("fmr_linear_contrast_mean") for item in analyses])
        array_path = args.out / f"{name}_arrays.npz"
        np.savez_compressed(array_path, b_mean=b, h_mean=h, fmr_linear_contrast=fmr)
        observations = [row for item in analyses for row in item["observations"]]
        report["tables"][name] = {
            "table_path": str(path.resolve()),
            "table_sha256": _sha256(path),
            "arrays": array_path.name,
            "arrays_sha256": _sha256(array_path),
            "captures": analyses,
            "summary": {
                "observations": len(observations),
                "mean_delta_margin": float(np.mean([row["delta_margin"] for row in observations])),
                "mean_delta_evidence_mass": float(np.mean([
                    row["candidate_evidence_mass"] - row["native_evidence_mass"]
                    for row in observations
                ])),
                "mean_delta_output_l2": float(np.mean([
                    row["delta_output_l2"] for row in observations
                ])),
                "mean_b": b.mean(axis=0).tolist(),
                "mean_hessian": h.mean(axis=0).tolist(),
                "fmr_linear_contrast_components": {
                    "muq_muk": float(fmr[:, 0].mean()),
                    "muq_rk": float(fmr[:, 1].mean()),
                    "rq_muk": float(fmr[:, 2].mean()),
                    "rq_rk": float(fmr[:, 3].mean()),
                    "identity": (
                        "Exact bilinear decomposition of the mean candidate-minus-native "
                        "evidence-vs-competitor score contrast within each capture; not an "
                        "additive decomposition of logsumexp margin or model semantics."
                    ),
                },
            },
        }
    temporary = args.out / "report.json.incomplete"
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.out / "report.json")
    print(json.dumps({"status": STATUS, "tables": sorted(tables), "captures": len(captures)}))


if __name__ == "__main__":
    main()
