#!/usr/bin/env python3
"""Checkpoint-only FMRoPE/EVQ frequency-band prediction and p-RoPE audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rebuttal.rebuttal_0723.experiments.fmrope_125m_l256_500m.protocol import (
    SPEC as SOURCE_SPEC,
)
from rebuttal.rebuttal_0723.experiments.fmrope_125m_l256_500m.run_experiment import (
    tensor_sha256,
)
from rebuttal.rebuttal_0723.experiments.profiled_residual_5090.run_experiment import (
    _bootstrap_ci,
    _evaluate_schedule,
    _load_model,
    _load_ready,
    _set_rope,
    _validate_cuda,
)


ARMS = (
    "paper_geo_base500k",
    "evq_cosh_tau4_paper_grid_base500k",
    "fmrope_base256",
)
LENGTHS = (256, 1_024, 4_096, 8_192)
X_STARS = (3.657, 4.493)


@dataclass(frozen=True)
class BandSpec:
    source_protocol_sha256: str = SOURCE_SPEC.fingerprint()
    train_length: int = SOURCE_SPEC.train_length
    head_dim: int = SOURCE_SPEC.head_dim
    arms: tuple[str, ...] = ARMS
    lengths: tuple[int, ...] = LENGTHS
    x_stars: tuple[float, ...] = X_STARS
    band_width_pairs: int = 3
    max_batch_tokens: int = 8_192
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 20_260_724

    def fingerprint(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()


SPEC = BandSpec()
ROOT = Path(__file__).resolve().parents[3]


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_fingerprint() -> str:
    paths = (
        Path(__file__).resolve(),
        ROOT / "rebuttal/rebuttal_0723/experiments/run_frequency_band_usage_5090.sh",
        ROOT / "rebuttal/rebuttal_0723/experiments/profiled_residual_5090/protocol.py",
        ROOT / "rebuttal/rebuttal_0723/experiments/profiled_residual_5090/run_experiment.py",
        ROOT / "experiments/native_rope_evq_150m/model.py",
        ROOT / "tests/test_frequency_band_usage_5090.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(bytes.fromhex(_file_sha256(path)))
    return digest.hexdigest()


def predicted_band_indices(inv_freq: torch.Tensor) -> dict[str, int]:
    """Nearest discrete channel to the two FMRoPE phase predictors."""
    inv = inv_freq.detach().cpu().double().contiguous()
    if inv.shape != (SPEC.head_dim // 2,) or not torch.all(inv > 0):
        raise ValueError("invalid training inverse-frequency table")
    phase = SPEC.train_length * inv
    return {
        str(value): int(torch.argmin(torch.abs(phase - value)))
        for value in SPEC.x_stars
    }


def intervention_windows(center: int, *, pairs: int = 32, width: int = 3) -> dict[str, list[int]]:
    """Return non-overlapping empirical-band, adjacent, and far controls."""
    if not 0 <= int(center) < int(pairs) or not 0 < int(width) * 3 <= int(pairs):
        raise ValueError("invalid band/window geometry")
    start = min(max(int(center) - int(width) // 2, 0), int(pairs) - int(width))
    band = tuple(range(start, start + int(width)))
    candidates = [
        tuple(range(index, index + int(width)))
        for index in range(0, int(pairs) - int(width) + 1)
        if set(range(index, index + int(width))).isdisjoint(band)
    ]
    band_mid = statistics.fmean(band)
    candidates.sort(key=lambda row: abs(statistics.fmean(row) - band_mid))
    adjacent = candidates[0]
    far = max(candidates, key=lambda row: abs(statistics.fmean(row) - band_mid))
    if not (set(band).isdisjoint(adjacent) and set(band).isdisjoint(far)):
        raise RuntimeError("intervention controls overlap the selected band")
    return {
        "band_nope": list(band),
        "adjacent_nope": list(adjacent),
        "far_nope": list(far),
    }


def apply_nope(inv_freq: torch.Tensor, indices: list[int]) -> torch.Tensor:
    """Set chosen rotary pairs to exact NoPE while preserving every other pair."""
    value = inv_freq.detach().clone().contiguous()
    if len(indices) != len(set(indices)) or any(not 0 <= index < value.numel() for index in indices):
        raise ValueError("invalid NoPE pair indices")
    value[indices] = 0.0
    return value


def runtime_inv_freq(training: torch.Tensor, arm: str, length: int) -> torch.Tensor:
    training = training.detach().cpu().float().clone()
    if arm != "fmrope_base256" or int(length) == SPEC.train_length:
        return training
    index = torch.arange(training.numel(), dtype=torch.float64)
    return torch.pow(float(length), -index / float(training.numel())).float()


@torch.inference_mode()
def collect_qk_profile(
    model: torch.nn.Module,
    validation: np.ndarray,
    anchors: np.ndarray,
) -> dict[str, Any]:
    """Collect pre-RoPE Q/K pair norms without materializing vocabulary logits."""
    layers = len(model.blocks)
    heads = int(model.config["num_heads"])
    pairs = int(model.config["head_dim"]) // 2
    sums = torch.zeros((layers, 2, heads, pairs), dtype=torch.float64)
    counts = torch.zeros(layers, dtype=torch.int64)
    hooks = []

    for layer, block in enumerate(model.blocks):
        def capture(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor, layer: int = layer) -> None:
            batch, length, _ = output.shape
            qkv = output.view(batch, length, 3, heads, pairs * 2)
            for kind in range(2):
                value = qkv[:, :, kind].float()
                pair_norm = torch.sqrt(value[..., :pairs].square() + value[..., pairs:].square())
                sums[layer, kind] += pair_norm.sum(dim=(0, 1)).double().cpu()
            counts[layer] += batch * length

        hooks.append(block.attention.qkv.register_forward_hook(capture))

    try:
        batch_size = max(1, SPEC.max_batch_tokens // SPEC.train_length)
        for start in range(0, len(anchors), batch_size):
            endpoints = anchors[start : start + batch_size]
            windows = np.stack(
                [
                    np.asarray(
                        validation[int(end) - SPEC.train_length : int(end)],
                        dtype=np.int64,
                    )
                    for end in endpoints
                ]
            )
            tokens = torch.from_numpy(windows).to("cuda", non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = model.embedding(tokens)
                hidden = hidden.to(dtype=torch.bfloat16)
                for block in model.blocks:
                    hidden = block(hidden)
            del tokens, hidden
    finally:
        for hook in hooks:
            hook.remove()

    if torch.any(counts <= 0):
        raise RuntimeError("Q/K hook did not observe every layer")
    means = sums / counts[:, None, None, None]
    aggregate = means.mean(dim=(0, 1, 2))
    per_layer_kind = means.mean(dim=2)
    return {
        "aggregate_pair_norm": [float(value) for value in aggregate],
        "empirical_band_index": int(torch.argmax(aggregate)),
        "per_layer_q_band_index": [
            int(torch.argmax(per_layer_kind[layer, 0])) for layer in range(layers)
        ],
        "per_layer_k_band_index": [
            int(torch.argmax(per_layer_kind[layer, 1])) for layer in range(layers)
        ],
        "observations_per_layer": [int(value) for value in counts],
    }


def preflight(source_ready_dir: Path, output_dir: Path) -> dict[str, Any]:
    source = _load_ready(source_ready_dir)
    if any(arm not in source["checkpoints"] for arm in SPEC.arms):
        raise ValueError("source READY lacks a registered checkpoint")
    result = {
        "status": "READY",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "protocol_sha256": SPEC.fingerprint(),
        "code_sha256": code_fingerprint(),
        "source_ready": str((source_ready_dir / "READY.json").resolve()),
        "source_ready_sha256": _file_sha256(source_ready_dir / "READY.json"),
        "source_profile_protocol_sha256": source["protocol_sha256"],
        "validation": source["validation"],
        "anchors": source["anchors"],
        "checkpoints": {arm: source["checkpoints"][arm] for arm in SPEC.arms},
        "claim_boundary": "single-seed checkpoint-only oracle diagnostic; no task or capability claim",
    }
    _atomic_json(output_dir / "READY.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _load_band_ready(output_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    ready = json.loads((output_dir / "READY.json").read_text())
    if ready.get("status") != "READY" or ready.get("protocol_sha256") != SPEC.fingerprint():
        raise ValueError("band READY protocol mismatch")
    if ready.get("code_sha256") != code_fingerprint():
        raise ValueError("band code changed after preflight")
    source_path = Path(ready["source_ready"])
    if _file_sha256(source_path) != ready["source_ready_sha256"]:
        raise ValueError("source READY changed after preflight")
    return ready, _load_ready(source_path.parent)


def run(output_dir: Path) -> dict[str, Any]:
    runtime = _validate_cuda()
    ready, source = _load_band_ready(output_dir)
    validation = np.load(ready["validation"]["path"], mmap_mode="r", allow_pickle=False).reshape(-1)
    anchors = {
        split: np.load(record["path"], allow_pickle=False)
        for split, record in ready["anchors"].items()
    }
    raw: dict[str, Any] = {"runtime": runtime, "arms": {}}

    for arm in SPEC.arms:
        model, _ = _load_model(source, arm)
        model = model.to("cuda").eval()
        training_inv = model.blocks[0].attention.rope.inv_freq.detach().cpu().float().clone()
        _set_rope(model, training_inv, SPEC.train_length)
        selection_profile = collect_qk_profile(model, validation, anchors["selection"])
        test_profile = collect_qk_profile(model, validation, anchors["test"])
        selected_center = int(selection_profile["empirical_band_index"])
        masks = intervention_windows(selected_center, pairs=training_inv.numel(), width=SPEC.band_width_pairs)
        arm_result: dict[str, Any] = {
            "checkpoint_sha256": ready["checkpoints"][arm]["checkpoint_sha256_from_train_receipt"],
            "training_inv_freq_sha256": tensor_sha256(training_inv),
            "predicted_band_indices": predicted_band_indices(training_inv),
            "selection_qk": selection_profile,
            "test_qk": test_profile,
            "masks": masks,
            "lengths": {},
        }
        for length in SPEC.lengths:
            inv = runtime_inv_freq(training_inv, arm, length)
            conditions = {"full_rope": inv}
            conditions.update({name: apply_nope(inv, mask) for name, mask in masks.items()})
            values = {
                name: _evaluate_schedule(
                    model,
                    validation,
                    anchors["test"],
                    inv_freq=value,
                    length=length,
                    max_batch_tokens=SPEC.max_batch_tokens,
                )
                for name, value in conditions.items()
            }
            baseline = values["full_rope"]
            arm_result["lengths"][str(length)] = {
                name: {
                    "mean_tail_nll": statistics.fmean(rows),
                    "delta_vs_full_rope": statistics.fmean(
                        [left - right for left, right in zip(rows, baseline, strict=True)]
                    ),
                    "paired_anchor_bootstrap_95ci": _bootstrap_ci(
                        [left - right for left, right in zip(rows, baseline, strict=True)]
                    ),
                    "tail_nll": rows,
                }
                for name, rows in values.items()
            }
        raw["arms"][arm] = arm_result
        del model
        torch.cuda.empty_cache()

    _atomic_json(output_dir / "raw.json", raw)
    return summarize(output_dir, ready, raw)


def summarize(output_dir: Path, ready: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    predictor_hits = 0
    causal_wins = 0
    in_domain_wins = 0
    rows = []
    bands = []
    for arm, value in raw["arms"].items():
        empirical = int(value["test_qk"]["empirical_band_index"])
        predictions = [int(index) for index in value["predicted_band_indices"].values()]
        error = min(abs(empirical - index) for index in predictions)
        predictor_hits += error <= 3
        bands.append({"arm": arm, "empirical": empirical, "predictions": predictions, "min_abs_error": error})
        for length, conditions in value["lengths"].items():
            band_delta = conditions["band_nope"]["delta_vs_full_rope"]
            controls = [
                conditions["adjacent_nope"]["delta_vs_full_rope"],
                conditions["far_nope"]["delta_vs_full_rope"],
            ]
            win = band_delta > max(controls)
            causal_wins += win
            in_domain_wins += win and int(length) == SPEC.train_length
            rows.append(
                {
                    "arm": arm,
                    "length": int(length),
                    "band_delta_nll": band_delta,
                    "adjacent_delta_nll": controls[0],
                    "far_delta_nll": controls[1],
                    "band_is_largest": win,
                }
            )
    proceed = predictor_hits >= 2 and in_domain_wins >= 2 and causal_wins >= len(rows) // 2
    result = {
        "status": "PASS",
        "protocol_sha256": SPEC.fingerprint(),
        "code_sha256": code_fingerprint(),
        "test_anchor_sha256": ready["anchors"]["test"]["sha256"],
        "decision": "SUPPORTS_BAND_BRIDGE" if proceed else "DOES_NOT_SUPPORT_BAND_BRIDGE",
        "single_seed_supporting": True,
        "paper_claim": False,
        "oracle_diagnostic": True,
        "predictor_hits_within_3_pairs": predictor_hits,
        "causal_band_wins": causal_wins,
        "causal_comparisons": len(rows),
        "in_domain_band_wins": in_domain_wins,
        "bands": bands,
        "ablation_rows": rows,
    }
    _atomic_json(output_dir / "summary.json", result)
    lines = [
        "# Generalized frequency-band and p-RoPE diagnostic",
        "",
        f"Decision: **{result['decision']}**.",
        "",
        "The band predictor is an FMRoPE-derived proxy. p-RoPE sets exactly three rotary pairs to NoPE; positive delta means worse NLL after deletion.",
        "",
        "| checkpoint | empirical band | predicted bands | min error |",
        "| --- | ---: | --- | ---: |",
    ]
    for row in bands:
        lines.append(f"| {row['arm']} | {row['empirical']} | {row['predictions']} | {row['min_abs_error']} |")
    lines.extend(["", "| checkpoint | length | band delta | adjacent delta | far delta | band largest |", "| --- | ---: | ---: | ---: | ---: | --- |"])
    for row in rows:
        lines.append(
            f"| {row['arm']} | {row['length']} | {row['band_delta_nll']:+.5f} | "
            f"{row['adjacent_delta_nll']:+.5f} | {row['far_delta_nll']:+.5f} | {row['band_is_largest']} |"
        )
    lines.extend(["", "Single-seed checkpoint-only oracle diagnostic; not a downstream capability or paper-level claim.", ""])
    (output_dir / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--source-ready-dir", type=Path, required=True)
    pre.add_argument("--output-dir", type=Path, required=True)
    execute = sub.add_parser("run")
    execute.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        preflight(args.source_ready_dir.resolve(), args.output_dir.resolve())
    else:
        run(args.output_dir.resolve())


if __name__ == "__main__":
    main()
