#!/usr/bin/env python3
"""Build the Track-Z linchpin figure from immutable readout trace files."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.lora_evq_v2.eval_official_yarn_capability import sha256_file
from experiments.lora_evq_v2.eval_sparse_conversion import (
    ASSOCIATION_SWAP_MANIFEST_SCHEMA,
    ASSOCIATION_SWAP_TRACE_SCHEMA,
    READOUT_TRACE_MANIFEST_SCHEMA,
    READOUT_TRACE_SCHEMA,
    causal_delta_rank,
)


SUMMARY_SCHEMA = "evq_cosh.readout_conversion_linchpin_summary.v1"
REQUIRED_SUBSTRATES = ("native_geo", "evq_cosh")


def _manifest(path: Path, schema: str) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != schema or document.get("status") != "complete":
        raise ValueError(f"incomplete or incompatible trace manifest: {path.name}")
    if document.get("measurement_label") != "oracle-diagnostic":
        raise ValueError(f"trace manifest has the wrong measurement label: {path.name}")
    if document.get("substrate") not in REQUIRED_SUBSTRATES:
        raise ValueError(f"trace manifest has an unknown substrate: {path.name}")
    if not isinstance(document.get("records"), list) or not document["records"]:
        raise ValueError(f"trace manifest has no records: {path.name}")
    return document


def _record(manifest_path: Path, item: Mapping[str, Any]) -> dict[str, Any]:
    relative = Path(str(item.get("file", "")))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("trace manifest contains an unsafe record path")
    root = manifest_path.parent.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("trace record escapes its manifest directory") from exc
    if not path.is_file() or sha256_file(path) != item.get("sha256"):
        raise ValueError(f"trace record receipt mismatch: {relative}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"trace record is not a mapping: {relative}")
    return payload


def _paired_manifests(
    paths: Sequence[Path], schema: str
) -> dict[str, tuple[Path, dict[str, Any]]]:
    manifests: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in paths:
        document = _manifest(path, schema)
        substrate = str(document["substrate"])
        if substrate in manifests:
            raise ValueError(f"duplicate {substrate} manifest")
        manifests[substrate] = (path, document)
    if tuple(sorted(manifests)) != tuple(sorted(REQUIRED_SUBSTRATES)):
        raise ValueError("analysis requires one matched Geo and one matched EVQ manifest")
    return manifests


def _causal_measurements(
    manifests: Mapping[str, tuple[Path, Mapping[str, Any]]]
) -> tuple[list[dict[str, Any]], dict[str, set[tuple[Any, ...]]]]:
    measurements = []
    case_keys: dict[str, set[tuple[Any, ...]]] = defaultdict(set)
    for substrate, (manifest_path, manifest) in manifests.items():
        for item in manifest["records"]:
            payload = _record(manifest_path, item)
            if payload.get("schema") != READOUT_TRACE_SCHEMA:
                raise ValueError("causal trace record schema mismatch")
            if payload.get("substrate") != substrate:
                raise ValueError("causal trace substrate mismatch")
            full = payload.get("full_logits")
            ablated = payload.get("ablated_logits")
            labels = payload.get("gold_token_ids")
            layer_indices = payload.get("layer_indices")
            if not all(torch.is_tensor(value) for value in (full, ablated, labels, layer_indices)):
                raise ValueError("causal trace record is missing tensor fields")
            if full.ndim != 3 or labels.ndim != 1 or layer_indices.ndim != 1:
                raise ValueError("causal trace tensor rank mismatch")
            if full.shape != ablated.shape or full.shape[:2] != (
                labels.numel(),
                layer_indices.numel(),
            ):
                raise ValueError("causal trace tensor shape mismatch")
            ranks = causal_delta_rank(full, ablated, labels)
            delta = full.float() - ablated.float()
            target_ids = labels.long().view(-1, 1, 1).expand(-1, full.shape[1], 1)
            target_delta = delta.gather(-1, target_ids).squeeze(-1)
            depth = float(payload["depth_percent"])
            case_keys[substrate].add(
                (
                    str(payload["prompt_sha256"]),
                    int(payload["target_length"]),
                    depth,
                    int(labels.numel()),
                    tuple(int(value) for value in layer_indices.tolist()),
                )
            )
            for position in range(labels.numel()):
                for layer_offset, layer in enumerate(layer_indices.tolist()):
                    measurements.append(
                        {
                            "substrate": substrate,
                            "answer_position_j": position + 1,
                            "layer": int(layer),
                            "depth_percent": depth,
                            "causal_delta_rank": int(ranks[position, layer_offset]),
                            "gold_causal_delta_logit": float(target_delta[position, layer_offset]),
                        }
                    )
    if case_keys["native_geo"] != case_keys["evq_cosh"]:
        raise ValueError("causal traces are not matched case-for-case across Geo and EVQ")
    return measurements, case_keys


def scalar_alpha_interval(
    logits: torch.Tensor,
    delta: torch.Tensor,
    gold_token_id: int,
) -> tuple[float, float] | None:
    """Return the non-negative alpha interval making gold weakly top-1."""
    if logits.ndim != 1 or delta.shape != logits.shape:
        raise ValueError("scalar feasibility requires matched vocabulary vectors")
    gold = int(gold_token_id)
    if not 0 <= gold < logits.numel():
        raise ValueError("scalar-feasibility gold token is outside the vocabulary")
    z = logits.float()
    d = delta.float()
    slopes = d[gold] - d
    gaps = z - z[gold]
    competitor = torch.ones_like(slopes, dtype=torch.bool)
    competitor[gold] = False
    zero = competitor & (slopes == 0)
    if bool((zero & (gaps > 0)).any()):
        return None
    positive = competitor & (slopes > 0)
    negative = competitor & (slopes < 0)
    lower = 0.0
    upper = float("inf")
    if bool(positive.any()):
        lower = max(lower, float((gaps[positive] / slopes[positive]).max()))
    if bool(negative.any()):
        upper = float((gaps[negative] / slopes[negative]).min())
    lower = max(0.0, lower)
    return (lower, upper) if lower <= upper else None


def _swap_measurements(
    manifests: Mapping[str, tuple[Path, Mapping[str, Any]]]
) -> tuple[
    list[dict[str, Any]],
    dict[str, set[tuple[Any, ...]]],
    list[dict[str, Any]],
]:
    measurements = []
    scalar_rows = []
    pair_keys: dict[str, set[tuple[Any, ...]]] = defaultdict(set)
    for substrate, (manifest_path, manifest) in manifests.items():
        for item in manifest["records"]:
            payload = _record(manifest_path, item)
            if payload.get("schema") != ASSOCIATION_SWAP_TRACE_SCHEMA:
                raise ValueError("association-swap trace record schema mismatch")
            if payload.get("substrate") != substrate:
                raise ValueError("association-swap trace substrate mismatch")
            full = payload.get("full_logits")
            ablated = payload.get("ablated_logits")
            candidates = payload.get("candidate_first_token_ids")
            layer_indices = payload.get("layer_indices")
            if not all(torch.is_tensor(value) for value in (full, ablated, candidates, layer_indices)):
                raise ValueError("association-swap trace is missing tensor fields")
            if full.ndim != 3 or full.shape[0] != 2 or candidates.shape != (2,):
                raise ValueError("association-swap trace tensor rank mismatch")
            if full.shape != ablated.shape or full.shape[1] != layer_indices.numel():
                raise ValueError("association-swap trace tensor shape mismatch")
            if bool(((candidates < 0) | (candidates >= full.shape[-1])).any()):
                raise ValueError("association-swap candidate token is outside the vocabulary")
            split = str(payload.get("split"))
            if split not in {"dev", "test"}:
                raise ValueError("association-swap split must be dev or test")
            delta = full.float() - ablated.float()
            token_a, token_b = (int(value) for value in candidates.tolist())
            scores = (
                delta[0, :, token_a]
                - delta[0, :, token_b]
                - delta[1, :, token_a]
                + delta[1, :, token_b]
            )
            depth = float(payload["depth_percent"])
            pair_keys[substrate].add(
                (
                    str(payload["pair_sha256"]),
                    split,
                    depth,
                    tuple(int(value) for value in layer_indices.tolist()),
                )
            )
            for layer_offset, layer in enumerate(layer_indices.tolist()):
                measurements.append(
                    {
                        "substrate": substrate,
                        "split": split,
                        "layer": int(layer),
                        "depth_percent": depth,
                        "swap_follow_score": float(scores[layer_offset]),
                    }
                )
            for condition, gold_token_id in enumerate((token_a, token_b)):
                interval = scalar_alpha_interval(
                    full[condition, -1],
                    delta[condition, -1],
                    gold_token_id,
                )
                scalar_rows.append(
                    {
                        "substrate": substrate,
                        "split": split,
                        "depth_percent": depth,
                        "feasible": interval is not None,
                        "lower_bound": None if interval is None else interval[0],
                    }
                )
    if pair_keys["native_geo"] != pair_keys["evq_cosh"]:
        raise ValueError("swap traces are not matched pair-for-pair across Geo and EVQ")
    return measurements, pair_keys, scalar_rows


def _aggregate_causal(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: defaultdict[tuple[Any, ...], list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        key = (
            row["substrate"],
            row["answer_position_j"],
            row["layer"],
            row["depth_percent"],
        )
        groups[key].append((row["causal_delta_rank"], row["gold_causal_delta_logit"]))
    output = []
    for (substrate, position, layer, depth), values in sorted(groups.items()):
        ranks = np.asarray([value[0] for value in values], dtype=np.float64)
        deltas = np.asarray([value[1] for value in values], dtype=np.float64)
        output.append(
            {
                "substrate": substrate,
                "answer_position_j": int(position),
                "layer": int(layer),
                "depth_percent": float(depth),
                "count": len(values),
                "median_causal_delta_rank": float(np.median(ranks)),
                "mean_causal_delta_rank": float(ranks.mean()),
                "median_gold_causal_delta_logit": float(np.median(deltas)),
            }
        )
    return output


def _bootstrap_mean_ci(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    if values.size == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        means[index] = rng.choice(values, size=values.size, replace=True).mean()
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def _aggregate_swap(
    rows: Sequence[Mapping[str, Any]], bootstrap_samples: int
) -> list[dict[str, Any]]:
    groups: defaultdict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        key = (row["substrate"], row["split"], row["layer"], row["depth_percent"])
        groups[key].append(row["swap_follow_score"])
    output = []
    for group_index, ((substrate, split, layer, depth), values) in enumerate(
        sorted(groups.items())
    ):
        scores = np.asarray(values, dtype=np.float64)
        ci_low, ci_high = _bootstrap_mean_ci(
            scores,
            samples=bootstrap_samples,
            seed=42 + group_index,
        )
        output.append(
            {
                "substrate": substrate,
                "split": split,
                "layer": int(layer),
                "depth_percent": float(depth),
                "count": len(values),
                "mean_swap_follow_score": float(scores.mean()),
                "median_swap_follow_score": float(np.median(scores)),
                "positive_fraction": float((scores > 0).mean()),
                "mean_ci95": [ci_low, ci_high],
            }
        )
    return output


def _aggregate_final_layer_swap(
    rows: Sequence[Mapping[str, Any]], bootstrap_samples: int
) -> list[dict[str, Any]]:
    final_layer = max(int(row["layer"]) for row in rows)
    groups: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if int(row["layer"]) == final_layer:
            groups[(str(row["substrate"]), str(row["split"]))].append(
                float(row["swap_follow_score"])
            )
    output = []
    for group_index, ((substrate, split), values) in enumerate(sorted(groups.items())):
        scores = np.asarray(values, dtype=np.float64)
        low, high = _bootstrap_mean_ci(
            scores,
            samples=bootstrap_samples,
            seed=4200 + group_index,
        )
        output.append(
            {
                "substrate": substrate,
                "split": split,
                "layer": final_layer,
                "count": len(values),
                "mean_swap_follow_score": float(scores.mean()),
                "median_swap_follow_score": float(np.median(scores)),
                "positive_fraction": float((scores > 0).mean()),
                "mean_ci95": [low, high],
            }
        )
    return output


def _aggregate_scalar_feasibility(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: defaultdict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["substrate"]), str(row["split"]))].append(row)
    output = []
    for (substrate, split), values in sorted(groups.items()):
        feasible = [row for row in values if bool(row["feasible"])]
        output.append(
            {
                "substrate": substrate,
                "split": split,
                "sample_count": len(values),
                "feasible_fraction": len(feasible) / len(values),
                "infeasible_fraction": 1.0 - len(feasible) / len(values),
                "median_feasible_lower_bound": (
                    None
                    if not feasible
                    else float(
                        np.median(
                            [float(row["lower_bound"]) for row in feasible]
                        )
                    )
                ),
            }
        )
    return output


def _plot_linchpin(
    causal: Sequence[Mapping[str, Any]],
    swap: Sequence[Mapping[str, Any]],
    output: Path,
) -> None:
    positions = sorted({int(row["answer_position_j"]) for row in causal})
    maximum = max(float(row["median_causal_delta_rank"]) for row in causal)
    figure, axes = plt.subplots(
        len(positions),
        len(REQUIRED_SUBSTRATES),
        figsize=(11, max(3.4, 2.8 * len(positions))),
        squeeze=False,
        constrained_layout=True,
    )
    image = None
    for column, substrate in enumerate(REQUIRED_SUBSTRATES):
        substrate_rows = [row for row in causal if row["substrate"] == substrate]
        layers = sorted({int(row["layer"]) for row in substrate_rows})
        depths = sorted({float(row["depth_percent"]) for row in substrate_rows})
        layer_offsets = {value: index for index, value in enumerate(layers)}
        depth_offsets = {value: index for index, value in enumerate(depths)}
        for row_offset, position in enumerate(positions):
            axis = axes[row_offset, column]
            matrix = np.full((len(depths), len(layers)), np.nan)
            for row in substrate_rows:
                if int(row["answer_position_j"]) == position:
                    matrix[depth_offsets[float(row["depth_percent"])], layer_offsets[int(row["layer"])]] = np.log10(
                        max(1.0, float(row["median_causal_delta_rank"]))
                    )
            image = axis.imshow(
                matrix,
                origin="lower",
                aspect="auto",
                cmap="viridis",
                vmin=0.0,
                vmax=max(1.0, np.log10(maximum)),
            )
            if position == 1:
                overlay = [
                    row
                    for row in swap
                    if row["substrate"] == substrate and row["split"] == "test"
                ]
                for row in overlay:
                    layer = int(row["layer"])
                    depth = float(row["depth_percent"])
                    if layer not in layer_offsets or depth not in depth_offsets:
                        raise ValueError("test swap cells do not align with causal layer/depth cells")
                    positive = float(row["mean_swap_follow_score"]) > 0.0
                    axis.scatter(
                        layer_offsets[layer],
                        depth_offsets[depth],
                        marker="o" if positive else "x",
                        facecolors="none" if positive else "#d81b60",
                        edgecolors="#ffffff" if positive else None,
                        color=None if positive else "#d81b60",
                        linewidths=0.8,
                        s=28,
                    )
            axis.set_xticks(range(len(layers)), labels=layers, fontsize=7)
            axis.set_yticks(range(len(depths)), labels=[f"{depth:g}" for depth in depths])
            axis.set_xlabel("decoder layer l")
            if column == 0:
                axis.set_ylabel(f"depth d (%)\nanswer position j={position}")
            if row_offset == 0:
                axis.set_title("Geo" if substrate == "native_geo" else "EVQ-Cosh")
    if image is None:
        raise ValueError("linchpin figure has no causal cells")
    figure.colorbar(image, ax=axes, label="log10 median causal-delta rank")
    figure.suptitle(
        "Causal-delta rank by answer position, layer, and depth\n"
        "test swap overlay: white circle = positive mean; magenta x = non-positive"
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)


def analyze(
    causal_manifests: Sequence[Path],
    swap_manifests: Sequence[Path],
    output_dir: Path,
    *,
    bootstrap_samples: int = 10_000,
    enforce_planned_counts: bool = True,
) -> dict[str, Any]:
    """Validate matched traces, aggregate metrics, and write sanitized outputs."""
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    temporary_dir = output_dir.with_name(output_dir.name + ".incomplete")
    if output_dir.exists() or temporary_dir.exists():
        raise FileExistsError(output_dir if output_dir.exists() else temporary_dir)

    causal_docs = _paired_manifests(causal_manifests, READOUT_TRACE_MANIFEST_SCHEMA)
    swap_docs = _paired_manifests(swap_manifests, ASSOCIATION_SWAP_MANIFEST_SCHEMA)
    causal_rows, causal_keys = _causal_measurements(causal_docs)
    swap_rows, swap_keys, scalar_rows = _swap_measurements(swap_docs)
    if enforce_planned_counts:
        if len(causal_keys["native_geo"]) != 10:
            raise ValueError("linchpin analysis requires the ten registered causal cases")
        split_counts = {
            split: sum(key[1] == split for key in swap_keys["native_geo"])
            for split in ("dev", "test")
        }
        if split_counts != {"dev": 128, "test": 128}:
            raise ValueError("linchpin analysis requires 128 dev and 128 test swap pairs")
    causal = _aggregate_causal(causal_rows)
    swap = _aggregate_swap(swap_rows, bootstrap_samples)
    final_layer_swap = _aggregate_final_layer_swap(swap_rows, bootstrap_samples)
    scalar_feasibility = _aggregate_scalar_feasibility(scalar_rows)

    temporary_dir.mkdir(parents=True)
    figure_name = "linchpin_causal_delta_rank.png"
    _plot_linchpin(causal, swap, temporary_dir / figure_name)
    inputs = []
    for kind, documents in (("causal", causal_docs), ("association_swap", swap_docs)):
        for substrate in REQUIRED_SUBSTRATES:
            path, document = documents[substrate]
            inputs.append(
                {
                    "kind": kind,
                    "substrate": substrate,
                    "manifest_sha256": sha256_file(path),
                    "record_count": len(document["records"]),
                }
            )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "complete",
        "measurement_labels": {
            "causal_delta_rank": "oracle-diagnostic",
            "association_swap": "oracle-diagnostic",
        },
        "single_seed_supporting": True,
        "paper_claim": False,
        "answer_position_indexing": "one_based",
        "matched_case_count": len(causal_keys["native_geo"]),
        "matched_swap_pair_count": len(swap_keys["native_geo"]),
        "bootstrap": {"samples": bootstrap_samples, "seed_base": 42},
        "inputs": inputs,
        "causal_delta_rank": causal,
        "association_swap": swap,
        "association_swap_final_layer": final_layer_swap,
        "oracle_scalar_feasibility": scalar_feasibility,
        "figure": figure_name,
    }
    temporary = temporary_dir / "summary.json.incomplete"
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, temporary_dir / "summary.json")
    os.replace(temporary_dir, output_dir)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--causal-manifest", action="append", type=Path, required=True)
    parser.add_argument("--swap-manifest", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = analyze(
        args.causal_manifest,
        args.swap_manifest,
        args.output_dir,
        bootstrap_samples=args.bootstrap_samples,
    )
    print(json.dumps({key: summary[key] for key in ("schema", "status", "figure")}, indent=2))


if __name__ == "__main__":
    main()
