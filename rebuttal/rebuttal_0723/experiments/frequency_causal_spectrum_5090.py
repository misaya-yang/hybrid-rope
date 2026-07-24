#!/usr/bin/env python3
"""Per-pair causal spectrum and frequency-assignment audit for frozen RoPE models."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rebuttal.rebuttal_0723.experiments.frequency_band_usage_5090 import (
    ARMS,
    LENGTHS,
    SPEC as BAND_SPEC,
    _atomic_json,
    _file_sha256,
    apply_nope,
    code_fingerprint as band_code_fingerprint,
    runtime_inv_freq,
)
from rebuttal.rebuttal_0723.experiments.profiled_residual_5090.run_experiment import (
    _bootstrap_ci,
    _evaluate_schedule,
    _load_model,
    _load_ready,
    _validate_cuda,
)


ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = 1


def code_fingerprint() -> str:
    paths = (
        Path(__file__).resolve(),
        ROOT / "rebuttal/rebuttal_0723/experiments/run_frequency_causal_spectrum_5090.sh",
        ROOT / "rebuttal/rebuttal_0723/experiments/frequency_band_usage_5090.py",
        ROOT / "tests/test_frequency_causal_spectrum_5090.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(bytes.fromhex(_file_sha256(path)))
    return digest.hexdigest()


def swap_pairs(inv_freq: torch.Tensor, left: list[int], right: list[int]) -> torch.Tensor:
    if len(left) != len(right) or set(left) & set(right):
        raise ValueError("swap windows must be equal-sized and disjoint")
    value = inv_freq.detach().clone().contiguous()
    before = torch.sort(value).values
    saved = value[left].clone()
    value[left] = value[right]
    value[right] = saved
    if not torch.equal(torch.sort(value).values, before):
        raise RuntimeError("frequency swap changed the multiset")
    return value


def phase_utility(inv_freq: torch.Tensor, length: int) -> np.ndarray:
    """Discrete maximum eigenvalue of Cov[cos(mw), sin(mw)] per pair."""
    omega = inv_freq.detach().cpu().double()
    positions = torch.arange(int(length), dtype=torch.float64)[:, None]
    phase = positions * omega[None]
    features = torch.stack((phase.cos(), phase.sin()), dim=-1)
    centered = features - features.mean(dim=0, keepdim=True)
    covariance = torch.einsum("lki,lkj->kij", centered, centered) / float(length)
    return torch.linalg.eigvalsh(covariance)[:, -1].numpy()


def spearman(left: list[float] | np.ndarray, right: list[float] | np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or a.size < 3:
        raise ValueError("Spearman inputs must be matched one-dimensional vectors")
    rank_a = np.argsort(np.argsort(a, kind="stable"), kind="stable").astype(np.float64)
    rank_b = np.argsort(np.argsort(b, kind="stable"), kind="stable").astype(np.float64)
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def preflight(source_band_dir: Path, output_dir: Path) -> dict[str, Any]:
    band_ready = json.loads((source_band_dir / "READY.json").read_text())
    if band_ready.get("status") != "READY" or band_ready.get("code_sha256") != band_code_fingerprint():
        raise ValueError("source band READY/code mismatch")
    band_raw = source_band_dir / "raw.json"
    band_summary = source_band_dir / "summary.json"
    for path in (band_raw, band_summary):
        if not path.is_file():
            raise FileNotFoundError(path)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "READY",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "code_sha256": code_fingerprint(),
        "band_code_sha256": band_ready["code_sha256"],
        "band_protocol_sha256": band_ready["protocol_sha256"],
        "source_band_dir": str(source_band_dir.resolve()),
        "source_band_ready_sha256": _file_sha256(source_band_dir / "READY.json"),
        "source_band_raw_sha256": _file_sha256(band_raw),
        "source_band_summary_sha256": _file_sha256(band_summary),
        "source_profile_ready": band_ready["source_ready"],
        "arms": list(ARMS),
        "lengths": list(LENGTHS),
        "test_anchor_sha256": band_ready["anchors"]["test"]["sha256"],
        "claim_boundary": "post-hoc single-seed frozen-checkpoint mechanistic exploration",
    }
    _atomic_json(output_dir / "READY.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _load_followup_ready(output_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    ready = json.loads((output_dir / "READY.json").read_text())
    if ready.get("status") != "READY" or ready.get("code_sha256") != code_fingerprint():
        raise ValueError("follow-up READY/code mismatch")
    source_dir = Path(ready["source_band_dir"])
    checks = {
        "source_band_ready_sha256": source_dir / "READY.json",
        "source_band_raw_sha256": source_dir / "raw.json",
        "source_band_summary_sha256": source_dir / "summary.json",
    }
    for field, path in checks.items():
        if _file_sha256(path) != ready[field]:
            raise ValueError(f"source artifact changed: {path}")
    source_profile = _load_ready(Path(ready["source_profile_ready"]).parent)
    return ready, source_profile, json.loads((source_dir / "raw.json").read_text())


def _condition_record(values: list[float], baseline: list[float]) -> dict[str, Any]:
    differences = [left - right for left, right in zip(values, baseline, strict=True)]
    return {
        "mean_tail_nll": statistics.fmean(values),
        "delta_vs_full_rope": statistics.fmean(differences),
        "paired_anchor_bootstrap_95ci": _bootstrap_ci(differences),
        "tail_nll": values,
    }


def run(output_dir: Path) -> dict[str, Any]:
    runtime = _validate_cuda()
    ready, source_profile, band_raw = _load_followup_ready(output_dir)
    band_ready = json.loads((Path(ready["source_band_dir"]) / "READY.json").read_text())
    validation = np.load(band_ready["validation"]["path"], mmap_mode="r", allow_pickle=False).reshape(-1)
    anchors = np.load(band_ready["anchors"]["test"]["path"], allow_pickle=False)
    raw: dict[str, Any] = {"runtime": runtime, "arms": {}}

    for arm in ARMS:
        model, _ = _load_model(source_profile, arm)
        model = model.to("cuda").eval()
        training_inv = model.blocks[0].attention.rope.inv_freq.detach().cpu().float().clone()
        band_source = band_raw["arms"][arm]
        masks = band_source["masks"]
        norm = band_source["test_qk"]["aggregate_pair_norm"]
        arm_result: dict[str, Any] = {"qk_pair_norm": norm, "lengths": {}}

        for length in LENGTHS:
            inv = runtime_inv_freq(training_inv, arm, length)
            baseline = band_source["lengths"][str(length)]["full_rope"]["tail_nll"]
            pair_rows = []
            for pair in range(inv.numel()):
                values = _evaluate_schedule(
                    model,
                    validation,
                    anchors,
                    inv_freq=apply_nope(inv, [pair]),
                    length=length,
                    max_batch_tokens=BAND_SPEC.max_batch_tokens,
                )
                pair_rows.append(_condition_record(values, baseline))

            swaps = {}
            for name, control in (
                ("band_adjacent_swap", masks["adjacent_nope"]),
                ("band_far_swap", masks["far_nope"]),
            ):
                values = _evaluate_schedule(
                    model,
                    validation,
                    anchors,
                    inv_freq=swap_pairs(inv, masks["band_nope"], control),
                    length=length,
                    max_batch_tokens=BAND_SPEC.max_batch_tokens,
                )
                swaps[name] = _condition_record(values, baseline)

            causal = [row["delta_vs_full_rope"] for row in pair_rows]
            utility = phase_utility(inv, length).tolist()
            arm_result["lengths"][str(length)] = {
                "runtime_inv_freq_sha256": hashlib.sha256(inv.numpy().tobytes()).hexdigest(),
                "per_pair_nope": pair_rows,
                "phase_utility": utility,
                "qk_norm_spearman": spearman(norm, causal),
                "phase_utility_spearman": spearman(utility, causal),
                "top_causal_pair": int(np.argmax(causal)),
                "top_qk_norm_pair": int(np.argmax(norm)),
                "top_phase_utility_pair": int(np.argmax(utility)),
                "harmful_pair_count": int(np.sum(np.asarray(causal) < 0.0)),
                "swaps": swaps,
            }
        raw["arms"][arm] = arm_result
        del model
        torch.cuda.empty_cache()

    _atomic_json(output_dir / "raw.json", raw)
    return summarize(output_dir, ready, raw)


def summarize(output_dir: Path, ready: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for arm, value in raw["arms"].items():
        for length, row in value["lengths"].items():
            rows.append(
                {
                    "arm": arm,
                    "length": int(length),
                    "qk_norm_spearman": row["qk_norm_spearman"],
                    "phase_utility_spearman": row["phase_utility_spearman"],
                    "top_causal_pair": row["top_causal_pair"],
                    "top_qk_norm_pair": row["top_qk_norm_pair"],
                    "top_phase_utility_pair": row["top_phase_utility_pair"],
                    "harmful_pair_count": row["harmful_pair_count"],
                    "band_adjacent_swap_delta": row["swaps"]["band_adjacent_swap"]["delta_vs_full_rope"],
                    "band_far_swap_delta": row["swaps"]["band_far_swap"]["delta_vs_full_rope"],
                    "band_adjacent_swap_ci": row["swaps"]["band_adjacent_swap"]["paired_anchor_bootstrap_95ci"],
                    "band_far_swap_ci": row["swaps"]["band_far_swap"]["paired_anchor_bootstrap_95ci"],
                }
            )
    median_norm = statistics.median(row["qk_norm_spearman"] for row in rows)
    median_utility = statistics.median(row["phase_utility_spearman"] for row in rows)
    proxy = "QK_NORM" if median_norm > 0.5 else "PHASE_UTILITY" if median_utility > 0.5 else "NO_SIMPLE_PROXY"
    material_swaps = sum(
        abs(row[field]) >= 0.05
        for row in rows
        for field in ("band_adjacent_swap_delta", "band_far_swap_delta")
    )
    result = {
        "status": "PASS",
        "code_sha256": code_fingerprint(),
        "test_anchor_sha256": ready["test_anchor_sha256"],
        "decision": proxy,
        "median_qk_norm_spearman": median_norm,
        "median_phase_utility_spearman": median_utility,
        "material_frequency_assignment_swaps": material_swaps,
        "swap_comparisons": 2 * len(rows),
        "single_seed_supporting": True,
        "post_hoc_exploratory": True,
        "paper_claim": False,
        "rows": rows,
    }
    _atomic_json(output_dir / "summary.json", result)
    lines = [
        "# Per-pair causal spectrum and frequency-assignment diagnostic",
        "",
        f"Decision: **{proxy}**.",
        "",
        "| checkpoint | length | rho(norm, causal) | rho(phase utility, causal) | top causal/norm/utility | harmful pairs | swap band-adjacent | swap band-far |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['arm']} | {row['length']} | {row['qk_norm_spearman']:+.3f} | "
            f"{row['phase_utility_spearman']:+.3f} | {row['top_causal_pair']}/{row['top_qk_norm_pair']}/{row['top_phase_utility_pair']} | "
            f"{row['harmful_pair_count']} | {row['band_adjacent_swap_delta']:+.4f} | {row['band_far_swap_delta']:+.4f} |"
        )
    lines.extend(
        [
            "",
            f"Median Spearman: Q/K norm `{median_norm:+.3f}`, phase utility `{median_utility:+.3f}`. Material assignment swaps: `{material_swaps}/{2 * len(rows)}`.",
            "",
            "Post-hoc single-seed frozen-checkpoint exploration. Per-pair NoPE effects are local interventions and are not additive.",
            "",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--source-band-dir", type=Path, required=True)
    pre.add_argument("--output-dir", type=Path, required=True)
    execute = sub.add_parser("run")
    execute.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        preflight(args.source_band_dir.resolve(), args.output_dir.resolve())
    else:
        run(args.output_dir.resolve())


if __name__ == "__main__":
    main()
