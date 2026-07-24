#!/usr/bin/env python3
"""Select harmful rotary pairs on dev anchors and test their joint removal."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rebuttal.rebuttal_0723.experiments.frequency_band_usage_5090 import (
    ARMS,
    LENGTHS,
    SPEC,
    _atomic_json,
    _file_sha256,
    apply_nope,
    runtime_inv_freq,
)
from rebuttal.rebuttal_0723.experiments.frequency_causal_spectrum_5090 import _condition_record
from rebuttal.rebuttal_0723.experiments.profiled_residual_5090.run_experiment import (
    _evaluate_schedule,
    _load_model,
    _load_ready,
    _validate_cuda,
)


ROOT = Path(__file__).resolve().parents[3]


def code_fingerprint() -> str:
    paths = (Path(__file__).resolve(), ROOT / "tests/test_heldout_causal_pruning_5090.py")
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(bytes.fromhex(_file_sha256(path)))
    return digest.hexdigest()


def choose_masks(pair_rows: list[dict[str, Any]]) -> dict[str, list[int]]:
    return {
        "negative_mean": [i for i, row in enumerate(pair_rows) if row["delta_vs_full_rope"] < 0.0],
        "negative_ci": [i for i, row in enumerate(pair_rows) if row["paired_anchor_bootstrap_95ci"][1] < 0.0],
    }


def preflight(source_dir: Path, output_dir: Path) -> dict[str, Any]:
    causal_ready = json.loads((source_dir / "READY.json").read_text())
    causal_raw = source_dir / "raw.json"
    causal_summary = source_dir / "summary.json"
    for path in (causal_raw, causal_summary):
        if not path.is_file():
            raise FileNotFoundError(path)
    band_ready_path = Path(causal_ready["source_band_dir"]) / "READY.json"
    band_ready = json.loads(band_ready_path.read_text())
    for split in ("selection", "test"):
        anchor = band_ready["anchors"][split]
        if _file_sha256(Path(anchor["path"])) != anchor["sha256"]:
            raise ValueError(f"{split} anchor hash mismatch")
    ready = {
        "status": "READY",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "code_sha256": code_fingerprint(),
        "source_dir": str(source_dir.resolve()),
        "source_ready_sha256": _file_sha256(source_dir / "READY.json"),
        "source_raw_sha256": _file_sha256(causal_raw),
        "source_summary_sha256": _file_sha256(causal_summary),
        "band_ready": str(band_ready_path.resolve()),
        "selection_anchor_sha256": band_ready["anchors"]["selection"]["sha256"],
        "test_anchor_sha256": band_ready["anchors"]["test"]["sha256"],
        "arms": list(ARMS),
        "lengths": list(LENGTHS),
        "claim_boundary": "post-hoc single-seed held-out mechanistic exploration",
    }
    _atomic_json(output_dir / "READY.json", ready)
    print(json.dumps(ready, indent=2, sort_keys=True))
    return ready


def _load_run_inputs(output_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    ready = json.loads((output_dir / "READY.json").read_text())
    if ready.get("status") != "READY" or ready.get("code_sha256") != code_fingerprint():
        raise ValueError("READY/code mismatch")
    source = Path(ready["source_dir"])
    for field, name in (
        ("source_ready_sha256", "READY.json"),
        ("source_raw_sha256", "raw.json"),
        ("source_summary_sha256", "summary.json"),
    ):
        if _file_sha256(source / name) != ready[field]:
            raise ValueError(f"source artifact changed: {name}")
    band_ready = json.loads(Path(ready["band_ready"]).read_text())
    profile_ready = _load_ready(Path(band_ready["source_ready"]).parent)
    return ready, band_ready, profile_ready


def run(output_dir: Path) -> dict[str, Any]:
    runtime = _validate_cuda()
    ready, band_ready, profile_ready = _load_run_inputs(output_dir)
    validation = np.load(band_ready["validation"]["path"], mmap_mode="r", allow_pickle=False).reshape(-1)
    selection = np.load(band_ready["anchors"]["selection"]["path"], allow_pickle=False)
    test = np.load(band_ready["anchors"]["test"]["path"], allow_pickle=False)
    raw: dict[str, Any] = {"runtime": runtime, "arms": {}}

    for arm in ARMS:
        model, _ = _load_model(profile_ready, arm)
        model = model.to("cuda").eval()
        training_inv = model.blocks[0].attention.rope.inv_freq.detach().cpu().float().clone()
        arm_result: dict[str, Any] = {"lengths": {}}
        for length in LENGTHS:
            inv = runtime_inv_freq(training_inv, arm, length)
            dev_full = _evaluate_schedule(
                model, validation, selection, inv_freq=inv, length=length, max_batch_tokens=SPEC.max_batch_tokens
            )
            pair_rows = []
            for pair in range(inv.numel()):
                values = _evaluate_schedule(
                    model,
                    validation,
                    selection,
                    inv_freq=apply_nope(inv, [pair]),
                    length=length,
                    max_batch_tokens=SPEC.max_batch_tokens,
                )
                pair_rows.append(_condition_record(values, dev_full))
            masks = choose_masks(pair_rows)
            test_full = _evaluate_schedule(
                model, validation, test, inv_freq=inv, length=length, max_batch_tokens=SPEC.max_batch_tokens
            )
            test_rows = {}
            for name, mask in masks.items():
                values = _evaluate_schedule(
                    model,
                    validation,
                    test,
                    inv_freq=apply_nope(inv, mask),
                    length=length,
                    max_batch_tokens=SPEC.max_batch_tokens,
                )
                test_rows[name] = _condition_record(values, test_full)
            arm_result["lengths"][str(length)] = {
                "selection_per_pair": pair_rows,
                "selected_masks": masks,
                "test_full_tail_nll": test_full,
                "test": test_rows,
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
            for criterion, result in row["test"].items():
                rows.append(
                    {
                        "arm": arm,
                        "length": int(length),
                        "criterion": criterion,
                        "selected_pairs": row["selected_masks"][criterion],
                        "test_delta": result["delta_vs_full_rope"],
                        "test_ci": result["paired_anchor_bootstrap_95ci"],
                    }
                )
    long_ci_wins = sum(
        row["length"] >= 4096
        and row["criterion"] == "negative_ci"
        and row["test_ci"][1] < 0.0
        for row in rows
    )
    decision = "HELDOUT_PRUNING_SIGNAL" if long_ci_wins else "NO_HELDOUT_PRUNING_SIGNAL"
    summary = {
        "status": "PASS",
        "decision": decision,
        "long_context_negative_ci_wins": long_ci_wins,
        "selection_anchor_sha256": ready["selection_anchor_sha256"],
        "test_anchor_sha256": ready["test_anchor_sha256"],
        "single_seed_supporting": True,
        "post_hoc_exploratory": True,
        "paper_claim": False,
        "rows": rows,
    }
    _atomic_json(output_dir / "summary.json", summary)
    lines = [
        "# Held-out causal rotary-pair pruning",
        "",
        f"Decision: **{decision}**.",
        "",
        "| checkpoint | length | selector | pairs | held-out NLL delta | 95% CI |",
        "| --- | ---: | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['arm']} | {row['length']} | {row['criterion']} | {len(row['selected_pairs'])} | "
            f"{row['test_delta']:+.4f} | [{row['test_ci'][0]:+.4f}, {row['test_ci'][1]:+.4f}] |"
        )
    lines.extend(
        [
            "",
            "Negative delta means that the independently selected joint NoPE mask improved held-out tail NLL.",
            "This is a post-hoc single-seed diagnostic; it does not establish an inference method or paper claim.",
            "",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--source-dir", type=Path, required=True)
    pre.add_argument("--output-dir", type=Path, required=True)
    execute = sub.add_parser("run")
    execute.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        preflight(args.source_dir.resolve(), args.output_dir.resolve())
    else:
        run(args.output_dir.resolve())


if __name__ == "__main__":
    main()
