#!/usr/bin/env python3
"""Measure Native OLMo-2 rotary-pair importance before any training."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rebuttal.rebuttal_0723.experiments.olmo2_lora_conversion import (
    load_model,
)
from rebuttal.rebuttal_0723.experiments.small_model_lora_conversion import (
    atomic_json,
    configure_cuda,
    seed_everything,
    sha256_file,
)

from .native_protected_evq import (
    CALIBRATION_ROWS,
    DIAGNOSTIC_PREPARED_STATUS,
    DIAGNOSTIC_STATUS,
    QUERY_POSITIONS,
    SEQUENCE_LENGTH,
    BandImportanceCollector,
    configure_band_importance_attention,
    select_protected_pairs,
)
from .train_4k_stage_a import ready_checkpoint_digest
from .train_screen import apply_frequency, load_fixed_view


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-ready-receipt", type=Path, required=True)
    parser.add_argument("--training-view", type=Path, required=True)
    parser.add_argument("--prepared-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=CALIBRATION_ROWS)
    parser.add_argument(
        "--query-position-count", type=int, default=QUERY_POSITIONS
    )
    parser.add_argument("--pair-chunk-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20_260_804)
    return parser.parse_args()


def file_receipt(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def deterministic_calibration_rows(
    *,
    split: np.ndarray,
    lengths: np.ndarray,
    rows: int,
    seed: int,
) -> np.ndarray:
    candidates = np.flatnonzero(
        (np.asarray(split) != 0)
        & (np.asarray(lengths) == SEQUENCE_LENGTH)
    )
    if len(candidates) < int(rows):
        raise RuntimeError(
            f"need {rows} held-out full-4K rows, found {len(candidates)}"
        )
    generator = np.random.default_rng(int(seed))
    return np.sort(generator.choice(candidates, size=int(rows), replace=False))


def deterministic_query_positions(
    *,
    count: int,
) -> np.ndarray:
    if int(count) < 4 or int(count) > 128:
        raise ValueError("query-position count must be in [4, 128]")
    positions = np.rint(
        np.linspace(
            SEQUENCE_LENGTH // 8,
            SEQUENCE_LENGTH - 1,
            num=int(count),
        )
    ).astype(np.int64)
    if len(np.unique(positions)) != int(count):
        raise RuntimeError("query-position construction produced duplicates")
    return positions


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    checkpoint = args.checkpoint.resolve()
    checkpoint_ready = args.checkpoint_ready_receipt.resolve()
    training_view = args.training_view.resolve()
    checkpoint_digest = ready_checkpoint_digest(
        checkpoint, checkpoint_ready
    )
    view = load_fixed_view(training_view)
    selected_rows = deterministic_calibration_rows(
        split=view.split,
        lengths=view.lengths,
        rows=int(args.rows),
        seed=int(args.seed),
    )
    query_positions = deterministic_query_positions(
        count=int(args.query_position_count)
    )
    prepared_path = args.prepared_receipt.resolve()
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    expected_protocol = {
        "rows": int(args.rows),
        "row_indices": [int(value) for value in selected_rows],
        "query_position_count": int(args.query_position_count),
        "query_positions": [int(value) for value in query_positions],
        "pair_chunk_size": int(args.pair_chunk_size),
        "physical_sequence_length": SEQUENCE_LENGTH,
        "seed": int(args.seed),
        "training_or_optimizer_steps": 0,
    }
    if (
        prepared.get("status") != DIAGNOSTIC_PREPARED_STATUS
        or prepared.get("protocol") != expected_protocol
        or prepared["inputs"]["checkpoint"]["composite_sha256"]
        != checkpoint_digest
        or Path(prepared["inputs"]["checkpoint"]["path"]).resolve()
        != checkpoint
        or Path(prepared["inputs"]["training_view"]["path"]).resolve()
        != training_view
        or Path(prepared["diagnostic_output"]).resolve() != output
    ):
        raise RuntimeError("diagnostic prepared receipt drift")
    for name, entry in prepared["inputs"]["training_view"]["files"].items():
        path = training_view / name
        if (
            int(path.stat().st_size) != int(entry["bytes"])
            or sha256_file(path) != str(entry["sha256"])
        ):
            raise RuntimeError(f"training-view changed after preflight: {name}")
    source = prepared["source"]
    if (
        sha256_file(Path(__file__).resolve())
        != source["diagnostic"]["sha256"]
        or sha256_file(
            Path(__file__).with_name("native_protected_evq.py")
        )
        != source["method"]["sha256"]
    ):
        raise RuntimeError("diagnostic source changed after preflight")
    seed_everything(int(args.seed))
    environment = configure_cuda()
    model = load_model(checkpoint)
    frequency = apply_frequency(model, "native")
    configure_band_importance_attention(model)
    model.eval()
    model.config.use_cache = False
    model.to("cuda")
    collector = BandImportanceCollector(
        query_positions=torch.from_numpy(query_positions),
        pair_chunk_size=int(args.pair_chunk_size),
    )
    raw_rows: list[np.ndarray] = []
    row_seconds: list[float] = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in selected_rows:
            collector.clear()
            input_ids = torch.from_numpy(
                np.asarray(view.input_ids[int(row)], dtype=np.int64)
            )[None, :].to("cuda")
            row_started = time.perf_counter()
            model.model(
                input_ids=input_ids,
                use_cache=False,
                return_dict=False,
                band_importance_collector=collector,
            )
            torch.cuda.synchronize()
            raw_rows.append(collector.require_row().numpy())
            row_seconds.append(time.perf_counter() - row_started)
            del input_ids
    raw = np.stack(raw_rows, axis=0)
    selection = select_protected_pairs(raw)
    row_digest = hashlib.sha256(
        np.ascontiguousarray(selected_rows, dtype="<i8").tobytes()
    ).hexdigest()
    query_digest = hashlib.sha256(
        np.ascontiguousarray(query_positions, dtype="<i8").tobytes()
    ).hexdigest()
    receipt = {
        "status": DIAGNOSTIC_STATUS,
        "passed": bool(selection["passed"]),
        "scientific_question": (
            "Does the untouched Native OLMo-2 model concentrate its actual "
            "4K post-RoPE causal attention function in a small, stable "
            "rotary-pair subset?"
        ),
        "metric": (
            "exact forward KL between Native causal attention and the same "
            "attention after removing one post-RoPE Q/K pair contribution"
        ),
        "checkpoint": {
            "path": str(checkpoint),
            "composite_sha256": checkpoint_digest,
            "ready_receipt": file_receipt(checkpoint_ready),
        },
        "training_view": {
            "path": str(training_view),
            "files": {
                name: file_receipt(training_view / name)
                for name in (
                    "manifest.json",
                    "input_ids.npy",
                    "assistant_mask.npy",
                    "lengths.npy",
                    "split.npy",
                )
            },
        },
        "native_frequency": frequency,
        "calibration": {
            "rows": [int(value) for value in selected_rows],
            "row_indices_sha256": row_digest,
            "split_contract": "view split != 0; never used for restoration",
            "physical_sequence_length": SEQUENCE_LENGTH,
            "query_positions": [int(value) for value in query_positions],
            "query_positions_sha256": query_digest,
            "pair_chunk_size": int(args.pair_chunk_size),
        },
        "selection": selection,
        "raw_per_row_layer_head_pair": raw.tolist(),
        "runtime": {
            "environment": environment,
            "elapsed_seconds": time.perf_counter() - started,
            "per_row_seconds": row_seconds,
            "peak_allocated_bytes": int(
                torch.cuda.max_memory_allocated()
            ),
            "peak_reserved_bytes": int(
                torch.cuda.max_memory_reserved()
            ),
        },
        "source": {
            "diagnostic": file_receipt(Path(__file__).resolve()),
            "method": file_receipt(
                Path(__file__).with_name("native_protected_evq.py")
            ),
        },
        "prepared_receipt": file_receipt(prepared_path),
        "claim_boundary": (
            "A pass licenses one calibrated retrofit experiment. It does not "
            "validate the analytic 2.205L anchor, prove causal additivity, "
            "or constitute capability evidence."
        ),
        "stop_condition": (
            "If selection.passed is false, do not train the protected-EVQ "
            "candidate and do not change the thresholds after seeing results."
        ),
    }
    atomic_json(output, receipt)
    print(
        json.dumps(
            {
                "status": DIAGNOSTIC_STATUS,
                "passed": receipt["passed"],
                "output": str(output),
                "output_sha256": sha256_file(output),
                "protected_pair_indices": selection[
                    "protected_pair_indices"
                ],
                "selection_gates": selection["gates"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not receipt["passed"]:
        raise RuntimeError(
            "Native band importance is not sufficiently concentrated/stable"
        )


if __name__ == "__main__":
    main()
