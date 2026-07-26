#!/usr/bin/env python3
"""Screen corrected Native/EVQ pair hybrids on independent 4K routing."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import torch

from rebuttal.rebuttal_0723.experiments.olmo2_lora_conversion import (
    install_adaptation,
    load_model,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_ood_factorial import (
    load_adapter,
)
from rebuttal.rebuttal_0723.experiments.small_model_lora_conversion import (
    atomic_json,
    configure_cuda,
    sha256_file,
)

from .evaluate_instruct_ruler_screen import apply_screen_frequency
from .train_4k_counterfactual_routing import (
    LENGTH,
    RoutingPairView,
    evaluate_routing,
)
from .train_4k_stage_a import ready_checkpoint_digest
from .train_screen import evaluate_natural_nll


PAIR_HYBRIDS = (
    "native",
    "hybrid_evq_low4",
    "hybrid_evq_low8",
    "hybrid_evq_low12",
    "hybrid_evq_low16",
    "hybrid_evq_low24",
    "hybrid_evq_low32",
    "hybrid_evq_low40",
    "hybrid_native_low32",
    "hybrid_native_low16",
    "hybrid_blend10",
    "hybrid_blend25",
    "hybrid_blend_evq_0p1pct",
    "hybrid_blend_evq_0p5pct",
    "hybrid_blend_evq_1pct",
    "hybrid_blend_evq_2pct",
    "hybrid_blend_evq_5pct",
    "hybrid_heads_evq1",
    "hybrid_heads_evq2",
    "hybrid_heads_evq4",
    "hybrid_heads_evq8",
    "evq",
)
RESULT_STATUS = "OLMO2_4K_NATURAL_MULTIQUERY_HYBRID_SCREEN_COMPLETE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ready-receipt", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--routing-data", type=Path, required=True)
    parser.add_argument("--background-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--frequencies",
        nargs="+",
        choices=PAIR_HYBRIDS,
        default=list(PAIR_HYBRIDS),
    )
    parser.add_argument(
        "--custom-evq-head-sets",
        nargs="*",
        default=(),
        metavar="INDEX[,INDEX...]",
        help=(
            "Additional head-hybrid settings. Each value is a comma-separated "
            "set of EVQ head indices selected only on the independent "
            "calibration task."
        ),
    )
    parser.add_argument("--calibration-rows", type=int, default=16)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=128.0)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--natural-eval-rows", type=int, default=16)
    return parser.parse_args()


def parse_custom_head_sets(
    values: list[str] | tuple[str, ...],
) -> list[tuple[str, tuple[int, ...]]]:
    parsed: list[tuple[str, tuple[int, ...]]] = []
    seen: set[tuple[int, ...]] = set()
    for raw_value in values:
        try:
            indices = tuple(
                sorted(
                    {
                        int(value)
                        for value in str(raw_value).split(",")
                        if value
                    }
                )
            )
        except ValueError as error:
            raise ValueError(
                f"invalid custom EVQ head set: {raw_value!r}"
            ) from error
        if (
            not indices
            or any(index < 0 or index >= 16 for index in indices)
            or len(indices) >= 16
        ):
            raise ValueError(
                f"custom EVQ head set escaped [0, 15]: {raw_value!r}"
            )
        if indices in seen:
            raise ValueError(f"duplicate custom EVQ head set: {indices}")
        seen.add(indices)
        parsed.append(
            (
                "hybrid_heads_custom__"
                + "_".join(str(index) for index in indices),
                indices,
            )
        )
    return parsed


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if not 1 <= int(args.calibration_rows) <= 128:
        raise ValueError("calibration rows must be in [1, 128]")
    checkpoint = args.checkpoint.resolve()
    ready = args.ready_receipt.resolve()
    adapter = args.adapter.resolve()
    checkpoint_digest = ready_checkpoint_digest(checkpoint, ready)
    view = RoutingPairView(
        args.routing_data.resolve() / "calibration"
    )
    payload = torch.load(adapter, map_location="cpu", weights_only=True)
    metadata = dict(payload.get("metadata", {}))
    expected = {
        "base_checkpoint_sha256": checkpoint_digest,
        "frequency": "native",
        "adaptation": "qkvo_answer",
        "rank": int(args.rank),
        "alpha": float(args.alpha),
        "training_sequence_length": LENGTH,
    }
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise RuntimeError(f"Native adapter metadata drift for {name}")

    runtime = configure_cuda()
    rows: dict[str, Any] = {}
    plans = [
        (str(name), str(name), ())
        for name in args.frequencies
    ]
    plans.extend(
        (result_name, "hybrid_heads_custom", head_indices)
        for result_name, head_indices in parse_custom_head_sets(
            tuple(args.custom_evq_head_sets)
        )
    )
    if len({result_name for result_name, _, _ in plans}) != len(plans):
        raise ValueError("duplicate frequency result name")
    for result_name, frequency_name, custom_head_indices in plans:
        model = load_model(checkpoint)
        frequency = apply_screen_frequency(
            model,
            frequency_name,
            custom_evq_head_indices=custom_head_indices,
        )
        readout = install_adaptation(
            model,
            "qkvo_answer",
            rank=int(args.rank),
            alpha=float(args.alpha),
        )
        if readout is not None:
            raise RuntimeError("hybrid screen forbids a readout")
        loaded = load_adapter(adapter, model, None)
        if loaded != metadata:
            raise RuntimeError("adapter metadata changed while loading")
        model.to("cuda")
        rows[result_name] = {
            "frequency": frequency,
            "calibration": evaluate_routing(
                model=model,
                view=view,
                rows=int(args.calibration_rows),
                pair_batch_size=2,
                margin=float(args.margin),
            ),
        }
        if args.background_dir is not None:
            rows[result_name]["natural_nll"] = evaluate_natural_nll(
                model=model,
                background_dir=args.background_dir.resolve(),
                lengths=(4_096, 8_192, 16_384),
                rows=int(args.natural_eval_rows),
                tail_tokens=1_024,
            )
        del model
        gc.collect()
        torch.cuda.empty_cache()

    result = {
        "status": RESULT_STATUS,
        "reviewer_concerns": ["R27bE.2", "R27bE.5", "AC.2"],
        "metric_boundary": (
            "Corrected inference-time pair-hybrid screen using a Native "
            "adapter trained only on independent natural multi-query data"
        ),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "checkpoint_sha256": checkpoint_digest,
        "adapter": {
            "path": str(adapter),
            "sha256": sha256_file(adapter),
            "metadata": metadata,
        },
        "routing_data": {
            "path": str(args.routing_data.resolve()),
            "manifest_sha256": sha256_file(
                args.routing_data.resolve() / "manifest.json"
            ),
        },
        "protocol": {
            "calibration_rows": int(args.calibration_rows),
            "rank": int(args.rank),
            "alpha": float(args.alpha),
            "margin": float(args.margin),
            "frequencies": [
                result_name for result_name, _, _ in plans
            ],
            "custom_evq_head_sets": {
                result_name: list(indices)
                for result_name, frequency_name, indices in plans
                if frequency_name == "hybrid_heads_custom"
            },
            "training_rows_from_ruler_or_niah": 0,
            "natural_eval_rows": (
                None
                if args.background_dir is None
                else int(args.natural_eval_rows)
            ),
        },
        "runtime": runtime,
        "results": rows,
    }
    output.mkdir(parents=True)
    atomic_json(output / "results.json", result)
    print(
        json.dumps(
            {
                "status": RESULT_STATUS,
                "output": str(output / "results.json"),
                "scores": {
                    name: value["calibration"]["source_token_exact"]
                    for name, value in rows.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
