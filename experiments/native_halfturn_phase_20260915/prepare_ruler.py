#!/usr/bin/env python3
"""Prepare a fresh Native-4K source-order RULER-13 panel for OLMo.

The panel is CPU-only and does not inspect model outputs.  A new generator seed
separates synthetic prompts from prior OLMo panels; ``--qa-offset`` freezes a
disjoint source-question range for qa_1 and qa_2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.iclr2027_strong_evidence_20260915.prepare_clean_transfer import (
    CONTRACT_REVISION,
    RULER_REVISION,
    TASKS,
    _atomic_json,
    _checkpoint_identity,
    _default_converter_main,
    _default_planb_main,
    _detect_ruler_revision,
    _freeze_panel_manifest,
    _sanitize_source_manifest,
    _sha256,
    _validate_model_id,
)


CONTRACT = "native-halfturn-ruler4k-source-order-v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows-per-task", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--qa-offset", type=int, default=5200)
    args = parser.parse_args()
    if args.rows_per_task != 60:
        raise ValueError("the frozen Native half-turn panel requires 60 rows per task")
    if args.seed < 0 or args.qa_offset < 0:
        raise ValueError("seed and QA offset must be nonnegative")

    model_id = _validate_model_id(args.model_id)
    model = args.model.resolve()
    data_root = args.data_root.resolve()
    out = args.out.resolve()
    identity = _checkpoint_identity(model)
    if int(identity["native_length"]) != 4096:
        raise ValueError("the first Native half-turn experiment is frozen to a 4096-token model")
    _detect_ruler_revision(data_root)
    request = {
        "contract": CONTRACT,
        "contract_revision": CONTRACT_REVISION,
        "upstream_revision": RULER_REVISION,
        "model_id": model_id,
        "model_identity": identity,
        "native_length": 4096,
        "rows_per_task": 60,
        "seed": args.seed,
        "qa_offset": args.qa_offset,
        "tasks": list(TASKS),
    }
    manifest_path = out / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text())
        if {key: existing.get(key) for key in request} != request:
            raise ValueError("existing Native half-turn panel has another frozen request")
        inputs = out / existing["panel"]["inputs"]
        if (
            existing.get("status") != "COMPLETE"
            or not inputs.is_file()
            or _sha256(inputs) != existing["panel"]["inputs_sha256"]
        ):
            raise ValueError("existing Native half-turn panel is incomplete or has drifted")
        print(json.dumps({"status": "SKIP_COMPLETE", "rows": existing["rows"]}, sort_keys=True))
        return

    out.mkdir(parents=True, exist_ok=True)
    for task_index, task in enumerate(TASKS):
        source_part = out / "source_parts" / task
        result = _default_planb_main([
            "--model", str(model),
            "--model-contract", "generic",
            "--upstream", str(data_root),
            "--out", str(source_part),
            "--stage", "H",
            "--contract", "planb",
            "--tasks", task,
            "--caps", "4096",
            "--counts-by-cap", "4096:60",
            "--selection-mode", "source-order",
            "--source-only",
            "--qa-base-offset", str(args.qa_offset),
            "--seed", str(args.seed + task_index * 100),
        ])
        if result not in (None, 0):
            raise RuntimeError(f"RULER source preparation failed for {task}: {result}")
        _sanitize_source_manifest(
            source_part / "manifest.json", model_id=model_id, model_name=model.name,
        )

    panel_dir = out / "panel"
    result = _default_converter_main([
        "--source-parts", str(out / "source_parts"),
        "--model", str(model),
        "--out", str(panel_dir),
        "--length", "4096",
        "--rows-per-task", "60",
        "--tasks", ",".join(TASKS),
    ])
    if result not in (None, 0):
        raise RuntimeError(f"clean RULER conversion failed: {result}")
    panel_manifest = _freeze_panel_manifest(
        panel_dir=panel_dir,
        out=out,
        model_id=model_id,
        identity=identity,
        scale=1.0,
        length=4096,
        rows_per_task=60,
    )
    manifest = {
        "status": "COMPLETE",
        **request,
        "rows": len(TASKS) * 60,
        "selection_mode": "source-order",
        "selection_uses_model_outputs": False,
        "content_padding": False,
        "freshness": {
            "synthetic_prompts": "new fixed generator seed relative to prior OLMo panels",
            "qa_questions": "source rows 5200..5259 for both QA tasks",
            "previous_olmo_qa_ranges_checked": ["5000..5029", "5600..5799"],
        },
        "panel": {
            "inputs": "panel/inputs.jsonl",
            "manifest": "panel/manifest.json",
            "rows": panel_manifest["rows"],
            "inputs_sha256": panel_manifest["inputs_sha256"],
        },
        "scope": (
            "Fresh OLMo Native-4K RULER Full-13 x 60 panel for a frozen analytic "
            "frequency intervention; no model-output selection and no content padding."
        ),
        "portable_paths": True,
    }
    _atomic_json(manifest_path, manifest)
    print(json.dumps({"status": "COMPLETE", "rows": manifest["rows"], "sha256": panel_manifest["inputs_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()

