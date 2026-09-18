#!/usr/bin/env python3
"""Prepare the frozen Kanana 64K paired RULER experiment without a GPU."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from experiments.fixed_rope_three_interfaces_20260913.tables import (
    atomic_json,
    build_analytic,
    make_receipt,
    model_geometry,
)
from experiments.iclr2027_strong_evidence_20260915.prepare_clean_transfer import (
    TASKS,
    prepare as prepare_clean_transfer,
)


PILOT_TASKS = ("niah_multiquery", "vt")
TARGET_LENGTH = 65_536
ROWS_PER_TASK = 10
OFFICIAL_YARN_FACTOR = 4.4
OFFICIAL_YARN_BETA_FAST = 64.0
OFFICIAL_YARN_BETA_SLOW = 2.0
TAILSPLINE_SCALE = 2.0
MODEL_ID = "kakaocorp_kanana_1.5_8b_instruct_2505"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _write_once_or_equal(path: Path, value: dict) -> None:
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"frozen artifact drift: {path}")
        return
    atomic_json(path, value)


def _check_geometry(config: dict) -> dict:
    geometry = model_geometry(config)
    expected = {
        "model_type": "llama",
        "attention_head_dim": 128,
        "partial_rotary_factor": 1.0,
        "head_dim": 128,
        "pairs": 64,
        "base": 8_000_000.0,
        "native_length": 32_768,
    }
    if geometry != expected:
        raise ValueError(f"unexpected Kanana public RoPE geometry: {geometry}")
    if (
        int(config.get("hidden_size", 0)) != 4096
        or int(config.get("num_hidden_layers", 0)) != 32
        or int(config.get("num_attention_heads", 0)) != 32
        or int(config.get("num_key_value_heads", 0)) != 8
    ):
        raise ValueError("unexpected Kanana architecture identity")
    return geometry


def _official_yarn_runtime_table(model: Path, config: dict) -> tuple[np.ndarray, float, dict]:
    """Ask the installed Transformers runtime for the exact requested YaRN table."""
    import torch
    from transformers import AutoConfig, modeling_rope_utils

    runtime_config = AutoConfig.from_pretrained(model, local_files_only=True)
    parameters = {
        "rope_type": "yarn",
        "rope_theta": 8_000_000.0,
        "factor": OFFICIAL_YARN_FACTOR,
        "original_max_position_embeddings": 32_768,
        "beta_fast": OFFICIAL_YARN_BETA_FAST,
        "beta_slow": OFFICIAL_YARN_BETA_SLOW,
    }
    runtime_config.rope_parameters = dict(parameters)
    runtime_config.rope_scaling = dict(parameters)
    compute = getattr(modeling_rope_utils, "_compute_yarn_parameters", None)
    if compute is None:
        raise RuntimeError("installed Transformers runtime lacks the YaRN initializer")
    values, gain = compute(runtime_config, device=torch.device("cpu"))
    values = values.detach().cpu().float().numpy()
    expected_gain = 1.0 + 0.1 * math.log(OFFICIAL_YARN_FACTOR)
    if (
        values.shape != (64,)
        or not np.isfinite(values).all()
        or not np.all(values[:-1] > values[1:])
        or not math.isclose(float(gain), expected_gain, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValueError("Transformers official YaRN runtime identity drift")
    construction = {
        "method": "transformers_official_runtime_yarn",
        "runtime_rope_parameters": parameters,
        "user_supplied_old_format": {
            "type": "yarn",
            "factor": OFFICIAL_YARN_FACTOR,
            "original_max_position_embeddings": 32_768,
            "beta_fast": OFFICIAL_YARN_BETA_FAST,
            "beta_slow": OFFICIAL_YARN_BETA_SLOW,
        },
        "transformers_version": __import__("transformers").__version__,
        "checkpoint_training_history": "unchanged; runtime RoPE replacement only",
        "same_table_all_layers_and_lengths": True,
    }
    return values, float(gain), construction


def freeze_tables(model: Path, root: Path) -> dict[str, dict]:
    config = json.loads((model / "config.json").read_text())
    geometry = _check_geometry(config)
    tables = root / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    ts_values, ts_gain, ts_construction = build_analytic(
        config, method="tailspline", scale=TAILSPLINE_SCALE,
        low=None, high=None, depth=1.0, gain=None,
    )
    tailspline = make_receipt(
        candidate_id="kanana_tailspline_s2_64k_v1",
        model_id=MODEL_ID,
        role="candidate",
        scale=TAILSPLINE_SCALE,
        geometry=geometry,
        values=ts_values,
        gain=ts_gain,
        construction={
            **ts_construction,
            "target_length": TARGET_LENGTH,
            "physical_extension_ratio": 2.0,
        },
        source="analytic:exact_finite_grid_tailspline_s2",
        changed_variables=["runtime_rope_frequency_table", "global_rotary_gain"],
    )
    yarn_values, yarn_gain, yarn_construction = _official_yarn_runtime_table(
        model, config,
    )
    yarn = make_receipt(
        candidate_id="kanana_official_runtime_yarn_factor4p4_v1",
        model_id=MODEL_ID,
        role="baseline",
        scale=OFFICIAL_YARN_FACTOR,
        geometry=geometry,
        values=yarn_values,
        gain=yarn_gain,
        construction=yarn_construction,
        source="transformers:runtime_yarn_factor4p4_beta64_beta2",
        changed_variables=["runtime_rope_frequency_table", "global_rotary_gain"],
    )
    outputs = {"tailspline": tailspline, "official_yarn": yarn}
    for name, receipt in outputs.items():
        _write_once_or_equal(tables / f"{name}.json", receipt)
    return outputs


def _validate_full_panel(rows: list[dict]) -> None:
    counts = Counter(str(row.get("task")) for row in rows)
    if len(rows) != len(TASKS) * ROWS_PER_TASK or counts != Counter({
        task: ROWS_PER_TASK for task in TASKS
    }):
        raise ValueError("Kanana source panel is not Full-13 x 10")
    for row in rows:
        if (
            int(row.get("length_cap", -1)) != TARGET_LENGTH
            or len(row.get("prompt_ids") or []) + int(row.get("max_new_tokens", 0))
            > TARGET_LENGTH
            or row.get("selection_mode") != "source-order"
            or row.get("selection_uses_model_outputs") is not False
            or int(row.get("irrelevant_padding_tokens", -1)) != 0
        ):
            raise ValueError("Kanana panel violates the unpadded source-order contract")


def freeze_subpanels(full_panel: Path, root: Path) -> dict[str, dict]:
    rows = read_jsonl(full_panel)
    _validate_full_panel(rows)
    groups = {
        "pilot2": [row for row in rows if row["task"] in PILOT_TASKS],
        "rest11": [row for row in rows if row["task"] not in PILOT_TASKS],
    }
    expected = {"pilot2": 20, "rest11": 110}
    result = {}
    full_ids = {row["row_id"] for row in rows}
    if {row["row_id"] for row in groups["pilot2"]} & {
        row["row_id"] for row in groups["rest11"]
    }:
        raise ValueError("pilot and continuation panels overlap")
    if {row["row_id"] for values in groups.values() for row in values} != full_ids:
        raise ValueError("pilot and continuation panels do not reconstruct Full-13")
    for name, selected in groups.items():
        if len(selected) != expected[name]:
            raise ValueError(f"{name} row count drift")
        path = root / "assets" / name / "inputs.jsonl"
        if path.exists():
            if read_jsonl(path) != selected:
                raise ValueError(f"existing {name} panel drift")
        else:
            atomic_jsonl(path, selected)
        manifest = {
            "status": "COMPLETE",
            "panel": name,
            "rows": len(selected),
            "rows_per_task": ROWS_PER_TASK,
            "tasks": list(PILOT_TASKS) if name == "pilot2" else [
                task for task in TASKS if task not in PILOT_TASKS
            ],
            "length_cap": TARGET_LENGTH,
            "selection_mode": "source-order",
            "selection_uses_model_outputs": False,
            "content_padding": False,
            "inputs_sha256": sha256(path),
            "source_full13_sha256": sha256(full_panel),
        }
        _write_once_or_equal(path.parent / "manifest.json", manifest)
        result[name] = manifest
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--ruler", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    model, ruler, root = args.model.resolve(), args.ruler.resolve(), args.root.resolve()
    receipts = freeze_tables(model, root)
    prepare_clean_transfer([
        "--model", str(model),
        "--model-id", MODEL_ID,
        "--data-root", str(ruler),
        "--out", str(root / "assets" / "full13"),
        "--scale", str(TAILSPLINE_SCALE),
        "--lengths", str(TARGET_LENGTH),
        "--rows-per-task", str(ROWS_PER_TASK),
        "--seed", "20260918",
        "--qa-offset", "5600",
    ])
    full_panel = root / "assets" / "full13" / "panels" / str(TARGET_LENGTH) / "inputs.jsonl"
    panels = freeze_subpanels(full_panel, root)
    minimal = {"evaluation_panels": {}, "lengths": []}
    _write_once_or_equal(root / "minimal_eval_manifest.json", minimal)
    ready = {
        "status": "KANANA_64K_YARN_TAILSPLINE_READY_V1",
        "model": str(model),
        "model_config_sha256": sha256(model / "config.json"),
        "target_length": TARGET_LENGTH,
        "physical_extension_ratio": 2.0,
        "pilot_tasks": list(PILOT_TASKS),
        "pilot_rows": panels["pilot2"]["rows"],
        "continuation_rows": panels["rest11"]["rows"],
        "full_rows": 130,
        "tables": {
            name: {
                "receipt": f"tables/{name}.json",
                "table_sha256_float32": receipt["table_sha256_float32"],
                "gain": receipt["gain"],
            }
            for name, receipt in receipts.items()
        },
        "gate": {
            "expand_when_absolute_pilot_delta_at_most": 0.10,
            "metric": "task-equal RULER official score, TailSpline minus official YaRN",
        },
    }
    _write_once_or_equal(root / "ready.json", ready)
    print(json.dumps(ready, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
