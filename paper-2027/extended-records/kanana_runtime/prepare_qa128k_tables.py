#!/usr/bin/env python3
"""Freeze Kanana TailSpline/MrRoPE S=4 and official runtime YaRN tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.fixed_rope_three_interfaces_20260913.tables import (
    build_analytic,
    make_receipt,
)

from .prepare import (
    MODEL_ID,
    _check_geometry,
    _official_yarn_runtime_table,
    _write_once_or_equal,
)


TARGET_LENGTH = 131_072
SCALE = 4.0


def freeze(model: Path, root: Path) -> dict[str, dict]:
    config = json.loads((model / "config.json").read_text())
    geometry = _check_geometry(config)
    tables = root / "qa128k" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for method, role in (("tailspline", "candidate"), ("mrpro", "baseline")):
        values, gain, construction = build_analytic(
            config, method=method, scale=SCALE,
            low=None, high=None, depth=1.0, gain=None,
        )
        outputs[method] = make_receipt(
            candidate_id=f"kanana_{method}_s4_128k_v1",
            model_id=MODEL_ID,
            role=role,
            scale=SCALE,
            geometry=geometry,
            values=values,
            gain=gain,
            construction={
                **construction,
                "target_length": TARGET_LENGTH,
                "physical_extension_ratio": SCALE,
            },
            source=f"analytic:{method}_s4",
            changed_variables=["runtime_rope_frequency_table", "global_rotary_gain"],
        )
    yarn_values, yarn_gain, yarn_construction = _official_yarn_runtime_table(model, config)
    outputs["official_yarn"] = make_receipt(
        candidate_id="kanana_official_runtime_yarn_factor4p4_128k_v1",
        model_id=MODEL_ID,
        role="baseline",
        scale=4.4,
        geometry=geometry,
        values=yarn_values,
        gain=yarn_gain,
        construction={**yarn_construction, "target_length": TARGET_LENGTH},
        source="transformers:runtime_yarn_factor4p4_beta64_beta2",
        changed_variables=["runtime_rope_frequency_table", "global_rotary_gain"],
    )
    for name, receipt in outputs.items():
        _write_once_or_equal(tables / f"{name}.json", receipt)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    outputs = freeze(args.model.resolve(), args.root.resolve())
    print(json.dumps({
        name: {
            "scale": receipt["scale"],
            "gain": receipt["gain"],
            "band": receipt["table"]["construction"].get("band_envelope"),
            "sha256": receipt["table_sha256_float32"],
        }
        for name, receipt in outputs.items()
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
