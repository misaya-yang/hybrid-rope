#!/usr/bin/env python3
"""Freeze canonical MrRoPE-Pro S=2 for the Kanana 64K pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.fixed_rope_three_interfaces_20260913.tables import (
    build_analytic,
    make_receipt,
    model_geometry,
)
from .prepare import MODEL_ID, TAILSPLINE_SCALE, _check_geometry, _write_once_or_equal


def freeze_mrpro(model: Path, root: Path) -> dict:
    config = json.loads((model / "config.json").read_text())
    geometry = _check_geometry(config)
    values, gain, construction = build_analytic(
        config, method="mrpro", scale=TAILSPLINE_SCALE,
        low=None, high=None, depth=1.0, gain=None,
    )
    receipt = make_receipt(
        candidate_id="kanana_canonical_mrrope_pro_s2_64k_v1",
        model_id=MODEL_ID,
        role="baseline",
        scale=TAILSPLINE_SCALE,
        geometry=geometry,
        values=values,
        gain=gain,
        construction={
            **construction,
            "target_length": 65_536,
            "physical_extension_ratio": 2.0,
        },
        source="analytic:canonical_mrrope_pro_s2",
        changed_variables=["runtime_rope_frequency_table", "global_rotary_gain"],
    )
    path = root / "tables" / "mrpro.json"
    _write_once_or_equal(path, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    receipt = freeze_mrpro(args.model.resolve(), args.root.resolve())
    print(json.dumps({
        "status": receipt["status"],
        "table_sha256_float32": receipt["table_sha256_float32"],
        "gain": receipt["gain"],
        "band": receipt["table"]["construction"].get("band_envelope"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
