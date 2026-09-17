#!/usr/bin/env python3
"""Freeze the unique NTS2 table from public OLMo RoPE geometry."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from experiments.fixed_rope_three_interfaces_20260913.tables import (
    atomic_json,
    build_analytic,
    make_receipt,
    model_geometry,
    tensor_sha256,
)


EXPECTED_TABLE_SHA256 = "0ccaecb736d94072579e76caa3ae12fa65ac565a2f1b7a4e7f10287c51938485"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    geometry = model_geometry(config)
    values, canonical_gain, construction = build_analytic(
        config, method="tailspline", scale=2.0, low=None, high=None,
        depth=1.0, gain=None,
    )
    if tensor_sha256(values) != EXPECTED_TABLE_SHA256:
        raise ValueError("NTS2 frequency table differs from the preregistered FP32 identity")
    midpoint_gain = math.sqrt(canonical_gain)
    receipt = make_receipt(
        candidate_id="native_tailspline_s2_midgain_v1",
        model_id="olmo2_0425_1b_instruct_native4k",
        role="candidate",
        scale=2.0,
        geometry=geometry,
        values=values,
        gain=midpoint_gain,
        construction={
            **construction,
            "gain_rule": "sqrt(1+0.1*ln(2)); geometric midpoint in attention-logit scale",
            "canonical_gain": canonical_gain,
            "midpoint_gain": midpoint_gain,
            "method": "native_tailspline_s2_midgain_v1",
        },
        source="analytic:tailspline_s2_plus_logit_midpoint_gain",
        changed_variables=["support", "internal_frequency_allocation", "global_rotary_gain"],
    )
    if args.out.exists():
        if json.loads(args.out.read_text()) != receipt:
            raise ValueError("existing NTS2 receipt drift")
    else:
        atomic_json(args.out, receipt)
    print(json.dumps({
        "status": receipt["status"],
        "table_sha256_float32": receipt["table_sha256_float32"],
        "gain": midpoint_gain,
        "band": receipt["table"]["construction"]["band_envelope"],
        "out": str(args.out),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
