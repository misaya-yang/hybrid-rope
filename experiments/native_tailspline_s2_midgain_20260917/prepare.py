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


SUPPORTED_MODELS = {
    ("olmo2", 4096, 500_000.0, 64): {
        "model_id": "olmo2_0425_1b_instruct_native4k",
        "table_sha256": "0ccaecb736d94072579e76caa3ae12fa65ac565a2f1b7a4e7f10287c51938485",
    },
    ("llama", 8192, 500_000.0, 64): {
        "model_id": "meta_llama3_8b_instruct_native8k",
        "table_sha256": "7d6739a3dc98c909a09dd05662d961a2408d8cefbceaf8dac3a6fe1d0983ced0",
    },
    ("glm4", 32768, 10_000.0, 32): {
        "model_id": "glm4_9b_0414_native32k",
        "table_sha256": "fac0cf96475697c8b7d978b62f0cafca449093955f59f444aa1eea400402ba26",
    },
    ("qwen2", 32768, 1_000_000.0, 64): {
        "model_id": "qwen2_5_3b_instruct_native32k",
        "table_sha256": "bb55a80244758dc071953c5e557e521490f1e1a1ee4c601c8bfb8bd814af19ca",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    geometry = model_geometry(config)
    key = (
        geometry["model_type"], geometry["native_length"],
        geometry["base"], geometry["pairs"],
    )
    if key not in SUPPORTED_MODELS:
        raise ValueError("NTS2 requires a preregistered public model geometry")
    model_identity = SUPPORTED_MODELS[key]
    values, canonical_gain, construction = build_analytic(
        config, method="tailspline", scale=2.0, low=None, high=None,
        depth=1.0, gain=None,
    )
    if tensor_sha256(values) != model_identity["table_sha256"]:
        raise ValueError("NTS2 frequency table differs from the preregistered FP32 identity")
    midpoint_gain = math.sqrt(canonical_gain)
    receipt = make_receipt(
        candidate_id="native_tailspline_s2_midgain_v1",
        model_id=model_identity["model_id"],
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
