"""Freeze output-blind, fixed-distance reference-risk predictions for binding.

The frozen NCP objective averages over causal separations. This diagnostic uses
the underlying single-distance loss r(d*omega), not that averaged risk R(phi).
It is a prediction of the declared aligned/unit-pair reference model only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.native_contrastive_proximal_20260915.tables import reference_fourier


def fixed_distance_risk(phases, a0, coefficients):
    phases = np.asarray(phases, dtype=float)
    harmonics = np.arange(1, len(coefficients) + 1)
    return a0 + np.sum(coefficients * np.cos(phases[..., None] * harmonics), axis=-1)


def freeze(rows, native_values, candidate_values):
    native, candidate = (np.asarray(v, dtype=float) for v in (native_values, candidate_values))
    if native.ndim != 1 or native.shape != candidate.shape or not np.isfinite(native).all() or not np.isfinite(candidate).all():
        raise ValueError("invalid table shapes or values")
    a0, coefficients = reference_fourier()
    predictions = []
    for row in rows:
        if row["task"] != "native_binding":
            continue
        evidence = [e for e in row["evidence_positions"] if e["source"] == row["query_node"]]
        if len(evidence) != 1:
            raise ValueError("query must identify exactly one evidence record")
        # Average every token in the correct evidence record, fixed before outputs.
        e = evidence[0]
        distances = row["input_tokens"] - 1 - np.arange(e["token_start"], e["token_end"])
        difference = fixed_distance_risk(distances[:, None] * native, a0, coefficients) - fixed_distance_risk(distances[:, None] * candidate, a0, coefficients)
        predictions.append({"row_id": row["row_id"], "group_id": row["group_id"],
                            "length_cap": row["length_cap"], "layout": row["intervention"]["layout"],
                            "prompt_sha256": row["prompt_sha256"],
                            "mean_evidence_distance": float(distances.mean()),
                            "reference_risk_reduction": float(difference.mean())})
    if not predictions:
        raise ValueError("no binding rows")
    summary = {}
    for length in sorted({p["length_cap"] for p in predictions}):
        values = {layout: float(np.mean([p["reference_risk_reduction"] for p in predictions
                                         if p["length_cap"] == length and p["layout"] == layout]))
                  for layout in ("near", "far")}
        summary[str(length)] = {**values, "predicted_far_minus_near": values["far"] - values["near"]}
    return {"status": "OUTPUT_BLIND_REFERENCE_PREDICTIONS_V1", "model_execution": False,
            "prediction_rows": len(predictions), "rows": predictions, "by_length": summary,
            "averaging": "equal frequency pairs and all tokens in the query-selected evidence record",
            "interpretation": "Positive reduction means lower reference loss; its sign and distance interaction are prospective hypotheses, not proven task-accuracy predictions. Three-hop chain behavior is not reduced to this one-pair retrieval loss."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--native-table", type=Path, required=True)
    parser.add_argument("--candidate-table", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    def values(path):
        raw = json.loads(path.read_text())
        return raw.get("table", raw)["values_float32"]
    rows = [json.loads(line) for line in args.panel.read_text().splitlines() if line.strip()]
    result = freeze(rows, values(args.native_table), values(args.candidate_table))
    if args.out.exists() and json.loads(args.out.read_text()) != result:
        raise ValueError("frozen predictions already exist for a different contract")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["by_length"]))


if __name__ == "__main__":
    main()
