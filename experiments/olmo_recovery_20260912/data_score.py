#!/usr/bin/env python3
"""Audit complete raw generations and score whole answers without substring shortcuts."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from experiments.evq_recovery.data import qa_scores


def read(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--generations", type=Path, required=True)
    p.add_argument("--eos-token-id", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    inputs = {row["id"]: row for row in read(args.inputs)}
    outputs = read(args.generations)
    if len(outputs) != len(inputs) or {row["id"] for row in outputs} != set(inputs):
        raise ValueError("generation panel is incomplete, duplicated, or unmatched")
    cells = defaultdict(list)
    details = []
    for result in outputs:
        source = inputs[result["id"]]
        tokens = result.get("generated_ids", [])
        if len(tokens) > source["generation_budget"]:
            raise ValueError(f"generation cap exceeded: {result['id']}")
        eos = bool(tokens and tokens[-1] == args.eos_token_id)
        if result.get("eos_terminated") != eos:
            raise ValueError(f"EOS receipt drift: {result['id']}")
        text = result["prediction"]
        scores = qa_scores(text, source["references"])
        literal_exact = any(text == reference for reference in source["references"])
        row = {"id": result["id"], "source_id": source["source_id"], "task": source["task"],
               "length_bucket": source["length_bucket"], "f1": scores["f1"], "normalized_exact": scores["exact"],
               "literal_complete_exact_plus_eos": float(literal_exact and eos), "eos": eos,
               "hit_cap": len(tokens) == source["generation_budget"] and not eos,
               "empty": not text.strip(), "generated_tokens_including_eos": len(tokens)}
        details.append(row)
        cells[source["task"], source["length_bucket"]].append(row)
    summary = {}
    for key, values in cells.items():
        summary[f"{key[0]}/{key[1]}"] = {"n": len(values),
            "whole_response_f1": float(np.mean([x["f1"] for x in values])),
            "normalized_exact": float(np.mean([x["normalized_exact"] for x in values])),
            "literal_complete_exact_plus_eos": float(np.mean([x["literal_complete_exact_plus_eos"] for x in values])),
            "eos_rate": float(np.mean([x["eos"] for x in values])), "cap_hits": sum(x["hit_cap"] for x in values),
            "empty_outputs": sum(x["empty"] for x in values),
            "mean_generated_tokens_including_eos": float(np.mean([x["generated_tokens_including_eos"] for x in values]))}
    payload = {"status": "COMPLETE", "summary": summary, "rows": details,
               "metric_boundary": "full response only; literal exact plus terminal EOS is distinct from normalized exact and token F1; no substring criterion"}
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "COMPLETE", "rows": len(details), "cells": len(summary)}))


if __name__ == "__main__":
    main()
