#!/usr/bin/env python3
"""Export the existing 32-document ProofPile PPL curve in MrRoPE-style form."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from experiments.fixed_rope_three_interfaces_20260913.tailspline_llama_classic_report import (
    LENGTHS,
    log_auc,
)


ARMS = ("tailspline", "mrpro")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def summarize(rows: list[dict], documents: list[int]) -> dict:
    curve = {}
    by_length = {}
    for length in LENGTHS:
        selected = [
            row for row in rows
            if int(row["document"]) in documents and int(row["length"]) == length
        ]
        if len(selected) != len(documents):
            raise ValueError(f"ProofPile PPL coverage drift at {length}")
        loss = sum(float(row["whole_loss_sum"]) for row in selected)
        tokens = sum(int(row["whole_target_count"]) for row in selected)
        ppl = math.exp(loss / tokens)
        curve[length] = ppl
        by_length[str(length)] = {
            "documents": len(selected), "target_tokens": tokens,
            "whole_nll": loss / tokens, "whole_ppl": ppl,
        }
    return {"by_length": by_length, "log_length_ppl_auc": log_auc(curve, list(LENGTHS))}


def bootstrap(runs, documents, *, draws=20_000, seed=20260929):
    mappings = {
        arm: {(int(row["document"]), int(row["length"])): row for row in rows}
        for arm, rows in runs.items()
    }
    rng = np.random.default_rng(seed)
    delta = np.empty(draws)
    for draw in range(draws):
        sampled = rng.choice(documents, size=len(documents), replace=True)
        auc = {}
        for arm, mapping in mappings.items():
            curve = {}
            for length in LENGTHS:
                rows = [mapping[(int(document), length)] for document in sampled]
                nll = sum(float(row["whole_loss_sum"]) for row in rows) / sum(
                    int(row["whole_target_count"]) for row in rows
                )
                curve[length] = math.exp(nll)
            auc[arm] = log_auc(curve, list(LENGTHS))
        delta[draw] = auc["tailspline"] - auc["mrpro"]
    return {
        "draws": draws, "seed": seed,
        "resampling": "paired ProofPile documents; same sampled document shared across lengths",
        "mean_delta_tailspline_minus_mrpro": float(delta.mean()),
        "ci95": [float(value) for value in np.quantile(delta, [0.025, 0.975])],
        "probability_delta_lt_zero": float(np.mean(delta < 0.0)),
    }


def plot(report: dict, prefix: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(5.4, 3.6), constrained_layout=True)
    for arm, color, marker in (("tailspline", "#2563eb", "o"), ("mrpro", "#dc2626", "s")):
        y = [report["arms"][arm]["by_length"][str(length)]["whole_ppl"] for length in LENGTHS]
        axis.plot([length / 1024 for length in LENGTHS], y, marker=marker, color=color, label=arm)
    axis.set_xlabel("Context length (K tokens)")
    axis.set_ylabel("ProofPile perplexity (lower is better)")
    axis.set_xticks([length / 1024 for length in LENGTHS])
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.savefig(prefix.with_suffix(".png"), dpi=220)
    fig.savefig(prefix.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppl-manifest", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, help="ARM=RUN_DIR")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = {}
    for item in args.run:
        arm, value = item.split("=", 1)
        paths[arm] = Path(value)
    if set(paths) != set(ARMS):
        raise ValueError("ProofPile report requires exactly tailspline and mrpro")
    manifest = json.loads(args.ppl_manifest.read_text())
    documents = [
        int(record["document"])
        for record in manifest["document_records"]
        if record["dataset"] == "proofpile"
    ]
    if len(documents) != 32:
        raise ValueError("frozen PPL manifest must contain 32 ProofPile documents")
    runs = {arm: read_jsonl(path / "lm_rows.jsonl") for arm, path in paths.items()}
    report = {
        "status": "TAILSPLINE_MRPRO_PROOFPILE32_PPL_PARITY_COMPLETE_V1",
        "model": "Meta-Llama-3-8B-Instruct",
        "documents": len(documents),
        "lengths": list(LENGTHS),
        "precision": "bfloat16 checkpoint forward with float32 logits/loss accumulation",
        "arms": {arm: summarize(rows, documents) for arm, rows in runs.items()},
        "paired_inference": bootstrap(runs, documents),
        "comparison_note": (
            "Uses 32 frozen ProofPile test documents versus the 10 randomly sampled "
            "ProofPile sequences disclosed by MrRoPE; evaluates only the claimed 1x-4x range."
        ),
    }
    report["delta_tailspline_minus_mrpro"] = {
        "by_length_ppl": {
            str(length): (
                report["arms"]["tailspline"]["by_length"][str(length)]["whole_ppl"]
                - report["arms"]["mrpro"]["by_length"][str(length)]["whole_ppl"]
            ) for length in LENGTHS
        },
        "log_length_ppl_auc": (
            report["arms"]["tailspline"]["log_length_ppl_auc"]
            - report["arms"]["mrpro"]["log_length_ppl_auc"]
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "proofpile32_ppl_curve.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    with (args.out_dir / "proofpile32_ppl_curve.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("arm", "length", "documents", "target_tokens", "whole_nll", "whole_ppl"))
        for arm in ARMS:
            for length in LENGTHS:
                row = report["arms"][arm]["by_length"][str(length)]
                writer.writerow((arm, length, row["documents"], row["target_tokens"], row["whole_nll"], row["whole_ppl"]))
    plot(report, args.out_dir / "proofpile32_ppl_curve")
    print(json.dumps({"status": report["status"], "delta": report["delta_tailspline_minus_mrpro"]}))


if __name__ == "__main__":
    main()
