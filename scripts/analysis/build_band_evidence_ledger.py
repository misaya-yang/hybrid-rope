#!/usr/bin/env python3
"""Normalize completed remote band screens into a portable JSON/CSV ledger."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


MODELS = {
    "Llama-3-8B-Instruct": {"base": 500_000.0, "native_length": 8192, "head_dim": 128, "pair_count": 64},
    "OLMo-2-0425-1B-Instruct": {"base": 500_000.0, "native_length": 4096, "head_dim": 128, "pair_count": 64},
    "Qwen2.5-1.5B-Instruct": {"base": 1_000_000.0, "native_length": 32768, "head_dim": 128, "pair_count": 64},
}


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def table_hash(table: dict) -> str:
    values = np.asarray(table["values_float32"], dtype="<f4")
    if values.shape != (64,):
        raise ValueError("ledger expects 64 RoPE pairs")
    return hashlib.sha256(values.tobytes()).hexdigest()


def table_from_registry(model: str, entry: dict, *, scale: float | None = None) -> dict:
    construction = entry["construction"]
    exponents = construction.get("exponents", construction.get("cumulative_exponents"))
    if exponents is None:
        raise ValueError("registered table lacks reconstructable exponents")
    spec = MODELS[model]
    if construction.get("method") == "mrpro":
        # The archived MrPro path scaled the runtime Native FP32 tensor.
        from scripts.experiments.cross_audit.tables import native_table

        native = native_table(spec["head_dim"], spec["base"]).astype(np.float64)
    else:
        native = np.power(
            spec["base"],
            -np.arange(spec["pair_count"], dtype=np.float64) / spec["pair_count"],
        )
    table_scale = float(construction.get("scale", scale))
    values = (native * np.power(table_scale, -np.asarray(exponents))).astype(np.float32)
    table = {"values_float32": values.tolist(), "gain": entry["gain"], "construction": construction}
    if table_hash(table) != entry["sha256_float32"]:
        raise ValueError("registered table reconstruction differs from stored hash")
    return table


def native_turns(model: str, band: tuple[int, int] | None) -> list[float] | None:
    if band is None:
        return None
    spec = MODELS[model]
    return [
        spec["native_length"] * spec["base"] ** (-slot / spec["pair_count"]) / (2.0 * math.pi)
        for slot in band
    ]


def metrics(summary: dict) -> dict:
    cells = defaultdict(list)
    for key, value in summary.get("generation_metrics", {}).items():
        parts = key.split("/")
        cells[int(parts[-1])].append((parts[-2], value))
    by_length = {}
    for length, values in sorted(cells.items()):
        tasks = {
            task: {
                "rows": value["rows"],
                "official": value.get("ruler_official_score"),
                "exact_plus_eos": value.get("exact_plus_eos"),
                "eos_rate": value.get("ended_eos"),
                "cap_rate": value.get("hit_cap"),
            }
            for task, value in values
        }
        official = [value["official"] for value in tasks.values() if value["official"] is not None]
        by_length[str(length)] = {
            "task_macro_official": sum(official) / len(official),
            "tasks": tasks,
        }
    lm = {}
    for length, value in summary.get("lm_metrics", {}).items():
        lm[length] = {
            "kind": "whole_next_token_nll",
            "documents": value["documents"],
            "nll": value["whole_nll"],
            "ppl": math.exp(value["whole_nll"]),
        }
    return {"generation": by_length, "language_modeling": lm}


def tail_metrics(summary: dict) -> dict:
    return {
        length: {
            "kind": summary["metric"],
            "documents": value["documents"],
            "nll": value["mean_tail_nll"],
            "ppl": value["tail_ppl"],
        }
        for length, value in summary["by_length"].items()
    }


def confirmed_metrics(initial: Path, extra: Path) -> dict:
    combined = [row for row in rows(initial) + rows(extra) if row["length_cap"] == 65536]
    cells = defaultdict(list)
    for row in combined:
        cells[row["task"]].append(row)
    tasks = {}
    for task, values in sorted(cells.items()):
        tasks[task] = {
            "rows": len(values),
            "official": sum(row["ruler_official_score"] for row in values) / len(values),
            "exact_plus_eos": sum(row["exact_plus_eos"] for row in values) / len(values),
            "eos_rate": sum(row["ended_eos"] for row in values) / len(values),
            "cap_rate": sum(row["hit_cap"] for row in values) / len(values),
        }
    return {
        "length": 65536,
        "task_macro_official": sum(value["official"] for value in tasks.values()) / len(tasks),
        "tasks": tasks,
    }


def band_from_label(label: str) -> tuple[int, int] | None:
    match = re.search(r"Band(\d+)_(\d+)", label)
    return tuple(map(int, match.groups())) if match else None


def add_record(
    output: list[dict], *, model: str, label: str, scale: float,
    band: tuple[int, int] | None, shape: str, table: dict,
    panel: str, result_type: str, result_metrics: dict,
    confirmation: dict | None = None,
) -> None:
    spec = MODELS[model]
    output.append({
        "model_identity": model,
        "checkpoint_identity": model,
        "head_dim": spec["head_dim"],
        "pair_count": spec["pair_count"],
        "rope_base": spec["base"],
        "native_length": spec["native_length"],
        "target_scale": scale,
        "arm": label,
        "band": list(band) if band else None,
        "normalized_band": [slot / (spec["pair_count"] - 1) for slot in band] if band else None,
        "native_turns_at_band": native_turns(model, band),
        "band_width": band[1] - band[0] if band else None,
        "shape_identity": shape,
        "gain": float(table["gain"]),
        "table_values_sha256": table_hash(table),
        "source_result_type": result_type,
        "input_manifest": panel,
        "metrics": result_metrics,
        "confirmation": confirmation,
    })


def build(root: Path) -> list[dict]:
    output: list[dict] = []

    for scale, directory in ((2, "llama_minimal_band_s2_20260913"), (4, "llama_minimal_band_s4_20260913")):
        screen = root / directory
        manifest = read(screen / "manifest.json")
        for label in manifest["arms"]:
            summary = read(screen / "runs" / label / "summary.json")
            band = band_from_label(label)
            if label.startswith("BM"):
                shape = "boundary_matched"
                band = (18, 35)
            elif label.startswith("MrPro"):
                shape = "MrRoPE-Pro"
                band = (18, 35)
            else:
                shape = "C42V24 remapped"
            add_record(
                output, model="Llama-3-8B-Instruct", label=label, scale=scale,
                band=band, shape=shape, table=summary["table"],
                panel=f"ignored_remote:{directory}/manifest.json", result_type="selection",
                result_metrics=metrics(summary),
            )

    g8 = root / "fixed_table_interval_20260913"
    g8_registry = read(g8 / "summary.json")["tables"]
    for label, band, shape in (
        ("BM_g8", (18, 35), "boundary_matched"),
        ("MrPro_g8", (18, 35), "MrRoPE-Pro"),
        ("SolverProfile_g8", (14, 32), "SolverC42 literal profile"),
        ("SolverProfileBand16_34_g8", (16, 34), "SolverC42 remapped"),
        ("SolverProfileBandRemap_g8", (18, 35), "SolverC42 remapped"),
        ("WindingMatched_g8", None, "WindingMatched author proposal"),
    ):
        summary = read(g8 / label / "summary.json")
        table = summary.get("table")
        if table is None:
            table = table_from_registry("Llama-3-8B-Instruct", g8_registry[label], scale=8)
        add_record(
            output, model="Llama-3-8B-Instruct", label=label, scale=8,
            band=band, shape=shape, table=table,
            panel="ignored_remote:fixed_table_interval_20260913/prepared_llama_g8_r0",
            result_type="selection", result_metrics=metrics(summary),
        )

    olmo = root / "olmo_minimal_band_s4_20260913"
    direct = root / "llama_s4_band16_34_to_olmo_20260913"
    conditioned = root / "model_conditioned_range_20260913"
    olmo_paths = {
        "C42Band12_30_s4": (olmo / "runs/C42Band12_30_s4/generation/summary.json", olmo / "runs/C42Band12_30_s4/ppl/summary.json"),
        "C42Band13_31_s4": (olmo / "runs/C42Band13_31_s4/generation/summary.json", olmo / "runs/C42Band13_31_s4/ppl/summary.json"),
        "C42Band13_32_s4": (olmo / "runs/C42Band13_32_s4/generation/summary.json", None),
        "C42Band14_30_s4": (olmo / "runs/C42Band14_30_s4/generation/summary.json", None),
        "C42Band14_31_s4": (olmo / "runs/C42Band14_31_s4/generation/summary.json", olmo / "runs/C42Band14_31_s4/ppl/summary.json"),
        "C42Band14_32_s4": (conditioned / "internal_confirm_r0/C42V24_g4/summary.json", direct / "ppl_band14_32/summary.json"),
        "C42Band14_33_s4": (olmo / "runs/C42Band14_33_s4/generation/summary.json", None),
        "C42Band15_32_s4": (olmo / "runs/C42Band15_32_s4/generation/summary.json", olmo / "runs/C42Band15_32_s4/ppl/summary.json"),
        "C42Band15_33_s4": (olmo / "runs/C42Band15_33_s4/generation/summary.json", olmo / "runs/C42Band15_33_s4/ppl/summary.json"),
        "C42Band16_34_s4": (direct / "generation/summary.json", direct / "ppl/summary.json"),
    }
    for label, (generation_path, ppl_path) in olmo_paths.items():
        summary = read(generation_path)
        result_metrics = metrics(summary)
        if ppl_path:
            result_metrics["language_modeling"] = tail_metrics(read(ppl_path))
        add_record(
            output, model="OLMo-2-0425-1B-Instruct", label=label, scale=4,
            band=band_from_label(label), shape="C42V24 remapped", table=summary["table"],
            panel="ignored_remote:model_conditioned_range_20260913/data_r0 internal_confirm",
            result_type="selection", result_metrics=result_metrics,
        )

    qwen = root / "qwen15_minimal_band_s2_20260913"
    for label in ("Native", "BM_s2", "MrPro_s2", "C42Band14_32_s2", "C42Band16_34_s2", "C42Band22_39_s2", "C42Band23_40_s2"):
        generation = read(qwen / "runs" / label / "generation" / "summary.json")
        ppl = read(qwen / "runs" / label / "ppl" / "summary.json")
        result_metrics = metrics(generation)
        result_metrics["language_modeling"] = tail_metrics(ppl)
        band = band_from_label(label)
        if label == "Native":
            shape = "identity"
        elif label.startswith("BM"):
            band, shape = (23, 40), "boundary_matched"
        elif label.startswith("MrPro"):
            band, shape = (23, 40), "MrRoPE-Pro"
        else:
            shape = "C42V24 remapped"
        confirm_path = qwen / "confirm_remaining" / label / "generation" / "generations.jsonl"
        confirmation = None
        if confirm_path.exists():
            confirmation = confirmed_metrics(
                qwen / "runs" / label / "generation" / "generations.jsonl",
                confirm_path,
            )
        add_record(
            output, model="Qwen2.5-1.5B-Instruct", label=label, scale=2,
            band=band, shape=shape, table=ppl["table"],
            panel="ignored_remote:qwen15_minimal_band_s2_20260913/niah_screen.jsonl",
            result_type="selection", result_metrics=result_metrics, confirmation=confirmation,
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path, required=True)
    args = parser.parse_args()
    records = build(args.root)
    payload = {
        "status": "COMPLETE",
        "coordinate": "native_turns_at_band = L_native * omega_slot / (2*pi)",
        "table_hash": "sha256 of little-endian float32 values only; gain is a separate field",
        "remote_root": "westc:53405 /root/autodl-tmp",
        "records": records,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    fields = [
        "model_identity", "checkpoint_identity", "head_dim", "pair_count", "rope_base",
        "native_length", "target_scale", "arm", "band", "normalized_band",
        "native_turns_at_band", "band_width", "shape_identity", "gain",
        "table_values_sha256", "source_result_type", "input_manifest", "metrics", "confirmation",
    ]
    with args.csv_out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({
                field: json.dumps(record[field], sort_keys=True) if isinstance(record[field], (dict, list)) else record[field]
                for field in fields
            })
    print(json.dumps({"status": "COMPLETE", "records": len(records)}))


if __name__ == "__main__":
    main()
