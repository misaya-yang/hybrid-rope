#!/usr/bin/env python3
"""Promote selected ignored result JSONs into portable rebuttal evidence.

The source files remain ignored because they are broad local audit artifacts.
This script verifies their exact SHA256 identities and emits only anonymous,
repo-relative JSON snapshots that can travel to another checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MLA_SOURCE = ROOT / "results" / "eval_3seeds_full_results.json"
DEFAULT_QUALITY_SOURCE = (
    ROOT
    / "results"
    / "core_text"
    / "phase21b"
    / "phase21b_quality_454m_full_eval.json"
)
DEFAULT_BASE_SOURCE_DIR = ROOT / "results" / "core_text" / "phase18_base_sweep"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "curated"

BASE_RUN_PATHS = {
    "base_10000_geo": "d64_base10000_geo_tau0.00_seed42/result.json",
    "base_10000_evq": "d64_base10000_evq_tau2.83_seed42/result.json",
    "base_500000_geo": "d64_base500000_geo_tau0.00_seed42/result.json",
    "base_500000_evq": "d64_base500000_evq_tau2.83_seed42/result.json",
}

EXPECTED_SHA256 = {
    "mla": "1e44d30bb880e4b7427ae55bd7034782989152bd2afca9217495f9b8ece30953",
    "phase11_raw": "6bdf97335365ea3a92c15ff84fc52f292ddad96b0f6e142f8b98199295dffa30",
    "phase11_yarn": "1f9550c46fa5b51b24b4d2e805c4dbbba8d639c19f664e812072bf8659b85321",
    "quality": "5fc3254cb7b44a918328056ccd505d01e5539dc596d4c06273c9914ec93e3caa",
    "base_10000_geo": "c367ff7ae073adb2c811a38f223305f96d44789c6fc2b293df2ba1675fc466d0",
    "base_10000_evq": "0821423633d821566bd5414e4fde75bb5a4f175197f8242bd00407d79d7571e5",
    "base_500000_geo": "9f495367a7f2975e00278512777978f7b219308534f34f75fa3799fcf1b40d7d",
    "base_500000_evq": "21447e44789fc13b86e8a389ba43f2eab4d31837acd478ac7267134057cee3dc",
}


def default_phase11_dir(root: Path = ROOT) -> Path:
    local = root / "results" / "core_text" / "phase11"
    if local.is_dir():
        return local
    return (
        root
        / "07 - rebuttal"
        / "all_paper_experiment_code"
        / "branch_archives"
        / "backup__2026-03-06"
        / "high_value_artifacts"
        / "results"
        / "phase11"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_verified_json(path: Path, expected_sha256: str) -> Any:
    actual = sha256(path)
    if actual != expected_sha256:
        raise ValueError(
            f"Unexpected source identity for {path.name}: "
            f"expected {expected_sha256}, got {actual}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def build_mla_snapshot(source: Path) -> dict[str, Any]:
    payload = load_verified_json(source, EXPECTED_SHA256["mla"])
    if payload.get("seeds") != [42, 43, 88]:
        raise ValueError("MLA source does not contain the expected seeds [42, 43, 88]")
    return {
        "schema_version": 1,
        "provenance_status": "raw-json-backed",
        "artifact_role": "Portable copy of the ignored three-seed MLA evaluation JSON.",
        "source": {
            "path_hint": "results/eval_3seeds_full_results.json",
            "sha256": EXPECTED_SHA256["mla"],
            "local_source_policy": "ignored; exact values promoted into this tracked snapshot",
        },
        "claim_boundary": (
            "Primary III 8K/500M scarce-channel stress test only. The tested "
            "tau=1.414 is an empirical d_eff=128 operating convention, not a "
            "global theorem or a direct comparison of d_eff conventions."
        ),
        "seeds": payload["seeds"],
        "eval_lengths": payload["eval_lengths"],
        "progression": payload["progression"],
        "extended": payload["extended"],
        "summary": payload["summary"],
    }


def build_phase11_snapshot(raw_source: Path, yarn_source: Path) -> dict[str, Any]:
    raw = load_verified_json(raw_source, EXPECTED_SHA256["phase11_raw"])
    yarn = load_verified_json(yarn_source, EXPECTED_SHA256["phase11_yarn"])
    if len(raw) != 9 or len(yarn) != 9:
        raise ValueError("Expected nine raw and nine YaRN Phase11 run records")
    if set(raw) != set(yarn):
        raise ValueError("Phase11 raw and YaRN run identifiers do not match")
    return {
        "schema_version": 1,
        "provenance_status": "raw-json-backed",
        "artifact_role": "Portable copy of two ignored archival Phase11 evaluator payloads.",
        "sources": {
            "raw": {
                "path_hint": "results_phase11_raw.json",
                "sha256": EXPECTED_SHA256["phase11_raw"],
            },
            "yarn": {
                "path_hint": "results_phase11_yarn.json",
                "sha256": EXPECTED_SHA256["phase11_yarn"],
            },
        },
        "claim_boundary": (
            "These are L_train=256 Geo/EVQ tau=2/tau=4 three-seed records, not "
            "the L_train=128 Geo/DAPE/EVQ replication requested for Primary II. "
            "The raw-only and scaling-evaluator payloads are preserved separately "
            "because their embedded raw evaluations are not numerically identical."
        ),
        "raw_runs": raw,
        "yarn_runs": yarn,
    }


def build_quality_snapshot(source: Path) -> dict[str, Any]:
    payload = load_verified_json(source, EXPECTED_SHA256["quality"])
    setup = payload.get("setup", {})
    if setup.get("eval_samples") != 2086:
        raise ValueError("QuALITY source does not contain the expected 2,086 examples")

    row_specs = (
        ("4k_raw", "4K raw", 4096, "raw", payload["results_raw"]["4k"]),
        ("8k_raw", "8K raw", 8192, "raw", payload["results_raw"]["8k"]),
        (
            "8k_yarn_s2",
            "8K YaRN",
            8192,
            "yarn_s2",
            payload["results_yarn"]["8k_yarn_scale2"],
        ),
        ("16k_raw", "16K raw", 16384, "raw", payload["results_raw"]["16k"]),
    )
    rows = []
    for row_id, label, context_length, mode, raw in row_specs:
        rows.append(
            {
                "id": row_id,
                "label": label,
                "context_length": context_length,
                "mode": mode,
                "geo": {
                    "accuracy": raw["geo_accuracy"],
                    "correct": raw["geo_correct"],
                    "gold_nll": raw["geo_gold_nll"],
                },
                "evq": {
                    "accuracy": raw["evq_accuracy"],
                    "correct": raw["evq_correct"],
                    "gold_nll": raw["evq_gold_nll"],
                },
            }
        )

    return {
        "schema_version": 1,
        "provenance_status": "raw-json-backed",
        "experiment": "Phase 21B QuALITY QA full evaluation",
        "source": {
            "path_hint": "results/core_text/phase21b/phase21b_quality_454m_full_eval.json",
            "sha256": EXPECTED_SHA256["quality"],
            "local_source_policy": "ignored; sanitized aggregate promoted into this tracked snapshot",
        },
        "source_report": "docs/exp/2026-03-12_phase21b_454m_full_eval_report.md",
        "protocol": {
            "model": "454M decoder-only transformer",
            "architecture": setup["architecture"],
            "pretrain_context": setup["pretrain_context"],
            "continue_train_context": setup["continue_train_context"],
            "finetune_context": setup["finetune_context"],
            "finetune_steps": setup["finetune_steps"],
            "finetune_seed": setup["finetune_seed"],
            "eval_samples": setup["eval_samples"],
            "eval_scoring": "length_normalized_option_nll",
            "random_baseline_accuracy": setup["random_baseline_accuracy"],
            "status": "full_eval_complete_except_32k",
        },
        "rows": rows,
        "supersedes": {
            "eval_samples": 200,
            "status": "superseded_small_sample_pilot",
            "reason": (
                "The pilot accuracy deltas were inflated by small-sample noise "
                "and must not be used in Figure 8."
            ),
        },
        "scope": (
            "Supporting, raw-JSON-backed probability-space diagnostic; all "
            "accuracy values remain near the 25% random baseline and do not "
            "establish a stable accuracy gain."
        ),
    }


def base_source_paths(source_dir: Path) -> dict[str, Path]:
    return {key: source_dir / rel for key, rel in BASE_RUN_PATHS.items()}


def build_base_snapshot(sources: dict[str, Path]) -> dict[str, Any]:
    expected_keys = set(BASE_RUN_PATHS)
    if set(sources) != expected_keys:
        raise ValueError(f"Base sweep sources must be exactly {sorted(expected_keys)}")
    payloads = {
        key: load_verified_json(path, EXPECTED_SHA256[key])
        for key, path in sources.items()
    }
    lengths = ("512", "1024", "2048", "4096")
    rows = []
    for base in (10000, 500000):
        geo = payloads[f"base_{base}_geo"]["ppl"]
        evq = payloads[f"base_{base}_evq"]["ppl"]
        if any(length not in geo or length not in evq for length in lengths):
            raise ValueError(f"Base {base} source is missing a required PPL length")
        rows.append(
            {
                "base": base,
                "geo_ppl": {length: geo[length] for length in lengths},
                "evq_ppl": {length: evq[length] for length in lengths},
                "evq_relative_change_percent": {
                    length: round((evq[length] / geo[length] - 1.0) * 100.0, 2)
                    for length in lengths
                },
            }
        )
    return {
        "schema_version": 1,
        "provenance_status": "raw-json-backed",
        "sources": {
            key: {
                "path_hint": f"results/core_text/phase18_base_sweep/{BASE_RUN_PATHS[key]}",
                "sha256": EXPECTED_SHA256[key],
            }
            for key in BASE_RUN_PATHS
        },
        "protocol": {
            "model_parameters": 151900000,
            "train_length": 512,
            "training_tokens": 50003968,
            "dataset": "FineWeb-Edu",
            "head_dim": 64,
            "seed": 42,
            "evq_tau": 2.828,
        },
        "rows": rows,
        "claim_boundary": (
            "Raw-JSON-backed single-seed supporting pilot. It shows that the "
            "direction is not unique to base 500K, but it is not a tuned-base "
            "sweep, does not compare c_pred at base 10K, and cannot establish "
            "the best geometric base."
        ),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mla-source", type=Path, default=DEFAULT_MLA_SOURCE)
    phase11_dir = default_phase11_dir()
    parser.add_argument(
        "--phase11-raw-source",
        type=Path,
        default=phase11_dir / "results_phase11_raw.json",
    )
    parser.add_argument(
        "--phase11-yarn-source",
        type=Path,
        default=phase11_dir / "results_phase11_yarn.json",
    )
    parser.add_argument("--quality-source", type=Path, default=DEFAULT_QUALITY_SOURCE)
    parser.add_argument(
        "--base-source-dir", type=Path, default=DEFAULT_BASE_SOURCE_DIR
    )
    parser.add_argument(
        "--only",
        action="append",
        choices=("mla", "phase11", "quality", "base"),
        help="Build only the selected component; repeat for multiple components.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    components = args.only or ["mla", "phase11", "quality", "base"]
    outputs: list[Path] = []
    if "mla" in components:
        outputs.append(args.output_dir / "table18_mla_3seed_aggregate.json")
        write_json(outputs[-1], build_mla_snapshot(args.mla_source))
    if "phase11" in components:
        outputs.append(args.output_dir / "phase11_l256_3seed_recovered.json")
        write_json(
            outputs[-1],
            build_phase11_snapshot(args.phase11_raw_source, args.phase11_yarn_source),
        )
    if "quality" in components:
        outputs.append(args.output_dir / "quality_454m_full_eval.json")
        write_json(outputs[-1], build_quality_snapshot(args.quality_source))
    if "base" in components:
        outputs.append(args.output_dir / "text_base_10k_500k_pilot.json")
        write_json(
            outputs[-1],
            build_base_snapshot(base_source_paths(args.base_source_dir)),
        )
    print(f"Wrote {len(outputs)} raw-JSON-backed portable rebuttal assets.")


if __name__ == "__main__":
    main()
