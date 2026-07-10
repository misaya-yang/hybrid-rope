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
DEFAULT_PHASE11_DIR = (
    ROOT
    / "07 - rebuttal"
    / "all_paper_experiment_code"
    / "branch_archives"
    / "backup__2026-03-06"
    / "high_value_artifacts"
    / "results"
    / "phase11"
)
DEFAULT_OUTPUT_DIR = ROOT / "data" / "curated"

EXPECTED_SHA256 = {
    "mla": "1e44d30bb880e4b7427ae55bd7034782989152bd2afca9217495f9b8ece30953",
    "phase11_raw": "6bdf97335365ea3a92c15ff84fc52f292ddad96b0f6e142f8b98199295dffa30",
    "phase11_yarn": "1f9550c46fa5b51b24b4d2e805c4dbbba8d639c19f664e812072bf8659b85321",
}


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mla-source", type=Path, default=DEFAULT_MLA_SOURCE)
    parser.add_argument(
        "--phase11-raw-source",
        type=Path,
        default=DEFAULT_PHASE11_DIR / "results_phase11_raw.json",
    )
    parser.add_argument(
        "--phase11-yarn-source",
        type=Path,
        default=DEFAULT_PHASE11_DIR / "results_phase11_yarn.json",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_json(
        args.output_dir / "table18_mla_3seed_aggregate.json",
        build_mla_snapshot(args.mla_source),
    )
    write_json(
        args.output_dir / "phase11_l256_3seed_recovered.json",
        build_phase11_snapshot(args.phase11_raw_source, args.phase11_yarn_source),
    )
    print("Wrote two raw-JSON-backed portable rebuttal assets.")


if __name__ == "__main__":
    main()
