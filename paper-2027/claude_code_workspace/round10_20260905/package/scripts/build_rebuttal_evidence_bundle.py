#!/usr/bin/env python3
"""Build portable rebuttal evidence from verified raw inputs or snapshots.

Tracked snapshots are the portable defaults. Historical raw result JSONs may be
provided explicitly; their exact SHA256 identities are verified before output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "curated"
DEFAULT_MLA_SOURCE = ROOT / "results" / "eval_3seeds_full_results.json"
DEFAULT_MLA_SNAPSHOT_SOURCE = DEFAULT_OUTPUT_DIR / "table18_mla_3seed_aggregate.json"
DEFAULT_PRIMARY1_SOURCE = (
    ROOT / "data" / "results_5090b" / "evq_yarn_10pct_allseeds.json"
)
DEFAULT_PRIMARY2_SEED42_SOURCE = (
    ROOT / "data" / "evq_128tok_results" / "extended_sweep" / "results_final.json"
)
DEFAULT_PRIMARY2_EXTRA_SEEDS_SOURCE = (
    ROOT
    / "data"
    / "evq_128tok_results"
    / "phase7"
    / "multiseed"
    / "results_final.json"
)
DEFAULT_PHASE11_SNAPSHOT_SOURCE = (
    DEFAULT_OUTPUT_DIR / "phase11_l256_3seed_recovered.json"
)
DEFAULT_PHASE11B_SNAPSHOT_SOURCE = (
    DEFAULT_OUTPUT_DIR / "phase11b_125m_l256_3seed.json"
)
DEFAULT_QUALITY_SOURCE = (
    ROOT
    / "results"
    / "core_text"
    / "phase21b"
    / "phase21b_quality_454m_full_eval.json"
)
DEFAULT_BASE_SOURCE_DIR = ROOT / "results" / "core_text" / "phase18_base_sweep"

BASE_RUN_PATHS = {
    "base_10000_geo": "d64_base10000_geo_tau0.00_seed42/result.json",
    "base_10000_evq": "d64_base10000_evq_tau2.83_seed42/result.json",
    "base_500000_geo": "d64_base500000_geo_tau0.00_seed42/result.json",
    "base_500000_evq": "d64_base500000_evq_tau2.83_seed42/result.json",
}

EXPECTED_SHA256 = {
    "mla": "1e44d30bb880e4b7427ae55bd7034782989152bd2afca9217495f9b8ece30953",
    "primary1": "1dbec88efac6d7442796d81fa1d073e3a76b1388dd815764bcb8b619f234511c",
    "primary2_seed42": "980246a9950d7e40e39278a4feef8115e6b35a9feb6e1fe1eb190faac1caf1fd",
    "primary2_extra_seeds": "4fd031f44d966117fa7473eaf405b4503329d81744a6938158fd588901535d47",
    "phase11_snapshot": "8af8bce33e96f70542d943745bebbbaaa7cc65117587950b75c584f06a2f68db",
    "phase11b_snapshot": "783fe586b0953b90329a8bf8ce0ccb52e82567eccc8c3d5fe3000bab831515c0",
    "phase11_raw": "6bdf97335365ea3a92c15ff84fc52f292ddad96b0f6e142f8b98199295dffa30",
    "phase11_yarn": "1f9550c46fa5b51b24b4d2e805c4dbbba8d639c19f664e812072bf8659b85321",
    "phase11b_scaling": "b8ae71708f18b54ed1249c77d26ba2e7aac6d4871a445648487c855740de6d94",
    "phase11b_dape": "44da360db7e8d0d6f964dde59ca52f294a3005d3cb820b6a066406164ef6f70d",
    "quality": "5fc3254cb7b44a918328056ccd505d01e5539dc596d4c06273c9914ec93e3caa",
    "base_10000_geo": "c367ff7ae073adb2c811a38f223305f96d44789c6fc2b293df2ba1675fc466d0",
    "base_10000_evq": "0821423633d821566bd5414e4fde75bb5a4f175197f8242bd00407d79d7571e5",
    "base_500000_geo": "9f495367a7f2975e00278512777978f7b219308534f34f75fa3799fcf1b40d7d",
    "base_500000_evq": "21447e44789fc13b86e8a389ba43f2eab4d31837acd478ac7267134057cee3dc",
}

MLA_RAW_KEYS = ("seeds", "eval_lengths", "progression", "extended", "summary")


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


def _load_snapshot(
    path: Path, expected_sha256: str
) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise ValueError(
            f"Unexpected snapshot identity for {path.name}: "
            f"expected {expected_sha256}, got {actual}"
        )
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in snapshot {path.name}")
    return payload, raw


def _verified_embedded_runs(
    snapshot: dict[str, Any], field: str, source: str, expected_key: str
) -> dict[str, Any]:
    runs = snapshot.get(field)
    if not isinstance(runs, dict):
        raise ValueError(f"Snapshot field {field} must be a JSON object")
    source_metadata = snapshot.get("sources", {}).get(source)
    if not isinstance(source_metadata, dict):
        raise ValueError(f"Snapshot is missing sources.{source} metadata")

    expected = EXPECTED_SHA256[expected_key]
    declared = source_metadata.get("sha256")
    if declared != expected:
        raise ValueError(
            f"Snapshot sources.{source}.sha256 mismatch: "
            f"expected {expected}, got {declared}"
        )
    reconstructed = json.dumps(runs, indent=2).encode("utf-8")
    actual = hashlib.sha256(reconstructed).hexdigest()
    if actual != expected:
        raise ValueError(
            f"Reconstructed {field} identity mismatch: expected {expected}, got {actual}"
        )
    return runs


def validated_phase11_snapshot_bytes(snapshot_source: Path) -> bytes:
    """Validate the tracked Phase11 snapshot and return its original bytes."""
    snapshot, raw = _load_snapshot(
        snapshot_source, EXPECTED_SHA256["phase11_snapshot"]
    )
    raw_runs = _verified_embedded_runs(
        snapshot, "raw_runs", "raw", "phase11_raw"
    )
    yarn_runs = _verified_embedded_runs(
        snapshot, "yarn_runs", "yarn", "phase11_yarn"
    )
    if len(raw_runs) != 9 or len(yarn_runs) != 9:
        raise ValueError("Expected nine raw and nine YaRN Phase11 run records")
    if set(raw_runs) != set(yarn_runs):
        raise ValueError("Phase11 raw and YaRN run identifiers do not match")
    return raw


def validated_phase11b_snapshot_bytes(snapshot_source: Path) -> bytes:
    """Validate the tracked Phase11B snapshot and return its original bytes."""
    snapshot, raw = _load_snapshot(
        snapshot_source, EXPECTED_SHA256["phase11b_snapshot"]
    )
    scaling_runs = _verified_embedded_runs(
        snapshot, "scaling_runs", "scaling", "phase11b_scaling"
    )
    dape_runs = _verified_embedded_runs(
        snapshot, "dape_runs", "dape", "phase11b_dape"
    )
    if len(scaling_runs) != 9 or len(dape_runs) != 6:
        raise ValueError("Expected nine scaling and six DAPE Phase11B run records")

    records = [*scaling_runs.values(), *dape_runs.values()]
    if any(not isinstance(record, dict) for record in records):
        raise ValueError("Phase11B run records must be JSON objects")
    seeds = sorted({record.get("seed") for record in records})
    if seeds != [42, 137, 256]:
        raise ValueError("Phase11B snapshot does not contain seeds [42, 137, 256]")
    if any(record.get("use_dape") is not False for record in scaling_runs.values()):
        raise ValueError("Phase11B scaling snapshot contains an invalid DAPE flag")
    if any(record.get("use_dape") is not True for record in dape_runs.values()):
        raise ValueError("Phase11B DAPE snapshot contains an invalid DAPE flag")
    return raw


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


def reconstruct_mla_source_bytes(snapshot_source: Path) -> bytes:
    """Recreate the evaluator's exact JSON bytes from the portable snapshot."""
    snapshot = json.loads(snapshot_source.read_text(encoding="utf-8"))
    missing = [key for key in MLA_RAW_KEYS if key not in snapshot]
    if missing:
        raise ValueError(f"MLA snapshot is missing raw source keys: {missing}")
    payload = {key: snapshot[key] for key in MLA_RAW_KEYS}
    raw = json.dumps(payload, indent=2).encode("utf-8")
    actual = hashlib.sha256(raw).hexdigest()
    if actual != EXPECTED_SHA256["mla"]:
        raise ValueError(
            "Reconstructed MLA source identity mismatch: "
            f"expected {EXPECTED_SHA256['mla']}, got {actual}"
        )
    return raw


def _passkey_rate_at_length(run: dict[str, Any], length: int) -> float:
    prefix = f"L={length}_"
    cells = [
        cell
        for key, cell in run["passkey_summary"].items()
        if key.startswith(prefix)
    ]
    if not cells:
        raise ValueError(f"Passkey payload has no cells for length {length}")
    return statistics.mean(cell["retrieval_rate"] for cell in cells)


def build_primary1_snapshot(source: Path) -> dict[str, Any]:
    payload = load_verified_json(source, EXPECTED_SHA256["primary1"])
    results = payload.get("results", {})
    if len(results) != 6:
        raise ValueError("Primary I source must contain six method/seed records")

    grouped: dict[str, dict[int, dict[str, Any]]] = {"Geo": {}, "EVQ": {}}
    for record in results.values():
        meta = record.get("meta", {})
        method = meta.get("method")
        seed = meta.get("seed")
        if method not in grouped or seed not in (7, 42, 123):
            raise ValueError("Primary I source has an unexpected method or seed")
        if meta.get("mix") != "10pct":
            raise ValueError("Primary I source is not the 10% passkey-mix protocol")
        grouped[method][seed] = record
    if any(sorted(records) != [7, 42, 123] for records in grouped.values()):
        raise ValueError("Primary I source must contain seeds 7, 42, and 123 per method")

    row_specs = (
        ("geo_raw", "Geo", "baseline"),
        ("geo_yarn_s8", "Geo", "yarn"),
        ("evq_raw", "EVQ", "baseline"),
        ("evq_yarn_s8", "EVQ", "yarn"),
    )
    rows = []
    for row_id, method, mode in row_specs:
        records = grouped[method]
        seedwise_pk = {
            str(seed): _passkey_rate_at_length(records[seed][mode], 8192)
            for seed in sorted(records)
        }
        rows.append(
            {
                "id": row_id,
                "method": method,
                "evaluation": "raw" if mode == "baseline" else "yarn_s8",
                "ppl_2048_mean": statistics.mean(
                    records[seed][mode]["ppl"]["2048"] for seed in records
                ),
                "ppl_8192_mean": statistics.mean(
                    records[seed][mode]["ppl"]["8192"] for seed in records
                ),
                "pk_8192_seedwise": seedwise_pk,
                "pk_8192_mean": statistics.mean(seedwise_pk.values()),
            }
        )

    return {
        "schema_version": 1,
        "provenance_status": "raw-json-backed",
        "artifact_role": "Full archival Primary I raw payload with recomputed Table 2 summaries.",
        "source": {
            "path": "data/results_5090b/evq_yarn_10pct_allseeds.json",
            "sha256": EXPECTED_SHA256["primary1"],
            "git_source": "backup/2026-03-06",
        },
        "protocol": {
            "model": "454M decoder-only transformer",
            "train_length": 2048,
            "passkey_mix": "10pct",
            "yarn_scale": payload["metadata"]["scale"],
            "seeds": [7, 42, 123],
            "pk_metric": "teacher-forced NLL-gap retrieval rate",
        },
        "claim_boundary": (
            "Primary I matched-scale EVQ x YaRN evidence. Autoregressive exact "
            "match is preserved as a distinct raw field and must not be relabeled "
            "as the reported teacher-forced PK metric."
        ),
        "recomputed_rows": rows,
        "raw_payload": payload,
    }


def build_primary2_tau5_snapshot(
    seed42_source: Path, extra_seeds_source: Path
) -> dict[str, Any]:
    seed42_payload = load_verified_json(
        seed42_source, EXPECTED_SHA256["primary2_seed42"]
    )
    extra_payload = load_verified_json(
        extra_seeds_source, EXPECTED_SHA256["primary2_extra_seeds"]
    )
    selected = [
        seed42_payload["experiments"]["125m_tau5.00_seed42"],
        extra_payload["experiments"]["125m_tau5.00_seed137"],
        extra_payload["experiments"]["125m_tau5.00_seed256"],
    ]
    selected.sort(key=lambda record: record["seed"])
    if [record["seed"] for record in selected] != [42, 137, 256]:
        raise ValueError("Primary II fixed-tau source must contain seeds 42, 137, 256")
    if any(record.get("tau") != 5.0 for record in selected):
        raise ValueError("Primary II recovered arm must contain only fixed tau=5")

    ppl_128 = [record["ppl"]["128"] for record in selected]
    ppl_8192 = [record["ppl"]["8192"] for record in selected]
    return {
        "schema_version": 1,
        "provenance_status": "raw-json-backed",
        "artifact_role": "Recovered three-seed fixed-EVQ arm from the L_train=128 diagnostic.",
        "sources": {
            "seed42_sweep": {
                "path": "data/evq_128tok_results/extended_sweep/results_final.json",
                "sha256": EXPECTED_SHA256["primary2_seed42"],
            },
            "extra_seed_sweep": {
                "path": "data/evq_128tok_results/phase7/multiseed/results_final.json",
                "sha256": EXPECTED_SHA256["primary2_extra_seeds"],
            },
            "git_source": "backup/2026-03-06",
        },
        "protocol": {
            "model": "125M decoder-only transformer",
            "train_length": 128,
            "training_tokens": 15000000,
            "dataset": "FineWeb-Edu",
            "base": 500000.0,
            "tau": 5.0,
            "seeds": [42, 137, 256],
            "eval_lengths": [128, 256, 512, 1024, 2048, 4096, 8192],
        },
        "summary": {
            "ppl_128_mean": statistics.mean(ppl_128),
            "ppl_128_sample_std": statistics.stdev(ppl_128),
            "ppl_8192_mean": statistics.mean(ppl_8192),
            "ppl_8192_sample_std": statistics.stdev(ppl_8192),
        },
        "claim_boundary": (
            "This recovers the fixed EVQ tau=5 arm only. It does not recover "
            "matched Geo or DAPE seeds 137/256, so it cannot upgrade the full "
            "Primary II Geo/DAPE/EVQ comparison to a three-seed claim."
        ),
        "runs": selected,
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


def build_phase11b_snapshot(
    scaling_source: Path, dape_source: Path
) -> dict[str, Any]:
    scaling = load_verified_json(
        scaling_source, EXPECTED_SHA256["phase11b_scaling"]
    )
    dape = load_verified_json(dape_source, EXPECTED_SHA256["phase11b_dape"])
    if len(scaling) != 9 or len(dape) != 6:
        raise ValueError("Expected nine scaling and six DAPE Phase11B run records")
    seeds = sorted({record["seed"] for record in [*scaling.values(), *dape.values()]})
    if seeds != [42, 137, 256]:
        raise ValueError("Phase11B sources do not contain seeds [42, 137, 256]")
    if any(record.get("use_dape") for record in scaling.values()):
        raise ValueError("Phase11B scaling source unexpectedly contains DAPE runs")
    if any(not record.get("use_dape") for record in dape.values()):
        raise ValueError("Phase11B DAPE source unexpectedly contains plain runs")
    return {
        "schema_version": 1,
        "provenance_status": "raw-json-backed",
        "artifact_role": (
            "Portable copy of the ignored 125M Phase11B three-seed scaling "
            "and DAPE-compatibility payloads."
        ),
        "sources": {
            "scaling": {
                "path_hint": "results/core_text/phase11b/results_125m_scaling.json",
                "sha256": EXPECTED_SHA256["phase11b_scaling"],
            },
            "dape": {
                "path_hint": "results/core_text/phase11b/results_125m_dape.json",
                "sha256": EXPECTED_SHA256["phase11b_dape"],
            },
        },
        "protocol": {
            "model": "125M decoder-only transformer",
            "train_length": 256,
            "training_tokens": 100000000,
            "dataset": "FineWeb-Edu",
            "base": 500000.0,
            "head_dim": 64,
            "seeds": seeds,
            "eval_lengths": [256, 512, 1024, 2048, 4096, 8192],
            "dape": "Kerple bias plus learned attention-score MLP refinement",
        },
        "claim_boundary": (
            "This is a separate L_train=256, 100M-token supporting protocol, "
            "not the L_train=128, 15M-token Primary II Geo/DAPE/EVQ diagnostic. "
            "It supports cross-scale plain-EVQ behavior and shows that EVQ+DAPE "
            "does not improve on Geo+DAPE here; it must not be used to claim "
            "EVQ-DAPE complementarity or to upgrade Primary II to three seeds."
        ),
        "scaling_runs": scaling,
        "dape_runs": dape,
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
        "source_report": "docs/exp/2026-03/2026-03-12_phase21b_454m_full_eval_report.md",
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


def write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mla-source", type=Path, default=DEFAULT_MLA_SOURCE)
    parser.add_argument(
        "--mla-snapshot-source", type=Path, default=DEFAULT_MLA_SNAPSHOT_SOURCE
    )
    parser.add_argument("--primary1-source", type=Path, default=DEFAULT_PRIMARY1_SOURCE)
    parser.add_argument(
        "--primary2-seed42-source",
        type=Path,
        default=DEFAULT_PRIMARY2_SEED42_SOURCE,
    )
    parser.add_argument(
        "--primary2-extra-seeds-source",
        type=Path,
        default=DEFAULT_PRIMARY2_EXTRA_SEEDS_SOURCE,
    )
    parser.add_argument(
        "--phase11-snapshot-source",
        type=Path,
        default=DEFAULT_PHASE11_SNAPSHOT_SOURCE,
    )
    parser.add_argument(
        "--phase11-raw-source",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--phase11-yarn-source",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--phase11b-snapshot-source",
        type=Path,
        default=DEFAULT_PHASE11B_SNAPSHOT_SOURCE,
    )
    parser.add_argument("--phase11b-scaling-source", type=Path, default=None)
    parser.add_argument("--phase11b-dape-source", type=Path, default=None)
    parser.add_argument("--quality-source", type=Path, default=DEFAULT_QUALITY_SOURCE)
    parser.add_argument(
        "--base-source-dir", type=Path, default=DEFAULT_BASE_SOURCE_DIR
    )
    parser.add_argument(
        "--only",
        action="append",
        choices=(
            "mla",
            "mla_raw",
            "primary1",
            "primary2_tau5",
            "phase11",
            "phase11b",
            "quality",
            "base",
        ),
        help="Build only the selected component; repeat for multiple components.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    if (args.phase11_raw_source is None) != (args.phase11_yarn_source is None):
        parser.error(
            "--phase11-raw-source and --phase11-yarn-source must be provided together"
        )
    if (args.phase11b_scaling_source is None) != (
        args.phase11b_dape_source is None
    ):
        parser.error(
            "--phase11b-scaling-source and --phase11b-dape-source "
            "must be provided together"
        )
    return args


def main() -> None:
    args = parse_args()
    components = args.only or ["mla", "phase11", "phase11b", "quality", "base"]
    outputs: list[Path] = []
    if "mla" in components:
        outputs.append(args.output_dir / "table18_mla_3seed_aggregate.json")
        write_json(outputs[-1], build_mla_snapshot(args.mla_source))
    if "mla_raw" in components:
        outputs.append(args.output_dir / "eval_3seeds_full_results.json")
        write_bytes(outputs[-1], reconstruct_mla_source_bytes(args.mla_snapshot_source))
    if "primary1" in components:
        outputs.append(args.output_dir / "primary1_evq_yarn_10pct_raw.json")
        write_json(outputs[-1], build_primary1_snapshot(args.primary1_source))
    if "primary2_tau5" in components:
        outputs.append(args.output_dir / "primary2_l128_fixed_tau5_3seed.json")
        write_json(
            outputs[-1],
            build_primary2_tau5_snapshot(
                args.primary2_seed42_source, args.primary2_extra_seeds_source
            ),
        )
    if "phase11" in components:
        outputs.append(args.output_dir / "phase11_l256_3seed_recovered.json")
        if args.phase11_raw_source is None:
            write_bytes(
                outputs[-1],
                validated_phase11_snapshot_bytes(args.phase11_snapshot_source),
            )
        else:
            write_json(
                outputs[-1],
                build_phase11_snapshot(
                    args.phase11_raw_source, args.phase11_yarn_source
                ),
            )
    if "phase11b" in components:
        outputs.append(args.output_dir / "phase11b_125m_l256_3seed.json")
        if args.phase11b_scaling_source is None:
            write_bytes(
                outputs[-1],
                validated_phase11b_snapshot_bytes(args.phase11b_snapshot_source),
            )
        else:
            write_json(
                outputs[-1],
                build_phase11b_snapshot(
                    args.phase11b_scaling_source, args.phase11b_dape_source
                ),
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
