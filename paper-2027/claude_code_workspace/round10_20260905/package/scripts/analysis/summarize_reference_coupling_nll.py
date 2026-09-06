#!/usr/bin/env python3
"""Hash-bound paired NLL reporting; no profile selection or model execution."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

ARMS = ("Native", "dimensionless_x", "normalized_raw_index", "official_equation_yarn")
SEED, REPLICATES, DOCUMENTS = 202609024, 10_000, 32


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_hash(value) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("missing or invalid identity SHA-256")
    return value


def finite_exp(value: float):
    try:
        result = math.exp(value)
        return result if math.isfinite(result) else None
    except OverflowError:
        return None


def summarize(result: dict, manifest: dict, rows: list[dict], raw_hashes=None) -> dict:
    if result.get("status") != "REFERENCE_COUPLING_NLL_COMPLETE" or manifest.get("status") != "REFERENCE_COUPLING_NLL_FROZEN":
        raise ValueError("only terminal NLL results with a frozen run manifest are admitted")
    lengths, scope = manifest["lengths"], manifest["profiles"]
    reference, target = scope["L_ref"], scope["target_length"]
    if (not lengths or any(type(value) is not int for value in lengths)
            or len(set(lengths)) != len(lengths) or reference not in lengths
            or min(lengths) != reference or max(lengths) > target
            or manifest.get("natural_documents") != DOCUMENTS or manifest.get("target_tokens") != 256
            or manifest.get("arm_order") != list(ARMS)):
        raise ValueError("invalid frozen length, document, target-token, or arm contract")
    profiles = {profile["name"]: profile for profile in scope["profiles"]}
    if set(profiles) != set(ARMS) or len(scope["profiles"]) != 4:
        raise ValueError("exactly four fixed profiles are required")
    identities = {}
    for arm, profile in profiles.items():
        gain = profile["attention_scaling"]
        if type(gain) not in (int, float) or not math.isfinite(gain) or gain <= 0 or (arm == "Native" and gain != 1):
            raise ValueError("invalid frozen attention gain")
        identities[arm] = {"tensor_sha256": checked_hash(profile["tensor_sha256"]), "attention_scaling": gain,
            "file_sha256": None if arm == "Native" else checked_hash(profile["file_sha256"])}
    paired, document_identity, cells = {}, {}, {}
    metadata = ("family", "variant", "split", "source_row", "source_text_sha256",
                "prompt_ids_sha256", "target_ids_sha256", "target_start", "target_tokens")
    for row in rows:
        arm, sid, length = row["arm"], row["sample_id"], row["length"]
        key = arm, sid, length
        if (arm not in ARMS or type(length) is not int or length not in lengths or key in cells
                or row.get("family") != "natural" or row.get("variant") != "natural" or row.get("split") != "holdout"
                or row.get("target_tokens") != 256 or row.get("target_start") != length - 256
                or row.get("table_sha256_float32") != identities[arm]["tensor_sha256"]
                or row.get("attention_scaling") != identities[arm]["attention_scaling"]):
            raise ValueError("duplicated/unregistered row or frozen profile metadata drift")
        loss = row["nll"]
        if type(loss) not in (int, float) or not math.isfinite(loss) or loss < 0:
            raise ValueError("NLL must be finite and nonnegative")
        identity = checked_hash(row["source_text_sha256"]), checked_hash(row["target_ids_sha256"])
        checked_hash(row["prompt_ids_sha256"])
        if sid in document_identity and document_identity[sid] != identity:
            raise ValueError("document source/target differs across lengths or arms")
        document_identity[sid] = identity
        row_metadata = tuple(row.get(field) for field in metadata)
        if (sid, length) in paired and paired[sid, length] != row_metadata:
            raise ValueError("paired document metadata differs across arms")
        paired[sid, length], cells[key] = row_metadata, float(loss)
    ids = [f"reference-coupling-natural-{index:03d}" for index in range(DOCUMENTS)]
    expected = {(arm, sid, length) for arm in ARMS for sid in ids for length in lengths}
    if set(cells) != expected or result.get("rows") != len(expected) or len({item[0] for item in document_identity.values()}) != DOCUMENTS:
        raise ValueError("requires exactly 32 unique, fully paired documents for every arm and length")
    lengths = sorted(lengths)
    values = np.array([[[cells[arm, sid, length] for arm in ARMS] for length in lengths] for sid in ids])
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, DOCUMENTS, size=(REPLICATES, DOCUMENTS))
    draws, means = values[indices].mean(axis=1), values.mean(axis=0)
    curves, contrasts = {}, {}
    pairs = {f"{arm}_minus_Native": (column, 0) for column, arm in enumerate(ARMS)}
    pairs.update(physical_minus_index=(1, 2), physical_minus_yarn=(1, 3))
    for li, length in enumerate(lengths):
        curves[str(length)], contrasts[str(length)] = {}, {}
        for ai, arm in enumerate(ARMS):
            mean = float(means[li, ai])
            stored = result.get("curves", {}).get(arm, {}).get(str(length))
            if stored is not None and (stored.get("documents") != DOCUMENTS or not math.isclose(stored["mean_tail_nll"], mean, rel_tol=0, abs_tol=1e-12)):
                raise ValueError("terminal aggregate differs from raw document NLL")
            curves[str(length)][arm] = {"mean_nll": mean, "ppl": finite_exp(mean),
                "ppl_retention_vs_native": finite_exp(float(means[li, 0] - mean))}
        for name, (left, right) in pairs.items():
            contrasts[str(length)][name] = {"paired_mean_delta_nll": float(means[li, left] - means[li, right]),
                "paired_document_ci95": [float(value) for value in np.quantile(draws[:, li, left] - draws[:, li, right], [.025, .975])]}
    ri = lengths.index(reference)
    gates = {arm: {"status": "PASS" if float(means[ri, ai] - means[ri, 0]) <= -math.log(.875) + 1e-12 else "FAIL",
                  "ppl_retention": curves[str(reference)][arm]["ppl_retention_vs_native"]} for ai, arm in enumerate(ARMS)}
    bound = {key: checked_hash(manifest[key]) for key in ("checkpoint_weight_sha256", "config_sha256",
        "data_manifest_sha256", "data_rows_sha256", "script_sha256", "model_source_sha256", "attention_source_sha256")}
    return {"status": "REFERENCE_COUPLING_NLL_SUMMARIZED", "lengths": lengths, "documents": DOCUMENTS,
        "L_config": scope["L_config"], "L_ref": reference, "target_length": target, "scale": scope["scale"],
        "curves": curves, "paired_contrasts": contrasts,
        "native_point_gate": {"length": reference, "threshold": .875, "arms": gates,
            "scope": "point-estimate natural PPL retention only; separate from RULER and uncertainty intervals"},
        "bootstrap": {"seed": SEED, "replicates": REPLICATES, "confidence": .95,
            "unit": "same 32 documents resampled jointly across all arms and lengths",
            "scope": "conditional on these documents and one checkpoint; not checkpoint or training-seed uncertainty"},
        "identity": {**bound, "profiles": identities,
            "coupling_manifest_sha256": checked_hash(scope["coupling_manifest_sha256"]),
            "baseline_manifest_sha256": checked_hash(scope["baseline_manifest_sha256"])},
        "raw_hashes": raw_hashes or {}, "summary_code_sha256": sha256(Path(__file__)),
        "profile_selection_performed": False,
        "evidence_limit": "Teacher-forced continuation NLL; no generated exact-match, K-causal, or SOTA claim; overflowing exponentials are null"}


def load_and_summarize(root: Path) -> dict:
    paths = {name: root / filename for name, filename in (("results", "results.json"),
        ("run_manifest", "run_manifest.json"), ("examples", "examples.jsonl"))}
    result = json.loads(paths["results"].read_text())
    if result.get("status") != "REFERENCE_COUPLING_NLL_COMPLETE":
        raise ValueError("NLL panel is not terminal")
    hashes = {f"{name}_sha256": sha256(path) for name, path in paths.items()}
    if any(result.get(key) != hashes[key] for key in ("examples_sha256", "run_manifest_sha256")):
        raise ValueError("terminal examples/run-manifest hash mismatch")
    rows = [json.loads(line) for line in paths["examples"].read_text().splitlines() if line.strip()]
    return summarize(result, json.loads(paths["run_manifest"].read_text()), rows, hashes)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = load_and_summarize(args.root)
    except (ValueError, KeyError, TypeError, OSError):
        report = {"status": "INVALID_OR_INCOMPLETE_NLL_PANEL", "reason": "input validation failed; no metrics promoted"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(report["status"])
    return 0 if report["status"] == "REFERENCE_COUPLING_NLL_SUMMARIZED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
