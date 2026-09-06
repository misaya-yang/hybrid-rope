#!/usr/bin/env python3
"""Frozen CPU-only Native reference-length calibration and confirmation.

The data manifest owns ``sample_ids[split][family]`` for both splits, or uses the
prepared-data V1 contract with fixed ``family-split-NNN`` IDs and split counts.
Confirmation IDs must be disjoint from calibration IDs within each family.
No model is loaded, no length/table is fit, and confirmation never selects again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import beta


GRID = (1024, 2048, 4096, 8192)
COUNTS = {"calibration": {"natural": 32, "capability": 64},
          "confirmation": {"natural": 64, "capability": 128}}
RETENTION = 0.875
COMPETENCE = 0.75
NLL_MARGIN = -math.log(RETENTION)
ALPHA = 0.0125
RATIO_TAIL_ALPHA = 0.00625
BOOTSTRAP_SEED = 202609013
BOOTSTRAP_SAMPLES = 10_000
SCHEMA = "native_reference_calibration.v1"


def digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def confirmation_lengths(selected: int) -> list[int]:
    index = GRID.index(selected)
    return sorted({1024, selected, *GRID[index + 1:index + 2]})


def contiguous_frontier(passed: list[bool]) -> tuple[int | None, bool]:
    """Return the passing prefix frontier and whether failure is followed by pass."""
    if len(passed) != len(GRID):
        raise ValueError("frontier requires every registered grid point")
    frontier = None
    failed = False
    reentry = False
    for length, is_pass in zip(GRID, passed):
        if not is_pass:
            failed = True
        elif failed:
            reentry = True
        else:
            frontier = length
    return frontier, reentry


def clopper_pearson(successes: int, count: int, alpha: float) -> tuple[float, float]:
    """Individual one-sided exact bounds, each with tail probability alpha."""
    if count <= 0 or not 0 <= successes <= count or not 0 < alpha < 0.5:
        raise ValueError("invalid binomial inputs")
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha, successes, count - successes + 1))
    upper = 1.0 if successes == count else float(beta.ppf(1 - alpha, successes + 1, count - successes))
    return lower, upper


def classify_lower_bound(lower: float, upper: float | None, floor: float) -> str:
    if lower >= floor:
        return "PASS"
    if upper is not None and upper < floor:
        return "FAIL"
    return "UNRESOLVED"


def _interval(values: np.ndarray) -> list[float] | None:
    if not np.isfinite(values).all():
        return None
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def paired_bootstrap(values: np.ndarray, strata: list[str] | None = None) -> np.ndarray:
    """Bootstrap rows jointly over columns, retaining each stratum's weight."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not values.shape[0]:
        raise ValueError("bootstrap needs a nonempty row-by-endpoint matrix")
    labels = ["all"] * len(values) if strata is None else strata
    if len(labels) != len(values):
        raise ValueError("bootstrap stratum count does not match rows")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.zeros((BOOTSTRAP_SAMPLES, values.shape[1]))
    for label in sorted(set(labels)):
        group = values[[i for i, item in enumerate(labels) if item == label]]
        indices = rng.integers(0, len(group), size=(BOOTSTRAP_SAMPLES, len(group)))
        draws += group[indices].sum(axis=1) / len(values)
    return draws


def _manifest_sample_ids(manifest: dict) -> dict:
    if "sample_ids" in manifest:
        return manifest["sample_ids"]
    if (manifest.get("status") != "NATIVE_REFERENCE_CALIBRATION_DATA_READY_V1"
            or manifest.get("schema_version") != 1 or manifest.get("grid") != list(GRID)):
        raise ValueError("manifest requires sample_ids or the prepared-data V1 identity contract")
    sample_ids = {}
    for split, counts in COUNTS.items():
        entry = manifest.get("files", {}).get(split, {})
        expected_rows = counts["natural"] * len(GRID) + counts["capability"] * (len(GRID) + 1)
        if (entry.get("natural_documents") != counts["natural"]
                or entry.get("capability_blueprints") != counts["capability"]
                or entry.get("rows") != expected_rows):
            raise ValueError(f"prepared-data V1 manifest count mismatch: {split}")
        sample_ids[split] = {family: [f"{family}-{split}-{index:03d}" for index in range(count)]
                             for family, count in counts.items()}
    return sample_ids


def validate_rows(rows: list[dict], manifest: dict, phase: str,
                  selection: dict | None) -> tuple[list[int], np.ndarray, np.ndarray, list[str], dict]:
    if phase not in COUNTS:
        raise ValueError("phase must be calibration or confirmation")
    sample_ids = _manifest_sample_ids(manifest)
    if not isinstance(sample_ids, dict):
        raise ValueError("manifest requires sample_ids[split][family]")
    for split, families in COUNTS.items():
        for family, count in families.items():
            ids = sample_ids.get(split, {}).get(family)
            if (not isinstance(ids, list) or len(ids) != count
                    or any(not isinstance(item, str) or not item for item in ids)
                    or len(set(ids)) != count):
                raise ValueError(f"manifest {split}/{family} requires {count} unique string sample IDs")
    for family in COUNTS[phase]:
        if set(sample_ids["calibration"][family]) & set(sample_ids["confirmation"][family]):
            raise ValueError(f"calibration and confirmation IDs overlap: {family}")
    lengths = list(GRID)
    if phase == "confirmation":
        if (not isinstance(selection, dict) or selection.get("schema") != SCHEMA
                or selection.get("phase") != "calibration"
                or selection.get("status") != "PROVISIONAL"
                or selection.get("selected_length") not in GRID):
            raise ValueError("confirmation requires a provisional calibration selection")
        if selection.get("manifest_digest") != digest(manifest):
            raise ValueError("selection is bound to a different data manifest")
        lengths = confirmation_lengths(selection["selected_length"])
        if selection.get("confirmation_lengths") != lengths:
            raise ValueError("selection confirmation lengths were modified")
    elif selection is not None:
        raise ValueError("calibration must not receive a selection")

    keyed: dict[tuple[str, str, str, int], dict] = {}
    depths: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("split") != phase:
            raise ValueError("examples must contain only the requested split")
        family, variant = row.get("family"), row.get("variant")
        if family not in COUNTS[phase]:
            raise ValueError(f"unexpected family: {family}")
        sample_id, length = row.get("sample_id"), row.get("length")
        if not isinstance(sample_id, str) or sample_id not in sample_ids[phase][family]:
            raise ValueError(f"sample ID not owned by manifest: {family}/{sample_id}")
        allowed = ((family == "natural" and variant == "natural" and length in lengths)
                   or (family == "capability" and variant == "distributed" and length in lengths)
                   or (family == "capability" and variant == "compact" and length == 0))
        if type(length) is not int or not allowed:
            raise ValueError(f"unregistered family/variant/length: {family}/{variant}/{length}")
        key = family, variant, sample_id, length
        if key in keyed:
            raise ValueError(f"duplicate example: {key}")
        keyed[key] = row
        if family == "natural":
            loss = row.get("nll")
            if isinstance(loss, bool) or not isinstance(loss, (int, float)) or not math.isfinite(loss) or loss < 0:
                raise ValueError("natural nll must be finite and nonnegative")
        else:
            if type(row.get("exact_match")) is not bool or type(row.get("terminated")) is not bool:
                raise ValueError("capability exact_match and terminated must be booleans")
            depth = row.get("depth_stratum")
            if isinstance(depth, bool) or not isinstance(depth, (str, int)) or str(depth) == "":
                raise ValueError("capability depth_stratum must be a nonempty string or integer")
            if sample_id in depths and depths[sample_id] != str(depth):
                raise ValueError(f"depth stratum changed across paired cells: {sample_id}")
            depths[sample_id] = str(depth)

    expected = set()
    for family in COUNTS[phase]:
        for sample_id in sample_ids[phase][family]:
            variant = "natural" if family == "natural" else "distributed"
            expected.update((family, variant, sample_id, length) for length in lengths)
            if family == "capability":
                expected.add((family, "compact", sample_id, 0))
    if set(keyed) != expected:
        raise ValueError(f"unpaired or missing required cells: {len(expected - set(keyed))} missing")
    natural_ids = sorted(sample_ids[phase]["natural"])
    cap_ids = sorted(sample_ids[phase]["capability"])
    natural = np.array([[keyed["natural", "natural", sid, length]["nll"]
                         for length in lengths] for sid in natural_ids], dtype=float)
    cap_columns = [("distributed", length) for length in lengths] + [("compact", 0)]
    capability = np.array([[float(keyed["capability", variant, sid, length]["exact_match"]
                                 and keyed["capability", variant, sid, length]["terminated"])
                            for variant, length in cap_columns] for sid in cap_ids])
    receipt = {"valid": True, "examples": len(rows), "natural_documents": len(natural_ids),
               "capability_blueprints": len(cap_ids), "lengths": lengths,
               "paired_sample_ids_verified": True, "split_ids_disjoint": True,
               "capability_success": "exact_match AND terminated",
               "exact_without_termination": sum(row.get("exact_match") is True
                   and row.get("terminated") is False for row in rows)}
    return lengths, natural, capability, [depths[sid] for sid in cap_ids], receipt


def _natural_metrics(lengths: list[int], values: np.ndarray) -> dict:
    delta = values - values[:, [0]]
    draws = paired_bootstrap(delta)
    metrics = {}
    for column, length in enumerate(lengths):
        point = float(delta[:, column].mean())
        lower, upper = [float(x) for x in np.quantile(draws[:, column], [ALPHA, 1 - ALPHA])]
        identity = length == 1024
        status = "PASS" if upper <= NLL_MARGIN else ("FAIL" if lower > NLL_MARGIN else "UNRESOLVED")
        metrics[str(length)] = {
            "mean_nll": float(values[:, column].mean()), "paired_delta_nll": point,
            "ppl_retention": math.exp(-point), "point_pass": point <= NLL_MARGIN,
            "paired_bootstrap_ci95_delta": _interval(draws[:, column]),
            "lower_one_sided_9875_delta": lower, "upper_one_sided_9875_delta": upper,
            "bound_status": status, "relative_identity": identity,
            "interval_interpretation": "algebraic identity, not population certainty" if identity else "paired document bootstrap",
        }
    return metrics


def _capability_metrics(lengths: list[int], values: np.ndarray, strata: list[str]) -> dict:
    count = len(values)
    draws = paired_bootstrap(values, strata)
    means = values.mean(axis=0)
    base_count = int(values[:, 0].sum())
    ratio_base_lower, ratio_base_upper = clopper_pearson(base_count, count, RATIO_TAIL_ALPHA)
    metrics = {}
    for column, length in enumerate(lengths):
        successes = int(values[:, column].sum())
        if length == 1024:
            point, lower, upper = (1.0, 1.0, 1.0) if means[0] > 0 else (None, 0.0, None)
            ratio_draws = np.ones(BOOTSTRAP_SAMPLES) if means[0] > 0 else np.full(BOOTSTRAP_SAMPLES, np.nan)
        else:
            point = float(means[column] / means[0]) if means[0] > 0 else None
            numerator_lower, numerator_upper = clopper_pearson(successes, count, RATIO_TAIL_ALPHA)
            lower = numerator_lower / ratio_base_upper
            upper = numerator_upper / ratio_base_lower if ratio_base_lower > 0 else None
            ratio_draws = np.divide(draws[:, column], draws[:, 0],
                                    out=np.full(BOOTSTRAP_SAMPLES, np.nan), where=draws[:, 0] > 0)
        metrics[str(length)] = {
            "successes": successes, "count": count, "success_rate": float(means[column]),
            "retention": point, "point_pass": point is not None and point >= RETENTION,
            "paired_stratified_bootstrap_ci95_retention": _interval(ratio_draws),
            "paired_stratified_bootstrap_ci95_difference": _interval(draws[:, column] - draws[:, 0]),
            "bootstrap_undefined_ratio_draws": int((~np.isfinite(ratio_draws)).sum()),
            "lower_conservative_cp_retention": lower, "upper_conservative_cp_retention": upper,
            "bound_status": classify_lower_bound(lower, upper, RETENTION),
            "relative_identity": length == 1024,
            "interval_interpretation": "algebraic identity, not population certainty" if length == 1024 else "CP numerator/denominator, each tail alpha=.00625",
        }
    instruments = {}
    for name, column in (("baseline", 0), ("compact", len(lengths))):
        successes = int(values[:, column].sum())
        lower, upper = clopper_pearson(successes, count, ALPHA)
        instruments[name] = {"successes": successes, "count": count,
                             "success_rate": float(means[column]),
                             "point_pass": bool(means[column] >= COMPETENCE),
                             "lower_one_sided_9875_cp": lower, "upper_one_sided_9875_cp": upper,
                             "bound_status": classify_lower_bound(lower, upper, COMPETENCE),
                             "stratified_bootstrap_ci95": _interval(draws[:, column])}
    return {"lengths": metrics, "instruments": instruments,
            "depth_stratum_counts": {label: strata.count(label) for label in sorted(set(strata))}}


def summarize(rows: list[dict], manifest: dict, *, phase: str,
              selection: dict | None = None) -> dict:
    """Return a JSON-safe decision; malformed input is an explicit abstention."""
    result: dict[str, Any] = {
        "schema": SCHEMA, "phase": phase, "status": "ABSTAIN", "selected_length": None,
        "reasons": [], "rule": {"grid": list(GRID), "retention": RETENTION,
            "competence": COMPETENCE, "nll_margin": NLL_MARGIN, "one_sided_alpha": ALPHA,
            "ratio_cp_each_tail_alpha": RATIO_TAIL_ALPHA, "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_samples": BOOTSTRAP_SAMPLES, "confirmation_never_reselects": True},
        "evidence_limit": "Native operating point; not training exposure, causal K, or population ceiling",
    }
    try:
        result["manifest_digest"] = digest(manifest)
        result["examples_digest"] = digest(rows)
        lengths, natural, capability, strata, receipt = validate_rows(rows, manifest, phase, selection)
    except (ValueError, TypeError, KeyError) as error:
        result["validation"] = {"valid": False}
        result["reasons"] = [f"INVALID_INPUT: {error}"]
        return result
    result["validation"] = receipt
    result["natural"] = _natural_metrics(lengths, natural)
    result["capability"] = _capability_metrics(lengths, capability, strata)
    cap = result["capability"]
    if phase == "calibration":
        frontiers, reentries = {}, {}
        for family, metrics in (("natural", result["natural"]), ("capability", cap["lengths"])):
            frontier, reentry = contiguous_frontier([metrics[str(length)]["point_pass"] for length in GRID])
            frontiers[family], reentries[family] = frontier, reentry
        result["frontiers"], result["fail_to_pass_reentry"] = frontiers, reentries
        for name, metric in cap["instruments"].items():
            if not metric["point_pass"]:
                result["reasons"].append(f"{name.upper()}_COMPETENCE_FAILED")
        if any(reentries.values()):
            result["reasons"].append("NONMONOTONE_FAIL_TO_PASS_REENTRY")
        if frontiers["natural"] != frontiers["capability"]:
            result["reasons"].append("INCOMPATIBLE_FAMILY_FRONTIERS")
        if frontiers["natural"] is None or frontiers["capability"] is None:
            result["reasons"].append("NO_PASSING_PREFIX")
        if not result["reasons"]:
            result["status"] = "PROVISIONAL"
            result["selected_length"] = frontiers["natural"]
            result["confirmation_lengths"] = confirmation_lengths(result["selected_length"])
        return result

    selected = selection["selected_length"]
    result["frozen_selection_digest"] = digest(selection)
    result["provisional_length"] = selected
    gates = {"natural": result["natural"][str(selected)]["bound_status"],
             "capability_retention": cap["lengths"][str(selected)]["bound_status"],
             **{name: metric["bound_status"] for name, metric in cap["instruments"].items()}}
    result["confirmation_gates"] = gates
    result["boundary_confirmed"] = False
    if all(status == "PASS" for status in gates.values()):
        result["status"] = "ACCEPTED_OPERATING_POINT"
        result["selected_length"] = selected
    else:
        result["reasons"] = [f"CONFIRMATION_{name.upper()}_{status}"
                             for name, status in gates.items() if status != "PASS"]
    next_lengths = GRID[GRID.index(selected) + 1:GRID.index(selected) + 2]
    if next_lengths:
        next_length = next_lengths[0]
        statuses = {"natural": result["natural"][str(next_length)]["bound_status"],
                    "capability_retention": cap["lengths"][str(next_length)]["bound_status"]}
        secondary_status = ("PASS" if all(value == "PASS" for value in statuses.values())
                            else "FAIL" if "FAIL" in statuses.values() else "UNRESOLVED")
        result["next_length_secondary"] = {"length": next_length, "status": secondary_status,
            "gates": statuses, "promotion_allowed": False,
            "multiplicity_note": "secondary only; not part of four primary acceptance gates"}
        result["boundary_confirmed"] = result["status"] == "ACCEPTED_OPERATING_POINT" and secondary_status == "FAIL"
    else:
        result["next_length_secondary"] = {"status": "NOT_AVAILABLE_AT_GRID_MAXIMUM", "promotion_allowed": False}
    return result


def verify_frozen_inputs(rows: list[dict], manifest: dict, manifest_path: Path,
                         phase: str) -> dict:
    """Verify input-file ownership without reading other-split model outcomes."""
    if manifest.get("status") != "NATIVE_REFERENCE_CALIBRATION_DATA_READY_V1":
        return {"status": "EXPLICIT_SAMPLE_ID_MANIFEST"}
    expected = {}
    receipts = {}
    identities = {split: {"natural": set(), "capability": set()} for split in COUNTS}
    for split in COUNTS:
        entry = manifest["files"][split]
        root = manifest_path.resolve().parent
        path = (root / entry["path"]).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("prepared input path leaves its manifest directory") from error
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if file_hash != entry["sha256"]:
            raise ValueError(f"frozen input file hash mismatch: {split}")
        seen = set()
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                source = json.loads(line)
                key = source["family"], source["variant"], source["sample_id"], source["length"]
                if source.get("split") != split or key in seen:
                    raise ValueError(f"duplicate or mixed-split frozen input: {split}")
                seen.add(key)
                identity_field = "source_text_sha256" if source["family"] == "natural" else "blueprint_sha256"
                if identity_field in source:
                    identities[split][source["family"]].add(source[identity_field])
                if split == phase:
                    expected[key] = {name: value for name, value in source.items() if name != "input_ids"}
        if len(seen) != entry["rows"]:
            raise ValueError(f"frozen input count mismatch: {split}")
        receipts[split] = {"sha256": file_hash, "rows": len(seen)}
    for family in COUNTS[phase]:
        if identities["calibration"][family] & identities["confirmation"][family]:
            raise ValueError(f"frozen source identities overlap across splits: {family}")
    for row in rows:
        key = row["family"], row["variant"], row["sample_id"], row["length"]
        source = expected.get(key)
        if source is None or any(row.get(name) != value for name, value in source.items()):
            raise ValueError(f"result metadata differs from frozen input: {key}")
    return {"status": "HASH_AND_METADATA_VERIFIED", "files": receipts,
            "other_split_model_outcomes_read": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=tuple(COUNTS), required=True)
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.examples.read_text().splitlines() if line.strip()]
    manifest = json.loads(args.data_manifest.read_text())
    selection = json.loads(args.selection.read_text()) if args.selection else None
    result = summarize(rows, manifest, phase=args.phase, selection=selection)
    if result["validation"]["valid"]:
        try:
            result["validation"]["frozen_inputs"] = verify_frozen_inputs(rows, manifest, args.data_manifest, args.phase)
        except (ValueError, KeyError, OSError) as error:
            result.update(status="ABSTAIN", selected_length=None, validation={"valid": False},
                          reasons=[f"INVALID_INPUT: {error}"])
    result["input_file_sha256"] = {"examples": hashlib.sha256(args.examples.read_bytes()).hexdigest(),
                                    "data_manifest": hashlib.sha256(args.data_manifest.read_bytes()).hexdigest()}
    result["data_manifest_sha256"] = result["input_file_sha256"]["data_manifest"]
    if any(row.get("data_manifest_sha256", result["data_manifest_sha256"])
           != result["data_manifest_sha256"] for row in rows):
        result.update(status="ABSTAIN", selected_length=None, validation={"valid": False},
                      reasons=["INVALID_INPUT: example data manifest file hash mismatch"])
    if args.selection:
        result["input_file_sha256"]["selection"] = hashlib.sha256(args.selection.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0 if result["validation"]["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
