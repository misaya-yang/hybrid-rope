#!/usr/bin/env python3
"""CPU-only audit of the fixed 24 K32 historical RULER replay canaries.

Reads only the supplied raw receipts; never imports a model or repairs scores.
Input identity means the receipted per-cell data-file hash, not token parity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

LENGTHS = (32768, 65536)
TASKS = ("niah_single_1", "niah_multikey_2", "niah_multikey_3", "vt")
ARMS = ("native", "physical_x", "normalized_index")
HISTORY = {
    "native": "prior_k32_native_{length}",
    "physical_x": "prior_k32_physical/physical_x_s2_{size}k",
    "normalized_index": "prior_k32_index/index_s2_{size}k",
}
COMPLETE = "TARGET_FREE_RULER_SMOKE_COMPLETE"
FROZEN = "TARGET_FREE_RULER_SMOKE_FROZEN"


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


def digest(value) -> str:
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            "invalid_sha256")
    return value


def score(row: dict) -> float:
    require(row["official_metric"] == "string_match_all", "metric_drift")
    refs, prediction = row["references"], row["prediction"]
    require(isinstance(prediction, str) and isinstance(refs, list) and bool(refs)
            and all(isinstance(ref, str) and bool(ref) for ref in refs), "invalid_prediction_or_references")
    value = row["official_task_score"]
    require(type(value) in (int, float) and math.isfinite(value), "invalid_score")
    computed = sum(ref.lower() in prediction.lower() for ref in refs) / len(refs)
    require(math.isclose(value, computed, rel_tol=0, abs_tol=1e-12), "score_not_reproducible")
    return float(value)


def load_run(path: Path, arm: str, lengths: tuple[int, ...], limit: int) -> dict:
    raw = {name: (path / name).read_bytes()
           for name in ("run_manifest.json", "results.json", "examples.jsonl")}
    protocol = json.loads(raw["run_manifest.json"])
    terminal = json.loads(raw["results.json"])
    rows = [json.loads(line) for line in raw["examples.jsonl"].splitlines() if line.strip()]
    require(terminal["status"] == COMPLETE and protocol["status"] == FROZEN, "run_not_complete")
    require(terminal["protocol"] == protocol, "terminal_protocol_mismatch")
    result, data, method = terminal["results"], terminal["data"], terminal["method"]
    require(digest(result["examples_sha256"]) == sha256(raw["examples.jsonl"]), "examples_hash_mismatch")
    require(protocol["lengths"] == list(lengths) and protocol["tasks"] == list(TASKS)
            and protocol["limit_per_cell"] == limit, "selection_contract_drift")
    require(result["examples"] == len(rows) == len(lengths) * len(TASKS) * limit, "row_count_drift")
    expected_method = "native" if arm == "native" else "external_table_static"
    require(protocol["method"] == method["method"] == expected_method, "method_drift")
    require(protocol["native_context_length"] == method["native_context_length"] == 32768
            and protocol["model_type"] == method["model_type"] == "qwen2", "checkpoint_geometry_drift")
    active = digest(method["active_sha256_float32"])
    native = active if arm == "native" else digest(method["native_sha256_float32"])
    gain = method["attention_scaling"] if arm == "native" else method["long_attention_scaling"]
    require(type(gain) in (int, float) and math.isfinite(gain) and gain > 0, "invalid_gain")
    if arm == "native":
        require(gain == 1 and protocol.get("table_sha256_float32") is None
                and protocol.get("table_file_sha256") is None, "native_intervention")
    else:
        require(protocol["table_factor"] == method["table_factor"] == 2
                and protocol["table_support"] == method["table_support"] == "native_div_factor"
                and protocol["long_attention_scaling"] == gain
                and digest(protocol["table_sha256_float32"]) == active
                and digest(protocol["expected_active_sha256"]) == active
                and method["initial_branch"] == "external_long", "static_table_identity_drift")
    if protocol.get("expected_native_sha256") is not None:
        require(digest(protocol["expected_native_sha256"]) == native, "native_hash_mismatch")
    data_hash = digest(data["manifest_sha256"])
    require(digest(protocol["data_manifest_sha256"]) == data_hash, "data_manifest_mismatch")
    if protocol.get("expected_data_manifest_sha256") is not None:
        require(protocol["expected_data_manifest_sha256"] == data_hash, "expected_data_manifest_mismatch")
    capacity = protocol.get("profile_target_length", 32768 * protocol["table_factor"])
    require(type(capacity) in (int, float) and math.isfinite(capacity)
            and capacity >= max(lengths), "insufficient_profile_capacity")
    indexed = {}
    for row in rows:
        key = (row["task"], row["nominal_length"], row["local_index"])
        require(type(key[1]) is int and type(key[2]) is int and key not in indexed, "duplicate_or_invalid_row")
        score(row)
        indexed[key] = row
    expected = {(task, length, index) for task in TASKS for length in lengths for index in range(limit)}
    require(set(indexed) == expected, "row_selection_drift")
    cells = {}
    cell_means = []
    for task in TASKS:
        for length in lengths:
            cell = data["cells"][task][str(length)]
            require(cell["selected_rows"] == limit and cell["rows"] == 20, "input_cell_count_drift")
            cells[task, length] = digest(cell["sha256"])
            mean = sum(score(indexed[task, length, index]) for index in range(limit)) / limit
            summary = result["cells"][task][str(length)]
            require(summary["rows"] == limit and math.isclose(summary["official_task_score"], mean,
                    rel_tol=0, abs_tol=1e-12), "terminal_cell_score_drift")
            cell_means.append(mean)
    require(math.isclose(result["macro_official_task_score"], sum(cell_means) / len(cell_means),
                        rel_tol=0, abs_tol=1e-12), "terminal_macro_drift")
    return {
        "rows": indexed, "cells": cells,
        "identity": {"checkpoint_sha256": digest(protocol["checkpoint_sha256"]),
                     "native_sha256_float32": native, "active_sha256_float32": active,
                     "attention_scaling": gain,
                     "table_file_sha256": None if arm == "native" else digest(protocol["table_file_sha256"]),
                     "data_manifest_sha256": data_hash, "tokenizer_sha256": digest(data["tokenizer_sha256"]),
                     "ruler_commit": data["ruler_commit"], "seed": data["seed"],
                     "torch": terminal["runtime"]["torch"], "cuda": terminal["runtime"]["cuda"]},
        "raw_hashes": {name: sha256(value) for name, value in raw.items()},
        "runner_sha256": digest(protocol["script_sha256"]), "profile_capacity": int(capacity),
    }


def audit(root: Path) -> dict:
    runs, receipts, errors = {}, {}, []
    for arm in ARMS:
        entries = [(f"replay_{arm}", root / "k32_historical_replay" / arm, LENGTHS, 1)]
        entries += [(f"historical_{arm}_{length}", root / HISTORY[arm].format(length=length, size=length // 1024),
                     (length,), 20) for length in LENGTHS]
        for label, path, lengths, limit in entries:
            try:
                run = load_run(path, arm, lengths, limit)
                runs[label] = run
                receipts[label] = {key: run[key] for key in ("raw_hashes", "runner_sha256", "profile_capacity")}
            except (OSError, KeyError, TypeError, ValueError, OverflowError):
                # Never serialize exception text: it can include private paths or raw input.
                errors.append({"run": label, "code": "invalid_or_incomplete_run"})
    cells = []
    for arm in ARMS:
        for length in LENGTHS:
            old = runs.get(f"historical_{arm}_{length}")
            new = runs.get(f"replay_{arm}")
            for task in TASKS:
                checks = {"receipts_valid": bool(old and new), "identity_match": False,
                          "input_cell_hash_match": False, "references_match": False,
                          "decoded_prediction_match": False, "official_score_match": False}
                cell = {"arm": arm, "length": length, "task": task, "local_index": 0, "checks": checks}
                if old and new:
                    prior, replay = old["rows"][task, length, 0], new["rows"][task, length, 0]
                    checks.update(identity_match=old["identity"] == new["identity"],
                                  input_cell_hash_match=old["cells"][task, length] == new["cells"][task, length],
                                  references_match=prior["references"] == replay["references"],
                                  decoded_prediction_match=prior["prediction"] == replay["prediction"],
                                  official_score_match=prior["official_task_score"] == replay["official_task_score"])
                    cell["input_cell_sha256"] = {"historical": old["cells"][task, length],
                                                 "replay": new["cells"][task, length]}
                cell["match"] = all(checks.values())
                cells.append(cell)
    # A matched pair must also belong to the same checkpoint/data across all arms.
    common_fields = ("checkpoint_sha256", "native_sha256_float32", "data_manifest_sha256",
                     "tokenizer_sha256", "ruler_commit", "seed", "torch", "cuda")
    common_identity = len(runs) == 9 and len({tuple(run["identity"][key] for key in common_fields)
                                            for run in runs.values()}) == 1
    matched = not errors and common_identity and all(cell["match"] for cell in cells)
    return {"schema": "HISTORICAL_RULER_REPLAY_AUDIT_V1", "status": "MATCH" if matched else "DRIFT",
            "expected_cells": 24, "matched_cells": sum(cell["match"] for cell in cells),
            "common_identity_match": common_identity, "cells": cells, "raw_receipts": receipts,
            "errors": errors, "comparison": "complete decoded prediction and official string_match_all score",
            "token_parity_established": False,
            "audit_script_sha256": sha256(Path(__file__).read_bytes()),
            "capacity_policy": "Native table_factor changes capacity only; compare Native tensor and unit gain, not factor.",
            "evidence_limit": "Only fixed local_index=0 canaries; decoded-only, not token/EOS parity or full-matrix execution equivalence. Input identity is the receipted cell-file hash; inputs are not retokenized."}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "expected_cells", "matched_cells")}))
    return 0 if result["status"] == "MATCH" else 1


if __name__ == "__main__":
    raise SystemExit(main())
