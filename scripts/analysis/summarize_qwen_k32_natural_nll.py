#!/usr/bin/env python3
"""CPU-only hash-bound summary of the fixed Qwen K32 packed-natural NLL panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

ARMS = ("Native", "normalized_raw_index", "official_equation_yarn")
LENGTHS = (32768, 65536)
STREAMS = 32
TARGET_TOKENS = 256
BOOTSTRAP_SEED = 202609030
BOOTSTRAP_SAMPLES = 10000
WEIGHT_SHA256 = "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
CONFIG_SHA256 = "18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45"
PROFILE = {
    "Native": ("6d1e10125bd0468a7cf91c6175a3af31c1bffca24592cf5630f0f8402a8746e3", None, 1.0),
    "normalized_raw_index": ("8c19ab976f71d30c6409f78a661209a8535ef9f101e8bf42f5bfce6f7817dc5f",
                             "36e09e014a86f8169296e6b5f2997ba7fea08ceb005b4e887085289d53db57a1",
                             1 + .074 * math.log(2)),
    "official_equation_yarn": ("d9eb5ac0185e84f2afa85997f10e4c51de97e3a2f937325769dd45ff86a0ea59",
                               "980d8d16b84d792fb7b50d42941747a881985ef0fa246e7fa0f0964a9b79ca03",
                               1 + .1 * math.log(2)),
}
PACKING_CONTRACT = (
    "source-order unique documents; insert one EOS between complete documents; truncate only "
    "the last document to finish each 65536-token stream and discard its unused suffix; never "
    "reuse a source document; 32768 is the suffix of the paired 65536 stream"
)
TERMINAL_STATUS = "QWEN_K32_PACKED_NATURAL_NLL_COMPLETE"
FROZEN_STATUS = "QWEN_K32_PACKED_NATURAL_NLL_FROZEN"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_hash(value, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"missing or invalid {name} SHA-256")
    return value


def finite_exp(value: float):
    try:
        result = math.exp(value)
        return result if math.isfinite(result) else None
    except OverflowError:
        return None


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def validate_tokenizer_files(value) -> list[dict]:
    if not isinstance(value, list) or not value:
        raise ValueError("tokenizer file receipts are missing")
    result, names = [], set()
    for item in value:
        if (not isinstance(item, dict) or set(item) != {"name", "sha256", "bytes"}
                or not isinstance(item["name"], str) or not item["name"]
                or "/" in item["name"] or "\\" in item["name"] or item["name"] in names
                or type(item["bytes"]) is not int or item["bytes"] <= 0):
            raise ValueError("invalid tokenizer file receipt")
        names.add(item["name"])
        result.append({"name": item["name"], "sha256": checked_hash(item["sha256"], "tokenizer file"),
                       "bytes": item["bytes"]})
    return result


def validate_source(value) -> dict:
    if (not isinstance(value, dict) or set(value) != {"name", "sha256", "start_row", "consumed_row_range"}
            or not isinstance(value["name"], str) or not value["name"]
            or "/" in value["name"] or "\\" in value["name"]
            or type(value["start_row"]) is not int or value["start_row"] < 0
            or not isinstance(value["consumed_row_range"], list) or len(value["consumed_row_range"]) != 2
            or any(type(item) is not int for item in value["consumed_row_range"])
            or value["consumed_row_range"][0] < value["start_row"]
            or value["consumed_row_range"][0] > value["consumed_row_range"][1]):
        raise ValueError("invalid packed-natural source receipt")
    return {**value, "sha256": checked_hash(value["sha256"], "source")}


def validate_profiles(manifest: dict) -> dict:
    profiles = manifest.get("profiles")
    if not isinstance(profiles, list) or [item.get("name") for item in profiles] != list(ARMS):
        raise ValueError("exactly three ordered frozen profiles are required")
    identities = {}
    for item in profiles:
        name = item["name"]
        tensor_hash, file_hash, gain = PROFILE[name]
        if (item.get("tensor_sha256") != tensor_hash or item.get("file_sha256") != file_hash
                or type(item.get("attention_scaling")) not in (int, float)
                or not math.isclose(item["attention_scaling"], gain, rel_tol=0, abs_tol=1e-12)):
            raise ValueError("frozen profile identity or gain drift")
        identities[name] = {"tensor_sha256": tensor_hash, "file_sha256": file_hash,
                            "attention_scaling": float(gain)}
    return identities


def summarize(result: dict, manifest: dict, rows: list[dict], raw_hashes=None) -> dict:
    if result.get("status") != TERMINAL_STATUS or manifest.get("status") != FROZEN_STATUS:
        raise ValueError("only terminal results with a frozen run manifest are admitted")
    if (manifest.get("checkpoint_weight_sha256") != WEIGHT_SHA256
            or manifest.get("config_sha256") != CONFIG_SHA256
            or manifest.get("lengths") != list(LENGTHS)
            or manifest.get("natural_streams") != STREAMS
            or manifest.get("target_tokens") != TARGET_TOKENS
            or manifest.get("arm_order") != list(ARMS)
            or manifest.get("packing_contract") != PACKING_CONTRACT
            or manifest.get("use_cache") is not False or manifest.get("compile") is not False
            or manifest.get("model_updates") != 0 or manifest.get("profile_selection") is not False
            or manifest.get("all_profiles_loaded_before_inference") is not True):
        raise ValueError("frozen packed-natural run contract drift")
    bound = {key: checked_hash(manifest.get(key), key) for key in (
        "checkpoint_weight_sha256", "config_sha256", "data_manifest_sha256", "data_rows_sha256",
        "script_sha256", "model_source_sha256", "attention_source_sha256")}
    evaluator = Path(__file__).resolve().parents[1] / "eval" / "eval_qwen_k32_natural_nll.py"
    if bound["script_sha256"] != sha256(evaluator):
        raise ValueError("evaluator source hash differs from the frozen run")
    source = validate_source(manifest.get("source"))
    tokenizer_files = validate_tokenizer_files(manifest.get("tokenizer_files"))
    profiles = validate_profiles(manifest)
    ids = [f"qwen-k32-natural-{index:03d}" for index in range(STREAMS)]
    expected = {(arm, sid, length) for arm in ARMS for sid in ids for length in LENGTHS}
    cells, paired, stream_identity = {}, {}, {}
    metadata = ("family", "variant", "split", "source_row_start", "source_row_end",
                "source_document_count", "source_set_sha256", "prompt_ids_sha256",
                "target_ids_sha256", "target_start", "target_tokens")
    for row in rows:
        arm, sid, length = row.get("arm"), row.get("sample_id"), row.get("length")
        key = arm, sid, length
        profile = profiles.get(arm)
        loss = row.get("nll")
        if (profile is None or key not in expected or key in cells
                or row.get("family") != "natural" or row.get("variant") != "packed_natural"
                or row.get("split") != "holdout" or row.get("target_start") != length - TARGET_TOKENS
                or row.get("target_tokens") != TARGET_TOKENS
                or row.get("table_sha256_float32") != profile["tensor_sha256"]
                or row.get("attention_scaling") != profile["attention_scaling"]
                or type(loss) not in (int, float) or not math.isfinite(loss) or loss < 0
                or type(row.get("source_row_start")) is not int
                or type(row.get("source_row_end")) is not int
                or row["source_row_start"] > row["source_row_end"]
                or type(row.get("source_document_count")) is not int
                or row["source_document_count"] <= 0):
            raise ValueError("duplicated/unregistered row or frozen row/profile metadata drift")
        for field in ("source_set_sha256", "prompt_ids_sha256", "target_ids_sha256"):
            checked_hash(row.get(field), field)
        row_metadata = tuple(row.get(field) for field in metadata)
        if (sid, length) in paired and paired[sid, length] != row_metadata:
            raise ValueError("paired stream metadata differs across profiles")
        paired[sid, length] = row_metadata
        identity = (row["source_row_start"], row["source_row_end"], row["source_document_count"],
                    row["source_set_sha256"], row["target_ids_sha256"])
        if sid in stream_identity and stream_identity[sid] != identity:
            raise ValueError("source stream or final-256 target differs across lengths/profiles")
        stream_identity[sid] = identity
        cells[key] = float(loss)
    if set(cells) != expected or result.get("rows") != len(expected) or len(stream_identity) != STREAMS:
        raise ValueError("requires exactly 32 fully paired streams for all profiles and lengths")
    values = np.array([[[cells[arm, sid, length] for arm in ARMS]
                        for length in LENGTHS] for sid in ids], dtype=np.float64)
    indices = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, STREAMS, size=(BOOTSTRAP_SAMPLES, STREAMS))
    draws, means = values[indices].mean(axis=1), values.mean(axis=0)
    curves = {}
    stored_curves = result.get("curves")
    if not isinstance(stored_curves, dict) or set(stored_curves) != set(ARMS):
        raise ValueError("terminal curves are missing or contain extra profiles")
    if any(set(stored_curves[arm]) != {str(length) for length in LENGTHS} for arm in ARMS):
        raise ValueError("terminal curves are missing or contain extra lengths")
    for length_index, length in enumerate(LENGTHS):
        curves[str(length)] = {}
        for arm_index, arm in enumerate(ARMS):
            mean = float(means[length_index, arm_index])
            stored = stored_curves[arm].get(str(length))
            paired_delta = float(mean - means[length_index, 0])
            if (not isinstance(stored, dict) or set(stored) != {
                    "streams", "mean_tail_nll", "mean_paired_delta_vs_native"}
                    or stored["streams"] != STREAMS
                    or not math.isclose(stored["mean_tail_nll"], mean, rel_tol=0, abs_tol=1e-12)
                    or not math.isclose(stored["mean_paired_delta_vs_native"], paired_delta,
                                        rel_tol=0, abs_tol=1e-12)):
                raise ValueError("terminal curve differs from raw stream NLL")
            curves[str(length)][arm] = {"mean_nll": mean, "ppl": finite_exp(mean)}
    index_retention = finite_exp(float(means[0, 0] - means[0, 1]))
    index_native_draws = draws[:, 1, 1] - draws[:, 1, 0]
    index_yarn_draws = draws[:, 1, 1] - draws[:, 1, 2]
    index_native_ci = np.quantile(index_native_draws, [.025, .975]).tolist()
    index_yarn_ci = np.quantile(index_yarn_draws, [.025, .975]).tolist()
    resolver = (index_retention is not None and index_retention >= .875
                and index_native_ci[1] < 0)
    ranking = ("INDEX_FAVORED" if index_yarn_ci[1] < 0 else
               "YARN_FAVORED" if index_yarn_ci[0] > 0 else "UNRESOLVED")
    return {
        "status": "QWEN_K32_PACKED_NATURAL_NLL_SUMMARIZED", "lengths": list(LENGTHS),
        "streams": STREAMS, "target_tokens": TARGET_TOKENS, "curves": curves,
        "index_native_ppl_retention_32k": {"value": index_retention, "threshold": .875,
                                           "passes_point_gate": bool(index_retention is not None and index_retention >= .875)},
        "index_minus_native_64k": {"mean_delta_nll": float(means[1, 1] - means[1, 0]),
                                    "joint_paired_stream_ci95": index_native_ci},
        "index_minus_yarn_64k": {"mean_delta_nll": float(means[1, 1] - means[1, 2]),
                                  "joint_paired_stream_ci95": index_yarn_ci},
        "classification": {"resolver": "PASS" if resolver else "NOT_PASS",
                           "index_vs_yarn": ranking},
        "decision_rule": {
            "resolver": "PASS iff 32K index PPL retention >= 0.875 and the 64K index-minus-Native NLL CI upper bound < 0.",
            "index_vs_yarn": "INDEX_FAVORED iff CI upper < 0; YARN_FAVORED iff CI lower > 0; otherwise UNRESOLVED."},
        "bootstrap": {"seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_SAMPLES,
            "confidence": .95, "unit": "same 32 stream indices resampled jointly across all profiles and lengths",
            "numpy_version": np.__version__, "quantile_interpolation": "linear"},
        "identity": {**bound, "profiles": profiles, "source": source,
                     "tokenizer_files": tokenizer_files,
                     "tokenizer_files_sha256": canonical_hash(tokenizer_files)},
        "raw_hashes": raw_hashes or {}, "summary_code_sha256": sha256(Path(__file__)),
        "profile_selection_performed": False,
        "evidence_limit": "Paired final-256 teacher-forced NLL on 32 packed streams; no generation, K causality, profile search, or SOTA claim."}


def load_and_summarize(root: Path) -> dict:
    paths = {name: root / filename for name, filename in (
        ("results", "results.json"), ("run_manifest", "run_manifest.json"),
        ("examples", "examples.jsonl"))}
    result = json.loads(paths["results"].read_text())
    if result.get("status") != TERMINAL_STATUS:
        raise ValueError("packed-natural panel is not terminal")
    hashes = {name + "_sha256": sha256(path) for name, path in paths.items()}
    if (result.get("examples_sha256") != hashes["examples_sha256"]
            or result.get("run_manifest_sha256") != hashes["run_manifest_sha256"]):
        raise ValueError("terminal examples/run-manifest hash mismatch")
    rows = [json.loads(line) for line in paths["examples"].read_text().splitlines() if line.strip()]
    return summarize(result, json.loads(paths["run_manifest"].read_text()), rows, hashes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = load_and_summarize(args.root)
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError):
        report = {"status": "INVALID_OR_INCOMPLETE_NLL_PANEL",
                  "reason": "input validation failed; no metrics promoted"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(report["status"])
    return 0 if report["status"] == "QWEN_K32_PACKED_NATURAL_NLL_SUMMARIZED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
