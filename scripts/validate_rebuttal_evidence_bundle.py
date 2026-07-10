#!/usr/bin/env python3
"""Validate the portable July rebuttal evidence bundle without ML dependencies."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CURATED = ROOT / "data" / "curated"
REBUTTAL_DOC = ROOT / "rebuttal_7" / "IGNORED_ASSET_RECONCILIATION.md"
CORE_ASSET_SUMMARY = ROOT / "rebuttal_7" / "LOCAL_CORE_ASSET_PROMOTION_SUMMARY.md"

EXPECTED_JSON = {
    "learnable_tau_128tok_evidence.json": "report-backed",
    "mla_channel_count_125m_pilot.json": "report-backed",
    "primary1_evq_yarn_10pct_raw.json": "raw-json-backed",
    "primary2_l128_fixed_tau5_3seed.json": "raw-json-backed",
    "phase11_l256_3seed_recovered.json": "raw-json-backed",
    "phase11b_125m_l256_3seed.json": "raw-json-backed",
    "phase16_99run_manifest.meta.json": "sanitized-run-manifest",
    "quality_454m_full_eval.json": "raw-json-backed",
    "table18_mla_3seed_aggregate.json": "raw-json-backed",
    "text_base_10k_500k_pilot.json": "raw-json-backed",
}

EXPECTED_RAW_SHA256 = {
    "data/curated/eval_3seeds_full_results.json": (
        "1e44d30bb880e4b7427ae55bd7034782989152bd2afca9217495f9b8ece30953"
    ),
    "data/results_5090b/evq_yarn_10pct_allseeds.json": (
        "1dbec88efac6d7442796d81fa1d073e3a76b1388dd815764bcb8b619f234511c"
    ),
    "data/evq_128tok_results/extended_sweep/results_final.json": (
        "980246a9950d7e40e39278a4feef8115e6b35a9feb6e1fe1eb190faac1caf1fd"
    ),
    "data/evq_128tok_results/phase7/multiseed/results_final.json": (
        "4fd031f44d966117fa7473eaf405b4503329d81744a6938158fd588901535d47"
    ),
}

EXPECTED_PHASE16_FIELDS = [
    "stage",
    "run_id",
    "config_id",
    "tier",
    "seq_len",
    "num_heads",
    "head_dim",
    "tau",
    "theory_tau",
    "seed",
    "train_tokens",
    "eval_lengths",
    "passkey_lengths",
    "passkey_trials",
    "completed_at",
    "train_time_sec",
    "eval_time_sec",
    "inv_freq_hash",
    "ppl_json",
    "passkey_summary_json",
]

FORBIDDEN = re.compile(
    r"misaya|yanghej|hejaz|/Users/|/root/autodl-tmp|sshpass|seetacloud|"
    r"BEGIN (?:OPENSSH|RSA) PRIVATE KEY|hf_[A-Za-z0-9]{20,}|"
    r"ghp_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}",
    re.IGNORECASE,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def validate_phase16_manifest(csv_path: Path, meta_path: Path) -> list[str]:
    errors: list[str] = []
    if not csv_path.is_file():
        return [f"missing: {csv_path}"]
    if not meta_path.is_file():
        return [f"missing: {meta_path}"]

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if fieldnames != EXPECTED_PHASE16_FIELDS:
        errors.append("phase16 CSV header does not match the exporter schema")
    if len(rows) != 99:
        errors.append(f"phase16 row count: expected 99, got {len(rows)}")

    stage_counts = Counter(row.get("stage") for row in rows)
    if stage_counts != Counter({"pilot": 45, "confirm": 54}):
        errors.append(
            "phase16 stage counts: expected pilot=45 and confirm=54, "
            f"got {dict(stage_counts)}"
        )

    run_ids = [row.get("run_id", "") for row in rows]
    if any(not run_id for run_id in run_ids) or len(set(run_ids)) != len(run_ids):
        errors.append("phase16 must contain 99 non-empty unique run_id values")

    for index, row in enumerate(rows, start=2):
        inv_freq_hash = row.get("inv_freq_hash", "")
        if re.fullmatch(r"[0-9a-f]{16}", inv_freq_hash) is None:
            errors.append(f"phase16 row {index} has an invalid inv_freq_hash")
        for field in (
            "eval_lengths",
            "passkey_lengths",
            "ppl_json",
            "passkey_summary_json",
        ):
            try:
                value = json.loads(row.get(field, ""))
            except json.JSONDecodeError:
                errors.append(f"phase16 row {index} has invalid JSON in {field}")
                continue
            if field.endswith("_json") and not isinstance(value, dict):
                errors.append(f"phase16 row {index} expects an object in {field}")
            if field.endswith("_lengths") and not isinstance(value, list):
                errors.append(f"phase16 row {index} expects a list in {field}")

    try:
        metadata = load_json(meta_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"invalid phase16 metadata: {exc}")
        return errors
    if metadata.get("row_count") != len(rows):
        errors.append("phase16 metadata row_count does not match the CSV")
    if metadata.get("stages") != {"pilot": 45, "confirm": 54}:
        errors.append("phase16 metadata stages do not match pilot=45/confirm=54")
    if metadata.get("sha256") != digest(csv_path):
        errors.append("phase16 CSV SHA256 does not match its metadata sidecar")
    return errors


def validate_bundle(require_tracked: bool = True) -> list[str]:
    errors: list[str] = []
    paths = [CURATED / name for name in EXPECTED_JSON]
    paths.extend(ROOT / rel for rel in EXPECTED_RAW_SHA256)
    csv_path = CURATED / "phase16_99run_manifest.csv"
    paths.extend([csv_path, REBUTTAL_DOC, CORE_ASSET_SUMMARY])

    for name, status in EXPECTED_JSON.items():
        path = CURATED / name
        if not path.is_file():
            errors.append(f"missing: {path.relative_to(ROOT)}")
            continue
        try:
            actual = load_json(path).get("provenance_status")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"invalid JSON {path.relative_to(ROOT)}: {exc}")
            continue
        if actual != status:
            errors.append(
                f"wrong provenance tier for {name}: expected {status}, got {actual}"
            )

    for rel, expected in EXPECTED_RAW_SHA256.items():
        path = ROOT / rel
        if not path.is_file():
            errors.append(f"missing: {rel}")
            continue
        actual = digest(path)
        if actual != expected:
            errors.append(
                f"raw SHA256 mismatch for {rel}: expected {expected}, got {actual}"
            )

    for path in CURATED.glob("*.json"):
        if load_json(path).get("provenance_status") == "trace-only":
            errors.append(f"trace-only asset stored in curated directory: {path.name}")

    errors.extend(
        validate_phase16_manifest(
            csv_path, CURATED / "phase16_99run_manifest.meta.json"
        )
    )

    if REBUTTAL_DOC.is_file():
        handoff = REBUTTAL_DOC.read_text(encoding="utf-8")
        for question in range(1, 19):
            heading = f"### F5-Q{question} "
            if handoff.count(heading) != 1:
                errors.append(f"handoff must contain one heading: {heading.strip()}")

    for path in paths:
        if path.is_file() and FORBIDDEN.search(path.read_text(encoding="utf-8")):
            errors.append(f"identity/path leak: {path.relative_to(ROOT)}")

    if require_tracked:
        existing = [str(path.relative_to(ROOT)) for path in paths if path.is_file()]
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", *existing],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode:
            errors.append("one or more bundle files are not tracked by git")

    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-tracked-check",
        action="store_true",
        help="Validate content before the new files have been staged.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    errors = validate_bundle(require_tracked=not args.skip_tracked_check)
    if errors:
        for error in errors:
            print(f"FAIL {error}")
        raise SystemExit(1)
    print("rebuttal evidence bundle: PASS")


if __name__ == "__main__":
    main()
