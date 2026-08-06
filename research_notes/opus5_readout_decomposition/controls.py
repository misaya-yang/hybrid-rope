#!/usr/bin/env python3
"""Control gate for the layer-resolved causal readout decomposition.

Verifies, before any interpretation is permitted:
  C1  every locally present record byte-matches its manifest SHA-256 and size
  C2  tensor schema / dtype / shape / layer-index contract
  C3  gold token ids agree across arms for the same prompt
  C4  final logit-lens parity (layer 31 == true model output)
  C5  arms are prompt-SHA matched; matched-pair count
  C6  shared provenance (passkey / code / script SHA) and distinct adapters
  C7  every matched readout prompt SHA resolves to a phase0 16K entry

Read-only. Loads no model, uses no GPU, downloads nothing.
Exit code 0 = all hard controls pass, 1 = gate failed (do not interpret).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[2]
TRACE_ROOT = REPO / "results/readout_conversion_s42_20260715/raw"
PHASE0_ROOT = REPO / "results/lora_sparse_conversion_s42_20260714"
ARMS = {"evq": "causal_evq_cosh", "geo": "causal_native_geo"}

EXPECTED_TRACE_SCHEMA = "evq_cosh.readout_conversion_trace.v1"
EXPECTED_MANIFEST_SCHEMA = "evq_cosh.readout_conversion_trace_manifest.v1"
EXPECTED_SHAPE = [3, 32, 128256]
PARITY_TOL = 1e-4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_controls(verbose: bool = True) -> dict[str, Any]:
    report: dict[str, Any] = {
        "repo_root": str(REPO),
        "trace_root": str(TRACE_ROOT.relative_to(REPO)),
        "phase0_root": str(PHASE0_ROOT.relative_to(REPO)),
        "hard_failures": [],
        "warnings": [],
        "arms": {},
    }

    def fail(msg: str) -> None:
        report["hard_failures"].append(msg)

    def warn(msg: str) -> None:
        report["warnings"].append(msg)

    manifests: dict[str, dict[str, Any]] = {}
    present: dict[str, dict[str, dict[str, Any]]] = {}

    # ---- C1/C2: manifest integrity and tensor contract -------------------
    for arm, folder in ARMS.items():
        mpath = TRACE_ROOT / folder / "manifest.json"
        if not mpath.is_file():
            fail(f"[C1] missing manifest: {mpath}")
            continue
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
        manifests[arm] = manifest
        if manifest.get("schema") != EXPECTED_MANIFEST_SCHEMA:
            fail(f"[C1] {arm}: manifest schema {manifest.get('schema')!r}")

        arm_present: dict[str, dict[str, Any]] = {}
        declared = len(manifest["records"])
        missing_records = []
        for rec in manifest["records"]:
            path = TRACE_ROOT / folder / rec["file"]
            if not path.is_file():
                missing_records.append((rec["file"], rec["depth_percent"]))
                continue
            digest = sha256_file(path)
            size = path.stat().st_size
            if digest != rec["sha256"]:
                fail(f"[C1] {arm}: SHA-256 mismatch {rec['file']}")
                continue
            if size != rec["size_bytes"]:
                fail(f"[C1] {arm}: size mismatch {rec['file']}")
                continue

            blob = torch.load(path, map_location="cpu", weights_only=False)
            if blob.get("schema") != EXPECTED_TRACE_SCHEMA:
                fail(f"[C1] {arm}: trace schema {blob.get('schema')!r} in {rec['file']}")
            if blob.get("prompt_sha256") != rec["prompt_sha256"]:
                fail(f"[C2] {arm}: prompt SHA disagrees with manifest in {rec['file']}")
            for key in ("full_logits", "ablated_logits"):
                tensor = blob[key]
                if list(tensor.shape) != EXPECTED_SHAPE:
                    fail(f"[C2] {arm}: {key} shape {list(tensor.shape)} in {rec['file']}")
                if tensor.dtype != torch.bfloat16:
                    fail(f"[C2] {arm}: {key} dtype {tensor.dtype} in {rec['file']}")
                if not torch.isfinite(tensor.float()).all():
                    fail(f"[C2] {arm}: non-finite values in {key} of {rec['file']}")
            layer_idx = blob["layer_indices"]
            if layer_idx.tolist() != list(range(EXPECTED_SHAPE[1])):
                fail(f"[C2] {arm}: layer_indices not 0..31 in {rec['file']}")
            parity = float(blob.get("final_logit_parity_max_abs", float("nan")))
            if not (parity <= PARITY_TOL):
                fail(f"[C4] {arm}: final-lens parity {parity} in {rec['file']}")
            gold = blob["gold_token_ids"].tolist()
            if len(gold) != EXPECTED_SHAPE[0]:
                fail(f"[C2] {arm}: gold_token_ids length {len(gold)} in {rec['file']}")

            arm_present[rec["prompt_sha256"]] = {
                "file": rec["file"],
                "depth_percent": rec["depth_percent"],
                "target_length": rec["target_length"],
                "gold_token_ids": gold,
                "parity_max_abs": parity,
                "sha256": digest,
            }
            del blob

        present[arm] = arm_present
        report["arms"][arm] = {
            "folder": folder,
            "manifest_status": manifest.get("status"),
            "measurement_label": manifest.get("measurement_label"),
            "single_seed_supporting": manifest.get("single_seed_supporting"),
            "substrate": manifest.get("substrate"),
            "records_declared": declared,
            "records_present": len(arm_present),
            "records_missing": [
                {"file": f, "depth_percent": d} for f, d in missing_records
            ],
            "adapter_sha256": manifest["adapter"]["adapter_sha256"],
            "protocol_sha256": manifest["adapter"]["protocol_sha256"],
            "code_sha256": manifest["adapter"]["code_sha256"],
            "passkey_sha256": manifest["passkey_sha256"],
            "script_sha256": manifest["script_sha256"],
        }
        if missing_records:
            warn(
                f"[C1] {arm}: {len(missing_records)}/{declared} manifest records are "
                f"absent locally (depths {sorted({d for _, d in missing_records})}); "
                "the remote run was complete, this machine holds a subset"
            )

    if len(present) != 2:
        fail("[C5] could not load both arms")
        report["gate_passed"] = False
        return report

    # ---- C5: pairing ------------------------------------------------------
    shared = sorted(
        set(present["evq"]) & set(present["geo"]),
        key=lambda s: present["evq"][s]["file"],
    )
    report["matched_pairs"] = [
        {
            "prompt_sha256": s,
            "depth_percent": present["evq"][s]["depth_percent"],
            "target_length": present["evq"][s]["target_length"],
            "evq_file": present["evq"][s]["file"],
            "geo_file": present["geo"][s]["file"],
            "gold_token_ids": present["evq"][s]["gold_token_ids"],
        }
        for s in shared
    ]
    report["geo_only"] = [
        {"prompt_sha256": s, "depth_percent": present["geo"][s]["depth_percent"]}
        for s in sorted(set(present["geo"]) - set(present["evq"]))
    ]
    report["evq_only"] = [
        {"prompt_sha256": s, "depth_percent": present["evq"][s]["depth_percent"]}
        for s in sorted(set(present["evq"]) - set(present["geo"]))
    ]
    if len(shared) < 5:
        fail(f"[C5] only {len(shared)} matched pairs (expected >= 5)")

    # ---- C3: gold ids agree across arms ----------------------------------
    for s in shared:
        if present["evq"][s]["gold_token_ids"] != present["geo"][s]["gold_token_ids"]:
            fail(f"[C3] gold_token_ids differ across arms for prompt {s[:12]}")
        if present["evq"][s]["target_length"] != present["geo"][s]["target_length"]:
            fail(f"[C3] target_length differs across arms for prompt {s[:12]}")

    # ---- C6: provenance ---------------------------------------------------
    a, g = report["arms"]["evq"], report["arms"]["geo"]
    for key in ("passkey_sha256", "code_sha256", "script_sha256"):
        if a[key] != g[key]:
            fail(f"[C6] arms disagree on {key}")
    if a["adapter_sha256"] == g["adapter_sha256"]:
        fail("[C6] arms share an adapter SHA-256 — they are not distinct arms")
    if a["substrate"] == g["substrate"]:
        fail("[C6] arms share a substrate label")

    # ---- C7: phase0 cross-match ------------------------------------------
    phase0_index: dict[str, dict[str, str]] = {}
    for arm, fname in (("evq", "phase0_evq.json"), ("geo", "phase0_geo.json")):
        blob = json.loads((PHASE0_ROOT / fname).read_text(encoding="utf-8"))
        if blob.get("query_contract") != "last_prompt_token_predicting_first_answer_token":
            fail(f"[C7] {arm}: unexpected phase0 query contract")
        phase0_index[arm] = {
            e["prompt_sha256"]: e["example_id"]
            for e in blob["results"]
            if int(e["target_length"]) == 16384
        }
    if set(phase0_index["evq"]) != set(phase0_index["geo"]):
        fail("[C7] phase0 arms cover different 16K prompts")
    resolved = []
    for s in shared:
        example_id = phase0_index["evq"].get(s)
        if example_id is None:
            fail(f"[C7] matched readout prompt {s[:12]} has no phase0 16K entry")
        resolved.append({"prompt_sha256": s, "phase0_example_id": example_id})
    report["phase0_crossmatch"] = resolved

    report["gate_passed"] = not report["hard_failures"]

    if verbose:
        print("=" * 72)
        print("CONTROL GATE — layer-resolved causal readout decomposition")
        print("=" * 72)
        for arm in ("evq", "geo"):
            info = report["arms"][arm]
            print(
                f"{arm:>4}: {info['records_present']}/{info['records_declared']} records "
                f"present, SHA-verified | substrate={info['substrate']} "
                f"| adapter={info['adapter_sha256'][:12]}"
            )
        print(f"matched pairs : {len(shared)}")
        for pair in report["matched_pairs"]:
            print(
                f"   {pair['evq_file'].split('/')[-1][:3]}  sha={pair['prompt_sha256'][:12]}"
                f"  depth={pair['depth_percent']:5.1f}  gold_ids={pair['gold_token_ids']}"
            )
        if report["geo_only"]:
            print(
                "geo-only extra:",
                [(p["prompt_sha256"][:8], p["depth_percent"]) for p in report["geo_only"]],
            )
        print(f"phase0 16K cross-match: {len(resolved)}/{len(shared)} resolved")
        for w in report["warnings"]:
            print(f"WARNING {w}")
        for f in report["hard_failures"]:
            print(f"FAIL    {f}")
        print(f"\nGATE: {'PASS' if report['gate_passed'] else 'FAIL'}")

    return report


def main() -> int:
    out_dir = Path(__file__).resolve().parent
    report = run_controls(verbose=True)
    (out_dir / "controls.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if report["gate_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
