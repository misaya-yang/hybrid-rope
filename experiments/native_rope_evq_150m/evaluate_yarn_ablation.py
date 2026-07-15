#!/usr/bin/env python3
"""Checkpoint-only official-YaRN frequency-versus-mscale ablation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.native_rope_evq_150m.evaluate import (
    EVAL_LENGTHS,
    YARN_ABLATION_OPERATORS,
    _load_checkpoint,
    _write_json,
    apply_registered_operator,
    evaluate_natural_text,
    evaluate_passkey,
    fixed_validation_offsets,
    validate_checkpoint_identity,
)
from experiments.native_rope_evq_150m.prepare_data import (
    sha256_file,
    validate_data_manifest,
)
from experiments.native_rope_evq_150m.protocol import ARMS
from experiments.native_rope_evq_150m.train import tensor_sha256


ABLATION_LENGTHS = (4_096, 8_192, 16_384)


def attribution_row(
    *,
    native_raw: float,
    evq_raw: float,
    native_operator: float,
    evq_operator: float,
) -> dict[str, float]:
    return {
        "substrate_gap": float(native_operator) - float(evq_operator),
        "interaction": (float(evq_operator) - float(evq_raw))
        - (float(native_operator) - float(native_raw)),
    }


def analyze_results(report: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    reproduction: dict[str, Any] = {}
    max_natural = 0.0
    max_passkey = 0.0
    case_identity = True
    for arm in ARMS:
        reproduction[arm] = {}
        for operator, reference_operator in (("raw", "raw"), ("full", "yarn")):
            actual = report["conditions"][arm]["operators"][operator]
            expected = reference["conditions"][arm]["operators"][reference_operator]
            natural_diff = max(
                abs(float(a) - float(b))
                for length in map(str, ABLATION_LENGTHS)
                for a, b in zip(
                    actual["natural_text"][length]["per_offset_nll"],
                    expected["natural_text"][length]["per_offset_nll"],
                )
            )
            passkey_diff = max(
                abs(float(a["nll_gap"]) - float(b["nll_gap"]))
                for a, b in zip(
                    actual["passkey"]["details"], expected["passkey"]["details"]
                )
            )
            identity_fields = (
                "length",
                "depth",
                "trial",
                "seed",
                "correct_passkey",
                "wrong_passkey",
            )
            identities_equal = len(actual["passkey"]["details"]) == len(
                expected["passkey"]["details"]
            ) and all(
                all(a[field] == b[field] for field in identity_fields)
                for a, b in zip(
                    actual["passkey"]["details"], expected["passkey"]["details"]
                )
            )
            reproduction[arm][operator] = {
                "natural_text_max_abs_diff": natural_diff,
                "passkey_gap_max_abs_diff": passkey_diff,
                "passkey_case_identity_equal": identities_equal,
            }
            max_natural = max(max_natural, natural_diff)
            max_passkey = max(max_passkey, passkey_diff)
            case_identity = case_identity and identities_equal

    attribution: dict[str, Any] = {}
    for length in map(str, ABLATION_LENGTHS):
        native_raw = report["conditions"]["native_rope"]["operators"]["raw"][
            "natural_text"
        ][length]["mean_nll"]
        evq_raw = report["conditions"]["endpoint_evq_tau1p5"]["operators"]["raw"][
            "natural_text"
        ][length]["mean_nll"]
        attribution[length] = {}
        for operator, _, _ in YARN_ABLATION_OPERATORS:
            native = report["conditions"]["native_rope"]["operators"][operator][
                "natural_text"
            ][length]
            evq = report["conditions"]["endpoint_evq_tau1p5"]["operators"][operator][
                "natural_text"
            ][length]
            attribution[length][operator] = {
                "native_mean_nll": native["mean_nll"],
                "native_ppl": native["ppl"],
                "evq_mean_nll": evq["mean_nll"],
                "evq_ppl": evq["ppl"],
                **attribution_row(
                    native_raw=native_raw,
                    evq_raw=evq_raw,
                    native_operator=native["mean_nll"],
                    evq_operator=evq["mean_nll"],
                ),
            }
    return {
        "reference_reproduction": {
            "passed": max_natural == 0.0
            and max_passkey == 0.0
            and case_identity,
            "natural_text_max_abs_diff": max_natural,
            "passkey_gap_max_abs_diff": max_passkey,
            "passkey_case_identity_equal": case_identity,
            "details": reproduction,
        },
        "natural_text_attribution": attribution,
    }


def preflight(work_dir: Path, manifest_path: Path, reference_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    reference = json.loads(reference_path.read_text())
    validate_data_manifest(manifest, check_files=True)
    if sha256_file(manifest_path) != reference.get("data_manifest_sha256"):
        raise ValueError("data manifest does not match the six-cell reference")

    identities: dict[str, Any] = {}
    operators: dict[str, Any] = {}
    for arm in ARMS:
        reference_arm = reference["conditions"][arm]
        identity = validate_checkpoint_identity(
            work_dir / arm, arm, reference_arm["train_meta"]
        )
        identities[arm] = {
            "checkpoint_sha256": identity["checkpoint_sha256"],
            "inv_freq_sha256": identity["inv_freq_sha256"],
        }
        operators[arm] = {}
        base = identity["inv_freq"]
        for name, use_frequency, use_scaling in YARN_ABLATION_OPERATORS:
            operators[arm][name] = {}
            for length in ABLATION_LENGTHS:
                inv, mscale, meta = apply_registered_operator(
                    base,
                    arm=arm,
                    operator=name,
                    length=length,
                    use_frequency_transform=use_frequency,
                    use_attention_scaling=use_scaling,
                )
                operators[arm][name][str(length)] = {
                    "inv_freq_sha256": tensor_sha256(inv),
                    "mscale": mscale,
                    "mode": meta["mode"],
                }
                reference_operator = "yarn" if name == "full" else "raw" if name == "raw" else None
                if reference_operator:
                    expected = reference_arm["operators"][reference_operator]["natural_text"][str(length)]
                    if tensor_sha256(inv) != expected["inv_freq_sha256"]:
                        raise ValueError(f"{arm}/{name}/{length} frequency hash mismatch")
                    if abs(mscale - float(expected["operator"]["mscale"])) > 1e-12:
                        raise ValueError(f"{arm}/{name}/{length} mscale mismatch")
                    offsets = fixed_validation_offsets(
                        int(manifest["validation"]["tokens"]), length, chunks=8
                    )
                    if offsets != expected["offsets"]:
                        raise ValueError(f"{arm}/{name}/{length} offset mismatch")

    raw_passkey = reference["conditions"][ARMS[0]]["operators"]["raw"]["passkey"]
    details = raw_passkey.get("details", [])
    if len(details) != 100 or raw_passkey.get("metric") != "teacher-forced NLL_wrong - NLL_correct":
        raise ValueError("reference Passkey contract is not the frozen 100-case NLL-gap suite")
    return {
        "dry_run": True,
        "data_manifest_sha256": sha256_file(manifest_path),
        "reference_sha256": sha256_file(reference_path),
        "checkpoint_identities": identities,
        "operators": operators,
        "natural_text_lengths": list(ABLATION_LENGTHS),
        "natural_text_offsets": 8,
        "passkey_cases": 100,
        "passkey_lengths": list(EVAL_LENGTHS),
        "passkey_metric": raw_passkey["metric"],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for evaluation")
    work_dir = args.work_dir.resolve()
    manifest_path = args.data_manifest.resolve()
    reference_path = args.reference.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    check = preflight(work_dir, manifest_path, reference_path)
    manifest = json.loads(manifest_path.read_text())

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        manifest["tokenizer"]["path"], local_files_only=True
    )
    validation = np.load(manifest["validation"]["path"], mmap_mode="r", allow_pickle=False)
    passkey_indices = np.load(manifest["passkey"]["indices_path"], allow_pickle=False)
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_role": "single-seed checkpoint-only official-YaRN component ablation",
        "claim_boundary": "supporting mechanistic evidence; no training and no primary-claim promotion",
        "preflight": check,
        "conditions": {},
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    for arm in ARMS:
        model, metadata, base = _load_checkpoint(work_dir / arm, arm)
        report["conditions"][arm] = {"train_meta": metadata, "operators": {}}
        for name, use_frequency, use_scaling in YARN_ABLATION_OPERATORS:
            started = time.time()
            natural = evaluate_natural_text(
                model,
                validation,
                base,
                arm=arm,
                operator=name,
                chunks=8,
                lengths=ABLATION_LENGTHS,
                use_frequency_transform=use_frequency,
                use_attention_scaling=use_scaling,
            )
            passkey = evaluate_passkey(
                model,
                tokenizer,
                validation,
                passkey_indices,
                base,
                arm=arm,
                operator=name,
                trials=5,
                use_frequency_transform=use_frequency,
                use_attention_scaling=use_scaling,
            )
            report["conditions"][arm]["operators"][name] = {
                "natural_text": natural,
                "passkey": passkey,
                "seconds": time.time() - started,
            }
        del model
        torch.cuda.empty_cache()
    report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "raw_results.json", report)
    analysis = analyze_results(report, json.loads(reference_path.read_text()))
    _write_json(output_dir / "analysis.json", analysis)
    if not analysis["reference_reproduction"]["passed"]:
        raise RuntimeError("raw/full reproduction gate failed; see analysis.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work_dir", type=Path, required=True)
    parser.add_argument("--data_manifest", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps(preflight(args.work_dir.resolve(), args.data_manifest.resolve(), args.reference.resolve()), indent=2, sort_keys=True))
        return
    if args.output_dir is None:
        parser.error("--output_dir is required unless --dry_run")
    run(args)


if __name__ == "__main__":
    main()
