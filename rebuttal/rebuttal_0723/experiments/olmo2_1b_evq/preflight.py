#!/usr/bin/env python3
"""Fail-closed no-GPU preflight for the paid OLMo-2 experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.contract import (
    ACTUAL_PARAMETER_COUNT,
    FIRST_GATE_STEPS,
    FIRST_GATE_TOKENS,
    GEO1000_REVISION,
    GEO1000_WEIGHT_FILES,
    GLOBAL_BATCH_SEQUENCES,
    MODEL_ID,
    OFFICIAL_CONFIG_SHA256,
    SEQUENCE_LENGTH,
    STEP0_REVISION,
    STEP0_WEIGHT_FILES,
    TOKENIZER_MARKERS,
    assert_frequency_contract,
    assert_model_config,
    named_parameter_metadata,
    parameter_identity_digest,
    patch_endpoint_evq,
    sha256_file,
    sha256_json,
    trainable_parameter_count,
    validate_weight_files,
)
from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.prepare_official_stream import (
    validate_existing as validate_existing_stream,
)


# One step-1000 FP32-model + Adam checkpoint occupies about 16.6 GiB.
# The fixed 50 GiB volume therefore requires 25 GiB free at launch, leaving
# explicit headroom for logs, validation output, and the atomic save.
MIN_CHECKPOINT_FREE_BYTES = 25 * 1024**3
ENVIRONMENT_CONTRACT = {
    "torch": "2.8.0+cu128",
    "transformers": "4.57.6",
    "safetensors": "0.8.0",
    "liger-kernel": "0.7.0",
    "numpy": "2.3.2",
}
CODE_FILES = (
    "__init__.py",
    "compare_eval.py",
    "compare_retrieval.py",
    "contract.py",
    "evaluate.py",
    "evaluate_retrieval.py",
    "prepare_assets.py",
    "prepare_eval_data.py",
    "prepare_official_stream.py",
    "prepare_retrieval_data.py",
    "preflight.py",
    "select_probe.py",
    "select_runtime.py",
    "train.py",
    "run_pro6000.sh",
    "SPEC.md",
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def free_bytes(path: Path) -> int:
    statistics = os.statvfs(path)
    return statistics.f_bavail * statistics.f_frsize


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def compiled_cuda_arches() -> list[str]:
    """Report compile-time CUDA architectures even when no GPU is attached."""
    get_flags = getattr(torch._C, "_cuda_getArchFlags", None)
    if get_flags is not None:
        flags = get_flags()
        if flags:
            return str(flags).split()
    return torch.cuda.get_arch_list()


def validate_dataset(
    manifest_path: Path,
    *,
    official_config: Path,
    tokenizer_path: Path,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    root = manifest_path.parent
    manifest = validate_existing_stream(
        root,
        official_config.resolve(),
        tokenizer_path=tokenizer_path.resolve(),
        steps=FIRST_GATE_STEPS,
    )
    stream = manifest["training_stream"]
    order = manifest["order"]
    if "GPT-NeoX" in json.dumps(manifest) or "gpt-neox" in json.dumps(manifest):
        raise RuntimeError("GPT-NeoX-tokenized data is forbidden for OLMo-2")
    return {
        "manifest_sha256": sha256_file(manifest_path),
        "stream_sha256": stream["sha256"],
        "first_global_batch_sha256": stream["first_global_batch_sha256"],
        "tokens": stream["tokens"],
        "instances": stream["instances"],
        "invalid_instances": stream["invalid_instances"],
        "source_manifest_sha256": manifest["source_manifest"]["sha256"],
        "configured_paths": manifest["source_manifest"]["configured_paths"],
        "unique_urls": manifest["source_manifest"]["unique_urls"],
        "order_indices_sha256": order["indices_sha256"],
        "decode_spotcheck_sha256": manifest["decode_spotcheck"]["sha256"],
        "order_stream_proof_sha256": manifest["order_stream_proof"]["sha256"],
    }


def validate_eval_dataset(manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "EVAL_DATA_VERIFIED":
        raise RuntimeError("evaluation dataset is not EVAL_DATA_VERIFIED")
    if manifest.get("tokenizer") != "allenai_dolma2":
        raise RuntimeError("evaluation tokenizer drift")
    if not manifest.get("held_out"):
        raise RuntimeError("evaluation data is not marked held-out")
    for source in manifest["sources"]:
        path = root / source["path"]
        if sha256_file(path) != source["sha256"]:
            raise RuntimeError(f"evaluation source hash drift: {path}")
    for source in manifest.get("raw_sources", []):
        path = root / source["path"]
        if path.stat().st_size != source["bytes"]:
            raise RuntimeError(f"evaluation raw-source size drift: {path}")
        if sha256_file(path) != source["sha256"]:
            raise RuntimeError(f"evaluation raw-source hash drift: {path}")
    expected_anchors = {
        "long_documents": (128, 16_384),
        "official_validation": (256, 4_096),
    }
    evidence: dict[str, Any] = {}
    for name, (rows, length) in expected_anchors.items():
        anchor = manifest["anchors"][name]
        path = root / anchor["path"]
        if anchor["rows"] != rows or anchor["length"] != length:
            raise RuntimeError(f"evaluation anchor contract drift: {name}")
        if sha256_file(path) != anchor["sha256"]:
            raise RuntimeError(f"evaluation anchor hash drift: {name}")
        metadata_path = root / anchor["metadata_path"]
        if sha256_file(metadata_path) != anchor["metadata_sha256"]:
            raise RuntimeError(f"evaluation metadata hash drift: {name}")
        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
        if len(metadata) != rows:
            raise RuntimeError(f"evaluation metadata row drift: {name}")
        if name == "long_documents":
            documents = [row["source"] for row in metadata]
            if len(set(documents)) != rows:
                raise RuntimeError(
                    "long evaluation is not document-disjoint by row"
                )
        evidence[name] = {
            "sha256": anchor["sha256"],
            "rows": rows,
            "length": length,
        }
    return {
        "manifest_sha256": sha256_file(manifest_path),
        "anchors": evidence,
    }


def validate_retrieval_dataset(
    manifest_path: Path, *, eval_manifest_path: Path
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "RETRIEVAL_DATA_VERIFIED":
        raise RuntimeError("retrieval dataset is not verified")
    if manifest.get("lengths") != [4_096, 8_192, 16_384]:
        raise RuntimeError("retrieval lengths drift")
    if manifest.get("source_fractions") != [0.1, 0.5, 0.9]:
        raise RuntimeError("retrieval source positions drift")
    if manifest.get("distractor_counts") != [0, 8]:
        raise RuntimeError("retrieval distractor densities drift")
    if manifest.get("examples_per_cell") != 4:
        raise RuntimeError("retrieval example count drift")
    if (
        manifest.get("natural_eval_manifest_sha256")
        != sha256_file(eval_manifest_path)
    ):
        raise RuntimeError("retrieval filler/evaluation manifest drift")
    expected_rows = (
        len(manifest["source_fractions"])
        * len(manifest["distractor_counts"])
        * manifest["examples_per_cell"]
    )
    expected_pairs = {
        (length, variant)
        for length in manifest["lengths"]
        for variant in ("sourced", "source_deleted", "metadata")
    }
    actual_pairs = {
        (int(row["length"]), str(row["variant"]))
        for row in manifest["arrays"]
    }
    if actual_pairs != expected_pairs or len(manifest["arrays"]) != len(
        expected_pairs
    ):
        raise RuntimeError("retrieval artifact matrix is incomplete or duplicated")
    evidence: list[dict[str, Any]] = []
    for row in manifest["arrays"]:
        path = root / row["path"]
        if int(row["rows"]) != expected_rows:
            raise RuntimeError(f"retrieval row-count drift: {path}")
        if sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"retrieval artifact hash drift: {path}")
        if row["variant"] == "metadata":
            metadata = json.loads(path.read_text(encoding="utf-8"))
            if len(metadata) != expected_rows:
                raise RuntimeError(f"retrieval metadata shape drift: {path}")
        else:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.dtype != np.uint32 or array.shape != (
                expected_rows,
                int(row["length"]),
            ):
                raise RuntimeError(f"retrieval array shape/dtype drift: {path}")
        evidence.append(
            {
                "path": row["path"],
                "length": row["length"],
                "variant": row["variant"],
                "rows": row["rows"],
                "sha256": row["sha256"],
            }
        )
    return {
        "manifest_sha256": sha256_file(manifest_path),
        "arrays": evidence,
    }


def validate_assets(asset_root: Path) -> dict[str, Any]:
    asset_root = asset_root.resolve()
    asset_manifest = json.loads(
        (asset_root / "asset_manifest.json").read_text(encoding="utf-8")
    )
    if asset_manifest.get("status") != "ASSETS_VERIFIED":
        raise RuntimeError("model assets are not ASSETS_VERIFIED")
    official_config = asset_root / asset_manifest["official_config"]["path"]
    if sha256_file(official_config) != OFFICIAL_CONFIG_SHA256:
        raise RuntimeError("official config SHA-256 drift")
    step0 = asset_root / "step0"
    geo1000 = asset_root / "geo1000"
    step0_weights = validate_weight_files(step0, STEP0_WEIGHT_FILES)
    geo_weights = validate_weight_files(geo1000, GEO1000_WEIGHT_FILES)
    config = AutoConfig.from_pretrained(step0, local_files_only=True)
    assert_model_config(config)
    tokenizer = AutoTokenizer.from_pretrained(step0, local_files_only=True)
    if tokenizer.eos_token_id != TOKENIZER_MARKERS["eos_token_id"]:
        raise RuntimeError("tokenizer EOS drift")
    if tokenizer.pad_token_id != TOKENIZER_MARKERS["pad_token_id"]:
        raise RuntimeError("tokenizer pad token drift")
    return {
        "asset_manifest_sha256": sha256_file(asset_root / "asset_manifest.json"),
        "model_id": MODEL_ID,
        "step0_revision": STEP0_REVISION,
        "geo1000_revision": GEO1000_REVISION,
        "official_config_sha256": OFFICIAL_CONFIG_SHA256,
        "step0_weights": step0_weights,
        "geo1000_weights": geo_weights,
        "tokenizer_sha256": sha256_file(step0 / "tokenizer.json"),
    }


def validate_model_intervention(step0: Path) -> dict[str, Any]:
    model = AutoModelForCausalLM.from_pretrained(
        step0,
        local_files_only=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    assert_model_config(model.config)
    count = trainable_parameter_count(model)
    if count != ACTUAL_PARAMETER_COUNT:
        raise RuntimeError(f"parameter count {count} != {ACTUAL_PARAMETER_COUNT}")
    metadata_before = named_parameter_metadata(model)
    digest_before = parameter_identity_digest(model)
    native = model.model.rotary_emb.inv_freq.detach().clone()
    frequency = patch_endpoint_evq(model)
    digest_after = parameter_identity_digest(model)
    metadata_after = named_parameter_metadata(model)
    if digest_before != digest_after or metadata_before != metadata_after:
        raise RuntimeError("EVQ intervention changed trainable parameters")
    if torch.equal(native, model.model.rotary_emb.inv_freq):
        raise RuntimeError("EVQ intervention did not change inv_freq")
    state_keys = set(model.state_dict().keys())
    if any("inv_freq" in key for key in state_keys):
        raise RuntimeError("inv_freq unexpectedly became persistent model state")
    return {
        "trainable_parameters": count,
        "parameter_sha256_before": digest_before,
        "parameter_sha256_after": digest_after,
        "only_intervention": "model.model.rotary_emb.inv_freq",
        "frequency": frequency,
    }


def validate_code(package_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for name in CODE_FILES:
        path = package_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(
            {
                "path": name,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {"files": rows, "code_sha256": sha256_json(rows)}


PORTABLE_BOUND_CHECKS = {
    "code": ("code_sha256",),
    "assets": ("asset_manifest_sha256",),
    "dataset": (
        "manifest_sha256",
        "stream_sha256",
        "first_global_batch_sha256",
    ),
    "evaluation_dataset": ("manifest_sha256",),
    "retrieval_dataset": ("manifest_sha256",),
}


def reuse_portable_model_intervention(
    portable_receipt_path: Path,
    checks: dict[str, Any],
) -> dict[str, Any]:
    """Reuse only the RAM-heavy model check, bound to revalidated artifacts."""
    portable_receipt_path = portable_receipt_path.resolve()
    portable = json.loads(portable_receipt_path.read_text(encoding="utf-8"))
    if portable.get("status") != "PORTABLE_PREFLIGHT_PASS":
        raise RuntimeError("portable preflight is not PASS")
    portable_checks = portable.get("checks", {})
    for name, fields in PORTABLE_BOUND_CHECKS.items():
        current_check = checks.get(name, {})
        portable_check = portable_checks.get(name, {})
        if current_check.get("status") != "PASS":
            raise RuntimeError(f"current {name} check is not PASS")
        if portable_check.get("status") != "PASS":
            raise RuntimeError(f"portable {name} check is not PASS")
        current_evidence = current_check["evidence"]
        portable_evidence = portable_check["evidence"]
        for field in fields:
            if current_evidence.get(field) != portable_evidence.get(field):
                raise RuntimeError(
                    f"portable {name}.{field} does not match current artifact"
                )
    current_frequency = checks.get("frequency_formula", {})
    portable_frequency = portable_checks.get("frequency_formula", {})
    if (
        current_frequency.get("status") != "PASS"
        or portable_frequency.get("status") != "PASS"
        or current_frequency.get("evidence") != portable_frequency.get("evidence")
    ):
        raise RuntimeError("portable frequency formula does not match")
    intervention = portable_checks.get("model_intervention", {})
    if intervention.get("status") != "PASS":
        raise RuntimeError("portable model intervention check is not PASS")
    evidence = dict(intervention["evidence"])
    if (
        evidence.get("parameter_sha256_before")
        != evidence.get("parameter_sha256_after")
    ):
        raise RuntimeError("portable intervention changed trainable parameters")
    evidence["evidence_mode"] = "REUSED_PORTABLE_HASH_BOUND"
    evidence["portable_receipt"] = str(portable_receipt_path)
    evidence["portable_receipt_sha256"] = sha256_file(portable_receipt_path)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--retrieval-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--compile-cache", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--gpu-command", required=True)
    parser.add_argument(
        "--portable-only",
        action="store_true",
        help="run the RAM-heavy model check but skip host-specific storage",
    )
    parser.add_argument(
        "--portable-receipt",
        type=Path,
        help="reuse a hash-bound model check from a high-memory no-GPU host",
    )
    args = parser.parse_args()
    if args.portable_only and args.portable_receipt is not None:
        parser.error("--portable-only and --portable-receipt are mutually exclusive")

    started = time.time()
    package_dir = Path(__file__).resolve().parent
    checks: dict[str, Any] = {}
    failures: list[str] = []

    def run_check(name: str, function: Any) -> None:
        try:
            checks[name] = {"status": "PASS", "evidence": function()}
        except Exception as error:
            checks[name] = {
                "status": "FAIL",
                "error": f"{type(error).__name__}: {error}",
            }
            failures.append(name)

    run_check("code", lambda: validate_code(package_dir))
    run_check("assets", lambda: validate_assets(args.asset_root))
    run_check(
        "dataset",
        lambda: validate_dataset(
            args.data_manifest,
            official_config=args.asset_root / "upstream" / "OLMo2-1B-stage1.yaml",
            tokenizer_path=args.asset_root / "step0",
        ),
    )
    run_check(
        "evaluation_dataset",
        lambda: validate_eval_dataset(args.eval_manifest),
    )
    run_check(
        "retrieval_dataset",
        lambda: validate_retrieval_dataset(
            args.retrieval_manifest,
            eval_manifest_path=args.eval_manifest,
        ),
    )
    run_check("frequency_formula", assert_frequency_contract)
    portable_reuse = args.portable_receipt is not None
    artifact_checks_pass = all(
        checks[name]["status"] == "PASS"
        for name in PORTABLE_BOUND_CHECKS
    ) and checks["frequency_formula"]["status"] == "PASS"
    if portable_reuse and artifact_checks_pass:
        run_check(
            "model_intervention",
            lambda: reuse_portable_model_intervention(
                args.portable_receipt,
                checks,
            ),
        )
    elif portable_reuse:
        failures.append("model_intervention")
        checks["model_intervention"] = {
            "status": "SKIP",
            "error": "current artifact validation failed",
        }
    elif checks["assets"]["status"] == "PASS":
        run_check(
            "model_intervention",
            lambda: validate_model_intervention(args.asset_root / "step0"),
        )
    else:
        failures.append("model_intervention")
        checks["model_intervention"] = {
            "status": "SKIP",
            "error": "asset validation failed",
        }

    def storage_check() -> dict[str, Any]:
        args.checkpoint_root.mkdir(parents=True, exist_ok=True)
        args.compile_cache.mkdir(parents=True, exist_ok=True)
        if args.train_output.exists() and any(args.train_output.iterdir()):
            raise RuntimeError(f"train output is non-empty: {args.train_output}")
        available = free_bytes(args.checkpoint_root)
        if available < MIN_CHECKPOINT_FREE_BYTES:
            raise RuntimeError(
                f"only {available/1024**3:.1f} GiB free; "
                f"need at least {MIN_CHECKPOINT_FREE_BYTES/1024**3:.1f} GiB"
            )
        return {
            "checkpoint_free_bytes": available,
            "minimum_free_bytes": MIN_CHECKPOINT_FREE_BYTES,
            "compile_cache": str(args.compile_cache.resolve()),
            "train_output": str(args.train_output.resolve()),
        }

    if args.portable_only:
        checks["storage"] = {
            "status": "SKIP",
            "evidence": {
                "reason": "host-specific storage is checked on the target host"
            },
        }
    else:
        run_check("storage", storage_check)

    def environment_check() -> dict[str, Any]:
        if torch.cuda.is_available():
            raise RuntimeError(
                "CPU preflight must run with CUDA_VISIBLE_DEVICES empty"
            )
        compiled_arches = compiled_cuda_arches()
        if "sm_120" not in compiled_arches:
            raise RuntimeError(
                f"PyTorch build lacks Blackwell SM120: {compiled_arches}"
            )
        actual_versions = {
            name: package_version(name) for name in ENVIRONMENT_CONTRACT
        }
        if actual_versions != ENVIRONMENT_CONTRACT:
            raise RuntimeError(
                "environment version drift: "
                f"expected={ENVIRONMENT_CONTRACT}, actual={actual_versions}"
            )
        return {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "compiled_cuda_arches": compiled_arches,
            "packages": actual_versions,
            "cuda_available": torch.cuda.is_available(),
        }

    run_check("environment", environment_check)

    pass_status = (
        "PORTABLE_PREFLIGHT_PASS"
        if args.portable_only
        else "CPU_PREFLIGHT_PASS"
    )
    receipt = {
        "status": pass_status if not failures else "NOT_READY",
        "receipt_mode": (
            "portable_source"
            if args.portable_only
            else (
                "target_with_portable_model_evidence"
                if portable_reuse
                else "target_full"
            )
        ),
        "started_unix": started,
        "finished_unix": time.time(),
        "checks": checks,
        "failures": sorted(set(failures)),
        "gpu_launch_command": args.gpu_command,
        "gpu_checks_pending": [
            "RTX Pro 6000 identity and compute capability",
            "Flash-only SDPA kernel eligibility",
            "native-vs-Liger loss and gradient parity",
            "compile latency and graph-break-free fullgraph capture",
            "finite first loss, peak VRAM, sustained tokens/s, and ETA",
        ],
    }
    write_json(args.receipt.resolve(), receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
