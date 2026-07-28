#!/usr/bin/env python3
"""Freeze the matched downstream evaluation matrix for fresh general Q/K LoRA."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from rebuttal.rebuttal_0723.experiments.small_model_lora_conversion import (
    atomic_json,
    sha256_file,
)

from .audit_general_qk_downstream_overlap import STATUS as OVERLAP_STATUS
from .train_4k_general_qk import RESULT_STATUS
from .train_4k_stage_a import ready_checkpoint_digest


STATUS = "OLMO2_GENERAL_QK_DOWNSTREAM_EVAL_READY_V1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-ready-receipt", type=Path, required=True)
    parser.add_argument("--native-run", type=Path, required=True)
    parser.add_argument("--evq-run", type=Path, required=True)
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument("--background-dir", type=Path, required=True)
    parser.add_argument("--overlap-receipt", type=Path, required=True)
    parser.add_argument("--two-wiki-data", type=Path, required=True)
    parser.add_argument("--ruler-data", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--minimum-free-bytes", type=int, default=8_000_000_000)
    return parser.parse_args()


def load_result(path: Path, frequency: str) -> dict[str, Any]:
    result_path = path.resolve() / "results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        result.get("status") != RESULT_STATUS
        or result.get("protocol", {}).get("frequency") != frequency
        or result.get("protocol", {}).get("adaptation") != "qk_answer"
        or result.get("protocol", {}).get("trainable_scope")
        != "fresh_qk_lora_only"
    ):
        raise RuntimeError(f"general-QK result identity drift: {path}")
    adapter = path.resolve() / "adapter.pt"
    if (
        not adapter.is_file()
        or result.get("adapter_sha256") != sha256_file(adapter)
    ):
        raise RuntimeError(f"general-QK adapter hash drift: {path}")
    return result


def protocol_without_frequency(result: dict[str, Any]) -> dict[str, Any]:
    value = dict(result["protocol"])
    value.pop("frequency", None)
    return value


def verify_manifest_files(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, entry in manifest["files"].items():
        path = root / name
        if (
            not path.is_file()
            or path.stat().st_size != int(entry["bytes"])
            or sha256_file(path) != entry["sha256"]
        ):
            raise RuntimeError(f"registered data file hash drift: {path}")
    return {
        "path": str(root),
        "manifest_sha256": sha256_file(manifest_path),
        "files": manifest["files"],
    }


def job(
    *,
    benchmark: str,
    name: str,
    frequency: str,
    adapter: str | None,
    lengths: list[int],
    factor: float | None = None,
) -> dict[str, Any]:
    return {
        "benchmark": benchmark,
        "name": name,
        "frequency": frequency,
        "adapter_role": adapter,
        "adaptation": "qk_answer" if adapter is not None else None,
        "lengths": lengths,
        "yarn_factor": factor,
        "limit_per_cell": 20 if benchmark == "ruler" else 200,
        "fill_to_budget": benchmark == "2wiki",
    }


def evaluation_jobs() -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for benchmark in ("ruler", "2wiki"):
        jobs.extend(
            [
                job(
                    benchmark=benchmark,
                    name=f"{benchmark}_base_native_4k",
                    frequency="native",
                    adapter=None,
                    lengths=[4_096],
                ),
                job(
                    benchmark=benchmark,
                    name=f"{benchmark}_general_qk_native_raw",
                    frequency="native",
                    adapter="native",
                    lengths=[4_096, 8_192, 16_384],
                ),
                job(
                    benchmark=benchmark,
                    name=f"{benchmark}_general_qk_evq_raw",
                    frequency="evq",
                    adapter="evq",
                    lengths=[4_096, 8_192, 16_384],
                ),
            ]
        )
        for label, native_frequency, evq_frequency in (
            (
                "official_yarn",
                "official_yarn",
                "evq_official_yarn",
            ),
            (
                "repo_fixed_ramp",
                "repo_fixed_ramp",
                "evq_repo_fixed_ramp",
            ),
        ):
            for factor, target in ((2.0, 8_192), (4.0, 16_384)):
                jobs.extend(
                    [
                        job(
                            benchmark=benchmark,
                            name=(
                                f"{benchmark}_general_qk_native_{label}"
                                f"_f{int(factor)}"
                            ),
                            frequency=native_frequency,
                            adapter="native",
                            lengths=[4_096, target],
                            factor=factor,
                        ),
                        job(
                            benchmark=benchmark,
                            name=(
                                f"{benchmark}_general_qk_evq_{label}"
                                f"_f{int(factor)}"
                            ),
                            frequency=evq_frequency,
                            adapter="evq",
                            lengths=[4_096, target],
                            factor=factor,
                        ),
                    ]
                )
    return jobs


def main() -> None:
    args = parse_args()
    receipt = args.receipt.resolve()
    output_root = args.output_root.resolve()
    if receipt.exists() or output_root.exists():
        raise FileExistsError(receipt if receipt.exists() else output_root)

    checkpoint = args.checkpoint.resolve()
    checkpoint_digest = ready_checkpoint_digest(
        checkpoint, args.checkpoint_ready_receipt.resolve()
    )
    native = load_result(args.native_run, "native")
    evq = load_result(args.evq_run, "evq")
    prepared = args.prepared_data.resolve()
    longalign = verify_manifest_files(
        prepared / "longalign_paired_L4096"
    )
    tulu = verify_manifest_files(prepared / "tulu3_replay_L4096")
    background = verify_manifest_files(args.background_dir)
    if (
        native["checkpoint_sha256"] != checkpoint_digest
        or evq["checkpoint_sha256"] != checkpoint_digest
        or protocol_without_frequency(native)
        != protocol_without_frequency(evq)
        or native["training"]["initial_adapter_sha256"]
        != evq["training"]["initial_adapter_sha256"]
        or native["training"]["selection_sha256"]
        != evq["training"]["selection_sha256"]
    ):
        raise RuntimeError("Native/EVQ general-QK matched contract drift")
    for result in (native, evq):
        if (
            result["inputs"]["longalign"]["manifest_sha256"]
            != longalign["manifest_sha256"]
            or result["inputs"]["tulu"]["manifest_sha256"]
            != tulu["manifest_sha256"]
            or result["inputs"]["background_manifest_sha256"]
            != background["manifest_sha256"]
        ):
            raise RuntimeError("general-QK executed data receipt drift")

    overlap_path = args.overlap_receipt.resolve()
    overlap = json.loads(overlap_path.read_text(encoding="utf-8"))
    if (
        overlap.get("status") != OVERLAP_STATUS
        or overlap.get("script_sha256")
        != sha256_file(
            Path(__file__).resolve().parent
            / "audit_general_qk_downstream_overlap.py"
        )
        or overlap.get("tokenizer_sha256")
        != sha256_file(checkpoint / "tokenizer.json")
        or overlap.get("training_views", {})
        .get("longalign", {})
        .get("manifest_sha256")
        != longalign["manifest_sha256"]
        or overlap.get("training_views", {})
        .get("tulu", {})
        .get("manifest_sha256")
        != tulu["manifest_sha256"]
        or overlap.get("two_wiki", {}).get("exact_question_matches") != 0
        or overlap.get("two_wiki", {}).get(
            "evaluation_rows_sha256"
        )
        != sha256_file(
            args.two_wiki_data.resolve() / "evaluation_rows.jsonl"
        )
        or overlap.get("ruler", {}).get(
            "exact_raw_or_chat_prompt_matches"
        )
        != 0
        or overlap.get("ruler", {}).get("manifest_sha256")
        != sha256_file(args.ruler_data.resolve() / "manifest.json")
    ):
        raise RuntimeError("downstream exact-overlap gate did not pass")

    two_wiki_manifest = args.two_wiki_data.resolve() / "manifest.json"
    ruler_manifest = args.ruler_data.resolve() / "manifest.json"
    jobs = evaluation_jobs()
    if len({value["name"] for value in jobs}) != len(jobs):
        raise RuntimeError("downstream evaluation job names are not unique")
    free_bytes = shutil.disk_usage(output_root.parent).free
    if free_bytes < int(args.minimum_free_bytes):
        raise RuntimeError("insufficient free space for evaluation matrix")

    code_root = Path(__file__).resolve().parent
    result = {
        "status": STATUS,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_digest,
        "matched_training_gate": {
            "protocol_without_frequency": protocol_without_frequency(native),
            "initial_adapter_sha256": native["training"][
                "initial_adapter_sha256"
            ],
            "selection_sha256": native["training"]["selection_sha256"],
            "native_adapter_sha256": native["adapter_sha256"],
            "evq_adapter_sha256": evq["adapter_sha256"],
        },
        "inputs": {
            "checkpoint_ready_receipt_sha256": sha256_file(
                args.checkpoint_ready_receipt.resolve()
            ),
            "native_results_sha256": sha256_file(
                args.native_run.resolve() / "results.json"
            ),
            "evq_results_sha256": sha256_file(
                args.evq_run.resolve() / "results.json"
            ),
            "overlap_receipt_sha256": sha256_file(overlap_path),
            "two_wiki_manifest_sha256": sha256_file(two_wiki_manifest),
            "ruler_manifest_sha256": sha256_file(ruler_manifest),
        },
        "registered_paths": {
            "native_run": str(args.native_run.resolve()),
            "evq_run": str(args.evq_run.resolve()),
            "prepared_data": str(prepared),
            "background_dir": str(args.background_dir.resolve()),
            "overlap_receipt": str(overlap_path),
            "two_wiki_data": str(args.two_wiki_data.resolve()),
            "ruler_data": str(args.ruler_data.resolve()),
        },
        "bound_code_sha256": {
            "preflight": sha256_file(Path(__file__).resolve()),
            "launcher": sha256_file(
                code_root / "run_general_qk_downstream_eval.py"
            ),
            "overlap_audit": sha256_file(
                code_root / "audit_general_qk_downstream_overlap.py"
            ),
            "ruler_evaluator": sha256_file(
                code_root / "evaluate_instruct_ruler_transfer.py"
            ),
            "two_wiki_evaluator": sha256_file(
                code_root / "evaluate_2wiki_phase_adaptation.py"
            ),
        },
        "output_root": str(output_root),
        "jobs": jobs,
        "execution": {
            "parallel_streams": ["ruler", "2wiki"],
            "maximum_concurrent_gpu_processes": 2,
            "stop_condition": (
                "stop on identity/hash failure, evaluator-contract drift, "
                "non-finite output, CUDA OOM, or insufficient disk"
            ),
        },
        "storage": {
            "free_bytes": int(free_bytes),
            "minimum_free_bytes": int(args.minimum_free_bytes),
        },
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(receipt, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
