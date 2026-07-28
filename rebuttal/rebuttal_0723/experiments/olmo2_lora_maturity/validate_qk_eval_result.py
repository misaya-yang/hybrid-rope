#!/usr/bin/env python3
"""Strict identity and coverage gate for OLMo-2 Q/K evaluation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


TWO_WIKI_STATUS = "OLMO2_2WIKI_PHASE_EVALUATION_COMPLETE_V1"
RULER_STATUS = "OLMO2_INSTRUCT_RULER_TRANSFER_COMPLETE"
RULER_TASKS = (
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--benchmark", choices=("2wiki", "ruler"), required=True
    )
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument(
        "--adapter-sha256",
        required=True,
        help="Use the literal value 'none' for an untouched base result.",
    )
    parser.add_argument(
        "--adapter-training-frequency",
        choices=("native", "evq"),
        required=True,
    )
    parser.add_argument("--frequency", required=True)
    parser.add_argument("--lengths", type=int, nargs="+", required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--data-manifest-sha256", required=True)
    parser.add_argument("--evaluator-sha256", required=True)
    parser.add_argument("--adaptation", default="qk_answer")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=128.0)
    parser.add_argument("--expected-role")
    parser.add_argument("--fill-to-budget", action="store_true")
    parser.add_argument("--yarn-factor", type=float)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def require_equal(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise RuntimeError(
            f"{name} drift: actual={actual!r}, expected={expected!r}"
        )


def require_probability(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise RuntimeError(f"invalid probability {name}: {number}")
    return number


def validate_common(
    args: argparse.Namespace,
    root: Path,
    manifest: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, str]:
    examples_path = root / "examples.jsonl"
    manifest_path = root / "run_manifest.json"
    result_path = root / "results.json"
    for path in (examples_path, manifest_path, result_path):
        if not path.is_file():
            raise RuntimeError(f"missing evaluation artifact: {path}")

    expected_status = (
        TWO_WIKI_STATUS if args.benchmark == "2wiki" else RULER_STATUS
    )
    require_equal(result.get("status"), expected_status, "result status")
    require_equal(
        result.get("checkpoint_sha256"),
        args.checkpoint_sha256,
        "result checkpoint",
    )
    require_equal(
        result.get("script_sha256"),
        args.evaluator_sha256,
        "result evaluator",
    )
    require_equal(
        result.get("run_manifest_sha256"),
        sha256_file(manifest_path),
        "run-manifest hash",
    )
    require_equal(
        result.get("results", {}).get("examples_sha256"),
        sha256_file(examples_path),
        "examples hash",
    )
    require_equal(
        result.get("data", {}).get("manifest_sha256"),
        args.data_manifest_sha256,
        "result data manifest",
    )
    if not result.get("frequency", {}).get("active_frequency"):
        raise RuntimeError("missing realized active frequency")
    if not result.get("frequency", {}).get("active_sha256_float32"):
        raise RuntimeError("missing realized frequency hash")

    require_equal(
        manifest.get("checkpoint_sha256"),
        args.checkpoint_sha256,
        "manifest checkpoint",
    )
    require_equal(
        manifest.get("data_manifest_sha256"),
        args.data_manifest_sha256,
        "manifest data",
    )
    require_equal(
        manifest.get("frequency"), args.frequency, "manifest frequency"
    )
    no_adapter = args.adapter_sha256 == "none"
    require_equal(
        manifest.get("adapter_sha256"),
        None if no_adapter else args.adapter_sha256,
        "manifest adapter",
    )
    if no_adapter:
        require_equal(
            manifest.get("adaptation"), None, "manifest adaptation"
        )
    else:
        require_equal(
            manifest.get("adaptation"),
            args.adaptation,
            "manifest adaptation",
        )
        require_equal(manifest.get("rank"), args.rank, "manifest rank")
        require_equal(
            float(manifest.get("alpha")), args.alpha, "manifest alpha"
        )
    require_equal(
        manifest.get("yarn_factor"),
        args.yarn_factor,
        "manifest YaRN factor",
    )

    adapter = result.get("adapter")
    if no_adapter:
        if adapter is not None:
            if not isinstance(adapter, dict) or any(
                adapter.get(name) is not None
                for name in ("metadata", "path", "sha256")
            ):
                raise RuntimeError(
                    f"base result adapter drift: {adapter!r}"
                )
    else:
        if not isinstance(adapter, dict):
            raise RuntimeError("missing adapter receipt")
        require_equal(
            adapter.get("sha256"), args.adapter_sha256, "result adapter"
        )
        metadata = adapter.get("metadata")
        if not isinstance(metadata, dict):
            raise RuntimeError("missing adapter metadata")
        require_equal(
            metadata.get("base_checkpoint_sha256"),
            args.checkpoint_sha256,
            "adapter base checkpoint",
        )
        require_equal(
            metadata.get("frequency"),
            args.adapter_training_frequency,
            "adapter training frequency",
        )
        require_equal(
            metadata.get("adaptation"),
            args.adaptation,
            "adapter adaptation",
        )
        require_equal(metadata.get("rank"), args.rank, "adapter rank")
        require_equal(
            float(metadata.get("alpha")), args.alpha, "adapter alpha"
        )

    return {
        "results_sha256": sha256_file(result_path),
        "run_manifest_sha256": sha256_file(manifest_path),
        "examples_sha256": sha256_file(examples_path),
        "adapter_sha256": None if no_adapter else args.adapter_sha256,
        "active_frequency": str(
            result["frequency"]["active_frequency"]
        ),
        "active_frequency_sha256_float32": str(
            result["frequency"]["active_sha256_float32"]
        ),
    }


def validate_two_wiki(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    lengths = [int(value) for value in args.lengths]
    protocol = result.get("protocol", {})
    if args.adapter_sha256 != "none":
        require_equal(
            protocol.get("adaptation"), args.adaptation, "adaptation"
        )
    require_equal(protocol.get("budgets"), lengths, "2Wiki budgets")
    require_equal(protocol.get("limit"), args.limit, "2Wiki limit")
    require_equal(
        protocol.get("fill_to_budget"),
        args.fill_to_budget,
        "2Wiki fill mode",
    )
    require_equal(protocol.get("greedy"), True, "2Wiki decoding")
    require_equal(manifest.get("budgets"), lengths, "manifest budgets")
    require_equal(manifest.get("limit"), args.limit, "manifest limit")
    require_equal(
        manifest.get("fill_to_budget"),
        args.fill_to_budget,
        "manifest fill mode",
    )
    if args.expected_role is not None:
        require_equal(
            manifest.get("role"), args.expected_role, "2Wiki role"
        )

    cells = result.get("results", {}).get("cells", {})
    require_equal(
        set(cells), {str(value) for value in lengths}, "2Wiki cells"
    )
    metrics: dict[str, Any] = {}
    for length in lengths:
        cell = cells[str(length)]
        require_equal(
            cell.get("examples"), args.limit, f"2Wiki L{length} rows"
        )
        metrics[str(length)] = {
            "mean_token_f1": require_probability(
                cell.get("mean_token_f1"), f"2Wiki L{length} F1"
            ),
            "normalized_exact": require_probability(
                cell.get("normalized_exact"), f"2Wiki L{length} exact"
            ),
            "terminal_eos": require_probability(
                cell.get("terminal_eos"), f"2Wiki L{length} EOS"
            ),
        }
    require_equal(
        result["results"].get("examples"),
        args.limit * len(lengths),
        "2Wiki total examples",
    )
    return metrics


def validate_ruler(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    lengths = [int(value) for value in args.lengths]
    protocol = result.get("protocol", {})
    if args.adapter_sha256 != "none":
        require_equal(
            protocol.get("adaptation"), args.adaptation, "adaptation"
        )
    require_equal(protocol.get("lengths"), lengths, "RULER lengths")
    require_equal(
        protocol.get("limit_per_cell"), args.limit, "RULER limit"
    )
    require_equal(
        tuple(protocol.get("tasks", ())), RULER_TASKS, "RULER tasks"
    )
    require_equal(protocol.get("greedy"), True, "RULER decoding")
    require_equal(manifest.get("lengths"), lengths, "manifest lengths")
    require_equal(
        manifest.get("limit_per_cell"), args.limit, "manifest limit"
    )
    require_equal(
        tuple(manifest.get("tasks", ())), RULER_TASKS, "manifest tasks"
    )

    cells = result.get("results", {}).get("cells", {})
    require_equal(set(cells), set(RULER_TASKS), "RULER cell tasks")
    by_length = {str(length): [] for length in lengths}
    for task in RULER_TASKS:
        task_cells = cells[task]
        require_equal(
            set(task_cells),
            {str(value) for value in lengths},
            f"RULER {task} lengths",
        )
        for length in lengths:
            cell = task_cells[str(length)]
            require_equal(
                cell.get("examples"),
                args.limit,
                f"RULER {task} L{length} rows",
            )
            by_length[str(length)].append(
                require_probability(
                    cell.get("official_task_score"),
                    f"RULER {task} L{length}",
                )
            )
    require_equal(
        result["results"].get("examples"),
        args.limit * len(lengths) * len(RULER_TASKS),
        "RULER total examples",
    )
    return {
        length: sum(values) / len(values)
        for length, values in by_length.items()
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    manifest = load_json(root / "run_manifest.json")
    result = load_json(root / "results.json")
    artifacts = validate_common(args, root, manifest, result)
    metrics = (
        validate_two_wiki(args, manifest, result)
        if args.benchmark == "2wiki"
        else validate_ruler(args, manifest, result)
    )
    print(
        json.dumps(
            {
                "status": "OLMO2_QK_EVAL_RESULT_VALID_V1",
                "benchmark": args.benchmark,
                "root": str(root),
                "artifacts": artifacts,
                "metrics": metrics,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
