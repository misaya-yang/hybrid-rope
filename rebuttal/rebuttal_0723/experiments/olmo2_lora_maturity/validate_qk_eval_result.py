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
EXPECTED_ACTIVE_FREQUENCY = {
    "native": "native_endpoint_rope",
    "evq": "evq_endpoint_cosh",
    "official_yarn": "official_transformers_yarn",
    "evq_official_yarn": (
        "evq_endpoint_cosh_plus_official_transformers_yarn"
    ),
    "repo_fixed_ramp": "native_plus_repo_fixed_index_smooth_ramp",
    "evq_repo_fixed_ramp": "evq_plus_repo_fixed_index_smooth_ramp",
}
EXPECTED_ACTIVE_FREQUENCY_SHA256 = {
    ("native", None): (
        "dde15c31724177356ae954d6e11fb337e6fccef56e4520a905cac3f0d9885b34"
    ),
    ("evq", None): (
        "917a52426b4ac986545c8ec73b115daae3c6515d6b9047f09d30c972ea1a4607"
    ),
    ("official_yarn", 2.0): (
        "8accc312855e440d24c9a3542a1cbff45a64774460c7aa7fd8513dc26c333039"
    ),
    ("evq_official_yarn", 2.0): (
        "45c4484495368a8dfc5b4fcc6f6efad446e004782a0625897457c25dad006d01"
    ),
    ("repo_fixed_ramp", 2.0): (
        "1355e594f8e72953c5ee73ac78df5a7779c239c8b7f4273cd2ab05c7e35c6bbf"
    ),
    ("evq_repo_fixed_ramp", 2.0): (
        "d11ddab909667b882ef59c465ed1a70bb98e25fa278635f1f7067ba3ffa9ed0d"
    ),
}


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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RuntimeError(
                f"expected JSON object at {path}:{line_number}"
            )
        rows.append(value)
    return rows


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


def require_sha256(value: Any, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise RuntimeError(f"invalid SHA-256 {name}: {text!r}")
    return text


def current_ruler_bound_code_sha256() -> dict[str, str]:
    maturity_root = Path(__file__).resolve().parent
    experiments_root = maturity_root.parent
    paths = {
        "evaluator": maturity_root / "evaluate_instruct_ruler_transfer.py",
        "screen_frequency": (
            maturity_root / "evaluate_instruct_ruler_screen.py"
        ),
        "hybrid_import_dependency": maturity_root / "olmo2_exact_method.py",
        "data_contract": (
            maturity_root / "prepare_instruct_ruler_transfer.py"
        ),
        "checkpoint_contract": maturity_root / "train_4k_stage_a.py",
        "frequency_application": maturity_root / "train_screen.py",
        "greedy_generation_and_row_identity": (
            experiments_root / "olmo2_1b_evq" / "evaluate_ruler.py"
        ),
        "evq_contract": experiments_root / "olmo2_1b_evq" / "contract.py",
        "lora_conversion": experiments_root / "olmo2_lora_conversion.py",
        "adapter_loader": (
            experiments_root / "olmo2_lora_ood_factorial.py"
        ),
    }
    return {
        name: sha256_file(path)
        for name, path in sorted(paths.items())
    }


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
    frequency = result.get("frequency", {})
    require_equal(
        frequency.get("active_frequency"),
        EXPECTED_ACTIVE_FREQUENCY[args.frequency],
        "realized active frequency",
    )
    active_frequency_sha = require_sha256(
        frequency.get("active_sha256_float32"),
        "realized frequency tensor",
    )
    expected_frequency_key = (args.frequency, args.yarn_factor)
    if expected_frequency_key not in EXPECTED_ACTIVE_FREQUENCY_SHA256:
        raise RuntimeError(
            "no independent realized-frequency anchor for "
            f"{expected_frequency_key!r}"
        )
    require_equal(
        active_frequency_sha,
        EXPECTED_ACTIVE_FREQUENCY_SHA256[expected_frequency_key],
        "independently anchored realized frequency tensor",
    )

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
        require_sha256(
            metadata.get("frequency_sha256_float32"),
            "adapter training frequency tensor",
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
        require_equal(
            metadata.get("training_sequence_length"),
            4096,
            "adapter training sequence length",
        )
        if args.frequency in {"native", "evq"}:
            require_equal(
                active_frequency_sha,
                metadata.get("frequency_sha256_float32"),
                "active/training frequency tensor",
            )

    if args.benchmark == "2wiki":
        current_evaluator_sha = sha256_file(
            Path(__file__).resolve().parent
            / "evaluate_2wiki_phase_adaptation.py"
        )
        require_equal(
            args.evaluator_sha256,
            current_evaluator_sha,
            "READY/current 2Wiki evaluator",
        )
        current_frequency_helper_sha = sha256_file(
            Path(__file__).resolve().parent
            / "evaluate_instruct_ruler_transfer.py"
        )
        require_equal(
            result.get("frequency_helper_sha256"),
            manifest.get("frequency_helper_sha256"),
            "2Wiki frequency-helper dependency",
        )
        frequency_helper_sha = require_sha256(
            result.get("frequency_helper_sha256"),
            "2Wiki frequency-helper dependency",
        )
        require_equal(
            frequency_helper_sha,
            current_frequency_helper_sha,
            "current 2Wiki frequency-helper dependency",
        )
    else:
        current_bound_code = current_ruler_bound_code_sha256()
        require_equal(
            args.evaluator_sha256,
            current_bound_code["evaluator"],
            "READY/current RULER evaluator",
        )
        bound_code = result.get("bound_code_sha256")
        if not isinstance(bound_code, dict) or not bound_code:
            raise RuntimeError("missing RULER bound-code receipt")
        require_equal(
            bound_code,
            manifest.get("bound_code_sha256"),
            "RULER bound-code dependencies",
        )
        for name, digest in bound_code.items():
            require_sha256(digest, f"RULER dependency {name}")
        require_equal(
            bound_code,
            current_bound_code,
            "current RULER bound-code dependencies",
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
    root: Path,
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
    raw_rows = load_jsonl(root / "examples.jsonl")
    raw_by_cell: dict[int, list[dict[str, Any]]] = {
        length: [] for length in lengths
    }
    observed_keys: set[tuple[int, int]] = set()
    for row in raw_rows:
        budget = int(row.get("budget"))
        local_index = int(row.get("local_index"))
        key = (budget, local_index)
        if key in observed_keys:
            raise RuntimeError(f"duplicate 2Wiki raw row: {key}")
        observed_keys.add(key)
        if budget not in raw_by_cell or not 0 <= local_index < args.limit:
            raise RuntimeError(f"2Wiki raw row escaped matrix: {key}")
        if args.expected_role is not None:
            require_equal(
                row.get("role"), args.expected_role, f"2Wiki role {key}"
            )
        truncation = row.get("truncation")
        if not isinstance(truncation, dict):
            raise RuntimeError(f"missing 2Wiki truncation receipt: {key}")
        chat_tokens = int(truncation.get("chat_input_tokens"))
        target_tokens = int(budget) - int(manifest.get("max_new_tokens"))
        if not 0 <= target_tokens - chat_tokens <= 2:
            raise RuntimeError(
                f"2Wiki physical budget drift {key}: "
                f"chat={chat_tokens}, target={target_tokens}"
            )
        raw_by_cell[budget].append(row)
    require_equal(
        len(raw_rows),
        args.limit * len(lengths),
        "2Wiki raw row count",
    )
    for length in lengths:
        cell = cells[str(length)]
        require_equal(
            cell.get("examples"), args.limit, f"2Wiki L{length} rows"
        )
        raw_cell = raw_by_cell[length]
        require_equal(
            len(raw_cell), args.limit, f"2Wiki raw L{length} rows"
        )
        cell_metrics = {
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
        for name, raw_name in (
            ("mean_token_f1", "token_f1"),
            ("normalized_exact", "normalized_exact"),
            ("terminal_eos", "terminal_eos"),
        ):
            raw_mean = sum(
                require_probability(
                    row.get(raw_name), f"2Wiki raw L{length} {raw_name}"
                )
                for row in raw_cell
            ) / len(raw_cell)
            if not math.isclose(
                cell_metrics[name], raw_mean, rel_tol=0.0, abs_tol=1e-12
            ):
                raise RuntimeError(
                    f"2Wiki L{length} {name} aggregate drift: "
                    f"{cell_metrics[name]} != {raw_mean}"
                )
        metrics[str(length)] = cell_metrics
    require_equal(
        result["results"].get("examples"),
        args.limit * len(lengths),
        "2Wiki total examples",
    )
    return metrics


def validate_ruler(
    args: argparse.Namespace,
    root: Path,
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
    raw_rows = load_jsonl(root / "examples.jsonl")
    raw_by_cell: dict[tuple[str, int], list[dict[str, Any]]] = {
        (task, length): [] for task in RULER_TASKS for length in lengths
    }
    observed_keys: set[tuple[str, int, int]] = set()
    for row in raw_rows:
        task = str(row.get("task"))
        length = int(row.get("nominal_length"))
        local_index = int(row.get("local_index"))
        key = (task, length, local_index)
        if key in observed_keys:
            raise RuntimeError(f"duplicate RULER raw row: {key}")
        observed_keys.add(key)
        if (
            (task, length) not in raw_by_cell
            or not 0 <= local_index < args.limit
        ):
            raise RuntimeError(f"RULER raw row escaped matrix: {key}")
        input_tokens = int(row.get("input_tokens"))
        generation_tokens = int(row.get("maximum_generation_tokens"))
        if (
            input_tokens <= 0
            or generation_tokens <= 0
            or input_tokens + generation_tokens > length
        ):
            raise RuntimeError(
                f"RULER physical budget drift {key}: "
                f"input={input_tokens}, generation={generation_tokens}"
            )
        raw_by_cell[(task, length)].append(row)
    require_equal(
        len(raw_rows),
        args.limit * len(lengths) * len(RULER_TASKS),
        "RULER raw row count",
    )
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
            cell_score = require_probability(
                cell.get("official_task_score"),
                f"RULER {task} L{length}",
            )
            raw_cell = raw_by_cell[(task, length)]
            require_equal(
                len(raw_cell),
                args.limit,
                f"RULER raw {task} L{length} rows",
            )
            raw_score = sum(
                require_probability(
                    row.get("official_task_score"),
                    f"RULER raw {task} L{length}",
                )
                for row in raw_cell
            ) / len(raw_cell)
            if not math.isclose(
                cell_score, raw_score, rel_tol=0.0, abs_tol=1e-12
            ):
                raise RuntimeError(
                    f"RULER {task} L{length} aggregate drift: "
                    f"{cell_score} != {raw_score}"
                )
            by_length[str(length)].append(cell_score)
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
        validate_two_wiki(args, root, manifest, result)
        if args.benchmark == "2wiki"
        else validate_ruler(args, root, manifest, result)
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
