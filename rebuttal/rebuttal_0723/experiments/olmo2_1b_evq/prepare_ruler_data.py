#!/usr/bin/env python3
"""Freeze the 13-task RULER suite for portable matched evaluation."""

from __future__ import annotations

import argparse
import functools
import gc
import hashlib
import json
import random
import sys
import types
from pathlib import Path
from typing import Any, Callable

import numpy as np


TASKS = (
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multiquery",
    "niah_multivalue",
    "ruler_vt",
    "ruler_cwe",
    "ruler_fwe",
    "ruler_qa_squad",
    "ruler_qa_hotpot",
)
LENGTHS = (4_096, 8_192, 16_384)
EXAMPLES_PER_CELL = 500
LM_EVAL_COMMIT = "f4d4b3de3ee6741a7151a9fe74945ee515262f4c"
PYTHON_RANDOM_SEED = 0
NUMPY_RANDOM_SEED = 1_234
MAX_GENERATOR_OVERRUN = {
    # The pinned upstream FWE generator reserves 50 generation tokens but its
    # per-example random vocabulary can overshoot the labeled cell by at most
    # that reserve. Preserve the upstream rows and record the exact incidence.
    "ruler_fwe": 50,
}
RULER_SOURCE_FILES = (
    "common_utils.py",
    "cwe_utils.py",
    "fwe_utils.py",
    "niah_utils.py",
    "prepare_niah.py",
    "qa_utils.py",
    "vt_utils.py",
)
QA_SOURCES = {
    (
        "https://rajpurkar.github.io/SQuAD-explorer/"
        "dataset/dev-v2.0.json"
    ): {
        "path": "squad_dev_v2.json",
        "sha256": (
            "80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8"
        ),
    },
    (
        "https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/"
        "7e54db4656209750ff487f6fdf8e39a66dba136b/"
        "hotpot_dev_distractor_v1.json"
    ): {
        "path": "hotpot_dev_distractor_v1.json",
        "sha256": (
            "e3da074df24e8369009918aa5cdbdd254dadcde4c63f7569d36afd6f2268caa8"
        ),
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def register_ruler_namespace(lm_eval_root: Path) -> None:
    """Load only the pinned RULER generators, not lm-eval's full task manager."""
    namespaces = {
        "lm_eval": lm_eval_root / "lm_eval",
        "lm_eval.tasks": lm_eval_root / "lm_eval" / "tasks",
        "lm_eval.tasks.ruler": lm_eval_root / "lm_eval" / "tasks" / "ruler",
    }
    for name, path in namespaces.items():
        if not path.is_dir():
            raise RuntimeError(f"missing lm-eval namespace path: {path}")
        module = types.ModuleType(name)
        module.__package__ = name
        module.__path__ = [str(path)]  # type: ignore[attr-defined]
        sys.modules[name] = module


def ruler_source_receipt(lm_eval_root: Path) -> dict[str, Any]:
    root = lm_eval_root / "lm_eval" / "tasks" / "ruler"
    files = {
        name: sha256_file(root / name)
        for name in RULER_SOURCE_FILES
    }
    return {
        "files": files,
        "combined_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }


def qa_source_receipt(root: Path) -> dict[str, Any]:
    files = {}
    for url, expected in QA_SOURCES.items():
        path = root / expected["path"]
        digest = sha256_file(path)
        if digest != expected["sha256"]:
            raise RuntimeError(f"RULER QA source hash drift: {path}")
        files[expected["path"]] = {
            "bytes": path.stat().st_size,
            "sha256": digest,
            "upstream_url": url,
        }
    return {"files": files}


def task_factories(
    lm_eval_root: Path,
    qa_source_root: Path,
) -> dict[str, Callable[..., dict]]:
    register_ruler_namespace(lm_eval_root)
    from lm_eval.tasks.ruler import (
        cwe_utils,
        fwe_utils,
        niah_utils,
        qa_utils,
        vt_utils,
    )

    local_sources = {
        url: qa_source_root / expected["path"]
        for url, expected in QA_SOURCES.items()
    }

    @functools.cache
    def local_download_json(url: str) -> Any:
        try:
            path = local_sources[url]
        except KeyError as error:
            raise RuntimeError(f"unregistered RULER QA source: {url}") from error
        return json.loads(path.read_text(encoding="utf-8"))

    qa_utils.download_json = local_download_json
    qa_utils.read_squad.cache_clear()
    qa_utils.read_hotpotqa.cache_clear()
    return {
        "niah_single_1": niah_utils.niah_single_1,
        "niah_single_2": niah_utils.niah_single_2,
        "niah_single_3": niah_utils.niah_single_3,
        "niah_multikey_1": niah_utils.niah_multikey_1,
        "niah_multikey_2": niah_utils.niah_multikey_2,
        "niah_multikey_3": niah_utils.niah_multikey_3,
        "niah_multiquery": niah_utils.niah_multiquery,
        "niah_multivalue": niah_utils.niah_multivalue,
        "ruler_vt": vt_utils.get_vt_dataset,
        "ruler_cwe": cwe_utils.get_cw_dataset,
        "ruler_fwe": fwe_utils.fwe_download,
        "ruler_qa_squad": qa_utils.get_squad,
        "ruler_qa_hotpot": qa_utils.get_hotpotqa,
    }


def tokenizer_receipt(root: Path) -> dict[str, Any]:
    names = (
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    )
    files = {
        name: sha256_file(root / name)
        for name in names
    }
    return {
        "files": files,
        "combined_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }


def validate_rows(
    rows: list[dict[str, Any]],
    task: str,
    *,
    lengths: tuple[int, ...] = LENGTHS,
) -> dict[str, Any]:
    expected = EXAMPLES_PER_CELL * len(lengths)
    if len(rows) != expected:
        raise RuntimeError(f"{task}: row count {len(rows)} != {expected}")
    per_length: dict[str, int] = {}
    overruns: list[int] = []
    for row in rows:
        length = int(row["max_length"])
        if length not in lengths:
            raise RuntimeError(f"{task}: unexpected target length {length}")
        overrun = int(row["length"]) - length
        if overrun > 0:
            overruns.append(overrun)
        if not row["input"] or not row["outputs"] or not row["gen_prefix"]:
            raise RuntimeError(f"{task}: incomplete RULER row")
        key = str(length)
        per_length[key] = per_length.get(key, 0) + 1
    expected_counts = {
        str(length): EXAMPLES_PER_CELL for length in lengths
    }
    if per_length != expected_counts:
        raise RuntimeError(f"{task}: per-length counts drift: {per_length}")
    maximum_overrun = max(overruns, default=0)
    allowed_overrun = MAX_GENERATOR_OVERRUN.get(task, 0)
    if maximum_overrun > allowed_overrun:
        raise RuntimeError(
            f"{task}: generator overrun {maximum_overrun} exceeds "
            f"allowed {allowed_overrun}"
        )
    return {
        "rows": len(rows),
        "per_length": per_length,
        "over_target_rows": len(overruns),
        "max_over_target_tokens": maximum_overrun,
        "first_row_sha256": hashlib.sha256(
            json.dumps(rows[0], sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "last_row_sha256": hashlib.sha256(
            json.dumps(rows[-1], sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lm-eval-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--qa-source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=TASKS,
        default=list(TASKS),
    )
    parser.add_argument(
        "--lengths",
        nargs="+",
        type=int,
        default=list(LENGTHS),
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    tasks = tuple(args.tasks)
    lengths = tuple(args.lengths)
    if len(set(tasks)) != len(tasks):
        raise RuntimeError("RULER tasks must be unique")
    if (
        not lengths
        or tuple(sorted(set(lengths))) != lengths
        or any(length < 2_048 or length > 32_768 for length in lengths)
        or any(length % 1_024 for length in lengths)
    ):
        raise RuntimeError(
            "RULER lengths must be unique ascending 1K multiples in [2K, 32K]"
        )

    lm_eval_root = args.lm_eval_root.resolve()
    pin = (lm_eval_root / "PINNED_COMMIT").read_text(encoding="utf-8").strip()
    if pin != LM_EVAL_COMMIT:
        raise RuntimeError(f"lm-eval commit drift: {pin}")
    sys.path.insert(0, str(lm_eval_root))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = tokenizer_receipt(args.tokenizer.resolve())
    qa_tasks_selected = any(task.startswith("ruler_qa_") for task in tasks)
    qa_sources = (
        qa_source_receipt(args.qa_source_root.resolve())
        if qa_tasks_selected
        else {"files": {}, "required": False}
    )
    manifest_path = output / "ruler_manifest.json"
    manifest: dict[str, Any] = {
        "status": "RULER_DATA_PREPARING",
        "implementation": {
            "repository": "EleutherAI/lm-evaluation-harness",
            "commit": LM_EVAL_COMMIT,
            "suite": "RULER",
            "tasks": list(tasks),
            "module_loading": "direct_pinned_ruler_namespace",
            "source": ruler_source_receipt(lm_eval_root),
        },
        "protocol": {
            "lengths": list(lengths),
            "examples_per_task_length": EXAMPLES_PER_CELL,
            "total_examples": (
                len(tasks) * len(lengths) * EXAMPLES_PER_CELL
            ),
            "python_random_seed_per_task": PYTHON_RANDOM_SEED,
            "numpy_random_seed_per_task": NUMPY_RANDOM_SEED,
            "tokenizer_sha256": tokenizer["combined_sha256"],
        },
        "tokenizer": tokenizer,
        "qa_sources": qa_sources,
        "files": {},
    }
    if args.resume and manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("implementation") != manifest["implementation"]:
            raise RuntimeError("existing RULER implementation drift")
        if previous.get("protocol") != manifest["protocol"]:
            raise RuntimeError("existing RULER protocol drift")
        manifest["files"] = previous.get("files", {})

    factories = task_factories(
        lm_eval_root,
        args.qa_source_root.resolve(),
    )
    for task in tasks:
        path = output / f"{task}.jsonl"
        prior = manifest["files"].get(task)
        if (
            args.resume
            and prior
            and path.is_file()
            and sha256_file(path) == prior.get("sha256")
        ):
            print(f"SKIP_VERIFIED {task}", flush=True)
            continue
        random.seed(PYTHON_RANDOM_SEED)
        np.random.seed(NUMPY_RANDOM_SEED)
        dataset = factories[task](
            tokenizer=str(args.tokenizer.resolve()),
            max_seq_lengths=list(lengths),
        )["test"]
        rows = [dict(row) for row in dataset]
        receipt = validate_rows(rows, task, lengths=lengths)
        temporary = path.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                )
        temporary.replace(path)
        receipt.update(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
        manifest["files"][task] = receipt
        write_json(manifest_path, manifest)
        print(
            f"FROZEN {task} rows={receipt['rows']} "
            f"bytes={receipt['bytes']} sha256={receipt['sha256']}",
            flush=True,
        )
        del rows
        del dataset
        gc.collect()

    if set(manifest["files"]) != set(tasks):
        raise RuntimeError("RULER suite is incomplete")
    manifest["status"] = "RULER_DATA_VERIFIED"
    manifest["total_bytes"] = sum(
        int(row["bytes"]) for row in manifest["files"].values()
    )
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
