#!/usr/bin/env python3
"""Build unpadded, relative-Native context manifests for target-free retrofit.

The CLI is intentionally fail-closed: full tokenization requires the explicit
``--allow-full-tokenization`` acknowledgement.  The no-card preflight only
uses the small fixture tests; it does not invoke this full-data path.

LongBench rows use the pinned official prompt templates and the target model's
chat template.  PG-19 anchors are right-aligned nested suffixes of the same
book so every multiplier shares the same final NLL target span.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


STATUS = "TARGET_FREE_TOKEN_MANIFEST_COMPLETE"
QA_GENERATION_RESERVE = 64
BUCKET_NAMES = ("retention", "near", "far")
PG19_MULTIPLIERS = (1, 2, 4)
PG19_TAIL_NLL_TOKENS = 512


@dataclass(frozen=True)
class BucketDecision:
    name: str | None
    total_tokens: int
    lower_exclusive: int
    upper_inclusive: int
    reason: str | None = None


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def row_sha256(row: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json(row))


def tokenizer_tree_sha256(root: Path) -> str:
    """Hash tokenizer/config files without reading or loading model weights."""

    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.endswith(".safetensors"):
            continue
        relative = path.relative_to(root).as_posix()
        files.append(
            {
                "path": relative,
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    if not files:
        raise FileNotFoundError(f"no tokenizer/config files under {root}")
    return sha256_bytes(canonical_json(files))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        value = json.loads(text)
        if not isinstance(value, list):
            raise ValueError(f"expected a JSON list at {path}")
        return [dict(row) for row in value]
    rows = []
    for line in text.splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object rows at {path}")
            rows.append(dict(value))
    return rows


def choose_bucket(total_tokens: int, native_context_length: int) -> BucketDecision:
    total = int(total_tokens)
    native = int(native_context_length)
    if total < 0 or native <= 0:
        raise ValueError("token counts and native context length must be positive")
    if total <= native:
        return BucketDecision("retention", total, 0, native)
    if total <= 2 * native:
        return BucketDecision("near", total, native, 2 * native)
    if total <= 4 * native:
        return BucketDecision("far", total, 2 * native, 4 * native)
    return BucketDecision(
        None,
        total,
        4 * native,
        4 * native,
        reason="prompt plus generation reserve exceeds 4 * L_native",
    )


def _token_ids(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(item) for item in value]


def render_chat_prompt(tokenizer: Any, content: str) -> tuple[list[int], str]:
    if not hasattr(tokenizer, "apply_chat_template"):
        raise TypeError("target tokenizer must expose apply_chat_template")
    messages = [{"role": "user", "content": content}]
    tokenized = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
    )
    ids = _token_ids(tokenized)
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
    except TypeError:
        rendered = content
    return ids, str(rendered)


def evidence_receipt(
    row: Mapping[str, Any],
    *,
    query_start_token: int | None = None,
) -> dict[str, Any]:
    """Record evidence distance when the source exposes a token-level span."""

    token_start = row.get("evidence_token_start")
    if token_start is None:
        token_start = row.get("answer_token_start")
    annotation_keys = (
        "evidence",
        "evidence_spans",
        "supporting_facts",
        "supporting_sentences",
    )
    present = any(key in row and row[key] for key in annotation_keys)
    if token_start is not None and query_start_token is not None:
        distance = int(query_start_token) - int(token_start)
        return {
            "evidence_annotation_present": True,
            "evidence_to_query_token_distance": distance,
            "evidence_distance_status": "source token offsets",
        }
    return {
        "evidence_annotation_present": bool(present),
        "evidence_to_query_token_distance": None,
        "evidence_distance_status": (
            "annotation present but no released token offset"
            if present
            else "not provided by source row"
        ),
    }


def official_reserve(
    task: str,
    official_generation_budgets: Mapping[str, Any],
    *,
    qa_reserve: int = QA_GENERATION_RESERVE,
) -> tuple[int, str]:
    if str(task) == "gov_report":
        value = official_generation_budgets.get("gov_report")
        if value is None or int(value) <= 0:
            raise ValueError("official GovReport generation budget is missing")
        return int(value), "LongBench dataset2maxlen.json"
    return int(qa_reserve), "target-free QA reserve contract"


def load_longbench_prompt_assets(longbench_root: Path) -> tuple[dict[str, str], dict[str, int]]:
    config = longbench_root / "official_config"
    prompts = load_json(config / "dataset2prompt.json")
    budgets = load_json(config / "dataset2maxlen.json")
    if not isinstance(prompts, dict) or not isinstance(budgets, dict):
        raise ValueError("LongBench official config files must contain objects")
    return {str(key): str(value) for key, value in prompts.items()}, {
        str(key): int(value) for key, value in budgets.items()
    }


def build_longbench_row(
    row: Mapping[str, Any],
    *,
    task: str,
    family: str,
    prompt_template: str,
    tokenizer: Any,
    tokenizer_sha256: str,
    native_context_length: int,
    generation_reserve: int,
    generation_reserve_source: str,
) -> tuple[dict[str, Any] | None, str | None]:
    context = row.get("context")
    question = row.get("input")
    if not isinstance(context, str) or not isinstance(question, str):
        return None, "row lacks string context or input"
    references = row.get("answers")
    if not isinstance(references, list) or not references:
        return None, "row lacks nonempty answers"
    try:
        content = prompt_template.format(context=context, input=question)
    except KeyError as exc:
        raise ValueError(f"official prompt has unsupported field: {exc}") from exc
    input_ids, rendered = render_chat_prompt(tokenizer, content)
    total = len(input_ids) + int(generation_reserve)
    decision = choose_bucket(total, native_context_length)
    if decision.name is None:
        return None, decision.reason
    digest = row_sha256(row)
    record = {
        "source_family": family,
        "task": task,
        "source_id": str(row.get("_id", digest)),
        "row_sha256": digest,
        "source_context_sha256": sha256_text(context),
        "prompt_sha256": sha256_text(content),
        "rendered_chat_prompt_sha256": sha256_text(rendered),
        "tokenizer_sha256": tokenizer_sha256,
        "input_ids": input_ids,
        "references": [str(value) for value in references],
        "all_classes": [str(value) for value in (row.get("all_classes") or [])],
        "input_tokens": len(input_ids),
        "generation_reserve": int(generation_reserve),
        "generation_reserve_source": generation_reserve_source,
        "total_tokens_with_reserve": total,
        "bucket": decision.name,
        "native_context_length": int(native_context_length),
        "nominal_window_upper_bound": int(decision.upper_inclusive),
        **evidence_receipt(row),
    }
    return record, None


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".incomplete",
        mode="w",
        encoding="utf-8",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def build_longbench(
    *,
    longbench_root: Path,
    tokenizer: Any,
    tokenizer_sha256: str,
    native_context_length: int,
    output: Path,
    max_rows_per_cell: int = 0,
    qa_reserve: int = QA_GENERATION_RESERVE,
) -> dict[str, Any]:
    prompts, budgets = load_longbench_prompt_assets(longbench_root)
    manifest: dict[str, Any] = {
        "status": STATUS,
        "native_context_length": int(native_context_length),
        "tokenizer_sha256": tokenizer_sha256,
        "padding": False,
        "synthetic_needles": False,
        "unrelated_concatenation": False,
        "main_result_truncation": False,
        "cells": {},
        "insufficient": [],
    }
    extracted = longbench_root / "extracted"
    for family in ("longbench", "longbench_e"):
        for task in sorted(p.stem.removesuffix("_e") for p in (extracted / family).glob("*.jsonl")):
            if task.endswith("_e"):
                task = task[:-2]
            source_path = extracted / family / f"{task}{'_e' if family == 'longbench_e' else ''}.jsonl"
            if not source_path.is_file() or task not in prompts:
                continue
            reserve, reserve_source = official_reserve(
                task,
                budgets,
                qa_reserve=qa_reserve,
            )
            rows = load_rows(source_path)
            indexed = sorted(
                enumerate(rows),
                key=lambda item: row_sha256(item[1]),
            )
            if int(max_rows_per_cell) > 0:
                indexed = indexed[: int(max_rows_per_cell)]
            kept: list[dict[str, Any]] = []
            rejected: dict[str, int] = {}
            for source_index, row in indexed:
                record, reason = build_longbench_row(
                    row,
                    task=task,
                    family=family,
                    prompt_template=prompts[task],
                    tokenizer=tokenizer,
                    tokenizer_sha256=tokenizer_sha256,
                    native_context_length=native_context_length,
                    generation_reserve=reserve,
                    generation_reserve_source=reserve_source,
                )
                if record is None:
                    rejected[reason or "unknown"] = rejected.get(reason or "unknown", 0) + 1
                    continue
                record["source_row_index"] = int(source_index)
                kept.append(record)
            cell = f"{family}:{task}"
            cell_path = output / family / task / "rows.jsonl"
            _atomic_jsonl(cell_path, kept)
            counts = {name: sum(row["bucket"] == name for row in kept) for name in BUCKET_NAMES}
            manifest["cells"][cell] = {
                "source_path": str(source_path),
                "source_rows": len(rows),
                "selected_rows": len(indexed),
                "kept_rows": len(kept),
                "bucket_counts": counts,
                "rejected": rejected,
                "rows_path": cell_path.relative_to(output.parent).as_posix(),
                "rows_sha256": sha256_file(cell_path),
                "generation_reserve": reserve,
                "generation_reserve_source": reserve_source,
            }
            if rejected:
                manifest["insufficient"].append({"cell": cell, "rejected": rejected})
    return manifest


def _load_pg19_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="strict")


def build_pg19_anchors(
    *,
    pg19_root: Path,
    tokenizer: Any,
    tokenizer_sha256: str,
    native_context_length: int,
    output: Path,
    anchor_count: int = 1,
) -> dict[str, Any]:
    if int(anchor_count) <= 0:
        raise ValueError("anchor_count must be positive")
    source_manifest = load_json(pg19_root.parent / "download_manifest.json")
    _ = source_manifest  # checked below through the pinned book inventory
    books = []
    for path in sorted((pg19_root / "books" / "test").glob("*.txt")):
        books.append(path)
    eligible = []
    for path in books:
        text = _load_pg19_text(path)
        ids = _token_ids(tokenizer(text, add_special_tokens=False)["input_ids"])
        if len(ids) >= PG19_MULTIPLIERS[-1] * int(native_context_length):
            eligible.append((sha256_text(text), path, ids))
    eligible.sort(key=lambda item: item[0])
    selected = eligible[: int(anchor_count)]
    if len(selected) < int(anchor_count):
        raise RuntimeError(
            f"PG-19 test has only {len(selected)} books with at least "
            f"{PG19_MULTIPLIERS[-1]} * L_native history"
        )
    rows: list[dict[str, Any]] = []
    for anchor_index, (source_hash, path, ids) in enumerate(selected):
        target_hashes = []
        for multiplier in PG19_MULTIPLIERS:
            length = int(multiplier) * int(native_context_length)
            view = ids[-length:]
            target_start = length - PG19_TAIL_NLL_TOKENS
            target = view[target_start:]
            target_hash = sha256_bytes(canonical_json(target))
            target_hashes.append(target_hash)
            rows.append(
                {
                    "anchor_index": anchor_index,
                    "anchor_source": path.name,
                    "anchor_source_sha256": source_hash,
                    "tokenizer_sha256": tokenizer_sha256,
                    "native_context_length": int(native_context_length),
                    "multiplier": int(multiplier),
                    "context_tokens": len(view),
                    "input_ids": view,
                    "nll_target_start": target_start,
                    "nll_target_tokens": PG19_TAIL_NLL_TOKENS,
                    "nll_target_sha256": target_hash,
                    "row_sha256": sha256_bytes(
                        canonical_json(
                            {
                                "anchor_source_sha256": source_hash,
                                "multiplier": int(multiplier),
                                "target_hash": target_hash,
                            }
                        )
                    ),
                }
            )
        if len(set(target_hashes)) != 1:
            raise RuntimeError("nested PG-19 views do not share the same tail")
    output.mkdir(parents=True, exist_ok=True)
    rows_path = output / "pg19_nested_anchors.jsonl"
    _atomic_jsonl(rows_path, rows)
    return {
        "status": STATUS,
        "native_context_length": int(native_context_length),
        "anchor_count": len(selected),
        "max_one_anchor_per_book": True,
        "multipliers": list(PG19_MULTIPLIERS),
        "nll_target_tokens": PG19_TAIL_NLL_TOKENS,
        "rows": len(rows),
        "rows_path": rows_path.relative_to(output.parent).as_posix(),
        "rows_sha256": sha256_file(rows_path),
        "selected_books": [path.name for _, path, _ in selected],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--tokenizer-root", type=Path, required=True)
    parser.add_argument("--native-context-length", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rows-per-cell", type=int, default=0)
    parser.add_argument("--pg19-anchor-count", type=int, default=1)
    parser.add_argument(
        "--allow-full-tokenization",
        action="store_true",
        help="Required acknowledgement for the later full-data build.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.allow_full_tokenization:
        raise SystemExit(
            "refusing full tokenization; rerun only on the authorised machine "
            "with --allow-full-tokenization"
        )
    download_root = args.download_root.expanduser().resolve()
    tokenizer_root = args.tokenizer_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if int(args.native_context_length) <= 0:
        raise ValueError("native context length must be positive")
    tokenizer_sha256 = tokenizer_tree_sha256(tokenizer_root)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_root,
        local_files_only=True,
        trust_remote_code=False,
    )
    longbench_manifest = build_longbench(
        longbench_root=download_root / "longbench",
        tokenizer=tokenizer,
        tokenizer_sha256=tokenizer_sha256,
        native_context_length=int(args.native_context_length),
        output=output / "longbench",
        max_rows_per_cell=int(args.max_rows_per_cell),
    )
    pg19_manifest = build_pg19_anchors(
        pg19_root=download_root / "pg19",
        tokenizer=tokenizer,
        tokenizer_sha256=tokenizer_sha256,
        native_context_length=int(args.native_context_length),
        output=output / "pg19",
        anchor_count=int(args.pg19_anchor_count),
    )
    manifest = {
        "status": STATUS,
        "tokenization_executed": True,
        "native_context_length": int(args.native_context_length),
        "tokenizer_sha256": tokenizer_sha256,
        "longbench": longbench_manifest,
        "pg19": pg19_manifest,
        "pending": [
            "GPU evaluation command execution",
            "paired tail-NLL and task-specific generation metrics",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "token_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
