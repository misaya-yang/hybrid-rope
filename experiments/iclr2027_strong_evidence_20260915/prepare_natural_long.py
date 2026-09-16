#!/usr/bin/env python3
"""Inventory and freeze complete natural long-context benchmark inputs.

This is a CPU-only data adapter for X3/X7.  It reads official local data,
renders the complete official prompt, and filters by the actual model-tokenizer
length.  It never downloads data, truncates a source row, or reads model
predictions.  ``candidates.jsonl`` retains every source row and its disposition;
``inputs.jsonl`` contains only the frozen rows intended for generation.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


LONGBENCH_V2 = "longbench_v2"
INFINITEBENCH = "infinitebench"
INFINITE_TASKS = ("longdialogue_qa_eng", "longbook_qa_eng")
MAX_NEW_TOKENS = {
    LONGBENCH_V2: 128,
    "longdialogue_qa_eng": 40,
    "longbook_qa_eng": 40,
}
SCORE_CONTRACT = {
    LONGBENCH_V2: "longbench_v2_mc_direct_answer_v1",
    "longdialogue_qa_eng": "infinitebench_en_dia_accuracy_v1",
    "longbook_qa_eng": "infinitebench_en_qa_rouge_f1_v1",
}
OFFICIAL_SOURCES = {
    LONGBENCH_V2: "https://huggingface.co/datasets/THUDM/LongBench-v2",
    INFINITEBENCH: "https://github.com/OpenBMB/InfiniteBench",
}


class DataRootError(ValueError):
    """The local official-data root is missing, ambiguous, or unsupported."""


@dataclass(frozen=True)
class PrepareConfig:
    benchmark: str
    model_id: str
    data_root: Path
    out: Path
    scale: int
    lengths: tuple[int, ...]
    native_length: int
    rows_per_task: int = 0
    tasks: tuple[str, ...] = ()
    minimum_input_tokens: int = 0
    maximum_input_tokens: int = 0

    def validate(self) -> None:
        if self.benchmark not in {LONGBENCH_V2, INFINITEBENCH}:
            raise ValueError(f"unsupported benchmark: {self.benchmark}")
        if not self.model_id.strip():
            raise ValueError("--model-id must be non-empty")
        if self.scale <= 1 or self.native_length <= 0:
            raise ValueError("scale must exceed one and native length must be positive")
        if self.benchmark == LONGBENCH_V2 and self.rows_per_task != 0:
            raise ValueError("LongBench-v2 requires --rows-per-task 0 for the complete eligible pool")
        if self.benchmark == INFINITEBENCH and self.rows_per_task <= 0:
            raise ValueError("InfiniteBench requires a positive --rows-per-task source-order cap")
        if self.benchmark == INFINITEBENCH:
            selected = self.tasks or INFINITE_TASKS
            if len(selected) != len(set(selected)) or not set(selected).issubset(INFINITE_TASKS):
                raise ValueError("InfiniteBench tasks must be a unique supported subset")
        elif self.tasks:
            raise ValueError("task selection is only supported for InfiniteBench")
        if not self.lengths or any(value <= 0 for value in self.lengths):
            raise ValueError("--lengths must contain positive caps")
        if tuple(sorted(set(self.lengths))) != self.lengths:
            raise ValueError("--lengths must be unique and strictly increasing")
        if not any(value > self.native_length for value in self.lengths):
            raise ValueError("--lengths must contain at least one cap above the native length")
        if self.lengths[-1] > self.native_length * self.scale:
            raise ValueError("largest requested length exceeds native_length * scale")
        if self.minimum_input_tokens < 0 or self.minimum_input_tokens > self.lengths[-1]:
            raise ValueError("minimum input tokens must be between zero and the largest cap")
        if self.maximum_input_tokens < 0 or self.maximum_input_tokens > self.lengths[-1]:
            raise ValueError("maximum input tokens must be between zero and the largest cap")
        if self.maximum_input_tokens and self.maximum_input_tokens < self.minimum_input_tokens:
            raise ValueError("maximum input tokens cannot be below the minimum")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_row_sha256(row: Mapping[str, Any]) -> str:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_text(payload)


def _portable_source_path(path: Path, data_root: Path) -> str:
    """Record a data-root-relative locator, never a workstation absolute path."""
    root = data_root if data_root.is_dir() else data_root.parent
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _find_unique(root: Path, names: set[str], description: str) -> Path:
    if root.is_file():
        if root.name not in names:
            raise DataRootError(f"{description}: unsupported file name {root.name!r}")
        return root
    if not root.exists():
        raise DataRootError(f"{description}: data root does not exist")
    matches = sorted({path.resolve() for path in root.rglob("*") if path.is_file() and path.name in names})
    if not matches:
        expected = ", ".join(sorted(names))
        raise DataRootError(f"{description}: no local official data file found; expected one of {expected}")
    if len(matches) != 1:
        shown = ", ".join(_portable_source_path(path, root) for path in matches)
        raise DataRootError(f"{description}: ambiguous local data files: {shown}")
    return matches[0]


def locate_sources(
    benchmark: str, data_root: Path, tasks: Sequence[str] = (),
) -> dict[str, Path]:
    """Locate already-present official files without network access."""
    data_root = Path(data_root)
    if benchmark == LONGBENCH_V2:
        distinctive_names = {
            "longbench_v2.jsonl", "longbench-v2.jsonl", "LongBench-v2.jsonl",
            "longbench_v2.json", "longbench-v2.json", "LongBench-v2.json",
            "longbench_v2.parquet", "longbench-v2.parquet", "LongBench-v2.parquet",
        }
        if data_root.is_file():
            allowed = distinctive_names | {"train.jsonl", "train.json", "train.parquet"}
            return {LONGBENCH_V2: _find_unique(data_root, allowed, "LongBench-v2")}
        if not data_root.exists():
            raise DataRootError("LongBench-v2: data root does not exist")
        matches = sorted({
            path.resolve() for path in data_root.rglob("*")
            if path.is_file() and path.name in distinctive_names
        })
        if not matches:
            conventional_dirs = (
                data_root,
                data_root / "LongBench-v2",
                data_root / "longbench-v2",
                data_root / "THUDM" / "LongBench-v2",
                data_root / "data" / "LongBench-v2",
            )
            matches = sorted({
                (directory / name).resolve()
                for directory in conventional_dirs
                for name in ("train.jsonl", "train.json", "train.parquet")
                if (directory / name).is_file()
            })
        if not matches:
            raise DataRootError(
                "LongBench-v2: no local official data file found; pass a downloaded JSONL, JSON, or parquet file"
            )
        if len(matches) != 1:
            shown = ", ".join(_portable_source_path(path, data_root) for path in matches)
            raise DataRootError(f"LongBench-v2: ambiguous local data files: {shown}")
        return {LONGBENCH_V2: matches[0]}
    if benchmark == INFINITEBENCH:
        if data_root.is_file():
            raise DataRootError("InfiniteBench requires a directory containing both official task files")
        sources: dict[str, Path] = {}
        missing: list[str] = []
        selected_tasks = tasks or INFINITE_TASKS
        if len(selected_tasks) != len(set(selected_tasks)) or not set(selected_tasks).issubset(INFINITE_TASKS):
            raise ValueError("InfiniteBench tasks must be a unique supported subset")
        for task in selected_tasks:
            matches = sorted({path.resolve() for path in data_root.rglob(f"{task}.jsonl") if path.is_file()}) if data_root.exists() else []
            if not matches:
                missing.append(f"{task}.jsonl")
            elif len(matches) > 1:
                shown = ", ".join(_portable_source_path(path, data_root) for path in matches)
                raise DataRootError(f"InfiniteBench {task}: ambiguous local data files: {shown}")
            else:
                sources[task] = matches[0]
        if missing:
            raise DataRootError("InfiniteBench: missing local official data files: " + ", ".join(missing))
        return sources
    raise ValueError(f"unsupported benchmark: {benchmark}")


def _iter_json(path: Path) -> Iterator[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise DataRootError(f"{path.name}:{line_number} is not a JSON object")
                yield value
        return
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            value = value.get("train", value.get("data"))
        if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
            raise DataRootError(f"{path.name} must contain a row list or a train/data row list")
        yield from value
        return
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - depends on execution image
            raise DataRootError("reading local parquet requires pyarrow; no download was attempted") from exc
        for row in pq.read_table(path).to_pylist():
            if not isinstance(row, dict):
                raise DataRootError(f"{path.name} contains a non-object row")
            yield row
        return
    raise DataRootError(f"unsupported local data format: {path.name}")


def longbench_v2_prompt(row: Mapping[str, Any]) -> str:
    """Official LongBench-v2 zero-shot direct-answer prompt."""
    context = str(row["context"]).strip()
    question = str(row["question"]).strip()
    choices = {key: str(row[f"choice_{key}"]).strip() for key in "ABCD"}
    return (
        "Please read the following text and answer the question below.\n\n"
        f"<text>\n{context}\n</text>\n\n"
        f"What is the correct answer to this question: {question}\n"
        "Choices:\n"
        f"(A) {choices['A']}\n(B) {choices['B']}\n"
        f"(C) {choices['C']}\n(D) {choices['D']}\n\n"
        'Format your response as follows: "The correct answer is (insert answer here)".'
    )


def infinitebench_prompt(task: str, row: Mapping[str, Any]) -> str:
    """Official GPT-4-style InfiniteBench prompt for the two X7 tasks."""
    if task == "longbook_qa_eng":
        question = row.get("input", row.get("question"))
        return (
            "Read the book below and answer a question.\n\n"
            f"{row['context']}\n\nQuestion: {question}\n\nBe very concise."
        )
    if task == "longdialogue_qa_eng":
        return (
            'Below is a dialogue script where one random occurrence of a character name is replaced with "$$MASK$$", '
            "and you should try to guess who that character is.\n\nThe dialogue:\n\n---\n\n"
            f"{row['context']}\n\n---\n\nEnd of dialogue.\n\n"
            'Which character is most likely "$$MASK$$"? Just say the name used by the scriptwriter '
            "(before the colon marks) of one single character and nothing else."
        )
    raise ValueError(f"unsupported InfiniteBench task: {task}")


def encode_chat_prompt(tokenizer: Any, prompt: str) -> list[int]:
    values = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True,
    )
    # Newer Transformers may return a BatchEncoding even without an explicit
    # return_dict request. Iterating it yields Encoding objects, not token IDs.
    if isinstance(values, Mapping):
        values = values["input_ids"]
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], list):
        if len(values) != 1:
            raise ValueError("tokenizer returned more than one chat row")
        values = values[0]
    if not isinstance(values, list) or not values or not all(isinstance(token, int) for token in values):
        raise ValueError("chat template did not return one non-empty integer token sequence")
    return values


def _references(value: Any) -> list[str]:
    values = value if isinstance(value, list) else [value]
    return [str(item) for item in values if item is not None and str(item) != ""]


def _length_bucket(input_tokens: int, native_length: int, lengths: Sequence[int]) -> str:
    lower = native_length + 1
    for upper in lengths:
        if upper <= native_length:
            continue
        if input_tokens <= upper:
            return f"{lower}-{upper}"
        lower = upper + 1
    return f">{lengths[-1]}"


def _execution_cap(input_tokens: int, budget: int, lengths: Sequence[int]) -> int | None:
    return next((cap for cap in lengths if input_tokens + budget <= cap), None)


def _base_candidate(
    *, benchmark: str, task: str, source_index: int, source: Mapping[str, Any],
) -> dict[str, Any]:
    source_hash = canonical_row_sha256(source)
    source_id = str(source.get("_id") or source.get("id") or source_hash)
    return {
        "row_id": f"{benchmark}:{source_id}" if benchmark == LONGBENCH_V2 else f"{benchmark}:{task}:{source_id}",
        "benchmark": benchmark,
        "task": task,
        "domain": source.get("domain"),
        "sub_domain": source.get("sub_domain"),
        "source_id": source_id,
        "source_index": source_index,
        "source_row_sha256": source_hash,
        "source_cluster_id": None,
        "input_tokens": None,
        "max_new_tokens": MAX_NEW_TOKENS[LONGBENCH_V2 if benchmark == LONGBENCH_V2 else task],
        "total_budget_tokens": None,
        "length_cap": None,
        "length_bucket": None,
        "eligible": False,
        "reason": "not_processed",
        "selected": False,
        "selection_reason": "not_eligible",
        "references": [],
        "score_contract": SCORE_CONTRACT[LONGBENCH_V2 if benchmark == LONGBENCH_V2 else task],
        "prompt_sha256": None,
    }


def _adapt_row(
    source: Mapping[str, Any], source_index: int, task: str, config: PrepareConfig, tokenizer: Any,
) -> tuple[dict[str, Any], list[int] | None]:
    candidate = _base_candidate(
        benchmark=config.benchmark, task=task, source_index=source_index, source=source,
    )
    if config.benchmark == LONGBENCH_V2:
        required = ("_id", "domain", "question", "choice_A", "choice_B", "choice_C", "choice_D", "answer", "context")
        missing = [key for key in required if source.get(key) is None or str(source.get(key)) == ""]
        if not missing and str(source["answer"]).strip().upper() not in {"A", "B", "C", "D"}:
            missing = ["valid_answer_A_to_D"]
        references = [str(source.get("answer", "")).strip().upper()] if not missing else []
        prompt = longbench_v2_prompt(source) if not missing else None
    else:
        required = ("context", "answer")
        if task == "longbook_qa_eng":
            if source.get("input", source.get("question")) in (None, ""):
                required += ("input_or_question",)
        missing = []
        for key in required:
            if key == "input_or_question":
                if source.get("input", source.get("question")) in (None, ""):
                    missing.append(key)
            elif source.get(key) is None or source.get(key) == "":
                missing.append(key)
        references = _references(source.get("answer")) if not missing else []
        if not missing and not references:
            missing = ["non_empty_answer"]
        prompt = infinitebench_prompt(task, source) if not missing else None
    if missing:
        candidate["reason"] = "missing_or_invalid_fields:" + ",".join(missing)
        return candidate, None

    context = str(source["context"])
    candidate["source_cluster_id"] = "context:" + sha256_text(context)
    candidate["references"] = references
    try:
        prompt_ids = encode_chat_prompt(tokenizer, str(prompt))
    except Exception as exc:  # retain the failed official row in the inventory
        candidate["reason"] = f"tokenization_error:{type(exc).__name__}"
        return candidate, None
    input_tokens = len(prompt_ids)
    budget = int(candidate["max_new_tokens"])
    candidate.update(
        input_tokens=input_tokens,
        total_budget_tokens=input_tokens + budget,
        length_bucket=_length_bucket(input_tokens, config.native_length, config.lengths),
        prompt_sha256=sha256_text(json.dumps(prompt_ids, separators=(",", ":"))),
    )
    if input_tokens <= config.native_length:
        candidate["reason"] = "within_native"
        return candidate, None
    if config.minimum_input_tokens and input_tokens < config.minimum_input_tokens:
        candidate["reason"] = "below_minimum_input_tokens"
        return candidate, None
    if config.maximum_input_tokens and input_tokens > config.maximum_input_tokens:
        candidate["reason"] = "above_maximum_input_tokens"
        return candidate, None
    cap = _execution_cap(input_tokens, budget, config.lengths)
    if cap is None:
        candidate["reason"] = "exceeds_max_complete_budget"
        return candidate, None
    candidate.update(eligible=True, reason="eligible", length_cap=cap)
    return candidate, prompt_ids


def _iter_benchmark_rows(
    sources: Mapping[str, Path], benchmark: str, tasks: Sequence[str] = (),
) -> Iterator[tuple[str, int, dict[str, Any]]]:
    if benchmark == LONGBENCH_V2:
        indexed_rows = list(enumerate(_iter_json(sources[LONGBENCH_V2])))
        indexed_rows.sort(key=lambda item: str(item[1].get("_id") or canonical_row_sha256(item[1])))
        for source_index, row in indexed_rows:
            yield str(row.get("domain") or "unknown"), source_index, row
        return
    for task in (tasks or INFINITE_TASKS):
        for index, row in enumerate(_iter_json(sources[task])):
            yield task, index, row


def _summarize(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, dict[str, Any]] = {}
    for task in dict.fromkeys(str(row["task"]) for row in candidates):
        rows = [row for row in candidates if row["task"] == task]
        eligible = [row for row in rows if row["eligible"]]
        selected = [row for row in rows if row["selected"]]
        lengths = [int(row["input_tokens"]) for row in rows if row["input_tokens"] is not None]
        by_task[task] = {
            "candidates": len(rows),
            "eligible": len(eligible),
            "selected": len(selected),
            "eligible_source_clusters": len({row["source_cluster_id"] for row in eligible}),
            "selected_source_clusters": len({row["source_cluster_id"] for row in selected}),
            **({"minimum_input_tokens": min(lengths), "maximum_input_tokens": max(lengths)} if lengths else {}),
        }
    selected_rows = [row for row in candidates if row["selected"]]
    task_length = Counter((str(row["task"]), str(row["length_bucket"])) for row in selected_rows)
    return {
        "candidate_rows": len(candidates),
        "eligible_rows": sum(bool(row["eligible"]) for row in candidates),
        "selected_rows": len(selected_rows),
        "by_reason": dict(sorted(Counter(str(row["reason"]) for row in candidates).items())),
        "by_task": by_task,
        "by_length_bucket": dict(sorted(Counter(str(row["length_bucket"]) for row in selected_rows).items())),
        "by_task_and_length": {
            f"{task}/{length}": count for (task, length), count in sorted(task_length.items())
        },
        "source_clusters": {
            "candidate": len({row["source_cluster_id"] for row in candidates if row["source_cluster_id"]}),
            "eligible": len({row["source_cluster_id"] for row in candidates if row["eligible"]}),
            "selected": len({row["source_cluster_id"] for row in selected_rows}),
        },
    }


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def prepare_dataset(
    config: PrepareConfig, tokenizer: Any, sources: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    config.validate()
    if config.out.exists():
        raise FileExistsError(f"refuse to overwrite frozen output: {config.out}")
    sources = dict(sources or locate_sources(config.benchmark, config.data_root))
    candidates: list[dict[str, Any]] = []
    prompt_ids_by_row: dict[str, list[int]] = {}
    seen_ids: set[str] = set()
    selected_per_task: Counter[str] = Counter()
    selected_tasks = config.tasks or (INFINITE_TASKS if config.benchmark == INFINITEBENCH else ())
    for task, source_index, source in _iter_benchmark_rows(
        sources, config.benchmark, selected_tasks,
    ):
        candidate, prompt_ids = _adapt_row(source, source_index, task, config, tokenizer)
        row_id = str(candidate["row_id"])
        if row_id in seen_ids:
            raise ValueError(f"duplicate stable row ID in official source: {row_id}")
        seen_ids.add(row_id)
        if candidate["eligible"]:
            if config.benchmark == LONGBENCH_V2 or selected_per_task[task] < config.rows_per_task:
                candidate.update(selected=True, selection_reason="selected")
                selected_per_task[task] += 1
                assert prompt_ids is not None
                prompt_ids_by_row[row_id] = prompt_ids
            else:
                candidate["selection_reason"] = "source_order_limit"
        candidates.append(candidate)
    if not candidates:
        raise DataRootError("official source contains no rows")

    selected_inputs: list[dict[str, Any]] = []
    for candidate in candidates:
        if not candidate["selected"]:
            continue
        row = dict(candidate)
        row.pop("eligible", None)
        row.pop("reason", None)
        row.pop("selected", None)
        row.pop("selection_reason", None)
        row["prompt_ids"] = prompt_ids_by_row[str(candidate["row_id"])]
        selected_inputs.append(row)

    work = config.out.with_name(config.out.name + ".building")
    if work.exists():
        raise FileExistsError(f"stale build directory exists: {work}")
    work.mkdir(parents=True)
    candidate_path = work / "candidates.jsonl"
    input_path = work / "inputs.jsonl"
    _write_jsonl(candidate_path, candidates)
    _write_jsonl(input_path, selected_inputs)
    summary = _summarize(candidates)
    source_receipts = {
        task: {
            "file": _portable_source_path(path, config.data_root),
            "sha256": sha256_file(path),
        }
        for task, path in sorted(sources.items())
    }
    manifest = {
        "status": "COMPLETE",
        "contract": "ICLR2027_STRONG_NATURAL_LONG_CPU_V1",
        "benchmark": config.benchmark,
        "model_id": config.model_id,
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "vocab_size": getattr(tokenizer, "vocab_size", None),
            "chat_template_sha256": (
                sha256_text(tokenizer.chat_template)
                if isinstance(getattr(tokenizer, "chat_template", None), str) else None
            ),
        },
        "scale": config.scale,
        "native_length": config.native_length,
        "lengths": list(config.lengths),
        "rows_per_task": config.rows_per_task,
        "tasks": list(selected_tasks),
        "minimum_input_tokens": config.minimum_input_tokens,
        "maximum_input_tokens": config.maximum_input_tokens,
        "selection": (
            "all complete LongBench-v2 rows with native_length < actual full prompt tokens and prompt+128 within a requested cap, sorted by _id"
            if config.benchmark == LONGBENCH_V2 else
            "source-order first rows-per-task among complete eligible official rows within the declared input-token bounds; no truncation or replacement"
        ),
        "prompt_contract": (
            "official LongBench-v2 prompts/0shot.txt wrapped once in the target model chat template"
            if config.benchmark == LONGBENCH_V2 else
            "official InfiniteBench GPT-4 task template wrapped once in the target model chat template"
        ),
        "complete_context_preserved": True,
        "model_outputs_read": False,
        "downloads_attempted": False,
        "official_source": OFFICIAL_SOURCES[config.benchmark],
        "source_files": source_receipts,
        "summary": summary,
        "candidates_sha256": sha256_file(candidate_path),
        "inputs_sha256": sha256_file(input_path),
    }
    _atomic_json(work / "manifest.json", manifest)
    os.replace(work, config.out)
    return manifest


def _missing_manifest(args: argparse.Namespace, message: str) -> dict[str, Any]:
    return {
        "status": "MISSING_DATA",
        "contract": "ICLR2027_STRONG_NATURAL_LONG_CPU_V1",
        "benchmark": args.benchmark.replace("-", "_"),
        "model_id": args.model_id,
        "scale": args.scale,
        "lengths": list(args.lengths),
        "tasks": list(args.task or []),
        "minimum_input_tokens": args.minimum_input_tokens,
        "maximum_input_tokens": args.maximum_input_tokens,
        "downloads_attempted": False,
        "error": message,
        "resolution": "place the official dataset files under --data-root, then run again with a new empty --out",
    }


def _native_length(model: Path) -> int:
    config_path = model / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"local model config not found: {config_path.name}; no download was attempted")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    value = config.get("max_position_embeddings")
    if not isinstance(value, int) or value <= 0:
        raise ValueError("model config has no positive max_position_embeddings")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("longbench-v2", INFINITEBENCH), required=True)
    parser.add_argument("--model", type=Path, required=True, help="existing local model/tokenizer directory")
    parser.add_argument("--model-id", required=True, help="portable model identity written to the manifest")
    parser.add_argument("--data-root", type=Path, required=True, help="existing official data root; never downloaded")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scale", type=int, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", required=True)
    parser.add_argument("--rows-per-task", type=int, default=0,
                        help="0 for all eligible LongBench-v2 rows; positive InfiniteBench source-order cap")
    parser.add_argument("--task", action="append", choices=INFINITE_TASKS,
                        help="prepare only this InfiniteBench task; repeat to include more")
    parser.add_argument("--minimum-input-tokens", type=int, default=0,
                        help="exclude otherwise eligible prompts shorter than this")
    parser.add_argument("--maximum-input-tokens", type=int, default=0,
                        help="exclude otherwise eligible prompts longer than this; 0 disables")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.benchmark = args.benchmark.replace("-", "_")
    args.lengths = tuple(args.lengths)
    if args.out.exists():
        raise FileExistsError(f"refuse to overwrite output: {args.out}")
    try:
        sources = locate_sources(args.benchmark, args.data_root, tuple(args.task or ()))
    except DataRootError as exc:
        args.out.mkdir(parents=True)
        manifest = _missing_manifest(args, str(exc))
        _atomic_json(args.out / "manifest.json", manifest)
        print(json.dumps(manifest, ensure_ascii=False, sort_keys=True), flush=True)
        raise SystemExit(2)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    config = PrepareConfig(
        benchmark=args.benchmark,
        model_id=args.model_id,
        data_root=args.data_root,
        out=args.out,
        scale=args.scale,
        lengths=args.lengths,
        native_length=_native_length(args.model),
        rows_per_task=args.rows_per_task,
        tasks=tuple(args.task or ()),
        minimum_input_tokens=args.minimum_input_tokens,
        maximum_input_tokens=args.maximum_input_tokens,
    )
    manifest = prepare_dataset(config, tokenizer, sources)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
