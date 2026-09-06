#!/usr/bin/env python3
"""Freeze a small natural-QA panel with all source evidence before 28K and the query near 64K."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any
import zipfile

from scripts.data.prepare_qwen_k32_natural_nll import (
    iter_source_texts,
    sha256_file,
    tokenizer_file_receipts,
)
from scripts.eval.eval_qwen_k32_natural_nll import GRID, load_data
from scripts.eval.longbench_metrics import normalize_text

STATUS = "QWEN_K32_FAR_EVIDENCE_QA_DATA_READY_V2"
TASKS = ("2wikimqa", "qasper", "hotpotqa")
ROWS_PER_TASK = 10
TARGET_LENGTH = 65_536
EVIDENCE_END_CEILING = 28_672
MIN_EVIDENCE_TO_QUERY_TOKENS = 32_768
EVIDENCE_MARKER = "__QWEN_FAR_QA_EVIDENCE_END_20260901__"
FILLER_MARKER = "__QWEN_FAR_QA_FILLER_20260901__"
COMMON_ANSWERS = {"yes", "no", "unanswerable"}
EXPECTED_MAIN_ZIP_SHA256 = "3824b6dae9738eb70f7c89cf1a1e13d37cb626b7dab779780d1a96f393c2280a"
EXPECTED_DATA_ZIP_SHA256 = "cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64"
EXPECTED_PACKED_MANIFEST_SHA256 = "988cce3e4f5e8d8cf9bbcdb1e3ec35034ebb629374b9273a8176d9315f811c39"
EXPECTED_PACKED_ROWS_SHA256 = "2301eb693bdf84963cf0d1645fd158d207358ec5743fc197b26f8c9ce3e16fa6"
EXPECTED_QUESTION_PANEL_MANIFEST_SHA256 = "b2c5bf42487fb3394cee6b1f66393b4f824849d778f3871c10d3b2ca1f8bc457"
EXPECTED_QUESTION_PANEL_ROWS_SHA256 = "fca23e32d8019e245635175ab4d951d255270cc9e98840c745be6c383cdf82d6"
EXPECTED_INVALID_RUN_HASHES = {
    "run_manifest_sha256": "d703d760902ced46d35942bfb3ad2734eb1e1c2a04a832cc66d923df79b2f686",
    "examples_sha256": "0ebd4d46cfe2340719eadeea913c35a7d40e275ef7de68cd7d00dd9e8b2d7325",
    "results_sha256": "9e66e3ad1af6593bb0d9e804a01bc595b403124796baeb84c8cecd46193735e3",
}
EXPECTED_SOURCE_SHA256 = "3fcf2dc69cd52503986276d3d2d26a8c356d0f2ea28a0de4fdbda8cf87755693"
NLL_SOURCE_RANGE = (650000, 652018)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode()).hexdigest()


def ids_sha256(values: list[int]) -> str:
    return canonical_sha256([int(value) for value in values])


def zip_member(archive: zipfile.ZipFile, suffix: str) -> str:
    matches = [name for name in archive.namelist() if name == suffix or name.endswith("/" + suffix)]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one zip member ending in {suffix}")
    return matches[0]


def load_sources(main_zip: Path, data_zip: Path) -> tuple[dict[str, str], dict[str, int], dict[str, list[dict]]]:
    with zipfile.ZipFile(main_zip) as archive:
        prompts = json.loads(archive.read(zip_member(archive, "config/dataset2prompt.json")))
        budgets = json.loads(archive.read(zip_member(archive, "config/dataset2maxlen.json")))
    rows = {}
    with zipfile.ZipFile(data_zip) as archive:
        for task in TASKS:
            rows[task] = [json.loads(line) for line in archive.read(
                zip_member(archive, f"data/{task}.jsonl")).decode().splitlines() if line.strip()]
            if len(rows[task]) != 200:
                raise ValueError(f"LongBench {task} row count drift")
    expected_budgets = {"2wikimqa": 32, "qasper": 128, "hotpotqa": 32}
    if any(int(budgets.get(task, 0)) != expected_budgets[task] for task in TASKS):
        raise ValueError("LongBench generation budget drift")
    return ({task: str(prompts[task]) for task in TASKS}, expected_budgets, rows)


def token_ids(tokenizer: Any, text: str) -> list[int]:
    value = tokenizer(text, add_special_tokens=False)["input_ids"]
    return [int(item) for item in value]


def prompt_segments(tokenizer: Any, template: str, source: dict) -> tuple[list[int], list[int], list[int]]:
    context, question = source.get("context"), source.get("input")
    if not isinstance(context, str) or not isinstance(question, str):
        raise ValueError("LongBench row lacks context or input")
    marked_context = (
        context + "\n\n" + EVIDENCE_MARKER
        + "\n\nThe following text is unrelated distractor material. Ignore it when answering.\n"
        + FILLER_MARKER
    )
    content = template.format(context=marked_context, input=question)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False)
    if rendered.count(EVIDENCE_MARKER) != 1 or rendered.count(FILLER_MARKER) != 1:
        raise ValueError("chat template did not preserve the two placement markers")
    evidence_text, remainder = rendered.split(EVIDENCE_MARKER)
    label_text, query_text = remainder.split(FILLER_MARKER)
    return token_ids(tokenizer, evidence_text), token_ids(tokenizer, label_text), token_ids(tokenizer, query_text)


def answer_leaks(tokenizer: Any, filler: list[int], references: list[str]) -> bool:
    normalized_filler = " " + normalize_text(tokenizer.decode(
        filler, skip_special_tokens=True, clean_up_tokenization_spaces=False)) + " "
    return any(
        normalized and normalized not in COMMON_ANSWERS and f" {normalized} " in normalized_filler
        for normalized in (normalize_text(value) for value in references)
    )


def sanitize_filler(tokenizer: Any, values: list[int]) -> tuple[list[int], list[int], int]:
    special = {int(value) for value in getattr(tokenizer, "all_special_ids", [])}
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        special.add(int(eos))
    separator = token_ids(tokenizer, "\n")
    if len(separator) != 1 or separator[0] in special:
        raise ValueError("ordinary newline separator tokenization is invalid")
    cleaned, positions = [], []
    for position, value in enumerate(values):
        if int(value) in special:
            cleaned.append(separator[0]); positions.append(position)
        else:
            cleaned.append(int(value))
    return cleaned, positions, separator[0]


def assemble_row(tokenizer: Any, *, task: str, source_index: int, source: dict,
                 template: str, generation_tokens: int, filler_row: dict) -> dict | None:
    references = [str(value) for value in source.get("answers") or []]
    if not references or all(normalize_text(value) in COMMON_ANSWERS for value in references):
        return None
    evidence, label, query = prompt_segments(tokenizer, template, source)
    target_input = TARGET_LENGTH - int(generation_tokens)
    filler_tokens = target_input - len(evidence) - len(label) - len(query)
    query_start = len(evidence) + len(label) + filler_tokens
    clean_filler, replacement_positions, separator_id = sanitize_filler(
        tokenizer, filler_row["input_ids"])
    if (
        len(evidence) > EVIDENCE_END_CEILING
        or filler_tokens <= 0
        or query_start - len(evidence) < MIN_EVIDENCE_TO_QUERY_TOKENS
        or filler_tokens > len(clean_filler)
    ):
        return None
    filler = clean_filler[:filler_tokens]
    special = {int(value) for value in getattr(tokenizer, "all_special_ids", [])}
    if any(value in special for value in filler):
        raise AssertionError("sanitized filler retains a tokenizer special token")
    if answer_leaks(tokenizer, filler, references):
        return None
    input_ids = evidence + label + filler + query
    evidence_special = [value for value in evidence if value in special]
    label_special = [value for value in label if value in special]
    query_special = [value for value in query if value in special]
    if (evidence_special != [151644, 151645, 151644] or label_special
            or query_special != [151645, 151644]):
        raise ValueError("Qwen chat wrapper special-token pattern drift")
    if len(input_ids) != target_input:
        raise AssertionError("far-evidence prompt length assembly drift")
    source_hash = canonical_sha256(source)
    identity = {
        "task": task,
        "source_index": int(source_index),
        "source_row_sha256": source_hash,
        "filler_sample_id": filler_row["sample_id"],
        "input_ids_sha256": ids_sha256(input_ids),
    }
    return {
        **identity,
        "row_sha256": canonical_sha256(identity),
        "input_ids": input_ids,
        "references": references,
        "all_classes": [str(value) for value in source.get("all_classes") or []],
        "input_tokens": len(input_ids),
        "generation_tokens": int(generation_tokens),
        "target_length_with_generation": TARGET_LENGTH,
        "evidence_end_token_exclusive": len(evidence),
        "query_suffix_start_token": query_start,
        "evidence_to_query_lower_bound": query_start - len(evidence),
        "filler_tokens": filler_tokens,
        "filler_prompt_sha256": filler_row["prompt_ids_sha256"],
        "filler_source_special_tokens_replaced": len(replacement_positions),
        "filler_source_special_positions_sha256": canonical_sha256(replacement_positions),
        "filler_replacement_token_id": separator_id,
        "filler_sanitized_ids_sha256": ids_sha256(filler),
        "filler_selected_special_tokens": 0,
        "query_suffix_ids_sha256": ids_sha256(query),
        "evidence_special_ids": evidence_special,
        "label_special_ids": label_special,
        "query_special_ids": query_special,
        "answer_leak_check": "normalized exact phrase absent from inserted filler",
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", mode="w",
                                     encoding="utf-8", delete=False) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush(); os.fsync(handle.fileno())
    temporary.replace(path)


def build(checkpoint: Path, packed_root: Path, source_parquet: Path,
          main_zip: Path, data_zip: Path,
          question_panel_root: Path, invalid_run_root: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("far-evidence output must be fresh")
    if (sha256_file(source_parquet) != EXPECTED_SOURCE_SHA256
            or sha256_file(main_zip) != EXPECTED_MAIN_ZIP_SHA256
            or sha256_file(data_zip) != EXPECTED_DATA_ZIP_SHA256):
        raise ValueError("frozen source or LongBench archive hash mismatch")
    packed, packed_rows = load_data(packed_root, checkpoint)
    if (packed["manifest_sha256"] != EXPECTED_PACKED_MANIFEST_SHA256
            or packed["file"]["sha256"] != EXPECTED_PACKED_ROWS_SHA256):
        raise ValueError("frozen packed-natural filler identity mismatch")
    fillers = [row for row in packed_rows if row["length"] == GRID[-1]]
    prompts, budgets, sources = load_sources(main_zip, data_zip)
    panel_manifest_path = question_panel_root / "manifest.json"
    panel_rows_path = question_panel_root / "rows.jsonl"
    if (sha256_file(panel_manifest_path) != EXPECTED_QUESTION_PANEL_MANIFEST_SHA256
            or sha256_file(panel_rows_path) != EXPECTED_QUESTION_PANEL_ROWS_SHA256):
        raise ValueError("frozen invalid-run question panel identity mismatch")
    panel_rows = [json.loads(line) for line in panel_rows_path.read_text().splitlines() if line.strip()]
    invalid_hashes = {
        "run_manifest_sha256": sha256_file(invalid_run_root / "run_manifest.json"),
        "examples_sha256": sha256_file(invalid_run_root / "examples.jsonl"),
        "results_sha256": sha256_file(invalid_run_root / "results.json"),
    }
    if invalid_hashes != EXPECTED_INVALID_RUN_HASHES:
        raise ValueError("prior invalid QA run identity mismatch")
    if (packed["source"]["start_row"] != NLL_SOURCE_RANGE[1] + 1
            or packed["source"]["consumed_row_range"][0] <= NLL_SOURCE_RANGE[1]):
        raise ValueError("fresh filler source rows overlap the NLL source range")
    fresh_range = tuple(int(value) for value in packed["source"]["consumed_row_range"])
    old_hashes, fresh_hashes = set(), set()
    for source_row, text in iter_source_texts(source_parquet, NLL_SOURCE_RANGE[0]):
        if source_row > fresh_range[1]:
            break
        digest = hashlib.sha256(text.encode()).hexdigest()
        if NLL_SOURCE_RANGE[0] <= source_row <= NLL_SOURCE_RANGE[1]:
            old_hashes.add(digest)
        if fresh_range[0] <= source_row <= fresh_range[1]:
            fresh_hashes.add(digest)
    source_intersection = old_hashes & fresh_hashes
    if source_intersection:
        raise ValueError("fresh filler duplicates source text from the NLL row range")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, local_files_only=True, trust_remote_code=False)
    _, _, newline_id = sanitize_filler(tokenizer, [])
    if newline_id != 198:
        raise ValueError("Qwen newline token identity drift")

    questions = []
    old_filler_hashes = {row["filler_prompt_sha256"] for row in panel_rows}
    if old_filler_hashes & {row["prompt_ids_sha256"] for row in fillers}:
        raise ValueError("fresh filler stream overlaps the prior invalid panel")
    for task in TASKS:
        frozen = [row for row in panel_rows if row.get("task") == task]
        if len(frozen) != ROWS_PER_TASK:
            raise ValueError("prior question panel task count drift")
        for frozen_row in frozen:
            source_index = int(frozen_row["source_index"])
            source = sources[task][source_index]
            if canonical_sha256(source) != frozen_row["source_row_sha256"]:
                raise ValueError("prior question panel source identity drift")
            candidates = []
            for filler_index, filler in enumerate(fillers):
                row = assemble_row(
                    tokenizer, task=task, source_index=source_index, source=source,
                    template=prompts[task], generation_tokens=budgets[task], filler_row=filler)
                if row is not None:
                    candidates.append((filler_index, row))
            if not candidates:
                raise RuntimeError(f"fixed question has no safe filler: {task}:{source_index}")
            questions.append((task, candidates))

    matched: dict[int, tuple[int, dict]] = {}
    def assign(question_index: int, seen: set[int]) -> bool:
        for filler_index, row in questions[question_index][1]:
            if filler_index in seen:
                continue
            seen.add(filler_index)
            previous = matched.get(filler_index)
            if previous is None or assign(previous[0], seen):
                matched[filler_index] = (question_index, row)
                return True
        return False
    for question_index in range(len(questions)):
        if not assign(question_index, set()):
            raise RuntimeError("fixed 30-question panel has no one-to-one safe filler assignment")
    selected_by_question = {question_index: row for question_index, row in matched.values()}
    selected = [selected_by_question[index] for index in range(len(questions))]
    task_counts = {task: sum(row["task"] == task for row in selected) for task in TASKS}

    output.mkdir(parents=True)
    rows_path = output / "rows.jsonl"
    write_jsonl(rows_path, selected)
    manifest = {
        "status": STATUS,
        "model_evaluation_status": "NOT_RUN",
        "selection_uses_model_outcomes": False,
        "prior_invalid_run_exists": True,
        "prior_invalid_run_hashes": invalid_hashes,
        "prior_invalid_data_manifest_sha256": EXPECTED_QUESTION_PANEL_MANIFEST_SHA256,
        "question_panel_frozen_from_rows_sha256": EXPECTED_QUESTION_PANEL_ROWS_SHA256,
        "repair_scope": ["fresh_disjoint_filler", "special_id_sanitization"],
        "tasks": list(TASKS),
        "rows_per_task": ROWS_PER_TASK,
        "rows": len(selected),
        "task_counts": task_counts,
        "target_length": TARGET_LENGTH,
        "evidence_end_ceiling": EVIDENCE_END_CEILING,
        "minimum_evidence_to_query_tokens": MIN_EVIDENCE_TO_QUERY_TOKENS,
        "filler_special_token_policy": "replace every tokenizer special id one-for-one with token 198 newline",
        "filler_source_start_row": packed["source"]["start_row"],
        "filler_source_consumed_row_range": packed["source"]["consumed_row_range"],
        "filler_source_document_set_sha256": packed["source_document_set_sha256"],
        "filler_source_disjoint_from_nll_by_row_range": True,
        "source_parquet_sha256": EXPECTED_SOURCE_SHA256,
        "nll_source_row_range": list(NLL_SOURCE_RANGE),
        "source_text_sha256_intersection_count": 0,
        "nll_range_source_text_set_sha256": canonical_sha256(sorted(old_hashes)),
        "fresh_range_source_text_set_sha256": canonical_sha256(sorted(fresh_hashes)),
        "tokenizer_all_special_ids": sorted(int(value) for value in tokenizer.all_special_ids),
        "filler_replacement_text": "\n",
        "filler_replacement_token_id": 198,
        "selection": "canonical source-row hash order; first model-free admissible rows",
        "longbench_main_zip_sha256": EXPECTED_MAIN_ZIP_SHA256,
        "longbench_data_zip_sha256": EXPECTED_DATA_ZIP_SHA256,
        "packed_manifest_sha256": packed["manifest_sha256"],
        "packed_rows_sha256": packed["file"]["sha256"],
        "config_sha256": packed["config_sha256"],
        "tokenizer_files": tokenizer_file_receipts(checkpoint),
        "file": {"path": rows_path.name, "sha256": sha256_file(rows_path)},
        "script_sha256": sha256_file(Path(__file__)),
        "metric_scope": "derived far-evidence LongBench QA with official-style token F1; not official LongBench",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--packed-root", type=Path, required=True)
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--longbench-main-zip", type=Path, required=True)
    parser.add_argument("--longbench-data-zip", type=Path, required=True)
    parser.add_argument("--question-panel-root", type=Path, required=True)
    parser.add_argument("--invalid-run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.checkpoint, args.packed_root, args.source_parquet,
                           args.longbench_main_zip, args.longbench_data_zip, args.question_panel_root,
                           args.invalid_run_root, args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
