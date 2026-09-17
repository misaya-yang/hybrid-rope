#!/usr/bin/env python3
"""Freeze 3 x 80 independent LongBench Natural-QA rows inside OLMo Native 4K."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile


TASKS = ("2wikimqa", "hotpotqa", "qasper")
ROWS_PER_TASK = 80
NATIVE_LENGTH = 4096
REVISION = "2e00731f8d0bff23dc4325161044d0ed8af94c1e"
CONFIG_FILES = ("dataset2prompt.json", "dataset2maxlen.json")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                      separators=(",", ":")).encode()).hexdigest()


def zip_member(archive: zipfile.ZipFile, suffix: str) -> str:
    matches = [name for name in archive.namelist()
               if name == suffix or name.endswith("/" + suffix)]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one archive member ending in {suffix}")
    return matches[0]


def ensure_config(root: Path, *, download: bool) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in CONFIG_FILES:
        path = root / name
        if path.is_file():
            continue
        if not download:
            raise FileNotFoundError(path)
        url = f"https://raw.githubusercontent.com/THUDM/LongBench/{REVISION}/config/{name}"
        temporary = path.with_name(path.name + ".incomplete")
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as stream:
            stream.write(response.read())
        temporary.replace(path)


def _ids(tokenizer, content: str) -> list[int]:
    value = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True, tokenize=True,
    )
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(token) for token in value]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--exclude-panel", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--download-config", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Natural-QA confirmation assets are immutable; use a new --out")
    ensure_config(args.config_root, download=args.download_config)
    prompts = json.loads((args.config_root / "dataset2prompt.json").read_text())
    budgets = json.loads((args.config_root / "dataset2maxlen.json").read_text())
    excluded = set()
    if args.exclude_panel:
        excluded = {
            str(json.loads(line).get("source_row_sha256"))
            for line in args.exclude_panel.read_text().splitlines() if line.strip()
        }
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    rows = []
    with zipfile.ZipFile(args.archive) as archive:
        for task in TASKS:
            source = [json.loads(line) for line in archive.read(
                zip_member(archive, f"data/{task}.jsonl")
            ).decode().splitlines() if line.strip()]
            eligible = []
            for source_index, item in enumerate(source):
                source_hash = canonical_sha256(item)
                if source_hash in excluded:
                    continue
                content = str(prompts[task]).format(
                    context=item["context"], input=item["input"],
                )
                prompt_ids = _ids(tokenizer, content)
                budget = int(budgets[task])
                if len(prompt_ids) + budget > NATIVE_LENGTH:
                    continue
                eligible.append((source_hash, source_index, item, prompt_ids, budget))
            eligible.sort(key=lambda item: (item[0], item[1]))
            if len(eligible) < ROWS_PER_TASK:
                raise ValueError(f"{task} has only {len(eligible)} independent Native-4K rows")
            for source_hash, source_index, item, prompt_ids, budget in eligible[:ROWS_PER_TASK]:
                references = [str(value) for value in item["answers"]]
                context_hash = hashlib.sha256(item["context"].encode()).hexdigest()
                rows.append({
                    "row_id": f"nativeqa_{task}_{source_index:04d}",
                    "task": task,
                    "family": "longbench_natural_qa_native_confirm",
                    "length_cap": NATIVE_LENGTH,
                    "prompt_ids": prompt_ids,
                    "prompt_sha256": canonical_sha256(prompt_ids),
                    "input_tokens": len(prompt_ids),
                    "actual_length": len(prompt_ids),
                    "max_new_tokens": budget,
                    "references": references,
                    "source_row_index": source_index,
                    "source_row_sha256": source_hash,
                    "source_document_id": str(item.get("_id", source_hash)),
                    "document_cluster_id": context_hash,
                    "source_context_sha256": context_hash,
                    "selection_uses_model_outputs": False,
                    "content_truncation": False,
                })
    counts = Counter(row["task"] for row in rows)
    if counts != Counter({task: ROWS_PER_TASK for task in TASKS}):
        raise AssertionError("Natural-QA task counts drifted")
    args.out.mkdir(parents=True)
    inputs = args.out / "inputs.jsonl"
    with inputs.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "status": "CPU_PREPARED",
        "contract": "olmo_native4k_longbench_naturalqa_3x80_independent_v1",
        "rows": len(rows), "rows_by_task": dict(counts),
        "native_length": NATIVE_LENGTH,
        "minimum_input_tokens": min(row["input_tokens"] for row in rows),
        "maximum_input_tokens": max(row["input_tokens"] for row in rows),
        "inputs_sha256": sha256(inputs),
        "source_archive_sha256": sha256(args.archive),
        "official_config_revision": REVISION,
        "official_config_sha256": {name: sha256(args.config_root / name) for name in CONFIG_FILES},
        "excluded_source_rows": len(excluded),
        "selection": "hash-order among complete untruncated rows fitting prompt plus official output budget in 4096 tokens",
        "selection_uses_model_outputs": False,
        "scope": "New Native-window source questions; three-task Natural-QA confirmation, not full LongBench.",
        "gpu_execution": False,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
