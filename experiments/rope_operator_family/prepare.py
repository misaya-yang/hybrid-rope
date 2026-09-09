"""Prepare natural-text inputs for one model and one experiment configuration."""
from __future__ import annotations

import hashlib
import heapq
import json
from pathlib import Path

from .study import digest, write_json


def documents(source: str | Path):
    """JSONL text/source_id records, plain text files, or a directory of .txt."""
    source = Path(source)
    if source.is_dir():
        for path in sorted(source.rglob("*.txt")):
            yield {"text": path.read_text(), "source_id": str(path.relative_to(source))}
    elif source.suffix == ".jsonl":
        with source.open() as stream:
            for index, line in enumerate(stream):
                if line.strip():
                    row = json.loads(line)
                    if not isinstance(row.get("text"), str):
                        raise ValueError(f"source line {index + 1} needs a text string")
                    yield row
    elif source.suffix == ".txt":
        yield {"text": source.read_text(), "source_id": source.name}
    else:
        raise ValueError("source must be a JSONL, a .txt, or a directory of .txt documents")


def prepare_text(source, tokenizer, output, calibration_documents=32, validation_documents=8,
                 calibration_length=2048, evaluation_length=8192, seed=42):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise FileExistsError("prepared data already exists; choose it as input or use a new output directory")
    if min(calibration_documents, validation_documents, calibration_length, evaluation_length) <= 0:
        raise ValueError("document counts and lengths must be positive")
    if evaluation_length < calibration_length:
        raise ValueError("evaluation_length should cover the calibration window")
    candidates, long_candidates, seen, seen_sources = [], [], set(), set()

    def retain(heap, score, item, count):
        entry = (-score, item["text_sha256"], item)
        if len(heap) < count:
            heapq.heappush(heap, entry)
        elif entry[:2] > heap[0][:2]:
            heapq.heapreplace(heap, entry)

    scanned = 0
    for row in documents(source):
        scanned += 1
        text = row["text"]
        sha = hashlib.sha256(" ".join(text.split()).encode()).hexdigest()
        if sha in seen:
            continue
        seen.add(sha)
        ids = tokenizer(text, add_special_tokens=False, truncation=True,
                        max_length=evaluation_length + 1)["input_ids"]
        if len(ids) < calibration_length:
            continue
        source_id = str(row.get("source_id", row.get("id", sha)))
        if source_id in seen_sources:
            continue
        seen_sources.add(source_id)
        item = {"source_id": source_id, "text_sha256": sha, "input_ids": ids}
        score = int(hashlib.sha256(f"{seed}:{sha}".encode()).hexdigest(), 16)
        retain(candidates, score, item, calibration_documents + validation_documents)
        if len(ids) >= evaluation_length + 1:
            retain(long_candidates, score, item, validation_documents)
    validation = [entry[2] for entry in sorted(long_candidates, key=lambda x: -x[0])]
    validation_sources = {row["source_id"] for row in validation}
    calibration = []
    selected_sources = set()
    for _, _, row in sorted(candidates, key=lambda x: -x[0]):
        if row["source_id"] in validation_sources or row["source_id"] in selected_sources:
            continue
        calibration.append(row)
        selected_sources.add(row["source_id"])
        if len(calibration) == calibration_documents:
            break
    # Enforce source-level separation even if a JSONL contains multiple rows/book.
    if len(validation_sources) != len(validation) or len(calibration) < calibration_documents or len(validation) < validation_documents:
        raise ValueError(f"source has insufficient distinct eligible documents: found calibration={len(calibration)}, validation={len(validation_sources)}; requested {calibration_documents}/{validation_documents}")
    capture_rows, evaluation_rows = [], []
    for split, selected in (("calibration", calibration), ("validation", validation)):
        for index, row in enumerate(selected):
            common = {key: row[key] for key in ("source_id", "text_sha256")}
            common.update(id=f"{split}_{index:04d}", split=split)
            capture_rows.append({**common, "input_ids": row["input_ids"][:calibration_length]})
            if split == "validation":
                evaluation_rows.append({**common, "input_ids": row["input_ids"][:evaluation_length + 1]})
    for name, rows in (("capture.jsonl", capture_rows), ("evaluate.jsonl", evaluation_rows)):
        with (output / name).open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    tokenizer_root = Path(tokenizer.name_or_path)
    tokenizer_files = {name: digest(tokenizer_root / name) for name in
                       ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt")
                       if (tokenizer_root / name).is_file()}
    manifest = dict(status="complete", source=str(Path(source).resolve()), scanned_documents=scanned,
                    calibration_documents=len(calibration), validation_documents=len(validation),
                    calibration_length=calibration_length, evaluation_length=evaluation_length,
                    seed=seed, tokenizer_name=tokenizer.name_or_path, tokenizer_files=tokenizer_files,
                    files={name: digest(output / name) for name in ("capture.jsonl", "evaluate.jsonl")})
    write_json(output / "manifest.json", manifest)
    return manifest
