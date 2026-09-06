#!/usr/bin/env python3
"""Download and freeze the real target-free retrofit data inputs.

This script intentionally downloads metadata, benchmark data, PG-19
validation/test books, and the OLMo-2 tokenizer/config only.  It never asks
Hugging Face for model weights and it never tokenizes text.  Raw assets belong
under a caller-provided persistent data root, not in the repository.

The resulting manifest is deliberately left in
``DOWNLOAD_COMPLETE_TOKEN_MANIFEST_PENDING`` state.  Token manifests require
the target model tokenizer and are built later on an authorised machine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


STATUS = "DOWNLOAD_COMPLETE_TOKEN_MANIFEST_PENDING"
LONGBENCH_REPOSITORY = "THUDM/LongBench"
LONGBENCH_DATASET_REVISION = "5e628be450b7e67fb7ae6e201bd6d8f7056f7672"
LONGBENCH_GITHUB_REVISION = "2e00731f8d0bff23dc4325161044d0ed8af94c1e"
PG19_REPOSITORY = "deepmind/pg19"
PG19_DATASET_REVISION = "4d28bd77e66947ad3835cf78ed7aaeb4dd87ad8b"
OLMO_REPOSITORY = "allenai/OLMo-2-0425-1B-Instruct"
OLMO_MODEL_REVISION = "48d788eca847d4d7548f375ad03d3c9312f6139e"

REQUESTED_TASKS = (
    "qasper",
    "narrativeqa",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "gov_report",
)
# LongBench's published LongBench-E list has no narrativeqa_e split.
PUBLISHED_E_TASKS = (
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "gov_report",
)
OLMO_METADATA_FILES = (
    "config.json",
    "generation_config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def composite_sha256(files: Iterable[tuple[str, dict[str, Any]]]) -> str:
    payload = json.dumps(
        [{"path": name, **receipt} for name, receipt in sorted(files)],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_receipt(path: Path, *, relative_to: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def atomic_json(path: Path, value: Any) -> None:
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
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def remote_content_length(url: str) -> int | None:
    request = urllib.request.Request(
        url,
        method="HEAD",
        headers={"User-Agent": "hybrid-rope-target-free-preflight/1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            value = response.headers.get("Content-Length")
    except Exception:
        return None
    try:
        return None if value is None else int(value)
    except ValueError:
        return None


def download(url: str, destination: Path) -> Path:
    """Download one immutable asset atomically, reusing an existing file."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_size = remote_content_length(url)
    if destination.is_file() and destination.stat().st_size > 0:
        if expected_size is None or destination.stat().st_size == expected_size:
            return destination
        destination.unlink()
    temporary = destination.with_name(destination.name + ".incomplete")
    if temporary.exists():
        temporary.unlink()
    try:
        for _attempt in range(8):
            start = temporary.stat().st_size if temporary.exists() else 0
            headers = {"User-Agent": "hybrid-rope-target-free-preflight/1"}
            if start > 0:
                headers["Range"] = f"bytes={start}-"
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                response_size = response.headers.get("Content-Length")
                if response_size is not None and start == 0:
                    expected_size = int(response_size)
                status = int(getattr(response, "status", 200))
                if start > 0 and status == 200:
                    # The server ignored Range; restart rather than duplicate.
                    temporary.unlink()
                    start = 0
                mode = "ab" if start > 0 else "wb"
                with temporary.open(mode) as handle:
                    shutil.copyfileobj(response, handle, length=8 << 20)
                    handle.flush()
                    os.fsync(handle.fileno())
            current = temporary.stat().st_size
            if expected_size is None or current >= expected_size:
                break
            if current <= start:
                raise IOError(f"download made no progress for {url}")
        if expected_size is not None and temporary.stat().st_size != expected_size:
            raise IOError(
                f"incomplete download for {url}: got {temporary.stat().st_size} "
                f"bytes, expected {expected_size}"
            )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def hf_url(repo_type: str, repository: str, revision: str, filename: str) -> str:
    prefix = "datasets" if repo_type == "dataset" else ""
    root = "https://huggingface.co/" + (prefix + "/" if prefix else "") + repository
    return f"{root}/resolve/{revision}/{filename}"


def github_raw_url(revision: str, filename: str) -> str:
    return f"https://raw.githubusercontent.com/THUDM/LongBench/{revision}/{filename}"


def json_row_count(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        value = json.loads(text)
        if not isinstance(value, list):
            raise ValueError(f"expected JSON list in {path}")
        return len(value)
    return sum(1 for line in text.splitlines() if line.strip())


def safe_zip_member(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"unsafe LongBench archive member: {name}")
    return path


def extract_selected_longbench(
    archive: Path,
    output_root: Path,
    *,
    task_names: Iterable[str],
    receipt_root: Path,
) -> dict[str, dict[str, Any]]:
    selected = {
        f"{task}.jsonl": (task, "longbench") for task in task_names
    }
    selected.update(
        {f"{task}_e.jsonl": (task, "longbench_e") for task in task_names}
    )
    records: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(archive) as handle:
        members = {}
        for member in handle.infolist():
            path = safe_zip_member(member.filename)
            members[path.name] = member
        for filename, (task, family) in selected.items():
            member = members.get(filename)
            if member is None:
                records.setdefault(family, {})[task] = {
                    "available": False,
                    "reason": "not published in the pinned LongBench archive",
                }
                continue
            destination = output_root / family / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".incomplete")
            if temporary.exists():
                temporary.unlink()
            with handle.open(member) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target, length=8 << 20)
                target.flush()
                os.fsync(target.fileno())
            temporary.replace(destination)
            records.setdefault(family, {})[task] = {
                "available": True,
                "rows": json_row_count(destination),
                **file_receipt(destination, relative_to=receipt_root),
                "archive_member": member.filename,
            }
    return records


def download_longbench(root: Path, revision: str) -> dict[str, Any]:
    output = root / "longbench"
    output.mkdir(parents=True, exist_ok=True)
    archive = download(
        hf_url("dataset", LONGBENCH_REPOSITORY, revision, "data.zip"),
        output / "data.zip",
    )
    config_root = output / "official_config"
    config_files = {}
    for filename in (
        "LongBench/LICENSE",
        "LongBench/README.md",
        "LongBench/config/dataset2maxlen.json",
        "LongBench/config/dataset2prompt.json",
    ):
        destination = config_root / Path(filename).name
        download(github_raw_url(LONGBENCH_GITHUB_REVISION, filename), destination)
        config_files[Path(filename).name] = file_receipt(
            destination, relative_to=root
        )
    extracted = extract_selected_longbench(
        archive,
        output / "extracted",
        task_names=REQUESTED_TASKS,
        receipt_root=root,
    )
    return {
        "repository": LONGBENCH_REPOSITORY,
        "dataset_revision": revision,
        "archive": file_receipt(archive, relative_to=root),
        "license": "MIT for the LongBench repository; constituent source licenses are acknowledged in its README",
        "official_config": config_files,
        "tasks": extracted,
        "longbench_e_unpublished_tasks": [
            task for task in REQUESTED_TASKS if task not in PUBLISHED_E_TASKS
        ],
    }


def download_pg19(root: Path, revision: str) -> dict[str, Any]:
    output = root / "pg19"
    metadata_root = output / "metadata"
    books_root = output / "books"
    metadata_root.mkdir(parents=True, exist_ok=True)
    books_root.mkdir(parents=True, exist_ok=True)
    split_records: dict[str, Any] = {}
    all_book_records: list[dict[str, Any]] = []
    for split in ("validation", "test"):
        list_path = download(
            hf_url("dataset", PG19_REPOSITORY, revision, f"data/{split}_files.txt"),
            metadata_root / f"{split}_files.txt",
        )
        names = [
            line.strip()
            for line in list_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        split_books = []
        for name in sorted(names):
            safe_name = PurePosixPath(name)
            if safe_name.is_absolute() or ".." in safe_name.parts:
                raise RuntimeError(f"unsafe PG-19 source path: {name}")
            relative_name = safe_name.as_posix()
            destination = books_root / split / Path(relative_name).name
            download(
                "https://storage.googleapis.com/deepmind-gutenberg/" + relative_name,
                destination,
            )
            receipt = file_receipt(destination, relative_to=root)
            record = {
                "split": split,
                "source_name": relative_name,
                **receipt,
            }
            split_books.append(record)
            all_book_records.append(record)
        split_records[split] = {
            "file_list": file_receipt(list_path, relative_to=root),
            "rows": len(split_books),
            "books": split_books,
        }
    metadata_csv = download(
        "https://storage.googleapis.com/deepmind-gutenberg/metadata.csv",
        metadata_root / "metadata.csv",
    )
    return {
        "repository": PG19_REPOSITORY,
        "dataset_revision": revision,
        "license": "Apache-2.0 (Hugging Face dataset card)",
        "metadata": file_receipt(metadata_csv, relative_to=root),
        "splits": split_records,
        "rows": sum(int(value["rows"]) for value in split_records.values()),
        "books": all_book_records,
    }


def download_olmo_metadata(root: Path, revision: str) -> dict[str, Any]:
    output = root / "olmo2_tokenizer_config"
    output.mkdir(parents=True, exist_ok=True)
    files: list[tuple[str, dict[str, Any]]] = []
    for filename in OLMO_METADATA_FILES:
        destination = download(
            hf_url("model", OLMO_REPOSITORY, revision, filename),
            output / filename,
        )
        receipt = file_receipt(destination, relative_to=root)
        files.append((filename, receipt))
    forbidden = output / "model.safetensors"
    if forbidden.exists():
        raise RuntimeError("refusing to keep a model weight in the metadata root")
    return {
        "repository": OLMO_REPOSITORY,
        "model_revision": revision,
        "license": "Apache-2.0 (model card)",
        "files": dict(files),
        "composite_sha256": composite_sha256(files),
        "weights_downloaded": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Persistent raw-data root outside the repository.",
    )
    parser.add_argument("--longbench-revision", default=LONGBENCH_DATASET_REVISION)
    parser.add_argument("--pg19-revision", default=PG19_DATASET_REVISION)
    parser.add_argument("--olmo-revision", default=OLMO_MODEL_REVISION)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": STATUS,
        "download_root": str(root),
        "tokenization_executed": False,
        "model_weights_downloaded": False,
        "sources": {
            "longbench": download_longbench(root, str(args.longbench_revision)),
            "pg19": download_pg19(root, str(args.pg19_revision)),
            "olmo2_tokenizer_config": download_olmo_metadata(
                root, str(args.olmo_revision)
            ),
        },
        "requested_tasks": list(REQUESTED_TASKS),
        "bucket_contract": {
            "retention": "total prompt plus reserve <= 1 * L_native",
            "near": "1 * L_native < total prompt plus reserve <= 2 * L_native",
            "far": "2 * L_native < total prompt plus reserve <= 4 * L_native",
            "qa_generation_reserve": 64,
            "gov_report_generation_budget_source": "LongBench dataset2maxlen.json",
        },
        "pending": [
            "target tokenizer token manifest",
            "relative-Native length bucket counts",
            "PG-19 tokenized nested anchors",
        ],
    }
    atomic_json(root / "download_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
