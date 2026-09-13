#!/usr/bin/env python3
"""Expand PG19 to 512 train books while reusing legacy validation/test assets.

The default is a local dry plan. ``--execute`` performs the bounded official
GCS listing and downloads only missing train books with two network workers.
Existing legacy files are symlinked, never copied or deleted. No MD5/SHA scan
is performed; resumable downloads are checked against listing/HTTP byte sizes.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from experiments.evq_recovery.acquire import PG_BUCKET, list_books


def write_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def child_text(node, name: str) -> str | None:
    for child in node:
        if child.tag.rsplit("}", 1)[-1] == name:
            return child.text
    return None


def paginated_train_books(count: int, max_pages: int = 20) -> list[dict]:
    """Apply acquire.list_books filtering across bounded GCS marker pages."""
    books = []
    marker = None
    for page in range(max_pages):
        query = {"prefix": "train/", "max-keys": "1000"}
        if marker:
            query["marker"] = marker
        url = PG_BUCKET + "?" + urllib.parse.urlencode(query)
        with urllib.request.urlopen(url, timeout=45) as response:
            root = ET.fromstring(response.read())
        contents = [node for node in root.iter() if node.tag.rsplit("}", 1)[-1] == "Contents"]
        page_keys = []
        for item in contents:
            key = child_text(item, "Key")
            size_text = child_text(item, "Size")
            etag = child_text(item, "ETag")
            if key is None or size_text is None:
                continue
            page_keys.append(key)
            size = int(size_text)
            if key.endswith(".txt") and size >= 180_000:
                books.append({"key": key, "bytes": size, "etag": (etag or "").strip('"')})
                if len(books) >= count:
                    return books[:count]
        truncated = next((node.text for node in root.iter() if node.tag.rsplit("}", 1)[-1] == "IsTruncated"), "false")
        if str(truncated).lower() != "true":
            break
        next_marker = next((node.text for node in root.iter() if node.tag.rsplit("}", 1)[-1] == "NextMarker"), None)
        marker = next_marker or (page_keys[-1] if page_keys else None)
        if not marker:
            raise RuntimeError(f"truncated GCS page {page + 1} supplied no continuation marker")
    if len(books) < count:
        raise ValueError(f"bounded official listing found {len(books)} qualifying train books; need {count}")
    return books[:count]


def official_train_books(count: int) -> list[dict]:
    for attempt in range(5):
        try:
            try:
                return list_books("train", count)
            except ValueError:
                return paginated_train_books(count)
        except (OSError, http.client.HTTPException):
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def resume_download(url: str, target: Path, expected_bytes: int, attempts: int = 4) -> dict:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size == expected_bytes:
        return {"status": "REUSED_COMPLETE", "path": str(target), "bytes": expected_bytes, "url": url}
    part = target.with_name(target.name + ".part")
    for attempt in range(attempts):
        offset = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": "OLMo-recovery-PG19-expansion", "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                content_range = response.headers.get("Content-Range")
                if offset and (response.status != 206 or not content_range or not content_range.startswith(f"bytes {offset}-")):
                    raise ValueError(f"server did not honor Range offset {offset}: {content_range}")
                advertised = int(content_range.rsplit("/", 1)[1]) if content_range else int(response.headers.get("Content-Length", expected_bytes))
                if advertised != expected_bytes:
                    raise ValueError(f"HTTP/listing size mismatch: {advertised} != {expected_bytes}")
                with part.open("ab" if offset else "wb") as output:
                    while True:
                        try:
                            block = response.read(256 * 1024)
                        except http.client.IncompleteRead as error:
                            block = error.partial
                        if not block:
                            break
                        output.write(block)
            size = part.stat().st_size
            if size == expected_bytes:
                os.replace(part, target)
                return {"status": "DOWNLOADED", "path": str(target), "bytes": size, "url": url}
            if size > expected_bytes:
                raise ValueError(f"partial file exceeds listing size: {size} > {expected_bytes}")
        except Exception:
            if attempt + 1 == attempts:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def ensure_link(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise ValueError(f"existing symlink targets another asset: {destination}")
        return
    if destination.exists():
        if destination.resolve() != source.resolve():
            raise FileExistsError(destination)
        return
    destination.symlink_to(source.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-books", type=int, default=512)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.train_books != 512:
        raise ValueError("this registered expansion targets exactly 512 train books")
    legacy = args.legacy_sources.resolve()
    legacy_manifest_path = legacy / "pg19_books.json"
    legacy_rows = json.loads(legacy_manifest_path.read_text())
    existing_train = [row for row in legacy_rows if row["split"] == "train"]
    heldout = [row for row in legacy_rows if row["split"] in ("validation", "test")]
    heldout_counts = {split: sum(row["split"] == split for row in heldout) for split in ("validation", "test")}
    if heldout_counts != {"validation": 24, "test": 24}:
        raise ValueError(f"legacy heldout contract differs: {heldout_counts}")
    plan = {"status": "DRY_PLAN" if not args.execute else "STARTING", "target_train_books": 512,
            "legacy_train_books": len(existing_train), "new_train_books_expected": 512 - len(existing_train),
            "validation_books": 24, "test_books": 24, "network_workers": 2,
            "integrity": "GCS listing size plus HTTP Content-Length/Content-Range; no MD5/SHA scan",
            "mutation": "new output only; legacy files symlinked, never copied or deleted"}
    print(json.dumps(plan, sort_keys=True), flush=True)
    if not args.execute:
        return
    if len(existing_train) > 512:
        raise ValueError("legacy train set already exceeds registered target")
    listed = official_train_books(512)
    by_key = {row["key"]: row for row in listed}
    selected = []
    for row in existing_train:
        selected.append({**by_key.get(row["key"], row), "split": "train"})
    for row in listed:
        if row["key"] not in {item["key"] for item in selected}:
            selected.append({**row, "split": "train"})
        if len(selected) == 512:
            break
    if len(selected) != 512 or len({row["key"] for row in selected}) != 512:
        raise ValueError("official listing could not extend the legacy selection to 512 unique train books")
    output = args.output.resolve()
    (output / "pg19").mkdir(parents=True, exist_ok=True)
    existing_keys = {row["key"] for row in existing_train}
    receipts = []
    for row in existing_train + heldout:
        source = legacy / "pg19" / row["key"]
        destination = output / "pg19" / row["key"]
        ensure_link(source, destination)
        receipts.append({"status": "SYMLINKED_LEGACY", "key": row["key"], "path": str(destination),
                         "bytes": destination.stat().st_size, "split": row["split"]})
    jobs = [row for row in selected if row["key"] not in existing_keys]
    failures = []
    downloaded = 0
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(resume_download, PG_BUCKET + row["key"], output / "pg19" / row["key"], int(row["bytes"])): row for row in jobs}
        for future in as_completed(futures):
            row = futures[future]
            try:
                receipt = future.result(); receipt.update(key=row["key"], split="train"); receipts.append(receipt); downloaded += 1
            except Exception as error:
                failures.append({"key": row["key"], "error": repr(error), "expected_bytes": int(row["bytes"])})
            print(json.dumps({"completed_network_jobs": downloaded + len(failures), "total_network_jobs": len(jobs),
                              "downloaded": downloaded, "failed": len(failures), "last_key": row["key"]}), flush=True)
    manifest_rows = selected + heldout
    write_json(output / "pg19_books.json", manifest_rows)
    receipt = {"status": "COMPLETE" if not failures else "PARTIAL_DOWNLOAD_FAILURE",
               "asset_identity_policy": "user_attested_clone/no_sha_validation",
               "train_books": len(selected), "validation_books": 24, "test_books": 24,
               "legacy_reused": len(existing_train) + len(heldout), "network_requested": len(jobs),
               "network_completed": downloaded, "network_failed": len(failures),
               "total_bytes_present": sum(int(item["bytes"]) for item in receipts),
               "receipts": receipts, "failures": failures,
               "layout": "pg19_books.json keys resolve under pg19/{split}/{file}; compatible with prepare_pg19_multiscale"}
    write_json(output / "expansion_receipt.json", receipt)
    print(json.dumps({key: receipt[key] for key in ("status", "train_books", "legacy_reused", "network_completed", "network_failed", "total_bytes_present")}, sort_keys=True))


if __name__ == "__main__":
    main()
