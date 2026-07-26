#!/usr/bin/env python3
"""Materialize the deterministic OLMo-2 stage-1 prefix for local training.

The official OLMo loader concatenates all configured raw uint32 token files,
chunks each file independently into 4096-token instances, then shuffles the
global instance indices with NumPy PCG64(seed=6198).  This script reconstructs
that order and fetches only the instances needed through step 1000.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import threading
import time
from pathlib import Path
from typing import Any, Generator, NamedTuple

import numpy as np
import requests
import yaml

from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.contract import (
    FIRST_GATE_STEPS,
    FIRST_GATE_TOKENS,
    GLOBAL_BATCH_SEQUENCES,
    MODEL_CONTRACT,
    OFFICIAL_CONFIG_SHA256,
    SEED,
    SEQUENCE_LENGTH,
    TOKENIZER_MARKERS,
    sha256_file,
    sha256_json,
)


USER_AGENT = "evq-olmo-data-preflight/1.0"
UINT32_BYTES = np.dtype(np.uint32).itemsize
CHUNK_BYTES = SEQUENCE_LENGTH * UINT32_BYTES
THREAD_LOCAL = threading.local()
EXPECTED_CONFIGURED_PATHS = 1_122
EXPECTED_UNIQUE_URLS = 1_117


class RepetitionTuple(NamedTuple):
    start: int
    end: int
    period: int
    times: int


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def get_session() -> requests.Session:
    session = getattr(THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT})
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=4, pool_maxsize=4, max_retries=0
        )
        session.mount("https://", adapter)
        THREAD_LOCAL.session = session
    return session


def https_url(url: str) -> str:
    if url.startswith("http://olmo-data.org/"):
        return "https://" + url[len("http://") :]
    return url


def range_get(url: str, start: int, end: int, *, attempts: int = 8) -> bytes:
    if start < 0 or end < start:
        raise ValueError(f"invalid byte range {start}-{end}")
    url = https_url(url)
    expected = end - start + 1
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = get_session().get(
                url,
                headers={"Range": f"bytes={start}-{end}"},
                timeout=(15, 120),
            )
            if response.status_code != 206:
                raise RuntimeError(
                    f"{url}: expected HTTP 206, received {response.status_code}"
                )
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {start}-{end}/"):
                raise RuntimeError(
                    f"{url}: unexpected Content-Range {content_range!r}"
                )
            if len(response.content) != expected:
                raise RuntimeError(
                    f"{url}: received {len(response.content)} bytes, expected {expected}"
                )
            return response.content
        except Exception as exc:  # network retry is explicit and bounded
            error = exc
            if attempt + 1 < attempts:
                time.sleep(min(30.0, 0.5 * (2**attempt)))
    assert error is not None
    raise error


def remote_size(url: str) -> int:
    response = get_session().get(
        https_url(url),
        headers={"Range": "bytes=0-0"},
        timeout=(15, 120),
    )
    if response.status_code != 206:
        raise RuntimeError(f"{url}: size probe returned HTTP {response.status_code}")
    content_range = response.headers.get("Content-Range", "")
    try:
        total = int(content_range.rsplit("/", 1)[1])
    except (IndexError, ValueError) as exc:
        raise RuntimeError(f"{url}: invalid Content-Range {content_range!r}") from exc
    if total <= 0 or total % UINT32_BYTES:
        raise RuntimeError(f"{url}: invalid uint32 byte size {total}")
    return total


def load_official_paths(config_path: Path) -> list[str]:
    if sha256_file(config_path) != OFFICIAL_CONFIG_SHA256:
        raise RuntimeError("official config hash mismatch")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    required = {
        "seed": SEED,
        "global_train_batch_size": GLOBAL_BATCH_SEQUENCES,
    }
    for field, expected in required.items():
        if config.get(field) != expected:
            raise RuntimeError(
                f"official config {field}={config.get(field)!r}, expected {expected!r}"
            )
    if config["model"]["max_sequence_length"] != SEQUENCE_LENGTH:
        raise RuntimeError("official sequence length drift")
    if config["data"]["memmap_dtype"] != "uint32":
        raise RuntimeError("official data dtype drift")
    if config["tokenizer"]["identifier"] != TOKENIZER_MARKERS["official_identifier"]:
        raise RuntimeError("official tokenizer identifier drift")
    paths = config["data"]["paths"]
    if not isinstance(paths, list) or not paths:
        raise RuntimeError("official data paths are missing")
    return [https_url(str(path)) for path in paths]


def probe_source_sizes(
    paths: list[str],
    *,
    workers: int,
    cached: dict[str, int] | None = None,
    cache_path: Path | None = None,
) -> dict[str, int]:
    sizes = dict(cached or {})
    missing = sorted(set(paths) - sizes.keys())
    if not missing:
        return sizes
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(remote_size, url): url for url in missing}
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            url = futures[future]
            sizes[url] = future.result()
            if completed % 50 == 0 or completed == len(missing):
                if cache_path is not None:
                    write_json(cache_path, sizes)
                elapsed = time.monotonic() - started
                print(
                    f"source-size progress {completed}/{len(missing)} "
                    f"({completed/max(elapsed, 1e-6):.1f} files/s)",
                    flush=True,
                )
    return sizes


def source_rows(paths: list[str], sizes: dict[str, int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    global_start = 0
    for order, url in enumerate(paths):
        size = int(sizes[url])
        instances = size // CHUNK_BYTES
        if instances <= 0:
            raise RuntimeError(f"{url}: no complete {SEQUENCE_LENGTH}-token instance")
        rows.append(
            {
                "order": order,
                "url": url,
                "bytes": size,
                "instances": instances,
                "discarded_tail_bytes": size - instances * CHUNK_BYTES,
                "global_start": global_start,
                "global_end": global_start + instances,
            }
        )
        global_start += instances
    if global_start >= np.iinfo(np.uint32).max:
        raise RuntimeError(
            f"official dataset has {global_start} instances, exceeding uint32 order"
        )
    return rows


def memory_available_bytes(
    *,
    proc_meminfo: Path = Path("/proc/meminfo"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> int | None:
    limits: list[int] = []
    if proc_meminfo.is_file():
        for line in proc_meminfo.read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                limits.append(int(line.split()[1]) * 1024)
                break
    maximum_path = cgroup_root / "memory.max"
    current_path = cgroup_root / "memory.current"
    if maximum_path.is_file() and current_path.is_file():
        maximum_text = maximum_path.read_text(encoding="ascii").strip()
        if maximum_text != "max":
            maximum = int(maximum_text)
            current = int(current_path.read_text(encoding="ascii").strip())
            limits.append(max(0, maximum - current))
    return min(limits) if limits else None


def generate_order_prefix(
    total_instances: int,
    needed_instances: int,
    *,
    seed: int,
) -> np.ndarray:
    if not 0 < needed_instances <= total_instances:
        raise ValueError(
            f"needed_instances={needed_instances}, total_instances={total_instances}"
        )
    required_bytes = total_instances * np.dtype(np.uint32).itemsize
    available_bytes = memory_available_bytes()
    if (
        available_bytes is not None
        and available_bytes < required_bytes * 2
    ):
        raise MemoryError(
            "exact OLMo permutation requires a full uint32 index array: "
            f"need a safe minimum of {required_bytes*2/2**30:.1f} GiB, "
            f"only {available_bytes/2**30:.1f} GiB is available"
        )
    indices = np.arange(total_instances, dtype=np.uint32)
    rng = np.random.Generator(np.random.PCG64(seed=seed))
    rng.shuffle(indices)
    prefix = indices[:needed_instances].copy()
    del indices
    return prefix


def map_indices(
    prefix: np.ndarray, rows: list[dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray]:
    ends = np.asarray([row["global_end"] for row in rows], dtype=np.uint64)
    starts = np.asarray([row["global_start"] for row in rows], dtype=np.uint64)
    source_ids = np.searchsorted(ends, prefix.astype(np.uint64), side="right")
    local_indices = prefix.astype(np.uint64) - starts[source_ids]
    return source_ids.astype(np.uint16), local_indices


def find_end_first_consecutive_true(array: np.ndarray) -> int:
    if not array[0]:
        return 0
    progress = np.cumsum(array)
    if progress[-1] == len(array):
        return len(array)
    locations = np.where(progress[:-1] == progress[1:])[0]
    return int(locations[0] + 1)


def find_start_last_consecutive_true(array: np.ndarray) -> int:
    reverse = find_end_first_consecutive_true(array[::-1])
    return len(array) - reverse if reverse > 0 else -1


def group_consecutive_values(array: np.ndarray) -> list[np.ndarray]:
    return list(np.split(array, np.where(np.diff(array) != 1)[0] + 1))


def find_periodic_sequences(
    array: np.ndarray, *, max_period: int, min_period: int = 1
) -> Generator[RepetitionTuple, None, None]:
    """Equivalent to OLMo's pinned instance-filter implementation."""
    mask_value = -1
    if bool((array == mask_value).sum()):
        raise ValueError("mask value appears in token array")
    max_period = min(max_period, len(array) // 3)
    for period in range(min_period, max_period + 1):
        padding = period - (len(array) % period)
        padded = np.pad(array, (0, padding), constant_values=mask_value)
        shaped = padded.reshape(-1, period)
        equal_previous = shaped == np.roll(shaped, shift=1, axis=0)
        rows, *_ = np.where(equal_previous.all(axis=1))
        if len(rows) == 0:
            continue
        for sequence in group_consecutive_values(rows):
            start_row = int(sequence[0])
            end_row = int(sequence[-1])
            start_offset = find_start_last_consecutive_true(
                equal_previous[start_row - 1]
            )
            start_offset = period - start_offset if start_offset > 0 else 0
            end_offset = find_end_first_consecutive_true(
                equal_previous[end_row + 1]
            )
            start = (start_row - 1) * period - start_offset
            end = (end_row + 1) * period + end_offset
            result = RepetitionTuple(
                start=start,
                end=end,
                period=period,
                times=(end - start) // period,
            )
            if result.times > 2:
                yield result


def official_instance_valid(tokens: np.ndarray) -> bool:
    for match in find_periodic_sequences(tokens, max_period=13, min_period=1):
        if match.times >= 32:
            return False
    return True


def fetch_instance(
    output_index: int,
    source_id: int,
    local_index: int,
    *,
    urls: list[str],
) -> tuple[int, bytes, bool, int, int]:
    start = int(local_index) * CHUNK_BYTES
    payload = range_get(urls[int(source_id)], start, start + CHUNK_BYTES - 1)
    tokens = np.frombuffer(payload, dtype=np.uint32)
    minimum = int(tokens.min())
    maximum = int(tokens.max())
    if minimum < 0 or maximum >= MODEL_CONTRACT["vocab_size"]:
        raise RuntimeError(
            f"sequence {output_index}: token range {minimum}..{maximum} "
            f"outside 0..{MODEL_CONTRACT['vocab_size']-1}"
        )
    return output_index, payload, official_instance_valid(tokens), minimum, maximum


def materialize(
    output_dir: Path,
    rows: list[dict[str, Any]],
    source_ids: np.ndarray,
    local_indices: np.ndarray,
    *,
    workers: int,
) -> dict[str, Any]:
    token_path = output_dir / "train_tokens.uint32.bin"
    bitmap_path = output_dir / "download_complete.uint8.bin"
    valid_path = output_dir / "instance_valid.uint8.bin"
    count = len(source_ids)
    expected_size = count * CHUNK_BYTES

    if not token_path.exists():
        with token_path.open("wb") as handle:
            handle.truncate(expected_size)
    if token_path.stat().st_size != expected_size:
        raise RuntimeError(
            f"{token_path}: size {token_path.stat().st_size} != {expected_size}"
        )
    if not bitmap_path.exists():
        with bitmap_path.open("wb") as handle:
            handle.truncate(count)
    if not valid_path.exists():
        with valid_path.open("wb") as handle:
            handle.truncate(count)

    bitmap = np.memmap(bitmap_path, dtype=np.uint8, mode="r+", shape=(count,))
    valid = np.memmap(valid_path, dtype=np.uint8, mode="r+", shape=(count,))
    pending = np.flatnonzero(bitmap != 1)
    urls = [row["url"] for row in rows]
    fd = os.open(token_path, os.O_RDWR)
    started = time.monotonic()
    minimum = MODEL_CONTRACT["vocab_size"]
    maximum = 0
    completed_now = 0
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            index_iterator = iter(int(value) for value in pending)
            futures: dict[concurrent.futures.Future, int] = {}

            def submit_one() -> bool:
                try:
                    index = next(index_iterator)
                except StopIteration:
                    return False
                future = pool.submit(
                    fetch_instance,
                    index,
                    int(source_ids[index]),
                    int(local_indices[index]),
                    urls=urls,
                )
                futures[future] = index
                return True

            for _ in range(max(workers * 4, 1)):
                if not submit_one():
                    break
            while futures:
                done, _ = concurrent.futures.wait(
                    futures,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    futures.pop(future)
                    index, payload, is_valid, row_min, row_max = future.result()
                    os.pwrite(fd, payload, index * CHUNK_BYTES)
                    valid[index] = 1 if is_valid else 0
                    bitmap[index] = 1
                    minimum = min(minimum, row_min)
                    maximum = max(maximum, row_max)
                    completed_now += 1
                    submit_one()
                    if (
                        completed_now % 1000 == 0
                        or completed_now == len(pending)
                    ):
                        bitmap.flush()
                        valid.flush()
                        elapsed = time.monotonic() - started
                        total_complete = int((bitmap == 1).sum())
                        rate = completed_now / max(elapsed, 1e-6)
                        eta = (len(pending) - completed_now) / max(
                            rate, 1e-6
                        )
                        print(
                            f"token progress {total_complete}/{count}; "
                            f"{rate:.1f} seq/s; ETA {eta/60:.1f} min",
                            flush=True,
                        )
    finally:
        os.close(fd)
        bitmap.flush()
        valid.flush()

    if not bool(np.all(bitmap == 1)):
        raise RuntimeError("materialization finished with incomplete instances")
    return {
        "path": token_path.name,
        "bytes": token_path.stat().st_size,
        "instances": count,
        "tokens": count * SEQUENCE_LENGTH,
        "sha256": sha256_file(token_path),
        "first_global_batch_sha256": first_global_batch_sha256(token_path),
        "instance_valid_path": valid_path.name,
        "instance_valid_sha256": sha256_file(valid_path),
        "invalid_instances": int((valid == 0).sum()),
        "token_min_observed_in_new_downloads": (
            minimum if completed_now else None
        ),
        "token_max_observed_in_new_downloads": (
            maximum if completed_now else None
        ),
    }


def first_global_batch_sha256(token_path: Path) -> str:
    with token_path.open("rb") as handle:
        payload = handle.read(GLOBAL_BATCH_SEQUENCES * CHUNK_BYTES)
    if len(payload) != GLOBAL_BATCH_SEQUENCES * CHUNK_BYTES:
        raise RuntimeError(f"{token_path}: incomplete first global batch")
    return hashlib.sha256(payload).hexdigest()


def array_values_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(array).tobytes(order="C")
    ).hexdigest()


def generate_order_stream_proof(
    output_dir: Path,
    official_config: Path,
    *,
    steps: int,
    spotcheck_count: int = 100,
) -> dict[str, Any]:
    """Regenerate PCG64 order and compare sampled bytes with official sources."""
    manifest_path = output_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "STREAM_VERIFIED":
        raise RuntimeError("cannot prove an unverified training stream")
    paths = load_official_paths(official_config)
    sizes_payload = json.loads(
        (output_dir / "source_sizes.json").read_text(encoding="utf-8")
    )
    sizes = {str(key): int(value) for key, value in sizes_payload.items()}
    rows = source_rows(paths, sizes)
    if (
        len(rows) != EXPECTED_CONFIGURED_PATHS
        or len(set(paths)) != EXPECTED_UNIQUE_URLS
    ):
        raise RuntimeError("official source path-count contract drift")
    needed_instances = steps * GLOBAL_BATCH_SEQUENCES
    expected_prefix = generate_order_prefix(
        rows[-1]["global_end"],
        needed_instances,
        seed=SEED,
    )
    order = manifest["order"]
    order_path = output_dir / order["indices_path"]
    saved_prefix = np.load(order_path, allow_pickle=False, mmap_mode="r")
    if saved_prefix.dtype != np.uint32 or saved_prefix.shape != (
        needed_instances,
    ):
        raise RuntimeError("saved order shape/dtype drift")
    if not np.array_equal(expected_prefix, saved_prefix):
        raise RuntimeError("saved order is not the exact PCG64(seed=6198) prefix")
    source_ids, local_indices = map_indices(expected_prefix, rows)
    saved_source_ids = np.load(
        output_dir / order["source_ids_path"],
        allow_pickle=False,
        mmap_mode="r",
    )
    saved_local_indices = np.load(
        output_dir / order["local_indices_path"],
        allow_pickle=False,
        mmap_mode="r",
    )
    if not np.array_equal(source_ids, saved_source_ids):
        raise RuntimeError("saved source mapping does not match regenerated order")
    if not np.array_equal(local_indices, saved_local_indices):
        raise RuntimeError("saved local mapping does not match regenerated order")

    stream = manifest["training_stream"]
    token_path = output_dir / stream["path"]
    valid_path = output_dir / stream["instance_valid_path"]
    if sha256_file(token_path) != stream["sha256"]:
        raise RuntimeError("training stream drift before source spotcheck")
    if sha256_file(valid_path) != stream["instance_valid_sha256"]:
        raise RuntimeError("instance-valid drift before source spotcheck")
    tokens = np.memmap(
        token_path,
        dtype=np.uint32,
        mode="r",
        shape=(needed_instances, SEQUENCE_LENGTH),
    )
    valid = np.memmap(
        valid_path,
        dtype=np.uint8,
        mode="r",
        shape=(needed_instances,),
    )
    rng = random.Random(SEED + 31_415)
    fixed = {0, needed_instances - 1}
    fixed.update(
        rng.sample(
            range(1, needed_instances - 1),
            min(spotcheck_count - len(fixed), needed_instances - 2),
        )
    )
    selected = sorted(fixed)

    def verify_one(index: int) -> dict[str, Any]:
        source_id = int(source_ids[index])
        local_index = int(local_indices[index])
        start = local_index * CHUNK_BYTES
        payload = range_get(
            paths[source_id],
            start,
            start + CHUNK_BYTES - 1,
        )
        local_payload = np.asarray(tokens[index]).tobytes(order="C")
        if payload != local_payload:
            raise RuntimeError(
                f"source byte spotcheck mismatch at output instance {index}"
            )
        is_valid = official_instance_valid(
            np.frombuffer(payload, dtype=np.uint32)
        )
        if int(valid[index]) != int(is_valid):
            raise RuntimeError(
                f"instance-valid spotcheck mismatch at output instance {index}"
            )
        return {
            "output_instance": index,
            "global_instance": int(expected_prefix[index]),
            "source_id": source_id,
            "local_instance": local_index,
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "instance_valid": bool(is_valid),
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        source_spotchecks = list(pool.map(verify_one, selected))
    proof = {
        "status": "ORDER_AND_SOURCE_PROOF_PASS",
        "official_config_sha256": OFFICIAL_CONFIG_SHA256,
        "seed": SEED,
        "order_algorithm": (
            "numpy.random.Generator(PCG64(seed)).shuffle("
            "arange(total_instances,dtype=uint32))"
        ),
        "total_instances": rows[-1]["global_end"],
        "needed_instances": needed_instances,
        "order_file_sha256": sha256_file(order_path),
        "order_values_sha256": array_values_sha256(saved_prefix),
        "source_manifest_sha256": manifest["source_manifest"]["sha256"],
        "training_stream_sha256": stream["sha256"],
        "instance_valid_sha256": stream["instance_valid_sha256"],
        "source_spotcheck_count": len(source_spotchecks),
        "source_spotchecks": source_spotchecks,
        "source_spotchecks_sha256": sha256_json(source_spotchecks),
    }
    proof_path = output_dir / "order_stream_proof.json"
    write_json(proof_path, proof)
    manifest["order_stream_proof"] = {
        "path": proof_path.name,
        "sha256": sha256_file(proof_path),
        "status": proof["status"],
    }
    write_json(manifest_path, manifest)
    return proof


def decode_spotcheck(
    output_dir: Path,
    tokenizer_path: Path,
    *,
    count: int = 100,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    manifest_path = output_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    instances = int(manifest["training_stream"]["instances"])
    token_path = output_dir / manifest["training_stream"]["path"]
    tokens = np.memmap(
        token_path,
        dtype=np.uint32,
        mode="r",
        shape=(instances, SEQUENCE_LENGTH),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True
    )
    if tokenizer.eos_token_id != TOKENIZER_MARKERS["eos_token_id"]:
        raise RuntimeError("tokenizer EOS marker drift")
    rng = random.Random(SEED)
    indices = sorted(rng.sample(range(instances), min(count, instances)))
    output = output_dir / "decode_spotcheck.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for index in indices:
            preview = tokens[index, :256].astype(np.int64).tolist()
            text = tokenizer.decode(preview, skip_special_tokens=False)
            record = {
                "instance_index": index,
                "token_prefix": preview[:16],
                "decoded_prefix": text[:400],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {
        "path": output.name,
        "samples": len(indices),
        "sha256": sha256_file(output),
    }


def validate_existing(
    output_dir: Path,
    official_config: Path,
    *,
    tokenizer_path: Path | None,
    steps: int,
) -> dict[str, Any]:
    """Validate the frozen stream without rewriting or re-sealing any artifact."""
    manifest_path = output_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "STREAM_VERIFIED":
        raise RuntimeError("dataset manifest is not STREAM_VERIFIED")
    expected_contract = {
        "tokenizer": "allenai_dolma2",
        "dtype": "uint32",
        "sequence_length": SEQUENCE_LENGTH,
        "seed": SEED,
        "global_batch_sequences": GLOBAL_BATCH_SEQUENCES,
        "steps": steps,
        "expected_tokens": steps * GLOBAL_BATCH_SEQUENCES * SEQUENCE_LENGTH,
        "official_config_sha256": OFFICIAL_CONFIG_SHA256,
    }
    contract = manifest["contract"]
    for key, expected in expected_contract.items():
        if contract.get(key) != expected:
            raise RuntimeError(
                f"frozen contract {key}={contract.get(key)!r}, "
                f"expected {expected!r}"
            )

    paths = load_official_paths(official_config)
    sizes_path = output_dir / "source_sizes.json"
    sizes_payload = json.loads(sizes_path.read_text(encoding="utf-8"))
    sizes = {str(key): int(value) for key, value in sizes_payload.items()}
    if set(sizes) != set(paths):
        raise RuntimeError("source-size cache does not match official paths")
    recomputed_rows = source_rows(paths, sizes)
    source_receipt = manifest["source_manifest"]
    rows_path = output_dir / source_receipt["path"]
    if sha256_file(rows_path) != source_receipt["sha256"]:
        raise RuntimeError("source manifest SHA-256 drift")
    stored_rows = json.loads(rows_path.read_text(encoding="utf-8"))
    if stored_rows != recomputed_rows:
        raise RuntimeError("source manifest content drift")
    source_expected = {
        "configured_paths": EXPECTED_CONFIGURED_PATHS,
        "unique_urls": EXPECTED_UNIQUE_URLS,
        "total_instances": recomputed_rows[-1]["global_end"],
    }
    if (
        len(recomputed_rows) != EXPECTED_CONFIGURED_PATHS
        or len(set(paths)) != EXPECTED_UNIQUE_URLS
    ):
        raise RuntimeError("official source path-count contract drift")
    for key, expected in source_expected.items():
        if source_receipt.get(key) != expected:
            raise RuntimeError(f"source manifest {key} drift")

    needed_instances = steps * GLOBAL_BATCH_SEQUENCES
    order = manifest["order"]
    order_path = output_dir / order["indices_path"]
    source_ids_path = output_dir / order["source_ids_path"]
    local_indices_path = output_dir / order["local_indices_path"]
    for path, hash_key in (
        (order_path, "indices_sha256"),
        (source_ids_path, "source_ids_sha256"),
        (local_indices_path, "local_indices_sha256"),
    ):
        if sha256_file(path) != order[hash_key]:
            raise RuntimeError(f"order artifact SHA-256 drift: {path.name}")
    prefix = np.load(order_path, allow_pickle=False, mmap_mode="r")
    source_ids = np.load(source_ids_path, allow_pickle=False, mmap_mode="r")
    local_indices = np.load(
        local_indices_path, allow_pickle=False, mmap_mode="r"
    )
    expected_arrays = (
        (prefix, np.dtype(np.uint32), "order indices"),
        (source_ids, np.dtype(np.uint16), "source ids"),
        (local_indices, np.dtype(np.uint64), "local indices"),
    )
    for array, dtype, name in expected_arrays:
        if array.dtype != dtype or array.shape != (needed_instances,):
            raise RuntimeError(f"{name} shape/dtype drift")
    remapped_source, remapped_local = map_indices(
        np.asarray(prefix), recomputed_rows
    )
    if not np.array_equal(remapped_source, source_ids):
        raise RuntimeError("saved source-id mapping drift")
    if not np.array_equal(remapped_local, local_indices):
        raise RuntimeError("saved local-index mapping drift")
    expected_mapping_sha = sha256_json(
        {
            "source_ids": order["source_ids_sha256"],
            "local_indices": order["local_indices_sha256"],
        }
    )
    if order.get("mapping_sha256") != expected_mapping_sha:
        raise RuntimeError("order mapping receipt drift")

    stream = manifest["training_stream"]
    token_path = output_dir / stream["path"]
    valid_path = output_dir / stream["instance_valid_path"]
    bitmap_path = output_dir / "download_complete.uint8.bin"
    expected_bytes = needed_instances * CHUNK_BYTES
    expected_sizes = (
        (token_path, expected_bytes),
        (valid_path, needed_instances),
        (bitmap_path, needed_instances),
    )
    for path, expected in expected_sizes:
        if path.stat().st_size != expected:
            raise RuntimeError(f"{path}: size drift")
    bitmap = np.memmap(
        bitmap_path,
        dtype=np.uint8,
        mode="r",
        shape=(needed_instances,),
    )
    valid = np.memmap(
        valid_path,
        dtype=np.uint8,
        mode="r",
        shape=(needed_instances,),
    )
    if not bool(np.all(bitmap == 1)):
        raise RuntimeError("data bitmap is incomplete")
    stream_expected = {
        "bytes": expected_bytes,
        "instances": needed_instances,
        "tokens": needed_instances * SEQUENCE_LENGTH,
        "sha256": sha256_file(token_path),
        "first_global_batch_sha256": first_global_batch_sha256(token_path),
        "instance_valid_sha256": sha256_file(valid_path),
        "invalid_instances": int((valid == 0).sum()),
    }
    for key, actual in stream_expected.items():
        if stream.get(key) != actual:
            raise RuntimeError(f"frozen training stream {key} drift")
    proof_reference = manifest.get("order_stream_proof")
    if not isinstance(proof_reference, dict):
        raise RuntimeError("order/source proof is missing")
    proof_path = output_dir / proof_reference["path"]
    if sha256_file(proof_path) != proof_reference["sha256"]:
        raise RuntimeError("order/source proof SHA-256 drift")
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof_expected = {
        "status": "ORDER_AND_SOURCE_PROOF_PASS",
        "official_config_sha256": OFFICIAL_CONFIG_SHA256,
        "seed": SEED,
        "total_instances": recomputed_rows[-1]["global_end"],
        "needed_instances": needed_instances,
        "order_file_sha256": order["indices_sha256"],
        "order_values_sha256": array_values_sha256(prefix),
        "source_manifest_sha256": source_receipt["sha256"],
        "training_stream_sha256": stream["sha256"],
        "instance_valid_sha256": stream["instance_valid_sha256"],
        "source_spotcheck_count": 100,
    }
    for key, expected in proof_expected.items():
        if proof.get(key) != expected:
            raise RuntimeError(f"order/source proof {key} drift")
    source_spotchecks = proof.get("source_spotchecks", [])
    if (
        len(source_spotchecks) != 100
        or sha256_json(source_spotchecks)
        != proof.get("source_spotchecks_sha256")
    ):
        raise RuntimeError("order/source spotcheck receipt drift")
    spotcheck_outputs = [
        int(row["output_instance"]) for row in source_spotchecks
    ]
    if (
        len(set(spotcheck_outputs)) != 100
        or min(spotcheck_outputs) < 0
        or max(spotcheck_outputs) >= needed_instances
    ):
        raise RuntimeError("order/source spotcheck index drift")
    for row in source_spotchecks:
        index = int(row["output_instance"])
        if (
            int(row["global_instance"]) != int(prefix[index])
            or int(row["source_id"]) != int(source_ids[index])
            or int(row["local_instance"]) != int(local_indices[index])
            or len(str(row["payload_sha256"])) != 64
            or not isinstance(row["instance_valid"], bool)
        ):
            raise RuntimeError("order/source spotcheck mapping drift")

    spotcheck = manifest.get("decode_spotcheck")
    if not isinstance(spotcheck, dict) or spotcheck.get("samples") != 100:
        raise RuntimeError("decode spotcheck receipt is missing or incomplete")
    spotcheck_path = output_dir / spotcheck["path"]
    if sha256_file(spotcheck_path) != spotcheck["sha256"]:
        raise RuntimeError("decode spotcheck SHA-256 drift")
    records = [
        json.loads(line)
        for line in spotcheck_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    spotcheck_indices = [int(row["instance_index"]) for row in records]
    if (
        len(records) != 100
        or len(set(spotcheck_indices)) != 100
        or min(spotcheck_indices) < 0
        or max(spotcheck_indices) >= needed_instances
    ):
        raise RuntimeError("decode spotcheck row contract drift")
    if tokenizer_path is not None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, local_files_only=True
        )
        if tokenizer.eos_token_id != TOKENIZER_MARKERS["eos_token_id"]:
            raise RuntimeError("tokenizer EOS marker drift")
        if tokenizer.pad_token_id != TOKENIZER_MARKERS["pad_token_id"]:
            raise RuntimeError("tokenizer pad marker drift")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--source-workers", type=int, default=64)
    parser.add_argument("--download-workers", type=int, default=128)
    parser.add_argument("--steps", type=int, default=FIRST_GATE_STEPS)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--prove-order-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only and args.prove_order_only:
        parser.error("--validate-only and --prove-order-only are mutually exclusive")

    output_dir = args.output_dir.resolve()
    official_config = args.official_config.resolve()
    if args.prove_order_only:
        proof = generate_order_stream_proof(
            output_dir,
            official_config,
            steps=args.steps,
        )
        print(json.dumps(proof, indent=2, sort_keys=True))
        return
    if args.validate_only:
        manifest = validate_existing(
            output_dir,
            official_config,
            tokenizer_path=(
                args.tokenizer_path.resolve()
                if args.tokenizer_path is not None
                else None
            ),
            steps=args.steps,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = load_official_paths(official_config)

    sizes_path = output_dir / "source_sizes.json"
    cached: dict[str, int] = {}
    if sizes_path.exists():
        cached_payload = json.loads(sizes_path.read_text(encoding="utf-8"))
        cached = {str(key): int(value) for key, value in cached_payload.items()}
    sizes = probe_source_sizes(
        paths,
        workers=args.source_workers,
        cached=cached,
        cache_path=sizes_path,
    )
    write_json(sizes_path, sizes)
    rows = source_rows(paths, sizes)
    rows_path = output_dir / "source_manifest.json"
    write_json(rows_path, rows)

    needed_instances = args.steps * GLOBAL_BATCH_SEQUENCES
    order_path = output_dir / "order_indices.uint32.npy"
    source_ids_path = output_dir / "order_source_ids.uint16.npy"
    local_indices_path = output_dir / "order_local_indices.uint64.npy"
    if not order_path.exists():
        prefix = generate_order_prefix(
            rows[-1]["global_end"],
            needed_instances,
            seed=SEED,
        )
        np.save(order_path, prefix, allow_pickle=False)
        source_ids, local_indices = map_indices(prefix, rows)
        np.save(source_ids_path, source_ids, allow_pickle=False)
        np.save(local_indices_path, local_indices, allow_pickle=False)
    prefix = np.load(order_path, allow_pickle=False, mmap_mode="r")
    source_ids = np.load(source_ids_path, allow_pickle=False, mmap_mode="r")
    local_indices = np.load(
        local_indices_path, allow_pickle=False, mmap_mode="r"
    )
    if len(prefix) != needed_instances:
        raise RuntimeError("order prefix length drift")
    remapped_source, remapped_local = map_indices(np.asarray(prefix), rows)
    if not np.array_equal(remapped_source, source_ids):
        raise RuntimeError("saved source-id mapping drift")
    if not np.array_equal(remapped_local, local_indices):
        raise RuntimeError("saved local-index mapping drift")

    stream_receipt: dict[str, Any] | None = None
    if not args.metadata_only:
        stream_receipt = materialize(
            output_dir,
            rows,
            source_ids,
            local_indices,
            workers=args.download_workers,
        )

    manifest = {
        "status": (
            "STREAM_VERIFIED"
            if stream_receipt is not None
            else "ORDER_METADATA_VERIFIED"
        ),
        "contract": {
            "tokenizer": "allenai_dolma2",
            "dtype": "uint32",
            "sequence_length": SEQUENCE_LENGTH,
            "seed": SEED,
            "global_batch_sequences": GLOBAL_BATCH_SEQUENCES,
            "steps": args.steps,
            "expected_tokens": args.steps
            * GLOBAL_BATCH_SEQUENCES
            * SEQUENCE_LENGTH,
            "official_config_sha256": OFFICIAL_CONFIG_SHA256,
            "order_algorithm": (
                "numpy.random.Generator(PCG64(seed)).shuffle("
                "arange(total_instances,dtype=uint32))"
            ),
            "instance_filter": {
                "repetition_min_period": 1,
                "repetition_max_period": 13,
                "repetition_max_count": 32,
            },
        },
        "provenance_level": (
            "deterministic released-recipe reconstruction; "
            "not promoted to paired trajectory until Geo sentinel validation"
        ),
        "source_manifest": {
            "path": rows_path.name,
            "sha256": sha256_file(rows_path),
            "configured_paths": len(rows),
            "unique_urls": len(set(paths)),
            "total_instances": rows[-1]["global_end"],
        },
        "order": {
            "indices_path": order_path.name,
            "indices_sha256": sha256_file(order_path),
            "source_ids_path": source_ids_path.name,
            "source_ids_sha256": sha256_file(source_ids_path),
            "local_indices_path": local_indices_path.name,
            "local_indices_sha256": sha256_file(local_indices_path),
            "mapping_sha256": sha256_json(
                {
                    "source_ids": sha256_file(source_ids_path),
                    "local_indices": sha256_file(local_indices_path),
                }
            ),
        },
        "training_stream": stream_receipt,
    }
    write_json(output_dir / "dataset_manifest.json", manifest)
    if args.tokenizer_path is not None and stream_receipt is not None:
        manifest["decode_spotcheck"] = decode_spotcheck(
            output_dir, args.tokenizer_path.resolve()
        )
        write_json(output_dir / "dataset_manifest.json", manifest)
    if stream_receipt is not None:
        generate_order_stream_proof(
            output_dir,
            official_config,
            steps=args.steps,
        )
        manifest = json.loads(
            (output_dir / "dataset_manifest.json").read_text(encoding="utf-8")
        )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
