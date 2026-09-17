"""Small on-disk receipt format for streamed pre-RoPE Q/K captures."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .core import ReplayCapture, validate_capture


FORMAT_V1 = "CHECKPOINT_ATTENTION_CAPTURE_V1"
FORMAT_V2 = "CHECKPOINT_ATTENTION_QKV_CAPTURE_V2"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _storage_array(name: str, value: np.ndarray) -> tuple[np.ndarray, str | None]:
    data = np.ascontiguousarray(value)
    if name in {"q", "k", "v"} and data.dtype == np.float32:
        words = data.view(np.uint32)
        if np.all((words & np.uint32(0xFFFF)) == 0):
            return (words >> np.uint32(16)).astype(np.uint16), "bfloat16_bits"
    return data, None


def _decode_storage(value: np.ndarray, encoding: str | None) -> np.ndarray:
    if encoding is None:
        return value
    if encoding != "bfloat16_bits" or value.dtype != np.uint16:
        raise ValueError(f"unsupported capture array encoding: {encoding}")
    words = value.astype(np.uint32) << np.uint32(16)
    return words.view(np.float32)


def save_capture(directory: Path, capture: ReplayCapture) -> dict:
    record = validate_capture(capture)
    directory = Path(directory)
    temporary = directory.with_name(directory.name + ".incomplete")
    if directory.exists() or temporary.exists():
        raise FileExistsError(directory)
    temporary.mkdir(parents=True, exist_ok=False)
    arrays = {
        "q": np.asarray(record.q, dtype=np.float32),
        "k": np.asarray(record.k, dtype=np.float32),
        "query_positions": np.asarray(record.query_positions, dtype=np.int64),
        "native_inv_freq": np.asarray(record.native_inv_freq, dtype=np.float32),
    }
    if record.v is not None:
        arrays["v"] = np.asarray(record.v, dtype=np.float32)
    if record.evidence_key_positions:
        lengths = [len(values) for values in record.evidence_key_positions]
        arrays["evidence_offsets"] = np.asarray(
            [0, *np.cumsum(lengths).tolist()], dtype=np.int64,
        )
        arrays["evidence_indices"] = np.asarray(
            [value for values in record.evidence_key_positions for value in values],
            dtype=np.int64,
        )
    stored = {}
    encodings = {}
    for name, value in arrays.items():
        stored[name], encodings[name] = _storage_array(name, value)
        np.save(temporary / f"{name}.npy", stored[name], allow_pickle=False)
    receipt = {
        "status": FORMAT_V2 if record.v is not None else FORMAT_V1,
        "row_id": record.row_id,
        "group": record.group,
        "layer": int(record.layer),
        "attention_scale": float(record.attention_scale),
        "reference_gain": float(record.reference_gain),
        "causal_lag_sign": "query_position - key_position",
        "rotary_layout": "split_half",
        "gqa_query_heads_per_kv_head": int(record.q.shape[0] // record.k.shape[0]),
        "complete_visible_key_prefix": True,
        "query_roles": list(record.query_roles),
        "arrays": {
            name: {
                "path": f"{name}.npy", "shape": list(stored[name].shape),
                "dtype": str(stored[name].dtype),
                **({"encoding": encodings[name]} if encodings[name] else {}),
                "sha256": file_sha256(temporary / f"{name}.npy"),
            }
            for name, value in arrays.items()
        },
    }
    (temporary / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(directory)
    return receipt


def load_capture(directory: Path, *, mmap_mode: str | None = None) -> ReplayCapture:
    directory = Path(directory)
    receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
    if receipt.get("status") not in (FORMAT_V1, FORMAT_V2):
        raise ValueError("unsupported capture receipt")
    arrays = {}
    for name, metadata in receipt["arrays"].items():
        path = directory / metadata["path"]
        if metadata.get("sha256") != file_sha256(path):
            raise ValueError(f"capture array hash differs: {name}")
        value = np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
        if list(value.shape) != metadata["shape"] or str(value.dtype) != metadata["dtype"]:
            raise ValueError(f"capture array identity differs: {name}")
        arrays[name] = _decode_storage(value, metadata.get("encoding"))
    evidence = ()
    if "evidence_offsets" in arrays or "evidence_indices" in arrays:
        if not {"evidence_offsets", "evidence_indices"}.issubset(arrays):
            raise ValueError("capture evidence arrays are incomplete")
        offsets = arrays.pop("evidence_offsets")
        indices = arrays.pop("evidence_indices")
        if offsets.ndim != 1 or len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != len(indices):
            raise ValueError("capture evidence offsets are invalid")
        evidence = tuple(
            tuple(int(value) for value in indices[offsets[index]:offsets[index + 1]])
            for index in range(len(offsets) - 1)
        )
    return validate_capture(ReplayCapture(
        q=arrays["q"],
        k=arrays["k"],
        query_positions=arrays["query_positions"],
        native_inv_freq=arrays["native_inv_freq"],
        attention_scale=float(receipt["attention_scale"]),
        reference_gain=float(receipt["reference_gain"]),
        group=str(receipt["group"]),
        row_id=str(receipt["row_id"]),
        layer=int(receipt["layer"]),
        v=arrays.get("v"),
        query_roles=tuple(str(value) for value in receipt.get("query_roles", [])),
        evidence_key_positions=evidence,
    ))
