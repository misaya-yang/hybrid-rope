"""Small on-disk receipt format for streamed pre-RoPE Q/K captures."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .core import ReplayCapture, validate_capture


FORMAT = "CHECKPOINT_ATTENTION_CAPTURE_V1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    for name, value in arrays.items():
        np.save(temporary / f"{name}.npy", value, allow_pickle=False)
    receipt = {
        "status": FORMAT,
        "row_id": record.row_id,
        "group": record.group,
        "layer": int(record.layer),
        "attention_scale": float(record.attention_scale),
        "reference_gain": float(record.reference_gain),
        "causal_lag_sign": "query_position - key_position",
        "rotary_layout": "split_half",
        "gqa_query_heads_per_kv_head": int(record.q.shape[0] // record.k.shape[0]),
        "complete_visible_key_prefix": True,
        "arrays": {
            name: {
                "path": f"{name}.npy", "shape": list(value.shape), "dtype": str(value.dtype),
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
    if receipt.get("status") != FORMAT:
        raise ValueError("unsupported capture receipt")
    arrays = {}
    for name, metadata in receipt["arrays"].items():
        path = directory / metadata["path"]
        if metadata.get("sha256") != file_sha256(path):
            raise ValueError(f"capture array hash differs: {name}")
        value = np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
        if list(value.shape) != metadata["shape"] or str(value.dtype) != metadata["dtype"]:
            raise ValueError(f"capture array identity differs: {name}")
        arrays[name] = value
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
    ))
