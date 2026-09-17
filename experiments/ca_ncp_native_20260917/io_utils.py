"""Small immutable-artifact helpers shared by CA-NCP entry points."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    temporary.replace(path)


def read_json_or_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    text = path.read_text()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(value, dict):
        value = value.get("documents", value.get("rows"))
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError("manifest must be a JSON list, JSONL rows, or an object with documents")
    return value


def model_identity(model: Path, *, include_checkpoint_files: bool = True) -> dict:
    model = Path(model).resolve()
    config_path = model / "config.json"
    tokenizer_path = model / "tokenizer.json"
    if not config_path.is_file() or not tokenizer_path.is_file():
        raise FileNotFoundError("model must contain config.json and tokenizer.json")
    config = json.loads(config_path.read_text())
    heads = int(config["num_attention_heads"])
    head_dim = int(config.get("head_dim") or int(config["hidden_size"]) // heads)
    rope = config.get("rope_parameters") or {}
    partial = float(config.get("partial_rotary_factor") or rope.get("partial_rotary_factor") or 1.0)
    tokenizer_files = {}
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        path = model / name
        if path.is_file():
            tokenizer_files[name] = file_sha256(path)
    result = {
        "model_path": str(model),
        "artifact_name": model.name,
        "config_sha256": file_sha256(config_path),
        "model_type": config.get("model_type"),
        "architectures": list(config.get("architectures") or []),
        "hidden_size": int(config["hidden_size"]),
        "num_hidden_layers": int(config["num_hidden_layers"]),
        "num_attention_heads": heads,
        "num_key_value_heads": int(config.get("num_key_value_heads", heads)),
        "head_dim": head_dim,
        "rotary_pairs": int(head_dim * partial) // 2,
        "native_length": int(config["max_position_embeddings"]),
        "rope_theta": float(config.get("rope_theta") or rope.get("rope_theta")),
        "rope_scaling": config.get("rope_scaling"),
        "tokenizer_files_sha256": tokenizer_files,
    }
    if include_checkpoint_files:
        weight_files = sorted(
            path for pattern in ("*.safetensors", "*.bin", "*.index.json")
            for path in model.glob(pattern) if path.is_file()
        )
        result["checkpoint_files"] = [
            {"name": path.name, "size": path.stat().st_size, "sha256": file_sha256(path)}
            for path in weight_files
        ]
    return result
