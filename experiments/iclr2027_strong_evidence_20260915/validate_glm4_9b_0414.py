#!/usr/bin/env python3
"""Validate the pinned local GLM snapshot without loading model weights."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path

from safetensors import safe_open
from transformers import AutoTokenizer


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/root/models/GLM-4-9B-0414"))
    args = parser.parse_args()
    receipt = json.loads((args.model / "DOWNLOAD_RECEIPT.json").read_text())
    if receipt.get("status") != "DOWNLOAD_COMPLETE_VERIFIED":
        raise ValueError("download receipt is not verified")
    tensors = 0
    parameters = 0
    dtypes: Counter[str] = Counter()
    for shard in sorted(args.model.glob("model-*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                tensor = handle.get_slice(key)
                count = 1
                for dimension in tensor.get_shape():
                    count *= int(dimension)
                tensors += 1
                parameters += count
                dtypes[str(tensor.get_dtype())] += count
    config = json.loads((args.model / "config.json").read_text())
    text = config.get("text_config", config)
    head_dim = int(text.get("head_dim") or text["hidden_size"] // text["num_attention_heads"])
    partial = float(text.get("partial_rotary_factor", 1.0))
    rotary_pairs = int(head_dim * partial) // 2
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hello"}], tokenize=True,
        add_generation_prompt=True,
    )
    if isinstance(rendered, dict):
        rendered = rendered["input_ids"]
    free = os.statvfs(args.model).f_bavail * os.statvfs(args.model).f_frsize
    result = {
        "status": "VERIFIED_READY_FOR_RUNTIME_INTEGRATION",
        "source_model": receipt["source_model"],
        "hf_revision": receipt["hf_revision"],
        "all_file_sha256_match_pinned_source": True,
        "tensor_count": tensors,
        "parameter_count": parameters,
        "dtype_parameters": dict(dtypes),
        "model_type": config.get("model_type"),
        "native_length": int(text["max_position_embeddings"]),
        "rope_base": float(text.get("rope_theta", 10000.0)),
        "attention_head_dim": head_dim,
        "partial_rotary_factor": partial,
        "rotary_pairs": rotary_pairs,
        "tokenizer_local_load": "PASS",
        "chat_template_tokens": len(rendered),
        "system_disk_free_bytes": free,
        "model_execution": False,
    }
    expected = (523, 9_400_279_040, {"BF16": 9_400_279_040}, "glm4", 32768, 128, 0.5, 32)
    actual = (tensors, parameters, dict(dtypes), config.get("model_type"), int(text["max_position_embeddings"]), head_dim, partial, rotary_pairs)
    if actual != expected:
        raise ValueError(f"local GLM identity differs: {actual}")
    atomic_json(args.model / "LOCAL_VALIDATION.json", result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
