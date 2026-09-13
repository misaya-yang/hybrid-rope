#!/usr/bin/env python3
"""Metadata-only inventory for the 0.5-core no-GPU instance.

No torch import, model load, tokenization, downloads, or large-file hashing.
This reports what remains to be validated; it never starts a GPU process.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path


def inspect(model: Path, sources: Path, data: Path) -> dict:
    config_path = model / "config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    index = model / "model.safetensors.index.json"
    weights = (sorted(set(json.loads(index.read_text())["weight_map"].values()))
               if index.is_file() else ["model.safetensors"])
    required = ["config.json", "tokenizer.json", "tokenizer_config.json", *weights]
    missing_model = [name for name in required if not (model / name).is_file()]
    raw_names = ["pg19_books.json", "longalign.jsonl", "qasper-train-dev.tgz",
                 "qasper-test.tgz"]
    missing_sources = [name for name in raw_names if not (sources / name).is_file()]
    if not any((sources / name).is_file() for name in ("native_rows.jsonl", "dolly.jsonl")):
        missing_sources.append("native_rows.jsonl OR dolly.jsonl")
    receipt_path = sources / "acquisition.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
    data_names = ["data_manifest.json", "cpt_train.npy", "sft_train.jsonl",
                  "cpt_train_8192.npy", "sft_train_8192.jsonl", "sft_train_16384.jsonl",
                  "native_train.jsonl", "native_dev.jsonl", "native_test.jsonl",
                  "qa_dev.jsonl", "qa_test.jsonl", "lm_validation.npy", "lm_test.npy"]
    missing_data = [name for name in data_names if not (data / name).is_file()]
    versions = {}
    for package in ("torch", "transformers", "peft", "numpy", "safetensors", "tokenizers"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "MISSING"
    return {
        "status": "METADATA_INVENTORY_ONLY",
        "current_priority": "OLMo recovery; Llama deferred until OLMo is stable",
        "model_type": config.get("model_type"),
        "native_length": config.get("max_position_embeddings"),
        "model_missing": missing_model,
        "model_sizes": {name: (model / name).stat().st_size for name in required
                        if (model / name).is_file()},
        "raw_sources_missing": missing_sources,
        "raw_download_status": receipt.get("status", "NO_RECEIPT"),
        "raw_download_completed_files": len(receipt.get("files", [])),
        "raw_download_failures": receipt.get("failures", []),
        "prepared_data_missing": missing_data,
        "versions": versions,
        "asset_validation_policy": "user_attested_clone; SHA validation is not required or scheduled",
        "not_tested": ["tokenization and source separation",
                       "actual PEFT forward/backward and resume", "GPU memory and kernels",
                       "training stability", "model capability"],
        "gpu_started": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = inspect(args.model, args.sources, args.data)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
