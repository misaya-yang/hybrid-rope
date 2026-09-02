#!/usr/bin/env python3
"""Evaluate the three frozen Qwen K32 profiles on paired packed-natural NLL rows."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.data.prepare_qwen_k32_natural_nll import (
    DOCUMENTS, GRID, STATUS, TARGET_TOKENS, ids_hash, sha256_file,
    tokenizer_file_receipts,
)

ARM_ORDER = ("Native", "normalized_raw_index", "official_equation_yarn")
EXPECTED_WEIGHT_SHA256 = "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
EXPECTED_CONFIG_SHA256 = "18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45"
EXPECTED_NATIVE_SHA256 = "6d1e10125bd0468a7cf91c6175a3af31c1bffca24592cf5630f0f8402a8746e3"
EXPECTED_INDEX_SHA256 = "8c19ab976f71d30c6409f78a661209a8535ef9f101e8bf42f5bfce6f7817dc5f"
EXPECTED_INDEX_FILE_SHA256 = "36e09e014a86f8169296e6b5f2997ba7fea08ceb005b4e887085289d53db57a1"
EXPECTED_YARN_SHA256 = "d9eb5ac0185e84f2afa85997f10e4c51de97e3a2f937325769dd45ff86a0ea59"
EXPECTED_YARN_FILE_SHA256 = "980d8d16b84d792fb7b50d42941747a881985ef0fa246e7fa0f0964a9b79ca03"
INDEX_GAIN = 1 + .074 * math.log(2)
YARN_GAIN = 1 + .1 * math.log(2)


def tensor_hash(values: np.ndarray) -> str:
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(values, dtype="<f4").tobytes()).hexdigest()


def child_path(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    path.relative_to(root.resolve())
    return path


def load_data(root: Path, checkpoint: Path) -> tuple[dict, list[dict]]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("status") != STATUS or manifest.get("model_outcomes_read") is not False
            or manifest.get("grid") != list(GRID) or manifest.get("natural_streams") != DOCUMENTS
            or manifest.get("rows") != DOCUMENTS * len(GRID)
            or manifest.get("target_tokens") != TARGET_TOKENS
            or "insert one EOS between complete documents" not in manifest.get("packing_contract", "")):
        raise ValueError("unregistered packed-natural data contract")
    if (manifest["config_sha256"] != sha256_file(checkpoint / "config.json")
            or manifest["tokenizer_files"] != tokenizer_file_receipts(checkpoint)):
        raise ValueError("data checkpoint/tokenizer identity mismatch")
    path = child_path(root, manifest["file"]["path"])
    if sha256_file(path) != manifest["file"]["sha256"]:
        raise ValueError("natural row file hash mismatch")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    cells, targets = set(), {}
    for row in rows:
        sid, length, ids = row["sample_id"], row["length"], row["input_ids"]
        key = sid, length
        if (key in cells or length not in GRID or row.get("family") != "natural"
                or row.get("variant") != "packed_natural" or row.get("split") != "holdout"
                or len(ids) != length or row.get("target_start") != length - TARGET_TOKENS
                or row.get("target_tokens") != TARGET_TOKENS
                or row.get("prompt_ids_sha256") != ids_hash(ids)
                or row.get("target_ids_sha256") != ids_hash(ids[-TARGET_TOKENS:])):
            raise ValueError("invalid or duplicated packed-natural row")
        cells.add(key)
        if sid in targets and targets[sid] != row["target_ids_sha256"]:
            raise ValueError("paired 32K/64K targets differ")
        targets[sid] = row["target_ids_sha256"]
    expected = {(f"qwen-k32-natural-{index:03d}", length)
                for index in range(DOCUMENTS) for length in GRID}
    if cells != expected:
        raise ValueError("packed-natural grid is incomplete")
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    return manifest, sorted(rows, key=lambda row: (row["length"], row["sample_id"]))


def load_table(path: Path, file_hash: str, tensor_sha: str) -> np.ndarray:
    if sha256_file(path) != file_hash:
        raise ValueError("frozen table file SHA-256 mismatch")
    values = np.load(path, allow_pickle=False)
    if (values.dtype != np.dtype("float32") or values.shape != (32,)
            or not np.isfinite(values).all() or not (values > 0).all()
            or not np.all(values[:-1] > values[1:]) or tensor_hash(values) != tensor_sha):
        raise ValueError("invalid frozen Qwen K32 table identity")
    return np.ascontiguousarray(values.copy())


def load_profiles(index_path: Path, yarn_path: Path) -> list[dict]:
    return [
        {"name": "Native", "values": None, "attention_scaling": 1.0,
         "tensor_sha256": EXPECTED_NATIVE_SHA256, "file_sha256": None},
        {"name": "normalized_raw_index",
         "values": load_table(index_path, EXPECTED_INDEX_FILE_SHA256, EXPECTED_INDEX_SHA256),
         "attention_scaling": INDEX_GAIN, "tensor_sha256": EXPECTED_INDEX_SHA256,
         "file_sha256": EXPECTED_INDEX_FILE_SHA256},
        {"name": "official_equation_yarn",
         "values": load_table(yarn_path, EXPECTED_YARN_FILE_SHA256, EXPECTED_YARN_SHA256),
         "attention_scaling": YARN_GAIN, "tensor_sha256": EXPECTED_YARN_SHA256,
         "file_sha256": EXPECTED_YARN_FILE_SHA256},
    ]


def aligned_suffix(logits, ids):
    if logits.shape[1] != TARGET_TOKENS + 1 or ids.shape[1] < TARGET_TOKENS + 1:
        raise ValueError("model must return exactly 257 logits for 256 aligned targets")
    return logits[:, :-1], ids[:, -TARGET_TOKENS:]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--index-table", type=Path, required=True)
    parser.add_argument("--yarn-table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("evaluation output directory must be fresh")
    config = json.loads((args.checkpoint / "config.json").read_text())
    head_dim = config.get("head_dim")
    if head_dim is None:
        head_dim = int(config["hidden_size"]) // int(config["num_attention_heads"])
    head_dim = int(head_dim)
    if (sha256_file(args.checkpoint / "config.json") != EXPECTED_CONFIG_SHA256
            or config.get("model_type") != "qwen2" or head_dim != 64
            or int(config.get("max_position_embeddings", 0)) != GRID[0]
            or config.get("rope_scaling") not in (None, {})
            or config.get("use_sliding_window", False) is not False):
        raise ValueError("checkpoint is not the frozen unscaled Qwen K32 geometry")
    from scripts.lib.checkpoint_identity import safetensors_weight_set_sha256
    weights = safetensors_weight_set_sha256(args.checkpoint)
    if weights != EXPECTED_WEIGHT_SHA256:
        raise ValueError("Qwen K32 checkpoint weight SHA-256 mismatch")
    data, rows = load_data(args.data_root, args.checkpoint)
    profiles = load_profiles(args.index_table, args.yarn_table)
    import torch
    import torch.nn.functional as F
    import transformers
    from transformers import AutoModelForCausalLM
    from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.evaluate_ruler import (
        configure_cuda, configure_ruler_flash_attention,
    )

    configure_cuda()
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16, attn_implementation="sdpa",
    ).eval().to("cuda")
    model.requires_grad_(False)
    configure_ruler_flash_attention(model)
    rotary = model.model.rotary_emb
    native = rotary.inv_freq.detach().cpu().float().numpy().copy()
    if tensor_hash(native) != EXPECTED_NATIVE_SHA256 or float(rotary.attention_scaling) != 1.0:
        raise RuntimeError("runtime Native table/gain mismatch")
    if getattr(rotary, "rope_type", "default") != "default":
        raise RuntimeError("runtime rotary must remain static Native")
    profiles[0]["values"] = native
    for profile in profiles:
        profile["active"] = torch.from_numpy(profile["values"]).to(
            device=rotary.inv_freq.device, dtype=rotary.inv_freq.dtype)
    args.output.mkdir(parents=True, exist_ok=False)
    run_manifest = {
        "status": "QWEN_K32_PACKED_NATURAL_NLL_FROZEN", "checkpoint_weight_sha256": weights,
        "config_sha256": data["config_sha256"], "data_manifest_sha256": data["manifest_sha256"],
        "data_rows_sha256": data["file"]["sha256"], "source": data["source"],
        "tokenizer_files": data["tokenizer_files"], "packing_contract": data["packing_contract"],
        "lengths": list(GRID), "natural_streams": DOCUMENTS, "target_tokens": TARGET_TOKENS,
        "arm_order": list(ARM_ORDER),
        "profiles": [{key: value for key, value in profile.items() if key not in {"values", "active"}}
                     for profile in profiles],
        "script_sha256": sha256_file(Path(__file__)),
        "model_source_sha256": sha256_file(Path(inspect.getsourcefile(type(model)))),
        "attention_source_sha256": sha256_file(Path(inspect.getsourcefile(configure_ruler_flash_attention))),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(), "use_cache": False, "compile": False,
        "model_updates": 0, "profile_selection": False, "all_profiles_loaded_before_inference": True,
    }
    (args.output / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n")
    examples = args.output / "examples.jsonl"
    results, started = [], time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with examples.open("x") as handle:
        for profile in profiles:
            with torch.no_grad():
                rotary.inv_freq.copy_(profile["active"])
                if hasattr(rotary, "original_inv_freq"):
                    rotary.original_inv_freq = rotary.inv_freq.detach().clone()
                rotary.attention_scaling = profile["attention_scaling"]
            for row in rows:
                ids = torch.tensor([row["input_ids"]], dtype=torch.long, device="cuda")
                batch_started = time.perf_counter()
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    raw = model(input_ids=ids, use_cache=False, logits_to_keep=257).logits
                    logits, targets = aligned_suffix(raw, ids)
                    losses = torch.cat([
                        F.cross_entropy(logits[:, left:left + 64].float().transpose(1, 2),
                                        targets[:, left:left + 64], reduction="none")
                        for left in range(0, TARGET_TOKENS, 64)
                    ], dim=1).mean(dim=1)
                if not bool(torch.isfinite(losses).all()):
                    raise RuntimeError("nonfinite packed-natural NLL")
                if (not torch.equal(rotary.inv_freq, profile["active"])
                        or float(rotary.attention_scaling) != profile["attention_scaling"]):
                    raise RuntimeError("static profile mutated during a full forward")
                torch.cuda.synchronize()
                record = {key: value for key, value in row.items() if key != "input_ids"}
                record.update(arm=profile["name"], nll=float(losses.item()),
                              table_sha256_float32=profile["tensor_sha256"],
                              attention_scaling=profile["attention_scaling"],
                              batch_seconds=time.perf_counter() - batch_started)
                results.append(record)
                handle.write(json.dumps(record) + "\n"); handle.flush()
                del ids, raw, logits, targets, losses
        os.fsync(handle.fileno())
    native = {(row["sample_id"], row["length"]): row["nll"]
              for row in results if row["arm"] == "Native"}
    curves = {}
    for arm in ARM_ORDER:
        curves[arm] = {}
        for length in GRID:
            cell = [row for row in results if row["arm"] == arm and row["length"] == length]
            curves[arm][str(length)] = {
                "streams": len(cell), "mean_tail_nll": float(np.mean([row["nll"] for row in cell])),
                "mean_paired_delta_vs_native": float(np.mean([
                    row["nll"] - native[row["sample_id"], length] for row in cell])),
            }
    result = {"status": "QWEN_K32_PACKED_NATURAL_NLL_COMPLETE", "curves": curves,
              "rows": len(results), "elapsed_seconds": time.perf_counter() - started,
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
              "examples_sha256": sha256_file(examples),
              "run_manifest_sha256": sha256_file(args.output / "run_manifest.json"),
              "evidence_limit": "Paired final-256 teacher-forced NLL; no generation, K causality, or profile selection"}
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
