#!/usr/bin/env python3
"""Evaluate four preregistered static profiles on fresh paired natural NLL rows."""

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
from scripts.data.prepare_reference_coupling_nll import (
    DOCUMENTS, GRID, STATUS, TARGET_TOKENS, child_path, ids_hash, sha256_file,
    tokenizer_file_receipts,
)
from scripts.analysis.export_frozen_coupling_transport import (
    DEFAULT_GAIN_COEFFICIENT, DEFAULT_X_HIGH, DEFAULT_X_LOW,
)

ARM_ORDER = ("Native", "dimensionless_x", "normalized_raw_index", "official_equation_yarn")


def tensor_hash(values: np.ndarray) -> str:
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(values, dtype="<f4").tobytes()).hexdigest()


def load_data(root: Path, checkpoint: Path, lengths: list[int]) -> tuple[dict, list[dict]]:
    if not lengths or len(set(lengths)) != len(lengths) or any(length not in GRID for length in lengths):
        raise ValueError("requested lengths must be unique members of the frozen holdout grid")
    manifest = json.loads((root / "manifest.json").read_text())
    if (manifest.get("status") != STATUS or manifest.get("grid") != list(GRID)
            or manifest.get("natural_documents") != DOCUMENTS or manifest.get("rows") != DOCUMENTS * len(GRID)
            or manifest.get("target_tokens") != TARGET_TOKENS
            or manifest.get("p0_inputs", {}).get("excluded_documents") != 96):
        raise ValueError("unregistered natural holdout contract")
    if (manifest["config_sha256"] != sha256_file(checkpoint / "config.json")
            or manifest["tokenizer_files"] != tokenizer_file_receipts(checkpoint)):
        raise ValueError("holdout config/tokenizer identity mismatch")
    path = child_path(root, manifest["file"]["path"])
    if sha256_file(path) != manifest["file"]["sha256"]:
        raise ValueError("holdout row file hash mismatch")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    seen, targets, sources = set(), {}, {}
    for row in rows:
        sid, length, ids = row["sample_id"], row["length"], row["input_ids"]
        key = sid, length
        if (key in seen or length not in GRID or row.get("split") != "holdout"
                or row.get("family") != "natural" or row.get("variant") != "natural"
                or len(ids) != length or ids[0] != manifest["bos_token_id"]
                or row["target_start"] != length - TARGET_TOKENS or row["target_tokens"] != TARGET_TOKENS
                or row["prompt_ids_sha256"] != ids_hash(ids)
                or row["target_ids_sha256"] != ids_hash(ids[-TARGET_TOKENS:])):
            raise ValueError("invalid or duplicated natural holdout row")
        seen.add(key)
        if sid in targets and (targets[sid] != row["target_ids_sha256"] or sources[sid] != row["source_text_sha256"]):
            raise ValueError("natural targets/source changed across paired lengths")
        targets[sid], sources[sid] = row["target_ids_sha256"], row["source_text_sha256"]
    expected = {(f"reference-coupling-natural-{i:03d}", length) for i in range(DOCUMENTS) for length in GRID}
    if seen != expected or len(set(sources.values())) != DOCUMENTS:
        raise ValueError("natural holdout lacks the complete 32-document paired grid")
    manifest["manifest_sha256"] = sha256_file(root / "manifest.json")
    return manifest, sorted((row for row in rows if row["length"] in lengths), key=lambda r: (r["length"], r["sample_id"]))


def load_profiles(coupling_path: Path, baseline_path: Path, config_path: Path,
                  weight_hash: str, lengths: list[int]) -> tuple[list[dict], dict]:
    """Load and hash every admitted table before any model outputs exist."""
    coupling, baseline = [json.loads(path.read_text()) for path in (coupling_path, baseline_path)]
    if (coupling.get("status") != "FROZEN_COUPLING_TRANSPORT_EXPORTED"
            or baseline.get("status") != "STATIC_ROPE_BASELINES_EXPORTED"):
        raise ValueError("profile manifest status mismatch")
    config = json.loads(config_path.read_text())
    if config.get("rope_scaling") not in (None, {}) or (config.get("rope_parameters") or {}).get("rope_type", "default") != "default":
        raise ValueError("model must load original default Native RoPE")
    scope = coupling.get("reference_scope", {})
    reference = scope.get("L_ref")
    target = scope.get("target_length")
    native_length = config["max_position_embeddings"]
    if (native_length != 8192 or reference != 4096 or target not in (8192, 16384)
            or scope.get("L_config") != native_length or scope.get("table_parameters_refit") is not False
            or baseline.get("reference_length") != reference or baseline.get("target_length") != target
            or any(length > target for length in lengths)):
        raise ValueError("L_config/L_ref/profile target or requested horizon mismatch")
    scale = target / reference
    if (coupling["law"].get("family") != "clipped_affine"
            or coupling["law"].get("x_high") != DEFAULT_X_HIGH
            or coupling["law"].get("x_low") != DEFAULT_X_LOW
            or coupling["law"].get("scale") != scale or scope.get("s") != scale
            or coupling["gain"].get("coefficient") != DEFAULT_GAIN_COEFFICIENT
            or baseline.get("factor") != scale or baseline.get("search_performed") is not False
            or baseline.get("long_benchmark_scores_used") is not False):
        raise ValueError("frozen construction identity drift")
    references = [coupling["checkpoint"].get("reference_calibration", {}), baseline.get("reference_receipt") or {}]
    config_hash = sha256_file(config_path)
    native_hash = coupling["checkpoint"]["native_sha256_float32"]
    pairs = int(config.get("head_dim", config.get("hidden_size", 0) // config.get("num_attention_heads", 1))) // 2
    for manifest, ref in zip((coupling, baseline), references):
        identity = manifest["checkpoint"]
        if (identity.get("config_sha256") != config_hash or identity.get("native_sha256_float32") != native_hash
                or identity.get("native_length") != native_length or identity.get("pairs") != pairs
                or identity.get("reference_length") != reference
                or ref.get("status") != "NATIVE_REFERENCE_CONFIRMED"
                or ref.get("checkpoint_weight_sha256") != weight_hash
                or ref.get("reference_length") != reference or ref.get("config_sha256") != config_hash
                or ref.get("native_sha256_float32") != native_hash):
            raise ValueError("profile checkpoint or confirmed-reference identity mismatch")
    for key in ("confirmation_decision_sha256", "data_manifest_sha256", "receipt_sha256"):
        if not references[0].get(key) or references[0][key] != references[1].get(key):
            raise ValueError("profile manifests do not share the same confirmed reference")
    coupling_gain = float(coupling["gain"]["attention_scaling"])
    if coupling_gain != 1 + DEFAULT_GAIN_COEFFICIENT * math.log(scale):
        raise ValueError("coupling gain drift")
    profiles = [{"name": "Native", "values": None, "attention_scaling": 1.0,
                 "tensor_sha256": native_hash, "file_sha256": None}]
    for name, manifest, path in (("dimensionless_x", coupling, coupling_path),
                                  ("normalized_raw_index", coupling, coupling_path),
                                  ("official_equation_yarn", baseline, baseline_path)):
        entry = manifest["tables"][name]
        table_path = child_path(path.parent, entry["path"])
        if sha256_file(table_path) != entry["file_sha256"]:
            raise ValueError(f"table file hash mismatch: {name}")
        values = np.load(table_path, allow_pickle=False)
        if (values.dtype != np.dtype("float32") or values.shape != (pairs,)
                or not np.isfinite(values).all() or not (values > 0).all()
                or not np.all(values[:-1] > values[1:]) or tensor_hash(values) != entry["tensor_sha256"]):
            raise ValueError(f"invalid table shape/order/tensor hash: {name}")
        gain = float(entry["attention_scaling"]) if name == "official_equation_yarn" else coupling_gain
        if name == "official_equation_yarn" and (gain != 1 + .1 * math.log(scale)
                or entry.get("beta_fast") != 32 or entry.get("beta_slow") != 1
                or entry.get("original_max_position_embeddings") != reference):
            raise ValueError("official-equation YaRN identity drift")
        profiles.append({"name": name, "values": np.ascontiguousarray(values.copy()),
                         "attention_scaling": gain, "tensor_sha256": entry["tensor_sha256"],
                         "file_sha256": entry["file_sha256"]})
    return profiles, {"L_config": native_length, "L_ref": reference, "target_length": target, "scale": scale,
        "native_sha256_float32": native_hash, "coupling_manifest_sha256": sha256_file(coupling_path),
        "baseline_manifest_sha256": sha256_file(baseline_path), "reference": references[0],
        "profiles": [{key: value for key, value in profile.items() if key != "values"} for profile in profiles]}


def aligned_suffix(logits, ids):
    if logits.shape[1] != TARGET_TOKENS + 1 or ids.shape[1] < TARGET_TOKENS + 1:
        raise ValueError("model must return exactly the last 257 logits for 256 aligned targets")
    return logits[:, :-1], ids[:, -TARGET_TOKENS:]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-weight-sha256", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--coupling-manifest", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[4096, 8192])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("evaluation output directory must be fresh")
    from scripts.lib.checkpoint_identity import safetensors_weight_set_sha256
    weights = safetensors_weight_set_sha256(args.checkpoint)
    if weights != args.expected_weight_sha256:
        raise ValueError("checkpoint weight hash mismatch")
    data, rows = load_data(args.data_root, args.checkpoint, args.lengths)
    profiles, profile_receipt = load_profiles(args.coupling_manifest, args.baseline_manifest,
                                             args.checkpoint / "config.json", weights, args.lengths)
    if profile_receipt["reference"]["data_manifest_sha256"] != data["p0_inputs"]["manifest_sha256"]:
        raise ValueError("holdout exclusions and confirmed reference use different P0 inputs")
    import torch
    import torch.nn.functional as F
    import transformers
    from transformers import AutoModelForCausalLM
    from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.evaluate_ruler import configure_cuda, configure_ruler_flash_attention

    configure_cuda()
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, local_files_only=True,
        trust_remote_code=False, dtype=torch.bfloat16, attn_implementation="sdpa").eval().to("cuda")
    model.requires_grad_(False)
    configure_ruler_flash_attention(model)
    rotary = model.model.rotary_emb
    native = rotary.inv_freq.detach().cpu().float().numpy().copy()
    if tensor_hash(native) != profile_receipt["native_sha256_float32"] or float(rotary.attention_scaling) != 1.0:
        raise RuntimeError("runtime Native table/gain mismatch")
    if getattr(rotary, "rope_type", "default") != "default":
        raise RuntimeError("runtime rotary must retain default non-dynamic Native implementation")
    profiles[0]["values"] = native
    for profile in profiles:
        profile["active"] = torch.from_numpy(profile["values"]).to(device=rotary.inv_freq.device, dtype=rotary.inv_freq.dtype)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "REFERENCE_COUPLING_NLL_FROZEN", "profiles": profile_receipt,
        "checkpoint_weight_sha256": weights, "config_sha256": data["config_sha256"],
        "data_manifest_sha256": data["manifest_sha256"], "data_rows_sha256": data["file"]["sha256"],
        "tokenizer_files": data["tokenizer_files"], "lengths": sorted(args.lengths),
        "arm_order": list(ARM_ORDER), "natural_documents": DOCUMENTS, "target_tokens": TARGET_TOKENS,
        "script_sha256": sha256_file(Path(__file__)), "model_source_sha256": sha256_file(Path(inspect.getsourcefile(type(model)))),
        "attention_source_sha256": sha256_file(Path(inspect.getsourcefile(configure_ruler_flash_attention))),
        "torch": torch.__version__, "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(),
        "batch_tokens": 16384, "max_batch_size": 8, "flash_only": True, "use_cache": False,
        "compile": False, "model_updates": 0, "candidate_search": False, "all_tables_loaded_before_inference": True}
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
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
            for length in sorted(args.lengths):
                cell = [row for row in rows if row["length"] == length]
                batch_size = max(1, min(8, 16384 // length))
                for offset in range(0, len(cell), batch_size):
                    batch = cell[offset:offset + batch_size]
                    ids = torch.tensor([row["input_ids"] for row in batch], dtype=torch.long, device="cuda")
                    batch_started = time.perf_counter()
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        raw_logits = model(input_ids=ids, use_cache=False, logits_to_keep=257).logits
                        logits, targets = aligned_suffix(raw_logits, ids)
                        chunks = [F.cross_entropy(logits[:, left:left+64].float().transpose(1, 2),
                            targets[:, left:left+64], reduction="none") for left in range(0, 256, 64)]
                        losses = torch.cat(chunks, dim=1).mean(dim=1)
                    if not bool(torch.isfinite(losses).all()):
                        raise RuntimeError("nonfinite natural NLL")
                    if not torch.equal(rotary.inv_freq, profile["active"]) or float(rotary.attention_scaling) != profile["attention_scaling"]:
                        raise RuntimeError("static profile mutated during an arm")
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - batch_started
                    values = losses.cpu().tolist()
                    for row, nll in zip(batch, values):
                        record = {key: value for key, value in row.items() if key != "input_ids"}
                        record.update(arm=profile["name"], nll=nll, table_sha256_float32=profile["tensor_sha256"],
                            attention_scaling=profile["attention_scaling"], batch_size=len(batch), batch_seconds=elapsed)
                        results.append(record)
                        handle.write(json.dumps(record) + "\n")
                    handle.flush()
                    print(json.dumps({"arm": profile["name"], "length": length, "completed": offset + len(batch),
                        "documents": DOCUMENTS, "first_batch_nll": values[0] if offset == 0 else None,
                        "tokens_per_second": len(batch) * length / elapsed,
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved()}), flush=True)
                    del ids, raw_logits, logits, targets, chunks, losses
        os.fsync(handle.fileno())
    curves = {}
    native_by_cell = {(row["sample_id"], row["length"]): row["nll"] for row in results if row["arm"] == "Native"}
    for name in ARM_ORDER:
        curves[name] = {}
        for length in sorted(args.lengths):
            cell = [row for row in results if row["arm"] == name and row["length"] == length]
            curves[name][str(length)] = {"documents": len(cell), "mean_tail_nll": float(np.mean([row["nll"] for row in cell])),
                "mean_paired_delta_vs_native": float(np.mean([row["nll"] - native_by_cell[row["sample_id"], length] for row in cell]))}
    summary = {"status": "REFERENCE_COUPLING_NLL_COMPLETE", "curves": curves,
        "rows": len(results), "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "examples_sha256": sha256_file(examples), "run_manifest_sha256": sha256_file(args.output / "run_manifest.json"),
        "evidence_limit": "Fresh paired teacher-forced natural NLL; not exact generation, K causality, or a selected winner"}
    (args.output / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
