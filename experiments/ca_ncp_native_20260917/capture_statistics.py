#!/usr/bin/env python3
"""Capture unlabeled Native post-norm/pre-RoPE Q/K signed moments.

Without ``--execute`` this command is PLAN_ONLY and does not import torch or
load a checkpoint. GPU execution is a separate explicit action.
"""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from . import METHOD_ID, NATIVE_LENGTH, PAIRS_PER_DOCUMENT
from .core import signed_moment, split_half_to_complex, tensor_sha256
from .io_utils import atomic_json, file_sha256


DISTANCE_EDGES = (0, 256, 512, 1024, 2048, 4096)
DISTANCE_LABELS = tuple(
    f"{DISTANCE_EDGES[i] + (1 if i == 0 else 0)}..{DISTANCE_EDGES[i + 1]}"
    for i in range(len(DISTANCE_EDGES) - 1)
)


def _head_layout(value, *, tokens: int, heads: int, head_dim: int):
    if value.ndim == 3 and tuple(value.shape) == (1, tokens, heads * head_dim):
        return value[0].view(tokens, heads, head_dim)
    if value.ndim == 4 and tuple(value.shape) == (1, tokens, heads, head_dim):
        return value[0]
    if value.ndim == 4 and tuple(value.shape) == (1, heads, tokens, head_dim):
        return value[0].permute(1, 0, 2)
    raise RuntimeError(
        f"unsupported Q/K layout {tuple(value.shape)} for T={tokens},H={heads},D={head_dim}"
    )


def _parameter_guard(model) -> dict[str, tuple[int, int]]:
    return {name: (id(parameter), int(parameter._version)) for name, parameter in model.named_parameters()}


def _save_npz(path: Path, **arrays) -> None:
    temporary = path.with_name(path.stem + ".incomplete.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--construction", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    asset_manifest = json.loads((args.assets / "manifest.json").read_text())
    method = json.loads((args.construction / "METHOD_RECEIPT.json").read_text())
    if not args.execute:
        print(json.dumps({
            "status": "PLAN_ONLY",
            "model_loaded": False,
            "gpu_execution": False,
            "documents": len(asset_manifest.get("documents", [])),
            "pairs_per_document": PAIRS_PER_DOCUMENT,
            "expected_native_tokens": len(asset_manifest.get("documents", [])) * NATIVE_LENGTH,
            "out": str(args.out),
        }, indent=2))
        return
    if asset_manifest.get("status") != "CPU_PREPARED" or len(asset_manifest.get("documents", [])) != 40:
        raise ValueError("statistics assets must contain the frozen 32 fit plus 8 report documents")
    if method.get("method_id") != METHOD_ID:
        raise ValueError("construction method differs")
    if asset_manifest["model_identity"]["config_sha256"] != method["model_identity"]["config_sha256"]:
        raise ValueError("statistics tokenizer/checkpoint differs from the construction")

    gpu_lock = open("/tmp/hybrid-rope-gpu0.lock", "a")
    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    active = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits",
    ], text=True).strip()
    if active:
        raise RuntimeError("a GPU process is active; statistics capture does not co-run or stop it")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("--execute requires exactly one visible CUDA device")
    from experiments.olmo_recovery_20260912.recovery_v2_runtime import load_model

    model, _, _ = load_model(args.model, "Native", checkpoint=None, training=False)
    model.eval()
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    guard = _parameter_guard(model)
    installed = model.model.rotary_emb.inv_freq.detach().cpu().float().numpy()
    if tensor_sha256(installed) != method["native_table_sha256_float32"]:
        raise RuntimeError("Native runtime table differs from METHOD_RECEIPT")
    config = json.loads((args.model / "config.json").read_text())
    layers = int(config["num_hidden_layers"])
    query_heads = int(config["num_attention_heads"])
    kv_heads = int(config.get("num_key_value_heads", query_heads))
    head_dim = int(config.get("head_dim") or int(config["hidden_size"]) // query_heads)
    if query_heads % kv_heads or head_dim % 2:
        raise ValueError("unsupported GQA layout")
    groups_per_kv = query_heads // kv_heads
    active = np.asarray(method["construction"]["active_indices_zero_based"], dtype=np.int64)
    wc = float(method["construction"]["carrier_frequency_float32"])
    active_size = int(active.size)
    args.out.mkdir(parents=True, exist_ok=True)
    moments_dir = args.out / "moments"
    moments_dir.mkdir(exist_ok=True)
    completed = []
    timing_rows = []

    for doc_index, document in enumerate(asset_manifest["documents"]):
        moment_path = moments_dir / f"{doc_index:02d}_{document['role']}.npz"
        if moment_path.is_file():
            completed.append({**document, "moment_file": str(moment_path.relative_to(args.out)),
                              "moment_file_sha256": file_sha256(moment_path)})
            with np.load(moment_path, allow_pickle=False) as prior:
                timing_rows.append({
                    "doc_id": document["doc_id"], "role": document["role"],
                    "native_forward_seconds": float(prior["native_forward_seconds"]),
                    "cpu_moment_seconds": float(prior["cpu_moment_seconds"]),
                })
            continue
        token_path = Path(document["token_file"])
        if not token_path.is_absolute():
            token_path = args.assets / token_path
        pair_path = args.assets / document["pair_file"]
        if file_sha256(token_path) != document["token_file_sha256"] or file_sha256(pair_path) != document["pair_file_sha256"]:
            raise ValueError("statistics token/pair asset drifted")
        tokens = np.load(token_path, allow_pickle=False).astype(np.int64)
        with np.load(pair_path, allow_pickle=False) as pairs:
            query_pos = pairs["query_pos"].astype(np.int64)
            key_pos = pairs["key_pos"].astype(np.int64)
        if tokens.shape != (NATIVE_LENGTH,) or query_pos.shape != (PAIRS_PER_DOCUMENT,) or np.any(query_pos <= key_pos):
            raise ValueError("statistics document shape or causal pairs differ")
        captured: dict[int, dict[str, np.ndarray]] = {layer: {} for layer in range(layers)}
        handles = []
        for layer_index, block in enumerate(model.model.layers):
            attention = block.self_attn
            if not hasattr(attention, "q_norm") or not hasattr(attention, "k_norm"):
                raise RuntimeError("CA-NCP requires post-norm/pre-RoPE Q/K hook points")

            def q_hook(_module, _inputs, output, *, layer=layer_index, sink=captured, positions=query_pos):
                layout = _head_layout(output, tokens=NATIVE_LENGTH, heads=query_heads, head_dim=head_dim)
                sink[layer]["q"] = layout[torch.as_tensor(positions, device=layout.device)].float().cpu().numpy()

            def k_hook(_module, _inputs, output, *, layer=layer_index, sink=captured, positions=key_pos):
                layout = _head_layout(output, tokens=NATIVE_LENGTH, heads=kv_heads, head_dim=head_dim)
                sink[layer]["k"] = layout[torch.as_tensor(positions, device=layout.device)].float().cpu().numpy()

            handles.append(attention.q_norm.register_forward_hook(q_hook))
            handles.append(attention.k_norm.register_forward_hook(k_hook))
        try:
            ids = torch.as_tensor(tokens, dtype=torch.long, device="cuda").unsqueeze(0)
            torch.cuda.synchronize()
            forward_start = time.perf_counter()
            with torch.inference_mode():
                model(input_ids=ids, use_cache=False, return_dict=True)
            torch.cuda.synchronize()
            forward_seconds = time.perf_counter() - forward_start
        finally:
            for handle in handles:
                handle.remove()
        if any(set(captured[layer]) != {"q", "k"} for layer in range(layers)):
            raise RuntimeError("one or more layers did not produce both Q and K captures")
        moment_start = time.perf_counter()
        lags = query_pos - key_pos
        moment = np.empty((layers, kv_heads, active_size, active_size), dtype=np.complex128)
        bin_moment = np.zeros((layers, kv_heads, len(DISTANCE_LABELS), active_size, active_size), dtype=np.complex128)
        bin_counts = np.asarray([
            int(np.sum((lags > DISTANCE_EDGES[index]) & (lags <= DISTANCE_EDGES[index + 1])))
            for index in range(len(DISTANCE_LABELS))
        ], dtype=np.int64)
        for layer_index, block in enumerate(model.model.layers):
            q = captured[layer_index]["q"]
            k = captured[layer_index]["k"]
            scale = float(block.self_attn.scaling)
            for group in range(kv_heads):
                start, stop = group * groups_per_kv, (group + 1) * groups_per_kv
                q_group = q[:, start:stop].mean(axis=1)
                q_complex = split_half_to_complex(q_group)[:, active]
                k_complex = split_half_to_complex(k[:, group])[:, active]
                moment[layer_index, group] = signed_moment(q_complex, k_complex, lags, wc, scale=scale)
                for bin_index in range(len(DISTANCE_LABELS)):
                    mask = (lags > DISTANCE_EDGES[bin_index]) & (lags <= DISTANCE_EDGES[bin_index + 1])
                    if np.any(mask):
                        bin_moment[layer_index, group, bin_index] = signed_moment(
                            q_complex[mask], k_complex[mask], lags[mask], wc, scale=scale,
                        )
        moment_seconds = time.perf_counter() - moment_start
        _save_npz(
            moment_path,
            moment_real=moment.real,
            moment_imag=moment.imag,
            bin_real=bin_moment.real,
            bin_imag=bin_moment.imag,
            bin_counts=bin_counts,
            query_pos=query_pos,
            key_pos=key_pos,
            native_forward_seconds=np.asarray(forward_seconds, dtype=np.float64),
            cpu_moment_seconds=np.asarray(moment_seconds, dtype=np.float64),
        )
        completed.append({**document, "moment_file": str(moment_path.relative_to(args.out)),
                          "moment_file_sha256": file_sha256(moment_path)})
        timing_rows.append({
            "doc_id": document["doc_id"], "role": document["role"],
            "native_forward_seconds": forward_seconds,
            "cpu_moment_seconds": moment_seconds,
        })
        atomic_json(args.out / "live.json", {"phase": "capture", "completed": len(completed), "total": 40})
        del captured
        torch.cuda.empty_cache()
    if _parameter_guard(model) != guard:
        raise RuntimeError("model parameter object/version changed during statistics capture")
    receipt = {
        "status": "STATISTICS_COMPLETE",
        "method_id": METHOD_ID,
        "method_receipt_sha256": file_sha256(args.construction / "METHOD_RECEIPT.json"),
        "asset_manifest_sha256": file_sha256(args.assets / "manifest.json"),
        "model_identity": method["model_identity"],
        "runtime": {
            "torch": torch.__version__,
            "dtype": "bfloat16 checkpoint; post-norm Q/K transferred as float32; moments complex128",
            "attention_backend": str(model.config._attn_implementation),
            "use_cache": False,
            "training": False,
            "inference_mode": True,
        },
        "layers": layers,
        "query_heads": query_heads,
        "kv_groups": kv_heads,
        "query_heads_per_group": groups_per_kv,
        "head_dim": head_dim,
        "active_indices_zero_based": active.tolist(),
        "carrier_frequency_float32": wc,
        "distance_bins": list(DISTANCE_LABELS),
        "documents": completed,
        "timing_by_document": timing_rows,
        "timing_totals": {
            "native_forward_seconds": float(sum(row["native_forward_seconds"] for row in timing_rows)),
            "cpu_moment_seconds": float(sum(row["cpu_moment_seconds"] for row in timing_rows)),
        },
        "document_counts": {"fit": sum(row["role"] == "fit" for row in completed),
                            "report": sum(row["role"] == "report" for row in completed)},
        "weights_unchanged": True,
        "parameter_guard": "same parameter object IDs and torch version counters before/after capture",
        "task_outputs_read": False,
        "loss_gradients_computed": False,
        "claim_boundary": "Real Native Q/K statistics only; no task-quality result.",
    }
    atomic_json(args.out / "STATISTICS_RECEIPT.json", receipt)
    print(json.dumps({"status": receipt["status"], "documents": len(completed), "out": str(args.out)}))


if __name__ == "__main__":
    main()
