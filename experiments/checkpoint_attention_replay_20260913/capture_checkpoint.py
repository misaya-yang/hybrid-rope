#!/usr/bin/env python3
"""Capture selected pre-RoPE Q and complete visible K prefixes on Native inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np

from experiments.fixed_rope_three_interfaces_20260913.tables import model_geometry
from .capture_io import load_capture, save_capture
from .core import ReplayCapture


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def prompt_ids_sha256(values: list[int]) -> str:
    payload = json.dumps(list(values), separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def query_positions(length: int, count: int) -> np.ndarray:
    if length < 2 or count < 1:
        raise ValueError("capture needs at least two tokens and one query")
    first = max(1, length // (count + 1))
    positions = np.linspace(first, length - 1, min(count, length - 1)).round().astype(np.int64)
    return np.unique(positions)


def annotated_queries(row: dict) -> tuple[np.ndarray, tuple[str, ...], tuple[tuple[int, ...], ...]]:
    records = row.get("capture_queries")
    if not isinstance(records, list) or not records:
        raise ValueError("annotated capture row lacks capture_queries")
    positions = np.asarray([int(item["position"]) for item in records], dtype=np.int64)
    roles = tuple(str(item["role"]) for item in records)
    evidence = tuple(tuple(int(value) for value in item["evidence_token_indices"])
                     for item in records)
    if len(set(positions.tolist())) != len(positions) or len(set(roles)) != len(roles):
        raise ValueError("annotated capture query positions and roles must be unique")
    return positions, roles, evidence


def balanced_rows(rows: list[dict], limit: int) -> list[dict]:
    if limit < 1:
        raise ValueError("row limit must be positive")
    selected = []
    per_task: dict[str, list[dict]] = {}
    for row in rows:
        per_task.setdefault(str(row.get("task", "unclassified")), []).append(row)
    offset = 0
    while len(selected) < limit:
        added = False
        for task in sorted(per_task):
            if offset < len(per_task[task]):
                selected.append(per_task[task][offset])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        offset += 1
    return selected


def head_layout(value, *, tokens: int, heads: int, head_dim: int):
    """Return ``[T,H,D]`` from projection or QK-normalization outputs."""
    if value.ndim == 3 and value.shape == (1, tokens, heads * head_dim):
        return value[0].view(tokens, heads, head_dim)
    if value.ndim == 4 and value.shape == (1, tokens, heads, head_dim):
        return value[0]
    if value.ndim == 4 and value.shape == (1, heads, tokens, head_dim):
        return value[0].permute(1, 0, 2)
    raise RuntimeError(
        f"unsupported pre-RoPE head layout {tuple(value.shape)} for T={tokens},H={heads},D={head_dim}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layer", type=int, action="append", required=True)
    parser.add_argument("--row-limit", type=int, default=4)
    parser.add_argument("--queries-per-row", type=int, default=8)
    parser.add_argument("--query-mode", choices=("uniform", "annotated"), default="uniform")
    parser.add_argument("--max-input-tokens", type=int, required=True)
    parser.add_argument("--max-capture-bytes", type=int, default=12 * 1024 ** 3)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    layers = sorted(set(args.layer))
    config = json.loads((args.model / "config.json").read_text())
    geometry = model_geometry(config)
    native_length = int(config.get("max_position_embeddings", 0))
    if not 0 < args.max_input_tokens <= native_length:
        raise ValueError("capture max-input must be within the checkpoint Native window")
    rows = balanced_rows([
        row for row in read_jsonl(args.panel)
        if len(row.get("prompt_ids", [])) <= args.max_input_tokens
    ], args.row_limit)
    heads = int(config["num_attention_heads"])
    kv_heads = int(config.get("num_key_value_heads", heads))
    head_dim = int(config.get("head_dim") or config["hidden_size"] // heads)
    query_counts = [
        len(row.get("capture_queries") or []) if args.query_mode == "annotated"
        else len(query_positions(len(row["prompt_ids"]), args.queries_per_row))
        for row in rows
    ]
    estimated_capture_bytes = len(layers) * sum(
        2 * kv_heads * len(row["prompt_ids"]) * head_dim * 2
        + heads * queries * head_dim * 2
        for row, queries in zip(rows, query_counts)
    )
    for row in rows:
        if row.get("prompt_sha256") != prompt_ids_sha256(row["prompt_ids"]):
            raise ValueError(f"selected capture row has a mismatched prompt hash: {row.get('row_id')}")
    plan = {
        "status": "PLAN_ONLY",
        "model": str(args.model), "model_id": args.model_id,
        "panel": str(args.panel), "out": str(args.out),
        "layers": layers, "rows": len(rows), "queries_per_row": args.queries_per_row,
        "query_mode": args.query_mode,
        "estimated_capture_bytes_bf16_payload": estimated_capture_bytes,
        "max_capture_bytes": args.max_capture_bytes,
        "native_length": native_length, "max_input_tokens": args.max_input_tokens,
        "selected_rows": [{
            "row_id": str(row.get("row_id")), "task": str(row.get("task")),
            "prompt_sha256": row["prompt_sha256"], "input_tokens": len(row["prompt_ids"]),
        } for row in rows],
        "scope": "Native input only; pre-RoPE Q and complete K/V sequence; detached replay proxy",
    }
    if not args.execute:
        print(json.dumps(plan, sort_keys=True))
        return
    if estimated_capture_bytes > args.max_capture_bytes:
        raise ValueError("frozen capture exceeds the declared on-disk byte cap")
    if not rows or args.row_limit < 1 or args.queries_per_row < 1:
        raise ValueError("capture rows/counts are invalid")
    capture_contract = {
        **plan,
        "status": "CHECKPOINT_QK_CAPTURE_CONTRACT_V1",
        "model_config_sha256": file_sha256(args.model / "config.json"),
        "panel_sha256": file_sha256(args.panel),
        "model_geometry": geometry,
        "capture_implementation": "post-qk-norm-if-present-else-projection-plus-v-v3",
    }
    contract_path = args.out / "contract.json"
    if args.out.exists():
        if not contract_path.is_file() or json.loads(contract_path.read_text()) != capture_contract:
            raise ValueError("capture output belongs to another frozen contract")
        if (args.out / "index.json").is_file():
            finished = json.loads((args.out / "index.json").read_text())
            if finished.get("status") == "CHECKPOINT_QK_CAPTURE_COMPLETE_V1":
                if finished.get("contract_sha256") != file_sha256(contract_path):
                    raise ValueError("completed capture index has another contract")
                if len(finished.get("captures", [])) != len(rows) * len(layers):
                    raise ValueError("completed capture index has the wrong work-item count")
                for record in finished.get("captures", []):
                    receipt_path = Path(record["path"]) / "receipt.json"
                    if record.get("receipt_sha256") != file_sha256(receipt_path):
                        raise ValueError("completed capture receipt changed")
                    load_capture(Path(record["path"]), mmap_mode="r")
                print(json.dumps({"status": finished["status"], "captures": len(finished["captures"])}, sort_keys=True))
                return
    else:
        args.out.mkdir(parents=True)
        atomic_json(contract_path, capture_contract)
    if shutil.disk_usage(args.out).free < estimated_capture_bytes + 2 * 1024 ** 3:
        raise RuntimeError("insufficient free disk for capture plus the 2 GiB safety reserve")

    import torch
    from experiments.olmo_recovery_20260912.recovery_v2_runtime import load_model
    from experiments.olmo_recovery_20260912.runtime import validate_cuda

    validate_cuda()
    model, wrapper, _ = load_model(args.model, "Native", checkpoint=None, training=False)
    if wrapper is not None:
        raise RuntimeError("Native capture unexpectedly loaded an adapter")
    model.eval()
    if model.training:
        raise RuntimeError("capture model failed to enter eval mode")
    blocks = model.model.layers
    if any(layer < 0 or layer >= len(blocks) for layer in layers):
        raise ValueError("requested layer is outside the checkpoint")
    head_dim = int(model.config.hidden_size // model.config.num_attention_heads)
    heads = int(model.config.num_attention_heads)
    kv_heads = int(getattr(model.config, "num_key_value_heads", heads))
    native_inv = model.model.rotary_emb.inv_freq.detach().cpu().float().numpy()
    if len(native_inv) * 2 != head_dim:
        raise ValueError("partial rotary dimensions are not implemented by this replay capture")
    reference_gain = float(model.model.rotary_emb.attention_scaling)
    device = next(model.parameters()).device
    capture_index = []
    capture_modules = {}

    for row_index, row in enumerate(rows):
        ids = torch.tensor([row["prompt_ids"]], dtype=torch.long, device=device)
        if args.query_mode == "annotated":
            positions, roles, evidence = annotated_queries(row)
        else:
            positions = query_positions(ids.shape[1], args.queries_per_row)
            roles = tuple(f"uniform_{index}" for index in range(len(positions)))
            evidence = ()
        if np.any(positions >= ids.shape[1]):
            raise ValueError("capture query position lies outside prompt")
        captured: dict[int, dict[str, np.ndarray]] = {layer: {} for layer in layers}
        handles = []

        def q_hook(layer: int):
            def hook(_module, _inputs, output):
                layout = head_layout(
                    output, tokens=ids.shape[1], heads=heads, head_dim=head_dim,
                )
                selected = layout[torch.as_tensor(positions, device=output.device)]
                value = selected.permute(1, 0, 2)
                captured[layer]["q"] = value.detach().cpu().float().numpy()
            return hook

        def k_hook(layer: int):
            def hook(_module, _inputs, output):
                layout = head_layout(
                    output, tokens=ids.shape[1], heads=kv_heads, head_dim=head_dim,
                )
                value = layout.permute(1, 0, 2)
                captured[layer]["k"] = value.detach().cpu().float().numpy()
            return hook

        def v_hook(layer: int):
            def hook(_module, _inputs, output):
                layout = head_layout(
                    output, tokens=ids.shape[1], heads=kv_heads, head_dim=head_dim,
                )
                captured[layer]["v"] = layout.permute(1, 0, 2).detach().cpu().float().numpy()
            return hook

        for layer in layers:
            attention = blocks[layer].self_attn
            q_module = getattr(attention, "q_norm", None) or attention.q_proj
            k_module = getattr(attention, "k_norm", None) or attention.k_proj
            capture_modules[str(layer)] = {
                "q": "q_norm" if q_module is not attention.q_proj else "q_proj",
                "k": "k_norm" if k_module is not attention.k_proj else "k_proj",
                "v": "v_proj",
            }
            handles.append(q_module.register_forward_hook(q_hook(layer)))
            handles.append(k_module.register_forward_hook(k_hook(layer)))
            handles.append(attention.v_proj.register_forward_hook(v_hook(layer)))
        try:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                model(input_ids=ids, use_cache=False, return_dict=True)
        finally:
            for handle in handles:
                handle.remove()
        for layer in layers:
            if set(captured[layer]) != {"q", "k", "v"}:
                raise RuntimeError(f"layer {layer} did not expose q/k/v")
            attention = blocks[layer].self_attn
            scale = float(getattr(attention, "scaling", 1.0 / math.sqrt(head_dim)))
            family = str(row.get("family") or row.get("task") or "unclassified")
            row_id = str(row.get("row_id", row_index))
            capture = ReplayCapture(
                q=captured[layer]["q"], k=captured[layer]["k"],
                query_positions=positions, native_inv_freq=native_inv,
                attention_scale=scale, reference_gain=reference_gain,
                group=f"{family}|row={row_id}|layer={layer}",
                row_id=row_id, layer=layer,
                v=captured[layer]["v"], query_roles=roles,
                evidence_key_positions=evidence,
            )
            directory = args.out / f"row_{row_index:03d}_layer_{layer:03d}"
            if directory.exists():
                restored = load_capture(directory, mmap_mode="r")
                if (
                    restored.row_id != capture.row_id or restored.layer != capture.layer
                    or not np.array_equal(restored.query_positions, capture.query_positions)
                    or not np.array_equal(restored.native_inv_freq, capture.native_inv_freq)
                    or not np.array_equal(restored.q, capture.q)
                    or not np.array_equal(restored.k, capture.k)
                    or not np.array_equal(restored.v, capture.v)
                    or restored.attention_scale != capture.attention_scale
                    or restored.reference_gain != capture.reference_gain
                    or restored.group != capture.group
                    or restored.query_roles != capture.query_roles
                    or restored.evidence_key_positions != capture.evidence_key_positions
                ):
                    raise ValueError(f"saved capture differs from frozen work item: {directory}")
                receipt = json.loads((directory / "receipt.json").read_text())
            else:
                receipt = save_capture(directory, capture)
            capture_index.append({
                "path": str(directory), "row_id": receipt["row_id"],
                "group": receipt["group"], "layer": layer,
                "queries": int(len(positions)), "keys": int(ids.shape[1]),
                "query_roles": list(roles),
                "receipt_sha256": file_sha256(directory / "receipt.json"),
            })
            atomic_json(args.out / "partial_index.json", {
                "status": "CHECKPOINT_QK_CAPTURE_PARTIAL_V1",
                "contract_sha256": file_sha256(contract_path),
                "captures": capture_index,
            })
        del ids, captured
        torch.cuda.empty_cache()

    overall = {
        **plan,
        "status": "CHECKPOINT_QK_CAPTURE_COMPLETE_V1",
        "contract_path": str(contract_path.resolve()),
        "contract_sha256": file_sha256(contract_path),
        "model_geometry": geometry,
        "native_inv_freq_sha256": hashlib.sha256(
            np.ascontiguousarray(native_inv, dtype="<f4").tobytes()
        ).hexdigest(),
        "captures": capture_index,
        "causal_lag_sign": "query_position - key_position",
        "rotary_layout": "split_half",
        "capture_modules": capture_modules,
        "hidden_state_feedback": False,
    }
    atomic_json(args.out / "index.json", overall)
    print(json.dumps({"status": overall["status"], "captures": len(capture_index)}, sort_keys=True))


if __name__ == "__main__":
    main()
