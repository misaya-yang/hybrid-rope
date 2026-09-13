#!/usr/bin/env python3
"""Small OLMo Q/K coefficient diagnostic for the paper's Sections 3--4.

The command is metadata-only unless ``--execute`` is supplied.  A GPU run uses
the frozen E2 preparation and exactly ten rows: the first two rows with more
than 4096 input tokens from each of its five natural-QA tasks.  Selection never
reads model output.  It runs the backbone under the observed Native-gain-1 and
BM-gain-g4 tables, captures normalized pre-RoPE Q/K at layers 0,5,10,15, and
streams keys in chunks.  It stores aggregate raw C/D coefficient statistics,
same-gain BM-minus-MrPro phase-only score changes, and a Taylor remainder bound.

Example (still no GPU without --execute)::

  python qk_diagnostic.py --prepared /path/to/e2_prepared --out /path/to/qk_diag
  python qk_diagnostic.py --prepared /path/to/e2_prepared --out /path/to/qk_diag --execute

This is an interpretability diagnostic, not a PPL predictor, an E5 training
crossing result, an E7 complete-output causal intervention, or evidence that
the captured activation would remain fixed after changing the runtime table.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np


TASKS = ("hotpotqa", "2wikimqa", "qasper", "narrativeqa", "multifieldqa_en")
LAYERS = (0, 5, 10, 15)
DISTANCE_BINS = ((0, 128, "0_128"), (128, 4097, "128_4096"), (4097, None, "gt_4096"))


def tensor_sha(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f4").tobytes()).hexdigest()


def rows(path: Path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def choose_rows(screen: Path) -> list[dict[str, Any]]:
    selected: dict[str, list[dict[str, Any]]] = {task: [] for task in TASKS}
    for row in rows(screen):
        task = row["task"]
        if task in selected and row["input_tokens"] > 4096 and len(selected[task]) < 2:
            selected[task].append(row)
    if any(len(value) != 2 for value in selected.values()):
        raise ValueError({task: len(value) for task, value in selected.items()})
    return [row for task in TASKS for row in selected[task]]


def canonical_table(entry: dict[str, Any], name: str, pairs: int) -> tuple[np.ndarray, float]:
    key = "values_float32" if "values_float32" in entry else "values"
    value = np.asarray(entry[key], dtype=np.float32)
    if value.shape != (pairs,) or not np.isfinite(value).all() or not np.all(value > 0) or not np.all(value[:-1] > value[1:]):
        raise ValueError(f"invalid table: {name}")
    gain = float(entry["gain"])
    if not math.isfinite(gain) or gain <= 0:
        raise ValueError(f"invalid gain: {name}")
    return value, gain


def normalized_shape(value, heads: int, head_dim: int):
    """Return [batch, heads, length, head_dim] for known HF norm layouts."""
    import torch
    if isinstance(value, tuple):
        value = value[0]
    if value.ndim == 3 and value.shape[-1] == heads * head_dim:
        return value.view(value.shape[0], value.shape[1], heads, head_dim).transpose(1, 2)
    if value.ndim == 4 and value.shape[1] == heads and value.shape[-1] == head_dim:
        return value
    if value.ndim == 4 and value.shape[2] == heads and value.shape[-1] == head_dim:
        return value.transpose(1, 2)
    raise RuntimeError(f"unexpected normalized Q/K shape: {tuple(value.shape)}")


def accumulator() -> dict[str, float | int]:
    return {"count": 0, "score_count": 0, "c_sum": 0.0, "d_sum": 0.0, "c_sq": 0.0, "d_sq": 0.0,
            "c_abs_max": 0.0, "d_abs_max": 0.0, "exact_delta_sum": 0.0,
            "exact_delta_sq": 0.0, "linear_delta_sum": 0.0, "remainder_bound_sum": 0.0}


def add_stats(total, c, dcoef, exact, linear, bound):
    total["count"] += c.numel(); total["c_sum"] += float(c.sum()); total["d_sum"] += float(dcoef.sum())
    total["c_sq"] += float(c.float().square().sum()); total["d_sq"] += float(dcoef.float().square().sum())
    total["c_abs_max"] = max(total["c_abs_max"], float(c.abs().max())); total["d_abs_max"] = max(total["d_abs_max"], float(dcoef.abs().max()))
    exact_score, linear_score, bound_score = exact.sum(-1), linear.sum(-1), bound.sum(-1)
    total["score_count"] += exact_score.numel(); total["exact_delta_sum"] += float(exact_score.sum()); total["exact_delta_sq"] += float(exact_score.square().sum())
    total["linear_delta_sum"] += float(linear_score.sum()); total["remainder_bound_sum"] += float(bound_score.sum())


def finish(total):
    n = total["count"]; scores = total["score_count"]
    return {"count": n, "c_mean": total["c_sum"] / n, "d_mean": total["d_sum"] / n,
            "c_rms": math.sqrt(total["c_sq"] / n), "d_rms": math.sqrt(total["d_sq"] / n),
            "c_abs_max": total["c_abs_max"], "d_abs_max": total["d_abs_max"],
            "score_count": scores, "phase_delta_mean": total["exact_delta_sum"] / scores,
            "phase_delta_rms": math.sqrt(total["exact_delta_sq"] / scores),
            "linear_delta_mean": total["linear_delta_sum"] / scores,
            "taylor_remainder_bound_mean": total["remainder_bound_sum"] / scores}


def analyze(q, k, bm, mr, gain, chunk_size):
    import torch
    # q: [H,D], k: [KV,L,D]. Repeat KV heads only for the score identity.
    if q.shape[0] % k.shape[0]:
        raise RuntimeError("query heads are not divisible by KV heads")
    k = k.repeat_interleave(q.shape[0] // k.shape[0], dim=0)
    pairs = q.shape[-1] // 2; qx, qy = q[:, :pairs].float(), q[:, pairs:].float()
    delta = torch.as_tensor(bm - mr, device=q.device, dtype=torch.float32)
    bm_t = torch.as_tensor(bm, device=q.device); mr_t = torch.as_tensor(mr, device=q.device)
    transition = np.flatnonzero(np.abs(bm - mr) > 0)
    lo, hi = (int(transition[0]), int(transition[-1]) + 1) if len(transition) else (0, pairs)
    bands = ((0, lo, "fast"), (lo, hi, "transition"), (hi, pairs, "slow"))
    totals = {(distance, band): accumulator() for _, _, distance in DISTANCE_BINS for _, _, band in bands}
    permutation_error = 0.0
    angle = torch.tensor(0.371, device=q.device); ca, sa = angle.cos(), angle.sin()
    for begin in range(0, k.shape[1], chunk_size):
        end = min(k.shape[1], begin + chunk_size); block = k[:, begin:end].float()
        kx, ky = block[..., :pairs], block[..., pairs:]
        c = qx[:, None, :] * kx + qy[:, None, :] * ky
        dcoef = qx[:, None, :] * ky - qy[:, None, :] * kx
        # A synchronous within-pair basis rotation must preserve C and D.
        qxr, qyr = ca * qx - sa * qy, sa * qx + ca * qy
        kxr, kyr = ca * kx - sa * ky, sa * kx + ca * ky
        permutation_error = max(permutation_error, float((c - (qxr[:, None] * kxr + qyr[:, None] * kyr)).abs().max()), float((dcoef - (qxr[:, None] * kyr - qyr[:, None] * kxr)).abs().max()))
        positions = torch.arange(begin, end, device=q.device); distance = (k.shape[1] - 1 - positions).float()
        for low, high, dname in DISTANCE_BINS:
            mask = distance >= low
            if high is not None: mask &= distance < high
            if not bool(mask.any()): continue
            dd = distance[mask][None, :, None]
            for left, right, bname in bands:
                if right <= left: continue
                cc, dc = c[:, mask, left:right], dcoef[:, mask, left:right]
                phase_b, phase_m = dd * bm_t[left:right], dd * mr_t[left:right]
                scale = gain * gain / math.sqrt(q.shape[-1])
                exact = scale * (cc * (phase_b.cos() - phase_m.cos()) + dc * (phase_b.sin() - phase_m.sin()))
                derivative = scale * (-cc * phase_m.sin() + dc * phase_m.cos())
                linear = derivative * dd * delta[left:right]
                bound = 0.5 * scale * torch.sqrt(cc.square() + dc.square()) * (dd * delta[left:right]).square()
                add_stats(totals[(dname, bname)], cc, dc, exact, linear, bound)
    return {f"{distance}/{band}": finish(value) for (distance, band), value in totals.items() if value["count"]}, permutation_error, {"transition_pair_range": [lo, hi]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--native"); parser.add_argument("--bm"); parser.add_argument("--mrpro")
    parser.add_argument("--chunk-size", type=int, default=512); parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(); prepared = args.prepared.resolve(); manifest = json.loads((prepared / "manifest.json").read_text())
    new_schema = (prepared / "inputs.jsonl").is_file() or isinstance(manifest.get("model"), dict)
    args.native = args.native or ("native_g1" if new_schema else "Native")
    args.bm = args.bm or ("bm_g4" if new_schema else "MrProBM")
    args.mrpro = args.mrpro or ("mrpro_g4" if new_schema else "MrPro")
    input_path = prepared / ("inputs.jsonl" if new_schema else "screen.jsonl")
    model_record = manifest["model"] if new_schema else {"path": manifest["model_path"], "id": manifest["model_id"], "revision": manifest["revision"]}
    metadata = {"status": "METADATA_ONLY" if not args.execute else "STARTING", "input_file": input_path.name, "selection": {"rule": "first two rows with input_tokens > 4096 from each task, without reading outputs", "tasks": list(TASKS), "rows_per_task": 2, "total_rows": 10}, "layers": list(LAYERS), "observed_tables": [args.native, args.bm], "phase_replay_tables": [args.bm, args.mrpro], "model": model_record, "scope": "phase-only coefficient diagnostic; not PPL prediction, E5 crossing, or E7 causal output evidence"}
    print(json.dumps(metadata, indent=2, sort_keys=True));
    if not args.execute: return
    chosen = choose_rows(input_path)
    metadata["rows"] = [{key: row[key] for key in ("row_id", "task", "input_tokens", "prompt_sha256")} for row in chosen]
    import torch
    from transformers import AutoModelForCausalLM
    from scripts.experiments.olmo_fast_screen.runtime import install, verify
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise RuntimeError("BF16 CUDA required")
    properties = torch.cuda.get_device_properties(0)
    if torch.cuda.get_device_capability(0) != (12, 0) or "5090" not in properties.name.upper(): raise RuntimeError("requires RTX 5090 sm120")
    torch.backends.cuda.enable_flash_sdp(True); torch.backends.cuda.enable_math_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_cudnn_sdp(False)
    config = json.loads((Path(model_record["path"]) / "config.json").read_text())
    if config["model_type"] != "olmo2" or config["num_hidden_layers"] != 16 or config["hidden_size"] != 2048 or config["num_attention_heads"] != 16: raise ValueError("unexpected OLMo model identity")
    tables = json.loads((prepared / "tables.json").read_text()); pairs = (config["hidden_size"] // config["num_attention_heads"]) // 2
    native, native_gain = canonical_table(tables[args.native], args.native, pairs); bm, bm_gain = canonical_table(tables[args.bm], args.bm, pairs); mr, mr_gain = canonical_table(tables[args.mrpro], args.mrpro, pairs)
    if bm_gain != mr_gain: raise ValueError("phase-only replay requires exactly matched gains")
    expected_parameters = int(manifest.get("actual_parameters", 1_484_916_736))
    started_load = time.monotonic(); model = AutoModelForCausalLM.from_pretrained(model_record["path"], local_files_only=True, dtype=torch.bfloat16, device_map={"": "cuda"}, attn_implementation="sdpa").eval(); load_seconds = time.monotonic() - started_load
    if sum(parameter.numel() for parameter in model.parameters()) != expected_parameters: raise ValueError("model parameter identity mismatch")
    args.out.mkdir(parents=True, exist_ok=False); output = args.out / "coefficients.jsonl"; run_started = time.monotonic(); records = []
    with output.open("x") as stream, torch.inference_mode():
        for observed_name, observed_values, observed_gain in ((args.native, native, native_gain), (args.bm, bm, bm_gain)):
            install(model, tables[observed_name]); verify(model, tables[observed_name])
            for row in chosen:
                captured = {layer: {} for layer in LAYERS}; handles = []
                for layer in LAYERS:
                    attention = model.model.layers[layer].self_attn
                    handles.append(attention.q_norm.register_forward_hook(lambda _m, _i, out, layer=layer: captured[layer].__setitem__("q", normalized_shape(out, config["num_attention_heads"], pairs * 2)[:, :, -1].detach())))
                    handles.append(attention.k_norm.register_forward_hook(lambda _m, _i, out, layer=layer: captured[layer].__setitem__("k", normalized_shape(out, config["num_key_value_heads"], pairs * 2)[0].detach())))
                tick = time.monotonic()
                try:
                    ids = torch.tensor([row["prompt_ids"]], device="cuda"); model.model(input_ids=ids, use_cache=False)
                finally:
                    for handle in handles: handle.remove()
                for layer in LAYERS:
                    if set(captured[layer]) != {"q", "k"}: raise RuntimeError("Q/K hook did not fire")
                    stats, parity, band = analyze(captured[layer]["q"][0], captured[layer]["k"], bm, mr, bm_gain, args.chunk_size)
                    record = {"row_id": row["row_id"], "task": row["task"], "prompt_sha256": row["prompt_sha256"], "input_tokens": row["input_tokens"], "layer": layer, "observed_table": observed_name, "observed_table_sha256": tensor_sha(observed_values), "observed_gain": observed_gain, "replayed_phase_tables": {args.bm: tensor_sha(bm), args.mrpro: tensor_sha(mr), "shared_gain": bm_gain}, "coefficient_statistics": stats, "synchronous_pair_rotation_max_error": parity, "band": band, "forward_seconds": time.monotonic() - tick}
                    stream.write(json.dumps(record, sort_keys=True) + "\n"); stream.flush(); records.append(record)
                del ids, captured
    summary = {**metadata, "status": "COMPLETE", "record_count": len(records), "model": {**model_record, "parameters": expected_parameters}, "load_seconds": load_seconds, "diagnostic_seconds": time.monotonic() - run_started, "peak_cuda_bytes": torch.cuda.max_memory_allocated(), "gpu": {"name": properties.name, "total_memory_bytes": properties.total_memory}}
    temporary = args.out / "summary.json.incomplete"; temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n"); os.replace(temporary, args.out / "summary.json")


if __name__ == "__main__": main()
