"""Replay saved Full-trajectory Q against one native prefix prefill per input.

This adds the real mass * raw ||V|| objective corresponding to the PM proxy's
value weighting. No generation, P/C/U/KeyDiff scoring, or keep-set changes occur.
Saved alpha=P(prefix | all visible keys) restores the original total denominator:
prefix_probability_j = alpha * softmax_prefix(q.K)_j. Only new audit output is
written; original traces remain untouched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from .run import digest, model_identity, records, write_json


@torch.inference_mode()
def replay_query(q_post, prefix_keys, prefix_values, alpha, keep_sets, *, attention_scale):
    """Inputs Q[Hq,D], K/V[Hkv,T,D], alpha[Hq]; original GQA order."""
    if q_post.ndim != 2 or prefix_keys.ndim != 3 or prefix_values.shape != prefix_keys.shape:
        raise ValueError("expected Q[Hq,D] and prefix K/V[Hkv,T,D]")
    hkv, length, dim = prefix_keys.shape
    hq = q_post.shape[0]
    if hq % hkv or q_post.shape[1] != dim:
        raise ValueError("native contiguous GQA required")
    if q_post.device.type == "cuda" and torch.backends.cuda.matmul.allow_tf32:
        raise ValueError("FP32 replay requires TF32 disabled")
    acc = torch.float64 if q_post.dtype == torch.float64 else torch.float32
    alpha = torch.as_tensor(alpha, device=q_post.device, dtype=acc)
    if alpha.shape != (hq,) or not torch.isfinite(alpha).all() or (alpha < 0).any() or (alpha > 1 + 1e-6).any():
        raise ValueError("saved alpha must be finite per-query-head prefix probability")
    with torch.autocast(device_type=q_post.device.type, enabled=False):
        q = q_post.to(acc).reshape(hkv, hq // hkv, dim)
        k, v = prefix_keys.to(acc), prefix_values.to(acc)
        conditional = (torch.einsum("hgd,htd->hgt", q, k) * attention_scale).softmax(-1)
        probabilities = conditional * alpha.reshape(hkv, hq // hkv, 1)
        vnorm = v.norm(dim=-1)
        weighted = probabilities * vnorm[:, None]
        conditional_weighted = conditional * vnorm[:, None]
        full_weighted = weighted.sum(-1)
        result = {"alpha_per_query_head": alpha.tolist(),
                  "full_prefix_weighted_value_mass_per_query_head": full_weighted.flatten().tolist(),
                  "arms": {}}
        for arm, selected in keep_sets.items():
            ids = torch.as_tensor(selected, dtype=torch.long, device=q.device)
            if ids.ndim != 2 or ids.shape[0] != hkv or not ids.shape[1]:
                raise ValueError("keep set must be [Hkv,B]")
            if (ids < 0).any() or (ids >= length).any() or (ids.shape[1] > 1 and not (ids[:, 1:] > ids[:, :-1]).all()):
                raise ValueError("keep set must be sorted unique original prefix positions")
            take = ids[:, None].expand(hkv, hq // hkv, -1)
            mass = probabilities.gather(-1, take).sum(-1)
            objective = weighted.gather(-1, take).sum(-1)
            cond_objective = conditional_weighted.gather(-1, take).sum(-1)
            result["arms"][arm] = {
                "reconstructed_prefix_mass_per_query_head": mass.flatten().tolist(),
                "real_mass_times_raw_vnorm_per_query_head": objective.flatten().tolist(),
                "conditional_mass_times_raw_vnorm_per_query_head": cond_objective.flatten().tolist(),
                "missed_weighted_prefix_mass_per_query_head": (full_weighted - objective).flatten().tolist(),
                "mean_real_mass_times_raw_vnorm": float(objective.mean()),
                "mean_conditional_mass_times_raw_vnorm": float(cond_objective.mean()),
                "mean_reconstructed_prefix_mass": float(mass.mean()),
            }
    result["P_minus_C_real_mass_times_raw_vnorm"] = (
        result["arms"]["P"]["mean_real_mass_times_raw_vnorm"] - result["arms"]["C"]["mean_real_mass_times_raw_vnorm"])
    result["P_minus_C_conditional_mass_times_raw_vnorm"] = (
        result["arms"]["P"]["mean_conditional_mass_times_raw_vnorm"] - result["arms"]["C"]["mean_conditional_mass_times_raw_vnorm"])
    return result


def verify_identity(contract, actual_model, row, trace_json):
    expected = contract["model"]
    # Relocation alone is not a scientific mismatch; compare the exact identity
    # fields recorded by the original run without re-hashing multi-GB weights.
    for field in ("config_sha256", "tokenizer_sha256", "weights"):
        if actual_model[field] != expected[field]:
            raise ValueError(f"model identity differs from saved trace: {field}")
    recorded = contract["inputs"][row["row_id"]]
    if digest(row["prompt_ids"]) != recorded["prompt_sha256"]:
        raise ValueError("input SHA differs from saved trace")
    if row["prefix_ids"] + row["suffix_ids"] != row["prompt_ids"]:
        raise ValueError("prefix/suffix boundary mismatch")
    if len(row["prefix_ids"]) != trace_json["prefix_choices"]["prefix_length"]:
        raise ValueError("prefix boundary differs from saved trace")


@torch.inference_mode()
def native_prefix(model, prefix_ids):
    """Exactly one native Full prefix, no sampling/session/scorer hooks."""
    device = next(model.parameters()).device
    ids = torch.as_tensor(prefix_ids, dtype=torch.long, device=device)[None]
    pos = torch.arange(ids.shape[1], device=device)
    return model(ids, position_ids=pos[None], cache_position=pos,
                 use_cache=True, logits_to_keep=1).past_key_values


def audit_row(model, row, original, saved):
    keep, queries = saved["keep_indices"], saved["queries"]
    for arm, sets in keep.items():
        if digest([x.cpu().tolist() for x in sets]) != original["prefix_choices"]["keep_sha256"][arm]:
            raise ValueError("saved keep set differs from original trace JSON")
    prefix = native_prefix(model, row["prefix_ids"])
    device = next(model.parameters()).device
    traced = {(r["layer_idx"], r["query_position"]): r for r in original["diagnostic"]["per_layer_query"]}
    observed = {(int(layer), int(pos)) for layer, qs in queries.items() for pos in qs}
    if observed != set(traced):
        raise ValueError("saved Q and JSON alpha records disagree")
    details = []
    for layer_index, by_position in queries.items():
        layer_index = int(layer_index)
        cache = prefix.layers[layer_index]
        for pos, q in sorted(by_position.items()):
            reference = traced[(layer_index, int(pos))]
            result = replay_query(q.to(device), cache.keys[0], cache.values[0],
                reference["full_prefix_mass_per_query_head"],
                {arm: sets[layer_index].to(device) for arm, sets in keep.items()},
                attention_scale=model.model.layers[layer_index].self_attn.scaling)
            result.update(layer_idx=layer_index, query_position=int(pos))
            result["unweighted_mass_replay_max_abs_error"] = max(
                max(abs(a - b) for a, b in zip(result["arms"][arm]["reconstructed_prefix_mass_per_query_head"],
                    reference["arms"][arm]["kept_prefix_mass_per_query_head"])) for arm in keep)
            details.append(result)
    del prefix
    by_position = {}
    for pos in sorted({r["query_position"] for r in details}):
        rows = [r for r in details if r["query_position"] == pos]
        by_position[str(pos)] = {
            key: sum(r[key] for r in rows) / len(rows) for key in
                ("P_minus_C_real_mass_times_raw_vnorm", "P_minus_C_conditional_mass_times_raw_vnorm")}
        by_position[str(pos)]["arms"] = {
            arm: {key: sum(r["arms"][arm][key] for r in rows) / len(rows) for key in
                ("mean_real_mass_times_raw_vnorm", "mean_conditional_mass_times_raw_vnorm")}
            for arm in keep}
    return {"row_id": row["row_id"], "task": row["task"],
            "original_proxy_P_minus_C": original["prefix_choices"]["P_objective_P_minus_C"],
            "per_position_summary": by_position, "per_layer_query": details,
            "unweighted_mass_replay_max_abs_error": max(r["unweighted_mass_replay_max_abs_error"] for r in details),
            "F_max_missed_weighted_prefix_mass_abs": max(abs(x) for r in details
                for x in r["arms"]["F"]["missed_weighted_prefix_mass_per_query_head"]),
            "original_full_answer": original["full_answer"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace-dir", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    a = p.parse_args()
    folder, output = Path(a.trace_dir), Path(a.output)
    if output.exists():
        raise FileExistsError("write a new audit; never overwrite original evidence")
    contract = json.loads((folder / "contract.json").read_text())
    actual_identity = model_identity(a.model)
    available = {r["row_id"]: r for r in records(a.data)}
    inputs = []
    for rid in contract["row_ids"]:
        row = available[rid]
        original = json.loads((folder / (rid + ".json")).read_text())
        verify_identity(contract, actual_identity, row, original)
        trace_path = folder / (rid + ".trace.pt")
        if not trace_path.is_file():
            raise FileNotFoundError(trace_path)
        inputs.append((row, original, trace_path))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if a.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent CPU fallback")
    from transformers import AutoModelForCausalLM
    dtype = getattr(torch, contract["dtype"])
    started = time.perf_counter()
    result = {"audit": "saved_future_Q_real_mass_times_raw_vnorm_v1", "status": "LOADING",
              "trace_dir": str(folder.resolve()), "model": actual_identity,
              "source_sha256": digest(Path(__file__).read_text()), "rows": [],
              "scope": "one native prefix prefill per saved input, no scoring or new generation; actual Q/sets/alpha are replayed",
              "normalization": "actual prefix probabilities = saved Full alpha * prefix-only softmax; new keys are represented by saved alpha",
              "objective_alignment": "raw V norm matches the original P/C/U proxy; conditional version separately removes actual alpha weighting",
              "limitation": "common Full-trajectory diagnostic at saved positions; neither weighted mass nor its improvement guarantees answers"}
    write_json(output, result)
    try:
        model = AutoModelForCausalLM.from_pretrained(a.model, local_files_only=True,
            torch_dtype=dtype, attn_implementation="sdpa").to(a.device).eval()
        for row, original, path in inputs:
            saved = torch.load(path, map_location="cpu", weights_only=True)
            result["rows"].append(audit_row(model, row, original, saved))
            result.update(status="RUNNING", completed_rows=len(result["rows"]), elapsed_seconds=time.perf_counter() - started)
            write_json(output, result)
        result.update(status="COMPLETE", elapsed_seconds=time.perf_counter() - started)
        write_json(output, result)
    except BaseException as error:
        result.update(status="FAILED", error_type=type(error).__name__, error=str(error))
        write_json(output, result)
        raise


if __name__ == "__main__":
    main()
