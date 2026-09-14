#!/usr/bin/env python3
"""Run the bounded E2 same-prefix attention-output intervention.

The default invocation only freezes up to eight existing divergence cases.
``--execute`` performs fresh teacher-forced GPU forwards and two recipient-side
continuations per case.  It never updates model weights or searches layers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from experiments.checkpoint_attention_replay_20260913.capture_checkpoint import head_layout
from .pipeline import atomic_json, file_sha256, prompt_hash, read_jsonl, validate_panel_rows
from .resident_eval import load_static_receipt
from .tables import model_geometry


MANIFEST_FORMAT = "ROPE_LOCAL_CAUSAL_INTERVENTION_MANIFEST_V1"
RESULT_FORMAT = "ROPE_LOCAL_CAUSAL_INTERVENTION_COMPLETE_V1"


def _score(row: dict) -> float:
    for name in ("ruler_official_score", "official_score"):
        if name in row:
            value = float(row[name])
            if math.isfinite(value):
                return value
    raise ValueError("generation row lacks a finite official score")


def _generation_map(path: Path) -> dict[str, dict]:
    result = {}
    for row in read_jsonl(path):
        identity = prompt_hash(row)
        if identity in result:
            raise ValueError(f"duplicate generation prompt in {path}: {identity}")
        tokens = row.get("generated_ids")
        if not isinstance(tokens, list) or not all(isinstance(value, int) for value in tokens):
            raise ValueError(f"generation row lacks integer generated_ids: {identity}")
        result[identity] = {**row, "official_score": _score(row)}
    return result


def first_divergence(left: list[int], right: list[int]) -> tuple[int, int | None, int | None] | None:
    shared = 0
    while shared < min(len(left), len(right)) and left[shared] == right[shared]:
        shared += 1
    if shared == len(left) == len(right):
        return None
    return (
        shared,
        left[shared] if shared < len(left) else None,
        right[shared] if shared < len(right) else None,
    )


def select_cases(
    panel: list[dict],
    recipient: dict[str, dict],
    donor: dict[str, dict],
    *,
    length: int,
    damage_limit: int = 4,
    benefit_limit: int = 2,
    concordant_limit: int = 2,
) -> list[dict]:
    """Select fixed-row-id result-conditioned cases without reading token margins."""
    panel_map = {prompt_hash(row): row for row in panel if int(row["length_cap"]) == length}
    common = set(panel_map) & set(recipient) & set(donor)
    if not common:
        raise ValueError("no paired generation rows overlap the requested diagnostic length")
    buckets: dict[str, list[dict]] = {"fwe_damage": [], "vt_benefit": [], "concordant": []}
    for identity in common:
        source = panel_map[identity]
        left, right = recipient[identity], donor[identity]
        if str(left["task"]) != str(source["task"]) or str(right["task"]) != str(source["task"]):
            raise ValueError(f"generation task differs from panel: {identity}")
        divergence = first_divergence(left["generated_ids"], right["generated_ids"])
        if divergence is None:
            continue
        offset, left_token, right_token = divergence
        # A missing token means one arm ended before a comparable argmax.  The
        # local two-token gamma contract therefore cannot be formed.
        if left_token is None or right_token is None:
            continue
        common_prefix = list(left["generated_ids"][:offset])
        if len(common_prefix) >= int(source["max_new_tokens"]):
            continue
        record = {
            "row_id": str(source["row_id"]),
            "prompt_sha256": identity,
            "task": str(source["task"]),
            "length_cap": int(source["length_cap"]),
            "prompt_ids": list(source["prompt_ids"]),
            "references": list(source["references"]),
            "max_new_tokens": int(source["max_new_tokens"]),
            "common_generated_prefix": common_prefix,
            "common_generated_tokens": offset,
            "recipient_divergent_token": int(left_token),
            "donor_divergent_token": int(right_token),
            "recipient_official_score": float(left["official_score"]),
            "donor_official_score": float(right["official_score"]),
            "recipient_original_generated_ids": list(left["generated_ids"]),
            "donor_original_generated_ids": list(right["generated_ids"]),
            "selection_uses_logits": False,
        }
        if record["task"] == "fwe" and record["recipient_official_score"] < record["donor_official_score"]:
            buckets["fwe_damage"].append(record)
        elif record["task"] == "vt" and record["recipient_official_score"] > record["donor_official_score"]:
            buckets["vt_benefit"].append(record)
        elif record["recipient_official_score"] == record["donor_official_score"]:
            buckets["concordant"].append(record)
    limits = {
        "fwe_damage": damage_limit, "vt_benefit": benefit_limit,
        "concordant": concordant_limit,
    }
    selected = []
    for category in ("fwe_damage", "vt_benefit", "concordant"):
        rows = sorted(buckets[category], key=lambda row: (row["row_id"], row["prompt_sha256"]))
        for row in rows[:limits[category]]:
            selected.append({**row, "category": category})
    return selected


def rotate_split_half(values: np.ndarray, positions: np.ndarray | float, inv_freq: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    inv = np.asarray(inv_freq, dtype=np.float64)
    if data.shape[-1] != 2 * len(inv):
        raise ValueError("rotary data and frequency dimensions differ")
    position = np.asarray(positions, dtype=np.float64)
    phase = position[..., None] * inv
    cosine, sine = np.cos(phase), np.sin(phase)
    left, right = np.split(data, 2, axis=-1)
    return np.concatenate((left * cosine - right * sine, right * cosine + left * sine), axis=-1)


def signed_pair_components(q: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``a,b`` where one pair contributes ``a*cos(delta)+b*sin(delta)``."""
    q_left, q_right = np.split(np.asarray(q, dtype=np.float64), 2, axis=-1)
    k_left, k_right = np.split(np.asarray(k, dtype=np.float64), 2, axis=-1)
    a = q_left[:, None, :] * k_left + q_right[:, None, :] * k_right
    b = q_left[:, None, :] * k_right - q_right[:, None, :] * k_left
    return a, b


def finite_attention_intervention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    query_position: int,
    recipient_inv_freq: np.ndarray,
    donor_inv_freq: np.ndarray,
    recipient_gain: float,
    donor_gain: float,
    attention_scale: float,
    output_projection: np.ndarray,
) -> dict:
    """Exact one-query finite phase intervention with full key competition and V."""
    query = np.asarray(q, dtype=np.float64)
    keys = np.asarray(k, dtype=np.float64)
    values = np.asarray(v, dtype=np.float64)
    projection = np.asarray(output_projection, dtype=np.float64)
    if query.ndim != 2 or keys.ndim != 3 or values.shape != keys.shape:
        raise ValueError("expected q[Hq,D], k/v[Hkv,T,D]")
    if query.shape[-1] != keys.shape[-1] or query.shape[0] % keys.shape[0]:
        raise ValueError("invalid GQA capture dimensions")
    if query_position < 0 or query_position >= keys.shape[1]:
        raise ValueError("query position lies outside captured visible keys")
    keys = keys[:, : query_position + 1]
    values = values[:, : query_position + 1]
    repeat = query.shape[0] // keys.shape[0]
    keys = np.repeat(keys, repeat, axis=0)
    values = np.repeat(values, repeat, axis=0)
    if projection.shape != (query.shape[0] * query.shape[1], query.shape[0] * query.shape[1]):
        raise ValueError("output projection must map concatenated heads to hidden size")
    key_positions = np.arange(query_position + 1, dtype=np.float64)

    def evaluate(inv_freq: np.ndarray, gain: float) -> dict:
        if not math.isfinite(gain) or gain <= 0.0:
            raise ValueError("RoPE gain must be finite and positive")
        q_rot = gain * rotate_split_half(query, float(query_position), inv_freq)
        k_rot = gain * rotate_split_half(keys, key_positions[None, :], inv_freq)
        logits = attention_scale * np.einsum("hd,htd->ht", q_rot, k_rot)
        shifted = logits - logits.max(axis=-1, keepdims=True)
        probability = np.exp(shifted)
        probability /= probability.sum(axis=-1, keepdims=True)
        context = np.einsum("ht,htd->hd", probability, values)
        projected = projection @ context.reshape(-1)
        a, b = signed_pair_components(query, keys)
        lag = query_position - key_positions
        phase = lag[None, :, None] * np.asarray(inv_freq, dtype=np.float64)[None, None, :]
        signed_logits = gain ** 2 * attention_scale * np.sum(
            a * np.cos(phase) + b * np.sin(phase), axis=-1,
        )
        return {
            "logits": logits,
            "probability": probability,
            "context": context,
            "attention_output": projected,
            "signed_formula_max_abs_error": float(np.max(np.abs(signed_logits - logits))),
        }

    recipient = evaluate(recipient_inv_freq, recipient_gain)
    phase_only = evaluate(donor_inv_freq, donor_gain)
    return {
        "recipient": recipient,
        "phase_only": phase_only,
        "delta_logits": phase_only["logits"] - recipient["logits"],
        "delta_probability": phase_only["probability"] - recipient["probability"],
        "delta_attention_output": phase_only["attention_output"] - recipient["attention_output"],
    }


def tensor_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _attention_tensor(output):
    return output[0] if isinstance(output, tuple) else output


def _replace_attention_tensor(output, replacement):
    tensor = _attention_tensor(output)
    patched = tensor.clone()
    patched[:, -1, :] = replacement.to(device=tensor.device, dtype=tensor.dtype)
    if isinstance(output, tuple):
        return (patched, *output[1:])
    return patched


def capture_forward(model, input_ids, *, layer: int) -> dict:
    """Capture one real last-query attention node and its next-token logits."""
    import torch

    block = model.model.layers[layer]
    attention = block.self_attn
    heads = int(model.config.num_attention_heads)
    kv_heads = int(getattr(model.config, "num_key_value_heads", heads))
    head_dim = int(model.config.hidden_size // heads)
    tokens = int(input_ids.shape[1])
    captured: dict[str, np.ndarray] = {}
    handles = []

    def projection_hook(name: str, count: int, *, last_only: bool = False):
        def hook(_module, _inputs, output):
            layout = head_layout(output, tokens=tokens, heads=count, head_dim=head_dim)
            value = layout[-1] if last_only else layout.permute(1, 0, 2)
            captured[name] = value.detach().cpu().float().numpy()
        return hook

    def attention_hook(_module, _inputs, output):
        captured["attention_output"] = (
            _attention_tensor(output)[0, -1].detach().cpu().float().numpy()
        )

    q_module = getattr(attention, "q_norm", None) or attention.q_proj
    k_module = getattr(attention, "k_norm", None) or attention.k_proj
    handles.append(q_module.register_forward_hook(projection_hook("q", heads, last_only=True)))
    handles.append(k_module.register_forward_hook(projection_hook("k", kv_heads)))
    handles.append(attention.v_proj.register_forward_hook(projection_hook("v", kv_heads)))
    handles.append(attention.register_forward_hook(attention_hook))
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(
                input_ids=input_ids,
                attention_mask=input_ids.new_ones(input_ids.shape),
                use_cache=False,
                logits_to_keep=1,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != {"q", "k", "v", "attention_output"}:
        raise RuntimeError(f"incomplete attention capture: {sorted(captured)}")
    captured["raw_logits"] = output.logits[0, -1].detach().cpu().float().numpy()
    captured["attention_scale"] = np.asarray(
        float(getattr(attention, "scaling", 1.0 / math.sqrt(head_dim))), dtype=np.float64,
    )
    return captured


def patched_continuation(
    model,
    input_ids,
    *,
    layer: int,
    replacement: np.ndarray,
    max_new_tokens: int,
    eos_ids: set[int],
) -> dict:
    """Patch one current-query attention output, then continue real greedy decode."""
    import torch

    if max_new_tokens < 1:
        raise ValueError("patched continuation needs at least one remaining token")
    block = model.model.layers[layer]
    replacement_tensor = torch.from_numpy(np.asarray(replacement, dtype=np.float32))

    def hook(_module, _inputs, output):
        return _replace_attention_tensor(output, replacement_tensor)

    handle = block.self_attn.register_forward_hook(hook)
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(
                input_ids=input_ids,
                attention_mask=input_ids.new_ones(input_ids.shape),
                use_cache=True,
                logits_to_keep=1,
                return_dict=True,
            )
    finally:
        handle.remove()
    logits = output.logits[:, -1, :].float()
    initial_logits = logits[0].detach().cpu().numpy()
    cache = output.past_key_values
    total = int(input_ids.shape[1])
    generated = []
    for _ in range(max_new_tokens):
        token = logits.argmax(dim=-1)
        value = int(token.item())
        generated.append(value)
        if value in eos_ids:
            break
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(
                input_ids=token[:, None],
                attention_mask=input_ids.new_ones((1, total + 1)),
                past_key_values=cache,
                use_cache=True,
                cache_position=torch.tensor([total], device=input_ids.device),
                logits_to_keep=1,
                return_dict=True,
            )
        cache = output.past_key_values
        logits = output.logits[:, -1, :].float()
        total += 1
    return {"generated_ids": generated, "initial_raw_logits": initial_logits}


def _array_receipt(path: Path, value: np.ndarray) -> dict:
    return {
        "path": path.name,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": file_sha256(path),
        "bytes": int(value.nbytes),
    }


def save_case(directory: Path, arrays: dict[str, np.ndarray], receipt: dict) -> dict:
    if directory.exists() or directory.with_name(directory.name + ".incomplete").exists():
        raise FileExistsError(directory)
    temporary = directory.with_name(directory.name + ".incomplete")
    temporary.mkdir(parents=True)
    array_metadata = {}
    for name, raw in arrays.items():
        value = np.asarray(raw)
        path = temporary / f"{name}.npy"
        np.save(path, value, allow_pickle=False)
        array_metadata[name] = _array_receipt(path, value)
    payload = {**receipt, "arrays": array_metadata}
    atomic_json(temporary / "receipt.json", payload)
    temporary.replace(directory)
    return payload


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.exists() else 0


def _two_token_margin(logits: np.ndarray, token_a: int, token_b: int) -> float:
    values = np.asarray(logits)
    if min(token_a, token_b) < 0 or max(token_a, token_b) >= len(values):
        raise ValueError("divergent token lies outside logits vocabulary")
    return float(values[token_a] - values[token_b])


def execute_cases(
    *,
    manifest: dict,
    model_path: Path,
    recipient_table_path: Path,
    donor_table_path: Path,
    out: Path,
    parity_atol: float,
    max_capture_bytes: int,
) -> dict:
    import torch
    from transformers import AutoTokenizer

    from experiments.olmo_recovery_20260912.recovery_v2_runtime import load_model
    from experiments.olmo_recovery_20260912.runtime import validate_cuda
    from scripts.eval.longbench_metrics import qa_f1_score
    from scripts.experiments.cross_audit.tables import install_static, verify_static
    from scripts.experiments.olmo_fast_screen.ruler_bench import score as ruler_score

    validate_cuda()
    model, wrapper, _ = load_model(model_path, "Native", checkpoint=None, training=False)
    model.requires_grad_(False)
    if wrapper is not None or model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("causal intervention requires one frozen eval model")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    pairs = int(model.config.hidden_size // model.config.num_attention_heads // 2)
    model_id = manifest["model_id"]
    recipient_receipt, recipient_table = load_static_receipt(
        recipient_table_path, model_id=model_id, pairs=pairs,
    )
    donor_receipt, donor_table = load_static_receipt(
        donor_table_path, model_id=model_id, pairs=pairs,
    )
    if float(recipient_table["gain"]) != float(donor_table["gain"]):
        raise ValueError("E2 requires same-gain recipient and donor tables")
    layer = int(manifest["layer"])
    attention = model.model.layers[layer].self_attn
    projection = attention.o_proj.weight.detach().cpu().float().numpy()
    projection_hash = tensor_sha256(projection)
    eos_value = model.generation_config.eos_token_id
    eos_ids = set(eos_value if isinstance(eos_value, list) else [eos_value])
    eos_ids.discard(None)
    device = next(model.parameters()).device
    cases_root = out / "cases"
    cases_root.mkdir(exist_ok=True)
    records = []
    for index, case in enumerate(manifest["cases"]):
        case_dir = cases_root / (
            f"case_{index:02d}_{case['category']}_{case['prompt_sha256'][:12]}"
        )
        if case_dir.exists():
            existing = json.loads((case_dir / "receipt.json").read_text())
            records.append({
                "path": str(case_dir), "receipt_sha256": file_sha256(case_dir / "receipt.json"),
                "row_id": existing["row_id"], "category": existing["category"],
            })
            continue
        prefix_tokens = list(case["prompt_ids"]) + list(case["common_generated_prefix"])
        input_ids = torch.tensor([prefix_tokens], dtype=torch.long, device=device)
        install_static(model, np.asarray(recipient_table["values_float32"], dtype=np.float32), recipient_table["gain"])
        verify_static(model, np.asarray(recipient_table["values_float32"], dtype=np.float32), recipient_table["gain"])
        recipient_state = capture_forward(model, input_ids, layer=layer)
        install_static(model, np.asarray(donor_table["values_float32"], dtype=np.float32), donor_table["gain"])
        verify_static(model, np.asarray(donor_table["values_float32"], dtype=np.float32), donor_table["gain"])
        donor_state = capture_forward(model, input_ids, layer=layer)
        token_a = int(case["recipient_divergent_token"])
        token_b = int(case["donor_divergent_token"])
        recipient_argmax = int(np.argmax(recipient_state["raw_logits"]))
        donor_argmax = int(np.argmax(donor_state["raw_logits"]))
        if recipient_argmax != token_a or donor_argmax != token_b:
            raise RuntimeError(
                f"generation parity failed for {case['row_id']}: "
                f"captured={recipient_argmax}/{donor_argmax}, saved={token_a}/{token_b}"
            )
        gamma_recipient = _two_token_margin(recipient_state["raw_logits"], token_a, token_b)
        gamma_donor = _two_token_margin(donor_state["raw_logits"], token_a, token_b)
        if not gamma_recipient > 0.0 or not gamma_donor < 0.0:
            raise RuntimeError(f"two-token gamma signs do not explain saved divergence: {case['row_id']}")
        finite = finite_attention_intervention(
            recipient_state["q"], recipient_state["k"], recipient_state["v"],
            query_position=len(prefix_tokens) - 1,
            recipient_inv_freq=np.asarray(recipient_table["values_float32"], dtype=np.float32),
            donor_inv_freq=np.asarray(donor_table["values_float32"], dtype=np.float32),
            recipient_gain=float(recipient_table["gain"]), donor_gain=float(donor_table["gain"]),
            attention_scale=float(recipient_state["attention_scale"]),
            output_projection=projection,
        )
        identity_error = float(np.max(np.abs(
            finite["recipient"]["attention_output"] - recipient_state["attention_output"]
        )))
        if identity_error > parity_atol:
            raise RuntimeError(
                f"captured attention replay parity failed for {case['row_id']}: {identity_error}"
            )
        common_count = int(case["common_generated_tokens"])
        remaining = int(case["max_new_tokens"]) - common_count
        install_static(model, np.asarray(recipient_table["values_float32"], dtype=np.float32), recipient_table["gain"])
        phase = patched_continuation(
            model, input_ids, layer=layer,
            replacement=finite["phase_only"]["attention_output"],
            max_new_tokens=remaining, eos_ids=eos_ids,
        )
        donor_output = patched_continuation(
            model, input_ids, layer=layer,
            replacement=donor_state["attention_output"],
            max_new_tokens=remaining, eos_ids=eos_ids,
        )

        def score_continuation(result: dict) -> dict:
            complete_ids = list(case["common_generated_prefix"]) + list(result["generated_ids"])
            decode_ids = complete_ids[:-1] if complete_ids and complete_ids[-1] in eos_ids else complete_ids
            text = tokenizer.decode(
                decode_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False,
            )
            row = {
                "task": case["task"], "references": case["references"],
                "length_cap": case["length_cap"],
            }
            return {
                "generated_ids": complete_ids,
                "output_text": text,
                "ruler_official_score": float(ruler_score(row, text)),
                "whole_response_f1": float(qa_f1_score(text, case["references"])),
                "ended_eos": bool(complete_ids and complete_ids[-1] in eos_ids),
                "hit_cap": len(complete_ids) == int(case["max_new_tokens"])
                    and not (complete_ids and complete_ids[-1] in eos_ids),
                "first_token": int(result["generated_ids"][0]),
                "gamma_a_minus_b": _two_token_margin(
                    result["initial_raw_logits"], token_a, token_b,
                ),
            }

        phase_score = score_continuation(phase)
        donor_output_score = score_continuation(donor_output)
        arrays = {
            "recipient_q": recipient_state["q"].astype(np.float32),
            "recipient_k": recipient_state["k"].astype(np.float32),
            "recipient_v": recipient_state["v"].astype(np.float32),
            "donor_q": donor_state["q"].astype(np.float32),
            "donor_k": donor_state["k"].astype(np.float32),
            "donor_v": donor_state["v"].astype(np.float32),
            "recipient_attention_output": recipient_state["attention_output"].astype(np.float32),
            "donor_attention_output": donor_state["attention_output"].astype(np.float32),
            "phase_only_attention_output": finite["phase_only"]["attention_output"].astype(np.float32),
            "recipient_inv_freq": np.asarray(recipient_table["values_float32"], dtype=np.float32),
            "donor_inv_freq": np.asarray(donor_table["values_float32"], dtype=np.float32),
            "recipient_divergence_logits": recipient_state["raw_logits"].astype(np.float32),
            "donor_divergence_logits": donor_state["raw_logits"].astype(np.float32),
            "phase_only_divergence_logits": phase["initial_raw_logits"].astype(np.float32),
            "donor_output_divergence_logits": donor_output["initial_raw_logits"].astype(np.float32),
        }
        projected_size = _directory_bytes(out) + sum(value.nbytes + 256 for value in arrays.values())
        if projected_size > max_capture_bytes:
            raise RuntimeError("diagnostic capture would exceed the frozen on-disk byte cap")
        receipt = save_case(case_dir, arrays, {
            "status": "ROPE_LOCAL_CAUSAL_INTERVENTION_CASE_V1",
            "row_id": case["row_id"], "category": case["category"], "task": case["task"],
            "prompt_sha256": case["prompt_sha256"], "layer": layer,
            "query_position": len(prefix_tokens) - 1,
            "visible_keys": len(prefix_tokens),
            "recipient_table_sha256_float32": recipient_receipt["table_sha256_float32"],
            "donor_table_sha256_float32": donor_receipt["table_sha256_float32"],
            "gain": float(recipient_table["gain"]),
            "attention_scale": float(recipient_state["attention_scale"]),
            "output_projection_sha256_float32": projection_hash,
            "identity_replay_max_abs_error": identity_error,
            "signed_formula_max_abs_error": {
                "recipient": finite["recipient"]["signed_formula_max_abs_error"],
                "phase_only": finite["phase_only"]["signed_formula_max_abs_error"],
            },
            "gamma": {"recipient": gamma_recipient, "donor": gamma_donor},
            "phase_only": phase_score,
            "donor_output": donor_output_score,
            "original": {
                "recipient_score": case["recipient_official_score"],
                "donor_score": case["donor_official_score"],
                "recipient_generated_ids": case["recipient_original_generated_ids"],
                "donor_generated_ids": case["donor_original_generated_ids"],
            },
            "decoder_contract": "raw greedy argmax; no logits processors; original row token cap and EOS",
            "scope": "one preregistered layer/query local diagnostic; not a deployable table",
        })
        records.append({
            "path": str(case_dir), "receipt_sha256": file_sha256(case_dir / "receipt.json"),
            "row_id": receipt["row_id"], "category": receipt["category"],
        })
        atomic_json(out / "partial_index.json", {
            "status": "ROPE_LOCAL_CAUSAL_INTERVENTION_PARTIAL_V1", "cases": records,
        })
        del input_ids, recipient_state, donor_state, finite, phase, donor_output
        torch.cuda.empty_cache()
    return {
        "status": RESULT_FORMAT,
        "manifest_sha256": file_sha256(out / "diagnostic_manifest.json"),
        "cases": records,
        "capture_bytes": _directory_bytes(cases_root),
        "max_capture_bytes": max_capture_bytes,
        "model_weight_updates": 0,
        "scope": "bounded same-prefix middle-layer current-query intervention",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--recipient-generations", type=Path, required=True)
    parser.add_argument("--donor-generations", type=Path, required=True)
    parser.add_argument("--recipient-table", type=Path, required=True)
    parser.add_argument("--donor-table", type=Path, required=True)
    parser.add_argument("--length", type=int, default=8192)
    parser.add_argument("--layer", type=int)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parity-atol", type=float, default=0.05)
    parser.add_argument("--max-capture-bytes", type=int, default=2 * 1024 ** 3)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    required_files = (
        args.model / "config.json", args.panel, args.recipient_generations,
        args.donor_generations, args.recipient_table, args.donor_table,
    )
    if any(not path.is_file() for path in required_files):
        raise FileNotFoundError([str(path) for path in required_files if not path.is_file()])
    config = json.loads((args.model / "config.json").read_text())
    geometry = model_geometry(config)
    if geometry["model_type"] != "llama":
        raise ValueError("today's E2 contract is preregistered only for the Llama checkpoint")
    layer = int(config["num_hidden_layers"] // 2 if args.layer is None else args.layer)
    if not 0 <= layer < int(config["num_hidden_layers"]):
        raise ValueError("diagnostic layer is outside the checkpoint")
    panel = validate_panel_rows(read_jsonl(args.panel), "e2_source")
    recipient = _generation_map(args.recipient_generations)
    donor = _generation_map(args.donor_generations)
    cases = select_cases(panel, recipient, donor, length=args.length)
    recipient_payload = json.loads(args.recipient_table.read_text())
    donor_payload = json.loads(args.donor_table.read_text())
    if recipient_payload.get("model_id") != donor_payload.get("model_id"):
        raise ValueError("recipient and donor receipts name different models")
    model_id = str(recipient_payload.get("model_id"))
    manifest = {
        "status": MANIFEST_FORMAT,
        "model": str(args.model.resolve()), "model_id": model_id,
        "model_config_sha256": file_sha256(args.model / "config.json"),
        "model_geometry": geometry,
        "panel": str(args.panel.resolve()), "panel_sha256": file_sha256(args.panel),
        "recipient_generations": str(args.recipient_generations.resolve()),
        "recipient_generations_sha256": file_sha256(args.recipient_generations),
        "donor_generations": str(args.donor_generations.resolve()),
        "donor_generations_sha256": file_sha256(args.donor_generations),
        "recipient_table": str(args.recipient_table.resolve()),
        "recipient_table_receipt_sha256": file_sha256(args.recipient_table),
        "donor_table": str(args.donor_table.resolve()),
        "donor_table_receipt_sha256": file_sha256(args.donor_table),
        "layer": layer, "length": args.length,
        "selection": {
            "rule": "row-id-first among existing divergent outputs: 4 FWE damage, 2 VT benefit, 2 equal-score controls",
            "selected": len(cases),
            "counts": {
                name: sum(case["category"] == name for case in cases)
                for name in ("fwe_damage", "vt_benefit", "concordant")
            },
            "result_conditioned_mechanism_sample_not_performance_estimate": True,
        },
        "cases": cases,
        "interventions": ["phase_only", "donor_output"],
        "same_gain_required": True,
        "max_capture_bytes": args.max_capture_bytes,
        "model_weight_updates": 0,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "diagnostic_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("diagnostic output contains another frozen manifest")
    atomic_json(manifest_path, manifest)
    if not args.execute:
        print(json.dumps({
            "status": "PLAN_ONLY", "cases": len(cases), "counts": manifest["selection"]["counts"],
            "layer": layer,
        }, sort_keys=True))
        return
    complete_path = args.out / "index.json"
    if complete_path.exists():
        complete = json.loads(complete_path.read_text())
        if complete.get("status") != RESULT_FORMAT or complete.get("manifest_sha256") != file_sha256(manifest_path):
            raise ValueError("completed diagnostic index differs from the frozen manifest")
        print(json.dumps({"status": complete["status"], "cases": len(complete["cases"])}, sort_keys=True))
        return
    result = execute_cases(
        manifest=manifest, model_path=args.model, recipient_table_path=args.recipient_table,
        donor_table_path=args.donor_table, out=args.out, parity_atol=args.parity_atol,
        max_capture_bytes=args.max_capture_bytes,
    )
    atomic_json(complete_path, result)
    print(json.dumps({
        "status": result["status"], "cases": len(result["cases"]),
        "capture_bytes": result["capture_bytes"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
