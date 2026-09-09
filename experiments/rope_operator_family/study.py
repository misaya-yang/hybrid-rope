"""Capture, fit and diagnose one operator-family configuration at a time."""
from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor

from .operator import OperatorFactors, Shape, native_frequencies, native_response


def digest(path: str | Path) -> str:
    sha = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def write_json(path: str | Path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def load_rows(path: str | Path):
    with Path(path).open() as source:
        return [json.loads(line) for line in source if line.strip()]


def model_shape(model) -> dict:
    config = model.config
    if getattr(config, "use_sliding_window", False):
        raise ValueError("this adapter currently targets full-attention RoPE models")
    if getattr(config, "model_type", None) not in ("qwen2", "llama"):
        raise ValueError("prepared model adapter supports Qwen2/Llama full RoPE attention")
    rope = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None) or {}
    if rope.get("rope_type", rope.get("type", "default")) != "default":
        raise ValueError("scaled/dynamic RoPE needs its own native operator implementation")
    attention = model.model.layers[0].self_attn
    if hasattr(attention, "q_norm") or hasattr(attention, "k_norm"):
        raise ValueError("Q/K normalization is not included in this affine adapter")
    d = attention.head_dim
    return dict(heads=config.num_attention_heads, kv_heads=config.num_key_value_heads,
                head_dim=d, theta=rope.get("rope_theta", getattr(config, "rope_theta", 10000.0)))


@torch.inference_mode()
def capture(model, rows: list[dict], output: str | Path, queries: int = 64):
    """Original-model Q/K/V. Store sampled Q rows and all causal keys/values."""
    output = Path(output)
    if (output / "manifest.json").exists():
        raise FileExistsError(f"capture already exists: {output}; reuse it instead of overwriting")
    output.mkdir(parents=True, exist_ok=True)
    shape = model_shape(model)
    device = model.model.embed_tokens.weight.device
    handles, captured = [], {}
    selection = None
    for layer_index, layer in enumerate(model.model.layers):
        for kind in ("q", "k", "v"):
            def hook(_module, _input, result, layer_index=layer_index, kind=kind):
                tensor = result[0]
                if kind == "q":
                    tensor = tensor[selection].reshape(-1, shape["heads"], shape["head_dim"])
                captured.setdefault(layer_index, {})[kind] = tensor.detach().cpu().contiguous()
            handles.append(getattr(layer.self_attn, f"{kind}_proj").register_forward_hook(hook))
    manifest = dict(status="running", model_shape=shape, layers=len(model.model.layers), queries=queries, records=[])
    started = time.monotonic()
    try:
        for index, row in enumerate(rows):
            ids = torch.tensor(row["input_ids"], device=device).unsqueeze(0)
            if ids.shape[1] < 2:
                raise ValueError("capture needs at least two tokens")
            selection = torch.linspace(1, ids.shape[1] - 1, min(queries, ids.shape[1] - 1), device=device).long().unique()
            captured.clear()
            # Skip the large vocabulary logits; the decoder emits every Q/K/V hook.
            model.model(input_ids=ids, use_cache=False)
            for layer_index in range(manifest["layers"]):
                relative = f"layer_{layer_index:03d}/record_{index:05d}.pt"
                path = output / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                record = captured[layer_index]
                record.update(query_positions=selection.cpu(), key_positions=torch.arange(ids.shape[1]),
                              id=str(row["id"]), source_id=str(row.get("source_id", row["id"])),
                              split=row["split"])
                torch.save(record, path)
            manifest["records"].append({"index": index, "id": str(row["id"]), "source_id": str(row.get("source_id", row["id"])),
                                        "split": row["split"], "tokens": ids.shape[1], "queries": len(selection)})
            manifest["seconds"] = time.monotonic() - started
            write_json(output / "progress.json", manifest)
            print(json.dumps({"stage": "capture", "documents": index + 1, "seconds": manifest["seconds"]}), flush=True)
    finally:
        for handle in handles:
            handle.remove()
    manifest.update(status="complete", seconds=time.monotonic() - started)
    write_json(output / "manifest.json", manifest)
    return manifest


def records_for(directory: str | Path, layer: int, split: str | None = None):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("capture did not finish")
    return [directory / f"layer_{layer:03d}/record_{r['index']:05d}.pt" for r in manifest["records"]
            if split is None or r["split"] == split]


def load_record(path: str | Path, device="cpu"):
    record = torch.load(path, map_location="cpu", weights_only=True)
    return {k: v.to(device=device, dtype=torch.float32 if k in ("q", "k", "v") else v.dtype)
            if isinstance(v, Tensor) else v for k, v in record.items()}


def responses(factors, record, position_scale=1.0):
    q, k, v = (record[name] for name in ("q", "k", "v"))
    original_qp, original_kp = record["query_positions"], record["key_positions"]
    # Causality belongs to token order, even if all phases are set to zero.
    valid = original_kp[None, :] <= original_qp[:, None]
    qp, kp = (original_qp.float() * position_scale).round(), (original_kp.float() * position_scale).round()
    with torch.no_grad():
        teacher = native_response(q, k, v, qp, kp, factors.shape, valid)
    student = factors.response(q, k, v, qp, kp, valid)
    return teacher, student


def squared_metrics(teacher, student):
    ts, to, tv, valid = teacher
    ss, so, sv, _ = student
    error, reference = (ss - ts)[:, valid], ts[:, valid]
    score = error.square().mean()
    output = (so - to).square().mean()
    value = (sv - tv).square().mean()
    return dict(score_mse=score, relative_score_mse=score / reference.square().mean().clamp_min(1e-8),
                output_mse=output, relative_output_mse=output / to.square().mean().clamp_min(1e-8),
                value_mse=value, relative_value_mse=value / tv.square().mean().clamp_min(1e-8))


@dataclass
class FitConfig:
    steps: int = 500
    learning_rate: float = 1e-3
    phase_learning_rate: float = 0.01
    max_position_scale: float = 8.0
    learn_projections: bool = True
    learn_frequency: bool = True
    learn_content: bool = True
    value_weight: float = 1.0
    output_weight: float = 0.0
    seed: int = 42


def fit_layer(factors: OperatorFactors, paths: list[Path], config: FitConfig, output: str | Path):
    """One configuration, one optimization trajectory. No arm/search loop."""
    if not paths or config.steps < 1 or config.max_position_scale < 1:
        raise ValueError("fit needs calibration records, positive steps and a scale >= 1")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(config.seed)
    device = factors.A.device
    groups = []
    for names, enabled, lr in ((["A", "B"], config.learn_projections, config.learning_rate),
                               (["C", "P", "U"], config.learn_content, config.learning_rate),
                               (["phase"], config.learn_frequency, config.phase_learning_rate)):
        parameters = [getattr(factors, name) for name in names]
        for parameter in parameters:
            parameter.requires_grad_(enabled)
        if enabled:
            groups.append({"params": parameters, "lr": lr})
    if not groups:
        raise ValueError("at least one part of the method must be learnable")
    optimizer = torch.optim.Adam(groups)
    started = time.monotonic()
    if (output / "fit.jsonl").exists():
        (output / "fit.jsonl").replace(output / f"fit_previous_attempt_{time.time_ns()}.jsonl")
    with (output / "fit.jsonl").open("w") as log:
        for step in range(config.steps):
            path = paths[step % len(paths)] if step < len(paths) else rng.choice(paths)
            record = load_record(path, device)
            scale = 1.0 if step % 2 == 0 else math.exp(rng.uniform(0, math.log(config.max_position_scale)))
            teacher, student = responses(factors, record, scale)
            metrics = squared_metrics(teacher, student)
            loss = metrics["relative_score_mse"] + config.value_weight * metrics["relative_value_mse"] + config.output_weight * metrics["relative_output_mse"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss at layer fit step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(factors.parameters(), 1.0)
            optimizer.step()
            item = {key: float(value.detach()) for key, value in metrics.items()}
            item.update(step=step + 1, document=record["id"], position_scale=scale,
                        loss=float(loss.detach()), gradient_norm=float(gradient_norm), seconds=time.monotonic() - started)
            log.write(json.dumps(item) + "\n")
            if step == 0 or (step + 1) % 50 == 0 or step + 1 == config.steps:
                log.flush()
                print(json.dumps({"stage": "fit", **item}), flush=True)
    return {"configuration": asdict(config), "seconds": time.monotonic() - started, "last": item}


@torch.inference_mode()
def diagnose(factors: OperatorFactors, paths: list[Path], position_scale: float = 1.0):
    """Content/phase and full-output diagnostics for this one fitted operator."""
    rows = []
    for path in paths:
        record = load_record(path, factors.A.device)
        teacher, student = responses(factors, record, position_scale)
        ts, to, _tv, valid = teacher
        ss, so, _sv, _ = student
        tp = ts.masked_fill(~valid[None], -torch.inf).log_softmax(-1)
        sp = ss.masked_fill(~valid[None], -torch.inf).log_softmax(-1)
        kl = (tp.exp() * (tp.masked_fill(~valid[None], 0) - sp.masked_fill(~valid[None], 0))).sum(-1).mean()
        item = {k: float(v) for k, v in squared_metrics(teacher, student).items()}
        static_teacher, static_student = responses(factors, record, 0.0)
        static_error = squared_metrics(static_teacher, static_student)["score_mse"]
        teacher_change = ts - static_teacher[0]
        student_change = ss - static_student[0]
        position_error = student_change - teacher_change
        position_mse = position_error[:, valid].square().mean()
        item.update(id=record["id"], source_id=record["source_id"], attention_kl=float(kl),
                    static_score_mse=float(static_error),
                    position_response_mse=float(position_mse),
                    relative_position_response_mse=float(position_mse / teacher_change[:, valid].square().mean().clamp_min(1e-8)),
                    log_mass_mae=float((ts.masked_fill(~valid[None], -torch.inf).logsumexp(-1) -
                                        ss.masked_fill(~valid[None], -torch.inf).logsumexp(-1)).abs().mean()))
        eligible = valid.sum(-1) >= 2
        if eligible.any():
            top = ts[:, eligible].masked_fill(~valid[eligible][None], -torch.inf).topk(2, dim=-1)
            student_top = ss[:, eligible].gather(-1, top.indices)
            item["teacher_top2_margin_mae"] = float(((top.values[..., 0] - top.values[..., 1]) -
                                                     (student_top[..., 0] - student_top[..., 1])).abs().mean())
        distance = (record["query_positions"].float() * position_scale).round()[:, None] - (record["key_positions"].float() * position_scale).round()[None, :]
        item["distance_bins"] = []
        for lo, hi in ((0, 256), (256, 2048), (2048, 4096), (4096, 16384), (16384, math.inf)):
            use = valid & (distance >= lo) & (distance < hi)
            if use.any():
                item["distance_bins"].append({"min": lo, "max_exclusive": None if math.isinf(hi) else hi,
                                               "pairs": int(use.sum()), "score_mse": float((ss - ts)[:, use].square().mean()),
                                               "position_response_mse": float(position_error[:, use].square().mean())})
        rows.append(item)
    scalar_keys = [key for key in rows[0] if isinstance(rows[0][key], (float, int))] if rows else []
    return {"position_scale": position_scale, "documents": len(rows), "rows": rows,
            "means": {key: sum(row[key] for row in rows if key in row) / sum(key in row for row in rows) for key in scalar_keys}}


@torch.inference_mode()
def generator_diagnostics(factors: OperatorFactors):
    """Geometric mechanism measurements; these are not model-quality guarantees."""
    s = factors.shape
    frequencies = native_frequencies(s, device=factors.B.device, dtype=factors.B.dtype)
    generator = torch.zeros(s.kv_width, s.kv_width, device=factors.B.device, dtype=factors.B.dtype)
    pair = torch.arange(s.head_dim // 2, device=factors.B.device)
    for head in range(s.kv_heads):
        real, imag = head * s.head_dim + pair, head * s.head_dim + s.head_dim // 2 + pair
        generator[real, imag], generator[imag, real] = -frequencies, frequencies
    compressed = torch.zeros(s.rotary_dim, s.rotary_dim, device=generator.device, dtype=generator.dtype)
    index = torch.arange(s.rotary_dim // 2, device=generator.device)
    compressed[2 * index, 2 * index + 1] = -factors.frequencies
    compressed[2 * index + 1, 2 * index] = factors.frequencies
    residual = generator @ factors.B - factors.B @ compressed
    basis, singular, _ = torch.linalg.svd(factors.B, full_matrices=False)
    tolerance = torch.finfo(singular.dtype).eps * max(factors.B.shape) * singular[0]
    rank = int((singular > tolerance).sum())
    basis = basis[:, :rank]
    leakage = generator @ basis - basis @ (basis.T @ generator @ basis)
    return dict(key_projection_rank=rank, key_projection_norm=float(factors.B.norm()),
                generator_intertwining_residual_frobenius=float(residual.norm()),
                relative_generator_residual=float(residual.norm() / factors.B.norm().clamp_min(1e-12)),
                subspace_leakage_frobenius=float(leakage.norm()),
                subspace_leakage_spectral=float(torch.linalg.matrix_norm(leakage, ord=2)) if rank else 0.0)
