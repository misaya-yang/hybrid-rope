#!/usr/bin/env python3
"""Evaluate one seed-42 LoRA substrate with the pinned official YaRN equations.

The evaluator consumes the frozen capability-suite manifest produced by
``prepare_seed42_capability_data.py``.  It scores every registered task with
gold-answer NLL, adds likelihood-normalized MCQA accuracy, and optionally runs
greedy generation for retrieval and long-document QA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import string
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

try:
    from .prepare_seed42_capability_data import (
        MANIFEST_SCHEMA,
        load_jsonl,
        validate_records,
    )
    from .train_positional_distill import causal_backbone
    from .train_evq_lora import (
        build_training_inv_freq,
        find_rotary_modules,
        load_frequency_artifact,
        resolve_model_rope_geometry,
        verify_model_inv_freq,
    )
except ImportError:
    from prepare_seed42_capability_data import MANIFEST_SCHEMA, load_jsonl, validate_records
    from train_positional_distill import causal_backbone
    from train_evq_lora import (
        build_training_inv_freq,
        find_rotary_modules,
        load_frequency_artifact,
        resolve_model_rope_geometry,
        verify_model_inv_freq,
    )

from scripts.lib.rope.official_yarn import official_yarn_on_inv_freq
from experiments.lora_evq_v2.prepare_legacy_model_manifest import (
    validate_model_manifest,
)
from experiments.lora_evq_v2.prepare_positional_distill_data import (
    tokenizer_source_fingerprint,
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().to(torch.float64).contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def parse_yarn_factors(value: str | Iterable[float]) -> tuple[float, ...]:
    pieces = value.split(",") if isinstance(value, str) else list(value)
    factors = tuple(float(piece) for piece in pieces)
    if factors != (2.0, 4.0):
        raise ValueError("registered official-YaRN capability evaluation requires factors 2 and 4")
    return factors


def capability_arm_contract(substrate: str, factors: Sequence[float]) -> dict[str, Any]:
    if tuple(float(value) for value in factors) != (2.0, 4.0):
        raise ValueError("arm contract requires official YaRN factors 2 and 4")
    if substrate == "native_geo":
        return {
            "substrate": substrate,
            "adapter": "geo_longalpaca_s42",
            "operator": "official_yarn",
            "factors": [2.0, 4.0],
            "seed": 42,
        }
    if substrate == "evq_cosh":
        return {
            "substrate": substrate,
            "adapter": "evq_longalpaca_tau1414_s42",
            "operator": "official_yarn_on_evq",
            "factors": [2.0, 4.0],
            "seed": 42,
            "tau": 1.414,
        }
    raise ValueError(f"unsupported substrate: {substrate!r}")


def validate_adapter_identity_metadata(
    metadata: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    substrate: str,
    training_manifest_sha256: str,
    adapter_sha256: str,
) -> dict[str, Any]:
    if metadata.get("objective") != "legacy_longalign_full_token_causal_lm_v2":
        raise ValueError("adapter objective mismatch")
    if metadata.get("status") != "complete" or int(metadata.get("global_step", -1)) != 300:
        raise ValueError("adapter is not a completed step-300 artifact")
    protocol = metadata.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("adapter protocol is missing")
    if protocol.get("method") != substrate:
        raise ValueError("adapter method mismatch")
    if int(protocol.get("seed", -1)) != 42:
        raise ValueError("adapter seed mismatch")
    for record in (metadata, protocol):
        if record.get("data_manifest_sha256") != training_manifest_sha256:
            raise ValueError("adapter training-data manifest mismatch")
    if metadata.get("adapter_sha256") != adapter_sha256:
        raise ValueError("adapter file hash mismatch")
    if int(config.get("r", -1)) != 64 or int(config.get("lora_alpha", -1)) != 128:
        raise ValueError("adapter rank/alpha mismatch")
    if not math.isclose(float(config.get("lora_dropout", -1.0)), 0.05, abs_tol=1e-12):
        raise ValueError("adapter dropout mismatch")
    if set(config.get("target_modules", [])) != {"q_proj", "k_proj", "v_proj", "o_proj"}:
        raise ValueError("adapter target modules mismatch")
    return dict(metadata)


def validate_adapter_identity(
    adapter_dir: str | Path,
    *,
    substrate: str,
    training_manifest_sha256: str,
) -> dict[str, Any]:
    root = Path(adapter_dir)
    adapter_path = root / "adapter_model.safetensors"
    config_path = root / "adapter_config.json"
    metadata_path = root / "experiment_meta.json"
    for path in (adapter_path, config_path, metadata_path, root / "custom_inv_freq.pt"):
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return validate_adapter_identity_metadata(
        metadata,
        config,
        substrate=substrate,
        training_manifest_sha256=training_manifest_sha256,
        adapter_sha256=sha256_file(adapter_path),
    )


def apply_official_yarn_runtime(
    model: torch.nn.Module,
    substrate_inv_freq: torch.Tensor,
    *,
    head_dim: int,
    base: float,
    factor: float,
    original_max_position_embeddings: int,
) -> dict[str, Any]:
    """Apply the full pinned YaRN frequency transform and cos/sin mscale."""
    transformed, mscale, operator = official_yarn_on_inv_freq(
        substrate_inv_freq,
        head_dim=head_dim,
        base=base,
        scale=factor,
        original_max_position_embeddings=original_max_position_embeddings,
        beta_fast=32.0,
        beta_slow=1.0,
        extrapolation_factor=1.0,
        attn_factor=1.0,
    )
    modules = find_rotary_modules(model)
    if not modules:
        raise RuntimeError("no rotary modules were found for official YaRN injection")
    changed = []
    for name, module in modules:
        if module.inv_freq.numel() != transformed.numel():
            raise RuntimeError(
                f"rotary frequency shape mismatch at {name}: "
                f"{module.inv_freq.numel()} != {transformed.numel()}"
            )
        target = transformed.to(device=module.inv_freq.device, dtype=module.inv_freq.dtype)
        with torch.no_grad():
            module.inv_freq.copy_(target)
            original = getattr(module, "original_inv_freq", None)
            if torch.is_tensor(original) and original.shape == target.shape:
                original.copy_(target.to(device=original.device, dtype=original.dtype))
        if not hasattr(module, "attention_scaling"):
            raise RuntimeError(f"rotary module {name} has no attention_scaling field")
        module.attention_scaling = float(mscale)
        for attr in (
            "_cos_cached",
            "_sin_cached",
            "cos_cached",
            "sin_cached",
            "_cos_cache",
            "_sin_cache",
            "max_seq_len_cached",
        ):
            if not hasattr(module, attr):
                continue
            current = getattr(module, attr)
            setattr(module, attr, 0 if isinstance(current, (int, float)) else None)
        changed.append(name)
    return {
        "patched_count": len(changed),
        "changed_modules": changed,
        "factor": float(factor),
        "mscale": float(mscale),
        "operator": operator,
        "input_inv_freq_sha256": _tensor_sha256(substrate_inv_freq),
        "output_inv_freq_sha256": _tensor_sha256(transformed),
        "inv_freq": transformed,
    }


def load_capability_suite(data_root: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = Path(data_root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("capability manifest schema mismatch")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("capability manifest has no files")
    rows: list[dict[str, Any]] = []
    for filename in sorted(files):
        if Path(filename).name != filename or not filename.endswith(".jsonl"):
            raise ValueError(f"unsafe capability filename: {filename!r}")
        record = files[filename]
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path) != record.get("sha256"):
            raise ValueError(f"capability file hash mismatch: {filename}")
        if path.stat().st_size != int(record.get("size_bytes", -1)):
            raise ValueError(f"capability file size mismatch: {filename}")
        file_rows = load_jsonl(path)
        if len(file_rows) != int(record.get("row_count", -1)):
            raise ValueError(f"capability file row-count mismatch: {filename}")
        rows.extend(file_rows)
    rows = validate_records(rows)
    if len(rows) != int(manifest.get("row_count", -1)):
        raise ValueError("capability manifest total row-count mismatch")
    return manifest, rows


def select_rows(rows: Sequence[dict[str, Any]], *, mode: str) -> list[dict[str, Any]]:
    if mode == "full":
        return list(rows)
    if mode != "pilot":
        raise ValueError("mode must be pilot or full")
    selected: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        cell = (row["task"], int(row["target_length"]), row.get("depth_percent"))
        if cell in seen:
            continue
        seen.add(cell)
        selected.append(row)
    return selected


def _length_label(length: int) -> str:
    return f"{length // 1024}K" if length % 1024 == 0 else str(length)


def _depth_label(depth: Any) -> str:
    if depth is None:
        return "all_depths"
    value = float(depth)
    return f"depth_{int(value) if value.is_integer() else value:g}"


def summarize_results(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[float, str, int, Any], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                float(row["factor"]),
                str(row["task"]),
                int(row["target_length"]),
                row.get("depth_percent"),
            )
        ].append(row)
    output: dict[str, Any] = {}
    for (factor, task, length, depth), cell_rows in sorted(grouped.items(), key=str):
        tokens = sum(int(row["answer_tokens"]) for row in cell_rows)
        nll_sum = sum(float(row["nll_sum"]) for row in cell_rows)
        scores = [float(row["metric_score"]) for row in cell_rows if row.get("metric_score") is not None]
        record = {
            "examples": len(cell_rows),
            "nll": nll_sum / tokens,
            "metric_mean": (sum(scores) / len(scores) if scores else None),
            "answer_tokens": tokens,
        }
        output.setdefault(f"x{factor:g}", {}).setdefault(task, {}).setdefault(
            _length_label(length), {}
        )[_depth_label(depth)] = record
    return output


def _answer_token_ids(tokenizer: Any, answer: str) -> list[int]:
    ids = tokenizer(answer, add_special_tokens=False, return_attention_mask=False)["input_ids"]
    if not ids:
        raise ValueError("answer tokenization produced no tokens")
    return [int(token_id) for token_id in ids]


@torch.no_grad()
def score_answer(
    backbone: torch.nn.Module,
    lm_head: torch.nn.Module,
    *,
    prompt_ids: Sequence[int],
    answer_ids: Sequence[int],
    device: torch.device,
) -> dict[str, float | int]:
    if not prompt_ids or not answer_ids:
        raise ValueError("answer scoring requires prompt and answer tokens")
    full_ids = [int(token_id) for token_id in prompt_ids] + [int(token_id) for token_id in answer_ids]
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    hidden = backbone(input_ids=input_ids, use_cache=False, return_dict=True).last_hidden_state
    answer_start = len(prompt_ids)
    predicting = hidden[:, answer_start - 1 : len(full_ids) - 1]
    labels = input_ids[:, answer_start:]
    logits = lm_head(predicting).float()
    nll_sum = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        reduction="sum",
    )
    token_count = int(labels.numel())
    value = float(nll_sum.detach().double().cpu())
    del input_ids, hidden, predicting, labels, logits, nll_sum
    return {
        "nll_sum": value,
        "answer_tokens": token_count,
        "mean_logprob": -value / token_count,
    }


def _normalize_answer(value: str) -> str:
    value = value.lower()
    value = value.translate(str.maketrans("", "", string.punctuation))
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def _qa_f1(prediction: str, reference: str) -> float:
    predicted = _normalize_answer(prediction).split()
    expected = _normalize_answer(reference).split()
    if not predicted or not expected:
        return float(predicted == expected)
    expected_counts: dict[str, int] = defaultdict(int)
    for token in expected:
        expected_counts[token] += 1
    overlap = 0
    for token in predicted:
        if expected_counts[token] > 0:
            overlap += 1
            expected_counts[token] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2.0 * precision * recall / (precision + recall)


@torch.no_grad()
def generate_answer_details(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    prompt_ids: Sequence[int],
    metric: str,
    device: torch.device,
) -> dict[str, Any]:
    input_ids = torch.tensor([[int(token_id) for token_id in prompt_ids]], dtype=torch.long, device=device)
    max_new_tokens = 32 if metric == "exact_match" else 96
    output = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = [int(token_id) for token_id in output[0, input_ids.shape[1] :].tolist()]
    eos_token_id = tokenizer.eos_token_id
    eos_index = (
        generated.index(int(eos_token_id))
        if eos_token_id is not None and int(eos_token_id) in generated
        else None
    )
    content = generated if eos_index is None else generated[:eos_index]
    prediction = tokenizer.decode(content, skip_special_tokens=True)
    return {
        "prediction": prediction.strip(),
        "generated_ids": generated,
        "generated_token_count": len(content),
        "eos_terminated": eos_index is not None,
    }


def generate_answer(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    prompt_ids: Sequence[int],
    metric: str,
    device: torch.device,
) -> str:
    """Backward-compatible prediction-only wrapper for existing entrypoints."""
    return str(
        generate_answer_details(
            model,
            tokenizer,
            prompt_ids=prompt_ids,
            metric=metric,
            device=device,
        )["prediction"]
    )


def score_generation_metrics(
    prediction: str,
    answers: Sequence[str],
    *,
    eos_terminated: bool,
    generated_token_count: int,
) -> dict[str, Any]:
    """Keep retrieval evidence distinct from formatting and stopping."""
    if not answers:
        raise ValueError("generation metrics require at least one reference answer")
    normalized_prediction = _normalize_answer(prediction)
    normalized_answers = [_normalize_answer(answer) for answer in answers]
    strict = normalized_prediction in normalized_answers
    containment = any(
        answer and answer in normalized_prediction for answer in normalized_answers
    )
    numeric_answers = {
        re.sub(r"\s+", "", str(answer))
        for answer in answers
        if re.fullmatch(r"\s*\d+\s*", str(answer))
    }
    if numeric_answers:
        first_number = re.search(r"(?<!\d)(\d+)(?!\d)", str(prediction))
        first_exact = first_number is not None and first_number.group(1) in numeric_answers
    else:
        first_exact = any(
            normalized_prediction == answer
            or normalized_prediction.startswith(f"{answer} ")
            for answer in normalized_answers
            if answer
        )
    return {
        "strict_exact": strict,
        "first_value_exact": first_exact,
        "gold_containment": containment,
        "eos_terminated": bool(eos_terminated),
        "generated_token_count": int(generated_token_count),
    }


def _score_metric(metric: str, prediction: str, answers: Sequence[str]) -> float:
    if metric == "exact_match":
        normalized = _normalize_answer(prediction)
        return float(any(normalized == _normalize_answer(answer) for answer in answers))
    if metric == "qa_f1":
        return max(_qa_f1(prediction, answer) for answer in answers)
    raise ValueError(f"generation metric is unsupported: {metric}")


def _score_one_record(
    *,
    row: Mapping[str, Any],
    model: torch.nn.Module,
    tokenizer: Any,
    backbone: torch.nn.Module,
    lm_head: torch.nn.Module,
    device: torch.device,
    generation: bool,
) -> dict[str, Any]:
    prompt_ids = [int(token_id) for token_id in row["prompt_ids"]]
    base = {
        "example_id": row["example_id"],
        "suite": row["suite"],
        "task": row["task"],
        "target_length": int(row["target_length"]),
        "depth_percent": row.get("depth_percent"),
        "prompt_sha256": row["prompt_sha256"],
        "metric": row["metric"],
    }
    if row["metric"] == "mcqa":
        choices = list(row["choices"])
        choice_scores = [
            score_answer(
                backbone,
                lm_head,
                prompt_ids=prompt_ids,
                answer_ids=_answer_token_ids(tokenizer, choice),
                device=device,
            )
            for choice in choices
        ]
        predicted = max(range(len(choice_scores)), key=lambda index: choice_scores[index]["mean_logprob"])
        gold = int(row["answer_index"])
        wrong_best = max(
            choice_scores[index]["mean_logprob"]
            for index in range(len(choice_scores))
            if index != gold
        )
        gold_score = choice_scores[gold]
        return {
            **base,
            "nll_sum": gold_score["nll_sum"],
            "answer_tokens": gold_score["answer_tokens"],
            "metric_score": float(predicted == gold),
            "prediction_index": predicted,
            "answer_index": gold,
            "correct_minus_best_wrong_mean_logprob": gold_score["mean_logprob"] - wrong_best,
            "choice_mean_logprobs": [score["mean_logprob"] for score in choice_scores],
        }

    answer_scores = []
    for answer in row["answers"]:
        score = score_answer(
            backbone,
            lm_head,
            prompt_ids=prompt_ids,
            answer_ids=_answer_token_ids(tokenizer, answer),
            device=device,
        )
        answer_scores.append(score)
    selected_index = max(range(len(answer_scores)), key=lambda index: answer_scores[index]["mean_logprob"])
    selected = answer_scores[selected_index]
    prediction = None
    generation_details = None
    metric_score = None
    if generation:
        generation_details = generate_answer_details(
            model,
            tokenizer,
            prompt_ids=prompt_ids,
            metric=row["metric"],
            device=device,
        )
        prediction = str(generation_details["prediction"])
        metric_score = _score_metric(row["metric"], prediction, row["answers"])
        generation_details.update(
            score_generation_metrics(
                prediction,
                row["answers"],
                eos_terminated=bool(generation_details["eos_terminated"]),
                generated_token_count=int(generation_details["generated_token_count"]),
            )
        )
    return {
        **base,
        "nll_sum": selected["nll_sum"],
        "answer_tokens": selected["answer_tokens"],
        "metric_score": metric_score,
        "prediction": prediction,
        "generation": generation_details,
        "selected_reference_index": selected_index,
        "reference_mean_logprobs": [score["mean_logprob"] for score in answer_scores],
    }


def _json_runtime_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != "inv_freq"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_manifest", type=Path, required=True)
    parser.add_argument("--adapter_dir", type=Path, required=True)
    parser.add_argument("--training_data_manifest", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--substrate", choices=("native_geo", "evq_cosh"), required=True)
    parser.add_argument("--yarn_factors", default="2,4")
    parser.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    factors = parse_yarn_factors(args.yarn_factors)
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest, all_rows = load_capability_suite(args.data_root)
    expected_tokenizer = tokenizer_source_fingerprint(args.model_name)
    recorded_tokenizer = manifest.get("tokenizer", {})
    if (
        recorded_tokenizer.get("identifier") != expected_tokenizer.get("identifier")
        or recorded_tokenizer.get("files") != expected_tokenizer.get("files")
    ):
        raise ValueError("capability suite tokenizer differs from the model tokenizer")
    rows = select_rows(all_rows, mode=args.mode)
    training_manifest_sha256 = sha256_file(args.training_data_manifest)
    model_manifest = json.loads(args.model_manifest.read_text(encoding="utf-8"))
    validate_model_manifest(Path(args.model_name), model_manifest, verify_hashes=False)
    model_manifest_sha256 = sha256_file(args.model_manifest)
    adapter_metadata = validate_adapter_identity(
        args.adapter_dir,
        substrate=args.substrate,
        training_manifest_sha256=training_manifest_sha256,
    )
    if adapter_metadata.get("model_manifest_sha256") != model_manifest_sha256:
        raise ValueError("adapter and evaluator model manifests differ")

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    local_model = Path(args.model_name).is_dir()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=local_model,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        local_files_only=local_model,
    )
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.to(torch.device("cuda"))
    model.eval()
    model.config.use_cache = True
    geometry = resolve_model_rope_geometry(model.config)
    substrate_inv_freq, frequency_record, frequency_provenance = load_frequency_artifact(
        args.adapter_dir / "custom_inv_freq.pt",
        expected_method=args.substrate,
    )
    canonical, _ = build_training_inv_freq(
        rope_method=args.substrate,
        head_dim=geometry.head_dim,
        base=geometry.rope_base,
        tau=1.414,
    )
    if not torch.allclose(
        substrate_inv_freq.to(torch.float64),
        canonical.to(torch.float64),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("adapter frequency artifact differs from the canonical substrate")

    device = torch.device("cuda")
    backbone = causal_backbone(model)
    lm_head = model.get_output_embeddings()
    raw_results: list[dict[str, Any]] = []
    operator_records: dict[str, Any] = {}
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    for factor in factors:
        runtime = apply_official_yarn_runtime(
            model,
            substrate_inv_freq,
            head_dim=geometry.head_dim,
            base=geometry.rope_base,
            factor=factor,
            original_max_position_embeddings=8192,
        )
        verify_model_inv_freq(model, runtime["inv_freq"])
        operator_records[f"x{factor:g}"] = _json_runtime_record(runtime)
        for index, row in enumerate(rows, start=1):
            scored = _score_one_record(
                row=row,
                model=model,
                tokenizer=tokenizer,
                backbone=backbone,
                lm_head=lm_head,
                device=device,
                generation=True,
            )
            scored["factor"] = factor
            raw_results.append(scored)
            print(
                json.dumps(
                    {
                        "factor": factor,
                        "progress": f"{index}/{len(rows)}",
                        "task": row["task"],
                        "target_length": row["target_length"],
                        "metric_score": scored["metric_score"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    output = {
        "schema": "evq_cosh.seed42_capability_official_yarn_eval.v1",
        "arm_contract": capability_arm_contract(args.substrate, factors),
        "mode": args.mode,
        "model": Path(args.model_name).name,
        "adapter_dir_name": args.adapter_dir.name,
        "adapter_sha256": adapter_metadata["adapter_sha256"],
        "training_data_manifest_sha256": training_manifest_sha256,
        "model_manifest_sha256": model_manifest_sha256,
        "capability_manifest_sha256": sha256_file(args.data_root / "manifest.json"),
        "capability_manifest": {
            "row_count": manifest["row_count"],
            "task_counts": manifest.get("task_counts", {}),
        },
        "frequency_artifact": {
            "metadata": {key: value for key, value in frequency_record.items() if key != "inv_freq"},
            "provenance": frequency_provenance,
        },
        "official_yarn": operator_records,
        "results": raw_results,
        "summary": summarize_results(raw_results),
        "runtime": {
            "seconds": time.time() - started,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
            "torch_version": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(0),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".incomplete")
    temporary.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "runtime": output["runtime"]}, indent=2))


if __name__ == "__main__":
    main()
