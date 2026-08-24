"""Held-out downstream evaluation for the phase AdaRoPE OLMo-2 protocol.

The natural final split is a prerequisite, not an evaluation input.  This
module only consumes released artifacts and prepared held-out data; it never
trains, chooses a method, or changes the RULER/LongBench rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .receipts import require_dual_authorization, sha256_file


PHASE_TARGET_KEY = "phase_chord_olmo_r0_lambda_0p1"
EXP_NEGATIVE_TARGET_KEY = "matched_exponential_control"
MOMENT_MATCHED_TARGET_KEY = "moment_matched_same_sign_control"

ARMS = (
    "lora_only_null",
    "native_scale",
    "context_stretch_exp_negative",
    "phase_chord",
    "moment_matched_same_sign_control",
)
ALLOWED_LENGTHS = (4_096, 8_192, 16_384)
EXPECTED_TARGET_MODULES = {"q_proj", "k_proj", "v_proj", "o_proj"}
OFFICIAL_2WIKI_MEMBER = "2wikimqa.jsonl"
OFFICIAL_2WIKI_PROMPT = (
    "Answer the question based on the given passages. Only give me the "
    "answer and do not output any other words.\n\n"
    "The following are given passages.\n{context}\n\n"
    "Answer the question based on the given passages. Only give me the "
    "answer and do not output any other words.\n\n"
    "Question: {question}\nAnswer:"
)
ARM_CONTRACT = {
    "lora_only_null": {
        "target_key": PHASE_TARGET_KEY,
        "target_name": "phase_chord",
        "receipt_arm": "lora_only_null",
        "modes": {"control"},
    },
    "native_scale": {
        "target_key": PHASE_TARGET_KEY,
        "target_name": "phase_chord",
        "receipt_arm": "native_scale",
        "modes": {"scale_only"},
    },
    "context_stretch_exp_negative": {
        "target_key": EXP_NEGATIVE_TARGET_KEY,
        "target_name": "matched_exponential",
        "receipt_arm": "context_stretch_exp_negative",
        "modes": {"joint"},
    },
    "phase_chord": {
        "target_key": PHASE_TARGET_KEY,
        "target_name": "phase_chord",
        "receipt_arm": "phase_chord",
        "modes": {"joint"},
    },
    "moment_matched_same_sign_control": {
        "target_key": MOMENT_MATCHED_TARGET_KEY,
        "target_name": "moment_matched_control",
        "receipt_arm": "moment_matched_same_sign_control",
        "modes": {"joint"},
    },
}
EXPECTED_CHECKPOINT_SHA256 = (
    "36d044c73655bb904f822915e6294ba3dae8e6e1af5e703e9d452f2d6a3a294f"
)
STATUS = "OLMO2_PHASE_ADAROPE_DOWNSTREAM_EVAL_COMPLETE_V1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=path.name + ".", mode="w", encoding="utf-8",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = path.read_bytes() if path.exists() else b""
    line = (
        json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=path.name + ".", mode="wb", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(previous)
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def strict_eos_boundary(
    token_ids: Sequence[int], eos_token_id: int,
) -> dict[str, Any]:
    values = [int(value) for value in token_ids]
    indices = [
        index for index, value in enumerate(values)
        if value == int(eos_token_id)
    ]
    first = indices[0] if indices else None
    return {
        "eos_token_id": int(eos_token_id),
        "eos_observed": first is not None,
        "first_eos_index": first,
        "eos_count": len(indices),
        "tokens_after_first_eos": (
            None if first is None else len(values) - first - 1
        ),
        "strict_terminal_eos_boundary": bool(
            first is not None
            and first == len(values) - 1
            and len(indices) == 1
        ),
    }


def tensor_float32_sha256(values: Any) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(values, dtype="<f4")).tobytes()
    ).hexdigest()


def tensor_state_sha256(values: Any) -> str:
    import torch

    value = torch.as_tensor(values).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def tree_hashes(root: Path) -> dict[str, dict[str, Any]]:
    root = root.resolve()
    return {
        str(path.relative_to(root)): {
            "sha256": sha256_file(path),
            "bytes": int(path.stat().st_size),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.name.endswith(".incomplete")
    }


def _target_records(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("assets", "frequencies", "candidates"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            return value
    return payload


def _load_target_record(
    manifest_path: Path, key: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    payload = _json(manifest_path)
    record = _target_records(payload).get(key)
    if not isinstance(record, Mapping):
        raise ValueError(f"target manifest lacks target key: {key}")
    if record.get("inv_freq") is not None:
        values = np.asarray(record["inv_freq"], dtype="<f4")
    else:
        relative = record.get("path")
        expected = record.get("sha256")
        if not relative or not expected:
            raise ValueError(f"target {key} lacks hash-bound asset")
        asset = (manifest_path.parent / str(relative)).resolve()
        if sha256_file(asset) != str(expected):
            raise ValueError(f"target asset hash drift: {key}")
        if asset.suffix == ".npy":
            values = np.asarray(np.load(asset, allow_pickle=False), dtype="<f4")
        else:
            import torch

            loaded = torch.load(asset, map_location="cpu", weights_only=True)
            if isinstance(loaded, Mapping):
                loaded = loaded.get("inv_freq", loaded.get("target_inv_freq"))
            values = np.asarray(loaded, dtype="<f4")
    if values.shape != (64,) or not np.isfinite(values).all():
        raise ValueError(f"target {key} must be finite with shape [64]")
    if not (values > 0).all() or not np.all(values[:-1] > values[1:]):
        raise ValueError(f"target {key} must be positive and decreasing")
    declared = record.get("inv_freq_float32_sha256")
    actual = tensor_float32_sha256(values)
    if declared is not None and str(declared) != actual:
        raise ValueError(f"target inline hash drift: {key}")
    return values, {
        "target_key": key,
        "target_float32_sha256": actual,
        "record": dict(record),
    }


def validate_target_manifest(path: Path, arm: str) -> dict[str, Any]:
    if arm not in ARM_CONTRACT:
        raise ValueError(f"unknown AdaRoPE arm: {arm}")
    key = str(ARM_CONTRACT[arm]["target_key"])
    values, record = _load_target_record(path.resolve(), key)
    return {
        "path": str(path.resolve()),
        "manifest_sha256": sha256_file(path),
        "target_key": key,
        "target_float32_sha256": record["target_float32_sha256"],
        "target_state_sha256": tensor_state_sha256(values),
        "values": values,
    }


def checkpoint_receipt(checkpoint: Path, ready_path: Path) -> dict[str, Any]:
    ready = _json(ready_path)
    if ready.get("status") not in {
        "VERIFIED_READY",
        "OLMO2_INSTRUCT_4K_CONVERSION_READY",
        "OLMO2_INSTRUCT_PRO6000_RULER_MATRIX_READY",
    }:
        raise RuntimeError("checkpoint READY status drift")
    weight = checkpoint.resolve() / "model.safetensors"
    config_path = checkpoint.resolve() / "config.json"
    if not weight.is_file() or not config_path.is_file():
        raise FileNotFoundError("released checkpoint weights/config are required")
    weight_sha = sha256_file(weight)
    if weight_sha != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("released checkpoint SHA drift")
    if ready.get("status") == "VERIFIED_READY":
        recorded = ready.get("canonical_identity", {}).get(
            "historical_model_sha256"
        )
        rows = [
            row
            for row in ready.get("repository_files", {}).get("files", [])
            if Path(str(row.get("path", ""))).name == "model.safetensors"
        ]
        if len(rows) != 1 or rows[0].get("sha256") != weight_sha:
            raise RuntimeError("READY repository weight digest drift")
        recorded_path = ready.get("model_dir")
    else:
        checkpoint_record = ready.get("checkpoint", {})
        recorded = checkpoint_record.get("composite_sha256")
        recorded_path = checkpoint_record.get("checkpoint_path")
    if recorded != weight_sha:
        raise RuntimeError("checkpoint READY canonical digest drift")
    if (
        recorded_path is not None
        and Path(str(recorded_path)).resolve() != checkpoint.resolve()
    ):
        raise RuntimeError("checkpoint READY path drift")
    config = _json(config_path)
    expected = {
        "model_type": "olmo2",
        "hidden_size": 2_048,
        "num_hidden_layers": 16,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "vocab_size": 100_352,
    }
    if any(config.get(name) != value for name, value in expected.items()):
        raise RuntimeError("released OLMo configuration drift")
    return {
        "path": str(checkpoint.resolve()),
        "weight_sha256": weight_sha,
        "ready_receipt_sha256": sha256_file(ready_path),
    }


def _adapter_tree_receipt(adapter: Path) -> dict[str, str]:
    return {
        name: row["sha256"]
        for name, row in tree_hashes(adapter).items()
    }


def _read_sidecar_metadata(path: Path) -> dict[str, Any]:
    import torch
    from . import phase_adarope

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, Mapping)
        or payload.get("method_id") != phase_adarope.METHOD_ID
        or int(payload.get("state_version", -1)) != int(phase_adarope.STATE_VERSION)
        or not isinstance(payload.get("metadata"), Mapping)
    ):
        raise RuntimeError("phase sidecar identity drift")
    state = payload.get("state")
    if not isinstance(state, Mapping):
        raise RuntimeError("phase sidecar state is malformed")
    state_keys = set(state)
    required = {"alpha", "raw_gamma"}
    if int(phase_adarope.STATE_VERSION) >= 2:
        required.add("raw_beta")
        valid_keys = state_keys == required
    else:
        temperature_keys = {
            key for key in state_keys
            if key.startswith("raw_") and key != "raw_gamma"
        }
        valid_keys = (
            state_keys == (required | temperature_keys)
            and len(temperature_keys) == 1
        )
    if not valid_keys:
        raise RuntimeError(
            f"phase sidecar state keys drift: {sorted(state_keys)}"
        )
    return dict(payload["metadata"])


def validate_bundle_contract(
    bundle: Path,
    *,
    arm: str,
    checkpoint_sha256: str,
    target: Mapping[str, Any],
) -> dict[str, Any]:
    contract = ARM_CONTRACT[arm]
    bundle = bundle.resolve()
    receipt_path = bundle / "receipt.json"
    adapter = bundle / "adapter"
    sidecar = bundle / "phase_adarope_state.pt"
    if not receipt_path.is_file() or not adapter.is_dir() or not sidecar.is_file():
        raise RuntimeError("every AdaRoPE bundle requires receipt, adapter, and sidecar")
    receipt = _json(receipt_path)
    if receipt.get("status") != "COMPLETE":
        raise RuntimeError("bundle receipt is not complete")
    if receipt.get("stage") != "stage2":
        raise RuntimeError("downstream evaluation requires a Stage2 bundle")
    if receipt.get("arm") != contract["receipt_arm"]:
        raise RuntimeError("bundle receipt arm drift")
    target_path = receipt.get("target_manifest")
    if target_path is not None and Path(str(target_path)).resolve() != Path(
        str(target["path"])
    ).resolve():
        raise RuntimeError("bundle target manifest path drift")
    if receipt.get("target_key") != target["target_key"]:
        raise RuntimeError("bundle target key drift")
    recorded_target_sha = receipt.get("target_manifest_sha256")
    if recorded_target_sha is not None and recorded_target_sha != target["manifest_sha256"]:
        raise RuntimeError("bundle target manifest hash drift")
    recorded_checkpoint_sha = receipt.get("checkpoint_sha256")
    if recorded_checkpoint_sha is not None and recorded_checkpoint_sha != checkpoint_sha256:
        raise RuntimeError("bundle checkpoint hash drift")
    bundle_info = receipt.get("bundle")
    if not isinstance(bundle_info, Mapping):
        raise RuntimeError("bundle receipt lacks bundle record")
    actual_sidecar_sha = sha256_file(sidecar)
    if bundle_info.get("phase_sidecar_sha256") != actual_sidecar_sha:
        raise RuntimeError("bundle sidecar hash drift")
    expected_adapter = _adapter_tree_receipt(adapter)
    if bundle_info.get("adapter_files") != expected_adapter:
        raise RuntimeError("bundle adapter tree drift")
    config = _json(adapter / "adapter_config.json")
    modules = config.get("target_modules")
    if isinstance(modules, str):
        modules = [modules]
    if (
        int(config.get("r", -1)) != 64
        or float(config.get("lora_alpha", -1)) != 128.0
        or float(config.get("lora_dropout", -1)) != 0.0
        or str(config.get("bias")) != "none"
        or set(str(value) for value in modules or []) != EXPECTED_TARGET_MODULES
    ):
        raise RuntimeError("standard PEFT QKVO rank64/alpha128 drift")
    metadata = _read_sidecar_metadata(sidecar)
    if metadata.get("target_name") != contract["target_name"]:
        raise RuntimeError("sidecar target_name drift")
    if metadata.get("mode") not in contract["modes"]:
        raise RuntimeError("sidecar mode drift")
    if arm in {"lora_only_null", "native_scale"}:
        import torch

        payload = torch.load(sidecar, map_location="cpu", weights_only=True)
        alpha = payload.get("state", {}).get("alpha") if isinstance(payload, Mapping) else None
        if alpha is None or not bool(torch.equal(torch.as_tensor(alpha), torch.zeros_like(torch.as_tensor(alpha)))):
            raise RuntimeError("control/scale sidecar must keep alpha at zero")
    if metadata.get("target_inv_freq_sha256") != target["target_state_sha256"]:
        raise RuntimeError("sidecar target frequency drift")
    return {
        "checkpoint_sha256": checkpoint_sha256,
        "target_manifest_sha256": target["manifest_sha256"],
        "target_key": target["target_key"],
        "target_float32_sha256": target["target_float32_sha256"],
        "sidecar_sha256": actual_sidecar_sha,
        "sidecar_metadata": metadata,
        "adapter_tree": expected_adapter,
        "receipt_sha256": sha256_file(receipt_path),
    }


def protocol_code_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    return {
        "trainer": sha256_file(package / "train_phase_adarope.py"),
        "data": sha256_file(package / "prepare_identifiable_data.py"),
        "attention": sha256_file(package / "phase_adarope.py"),
        "receipts": sha256_file(package / "receipts.py"),
    }


def downstream_code_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    maturity = package.parent / "olmo2_lora_maturity"
    return {
        "evaluator": sha256_file(Path(__file__).resolve()),
        "official_ruler_scorer": sha256_file(
            maturity / "evaluate_instruct_ruler_transfer.py"
        ),
        "ruler_preparer": sha256_file(
            maturity / "prepare_instruct_ruler_transfer.py"
        ),
    }


def validate_natural_final_gate(
    path: Path,
    *,
    natural_data_root: Path,
    parent_receipt: Path,
    target: Mapping[str, Any],
    bundle_receipt: Path,
) -> dict[str, Any]:
    gate = _json(path)
    if gate.get("status") != "COMPLETE":
        raise RuntimeError("natural gate must be the trainer COMPLETE output")
    data_manifest = natural_data_root.resolve() / "manifest.json"
    if not data_manifest.is_file():
        raise FileNotFoundError(data_manifest)
    if gate.get("data_manifest_sha256") != sha256_file(data_manifest):
        raise RuntimeError("natural final split manifest hash drift")
    manifest = _json(data_manifest)
    if manifest.get("split") != "final_validation":
        raise RuntimeError("natural gate is not on final_validation split")
    if manifest.get("method_selection_allowed") is not False:
        raise RuntimeError("natural final_validation split is method-selectable")
    if int(manifest.get("length", 0)) not in ALLOWED_LENGTHS:
        raise RuntimeError("natural final evidence length is unsupported")
    if int(gate.get("examples", 0)) <= 0:
        raise RuntimeError("natural gate has no examples")
    parent = _json(parent_receipt)
    if parent_receipt.resolve() != bundle_receipt.resolve():
        raise RuntimeError("natural parent receipt is not the evaluated bundle receipt")
    if gate.get("parent_receipt_sha256") != sha256_file(parent_receipt):
        raise RuntimeError("natural parent receipt hash drift")
    if gate.get("parent_receipt_sha256") != sha256_file(bundle_receipt):
        raise RuntimeError("natural gate does not bind the evaluated bundle")
    if parent.get("target_key") != target["target_key"]:
        raise RuntimeError("natural parent target key drift")
    if parent.get("target_manifest_sha256") not in {
        None, target["manifest_sha256"]
    }:
        raise RuntimeError("natural parent target manifest hash drift")
    parent_target = parent.get("target_manifest")
    if parent_target is not None and Path(str(parent_target)).resolve() != Path(
        str(target["path"])
    ).resolve():
        raise RuntimeError("natural parent target manifest drift")
    return {
        "receipt_sha256": sha256_file(path),
        "data_manifest_sha256": sha256_file(data_manifest),
        "parent_receipt_sha256": sha256_file(parent_receipt),
        "protocol_code_sha256": protocol_code_hashes(),
        "length": int(manifest.get("length", 0)),
        "metrics": {
            key: gate.get(key)
            for key in (
                "exact_answer_terminal_eos",
                "terminal_eos_rate",
                "source_follow_mean_logprob_delta",
                "source_follow_positive_fraction",
                "retention_4k",
            )
            if key in gate
        },
    }


def validate_natural_final_gates(
    gates: Sequence[Path],
    data_roots: Sequence[Path],
    *,
    parent_receipt: Path,
    target: Mapping[str, Any],
    bundle_receipt: Path,
) -> dict[str, Any]:
    if len(gates) != 3 or len(data_roots) != 3:
        raise ValueError("exactly three natural gates/data roots are required")
    validated = [
        validate_natural_final_gate(
            gate,
            natural_data_root=root,
            parent_receipt=parent_receipt,
            target=target,
            bundle_receipt=bundle_receipt,
        )
        for gate, root in zip(gates, data_roots)
    ]
    lengths = [int(item["length"]) for item in validated]
    if sorted(lengths) != [4_096, 8_192, 16_384]:
        raise RuntimeError(
            f"natural evidence must cover exactly 4K/8K/16K, got {lengths}"
        )
    return {
        "evidence_by_length": {
            str(item["length"]): item for item in validated
        },
        "lengths": sorted(lengths),
        "threshold_selection": False,
    }


def load_resumable_rows(
    path: Path,
    expected_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    expected = {str(row["row_id"]): row for row in expected_rows}
    result: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            raise RuntimeError(f"blank resumable row at line {line_number}")
        row = json.loads(line)
        row_id = str(row.get("row_id", ""))
        if not row_id or row_id in result:
            raise RuntimeError(f"duplicate/missing resumable row id: {row_id}")
        if row_id not in expected:
            raise RuntimeError(f"resumable row is outside current run: {row_id}")
        if row.get("row_sha256") != expected[row_id]["row_sha256"]:
            raise RuntimeError(f"resumable row identity drift: {row_id}")
        result[row_id] = row
    return result


def generation_cache_key(job: Mapping[str, Any]) -> str:
    """Identity of deterministic prefill/decode including phase budget."""
    return canonical_sha256({
        "input_ids": [int(value) for value in job["input_ids"]],
        "max_new_tokens": int(job["max_new_tokens"]),
        "references": [str(value) for value in job["references"]],
        "metric": str(job["metric"]),
        "phase_context_budget": int(job["nominal_length"]),
    })


def normalize_2wiki(text: str) -> str:
    import re
    import string

    value = re.sub(r"\b(a|an|the)\b", " ", str(text).lower())
    value = "".join(
        character for character in value
        if character not in string.punctuation
    )
    return " ".join(value.split())


def token_f1(prediction: str, references: Sequence[str]) -> float:
    pred = normalize_2wiki(prediction).split()
    best = 0.0
    for reference in references:
        gold = normalize_2wiki(reference).split()
        if not pred or not gold:
            continue
        overlap = sum(
            min(pred.count(token), gold.count(token))
            for token in set(pred)
        )
        if overlap:
            precision = overlap / len(pred)
            recall = overlap / len(gold)
            best = max(
                best,
                2 * precision * recall / (precision + recall),
            )
    return float(best)


def normalized_exact(prediction: str, references: Sequence[str]) -> float:
    value = normalize_2wiki(prediction)
    return float(any(value == normalize_2wiki(ref) for ref in references))


def load_official_2wiki_rows(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path.resolve()) as archive:
        candidates = [
            name for name in archive.namelist()
            if name == OFFICIAL_2WIKI_MEMBER
            or name.endswith("/" + OFFICIAL_2WIKI_MEMBER)
        ]
        if len(candidates) != 1:
            raise RuntimeError(f"expected one official 2Wiki member, got {candidates}")
        member = candidates[0]
        rows = [
            json.loads(line)
            for line in archive.read(member).decode("utf-8").splitlines()
            if line.strip()
        ]
    if len(rows) != 200:
        raise RuntimeError(f"LongBench 2wikimqa must contain 200 rows, got {len(rows)}")
    for index, row in enumerate(rows):
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("input"), str)
            or not isinstance(row.get("answers"), list)
        ):
            raise RuntimeError(f"malformed official 2Wiki row {index}")
    return {
        "dataset": "THUDM/LongBench:2wikimqa",
        "zip_sha256": sha256_file(path),
        "member": member,
        "rows": rows,
        "rows_sha256": canonical_sha256(rows),
    }


def _chat_ids(tokenizer: Any, prompt: str) -> list[int]:
    value = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if isinstance(value, Mapping):
        value = value.get("input_ids")
        if value is None:
            raise RuntimeError("chat template output lacks input_ids")
    if getattr(value, "ndim", None) == 2:
        return [int(item) for item in value[0].tolist()]
    return [int(item) for item in value]


def _fit_chat_prompt(
    tokenizer: Any, prompt: str, length: int, max_new_tokens: int,
) -> tuple[list[int], bool]:
    raw = list(tokenizer(prompt, add_special_tokens=False).input_ids)
    ids = _chat_ids(tokenizer, prompt)
    if len(ids) + max_new_tokens <= length:
        return ids, False
    candidate = raw
    while True:
        content = tokenizer.decode(
            candidate,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        ids = _chat_ids(tokenizer, content)
        if len(ids) + max_new_tokens <= length:
            return ids, True
        allowed = len(candidate) - (len(ids) + max_new_tokens - length)
        if allowed <= 64:
            raise RuntimeError(f"2Wiki query cannot fit physical L{length}")
        head = allowed // 2
        candidate = raw[:head] + raw[-(allowed - head):]


def build_2wiki_jobs(
    tokenizer: Any,
    package: Mapping[str, Any],
    *,
    lengths: Sequence[int] = ALLOWED_LENGTHS,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for index, source in enumerate(package["rows"]):
        prompt = OFFICIAL_2WIKI_PROMPT.format(
            context=str(source["context"]),
            question=str(source["input"]),
        )
        source_hash = canonical_sha256(source)
        for length in lengths:
            ids, truncated = _fit_chat_prompt(
                tokenizer, prompt, int(length), max_new_tokens=32,
            )
            row_id = f"2wikimqa:{source_hash}:L{int(length)}"
            jobs.append({
                "row_id": row_id,
                "family": "2wikimqa",
                "task": "2wikimqa",
                "nominal_length": int(length),
                "input_ids": ids,
                "references": [str(value) for value in source["answers"]],
                "metric": "longbench_token_f1_exact",
                "max_new_tokens": 32,
                "truncated": truncated,
                "row_sha256": canonical_sha256({
                    "row_id": row_id,
                    "source": source_hash,
                    "length": int(length),
                    "input_ids": ids,
                }),
                "source_row_sha256": source_hash,
                "source_index": index,
            })
    return jobs


def load_ruler_jobs(
    root: Path,
    checkpoint: Path,
    tokenizer: Any,
    *,
    tasks: tuple[str, ...] | None,
    lengths: tuple[int, ...] | None,
    limit_per_cell: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.evaluate_instruct_ruler_transfer import (
        _validate_data,
        row_sha256,
    )
    from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.prepare_instruct_ruler_transfer import (
        TASK_CONFIGS,
        chat_input_ids,
    )

    receipt, prepared = _validate_data(
        root=root.resolve(),
        checkpoint=checkpoint.resolve(),
        requested_tasks=tasks,
        requested_lengths=lengths,
        limit_per_cell=limit_per_cell,
    )
    jobs: list[dict[str, Any]] = []
    for row in prepared:
        task = str(row["_task"])
        length = int(row["_nominal_length"])
        local_index = int(row["_local_index"])
        chat = chat_input_ids(tokenizer.apply_chat_template(
            [{"role": "user", "content": row["input"]}],
            add_generation_prompt=True,
            return_tensors="pt",
        ))
        prefix = tokenizer(
            row.get("answer_prefix", ""),
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids
        input_ids = [int(value) for value in np.concatenate(
            (chat.cpu().numpy(), prefix.cpu().numpy()), axis=1,
        )[0].tolist()]
        source = {
            name: value for name, value in row.items()
            if not name.startswith("_")
        }
        source_hash = row_sha256(source)
        row_id = f"ruler:{task}:L{length}:I{local_index}"
        jobs.append({
            "row_id": row_id,
            "family": str(TASK_CONFIGS[task]["role"]),
            "task": task,
            "nominal_length": length,
            "input_ids": input_ids,
            "references": [str(value) for value in row["outputs"]],
            "metric": str(TASK_CONFIGS[task]["official_metric"]),
            "max_new_tokens": int(row["_generation_tokens"]),
            "row_sha256": canonical_sha256({
                "row_id": row_id,
                "source_row_sha256": source_hash,
                "input_ids": input_ids,
                "references": row["outputs"],
            }),
            "source_row_sha256": source_hash,
            "local_index": local_index,
        })
    return receipt, jobs


def prefill_decode_greedy(
    model: Any,
    input_ids: Any,
    *,
    max_new_tokens: int,
    eos_token_id: int | None,
    phase_context_budget: int | None = None,
) -> Any:
    import torch

    if getattr(input_ids, "ndim", None) != 2 or int(input_ids.shape[0]) != 1:
        raise ValueError("input_ids must have shape [1,T]")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    budget = int(
        input_ids.shape[1] + max_new_tokens
        if phase_context_budget is None else phase_context_budget
    )
    if budget < int(input_ids.shape[1]) + max_new_tokens or budget > 16_384:
        raise ValueError("phase_context_budget must cover the sequence and be <= 16384")
    with torch.inference_mode():
        try:
            outputs = model(
                input_ids=input_ids,
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
                phase_context_budget=budget,
            )
        except TypeError as exc:
            raise RuntimeError("prefill lacks logits_to_keep=1") from exc
        past = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1)
        generated = []
        for _ in range(max_new_tokens):
            generated.append(next_token)
            if (
                eos_token_id is not None
                and bool(torch.all(next_token == int(eos_token_id)))
            ):
                break
            try:
                outputs = model(
                    input_ids=next_token[:, None],
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                    logits_to_keep=1,
                    phase_context_budget=budget,
                )
            except TypeError as exc:
                raise RuntimeError("decode lacks logits_to_keep=1") from exc
            past = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1)
    return torch.stack(generated, dim=1)


def aggregate_cells(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    cells: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        family = str(row["family"])
        length = str(int(row["nominal_length"]))
        cells.setdefault(family, {}).setdefault(length, []).append(
            float(row["score"])
        )
    if not cells:
        raise RuntimeError("no downstream rows to aggregate")
    cell_scores = {
        family: {
            length: sum(values) / len(values)
            for length, values in by_length.items()
        }
        for family, by_length in cells.items()
    }
    family_macro = {
        family: sum(values.values()) / len(values)
        for family, values in cell_scores.items()
    }
    flat = [
        value for values in cell_scores.values() for value in values.values()
    ]
    return {
        "cells": cell_scores,
        "family_macro": family_macro,
        "macro": sum(flat) / len(flat),
        "macro_definition": "unweighted mean over family-length cells",
    }


def _official_ruler_score(
    prediction: str, references: Sequence[str], metric: str,
) -> float:
    from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.evaluate_instruct_ruler_transfer import (
        official_task_score,
    )

    return float(official_task_score(prediction, list(references), metric))


def _official_reference_recall(
    prediction: str, references: Sequence[str],
) -> float:
    from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.evaluate_instruct_ruler_transfer import (
        official_string_match_all,
    )

    return float(official_string_match_all(prediction, list(references)))


def evaluate_jobs(
    model: Any,
    tokenizer: Any,
    jobs: Sequence[Mapping[str, Any]],
    rows_path: Path,
) -> dict[str, Any]:
    import torch

    expected = list(jobs)
    completed = load_resumable_rows(rows_path, expected)
    generation_cache: dict[str, dict[str, Any]] = {}
    for job in expected:
        row_id = str(job["row_id"])
        prior = completed.get(row_id)
        if prior is not None:
            generation_cache[generation_cache_key(job)] = {
                "generated_token_ids": list(prior["generated_token_ids"]),
                "prediction": str(prior["prediction"]),
                "source_row_id": row_id,
            }
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    for ordinal, job in enumerate(expected):
        row_id = str(job["row_id"])
        if row_id in completed:
            continue
        cache_key = generation_cache_key(job)
        cached = generation_cache.get(cache_key)
        started = time.perf_counter()
        phase_context_budget = int(job["nominal_length"])
        if len(job["input_ids"]) + int(job["max_new_tokens"]) > phase_context_budget:
            raise RuntimeError(
                f"row {row_id} exceeds its nominal phase budget"
            )
        if cached is None:
            input_ids = torch.tensor(
                [list(job["input_ids"])], dtype=torch.long, device="cuda",
            )
            generated = prefill_decode_greedy(
                model,
                input_ids,
                max_new_tokens=int(job["max_new_tokens"]),
                eos_token_id=eos_token_id,
                phase_context_budget=phase_context_budget,
            )[0].detach().cpu().tolist()
            prediction = tokenizer.decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            generation_cache[cache_key] = {
                "generated_token_ids": list(generated),
                "prediction": prediction,
                "source_row_id": row_id,
            }
            generation_reused_from = None
        else:
            generated = list(cached["generated_token_ids"])
            prediction = str(cached["prediction"])
            generation_reused_from = str(cached["source_row_id"])
        references = list(job["references"])
        if job["metric"] == "longbench_token_f1_exact":
            score = token_f1(prediction, references)
            exact = normalized_exact(prediction, references)
            reference_recall = None
        else:
            score = _official_ruler_score(
                prediction, references, str(job["metric"]),
            )
            exact = None
            reference_recall = _official_reference_recall(
                prediction, references,
            )
        row = {
            "ordinal": ordinal,
            "row_id": row_id,
            "row_sha256": str(job["row_sha256"]),
            "generation_cache_key": cache_key,
            "generation_reused_from": generation_reused_from,
            "input_tokens": len(job["input_ids"]),
            "input_sha256": canonical_sha256(job["input_ids"]),
            "family": str(job["family"]),
            "task": str(job["task"]),
            "nominal_length": int(job["nominal_length"]),
            "prediction": prediction,
            "generated_token_ids": [int(value) for value in generated],
            "generated_tokens": len(generated),
            "score": float(score),
            "exact": exact,
            "reference_recall": reference_recall,
            "eos_boundary": (
                strict_eos_boundary(generated, int(eos_token_id))
                if eos_token_id is not None else None
            ),
            "elapsed_seconds": time.perf_counter() - started,
        }
        atomic_append_jsonl(rows_path, row)
        completed[row_id] = row
    if len(completed) != len(expected):
        raise RuntimeError(
            f"evaluation incomplete: {len(completed)} of {len(expected)}"
        )
    rows = [completed[str(job["row_id"])] for job in expected]
    aggregation = aggregate_cells(rows)
    cache_rows = {
        str(row["generation_cache_key"]): row for row in rows
    }
    aggregation["unique_generations"] = len(cache_rows)
    aggregation["duplicate_generation_aliases"] = sum(
        row.get("generation_reused_from") is not None for row in rows
    )
    aggregation["input_tokens_by_window"] = {
        str(length): sorted({int(row["input_tokens"]) for row in rows if int(row["nominal_length"]) == length})
        for length in sorted({int(row["nominal_length"]) for row in rows})
    }
    aggregation["generation_reuse_by_window"] = {
        str(length): {
            "rows": sum(int(row["nominal_length"]) == length for row in rows),
            "unique_generations": len({
                str(row["generation_cache_key"])
                for row in rows if int(row["nominal_length"]) == length
            }),
            "duplicate_aliases": sum(
                int(row["nominal_length"]) == length
                and row.get("generation_reused_from") is not None
                for row in rows
            ),
        }
        for length in sorted({int(row["nominal_length"]) for row in rows})
    }
    aggregation["generation_cache_definition"] = (
        "input_ids + max_new_tokens + references + metric"
    )
    wiki = [row for row in rows if row["task"] == "2wikimqa"]
    aggregation["2wiki_exact_macro"] = (
        sum(float(row["exact"]) for row in wiki) / len(wiki)
        if wiki else None
    )
    return {"rows": rows, "aggregation": aggregation}


def _load_model(
    checkpoint: Path,
    bundle: Path,
    *,
    target: Mapping[str, Any],
    sidecar_metadata: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    original_max_positions = int(
        getattr(base.config, "max_position_embeddings", 0)
    )
    if original_max_positions != 4_096:
        raise RuntimeError(
            "released OLMo runtime base must retain max_position_embeddings=4096"
        )
    base.config.original_max_position_embeddings = original_max_positions
    base.config.max_position_embeddings = 16_384
    native_hash = tensor_state_sha256(
        base.model.rotary_emb.inv_freq.detach().cpu()
    )
    if native_hash != sidecar_metadata.get("native_inv_freq_sha256"):
        raise RuntimeError("released Native frequency hash differs from sidecar")
    model = PeftModel.from_pretrained(
        base, str(bundle / "adapter"), is_trainable=False,
    )
    model.config.original_max_position_embeddings = original_max_positions
    model.config.max_position_embeddings = 16_384
    from rebuttal.rebuttal_0723.experiments.olmo2_phase_adarope_5090 import (
        phase_adarope,
    )

    bank, install_receipt = phase_adarope.install_phase_adarope(
        model.get_base_model(),
        target_inv_freq=target["values"],
        target_name=str(sidecar_metadata["target_name"]),
        mode=str(sidecar_metadata["mode"]),
        strict=True,
    )
    loaded = phase_adarope.load_phase_adarope_state(
        bundle / "phase_adarope_state.pt", bank, strict=True,
    )
    model.eval().to("cuda")
    return model, {
        "install": install_receipt,
        "sidecar": loaded,
        "native_inv_freq_sha256": native_hash,
        "runtime_context": {
            "original_max_position_embeddings": original_max_positions,
            "max_position_embeddings": 16_384,
        },
        "adapter_tree": _adapter_tree_receipt(bundle / "adapter"),
        "fresh_load": True,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorize", action="store_true")
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-ready-receipt", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--adapter-bundle", type=Path, required=True)
    parser.add_argument("--natural-gate", type=Path, nargs=3, required=True)
    parser.add_argument("--natural-data-root", type=Path, nargs=3, required=True)
    parser.add_argument("--natural-parent-receipt", type=Path, required=True)
    parser.add_argument("--ruler-data", type=Path)
    parser.add_argument("--ruler-tasks", nargs="+")
    parser.add_argument("--ruler-lengths", nargs="+", type=int)
    parser.add_argument("--limit-per-cell", type=int, default=20)
    parser.add_argument("--longbench-zip", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    require_dual_authorization(
        cli_authorize=bool(args.authorize), environment=os.environ,
    )
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    checkpoint = checkpoint_receipt(
        args.checkpoint.resolve(), args.checkpoint_ready_receipt.resolve(),
    )
    arm = str(args.arm)
    target = validate_target_manifest(args.target_manifest, arm)
    bundle = validate_bundle_contract(
        args.adapter_bundle.resolve(),
        arm=arm,
        checkpoint_sha256=checkpoint["weight_sha256"],
        target=target,
    )
    natural = validate_natural_final_gates(
        [path.resolve() for path in args.natural_gate],
        [path.resolve() for path in args.natural_data_root],
        parent_receipt=args.natural_parent_receipt.resolve(),
        target=target,
        bundle_receipt=args.adapter_bundle.resolve() / "receipt.json",
    )
    # The trainer's stage receipt carries the target/sidecar contract; this
    # evaluator emits the completed binding to the independently verified
    # READY checkpoint as part of its own final receipt.
    bundle.update({
        "checkpoint_path": checkpoint["path"],
        "checkpoint_sha256": checkpoint["weight_sha256"],
        "checkpoint_ready_receipt_sha256": checkpoint["ready_receipt_sha256"],
        "native_inv_freq_sha256": bundle["sidecar_metadata"][
            "native_inv_freq_sha256"
        ],
    })
    if args.ruler_data is None and args.longbench_zip is None:
        raise ValueError("at least one official held-out suite is required")
    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        (args.tokenizer or args.checkpoint).resolve(),
        local_files_only=True,
        trust_remote_code=False,
    )
    ruler_receipt = None
    jobs: list[dict[str, Any]] = []
    if args.ruler_data is not None:
        requested_tasks = (
            None if args.ruler_tasks is None
            else tuple(str(value) for value in args.ruler_tasks)
        )
        requested_lengths = (
            None if args.ruler_lengths is None
            else tuple(int(value) for value in args.ruler_lengths)
        )
        ruler_receipt, ruler_jobs = load_ruler_jobs(
            args.ruler_data.resolve(),
            args.checkpoint.resolve(),
            tokenizer,
            tasks=requested_tasks,
            lengths=requested_lengths,
            limit_per_cell=int(args.limit_per_cell),
        )
        jobs.extend(ruler_jobs)
    longbench = None
    if args.longbench_zip is not None:
        longbench = load_official_2wiki_rows(args.longbench_zip.resolve())
        jobs.extend(build_2wiki_jobs(tokenizer, longbench))
    model, load_receipt = _load_model(
        args.checkpoint.resolve(),
        args.adapter_bundle.resolve(),
        target=target,
        sidecar_metadata=bundle["sidecar_metadata"],
    )
    rows_path = output.with_name(output.name + ".rows.jsonl")
    evaluated = evaluate_jobs(model, tokenizer, jobs, rows_path)
    del model
    result = {
        "status": STATUS,
        "arm": arm,
        "checkpoint": checkpoint,
        "target": {
            key: value for key, value in target.items() if key != "values"
        },
        "bundle": bundle,
        "natural_gate": natural,
        "protocol_code_sha256": protocol_code_hashes(),
        "downstream_code_sha256": downstream_code_hashes(),
        "ruler_data": ruler_receipt,
        "longbench": (
            None if longbench is None
            else {
                key: value for key, value in longbench.items()
                if key != "rows"
            }
        ),
        "rows_path": str(rows_path.resolve()),
        **evaluated,
    }
    atomic_json(output, result)
    return result


if __name__ == "__main__":
    run(parse_args())
