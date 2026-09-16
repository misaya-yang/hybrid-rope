#!/usr/bin/env python3
"""Validate the frozen 128K queue before an expensive GPU is started.

The default path is CPU-only.  ``--check-gpu`` additionally verifies the
destination CUDA runtime and a forced Flash-SDPA operation.  It never launches
an experiment or creates model generations.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


TASKS = (
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe", "qa_1", "qa_2",
)
QWEN_LENGTHS = (65536, 131072)
QWEN_ROWS_PER_TASK = 50
QWEN_SEED = 20261101
QWEN_QA_OFFSET = 5600


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def expected_analytic_table(config: dict, *, method: str, scale: float):
    from experiments.fixed_rope_three_interfaces_20260913 import tables
    return tables.build_analytic(
        config, method=method, scale=scale, low=None, high=None, depth=1.0, gain=None,
    )


def validate_llama_assets(root: Path, model: Path) -> dict[str, Any]:
    ready_path = _require_file(root / "assets" / "ready.json")
    ready = read_json(ready_path)
    expected = {
        "status": "TAILSPLINE_LLAMA_S16_128K_ASSETS_READY_V1",
        "ruler_rows": 130,
        "ppl_documents": 10,
        "length": 131072,
        "band": [18, 35],
        "gpu_execution": False,
    }
    if any(ready.get(key) != value for key, value in expected.items()):
        raise ValueError("Llama S16 ready receipt does not match the frozen 128K gate")

    panel = _require_file(root / "assets" / "full13" / "inputs.jsonl")
    panel_manifest = read_json(_require_file(root / "assets" / "full13" / "manifest.json"))
    lm_array = _require_file(root / "assets" / "ppl10" / "lm.npy")
    lm_manifest = read_json(_require_file(root / "assets" / "ppl10" / "manifest.json"))
    config_path = _require_file(model / "config.json")
    config = read_json(config_path)
    counts: Counter[str] = Counter()
    rows = 0
    with panel.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            counts[str(row.get("task"))] += 1
            prompt = row.get("prompt_ids")
            if (
                row.get("length_cap") != 131072
                or not isinstance(prompt, list)
                or len(prompt) + int(row.get("max_new_tokens", 0)) > 131072
            ):
                raise ValueError("Llama S16 panel row violates the frozen 128K contract")
    if (
        panel_manifest.get("rows") != 130
        or rows != 130
        or counts != Counter({task: 10 for task in TASKS})
        or panel_manifest.get("inputs_sha256") != ready.get("inputs_sha256")
        or sha256(panel) != ready.get("inputs_sha256")
        or lm_manifest.get("documents") != 10
        or lm_manifest.get("lengths") != [131072]
        or lm_manifest.get("array_shape") != [10, 131073]
        or lm_manifest.get("lm_array_sha256") != ready.get("lm_array_sha256")
        or sha256(lm_array) != ready.get("lm_array_sha256")
    ):
        raise ValueError("Llama S16 128K panel or LM asset drift")
    array = np.load(lm_array, mmap_mode="r", allow_pickle=False)
    if array.shape != (10, 131073) or array.dtype != np.int64:
        raise ValueError("Llama S16 LM array shape or dtype drift")

    table_hashes = {}
    from experiments.fixed_rope_three_interfaces_20260913 import tables as table_tools
    for arm in ("tailspline", "mrpro"):
        receipt = read_json(_require_file(root / "tables" / f"{arm}.json"))
        expected_values, expected_gain, _ = expected_analytic_table(
            config, method=arm, scale=16.0,
        )
        table = receipt.get("table", receipt)
        actual_values = np.asarray(table.get("values_float32"), dtype=np.float32)
        if (
            receipt.get("candidate_id") != f"llama3_8b_s16_128k_{arm}"
            or receipt.get("scale") != 16.0
            or receipt.get("band_envelope") != [18, 35]
            or not np.array_equal(actual_values, expected_values)
            or float(table.get("gain", float("nan"))) != float(expected_gain)
            or receipt.get("table_sha256_float32") != table_tools.tensor_sha256(actual_values)
        ):
            raise ValueError(f"Llama S16 analytic table drift: {arm}")
        table_hashes[arm] = receipt.get("table_sha256_float32")
    if table_hashes != ready.get("table_sha256"):
        raise ValueError("Llama S16 table receipt drift")
    return {
        "status": "COMPLETE",
        "ready_sha256": sha256(ready_path),
        "inputs_sha256": ready["inputs_sha256"],
        "lm_array_sha256": ready["lm_array_sha256"],
        "table_sha256": table_hashes,
        "rows_per_arm": 130,
        "lm_documents_per_arm": 10,
    }


def validate_qwen_assets(root: Path, model: Path) -> dict[str, Any]:
    assets = root / "assets"
    manifest_path = _require_file(assets / "manifest.json")
    manifest = read_json(manifest_path)
    config_path = _require_file(model / "config.json")
    expected = {
        "status": "COMPLETE",
        "model_id": "qwen25_3b",
        "scale": 4.0,
        "lengths": list(QWEN_LENGTHS),
        "rows_per_task": QWEN_ROWS_PER_TASK,
        "seed": QWEN_SEED,
        "qa_offset": QWEN_QA_OFFSET,
        "selection_mode": "source-order",
        "selection_uses_model_outputs": False,
        "content_padding": False,
        "rows": len(TASKS) * len(QWEN_LENGTHS) * QWEN_ROWS_PER_TASK,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Qwen clean-transfer manifest does not match X2")
    if manifest.get("model_identity", {}).get("config_sha256") != sha256(config_path):
        raise ValueError("Qwen clean assets belong to a different checkpoint config")

    panels = {}
    for length in QWEN_LENGTHS:
        entry = manifest.get("panels", {}).get(str(length))
        if not isinstance(entry, dict):
            raise ValueError(f"Qwen manifest lacks {length} panel")
        inputs = _require_file(assets / str(entry.get("inputs", "")))
        child_path = _require_file(assets / str(entry.get("manifest", "")))
        child = read_json(child_path)
        expected_rows = len(TASKS) * QWEN_ROWS_PER_TASK
        if (
            entry.get("rows") != expected_rows
            or sha256(inputs) != entry.get("inputs_sha256")
            or sha256(child_path) != entry.get("manifest_sha256")
            or child.get("rows") != expected_rows
            or child.get("length_cap") != length
            or child.get("tasks") != list(TASKS)
            or child.get("selection_mode") != "source-order"
            or child.get("content_padding") is not False
        ):
            raise ValueError(f"Qwen {length} clean panel drift")
        counts: Counter[str] = Counter()
        rows = 0
        with inputs.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                rows += 1
                counts[str(row.get("task"))] += 1
                prompt = row.get("prompt_ids")
                if (
                    row.get("length_cap") != length
                    or row.get("selection_mode") != "source-order"
                    or row.get("selection_uses_model_outputs") is not False
                    or row.get("irrelevant_padding_tokens") != 0
                    or not isinstance(prompt, list)
                    or len(prompt) != row.get("input_tokens")
                    or len(prompt) + int(row.get("max_new_tokens", 0)) > length
                ):
                    raise ValueError(f"Qwen {length} row violates the clean contract")
        if rows != expected_rows or counts != Counter({task: QWEN_ROWS_PER_TASK for task in TASKS}):
            raise ValueError(f"Qwen {length} panel is not Full-13 x {QWEN_ROWS_PER_TASK}")
        panels[str(length)] = {
            "rows": rows,
            "inputs_sha256": entry["inputs_sha256"],
            "manifest_sha256": entry["manifest_sha256"],
        }
    from experiments.iclr2027_strong_evidence_20260915 import run_clean_matrix

    table_args = argparse.Namespace(model=model, model_id="qwen25_3b", scale=4.0)
    table_hashes = {}
    for arm in ("tailspline", "mrpro"):
        receipt = run_clean_matrix.validate_table(
            table_args, arm, _require_file(root / "tables" / f"{arm}.json"),
        )
        table_hashes[arm] = receipt["table_sha256_float32"]
    return {
        "status": "COMPLETE",
        "manifest_sha256": sha256(manifest_path),
        "model_config_sha256": sha256(config_path),
        "rows_per_arm": len(TASKS) * len(QWEN_LENGTHS) * QWEN_ROWS_PER_TASK,
        "panels": panels,
        "table_sha256": table_hashes,
    }


def version_pair(value: str) -> tuple[int, int]:
    parts = value.split("+", 1)[0].split(".")
    return int(parts[0]), int(parts[1])


def cuda_pair(value: str | None) -> tuple[int, int]:
    if not value:
        return (0, 0)
    parts = value.split(".")
    return int(parts[0]), int(parts[1])


def validate_gpu(*, minimum_vram_mib: int, require_blackwell: bool) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one CUDA GPU is required")
    properties = torch.cuda.get_device_properties(0)
    total_mib = int(properties.total_memory // (1024 * 1024))
    capability = tuple(torch.cuda.get_device_capability(0))
    arch = f"sm_{capability[0]}{capability[1]}"
    if total_mib < minimum_vram_mib:
        raise RuntimeError(f"need >= {minimum_vram_mib} MiB VRAM, found {total_mib}")
    if version_pair(torch.__version__) < (2, 7) or cuda_pair(torch.version.cuda) < (12, 8):
        raise RuntimeError("Blackwell requires PyTorch >=2.7 with CUDA >=12.8")
    if require_blackwell and capability != (12, 0):
        raise RuntimeError(f"expected sm_120 Blackwell, found {capability}")
    if require_blackwell and arch not in torch.cuda.get_arch_list():
        raise RuntimeError(f"PyTorch binary does not include {arch}")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("destination GPU does not support BF16")
    if not torch.backends.cuda.is_flash_attention_available():
        raise RuntimeError("PyTorch Flash-SDPA backend is unavailable")

    query = torch.randn((1, 8, 64, 128), device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        output = functional.scaled_dot_product_attention(
            query, query, query, is_causal=True, dropout_p=0.0,
        )
    torch.cuda.synchronize()
    if not torch.isfinite(output).all().item():
        raise RuntimeError("forced Flash-SDPA smoke produced non-finite output")
    return {
        "status": "COMPLETE",
        "name": properties.name,
        "total_mib": total_mib,
        "capability": list(capability),
        "arch": arch,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "arch_list": torch.cuda.get_arch_list(),
        "bf16": True,
        "forced_flash_sdpa": True,
    }


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-root", type=Path, default=Path("/root/autodl-tmp/today_rope_plan_20260914"))
    parser.add_argument("--qwen-model", type=Path,
                        default=Path("/root/autodl-tmp/rope_qwen_baseline_20260907/model"))
    parser.add_argument("--llama-model", type=Path,
                        default=Path("/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct"))
    parser.add_argument("--check-gpu", action="store_true")
    parser.add_argument("--minimum-vram-mib", type=int, default=80000)
    parser.add_argument("--allow-non-blackwell", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.minimum_vram_mib <= 0:
        raise ValueError("minimum VRAM must be positive")
    payload: dict[str, Any] = {
        "status": "PRO6000_128K_PREFLIGHT_COMPLETE_V1",
        "gpu_execution": False,
        "llama_s16_128k": validate_llama_assets(
            args.plan_root / "tailspline_llama_s16_128k_gate", args.llama_model,
        ),
        "qwen_s4_64k128k": validate_qwen_assets(
            args.plan_root / "tailspline_qwen25_s4_64k128k_clean", args.qwen_model,
        ),
    }
    if args.check_gpu:
        payload["gpu"] = validate_gpu(
            minimum_vram_mib=args.minimum_vram_mib,
            require_blackwell=not args.allow_non_blackwell,
        )
    if args.out:
        atomic_json(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
