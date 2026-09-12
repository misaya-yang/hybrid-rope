#!/usr/bin/env python3
"""Run all seven frozen E2 OLMo natural-QA arms, or validate them with --dry-run."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path

import numpy as np

from scripts.eval.longbench_metrics import qa_f1_score
from scripts.experiments.cross_audit.tables import tensor_sha
from experiments.rope_fast_5090_20260912.e2_prepare import CAPS, digest


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic(path: Path, value) -> None:
    tmp = path.with_name(path.name + ".incomplete")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def validate(prepared: Path, model: Path, hash_weights: bool) -> tuple[dict, list[dict], dict, dict]:
    manifest = json.loads((prepared / "manifest.json").read_text())
    if manifest.get("status") != "READY_GPU_NOT_RUN" or manifest["pool"]["rows"] != 778 or manifest["pool"]["long_rows"] != 631:
        raise ValueError("E2 manifest contract differs")
    for name, expected in manifest["files"].items():
        if sha_file(prepared / name) != expected:
            raise ValueError(f"prepared file drift: {name}")
    data = rows(prepared / "inputs.jsonl")
    if len(data) != 778 or len({row["row_id"] for row in data}) != 778:
        raise ValueError("input pool is incomplete or duplicated")
    for row in data:
        if len(row["prompt_ids"]) != row["input_tokens"] or digest(row["prompt_ids"]) != row["prompt_sha256"]:
            raise ValueError(f"prompt token identity drift: {row['row_id']}")
        if row["max_new_tokens"] != CAPS[row["task"]] or row["input_tokens"] + row["max_new_tokens"] > 16_384:
            raise ValueError(f"task cap drift: {row['row_id']}")
    tables = json.loads((prepared / "tables.json").read_text())
    if len(tables) != len(manifest["arms"]) or set(tables) != set(manifest["arms"]):
        raise ValueError("arm order drift")
    for name, table in tables.items():
        values = np.asarray(table["values_float32"], dtype=np.float32)
        if values.shape != (64,) or tensor_sha(values) != table["tensor_sha256"]:
            raise ValueError(f"table drift: {name}")
    if sha_file(model / "tokenizer.json") != manifest["model"]["tokenizer_sha256"]:
        raise ValueError("tokenizer drift")
    if hash_weights and sha_file(model / manifest["model"]["weight_file"]) != manifest["model"]["weight_sha256"]:
        raise ValueError("model weight drift")
    root = Path(__file__).resolve().parents[2]
    for name, expected in manifest["code_files"].items():
        if sha_file(root / name) != expected:
            raise ValueError(f"runtime code drift: {name}")
    for package, expected in manifest["required_software"].items():
        actual = importlib.metadata.version(package)
        if actual != expected and not (package == "torch" and actual.startswith(expected + "+")):
            raise RuntimeError(f"software drift: {package}={actual}, expected {expected}")
    return manifest, data, tables, json.loads((prepared / "generation_config.json").read_text())


def validate_saved_rows(saved: list[dict], expected: list[dict], arm: str, eos_ids: set[int]) -> None:
    if len(saved) > len(expected):
        raise ValueError(f"too many saved rows for {arm}")
    for index, row in enumerate(saved):
        source = expected[index]
        for key in ("row_id", "task", "prompt_sha256", "input_tokens", "max_new_tokens", "references"):
            if row.get(key) != source[key]:
                raise ValueError(f"resume row mismatch: {arm}/{index}/{key}")
        ended = bool(row.get("generated_ids") and row["generated_ids"][-1] in eos_ids)
        if row.get("arm") != arm or len(row.get("generated_ids", [])) > row["max_new_tokens"] or row.get("ended_eos") != ended:
            raise ValueError(f"resume output contract mismatch: {arm}/{index}")
        if abs(qa_f1_score(row["output_text"], row["references"]) - row["whole_response_f1"]) > 1e-12:
            raise ValueError(f"resume score mismatch: {arm}/{index}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    p.add_argument("--skip-weight-hash", action="store_true", help="Only for CPU preparation; GPU execution always hashes weights")
    args = p.parse_args()
    prepared, model = args.prepared.resolve(), args.model.resolve()
    manifest, data, tables, decoding_dict = validate(prepared, model, not args.skip_weight_hash or not args.dry_run)
    if args.dry_run:
        print(json.dumps({"status": "DRY_RUN_PASS", "rows": len(data), "long_rows": sum(r["input_tokens"] > 4096 for r in data), "arms": list(tables)}))
        return
    if args.output is None:
        p.error("--output is required with --execute")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    from scripts.experiments.olmo_fast_screen.runtime import install, verify

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 device unavailable")
    required = manifest["required_device"]
    device_name = torch.cuda.get_device_name(0)
    capability = list(torch.cuda.get_device_capability(0))
    if required["name_contains"] not in device_name or capability != required["cuda_capability"] or required["torch_arch"] not in torch.cuda.get_arch_list():
        raise RuntimeError(f"requires RTX 5090 sm120, got {device_name}/{capability}/{torch.cuda.get_arch_list()}")
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    model_obj = AutoModelForCausalLM.from_pretrained(model, local_files_only=True, dtype=torch.bfloat16,
                                                     device_map={"": "cuda"}, attn_implementation="sdpa").eval()
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    decoding = GenerationConfig.from_dict(decoding_dict)
    runtime_contract = {"manifest_sha256": sha_file(prepared / "manifest.json"), "code_files": manifest["code_files"],
                        "device_name": device_name, "cuda_capability": capability,
                        "torch_arch_list": torch.cuda.get_arch_list(),
                        "software": {name: importlib.metadata.version(name) for name in manifest["required_software"]},
                        "model_weight_sha256": manifest["model"]["weight_sha256"], "decoder": manifest["decoder_contract"]}
    if (output / "runtime.json").exists() and json.loads((output / "runtime.json").read_text()) != runtime_contract:
        raise ValueError("resume runtime contract differs")
    atomic(output / "runtime.json", runtime_contract)
    status = {"status": "RUNNING", "manifest_sha256": runtime_contract["manifest_sha256"],
              "model_weight_sha256": manifest["model"]["weight_sha256"], "arms": list(tables),
              "started_at": time.time(), "completed_arms": []}
    if (output / "status.json").exists():
        previous = json.loads((output / "status.json").read_text())
        if previous.get("manifest_sha256") != status["manifest_sha256"]:
            raise ValueError("resume manifest differs")
        status["completed_arms"] = previous.get("completed_arms", [])
    atomic(output / "status.json", status)
    eos_ids = decoding.eos_token_id
    eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids)
    with torch.inference_mode():
        for arm in manifest["arms"]:
            table = tables[arm]
            install(model_obj, table)
            verify(model_obj, table)
            raw_path = output / f"{arm}.jsonl"
            complete_path = output / f"{arm}.json"
            saved = rows(raw_path) if raw_path.exists() else []
            validate_saved_rows(saved, data, arm, eos_ids)
            if complete_path.exists():
                receipt = json.loads(complete_path.read_text())
                if len(saved) != len(data) or receipt.get("raw_sha256") != sha_file(raw_path) or receipt.get("table") != table:
                    raise ValueError(f"completed arm receipt drift: {arm}")
                if arm not in status["completed_arms"]:
                    status["completed_arms"].append(arm)
                continue
            started = time.monotonic()
            with raw_path.open("a") as stream:
                for index, row in enumerate(data[len(saved):], start=len(saved)):
                    ids = torch.tensor([row["prompt_ids"]], dtype=torch.long, device="cuda")
                    generated = model_obj.generate(ids, attention_mask=torch.ones_like(ids),
                                                   generation_config=decoding, max_new_tokens=row["max_new_tokens"])
                    new_ids = generated[0, ids.shape[1]:].tolist()
                    ended_eos = bool(new_ids and new_ids[-1] in eos_ids)
                    score_ids = new_ids[:-1] if ended_eos else new_ids
                    output_text = tokenizer.decode(score_ids, skip_special_tokens=False)
                    record = {key: row[key] for key in ("row_id", "task", "prompt_sha256", "input_tokens", "max_new_tokens", "references")}
                    record.update(arm=arm, generated_ids=new_ids, ended_eos=ended_eos,
                                  hit_cap=len(new_ids) == row["max_new_tokens"] and not ended_eos,
                                  output_text=output_text, whole_response_f1=qa_f1_score(output_text, row["references"]))
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
                    stream.flush()
                    atomic(output / "live.json", {"arm": arm, "completed": index + 1, "total": len(data)})
            verify(model_obj, table)
            atomic(complete_path, {"status": "COMPLETE", "rows": len(data),
                   "raw_sha256": sha_file(raw_path), "table": table, "elapsed_seconds": time.monotonic() - started})
            if arm not in status["completed_arms"]:
                status["completed_arms"].append(arm)
            atomic(output / "status.json", status)
    status.update(status="COMPLETE", finished_at=time.time(), peak_cuda_bytes=int(torch.cuda.max_memory_allocated()))
    atomic(output / "status.json", status)
    print(json.dumps(status, sort_keys=True))


if __name__ == "__main__":
    main()
